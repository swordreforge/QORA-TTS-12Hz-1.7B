//! Code Predictor — 5-layer transformer that expands code group 0 to groups 1..15.
//!
//! Architecture (config-driven, supports 0.6B and 1.7B):
//!   small_to_mtp_projection: [cp_hidden, talker_hidden] + bias — input proj
//!   15 codec_embeddings: [cp_vocab, embed_dim] each — for code groups 0..14
//!   N layers: hidden=cp_hidden, heads/kv_heads, head_dim=128, intermediate
//!   QK-norm: q_norm [128], k_norm [128] per layer
//!   15 lm_heads: [cp_vocab, cp_hidden] — output predictions
//!   Standard interleaved RoPE (no multimodal), theta=1M

use half::f16;
use crate::gemv::*;

pub struct PredictorLayerWeights {
    pub q_proj: Weight,     // [num_heads*head_dim, hidden_size]
    pub k_proj: Weight,     // [num_kv_heads*head_dim, hidden_size]
    pub v_proj: Weight,     // [num_kv_heads*head_dim, hidden_size]
    pub o_proj: Weight,     // [hidden_size, num_heads*head_dim]
    pub q_norm: Vec<f16>,   // [head_dim]
    pub k_norm: Vec<f16>,   // [head_dim]
    pub gate_proj: Weight,  // [intermediate, hidden_size]
    pub up_proj: Weight,    // [intermediate, hidden_size]
    pub down_proj: Weight,  // [hidden_size, intermediate]
    pub input_norm_gamma: Vec<f16>,
    pub post_attn_norm_gamma: Vec<f16>,
}

pub struct CodePredictorWeights {
    pub input_proj: Option<Weight>,             // [cp_hidden, talker_hidden] small_to_mtp_projection (None if same size)
    pub input_proj_bias: Option<Vec<f32>>,      // [cp_hidden] (None if no projection)
    pub codec_embeddings: Vec<Weight>,          // 15 × [cp_vocab, embed_dim]
    pub layers: Vec<PredictorLayerWeights>,     // N layers
    pub lm_heads: Vec<Weight>,                  // 15 × [cp_vocab, cp_hidden]
    pub final_norm_gamma: Vec<f16>,

    // RoPE tables (standard interleaved, half_dim = head_dim/2 = 64)
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,

    // Config
    pub hidden_size: usize,     // 1024
    pub num_heads: usize,       // 16
    pub num_kv_heads: usize,    // 8
    pub head_dim: usize,        // 128
    pub num_kv_groups: usize,   // 2
}

impl CodePredictorWeights {
    /// Apply optional input projection (talker_hidden → cp_hidden).
    /// When talker_hidden == cp_hidden, input_proj is None and we return a clone.
    pub fn project_input(&self, input: &[f32]) -> Vec<f32> {
        if let Some(ref proj) = self.input_proj {
            let mut out = gemv(input, proj);
            if let Some(ref bias) = self.input_proj_bias {
                for j in 0..out.len() { out[j] += bias[j]; }
            }
            out
        } else {
            input.to_vec()
        }
    }

    pub fn memory_bytes(&self) -> usize {
        let mut total = self.input_proj.as_ref().map(|w| w.memory_bytes()).unwrap_or(0);
        total += self.input_proj_bias.as_ref().map(|b| b.len() * 4).unwrap_or(0);
        for e in &self.codec_embeddings { total += e.memory_bytes(); }
        for l in &self.layers {
            total += l.q_proj.memory_bytes() + l.k_proj.memory_bytes()
                + l.v_proj.memory_bytes() + l.o_proj.memory_bytes()
                + l.gate_proj.memory_bytes() + l.up_proj.memory_bytes()
                + l.down_proj.memory_bytes();
            total += (l.input_norm_gamma.len() + l.post_attn_norm_gamma.len()
                + l.q_norm.len() + l.k_norm.len()) * 2;
        }
        for h in &self.lm_heads { total += h.memory_bytes(); }
        total += self.final_norm_gamma.len() * 2;
        total += (self.rope_cos.len() + self.rope_sin.len()) * 4;
        total
    }
}

/// Apply QK-norm per head.
#[inline]
fn apply_qk_norm(data: &mut [f32], num_heads: usize, head_dim: usize, norm_gamma: &[f16]) {
    for h in 0..num_heads {
        let start = h * head_dim;
        let sum_sq: f32 = data[start..start + head_dim].iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + 1e-6).sqrt();
        for d in 0..head_dim {
            data[start + d] = data[start + d] * inv_rms * norm_gamma[d].to_f32();
        }
    }
}

/// Apply standard SPLIT RoPE (first half, second half) - matching working code.
#[inline]
fn apply_rope_split(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    position: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) {
    let half_dim = head_dim / 2;
    let cos_offset = position * half_dim;
    if cos_offset + half_dim > cos_table.len() { return; }
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half_dim {
            // Split pattern: x1 = first half, x2 = second half
            let x1 = data[base + i];
            let x2 = data[base + half_dim + i];
            let c = cos_table[cos_offset + i];
            let s = sin_table[cos_offset + i];
            // Rotate: [x1*cos - x2*sin, x2*cos + x1*sin]
            data[base + i] = x1 * c - x2 * s;
            data[base + half_dim + i] = x2 * c + x1 * s;
        }
    }
}

/// Forward pass for code predictor — single step, predicts 15 code groups.
/// talker_hidden: [2048] — the talker's final hidden state for this timestep.
/// Returns 15 sets of logits [2048] each.
pub fn predict_codes(
    weights: &CodePredictorWeights,
    talker_hidden: &[f32],
    kv_cache: &mut RawKvCache,
    _position: usize,
) -> Vec<Vec<f32>> {
    let hidden = weights.hidden_size;

    // Project from talker hidden to predictor hidden (optional, identity if same size)
    let mut x = weights.project_input(talker_hidden);

    for i in 0..weights.layers.len() {
        let lw = &weights.layers[i];
        let (ref mut cached_k, ref mut cached_v, ref mut cached_len) = kv_cache[i];
        let offset = *cached_len;

        let x_norm = rms_norm_f16(&x, &lw.input_norm_gamma);

        let mut q = gemv(&x_norm, &lw.q_proj);
        let mut k_new = gemv(&x_norm, &lw.k_proj);
        let v_new = gemv(&x_norm, &lw.v_proj);

        // QK-norm
        apply_qk_norm(&mut q, weights.num_heads, weights.head_dim, &lw.q_norm);
        apply_qk_norm(&mut k_new, weights.num_kv_heads, weights.head_dim, &lw.k_norm);

        // Standard interleaved RoPE
        apply_rope_split(
            &mut q, weights.num_heads, weights.head_dim,
            offset, &weights.rope_cos, &weights.rope_sin,
        );
        apply_rope_split(
            &mut k_new, weights.num_kv_heads, weights.head_dim,
            offset, &weights.rope_cos, &weights.rope_sin,
        );

        cached_k.extend_from_slice(&k_new);
        cached_v.extend_from_slice(&v_new);
        *cached_len = offset + 1;
        let kv_seq_len = *cached_len;

        // GQA Attention
        let scale = 1.0 / (weights.head_dim as f32).sqrt();
        let kv_stride = weights.num_kv_heads * weights.head_dim;
        let mut attn_output = vec![0.0f32; weights.num_heads * weights.head_dim];

        for h in 0..weights.num_heads {
            let kv_h = h / weights.num_kv_groups;
            let q_offset = h * weights.head_dim;
            let q_vec = &q[q_offset..q_offset + weights.head_dim];
            let mut scores = vec![0.0f32; kv_seq_len];
            for s in 0..kv_seq_len {
                let k_offset = s * kv_stride + kv_h * weights.head_dim;
                let mut dot = 0.0f32;
                for d in 0..weights.head_dim { dot += q_vec[d] * cached_k[k_offset + d]; }
                scores[s] = dot * scale;
            }
            softmax_raw(&mut scores);
            let out_offset = h * weights.head_dim;
            for s in 0..kv_seq_len {
                let v_offset = s * kv_stride + kv_h * weights.head_dim;
                let score = scores[s];
                for d in 0..weights.head_dim {
                    attn_output[out_offset + d] += score * cached_v[v_offset + d];
                }
            }
        }

        let attn_out = gemv(&attn_output, &lw.o_proj);
        for j in 0..hidden { x[j] += attn_out[j]; }

        let x_norm = rms_norm_f16(&x, &lw.post_attn_norm_gamma);
        let gate = gemv(&x_norm, &lw.gate_proj);
        let up = gemv(&x_norm, &lw.up_proj);
        let inter_size = gate.len();
        let mut intermediate = vec![0.0f32; inter_size];
        for j in 0..inter_size { intermediate[j] = silu(gate[j]) * up[j]; }
        let mlp_out = gemv(&intermediate, &lw.down_proj);
        for j in 0..hidden { x[j] += mlp_out[j]; }
    }

    let normed = rms_norm_f16(&x, &weights.final_norm_gamma);

    // 15 lm_heads → 15 sets of logits
    weights.lm_heads.iter()
        .map(|head| gemv(&normed, head))
        .collect()
}

/// Generate 15 acoustic codes AUTOREGRESSIVELY.
/// Predicts codes 1-15 autoregressively from code 0.
/// Returns sampled codes [15] as u32.
pub fn generate_acoustic_codes(
    weights: &CodePredictorWeights,
    talker_hidden: &[f32],
    semantic_embed: &[f32],
    kv_cache: &mut RawKvCache,
) -> Vec<u32> {
    let hidden = weights.hidden_size;

    // Step 1: Reset KV cache (line 330-332)
    for cache_entry in kv_cache.iter_mut() {
        cache_entry.0.clear();
        cache_entry.1.clear();
        cache_entry.2 = 0;
    }

    // Step 2: Prefill with [talker_hidden, semantic_embed] (line 338)
    // Project both to predictor hidden size (optional, identity if same size)
    let proj_hidden = weights.project_input(talker_hidden);
    let proj_semantic = weights.project_input(semantic_embed);

    // Concatenate as [2, hidden] input (line 338)
    let mut prefill_input = vec![0.0f32; 2 * hidden];
    prefill_input[..hidden].copy_from_slice(&proj_hidden);
    prefill_input[hidden..].copy_from_slice(&proj_semantic);

    // Step 3: Run prefill through layers (line 358-366)
    let last_hidden = prefill_code_predictor(weights, &prefill_input, 2, kv_cache);

    // Step 4: Predict first acoustic code from last position (line 374-375)
    let first_logits = gemv(&last_hidden, &weights.lm_heads[0]);
    let first_code = sample_argmax(&first_logits);
    let mut all_codes = vec![first_code];

    // Step 5: Autoregressively generate remaining 14 codes (line 386-413)
    let mut offset = 2;  // Already processed 2 tokens in prefill
    for group_idx in 1..15 {
        // Embed previous code using previous group's embedding (line 388)
        let prev_code = all_codes[group_idx - 1];
        let embed_dim = weights.codec_embeddings[group_idx - 1].n();
        let code_embed = embed_lookup(&weights.codec_embeddings[group_idx - 1], prev_code as usize, embed_dim);

        // Project embedding (optional, identity if same size)
        let proj_embed = weights.project_input(&code_embed);

        // Run through layers (line 401-404)
        let h = decode_one_code_predictor_token(weights, &proj_embed, offset, kv_cache);

        // Predict next code (line 407-408)
        let logits = gemv(&h, &weights.lm_heads[group_idx]);
        let next_code = sample_argmax(&logits);
        all_codes.push(next_code);
        offset += 1;
    }

    all_codes
}

/// Get embedding for an acoustic code token.
/// acoustic_embeddings shape: [codebook_size, hidden_size]
/// Get embedding for a single acoustic code from one codebook group.
/// Used by ICL prompt construction (per-frame sum over 16 groups).
pub fn get_acoustic_embedding(weights: &CodePredictorWeights, group_idx: usize, code: u32) -> Vec<f32> {
    let embed_dim = weights.codec_embeddings[0].n();
    embed_lookup(&weights.codec_embeddings[group_idx], code as usize, embed_dim)
}

/// Get sum of embeddings for all 15 acoustic codes.
/// Sums embeddings from all 15 codebooks for the given acoustic codes.
/// Sums embeddings from all 15 codebooks for the given acoustic codes.
pub fn get_acoustic_embeddings_sum(weights: &CodePredictorWeights, acoustic_codes: &[u32]) -> Vec<f32> {
    let embed_dim = weights.codec_embeddings[0].n();  // Derive from actual weight dimensions
    let mut sum = vec![0.0f32; embed_dim];

    // Sum embeddings from all 15 codebooks
    for (codebook_idx, &code) in acoustic_codes.iter().enumerate() {
        let embed = embed_lookup(&weights.codec_embeddings[codebook_idx], code as usize, embed_dim);
        for j in 0..embed_dim {
            sum[j] += embed[j];
        }
    }

    sum
}

/// Lookup embedding from a weight matrix (same as talker.rs)
fn embed_lookup(weight: &Weight, token_id: usize, hidden_size: usize) -> Vec<f32> {
    match weight {
        Weight::Q4(q4) => {
            let groups_per_row = hidden_size / Q4_GROUP_SIZE;
            let packed_per_group = Q4_GROUP_SIZE / 2;
            let scale_base = token_id * groups_per_row;
            let pack_base = token_id * groups_per_row * packed_per_group;

            let mut output = vec![0.0f32; hidden_size];
            for g in 0..groups_per_row {
                let scale = q4.scales[scale_base + g].to_f32();
                let pack_offset = pack_base + g * packed_per_group;
                let out_offset = g * Q4_GROUP_SIZE;
                for j in 0..packed_per_group {
                    let byte = q4.packed[pack_offset + j];
                    let q0 = (byte & 0x0F) as i32 - 8;
                    let q1 = ((byte >> 4) & 0x0F) as i32 - 8;
                    output[out_offset + j * 2] = scale * q0 as f32;
                    output[out_offset + j * 2 + 1] = scale * q1 as f32;
                }
            }
            output
        }
        Weight::F16(f16) => {
            let start = token_id * hidden_size;
            let end = start + hidden_size;
            f16.data[start..end].iter().map(|&v| v.to_f32()).collect()
        }
    }
}

/// Simple argmax sampling (greedy)
fn sample_argmax(logits: &[f32]) -> u32 {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

/// Prefill code predictor with 2-token input [talker_hidden, semantic_embed].
/// Returns last hidden state.
fn prefill_code_predictor(
    weights: &CodePredictorWeights,
    input: &[f32],  // [2 * hidden_size]
    seq_len: usize,  // Should be 2
    kv_cache: &mut RawKvCache,
) -> Vec<f32> {
    let hidden = weights.hidden_size;
    let mut x = input.to_vec();

    for i in 0..weights.layers.len() {
        let lw = &weights.layers[i];

        // Norm
        let mut x_norm = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            let normed = rms_norm_f16(&x[t * hidden..(t + 1) * hidden], &lw.input_norm_gamma);
            x_norm[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
        }

        // QKV
        let mut q_all = gemm(&x_norm, seq_len, &lw.q_proj);
        let mut k_all = gemm(&x_norm, seq_len, &lw.k_proj);
        let v_all = gemm(&x_norm, seq_len, &lw.v_proj);

        // QK-norm and RoPE per-token (EXACT working code pattern)
        let q_dim = weights.num_heads * weights.head_dim;
        let k_dim = weights.num_kv_heads * weights.head_dim;
        for t in 0..seq_len {
            apply_qk_norm(&mut q_all[t * q_dim..(t + 1) * q_dim], weights.num_heads, weights.head_dim, &lw.q_norm);
            apply_qk_norm(&mut k_all[t * k_dim..(t + 1) * k_dim], weights.num_kv_heads, weights.head_dim, &lw.k_norm);
            // Apply RoPE with correct position for each token
            apply_rope_split(&mut q_all[t * q_dim..(t + 1) * q_dim], weights.num_heads, weights.head_dim, t, &weights.rope_cos, &weights.rope_sin);
            apply_rope_split(&mut k_all[t * k_dim..(t + 1) * k_dim], weights.num_kv_heads, weights.head_dim, t, &weights.rope_cos, &weights.rope_sin);
        }

        // Store in KV cache
        let (ref mut cached_k, ref mut cached_v, ref mut cached_len) = kv_cache[i];
        cached_k.extend_from_slice(&k_all);
        cached_v.extend_from_slice(&v_all);
        *cached_len = seq_len;

        // Attention
        let scale = 1.0 / (weights.head_dim as f32).sqrt();
        let kv_stride = weights.num_kv_heads * weights.head_dim;
        let mut attn_output = vec![0.0f32; seq_len * q_dim];

        for h in 0..weights.num_heads {
            let kv_h = h / weights.num_kv_groups;
            for t1 in 0..seq_len {
                let attend_len = t1 + 1;
                let q_off = t1 * q_dim + h * weights.head_dim;
                let q_vec = &q_all[q_off..q_off + weights.head_dim];
                let mut scores = vec![0.0f32; attend_len];
                for t2 in 0..attend_len {
                    let k_off = t2 * kv_stride + kv_h * weights.head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..weights.head_dim { dot += q_vec[d] * cached_k[k_off + d]; }
                    scores[t2] = dot * scale;
                }
                softmax_raw(&mut scores);
                let out_off = t1 * q_dim + h * weights.head_dim;
                for t2 in 0..attend_len {
                    let v_off = t2 * kv_stride + kv_h * weights.head_dim;
                    let score = scores[t2];
                    for d in 0..weights.head_dim {
                        attn_output[out_off + d] += score * cached_v[v_off + d];
                    }
                }
            }
        }

        let attn_out_all = gemm(&attn_output, seq_len, &lw.o_proj);
        for j in 0..x.len() { x[j] += attn_out_all[j]; }

        // MLP
        let mut x_norm2 = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            let normed = rms_norm_f16(&x[t * hidden..(t + 1) * hidden], &lw.post_attn_norm_gamma);
            x_norm2[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
        }

        let gate = gemm(&x_norm2, seq_len, &lw.gate_proj);
        let up = gemm(&x_norm2, seq_len, &lw.up_proj);
        let mut intermediate = vec![0.0f32; gate.len()];
        for j in 0..intermediate.len() { intermediate[j] = silu(gate[j]) * up[j]; }
        let mlp_out = gemm(&intermediate, seq_len, &lw.down_proj);
        for j in 0..x.len() { x[j] += mlp_out[j]; }
    }

    // Return last position's hidden state
    let last_hidden = &x[(seq_len - 1) * hidden..seq_len * hidden];
    let normed = rms_norm_f16(last_hidden, &weights.final_norm_gamma);
    normed
}

/// Decode one token in code predictor (autoregressive step).
fn decode_one_code_predictor_token(
    weights: &CodePredictorWeights,
    embedding: &[f32],
    _position: usize,
    kv_cache: &mut RawKvCache,
) -> Vec<f32> {
    let hidden = weights.hidden_size;
    let mut x = embedding.to_vec();

    for i in 0..weights.layers.len() {
        let lw = &weights.layers[i];
        let (ref mut cached_k, ref mut cached_v, ref mut cached_len) = kv_cache[i];
        let offset = *cached_len;

        let x_norm = rms_norm_f16(&x, &lw.input_norm_gamma);

        let mut q = gemv(&x_norm, &lw.q_proj);
        let mut k_new = gemv(&x_norm, &lw.k_proj);
        let v_new = gemv(&x_norm, &lw.v_proj);

        apply_qk_norm(&mut q, weights.num_heads, weights.head_dim, &lw.q_norm);
        apply_qk_norm(&mut k_new, weights.num_kv_heads, weights.head_dim, &lw.k_norm);

        // Standard interleaved RoPE
        apply_rope_split(&mut q, weights.num_heads, weights.head_dim, offset, &weights.rope_cos, &weights.rope_sin);
        apply_rope_split(&mut k_new, weights.num_kv_heads, weights.head_dim, offset, &weights.rope_cos, &weights.rope_sin);

        cached_k.extend_from_slice(&k_new);
        cached_v.extend_from_slice(&v_new);
        *cached_len = offset + 1;
        let kv_seq_len = *cached_len;

        // Attention
        let scale = 1.0 / (weights.head_dim as f32).sqrt();
        let kv_stride = weights.num_kv_heads * weights.head_dim;
        let mut attn_output = vec![0.0f32; weights.num_heads * weights.head_dim];

        for h in 0..weights.num_heads {
            let kv_h = h / weights.num_kv_groups;
            let q_offset = h * weights.head_dim;
            let q_vec = &q[q_offset..q_offset + weights.head_dim];
            let mut scores = vec![0.0f32; kv_seq_len];
            for s in 0..kv_seq_len {
                let k_offset = s * kv_stride + kv_h * weights.head_dim;
                let mut dot = 0.0f32;
                for d in 0..weights.head_dim { dot += q_vec[d] * cached_k[k_offset + d]; }
                scores[s] = dot * scale;
            }
            softmax_raw(&mut scores);
            let out_offset = h * weights.head_dim;
            for s in 0..kv_seq_len {
                let v_offset = s * kv_stride + kv_h * weights.head_dim;
                let score = scores[s];
                for d in 0..weights.head_dim {
                    attn_output[out_offset + d] += score * cached_v[v_offset + d];
                }
            }
        }

        let attn_out = gemv(&attn_output, &lw.o_proj);
        for j in 0..hidden { x[j] += attn_out[j]; }

        let x_norm = rms_norm_f16(&x, &lw.post_attn_norm_gamma);
        let gate = gemv(&x_norm, &lw.gate_proj);
        let up = gemv(&x_norm, &lw.up_proj);
        let mut intermediate = vec![0.0f32; gate.len()];
        for j in 0..intermediate.len() { intermediate[j] = silu(gate[j]) * up[j]; }
        let mlp_out = gemv(&intermediate, &lw.down_proj);
        for j in 0..hidden { x[j] += mlp_out[j]; }
    }

    rms_norm_f16(&x, &weights.final_norm_gamma)
}
