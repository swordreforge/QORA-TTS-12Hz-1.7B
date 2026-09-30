//! Talker forward pass — N-layer transformer LM for TTS code generation.
//!
//! Architecture (config-driven, supports 0.6B and 1.7B):
//! - ONE codec_embedding [vocab_size, hidden_size] (shared for all code groups)
//! - text_embedding [text_vocab_size, text_hidden_size]
//! - text_projection: 2-layer MLP (text_hidden_size → hidden_size, both with bias)
//! - N transformer layers: GQA (heads/kv_heads), QK-norm, SwiGLU MLP
//! - ONE codec_head [vocab_size, hidden_size] → predicts first code group
//! - Multimodal RoPE (interleaved, sections [24, 20, 20], theta=1M)

use half::f16;
use crate::gemv::*;
use crate::rope::{self, apply_mrope_interleaved};

// ============================================================
// Weight structures
// ============================================================

pub struct TalkerLayerWeights {
    pub q_proj: Weight,     // [num_heads*head_dim, hidden_size]
    pub k_proj: Weight,     // [num_kv_heads*head_dim, hidden_size]
    pub v_proj: Weight,     // [num_kv_heads*head_dim, hidden_size]
    pub o_proj: Weight,     // [hidden_size, num_heads*head_dim]
    pub q_norm: Vec<f16>,   // [128] per-head QK norm
    pub k_norm: Vec<f16>,   // [128]
    pub gate_proj: Weight,  // [intermediate, hidden_size]
    pub up_proj: Weight,    // [intermediate, hidden_size]
    pub down_proj: Weight,  // [hidden_size, intermediate]
    pub input_norm_gamma: Vec<f16>,
    pub post_attn_norm_gamma: Vec<f16>,
}

pub struct TalkerWeights {
    pub layers: Vec<TalkerLayerWeights>,
    pub codec_embedding: Weight,        // [vocab_size, hidden_size] — ONE shared embedding
    pub text_embedding: Weight,         // [text_vocab_size, text_hidden_size]
    pub text_proj_fc1: Weight,          // [text_hidden_size, hidden_size] (or same if equal)
    pub text_proj_fc1_bias: Vec<f32>,
    pub text_proj_fc2: Weight,          // [hidden_size, hidden_size]
    pub text_proj_fc2_bias: Vec<f32>,
    pub codec_head: Weight,             // [vocab_size, hidden_size] — ONE lm_head for code group 0
    pub final_norm_gamma: Vec<f16>,

    // RoPE tables
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,

    // Config (from config.json)
    pub hidden_size: usize,
    pub text_hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_kv_groups: usize,
    pub mrope_section: Vec<usize>,
}

impl TalkerWeights {
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0;
        for l in &self.layers {
            total += l.q_proj.memory_bytes() + l.k_proj.memory_bytes()
                + l.v_proj.memory_bytes() + l.o_proj.memory_bytes()
                + l.gate_proj.memory_bytes() + l.up_proj.memory_bytes()
                + l.down_proj.memory_bytes();
            total += (l.input_norm_gamma.len() + l.post_attn_norm_gamma.len()
                + l.q_norm.len() + l.k_norm.len()) * 2;
        }
        total += self.codec_embedding.memory_bytes();
        total += self.text_embedding.memory_bytes();
        total += self.text_proj_fc1.memory_bytes() + self.text_proj_fc2.memory_bytes();
        total += (self.text_proj_fc1_bias.len() + self.text_proj_fc2_bias.len()) * 4;
        total += self.codec_head.memory_bytes();
        total += self.final_norm_gamma.len() * 2;
        total += (self.rope_cos.len() + self.rope_sin.len()) * 4;
        total
    }

    pub fn num_layers(&self) -> usize { self.layers.len() }
}

// ============================================================
// QK-Norm helper
// ============================================================

/// Apply RMS norm per-head on Q or K vectors.
/// data: [num_heads * head_dim], norm: [head_dim] (shared across heads)
#[inline]
fn apply_qk_norm(data: &mut [f32], num_heads: usize, head_dim: usize, norm_gamma: &[f16]) {
    for h in 0..num_heads {
        let start = h * head_dim;
        let head_slice = &data[start..start + head_dim];
        let sum_sq: f32 = head_slice.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + 1e-6).sqrt();
        for d in 0..head_dim {
            data[start + d] = data[start + d] * inv_rms * norm_gamma[d].to_f32();
        }
    }
}

// ============================================================
// Embedding
// ============================================================

pub fn embed_text_token(weights: &TalkerWeights, token_id: u32) -> Vec<f32> {
    // text_embedding is [text_vocab_size, text_hidden_size] — may differ from hidden_size
    let raw = embed_lookup(&weights.text_embedding, token_id as usize, weights.text_hidden_size);
    // 2-layer MLP text projection with bias (text_hidden_size → hidden_size)
    let mut h = gemv(&raw, &weights.text_proj_fc1);
    for j in 0..h.len() { h[j] += weights.text_proj_fc1_bias[j]; }
    // SILU activation between layers (NOT GELU!)
    for j in 0..h.len() { h[j] = silu(h[j]); }
    let mut out = gemv(&h, &weights.text_proj_fc2);
    for j in 0..out.len() { out[j] += weights.text_proj_fc2_bias[j]; }
    out
}

/// Lookup embedding for a token from a weight matrix
fn embed_lookup(weight: &Weight, token_id: usize, hidden_size: usize) -> Vec<f32> {
    match weight {
        Weight::Q4(q4) => {
            // Q4 embedding lookup
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
            // F16 embedding lookup
            let start = token_id * hidden_size;
            let end = start + hidden_size;
            f16.data[start..end].iter().map(|&v| v.to_f32()).collect()
        }
    }
}

pub fn embed_codec_token(weights: &TalkerWeights, token_id: u32) -> Vec<f32> {
    embed_lookup(&weights.codec_embedding, token_id as usize, weights.hidden_size)
}

// ============================================================
// Forward pass — single token decode
// ============================================================

/// Forward one token through the talker. Returns hidden state (before codec_head).
pub fn forward_talker_decode(
    weights: &TalkerWeights,
    token_id: u32,
    is_text: bool,
    kv_cache: &mut RawKvCache,
    position: usize,
) -> Vec<f32> {
    let hidden = weights.hidden_size;

    let mut x = if is_text {
        embed_text_token(weights, token_id)
    } else {
        embed_codec_token(weights, token_id)
    };

    let position_ids = [position, position, position];

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

        // Multimodal RoPE (interleaved)
        rope::apply_mrope_interleaved(
            &mut q, weights.num_heads, weights.head_dim,
            &weights.mrope_section, &position_ids,
            &weights.rope_cos, &weights.rope_sin,
        );
        rope::apply_mrope_interleaved(
            &mut k_new, weights.num_kv_heads, weights.head_dim,
            &weights.mrope_section, &position_ids,
            &weights.rope_cos, &weights.rope_sin,
        );

        // KV cache append
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
                for d in 0..weights.head_dim {
                    dot += q_vec[d] * cached_k[k_offset + d];
                }
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
        for j in 0..inter_size {
            intermediate[j] = silu(gate[j]) * up[j];
        }
        let mlp_out = gemv(&intermediate, &lw.down_proj);
        for j in 0..hidden { x[j] += mlp_out[j]; }
    }

    rms_norm_f16(&x, &weights.final_norm_gamma)
}

/// Forward decode with a pre-built embedding (for trailing text fusion).
/// Takes a combined embedding (semantic + acoustic + text) and runs through layers.
/// Returns hidden state with logits at the end.
pub fn forward_with_embedding(
    weights: &TalkerWeights,
    embedding: &[f32],
    kv_cache: &mut RawKvCache,
    position: usize,
) -> Vec<f32> {
    let hidden = weights.hidden_size;

    let mut x = embedding.to_vec();
    let position_ids = [position, position, position];

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

        // Multimodal RoPE (interleaved) — applied to ALL layers
        apply_mrope_interleaved(
            &mut q, weights.num_heads, weights.head_dim,
            &weights.mrope_section, &position_ids, &weights.rope_cos, &weights.rope_sin,
        );
        apply_mrope_interleaved(
            &mut k_new, weights.num_kv_heads, weights.head_dim,
            &weights.mrope_section, &position_ids, &weights.rope_cos, &weights.rope_sin,
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
                for d in 0..weights.head_dim {
                    dot += q_vec[d] * cached_k[k_offset + d];
                }
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
        for j in 0..hidden {
            x[j] += attn_out[j];
        }

        let x_norm = rms_norm_f16(&x, &lw.post_attn_norm_gamma);
        let gate = gemv(&x_norm, &lw.gate_proj);
        let up = gemv(&x_norm, &lw.up_proj);
        let inter_size = gate.len();
        let mut intermediate = vec![0.0f32; inter_size];
        for j in 0..inter_size {
            intermediate[j] = silu(gate[j]) * up[j];
        }
        let mlp_out = gemv(&intermediate, &lw.down_proj);
        for j in 0..hidden {
            x[j] += mlp_out[j];
        }
    }

    // Apply final norm + codec head to get logits
    let normed = rms_norm_f16(&x, &weights.final_norm_gamma);
    let logits = gemv(&normed, &weights.codec_head);

    // Return: [hidden_state, logits]
    let mut result = normed;
    result.extend_from_slice(&logits);
    result
}

/// Forward prefill with pre-built embeddings (for dual-stream architecture).
/// Takes raw embeddings [seq_len * hidden_size] and runs through all layers.
pub fn prefill_talker_raw(
    weights: &TalkerWeights,
    x_in: &[f32],
    seq_len: usize,
    kv_cache: &mut RawKvCache,
) -> Vec<f32> {
    prefill_talker_raw_from(weights, x_in, seq_len, kv_cache, 0)
}

/// Prefill a suffix starting at absolute position `start_pos`, appending to
/// the existing KV cache (which must already hold `start_pos` entries).
/// With empty cache + start_pos 0 this is exactly `prefill_talker_raw`.
/// Used for ICL prefix-cache reuse: the chunk-independent prefix
/// (role + codec header + ref-text part of the ICL block) is prefilled once,
/// each chunk then prefills only its text-dependent suffix.
pub fn prefill_talker_raw_from(
    weights: &TalkerWeights,
    x_in: &[f32],
    seq_len: usize,
    kv_cache: &mut RawKvCache,
    start_pos: usize,
) -> Vec<f32> {
    let hidden = weights.hidden_size;
    let mut x = x_in.to_vec();

    for i in 0..weights.layers.len() {
        let lw = &weights.layers[i];

        let mut x_norm = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            let normed = rms_norm_f16(&x[t * hidden..(t + 1) * hidden], &lw.input_norm_gamma);
            x_norm[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
        }

        let mut q_all = gemm(&x_norm, seq_len, &lw.q_proj);
        let mut k_all = gemm(&x_norm, seq_len, &lw.k_proj);
        let v_all = gemm(&x_norm, seq_len, &lw.v_proj);

        // QK-norm per token
        let q_dim = weights.num_heads * weights.head_dim;
        let k_dim = weights.num_kv_heads * weights.head_dim;
        for t in 0..seq_len {
            apply_qk_norm(&mut q_all[t * q_dim..(t + 1) * q_dim], weights.num_heads, weights.head_dim, &lw.q_norm);
            apply_qk_norm(&mut k_all[t * k_dim..(t + 1) * k_dim], weights.num_kv_heads, weights.head_dim, &lw.k_norm);
        }

        // Multimodal RoPE at absolute positions
        rope::apply_mrope_interleaved_batch(
            &mut q_all, seq_len, weights.num_heads, weights.head_dim,
            &weights.mrope_section, start_pos, &weights.rope_cos, &weights.rope_sin,
        );
        rope::apply_mrope_interleaved_batch(
            &mut k_all, seq_len, weights.num_kv_heads, weights.head_dim,
            &weights.mrope_section, start_pos, &weights.rope_cos, &weights.rope_sin,
        );

        // KV cache append (prefix entries already present)
        let (ref mut cached_k, ref mut cached_v, ref mut cached_len) = kv_cache[i];
        debug_assert_eq!(*cached_len, start_pos,
            "prefix cache length mismatch at layer {i}");
        cached_k.extend_from_slice(&k_all);
        cached_v.extend_from_slice(&v_all);
        *cached_len = start_pos + seq_len;

        // Causal attention over the FULL context (prefix + suffix).
        // Suffix query t1 (global start_pos+t1) attends globals 0..=start_pos+t1,
        // same ascending order as a single full prefill → bit-identical.
        let scale = 1.0 / (weights.head_dim as f32).sqrt();
        let kv_stride = weights.num_kv_heads * weights.head_dim;
        let mut attn_output = vec![0.0f32; seq_len * q_dim];

        for h in 0..weights.num_heads {
            let kv_h = h / weights.num_kv_groups;
            for t1 in 0..seq_len {
                let attend_len = start_pos + t1 + 1;
                let q_off = t1 * q_dim + h * weights.head_dim;
                let q_vec = &q_all[q_off..q_off + weights.head_dim];
                let mut scores = vec![0.0f32; attend_len];
                for s in 0..attend_len {
                    let k_off = s * kv_stride + kv_h * weights.head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..weights.head_dim { dot += q_vec[d] * cached_k[k_off + d]; }
                    scores[s] = dot * scale;
                }
                softmax_raw(&mut scores);
                let out_off = t1 * q_dim + h * weights.head_dim;
                for s in 0..attend_len {
                    let v_off = s * kv_stride + kv_h * weights.head_dim;
                    let score = scores[s];
                    for d in 0..weights.head_dim {
                        attn_output[out_off + d] += score * cached_v[v_off + d];
                    }
                }
            }
        }

        let attn_out_all = gemm(&attn_output, seq_len, &lw.o_proj);
        for j in 0..x.len() { x[j] += attn_out_all[j]; }

        // MLP
        let mut x_norm = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            let normed = rms_norm_f16(&x[t * hidden..(t + 1) * hidden], &lw.post_attn_norm_gamma);
            x_norm[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
        }

        let gate = gemm(&x_norm, seq_len, &lw.gate_proj);
        let up = gemm(&x_norm, seq_len, &lw.up_proj);
        let inter_size = gate.len();
        let mut intermediate = vec![0.0f32; inter_size];
        for j in 0..inter_size { intermediate[j] = silu(gate[j]) * up[j]; }
        let mlp_out = gemm(&intermediate, seq_len, &lw.down_proj);
        for j in 0..x.len() { x[j] += mlp_out[j]; }
    }

    // Return final hidden state from last token (with final norm)
    let last_hidden = &x[(seq_len - 1) * hidden..seq_len * hidden];
    rms_norm_f16(last_hidden, &weights.final_norm_gamma)
}

/// Forward prefill for a sequence of tokens (text or codec).
/// Each token is (token_id, is_text). Text tokens use text_embedding + projection,
/// codec tokens use codec_embedding directly.
pub fn prefill_talker(
    weights: &TalkerWeights,
    tokens: &[(u32, bool)],
    kv_cache: &mut RawKvCache,
) -> Vec<f32> {
    let hidden = weights.hidden_size;
    let seq_len = tokens.len();

    // Embed all tokens based on type
    let mut x = vec![0.0f32; seq_len * hidden];
    for (t, &(tid, is_text)) in tokens.iter().enumerate() {
        let emb = if is_text {
            embed_text_token(weights, tid)
        } else {
            embed_codec_token(weights, tid)
        };
        x[t * hidden..(t + 1) * hidden].copy_from_slice(&emb);
    }

    // Use prefill_talker_raw for the rest
    prefill_talker_raw(weights, &x, seq_len, kv_cache)
}

/// Prefill talker with text tokens + optional voice embeddings (ICL voice cloning).
///
/// Sequence layout:
///   [text_token_0, ..., text_token_N-1]                      (no voice)
///   [text_token_0, ..., text_token_N-1, codec_bos, voice_0, ..., voice_M-1]  (with voice)
///
/// Voice embeddings are pre-computed (sum of all 16 codebook embeddings per timestep).
pub fn prefill_talker_with_voice(
    weights: &TalkerWeights,
    text_tokens: &[u32],
    voice_embeds: Option<&[Vec<f32>]>,
    codec_bos_id: u32,
    kv_cache: &mut RawKvCache,
) -> Vec<f32> {
    let hidden = weights.hidden_size;
    let num_text = text_tokens.len();
    let num_voice = voice_embeds.map(|v| v.len() + 1).unwrap_or(0); // +1 for codec_bos
    let seq_len = num_text + num_voice;

    // Embed all tokens
    let mut x = vec![0.0f32; seq_len * hidden];

    // Text tokens: text_embedding + 2-layer MLP projection
    for t in 0..num_text {
        let emb = embed_text_token(weights, text_tokens[t]);
        x[t * hidden..(t + 1) * hidden].copy_from_slice(&emb);
    }

    // Voice: codec_bos + pre-computed summed embeddings
    if let Some(embeds) = voice_embeds {
        let bos_pos = num_text;
        let bos_emb = embed_codec_token(weights, codec_bos_id);
        x[bos_pos * hidden..(bos_pos + 1) * hidden].copy_from_slice(&bos_emb);

        for (i, emb) in embeds.iter().enumerate() {
            let pos = bos_pos + 1 + i;
            x[pos * hidden..(pos + 1) * hidden].copy_from_slice(emb);
        }
    }

    // Transformer layers (same as prefill_talker)
    for i in 0..weights.layers.len() {
        let lw = &weights.layers[i];

        let mut x_norm = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            let normed = rms_norm_f16(&x[t * hidden..(t + 1) * hidden], &lw.input_norm_gamma);
            x_norm[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
        }

        let mut q_all = gemm(&x_norm, seq_len, &lw.q_proj);
        let mut k_all = gemm(&x_norm, seq_len, &lw.k_proj);
        let v_all = gemm(&x_norm, seq_len, &lw.v_proj);

        // QK-norm per token
        let q_dim = weights.num_heads * weights.head_dim;
        let k_dim = weights.num_kv_heads * weights.head_dim;
        for t in 0..seq_len {
            apply_qk_norm(&mut q_all[t * q_dim..(t + 1) * q_dim], weights.num_heads, weights.head_dim, &lw.q_norm);
            apply_qk_norm(&mut k_all[t * k_dim..(t + 1) * k_dim], weights.num_kv_heads, weights.head_dim, &lw.k_norm);
        }

        // Multimodal RoPE
        rope::apply_mrope_interleaved_batch(
            &mut q_all, seq_len, weights.num_heads, weights.head_dim,
            &weights.mrope_section, 0, &weights.rope_cos, &weights.rope_sin,
        );
        rope::apply_mrope_interleaved_batch(
            &mut k_all, seq_len, weights.num_kv_heads, weights.head_dim,
            &weights.mrope_section, 0, &weights.rope_cos, &weights.rope_sin,
        );

        // KV cache
        let (ref mut cached_k, ref mut cached_v, ref mut cached_len) = kv_cache[i];
        cached_k.extend_from_slice(&k_all);
        cached_v.extend_from_slice(&v_all);
        *cached_len = seq_len;

        // Causal attention
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

        let o_out = gemm(&attn_output, seq_len, &lw.o_proj);
        for j in 0..seq_len * hidden { x[j] += o_out[j]; }

        let mut x_norm2 = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            let normed = rms_norm_f16(&x[t * hidden..(t + 1) * hidden], &lw.post_attn_norm_gamma);
            x_norm2[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
        }

        let gate = gemm(&x_norm2, seq_len, &lw.gate_proj);
        let up = gemm(&x_norm2, seq_len, &lw.up_proj);
        let inter_size = lw.gate_proj.n();
        let mut intermediate = vec![0.0f32; seq_len * inter_size];
        for j in 0..seq_len * inter_size {
            intermediate[j] = silu(gate[j]) * up[j];
        }
        let mlp_out = gemm(&intermediate, seq_len, &lw.down_proj);
        for j in 0..seq_len * hidden { x[j] += mlp_out[j]; }

        if i % 6 == 0 || i == weights.layers.len() - 1 {
            eprintln!("  Prefill layer {i}/{}", weights.layers.len());
        }
    }

    let last = &x[(seq_len - 1) * hidden..seq_len * hidden];
    rms_norm_f16(last, &weights.final_norm_gamma)
}

/// Apply codec_head to get logits for code group 0.
pub fn apply_codec_head(weights: &TalkerWeights, hidden_state: &[f32]) -> Vec<f32> {
    gemv(hidden_state, &weights.codec_head)
}
