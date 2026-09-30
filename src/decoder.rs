//! Speech Decoder — converts 16 codebook indices per timestep to 24kHz audio.
//!
//! Architecture (from speech_tokenizer weights):
//!   Codebook: 16 VQ codebooks (dim=256) + output_proj → 512
//!   Pre-conv: CausalConv1d(512→1024, k=3)
//!   Pre-transformer: input_proj(1024→512) → 8 layers → norm → output_proj(512→1024)
//!     Each layer: RMSNorm → SelfAttn(16 heads, hdim=64, sliding_window=72) → LayerScale
//!                 RMSNorm → SwiGLU MLP(512→1024→512) → LayerScale
//!   Upsample: 2× ConvTranspose1d(1024, k=2, s=2) + ConvNeXt → [1024, 4T]
//!   Vocos: CausalConv1d(1024→1536, k=7) → 4 blocks (rates [8,5,4,3])
//!     Each block: SnakeBeta → ConvTranspose1d → 3 residuals
//!   Output: SnakeBeta(96) → Conv1d(96→1, k=7) → clamp [-1,1]
//!   Total upsample: 2×2×8×5×4×3 = 1920x → 12.5Hz × 1920 = 24kHz

use std::time::Instant;
use crate::conv::*;
use crate::gemv::softmax_raw;

// ============================================================
// Weight structures
// ============================================================

pub struct SpeechDecoderWeights {
    // VQ codebooks (EMA-normalized: embedding_sum / cluster_usage)
    pub first_codebook: Vec<f32>,       // [2048, 256]
    pub first_output_proj: Vec<f32>,    // [256, 512] (transposed for gemv)
    pub rest_codebooks: Vec<Vec<f32>>,  // 15 × [2048, 256]
    pub rest_output_proj: Vec<f32>,     // [256, 512] (transposed for gemv, shared)

    // Pre-conv: CausalConv1d(512→1024, k=3)
    pub pre_conv: Conv1dWeight,

    // Pre-transformer
    pub tf_input_w: Vec<f32>,   // [1024, 512] transposed for gemv
    pub tf_input_b: Vec<f32>,   // [512]
    pub tf_layers: Vec<DecoderTfLayer>,
    pub tf_norm: Vec<f32>,      // [512]
    pub tf_output_w: Vec<f32>,  // [512, 1024] transposed for gemv
    pub tf_output_b: Vec<f32>,  // [1024]

    // Upsample: 2 stages
    pub upsample: Vec<UpsampleStage>,

    // Vocos initial conv: CausalConv1d(1024→1536, k=7)
    pub vocos_init: Conv1dWeight,

    // Vocos blocks: 4
    pub vocos_blocks: Vec<VocosBlock>,

    // Output
    pub final_snake_alpha: Vec<f32>,  // [96]
    pub final_snake_beta: Vec<f32>,   // [96]
    pub output_conv: Conv1dWeight,    // Conv1d(96→1, k=7)

    // RoPE tables
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,

    // Config
    pub hidden_size: usize,       // 512
    pub num_heads: usize,         // 16
    pub head_dim: usize,          // 64
    pub intermediate_size: usize, // 1024
    pub sliding_window: usize,    // 72
    pub codebook_dim: usize,      // 256
}

pub struct DecoderTfLayer {
    pub q_proj: Vec<f32>,           // [512, 1024] for gemv
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,           // [1024, 512] for gemv
    pub gate_proj: Vec<f32>,        // [512, 1024] for gemv
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,        // [1024, 512] for gemv
    pub input_norm: Vec<f32>,       // [512]
    pub post_attn_norm: Vec<f32>,   // [512]
    pub attn_scale: Vec<f32>,       // [512] layer scale
    pub mlp_scale: Vec<f32>,        // [512] layer scale
}

pub struct UpsampleStage {
    pub conv_t: ConvTranspose1dWeight,
    pub dw_conv: DepthwiseConv1dWeight,
    pub norm_w: Vec<f32>,     // [1024] LayerNorm weight
    pub norm_b: Vec<f32>,     // [1024] LayerNorm bias
    pub pw1_w: Vec<f32>,      // [1024, 4096] transposed for gemv
    pub pw1_b: Vec<f32>,      // [4096]
    pub pw2_w: Vec<f32>,      // [4096, 1024] transposed for gemv
    pub pw2_b: Vec<f32>,      // [1024]
    pub gamma: Vec<f32>,      // [1024] residual scale
    pub channels: usize,
}

pub struct VocosBlock {
    pub pre_snake_alpha: Vec<f32>,
    pub pre_snake_beta: Vec<f32>,
    pub upsample: ConvTranspose1dWeight,
    pub residuals: Vec<VocosResidual>,  // 3
}

pub struct VocosResidual {
    pub act1_alpha: Vec<f32>,
    pub act1_beta: Vec<f32>,
    pub conv1: Conv1dWeight,      // k=7, causal
    pub act2_alpha: Vec<f32>,
    pub act2_beta: Vec<f32>,
    pub conv2: Conv1dWeight,      // k=1
}

impl SpeechDecoderWeights {
    pub fn memory_bytes(&self) -> usize {
        let mut t = 0;
        t += self.first_codebook.len() * 4;
        t += self.first_output_proj.len() * 4;
        for cb in &self.rest_codebooks { t += cb.len() * 4; }
        t += self.rest_output_proj.len() * 4;
        t += self.pre_conv.weight.len() * 4;
        t += self.tf_input_w.len() * 4 + self.tf_output_w.len() * 4;
        for l in &self.tf_layers {
            t += (l.q_proj.len() + l.k_proj.len() + l.v_proj.len() + l.o_proj.len()) * 4;
            t += (l.gate_proj.len() + l.up_proj.len() + l.down_proj.len()) * 4;
        }
        for u in &self.upsample {
            t += u.conv_t.weight.len() * 4 + u.dw_conv.weight.len() * 4;
            t += u.pw1_w.len() * 4 + u.pw2_w.len() * 4;
        }
        t += self.vocos_init.weight.len() * 4;
        for vb in &self.vocos_blocks {
            t += vb.upsample.weight.len() * 4;
            for r in &vb.residuals { t += r.conv1.weight.len() * 4 + r.conv2.weight.len() * 4; }
        }
        t += self.output_conv.weight.len() * 4;
        t
    }
}

// ============================================================
// Codebook dequantize
// ============================================================

/// Dequantize 16 codebook indices → [512, T] channel-first.
fn codebook_lookup(w: &SpeechDecoderWeights, codes: &[Vec<u32>]) -> Vec<f32> {
    let t = codes[0].len();
    let cb_dim = w.codebook_dim; // 256

    // Sum embeddings in 256-dim for each VQ group, then project to 512
    let mut out = vec![0.0f32; 512 * t];

    // Group 0 (rvq_first)
    for step in 0..t {
        let idx = codes[0][step] as usize;
        let emb = &w.first_codebook[idx * cb_dim..(idx + 1) * cb_dim];
        // Project 256→512 and accumulate
        let proj = f32_gemv(emb, &w.first_output_proj, cb_dim, 512);
        for c in 0..512 {
            out[c * t + step] += proj[c];
        }
    }

    // Groups 1-15 (rvq_rest, shared output_proj)
    for step in 0..t {
        // Sum all 15 codebook lookups in 256-dim
        let mut sum_emb = vec![0.0f32; cb_dim];
        for g in 0..15 {
            let idx = codes[g + 1][step] as usize;
            let emb = &w.rest_codebooks[g][idx * cb_dim..(idx + 1) * cb_dim];
            for d in 0..cb_dim {
                sum_emb[d] += emb[d];
            }
        }
        // Project sum 256→512
        let proj = f32_gemv(&sum_emb, &w.rest_output_proj, cb_dim, 512);
        for c in 0..512 {
            out[c * t + step] += proj[c];
        }
    }

    // Debug: print codebook lookup stats
    let vq_min = out.iter().copied().fold(f32::INFINITY, f32::min);
    let vq_max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let vq_mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
    eprintln!("  VQ output: range=[{vq_min:.4}, {vq_max:.4}], mean={vq_mean:.6}");
    eprintln!("  VQ [0,:5]: {:.4} {:.4} {:.4} {:.4} {:.4}",
        out[0*t], out[0*t+1], out[0*t+2], out[0*t+3], out[0*t+4]);
    eprintln!("  VQ [1,:5]: {:.4} {:.4} {:.4} {:.4} {:.4}",
        out[1*t], out[1*t+1], out[1*t+2], out[1*t+3], out[1*t+4]);

    out
}

// ============================================================
// Transformer (full-sequence, causal + sliding window)
// ============================================================

fn decoder_transformer(w: &SpeechDecoderWeights, x_ch: &[f32]) -> Vec<f32> {
    let hidden = w.hidden_size; // 512
    let attn_dim = w.num_heads * w.head_dim; // 1024
    let _channels = x_ch.len() / (x_ch.len() / 1024); // infer from pre_conv output
    let in_dim = 1024; // latent_dim
    let t_len = x_ch.len() / in_dim;

    // Transpose [channels=1024, T] → [T, 1024]
    let mut x_row = vec![0.0f32; t_len * in_dim];
    for c in 0..in_dim {
        for t in 0..t_len {
            x_row[t * in_dim + c] = x_ch[c * t_len + t];
        }
    }

    // Input projection: [T, 1024] → [T, 512]
    let mut x = vec![0.0f32; t_len * hidden];
    for t in 0..t_len {
        let proj = f32_gemv_bias(
            &x_row[t * in_dim..(t + 1) * in_dim],
            &w.tf_input_w, in_dim, hidden, &w.tf_input_b,
        );
        x[t * hidden..(t + 1) * hidden].copy_from_slice(&proj);
    }

    // 8 transformer layers
    for (li, layer) in w.tf_layers.iter().enumerate() {
        // Pre-compute Q, K, V for all positions
        let mut all_q = vec![0.0f32; t_len * attn_dim];
        let mut all_k = vec![0.0f32; t_len * attn_dim];
        let mut all_v = vec![0.0f32; t_len * attn_dim];

        for t in 0..t_len {
            let x_norm = rms_norm_f32(&x[t * hidden..(t + 1) * hidden], &layer.input_norm, 1e-5);
            let q = f32_gemv(&x_norm, &layer.q_proj, hidden, attn_dim);
            let k = f32_gemv(&x_norm, &layer.k_proj, hidden, attn_dim);
            let v = f32_gemv(&x_norm, &layer.v_proj, hidden, attn_dim);
            all_q[t * attn_dim..(t + 1) * attn_dim].copy_from_slice(&q);
            all_k[t * attn_dim..(t + 1) * attn_dim].copy_from_slice(&k);
            all_v[t * attn_dim..(t + 1) * attn_dim].copy_from_slice(&v);
        }

        // Apply RoPE
        let _half_dim = w.head_dim / 2;
        for t in 0..t_len {
            apply_rope_split_half(
                &mut all_q[t * attn_dim..(t + 1) * attn_dim],
                w.num_heads, w.head_dim, t, &w.rope_cos, &w.rope_sin,
            );
            apply_rope_split_half(
                &mut all_k[t * attn_dim..(t + 1) * attn_dim],
                w.num_heads, w.head_dim, t, &w.rope_cos, &w.rope_sin,
            );
        }

        // Causal attention with sliding window
        let scale = 1.0 / (w.head_dim as f32).sqrt();
        for t in 0..t_len {
            let start = if t + 1 > w.sliding_window { t + 1 - w.sliding_window } else { 0 };
            let attend_len = t + 1 - start;

            let mut attn_out = vec![0.0f32; attn_dim];
            for h in 0..w.num_heads {
                let q_off = t * attn_dim + h * w.head_dim;
                let mut scores = vec![0.0f32; attend_len];
                for s in 0..attend_len {
                    let k_off = (start + s) * attn_dim + h * w.head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..w.head_dim {
                        dot += all_q[q_off + d] * all_k[k_off + d];
                    }
                    scores[s] = dot * scale;
                }
                softmax_raw(&mut scores);
                for s in 0..attend_len {
                    let v_off = (start + s) * attn_dim + h * w.head_dim;
                    let sc = scores[s];
                    for d in 0..w.head_dim {
                        attn_out[h * w.head_dim + d] += sc * all_v[v_off + d];
                    }
                }
            }

            // O projection + layer scale + residual
            let o_out = f32_gemv(&attn_out, &layer.o_proj, attn_dim, hidden);
            for j in 0..hidden {
                x[t * hidden + j] += o_out[j] * layer.attn_scale[j];
            }
        }

        // MLP
        for t in 0..t_len {
            let x_norm = rms_norm_f32(&x[t * hidden..(t + 1) * hidden], &layer.post_attn_norm, 1e-5);
            let gate = f32_gemv(&x_norm, &layer.gate_proj, hidden, w.intermediate_size);
            let up = f32_gemv(&x_norm, &layer.up_proj, hidden, w.intermediate_size);
            let mut inter = vec![0.0f32; w.intermediate_size];
            for j in 0..w.intermediate_size {
                let g = gate[j];
                inter[j] = (g / (1.0 + (-g).exp())) * up[j]; // SiLU(gate) * up
            }
            let mlp_out = f32_gemv(&inter, &layer.down_proj, w.intermediate_size, hidden);
            for j in 0..hidden {
                x[t * hidden + j] += mlp_out[j] * layer.mlp_scale[j];
            }
        }

        if li % 2 == 0 {
            eprintln!("    Transformer layer {li}/{}...", w.tf_layers.len());
        }
    }

    // Final norm
    for t in 0..t_len {
        let normed = rms_norm_f32(&x[t * hidden..(t + 1) * hidden], &w.tf_norm, 1e-5);
        x[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
    }

    // Output projection: [T, 512] → [T, 1024]
    let mut out_row = vec![0.0f32; t_len * in_dim];
    for t in 0..t_len {
        let proj = f32_gemv_bias(
            &x[t * hidden..(t + 1) * hidden],
            &w.tf_output_w, hidden, in_dim, &w.tf_output_b,
        );
        out_row[t * in_dim..(t + 1) * in_dim].copy_from_slice(&proj);
    }

    // Transpose [T, 1024] → [1024, T]
    let mut out_ch = vec![0.0f32; in_dim * t_len];
    for c in 0..in_dim {
        for t in 0..t_len {
            out_ch[c * t_len + t] = out_row[t * in_dim + c];
        }
    }
    out_ch
}

// ============================================================
// ConvNeXt block
// ============================================================

fn convnext_forward(input: &[f32], stage: &UpsampleStage) -> Vec<f32> {
    let ch = stage.channels;
    let length = input.len() / ch;
    let residual = input.to_vec();

    // Depthwise conv (same-padding)
    let dw_out = depthwise_conv1d(input, &stage.dw_conv);

    // LayerNorm per-timestep across channels
    let normed = layer_norm_channel_first(&dw_out, &stage.norm_w, &stage.norm_b, ch, 1e-5);

    // Pointwise conv1+2, batched over time (was: per-timestep gather →
    // scalar f32_gemv → scatter, re-reading weights `length` times).
    // Layouts already align: normed is row-major [ch, length], pw1_w is
    // [inner, ch] → one GEMM gives row-major [inner, length] = channel-first.
    // Same per-element order (bias-first, k-ascending) → bit-exact.
    let inner_dim = stage.pw1_b.len();
    // pw weights are stored [k, n] for per-t GEMV (W[ic][c]); batched GEMM
    // needs [inner, ch]: transpose once per decode (ms-level, vs ~1s GEMM).
    let w1t = crate::conv::transpose2d(&stage.pw1_w, ch, inner_dim);
    let mut pw1_out = crate::conv::f32_gemm_bias(
        &w1t, &normed, &stage.pw1_b, inner_dim, length, ch,
    );
    // GELU is elementwise (order-free) → in place, still channel-first.
    for v in pw1_out.iter_mut() {
        *v = gelu(*v);
    }

    // Pointwise conv2: [4*ch] → [ch], same batching (output already [ch, length]).
    let w2t = crate::conv::transpose2d(&stage.pw2_w, inner_dim, ch);
    let pw2_out = crate::conv::f32_gemm_bias(
        &w2t, &pw1_out, &stage.pw2_b, ch, length, inner_dim,
    );

    // gamma * output + residual
    let mut output = vec![0.0f32; ch * length];
    for c in 0..ch {
        let g = stage.gamma[c];
        for t in 0..length {
            output[c * length + t] = residual[c * length + t] + g * pw2_out[c * length + t];
        }
    }
    output
}

// ============================================================
// Full decode pipeline
// ============================================================

/// Decode audio codes to waveform.
/// codes: [16][T] — 16 codebook indices per timestep.
/// Returns: f32 audio samples at 24kHz.
///
/// `warmup`: optional reference codes [16][Tv] prepended before decoding
/// (official ICL parity: the decoder must not start cold — conv state and
/// sliding-window attention need left context, otherwise chunk attacks
/// render in the wrong voice). The reference portion is trimmed afterwards
/// with the official proportional cut: `Tv / (Tv + T) * len(samples)`.
pub fn decode_to_audio(
    weights: &SpeechDecoderWeights,
    codes: &[Vec<u32>],
    warmup: Option<&[Vec<u32>]>,
) -> Vec<f32> {
    let t = codes[0].len();
    if t == 0 {
        eprintln!("  No audio codes to decode!");
        return Vec::new();
    }

    // Official ICL parity: prepend ref codes so the decoder starts warm.
    // trim = (Tv, Tv + T); cut = Tv / (Tv + T) * samples at the end.
    let full: Vec<Vec<u32>>;
    let (work, trim): (&[Vec<u32>], Option<(usize, usize)>) = match warmup {
        Some(w)
            if w.len() == codes.len()
                && !w.is_empty()
                && w.iter().all(|q| q.len() == w[0].len())
                && !w[0].is_empty() =>
        {
            let tv = w[0].len();
            eprintln!("  Decoder warmup: {tv} ref frames prepended");
            full = w
                .iter()
                .zip(codes.iter())
                .map(|(a, b)| {
                    let mut v = Vec::with_capacity(a.len() + b.len());
                    v.extend_from_slice(a);
                    v.extend_from_slice(b);
                    v
                })
                .collect();
            (&full, Some((tv, tv + t)))
        }
        _ => (codes, None),
    };
    let t = work[0].len();
    eprintln!("  Decoding {t} timesteps to audio...");

    // 1. Codebook lookup → [512, T]
    let vq_out = codebook_lookup(weights, work);
    eprintln!("  Codebook lookup: [512, {t}]");

    // 2. Pre-conv: CausalConv1d(512→1024, k=3) → [1024, T]
    let pre_conv_out = causal_conv1d(&vq_out, &weights.pre_conv);
    let t1 = pre_conv_out.len() / 1024;
    {
        let mn = pre_conv_out.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = pre_conv_out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let avg: f32 = pre_conv_out.iter().sum::<f32>() / pre_conv_out.len() as f32;
        eprintln!("  Pre-conv: [1024, {t1}], range=[{mn:.4}, {mx:.4}], mean={avg:.6}");
        eprintln!("  Pre-conv [0,:5]: {:.4} {:.4} {:.4} {:.4} {:.4}",
            pre_conv_out[0*t1], pre_conv_out[0*t1+1], pre_conv_out[0*t1+2],
            pre_conv_out[0*t1+3], pre_conv_out[0*t1+4]);
    }

    // 3. Transformer → [1024, T] (single-threaded per-position GEMVs —
    // see execution-graph notes; timed separately from Vocos below)
    let s_tf = Instant::now();
    let tf_out = decoder_transformer(weights, &pre_conv_out);
    let t_tf = s_tf.elapsed();
    let t2 = tf_out.len() / 1024;
    {
        let mn = tf_out.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = tf_out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let avg: f32 = tf_out.iter().sum::<f32>() / tf_out.len() as f32;
        eprintln!("  Transformer: [1024, {t2}], range=[{mn:.4}, {mx:.4}], mean={avg:.6}");
        eprintln!("  Transformer [0,:5]: {:.4} {:.4} {:.4} {:.4} {:.4}",
            tf_out[0*t2], tf_out[0*t2+1], tf_out[0*t2+2],
            tf_out[0*t2+3], tf_out[0*t2+4]);
    }

    // 4. Upsample: 2× ConvT(s=2) + ConvNeXt → [1024, 4T]
    let mut signal = tf_out;
    let mut t_up = std::time::Duration::ZERO;
    for (i, stage) in weights.upsample.iter().enumerate() {
        let s = Instant::now();
        signal = conv_transpose1d(&signal, &stage.conv_t);
        // Causal right-crop: remove (kernel_size - stride) from right
        let right_crop = stage.conv_t.kernel_size - stage.conv_t.stride;
        if right_crop > 0 {
            let ch = stage.channels;
            let full_len = signal.len() / ch;
            let cropped_len = full_len - right_crop;
            let mut cropped = vec![0.0f32; ch * cropped_len];
            for c in 0..ch {
                cropped[c * cropped_len..(c + 1) * cropped_len]
                    .copy_from_slice(&signal[c * full_len..c * full_len + cropped_len]);
            }
            signal = cropped;
        }
        signal = convnext_forward(&signal, stage);
        t_up += s.elapsed();
        let ch = stage.channels;
        let len = signal.len() / ch;
        eprintln!("  Upsample {i}: [{ch}, {len}]");
    }

    // 5. Vocos initial conv: CausalConv1d(1024→1536, k=7)
    signal = causal_conv1d(&signal, &weights.vocos_init);
    let vocos_ch = weights.vocos_init.out_channels;
    let vocos_len = signal.len() / vocos_ch;
    eprintln!("  Vocos init: [{vocos_ch}, {vocos_len}]");

    // 6. Vocos blocks
    let mut t_blk = std::time::Duration::ZERO;
    for (i, vb) in weights.vocos_blocks.iter().enumerate() {
        let s = Instant::now();
        let mut t_sn = std::time::Duration::ZERO;
        let mut t_tr = std::time::Duration::ZERO;
        let mut t_ca = std::time::Duration::ZERO;
        let in_ch = vb.pre_snake_alpha.len();

        // SnakeBeta → ConvTranspose1d (causal: right-crop kernel_size - stride)
        let q = Instant::now();
        signal = snake_beta(&signal, &vb.pre_snake_alpha, &vb.pre_snake_beta, in_ch);
        t_sn += q.elapsed();
        let q = Instant::now();
        signal = conv_transpose1d(&signal, &vb.upsample);
        t_tr += q.elapsed();
        let out_ch = vb.upsample.out_channels;
        let right_crop = vb.upsample.kernel_size - vb.upsample.stride;
        if right_crop > 0 {
            let full_len = signal.len() / out_ch;
            let cropped_len = full_len - right_crop;
            let mut cropped = vec![0.0f32; out_ch * cropped_len];
            for c in 0..out_ch {
                cropped[c * cropped_len..(c + 1) * cropped_len]
                    .copy_from_slice(&signal[c * full_len..c * full_len + cropped_len]);
            }
            signal = cropped;
        }
        let len = signal.len() / out_ch;

        // 3 residual units
        for res in &vb.residuals {
            let residual = signal.clone();
            let q = Instant::now();
            signal = snake_beta(&signal, &res.act1_alpha, &res.act1_beta, out_ch);
            t_sn += q.elapsed();
            let q = Instant::now();
            signal = causal_conv1d(&signal, &res.conv1);
            t_ca += q.elapsed();
            let q = Instant::now();
            signal = snake_beta(&signal, &res.act2_alpha, &res.act2_beta, out_ch);
            t_sn += q.elapsed();
            let q = Instant::now();
            signal = causal_conv1d(&signal, &res.conv2);
            t_ca += q.elapsed();
            // Add residual
            for j in 0..signal.len() {
                signal[j] += residual[j];
            }
        }
        eprintln!("  Vocos block {i}: [{out_ch}, {len}] in {:.1?} (snake {:.1?}, trans {:.1?}, causal {:.1?})",
            s.elapsed(), t_sn, t_tr, t_ca);
        t_blk += s.elapsed();
    }

    // 7. Final SnakeBeta + output conv → [1, samples]
    let final_ch = weights.final_snake_alpha.len();
    signal = snake_beta(&signal, &weights.final_snake_alpha, &weights.final_snake_beta, final_ch);
    signal = causal_conv1d(&signal, &weights.output_conv);

    // Clamp to [-1, 1]
    for s in &mut signal {
        *s = s.clamp(-1.0, 1.0);
    }

    // Official ICL parity: trim the prepended reference portion.
    if let Some((tv, total)) = trim {
        let cut = warmup_trim_len(tv, total, signal.len());
        eprintln!("  Decoder warmup trimmed: {cut} samples ({tv}/{total} frames)");
        signal.drain(..cut);
    }

    eprintln!("  Audio: {} samples ({:.1}s at 24kHz)", signal.len(), signal.len() as f32 / 24000.0);
    eprintln!("  Decode split: transformer {:.1?}, upsample {:.1?}, vocos-blocks {:.1?}, other {:.1?}",
        t_tf, t_up, t_blk, s_tf.elapsed() - t_tf - t_up - t_blk);
    if let Some(rep) = crate::conv::snake_stats_report() {
        eprintln!("  {rep}");
    }
    signal
}

// ============================================================
// F32 compute helpers
// ============================================================

fn rms_norm_f32(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = x[i] * inv_rms * gamma[i];
    }
    out
}

/// Matrix-vector multiply. Weight layout: [k=input_dim, n=output_dim].
fn f32_gemv(input: &[f32], weight: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for ki in 0..k {
        let iv = input[ki];
        if iv == 0.0 { continue; }
        let w_start = ki * n;
        for j in 0..n {
            out[j] += iv * weight[w_start + j];
        }
    }
    out
}

/// Matrix-vector multiply with bias.
fn f32_gemv_bias(input: &[f32], weight: &[f32], k: usize, n: usize, bias: &[f32]) -> Vec<f32> {
    let mut out = bias.to_vec();
    for ki in 0..k {
        let iv = input[ki];
        if iv == 0.0 { continue; }
        let w_start = ki * n;
        for j in 0..n {
            out[j] += iv * weight[w_start + j];
        }
    }
    out
}

/// Apply split-half RoPE: pairs (x[i], x[i+half_dim]) for rotation.
/// This matches the `rotate_half` convention used in HuggingFace transformers.
fn apply_rope_split_half(
    data: &mut [f32], num_heads: usize, head_dim: usize,
    position: usize, cos_table: &[f32], sin_table: &[f32],
) {
    let half_dim = head_dim / 2;
    let cos_off = position * half_dim;
    if cos_off + half_dim > cos_table.len() { return; }
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half_dim {
            let x0 = data[base + i];
            let x1 = data[base + half_dim + i];
            let c = cos_table[cos_off + i];
            let s = sin_table[cos_off + i];
            // rotate_half: [-x2, x1], so:
            // output[i] = x0 * cos - x1 * sin
            // output[i+half] = x1 * cos + x0 * sin
            data[base + i] = x0 * c - x1 * s;
            data[base + half_dim + i] = x1 * c + x0 * s;
        }
    }
}

/// Official proportional cut for warmed decode output:
/// `Tv / (Tv + T) * samples`. Pure helper so the arithmetic is unit-tested.
pub fn warmup_trim_len(tv_frames: usize, total_frames: usize, out_len: usize) -> usize {
    if total_frames == 0 {
        return 0;
    }
    (tv_frames * out_len / total_frames).min(out_len)
}
/// LayerNorm applied per-timestep across channels (channel-first format).
fn layer_norm_channel_first(
    input: &[f32], gamma: &[f32], beta: &[f32], channels: usize, eps: f32,
) -> Vec<f32> {
    let length = input.len() / channels;
    let mut output = vec![0.0f32; input.len()];
    for t in 0..length {
        let mut mean = 0.0f32;
        for c in 0..channels {
            mean += input[c * length + t];
        }
        mean /= channels as f32;
        let mut var = 0.0f32;
        for c in 0..channels {
            let d = input[c * length + t] - mean;
            var += d * d;
        }
        let inv_std = 1.0 / (var / channels as f32 + eps).sqrt();
        for c in 0..channels {
            output[c * length + t] = (input[c * length + t] - mean) * inv_std * gamma[c] + beta[c];
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_trim_len() {
        // 141 ref + 34 gen frames, 175*1920 samples: cut == 141*1920 exactly
        assert_eq!(warmup_trim_len(141, 175, 175 * 1920), 141 * 1920);
        // rounding down + clamp
        assert_eq!(warmup_trim_len(1, 3, 100), 33);
        assert_eq!(warmup_trim_len(0, 10, 100), 0);
        assert_eq!(warmup_trim_len(5, 0, 100), 0);
        assert_eq!(warmup_trim_len(10, 10, 50), 50);
    }
}
