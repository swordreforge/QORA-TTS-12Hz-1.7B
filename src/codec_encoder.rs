//! Codec encoder (Mimi/SEANet) for ICL voice cloning.
//!
//! Ports `Qwen3TTSTokenizerV2Encoder` (a `MimiModel` with decoder parts removed):
//! raw waveform [1, T] → SEANet causal convs (24000Hz → 25Hz) → 8-layer transformer
//! (RoPE + sliding-window 250) → stride-2 downsample (25Hz → 12.5Hz) → split RVQ
//! (1 semantic + 31 acoustic codebooks, codebook_dim 256) → codes [16, T/1920].
//!
//! Reference: transformers `models/mimi/modeling_mimi.py` + Space
//! `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`.
//! Safetensors keys use the V2 nested layout, e.g.
//! `encoder.encoder.layers.0.conv.weight`, `encoder.downsample.conv.weight`,
//! `encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum`.

use safetensors::SafeTensors;

// ============================================================
// Weight structures
// ============================================================

/// Causal Conv1d with Mimi padding semantics (see `causal_pad` below).
/// `pad_replicate == true` selects edge-replicate padding (used only by the
/// final stride-2 downsample); otherwise zero padding (`pad_mode="constant"`).
pub struct MimiConv {
    pub weight: Vec<f32>, // [out_ch, in_ch, k]
    pub bias: Option<Vec<f32>>,
    pub in_ch: usize,
    pub out_ch: usize,
    pub k: usize,
    pub stride: usize,
    pub dilation: usize,
    pub pad_replicate: bool,
}

/// SEANet residual block: ELU → Conv(k=3) → ELU → Conv(k=1) + identity.
/// (`use_conv_shortcut=false`, so the shortcut is always identity.)
pub struct SeanetResBlock {
    pub convs: [MimiConv; 2],
}

/// One downsampling stage: ResNet block(s) + ELU + strided conv.
/// With `num_residual_layers=1` there is exactly one res block per stage.
pub struct DownStage {
    pub res: SeanetResBlock,
    pub down: MimiConv,
}

/// SEANet conv stack: initial conv → 4 stages (ratios 4,5,6,8) → ELU → final conv.
/// Output rate: 24000 / (4*5*6*8) = 25 Hz.
pub struct SeanetEncoder {
    pub initial: MimiConv, // [64, 1, 7], s=1
    pub stages: Vec<DownStage>,
    pub final_conv: MimiConv, // [512, 512, 3], s=1
}

/// One encoder transformer layer (pre-LN, RoPE, sliding-window causal attn,
/// LayerScale on both branches, GELU MLP without biases).
pub struct MimiTransformerLayerW {
    pub q_proj: Vec<f32>, // [512, 512] row-major [out, in]
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub in_ln_w: Vec<f32>,
    pub in_ln_b: Vec<f32>,
    pub post_ln_w: Vec<f32>,
    pub post_ln_b: Vec<f32>,
    pub fc1: Vec<f32>, // [2048, 512]
    pub fc2: Vec<f32>, // [512, 2048]
    pub attn_scale: Vec<f32>, // [512]
    pub mlp_scale: Vec<f32>,  // [512]
}

/// One side (semantic / acoustic) of the split RVQ.
pub struct RvqSide {
    pub input_proj: Vec<f32>,  // [256, 512] (k=1 conv, no bias)
    pub output_proj: Vec<f32>, // [512, 256] (decode path; loaded for completeness)
    /// Per-layer codebook embeddings [2048, 256], computed as
    /// embed_sum / cluster_usage (matches `MimiEuclideanCodebook.embed`).
    pub codebooks: Vec<Vec<f32>>,
}

pub struct CodecEncoderWeights {
    pub seanet: SeanetEncoder,
    pub transformer: Vec<MimiTransformerLayerW>, // 8 layers
    pub downsample: MimiConv,                    // [512, 512, 4], s=2, replicate pad
    pub semantic: RvqSide,                       // 1 codebook
    pub acoustic: RvqSide,                       // 31 codebooks
}

// ============================================================
// Mimi causal padding semantics
// ============================================================

/// Extra right padding so the padded length lands on the stride grid.
/// Mirrors `MimiConv1d._get_extra_padding_for_conv1d`:
/// n = ceil((L - eff + pt) / s + 1) - 1; extra = n*s + eff - pt - L (>= 0).
pub fn extra_padding(input_len: usize, eff_kernel: usize, padding_total: usize, stride: usize) -> usize {
    let y = (input_len as f64 - eff_kernel as f64 + padding_total as f64) / stride as f64 + 1.0;
    let n_frames = y.ceil() as i64 - 1;
    let ideal = n_frames * stride as i64 + eff_kernel as i64 - padding_total as i64;
    (ideal - input_len as i64).max(0) as usize
}

/// Effective kernel size with dilation.
pub fn eff_kernel_size(k: usize, dilation: usize) -> usize {
    (k - 1) * dilation + 1
}

/// `padding_total = eff_kernel - stride` (== left pad for causal convs).
pub fn padding_total(k: usize, stride: usize, dilation: usize) -> usize {
    eff_kernel_size(k, dilation).saturating_sub(stride)
}

/// Causal pad of one channel: left=`padding_total`, right=`extra`.
/// Zeros for `constant` mode, edge replication for `replicate` mode.
pub fn causal_pad_channel(signal: &[f32], pad_left: usize, pad_right: usize, replicate: bool) -> Vec<f32> {
    let n = signal.len();
    let mut out = Vec::with_capacity(pad_left + n + pad_right);
    for _ in 0..pad_left {
        out.push(if replicate && n > 0 { signal[0] } else { 0.0 });
    }
    out.extend_from_slice(signal);
    for _ in 0..pad_right {
        out.push(if replicate && n > 0 { signal[n - 1] } else { 0.0 });
    }
    out
}

/// ELU (alpha=1), as `nn.ELU()` in the SEANet stack.
pub fn elu(x: f32) -> f32 {
    if x >= 0.0 { x } else { x.exp() - 1.0 }
}

// ============================================================
// Exact erf + GELU (matches torch ACT2FN["gelu"]; repo conv::gelu is tanh approx)
// ============================================================

/// erf via Abramowitz–Stegun 7.1.26 (|eps| <= 1.5e-7).
pub fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    // A&S constants
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
pub fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_approx(x * 0.7071067811865476))
}

// ============================================================
// Forward: causal conv + SEANet
// ============================================================

/// Causal Conv1d forward on channel-first input.
/// Output [out_ch, out_len], out_len = (padded - eff) / stride + 1.
pub fn mimi_conv_forward(input: &[f32], conv: &MimiConv) -> Vec<f32> {
    let in_ch = conv.in_ch;
    debug_assert_eq!(input.len() % in_ch, 0);
    let t_len = input.len() / in_ch;
    let eff = eff_kernel_size(conv.k, conv.dilation);
    let pt = padding_total(conv.k, conv.stride, conv.dilation);
    let extra = extra_padding(t_len, eff, pt, conv.stride);
    let padded_len = t_len + pt + extra;
    let mut padded = vec![0.0f32; in_ch * padded_len];
    for c in 0..in_ch {
        let ch = causal_pad_channel(&input[c * t_len..(c + 1) * t_len], pt, extra, conv.pad_replicate);
        padded[c * padded_len..(c + 1) * padded_len].copy_from_slice(&ch);
    }
    let out_len = (padded_len - eff) / conv.stride + 1;
    let mut out = vec![0.0f32; conv.out_ch * out_len];
    for oc in 0..conv.out_ch {
        for t in 0..out_len {
            let mut sum = conv.bias.as_ref().map(|b| b[oc]).unwrap_or(0.0);
            let base = t * conv.stride;
            for ic in 0..in_ch {
                for k in 0..conv.k {
                    let w = conv.weight[(oc * in_ch + ic) * conv.k + k];
                    sum += padded[ic * padded_len + base + k * conv.dilation] * w;
                }
            }
            out[oc * out_len + t] = sum;
        }
    }
    out
}

/// SEANet resnet block: ELU→Conv→ELU→Conv→(+identity). Length-preserving (s=1).
pub fn seanet_resblock_forward(input: &[f32], channels: usize, block: &SeanetResBlock) -> Vec<f32> {
    let t_len = input.len() / channels;
    let mut x: Vec<f32> = input.iter().map(|&v| elu(v)).collect();
    x = mimi_conv_forward(&x, &block.convs[0]);
    for v in &mut x {
        *v = elu(*v);
    }
    x = mimi_conv_forward(&x, &block.convs[1]);
    // residual (lengths must match; both s=1 causal convs preserve length here)
    let x_len = x.len() / channels;
    let use_len = x_len.min(t_len);
    let mut out = vec![0.0f32; channels * use_len];
    for c in 0..channels {
        for t in 0..use_len {
            out[c * use_len + t] = input[c * t_len + t] + x[c * x_len + t];
        }
    }
    out
}

/// Full SEANet encoder: waveform [T] → [512, T/960].
pub fn seanet_forward(enc: &SeanetEncoder, wave: &[f32]) -> Vec<f32> {
    let mut x = mimi_conv_forward(wave, &enc.initial);
    let mut channels = 64usize;
    for stage in enc.stages.iter() {
        x = seanet_resblock_forward(&x, channels, &stage.res);
        for v in &mut x {
            *v = elu(*v);
        }
        let t_before = x.len() / channels;
        let pt = padding_total(stage.down.k, stage.down.stride, stage.down.dilation);
        let eff = eff_kernel_size(stage.down.k, stage.down.dilation);
        let extra = extra_padding(t_before, eff, pt, stage.down.stride);
        let expect = (t_before + pt + extra - eff) / stage.down.stride + 1;
        x = mimi_conv_forward(&x, &stage.down);
        channels *= 2;
        debug_assert_eq!(x.len() / channels, expect);
    }
    for v in &mut x {
        *v = elu(*v);
    }
    mimi_conv_forward(&x, &enc.final_conv)
}

// ============================================================
// Loading (V2 nested key layout, all F32)
// ============================================================

fn read_f32(st: &SafeTensors, key: &str) -> Result<Vec<f32>, String> {
    let t = st.tensor(key).map_err(|_| format!("missing tensor: {key}"))?;
    let shape: Vec<usize> = t.shape().to_vec();
    let data = t.data();
    let mut out = Vec::with_capacity(data.len() / 4);
    for i in 0..data.len() / 4 {
        out.push(f32::from_le_bytes([data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]]));
    }
    let expect: usize = shape.iter().product();
    if out.len() != expect {
        return Err(format!("shape mismatch for {key}: {:?} vs {} elems", shape, out.len()));
    }
    Ok(out)
}

fn read_f32_opt(st: &SafeTensors, key: &str) -> Result<Option<Vec<f32>>, String> {
    match st.tensor(key) {
        Ok(_) => Ok(Some(read_f32(st, key)?)),
        Err(_) => Ok(None),
    }
}

fn load_mimi_conv(
    st: &SafeTensors,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    pad_replicate: bool,
) -> Result<MimiConv, String> {
    let w = read_f32(st, &format!("{prefix}.weight"))?;
    if w.len() != out_ch * in_ch * k {
        return Err(format!("bad shape for {prefix}.weight: {} elems", w.len()));
    }
    let b = read_f32_opt(st, &format!("{prefix}.bias"))?;
    Ok(MimiConv { weight: w, bias: b, in_ch, out_ch, k, stride, dilation, pad_replicate })
}

/// Load codec encoder weights from `speech_tokenizer/model.safetensors`.
/// Keys carry the V2 `encoder.` prefix (e.g. `encoder.encoder.layers.0.conv.weight`).
pub fn load_codec_encoder(st_path: &std::path::Path) -> Result<CodecEncoderWeights, String> {
    let data = std::fs::read(st_path).map_err(|e| format!("read {st_path:?}: {e}"))?;
    let st = SafeTensors::deserialize(&data).map_err(|e| format!("safetensors: {e}"))?;
    load_codec_encoder_from_tensors(&st)
}

fn load_codec_encoder_from_tensors(st: &SafeTensors) -> Result<CodecEncoderWeights, String> {
    const E: &str = "encoder.encoder";
    const T: &str = "encoder.encoder_transformer";
    const Q: &str = "encoder.quantizer";

    // SEANet: layers.0 initial [64,1,7] s=1; then per stage s in 0..4 (ELUs take
    // weightless slots in between):
    //   layers.(1+3s) = resnet block (block.1: k=3, block.3: k=1),
    //   layers.(3+3s) = downsample conv (k=2r, s=r); layers.14 = final [512,512,3].
    // Ratios (reversed upsampling_ratios [8,6,5,4]) = [4,5,6,8], num_residual_layers=1.
    let initial = load_mimi_conv(st, &format!("{E}.layers.0.conv"), 1, 64, 7, 1, 1, false)?;
    let ratios = [4usize, 5, 6, 8];
    let mut channels = 64usize;
    let mut stages = Vec::with_capacity(4);
    for (s, &r) in ratios.iter().enumerate() {
        let rb = 1 + 3 * s;
        let b0 = load_mimi_conv(st, &format!("{E}.layers.{rb}.block.1.conv"), channels, channels / 2, 3, 1, 1, false)?;
        let b1 = load_mimi_conv(st, &format!("{E}.layers.{rb}.block.3.conv"), channels / 2, channels, 1, 1, 1, false)?;
        let down = load_mimi_conv(st, &format!("{E}.layers.{}.conv", rb + 2), channels, channels * 2, 2 * r, r, 1, false)?;
        stages.push(DownStage { res: SeanetResBlock { convs: [b0, b1] }, down });
        channels *= 2;
    }
    // final conv [512,1024,3] s=1 (last_kernel_size=3, 1024→512)
    let final_conv = load_mimi_conv(st, &format!("{E}.layers.14.conv"), 1024, 512, 3, 1, 1, false)?;

    // Transformer x8 (hidden 512, no qkv/o bias, LayerNorm eps 1e-5, GELU MLP 2048)
    let mut transformer = Vec::with_capacity(8);
    for i in 0..8 {
        let p = format!("{T}.layers.{i}");
        let rd = |s: &str| read_f32(st, s);
        let layer = MimiTransformerLayerW {
            q_proj: rd(&format!("{p}.self_attn.q_proj.weight"))?,
            k_proj: rd(&format!("{p}.self_attn.k_proj.weight"))?,
            v_proj: rd(&format!("{p}.self_attn.v_proj.weight"))?,
            o_proj: rd(&format!("{p}.self_attn.o_proj.weight"))?,
            in_ln_w: rd(&format!("{p}.input_layernorm.weight"))?,
            in_ln_b: rd(&format!("{p}.input_layernorm.bias"))?,
            post_ln_w: rd(&format!("{p}.post_attention_layernorm.weight"))?,
            post_ln_b: rd(&format!("{p}.post_attention_layernorm.bias"))?,
            fc1: rd(&format!("{p}.mlp.fc1.weight"))?,
            fc2: rd(&format!("{p}.mlp.fc2.weight"))?,
            attn_scale: rd(&format!("{p}.self_attn_layer_scale.scale"))?,
            mlp_scale: rd(&format!("{p}.mlp_layer_scale.scale"))?,
        };
        for w in [&layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.o_proj] {
            if w.len() != 512 * 512 {
                return Err(format!("bad attn proj shape at {p}: {} elems", w.len()));
            }
        }
        for w in [&layer.in_ln_w, &layer.in_ln_b, &layer.post_ln_w, &layer.post_ln_b,
                  &layer.attn_scale, &layer.mlp_scale] {
            if w.len() != 512 {
                return Err(format!("bad norm/scale shape at {p}: {} elems", w.len()));
            }
        }
        if layer.fc1.len() != 2048 * 512 || layer.fc2.len() != 512 * 2048 {
            return Err(format!("bad mlp shape at {p}"));
        }
        transformer.push(layer);
    }

    // Downsample [512,512,4] s=2, replicate pad, no bias (frame 25Hz → 12.5Hz)
    let downsample = load_mimi_conv(st, "encoder.downsample.conv", 512, 512, 4, 2, 1, true)?;

    // Split RVQ sides
    let load_side = |name: &str, n_books: usize| -> Result<RvqSide, String> {
        let p = format!("{Q}.{name}");
        let flat = |s: &str| -> Result<Vec<f32>, String> {
            let v = read_f32(st, s)?;
            Ok(v)
        };
        // input_proj Conv1d [256,512,1] → flatten k dim
        let input_proj = flat(&format!("{p}.input_proj.weight"))?;
        let output_proj = flat(&format!("{p}.output_proj.weight"))?;
        let mut codebooks = Vec::with_capacity(n_books);
        for b in 0..n_books {
            let sum = read_f32(st, &format!("{p}.layers.{b}.codebook.embed_sum"))?;
            let usage = read_f32(st, &format!("{p}.layers.{b}.codebook.cluster_usage"))?;
            if sum.len() != 2048 * 256 || usage.len() != 2048 {
                return Err(format!("bad codebook shape at {p}.layers.{b}"));
            }
            let mut emb = vec![0.0f32; 2048 * 256];
            for i in 0..2048 {
                let u = usage[i].max(1e-5); // matches epsilon guard
                for d in 0..256 {
                    emb[i * 256 + d] = sum[i * 256 + d] / u;
                }
            }
            codebooks.push(emb);
        }
        Ok(RvqSide { input_proj, output_proj, codebooks })
    };
    let semantic = load_side("semantic_residual_vector_quantizer", 1)?;
    let acoustic = load_side("acoustic_residual_vector_quantizer", 31)?;

    Ok(CodecEncoderWeights { seanet: SeanetEncoder { initial, stages, final_conv }, transformer, downsample, semantic, acoustic })
}

// ============================================================
// Forward: transformer (RoPE + sliding-window causal attn)
// ============================================================

/// Per-row LayerNorm.
pub fn layer_norm_row(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len();
    let mean = x.iter().sum::<f32>() / d as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter().enumerate().map(|(i, v)| (v - mean) * inv * w[i] + b[i]).collect()
}

pub const MIMI_ROPE_DIM: usize = 64;
pub const MIMI_ROPE_THETA: f32 = 10000.0;
pub const MIMI_SLIDING_WINDOW: usize = 250;

/// inv_freq[i] = 1 / (theta ^ (2i / 64)), i = 0..32.
pub fn mimi_inv_freq() -> Vec<f32> {
    (0..32).map(|i| 1.0 / MIMI_ROPE_THETA.powf((2 * i) as f32 / MIMI_ROPE_DIM as f32)).collect()
}

/// Standard llama-style RoPE on one 64-dim head vector (rotate_half convention).
pub fn apply_rope_64(v: &mut [f32], pos: usize, inv_freq: &[f32]) {
    debug_assert_eq!(v.len(), 64);
    for i in 0..32 {
        let a = pos as f32 * inv_freq[i];
        let (c, s) = (a.cos(), a.sin());
        let x0 = v[i];
        let x1 = v[i + 32];
        v[i] = x0 * c - x1 * s;
        v[i + 32] = x1 * c + x0 * s;
    }
}

/// Sliding-window causal mask predicate: key k visible from query p iff
/// k <= p and p - k < 250.
pub fn mimi_attn_allowed(q_pos: usize, k_pos: usize) -> bool {
    k_pos <= q_pos && q_pos - k_pos < MIMI_SLIDING_WINDOW
}

/// y = W x with W row-major [out_dim, in_dim].
pub fn matvec(w: &[f32], out_dim: usize, x: &[f32]) -> Vec<f32> {
    let in_dim = x.len();
    debug_assert_eq!(w.len(), out_dim * in_dim);
    let mut y = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut s = 0.0f32;
        for (i, &xv) in x.iter().enumerate() {
            s += w[o * in_dim + i] * xv;
        }
        y[o] = s;
    }
    y
}

/// One transformer layer on [T, 512] rows (8 heads x 64, no GQA, fp32 softmax).
pub fn mimi_transformer_layer_forward(
    hidden: &[f32],
    t_len: usize,
    w: &MimiTransformerLayerW,
    inv_freq: &[f32],
) -> Vec<f32> {
    const D: usize = 512;
    const H: usize = 8;
    const HD: usize = 64;
    debug_assert_eq!(hidden.len(), t_len * D);

    // Q/K/V with RoPE on Q,K
    let mut q: Vec<f32> = Vec::with_capacity(t_len * D);
    let mut k: Vec<f32> = Vec::with_capacity(t_len * D);
    let mut v: Vec<f32> = Vec::with_capacity(t_len * D);
    for t in 0..t_len {
        let row = &hidden[t * D..(t + 1) * D];
        let n = layer_norm_row(row, &w.in_ln_w, &w.in_ln_b, 1e-5);
        let mut qt = matvec(&w.q_proj, D, &n);
        let mut kt = matvec(&w.k_proj, D, &n);
        for h in 0..H {
            apply_rope_64(&mut qt[h * HD..(h + 1) * HD], t, inv_freq);
            apply_rope_64(&mut kt[h * HD..(h + 1) * HD], t, inv_freq);
        }
        let vt = matvec(&w.v_proj, D, &n);
        q.extend_from_slice(&qt);
        k.extend_from_slice(&kt);
        v.extend_from_slice(&vt);
    }

    // Sliding-window causal attention
    let scale = 1.0 / (HD as f32).sqrt();
    let mut attn_out = vec![0.0f32; t_len * D];
    for h in 0..H {
        for p in 0..t_len {
            let mut max_s = f32::NEG_INFINITY;
            let mut scores = vec![f32::NEG_INFINITY; t_len];
            for kk in 0..t_len {
                if !mimi_attn_allowed(p, kk) {
                    continue;
                }
                let mut s = 0.0f32;
                for d in 0..HD {
                    s += q[(p * H + h) * HD + d] * k[(kk * H + h) * HD + d];
                }
                s *= scale;
                scores[kk] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut denom = 0.0f32;
            for kk in 0..t_len {
                if scores[kk].is_finite() {
                    scores[kk] = (scores[kk] - max_s).exp();
                    denom += scores[kk];
                } else {
                    scores[kk] = 0.0;
                }
            }
            for d in 0..HD {
                let mut acc = 0.0f32;
                for kk in 0..t_len {
                    acc += scores[kk] / denom * v[(kk * H + h) * HD + d];
                }
                attn_out[(p * H + h) * HD + d] = acc;
            }
        }
    }
    // O proj + LayerScale residual
    let mut y = vec![0.0f32; t_len * D];
    for t in 0..t_len {
        let o = matvec(&w.o_proj, D, &attn_out[t * D..(t + 1) * D]);
        for d in 0..D {
            y[t * D + d] = hidden[t * D + d] + w.attn_scale[d] * o[d];
        }
    }
    // MLP branch
    let mut out = vec![0.0f32; t_len * D];
    for t in 0..t_len {
        let row = &y[t * D..(t + 1) * D];
        let n = layer_norm_row(row, &w.post_ln_w, &w.post_ln_b, 1e-5);
        let h1 = matvec(&w.fc1, 2048, &n);
        let h1a: Vec<f32> = h1.iter().map(|&v| gelu_exact(v)).collect();
        let h2 = matvec(&w.fc2, D, &h1a);
        for d in 0..D {
            out[t * D + d] = row[d] + w.mlp_scale[d] * h2[d];
        }
    }
    out
}

/// Full 8-layer transformer on rows [T, 512].
pub fn mimi_transformer_forward(layers: &[MimiTransformerLayerW], hidden: &[f32], t_len: usize) -> Vec<f32> {
    let inv = mimi_inv_freq();
    let mut x = hidden.to_vec();
    for w in layers {
        x = mimi_transformer_layer_forward(&x, t_len, w, &inv);
    }
    x
}

// ============================================================
// Forward: 1x1 proj + split RVQ + top-level encode
// ============================================================

/// 1x1 conv (no bias) on channel-first input: y[o,t] = sum_i W[o,i] x[i,t].
pub fn conv1x1_forward(x: &[f32], in_ch: usize, w: &[f32], out_ch: usize) -> Vec<f32> {
    let t_len = x.len() / in_ch;
    debug_assert_eq!(w.len(), out_ch * in_ch);
    let mut y = vec![0.0f32; out_ch * t_len];
    for o in 0..out_ch {
        for t in 0..t_len {
            let mut s = 0.0f32;
            for i in 0..in_ch {
                s += w[o * in_ch + i] * x[i * t_len + t];
            }
            y[o * t_len + t] = s;
        }
    }
    y
}

/// Nearest codebook index (fp32 squared Euclidean, first-wins ties like argmin).
pub fn rvq_argmin(frame: &[f32], codebook: &[f32], dim: usize) -> u32 {
    let n = codebook.len() / dim;
    let mut best = 0u32;
    let mut best_d = f32::INFINITY;
    for i in 0..n {
        let mut d = 0.0f32;
        for dd in 0..dim {
            let diff = frame[dd] - codebook[i * dim + dd];
            d += diff * diff;
        }
        if d < best_d {
            best_d = d;
            best = i as u32;
        }
    }
    best
}

/// Encode one RVQ side: project 512→256, then `n_layers` residual steps.
/// Returns [n_layers][T] codes.
pub fn rvq_side_encode(side: &RvqSide, x512: &[f32], t_len: usize, n_layers: usize) -> Vec<Vec<u32>> {
    assert!(n_layers <= side.codebooks.len());
    let mut residual = conv1x1_forward(x512, 512, &side.input_proj, 256);
    let mut codes = vec![Vec::with_capacity(t_len); n_layers];
    for l in 0..n_layers {
        let cb = &side.codebooks[l];
        for t in 0..t_len {
            let mut frame = vec![0.0f32; 256];
            for d in 0..256 {
                frame[d] = residual[d * t_len + t];
            }
            let idx = rvq_argmin(&frame, cb, 256) as usize;
            codes[l].push(idx as u32);
            for d in 0..256 {
                residual[d * t_len + t] -= cb[idx * 256 + d];
            }
        }
    }
    codes
}

/// Valid code length for raw audio of `audio_len` samples: ceil(len / 1920).
/// Mirrors V2 `code[..., :-(-mask.sum() // encode_downsample_rate)]`.
pub fn codec_valid_len(audio_len: usize) -> usize {
    audio_len.div_ceil(1920)
}

/// Top-level encode: waveform [L] → 16 codebooks x Tv codes (valid-16 of 32:
/// 1 semantic + first 15 acoustic), trimmed to `codec_valid_len(L)`.
pub fn encode_waveform_to_codes(w: &CodecEncoderWeights, wave: &[f32]) -> Vec<Vec<u32>> {
    // SEANet: [T] → [512, T/960]
    let s = seanet_forward(&w.seanet, wave);
    let t25 = s.len() / 512;
    // transformer on rows [T, 512]
    let mut rows = vec![0.0f32; s.len()];
    for t in 0..t25 {
        for c in 0..512 {
            rows[t * 512 + c] = s[c * t25 + t];
        }
    }
    let rows = mimi_transformer_forward(&w.transformer, &rows, t25);
    let mut back = vec![0.0f32; rows.len()];
    for t in 0..t25 {
        for c in 0..512 {
            back[c * t25 + t] = rows[t * 512 + c];
        }
    }
    // downsample: [512, T/960] → [512, T/1920]
    let d = mimi_conv_forward(&back, &w.downsample);
    let t12 = d.len() / 512;
    // split RVQ (acoustic re-encodes the same embeddings, not the residual)
    let mut codes = rvq_side_encode(&w.semantic, &d, t12, 1);
    codes.extend(rvq_side_encode(&w.acoustic, &d, t12, 15));
    debug_assert_eq!(codes.len(), 16);
    // trim to valid length
    let tv = codec_valid_len(wave.len()).min(t12);
    for q in codes.iter_mut() {
        q.truncate(tv);
    }
    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eff_kernel_and_padding_total() {
        // k=7,s=1,d=1 → eff 7, pt 6
        assert_eq!(eff_kernel_size(7, 1), 7);
        assert_eq!(padding_total(7, 1, 1), 6);
        // downsample convs: k=2r, s=r → pt = r
        for r in [4, 5, 6, 8] {
            assert_eq!(padding_total(2 * r, r, 1), r);
        }
        // final downsample k=4 s=2 → pt 2
        assert_eq!(padding_total(4, 2, 1), 2);
        // resnet k=3 d=1 s=1 → pt 2; k=1 → pt 0
        assert_eq!(padding_total(3, 1, 1), 2);
        assert_eq!(padding_total(1, 1, 1), 0);
    }

    #[test]
    fn test_extra_padding_grid() {
        // Hand-computed against MimiConv1d._get_extra_padding_for_conv1d
        // (n = ceil((L-eff+pt)/s + 1) - 1; extra = n*s + eff - pt - L):
        // L=24000, eff=7, pt=6, s=1 → n=23999 → ideal=24000 → 0
        assert_eq!(extra_padding(24000, 7, 6, 1), 0);
        // L=10, eff=8, pt=4, s=4 → y=1.5 → n=2 → ideal=12 → 2
        assert_eq!(extra_padding(10, 8, 4, 4), 2);
        // L=11 → y=1.75 → n=2 → ideal=12 → 1
        assert_eq!(extra_padding(11, 8, 4, 4), 1);
        // L=13 → y=2.25 → n=3 → ideal=16 → 3
        assert_eq!(extra_padding(13, 8, 4, 4), 3);
        // L=14 → y=2.5 → n=3 → ideal=16 → 2
        assert_eq!(extra_padding(14, 8, 4, 4), 2);
        // L=9 → y=1.25 → n=2 → ideal=12 → 3
        assert_eq!(extra_padding(9, 8, 4, 4), 3);
        // L=7 → y=0.75 → n=1 → ideal=8 → 1
        assert_eq!(extra_padding(7, 8, 4, 4), 1);
        // s=1 always 0
        assert_eq!(extra_padding(24001, 7, 6, 1), 0);
        // L=101, eff=4, pt=2, s=2 → y=49.5 → n=50 → ideal=102 → 1
        assert_eq!(extra_padding(101, 4, 2, 2), 1);
        // L=100 → y=49 → n=49 → ideal=100 → 0
        assert_eq!(extra_padding(100, 4, 2, 2), 0);
    }

    #[test]
    fn test_causal_pad_channel() {
        let s = vec![1.0, 2.0, 3.0];
        // zero mode
        assert_eq!(causal_pad_channel(&s, 2, 1, false), vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
        // replicate mode
        assert_eq!(causal_pad_channel(&s, 2, 1, true), vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0]);
        // empty edge
        assert_eq!(causal_pad_channel(&[], 2, 1, true), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_elu() {
        assert_eq!(elu(2.0), 2.0);
        assert_eq!(elu(0.0), 0.0);
        assert!((elu(-1.0) - (1.0f32.exp().recip() - 1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_seanet_length_chain() {
        // Geometry-only check with zero weights (lengths don't depend on
        // channels/values): 44160 samples must flow
        // 44160 → 44160 → 11040 → 2208 → 368 → 46 (seanet) → 23 (downsample).
        fn dummy(k: usize, s: usize) -> MimiConv {
            MimiConv {
                weight: vec![0.0; k], bias: None,
                in_ch: 1, out_ch: 1, k, stride: s, dilation: 1, pad_replicate: false,
            }
        }
        let mut x = vec![0.0f32; 44160];
        let steps = [(7, 1, 44160), (8, 4, 11040), (10, 5, 2208), (12, 6, 368), (16, 8, 46), (3, 1, 46), (4, 2, 23)];
        for (k, s, expect) in steps {
            x = mimi_conv_forward(&x, &dummy(k, s));
            assert_eq!(x.len(), expect, "k={k} s={s}");
        }
    }

    #[test]
    fn test_mimi_conv_ones() {
        // in=1, k=3, s=1, all-ones weight, no bias, x=[1,2,3,4]:
        // pad_left=2 zeros → [0,0,1,2,3,4] → [1,3,6,9]
        let conv = MimiConv {
            weight: vec![1.0; 3], bias: None,
            in_ch: 1, out_ch: 1, k: 3, stride: 1, dilation: 1, pad_replicate: false,
        };
        assert_eq!(mimi_conv_forward(&[1.0, 2.0, 3.0, 4.0], &conv), vec![1.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_mimi_conv_strided_len_and_bias() {
        // k=8 s=4, L=10 → padded 16 → out 3; weight ones, bias 0.5
        let conv = MimiConv {
            weight: vec![1.0; 8], bias: Some(vec![0.5]),
            in_ch: 1, out_ch: 1, k: 8, stride: 4, dilation: 1, pad_replicate: false,
        };
        let x: Vec<f32> = (1..=10).map(|v| v as f32).collect();
        // padded [0,0,0,0,1..10,0,0] (extra=2); windows at 0,4,8:
        // w0: 0+0+0+0+1+2+3+4=10; w1: 1+2+3+4+5+6+7+8=36; w2: 5+6+7+8+9+10+0+0=45
        assert_eq!(mimi_conv_forward(&x, &conv), vec![10.5, 36.5, 45.5]);
    }

    #[test]
    fn test_resblock_zero_weights_is_identity() {
        // Zero convs → branch output zeros → out = input.
        let z3 = MimiConv {
            weight: vec![0.0; 2 * 1 * 3], bias: Some(vec![0.0; 2]),
            in_ch: 2, out_ch: 2, k: 1, stride: 1, dilation: 1, pad_replicate: false,
        };
        // NOTE: convs[0] maps ch→ch/2; build matching tiny block with channels=2:
        // use k=1 so padding is trivially zero.
        let c0 = MimiConv { weight: vec![0.0; 1 * 2 * 1], bias: None, in_ch: 2, out_ch: 1, k: 1, stride: 1, dilation: 1, pad_replicate: false };
        let c1 = MimiConv { weight: vec![0.0; 2 * 1 * 1], bias: None, in_ch: 1, out_ch: 2, k: 1, stride: 1, dilation: 1, pad_replicate: false };
        let _ = z3;
        let block = SeanetResBlock { convs: [c0, c1] };
        let x = vec![0.5, -1.0, 2.0, -0.25, 1.5, 3.0]; // [2, 3]
        assert_eq!(seanet_resblock_forward(&x, 2, &block), x);
    }

    #[test]
    fn test_layer_norm_row() {
        // x=[1,2,3], w=1, b=0, eps=0 → mean 2, var 2/3
        let y = layer_norm_row(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], 0.0);
        let inv = (2.0f32 / 3.0).sqrt().recip();
        assert!((y[0] - -1.0 * inv).abs() < 1e-6);
        assert!(y[1].abs() < 1e-6);
        assert!((y[2] - inv).abs() < 1e-6);
    }

    #[test]
    fn test_erf_and_gelu_exact() {
        assert!(erf_approx(0.0).abs() < 1e-7);
        assert!((erf_approx(1.0) - 0.84270079).abs() < 1e-6);
        assert!((erf_approx(0.5) - 0.52049988).abs() < 1e-6);
        assert!((erf_approx(-1.0) + 0.84270079).abs() < 1e-6);
        assert_eq!(gelu_exact(0.0), 0.0);
        assert!((gelu_exact(1.0) - 0.84134475).abs() < 1e-5);
        assert!((gelu_exact(-1.0) + 0.15865525).abs() < 1e-5);
        // tanh-approx differs from exact (guard against accidental swap to approx)
        assert!((gelu_exact(1.0) - crate::conv::gelu(1.0)).abs() > 1e-5);
    }

    #[test]
    fn test_rope_pos0_identity_and_rotation() {
        let inv = mimi_inv_freq();
        assert_eq!(inv.len(), 32);
        assert!((inv[0] - 1.0).abs() < 1e-7); // theta^0 = 1
        // pos 0 → identity
        let mut v: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let orig = v.clone();
        apply_rope_64(&mut v, 0, &inv);
        for (a, b) in v.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        // x = e0 at pos p → out[0]=cos p, out[32]=sin p (inv[0]=1)
        let p = 2usize;
        let mut w = vec![0.0f32; 64];
        w[0] = 1.0;
        apply_rope_64(&mut w, p, &inv);
        assert!((w[0] - (p as f32).cos()).abs() < 1e-6);
        assert!((w[32] - (p as f32).sin()).abs() < 1e-6);
    }

    #[test]
    fn test_attn_window_predicate() {
        assert!(mimi_attn_allowed(0, 0));
        assert!(mimi_attn_allowed(5, 5));
        assert!(mimi_attn_allowed(5, 0));
        assert!(!mimi_attn_allowed(3, 4)); // future
        assert!(!mimi_attn_allowed(300, 50)); // 250 back: excluded
        assert!(mimi_attn_allowed(300, 51)); // 249 back: included
        assert!(mimi_attn_allowed(249, 0));
        assert!(!mimi_attn_allowed(250, 0));
    }

    #[test]
    fn test_rvq_argmin_first_wins() {
        let cb = vec![0.0, 0.0, 10.0, 10.0]; // 2 entries, dim 2
        assert_eq!(rvq_argmin(&[1.0, 1.0], &cb, 2), 0);
        assert_eq!(rvq_argmin(&[9.0, 9.0], &cb, 2), 1);
        // tie [5,5] → first index (torch argmin convention)
        assert_eq!(rvq_argmin(&[5.0, 5.0], &cb, 2), 0);
    }

    #[test]
    fn test_codec_valid_len() {
        assert_eq!(codec_valid_len(1920), 1);
        assert_eq!(codec_valid_len(1921), 2);
        assert_eq!(codec_valid_len(44160), 23); // 1.84s ref
        assert_eq!(codec_valid_len(111360), 58); // 4.64s ref
    }

    /// End-to-end encode on a real wav with real weights (A4 gate step 1).
    /// Ignored by default:
    /// `QORA_ST_WEIGHTS=/tmp/opencode/st_model.safetensors cargo test --lib codec_encoder -- --ignored`
    #[test]
    #[ignore]
    fn test_encode_real_audio() {
        let path = std::env::var("QORA_ST_WEIGHTS").expect("set QORA_ST_WEIGHTS");
        let w = load_codec_encoder(std::path::Path::new(&path)).expect("load");
        let (audio, sr) = crate::wav::read_wav(std::path::Path::new("voice/zstym2.wav")).expect("wav");
        assert_eq!(sr, 24000);
        let codes = encode_waveform_to_codes(&w, &audio);
        assert_eq!(codes.len(), 16);
        let tv = codec_valid_len(audio.len());
        assert_eq!(tv, 23);
        for (q, c) in codes.iter().enumerate() {
            assert_eq!(c.len(), tv, "codebook {q}");
            assert!(c.iter().all(|&v| v < 2048), "codebook {q} range");
        }
        // sanity: not collapsed to a single value
        for (q, c) in codes.iter().enumerate().take(4) {
            let uniq: std::collections::HashSet<u32> = c.iter().copied().collect();
            eprintln!("codebook {q}: len={} uniq={} first5={:?}", c.len(), uniq.len(), &c[..5.min(c.len())]);
            assert!(uniq.len() > 1, "codebook {q} collapsed");
        }
    }

    /// Stage dump for differential test vs torch ground truth (gt_encode.py).
    /// Writes /tmp/opencode/rs_{seanet,transformer,downsample,codes}.txt
    /// Ignored by default (needs weights + wav).
    #[test]
    #[ignore]
    fn test_dump_stages() {
        use std::io::Write;
        let path = std::env::var("QORA_ST_WEIGHTS").expect("set QORA_ST_WEIGHTS");
        let w = load_codec_encoder(std::path::Path::new(&path)).expect("load");
        let (audio, _) =
            crate::wav::read_wav(std::path::Path::new("voice/zstym2.wav")).expect("wav");
        let s = seanet_forward(&w.seanet, &audio);
        let t25 = s.len() / 512;
        let mut rows = vec![0.0f32; s.len()];
        for t in 0..t25 {
            for c in 0..512 {
                rows[t * 512 + c] = s[c * t25 + t];
            }
        }
        let tro = mimi_transformer_forward(&w.transformer, &rows, t25);
        let mut back = vec![0.0f32; tro.len()];
        for t in 0..t25 {
            for c in 0..512 {
                back[c * t25 + t] = tro[t * 512 + c];
            }
        }
        let d = mimi_conv_forward(&back, &w.downsample);
        let dump = |name: &str, v: &[f32]| {
            let mut f = std::fs::File::create(format!("/tmp/opencode/rs_{name}.txt")).unwrap();
            for x in v {
                write!(f, "{x:.6} ").unwrap();
            }
        };
        dump("seanet", &s);
        dump("transformer", &tro);
        dump("downsample", &d);
        let codes = encode_waveform_to_codes(&w, &audio);
        let mut f = std::fs::File::create("/tmp/opencode/rs_codes.txt").unwrap();
        for q in &codes {
            for c in q {
                write!(f, "{c} ").unwrap();
            }
            writeln!(f).unwrap();
        }
        eprintln!("t25={t25} t12={}", d.len() / 512);
    }

    /// Golden roundtrip (A4 gate step 2): ref_code → existing decoder → waveform,
    /// must resemble the original reference. Needs the full model binary:
    /// `QORA_ST_WEIGHTS=... QORA_MODEL=target/release/model.qora-tts cargo test --release --lib codec_encoder -- --ignored`
    #[test]
    #[ignore]
    fn test_golden_roundtrip() {
        let enc_path = std::env::var("QORA_ST_WEIGHTS").expect("set QORA_ST_WEIGHTS");
        let model_path = std::env::var("QORA_MODEL")
            .unwrap_or_else(|_| "target/release/model.qora-tts".to_string());
        let w = load_codec_encoder(std::path::Path::new(&enc_path)).expect("encoder load");
        let (audio, _) =
            crate::wav::read_wav(std::path::Path::new("voice/zstym2.wav")).expect("wav");
        let codes = encode_waveform_to_codes(&w, &audio);
        let (_talker, _predictor, decoder, _) =
            crate::save::load_model(std::path::Path::new(&model_path)).expect("model load");
        let out = crate::decoder::decode_to_audio(&decoder, &codes);
        let expect_len = codes[0].len() * 1920;
        assert_eq!(out.len(), expect_len, "decoded length");
        assert!(out.iter().all(|v| v.is_finite()), "finite output");
        let n = out.len().min(audio.len());
        let (a, b) = (&audio[..n], &out[..n]);
        let ma: f32 = a.iter().sum::<f32>() / n as f32;
        let mb: f32 = b.iter().sum::<f32>() / n as f32;
        let mut num = 0.0f32;
        let (mut da, mut db) = (0.0f32, 0.0f32);
        for i in 0..n {
            num += (a[i] - ma) * (b[i] - mb);
            da += (a[i] - ma) * (a[i] - ma);
            db += (b[i] - mb) * (b[i] - mb);
        }
        let corr = num / (da.sqrt() * db.sqrt() + 1e-9);
        let rms_in = (da / n as f32).sqrt();
        let rms_out = (db / n as f32).sqrt();
        eprintln!("roundtrip: n={n} corr={corr:.4} rms_in={rms_in:.4} rms_out={rms_out:.4}");
        crate::wav::write_wav(
            std::path::Path::new("/tmp/opencode/roundtrip.wav"),
            &out,
            24000,
        )
        .expect("write");
        assert!(corr > 0.7, "roundtrip correlation {corr:.4} (encoder likely wrong)");
        assert!(
            (rms_out / (rms_in + 1e-9) - 1.0).abs() < 0.5,
            "rms ratio out of range"
        );
    }

    /// Full weight load against the real `speech_tokenizer/model.safetensors`.
    /// Ignored by default (needs 651MB file): 
    /// `QORA_ST_WEIGHTS=/tmp/opencode/st_model.safetensors cargo test --lib codec_encoder -- --ignored`
    #[test]
    #[ignore]
    fn test_load_real_weights() {
        let path = std::env::var("QORA_ST_WEIGHTS").expect("set QORA_ST_WEIGHTS");
        let w = load_codec_encoder(std::path::Path::new(&path)).expect("load");
        assert_eq!(w.seanet.stages.len(), 4);
        assert_eq!(w.transformer.len(), 8);
        assert_eq!(w.semantic.codebooks.len(), 1);
        assert_eq!(w.acoustic.codebooks.len(), 31);
        assert_eq!(w.downsample.weight.len(), 512 * 512 * 4);
        assert!(w.downsample.bias.is_none());
        assert_eq!(w.semantic.input_proj.len(), 256 * 512);
        // codebooks are finite
        for cb in w.semantic.codebooks.iter().chain(w.acoustic.codebooks.iter().take(1)) {
            assert_eq!(cb.len(), 2048 * 256);
            assert!(cb.iter().all(|v| v.is_finite()));
        }
    }
}
