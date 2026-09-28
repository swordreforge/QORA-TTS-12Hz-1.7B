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
        transformer.push(MimiTransformerLayerW {
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
        });
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
