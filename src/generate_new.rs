//! New generation matching qwen3-tts-rs architecture
//!
//! Key differences from old version:
//! 1. Text projection uses SILU not GELU
//! 2. Dual-stream embeddings: text + codec ADDED together
//! 3. Proper TTS_PAD/TTS_BOS overlay on codec tokens
//! 4. Trailing text fusion during generation

use std::time::Instant;
use crate::talker::TalkerWeights;
use crate::code_predictor::CodePredictorWeights;
use crate::decoder::SpeechDecoderWeights;
use crate::tokenizer::TTSTokenizer;
use crate::gemv;

/// Simple deterministic PRNG (xorshift64).
fn simple_rand(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f32) / (u64::MAX as f32)
}

// Special token constants
const IM_START: u32 = 151644;
const ASSISTANT: u32 = 77091;
const NEWLINE: u32 = 198;
const TTS_PAD: u32 = 151671;
const TTS_BOS: u32 = 151672;
const TTS_EOS: u32 = 151673;
const CODEC_THINK: u32 = 2154;
const CODEC_THINK_BOS: u32 = 2156;
const CODEC_THINK_EOS: u32 = 2157;
const CODEC_PAD: u32 = 2148;
const CODEC_BOS: u32 = 2149;
const CODEC_EOS: u32 = 2150;

pub struct TTSParams {
    pub max_codes: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub codec_eos_id: u32,
    pub codec_bos_id: u32,
    /// Onset guard: first `onset_frames` frames of each chunk sample with
    /// `onset_temperature` (chunk attacks have the widest first-draw entropy
    /// and collect most roughness defects). 0 = off.
    pub onset_frames: usize,
    pub onset_temperature: f32,
    /// ICL decoder warmup: prepend ref codes so conv state and sliding-window
    /// attention start warm (official parity; fixes cold-start attacks).
    /// Costs decode time ∝ ref length. false = fast cold path (default).
    pub decode_warmup: bool,
    /// Cap for warmup context: use only the trailing N ref frames instead of
    /// the full reference (0 = full prepend per official). The decoder's only
    /// long-memory part is the 72-frame sliding attention window; Vocos conv
    /// state flushes within ~2 frames — so trailing-32 ≈ full at ~1/4 cost.
    /// Ignored unless decode_warmup is set.
    pub warmup_frames: usize,
}

impl Default for TTSParams {
    fn default() -> Self {
        Self {
            max_codes: 1000,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            codec_eos_id: CODEC_EOS,
            codec_bos_id: CODEC_BOS,
            onset_frames: 0,
            onset_temperature: 0.3,
            decode_warmup: false,
            warmup_frames: 0,
        }
    }
}

/// Effective sampling temperature for a frame (onset schedule).
pub fn frame_temperature(params: &TTSParams, frame_idx: usize) -> f32 {
    if frame_idx < params.onset_frames {
        params.onset_temperature
    } else {
        params.temperature
    }
}

/// Build role prefix: text_proj([IM_START, ASSISTANT, NEWLINE])
fn build_role_prefix(talker: &TalkerWeights) -> Vec<Vec<f32>> {
    vec![
        crate::talker::embed_text_token(talker, IM_START),
        crate::talker::embed_text_token(talker, ASSISTANT),
        crate::talker::embed_text_token(talker, NEWLINE),
    ]
}

/// Build TTS_PAD/TTS_BOS overlay: [TTS_PAD × count, TTS_BOS × 1]
fn build_tts_pad_bos(talker: &TalkerWeights, pad_count: usize) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(pad_count + 1);
    let tts_pad_proj = crate::talker::embed_text_token(talker, TTS_PAD);
    for _ in 0..pad_count {
        result.push(tts_pad_proj.clone());
    }
    result.push(crate::talker::embed_text_token(talker, TTS_BOS));
    result
}

/// Streaming-branch ICL block layout (official `generate_icl_prompt`).
/// text_len = R + T + 1 (ref tokens + text tokens + eos);
/// codec_len = Tv + 1 (codec_bos + ref frames).
/// text_len > codec_len: icl takes the first codec_len pairs, rest is remainder.
/// else: text side padded with tts_pad to codec_len, remainder empty.
pub struct IclLayout {
    pub icl_len: usize,
    pub remainder_len: usize,
}

pub fn icl_block_layout(ref_t: usize, text_t: usize, tv: usize) -> IclLayout {
    let text_len = ref_t + text_t + 1;
    let codec_len = tv + 1;
    if text_len > codec_len {
        IclLayout { icl_len: codec_len, remainder_len: text_len - codec_len }
    } else {
        IclLayout { icl_len: codec_len, remainder_len: 0 }
    }
}

fn add_embed(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Build the ICL block: (icl_embeds, remainder).
/// text side = text_proj(ref_tokens + text_tokens) + tts_eos;
/// codec side = codec_bos + per-frame sums over all 16 code groups
/// (group 0 via talker codec table, groups 1-15 via predictor tables).
/// Combined with the streaming branch rule (see `icl_block_layout`).
/// `ref_codes` is [16][Tv] from the codec encoder; values must be < 2048.
pub fn build_icl_block(
    talker: &TalkerWeights,
    predictor: &CodePredictorWeights,
    ref_tokens: &[u32],
    text_tokens: &[u32],
    ref_codes: &[Vec<u32>],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    assert_eq!(ref_codes.len(), 16, "ref_codes must have 16 codebooks");
    let tv = ref_codes[0].len();
    assert!(tv > 0, "ref_codes must be non-empty");
    assert!(ref_codes.iter().all(|c| c.len() == tv), "ragged ref_codes");

    let tts_pad = crate::talker::embed_text_token(talker, TTS_PAD);
    let tts_eos = crate::talker::embed_text_token(talker, TTS_EOS);

    let mut text_full = Vec::with_capacity(ref_tokens.len() + text_tokens.len() + 1);
    for &tok in ref_tokens.iter().chain(text_tokens.iter()) {
        text_full.push(crate::talker::embed_text_token(talker, tok));
    }
    text_full.push(tts_eos);

    let mut codec_full = Vec::with_capacity(tv + 1);
    codec_full.push(crate::talker::embed_codec_token(talker, CODEC_BOS));
    for t in 0..tv {
        let mut sum = crate::talker::embed_codec_token(talker, ref_codes[0][t]);
        for g in 1..16 {
            let e = crate::code_predictor::get_acoustic_embedding(predictor, g - 1, ref_codes[g][t]);
            for j in 0..talker.hidden_size {
                sum[j] += e[j];
            }
        }
        codec_full.push(sum);
    }

    let layout = icl_block_layout(ref_tokens.len(), text_tokens.len(), tv);
    let (icl, remainder) = if text_full.len() > codec_full.len() {
        let mut icl = Vec::with_capacity(codec_full.len());
        for (a, b) in text_full.iter().take(codec_full.len()).zip(codec_full.iter()) {
            icl.push(add_embed(a, b));
        }
        (icl, text_full[codec_full.len()..].to_vec())
    } else {
        let mut icl = Vec::with_capacity(codec_full.len());
        for (i, b) in codec_full.iter().enumerate() {
            if i < text_full.len() {
                icl.push(add_embed(&text_full[i], b));
            } else {
                icl.push(add_embed(&tts_pad, b));
            }
        }
        (icl, Vec::new())
    };
    debug_assert_eq!(icl.len(), layout.icl_len);
    debug_assert_eq!(remainder.len(), layout.remainder_len);
    (icl, remainder)
}

/// The 6 dual-stream codec header positions shared by every chunk
/// (voice-cloning path: 4 codec+PAD, voice+PAD at pos 4, PAD+BOS).
/// Chunk-independent: same voice + language → identical vectors.
fn build_codec_prefix_voice(talker: &TalkerWeights, language_id: u32, voice_embedding: &[f32]) -> Vec<Vec<f32>> {
    let hidden_size = talker.hidden_size;
    let mut out = Vec::with_capacity(6);
    let prefix_codec = [CODEC_THINK, CODEC_THINK_BOS, language_id, CODEC_THINK_EOS];
    let tts_pad_proj = crate::talker::embed_text_token(talker, TTS_PAD);
    let tts_bos_proj = crate::talker::embed_text_token(talker, TTS_BOS);
    for &tok in &prefix_codec {
        let codec_emb = crate::talker::embed_codec_token(talker, tok);
        let mut combined = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            combined[j] = codec_emb[j] + tts_pad_proj[j];
        }
        out.push(combined);
    }
    let mut pos4 = vec![0.0f32; hidden_size];
    for j in 0..hidden_size.min(voice_embedding.len()) {
        pos4[j] = voice_embedding[j] + tts_pad_proj[j];
    }
    out.push(pos4);
    let pad_emb = crate::talker::embed_codec_token(talker, CODEC_PAD);
    let mut pad_combined = vec![0.0f32; hidden_size];
    for j in 0..hidden_size {
        pad_combined[j] = pad_emb[j] + tts_bos_proj[j];
    }
    out.push(pad_combined);
    out
}

/// Built-in speaker variant of the 6 header positions (no voice embedding).
fn build_codec_prefix_builtin(talker: &TalkerWeights, speaker_id: u32, language_id: u32) -> Vec<Vec<f32>> {
    let hidden_size = talker.hidden_size;
    let codec_tokens = [CODEC_THINK, CODEC_THINK_BOS, language_id, CODEC_THINK_EOS, speaker_id, CODEC_PAD];
    let mut codec_embeds: Vec<Vec<f32>> = codec_tokens.iter()
        .map(|&t| crate::talker::embed_codec_token(talker, t))
        .collect();
    let tts_overlay = build_tts_pad_bos(talker, 5);
    for i in 0..6 {
        for j in 0..hidden_size {
            codec_embeds[i][j] += tts_overlay[i][j];
        }
    }
    codec_embeds
}

/// Chunk-independent head of the prefill sequence: role (3) + codec header (6).
fn build_prefill_head(
    talker: &TalkerWeights,
    speaker_id: u32,
    language_id: u32,
    voice_embedding: Option<&[f32]>,
) -> Vec<Vec<f32>> {
    let mut head = build_role_prefix(talker);
    if let Some(emb) = voice_embedding {
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("  Voice embedding norm={:.4}, injected with TTS_PAD overlay at position 4", norm);
        head.extend(build_codec_prefix_voice(talker, language_id, emb));
    } else {
        head.extend(build_codec_prefix_builtin(talker, speaker_id, language_id));
    }
    head
}

/// Chunk-independent prefix of the ICL block: the first R pairs, where only
/// ref_tokens meet the codec side (text_full[0..R] + codec_full[0..R],
/// codec_full[0] = BOS then ref frames). Must use the exact same summation
/// order as `build_icl_block` so cached KVs stay bit-identical.
fn build_icl_prefix(
    talker: &TalkerWeights,
    predictor: &CodePredictorWeights,
    ref_tokens: &[u32],
    ref_codes: &[Vec<u32>],
) -> Vec<Vec<f32>> {
    let tv = ref_codes[0].len();
    let mut prefix = Vec::with_capacity(ref_tokens.len());
    for (i, &tok) in ref_tokens.iter().enumerate() {
        let text_e = crate::talker::embed_text_token(talker, tok);
        let codec_e = if i == 0 {
            crate::talker::embed_codec_token(talker, CODEC_BOS)
        } else {
            // ref frame i-1 (codec_full[i] for i>=1); i <= R <= Tv holds in
            // the pad branch (R+1 <= Tv+1); in the remainder branch the block
            // is truncated to codec_len so i < codec_len <= Tv+1 either way.
            // Guard the degenerate R > Tv case with PAD (never hit in practice).
            if i - 1 < tv {
                let t = i - 1;
                let mut sum = crate::talker::embed_codec_token(talker, ref_codes[0][t]);
                for g in 1..16 {
                    let e = crate::code_predictor::get_acoustic_embedding(predictor, g - 1, ref_codes[g][t]);
                    for j in 0..talker.hidden_size {
                        sum[j] += e[j];
                    }
                }
                sum
            } else {
                crate::talker::embed_codec_token(talker, CODEC_PAD)
            }
        };
        prefix.push(add_embed(&text_e, &codec_e));
    }
    prefix
}

/// Pure split rule for the prefix-cache suffix path (unit-tested, no weights).
/// `block_len` = full ICL block length, `r` = ref-token count, `prefix_len` =
/// cached prefix length. Returns the suffix length to prefill, or None when
/// the cache is unusable (length mismatch, degenerate coverage, empty suffix)
/// and the caller must fall back to full prefill.
pub fn prefix_suffix_split(block_len: usize, r: usize, prefix_len: usize) -> Option<usize> {
    if prefix_len != 9 + r {
        return None;
    }
    if r < block_len {
        Some(block_len - r)
    } else {
        None
    }
}
/// Cached chunk-independent talker prefix: KV after prefilling
/// role(3) + codec header(6) + ICL ref part(R). `len` = 9+R (ICL) or 9.
/// Per chunk, only the text-dependent suffix (block[R..]) is prefilled
/// on top of a clone. Positions are absolute → the suffix lands exactly
/// where a full prefill would put it (bit-identical, verified by sha256).
pub struct TalkerPrefixCache {
    pub kv: gemv::RawKvCache,
    pub len: usize,
}

/// Prefill the chunk-independent prefix once per run. Returns None when
/// there is nothing worth caching (no voice embedding and no ICL — the
/// 9-position head alone saves too little to matter; caller falls back).
pub fn build_talker_prefix(
    talker: &TalkerWeights,
    predictor: &CodePredictorWeights,
    speaker_id: u32,
    language_id: u32,
    voice_embedding: Option<&[f32]>,
    ref_text_tokens: Option<&[u32]>,
    ref_codes: Option<&[Vec<u32>]>,
) -> Option<TalkerPrefixCache> {
    // Without ICL the reusable head is 9 positions of ~10 — skip.
    let (rt, rc) = match (ref_text_tokens, ref_codes) {
        (Some(rt), Some(rc)) => (rt, rc),
        _ => return None,
    };
    let t = Instant::now();
    let mut embeds = build_prefill_head(talker, speaker_id, language_id, voice_embedding);
    embeds.extend(build_icl_prefix(talker, predictor, rt, rc));
    let len = embeds.len();
    let hidden = talker.hidden_size;
    let flat: Vec<f32> = embeds.iter().flat_map(|e| e.iter().copied()).collect();
    debug_assert_eq!(flat.len(), len * hidden);
    let mut kv = gemv::empty_kv_cache(talker.num_layers(), talker.num_kv_heads, talker.head_dim);
    let _ = crate::talker::prefill_talker_raw(talker, &flat, len, &mut kv);
    eprintln!("Talker prefix cache: {len} positions in {:.1?} (reused by every chunk)", t.elapsed());
    Some(TalkerPrefixCache { kv, len })
}

/// Prefill for CustomVoice matching qwen3-tts-rs structure
/// Returns (last_hidden_state, logits) where:
/// - last_hidden_state: [hidden_size] - the hidden state from the last position
/// - logits: [vocab_size] - the codec head output for sampling first token
fn prefill_custom_voice(
    talker: &TalkerWeights,
    text_tokens: &[u32],
    speaker_id: u32,
    language_id: u32,
    voice_embedding: Option<&[f32]>,
    kv_cache: &mut gemv::RawKvCache,
    icl_embeds: Option<&[Vec<f32>]>,
) -> (Vec<f32>, Vec<f32>) {
    let hidden_size = talker.hidden_size;

    // 1. Role prefix: [IM_START, ASSISTANT, NEWLINE] projected
    let role_prefix = build_role_prefix(talker);

    // 2. Build codec + text overlay sequence
    // Structure depends on whether voice embedding is provided:
    //   With voice: [THINK+PAD, THINK_BOS+PAD, lang+PAD, THINK_EOS+PAD, voice+PAD, PAD+TTS_BOS, BOS+first_text]
    //   Without:    [THINK+PAD, THINK_BOS+PAD, lang+PAD, THINK_EOS+PAD, spk+PAD, PAD+TTS_BOS, BOS+first_text]

    let mut all_embeds = Vec::new();
    all_embeds.extend(role_prefix);           // 3 positions (text-only)

    if voice_embedding.is_some() {
        // Voice cloning path: 4 codec+TTS_PAD, 1 voice+TTS_PAD, PAD+TTS_BOS, BOS+first_text
        let prefix_codec = [CODEC_THINK, CODEC_THINK_BOS, language_id, CODEC_THINK_EOS];
        let tts_pad_proj = crate::talker::embed_text_token(talker, TTS_PAD);
        let tts_bos_proj = crate::talker::embed_text_token(talker, TTS_BOS);

        // Positions 0-3: codec + TTS_PAD overlay (dual-stream)
        for &tok in &prefix_codec {
            let codec_emb = crate::talker::embed_codec_token(talker, tok);
            let mut combined = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                combined[j] = codec_emb[j] + tts_pad_proj[j];
            }
            all_embeds.push(combined);
        }

        // Position 4: voice embedding + TTS_PAD overlay (matches official:
        // _talker_input_embed = cat(tts_pad x5, tts_bos) + codec[:, :-1])
        let emb = voice_embedding.unwrap();
        let voice_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("  Voice embedding norm={:.4}, injected with TTS_PAD overlay at position 4", voice_norm);
        let mut pos4 = vec![0.0f32; hidden_size];
        for j in 0..hidden_size.min(emb.len()) {
            pos4[j] = emb[j] + tts_pad_proj[j];
        }
        all_embeds.push(pos4);

        // Position 5: PAD + TTS_BOS overlay
        let pad_emb = crate::talker::embed_codec_token(talker, CODEC_PAD);
        let mut pad_combined = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            pad_combined[j] = pad_emb[j] + tts_bos_proj[j];
        }
        all_embeds.push(pad_combined);

        // ICL mode (official streaming branch): base is role + 6 codec positions,
        // then the ICL block; the BOS+first_text position is NOT appended.
        // Non-ICL: unchanged legacy path below.
        if let Some(icl) = icl_embeds {
            all_embeds.extend(icl.iter().cloned());
        } else if !text_tokens.is_empty() {
            // Position 6: BOS + first_text (if text exists)
            let bos_emb = crate::talker::embed_codec_token(talker, CODEC_BOS);
            let first_text_proj = crate::talker::embed_text_token(talker, text_tokens[0]);
            let mut combined = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                combined[j] = bos_emb[j] + first_text_proj[j];
            }
            all_embeds.push(combined);
        }
    } else {
        // Built-in speaker path: all 7 codec positions with TTS overlay
        let codec_tokens = [CODEC_THINK, CODEC_THINK_BOS, language_id, CODEC_THINK_EOS, speaker_id, CODEC_PAD, CODEC_BOS];
        let mut codec_embeds: Vec<Vec<f32>> = codec_tokens.iter()
            .map(|&t| crate::talker::embed_codec_token(talker, t))
            .collect();

        // TTS overlay: [TTS_PAD × 5, TTS_BOS × 1] applied to first 6 positions
        let tts_overlay = build_tts_pad_bos(talker, 5);
        for i in 0..6 {
            for j in 0..hidden_size {
                codec_embeds[i][j] += tts_overlay[i][j];
            }
        }

        // Add codec positions 0-5 (dual-stream)
        all_embeds.extend(codec_embeds.into_iter().take(6));

        // BOS + first_text
        if !text_tokens.is_empty() {
            let bos_emb = crate::talker::embed_codec_token(talker, CODEC_BOS);
            let first_text_proj = crate::talker::embed_text_token(talker, text_tokens[0]);
            let mut combined = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                combined[j] = bos_emb[j] + first_text_proj[j];
            }
            all_embeds.push(combined);
        }
    }

    // 7. Run prefill through transformer layers
    let seq_len = all_embeds.len();
    let mut x = vec![0.0f32; seq_len * hidden_size];
    for (t, emb) in all_embeds.iter().enumerate() {
        x[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(emb);
    }

    let last_hidden = crate::talker::prefill_talker_raw(talker, &x, seq_len, kv_cache);

    // Apply codec head to get logits
    let logits = crate::talker::apply_codec_head(talker, &last_hidden);

    (last_hidden, logits)
}

/// Generate speech matching qwen3-tts-rs architecture
pub fn generate_speech(
    talker: &TalkerWeights,
    predictor: &CodePredictorWeights,
    decoder: &SpeechDecoderWeights,
    tokenizer: &TTSTokenizer,
    text: &str,
    speaker_id: u32,
    language_id: u32,
    _voice_codes: Option<&[Vec<u32>]>,
    voice_embedding: Option<&[f32]>,
    params: &TTSParams,
    seed: Option<u64>,
    ref_text_tokens: Option<Vec<u32>>,
    ref_codes: Option<&[Vec<u32>]>,
    // Chained decoder warmup: tail codes of the previous chunk (same voice,
    // freshest context). Takes precedence over ref warmup when present.
    // Talker ICL still uses `ref_codes` — only the decoder source switches.
    chain_codes: Option<&[Vec<u32>]>,
    // Talker prefix cache (P0): chunk-independent KV built once per run.
    // When present and compatible (ICL block covers the cached R), only the
    // text-dependent suffix is prefilled on top of a clone. None = legacy
    // full prefill (single-chunk runs, non-ICL, or --no-prefix-cache).
    talker_prefix: Option<&TalkerPrefixCache>,
) -> (Vec<f32>, Vec<Vec<u32>>) {
    let t0 = Instant::now();

    // Tokenize text
    let text_tokens = tokenizer.encode(text);
    eprintln!("Tokenized {} tokens", text_tokens.len());

    // ICL mode: build the ICL block (official streaming branch) from reference
    // text tokens + reference codes. In ICL mode the per-step fusion is the
    // block remainder only (no legacy trailing-text fusion).
    let icl: Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> = match (&ref_text_tokens, &ref_codes) {
        (Some(rt), Some(rc)) => {
            let (block, remainder) = build_icl_block(talker, predictor, rt, &text_tokens, rc);
            eprintln!("ICL mode: ref_text {} tokens, ref_codes 16x{}, block {}, remainder {}",
                rt.len(), rc[0].len(), block.len(), remainder.len());
            Some((block, remainder))
        }
        _ => None,
    };

    // Initialize KV cache (prefix-cache hit → clone, else fresh)
    let mut talker_kv: gemv::RawKvCache;
    let t_prefill = Instant::now();
    let (mut last_hidden, mut logits);
    let mut position: usize;
    // Suffix path: usable iff ICL is active, the cache exists, the cached
    // length matches 9+R, and the block covers R with a non-empty suffix.
    // (Empty-suffix edge — empty text — falls back to exact full prefill.)
    let use_suffix: bool = match (&icl, talker_prefix) {
        (Some((block, _)), Some(pfx)) => {
            let r = ref_text_tokens.as_ref().map(|t| t.len()).unwrap_or(0);
            prefix_suffix_split(block.len(), r, pfx.len).is_some()
        }
        _ => false,
    };
    if use_suffix {
        let (block, _) = icl.as_ref().unwrap();
        let pfx = talker_prefix.unwrap();
        let r = ref_text_tokens.as_ref().unwrap().len();
        talker_kv = pfx.kv.clone();
        let hidden = talker.hidden_size;
        let suffix: Vec<f32> = block[r..].iter().flat_map(|e| e.iter().copied()).collect();
        let suffix_len = block.len() - r;
        debug_assert_eq!(suffix.len(), suffix_len * hidden);
        last_hidden = crate::talker::prefill_talker_raw_from(
            talker, &suffix, suffix_len, &mut talker_kv, pfx.len);
        logits = crate::talker::apply_codec_head(talker, &last_hidden);
        position = 9 + block.len();
        eprintln!("Prefill done in {:.1?}, position={} (suffix {} on cached {})",
            t_prefill.elapsed(), position, suffix_len, pfx.len);
    } else {
        talker_kv = gemv::empty_kv_cache(talker.num_layers(), talker.num_kv_heads, talker.head_dim);
        // Prefill with proper dual-stream architecture
        let full = prefill_custom_voice(talker, &text_tokens, speaker_id, language_id, voice_embedding, &mut talker_kv, icl.as_ref().map(|(b, _)| b.as_slice()));
        last_hidden = full.0;
        logits = full.1;
        position = match &icl {
            // 3 role + 6 codec + ICL block (no first_text position, official streaming ICL)
            Some((block, _)) => 9 + block.len(),
            // legacy: 3 role + 6 codec + 1 first_text
            None => if text_tokens.is_empty() { 9 } else { 10 },
        };
        eprintln!("Prefill done in {:.1?}, position={}", t_prefill.elapsed(), position);
    }

    // Build trailing text embeddings (remaining text tokens after first + TTS_EOS).
    // ICL mode overrides this with the block remainder (may be empty → always pad).
    let trailing_text: Vec<Vec<f32>> = match &icl {
        Some((_, remainder)) => remainder.clone(),
        None => build_trailing_text(talker, &text_tokens),
    };
    let trailing_text_len = trailing_text.len();
    let tts_pad_embed = crate::talker::embed_text_token(talker, TTS_PAD);

    eprintln!("Trailing text length: {}", trailing_text_len);

    // Initialize code predictor KV cache
    let mut predictor_kv = gemv::empty_kv_cache(predictor.layers.len(), predictor.num_kv_heads, predictor.head_dim);

    // Generate codes frame by frame
    let mut all_codes: Vec<Vec<u32>> = vec![Vec::new(); 16];
    let mut prev_tokens = vec![0u32; 16];  // For repetition penalty

    let mut rng_state: u64 = match seed {
        Some(s) => {
            eprintln!("Seed: {s} (explicit)");
            s
        }
        None => {
            let s = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            eprintln!("Seed: {s} (random, reuse with --seed {s})");
            s
        }
    };

    let t_gen = Instant::now();
    // Phase split (GPT review P0): talker forward vs predictor per frame.
    // Sampling/pushes are unattributed (µs). Printed on the done line.
    let mut t_talker = std::time::Duration::ZERO;
    let mut t_pred = std::time::Duration::ZERO;

    for frame_idx in 0..params.max_codes {
        // Sample semantic token from logits (onset schedule for chunk attacks)
        let eff_temp = frame_temperature(params, frame_idx);
        let semantic_token = sample_token(&logits, params, &prev_tokens, &mut rng_state, eff_temp);

        if semantic_token == params.codec_eos_id {
            eprintln!("EOS at frame {}", frame_idx);
            break;
        }

        all_codes[0].push(semantic_token);
        prev_tokens[0] = semantic_token;

        // Get semantic embedding
        let semantic_embed = crate::talker::embed_codec_token(talker, semantic_token);

        // Generate 15 acoustic codes using code predictor
        // (prefill 2 + 14 AR steps, 5 layers each — timed separately)
        let s_pred = Instant::now();
        let acoustic_codes = crate::code_predictor::generate_acoustic_codes(
            predictor,
            &last_hidden,
            &semantic_embed,
            &mut predictor_kv,
        );
        t_pred += s_pred.elapsed();

        for (i, &code) in acoustic_codes.iter().enumerate() {
            all_codes[i + 1].push(code);
            prev_tokens[i + 1] = code;
        }

        // Build input for next token: semantic + (SUM of all 15 acoustic embeddings) + text fusion
        let mut combined_embed = semantic_embed;
        let acoustic_embed_sum = crate::code_predictor::get_acoustic_embeddings_sum(predictor, &acoustic_codes);
        for j in 0..talker.hidden_size {
            combined_embed[j] += acoustic_embed_sum[j];
        }

        // Add trailing text fusion
        let text_addition = if frame_idx < trailing_text_len {
            &trailing_text[frame_idx]
        } else {
            &tts_pad_embed
        };

        for j in 0..talker.hidden_size {
            combined_embed[j] += text_addition[j];
        }

        // Forward through talker with combined embedding
        let s_talk = Instant::now();
        let result = crate::talker::forward_with_embedding(talker, &combined_embed, &mut talker_kv, position);
        t_talker += s_talk.elapsed();

        // Extract hidden state (first hidden_size elements) and logits (rest)
        let hidden_size = talker.hidden_size;
        last_hidden = result[..hidden_size].to_vec();
        logits = result[hidden_size..].to_vec();
        position += 1;

        if frame_idx % 10 == 0 {
            eprint!("\rGenerating frame {}/{}...", frame_idx + 1, params.max_codes);
        }
    }

    eprintln!("\nGeneration done in {:.1?} ({} frames, talker {:.1?}, predictor {:.1?})",
        t_gen.elapsed(), all_codes[0].len(), t_talker, t_pred);

    // Diagnostic dump of generated codes (VCOD format, same as --voice-codes
    // input). Gated by QORA_DUMP_CODES=<path>; used to inspect frame-level
    // patterns behind audible artifacts (stuck runs, oscillation, thrash).
    if let Ok(dump_path) = std::env::var("QORA_DUMP_CODES") {
        if let Err(e) = write_codes_file(std::path::Path::new(&dump_path), &all_codes) {
            eprintln!("Failed to dump codes to {dump_path}: {e}");
        } else {
            eprintln!("Codes dumped to {dump_path} ({} codebooks x {} frames)",
                all_codes.len(), all_codes[0].len());
        }
    }

    // Decode to audio
    let t_decode = Instant::now();
    let warmed: Vec<Vec<u32>>;
    // Chained tail (previous chunk, same voice) wins over ref warmup:
    // fresher context, bounded cost independent of ref length.
    let warmup_opt: Option<&[Vec<u32>]> = match chain_codes {
        Some(cc) if !cc.is_empty() && !cc[0].is_empty() => {
            eprintln!("  Decoder warmup: chained {} frames (prev chunk tail)", cc[0].len());
            Some(cc)
        }
        _ if params.decode_warmup => match ref_codes.as_deref() {
            Some(rc) if params.warmup_frames > 0 => {
                warmed = warmup_tail(rc, params.warmup_frames);
                eprintln!("  Decoder warmup: trailing {} of {} ref frames",
                    warmed[0].len(), rc[0].len());
                Some(&warmed)
            }
            other => other,
        },
        _ => None,
    };
    let audio = crate::decoder::decode_to_audio(
        decoder,
        &all_codes,
        warmup_opt,
    );
    eprintln!("Decode done in {:.1?}", t_decode.elapsed());

    eprintln!("Total: {:.1?}", t0.elapsed());
    (audio, all_codes)
}

/// Trailing-K slice of ref codes for decoder warmup (see TTSParams).
/// k == 0 or k >= frames → full passthrough (official behavior).
/// Returns owned per-book tails (decoder takes slices; Vec<Vec> can't
/// sub-slice frames without copying).
pub fn warmup_tail(codes: &[Vec<u32>], k: usize) -> Vec<Vec<u32>> {
    if codes.is_empty() {
        return Vec::new();
    }
    let n = codes[0].len();
    if k == 0 || k >= n {
        return codes.to_vec();
    }
    codes.iter().map(|q| q[q.len() - k..].to_vec()).collect()
}

/// Write codes in VCOD format (magic + u32 groups + u32 frames + u16 codes,
/// group-major), readable by --voice-codes / load_all_voice_codes.
fn write_codes_file(path: &std::path::Path, codes: &[Vec<u32>]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"VCOD")?;
    f.write_all(&(codes.len() as u32).to_le_bytes())?;
    f.write_all(&(codes[0].len() as u32).to_le_bytes())?;
    for q in codes {
        for &c in q {
            f.write_all(&(c as u16).to_le_bytes())?;
        }
    }
    f.flush()
}

/// Build trailing text embeddings: text_proj(text[1..]) + TTS_EOS
fn build_trailing_text(talker: &TalkerWeights, text_tokens: &[u32]) -> Vec<Vec<f32>> {    let mut trailing = Vec::new();

    // Add remaining text tokens (skip first token which was used in prefill)
    for &token_id in text_tokens.iter().skip(1) {
        trailing.push(crate::talker::embed_text_token(talker, token_id));
    }

    // Add TTS_EOS at the end
    trailing.push(crate::talker::embed_text_token(talker, TTS_EOS));

    trailing
}

/// Sample next token with temperature, top-k, top-p, and repetition penalty.
/// `temperature` is passed per-frame (onset schedule); the rest from params.
fn sample_token(logits: &[f32], params: &TTSParams, prev_tokens: &[u32], rng: &mut u64, temperature: f32) -> u32 {
    let mut scores = logits.to_vec();

    // Apply repetition penalty
    if params.repetition_penalty != 1.0 {
        for &prev_token in prev_tokens {
            if (prev_token as usize) < scores.len() {
                if scores[prev_token as usize] > 0.0 {
                    scores[prev_token as usize] /= params.repetition_penalty;
                } else {
                    scores[prev_token as usize] *= params.repetition_penalty;
                }
            }
        }
    }

    // Apply temperature
    if temperature != 1.0 && temperature > 0.0 {
        for s in &mut scores {
            *s /= temperature;
        }
    }

    // Softmax
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for s in &mut scores {
        *s = (*s - max_score).exp();
    }
    let sum: f32 = scores.iter().sum();
    for s in &mut scores {
        *s /= sum;
    }

    // Top-k filtering
    if params.top_k > 0 && params.top_k < scores.len() {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (i, _) in indexed.iter().skip(params.top_k) {
            scores[*i] = 0.0;
        }

        // Renormalize
        let sum: f32 = scores.iter().sum();
        if sum > 0.0 {
            for s in &mut scores {
                *s /= sum;
            }
        }
    }

    // Top-p (nucleus) filtering: keep smallest set with cumulative mass >= top_p
    // (always keeps the argmax). top_p >= 1.0 or <= 0.0 = off.
    if params.top_p > 0.0 && params.top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cum = 0.0f32;
        let mut cutoff = indexed.len();
        for (rank, &(_, p)) in indexed.iter().enumerate() {
            cum += p;
            if cum >= params.top_p {
                cutoff = rank + 1;
                break;
            }
        }
        let cutoff = cutoff.max(1);
        for (i, _) in indexed.iter().skip(cutoff) {
            scores[*i] = 0.0;
        }

        // Renormalize
        let sum: f32 = scores.iter().sum();
        if sum > 0.0 {
            for s in &mut scores {
                *s /= sum;
            }
        }
    }

    // Sample from distribution
    let r = simple_rand(rng);
    let mut cumsum = 0.0;
    for (i, &p) in scores.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icl_layout_text_longer() {
        // R=8 ref tokens, T=8 text tokens, Tv=23 frames:
        // text_len=17 > codec_len=24? No: 8+8+1=17 < 24 → else branch.
        let l = icl_block_layout(8, 8, 23);
        assert_eq!(l.icl_len, 24);
        assert_eq!(l.remainder_len, 0);
    }

    #[test]
    fn test_icl_layout_remainder() {
        // R=30, T=30, Tv=23: text_len=61 > codec_len=24 → remainder 37.
        let l = icl_block_layout(30, 30, 23);
        assert_eq!(l.icl_len, 24);
        assert_eq!(l.remainder_len, 37);
    }

    #[test]
    fn test_icl_layout_equal() {
        // text_len == codec_len → else branch, no remainder.
        // R=11, T=11, Tv=22: 11+11+1=23 == 22+1.
        let l = icl_block_layout(11, 11, 22);
        assert_eq!(l.icl_len, 23);
        assert_eq!(l.remainder_len, 0);
    }

    #[test]
    fn test_icl_layout_short_ref() {
        // R=1, T=1, Tv=23: text_len=3 < 24 → pad branch.
        let l = icl_block_layout(1, 1, 23);
        assert_eq!(l.icl_len, 24);
        assert_eq!(l.remainder_len, 0);
    }

    fn test_params_top_p(top_p: f32) -> TTSParams {
        TTSParams {
            max_codes: 10,
            temperature: 1.0,
            top_k: 0,
            top_p,
            repetition_penalty: 1.0,
            codec_eos_id: CODEC_EOS,
            codec_bos_id: CODEC_BOS,
            onset_frames: 0,
            onset_temperature: 0.3,
            decode_warmup: false,
            warmup_frames: 0,
        }
    }

    #[test]
    fn test_top_p_off_by_default() {
        // top_p=1.0 (and <=0) must not filter: every token still reachable
        let logits = vec![2.0, 1.0, 0.5, 0.1];
        for tp in [1.0, 0.0, -0.5, 2.0] {
            let p = test_params_top_p(tp);
            // try many rng states; all 4 tokens must be drawable
            let mut seen = [false; 4];
            let mut rng: u64 = 12345;
            for _ in 0..200 {
                let t = sample_token(&logits, &p, &[], &mut rng, 1.0);
                seen[t as usize] = true;
            }
            assert!(seen.iter().all(|&s| s), "top_p={tp}: all tokens reachable");
        }
    }

    #[test]
    fn test_top_p_cuts_tail() {
        // logits [5,1,1,1] @ temp 1: top token p ≈ 0.93; top_p=0.5 keeps only it
        let logits = vec![5.0, 1.0, 1.0, 1.0];
        let p = test_params_top_p(0.5);
        let mut rng: u64 = 999;
        for _ in 0..50 {
            assert_eq!(sample_token(&logits, &p, &[], &mut rng, 1.0), 0);
        }
    }

    #[test]
    fn test_top_p_keeps_argmax_always() {
        // even tiny top_p keeps at least the argmax (cutoff.max(1))
        let logits = vec![3.0, 2.9, 2.8];
        let p = test_params_top_p(0.01);
        let mut rng: u64 = 7;
        for _ in 0..20 {
            assert_eq!(sample_token(&logits, &p, &[], &mut rng, 1.0), 0);
        }
    }

    #[test]
    fn test_frame_temperature_schedule() {
        let mut p = TTSParams::default();
        p.temperature = 0.8;
        p.onset_frames = 8;
        p.onset_temperature = 0.3;
        assert_eq!(frame_temperature(&p, 0), 0.3);
        assert_eq!(frame_temperature(&p, 7), 0.3);
        assert_eq!(frame_temperature(&p, 8), 0.8);
        assert_eq!(frame_temperature(&p, 500), 0.8);
        p.onset_frames = 0;
        assert_eq!(frame_temperature(&p, 0), 0.8); // off by default
    }

    #[test]
    fn test_warmup_tail() {
        let codes = vec![vec![1u32, 2, 3, 4, 5], vec![6u32, 7, 8, 9, 10]];
        // k=0 → full passthrough
        assert_eq!(warmup_tail(&codes, 0), codes);
        // k>=frames → full passthrough
        assert_eq!(warmup_tail(&codes, 5), codes);
        assert_eq!(warmup_tail(&codes, 99), codes);
        // trailing slice
        assert_eq!(warmup_tail(&codes, 2), vec![vec![4u32, 5], vec![9u32, 10]]);
        // empty
        assert!(warmup_tail(&[], 3).is_empty());
    }

    #[test]
    fn test_prefix_suffix_split_hit() {
        // R=35, block=162 → prefix 44, suffix 127
        assert_eq!(prefix_suffix_split(162, 35, 44), Some(127));
    }

    #[test]
    fn test_prefix_suffix_split_miss() {
        // stale cache length
        assert_eq!(prefix_suffix_split(162, 35, 43), None);
        // degenerate: R covers the whole block (empty suffix → fallback)
        assert_eq!(prefix_suffix_split(35, 35, 44), None);
        // degenerate: R beyond block
        assert_eq!(prefix_suffix_split(30, 35, 44), None);
        // zero ref (non-ICL shape never reaches here, still defined)
        assert_eq!(prefix_suffix_split(162, 0, 9), Some(162));
    }
}

#[cfg(test)]
mod codes_tests {
    use super::*;

    #[test]
    fn test_codes_roundtrip() {
        let codes = vec![vec![1u32, 2047, 0], vec![4095u32, 7, 13]];
        let p = std::env::temp_dir().join("qora_codes_rt.vcod");
        write_codes_file(&p, &codes).unwrap();
        // read back with the same layout main.rs uses
        use std::io::Read;
        let mut f = std::fs::File::open(&p).unwrap();
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic).unwrap();
        assert_eq!(&magic, b"VCOD");
        let mut b4 = [0u8; 4];
        f.read_exact(&mut b4).unwrap();
        assert_eq!(u32::from_le_bytes(b4), 2);
        f.read_exact(&mut b4).unwrap();
        assert_eq!(u32::from_le_bytes(b4), 3);
        let mut raw = vec![0u8; 12];
        f.read_exact(&mut raw).unwrap();
        let vals: Vec<u32> = raw.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect();
        assert_eq!(vals, vec![1, 2047, 0, 4095, 7, 13]);
    }
}
