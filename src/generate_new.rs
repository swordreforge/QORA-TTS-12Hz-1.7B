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
    pub repetition_penalty: f32,
    pub codec_eos_id: u32,
    pub codec_bos_id: u32,
}

impl Default for TTSParams {
    fn default() -> Self {
        Self {
            max_codes: 1000,
            temperature: 0.9,
            top_k: 50,
            repetition_penalty: 1.05,
            codec_eos_id: CODEC_EOS,
            codec_bos_id: CODEC_BOS,
        }
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

        // Position 6: BOS + first_text (if text exists)
        if !text_tokens.is_empty() {
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
) -> Vec<f32> {
    let t0 = Instant::now();

    // Tokenize text
    let text_tokens = tokenizer.encode(text);
    eprintln!("Tokenized {} tokens", text_tokens.len());

    // Initialize KV cache
    let mut talker_kv = gemv::empty_kv_cache(talker.num_layers(), talker.num_kv_heads, talker.head_dim);

    // Prefill with proper dual-stream architecture
    let t_prefill = Instant::now();
    let (mut last_hidden, mut logits) = prefill_custom_voice(talker, &text_tokens, speaker_id, language_id, voice_embedding, &mut talker_kv);
    let mut position = if text_tokens.is_empty() { 9 } else { 10 };  // 3 role + 6 codec + 1 first_text
    eprintln!("Prefill done in {:.1?}, position={}", t_prefill.elapsed(), position);

    // Build trailing text embeddings (remaining text tokens after first + TTS_EOS)
    let trailing_text = build_trailing_text(talker, &text_tokens);
    let trailing_text_len = trailing_text.len();
    let tts_pad_embed = crate::talker::embed_text_token(talker, TTS_PAD);

    eprintln!("Trailing text length: {}", trailing_text_len);

    // Initialize code predictor KV cache
    let mut predictor_kv = gemv::empty_kv_cache(predictor.layers.len(), predictor.num_kv_heads, predictor.head_dim);

    // Generate codes frame by frame
    let mut all_codes: Vec<Vec<u32>> = vec![Vec::new(); 16];
    let mut prev_tokens = vec![0u32; 16];  // For repetition penalty

    let mut rng_state: u64 = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    });

    let t_gen = Instant::now();

    for frame_idx in 0..params.max_codes {
        // Sample semantic token from logits
        let semantic_token = sample_token(&logits, params, &prev_tokens, &mut rng_state);

        if semantic_token == params.codec_eos_id {
            eprintln!("EOS at frame {}", frame_idx);
            break;
        }

        all_codes[0].push(semantic_token);
        prev_tokens[0] = semantic_token;

        // Get semantic embedding
        let semantic_embed = crate::talker::embed_codec_token(talker, semantic_token);

        // Generate 15 acoustic codes using code predictor
        let acoustic_codes = crate::code_predictor::generate_acoustic_codes(
            predictor,
            &last_hidden,
            &semantic_embed,
            &mut predictor_kv,
        );

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
        let result = crate::talker::forward_with_embedding(talker, &combined_embed, &mut talker_kv, position);

        // Extract hidden state (first hidden_size elements) and logits (rest)
        let hidden_size = talker.hidden_size;
        last_hidden = result[..hidden_size].to_vec();
        logits = result[hidden_size..].to_vec();
        position += 1;

        if frame_idx % 10 == 0 {
            eprint!("\rGenerating frame {}/{}...", frame_idx + 1, params.max_codes);
        }
    }

    eprintln!("\nGeneration done in {:.1?} ({} frames)", t_gen.elapsed(), all_codes[0].len());

    // Decode to audio
    let t_decode = Instant::now();
    let audio = crate::decoder::decode_to_audio(decoder, &all_codes);
    eprintln!("Decode done in {:.1?}", t_decode.elapsed());

    eprintln!("Total: {:.1?}", t0.elapsed());
    audio
}

/// Build trailing text embeddings: text_proj(text[1..]) + TTS_EOS
fn build_trailing_text(talker: &TalkerWeights, text_tokens: &[u32]) -> Vec<Vec<f32>> {
    let mut trailing = Vec::new();

    // Add remaining text tokens (skip first token which was used in prefill)
    for &token_id in text_tokens.iter().skip(1) {
        trailing.push(crate::talker::embed_text_token(talker, token_id));
    }

    // Add TTS_EOS at the end
    trailing.push(crate::talker::embed_text_token(talker, TTS_EOS));

    trailing
}

/// Sample next token with temperature, top-k, and repetition penalty
fn sample_token(logits: &[f32], params: &TTSParams, prev_tokens: &[u32], rng: &mut u64) -> u32 {
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
    if params.temperature != 1.0 {
        for s in &mut scores {
            *s /= params.temperature;
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
