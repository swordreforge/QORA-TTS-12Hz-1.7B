//! TTS generation loop — text → codec codes → audio.
//!
//! Flow:
//!   1. Tokenize text, build prompt
//!   2. Prefill text through talker
//!   3. Voice conditioning: feed reference codes through talker (if voice cloning)
//!   4. Autoregressive loop: talker → codec_head → sample code 0
//!   5. Code predictor → codes 1..15
//!   6. Decode all codes through speech decoder → waveform

use std::time::Instant;

use crate::talker::TalkerWeights;
use crate::code_predictor::CodePredictorWeights;
use crate::decoder::SpeechDecoderWeights;
use crate::tokenizer::TTSTokenizer;
use crate::gemv;

pub struct TTSParams {
    pub max_codes: usize,       // max audio code timesteps to generate
    pub temperature: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub codec_eos_id: u32,
    pub codec_bos_id: u32,
}

impl Default for TTSParams {
    fn default() -> Self {
        Self {
            max_codes: 1000, // ~80 seconds at 12.5Hz
            temperature: 0.9,
            top_k: 50,
            repetition_penalty: 1.05,
            codec_eos_id: 2150,
            codec_bos_id: 2149,
        }
    }
}

/// Simple PRNG (xorshift64).
fn simple_rand(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f32) / (u64::MAX as f32)
}

/// Top-K sampling with temperature.
fn sample_top_k(logits: &[f32], temperature: f32, top_k: usize, rng: &mut u64) -> u32 {
    if temperature <= 0.0 {
        // Greedy
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0 as u32;
    }

    let vocab = logits.len();
    let k = top_k.min(vocab);

    // Find top-K indices
    let mut indices: Vec<usize> = (0..vocab).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    indices.truncate(k);

    // Apply temperature and softmax
    let max_logit = logits[indices[0]];
    let mut probs: Vec<f32> = indices.iter()
        .map(|&i| ((logits[i] - max_logit) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs { *p /= sum; }

    // Sample
    let r = simple_rand(rng);
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return indices[i] as u32;
        }
    }
    indices[k - 1] as u32
}

/// Apply repetition penalty to logits.
fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    for &token in generated {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Generate speech from text.
/// Returns audio samples as f32 at 24kHz.
///
/// If `voice_codes` is provided (all 16 codebooks from a reference audio),
/// code 0 values are fed through the talker as voice conditioning context.
/// The model "sees" the reference audio as if it generated it, building up
/// a consistent KV cache that captures the voice pattern.
pub fn generate_speech(
    talker: &TalkerWeights,
    predictor: &CodePredictorWeights,
    decoder: &SpeechDecoderWeights,
    tokenizer: &TTSTokenizer,
    text: &str,
    speaker_id: u32,
    language_id: u32,
    voice_codes: Option<&[Vec<u32>]>,
    params: &TTSParams,
) -> Vec<f32> {
    let t0 = Instant::now();

    // 1. Build text prompt
    let text_tokens = tokenizer.format_tts_prompt(text, speaker_id, language_id);
    let num_tokens = text_tokens.len();
    eprintln!("Prompt: {} tokens", num_tokens);

    // 2. Prefill through talker
    // Format: [IM_START, ASSISTANT, NEWLINE (text)] +
    //         [think, think_bos, lang, think_eos, speaker, pad, bos (codec)] +
    //         [text tokens... (text)]
    let mut talker_kv = gemv::empty_kv_cache(talker.num_layers(), talker.num_kv_heads, talker.head_dim);
    let t_prefill = Instant::now();

    let mut prompt: Vec<(u32, bool)> = Vec::with_capacity(num_tokens);
    for (i, &token_id) in text_tokens.iter().enumerate() {
        // First 3 are text (role prefix), next 7 are codec, rest are text
        let is_text = i < 3 || i >= 10;
        prompt.push((token_id, is_text));
    }
    let mut hidden = crate::talker::prefill_talker(talker, &prompt, &mut talker_kv);
    let mut position = num_tokens;
    eprintln!("Text prefill done in {:.1?}", t_prefill.elapsed());

    // 3. Voice conditioning: feed reference codes through talker
    if let Some(codes) = voice_codes {
        let num_ref = codes[0].len();
        eprintln!("Voice conditioning: {} reference codes (~{:.1}s audio)",
            num_ref, num_ref as f32 / 12.5);
        let t_vc = Instant::now();

        // Feed codec_bos to start audio context
        hidden = crate::talker::forward_talker_decode(
            talker, params.codec_bos_id, false, &mut talker_kv, position,
        );
        position += 1;

        // Feed each reference code 0 through the talker
        // This builds up KV cache with the voice pattern
        for t in 0..num_ref {
            hidden = crate::talker::forward_talker_decode(
                talker, codes[0][t], false, &mut talker_kv, position,
            );
            position += 1;
            if t % 25 == 0 && t > 0 {
                eprintln!("  Voice ref {t}/{num_ref} ({:.1} codes/s)",
                    t as f32 / t_vc.elapsed().as_secs_f32());
            }
        }
        eprintln!("Voice conditioning done in {:.1?} ({num_ref} codes)", t_vc.elapsed());
    }

    // 4. Autoregressive code generation
    let mut all_codes: Vec<Vec<u32>> = vec![Vec::new(); 16];
    let mut prev_code0: Vec<u32> = Vec::new();
    let mut rng_state: u64 = 12345;

    let mut predictor_kv = gemv::empty_kv_cache(
        predictor.layers.len(), predictor.num_kv_heads, predictor.head_dim,
    );

    eprintln!("Generating codes (max {})...", params.max_codes);
    let t_gen = Instant::now();

    let codebook_size: u32 = 2048; // codes >= this are special tokens (BOS/EOS/PAD/etc.)
    let mut audio_steps = 0usize;

    for step in 0..params.max_codes {
        // Get logits for code group 0 from talker
        let mut logits0 = crate::talker::apply_codec_head(talker, &hidden);

        // Repetition penalty
        if params.repetition_penalty > 1.0 {
            apply_repetition_penalty(&mut logits0, &prev_code0, params.repetition_penalty);
        }

        // Sample code 0
        let code0 = sample_top_k(&logits0, params.temperature, params.top_k, &mut rng_state);

        // Check EOS
        if code0 == params.codec_eos_id {
            eprintln!("  EOS at step {step} ({audio_steps} audio codes)");
            break;
        }

        prev_code0.push(code0);

        // Only collect actual audio codes (< codebook_size) for the decoder
        if code0 < codebook_size {
            all_codes[0].push(code0);

            // Code predictor → codes 1..15 (only for actual audio codes)
            let code_logits = crate::code_predictor::predict_codes(
                predictor, &hidden, &mut predictor_kv, audio_steps,
            );

            for g in 0..15 {
                let logits = &code_logits[g];
                let code = sample_top_k(logits, params.temperature, params.top_k, &mut rng_state);
                all_codes[g + 1].push(code);
            }
            audio_steps += 1;
        } else {
            eprintln!("  Step {step}: special token {code0} (skipped for audio)");
        }

        // Feed code0 back to talker for next step (including special tokens)
        hidden = crate::talker::forward_talker_decode(
            talker, code0, false, &mut talker_kv, position,
        );
        position += 1;

        if step % 50 == 0 && step > 0 {
            let elapsed = t_gen.elapsed().as_secs_f32();
            let codes_per_sec = step as f32 / elapsed;
            eprintln!("  Step {step}: {codes_per_sec:.1} codes/s ({audio_steps} audio codes)");
        }
    }

    let num_steps = all_codes[0].len();
    let gen_time = t_gen.elapsed();
    eprintln!(
        "Generated {num_steps} code steps in {:.1?} ({:.1} codes/s, ~{:.1}s audio)",
        gen_time,
        num_steps as f32 / gen_time.as_secs_f32(),
        num_steps as f32 / 12.5,
    );

    // 5. Decode to audio
    let t_decode = Instant::now();
    let audio = crate::decoder::decode_to_audio(decoder, &all_codes, None);
    eprintln!("Audio decoded in {:.1?}", t_decode.elapsed());

    eprintln!("Total: {:.1?}", t0.elapsed());
    audio
}
