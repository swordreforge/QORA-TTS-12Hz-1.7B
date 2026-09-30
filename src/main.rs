use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse key-value arguments
    let mut text = String::from("Hello, how are you today?");
    let mut speaker = String::from("ryan");
    let mut language = String::from("english");
    let mut output_path = PathBuf::from("output.wav");
    let mut max_codes: usize = 500;
    let mut max_codes_explicit = false;
    let mut temperature: f32 = 0.8;
    let mut top_k: usize = 50;
    let mut top_p: f32 = 1.0;
    let mut onset_frames: usize = 0;
    let mut onset_temperature: f32 = 0.3;
    let mut seed: Option<u64> = None;
    let exe_dir = std::env::current_exe()
        .expect("Cannot determine executable path")
        .parent().unwrap().to_path_buf();
    let mut load_path = exe_dir.join("model.qora-tts");
    let mut voice_codes_path: Option<PathBuf> = None;
    let mut decode_codes_path: Option<PathBuf> = None;
    let mut ref_audio_path: Option<PathBuf> = None;
    let mut ref_text: Option<String> = None;
    let mut encoder_weights_path: Option<PathBuf> = None;
    let mut trim_silence: f32 = 0.0;
    let mut text_file: Option<PathBuf> = None;
    let mut load_voice_path: Option<PathBuf> = None;
    let mut save_voice_path: Option<PathBuf> = None;
    let mut decode_warmup = false;
    let mut check_target: Option<Option<PathBuf>> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--text" => {
                if i + 1 < args.len() {
                    text = args[i + 1].clone();
                    i += 1;
                }
            }
            "--speaker" => {
                if i + 1 < args.len() {
                    speaker = args[i + 1].clone();
                    i += 1;
                }
            }
            "--language" => {
                if i + 1 < args.len() {
                    language = args[i + 1].clone();
                    i += 1;
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--max-codes" => {
                if i + 1 < args.len() {
                    max_codes = args[i + 1].parse().unwrap_or(500);
                    max_codes_explicit = true;
                    i += 1;
                }
            }
            "--temperature" => {
                if i + 1 < args.len() {
                    temperature = args[i + 1].parse().unwrap_or(0.8);
                    i += 1;
                }
            }
            "--top-k" => {
                if i + 1 < args.len() {
                    top_k = args[i + 1].parse().unwrap_or(50);
                    i += 1;
                }
            }
            "--top-p" => {
                if i + 1 < args.len() {
                    top_p = args[i + 1].parse().unwrap_or(1.0);
                    i += 1;
                }
            }
            "--onset-frames" => {
                if i + 1 < args.len() {
                    onset_frames = args[i + 1].parse().unwrap_or(0);
                    i += 1;
                }
            }
            "--onset-temperature" => {
                if i + 1 < args.len() {
                    onset_temperature = args[i + 1].parse().unwrap_or(0.3);
                    i += 1;
                }
            }
            "--seed" => {
                if i + 1 < args.len() {
                    seed = Some(args[i + 1].parse().unwrap_or(12345));
                    i += 1;
                }
            }
            "--load" => {
                if i + 1 < args.len() {
                    load_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--voice-codes" => {
                if i + 1 < args.len() {
                    voice_codes_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--decode-codes" => {
                if i + 1 < args.len() {
                    decode_codes_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--ref-audio" => {
                if i + 1 < args.len() {
                    ref_audio_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--ref-text" => {
                if i + 1 < args.len() {
                    ref_text = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--encoder-weights" => {
                if i + 1 < args.len() {
                    encoder_weights_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--trim-silence" => {
                if i + 1 < args.len() {
                    trim_silence = args[i + 1].parse().unwrap_or(0.0);
                    i += 1;
                }
            }
            "--text-file" => {
                if i + 1 < args.len() {
                    text_file = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--load-voice" => {
                if i + 1 < args.len() {
                    load_voice_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--decode-warmup" => {
                decode_warmup = true;
            }
            "--save-voice" => {
                if i + 1 < args.len() {
                    save_voice_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--check" => {
                // Optional wav path: --check <file.wav> (only if next arg isn't a flag)
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    check_target = Some(Some(PathBuf::from(&args[i + 1])));
                    i += 1;
                } else {
                    check_target = Some(None);
                }
            }
            _ => {}
        }
        i += 1;
    }

    // Self-test mode: runs before model loading, exits with 0/1
    if let Some(target) = check_target {
        eprintln!("QORA-TTS self-check");
        let (items, runnable) = qora_tts::check::check_env(&exe_dir, &load_path);
        for it in &items {
            eprintln!("{}", it.line());
        }
        let mut fail = !runnable;
        if let Some(wav) = target {
            eprintln!("--- reference audio: {} ---", wav.display());
            let wavs = qora_tts::check::analyze_wav(&wav);
            for it in &wavs {
                eprintln!("{}", it.line());
            }
            // 'format'/'sample-rate' notes are informational; the rest must pass
            fail |= wavs.iter().any(|it| !it.ok && it.name != "sample-rate");
        }
        if fail {
            eprintln!("CHECK FAILED");
            std::process::exit(1);
        }
        eprintln!("CHECK PASSED");
        return;
    }

    // System awareness
    let sys = qora_tts::system::SystemInfo::detect();
    let limits = sys.smart_limits();
    eprintln!("QORA-TTS — Pure Rust Text-to-Speech Engine");
    eprintln!("System: {} MB RAM ({} MB free), {} threads",
        sys.total_ram_mb, sys.available_ram_mb, sys.cpu_threads);

    // Apply smart defaults if user didn't specify
    if !max_codes_explicit { max_codes = limits.default_max_codes; }

    // Hard cap even explicit values on weak systems
    if max_codes > limits.max_codes {
        eprintln!("System cap: max-codes {} → {}", max_codes, limits.max_codes);
        max_codes = limits.max_codes;
    }

    if let Some(msg) = limits.warning {
        eprintln!("WARNING: {msg}");
    }

    // NOTE: `text` defaults to "Hello, how are you today?" only when neither
    // --text nor --text-file is given; with --text-file the real input is
    // split into chunks later (see "Text file:" line).
    if text_file.is_some() {
        eprintln!("Text: <from {}>", text_file.as_ref().unwrap().display());
    } else {
        eprintln!("Text: \"{text}\"");
    }
    if voice_codes_path.is_some() {
        eprintln!("Voice: custom (from .codes file)");
    } else {
        eprintln!("Speaker: {speaker}, Language: {language}");
    }
    eprintln!("Max codes: {max_codes}");
    eprintln!();

    // Look for config/tokenizer next to the .qora-tts file
    let base_path = load_path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();

    // Load config
    let config = qora_tts::config::QoraTTSConfig::from_file(base_path.join("config.json"))
        .expect("Failed to load config.json");

    // Get speaker and language IDs
    // If using --ref-audio, speaker_id doesn't matter (will be overridden by embedding)
    let speaker_id = if ref_audio_path.is_some() {
        config.talker_config.spk_id.get(&speaker).copied().unwrap_or(0)
    } else {
        config.talker_config.spk_id.get(&speaker)
            .copied()
            .unwrap_or_else(|| {
                eprintln!("Unknown speaker '{speaker}', available: {:?}", config.talker_config.spk_id.keys().collect::<Vec<_>>());
                std::process::exit(1);
            })
    };
    let language_id = config.talker_config.codec_language_id.get(&language)
        .copied()
        .unwrap_or_else(|| {
            eprintln!("Unknown language '{language}', available: {:?}", config.talker_config.codec_language_id.keys().collect::<Vec<_>>());
            std::process::exit(1);
        });

    eprintln!("Speaker ID: {speaker_id}, Language ID: {language_id}");

    // === Load model from .qora-tts binary ===
    eprintln!("Loading from {}...", load_path.display());
    let t0 = Instant::now();
    let (talker, predictor, decoder, speaker_encoder_opt) = qora_tts::save::load_model(&load_path)
        .expect("Failed to load .qora-tts model");
    let mb = (talker.memory_bytes() + predictor.memory_bytes() + decoder.memory_bytes()) / (1024 * 1024);
    eprintln!("All weights loaded in {:.1?} ({mb} MB)", t0.elapsed());

    // Decode-only mode: decode codes from .codes file without running the talker
    if let Some(ref dcp) = decode_codes_path {
        eprintln!("Decode-only mode: loading codes from {}...", dcp.display());
        let all_codes = load_all_voice_codes(dcp);
        let t_decode = Instant::now();
        let audio = qora_tts::decoder::decode_to_audio(&decoder, &all_codes, None);
        eprintln!("Audio decoded in {:.1?}", t_decode.elapsed());
        qora_tts::wav::write_wav(&output_path, &audio, 24000).expect("Failed to write WAV");
        eprintln!("Saved to {}", output_path.display());
        return;
    }

    // Load tokenizer
    let tokenizer_path = base_path.join("tokenizer.json");
    let tokenizer = qora_tts::tokenizer::TTSTokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");

    eprintln!("Temperature: {temperature}, Top-K: {top_k}, Top-P: {top_p}, Seed: {}",
        seed.map(|s| s.to_string()).unwrap_or("random".into()));
    if onset_frames > 0 {
        eprintln!("Onset schedule: first {onset_frames} frames at temperature {onset_temperature}");
    }

    // Load reference audio once (auto-resampled to 24kHz mono) for both
    // the speaker embedding and, in ICL mode, the codec encoder.
    // Raw bytes are hashed for voice-profile freshness checks.
    let (ref_audio_24k, ref_audio_hash): (Option<Vec<f32>>, Option<[u8; 32]>) =
        if let Some(ref ref_path) = ref_audio_path {
            eprintln!("Loading reference audio {}...", ref_path.display());
            let raw = std::fs::read(ref_path).unwrap_or_else(|e| {
                eprintln!("Failed to load reference audio: {e}");
                std::process::exit(1);
            });
            let hash = qora_tts::voice_profile::hash_bytes(&raw);
            let (audio, converted) = qora_tts::wav::read_wav_mono_24k(ref_path)
                .unwrap_or_else(|e| {
                    eprintln!("Failed to load reference audio: {e}");
                    std::process::exit(1);
                });
            if converted {
                eprintln!("Reference resampled to 24kHz mono ({} samples)", audio.len());
            }
            (Some(audio), Some(hash))
        } else {
            (None, None)
        };

    // Voice profile cache ("style file"): text-independent conditioning
    // (embedding + ICL codes) reused across runs. Embedding usable iff the
    // audio hash matches; codes additionally require ref_text equality.
    // Anything unusable is recomputed below; --save-voice stores the result.
    struct ProfileHit {
        embedding: Option<Vec<f32>>,
        codes: Option<Vec<Vec<u32>>>,
    }
    let profile_hit: ProfileHit = match (&load_voice_path, &ref_audio_hash) {
        (Some(pp), Some(hash)) => match qora_tts::voice_profile::load_profile(pp) {
            Err(e) => {
                eprintln!("Voice profile {} unusable ({e}), recomputing", pp.display());
                ProfileHit { embedding: None, codes: None }
            }
            Ok(prof) if prof.audio_sha256 != *hash => {
                eprintln!("Voice profile audio mismatch, recomputing");
                ProfileHit { embedding: None, codes: None }
            }
            Ok(prof) => {
                let codes_ok = prof.ref_text == ref_text;
                if !codes_ok && ref_text.is_some() {
                    eprintln!("Voice profile ref_text differs, re-encoding codes");
                }
                eprintln!("Voice profile hit: embedding reused{}",
                    if codes_ok && prof.ref_codes.is_some() { ", codes reused" } else { "" });
                ProfileHit {
                    embedding: Some(prof.embedding),
                    codes: if codes_ok { prof.ref_codes } else { None },
                }
            }
        },
        (Some(pp), None) => {
            eprintln!("--load-voice given without --ref-audio, ignoring {}", pp.display());
            ProfileHit { embedding: None, codes: None }
        }
        _ => ProfileHit { embedding: None, codes: None },
    };

    // Load speaker encoder and extract voice embedding if --ref-audio provided
    let voice_embedding: Option<Vec<f32>> = if let Some(emb) = profile_hit.embedding {
        eprintln!("Voice embedding from profile: {} dims", emb.len());
        Some(emb)
    } else if let Some(ref audio) = ref_audio_24k {
        eprintln!("Loading speaker encoder for voice cloning...");
        let t0 = Instant::now();
        let speaker_encoder = speaker_encoder_opt
            .expect("Model binary does not contain speaker encoder — cannot use --ref-audio");
        let enc_mb = speaker_encoder.memory_bytes() / (1024 * 1024);
        eprintln!("Speaker encoder: {enc_mb} MB, loaded in {:.1?}", t0.elapsed());

        // Extract mel-spectrogram (speaker encoder uses different params than codec)
        eprintln!("Extracting voice embedding...");
        let mel_config = qora_tts::audio_features::MelConfig::speaker_encoder();
        let mel_spec = qora_tts::audio_features::extract_mel_spectrogram(audio, &mel_config);

        // Extract speaker embedding
        let embedding = qora_tts::speaker_encoder::extract_speaker_embedding(
            &mel_spec,
            &speaker_encoder,
            128  // n_mels
        );

        eprintln!("Voice embedding extracted: {} dims", embedding.len());
        Some(embedding)
    } else {
        None
    };

    // ICL mode requires both reference audio and reference transcript
    if ref_text.is_some() && ref_audio_path.is_none() {
        eprintln!("--ref-text requires --ref-audio (ICL needs reference codes from audio)");
        std::process::exit(1);
    }
    // Rule-based number normalization (Chinese only): ref_text must read the
    // way the reference audio actually speaks the digits.
    let ref_text: Option<String> =
        ref_text.map(|rt| qora_tts::normalize::normalize_for_language(&rt, &language));

    // Load voice codes if --voice-codes provided
    let voice_codes = if let Some(ref vcp) = voice_codes_path {
        eprintln!("Loading voice codes from {}...", vcp.display());
        Some(load_all_voice_codes(vcp))
    } else {
        None
    };

    // ICL mode: encode reference audio to codes + tokenize reference text.
    // ref_tokens == official ref_ids[:, 3:-2] (template stripped on both ends).
    // Profile-cached codes are reused when the audio hash and ref_text match.
    let (ref_text_tokens, ref_codes): (Option<Vec<u32>>, Option<Vec<Vec<u32>>>) =
        match (&ref_text, &ref_audio_24k) {
            (Some(rt), Some(audio_icl)) => {
                let toks = tokenizer.encode(rt);
                eprintln!("Reference text: {} tokens", toks.len());
                if let Some(cached) = profile_hit.codes {
                    eprintln!("Reference codes from profile: 16x{}", cached[0].len());
                    (Some(toks), Some(cached))
                } else {
                    let enc_path = encoder_weights_path.clone().unwrap_or_else(|| {
                        base_path.join("speech_tokenizer").join("model.safetensors")
                    });
                    eprintln!("ICL mode: loading codec encoder from {}...", enc_path.display());
                    let enc = qora_tts::codec_encoder::load_codec_encoder(&enc_path)
                        .unwrap_or_else(|e| {
                            eprintln!("Failed to load codec encoder: {e}");
                            eprintln!("hint: pass --encoder-weights <speech_tokenizer/model.safetensors>");
                            std::process::exit(1);
                        });
                    // Reference audio already loaded + resampled above
                    let t_enc = Instant::now();
                    let codes = qora_tts::codec_encoder::encode_waveform_to_codes(&enc, &audio_icl);
                    eprintln!("Reference encoded to 16x{} codes in {:.1?}",
                        codes[0].len(), t_enc.elapsed());
                    (Some(toks), Some(codes))
                }
            }
            _ => (None, None),
        };

    // Persist the voice profile if requested (stores whatever was computed
    // or reused this run: embedding + codes + hash + ref_text).
    if let Some(ref sp) = save_voice_path {
        match (&ref_audio_hash, &voice_embedding) {
            (Some(hash), Some(emb)) => {
                let prof = qora_tts::voice_profile::VoiceProfile {
                    audio_len: ref_audio_24k.as_ref().map(|a| a.len() as u64).unwrap_or(0),
                    audio_sha256: *hash,
                    ref_text: ref_text.clone(),
                    embedding: emb.clone(),
                    ref_codes: ref_codes.clone(),
                };
                match qora_tts::voice_profile::save_profile(sp, &prof) {
                    Ok(()) => eprintln!("Voice profile saved to {}", sp.display()),
                    Err(e) => eprintln!("Failed to save voice profile: {e}"),
                }
            }
            _ => eprintln!("--save-voice needs --ref-audio, nothing saved"),
        }
    }

    // === CPU inference (per chunk; --text-file splits long text) ===
    let chunks: Vec<String> = if let Some(ref tf) = text_file {
        let content = std::fs::read_to_string(tf).unwrap_or_else(|e| {
            eprintln!("Failed to read text file {}: {e}", tf.display());
            std::process::exit(1);
        });
        let parts = qora_tts::chunk::split_sentences(&content);
        if parts.is_empty() {
            eprintln!("Text file produced no chunks");
            std::process::exit(1);
        }
        eprintln!("Text file: {} chunks", parts.len());
        parts
    } else {
        vec![text.clone()]
    };
    // Rule-based number normalization (Chinese only; other languages pass
    // through). Applied after splitting so chunk boundaries stay stable.
    // ref_text is normalized too: it must read the way the reference audio
    // actually speaks the digits.
    let chunks: Vec<String> = chunks
        .into_iter()
        .map(|c| qora_tts::normalize::normalize_for_language(&c, &language))
        .collect();
    if language.eq_ignore_ascii_case("chinese") || language.eq_ignore_ascii_case("japanese") {
        eprintln!("Number normalization applied ({language})");
    }
    let mut audios: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
    for (idx, chunk_text) in chunks.iter().enumerate() {
        if chunks.len() > 1 {
            eprintln!("--- Chunk {}/{} ({} chars) ---", idx + 1, chunks.len(), chunk_text.chars().count());
        }
        // Deterministic per-chunk seeds when a base seed is given
        let chunk_seed = seed.map(|s| s.wrapping_add(idx as u64));
        audios.push(qora_tts::generate_new::generate_speech(
            &talker, &predictor, &decoder, &tokenizer,
            chunk_text, speaker_id, language_id,
            voice_codes.as_deref(),
            voice_embedding.as_deref(),
            &qora_tts::generate_new::TTSParams {
                max_codes,
                temperature,
                top_k,
                top_p,
                repetition_penalty: 1.05,
                codec_eos_id: 2150,
                codec_bos_id: 2149,
                onset_frames,
                onset_temperature,
                decode_warmup,
            },
            chunk_seed,
            ref_text_tokens.clone(),
            ref_codes.as_deref(),
        ));
    }
    // 30ms crossfade between chunks to avoid clicks.
    // Seam hygiene: trim each chunk BEFORE joining (when requested) so the
    // crossfade blends silence into silence. Trimming after the join cannot
    // undo a speech-on-speech crossfade already baked into the seam.
    // (Single-chunk runs trim once below; multi-chunk tails are covered by
    // the per-chunk pass, so no final pass is needed there.)
    let multi = audios.len() > 1;
    let audio = if multi {
        // Seam hygiene: per-chunk trim (if requested) + guaranteed 0.15s
        // tail pad, THEN crossfade. The pad makes hot tails end in digital
        // silence so the 30ms fade blends silence into attack instead of
        // mixing two phonemes (the glitch seen in long-form runs).
        let joined: Vec<Vec<f32>> = if trim_silence > 0.0 {
            eprintln!("Trimming {} chunks (>{trim_silence}s) before join...", audios.len());
            audios
                .iter()
                .map(|a| {
                    let t = qora_tts::wav::trim_silence(a, 24000, trim_silence);
                    qora_tts::chunk::pad_tail(&t, 24000, 0.15)
                })
                .collect()
        } else {
            audios
                .iter()
                .map(|a| qora_tts::chunk::pad_tail(a, 24000, 0.15))
                .collect()
        };
        qora_tts::chunk::crossfade_concat(&joined, 720)
    } else {
        audios.into_iter().next().unwrap()
    };

    // Optional silence compression for single-chunk output
    // (internal gaps + trailing tail).
    let audio = if trim_silence > 0.0 && !multi {
        let before = audio.len();
        let trimmed = qora_tts::wav::trim_silence(&audio, 24000, trim_silence);
        eprintln!("Trim silence (>{trim_silence}s): {} -> {} samples ({:.2}s -> {:.2}s)",
            before, trimmed.len(), before as f32 / 24000.0, trimmed.len() as f32 / 24000.0);
        trimmed
    } else {
        audio
    };

    // Save WAV
    qora_tts::wav::write_wav(&output_path, &audio, 24000)
        .expect("Failed to write WAV");

    eprintln!("Saved to {}", output_path.display());
}

/// Load ALL codebook codes from a .codes file as [16][T] Vec<Vec<u32>>.
/// Used for decode-only mode to test the decoder in isolation.
fn load_all_voice_codes(path: &std::path::Path) -> Vec<Vec<u32>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)
        .unwrap_or_else(|e| { eprintln!("Failed to open codes: {e}"); std::process::exit(1); });

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).expect("Failed to read magic");
    if &magic != b"VCOD" {
        eprintln!("Invalid codes file (bad magic)");
        std::process::exit(1);
    }

    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4).expect("Failed to read num_codebooks");
    let num_codebooks = u32::from_le_bytes(buf4) as usize;
    f.read_exact(&mut buf4).expect("Failed to read timesteps");
    let timesteps = u32::from_le_bytes(buf4) as usize;

    let total = num_codebooks * timesteps;
    let mut raw = vec![0u8; total * 2];
    f.read_exact(&mut raw).expect("Failed to read codes");

    eprintln!("Loaded {num_codebooks} codebooks x {timesteps} timesteps");

    // Parse into [codebooks][timesteps] Vec<Vec<u32>>
    let mut codes = vec![Vec::with_capacity(timesteps); num_codebooks];
    for cb in 0..num_codebooks {
        for t in 0..timesteps {
            let offset = (cb * timesteps + t) * 2;
            let val = u16::from_le_bytes([raw[offset], raw[offset + 1]]) as u32;
            codes[cb].push(val);
        }
    }

    // Print first few values
    for cb in 0..num_codebooks.min(3) {
        eprintln!("  Code {cb} first 5: {:?}", &codes[cb][..codes[cb].len().min(5)]);
    }

    codes
}
