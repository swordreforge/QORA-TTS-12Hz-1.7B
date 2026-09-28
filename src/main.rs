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
            _ => {}
        }
        i += 1;
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

    eprintln!("Text: \"{text}\"");
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
        let audio = qora_tts::decoder::decode_to_audio(&decoder, &all_codes);
        eprintln!("Audio decoded in {:.1?}", t_decode.elapsed());
        qora_tts::wav::write_wav(&output_path, &audio, 24000).expect("Failed to write WAV");
        eprintln!("Saved to {}", output_path.display());
        return;
    }

    // Load tokenizer
    let tokenizer_path = base_path.join("tokenizer.json");
    let tokenizer = qora_tts::tokenizer::TTSTokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");

    eprintln!("Temperature: {temperature}, Top-K: {top_k}, Seed: {}",
        seed.map(|s| s.to_string()).unwrap_or("random".into()));

    // Load reference audio once (auto-resampled to 24kHz mono) for both
    // the speaker embedding and, in ICL mode, the codec encoder.
    let ref_audio_24k: Option<Vec<f32>> = if let Some(ref ref_path) = ref_audio_path {
        eprintln!("Loading reference audio {}...", ref_path.display());
        let (audio, converted) = qora_tts::wav::read_wav_mono_24k(ref_path)
            .unwrap_or_else(|e| {
                eprintln!("Failed to load reference audio: {e}");
                std::process::exit(1);
            });
        if converted {
            eprintln!("Reference resampled to 24kHz mono ({} samples)", audio.len());
        }
        Some(audio)
    } else {
        None
    };

    // Load speaker encoder and extract voice embedding if --ref-audio provided
    let voice_embedding: Option<Vec<f32>> = if let Some(ref audio) = ref_audio_24k {
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

    // Load voice codes if --voice-codes provided
    let voice_codes = if let Some(ref vcp) = voice_codes_path {
        eprintln!("Loading voice codes from {}...", vcp.display());
        Some(load_all_voice_codes(vcp))
    } else {
        None
    };

    // ICL mode: encode reference audio to codes + tokenize reference text.
    // ref_tokens == official ref_ids[:, 3:-2] (template stripped on both ends).
    let (ref_text_tokens, ref_codes): (Option<Vec<u32>>, Option<Vec<Vec<u32>>>) =
        match (&ref_text, &ref_audio_24k) {
            (Some(rt), Some(audio_icl)) => {
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
                let toks = tokenizer.encode(rt);
                eprintln!("Reference text: {} tokens", toks.len());
                (Some(toks), Some(codes))
            }
            _ => (None, None),
        };

    // === CPU inference ===
    let audio = qora_tts::generate_new::generate_speech(
        &talker, &predictor, &decoder, &tokenizer,
        &text, speaker_id, language_id,
        voice_codes.as_deref(),
        voice_embedding.as_deref(),
        &qora_tts::generate_new::TTSParams {
            max_codes,
            temperature,
            top_k,
            repetition_penalty: 1.05,
            codec_eos_id: 2150,
            codec_bos_id: 2149,
        },
        seed,
        ref_text_tokens,
        ref_codes.as_deref(),
    );

    // Optional silence compression (internal gaps + trailing tail)
    let audio = if trim_silence > 0.0 {
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
