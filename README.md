

# QORA-TTS 1.7B - Pure Rust Text-to-Speech with Voice Cloning

Pure Rust TTS engine with voice cloning. No Python, no CUDA, no external ML frameworks. Single executable + model weights = portable text-to-speech that runs on any machine.

**Smart system awareness** — automatically detects your hardware (RAM, CPU threads) and adjusts generation limits so TTS runs well even on constrained systems. **Voice cloning** — clone any voice from a 3-10 second WAV recording. **25 included voices** — ready to use out of the box.

Based on [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) (Apache 2.0).

## License

This project is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). The base model [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) is released by the Qwen team under Apache 2.0.

## What It Does

QORA-TTS 1.7B converts text to natural-sounding speech. It can:

- **Voice cloning** — clone any voice from a short WAV recording (3-10 seconds)
- **25 built-in voices** — 13 female + 12 male voices included, ready to use
- **10 languages** — English, Chinese, German, Italian, Portuguese, Spanish, Japanese, Korean, French, Russian
- **24 kHz output** — high-quality mono WAV audio
- **Controllable generation** — adjust length, temperature, and sampling parameters

## Platform Support

| Platform | Binary | Status |
|----------|--------|--------|
| **Windows x86_64** | `qora-tts.exe` | Tested |
| **Linux x86_64** | `qora-tts` | Supported |
| **macOS aarch64** | `qora-tts` | Supported |

CPU-only — no GPU needed. Pre-built binaries on the [Releases](https://github.com/qora-protocol/QORA-TTS-12Hz-1.7B/releases) page.

## Quick Start

1. Download from the [Releases](https://github.com/qora-protocol/QORA-TTS-12Hz-1.7B/releases) page
2. Run:

```bash
# Voice cloning with included voice
qora-tts.exe --ref-audio voices/luna.wav --text "Hello, how are you?" --language english

# Different voice
qora-tts.exe --ref-audio voices/adam.wav --text "Good morning!" --language english

# Clone your own voice (any 24kHz WAV)
qora-tts.exe --ref-audio my_recording.wav --text "Custom voice" --language english

# Control length (codes = seconds x 12.5)
qora-tts.exe --ref-audio voices/luna.wav --text "Short" --max-codes 100

# Custom output path
qora-tts.exe --ref-audio voices/sagar.wav --text "Hi there" --language english --output greeting.wav

# Reproducible output with seed (the actual seed is printed each run when omitted)
qora-tts.exe --ref-audio voices/luna.wav --text "Same every time" --seed 42

# ICL mode: clone prosody too (needs reference transcript WITH punctuation)
qora-tts.exe --ref-audio my_recording.wav --ref-text "Hello, how are you today?" --text "Custom voice" --language english
```

## Files

```
qora-tts.exe          4.1 MB   Inference engine
model.qora-tts     1559 MB     Q4 weights (talker + predictor + decoder + speaker encoder)
config.json           4.4 KB   Model configuration
tokenizer.json         11 MB   Tokenizer (151,936 vocab)
vocab.json            2.7 MB   Vocabulary
merges.txt            1.6 MB   BPE merges
tokenizer_config.json 7.2 KB   Tokenizer config
voices/                         25 reference WAV files for voice cloning
```

**No safetensors needed.** Everything loads from `model.qora-tts`. The exe auto-finds all files in its own directory.

## Architecture

| Component | Details |
|-----------|---------|
| **Parameters** | 1.7B total |
| **Talker** | 28 layers, hidden=2048, 16/8 GQA heads, SwiGLU FFN 6144 |
| **Code Predictor** | 5 layers, hidden=1024, input_proj [1024, 2048], 16 code groups |
| **Speech Decoder** | 8-layer transformer + Vocos vocoder, 16 VQ codebooks |
| **Speaker Encoder** | ECAPA-TDNN (3 Res2Net blocks, 2048-dim output) |
| **Quantization** | Q4 (4-bit symmetric, group_size=32) with LUT-optimized dequantization |
| **Sample Rate** | 24 kHz mono WAV |
| **Code Rate** | 12.5 Hz (1 code = 80ms of audio) |

### How It Works

1. **Text encoding** — tokenize input text with 151K BPE vocabulary
2. **Voice extraction** — ECAPA-TDNN speaker encoder extracts voice embedding from reference audio
3. **Code generation** — 28-layer transformer (Talker) generates speech codes autoregressively
4. **Code expansion** — 5-layer Code Predictor expands code0 into 16 codebooks (codes 0-15)
5. **Audio synthesis** — VQ decoder + Vocos vocoder converts codes to 24kHz waveform

### AVX-512 SIMD Acceleration

On CPUs with AVX-512 support (Intel 11th gen+, AMD Zen 4+), QORA-TTS automatically uses hand-written AVX-512 SIMD kernels for faster inference:

| Kernel | Technique | Speedup |
|--------|-----------|---------|
| **Q4 GEMV** | `permutexvar_ps` 16-entry LUT lookup, nibble extract via `cvtepu8_epi32` | ~2.5x |
| **F16 GEMV** | `cvtph_ps` f16→f32 + `fmadd_ps` FMA accumulation | ~2.5x |

Detection is automatic at runtime — falls back to scalar code on non-AVX-512 CPUs with zero overhead.

## Smart System Awareness

QORA-TTS detects your system at startup and automatically adjusts generation limits:

```
QORA-TTS — Pure Rust Text-to-Speech Engine
System: 16384 MB RAM (9856 MB free), 12 threads
```

| Available RAM | Max Codes | Default | Audio Length |
|---------------|-----------|---------|-------------|
| < 4 GB | 200 | 100 | ~8s |
| 4-8 GB | 500 | 300 | ~20s |
| 8-12 GB | 1000 | 500 | ~40s |
| >= 12 GB | 2000 | 500 | ~80s |

**Hard caps apply even to explicit user values** — if you pass `--max-codes 2000` on a system with 6 GB free RAM, it gets clamped to 500 automatically. This prevents the model from running for too long on weak systems.

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--text <text>` | "Hello, how are you today?" | Text to synthesize |
| `--ref-audio <wav>` | - | **Required** - reference WAV for voice cloning (24kHz mono, 3-10s clean speech) |
| `--ref-text <text>` | - | Reference transcript → ICL mode: prosody (pauses/intonation/rate) follows the reference. **Must be accurate, punctuation included** |
| `--encoder-weights <file>` | `<exe-dir>/speech_tokenizer/model.safetensors` | Codec encoder weights, only needed for ICL mode |
| `--language <name>` | english | Target language |
| `--output <path>` | output.wav | Output WAV path |
| `--max-codes <n>` | 500 | Max code timesteps (~n/12.5 seconds) |
| `--temperature <f>` | 0.8 | Sampling temperature |
| `--top-k <n>` | 50 | Top-K sampling |
| `--seed <n>` | random | Random seed for reproducibility (printed each run) |

### Clone modes

| Mode | Flags | Copies timbre | Copies prosody |
|------|-------|---------------|----------------|
| x-vector only | `--ref-audio` | Yes | No (model default rhythm) |
| ICL | `--ref-audio` + `--ref-text` | Yes | Yes (pauses/intonation follow reference) |

## Included Voices

| Female | Male |
|--------|------|
| luna, anushri, beth, caty, cherie, ember, faith, hope, jessica, kea, riya, vidhi, velvety | adam, charles, david, hale, heisenberg, joe, peter, quentin, sagar, steven, titan, true |

All 24kHz WAV files in `voices/`. Use any 3-10 second clean speech recording for custom voice cloning.

## Supported Languages

| Language | Flag Value |
|----------|-----------|
| English | `english` |
| Chinese | `chinese` |
| German | `german` |
| Italian | `italian` |
| Portuguese | `portuguese` |
| Spanish | `spanish` |
| Japanese | `japanese` |
| Korean | `korean` |
| French | `french` |
| Russian | `russian` |

## Performance

Tested on i5-11500 (6C/12T), 16GB RAM, CPU-only:

| Phase | Time | Notes |
|-------|------|-------|
| Model Load | ~0.8s | From binary, 1559 MB |
| Voice Extraction | ~5-10s | ECAPA-TDNN speaker encoder |
| Prefill | ~3-8s | Text + voice embedding processing |
| Code Generation | ~2.5s/code | Autoregressive, 12.5 codes/sec of audio |
| Code Expansion | ~0.1s | 5-layer predictor, 16 codebooks |
| Audio Decode | ~0.5s/frame | VQ + Vocos vocoder |
| RAM Usage | ~1560 MB | Q4 model in memory |

**Example:** "Hello, how are you?" (~3 seconds of audio) takes ~15-20 seconds total.

## Comparison with 0.6B

| | QORA-TTS 1.7B | QORA-TTS 0.6B |
|---|---------------|---------------|
| Parameters | 1.7B | 0.6B |
| Model size | 1559 MB | 971 MB |
| Voice cloning | Yes (ECAPA-TDNN) | No |
| Built-in speakers | 25 (via voice files) | 9 (embedded) |
| Code generation | ~2.5s/code | ~1.5s/code |
| Quality | Higher | Good |
| Best for | Quality + cloning | Speed + simplicity |

## Building from Source

```bash
cargo build --release
```

### Dependencies

- **Language**: Pure Rust (2021 edition)
- `half` — F16 support
- `tokenizers` — HuggingFace tokenizer
- `safetensors` — Weight loading
- `serde_json` — Config parsing
- **No ML framework** for inference — all matrix ops are hand-written Rust

### Cross-Platform Releases

Pre-built binaries via GitHub Actions for Windows x86_64, Linux x86_64, macOS aarch64.

## Model Binary Format (.qora-tts)

Custom binary format for fast loading:

```
Header:  "QTTS" magic + version + format byte
Talker:  28 transformer layers (Q4 quantized)
Predictor: 5 transformer layers + code embeddings
Decoder: VQ codebooks + 8 transformer layers + Vocos vocoder
Speaker Encoder: ECAPA-TDNN (3 Res2Net blocks)
```
