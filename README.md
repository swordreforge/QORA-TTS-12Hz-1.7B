

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
# 0. Self-check first (files, LFS pointers, CPU/RAM) — exits 0/1
qora-tts.exe --check
qora-tts.exe --check my_recording.wav   # + reference-audio analysis

# 1. Basic voice cloning (timbre only)
qora-tts.exe --ref-audio voices/luna.wav --text "Hello, how are you?" --language english

# 2. Clone your own voice (any sample rate: auto-resampled to 24kHz)
qora-tts.exe --ref-audio my_recording.wav --text "Custom voice" --language chinese

# 3. ICL mode: clone prosody too (needs reference transcript WITH punctuation)
qora-tts.exe --ref-audio my_recording.wav --ref-text "Hello, how are you today?" --text "Custom voice" --language english

# 4. Reuse a voice: save once, skip encoder runs afterwards (ICL+11s ref: 79s -> 36s)
qora-tts.exe --ref-audio my_recording.wav --ref-text "Hello, how are you today?" --text "First run" --save-voice luna.qvoice
qora-tts.exe --ref-audio my_recording.wav --ref-text "Hello, how are you today?" --text "Second run" --load-voice luna.qvoice --output second.wav

# 5. Long article: per-sentence synthesis + crossfade join + silence trim
qora-tts.exe --ref-audio my_recording.wav --text-file article.txt --trim-silence 0.25 --output article.wav

# 6. The works: ICL voice + profile + long text + trim
qora-tts.exe --ref-audio my_recording.wav --ref-text "Hello, how are you today?" --text-file article.txt --load-voice luna.qvoice --trim-silence 0.25 --output article.wav

# 7. Reproducible output (the actual seed is printed each run when omitted)
qora-tts.exe --ref-audio voices/luna.wav --text "Same every time" --seed 42

# 8. Quick preview: cap length (codes = seconds x 12.5, 100 ~= 8s audio max)
qora-tts.exe --ref-audio voices/luna.wav --text "Short preview" --max-codes 100

# 9. Tune randomness (higher temperature = livelier prosody)
qora-tts.exe --ref-audio voices/adam.wav --text "Good morning!" --temperature 1.0 --top-k 80

# 10. Diagnose artifacts: dump frame-level codes (VCOD) for inspection
QORA_DUMP_CODES=debug.codes qora-tts.exe --ref-audio my_recording.wav --text "Suspect sentence" --seed 42
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
speech_tokenizer/model.safetensors  651 MB  Codec encoder (only needed for --ref-text ICL mode)
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
| `--text-file <path>` | - | Long text file: auto-split into sentences, synthesized per chunk, joined with 30ms crossfade (deterministic per-chunk seeds when `--seed` given) |
| `--ref-audio <wav>` | - | **Required** - reference WAV for voice cloning (any sample rate: auto-resampled to 24kHz mono; 3-10s clean speech) |
| `--ref-text <text>` | - | Reference transcript → ICL mode: prosody (pauses/intonation/rate) follows the reference. **Must be accurate, punctuation included** |
| `--encoder-weights <file>` | `<exe-dir>/speech_tokenizer/model.safetensors` | Codec encoder weights, only needed for ICL mode |
| `--trim-silence <secs>` | 0 (off) | Compress internal/trailing silences longer than this to the given length (0.25 recommended) |
| `--check` | - | Self-test and exit: model/tokenizer files (catches undownloaded git-lfs pointers), ICL sidecar, CPU/RAM |
| `--check <wav>` | - | Above plus reference-audio analysis (sample rate, duration ≥3s, level, silence ratio) |
| `--save-voice <file>` | - | Save voice profile (embedding + ICL codes + audio hash, ~13KB) for reuse |
| `--load-voice <file>` | - | Reuse profile: skips speaker/codec encoder runs (ICL+11s ref: 79s → 36s wall). Falls back to recompute on audio/ref_text mismatch |
| `--language <name>` | english | Target language |
| `--output <path>` | output.wav | Output WAV path |
| `--max-codes <n>` | 500 | Max code timesteps (~n/12.5 seconds) |
| `--temperature <f>` | 0.8 | Sampling temperature |
| `--top-k <n>` | 50 | Top-K sampling |
| `--seed <n>` | random | Random seed for reproducibility (printed each run) |
| `--top-p <f>` | 1.0 (off) | Nucleus sampling: keep smallest set with cumulative mass ≥ p (cuts sampling tail; onset-roughness tool) |
| `--onset-frames <n>` / `--onset-temperature <f>` | 0 / 0.3 | First n frames of each chunk sample at the given temperature (chunk-attack guard; off by default) |

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

## Benchmark (Intel Core Ultra 7 155H, 22 threads, 32GB RAM, CPU-only)

Measured with `--seed` fixed (reproducible). Time scales with **frames**, not text
length — sampling variance changes EOS timing run to run.

| Text | Frames | Audio | Total | Notes |
|------|--------|-------|-------|-------|
| 你好 (chinese) | 8 | 0.6s | 20.5s | x-vector |
| 你好初次见面我是Nori (chinese) | 35 | 2.8s | ~78s | x-vector, no commas |
| 你好,初次见面,我是Nori (chinese) | 45 | 3.6s | 95.5s | x-vector, commas add pauses |
| 今天天气真好 (chinese) | 27 | 2.2s | ~82s | ICL (ref 35 frames) |
| Good morning, how are you today? I hope you are well. (english) | 47 | 3.8s | 104.3s | x-vector |
| same (english) | 33 | 2.6s | 104.4s | ICL (ref 114 frames): fewer frames, tighter pauses |

Per-phase rates on this machine:

| Phase | Rate |
|-------|------|
| Model load | ~1.0s (1511 MB) |
| Voice extraction | ~1-2s (scales with ref length) |
| Prefill | ~3s, ~12s with ICL block (142 positions) |
| Generation | **~1.8s/frame** (talker, single-threaded) |
| Decode | **~0.4s/frame** (VQ + Vocos) |
| ICL encode (one-time) | ~8s (codec encoder from sidecar weights) |

Rule of thumb: `Total ≈ 5s + frames × 2.2s (+ 8s ICL)`, with 1 frame = 80ms audio.
Cap runaway runs with `--max-codes` (e.g. 100 ≈ 8s audio max).

## Speedup notes (same machine, baseline: `--ref-audio voice/dpsng9.wav --text "你好初次见面我叫nori" --language chinese --seed 1790574344731051634`, 34 frames)

`perf` showed 98% of generation cycles in scalar `gemv_q4_inner`; the code
predictor's 15-step loop (≈1M-MAC GEMVs) ran single-threaded under the old 4M
threading threshold.

| Build | Generation | Total | vs baseline |
|-------|-----------|-------|-------------|
| generic release | 60.4s (1.78s/frame) | 77.4s | — |
| `RUSTFLAGS="-C target-cpu=native"` | 56.0s | 73.6s | -5% total |
| native + persistent GEMV thread pool | 41.0s (1.21s/frame) | 58.1s | -25% total |
| native + pool + hand-written AVX2 Q4 kernel | 25.9s (0.76s/frame) | 42.1s | -46% total |
| + AVX2 causal-conv kernel (decode) | 25.7s | 32.7s (decode 13.8s → 4.2s) | -58% total |
| + GEMV overdecomposition (4x chunks) | 19.1s (0.56s/frame) | 25.6s | -67% total |
| + integer-arithmetic Q4 LUT | **12.0s (0.35s/frame)** | **17.7s** | **-77% total** |

Bit-identical chain: the first five builds produce identical audio for the same
seed. Overdecomposition changed FP summation order, so bit-identity is no
longer guaranteed in general — this run happened to be bit-identical
(corr=1.0, max abs diff 0.0); treat tolerance + listening as the gate now.

### Hybrid-CPU note (Intel Ultra 7 155H: 6P + 8E + 2LP-E)

The pool splits work evenly, so join barriers wait for 2.5GHz LP-E cores while
22 active cores also drag P-core clocks down. Pinning to P-cores only:

```bash
taskset -c 0-11 ./target/release/qora-tts --ref-audio ...  # gen 19.1s → 10.8s, total 25.6s → 18.3s
```

(`QORA_THREADS=N` overrides pool size portably, but only `taskset`/affinity
keeps threads off E-cores. Note: decode convs scale with thread count, so
P-only pinning speeds generation but slows decode 4.2s → 6.4s.)

### FMA evaluation (measured, kept opt-in)

`QORA_FMA=1` fuses conv `out += x * w` into one rounding (tolerance-tested:
worst abs 9.5e-7). Result on this machine: **decode 4.3s → 4.3s, no gain**
(the loop is stream-bound; one fewer ALU op changes nothing), while output
sha changes (max 1 int16 LSB). Verdict: default stays bit-exact mul+add;
FMA kept only as an option for other microarchitectures. The Q4 kernel has
no FMA opportunity at all (LUT build is pure mul, accumulation pure add).
The pool (22 persistent workers, no per-call spawn) also lowered the threading
threshold so predictor GEMVs parallelize. The AVX2 kernel (`simd::gemv_q4_avx2`,
8-wide LUT via split `permutevar8x32`+blend) is runtime-dispatched below AVX-512;
differential tests assert bit-exact equality with the scalar oracle.

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
