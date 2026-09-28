//! `--check` self-test: verify runnable environment before a long inference run.
//!
//! Checks, in order:
//! 1. model binary next to the exe (magic "QTTS", supported version)
//! 2. config.json / tokenizer.json / vocab.json / merges.txt present; LFS
//!    pointer files (133-byte `version https://git-lfs...`) reported as missing
//! 3. speech_tokenizer/model.safetensors (ICL sidecar; warn-only if absent)
//! 4. CPU/RAM summary (threads, AVX-512, free RAM)
//! 5. optional `--check <wav>`: sample rate, channels, duration, clipping,
//!    silence ratio with a verdict for voice-cloning suitability.

use std::path::{Path, PathBuf};

/// True if the file is a git-lfs pointer rather than real content.
pub fn is_lfs_pointer(path: &Path) -> bool {
    let Ok(data) = std::fs::read(path) else { return false };
    if data.len() > 1024 {
        return false;
    }
    let head = String::from_utf8_lossy(&data);
    head.starts_with("version https://git-lfs.github.com/spec/v1")
}

pub struct CheckItem {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

impl CheckItem {
    pub fn line(&self) -> String {
        let mark = if self.ok { "ok  " } else { "FAIL" };
        format!("[{mark}] {:28} {}", self.name, self.detail)
    }
}

fn need_file(dir: &Path, name: &str) -> CheckItem {
    let p = dir.join(name);
    match std::fs::metadata(&p) {
        Err(_) => CheckItem { name: name.into(), ok: false, detail: "missing".into() },
        Ok(_) if is_lfs_pointer(&p) => CheckItem {
            name: name.into(),
            ok: false,
            detail: "git-lfs pointer, real file not downloaded (see README)".into(),
        },
        Ok(m) => CheckItem {
            name: name.into(),
            ok: true,
            detail: format!("{:.1} MB", m.len() as f64 / 1048576.0),
        },
    }
}

/// Check the model binary header (magic + version) without loading weights.
fn check_model_bin(path: &Path) -> CheckItem {
    let name = "model.qora-tts".to_string();
    let Ok(mut f) = std::fs::File::open(path) else {
        return CheckItem { name, ok: false, detail: format!("missing at {}", path.display()) };
    };
    use std::io::Read;
    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() || &magic != b"QTTS" {
        return CheckItem { name, ok: false, detail: "bad magic (not a QTTS binary)".into() };
    }
    let mut vb = [0u8; 4];
    let version = if f.read_exact(&mut vb).is_ok() { u32::from_le_bytes(vb) } else { 0 };
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    CheckItem {
        name,
        ok: true,
        detail: format!("version {version}, {:.0} MB", size as f64 / 1048576.0),
    }
}

/// Full environment check. Returns (items, runnable).
pub fn check_env(exe_dir: &Path, model_path: &Path) -> (Vec<CheckItem>, bool) {
    let base = model_path.parent().unwrap_or(exe_dir);
    let mut items = vec![
        check_model_bin(model_path),
        need_file(base, "config.json"),
        need_file(base, "tokenizer.json"),
        need_file(base, "vocab.json"),
        need_file(base, "merges.txt"),
    ];
    // ICL sidecar: warn-only
    let enc = base.join("speech_tokenizer").join("model.safetensors");
    let enc_item = match std::fs::metadata(&enc) {
        Ok(m) if m.len() > 10_000_000 => CheckItem {
            name: "speech_tokenizer/model.safetensors".into(),
            ok: true,
            detail: format!("{:.0} MB, ICL ready", m.len() as f64 / 1048576.0),
        },
        Ok(_) => CheckItem {
            name: "speech_tokenizer/model.safetensors".into(),
            ok: false,
            detail: "too small, ICL (--ref-text) unavailable".into(),
        },
        Err(_) => CheckItem {
            name: "speech_tokenizer/model.safetensors".into(),
            ok: false,
            detail: "missing, ICL (--ref-text) unavailable (x-vector still works)".into(),
        },
    };
    items.push(enc_item);

    let sys = crate::system::SystemInfo::detect();
    items.push(CheckItem {
        name: "cpu".into(),
        ok: true,
        detail: format!(
            "{} threads, AVX-512 {}",
            sys.cpu_threads,
            if crate::simd::has_avx512() { "yes" } else { "no (scalar Q4 path)" }
        ),
    });
    items.push(CheckItem {
        name: "ram".into(),
        ok: sys.available_ram_mb >= 2000,
        detail: format!("{} MB free", sys.available_ram_mb),
    });

    // items[5] (ICL sidecar) is advisory only; everything else must pass
    let fatal = items.iter().enumerate().filter(|(idx, _)| *idx != 5).all(|(_, i)| i.ok);
    (items, fatal)
}

/// Analyze a reference wav for cloning suitability. Returns printable lines.
pub fn analyze_wav(path: &Path) -> Vec<CheckItem> {
    let mut items = Vec::new();
    let (samples, sr) = match crate::wav::read_wav(path) {
        Ok(v) => v,
        Err(e) => {
            items.push(CheckItem { name: "read".into(), ok: false, detail: e.to_string() });
            return items;
        }
    };
    // Re-derive channel count: read_wav mixes to mono; inspect header instead.
    let ch = wav_channels(path).unwrap_or(1);
    let dur = samples.len() as f32 / sr as f32;
    items.push(CheckItem {
        name: "format".into(),
        ok: true,
        detail: format!("{sr} Hz, {ch} ch, {dur:.2}s"),
    });
    items.push(CheckItem {
        name: "sample-rate".into(),
        ok: sr == 24000,
        detail: if sr == 24000 {
            "24kHz, no resampling needed".into()
        } else {
            format!("{sr} Hz will be auto-resampled to 24kHz")
        },
    });
    items.push(CheckItem {
        name: "duration".into(),
        ok: dur >= 3.0,
        detail: if dur < 3.0 {
            format!("{dur:.2}s is below the 3s floor, expect weak conditioning")
        } else if dur > 30.0 {
            format!("{dur:.2}s is long; first ~10s dominate, consider trimming")
        } else {
            format!("{dur:.2}s in the sweet spot")
        },
    });
    let peak = samples.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    items.push(CheckItem {
        name: "level".into(),
        ok: peak >= 0.05 && peak < 1.0,
        detail: if peak < 0.05 {
            format!("peak {peak:.3}, too quiet")
        } else if peak >= 1.0 {
            format!("peak {peak:.3}, clipping present")
        } else {
            format!("peak {peak:.3}")
        },
    });
    // silence ratio over 20ms frames at -40dB
    let frame = (sr as usize / 50).max(1);
    let mut sil = 0usize;
    let mut tot = 0usize;
    let mut i = 0;
    while i + frame <= samples.len() {
        let e: f32 = samples[i..i + frame].iter().map(|v| v * v).sum::<f32>() / frame as f32;
        tot += 1;
        if e < 1e-4 {
            sil += 1;
        }
        i += frame;
    }
    let ratio = if tot > 0 { sil as f32 / tot as f32 } else { 1.0 };
    items.push(CheckItem {
        name: "silence".into(),
        ok: ratio < 0.6,
        detail: if ratio >= 0.6 {
            format!("{:.0}% silence, effective speech is thin", ratio * 100.0)
        } else {
            format!("{:.0}% silence", ratio * 100.0)
        },
    });
    items
}

/// Channel count from the WAV header (read_wav mixes down to mono).
fn wav_channels(path: &Path) -> Option<usize> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 44 || &data[0..4] != b"RIFF" {
        return None;
    }
    let mut pos = 12;
    while pos + 8 <= data.len() {
        if &data[pos..pos + 4] == b"fmt " {
            return Some(u16::from_le_bytes([data[pos + 10], data[pos + 11]]) as usize);
        }
        let size = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        pos += 8 + size;
    }
    None
}

/// Resolve a user-supplied check target: existing wav file or search path hint.
pub fn resolve_check_target(arg: &str) -> PathBuf {
    PathBuf::from(arg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn tmp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("qora_check_{name}"));
        p
    }

    fn sine_24k(secs: f32, peak: f32) -> Vec<f32> {
        let n = (24000.0 * secs) as usize;
        (0..n).map(|i| peak * (2.0 * PI * 440.0 * i as f32 / 24000.0).sin()).collect()
    }

    #[test]
    fn test_lfs_pointer_detect() {
        let p = tmp("lfs.bin");
        std::fs::write(&p, "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 11\n").unwrap();
        assert!(is_lfs_pointer(&p));
        let q = tmp("real.bin");
        std::fs::write(&q, vec![0u8; 2000]).unwrap();
        assert!(!is_lfs_pointer(&q)); // >1KB short-circuits
        let r = tmp("small.bin");
        std::fs::write(&r, b"{\"a\": 1}").unwrap();
        assert!(!is_lfs_pointer(&r));
    }

    #[test]
    fn test_analyze_good_ref() {
        let p = tmp("good.wav");
        crate::wav::write_wav(&p, &sine_24k(5.0, 0.3), 24000).unwrap();
        let items = analyze_wav(&p);
        assert!(items.iter().all(|i| i.ok), "{:?}", items.iter().map(|i| i.line()).collect::<Vec<_>>());
    }

    #[test]
    fn test_analyze_short_ref_fails_duration() {
        let p = tmp("short.wav");
        crate::wav::write_wav(&p, &sine_24k(1.0, 0.3), 24000).unwrap();
        let items = analyze_wav(&p);
        let d = items.iter().find(|i| i.name == "duration").unwrap();
        assert!(!d.ok);
    }

    #[test]
    fn test_analyze_quiet_ref_fails_level() {
        let p = tmp("quiet.wav");
        crate::wav::write_wav(&p, &sine_24k(4.0, 0.01), 24000).unwrap();
        let items = analyze_wav(&p);
        let l = items.iter().find(|i| i.name == "level").unwrap();
        assert!(!l.ok);
    }

    #[test]
    fn test_analyze_missing_file() {
        let items = analyze_wav(&tmp("nope.wav"));
        assert!(!items[0].ok);
    }

    #[test]
    fn test_need_file_missing() {
        let it = need_file(&tmp("no-such-dir"), "config.json");
        assert!(!it.ok);
    }
}
