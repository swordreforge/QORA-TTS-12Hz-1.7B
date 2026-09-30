//! Voice profile cache ("style file"): reuse text-independent voice
//! conditioning across runs.
//!
//! A profile stores everything derived from the reference AUDIO alone:
//! speaker embedding (2048 f32) + ICL ref codes (16 x Tv u16) + the ref_text
//! they were built with + sha256 of the source audio bytes (freshness).
//! The ICL prefill block itself is NOT cacheable (it mixes in the current
//! text), so it is rebuilt every run from cached codes + new text.
//!
//! Format: magic "QVOI" + u32 version + u64 audio_len + 32B sha256 +
//! ref_text (u32 len + bytes, u32::MAX = none) + embedding (u32 dims + f32s)
//! + codes (u32 groups, u32 frames, u16s; groups=0 = none).
//!
//! Load rules (partial reuse): embedding usable iff audio hash matches;
//! codes usable iff hash matches AND stored ref_text == current --ref-text
//! (both none counts as equal). Anything unusable is recomputed.

use std::io::{Read, Write};
use std::path::Path;

pub const PROFILE_VERSION: u32 = 1;
const MAGIC: &[u8; 4] = b"QVOI";

pub struct VoiceProfile {
    pub audio_len: u64,
    pub audio_sha256: [u8; 32],
    pub ref_text: Option<String>,
    pub embedding: Vec<f32>,
    pub ref_codes: Option<Vec<Vec<u32>>>,
}

/// sha256 of raw file bytes (freshness check, not security).
pub fn hash_bytes(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(data);
    h.finalize().into()
}

fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub fn save_profile(path: &Path, p: &VoiceProfile) -> std::io::Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    w.write_all(MAGIC)?;
    write_u32(&mut w, PROFILE_VERSION)?;
    w.write_all(&p.audio_len.to_le_bytes())?;
    w.write_all(&p.audio_sha256)?;
    match &p.ref_text {
        Some(t) => {
            write_u32(&mut w, t.len() as u32)?;
            w.write_all(t.as_bytes())?;
        }
        None => write_u32(&mut w, u32::MAX)?,
    }
    write_u32(&mut w, p.embedding.len() as u32)?;
    for v in &p.embedding {
        w.write_all(&v.to_le_bytes())?;
    }
    match &p.ref_codes {
        Some(codes) => {
            write_u32(&mut w, codes.len() as u32)?;
            write_u32(&mut w, codes[0].len() as u32)?;
            for q in codes {
                for &c in q {
                    w.write_all(&(c as u16).to_le_bytes())?;
                }
            }
        }
        None => write_u32(&mut w, 0)?,
    }
    w.flush()
}

pub fn load_profile(path: &Path) -> Result<VoiceProfile, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(|e| format!("read: {e}"))?;
    if &magic != MAGIC {
        return Err("bad magic (not a voice profile)".into());
    }
    let version = read_u32(&mut f).map_err(|e| format!("read: {e}"))?;
    if version != PROFILE_VERSION {
        return Err(format!("version {version}, expected {PROFILE_VERSION}"));
    }
    let mut lb = [0u8; 8];
    f.read_exact(&mut lb).map_err(|e| format!("read: {e}"))?;
    let audio_len = u64::from_le_bytes(lb);
    let mut sha = [0u8; 32];
    f.read_exact(&mut sha).map_err(|e| format!("read: {e}"))?;
    let rl = read_u32(&mut f).map_err(|e| format!("read: {e}"))?;
    let ref_text = if rl == u32::MAX {
        None
    } else {
        if rl > 1_000_000 {
            return Err("ref_text too long".into());
        }
        let mut buf = vec![0u8; rl as usize];
        f.read_exact(&mut buf).map_err(|e| format!("read: {e}"))?;
        Some(String::from_utf8(buf).map_err(|_| "ref_text not utf-8".to_string())?)
    };
    let nd = read_u32(&mut f).map_err(|e| format!("read: {e}"))? as usize;
    if nd == 0 || nd > 1_000_000 {
        return Err("bad embedding dims".into());
    }
    let mut embedding = Vec::with_capacity(nd);
    for _ in 0..nd {
        let mut b = [0u8; 4];
        f.read_exact(&mut b).map_err(|e| format!("read: {e}"))?;
        embedding.push(f32::from_le_bytes(b));
    }
    let ng = read_u32(&mut f).map_err(|e| format!("read: {e}"))? as usize;
    let ref_codes = if ng == 0 {
        None
    } else {
        if ng > 64 {
            return Err("bad codebook count".into());
        }
        let nf = read_u32(&mut f).map_err(|e| format!("read: {e}"))? as usize;
        if nf == 0 || nf > 100_000 {
            return Err("bad frame count".into());
        }
        let mut codes = Vec::with_capacity(ng);
        for _ in 0..ng {
            let mut q = Vec::with_capacity(nf);
            for _ in 0..nf {
                let mut b = [0u8; 2];
                f.read_exact(&mut b).map_err(|e| format!("read: {e}"))?;
                q.push(u16::from_le_bytes(b) as u32);
            }
            codes.push(q);
        }
        Some(codes)
    };
    Ok(VoiceProfile { audio_len, audio_sha256: sha, ref_text, embedding, ref_codes })
}

/// Reuse plan for a loaded profile. Pure decision logic (unit-tested):
/// - `audio_hash`: None = no --ref-audio given (trust mode, freshness
///   unverified); Some(h) = verify against stored hash, mismatch → recompute.
/// - `explicit_ref_text`: current --ref-text (already normalized).
/// - effective text = explicit if given, else the profile's stored text.
/// - codes reusable iff present AND stored text == effective text.
pub struct ProfilePlan {
    pub use_embedding: bool,
    pub use_codes: bool,
    pub effective_ref_text: Option<String>,
}

pub fn plan_profile_use(
    profile: Option<&VoiceProfile>,
    audio_hash: Option<[u8; 32]>,
    explicit_ref_text: Option<&str>,
) -> ProfilePlan {
    let prof = match profile {
        Some(p) => p,
        None => {
            return ProfilePlan {
                use_embedding: false,
                use_codes: false,
                effective_ref_text: explicit_ref_text.map(|s| s.to_string()),
            }
        }
    };
    if let Some(h) = audio_hash {
        if h != prof.audio_sha256 {
            return ProfilePlan {
                use_embedding: false,
                use_codes: false,
                effective_ref_text: explicit_ref_text.map(|s| s.to_string()),
            };
        }
    }
    let effective = explicit_ref_text
        .map(|s| s.to_string())
        .or_else(|| prof.ref_text.clone());
    let codes_ok =
        prof.ref_codes.is_some() && prof.ref_text.as_deref() == effective.as_deref();
    ProfilePlan { use_embedding: true, use_codes: codes_ok, effective_ref_text: effective }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample_profile() -> VoiceProfile {
        VoiceProfile {
            audio_len: 12345,
            audio_sha256: hash_bytes(b"fake-audio"),
            ref_text: Some("你好。".into()),
            embedding: vec![0.5, -1.25, 3.0],
            ref_codes: Some(vec![vec![1, 2, 3], vec![4, 5, 6]]),
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("qora_prof_{name}"));
        p
    }

    #[test]
    fn test_roundtrip_full() {
        let p = tmp("full.qvoice");
        save_profile(&p, &sample_profile()).unwrap();
        let q = load_profile(&p).unwrap();
        assert_eq!(q.audio_len, 12345);
        assert_eq!(q.audio_sha256, hash_bytes(b"fake-audio"));
        assert_eq!(q.ref_text, Some("你好。".into()));
        assert_eq!(q.embedding, vec![0.5, -1.25, 3.0]);
        assert_eq!(q.ref_codes, Some(vec![vec![1, 2, 3], vec![4, 5, 6]]));
    }

    #[test]
    fn test_roundtrip_minimal() {
        let p = tmp("min.qvoice");
        let prof = VoiceProfile {
            audio_len: 1,
            audio_sha256: [7u8; 32],
            ref_text: None,
            embedding: vec![1.0],
            ref_codes: None,
        };
        save_profile(&p, &prof).unwrap();
        let q = load_profile(&p).unwrap();
        assert_eq!(q.ref_text, None);
        assert_eq!(q.ref_codes, None);
        assert_eq!(q.embedding, vec![1.0]);
    }

    #[test]
    fn test_bad_magic() {
        let p = tmp("bad.qvoice");
        std::fs::write(&p, b"NOPE1234").unwrap();
        assert!(load_profile(&p).is_err());
    }

    #[test]
    fn test_truncated() {
        let p = tmp("trunc.qvoice");
        save_profile(&p, &sample_profile()).unwrap();
        let mut data = std::fs::read(&p).unwrap();
        data.truncate(data.len() / 2);
        std::fs::write(&p, data).unwrap();
        assert!(load_profile(&p).is_err());
    }

    #[test]
    fn test_hash_stable_and_sensitive() {
        assert_eq!(hash_bytes(b"abc"), hash_bytes(b"abc"));
        assert_ne!(hash_bytes(b"abc"), hash_bytes(b"abd"));
    }
}

#[cfg(test)]
mod plan_tests {
    use super::*;

    fn prof() -> VoiceProfile {
        VoiceProfile {
            audio_len: 10,
            audio_sha256: [9u8; 32],
            ref_text: Some("ref text".into()),
            embedding: vec![1.0, 2.0],
            ref_codes: Some(vec![vec![1u32, 2]]),
        }
    }

    #[test]
    fn test_plan_no_profile() {
        let p = plan_profile_use(None, Some([9u8; 32]), Some("x"));
        assert!(!p.use_embedding && !p.use_codes);
        assert_eq!(p.effective_ref_text, Some("x".into()));
    }

    #[test]
    fn test_plan_hash_mismatch() {
        let pr = prof();
        let p = plan_profile_use(Some(&pr), Some([1u8; 32]), Some("ref text"));
        assert!(!p.use_embedding && !p.use_codes);
    }

    #[test]
    fn test_plan_full_hit_explicit_match() {
        let pr = prof();
        let p = plan_profile_use(Some(&pr), Some([9u8; 32]), Some("ref text"));
        assert!(p.use_embedding && p.use_codes);
        assert_eq!(p.effective_ref_text, Some("ref text".into()));
    }

    #[test]
    fn test_plan_trust_mode_no_audio() {
        // no audio hash: accept profile as-is, effective text from profile
        let pr = prof();
        let p = plan_profile_use(Some(&pr), None, None);
        assert!(p.use_embedding && p.use_codes);
        assert_eq!(p.effective_ref_text, Some("ref text".into()));
    }

    #[test]
    fn test_plan_explicit_differs() {
        // explicit ref_text wins; stored codes no longer match it
        let pr = prof();
        let p = plan_profile_use(Some(&pr), None, Some("other"));
        assert!(p.use_embedding && !p.use_codes);
        assert_eq!(p.effective_ref_text, Some("other".into()));
    }

    #[test]
    fn test_plan_no_codes_in_profile() {
        let mut pr = prof();
        pr.ref_codes = None;
        let p = plan_profile_use(Some(&pr), Some([9u8; 32]), None);
        assert!(p.use_embedding && !p.use_codes);
        assert_eq!(p.effective_ref_text, Some("ref text".into()));
    }
}
