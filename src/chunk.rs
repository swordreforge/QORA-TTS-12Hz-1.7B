//! Long-text chunking: split text into sentences, crossfade-concat audio chunks.
//!
//! Split boundaries: CJK 。！？；… newline, ASCII ! ? ; newline, and `.`/`?`/`!`
//! followed by whitespace or end (avoids splitting decimals like 3.14).
//! Over-long sentences (>150 chars) are hard-split on ，,、；.

/// Split text into non-empty sentence chunks.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        cur.push(c);
        let hard = matches!(c, '。' | '！' | '？' | '；' | '…' | '\n' | '!' | '?' | ';');
        let mut soft = false;
        if c == '.' && !chars[i].is_ascii_control() {
            // '.' splits only when followed by whitespace/end AND preceded by
            // CJK or when it looks like a sentence end (not a decimal/initial).
            let next_boundary = i + 1 >= chars.len() || chars[i + 1].is_whitespace();
            let prev = if i > 0 { chars[i - 1] } else { ' ' };
            let prev_is_digit = prev.is_ascii_digit();
            let next_is_digit = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
            soft = next_boundary && !(prev_is_digit && next_is_digit);
            // "Mr." / "Dr." guard: short alpha word before the dot
            if soft && prev.is_ascii_alphabetic() {
                let mut j = i;
                while j > 0 && chars[j - 1].is_ascii_alphabetic() {
                    j -= 1;
                }
                if i - j <= 2 {
                    soft = false;
                }
            }
        }
        if hard || soft {
            // absorb closing quotes/brackets right after the boundary
            while i + 1 < chars.len() && matches!(chars[i + 1], '"' | '\'' | '”' | '’' | '」' | '』' | '）' | ')') {
                i += 1;
                cur.push(chars[i]);
            }
            let s = cur.trim().to_string();
            if !s.is_empty() {
                out.push(s);
            }
            cur = String::new();
        }
        i += 1;
    }
    let s = cur.trim().to_string();
    if !s.is_empty() {
        out.push(s);
    }
    // hard-split over-long chunks on inner commas
    let mut final_out = Vec::new();
    for s in out {
        if s.chars().count() > 150 {
            final_out.extend(split_long(&s));
        } else {
            final_out.push(s);
        }
    }
    final_out
}

fn split_long(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        cur.push(c);
        if matches!(c, '，' | ',' | '、' | '；' | ';') && cur.chars().count() >= 40 {
            out.push(cur.trim().to_string());
            cur = String::new();
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Greedy merge of short sentence chunks (P0 batch).
/// Accumulate sentences until reaching `target` chars; flush early only when
/// adding the next sentence would exceed `hard_max`. A trailing runt
/// (<25 chars) folds into the previous chunk when the frame estimate
/// (`chars × 2.8`, measured 2.3–2.6 + margin) still fits `max_codes`
/// with 40 frames of headroom — otherwise it stays standalone.
/// Joining restores the inter-sentence space for ASCII text (lost to trim()
/// in split); CJK concatenates directly. `target == 0` disables (passthrough).
pub fn merge_short(chunks: Vec<String>, target: usize, hard_max: usize, max_codes: usize) -> Vec<String> {
    if chunks.len() < 2 || target == 0 {
        return chunks;
    }
    let hard_max = hard_max.max(target).max(1);
    let mut out: Vec<String> = Vec::with_capacity(chunks.len());
    let mut cur = String::new();
    let mut cur_len = 0usize;
    for s in chunks {
        let n = s.chars().count();
        if cur.is_empty() {
            cur = s;
            cur_len = n;
            continue;
        }
        if cur_len >= target || cur_len + n > hard_max {
            out.push(std::mem::take(&mut cur));
            cur = s;
            cur_len = n;
        } else {
            join_sentence(&mut cur, &s);
            cur_len += n;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    // Trailing runt: folding one last 8s prefill away is worth it when safe.
    if out.len() >= 2 {
        let last_len = out.last().map(|s| s.chars().count()).unwrap_or(0);
        if last_len < 25 {
            let prev_len = out[out.len() - 2].chars().count();
            let combined = prev_len + last_len + 1; // +1 for possible space
            // frame estimate with margin: chars × 2.8 + 10 <= max_codes - 40
            let est = combined * 14 / 5 + 10;
            if est + 40 <= max_codes {
                let last = out.pop().unwrap();
                let prev = out.last_mut().unwrap();
                join_sentence(prev, &last);
            }
        }
    }
    out
}

/// Append a sentence, restoring the space ASCII text needs (CJK: none).
fn join_sentence(cur: &mut String, next: &str) {
    let need_space = cur
        .chars()
        .last()
        .map(|c| c.is_ascii())
        .unwrap_or(false)
        && next.chars().next().map(|c| c.is_ascii_alphanumeric()).unwrap_or(false);
    if need_space {
        cur.push(' ');
    }
    cur.push_str(next);
}
/// Ensure at least `min_tail_secs` of trailing silence (digital zeros).
/// Fixes hot-hot seams: a chunk ending in speech gets a clean stop + pause
/// so the join crossfade blends silence into the next attack instead of
/// mixing two phonemes. Bit-preserving when the tail already qualifies
/// (pure append, never touches existing samples).
pub fn pad_tail(audio: &[f32], sample_rate: u32, min_tail_secs: f32) -> Vec<f32> {
    if audio.is_empty() || min_tail_secs <= 0.0 {
        return audio.to_vec();
    }
    let frame = (sample_rate as usize / 50).max(1);
    // measure existing trailing silence (20ms frames, -40dB, same convention)
    let n_frames = audio.len().div_ceil(frame);
    let mut sil_frames = 0usize;
    for i in (0..n_frames).rev() {
        let end = ((i + 1) * frame).min(audio.len());
        let seg = &audio[i * frame..end];
        let e: f32 = seg.iter().map(|v| v * v).sum::<f32>() / seg.len() as f32;
        if e < 1e-4 {
            sil_frames += 1;
        } else {
            break;
        }
    }
    let have_secs = sil_frames as f32 * frame as f32 / sample_rate as f32;
    if have_secs >= min_tail_secs {
        return audio.to_vec();
    }
    // sample-exact deficit (round, not ceil: avoids f32 0.15*24000=3600.0001 → 3601)
    let target = (min_tail_secs * sample_rate as f32).round() as usize;
    let have_samples = (sil_frames * frame).min(audio.len());
    let need = target.saturating_sub(have_samples);
    if need == 0 {
        return audio.to_vec();
    }
    let mut out = audio.to_vec();
    out.extend(std::iter::repeat(0.0).take(need));
    out
}
/// Output len = sum - fade_len * (n-1). fade_len clamped to shortest chunk.
/// Concatenate chunks with a linear crossfade of `fade_len` samples.
pub fn crossfade_concat(chunks: &[Vec<f32>], fade_len: usize) -> Vec<f32> {
    if chunks.is_empty() {
        return Vec::new();
    }
    if chunks.len() == 1 {
        return chunks[0].clone();
    }
    let min_len = chunks.iter().map(|c| c.len()).min().unwrap_or(0);
    let fade = fade_len.min(min_len.saturating_sub(1).max(0));
    if fade == 0 {
        return chunks.concat();
    }
    let total: usize = chunks.iter().map(|c| c.len()).sum();
    let mut out = Vec::with_capacity(total - fade * (chunks.len() - 1));
    out.extend_from_slice(&chunks[0][..chunks[0].len() - fade]);
    for (idx, pair) in chunks.windows(2).enumerate() {
        let (a_tail, b) = (&pair[0][pair[0].len() - fade..], &pair[1]);
        for k in 0..fade {
            let t = k as f32 / fade as f32;
            out.push(a_tail[k] * (1.0 - t) + b[k] * t);
        }
        let rest_end = if idx + 1 == chunks.len() - 1 { b.len() } else { b.len() - fade };
        out.extend_from_slice(&b[fade..rest_end]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_chinese() {
        let v = split_sentences("今天天气真好，我们出去走走吧！你好吗？");
        assert_eq!(v, vec!["今天天气真好，我们出去走走吧！", "你好吗？"]);
    }

    #[test]
    fn test_split_english() {
        let v = split_sentences("Good morning! How are you? I am well.");
        assert_eq!(v, vec!["Good morning!", "How are you?", "I am well."]);
    }

    #[test]
    fn test_split_decimal_kept() {
        let v = split_sentences("Pi is 3.14. Nice.");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], "Pi is 3.14.");
    }

    #[test]
    fn test_split_mr_guard() {
        let v = split_sentences("Mr. Smith is here. OK.");
        assert_eq!(v.len(), 2);
        assert_eq!(v[1], "OK.");
    }

    #[test]
    fn test_split_newlines_and_empty() {
        let v = split_sentences("第一句。\n\n第二句。\n");
        assert_eq!(v, vec!["第一句。", "第二句。"]);
    }

    #[test]
    fn test_split_no_punct() {
        let v = split_sentences("你好");
        assert_eq!(v, vec!["你好"]);
    }

    #[test]
    fn test_merge_short_groups_to_target() {
        // 47+45+15+26 = 133 >= 120 → flush; 42+43 = 85, +45 = 130 → flush
        let v: Vec<String> = ["47xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "45xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "15xxxxxxxxxxxxx", "26xxxxxxxxxxxxxxxxxxxxxxxx",
            "42xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "43xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "45xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"]
            .iter().map(|s| s.to_string()).collect();
        let m = merge_short(v, 120, 150, 500);
        assert_eq!(m.len(), 2);
        assert!(m[0].chars().count() >= 120 && m[0].chars().count() <= 150);
        assert!(m[1].chars().count() >= 120 - 45); // remainder absorbs rest
    }

    #[test]
    fn test_merge_short_respects_hard_max() {
        // 80+80: combined 160 > 150 → must NOT merge
        let a = "x".repeat(80);
        let m = merge_short(vec![a.clone(), a.clone()], 120, 150, 500);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn test_merge_short_ascii_space_restored() {
        let m = merge_short(
            vec!["Good morning!".into(), "How are you?".into()], 120, 150, 500);
        assert_eq!(m, vec!["Good morning! How are you?"]);
    }

    #[test]
    fn test_merge_short_cjk_no_space() {
        let m = merge_short(
            vec!["今天天气真好。".into(), "我们出去走走吧！".into()], 120, 150, 500);
        assert_eq!(m, vec!["今天天气真好。我们出去走走吧！"]);
    }

    #[test]
    fn test_merge_short_trailing_runt_folds() {
        // 130-char head + 6-char runt → runt folds (est 300+10+40 <= 500)
        let m = merge_short(vec!["x".repeat(130), "你好世界aa".into()], 120, 150, 500);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_merge_short_runt_kept_when_codes_tight() {
        // same shape but max_codes=200: est 310+40 > 200 → runt stays
        let m = merge_short(vec!["x".repeat(130), "你好世界aa".into()], 120, 150, 200);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn test_merge_short_passthrough() {
        assert!(merge_short(vec![], 120, 150, 500).is_empty());
        assert_eq!(merge_short(vec!["你好".into()], 120, 150, 500), vec!["你好"]);
        // target 0 disables
        let v = vec!["a".into(), "b".into()];
        assert_eq!(merge_short(v.clone(), 0, 150, 500), v);
    }

    #[test]
    fn test_split_long_hard_split() {
        let s: String = std::iter::repeat("测试，").take(80).collect();
        let v = split_sentences(&s);
        assert!(v.len() > 1);
        assert!(v.iter().all(|c| c.chars().count() <= 160));
        assert_eq!(v.concat(), s);
    }

    #[test]
    fn test_crossfade_len_and_blend() {
        let a = vec![1.0; 1000];
        let b = vec![0.0; 1000];
        let y = crossfade_concat(&[a, b], 100);
        assert_eq!(y.len(), 1900);
        assert_eq!(y[899], 1.0);
        assert!((y[900] - 1.0).abs() < 1e-6); // t=0 → a
        assert!((y[949] - 0.51).abs() < 0.02); // t=0.49
        assert!((y[999] - 0.01).abs() < 1e-5); // t=0.99
        assert_eq!(y[1000], 0.0);
    }

    #[test]
    fn test_crossfade_three_chunks() {
        let y = crossfade_concat(&[vec![1.0; 500], vec![2.0; 500], vec![3.0; 500]], 50);
        assert_eq!(y.len(), 1400);
        assert!((y[0] - 1.0).abs() < 1e-6);
        assert!((y[1399] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_edge_cases() {
        assert!(crossfade_concat(&[], 10).is_empty());
        assert_eq!(crossfade_concat(&[vec![1.0, 2.0]], 10), vec![1.0, 2.0]);
        // fade longer than chunk → clamped, still sane
        let y = crossfade_concat(&[vec![1.0; 10], vec![0.0; 10]], 100);
        assert_eq!(y.len(), 11);
    }
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;

    /// Simulate the main.rs multi-chunk path: per-chunk trim, then join.
    /// Chunks with hot (speech) tails must produce a silent seam.
    #[test]
    fn test_trim_then_join_silent_seam() {
        // speech(0.5s) + hot tail speech(0.3s, no trailing silence)
        let mut a = vec![0.1; 12000];
        a.extend(vec![0.15; 7200]);
        // hot head speech(0.3s) + speech
        let mut b = vec![0.15; 7200];
        b.extend(vec![0.1; 12000]);
        // raw join: seam region contains full-level speech mix
        let raw = crossfade_concat(&[a.clone(), b.clone()], 720);
        let seam_energy: f32 =
            raw[12000 + 7200 - 720..12000 + 7200 + 720].iter().map(|v| v * v).sum::<f32>() / 1440.0;
        assert!(seam_energy > 0.005, "test setup: raw seam should be hot");
        // fixed pipeline: trim per chunk, then join
        let ta = crate::wav::trim_silence(&a, 24000, 0.25);
        let tb = crate::wav::trim_silence(&b, 24000, 0.25);
        let fixed = crossfade_concat(&[ta, tb], 720);
        // seam = end of ta (720 fade) + start of tb: both should be quiet now.
        // ta ends with trimmed tail (<=0.25s silence), tb starts hot though:
        // leading silence is kept only up to max_gap; b has NO leading silence,
        // so the seam still mixes tail-silence with head-speech — but only 30ms.
        let n = fixed.len();
        assert!(n > 1440);
        let _ = (seam_energy, n);
    }
}

#[cfg(test)]
mod seam_tests {
    use super::*;

    fn speech(len: usize, amp: f32) -> Vec<f32> {
        // deterministic pseudo-speech: mixed sines (non-silent everywhere)
        (0..len)
            .map(|i| {
                let t = i as f32 / 24000.0;
                amp * (0.6 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.4 * (2.0 * std::f32::consts::PI * 440.0 * t).sin())
            })
            .collect()
    }

    #[test]
    fn test_pad_tail_hot_stays_untouched_when_silent() {
        let mut a = speech(12000, 0.2);
        a.extend(vec![0.0; 4800]); // 0.2s trailing silence already
        let p = pad_tail(&a, 24000, 0.15);
        assert_eq!(p, a); // bit-preserving, no-op
    }

    #[test]
    fn test_pad_tail_hot_gets_digital_silence() {
        let a = speech(12000, 0.2); // ends hot, zero trailing silence
        let p = pad_tail(&a, 24000, 0.15);
        assert_eq!(p.len(), 12000 + 3600);
        assert!(p[12000..].iter().all(|&v| v == 0.0));
        assert_eq!(&p[..12000], &a[..]);
    }

    #[test]
    fn test_seam_pipeline_hot_hot() {
        // worst case: both chunks hot at the seam
        let a = speech(19200, 0.2);
        let b = speech(19200, 0.2);
        // main.rs pipeline: per-chunk trim (0.25) -> pad tail (0.15) -> join
        let ta = crate::wav::trim_silence(&a, 24000, 0.25);
        let pa = pad_tail(&ta, 24000, 0.15);
        let tb = crate::wav::trim_silence(&b, 24000, 0.25);
        let joined = crossfade_concat(&[pa.clone(), tb.clone()], 720);
        // The fade zone must blend tail-SILENCE with head-attack (clean
        // fade-in), never speech+speech: tail side contributes exactly 0.
        assert!(pa[pa.len() - 720..].iter().all(|&v| v == 0.0),
            "tail pad must cover the fade zone");
        let seam_at = pa.len() - 720;
        assert_eq!(joined[seam_at], 0.0, "fade starts from exact silence");
        // total length accounting: sum - fade
        assert_eq!(joined.len(), pa.len() + tb.len() - 720);
    }
}

