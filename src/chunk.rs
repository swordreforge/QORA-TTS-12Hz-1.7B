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

/// Concatenate chunks with a linear crossfade of `fade_len` samples.
/// Output len = sum - fade_len * (n-1). fade_len clamped to shortest chunk.
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
