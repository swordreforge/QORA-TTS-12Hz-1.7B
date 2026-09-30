//! Rule-based Japanese text normalization for TTS (numbers and units).
//!
//! Same v1 scope as the Chinese module — numeric tokens only, readings in
//! katakana, everything else passes through byte-identical:
//! - integers → Sino-Japanese grouped reading (4-digit 万/億/兆 grouping,
//!   same skeleton as Chinese; NO zero serialization: 101 → ヒャクイチ)
//! - euphony: 300 サンビャク / 600 ロッピャク / 800 ハッピャク /
//!   3000 サンゼン / 8000 ハッセン; 100→ヒャク, 1000→セン (drop イチ),
//!   but 一万 keeps イチ (イチマン); 4→ヨン, 7→ナナ, 9→キュウ
//! - decimals → テン (ヨンテンナナハチ), digits one by one, 0 → ゼロ
//! - trailing %/％ → append パーセント (unlike Chinese prefix!)
//! - 4 digits + 年 → regular reading + ネン (2020年 → ニセンニジュウネン)
//! - N + 月 (1-12) → month table (4月 シガツ, 7月 シチガツ, 9月 クガツ)
//! - N + 日 (1-31) → day table (1日 ツイタチ, 20日 ハツカ)
//! - N + ヶ/ケ/か + 月 → duration (1ヶ月 イッカゲツ, else X+カゲツ)
//! - N~M / N～M / N-M (digits both sides) → NからM
//! - leading -/－ before digits → マイナス
//! - ° → ド, ×/*/÷ → カケル/カケル/ワル
//! Out of scope (need semantic context): counters (つ/人/個…,
//! native kun readings), time (時/分/秒), weekday disambiguation.
//! ASCII digits only (same scope as the Chinese module).

const D: [&str; 10] = [
    "ゼロ", "イチ", "ニ", "サン", "ヨン", "ゴ", "ロク", "ナナ", "ハチ", "キュウ",
];

const MONTH: [&str; 12] = [
    "イチガツ", "ニガツ", "サンガツ", "シガツ", "サツガツ", "ロクガツ",
    "シチガツ", "ハチガツ", "クガツ", "ジュウガツ", "ジュウイチガツ", "ジュウニガツ",
];

const DAY: [&str; 31] = [
    "ツイタチ", "フツカ", "ミッカ", "ヨッカ", "イツカ", "ムイカ",
    "ナノカ", "ヨウカ", "ココノカ", "トオカ",
    "ジュウイチニチ", "ジュウニニチ", "ジュウサンニチ", "ジュウヨッカ",
    "ジュウゴニチ", "ジュウロクニチ", "ジュウナナニチ", "ジュウハチニチ",
    "ジュウクニチ", "ハツカ",
    "ニジュウイチニチ", "ニジュウニニチ", "ニジュウサンニチ", "ニジュウヨッカ",
    "ニジュウゴニチ", "ニジュウロクニチ", "ニジュウナナニチ", "ニジュウハチニチ",
    "ニジュウクニチ", "サンジュウニチ", "サンジュウイチニチ",
];

/// Integer reading with 万/億/兆 grouping (0..=10^16-1; larger → digitwise).
/// Zero digits are dropped, never serialized (101 → ヒャクイチ).
pub fn int_ja(n: u64) -> String {
    if n == 0 {
        return "ゼロ".to_string();
    }
    if n >= 10_000_000_000_000_000 {
        return digitwise(&n.to_string());
    }
    let groups = ["", "マン", "オク", "チョウ"];
    let mut parts: Vec<u16> = Vec::new();
    let mut v = n;
    while v > 0 {
        parts.push((v % 10000) as u16);
        v /= 10000;
    }
    let mut out = String::new();
    for (gi, &g) in parts.iter().enumerate().rev() {
        if g == 0 {
            continue; // no zero serialization in Japanese
        }
        out.push_str(&group4(g));
        out.push_str(groups[gi]);
    }
    out
}

/// 4-digit group 1..=9999 with euphonic changes, no group unit.
fn group4(g: u16) -> String {
    debug_assert!((1..=9999).contains(&g));
    let t = (g / 1000) as usize;
    let h = ((g / 100) % 10) as usize;
    let te = ((g / 10) % 10) as usize;
    let o = (g % 10) as usize;
    let mut out = String::new();
    match t {
        0 => {}
        1 => out.push_str("セン"),
        3 => out.push_str("サンゼン"),
        8 => out.push_str("ハッセン"),
        _ => {
            out.push_str(D[t]);
            out.push_str("セン");
        }
    }
    match h {
        0 => {}
        1 => out.push_str("ヒャク"),
        3 => out.push_str("サンビャク"),
        6 => out.push_str("ロッピャク"),
        8 => out.push_str("ハッピャク"),
        _ => {
            out.push_str(D[h]);
            out.push_str("ヒャク");
        }
    }
    match te {
        0 => {}
        1 => out.push_str("ジュウ"),
        _ => {
            out.push_str(D[te]);
            out.push_str("ジュウ");
        }
    }
    if o > 0 {
        out.push_str(D[o]);
    }
    out
}

fn digitwise(s: &str) -> String {
    s.chars().map(|c| D[c.to_digit(10).unwrap() as usize]).collect()
}

/// Normalize one text. Idempotent-ish for already-Japanese prose.
pub fn normalize(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if (c == '-' || c == '－') && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            let prev = if i > 0 { chars[i - 1] } else { ' ' };
            if prev.is_ascii_alphabetic() {
                i += 1; // drop: glued to a model name
                continue;
            }
            if prev.is_ascii_digit() || prev == '.' {
                out.push_str("カラ");
            } else {
                out.push_str("マイナス");
            }
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            let (word, j) = parse_number(&chars, i, false);
            out.push_str(&word);
            i = j;
            continue;
        }
        match c {
            '％' | '%' => out.push_str("パーセント"),
            '°' => out.push_str("ド"),
            '×' | '*' => out.push_str("カケル"),
            '÷' => out.push_str("ワル"),
            '~' | '～' => out.push_str("カラ"),
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

/// Parse a number starting at chars[i] (chars[i] is a digit).
/// Suffix priority after int[.frac][%]: ヶ月 duration → 月 month → 日 day →
/// 年 year. Ranges and letter-adjacent rules mirror the Chinese module.
fn parse_number(chars: &[char], i: usize, force_dw: bool) -> (String, usize) {
    // look back past dropped version hyphens and spaces
    let mut b = i;
    while b > 0 && (chars[b - 1] == '-' || chars[b - 1] == '－' || chars[b - 1] == ' ' || chars[b - 1] == '　') {
        b -= 1;
    }
    let left_letter = force_dw || (b > 0 && chars[b - 1].is_ascii_alphabetic());
    let mut j = i;
    while j < chars.len() && chars[j].is_ascii_digit() {
        j += 1;
    }
    let int_part = chars[i..j].iter().collect::<String>();
    // decimal fraction?
    let mut frac: Option<String> = None;
    if j < chars.len() && chars[j] == '.' && j + 1 < chars.len() && chars[j + 1].is_ascii_digit() {
        let mut k = j + 1;
        while k < chars.len() && chars[k].is_ascii_digit() {
            k += 1;
        }
        frac = Some(chars[j + 1..k].iter().collect());
        j = k;
    }
    // percent suffix (space-tolerant), appended AFTER the number
    let mut k = j;
    while k < chars.len() && (chars[k] == ' ' || chars[k] == '　') {
        k += 1;
    }
    let mut pct = false;
    if k < chars.len() && (chars[k] == '%' || chars[k] == '％') {
        pct = true;
        j = k + 1;
    }

    // unit suffix lookahead (spaces tolerated): ヶ/ケ/か+月, 月, 日, 年
    let mut k = j;
    while k < chars.len() && (chars[k] == ' ' || chars[k] == '　') {
        k += 1;
    }
    // duration Nヶ月 (ヵ/ヶ/ケ/か + 月)
    if k + 1 < chars.len()
        && matches!(chars[k], 'ヶ' | 'ヵ' | 'ケ' | 'か')
        && chars[k + 1] == '月'
        && frac.is_none()
    {
        // 1ヶ月 is irregular (イッカゲツ); else regular/digitwise + カゲツ
        let word = if int_part == "1" && !left_letter {
            "イッカゲツ".to_string()
        } else if left_letter {
            format!("{}カゲツ", digitwise(&int_part))
        } else {
            let reading = int_part
                .parse::<u64>()
                .map(int_ja)
                .unwrap_or_else(|_| digitwise(&int_part));
            format!("{reading}カゲツ")
        };
        return (word, k + 2);
    }
    // month N月 (1-12)
    if k < chars.len() && chars[k] == '月' && frac.is_none() {
        if let Ok(n) = int_part.parse::<usize>() {
            if (1..=12).contains(&n) && !left_letter {
                return (MONTH[n - 1].to_string(), k + 1);
            }
        }
    }
    // day N日 (1-31)
    if k < chars.len() && chars[k] == '日' && frac.is_none() {
        if let Ok(n) = int_part.parse::<usize>() {
            if (1..=31).contains(&n) && !left_letter {
                return (DAY[n - 1].to_string(), k + 1);
            }
        }
    }
    // year: exactly 4 digits + 年 → regular reading + ネン (consumes 年 too,
    // unlike Chinese where 年 emits separately)
    let is_year = int_part.len() == 4 && frac.is_none() && k < chars.len() && chars[k] == '年';
    let mut word = if is_year {
        j = k + 1;
        format!("{}ネン", int_ja(int_part.parse().unwrap_or(0)))
    } else if left_letter {
        digitwise(&int_part)
    } else {
        int_part.parse::<u64>().map(int_ja).unwrap_or_else(|_| digitwise(&int_part))
    };
    if let Some(f) = frac {
        word.push_str("テン");
        word.push_str(&digitwise(&f));
    }
    if pct {
        word.push_str("パーセント");
    }
    // range: sep + digits → カラ + number (dropped when glued to a model name)
    if j < chars.len() && matches!(chars[j], '~' | '～' | '-' | '－' | '–')
        && j + 1 < chars.len()
        && chars[j + 1].is_ascii_digit()
    {
        let (w2, j2) = parse_number(chars, j + 1, left_letter);
        if !left_letter {
            word.push_str("カラ");
        }
        word.push_str(&w2);
        j = j2;
    }
    (word, j)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_ja_basics() {
        assert_eq!(int_ja(0), "ゼロ");
        assert_eq!(int_ja(1), "イチ");
        assert_eq!(int_ja(4), "ヨン");
        assert_eq!(int_ja(7), "ナナ");
        assert_eq!(int_ja(9), "キュウ");
        assert_eq!(int_ja(10), "ジュウ");
        assert_eq!(int_ja(17), "ジュウナナ");
        assert_eq!(int_ja(20), "ニジュウ");
        assert_eq!(int_ja(100), "ヒャク");
        assert_eq!(int_ja(101), "ヒャクイチ");
        assert_eq!(int_ja(300), "サンビャク");
        assert_eq!(int_ja(600), "ロッピャク");
        assert_eq!(int_ja(800), "ハッピャク");
        assert_eq!(int_ja(1000), "セン");
        assert_eq!(int_ja(1010), "センジュウ");
        assert_eq!(int_ja(1001), "センイチ");
        assert_eq!(int_ja(3000), "サンゼン");
        assert_eq!(int_ja(8000), "ハッセン");
    }

    #[test]
    fn test_int_ja_groups() {
        assert_eq!(int_ja(10000), "イチマン");
        assert_eq!(int_ja(10001), "イチマンイチ");
        assert_eq!(int_ja(12000), "イチマンニセン");
        assert_eq!(int_ja(10100), "イチマンヒャク");
        assert_eq!(int_ja(100000), "ジュウマン");
        assert_eq!(int_ja(2020), "ニセンニジュウ");
        assert_eq!(int_ja(100000000), "イチオク");
    }

    #[test]
    fn test_month_day() {
        assert_eq!(normalize("4月"), "シガツ");
        assert_eq!(normalize("7月"), "シチガツ");
        assert_eq!(normalize("9月"), "クガツ");
        assert_eq!(normalize("12月"), "ジュウニガツ");
        assert_eq!(normalize("1日"), "ツイタチ");
        assert_eq!(normalize("20日"), "ハツカ");
        assert_eq!(normalize("15日"), "ジュウゴニチ");
        assert_eq!(normalize("3日"), "ミッカ");
        assert_eq!(normalize("1ヶ月"), "イッカゲツ");
        assert_eq!(normalize("3ヶ月"), "サンカゲツ");
    }

    #[test]
    fn test_normalize_ja_sentences() {
        assert_eq!(normalize("4.78"), "ヨンテンナナハチ");
        assert_eq!(normalize("35%"), "サンジュウゴパーセント");
        assert_eq!(normalize("2020年"), "ニセンニジュウネン");
        assert_eq!(normalize("3~5"), "サンカラゴ");
        assert_eq!(normalize("-40"), "マイナスヨンジュウ");
        assert_eq!(normalize("RTX 4090"), "RTX ヨンゼロキュウゼロ");
        assert_eq!(normalize("Qwen3"), "Qwenサン");
        assert_eq!(normalize("こんにちは"), "こんにちは");
        assert_eq!(normalize("今日は4月1日です"), "今日はシガツツイタチです");
    }
}
