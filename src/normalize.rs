//! Rule-based Chinese text normalization for TTS (numbers, units, code tokens).
//!
//! The small LM mangles raw digits ("RTF为4.78", "2020年", "-40dB").
//! v1 scope is deliberately narrow — numeric tokens only; Latin prose and
//! code pass through untouched:
//! - integers → grouped reading (一百七十一, 四千零九十, 一万二千)
//! - decimals → 点 reading (四点七八), fractional digits read one by one
//! - trailing %/％ → 百分之X (百分之三十五)
//! - 4 digits + 年 → digit-by-digit (二零二零年)
//! - N~M / N～M / N-M (digits both sides) → N到M
//! - leading -/－ before digits → 负 (负四十)
//! - ° → 度, ×/*/÷ → 乘/乘/除以
//! v2 adds code-token stabilization (runs AFTER numbers, so decimal dots
//! are already gone and remaining ASCII dots are code/abbreviation dots):
//! - identifier dots → 点 (torch.no_grad → torch点no grad): kills the
//!   English-period prosody stutter on code and keeps dotted names in one
//!   prosodic unit (code-dense text otherwise wanders into silence defects)
//! - underscores inside words → space (no_grad → no grad)
//! - standalone True/False/None → 真/假/空 (Python booleans read naturally)
//! - sentence dots (followed by space/end/punct) are kept for prosody
//! v3 adds universal cleaning FIRST (all languages via normalize_for_language):
//! - invisible junk deleted: zero-width (U+200B/C/D), BOM (U+FEFF), bidi
//!   controls (U+200E/F, U+202A-2E, U+2060-69), variation selectors
//!   (U+FE00-0F incl. emoji VS16, U+E0000-7F tags, U+E0100-1FD),
//!   keycap joiner (U+20E3), emoji pictographs (U+1F000-1FAFF)
//! - control chars deleted except \n \r \t (which shape chunking/prosody)
//! - horizontal whitespace runs collapsed to one space (space/tab/VT/FF/
//!   NBSP/U+3000; newlines untouched — the splitter owns them)
//! - light markdown: `ticks`→content, line-start #/>/list markers stripped,
//!   --- hr lines dropped, [text](url)→text, paired **/__/~~/*…* stripped
//!   (content kept), | → space
//! Boundaries (documented, not bugs): ZWJ/ZWNJ deletion assumes CJK/EN
//! corpus (Indic scripts use ZWJ orthographically); dingbats/misc-symbols
//! (★☎♥) are KEPT (often meaningful); lowercase true/false untouched.
//! Anything unrecognized passes through byte-identical.
//!
//! Panic-safety contract: this file NEVER slices &str by byte index.
//! Everything runs on Vec<char> with bounds-checked indexing, so CJK
//! (multi-byte UTF-8) cannot split mid-codepoint. Keep it that way.

const D: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];

/// Integer reading with 万/亿/兆 grouping (0..=10^16-1; larger → digitwise).
pub fn int_zh(n: u64) -> String {
    if n == 0 {
        return "零".to_string();
    }
    if n >= 10_000_000_000_000_000 {
        return digitwise(&n.to_string());
    }
    let groups = ["", "万", "亿", "兆"];
    // split into 4-digit groups, low to high
    let mut parts: Vec<u16> = Vec::new();
    let mut v = n;
    while v > 0 {
        parts.push((v % 10000) as u16);
        v /= 10000;
    }
    let mut out = String::new();
    let mut need_zero = false; // a skipped zero-group (or group <1000) forces 零
    for (gi, &g) in parts.iter().enumerate().rev() {
        if g == 0 {
            need_zero = true;
            continue;
        }
        if !out.is_empty() {
            if need_zero || g < 1000 {
                out.push_str("零");
            }
        }
        need_zero = false;
        out.push_str(&group4(g));
        out.push_str(groups[gi]);
    }
    // leading 一十 → 十 ("十", not "一十"; but "一百一十" keeps inner 一十)
    if let Some(rest) = out.strip_prefix("一十") {
        out = format!("十{rest}");
    }
    out
}

/// 4-digit group 1..=9999, no group unit.
fn group4(g: u16) -> String {
    debug_assert!((1..=9999).contains(&g));
    let d = [(g / 1000) as usize, ((g / 100) % 10) as usize, ((g / 10) % 10) as usize, (g % 10) as usize];
    let u = ["千", "百", "十", ""];
    let mut out = String::new();
    let mut started = false;
    let mut zero = false;
    for (i, &x) in d.iter().enumerate() {
        if x == 0 {
            if started {
                zero = true;
            }
            continue;
        }
        if zero {
            out.push_str("零");
            zero = false;
        }
        // inner 一十 → 十? No: 一百一十 keeps 一十. Only strip at very start
        // (handled by caller). "十" alone inside group: 10 → 一十 handled
        // by caller strip; 1010 → 一千零一十? Standard: 一千零一十.
        // Keep 一十 inside groups (一千零一十, not 一千零十 — both heard,
        // former is standard). Simplify: tens digit 1 → 十 without 一?
        // Common speech: 一百一十, 一千零一十. So keep 一.
        out.push_str(D[x]);
        out.push_str(u[i]);
        started = true;
    }
    // group "10" → 一十 (caller strips leading 一十 only at string start)
    out
}

fn digitwise(s: &str) -> String {
    s.chars().map(|c| D[c.to_digit(10).unwrap() as usize]).collect()
}

/// Universal cleaning v3: invisible junk, controls, whitespace, light markdown.
/// Runs FIRST for every language (numbers/code rules see clean text).
/// Char-scan only — no byte slicing anywhere (see module panic contract).
pub fn clean(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    // collapse horizontal whitespace runs (space-like class → one ' ');
    // leading/trailing runs vanish (flush only between content)
    let mut pend_space = false;
    let flush_space = |out: &mut String, pend: &mut bool| {
        if *pend {
            if !out.is_empty() {
                out.push(' ');
            }
            *pend = false;
        }
    };
    while i < chars.len() {
        let c = chars[i];
        // --- markdown line constructs (only at line start) ---
        if i == 0 || chars[i - 1] == '\n' {
            let mut j = i;
            while j < chars.len() && (chars[j] == ' ' || chars[j] == '\t') {
                j += 1;
            }
            // hr: line of only [-*_ \t], ≥3 markers → drop whole line
            let mut k = j;
            let mut marks = 0;
            while k < chars.len() && chars[k] != '\n'
                && matches!(chars[k], '-' | '*' | '_' | ' ' | '\t')
            {
                if matches!(chars[k], '-' | '*' | '_') {
                    marks += 1;
                }
                k += 1;
            }
            if marks >= 3 && (k >= chars.len() || chars[k] == '\n') {
                i = k; // skip to \n (kept next iteration) or end
                pend_space = false;
                continue;
            }
            // headers: 1-6 '#' + space/EOL (one following space eaten)
            if chars[j] == '#' {
                let mut h = j;
                while h < chars.len() && chars[h] == '#' && h - j < 6 {
                    h += 1;
                }
                if h == chars.len() || chars[h] == ' ' || chars[h] == '\t' || chars[h] == '\n' {
                    i = h;
                    if i < chars.len() && (chars[i] == ' ' || chars[i] == '\t') {
                        i += 1;
                    }
                    continue;
                }
            }
            // quote: '>' + space/EOL (one following space eaten)
            if chars[j] == '>'
                && (j + 1 >= chars.len()
                    || chars[j + 1] == ' '
                    || chars[j + 1] == '\t'
                    || chars[j + 1] == '\n')
            {
                i = j + 1;
                if i < chars.len() && (chars[i] == ' ' || chars[i] == '\t') {
                    i += 1;
                }
                continue;
            }
            // list: single -/+/* + space/EOL ("-40dB" has no space → safe)
            if matches!(chars[j], '-' | '+' | '*')
                && (j + 1 >= chars.len()
                    || chars[j + 1] == ' '
                    || chars[j + 1] == '\t'
                    || chars[j + 1] == '\n')
            {
                i = j + 1;
                if i < chars.len() && (chars[i] == ' ' || chars[i] == '\t') {
                    i += 1;
                }
                continue;
            }
        }
        // --- invisible junk: delete ---
        if is_invisible(c) {
            i += 1;
            continue;
        }
        // --- controls (keep \n \r \t) ---
        if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
            i += 1;
            continue;
        }
        // --- horizontal whitespace → collapse ---
        if c == ' ' || c == '\t' || c == '\x0B' || c == '\x0C' || c == '\u{A0}' || c == '\u{3000}' {
            pend_space = true;
            i += 1;
            continue;
        }
        flush_space(&mut out, &mut pend_space);
        // --- backtick: always junk, drop the tick, keep content ---
        if c == '`' {
            i += 1;
            continue;
        }
        // --- pipe: table border / shell pipe → neutral space ---
        if c == '|' {
            out.push(' ');
            i += 1;
            continue;
        }
        // --- [text](url) → text (malformed → all literal) ---
        if c == '[' {
            if let Some((end, inner)) = try_link(&chars, i) {
                flush_space(&mut out, &mut pend_space);
                out.push_str(&inner);
                i = end;
                continue;
            }
        }
        // --- paired markers **/__/~~ (and single *…*, never single _…_) ---
        if (c == '*' || c == '~') || (c == '_' && i + 1 < chars.len() && chars[i + 1] == '_') {
            let marker: &str = if c == '_' {
                "__"
            } else if c == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
                "**"
            } else if c == '~' && i + 1 < chars.len() && chars[i + 1] == '~' {
                "~~"
            } else if c == '*' {
                "*"
            } else {
                ""
            };
            if !marker.is_empty() {
                if let Some((end, inner)) = try_paired(&chars, i, marker) {
                    flush_space(&mut out, &mut pend_space);
                    out.push_str(&inner);
                    i = end;
                    continue;
                }
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Invisible characters deleted by clean(). ZWJ/ZWNJ included by instruction
/// (CJK/EN corpus assumption — Indic scripts use ZWJ orthographically).
/// Dingbats/misc-symbols (★☎♥ U+2600-27BF) deliberately NOT here (kept).
fn is_invisible(c: char) -> bool {
    matches!(c,
        '\u{200B}' | '\u{200C}' | '\u{200D}' | // ZWSP, ZWNJ, ZWJ
        '\u{FEFF}' | // BOM / ZWNBSP
        '\u{200E}' | '\u{200F}' | // bidi marks
        '\u{202A}'..='\u{202E}' | // bidi embeddings/overrides
        '\u{2060}'..='\u{2069}' | // word joiner, invisible ops, bidi isolates
        '\u{FE00}'..='\u{FE0F}' | // variation selectors (incl. VS16)
        '\u{E0000}'..='\u{E007F}' | // tag characters
        '\u{E0100}'..='\u{E01FD}' | // IVS
        '\u{20E3}' // combining enclosing keycap
    ) || ('\u{1F000}'..='\u{1FAFF}').contains(&c) // emoji blocks
}

/// Parse [text](url) at chars[i]=='['. Returns (index past ')', inner text).
/// Newlines/nesting rejected; malformed input → None (emit literally).
fn try_link(chars: &[char], i: usize) -> Option<(usize, String)> {
    let mut j = i + 1;
    while j < chars.len() && chars[j] != ']' && chars[j] != '\n' && chars[j] != '[' {
        j += 1;
    }
    if j >= chars.len() || chars[j] != ']' || j + 1 >= chars.len() || chars[j + 1] != '(' {
        return None;
    }
    let inner: String = chars[i + 1..j].iter().collect();
    if inner.is_empty() || inner.chars().count() > 200 {
        return None;
    }
    let mut k = j + 2;
    while k < chars.len() && chars[k] != ')' && chars[k] != '\n' && chars[k] != ' ' && chars[k] != '\t' {
        k += 1;
    }
    if k >= chars.len() || chars[k] != ')' {
        return None;
    }
    Some((k + 1, inner))
}

/// Strip a paired marker (opener at i). Content rules: non-empty, same line,
/// ≤60 chars, no marker char inside (nearest-close pairing), and must look
/// like prose — contains an ASCII letter or CJK char (so "2**3" / "**2**"
/// fall through to the arithmetic rules instead of being eaten).
/// Single '_' is NEVER paired here (snake_case glue owns it downstream).
fn try_paired(chars: &[char], i: usize, marker: &str) -> Option<(usize, String)> {
    let m: Vec<char> = marker.chars().collect();
    let ml = m.len();
    if chars[i..].len() < ml * 2 + 1 {
        return None;
    }
    // find nearest closer on the same line
    let mut k = i + ml;
    let mut found = None;
    while k + ml <= chars.len() && chars[k] != '\n' && k - (i + ml) <= 60 {
        if chars[k..].starts_with(&m) {
            found = Some(k);
            break;
        }
        // a lone same-head char inside aborts ("*a*b*c" pairs a-b first,
        // which is correct nearest-close; but "**a**" handled by ml=2 scan)
        k += 1;
    }
    let e = found?;
    let inner: String = chars[i + ml..e].iter().collect();
    if inner.is_empty() {
        return None;
    }
    let mhead = m[0];
    if inner.contains(mhead) {
        return None;
    }
    // strikethrough applies to anything; emphasis needs prose evidence
    if marker != "~~"
        && !inner.chars().any(|c| c.is_ascii_alphabetic() || is_cjk(c))
    {
        return None;
    }
    Some((e + ml, inner))
}

fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}' | // CJK unified
        '\u{3400}'..='\u{4DBF}' | // ext A
        '\u{3040}'..='\u{309F}' | // hiragana
        '\u{30A0}'..='\u{30FF}' | // katakana
        '\u{AC00}'..='\u{D7AF}' // hangul
    )
}

/// Normalize one text. Idempotent-ish for already-Chinese prose.
pub fn normalize(text: &str) -> String {
    let text = clean(text);
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // range minus / leading minus / version hyphen (AVX-512 → AVX512)
        if (c == '-' || c == '－') && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            let prev = if i > 0 { chars[i - 1] } else { ' ' };
            if prev.is_ascii_alphabetic() {
                i += 1; // drop: glued to a model name, digits read by parser
                continue;
            }
            if prev.is_ascii_digit() || prev == '.' {
                // N-M handled by the number parser's range lookahead; a lone
                // '-' between digits here means parser already passed — emit 到
                out.push_str("到");
            } else {
                out.push_str("负");
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
            // lone % (not after digits — digit trails are consumed by the
            // parser): keep position, best-effort.
            '％' | '%' => out.push_str("百分之"),
            '°' => out.push_str("度"),
            '×' | '*' => out.push_str("乘"),
            '÷' => out.push_str("除以"),
            '~' | '～' => out.push_str("到"),
            _ => out.push(c),
        }
        i += 1;
    }
    normalize_code(&out)
}

/// Code-token stabilization for Chinese TTS (v2, runs after numbers).
/// Dotted identifiers (`torch.no_grad`) otherwise trigger English-period
/// prosody at every dot and destabilize sampling on code-dense text.
/// Only dots/glue INSIDE words are rewritten; sentence dots survive.
pub fn normalize_code(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    while i < chars.len() {
        let c = chars[i];
        // Standalone Python booleans (word-bounded, exact case only).
        // "NoneType" / "Falsehood" must not match: check both sides.
        if (c == 'T' || c == 'F' || c == 'N')
            && (i == 0 || !is_word(chars[i - 1]) && chars[i - 1] != '.')
        {
            let rest = if c == 'T' { "rue" } else if c == 'F' { "alse" } else { "one" };
            let word: String = chars[i..].iter().take(rest.len() + 1).collect();
            let expect = format!("{c}{rest}");
            if word == expect {
                let after = i + rest.len() + 1;
                if after >= chars.len() || (!is_word(chars[after]) && chars[after] != '.') {
                    out.push_str(if c == 'T' { "真" } else if c == 'F' { "假" } else { "空" });
                    i = after;
                    continue;
                }
            }
        }
        if c == '.' {
            let next = if i + 1 < chars.len() { Some(chars[i + 1]) } else { None };
            match next {
                // identifier dot: torch.no_grad, cudnn.benchmark, e.g.
                Some(n) if n.is_ascii_alphanumeric() || n == '_' => {
                    out.push_str("点");
                    i += 1;
                    continue;
                }
                // leading-dot decimal (.5秒): prev must not be a word char
                // (digit.digit was already consumed by the number parser).
                Some(n) if n.is_ascii_digit() && (i == 0 || !is_word(chars[i - 1])) => {
                    out.push_str("零点");
                    i += 1;
                    continue;
                }
                _ => {}
            }
            out.push(c);
            i += 1;
            continue;
        }
        // underscore glue inside words: no_grad → no grad.
        if c == '_' && i > 0 && i + 1 < chars.len()
            && chars[i - 1].is_ascii_alphanumeric()
            && chars[i + 1].is_ascii_alphanumeric()
        {
            out.push(' ');
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Parse a number starting at chars[i] (chars[i] is a digit).
/// Returns (reading, next index). Consumes optional .frac, % suffix,
/// and N~M / N-M ranges (emits 到 + second number, recursive once).
/// Letter-adjacent integers (Qwen3, H100, x86, RTX4090) read digit by
/// digit — grouped reading ("四千零九十") is for standalone quantities.
/// Unit suffixes (80GB, 40dB) keep grouped reading; only a letter on the
/// LEFT (model names) or an explicit force flag (version ranges like
/// x86-64) triggers digitwise. A 4-digit run before 年 (spaces tolerated:
/// "2020 年") is a year.
fn parse_number(chars: &[char], i: usize, force_dw: bool) -> (String, usize) {
    // look back past dropped version hyphens (AVX-512 → AVX512: prev is X)
    // and past spaces (RTX 4090: model number after a Latin token).
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
    // percent suffix (space-tolerant: "35 %" → 百分之三十五)?
    let mut k = j;
    while k < chars.len() && (chars[k] == ' ' || chars[k] == '　') {
        k += 1;
    }
    let mut pct = false;
    if k < chars.len() && (chars[k] == '%' || chars[k] == '％') {
        pct = true;
        j = k + 1;
    }
    // year: exactly 4 digits + 年 (spaces tolerated) → digit-by-digit
    let mut k = j;
    while k < chars.len() && (chars[k] == ' ' || chars[k] == '　') {
        k += 1;
    }
    let is_year = int_part.len() == 4 && frac.is_none() && k < chars.len() && chars[k] == '年';
    let mut word = if is_year {
        j = k; // consume the spaces; 年 emits normally next iteration
        digitwise(&int_part)
    } else if left_letter {
        digitwise(&int_part)
    } else {
        int_part.parse::<u64>().map(int_zh).unwrap_or_else(|_| digitwise(&int_part))
    };
    if let Some(f) = frac {
        word.push_str("点");
        word.push_str(&digitwise(&f));
    }
    if pct {
        word = format!("百分之{word}");
    }
    // range: sep + digits → 到 + number (dropped when glued to a model
    // name: x86-64 → 八六六四 — the right part inherits digitwise)
    if j < chars.len() && matches!(chars[j], '~' | '～' | '-' | '－' | '–')
        && j + 1 < chars.len()
        && chars[j + 1].is_ascii_digit()
    {
        let (w2, j2) = parse_number(chars, j + 1, left_letter);
        if !left_letter {
            word.push_str("到");
        }
        word.push_str(&w2);
        j = j2;
    }
    (word, j)
}

/// Language-aware entry: v3 cleaning is universal (all languages);
/// Chinese adds numbers+code, Japanese its own rules, English passes
/// through otherwise (cleaning only).
pub fn normalize_for_language(text: &str, language: &str) -> String {
    if language.eq_ignore_ascii_case("chinese") {
        normalize(text)
    } else if language.eq_ignore_ascii_case("japanese") {
        crate::normalize_ja::normalize(&clean(text))
    } else {
        clean(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_zh_basics() {
        assert_eq!(int_zh(0), "零");
        assert_eq!(int_zh(5), "五");
        assert_eq!(int_zh(10), "十");
        assert_eq!(int_zh(11), "十一");
        assert_eq!(int_zh(17), "十七");
        assert_eq!(int_zh(20), "二十");
        assert_eq!(int_zh(27), "二十七");
        assert_eq!(int_zh(100), "一百");
        assert_eq!(int_zh(101), "一百零一");
        assert_eq!(int_zh(110), "一百一十");
        assert_eq!(int_zh(171), "一百七十一");
        assert_eq!(int_zh(1000), "一千");
        assert_eq!(int_zh(1010), "一千零一十");
        assert_eq!(int_zh(4090), "四千零九十");
    }

    #[test]
    fn test_int_zh_groups() {
        assert_eq!(int_zh(10000), "一万");
        assert_eq!(int_zh(10001), "一万零一");
        assert_eq!(int_zh(12000), "一万二千");
        assert_eq!(int_zh(10100), "一万零一百");
        assert_eq!(int_zh(100000), "十万");
        assert_eq!(int_zh(100000000), "一亿");
        assert_eq!(int_zh(2020), "二千零二十");
    }

    #[test]
    fn test_normalize_sentences() {
        assert_eq!(normalize("RTF为4.78"), "RTF为四点七八");
        assert_eq!(normalize("2020年"), "二零二零年");
        assert_eq!(normalize("2020 年的笔记本"), "二零二零年的笔记本");
        assert_eq!(normalize("RTX 4090"), "RTX 四零九零");
        assert_eq!(normalize("x86-64"), "x八六六四");
        assert_eq!(normalize("AVX-512"), "AVX五一二");
        assert_eq!(normalize("-40dB"), "负四十dB");
        assert_eq!(normalize("35%"), "百分之三十五");
        assert_eq!(normalize("171毫秒"), "一百七十一毫秒");
        assert_eq!(normalize("H100"), "H一零零");
        assert_eq!(normalize("Qwen3"), "Qwen三");
        assert_eq!(normalize("5.7秒"), "五点七秒");
        assert_eq!(normalize("2核"), "二核");
        assert_eq!(normalize("27s"), "二十七s");
        assert_eq!(normalize("0.6B模型"), "零点六B模型");
    }

    #[test]
    fn test_normalize_passthrough() {
        assert_eq!(normalize("你好，世界。"), "你好，世界。");
        assert_eq!(normalize("Hello world"), "Hello world");
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn test_normalize_code_dots() {
        assert_eq!(normalize_code("torch.no_grad"), "torch点no grad");
        assert_eq!(
            normalize_code("torch.backends.cudnn.benchmark"),
            "torch点backends点cudnn点benchmark"
        );
        // sentence dots survive (prosody)
        assert_eq!(normalize_code("Hello. World"), "Hello. World");
        assert_eq!(normalize_code("结尾."), "结尾.");
        // decimals already handled upstream; code pass must not touch CJK 点
        assert_eq!(normalize_code("四点七八"), "四点七八");
        // full pipeline: numbers first, code second
        assert_eq!(normalize("RTF为4.78"), "RTF为四点七八");
        assert_eq!(normalize("torch.no_grad的开销3.5秒"), "torch点no grad的开销三点五秒");
    }

    #[test]
    fn test_normalize_code_underscore_edges() {
        assert_eq!(normalize_code("no_grad"), "no grad");
        assert_eq!(normalize_code("empty_cache"), "empty cache");
        assert_eq!(normalize_code("_private"), "_private");
        assert_eq!(normalize_code("trailing_"), "trailing_");
        assert_eq!(normalize_code("a__b"), "a__b"); // only single glue between alnums
    }

    #[test]
    fn test_normalize_code_booleans() {
        assert_eq!(normalize_code("设置为 True 时"), "设置为 真 时");
        assert_eq!(normalize_code("(False、None)"), "(假、空)");
        // must not fire inside longer words or dotted paths
        assert_eq!(normalize_code("NoneType"), "NoneType");
        assert_eq!(normalize_code("Falsehood"), "Falsehood");
        assert_eq!(normalize_code("torch.True"), "torch点True");
        // lowercase untouched (plain English words, out of scope)
        assert_eq!(normalize_code("true cost"), "true cost");
    }

    #[test]
    fn test_normalize_code_suspect_sentence() {
        // the pytorch_issue_on_gpu.py chunk-8 sentence (122s silence hole):
        // dots must become 点, True → 真, no sentence split inside.
        let inp = "当 torch.backends.cudnn.benchmark 设置为 True 时，cuDNN 会在第一次运行某个 shape 的卷积时，测试几种不同的卷积算法，选出最快的一种缓存下来，之后相同 shape 都使用这个算法。";
        let exp = "当 torch点backends点cudnn点benchmark 设置为 真 时，cuDNN 会在第一次运行某个 shape 的卷积时，测试几种不同的卷积算法，选出最快的一种缓存下来，之后相同 shape 都使用这个算法。";
        assert_eq!(normalize(inp), exp);
        // idempotent: second pass changes nothing
        assert_eq!(normalize(&normalize(inp)), exp);
    }

    #[test]
    fn test_language_gate() {
        assert_eq!(normalize_for_language("2020年", "chinese"), "二零二零年");
        assert_eq!(normalize_for_language("2020年", "english"), "2020年");
        assert_eq!(normalize_for_language("2020年", "Chinese"), "二零二零年");
        assert_eq!(normalize_for_language("4月1日", "japanese"), "シガツツイタチ");
        assert_eq!(normalize_for_language("4月1日", "english"), "4月1日");
        assert_eq!(normalize_for_language("2020年", "japanese"), "ニセンニジュウネン");
    }

    #[test]
    fn test_clean_invisible() {
        assert_eq!(clean("a​b‍c﻿d"), "abcd"); // ZWSP ZWJ BOM
        assert_eq!(clean("a\u{202A}b\u{202E}c\u{2066}d\u{2069}e"), "abcde"); // bidi controls
        assert_eq!(clean("a︀b️c"), "abc"); // VS1 + VS16
        // emoji pictographs gone (incl. ZWJ family glue, keycap, tags)
        assert_eq!(clean("好吃😂"), "好吃");
        assert_eq!(clean("👨‍👩‍👧"), "");
        assert_eq!(clean("按1⃣确认"), "按1确认");
        // dingbats/misc-symbols KEPT (documented boundary)
        assert_eq!(clean("五星★★★★★"), "五星★★★★★");
        assert_eq!(clean("电话☎"), "电话☎");
    }

    #[test]
    fn test_clean_controls_and_space() {
        assert_eq!(clean("a\0b\x07c\x7Fd\u{80}e"), "abcde");
        assert_eq!(clean("a\nb\rc\td"), "a\nb\rc d"); // \n\r kept; tab folds to space
        assert_eq!(clean("a   b\t\tc\u{3000}\u{3000}d\u{A0}e"), "a b c d e");
        assert_eq!(clean("  首尾空格  "), "首尾空格");
    }

    #[test]
    fn test_clean_markdown() {
        assert_eq!(clean("用`code`表示"), "用code表示");
        assert_eq!(clean("# 标题"), "标题");
        assert_eq!(clean("### 深标题"), "深标题");
        assert_eq!(clean("> 引用"), "引用");
        assert_eq!(clean("- 列表项"), "列表项");
        assert_eq!(clean("C#语言"), "C#语言"); // mid-line # kept
        assert_eq!(clean("a>b"), "a>b");
        assert_eq!(clean("**粗体**和*斜体*"), "粗体和斜体");
        assert_eq!(clean("__init__"), "init");
        assert_eq!(clean("~~删除~~"), "删除");
        assert_eq!(clean("看[文档](http://x.y/z)吧"), "看文档吧");
        assert_eq!(clean("[坏链接"), "[坏链接");
        assert_eq!(clean("a|b"), "a b");
        assert_eq!(clean("---\n正文"), "\n正文");
        // arithmetic guards: single markers still mean math downstream
        assert_eq!(normalize("2*3"), "二乘三");
        assert_eq!(normalize("**2**"), "乘乘二乘乘"); // no prose → untouched
    }

    #[test]
    fn test_clean_no_panic_multibyte_soup() {
        // every rule trigger adjacent to multi-byte CJK (byte-slicing here
        // would panic mid-codepoint); must survive + converge.
        let soup = "中​文😂a.b_c­d\u{202E}**重**[链](u)　　3.5% True\n# 标`题`";
        let mut s = soup.to_string();
        for _ in 0..3 {
            s = normalize(&s);
        }
        let again = normalize(&s);
        assert_eq!(s, again); // fixed point, no panic
        assert!(!s.contains('\u{200B}'));
        assert!(!s.chars().any(|c| ('\u{1F000}'..='\u{1FAFF}').contains(&c)));
    }

    #[test]
    fn test_clean_idempotent_and_passthrough() {
        let nasty = "# 标​题\n- `a.b` 用True😂  3.5%  **x** [t](u) a|b ---";
        let once = normalize(nasty);
        assert_eq!(normalize(&once), once);
        assert_eq!(normalize("你好，世界。"), "你好，世界。");
        // v2 suspect sentence unchanged by v3 (no v3 chars in it)
        let s = "当 torch点backends点cudnn点benchmark 设置为 真 时。";
        assert_eq!(normalize(s), s);
    }
}
