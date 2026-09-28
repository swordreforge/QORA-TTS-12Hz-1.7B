//! PCM 16-bit WAV writer/reader (mono).

use std::io::{self, Write};
use std::path::Path;

/// Write f32 audio samples [-1.0, 1.0] as a 16-bit PCM WAV file.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);

    let num_samples = samples.len() as u32;
    let bytes_per_sample: u16 = 2;
    let num_channels: u16 = 1;
    let data_size = num_samples * bytes_per_sample as u32;
    let file_size = 36 + data_size;

    // RIFF header
    f.write_all(b"RIFF")?;
    f.write_all(&file_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;

    // fmt chunk
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;         // chunk size
    f.write_all(&1u16.to_le_bytes())?;           // PCM format
    f.write_all(&num_channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    let byte_rate = sample_rate * num_channels as u32 * bytes_per_sample as u32;
    f.write_all(&byte_rate.to_le_bytes())?;
    let block_align = num_channels * bytes_per_sample;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&(bytes_per_sample * 8).to_le_bytes())?; // bits per sample

    // data chunk
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        f.write_all(&i16_val.to_le_bytes())?;
    }

    f.flush()?;
    Ok(())
}

/// Read WAV file and return (samples, sample_rate)
/// Converts to mono f32 samples [-1.0, 1.0]
pub fn read_wav(path: &Path) -> io::Result<(Vec<f32>, u32)> {
    let all_data = std::fs::read(path)?;

    if all_data.len() < 44 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "WAV file too small"));
    }

    // Check RIFF header
    if &all_data[0..4] != b"RIFF" || &all_data[8..12] != b"WAVE" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a valid WAV file"));
    }

    // Find fmt chunk
    let mut pos = 12;
    let mut num_channels = 0usize;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut fmt_found = false;

    while pos + 8 <= all_data.len() {
        let chunk_id = &all_data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            all_data[pos + 4],
            all_data[pos + 5],
            all_data[pos + 6],
            all_data[pos + 7],
        ]) as usize;

        if chunk_id == b"fmt " {
            if pos + 8 + chunk_size > all_data.len() {
                break;
            }
            num_channels = u16::from_le_bytes([all_data[pos + 10], all_data[pos + 11]]) as usize;
            sample_rate = u32::from_le_bytes([
                all_data[pos + 12],
                all_data[pos + 13],
                all_data[pos + 14],
                all_data[pos + 15],
            ]);
            bits_per_sample = u16::from_le_bytes([all_data[pos + 22], all_data[pos + 23]]);
            fmt_found = true;
        } else if chunk_id == b"data" && fmt_found {
            // Found data chunk
            if bits_per_sample != 16 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Only 16-bit WAV supported"));
            }

            let data_size = chunk_size;
            let data_start = pos + 8;

            if data_start + data_size > all_data.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Data chunk size exceeds file"));
            }

            let data = &all_data[data_start..data_start + data_size];
            let num_samples = data_size / 2 / num_channels;

            // Convert to mono f32
            let mut samples = vec![0.0f32; num_samples];

            for i in 0..num_samples {
                let mut sum = 0.0f32;
                for ch in 0..num_channels {
                    let offset = (i * num_channels + ch) * 2;
                    if offset + 1 < data.len() {
                        let i16_val = i16::from_le_bytes([data[offset], data[offset + 1]]);
                        sum += i16_val as f32 / 32768.0;
                    }
                }
                samples[i] = sum / num_channels as f32;
            }

            return Ok((samples, sample_rate));
        }

        pos += 8 + chunk_size;
        // Chunks are word-aligned
        if chunk_size % 2 == 1 {
            pos += 1;
        }
    }

    Err(io::Error::new(io::ErrorKind::InvalidData, "Missing data chunk"))
}

/// Linear-interpolation resample of mono audio to a new sample rate.
/// Duration-preserving: out_len = round(in_len * to_sr / from_sr).
pub fn resample_mono(input: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if input.is_empty() || from_sr == 0 || to_sr == 0 {
        return input.to_vec();
    }
    if from_sr == to_sr {
        return input.to_vec();
    }
    let ratio = to_sr as f64 / from_sr as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    if out_len == 0 {
        return Vec::new();
    }
    let last = input.len() - 1;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 / ratio;
        let i0 = (pos.floor() as usize).min(last);
        let i1 = (i0 + 1).min(last);
        let frac = (pos - i0 as f64) as f32;
        out.push(input[i0] + (input[i1] - input[i0]) * frac);
    }
    out
}

/// Read any mono/stereo 16-bit WAV and return mono f32 at 24000 Hz,
/// resampling automatically when needed. Returns (samples, was_converted).
pub fn read_wav_mono_24k(path: &Path) -> io::Result<(Vec<f32>, bool)> {
    let (samples, sr) = read_wav(path)?;
    if sr == 24000 {
        return Ok((samples, false));
    }
    Ok((resample_mono(&samples, sr, 24000), true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(sr: u32, freq: f32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n).map(|i| (2.0 * PI * freq * i as f32 / sr as f32).sin()).collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    fn zero_crossings(x: &[f32]) -> usize {
        x.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count()
    }

    #[test]
    fn test_resample_identity() {
        let x = sine(24000, 440.0, 0.1);
        let y = resample_mono(&x, 24000, 24000);
        assert_eq!(y, x);
    }

    #[test]
    fn test_resample_48k_to_24k() {
        let x = sine(48000, 1000.0, 1.0);
        let y = resample_mono(&x, 48000, 24000);
        assert_eq!(y.len(), 24000);
        // 1kHz preserved: ~2000 zero crossings/sec
        let zc = zero_crossings(&y);
        assert!((1900..2100).contains(&zc), "zc={zc}");
        // level preserved
        assert!((rms(&y) - rms(&x)).abs() < 0.02, "rms {} vs {}", rms(&y), rms(&x));
    }

    #[test]
    fn test_resample_16k_to_24k() {
        let x = sine(16000, 500.0, 0.5);
        let y = resample_mono(&x, 16000, 24000);
        assert_eq!(y.len(), 12000);
        let zc = zero_crossings(&y);
        assert!((450..550).contains(&zc), "zc={zc}");
    }

    #[test]
    fn test_resample_empty() {
        assert!(resample_mono(&[], 48000, 24000).is_empty());
    }

    fn speech_frame() -> Vec<f32> {
        vec![0.1; 480] // energy 0.01 >> 1e-4
    }

    #[test]
    fn test_trim_silence_compresses_gap() {
        // speech + 1s silence (50 frames) + speech, max_gap 0.25s (13 frames ceil)
        let mut x = speech_frame();
        x.extend(vec![0.0; 480 * 50]);
        x.extend(speech_frame());
        let y = trim_silence(&x, 24000, 0.25);
        // 1 + 13 + 1 frames
        assert_eq!(y.len(), 480 * 15, "len={}", y.len());
    }

    #[test]
    fn test_trim_silence_cuts_trailing() {
        let mut x = speech_frame();
        x.extend(vec![0.0; 480 * 50]);
        let y = trim_silence(&x, 24000, 0.25);
        assert_eq!(y.len(), 480 * 14, "len={}", y.len()); // 1 speech + 13 gap
    }

    #[test]
    fn test_trim_silence_short_gap_untouched() {
        // 0.2s gap < 0.25s max → unchanged
        let mut x = speech_frame();
        x.extend(vec![0.0; 480 * 10]);
        x.extend(speech_frame());
        let y = trim_silence(&x, 24000, 0.25);
        assert_eq!(y, x);
    }

    #[test]
    fn test_trim_silence_all_silence() {
        let x = vec![0.0; 480 * 5];
        assert!(trim_silence(&x, 24000, 0.25).is_empty());
    }

    #[test]
    fn test_trim_silence_leading_kept() {
        // leading silence shorter than max is kept (only internal/trailing touched)
        let mut x = vec![0.0; 480 * 5];
        x.extend(speech_frame());
        let y = trim_silence(&x, 24000, 0.25);
        assert_eq!(y, x);
    }
}

/// Compress internal silences longer than `max_gap_secs` down to `max_gap_secs`
/// and trim trailing silence beyond `max_gap_secs`.
/// Frame = 20ms, speech threshold = mean square energy >= 1e-4
/// (same convention as the analysis scripts in docs).
pub fn trim_silence(audio: &[f32], sample_rate: u32, max_gap_secs: f32) -> Vec<f32> {
    if audio.is_empty() || max_gap_secs <= 0.0 {
        return audio.to_vec();
    }
    let frame = (sample_rate as usize / 50).max(1);
    let max_gap_frames = ((max_gap_secs * sample_rate as f32) / frame as f32).ceil() as usize;

    // Classify frames
    let n_frames = audio.len().div_ceil(frame);
    let mut is_speech = vec![false; n_frames];
    for (i, s) in is_speech.iter_mut().enumerate() {
        let end = ((i + 1) * frame).min(audio.len());
        let seg = &audio[i * frame..end];
        let e: f32 = seg.iter().map(|v| v * v).sum::<f32>() / seg.len() as f32;
        *s = e >= 1e-4;
    }

    // Find last speech frame; drop everything after last_speech + max_gap_frames
    let Some(last_speech) = is_speech.iter().rposition(|&s| s) else {
        return Vec::new();
    };
    let keep_frames = (last_speech + 1 + max_gap_frames).min(n_frames);

    // Copy, skipping the middle of over-long internal gaps
    let mut out: Vec<f32> = Vec::with_capacity(audio.len());
    let mut i = 0;
    while i < keep_frames {
        if is_speech[i] {
            let end = ((i + 1) * frame).min(audio.len());
            out.extend_from_slice(&audio[i * frame..end]);
            i += 1;
        } else {
            let mut j = i;
            while j < keep_frames && !is_speech[j] {
                j += 1;
            }
            let gap = j - i;
            let keep = gap.min(max_gap_frames);
            for k in i..i + keep {
                let end = ((k + 1) * frame).min(audio.len());
                out.extend_from_slice(&audio[k * frame..end]);
            }
            i = j;
        }
    }
    out
}
