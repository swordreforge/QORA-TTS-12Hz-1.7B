//! Audio preprocessing for speaker encoder
//! Extracts 128-bin mel-spectrogram from 24kHz audio

use std::f32::consts::PI;

pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
    pub pre_emphasis: bool,
    pub slaney_norm: bool,      // librosa norm='slaney'
    pub reflect_pad: bool,      // Pad signal with (n_fft - hop) // 2 reflect before STFT
    pub periodic_window: bool,  // PyTorch-style periodic Hann window
}

impl Default for MelConfig {
    /// Default for speech tokenizer (codec encoder)
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            n_fft: 512,
            hop_length: 240,
            win_length: 480,
            n_mels: 128,
            fmin: 0.0,
            fmax: 12000.0,
            pre_emphasis: true,
            slaney_norm: false,
            reflect_pad: false,
            periodic_window: false,
        }
    }
}

impl MelConfig {
    /// Config for speaker encoder (ECAPA-TDNN) — matches Python mel_spectrogram()
    /// Uses librosa mel filterbank (Slaney norm), PyTorch STFT (periodic Hann, reflect pad)
    pub fn speaker_encoder() -> Self {
        Self {
            sample_rate: 24000,
            n_fft: 1024,
            hop_length: 256,
            win_length: 1024,
            n_mels: 128,
            fmin: 0.0,
            fmax: 12000.0,
            pre_emphasis: false,
            slaney_norm: true,
            reflect_pad: true,
            periodic_window: true,
        }
    }
}

/// Extract mel-spectrogram from audio samples
/// Returns [n_mels, T] in channel-first format
pub fn extract_mel_spectrogram(audio: &[f32], config: &MelConfig) -> Vec<f32> {
    // 1. Optional pre-emphasis filter
    let processed = if config.pre_emphasis {
        pre_emphasis(audio, 0.97)
    } else {
        audio.to_vec()
    };

    // 2. Reflect-pad signal if configured (matches Python STFT center=False with explicit padding)
    //    Python: padding = (n_fft - hop_size) // 2, then F.pad(y, (padding, padding), mode="reflect")
    let padded_signal = if config.reflect_pad {
        let pad = (config.n_fft - config.hop_length) / 2;
        reflect_pad_signal(&processed, pad)
    } else {
        processed
    };

    // 3. Frame the signal
    let frames = frame_signal(&padded_signal, config.win_length, config.hop_length);
    let n_frames = frames.len();

    // 4. Apply Hann window (periodic for PyTorch compatibility)
    let window = if config.periodic_window {
        hann_window_periodic(config.win_length)
    } else {
        hann_window(config.win_length)
    };

    // 5. Compute STFT magnitude
    let mut spectrogram = Vec::new();
    for frame in frames {
        let windowed = apply_window(&frame, &window);
        let spectrum = compute_fft_magnitude(&windowed, config.n_fft);
        spectrogram.push(spectrum);
    }

    // 6. Apply mel filterbank
    let mel_filters = create_mel_filterbank(
        config.n_mels,
        config.n_fft,
        config.sample_rate,
        config.fmin,
        config.fmax,
        config.slaney_norm,
    );

    let mut mel_spec = vec![0.0f32; config.n_mels * n_frames];
    for (t, spectrum) in spectrogram.iter().enumerate() {
        for m in 0..config.n_mels {
            let mut energy = 0.0f32;
            for (f, &weight) in mel_filters[m].iter().enumerate() {
                if f < spectrum.len() {
                    energy += spectrum[f] * weight;
                }
            }
            // Log compression: ln(max(energy, 1e-5)) matching Python dynamic_range_compression
            mel_spec[m * n_frames + t] = energy.max(1e-5).ln();
        }
    }

    mel_spec
}

/// Reflect-pad a signal on both sides
fn reflect_pad_signal(signal: &[f32], pad: usize) -> Vec<f32> {
    let len = signal.len();
    let mut out = vec![0.0f32; pad + len + pad];

    // Left reflect: signal[pad], signal[pad-1], ..., signal[1]
    for i in 0..pad {
        let src_idx = if pad - i < len { pad - i } else { (pad - i) % len };
        out[i] = signal[src_idx];
    }

    // Center: copy signal
    out[pad..pad + len].copy_from_slice(signal);

    // Right reflect: signal[len-2], signal[len-3], ..., signal[len-1-pad]
    for i in 0..pad {
        let src_idx = if len >= 2 + i { len - 2 - i } else { i % len };
        out[pad + len + i] = signal[src_idx];
    }

    out
}

/// Pre-emphasis filter to boost high frequencies
fn pre_emphasis(audio: &[f32], coef: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; audio.len()];
    out[0] = audio[0];
    for i in 1..audio.len() {
        out[i] = audio[i] - coef * audio[i - 1];
    }
    out
}

/// Frame signal into overlapping windows
fn frame_signal(signal: &[f32], frame_length: usize, hop_length: usize) -> Vec<Vec<f32>> {
    let mut frames = Vec::new();
    let mut start = 0;

    while start + frame_length <= signal.len() {
        frames.push(signal[start..start + frame_length].to_vec());
        start += hop_length;
    }

    // Pad last frame if needed
    if start < signal.len() {
        let mut last_frame = signal[start..].to_vec();
        last_frame.resize(frame_length, 0.0);
        frames.push(last_frame);
    }

    frames
}

/// Create symmetric Hann window (traditional)
fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (length - 1) as f32).cos()))
        .collect()
}

/// Create periodic Hann window (PyTorch torch.hann_window default)
/// Uses N instead of N-1 in denominator
fn hann_window_periodic(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / length as f32).cos()))
        .collect()
}

/// Apply window to frame
fn apply_window(frame: &[f32], window: &[f32]) -> Vec<f32> {
    frame.iter()
        .zip(window.iter())
        .map(|(x, w)| x * w)
        .collect()
}

/// Compute FFT magnitude spectrum
/// Returns sqrt(real^2 + imag^2 + 1e-9) matching PyTorch
fn compute_fft_magnitude(signal: &[f32], n_fft: usize) -> Vec<f32> {
    // Zero-pad to n_fft
    let mut real = signal.to_vec();
    real.resize(n_fft, 0.0);
    let mut imag = vec![0.0f32; n_fft];

    // Use Cooley-Tukey FFT (requires power-of-2 size, which n_fft=512/1024 always is)
    fft_in_place(&mut real, &mut imag);

    // Return magnitude for bins 0..=n_fft/2
    let mut magnitude = Vec::with_capacity(n_fft / 2 + 1);
    for k in 0..=n_fft / 2 {
        magnitude.push((real[k] * real[k] + imag[k] * imag[k] + 1e-9).sqrt());
    }
    magnitude
}

/// In-place Cooley-Tukey radix-2 FFT
fn fft_in_place(real: &mut [f32], imag: &mut [f32]) {
    let n = real.len();
    assert!(n.is_power_of_two(), "FFT requires power-of-2 size");

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }

    // Butterfly stages
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_step = -2.0 * PI / len as f32;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let angle = angle_step * k as f32;
                let wr = angle.cos();
                let wi = angle.sin();
                let a = start + k;
                let b = start + k + half;
                let tr = wr * real[b] - wi * imag[b];
                let ti = wr * imag[b] + wi * real[b];
                real[b] = real[a] - tr;
                imag[b] = imag[a] - ti;
                real[a] += tr;
                imag[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// Create mel filterbank with optional Slaney normalization
fn create_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: usize,
    fmin: f32,
    fmax: f32,
    slaney_norm: bool,
) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;

    // Convert Hz to mel
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // Create linearly-spaced mel points
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert mel back to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin (using float for precision, matching librosa)
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    // Create filterbank using librosa-style ramps for precise interpolation
    let mut filterbank = vec![vec![0.0f32; n_freqs]; n_mels];

    for m in 0..n_mels {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];

        for k in 0..n_freqs {
            let freq = fft_freqs[k];

            if freq >= f_left && freq <= f_center && f_center > f_left {
                // Rising slope
                filterbank[m][k] = (freq - f_left) / (f_center - f_left);
            } else if freq > f_center && freq <= f_right && f_right > f_center {
                // Falling slope
                filterbank[m][k] = (f_right - freq) / (f_right - f_center);
            }
        }

        // Slaney normalization: divide each filter by its bandwidth
        // enorm = 2.0 / (hz_points[m+2] - hz_points[m])
        if slaney_norm {
            let bandwidth = f_right - f_left;
            if bandwidth > 0.0 {
                let enorm = 2.0 / bandwidth;
                for k in 0..n_freqs {
                    filterbank[m][k] *= enorm;
                }
            }
        }
    }

    filterbank
}

/// Convert Hz to mel scale (Slaney / Auditory Toolbox formula, matching librosa htk=False)
/// This is what the official Qwen3-TTS mel_spectrogram() uses via librosa.filters.mel.
fn hz_to_mel(hz: f32) -> f32 {
    let f_min = 0.0f32;
    let f_sp = 200.0f32 / 3.0f32;
    let min_log_hz = 1000.0f32;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = 6.4f32.ln() / 27.0f32;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        (hz - f_min) / f_sp
    }
}

/// Convert mel to Hz scale (Slaney, matching librosa htk=False)
fn mel_to_hz(mel: f32) -> f32 {
    let f_min = 0.0f32;
    let f_sp = 200.0f32 / 3.0f32;
    let min_log_hz = 1000.0f32;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = 6.4f32.ln() / 27.0f32;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_min + f_sp * mel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_conversion() {
        // Slaney scale: 1000 Hz <-> 15.0 mel (linear region boundary)
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-3);
        assert!((mel_to_hz(15.0) - 1000.0).abs() < 1.0);
        // Roundtrip across the log region
        for &hz in &[500.0f32, 4000.0, 8000.0, 12000.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < hz * 1e-4 + 1e-2);
        }
    }

    #[test]
    fn test_hann_window() {
        let window = hann_window(512);
        assert_eq!(window.len(), 512);
        assert!(window[0] < 0.01); // Start near 0
        assert!(window[256] > 0.99); // Peak near 1
    }

    #[test]
    fn test_reflect_pad() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = reflect_pad_signal(&signal, 2);
        // Left: signal[2], signal[1] = 3.0, 2.0
        // Center: 1.0, 2.0, 3.0, 4.0, 5.0
        // Right: signal[3], signal[2] = 4.0, 3.0
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_slaney_norm() {
        let filters = create_mel_filterbank(4, 512, 24000, 0.0, 12000.0, true);
        // With Slaney norm, each filter should have normalized area
        for f in &filters {
            let sum: f32 = f.iter().sum();
            assert!(sum > 0.0, "Filter should have positive sum");
        }
    }
}
