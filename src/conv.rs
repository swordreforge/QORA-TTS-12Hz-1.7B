//! Convolution primitives for the speech decoder.
//!
//! All operations work on f32 buffers in channel-first format: [channels, length].
//! Heavy operations are multi-threaded for CPU parallelism.

use std::thread;

fn num_threads() -> usize {
    thread::available_parallelism().map(|n| n.get()).unwrap_or(6)
}

/// Wrapper to send raw pointers across threads (we guarantee non-overlapping writes).
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline]
    unsafe fn add(self, count: usize) -> *mut f32 {
        self.0.add(count)
    }
}

// ============================================================
// Conv1d
// ============================================================

pub struct Conv1dWeight {
    pub weight: Vec<f32>,  // [out_channels, in_channels, kernel_size]
    pub bias: Vec<f32>,    // [out_channels]
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

/// Standard 1D convolution — multi-threaded by output channels.
/// Input: [in_channels, in_length], Output: [out_channels, out_length]
pub fn conv1d(input: &[f32], w: &Conv1dWeight) -> Vec<f32> {
    let in_len = input.len() / w.in_channels;
    let out_len = (in_len + 2 * w.padding - w.dilation * (w.kernel_size - 1) - 1) / w.stride + 1;
    let total = w.out_channels * out_len;
    let mut output = vec![0.0f32; total];

    let ops = w.out_channels * w.in_channels * w.kernel_size * out_len;
    if ops < 500_000 {
        // Small: single-threaded
        conv1d_range(input, w, 0, w.out_channels, in_len, out_len, &mut output);
        return output;
    }

    let n_threads = num_threads().min(w.out_channels);
    let chunk = (w.out_channels + n_threads - 1) / n_threads;
    let out_ptr = SendPtr(output.as_mut_ptr());

    thread::scope(|s| {
        for tid in 0..n_threads {
            let oc_start = tid * chunk;
            let oc_end = (oc_start + chunk).min(w.out_channels);
            if oc_start >= oc_end { break; }
            let ptr = out_ptr;
            s.spawn(move || {
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.add(oc_start * out_len),
                        (oc_end - oc_start) * out_len,
                    )
                };
                conv1d_range(input, w, oc_start, oc_end, in_len, out_len, out_slice);
            });
        }
    });
    output
}

#[inline]
fn conv1d_range(
    input: &[f32], w: &Conv1dWeight,
    oc_start: usize, oc_end: usize,
    in_len: usize, out_len: usize,
    output: &mut [f32],
) {
    let ick = w.in_channels * w.kernel_size;
    for oc in oc_start..oc_end {
        let local_oc = oc - oc_start;
        let w_base = oc * ick;
        for o in 0..out_len {
            let mut sum = w.bias[oc];
            for ic in 0..w.in_channels {
                let in_row = ic * in_len;
                let w_row = w_base + ic * w.kernel_size;
                for k in 0..w.kernel_size {
                    let in_pos = o * w.stride + k * w.dilation;
                    if in_pos >= w.padding && in_pos < in_len + w.padding {
                        let idx = in_pos - w.padding;
                        sum += input[in_row + idx] * w.weight[w_row + k];
                    }
                }
            }
            output[local_oc * out_len + o] = sum;
        }
    }
}

// ============================================================
// CausalConv1d
// ============================================================

/// Causal Conv1d — left-padded, multi-threaded by output channels.
/// Padding = (kernel_size - 1) * dilation.
pub fn causal_conv1d(input: &[f32], w: &Conv1dWeight) -> Vec<f32> {
    let in_len = input.len() / w.in_channels;
    let out_len = in_len; // causal: same length

    let mut output = vec![0.0f32; w.out_channels * out_len];

    let ops = w.out_channels * w.in_channels * w.kernel_size * out_len;
    if ops < 500_000 {
        causal_conv1d_range(input, w, 0, w.out_channels, in_len, out_len, &mut output);
        return output;
    }

    let n_threads = num_threads().min(w.out_channels);
    let chunk = (w.out_channels + n_threads - 1) / n_threads;
    let out_ptr = SendPtr(output.as_mut_ptr());

    thread::scope(|s| {
        for tid in 0..n_threads {
            let oc_start = tid * chunk;
            let oc_end = (oc_start + chunk).min(w.out_channels);
            if oc_start >= oc_end { break; }
            let ptr = out_ptr;
            s.spawn(move || {
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.add(oc_start * out_len),
                        (oc_end - oc_start) * out_len,
                    )
                };
                causal_conv1d_range(input, w, oc_start, oc_end, in_len, out_len, out_slice);
            });
        }
    });
    output
}

/// Dispatch: AVX2 vectorized range when available, scalar oracle otherwise.
/// Both produce bit-identical output (mul+add op sequence preserved per
/// output element); see differential tests in `tests` module below.
/// `QORA_FMA=1` selects the FMA variant (single rounding, NOT bit-identical,
/// ~10-15% faster); requires AVX2+FMA hardware, otherwise ignored.
#[inline]
fn causal_conv1d_range(
    input: &[f32], w: &Conv1dWeight,
    oc_start: usize, oc_end: usize,
    in_len: usize, out_len: usize,
    output: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::has_avx2() {
        use std::sync::OnceLock;
        static FMA: OnceLock<bool> = OnceLock::new();
        let use_fma = *FMA.get_or_init(|| {
            std::env::var("QORA_FMA").map(|v| v == "1").unwrap_or(false)
                && crate::simd::has_avx2_fma()
        });
        unsafe {
            if use_fma {
                crate::simd::causal_conv1d_range_avx2_fma(
                    input, &w.weight, &w.bias,
                    w.in_channels, w.kernel_size, w.dilation,
                    oc_start, oc_end, in_len, out_len, output,
                );
            } else {
                crate::simd::causal_conv1d_range_avx2(
                    input, &w.weight, &w.bias,
                    w.in_channels, w.kernel_size, w.dilation,
                    oc_start, oc_end, in_len, out_len, output,
                );
            }
        }
        return;
    }
    causal_conv1d_range_scalar(input, w, oc_start, oc_end, in_len, out_len, output);
}

#[inline]
fn causal_conv1d_range_scalar(
    input: &[f32], w: &Conv1dWeight,
    oc_start: usize, oc_end: usize,
    in_len: usize, out_len: usize,
    output: &mut [f32],
) {
    let causal_pad = (w.kernel_size - 1) * w.dilation;
    let ick = w.in_channels * w.kernel_size;
    // Length tiling: mirrors the AVX2 macro (same tile geometry → same
    // output bits; each element's full sum lives inside one o-iteration,
    // so reordering positions is exact). Keeps the input window LLC-resident.
    let tile = ((out_len.min(262144 / w.in_channels.max(1))).clamp(128, 16384)).max(1);
    for oc in oc_start..oc_end {
        let local_oc = oc - oc_start;
        let w_base = oc * ick;
        let mut ts = 0;
        while ts < out_len {
            let te = (ts + tile).min(out_len);
            for o in ts..te {
            let mut sum = w.bias[oc];
            for ic in 0..w.in_channels {
                let in_row = ic * in_len;
                let w_row = w_base + ic * w.kernel_size;
                for k in 0..w.kernel_size {
                    let in_pos = o + k * w.dilation;
                    if in_pos >= causal_pad {
                        let idx = in_pos - causal_pad;
                        if idx < in_len {
                            sum += input[in_row + idx] * w.weight[w_row + k];
                        }
                    }
                }
            }
            output[local_oc * out_len + o] = sum;
            }
            ts = te;
            }
        }
}

// ============================================================
// ConvTranspose1d
// ============================================================

pub struct ConvTranspose1dWeight {
    pub weight: Vec<f32>,  // [in_channels, out_channels, kernel_size]
    pub bias: Vec<f32>,    // [out_channels]
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

/// Transposed 1D convolution — multi-threaded by output channels.
/// Input: [in_channels, in_length], Output: [out_channels, out_length]
pub fn conv_transpose1d(input: &[f32], w: &ConvTranspose1dWeight) -> Vec<f32> {
    let in_len = input.len() / w.in_channels;
    let out_len = (in_len - 1) * w.stride - 2 * w.padding + w.kernel_size;
    if std::env::var("QORA_CONVT_SHAPES").as_deref() == Ok("1") {
        eprintln!("  [convt] ic={} oc={} k={} s={} in={} out={}",
            w.in_channels, w.out_channels, w.kernel_size, w.stride, in_len, out_len);
    }
    let mut output = vec![0.0f32; w.out_channels * out_len];

    let ops = w.in_channels * w.out_channels * w.kernel_size * in_len;
    if ops < 500_000 {
        conv_transpose1d_range(input, w, 0, w.out_channels, in_len, out_len, &mut output);
        return output;
    }

    let n_threads = num_threads().min(w.out_channels);
    let chunk = (w.out_channels + n_threads - 1) / n_threads;
    let out_ptr = SendPtr(output.as_mut_ptr());

    thread::scope(|s| {
        for tid in 0..n_threads {
            let oc_start = tid * chunk;
            let oc_end = (oc_start + chunk).min(w.out_channels);
            if oc_start >= oc_end { break; }
            let ptr = out_ptr;
            s.spawn(move || {
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.add(oc_start * out_len),
                        (oc_end - oc_start) * out_len,
                    )
                };
                conv_transpose1d_range(input, w, oc_start, oc_end, in_len, out_len, out_slice);
            });
        }
    });
    output
}

/// Per output channel: gather contributions from all input channels.
/// Dispatch: AVX2 8-tap kernel when available, scalar oracle otherwise.
/// Both produce bit-identical output (same visit order, mul+add per lane);
/// the golden test below guards this on multi-tile shapes.
#[inline]
fn conv_transpose1d_range(
    input: &[f32], w: &ConvTranspose1dWeight,
    oc_start: usize, oc_end: usize,
    in_len: usize, out_len: usize,
    output: &mut [f32],
) {
    // Bias fill (once per row, both paths — the kernels only accumulate)
    for oc in oc_start..oc_end {
        let local_oc = oc - oc_start;
        for o in 0..out_len {
            output[local_oc * out_len + o] = w.bias[oc];
        }
    }
    #[cfg(target_arch = "x86_64")]
    if crate::simd::has_avx2() {
        unsafe {
            crate::simd::conv_transpose1d_range_avx2(
                input, &w.weight,
                w.in_channels, w.out_channels,
                w.kernel_size, w.stride, w.padding,
                oc_start, oc_end, in_len, out_len, output,
            );
        }
        return;
    }
    conv_transpose1d_range_scalar(input, w, oc_start, oc_end, in_len, out_len, output);
}

#[inline]
fn conv_transpose1d_range_scalar(
    input: &[f32], w: &ConvTranspose1dWeight,
    oc_start: usize, oc_end: usize,
    in_len: usize, out_len: usize,
    output: &mut [f32],
) {
    for oc in oc_start..oc_end {
        let local_oc = oc - oc_start;
        let out_row = local_oc * out_len;
        // Gather from all input channels
        for ic in 0..w.in_channels {
            let in_row = ic * in_len;
            let w_base = ic * w.out_channels * w.kernel_size + oc * w.kernel_size;
            for i in 0..in_len {
                let val = input[in_row + i];
                if val == 0.0 { continue; }
                for k in 0..w.kernel_size {
                    let o_pos_raw = i as isize * w.stride as isize + k as isize - w.padding as isize;
                    if o_pos_raw >= 0 && (o_pos_raw as usize) < out_len {
                        output[out_row + o_pos_raw as usize] += val * w.weight[w_base + k];
                    }
                }
            }
        }
    }
}

// ============================================================
// Depthwise Conv1d (for ConvNeXt)
// ============================================================

pub struct DepthwiseConv1dWeight {
    pub weight: Vec<f32>,  // [channels, 1, kernel_size]
    pub bias: Vec<f32>,    // [channels]
    pub channels: usize,
    pub kernel_size: usize,
    pub padding: usize,
}

/// Depthwise 1D convolution — multi-threaded by channels.
/// Uses causal (left-only) padding: pad = kernel_size - 1.
pub fn depthwise_conv1d(input: &[f32], w: &DepthwiseConv1dWeight) -> Vec<f32> {
    let in_len = input.len() / w.channels;
    let causal_pad = w.kernel_size - 1;
    let out_len = in_len;
    let mut output = vec![0.0f32; w.channels * out_len];

    let ops = w.channels * w.kernel_size * out_len;
    if ops < 500_000 {
        dw_conv1d_range(input, w, 0, w.channels, in_len, out_len, causal_pad, &mut output);
        return output;
    }

    let n_threads = num_threads().min(w.channels);
    let chunk = (w.channels + n_threads - 1) / n_threads;
    let out_ptr = SendPtr(output.as_mut_ptr());

    thread::scope(|s| {
        for tid in 0..n_threads {
            let c_start = tid * chunk;
            let c_end = (c_start + chunk).min(w.channels);
            if c_start >= c_end { break; }
            let ptr = out_ptr;
            s.spawn(move || {
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.add(c_start * out_len),
                        (c_end - c_start) * out_len,
                    )
                };
                dw_conv1d_range(input, w, c_start, c_end, in_len, out_len, causal_pad, out_slice);
            });
        }
    });
    output
}

#[inline]
fn dw_conv1d_range(
    input: &[f32], w: &DepthwiseConv1dWeight,
    c_start: usize, c_end: usize,
    in_len: usize, out_len: usize, causal_pad: usize,
    output: &mut [f32],
) {
    for c in c_start..c_end {
        let local_c = c - c_start;
        for o in 0..out_len {
            let mut sum = w.bias[c];
            for k in 0..w.kernel_size {
                let in_pos = o + k;
                if in_pos >= causal_pad {
                    let idx = in_pos - causal_pad;
                    if idx < in_len {
                        sum += input[c * in_len + idx] * w.weight[c * w.kernel_size + k];
                    }
                }
            }
            output[local_c * out_len + o] = sum;
        }
    }
}

// ============================================================
// SnakeBeta activation
// ============================================================

/// SnakeBeta: x + (1/exp(beta)) * sin^2(exp(alpha) * x)
/// Multi-threaded by channels.
pub fn snake_beta(input: &[f32], alpha: &[f32], beta: &[f32], channels: usize) -> Vec<f32> {
    let t0 = snake_stats_on().then(std::time::Instant::now);
    let length = input.len() / channels;
    let mut output = vec![0.0f32; input.len()];

    if channels * length < 500_000 {
        snake_beta_range(input, alpha, beta, 0, channels, length, &mut output);
        snake_stats_record(t0, input.len() as u64);
        return output;
    }

    let n_threads = num_threads().min(channels);
    let chunk = (channels + n_threads - 1) / n_threads;
    let out_ptr = SendPtr(output.as_mut_ptr());

    thread::scope(|s| {
        for tid in 0..n_threads {
            let c_start = tid * chunk;
            let c_end = (c_start + chunk).min(channels);
            if c_start >= c_end { break; }
            let ptr = out_ptr;
            s.spawn(move || {
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.add(c_start * length),
                        (c_end - c_start) * length,
                    )
                };
                snake_beta_range(input, alpha, beta, c_start, c_end, length, out_slice);
            });
        }
    });
    snake_stats_record(t0, input.len() as u64);
    output
}

// --- snake stats (QORA_SNAKE_STATS=1): call count, total ns, total elements.
// Accumulates only when enabled (OnceLock gate, zero cost otherwise).
static SNAKE_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static SNAKE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SNAKE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SNAKE_ELS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn snake_stats_on() -> bool {
    *SNAKE_ON.get_or_init(|| {
        std::env::var("QORA_SNAKE_STATS").map(|v| v == "1").unwrap_or(false)
    })
}

fn snake_stats_record(t0: Option<std::time::Instant>, els: u64) {
    if let Some(t) = t0 {
        use std::sync::atomic::Ordering::Relaxed;
        SNAKE_CALLS.fetch_add(1, Relaxed);
        SNAKE_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        SNAKE_ELS.fetch_add(els, Relaxed);
    }
}

/// One-line report for the decode split log. Returns None when disabled.
pub fn snake_stats_report() -> Option<String> {
    if !snake_stats_on() {
        return None;
    }
    use std::sync::atomic::Ordering::Relaxed;
    let c = SNAKE_CALLS.load(Relaxed);
    let ns = SNAKE_NS.load(Relaxed);
    let els = SNAKE_ELS.load(Relaxed);
    Some(format!(
        "snake x{c} calls, {:.1}s total, {:.1}M el/call",
        ns as f64 / 1e9,
        els as f64 / c.max(1) as f64 / 1e6
    ))
}

#[inline]
fn snake_beta_range(
    input: &[f32], alpha: &[f32], beta: &[f32],
    c_start: usize, c_end: usize, length: usize,
    output: &mut [f32],
) {
    for c in c_start..c_end {
        let local_c = c - c_start;
        let a = alpha[c].exp();
        let inv_b = (-beta[c]).exp();
        for t in 0..length {
            let x = input[c * length + t];
            let sin_val = (a * x).sin();
            output[local_c * length + t] = x + inv_b * sin_val * sin_val;
        }
    }
}

// ============================================================
// GroupNorm (used in ConvNeXt)
// ============================================================

/// Group normalization.
/// Input/output: [channels, length], with channels split into num_groups.
pub fn group_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    length: usize,
    num_groups: usize,
    eps: f32,
) -> Vec<f32> {
    let channels_per_group = channels / num_groups;
    let mut output = vec![0.0f32; channels * length];

    for g in 0..num_groups {
        let c_start = g * channels_per_group;
        let c_end = c_start + channels_per_group;
        let group_size = channels_per_group * length;

        let mut sum = 0.0f32;
        for c in c_start..c_end {
            for t in 0..length {
                sum += input[c * length + t];
            }
        }
        let mean = sum / group_size as f32;

        let mut var_sum = 0.0f32;
        for c in c_start..c_end {
            for t in 0..length {
                let diff = input[c * length + t] - mean;
                var_sum += diff * diff;
            }
        }
        let inv_std = 1.0 / (var_sum / group_size as f32 + eps).sqrt();

        for c in c_start..c_end {
            for t in 0..length {
                output[c * length + t] = (input[c * length + t] - mean) * inv_std * gamma[c] + beta[c];
            }
        }
    }
    output
}

// ============================================================
// GELU activation (used in ConvNeXt)
// ============================================================

/// GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
#[inline]
pub fn gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + (x * 0.7071067811865476).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xrng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn rvec(state: &mut u64, n: usize, mag: f32) -> Vec<f32> {
        (0..n)
            .map(|_| ((xrng(state) % 2000) as f32 / 1000.0 - 1.0) * mag)
            .collect()
    }


    /// Golden regression for transpose + depthwise (multi-tile shapes).
    /// Values captured from the pre-tiling implementation; tiling must be
    /// bit-exact (output-position reorder only, accumulation order kept).
    #[test]
    fn test_transpose_dw_golden() {
        let mut st = 777u64;
        let inp = rvec(&mut st, 192 * 3000, 1.0);
        let w: Vec<f32> = rvec(&mut st, 192 * 96 * 4, 0.2);
        let b: Vec<f32> = rvec(&mut st, 96, 0.1);
        let tw = ConvTranspose1dWeight {
            weight: w, bias: b, in_channels: 192, out_channels: 96,
            kernel_size: 4, stride: 3, padding: 1,
        };
        let out_len = (3000 - 1) * 3 - 2 * 1 + 4;
        let mut y = vec![0.0f32; 96 * out_len];
        conv_transpose1d_range(&inp, &tw, 0, 96, 3000, out_len, &mut y);
        assert!((y.iter().sum::<f32>() - 10309.475586).abs() < 1.0);
        assert_eq!(y[0].to_bits(), 0.11413957f32.to_bits());
        assert_eq!(y[12345].to_bits(), 0.30676275f32.to_bits());
        assert_eq!(y[y.len() - 1].to_bits(), 0.7365111f32.to_bits());
        // dense strided sweep: accumulation-order changes flip ulps
        // somewhere in 864k outputs (the 3 samples above missed one
        // during development); 64 points make golden sensitive.
        {
            let mut st2 = 777u64;
            let inp0 = rvec(&mut st2, 192 * 3000, 1.0);
            let w0: Vec<f32> = rvec(&mut st2, 192 * 96 * 4, 0.2);
            let b0: Vec<f32> = rvec(&mut st2, 96, 0.1);
            let tw0 = ConvTranspose1dWeight {
                weight: w0, bias: b0, in_channels: 192, out_channels: 96,
                kernel_size: 4, stride: 3, padding: 1,
            };
            let mut refy = vec![0.0f32; 96 * out_len];
            // reference rebuilt with identical code path+seed: sanity that
            // the sweep itself is deterministic (not a golden check yet)
            conv_transpose1d_range(&inp0, &tw0, 0, 96, 3000, out_len, &mut refy);
            assert_eq!(refy, y);
            let step = y.len() / 64;
            let mut acc: u64 = 0;
            for (n, v) in y.iter().enumerate().step_by(step).take(64) {
                acc = acc.wrapping_add((v.to_bits() as u64).wrapping_mul(n as u64 + 1));
            }
            assert_eq!(acc, 59818250823433361u64, "transpose sweep hash");
        }
        let inp2 = rvec(&mut st, 96 * 20000, 1.0);
        let w2 = rvec(&mut st, 96 * 7, 0.2);
        let b2 = rvec(&mut st, 96, 0.1);
        let dw = DepthwiseConv1dWeight {
            weight: w2, bias: b2, channels: 96, kernel_size: 7, padding: 6,
        };
        let mut y2 = vec![0.0f32; 96 * 20000];
        dw_conv1d_range(&inp2, &dw, 0, 96, 20000, 20000, 6, &mut y2);
        assert!((y2.iter().sum::<f32>() - -3754.235596).abs() < 1.0);
        assert_eq!(y2[0].to_bits(), 0.030722003f32.to_bits());
        assert_eq!(y2[765432].to_bits(), (-0.250796f32).to_bits());
        assert_eq!(y2[y2.len() - 1].to_bits(), (-0.10229f32).to_bits());
    }

    fn mkconv(state: &mut u64, oc: usize, ic: usize, k: usize, d: usize) -> Conv1dWeight {
        Conv1dWeight {
            weight: rvec(state, oc * ic * k, 0.5),
            bias: rvec(state, oc, 0.1),
            in_channels: ic,
            out_channels: oc,
            kernel_size: k,
            stride: 1,
            padding: 0,
            dilation: d,
        }
    }

    /// Scalar oracle vs AVX2 range kernel, bit-exact.
    #[cfg(target_arch = "x86_64")]
    fn check_conv(oc: usize, ic: usize, k: usize, d: usize, len: usize, seed: u64) {
        if !crate::simd::has_avx2() {
            eprintln!("no AVX2, skipping");
            return;
        }
        let mut st = seed;
        let w = mkconv(&mut st, oc, ic, k, d);
        let input = rvec(&mut st, ic * len, 1.0);
        let mut expect = vec![0.0f32; oc * len];
        causal_conv1d_range_scalar(&input, &w, 0, oc, len, len, &mut expect);
        let mut got = vec![0.0f32; oc * len];
        unsafe {
            crate::simd::causal_conv1d_range_avx2(
                &input, &w.weight, &w.bias, ic, k, d, 0, oc, len, len, &mut got,
            );
        }
        assert_eq!(got.len(), expect.len());
        for (i, (a, b)) in got.iter().zip(expect.iter()).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "bit mismatch oc={oc} ic={ic} k={k} d={d} len={len} [{i}]: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_causal_avx2_shapes() {
        check_conv(4, 4, 3, 1, 33, 1); // non-8 length: edge paths
        check_conv(8, 16, 7, 1, 100, 2); // vocos-init-like k=7
        check_conv(2, 2, 1, 1, 17, 3); // k=1, zero pad
        check_conv(16, 32, 3, 1, 200, 4); // wider
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_causal_tiled_large() {
        // lengths past the tile size (tile=262144/ic): multi-tile runs and
        // non-multiple tile edges must stay bit-identical across paths.
        check_conv(4, 96, 7, 1, 3000, 21); // 2730+270 edge split
        check_conv(2, 96, 7, 1, 20000, 22); // b3-like, 8 tiles
        check_conv(4, 768, 3, 1, 2000, 23); // b0-like wide channels
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_causal_avx2_dilation() {
        check_conv(4, 8, 3, 2, 64, 11);
        check_conv(4, 4, 3, 4, 65, 12); // odd length + dilation
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_causal_avx2_subrange() {
        // oc sub-range (thread-pool chunking) agrees with scalar sub-range
        if !crate::simd::has_avx2() {
            return;
        }
        let mut st = 21;
        let (oc, ic, k, d, len) = (8, 8, 3, 1, 48);
        let w = mkconv(&mut st, oc, ic, k, d);
        let input = rvec(&mut st, ic * len, 1.0);
        let mut expect = vec![0.0f32; 4 * len];
        causal_conv1d_range_scalar(&input, &w, 2, 6, len, len, &mut expect);
        let mut got = vec![0.0f32; 4 * len];
        unsafe {
            crate::simd::causal_conv1d_range_avx2(
                &input, &w.weight, &w.bias, ic, k, d, 2, 6, len, len, &mut got,
            );
        }
        assert_eq!(got, expect);
    }
    /// FMA variant vs scalar: single-vs-double rounding. Near-zero outputs
    /// amplify relative diffs, so gate on absolute error (< 5e-5, chains of
    /// ~100 terms x ~3e-8 rounding each) plus relative error where |b| > 1.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_causal_fma_tolerance() {
        if !crate::simd::has_avx2_fma() {
            eprintln!("no AVX2+FMA, skipping");
            return;
        }
        for (oc, ic, k, d, len, seed) in
            [(4, 4, 3, 1, 33, 1u64), (8, 16, 7, 1, 100, 2), (4, 8, 3, 2, 64, 11)]
        {
            let mut st = seed;
            let w = mkconv(&mut st, oc, ic, k, d);
            let input = rvec(&mut st, ic * len, 1.0);
            let mut expect = vec![0.0f32; oc * len];
            causal_conv1d_range_scalar(&input, &w, 0, oc, len, len, &mut expect);
            let mut got = vec![0.0f32; oc * len];
            unsafe {
                crate::simd::causal_conv1d_range_avx2_fma(
                    &input, &w.weight, &w.bias, ic, k, d, 0, oc, len, len, &mut got,
                );
            }
            let mut worst_abs = 0.0f32;
            let mut worst_rel = 0.0f32;
            for (a, b) in got.iter().zip(expect.iter()) {
                worst_abs = worst_abs.max((a - b).abs());
                if b.abs() > 1.0 {
                    worst_rel = worst_rel.max((a - b).abs() / b.abs());
                }
            }
            eprintln!("oc={oc} ic={ic} k={k} d={d}: worst abs {worst_abs:.2e} rel {worst_rel:.2e}");
            assert!(worst_abs < 5e-5, "worst abs diff {worst_abs:.2e}");
            assert!(worst_rel < 1e-6, "worst rel diff {worst_rel:.2e}");
        }
    }
}
