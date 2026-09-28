//! AVX-512 SIMD kernels for QORA-TTS CPU inference.
//!
//! Optimized GEMV for Q4 and F16 weights.
//! Falls back to scalar code on non-AVX-512 CPUs (dispatch in gemv.rs).

// All functions in this module are `unsafe fn` wrapping SIMD intrinsics.
// Every operation inside them is inherently unsafe, so suppress the per-op warning.
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

const Q4_GROUP_SIZE: usize = 32;

/// Check if AVX-512F is available at runtime.
pub fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Check if AVX2 is available at runtime.
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Check if AVX2+ FMA are available at runtime.
pub fn has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ============================================================
// Q4 GEMV — AVX2 (8-wide)
// ============================================================

// ============================================================
// Q4 GEMV — AVX2 (8-wide)
// ============================================================

/// AVX2 Q4 GEMV inner kernel. Bit-exact mirror of the scalar reference
/// (`gemv_q4_scalar` in gemv.rs): same loop order, same FP op sequence
/// (`lut[i] = s * (i - 8)` per lane, k-major `output[j] += …` accumulation).
///
/// Per group of 32 outputs, two 8-wide iterations handle 8 packed bytes
/// (16 values) each. The 16-entry LUT is split in halves for
/// `_mm256_permutevar8x32_ps` (which indexes modulo 8); lanes with
/// `idx > 7` blend in the high half. Out-of-range indices never fault
/// (hardware masks to 3 bits); the blend mask selects the correct half.
///
/// Only full groups are vectorized (`groups_per_row = n / 32`); a short tail
/// (n not a multiple of 32) is left as zeros, exactly like the scalar path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn gemv_q4_avx2(
    input: &[f32],
    packed: &[u8],
    scales: &[f16],
    n: usize,
    k_start: usize,
    k_end: usize,
) -> Vec<f32> {
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2; // 16
    let packed_per_row = groups_per_row * packed_per_group;

    let mut output = vec![0.0f32; n];

    // LUT factor halves: (q - 8) for q = 0..8 and 8..16
    let factors_lo = _mm256_loadu_ps(Q4_FACTORS.as_ptr());
    let factors_hi = _mm256_loadu_ps(Q4_FACTORS.as_ptr().add(8));
    let nibble_mask = _mm256_set1_epi32(0x0F);
    let seven = _mm256_set1_epi32(7);

    for ki in k_start..k_end {
        let val = input[ki];
        if val == 0.0 {
            continue;
        }

        let scale_base = ki * groups_per_row;
        let pack_base = ki * packed_per_row;

        for g in 0..groups_per_row {
            let s = scales[scale_base + g].to_f32() * val;
            if s == 0.0 {
                continue;
            }

            // Build LUT halves: lut[i] = s * (i - 8)
            let s_vec = _mm256_set1_ps(s);
            let lut0 = _mm256_mul_ps(factors_lo, s_vec);
            let lut1 = _mm256_mul_ps(factors_hi, s_vec);

            // 16 packed bytes cover 32 outputs; process 8 bytes (16 outs) at a time
            let po = pack_base + g * packed_per_group;
            let bytes = _mm_loadu_si128(packed.as_ptr().add(po) as *const __m128i);

            for h in 0..2 {
                let half = if h == 0 { bytes } else { _mm_srli_si128::<8>(bytes) };
                // 8 bytes -> 8 u32 lanes
                let v8 = _mm256_cvtepu8_epi32(half);
                let lo = _mm256_and_si256(v8, nibble_mask);
                let hi = _mm256_srli_epi32::<4>(v8);

                // 16-entry lookup via two 8-entry permutes + blend on idx > 7
                let lo0 = _mm256_permutevar8x32_ps(lut0, lo);
                let lo1 = _mm256_permutevar8x32_ps(lut1, lo);
                let lo_m = _mm256_cmpgt_epi32(lo, seven);
                let lo_v = _mm256_blendv_ps(lo0, lo1, _mm256_castsi256_ps(lo_m));

                let hi0 = _mm256_permutevar8x32_ps(lut0, hi);
                let hi1 = _mm256_permutevar8x32_ps(lut1, hi);
                let hi_m = _mm256_cmpgt_epi32(hi, seven);
                let hi_v = _mm256_blendv_ps(hi0, hi1, _mm256_castsi256_ps(hi_m));

                // Interleave: out[2i] = lo[i], out[2i+1] = hi[i].
                // NOTE: unpack is lane-local (128-bit lanes), so a lane
                // cross via permute2f128 is required for linear order.
                let a = _mm256_unpacklo_ps(lo_v, hi_v);
                let b = _mm256_unpackhi_ps(lo_v, hi_v);
                let first = _mm256_permute2f128_ps(a, b, 0x20);
                let second = _mm256_permute2f128_ps(a, b, 0x31);

                let oo = g * Q4_GROUP_SIZE + h * 16;
                let acc1 = _mm256_loadu_ps(output.as_ptr().add(oo));
                _mm256_storeu_ps(output.as_mut_ptr().add(oo), _mm256_add_ps(acc1, first));
                let acc2 = _mm256_loadu_ps(output.as_ptr().add(oo + 8));
                _mm256_storeu_ps(
                    output.as_mut_ptr().add(oo + 8),
                    _mm256_add_ps(acc2, second),
                );
            }
        }
    }

    output
}

// ============================================================
// AVX-512 helper
// ============================================================

/// Horizontal sum of 16 f32 lanes -> scalar f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn hsum_ps(v: __m512) -> f32 {
    // Fold 512 -> 128 via extractf32x4
    let a = _mm512_extractf32x4_ps(v, 0);
    let b = _mm512_extractf32x4_ps(v, 1);
    let c = _mm512_extractf32x4_ps(v, 2);
    let d = _mm512_extractf32x4_ps(v, 3);
    let sum = _mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(c, d));
    // Fold 128 -> scalar
    let hi = _mm_movehl_ps(sum, sum);
    let sum2 = _mm_add_ps(sum, hi);
    let hi2 = _mm_shuffle_ps(sum2, sum2, 1);
    let sum3 = _mm_add_ss(sum2, hi2);
    _mm_cvtss_f32(sum3)
}

// Suppress unused warning on non-x86_64
#[cfg(target_arch = "x86_64")]
const _: () = { let _ = hsum_ps; };

// ============================================================
// Q4 GEMV — AVX-512
// ============================================================

/// Q4 dequant factors: (q - 8) for q = 0..15
static Q4_FACTORS: [f32; 16] = [
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
];

/// Interleave indices for lo/hi nibble -> contiguous output (first 16 of 32).
static INTERLEAVE_FIRST: [i32; 16] = [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23];
/// Interleave indices for lo/hi nibble -> contiguous output (second 16 of 32).
static INTERLEAVE_SECOND: [i32; 16] = [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31];

/// AVX-512 Q4 GEMV inner kernel. Replaces gemv_q4_inner.
///
/// Processes k_start..k_end rows of the weight matrix, accumulating into output[0..n].
/// Uses permutexvar for 16-entry LUT lookup of dequantized Q4 nibbles.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemv_q4_avx512(
    input: &[f32],
    packed: &[u8],
    scales: &[f16],
    n: usize,
    k_start: usize,
    k_end: usize,
) -> Vec<f32> {
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2; // 16
    let packed_per_row = groups_per_row * packed_per_group;

    let mut output = vec![0.0f32; n];

    // Load constant vectors
    let factors = _mm512_loadu_ps(Q4_FACTORS.as_ptr());
    let interleave_lo = _mm512_loadu_si512(INTERLEAVE_FIRST.as_ptr() as *const _);
    let interleave_hi = _mm512_loadu_si512(INTERLEAVE_SECOND.as_ptr() as *const _);
    let nibble_mask = _mm512_set1_epi32(0x0F);

    for ki in k_start..k_end {
        let val = input[ki];
        if val == 0.0 {
            continue;
        }

        let scale_base = ki * groups_per_row;
        let pack_base = ki * packed_per_row;

        for g in 0..groups_per_row {
            let s = scales[scale_base + g].to_f32() * val;
            if s == 0.0 {
                continue;
            }

            // Build LUT: lut[i] = s * (i - 8)
            let s_vec = _mm512_set1_ps(s);
            let lut = _mm512_mul_ps(factors, s_vec);

            // Load 16 packed bytes and zero-extend to 16 x i32
            let po = pack_base + g * packed_per_group;
            let bytes = _mm_loadu_si128(packed.as_ptr().add(po) as *const __m128i);
            let bytes_i32 = _mm512_cvtepu8_epi32(bytes);

            // Extract nibbles
            let lo_nib = _mm512_and_epi32(bytes_i32, nibble_mask);
            let hi_nib = _mm512_srli_epi32(bytes_i32, 4);

            // LUT lookup: 16 values for even positions, 16 for odd
            let lo_vals = _mm512_permutexvar_ps(lo_nib, lut);
            let hi_vals = _mm512_permutexvar_ps(hi_nib, lut);

            // Interleave to get 32 output values in contiguous order
            let first_16 = _mm512_permutex2var_ps(lo_vals, interleave_lo, hi_vals);
            let second_16 = _mm512_permutex2var_ps(lo_vals, interleave_hi, hi_vals);

            // Accumulate into output
            let oo = g * Q4_GROUP_SIZE;
            let acc1 = _mm512_loadu_ps(output.as_ptr().add(oo));
            let acc2 = _mm512_loadu_ps(output.as_ptr().add(oo + 16));
            _mm512_storeu_ps(output.as_mut_ptr().add(oo), _mm512_add_ps(acc1, first_16));
            _mm512_storeu_ps(output.as_mut_ptr().add(oo + 16), _mm512_add_ps(acc2, second_16));
        }
    }

    output
}

// ============================================================
// Causal Conv1d range — AVX2 (8-wide over output time)
// ============================================================

/// AVX2 causal Conv1d over output channels `[oc_start, oc_end)`.
/// Bit-exact mirror of `causal_conv1d_range_scalar` (mul+add only, no FMA):
/// bias fill, then per (oc, ic, k) `out[o] += in[o+off] * w` with the same
/// (ic, k) accumulation order per output lane. Boundary taps that fall
/// outside `[0, in_len)` stay scalar; the 8-aligned interior is vectorized.
/// Weight/bias indexed by global `oc`, output rows by `oc - oc_start`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn causal_conv1d_range_avx2(
    input: &[f32],
    weight: &[f32], // [oc_total, in_ch, ksize] row-major
    bias: &[f32],
    in_ch: usize,
    ksize: usize,
    dilation: usize,
    oc_start: usize,
    oc_end: usize,
    in_len: usize,
    out_len: usize,
    output: &mut [f32],
) {
    causal_range_exact(
        input, weight, bias, in_ch, ksize, dilation,
        oc_start, oc_end, in_len, out_len, output,
    )
}

/// FMA variant of the kernel above: `out += x * w` fused into one rounding.
/// ~10-15% faster (4 ops vs 5 per 8 outputs) but NOT bit-identical to scalar
/// (single vs double rounding, ~1e-7 relative). Gated by `QORA_FMA=1`;
/// default stays exact. See tolerance test below.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn causal_conv1d_range_avx2_fma(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_ch: usize,
    ksize: usize,
    dilation: usize,
    oc_start: usize,
    oc_end: usize,
    in_len: usize,
    out_len: usize,
    output: &mut [f32],
) {
    causal_range_fma(
        input, weight, bias, in_ch, ksize, dilation,
        oc_start, oc_end, in_len, out_len, output,
    )
}

/// Shared loop body for both kernel variants. `$sum(acc, x, wb)` computes the
/// 8-lane update; exact passes mul+add, FMA passes fmadd. Macro (not generic)
/// so each intrinsic resolves under its wrapper's own `target_feature`.
#[cfg(target_arch = "x86_64")]
macro_rules! causal_range_body {
    ($input:expr, $weight:expr, $bias:expr, $in_ch:expr, $ksize:expr,
     $dilation:expr, $oc_start:expr, $oc_end:expr, $in_len:expr, $out_len:expr,
     $output:expr, $sum:expr) => {{
        let causal_pad = ($ksize - 1) * $dilation;
        let ick = $in_ch * $ksize;

        for oc in $oc_start..$oc_end {
            let local_oc = oc - $oc_start;
            let out_row = &mut $output[local_oc * $out_len..(local_oc + 1) * $out_len];
            out_row.fill($bias[oc]);

            let w_base = oc * ick;
            for ic in 0..$in_ch {
                let in_base = ic * $in_len;
                let w_row = w_base + ic * $ksize;
                for k in 0..$ksize {
                    let wk = $weight[w_row + k];
                    let off = k as isize * $dilation as isize - causal_pad as isize;
                    let lo = (0isize).max(-off);
                    let hi = ($out_len as isize).min($in_len as isize - off);
                    if hi <= lo {
                        continue;
                    }
                    let (lo, hi) = (lo as usize, hi as usize);
                    let vs = (lo + 7) / 8 * 8;
                    let ve = hi / 8 * 8;
                    for o in lo..vs.min(hi) {
                        let idx = in_base + ((o as isize + off) as usize);
                        out_row[o] += $input[idx] * wk;
                    }
                    if ve > vs {
                        debug_assert!(ve <= $out_len);
                        debug_assert!(vs as isize + off >= 0);
                        debug_assert!(ve as isize + off <= $in_len as isize);
                        let wb = _mm256_set1_ps(wk);
                        let combine = $sum;
                        let mut o = vs;
                        while o < ve {
                            let acc = _mm256_loadu_ps(out_row.as_ptr().add(o));
                            let x = _mm256_loadu_ps(
                                $input.as_ptr().add(in_base + ((o as isize + off) as usize)),
                            );
                            _mm256_storeu_ps(out_row.as_mut_ptr().add(o), combine(acc, x, wb));
                            o += 8;
                        }
                    }
                    for o in ve.max(vs)..hi {
                        let idx = in_base + ((o as isize + off) as usize);
                        out_row[o] += $input[idx] * wk;
                    }
                }
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn causal_range_exact(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_ch: usize,
    ksize: usize,
    dilation: usize,
    oc_start: usize,
    oc_end: usize,
    in_len: usize,
    out_len: usize,
    output: &mut [f32],
) {
    causal_range_body!(
        input, weight, bias, in_ch, ksize, dilation,
        oc_start, oc_end, in_len, out_len, output,
        (|acc: __m256, x: __m256, wb: __m256| _mm256_add_ps(acc, _mm256_mul_ps(x, wb)))
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn causal_range_fma(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_ch: usize,
    ksize: usize,
    dilation: usize,
    oc_start: usize,
    oc_end: usize,
    in_len: usize,
    out_len: usize,
    output: &mut [f32],
) {
    causal_range_body!(
        input, weight, bias, in_ch, ksize, dilation,
        oc_start, oc_end, in_len, out_len, output,
        (|acc: __m256, x: __m256, wb: __m256| _mm256_fmadd_ps(x, wb, acc))
    );
}

// ============================================================
// F16 GEMV — AVX-512
// ============================================================

/// AVX-512 F16 GEMV: input[k] @ weight[k,n] -> output[n].
/// Uses _mm512_cvtph_ps for f16->f32 and FMA for accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemv_f16_avx512(
    input: &[f32],
    weight_data: &[f16],
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; n];
    let n16 = n / 16 * 16; // aligned to 16

    for ki in 0..k {
        let val = input[ki];
        if val == 0.0 {
            continue;
        }
        let val_vec = _mm512_set1_ps(val);
        let row = ki * n;

        // Process 16 elements at a time
        let mut j = 0usize;
        while j < n16 {
            // Load 16 x f16 as raw bits in __m256i, convert to __m512 f32
            let w_f16 = _mm256_loadu_si256(weight_data.as_ptr().add(row + j) as *const __m256i);
            let w_f32 = _mm512_cvtph_ps(w_f16);
            let acc = _mm512_loadu_ps(output.as_ptr().add(j));
            let result = _mm512_fmadd_ps(val_vec, w_f32, acc);
            _mm512_storeu_ps(output.as_mut_ptr().add(j), result);
            j += 16;
        }

        // Scalar tail
        while j < n {
            output[j] += val * weight_data[row + j].to_f32();
            j += 1;
        }
    }

    output
}
