//! AVX-512 SIMD kernels for QORA-TTS CPU inference.
//!
//! Optimized GEMV for Q4 and F16 weights.
//! Falls back to scalar code on non-AVX-512 CPUs (dispatch in gemv.rs).

// All functions in this module are `unsafe fn` wrapping SIMD intrinsics.
// Every operation inside them is inherently unsafe, so suppress the per-op warning.
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[allow(unused_imports)] // non-x86_64 cfgs out most fns using this (CI hygiene)
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

    // Integer-arithmetic LUT: dequant value = (nibble - 8) * s, computed as
    // cvt(nibble) -> sub(8.0) -> mul(s). Bit-identical to the split-table
    // permutevar version: cvt/sub are exact for 0..15, and IEEE mul commutes
    // bitwise (a*b == b*a), so (i-8)*s == s*(i-8) lane for lane. This removes
    // 4 permutevar + 2 cmpgt + 2 blendv per 16 outputs (all port-5 pressure
    // on Intel) at the cost of 2 cvt + 2 sub + 2 mul on spread ports.
    let nibble_mask = _mm256_set1_epi32(0x0F);
    let eight = _mm256_set1_ps(8.0);

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

            // Build once per group: broadcast scale (LUT folded into lanes below)
            let s_vec = _mm256_set1_ps(s);

            // 16 packed bytes cover 32 outputs; process 8 bytes (16 outs) at a time
            let po = pack_base + g * packed_per_group;
            let bytes = _mm_loadu_si128(packed.as_ptr().add(po) as *const __m128i);

            for h in 0..2 {
                let half = if h == 0 { bytes } else { _mm_srli_si128::<8>(bytes) };
                // 8 bytes -> 8 u32 lanes
                let v8 = _mm256_cvtepu8_epi32(half);
                let lo = _mm256_and_si256(v8, nibble_mask);
                let hi = _mm256_srli_epi32::<4>(v8);

                // (nibble - 8) * s per lane; bit-identical to LUT (see above)
                let lo_v = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(lo), eight), s_vec);
                let hi_v = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(hi), eight), s_vec);

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

            // Length tiling: keep one tile's input window
            // (~tile*in_ch floats ≈ 1MB) LLC/L2-resident. The untiled code
            // streams input rows strided by in_len (up to 1.8MB @462k),
            // thrashing LLC on every (ic,k) pass. Tiling only reorders
            // OUTPUT positions; per-lane (ic,k) accumulation is untouched,
            // so this is bit-exact (scalar/vector split points may shift,
            // but both paths do the same per-lane mul+add sequence).
            let tile = (($out_len.min(262144 / $in_ch.max(1))).clamp(128, 16384)).max(1);
            let mut ts = 0;
            while ts < $out_len {
                let te = (ts + tile).min($out_len);
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
                    let (lo, hi) = ((lo as usize).max(ts), (hi as usize).min(te));
                    if hi <= lo {
                        continue;
                    }
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
                ts = te;
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
        |acc: __m256, x: __m256, wb: __m256| _mm256_add_ps(acc, _mm256_mul_ps(x, wb))
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
        |acc: __m256, x: __m256, wb: __m256| _mm256_fmadd_ps(x, wb, acc)
    );
}

// ============================================================
// ConvTranspose1d range — AVX2 (8-wide over kernel taps)
// ============================================================

/// AVX2 transposed-conv accumulation over output channels
/// `[oc_start, oc_end)`. Bit-exact mirror of `conv_transpose1d_range`
/// (scalar): same visit order (oc → ic → i → ascending k), bias NOT
/// touched here (caller fills once). For fixed (ic,i) the k taps hit
/// CONSECUTIVE outputs, so an 8-tap chunk is one vector load (weights) +
/// one vector load/mul-add/store (outputs); tail taps stay scalar.
/// Bounds are hoisted per i (klo/khi) instead of per (i,k).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn conv_transpose1d_range_avx2(
    input: &[f32],
    weight: &[f32], // [in_ch, out_ch_total, ksize] row-major
    in_ch: usize,
    out_ch_total: usize,
    ksize: usize,
    stride: usize,
    padding: usize,
    oc_start: usize,
    oc_end: usize,
    in_len: usize,
    out_len: usize,
    output: &mut [f32],
) {
    let pad = padding as isize;
    let stride = stride as isize;
    for oc in oc_start..oc_end {
        let local_oc = oc - oc_start;
        let out_base = local_oc * out_len;
        for ic in 0..in_ch {
            let in_row = ic * in_len;
            let w_base = ic * out_ch_total * ksize + oc * ksize;
            for i in 0..in_len {
                let val = *input.get_unchecked(in_row + i);
                if val == 0.0 {
                    continue;
                }
                // valid tap range for this i (hoisted, was per-(i,k) branch)
                let klo = (0isize).max(pad - i as isize * stride);
                let khi = (ksize as isize).min(out_len as isize + pad - i as isize * stride);
                if khi <= klo {
                    continue;
                }
                let (klo, khi) = (klo as usize, khi as usize);
                let vb = _mm256_set1_ps(val);
                // 8-tap vector chunks (ascending k, same per-lane order)
                let mut kb = klo;
                while kb + 8 <= khi {
                    let o0 = (i as isize * stride + kb as isize - pad) as usize;
                    let wv = _mm256_loadu_ps(weight.as_ptr().add(w_base + kb));
                    let acc = _mm256_loadu_ps(output.as_ptr().add(out_base + o0));
                    let res = _mm256_add_ps(acc, _mm256_mul_ps(vb, wv));
                    _mm256_storeu_ps(output.as_mut_ptr().add(out_base + o0), res);
                    kb += 8;
                }
                // tail taps scalar (k < 8 kernels live entirely here)
                for k in kb..khi {
                    let o = (i as isize * stride + k as isize - pad) as usize;
                    let out_ptr = output.as_mut_ptr().add(out_base + o);
                    *out_ptr += val * *weight.get_unchecked(w_base + k);
                }
            }
        }
    }
}

// ============================================================
// F32 GEMM — AVX2 (8-wide over n)
// ============================================================

/// AVX2 row-major GEMM micro-block over rows `[m0, m1)`.
/// C[m,n] += A[m,k]·B[k,n] with k ascending per element — same per-lane
/// order as scalar `f32_gemv`, so bit-exact. Bias NOT handled here
/// (caller pre-fills C rows); n-tail scalar.
/// `c` covers exactly rows m0..m1 (C indexing is m0-relative).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn f32_gemm_block_avx2(
    a: &[f32],
    b: &[f32],
    m0: usize,
    m1: usize,
    n: usize,
    kdim: usize,
    c: &mut [f32],
) {
    for m in m0..m1 {
        let a_row = m * kdim;
        let c_row = (m - m0) * n;
        for k in 0..kdim {
            let v = *a.get_unchecked(a_row + k);
            // Zero-skip: matches legacy per-t f32_gemv_bias EXACTLY, including
            // the -0.0 edge (skip keeps -0.0, adding +0.0 would flip it to +0.0).
            if v == 0.0 {
                continue;
            }
            let av = _mm256_set1_ps(v);
            let b_row = k * n;
            let mut j = 0;
            while j + 8 <= n {
                let cv = _mm256_loadu_ps(c.as_ptr().add(c_row + j));
                let bv = _mm256_loadu_ps(b.as_ptr().add(b_row + j));
                _mm256_storeu_ps(
                    c.as_mut_ptr().add(c_row + j),
                    _mm256_add_ps(cv, _mm256_mul_ps(av, bv)),
                );
                j += 8;
            }
            while j < n {
                let c_ptr = c.as_mut_ptr().add(c_row + j);
                *c_ptr += v * *b.get_unchecked(b_row + j);
                j += 1;
            }
        }
    }
}

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
