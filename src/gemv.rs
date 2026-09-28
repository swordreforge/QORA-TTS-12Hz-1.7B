//! Core compute kernels: GEMV, GEMM, quantization, norms.
//!
//! Supports two weight formats:
//! - **F16**: Half precision. Better quality.
//! - **Q4**: 4-bit symmetric quantization. Faster, lower memory.
//!
//! Q4 uses per-group (32 values) symmetric quantization:
//!   scale = absmax / 7, q = round(val/scale) + 8, packed 2 per byte.
//!   Dequant: val = (q - 8) * scale

use half::f16;
use std::sync::Arc;

// ============================================================
// Weight format types
// ============================================================

pub const Q4_GROUP_SIZE: usize = 32;

pub struct F16Weight {
    pub data: Vec<f16>,
    pub k: usize,
    pub n: usize,
}

pub struct Q4Weight {
    pub packed: Vec<u8>,
    pub scales: Vec<f16>,
    pub k: usize,
    pub n: usize,
}

pub enum Weight {
    F16(F16Weight),
    Q4(Q4Weight),
}

impl Weight {
    pub fn k(&self) -> usize {
        match self { Weight::F16(w) => w.k, Weight::Q4(w) => w.k }
    }
    pub fn n(&self) -> usize {
        match self { Weight::F16(w) => w.n, Weight::Q4(w) => w.n }
    }
    pub fn memory_bytes(&self) -> usize {
        match self {
            Weight::F16(w) => w.data.len() * 2,
            Weight::Q4(w) => w.packed.len() + w.scales.len() * 2,
        }
    }
}

// ============================================================
// Serializable weight data (for save/load)
// ============================================================

pub enum WeightData {
    F16 { data: Vec<f16>, k: usize, n: usize },
    Q4 { packed: Vec<u8>, scales: Vec<f16>, k: usize, n: usize },
}

impl From<WeightData> for Weight {
    fn from(wd: WeightData) -> Self {
        match wd {
            WeightData::F16 { data, k, n } => Weight::F16(F16Weight { data, k, n }),
            WeightData::Q4 { packed, scales, k, n } => Weight::Q4(Q4Weight { packed, scales, k, n }),
        }
    }
}

// ============================================================
// KV cache types
// ============================================================

/// Raw KV cache: per-layer (k_data, v_data, seq_len).
/// Layout: [seq_len, kv_heads, head_dim] token-major, f32.
pub type RawKvCache = Vec<(Vec<f32>, Vec<f32>, usize)>;

pub fn empty_kv_cache(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> RawKvCache {
    (0..num_layers)
        .map(|_| {
            let cap = num_kv_heads * 512 * head_dim;
            (Vec::with_capacity(cap), Vec::with_capacity(cap), 0usize)
        })
        .collect()
}

// ============================================================
// Quantization
// ============================================================

pub fn quantize_f32_to_q4(data: &[f32], k: usize, n: usize) -> Q4Weight {
    debug_assert_eq!(data.len(), k * n);
    debug_assert_eq!(n % Q4_GROUP_SIZE, 0);

    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2;
    let total_groups = k * groups_per_row;

    let mut packed = vec![0u8; total_groups * packed_per_group];
    let mut scales = vec![f16::ZERO; total_groups];

    for ki in 0..k {
        for g in 0..groups_per_row {
            let group_idx = ki * groups_per_row + g;
            let col_start = g * Q4_GROUP_SIZE;

            let mut absmax = 0.0f32;
            for j in 0..Q4_GROUP_SIZE {
                let v = data[ki * n + col_start + j].abs();
                if v > absmax { absmax = v; }
            }

            let scale = absmax / 7.0;
            scales[group_idx] = f16::from_f32(scale);
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

            let pack_start = group_idx * packed_per_group;
            for j in (0..Q4_GROUP_SIZE).step_by(2) {
                let v0 = data[ki * n + col_start + j];
                let v1 = data[ki * n + col_start + j + 1];
                let q0 = ((v0 * inv_scale).round().clamp(-8.0, 7.0) + 8.0) as u8;
                let q1 = ((v1 * inv_scale).round().clamp(-8.0, 7.0) + 8.0) as u8;
                packed[pack_start + j / 2] = q0 | (q1 << 4);
            }
        }
    }

    Q4Weight { packed, scales, k, n }
}

pub fn f32_to_f16(data: &[f32]) -> Vec<f16> {
    data.iter().map(|&v| f16::from_f32(v)).collect()
}

pub fn build_weight(data: &[f32], k: usize, n: usize, use_q4: bool) -> Weight {
    if use_q4 && n % Q4_GROUP_SIZE == 0 {
        Weight::Q4(quantize_f32_to_q4(data, k, n))
    } else {
        Weight::F16(F16Weight { data: f32_to_f16(data), k, n })
    }
}

// ============================================================
// GEMV dispatch (single-token decode)
// ============================================================

#[inline]
pub fn gemv(input: &[f32], weight: &Weight) -> Vec<f32> {
    match weight {
        Weight::F16(w) => gemv_f16(input, w),
        Weight::Q4(w) => gemv_q4(input, w),
    }
}

#[inline]
pub fn gemm(x: &[f32], seq_len: usize, weight: &Weight) -> Vec<f32> {
    match weight {
        Weight::F16(w) => gemm_f16(x, seq_len, w),
        Weight::Q4(w) => gemm_q4(x, seq_len, w),
    }
}

#[inline]
pub fn embed_lookup(weight: &Weight, token_id: usize, hidden: usize) -> Vec<f32> {
    match weight {
        Weight::F16(w) => {
            let row_start = token_id * hidden;
            w.data[row_start..row_start + hidden]
                .iter()
                .map(|v| v.to_f32())
                .collect()
        }
        Weight::Q4(w) => embed_lookup_q4(w, token_id),
    }
}

// ============================================================
// F16 compute kernels
// ============================================================

#[inline]
fn gemv_f16(input: &[f32], weight: &F16Weight) -> Vec<f32> {
    let k = weight.k;
    let n = weight.n;
    let w = &weight.data;

    // AVX-512 fast path
    if crate::simd::has_avx512() {
        return unsafe { crate::simd::gemv_f16_avx512(input, w, k, n) };
    }

    let mut output = vec![0.0f32; n];
    for ki in 0..k {
        let input_val = input[ki];
        let row_start = ki * n;
        for j in 0..n {
            output[j] += input_val * w[row_start + j].to_f32();
        }
    }
    output
}

#[inline]
fn gemm_f16(x: &[f32], seq_len: usize, weight: &F16Weight) -> Vec<f32> {
    let k = weight.k;
    let n = weight.n;
    let w = &weight.data;
    let mut output = vec![0.0f32; seq_len * n];
    for t in 0..seq_len {
        let x_row = &x[t * k..(t + 1) * k];
        let out_row = &mut output[t * n..(t + 1) * n];
        for ki in 0..k {
            let input_val = x_row[ki];
            let w_start = ki * n;
            for j in 0..n {
                out_row[j] += input_val * w[w_start + j].to_f32();
            }
        }
    }
    output
}

// ============================================================
// Q4 compute kernels
// ============================================================

#[inline]
fn gemv_q4_inner(input: &[f32], packed: &[u8], scales: &[f16],
                  _k: usize, n: usize, k_start: usize, k_end: usize) -> Vec<f32> {
    // AVX-512 fast path
    if crate::simd::has_avx512() {
        return unsafe { crate::simd::gemv_q4_avx512(input, packed, scales, n, k_start, k_end) };
    }
    // AVX2 fast path
    if crate::simd::has_avx2() {
        return unsafe { crate::simd::gemv_q4_avx2(input, packed, scales, n, k_start, k_end) };
    }
    gemv_q4_scalar(input, packed, scales, n, k_start, k_end)
}

/// Scalar Q4 GEMV reference. This is the bit-exact oracle for the SIMD
/// kernels: same loop order, same FP op sequence. Differential tests assert
/// `avx2 == scalar` element-wise (not approximate).
#[inline]
fn gemv_q4_scalar(input: &[f32], packed: &[u8], scales: &[f16],
                  n: usize, k_start: usize, k_end: usize) -> Vec<f32> {
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2;
    let packed_per_row = groups_per_row * packed_per_group;

    let mut output = vec![0.0f32; n];

    for ki in k_start..k_end {
        let input_val = input[ki];
        if input_val == 0.0 { continue; }

        let scale_base = ki * groups_per_row;
        let pack_base = ki * packed_per_row;

        for g in 0..groups_per_row {
            let s = scales[scale_base + g].to_f32() * input_val;
            if s == 0.0 { continue; }

            let lut = [
                s * -8.0, s * -7.0, s * -6.0, s * -5.0,
                s * -4.0, s * -3.0, s * -2.0, s * -1.0,
                0.0,      s,        s * 2.0,  s * 3.0,
                s * 4.0,  s * 5.0,  s * 6.0,  s * 7.0,
            ];

            let pack_offset = pack_base + g * packed_per_group;
            let out_offset = g * Q4_GROUP_SIZE;

            for j in 0..packed_per_group {
                let byte = packed[pack_offset + j];
                output[out_offset + j * 2] += lut[(byte & 0x0F) as usize];
                output[out_offset + j * 2 + 1] += lut[(byte >> 4) as usize];
            }
        }
    }

    output
}

#[inline]
fn gemv_q4(input: &[f32], weight: &Q4Weight) -> Vec<f32> {
    let k = weight.k;
    let n = weight.n;

    // Thread budget proportional to work: ~500K MACs per worker keeps
    // dispatch overhead negligible while letting the predictor's ~1M-MAC
    // GEMVs (previously single-threaded under the 4M threshold) parallelize.
    let workers = crate::gpool::num_workers();
    let mut num_threads = (k * n / 500_000).clamp(1, workers);
    if k * n < 500_000 {
        num_threads = 1;
    }

    if num_threads <= 1 {
        return gemv_q4_inner(input, &weight.packed, &weight.scales, k, n, 0, k);
    }

    let chunk_k = (k + num_threads - 1) / num_threads;
    let input_ptr = input.as_ptr() as usize;
    let input_len = input.len();
    let packed_ptr = weight.packed.as_ptr() as usize;
    let packed_len = weight.packed.len();
    let scales_ptr = weight.scales.as_ptr() as usize;
    let scales_len = weight.scales.len();

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let k_start = t * chunk_k;
            let k_end = ((t + 1) * chunk_k).min(k);
            let ip = input_ptr; let il = input_len;
            let pp = packed_ptr; let pl = packed_len;
            let sp = scales_ptr; let sl = scales_len;
            let nn = n; let kk = k;
            crate::gpool::dispatch(move || {
                let inp = unsafe { std::slice::from_raw_parts(ip as *const f32, il) };
                let packed = unsafe { std::slice::from_raw_parts(pp as *const u8, pl) };
                let scales = unsafe { std::slice::from_raw_parts(sp as *const f16, sl) };
                gemv_q4_inner(inp, packed, scales, kk, nn, k_start, k_end)
            })
        })
        .collect();

    let mut output = vec![0.0f32; n];
    for h in handles {
        let partial = h.recv().unwrap();
        for j in 0..n { output[j] += partial[j]; }
    }
    output
}

#[inline]
fn gemm_q4(x: &[f32], seq_len: usize, weight: &Q4Weight) -> Vec<f32> {
    let k = weight.k;
    let n = weight.n;

    let num_threads = if k * n >= 4_000_000 && seq_len > 1 {
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(6)
    } else {
        1
    };

    if num_threads <= 1 || seq_len <= 1 {
        let mut output = vec![0.0f32; seq_len * n];
        for t in 0..seq_len {
            let row = gemv_q4_inner(
                &x[t * k..(t + 1) * k],
                &weight.packed, &weight.scales, k, n, 0, k,
            );
            output[t * n..(t + 1) * n].copy_from_slice(&row);
        }
        return output;
    }

    let chunk = (seq_len + num_threads - 1) / num_threads;
    let x_ptr = x.as_ptr() as usize;
    let x_len = x.len();
    let packed_ptr = weight.packed.as_ptr() as usize;
    let packed_len = weight.packed.len();
    let scales_ptr = weight.scales.as_ptr() as usize;
    let scales_len = weight.scales.len();

    let handles: Vec<_> = (0..num_threads)
        .filter_map(|t| {
            let t_start = t * chunk;
            let t_end = ((t + 1) * chunk).min(seq_len);
            if t_start >= seq_len { return None; }
            let xp = x_ptr; let xl = x_len;
            let pp = packed_ptr; let pl = packed_len;
            let sp = scales_ptr; let sl = scales_len;
            let kk = k; let nn = n;
            Some(std::thread::spawn(move || {
                let xd = unsafe { std::slice::from_raw_parts(xp as *const f32, xl) };
                let packed = unsafe { std::slice::from_raw_parts(pp as *const u8, pl) };
                let scales = unsafe { std::slice::from_raw_parts(sp as *const f16, sl) };
                let count = t_end - t_start;
                let mut partial = vec![0.0f32; count * nn];
                for t in 0..count {
                    let global_t = t_start + t;
                    let x_row = &xd[global_t * kk..(global_t + 1) * kk];
                    let row_out = gemv_q4_inner(x_row, packed, scales, kk, nn, 0, kk);
                    partial[t * nn..(t + 1) * nn].copy_from_slice(&row_out);
                }
                (t_start, partial)
            }))
        })
        .collect();

    let mut output = vec![0.0f32; seq_len * n];
    for h in handles {
        let (t_start, partial) = h.join().unwrap();
        let count = partial.len() / n;
        output[t_start * n..(t_start + count) * n].copy_from_slice(&partial);
    }
    output
}

#[inline]
fn embed_lookup_q4(weight: &Q4Weight, token_id: usize) -> Vec<f32> {
    let n = weight.n;
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2;

    let scale_base = token_id * groups_per_row;
    let pack_base = token_id * groups_per_row * packed_per_group;

    let mut output = vec![0.0f32; n];
    for g in 0..groups_per_row {
        let scale = weight.scales[scale_base + g].to_f32();
        let pack_offset = pack_base + g * packed_per_group;
        let out_offset = g * Q4_GROUP_SIZE;
        for j in 0..packed_per_group {
            let byte = weight.packed[pack_offset + j];
            let q0 = (byte & 0x0F) as i32 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i32 - 8;
            output[out_offset + j * 2] = scale * q0 as f32;
            output[out_offset + j * 2 + 1] = scale * q1 as f32;
        }
    }
    output
}

// ============================================================
// Shared compute kernels
// ============================================================

/// RmsNorm with f16 gamma.
#[inline]
pub fn rms_norm_f16(x: &[f32], gamma: &[f16]) -> Vec<f32> {
    let size = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / size as f32 + 1e-6).sqrt();
    let mut out = vec![0.0f32; size];
    for i in 0..size {
        out[i] = x[i] * inv_rms * gamma[i].to_f32();
    }
    out
}

/// LayerNorm with f16 gamma and beta.
#[inline]
pub fn layer_norm_f16(x: &[f32], gamma: &[f16], beta: &[f16], eps: f32) -> Vec<f32> {
    let size = x.len();
    let mean: f32 = x.iter().sum::<f32>() / size as f32;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / size as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let mut out = vec![0.0f32; size];
    for i in 0..size {
        out[i] = (x[i] - mean) * inv_std * gamma[i].to_f32() + beta[i].to_f32();
    }
    out
}

#[inline]
pub fn softmax_raw(scores: &mut [f32]) {
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum += *s;
    }
    let inv_sum = 1.0 / sum;
    for s in scores.iter_mut() {
        *s *= inv_sum;
    }
}

/// SiLU activation: x * sigmoid(x)
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Parallel lm_head using Weight enum.
#[inline(never)]
pub fn lm_head_parallel(input: &[f32], weight: &Weight, vocab: usize, hidden: usize) -> Vec<f32> {
    match weight {
        Weight::F16(w) => lm_head_parallel_f16(input, &w.data, vocab, hidden),
        Weight::Q4(w) => lm_head_parallel_q4(input, w, vocab, hidden),
    }
}

fn lm_head_parallel_f16(input: &[f32], embed_data: &[f16], vocab: usize, hidden: usize) -> Vec<f32> {
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(6);
    let chunk_size = (vocab + num_threads - 1) / num_threads;

    let input = Arc::new(input.to_vec());
    let embed_ptr = embed_data.as_ptr() as usize;
    let embed_len = embed_data.len();

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let input = Arc::clone(&input);
            let start = t * chunk_size;
            let end = (start + chunk_size).min(vocab);
            let ep = embed_ptr; let el = embed_len; let h = hidden;
            std::thread::spawn(move || {
                let embed = unsafe { std::slice::from_raw_parts(ep as *const f16, el) };
                let mut partial = Vec::with_capacity(end - start);
                for r in start..end {
                    let row = &embed[r * h..(r + 1) * h];
                    let mut sum = 0.0f32;
                    for j in 0..h { sum += input[j] * row[j].to_f32(); }
                    partial.push(sum);
                }
                partial
            })
        })
        .collect();

    let mut output = Vec::with_capacity(vocab);
    for h in handles { output.extend(h.join().unwrap()); }
    output
}

fn lm_head_parallel_q4(input: &[f32], embed: &Q4Weight, vocab: usize, hidden: usize) -> Vec<f32> {
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(6);
    let chunk_size = (vocab + num_threads - 1) / num_threads;

    let input = Arc::new(input.to_vec());
    let packed_ptr = embed.packed.as_ptr() as usize;
    let packed_len = embed.packed.len();
    let scales_ptr = embed.scales.as_ptr() as usize;
    let scales_len = embed.scales.len();

    let groups_per_row = hidden / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2;
    let packed_per_row = groups_per_row * packed_per_group;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let input = Arc::clone(&input);
            let start = t * chunk_size;
            let end = (start + chunk_size).min(vocab);
            let pp = packed_ptr; let pl = packed_len;
            let sp = scales_ptr; let sl = scales_len;
            let gpr = groups_per_row; let ppg = packed_per_group; let ppr = packed_per_row;
            std::thread::spawn(move || {
                let packed = unsafe { std::slice::from_raw_parts(pp as *const u8, pl) };
                let scales = unsafe { std::slice::from_raw_parts(sp as *const f16, sl) };
                let mut partial = Vec::with_capacity(end - start);
                for v in start..end {
                    let mut dot = 0.0f32;
                    let scale_base = v * gpr;
                    let pack_base = v * ppr;
                    for g in 0..gpr {
                        let scale = scales[scale_base + g].to_f32();
                        if scale == 0.0 { continue; }
                        let pack_offset = pack_base + g * ppg;
                        let inp_offset = g * Q4_GROUP_SIZE;
                        let lut = [
                            scale * -8.0, scale * -7.0, scale * -6.0, scale * -5.0,
                            scale * -4.0, scale * -3.0, scale * -2.0, scale * -1.0,
                            0.0,          scale,         scale * 2.0,  scale * 3.0,
                            scale * 4.0,  scale * 5.0,  scale * 6.0,  scale * 7.0,
                        ];
                        for j in 0..ppg {
                            let byte = packed[pack_offset + j];
                            dot += input[inp_offset + j * 2] * lut[(byte & 0x0F) as usize];
                            dot += input[inp_offset + j * 2 + 1] * lut[(byte >> 4) as usize];
                        }
                    }
                    partial.push(dot);
                }
                partial
            })
        })
        .collect();

    let mut output = Vec::with_capacity(vocab);
    for h in handles { output.extend(h.join().unwrap()); }
    output
}

// ============================================================
// I/O helpers (used by save.rs)
// ============================================================

pub fn write_weight_io(w: &mut impl std::io::Write, weight: &Weight) -> std::io::Result<()> {
    match weight {
        Weight::F16(fw) => {
            w.write_all(&(fw.k as u64).to_le_bytes())?;
            w.write_all(&(fw.n as u64).to_le_bytes())?;
            write_f16_vec_io(w, &fw.data)?;
        }
        Weight::Q4(qw) => {
            w.write_all(&(qw.k as u64).to_le_bytes())?;
            w.write_all(&(qw.n as u64).to_le_bytes())?;
            write_bytes_io(w, &qw.packed)?;
            write_f16_vec_io(w, &qw.scales)?;
        }
    }
    Ok(())
}

pub fn write_f16_vec_io(w: &mut impl std::io::Write, data: &[f16]) -> std::io::Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
    };
    w.write_all(&(data.len() as u64).to_le_bytes())?;
    w.write_all(bytes)
}

pub fn write_f32_vec_io(w: &mut impl std::io::Write, data: &[f32]) -> std::io::Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    w.write_all(&(data.len() as u64).to_le_bytes())?;
    w.write_all(bytes)
}

pub fn write_bytes_io(w: &mut impl std::io::Write, data: &[u8]) -> std::io::Result<()> {
    w.write_all(&(data.len() as u64).to_le_bytes())?;
    w.write_all(data)
}

pub fn read_u32_io(r: &mut impl std::io::Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; Ok(u32::from_le_bytes(buf))
}

pub fn read_u64_io(r: &mut impl std::io::Read) -> std::io::Result<u64> {
    let mut buf = [0u8; 8]; r.read_exact(&mut buf)?; Ok(u64::from_le_bytes(buf))
}

pub fn read_u8_io(r: &mut impl std::io::Read) -> std::io::Result<u8> {
    let mut buf = [0u8; 1]; r.read_exact(&mut buf)?; Ok(buf[0])
}

pub fn read_f16_vec_io(r: &mut impl std::io::Read) -> std::io::Result<Vec<f16>> {
    let len = read_u64_io(r)? as usize;
    let mut bytes = vec![0u8; len * 2];
    r.read_exact(&mut bytes)?;
    let data = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f16, len).to_vec()
    };
    Ok(data)
}

pub fn read_f32_vec_io(r: &mut impl std::io::Read) -> std::io::Result<Vec<f32>> {
    let len = read_u64_io(r)? as usize;
    let mut bytes = vec![0u8; len * 4];
    r.read_exact(&mut bytes)?;
    let data = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec()
    };
    Ok(data)
}

pub fn read_bytes_io(r: &mut impl std::io::Read) -> std::io::Result<Vec<u8>> {
    let len = read_u64_io(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

pub fn read_weight_io(r: &mut impl std::io::Read, format_id: u8) -> std::io::Result<Weight> {
    let k = read_u64_io(r)? as usize;
    let n = read_u64_io(r)? as usize;
    match format_id {
        0 => {
            let data = read_f16_vec_io(r)?;
            Ok(Weight::F16(F16Weight { data, k, n }))
        }
        1 => {
            let packed = read_bytes_io(r)?;
            let scales = read_f16_vec_io(r)?;
            Ok(Weight::Q4(Q4Weight { packed, scales, k, n }))
        }
        _ => Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unknown format"))
    }
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

    fn rbytes(state: &mut u64, n: usize) -> Vec<u8> {
        (0..n).map(|_| (xrng(state) & 0xFF) as u8).collect()
    }

    fn rscales(state: &mut u64, n: usize, zero_frac: f64) -> Vec<f16> {
        (0..n)
            .map(|_| {
                let r = (xrng(state) % 1000) as f64 / 1000.0;
                if r < zero_frac {
                    f16::from_f32(0.0)
                } else {
                    // small magnitudes incl. negatives and subnormals-adjacent
                    let v = ((xrng(state) % 2000) as f32 / 1000.0 - 1.0) * 0.05;
                    f16::from_f32(v)
                }
            })
            .collect()
    }

    fn rinput(state: &mut u64, n: usize, zero_frac: f64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let r = (xrng(state) % 1000) as f64 / 1000.0;
                if r < zero_frac {
                    0.0
                } else {
                    ((xrng(state) % 2000) as f32 / 1000.0 - 1.0) * 2.0
                }
            })
            .collect()
    }

    /// Run scalar oracle vs AVX2 kernel on one synthetic case, assert
    /// bit-exact equality (same FP op sequence by construction).
    #[cfg(target_arch = "x86_64")]
    fn check_case(k: usize, n: usize, seed: u64, in_zero: f64, sc_zero: f64) {
        if !crate::simd::has_avx2() {
            eprintln!("no AVX2, skipping differential test");
            return;
        }
        let mut st = seed;
        let groups = n / Q4_GROUP_SIZE;
        let packed = rbytes(&mut st, k * groups * (Q4_GROUP_SIZE / 2));
        let scales = rscales(&mut st, k * groups, sc_zero);
        let input = rinput(&mut st, k, in_zero);
        let expect = gemv_q4_scalar(&input, &packed, &scales, n, 0, k);
        let got = unsafe { crate::simd::gemv_q4_avx2(&input, &packed, &scales, n, 0, k) };
        assert_eq!(got.len(), expect.len(), "len k={k} n={n}");
        for (i, (a, b)) in got.iter().zip(expect.iter()).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "bit mismatch k={k} n={n} seed={seed} [{i}]: got {a:?} want {b:?}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_matches_scalar_shapes() {
        // talker q (2048x2048) and mlp (2048x6144) shapes, predictor 1024x1024
        check_case(64, 2048, 1, 0.0, 0.0);
        check_case(64, 6144, 2, 0.0, 0.0);
        check_case(32, 1024, 3, 0.0, 0.0);
        check_case(16, 256, 4, 0.0, 0.0);
        check_case(1, 32, 5, 0.0, 0.0); // single row, single group
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_matches_scalar_sparse() {
        // zero-skipping paths (input==0, scale==0), incl. negative scales
        check_case(48, 1024, 11, 0.3, 0.2);
        check_case(32, 2048, 12, 0.5, 0.5);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_subrange_matches() {
        // k-chunk path used by the thread pool must agree per chunk
        if !crate::simd::has_avx2() {
            return;
        }
        let mut st = 21;
        let (k, n) = (64, 1024);
        let groups = n / Q4_GROUP_SIZE;
        let packed = rbytes(&mut st, k * groups * (Q4_GROUP_SIZE / 2));
        let scales = rscales(&mut st, k * groups, 0.1);
        let input = rinput(&mut st, k, 0.1);
        let full = gemv_q4_scalar(&input, &packed, &scales, n, 0, k);
        let mut acc = vec![0.0f32; n];
        for (a, b) in [(0, 16), (16, 48), (48, 64)] {
            let part = unsafe { crate::simd::gemv_q4_avx2(&input, &packed, &scales, n, a, b) };
            for j in 0..n {
                acc[j] += part[j];
            }
        }
        // NOTE: chunked accumulation order differs from single-pass, so compare
        // against chunked scalar accumulation (same op order), not `full`.
        let mut acc_ref = vec![0.0f32; n];
        for (a, b) in [(0, 16), (16, 48), (48, 64)] {
            let part = gemv_q4_scalar(&input, &packed, &scales, n, a, b);
            for j in 0..n {
                acc_ref[j] += part[j];
            }
        }
        assert_eq!(acc, acc_ref);
        // sanity: chunked ≈ full within float tolerance
        for (a, b) in acc.iter().zip(full.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_tail_n() {
        // n not a multiple of 32: full groups agree, remainder stays zero
        if !crate::simd::has_avx2() {
            return;
        }
        let mut st = 31;
        let (k, n) = (16, 100); // 3 full groups (96) + 4 untouched
        let groups = n / Q4_GROUP_SIZE;
        let packed = rbytes(&mut st, k * groups * (Q4_GROUP_SIZE / 2));
        let scales = rscales(&mut st, k * groups, 0.0);
        let input = rinput(&mut st, k, 0.0);
        let expect = gemv_q4_scalar(&input, &packed, &scales, n, 0, k);
        let got = unsafe { crate::simd::gemv_q4_avx2(&input, &packed, &scales, n, 0, k) };
        assert_eq!(got, expect);
        assert!(got[96..].iter().all(|&v| v == 0.0));
    }
}
