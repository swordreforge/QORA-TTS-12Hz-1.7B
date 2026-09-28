# AVX2 Q4 kernel 实施方案（✅ 已实现，2026-09-28）

> 背景：`perf` 实测 generation 98% 周期烧在标量 `gemv_q4_inner`；
> 本机（Ultra 7 155H）无 AVX-512，`simd.rs` 的加速分支永远进不去。
> 目标：8-wide AVX2 kernel，把单线程吞吐翻 ~2x，再与线程池乘法叠加。
>
> 结果：generation 41.0s → 25.9s（0.76s/frame），total 58.1s → 42.1s；
> 同 seed sha256 与之前三版完全一致。实现与本方案一致，两处 Abweichungen：
> 门控取纯 `avx2`（未绑 fma，kernel 内只用 mul/add）；n 非 32 倍数的尾巴
> 与标量版同样留零（不另行处理）。

---

## 1. 不变量（动不得）

- Q4 行布局：每行按 group=32 切分，每组 16 packed bytes（低 nibble→偶下标，
  高 nibble→奇下标）+ 1 个 f16 scale；反量化 `(q-8)*scale`。
- 循环顺序：k-major（逐行），`output[j] += lut[...]` 累加；`input==0` / `s==0` 跳过。
- 标量 LUT 语义：`[s*-8.0, …, s*-1.0, 0.0, s, s*2.0, …, s*7.0]`（注意下标 8 是字面量
  `0.0`，下标 9 是 `s` 本身）。

## 2. AVX2 算法（与现有 AVX-512 kernel 镜像）

每个 group（32 输出）分两次 8-wide 迭代，每次处理 8 packed bytes → 16 输出：

1. `_mm_loadu_si128` 读 16 bytes（一次读一组，复用两次 8 字节半）。
2. `_mm256_cvtepu8_epi32`：8 bytes → 8×i32；`and 0x0F` 取低 nibble，`srli 4` 取高 nibble。
3. **16-entry LUT 的 8-wide 查表**：把 LUT 拆成 `lut0[0..8]` / `lut1[8..16]`，
   两次 `_mm256_permutevar8x32_ps` + 按 `idx>=8` blend。
   LUT 构建：`mul(factors_broadcast, s_broadcast)`，2 条 8-wide mul。
4. 交织偶/奇位（`unpacklo/unpackhi`），两次 load-add-store 累加到 output。
5. FMA 用途：累加是纯 add（无乘），FMA 只在 LUT 构建可有可无；
   门控取 `avx2+fma`（本机有 fma；实际只用 mul/add，fma 为未来 F16 kernel 铺路）。

非 32 倍数的 n：理论上不存在（2048/6144/1024/3072 全是 32 的倍数），
kernel 内 `debug_assert!(n % 32 == 0)`，尾巴走现有安全标量循环。

## 3. unsafe 收敛策略（本方案核心）

先说实话：stable Rust 写 SIMD **不可能零 unsafe**——每条 intrinsic 都是
`unsafe fn`。能做的是把 unsafe 压到最小包络：

- **单点包络**：所有 unsafe 只存在于 `simd::gemv_q4_avx2` 这一个函数内
  （延续现有 `simd.rs` 风格：`#![allow(unsafe_op_in_unsafe_fn)]` + `unsafe fn`）。
  函数签名全是安全类型（`&[f32]`/`&[u8]`/`&[f16]`/usize），调用方（`gemv.rs`
  dispatch）零改动、零 unsafe。
- **包络内部纪律**：
  - 只做 `loadu/storeu`（不对齐假设，杜绝 SEGV 类错误）；
  - 所有偏移量来自 `g * 32` / `po + i` 这类闭式表达式，进循环前
    `debug_assert` 验证上界；tail 不进 unsafe 区；
  - 不手写裸指针算术做数据搬运，只传指针给 intrinsic。
- **门控**：`#[target_feature(enable = "avx2,fma")]` + 运行时
  `is_x86_feature_detected!("avx2")`；非 x86_64 走 `#[cfg]` 直接编译掉，
  与现有 AVX-512 三层 dispatch（512 > AVX2 > scalar）一致。
- **为什么不用 `safe_arch` / `wide` / nightly `std::simd`**：
  零 unsafe 的代价是新依赖 + API 跟随成本，且 codegen 与 stdarch  intrinsics
  同构（safe_arch 就是薄封装），省不掉查表算法本身的正确性负担；
  保持零依赖、与仓库现有风格一致。nightly `std::simd` 在 release 构建链上不可接受。
- **真正的安全网是测试**（见 §4），不是少写几个 unsafe 块：
  标量实现就是 oracle。

### 比特一致性论证（differential test 的理论依据）

AVX2 版与标量版 FP 操作序列逐项同构：LUT 项 `s*c`（c 为小整数 f32，乘法精确，
`factors*s` 与手写 `s*-8.0…` 逐 lane 相同）；累加顺序同为 k-major，
`output[j]` 的加法链完全一致 → **要求测试断言比特级相等**（`assert_eq!` 整数组，
不是近似比较）。唯一特例：下标 8 处标量用字面量 `+0.0`，向量版算出 `s*0.0`
（s<0 时为 `-0.0`）；IEEE 加法中 `x + (±0.0) == x` 恒成立（RN 舍入下
`(+0)+(−0)=+0`），不改变任何比特——且 AVX-512 版已是同样做法，e2e sha 已验证。

## 4. 验证矩阵（先写测试再写 kernel）

1. **随机差分**：多组 (k,n)（2048×2048 / 2048×6144 / 1024×1024 / 含 0 输入 /
   含 0 scale / 全随机），`assert_eq!(avx2_out, scalar_out)` 逐元素。
2. **n 非 32 倍数**（人工构造）：触发 scalar tail，同样比特一致。
3. **线程池开/关**：`num_workers=1` vs 22，结果一致（kernel 与切分正交）。
4. **e2e**：基线命令同 seed 跑，sha256 必须与现有三版输出完全一致。
5. **perf 抽查**：`gemv_q4_inner`（届时改名 `gemv_q4_avx2`）占比应显著下降，
   且无新热点（如有，说明被内存带宽封顶，见 §6）。

## 5. 改动清单（预计 +250/-10 行）

- `src/simd.rs`：+ `has_avx2()`、`gemv_q4_avx2`（唯一 unsafe 增量，~120 行含注释）。
- `src/gemv.rs`：dispatch 加一层（512 → AVX2 → scalar），~5 行。
- `src/simd.rs` tests 或 `gemv.rs` tests：差分测试 ~80 行。
- 不碰：线程池、权重格式、调用方、非 x86 路径。

## 6. 收益估算与封顶风险

- 标量 LUT 循环每输出约 ~6-8 周期（含 f16→f32 转换开销，perf 里 `f16_to_f32` 占 1%+）；
  AVX2 版每 16 输出约 ~12 条向量指令 → 理论 ~2-2.5x 单线程吞吐。
- 与线程池叠加：generation 41s → **~20-25s**（若未触带宽墙）。
- 封顶风险：Q4 talker 每 frame 重读 ~0.9GB 权重；22 核并发下若内存带宽先满，
  加速比打折——`perf stat -e cache-misses` 可在验证阶段确认；若封顶，
  下一步才是 F16/权重量化或 KV 结构优化，不在本方案内。
- 回退：单函数 + dispatch 一行，删掉即回标量，零耦合。
