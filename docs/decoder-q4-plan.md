# Speech Decoder 量化立项书（❌ 已关闭，2026-09-30，门禁 0 未过）

> 地位：P0（前缀缓存 -3%）+ P0（合块 -10%）+ P1（fat LTO -5%）全部落地后，
> P2 里唯一有结构性 headroom 的项目。talker Q4、predictor Q4（compress 双双
> `use_q4=true`）、AVX2 手写核均已到屋顶；唯 speech decoder 是 f32。
> **2026-09-30 探索结论：门禁 0 未过，立项关闭，不实施。**（下文保留设计，
> 重开需先推翻 §8 的测量。）

---

## 1. 事实基线（已查，不猜）

- 精度版图：talker Q4（`gemv.rs:237` AVX2 直读 packed）/ predictor Q4 /
  decoder **f32**（`save.rs:save_decoder` 全 `write_f32_vec_io`；`Conv1dWeight`
  也是 `Vec<f32>`）。
- 占比：decode 占 long-doc wall 约 31%（alynaze §1：450.8s / 1433.4s）。
- decode 内部分量（`decoder.rs`）：codebook 查表 → pre-conv → 8 层 transformer
 （全序列，sliding window 72）→ Vocos 上采样 + 卷积。注意 transformer 是
  全序列 GEMM（算术强度高于 GEMV），Vocos 侧是大 activation 搬运
 （合块实测：单帧成本随块长 0.11→0.18s，疑似巨型 activation 内存流量）。
- 推论：量化只压得动**权重流量**那部分；activation 流量压不动。
  上限估算：decode 31% × 权重占比（未知，待 perf 量）— 乐观 5~15% total，
  悲观 ~0%。**先量权重占比再动手**（门禁 0）。

## 2. 不变量（动不得）

- Q4 行布局沿用 talker 系：group=32，每组 16 packed bytes + 1 个 f16 scale，
  反量化 `(q-8)*scale`；AVX2 kernel 复用（`simd.rs`），bit-exact 语义不变。
- `.qora-tts` 文件格式向后兼容：decoder 段加 format 标记，旧文件照读
  （f32 路径保留，默认回退）。
- pure-Rust 立项不变：不引任何 C/C++ 量化库（RAGE-QUANT 类只借思路，不碰代码，AGPL 红线）。
- f32 decoder 永不删除：`--decoder-prec f32` 永远可用，当质量仲裁者。

## 3. 分阶段设计（F16 先行，Q4 在后）

- **Stage 1（F16，低风险探路）**：decoder 权重 f32→f16（`half` crate 已在依赖里），
  GEMM/conv 输入输出保持 f32 累加（只切权重流量一半）。预期 ~0~5%，主要价值是
  验证整套质量门禁跑得通。
- **Stage 2（Q4，正餐）**：decoder transformer 的 q/k/v/o + Vocos 卷积权重量化到
  group-32 Q4，复用 AVX2 核；codebook（纯查表，量小）保持 f32；norm/bias/scale
  类向量保持 f32。预期上限见 §1。
- 开关：`--decoder-prec f32|f16|q4`，默认 f32 直到耳验通过；通过后默认 q4，
  f32 留作仲裁。

## 4. 质量门禁（任一不过即回退，不合并）

> 本项目**不可能 bit-exact**（量化 by design 改变数值）。验收全部走质量门禁，
> 不走 sha。

- 门禁 0（动手前）：perf 量 decode 内权重流量占比；<20% 则直接关闭立项。
- 门禁 1（客观）：`--check` HNR 全套 + onset/ending 分析，q4 vs f32 同文本同 seed，
  退化阈值：整体 HNR 下降 < 0.05，首尾 1s 无新增糊/哑帧。
- 门禁 2（主观）：用户耳验 A/B（f32 vs q4 同句盲听，至少覆盖女声 mum 系 + 32uwgy 系，
  短句 + 200 帧以上长块）。
- 门禁 3（回归）：全量单测 101 通过；`--decode-codes` 路径单独 A/B（隔离 vocoder 退化）。
- 门禁 4（性能）：sub8 基准（RTF 口径）实测提升，否则 Stage 2 不合入（Stage 1 同理）。

## 5. 风险清单

1. vocoder 对量化敏感：Vocos 卷积误差会被上采样放大，Q4 可能出现可闻底噪/金属音
   → Stage 1 先行、codebook 不动就是为此。
2. 全序列 GEMM 若是计算 bound 而非访存 bound，Q4 收益→0（门禁 0 就是筛这个）。
3. 超长块（~400 帧）误差累积：门禁 2 强制覆盖长块。
4. 构建产物膨胀：decoder 双精度版本（f32+q4）进 `.qora-tts`，文件变大；
   对策：compress_model 加 `--decoder-q4` 只产出选定版本，不双持。

## 6. 实施步骤（拍板后按序执行）

1. perf 量 decode 权重流量占比（门禁 0）。
2. Stage 1：f16 decoder 路径 + `--decoder-prec` 开关 + 门禁 1~4 跑通。
3. 耳验 Stage 1（门禁 2）。过 → Stage 2；不过 → 关闭立项。
4. Stage 2：Q4 decoder（复用 kernel）+ compress_model 出 q4 decoder 产物。
5. 全门禁 + 用户终验 → 默认 q4 → 写实测数回本文档，状态翻 ✅。

## 7. 非目标

- talker/predictor 再量化（已 Q4，无事可做）。
- iGPU（已否决，见 alynaze 注 11）。
- IR/外部推理库（已否决）。
- KV cache 量化（cache 小，无意义，见 alynaze 注）。

## 状态

- [x] 门禁 0（perf 权重占比）→ **未过，见 §8，立项关闭**
- [ ] Stage 1 + 门禁 1~4（不执行）
- [ ] Stage 2 + 全门禁 + 终验（不执行）

## 8. 门禁 0 实测（2026-09-30，关闭依据）

方法：`QORA_DUMP_CODES` 取 241 帧 codes → `--decode-codes` 隔离 decoder，
perf 抓 DRAM + 双长度计时分解固定/可变成本。零代码改动。

| 测量 | 数值 |
|---|---|
| decode-only 241f（taskset P 核钉死） | 54.5s，LLC-load-miss 1.9B ≈ **123GB DRAM 读** |
| decoder f32 权重总量（估算：8 层×4×[1024×512] + Vocos + codebook） | ~130MB |
| 权重流量占比 | ~130MB / 123GB ≈ **0.1%**（门禁要 ≥20%） |
| 42f vs 241f decode wall | 3.2s vs 27.4s（8.6x 时间 / 5.7x 帧，超线性） |
| 线性外推固定截距（权重一次性成本） | ≈ 0（斜率 0.122s/f，截距为负） |

结论：decode 成本几乎纯可变（activation 流量，Vocos 巨型 tensor，
`[96, 289920]` 级），权重只读一次、贡献归零。**量化权重省 100MB 级流量，
相对 123GB 无意义**；且 decode 带宽利用率仅 ~2GB/s（延迟/抖动 bound，
非吞吐 bound），int8 之类同样无收益。 Stage 1（F16）同理无收益。
重开本立项的唯一路径：先拿出权重占比 ≥20% 的反例测量。
