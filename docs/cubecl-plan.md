# cubecl GPU 立项书（🟡 已提议，2026-10-02，门禁 0 未量）

> 地位：CPU 已到屋顶（P0 前缀缓存 / 合块 / fat LTO 全落地；decoder-Q4 门禁 0 未过已关闭，
> 见 `decoder-q4-plan.md:88-105`）。本立项是**未来 dGPU / 跨平台**留门，
> 不是本机 iGPU 提速项（本机预期仍是负优化，见 §1）。
> 动手前必须先过门禁 0，否则不碰主树。

---

## 1. 事实基线（已查，不猜）

- 工作负载性质（`decoder-q4-plan.md:18-23` + `alynaze.txt:12`）：
  decode 占 long-doc wall 约 31%；decode-only 241f 实测 123GB DRAM 读，
  decoder f32 权重仅 ~130MB（占比 0.1%）——**activation 流量 bound，非权重 bound**，
  且带宽利用率仅 ~2GB/s（延迟/抖动 bound）。
- 推论：GPU 能压的是**算术强度高**的部分（全序列 transformer GEMM、
  Vocos 上采样大 GEMM 即 `conv.rs:434 f32_gemm_bias`），压不动 GEMV 小 batch
  访存（`gemv.rs:1-30` Q4/F16 GEMV 系）与巨型 activation 搬运。
- Amdahl 上限：decode 31% 全搬上 GPU ≈ 25% total 上限；只搬 GEMM 子集则更低。
  **只看 kernel 微基准一律不算数，看端到端 RTF**（`docs/pmu-pitfall.md` 同理）。
- 本机（Ultra 7 155H，Arc iGPU，共享内存）：CPU 与 iGPU 争同一份 DRAM 带宽，
  H2D/D2H 无独立显存红利 + shader 编译冷启动，此前已否决（`alynaze.txt:12`）。
  cubecl 只消掉"C++ 栈"这一条否决理由，另两条（共享内存、Amdahl）仍在。
- 当前依赖（`Cargo.toml:14-20`）：`half/serde/serde_json/tokenizers/safetensors/sha2`，
  无 GPU 依赖；测试基线 114 通过。

## 2. 不变量（动不得）

- pure-Rust 立项不变：默认构建**不引 CUDA toolkit、不引任何 C/C++ 链接依赖**。
  只用 cubecl + wgpu-runtime 后端（Vulkan/Metal/DX12 走系统驱动，非 vendored C++）。
  CUDA/HIP 后端如要试，只能做**非默认独立 feature**，永不进 portable release。
- CPU 路径永远默认、永远 bit-exact：端到端 sha 仍是真门禁；GPU 路径走容差门禁（§4），
  永不替换 CPU 口径。
- 默认关：`--gpu off` 默认；无 GPU 适配器 / 初始化失败时静默回退 CPU，不炸。
- 三件套 cfg 卫生（此前 macOS E0425 教训）：detection + `#[cfg]` 调用点 + 标量 fallback，
  测试函数整个 `#[cfg]`；`cargo check --target aarch64-apple-darwin` 类问题由 macOS CI 终审。
- 测试先行：新模块先写 differential/oracle 测试再接主树；复跑铁律 + quiet-machine
  纪律沿用（环境 2x 毛刺已有前科）。
- portable release 不受影响：generic x86-64 隔离构建 + `scripts/build-release.sh`
  产物尺寸/sha 流程不变；GPU feature 永不污染默认 `cargo build --release --bins`。

## 3. 分阶段设计（Stage 0 不碰主树）

- **Stage 0（spike，scratch crate，1~2 天封顶）**：
  独立小 crate 依赖 `cubecl 0.8.x`（wgpu backend only），复刻两个形状：
  (a) `f32_gemm_bias` 大 ops 形状（`conv.rs:453-458` 的 `ops >= 500_000` 分支，
  m/n/k 取 Vocos/transformer 真实尺寸）；(b) Vocos 上采样 conv 形状。
  只量三数：正确性 max-abs-diff vs `f32_gemm_block_scalar`、
  冷启动 shader 编译耗时、H2D/D2H + 执行 wall vs P 核 AVX2 wall。
  **门禁 0 不过 → 关闭立项，主树零 diff。**
- **Stage 1（单核 wiring，opt-in）**：`Cargo.toml` 加
  `[features] gpu = ["cubecl/wgpu"]`（默认关闭，CUDA/HIP 另起 `gpu-cuda` 非默认），
  新增 `src/gpu.rs`（device 单例常驻 + 调度 + CPU 回退），只接
  `f32_gemm_bias` 大 ops 路径（`conv.rs:449-458` 调度点），开关 `--gpu off|vocos`
  默认 off。配套 differential 测试（CPU oracle vs GPU，容差见 §4）。
- **Stage 2（扩面，仅 Stage 1 端到端赢了才开）**：候选 transformer 全序列 GEMM；
  GEMV 系（`gemv.rs:183-237` talker/predictor Q4/F16 小 batch）**默认不搬**
  （memory-bound + 调度开销大概率为负），要搬需单独门禁 4 重验。
- 开关语义：`--gpu off`（默认）= 现状逐字节一致；`--gpu vocos` = 仅 Stage 1 内核；
  无 `full` 档直到 Stage 2 立项。

## 4. 质量门禁（任一不过即回退）

> GPU by design 不可能 bit-exact（浮点求和顺序/融合指令不同）。CPU sha 门禁保留给
> 默认路径；GPU 走以下容差门禁。

- 门禁 0（动手前，Stage 0 数据）：大 GEMM 形状 GPU wall（含全部传输+同步）
  < P 核 AVX2 wall × 0.8，且冷启动 shader 编译 < 5s（否则每次 CLI 启动摊销不起）。
  不过 → 关闭。
- 门禁 1（客观）：`--check` HNR 全套 + onset/ending，`--gpu vocos` vs CPU 同文本同 seed，
  HNR 下降 < 0.05，首尾 1s 无新增糊/哑帧；另 kernel 级 max-abs-diff < 1e-4（f32 累加语义下）。
- 门禁 2（主观）：用户耳验 A/B（f32-CPU vs GPU 同句盲听，mum 系 + 32uwgy 系，
  短句 + 200 帧以上长块）。
- 门禁 3（回归）：全量单测（含新增 GPU diff 测试，**无 GPU 机器上自动 skip 不 fail**）
  通过；`--decode-codes` 路径单独 A/B。
- 门禁 4（性能）：sub8/long-doc 基准端到端 RTF 实测提升 ≥5%（kernel 微基准提升不算），
  否则 Stage 不合入。

## 5. 风险清单

1. shader 冷编译抖动 CLI 延迟 → 常驻 device + 预热复用，冷启动单独计时公示。
2. 小 batch GEMV 上 GPU 为负 → Stage 1 不碰 GEMV，Stage 2 需重过门禁 4。
3. 与 `gpool.rs` 16-worker 线程池争抢（CPU 喂数 vs GPU 队列串行）→ Stage 1 只单队列提交，
   不做 CPU/GPU 流水线重叠（~10% 上限 defer 项，此前已评估）。
4. 依赖膨胀：cubecl + wgpu 传递依赖多，构建时间/二进制尺寸涨 → 默认 feature 关闭，
   portable tar 体积门禁（`scripts/build-release.sh` 口径）单独记录。
5. 非 x86/驱动缺失 matrix（macOS Metal、Win DX12、无头 Linux）→ 全部走"无适配器回退 CPU"，
   macOS CI job 为终审。
6. NVIDIA 2026-09 CUDA-Rust 双轨（官方 Rust kernels）只吃 N 卡，破可移植 → 不跟，
   除非日后 `gpu-cuda` 非默认 feature 单独立项。

## 6. 实施步骤（按序，前一步门禁过了才走下一步）

1. Stage 0 spike + 门禁 0 数据回填 §8（未过 → 关闭，主树零 diff）。
2. `gpu` feature + `src/gpu.rs` + `--gpu` 开关 + diff 测试（默认 off，CI 默认构建零变化）。
3. 门禁 1~4（§4）→ 过则合入 Stage 1，文档写实测数翻 ✅；任一不过则回退。
4. Stage 2 是否开，凭 Stage 1 端到端数据另议（另立门禁，不自动延续）。

## 7. 非目标

- Burn 整框重写（autodiff/训练/ONNX 转码，数周级，见上轮结论）。
- candle 引入（CUDA toolkit 绑定 + 重写模型层，无额外收益）。
- rust-gpu / vulkano / ash 裸写（cubecl 已盖住，无必要吃维护成本）。
- 训练/微调、KV cache 量化、talker/predictor 再量化。
- iGPU 当默认路径（本机结论不变：预期负优化，只当回退测试机）。

## 状态

- [ ] 门禁 0（Stage 0 spike 数据，§8 待填）
- [ ] Stage 1 wiring + `--gpu vocos` + 门禁 1~4（不执行直到门禁 0 过）
- [ ] Stage 2（不自动开，另议）

## 8. 门禁 0 实测（待填，Stage 0 产出回填此处）

| 测量 | 数值 |
|---|---|
| 大 GEMM 形状 P 核 AVX2 wall（`f32_gemm_bias` 真实 m/n/k） | — |
| 同形状 cubecl/wgpu wall（含 H2D/D2H/同步） | — |
| shader 冷编译耗时 | — |
| kernel max-abs-diff vs scalar oracle | — |
| 本机 iGPU 端到端 RTF（预期为负，仅记录） | — |
