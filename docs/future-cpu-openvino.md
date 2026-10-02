# 延期备忘：CPU 微架构缝隙 + OpenVINO IR 实验田（⏸️ 待试，2026-10-02）

> 地位：想法已记录，不立项、不动手。重开任一条需先过各自门禁 0，否则维持关闭。
> 不变量沿用：pure-Rust（默认构建零 C/C++）、CPU 默认路径 bit-exact + sha 门禁、
> 复跑铁律 + quiet-machine、portable release 不受污染。

## 1. CPU 微架构缝隙（低 hanging fruit 已摘完）

现状：P0（前缀缓存/合块）+ P1（fat LTO）+ decoder 执行图（causal-conv tiling、
transpose 8-tap、pw-GEMM）全落地；PGO/mimalloc/in-place/transpose-dw 已 revert（零收益）。
P/E 调度裁决：faults/TLB/migration ≈ 1%，默认 all-core 最优。

候选（全是 1~3% 级 + 破 bit-exact 风险，任一需 sha + 端到端 RTF≥3% 才合入）：
- 软件预取/分块调参、pack 布局再压 TLB（对手是 123GB activation 搬运，非发射宽度）。
- AVX-VNNI int8：只吃权重流量 0.1%（`decoder-q4-plan.md` 门禁 0），预期归零，不优先。
- 大页/亲和：单 socket + 共享 DRAM，历史同类全零收益，需重验。

重开门禁 0：phase-split timers（`generate_new.rs`/`decoder.rs`）指认最大 wall 分量 +
perf 量该分量访存/计算 bound， теоретический 上限 <3% 则不开。

## 2. OpenVINO IR 实验田（只借思路，不引库）

- 红线：引 `openvino` runtime/crate = 拉 C++，破立项，维持否决；CUDA-toolkit 同理。
- 允许：读 IR 开放格式（XML+bin，Apache-2.0）+ fusion/tiling 文档，手工 Rust 重实现
  （对标 RAGE-QUANT 处理：借思路不碰代码）；NPU（155H AI Boost）对自回归延迟 bound
  先天不亲和，不列目标。
- 可行形态（未立项）：`docs/openvino-ir-notes.md` 研究笔记 + `src/ir_import.rs`
  纯 Rust 解析器（默认关 feature），节点映射到现有 `decoder.rs`/`conv.rs`/`simd.rs`
  kernel 做对照实验；门禁同 cubecl（端到端 RTF 说话）。

## 状态

- [ ] CPU 缝隙门禁 0（未量，不执行）
- [ ] IR 笔记 + importer（未立项，不执行）
