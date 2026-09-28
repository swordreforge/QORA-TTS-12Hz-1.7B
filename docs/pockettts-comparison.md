# PocketTTS 对比结论（存档，2026-09-28，只对比不实施）

## PocketTTS 快照（Kyutai，2026-01）

- 100M 参数（313M teacher → 6 层 student，隐变量蒸馏），code MIT。
- CPU RTF 0.71（Xeon 8272CL 4 核）；M4 约 6x realtime；200ms 首包；可流式。
- 5s 参考零样本克隆（voice prompt 前缀 + codec encoder，思路同 ICL）。
- 客观质量 UTMOS 4.10（同场 Kokoro 4.44）。
- 语言：英/法/德/葡/意/西 —— **无中文、无日文**。
- 权重许可待核实（code MIT ≠ weights，需单独确认）。

## Rust 路线（官方仅 Python）

| 路线 | 说明 | 成熟度 |
|---|---|---|
| pocket-tts-xn | LaurentMazare，XN 后端移植 | 社区个人项目 |
| pocket-tts-candle | Candle 版，WASM/PyO3 | 社区个人项目 |
| sherpa-onnx | 12 语言绑定（含 Rust），有现成 int8，上过树莓派/Jetson | 最稳 |
| PocketTTS.cpp | 单文件 ONNX Runtime，CLI+HTTP+FFI | 可用 |

共同代价：引入外部推理运行时，与本仓库手写纯 Rust 零依赖路线相悖。

## Head-to-head（vs 本仓库 Qwen3-TTS-1.7B 栈）

| | QORA 现状 | PocketTTS |
|---|---|---|
| 参数 | 1.7B | 100M（1/17） |
| 速度（U7） | 0.35s/frame，RTF ~4 | RTF 0.7（快 ~6x） |
| 中文/日文 | 主力场景已验证 | 无 |
| 克隆+韵律跟随 | ICL 已验证 | 有克隆，跟随未验证 |
| 流式 | 无（整句出） | 有 |
| 授权 | Apache 2.0 清晰 | 权重待核实 |
| Rust | 手写核心 | 第三方运行时/社区移植 |

## 结论

1. 三大刚需（中日英 + 克隆韵律 + 纯 Rust）PocketTTS 都不满足，**不替代**。
2. 它赢的维度（实时/流式/低资源）是当前不痛的维度（离线合成可接受）。
3. 适用场景（未来）：纯拉丁语系边缘实时（如英文播报+流式），或做第二引擎专跑英文。
4. 真要速度且固定音色：先用仓库内 `model/0.6B`（9 内置音色，无克隆），与 1.7B 双引擎搭配。
5. 待办（未启动）：英文同题 A/B（sherpa-onnx int8 vs 本引擎：音质+速度+克隆相似度），约半天工作量，拿数据再定。
