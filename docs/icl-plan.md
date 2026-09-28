# QORA-TTS ICL（参考文本 Prompt）实施方案

> 目标：新增 `--ref-text`，让生成韵律（停顿 / 语调 / 语速）跟随参考音频。
> 现状基线：x-vector-only，只搬音色。实测"你好,初次见面,我是Nori"→ 45 frames / 3.6s，
> 其中静音 1.76s（0.7~0.9s 大停顿 ×2），temperature/seed 对照证实停顿结构基本不变。

---

## 1. 官方机制（已从源码确认的部分）

官方实现：HF Space `Qwen/Qwen3-TTS`，`qwen_tts/core/models/modeling_qwen3_tts.py`。

- `voice_clone_prompt` 结构：`ref_spk_embedding` / `ref_code` / `icl_mode` / `x_vector_only_mode`
  四件套，ICL 与 x-vector 可双开。
- 调用链：`extract_speaker_embedding`（已对齐：mel n_fft=1024/hop=256/128维/Slaney，
  embedding raw 注入不归一化）→ `generate_speaker_prompt` →
  prefill embedding 手术（`cat(tts_pad×5, tts_bos) + codec[:, :-1]`，pos4 = spk_emb + tts_pad，
  本仓库已按此修复）→ 有 ICL 时调 `generate_icl_prompt`，无 ICL 时走 trailing-text 路径。
- **此前只看到 `generate_icl_prompt` 的调用点，没看到函数体**（见 §4-A2）。

## 2. 本仓库现状盘点

| 模块 | 状态 |
|---|---|
| talker / predictor / decoder / speaker encoder（f32，二进制） | ✅ 推理已对齐官方 |
| 文本 tokenizer（11MB，已恢复实体） | ✅ |
| `--ref-audio`（x-vector） | ✅ |
| `speech_tokenizer/model.safetensors` | ❌ 不存在（目录下只有 3 个 json） |
| codec **encoder** 推理代码 | ❌ 零行（`loader.rs:509` 只读了 decoder 部分） |
| `generate_icl_prompt` 移植 | ❌ 零行 |
| `--ref-text` | ❌ |
| `talker.rs:prefill_talker_with_voice` | ⚠️ 旧版（0.6B 风格）脚手架，布局与 Qwen3-TTS ICL 不一致，仅"16 码本 embedding 求和"思路可参考，不可直接复用 |

`encoder_config`（`model/1.7B/speech_tokenizer/config.json`，已在本地）关键参数：
`encode_downsample_rate=1920`（=24000/12.5）、conv `kernel_size=7` + causal conv、
8 层 transformer（hidden 512 / 8 头 / RoPE theta=10000 / sliding window 250）、
RVQ `num_quantizers=32`、`encoder_valid_num_quantizers=16`、`num_semantic_quantizers=1`
（semantic 码本 4096）。

## 3. 总体方案（分三阶段，每阶段独立 commit、可回退）

- **Phase A**：拿权重 → 精读官方 encoder 源码 → Rust 移植 encoder → **离线验证**（门禁）。
  ✅ 已完成（2026-09-28）：`src/codec_encoder.rs` 前向全实现，与官方 torch 版
  逐阶段对齐（seanet/transformer/downsample 最大差 <1e-3，16×23 码 100% 一致），
  黄金往返 corr=0.9676。途中修掉 `mimi_conv_forward` 通道数参数脚枪
  （改用 `conv.in_ch`，见 commit）。
- **Phase B**：移植 `generate_icl_prompt` → `--ref-text` 接线 + 双开关 → 无 ref-text 时行为零变化。
  ✅ 已完成（2026-09-28）：`--ref-text` / `--encoder-weights`（默认
  `<model>/speech_tokenizer/model.safetensors`）；`build_icl_block` 按官方
  streaming 分支（text+pad 与 codec 对齐、余数进 trailing）；ICL 时 prefill 为
  9+block（无 first_text 位），逐帧只融 remainder/pad。
  B3 回归：同 seed 无 ref-text 双跑 sha256 一致。ICL 首测通路正常
  （ref 6 tokens/35 帧 → block 36/remainder 0 → 27 帧出声），但出现 0.8s 句首
  停顿（疑似 continuation 效应），韵律质量待 Phase C 用真人参考 + 试听评估。
- **Phase C**：同 seed A/B 评测 + 回归 + 性能记录。

## 4. Phase A 细化

### A1. 权重获取
- 从 `Qwen/Qwen3-TTS-Tokenizer-12Hz` 拉 `speech_tokenizer/model.safetensors`，
  记录 sha256 与大小（参考：此前 `tokenizer.json` 走 GitHub raw 直链成功，LFS 曾静默失败，
  准备备用下载方式）。
- 用其中 decoder 权重与现有 `model.qora-tts` 内 decoder 做交叉验证（同名 tensor 逐个比对
  shape/均值），确认权重来源一致后再动 encoder。

> ✅ 已验证（2026-09-28，脚本 `/tmp/opencode/a1_inventory.py` 全过）：
> `model.safetensors` 651MB，sha256 `836b7b35…571258`，496 tensors 全 F32，
> encoder 225 / decoder 271，无 weight_norm。
> decoder 侧 17 个 `src/loader.rs` 字面 key + 4 组 prefix（含 15 层 rvq_rest）全部命中，
> 与现有二进制兼容。
> encoder 侧键名为 V2 嵌套布局（`encoder.encoder.*` / `encoder.encoder_transformer.*` /
> `encoder.downsample.*` / `encoder.quantizer.{semantic,acoustic}_residual_vector_quantizer.*`），
> 关键 shape 全对：首 conv [64,1,7]、downsample [512,512,4] 无 bias、
> 码本 embed_sum [2048,256]、semantic RVQ 1 层 / acoustic 31 层 / transformer 8 层。
> ⚠️ 注意 encoder 码本键是 Mimi 风格 `codebook.embed_sum`，与 decoder 侧
> `_codebook.embedding_sum` 命名不同，loader 不可混用。

### A2. 官方源码精读清单（按顺序，带着问题读）
1. **`generate_icl_prompt` 完整函数体**（最高优先级，之前只见到调用）：
   `ref_id=input_id[:, 3:-2]` 切片语义、`ref_code` 如何转 embedding、
   `trailing_text_hidden` 在 ICL vs 非 ICL 下的构造差异。
2. **Encoder 类**：输入是原始波形还是 mel？（1920 下采样比指向波形卷积栈，需确认；
   若是波形，注意 `pad_mode=constant`、`use_causal_conv=true` 与现有 reflect-pad 代码的区别。）
3. **32 quantizers vs valid 16**：`ref_code` 取哪 16 个码本？semantic quantizer 是否参与
   ICL？`num_semantic_quantizers=1` 的码流去向。
4. **Encoder transformer**：RoPE(10000) + sliding window(250) + causal conv，
   评估 `src/rope.rs` 复用度；`trim_right_ratio=1.0`、`compress=2` 的含义。
5. **参考音频预处理**：重采样 / 幅度归一 / 长度截断与补齐要求（对照
   `extract_speaker_embedding` 的 `assert sr == 24000`）。
6. 精度：官方 encoder 跑什么 dtype；本仓库统一 f32 是否足够（现有权重全 f32 存，
   继续沿用）。

### A3. Encoder Rust 移植
- 新模块 `src/codec_encoder.rs`（权重结构 + 前向），`loader.rs` 加 `load_codec_encoder`，
  复用 `src/conv.rs` 多线程卷积与 `src/rope.rs`（若 sliding-window/RoPE 可套用，不行则单列）。
- 单测：输出 shape（T = samples/1920）、码值范围（<2048）、与 Python 版同输入数值比对
  （容差按 f32 累积误差定，建议相对误差 <1e-3）。

### A4. 离线验证（Phase A 门禁，不通过不进 B）
- Encode 一段参考音频 → 检查码率 12.5Hz、16 码本命中分布无坍缩。
- **黄金验证**：`ref_code` 直接喂现有 decoder 解码 → 与原参考音频对比试听/频谱。
  能过这一关，证明 encoder 精度达标（decoder 已验证正确）。

## 5. Phase B 细化

- **B1**：逐行移植 `generate_icl_prompt` 的 embedding 拼接（只动 `generate_new.rs` 及
  可能的 `talker.rs` 辅助函数，不碰权重）。
- **B2**：CLI 加 `--ref-text`；组装 `voice_clone_prompt`（`icl_mode=true` 当且仅当
  ref-audio + ref-text 齐备；保留 `x_vector_only_mode` 兼容纯 embedding 路径）。
- **B3**：回归基线——无 `--ref-text` 时，固定 seed 下输出与改前一致
  （至少帧数/时长一致，最好首帧 logits 比特级一致）。

## 6. Phase C 验收

- 同 seed A/B：沿用已有的静音分段脚本（20ms 帧、-40dB 门限），对比停顿时长/位置分布；
  语调主观对比（同一文本 × 有/无 ref-text）。
- 不同参考音频（中/英文、快/慢节奏）各一组，确认韵律跟随而非固定模板。
- 性能：encoder 为一次性开销（参考 45MB speaker encoder 秒级完成），记录 Total 变化；
  生成阶段单帧耗时不应变化。

> ✅ 首轮 A/B（2026-09-28，mum3bv 11.28s 真人单人声库，seed 777，文本"今天天气真好，我们出去走走吧"）：
> 参考本身：快节奏句内（间隙 0.04~0.1s）+ 3 个长句间停顿（0.66/1.36/0.68s）。
> | 版本 | 总长 | speech | silence | 逗号停顿 | 尾部静音 |
> |---|---|---|---|---|---|
> | xvec | 3.04s | 1.82s | 1.22s | 0.74s | 0.10s |
> | ICL（转写无标点） | 3.76s | 1.58s | 2.18s | 0.78s | 0.96s |
> | ICL（转写加标点） | 3.36s | 1.62s | 1.74s | 0.62s | 0.82s |
> 结论：通路正常、对齐机制存活（标点使逗号停顿 0.78→0.62s）；ICL 固定带 ~0.8~1s
> 尾部静音 continuation（后处理可压）；韵律是否真正跟随需试听 + 更多样本。
> 音频：`output_mum_xvec.wav` / `output_mum_icl.wav` / `output_mum_icl2.wav`（未进版本库）。
> ✅ 试听确认（2026-09-28）：语调与参考本人说话的调子完美符合。ICL 目标达成。

## 7. 模型文件格式升级

- `model.qora-tts`：`VERSION + 1`（见 `src/save.rs`），尾部 append encoder 段。
- `load` 必须兼容旧版：无 encoder 段 → ICL 报错但 x-vector 照常工作。
- `compress_model`：新增 encoder 加载步骤；量化策略先 f32 保精度，Phase C 后视体积/速度
  再定是否 Q4（参考：现 talker Q4、speaker encoder f32）。

## 8. 风险与回退

| 风险 | 缓解 |
|---|---|
| encoder 数值精度不够（causal conv / sliding window / RVQ 任一错位） | Phase A4 门禁；Python 对照脚本先行 |
| 权重下不到（LFS/网络） | GitHub raw 直链备用（有成功先例） |
| ICL 效果不及预期（韵律仍不跟随） | 先单样本验证再批量评测；B2 开关保证随时回退 x-vector |
| scope 蔓延（顺手改采样/解码） | 本方案冻结除 ICL 外的一切改动，另开分支议 |

## 9. 工作量估计

Phase A 约占 60%（encoder 移植是深水区），B 约 30%，C 约 10%。
建议 A2 精读输出一份"函数级对齐清单"（官方函数 → Rust 对应位置/待写位置）后再写码。
