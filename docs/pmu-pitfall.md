# PMU 坑：core LLC-miss 看不见 streaming 流量（2026-09-30，实测）

> 背景：decoder-Q4 门禁 0 要量"权重流量占比"，`cpu_core/LLC-load-misses`
> 给出 123GB/241f，看似够用；但 P 钉死复算 two2（prefill+gen54+decode54）
> 只给出 4.5GB——而 talker Q4 一帧就要扫 ~1GB 权重，54 帧理论下限 60GB+。
> 差 14 倍。不是 workload 变了，是计数器瞎了。

## 1. 机制

`cpu_core/LLC-load-misses`（LONGEST_LAT_CACHE.MISS 系）只统计 **demand miss**
（CPU 缺数 Stall 去访存的那次）。GEMV/GEMM 扫权重的 streaming 访问被硬件
prefetcher 提前搬进 LLC，demand 侧命中 → **不计数，但 DRAM 流量真实发生**。
decoder 的 activation 是跨步/延迟 bound 访问，prefetcher 够不着，全落成
demand miss → 全计数。于是同一个计数器下：

| 流量类型 | 是否计数 | 后果 |
|---|---|---|
| 权重 streaming（talker/predictor GEMV） | 否（prefetch 搬运） | 绝对值少 10 倍以上 |
| activation 跨步（Vocos 巨 tensor） | 是 | 看起来占了"几乎全部" |

用它做"权重占比 = 权重 miss / 总 miss"会得出 0.1% 这种**方向正确、
数值全错**的结论（0.1% 是 demand-miss 视角的占比，不是 DRAM 视角）。

## 2. 正解（本机 uid 1000 / paranoid 2，逐个试过）

| 手段 | 状态 |
|---|---|
| `uncore_imc_free_running/data_read/`（真 DRAM 总量） | ❌ 要 root，EINVAL |
| `ocr.*` offcore（per-process，含 prefetch 的 DRAM 变体） | ❌ 只暴露了 demand 变体（`ocr.demand_data_rd.l3_miss`），无 hwpf 变体 |
| 解析 traffic model（权重字节 × pass 数 + activation 形状） | ✅ 唯一可用。权重侧是精确数（格式已知），activation 侧从代码形状估 |
| 时钟比（phase split timing） | ✅ 硬信号：predictor/talker 1.19 ≈ 参数比 1.24，坐实 traffic-proportional |

## 3. 规矩（以后量 perf 先读这节）

1. `LLC-*-misses` 绝对值只看数量级 + 只做**同形状比较**（demand-miss 视角内部比，
   如 42f vs 241f 的超线性判断有效；跨 workload 比权重占比无效）。
2. 要总量：先算 analytical（§2 表），再用 demand-miss 做下界交叉验证。
3. 要 root 时：IMC free-running 是正解（`data_read`/`data_write`，system-wide，
   安静机器 + 前后差值）。没 root 别碰 uncore。
4. 环境比：multiplexing 缩放（括号百分比 <100%）只做趋势看；要精确就钉核 +
   减事件数（taskset + QORA_THREADS + 2~3 个事件）。
5. 本文件的教训只适用于 Intel hybrid + 无 root；换机器重验 paranoid 和事件集。
