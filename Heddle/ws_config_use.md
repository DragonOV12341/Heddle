# Per-Op Warp Dispatch (Plan B) 使用指南

## 概述

Per-Op Warp Dispatch 允许 CP-SAT 求解器为每个消费者操作独立分配 warp group，取代原有的位置分裂（`pv_split_idx`）方式。求解器基于 FU 容量、寄存器活跃度、blocking sync 等约束，自动找到最优的 warp 分配方案。

## 数据流

```
CP-SAT Solver (Python)
    │  kernel_warp_assigns = {"k0": {"A":0, "B":1, "C":0}}
    ▼
heddle_consumer_schedule.py
    │  annotation: tl_pcws_warp_assigns = "s0:0,s1:1,s2:0"
    ▼
TVM IR annotation (consumer loop)
    │  PassContext config: tl.finegrainedws_warp_assigns
    ▼
finegrained_ws.cc (C++ PCWS pass)
    │  ParseWarpAssigns → per-op warp dispatch codegen
    ▼
CUDA kernel (N consumer warp groups + producer)
```

## 使用方式

### 方式一：自动路径（推荐）

Heddle 的 `HeddleConsumerSchedule` pass 在 Phase B 求解时自动分配 warp。只要 `num_warps > 1`，结果会自动注入 IR annotation，C++ pass 自动读取并生成 N-way dispatch 代码。

```python
import heddle
heddle.apply_all_patches()

# TileLang kernel 正常编写，warp 分配由调度器自动完成
```

自动路径的触发需要在 `_phase_b_consumer_ordering()` 调用处设置 `num_warps`：

```python
phase_b_result = _phase_b_consumer_ordering(
    infos_list, consumer_indices, deps,
    use_precise_latency=use_precise_latency,
    num_warps=2,                  # 启用 2 个消费者 warp group
    barrier_edges=barrier_edges,  # 可选：blocking sync 边集合
    timeout_ms=adaptive_timeout,
)
# 返回值: (ordering, schedule_times, warp_assigns)
# warp_assigns = {"s0": 0, "s1": 1, "s2": 0, "s3": 1}
```

注解自动注入逻辑（`heddle_consumer_schedule.py` 行 1621-1624）：

```python
if phase_b_warps:
    warp_str = ",".join(f"{k}:{v}" for k, v in sorted(phase_b_warps.items()))
    _set_ws_annotation(new_annotations, "tl_pcws_warp_assigns", warp_str)
```

### 方式二：手动指定（通过 PassContext）

通过 TVM PassContext 显式设置 warp 分配字符串：

```python
with tvm.transform.PassContext(config={
    "tl.finegrainedws_warp_assigns": "s0:0,s1:1,s2:0,s3:1",
}):
    mod = tilelang.transform.FineGrainedWarpSpecialized()(mod)
```

可同时设置的相关 config key：

| Config Key | 说明 |
|------------|------|
| `tl.finegrainedws_warp_assigns` | warp 分配字符串（推荐） |
| `tl.pcws_warp_assigns` | warp 分配字符串（兼容旧路径） |
| `tl.finegrainedws_barrier_hints` | barrier hint 配置 |
| `tl.finegrainedws_stage_offsets` | stage offset 配置 |

## Warp Assigns 字符串格式

格式：`"s0:0,s1:1,s2:0,s3:1"`

- `sN` — compute_stmt 索引（N 从 0 开始）
- `:` 后的数字 — warp group ID（从 0 开始）
- 多个条目用 `,` 分隔

示例：

| 字符串 | 含义 |
|--------|------|
| `"s0:0,s1:0"` | 2 个 op 都在 WG0（退化为标准双角色） |
| `"s0:0,s1:1"` | op0 → WG0, op1 → WG1（2-way dispatch） |
| `"s0:0,s1:1,s2:2,s3:0"` | 3 个 warp group，op0/op3 → WG0, op1 → WG1, op2 → WG2 |

warp group 数量由 `max(warp_id) + 1` 自动推导。如果只有 1 个 warp group，自动 fallback 到标准双角色模式。

## 运行时线程布局

启用 N 个消费者 warp group 后的 GPU 线程分布：

```
Thread   0 ~ 127    →  Warp Group 0 (consumer)
Thread 128 ~ 255    →  Warp Group 1 (consumer)
Thread 256 ~ 383    →  Warp Group 2 (consumer, N≥3 时)
...
Thread N*128 ~ ...  →  Producer (extent = producer_threads)
```

每个 warp group 内的 `threadIdx.x` 会被 rewrite 为 `[0, 128)` 局部偏移：
- WG0：直接使用 `PCThreadIdxRewriter`
- WG1..N-1：先用 `ThreadIdxSubstitutor` 减去 `w*128` 偏移，再用 `PCThreadIdxRewriter`

## Dispatch 链优先级

C++ `finegrained_ws.cc` 中的 dispatch 判断顺序（前面优先）：

| 优先级 | 条件 | 模式 |
|--------|------|------|
| 1 | `has_three_role` | 三角色（producer + 2 fixed consumer） |
| 2 | `dual_consumer_enabled_` | 双消费者（固定 2 路位置分裂） |
| 3 | `track_warp_groups && warp_assigns 非空` | **Per-op warp dispatch** |
| 4 | fallback | 标准双角色（producer + consumer） |

## CP-SAT 求解器参数

`UnifiedScheduler` 构造时的关键参数：

```python
solver = UnifiedScheduler(
    [partition],
    fu_caps={                          # 各 FU 类型的并发容量
        ResourceType.TMA: 1,
        ResourceType.TensorCore: 1,
        ResourceType.ALU: 2,
        ResourceType.SFU: 1,
    },
    reg_limit=256,                     # 每个 warp 的寄存器上限
    num_warps=2,                       # 消费者 warp group 数量
    timeout_s=5,                       # 求解超时（秒）
)
```

`OpSpec` 中与 warp 相关的字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `fixed_warp` | `int` | 固定分配到指定 warp（-1 表示自由分配） |
| `spill_cost` | `int` | 跨 warp 依赖的额外延迟代价 |
| `deps` 第 3 元素 | `bool` | `blocking_sync`：True 时强制 producer/consumer 同 warp |

## 涉及的文件

| 文件 | 职责 |
|------|------|
| `heddle/scheduler/cp_sat.py` | CP-SAT 求解器，输出 `kernel_warp_assigns` |
| `heddle/transform/heddle_consumer_schedule.py` | 调用求解器，将 warp 分配写入 IR annotation |
| `heddle/_monkey_patch.py` | 注册 PassContext config key |
| `patches/cpp/src/op/builtin.h` | 定义 `kFineGrainedWsWarpAssigns` 常量 |
| `patches/cpp/src/transform/finegrained_ws.cc` | C++ PCWS pass，解析 warp assigns 并生成 dispatch 代码 |
| `tests/test_warp_dispatch_integration.py` | 集成测试（格式解析、源码结构、端到端） |
| `tests/test_cp_sat_warp.py` | 求解器单元测试（需要 ortools） |
