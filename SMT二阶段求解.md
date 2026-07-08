# SMT 二阶段求解流程

本文档按当前代码结构说明 `heddle_consumer_schedule.py` 中的二阶段求解路径。这里的“SMT”是历史命名，当前实现实际使用 OR-Tools CP-SAT。

整体路径是：

1. `_solve_naive_modulo_sched()`：先求一个基础 modulo issue plan，得到 `I/M/L`。
2. `_solve_smt_joint_optimize()`：把基础 plan 转成 `HeddleScheduler` 的 `OpNode` 图，在固定 `I` 下联合求 `Tv`、warp 分配、FU/issue 资源和容量复核。

## 入口

入口在 `Heddle/heddle/transform/heddle_consumer_schedule.py` 的 loop-body 重排流程中：

```python
mod_sched_plans = _solve_naive_modulo_sched(deps_all, infos_list, all_indices)
for plan in mod_sched_plans:
    optimized = _solve_smt_joint_optimize(
        deps_all,
        infos_list,
        all_indices,
        plan,
        kernel_num_threads=func_num_threads,
    )
```

进入这里前，transform 侧已经从当前 TIR loop body 中抽取出几类输入：

- `infos_list`：每条 stmt 的读写 buffer、producer/consumer 标记、WGMMA/TMA/barrier/sync 标记，以及原始 TIR stmt。
- `all_indices`：参与当前调度的 stmt/op id。
- `deps_all`：保守依赖图，包含数据依赖和必要的 barrier/sync 顺序。
- `func_num_threads`：当前 loop/kernel 的线程数，用于估计 warp 域。
- `num_stages` annotation：不是 joint 阶段的硬约束，但会影响预估 stage 数和 shared buffer 多版本 footprint。

## 第一阶段：naive modulo schedule

`_solve_naive_modulo_sched(op_deps, infos, op_indices)` 先找到一个基础 issue plan。它只关心启动时间、依赖和 issue/FU 容量，不解 warp 分配，也不把完整 RMEM/SMEM liveness 放进模型。

### 输入如何变成约束

每个 op `v` 会从 `_StmtInfo` 派生这些量：

- `latencies[v]`：依赖 readiness delay，来自 `_detect_op_latency_and_resource()` 等分析。
- `duration[v]`：issue slot 占用长度，来自 `_detect_op_issue_cycles()` 或 `_detect_wgmma_issue_cycles()`。
- `rrt[v][c]`：第 `c` 个 issue cycle 消耗哪些资源，例如 `TC`、`ALU`、`SFU`、`TMA`、`BARRIER`。
- `edges`：从 `op_deps/deps_all` 生成同迭代依赖；如果 op 自读写同一个 buffer 或是 wait/sync-like 语义，再补跨迭代 self hazard。

注意这里明确区分：

- `latency` 用于“消费者什么时候能读到结果”。
- `duration/reservation` 用于“发射槽被占多久”。
- 跨迭代 self hazard 使用 issue delay，不把所有 op 都强行加 `latency` 级别的自依赖。

### 主要变量

对一个固定的 `I`，naive 阶段创建：

- `M[v]`：op `v` 在展开窗口里的启动 issue 时间。
- `phase[v] = M[v] mod I`：op `v` 的 modulo 相位。
- `L = max(M[v] + duration[v])`：所有 op 发射完成所需的窗口长度。

`M[v]` 本身就是“每个 op 恰好出现一次”的表达，因此不再创建 `op[v,t]` 这种 one-hot 布尔网格。

### 主要约束

依赖约束：

```text
M[v] - M[u] + delta * I >= d
```

其中：

- `u -> v` 是依赖边。
- `d` 是 producer 侧提供的 delay。
- `delta = 0` 表示同一轮迭代。
- `delta = 1` 表示相邻迭代的 loop-carried hazard。

资源约束：

- `phase[v] = M[v] mod I` 把绝对时间折叠到 modulo 周期。
- 对 `cap = 1` 且连续占用的资源，用环形 interval no-overlap 表达。
- WGMMA 的 TensorCore issue 会按多个连续 `TC` slot 建模。
- wait/try_wait barrier 使用独立 `BARRIER` issue slot，并且 barrier 的 issue interval 与其它 op 互斥。

窗口和目标：

```text
L = max(M[v] + duration[v])
minimize 100 * L - sum(M[v])
```

目标优先压短 issue 窗口；在相同窗口下，倾向把 op 往后放一些。

### 搜索多个 plan

`solve_min_I()` 会先用二分缩小 `I` 的搜索区间，再在线性区间找最小可行 `I`。当前代码还有 `plan_cache`，同一个 `I` 只求一次。

找到最小 `I` 后，如果 `start_ii <= 1`，还会按 `estimated_total_latency` 的若干分位补充额外 plan。也就是说 joint 阶段可能会尝试不止一个 naive plan。

每个 naive plan 形如：

```python
{
    "I": I,
    "L": L,
    "M": {op_idx: start_time},
    "num_stages": ceil(L / I),
    "modular_rrt": [...],
}
```

## 第二阶段：joint optimize

`_solve_smt_joint_optimize()` 接收 naive plan，并把 `_StmtInfo` 转成 `HeddleScheduler` 的 `OpNode` 图。joint 阶段固定 `base_I`，不会重新搜索 II；它只在候选 `L` 窗口里重新安排启动时间、warp 和资源。

```python
solver.schedule_joint(
    min_ii=base_I,
    max_ii=base_I,
    window=solve_window,
    optimize=optimize,
)
```

### 输入如何变成 joint 模型

joint 阶段先读 naive plan：

- `base_M = plan["M"]`：作为 CP-SAT hint，并用于估算 buffer 多版本 footprint。
- `base_I = plan["I"]`：作为固定 II。
- `base_L = plan["L"]`：作为基础窗口和 stage 估计来源。
- `expect_num_stage = ceil(base_L / base_I)`：用于估算 MultiVersionBuffer 前的 shared buffer 版本数。

然后构造统一 buffer registry：

- 收集每个 op 的 `reads/writes`。
- 按物理 `buffer.name` 聚合，而不是按 op-prefixed output 名称聚合。
- shared buffer 和 `shared.barrier` 归为 `SMEM`，其它归为 `RMEM`。
- 对 shared producer/consumer buffer 估算 `version_factor`，把后续 MultiVersionBuffer 会扩出的 ring-buffer footprint 提前反映到 solver 的 SMEM footprint 中。

每个 op 会被转换为一个 `OpNode`：

- `name = f"s{idx}"`。
- `resource_type`：TMA、TensorCore、ALU、SFU 或 Barrier。
- `latency`：依赖 readiness delay。
- `reservation`：issue slot 占用序列。
- `outputs`：写出的 `OutputValue`，包含 storage、footprint、buffer_name。
- WGMMA：`warp_count = 4`，`warp_align = 4`。
- true TMA：`replicable = True`，并标记为 variable latency op。

依赖边输入：

- 同迭代依赖来自 `deps_all`，建成 `distance=0`。
- 真实跨迭代 hazard 才建 `distance=1`，当前包括自读写同一 buffer 和 wait barrier。
- 不会给所有 op 加 `v -> v, distance=1` 的 consistency self-edge，因为“每个 op 一个 `Tv[v]` 且固定 `ii`”已经表达了 modulo 模板的相位一致性。

额外语义约束输入：

- `same_warpgroup_pairs`：wait barrier 和其保护的 shared-memory reader 需要在同一 raw warpgroup。
- `not_all_same_warpgroup_sets`：当后置 reg liveness 发现多个 peak producer 挤在同一 warpgroup 时，Round2 会追加“这些 producer 不能全在同一 raw warpgroup”的反馈约束。
- `start_hints`：来自 naive `base_M`，只作为 hint，不是硬约束。

### Phase B 主要变量

`HeddleScheduler._solve_phase_b(ii, L, optimize=True)` 中的核心变量是：

- `Tv[v]`：op `v` 的启动时间，域为 `[0, L - 1]`。
- `warp[(v, w)]`：op `v` 是否占用 warp `w`。
- `issue_warp[(v, w)]`：op 的 issue 起点 warp；多 warp op 用它表达连续 warp block 的起点。
- `phase[v] = Tv[v] mod ii`：用于 modulo issue/FU 资源约束。
- `start_le(v, bound)`：`Tv[v] <= bound` 的 reified Bool，用于 liveness 推导。
- `smem_live/rmem_live` 相关 Bool：当内部 liveness 开启时，用来表示某个 output copy 在时间 `tau` 是否 live。

变量含义上的边界：

- `Tv/M` 是绝对模板时间，不是 modulo phase。
- `phase` 才是 modulo 相位。
- `ii/base_I` 是 steady-state 相邻迭代启动间隔，是 joint 阶段硬约束。
- `window/L` 在 Phase B 是 `Tv` 的搜索域上界；最终返回的 `optimized_L` 会用 `Tv + reservation_len` 重新计算。
- `live_start/live_end/live_end_exclusive` 是按 `Tv + iter_offset * ii` 展开的绝对时间，不是 modulo phase。

## 约束部分：当前到底输入了哪些约束

这一节按“输入来源 -> CP-SAT 表达”列出当前主路径中真正进入模型或后置复核的约束。

### 1. 启动时间与固定 II

输入来源：

- naive plan 的 `base_I`。
- `_solve_smt_joint_optimize()` 给出的候选 `solve_window`。

模型表达：

```text
Tv[v] in [0, L - 1]
phase[v] = Tv[v] mod ii
ii = base_I
```

这里没有 `op[v, i, t]` 三维布尔变量。每个 op 只有一个 `Tv[v]`，表示 steady-state 模板中的启动时间；第 `k` 个迭代副本默认在 `Tv[v] + k * ii` 发生。

### 2. warp 分配约束

输入来源：

- `OpNode.warp_count`
- `OpNode.warp_align`
- `OpNode.replicable`
- `num_warps`

模型表达：

- 单 warp op 或 replicable op：`sum_w warp[v,w] == 1`。
- 多 warp op：选择唯一连续 warp block 起点。
- `warp_align` 要求起点满足对齐，例如 WGMMA 起点按 4 对齐。
- 如果 `warp_count > num_warps` 或没有合法起点，直接返回不可行。

对 WGMMA 来说，它会选择一个连续 4 warp 的 raw warpgroup；对 true TMA 来说，它是 replicable 的单 warp variable latency op。

### 3. variable latency warpgroup 约束

输入来源：

- true TMA op 被标记为 `is_varialble_latency=True`。

模型表达：

- 所有 variable latency op 必须在同一个 raw warpgroup，即 `warpId // 4` 相同。
- variable latency op 和 non-variable latency op 不能在同一个 raw warpgroup。

这个约束把 TMA producer 角色和普通 consumer/WGMMA 角色分开，避免 PCWS lowering 后角色混用。

### 4. wait/shared-reader 同 warpgroup 约束

输入来源：

- `_shared_wait_reader_pairs()` 从 `infos` 中找到 wait barrier 与后续读取 producer shared buffer 的 consumer。

模型表达：

```text
warp(wait) // 4 == warp(reader) // 4
```

它不是时间依赖，而是 warpgroup placement 约束。时间顺序仍由 `deps_all` 中的依赖边表达。

### 5. 依赖约束

输入来源：

- `deps_all` 生成同迭代边。
- 自读写 buffer 和 wait barrier 生成跨迭代 self edge。
- `OutputValue.spill_cost` 可给跨 warp producer/consumer 增加延迟。

模型表达：

```text
Tv[v] - Tv[u] >= delay - distance * ii
```

其中：

- `distance=0`：同一轮迭代。
- `distance=1`：下一轮迭代的消费者/同一 op。
- `delay`：通常取 producer 的 dependency delay。

如果 edge 有 spill cost：

- producer 和 consumer 在同一 warp：使用原始 `delay`。
- 不同 warp：使用 `delay + spill_cost`。
- 如果 `disallow_spills=True`：强制同 warp。

### 6. blocking sync 与 spill 并发约束

输入来源：

- `edge.blocking_sync`
- `OutputValue.spill_cost`

模型表达：

- blocking sync：除了普通时间依赖，还要求 parent warp 覆盖 child warp，并把 `[Tv[v] - delay, Tv[v])` 近似成阻塞窗口；同一 warp 上其它 op 不能与这个窗口重叠。
- spill concurrency：当 producer/consumer 跨 warp 且有 spill cost，把 `[Tv[v] - spill_cost, Tv[v])` 看成占用接收方 warp 的 spill 窗口；接收方 warp 上其它 op 不能与之重叠。

这两类约束用 optional interval + `AddNoOverlap` 表达，避免枚举大量 `(time, other-op)` 冲突子句。

### 7. subcore issue 约束

输入来源：

- 已经求出的或待求的 `issue_warp`。
- 每个 op 的 `reservation` 长度。

模型表达：

Hopper 中 raw warpgroup 内同号 warp 共享 subcore issue 槽，近似为：

```text
subcoreId = warpId % 4
```

如果两个 op 的 issue 起点 warp 满足 `wu % 4 == wv % 4`，则它们的 modulo issue interval 不能重叠。不同 subcore 不加这条互斥。

### 8. FU / issue reservation 容量约束

输入来源：

- `OpNode.reservation`
- `fu_caps`

当前容量大致是：

```python
{
    TMA: 255,
    TensorCore: 1,
    ALU: 64,
    SFU: 16,
    Barrier: 1,
}
```

模型表达分两类：

- `cap = 1` 且 reservation 是连续区间：用 modulo 环形 interval no-overlap。
- 其它资源形状：先 `_fold_reservations(ii)` 折叠，再对每个 modulo phase 做容量求和。

TensorCore、Barrier 这类 `cap=1` 资源通常走 interval 互斥；ALU/SFU/TMA 这种容量较大的资源通常走逐 phase 求和。

### 9. barrier issue slot 独占

输入来源：

- wait/try_wait barrier 的 `reservation = [{Barrier: 1}]`。

模型表达：

barrier op 不只和其它 barrier 互斥，还和所有其它 op 的 issue interval 互斥。它表达的是“同步 issue slot 单独发射”，不是把 barrier 当成同时占满 TMA/TC/ALU/SFU。

### 10. Liveness / memory 约束总览

Twill 的 memory/liveness 约束可以理解成三层：

- 先定义 value 什么时候 live。
- 再把 live value 映射到 memory 类型。
- 最后对任意时间 `t` 加 capacity 上限。

如果沿用 Twill 的三维布尔视角，核心对象通常是：

```text
op[v, i, t]      = 第 i 个迭代副本里的 op v 是否在 t 启动
live[x, i, t]    = 第 i 个迭代副本产生的 value x 在 t 是否 live
dead[x, i, t]    = value x 在 t 前是否已经死亡或可释放
```

当前 Heddle 没有显式建立完整的 `op/live/dead` 三维网格，而是用 `Tv[v]` 直接推导 live interval：

```text
producer_time(x, k) = Tv[producer(x)] + k * ii
consumer_time(x, k) = Tv[consumer(x)] + (k + distance) * ii
```

其中 `k` 是 `iter_offset`。也就是说，Twill 里由 `LIVEPROP/DEADPROP` 逐点传播出来的 live 状态，在当前代码里被压缩成：

```text
is_produced(x,k,tau) = producer_time(x,k) <= tau
all_consumers_done(x,k,tau) = 所有 consumer_time(x,k) <= tau
live(x,k,tau) = is_produced && !all_consumers_done
```

对于 `LifetimeSemantic.DEAD_ON_ENTRY` 的 output，一份 copy 在 producer 启动后开始 live，直到最后一个消费者启动/消费边界为止。如果没有消费者，这份临时 value 直接视为不需要保留。非 `DEAD_ON_ENTRY` 的 output 目前按 produced 后持续 live 处理。

### 11. Heddle 的 value 和 copy 如何对应 Twill

Twill 的公式通常按 value `x`、迭代副本 `i`、时间 `t` 展开。当前代码中的对应关系是：

| Twill 概念 | 当前代码 |
| --- | --- |
| `x` / value | `OutputValue`，来自某个 op 的写 buffer |
| `i` / iteration copy | `iter_offset` |
| `t` / time | `tau` |
| `op[v,i,t]` | 不显式建；由 `Tv[v] + i * ii == t` 隐含 |
| `live[x,i,t]` | `smem_l_*` / `rmem_l_*`，或 fixed check 里的 `_copy_live()` |
| `mem(x)` | `OutputValue.storage`，即 `RMEM` 或 `SMEM` |
| `size(x)` | `OutputValue.footprint_bytes` |
| `capacity(mem)` | `reg_limit` 或 `smem_limit` |

`OutputValue` 的来源是 joint 阶段的 unified buffer registry：

- 每个写 buffer 生成一个 `OutputValue`。
- `buffer_name` 保留物理 buffer 身份。
- shared buffer 和 `shared.barrier` 归为 `SMEM`。
- 其它 buffer 归为 `RMEM`。
- footprint 使用 `_estimate_buffer_footprint_bytes()`，并按物理 buffer 聚合取最大值。

这样做是为了避免 `s3_w_acc`、`s5_w_acc` 这类 op-prefixed output 名字把同一个物理 buffer 重复计入容量。

### 12. 多迭代 live copy

Twill 的 modulo schedule 会在一个 steady-state 窗口里同时看到多个迭代副本。当前实现用 `iter_offsets` 表达这个展开范围：

```python
max_iter_overlap = (L - 1) // ii
iter_offsets = range(-max_iter_overlap, max_iter_overlap + 1)
```

含义是：在求解窗口 `[0, L)` 内，当前模板迭代前后的若干副本都可能与当前迭代重叠。对每个 output copy `(x, iter_offset)`：

```text
live_start = Tv[producer] + iter_offset * ii
consume_time = Tv[consumer] + (iter_offset + distance) * ii
```

如果 `distance > 0`，消费者读的是下一轮或更晚迭代对应的值，这就是 Twill 里 loop-carried live value / incoming live 的来源。`smt.py` 里会把有 `distance > 0` 消费者的 output 放进 `loop_carried` 集合；当前主要用于解释和后续 live copy 枚举，主公式直接通过 `iter_offset + distance` 体现跨迭代关系。

### 13. SMEM 容量约束

输入来源：

- unified buffer registry 里的 SMEM `OutputValue`。
- `smem_allocations_by_buffer`：按物理 buffer 聚合后的静态/多版本 shared allocation footprint。
- `smem_limit`：默认约为 H100 CTA shared memory 上限；如果静态 allocation floor 已超过默认 limit，会把 limit 提高到该 floor，避免 trivially UNSAT。

SMEM 在 Phase B 的 CP-SAT 主模型里仍会建容量约束。对每个 `tau in [0, L)`：

1. 遍历所有 SMEM output `x`。
2. 遍历可能重叠的 `iter_offset`。
3. 建 `is_produced(x,k,tau)`：

   ```text
   Tv[producer] + k * ii <= tau
   ```

4. 如果 output 是 `DEAD_ON_ENTRY` 且有消费者，建 `all_c_done(x,k,tau)`：

   ```text
   对所有 consumer c:
   Tv[c] + k * ii <= tau - distance(c) * ii - zero_latency_offset
   ```

5. 建 live：

   ```text
   live(x,k,tau) = is_produced(x,k,tau) && !all_c_done(x,k,tau)
   ```

6. 对同一个 physical buffer 做 OR 聚合：

   ```text
   buffer_live(buffer,tau) = OR(live(x,k,tau) for x 写同一 buffer)
   ```

7. 加容量约束：

   ```text
   static_smem_total
     + sum(buffer_footprint[buffer] * buffer_live(buffer,tau))
     <= smem_limit
   ```

这里有两个工程化处理：

- 如果某个 buffer 已经在 `smem_allocations_by_buffer` 中，说明它已经作为静态/多版本 allocation 计入 `static_smem_total`，就不再按 output copy 重复计入动态项。
- 同一个 SMEM physical buffer 只按 `buffer_name` 计一次，不因为多个 producer output 名称不同而重复计。

这和 Twill 的 `sum live[x,i,t] * size[x] <= capacity[mem]` 是同一类约束，只是 Heddle 先按 physical buffer 合并，再区分静态 allocation 与动态 output 项。

### 14. RMEM / register 容量约束

输入来源：

- RMEM `OutputValue`。
- `reg_limit`。
- output 的消费者列表和 `distance`。
- producer 的 warp assignment。

`_solve_phase_b()` 里保留了内部 RMEM checkpoint liveness 编码，但当前 `_solve_smt_joint_optimize()` 主路径调用 joint solver 时传的是：

```python
optimize=True,
enable_liveness=False,
```

所以主路径不会在 CP-SAT 内展开 RMEM liveness 网格，而是求出固定 `schedule + warp_assign` 后调用 `_check_fixed_liveness()` 做确定性复核。

如果打开内部 RMEM liveness，它的 CP-SAT 形式是：

1. 选择 checkpoint 集合：

   ```text
   tau in {0, step, 2*step, ..., L-1}
   再加上 start_hints 附近和 consumer hint 附近的 tau
   ```

2. 对每个 checkpoint、RMEM output、`iter_offset` 建 live。
3. 把 live 和 producer warp 绑定：

   ```text
   live_on_warp(w,x,k,tau) = live(x,k,tau) && warp[producer(x), w]
   ```

4. 按 `(warp, buffer_key, iter_offset)` 聚合：

   ```text
   live_copy(w,buffer,k,tau) = OR(live_on_warp(...))
   ```

5. 对每个 warp 加 register capacity：

   ```text
   sum(footprint(buffer,k) * live_copy(w,buffer,k,tau)) <= reg_limit
   ```

当前主路径的 fixed liveness check 做的是同一件事，只是不用 CP-SAT 变量，而是在 Python 里对固定解直接计算：

```text
if fixed_M[producer] + iter_offset * base_I > tau:
    not live
elif 所有 consumer_time <= tau:
    not live
else:
    live
```

然后：

- RMEM 按 warp 统计。
- 同一物理 RMEM buffer 按 `(buffer_key, iter_offset)` 聚合。
- `reg_peak[w]` 记录每个 warp 的峰值。
- 超过 `reg_limit` 时返回 `reason="reg_limit"`、`warp`、`tau`、`usage`、`limit` 和 top live buffers。

这相当于 Twill memory capacity 约束的后验检查版本：

```text
forall tau, warp:
  sum(live[x,k,tau] * footprint[x]) <= reg_limit
```

区别是 Heddle 的 RMEM capacity 是 per warp 的，而 Twill 论文里的 memory capacity 通常按 memory 类型 `m` 全局描述。

### 15. MultiVersionBuffer 前的 SMEM footprint

Heddle joint optimize 当前运行在 `tilelang.transform.MultiVersionBuffer()` 之前，因此 TIR 里仍然可能只看到一个逻辑 shared buffer。但下游 pass 会按 pipeline stage 把 producer/consumer shared buffer 扩成 ring buffer。

为避免 solver 低估 SMEM，用 `_pipeline_version_factors()` 预估版本数：

- `expect_num_stage = ceil(base_L / base_I)` 给出 stage 上界。
- 按 naive `base_M` 收集每个 shared buffer 的 writes/reads。
- 在 `version_window` 内估算同一时间最多有多少个 iteration copy live。
- `version_factor = min(expect_num_stage, max_live_copies)`。
- 最终 `footprint_bytes = base_footprint_bytes * version_factor`。

这一步不是 Twill 公式里的 live propagation，而是 Heddle 针对 TileLang pass 顺序的保守 footprint 修正。

### 16. liveness 失败后的反馈约束

如果 fixed liveness check 因 `reg_limit` 失败，transform 侧不会马上放弃，而是从失败信息里取峰值 live buffers：

- `live_detail` 里按 bytes 从大到小列出 top buffer。
- 反查这些 RMEM buffer 的 producer op。
- 如果能找到至少两个 producer，就追加：

```text
not_all_same_warpgroup(producer_1, producer_2, ...)
```

在 CP-SAT 里表达为：对每个 raw warpgroup，这些 producer 不能全部落在同一个 warpgroup。它不是 Twill 原始 liveness 公式的一部分，而是 Heddle 为了避免重建完整 RMEM CP-SAT live 网格加的轻量反馈约束。

### 17. 优化目标

输入来源：

- `optimize=True`。
- `Tv` 和 reservation 长度。

模型表达：

```text
max_end = max(Tv[v] + len(reservation[v]))
minimize L * N * max_end + sum(Tv)
```

它倾向压短实际 issue span，同时让整体启动时间更靠前。这里的 `L` 是当前求解窗口常量，不是被最小化的变量；最终输出的 `optimized_L` 会在 transform 侧重新计算。

## 当前 joint 求解策略

当前 `_solve_smt_joint_optimize()` 主路径不是先跑一次 `optimize=False` feasibility，再跑 optimize。实际流程是：

1. 构造候选窗口：

   ```text
   window = max(base_L, max(base_M) + 1, base_I)
   candidates = [window, window + I, window + 2I, window + 3I]
   ```

2. 对每个候选 `solve_window` 直接运行：

   ```python
   _run_joint_solver(
       ii=base_I,
       solve_window=solve_window,
       optimize=True,
       enable_liveness=False,
   )
   ```

3. 如果 CP-SAT 返回解，立刻调用 `_post_check_liveliness()`，也就是后置 `_check_fixed_liveness()`。

4. 如果 liveness 通过，返回该 joint 解。

5. 如果因为 `reg_limit` 失败，并能提取到至少两个 peak producer，则追加 `not_all_same_warpgroup` 约束做 Round2。

6. 如果当前窗口失败，继续尝试更大的 `L`。

7. 所有候选窗口都失败时返回 `None`，外层流程再决定是否 fallback 到 naive/旧 Phase B 路径。

因此当前主路径的分工是：

- CP-SAT 主模型：负责结构性约束、warp placement、FU/issue 资源、SMEM 容量和紧凑目标。
- Python fixed liveness：负责最终 RMEM/SMEM 峰值复核，并给 reg overflow 生成轻量反馈约束。

## 返回结果

joint 成功后，transform 侧把 scheduler 的 `schedule` 和 `warp_assign` 转回 op id：

```python
optimized_M = {
    idx: int(schedule[f"s{idx}"])
    for idx in ops
    if f"s{idx}" in schedule
}
```

然后重新计算：

```text
optimized_L = max(optimized_M[idx] + len(reservation[idx]))
```

最终 plan 会更新：

- `I`：固定为 `base_I`。
- `L`：joint 结果重新计算出的 `optimized_L`。
- `M`：joint 后的启动时间。
- `window`：当前 CP-SAT 使用的搜索窗口。
- `warp_assign`：op id 到起始 warp 的映射。
- `reg_peak` / `smem_peak`：后置 liveness 复核得到的峰值。
- `variable_lifetimes`：按物理 buffer 聚合的生命周期信息。
- `modular_rrt`：按 joint 后 `M` 和 reservation 重新折叠。
- `ordering`：按 `(time, op_id)` 排序得到的执行顺序。

`variable_lifetimes` 中每个 copy 的时间含义是绝对展开时间：

- `iter_offset`：相对模板迭代的偏移。
- `live_start = Tv[producer] + iter_offset * I`。
- `consume_time = Tv[consumer] + (iter_offset + distance) * I`。
- `live_end_exclusive`：最后一个消费者消费后的释放边界。
- `live_end = live_end_exclusive - 1`。

这些字段不是 modulo phase。

## 和 Twill 4.1 约束的对应关系

Twill 论文里的 constrained modulo scheduling 常用三维布尔变量：

```text
op[v, i, t] = 操作 v 的第 i 个迭代副本是否在时间 t 被调度
```

当前代码没有逐字实现这个网格，而是用更紧凑的 IntVar/phase/interval 表达同类约束。

| Twill 约束 | 当前实现 |
| --- | --- |
| UNIQUENESS | 每个 op 一个 `M[v]` 或 `Tv[v]`，天然只调度一次 |
| CONSISTENCY | 固定 `I/ii`，每个 op 一个 steady-state 模板时间，副本隐式按 `+ k * I` 展开 |
| COMPLETION | naive 用 `M[v] <= H - duration[v]` 和 `L=max(end)`；joint 用 `Tv[v] in [0,L-1]`，最终 `optimized_L` 另算 |
| DEPENDENCE | `M[v]-M[u]+delta*I>=d` 或 `Tv[v]-Tv[u]>=delay-distance*ii` |
| CAPACITY | 连续 `cap=1` reservation 用 modulo interval no-overlap，其它资源逐 phase 求和 |
| MEMORY/LIVENESS | SMEM 在 CP-SAT 内按 `tau` 建 `live(x,k,tau)` 和容量约束；RMEM 支持 checkpoint live 约束，但当前主路径关闭内部 RMEM liveness，改用 fixed liveness 后验复核和 Round2 warpgroup 反馈 |

### Memory / Liveness 逐项对齐

下面按 `twill_paper.md` 中图 5 的 memory/liveness 约束逐项对齐当前实现。为了避免符号混淆，先给出变量映射：

| Twill 符号 | 含义 | Heddle 当前对应 |
| --- | --- | --- |
| `v` | 产生某个 SSA value 的 op | producer op，对应 `OpNode` / `OutputValue` 的 producer |
| `u` | 使用 `v` 结果的 consumer op | `consumers_of[xi]` 中的 consumer |
| `i` | 展开后的 iteration copy | `iter_offset` |
| `t` | 展开时间 | `tau` |
| `live[v,i,t]` | 第 `i` 个 copy 中，`v` 的结果在 `t` 是否 live | CP-SAT 中的 `smem_l_*` / `rmem_l_*`，或 fixed check 的 `_copy_live()` |
| `footprint(v,m)` | value 在 memory `m` 中占用多少空间 | `OutputValue.footprint_bytes`，按 `buffer_name` 聚合 |
| `capacity(m)` | memory 类型容量 | `smem_limit` 或 `reg_limit` |
| `δ` | dependence distance | `dependency_distance` / `distance` |

#### 1. MEMORY CAPACITY

Twill 约束：

```text
forall t, m:
  sum live[v,i,t] * footprint(v,m) <= capacity(m)
```

语义：任意时间、任意 memory 类型上，同时 live 的 value 总 footprint 不能超过容量。

Heddle 对齐：

- SMEM：在 CP-SAT 主模型里按每个 `tau in [0, L)` 建容量约束。
- RMEM：`smt.py` 保留 checkpoint 版 CP-SAT 约束；但当前 `_solve_smt_joint_optimize()` 主路径传 `enable_liveness=False`，所以 RMEM 主要由 `_check_fixed_liveness()` 在固定解上后验检查。
- Twill 按 value 累加；Heddle 先按 physical buffer 聚合，避免同一 buffer 因多个 op-prefixed output 名字重复计数。

SMEM 的模型形式近似是：

```text
static_smem_total
  + sum(buffer_footprint[b] * buffer_live[b,tau])
  <= smem_limit
```

RMEM fixed check 的形式近似是：

```text
forall tau, warp:
  sum(live[buffer,iter_offset,tau] * footprint(buffer))
  <= reg_limit
```

这里 RMEM 是 per-warp capacity，比 Twill 图 5 的 memory 类型容量更接近 Twill 图 6 的 register limit 约束。

#### 2. INIT

Twill 约束：

```text
forall v:
  exists (v,u,d,delta>0) in E
  iff live[v, ceil(L/I)-1, T]
```

语义：在展开片段的最后时间 `T`，只有会被未来 iteration 使用的 loop-carried value 仍然 live。

Heddle 对齐：

- Heddle 不显式设置 `live[v,last_copy,T]` 这种 INIT Bool。
- 它从依赖边里收集 `consumers_of[xi]`，如果某条边 `distance > 0`，说明这个 output 可能跨迭代被使用。
- live copy 的时间直接用 consumer 时间表达：

```text
consume_time = Tv[consumer] + (iter_offset + distance) * ii
```

因此 loop-carried live 不是靠末尾 INIT 传播出来，而是在枚举 `iter_offsets` 时自然覆盖。只要 `distance > 0`，当前 copy 的释放时间就会被推到未来 iteration 的 consumer 之后。

#### 3. LIVEPROP-1

Twill 约束：

```text
live[v,i,t] && op[v,i,t] => not live[v,i,t-1]
```

语义：如果 value 在 `t` live，且它正是在 `t` 被定义，那么定义点之前 `t-1` 不可能 live。

Heddle 对齐：

- Heddle 不做 backward propagation。
- 它直接定义 produced 条件：

```text
is_produced(x,k,tau) = Tv[producer] + k * ii <= tau
```

如果 `tau` 早于 producer time，`is_produced=False`，则 live 必然为 false。这等价于把 LIVEPROP-1 的“定义点之前不 live”直接编码进 live interval 的左边界。

#### 4. LIVEPROP-2

Twill 约束：

```text
live[v,i,t] && not op[v,i,t] => live[v,i,t-1]
```

语义：如果 value 在 `t` live，且 `t` 不是定义点，那么 liveness 向前传播到 `t-1`。

Heddle 对齐：

- Heddle 用区间表达替代逐点传播。
- 对 `DEAD_ON_ENTRY` output，只要满足：

```text
producer_time <= tau
and not all_consumers_done(tau)
```

就认为该 copy live。

这个条件一次性覆盖了从 producer 到最后一个 consumer 之前的整段区间，不需要逐个 `tau` 写 `live(t) -> live(t-1)`。

#### 5. DEADPROP-1

Twill 约束：

```text
not live[v,i,t] && OR op[u,i+delta,t] => live[v,i,t-1]
```

其中 `(v,u,_,delta) in E`，`u` 是使用 `v` 结果的 consumer。

语义：如果 `t` 时刻有 consumer 使用 `v`，那么在使用发生之前，`v` 的结果必须 live；使用之后如果没有其它 use，才可以 dead。

Heddle 对齐：

- Heddle 通过 consumer time 决定右边界：

```text
consumer_time = Tv[consumer] + (iter_offset + distance) * ii
```

- fixed check 中 `_copy_live()` 的判断是：只要还有任意 `consume_time > tau`，这份 copy 就 live。
- CP-SAT 内部 liveness 也是建 `all_c_done`，然后用 `!all_c_done` 表示仍 live。

所以 DEADPROP-1 对应到当前实现就是：consumer 发生之前必须 live；consumer 发生后，如果所有 consumer 都完成，才允许 dead。

#### 6. DEADPROP-2

Twill 约束：

```text
not live[v,i,t] && AND not op[u,i+delta,t] => not live[v,i,t-1]
```

语义：如果 value 在 `t` 已经 dead，而且 `t` 没有任何 consumer 使用它，那么 deadness 向前传播，直到遇到某个 use。

Heddle 对齐：

- Heddle 不显式建 `dead[v,i,t]`。
- 它用 `all_consumers_done(tau)` 表达 dead 区间：

```text
all_consumers_done(x,k,tau) =
  forall consumer:
    consume_time <= tau
```

一旦所有 consumer 都已经完成，`live=False`；如果继续往后看，仍然满足所有 consumer done，因此一直 dead。这个区间式表达等价于 DEADPROP-2 的传播结果。

#### 7. REGISTER LIMIT

`twill_paper.md` 还把 register limit 写成：

```text
forall t,w:
  sum live[v,i,t] * opw[v,w] * regs(v) <= reg_limit()
```

语义：任意时间、任意 warp 上，分配给该 warp 且仍 live 的值，其寄存器总量不能超过限制。

Heddle 对齐：

- `warp[(producer, w)]` 对应 Twill 的 `opw[v,w]`。
- `OutputValue.footprint_bytes` 对应 `regs(v)`，单位是 bytes。
- 内部 checkpoint liveness 会建：

```text
live_on_warp(w,x,k,tau) = live(x,k,tau) && warp[producer(x), w]
sum footprint * live_on_warp <= reg_limit
```

- 当前主路径的 fixed check 则直接展开 producer 的 selected warps，按 warp 累加 live RMEM bytes。

如果 fixed check 发现 `reg_limit` 超限，Heddle 会从 `live_detail` 里找 top live buffers 和 producer，追加 `not_all_same_warpgroup` 反馈约束重试。这是工程化近似：它不等价于 Twill 的完整 register SMT 约束，但能把 peak producer 分散到不同 warpgroup，降低单 warpgroup 的 register pressure。

## 调试日志阅读顺序

推荐按这些标记看：

1. `enter _solve_naive_modulo_sched`：进入 naive 阶段。
2. `latencies=...` / `duration=...`：确认 readiness delay 和 issue occupancy 没混。
3. `[Modulo Sched] Trying : I = ...`：查看当前 II 搜索。
4. `solve done`、`I = ...`、`L = ...`、`M = ...`：确认 naive plan 是否非空。
5. `----start  _solve_smt_joint_optimize`：确认进入 joint。
6. `expect_num_stage=...`：确认 `ceil(base_L/base_I)`。
7. `---- num_warps (consumer only) = ...`：确认 warp 域。
8. `----- HeddleSCheduler: reg_limit=..., smem_limit=..., num_warps=..., optimize=True`：进入 Phase B。
9. `[SMT] 遍历空间找最优解 : I= ... L = ...`：当前候选窗口。
10. `--- SMT liveness check success`：后置 liveness 复核通过。
11. `Round2 reg feedback`：reg 超限后追加分散 warpgroup 约束重试。
12. `---warp_assign=...`、`---optimized_L=...`、`---optimized_M=...`：最终 joint 输出。

## 压缩心智模型

可以把当前实现理解成：

```text
TIR statements
  -> infos_list / deps_all / all_indices
  -> naive CP-SAT:
       search I, solve M, issue/FU modulo capacity
  -> joint input lowering:
       OpNode + OutputValue + buffer registry + dependency edges
  -> joint CP-SAT:
       fixed I, solve Tv + warp_assign + warpgroup + FU/subcore/barrier + SMEM
  -> fixed liveness check:
       verify RMEM/SMEM peaks on the concrete schedule
  -> optimized plan:
       M/L/order/warp_assign/lifetimes/peaks
```

最关键的变量边界是：

- `I/base_I/ii`：固定 steady-state 启动间隔。
- `M`：naive 阶段的启动时间。
- `Tv`：joint Phase B 的启动时间。
- `phase`：`M` 或 `Tv` 对 `I` 取模后的 issue 相位。
- `L/window`：求解窗口；最终 `optimized_L` 是结果 span。
- `latency`：依赖 readiness。
- `duration/reservation`：issue slot occupancy。
- `warp_assign`：joint 阶段得到的起始 warp。
- `variable_lifetimes`：按物理 buffer 聚合、按绝对展开时间描述的生命周期。
