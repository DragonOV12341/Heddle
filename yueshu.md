# SMT joint 阶段 CP-SAT 变量与约束说明

本文档整理当前 `SMT joint` 路径中所有主要变量和约束的 CP-SAT 含义及使用方式。当前实现虽然函数名里仍保留 `SMT`，但 joint 阶段真正的后端是 OR-Tools CP-SAT：

- 入口：`Heddle/heddle/transform/heddle_consumer_schedule.py::_solve_smt_joint_optimize`
- 求解器：`Heddle/heddle/scheduler/smt.py::HeddleScheduler._solve_phase_b`
- 调用方式：`_solve_smt_joint_optimize()` 先把 TIR 语句、依赖、buffer、latency、resource、warp 需求整理成 `OpNode` / `OutputValue`，再调用 `HeddleScheduler.schedule_joint(min_ii=I, max_ii=I, window=L, optimize=True)`。

## 1. 输入对象

### 1.1 基础调度输入

`_solve_smt_joint_optimize(op_deps, infos, all_indices, mod_sched_plan, kernel_num_threads)` 使用 naive modulo scheduler 的结果作为基础：

- `base_M = mod_sched_plan["M"]`：原始语句 id 到单次循环内启动时间的映射。
- `base_I = mod_sched_plan["I"]`：固定 initiation interval。joint 阶段当前不搜索新的 `I`，而是在 `base_I` 上优化 `M` 和 warp 分配。
- `base_L = mod_sched_plan["L"]`：基础窗口长度。
- `expect_num_stage = ceil(base_L / base_I)`：估计 pipeline stage 数，用于 shared buffer 多版本 footprint 估算。
- `expect_consumer_warps = mod_sched_plan["heddle_expect_consumer_warps"]`：consumer 侧 warp 数。
- `num_warps = expect_consumer_warps + 4`：传给 joint solver 的 raw warp 域，额外 `+4` 表示 producer/TMA warpgroup。

### 1.2 语句展开

每个原始 stmt `idx` 可能有 `op_instance_count` 个子实例：

- child 表示为 `(idx, inst)`。
- child name 是 `s{idx}` 或 `s{idx}__u{inst}`。
- CP-SAT 里每个 child 对应一个 `OpNode`。

这个展开用于 WGMMA/TMA 等多 issue slice 语句，使求解器可以约束同一 TIR stmt 内多个 issue slice 的顺序。

### 1.3 OpNode 字段

每个 `OpNode` 会带入 CP-SAT：

- `name`：求解器内 op 名。
- `resource_type`：`TMA` / `TC` / `ALU` / `SFU` / `BARRIER`。
- `latency`：依赖边使用的 ready latency。
- `reservation`：issue 资源占用表，例如 `[{TC: 1}, {TC: 1}, ...]`。
- `outputs`：该 op 写出的 `OutputValue`，用于 RMEM/SMEM liveness。
- `warp_count`：该 op 需要占用几个 warp。当前 WGMMA 为 `4`，普通 op 为 `1`。
- `warp_align`：多 warp op 的起始 warp 对齐。当前 WGMMA 为 `4`。
- `replicable`：true TMA 可复制，按单 warp 选择建模。
- `is_varialble_latency`：true TMA 被标成 variable latency op，用于 warpgroup 隔离。

### 1.4 OutputValue 字段

`OutputValue` 表示一个被写出的 buffer/value：

- `name`：内部 value 名。
- `storage`：`RMEM` 或 `SMEM`。
- `footprint_bytes`：容量占用，SMEM 会根据 pipeline stage 做版本系数放大。
- `spill_cost`：跨 warp 使用时的额外代价，目前 transform 侧一般没有显式设置。
- `lifetime`：当前 joint transform 写出的值使用 `DEAD_ON_ENTRY`。
- `buffer_name`：聚合相同物理 buffer 的 key。

## 2. CP-SAT 决策变量

以下变量都在 `HeddleScheduler._solve_phase_b(ii, L)` 中创建。

### 2.1 启动时间变量 `Tv[v]`

```text
Tv[v] = IntVar(0, L - 1)
```

含义：op `v` 在当前 modulo window 内的绝对启动时间。

使用方式：

- 依赖约束直接比较 `Tv[consumer] - Tv[producer]`。
- `phase[v] = Tv[v] mod ii` 用于 modulo FU 资源约束。
- objective 最小化 `max(Tv + issue_duration)` 和 `sum(Tv)`。
- 求解结果中的 `schedule[node.name]` 来自 `Tv[v]`。

### 2.2 warp 选择变量 `warp[(v, w)]`

```text
warp[(v, w)] = BoolVar
```

含义：op `v` 是否占用 raw logical warp `w`。

使用方式：

- 单 warp op：`ExactlyOne(warp[v, 0..W-1])`。
- 多 warp op：由连续 warp block 的起点变量推导出多个 `warp[v,w] = 1`。
- variable latency/TMA warpgroup 隔离直接约束 `warp`。
- liveness 中，RMEM value 归属到 producer 所在 warp。
- 返回结果 `warp_assign[node.name]` 取第一个为 true 的 warp。

注意：返回的 `warp_assign` 是 raw logical warp id。后续 FineGrainedWS annotation 会再压缩为 consumer warpgroup id，不能逐字和 `tl_finegrainedws_warp_assigns` 比较。

### 2.3 多 warp 起点变量 `warp_start_v={v}_w={s}`

```text
warp_start[v, s] = BoolVar
ExactlyOne(start slots)
```

只对 `warp_count > 1` 且非 replicable 的 op 创建。

含义：多 warp op `v` 的连续 warp block 从 raw warp `s` 开始。

使用方式：

- 可选起点 `s` 必须满足 `s % warp_align == 0`。
- 每个 `warp[v,w]` 由所有覆盖它的 start slot 做 `MaxEquality` 得到。
- WGMMA 目前 `warp_count=4, warp_align=4`，因此被分配到一个完整 raw warpgroup。

### 2.4 `issue_warp[(v,w)]`

`issue_warp` 是 Python 字典，不是额外独立的 CP-SAT 决策本体：

- 单 warp / replicable op：`issue_warp[(v,w)] = warp[(v,w)]`。
- 多 warp op：`issue_warp[(v,w)]` 只在连续 block 起点为 true 时为 true。

当前后续约束主要仍使用 `warp[(v,w)]`。

### 2.5 optional interval 变量

#### `op_iv_v={v}_w={w}`

```text
end = Tv[v] + latency[v]
OptionalInterval(start=Tv[v], size=latency[v], end=end, presence=warp[v,w])
```

含义：op `v` 在 warp `w` 上执行的可选时间区间。

使用方式：

- blocking sync 的阻塞窗口通过 `NoOverlap(block_interval, op_interval)` 排斥其它 op。
- spill concurrency 通过 `NoOverlap(spill_interval, op_interval)` 排斥接收方 warp 上其它 op。

#### `block_iv_u={u}_v={v}_w={w}`

只在 edge `u -> v` 标记 `blocking_sync` 时创建。

含义：`v` 启动前 `[Tv[v] - delay, Tv[v])` 的阻塞窗口。

当前 `_solve_smt_joint_optimize()` 对普通 sync/barrier 边没有批量设置 `blocking_sync=True`，因为该约束比 modulo issue slot 强，历史上容易让实际图 presolve UNSAT。

#### `spill_iv_u={u}_v={v}_ws={w_src}_wd={w_dst}`

只在 producer output 有 `spill_cost > 0`，且 `w_src != w_dst` 时创建。

含义：跨 warp 使用 producer 输出时，在 consumer warp 上插入 `[Tv[v] - spill_cost, Tv[v])` 的 spill 区间。

### 2.6 modulo 相位变量 `phase[v]`

```text
phase[v] = IntVar(0, ii - 1)
AddModuloEquality(phase[v], Tv[v], ii)
```

含义：op `v` 的 modulo issue 相位。

使用方式：

- FU interval mode 通过两个 op 的相位差 `delta_uv` 保证 issue interval 不重叠。
- exact fallback mode 通过 `phase_eq_v={v}_p={p}` 统计每个 modulo 槽位的资源用量。
- barrier issue slot 也使用相位差做互斥。

### 2.7 `phase_eq_v={v}_p={p}`

```text
phase_eq[v,p] = BoolVar
phase[v] == p <=> phase_eq[v,p]
```

含义：op `v` 是否落在 modulo 相位 `p`。

使用方式：在多容量资源 fallback 里统计：

```text
sum(resource_usage_at_phase_q) <= cap
```

### 2.8 FU / barrier 相位差变量

```text
fu_delta_{resource}_u={u}_v={v} = IntVar(0, ii - 1)
barrier_delta_u={u}_v={v} = IntVar(0, ii - 1)
```

含义：两个 issue interval 在 modulo 环上的相对距离。

使用方式：

```text
delta = (phase[v] + off_v - phase[u] - off_u + ii) mod ii
delta >= dur_u
delta <= ii - dur_v
```

这表示 `u` 和 `v` 的 modulo issue 区间不重叠。

### 2.9 liveness reification 变量

SMEM 和 RMEM liveness 都会创建一组 BoolVar，把“某个 value copy 在 tau 时刻是否 live”变成线性容量约束。

常见变量：

- `smem_p_x={xi}_k={iter_offset}_t={tau}`：SMEM copy 是否已 produced。
- `smem_c_x={xi}_k={iter_offset}_t={tau}`：所有 consumer 是否已完成。
- `smem_l_x={xi}_k={iter_offset}_t={tau}`：SMEM copy 是否 live。
- `smem_live_x={xi}_t={tau}`：某个 output 在 tau 是否任一 copy live。
- `smem_live_buf={buffer_key}_t={tau}`：某个 SMEM buffer 在 tau 是否 live。
- `rmem_p_x=...`：RMEM copy 是否已 produced。
- `rmem_c_x=...`：某个 consumer 是否已消费。
- `rmem_done_x=...`：所有 consumers 是否已消费。
- `rmem_l_x=...`：RMEM copy 是否 live。
- `rmem_live_w={w}_x=...`：该 RMEM live copy 是否归属于 warp `w`。
- `rmem_live_buf_w={w}_b={buffer}_k={iter_offset}_t={tau}`：按 `(warp, buffer, iter_offset)` 聚合后的 live copy。

使用方式：

- SMEM：每个 `tau` 上统计 live buffer footprint，加 static smem 后不得超过 `smem_limit`。
- RMEM：在稀疏 checkpoint 上统计每个 warp 的 live bytes，不得超过 `reg_limit`。

注意：当前 `_solve_smt_joint_optimize()` 调用 `_run_joint_solver(... enable_liveness=False)`，也就是 CP-SAT 内部 RMEM/SMEM liveness 约束默认不打开；它会在求解后用 Python `_check_fixed_liveness()` 做固定调度的完整检查。文档仍列出 CP-SAT liveness 变量，因为 `HeddleScheduler._solve_phase_b()` 支持它们。

### 2.10 objective 变量

```text
end_T_v[v] = Tv[v] + issue_duration[v]
max_end_T = max(end_T_v)
Minimize(L * N * max_end_T + sum(Tv))
```

含义：在固定 `ii` 和候选 `L` 下，偏好更短的实际 makespan，其次让所有 op 尽早启动。

## 3. CP-SAT 约束分类

## 3.1 warp 分配约束

### 单 warp / replicable op

```text
ExactlyOne(warp[v,w] for w in W)
```

含义：普通 op 和 replicable true TMA 在 raw warp 域中选一个 warp。

### 多 warp op 连续 block

```text
ExactlyOne(warp_start[v,s])
warp[v,w] = max(starts covering w)
```

含义：WGMMA 这类 `warp_count=4` 的 op 必须占据连续 4 个 warp，并且起点按 `warp_align=4` 对齐。这样天然对应一个 Hopper warpgroup。

### early return

如果 `warp_count > W` 或没有合法 aligned start slot，直接返回 `None`，不构造无意义模型。

## 3.2 variable latency / TMA warpgroup 约束

### variable latency ops 同 warpgroup

对任意两个 variable latency op `u, v`：

```text
if wu // 4 != wv // 4:
    not (warp[u,wu] and warp[v,wv])
```

含义：所有 true TMA/variable latency op 放到同一个 raw warpgroup。

### variable latency 和非 variable latency 分离

对 variable latency op `u` 和普通 op `v`：

```text
if wu // 4 == wv // 4:
    not (warp[u,wu] and warp[v,wv])
```

含义：TMA producer warpgroup 和 consumer/non-TMA warpgroup 隔离。这个约束会消耗一个 raw warpgroup，所以 `_solve_smt_joint_optimize()` 给 solver 的 `num_warps` 是 consumer warps 加 4。

## 3.3 wait barrier 与 shared-memory reader 同 warpgroup约束

入口侧先扫描：

- 找到 producer/TMA 写过的 shared buffer。
- 遇到 consumer wait barrier 后，把后续读取 producer-written shared buffer 的 consumer stmt 记为 reader。
- 形成 `(wait_idx, reader_idx)`。

传入 scheduler 后，在 CP-SAT 中约束：

```text
if wu // 4 != wv // 4:
    not (warp[wait,wu] and warp[reader,wv])
```

含义：保护 shared-memory 值的 wait 必须和真正使用该 shared 值的 reader 在同一个 raw warpgroup。否则后续 PCWS/FineGrainedWS 可能把 wait 放在一个 WG，把 shared load/use 放在另一个 WG。

## 3.4 reg feedback 的 not-all-same-warpgroup约束

如果求解后的 Python fixed liveness 检查发现某些 producer 造成 `reg_limit` overflow，入口侧可选出 peak producer 集合，第二轮加约束：

```text
for each raw warpgroup wg:
    not all(producer in wg for producer in feedback_set)
```

CP-SAT 表达方式：

- `feedback_in_wg_set=...` 表示某 producer 是否落在某个 raw WG。
- `BoolOr([term.negated() for term in in_group_terms])` 表示这组 producer 不能全部在同一个 WG。

用途：轻量地把高寄存器压力 producer 分散开，而不在 CP-SAT 中重建完整 RMEM liveness。

## 3.5 start hints

```text
model.add_hint(Tv[v], hint_t)
```

含义：给 CP-SAT 一个来自 naive `M` / expanded `M` 的启动时间提示。

注意：hint 不是硬约束。它只帮助搜索，求解结果可以偏离 hint。

## 3.6 依赖约束

对每条 parent `u -> v`：

```text
Tv[v] - Tv[u] >= base_delay - distance * ii
```

含义：

- `base_delay` 来自 edge 显式 delay，否则用 producer latency。
- `distance=0` 是同一轮迭代内依赖。
- `distance=1` 是跨迭代 hazard，例如自读写同一 buffer 或 wait barrier 的相邻迭代间隔。
- `distance * ii` 体现 modulo schedule 中跨迭代的时间平移。

入口侧当前添加的依赖包括：

- `op_deps` 中的同迭代依赖。
- WGMMA/TMA issue slices 的内部顺序依赖。
- stmt 同时读写同一 buffer 的 loop-carried self dependency。
- wait barrier 的 loop-carried self dependency。

## 3.7 spill delay 约束

如果 producer output 有 `spill_cost > 0`：

先构造：

```text
same_w = OR_w( warp[u,w] AND warp[v,w] )
```

然后：

```text
same_w     => Tv[v] - Tv[u] >= base_delay - distance * ii
not same_w => Tv[v] - Tv[u] >= base_delay + spill_cost - distance * ii
```

如果 `disallow_spills=True`，则强制 `same_w == 1`。

含义：跨 warp 使用 producer value 时，consumer 需要额外等待 spill/move 代价。

## 3.8 blocking sync 并发约束

如果 edge 标记 `blocking_sync=True`：

1. 约束 producer warp 覆盖到 consumer warp：

```text
warp[u,w] => warp[v,w]
```

2. 创建 `[Tv[v] - delay, Tv[v])` 的 block interval。
3. 对同一 warp 上其它 op 加：

```text
NoOverlap(block_interval, op_interval(other,w))
```

含义：同步阻塞窗口内，同一个 warp 上不能安排其它重叠 op。

当前 joint transform 对普通 sync/barrier 没有默认启用这类强 blocking edge，而是主要用 barrier issue slot 和显式依赖来建模。

## 3.9 spill concurrency 约束

当 `use_spill_concurrency=True` 且 `spill_cost > 0`：

如果 producer 在 `w_src`，consumer 在 `w_dst`，且 `w_src != w_dst`，创建 spill interval：

```text
spill_present = warp[u,w_src] AND warp[v,w_dst]
spill interval = [Tv[v] - spill_cost, Tv[v])
NoOverlap(spill_interval, op_interval(other,w_dst))
```

含义：跨 warp spill 占用接收方 warp 的一段时间，这段时间内接收方 warp 不能执行其它 op。

## 3.10 FU capacity约束

FU 资源来自每个 node 的 `reservation`。入口侧目前的 capacity：

```text
TMA: 255
TC: 1
ALU: 64
SFU: 16
BARRIER: 1
```

### cap=1 且连续 reservation：interval mode

对 `TC`、`BARRIER` 等 cap=1 且每个 op 使用连续 offset 的资源，使用 modulo interval 不重叠：

```text
delta = (phase[v] + off_v - phase[u] - off_u + ii) mod ii
delta >= dur_u
delta <= ii - dur_v
```

含义：两个占用同一 cap=1 资源的 issue interval 在 modulo 环上不能重叠。

如果某个 span 大于 `ii`，或两个 span 的长度和大于 `ii`，当前实现会 early return `None`。

### 其它资源：exact fallback

对非 interval mode 的资源，按每个 modulo 相位 `q` 统计：

```text
sum(c * phase_eq[v, q-l]) <= cap
```

含义：在每个 modulo slot 上，该 FU 的总 issue 使用量不能超过 capacity。

如果 `cap > 1` 且 `ii > 256`，当前实现会跳过 exact multi-cap FU 约束以避免模型过大，并打印 skip 日志。

## 3.11 barrier issue slot互斥

wait/try_wait barrier 被建模成专门的 `BARRIER` issue slot：

```text
reservation = [{BARRIER: 1}]
```

除了参与 `BARRIER` 自身 cap=1 的 interval mode 外，还额外和所有 op 的 full issue span 做互斥：

```text
barrier_delta = (phase[other] + off_other - phase[barrier] - off_barrier + ii) mod ii
barrier_delta >= barrier_dur
barrier_delta <= ii - other_dur
```

含义：barrier issue 不能和任何 op 的 issue interval 共享同一个 modulo slot。

重要区别：barrier 不再被当成同时占满 TMA/TC/ALU/SFU，而是一个独立同步 issue 槽。

## 3.12 SMEM 容量约束

当 `enable_liveness=True` 时，CP-SAT 内部会对每个 `tau in [0, L)` 建 SMEM live 变量。

生产条件：

```text
is_produced <=> Tv[pv] + iter_offset * ii <= tau
```

释放条件：

- 对 `DEAD_ON_ENTRY` 且有 consumers 的值，所有 consumer 都已启动/消费后释放。
- 否则 produced 后视为 live。

buffer 聚合：

```text
buffer_live = OR(live copies of same buffer)
sum(static_smem + footprint(buffer) * buffer_live) <= smem_limit
```

含义：同一物理 SMEM buffer 在同一时刻只计一次 footprint。入口侧也会把已知 shared allocations 作为 `static_smem_terms` 加入容量约束。

当前 `_solve_smt_joint_optimize()` 默认 `enable_liveness=False` 调用 joint solver，所以这部分通常不进入当前主路径的 CP-SAT 模型，而由后验 Python check 接管。

## 3.13 RMEM / register 容量约束

当 `enable_liveness=True` 且有 outputs 且 `reg_limit > 0` 时启用。

为了降低模型规模，不对所有 `tau` 建完整 interval，而是构造 sparse checkpoints：

- `range(0, L, liveness_checkpoint_step)`
- `L - 1`
- start hints 附近 `t-1, t, t+1`
- consumer hints 附近 `t-1, t, t+1`

对每个 checkpoint `tau`：

1. 判断 RMEM copy 是否 produced。
2. 判断所有 consumers 是否已经消费。
3. 得到 `live_var`。
4. 和 producer 的 `warp[pv,w]` 做 AND，得到 `live_on_warp`。
5. 按 `(warp, buffer, iter_offset)` 聚合，同一物理 buffer copy 只计一次。
6. 加容量约束：

```text
sum(live_bytes_on_warp[w]) <= reg_limit
```

含义：每个 raw warp 上 live RMEM bytes 不得超过寄存器预算。

同 SMEM 一样，当前主调用默认关闭 CP-SAT 内部 liveness，随后用 `_check_fixed_liveness()` 对求解出的固定 schedule/warp assignment 做完整逐 tau 检查。

## 3.14 incoming_live / iter_offsets

CP-SAT liveness 使用：

```text
max_iter_overlap = (L - 1) // ii
iter_offsets = range(-max_iter_overlap, max_iter_overlap + 1)
```

含义：考虑当前 window 内可能同时活跃的前后迭代 copy。若 `include_incoming_live=False`，只考虑当前迭代 `0`。

当前 `HeddleScheduler` 默认 `include_incoming_live=True`。

## 3.15 objective约束

当 `optimize=True`：

```text
end_v = Tv[v] + max(len(reservation[v]), 1)
mx = max(end_v)
minimize(L * N * mx + sum(Tv))
```

含义：

1. 第一优先级：让实际结束时间 `mx` 尽量小。
2. 第二优先级：在相同 `mx` 下，让所有 op 尽早启动。

因为外层已经固定候选 `L`，这里不是直接最小化 `L`，而是在这个窗口中找更紧凑的排布。

## 4. 后验 fixed liveness 检查

当前 `_solve_smt_joint_optimize()` 的主路径：

```text
_run_joint_solver(enable_liveness=False)
_post_check_liveliness(sol)
```

也就是说，CP-SAT 负责：

- 时间 `Tv`
- warp assignment
- dependency
- FU capacity
- barrier slot
- TMA/consumer warpgroup 隔离
- wait/shared-reader 同 WG

然后 Python `_check_fixed_liveness()` 对固定结果做：

- FU 使用复查。
- subcore 使用复查。
- RMEM live bytes 逐 `tau` 检查。
- SMEM live bytes 逐 `tau` 检查。

如果后验检查失败：

- `reg_limit` 失败时，可能根据 top live buffers 找 peak producers，第二轮加入 `not_all_same_warpgroup` 反馈约束。
- `subcore_issue_limit` 失败时，代码准备了 `same_subcore_exclusion_pairs` 反馈入口，但当前 `smt.py` 内没有实际消费该参数建约束，且 subcore CP-SAT 约束被注释为 disabled。
- 如果候选 window 失败，外层尝试 `window, window + I, window + 2I, window + 3I`。

## 5. 求解结果字段

`HeddleScheduler._solve_phase_b()` 返回：

- `ii`：固定 II。
- `window`：当前求解窗口。
- `schedule`：`{node.name: Tv}`。
- `warp_assign`：`{node.name: first selected raw warp}`。
- `reg_peak`：CP-SAT 内部 liveness 打开时的 sparse checkpoint peak；当前主路径通常为空。
- `variable_lifetimes`：后处理出来的 buffer lifetime 信息，不是 CP-SAT 决策变量。

`_solve_smt_joint_optimize()` 再整理成：

- `I`：`base_I`。
- `L`：按 collapsed schedule 和 issue duration 计算出的 optimized length。
- `M`：从 child schedule collapse 回原 stmt id。
- `status`：当前代码里 `solved_with_optimize` 没有被置 true，因此成功时通常是 `SMT_FEASIBLE`，虽然调用 CP-SAT 时使用了 `optimize=True`。
- `warp_assign`：从 child raw warp collapse 回原 stmt id。
- `reg_peak` / `smem_peak`：来自后验 liveness。
- `variable_lifetimes`：buffer lifetime 汇总。
- `modular_rrt`：按 `I` 折叠后的资源占用表。
- `ordering`：按 optimized `M` 排序后的 stmt id 顺序。

## 6. 当前实现中的几个重要注意点

1. `SMT joint` 名称是历史遗留，当前 joint backend 是 CP-SAT。

2. `I` 是 hard input，不在 joint 阶段搜索。joint 只在固定 `I` 和候选 `L` 上优化 `Tv` 与 warp assignment。

3. `start_hints` 不是约束。naive `M` 只给 CP-SAT 搜索提示，结果可以改变。

4. TMA/variable latency 会占一个独立 raw warpgroup，并且和非 TMA op 分离。

5. WGMMA 通过 `warp_count=4, warp_align=4` 建模成完整 raw warpgroup。

6. wait barrier 不再占满所有 FU，而是独立 `BARRIER` issue slot。

7. wait barrier 和它保护的 shared-memory reader 会被硬约束在同一个 raw warpgroup。

8. CP-SAT 内部 liveness 代码仍存在，但当前 transform 主路径默认关闭，改用 Python fixed liveness 检查。这意味着看到 `INFEASIBLE` 时要先区分是 CP-SAT 结构约束失败，还是求解后 liveness check 失败。

9. raw solver `warp_assign` 和最终 `tl_finegrainedws_warp_assigns` 单位不同。前者是 raw logical warp id，后者通常是压缩后的 consumer warpgroup id。

10. `same_subcore_exclusion_pairs` 已从 transform 传入 scheduler 构造函数，但当前 `_solve_phase_b()` 没有使用它生成 CP-SAT 约束；subcore issue 精确模型也明确处于 disabled 状态。
