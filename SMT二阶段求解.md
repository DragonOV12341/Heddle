# SMT 二阶段求解流程

本文档总结当前 `heddle_consumer_schedule.py` 中，在 naive modulo schedule 之后继续进入 joint optimize 的整体执行流程。这里的“二阶段”指：

1. 先用 `_solve_naive_modulo_sched()` 求一个基础模调度计划。
2. 再用 `_solve_smt_joint_optimize()` 把这个计划交给 `HeddleScheduler.schedule_joint()`，联合优化启动时间、warp 分配、资源占用和 liveness。

需要注意：当前实现里两阶段都使用 OR-Tools CP-SAT，而不是旧版 Z3。

## 入口位置

入口在 `Heddle/heddle/transform/heddle_consumer_schedule.py` 的 loop-body 重排逻辑中：

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

进入这条路径前，代码已经从当前 TIR loop body 中构造出：

- `infos_list`：每条 statement 的读写 buffer、op 类型、TMA/WGMMA/barrier/sync 标记等信息。
- `all_indices`：参与当前模调度的所有 op id。
- `deps_all`：更保守的全量依赖图，包括普通数据依赖、barrier/sync 顺序等。
- `func_num_threads`：用于 joint 阶段估算可用 warp 数。

## 第一阶段：naive modulo schedule

`_solve_naive_modulo_sched(op_deps, infos, op_indices)` 的目标是先找到一个可行的 modulo issue plan。它不解 warp 分配，也不建完整寄存器/SMEM 活跃区间，只负责给后续 joint refine 一个基础形状。

### 1. 建模对象

输入 op 被保存为 `ops = list(op_indices)`。每个 op 有三个关键量：

- `duration[idx]`：issue slot 占用时长，不是数据就绪 latency。
- `latencies[idx]`：依赖边使用的数据就绪延迟。
- `rrt[idx]`：每个 issue cycle 消耗哪些资源。

当前资源容量大致为：

```python
capacity = {
    "TMA": 255,
    "TC": 1,
    "ALU": 64,
    "SFU": 16,
    "BARRIER": 1,
}
```

特殊处理包括：

- `is_wait_barrier` 使用独立 `BARRIER` issue slot，latency 设为 1。
- WGMMA 的 `duration` 来自 `_detect_wgmma_issue_cycles()`，资源为连续多个 `TC` slot。
- 普通 op 的 `duration` 来自 `_detect_op_issue_cycles()`，latency/resource 来自 `_detect_op_latency_and_resource()`。
- 跨迭代 self hazard 使用 issue delay，而不是完整 ready latency，避免把约束强化成 `I >= latency[v]`。

### 2. 对固定 II 求解

内部 `solve_for_I(I)` 会建立 CP-SAT 模型：

- 决策变量 `M[v]`：op `v` 在展开窗口中的启动 issue 时间。
- `phase[v] = M[v] mod I`：op 在 modulo 周期里的相位。
- 依赖约束：

```text
M[v] - M[u] + delta * I >= d
```

其中同一迭代边 `delta = 0`，跨迭代 self hazard 边 `delta = 1`。

资源约束主要按 modulo phase 建：

- 对 `cap = 1` 且连续 reservation 的资源，用环形 interval no-overlap 表达。
- barrier op 额外和所有其它 op 的 issue interval 互斥，保证 wait/try_wait 单独占一个同步 issue slot。

调度长度：

```text
L = max(M[v] + duration[v])
```

这里的 `L` 表示当前所有 op 发射完成所需窗口，不是最后一条指令执行完成的 ready time。

目标函数是：

```text
minimize 100 * L - sum(M[v])
```

也就是优先压短 issue 窗口，同时在同样窗口内倾向把 op 往后放一点。

### 3. 搜索最小 II

`solve_min_I()` 先用 `estimated_total_latency` 作为上界，尝试收缩 `lb/ub`，然后在线性区间内找第一个可行 `I`。返回值是一个 plan 列表，当前通常只取第一个可行 plan。

每个 naive plan 形如：

```python
{
    "I": I,
    "L": L,
    "M": {op_idx: start_time},
    "modular_rrt": [...]
}
```

其中 `modular_rrt` 只是按 `I` 折叠后的资源占用表，用于打印和观察。

## 第二阶段：joint optimize

`_solve_smt_joint_optimize()` 接收 naive plan，并把它转换成 `HeddleScheduler` 的 `OpNode` 图。它的核心作用是：固定 naive 得到的 `base_I`，在一个可控窗口内重新求更完整的联合排布。

### 1. 读取 naive plan

函数先提取：

- `base_M = mod_sched_plan["M"]`
- `base_I = mod_sched_plan["I"]`
- `base_L = mod_sched_plan["L"]`

如果 `base_I <= 0`，直接返回原 plan；如果 plan 为空，则返回 `None`。

joint 阶段不会重新搜索 II，而是调用：

```python
solver.schedule_joint(
    min_ii=base_I,
    max_ii=base_I,
    window=solve_window,
    optimize=optimize,
)
```

所以这一阶段是在固定 II 下 refine `M`、warp 和资源/liveness。

### 2. 从 `_StmtInfo` 构造 `OpNode`

每个 `info` 会被转换为一个 `OpNode`：

- `name = f"s{idx}"`
- `resource_type` 来自 TMA/barrier 特判或 `_detect_op_latency_and_resource()`。
- `latency` 来自 `_latency_for_info()`。
- `reservation` 来自 `_reservation_for_info()`，表示 issue slot 占用。
- `outputs` 用 `OutputValue` 记录写出的 RMEM/SMEM footprint，用于 liveness 容量约束。
- WGMMA 设置 `warp_count = 4`、`warp_align = 4`。
- true TMA 设置 `replicable = True`，并标记 variable latency。

这里依然明确区分：

- `latency`：依赖 readiness。
- `reservation`：issue slot occupancy。
- `OutputValue`：后续寄存器/SMEM 活跃区间容量统计。

### 3. 依赖边转换

同一迭代依赖来自 `op_deps/deps_all`：

```python
node_by_idx[v].add_dependency(
    node_by_idx[u],
    distance=0,
    delay=_dependency_delay_for_info(infos[u]),
)
```

跨迭代 hazard 只对真实需要的情况添加：

- op 同时读写同一个 buffer。
- wait barrier 这类同步语义。

这类边使用：

```python
distance = 1
delay = _issue_delay_for_self_edge(info)
```

不要把“所有 op 在不同迭代保持同相位”建成 `v -> v, distance=1, delay=latency`。当前 joint 模型里每个 op 只有一个 `Tv[v]`，且 `ii = base_I` 固定，已经隐含表达了相位一致性；额外加全量 self-edge 会把模型过度收紧。

### 4. 资源和求解参数

joint 阶段会设置：

- `num_warps = ceil(kernel_num_threads / 32) * 2`，因为考虑 PCWS 后乘以 2。
- `reg_limit` 默认来自 plan，缺省为 `32 * 240 * 4` bytes。
- `smem_limit` 默认约为 H100 CTA shared memory 上限。
- 如果简单求和的 SMEM footprint 已经超过 `smem_limit`，会把 limit 提高到该 floor，避免 trivially UNSAT。
- `start_hints` 使用 naive 的 `base_M`，传给 CP-SAT hint。

基础窗口：

```text
window = max(base_L, max(base_M) + 1, base_I)
```

后续会尝试：

```text
window, window + base_I, window + 2 * base_I
```

同时也会尝试放宽 `reg_limit` / `smem_limit`，用于判断是否是容量约束导致不可行。

## `HeddleScheduler.schedule_joint()` / `_solve_phase_b()`

`_run_joint_solver()` 内部实例化 `HeddleScheduler`，然后进入 scheduler 侧 Phase B：

```python
HeddleScheduler(...).schedule_joint(...)
```

`schedule_joint()` 会枚举给定 II 和窗口 `L`，对每个组合调用 `_solve_phase_b(ii, L, optimize)`。在当前 joint optimize 路径里，II 被固定为 `base_I`，窗口由 transform 侧提供。

### Phase B 的主要变量

`_solve_phase_b()` 里的核心 CP-SAT 变量包括：

- `Tv[v]`：op `v` 在窗口 `[0, L)` 中的启动时间。
- `warp[(v, w)]`：op `v` 是否分配到 warp `w`。
- `issue_warp[(v, w)]`：op 的 issue 起点 warp，用于多 warp op 的连续块建模。
- `phase[v] = Tv[v] mod ii`：用于 modulo FU/issue 资源约束。
- `live` / `incoming_live` / `iter_live`：在 optimize+liveness 模式下统计 RMEM/SMEM 活跃区间。

### Phase B 的主要约束

1. warp 分配
   - 单 warp 或 replicable op：exactly one warp。
   - 多 warp op：选择一个连续 warp 块，且满足 `warp_align`。
   - variable latency op 约束到同一个 warpgroup，并与非 variable latency op 分离 warpgroup。

2. 依赖约束
   - 普通边：

   ```text
   Tv[v] - Tv[u] >= delay - distance * ii
   ```

   - 如果 producer/consumer 跨 warp 且有 spill cost，会在 delay 上加 spill cost。
   - blocking sync 边会额外加入同 warp 覆盖和阻塞窗口 no-overlap。

3. issue/FU 资源约束
   - 同 subcore 约束：`warpId % 4` 相同的 op，不能在同一个 modulo issue interval 重叠。
   - `cap = 1` 且连续 reservation 的 FU，用 modulo interval no-overlap。
   - barrier issue slot 和所有其它 op 的 issue interval 互斥。
   - 其它资源形状使用逐 modulo phase 容量求和。

4. liveness 和容量约束
   - optimize 且 `enable_liveness=True` 时，建 RMEM/SMEM live 变量。
   - RMEM 按 warp 统计，约束不超过 `reg_limit`。
   - SMEM 按全局统计，且同一个 allocation 不按迭代副本重复累加。
   - incoming/iter copy 用来表达 steady-state modulo schedule 中多个迭代副本同时 live 的情况。

5. 优化目标
   - `optimize=True` 时，目标偏好更紧凑的 `max_end` 和更小的 `sum(Tv)`。

返回值形如：

```python
{
    "ii": ii,
    "window": L,
    "schedule": {"s0": 0, ...},
    "warp_assign": {"s0": 4, ...},
    "reg_peak": {...},
}
```

## 对照 Twill 4.1 的约束

Notion 页面 `twill 论文阅读`（<https://magnificent-fine-259.notion.site/twill-38017570dadc803b97b9d26336a39df6>）的 `4.1 带约束的模调度` 部分，把 Twill 的 constrained modulo scheduling 写成一个基于三维布尔变量的模型：

```text
op[v, i, t] = 操作 v 的第 i 个迭代副本是否在时间 t 被调度
```

它列出的核心模调度约束是：

- `UNIQUENESS`
- `CONSISTENCY`
- `COMPLETION`
- `DEPENDENCE`
- `CAPACITY`

当前代码没有逐字实现 `op[v,i,t]` 这种三维 one-hot 布尔网格，而是用更紧凑的 IntVar/phase/interval 形式表达同一类约束。这样做主要是为了避免 `op * iteration * time` 级别的变量爆炸。

### UNIQUENESS：每个 op 恰好调度一次

Twill 公式的含义是：对每个 `v, i`，所有时间 `t` 上只能且必须有一个 `op[v,i,t] = true`。

当前实现：

- naive 阶段用一个整数变量 `M[v]` 表示 op 的启动时间。
- joint 阶段用一个整数变量 `Tv[v]` 表示 op 的启动时间。
- 因为每个 op 只有一个 `M[v]` / `Tv[v]`，所以“恰好一次”是变量形态天然保证的，不需要再加 `sum_t op[v,i,t] == 1`。

代码位置：

- `_solve_naive_modulo_sched.solve_for_I()`：`M[v] = model.NewIntVar(...)`
- `HeddleScheduler._solve_phase_b()`：`Tv = [model.new_int_var(...)]`

差异：

- Twill 显式枚举每个迭代副本 `i`。
- 当前实现只建一个 steady-state 模板时间，迭代副本由固定 II 隐含展开。

### CONSISTENCY：不同迭代副本相隔固定 II

Twill 公式的含义是：如果第 0 个迭代副本的 `v` 在 `t` 发射，那么第 `i` 个副本必须在 `t + i * I` 发射。

当前实现：

- naive 阶段只求一个 `M[v]`，并通过 `phase[v] = M[v] mod I` 表示 modulo 相位。
- joint 阶段只求一个 `Tv[v]`，并固定 `ii = base_I`。
- 因为每个 op 只有一个 steady-state 启动时间，所以后续迭代默认按 `+ k * II` 重复。

代码里还有一段注释明确说明：`op[v,0,t] => op[v,i,t+i*II]` 这种相位一致性已经由“每个 op 一个 `Tv[v]` 且固定 `ii=base_I`”隐含表示，不应该把它误建成所有 op 的 loop-carried self dependency。

差异：

- Twill 把 consistency 写成跨副本蕴含。
- 当前实现把它折叠成单模板 + 固定 II；只有真实 loop-carried hazard 才额外加 `distance=1` 依赖边。

### COMPLETION：不能越过求解窗口

Twill 公式的含义是：如果 `t + cycles(v) > T`，则不能在 `t` 启动 `v`。

当前实现分两层：

- naive 阶段比较接近 Twill：`M[v]` 的上界是 `H - duration[v]`，并用 `L = max(M[v] + duration[v])` 计算发射窗口长度。
- joint 阶段的 `window=L` 主要是 `Tv[v]` 的启动时间域：`Tv[v] in [0, L-1]`。op 的结束时间用于目标函数和最终 `optimized_L` 重新计算，但没有完全照搬 `Tv[v] + reservation_len <= window` 作为硬约束。

因此，当前 joint 阶段更像是在给定启动窗口里找一个可行 steady-state 模板，然后用求出的 `Tv + reservation` 回算实际 `optimized_L`。

差异：

- Twill 的 `T` 是直线程序 `Q*` 的硬边界。
- 当前 joint 的 `window` 更偏求解搜索窗口；最终 `L` 由结果重算。

### DEPENDENCE：依赖消费者不能抢跑

Twill 公式的含义是：若边为 `(u, v, d, delta)`，则 `v` 的对应迭代副本不能早于 `u + d`。

当前实现：

naive 阶段写成线性约束：

```text
M[v] - M[u] + delta * I >= d
```

joint 阶段写成：

```text
Tv[v] - Tv[u] >= delay - distance * ii
```

两者是同一个 modulo dependence 约束的等价写法。这里 `distance` 对应 Twill 里的 `delta`，`delay` 对应 `d`。

当前代码还做了两个重要收窄：

- 普通同迭代依赖来自 `deps_all`，`distance=0`。
- 跨迭代 `distance=1` 只给真实 loop-carried hazard，例如自读写同一 buffer、wait barrier 等；不会给所有 op 强行加 self-edge。

差异：

- Twill 的公式基于 `op[v,i,t]` 排除非法时间。
- 当前实现直接在启动时间变量上加差分不等式，更紧凑。

### CAPACITY：任意时刻资源使用不能超过容量

Twill 公式的含义是：对任意时间 `t` 和功能单元 `f`，所有正在执行的 op 的 RRT 使用量总和不能超过 `cap(f)`。

当前实现：

- naive 阶段用 `rrt[v]` 和 `duration[v]` 描述 issue reservation。
- 对 `cap = 1` 且 reservation 连续的资源，naive 用 modulo 环形 interval no-overlap，避免逐时间点枚举。
- barrier 被建成独立 `BARRIER` issue slot，并额外与所有其它 op 的 issue interval 互斥。
- joint 阶段先用 `phase[v] = Tv[v] mod ii`，再建：
  - 同 subcore issue interval 互斥：`warpId % 4` 相同才互斥。
  - `cap = 1` 且连续 reservation 的 FU interval no-overlap。
  - barrier issue slot 与所有其它 op 互斥。
  - 其它资源形状走 `_fold_reservations(ii)` 后的逐 modulo phase 容量求和。

差异：

- Twill 的容量约束按 `op[v,i,t-c] * RRT[v][f,c]` 做显式求和。
- 当前实现把常见连续 reservation 压缩成 interval no-overlap；只有不适合 interval 表达的资源形状才退回逐相位求和。

### 后续 liveness / memory capacity 约束

Notion 页面在 4.1 后面继续解释图 5 的 liveness/memory 约束，包括：

- 任意时间 `t`、memory 类型 `m` 上，所有 live value 的 footprint 不能超过 capacity。
- loop-carried value 在展开片段末尾仍然 live。
- `LIVEPROP` / `DEADPROP` 从后往前传播活跃性。

当前代码没有显式建立 `dead[v,i,t]`，也没有完全按论文的 backward propagation 公式逐项编码，而是用 producer/consumer start-time 直接定义 live interval：

- `live[(xi, tau)]`：同一迭代内第 `xi` 个 output 在 `tau` 是否 live。
- `incoming_live[(xi, tau)]`：loop-carried output 是否从上一轮带入。
- `iter_live[(xi, iter_offset, tau)]`：steady-state 下多个迭代副本重叠时的 live copy。
- RMEM 容量按 warp 统计：`sum(footprint * live_on_warp) <= reg_limit`。
- SMEM 容量按 CTA 全局统计：同一个 allocation 只统计一次，不按迭代副本重复累加。

另外，transform 侧 `_check_fixed_liveness()` 会对 joint 求出的固定 `schedule + warp_assign` 再做一次外部 liveness 复核。也就是说，当前实现的 liveness 语义和 Twill 图 5 同方向，但表达方式更工程化：

- 不用 `op[v,i,t]` / `dead[v,i,t]` 全量布尔传播。
- 用 `Tv`、`_start_le_var()`、consumer 列表和迭代偏移直接推导 live 区间。
- 为控制模型大小，优化解会先关掉内部 liveness，再用固定解复核。

## joint optimize 的求解策略

transform 侧不是直接只跑一次优化模型，而是分几步。每一步都会固定 `base_I`，并在 `candidate_windows`、`reg_limit_candidates`、`smem_limit_candidates` 的组合上尝试求解。

### 阶段 0：构造 joint 模型输入

进入求解循环前，`_solve_smt_joint_optimize()` 会先把 naive plan 和 `_StmtInfo` 转成 scheduler 侧的 `OpNode` 图。这个阶段不调用 CP-SAT，但决定后面会有哪些约束。

启用的语义包括：

- 固定 `base_I`：joint 阶段只 refine 同一个 II，不重新搜索 II。
- `start_hints`：把 naive 的 `base_M` 作为 CP-SAT hint。
- `resource_type` / `reservation`：为每个 op 建 issue reservation。
- `latency` / edge delay：为依赖边准备 readiness delay。
- `outputs`：为 RMEM/SMEM liveness 和容量检查准备 footprint。
- WGMMA warp 语义：`warp_count=4`，`warp_align=4`。
- true TMA 语义：`replicable=True`，并标记 variable latency。
- 同迭代依赖：来自 `deps_all`，`distance=0`。
- 跨迭代依赖：只给真实 loop-carried hazard，例如自读写同一 buffer、wait barrier，`distance=1`。

这里不会启用的语义：

- 不会给所有 op 加 `v -> v, distance=1` 的 consistency self-edge。
- 不会在 transform 侧直接生成 `op[v,i,t]` 三维布尔变量。

### 阶段 1：feasibility 求一个可行联合排布

第一次调用：

```python
_run_joint_solver(
    solve_window=solve_window,
    optimize=False,
    solve_reg_limit=solve_reg_limit,
    solve_smem_limit=solve_smem_limit,
)
```

默认 `enable_liveness=True` 会被传进 `HeddleScheduler`，但 `_solve_phase_b()` 内部实际用：

```python
track_liveness = self.enable_liveness and optimize and all_outputs and self.reg_limit > 0
```

所以在 `optimize=False` 的 feasibility 阶段，内部 `live/incoming_live/iter_live` 网格不会展开，RMEM/SMEM 容量约束也不会进入 CP-SAT 主模型。

此阶段启用的约束：

- 唯一启动时间：每个 op 一个 `Tv[v]`。
- 固定窗口：`Tv[v] in [0, L - 1]`。
- warp 分配：
  - 单 warp / replicable op：exactly one warp。
  - 多 warp op：选择一个连续 warp block。
  - `warp_align` 对齐约束。
- variable latency warpgroup 约束：
  - variable latency op 之间必须在同一个 warpgroup。
  - variable latency op 和 non-variable latency op 不能在同一个 warpgroup。
- start hint：给 `Tv[v]` 加 naive `base_M` hint。
- 依赖约束：
  - `Tv[v] - Tv[u] >= delay - distance * ii`。
  - 如果有 spill cost，跨 warp 时加 spill delay。
  - 如果 `disallow_spills=True`，强制相关边同 warp。
- blocking sync 约束：
  - parent warp 覆盖到 child warp。
  - 阻塞窗口和同 warp 上其它 op 做 `NoOverlap`。
- spill 并发约束：
  - 跨 warp spill window 和接收 warp 上其它 op 做 `NoOverlap`。
- modulo phase：
  - `phase[v] = Tv[v] mod ii`。
- subcore issue 约束：
  - `warpId % 4` 相同的 issue interval 不重叠。
- FU / issue reservation 容量：
  - `cap=1` 且连续 reservation 的资源用 modulo interval no-overlap。
  - 其它资源用 `_fold_reservations(ii)` 后逐 modulo phase 求和。
- barrier issue slot：
  - barrier reservation 和所有其它 op 的 issue interval 互斥。

此阶段不启用的约束/目标：

- 不展开 `live`、`incoming_live`、`iter_live`。
- 不在 CP-SAT 主模型内约束 `reg_limit` / `smem_limit`。
- 不设置优化目标，只要求可行解。

阶段结果：

- 找到的第一个解保存为 `feasible_sol`。
- 记录对应的 `feasible_window`、`used_reg_limit`、`used_smem_limit`。
- 如果所有候选组合都失败，进入阶段 4 的 retry optimize。

### 阶段 2：在同一窗口上求优化解

如果阶段 1 成功，会在 `feasible_window` 和同一组 limit 上再跑一次：

```python
_run_joint_solver(
    solve_window=feasible_window,
    optimize=True,
    solve_reg_limit=used_reg_limit,
    solve_smem_limit=used_smem_limit,
    enable_liveness=False,
)
```

因为显式传了 `enable_liveness=False`，所以即使 `optimize=True`，内部 liveness 网格也不会展开。

此阶段启用的约束：

- 阶段 1 中列出的所有结构性约束：
  - warp 分配。
  - variable latency warpgroup。
  - 依赖 / spill delay / blocking sync。
  - subcore issue。
  - FU reservation。
  - barrier issue slot。
以及
  - 约束SMEM 容量
- 优化目标：

```text
minimize L * N * max_end + sum(Tv)
```

其中 `max_end = max(Tv[v] + len(reservation[v]))`。

此阶段不启用的约束：

- 不展开内部 `live/incoming_live/iter_live`。
- 不在 CP-SAT 主模型内约束 RMEM 容量。

阶段结果：

- 如果 `opt_sol` 不存在，直接使用阶段 1 的 `feasible_sol`。
- 如果 `opt_sol` 存在，进入阶段 3 做固定 liveness 复核。

### 阶段 3：对优化解做固定 liveness 复核

优化解出来后，transform 侧不会直接信任它，而是调用：

```python
_check_fixed_liveness(
    opt_schedule,
    opt_warp_assign,
    check_L=opt_L,
    check_reg_limit=used_reg_limit,
    check_smem_limit=used_smem_limit,
)
```

这一步不是 CP-SAT 求解，而是对固定的 `schedule + warp_assign` 做确定性检查。

启用的检查包括：

- schedule 完整性：所有 op 都必须有固定启动时间。
- warp 展开：
  - 单 warp op 取一个 warp。
  - WGMMA 等多 warp op 展开成连续 warp 集合。
- consumer 关系：
  - 根据 `OpNode.parents` 和 `dependency_distance` 找每个 output 的消费者。
- 多迭代 live copy：
  - 根据 `check_L` 和 `base_I` 枚举 `iter_offsets`。
  - 判断一个 output copy 在 `tau` 是否 live。
- RMEM 容量：
  - 按 warp 累加 live RMEM footprint。
  - 超过 `check_reg_limit` 则复核失败。
- SMEM 容量：
  - 按 CTA 全局累加 live SMEM allocation。
  - 同一个 SMEM output 只按一次 allocation 统计，不按迭代副本重复累加。
  - 超过 `check_smem_limit` 则复核失败。

阶段结果：

- 如果通过：

```text
sol = opt_sol
status = SMT_OPTIMIZED
```

- 如果失败：

```text
sol = feasible_sol
status = SMT_FEASIBLE
```

也就是说，优化解 liveness 失败时，会退回阶段 1 的可行解，而不是立刻退回 naive plan。

### 阶段 4：feasibility 失败后的 retry optimize

如果阶段 1 完全没有找到可行解，会进入 retry：

```python
_run_joint_solver(
    solve_window=solve_window,
    optimize=True,
    solve_reg_limit=solve_reg_limit,
    solve_smem_limit=solve_smem_limit,
    enable_liveness=False,
)
```

此阶段启用的约束：

- 和阶段 2 相同：
  - 所有结构性约束。
  - 优化目标。

此阶段不启用的约束：

- 不展开内部 liveness。
- 不在 CP-SAT 主模型内约束 RMEM/SMEM 容量。

阶段结果：

- 如果 retry 仍然没有解：

```text
fallback = naive plan
status = SMT_UNSAT
```

- 如果 retry 找到解，进入阶段 5 做固定 liveness 复核。

### 阶段 5：对最终 joint 解做固定 liveness 复核

只要前面还没有得到 `liveness_info`，都会对当前 `sol` 再跑一次 `_check_fixed_liveness()`。

常见情况：

- 阶段 2 没有优化解，当前 `sol = feasible_sol`，这里复核 feasibility 解。
- 阶段 4 retry optimize 找到了解，当前 `sol = retry opt sol`，这里复核 retry 解。

启用的检查和阶段 3 相同：

- schedule 完整性。
- warp 展开。
- 多迭代 RMEM live copy。
- SMEM live allocation。
- `reg_limit` / `smem_limit` 上限。

阶段结果：

- 如果复核通过，记录 `reg_peak` / `smem_peak`。
- 如果复核失败且当前解来自 retry optimize：

```text
fallback = naive plan
status = SMT_LIVENESS_FAIL
```

- 如果复核失败但当前解是 feasibility 解，代码仍保留该 joint 解，并把失败信息记录到 `liveness_info`。因此这种情况下最终仍可能返回 `SMT_FEASIBLE`，但日志会出现 `SMT liveness check failed`。

### 汇总表

| 阶段 | 调用形态 | 结构性约束 | 内部 liveness 约束 | 优化目标 | 外部 liveness 复核 |
| --- | --- | --- | --- | --- | --- |
| 阶段 1 feasibility | `optimize=False` | 开 | 关 | 关 | 不立即复核 |
| 阶段 2 optimize | `optimize=True, enable_liveness=False` | 开 | 关 | 开 | 对 `opt_sol` 复核 |
| 阶段 4 retry optimize | `optimize=True, enable_liveness=False` | 开 | 关 | 开 | 阶段 5 复核 |
| 阶段 5 fixed check | 非 CP-SAT | 固定 schedule 检查 | 外部计算 | 无 | 开 |

这里的“结构性约束”包括 warp assignment、warpgroup、依赖、spill、blocking sync、subcore issue、FU reservation 和 barrier slot。当前 live 路径中，joint optimize 的主 CP-SAT 模型主要负责找结构性可行/紧凑排布，RMEM/SMEM 容量更多依赖后置的固定解复核。

## 最终 optimized plan

joint 成功后会把 scheduler 侧返回的 `schedule` 和 `warp_assign` 转回 op id：

```python
optimized_M = {
    idx: int(schedule[f"s{idx}"])
    for idx in ops
    if f"s{idx}" in schedule
}
```

如果某个 op 没有出现在 joint schedule 里，但 naive `base_M` 有记录，会用 `base_M` 补齐。

最终 plan 会更新：

- `I = base_I`
- `L = optimized_L`
- `M = optimized_M`
- `status = SMT_OPTIMIZED` 或 `SMT_FEASIBLE`
- `window`
- `warp_assign`
- `reg_peak`
- `smem_peak`
- `modular_rrt`
- `ordering`

其中新的 `modular_rrt` 根据 joint 后的 `optimized_M` 和 `OpNode.reservation` 重新折叠。

## 和后续旧 Phase B 路径的关系

在这段 joint optimize 之后，文件里还会打印：

```text
---- [原始路径] start smt solving -----
```

后面是旧的 `_phase_b_consumer_ordering()` 路径，用于 consumer ordering / register-aware ordering。本文档总结的是当前新增的 naive modulo 后接 `_solve_smt_joint_optimize()` 这条路径；它和后面的旧路径目前仍然在同一个 transform 流程里相邻存在。

## 调试时看日志的顺序

推荐按下面的标记读日志：

1. `enter _solve_naive_modulo_sched`
   - 进入 naive modulo。
2. `latencies=...` / `duration=...`
   - 检查 readiness latency 和 issue occupancy 是否混了。
3. `[Modulo Sched] Testing Initiation Interval`
   - 查看当前尝试的 II。
4. `solve done`、`I = ...`、`L = ...`、`M = ...`、`Modular RRT`
   - naive plan 是否非空。
5. `----start  _solve_smt_joint_optimize`
   - 进入 joint refine。
6. `---- num_warps ...`
   - 检查 PCWS 后的 warp 数。
7. `----- HeddleSCheduler: reg_limit=..., smem_limit=..., num_warps=...`
   - 进入 scheduler Phase B。
8. `SMT Optimize success` / `SMT liveness check success`
   - joint 优化和复核成功。
9. `SMT feasibility failed` / `Retry failed. Fallback to naive sched plan`
   - joint 不可行，回退到 naive。
10. `----warp_assign=...`、`optimized_M=...`
    - 最终 joint 输出。

## 一个压缩版心智模型

可以把当前流程理解成：

```text
TIR statements
  -> infos_list / deps_all / all_indices
  -> naive CP-SAT:
       fixed issue model, search minimal I, produce base_M/base_I/base_L
  -> joint CP-SAT:
       fixed base_I, use base_M as hint,
       solve Tv + warp_assign + FU/subcore/barrier + liveness
  -> optimized plan:
       updated M/L/RRT/ordering/warp_assign/reg_peak/smem_peak
  -> if joint fails:
       fallback to naive plan with status
```

最关键的语义边界是：

- `latency` 负责依赖 readiness。
- `duration` / `reservation` 负责 issue slot occupancy。
- `I` 是 steady-state 启动间隔。
- `L` 是求解窗口或发射窗口。
- `warp_assign` 只在 joint 阶段产生。
- liveness/容量约束主要在 joint 阶段处理，且当前会用固定解复核来降低主模型压力。
