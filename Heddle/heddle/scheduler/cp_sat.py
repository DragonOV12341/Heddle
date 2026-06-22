"""
Unified Partition + Schedule optimizer using CP-SAT.

Instead of the two-stage approach (AutoMixed enumerate → Heddle schedule each),
this module jointly decides:
  1. Which partition to use (e.g., combined vs split_dq_dkdv vs split_3way)
  2. How to schedule consumer statements within each kernel
  3. Register liveness across the schedule

Uses Google OR-Tools CP-SAT solver, which is purpose-built for scheduling
problems and typically 10-100x faster than Z3 for this problem class.

Canonical import:
    from heddle.scheduler.cp_sat import UnifiedScheduler, KernelSpec, OpSpec
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


# ====================================================================== #
# Data types (intentionally parallel to smt.py but decoupled)
# ====================================================================== #

class ResourceType(enum.Enum):
    TMA = "TMA"
    TensorCore = "TC"
    ALU = "ALU"
    SFU = "SFU"


class StorageKind(enum.Enum):
    RMEM = "RMEM"
    SMEM = "SMEM"


@dataclass
class OutputSpec:
    """An output value produced by an op."""
    name: str
    storage: StorageKind = StorageKind.RMEM
    footprint_bytes: int = 0


@dataclass
class OpSpec:
    """A single operation in the computation graph."""
    name: str
    resource_type: ResourceType
    latency: int
    outputs: List[OutputSpec] = field(default_factory=list)
    # Dependencies: list of (parent_op_name, iteration_distance[, blocking_sync])
    # 2-tuple (name, dist) is accepted for backward compatibility.
    deps: List[Union[Tuple[str, int], Tuple[str, int, bool]]] = field(default_factory=list)
    # Warp assignment: how many warps this op requires (1 = any single warp)
    warp_count: int = 1
    # Spill cost (cycles) for cross-warp data transfer
    spill_cost: int = 0
    # Force this op to a specific warp (-1 = solver decides)
    fixed_warp: int = -1


@dataclass
class KernelSpec:
    """A single kernel in a partition."""
    name: str
    ops: List[OpSpec]
    # Shared memory per CTA in bytes (for occupancy calculation)
    smem_bytes: int = 0
    # Threads per CTA
    threads: int = 128
    # Total number of CTA launches (grid size). 0 = auto from problem/tile size.
    grid_size: int = 0


@dataclass
class PartitionSpec:
    """A complete partition strategy."""
    name: str
    kernels: List[KernelSpec]
    description: str = ""


# ── H100 SM resource limits ──
@dataclass
class SMConfig:
    """Hardware SM resource limits."""
    regs_per_sm: int = 65536
    smem_per_sm: int = 232448       # 227 KB usable
    max_threads_per_sm: int = 2048
    max_blocks_per_sm: int = 32
    num_sms: int = 132              # H100 SXM

    def occupancy(self, regs_per_thread: int, threads: int, smem_bytes: int) -> int:
        """Compute active CTAs per SM."""
        if threads == 0 or smem_bytes == 0:
            return 1
        by_regs = self.regs_per_sm // max(regs_per_thread * threads, 1)
        by_smem = self.smem_per_sm // max(smem_bytes, 1)
        by_threads = self.max_threads_per_sm // max(threads, 1)
        return max(1, min(by_regs, by_smem, by_threads, self.max_blocks_per_sm))


H100 = SMConfig()


def _unpack_dep(dep_tuple) -> Tuple[str, int, bool]:
    """Unpack a dependency tuple, handling both 2-tuple and 3-tuple formats."""
    if len(dep_tuple) == 3:
        return dep_tuple[0], dep_tuple[1], dep_tuple[2]
    return dep_tuple[0], dep_tuple[1], False


@dataclass
class UnifiedResult:
    """Result of the unified solver."""
    partition: str
    kernel_schedules: Dict[str, Dict[str, int]]   # kernel_name -> {op_name: time}
    kernel_warp_assigns: Dict[str, Dict[str, int]] = field(default_factory=dict)
    kernel_reg_peaks: Dict[str, int] = field(default_factory=dict)
    kernel_occupancy: Dict[str, int] = field(default_factory=dict)
    total_makespan: int = -1
    solve_time_ms: float = 0.0
    status: str = ""


# ====================================================================== #
# Solver
# ====================================================================== #

class UnifiedScheduler:
    """CP-SAT based joint partition + schedule optimizer."""

    def __init__(
        self,
        partitions: List[PartitionSpec],
        *,
        fu_caps: Optional[Dict[ResourceType, int]] = None,
        reg_limit: int = 240 * 4,    # bytes (240 regs × 4 bytes)
        reg_safe_threshold: int = 200 * 4,  # bytes; above this, apply spill penalty
        spill_penalty_per_byte: int = 2,    # cycles penalty per byte over threshold
        num_warps: int = 1,           # consumer warp groups
        sm_config: Optional[SMConfig] = None,
        occupancy_weight: int = 10,   # penalty per lost CTA slot
        horizon: int = 120,
        timeout_s: float = 30.0,
    ):
        self.partitions = partitions
        self.reg_safe_threshold = reg_safe_threshold
        self.spill_penalty_per_byte = spill_penalty_per_byte
        self.sm_config = sm_config or H100
        self.occupancy_weight = occupancy_weight
        self.fu_caps = fu_caps or {
            ResourceType.TMA: 1,
            ResourceType.TensorCore: 1,
            ResourceType.ALU: 64,
            ResourceType.SFU: 16,
        }
        self.reg_limit = reg_limit
        self.num_warps = num_warps
        self.horizon = horizon
        self.timeout_s = timeout_s

    def solve(self) -> Optional[UnifiedResult]:
        from ortools.sat.python import cp_model

        # CP-SAT 的建模套路是先声明所有候选变量，再用约束把“不合法”
        # 的组合排除掉，最后给一个优化目标让 solver 在可行解中选最优。
        model = cp_model.CpModel()
        H = self.horizon
        W = max(self.num_warps, 1)

        # ============================================================
        # 1. Partition selection: exactly one partition is active
        # ============================================================
        part_vars = {}
        for p in self.partitions:
            # 每个 partition 一个 BoolVar；值为 1 表示最终选择该切分策略。
            part_vars[p.name] = model.new_bool_var(f"part_{p.name}")
        # 所有候选 partition 中必须且只能选一个。当前实际调用点常常只传
        # 一个 partition，但这里仍保留了联合选择多个策略的通用建模。
        model.add_exactly_one(part_vars.values())

        # ============================================================
        # 2. Per-partition, per-kernel, per-op: optional intervals + warp
        # ============================================================
        op_present: Dict[Tuple[str, str, str], any] = {}
        op_start: Dict[Tuple[str, str, str], any] = {}
        op_end: Dict[Tuple[str, str, str], any] = {}
        op_interval: Dict[Tuple[str, str, str], any] = {}
        op_warp: Dict[Tuple[str, str, str], any] = {}

        kernel_makespan: Dict[Tuple[str, str], any] = {}

        for p in self.partitions:
            pv = part_vars[p.name]
            for k in p.kernels:
                # kernel_makespan 是该 kernel 内所有 op 结束时间的上界。
                km_var = model.new_int_var(0, H, f"km_{p.name}_{k.name}")
                kernel_makespan[(p.name, k.name)] = km_var

                for op in k.ops:
                    key = (p.name, k.name, op.name)

                    # pres 表示这个 op 是否存在于当前解中。因为每个 op
                    # 归属于某个 partition，所以它跟 partition 选择变量绑定。
                    pres = model.new_bool_var(f"pres_{p.name}_{k.name}_{op.name}")
                    op_present[key] = pres

                    model.add(pres == 1).only_enforce_if(pv)
                    model.add(pres == 0).only_enforce_if(pv.negated())

                    # s/e 是调度时间；optional interval 把 “存在性 + 固定时长
                    # + 起止时间” 包成 CP-SAT 的调度对象，后面 cumulative /
                    # no_overlap 这类全局约束会直接消费它。
                    s = model.new_int_var(0, H, f"s_{p.name}_{k.name}_{op.name}")
                    e = model.new_int_var(0, H + op.latency, f"e_{p.name}_{k.name}_{op.name}")
                    iv = model.new_optional_interval_var(
                        s, op.latency, e, pres,
                        f"iv_{p.name}_{k.name}_{op.name}")

                    op_start[key] = s
                    op_end[key] = e
                    op_interval[key] = iv

                    # Warp assignment
                    # 每个 op 绑定一个 warp id。fixed_warp >= 0 时不让 solver
                    # 自由选择，而是强制放到调用方指定的 warp。
                    w_var = model.new_int_var(0, W - 1, f"w_{p.name}_{k.name}_{op.name}")
                    op_warp[key] = w_var
                    if op.fixed_warp >= 0:
                        model.add(w_var == op.fixed_warp)

                    # 只要 op 存在，kernel makespan 至少要覆盖它的结束时间。
                    model.add(km_var >= e).only_enforce_if(pres)

        # ============================================================
        # 3. Dependency constraints + spill cost + blocking_sync
        # ============================================================
        # Collect barrier intervals for per-warp no-overlap (Section 3b)
        # barrier_intervals[(p.name, k.name, w)] = list of optional intervals
        barrier_intervals: Dict[Tuple[str, str, int], list] = {}

        for p in self.partitions:
            for k in p.kernels:
                op_map = {op.name: op for op in k.ops}
                for op in k.ops:
                    key = (p.name, k.name, op.name)
                    pres = op_present[key]
                    for dep_tuple in op.deps:
                        dep_name, dist, is_blocking = _unpack_dep(dep_tuple)
                        if dep_name not in op_map:
                            continue
                        dep_key = (p.name, k.name, dep_name)
                        dep_op = op_map[dep_name]
                        base_lat = dep_op.latency

                        # 3a. Dependency timing with optional spill cost
                        # 普通依赖：consumer 不能早于 producer latency 之后开始。
                        # 如果有多个 warp 且 producer 标了 spill_cost，则把
                        # “同 warp / 跨 warp” reify 成 same_w，跨 warp 时额外加
                        # spill_cost，近似表达数据搬运或同步代价。
                        if W > 1 and dep_op.spill_cost > 0:
                            same_w = model.new_bool_var(
                                f"sw_{p.name}_{k.name}_{op.name}_{dep_name}")
                            model.add(
                                op_warp[key] == op_warp[dep_key]
                            ).only_enforce_if(same_w)
                            model.add(
                                op_warp[key] != op_warp[dep_key]
                            ).only_enforce_if(same_w.negated())
                            model.add(
                                op_start[key] >= op_start[dep_key] + base_lat
                            ).only_enforce_if([pres, same_w])
                            model.add(
                                op_start[key] >= op_start[dep_key] + base_lat + dep_op.spill_cost
                            ).only_enforce_if([pres, same_w.negated()])
                        else:
                            model.add(
                                op_start[key] >= op_start[dep_key] + base_lat
                            ).only_enforce_if(pres)

                        # 3b. Blocking sync constraints
                        if is_blocking and W > 1:
                            # Same-warp enforcement: producer and consumer must share a warp
                            # blocking_sync 表示这条边不能靠跨 warp spill 解决；
                            # producer 和 consumer 必须落在同一个 warp 上。
                            model.add(
                                op_warp[key] == op_warp[dep_key]
                            ).only_enforce_if(pres)

                            # Exclusive execution: create a barrier interval covering
                            # [consumer_start - producer_latency, consumer_start).
                            # Other ops on the same warp must not overlap this window.
                            # 这里把 “consumer 开始前必须保留的一段同步窗口”
                            # 建成一个 interval，稍后放进该 warp 的 no_overlap。
                            if base_lat > 0:
                                b_start = model.new_int_var(
                                    0, H,
                                    f"bs_{p.name}_{k.name}_{op.name}_{dep_name}")
                                model.add(
                                    b_start == op_start[key] - base_lat
                                ).only_enforce_if(pres)
                                model.add(b_start == 0).only_enforce_if(pres.negated())
                                b_iv = model.new_optional_interval_var(
                                    b_start, base_lat, op_start[key], pres,
                                    f"biv_{p.name}_{k.name}_{op.name}_{dep_name}")
                                for w in range(W):
                                    bk = (p.name, k.name, w)
                                    barrier_intervals.setdefault(bk, []).append(
                                        (b_iv, key))

        # ============================================================
        # 3c. Per-warp no-overlap for barrier exclusion zones
        # ============================================================
        if barrier_intervals:
            for p in self.partitions:
                for k in p.kernels:
                    for w in range(W):
                        bk = (p.name, k.name, w)
                        barriers = barrier_intervals.get(bk, [])
                        if not barriers:
                            continue
                        # Collect all op intervals on this warp (conditional)
                        # plus barrier intervals (also conditional on same warp)
                        warp_no_overlap = []
                        for op in k.ops:
                            okey = (p.name, k.name, op.name)
                            # Skip ops that are endpoints of a barrier on this warp
                            # on_w/both 用来重新包装 interval：只有 op 存在且
                            # 被分配到当前 warp 时，它才参与这个 warp 的互斥检查。
                            on_w = model.new_bool_var(
                                f"noo_{p.name}_{k.name}_{op.name}_w{w}")
                            model.add(op_warp[okey] == w).only_enforce_if(on_w)
                            model.add(op_warp[okey] != w).only_enforce_if(on_w.negated())
                            both = model.new_bool_var(
                                f"noob_{p.name}_{k.name}_{op.name}_w{w}")
                            model.add_bool_and(
                                [op_present[okey], on_w]
                            ).only_enforce_if(both)
                            model.add_bool_or(
                                [op_present[okey].negated(), on_w.negated()]
                            ).only_enforce_if(both.negated())
                            warp_iv = model.new_optional_interval_var(
                                op_start[okey], op.latency, op_end[okey], both,
                                f"noiv_{p.name}_{k.name}_{op.name}_w{w}")
                            warp_no_overlap.append(warp_iv)

                        for b_iv, consumer_key in barriers:
                            # Barrier interval is active only when consumer is on this warp
                            # barrier interval 原本只由 consumer 是否存在控制；
                            # 这里再加一层 “consumer 是否在当前 warp” 的条件。
                            on_w_b = model.new_bool_var(
                                f"bw_{p.name}_{k.name}_{consumer_key[2]}_w{w}")
                            model.add(op_warp[consumer_key] == w).only_enforce_if(on_w_b)
                            model.add(op_warp[consumer_key] != w).only_enforce_if(on_w_b.negated())
                            # Re-wrap barrier interval conditioned on warp assignment
                            b_start_var = b_iv.StartExpr()
                            b_size = b_iv.SizeExpr()
                            b_end_var = b_iv.EndExpr()
                            cond_b = model.new_bool_var(
                                f"cb_{p.name}_{k.name}_{consumer_key[2]}_w{w}")
                            model.add_bool_and(
                                [op_present[consumer_key], on_w_b]
                            ).only_enforce_if(cond_b)
                            model.add_bool_or(
                                [op_present[consumer_key].negated(), on_w_b.negated()]
                            ).only_enforce_if(cond_b.negated())
                            cond_biv = model.new_optional_interval_var(
                                b_start_var, b_size, b_end_var, cond_b,
                                f"cbiv_{p.name}_{k.name}_{consumer_key[2]}_w{w}")
                            warp_no_overlap.append(cond_biv)

                        if len(warp_no_overlap) > 1:
                            # 同一个 warp 上，真实 op interval 和 blocking sync
                            # 的保护窗口不能互相重叠。
                            model.add_no_overlap(warp_no_overlap)

        # ============================================================
        # 4. FU capacity (per kernel, per warp when W > 1)
        # ============================================================
        for p in self.partitions:
            for k in p.kernels:
                for fu_type, cap in self.fu_caps.items():
                    fu_ops = [(op, (p.name, k.name, op.name))
                              for op in k.ops
                              if op.resource_type == fu_type]
                    if not fu_ops:
                        continue
                    if W == 1:
                        # 单 warp 情况：同类 FU 的所有 op 共享一个 cumulative
                        # 容量约束，cap 表示同一时间最多可并发的同类 op 数。
                        intervals_for_fu = []
                        demands_for_fu = []
                        for op, key in fu_ops:
                            intervals_for_fu.append(op_interval[key])
                            demands_for_fu.append(1)
                        if intervals_for_fu:
                            model.add_cumulative(
                                intervals_for_fu, demands_for_fu, cap)
                    else:
                        # 多 warp 情况：先按 warp 条件化地重包 interval，再对
                        # 每个 warp 单独做 FU 容量约束。
                        for w in range(W):
                            warp_intervals = []
                            warp_demands = []
                            for op, key in fu_ops:
                                on_w = model.new_bool_var(
                                    f"fuw_{p.name}_{k.name}_{op.name}_w{w}")
                                model.add(op_warp[key] == w).only_enforce_if(on_w)
                                model.add(op_warp[key] != w).only_enforce_if(on_w.negated())
                                both = model.new_bool_var(
                                    f"fub_{p.name}_{k.name}_{op.name}_w{w}")
                                model.add_bool_and(
                                    [op_present[key], on_w]
                                ).only_enforce_if(both)
                                model.add_bool_or(
                                    [op_present[key].negated(), on_w.negated()]
                                ).only_enforce_if(both.negated())
                                warp_iv = model.new_optional_interval_var(
                                    op_start[key], op.latency, op_end[key],
                                    both,
                                    f"fuiv_{p.name}_{k.name}_{op.name}_w{w}")
                                warp_intervals.append(warp_iv)
                                warp_demands.append(1)
                            if warp_intervals:
                                model.add_cumulative(
                                    warp_intervals, warp_demands, cap)

        # ============================================================
        # 5. Register liveness tracking (per kernel, per warp)
        # ============================================================
        CHECKPOINT_STEP = 1
        # 用离散 checkpoint 近似寄存器活跃区间。这里步长为 1，所以会检查
        # horizon 内每个整数时刻的 RMEM live bytes。
        checkpoints = list(range(0, H + 1, CHECKPOINT_STEP))

        for p in self.partitions:
            pv = part_vars[p.name]
            for k in p.kernels:
                op_map = {op.name: op for op in k.ops}
                consumer_map: Dict[str, List[str]] = {}
                for op in k.ops:
                    for dep_tuple in op.deps:
                        dep_name, _, _ = _unpack_dep(dep_tuple)
                        if dep_name in op_map:
                            for out in op_map[dep_name].outputs:
                                # 反向索引：某个 output 会被哪些 consumer 使用。
                                # 后面用它判断 output 是否已经被全部消费。
                                consumer_map.setdefault(out.name, []).append(op.name)

                rmem_outputs: List[Tuple[OpSpec, OutputSpec]] = []
                for op in k.ops:
                    for out in op.outputs:
                        if out.storage.value == StorageKind.RMEM.value and out.footprint_bytes > 0:
                            rmem_outputs.append((op, out))

                if not rmem_outputs:
                    continue

                for w in range(W):
                    for tau in checkpoints:
                        live_terms = []
                        for prod_op, out in rmem_outputs:
                            prod_key = (p.name, k.name, prod_op.name)
                            consumers = consumer_map.get(out.name, [])

                            # on_warp: producer is on this warp
                            # 只把该 warp 生产的 RMEM output 计入该 warp 的压力。
                            on_warp = model.new_bool_var(
                                f"ow_{p.name}_{k.name}_{out.name}_w{w}_t{tau}")
                            model.add(op_warp[prod_key] == w).only_enforce_if(on_warp)
                            model.add(op_warp[prod_key] != w).only_enforce_if(on_warp.negated())

                            # output 在 tau 时刻前已经产生，才可能进入 live 集合。
                            is_produced = model.new_bool_var(
                                f"prod_{p.name}_{k.name}_{out.name}_w{w}_t{tau}")
                            model.add(
                                op_start[prod_key] <= tau
                            ).only_enforce_if(is_produced)
                            model.add(
                                op_start[prod_key] > tau
                            ).only_enforce_if(is_produced.negated())

                            if consumers:
                                consumer_started = []
                                for c_name in consumers:
                                    c_key = (p.name, k.name, c_name)
                                    # 这里以 consumer 的 start 作为 “已经读取/消费”
                                    # 的近似边界；所有 consumer 都开始后认为该
                                    # output 不再需要保持 live。
                                    cs = model.new_bool_var(
                                        f"cs_{p.name}_{k.name}_{out.name}_{c_name}_w{w}_t{tau}")
                                    model.add(
                                        op_start[c_key] <= tau
                                    ).only_enforce_if(cs)
                                    model.add(
                                        op_start[c_key] > tau
                                    ).only_enforce_if(cs.negated())
                                    consumer_started.append(cs)

                                all_consumed = model.new_bool_var(
                                    f"ac_{p.name}_{k.name}_{out.name}_w{w}_t{tau}")
                                model.add_bool_and(consumer_started).only_enforce_if(all_consumed)
                                for cs in consumer_started:
                                    model.add_bool_or([all_consumed.negated(), cs])
                                model.add_bool_or(
                                    [cs.negated() for cs in consumer_started] + [all_consumed]
                                )

                                # live = 已产生 且 尚未被全部 consumer 消费 且
                                # producer 在当前 warp 且 partition 被选中。
                                is_live = model.new_bool_var(
                                    f"live_{p.name}_{k.name}_{out.name}_w{w}_t{tau}")
                                model.add_bool_and(
                                    [is_produced, all_consumed.negated(), on_warp, pv]
                                ).only_enforce_if(is_live)
                                model.add_bool_or(
                                    [is_produced.negated(), all_consumed, on_warp.negated(), pv.negated()]
                                ).only_enforce_if(is_live.negated())

                                live_terms.append((is_live, out.footprint_bytes))
                            else:
                                # 没有显式 consumer 的 output 一旦产生就一直计入
                                # live pressure，直到 kernel 结束。
                                is_live = model.new_bool_var(
                                    f"live_{p.name}_{k.name}_{out.name}_w{w}_t{tau}")
                                model.add_bool_and(
                                    [is_produced, on_warp, pv]
                                ).only_enforce_if(is_live)
                                model.add_bool_or(
                                    [is_produced.negated(), on_warp.negated(), pv.negated()]
                                ).only_enforce_if(is_live.negated())
                                live_terms.append((is_live, out.footprint_bytes))

                        if live_terms:
                            # 这是硬约束：每个 warp 在每个 checkpoint 的 RMEM
                            # live bytes 不能超过 reg_limit。
                            model.add(
                                sum(bv * fb for bv, fb in live_terms) <= self.reg_limit
                            )

        # ============================================================
        # 6. Objective: minimize total execution time (sum of kernels)
        # ============================================================
        # total 只绑定到被选中的 partition：它等于该 partition 内所有 kernel
        # makespan 的和。注意这里的目标只最小化时间，寄存器压力目前是硬约束，
        # 不是 tie-breaker。
        total = model.new_int_var(0, H * 10, "total_makespan")
        for p in self.partitions:
            pv = part_vars[p.name]
            km_sum = sum(kernel_makespan[(p.name, k.name)] for k in p.kernels)
            model.add(total == km_sum).only_enforce_if(pv)

        model.minimize(total)

        # ============================================================
        # 7. Solve
        # ============================================================
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout_s
        solver.parameters.num_workers = 4

        t0 = time.monotonic()
        status = solver.solve(model)
        solve_ms = (time.monotonic() - t0) * 1000

        status_name = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN",
        }.get(status, f"STATUS_{status}")

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # 不可行/超时无可行解时也返回 UnifiedResult，方便调用方看 status
            # 和 solve_time，而不是只拿到 None。
            return UnifiedResult(
                partition="NONE", kernel_schedules={}, kernel_reg_peaks={},
                total_makespan=-1, solve_time_ms=solve_ms, status=status_name,
            )

        # ============================================================
        # 8. Extract solution
        # ============================================================
        # 先根据 partition BoolVar 找到被选中的 PartitionSpec。
        chosen_partition = None
        for p in self.partitions:
            if solver.value(part_vars[p.name]):
                chosen_partition = p
                break

        schedules = {}
        warp_assigns = {}
        reg_peaks = {}
        for k in chosen_partition.kernels:
            sched = {}
            warps = {}
            for op in k.ops:
                key = (chosen_partition.name, k.name, op.name)
                # 调用方主要消费两类结果：op 的开始时间用于排序，
                # op_warp 用于后续 warp-specialized lowering 的 hint。
                sched[op.name] = solver.value(op_start[key])
                warps[op.name] = solver.value(op_warp[key])
            schedules[k.name] = sched
            warp_assigns[k.name] = warps

            # Compute register peak per warp from solution
            # 下面不是再加约束，而是用已求出的 schedule/warp assignment
            # 重新扫描一遍，计算实际 peak，写进 UnifiedResult 供外部诊断。
            op_map = {op.name: op for op in k.ops}
            consumer_map: Dict[str, List[str]] = {}
            for op in k.ops:
                for dep_tuple in op.deps:
                    dep_name, _, _ = _unpack_dep(dep_tuple)
                    if dep_name in op_map:
                        for out in op_map[dep_name].outputs:
                            consumer_map.setdefault(out.name, []).append(op.name)

            max_peak = 0
            km_val = solver.value(kernel_makespan[(chosen_partition.name, k.name)])
            for w in range(W):
                peak = 0
                for tau in range(km_val + 1):
                    total_live = 0
                    for op in k.ops:
                        if warps[op.name] != w:
                            continue
                        prod_t = sched[op.name]
                        for out in op.outputs:
                            if out.storage.value != StorageKind.RMEM.value or out.footprint_bytes <= 0:
                                continue
                            if prod_t > tau:
                                continue
                            consumers = consumer_map.get(out.name, [])
                            if consumers:
                                all_done = all(
                                    sched[c] <= tau for c in consumers
                                )
                                if not all_done:
                                    total_live += out.footprint_bytes
                            else:
                                total_live += out.footprint_bytes
                    peak = max(peak, total_live)
                max_peak = max(max_peak, peak)
            reg_peaks[k.name] = max_peak

        return UnifiedResult(
            partition=chosen_partition.name,
            kernel_schedules=schedules,
            kernel_warp_assigns=warp_assigns,
            kernel_reg_peaks=reg_peaks,
            total_makespan=solver.value(total),
            solve_time_ms=solve_ms,
            status=status_name,
        )

    # ================================================================== #
    # Modulo scheduling: steady-state pipelined loop
    # ================================================================== #

    def solve_modulo(
        self, *, min_ii: int = 1, max_ii: int = 10,
    ) -> Optional[UnifiedResult]:
        """Joint partition + modulo schedule.

        For each candidate II (initiation interval), builds a CP-SAT model
        where FU capacity is checked per modulo slot and register liveness
        accounts for cross-iteration overlap (incoming_live).

        Returns the solution with the smallest feasible II.
        """
        for ii in range(min_ii, max_ii + 1):
            result = self._solve_modulo_for_ii(ii)
            if result is not None:
                result.status = f"OPTIMAL(II={ii})"
                return result
        return None

    def _solve_modulo_for_ii(self, ii: int) -> Optional[UnifiedResult]:
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        H = self.horizon
        W = max(self.num_warps, 1)

        # ============================================================
        # 1. Partition selection
        # ============================================================
        part_vars = {}
        for p in self.partitions:
            part_vars[p.name] = model.new_bool_var(f"part_{p.name}")
        model.add_exactly_one(part_vars.values())

        # ============================================================
        # 2. Per-partition, per-kernel, per-op variables
        # ============================================================
        op_present: Dict[Tuple[str, str, str], any] = {}
        op_start: Dict[Tuple[str, str, str], any] = {}
        op_slot: Dict[Tuple[str, str, str], any] = {}   # start % II
        op_warp: Dict[Tuple[str, str, str], any] = {}   # warp assignment (int 0..W-1)
        kernel_makespan: Dict[Tuple[str, str], any] = {}

        for p in self.partitions:
            pv = part_vars[p.name]
            for k in p.kernels:
                km_var = model.new_int_var(0, H, f"km_{p.name}_{k.name}")
                kernel_makespan[(p.name, k.name)] = km_var

                for op in k.ops:
                    key = (p.name, k.name, op.name)

                    pres = model.new_bool_var(f"pres_{p.name}_{k.name}_{op.name}")
                    op_present[key] = pres
                    model.add(pres == 1).only_enforce_if(pv)
                    model.add(pres == 0).only_enforce_if(pv.negated())

                    s = model.new_int_var(0, H, f"s_{p.name}_{k.name}_{op.name}")
                    op_start[key] = s

                    s_mod = model.new_int_var(0, ii - 1, f"sm_{p.name}_{k.name}_{op.name}")
                    op_slot[key] = s_mod
                    q = model.new_int_var(0, H // ii, f"q_{p.name}_{k.name}_{op.name}")
                    model.add(s == q * ii + s_mod)

                    # Warp assignment
                    w = model.new_int_var(0, W - 1, f"w_{p.name}_{k.name}_{op.name}")
                    op_warp[key] = w
                    # Fix warp if specified (e.g., TMA producers → warp 0)
                    if op.fixed_warp >= 0:
                        model.add(w == op.fixed_warp)

                    model.add(km_var >= s + op.latency).only_enforce_if(pres)

        # ============================================================
        # 3. Dependencies (with iteration distance + spill cost)
        # ============================================================
        for p in self.partitions:
            for k in p.kernels:
                op_map = {op.name: op for op in k.ops}
                for op in k.ops:
                    key = (p.name, k.name, op.name)
                    for dep_tuple in op.deps:
                        dep_name, dist, _ = _unpack_dep(dep_tuple)
                        if dep_name not in op_map:
                            continue
                        dep_key = (p.name, k.name, dep_name)
                        dep_op = op_map[dep_name]
                        base_lat = dep_op.latency

                        if W > 1 and dep_op.spill_cost > 0:
                            # Cross-warp: add spill cost
                            same_w = model.new_bool_var(
                                f"sw_{p.name}_{k.name}_{op.name}_{dep_name}")
                            model.add(
                                op_warp[key] == op_warp[dep_key]
                            ).only_enforce_if(same_w)
                            model.add(
                                op_warp[key] != op_warp[dep_key]
                            ).only_enforce_if(same_w.negated())
                            # If same warp: base latency; if cross-warp: + spill_cost
                            model.add(
                                op_start[key] >= op_start[dep_key] + base_lat - dist * ii
                            ).only_enforce_if([op_present[key], same_w])
                            model.add(
                                op_start[key] >= op_start[dep_key] + base_lat + dep_op.spill_cost - dist * ii
                            ).only_enforce_if([op_present[key], same_w.negated()])
                        else:
                            model.add(
                                op_start[key] >= op_start[dep_key] + base_lat - dist * ii
                            ).only_enforce_if(op_present[key])

        # ============================================================
        # 4. FU capacity per modulo slot per warp (per kernel)
        # ============================================================
        for p in self.partitions:
            for k in p.kernels:
                for fu_type, cap in self.fu_caps.items():
                    fu_ops = [(op, (p.name, k.name, op.name))
                              for op in k.ops
                              if op.resource_type.value == fu_type.value]
                    if not fu_ops:
                        continue
                    if W == 1:
                        # Single warp: global modulo constraint
                        for slot in range(ii):
                            at_slot = []
                            for op, key in fu_ops:
                                b = model.new_bool_var(
                                    f"at_{p.name}_{k.name}_{op.name}_s{slot}")
                                model.add(op_slot[key] == slot).only_enforce_if(b)
                                model.add(op_slot[key] != slot).only_enforce_if(b.negated())
                                at_slot.append(b)
                            model.add(sum(at_slot) <= cap)
                    else:
                        # Multi-warp: FU capacity per warp per slot
                        for w in range(W):
                            for slot in range(ii):
                                at_slot_warp = []
                                for op, key in fu_ops:
                                    b = model.new_bool_var(
                                        f"atw_{p.name}_{k.name}_{op.name}_w{w}_s{slot}")
                                    # b ↔ (op at this slot AND on this warp)
                                    slot_match = model.new_bool_var(
                                        f"sm_{p.name}_{k.name}_{op.name}_w{w}_s{slot}")
                                    model.add(op_slot[key] == slot).only_enforce_if(slot_match)
                                    model.add(op_slot[key] != slot).only_enforce_if(slot_match.negated())
                                    warp_match = model.new_bool_var(
                                        f"wm_{p.name}_{k.name}_{op.name}_w{w}_s{slot}")
                                    model.add(op_warp[key] == w).only_enforce_if(warp_match)
                                    model.add(op_warp[key] != w).only_enforce_if(warp_match.negated())
                                    model.add_bool_and([slot_match, warp_match]).only_enforce_if(b)
                                    model.add_bool_or([slot_match.negated(), warp_match.negated()]).only_enforce_if(b.negated())
                                    at_slot_warp.append(b)
                                model.add(sum(at_slot_warp) <= cap)

        # ============================================================
        # 5. Register liveness per warp (within-iteration + incoming_live)
        # ============================================================
        CHECKPOINT_STEP = max(1, ii)
        checkpoints = list(range(0, H + 1, CHECKPOINT_STEP))

        for p in self.partitions:
            pv = part_vars[p.name]
            for k in p.kernels:
                op_map = {op.name: op for op in k.ops}
                consumer_map: Dict[str, List[Tuple[str, int]]] = {}
                for op in k.ops:
                    for dep_tuple in op.deps:
                        dep_name, dist, _ = _unpack_dep(dep_tuple)
                        if dep_name in op_map:
                            for out in op_map[dep_name].outputs:
                                consumer_map.setdefault(out.name, []).append((op.name, dist))

                rmem_outputs: List[Tuple[OpSpec, OutputSpec]] = []
                for op in k.ops:
                    for out in op.outputs:
                        if out.storage.value == StorageKind.RMEM.value and out.footprint_bytes > 0:
                            rmem_outputs.append((op, out))

                if not rmem_outputs:
                    continue

                for w in range(W):
                    for tau in checkpoints:
                        live_terms = []
                        for prod_op, out in rmem_outputs:
                            prod_key = (p.name, k.name, prod_op.name)
                            consumers = consumer_map.get(out.name, [])
                            same_iter = [(c, d) for c, d in consumers if d == 0]
                            cross_iter = [(c, d) for c, d in consumers if d > 0]

                            # on_warp: producer is on this warp
                            on_warp = model.new_bool_var(
                                f"ow_{p.name}_{k.name}_{out.name}_w{w}_t{tau}")
                            model.add(op_warp[prod_key] == w).only_enforce_if(on_warp)
                            model.add(op_warp[prod_key] != w).only_enforce_if(on_warp.negated())

                            is_produced = model.new_bool_var(
                                f"mp_{p.name}_{k.name}_{out.name}_w{w}_t{tau}")
                            model.add(op_start[prod_key] <= tau).only_enforce_if(is_produced)
                            model.add(op_start[prod_key] > tau).only_enforce_if(is_produced.negated())

                            if same_iter:
                                cs_list = []
                                for c_name, _ in same_iter:
                                    c_key = (p.name, k.name, c_name)
                                    cs = model.new_bool_var(
                                        f"mcs_{p.name}_{k.name}_{out.name}_{c_name}_w{w}_t{tau}")
                                    model.add(op_start[c_key] <= tau).only_enforce_if(cs)
                                    model.add(op_start[c_key] > tau).only_enforce_if(cs.negated())
                                    cs_list.append(cs)

                                all_consumed = model.new_bool_var(
                                    f"mac_{p.name}_{k.name}_{out.name}_w{w}_t{tau}")
                                model.add_bool_and(cs_list).only_enforce_if(all_consumed)
                                model.add_bool_or(
                                    [c.negated() for c in cs_list] + [all_consumed])

                                is_live = model.new_bool_var(
                                    f"ml_{p.name}_{k.name}_{out.name}_w{w}_t{tau}")
                                model.add_bool_and(
                                    [is_produced, all_consumed.negated(), on_warp, pv]
                                ).only_enforce_if(is_live)
                                model.add_bool_or(
                                    [is_produced.negated(), all_consumed, on_warp.negated(), pv.negated()]
                                ).only_enforce_if(is_live.negated())
                                live_terms.append((is_live, out.footprint_bytes))
                            elif not cross_iter:
                                is_live = model.new_bool_var(
                                    f"ml_{p.name}_{k.name}_{out.name}_w{w}_t{tau}")
                                model.add_bool_and([is_produced, on_warp, pv]).only_enforce_if(is_live)
                                model.add_bool_or(
                                    [is_produced.negated(), on_warp.negated(), pv.negated()]
                                ).only_enforce_if(is_live.negated())
                                live_terms.append((is_live, out.footprint_bytes))

                            if cross_iter:
                                ci_pending = []
                                for c_name, _ in cross_iter:
                                    c_key = (p.name, k.name, c_name)
                                    cp_var = model.new_bool_var(
                                        f"mip_{p.name}_{k.name}_{out.name}_{c_name}_w{w}_t{tau}")
                                    model.add(op_start[c_key] > tau).only_enforce_if(cp_var)
                                    model.add(op_start[c_key] <= tau).only_enforce_if(cp_var.negated())
                                    ci_pending.append(cp_var)

                                any_pending = model.new_bool_var(
                                    f"map_{p.name}_{k.name}_{out.name}_w{w}_t{tau}")
                                model.add_bool_or(ci_pending).only_enforce_if(any_pending)
                                model.add_bool_and(
                                    [c.negated() for c in ci_pending]
                                ).only_enforce_if(any_pending.negated())

                                inc_live = model.new_bool_var(
                                    f"mil_{p.name}_{k.name}_{out.name}_w{w}_t{tau}")
                                model.add_bool_and([any_pending, on_warp, pv]).only_enforce_if(inc_live)
                                model.add_bool_or(
                                    [any_pending.negated(), on_warp.negated(), pv.negated()]
                                ).only_enforce_if(inc_live.negated())
                                live_terms.append((inc_live, out.footprint_bytes))

                        if live_terms:
                            model.add(sum(bv * fb for bv, fb in live_terms) <= self.reg_limit)

        # ============================================================
        # 6. Objective: makespan + spill penalty - overlap bonus
        # ============================================================
        total = model.new_int_var(0, H * 10, "total_makespan")
        for p in self.partitions:
            pv = part_vars[p.name]
            km_sum = sum(kernel_makespan[(p.name, k.name)] for k in p.kernels)
            model.add(total == km_sum).only_enforce_if(pv)

        # 6a. Spill penalty: penalize reg peaks near the hardware limit.
        # ptxas often spills when the model says "just under the limit"
        # due to address computation regs, control flow, etc.
        spill_penalty = model.new_int_var(0, H * 100, "spill_penalty")
        safe = self.reg_safe_threshold
        coeff = self.spill_penalty_per_byte
        if safe > 0 and coeff > 0:
            # For each partition, compute max per-warp reg peak across kernels.
            # Use checkpoint liveness already tracked: the max live_terms sum
            # at any checkpoint gives the peak. We approximate by adding a
            # penalty proportional to (estimated_peak - safe_threshold)+.
            # Since exact peak is hard to linearize, we use a per-kernel proxy:
            # sum of all RMEM output footprints that are simultaneously alive
            # in the worst case (all produced, none consumed) as an upper bound.
            for p in self.partitions:
                pv = part_vars[p.name]
                for k in p.kernels:
                    op_map = {op.name: op for op in k.ops}
                    # Build consumer set: which outputs have consumers?
                    consumed_outputs = set()
                    for op in k.ops:
                        for dep_tuple in op.deps:
                            dep_name, _, _ = _unpack_dep(dep_tuple)
                            if dep_name in op_map:
                                for out in op_map[dep_name].outputs:
                                    consumed_outputs.add(out.name)
                    # Accumulator footprint: outputs with NO consumers
                    # (they persist across the entire loop body)
                    accum_bytes = sum(
                        out.footprint_bytes
                        for op in k.ops for out in op.outputs
                        if out.storage.value == StorageKind.RMEM.value
                        and out.footprint_bytes > 0
                        and out.name not in consumed_outputs
                    )
                    overshoot = max(0, accum_bytes - safe)
                    if overshoot > 0:
                        model.add(spill_penalty >= overshoot * coeff).only_enforce_if(pv)
            # If no partition overshoots, penalty = 0
            for p in self.partitions:
                pv = part_vars[p.name]
                all_under = True
                for k in p.kernels:
                    op_map_check = {op.name: op for op in k.ops}
                    consumed_check = set()
                    for op in k.ops:
                        for dep_tuple in op.deps:
                            dep_name, _, _ = _unpack_dep(dep_tuple)
                            if dep_name in op_map_check:
                                for out in op_map_check[dep_name].outputs:
                                    consumed_check.add(out.name)
                    acc = sum(
                        out.footprint_bytes
                        for op in k.ops for out in op.outputs
                        if out.storage.value == StorageKind.RMEM.value
                        and out.footprint_bytes > 0
                        and out.name not in consumed_check
                    )
                    if acc > safe:
                        all_under = False
                if all_under:
                    model.add(spill_penalty == 0).only_enforce_if(pv)
        else:
            model.add(spill_penalty == 0)

        # 6b. Overlap bonus: maximize producer-consumer overlap for TMA ops.
        # For each TMA producer with a consumer dependency (forward_wait),
        # the overlap = consumer_start - producer_start. Larger overlap
        # means more latency hiding. We reward this as a negative cost.
        overlap_bonus = model.new_int_var(0, H * 10, "overlap_bonus")
        tma_overlap_terms = []
        for p in self.partitions:
            pv = part_vars[p.name]
            for k in p.kernels:
                op_map = {op.name: op for op in k.ops}
                for op in k.ops:
                    if op.fixed_warp != 0:
                        continue  # not a producer
                    key = (p.name, k.name, op.name)
                    # Find consumers of this producer (ops that depend on it with dist=0)
                    for c_op in k.ops:
                        for dep_tuple in c_op.deps:
                            dep_name, dist, _ = _unpack_dep(dep_tuple)
                            if dep_name == op.name and dist == 0 and c_op.fixed_warp != 0:
                                c_key = (p.name, k.name, c_op.name)
                                # overlap = consumer_start - producer_start
                                gap = model.new_int_var(0, H, f"gap_{p.name}_{k.name}_{op.name}_{c_op.name}")
                                model.add(gap == op_start[c_key] - op_start[key]).only_enforce_if(pv)
                                tma_overlap_terms.append((gap, pv))
        if tma_overlap_terms:
            # Sum all overlaps for the chosen partition
            for gap, pv in tma_overlap_terms:
                model.add(overlap_bonus >= gap).only_enforce_if(pv)
        else:
            model.add(overlap_bonus == 0)

        # 6c. Occupancy penalty: penalize low SM occupancy.
        # Fewer active CTAs per SM → worse latency hiding.
        # Compute occupancy statically per partition (function of SMEM, threads, regs).
        occ_penalty = model.new_int_var(0, H * 100, "occ_penalty")
        sm = self.sm_config
        ow = self.occupancy_weight
        if ow > 0:
            for p in self.partitions:
                pv = part_vars[p.name]
                # Partition occupancy = min occupancy across its kernels.
                # Use register estimate from RMEM outputs to compute regs/thread.
                min_occ = 99
                for k in p.kernels:
                    if k.smem_bytes <= 0:
                        occ = 4
                    else:
                        # Estimate regs/thread from accumulator footprints
                        total_rmem = sum(
                            out.footprint_bytes
                            for op in k.ops for out in op.outputs
                            if out.storage.value == StorageKind.RMEM.value
                        )
                        est_regs = total_rmem // 4 + 20  # bytes → 32b regs + overhead
                        occ = sm.occupancy(est_regs, k.threads, k.smem_bytes)
                    min_occ = min(min_occ, occ)
                if not p.kernels:
                    min_occ = 1
                max_occ = 4
                penalty = ow * max(0, max_occ - min_occ)
                model.add(occ_penalty == penalty).only_enforce_if(pv)
        else:
            model.add(occ_penalty == 0)

        # 6d. Iteration overhead: more loop iterations = more overhead.
        # Penalize partitions with more total grid work (smaller tiles → more CTAs).
        iter_penalty = model.new_int_var(0, H * 100, "iter_penalty")
        for p in self.partitions:
            pv = part_vars[p.name]
            total_grid = sum(k.grid_size for k in p.kernels if k.grid_size > 0)
            if total_grid > 0:
                # Each inner loop iteration incurs TMA issue + barrier overhead.
                # grid_size here represents inner iteration count per CTA.
                model.add(iter_penalty == total_grid).only_enforce_if(pv)
            else:
                model.add(iter_penalty == 0).only_enforce_if(pv)

        # Combined objective:
        # minimize(makespan + spill + occupancy + iteration - overlap)
        objective = model.new_int_var(-H * 100, H * 200, "objective")
        model.add(objective == total + spill_penalty + occ_penalty + iter_penalty - overlap_bonus)
        model.minimize(objective)

        # ============================================================
        # 7. Solve
        # ============================================================
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout_s
        solver.parameters.num_workers = 4

        t0 = time.monotonic()
        status = solver.solve(model)
        solve_ms = (time.monotonic() - t0) * 1000

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        # ============================================================
        # 8. Extract
        # ============================================================
        chosen_partition = None
        for p in self.partitions:
            if solver.value(part_vars[p.name]):
                chosen_partition = p
                break

        schedules = {}
        warp_assigns = {}
        reg_peaks = {}
        for k in chosen_partition.kernels:
            sched = {}
            warps = {}
            for op in k.ops:
                key = (chosen_partition.name, k.name, op.name)
                sched[op.name] = solver.value(op_start[key])
                warps[op.name] = solver.value(op_warp[key])
            schedules[k.name] = sched
            warp_assigns[k.name] = warps

            # Compute register peak per warp
            op_map = {op.name: op for op in k.ops}
            cm: Dict[str, List[Tuple[str, int]]] = {}
            for op in k.ops:
                for dep_tuple in op.deps:
                    dep_name, dist, _ = _unpack_dep(dep_tuple)
                    if dep_name in op_map:
                        for out in op_map[dep_name].outputs:
                            cm.setdefault(out.name, []).append((op.name, dist))

            max_peak = 0
            km_val = solver.value(kernel_makespan[(chosen_partition.name, k.name)])
            for w in range(W):
                peak = 0
                for tau in range(km_val + 1):
                    total_live = 0
                    for op in k.ops:
                        if warps[op.name] != w:
                            continue
                        prod_t = sched[op.name]
                        for out in op.outputs:
                            if out.storage.value != StorageKind.RMEM.value or out.footprint_bytes <= 0:
                                continue
                            consumers = cm.get(out.name, [])
                            same_iter = [(c, d) for c, d in consumers if d == 0]
                            cross_iter = [(c, d) for c, d in consumers if d > 0]
                            if prod_t <= tau:
                                if same_iter:
                                    if not all(sched[c] <= tau for c, _ in same_iter):
                                        total_live += out.footprint_bytes
                                elif not cross_iter:
                                    total_live += out.footprint_bytes
                            if cross_iter:
                                if any(sched[c] > tau for c, _ in cross_iter):
                                    total_live += out.footprint_bytes
                    peak = max(peak, total_live)
                max_peak = max(max_peak, peak)
            reg_peaks[k.name] = max_peak

        # Compute occupancy per kernel
        occupancies = {}
        for k in chosen_partition.kernels:
            occupancies[k.name] = self.sm_config.occupancy(240, k.threads, k.smem_bytes)

        return UnifiedResult(
            partition=chosen_partition.name,
            kernel_schedules=schedules,
            kernel_warp_assigns=warp_assigns,
            kernel_reg_peaks=reg_peaks,
            kernel_occupancy=occupancies,
            total_makespan=solver.value(total),
            solve_time_ms=solve_ms,
            status=f"II={ii}",
        )
