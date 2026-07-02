"""
Heddle Scheduler (modulo scheduler).

Canonical import path:
  - `from heddle.scheduler.smt import HeddleScheduler, OpNode, ResourceType`

Phase A: Find minimum II with dependency + FU capacity constraints.
Phase B: CP-SAT joint schedule + warp assignment + liveness (incl.
         incoming_live for loop-carried deps) + register/SMEM capacity
         + spill cost + full concurrency constraints + APLSP bound tightening.
"""

from __future__ import annotations

import enum
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ====================================================================== #
# Data types
# ====================================================================== #

class ResourceType(enum.Enum):
    TMA = "TMA"
    TensorCore = "TC"
    ALU = "ALU"
    SFU = "SFU"
    Barrier = "BARRIER"


SFU_ISSUE_CYCLES = 8


class StorageKind(enum.Enum):
    RMEM = "RMEM"
    SMEM = "SMEM"
    TMEM = "TMEM"


class LifetimeSemantic(enum.Enum):
    DEAD_ON_ENTRY = "dead_on_entry"
    DEAD_ON_EXIT = "dead_on_exit"


@dataclass
class OutputValue:
    name: str
    storage: StorageKind = StorageKind.RMEM
    footprint_bytes: int = 0
    spill_cost: int = 0
    lifetime: LifetimeSemantic = LifetimeSemantic.DEAD_ON_ENTRY
    buffer_name: Optional[str] = None

    def smem_buffer_key(self) -> str:
        return self.buffer_name or self.name

    def rmem_buffer_key(self) -> str:
        return self.buffer_name or self.name


@dataclass
class EdgeInfo:
    blocking_sync: bool = False
    delay: Optional[int] = None


@dataclass
class OpNode:
    name: str
    resource_type: ResourceType
    latency: int

    reservation: List[Dict[ResourceType, int]] = field(default_factory=list)

    parents: List["OpNode"] = field(default_factory=list)
    children: List["OpNode"] = field(default_factory=list)

    dependency_distance: Dict[str, int] = field(default_factory=dict)
    edge_info: Dict[str, EdgeInfo] = field(default_factory=dict)

    outputs: List[OutputValue] = field(default_factory=list)

    warp_count: int = 1
    warp_align: int = 1
    replicable: bool = False
    is_varialble_latency : bool = False

    def add_dependency(self, parent: "OpNode", distance: int = 0,
                       blocking_sync: bool = False,
                       delay: Optional[int] = None):
        self.parents.append(parent)
        parent.children.append(self)
        self.dependency_distance[parent.name] = distance
        self.edge_info[parent.name] = EdgeInfo(
            blocking_sync=blocking_sync,
            delay=delay,
        )


# ====================================================================== #
# APLSP (All-Pairs Longest Simple Paths) for bound tightening
# ====================================================================== #

def _compute_aplsp(nodes: List[OpNode], idx: Dict[str, int], ii: int
                   ) -> Dict[Tuple[int, int], int]:
    """Compute all-pairs longest effective delay for the given II.

    For edge u->v with delay d and iteration distance delta:
        effective_delay = d - delta * ii

    Uses Floyd–Warshall on effective delays (max instead of min).
    Returns dict {(u_idx, v_idx): longest_path_length}.
    """
    N = len(nodes)
    NEG_INF = -(10**9)
    dist = [[NEG_INF] * N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0

    for v_node in nodes:
        vi = idx[v_node.name]
        for p in v_node.parents:
            ui = idx[p.name]
            delta = int(v_node.dependency_distance.get(p.name, 0))
            delay = int(p.latency)
            eff = delay - delta * ii
            if eff > dist[ui][vi]:
                dist[ui][vi] = eff

    for k in range(N):
        for i in range(N):
            if dist[i][k] == NEG_INF:
                continue
            for j in range(N):
                if dist[k][j] == NEG_INF:
                    continue
                cand = dist[i][k] + dist[k][j]
                if cand > dist[i][j]:
                    dist[i][j] = cand

    result: Dict[Tuple[int, int], int] = {}
    for i in range(N):
        for j in range(N):
            if dist[i][j] > NEG_INF and i != j:
                result[(i, j)] = dist[i][j]
    return result


# ====================================================================== #
# Scheduler
# ====================================================================== #

class HeddleScheduler:
    """Modulo scheduler for Hopper pipelines."""

    def __init__(
        self,
        nodes: List[OpNode],
        *,
        fu_caps: Optional[Dict[ResourceType, int]] = None,
        reg_limit: int = 240,
        smem_limit: int = 233472,
        num_warps: int = 1,
        timeout_ms: int = 30000,
        disallow_spills: bool = False,
        use_spill_concurrency: bool = True,
        include_incoming_live: bool = True,
        enable_liveness: bool = True,
        start_hints: Optional[Dict[str, int]] = None,
        smem_allocations: Optional[Dict[str, int]] = None,
    ):
        self.nodes = nodes
        self.fu_caps = fu_caps or {
            ResourceType.TMA: 1, ResourceType.TensorCore: 1,
            ResourceType.ALU: 1, ResourceType.SFU: 1,
            ResourceType.Barrier: 1,
        }
        self.reg_limit = reg_limit
        self.smem_limit = smem_limit
        self.num_warps = num_warps
        self.timeout_ms = timeout_ms
        self.disallow_spills = disallow_spills
        self.use_spill_concurrency = use_spill_concurrency
        self.include_incoming_live = include_incoming_live
        self.enable_liveness = enable_liveness
        self.start_hints = start_hints or {}
        self.smem_allocations = dict(smem_allocations or {})

    # ------------------------------------------------------------------ #
    # Phase A  (unchanged API)
    # ------------------------------------------------------------------ #

    def schedule(self, *, min_ii: int = 1, max_ii: int = 10,
                 optimize: bool = False) -> Optional[Dict[str, int]]:
        for ii in range(min_ii, max_ii + 1):
            sol = self._solve_phase_a(ii, optimize=optimize)
            if sol is not None:
                return sol
        return None

    # ------------------------------------------------------------------ #
    # Phase B
    # ------------------------------------------------------------------ #

    def schedule_joint(self, *, min_ii: int = 1, max_ii: int = 10,
                       window: Optional[int] = None,
                       max_window: Optional[int] = None,
                       window_step: Optional[int] = None,
                       optimize: bool = True,
                       ) -> Optional[Dict[str, object]]:
        for ii in range(min_ii, max_ii + 1):
            if window is not None:
                windows = [max(int(window), ii)]
            else:
                start_L = max(ii * 3, ii)
                end_L = max_window if max_window is not None else max(ii * 8, start_L)
                # Guarantee start_L <= end_L so range() is never empty.
                # When max_window < ii * 3 (caller budget tighter than minimum
                # useful window), still try at start_L; the solver will return
                # UNSAT quickly if the window is too small.
                end_L = max(end_L, start_L)
                step_L = max(int(window_step or ii), 1)
                windows = list(range(start_L, end_L + 1, step_L))
                if windows[-1] != end_L:
                    windows.append(end_L)

            for L in windows:
                sol = self._solve_phase_b(ii, L, optimize=optimize)
                if sol is not None:
                    return sol
        return None

    # ================================================================== #
    # Phase A
    # ================================================================== #

    def _solve_phase_a(self, ii: int, *, optimize: bool) -> Optional[Dict[str, int]]:
        # Pure-Python ASAP modulo scheduler — no Z3.
        #
        # Motivation: An ASAP longest-path solver is sufficient for Phase A's
        # purpose (finding any valid schedule for ordering), and keeps this
        # preliminary feasibility pass independent from the CP-SAT model below.
        #
        # Algorithm: Bellman-Ford longest-path to compute ASAP times.
        #   T[v] = max over parents p of (T[p] + latency[p] - delta[v,p] * ii)
        # Check feasibility: ensure no T[v] < required by any constraint.
        # Resource check: verify FU capacity at each slot (mod ii).
        self._ensure_reservations()
        N = len(self.nodes)
        if N == 0:
            return {}
        idx = {n.name: i for i, n in enumerate(self.nodes)}

        # ASAP pass: propagate T values until stable (Bellman-Ford, N rounds)
        T: list[int] = [0] * N
        for _ in range(N):
            changed = False
            for v in self.nodes:
                vi = idx[v.name]
                for par in v.parents:
                    ui = idx[par.name]
                    delta = int(v.dependency_distance.get(par.name, 0))
                    lat = int(par.latency)
                    req = T[ui] + lat - delta * ii
                    if req > T[vi]:
                        T[vi] = req
                        changed = True
            if not changed:
                break

        # Check for negative-cycle (positive cycle in negated graph → infeasible)
        for v in self.nodes:
            vi = idx[v.name]
            for par in v.parents:
                ui = idx[par.name]
                delta = int(v.dependency_distance.get(par.name, 0))
                lat = int(par.latency)
                if T[vi] - T[ui] < lat - delta * ii:
                    return None  # infeasible for this ii

        # Resource-feasibility check: FU capacity at each slot modulo ii
        expanded = self._fold_reservations(ii)
        slot_usage: dict[tuple[int, int], int] = {}  # (slot, resource) -> count
        for i in range(N):
            slot = T[i] % ii
            for l in range(ii):
                for r, c in expanded[i][l].items():
                    effective_slot = (slot + l) % ii
                    key = (effective_slot, r)
                    slot_usage[key] = slot_usage.get(key, 0) + int(c)
        for (slot, r), usage in slot_usage.items():
            cap = int(self.fu_caps.get(r, 1))
            if usage > cap:
                return None  # infeasible for this ii

        return {self.nodes[i].name: T[i] for i in range(N)}

    # ================================================================== #
    # Phase B：联合求解启动时间、warp 分配、活跃区间和资源约束。
    # P1 表示跨迭代 incoming_live，P2 表示并发/阻塞约束，
    # P3 表示 APLSP 派生的时间下界。
    # ================================================================== #

    def _solve_phase_b(self, ii: int, L: int, *, optimize: bool = True) -> Optional[Dict[str, object]]:
        from ortools.sat.python import cp_model
        print(f"----- HeddleSCheduler: {self.reg_limit=} bytes, {self.smem_limit=} bytes, {self.num_warps=}")
        self._ensure_reservations()
        N = len(self.nodes)
        idx = {n.name: i for i, n in enumerate(self.nodes)}
        if N == 0:
            return {"ii": ii, "window": L, "schedule": {}, "warp_assign": {}, "reg_peak": {}}

        W = max(self.num_warps, 1)
        model = cp_model.CpModel()

        def _false_var(name: str):
            b = model.new_bool_var(name)
            model.add(b == 0)
            return b

        def _true_var(name: str):
            b = model.new_bool_var(name)
            model.add(b == 1)
            return b

        def _or_var(name: str, lits):  # 等价性约束 : b True <=> lit中最少有一个为True
            lits = list(lits)
            if not lits:
                return _false_var(name)
            b = model.new_bool_var(name)
            model.add_bool_or(lits).only_enforce_if(b)  # b True => lit中最少有一个为True
            for lit in lits:
                model.add_implication(lit, b)  #  lit中有一个为True => b True
            return b

        def _and_var(name: str, lits): # b True <=> lit中都为True, lit中有一个False b FAlse
            lits = list(lits)
            if not lits:
                return _true_var(name)
            b = model.new_bool_var(name)
            for lit in lits:
                model.add_implication(b, lit)
            model.add_bool_or([lit.negated() for lit in lits] + [b])
            return b

        le_cache = {}

        def _start_le_var(v: int, bound: int):
            # Reified form of Tv[v] <= bound. This replaces large prefix ORs
            # over op[(v, 0..bound)] with two linear half-reifications.
            key = (v, bound)
            if key in le_cache:
                return le_cache[key]
            if bound < 0:
                b = _false_var(f"start_le_false_v={v}_b={bound}")
            elif bound >= L - 1:
                b = _true_var(f"start_le_true_v={v}_b={bound}")
            else:
                b = model.new_bool_var(f"start_le_v={v}_b={bound}")
                model.add(Tv[v] <= bound).only_enforce_if(b)
                model.add(Tv[v] >= bound + 1).only_enforce_if(b.negated())
            le_cache[key] = b
            return b

        # ---- P3：APLSP 时间边界收紧 --------------------------------------
        aplsp = _compute_aplsp(self.nodes, idx, ii)

        # P4 的对称性破除会在变量定义之后添加。

        # ---- 决策变量 ------------------------------------------------------
        Tv = [model.new_int_var(0, L - 1, f"T_v={v}") for v in range(N)]
        warp = {}
        for v in range(N):
            for w in range(W):
                warp[(v, w)] = model.new_bool_var(f"warp_v={v}_w={w}")
        
        # 特殊标记：是否是 variable latency 操作（如 TMA）。
        is_varialble_latency_op = [
            bool(self.nodes[v].is_varialble_latency)
            for v in range(N)
        ]

        # ---- warp 分配（支持多 warp op 和可复制 op） -----------------------
        issue_warp = {}
        for v in range(N):
            nd = self.nodes[v]
            wc = max(nd.warp_count, 1)
            if nd.replicable or wc == 1:
                model.add_exactly_one(warp[(v, w)] for w in range(W))
                for w in range(W):
                    issue_warp[(v, w)] = warp[(v, w)]
            else:
                if wc > W:
                    return None

                # 多 warp op 需要占用一段连续 warp。不能直接写
                # warp[v,w] -> warp[v,w+1..]，因为连续块内部的每个
                # selected warp 都会再次触发蕴含，导致约束向后级联。
                # 因此这里显式引入“连续块起点”变量，再由起点决定
                # 每个 warp 是否落在这段长度为 wc 的区间里。
                align = max(int(nd.warp_align), 1)
                start_slots = [
                    s for s in range(W - wc + 1)
                    if s % align == 0
                ]
                if not start_slots:
                    return None

                starts = {
                    s: model.new_bool_var(f"warp_start_v={v}_w={s}")
                    for s in start_slots
                }
                model.add_exactly_one(starts.values())  # op的 start warp唯一
                for w in range(W):
                    issue_warp[(v, w)] = starts.get(
                        w, _false_var(f"issue_warp_false_v={v}_w={w}")
                    )
                for w in range(W):
                    covering_starts = [
                        starts[s]
                        for s in start_slots
                        if s <= w < s + wc
                    ]
                    if covering_starts:
                        model.add_max_equality(warp[(v, w)], covering_starts)
                    else:
                        model.add(warp[(v, w)] == 0)

        # 对于 variable latency op，应将它们放到同一个 warpgroup 内。
        # 这里按 Hopper warpgroup 语义建模：warpId // 4 相同。
        variable_latency_ops = [
            v for v, is_variable in enumerate(is_varialble_latency_op)
            if is_variable
        ]
        non_variable_latency_ops = [
            v for v, is_variable in enumerate(is_varialble_latency_op)
            if not is_variable
        ]
        for i, u in enumerate(variable_latency_ops):
            for v in variable_latency_ops[i + 1:]:
                for wu in range(W):
                    for wv in range(W):
                        if wu // 4 != wv // 4:  # 如果两个 wgid (wu wv)不同，那么 u分到wu 和 v分到wv 不能同时发生 （可行域添加）
                            model.add_bool_or([
                                warp[(u, wu)].negated(),
                                warp[(v, wv)].negated(),
                            ])

        # variable latency op（如 TMA）需要和其他op分配到不同 warpgroup。

        for u in variable_latency_ops:
            for v in non_variable_latency_ops:
                for wu in range(W):
                    for wv in range(W):
                        if wu // 4 == wv // 4:  # 如果 wu wv wgid 相同，u,v 其中一个为varLat op 则可行域为： not((u分到wu) and (v 分到wv)) 
                            model.add_bool_or([
                                warp[(u, wu)].negated(),
                                warp[(v, wv)].negated(),
                            ])
        
        for v, node in enumerate(self.nodes):
            hint_t = self.start_hints.get(node.name)
            if hint_t is None:
                continue
            hint_t = int(hint_t)
            if hint_t < 0 or hint_t >= L:
                continue
            model.add_hint(Tv[v], hint_t)

        # CP-SAT 的 interval/no_overlap 比手工枚举所有 (time, other-op)
        # 冲突子句紧凑得多。这里按需缓存“op v 在 warp w 上执行”的可选区间，
        # 后续 blocking_sync / spill 并发约束都复用它们。
        max_latency = max((max(int(n.latency), 1) for n in self.nodes), default=1)
        interval_end_max = L - 1 + max_latency
        op_intervals: dict[tuple[int, int], object] = {}

        def _op_interval(v: int, w: int):
            key = (v, w)
            if key not in op_intervals:
                size = max(int(self.nodes[v].latency), 1)
                end = model.new_int_var(0, interval_end_max, f"op_end_v={v}_w={w}")
                model.add(end == Tv[v] + size)
                op_intervals[key] = model.new_optional_interval_var(
                    Tv[v], size, end, warp[(v, w)], f"op_iv_v={v}_w={w}"
                )
            return op_intervals[key]

        # debug : 去掉 APLSP 的时间下界收紧
        # # ---- P3：由 APLSP 推导出的时间下界 -------------------------------
        # for (u, v), d in aplsp.items():
        #     if d > 0:
        #         solver.add(Tv[v] - Tv[u] >= d)

        # # ---- P4：对独立同类节点做对称性破除 -------------------------------
        # # 如果两个节点没有依赖关系，且 FU 类型和 latency 完全相同，
        # # 则强制前者不晚于后者启动，减少等价调度带来的搜索空间。
        # dep_pairs = set(aplsp.keys())
        # for v1 in range(N):
        #     for v2 in range(v1 + 1, N):
        #         n1, n2 = self.nodes[v1], self.nodes[v2]
        #         if (n1.resource_type == n2.resource_type and n1.latency == n2.latency
        #                 and (v1, v2) not in dep_pairs and (v2, v1) not in dep_pairs):
        #             solver.add(Tv[v1] <= Tv[v2])

        # ---- 依赖、跨 warp spill 代价和 blocking sync ---------------------
        for v_node in self.nodes:
            vi = idx[v_node.name]
            for par in v_node.parents:
                ui = idx[par.name]
                # uv的跨迭代依赖距离
                delta = int(v_node.dependency_distance.get(par.name, 0))
                edge = v_node.edge_info.get(par.name)
                # uv延迟
                base_delay = int(edge.delay) if edge and edge.delay is not None else int(par.latency)
                # spill 代价
                spill_cost = max((o.spill_cost for o in par.outputs), default=0)
                # 如果 op有spill代价 && 可用W不止一个
                if spill_cost > 0 and W > 1:
                    same_pairs = [
                        _and_var(f"same_pair_u={ui}_v={vi}_w={w}", [warp[(ui, w)], warp[(vi, w)]])
                        for w in range(W)
                    ]
                    same_w = _or_var(f"same_w_u={ui}_v={vi}", same_pairs)
                    if self.disallow_spills:
                        # 禁用spill时， uv只能在同warp内
                        model.add(same_w == 1)
                        model.add(Tv[vi] - Tv[ui] >= base_delay - delta * ii)
                    else:
                        # 若 uv在同warp，正常计算 Tv[vi] - Tv[ui] 启动间隔
                        model.add(Tv[vi] - Tv[ui] >= base_delay - delta * ii).only_enforce_if(same_w)
                        # 否则 启动间隔需要考虑 spill代价
                        model.add(
                            Tv[vi] - Tv[ui] >= base_delay + spill_cost - delta * ii
                        ).only_enforce_if(same_w.negated())
                else:
                    model.add(Tv[vi] - Tv[ui] >= base_delay - delta * ii)

                # P2：blocking sync 的完整并发约束。
                # 这种边不仅要求时间顺序，还会约束相关 warp 的覆盖关系；
                # 在同步阻塞窗口内，同一个 warp 上不能安排其他重叠 op。
                if edge and edge.blocking_sync:
                    if W > 1:
                        for w in range(W):
                            model.add_implication(warp[(ui, w)], warp[(vi, w)])

                    # v 在 t 启动时，u 的阻塞同步窗口近似为
                    # [t - base_delay, t)。用 NoOverlap 表达“同一 warp
                    # 上其他 op 不能与该窗口重叠”，避免按每个 t/to 展开
                    # 成海量 BoolOr。
                    block_size = max(base_delay, 1)
                    block_start_min = -block_size
                    block_end_min = 0
                    for w in range(W):
                        block_start = model.new_int_var(
                            block_start_min, L - 1, f"block_start_u={ui}_v={vi}_w={w}"
                        )
                        block_end = model.new_int_var(
                            block_end_min, L - 1, f"block_end_u={ui}_v={vi}_w={w}"
                        )
                        model.add(block_start == Tv[vi] - block_size)
                        model.add(block_end == Tv[vi])
                        block_interval = model.new_optional_interval_var(
                            block_start,
                            block_size,
                            block_end,
                            warp[(vi, w)],
                            f"block_iv_u={ui}_v={vi}_w={w}",
                        )
                        # 注意：NoOverlap 会约束列表中任意两个 interval
                        # 都不重叠。这里需要的是“阻塞窗口 vs 其他 op”的
                        # 星形排斥，而不是把所有 other op 彼此串行化。
                        for other in range(N):
                            if other == ui or other == vi:
                                continue
                            model.add_no_overlap([block_interval, _op_interval(other, w)])

        # P2：spill 并发约束。
        # 当 producer/consumer 分配到不同 warp，且 producer 输出带 spill_cost
        # 时，把 spill 看成占用接收方 warp 的一段时间；这段时间内接收方
        # warp 不能再执行其他 op。
        if self.use_spill_concurrency and W > 1:
            for v_node in self.nodes:
                vi = idx[v_node.name]
                for par in v_node.parents:
                    ui = idx[par.name]
                    sc = max((o.spill_cost for o in par.outputs), default=0)
                    if sc <= 0:
                        continue
                    for w_src in range(W):
                        for w_dst in range(W):
                            if w_src == w_dst:
                                continue
                            spill_present = _and_var(
                                f"spill_present_u={ui}_v={vi}_ws={w_src}_wd={w_dst}",
                                [warp[(ui, w_src)], warp[(vi, w_dst)]],
                            )
                            spill_start = model.new_int_var(
                                -sc, L - 1, f"spill_start_u={ui}_v={vi}_ws={w_src}_wd={w_dst}"
                            )
                            spill_end = model.new_int_var(
                                0, L - 1, f"spill_end_u={ui}_v={vi}_ws={w_src}_wd={w_dst}"
                            )
                            model.add(spill_start == Tv[vi] - sc)
                            model.add(spill_end == Tv[vi])
                            spill_interval = model.new_optional_interval_var(
                                spill_start,
                                sc,
                                spill_end,
                                spill_present,
                                f"spill_iv_u={ui}_v={vi}_ws={w_src}_wd={w_dst}",
                            )
                            # 同理只禁止 spill 窗口和接收方 warp 上的其他 op
                            # 重叠，不约束这些 other op 之间的相互重叠。
                            for other in range(N):
                                if other == vi:
                                    continue
                                model.add_no_overlap([spill_interval, _op_interval(other, w_dst)])

        # ---- FU 容量约束 ---------------------------------------------------
        # 对 cap=1 且 reservation 为连续区间的资源，用相位区间不重叠表达。
        # 这避免 WGMMA 多拍 issue 展开后在 (L * ii * L) 枚举里制造海量
        # Bool/linear 项。其它更复杂的资源形状继续走旧的逐槽容量约束。
        phase = [model.new_int_var(0, ii - 1, f"phase_v={v}") for v in range(N)]
        for v in range(N):
            model.add_modulo_equality(phase[v], Tv[v], ii)

        phase_eq_cache = {}

        def _phase_eq_var(v: int, p: int):
            p = int(p) % ii
            key = (v, p)
            if key in phase_eq_cache:
                return phase_eq_cache[key]
            b = model.new_bool_var(f"phase_eq_v={v}_p={p}")
            model.add(phase[v] == p).only_enforce_if(b)
            model.add(phase[v] != p).only_enforce_if(b.negated())
            phase_eq_cache[key] = b
            return b

        # Hopper 每个 warpgroup 内的同号 warp 共享一个 subcore issue 槽：
        # subcoreId = warpId % 4。同 subcore 上的两个 op 不能在同一个
        # modulo issue interval 内重叠；不同 subcore 的 warp 组合不加限制。
        issue_spans = [
            (v, 0, max(len(node.reservation), 1))
            for v, node in enumerate(self.nodes)
        ]
        for i, (u, off_u, dur_u) in enumerate(issue_spans):
            for v, off_v, dur_v in issue_spans[i + 1:]:
                for wu in range(W):
                    for wv in range(W):
                        if wu % 4 != wv % 4:
                            continue
                        same_subcore = _and_var(
                            f"same_subcore_u={u}_v={v}_wu={wu}_wv={wv}",
                            [issue_warp[(u, wu)], issue_warp[(v, wv)]],
                        )
                        if dur_u + dur_v > ii:
                            model.add(same_subcore == 0)
                            continue
                        delta_uv = model.new_int_var(
                            0, ii - 1,
                            f"subcore_delta_u={u}_v={v}_wu={wu}_wv={wv}",
                        )
                        model.add_modulo_equality(
                            delta_uv,
                            phase[v] + off_v - phase[u] - off_u + ii,
                            ii,
                        )
                        model.add(delta_uv >= dur_u).only_enforce_if(same_subcore)
                        model.add(delta_uv <= ii - dur_v).only_enforce_if(same_subcore)

        interval_mode_resources: set[ResourceType] = set()
        resource_spans: dict[ResourceType, list[tuple[int, int, int]]] = {}
        for r, cap in self.fu_caps.items():
            if int(cap) != 1:
                continue
            spans: list[tuple[int, int, int]] = []
            ok = True
            for v, node in enumerate(self.nodes):
                used_offsets = [
                    j for j, per_cycle in enumerate(node.reservation)
                    if int(per_cycle.get(r, 0)) > 0
                ]
                if not used_offsets:
                    continue
                if any(int(node.reservation[j].get(r, 0)) != 1 for j in used_offsets):
                    ok = False
                    break
                first = min(used_offsets)
                last = max(used_offsets)
                if used_offsets != list(range(first, last + 1)):
                    ok = False
                    break
                span = last - first + 1
                if span > ii:
                    return None
                spans.append((v, first, span))
            if ok and spans:
                interval_mode_resources.add(r)
                resource_spans[r] = spans

        for r, spans in resource_spans.items():
            for i, (u, off_u, dur_u) in enumerate(spans):
                for v, off_v, dur_v in spans[i + 1:]:
                    if dur_u + dur_v > ii:
                        return None
                    delta_uv = model.new_int_var(0, ii - 1, f"fu_delta_{r.value}_u={u}_v={v}")
                    model.add_modulo_equality(
                        delta_uv,
                        phase[v] + off_v - phase[u] - off_u + ii,
                        ii,
                    )
                    model.add(delta_uv >= dur_u)
                    model.add(delta_uv <= ii - dur_v)

        # Barrier issue 槽互斥。
        # wait/try_wait barrier 使用专门的同步 issue 槽：它不能和其它
        # op 的 issue interval 共享同一个 modulo 槽。
        # 这里刻意和 FU 容量分开建模，避免把 barrier 误看成同时消耗
        # TMA/TC/ALU/SFU 资源。
        barrier_spans = resource_spans.get(ResourceType.Barrier, [])
        if barrier_spans:
            full_issue_spans = [
                (v, 0, max(len(node.reservation), 1))
                for v, node in enumerate(self.nodes)
            ]
            barrier_pairs: set[tuple[int, int]] = set()
            for b, off_b, dur_b in barrier_spans:
                for v, off_v, dur_v in full_issue_spans:
                    if v == b:
                        continue
                    u0, off_u0, dur_u0 = (b, off_b, dur_b)
                    v0, off_v0, dur_v0 = (v, off_v, dur_v)
                    key = (u0, v0) if u0 < v0 else (v0, u0)
                    if key in barrier_pairs:
                        continue
                    barrier_pairs.add(key)
                    if dur_u0 + dur_v0 > ii:
                        return None
                    delta_uv = model.new_int_var(
                        0, ii - 1, f"barrier_delta_u={u0}_v={v0}"
                    )
                    model.add_modulo_equality(
                        delta_uv,
                        phase[v0] + off_v0 - phase[u0] - off_u0 + ii,
                        ii,
                    )
                    model.add(delta_uv >= dur_u0)
                    model.add(delta_uv <= ii - dur_v0)

        expanded = self._fold_reservations(ii)
        for r, cap in self.fu_caps.items():
            if r in interval_mode_resources:
                continue
            for q in range(ii):
                terms = []
                for v in range(N):
                    for l in range(ii):
                        c = int(expanded[v][l].get(r, 0))
                        if c:
                            terms.append(c * _phase_eq_var(v, q - l))
                if terms:
                    model.add(sum(terms) <= int(cap))  # 每个 modulo 相位上的 FU 使用量不得超过上限。

        # ---- 活跃区间、P1 incoming_live 和容量约束 ------------------------
        all_outputs: list[tuple[int, OutputValue]] = []  #[(opId, output)]
        for v in range(N):
            for out in self.nodes[v].outputs:
                all_outputs.append((v, out))

        # 检测跨迭代传递的输出：只要某个消费者边的 distance(delta) > 0，
        # 该输出就可能在当前迭代开始时已经来自上一轮迭代并保持 live。
        loop_carried: set[int] = set()
        consumers_of: dict[int, list[tuple[int, int]]] = defaultdict(list)
        output_name_to_xi: dict[str, int] = {}
        for xi, (pv, oval) in enumerate(all_outputs):
            output_name_to_xi[oval.name] = xi  # {outbuffer.name : opid}

        for v_node in self.nodes:
            vi = idx[v_node.name]
            for par in v_node.parents:  # 遍历每个parentOp (自己读写同一buffer这种自依赖 已经在 solve_joint 的准备阶段加入了)
                delta = int(v_node.dependency_distance.get(par.name, 0))
                for oval in par.outputs:
                    xi = output_name_to_xi.get(oval.name)
                    if xi is not None:
                        consumers_of[xi].append((vi, delta))  # {vi : (ui,delta)}
                        if delta > 0:
                            loop_carried.add(xi)  # opvi 有跨迭代依赖

        # RMEM 容量约束是可行性的一部分，feasibility 和 optimize 两个阶段
        # 都必须建模；SMEM footprint 是 CTA 全局容量约束，也独立建模。
        track_liveness = bool(self.enable_liveness and all_outputs and self.reg_limit > 0)
        max_iter_overlap = (L - 1) // max(int(ii), 1)
        iter_offsets = (
            range(-max_iter_overlap, max_iter_overlap + 1)
            if self.include_incoming_live else
            range(0, 1)
        )
        iter_live = {}

        def _iter_live_var(xi: int, iter_offset: int, tau: int):
            key = (xi, iter_offset, tau)
            if key in iter_live:
                return iter_live[key]

            producer_v, oval = all_outputs[xi]
            produced_by = _start_le_var(
                producer_v, tau - iter_offset * ii)

            if oval.lifetime == LifetimeSemantic.DEAD_ON_ENTRY:
                consumers = (
                    consumers_of[xi] if self.include_incoming_live else
                    [(cv, d) for cv, d in consumers_of[xi] if d == 0]
                )
                if not consumers:
                    b = _false_var(
                        f"iter_live_false_x={xi}_k={iter_offset}_t={tau}")
                else:
                    producer_lat = int(self.nodes[producer_v].latency)
                    consumed_terms = []
                    for cv, d in consumers:
                        consume_bound = tau - (iter_offset + d) * ii
                        # Zero-latency values can be produced and consumed
                        # in the same cycle, but they still occupy the
                        # register during that cycle. Treat consumption as
                        # complete only after the consumer's start cycle.
                        if producer_lat == 0:
                            consume_bound -= 1
                        consumed_terms.append(
                            _start_le_var(cv, consume_bound))
                    all_consumed = _and_var(
                        f"iter_consumed_x={xi}_k={iter_offset}_t={tau}",
                        consumed_terms,
                    )
                    b = model.new_bool_var(
                        f"iter_live_x={xi}_k={iter_offset}_t={tau}")
                    model.add_implication(b, produced_by)
                    model.add_implication(b, all_consumed.negated())
                    model.add_bool_or([
                        produced_by.negated(),
                        all_consumed,
                        b,
                    ])
            else:
                b = produced_by

            iter_live[key] = b
            return b

        # ---- SMEM 容量约束（全局统计） -------------------------------
        # SMEM footprint 表示 shared buffer allocation 的大小。这个
        # allocation 已经包含 pipeline stage / double-buffer 空间，
        # 不能像 RMEM value 一样按重叠迭代副本重复累加；同一个
        # shared buffer 被多个 stmt 写到时也只按 bufferName 统计一次。
        static_smem_terms = [
            int(footprint)
            for footprint in self.smem_allocations.values()
            if int(footprint) > 0
        ]
        for tau in range(L):
            smem_live_by_buffer: dict[str, list] = defaultdict(list)
            smem_footprint_by_buffer: dict[str, int] = {}
            for xi, (pv, oval) in enumerate(all_outputs):
                if oval.storage != StorageKind.SMEM or oval.footprint_bytes <= 0:
                    continue
                buffer_key = oval.smem_buffer_key()
                if buffer_key in self.smem_allocations:
                    continue
                any_copy_live = _or_var(
                    f"smem_live_x={xi}_t={tau}",
                    [_iter_live_var(xi, iter_offset, tau)
                        for iter_offset in iter_offsets],
                )
                smem_live_by_buffer[buffer_key].append(any_copy_live)
                smem_footprint_by_buffer[buffer_key] = max(
                    smem_footprint_by_buffer.get(buffer_key, 0),
                    int(oval.footprint_bytes),
                )
            smem_terms = []
            for buffer_key, live_terms in smem_live_by_buffer.items():
                buffer_live = _or_var(
                    f"smem_live_buf={buffer_key}_t={tau}",
                    live_terms,
                )
                smem_terms.append(
                    smem_footprint_by_buffer[buffer_key] * buffer_live)
            smem_total_terms = static_smem_terms + smem_terms
            if smem_total_terms:
                model.add(sum(smem_total_terms) <= self.smem_limit)

        if track_liveness:
            # live[xi, tau]：第 xi 个输出在第 0 轮迭代的 tau 时刻是否 live。
            live = {}
            for xi in range(len(all_outputs)):
                for tau in range(L):
                    live[(xi, tau)] = model.new_bool_var(f"live_x={xi}_t={tau}")

            # P1：incoming_live[xi, tau] 表示跨迭代值 xi 是否在 tau 时刻
            # 仍然由“上一轮迭代”带入并保持 live，用于统计跨迭代寄存器压力。
            incoming_live = {}
            if self.include_incoming_live:
                for xi in loop_carried:
                    for tau in range(L):
                        incoming_live[(xi, tau)] = model.new_bool_var(f"ilive_x={xi}_t={tau}")

            # ---- 活跃区间约束 ---------------------------------------------
            for xi, (producer_v, oval) in enumerate(all_outputs):
                same_iter_consumers = [(cv, d) for cv, d in consumers_of[xi] if d == 0]
                producer_lat = int(self.nodes[producer_v].latency)

                for tau in range(L):
                    produced_by = _start_le_var(producer_v, tau)  # producer_v 是否在 tau时刻前启动

                    if oval.lifetime == LifetimeSemantic.DEAD_ON_ENTRY:
                        if same_iter_consumers:
                            # 对 zero-latency producer，consumer 可能和 producer
                            # 在同一个时间步启动。该 cycle 内输出仍占寄存器，
                            # 因此判断“是否已消费完”时必须严格早于 tau。
                            # 这个语义对应 Twill warpspecialization.rs L371-409。
                            if producer_lat == 0:
                                all_consumed = _and_var(f"consumed_x={xi}_t={tau}", [
                                    _start_le_var(cv, tau - 1)
                                    for cv, _ in same_iter_consumers
                                ]) if tau > 1 else _false_var(f"consumed_false_x={xi}_t={tau}")
                            else:
                                all_consumed = _and_var(f"consumed_x={xi}_t={tau}", [
                                    _start_le_var(cv, tau)
                                    for cv, _ in same_iter_consumers
                                ]) if tau > 0 else _false_var(f"consumed_false_x={xi}_t={tau}")
                            # 约束 1：如果 producer 还没启动 (produced_by=0)，则数据绝对不可能 live
                            model.add_bool_or([
                                live[(xi, tau)].negated(), 
                                produced_by,  # producer_v 在 tau时刻后启动 与 xi在tau时刻 live 不可能同True
                            ])
                            # 约束 2：如果所有消费者已经消费完 (all_consumed=1)，则数据绝对不可能 live
                            model.add_bool_or([
                                live[(xi, tau)].negated(),  # 可行域为： not (live[xi,tau] && 消费者在tau时刻前启动 )
                                all_consumed.negated(),
                            ])
                            # 约束 3：如果 producer 已经启动，且消费者还没消费完，则数据【必须】是 live 的
                            model.add_bool_or([
                                produced_by.negated(),
                                all_consumed,
                                live[(xi, tau)],
                            ])
                        else:
                            # 无sameiterconsumer, live可以判False
                            model.add(live[(xi, tau)] == 0)
                    else:
                        model.add(live[(xi, tau)] == produced_by)

                # P1：为跨迭代值建立 incoming_live。
                if self.include_incoming_live and xi in loop_carried:
                    cross_iter_consumers = [(cv, d) for cv, d in consumers_of[xi] if d > 0]
                    for tau in range(L):
                        if cross_iter_consumers:
                            # 如果第 0 轮迭代中仍有消费者在 tau 时刻前尚未启动，
                            # 那么来自上一轮的值在 tau 时刻仍需要保持 live。
                            some_consumer_pending = _or_var(f"pending_x={xi}_t={tau}", [
                                _start_le_var(cv, tau).negated()
                                for cv, _ in cross_iter_consumers
                            ]) if tau > 0 else _true_var(f"pending_true_x={xi}_t={tau}")
                            model.add(incoming_live[(xi, tau)] == some_consumer_pending)
                        else:
                            model.add(incoming_live[(xi, tau)] == 0)

            # ---- 每个 warp、每个时间步的寄存器容量约束 --------------------
            for w in range(W):
                for tau in range(L):
                    rmem_live_by_copy: dict[tuple[str, int], list] = defaultdict(list)
                    rmem_footprint_by_copy: dict[tuple[str, int], int] = {}
                    for xi, (pv, oval) in enumerate(all_outputs):
                        if oval.storage != StorageKind.RMEM or oval.footprint_bytes <= 0:
                            continue
                        buffer_key = oval.rmem_buffer_key()
                        rmem_iter_offsets = (
                            iter_offsets
                            if oval.lifetime == LifetimeSemantic.DEAD_ON_ENTRY else
                            range(0, 1)
                        )
                        for iter_offset in rmem_iter_offsets:
                            copy_key = (buffer_key, int(iter_offset))
                            copy_live = _iter_live_var(
                                xi, iter_offset, tau)
                            live_on_warp = _and_var(
                                f"live_on_warp_x={xi}_k={iter_offset}_w={w}_t={tau}",
                                [warp[(pv, w)], copy_live],
                            )
                            rmem_live_by_copy[copy_key].append(live_on_warp)
                            rmem_footprint_by_copy[copy_key] = max(
                                rmem_footprint_by_copy.get(copy_key, 0),
                                int(oval.footprint_bytes),
                            )
                    rmem_terms = []
                    for (buffer_key, iter_offset), live_terms in rmem_live_by_copy.items():
                        buffer_live = _or_var(
                            f"rmem_live_buf={buffer_key}_k={iter_offset}_w={w}_t={tau}",
                            live_terms,
                        )
                        rmem_terms.append(
                            rmem_footprint_by_copy[(buffer_key, iter_offset)] * buffer_live)
                    if rmem_terms:
                        model.add(sum(rmem_terms) <= self.reg_limit)

        # ---- 优化目标：偏好更紧凑的调度 -------------------------------
        if optimize:
            end_times = []
            for v, node in enumerate(self.nodes):
                end_v = model.new_int_var(0, L - 1 + max(len(node.reservation), 1), f"end_T_v={v}")
                model.add(end_v == Tv[v] + max(len(node.reservation), 1))
                end_times.append(end_v)
            mx = model.new_int_var(0, L - 1 + max((max(len(n.reservation), 1) for n in self.nodes), default=1), "max_end_T")
            model.add_max_equality(mx, end_times)
            model.minimize(L * N * mx + sum(Tv))

        # ---- 求解 ---------------------------------------------------------
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(float(self.timeout_ms) / 1000.0, 1)
        solver.parameters.num_workers = 4
        solver.parameters.log_search_progress=False
        
        
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        schedule = {
            self.nodes[v].name: int(solver.value(Tv[v]))
            for v in range(N)
        }

        warp_assign = {}
        for v in range(N):
            for w in range(W):
                if solver.value(warp[(v, w)]):
                    warp_assign[self.nodes[v].name] = w
                    break

        reg_peak: Dict[int, int] = {}  # warp : 寄存器用量最大值
        if track_liveness:
            for w in range(W):
                peak = 0
                for tau in range(L):
                    live_bytes_by_copy: Dict[tuple[str, int], int] = {}
                    for xi, (pv, oval) in enumerate(all_outputs):
                        if oval.storage != StorageKind.RMEM:
                            continue
                        owned = bool(solver.value(warp[(pv, w)]))
                        if not owned:
                            continue
                        buffer_key = oval.rmem_buffer_key()
                        rmem_iter_offsets = (
                            iter_offsets
                            if oval.lifetime == LifetimeSemantic.DEAD_ON_ENTRY else
                            range(0, 1)
                        )
                        for iter_offset in rmem_iter_offsets:
                            live_var = iter_live.get((xi, iter_offset, tau))
                            if live_var is not None and bool(solver.value(live_var)):
                                copy_key = (buffer_key, int(iter_offset))
                                live_bytes_by_copy[copy_key] = max(
                                    live_bytes_by_copy.get(copy_key, 0),
                                    int(oval.footprint_bytes),
                                )
                    total = sum(live_bytes_by_copy.values())
                    peak = max(peak, total)
                reg_peak[w] = peak

        return {
            "ii": ii,
            "window": L,
            "schedule": schedule,
            "warp_assign": warp_assign,
            "reg_peak": reg_peak,
        }

    # ================================================================== #
    # Helpers
    # ================================================================== #

    def _ensure_reservations(self):
        for n in self.nodes:
            if not n.reservation:
                issue_cycles = (
                    SFU_ISSUE_CYCLES
                    if n.resource_type is ResourceType.SFU
                    else max(int(n.latency), 0)
                )
                n.reservation = [{n.resource_type: 1} for _ in range(issue_cycles)]  # [{ALU: 1}, {ALU: 1}, {ALU: 1}]

    def _fold_reservations(self, ii: int) -> list[dict[int, dict[ResourceType, int]]]:
        expanded: list[dict[int, dict[ResourceType, int]]] = []
        for n in self.nodes:
            # 1. 为当前节点初始化一个大小为 ii 的空折叠表（周期从 0 到 ii-1）
            tbl: dict[int, dict[ResourceType, int]] = {l: {} for l in range(ii)}
            # 2. 遍历该节点在原始时间线（j）上的资源占用
            for j, per_cycle in enumerate(n.reservation):
                l = j % ii
                # 3. 将资源合并到对应的模周期槽位中
                #  假如 原始占用：[{ALU: 1}, {ALU: 1}, {ALU: 1}] ， ii=2时，则折叠为： [{ALU: 2}, {ALU: 1}] 
                for r, cnt in per_cycle.items():
                    tbl[l][r] = int(tbl[l].get(r, 0)) + int(cnt)  
            expanded.append(tbl)
        return expanded
