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
        reg_limit: int = 240*4,  # per thread f32 register counts * 4bytes per f32
        smem_limit: int = 233472,  # per block bytes
        num_warps: int = 1,
        timeout_ms: int = 30000,
        disallow_spills: bool = False,
        use_spill_concurrency: bool = True,
        include_incoming_live: bool = True,
        enable_liveness: bool = True,
        enable_rmem_liveness: Optional[bool] = None,
        enable_smem_liveness: Optional[bool] = None,
        fold_rmem_liveness_by_ii: bool = True,
        liveness_checkpoint_step: int = 8,
        start_hints: Optional[Dict[str, int]] = None,
        smem_allocations: Optional[Dict[str, int]] = None,
        log_search_progress : bool = False,
        same_warpgroup_pairs: Optional[List[Tuple[str, str]]] = None,
        not_all_same_warpgroup_sets: Optional[List[Tuple[str, ...]]] = None,
        same_subcore_exclusion_pairs: Optional[List[Tuple[str, str]]] = None,
        cross_wg_rmem_penalty: int = 0,
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
        self.enable_rmem_liveness = (
            bool(enable_liveness)
            if enable_rmem_liveness is None else
            bool(enable_rmem_liveness)
        )
        self.enable_smem_liveness = (
            bool(enable_liveness)
            if enable_smem_liveness is None else
            bool(enable_smem_liveness)
        )
        self.fold_rmem_liveness_by_ii = bool(fold_rmem_liveness_by_ii)
        self.liveness_checkpoint_step = max(int(liveness_checkpoint_step), 1)
        self.start_hints = start_hints or {}
        self.smem_allocations = dict(smem_allocations or {})
        self.log_search_progress = log_search_progress
        self.same_warpgroup_pairs = list(same_warpgroup_pairs or [])
        self.not_all_same_warpgroup_sets = list(not_all_same_warpgroup_sets or [])
        self.same_subcore_exclusion_pairs = list(same_subcore_exclusion_pairs or [])
        self.cross_wg_rmem_penalty = max(int(cross_wg_rmem_penalty), 0)

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
        print(f"----- HeddleSCheduler: {self.reg_limit=} bytes, {self.smem_limit=} bytes, {self.num_warps=}, {optimize=}")
        self._ensure_reservations()
        N = len(self.nodes)
        idx = {n.name: i for i, n in enumerate(self.nodes)}
        if N == 0:
            return {
                "ii": ii,
                "window": L,
                "schedule": {},
                "warp_assign": {},
                "reg_peak": {},
                "variable_lifetimes": {},
            }

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
        # aplsp = _compute_aplsp(self.nodes, idx, ii)

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
                    print(
                        f"----- HeddleScheduler early return: reason=warp_count_gt_domain "
                        f"node={nd.name} warp_count={wc} num_warps={W} ii={ii} L={L}",
                        flush=True,
                    )
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
                    print(
                        f"----- HeddleScheduler early return: reason=no_warp_start_slots "
                        f"node={nd.name} warp_count={wc} warp_align={align} "
                        f"num_warps={W} ii={ii} L={L}",
                        flush=True,
                    )
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

        # Explicit semantic constraints from the frontend.  Barrier wait nodes
        # that protect a shared-memory value must execute in the same raw
        # warpgroup as the consumer that reads that value; otherwise PCWS can
        # place the wait in one WG and the shared-memory use in another WG.
        for left_name, right_name in self.same_warpgroup_pairs:
            if left_name not in idx or right_name not in idx:
                continue
            u = idx[left_name]
            v = idx[right_name]
            for wu in range(W):
                for wv in range(W):
                    if wu // 4 != wv // 4:
                        model.add_bool_or([
                            warp[(u, wu)].negated(),
                            warp[(v, wv)].negated(),
                        ])

        # Lightweight register-pressure feedback: after an external fixed
        # liveness check finds that several peak producers overflow one
        # warpgroup, callers can require that this small set is not all placed
        # in the same raw warpgroup. This avoids rebuilding the full RMEM
        # liveness model in CP-SAT.
        for feedback_idx, raw_names in enumerate(self.not_all_same_warpgroup_sets):
            names = [name for name in raw_names if name in idx]
            if len(names) < 2:
                continue
            producer_ids = [idx[name] for name in names]
            for wg in range((W + 3) // 4):
                group_warps = [w for w in range(wg * 4, min((wg + 1) * 4, W))]
                if not group_warps:
                    continue
                in_group_terms = []
                for v in producer_ids:
                    in_group_terms.append(
                        _or_var(
                            f"feedback_in_wg_set={feedback_idx}_v={v}_wg={wg}",
                            [warp[(v, w)] for w in group_warps],
                        )
                    )
                model.add_bool_or([term.negated() for term in in_group_terms])
        
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

        cross_wg_rmem_penalty_terms = []

        def _node_in_warpgroup(v: int, wg: int):
            group_warps = [w for w in range(wg * 4, min((wg + 1) * 4, W))]
            if not group_warps:
                return _false_var(f"in_wg_empty_v={v}_wg={wg}")
            return _or_var(
                f"in_wg_v={v}_wg={wg}",
                [warp[(v, w)] for w in group_warps],
            )

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
                has_rmem_spill_output = any(
                    o.storage == StorageKind.RMEM and o.footprint_bytes > 0 and o.spill_cost > 0
                    for o in par.outputs
                )
                if has_rmem_spill_output and self.cross_wg_rmem_penalty > 0 and W > 1:
                    same_wg_terms = []
                    for wg in range((W + 3) // 4):
                        same_wg_terms.append(
                            _and_var(
                                f"same_rmem_wg_u={ui}_v={vi}_wg={wg}",
                                [_node_in_warpgroup(ui, wg), _node_in_warpgroup(vi, wg)],
                            )
                        )
                    same_wg = _or_var(f"same_rmem_wg_u={ui}_v={vi}", same_wg_terms)
                    cross_wg_rmem_penalty_terms.append(
                        int(self.cross_wg_rmem_penalty) * int(max(spill_cost, 1)) * same_wg.negated()
                    )
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
                        # 星形排斥，而不是把所有 other op 
                        # 彼此串行化。
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
                            # 同理只禁止 spill 窗口 and 接收方 warp 上的其他 op
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

        # Hopper 每个 warpgroup 内的同号 warp 共享一个 subcore issue 槽。
        # 不在首次求解里对所有 op pair 全局展开；log4 这类 expanded op
        # 数量较大时，这会在进入 CP-SAT 前制造百万级 Python 侧建模开销。
        # 只对 post-check 反馈回来的少量冲突 pair 加同 subcore 排斥。
        seen_subcore_pairs: set[tuple[int, int]] = set()
        for left_name, right_name in self.same_subcore_exclusion_pairs:
            if left_name not in idx or right_name not in idx:
                continue
            u = idx[left_name]
            v = idx[right_name]
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in seen_subcore_pairs:
                continue
            seen_subcore_pairs.add((u, v))
            dur_u = max(len(self.nodes[u].reservation), 1)
            dur_v = max(len(self.nodes[v].reservation), 1)
            for wu in range(W):
                for wv in range(W):
                    if wu % 4 != wv % 4:
                        continue
                    same_subcore = _and_var(
                        f"same_subcore_feedback_u={u}_v={v}_wu={wu}_wv={wv}",
                        [issue_warp[(u, wu)], issue_warp[(v, wv)]],
                    )
                    if dur_u + dur_v > ii:
                        model.add(same_subcore == 0)
                        continue
                    delta_uv = model.new_int_var(
                        0, ii - 1,
                        f"subcore_feedback_delta_u={u}_v={v}_wu={wu}_wv={wv}",
                    )
                    model.add_modulo_equality(
                        delta_uv,
                        phase[v] - phase[u] + ii,
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
                    print(
                        f"----- HeddleScheduler early return: reason=resource_span_gt_ii "
                        f"resource={r.value} node={node.name} span={span} ii={ii} L={L} "
                        f"used_offsets={used_offsets}",
                        flush=True,
                    )
                    return None
                spans.append((v, first, span))
            if ok and spans:
                interval_mode_resources.add(r)
                resource_spans[r] = spans

        for r, spans in resource_spans.items():
            for i, (u, off_u, dur_u) in enumerate(spans):
                for v, off_v, dur_v in spans[i + 1:]:
                    if dur_u + dur_v > ii:
                        print(
                            f"----- HeddleScheduler early return: reason=resource_pair_span_gt_ii "
                            f"resource={r.value} left={self.nodes[u].name} right={self.nodes[v].name} "
                            f"dur_left={dur_u} dur_right={dur_v} ii={ii} L={L}",
                            flush=True,
                        )
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
        # 这里刻意和 
        # FU 容量分开建模，避免把 barrier 误看成同时消耗
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
                        print(
                            f"----- HeddleScheduler early return: reason=barrier_pair_span_gt_ii "
                            f"barrier={self.nodes[u0].name} other={self.nodes[v0].name} "
                            f"barrier_dur={dur_u0} other_dur={dur_v0} ii={ii} L={L}",
                            flush=True,
                        )
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

        resource_has_usage: dict[ResourceType, bool] = {
            r: any(
                int(per_cycle.get(r, 0)) > 0
                for node in self.nodes
                for per_cycle in node.reservation
            )
            for r in self.fu_caps
        }
        exact_fallback_resources: list[tuple[ResourceType, int]] = []
        for r, cap in self.fu_caps.items():
            if r in interval_mode_resources:
                continue
            if not resource_has_usage.get(r, False):
                continue
            exact_fallback_resources.append((r, int(cap)))
        for r, cap in exact_fallback_resources:
            usage_by_node: list[list[tuple[int, int]]] = []
            max_possible_usage = 0
            for node in self.nodes:
                usage: dict[int, int] = {}
                for off, per_cycle in enumerate(node.reservation):
                    c = int(per_cycle.get(r, 0))
                    if c:
                        folded_off = int(off) % ii
                        usage[folded_off] = usage.get(folded_off, 0) + c
                node_usage = sorted(usage.items())
                usage_by_node.append(node_usage)
                if node_usage:
                    max_possible_usage += max(c for _, c in node_usage)
            if max_possible_usage <= int(cap):
                continue
            for q in range(ii):
                terms = []
                for v, node_usage in enumerate(usage_by_node):
                    for off, c in node_usage:
                        terms.append(c * _phase_eq_var(v, q - off))
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
            for par in v_node.parents:  # 遍历每个parentOp (自己读写同一buffer这种自依赖 已经在 
                # solve_joint 的准备阶段加入了)
                delta = int(v_node.dependency_distance.get(par.name, 0))
                for oval in par.outputs:
                    xi = output_name_to_xi.get(oval.name)
                    if xi is not None:
                        consumers_of[xi].append((vi, delta))  # {vi : (ui,delta)}
                        if delta > 0:
                            loop_carried.add(xi)  # opvi 有跨迭代依赖

        # RMEM / SMEM 动态 liveness 分开控制。主 joint path 可以只打开
        # sparse RMEM 容量约束，避免同时引入逐 tau 的 SMEM live 网格。
        track_rmem_liveness = bool(self.enable_rmem_liveness and all_outputs and self.reg_limit > 0)
        track_smem_liveness = bool(self.enable_smem_liveness and all_outputs)
        max_iter_overlap = (L - 1) // max(int(ii), 1)
        max_dependency_distance = max(
            (
                int(distance)
                for consumers in consumers_of.values()
                for _, distance in consumers
            ),
            default=0,
        )
        # A loop-carried dependency with distance=1 contributes the previous
        # iteration's value to the current window even when I == L.
        incoming_iter_overlap = max(max_iter_overlap, max_dependency_distance)
        iter_offsets = (
            range(-incoming_iter_overlap, max_iter_overlap + 1)
            if self.include_incoming_live else
            range(0, 1)
        )

        # 修复原始代码中未声明 static_smem_terms 的问题
        static_smem_terms = [
            int(footprint)
            for footprint in self.smem_allocations.values()
            if int(footprint) > 0
        ]
        if static_smem_terms:
            model.add(sum(static_smem_terms) <= self.smem_limit)

        # ---- SMEM 容量约束（全局统计，基于区间简化） -------------------------------
        # 针对 SMEM，我们将传统的逐 tau 枚举替换为轻量的绝对时间跨度区间检查，缩减变量规模
        if track_smem_liveness:
            for tau in range(L):
                smem_live_by_buffer: dict[str, list] = defaultdict(list)
                smem_footprint_by_buffer: dict[str, int] = {}
                spill_smem_terms = []
                for xi, (pv, oval) in enumerate(all_outputs):
                    if oval.storage != StorageKind.SMEM or oval.footprint_bytes <= 0:
                        continue
                    buffer_key = oval.smem_buffer_key()
                    if buffer_key in self.smem_allocations:
                        continue
                    
                    producer_lat = int(self.nodes[pv].latency)
                    consumers = consumers_of[xi] if self.include_incoming_live else [(cv, d) for cv, d in consumers_of[xi] if d == 0]
                    
                    any_copy_live_terms = []
                    for iter_offset in iter_offsets:
                        p_start = Tv[pv] + iter_offset * ii
                        is_produced = model.new_bool_var(f"smem_p_x={xi}_k={iter_offset}_t={tau}")
                        model.add(p_start <= tau).only_enforce_if(is_produced)
                        model.add(p_start > tau).only_enforce_if(is_produced.negated())
                        
                        if oval.lifetime == LifetimeSemantic.DEAD_ON_ENTRY and consumers:
                            all_c_done = model.new_bool_var(f"smem_c_x={xi}_k={iter_offset}_t={tau}")
                            c_terms = []
                            for cv, d in consumers:
                                c_bound = tau - d * ii - (1 if producer_lat == 0 else 0)
                                c_started = model.new_bool_var(f"smem_cv={cv}_t={tau}")
                                model.add(Tv[cv] + iter_offset * ii <= c_bound).only_enforce_if(c_started)
                                model.add(Tv[cv] + iter_offset * ii > c_bound).only_enforce_if(c_started.negated())
                                c_terms.append(c_started)
                            model.add_bool_and(c_terms).only_enforce_if(all_c_done)
                            model.add_bool_or([t.negated() for t in c_terms]).only_enforce_if(all_c_done.negated())
                            
                            live_var = model.new_bool_var(f"smem_l_x={xi}_k={iter_offset}_t={tau}")
                            model.add_bool_and([is_produced, all_c_done.negated()]).only_enforce_if(live_var)
                            model.add_bool_or([is_produced.negated(), all_c_done]).only_enforce_if(live_var.negated())
                            any_copy_live_terms.append(live_var)
                        else:
                            any_copy_live_terms.append(is_produced)
                            
                    any_copy_live = _or_var(f"smem_live_x={xi}_t={tau}", any_copy_live_terms)
                    smem_live_by_buffer[buffer_key].append(any_copy_live)
                    smem_footprint_by_buffer[buffer_key] = max(smem_footprint_by_buffer.get(buffer_key, 0), int(oval.footprint_bytes))
                    
                smem_terms = []
                for buffer_key, live_terms in smem_live_by_buffer.items():
                    buffer_live = _or_var(f"smem_live_buf={buffer_key}_t={tau}", live_terms)
                    smem_terms.append(smem_footprint_by_buffer[buffer_key] * buffer_live)
                if W > 1:
                    for xi, (pv, oval) in enumerate(all_outputs):
                        if (
                            oval.storage != StorageKind.RMEM
                            or oval.footprint_bytes <= 0
                            or oval.spill_cost <= 0
                        ):
                            continue
                        sc = int(oval.spill_cost)
                        consumers = (
                            consumers_of[xi]
                            if self.include_incoming_live else
                            [(cv, d) for cv, d in consumers_of[xi] if d == 0]
                        )
                        for cv, d in consumers:
                            for iter_offset in iter_offsets:
                                consume_time = Tv[cv] + (iter_offset + int(d)) * ii
                                spill_started = model.new_bool_var(
                                    f"spill_smem_started_x={xi}_c={cv}_k={iter_offset}_t={tau}"
                                )
                                spill_not_ended = model.new_bool_var(
                                    f"spill_smem_not_ended_x={xi}_c={cv}_k={iter_offset}_t={tau}"
                                )
                                model.add(consume_time - sc <= tau).only_enforce_if(spill_started)
                                model.add(consume_time - sc > tau).only_enforce_if(spill_started.negated())
                                model.add(consume_time > tau).only_enforce_if(spill_not_ended)
                                model.add(consume_time <= tau).only_enforce_if(spill_not_ended.negated())
                                spill_active = _and_var(
                                    f"spill_smem_active_x={xi}_c={cv}_k={iter_offset}_t={tau}",
                                    [spill_started, spill_not_ended],
                                )
                                same_warp_terms = [
                                    _and_var(
                                        f"spill_smem_same_x={xi}_c={cv}_k={iter_offset}_w={w}_t={tau}",
                                        [warp[(pv, w)], warp[(cv, w)]],
                                    )
                                    for w in range(W)
                                ]
                                has_same_warp = _or_var(
                                    f"spill_smem_has_same_x={xi}_c={cv}_k={iter_offset}_t={tau}",
                                    same_warp_terms,
                                )
                                spill_present = _and_var(
                                    f"spill_smem_x={xi}_c={cv}_k={iter_offset}_t={tau}",
                                    [spill_active, has_same_warp.negated()],
                                )
                                spill_smem_terms.append(
                                    int(oval.footprint_bytes) * spill_present
                                )
                all_smem_terms = static_smem_terms + smem_terms + spill_smem_terms
                if all_smem_terms:
                    model.add(sum(all_smem_terms) <= self.smem_limit)

        # ---- RMEM liveness 容量约束（稀疏 checkpoint） --------------------
        # 不再把每个 live range 建成 OptionalInterval + Cumulative。那种
        # 编码在 log4 的大窗口上会产生数百个 variable-size interval，
        # CP-SAT presolve 容易直接闭死。这里改为在少量 checkpoint 上
        # 约束 per-warp live bytes，并按 (buffer, iter_offset) 聚合，和
        # Python fixed liveness check 的“同一物理 RMEM buffer 只计一次”
        # 语义保持一致。
        rmem_liveness_checkpoints: list[int] = []
        rmem_folded_offsets = (
            range(-incoming_iter_overlap, 1)
            if self.include_incoming_live else
            range(0, 1)
        )
        if track_rmem_liveness:
            if self.fold_rmem_liveness_by_ii:
                checkpoint_set = set(range(0, ii, self.liveness_checkpoint_step))
                checkpoint_set.update({0, max(ii - 1, 0)})
            else:
                checkpoint_set = set(range(0, L, self.liveness_checkpoint_step))
                checkpoint_set.add(max(L - 1, 0))
            hint_times = {
                v: int(self.start_hints.get(node.name))
                for v, node in enumerate(self.nodes)
                if self.start_hints.get(node.name) is not None
            }
            for xi, (producer_v, oval) in enumerate(all_outputs):
                rmem_iter_offsets = (
                    (rmem_folded_offsets if self.fold_rmem_liveness_by_ii else iter_offsets)
                    if oval.lifetime == LifetimeSemantic.DEAD_ON_ENTRY else
                    range(0, 1)
                )
                for iter_offset in rmem_iter_offsets:
                    producer_hint = hint_times.get(producer_v)
                    if producer_hint is not None:
                        t = (
                            producer_hint % ii
                            if self.fold_rmem_liveness_by_ii else
                            producer_hint + iter_offset * ii
                        )
                        for dt in (-1, 0, 1):
                            limit = ii if self.fold_rmem_liveness_by_ii else L
                            if 0 <= t + dt < limit:
                                checkpoint_set.add(t + dt)
                    for cv, d in consumers_of.get(xi, []):
                        consumer_hint = hint_times.get(cv)
                        if consumer_hint is None:
                            continue
                        t = (
                            consumer_hint % ii
                            if self.fold_rmem_liveness_by_ii else
                            consumer_hint + (iter_offset + int(d)) * ii
                        )
                        for dt in (-1, 0, 1):
                            limit = ii if self.fold_rmem_liveness_by_ii else L
                            if 0 <= t + dt < limit:
                                checkpoint_set.add(t + dt)
            rmem_liveness_checkpoints = sorted(checkpoint_set)

        def _time_le_checkpoint_var(name: str, time_expr, tau: int):
            b = model.new_bool_var(name)
            model.add(time_expr <= tau).only_enforce_if(b)
            model.add(time_expr > tau).only_enforce_if(b.negated())
            return b

        if track_rmem_liveness:
            for tau in rmem_liveness_checkpoints:
                live_terms_by_warp_buffer: dict[tuple[int, str, int], list] = defaultdict(list)
                footprint_by_warp_buffer: dict[tuple[int, str, int], int] = {}
                for xi, (producer_v, oval) in enumerate(all_outputs):
                    if oval.storage != StorageKind.RMEM or oval.footprint_bytes <= 0:
                        continue

                    buffer_key = oval.rmem_buffer_key()
                    rmem_iter_offsets = (
                        (rmem_folded_offsets if self.fold_rmem_liveness_by_ii else iter_offsets)
                        if oval.lifetime == LifetimeSemantic.DEAD_ON_ENTRY else
                        range(0, 1)
                    )
                    producer_lat = int(self.nodes[producer_v].latency)
                    consumers = consumers_of[xi] if self.include_incoming_live else [(cv, d) for cv, d in consumers_of[xi] if d == 0]

                    for iter_offset in rmem_iter_offsets:
                        p_start = (
                            phase[producer_v] + iter_offset * ii
                            if self.fold_rmem_liveness_by_ii else
                            Tv[producer_v] + iter_offset * ii
                        )
                        is_produced = _time_le_checkpoint_var(
                            f"rmem_p_x={xi}_k={iter_offset}_t={tau}",
                            p_start,
                            tau,
                        )

                        if oval.lifetime == LifetimeSemantic.DEAD_ON_ENTRY:
                            if not consumers:
                                live_var = _false_var(
                                    f"rmem_l_false_x={xi}_k={iter_offset}_t={tau}"
                                )
                            else:
                                c_terms = []
                                offset = 1 if producer_lat == 0 else 0
                                # 所有消费者都已经启动/消费后，这份 copy 才能释放。
                                # c_started: phase/Tv[cv] + (iter_offset+d)*ii + offset <= tau
                                for cv, d in consumers:
                                    c_time = (
                                        phase[cv] + (iter_offset + int(d)) * ii + offset
                                        if self.fold_rmem_liveness_by_ii else
                                        Tv[cv] + (iter_offset + int(d)) * ii + offset
                                    )
                                    c_started = _time_le_checkpoint_var(
                                        f"rmem_c_x={xi}_k={iter_offset}_cv={cv}_t={tau}",
                                        c_time,
                                        tau,
                                    )
                                    c_terms.append(c_started)

                                all_c_done = _and_var(
                                    f"rmem_done_x={xi}_k={iter_offset}_t={tau}",
                                    c_terms,
                                )
                                live_var = model.new_bool_var(
                                    f"rmem_l_x={xi}_k={iter_offset}_t={tau}"
                                )
                                model.add_bool_and(
                                    [is_produced, all_c_done.negated()]
                                ).only_enforce_if(live_var)
                                model.add_bool_or(
                                    [is_produced.negated(), all_c_done]
                                ).only_enforce_if(live_var.negated())
                        else:
                            live_var = is_produced

                        for w in range(W):
                            live_on_warp = _and_var(
                                f"rmem_live_w={w}_x={xi}_k={iter_offset}_t={tau}",
                                [live_var, warp[(producer_v, w)]],
                            )
                            copy_key = (w, buffer_key, int(iter_offset))
                            live_terms_by_warp_buffer[copy_key].append(live_on_warp)
                            footprint_by_warp_buffer[copy_key] = max(
                                footprint_by_warp_buffer.get(copy_key, 0),
                                int(oval.footprint_bytes),
                            )

                terms_by_warp: dict[int, list] = defaultdict(list)
                for copy_key, live_terms in live_terms_by_warp_buffer.items():
                    w, buffer_key, iter_offset = copy_key
                    live_copy = _or_var(
                        f"rmem_live_buf_w={w}_b={buffer_key}_k={iter_offset}_t={tau}",
                        live_terms,
                    )
                    terms_by_warp[w].append(
                        footprint_by_warp_buffer[copy_key] * live_copy
                    )
                for w, terms in terms_by_warp.items():
                    if terms:
                        model.add(sum(terms) <= self.reg_limit)

        # ---- 优化目标：偏好更紧凑的调度 -------------------------------
        if optimize:
            end_times = []
            for v, node in enumerate(self.nodes):
                end_v = model.new_int_var(0, L - 1 + max(len(node.reservation), 1), f"end_T_v={v}")
                model.add(end_v == Tv[v] + max(len(node.reservation), 1))
                end_times.append(end_v)
            mx = model.new_int_var(0, L - 1 + max((max(len(n.reservation), 1) for n in self.nodes), default=1), "max_end_T")
            model.add_max_equality(mx, end_times)
            # Keep makespan compact, then strongly discourage cross-warpgroup
            # RMEM communication because codegen does not yet materialize a
            # real cross-WG register spill/copy path.
            model.minimize(
                L * N * mx + sum(Tv)
            )

        # ---- 求解 ---------------------------------------------------------
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(float(self.timeout_ms) / 1000.0, 1)
        solver.parameters.num_workers = 4
        solver.parameters.log_search_progress = self.log_search_progress
        try:
            proto_getter = getattr(model, "proto", None)
            if callable(proto_getter):
                proto = proto_getter()
            elif proto_getter is not None:
                proto = proto_getter
            else:
                proto = model.Proto()
            print(
                f"----- HeddleScheduler model built: "
                f"vars={len(proto.variables)} constraints={len(proto.constraints)}",
                flush=True,
            )
        except Exception:
            pass
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            status_name = {
                cp_model.OPTIMAL: "OPTIMAL",
                cp_model.FEASIBLE: "FEASIBLE",
                cp_model.INFEASIBLE: "INFEASIBLE",
                cp_model.MODEL_INVALID: "MODEL_INVALID",
                cp_model.UNKNOWN: "UNKNOWN",
            }.get(status, str(status))
            try:
                wall_time = float(solver.WallTime())
            except Exception:
                wall_time = 0.0
            print(
                f"----- HeddleScheduler solve failed: reason=cp_sat_status "
                f"status={status_name} ii={ii} L={L} walltime={wall_time:.6f}",
                flush=True,
            )
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

        # ---- 结果后处理（在外部 Python 环境执行，不再消耗约束算力） -------
        variable_lifetimes: Dict[str, Dict[str, object]] = {}
        for xi, (pv, oval) in enumerate(all_outputs):
            producer_time = int(solver.value(Tv[pv]))
            producer_lat = int(self.nodes[pv].latency)
            producer_name = self.nodes[pv].name
            producer_warp = int(warp_assign.get(producer_name, 0))
            buffer_key = (
                oval.smem_buffer_key()
                if oval.storage == StorageKind.SMEM else
                oval.rmem_buffer_key()
            )
            consumers = (
                consumers_of[xi]
                if self.include_incoming_live else
                [(cv, d) for cv, d in consumers_of[xi] if d == 0]
            )
            value_iter_offsets = (
                iter_offsets
                if oval.lifetime == LifetimeSemantic.DEAD_ON_ENTRY else
                range(0, 1)
            )

            buffer_lifetime = variable_lifetimes.setdefault(buffer_key, {
                "name": buffer_key,
                "storage": oval.storage.value,
                "buffer": buffer_key,
                "footprint_bytes": int(oval.footprint_bytes),
                "lifetime": oval.lifetime.value,
                "producers": [],
                "copies": [],
            })
            buffer_lifetime["footprint_bytes"] = max(
                int(buffer_lifetime.get("footprint_bytes", 0)),
                int(oval.footprint_bytes),
            )
            producers = buffer_lifetime.setdefault("producers", [])
            if producer_name not in producers:
                producers.append(producer_name)

            for iter_offset in value_iter_offsets:
                live_start = producer_time + int(iter_offset) * ii
                consumer_entries = []
                release_times = []
                for cv, d in consumers:
                    consume_time = (
                        int(solver.value(Tv[cv]))
                        + (int(iter_offset) + int(d)) * ii
                    )
                    if producer_lat == 0:
                        consume_time += 1
                    consumer_entries.append({
                        "consumer": self.nodes[cv].name,
                        "distance": int(d),
                        "consume_time": int(consume_time),
                    })
                    release_times.append(int(consume_time))

                if oval.lifetime == LifetimeSemantic.DEAD_ON_ENTRY:
                    if not release_times:
                        live_end_exclusive: Optional[int] = live_start
                        live_end: Optional[int] = None
                    else:
                        live_end_exclusive = max(release_times)
                        live_end = live_end_exclusive - 1
                else:
                    live_end_exclusive = None
                    live_end = None

                buffer_lifetime["copies"].append({
                    "producer": producer_name,
                    "producer_warp": producer_warp,
                    "iter_offset": int(iter_offset),
                    "live_start": int(live_start),
                    "live_end": live_end,
                    "live_end_exclusive": live_end_exclusive,
                    "consumers": consumer_entries,
                })

        reg_peak: Dict[int, int] = {}  # warp 内每个线程 寄存器用量最大值
        if track_rmem_liveness:
            for w in range(W):
                peak = 0
                for tau in rmem_liveness_checkpoints:
                    live_bytes_by_copy: Dict[tuple[str, int], int] = {}
                    for xi, (pv, oval) in enumerate(all_outputs):
                        if oval.storage != StorageKind.RMEM:
                            continue
                        owned = bool(solver.value(warp[(pv, w)]))
                        if not owned:
                            continue
                        buffer_key = oval.rmem_buffer_key()
                        rmem_iter_offsets = (
                            (rmem_folded_offsets if self.fold_rmem_liveness_by_ii else iter_offsets)
                            if oval.lifetime == LifetimeSemantic.DEAD_ON_ENTRY else
                            range(0, 1)
                        )
                        producer_lat = int(self.nodes[pv].latency)
                        consumers = consumers_of[xi] if self.include_incoming_live else [(cv, d) for cv, d in consumers_of[xi] if d == 0]
                        for iter_offset in rmem_iter_offsets:
                            base_produced_at = (
                                solver.value(phase[pv])
                                if self.fold_rmem_liveness_by_ii else
                                solver.value(Tv[pv])
                            )
                            produced_at = base_produced_at + iter_offset * ii
                            if produced_at > tau:
                                continue
                            if oval.lifetime == LifetimeSemantic.DEAD_ON_ENTRY:
                                if not consumers:
                                    continue
                                live = False
                                for cv, d in consumers:
                                    base_consume_time = (
                                        solver.value(phase[cv])
                                        if self.fold_rmem_liveness_by_ii else
                                        solver.value(Tv[cv])
                                    )
                                    consume_time = (
                                        base_consume_time + (iter_offset + int(d)) * ii
                                    )
                                    if producer_lat == 0:
                                        consume_time += 1
                                    if consume_time > tau:
                                        live = True
                                        break
                            else:
                                live = True
                            if live:
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
            "variable_lifetimes": variable_lifetimes,
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
