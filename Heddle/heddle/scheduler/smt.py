"""
Heddle Scheduler (SMT-based modulo scheduler).

Canonical import path:
  - `from heddle.scheduler.smt import HeddleScheduler, OpNode, ResourceType`

Phase A: Find minimum II with dependency + FU capacity constraints.
Phase B: Joint schedule + warp assignment + liveness (incl. incoming_live for
         loop-carried deps) + register/SMEM capacity + spill cost + full
         concurrency constraints + APLSP bound tightening.
"""

from __future__ import annotations

import enum
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Thread-safe global call counter for unique Z3 variable names.
# Z3 uses a global hash-cons table keyed by variable name.  When many solver
# instances reuse the same names, the reference-counting can go wrong if
# Python's GC frees a BoolRef/IntRef while a C-level Z3 struct still holds a
# raw pointer to the same AST node (or vice-versa).  Unique names per call
# eliminate the collision entirely at the cost of a small counter bump.
_z3_call_counter_lock = threading.Lock()
_z3_call_counter = 0

def _next_call_id() -> int:
    global _z3_call_counter
    with _z3_call_counter_lock:
        _z3_call_counter += 1
        return _z3_call_counter

import z3


# ====================================================================== #
# Data types
# ====================================================================== #

class ResourceType(enum.Enum):
    TMA = "TMA"
    TensorCore = "TC"
    ALU = "ALU"
    SFU = "SFU"


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


@dataclass
class EdgeInfo:
    blocking_sync: bool = False
    warp_disjoint: bool = False
    warpgroup_disjoint: bool = False


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

    def add_dependency(self, parent: "OpNode", distance: int = 0,
                       blocking_sync: bool = False,
                       warp_disjoint: bool = False,
                       warpgroup_disjoint: bool = False):
        self.parents.append(parent)
        parent.children.append(self)
        self.dependency_distance[parent.name] = distance
        self.edge_info[parent.name] = EdgeInfo(
            blocking_sync=blocking_sync,
            warp_disjoint=warp_disjoint,
            warpgroup_disjoint=warpgroup_disjoint,
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
    """SMT-based Modulo Scheduler for Hopper Pipelines."""

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
        hard_warp_disjoint: bool = True,
    ):
        self.nodes = nodes
        self.fu_caps = fu_caps or {
            ResourceType.TMA: 1, ResourceType.TensorCore: 1,
            ResourceType.ALU: 1, ResourceType.SFU: 1,
        }
        self.reg_limit = reg_limit
        self.smem_limit = smem_limit
        self.num_warps = num_warps
        self.timeout_ms = timeout_ms
        self.disallow_spills = disallow_spills
        self.use_spill_concurrency = use_spill_concurrency
        self.include_incoming_live = include_incoming_live
        self.hard_warp_disjoint = hard_warp_disjoint

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
        # Motivation: Z3 (v4.15.4) crashes during model evaluation when
        # _solve_phase_b's z3.Optimize() runs first (global context state
        # becomes corrupted). An ASAP longest-path solver is sufficient for
        # Phase A's purpose (finding any valid schedule for ordering), and
        # has no Z3 dependency that can be corrupted.
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
        self._ensure_reservations()
        N = len(self.nodes)
        idx = {n.name: i for i, n in enumerate(self.nodes)}
        if N == 0:
            return {"ii": ii, "schedule": {}, "warp_assign": {}, "reg_peak": {}}

        W = max(self.num_warps, 1)
        # 每次调用都给 Z3 变量加唯一前缀，避免不同 solver 调用之间
        # 因变量名相同而在 Z3 全局 hash-cons 表里发生 AST 别名冲突。
        p = _next_call_id()
        solver: z3.Solver | z3.Optimize
        solver = z3.Optimize() if optimize else z3.Solver()
        solver.set("timeout", self.timeout_ms)

        # ---- P3：APLSP 时间边界收紧 --------------------------------------
        aplsp = _compute_aplsp(self.nodes, idx, ii)

        # P4 的对称性破除会在变量定义之后添加。

        # ---- 决策变量 ------------------------------------------------------
        op: dict[tuple[int, int], z3.BoolRef] = {}
        for v in range(N):
            for t in range(L):
                op[(v, t)] = z3.Bool(f"PB{p}_op_v={v}_t={t}")

        warp: dict[tuple[int, int], z3.BoolRef] = {}
        for v in range(N):
            for w in range(W):
                warp[(v, w)] = z3.Bool(f"PB{p}_warp_v={v}_w={w}")

        # ---- 唯一启动时间 --------------------------------------------------
        for v in range(N):
            solver.add(z3.Sum([z3.If(op[(v, t)], 1, 0) for t in range(L)]) == 1)

        # ---- warp 分配（支持多 warp op 和可复制 op） -----------------------
        for v in range(N):
            nd = self.nodes[v]
            wc = max(nd.warp_count, 1)
            if nd.replicable or wc == 1:
                solver.add(z3.Sum([z3.If(warp[(v, w)], 1, 0) for w in range(W)]) == 1)
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
                    s: z3.Bool(f"PB{p}_warp_start_v={v}_w={s}")
                    for s in start_slots
                }
                solver.add(z3.Sum([z3.If(st, 1, 0) for st in starts.values()]) == 1)
                for w in range(W):
                    covering_starts = [
                        starts[s]
                        for s in start_slots
                        if s <= w < s + wc
                    ]
                    solver.add(warp[(v, w)] == (
                        z3.Or(covering_starts) if covering_starts else z3.BoolVal(False)
                    ))

        # ---- 整数启动时间表达式 -------------------------------------------
        Tv = [z3.Sum([z3.If(op[(v, t)], t, 0) for t in range(L)]) for v in range(N)]

        # ---- P3：由 APLSP 推导出的时间下界 -------------------------------
        for (u, v), d in aplsp.items():
            if d > 0:
                solver.add(Tv[v] - Tv[u] >= d)

        # ---- P4：对独立同类节点做对称性破除 -------------------------------
        # 如果两个节点没有依赖关系，且 FU 类型和 latency 完全相同，
        # 则强制前者不晚于后者启动，减少等价调度带来的搜索空间。
        dep_pairs = set(aplsp.keys())
        for v1 in range(N):
            for v2 in range(v1 + 1, N):
                n1, n2 = self.nodes[v1], self.nodes[v2]
                if (n1.resource_type == n2.resource_type and n1.latency == n2.latency
                        and (v1, v2) not in dep_pairs and (v2, v1) not in dep_pairs):
                    solver.add(Tv[v1] <= Tv[v2])

        # ---- 依赖、跨 warp spill 代价和 blocking sync ---------------------
        warp_disjoint_penalties: list[z3.ArithRef] = []
        for v_node in self.nodes:
            vi = idx[v_node.name]
            for par in v_node.parents:
                ui = idx[par.name]
                delta = int(v_node.dependency_distance.get(par.name, 0))
                base_delay = int(par.latency)
                edge = v_node.edge_info.get(par.name)

                spill_cost = max((o.spill_cost for o in par.outputs), default=0)

                if spill_cost > 0 and W > 1:
                    same_w = z3.Or([z3.And(warp[(ui, w)], warp[(vi, w)]) for w in range(W)])
                    if self.disallow_spills:
                        solver.add(same_w)
                        solver.add(Tv[vi] - Tv[ui] >= base_delay - delta * ii)
                    else:
                        eff = z3.If(same_w, base_delay, base_delay + spill_cost)
                        solver.add(Tv[vi] - Tv[ui] >= eff - delta * ii)
                else:
                    solver.add(Tv[vi] - Tv[ui] >= base_delay - delta * ii)

                # producer / WGMMA consumer 必须分属不同 warpgroup：
                # 任意被选中的 warp id 都不能满足 wu // 4 == wv // 4。
                # 这是硬约束，即使 fallback 到 soft warp_disjoint 也不能放松。
                if edge and edge.warpgroup_disjoint and W > 1:
                    for wu in range(W):
                        for wv in range(W):
                            if wu // 4 == wv // 4:
                                solver.add(z3.Not(z3.And(warp[(ui, wu)], warp[(vi, wv)])))

                # producer / consumer 角色分离偏好：用于较弱的“不要同 warp”
                # 情况；hard_warp_disjoint=False 时可降级成 penalty。
                if edge and edge.warp_disjoint and W > 1:
                    for w in range(W):
                        overlap = z3.And(warp[(ui, w)], warp[(vi, w)])
                        if self.hard_warp_disjoint:
                            solver.add(z3.Not(overlap))
                        else:
                            warp_disjoint_penalties.append(z3.If(overlap, 1, 0))

                # P2：blocking sync 的完整并发约束。
                # 这种边不仅要求时间顺序，还会约束相关 warp 的覆盖关系；
                # 在同步阻塞窗口内，同一个 warp 上不能安排其他重叠 op。
                if edge and edge.blocking_sync:
                    if W > 1:
                        for w in range(W):
                            solver.add(z3.Implies(
                                warp[(ui, w)], warp[(vi, w)]))
                    lat_u = max(int(par.latency), 1)
                    for w in range(W):
                        for other in range(N):
                            if other == ui or other == vi:
                                continue
                            lat_o = max(int(self.nodes[other].latency), 1)
                            for t in range(L):
                                # v 在 t 启动时，u 近似活跃于 [t - base_delay, t)。
                                # 若 other 的执行窗口与该阻塞区间重叠，则禁止
                                # other 使用同一个 warp。
                                for to in range(max(0, t - base_delay - lat_o + 1), t + 1):
                                    if to >= L:
                                        continue
                                    solver.add(z3.Implies(
                                        z3.And(op[(vi, t)], warp[(vi, w)],
                                               op[(other, to)]),
                                        z3.Not(warp[(other, w)])))

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
                            for other in range(N):
                                if other == vi:
                                    continue
                                for t in range(L):
                                    for to in range(max(0, t - sc + 1), t + 1):
                                        if to >= L:
                                            continue
                                        solver.add(z3.Implies(
                                            z3.And(warp[(ui, w_src)], warp[(vi, w_dst)],
                                                   op[(vi, t)], op[(other, to)]),
                                            z3.Not(warp[(other, w_dst)])))

        # ---- FU 容量约束 ---------------------------------------------------
        expanded = self._fold_reservations(ii)
        for t in range(L):
            for r, cap in self.fu_caps.items():
                terms = []
                for v in range(N):
                    for l in range(ii):
                        c = int(expanded[v][l].get(r, 0))
                        if c:
                            tp = (t - l) % ii
                            if tp < L:
                                terms.append(z3.If(op[(v, tp)], c, 0))
                if terms:
                    solver.add(z3.Sum(terms) <= int(cap))

        # ---- 活跃区间、P1 incoming_live 和容量约束 ------------------------
        all_outputs: list[tuple[int, OutputValue]] = []
        for v in range(N):
            for out in self.nodes[v].outputs:
                all_outputs.append((v, out))

        # 检测跨迭代传递的输出：只要某个消费者边的 distance(delta) > 0，
        # 该输出就可能在当前迭代开始时已经来自上一轮迭代并保持 live。
        loop_carried: set[int] = set()
        consumers_of: dict[int, list[tuple[int, int]]] = defaultdict(list)
        output_name_to_xi: dict[str, int] = {}
        for xi, (pv, oval) in enumerate(all_outputs):
            output_name_to_xi[oval.name] = xi

        for v_node in self.nodes:
            vi = idx[v_node.name]
            for par in v_node.parents:
                delta = int(v_node.dependency_distance.get(par.name, 0))
                for oval in par.outputs:
                    xi = output_name_to_xi.get(oval.name)
                    if xi is not None:
                        consumers_of[xi].append((vi, delta))
                        if delta > 0:
                            loop_carried.add(xi)

        if all_outputs and self.reg_limit > 0:
            # live[xi, tau]：第 xi 个输出在第 0 轮迭代的 tau 时刻是否 live。
            live: dict[tuple[int, int], z3.BoolRef] = {}
            for xi in range(len(all_outputs)):
                for tau in range(L):
                    live[(xi, tau)] = z3.Bool(f"PB{p}_live_x={xi}_t={tau}")

            # P1：incoming_live[xi, tau] 表示跨迭代值 xi 是否在 tau 时刻
            # 仍然由“上一轮迭代”带入并保持 live，用于统计跨迭代寄存器压力。
            incoming_live: dict[tuple[int, int], z3.BoolRef] = {}
            if self.include_incoming_live:
                for xi in loop_carried:
                    for tau in range(L):
                        incoming_live[(xi, tau)] = z3.Bool(f"PB{p}_ilive_x={xi}_t={tau}")

            # ---- 活跃区间约束 ---------------------------------------------
            for xi, (producer_v, oval) in enumerate(all_outputs):
                same_iter_consumers = [(cv, d) for cv, d in consumers_of[xi] if d == 0]
                producer_lat = int(self.nodes[producer_v].latency)

                for tau in range(L):
                    produced_by = z3.Or([op[(producer_v, t)] for t in range(tau + 1)])

                    if oval.lifetime == LifetimeSemantic.DEAD_ON_ENTRY:
                        if same_iter_consumers:
                            # 对 zero-latency producer，consumer 可能和 producer
                            # 在同一个时间步启动。该 cycle 内输出仍占寄存器，
                            # 因此判断“是否已消费完”时必须严格早于 tau。
                            # 这个语义对应 Twill warpspecialization.rs L371-409。
                            if producer_lat == 0:
                                all_consumed = z3.And([
                                    z3.Or([op[(cv, tc)] for tc in range(tau)])
                                    for cv, _ in same_iter_consumers
                                ]) if tau > 1 else z3.BoolVal(False)
                            else:
                                all_consumed = z3.And([
                                    z3.Or([op[(cv, tc)] for tc in range(tau + 1)])
                                    for cv, _ in same_iter_consumers
                                ]) if tau > 0 else z3.BoolVal(False)
                            solver.add(live[(xi, tau)] == z3.And(produced_by, z3.Not(all_consumed)))
                        else:
                            solver.add(live[(xi, tau)] == z3.BoolVal(False))
                    else:
                        solver.add(live[(xi, tau)] == produced_by)

                # P1：为跨迭代值建立 incoming_live。
                if self.include_incoming_live and xi in loop_carried:
                    cross_iter_consumers = [(cv, d) for cv, d in consumers_of[xi] if d > 0]
                    for tau in range(L):
                        if cross_iter_consumers:
                            # 如果第 0 轮迭代中仍有消费者在 tau 时刻前尚未启动，
                            # 那么来自上一轮的值在 tau 时刻仍需要保持 live。
                            some_consumer_pending = z3.Or([
                                z3.Not(z3.Or([op[(cv, tc)] for tc in range(tau + 1)]))
                                for cv, _ in cross_iter_consumers
                            ]) if tau > 0 else z3.BoolVal(True)
                            solver.add(incoming_live[(xi, tau)] == some_consumer_pending)
                        else:
                            solver.add(incoming_live[(xi, tau)] == z3.BoolVal(False))

            # ---- 每个 warp、每个时间步的寄存器容量约束 --------------------
            for w in range(W):
                for tau in range(L):
                    rmem_terms = []
                    for xi, (pv, oval) in enumerate(all_outputs):
                        if oval.storage != StorageKind.RMEM or oval.footprint_bytes <= 0:
                            continue
                        # 同一轮迭代内的 live 值。
                        rmem_terms.append(
                            z3.If(z3.And(warp[(pv, w)], live[(xi, tau)]),
                                  oval.footprint_bytes, 0))
                        # P1：跨迭代 live 值，计入 producer 所在 warp 的压力。
                        if self.include_incoming_live and xi in loop_carried:
                            rmem_terms.append(
                                z3.If(z3.And(warp[(pv, w)], incoming_live[(xi, tau)]),
                                      oval.footprint_bytes, 0))
                    if rmem_terms:
                        solver.add(z3.Sum(rmem_terms) <= self.reg_limit)

            # ---- SMEM 容量约束（全局统计，也包含 incoming_live） ----------
            for tau in range(L):
                smem_terms = []
                for xi, (pv, oval) in enumerate(all_outputs):
                    if oval.storage != StorageKind.SMEM or oval.footprint_bytes <= 0:
                        continue
                    smem_terms.append(z3.If(live[(xi, tau)], oval.footprint_bytes, 0))
                    if self.include_incoming_live and xi in loop_carried:
                        smem_terms.append(
                            z3.If(incoming_live[(xi, tau)], oval.footprint_bytes, 0))
                if smem_terms:
                    solver.add(z3.Sum(smem_terms) <= self.smem_limit)

        # ---- 优化目标：偏好更紧凑的调度 -------------------------------
        if optimize:
            if warp_disjoint_penalties:
                solver.minimize(z3.Sum(warp_disjoint_penalties))
            mx = z3.Int(f"PB{p}_max_T")
            solver.add(mx >= 0)
            for texpr in Tv:
                solver.add(mx >= texpr)
            solver.minimize(mx)
            if Tv:
                solver.minimize(z3.Sum(Tv))

        # ---- 求解 ---------------------------------------------------------
        if solver.check() != z3.sat:
            return None

        model = solver.model()

        schedule = {}
        for v in range(N):
            for t in range(L):
                if z3.is_true(model.eval(op[(v, t)])):
                    schedule[self.nodes[v].name] = t
                    break

        warp_assign = {}
        for v in range(N):
            for w in range(W):
                if z3.is_true(model.eval(warp[(v, w)])):
                    warp_assign[self.nodes[v].name] = w
                    break

        reg_peak: Dict[int, int] = {}
        if all_outputs and self.reg_limit > 0:
            for w in range(W):
                peak = 0
                for tau in range(L):
                    total = 0
                    for xi, (pv, oval) in enumerate(all_outputs):
                        if oval.storage != StorageKind.RMEM:
                            continue
                        owned = z3.is_true(model.eval(warp[(pv, w)]))
                        is_live = z3.is_true(model.eval(live[(xi, tau)]))
                        is_incoming = (self.include_incoming_live and xi in loop_carried and
                                       z3.is_true(model.eval(incoming_live[(xi, tau)])))
                        if owned and (is_live or is_incoming):
                            total += oval.footprint_bytes
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
                n.reservation = [{n.resource_type: 1} for _ in range(max(int(n.latency), 0))]

    def _fold_reservations(self, ii: int) -> list[dict[int, dict[ResourceType, int]]]:
        expanded: list[dict[int, dict[ResourceType, int]]] = []
        for n in self.nodes:
            tbl: dict[int, dict[ResourceType, int]] = {l: {} for l in range(ii)}
            for j, per_cycle in enumerate(n.reservation):
                l = j % ii
                for r, cnt in per_cycle.items():
                    tbl[l][r] = int(tbl[l].get(r, 0)) + int(cnt)
            expanded.append(tbl)
        return expanded
