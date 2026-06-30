"""
Heddle Consumer Schedule — SMT-based consumer-side reordering for PCWS.

This pass runs BEFORE ProducerConsumerWarpSpecialized (PCWS).  It analyzes
the pipeline loop body, classifies statements as producer or consumer, and
uses the Heddle SMT scheduler to find an optimal consumer ordering.  The
reordered IR is then consumed by PCWS, which performs barrier placement
based on the new statement order — yielding provably optimal barrier
positions.

Key insight: PCWS places forward-wait at first_read and backpressure-arrive
at last_access+1 (per buffer).  By reordering consumer statements, Heddle
can shift these positions to minimize the critical path and maximize
producer/consumer overlap.

Optimizations:
  1. ALAP + slack-based scheduling: delays non-critical consumers to reduce
     peak register liveness without extending the critical path.
  2. Buffer-span-aware priority: groups consumers reading the same producer
     buffer and delays first-readers to hide producer latency.
  3. Relaxed producer-boundary constraints: per-buffer ordering instead of
     global ordering across producer boundaries when buffers are disjoint.

Usage:
    with PassContext(config={
        "tl.enable_heddle_consumer_schedule": True,
    }):
        mod = tilelang.transform.HeddleConsumerSchedule()(mod)

Or automatically via phase.py when the config key is set.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from tvm.ir.expr import PrimExpr

import tilelang
from tilelang import tvm as tvm

if TYPE_CHECKING:
    from heddle.scheduler.smt import OpNode, ResourceType

# ---------------------------------------------------------------------------
# Re-use helpers from auto_tl_pipeline_smt
# ---------------------------------------------------------------------------
from heddle.transform.auto_tl_pipeline_smt import (
    _build_stmt_infos,
    _call_op_names,
    _collect_func_alloc_buffers,
    _collect_rw_regions,
    _detect_op_latency_and_resource,
    _detect_op_issue_cycles,
    _detect_wgmma_issue_cycles,
    _estimate_buffer_footprint_bytes,
    _extract_num_threads,
    _is_shared,
    _StmtInfo,
    _unwrap_to_seqstmt,
)


# ---------------------------------------------------------------------------
# Three-role WS auto-detection
# ---------------------------------------------------------------------------

def _detect_tma_reduce_add(seq) -> bool:
    """Detect T.atomic_add(..., use_tma=True) in a SeqStmt.

    Walks all statements looking for Call nodes to ``tl.tileop.atomicadd``
    whose annotations contain ``use_tma = 1``.  When found, this indicates
    the kernel uses TMA-based reduce-add (e.g. dQ writeback in FA BWD),
    and PCWS should extract those into a dedicated third-role warp.
    """

    def _walk(node) -> bool:
        if isinstance(node, tvm.tir.Call):
            op = node.op
            if hasattr(op, "name") and "atomicadd" in str(op.name):
                # Check annotations arg (last arg by convention)
                for arg in node.args:
                    if isinstance(arg, tvm.tir.StringImm) and "use_tma" in arg.value:
                        return True
                # Also check via call attributes if available
                if hasattr(node, "attrs") and node.attrs:
                    try:
                        ann = dict(node.attrs)
                        if ann.get("use_tma"):
                            return True
                    except Exception:
                        pass
            return False
        if isinstance(node, tvm.tir.Evaluate):
            return _walk(node.value)
        if isinstance(node, tvm.tir.SeqStmt):
            return any(_walk(s) for s in node.seq)
        if isinstance(node, tvm.tir.LetStmt):
            return _walk(node.body)
        if isinstance(node, tvm.tir.AttrStmt):
            return _walk(node.body)
        if isinstance(node, tvm.tir.IfThenElse):
            if _walk(node.then_case):
                return True
            if node.else_case is not None and _walk(node.else_case):
                return True
        if isinstance(node, tvm.tir.For):
            return _walk(node.body)
        if isinstance(node, tvm.tir.Block):
            return _walk(node.body)
        if isinstance(node, tvm.tir.BlockRealize):
            return _walk(node.block)
        return False

    return _walk(seq)


# ---------------------------------------------------------------------------
# Barrier hint protocol (PCWS performance upgrade: Proposal 2)
# ---------------------------------------------------------------------------

@dataclass
class BarrierHint:
    """Per-buffer barrier position hint from Phase B for PCWS."""
    buffer_name: str
    suggested_wait_pos: int    # compute_stmt index for forward-wait
    suggested_arrive_pos: int  # compute_stmt index for backpressure-arrive
    delay_cycles: int = 0      # cycles the wait can be safely delayed


@dataclass
class PhaseB_PCWS_Hint:
    """Complete hint package from Phase B to PCWS.

    Contains the consumer ordering (already used) plus new barrier position
    hints and stage offset recommendations.
    """
    consumer_ordering: List[int]                           # existing
    barrier_hints: Dict[str, BarrierHint] = field(default_factory=dict)  # new
    stage_offsets: Dict[int, int] = field(default_factory=dict)          # new

    def to_barrier_hints_config(self) -> str:
        """Serialize barrier hints to PCWS config string format.

        Format: "buffer_name:wait=W,arrive=A;buffer_name2:wait=W2,arrive=A2"
        """
        parts = []
        for name, hint in self.barrier_hints.items():
            parts.append(f"{name}:wait={hint.suggested_wait_pos},arrive={hint.suggested_arrive_pos}")
        return ";".join(parts)

    def to_stage_offsets_config(self) -> str:
        """Serialize stage offsets to PCWS config string format.

        Format: "idx1=offset1,idx2=offset2"
        """
        parts = []
        for idx, offset in self.stage_offsets.items():
            if offset != 0:
                parts.append(f"{idx}={offset}")
        return ",".join(parts)


def _get_bool_config(ctx: tvm.ir.transform.PassContext, key: str, default: bool) -> bool:
    try:
        from heddle._monkey_patch import get_heddle_pass_config
        return bool(ctx.config.get(key, get_heddle_pass_config(key, default)))
    except Exception:
        return default


def _get_int_config(ctx: tvm.ir.transform.PassContext, key: str, default: int) -> int:
    try:
        from heddle._monkey_patch import get_heddle_pass_config
        return int(ctx.config.get(key, get_heddle_pass_config(key, default)))
    except Exception:
        return default


def _ws_annotation_aliases(key: str) -> Tuple[str, ...]:
    """Return annotation keys for main PCWS and legacy FineGrainedWS backends."""
    if key.startswith("tl_pcws_"):
        return (key, "tl_finegrainedws_" + key[len("tl_pcws_"):])
    return (key,)


def _has_ws_annotation(annotations: Dict[str, object], key: str) -> bool:
    return any(alias in annotations for alias in _ws_annotation_aliases(key))


def _set_ws_annotation(annotations: Dict[str, object], key: str, value: object) -> None:
    for alias in _ws_annotation_aliases(key):
        annotations[alias] = value


def _ensure_ws_annotation(annotations: Dict[str, object], key: str, value: object) -> bool:
    aliases = _ws_annotation_aliases(key)
    existing = next((annotations[alias] for alias in aliases if alias in annotations), value)
    changed = False
    for alias in aliases:
        if annotations.get(alias) != existing:
            annotations[alias] = existing
            changed = True
    return changed


# ---------------------------------------------------------------------------
# Consumer dependency graph builder
# ---------------------------------------------------------------------------

def _build_consumer_dep_graph(
    infos: List[_StmtInfo], barrier_infos: Dict[PrimExpr, List[Tuple[str, int]]],
    *,
    relax_producer_boundary: bool = False,
) -> Tuple[List[int], Dict[int, List[int]], List[int], Dict[int, List[int]]]:
    """Build a dependency graph for consumer statements.

    Returns:
        consumer_indices: list of statement indices that are consumers
        deps: dict mapping consumer_idx -> list of consumer_idx it depends on
        all_indices: list of all statement indices
        deps_all: dict mapping stmt_idx -> list of stmt_idx it depends on,
            including producer/consumer dependencies

    Dependencies include:
    1. Buffer RAW (read-after-write) through any buffer
    2. Sync ordering: sync statements (barrier waits) get a program-order
       chain to preserve barrier timing relative to compute
    3. Producer-consumer ordering: the first consumer after each producer
       must stay ordered (preserves PCWS barrier placement semantics)
    """
    consumer_indices = [info.idx for info in infos if not info.is_producer]
    all_indices = [info.idx for info in infos]

    last_writer: Dict[str, int] = {}  # buffer_name -> stmt_idx
    deps: Dict[int, List[int]] = {idx: [] for idx in consumer_indices}
    deps_all: Dict[int, List[int]] = {idx: [] for idx in all_indices}

    consumer_set = set(consumer_indices)

    def _add_all_dep(dst: int, src: int) -> None:
        if src != dst and src not in deps_all[dst]:
            deps_all[dst].append(src)

    def _add_consumer_dep(dst: int, src: int) -> None:
        if dst in consumer_set and src in consumer_set:
            if src != dst and src not in deps[dst]:
                deps[dst].append(src)
        _add_all_dep(dst, src)

    def _is_barrier_wait(info: _StmtInfo) -> bool:
        if not (info.is_sync_top or info.is_sync_nested):
            return False
        return bool(_call_op_names(info.stmt) & {
            "tl.mbarrier_wait_parity",
            "tir.ptx_wait_barrier",
        })

    for info in infos:
        idx = info.idx
        for rd in info.reads:
            buf_name = rd.buffer.name
            if buf_name in last_writer:
                _add_consumer_dep(idx, last_writer[buf_name])

        if info.is_producer:
            for wr in info.writes:
                if _is_shared(wr.buffer):
                    last_writer[wr.buffer.name] = idx
            continue

        for wr in info.writes:
            last_writer[wr.buffer.name] = idx

    prev_producer = None
    for info in infos:
        if info.is_producer:
            if any(_is_shared(wr.buffer) for wr in info.writes):
                prev_producer = info.idx
            continue
        if prev_producer is not None and _is_barrier_wait(info):
            _add_all_dep(info.idx, prev_producer)

    # Add sync-chain constraints: sync statements must stay in program order
    # relative to each other AND relative to compute that reads producer buffers.
    # This prevents moving barrier waits before the compute they protect.
    prev_sync = None
    for ci in consumer_indices:
        info = infos[ci]
        if info.is_sync_top or info.is_sync_nested:
            if prev_sync is not None:
                _add_consumer_dep(ci, prev_sync)
            prev_sync = ci

    # A barrier wait does not necessarily read the shared buffer it protects, so
    # region analysis cannot infer wait -> reader.  Add that semantic edge for
    # readers of producer-written shared buffers after the nearest wait.
    producer_written_bufs: Set[str] = set()
    for info in infos:
        if info.is_producer:
            producer_written_bufs.update(
                wr.buffer.name for wr in info.writes if _is_shared(wr.buffer)
            )

    prev_wait = None
    for ci in consumer_indices:
        info = infos[ci]
        if _is_barrier_wait(info):
            prev_wait = ci
            continue
        if prev_wait is None:
            continue
        if any(rd.buffer.name in producer_written_bufs for rd in info.reads):
            _add_consumer_dep(ci, prev_wait)

    # Add producer-boundary constraints
    producer_groups: List[Tuple[int, Set[str]]] = []  # (producer_idx, written_buffer_names)
    for info in infos:
        if info.is_producer:
            bufs = {wr.buffer.name for wr in info.writes if _is_shared(wr.buffer)}
            if bufs:
                producer_groups.append((info.idx, bufs))

    if len(producer_groups) >= 2:
        for pg_idx in range(1, len(producer_groups)):
            prev_bufs = producer_groups[pg_idx - 1][1]
            curr_bufs = producer_groups[pg_idx][1]

            # Opt 3: When buffers are disjoint, skip the global ordering
            # constraint. Only enforce when buffers overlap.
            if relax_producer_boundary and not (prev_bufs & curr_bufs):
                continue

            # Find last consumer that reads from prev_producer's buffers
            last_prev_reader = None
            for ci in consumer_indices:
                c_info = infos[ci]
                if any(rd.buffer.name in prev_bufs for rd in c_info.reads):
                    last_prev_reader = ci

            # Find first consumer that reads from curr_producer's buffers
            first_curr_reader = None
            for ci in consumer_indices:
                c_info = infos[ci]
                if any(rd.buffer.name in curr_bufs for rd in c_info.reads):
                    first_curr_reader = ci
                    break

            if last_prev_reader is not None and first_curr_reader is not None:
                _add_consumer_dep(first_curr_reader, last_prev_reader)

    # barrier_infos groups barrier-touching statements by the actual barrier
    # expression.  Preserve the per-barrier program order in deps_all so the
    # full graph contains explicit expect/load/arrive/wait dependencies even
    # when those calls do not expose normal buffer read/write regions.
    for _, raw_ops in (barrier_infos or {}).items():
        if isinstance(raw_ops, tuple) and len(raw_ops) >= 2 and isinstance(raw_ops[0], str):
            raw_iter = [raw_ops]
        else:
            raw_iter = list(raw_ops)

        barrier_ops: List[Tuple[int, str]] = []
        for item in raw_iter:
            if not isinstance(item, tuple) or len(item) < 2:
                continue
            op_type, stmt_idx = item[0], item[1]
            try:
                idx = int(stmt_idx)
            except (TypeError, ValueError):
                continue
            if idx in deps_all:
                barrier_ops.append((idx, str(op_type)))

        barrier_ops.sort(key=lambda x: x[0])
        for pos, (idx, _) in enumerate(barrier_ops):
            for prev_idx, _ in barrier_ops[:pos]:
                _add_all_dep(idx, prev_idx)

    return consumer_indices, deps, all_indices, deps_all


def _solve_naive_modulo_sched(op_deps: Dict[int, List[int]], infos: List['_StmtInfo'], op_indices: List[int]) -> List[Dict]:
    from ortools.sat.python import cp_model
    from heddle.scheduler.smt import ResourceType
    print('enter _solve_naive_modulo_sched', flush=True)
    
    ops: List[int] = list(op_indices)
    if not ops:
        return None

    # duration 代表发射占用时长（Issue Cycle），单发射模型下统一为 1
    duration = { key : 1 for key in ops }
    
    # 硬件发射槽位容量模型 （TMA暂且认为无发射限制。其受带宽影响）
    # H100 一个SM有4个subcore，每个SM上有: 1TMA, 4Tensorcore(但在实际使用时，其需要4个warp协作，一般需要跨subcore), 故总体建模为1个；
    capacity = { "TMA": 255, "TC": 1, "ALU": 64, "SFU": 16, "BARRIER": 1 }
    
    # latencies - 指令执行耗时
    latencies = {} 
    rrt = {}
    estimated_total_latency = 0

    def _issue_delay_for_self_edge(idx: int) -> int:
        # 跨迭代 self hazard 在这里表达的是发射/顺序间隔。
        # 如果对这类边使用完整数据就绪 latency，会强行要求
        # I >= latency，导致后续 joint SMT refine 根本没有机会运行。
        return max(int(duration.get(idx, 1)), 1)
    
    for idx in ops:
        info = infos[idx]
        if info.is_wait_barrier:
            # mbarrier_wait 使用一个独立同步 issue slot；它不能和任意
            # 其它 op 同槽发射，但不能伪装成占满所有 FU。
            rrt[info.idx] = [{"BARRIER": 1}]
            latencies[info.idx] = 1  # 屏障等待本身阻塞发射或紧邻同步，设为 1
            estimated_total_latency += 1
        else:
            if info.is_wgmma:
                issue_cycles = _detect_wgmma_issue_cycles(info.stmt)
                duration[info.idx] = issue_cycles  # 指令发射所占的时钟周期
                rrt[info.idx] = [{"TC": 1} for _ in range(issue_cycles)]
                latency, _ = _detect_op_latency_and_resource(info.stmt)
            else:
                latency, rty = _detect_op_latency_and_resource(info.stmt)
                issue_cycles = _detect_op_issue_cycles(info.stmt)
                duration[info.idx] = issue_cycles  # 指令发射所占的时钟周期
                rrt[info.idx] = [{rty.value : 1} for _ in range(issue_cycles)]  # 展开 rrt为 issue 周期数对应的表
            latencies[info.idx] = latency  # 记录 op总体的 issue+execute 延迟
            estimated_total_latency += latency


    print(f"-------- {latencies=}")
    print(f"-------- {duration=}", flush=True)
    
    def solve_for_I(I: int):
        # ---------------------------------------------------------
        # 【修改点 2】修正数据依赖与跨循环依赖的边权重（使用真实时延）
        # ---------------------------------------------------------
        # Edges: (producer, consumer, latency d, iteration distance δ)
        edges = []
        H = estimated_total_latency
        for v, deps in op_deps.items():
            if v not in ops:
                continue
            for u in deps:
                if u not in ops:
                    continue
                # 消费者 v 必须等生产者 u 的真实计算时延（latencies[u]）结束后才能发射
                edges.append((u, v, latencies[u], 0))
                
            # 跨循环依赖关系: 如果读写同一 buffer 则自己存在跨循环依赖
            write_buf_names = set()
            read_buf_names = set()
            for buf in infos[v].writes:
                write_buf_names.add(buf.buffer.name)
            for buf in infos[v].reads:
                read_buf_names.add(buf.buffer.name)
            intersect = write_buf_names & read_buf_names
            if intersect:
                edges.append((v, v, _issue_delay_for_self_edge(v), 1))
            if infos[v].is_sync_top and infos[v].is_sync_nested:
                edges.append((v, v, _issue_delay_for_self_edge(v), 1))

        model = cp_model.CpModel()

        # M[v] = scheduled time of op v (发射槽位时间戳).
        # 这里直接使用 IntVar 表达一次出现的位置。旧实现为每个
        # (op, absolute_time) 建 BoolVar；WGMMA issue duration 展开后会导致
        # H * duration * I 级别的资源约束爆炸。
        M = {}
        for v in ops:
            latest_start = max(0, H - duration[v])
            M[v] = model.NewIntVar(0, latest_start, f"M_{v}")

        # Symmetry breaking
        model.Add(M[ops[0]] == 0)
        
        # Dependency constraints:
        # M[v] - M[u] + δ*I >= d (此时 d 已经是真实的硬件 latency)
        for u, v, d, delta in edges:
            model.Add(M[v] - M[u] + delta * I >= d)

        # Modular resource capacity constraints.
        # 对 cap=1 且 reservation 连续的 issue 资源，用“相位区间在
        # 模 I 环上不重叠”表达，避免逐时间点枚举 Bool。WGMMA 和 SFU
        # 都会连续占用多个 issue slot。
        phase = {}
        for v in ops:
            phase[v] = model.NewIntVar(0, I - 1, f"phase_{v}")
            model.AddModuloEquality(phase[v], M[v], I)

        def _add_modular_no_overlap(u: int, v: int, tag: str) -> bool:
            du = duration[u]
            dv = duration[v]
            if du > I or dv > I or du + dv > I:
                return False

            # delta_uv = (phase[v] - phase[u]) mod I.
            # 两个环形 issue interval [u, u+du) 与 [v, v+dv) 不重叠：
            #   du <= delta_uv <= I - dv
            delta_uv = model.NewIntVar(0, I - 1, f"delta_{tag}_{u}_{v}")
            model.AddModuloEquality(delta_uv, phase[v] - phase[u] + I, I)
            model.Add(delta_uv >= du)
            model.Add(delta_uv <= I - dv)
            return True

        for resource_name, cap in capacity.items():
            if int(cap) != 1:
                continue
            resource_ops = [
                v for v in ops
                if any(per_cycle.get(resource_name, 0) for per_cycle in rrt[v])
            ]
            for pos, u in enumerate(resource_ops):
                du = duration[u]
                if du > I:
                    return None
                for v in resource_ops[pos + 1:]:
                    dv = duration[v]
                    if dv > I or du + dv > I:
                        return None

                    if not _add_modular_no_overlap(u, v, resource_name):
                        return None

        # Barrier issue slot exclusivity.
        # wait/try_wait barrier 必须单独占一个 modulo issue slot：同一个
        # phase 上不能有任何其它 op 的 issue interval 覆盖它。这个约束
        # 只表达发射槽独占，不再通过占满 TMA/TC/ALU/SFU 来间接实现。
        barrier_ops = [
            v for v in ops
            if getattr(infos[v], "is_wait_barrier", False)
        ]
        barrier_exclusive_pairs = set()
        for b in barrier_ops:
            for v in ops:
                if v == b:
                    continue
                u0, v0 = (b, v) if b < v else (v, b)
                if (u0, v0) in barrier_exclusive_pairs:
                    continue
                barrier_exclusive_pairs.add((u0, v0))
                if not _add_modular_no_overlap(u0, v0, "barrier_slot"):
                    return None
        
        # Schedule length L = max(M[v] + duration[v]).
        # 注：如果你希望 L 代表全流水线完全排空（包含最后一条指令执行完）的长度，
        # 可以把这里的 duration[v] 替换为 latencies[v]。
        # 目前保持 duration[v] 代表“所有指令发射完毕所需的总周期数”。
        end = {}
        for v in ops:
            end[v] = model.NewIntVar(0, H + duration[v], f"end_{v}")
            model.Add(end[v] == M[v] + duration[v])

        L = model.NewIntVar(0, H + max(duration.values()), "L")
        model.AddMaxEquality(L, [end[v] for v in ops])

        BIG = 100
        model.Minimize(BIG * L - sum(M[v] for v in ops))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        result = {
            "I": I,
            "L": solver.Value(L),
            "M": {v: solver.Value(M[v]) for v in ops},
        }

        # Build modular RRT table for printing.
        table = []
        for r in range(I):
            modified = False
            row = {"slot": f"{r} mod {I}"}
            for f in capacity:
                row[f] = []
            for v in ops:
                t = result["M"][v]
                for c in range(duration[v]):
                    slot = (t + c) % I
                    if slot == r:
                        for f in capacity:
                            if rrt[v][c].get(f, 0):
                                row[f].append(v)
                                modified = True
            if modified:
                table.append(row)

        result["modular_rrt"] = table
        return result

    def solve_min_I(max_I: int = 6) -> List:
        rets = []
        print(f'{estimated_total_latency=}') 
        lb = 1
        ub = estimated_total_latency
        ans_ub = None
        ans_lb = None
        last_lb = lb
        while True :
            print(f"\r[Modulo Sched] Testing Initiation Interval: {lb=},{ub=} ...", end="", flush=True)

            if ans_ub is None:
                ans_ub = solve_for_I(ub)
            if ans_lb is None: 
                ans_lb = solve_for_I(lb)
            assert ans_ub is not None 
            if ans_lb is None :
                if ub-lb <= 10 :
                    break
                if lb > last_lb :
                    last_lb = lb
                lb = (ub + lb) // 2
            else:
                ub = lb
                lb = last_lb
                ans_ub = ans_lb
                ans_lb = None

        
        for i in range(lb,ub+1) :
            print(f"\r[Modulo Sched] Testing Initiation Interval: I = {i}/{max_I} ...", end="", flush=True)
            ans = solve_for_I(i)
            if ans is not None :
                rets.append(ans);break
                    
        # for I in range(1, max_I + 1):
        #     print(f"\r[Modulo Sched] Testing Initiation Interval: I = {I}/{max_I} ...", end="", flush=True)
        #     ans = solve_for_I(I)
        #     if ans is not None and ans["I"] < ans["L"]:
        #         rets.append(ans)
        #         break
        return rets

    results = None
    try:
        results = solve_min_I(max(estimated_total_latency, 1))
    except Exception as e:
        print(e, flush=True)
        results = []
        
    print('solve done')
    if results is not None:
        print('ans not None')
        for ans in results:
            print("I =", ans["I"], flush=True)
            print("L =", ans["L"], flush=True)
            print("M =", ans["M"], flush=True)
            print("Modular RRT:")
            for row in ans["modular_rrt"]:
                print(row)
    else:
        print('ans none')
    return results

def _topo_sort_with_priority(
    consumer_indices: List[int],
    deps: Dict[int, List[int]],
    priorities: Dict[int, int],
    infos: Optional[List["_StmtInfo"]] = None,
    reg_limit: int = 0,
) -> List[int]:
    """Topological sort of consumer indices, breaking ties by priority.

    When *infos* and *reg_limit* are provided, the sort becomes
    register-pressure-aware: among ready nodes with equal priority,
    it prefers nodes whose scheduling would **release** the most
    register bytes (i.e. nodes that are the last reader of a live
    buffer).  This is a lightweight O(N log N) approximation of the
    Phase-B C6 register-capacity constraint.

    Lower priority value = scheduled earlier.
    """
    import heapq

    in_degree: Dict[int, int] = {idx: 0 for idx in consumer_indices}
    children: Dict[int, List[int]] = {idx: [] for idx in consumer_indices}

    for idx, dep_list in deps.items():
        for src in dep_list:
            children[src].append(idx)
            in_degree[idx] += 1

    # --- Resource-awareness bookkeeping ---
    # For each shared buffer written by a producer, track which consumers
    # read it and how many unscheduled readers remain.  When a consumer is
    # the *last* reader of a buffer, scheduling it "releases" that buffer's
    # register footprint.
    buf_readers: Dict[str, Set[int]] = {}        # buf_name -> set of reader indices
    buf_remaining: Dict[str, int] = {}            # buf_name -> unscheduled reader count
    buf_footprint: Dict[str, int] = {}            # buf_name -> estimated bytes
    node_reads_bufs: Dict[int, Set[str]] = {}     # consumer idx -> set of buf names it reads
    resource_aware = (infos is not None and reg_limit > 0)

    if resource_aware:
        consumer_set = set(consumer_indices)
        for ci in consumer_indices:
            read_bufs: Set[str] = set()
            for rd in infos[ci].reads:
                bname = rd.buffer.name
                read_bufs.add(bname)
                if bname not in buf_readers:
                    buf_readers[bname] = set()
                    buf_footprint[bname] = _estimate_buffer_footprint_bytes(rd.buffer)
                buf_readers[bname].add(ci)
            node_reads_bufs[ci] = read_bufs
        for bname, readers in buf_readers.items():
            buf_remaining[bname] = len(readers)

    def _release_score(idx: int) -> int:
        """Bytes released if *idx* is scheduled now (negative = releases more)."""
        if not resource_aware:
            return 0
        released = 0
        for bname in node_reads_bufs.get(idx, ()):
            if buf_remaining.get(bname, 0) == 1:  # last reader
                released += buf_footprint.get(bname, 0)
        return -released  # negative so min-heap prefers larger release

    # Min-heap: (priority, release_score, original_position, idx)
    ready: list = []
    for idx in consumer_indices:
        if in_degree[idx] == 0:
            heapq.heappush(ready, (priorities.get(idx, 0), _release_score(idx), idx, idx))

    result: List[int] = []
    while ready:
        _, _, _, idx = heapq.heappop(ready)
        result.append(idx)

        # Update remaining reader counts
        if resource_aware:
            for bname in node_reads_bufs.get(idx, ()):
                if bname in buf_remaining:
                    buf_remaining[bname] -= 1

        for child in children[idx]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                heapq.heappush(ready, (
                    priorities.get(child, 0),
                    _release_score(child),
                    child,
                    child,
                ))

    return result


def _compute_asap_times(
    infos: List[_StmtInfo],
    consumer_indices: List[int],
    deps: Dict[int, List[int]],
    *,
    use_precise_latency: bool = False,
    debug: bool = False,
) -> Optional[Dict[int, int]]:
    """Compute ASAP (As Soon As Possible) times for consumer statements.

    Uses Bellman-Ford longest-path on the dependency graph with latency
    weights. This is NOT modulo scheduling — it computes the ASAP time
    for a single iteration, which determines optimal statement ordering.

    Returns:
        Dict mapping consumer stmt_idx -> ASAP time, or None if no reordering is needed.
    """
    if len(consumer_indices) <= 1:
        return None

    # Compute latency for each consumer
    latency: Dict[int, int] = {}
    for ci in consumer_indices:
        info = infos[ci]
        lat, _ = _detect_op_latency_and_resource(info.stmt)
        if not use_precise_latency:
            lat = 1
        latency[ci] = lat

    # Bellman-Ford longest-path ASAP
    asap: Dict[int, int] = {ci: 0 for ci in consumer_indices}

    # Iterate until convergence (max |V| iterations)
    for _ in range(len(consumer_indices)):
        changed = False
        for ci in consumer_indices:
            for dep_idx in deps[ci]:
                new_time = asap[dep_idx] + latency[dep_idx]
                if new_time > asap[ci]:
                    asap[ci] = new_time
                    changed = True
        if not changed:
            break

    if debug:
        print(f"[Heddle] ASAP times: { {f's{k}': v for k, v in sorted(asap.items())} }", file=sys.stderr, flush=True)

    # Check if reordering would change anything
    # If ASAP times are monotonically increasing in original order, no reorder needed
    sorted_by_asap = sorted(consumer_indices, key=lambda ci: (asap[ci], ci))
    if sorted_by_asap == consumer_indices:
        if debug:
            print("[Heddle] ASAP order matches original order, no reorder needed", file=sys.stderr, flush=True)
        return None

    return asap


# ---------------------------------------------------------------------------
# Opt 1: ALAP (As Late As Possible) computation
# ---------------------------------------------------------------------------

def _compute_alap_times(
    infos: List[_StmtInfo],
    consumer_indices: List[int],
    deps: Dict[int, List[int]],
    asap: Dict[int, int],
    *,
    use_precise_latency: bool = False,
    debug: bool = False,
) -> Dict[int, int]:
    """Compute ALAP (As Late As Possible) times for consumer statements.

    Uses reverse Bellman-Ford: propagate from sinks backward.
    ALAP[v] = makespan - reverse_longest_path_from_v.

    The makespan is the maximum (ASAP + latency) over all consumers,
    representing the earliest possible completion time of the iteration.
    """
    # Compute latency for each consumer
    latency: Dict[int, int] = {}
    for ci in consumer_indices:
        info = infos[ci]
        lat, _ = _detect_op_latency_and_resource(info.stmt)
        if not use_precise_latency:
            lat = 1
        latency[ci] = lat

    # Makespan = max(ASAP[v] + latency[v]) over all consumers
    makespan = max(asap[ci] + latency[ci] for ci in consumer_indices)

    # Build reverse dependency graph: children[ci] -> list of ci's that depend on ci
    children: Dict[int, List[int]] = {ci: [] for ci in consumer_indices}
    for ci in consumer_indices:
        for dep_idx in deps[ci]:
            if dep_idx in children:
                children[dep_idx].append(ci)

    # Reverse Bellman-Ford: ALAP[v] = makespan - latency[v], then tighten
    # ALAP[v] = min over children c of (ALAP[c] - latency[v])
    alap: Dict[int, int] = {ci: makespan - latency[ci] for ci in consumer_indices}

    for _ in range(len(consumer_indices)):
        changed = False
        for ci in consumer_indices:
            for child_idx in children[ci]:
                new_time = alap[child_idx] - latency[ci]
                if new_time < alap[ci]:
                    alap[ci] = new_time
                    changed = True
        if not changed:
            break

    if debug:
        slack = {ci: alap[ci] - asap[ci] for ci in consumer_indices}
        print(f"[Heddle] ALAP times: { {f's{k}': v for k, v in sorted(alap.items())} }", file=sys.stderr, flush=True)
        print(f"[Heddle] Slack: { {f's{k}': v for k, v in sorted(slack.items())} }", file=sys.stderr, flush=True)

    return alap


# ---------------------------------------------------------------------------
# Opt 2: Buffer-span-aware priority adjustment
# ---------------------------------------------------------------------------

def _compute_buffer_span_priorities(
    infos: List[_StmtInfo],
    consumer_indices: List[int],
    base_priorities: Dict[int, int],
    *,
    use_precise_latency: bool = False,
    debug: bool = False,
) -> Dict[int, int]:
    """Adjust priorities to release producer-buffer slots as early as possible.

    Rationale (redesigned):
    In a steady-state software-pipelined loop, a shared-buffer slot can only
    be overwritten by the next-iteration producer after the current
    iteration's *last reader* has consumed it. The sooner the last reader
    runs, the sooner the slot frees and the sooner the next TMA can fire.
    PCWS derives arrive_insert_pos from the new SeqStmt order as
    ``last_access + 1``, so pulling the last reader earlier in the consumer
    sequence directly translates to an earlier mbarrier.arrive.

    The previous implementation *delayed the first reader* on the theory
    that this gave TMA more head-room. That is a single-shot assumption —
    in a pipelined loop the current-iter TMA is expected to have completed
    during the previous iteration, so delaying consumers only loses overlap
    against the following iteration's producer.

    New policy: give the *last reader* of each shared buffer a small
    priority bonus so it is scheduled at its ASAP time, even when the
    topological sort would otherwise push it to ALAP. Sync statements
    and critical-path nodes are left untouched.
    """
    producer_bufs: Dict[str, List[int]] = {}
    producer_buf_set: Set[str] = set()

    for info in infos:
        if info.is_producer:
            for wr in info.writes:
                if _is_shared(wr.buffer):
                    producer_buf_set.add(wr.buffer.name)

    for ci in consumer_indices:
        for rd in infos[ci].reads:
            if rd.buffer.name in producer_buf_set:
                producer_bufs.setdefault(rd.buffer.name, []).append(ci)

    if not producer_bufs:
        return base_priorities

    adjusted = dict(base_priorities)
    changes: Dict[int, int] = {}

    # For each shared producer buffer, find its last reader by original idx
    # and hoist it by subtracting a bonus proportional to its slack. Using a
    # small fixed bonus (1) rather than a full-slack shift preserves RAW
    # ordering between readers of the same buffer; the topo sort breaks ties
    # by original index when priorities match.
    for buf_name, readers in producer_bufs.items():
        if not readers:
            continue
        last_reader = max(readers)
        info = infos[last_reader]
        if info.is_sync_top or info.is_sync_nested:
            continue
        old = adjusted[last_reader]
        # Subtract 1 to ensure the last reader beats any non-last-reader at
        # the same ALAP level. We clamp to 0 to avoid negative priorities
        # that would confuse the topo-sort tie-breaker.
        new = max(0, old - 1)
        if new != old:
            adjusted[last_reader] = new
            changes[last_reader] = new - old

    if debug and changes:
        print(f"[Heddle] Buffer-span (last-reader hoist): "
              f"{ {f's{k}': v for k, v in changes.items()} }",
              file=sys.stderr, flush=True)

    return adjusted

def _solve_smt_joint_optimize(
    op_deps : Dict[int, List[int]], infos : List[_StmtInfo], all_indices : List[int], mod_sched_plan : Dict
):
    '''
    参照 _phase_b_consumer_ordering 实现 SMT 求解 optimized 模调度方案.
    [输入] 基础模调度方案 mod_sched_plan, 来自 _solve_naive_modulo_sched . mod_sched_plan含 ans["M"] ans["I"] ans["L"];  
        ans["M"] 表达单次循环内， opId:slot 位置分布
        ans['I'] 为最小启动周期； ans['L'] 为循环体长度  还有一个表格 用于展示调度结果
        op依赖关系 op_deps ; 
        all_indices 所有op的id ; 
        op信息 infos
    [输出] optimized_sched_plan 
    [过程] 参考  _phase_b_consumer_ordering ，用SMT。添加 warp_assign,  FU capacity, memory capacity 约束。
    '''
    from heddle.scheduler.smt import (
        HeddleScheduler,
        LifetimeSemantic,
        OpNode,
        OutputValue,
        ResourceType,
        StorageKind,
    )

    if not mod_sched_plan:
        return None

    ops = list(all_indices)
    if not ops:
        return None

    base_M = dict(mod_sched_plan.get("M", {}))
    try:
        base_I = int(mod_sched_plan.get("I", 0))
    except (TypeError, ValueError):
        base_I = 0
    try:
        base_L = int(mod_sched_plan.get("L", 0))
    except (TypeError, ValueError):
        base_L = 0
    if base_I <= 0:
        return dict(mod_sched_plan)

    _rtype_map = {
        "TMA": ResourceType.TMA,
        "TC": ResourceType.TensorCore,
        "ALU": ResourceType.ALU,
        "SFU": ResourceType.SFU,
        "BARRIER": ResourceType.Barrier,
    }
    # TC指令为跨subcore协作指令，无法多发射；其他指令，如果 subcoreId = warpId % 4 相同，则不能多发射，否则可以同时发射
    capacity = {
        ResourceType.TMA: 255,
        ResourceType.TensorCore: 1,
        ResourceType.ALU: 64,
        ResourceType.SFU: 16,
        ResourceType.Barrier: 1,
    }
    
    def _get_warpgroup_count_from_info() :
        warps = {
            "tma" : 0,
            "wgmma_consumer" : 0,
            "alu_consumer" : 4  # 暂且认为 ALU consumer 使用一个warpgroup
        }
        for info in infos :
            if info.is_wgmma :
                warps["wgmma_consumer"] = 4
            if info.is_true_tma :
                warps["tma"] = 4  # tma copy global->shm 暂且认为是 wg级别的？
        return  warps['tma'] +warps['wgmma_consumer'] +warps['alu_consumer']
    
    def _resource_for_info(info: _StmtInfo) -> ResourceType:
        if getattr(info, "is_wait_barrier", False):
            return ResourceType.Barrier
        if getattr(info, "is_true_tma", False):
            return ResourceType.TMA
        _, rty = _detect_op_latency_and_resource(info.stmt)
        return _rtype_map.get(getattr(rty, "value", str(rty)), ResourceType.ALU)

    def _latency_for_info(info: _StmtInfo) -> int:
        # Keep the joint solver at the same abstraction level as
        # _solve_naive_modulo_sched: TMA and wait barriers consume one issue
        # slot here. Their long data-ready / synchronization semantics are
        # represented by explicit barrier ordering and resource constraints,
        # not by turning every dependent edge into a long latency edge.
        if getattr(info, "is_wait_barrier", False) or getattr(info, "is_true_tma", False):
            return 1
        latency, _ = _detect_op_latency_and_resource(info.stmt)
        return max(int(latency), 1)

    def _dependency_delay_for_info(info: _StmtInfo) -> int:
        if getattr(info, "is_wait_barrier", False):
            return 1
        latency, _ = _detect_op_latency_and_resource(info.stmt)
        return max(int(latency), 1)

    def _issue_delay_for_self_edge(info: _StmtInfo) -> int:
        if getattr(info, "is_wait_barrier", False):
            return 1
        if getattr(info, "is_wgmma", False):
            return max(int(_detect_wgmma_issue_cycles(info.stmt)), 1)
        return max(int(_detect_op_issue_cycles(info.stmt)), 1)

    def _reservation_for_info(info: _StmtInfo, rty: ResourceType) -> List[Dict[ResourceType, int]]:
        # wait/try_wait barrier 会阻塞当前发射 warp；这里把它建模成
        # 一个同步 issue slot。不要按 ALU/SFU 总容量填满，否则长 SFU/ALU
        # reservation 折叠到整个 II 时，会把所有 wait barrier 都判成不可行。
        if getattr(info, "is_wait_barrier", False):
            return [{ResourceType.Barrier: 1}]
        if getattr(info, "is_wgmma", False):
            issue_cycles = _detect_wgmma_issue_cycles(info.stmt)
            return [{ResourceType.TensorCore: 1} for _ in range(issue_cycles)]
        issue_cycles = _detect_op_issue_cycles(info.stmt)
        return [{rty: 1} for _ in range(max(int(issue_cycles), 1))]

    nodes: List[OpNode] = []
    node_by_idx: Dict[int, OpNode] = {}
    for idx in ops:
        info = infos[idx]
        rty = _resource_for_info(info)
        latency = _latency_for_info(info)
        outputs: List[OutputValue] = []
        for wr in info.writes:
            storage = StorageKind.SMEM if _is_shared(wr.buffer) else StorageKind.RMEM
            footprint = _estimate_buffer_footprint_bytes(wr.buffer)
            outputs.append(OutputValue(
                name=f"s{idx}_w_{wr.buffer.name}",
                storage=storage,
                footprint_bytes=footprint,
                lifetime=LifetimeSemantic.DEAD_ON_ENTRY,
            ))
        
        need_warpgroup = info.is_wgmma
        single_warp_eligible = info.is_true_tma
        wc = 4 if need_warpgroup else 1  # producer consumer 都按WG安排；barrier可能需要特殊处理
        _is_variable_latency = info.is_true_tma  # 只有真正的 TMA load 按 variable latency 建模
        node = OpNode(
            name=f"s{idx}",
            resource_type=rty,
            latency=latency,
            reservation=_reservation_for_info(info, rty),
            outputs=outputs,
            warp_count=wc,
            warp_align=4 if need_warpgroup else 1,
            is_varialble_latency=_is_variable_latency,
            replicable=single_warp_eligible
        )
        nodes.append(node)
        node_by_idx[idx] = node

    # 同一轮迭代内的依赖边，来自 IR 分析得到的 op_deps 图。
    for v, deps in op_deps.items():
        if v not in node_by_idx:
            continue
        for u in deps:
            if u not in node_by_idx:
                continue
            # deps_all already contains the conservative ordering needed for
            # sync/barrier statements. Do not upgrade every sync parent edge to
            # HeddleScheduler.blocking_sync: that additionally enforces same
            # warp and a long active-window exclusion, which is stronger than
            # the issue-slot model used by the modulo scheduler and makes real
            # producer/consumer graphs presolve UNSAT.
            node_by_idx[v].add_dependency(
                node_by_idx[u],
                distance=0,
                delay=_dependency_delay_for_info(infos[u]),
            )

    # 跨迭代 hazard 与 _solve_naive_modulo_sched 保持一致：
    # 如果一个 stmt 同时读写同一个 buffer，或者它是 sync-like 阻塞语句，
    # 那么相邻两轮迭代中的该 stmt 必须至少间隔一个依赖延迟。
    for idx in ops:
        info = infos[idx]
        write_bufs = {wr.buffer.name for wr in info.writes}
        read_bufs = {rd.buffer.name for rd in info.reads}
        if write_bufs & read_bufs:
            node_by_idx[idx].add_dependency(
                node_by_idx[idx],
                distance=1,
                delay=_issue_delay_for_self_edge(info),
            )
        if info.is_wait_barrier :
            node_by_idx[idx].add_dependency(
                node_by_idx[idx],
                distance=1,
                delay=_issue_delay_for_self_edge(info),
            )
        # 一致性约束 : op[v,0,t] => op[v,i,t+i*II]  应用于所有op。不再限制 自读写 & sync语义
        # 关键点：op[v,0,t] => op[v,i,t+i*II] 这种“相位一致性”在当前 joint SMT 表达里已经由“每个 op 一个 Tv[v]，并固定 ii=base_I”隐含表示了；它不是 v -> v, distance=1, delay=latency 这种依赖边。后者表达的是“下一轮同一个 op 必须等上一轮这个 op 的结果/资源 hazard 结束”，只应该用于真实 loop-carried hazard，比如自读写 buffer 或 sync-like 语义。
        #  node_by_idx[idx].add_dependency(
        #     node_by_idx[idx],
        #     distance=1,
        #     delay=_dependency_delay_for_info(info),
        # )
        
    nwarps = _get_warpgroup_count_from_info()
    mod_sched_plan['num_warps'] = nwarps
    print(f'---- num_warps = {nwarps}')
    num_warps = max(1, int(mod_sched_plan.get("num_warps", 1)))
    reg_limit = int(mod_sched_plan.get("reg_limit", 32* 240 * 4)) # 单个线程 255个 f32 寄存器；每个warp内需*32，阈值设置略低于 255
    smem_limit = int(mod_sched_plan.get("smem_limit", 227 * 1024))  # h100 : 227 Kbytes for CTA
    smem_output_floor = sum(
        out.footprint_bytes
        for node in nodes
        for out in node.outputs
        if out.storage == StorageKind.SMEM and out.footprint_bytes > 0
    )
    if smem_output_floor > smem_limit:
        print(
            f"---- SHM 到达 smem_limit 限制（未考虑复用的简单求和）: {smem_limit} -> {smem_output_floor}",
            flush=True,
        )
        smem_limit = smem_output_floor
        print(f"--- 调整 smem_limit 为 {smem_limit}")

    # SMT 优化器使用 naive plan 的基础窗口；如果 naive 的绝对调度时间
    # 比记录的 L 更宽，则补一点 slack，避免可行解被窗口截断。
    base_max_time = max((int(base_M.get(v, 0)) for v in ops), default=0)
    window = max(base_L, base_max_time + 1, base_I)

    def _run_joint_solver(
        *, solve_window: int, optimize: bool,
        solve_reg_limit: int, solve_smem_limit: int,
    ):
        solver = HeddleScheduler(
            nodes,
            fu_caps=capacity,
            reg_limit=solve_reg_limit,
            smem_limit=solve_smem_limit,
            num_warps=num_warps,
            timeout_ms=int(mod_sched_plan.get("timeout_ms", 15000)),
            start_hints={
                f"s{idx}": int(t)
                for idx, t in base_M.items()
                if idx in node_by_idx
            },
        )
        return solver.schedule_joint(
            min_ii=base_I,
            max_ii=base_I,
            window=solve_window,
            optimize=optimize,
        )

    candidate_windows = []
    for candidate in (window, window + base_I, window + 2 * base_I):
        if candidate not in candidate_windows:
            candidate_windows.append(candidate)

    reg_limit_candidates = []
    for candidate in (reg_limit, reg_limit * 4, 10**9):
        if candidate > 0 and candidate not in reg_limit_candidates:
            reg_limit_candidates.append(candidate)

    smem_limit_candidates = []
    for candidate in (smem_limit, smem_limit * 2, 233472):
        if candidate > 0 and candidate not in smem_limit_candidates:
            smem_limit_candidates.append(candidate)

    sol = None
    solved_with_optimize = False
    used_reg_limit = reg_limit
    used_smem_limit = smem_limit
    for solve_window in candidate_windows:
        for solve_reg_limit in reg_limit_candidates:
            for solve_smem_limit in smem_limit_candidates:
                used_reg_limit = solve_reg_limit
                used_smem_limit = solve_smem_limit
                sol = _run_joint_solver(
                    solve_window=solve_window,
                    optimize=False,
                    solve_reg_limit=solve_reg_limit,
                    solve_smem_limit=solve_smem_limit,
                )
                if sol is not None:
                    break
                if solve_reg_limit != reg_limit or solve_smem_limit != smem_limit:
                    print(
                        f"---- SMT feasibility failed at window={solve_window}, "
                        f"reg_limit={solve_reg_limit}, smem_limit={solve_smem_limit}",
                        flush=True,
                    )
            if sol is not None:
                break
        if sol is not None:
            break

    if sol is None:
        print("---- SMT feasibility failed; retry optimize", flush=True)
        for solve_window in candidate_windows:
            for solve_reg_limit in reg_limit_candidates:
                for solve_smem_limit in smem_limit_candidates:
                    used_reg_limit = solve_reg_limit
                    used_smem_limit = solve_smem_limit
                    sol = _run_joint_solver(
                        solve_window=solve_window,
                        optimize=True,
                        solve_reg_limit=solve_reg_limit,
                        solve_smem_limit=solve_smem_limit,
                    )
                    if sol is not None:
                        solved_with_optimize = True
                        break
                    if solve_reg_limit != reg_limit or solve_smem_limit != smem_limit:
                        print(
                            f"---- SMT optimize failed at window={solve_window}, "
                            f"reg_limit={solve_reg_limit}, smem_limit={solve_smem_limit}",
                            flush=True,
                        )
                if sol is not None:
                    break
            if sol is not None:
                break
    if sol is None:
        fallback = dict(mod_sched_plan)
        fallback.setdefault("status", "SMT_UNSAT")
        return fallback
    if used_reg_limit != reg_limit:
        print(
            f"---- SMT joint succeeded with relaxed reg_limit={used_reg_limit} "
            f"(original={reg_limit})",
            flush=True,
        )
    if used_smem_limit != smem_limit:
        print(
            f"---- SMT joint succeeded with relaxed smem_limit={used_smem_limit} "
            f"(original={smem_limit})",
            flush=True,
        )

    schedule = sol.get("schedule", {})
    warp_assign = sol.get("warp_assign", {})
    print(f"----{warp_assign=}")
    optimized_M = {
        idx: int(schedule[f"s{idx}"])
        for idx in ops
        if f"s{idx}" in schedule
    }
    for idx in ops:
        if idx not in optimized_M and idx in base_M:
            optimized_M[idx] = int(base_M[idx])

    optimized_L = max(
        (
            t + max(len(node_by_idx[idx].reservation), 1)
            for idx, t in optimized_M.items()
            if idx in node_by_idx
        ),
        default=0,
    )
    print(f'---{optimized_L=}')
    print(f'---{base_I=}')
    print(f'---{optimized_M=}')

    table = []
    for r in range(base_I):
        row = {"slot": f"{r} mod {base_I}"}
        for f in ("TMA", "TC", "ALU", "SFU"):
            row[f] = []
        modified=False
        for idx in ops:
            t = optimized_M.get(idx)
            if t is None or idx not in node_by_idx:
                continue
            for c, per_cycle in enumerate(node_by_idx[idx].reservation):
                if (t + c) % base_I != r:
                    continue
                for rty, used in per_cycle.items():
                    if int(used):
                        row[rty.value].append(idx)
                        modified=True
        if modified:
            table.append(row)

    optimized = dict(mod_sched_plan)
    optimized.update({
        "I": base_I,
        "L": optimized_L,
        "M": optimized_M,
        "status": "SMT_OPTIMIZED" if solved_with_optimize else "SMT_FEASIBLE",
        "window": int(sol.get("window", window)),
        "warp_assign": {
            int(name[1:]): int(w)
            for name, w in warp_assign.items()
            if isinstance(name, str) and name.startswith("s") and name[1:].isdigit()
        },
        "reg_peak": sol.get("reg_peak", {}),
        "modular_rrt": table,
        "ordering": [idx for idx, _ in sorted(optimized_M.items(), key=lambda item: (item[1], item[0]))],
    })
    return optimized

# ---------------------------------------------------------------------------
# Phase B: SMT-based joint consumer ordering with register awareness
# ---------------------------------------------------------------------------

def _phase_b_consumer_ordering(
    infos: List[_StmtInfo],
    consumer_indices: List[int],
    deps: Dict[int, List[int]],
    *,
    use_precise_latency: bool = False,
    reg_limit: int = 960,
    timeout_ms: int = 15000,
    num_warps: int = 1,
    barrier_edges: Optional[Set[Tuple[int, int]]] = None,
    debug: bool = False,
) -> Optional[Tuple[List[int], Dict[str, int], Dict[str, int]]]:
    """Use Phase B (CP-SAT solver) to find an optimal consumer ordering.

    Unlike Phase A (ASAP heuristic), Phase B jointly optimizes:
    - Statement time-slot assignment (resolves FU resource conflicts)
    - Register liveness tracking (including incoming_live for loop-carried values)
    - Register capacity constraints per warp
    - Warp assignment (when num_warps > 1)
    - Blocking sync barrier constraints (when barrier_edges provided)

    Uses Google OR-Tools CP-SAT solver which is 100-1000x faster than Z3
    for scheduling problems (38ms vs 103s on FA BWD 21-consumer graphs).

    Args:
        barrier_edges: Set of (producer_idx, consumer_idx) pairs that require
            blocking synchronization (same warp + exclusive execution).

    Returns (consumer_ordering, schedule_times, warp_assigns) or None if UNSAT/timeout.
    schedule_times maps node name (e.g. "s3") to its start time in the schedule.
    warp_assigns maps node name to its warp assignment (int).
    Falls back to None so caller can use Phase A ordering instead.
    """
    from heddle.scheduler.cp_sat import (
        UnifiedScheduler, PartitionSpec, KernelSpec, OpSpec, OutputSpec,
        ResourceType, StorageKind,
    )

    if len(consumer_indices) <= 1:
        return None

    # Build OpSpec list from consumer statements
    ops: List[OpSpec] = []
    # Map from smt.ResourceType names to cp_sat.ResourceType
    _rtype_map = {
        "TMA": ResourceType.TMA,
        "TC": ResourceType.TensorCore,
        "ALU": ResourceType.ALU,
        "SFU": ResourceType.SFU,
    }

    for ci in consumer_indices:
        info = infos[ci]
        if use_precise_latency:
            latency, rty = _detect_op_latency_and_resource(info.stmt)
            cpsat_rtype = _rtype_map.get(rty.value, ResourceType.ALU)
        else:
            cpsat_rtype = ResourceType.ALU
            latency = 1

        # Build output specs with RMEM footprint
        outputs: List[OutputSpec] = []
        for wr in info.writes:
            storage = StorageKind.SMEM if _is_shared(wr.buffer) else StorageKind.RMEM
            fp = _estimate_buffer_footprint_bytes(wr.buffer)
            outputs.append(OutputSpec(
                name=f"s{ci}_w_{wr.buffer.name}",
                storage=storage,
                footprint_bytes=fp,
            ))

        # Build dependency list (distance=0, within iteration)
        op_deps = []
        for dep_idx in deps.get(ci, []):
            if dep_idx in set(consumer_indices):
                is_blocking = bool(barrier_edges and (dep_idx, ci) in barrier_edges)
                op_deps.append((f"s{dep_idx}", 0, is_blocking))

        ops.append(OpSpec(
            name=f"s{ci}",
            resource_type=cpsat_rtype,
            latency=latency,
            outputs=outputs,
            deps=op_deps,
        ))

        if debug and outputs:
            total_fp = sum(o.footprint_bytes for o in outputs)
            print(f"[Heddle PB] s{ci}: {len(outputs)} outputs, total_fp={total_fp}B "
                  f"rtype={cpsat_rtype.value} lat={latency} "
                  f"({', '.join(f'{o.name}={o.footprint_bytes}B' for o in outputs)})",
                  file=sys.stderr, flush=True)

    fu_caps = {
        ResourceType.TMA: 1, ResourceType.TensorCore: 1,
        ResourceType.ALU: 64, ResourceType.SFU: 16,
    }

    # Estimate horizon from total latency chain
    total_lat = sum(op.latency for op in ops)
    max_lat = max(op.latency for op in ops) if ops else 1
    horizon = total_lat + max_lat + 8

    # Wrap in single-kernel, single-partition for the unified solver
    kernel = KernelSpec("consumer_loop", ops)
    partition = PartitionSpec("single", [kernel])

    timeout_s = timeout_ms / 1000.0

    # Try with the given reg_limit first. If INFEASIBLE (common when the
    # kernel inherently requires spilling, e.g. FA BWD with 5 WGMMAs),
    # retry with a relaxed limit. The solver's objective still minimizes
    # register pressure, so we get the best feasible ordering.
    result = None
    for attempt_reg_limit in [reg_limit, reg_limit * 4, 0]:
        solver = UnifiedScheduler(
            [partition],
            fu_caps=fu_caps,
            reg_limit=attempt_reg_limit if attempt_reg_limit > 0 else 10**7,
            num_warps=num_warps,
            horizon=horizon,
            timeout_s=timeout_s,
        )
        result = solver.solve()
        if result is not None and result.status in ("OPTIMAL", "FEASIBLE"):
            if debug and attempt_reg_limit != reg_limit:
                print(f"[Heddle] Phase B (CP-SAT) INFEASIBLE at reg_limit={reg_limit}, "
                      f"succeeded with relaxed limit={attempt_reg_limit}",
                      file=sys.stderr, flush=True)
            break
        if debug:
            status = result.status if result else "None"
            print(f"[Heddle] Phase B (CP-SAT) {status} at reg_limit={attempt_reg_limit}",
                  file=sys.stderr, flush=True)

    if result is None or result.status not in ("OPTIMAL", "FEASIBLE"):
        if debug:
            print(f"[Heddle] Phase B (CP-SAT) failed for {len(consumer_indices)} consumers "
                  f"(horizon={horizon})",
                  file=sys.stderr, flush=True)
        return None

    times_b = result.kernel_schedules.get("consumer_loop", {})
    warp_assigns_b = result.kernel_warp_assigns.get("consumer_loop", {})
    reg_peak = result.kernel_reg_peaks

    if debug:
        print(f"[Heddle] Phase B (CP-SAT) succeeded: "
              f"makespan={result.total_makespan}, "
              f"solve_time={result.solve_time_ms:.1f}ms, "
              f"reg_peak={reg_peak}, status={result.status}",
              file=sys.stderr, flush=True)
        print(f"[Heddle] Phase B times: {times_b}", file=sys.stderr, flush=True)

    # Convert schedule times to consumer ordering
    name_to_idx = {f"s{ci}": ci for ci in consumer_indices}
    ordered = []
    for name, t in sorted(times_b.items(), key=lambda x: (x[1], x[0])):
        if name in name_to_idx:
            ordered.append(name_to_idx[name])

    # Ensure all consumers are included
    ordered_set = set(ordered)
    for ci in consumer_indices:
        if ci not in ordered_set:
            ordered.append(ci)

    return ordered, times_b, warp_assigns_b


def _extract_barrier_hints(
    infos: List[_StmtInfo],
    consumer_indices: List[int],
    consumer_ordering: List[int],
    times_b: Dict[str, int],
    *,
    use_precise_latency: bool = False,
    debug: bool = False,
) -> PhaseB_PCWS_Hint:
    """Extract barrier hints and stage offsets from Phase B solution.

    Uses the Phase B schedule times to compute optimal barrier positions:
    - For each producer buffer, find the first/last reader times in the schedule
    - Suggest wait position = first reader position in the ordering
    - Suggest arrive position = last reader position + 1 in the ordering
    - Compute delay = first_reader_time - estimated TMA latency

    Returns a PhaseB_PCWS_Hint with ordering, barrier_hints, and stage_offsets.
    """
    name_to_idx = {f"s{ci}": ci for ci in consumer_indices}
    idx_to_pos = {ci: pos for pos, ci in enumerate(consumer_ordering)}

    # Build buffer -> reader map
    buffer_readers: Dict[str, List[int]] = {}  # buffer_name -> [ci, ...]
    for ci in consumer_indices:
        info = infos[ci]
        for rd in info.reads:
            if _is_shared(rd.buffer):
                buf_name = rd.buffer.name
                if buf_name not in buffer_readers:
                    buffer_readers[buf_name] = []
                buffer_readers[buf_name].append(ci)
    if debug:
        _summary = {n: rs for n, rs in buffer_readers.items()}
        print(f"[Heddle] _extract_barrier_hints: buffer_readers={_summary}",
              file=sys.stderr, flush=True)

    # Build time -> position mapping from Phase B schedule
    time_to_pos: Dict[int, int] = {}
    for pos, ci in enumerate(consumer_ordering):
        key = f"s{ci}"
        if key in times_b:
            time_to_pos[times_b[key]] = pos

    hints: Dict[str, BarrierHint] = {}
    for buf_name, readers in buffer_readers.items():
        if not readers:
            continue
        # Find first and last reader in the consumer ordering
        reader_positions = sorted(idx_to_pos.get(ci, 999) for ci in readers)
        first_pos = reader_positions[0]
        last_pos = reader_positions[-1]

        # Get schedule times from Phase B
        reader_times = []
        reader_ci_times = []
        for ci in readers:
            key = f"s{ci}"
            if key in times_b:
                reader_times.append(times_b[key])
                reader_ci_times.append((ci, times_b[key]))

        if not reader_times:
            # No schedule data — use ordering positions as fallback
            hints[buf_name] = BarrierHint(
                buffer_name=buf_name,
                suggested_wait_pos=first_pos,
                suggested_arrive_pos=last_pos + 1,
                delay_cycles=0,
            )
            continue

        first_reader_time = min(reader_times)
        last_reader_time = max(reader_times)

        # Derive optimal wait position from schedule times.
        # The wait is placed BEFORE the compute_stmt at the returned position.
        # It must be ≤ first_pos (the first reader) to ensure data is ready.
        # The optimization: if earlier consumers don't need this buffer,
        # we can place the wait later (closer to first_pos) to hide latency.
        # But we never exceed first_pos.
        tma_latency_est = 25
        delay_cycles = max(0, first_reader_time - tma_latency_est)

        # The suggested wait position is the first_pos (safe default).
        # Phase B can suggest an earlier position if TMA completes early
        # enough that earlier compute can overlap with the transfer.
        # For now, keep wait at first_pos — the main optimization is in
        # the arrive position (allowing earlier buffer release).
        delayed_wait_pos = first_pos

        # Derive arrive position from last reader time + latency.
        # Find the first position after the last reader completes.
        last_reader_ci = max(reader_ci_times, key=lambda x: x[1])[0]
        last_reader_info = infos[last_reader_ci]
        if use_precise_latency:
            _, _ = _detect_op_latency_and_resource(last_reader_info.stmt)
        arrive_pos = idx_to_pos.get(last_reader_ci, last_pos) + 1

        hints[buf_name] = BarrierHint(
            buffer_name=buf_name,
            suggested_wait_pos=delayed_wait_pos,
            suggested_arrive_pos=arrive_pos,
            delay_cycles=delay_cycles,
        )

    if debug and hints:
        print(f"[Heddle] Barrier hints: {[(n, h.suggested_wait_pos, h.suggested_arrive_pos, h.delay_cycles) for n, h in hints.items()]}",
              file=sys.stderr, flush=True)

    return PhaseB_PCWS_Hint(
        consumer_ordering=consumer_ordering,
        barrier_hints=hints,
        stage_offsets={},  # Stage offsets require cross-stage Z3 extension
    )


# ---------------------------------------------------------------------------
# IR rewriting: reorder consumer statements in SeqStmt
# ---------------------------------------------------------------------------

def _reorder_loop_body(
    seq: tvm.tir.SeqStmt,
    infos: List[_StmtInfo],
    new_consumer_order: List[int],
) -> tvm.tir.SeqStmt:
    """Reconstruct the SeqStmt with producers hoisted and consumers reordered.

    Producer hoisting policy (redesigned):
    The previous policy placed each producer *just before* its first reader.
    For a pipelined loop this collapses TMA issue into the reader's critical
    path and wastes the async-overlap window. The new policy hoists every
    producer to the earliest position that still respects RAW/WAR
    dependencies on other in-loop statements:

        - No shared-buffer dependency on a later in-loop producer → hoist to
          the front of the body (right after any other front-hoisted producer).
        - Shared-buffer RAW from producer P2 to P1 (rare) → keep P2 after P1.

    This lets TMA intrinsics fire asynchronously at loop entry and overlap
    with the full consumer compute window, instead of stalling behind the
    serial WGMMA chain.
    """
    producer_indices = [info.idx for info in infos if info.is_producer]
    consumer_set = set(new_consumer_order)

    # Build producer-producer RAW chain (if any).
    # prod_deps[p] = set of producer indices P' whose shared write P reads.
    prod_deps: Dict[int, Set[int]] = {p: set() for p in producer_indices}
    prod_writes: Dict[int, Set[str]] = {}
    for p in producer_indices:
        prod_writes[p] = {w.buffer.name for w in infos[p].writes if _is_shared(w.buffer)}
    for p in producer_indices:
        p_reads = {r.buffer.name for r in infos[p].reads if _is_shared(r.buffer)}
        for p2 in producer_indices:
            if p2 == p:
                continue
            if prod_writes[p2] & p_reads:
                prod_deps[p].add(p2)

    # Topologically sort producers (keep original relative order for ties).
    hoisted_producers: List[int] = []
    remaining = list(producer_indices)
    while remaining:
        ready = [p for p in remaining if not (prod_deps[p] - set(hoisted_producers))]
        if not ready:
            # Cycle — fall back to original order
            hoisted_producers.extend(remaining)
            break
        next_p = ready[0]
        hoisted_producers.append(next_p)
        remaining.remove(next_p)

    # Consumer → producer anchor map for the rare case where a consumer reads
    # a buffer that is written by *another* producer we could not hoist.
    # This defensive path is unused when all producers are cleanly hoistable.
    producer_before: Dict[int, int] = {}
    for pi in producer_indices:
        if pi in hoisted_producers:
            continue
        p_written = prod_writes.get(pi, set())
        for pos, ci in enumerate(new_consumer_order):
            c_reads = {r.buffer.name for r in infos[ci].reads}
            if p_written & c_reads:
                producer_before[pi] = ci
                break

    new_stmts: List[tvm.tir.Stmt] = []
    # 1. All hoistable producers first (in topo order).
    for p in hoisted_producers:
        new_stmts.append(seq.seq[p])
    # 2. Consumers in the scheduled order, interleaved with any non-hoisted
    #    producer that must precede them.
    non_hoisted = [p for p in producer_indices if p not in hoisted_producers]
    for ci in new_consumer_order:
        to_insert = [p for p in non_hoisted if producer_before.get(p) == ci]
        for p in to_insert:
            new_stmts.append(seq.seq[p])
            non_hoisted.remove(p)
        new_stmts.append(seq.seq[ci])
    for p in non_hoisted:
        new_stmts.append(seq.seq[p])

    return tvm.tir.SeqStmt(new_stmts)


# ---------------------------------------------------------------------------
# Unified scheduling cost model
# ---------------------------------------------------------------------------

@dataclass
class SchedulingConfig:
    """A point in the scheduling configuration space."""
    async_wgmma: bool = False     # warpgroup_wait<1> + post-loop drain
    early_bp: bool = False        # bp_arrive before warpgroup_wait
    dual_consumer: bool = False   # WG0=QK+softmax, WG1=PV
    three_role: bool = False      # dedicated dQ writer warp


@dataclass
class SchedulingResult:
    """Output of the cost model."""
    config: SchedulingConfig
    T_producer: float   # estimated producer per-iter cycles
    T_consumer: float   # estimated consumer per-iter cycles
    T_iter: float       # max(T_producer, T_consumer)
    annotations: Dict[str, str]
    breakdown: str      # human-readable explanation


def _evaluate_scheduling_cost_model(
    infos_list: List[_StmtInfo],
    consumer_indices: List[int],
    consumer_wgmma_indices: List[int],
    has_non_tc_gap: bool,
    max_wgmma_out_elems: int,
    has_tma_reduce_add: bool,
    *,
    debug: bool = False,
) -> SchedulingResult:
    """Unified cost model for Hopper warp-specialized pipeline scheduling.

    Optimizes:  minimize  T_iter = max(T_producer, T_consumer)
    Subject to: correctness constraints (RAW/WAR on shared buffers)

    T_producer = bp_wait + TMA_latency + fwd_signal
    T_consumer = Σ(stmt_latency) + barrier_overhead - overlap_savings

    Each boolean in SchedulingConfig affects one or both terms. The model
    enumerates all *feasible* configurations and picks the one with minimum
    T_iter. Feasibility is determined by hardware constraints (WGMMA count,
    fragment size, atomic_add presence).

    Latency constants are calibrated on H100 SXM (SM90a, 1980 MHz).
    """
    # ── Hopper per-instruction latency (cycles @ 1980 MHz) ──
    WGMMA_ISSUE_PER_K = 27    # 1 wgmma_ss/rs issue (m64n128k16)
    WGMMA_ACCUM_BASE = 40     # accumulator write-back (overlappable)
    TMA_BYTES_PER_CYCLE = 640 # ~1 TB/s TMA at 1980 MHz
    BARRIER_WAIT = 5          # mbarrier try_wait (uncontested)
    BARRIER_SIGNAL = 1        # mbarrier arrive
    XFER_BARRIER = 25         # named barrier rendezvous (cross-WG)
    XFER_PER_ELEM = 0.25      # flat smem copy per fp16 element
    ATOMIC_ADD_LAT = 50       # TMA reduce-add issue + smem copy

    # ── IR-derived latency per consumer stmt ──
    n_wgmma = len(consumer_wgmma_indices)
    n_producers = sum(1 for i in infos_list if i.is_producer)

    # Estimate K-dim iterations from shared buffer shapes. For a WGMMA
    # stmt, one of its shared-memory read buffers has shape (*, K_tile);
    # K_ITERS = K_tile / 16 (fp16 wgmma k-dim = 16).
    def _wgmma_k_iters(info: _StmtInfo) -> int:
        for rd in info.reads:
            if _is_shared(rd.buffer) and len(rd.buffer.shape) == 2:
                try:
                    k_dim = int(rd.buffer.shape[-1])
                    return max(1, k_dim // 16)
                except (TypeError, ValueError):
                    pass
        return 4  # conservative default

    # Estimate TMA latency from producer buffer sizes
    total_tma_bytes = 0
    for info in infos_list:
        if info.is_producer:
            for wr in info.writes:
                if _is_shared(wr.buffer):
                    total_tma_bytes += _estimate_buffer_footprint_bytes(wr.buffer)
    tma_lat_per_producer = max(10, total_tma_bytes // max(1, n_producers) // TMA_BYTES_PER_CYCLE)

    stmt_lat: Dict[int, int] = {}
    for ci in consumer_indices:
        info = infos_list[ci]
        lat, _ = _detect_op_latency_and_resource(info.stmt)
        if info.is_wgmma:
            k_iters = _wgmma_k_iters(info)
            lat = WGMMA_ISSUE_PER_K * k_iters
        stmt_lat[ci] = lat

    consumer_serial = sum(stmt_lat.values())
    barrier_overhead = n_producers * (BARRIER_WAIT + BARRIER_SIGNAL)

    # Overlap window: sum of non-WGMMA stmts between last WGMMA and
    # first WGMMA (wrapping around the loop body). This is the ALU/barrier
    # work that can run while the last WGMMA's accumulation is in-flight.
    sorted_ci = sorted(consumer_indices)
    if n_wgmma >= 1:
        last_wgmma_pos = max(i for i, ci in enumerate(sorted_ci) if ci in consumer_wgmma_indices)
        first_wgmma_pos = min(i for i, ci in enumerate(sorted_ci) if ci in consumer_wgmma_indices)
        wrap_stmts = sorted_ci[last_wgmma_pos + 1:] + sorted_ci[:first_wgmma_pos]
        overlappable_work = sum(stmt_lat.get(ci, 0) for ci in wrap_stmts)
        overlappable_work += barrier_overhead  # barriers also overlap
    else:
        overlappable_work = 0

    # ── T_producer baseline (steady state) ──
    # Producer waits for bp, loads K/V via TMA, signals fwd.
    T_prod_base = BARRIER_WAIT + tma_lat_per_producer + BARRIER_SIGNAL

    # ── Enumerate feasible configurations ──
    configs: List[SchedulingConfig] = [SchedulingConfig()]  # baseline

    # async_wgmma: feasible when ≥1 WGMMA in loop body
    if n_wgmma >= 1:
        configs.append(SchedulingConfig(async_wgmma=True, early_bp=True))
        configs.append(SchedulingConfig(async_wgmma=True, early_bp=False))

    # dual_consumer: feasible when ≥2 WGMMAs + non-TC gap + full-tile fragment
    if n_wgmma >= 2 and has_non_tc_gap and max_wgmma_out_elems > 64:
        configs.append(SchedulingConfig(async_wgmma=True, early_bp=True,
                                        dual_consumer=True))

    # three_role: feasible when TMA atomic_add detected (FA BWD)
    if has_tma_reduce_add:
        configs.append(SchedulingConfig(async_wgmma=True, early_bp=True,
                                        three_role=True))

    # ── Evaluate each configuration ──
    results: List[SchedulingResult] = []
    for cfg in configs:
        # T_consumer
        T_cons = consumer_serial + barrier_overhead

        # async_wgmma: last WGMMA's accumulation overlaps with next iter's
        # barrier waits + init + non-WGMMA work (computed from IR above)
        overlap = 0
        if cfg.async_wgmma and n_wgmma >= 1:
            overlap = min(WGMMA_ACCUM_BASE, overlappable_work)
            T_cons -= overlap

        # three_role: offload atomic_add to dedicated warp
        if cfg.three_role:
            T_cons -= ATOMIC_ADD_LAT

        # dual_consumer: split consumer into 2 parallel WGs
        if cfg.dual_consumer:
            sorted_c = sorted(consumer_indices)
            wgmma_set = set(consumer_wgmma_indices)
            best_dual = float('inf')
            for ci in sorted_c:
                if ci not in wgmma_set:
                    continue
                nxt = next((c for c in sorted_c if c > ci), None)
                if nxt is None:
                    continue
                wg0 = [c for c in sorted_c if c < nxt]
                wg1 = [c for c in sorted_c if c >= nxt]
                if not any(c in wgmma_set for c in wg0):
                    continue
                if not any(c in wgmma_set for c in wg1):
                    continue
                t0 = sum(stmt_lat.get(c, 0) for c in wg0)
                t1 = sum(stmt_lat.get(c, 0) for c in wg1)
                w0 = {w.buffer.name for c in wg0 for w in infos_list[c].writes}
                r1 = {r.buffer.name for c in wg1 for r in infos_list[c].reads}
                xfer = len(w0 & r1) * 32 * XFER_PER_ELEM + 2 * XFER_BARRIER
                dual_t = max(t0, t1) + xfer
                if dual_t < best_dual:
                    best_dual = dual_t
            if best_dual < float('inf'):
                T_cons = best_dual - (overlap if cfg.async_wgmma else 0)

        # T_producer: early_bp gives producer a head start
        T_prod = T_prod_base
        if cfg.early_bp and n_wgmma >= 1:
            T_prod -= min(WGMMA_ACCUM_BASE, tma_lat_per_producer)

        T_iter = max(T_prod, T_cons)

        # Build annotations
        annos: Dict[str, str] = {}
        parts = []
        if cfg.async_wgmma:
            _set_ws_annotation(annos, "tl_pcws_async_pv", "1")
            parts.append(f"wait<1> overlap={overlap:.0f}")
        if cfg.dual_consumer:
            _set_ws_annotation(annos, "tl_pcws_dual_consumer", "1")
            parts.append("dual-WG")
        if cfg.three_role:
            _set_ws_annotation(annos, "tl_pcws_three_role", "1")
            parts.append("dQ-writer")
        desc = ", ".join(parts) if parts else "baseline"

        results.append(SchedulingResult(
            config=cfg, T_producer=T_prod, T_consumer=T_cons,
            T_iter=T_iter, annotations=annos, breakdown=desc,
        ))

    # ── Pick minimum T_iter ──
    best = min(results, key=lambda r: r.T_iter)

    if debug:
        print(f"[Heddle] Cost model: minimize max(T_prod, T_cons)",
              file=sys.stderr, flush=True)
        for r in results:
            m = " ◀" if r is best else ""
            print(f"  T_prod={r.T_producer:5.0f}  T_cons={r.T_consumer:5.0f}  "
                  f"→ T_iter={r.T_iter:5.0f}  ({r.breakdown}){m}",
                  file=sys.stderr, flush=True)

    return best


# ---------------------------------------------------------------------------
# Main pass logic
# ---------------------------------------------------------------------------

def _transform_pipeline_loop(
    func: tvm.tir.PrimFunc,
    *,
    use_precise_latency: bool = False,
    use_alap_priority: bool = True,
    buffer_span_aware: bool = True,
    relax_producer_boundary: bool = True,
    use_phase_b: bool = False,
    consumer_num_warps: int = 1,
    debug: bool = False,
) -> tvm.tir.PrimFunc:
    """Find pipeline loops and reorder consumer statements using Heddle."""

    func_alloc_vars = set(_collect_func_alloc_buffers(func).keys())
    buffer_var_map = _collect_func_alloc_buffers(func)
    # Extract the user-specified threadIdx.x extent. Dual-consumer WS is
    # only correct for threads=128 kernels (single consumer WG that is
    # safe to split by *adding* a second WG). For threads=256 the PV WGMMA
    # fragment covers only half the M-tile on a single WG, so routing PV
    # to WG1 alone produces an incomplete output. We gate the auto-trigger
    # accordingly.
    func_num_threads = _extract_num_threads(func)
    # Also look at the function attrs for the original kernel thread count
    # (set by T.Kernel(threads=...)). LowerOpaqueBlock and other early
    # passes may rewrite the threadIdx.x extent to the per-iteration
    # consumer count, so we fall back to the attr when present.
    attr_threads = 0
    try:
        if func.attrs:
            for key in ("threads", "tir.noalias", "tl_num_threads"):
                v = func.attrs.get(key)
                if v is not None and str(v).strip().isdigit():
                    attr_threads = int(v)
                    break
            if debug:
                print(f"[Heddle] func.attrs keys: {list(func.attrs.keys())}",
                      file=sys.stderr, flush=True)
    except Exception:
        attr_threads = 0
    if attr_threads > 0:
        func_num_threads = attr_threads
    # Fallback: scan kernel block/iter_vars for the threadIdx.x extent at
    # PrimFunc entry. Earlier passes (MultiVersionBuffer etc.) sometimes
    # present per-iter thread extent (e.g. 128) in AttrStmt even when the
    # user spec is 256; in that case we also look at the kernel-level
    # block's iter_vars and pick the largest observed thread extent.
    scan_max = [func_num_threads]
    def _scan(stmt):
        if isinstance(stmt, tvm.tir.For):
            tb = stmt.thread_binding
            if tb is not None:
                try:
                    tag = str(tb.thread_tag) if hasattr(tb, "thread_tag") else ""
                    if tag == "threadIdx.x":
                        v = int(stmt.extent)
                        if v > scan_max[0]:
                            scan_max[0] = v
                except Exception:
                    pass
        if isinstance(stmt, tvm.tir.AttrStmt):
            if str(stmt.attr_key) == "thread_extent":
                try:
                    if hasattr(stmt.node, "var") and stmt.node.var.name == "threadIdx.x":
                        v = int(stmt.value)
                        if v > scan_max[0]:
                            scan_max[0] = v
                except Exception:
                    pass
    tvm.tir.stmt_functor.post_order_visit(func.body, _scan)
    func_num_threads = scan_max[0]
    if debug:
        print(f"[Heddle] func_num_threads={func_num_threads} (attr={attr_threads})",
              file=sys.stderr, flush=True)
    changed = [False]

    def _visit_for(stmt):
        """Callback for ir_transform: process For loops with num_stages."""
        if not isinstance(stmt, tvm.tir.For):
            return None  # unchanged

        # Check for num_stages annotation
        ann = stmt.annotations
        if ann is None:
            return None
        num_stages = None
        for key in ann:
            if str(key) == "num_stages":
                try:
                    num_stages = int(ann[key])
                except (TypeError, ValueError):
                    pass
        if num_stages is None or num_stages <= 0:
            return None

        # Unwrap to SeqStmt
        # print('----- stmt :\n', stmt.script())

        seq, local_buf_map = _unwrap_to_seqstmt(stmt.body)
        if seq is None or len(seq.seq) < 2:
            return None

        # Merge buffer maps
        merged_buf_map = dict(buffer_var_map)
        merged_buf_map.update(local_buf_map)

        # Build statement infos
        infos_list, barrier_infos = _build_stmt_infos(seq, merged_buf_map, func_alloc_vars)

        # Check we have both producers and consumers
        has_producer = any(info.is_producer for info in infos_list)
        has_consumer = any(not info.is_producer for info in infos_list)
        if not has_producer or not has_consumer:
            if debug:
                print(f"[Heddle] Skip: has_producer={has_producer}, has_consumer={has_consumer}", file=sys.stderr, flush=True)
            return None

        consumer_count = sum(1 for info in infos_list if not info.is_producer)
        if debug:
            print(f"[Heddle] Found {len(infos_list)} stmts, {consumer_count} consumers, {len(infos_list) - consumer_count} producers", file=sys.stderr, flush=True)
            for info in infos_list:
                latency, rty = _detect_op_latency_and_resource(info.stmt)
                role = "PRODUCER" if info.is_producer else "consumer"
                flags = []
                if info.is_wgmma: flags.append("wgmma")
                if info.is_sync_top: flags.append("sync_top")
                if info.is_sync_nested: flags.append("sync_nest")
                if info.touches_external_local: flags.append("ext_local")
                print(f"  s{info.idx}: {role} lat={latency} rty={rty} flags={flags} "
                      f"reads=[{','.join(r.buffer.name for r in info.reads)}] "
                      f"writes=[{','.join(w.buffer.name for w in info.writes)}]",
                      file=sys.stderr, flush=True)
        if consumer_count <= 1:
            return None  # nothing to reorder

        # ── Auto-detect dual-consumer with cost model + split search ──
        # 1. Detect pattern: ≥2 WGMMAs with non-TC gap
        # 2. Enumerate possible split points between WGMMA groups
        # 3. Estimate cost of each split (workload balance + transfer overhead)
        # 4. Only enable if predicted speedup > 1.0

        consumer_wgmma_indices = [
            info.idx for info in infos_list
            if not info.is_producer and info.is_wgmma
        ]
        has_non_tc_gap = False
        dual_consumer_split_idx = -1  # will be set if beneficial

        if len(consumer_wgmma_indices) >= 2:
            first_wgmma = consumer_wgmma_indices[0]
            last_wgmma = consumer_wgmma_indices[-1]
            has_non_tc_gap = any(
                not info.is_producer and not info.is_wgmma
                and first_wgmma < info.idx < last_wgmma
                for info in infos_list
            )

        if has_non_tc_gap and len(consumer_wgmma_indices) >= 2:
            # ── Cost model for dual-consumer benefit estimation ──
            # Latency constants (Hopper cycles). These were previously too
            # small — a single fp16 m64n128k16 WGMMA issues 8 inner iters of
            # ~27 cycles each (~216 cycles), not the 27 single-op latency
            # returned by _detect_op_latency_and_resource. We scale the raw
            # latency of WGMMA ops to reflect realistic pipelined issue time
            # inside the dual-WG balance estimate; otherwise the solver sees
            # a tiny single-WG time and always rejects the split.
            # Dual-WG cross-transfer cost on Hopper. Two named barriers (arrive
            # on WG0, wait on WG1) ~25 cycles each under contention; the flat
            # att/m/l store is vectorized via smem. Previous setting charged
            # 4 barriers at 50 cycles plus a heavy per-elem cost and vetoed
            # every realistic split — FA FWD's QK/PV overlap was never tried.
            BARRIER_OVERHEAD = 25      # one named barrier rendezvous (~25 cyc)
            FLAT_COPY_PER_ELEM = 0.25  # vectorized smem copy per half elem
            WGMMA_ISSUE_MULT = 8       # WGMMA serial inner iterations

            stmt_latencies = {}
            for info in infos_list:
                if info.is_producer:
                    continue
                lat, _ = _detect_op_latency_and_resource(info.stmt)
                if info.is_wgmma:
                    lat *= WGMMA_ISSUE_MULT
                stmt_latencies[info.idx] = lat

            consumer_indices_set = {
                info.idx for info in infos_list if not info.is_producer
            }

            # Single-WG time = strict serial sum (one resource, no overlap).
            single_wg_time = sum(stmt_latencies.get(ci, 0) for ci in consumer_indices_set)

            best_score = single_wg_time
            best_split = -1

            sorted_consumers = sorted(consumer_indices_set)
            # Restrict candidate splits to the statement immediately after
            # each WGMMA boundary. Splitting mid-softmax produces dangling
            # register references that the dual-consumer codegen in PCWS
            # cannot resolve and triggers launch-time faults.
            wgmma_set = set(consumer_wgmma_indices)
            candidate_splits = []
            for ci in sorted_consumers:
                if ci in wgmma_set:
                    # Next consumer index after this WGMMA is a valid split
                    idx_after = next(
                        (c for c in sorted_consumers if c > ci), None
                    )
                    if idx_after is not None:
                        candidate_splits.append(idx_after)
            for split_idx in candidate_splits:
                wg0_stmts = [ci for ci in sorted_consumers if ci < split_idx]
                wg1_stmts = [ci for ci in sorted_consumers if ci >= split_idx]

                wg0_has_wgmma = any(ci in consumer_wgmma_indices for ci in wg0_stmts)
                wg1_has_wgmma = any(ci in consumer_wgmma_indices for ci in wg1_stmts)
                if not (wg0_has_wgmma and wg1_has_wgmma):
                    continue

                wg0_time = sum(stmt_latencies.get(ci, 0) for ci in wg0_stmts)
                wg1_time = sum(stmt_latencies.get(ci, 0) for ci in wg1_stmts)

                wg0_writes = set()
                wg1_reads = set()
                for ci in wg0_stmts:
                    for w in infos_list[ci].writes:
                        wg0_writes.add(w.buffer.name)
                for ci in wg1_stmts:
                    for r in infos_list[ci].reads:
                        wg1_reads.add(r.buffer.name)
                xfer_bufs = wg0_writes & wg1_reads
                xfer_elems = len(xfer_bufs) * 32
                # Two barriers (arrive on WG0, wait on WG1). The flat att/m/l
                # store overlaps with WG1's rescale, so per-iter only the
                # barrier rendezvous adds to the makespan in steady state.
                xfer_cost = xfer_elems * FLAT_COPY_PER_ELEM + 2 * BARRIER_OVERHEAD

                # Steady-state dual-WG cost per iteration. Two WGs run
                # concurrently; the slower half dominates, plus the
                # single-barrier rendezvous that cannot overlap anything.
                # The previous formula ``max(wg0, wg1 + xfer)`` was algebraically
                # wrong: it modeled parallelism between wg0 and (wg1+xfer)
                # and undercharged when wg0 dominated.
                dual_time = max(wg0_time, wg1_time) + xfer_cost

                if dual_time < best_score:
                    best_score = dual_time
                    best_split = split_idx

            # Decision: enable dual-consumer if beneficial
            if best_split >= 0:
                speedup = single_wg_time / best_score if best_score > 0 else 0
                dual_consumer_split_idx = best_split
                if debug:
                    print(f"[Heddle] Dual-consumer cost model: "
                          f"single={single_wg_time} dual={best_score:.0f} "
                          f"speedup={speedup:.2f}x split_at={best_split} "
                          f"({len(consumer_wgmma_indices)} WGMMAs)",
                          file=sys.stderr, flush=True)
            else:
                has_non_tc_gap = False  # disable dual-consumer
                if debug:
                    print(f"[Heddle] Dual-consumer cost model: "
                          f"no beneficial split found (single={single_wg_time})",
                          file=sys.stderr, flush=True)

        # Build consumer dependency graph (Opt 3: relaxed producer boundaries)
        consumer_indices, deps, all_indices ,deps_all = _build_consumer_dep_graph(
            infos_list, barrier_infos,
            relax_producer_boundary=relax_producer_boundary,
        )
        print(f'----{consumer_indices=} ')
        print(f'----{deps=} ')
        print(f'----{deps_all=} ')
        print(f'----{barrier_infos=} ')
        print('---- infos_list : ')
        for info in infos_list :
            msg = f"[{info.idx}] wr:"
            for wr in info.writes :
                msg += f"{wr.buffer.name}, "
            msg += " / rd:"
            for rd in info.reads :
                msg += f"{rd.buffer.name}, "
            print(msg)

        # ── Skip reordering for simple kernels ──
        # When there is at most 1 WGMMA consumer (e.g. GEMM, Dequant GEMM),
        # consumer reordering has no benefit — the single compute op has no
        # parallel siblings to reorder with. Applying the scheduling passes
        # For single-WGMMA kernels (GEMM), skip consumer reorder (only 1
        # WGMMA → nothing to reorder) but still run the cost model to
        # inject async WGMMA / early bp annotations.
        if len(consumer_wgmma_indices) <= 1:
            if debug:
                print(f"[Heddle] Single WGMMA — skip reorder, run cost model only",
                      file=sys.stderr, flush=True)
            max_wgmma_out_elems = 0
            for ci in consumer_wgmma_indices:
                for wr in infos_list[ci].writes:
                    fp = _estimate_buffer_footprint_bytes(wr.buffer)
                    elems = fp // 4
                    if elems > max_wgmma_out_elems:
                        max_wgmma_out_elems = elems
            strategy = _evaluate_scheduling_cost_model(
                infos_list, consumer_indices, consumer_wgmma_indices,
                has_non_tc_gap, max_wgmma_out_elems,
                has_tma_reduce_add=_detect_tma_reduce_add(seq),
                debug=debug,
            )
            if strategy.annotations:
                if debug:
                    print(f"[Heddle] Selected: {strategy.breakdown} ({strategy.T_iter:.0f} cyc)",
                          file=sys.stderr, flush=True)
                phase_a_annotations = dict(stmt.annotations) if stmt.annotations else {}
                for k, v in strategy.annotations.items():
                    _ensure_ws_annotation(phase_a_annotations, k, v)
                changed[0] = True
                return tvm.tir.For(
                    stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
                    stmt.body, stmt.thread_binding, phase_a_annotations,
                )
            return None

        # ---- TWill : step 1 求解基础模调度M
        mod_sched_plans = _solve_naive_modulo_sched(deps_all, infos_list, all_indices)
        # ---- TWill : step 2 求解联合优化问题： 基础模调度M + warp_spec
        for plan in mod_sched_plans :
            print('----start  _solve_smt_joint_optimize', flush=True)
            optimized =  _solve_smt_joint_optimize(deps_all, infos_list, all_indices, plan)
            if optimized and optimized['modular_rrt'] is not None :
                for row in optimized['modular_rrt'] :
                    print(row)

        # ── Phase B: SMT-based joint ordering ──
        # Policy: run Phase B if (a) explicitly enabled, or (b) ≥3 WGMMA
        # consumer ops detected (complex dependency → heuristic may produce
        # FU conflicts that only joint solve can resolve).
        print("---- [原始路径] start smt solving -----")
        n_tc_ops = len(consumer_wgmma_indices)
        auto_phase_b = (n_tc_ops >= 3) and (len(consumer_indices) >= 6)
        run_phase_b = use_phase_b or auto_phase_b or consumer_num_warps > 1

        if run_phase_b:
            # Adaptive timeout: 500ms base + 100ms per consumer node
            adaptive_timeout = min(500 + len(consumer_indices) * 100, 15000)
            if debug:
                trigger = "explicit" if use_phase_b else f"auto (TC={n_tc_ops}, consumers={len(consumer_indices)})"
                print(f"[Heddle] Phase B triggered ({trigger}), timeout={adaptive_timeout}ms",
                      file=sys.stderr, flush=True)

            import time as _time
            _pb_t0 = _time.perf_counter()
            phase_b_result = _phase_b_consumer_ordering(
                infos_list, consumer_indices, deps,
                use_precise_latency=use_precise_latency,
                num_warps=consumer_num_warps,
                timeout_ms=adaptive_timeout,
                debug=debug,
            )
            _pb_elapsed = (_time.perf_counter() - _pb_t0) * 1000
            if phase_b_result is not None:
                phase_b_order, phase_b_times, phase_b_warps = phase_b_result
            else:
                phase_b_order, phase_b_times, phase_b_warps = None, {}, {}

            if consumer_num_warps > 1 and phase_b_order is None:
                phase_b_order = list(consumer_indices)
                phase_b_times = {}
                phase_b_warps = {}
                if debug:
                    print(f"[Heddle] Phase B failed but multi-consumer warp is "
                          f"requested; using original order and letting PCWS "
                          f"choose a structured consumer split",
                          file=sys.stderr, flush=True)

            if consumer_num_warps > 1 and phase_b_order is not None:
                warp_keys = {int(name[1:]) for name in phase_b_warps}
                if phase_b_warps and warp_keys != set(consumer_indices):
                    phase_b_warps = {}
                    if debug:
                        print(f"[Heddle] Phase B returned partial warp assigns; "
                              f"skipping per-op dispatch and letting PCWS choose "
                              f"a structured consumer split",
                              file=sys.stderr, flush=True)

            if debug:
                status = "SAT" if phase_b_order is not None else "UNSAT/timeout"
                print(f"[Heddle] Phase B completed in {_pb_elapsed:.1f}ms ({status})",
                      file=sys.stderr, flush=True)
            if phase_b_order is not None:
                order_changed = (phase_b_order != consumer_indices)
                if debug:
                    if order_changed:
                        print(f"[Heddle] Using Phase B ordering: {consumer_indices} -> {phase_b_order}",
                              file=sys.stderr, flush=True)
                    else:
                        print(f"[Heddle] Phase B ordering unchanged, but extracting barrier hints",
                              file=sys.stderr, flush=True)

                # Extract barrier hints from Phase B schedule times.
                # Hints are valuable even when the ordering is unchanged,
                # because they carry delay information from the schedule.
                pcws_hint = None
                try:
                    pcws_hint = _extract_barrier_hints(
                        infos_list, consumer_indices, phase_b_order,
                        phase_b_times,
                        use_precise_latency=use_precise_latency,
                        debug=debug,
                    )
                except Exception as e:
                    if debug:
                        print(f"[Heddle] Barrier hint extraction failed: {e}",
                              file=sys.stderr, flush=True)

                # Build new loop body (reordered or original)
                if order_changed:
                    new_seq = _reorder_loop_body(seq, infos_list, phase_b_order)
                    new_body = _rewrap_body(stmt.body, seq, new_seq)
                    changed[0] = True
                else:
                    new_body = stmt.body

                # Attach barrier hints as loop annotations for PCWS.
                # Always attach when Phase B produced hints, even if
                # the ordering is unchanged.
                new_annotations = dict(stmt.annotations) if stmt.annotations else {}
                annotations_changed = False

                # Auto-inject dual-consumer annotation when pattern detected
                if not phase_b_warps and len(consumer_wgmma_indices) >= 2 and has_non_tc_gap:
                    annotations_changed = (
                        _ensure_ws_annotation(new_annotations, "tl_pcws_dual_consumer", "1")
                        or annotations_changed
                    )
                    if dual_consumer_split_idx >= 0:
                        annotations_changed = (
                            _ensure_ws_annotation(
                                new_annotations,
                                "tl_pcws_dual_consumer_split",
                                str(dual_consumer_split_idx),
                            )
                            or annotations_changed
                        )

                # Auto-inject three-role annotation when TMA reduce-add
                # pattern is detected (T.atomic_add with use_tma=True).
                # This tells PCWS to extract dQ writer warps into a
                # dedicated third role in the producer warp group.
                if _detect_tma_reduce_add(seq):
                    three_role_added = _ensure_ws_annotation(
                        new_annotations, "tl_pcws_three_role", "1"
                    )
                    annotations_changed = three_role_added or annotations_changed
                    if three_role_added and debug:
                        print("[Heddle] Auto-detected TMA reduce-add pattern, "
                              "enabling three-role WS",
                              file=sys.stderr, flush=True)

                if pcws_hint is not None:
                    hints_str = pcws_hint.to_barrier_hints_config()
                    offsets_str = pcws_hint.to_stage_offsets_config()
                    if hints_str:
                        _set_ws_annotation(new_annotations, "tl_pcws_barrier_hints", hints_str)
                        annotations_changed = True
                    if offsets_str:
                        _set_ws_annotation(new_annotations, "tl_pcws_stage_offsets", offsets_str)
                        annotations_changed = True

                if phase_b_warps:
                    pcws_order = phase_b_order if phase_b_order is not None else consumer_indices
                    pcws_warp_keys = {
                        f"s{ci}": f"s{pos}"
                        for pos, ci in enumerate(pcws_order)
                    }
                    warp_str = ",".join(
                        f"{pcws_warp_keys.get(k, k)}:{v}"
                        for k, v in sorted(phase_b_warps.items())
                    )
                    _set_ws_annotation(new_annotations, "tl_pcws_warp_assigns", warp_str)
                    annotations_changed = True
                    if debug:
                        print(f"[Heddle] Injected per-op warp assigns: {warp_str}",
                              file=sys.stderr, flush=True)

                if order_changed or annotations_changed:
                    changed[0] = True
                    return tvm.tir.For(
                        stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
                        new_body, stmt.thread_binding, new_annotations,
                    )

            if debug:
                print(f"[Heddle] Phase B returned None, falling back to Phase A",
                      file=sys.stderr, flush=True)

        # Phase A: ASAP-based scheduling (fallback or primary)
        import time as _time
        _pa_t0 = _time.perf_counter()

        # Compute ASAP times for consumer statements
        asap_times = _compute_asap_times(
            infos_list, consumer_indices, deps,
            use_precise_latency=use_precise_latency,
            debug=debug,
        )

        if asap_times is None:
            if debug:
                print(f"[Heddle] Phase A returned None for {len(consumer_indices)} consumers, keeping original order", file=sys.stderr, flush=True)
                print(f"[Heddle] Consumer deps: {deps}", file=sys.stderr, flush=True)
            return None

        # Compute scheduling priorities
        priorities = dict(asap_times)

        # Opt 1: ALAP + slack-based scheduling
        if use_alap_priority:
            alap_times = _compute_alap_times(
                infos_list, consumer_indices, deps, asap_times,
                use_precise_latency=use_precise_latency,
                debug=debug,
            )
            # Blended priority: critical-path nodes (slack=0) use ASAP,
            # non-critical nodes (slack>0) use ALAP (delayed scheduling).
            # Exception: sync statements always keep ASAP priority — they
            # coordinate producer-consumer timing and should not be delayed.
            for ci in consumer_indices:
                info = infos_list[ci]
                if info.is_sync_top or info.is_sync_nested:
                    continue  # keep ASAP priority for syncs
                slack = alap_times[ci] - asap_times[ci]
                if slack > 0:
                    priorities[ci] = alap_times[ci]

        # Opt 2: Buffer-span-aware priority adjustment
        if buffer_span_aware:
            priorities = _compute_buffer_span_priorities(
                infos_list, consumer_indices, priorities,
                use_precise_latency=use_precise_latency,
                debug=debug,
            )

        # Compute new consumer ordering via topo sort with priorities
        # Pass infos + reg_limit for resource-aware tie-breaking
        new_order = _topo_sort_with_priority(
            consumer_indices, deps, priorities,
            infos=infos_list, reg_limit=960,
        )

        _pa_elapsed = (_time.perf_counter() - _pa_t0) * 1000
        if debug:
            print(f"[Heddle] Phase A completed in {_pa_elapsed:.1f}ms "
                  f"({len(consumer_indices)} consumers)",
                  file=sys.stderr, flush=True)

        # When dual-consumer WS is about to be enabled, skip consumer
        # reordering. PCWS's dual-consumer codegen detects the split point
        # by scanning stmt order for the V-wait position and for specific
        # smem/fragment patterns (att store, sscl rescale). Reordering
        # consumers breaks that pattern match and produces an invalid
        # kernel at launch time. Keep the original order; the structural
        # parallelism from dual-WG already dominates any micro-reorder.
        dual_consumer_will_fire = (
            func_num_threads == 128
            and len(consumer_wgmma_indices) >= 2 and has_non_tc_gap
            and not _has_ws_annotation(stmt.annotations or {}, "tl_pcws_dual_consumer")
        )
        if dual_consumer_will_fire:
            if debug:
                print("[Heddle] Skipping consumer reorder (dual-consumer mode)",
                      file=sys.stderr, flush=True)
            new_order = list(consumer_indices)

        # Check if order actually changed
        if new_order == consumer_indices and not dual_consumer_will_fire:
            if debug:
                print("[Heddle] Consumer order unchanged after scheduling", file=sys.stderr, flush=True)
            return None

        if debug and new_order != consumer_indices:
            print(f"[Heddle] Reordering consumers: {consumer_indices} -> {new_order}", file=sys.stderr, flush=True)

        # Reorder the SeqStmt (no-op when new_order matches original)
        if new_order != list(consumer_indices):
            new_seq = _reorder_loop_body(seq, infos_list, new_order)
            new_body = _rewrap_body(stmt.body, seq, new_seq)
        else:
            new_body = stmt.body

        changed[0] = True

        # ── Phase A: extract barrier hints from ASAP schedule ──
        # Build fake schedule times from ASAP (Phase B uses real schedule_times).
        # _extract_barrier_hints expects keys of the form f"s{ci}" (string).
        phase_a_times = {f"s{ci}": int(asap_times[ci]) for ci in consumer_indices if ci in asap_times}
        pcws_hint_a = None
        try:
            pcws_hint_a = _extract_barrier_hints(
                infos_list, consumer_indices, new_order,
                phase_a_times,
                use_precise_latency=use_precise_latency,
                debug=debug,
            )
            if debug:
                n_hints = len(pcws_hint_a.barrier_hints) if pcws_hint_a else 0
                print(f"[Heddle] Phase A extracted {n_hints} barrier hints "
                      f"from {len(phase_a_times)} ASAP times",
                      file=sys.stderr, flush=True)
        except Exception as e:
            if debug:
                print(f"[Heddle] Phase A barrier hint extraction failed: {e}",
                      file=sys.stderr, flush=True)

        # ── Unified cost model: evaluate all scheduling strategies ──
        phase_a_annotations = dict(stmt.annotations) if stmt.annotations else {}

        max_wgmma_out_elems = 0
        for ci in consumer_wgmma_indices:
            for wr in infos_list[ci].writes:
                fp = _estimate_buffer_footprint_bytes(wr.buffer)
                elems = fp // 4
                if elems > max_wgmma_out_elems:
                    max_wgmma_out_elems = elems

        strategy = _evaluate_scheduling_cost_model(
            infos_list, consumer_indices, consumer_wgmma_indices,
            has_non_tc_gap, max_wgmma_out_elems,
            has_tma_reduce_add=_detect_tma_reduce_add(seq),
            debug=debug,
        )
        if debug:
            print(f"[Heddle] Selected strategy: {strategy.breakdown} "
                  f"({strategy.T_iter:.0f} cyc)",
                  file=sys.stderr, flush=True)

        for key, val in strategy.annotations.items():
            _ensure_ws_annotation(phase_a_annotations, key, val)
        # Inject barrier_hints annotation so PCWS uses Heddle's wait/release positions
        if pcws_hint_a is not None:
            hints_str = pcws_hint_a.to_barrier_hints_config()
            if hints_str:
                _set_ws_annotation(phase_a_annotations, "tl_pcws_barrier_hints", hints_str)
                if debug:
                    print(f"[Heddle] Phase A injected barrier hints: {hints_str}",
                          file=sys.stderr, flush=True)
        return tvm.tir.For(
            stmt.loop_var,
            stmt.min,
            stmt.extent,
            stmt.kind,
            new_body,
            stmt.thread_binding,
            phase_a_annotations,
        )

    print("[d] ---- func before SMT -----", func.script())
    new_body = tvm.tir.stmt_functor.ir_transform(
        func.body, None, _visit_for, ["tir.For"]
    )

    if changed[0]:
        return func.with_body(new_body)
    return func


def _rewrap_body(
    original_body: tvm.tir.Stmt,
    old_seq: tvm.tir.SeqStmt,
    new_seq: tvm.tir.SeqStmt,
) -> tvm.tir.Stmt:
    """Re-wrap the new SeqStmt with the original Block/BlockRealize/Let/Attr layers."""
    # Walk the original body to find the old SeqStmt, then substitute
    def _substitute(node):
        if node is old_seq:
            return new_seq
        if isinstance(node, tvm.tir.BlockRealize):
            blk = node.block
            new_blk_body = _substitute(blk.body)
            if new_blk_body is blk.body:
                return node
            new_blk = tvm.tir.Block(
                blk.iter_vars, blk.reads, blk.writes, blk.name_hint,
                new_blk_body, blk.init, blk.alloc_buffers, blk.match_buffers,
                blk.annotations,
            )
            return tvm.tir.BlockRealize(node.iter_values, node.predicate, new_blk)
        if isinstance(node, tvm.tir.Block):
            new_body = _substitute(node.body)
            if new_body is node.body:
                return node
            return tvm.tir.Block(
                node.iter_vars, node.reads, node.writes, node.name_hint,
                new_body, node.init, node.alloc_buffers, node.match_buffers,
                node.annotations,
            )
        if isinstance(node, tvm.tir.LetStmt):
            new_body = _substitute(node.body)
            if new_body is node.body:
                return node
            return tvm.tir.LetStmt(node.var, node.value, new_body)
        if isinstance(node, tvm.tir.AttrStmt):
            new_body = _substitute(node.body)
            if new_body is node.body:
                return node
            return tvm.tir.AttrStmt(node.node, node.attr_key, node.value, new_body)
        if isinstance(node, tvm.tir.IfThenElse) and node.else_case is None:
            new_then = _substitute(node.then_case)
            if new_then is node.then_case:
                return node
            return tvm.tir.IfThenElse(node.condition, new_then, None)
        return node

    return _substitute(original_body)


# ---------------------------------------------------------------------------
# TVM pass registration
# ---------------------------------------------------------------------------

def HeddleConsumerSchedule():
    """Create a TVM pass that reorders consumer statements using Heddle SMT.

    This pass should be applied BEFORE ProducerConsumerWarpSpecialized.
    It uses the Heddle scheduler to find an optimal consumer ordering
    and reorders the IR statements accordingly. PCWS then performs
    barrier placement on the optimally-ordered IR.
    """

    @tvm.tir.transform.prim_func_pass(opt_level=0, name="tl.HeddleConsumerSchedule")
    def _pass_func(func: tvm.tir.PrimFunc, mod, ctx: tvm.ir.transform.PassContext):
        debug = os.environ.get("TL_HEDDLE_DEBUG", "0") == "1"
        enable = _get_bool_config(ctx, "tl.enable_heddle_consumer_schedule", False)
        if debug:
            print(f"[Heddle] Pass entry: enable={enable}", file=sys.stderr, flush=True)
        if not enable:
            return func

        use_precise_latency = _get_bool_config(ctx, "tl.heddle_use_precise_latency", True)
        use_alap_priority = _get_bool_config(ctx, "tl.heddle_use_alap_priority", True)
        buffer_span_aware = _get_bool_config(ctx, "tl.heddle_buffer_span_aware", True)
        relax_producer_boundary = _get_bool_config(ctx, "tl.heddle_relax_producer_boundary", True)
        use_phase_b = _get_bool_config(ctx, "tl.heddle_use_phase_b", False)
        consumer_num_warps = max(1, _get_int_config(ctx, "tl.heddle_consumer_num_warps", 1))

        if debug:
            print(f"[Heddle] HeddleConsumerSchedule pass running "
                  f"(alap={use_alap_priority}, buf_span={buffer_span_aware}, "
                  f"relax_pb={relax_producer_boundary}, phase_b={use_phase_b}, "
                  f"consumer_warps={consumer_num_warps})",
                  file=sys.stderr, flush=True)

        try:
            result = _transform_pipeline_loop(
                func,
                use_precise_latency=use_precise_latency,
                use_alap_priority=use_alap_priority,
                buffer_span_aware=buffer_span_aware,
                relax_producer_boundary=relax_producer_boundary,
                use_phase_b=use_phase_b,
                consumer_num_warps=consumer_num_warps,
                debug=debug,
            )
            print(" --after smt ----\n " ,result.script())
            
            return result
        except Exception as e:
            if debug:
                print(f"[Heddle] Pass failed with error: {e}", file=sys.stderr, flush=True)
            return func  # graceful fallback: return unchanged

    return _pass_func
