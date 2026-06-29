from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from tvm import tir

import tilelang
from tilelang import tvm as tvm

if TYPE_CHECKING:
    try:
        from heddle.scheduler.smt import OpNode, ResourceType  # type: ignore
    except Exception:  # pragma: no cover
        from heddle.scheduler.smt import OpNode, ResourceType  # type: ignore


def _get_bool_config(ctx: tvm.ir.transform.PassContext, key: str, default: bool) -> bool:
    try:
        from heddle._monkey_patch import get_heddle_pass_config
        # PassContext.config is a mapping (python-side); values are python bool/int/str.
        return bool(ctx.config.get(key, get_heddle_pass_config(key, default)))
    except Exception:
        return default


def _get_int_config(ctx: tvm.ir.transform.PassContext, key: str, default: int) -> int:
    try:
        from heddle._monkey_patch import get_heddle_pass_config
        v = ctx.config.get(key, get_heddle_pass_config(key, default))
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _get_str_config(ctx: tvm.ir.transform.PassContext, key: str, default: str) -> str:
    try:
        from heddle._monkey_patch import get_heddle_pass_config
        v = ctx.config.get(key, get_heddle_pass_config(key, default))
        if v is None:
            return default
        return str(v)
    except Exception:
        return default


def _parse_group_override(raw: str) -> Optional[List[List[int]]]:
    """Parse a group override string into a list of statement index groups.

    Supported formats:
      - "0;1,2;3-11;12;13;14"
      - "[[0],[1,2],[3..11],[12],[13],[14]]"
    """
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    # Normalize separators and ranges
    text = text.replace("[", "").replace("]", "")
    text = text.replace("..", "-")
    parts = [p.strip() for p in re.split(r"[;|]+", text) if p.strip()]
    groups: List[List[int]] = []
    for part in parts:
        items = [x for x in re.split(r"[\s,]+", part) if x]
        group: List[int] = []
        for item in items:
            if "-" in item:
                a, b = item.split("-", 1)
                try:
                    start = int(a.strip())
                    end = int(b.strip())
                except ValueError:
                    continue
                if end < start:
                    start, end = end, start
                group.extend(list(range(start, end + 1)))
            else:
                try:
                    group.append(int(item))
                except ValueError:
                    continue
        if group:
            groups.append(group)
    return groups if groups else None


def _parse_order_override(raw: str) -> Optional[List[int]]:
    """Parse an order override string into a list of ints."""
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    text = text.replace("[", "").replace("]", "")
    items = [x.strip() for x in re.split(r"[,\s]+", text) if x.strip()]
    out: List[int] = []
    for item in items:
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out if out else None


def _parse_stage_override(raw: str) -> Optional[List[int]]:
    """Parse a stage override string into a list of ints."""
    return _parse_order_override(raw)


def _compute_semantic_stmt_stage(infos: List[_StmtInfo]) -> List[int]:
    """Compute per-stmt stage based on shared-buffer producer stages."""
    buffer_stage: Dict[tvm.tir.Buffer, int] = {}
    producer_stage = 0
    last_consumer_stage = 0
    last_producer_stage = -1
    stmt_stage: List[int] = []
    for info in infos:
        if info.is_producer:
            # Assign stage to buffers written by this producer
            for wr in info.writes:
                if _is_shared(wr.buffer):
                    buffer_stage[wr.buffer] = producer_stage
            stmt_stage.append(-1)
            last_producer_stage = producer_stage
            producer_stage += 1
            continue
        # Consumer: stage based on max producer stage of shared reads
        stages = []
        for rd in info.reads:
            if _is_shared(rd.buffer) and rd.buffer in buffer_stage:
                stages.append(buffer_stage[rd.buffer])
        if stages:
            st = max(stages)
        else:
            # If no shared reads, inherit the most recent producer stage when available.
            st = last_producer_stage if last_producer_stage >= 0 else last_consumer_stage
        stmt_stage.append(int(st))
        last_consumer_stage = int(st)
    return stmt_stage


def _compute_group_stage_from_stmt_stage(
    groups: List[List[int]],
    stmt_stage: List[int],
    *,
    num_stages: int,
    stage_offset: int = 0,
) -> List[int]:
    """Compute group stage from per-stmt stage with optional offset."""
    smax = max(1, num_stages) - 1
    out: List[int] = []
    for grp in groups:
        vals = [stmt_stage[i] for i in grp if i < len(stmt_stage) and stmt_stage[i] >= 0]
        if not vals:
            out.append(-1)
            continue
        st = max(vals) + max(0, stage_offset)
        if st > smax:
            st = smax
        out.append(int(st))
    return out


def _unwrap_to_seqstmt(body: tvm.tir.Stmt) -> Tuple[Optional[tvm.tir.SeqStmt], Dict[tvm.tir.Var, tvm.tir.Buffer]]:
    """Try to unwrap a loop body to a top-level SeqStmt, collecting alloc_buffers into buffer_var_map."""
    buffer_var_map: Dict[tvm.tir.Var, tvm.tir.Buffer] = {}

    cur = body
    while True:
        if isinstance(cur, tvm.tir.SeqStmt):
            return cur, buffer_var_map
        if isinstance(cur, tvm.tir.BlockRealize):
            blk = cur.block
            for buf in blk.alloc_buffers:
                buffer_var_map[buf.data] = buf
            cur = blk.body
            continue
        if isinstance(cur, tvm.tir.Block):
            for buf in cur.alloc_buffers:
                buffer_var_map[buf.data] = buf
            cur = cur.body
            continue
        if isinstance(cur, tvm.tir.LetStmt):
            cur = cur.body
            continue
        if isinstance(cur, tvm.tir.AttrStmt):
            cur = cur.body
            continue
        if isinstance(cur, tvm.tir.IfThenElse):
            # Conservative: only unwrap single-branch If.
            if cur.else_case is not None:
                return None, {}
            cur = cur.then_case
            continue
        return None, {}


def _collect_func_alloc_buffers(func: tvm.tir.PrimFunc) -> Dict[tvm.tir.Var, tvm.tir.Buffer]:
    """Collect all alloc_buffers in the PrimFunc into a buffer var map.

    TVM region analysis APIs often require a complete mapping from `Buffer.data` vars to `Buffer`
    objects. Collecting alloc_buffers only along one unwrap path is fragile after passes like
    MultiVersionBuffer / WarpSpecialized that restructure blocks.
    """
    out: Dict[tvm.tir.Var, tvm.tir.Buffer] = {}

    def visit(stmt):
        if isinstance(stmt, tvm.tir.Block):
            for b in stmt.alloc_buffers:
                out[b.data] = b

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return out


def _call_op_names(stmt: tvm.tir.Stmt) -> Set[str]:
    names: Set[str] = set()

    def _v(node):
        if isinstance(node, tvm.tir.Call):
            try:
                if isinstance(node.op, tvm.ir.Op):
                    names.add(node.op.name)
                # also detect call_extern target name (string at args[0])
                if isinstance(node.op, tvm.ir.Op) and node.op.name == "tir.call_extern":
                    if node.args and isinstance(node.args[0], tvm.tir.StringImm):
                        names.add(f"extern:{node.args[0].value}")
            except Exception:
                pass

    tvm.tir.stmt_functor.post_order_visit(stmt, _v)
    return names


def _static_positive_int(expr: tvm.tir.PrimExpr) -> Optional[int]:
    if isinstance(expr, tvm.tir.IntImm):
        v = int(expr.value)
        return v if v > 0 else None
    return None


def _is_wgmma_call(call: tvm.tir.Call) -> bool:
    if not isinstance(call.op, tvm.ir.Op):
        return False
    op_name = call.op.name
    if op_name in {"tl.tl_gemm", "tl.tl_gemm_sp", "tl.ptx_wgmma_ss", "tl.ptx_wgmma_rs"}:
        return True
    if op_name == "tir.call_extern" and call.args and isinstance(call.args[0], tvm.tir.StringImm):
        f = call.args[0].value
        return (
            f.startswith("tl::tcgen5mma_gemm_")
            or f.startswith("tl::wgmma")
            or f.startswith("tl::tcgen05")
        )
    return False


def _parse_wgmma_mnk_from_string(s: str) -> Optional[Tuple[int, int, int]]:
    m = re.search(r"\bm(\d+)n(\d+)k(\d+)\b", s)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _parse_wgmma_kind_from_string(s: str) -> Optional[str]:
    if re.search(r"(?:^|[_:])wgmma_ss(?:[<_]|$)", s):
        return "ss"
    if re.search(r"(?:^|[_:])wgmma_rs(?:[<_]|$)", s):
        return "rs"
    return None


def _wgmma_call_kind(call: tvm.tir.Call) -> Optional[str]:
    if not isinstance(call.op, tvm.ir.Op):
        return None

    op_name = call.op.name
    if op_name == "tl.ptx_wgmma_ss":
        return "ss"
    if op_name == "tl.ptx_wgmma_rs":
        return "rs"
    if op_name == "tir.call_extern" and call.args and isinstance(call.args[0], tvm.tir.StringImm):
        return _parse_wgmma_kind_from_string(call.args[0].value)
    return None


def _wgmma_call_mnk(call: tvm.tir.Call) -> Optional[Tuple[int, int, int]]:
    if not isinstance(call.op, tvm.ir.Op):
        return None

    op_name = call.op.name
    if op_name in {"tl.ptx_wgmma_ss", "tl.ptx_wgmma_rs"}:
        if call.args and isinstance(call.args[0], tvm.tir.StringImm):
            return _parse_wgmma_mnk_from_string(call.args[0].value)
        return None

    if op_name == "tir.call_extern" and call.args and isinstance(call.args[0], tvm.tir.StringImm):
        return _parse_wgmma_mnk_from_string(call.args[0].value)

    return None


def _wgmma_call_desc(call: tvm.tir.Call) -> Optional[Tuple[str, Optional[Tuple[int, int, int]]]]:
    if not _is_wgmma_call(call):
        return None
    return _wgmma_call_kind(call) or "unknown", _wgmma_call_mnk(call)


def _is_tma_call(call: tvm.tir.Call) -> bool:
    if not isinstance(call.op, tvm.ir.Op):
        return False
    return call.op.name in {"tl.tma_load", "tl.tma_load_im2col", "tir.tma_load"}


def _count_calls_with_static_loop_multiplier(stmt: tvm.tir.Stmt, pred) -> int:
    count = 0
    loop_multiplier = 1

    @tir.functor.visitor
    class CountVisitor(tir.PyStmtExprVisitor):
        def visit_for_(self, op):
            nonlocal loop_multiplier
            extent = _static_positive_int(op.extent)
            if extent is None:
                super().visit_for_(op)
                return

            prev = loop_multiplier
            loop_multiplier *= extent
            super().visit_for_(op)
            loop_multiplier = prev

        def visit_call_(self, call):
            nonlocal count
            if pred(call):
                count += loop_multiplier
            super().visit_call_(call)

    CountVisitor().visit_stmt(stmt)
    return count


def _count_wgmma_ops_with_static_loop_multiplier(stmt: tvm.tir.Stmt) -> int:
    return _count_calls_with_static_loop_multiplier(stmt, _is_wgmma_call)


def _collect_wgmma_mnks_with_static_loop_multiplier(stmt: tvm.tir.Stmt) -> Dict[Tuple[int, int, int], int]:
    mnks: Dict[Tuple[int, int, int], int] = {}
    loop_multiplier = 1

    @tir.functor.visitor
    class CollectVisitor(tir.PyStmtExprVisitor):
        def visit_for_(self, op):
            nonlocal loop_multiplier
            extent = _static_positive_int(op.extent)
            if extent is None:
                super().visit_for_(op)
                return

            prev = loop_multiplier
            loop_multiplier *= extent
            super().visit_for_(op)
            loop_multiplier = prev

        def visit_call_(self, call):
            mnk = _wgmma_call_mnk(call)
            if mnk is not None:
                mnks[mnk] = mnks.get(mnk, 0) + loop_multiplier
            super().visit_call_(call)

    CollectVisitor().visit_stmt(stmt)
    return mnks


def _collect_wgmma_descs_with_static_loop_multiplier(
    stmt: tvm.tir.Stmt,
) -> Dict[Tuple[str, Optional[Tuple[int, int, int]]], int]:
    descs: Dict[Tuple[str, Optional[Tuple[int, int, int]]], int] = {}
    loop_multiplier = 1

    @tir.functor.visitor
    class CollectVisitor(tir.PyStmtExprVisitor):
        def visit_for_(self, op):
            nonlocal loop_multiplier
            extent = _static_positive_int(op.extent)
            if extent is None:
                super().visit_for_(op)
                return

            prev = loop_multiplier
            loop_multiplier *= extent
            super().visit_for_(op)
            loop_multiplier = prev

        def visit_call_(self, call):
            desc = _wgmma_call_desc(call)
            if desc is not None:
                descs[desc] = descs.get(desc, 0) + loop_multiplier
            super().visit_call_(call)

    CollectVisitor().visit_stmt(stmt)
    return descs


def _count_tma_ops_with_static_loop_multiplier(stmt: tvm.tir.Stmt) -> int:
    count = _count_calls_with_static_loop_multiplier(stmt, _is_tma_call)
    return count


def _is_sync_like(stmt: tvm.tir.Stmt, *, nested: bool) -> bool:
    sync_ops = {
        "tl.mbarrier_wait_parity",
        "tir.ptx_arrive_barrier",
        "tir.ptx_arrive_barrier_expect_tx",
        "tir.ptx_wait_barrier",
        "tir.ptx_cp_async_barrier",
        "tl.ptx_cp_async_barrier_noinc",
        "tl.ptx_fence_barrier_init",
        # WGMMA completion fence — must not be reordered by the SMT scheduler.
        "tl.wait_wgmma",
    }

    found = False

    def _v(node):
        nonlocal found
        if found:
            return
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            if node.op.name in sync_ops:
                found = True

    if nested:
        tvm.tir.stmt_functor.post_order_visit(stmt, _v)
        return found

    # top-level only: Evaluate(Call(...))
    if isinstance(stmt, tvm.tir.Evaluate) and isinstance(stmt.value, tvm.tir.Call) and isinstance(stmt.value.op, tvm.ir.Op):
        return stmt.value.op.name in sync_ops
    return False


def _buffer_scopes(bufs: Set[tvm.tir.Buffer]) -> Set[str]:
    scopes: Set[str] = set()
    for b in bufs:
        try:
            scopes.add(b.scope)
        except Exception:
            pass
    return scopes


def _collect_rw_regions(
    stmt: tvm.tir.Stmt, buffer_var_map: Dict[tvm.tir.Var, tvm.tir.Buffer]
) -> Tuple[List[tvm.tir.BufferRegion], List[tvm.tir.BufferRegion]]:
    # Wrap statement into a Block to leverage TVM's region analysis.
    blk = tvm.tir.Block([], [], [], "", stmt)
    reads, writes = tvm.tir.analysis.get_block_read_write_region(blk, buffer_var_map)
    return list(reads), list(writes)


def _buf_scope_str(buf: tvm.tir.Buffer) -> str:
    # tvm.tir.Buffer.scope is a method in modern TVM; older builds expose a property.
    s = buf.scope
    if callable(s):
        try:
            s = s()
        except Exception:
            return ""
    return str(s) if s is not None else ""


def _is_shared(buf: tvm.tir.Buffer) -> bool:
    return _buf_scope_str(buf) in ("shared", "shared.dyn")


def _is_global(buf: tvm.tir.Buffer) -> bool:
    # In TVM/TIR the default global-memory scope is an empty string "" rather than the
    # explicit string "global".  Matching only "global" would miss most T.copy(global->shared)
    # patterns and cause the SMT path to exit early with reason_code=2 (no_producer).
    return _buf_scope_str(buf) in ("", "global")


def _is_intrinsic_producer(stmt: tvm.tir.Stmt) -> bool:
    # Match TileLang intrinsics: tl.tma_load / tl.tma_load_im2col / tl.ptx_cp_async
    op_names = _call_op_names(stmt)
    return ("tl.tma_load" in op_names) or ("tl.tma_load_im2col" in op_names) or ("tl.ptx_cp_async" in op_names)


def _is_true_tma_producer(stmt: tvm.tir.Stmt) -> bool:
    op_names = _call_op_names(stmt)
    return ("tl.tma_load" in op_names) or ("tl.tma_load_im2col" in op_names)


def _is_copy_pattern_producer(reads: List[tvm.tir.BufferRegion], writes: List[tvm.tir.BufferRegion]) -> bool:
    has_shared_write = any(_is_shared(wr.buffer) for wr in writes)
    if not has_shared_write:
        return False
    if not reads:
        return False
    for rd in reads:
        if not _is_global(rd.buffer):
            return False
    return True


def _touches_non_shared_global(
    reads: List[tvm.tir.BufferRegion], writes: List[tvm.tir.BufferRegion]
) -> bool:
    """Conservative safety predicate for software pipelining.

    Return True if stmt touches buffers outside {global, shared, shared.dyn}.
    Moving such statements across pipeline stages without precise lifetime/liveness
    modeling is often unsafe.
    """
    for r in reads:
        if r.buffer.scope not in ("global", "shared", "shared.dyn"):
            return True
    for w in writes:
        if w.buffer.scope not in ("global", "shared", "shared.dyn"):
            return True
    return False


def _collect_internal_alloc_data_vars(stmt: tvm.tir.Stmt) -> Set[tvm.tir.Var]:
    """Collect Buffer.data vars for alloc_buffers inside this statement subtree."""
    out: Set[tvm.tir.Var] = set()

    def visit(s):
        if isinstance(s, tvm.tir.Block):
            for b in s.alloc_buffers:
                out.add(b.data)

    tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    return out


def _touches_external_non_shared_global(
    stmt: tvm.tir.Stmt, reads: List[tvm.tir.BufferRegion], writes: List[tvm.tir.BufferRegion],
    func_alloc_vars: Optional[Set[tvm.tir.Var]] = None
) -> bool:
    """Return True if stmt touches non-(global/shared) buffers that are *external* to this stmt.

    This treats local/descriptor scratch buffers allocated *inside* the stmt as safe, because
    moving the stmt across stages does not create cross-stmt lifetime hazards for those buffers.
    
    If func_alloc_vars is provided, buffers allocated at function level (e.g., fragment buffers
    in kernel body) are also treated as internal/safe for stage movement within a single iteration.
    """
    internal = _collect_internal_alloc_data_vars(stmt)
    # Also include function-level allocs as "internal" for stage movement purposes
    if func_alloc_vars:
        internal = internal | func_alloc_vars

    def _is_external(buf: tvm.tir.Buffer) -> bool:
        try:
            return buf.data not in internal
        except Exception:
            return True

    for r in reads:
        if r.buffer.scope not in ("global", "shared", "shared.dyn") and _is_external(r.buffer):
            return True
    for w in writes:
        if w.buffer.scope not in ("global", "shared", "shared.dyn") and _is_external(w.buffer):
            return True
    return False

from heddle.scheduler.smt import ResourceType  # type: ignore
class OpIssueAndLatencyTable :
    
    # 指令的发射延迟
    @staticmethod
    def get_issue_execute_cycle(opTy : ResourceType, desc_info : List) -> List[int, int] :  # issue, execute
        if opTy is ResourceType.TensorCore :
            '''
            rs & ss diff :
            SM90_64xNx16_F32F16F16_SS   M=64,N=64,K=16 elapsed_time: 2.87264ms 4.56277TFLOPS, latancy=32.0051
            SM90_64xNx16_F32F16F16_SS   M=64,N=32,K=16 elapsed_time: 2.15971ms 3.03448TFLOPS, latancy=24.0035
            SM90_64xNx16_F32F16F16_RS   M=64,N=32,K=16 elapsed_time: 1.45213ms 4.5131TFLOPS, latancy=16.0041
            SM90_64xNx16_F32F16F16_SS   M=64,N=16,K=16 elapsed_time: 1.80304ms 1.81738TFLOPS, latancy=20.0044
            SM90_64xNx16_F32F16F16_RS   M=64,N=16,K=16 elapsed_time: 1.18122ms 2.77409TFLOPS, latancy=13.0048
            SM90_64xNx16_F32F16F16_SS   M=64,N=8,K=16 elapsed_time: 1.62624ms 1.00748TFLOPS, latancy=18.0046
            SM90_64xNx16_F32F16F16_RS   M=64,N=8,K=16 elapsed_time: 1.18074ms 1.38761TFLOPS, latancy=13.005

            rs & ss same :
            SM90_64xNx16_F16F16F16_SS   M=64,N=256,K=16  elapsed_time: 11.4083ms 4.59568TFLOPS, latancy=128.003
            SM90_64xNx8_F32TF32TF32_SS_TN   M=64,N=256,K=8  elapsed_time: 11.4115ms 2.29719TFLOPS, latancy=128.005
            SM90_64xNx32_F32E4M3E4M3_SS_TN   M=64,N=256,K=32  elapsed_time: 11.4289ms 9.17481TFLOPS, latancy=128.005
            SM90_64xNx16_F32F16F16_SS   M=64,N=256,K=16 elapsed_time: 11.4108ms 4.59467TFLOPS, latancy=128.005
            SM90_64xNx16_F32F16F16_SS   M=64,N=128,K=16 elapsed_time: 5.71731ms 4.58509TFLOPS, latancy=64.0046
            
            该表格实际包含了issue + execute。真实 issue/execute 需求解：
            设 m64n8k16 发射为I，执行为T: 
            m64n8k16 -> I+T = 18 
            m64n32k16 -> 4*I + T1 = 24    
            m64n256k16  ->   32*I + T2 = 128  
            m64n64k16 : 8*I+T3 = 32 
            
            通过两两比对，且假定 mnk 规模越大，T越久，I严格遵守正比关系，那么:
            3*I + T1 - T = 3*I + (4*delta-1)*T  = 6  
            31*I + T2-T = 31*I + (32*delta-1)*T = 110
            7*I+T3 - T = 7*I+ (8*delta-1)*T = 14  
            
            差距过大，说明 I 之间存在比例关系， T也有比例关系
            
            '''
            [mnk_key, kind] = desc_info
            _table ={
                                # ss_total,rs_total, issue_estimated
                "m64n32k16" :   [24,  16 , 4] ,
                "m64n16k16" :   [20,  13,  2],
                "m64n8k16" :    [18,  13 , 1] ,
                "m64n256k16"  : [128, 128, 32] ,
                "m64n256k8"  :  [128, 128, 16] ,
                "m64n256k32"  : [128, 128, 64] ,
                "m64n256k16" :  [128, 128, 32] ,
                "m64n128k16" :  [64 , 64, 16] ,
                "m64n64k16" :   [32 , 32,  8] ,
            }
            row = _table.get(mnk_key)
            return [row[2], row[kind] - row[2]]  # 执行周期需要减掉发射周期
        
        if opTy is ResourceType.TMA :
            return [1, 280]
        if opTy is ResourceType.ALU :
            return [1,4]
        if opTy is ResourceType.SFU :
            return [1,18]


def _detect_wgmma_issue_cycles(stmt: tvm.tir.Stmt) -> int:
    """Return the summed TensorCore issue occupancy for WGMMA calls."""
    wgmma_descs = _collect_wgmma_descs_with_static_loop_multiplier(stmt)
    if not wgmma_descs:
        return max(_count_wgmma_ops_with_static_loop_multiplier(stmt), 1)

    issue_cycles = 0
    for (kind, mnk), count in wgmma_descs.items():
        if mnk is None:
            issue_cycles += int(count)
            continue
        m, n, k = mnk
        mnk_key = f"m{m}n{n}k{k}"
        if kind == "rs":
            kind_idx = 1
        elif kind == "ss":
            kind_idx = 0
        else:
            assert False, f"invalid kind detected - {kind}"
        issue, _ = OpIssueAndLatencyTable.get_issue_execute_cycle(
            ResourceType.TensorCore, [mnk_key, kind_idx]
        )
        issue_cycles += int(issue) * int(count)
    return max(issue_cycles, 1)



def _detect_op_latency_and_resource(stmt: tvm.tir.Stmt) -> tuple:
    """Detect operation type and return (latency, ResourceType).

    Reference latencies from Hopper architecture:
    - WGMMA (warp_group_dot): ~27 cycles, TensorCore
    - TMA load issue: ~20 cycles, TMA
    - Reduce (max/sum): ~3 cycles, ALU
    - SFU (exp2, rsqrt, log2): ~25 cycles, SFU
    - Copy/Fill: ~1 cycle, ALU
    - ALU (add, mul, etc.): ~1 cycle, ALU
    """
    try:
        from heddle.scheduler.smt import ResourceType  # type: ignore
    except Exception:  # pragma: no cover
        from heddle.scheduler.smt import ResourceType  # type: ignore

    wgmma_op_count = _count_wgmma_ops_with_static_loop_multiplier(stmt)
    if wgmma_op_count > 0:
        idx = 0  # ss=0, rs = 1
        mnk_key = ""
        wgmma_descs = _collect_wgmma_descs_with_static_loop_multiplier(stmt)
        if wgmma_descs:
            desc_parts = []
            for (kind, mnk), count in sorted(wgmma_descs.items()):
                if mnk is None:
                    desc_parts.append(f"{kind}:unknownx{count}")
                else:
                    m, n, k = mnk
                    mnk_key = f"m{m}n{n}k{k}"
                    desc_parts.append(f"{kind}:m{m}n{n}k{k}x{count}")
                    if kind == 'rs':
                        idx = 1
                    elif kind == 'ss':
                        idx = 0
                    else:
                        assert False, f"invalid kind detected - {kind}"
            desc_str = ", ".join(desc_parts)
        else:
            desc_str = "unknown"
        # print(f'[d] wgmma_op_count = {wgmma_op_count}, wgmma_desc = {desc_str}, stmt = {stmt.script()}' )
        # print('--------------\n')
        # 单个wgmma执行周期
        [issue , execute] = OpIssueAndLatencyTable.get_issue_execute_cycle(ResourceType.TensorCore, [mnk_key,idx] )
        # 循环：不能简单xN。 正确计算方式 = interval * (N-1) + Latency, 这里 interval 简化为等于issue_time
        return _detect_wgmma_issue_cycles(stmt) + execute, ResourceType.TensorCore

    tma_op_count = _count_tma_ops_with_static_loop_multiplier(stmt)
    if tma_op_count > 0:
        [issue, execute] = OpIssueAndLatencyTable.get_issue_execute_cycle(ResourceType.TMA, [])
        return  execute, ResourceType.TMA
        # return 20 * tma_op_count, ResourceType.TMA

    op_names = _call_op_names(stmt)
    
    # Check for legacy PTX MMA (non-WGMMA, ~8 cycles on Hopper)
    ptx_mma_ops = {"tl.ptx_mma", "tl.ptx_mma_sp", "tl.ptx_mma_sm70"}
    if op_names & ptx_mma_ops:
        return 8, ResourceType.TensorCore

    # Check for ldmatrix (shared memory → register, ~1 cycle ALU)
    if any(n.startswith("tl.ptx_ldmatrix") for n in op_names):
        return 1, ResourceType.ALU
    
    # Check for Reduce operations
    if ("tl.tl_reduce" in op_names) or ("tl.reduce_max" in op_names) or ("tl.reduce_sum" in op_names):
        return 3, ResourceType.ALU
    
    for n in op_names:
        if n.startswith("extern:") and "reduce" in n.lower():
            return 3, ResourceType.ALU
    
    # Check for SFU operations (exp2, rsqrt, log2, etc.)
    sfu_ops = {"tir.exp2", "tir.rsqrt", "tir.log2", "tir.exp", "tir.log", "tir.sqrt", "tir.tanh", "tir.sigmoid"}
    if op_names & sfu_ops:
        return 25, ResourceType.SFU
    
    # Check for copy/fill operations
    if ("tl.tl_copy" in op_names) or ("tl.tl_fill" in op_names):
        return 1, ResourceType.ALU
    
    # Default: ALU with latency 1
    return 1, ResourceType.ALU


def _extract_num_threads(func: tvm.tir.PrimFunc) -> int:
    """Extract the threadIdx.x launch extent from a GPU kernel PrimFunc body.

    Walks the IR looking for ``thread_extent`` AttrStmt nodes whose IterVar
    name is ``threadIdx.x``.  Returns 128 if none is found (conservative
    default: overestimates per-thread footprint rather than under-estimating).
    """
    result = [128]

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.AttrStmt):
            try:
                if (str(stmt.attr_key) == "thread_extent"
                        and hasattr(stmt.node, "var")
                        and stmt.node.var.name == "threadIdx.x"):
                    result[0] = int(stmt.value)
            except Exception:
                pass

    tvm.tir.stmt_functor.post_order_visit(func.body, _visit)
    return result[0]


def _is_stage_movable(info: "_StmtInfo") -> bool:
    """Whether a statement is safe to move across stages (conservative)."""
    if info.is_producer:
        return False
    # Only treat *external* local/fragment temporaries as unsafe. Internal alloc_buffers
    # (e.g. descriptor scratch) do not create cross-stmt hazards.
    if info.touches_external_local:
        return False
    if info.is_wgmma:
        return False
    if info.is_sync_top or info.is_sync_nested:
        return False
    return True


def _is_wgmma_like(stmt: tvm.tir.Stmt) -> bool:
    op_names = _call_op_names(stmt)
    # High-level TL gemm (before lowering) shows up as tl.tl_gemm / tl.tl_gemm_sp sometimes.
    if ("tl.tl_gemm" in op_names) or ("tl.tl_gemm_sp" in op_names):
        return True
    # Lowered PTX-level intrinsics
    if ("tl.ptx_wgmma_ss" in op_names) or ("tl.ptx_wgmma_rs" in op_names):
        return True
    # Or lowered via call_extern to tcgen5mma/wgmma markers.
    for n in op_names:
        if not n.startswith("extern:"):
            continue
        f = n[len("extern:") :]
        if f.startswith("tl::tcgen5mma_gemm_") or f.startswith("tl::wgmma") or f.startswith("tl::tcgen05"):
            return True
    return False


@dataclass
class _StmtInfo:
    idx: int
    stmt: tvm.tir.Stmt
    reads: List[tvm.tir.BufferRegion]
    writes: List[tvm.tir.BufferRegion]
    is_producer: bool
    is_true_tma: bool
    is_sync_top: bool
    is_sync_nested: bool
    is_wgmma: bool
    touches_local: bool
    touches_external_local: bool
    is_wait_barrier : bool


import tvm
from tvm import tir

@tir.functor.visitor
class FullBarrierExtractor(tir.PyStmtExprVisitor):
    def __init__(self):
        super().__init__()
        # 用来存放提取到的 barrier 及其对应的操作类型
        # 格式: (barrier_expr, op_type, stmt_or_call_node)
        self.barrier_operations = []

    def visit_call_(self, call):
        # 针对以 Call 形式存在的原语 (通常是外部函数调用或特定 Op)
        if isinstance(call.op, tvm.ir.Op):
            op_name = call.op.name

            # 1. 处理 tma_load
            if op_name in {"tl.tma_load", "tl.tma_load_im2col", "tir.tma_load"}:
                self.barrier_operations.append((call.args[1], "TMA_LOAD", call))

            # 2. 处理 ptx_arrive_barrier
            elif op_name == "tir.ptx_arrive_barrier":
                self.barrier_operations.append((call.args[0], "ARRIVE", call))

            elif op_name == "tir.ptx_arrive_barrier_expect_tx":
                self.barrier_operations.append((call.args[0], "ARRIVE_EXPECT_TX", call))

            # 3. 处理 mbarrier_expect_tx
            elif op_name in {"tl.mbarrier_expect_tx", "tir.mbarrier_expect_tx"}:
                self.barrier_operations.append((call.args[0], "EXPECT_TX", call))

            # 4. 处理 mbarrier_wait_parity
            elif op_name in {"tl.mbarrier_wait_parity", "tir.mbarrier_wait_parity"}:
                self.barrier_operations.append((call.args[0], "WAIT", call))

        # 继续向下遍历
        super().visit_call_(call)



def _build_stmt_infos(
    seq: tvm.tir.SeqStmt,
    buffer_var_map: Dict[tvm.tir.Var, tvm.tir.Buffer],
    func_alloc_vars: Set[tvm.tir.Var],
) -> Tuple[List[_StmtInfo], Dict[tvm.tir.PrimExpr, Tuple[str, int]]]:
    import tvm
    from collections import defaultdict
    grouped_barriers = defaultdict(list)  # Dict[ simplified_expr, (op_type, op_idx) ]
    analyzer = tvm.arith.Analyzer()

    """Build a list of _StmtInfo for every statement in *seq*."""
    infos: List[_StmtInfo] = []
    for i, s in enumerate(seq.seq):
        try:
            reads, writes = _collect_rw_regions(s, buffer_var_map)
        except Exception:
            reads, writes = [], []
        is_intr = _is_intrinsic_producer(s)
        is_true_tma = _is_true_tma_producer(s)
        is_copy = _is_copy_pattern_producer(reads, writes)
        is_prod = bool(is_intr or is_copy)

        extractor = FullBarrierExtractor()
        extractor.visit_stmt(s)
        
        is_wait_barrier = False
        
        # 假设已经通过 visitor 拿到了 self.barrier_operations
        for barrier_expr, op_type, node in extractor.barrier_operations:
            # 1. 表达式化简 (比如把 k % 2 + 1 化简为标准形式)
            simplified_expr = analyzer.simplify(barrier_expr)

            # 2. 寻找是否已有结构相同的 barrier 分组
            found_key = None
            for existing_key in grouped_barriers.keys():
                if tvm.ir.structural_equal(simplified_expr, existing_key):
                    found_key = existing_key
                    break
            if found_key is not None:
                grouped_barriers[found_key].append((op_type, i))
            else:
                grouped_barriers[simplified_expr].append((op_type, i))
            # 3. 记录信息-是否为 wait_barrier 操作
            if op_type == 'WAIT':
                is_wait_barrier = True

        # 此时：
        # grouped_barriers 里面就会有两个明确的 key：
        # 一个代表 `k % 2 + 1` (控制 K 矩阵的 TMA)
        # 一个代表 `k % 2 + 3` (控制 V 矩阵的 TMA)

        infos.append(
            _StmtInfo(
                idx=i,
                stmt=s,
                reads=reads,
                writes=writes,
                is_producer=is_prod,
                is_true_tma=is_true_tma,
                is_sync_top=_is_sync_like(s, nested=False),
                is_sync_nested=_is_sync_like(s, nested=True),
                is_wgmma=_is_wgmma_like(s),
                touches_local=_touches_non_shared_global(reads, writes),
                touches_external_local=_touches_external_non_shared_global(
                    s, reads, writes, func_alloc_vars
                ),
                is_wait_barrier=is_wait_barrier
            )
        )
    return infos, grouped_barriers


def _estimate_buffer_footprint_bytes(buf: tvm.tir.Buffer) -> int:
    """Estimate the byte size of a TIR buffer from its shape and dtype."""
    dtype_bits = {"float16": 16, "float32": 32, "float64": 64,
                  "int8": 8, "int16": 16, "int32": 32, "int64": 64,
                  "uint8": 8, "uint16": 16, "uint32": 32, "bool": 8}
    bits = dtype_bits.get(str(buf.dtype), 16)
    elems = 1
    for dim in buf.shape:
        try:
            elems *= int(dim)
        except (TypeError, ValueError):
            elems *= 128  # conservative fallback for symbolic dims
    return elems * bits // 8


def _schedule_with_smt(
    infos: List[_StmtInfo], *,
    ii: int,
    use_precise_latency: bool = False,
    debug: bool = False,
    max_ii_search: int = 0,
    disallow_spills: bool = False,
    use_spill_concurrency: bool = True,
    include_incoming_live: bool = True,
    num_threads: int = 128,
) -> Optional[Dict[int, int]]:
    """Return absolute time T per statement idx (only for consumer stmts).

    The scheduler has two modes:
      - Phase A only (original): minimizes II with FU capacity constraints.
      - Phase A+B (joint): additionally enforces register-capacity liveness
        constraints so that the returned schedule is guaranteed to fit within
        the hardware register budget. If the joint solve is UNSAT (registers
        cannot hold the pipeline), it automatically reduces num_stages and
        retries before falling back to Phase A.

    Args:
        infos: List of statement info objects
        ii: Initiation interval (typically num_stages)
        use_precise_latency: If True, use architecture-aware latency modeling
                            based on operation type. If False, use latency=1 for all.
        debug: If True, print debug information about the scheduling process.
        max_ii_search: If > 0, when the initial II is UNSAT, search up to this II.
                      This enables auto-relaxation when precise latency makes constraints tight.
    """
    try:
        from heddle.scheduler.smt import HeddleScheduler, OpNode, ResourceType, OutputValue, StorageKind  # type: ignore
    except Exception:  # pragma: no cover
        from heddle.scheduler.smt import HeddleScheduler, OpNode, ResourceType, OutputValue, StorageKind  # type: ignore

    nodes: List[OpNode] = []
    idx_to_node: Dict[int, OpNode] = {}

    if debug:
        print(f"[SMT Debug] Total statements: {len(infos)}, II={ii}, precise_latency={use_precise_latency}")
        movable_count = sum(1 for info in infos if _is_stage_movable(info))
        producer_count = sum(1 for info in infos if info.is_producer)
        wgmma_count = sum(1 for info in infos if info.is_wgmma)
        sync_count = sum(1 for info in infos if info.is_sync_top or info.is_sync_nested)
        local_count = sum(1 for info in infos if info.touches_external_local)
        print(f"[SMT Debug] Movable: {movable_count}, Producers: {producer_count}, WGMMA: {wgmma_count}, Sync: {sync_count}, TouchesLocal: {local_count}")

    for info in infos:
        if not _is_stage_movable(info):
            continue

        if use_precise_latency:
            latency, rty = _detect_op_latency_and_resource(info.stmt)
        else:
            rty = ResourceType.ALU
            latency = 1

        if debug:
            op_names = _call_op_names(info.stmt)
            print(f"[SMT Debug] stmt[{info.idx}]: latency={latency}, resource={rty.value}, ops={op_names}")

        # --- Build OutputValue list from written buffers ---
        # For RMEM (register/fragment) outputs, Phase B's register-capacity
        # constraint is per-thread.  _estimate_buffer_footprint_bytes returns
        # the *total* buffer size; divide by num_threads to get per-thread bytes.
        outputs: List[OutputValue] = []
        for wr in info.writes:
            storage = StorageKind.SMEM if _is_shared(wr.buffer) else StorageKind.RMEM
            fp = _estimate_buffer_footprint_bytes(wr.buffer)
            if storage == StorageKind.RMEM and num_threads > 1:
                fp = max(1, fp // num_threads)
            outputs.append(OutputValue(
                name=f"s{info.idx}_w_{wr.buffer.name}",
                storage=storage,
                footprint_bytes=fp,
            ))

        n = OpNode(name=f"s{info.idx}", resource_type=rty, latency=latency, outputs=outputs)
        nodes.append(n)
        idx_to_node[info.idx] = n

    # --- Program-order chain ---
    prev_movable: Optional[int] = None
    for info in infos:
        if not _is_stage_movable(info):
            continue
        if prev_movable is None:
            prev_movable = info.idx
            continue
        a = idx_to_node.get(prev_movable)
        b = idx_to_node.get(info.idx)
        if a is not None and b is not None:
            b.add_dependency(a, distance=0)
        prev_movable = info.idx

    # --- Shared-buffer RAW dependencies ---
    last_shared_writer: Dict[tvm.tir.Buffer, int] = {}
    for info in infos:
        for wr in info.writes:
            if _is_shared(wr.buffer):
                last_shared_writer[wr.buffer] = info.idx

        if not _is_stage_movable(info):
            continue
        for rd in info.reads:
            if not _is_shared(rd.buffer):
                continue
            widx = last_shared_writer.get(rd.buffer)
            if widx is None or widx == info.idx:
                continue
            if widx in idx_to_node and info.idx in idx_to_node:
                idx_to_node[info.idx].add_dependency(idx_to_node[widx], distance=0)

    fu_caps = {
        ResourceType.TMA: 1,
        ResourceType.TensorCore: 1,
        ResourceType.ALU: 64,
        ResourceType.SFU: 16,
    }
    actual_max_ii = max(ii, max_ii_search) if max_ii_search > 0 else ii

    # --- Try Phase A+B (joint) first: schedule with register capacity ---
    _phase_b_reg_limit = 240 * 4  # 240 regs × 4 bytes/reg = 960 bytes (per-thread)
    has_outputs = any(n.outputs for n in nodes)
    # Skip Phase B if the largest *per-thread* RMEM footprint exceeds the register limit.
    # Only RMEM outputs matter for register capacity; SMEM outputs are unlimited here.
    # footprint_bytes for RMEM outputs has already been divided by num_threads above,
    # so this comparison is in per-thread bytes.
    max_rmem_footprint = max(
        (o.footprint_bytes for n in nodes for o in n.outputs
         if o.storage == StorageKind.RMEM), default=0
    )
    phase_b_feasible = max_rmem_footprint <= _phase_b_reg_limit
    # Phase A is needed either as: (a) window-sizer for Phase B, or (b) direct fallback.
    # Run it once here so we never compute it twice.
    _sched_a = HeddleScheduler(nodes, fu_caps=fu_caps)
    _phase_a_sol = _sched_a.schedule(
        min_ii=max(1, ii), max_ii=max(1, actual_max_ii), optimize=True
    )
    _max_asap_time = max(_phase_a_sol.values()) if _phase_a_sol else 0

    if has_outputs and ii > 1 and phase_b_feasible:
        # Use Phase A's ASAP times to size the Phase B modulo-scheduling window.
        # Without this, max_window=try_ii*8 (e.g. 16 for ii=2) is too small
        # when ASAP times are large (e.g. 62 for FA FWD), causing Phase B to
        # always return UNSAT.
        for try_ii in range(ii, 0, -1):
            sched_joint = HeddleScheduler(
                nodes,
                fu_caps=fu_caps,
                reg_limit=_phase_b_reg_limit,
                disallow_spills=disallow_spills,
                use_spill_concurrency=use_spill_concurrency,
                include_incoming_live=include_incoming_live,
            )
            _dynamic_max_window = max(try_ii * 8, _max_asap_time + 2 * try_ii)
            joint_result = sched_joint.schedule_joint(
                min_ii=max(1, try_ii),
                max_ii=max(1, actual_max_ii),
                max_window=_dynamic_max_window,
                window_step=max(1, try_ii),
                optimize=True,
            )
            if joint_result is not None:
                sol = joint_result["schedule"]
                if debug:
                    peak = joint_result.get("reg_peak", {})
                    print(f"[SMT Debug] Joint schedule found! II={joint_result['ii']}, "
                          f"L={joint_result.get('window', 'n/a')}, reg_peak={peak}, times={sol}")
                out: Dict[int, int] = {}
                for i, n in idx_to_node.items():
                    if n.name in sol:
                        out[i] = int(sol[n.name])
                return out
            if debug:
                print(f"[SMT Debug] Joint UNSAT at II={try_ii}, trying II={try_ii - 1}")
        if debug:
            print(f"[SMT Debug] Joint scheduling exhausted, falling back to Phase A")

    # --- Fallback: Phase A only (result already computed above) ---
    sol = _phase_a_sol
    if sol is None:
        if debug:
            print(f"[SMT Debug] Phase A UNSAT! (tried II from {ii} to {actual_max_ii})")
        return None

    if debug:
        print(f"[SMT Debug] Phase A schedule found! Times: {sol}")

    out = {}
    for i, n in idx_to_node.items():
        out[i] = int(sol[n.name])
    return out


def _build_group_schedule_nodes(
    groups: List[List[int]],
    infos: List[_StmtInfo],
    *,
    use_precise_latency: bool,
) -> Tuple[List[OpNode], Dict[int, OpNode], Dict[int, bool], Dict[int, int]]:
    """Build group-level OpNodes and dependencies for modulo scheduling."""
    try:
        from heddle.scheduler.smt import OpNode, ResourceType  # type: ignore
    except Exception:  # pragma: no cover
        from heddle.scheduler.smt import OpNode, ResourceType  # type: ignore
    # Map stmt idx -> group idx
    stmt_to_group: Dict[int, int] = {}
    for gi, grp in enumerate(groups):
        for idx in grp:
            stmt_to_group[idx] = gi

    group_is_producer: Dict[int, bool] = {}
    group_nodes: List[OpNode] = []
    group_idx_to_node: Dict[int, OpNode] = {}

    for gi, grp in enumerate(groups):
        # Producer group if all statements are producers
        is_prod = True
        for idx in grp:
            if idx >= len(infos):
                continue
            if not infos[idx].is_producer:
                is_prod = False
                break
        group_is_producer[gi] = is_prod
        if is_prod:
            continue

        reservation: List[Dict[ResourceType, int]] = []
        total_latency = 0
        for idx in grp:
            if idx >= len(infos):
                continue
            info = infos[idx]
            if info.is_producer:
                continue
            if use_precise_latency:
                latency, rty = _detect_op_latency_and_resource(info.stmt)
            else:
                latency, rty = 1, ResourceType.ALU
            lat = max(int(latency), 1)
            reservation.extend([{rty: 1} for _ in range(lat)])
            total_latency += lat

        if not reservation:
            reservation = [{ResourceType.ALU: 1}]
            total_latency = 1

        node = OpNode(
            name=f"g{gi}",
            resource_type=ResourceType.ALU,
            latency=total_latency,
            reservation=reservation,
        )
        group_nodes.append(node)
        group_idx_to_node[gi] = node

    # Build dependencies based on shared-buffer RAW among consumer groups
    last_shared_writer: Dict[tvm.tir.Buffer, int] = {}
    for info in infos:
        gi = stmt_to_group.get(info.idx, -1)
        # record writes
        for wr in info.writes:
            if _is_shared(wr.buffer):
                last_shared_writer[wr.buffer] = gi
        # add deps for reads
        if info.is_producer or gi < 0:
            continue
        if gi not in group_idx_to_node:
            continue
        for rd in info.reads:
            if not _is_shared(rd.buffer):
                continue
            wgi = last_shared_writer.get(rd.buffer, -1)
            if wgi < 0 or wgi == gi:
                continue
            if wgi in group_idx_to_node:
                group_idx_to_node[gi].add_dependency(group_idx_to_node[wgi], distance=0)

    return group_nodes, group_idx_to_node, group_is_producer, stmt_to_group


def _schedule_groups_with_order(
    groups: List[List[int]],
    infos: List[_StmtInfo],
    *,
    ii: int,
    use_precise_latency: bool,
    order_perm: Optional[Tuple[int, ...]] = None,
    max_ii_search: int = 0,
    relax_resources: bool = False,
) -> Optional[Dict[int, int]]:
    """Schedule groups with optional order constraints, returning group->time."""
    try:
        from heddle.scheduler.smt import HeddleScheduler, ResourceType  # type: ignore
    except Exception:  # pragma: no cover
        from heddle.scheduler.smt import HeddleScheduler, ResourceType  # type: ignore
    nodes, idx_to_node, _, _ = _build_group_schedule_nodes(
        groups, infos, use_precise_latency=use_precise_latency
    )
    # Add order constraints between consumer groups
    if order_perm:
        for a, b in zip(order_perm, order_perm[1:]):
            if a in idx_to_node and b in idx_to_node:
                idx_to_node[b].add_dependency(idx_to_node[a], distance=0)

    if relax_resources:
        fu_caps = {
            ResourceType.TMA: 1024,
            ResourceType.TensorCore: 1024,
            ResourceType.ALU: 1024,
            ResourceType.SFU: 1024,
        }
    else:
        fu_caps = {
            ResourceType.TMA: 1,
            ResourceType.TensorCore: 1,
            ResourceType.ALU: 64,
            ResourceType.SFU: 16,
        }
    sched = HeddleScheduler(nodes, fu_caps=fu_caps)
    actual_max_ii = max(ii, max_ii_search) if max_ii_search > 0 else ii
    sol = sched.schedule(min_ii=max(1, ii), max_ii=max(1, actual_max_ii), optimize=True)
    if sol is None:
        return None

    out: Dict[int, int] = {}
    for gi, n in idx_to_node.items():
        out[gi] = int(sol[n.name])
    return out


def _search_group_order_stage(
    groups: List[List[int]],
    infos: List[_StmtInfo],
    *,
    num_stages: int,
    use_precise_latency: bool,
    search_order: bool,
    debug: bool,
    stage_policy: str = "time",
    stage_offset: int = 0,
) -> Optional[Tuple[List[int], List[int], Dict[int, int]]]:
    """Search order/stage for given grouping; returns (order, stage, group_times)."""
    if not groups:
        return None

    # Identify consumer groups
    _, _, group_is_producer, _ = _build_group_schedule_nodes(
        groups, infos, use_precise_latency=use_precise_latency
    )
    consumer_groups = [gi for gi in range(len(groups)) if not group_is_producer.get(gi, False)]

    max_ii_search = num_stages * 4 if use_precise_latency else 0

    best_perm: Optional[Tuple[int, ...]] = None
    best_sol: Optional[Dict[int, int]] = None
    best_cost: Optional[int] = None
    top_candidates: List[Tuple[int, int, Tuple[int, ...], Dict[int, int]]] = []

    perms = [tuple(consumer_groups)]
    if search_order and len(consumer_groups) <= 8:
        perms = list(permutations(consumer_groups))
        if len(perms) > 24:
            perms = perms[:24]

    for perm in perms:
        sol = _schedule_groups_with_order(
            groups,
            infos,
            ii=max(1, num_stages),
            use_precise_latency=use_precise_latency,
            order_perm=perm if search_order else None,
            max_ii_search=max_ii_search,
            relax_resources=bool(use_precise_latency and search_order),
        )
        if sol is None:
            continue
        cost = max(sol.values()) if sol else 0
        # Track top candidates for debug
        sum_time = int(sum(sol.values())) if sol else 0
        top_candidates.append((cost, sum_time, perm, sol))
        top_candidates.sort(key=lambda x: (x[0], x[1]))
        if len(top_candidates) > 3:
            top_candidates = top_candidates[:3]

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm if search_order else None
            best_sol = sol

    if best_sol is None:
        return None

    def _build_order_stage_for_candidate(perm: Tuple[int, ...], sol: Dict[int, int]) -> Tuple[List[int], List[int]]:
        if not search_order:
            perm = tuple(sorted(consumer_groups, key=lambda gi: sol.get(gi, 0)))
        order: List[int] = [-1] * len(groups)
        for rank, gi in enumerate(perm):
            order[gi] = rank
        if stage_policy == "semantic":
            stmt_stage = _compute_semantic_stmt_stage(infos)
            stage = _compute_group_stage_from_stmt_stage(
                groups, stmt_stage, num_stages=num_stages, stage_offset=stage_offset
            )
        else:
            stage = [-1] * len(groups)
            consumer_times = [sol.get(gi, 0) for gi in consumer_groups]
            t0 = min(consumer_times) if consumer_times else 0
            smax = max(1, num_stages) - 1
            for gi in consumer_groups:
                t = sol.get(gi, t0)
                st = t - t0
                if st < 0:
                    st = 0
                if st > smax:
                    st = smax
                st = st + max(0, stage_offset)
                if st > smax:
                    st = smax
                stage[gi] = int(st)
        return order, stage

    # Build best order/stage
    if best_perm is None:
        best_perm = tuple(consumer_groups)
    order, stage = _build_order_stage_for_candidate(best_perm, best_sol)

    if debug:
        print(f"[SMT Debug] Group order search (top candidates):")
        print(f"[SMT Debug]   groups: {groups}")
        for rank, (cost, sum_time, perm, sol) in enumerate(top_candidates, 1):
            cand_order, cand_stage = _build_order_stage_for_candidate(perm, sol)
            print(f"[SMT Debug]   rank {rank}: cost={cost}, sum={sum_time}, order={cand_order}, stage={cand_stage}, times={sol}")

        # For small search spaces, dump all permutations for inspection
        if search_order and len(consumer_groups) <= 6:
            all_candidates: List[Tuple[int, int, Tuple[int, ...], Dict[int, int]]] = []
            for perm in perms:
                sol = _schedule_groups_with_order(
                    groups,
                    infos,
                    ii=max(1, num_stages),
                    use_precise_latency=use_precise_latency,
                    order_perm=perm,
                    max_ii_search=max_ii_search,
                    relax_resources=bool(use_precise_latency and search_order),
                )
                if sol is None:
                    continue
                cost = max(sol.values()) if sol else 0
                sum_time = int(sum(sol.values())) if sol else 0
                all_candidates.append((cost, sum_time, perm, sol))
            all_candidates.sort(key=lambda x: (x[0], x[1]))
            print(f"[SMT Debug] Group order search (all candidates, sorted by cost,sum):")
            for rank, (cost, sum_time, perm, sol) in enumerate(all_candidates, 1):
                cand_order, cand_stage = _build_order_stage_for_candidate(perm, sol)
                print(f"[SMT Debug]   rank {rank}: cost={cost}, sum={sum_time}, order={cand_order}, stage={cand_stage}, times={sol}")

    return order, stage, best_sol


def _build_group_templates(
    infos: List[_StmtInfo],
    stmt_stage: List[int],
    *,
    mode: str,
) -> List[List[int]]:
    """Build candidate group templates for WS pipeline pattern search."""
    groups: List[List[int]] = []

    if mode == "all_consumers":
        # Build groups in sequential statement order so that TVM's WarpSpecialized
        # pass (which uses a sequential cur_id counter) matches groups to statements
        # correctly.  Each producer is isolated; consecutive consumers between
        # producers are merged into one group.  This preserves the "all consumers
        # at the same stage" intent while keeping the group list contiguous.
        current_consumers: List[int] = []
        for info in infos:
            if info.is_producer:
                if current_consumers:
                    groups.append(current_consumers)
                    current_consumers = []
                groups.append([info.idx])
            else:
                current_consumers.append(info.idx)
        if current_consumers:
            groups.append(current_consumers)
        return groups

    current: List[int] = []
    prev_stage: Optional[int] = None
    prev_is_wgmma: Optional[bool] = None

    def _flush():
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for info in infos:
        if info.is_producer:
            _flush()
            groups.append([info.idx])
            prev_stage = None
            prev_is_wgmma = None
            continue

        st = stmt_stage[info.idx] if info.idx < len(stmt_stage) else 0
        is_wgmma = bool(info.is_wgmma)

        if not current:
            current = [info.idx]
            prev_stage = st
            prev_is_wgmma = is_wgmma
            continue

        split = False
        if mode == "stage_runs":
            split = (st != prev_stage)
        elif mode == "producer_boundary":
            split = False
        elif mode == "wgmma_split":
            split = (st != prev_stage) or (is_wgmma != prev_is_wgmma)
        else:
            split = (st != prev_stage)

        if split:
            _flush()
            current = [info.idx]
        else:
            current.append(info.idx)

        prev_stage = st
        prev_is_wgmma = is_wgmma

    _flush()
    return groups


def AutoTLPipelineSMTAnnotations():
    """SMT-based auto generation of tl_pipeline_group/order/stage for T.Pipelined loops.

    Minimal MVP:
    - Consumer nodes only (producer groups stay as -1/-1 markers)
    - Build RAW deps via shared buffer reads/writes
    - Use HeddleScheduler with fixed II == effective_num_stages
    - Emit per-stmt grouping for simplicity
    """

    @tvm.tir.transform.prim_func_pass(opt_level=0, name="tl.AutoTLPipelineSMTAnnotations")
    def _pass(func: tvm.tir.PrimFunc, mod, ctx: tvm.ir.transform.PassContext):  # type: ignore[no-redef]
        smt_debug_early = os.environ.get("TL_SMT_DEBUG", "0") == "1"
        smt_enabled = _get_bool_config(ctx, tilelang.PassConfigKey.TL_ENABLE_AUTO_TL_PIPELINE_SMT.value, False)
        # Fallback: check env var if pass config not registered in C++ yet
        if not smt_enabled and os.environ.get("TL_ENABLE_AUTO_TL_PIPELINE_SMT", "0") == "1":
            smt_enabled = True
        if smt_debug_early:
            print(f"[SMT Debug] Pass check: TL_ENABLE_AUTO_TL_PIPELINE_SMT = {smt_enabled}", file=sys.stderr, flush=True)
        if not smt_enabled:
            return func

        func_alloc_map = _collect_func_alloc_buffers(func)
        num_threads = _extract_num_threads(func)

        overwrite = _get_bool_config(ctx, tilelang.PassConfigKey.TL_OVERWRITE_AUTO_TL_PIPELINE_ANNOTATIONS.value, False)
        override_num_stages = _get_int_config(ctx, tilelang.PassConfigKey.TL_AUTO_TL_PIPELINE_NUM_STAGES.value, 0)
        # Experimental: allow emitting multi-stage schedule from SMT. Default off for correctness.
        enable_multistage = _get_bool_config(
            ctx, tilelang.PassConfigKey.TL_AUTO_TL_PIPELINE_SMT_MULTISTAGE.value, False
        )
        # Experimental: use architecture-aware latency modeling in SMT scheduler.
        use_precise_latency = _get_bool_config(
            ctx, tilelang.PassConfigKey.TL_SMT_USE_PRECISE_LATENCY.value, False
        )
        # Experimental: force group list and/or search order.
        force_group_raw = _get_str_config(ctx, tilelang.PassConfigKey.TL_SMT_FORCE_GROUP.value, "")
        forced_groups = _parse_group_override(force_group_raw)
        force_order_raw = _get_str_config(ctx, tilelang.PassConfigKey.TL_SMT_FORCE_ORDER.value, "")
        forced_order = _parse_order_override(force_order_raw)
        force_stage_raw = _get_str_config(ctx, tilelang.PassConfigKey.TL_SMT_FORCE_STAGE.value, "")
        forced_stage = _parse_stage_override(force_stage_raw)
        search_order = _get_bool_config(ctx, tilelang.PassConfigKey.TL_SMT_SEARCH_ORDER.value, False)
        stage_offset = _get_int_config(ctx, tilelang.PassConfigKey.TL_SMT_STAGE_OFFSET.value, 0)
        # Phase-B constraint flags (twill-style ablation knobs)
        disallow_spills = _get_bool_config(
            ctx, tilelang.PassConfigKey.TL_SMT_DISALLOW_SPILLS.value, False
        )
        use_spill_concurrency = _get_bool_config(
            ctx, tilelang.PassConfigKey.TL_SMT_USE_SPILL_CONCURRENCY.value, True
        )
        include_incoming_live = _get_bool_config(
            ctx, tilelang.PassConfigKey.TL_SMT_INCLUDE_INCOMING_LIVE.value, True
        )
        pattern_search_enable = _get_bool_config(
            ctx, tilelang.PassConfigKey.TL_SMT_PATTERN_SEARCH_ENABLE.value, False
        )
        # Debug: print SMT scheduling details (controlled by env var TL_SMT_DEBUG=1)
        smt_debug = os.environ.get("TL_SMT_DEBUG", "0") == "1"
        
        if smt_debug:
            print(f"[SMT Debug] Pass entered, smt_debug={smt_debug}", file=sys.stderr, flush=True)
            if forced_groups is not None:
                print(f"[SMT Debug] Forced groups enabled: {forced_groups}")
            if forced_order is not None:
                print(f"[SMT Debug] Forced order enabled: {forced_order}")
            if forced_stage is not None:
                print(f"[SMT Debug] Forced stage enabled: {forced_stage}")
            if search_order:
                print(f"[SMT Debug] Order search enabled: True")
            if stage_offset:
                print(f"[SMT Debug] Stage offset enabled: {stage_offset}")
            if pattern_search_enable:
                print(f"[SMT Debug] Pattern search enabled: True")

        # NOTE: This TVM build doesn't expose a Python StmtMutator class; use ir_transform instead.
        for_count = [0]  # Use list to allow mutation in nested function
        def _post(stmt):
            if not isinstance(stmt, tvm.tir.For):
                return stmt
            for_count[0] += 1

            ann = dict(stmt.annotations) if stmt.annotations is not None else {}
            num_stages = ann.get("num_stages")
            
            if num_stages is None or not isinstance(num_stages, tvm.tir.IntImm) or int(num_stages.value) <= 0:
                return stmt

            effective_num_stages = int(num_stages.value)
            if override_num_stages and override_num_stages > 0:
                effective_num_stages = int(override_num_stages)

            # ns=1 means no software pipelining; writing group/order/stage annotations for a
            # single-stage pipeline confuses WarpSpecialized (it expects multi-stage grouping).
            if effective_num_stages <= 1:
                if smt_debug:
                    print(f"[SMT Debug] Skipping: effective_num_stages={effective_num_stages} <= 1, no pipelining needed")
                return stmt

            if not overwrite and (("tl_pipeline_group" in ann) or ("tl_pipeline_order" in ann) or ("tl_pipeline_stage" in ann)):
                if smt_debug:
                    print(f"[SMT Debug] Skipping: pipeline annotations already exist and overwrite=False")
                return stmt

            # A-mode: reuse grouping produced by AutoTLPipelineAnnotations.
            #
            # The C++ auto pass emits `tl.debug_auto_pipeline_group/order/stage` (and may or may not
            # force `tl_pipeline_*` emission depending on config). For rapid convergence, when these
            # debug values exist, we treat the group as *fixed* and only emit `tl_pipeline_*` from it.
            #
            # This makes the SMT path align with the manual/auto baseline structure (few coarse groups)
            # instead of the previous per-statement singleton grouping.
            auto_group = ann.get("tl.debug_auto_pipeline_group", None)
            auto_order = ann.get("tl.debug_auto_pipeline_order", None)
            auto_stage = ann.get("tl.debug_auto_pipeline_stage", None)
            
            if smt_debug:
                print(f"[SMT Debug] Loop found: num_stages={effective_num_stages}, auto_group exists={auto_group is not None}")
            
            if auto_group is not None and forced_groups is None and not pattern_search_enable:
                # A-mode: Reuse grouping from C++ auto pass, but optimize order using SMT.
                # 
                # Strategy:
                # 1. Keep the grouping structure from auto_group
                # 2. Run SMT scheduling to get optimal times for each statement
                # 3. Reorder consumer groups based on their minimum SMT time
                # 4. Keep stage from auto_stage (it encodes correctness constraints)
                
                ann["tl.auto_tl_pipeline_smt_applied"] = tvm.tir.IntImm("int32", 1)
                ann["tl.debug_smt_pipeline_used_auto_group"] = tvm.tir.IntImm("int32", 1)

                # Validate auto_group: ensure no group mixes producers with
                # non-producers.  The WarpSpecialized rewriter requires
                # group_size == 1 for producer groups; if the annotation pass
                # created a mixed group, split it here to avoid a downstream
                # assertion failure.
                _validated_group = auto_group
                try:
                    _ag_seq, _ag_extra = _unwrap_to_seqstmt(stmt.body)
                    if _ag_seq is not None:
                        _ag_bvm: Dict[tvm.tir.Var, tvm.tir.Buffer] = {}
                        for _, b in func.buffer_map.items():
                            _ag_bvm[b.data] = b
                        _ag_bvm.update(func_alloc_map)
                        _ag_bvm.update(_ag_extra)
                        _ag_infos, _ = _build_stmt_infos(_ag_seq, _ag_bvm, set(func_alloc_map.keys()))
                        _n_ag = len(_ag_infos)
                        _needs_split = False
                        for g in auto_group:
                            g_list = [int(x) for x in g]
                            if len(g_list) > 1:
                                has_prod = any(_ag_infos[idx].is_producer for idx in g_list if idx < _n_ag)
                                if has_prod:
                                    _needs_split = True
                                    break
                        if _needs_split:
                            if smt_debug:
                                print("[SMT Debug] A-mode: auto_group has mixed producer+consumer group, "
                                      "splitting to per-stmt groups", file=sys.stderr, flush=True)
                            _validated_group = tvm.runtime.convert([[info.idx] for info in _ag_infos])
                            # Also fix order/stage to per-stmt
                            if auto_order is not None and auto_stage is not None:
                                _new_order = []
                                _new_stage = []
                                _rank = 0
                                for info in _ag_infos:
                                    if info.is_producer:
                                        _new_order.append(-1)
                                        _new_stage.append(-1)
                                    else:
                                        _new_order.append(_rank)
                                        _new_stage.append(0)
                                        _rank += 1
                                auto_order = tvm.runtime.convert(_new_order)
                                auto_stage = tvm.runtime.convert(_new_stage)
                except Exception:
                    pass  # validation failed; proceed with original auto_group

                ann["tl_pipeline_group"] = _validated_group

                # Try to run SMT to optimize order
                smt_optimized_order = False
                try:
                    seq, extra_map = _unwrap_to_seqstmt(stmt.body)
                    if seq is not None:
                        buffer_var_map: Dict[tvm.tir.Var, tvm.tir.Buffer] = {}
                        for _, b in func.buffer_map.items():
                            buffer_var_map[b.data] = b
                        buffer_var_map.update(func_alloc_map)
                        buffer_var_map.update(extra_map)

                        func_alloc_vars = set(func_alloc_map.keys())
                        infos_amode, _  = _build_stmt_infos(seq, buffer_var_map, func_alloc_vars)

                        # A-mode only needs relative ordering for consumer group reordering;
                        # use Phase A only (precise=False) to avoid Z3.Optimize crashes on
                        # large kernels where Phase B's register constraints are unsatisfiable.
                        sol_amode = _schedule_with_smt(
                            infos_amode, ii=max(1, effective_num_stages),
                            use_precise_latency=False, debug=smt_debug,
                            max_ii_search=0,
                            disallow_spills=disallow_spills,
                            use_spill_concurrency=use_spill_concurrency,
                            include_incoming_live=include_incoming_live,
                            num_threads=num_threads,
                        )
                        
                        if sol_amode is not None and auto_order is not None and auto_stage is not None:
                            # Parse auto_group, auto_order, auto_stage
                            groups_list = [[int(x) for x in g] for g in auto_group]
                            order_list = [int(x) for x in auto_order]
                            stage_list = [int(x) for x in auto_stage]
                            
                            if smt_debug:
                                print(f"[SMT Debug] A-mode analysis:")
                                print(f"[SMT Debug]   auto_group: {groups_list}")
                                print(f"[SMT Debug]   auto_order: {order_list}")
                                print(f"[SMT Debug]   auto_stage: {stage_list}")
                            
                            # ========== Order optimization only (no group splitting) ==========
                            # Keep the original grouping from C++ auto pass, only optimize order
                            # based on SMT times. This is safer and avoids breaking the backend.
                            #
                            # Note: Group splitting is disabled for now because it can break
                            # the PipelineInfo constructor which expects consistent annotations.
                            
                            # Compute min SMT time for each consumer group
                            consumer_group_data = []
                            for gi, (grp, ord_val, stg_val) in enumerate(zip(groups_list, order_list, stage_list)):
                                if ord_val == -1:
                                    continue
                                min_t = min(int(sol_amode.get(idx, 0)) for idx in grp)
                                consumer_group_data.append((gi, min_t, ord_val, stg_val))
                            
                            if smt_debug:
                                print(f"[SMT Debug]   consumer_groups: {[(gi, f'time={t}', f'ord={o}', f'stg={s}') for gi, t, o, s in consumer_group_data]}")
                            
                            # Sort by min_time only to allow cross-stage reordering
                            consumer_group_data_sorted = sorted(consumer_group_data, key=lambda x: x[1])
                            
                            # Build new order assignment
                            new_order_list = order_list.copy()
                            for new_rank, (gi, _, old_ord, _) in enumerate(consumer_group_data_sorted):
                                new_order_list[gi] = new_rank
                            
                            if smt_debug:
                                print(f"[SMT Debug]   sorted by time: {[(gi, t) for gi, t, _, _ in consumer_group_data_sorted]}")
                                print(f"[SMT Debug]   original order: {order_list}")
                                print(f"[SMT Debug]   new order:      {new_order_list}")
                            
                            # Check if order changed
                            if new_order_list != order_list:
                                smt_optimized_order = True
                                if smt_debug:
                                    print(f"[SMT Debug]   -> Order CHANGED!")
                                ann["tl_pipeline_order"] = tvm.runtime.convert(new_order_list)
                                ann["tl.debug_smt_pipeline_optimized_order"] = tvm.tir.IntImm("int32", 1)
                            else:
                                if smt_debug:
                                    print(f"[SMT Debug]   -> Order unchanged (SMT agrees with auto pass)")
                except Exception as e:
                    if smt_debug:
                        print(f"[SMT Debug] A-mode SMT optimization failed: {e}")
                
                # If SMT didn't optimize order, use original auto_order
                if not smt_optimized_order:
                    if auto_order is not None:
                        ann["tl_pipeline_order"] = auto_order
                        ann["tl.debug_smt_pipeline_used_auto_order"] = tvm.tir.IntImm("int32", 1)
                
                if auto_stage is not None:
                    ann["tl_pipeline_stage"] = auto_stage
                    ann["tl.debug_smt_pipeline_used_auto_stage"] = tvm.tir.IntImm("int32", 1)
                if override_num_stages and override_num_stages > 0:
                    ann["num_stages"] = tvm.tir.IntImm("int32", int(override_num_stages))
                return tvm.tir.For(
                    stmt.loop_var,
                    stmt.min,
                    stmt.extent,
                    stmt.kind,
                    stmt.body,
                    stmt.thread_binding,
                    ann,
                )

            seq, extra_map = _unwrap_to_seqstmt(stmt.body)
            if seq is None:
                ann["tl.debug_smt_pipeline_reason_code"] = tvm.tir.IntImm("int32", 1)  # unwrap_failed
                return tvm.tir.For(
                    stmt.loop_var,
                    stmt.min,
                    stmt.extent,
                    stmt.kind,
                    stmt.body,
                    stmt.thread_binding,
                    ann,
                )

            # Build buffer var map: PrimFunc buffers + alloc_buffers discovered in body.
            buffer_var_map: Dict[tvm.tir.Var, tvm.tir.Buffer] = {}
            for _, b in func.buffer_map.items():
                buffer_var_map[b.data] = b
            buffer_var_map.update(func_alloc_map)
            buffer_var_map.update(extra_map)

            func_alloc_vars = set(func_alloc_map.keys())
            infos, _ = _build_stmt_infos(seq, buffer_var_map, func_alloc_vars)

            # --- Compute-bound quality gate ---
            # For loops dominated by compute (many WGMMAs, SFU ops), SMT pipeline
            # annotations add code-gen overhead without enabling useful overlap
            # (GEMM1→softmax→GEMM2 dependency chain is sequential).
            # The WS rewriter's auto-detection handles these optimally.
            #
            # Heuristic: if ≥2 WGMMAs per iteration AND stages ≤ 2, the kernel is
            # attention-like (compute-bound) and annotations are counter-productive.
            # Evidence: FA FWD baseline 553 TFLOPS vs SMT-annotated 482 (-12.9%).
            _wgmma_count = sum(1 for x in infos if x.is_wgmma)
            _producer_count = sum(1 for x in infos if x.is_producer)
            _total_stmts = len(infos)
            _compute_ratio = _wgmma_count / max(1, _total_stmts - _producer_count)
            if _wgmma_count >= 2 and _compute_ratio >= 0.15:
                if smt_debug:
                    print(f"[SMT Debug] Compute-bound bypass: {_wgmma_count} WGMMAs in "
                          f"{_total_stmts} stmts (ratio={_compute_ratio:.2f}), "
                          f"stages={effective_num_stages} — skipping annotations")
                ann["tl.debug_smt_pipeline_reason_code"] = tvm.tir.IntImm("int32", 5)  # compute_bound_bypass
                ann["tl.debug_smt_pipeline_wgmma_count"] = tvm.tir.IntImm("int32", _wgmma_count)
                ann["tl.debug_smt_pipeline_compute_ratio"] = tvm.tir.FloatImm("float32", _compute_ratio)
                return tvm.tir.For(
                    stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
                    stmt.body, stmt.thread_binding, ann,
                )

            if not any(x.is_producer for x in infos):
                # Do not perturb loops without producers.
                ann["tl.debug_smt_pipeline_reason_code"] = tvm.tir.IntImm("int32", 2)  # no_producer
                ann["tl.debug_smt_pipeline_num_stmts"] = tvm.tir.IntImm("int32", int(len(infos)))
                ann["tl.debug_smt_pipeline_stmt_is_copy"] = tvm.runtime.convert(
                    [1 if x.is_producer and not x.is_true_tma else 0 for x in infos]
                )
                ann["tl.debug_smt_pipeline_stmt_is_intr"] = tvm.runtime.convert(
                    [1 if x.is_true_tma else 0 for x in infos]
                )
                ann["tl.debug_smt_pipeline_stmt_is_producer"] = tvm.runtime.convert(
                    [1 if x.is_producer else 0 for x in infos]
                )
                ann["tl.debug_smt_pipeline_stmt_reads_n"] = tvm.runtime.convert(
                    [len(x.reads) for x in infos]
                )
                ann["tl.debug_smt_pipeline_stmt_writes_n"] = tvm.runtime.convert(
                    [len(x.writes) for x in infos]
                )
                return tvm.tir.For(
                    stmt.loop_var,
                    stmt.min,
                    stmt.extent,
                    stmt.kind,
                    stmt.body,
                    stmt.thread_binding,
                    ann,
                )

            # Forced grouping: bypass auto grouping and search order/stage directly.
            if forced_groups is not None:
                flat = [idx for grp in forced_groups for idx in grp]
                if flat and max(flat) < len(infos):
                    search_res = _search_group_order_stage(
                        forced_groups,
                        infos,
                        num_stages=effective_num_stages,
                        use_precise_latency=use_precise_latency,
                        search_order=search_order,
                        debug=smt_debug,
                        stage_policy="semantic",
                        stage_offset=stage_offset,
                    )
                    if search_res is None and use_precise_latency:
                        # Fallback: try uniform latency if precise is UNSAT
                        if smt_debug:
                            print("[SMT Debug] Forced-group search UNSAT with precise latency; retrying with uniform latency")
                        search_res = _search_group_order_stage(
                            forced_groups,
                            infos,
                            num_stages=effective_num_stages,
                            use_precise_latency=False,
                            search_order=search_order,
                            debug=smt_debug,
                            stage_policy="semantic",
                            stage_offset=stage_offset,
                        )
                    if search_res is not None:
                        group_order, group_stage, _group_times = search_res
                        # If forced order provided, override SMT order (length must match groups)
                        if forced_order is not None:
                            if len(forced_order) == len(forced_groups):
                                group_order = forced_order
                            else:
                                if smt_debug:
                                    print(
                                        f"[SMT Debug] Forced order length mismatch: "
                                        f"{len(forced_order)} vs groups {len(forced_groups)}"
                                    )
                        # If forced stage provided, override SMT stage (length must match groups)
                        if forced_stage is not None:
                            if len(forced_stage) == len(forced_groups):
                                group_stage = forced_stage
                            else:
                                if smt_debug:
                                    print(
                                        f"[SMT Debug] Forced stage length mismatch: "
                                        f"{len(forced_stage)} vs groups {len(forced_groups)}"
                                    )
                        ann["tl.auto_tl_pipeline_smt_applied"] = tvm.tir.IntImm("int32", 1)
                        ann["tl.debug_smt_pipeline_forced_group"] = tvm.tir.IntImm("int32", 1)
                        ann["tl_pipeline_group"] = tvm.runtime.convert(forced_groups)
                        ann["tl_pipeline_order"] = tvm.runtime.convert(group_order)
                        ann["tl_pipeline_stage"] = tvm.runtime.convert(group_stage)
                        # Keep debug copies so dumps after InjectSoftwarePipeline can still show them
                        ann["tl.debug_smt_pipeline_forced_order"] = tvm.runtime.convert(group_order)
                        ann["tl.debug_smt_pipeline_forced_stage"] = tvm.runtime.convert(group_stage)
                        if override_num_stages and override_num_stages > 0:
                            ann["num_stages"] = tvm.tir.IntImm("int32", int(override_num_stages))
                        return tvm.tir.For(
                            stmt.loop_var,
                            stmt.min,
                            stmt.extent,
                            stmt.kind,
                            stmt.body,
                            stmt.thread_binding,
                            ann,
                        )
                else:
                    if smt_debug:
                        print(f"[SMT Debug] Forced groups invalid for this loop: {forced_groups}")

            # When using precise latency, allow searching for larger II if initial II is too tight
            # This handles cases where high-latency ops (like SFU) make the original II infeasible
            max_ii_search = effective_num_stages * 4 if use_precise_latency else 0
            if smt_debug:
                print(f"[SMT Debug] Main mode: calling _schedule_with_smt ii={max(1,effective_num_stages)} precise={use_precise_latency}", file=sys.stderr, flush=True)
            sol = _schedule_with_smt(
                infos,
                ii=max(1, effective_num_stages),
                use_precise_latency=use_precise_latency,
                debug=smt_debug,
                max_ii_search=max_ii_search,
                disallow_spills=disallow_spills,
                use_spill_concurrency=use_spill_concurrency,
                include_incoming_live=include_incoming_live,
                num_threads=num_threads,
            )
            if smt_debug:
                print(f"[SMT Debug] Main mode: _schedule_with_smt returned sol={sol is not None}", file=sys.stderr, flush=True)
            if sol is None:
                ann["tl.debug_smt_pipeline_reason_code"] = tvm.tir.IntImm("int32", 3)  # unsat
                ann["tl.debug_smt_pipeline_num_stmts"] = tvm.tir.IntImm("int32", int(len(infos)))
                return tvm.tir.For(
                    stmt.loop_var,
                    stmt.min,
                    stmt.extent,
                    stmt.kind,
                    stmt.body,
                    stmt.thread_binding,
                    ann,
                )

            # ========== Compute order/stage for each statement ==========
            order: List[int] = []
            stage: List[int] = []
            consumer_o = 0
            # Monotone stage mapping:
            # InjectSoftwarePipeline requires (for its internal deps) that stage does not decrease
            # along the scheduled order/dependences. A modulo mapping (t % S) can wrap and violate
            # this invariant. Instead, bucketize absolute time into [0..S-1] without wrap:
            # stage = clamp(t - t_min, 0, S-1).
            consumer_ts = [int(sol.get(info.idx, 0)) for info in infos if not info.is_producer]
            t0 = min(consumer_ts) if consumer_ts else 0
            # Safety gate:
            # Our current SMT graph is still a MVP and does not model all real dependencies
            # (especially for WS + barrier/wgmma paths). By default, we DO NOT apply the
            # multi-stage assignment to avoid breaking correctness. We still keep the raw
            # SMT stages for debugging.
            raw_stage: List[int] = []
            for info in infos:
                if info.is_producer:
                    order.append(-1)
                    stage.append(-1)
                    raw_stage.append(-1)
                else:
                    order.append(consumer_o)
                    consumer_o += 1
                    t = int(sol.get(info.idx, 0))
                    smax = max(1, effective_num_stages) - 1
                    st = t - int(t0)
                    if st < 0:
                        st = 0
                    if st > smax:
                        st = smax
                    raw_stage.append(int(st))
                    # If we enable multistage, only move statements that are conservatively
                    # considered "safe": global/shared-only, non-sync, non-wgmma.
                    if not enable_multistage:
                        stage.append(int(smax))
                    else:
                        stage.append(int(st) if _is_stage_movable(info) else int(smax))

            if smt_debug:
                print(f"[SMT Debug] Entering pattern search, order={order[:5]}..., stage={stage[:5]}...", file=sys.stderr, flush=True)
            # ========== WS pattern search over multiple group templates ==========
            modes = ["stage_runs"]
            if pattern_search_enable:
                modes = ["stage_runs", "producer_boundary", "wgmma_split", "all_consumers"]

            best_groups: Optional[List[List[int]]] = None
            best_order: Optional[List[int]] = None
            best_stage: Optional[List[int]] = None
            best_score: Optional[Tuple[int, int, int]] = None
            best_mode: Optional[str] = None

            for mode in modes:
                if smt_debug:
                    print(f"[SMT Debug] Pattern mode={mode}", file=sys.stderr, flush=True)
                cand_groups = _build_group_templates(infos, stage, mode=mode)
                if not cand_groups:
                    continue
                if smt_debug:
                    print(f"[SMT Debug] Pattern mode={mode} groups={cand_groups}", file=sys.stderr, flush=True)
                res = _search_group_order_stage(
                    cand_groups,
                    infos,
                    num_stages=effective_num_stages,
                    use_precise_latency=use_precise_latency,
                    search_order=search_order,
                    debug=smt_debug,
                    stage_policy="semantic",
                    stage_offset=stage_offset,
                )
                if smt_debug:
                    print(f"[SMT Debug] Pattern mode={mode} res={res is not None}", file=sys.stderr, flush=True)
                if res is None:
                    continue
                cand_order, cand_stage, cand_times = res
                makespan = max(cand_times.values()) if cand_times else 0
                total = int(sum(cand_times.values())) if cand_times else 0
                score = (makespan, total, len(cand_groups))
                if best_score is None or score < best_score:
                    best_score = score
                    best_groups = cand_groups
                    best_order = cand_order
                    best_stage = cand_stage
                    best_mode = mode
                if smt_debug:
                    print(f"[SMT Debug] Pattern mode={mode} score={score} groups={cand_groups}")

            if best_groups is None or best_order is None or best_stage is None:
                # fallback: conservative single template
                best_groups = _build_group_templates(infos, stage, mode="stage_runs")
                best_order = []
                next_rank = 0
                for g in best_groups:
                    if all(infos[idx].is_producer for idx in g):
                        best_order.append(-1)
                    else:
                        best_order.append(next_rank)
                        next_rank += 1
                best_stage = [(-1 if all(infos[idx].is_producer for idx in g)
                               else max(stage[idx] for idx in g if idx < len(stage)))
                              for g in best_groups]
                best_mode = "stage_runs_fallback"

            # Validate that groups are contiguous and don't mix producers/consumers
            # in the same group (WarpSpecialized requires group_size==1 for producers).
            _n_seq = len(infos)
            _total_stmts = sum(len(g) for g in best_groups)
            if _total_stmts != _n_seq:
                # Fallback: use safe conservative grouping
                print(f"[SMT] WARNING: group total {_total_stmts} != seq len {_n_seq}, "
                      f"groups={best_groups}, falling back to per-stmt", file=sys.stderr, flush=True)
                best_groups = [[info.idx] for info in infos]
                best_order = []
                best_stage = []
                next_rank = 0
                for info in infos:
                    if info.is_producer:
                        best_order.append(-1)
                        best_stage.append(-1)
                    else:
                        best_order.append(next_rank)
                        best_stage.append(int(stage[info.idx]) if info.idx < len(stage) else 0)
                        next_rank += 1
            # Ensure no group mixes producers with multiple statements
            for gi, g in enumerate(best_groups):
                if len(g) > 1:
                    has_producer = any(infos[idx].is_producer for idx in g if idx < _n_seq)
                    if has_producer:
                        print(f"[SMT] WARNING: group {gi}={g} has producer+others, "
                              f"falling back to per-stmt", file=sys.stderr, flush=True)
                        best_groups = [[info.idx] for info in infos]
                        best_order = []
                        best_stage = []
                        next_rank2 = 0
                        for info in infos:
                            if info.is_producer:
                                best_order.append(-1)
                                best_stage.append(-1)
                            else:
                                best_order.append(next_rank2)
                                best_stage.append(int(stage[info.idx]) if info.idx < len(stage) else 0)
                                next_rank2 += 1
                        break

            group = tvm.runtime.convert(best_groups)
            order = best_order
            stage = best_stage

            ann["tl.auto_tl_pipeline_smt_applied"] = tvm.tir.IntImm("int32", 1)
            # T debug:
            # -1: producer marker
            # -2: non-movable (not scheduled by SMT)
            # >=0: scheduled movable stmt time
            t_dbg: List[int] = []
            for i, info in enumerate(infos):
                if info.is_producer:
                    t_dbg.append(-1)
                elif not _is_stage_movable(info):
                    t_dbg.append(-2)
                else:
                    t_dbg.append(int(sol.get(i, -3)))
            ann["tl.debug_smt_pipeline_T"] = tvm.runtime.convert(t_dbg)
            ann["tl.debug_smt_pipeline_ii"] = tvm.tir.IntImm("int32", int(max(1, effective_num_stages)))
            ann["tl.debug_smt_pipeline_stage_raw"] = tvm.runtime.convert(raw_stage)
            if not enable_multistage:
                ann["tl.debug_smt_pipeline_stage_sanitized"] = tvm.tir.IntImm("int32", 1)
            ann["tl.debug_smt_pipeline_multistage_enabled"] = tvm.tir.IntImm("int32", 1 if enable_multistage else 0)
            ann["tl.debug_smt_pipeline_stmt_is_wgmma"] = tvm.runtime.convert([1 if x.is_wgmma else 0 for x in infos])
            ann["tl.debug_smt_pipeline_stmt_is_sync"] = tvm.runtime.convert(
                [1 if (x.is_sync_top or x.is_sync_nested) else 0 for x in infos]
            )
            ann["tl.debug_smt_pipeline_stmt_touches_local"] = tvm.runtime.convert([1 if x.touches_local else 0 for x in infos])
            ann["tl.debug_smt_pipeline_stmt_touches_external_local"] = tvm.runtime.convert(
                [1 if x.touches_external_local else 0 for x in infos]
            )
            ann["tl.debug_smt_pipeline_stmt_movable"] = tvm.runtime.convert([1 if _is_stage_movable(x) else 0 for x in infos])
            ann["tl_pipeline_group"] = group
            ann["tl_pipeline_order"] = tvm.runtime.convert(order)
            ann["tl_pipeline_stage"] = tvm.runtime.convert(stage)

            # Always emit a compact summary line so that downstream tooling
            # (e.g. joint_ablation.py's parse_smt_pattern) can capture it
            # without requiring TL_SMT_DEBUG=1.
            print(
                f"[SMT] chosen_pattern_mode: {best_mode}  "
                f"chosen_pattern_score: {best_score}  "
                f"groups: {best_groups}  "
                f"order: {order}  "
                f"stage: {stage}",
                file=sys.stderr, flush=True,
            )

            if smt_debug:
                print(f"[SMT Debug] Generated annotations:")
                print(f"[SMT Debug]   chosen_pattern_mode: {best_mode}")
                print(f"[SMT Debug]   chosen_pattern_score: {best_score}")
                print(f"[SMT Debug]   num_groups: {len(best_groups)}")
                print(f"[SMT Debug]   groups: {best_groups}")
                print(f"[SMT Debug]   order: {order}")
                print(f"[SMT Debug]   stage: {stage}")
                print(f"[SMT Debug]   raw_stage: {raw_stage}")

            if override_num_stages and override_num_stages > 0:
                ann["num_stages"] = tvm.tir.IntImm("int32", int(override_num_stages))

            return tvm.tir.For(
                stmt.loop_var,
                stmt.min,
                stmt.extent,
                stmt.kind,
                stmt.body,
                stmt.thread_binding,
                ann,
            )

        new_body = tvm.tir.stmt_functor.ir_transform(func.body, None, _post, None)
        if smt_debug:
            print(f"[SMT Debug] Pass completed, visited {for_count[0]} For loops", file=sys.stderr, flush=True)
        return func.with_body(new_body).with_attr("tl.auto_tl_pipeline_smt_applied", 1)

    return _pass
