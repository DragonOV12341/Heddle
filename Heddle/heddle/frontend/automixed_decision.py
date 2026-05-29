from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from heddle.tools.ptxas import (
    capture_compile_log,
    dyn_shared_memory_bytes_from_jit_kernel,
    get_max_dynamic_smem_per_block_bytes,
    nvcc_ptxas_stats_from_cuda_source,
    occupancy_proxy_from_regs,
    parse_ptxas_stats,
)

RejectCode = Literal[
    "OK",
    "PTXAS_FEEDBACK_DISABLED",
    "PTXAS_STATS_MISSING",
    "SMEM_EST_EXCEEDS_LIMIT",
    "DYN_SMEM_EXCEEDS_LIMIT",
    "REGS_EXCEEDS_HARD",
    "STACK_FRAME_NONZERO",
    "SPILL_STORES_NONZERO",
    "SPILL_LOADS_NONZERO",
    "COMPILE_OR_PTXAS_ERROR",
]

RiskLevel = Literal["unknown", "safe", "risk", "hard"]


def _risk_from_regs(regs: int | None, *, regs_safe: int, regs_hard: int) -> RiskLevel:
    if regs is None:
        return "unknown"
    r = int(regs)
    if r <= int(regs_safe):
        return "safe"
    if r <= int(regs_hard):
        return "risk"
    return "hard"


def _infer_fallback_rule(src: "Candidate", dst: "Candidate") -> dict[str, Any]:
    """
    Best-effort inference of fallback rule for evidence tracing.
    (We keep fallback_fn signature simple, so we infer rule from field diffs.)
    """
    changes: list[str] = []
    if src.pv_gemm_type != dst.pv_gemm_type:
        changes.append("pv_gemm_type")
    if int(dst.num_stages) < int(src.num_stages):
        changes.append("num_stages")
    if int(dst.block_M) < int(src.block_M) or int(dst.block_N) < int(src.block_N):
        changes.append("tile")
    if int(dst.threads) > int(src.threads):
        changes.append("threads")
    if changes == ["pv_gemm_type"] and src.pv_gemm_type == "rs" and dst.pv_gemm_type == "ss":
        rule = "RS_TO_SS"
    elif changes == ["num_stages"]:
        rule = "REDUCE_STAGES"
    elif changes == ["tile"]:
        rule = "SHRINK_TILE"
    elif changes == ["threads"]:
        rule = "INCREASE_THREADS"
    else:
        rule = "FALLBACK_OTHER"
    return {"rule": rule, "changes": changes}


def _maybe_append_jsonl(path: str | None, obj: dict) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except Exception:
        pass
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        # Best-effort logging only.
        return


@dataclass(frozen=True)
class AutoMixedPolicy:
    """
    Policy knobs aligned with docs/design.md and docs/Heddle_config.md.
    """

    enable_ptxas_feedback: bool = False
    regs_safe: int = 180
    regs_hard: int = 240
    require_no_spill: bool = True
    try_topk: int = 8
    log: bool = False
    trace_path: str | None = None


@dataclass(frozen=True)
class Candidate:
    block_M: int
    block_N: int
    num_stages: int
    threads: int
    pv_gemm_type: Literal["rs", "ss"] | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "block_M": int(self.block_M),
            "block_N": int(self.block_N),
            "num_stages": int(self.num_stages),
            "threads": int(self.threads),
        }
        if self.pv_gemm_type is not None:
            d["pv_gemm_type"] = str(self.pv_gemm_type)
        return d


@dataclass(frozen=True)
class StaticEstimate:
    smem_est_bytes: int | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"smem_est_bytes": self.smem_est_bytes, "note": self.note}


def _ptxas_eval(
    *,
    build_fn: Callable[[], Any],
    policy: AutoMixedPolicy,
    threads_per_block: int | None = None,
    cache: dict[tuple, dict] | None = None,
    cache_key: tuple | None = None,
) -> dict | None:
    """
    Compile a candidate kernel and extract ptxas stats (regs/stack/spill/smem) + dyn_smem + occ proxy.
    Returns None when feedback is disabled or not available.
    """
    if not policy.enable_ptxas_feedback:
        return None
    if cache is not None and cache_key is not None and cache_key in cache:
        return cache[cache_key]

    try:
        kern, log_text = capture_compile_log(build_fn)
        stats = parse_ptxas_stats(log_text or "")
        # If compilation was served from cache, we may see no ptxas text.
        if stats.regs is None and hasattr(kern, "get_kernel_source"):
            try:
                src = kern.get_kernel_source()
                if isinstance(src, str) and src.strip():
                    stats2, _ = nvcc_ptxas_stats_from_cuda_source(src)
                    if stats2.regs is not None or stats2.smem_bytes is not None:
                        stats = stats2
            except Exception as e:
                if policy.log:
                    print(f"[AutoMixed] nvcc-ptxas fallback failed: {e}")
        dyn_smem = dyn_shared_memory_bytes_from_jit_kernel(kern)
        dyn_lim = get_max_dynamic_smem_per_block_bytes()
        occ0 = occupancy_proxy_from_regs(regs_per_thread=stats.regs, threads_per_block=threads_per_block)
        out = {
            "regs": stats.regs,
            "smem_bytes": stats.smem_bytes,
            "stack_frame_bytes": stats.stack_frame_bytes,
            "spill_stores_bytes": stats.spill_stores_bytes,
            "spill_loads_bytes": stats.spill_loads_bytes,
            "dyn_smem_bytes": dyn_smem,
            "dyn_smem_limit_bytes": dyn_lim,
            "occ_reg_pct": occ0.occ_reg_pct if occ0.ok else None,
            "occ_reason": occ0.reason,
            "error": "",
        }
    except Exception as e:
        if policy.log:
            print(f"[AutoMixed] ptxas-eval failed: {type(e).__name__}: {e}")
        return {
            "regs": None,
            "smem_bytes": None,
            "stack_frame_bytes": None,
            "spill_stores_bytes": None,
            "spill_loads_bytes": None,
            "dyn_smem_bytes": None,
            "dyn_smem_limit_bytes": get_max_dynamic_smem_per_block_bytes(),
            "occ_reg_pct": None,
            "occ_reason": f"error:{type(e).__name__}",
            "error": f"{type(e).__name__}: {e}",
        }

    if cache is not None and cache_key is not None:
        cache[cache_key] = out
    return out


def _ptxas_accept(ptx: dict | None, *, policy: AutoMixedPolicy) -> tuple[bool, RejectCode, dict[str, Any]]:
    if not policy.enable_ptxas_feedback:
        return True, "PTXAS_FEEDBACK_DISABLED", {}
    if ptx is None:
        return True, "PTXAS_STATS_MISSING", {}
    if isinstance(ptx.get("error", ""), str) and ptx.get("error"):
        return False, "COMPILE_OR_PTXAS_ERROR", {"error": ptx.get("error", "")}

    dyn = ptx.get("dyn_smem_bytes", None)
    dyn_lim = ptx.get("dyn_smem_limit_bytes", None)
    if isinstance(dyn, int) and isinstance(dyn_lim, int) and dyn_lim > 0 and dyn > dyn_lim:
        return False, "DYN_SMEM_EXCEEDS_LIMIT", {"dyn_smem_bytes": int(dyn), "dyn_smem_limit_bytes": int(dyn_lim)}

    regs = ptx.get("regs", None)
    if isinstance(regs, int) and regs > int(policy.regs_hard):
        return False, "REGS_EXCEEDS_HARD", {"regs": int(regs), "regs_hard": int(policy.regs_hard)}

    if policy.require_no_spill:
        stack = ptx.get("stack_frame_bytes", None)
        if isinstance(stack, int) and stack > 0:
            return False, "STACK_FRAME_NONZERO", {"stack_frame_bytes": int(stack)}
        spill_st = ptx.get("spill_stores_bytes", None)
        if isinstance(spill_st, int) and spill_st > 0:
            return False, "SPILL_STORES_NONZERO", {"spill_stores_bytes": int(spill_st)}
        spill_ld = ptx.get("spill_loads_bytes", None)
        if isinstance(spill_ld, int) and spill_ld > 0:
            return False, "SPILL_LOADS_NONZERO", {"spill_loads_bytes": int(spill_ld)}

    return True, "OK", {}


def _cpsat_rank_candidates(
    candidates: list[Candidate],
    op: str,
    shape: dict[str, Any],
) -> list[Candidate]:
    """Use CP-SAT constraint solver to rank candidates by predicted performance.

    Falls back to original order if CP-SAT is not available or fails.
    """
    try:
        from heddle.scheduler.cp_sat import (
            UnifiedScheduler, PartitionSpec, KernelSpec, OpSpec, OutputSpec,
            ResourceType, StorageKind, H100,
        )
    except ImportError:
        return candidates

    D = int(shape.get("D", shape.get("dim", 128)))
    TC, SFU, TMA_L = 27, 25, 20
    FU = {ResourceType.TensorCore: 1, ResourceType.SFU: 16,
          ResourceType.ALU: 64, ResourceType.TMA: 1}

    scored: list[tuple[float, Candidate]] = []
    for c in candidates:
        bM, bN, ns, t = int(c.block_M), int(c.block_N), int(c.num_stages), int(c.threads)
        acc = bM * bN * 4 // t
        acc_d = bM * D * 4 // t
        S = ns
        smem = bM * D * 2 + bN * D * 2 * S * 2  # Q + K×S + V×S

        ops = [
            OpSpec("tma_k", ResourceType.TMA, TMA_L, deps=[("qk", S)], fixed_warp=0),
            OpSpec("tma_v", ResourceType.TMA, TMA_L, deps=[("tma_k", 0), ("pv", S)], fixed_warp=0),
            OpSpec("qk", ResourceType.TensorCore, TC,
                   outputs=[OutputSpec("qk_acc", StorageKind.RMEM, acc)],
                   deps=[("tma_k", 0)]),
            OpSpec("sfmx", ResourceType.SFU, SFU,
                   outputs=[OutputSpec("P", StorageKind.RMEM, acc)],
                   deps=[("qk", 0)]),
            OpSpec("pv", ResourceType.TensorCore, TC,
                   outputs=[OutputSpec("O_acc", StorageKind.RMEM, acc_d)],
                   deps=[("sfmx", 0), ("tma_v", 0), ("pv", 1)]),
        ]
        spec = PartitionSpec(f"c_{bM}_{bN}_{ns}_{t}",
                             [KernelSpec("k", ops, smem_bytes=smem, threads=t + 128)])
        try:
            solver = UnifiedScheduler(
                [spec], fu_caps=FU, reg_limit=960,
                reg_safe_threshold=600, spill_penalty_per_byte=2,
                occupancy_weight=10, num_warps=2,
                horizon=200, timeout_s=2.0,
            )
            r = solver.solve_modulo(min_ii=1, max_ii=60)
            score = r.total_makespan if r else 9999
        except Exception:
            score = 9999
        scored.append((score, c))

    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored]


def decide_with_fallback(
    *,
    op: str,
    shape: dict[str, Any],
    candidates: list[Candidate],
    static_estimator: Callable[[Candidate], StaticEstimate] | None,
    build_fn_of_candidate: Callable[[Candidate], Callable[[], Any]],
    policy: AutoMixedPolicy,
    fallback_fn: Callable[[Candidate], list[Candidate]],
) -> dict[str, Any]:
    """
    Core AutoMixed decision loop:
    - CP-SAT pre-ranking (if available)
    - static estimate + pruning
    - Top-K ptxas sampling
    - fallback chain when a candidate is rejected
    - JSONL evidence chain (optional)

    Returns: a dict config (plus optionally pv_gemm_type) suitable to feed into kernel builders.
    """
    # Pre-rank candidates with CP-SAT (fast, ~80ms for all candidates)
    enable_cpsat = os.environ.get("TILELANG_AUTOMIXED_CPSAT_RANK", "1").strip() == "1"
    if enable_cpsat:
        candidates = _cpsat_rank_candidates(candidates, op, shape)

    # Ranking: prefer larger tiles and higher stages first (aggressive-first), then threads=256 (often safer for regs).
    def _static_rank(c: Candidate) -> tuple:
        tile = int(c.block_M) * int(c.block_N)
        th = int(c.threads)
        stg = int(c.num_stages)
        pv = 0 if c.pv_gemm_type is None else (0 if c.pv_gemm_type == "rs" else 1)
        # Larger tile, larger stages, prefer 256 threads, then rs before ss (ss increases SMEM traffic).
        return (-tile, -stg, -th, pv)

    trace = {
        "ts": time.time(),
        "op": str(op),
        "shape": shape,
        "policy": {
            "enable_ptxas_feedback": bool(policy.enable_ptxas_feedback),
            "regs_safe": int(policy.regs_safe),
            "regs_hard": int(policy.regs_hard),
            "require_no_spill": bool(policy.require_no_spill),
            "try_topk": int(policy.try_topk),
        },
        "candidates": [],
        "selected": None,
    }

    _cache: dict[tuple, dict] = {}

    # Best-effort parent edges for evidence chaining (candidate_key -> edge info).
    parent_edge: dict[tuple, dict[str, Any]] = {}

    # Phase 1: static prune (SMEM only; regs is unreliable without ptxas).
    pruned: list[Candidate] = []
    for c in sorted(candidates, key=_static_rank):
        est = static_estimator(c) if static_estimator is not None else StaticEstimate()
        ok = True
        reject: dict[str, Any] = {"code": "OK", "details": {}}
        # Best-effort SMEM hard limit: use per-block opt-in limit when present; otherwise don't reject.
        smem_lim = get_max_dynamic_smem_per_block_bytes()
        if est.smem_est_bytes is not None and isinstance(smem_lim, int) and smem_lim > 0:
            if int(est.smem_est_bytes) > int(smem_lim):
                ok = False
                reject = {
                    "code": "SMEM_EST_EXCEEDS_LIMIT",
                    "details": {"smem_est_bytes": int(est.smem_est_bytes), "smem_limit_bytes": int(smem_lim)},
                }
        trace["candidates"].append(
            {
                "candidate": c.as_dict(),
                "static": est.as_dict(),
                "accept": bool(ok),
                "reject": reject,
                "phase": "static",
                "parent": parent_edge.get((c.block_M, c.block_N, c.num_stages, c.threads, c.pv_gemm_type)),
            }
        )
        if ok:
            pruned.append(c)

    # Phase 2: Top-K ptxas sampling with fallback chain.
    tried = 0
    visited: set[tuple] = set()
    queue: list[Candidate] = pruned[:]

    while queue:
        c = queue.pop(0)
        key = (c.block_M, c.block_N, c.num_stages, c.threads, c.pv_gemm_type)
        if key in visited:
            continue
        visited.add(key)

        # If feedback is disabled, pick the first statically-ok candidate.
        if not policy.enable_ptxas_feedback:
            trace["selected"] = c.as_dict()
            _maybe_append_jsonl(policy.trace_path, trace)
            return c.as_dict()

        ptx = None
        accept = True
        reject = {"code": "OK", "details": {}}
        if tried < int(policy.try_topk):
            build_fn = build_fn_of_candidate(c)
            ptx = _ptxas_eval(
                build_fn=build_fn,
                policy=policy,
                threads_per_block=int(c.threads),
                cache=_cache,
                cache_key=key,
            )
            accept, code, details = _ptxas_accept(ptx, policy=policy)
            reject = {"code": code, "details": details}
            tried += 1

        regs = None if ptx is None else ptx.get("regs", None)
        regs_i = int(regs) if isinstance(regs, int) else None
        risk_level: RiskLevel = _risk_from_regs(regs_i, regs_safe=policy.regs_safe, regs_hard=policy.regs_hard)

        trace["candidates"].append(
            {
                "candidate": c.as_dict(),
                "ptxas": ptx,
                "accept": bool(accept),
                "reject": reject,
                "risk": {"level": risk_level, "regs_safe": int(policy.regs_safe), "regs_hard": int(policy.regs_hard)},
                "phase": "ptxas",
                "parent": parent_edge.get(key),
            }
        )
        if policy.log:
            print(f"[AutoMixed] cand={c.as_dict()} -> {reject.get('code')} ptxas={ptx}")

        if accept:
            trace["selected"] = c.as_dict()
            _maybe_append_jsonl(policy.trace_path, trace)
            return c.as_dict()

        for fc in fallback_fn(c):
            fkey = (fc.block_M, fc.block_N, fc.num_stages, fc.threads, fc.pv_gemm_type)
            if fkey not in visited:
                if fkey not in parent_edge:
                    parent_edge[fkey] = {"from": c.as_dict(), "via": _infer_fallback_rule(c, fc)}
                queue.append(fc)

    # Final fallback: return the most conservative candidate and dump trace for debugging.
    last = sorted(candidates, key=_static_rank)[-1] if candidates else Candidate(64, 64, 1, 128, None)
    trace["selected"] = last.as_dict()
    trace["warning"] = "no_candidate_accepted; falling back to most conservative"
    _maybe_append_jsonl(policy.trace_path, trace)
    return last.as_dict()


def estimate_flashattn_mha_fwd_bshd_smem_bytes(
    *,
    block_M: int,
    block_N: int,
    dim: int,
    qkv_dtype: str,
    pv_gemm_type: str,
) -> int:
    """
    Static SMEM estimate for `examples/analyze/pressure_flashattn_mha_fwd_bshd_stage_sweep.py::flashattn_mha_fwd_bshd`.
    Important: shared allocations in that kernel are NOT multiplied by num_stages.
    """
    qkv_dtype = (qkv_dtype or "float16").lower()
    pv_gemm_type = (pv_gemm_type or "rs").lower()

    # Q_qkv_shared/K_qkv_shared/V_qkv_shared are in qkv dtype (fp16/fp8)
    if qkv_dtype in ("float16", "fp16"):
        qkv_bytes = 2
    elif qkv_dtype in (
        "float8_e4m3fn",
        "fp8_e4m3",
        "fp8",
        "e4m3",
        "float8_e5m2",
        "fp8_e5m2",
        "e5m2",
    ):
        qkv_bytes = 1
    else:
        qkv_bytes = 2

    out_bytes = 2  # out_dtype is fp16
    smem = 0

    # qkv shared
    smem += int(block_M) * int(dim) * int(qkv_bytes)  # Q_qkv_shared
    smem += int(block_N) * int(dim) * int(qkv_bytes)  # K_qkv_shared
    smem += int(block_N) * int(dim) * int(qkv_bytes)  # V_qkv_shared

    # fp16 shared
    smem += int(block_M) * int(dim) * int(out_bytes)  # Q_shared
    smem += int(block_N) * int(dim) * int(out_bytes)  # K_shared
    smem += int(block_N) * int(dim) * int(out_bytes)  # V_shared
    smem += int(block_M) * int(dim) * int(out_bytes)  # O_shared

    if pv_gemm_type == "ss":
        smem += int(block_M) * int(block_N) * int(out_bytes)  # P_shared

    return int(smem)

