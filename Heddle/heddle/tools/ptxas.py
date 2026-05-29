from __future__ import annotations

"""
Utilities for parsing ptxas verbose output and building lightweight occupancy proxies.

This is a productionized subset of `examples/analyze/pressure_utils.py`, so frontend/runtime
code (e.g. AutoMixed decision in `tilelang/frontend/patterns.py`) can close the loop:

  compile -> ptxas --verbose -> (regs, spill, smem) -> fallback
"""

import contextlib
import io
import re
import os
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class PtxasStats:
    regs: int | None
    stack_frame_bytes: int | None
    spill_stores_bytes: int | None
    spill_loads_bytes: int | None
    smem_bytes: int | None


@dataclass(frozen=True)
class OccupancyProxy:
    """
    A lightweight occupancy proxy derived from regs-per-thread and threads-per-block.

    NOTE: This is NOT exact hardware occupancy (it ignores block limits, SMEM, and allocator granularity),
    but is extremely useful for diagnosing the reg/occupancy cliff discussed in docs/design.md.
    """

    ok: bool
    regs_per_thread: int | None = None
    threads_per_block: int | None = None
    warp_size: int | None = None
    regs_per_sm: int | None = None
    max_threads_per_sm: int | None = None
    max_warps_per_sm: int | None = None
    active_warps_reg_limited: int | None = None
    occ_reg_pct: float | None = None
    reason: str = ""


_RE_PTXAS_REGS = re.compile(r"Used\s+(?P<regs>\d+)\s+registers", re.IGNORECASE)
# ptxas often prints:
#   "ptxas info    : Used 168 registers, 196608 bytes smem, 0 bytes cmem[0]"
# So we match the generic "NN bytes smem" and pick the first occurrence.
_RE_PTXAS_SMEM = re.compile(r"(?P<smem>\d+)\s+bytes\s+smem", re.IGNORECASE)
_RE_PTXAS_STACK_SPILL = re.compile(
    r"(?P<stack>\d+)\s+bytes stack frame,\s+(?P<stores>\d+)\s+bytes spill stores,\s+(?P<loads>\d+)\s+bytes spill loads",
    re.IGNORECASE,
)
_RE_PTXAS_SPILL_FALLBACK = re.compile(
    r"(?P<stores>\d+)\s+bytes spill stores,\s+(?P<loads>\d+)\s+bytes spill loads|spill stores,\s+(?P<loads2>\d+)\s+bytes spill loads",
    re.IGNORECASE,
)

_RE_CUDA_FUNC_SETATTR_DYN_SMEM = re.compile(
    r"cudaFuncSetAttribute\s*\([^,]+,\s*cudaFuncAttributeMaxDynamicSharedMemorySize\s*,\s*(?P<n>\d+)\s*\)",
    re.IGNORECASE,
)


def parse_ptxas_stats(log: str) -> PtxasStats:
    regs = None
    stack_frame_bytes = None
    spill_stores_bytes = None
    spill_loads_bytes = None
    smem_bytes = None

    m = _RE_PTXAS_REGS.search(log)
    if m:
        regs = int(m.group("regs"))

    m = _RE_PTXAS_SMEM.search(log)
    if m:
        smem_bytes = int(m.group("smem"))

    m = _RE_PTXAS_STACK_SPILL.search(log)
    if m:
        stack_frame_bytes = int(m.group("stack"))
        spill_stores_bytes = int(m.group("stores"))
        spill_loads_bytes = int(m.group("loads"))
        return PtxasStats(
            regs=regs,
            stack_frame_bytes=stack_frame_bytes,
            spill_stores_bytes=spill_stores_bytes,
            spill_loads_bytes=spill_loads_bytes,
            smem_bytes=smem_bytes,
        )

    m = _RE_PTXAS_SPILL_FALLBACK.search(log)
    if m:
        if m.group("stores") is not None and m.group("loads") is not None:
            spill_stores_bytes = int(m.group("stores"))
            spill_loads_bytes = int(m.group("loads"))
        elif m.group("loads2") is not None:
            spill_loads_bytes = int(m.group("loads2"))

    return PtxasStats(
        regs=regs,
        stack_frame_bytes=stack_frame_bytes,
        spill_stores_bytes=spill_stores_bytes,
        spill_loads_bytes=spill_loads_bytes,
        smem_bytes=smem_bytes,
    )


def capture_compile_log(build_fn):
    """
    Capture stdout+stderr during TileLang compilation.
    Returns (obj, text_log).
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        obj = build_fn()
    return obj, buf.getvalue()


def occupancy_proxy_from_regs(*, regs_per_thread: int | None, threads_per_block: int | None) -> OccupancyProxy:
    """
    Estimate occupancy limited by register file only.
    Returns occ_reg_pct in [0,1], and active_warps_reg_limited.
    """
    if regs_per_thread is None or threads_per_block is None:
        return OccupancyProxy(ok=False, regs_per_thread=regs_per_thread, threads_per_block=threads_per_block, reason="missing regs/threads")
    if regs_per_thread <= 0 or threads_per_block <= 0:
        return OccupancyProxy(ok=False, regs_per_thread=regs_per_thread, threads_per_block=threads_per_block, reason="invalid regs/threads")

    try:
        import torch  # type: ignore
    except Exception:
        return OccupancyProxy(ok=False, regs_per_thread=regs_per_thread, threads_per_block=threads_per_block, reason="torch not available")
    if not torch.cuda.is_available():
        return OccupancyProxy(ok=False, regs_per_thread=regs_per_thread, threads_per_block=threads_per_block, reason="cuda not available")

    p = torch.cuda.get_device_properties(0)
    warp_size = int(getattr(p, "warp_size", 32))
    regs_per_sm = int(getattr(p, "regs_per_multiprocessor", 0))
    max_threads_per_sm = int(getattr(p, "max_threads_per_multi_processor", 0))
    if regs_per_sm <= 0 or max_threads_per_sm <= 0:
        return OccupancyProxy(
            ok=False,
            regs_per_thread=regs_per_thread,
            threads_per_block=threads_per_block,
            warp_size=warp_size,
            regs_per_sm=regs_per_sm or None,
            max_threads_per_sm=max_threads_per_sm or None,
            reason="missing device limits",
        )

    # Reg-limited max threads (very rough; real alloc is granular and per-warp/block).
    max_threads_reg = regs_per_sm // regs_per_thread
    # Align down to warp boundary.
    max_threads_reg = (max_threads_reg // warp_size) * warp_size
    active_threads = min(max_threads_reg, max_threads_per_sm)
    max_warps_per_sm = max_threads_per_sm // warp_size
    active_warps = active_threads // warp_size
    occ = 0.0 if max_warps_per_sm <= 0 else float(active_warps) / float(max_warps_per_sm)
    return OccupancyProxy(
        ok=True,
        regs_per_thread=regs_per_thread,
        threads_per_block=threads_per_block,
        warp_size=warp_size,
        regs_per_sm=regs_per_sm,
        max_threads_per_sm=max_threads_per_sm,
        max_warps_per_sm=max_warps_per_sm,
        active_warps_reg_limited=active_warps,
        occ_reg_pct=occ,
    )


def sm_arch_flag() -> str:
    """
    Best-effort "sm_XX[a]" arch flag for nvcc.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return "sm_80"
    if not torch.cuda.is_available():
        return "sm_80"
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    if major >= 9:
        arch += "a"
    return arch


def get_max_dynamic_smem_per_block_bytes(device_id: int = 0) -> int | None:
    """
    Best-effort per-block dynamic shared memory limit (opt-in).

    This is the limit enforced by `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, N)`.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(device_id)
    # Newer PyTorch exposes opt-in per-block shared memory limit.
    v = getattr(p, "shared_memory_per_block_optin", None)
    if isinstance(v, int) and v > 0:
        return int(v)
    # Fallback: non-opt-in per-block shared memory.
    v2 = getattr(p, "shared_memory_per_block", None)
    if isinstance(v2, int) and v2 > 0:
        return int(v2)
    return None


def dyn_shared_memory_bytes_from_jit_kernel(kernel) -> int | None:
    """
    Extract requested dynamic shared memory bytes from a TileLang JITKernel.

    We prefer reading `dyn_shared_memory_buf` from device PrimFunc attrs (set by TVM lowering).
    Returns the max across device functions, because multi-function modules can exist.
    """
    # Lazy imports to avoid importing TVM at module import time.
    try:
        from tvm import tir  # type: ignore
    except Exception:
        tir = None  # type: ignore

    # Most common cases:
    # - TileLang JITKernel: kernel.adapter.device_mod is an IRModule
    # - Adapter object returned directly: kernel.device_mod is an IRModule
    try:
        dev_mod = getattr(kernel, "device_mod", None)
        if dev_mod is None:
            adapter = getattr(kernel, "adapter", None)
            dev_mod = getattr(adapter, "device_mod", None)
        if dev_mod is not None and hasattr(dev_mod, "functions"):
            best = None
            for _, fn in dev_mod.functions.items():
                attrs = getattr(fn, "attrs", None)
                if attrs is None:
                    continue
                if "dyn_shared_memory_buf" in attrs:
                    try:
                        v = int(attrs["dyn_shared_memory_buf"])
                        best = v if best is None else max(best, v)
                    except Exception:
                        continue
            if best is not None:
                return int(best)
    except Exception:
        pass

    # Fallback: sometimes users pass a PrimFunc directly.
    try:
        prim = getattr(kernel, "prim_func", None)
        if prim is not None and getattr(prim, "attrs", None) is not None:
            attrs = prim.attrs
            if "dyn_shared_memory_buf" in attrs:
                return int(attrs["dyn_shared_memory_buf"])
    except Exception:
        pass

    # Last resort: parse from generated *host wrapper* source. This is robust across backends,
    # because the wrapper literally embeds the cudaFuncSetAttribute(...) call with the requested bytes.
    try:
        src = None
        # Prefer explicit host source if available (this is where init()/cudaFuncSetAttribute lives).
        if hasattr(kernel, "get_host_source"):
            try:
                src = kernel.get_host_source()
            except Exception:
                src = None
        # Some adapters expose host source on the adapter object.
        if not src:
            adapter = getattr(kernel, "adapter", None)
            if adapter is not None and hasattr(adapter, "get_host_source"):
                try:
                    src = adapter.get_host_source()
                except Exception:
                    src = None
        # Fallback: some backends may include host wrapper in "kernel source" when kernel_only=False.
        if not src and hasattr(kernel, "get_kernel_source"):
            try:
                src = kernel.get_kernel_source(kernel_only=False)
            except TypeError:
                # Some implementations use positional arg instead of kw.
                src = kernel.get_kernel_source(False)
        if isinstance(src, str) and src:
            m = _RE_CUDA_FUNC_SETATTR_DYN_SMEM.search(src)
            if m:
                return int(m.group("n"))
    except Exception:
        pass

    return None


def nvcc_ptxas_stats_from_cuda_source(code: str) -> tuple[PtxasStats, str]:
    """
    Compile generated CUDA source with nvcc and capture ptxas verbose output.

    This is a robust fallback when kernel compilation is served from cache and no ptxas text is emitted.
    """
    # Local import to avoid paying heavy deps on import.
    from tilelang.contrib.nvcc import get_nvcc_compiler
    from tilelang.env import CUDA_HOME, CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

    arch = sm_arch_flag()
    with tempfile.TemporaryDirectory() as td:
        cu_path = os.path.join(td, "kernel.cu")
        cubin_path = os.path.join(td, "kernel.cubin")
        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(code)

        include_flags: list[str] = []
        if TILELANG_TEMPLATE_PATH:
            include_flags.append(f"-I{TILELANG_TEMPLATE_PATH}")
        if CUTLASS_INCLUDE_DIR:
            include_flags.append(f"-I{CUTLASS_INCLUDE_DIR}")
        if CUDA_HOME:
            include_flags.append(f"-I{os.path.join(CUDA_HOME, 'include')}")

        cmd = [
            get_nvcc_compiler(),
            "--cubin",
            "-O3",
            "-lineinfo",
            f"-arch={arch}",
            "--ptxas-options=--verbose",
            "-w",
            *include_flags,
            "-o",
            cubin_path,
            cu_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = proc.stdout or ""
        if proc.returncode != 0:
            raise RuntimeError(f"nvcc failed (rc={proc.returncode}). Output:\n{out}")
        return parse_ptxas_stats(out), out

