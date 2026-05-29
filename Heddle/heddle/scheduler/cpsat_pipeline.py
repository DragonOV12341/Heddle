"""
Unified CP-SAT → Heddle → Benchmark pipeline.

This module connects the CP-SAT pre-filter to the actual TileLang
compilation pipeline, closing the loop from constraint solving to
GPU-measured performance.

Pipeline:
  1. CP-SAT: enumerate partition × tile × stages candidates,
     solve joint constraint program, rank by objective → Top-K
  2. Compile: for each Top-K candidate, build TileLang kernel
     with Heddle consumer schedule (Z3) + PCWS
  3. Benchmark: run each compiled kernel on GPU, pick the fastest

Usage:
    from heddle.scheduler.cpsat_pipeline import AutoSearch

    search = AutoSearch(
        op_name="fa_bwd",
        shapes={"B": 4, "H": 32, "D": 128, "T": 4096},
    )
    best = search.run(top_k=3)
    print(best.config, best.tflops)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Candidate:
    """A single search candidate."""
    name: str
    partition: str
    config: Dict[str, Any]  # {block_M, block_N, num_stages, threads, strategy, ...}
    # Filled by CP-SAT pre-filter
    cpsat_objective: float = 0.0
    cpsat_ii: int = 0
    cpsat_makespan: int = 0
    cpsat_reg_peak: int = 0
    cpsat_occupancy: int = 0
    # Filled by compilation
    compile_ok: bool = False
    compile_time_ms: float = 0.0
    ptxas_regs: int = 0
    ptxas_spill: int = 0
    # Filled by benchmark
    latency_ms: float = 0.0
    tflops: float = 0.0


@dataclass
class SearchResult:
    """Result of the full search pipeline."""
    best: Optional[Candidate] = None
    all_candidates: List[Candidate] = field(default_factory=list)
    cpsat_time_ms: float = 0.0
    compile_time_ms: float = 0.0
    bench_time_ms: float = 0.0

    def summary(self) -> str:
        lines = []
        lines.append(f"Search complete: {len(self.all_candidates)} candidates")
        lines.append(f"  CP-SAT:   {self.cpsat_time_ms:.0f}ms")
        lines.append(f"  Compile:  {self.compile_time_ms:.0f}ms")
        lines.append(f"  Bench:    {self.bench_time_ms:.0f}ms")
        compiled = [c for c in self.all_candidates if c.compile_ok]
        lines.append(f"  Compiled: {len(compiled)}/{len(self.all_candidates)}")
        if self.best:
            lines.append(f"  Best:     {self.best.name} → {self.best.tflops:.1f} TFLOPS")
            lines.append(f"            {self.best.config}")
        return "\n".join(lines)


def run_pipeline(
    candidates: List[Candidate],
    builder_fn: Callable[[Dict[str, Any]], Any],
    bench_fn: Callable[[Any], float],
    *,
    flops: float = 0.0,
    top_k: int = 5,
) -> SearchResult:
    """Run the full CP-SAT → compile → benchmark pipeline.

    Args:
        candidates: pre-ranked by CP-SAT objective (lowest first)
        builder_fn: (config) -> compiled TileLang program
        bench_fn: (program) -> latency_ms
        flops: total FLOPs for TFLOPS calculation
        top_k: how many candidates to actually compile and benchmark
    """
    result = SearchResult()
    result.all_candidates = candidates

    # Take Top-K by CP-SAT objective
    ranked = sorted(candidates, key=lambda c: c.cpsat_objective)
    top = ranked[:top_k]

    # Compile
    t0 = time.monotonic()
    for c in top:
        try:
            tc0 = time.monotonic()
            program = builder_fn(c.config)
            c.compile_time_ms = (time.monotonic() - tc0) * 1000
            c.compile_ok = True
            c._program = program  # stash for benchmark
        except Exception as e:
            c.compile_ok = False
            c.compile_time_ms = (time.monotonic() - tc0) * 1000
    result.compile_time_ms = (time.monotonic() - t0) * 1000

    # Benchmark compiled candidates
    t0 = time.monotonic()
    for c in top:
        if not c.compile_ok:
            continue
        try:
            c.latency_ms = bench_fn(c._program)
            if flops > 0 and c.latency_ms > 0:
                c.tflops = flops / (c.latency_ms / 1000) / 1e12
        except Exception:
            c.latency_ms = -1
    result.bench_time_ms = (time.monotonic() - t0) * 1000

    # Pick best
    benched = [c for c in top if c.compile_ok and c.latency_ms > 0]
    if benched:
        result.best = min(benched, key=lambda c: c.latency_ms)

    return result
