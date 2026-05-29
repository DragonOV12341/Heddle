"""
Bridge from CP-SAT unified solver to HeddleConsumerSchedule / PCWS.

Converts CP-SAT schedule output (op_name -> time) into a consumer_ordering
list that PCWS can consume, and provides a convenience function to run
an end-to-end benchmark.

Usage:
    from heddle.scheduler.cpsat_bridge import schedule_to_consumer_ordering
    ordering = schedule_to_consumer_ordering(cpsat_schedule, op_name_to_stmt_idx)
"""

from __future__ import annotations
from typing import Dict, List, Optional


def schedule_to_consumer_ordering(
    schedule: Dict[str, int],
    op_name_to_stmt_idx: Dict[str, int],
) -> List[int]:
    """Convert a CP-SAT schedule to a PCWS consumer ordering.

    Args:
        schedule: {op_name: start_time} from CP-SAT result
        op_name_to_stmt_idx: {op_name: statement_index} mapping ops to
            their position in the original TIR SeqStmt

    Returns:
        consumer_ordering: List[int] of statement indices sorted by
            scheduled start time (ascending), breaking ties by
            original position
    """
    items = []
    for op_name, start_time in schedule.items():
        if op_name in op_name_to_stmt_idx:
            items.append((start_time, op_name_to_stmt_idx[op_name], op_name))

    # Sort by (time, original_position) for determinism
    items.sort(key=lambda x: (x[0], x[1]))
    return [stmt_idx for _, stmt_idx, _ in items]


def benchmark_partition(
    partition_name: str,
    kernel_builders: Dict[str, callable],
    kernel_configs: Dict[str, dict],
    *,
    warmup: int = 5,
    repeat: int = 20,
) -> Dict[str, float]:
    """Benchmark all kernels in a partition and return TFLOPS per kernel.

    Args:
        partition_name: name of the chosen partition
        kernel_builders: {kernel_name: callable(**config) -> tilelang program}
        kernel_configs: {kernel_name: {config dict}}
        warmup: warmup iterations
        repeat: benchmark iterations

    Returns:
        {kernel_name: median_time_ms}
    """
    import torch

    results = {}
    for kname, builder in kernel_builders.items():
        config = kernel_configs.get(kname, {})
        try:
            program = builder(**config)
            kernel = program.get_kernel()

            # Warmup
            for _ in range(warmup):
                kernel()

            # Benchmark
            torch.cuda.synchronize()
            import time
            times = []
            for _ in range(repeat):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                kernel()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            times.sort()
            median_ms = times[len(times) // 2]
            results[kname] = median_ms
        except Exception as e:
            results[kname] = f"ERROR: {e}"

    return results
