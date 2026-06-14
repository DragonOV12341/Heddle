"""Unit tests for warp-specialized solve() in cp_sat.py.

These tests exercise the CP-SAT solver directly without tilelang.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from heddle.scheduler.cp_sat import (
    UnifiedScheduler, PartitionSpec, KernelSpec, OpSpec,
    OutputSpec, ResourceType, StorageKind, _unpack_dep,
)


def _make_scheduler(ops, *, num_warps=1, reg_limit=256, timeout_s=5):
    kernel = KernelSpec(name="k0", ops=ops, smem_bytes=1024, threads=128)
    partition = PartitionSpec(name="p0", kernels=[kernel])
    return UnifiedScheduler(
        [partition],
        fu_caps={ResourceType.TMA: 1, ResourceType.TensorCore: 1,
                 ResourceType.ALU: 2, ResourceType.SFU: 1},
        reg_limit=reg_limit,
        num_warps=num_warps,
        timeout_s=timeout_s,
    )


class TestUnpackDep:

    def test_2tuple(self):
        name, dist, blocking = _unpack_dep(("a", 1))
        assert (name, dist, blocking) == ("a", 1, False)

    def test_3tuple(self):
        name, dist, blocking = _unpack_dep(("a", 1, True))
        assert (name, dist, blocking) == ("a", 1, True)

    def test_3tuple_false(self):
        name, dist, blocking = _unpack_dep(("a", 0, False))
        assert (name, dist, blocking) == ("a", 0, False)


class TestSolveBasicNoWarps:

    def test_chain_schedule(self):
        """A -> B -> C chain with 1 warp: basic scheduling, all warp 0."""
        ops = [
            OpSpec("A", ResourceType.TMA, latency=2,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 128)]),
            OpSpec("B", ResourceType.ALU, latency=3,
                   deps=[("A", 0)],
                   outputs=[OutputSpec("oB", StorageKind.RMEM, 64)]),
            OpSpec("C", ResourceType.TensorCore, latency=4,
                   deps=[("B", 0)]),
        ]
        solver = _make_scheduler(ops, num_warps=1)
        result = solver.solve()

        assert result is not None
        assert result.status in ("OPTIMAL", "FEASIBLE")
        sched = result.kernel_schedules["k0"]
        assert sched["A"] < sched["B"] < sched["C"]

        warps = result.kernel_warp_assigns.get("k0", {})
        for op_name in ("A", "B", "C"):
            assert warps.get(op_name, 0) == 0


class TestSolveWithWarps:

    def test_fixed_warp_respected(self):
        """Ops with fixed_warp are placed on their designated warps."""
        ops = [
            OpSpec("A", ResourceType.TMA, latency=2, fixed_warp=0,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 128)]),
            OpSpec("B", ResourceType.ALU, latency=3, fixed_warp=1,
                   deps=[("A", 0)],
                   outputs=[OutputSpec("oB", StorageKind.RMEM, 64)]),
            OpSpec("C", ResourceType.TensorCore, latency=4, fixed_warp=0,
                   deps=[("B", 0)]),
        ]
        solver = _make_scheduler(ops, num_warps=2)
        result = solver.solve()

        assert result is not None
        warps = result.kernel_warp_assigns["k0"]
        assert warps["A"] == 0
        assert warps["B"] == 1
        assert warps["C"] == 0

    def test_solver_assigns_warps(self):
        """With multiple warps and no fixed assignments, solver assigns freely."""
        ops = [
            OpSpec("A", ResourceType.TMA, latency=2,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 128)]),
            OpSpec("B", ResourceType.ALU, latency=3,
                   outputs=[OutputSpec("oB", StorageKind.RMEM, 64)]),
        ]
        solver = _make_scheduler(ops, num_warps=2)
        result = solver.solve()

        assert result is not None
        warps = result.kernel_warp_assigns["k0"]
        assert warps["A"] in (0, 1)
        assert warps["B"] in (0, 1)


class TestSolveBlockingSync:

    def test_blocking_sync_same_warp(self):
        """blocking_sync dependency forces producer and consumer onto same warp."""
        ops = [
            OpSpec("A", ResourceType.TMA, latency=4,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 128)]),
            OpSpec("B", ResourceType.ALU, latency=3,
                   deps=[("A", 0, True)]),  # blocking_sync
        ]
        solver = _make_scheduler(ops, num_warps=2)
        result = solver.solve()

        assert result is not None
        warps = result.kernel_warp_assigns["k0"]
        assert warps["A"] == warps["B"], \
            f"blocking_sync should force same warp, got A={warps['A']}, B={warps['B']}"

    def test_blocking_sync_with_bystander(self):
        """A third op cannot overlap the barrier window on the barrier warp."""
        ops = [
            OpSpec("prod", ResourceType.TMA, latency=5,
                   outputs=[OutputSpec("oP", StorageKind.RMEM, 128)]),
            OpSpec("cons", ResourceType.ALU, latency=2,
                   deps=[("prod", 0, True)]),  # blocking_sync
            OpSpec("other", ResourceType.ALU, latency=3,
                   outputs=[OutputSpec("oO", StorageKind.RMEM, 64)]),
        ]
        solver = _make_scheduler(ops, num_warps=2)
        result = solver.solve()

        assert result is not None
        sched = result.kernel_schedules["k0"]
        warps = result.kernel_warp_assigns["k0"]

        assert warps["prod"] == warps["cons"]

        if warps["other"] == warps["prod"]:
            barrier_start = sched["cons"] - 5
            barrier_end = sched["cons"]
            other_start = sched["other"]
            other_end = sched["other"] + 3
            overlap = other_start < barrier_end and other_end > barrier_start
            assert not overlap, \
                f"other op overlaps barrier window [{barrier_start},{barrier_end})"


class TestSolveSpillCost:

    def test_cross_warp_adds_delay(self):
        """Cross-warp dep with spill_cost adds extra latency."""
        spill = 10
        base_lat = 2
        ops = [
            OpSpec("A", ResourceType.TMA, latency=base_lat, spill_cost=spill,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 128)]),
            OpSpec("B", ResourceType.ALU, latency=3,
                   deps=[("A", 0)]),
        ]
        solver = _make_scheduler(ops, num_warps=2)
        result = solver.solve()

        assert result is not None
        sched = result.kernel_schedules["k0"]
        warps = result.kernel_warp_assigns["k0"]

        gap = sched["B"] - sched["A"]
        if warps["A"] != warps["B"]:
            assert gap >= base_lat + spill, \
                f"cross-warp gap should be >= {base_lat + spill}, got {gap}"
        else:
            assert gap >= base_lat, \
                f"same-warp gap should be >= {base_lat}, got {gap}"


class TestSolvePerWarpFUCapacity:

    def test_per_warp_capacity(self):
        """With 2 warps and FU cap=1, at most 1 op per warp at a time."""
        ops = [
            OpSpec("A", ResourceType.ALU, latency=3, fixed_warp=0,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 32)]),
            OpSpec("B", ResourceType.ALU, latency=3, fixed_warp=0,
                   outputs=[OutputSpec("oB", StorageKind.RMEM, 32)]),
            OpSpec("C", ResourceType.ALU, latency=3, fixed_warp=1,
                   outputs=[OutputSpec("oC", StorageKind.RMEM, 32)]),
        ]
        solver = _make_scheduler(ops, num_warps=2,
                                 reg_limit=1024)
        # ALU cap is 2 globally, but per-warp should be 2 as well.
        # With A and B both on warp 0 and ALU cap=2, they can overlap.
        # Override to cap=1 to test serialization.
        solver_caps = {ResourceType.TMA: 1, ResourceType.TensorCore: 1,
                       ResourceType.ALU: 1, ResourceType.SFU: 1}
        kernel = KernelSpec(name="k0", ops=ops, smem_bytes=1024, threads=128)
        partition = PartitionSpec(name="p0", kernels=[kernel])
        solver2 = UnifiedScheduler(
            [partition], fu_caps=solver_caps, reg_limit=1024,
            num_warps=2, timeout_s=5,
        )
        result = solver2.solve()

        assert result is not None
        sched = result.kernel_schedules["k0"]
        # A and B are both on warp 0 with ALU cap 1: they must not overlap
        a_start, a_end = sched["A"], sched["A"] + 3
        b_start, b_end = sched["B"], sched["B"] + 3
        overlap_ab = a_start < b_end and b_start < a_end
        assert not overlap_ab, \
            f"A [{a_start},{a_end}) and B [{b_start},{b_end}) overlap on same warp with cap=1"

        # C is on warp 1, so it can freely overlap with A or B
        c_start = sched["C"]
        assert c_start >= 0


class TestBackwardCompat2TupleDeps:

    def test_old_format_works(self):
        """2-tuple deps format still works (backward compat)."""
        ops = [
            OpSpec("X", ResourceType.TMA, latency=2,
                   outputs=[OutputSpec("oX", StorageKind.RMEM, 64)]),
            OpSpec("Y", ResourceType.ALU, latency=3,
                   deps=[("X", 0)]),  # 2-tuple, no blocking_sync
        ]
        solver = _make_scheduler(ops, num_warps=1)
        result = solver.solve()

        assert result is not None
        sched = result.kernel_schedules["k0"]
        assert sched["Y"] >= sched["X"] + 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
