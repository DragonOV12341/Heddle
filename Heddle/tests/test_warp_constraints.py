"""Deep verification of warp-related constraints in both CP-SAT and SMT solvers.

Tests exercise solve() and _solve_phase_b() with specific constraint scenarios
to verify that warp assignment, blocking_sync, spill cost, per-warp FU capacity,
and per-warp register liveness all enforce correct behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from heddle.scheduler.cp_sat import (
    UnifiedScheduler, PartitionSpec, KernelSpec, OpSpec,
    OutputSpec, ResourceType, StorageKind, _unpack_dep,
)


# ── helpers ────────────────────────────────────────────────────────────

def _make_cpsat(ops, *, num_warps=1, reg_limit=256, timeout_s=5,
                fu_caps=None):
    kernel = KernelSpec(name="k0", ops=ops, smem_bytes=1024, threads=128)
    partition = PartitionSpec(name="p0", kernels=[kernel])
    caps = fu_caps or {
        ResourceType.TMA: 1, ResourceType.TensorCore: 1,
        ResourceType.ALU: 2, ResourceType.SFU: 1,
    }
    return UnifiedScheduler(
        [partition], fu_caps=caps, reg_limit=reg_limit,
        num_warps=num_warps, timeout_s=timeout_s,
    )


def _has_z3():
    try:
        import z3
        return True
    except ImportError:
        return False


def _make_smt_nodes(specs, *, num_warps=1, reg_limit=240, timeout_ms=5000,
                    fu_caps=None):
    """Build an SMT HeddleScheduler from a lightweight spec list.

    Each spec: (name, resource_type_str, latency, deps, outputs, kwargs)
    deps: [(parent_name, distance, blocking_sync?)]
    outputs: [(name, storage_str, footprint, spill_cost)]
    """
    from heddle.scheduler.smt import (
        HeddleScheduler, OpNode, OutputValue, ResourceType as SRT,
        StorageKind as SSK,
    )
    _rt = {"TMA": SRT.TMA, "ALU": SRT.ALU, "TC": SRT.TensorCore, "SFU": SRT.SFU}
    _sk = {"RMEM": SSK.RMEM, "SMEM": SSK.SMEM}

    node_map = {}
    nodes = []
    for spec in specs:
        name, rt, lat = spec[0], spec[1], spec[2]
        deps = spec[3] if len(spec) > 3 else []
        outs = spec[4] if len(spec) > 4 else []
        kw = spec[5] if len(spec) > 5 else {}
        ov_list = [
            OutputValue(n, _sk.get(s, SSK.RMEM), fp, sc)
            for n, s, fp, sc in outs
        ]
        nd = OpNode(name, _rt[rt], lat, outputs=ov_list, **kw)
        node_map[name] = nd
        nodes.append(nd)

    for spec in specs:
        name = spec[0]
        deps = spec[3] if len(spec) > 3 else []
        nd = node_map[name]
        for dep_tuple in deps:
            pname = dep_tuple[0]
            dist = dep_tuple[1] if len(dep_tuple) > 1 else 0
            blocking = dep_tuple[2] if len(dep_tuple) > 2 else False
            nd.add_dependency(node_map[pname], distance=dist,
                              blocking_sync=blocking)

    caps = fu_caps or {SRT.TMA: 1, SRT.TensorCore: 1, SRT.ALU: 2, SRT.SFU: 1}
    return HeddleScheduler(
        nodes, fu_caps=caps, reg_limit=reg_limit, num_warps=num_warps,
        timeout_ms=timeout_ms,
    )


# ======================================================================
# CP-SAT solve() — warp constraint verification
# ======================================================================

class TestCPSatWarpAssignment:
    """Verify warp variable creation and fixed_warp enforcement."""

    def test_all_ops_get_warp_assigns(self):
        ops = [
            OpSpec("A", ResourceType.TMA, latency=2,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 64)]),
            OpSpec("B", ResourceType.ALU, latency=3, deps=[("A", 0)]),
            OpSpec("C", ResourceType.TensorCore, latency=4, deps=[("B", 0)]),
        ]
        result = _make_cpsat(ops, num_warps=2).solve()
        assert result is not None
        warps = result.kernel_warp_assigns["k0"]
        assert set(warps.keys()) == {"A", "B", "C"}
        for v in warps.values():
            assert v in (0, 1)

    def test_three_warp_groups(self):
        ops = [
            OpSpec("A", ResourceType.TMA, latency=2, fixed_warp=0),
            OpSpec("B", ResourceType.ALU, latency=3, fixed_warp=1),
            OpSpec("C", ResourceType.TensorCore, latency=4, fixed_warp=2),
        ]
        result = _make_cpsat(ops, num_warps=3).solve()
        assert result is not None
        warps = result.kernel_warp_assigns["k0"]
        assert warps["A"] == 0
        assert warps["B"] == 1
        assert warps["C"] == 2

    def test_single_warp_all_zero(self):
        ops = [
            OpSpec("A", ResourceType.TMA, latency=2),
            OpSpec("B", ResourceType.ALU, latency=3),
        ]
        result = _make_cpsat(ops, num_warps=1).solve()
        assert result is not None
        warps = result.kernel_warp_assigns["k0"]
        assert all(v == 0 for v in warps.values())


class TestCPSatBlockingSync:
    """Verify blocking_sync constraints in solve()."""

    def test_same_warp_enforced(self):
        ops = [
            OpSpec("P", ResourceType.TMA, latency=4,
                   outputs=[OutputSpec("oP", StorageKind.RMEM, 128)]),
            OpSpec("C", ResourceType.ALU, latency=2,
                   deps=[("P", 0, True)]),
        ]
        result = _make_cpsat(ops, num_warps=2).solve()
        assert result is not None
        warps = result.kernel_warp_assigns["k0"]
        assert warps["P"] == warps["C"]

    def test_barrier_exclusion_window(self):
        """Third op on the same warp must not overlap the barrier window."""
        ops = [
            OpSpec("P", ResourceType.TMA, latency=5,
                   outputs=[OutputSpec("oP", StorageKind.RMEM, 128)]),
            OpSpec("C", ResourceType.ALU, latency=2,
                   deps=[("P", 0, True)]),
            OpSpec("X", ResourceType.ALU, latency=3,
                   outputs=[OutputSpec("oX", StorageKind.RMEM, 64)]),
        ]
        result = _make_cpsat(ops, num_warps=2).solve()
        assert result is not None
        sched = result.kernel_schedules["k0"]
        warps = result.kernel_warp_assigns["k0"]
        assert warps["P"] == warps["C"]

        if warps["X"] == warps["P"]:
            barrier_start = sched["C"] - 5
            barrier_end = sched["C"]
            x_start = sched["X"]
            x_end = sched["X"] + 3
            overlap = x_start < barrier_end and x_end > barrier_start
            assert not overlap, (
                f"X [{x_start},{x_end}) overlaps barrier [{barrier_start},{barrier_end})"
            )

    def test_multiple_blocking_edges(self):
        """Two blocking_sync edges: both enforce same-warp."""
        ops = [
            OpSpec("P1", ResourceType.TMA, latency=3,
                   outputs=[OutputSpec("o1", StorageKind.RMEM, 64)]),
            OpSpec("C1", ResourceType.ALU, latency=2,
                   deps=[("P1", 0, True)]),
            OpSpec("P2", ResourceType.TMA, latency=4,
                   outputs=[OutputSpec("o2", StorageKind.RMEM, 64)]),
            OpSpec("C2", ResourceType.TensorCore, latency=2,
                   deps=[("P2", 0, True)]),
        ]
        result = _make_cpsat(ops, num_warps=2).solve()
        assert result is not None
        warps = result.kernel_warp_assigns["k0"]
        assert warps["P1"] == warps["C1"]
        assert warps["P2"] == warps["C2"]


class TestCPSatSpillCost:
    """Verify cross-warp spill cost constraints."""

    def test_spill_adds_delay_cross_warp(self):
        spill = 8
        base_lat = 2
        ops = [
            OpSpec("A", ResourceType.TMA, latency=base_lat, spill_cost=spill,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 128)]),
            OpSpec("B", ResourceType.ALU, latency=3, deps=[("A", 0)]),
        ]
        result = _make_cpsat(ops, num_warps=2).solve()
        assert result is not None
        sched = result.kernel_schedules["k0"]
        warps = result.kernel_warp_assigns["k0"]
        gap = sched["B"] - sched["A"]
        if warps["A"] != warps["B"]:
            assert gap >= base_lat + spill
        else:
            assert gap >= base_lat

    def test_zero_spill_no_penalty(self):
        ops = [
            OpSpec("A", ResourceType.TMA, latency=3, spill_cost=0,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 128)]),
            OpSpec("B", ResourceType.ALU, latency=2, deps=[("A", 0)]),
        ]
        result = _make_cpsat(ops, num_warps=2).solve()
        assert result is not None
        sched = result.kernel_schedules["k0"]
        gap = sched["B"] - sched["A"]
        assert gap >= 3


class TestCPSatPerWarpFUCapacity:
    """Verify per-warp FU capacity constraints."""

    def test_same_warp_serialized(self):
        """Two ALU ops on same warp with cap=1 must not overlap."""
        ops = [
            OpSpec("A", ResourceType.ALU, latency=4, fixed_warp=0,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 32)]),
            OpSpec("B", ResourceType.ALU, latency=4, fixed_warp=0,
                   outputs=[OutputSpec("oB", StorageKind.RMEM, 32)]),
        ]
        result = _make_cpsat(
            ops, num_warps=2,
            fu_caps={ResourceType.TMA: 1, ResourceType.TensorCore: 1,
                     ResourceType.ALU: 1, ResourceType.SFU: 1},
        ).solve()
        assert result is not None
        sched = result.kernel_schedules["k0"]
        a_range = (sched["A"], sched["A"] + 4)
        b_range = (sched["B"], sched["B"] + 4)
        overlap = a_range[0] < b_range[1] and b_range[0] < a_range[1]
        assert not overlap

    def test_different_warps_can_overlap(self):
        """Two ALU ops on different warps with cap=1 can run in parallel."""
        ops = [
            OpSpec("A", ResourceType.ALU, latency=4, fixed_warp=0,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 32)]),
            OpSpec("B", ResourceType.ALU, latency=4, fixed_warp=1,
                   outputs=[OutputSpec("oB", StorageKind.RMEM, 32)]),
        ]
        result = _make_cpsat(
            ops, num_warps=2,
            fu_caps={ResourceType.TMA: 1, ResourceType.TensorCore: 1,
                     ResourceType.ALU: 1, ResourceType.SFU: 1},
        ).solve()
        assert result is not None
        sched = result.kernel_schedules["k0"]
        # With separate warps, makespan should be 4 (parallel), not 8
        makespan = result.total_makespan
        assert makespan <= 4


class TestCPSatPerWarpRegLiveness:
    """Verify per-warp register liveness constraints."""

    def test_reg_pressure_forces_different_warps(self):
        """Two big outputs on same warp would exceed reg_limit, solver splits them."""
        ops = [
            OpSpec("A", ResourceType.TMA, latency=2,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 200)]),
            OpSpec("B", ResourceType.ALU, latency=2,
                   outputs=[OutputSpec("oB", StorageKind.RMEM, 200)]),
            OpSpec("C", ResourceType.TensorCore, latency=2,
                   deps=[("A", 0), ("B", 0)]),
        ]
        # reg_limit=250: each output 200, both alive together = 400 > 250
        result = _make_cpsat(ops, num_warps=2, reg_limit=250).solve()
        assert result is not None
        warps = result.kernel_warp_assigns["k0"]
        # A and B should be on different warps to avoid exceeding limit
        assert warps["A"] != warps["B"], (
            f"Expected different warps due to reg pressure, got A={warps['A']}, B={warps['B']}"
        )

    def test_small_outputs_can_share_warp(self):
        """Small outputs don't force warp splitting."""
        ops = [
            OpSpec("A", ResourceType.TMA, latency=2,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 32)]),
            OpSpec("B", ResourceType.ALU, latency=2,
                   outputs=[OutputSpec("oB", StorageKind.RMEM, 32)]),
            OpSpec("C", ResourceType.TensorCore, latency=2,
                   deps=[("A", 0), ("B", 0)]),
        ]
        result = _make_cpsat(ops, num_warps=2, reg_limit=256).solve()
        assert result is not None
        # Both can fit on same warp (64 < 256), solver may or may not split


class TestCPSatResultIntegrity:
    """Verify the structure and completeness of solve() output."""

    def test_schedules_respect_deps(self):
        ops = [
            OpSpec("A", ResourceType.TMA, latency=3,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 64)]),
            OpSpec("B", ResourceType.ALU, latency=2, deps=[("A", 0)]),
            OpSpec("C", ResourceType.TensorCore, latency=4, deps=[("B", 0)]),
        ]
        result = _make_cpsat(ops, num_warps=2).solve()
        assert result is not None
        s = result.kernel_schedules["k0"]
        assert s["B"] >= s["A"] + 3
        assert s["C"] >= s["B"] + 2

    def test_reg_peak_per_warp(self):
        ops = [
            OpSpec("A", ResourceType.TMA, latency=2, fixed_warp=0,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 100)]),
            OpSpec("B", ResourceType.ALU, latency=2, fixed_warp=1,
                   outputs=[OutputSpec("oB", StorageKind.RMEM, 80)]),
            OpSpec("C", ResourceType.TensorCore, latency=2,
                   deps=[("A", 0), ("B", 0)]),
        ]
        result = _make_cpsat(ops, num_warps=2, reg_limit=256).solve()
        assert result is not None
        peak = result.kernel_reg_peaks.get("k0")
        assert peak is not None
        if isinstance(peak, dict):
            assert all(isinstance(v, int) for v in peak.values())
        else:
            assert isinstance(peak, int)
            assert peak <= 256

    def test_empty_ops(self):
        result = _make_cpsat([], num_warps=2).solve()
        assert result is not None
        assert result.total_makespan == 0


# ======================================================================
# SMT (Z3) _solve_phase_b — warp constraint verification
# ======================================================================

@pytest.mark.skipif(not _has_z3(), reason="z3-solver not installed")
class TestSMTWarpAssignment:

    def test_all_ops_get_warp(self):
        smt = _make_smt_nodes([
            ("A", "TMA", 2, [], [("oA", "RMEM", 64, 0)]),
            ("B", "ALU", 3, [("A", 0)], []),
        ], num_warps=2)
        result = smt._solve_phase_b(ii=6, L=12)
        assert result is not None
        wa = result["warp_assign"]
        assert "A" in wa and "B" in wa
        for v in wa.values():
            assert v in (0, 1)

    def test_single_warp_all_zero(self):
        smt = _make_smt_nodes([
            ("A", "TMA", 2, []),
            ("B", "ALU", 3, [("A", 0)]),
        ], num_warps=1)
        result = smt._solve_phase_b(ii=6, L=12)
        assert result is not None
        wa = result["warp_assign"]
        assert all(v == 0 for v in wa.values())


@pytest.mark.skipif(not _has_z3(), reason="z3-solver not installed")
class TestSMTBlockingSync:

    def test_same_warp_enforced(self):
        smt = _make_smt_nodes([
            ("P", "TMA", 4, [], [("oP", "RMEM", 128, 0)]),
            ("C", "ALU", 2, [("P", 0, True)]),
        ], num_warps=2)
        result = smt._solve_phase_b(ii=8, L=16)
        assert result is not None
        wa = result["warp_assign"]
        assert wa["P"] == wa["C"]

    def test_barrier_exclusion(self):
        """Third op on same warp must not overlap the blocking window."""
        smt = _make_smt_nodes([
            ("P", "TMA", 5, [], [("oP", "RMEM", 128, 0)]),
            ("C", "ALU", 2, [("P", 0, True)]),
            ("X", "ALU", 3, [], [("oX", "RMEM", 64, 0)]),
        ], num_warps=2)
        result = smt._solve_phase_b(ii=12, L=20)
        assert result is not None
        sched = result["schedule"]
        wa = result["warp_assign"]
        assert wa["P"] == wa["C"]

        if wa["X"] == wa["P"]:
            barrier_start = sched["C"] - 5
            barrier_end = sched["C"]
            x_start = sched["X"]
            x_end = sched["X"] + 3
            overlap = x_start < barrier_end and x_end > barrier_start
            assert not overlap


@pytest.mark.skipif(not _has_z3(), reason="z3-solver not installed")
class TestSMTSpillCost:

    def test_cross_warp_adds_delay(self):
        spill = 8
        base_lat = 2
        smt = _make_smt_nodes([
            ("A", "TMA", base_lat, [], [("oA", "RMEM", 128, spill)]),
            ("B", "ALU", 3, [("A", 0)]),
        ], num_warps=2)
        result = smt._solve_phase_b(ii=15, L=20)
        assert result is not None
        sched = result["schedule"]
        wa = result["warp_assign"]
        gap = sched["B"] - sched["A"]
        if wa["A"] != wa["B"]:
            assert gap >= base_lat + spill
        else:
            assert gap >= base_lat


@pytest.mark.skipif(not _has_z3(), reason="z3-solver not installed")
class TestSMTPerWarpRegCapacity:

    def test_reg_limit_per_warp(self):
        """Reg capacity is enforced per-warp: both outputs alive = 400 > 250."""
        smt = _make_smt_nodes([
            ("A", "TMA", 2, [], [("oA", "RMEM", 200, 0)]),
            ("B", "ALU", 2, [], [("oB", "RMEM", 200, 0)]),
            ("C", "TC", 2, [("A", 0), ("B", 0)]),
        ], num_warps=2, reg_limit=250)
        result = smt._solve_phase_b(ii=8, L=16)
        assert result is not None
        wa = result["warp_assign"]
        assert wa["A"] != wa["B"], (
            f"Expected different warps due to reg pressure, got A={wa['A']}, B={wa['B']}"
        )


@pytest.mark.skipif(not _has_z3(), reason="z3-solver not installed")
class TestSMTFUCapacity:

    def test_fu_cap_respected(self):
        """Two ALU ops at same time: if cap=1, they must be serialized."""
        smt = _make_smt_nodes([
            ("A", "ALU", 4, []),
            ("B", "ALU", 4, []),
        ], num_warps=1, fu_caps={
            ResourceType.TMA: 1, ResourceType.TensorCore: 1,
            ResourceType.ALU: 1, ResourceType.SFU: 1,
        })
        # Import SMT ResourceType
        from heddle.scheduler.smt import ResourceType as SRT
        smt.fu_caps = {SRT.TMA: 1, SRT.TensorCore: 1, SRT.ALU: 1, SRT.SFU: 1}
        result = smt._solve_phase_b(ii=8, L=16)
        assert result is not None
        sched = result["schedule"]
        a_range = (sched["A"], sched["A"] + 4)
        b_range = (sched["B"], sched["B"] + 4)
        overlap = a_range[0] < b_range[1] and b_range[0] < a_range[1]
        assert not overlap


@pytest.mark.skipif(not _has_z3(), reason="z3-solver not installed")
class TestSMTResultStructure:

    def test_result_has_all_fields(self):
        smt = _make_smt_nodes([
            ("A", "TMA", 2, [], [("oA", "RMEM", 64, 0)]),
            ("B", "ALU", 3, [("A", 0)]),
        ], num_warps=2)
        result = smt._solve_phase_b(ii=6, L=12)
        assert result is not None
        assert "ii" in result
        assert "schedule" in result
        assert "warp_assign" in result
        assert "reg_peak" in result

    def test_schedule_respects_deps(self):
        smt = _make_smt_nodes([
            ("A", "TMA", 3, []),
            ("B", "ALU", 2, [("A", 0)]),
            ("C", "TC", 4, [("B", 0)]),
        ], num_warps=1)
        result = smt._solve_phase_b(ii=10, L=20)
        assert result is not None
        s = result["schedule"]
        assert s["B"] >= s["A"] + 3
        assert s["C"] >= s["B"] + 2


# ======================================================================
# Cross-solver consistency: CP-SAT vs SMT on same input
# ======================================================================

@pytest.mark.skipif(not _has_z3(), reason="z3-solver not installed")
class TestCrossSolverConsistency:

    def test_fixed_warp_consistent(self):
        """Both solvers respect fixed_warp assignments."""
        # CP-SAT
        ops = [
            OpSpec("A", ResourceType.TMA, latency=2, fixed_warp=0),
            OpSpec("B", ResourceType.ALU, latency=3, fixed_warp=1,
                   deps=[("A", 0)]),
        ]
        cpsat_result = _make_cpsat(ops, num_warps=2).solve()
        assert cpsat_result is not None
        cw = cpsat_result.kernel_warp_assigns["k0"]
        assert cw["A"] == 0 and cw["B"] == 1

        # SMT — no fixed_warp field, but we can verify warp_assign exists
        smt = _make_smt_nodes([
            ("A", "TMA", 2, []),
            ("B", "ALU", 3, [("A", 0)]),
        ], num_warps=2)
        smt_result = smt._solve_phase_b(ii=6, L=12)
        assert smt_result is not None
        sw = smt_result["warp_assign"]
        assert "A" in sw and "B" in sw

    def test_blocking_sync_both_enforce_same_warp(self):
        """Both solvers enforce same-warp for blocking_sync edges."""
        # CP-SAT
        ops = [
            OpSpec("P", ResourceType.TMA, latency=4,
                   outputs=[OutputSpec("oP", StorageKind.RMEM, 128)]),
            OpSpec("C", ResourceType.ALU, latency=2,
                   deps=[("P", 0, True)]),
        ]
        cpsat_result = _make_cpsat(ops, num_warps=2).solve()
        assert cpsat_result is not None
        cw = cpsat_result.kernel_warp_assigns["k0"]
        assert cw["P"] == cw["C"]

        # SMT
        smt = _make_smt_nodes([
            ("P", "TMA", 4, [], [("oP", "RMEM", 128, 0)]),
            ("C", "ALU", 2, [("P", 0, True)]),
        ], num_warps=2)
        smt_result = smt._solve_phase_b(ii=8, L=16)
        assert smt_result is not None
        sw = smt_result["warp_assign"]
        assert sw["P"] == sw["C"]


# ======================================================================
# _unpack_dep edge cases
# ======================================================================

class TestUnpackDepEdgeCases:

    def test_2tuple_defaults_blocking_false(self):
        name, dist, blocking = _unpack_dep(("x", 5))
        assert (name, dist, blocking) == ("x", 5, False)

    def test_3tuple_true(self):
        name, dist, blocking = _unpack_dep(("y", 0, True))
        assert blocking is True

    def test_3tuple_false(self):
        name, dist, blocking = _unpack_dep(("z", 3, False))
        assert blocking is False

    def test_zero_distance(self):
        name, dist, blocking = _unpack_dep(("a", 0))
        assert dist == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
