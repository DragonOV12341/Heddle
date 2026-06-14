"""Integration tests for the per-op warp dispatch (Plan B) pipeline.

Tests the Python → annotation → C++ config path without tilelang.
Verifies format consistency between the Python warp_assigns string
producer (heddle_consumer_schedule.py) and the C++ consumer
(ParseWarpAssigns in finegrained_ws.cc).
"""

import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def _has_ortools():
    try:
        from ortools.sat.python import cp_model
        return True
    except ImportError:
        return False


def parse_warp_assigns_py(config_str: str) -> dict:
    """Python mirror of C++ ParseWarpAssigns for format validation."""
    result = {}
    if not config_str:
        return result
    for entry in config_str.split(","):
        if ":" not in entry:
            continue
        name, warp_id_str = entry.split(":", 1)
        warp_id = int(warp_id_str)
        if len(name) > 1 and name[0] == "s":
            ci = int(name[1:])
            result[ci] = warp_id
    return result


class TestParseWarpAssigns:

    def test_empty(self):
        assert parse_warp_assigns_py("") == {}

    def test_single(self):
        assert parse_warp_assigns_py("s0:0") == {0: 0}

    def test_multiple(self):
        result = parse_warp_assigns_py("s0:0,s1:1,s2:0,s3:1")
        assert result == {0: 0, 1: 1, 2: 0, 3: 1}

    def test_non_sequential(self):
        result = parse_warp_assigns_py("s3:2,s7:0,s1:1")
        assert result == {3: 2, 7: 0, 1: 1}

    def test_malformed_entries_skipped(self):
        result = parse_warp_assigns_py("s0:0,badentry,s1:1,x2:3")
        assert result == {0: 0, 1: 1}

    def test_three_warp_groups(self):
        result = parse_warp_assigns_py("s0:0,s1:1,s2:2,s3:0")
        assert result == {0: 0, 1: 1, 2: 2, 3: 0}
        max_wg = max(result.values())
        assert max_wg + 1 == 3  # 3 warp groups


class TestWarpAssignsFormatConsistency:
    """Verify the Python producer format matches C++ consumer expectations."""

    def test_format_from_solver_result(self):
        """Simulate what heddle_consumer_schedule.py produces."""
        warp_assigns = {"s0": 0, "s1": 1, "s2": 0, "s3": 1}
        warp_str = ",".join(f"{k}:{v}" for k, v in sorted(warp_assigns.items()))
        assert warp_str == "s0:0,s1:1,s2:0,s3:1"

        parsed = parse_warp_assigns_py(warp_str)
        assert parsed == {0: 0, 1: 1, 2: 0, 3: 1}

    def test_roundtrip_from_dict(self):
        """Dict -> string -> parse roundtrip."""
        original = {0: 0, 1: 1, 2: 0, 3: 2, 4: 1}
        warp_str = ",".join(f"s{k}:{v}" for k, v in sorted(original.items()))
        parsed = parse_warp_assigns_py(warp_str)
        assert parsed == original


class TestMonkeyPatchConfigKeys:

    def test_warp_assigns_keys_registered(self):
        from heddle._monkey_patch import HEDDLE_CONFIG_KEYS

        assert "TL_PCWS_WARP_ASSIGNS" in HEDDLE_CONFIG_KEYS
        assert HEDDLE_CONFIG_KEYS["TL_PCWS_WARP_ASSIGNS"] == "tl.pcws_warp_assigns"
        assert "TL_FINEGRAINEDWS_WARP_ASSIGNS" in HEDDLE_CONFIG_KEYS
        assert HEDDLE_CONFIG_KEYS["TL_FINEGRAINEDWS_WARP_ASSIGNS"] == "tl.finegrainedws_warp_assigns"


class TestCppSourceConsistency:
    """Read the C++ source and verify structural consistency."""

    CPP_PATH = os.path.join(
        os.path.dirname(__file__), "..",
        "patches", "cpp", "src", "transform", "finegrained_ws.cc"
    )
    BUILTIN_PATH = os.path.join(
        os.path.dirname(__file__), "..",
        "patches", "cpp", "src", "op", "builtin.h"
    )

    def test_builtin_h_has_constant(self):
        with open(self.BUILTIN_PATH) as f:
            src = f.read()
        assert 'kFineGrainedWsWarpAssigns' in src
        assert '"tl.finegrainedws_warp_assigns"' in src

    def test_parse_warp_assigns_exists(self):
        with open(self.CPP_PATH) as f:
            src = f.read()
        assert "ParseWarpAssigns" in src

    def test_warp_assigns_map_member(self):
        with open(self.CPP_PATH) as f:
            src = f.read()
        assert "warp_assigns_map_" in src

    def test_substitute_signature_includes_warp_assigns(self):
        with open(self.CPP_PATH) as f:
            src = f.read()
        assert "warp_assigns_str" in src

    def test_per_op_dispatch_block_exists(self):
        with open(self.CPP_PATH) as f:
            src = f.read()
        assert "Plan B: Per-op warp dispatch" in src
        assert "track_warp_groups" in src
        assert "consumer_stmt_warp_group" in src

    def test_pass_registration_reads_warp_assigns(self):
        with open(self.CPP_PATH) as f:
            src = f.read()
        assert "kFineGrainedWsWarpAssigns" in src
        # The constant is defined in builtin.h and used in finegrained_ws.cc
        with open(self.BUILTIN_PATH) as f:
            builtin_src = f.read()
        assert "kFineGrainedWsWarpAssigns" in builtin_src
        # finegrained_ws.cc should reference it (pass config reading)
        assert "warp_assigns_str" in src

    def test_annotation_reading_both_variants(self):
        with open(self.CPP_PATH) as f:
            src = f.read()
        assert "tl_finegrainedws_warp_assigns" in src
        assert "tl_pcws_warp_assigns" in src

    def test_dispatch_chain_order(self):
        """Verify dispatch chain: three_role → dual_consumer → per_op → standard."""
        with open(self.CPP_PATH) as f:
            src = f.read()
        three_role_pos = src.find("if (has_three_role)")
        dual_consumer_pos = src.find("else if (dual_consumer_enabled_)")
        per_op_pos = src.find("else if (track_warp_groups")
        standard_pos = src.find("standard_two_role:")

        assert three_role_pos < dual_consumer_pos < per_op_pos < standard_pos, \
            f"Dispatch chain order wrong: three_role={three_role_pos}, " \
            f"dual_consumer={dual_consumer_pos}, per_op={per_op_pos}, " \
            f"standard={standard_pos}"


class TestSolverWarpAssignsEndToEnd:
    """Test that cp_sat solve() produces warp_assigns that parse correctly."""

    @pytest.mark.skipif(
        not _has_ortools(), reason="ortools not installed"
    )
    def test_solver_output_parseable(self):
        from heddle.scheduler.cp_sat import (
            UnifiedScheduler, PartitionSpec, KernelSpec, OpSpec,
            OutputSpec, ResourceType, StorageKind,
        )

        ops = [
            OpSpec("A", ResourceType.TMA, latency=2, fixed_warp=0,
                   outputs=[OutputSpec("oA", StorageKind.RMEM, 128)]),
            OpSpec("B", ResourceType.ALU, latency=3, fixed_warp=1,
                   deps=[("A", 0)],
                   outputs=[OutputSpec("oB", StorageKind.RMEM, 64)]),
            OpSpec("C", ResourceType.TensorCore, latency=4, fixed_warp=0,
                   deps=[("B", 0)]),
        ]
        kernel = KernelSpec(name="k0", ops=ops, smem_bytes=1024, threads=128)
        partition = PartitionSpec(name="p0", kernels=[kernel])
        solver = UnifiedScheduler(
            [partition],
            fu_caps={ResourceType.TMA: 1, ResourceType.TensorCore: 1,
                     ResourceType.ALU: 2, ResourceType.SFU: 1},
            reg_limit=256,
            num_warps=2,
            timeout_s=5,
        )
        result = solver.solve()

        assert result is not None
        warps = result.kernel_warp_assigns["k0"]
        assert warps["A"] == 0
        assert warps["B"] == 1
        assert warps["C"] == 0

        # Convert to annotation format (what heddle_consumer_schedule.py does)
        # Note: solver uses op names, annotation uses sN indices
        # Simulate: map op index → warp
        warp_str = ",".join(
            f"s{i}:{warps[op.name]}" for i, op in enumerate(ops)
        )
        # Verify it round-trips through parser
        parsed = parse_warp_assigns_py(warp_str)
        assert parsed == {0: 0, 1: 1, 2: 0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
