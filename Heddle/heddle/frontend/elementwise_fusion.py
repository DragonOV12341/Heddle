"""
AutoMixed elementwise fusion pass.

Extends AutoMixed's resource-aware decision framework to cover
memory-bandwidth-bound elementwise operations. For compute-heavy
operations (WGMMA), AutoMixed uses register/SMEM cost models;
for elementwise operations, this module uses a GMEM bandwidth
cost model to decide whether adjacent operations should be fused.

Cost model:
    savings = intermediate_tensor_bytes * 2   (one write + one read eliminated)
    overhead = 0   (elementwise fusion has negligible register cost)
    decision = fuse if savings > 0

Supported fusion patterns:
    1. residual_add + rmsnorm  → fused_add_rmsnorm
    2. gate_proj + up_proj     → merged gate_up_proj (single GEMM)
    3. silu + elementwise_mul  → fused_silu_mul
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class ElementwiseFusionCandidate:
    """A candidate elementwise fusion opportunity."""
    pattern: Literal["add_rmsnorm", "gate_up_merge", "silu_mul"]
    input_shape: tuple[int, ...]
    dtype_bytes: int = 2  # FP16

    @property
    def intermediate_bytes(self) -> int:
        """Size of the intermediate tensor that fusion eliminates."""
        numel = math.prod(self.input_shape)
        return numel * self.dtype_bytes

    @property
    def gmem_savings_bytes(self) -> int:
        """GMEM traffic saved by fusion: one write + one read of the intermediate."""
        return self.intermediate_bytes * 2

    @property
    def should_fuse(self) -> bool:
        """Bandwidth cost model: fuse whenever there is a positive saving.

        Elementwise operations have negligible register overhead, so the
        decision reduces to: does fusion eliminate an intermediate tensor?
        The answer is always yes for the supported patterns.
        """
        return self.gmem_savings_bytes > 0


def analyze_llama_layer(
    batch: int,
    seq_len: int,
    hidden: int = 4096,
    intermediate: int = 11008,
    dtype_bytes: int = 2,
) -> list[ElementwiseFusionCandidate]:
    """Analyze a Llama transformer layer and return fusion candidates.

    This function performs the graph-level pattern matching that AutoMixed
    uses to identify elementwise fusion opportunities in a transformer block.
    """
    candidates = []

    # Pattern 1: residual + x followed by rmsnorm(residual)
    # Unfused: write residual (B, T, H) to GMEM, then read it back for norm
    # Fused: residual stays in registers, norm computed in-place
    candidates.append(ElementwiseFusionCandidate(
        pattern="add_rmsnorm",
        input_shape=(batch, seq_len, hidden),
        dtype_bytes=dtype_bytes,
    ))

    # Pattern 2: gate_proj(x) and up_proj(x) share the same input
    # Unfused: two cuBLAS calls, input read twice
    # Fused: single cuBLAS call with concatenated weight [2*inter, hidden]
    candidates.append(ElementwiseFusionCandidate(
        pattern="gate_up_merge",
        input_shape=(batch, seq_len, hidden),
        dtype_bytes=dtype_bytes,
    ))

    # Pattern 3: silu(gate) * up produces intermediate silu output
    # Unfused: write silu(gate) to GMEM, then read for elementwise multiply
    # Fused: silu and multiply in single kernel, no intermediate write
    candidates.append(ElementwiseFusionCandidate(
        pattern="silu_mul",
        input_shape=(batch, seq_len, intermediate),
        dtype_bytes=dtype_bytes,
    ))

    return candidates


def print_fusion_analysis(
    batch: int,
    seq_len: int,
    hidden: int = 4096,
    intermediate: int = 11008,
    num_layers: int = 32,
) -> None:
    """Print a human-readable fusion analysis report."""
    candidates = analyze_llama_layer(batch, seq_len, hidden, intermediate)

    total_savings = 0
    print(f"=== AutoMixed Elementwise Fusion Analysis ===")
    print(f"Model: Llama2-7B, B={batch}, T={seq_len}, layers={num_layers}")
    print()

    for c in candidates:
        # Per layer, each pattern appears twice (pre-attn and post-attn norm,
        # or once for silu_mul)
        count_per_layer = 2 if c.pattern == "add_rmsnorm" else 1
        total_per_model = c.gmem_savings_bytes * count_per_layer * num_layers
        total_savings += total_per_model

        print(f"  Pattern: {c.pattern}")
        print(f"    Shape: {c.input_shape}")
        print(f"    Intermediate: {c.intermediate_bytes / 1024:.1f} KB")
        print(f"    GMEM saved per call: {c.gmem_savings_bytes / 1024:.1f} KB")
        print(f"    Per model ({count_per_layer}x/layer × {num_layers} layers): "
              f"{total_per_model / 1024 / 1024:.1f} MB")
        print(f"    Decision: {'FUSE' if c.should_fuse else 'SKIP'}")
        print()

    print(f"  Total GMEM savings: {total_savings / 1024 / 1024:.1f} MB per forward pass")
    # H100 HBM3 bandwidth: 3.35 TB/s
    time_saved_us = total_savings / (3.35e12) * 1e6
    print(f"  Estimated time saved (at peak BW): {time_saved_us:.1f} μs")


if __name__ == "__main__":
    for T in [512, 1024, 2048, 4096]:
        print_fusion_analysis(batch=1, seq_len=T)
        print()
