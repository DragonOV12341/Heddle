# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
BWD Separator: Automatic Backward Pass Separation Detection and Code Generation

This module implements automatic detection and separation of backward passes
for attention-like operators. The key insight is that BWD can often be split
into independent kernels (dKdV and dQ) to avoid atomic_add overhead.

Design Document:
----------------
Problem: Unified BWD kernels for attention require atomic_add to accumulate
         gradients, which creates contention and limits performance.

Solution: Automatically detect when BWD can be separated into:
         - dKdV kernel: For each (K,V) block, iterate over Q blocks
         - dQ kernel: For each Q block, iterate over (K,V) blocks

Integration Points:
1. patterns.py: TileLangFlashAttnFunc.backward() calls separated kernels
2. partitioner.py: Detect BWD subgraph separability in graph-level fusion
3. decomposition.py: Decompose unified BWD into separated form

Usage:
------
# Method 1: Via torch.autograd.Function (recommended for now)
class MyAttnFunc(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_out):
        return separated_bwd(ctx, grad_out)

# Method 2: Via graph analysis (future work)
if analyze_bwd_separability(fx_graph):
    dkdv_graph, dq_graph = split_bwd_graph(fx_graph)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.fx as fx


class BWDSeparationMode(Enum):
    """BWD separation mode selection."""
    AUTO = "auto"           # Automatically decide based on shape/heuristics
    UNIFIED = "unified"     # Force unified (single kernel) BWD
    SEPARATED = "separated" # Force separated (dKdV + dQ) BWD


@dataclass
class BWDSeparationConfig:
    """Configuration for BWD separation decision."""
    mode: BWDSeparationMode = BWDSeparationMode.AUTO
    
    # Thresholds for AUTO mode
    min_seq_len_for_separation: int = 1024  # Below this, unified is faster
    min_batch_heads: int = 4  # B*H threshold
    
    # Performance tuning
    dkdv_block_m: int = 64
    dkdv_block_n: int = 128
    dkdv_num_stages: int = 2
    dkdv_threads: int = 128
    
    dq_block_m: int = 128
    dq_block_n: int = 64
    dq_num_stages: int = 2
    dq_threads: int = 128
    
    @classmethod
    def from_env(cls) -> "BWDSeparationConfig":
        """Load configuration from environment variables."""
        mode_str = os.environ.get("TILELANG_BWD_SEPARATION_MODE", "auto").strip().lower()
        mode = {
            "auto": BWDSeparationMode.AUTO,
            "unified": BWDSeparationMode.UNIFIED,
            "separated": BWDSeparationMode.SEPARATED,
        }.get(mode_str, BWDSeparationMode.AUTO)
        
        return cls(
            mode=mode,
            min_seq_len_for_separation=int(os.environ.get(
                "TILELANG_BWD_MIN_SEQ_FOR_SEP", "1024")),
            min_batch_heads=int(os.environ.get(
                "TILELANG_BWD_MIN_BATCH_HEADS", "4")),
        )


def should_separate_bwd(
    B: int,
    H: int,
    T: int,
    D: int,
    config: Optional[BWDSeparationConfig] = None,
) -> bool:
    """
    Decide whether to use separated BWD based on shape and configuration.
    
    Heuristics:
    1. Larger sequences benefit more from separation (less atomic contention)
    2. Larger batch*heads means more parallel work, favoring separation
    3. Separation has kernel launch overhead, so avoid for small shapes
    
    Args:
        B: Batch size
        H: Number of attention heads
        T: Sequence length
        D: Head dimension
        config: Optional configuration (loads from env if None)
        
    Returns:
        True if separated BWD should be used
    """
    if config is None:
        config = BWDSeparationConfig.from_env()
    
    if config.mode == BWDSeparationMode.UNIFIED:
        return False
    if config.mode == BWDSeparationMode.SEPARATED:
        return True
    
    # AUTO mode heuristics
    if T < config.min_seq_len_for_separation:
        return False
    if B * H < config.min_batch_heads:
        return False
    
    # For very long sequences, separation is almost always better
    if T >= 4096:
        return True
    
    # Middle ground: use separation if enough parallelism
    return B * H >= 8


# ============================================================================
# Graph-Level BWD Separation Analysis (Future Work)
# ============================================================================

@dataclass
class BWDGradientNode:
    """Represents a gradient computation node in BWD graph."""
    name: str  # e.g., "dQ", "dK", "dV"
    fx_nodes: Set[fx.Node]
    inputs: Set[fx.Node]  # FWD tensors needed (Q, K, V, O, LSE, grad_out)
    outputs: Set[fx.Node]
    
    # Dependency analysis
    depends_on: Set[str] = None  # Other gradient names this depends on
    produces_atomic: bool = False  # Whether this needs atomic_add


class BWDGraphAnalyzer:
    """
    Analyze BWD computation graph to detect separability.
    
    A BWD graph is separable if:
    1. dK, dV computation does not depend on dQ intermediate results
    2. dQ computation does not depend on dK, dV intermediate results
    3. The only shared computation is FWD outputs (Q, K, V, O, LSE)
    
    This is typically true for attention BWD because:
    - dK = sum_over_q(dS^T @ Q) where dS = grad_out @ V^T * softmax_grad
    - dV = sum_over_q(S^T @ grad_out)
    - dQ = sum_over_kv(dS @ K)
    
    Each gradient accumulates over a different dimension, so they can
    be computed independently (with only FWD data as shared input).
    """
    
    def __init__(self, gm: fx.GraphModule):
        self.gm = gm
        self.gradient_nodes: Dict[str, BWDGradientNode] = {}
        
    def analyze(self) -> Tuple[bool, Optional[Dict[str, BWDGradientNode]]]:
        """
        Analyze the graph and return (is_separable, gradient_nodes).
        
        Returns:
            (True, {gradient_nodes}) if BWD can be separated
            (False, None) if BWD must be unified
        """
        # Step 1: Identify gradient output nodes (dQ, dK, dV)
        grad_outputs = self._find_gradient_outputs()
        if not grad_outputs or len(grad_outputs) < 3:
            return False, None
        
        # Step 2: Trace backward to find computation subgraphs
        for name, out_node in grad_outputs.items():
            self.gradient_nodes[name] = self._trace_gradient_subgraph(name, out_node)
        
        # Step 3: Check for cross-dependencies
        is_separable = self._check_separability()
        
        return is_separable, self.gradient_nodes if is_separable else None
    
    def _find_gradient_outputs(self) -> Dict[str, fx.Node]:
        """
        Find nodes that produce dQ, dK, dV in the graph.
        
        This is heuristic-based: we look for patterns like:
        - Output tuple with 3 elements (assuming dQ, dK, dV order)
        - Nodes with names containing "grad_q", "grad_k", "grad_v"
        - Nodes that are the result of sum/accumulate operations
        """
        outputs = {}
        for node in self.gm.graph.nodes:
            if node.op == "output":
                out_args = node.args[0]
                if isinstance(out_args, tuple) and len(out_args) >= 3:
                    # Assume order: dQ, dK, dV (standard PyTorch convention)
                    outputs["dQ"] = out_args[0] if isinstance(out_args[0], fx.Node) else None
                    outputs["dK"] = out_args[1] if isinstance(out_args[1], fx.Node) else None
                    outputs["dV"] = out_args[2] if isinstance(out_args[2], fx.Node) else None
                break
        return {k: v for k, v in outputs.items() if v is not None}
    
    def _trace_gradient_subgraph(self, name: str, out_node: fx.Node) -> BWDGradientNode:
        """
        Trace backward from gradient output to find all nodes in its computation.
        """
        visited = set()
        inputs = set()
        
        def trace(node: fx.Node):
            if node in visited:
                return
            visited.add(node)
            
            if node.op == "placeholder":
                inputs.add(node)
                return
            
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    trace(arg)
                elif isinstance(arg, (list, tuple)):
                    for a in arg:
                        if isinstance(a, fx.Node):
                            trace(a)
        
        trace(out_node)
        
        # Check if any node uses atomic_add
        produces_atomic = any(
            self._is_atomic_op(n) for n in visited
        )
        
        return BWDGradientNode(
            name=name,
            fx_nodes=visited,
            inputs=inputs,
            outputs={out_node},
            depends_on=set(),
            produces_atomic=produces_atomic,
        )
    
    def _is_atomic_op(self, node: fx.Node) -> bool:
        """Check if a node represents an atomic operation."""
        if node.op != "call_function":
            return False
        target_name = str(node.target)
        return "atomic" in target_name.lower() or "scatter_add" in target_name.lower()
    
    def _check_separability(self) -> bool:
        """
        Check if gradient computations are separable.
        
        Separable if: dK/dV nodes ∩ dQ nodes == shared FWD inputs only
        """
        if "dQ" not in self.gradient_nodes:
            return False
        if "dK" not in self.gradient_nodes or "dV" not in self.gradient_nodes:
            return False
        
        dq_nodes = self.gradient_nodes["dQ"].fx_nodes
        dk_nodes = self.gradient_nodes["dK"].fx_nodes
        dv_nodes = self.gradient_nodes["dV"].fx_nodes
        
        # Combine dK and dV (they're usually computed together)
        dkdv_nodes = dk_nodes | dv_nodes
        
        # Find intersection (shared computation)
        shared = dq_nodes & dkdv_nodes
        
        # Remove placeholder nodes (FWD inputs are expected to be shared)
        shared_compute = {n for n in shared if n.op != "placeholder"}
        
        # If there's significant shared compute, not separable
        # Allow some threshold for common preprocessing (e.g., delta computation)
        max_shared_compute = 5  # Heuristic threshold
        
        if len(shared_compute) > max_shared_compute:
            return False
        
        # Check for atomic operations
        dq_atomic = self.gradient_nodes["dQ"].produces_atomic
        dkdv_atomic = any(
            self.gradient_nodes[g].produces_atomic 
            for g in ["dK", "dV"] if g in self.gradient_nodes
        )
        
        # Separation eliminates atomic ops if they exist
        if dq_atomic or dkdv_atomic:
            # Separation is beneficial
            return True
        
        return True


def analyze_bwd_separability(gm: fx.GraphModule) -> bool:
    """
    High-level API to check if a BWD graph can be separated.
    
    Args:
        gm: FX GraphModule representing BWD computation
        
    Returns:
        True if BWD can be separated into dKdV and dQ kernels
    """
    analyzer = BWDGraphAnalyzer(gm)
    is_sep, _ = analyzer.analyze()
    return is_sep


def split_bwd_graph(
    gm: fx.GraphModule,
) -> Optional[Tuple[fx.GraphModule, fx.GraphModule]]:
    """
    Split a BWD graph into separate dKdV and dQ subgraphs.
    
    This is a more advanced operation that creates two independent
    GraphModules that can be compiled and executed separately.
    
    Args:
        gm: Unified BWD GraphModule
        
    Returns:
        (dkdv_gm, dq_gm) if separable, None otherwise
        
    Note: This is a placeholder for future implementation.
    Currently, we recommend using the autograd.Function approach
    where separation is done at the Python level.
    """
    # TODO: Implement graph splitting
    # This would involve:
    # 1. Analyzing the graph with BWDGraphAnalyzer
    # 2. Extracting dKdV and dQ subgraphs
    # 3. Creating new GraphModules with proper inputs/outputs
    # 4. Handling shared preprocessing (delta, softmax grad)
    raise NotImplementedError(
        "Graph-level BWD splitting is not yet implemented. "
        "Please use the torch.autograd.Function approach in patterns.py."
    )


# ============================================================================
# Separated BWD Kernel Interface (for patterns.py integration)
# ============================================================================

def create_separated_bwd_function(
    flashattn_bwd_preprocess: Callable,
    flashattn_bwd_dkdv: Callable,
    flashattn_bwd_dq: Callable,
) -> Callable:
    """
    Create a separated BWD function from three kernel factories.
    
    This is the main integration point for patterns.py. It wraps
    the separated kernels into a single backward function that can
    be used in torch.autograd.Function.
    
    Args:
        flashattn_bwd_preprocess: Kernel that computes delta = rowsum(O * dO)
        flashattn_bwd_dkdv: Kernel that computes dK, dV
        flashattn_bwd_dq: Kernel that computes dQ
        
    Returns:
        A function that takes (Q, K, V, O, dO, LSE) and returns (dQ, dK, dV)
        
    Example:
        # In patterns.py:
        from heddle.tileop.flash_attention_bwd import (
            flashattn_bwd_preprocess,
            flashattn_bwd_dkdv,
            flashattn_bwd_dq,
        )
        
        separated_bwd = create_separated_bwd_function(
            flashattn_bwd_preprocess,
            flashattn_bwd_dkdv,
            flashattn_bwd_dq,
        )
        
        class TileLangFlashAttnFunc(torch.autograd.Function):
            @staticmethod
            def backward(ctx, grad_out):
                Q, K, V, O, LSE = ctx.saved_tensors
                if should_separate_bwd(B, H, T, D):
                    return separated_bwd(Q, K, V, O, grad_out, LSE)
                else:
                    return unified_bwd(Q, K, V, O, grad_out, LSE)
    """
    def separated_bwd(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        O: torch.Tensor,
        grad_out: torch.Tensor,
        LSE: torch.Tensor,
        is_causal: bool = False,
        config: Optional[BWDSeparationConfig] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute separated BWD: preprocess -> dKdV -> dQ
        
        Each kernel is independent and can be optimized separately.
        No atomic_add is needed because each output is written exclusively.
        """
        if config is None:
            config = BWDSeparationConfig.from_env()
        
        B, H, T, D = Q.shape
        
        # Step 1: Preprocess - compute delta = rowsum(O * dO)
        delta = flashattn_bwd_preprocess(B, H, T, D)(O, grad_out)
        
        # Step 2: dKdV - for each (k,v) block, iterate over q blocks
        dK, dV = flashattn_bwd_dkdv(
            B, H, T, T, D,
            is_causal=is_causal,
            block_M=config.dkdv_block_m,
            block_N=config.dkdv_block_n,
            num_stages=config.dkdv_num_stages,
            threads=config.dkdv_threads,
        )(Q, K, V, O, grad_out, LSE, delta)
        
        # Step 3: dQ - for each q block, iterate over (k,v) blocks
        dQ = flashattn_bwd_dq(
            B, H, T, T, D,
            is_causal=is_causal,
            block_M=config.dq_block_m,
            block_N=config.dq_block_n,
            num_stages=config.dq_num_stages,
            threads=config.dq_threads,
        )(Q, K, V, O, grad_out, LSE, delta)
        
        return dQ, dK, dV
    
    return separated_bwd


# ============================================================================
# Integration with TileLang Frontend Pipeline
# ============================================================================

def register_bwd_separation_pass():
    """
    Register BWD separation as a pass in the TileLang frontend pipeline.
    
    This enables automatic BWD separation during torch.compile:
    
    1. Detect attention BWD patterns in the FX graph
    2. Analyze separability
    3. Replace unified BWD with separated kernels
    
    Note: This is a placeholder for future integration with backend.py
    """
    # TODO: Integrate with tilelang_backend() in backend.py
    # 
    # The integration would look like:
    # 
    # def tilelang_backend(gm, example_inputs):
    #     ...
    #     # After pattern matching, check for BWD patterns
    #     if detect_attention_bwd_pattern(gm):
    #         if analyze_bwd_separability(gm):
    #             gm = rewrite_to_separated_bwd(gm)
    #     ...
    pass


# ============================================================================
# Diagnostic Utilities
# ============================================================================

def diagnose_bwd_graph(gm: fx.GraphModule) -> str:
    """
    Generate a diagnostic report for a BWD graph.
    
    Useful for understanding why separation is/isn't possible.
    """
    analyzer = BWDGraphAnalyzer(gm)
    is_sep, grad_nodes = analyzer.analyze()
    
    lines = [
        "=" * 60,
        "BWD Graph Diagnostic Report",
        "=" * 60,
        f"Separable: {is_sep}",
        "",
    ]
    
    if grad_nodes:
        for name, gnode in grad_nodes.items():
            lines.extend([
                f"Gradient: {name}",
                f"  - Compute nodes: {len(gnode.fx_nodes)}",
                f"  - Input nodes: {len(gnode.inputs)}",
                f"  - Uses atomic: {gnode.produces_atomic}",
                "",
            ])
    
    lines.append("=" * 60)
    return "\n".join(lines)
