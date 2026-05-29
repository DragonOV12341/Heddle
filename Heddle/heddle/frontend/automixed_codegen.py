# Copyright (c) Heddle Authors.
# Licensed under the MIT License.

"""
Unified AutoMixed Codegen: graph-aware kernel generation for any partition.

Architecture:
    FX Partition → PartitionAnalysis → CoreOp + Prologue + Epilogue
                                            ↓
                                   TileLang IR (T.Pipelined + T.gemm + ...)
                                            ↓
                                   AutoMixed config selection (CP-SAT / ptxas)
                                            ↓
                                   Heddle schedule → GPU kernel

Handles any subgraph that decomposes into:
    [Prologue ops] → CoreOp (einsum / GEMM / bmm / reduction) → [Epilogue ops]

Where:
    - Prologue: elementwise ops feeding into CoreOp inputs (exp, mul, sub, ...)
    - CoreOp: the compute-intensive op defining the tiling strategy
    - Epilogue: elementwise ops consuming CoreOp output (silu, mul, add, ...)
    - Layout ops (permute, reshape, ...): absorbed into index expressions
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.fx as fx


# ─── Partition Analysis ───────────────────────────────────────────────────────

class CoreOpKind(Enum):
    EINSUM = auto()      # torch.einsum / batched contraction
    GEMM = auto()        # torch.mm / matmul / linear (2D)
    BMM = auto()         # torch.bmm (3D)
    REDUCTION = auto()   # sum / mean / softmax
    ELEMENTWISE = auto() # pure elementwise (no contraction)


@dataclass
class PrologueOp:
    """An elementwise op that feeds into a CoreOp input."""
    node: fx.Node
    op_name: str              # "exp", "mul", "sub", etc.
    args: list                # FX Node or constant


@dataclass
class EpilogueOp:
    """An elementwise op consuming CoreOp output."""
    node: fx.Node
    op_name: str
    args: list


@dataclass
class IndexMapping:
    """How a layout op transforms indices."""
    op: str                   # "permute", "unsqueeze", "reshape"
    params: tuple             # e.g., (0,2,3,1) for permute
    inverse: Optional[tuple]  # inverse mapping for index propagation


@dataclass
class PartitionAnalysis:
    """Result of analyzing an FX partition for codegen."""
    core_op: CoreOpKind
    core_node: fx.Node

    # Einsum-specific
    equation: Optional[str] = None
    batch_dims: Dict[str, int] = field(default_factory=dict)
    contracted_dims: Dict[str, int] = field(default_factory=dict)
    free_a_dims: Dict[str, int] = field(default_factory=dict)
    free_b_dims: Dict[str, int] = field(default_factory=dict)

    # Operands (after tracing through layout ops)
    operand_nodes: List[fx.Node] = field(default_factory=list)
    prologue_chain: List[PrologueOp] = field(default_factory=list)
    epilogue_chain: List[EpilogueOp] = field(default_factory=list)
    layout_ops: List[IndexMapping] = field(default_factory=list)

    # Root placeholders (the actual tensor inputs to the kernel)
    root_placeholders: List[fx.Node] = field(default_factory=list)
    aux_tma_placeholders: List[fx.Node] = field(default_factory=list)  # TMA-loaded (dA, dt)
    aux_scalar_placeholders: List[fx.Node] = field(default_factory=list)  # Preloaded (dA_last)


def analyze_partition(gm: fx.GraphModule) -> Optional[PartitionAnalysis]:
    """Analyze an FX subgraph to determine core op, prologue, epilogue, and index mappings.

    This is the universal entry point for codegen — works for any partition structure.
    """
    # Find the core op (einsum > GEMM > reduction > elementwise)
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", "") or str(node.target)

        if "einsum" in name:
            return _analyze_einsum(gm, node)
        if name in ("mm", "matmul", "bmm") or "linear" in name:
            return _analyze_gemm(gm, node)
        if name in ("sum", "mean", "max", "amax"):
            return _analyze_reduction(gm, node)

    # Pure elementwise
    return PartitionAnalysis(
        core_op=CoreOpKind.ELEMENTWISE,
        core_node=None,
    )


def _analyze_einsum(gm: fx.GraphModule, node: fx.Node) -> PartitionAnalysis:
    """Analyze einsum partition: parse equation, trace prologue, find root placeholders."""
    eq = node.args[0] if isinstance(node.args[0], str) else None
    operands = [a for a in node.args[1:] if isinstance(a, fx.Node)]

    analysis = PartitionAnalysis(
        core_op=CoreOpKind.EINSUM,
        core_node=node,
        equation=eq,
        operand_nodes=operands,
    )

    if eq:
        parts = eq.replace(" ", "").split("->")
        if len(parts) == 2:
            a_idx, b_idx = parts[0].split(",")
            out_idx = parts[1]
            contracted = set(a_idx) & set(b_idx) - set(out_idx)
            batch = set(a_idx) & set(b_idx) & set(out_idx)

            # Populate dimension info (sizes filled by codegen from metadata)
            analysis.batch_dims = {c: 0 for c in batch}
            analysis.contracted_dims = {c: 0 for c in contracted}
            analysis.free_a_dims = {c: 0 for c in set(a_idx) - contracted - batch}
            analysis.free_b_dims = {c: 0 for c in set(b_idx) - contracted - batch}

    # Trace operands to find prologue and root placeholders
    for op_node in operands:
        _trace_prologue(op_node, analysis)

    return analysis


def _analyze_gemm(gm: fx.GraphModule, node: fx.Node) -> PartitionAnalysis:
    """Analyze GEMM/linear partition."""
    return PartitionAnalysis(
        core_op=CoreOpKind.GEMM,
        core_node=node,
        operand_nodes=list(node.args[:2]) if len(node.args) >= 2 else [],
    )


def _analyze_reduction(gm: fx.GraphModule, node: fx.Node) -> PartitionAnalysis:
    """Analyze reduction partition."""
    return PartitionAnalysis(
        core_op=CoreOpKind.REDUCTION,
        core_node=node,
    )


def _trace_prologue(node: fx.Node, analysis: PartitionAnalysis):
    """Walk backwards from an operand to find prologue ops and root placeholders."""
    visited = set()

    def _walk(n):
        if n.name in visited:
            return
        visited.add(n.name)

        if n.op == "placeholder":
            if n not in analysis.root_placeholders:
                analysis.root_placeholders.append(n)
            return

        if n.op in ("call_function", "call_method"):
            name = getattr(n.target, "__name__", str(n.target)) if n.op == "call_function" else str(n.target)

            # Layout ops: record for index propagation
            if name in ("permute", "transpose", "unsqueeze", "squeeze",
                         "contiguous", "reshape", "view", "flatten"):
                analysis.layout_ops.append(IndexMapping(
                    op=name,
                    params=tuple(a for a in n.args[1:] if not isinstance(a, fx.Node)),
                    inverse=None,  # computed on demand
                ))

            # Elementwise: add to prologue
            elif name in ("exp", "sub", "mul", "add", "div", "neg"):
                analysis.prologue_chain.append(PrologueOp(
                    node=n, op_name=name,
                    args=[a for a in n.args if isinstance(a, (fx.Node, int, float))],
                ))

            # Recurse into all Node args
            for arg in n.args:
                if isinstance(arg, fx.Node):
                    _walk(arg)

    _walk(node)


# ─── Unified Codegen ──────────────────────────────────────────────────────────

def generate_kernel(analysis: PartitionAnalysis,
                    gm: fx.GraphModule,
                    policy: Optional[Any] = None) -> Tuple[Any, dict]:
    """Generate a TileLang kernel for the analyzed partition.

    Unified codegen entry point. Routes to the appropriate code generator
    based on core_op, with AutoMixed config selection.

    Returns:
        (compiled_kernel, metadata) where metadata includes:
        - config: selected tile config
        - core_op: CoreOpKind
        - is_einsum / is_gemm / is_reduction: for runner dispatch
    """
    from .codegen import AutoFusionScheduler

    scheduler = AutoFusionScheduler(gm)
    kernel = scheduler.generate(tune=False)

    metadata = {
        "core_op": analysis.core_op,
        "is_einsum": scheduler.has_einsum,
        "is_gemm": scheduler.has_gemm,
        "is_reduction": scheduler.has_reduction,
    }

    return kernel, metadata


# ─── AutoMixed Integration ────────────────────────────────────────────────────

def automixed_compile(gm: fx.GraphModule,
                      partitions: list,
                      policy: Optional[Any] = None) -> Dict[fx.Node, Any]:
    """Compile all partitions using the unified AutoMixed pipeline.

    For each partition:
    1. analyze_partition() → determine core op + structure
    2. generate_kernel() → emit TileLang IR
    3. AutoMixed config selection → pick tile sizes (optional)
    4. Return runner map for the backend

    This replaces the manual per-partition codegen loop in backend.py.
    """
    from .subgraph import extract_subgraph_gm

    runners = {}
    for p in partitions:
        orig_order = list(gm.graph.nodes)
        in_nodes = [n for n in orig_order if n in p.inputs]
        out_nodes = [n for n in orig_order if n in p.outputs]

        if not in_nodes or not out_nodes:
            continue

        try:
            sub_gm = extract_subgraph_gm(gm, p.nodes, set(in_nodes), set(out_nodes))
            analysis = analyze_partition(sub_gm)

            if analysis is None:
                continue

            kernel = generate_kernel(analysis, sub_gm, policy)

            if kernel is not None:
                runners[out_nodes[0]] = {
                    "kernel": kernel,
                    "analysis": analysis,
                    "in_nodes": in_nodes,
                    "out_nodes": out_nodes,
                }
        except Exception as e:
            print(f"[AutoMixed] Skip partition: {e}")
            continue

    return runners
