# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Support / capability policy for TileLang frontend.

Goal:
- Centralize "what TileLang can codegen" vs "what must be boundary".
- Make partitioning / selection depend on capability analysis, not ad-hoc pattern matching.

This file is intentionally conservative: UNSUPPORTED by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Set, Tuple
import operator
import os

import torch
import torch.fx as fx


class Capability(Enum):
    SUPPORTED = auto()
    SOFT_BOUNDARY = auto()
    HARD_BOUNDARY = auto()
    UNSUPPORTED = auto()


def _target_key(target: Any) -> str:
    if target is None:
        return "None"
    try:
        if hasattr(target, "__name__"):
            return str(target.__name__)
    except Exception:
        pass
    try:
        if hasattr(target, "name") and callable(getattr(target, "name")):
            return str(target.name())
    except Exception:
        pass
    try:
        return str(target)
    except Exception:
        return repr(target)


@dataclass
class SupportPolicy:
    """
    Capability policy used by GraphPartitioner.

    Notes:
    - HARD_BOUNDARY: layout / aliasing / module attrs we don't model inside kernels.
    - SOFT_BOUNDARY: allowed at partition IO but not inside partition (e.g. dtype casts).
    """

    supported_call_function_targets: Set[Any]
    supported_call_method_targets: Set[str]
    supported_aten_substr: Tuple[str, ...]

    hard_boundary_call_method_targets: Set[str]
    soft_boundary_call_method_targets: Set[str]
    hard_boundary_aten_substr: Tuple[str, ...]
    soft_boundary_aten_substr: Tuple[str, ...]

    @staticmethod
    def default() -> "SupportPolicy":
        return SupportPolicy(
            supported_call_function_targets={
                operator.add,
                operator.mul,
                operator.sub,
                operator.truediv,
                torch.add,
                torch.mul,
                torch.sub,
                torch.div,
                torch.exp,
                torch.sigmoid,
                torch.relu,
                torch.nn.functional.silu,
                torch.nn.functional.gelu,
                torch.tanh,
                torch.clamp,
                torch.sum,
                torch.mean,
                torch.amax,
                torch.sqrt,
                torch.rsqrt,
                torch.pow,
                torch.log,
                torch.log2,
                operator.neg,
                torch.neg,
                torch.abs,
                torch.where,
                torch.matmul,
                torch.mm,
                torch.bmm,
                torch._C._nn.linear,
                torch.einsum,
                torch.functional.einsum,
            },
            supported_call_method_targets={
                "add",
                "mul",
                "sub",
                "div",
                "exp",
                "sigmoid",
                "silu",
                "relu",
                "tanh",
                "clamp",
                "sum",
                "mean",
                "amax",
                "sqrt",
                "rsqrt",
                "pow",
                "log",
                "log2",
                "neg",
                "abs",
                "to",
                "type",
                "float",
                "half",
                "bfloat16",
                "matmul",
                "mm",
                "bmm",
                "chunk",
                "split",
                "view",
                "reshape",
                "permute",
                "transpose",
                "contiguous",
                "flatten",
                "unsqueeze",
                "squeeze",
            },
            supported_aten_substr=(
                "aten.add",
                "aten.mul",
                "aten.sub",
                "aten.div",
                "aten.exp",
                "aten.sigmoid",
                "aten.silu",
                "aten.gelu",
                "aten.relu",
                "aten.tanh",
                "aten.clamp",
                "aten.sum",
                "aten.mean",
                "aten.amax",
                "aten.sqrt",
                "aten.rsqrt",
                "aten.pow",
                "aten.log",
                "aten.neg",
                "aten.abs",
                "aten.where",
                "aten.to",
                "aten.type_as",
                "aten.mm",
                "aten.bmm",
                "aten.matmul",
                "aten.linear",
                "aten.einsum",
                "einsum",
                "aten.chunk",
                "aten.split",
                "aten.reshape",
                "aten.view",
                "aten._unsafe_view",
                "aten.permute",
                "aten.transpose",
                "aten.t",
                "aten.contiguous",
            ),
            hard_boundary_call_method_targets={
                "expand",
                "repeat",
                "detach",
                "clone",
            },
            soft_boundary_call_method_targets={"chunk", "split"},
            hard_boundary_aten_substr=(
                "aten.clone",
            ),
            soft_boundary_aten_substr=(
                "aten.chunk",
                "aten.split",
            ),
        )

    def classify(self, n: fx.Node) -> Capability:
        # Non-call nodes are boundaries by definition for now.
        if n.op in ("get_attr", "call_module"):
            return Capability.HARD_BOUNDARY
        if n.op == "call_method":
            tgt = str(n.target)
            if tgt in self.hard_boundary_call_method_targets:
                return Capability.HARD_BOUNDARY
            if tgt in self.soft_boundary_call_method_targets:
                return Capability.SOFT_BOUNDARY
            if tgt in self.supported_call_method_targets:
                return Capability.SUPPORTED
            return Capability.UNSUPPORTED
        if n.op == "call_function":
            if n.target in self.supported_call_function_targets:
                return Capability.SUPPORTED
            k = _target_key(n.target)
            if any(s in k for s in self.hard_boundary_aten_substr):
                return Capability.HARD_BOUNDARY
            if any(s in k for s in self.soft_boundary_aten_substr):
                return Capability.SOFT_BOUNDARY
            if any(s in k for s in self.supported_aten_substr):
                return Capability.SUPPORTED
            return Capability.UNSUPPORTED
        return Capability.UNSUPPORTED

    def is_supported_inside_partition(self, n: fx.Node) -> bool:
        return self.classify(n) == Capability.SUPPORTED

    def is_any_boundary(self, n: fx.Node) -> bool:
        c = self.classify(n)
        return c in (Capability.SOFT_BOUNDARY, Capability.HARD_BOUNDARY)

    def is_hard_boundary(self, n: fx.Node) -> bool:
        return self.classify(n) == Capability.HARD_BOUNDARY

    def debug_dump(self, gm: fx.GraphModule, head: int = 50) -> None:
        if os.environ.get("TILELANG_FRONTEND_DUMP_CAPABILITY", "0").strip() != "1":
            return
        i = 0
        for n in gm.graph.nodes:
            if n.op in ("call_function", "call_method", "call_module", "get_attr"):
                print(f"[SupportPolicy] {n.name}: {n.op} {n.target} => {self.classify(n).name}")
                i += 1
                if i >= head:
                    break




