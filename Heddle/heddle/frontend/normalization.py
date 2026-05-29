# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.fx as fx
import operator
from typing import Dict, Any

def canonicalize_ops(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Standardize the FX graph to improve fusion opportunities:
    1. Replace in-place ops (iadd, imul) with functional equivalents.
    2. Remove redundant identity-like ops (detach, clone, contiguous).
    3. (Future) Constant folding.
    """
    graph = gm.graph
    
    # Mapping of in-place / specialized targets to canonical ones
    _CANONICAL_MAP = {
        operator.iadd: operator.add,
        operator.imul: operator.mul,
        operator.isub: operator.sub,
        operator.itruediv: operator.truediv,
        # Common aten in-place versions
        torch.ops.aten.add_.Tensor: torch.ops.aten.add.Tensor,
        torch.ops.aten.mul_.Tensor: torch.ops.aten.mul.Tensor,
    }

    for node in list(graph.nodes):
        if node.op == "call_function":
            # 1. Standardize arithmetic
            if node.target in _CANONICAL_MAP:
                node.target = _CANONICAL_MAP[node.target]
            
            # 2. Handle redundant views/copies at fusion boundaries
            if node.target in (torch.ops.aten.detach.default, torch.ops.aten.clone.default):
                node.replace_all_uses_with(node.args[0])
                graph.erase_node(node)
                
        elif node.op == "call_method":
            # Normalize method names to strings for easier matching
            target_str = str(node.target)
            if target_str in ("detach", "clone"):
                node.replace_all_uses_with(node.args[0])
                graph.erase_node(node)

    graph.lint()
    gm.recompile()
    return gm

