# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.fx as fx
import math

def decompose_ops(gm: torch.fx.GraphModule):
    """
    Decompose composite operators (Softmax, LayerNorm, einsum, etc.) into elementary
    operations (Exp, Add, Mul, Reduce, bmm) to expose fine-grained fusion opportunities.
    """
    for node in list(gm.graph.nodes):
        if node.op == 'call_function':
            if node.target == torch.nn.functional.softmax or node.target == torch.softmax:
                _decompose_softmax(gm, node)
            target_name = getattr(node.target, "__name__", "") or str(node.target)
            # einsum: handled directly by codegen (ND-GEMM kernel),
            # NOT decomposed to bmm+permute (which adds 6 memory ops).
            # Set TILELANG_FRONTEND_DECOMPOSE_EINSUM=1 to force decomposition.
            if "einsum" in target_name:
                import os
                if os.environ.get("TILELANG_FRONTEND_DECOMPOSE_EINSUM", "0").strip() == "1":
                    try:
                        _decompose_einsum(gm, node)
                    except Exception as e:
                        print(f"[Decompose] einsum decomposition failed: {e}")

    gm.graph.lint()
    gm.recompile()
    return gm

def _decompose_einsum(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """Decompose einsum into reshape + bmm + reshape.

    Handles common patterns found in Mamba/linear attention:
    - 'bclhn,bclhp->bchpn': contract over l (chunk_size), batched over b,c,h
      → reshape to (B*C*H, L, N) @ (B*C*H, L, P) → (B*C*H, P, N) → reshape
    """
    eq = node.args[0] if isinstance(node.args[0], str) else None
    if eq is None:
        return

    operands = [a for a in node.args[1:] if isinstance(a, fx.Node)]
    if len(operands) != 2:
        return  # only handle 2-operand einsum for now

    A_node, B_node = operands

    def _get_shape(n):
        for key in ("tensor_meta", "val", "example_value"):
            m = n.meta.get(key)
            if m is not None and hasattr(m, "shape"):
                return list(m.shape)
        return None

    A_shape = _get_shape(A_node)
    B_shape = _get_shape(B_node)
    if A_shape is None or B_shape is None:
        return

    # Parse einsum equation
    parts = eq.replace(" ", "").split("->")
    if len(parts) != 2:
        return
    inputs_str, output_str = parts
    in_parts = inputs_str.split(",")
    if len(in_parts) != 2:
        return
    a_idx, b_idx = in_parts

    # Find contracted indices (in both inputs but not in output)
    contracted = set(a_idx) & set(b_idx) - set(output_str)
    batch_dims = set(a_idx) & set(b_idx) & set(output_str)
    a_free = set(a_idx) - contracted - batch_dims
    b_free = set(b_idx) - contracted - batch_dims

    if not contracted:
        return  # no contraction, not a matmul-like einsum

    A_shape = [int(s) for s in A_shape]
    B_shape = [int(s) for s in B_shape]

    # Build dimension maps
    a_dim_map = {c: i for i, c in enumerate(a_idx)}
    b_dim_map = {c: i for i, c in enumerate(b_idx)}

    # Compute sizes
    batch_size = 1
    for c in sorted(batch_dims):
        batch_size *= int(A_shape[a_dim_map[c]])
    contract_size = 1
    for c in sorted(contracted):
        contract_size *= int(A_shape[a_dim_map[c]])
    a_free_size = 1
    for c in sorted(a_free):
        a_free_size *= int(A_shape[a_dim_map[c]])
    b_free_size = 1
    for c in sorted(b_free):
        b_free_size *= int(B_shape[b_dim_map[c]])

    # Build permutation orders: batch_dims, contracted, free
    a_batch_pos = [a_dim_map[c] for c in sorted(batch_dims)]
    a_contract_pos = [a_dim_map[c] for c in sorted(contracted)]
    a_free_pos = [a_dim_map[c] for c in sorted(a_free)]
    a_perm = a_batch_pos + a_free_pos + a_contract_pos

    b_batch_pos = [b_dim_map[c] for c in sorted(batch_dims)]
    b_contract_pos = [b_dim_map[c] for c in sorted(contracted)]
    b_free_pos = [b_dim_map[c] for c in sorted(b_free)]
    b_perm = b_batch_pos + b_contract_pos + b_free_pos

    with gm.graph.inserting_before(node):
        # Permute A: batch, free_a, contracted → (B, M, K)
        a_perm_node = gm.graph.call_method("permute", (A_node,), {"dims": tuple(a_perm)})
        a_contig = gm.graph.call_method("contiguous", (a_perm_node,))
        a_reshape = gm.graph.call_method("reshape", (a_contig,),
                                          {"shape": (batch_size, a_free_size, contract_size)})

        # Permute B: batch, contracted, free_b → (B, K, N)
        b_perm_node = gm.graph.call_method("permute", (B_node,), {"dims": tuple(b_perm)})
        b_contig = gm.graph.call_method("contiguous", (b_perm_node,))
        b_reshape = gm.graph.call_method("reshape", (b_contig,),
                                          {"shape": (batch_size, contract_size, b_free_size)})

        # BMM: (B, M, K) @ (B, K, N) → (B, M, N) where M=free_a, N=free_b
        bmm_node = gm.graph.call_function(torch.bmm, args=(a_reshape, b_reshape))

        # Determine output dimension order from einsum equation
        # BMM result has dims: [batch_dim_0, ..., batch_dim_k, free_a_0, ..., free_b_0, ...]
        # But einsum output might have different ordering of free dims
        # Build the BMM result dim labels
        bmm_labels = (list(sorted(batch_dims))
                      + list(sorted(a_free))
                      + list(sorted(b_free)))

        # Check if we need to permute to match output order
        out_labels = list(output_str)
        if bmm_labels != out_labels:
            # Need to unflatten, permute, reflatten
            # First: unflatten BMM result to individual dims
            batch_dim_sizes = [int(A_shape[a_dim_map[c]]) for c in sorted(batch_dims)]
            a_free_sizes = [int(A_shape[a_dim_map[c]]) for c in sorted(a_free)]
            b_free_sizes = [int(B_shape[b_dim_map[c]]) for c in sorted(b_free)]
            unflatten_shape = tuple(batch_dim_sizes + a_free_sizes + b_free_sizes)
            unflat = gm.graph.call_method("reshape", (bmm_node,), {"shape": unflatten_shape})

            # Build permutation from bmm_labels order → output_str order
            label_to_pos = {c: i for i, c in enumerate(bmm_labels)}
            perm = [label_to_pos[c] for c in out_labels]
            perm_node = gm.graph.call_method("permute", (unflat,), {"dims": tuple(perm)})
            result = gm.graph.call_method("contiguous", (perm_node,))
        else:
            # Direct reshape to output shape
            out_shape_list = _get_shape(node)
            if out_shape_list is not None:
                out_shape = tuple(int(s) for s in out_shape_list)
                result = gm.graph.call_method("reshape", (bmm_node,), {"shape": out_shape})
            else:
                result = bmm_node

    node.replace_all_uses_with(result)
    gm.graph.erase_node(node)
    print(f"[Decompose] einsum '{eq}' → permute + bmm({batch_size}, {a_free_size}, {contract_size}, {b_free_size})")


def _decompose_softmax(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """
    Decompose Softmax(x, dim) -> Safe Softmax
        max_val = x.amax(dim, keepdim=True)
        x_safe = x - max_val
        exp_x = exp(x_safe)
        sum_exp = sum(exp_x, dim, keepdim=True)
        out = exp_x / sum_exp
    """
    with gm.graph.inserting_before(node):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', -1)
        
        # 1. Max Reduction (Safe Softmax)
        # Use torch.amax to get values directly (avoiding tuple return of torch.max)
        max_node = gm.graph.call_function(torch.amax, args=(input_node,), kwargs={'dim': dim, 'keepdim': True})
        
        # 2. Sub (x - max)
        sub_node = gm.graph.call_function(torch.sub, args=(input_node, max_node))
        
        # 3. Exp
        exp_node = gm.graph.call_function(torch.exp, args=(sub_node,))
        
        # 4. Sum Reduction
        sum_node = gm.graph.call_function(torch.sum, args=(exp_node,), kwargs={'dim': dim, 'keepdim': True})
        
        # 5. Div
        div_node = gm.graph.call_function(torch.div, args=(exp_node, sum_node))
        
        # Replace usages
        node.replace_all_uses_with(div_node)
        gm.graph.erase_node(node)
