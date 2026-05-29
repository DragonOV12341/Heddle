# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import operator
import tilelang
import tilelang.language as T
from typing import List, Optional, Dict, Any, Set
import sys
import importlib.util
import tempfile
import os
import uuid

class Emitter:
    """Helper to manage code generation with automatic indentation."""
    def __init__(self):
        self.lines = []
        self._indent = 0

    def __call__(self, line):
        if line.strip() == "":
            self.lines.append("")
        else:
            self.lines.append("    " * self._indent + line)

    def indent(self):
        class Indenter:
            def __init__(self, e): self.e = e
            def __enter__(self): self.e._indent += 1
            def __exit__(self, *args): self.e._indent -= 1
        return Indenter(self)

    def get_code(self):
        return "\n".join(self.lines)

def _is_linear_node(node: torch.fx.Node) -> bool:
    """Check if an FX node is a linear/mm/matmul operation."""
    if node.op != "call_function":
        return False
    name = getattr(node.target, "__name__", "") or str(node.target)
    return "linear" in name or name in ("mm", "matmul", "bmm")


def _apply_inverse_layout_ops(layout_ops, idx_list):
    """Apply the inverse of a sequence of layout ops to an index list.

    layout_ops: list of (op_name, args, kwargs) recorded in forward order.
    idx_list: current index expressions (one per dim of the node's output shape).

    Returns new idx_list corresponding to the input tensor's indices.
    Each element is either a string expression or None (broadcast/unknown).
    """
    # Process ops in reverse (we want to map output indices → input indices)
    for op_name, args, kwargs in reversed(layout_ops):
        if op_name == "unsqueeze":
            dim_arg = args[0] if args else (kwargs.get("dim", -1))
            if isinstance(dim_arg, int):
                d = dim_arg if dim_arg >= 0 else len(idx_list) + dim_arg
                # Remove the dim we inserted (it was broadcast/size-1)
                idx_list = list(idx_list)
                if 0 <= d < len(idx_list):
                    idx_list.pop(d)
        elif op_name in ("squeeze",):
            # squeeze adds dims back in forward; inverse: don't know which, skip
            pass
        elif op_name == "permute":
            # permute(*dims): output[i0,...] = input[dims[0]-th, ...]
            # Forward: out[i] = in[dims[i]]
            # Inverse: in[j] = out[where dims==j]
            dims = [a for a in args if isinstance(a, int)]
            if len(dims) == len(idx_list):
                new_idx = [None] * len(dims)
                for out_pos, in_pos in enumerate(dims):
                    new_idx[in_pos] = idx_list[out_pos]
                idx_list = new_idx
        elif op_name == "transpose":
            if len(args) >= 2:
                d0, d1 = int(args[0]), int(args[1])
                n = len(idx_list)
                if d0 < 0: d0 += n
                if d1 < 0: d1 += n
                idx_list = list(idx_list)
                idx_list[d0], idx_list[d1] = idx_list[d1], idx_list[d0]
        elif op_name in ("contiguous", "flatten", "view", "reshape"):
            # contiguous: no index change
            # flatten/view/reshape: complex, skip (would need shape info)
            pass
    return idx_list


def _build_scale_expr(scale_node, idx_list, input_name_map, placeholder_shapes):
    """Recursively build an inline expression string for the scale chain.

    scale_node: FX node for the scale (could be placeholder, permute, unsqueeze, mul, exp, sub, etc.)
    idx_list: current index expressions for each dim of scale_node's output (list of str or None)
    input_name_map: {node.name: kernel_param_name} for all partition inputs
    placeholder_shapes: {node.name: shape_list} for all partition inputs

    Returns a string expression (Python/TileLang) that evaluates the scale value.
    Returns None if tracing fails.
    """
    if scale_node.op == "placeholder":
        # Build index into this input tensor
        shape = placeholder_shapes.get(scale_node.name, [])
        param_name = input_name_map.get(scale_node.name, f"inp_{scale_node.name}")
        idx_parts = []
        for i, (sz, ix) in enumerate(zip(shape, idx_list)):
            if ix is None:
                idx_parts.append("0")
            elif isinstance(sz, int) and sz == 1:
                idx_parts.append("0")  # broadcast dim
            else:
                idx_parts.append(str(ix))
        # Pad with 0 if fewer idx_list entries than shape dims (shouldn't happen)
        while len(idx_parts) < len(shape):
            idx_parts.append("0")
        return f"T.cast({param_name}[{', '.join(idx_parts)}], 'float32')"

    if scale_node.op not in ("call_function", "call_method"):
        return None

    target_name = (getattr(scale_node.target, "__name__", str(scale_node.target))
                   if scale_node.op == "call_function" else str(scale_node.target))

    # Layout ops: undo the layout transformation, recurse into the input tensor
    if target_name in ("permute", "transpose", "unsqueeze", "squeeze", "contiguous",
                        "flatten", "view", "reshape"):
        if not scale_node.args or not isinstance(scale_node.args[0], torch.fx.Node):
            return None
        layout_ops = [(target_name, scale_node.args[1:], scale_node.kwargs)]
        inner_idx = _apply_inverse_layout_ops(layout_ops, list(idx_list))
        return _build_scale_expr(scale_node.args[0], inner_idx, input_name_map, placeholder_shapes)

    # Elementwise unary: exp → exp2(x * log2e) for faster SFU path
    if target_name == "exp":
        if not scale_node.args or not isinstance(scale_node.args[0], torch.fx.Node):
            return None
        inner = _build_scale_expr(scale_node.args[0], list(idx_list), input_name_map, placeholder_shapes)
        if inner is None:
            return None
        return f"T.exp2(({inner}) * 1.4426950408889634)"

    if target_name == "neg":
        if not scale_node.args or not isinstance(scale_node.args[0], torch.fx.Node):
            return None
        inner = _build_scale_expr(scale_node.args[0], list(idx_list), input_name_map, placeholder_shapes)
        if inner is None:
            return None
        return f"(-{inner})"

    # Elementwise binary: sub, mul, add, div
    if target_name in ("sub", "mul", "add", "div"):
        if len(scale_node.args) < 2:
            return None
        args_exprs = []
        for a in scale_node.args[:2]:
            if isinstance(a, torch.fx.Node):
                ex = _build_scale_expr(a, list(idx_list), input_name_map, placeholder_shapes)
                if ex is None:
                    return None
                args_exprs.append(ex)
            else:
                args_exprs.append(f"T.cast({a!r}, 'float32')" if not isinstance(a, str) else a)
        ops_map = {"sub": "-", "mul": "*", "add": "+", "div": "/"}
        return f"({args_exprs[0]} {ops_map[target_name]} {args_exprs[1]})"

    return None


def _trace_index_expr(node, idx_vars, gm_nodes_set):
    """Trace backwards from a node through layout/elementwise ops.

    Returns (placeholder_name, index_expr_str, elementwise_expr_str) or None.
    For layout ops: transforms idx_vars.
    For elementwise ops: builds inline expression.

    Example: unsqueeze(permute(exp(sub(getitem(dA), dA)) * dt), 0,2,3,1)
    → reads dA[b,h,c,k*bK+ki] and dt[b,h,c,k*bK+ki], computes exp(dA_last-dA)*dt
    """
    # Walk backwards, building the expression
    if node.op == "placeholder":
        return {"type": "placeholder", "name": node.name, "node": node}

    if node.op != "call_function" and node.op != "call_method":
        return None

    target_name = getattr(node.target, "__name__", str(node.target)) if node.op == "call_function" else str(node.target)

    # Layout ops: just record the index transformation
    if target_name in ("permute", "transpose", "unsqueeze", "squeeze", "reshape", "view", "contiguous", "flatten"):
        if len(node.args) < 1 or not isinstance(node.args[0], torch.fx.Node):
            return None
        inner = _trace_index_expr(node.args[0], idx_vars, gm_nodes_set)
        if inner is None:
            return None
        # For permute/transpose/unsqueeze: the layout change is absorbed at the call site
        # We just pass through — the caller will handle the index mapping
        inner["layout_ops"] = inner.get("layout_ops", []) + [(target_name, node.args[1:], node.kwargs)]
        return inner

    # Elementwise ops: build expression
    if target_name in ("exp", "sub", "mul", "add", "div", "neg"):
        args_info = []
        for a in node.args:
            if isinstance(a, torch.fx.Node):
                info = _trace_index_expr(a, idx_vars, gm_nodes_set)
                if info is None:
                    return None
                args_info.append(info)
            else:
                args_info.append({"type": "const", "value": a})

        return {"type": "elementwise", "op": target_name, "args": args_info,
                "layout_ops": [], "node": node}

    # getitem (slice): record the slice
    if target_name in ("getitem", "__getitem__"):
        if len(node.args) >= 2 and isinstance(node.args[0], torch.fx.Node):
            inner = _trace_index_expr(node.args[0], idx_vars, gm_nodes_set)
            if inner is not None:
                inner["slice"] = node.args[1]
                return inner

    return None


def _is_einsum_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    name = getattr(node.target, "__name__", "") or str(node.target)
    return "einsum" in name


class AutoFusionScheduler:
    """
    Advanced Scheduler that generates tunable TileLang kernels.
    Supports: Elementwise, Reduction, GEMM + Epilogue, Einsum (ND-GEMM).
    """
    def __init__(self, gm: torch.fx.GraphModule):
        self.gm = gm
        self.input_nodes = []
        self.output_nodes = []
        self.has_reduction = False
        self.has_gemm = False
        self.has_einsum = False
        self.gemm_nodes = []
        self.einsum_nodes = []

    def generate(self, tune: bool = False):
        self._analyze_graph()
        if self.has_einsum:
            return self._generate_einsum_kernel()
        elif self.has_gemm:
            return self._generate_gemm_epilogue_kernel()
        elif self.has_reduction:
            return self._generate_reduction_kernel()
        else:
            search_space = [{'block_size': 256, 'thread_num': 128, 'vectorize': 2}]
            return self._codegen_elementwise(search_space[0])

    def _analyze_graph(self):
        self.input_nodes = []
        self.output_nodes = []
        self.has_reduction = False
        self.has_gemm = False
        self.gemm_nodes = []
        for node in self.gm.graph.nodes:
            if node.op == 'placeholder':
                self.input_nodes.append(node)
            elif node.op == 'output':
                outs = node.args[0]
                if isinstance(outs, (tuple, list)):
                    self.output_nodes = [n for n in outs if isinstance(n, torch.fx.Node)]
                elif isinstance(outs, torch.fx.Node):
                    self.output_nodes = [outs]
            elif node.op == 'call_function':
                if node.target in [torch.sum, torch.mean, torch.max, torch.amax]:
                    self.has_reduction = True
                if _is_linear_node(node):
                    self.has_gemm = True
                    self.gemm_nodes.append(node)
                if _is_einsum_node(node):
                    self.has_einsum = True
                    self.einsum_nodes.append(node)
                    pass
            elif node.op == 'call_method':
                if node.target in ['sum', 'mean', 'max', 'amax']:
                    self.has_reduction = True

    def _select_gemm_config(self, M, N, K):
        """Select tile config for GEMM kernel."""
        import os
        if os.environ.get("TILELANG_AUTOMIXED_CPSAT_RANK", "0").strip() == "1":
            try:
                from .automixed_decision import _cpsat_rank_candidates, Candidate
                candidates = []
                for bM in [64, 128]:
                    for bN in [64, 128]:
                        for ns in [2, 3]:
                            candidates.append(Candidate(block_M=min(bM, M), block_N=min(bN, N),
                                                         num_stages=ns, threads=128))
                ranked = _cpsat_rank_candidates(candidates, "gemm", {"M": M, "N": N, "K": K})
                if ranked:
                    b = ranked[0]
                    return {"block_M": b.block_M, "block_N": b.block_N, "block_K": 64, "num_stages": b.num_stages}
            except Exception:
                pass
        # Heuristic fallback
        bM = 128 if M >= 128 else max(16, (M // 16) * 16)
        bN = 128 if N >= 128 else max(16, (N // 16) * 16)
        bK = 64 if K >= 64 else max(16, (K // 16) * 16)
        ns = 3 if K >= 256 else 2
        return {"block_M": bM, "block_N": bN, "block_K": bK, "num_stages": ns}

    def _select_einsum_config(self, total_a_free, total_b_free, total_contract, total_batch, swap_tiles):
        """Select tile config for einsum kernel using AutoMixed pipeline.

        Priority:
        1. CP-SAT ranking (fast, no compilation)
        2. Heuristic fallback (if CP-SAT unavailable)
        """
        # Generate candidate configs
        candidates = []
        for bM_cand in [64, 128]:
            for bN_cand in [64, 128]:
                for bK_cand in [32, 64]:
                    for ns_cand in [2, 3, 4]:
                        if swap_tiles:
                            eff_M = min(bM_cand, total_b_free)
                            eff_N = min(bN_cand, total_a_free)
                        else:
                            eff_M = min(bM_cand, total_a_free)
                            eff_N = min(bN_cand, total_b_free)
                        eff_K = min(bK_cand, total_contract)
                        if eff_M < 16 or eff_N < 16 or eff_K < 16:
                            continue
                        eff_M = (eff_M // 16) * 16
                        eff_N = (eff_N // 16) * 16
                        eff_K = (eff_K // 16) * 16
                        candidates.append({
                            "block_M": eff_M, "block_N": eff_N,
                            "block_K": eff_K, "num_stages": ns_cand,
                        })

        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            key = (c["block_M"], c["block_N"], c["block_K"], c["num_stages"])
            if key not in seen:
                seen.add(key)
                unique.append(c)
        candidates = unique

        # Try AutoMixed CP-SAT ranking (skip if it takes too long)
        import os
        if os.environ.get("TILELANG_AUTOMIXED_CPSAT_RANK", "0").strip() == "1":
            try:
                from .automixed_decision import _cpsat_rank_candidates, Candidate
                cpsat_candidates = [
                    Candidate(block_M=c["block_M"], block_N=c["block_N"],
                              num_stages=c["num_stages"], threads=128)
                    for c in candidates
                ]
                ranked = _cpsat_rank_candidates(
                    cpsat_candidates, "einsum",
                    {"M": total_a_free, "N": total_b_free, "K": total_contract, "batch": total_batch})
                if ranked:
                    best = ranked[0]
                    for c in candidates:
                        if c["block_M"] == best.block_M and c["block_N"] == best.block_N and c["num_stages"] == best.num_stages:
                            return c
            except Exception:
                pass

        # Heuristic fallback (matches hand-written kernel config)
        if swap_tiles:
            bM = min(128, total_b_free) if total_b_free >= 16 else 16
            bN = min(128, total_a_free) if total_a_free >= 16 else 16
        else:
            bM = min(128, total_a_free) if total_a_free >= 16 else 16
            bN = min(128, total_b_free) if total_b_free >= 16 else 16
        bM = (bM // 16) * 16
        bN = (bN // 16) * 16
        bK = 64 if total_contract >= 64 else max(16, (total_contract // 16) * 16)
        ns = 4 if total_contract >= 256 else 2
        return {"block_M": bM, "block_N": bN, "block_K": bK, "num_stages": ns}

    def _collect_aux_placeholders(self, scale_node, data_node):
        """Walk the scale chain to find all placeholder inputs (auxiliary tensors like dA, dt).
        Returns list of placeholder names (excluding the data placeholder)."""
        result = []
        visited = set()
        data_name = data_node.name if data_node.op == "placeholder" else None

        def _walk(node):
            if node.name in visited:
                return
            visited.add(node.name)
            if node.op == "placeholder":
                if node.name != data_name:
                    result.append(node.name)
                return
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    _walk(arg)

        _walk(scale_node)
        return result

    def _generate_einsum_kernel(self):
        """Generate a direct ND-indexed GEMM kernel for einsum.

        Instead of decomposing to permute+bmm+reshape (6 memory ops),
        generates a single kernel that reads/writes ND tensors directly
        with computed multi-dimensional indices. Zero intermediate tensors.

        Also inlines prologue elementwise ops (e.g., exp, mul) into the K-loop.
        """
        kernel_id = str(uuid.uuid4()).replace("-", "_")[:8]
        kernel_name = f"auto_einsum_{kernel_id}"
        prim_func_name = f"kernel_{kernel_id}"

        en = self.einsum_nodes[0]
        eq = en.args[0] if isinstance(en.args[0], str) else None
        if eq is None:
            return self._codegen_elementwise({'block_size': 256, 'thread_num': 128, 'vectorize': 2})

        operand_nodes = [a for a in en.args[1:] if isinstance(a, torch.fx.Node)]
        if len(operand_nodes) != 2:
            return self._codegen_elementwise({'block_size': 256, 'thread_num': 128, 'vectorize': 2})

        A_node, B_node = operand_nodes

        def _get_shape(n):
            for key in ("tensor_meta", "val", "example_value"):
                m = n.meta.get(key)
                if m is not None and hasattr(m, "shape"):
                    return [int(s) for s in m.shape]
            return None

        A_shape = _get_shape(A_node)
        B_shape = _get_shape(B_node)
        out_shape = _get_shape(en)
        if not A_shape or not B_shape:
            return self._codegen_elementwise({'block_size': 256, 'thread_num': 128, 'vectorize': 2})

        in_dtype = self._tir_dtype_from_meta(self.input_nodes[0], default="float16")
        out_dtype = self._tir_dtype_from_meta(self.output_nodes[0], default="float16") if self.output_nodes else in_dtype

        # Parse einsum equation
        parts = eq.replace(" ", "").split("->")
        if len(parts) != 2:
            return self._codegen_elementwise({'block_size': 256, 'thread_num': 128, 'vectorize': 2})
        a_idx, b_idx = parts[0].split(",")
        out_idx = parts[1]

        contracted = set(a_idx) & set(b_idx) - set(out_idx)
        batch_dims = set(a_idx) & set(b_idx) & set(out_idx)
        a_free = set(a_idx) - contracted - batch_dims
        b_free = set(b_idx) - contracted - batch_dims

        a_dim_map = {c: i for i, c in enumerate(a_idx)}
        b_dim_map = {c: i for i, c in enumerate(b_idx)}

        # Compute dimension sizes
        batch_labels = [c for c in out_idx if c in batch_dims]
        a_free_labels = [c for c in a_idx if c in a_free]
        b_free_labels = [c for c in b_idx if c in b_free]
        contract_labels = [c for c in a_idx if c in contracted]

        batch_sizes = {c: A_shape[a_dim_map[c]] for c in batch_labels}
        contract_sizes = {c: A_shape[a_dim_map[c]] for c in contract_labels}
        a_free_sizes = {c: A_shape[a_dim_map[c]] for c in a_free_labels}
        b_free_sizes = {c: B_shape[b_dim_map[c]] for c in b_free_labels}

        total_batch = 1
        for s in batch_sizes.values(): total_batch *= s
        total_contract = 1
        for s in contract_sizes.values(): total_contract *= s
        total_a_free = 1
        for s in a_free_sizes.values(): total_a_free *= s
        total_b_free = 1
        for s in b_free_sizes.values(): total_b_free *= s

        # Check if output has b_free before a_free → swap tile assignment
        # so C_l matches output layout directly (no output transpose needed)
        out_free_order = [c for c in out_idx if c in a_free or c in b_free]
        cl_natural = list(sorted(a_free)) + list(sorted(b_free))
        swap_tiles = (out_free_order != cl_natural)

        # Tile config: use AutoMixed if available, else heuristic
        config = self._select_einsum_config(
            total_a_free, total_b_free, total_contract, total_batch, swap_tiles)
        bM, bN, bK, ns = config["block_M"], config["block_N"], config["block_K"], config["num_stages"]

        # Detect prologue: elementwise ops in the partition that feed into einsum.
        # If einsum operand B is produced by mul(x_r, scale), we inline:
        #   load x_r tile → load scale tile (possibly through layout ops) → x_r * scale → T.gemm
        prologue_info = None  # {op, data_node, scale_node, data_shape, scale_shape, operand}

        # Build input_name_map and placeholder_shapes for _build_scale_expr
        input_name_map = {}
        placeholder_shapes = {}
        for pn in self.input_nodes:
            input_name_map[pn.name] = f"inp_{pn.name}"
            s = _get_shape(pn)
            placeholder_shapes[pn.name] = s if s else []

        # Check if one of the einsum operands is a mul within the partition
        def _is_call_target_mul(node):
            if node.op == "call_function":
                return node.target in (operator.mul, torch.mul) or "mul" in getattr(node.target, "__name__", "")
            if node.op == "call_method":
                return node.target == "mul"
            return False

        for operand_label, operand_node, op_idx in [("B", B_node, b_idx), ("A", A_node, a_idx)]:
            if not _is_call_target_mul(operand_node):
                continue
            if len(operand_node.args) < 2:
                continue
            arg0, arg1 = operand_node.args[0], operand_node.args[1]
            if not (isinstance(arg0, torch.fx.Node) and isinstance(arg1, torch.fx.Node)):
                continue
            s0, s1 = _get_shape(arg0), _get_shape(arg1)
            if not s0 or not s1:
                continue
            # Larger tensor = data (same shape as einsum B operand), smaller = scale
            n0, n1 = 1, 1
            for x in s0: n0 *= x
            for x in s1: n1 *= x
            data_node = arg0 if n0 >= n1 else arg1
            scale_node = arg1 if n0 >= n1 else arg0
            data_shape = s0 if n0 >= n1 else s1
            scale_shape = s1 if n0 >= n1 else s0
            prologue_info = {"op": "mul", "operand": operand_label,
                             "data_node": data_node, "scale_node": scale_node,
                             "data_shape": data_shape, "scale_shape": scale_shape,
                             "operand_idx": op_idx}
            print(f"[Codegen] Prologue detected: {operand_label} = mul(data{data_shape}, scale{scale_shape})")
            break

        # ── Prologue aux analysis (must happen before kernel param generation) ──
        # Compute aux_placeholders and scalar_aux early so we can filter kernel params.
        all_aux = []
        aux_placeholders = []
        scalar_aux = []
        if prologue_info:
            all_aux = self._collect_aux_placeholders(
                prologue_info["scale_node"],
                prologue_info["data_node"])
            for aux_name in all_aux:
                aux_shape = placeholder_shapes.get(aux_name, [])
                if aux_shape and aux_shape[-1] >= bK:
                    aux_placeholders.append(aux_name)
                else:
                    scalar_aux.append(aux_name)
            if aux_placeholders and scalar_aux:
                print(f"[Codegen] Aux TMA: {aux_placeholders}, scalar (read from main aux): {scalar_aux}")

        # Build helper: multi-dim index from flat batch_idx and tile indices
        def _gen_nd_index(var_prefix, dim_labels, dim_map, shape, batch_var, tile_dim, tile_var, tile_block):
            """Generate index expressions for an ND tensor access.
            Returns a list of (dim_position, index_expr) tuples.
            """
            indices = []
            # Decompose batch_var into individual batch dim indices
            batch_decompose = []
            remaining = batch_var
            for i, c in enumerate(batch_labels):
                size = batch_sizes[c]
                if i < len(batch_labels) - 1:
                    prod = 1
                    for c2 in batch_labels[i+1:]: prod *= batch_sizes[c2]
                    batch_decompose.append((c, f"({remaining} // {prod})"))
                    remaining = f"({remaining} % {prod})"
                else:
                    batch_decompose.append((c, remaining))
            batch_idx_map = dict(batch_decompose)

            for dim_char in dim_labels:
                pos = dim_map[dim_char]
                if dim_char in batch_dims:
                    indices.append((pos, batch_idx_map[dim_char]))
                elif dim_char == tile_dim:
                    indices.append((pos, f"{tile_var}*{tile_block}:({tile_var}+1)*{tile_block}"))
                elif dim_char in contracted:
                    indices.append((pos, f"k*{bK}:(k+1)*{bK}"))
                else:
                    indices.append((pos, f"{tile_var}*{tile_block}:({tile_var}+1)*{tile_block}"))
            return indices

        e = Emitter()
        e("import tilelang")
        e("import tilelang.language as T")
        e("from tilelang.transform import PassConfigKey")
        e("")

        # Filter needed_inputs: remove scalar_aux (their values are read
        # from the main TMA-loaded aux tensor inside the kernel).
        scalar_aux_set = set(scalar_aux)
        needed_inputs = [pn for pn in self.input_nodes if pn.name not in scalar_aux_set]

        inp_params = []
        inp_shapes = []
        for pn in needed_inputs:
            s = _get_shape(pn)
            inp_params.append(pn.name)
            inp_shapes.append(s)

        out_idx_val = len(needed_inputs)
        e("@tilelang.jit(")
        e(f"    out_idx=[{out_idx_val}],")
        e("    pass_configs={")
        e("        PassConfigKey.TL_ENABLE_FAST_MATH: True,")
        e("        PassConfigKey.TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE: True,")
        e("        PassConfigKey.TL_HEDDLE_USE_PRECISE_LATENCY: True,")
        e("    },")
        e(")")
        e(f"def {kernel_name}():")
        with e.indent():
            e("@T.prim_func")

            # Parameters: only needed_inputs (scalar_aux filtered out)
            params = []
            for i, pn in enumerate(needed_inputs):
                s = inp_shapes[i]
                shape_str = ", ".join(str(x) for x in s) if s else "1"
                params.append(f"inp_{pn.name}: T.Tensor(({shape_str},), '{in_dtype}')")
            out_shape_str = ", ".join(str(x) for x in out_shape) if out_shape else "1"
            params.append(f"out: T.Tensor(({out_shape_str},), '{out_dtype}')")
            e(f"def {prim_func_name}({', '.join(params)}):")

            with e.indent():
                if swap_tiles:
                    M_tiles = f"T.ceildiv({total_b_free}, {bM})"
                    N_tiles = f"T.ceildiv({total_a_free}, {bN})"
                else:
                    M_tiles = f"T.ceildiv({total_a_free}, {bM})"
                    N_tiles = f"T.ceildiv({total_b_free}, {bN})"
                # Tile variable mapping: which grid dim tiles which free dim
                # Grid: put batch on first axis for SM scheduling (matches hand-written)
                # (batch, M_tiles, N_tiles) → bx=batch, by=M, bz=N
                # Grid layout: split batch into (largest_dim, rest_batch * tiles)
                # Matching hand-written pattern: (nheads, tiles, batch*nchunks)
                largest_batch_label = max(batch_labels, key=lambda c: batch_sizes[c])
                largest_batch_sz = batch_sizes[largest_batch_label]
                rest_batch_sz = total_batch // largest_batch_sz
                rest_batch_labels = [c for c in batch_labels if c != largest_batch_label]

                # Tile variables: since M_tiles and N_tiles are typically 1,
                # use constant 0 for tile access (no grid dim needed for tiles)
                a_tile_var, a_tile_sz = "0", (bN if swap_tiles else bM)
                b_tile_var, b_tile_sz = "0", (bM if swap_tiles else bN)

                e(f"with T.Kernel({largest_batch_sz}, {rest_batch_sz}, 1, threads=128) as (bx, by, bz):")
                with e.indent():
                    e(f"A_s = T.alloc_shared(({bK}, {a_tile_sz}), '{in_dtype}')")
                    e(f"B_s = T.alloc_shared(({bK}, {b_tile_sz}), '{in_dtype}')")
                    e(f"C_l = T.alloc_fragment(({bM}, {bN}), 'float32')")
                    if prologue_info:
                        if swap_tiles:
                            e(f"B_local_s = T.alloc_fragment(({bK}, {b_tile_sz}), '{in_dtype}')")
                            e(f"Bt_scaled_s = T.alloc_fragment(({b_tile_sz}, {bK}), '{in_dtype}')")
                        else:
                            e(f"A_local = T.alloc_fragment(({bK}, {a_tile_sz}), '{in_dtype}')")
                            e(f"At_scaled = T.alloc_fragment(({a_tile_sz}, {bK}), '{in_dtype}')")
                        e(f"S_frag = T.alloc_fragment(({bK},), 'float32')")
                        # TMA buffers for auxiliary inputs (pre-computed above)
                        for aux_name in aux_placeholders:
                            e(f"aux_{aux_name}_s = T.alloc_shared(({bK},), '{in_dtype}')")
                            e(f"aux_{aux_name}_l = T.alloc_fragment(({bK},), 'float32')")
                    # Swizzle the data operand's shared buffer (the one that gets transpose+scaled)
                    if swap_tiles:
                        e(f"T.annotate_layout({{B_s: tilelang.layout.make_swizzled_layout(B_s)}})")
                    else:
                        e(f"T.annotate_layout({{A_s: tilelang.layout.make_swizzled_layout(A_s)}})")
                    e(f"T.clear(C_l)")

                    # Decompose grid → batch dim indices
                    # bx = largest batch dim (direct), by = rest batch dims (decompose)
                    e(f"bi_{largest_batch_label} = bx")
                    if len(rest_batch_labels) == 0:
                        pass
                    elif len(rest_batch_labels) == 1:
                        e(f"bi_{rest_batch_labels[0]} = by")
                    else:
                        remaining = "by"
                        for i, c in enumerate(rest_batch_labels):
                            size = batch_sizes[c]
                            if i < len(rest_batch_labels) - 1:
                                prod = 1
                                for c2 in rest_batch_labels[i+1:]: prod *= batch_sizes[c2]
                                e(f"bi_{c} = {remaining} // {prod}")
                                remaining = f"({remaining} % {prod})"
                            else:
                                e(f"bi_{c} = {remaining}")

                    # Preload scalar auxiliary values (e.g., dA_last) before K-loop
                    # Only when there ARE TMA-loaded auxiliaries (dA, dt) — meaning
                    # the full prologue chain is captured in the partition.
                    if prologue_info and aux_placeholders and scalar_aux:
                        # dA_last is a slice of dA_cumsum — read from the TMA-loaded dA tensor
                        # using the last element of the contracted dimension
                        main_aux = aux_placeholders[0]  # dA_cumsum
                        main_shape = placeholder_shapes.get(main_aux, [])
                        idx_parts = []
                        used = set()
                        for dim_i, sz in enumerate(main_shape):
                            if sz == total_contract:
                                idx_parts.append(str(total_contract - 1))  # last element
                            elif sz == 1:
                                idx_parts.append("0")
                            else:
                                matched = False
                                for bc in batch_labels:
                                    if batch_sizes[bc] == sz and bc not in used:
                                        idx_parts.append(f"bi_{bc}")
                                        used.add(bc)
                                        matched = True
                                        break
                                if not matched:
                                    idx_parts.append("0")
                        e(f"dA_last_val = T.cast(inp_{main_aux}[{', '.join(idx_parts)}], 'float32')")

                    K_iters = f"T.ceildiv({total_contract}, {bK})"
                    e(f"for k in T.Pipelined({K_iters}, num_stages={ns}):")
                    with e.indent():
                        # Generate A load: T.copy(A[batch_dims, k_range, free_a_range], A_s)
                        # A has shape matching a_idx, need to build index string
                        a_input_name = f"inp_{A_node.name}" if A_node in needed_inputs else f"inp_{needed_inputs[0].name}"
                        if prologue_info and prologue_info["operand"] == "B":
                            # Prologue: B_node = mul(data, scale) or deeper chain.
                            # Three optimizations vs naive approach:
                            # 1. TMA pipeline: load dA/dt via T.copy→shared→fragment (not global element-wise)
                            # 2. exp → exp2(x * log2e): faster SFU path (in _build_scale_expr)
                            # 3. Scale+transpose in fragment → T.gemm(fragment, shared): no shared roundtrip
                            data_node = prologue_info["data_node"]
                            scale_node = prologue_info["scale_node"]
                            data_name = f"inp_{data_node.name}"

                            # Load data tile (x_r)
                            b_indices = []
                            for dim_char in b_idx:
                                if dim_char in batch_dims: b_indices.append(f"bi_{dim_char}")
                                elif dim_char in contracted: b_indices.append(f"k*{bK}:(k+1)*{bK}")
                                elif dim_char in b_free: b_indices.append(f"{b_tile_var}*{b_tile_sz}:({b_tile_var}+1)*{b_tile_sz}")
                                else: b_indices.append(":")
                            b_idx_str = ", ".join(b_indices)
                            e(f"T.copy({data_name}[{b_idx_str}], B_s)")

                            # TMA load auxiliary inputs (dA, dt) into shared → fragment
                            for aux_name in aux_placeholders:
                                aux_shape = placeholder_shapes.get(aux_name, [])
                                # Build T.copy index for this auxiliary tensor.
                                # Find the contracted dim (size == total_contract).
                                # Assign batch dims by matching size, deduplicate.
                                aux_idx_parts = []
                                used_bi = set()
                                contracted_dim_found = False
                                for dim_i, sz in enumerate(aux_shape):
                                    if sz == total_contract and not contracted_dim_found:
                                        aux_idx_parts.append(f"k*{bK}:(k+1)*{bK}")
                                        contracted_dim_found = True
                                    elif sz == 1:
                                        aux_idx_parts.append("0")
                                    else:
                                        # Match to a batch variable by size (avoid duplicates)
                                        matched = False
                                        for bc in batch_labels:
                                            if batch_sizes[bc] == sz and bc not in used_bi:
                                                aux_idx_parts.append(f"bi_{bc}")
                                                used_bi.add(bc)
                                                matched = True
                                                break
                                        if not matched:
                                            aux_idx_parts.append("0")
                                e(f"T.copy(inp_{aux_name}[{', '.join(aux_idx_parts)}], aux_{aux_name}_s)")
                                e(f"T.copy(aux_{aux_name}_s, aux_{aux_name}_l)")

                            # Load A_s (large tile) AFTER B_s and aux for better pipeline overlap
                            a_indices = []
                            for dim_char in a_idx:
                                if dim_char in batch_dims: a_indices.append(f"bi_{dim_char}")
                                elif dim_char in contracted: a_indices.append(f"k*{bK}:(k+1)*{bK}")
                                elif dim_char in a_free: a_indices.append(f"{a_tile_var}*{a_tile_sz}:({a_tile_var}+1)*{a_tile_sz}")
                                else: a_indices.append(":")
                            a_idx_str = ", ".join(a_indices)
                            # NOTE: A_s load moved AFTER scale computation (see below)
                            # to match hand-written pattern for better TMA/compute overlap.

                            # Build scale expression using fragment-local variables
                            def _make_scale_idx_list(ki_var="ki"):
                                idx = []
                                for dim_char in b_idx:
                                    if dim_char in batch_dims: idx.append(f"bi_{dim_char}")
                                    elif dim_char in contracted: idx.append(f"k*{bK}+{ki_var}")
                                    elif dim_char in b_free: idx.append("0")
                                    else: idx.append("0")
                                return idx

                            # Build the expression, but with aux tensors reading from fragments
                            # Create a modified input_name_map that points aux reads to fragment vars
                            tma_name_map = dict(input_name_map)
                            for aux_name in aux_placeholders:
                                tma_name_map[aux_name] = f"aux_{aux_name}_l"

                            # For TMA-loaded auxiliaries, override _build_scale_expr to use fragment index [ki]
                            # instead of the full ND index. We do this by creating a custom placeholder handler.
                            scale_idx_list = _make_scale_idx_list("ki")

                            def _build_tma_scale_expr(node, idx_list):
                                """Like _build_scale_expr but reads aux tensors from 1D fragments."""
                                if node.op == "placeholder":
                                    pn = node.name
                                    if pn in aux_placeholders:
                                        # TMA-loaded: read from 1D fragment
                                        return f"T.cast(aux_{pn}_l[ki], 'float32')"
                                    elif aux_placeholders and pn in scalar_aux:
                                        return "dA_last_val"
                                    else:
                                        return _build_scale_expr(node, idx_list, input_name_map, placeholder_shapes)
                                if node.op not in ("call_function", "call_method"):
                                    return None
                                tn = (getattr(node.target, "__name__", str(node.target))
                                      if node.op == "call_function" else str(node.target))
                                if tn in ("permute", "transpose", "unsqueeze", "squeeze",
                                          "contiguous", "flatten", "view", "reshape"):
                                    if not node.args or not isinstance(node.args[0], torch.fx.Node):
                                        return None
                                    layout_ops = [(tn, node.args[1:], node.kwargs)]
                                    inner_idx = _apply_inverse_layout_ops(layout_ops, list(idx_list))
                                    return _build_tma_scale_expr(node.args[0], inner_idx)
                                if tn == "exp":
                                    inner = _build_tma_scale_expr(node.args[0], list(idx_list))
                                    return f"T.exp2(({inner}) * 1.4426950408889634)" if inner else None
                                if tn == "neg":
                                    inner = _build_tma_scale_expr(node.args[0], list(idx_list))
                                    return f"(-{inner})" if inner else None
                                if tn in ("sub", "mul", "add", "div"):
                                    if len(node.args) < 2: return None
                                    exprs = []
                                    for a in node.args[:2]:
                                        if isinstance(a, torch.fx.Node):
                                            ex = _build_tma_scale_expr(a, list(idx_list))
                                            if ex is None: return None
                                            exprs.append(ex)
                                        else:
                                            exprs.append(str(a))
                                    ops = {"sub": "-", "mul": "*", "add": "+", "div": "/"}
                                    return f"({exprs[0]} {ops[tn]} {exprs[1]})"
                                return None

                            scale_expr = _build_tma_scale_expr(scale_node, scale_idx_list)

                            # Compute scale and apply with transpose:
                            # B_s(bK, bN) → fragment, transpose+scale → Bt_scaled(bM_or_bN, bK)
                            # Then T.gemm(Bt_scaled, A_s) where A_s(bK, bM)
                            # Bt_scaled needs to be (bN, bK) for gemm(bN,bK) × (bK,bM) → (bN,bM)
                            # But C_l is (bM, bN)... need to think about dim ordering.
                            # Simpler: keep existing pattern but use fragment for scale application
                            # Compute scale from TMA-loaded fragments
                            e(f"for ki in T.Parallel({bK}):")
                            with e.indent():
                                if scale_expr is not None:
                                    e(f"S_frag[ki] = {scale_expr}")
                                else:
                                    e(f"S_frag[ki] = T.cast(1.0, 'float32')")

                            # Key optimization: apply scale to A (not B), then transpose.
                            # Math: C[m,n] = sum_k A[k,m] * scale[k] * B[k,n]
                            #              = sum_k At_scaled[m,k] * B[k,n]
                            # where At_scaled[m,k] = A[k,m] * scale[k]
                            #
                            # This way B_s stays in shared (zero roundtrip), and
                            # T.gemm(At_scaled_fragment, B_s_shared) → same as hand-written pattern.
                            # Load A_s (large tile) AFTER scale — enables TMA/compute overlap
                            e(f"T.copy({a_input_name}[{a_idx_str}], A_s)")

                            if swap_tiles:
                                e(f"T.copy(B_s, B_local_s)")
                                e(f"for mi, ki in T.Parallel({b_tile_sz}, {bK}):")
                                with e.indent():
                                    e(f"Bt_scaled_s[mi, ki] = B_local_s[ki, mi] * T.cast(S_frag[ki], '{in_dtype}')")
                                e(f"T.gemm(Bt_scaled_s, A_s, C_l)")
                            else:
                                e(f"T.copy(A_s, A_local)")
                                e(f"for mi, ki in T.Parallel({a_tile_sz}, {bK}):")
                                with e.indent():
                                    e(f"At_scaled[mi, ki] = A_local[ki, mi] * T.cast(S_frag[ki], '{in_dtype}')")
                                e(f"T.gemm(At_scaled, B_s, C_l)")
                        else:
                            # No prologue: standard A + B load + GEMM
                            a_indices = []
                            for dim_char in a_idx:
                                if dim_char in batch_dims: a_indices.append(f"bi_{dim_char}")
                                elif dim_char in contracted: a_indices.append(f"k*{bK}:(k+1)*{bK}")
                                elif dim_char in a_free: a_indices.append(f"{a_tile_var}*{a_tile_sz}:({a_tile_var}+1)*{a_tile_sz}")
                                else: a_indices.append(":")
                            e(f"T.copy({a_input_name}[{', '.join(a_indices)}], A_s)")
                            b_input_name = (f"inp_{B_node.name}" if B_node in needed_inputs
                                            else f"inp_{needed_inputs[-1].name}")
                            b_indices = []
                            for dim_char in b_idx:
                                if dim_char in batch_dims: b_indices.append(f"bi_{dim_char}")
                                elif dim_char in contracted: b_indices.append(f"k*{bK}:(k+1)*{bK}")
                                elif dim_char in b_free: b_indices.append(f"{b_tile_var}*{b_tile_sz}:({b_tile_var}+1)*{b_tile_sz}")
                                else: b_indices.append(":")
                            b_idx_str = ", ".join(b_indices)
                            e(f"T.copy({b_input_name}[{b_idx_str}], B_s)")
                            e(f"T.gemm(A_s, B_s, C_l, transpose_A=True)")

                    # Write output: when swap_tiles, C_l is (bM=b_free, bN=a_free) matching output
                    need_transpose = not swap_tiles and (
                        [c for c in out_idx if c in a_free or c in b_free] !=
                        list(sorted(a_free)) + list(sorted(b_free))
                    )

                    if need_transpose:
                        # Output has b_free before a_free (e.g., 'bchpn' → p before n)
                        # Transpose C_l(bM,bN) to C_t(bN,bM) in shared, then write
                        e(f"O_s = T.alloc_shared(({bM}, {bN}), '{out_dtype}')")
                        e(f"T.copy(C_l, O_s)")
                        e(f"O_t = T.alloc_shared(({bN}, {bM}), '{out_dtype}')")
                        e(f"for mi, ni in T.Parallel({bM}, {bN}):")
                        with e.indent():
                            e(f"O_t[ni, mi] = O_s[mi, ni]")
                        # Build output index with b_free and a_free in output order
                        out_indices = []
                        for dim_char in out_idx:
                            if dim_char in batch_dims:
                                out_indices.append(f"bi_{dim_char}")
                            elif dim_char in b_free:
                                out_indices.append(f"by*{bN}:(by+1)*{bN}")
                            elif dim_char in a_free:
                                out_indices.append(f"bx*{bM}:(bx+1)*{bM}")
                            else:
                                out_indices.append(":")
                        e(f"T.copy(O_t, out[{', '.join(out_indices)}])")
                    else:
                        # C_l matches output layout directly
                        out_indices = []
                        for dim_char in out_idx:
                            if dim_char in batch_dims:
                                out_indices.append(f"bi_{dim_char}")
                            elif dim_char in a_free:
                                out_indices.append(f"{a_tile_var}*{a_tile_sz}:({a_tile_var}+1)*{a_tile_sz}")
                            elif dim_char in b_free:
                                out_indices.append(f"{b_tile_var}*{b_tile_sz}:({b_tile_var}+1)*{b_tile_sz}")
                            else:
                                out_indices.append(":")
                        e(f"acc_s = T.alloc_shared(({bM}, {bN}), '{out_dtype}')")
                        e(f"T.copy(C_l, acc_s)")
                        e(f"T.copy(acc_s, out[{', '.join(out_indices)}])")

            e(f"return {prim_func_name}")

        e(f"_raw_kernel = {kernel_name}()")
        e("")

        # Compute which argument indices to keep (drop scalar_aux inputs)
        if scalar_aux:
            keep_indices = [i for i, pn in enumerate(self.input_nodes) if pn.name not in scalar_aux_set]
            e(f"_keep_idx = {keep_indices}")
            e(f"def {kernel_name}_wrapper(*args):")
            with e.indent():
                e(f"args = [args[i].contiguous() if hasattr(args[i], 'contiguous') else args[i] for i in _keep_idx]")
                e(f"return _raw_kernel(*args)")
        else:
            e(f"def {kernel_name}_wrapper(*args):")
            with e.indent():
                e(f"args = [a.contiguous() if hasattr(a, 'contiguous') else a for a in args]")
                e(f"return _raw_kernel(*args)")
        e(f"{kernel_name} = {kernel_name}_wrapper")

        return self._compile_code(e.get_code().split("\n"), kernel_name)

    def _generate_gemm_epilogue_kernel(self):
        """Generate a fused GEMM + epilogue kernel.

        Architecture:
        - Wrapper: flatten ND input → 2D, transpose weights (N,K)→(K,N) once
        - Kernel: T.copy for ALL loads (TMA coalesced), T.gemm, inline epilogue
        - Same pattern as hand-written benchmarks in experiment/benchmarks/

        Handles:
        - Single GEMM + epilogue (linear → silu/relu/... → output)
        - Dual GEMM + epilogue (SwiGLU: silu(linear(x,W1)) * linear(x,W2))
        """
        kernel_id = str(uuid.uuid4()).replace("-", "_")[:8]
        kernel_name = f"auto_gemm_{kernel_id}"
        prim_func_name = f"kernel_{kernel_id}"

        gemm_infos = []
        for gn in self.gemm_nodes:
            gemm_infos.append({
                "node": gn, "input": gn.args[0], "weight": gn.args[1],
                "bias": gn.args[2] if len(gn.args) > 2 else None, "name": gn.name,
            })

        epilogue_nodes = []
        for node in self.gm.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            if _is_linear_node(node):
                continue
            if node.op in ("call_function", "call_method"):
                epilogue_nodes.append(node)

        in_dtype = self._tir_dtype_from_meta(self.input_nodes[0], default="float16")
        out_dtype = self._tir_dtype_from_meta(self.output_nodes[0], default="float16") if self.output_nodes else in_dtype

        gi = gemm_infos[0]
        in_meta = gi["input"].meta.get("tensor_meta") or gi["input"].meta.get("val")
        w_meta = gi["weight"].meta.get("tensor_meta") or gi["weight"].meta.get("val")
        if in_meta is None or w_meta is None:
            print(f"[Codegen] GEMM kernel: cannot determine shapes, falling back to elementwise")
            return self._codegen_elementwise({'block_size': 256, 'thread_num': 128, 'vectorize': 2})

        in_shape = list(in_meta.shape)
        K = int(in_shape[-1])
        M_total = 1
        for s in in_shape[:-1]:
            M_total *= int(s)

        n_gemms = len(gemm_infos)
        N_vals = []
        for i, gi in enumerate(gemm_infos):
            wm = gi["weight"].meta.get("tensor_meta") or w_meta
            N_vals.append(int(wm.shape[0]))
        N_out_val = int(self.output_nodes[0].meta.get("tensor_meta").shape[-1]) if self.output_nodes and self.output_nodes[0].meta.get("tensor_meta") else N_vals[0]

        # Tile config via AutoMixed
        config = self._select_gemm_config(M_total, max(N_vals), K)
        bM, bN, bK, ns = config["block_M"], config["block_N"], config["block_K"], config["num_stages"]

        # ── Generate kernel code ──
        # Key: weights are passed as TRANSPOSED (K, N), so T.copy works directly.
        e = Emitter()
        e("import tilelang")
        e("import tilelang.language as T")
        e("from tilelang.transform import PassConfigKey")
        e("")
        out_idx = 1 + n_gemms  # inp + n weights → output is last
        e("@tilelang.jit(")
        e(f"    out_idx=[{out_idx}],")
        e("    pass_configs={")
        e("        PassConfigKey.TL_ENABLE_FAST_MATH: True,")
        e("        PassConfigKey.TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE: True,")
        e("        PassConfigKey.TL_HEDDLE_USE_PRECISE_LATENCY: True,")
        e("    },")
        e(")")
        if n_gemms == 1:
            N_params = "N_0"
            e(f"def {kernel_name}(M, K, {N_params}):")
            with e.indent():
                e("@T.prim_func")
                params = [f"inp: T.Tensor((M, K), '{in_dtype}')",
                          f"wt_0: T.Tensor((K, N_0), '{in_dtype}')",
                          f"out: T.Tensor((M, {N_out_val}), '{out_dtype}')"]
                e(f"def {prim_func_name}({', '.join(params)}):")
                with e.indent():
                    self._emit_single_gemm_epilogue(e, epilogue_nodes, bM, bN, bK, ns, in_dtype, out_dtype, N_vals[0])
                e(f"return {prim_func_name}")
        else:
            # Dual GEMM: concatenate weights → single GEMM on (K, 2N), then split + epilogue in wrapper
            N_cat = sum(N_vals[:2])
            e(f"def {kernel_name}(M, K, N_cat):")
            with e.indent():
                e("@T.prim_func")
                params = [f"inp: T.Tensor((M, K), '{in_dtype}')",
                          f"wt_cat: T.Tensor((K, N_cat), '{in_dtype}')",
                          f"out: T.Tensor((M, N_cat), '{out_dtype}')"]
                e(f"def {prim_func_name}({', '.join(params)}):")
                with e.indent():
                    self._emit_dual_gemm_epilogue(e, epilogue_nodes, bM, bN, bK, ns, in_dtype, out_dtype, N_vals[:2])
                e(f"return {prim_func_name}")

        # Instantiate kernel and generate wrapper
        e("")
        e("import torch as _torch")
        e("_wt_cache = {}  # data_ptr → transposed/concatenated weight")
        e("")

        if n_gemms == 1:
            # Check if partition has prologue ops feeding INTO the linear
            # (these are not supported by GEMM codegen — only epilogue is supported)
            if epilogue_nodes:
                # Check if any epilogue node is actually a PROLOGUE (feeds into linear input)
                linear_input = gemm_infos[0]["input"]
                has_prologue = any(n == linear_input or
                                  (isinstance(linear_input, torch.fx.Node) and
                                   n in [a for a in linear_input.args if isinstance(a, torch.fx.Node)])
                                  for n in epilogue_nodes)
                if has_prologue:
                    print(f"[Codegen] GEMM partition has prologue ops — skipping (not supported)")
                    return self._codegen_elementwise({'block_size': 256, 'thread_num': 128, 'vectorize': 2})

            # The weight shape from metadata is (N, K). Find which arg matches.
            w_shape = tuple(w_meta.shape) if w_meta else None

            e(f"_raw_kernel = {kernel_name}({M_total}, {K}, {N_vals[0]})")
            e(f"_weight_shape = {w_shape}")
            e(f"def {kernel_name}_wrapper(*args):")
            with e.indent():
                # Find weight by exact shape match, data is the other one
                e("weight_arg = None; data_arg = None")
                e("for a in args:")
                with e.indent():
                    e("if _weight_shape and tuple(a.shape) == _weight_shape:")
                    with e.indent():
                        e("weight_arg = a")
                e("if weight_arg is None:")
                with e.indent():
                    # Fallback: smallest tensor is weight
                    e("weight_arg = min(args, key=lambda a: a.numel())")
                e("for a in args:")
                with e.indent():
                    e("if a is not weight_arg:")
                    with e.indent():
                        e("data_arg = a; break")
                e("orig_batch_shape = data_arg.shape[:-1]")
                e("M_flat = 1")
                e("for s in orig_batch_shape: M_flat *= s")
                e("flat_inp = data_arg.reshape(M_flat, -1).contiguous()")
                e("key = weight_arg.data_ptr()")
                e("if key not in _wt_cache: _wt_cache[key] = weight_arg.reshape(-1, weight_arg.shape[-1]).transpose(0,1).contiguous()")
                e("out_2d = _raw_kernel(flat_inp, _wt_cache[key])")
                e("return out_2d.reshape(*orig_batch_shape, -1)")
        else:
            # Dual GEMM: concatenate transposed weights → single GEMM → split + epilogue
            N_cat = sum(N_vals[:2])
            e(f"_raw_kernel = {kernel_name}({M_total}, {K}, {N_cat})")
            has_silu = any("silu" in getattr(ep.target, "__name__", str(ep.target)) for ep in epilogue_nodes)
            e(f"def {kernel_name}_wrapper(*args):")
            with e.indent():
                e("sorted_args = sorted(args, key=lambda a: -a.ndim)")
                e("data_arg = sorted_args[0]")
                e("weight_args = sorted_args[1:]")
                e("orig_batch_shape = data_arg.shape[:-1]")
                e("M_flat = 1")
                e("for s in orig_batch_shape: M_flat *= s")
                e("flat_inp = data_arg.reshape(M_flat, -1).contiguous()")
                e("# Concatenate transposed weights: (K, N0) cat (K, N1) → (K, N0+N1)")
                e("cache_key = tuple(w.data_ptr() for w in weight_args)")
                e("if cache_key not in _wt_cache:")
                with e.indent():
                    e("wt_list = [w.transpose(0,1).contiguous() for w in weight_args]")
                    e("_wt_cache[cache_key] = _torch.cat(wt_list, dim=1).contiguous()")
                e("wt_cat = _wt_cache[cache_key]")
                e("combined = _raw_kernel(flat_inp, wt_cat)")
                e(f"# Split and apply epilogue")
                e(f"g, u = combined[:, :{N_vals[0]}], combined[:, {N_vals[0]}:]")
                if has_silu:
                    e("result = _torch.nn.functional.silu(g) * u")
                else:
                    e("result = g * u")
                e("return result.reshape(*orig_batch_shape, -1)")
        e("")
        e(f"{kernel_name} = {kernel_name}_wrapper")

        return self._compile_code(e.get_code().split("\n"), kernel_name)

    def _emit_single_gemm_epilogue(self, e, epilogue_nodes, bM, bN, bK, ns, in_dtype, out_dtype, N):
        """Emit single GEMM + epilogue body — all loads via T.copy."""
        e(f"with T.Kernel(T.ceildiv(M, {bM}), T.ceildiv({N}, {bN}), threads=128) as (bx, by):")
        with e.indent():
            e(f"A_s = T.alloc_shared(({bM}, {bK}), '{in_dtype}')")
            e(f"B_s = T.alloc_shared(({bK}, {bN}), '{in_dtype}')")
            e(f"C_l = T.alloc_fragment(({bM}, {bN}), 'float32')")
            e(f"T.clear(C_l)")
            e(f"for k in T.Pipelined(T.ceildiv(K, {bK}), num_stages={ns}):")
            with e.indent():
                e(f"T.copy(inp[bx*{bM}:(bx+1)*{bM}, k*{bK}:(k+1)*{bK}], A_s)")
                e(f"T.copy(wt_0[k*{bK}:(k+1)*{bK}, by*{bN}:(by+1)*{bN}], B_s)")
                e(f"T.gemm(A_s, B_s, C_l)")
            self._emit_epilogue(e, epilogue_nodes, bM, bN, out_dtype, ["C_l"])

    def _emit_dual_gemm_epilogue(self, e, epilogue_nodes, bM, bN, bK, ns, in_dtype, out_dtype, N_vals):
        """Emit dual GEMM (SwiGLU) + epilogue.

        Strategy: single GEMM on concatenated weight [W0; W1] (K, 2N),
        then split + epilogue in registers. This is better than serial
        dual GEMM because the larger N dimension improves utilization.
        """
        N = N_vals[0]  # assume N0 == N1
        N2 = N * 2
        e(f"with T.Kernel(T.ceildiv(M, {bM}), T.ceildiv({N2}, {bN}), threads=128) as (bx, by):")
        with e.indent():
            e(f"A_s = T.alloc_shared(({bM}, {bK}), '{in_dtype}')")
            e(f"B_s = T.alloc_shared(({bK}, {bN}), '{in_dtype}')")
            e(f"C_l = T.alloc_fragment(({bM}, {bN}), 'float32')")
            e(f"T.clear(C_l)")
            e(f"for k in T.Pipelined(T.ceildiv(K, {bK}), num_stages={ns}):")
            with e.indent():
                e(f"T.copy(inp[bx*{bM}:(bx+1)*{bM}, k*{bK}:(k+1)*{bK}], A_s)")
                # Select from concatenated weight based on by index
                e(f"T.copy(wt_cat[k*{bK}:(k+1)*{bK}, by*{bN}:(by+1)*{bN}], B_s)")
                e(f"T.gemm(A_s, B_s, C_l)")
            e(f"T.copy(C_l, out[bx*{bM}:(bx+1)*{bM}, by*{bN}:(by+1)*{bN}])")

    def _emit_epilogue(self, e, epilogue_nodes, bM, bN, out_dtype, accum_names):
        """Emit fused epilogue: elementwise ops on GEMM accumulator(s), then write output."""
        if not epilogue_nodes and len(accum_names) == 1:
            e(f"T.copy({accum_names[0]}, out[bx*{bM}:(bx+1)*{bM}, by*{bN}:(by+1)*{bN}])")
            return

        e(f"for mi, ni in T.Parallel({bM}, {bN}):")
        with e.indent():
            if len(accum_names) == 1:
                e(f"val = {accum_names[0]}[mi, ni]")
                for ep in epilogue_nodes:
                    self._emit_epilogue_op(e, ep, "val")
                e(f"out[bx*{bM}+mi, by*{bN}+ni] = T.cast(val, '{out_dtype}')")
            elif len(accum_names) == 2:
                # Dual GEMM: determine which accumulator feeds which epilogue op
                # Common pattern: silu(C0) * C1 (SwiGLU)
                has_silu = any("silu" in (getattr(ep.target, "__name__", str(ep.target))) for ep in epilogue_nodes)
                if has_silu:
                    e(f"g = {accum_names[0]}[mi, ni]")
                    e(f"u = {accum_names[1]}[mi, ni]")
                    e(f"val = g * (1.0 / (1.0 + T.exp(-g))) * u")
                else:
                    # Generic: apply epilogue to first, multiply with second
                    e(f"v0 = {accum_names[0]}[mi, ni]")
                    e(f"v1 = {accum_names[1]}[mi, ni]")
                    e(f"val = v0 * v1")
                e(f"out[bx*{bM}+mi, by*{bN}+ni] = T.cast(val, '{out_dtype}')")

    def _emit_epilogue_op(self, e, ep_node, var_name):
        """Emit a single epilogue elementwise op inline."""
        ep_name = getattr(ep_node.target, "__name__", str(ep_node.target))
        if "silu" in ep_name:
            e(f"{var_name} = {var_name} * (1.0 / (1.0 + T.exp(-{var_name})))")
        elif "relu" in ep_name:
            e(f"{var_name} = T.max({var_name}, 0.0)")
        elif "gelu" in ep_name:
            e(f"{var_name} = {var_name} * 0.5 * (1.0 + T.tanh(0.7978845608 * ({var_name} + 0.044715 * T.pow({var_name}, 3))))")
        elif "sigmoid" in ep_name:
            e(f"{var_name} = 1.0 / (1.0 + T.exp(-{var_name}))")
        elif "tanh" in ep_name:
            e(f"{var_name} = T.tanh({var_name})")

    def _generate_reduction_kernel(self):
        # Unique kernel ID to avoid cache pollution
        kernel_id = str(uuid.uuid4()).replace("-", "_")[:8]
        kernel_name = f"auto_reduce_{kernel_id}"
        prim_func_name = f"kernel_{kernel_id}"
        # Use a more accurate log2(e) constant to reduce numerical drift vs PyTorch exp().
        LOG2E = "1.4426950408889634"
        
        in_dtype = self._tir_dtype_from_meta(self.input_nodes[0] if self.input_nodes else None, default="float16")
        out_dtypes = [self._tir_dtype_from_meta(node, default=in_dtype) for node in self.output_nodes]

        reduction_nodes = [n for n in self.gm.graph.nodes if n.op == 'call_function' and n.target in [torch.sum, torch.mean, torch.max, torch.amax]]
        reduction_nodes += [n for n in self.gm.graph.nodes if n.op == 'call_method' and n.target in ['sum', 'mean', 'max', 'amax']]

        def _is_max_node(n: torch.fx.Node) -> bool:
            return (n.op == "call_function" and n.target in [torch.max, torch.amax]) or (n.op == "call_method" and n.target in ["max", "amax"])

        def _is_sum_node(n: torch.fx.Node) -> bool:
            return (n.op == "call_function" and n.target in [torch.sum, torch.mean]) or (n.op == "call_method" and n.target in ["sum", "mean"])

        def _is_div_node(n: torch.fx.Node) -> bool:
            return (n.op == "call_function" and n.target in [torch.div, operator.truediv]) or (n.op == "call_method" and n.target in ["div", "true_divide"])

        max_nodes = [n for n in reduction_nodes if _is_max_node(n)]
        sum_nodes = [n for n in reduction_nodes if _is_sum_node(n)]
        softmax_like = (
            len(max_nodes) >= 1
            and len(sum_nodes) >= 1
            and len(self.output_nodes) == 1
            and _is_div_node(self.output_nodes[0])
        )

        e = Emitter()
        e("import tilelang")
        e("import tilelang.language as T")
        e("")
        e("@tilelang.jit")
        e(f"def {kernel_name}(rows, cols):")
        with e.indent():
            # Use 32 threads per row to stay within register budget.
            # NTHREADS is a Python constant here — it controls the generated code template.
            NTHREADS = 32
            # Compute nblk as a Python integer expression at JIT-specialization time
            # (when rows/cols are concrete Python ints).  Placing it here — before
            # @T.prim_func — ensures T.alloc_shared([nblk], ...) gets a static size.
            e(f"nblk = (cols + {NTHREADS - 1}) // {NTHREADS}")
            e("@T.prim_func")
            params = [f"in_{node.name}: T.Buffer((rows, cols), '{in_dtype}')" for node in self.input_nodes]
            params += [f"out_{node.name}: T.Buffer((rows, cols), '{out_dtypes[i]}')" for i, node in enumerate(self.output_nodes)]
            # Use a unique PrimFunc name to avoid TileLang kernel cache collisions.
            e(f"def {prim_func_name}({', '.join(params)}):")
            with e.indent():
                e(f"with T.Kernel(rows, threads={NTHREADS}) as row_idx:")
                with e.indent():
                    # Specialize softmax-like pattern: exp2 + online logsum (stable) rather than exp + 3-pass reduce.
                    if softmax_like:
                        max_node = max_nodes[0]
                        x_node = max_node.args[0] if len(max_node.args) >= 1 else None
                        if not isinstance(x_node, torch.fx.Node):
                            raise RuntimeError("Softmax-like pattern detected but max input is not a Node.")

                        e("# Softmax (stable, flash-style): block max + block exp2-sum + merge (m,s)")
                        # nblk is already defined as a Python int in the outer wrapper scope.
                        # Store per-element exp2(x - blk_max) (base2 domain) so we only compute the elementwise
                        # expression once, and avoid exp2 in the final pass (only per-block scaling).
                        e("exp_buf = T.alloc_shared([cols], 'float32')")
                        e("blk_max_arr = T.alloc_shared([nblk], 'float32')")
                        e("blk_sum_arr = T.alloc_shared([nblk], 'float32')")
                        e("m_sh = T.alloc_shared([1], 'float32')")
                        e("inv_s_sh = T.alloc_shared([1], 'float32')")

                        # Pass 1: per-block max and per-block sum(exp2((x-blk_max)*log2e)), plus exp_buf.
                        e("for k in T.Pipelined(nblk, num_stages=0):")
                        with e.indent():
                            e(f"accv_x = T.alloc_fragment(({NTHREADS},), 'float32')")
                            e("T.fill(accv_x, -T.infinity('float32'))")
                            e(f"for tx in T.Parallel({NTHREADS}):")
                            with e.indent():
                                e(f"k_idx = k * {NTHREADS} + tx")
                                e("if k_idx < cols:")
                                with e.indent():
                                    body = self._generate_body_until(max_node, "row_idx", "k_idx", override={})
                                    for line in body:
                                        e(line)
                                    e(f"accv_x[tx] = val_{x_node.name}")
                            e("blk_max = T.alloc_fragment((1,), 'float32')")
                            e("T.reduce_max(accv_x, blk_max, dim=0, clear=True)")
                            e(f"for tx in T.Parallel({NTHREADS}):")
                            with e.indent():
                                e("if tx == 0:")
                                with e.indent():
                                    e("blk_max_arr[k] = blk_max[0]")

                            e(f"accv_e = T.alloc_fragment(({NTHREADS},), 'float32')")
                            e("T.fill(accv_e, 0.0)")
                            e(f"for tx in T.Parallel({NTHREADS}):")
                            with e.indent():
                                e(f"k_idx = k * {NTHREADS} + tx")
                                e("if k_idx < cols:")
                                with e.indent():
                                    # Guard: if blk_max is -inf, the whole block is masked => contribution is 0.
                                    # Use a single expression (no if-frame) to satisfy TileLang V2 scoping.
                                    e(f"ev_expr = T.if_then_else(blk_max[0] == -T.infinity('float32'), 0.0, T.exp2((accv_x[tx] - blk_max[0]) * {LOG2E}))")
                                    e("accv_e[tx] = ev_expr")
                                    e("exp_buf[k_idx] = ev_expr")
                            e("blk_sum = T.alloc_fragment((1,), 'float32')")
                            e("T.reduce_sum(accv_e, blk_sum, dim=0, clear=True)")
                            e(f"for tx in T.Parallel({NTHREADS}):")
                            with e.indent():
                                e("if tx == 0:")
                                with e.indent():
                                    # Guard: if blk_max is -inf, force blk_sum to 0 (avoid -inf - -inf in merge).
                                    e("blk_sum_arr[k] = T.if_then_else(blk_max[0] == -T.infinity('float32'), 0.0, blk_sum[0])")

                        # Ensure exp_buf / blk_* are fully written before merge + final pass.
                        e("T.sync_threads()")

                        # Merge blocks: all threads run the same serial reduction (redundant but correct).
                        # Using T.Parallel(N) with if tx==0 causes register-layout failures in TileLang
                        # because T.alloc_fragment inside a conditional branch of T.Parallel is unsupported.
                        e("m_loc = T.alloc_fragment((1,), 'float32')")
                        e("s_loc = T.alloc_fragment((1,), 'float32')")
                        e("m_tmp = T.alloc_fragment((1,), 'float32')")
                        e("m_loc[0] = -T.infinity('float32')")
                        e("s_loc[0] = 0.0")
                        e("for kb in T.serial(nblk):")
                        with e.indent():
                            e("m_tmp[0] = T.max(m_loc[0], blk_max_arr[kb])")
                            e(
                                f"s_loc[0] = (s_loc[0] * T.if_then_else((m_loc[0] == -T.infinity('float32')) & (m_tmp[0] == -T.infinity('float32')), 1.0, T.exp2((m_loc[0] - m_tmp[0]) * {LOG2E})))"
                            )
                            e(
                                f"s_loc[0] = s_loc[0] + T.if_then_else(blk_max_arr[kb] == -T.infinity('float32'), 0.0, blk_sum_arr[kb] * T.exp2((blk_max_arr[kb] - m_tmp[0]) * {LOG2E}))"
                            )
                            e("m_loc[0] = m_tmp[0]")
                        e("m_sh[0] = m_loc[0]")
                        e("inv_s_sh[0] = T.if_then_else(s_loc[0] == 0.0, 0.0, 1.0 / s_loc[0])")

                        e("T.sync_threads()")

                        # Final Pass: output = exp2((x - m)*log2e)/sum = exp_buf * exp2((blk_max - m)*log2e) / s
                        e("# Final Pass: Output")
                        e("for k in T.Pipelined(nblk, num_stages=0):")
                        with e.indent():
                            e(f"scale_blk = T.exp2((blk_max_arr[k] - m_sh[0]) * {LOG2E}) * inv_s_sh[0]")
                            e(f"for tx in T.Parallel({NTHREADS}):")
                            with e.indent():
                                e(f"k_idx = k * {NTHREADS} + tx")
                                e("if k_idx < cols:")
                                with e.indent():
                                    e(f"out_{self.output_nodes[0].name}[row_idx, k_idx] = T.cast(exp_buf[k_idx] * scale_blk, '{out_dtypes[0]}')")
                    else:
                        reduction_vars = {}

                        for i, r_node in enumerate(reduction_nodes):
                            e(f"# Reduction Pass {i+1}: {r_node.name}")
                            is_max = r_node.target in [torch.max, torch.amax]
                            neutral = "-T.infinity('float32')" if is_max else "0.0"
                            # Per-thread accumulator to avoid data races inside T.Parallel.
                            e(f"accv_{i} = T.alloc_fragment(({NTHREADS},), 'float32')")
                            e(f"T.fill(accv_{i}, {neutral})")

                            # Use T.Pipelined for robust symbolic loops in TileLang V2
                            e(f"for k in T.Pipelined(T.ceildiv(cols, {NTHREADS}), num_stages=0):")
                            with e.indent():
                                e(f"for tx in T.Parallel({NTHREADS}):")
                                with e.indent():
                                    e(f"k_idx = k * {NTHREADS} + tx")
                                    e("if k_idx < cols:")
                                    with e.indent():
                                        body = self._generate_body_until(r_node, "row_idx", "k_idx", override=reduction_vars)
                                        for line in body: e(line)
                                        input_var = f"val_{r_node.args[0].name}"
                                        if is_max:
                                            e(f"accv_{i}[tx] = T.max(accv_{i}[tx], {input_var})")
                                        else:
                                            e(f"accv_{i}[tx] = accv_{i}[tx] + {input_var}")

                            # Perform cross-thread reduction AFTER the Parallel loop
                            e(f"acc_{i} = T.alloc_fragment((1,), 'float32')")
                            if is_max:
                                e(f"T.reduce_max(accv_{i}, acc_{i}, dim=0, clear=True)")
                            else:
                                e(f"T.reduce_sum(accv_{i}, acc_{i}, dim=0, clear=True)")

                            if r_node.target == torch.mean:
                                e(f"acc_{i}[0] = acc_{i}[0] / T.cast(cols, 'float32')")

                            res_var = f"reduce_res_{i}"
                            e(f"{res_var} = acc_{i}[0]")
                            # Keep reduction results in fp32 to avoid numerical issues (e.g., sum underflow to 0).
                            reduction_vars[r_node] = f"{res_var}"

                        e("# Final Pass: Output")
                        e(f"for k in T.Pipelined(T.ceildiv(cols, {NTHREADS}), num_stages=0):")
                        with e.indent():
                            e(f"for tx in T.Parallel({NTHREADS}):")
                            with e.indent():
                                e(f"k_idx = k * {NTHREADS} + tx")
                                e("if k_idx < cols:")
                                with e.indent():
                                    body_final = self._generate_body_full("row_idx", "k_idx", override=reduction_vars)
                                    for line in body_final: e(line)
                                    for i, node in enumerate(self.output_nodes):
                                        e(f"out_{node.name}[row_idx, k_idx] = T.cast(val_{node.name}, '{out_dtypes[i]}')")

            # Important: tilelang.jit expects the wrapper to return a PrimFunc.
            e(f"return {prim_func_name}")

        return self._compile_code(e.get_code().split("\n"), kernel_name)

    def _codegen_elementwise(self, config: Dict[str, Any]):
        kernel_id = str(uuid.uuid4()).replace("-", "_")[:8]
        kernel_name = f"auto_elem_{kernel_id}"
        prim_func_name = f"kernel_{kernel_id}"
        thread_num = config['thread_num']
        vectorize = config.get('vectorize', 1)
        
        in_dtype = self._tir_dtype_from_meta(self.input_nodes[0] if self.input_nodes else None, default="float16")
        out_dtypes = [self._tir_dtype_from_meta(node, default=in_dtype) for node in self.output_nodes]

        e = Emitter()
        e("import tilelang")
        e("import tilelang.language as T")
        e("")
        e("@tilelang.jit")
        e(f"def {kernel_name}(total_elements):")
        with e.indent():
            e("@T.prim_func")
            params = [f"in_{node.name}: T.Buffer((total_elements,), '{in_dtype}')" for node in self.input_nodes]
            params += [f"out_{node.name}: T.Buffer((total_elements,), '{out_dtypes[i]}')" for i, node in enumerate(self.output_nodes)]
            # Use a unique PrimFunc name to avoid TileLang kernel cache collisions.
            e(f"def {prim_func_name}({', '.join(params)}):")
            with e.indent():
                elems_per_block = thread_num * vectorize
                e(f"with T.Kernel(T.ceildiv(total_elements, {elems_per_block}), threads={thread_num}) as bx:")
                with e.indent():
                    e(f"for i in T.Parallel({thread_num}):")
                    with e.indent():
                        e("idx = bx * " + str(elems_per_block) + " + i * " + str(vectorize))
                        for v in range(vectorize):
                            e(f"v_idx = idx + {v}")
                            e("if v_idx < total_elements:")
                            with e.indent():
                                body_code = self._generate_body_full_flat("v_idx")
                                for line in body_code: e(line)
                                for i, node in enumerate(self.output_nodes):
                                    e(f"out_{node.name}[v_idx] = T.cast(val_{node.name}, '{out_dtypes[i]}')")

            # Important: tilelang.jit expects the wrapper to return a PrimFunc.
            e(f"return {prim_func_name}")
        
        return self._compile_code(e.get_code().split("\n"), kernel_name)

    def _tir_dtype_from_meta(self, node, default="float16"):
        if node is None: return default
        try:
            tm = node.meta.get("tensor_meta", None)
            if tm is not None and hasattr(tm, "dtype"):
                dt = tm.dtype
                if dt == torch.float16: return "float16"
                if dt == torch.float32: return "float32"
                if dt == torch.bfloat16: return "bfloat16"
        except Exception: pass
        return default

    def _compile_code(self, code_lines, kernel_name):
        full_code = "\n".join(code_lines)
        print(f"[Codegen] Generated Kernel:\n{full_code}")
        
        fd, path = tempfile.mkstemp(suffix=".py", prefix=kernel_name + "_")
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(full_code)
            spec = importlib.util.spec_from_file_location(kernel_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[kernel_name] = module
            spec.loader.exec_module(module)
            return getattr(module, kernel_name)
        except Exception as e:
            raise e

    def _generate_body_full_flat(self, idx):
        return self._generate_body_full(idx, None)

    def _generate_body_full(self, row_idx, col_idx, override=None):
        if override is None: override = {}
        lines = []
        for node in self.input_nodes:
            if col_idx:
                lines.append(f"val_{node.name} = T.cast(in_{node.name}[{row_idx}, {col_idx}], 'float32')")
            else:
                lines.append(f"val_{node.name} = T.cast(in_{node.name}[{row_idx}], 'float32')")
        for node in self.gm.graph.nodes:
            if node in override:
                lines.append(f"val_{node.name} = {override[node]}")
                continue
            if node.op == 'call_function':
                if node.target in [torch.sum, torch.mean, torch.max, torch.amax]: continue 
                args_str = [f"val_{arg.name}" if isinstance(arg, torch.fx.Node) else str(arg) for arg in node.args]
                op_str = self._map_op(node.target, args_str)
                lines.append(f"val_{node.name} = {op_str}")
            elif node.op == 'call_method':
                if node.target in ['sum', 'mean', 'max', 'amax']: continue
                args_str = [f"val_{arg.name}" if isinstance(arg, torch.fx.Node) else str(arg) for arg in node.args]
                op_str = self._map_op(node.target, args_str)
                lines.append(f"val_{node.name} = {op_str}")
        return lines

    def _generate_body_until(self, target_node, row_idx, col_idx, override=None):
        if override is None: override = {}
        lines = []
        for node in self.input_nodes:
            lines.append(f"val_{node.name} = T.cast(in_{node.name}[{row_idx}, {col_idx}], 'float32')")
        for node in self.gm.graph.nodes:
            if node == target_node: break
            if node in override:
                lines.append(f"val_{node.name} = {override[node]}")
                continue
            if node.op == 'call_function':
                if node.target in [torch.sum, torch.mean, torch.max, torch.amax]: continue 
                args_str = [f"val_{arg.name}" if isinstance(arg, torch.fx.Node) else str(arg) for arg in node.args]
                op_str = self._map_op(node.target, args_str)
                lines.append(f"val_{node.name} = {op_str}")
            elif node.op == 'call_method':
                if node.target in ['sum', 'mean', 'max', 'amax']: continue
                args_str = [f"val_{arg.name}" if isinstance(arg, torch.fx.Node) else str(arg) for arg in node.args]
                op_str = self._map_op(node.target, args_str)
                lines.append(f"val_{node.name} = {op_str}")
        return lines

    def _map_op(self, target, args_str):
        if target == operator.add or target == torch.add or target == 'add': return f"{args_str[0]} + {args_str[1]}"
        elif target == operator.mul or target == torch.mul or target == 'mul': return f"{args_str[0]} * {args_str[1]}"
        elif target == torch.sub or target == 'sub': return f"{args_str[0]} - {args_str[1]}"
        elif target == torch.div or target == operator.truediv or target == 'div': return f"{args_str[0]} / {args_str[1]}"
        elif target == torch.exp or target == 'exp': return f"T.exp({args_str[0]})"
        elif target == torch.sigmoid or target == 'sigmoid': return f"1.0 / (1.0 + T.exp(-{args_str[0]}))"
        elif target == torch.relu or target == 'relu': return f"T.max({args_str[0]}, 0.0)"
        elif target == torch.nn.functional.silu or target == 'silu' or 'silu' in str(target):
            return f"({args_str[0]} * (1.0 / (1.0 + T.exp(-{args_str[0]}))))"
        elif target == torch.nn.functional.gelu or target == 'gelu' or 'gelu' in str(target):
            return f"({args_str[0]} * 0.5 * (1.0 + T.tanh(0.7978845608 * ({args_str[0]} + 0.044715 * T.pow({args_str[0]}, 3)))))"
        elif target == torch.tanh or target == 'tanh': return f"T.tanh({args_str[0]})"
        elif target == torch.log or target == 'log': return f"T.log({args_str[0]})"
        elif target == torch.log2 or target == 'log2': return f"T.log2({args_str[0]})"
        elif target in (torch.sqrt, 'sqrt'): return f"T.sqrt({args_str[0]})"
        elif target in (torch.rsqrt, 'rsqrt'): return f"1.0 / T.sqrt({args_str[0]})"
        elif target in (operator.neg, torch.neg, 'neg'): return f"-{args_str[0]}"
        elif target in (torch.abs, 'abs'): return f"T.abs({args_str[0]})"
        elif target == torch.where: return f"T.if_then_else({args_str[0]}, {args_str[1]}, {args_str[2]})"
        elif target in (torch.pow, 'pow'): return f"T.pow({args_str[0]}, {args_str[1]})"
        elif any(x in str(target) for x in ("to", "type", "float", "half", "bfloat16")):
            t_str = str(target)
            if "float32" in t_str or "float" in t_str: dtype = "float32"
            elif "float16" in t_str or "half" in t_str: dtype = "float16"
            elif "bfloat16" in t_str: dtype = "bfloat16"
            else: dtype = "float32"
            return f"T.cast({args_str[0]}, '{dtype}')"
        elif target in (torch.matmul, torch.mm, torch.bmm):
            raise NotImplementedError("MatMul must be handled by a specialized template.")
        raise NotImplementedError(f"Unsupported op for TileLang ElementwiseCodegen: {target!r}")

ElementwiseCodegen = AutoFusionScheduler
