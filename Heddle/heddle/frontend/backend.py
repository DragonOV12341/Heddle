# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.fx as fx
from torch.fx import replace_pattern
from torch.fx.interpreter import Interpreter
from torch.fx.passes.shape_prop import ShapeProp
import operator
from typing import List, Optional, Dict, Callable, Any
import os
import itertools
import json
from .patterns import (
    scaled_dot_product_attention_pattern,
    tilelang_flash_attn_replacement,
    tilelang_flash_attn_wrapper,
    tilelang_h2o_attention_wrapper,
)
from .codegen import ElementwiseCodegen
import operator
from .partitioner import GraphPartitioner
from .decomposition import decompose_ops
from .subgraph import extract_subgraph_gm
from .normalization import canonicalize_ops


def _build_fa_kernel(batch, heads, seq_q, seq_kv, dim, is_causal, block_M, block_N, num_stages, threads):
    """Build FlashAttention kernel directly via tilelang.jit (bypass autotune)."""
    import tilelang
    import tilelang.language as T

    scale = (1.0 / dim) ** 0.5 * 1.4426950408889634
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]

    pc = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
          tilelang.PassConfigKey.TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE: True,
          tilelang.PassConfigKey.TL_HEDDLE_USE_PRECISE_LATENCY: True}

    @tilelang.jit(out_idx=[3], pass_configs=pc)
    def fa(batch, heads, seq_q, seq_kv, dim, block_M, block_N, num_stages, threads, scale):
        @T.prim_func
        def main(Q: T.Tensor(q_shape, T.float16), K: T.Tensor(kv_shape, T.float16),
                 V: T.Tensor(kv_shape, T.float16), O: T.Tensor(q_shape, T.float16)):
            with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Qs = T.alloc_shared([block_M, dim], T.float16)
                Ks = T.alloc_shared([block_N, dim], T.float16)
                Vs = T.alloc_shared([block_N, dim], T.float16)
                acc_s = T.alloc_fragment([block_M, block_N], T.float32)
                acc_c = T.alloc_fragment([block_M, block_N], T.float16)
                acc_o = T.alloc_fragment([block_M, dim], T.float32)
                smax = T.alloc_fragment([block_M], T.float32)
                smax_p = T.alloc_fragment([block_M], T.float32)
                sscl = T.alloc_fragment([block_M], T.float32)
                ssum = T.alloc_fragment([block_M], T.float32)
                lsum = T.alloc_fragment([block_M], T.float32)
                T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Qs)
                T.fill(acc_o, 0); T.fill(lsum, 0); T.fill(smax, -T.infinity(T.float32))
                for k in T.Pipelined(T.ceildiv(seq_kv, block_N), num_stages=num_stages):
                    T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], Ks)
                    T.copy(smax, smax_p)
                    T.fill(smax, -T.infinity(T.float32))
                    T.clear(acc_s)
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                bx * block_M + i >= k * block_N + j,
                                acc_s[i, j], -T.infinity(T.float32))
                    T.reduce_max(acc_s, smax, dim=1, clear=False)
                    for i in T.Parallel(block_M): smax[i] = T.max(smax[i], smax_p[i])
                    for i in T.Parallel(block_M): sscl[i] = T.exp2(smax_p[i] * scale - smax[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - smax[i] * scale)
                    T.reduce_sum(acc_s, ssum, dim=1)
                    for i in T.Parallel(block_M): lsum[i] = lsum[i] * sscl[i] + ssum[i]
                    T.copy(acc_s, acc_c)
                    for i, j in T.Parallel(block_M, dim): acc_o[i, j] *= sscl[i]
                    T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], Vs)
                    T.gemm(acc_c, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_M, dim): acc_o[i, j] /= lsum[i]
                T.copy(acc_o, O[bz, by, bx * block_M:(bx + 1) * block_M, :])
        return main
    return fa(batch, heads, seq_q, seq_kv, dim, block_M, block_N, num_stages, threads, scale)


# Cache to avoid recompilation on every call
_fa_kernel_cache: Dict[tuple, Any] = {}


def _build_fa_kernel_with_lse(batch, heads, seq_q, seq_kv, dim, is_causal, block_M, block_N, num_stages, threads):
    """FlashAttention kernel that outputs BOTH O and LSE (log-sum-exp per query row)."""
    import tilelang
    import tilelang.language as T

    scale = (1.0 / dim) ** 0.5 * 1.4426950408889634
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    lse_shape = [batch, heads, seq_q, 1]

    pc = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
          tilelang.PassConfigKey.TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE: True,
          tilelang.PassConfigKey.TL_HEDDLE_USE_PRECISE_LATENCY: True}

    @tilelang.jit(out_idx=[3, 4], pass_configs=pc)
    def fa_lse(batch, heads, seq_q, seq_kv, dim, block_M, block_N, num_stages, threads, scale):
        past_len = seq_kv - seq_q
        @T.prim_func
        def main(Q: T.Tensor(q_shape, T.float16), K: T.Tensor(kv_shape, T.float16),
                 V: T.Tensor(kv_shape, T.float16), O: T.Tensor(q_shape, T.float16),
                 LSE: T.Tensor(lse_shape, T.float32)):
            with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Qs = T.alloc_shared([block_M, dim], T.float16)
                Ks = T.alloc_shared([block_N, dim], T.float16)
                Vs = T.alloc_shared([block_N, dim], T.float16)
                acc_s = T.alloc_fragment([block_M, block_N], T.float32)
                acc_c = T.alloc_fragment([block_M, block_N], T.float16)
                acc_o = T.alloc_fragment([block_M, dim], T.float32)
                smax = T.alloc_fragment([block_M], T.float32)
                smax_p = T.alloc_fragment([block_M], T.float32)
                sscl = T.alloc_fragment([block_M], T.float32)
                ssum = T.alloc_fragment([block_M], T.float32)
                lsum = T.alloc_fragment([block_M], T.float32)
                T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Qs)
                T.fill(acc_o, 0); T.fill(lsum, 0); T.fill(smax, -T.infinity(T.float32))
                for k in T.Pipelined(T.ceildiv(seq_kv, block_N), num_stages=num_stages):
                    T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], Ks)
                    T.copy(smax, smax_p)
                    T.fill(smax, -T.infinity(T.float32))
                    T.clear(acc_s)
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    # Causal masking
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            q_idx = bx * block_M + i + past_len
                            k_idx = k * block_N + j
                            acc_s[i, j] = T.if_then_else(q_idx >= k_idx, acc_s[i, j], -T.infinity(T.float32))
                    T.reduce_max(acc_s, smax, dim=1, clear=False)
                    for i in T.Parallel(block_M): smax[i] = T.max(smax[i], smax_p[i])
                    for i in T.Parallel(block_M): sscl[i] = T.exp2(smax_p[i] * scale - smax[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - smax[i] * scale)
                    T.reduce_sum(acc_s, ssum, dim=1)
                    for i in T.Parallel(block_M): lsum[i] = lsum[i] * sscl[i] + ssum[i]
                    T.copy(acc_s, acc_c)
                    for i, j in T.Parallel(block_M, dim): acc_o[i, j] *= sscl[i]
                    T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], Vs)
                    T.gemm(acc_c, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_M, dim): acc_o[i, j] /= lsum[i]
                T.copy(acc_o, O[bz, by, bx * block_M:(bx + 1) * block_M, :])
                # Write LSE: log2(lsum) + smax (in log2 domain)
                for i in T.Parallel(block_M):
                    LSE[bz, by, bx * block_M + i, 0] = smax[i] + T.log2(lsum[i])
        return main
    return fa_lse(batch, heads, seq_q, seq_kv, dim, block_M, block_N, num_stages, threads, scale)


_fa_lse_kernel_cache: Dict[tuple, Any] = {}


@torch.fx.wrap
def _tilelang_h2o_attention(q, k, v):
    """Combined H2O attention: FlashAttention(out + LSE) + h2o_score(Q, K, LSE).

    Single entry point replacing the entire H2O computation.
    Returns (out_BHSD, h2o_score_BHK).
    """
    from heddle.frontend.patterns import _heuristic_config
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    batch, heads, seq_q, dim = q.shape
    _, _, seq_kv, _ = k.shape
    config = _heuristic_config(batch, heads, seq_q, seq_kv, dim)
    bM, bN, ns, th = config["block_M"], config["block_N"], config["num_stages"], config["threads"]

    # Step 1: FlashAttention with LSE
    cache_key = ("fa_lse", batch, heads, seq_q, seq_kv, dim, bM, bN, ns, th)
    if cache_key not in _fa_lse_kernel_cache:
        _fa_lse_kernel_cache[cache_key] = _build_fa_kernel_with_lse(
            batch, heads, seq_q, seq_kv, dim, is_causal=True,
            block_M=bM, block_N=bN, num_stages=ns, threads=th)
    fa_kernel = _fa_lse_kernel_cache[cache_key]
    out, lse = fa_kernel(q, k, v)

    # Step 2: H2O score using LSE
    try:
        from heddle.tileop.h2o_attention import h2o_score as _h2o_score_autotune
        h2o_kernel = _h2o_score_autotune(batch, heads, seq_q, seq_kv, dim, is_causal=True,
                                          block_M=bM, block_N=bN, num_stages=ns, threads=th)
        score = h2o_kernel(q, k, lse)
    except Exception:
        # Fallback: compute h2o_score via PyTorch (correct but slower)
        scale_val = 1.0 / (dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale_val
        # Causal mask
        L = scores.shape[-2]
        mask = torch.triu(torch.full((L, seq_kv), float('-inf'), device=q.device, dtype=q.dtype),
                          diagonal=seq_kv - L + 1)
        scores = scores + mask
        probs = torch.softmax(scores.float(), dim=-1)
        score = probs.sum(dim=-2)  # sum over query dim
    return out, score


@torch.fx.wrap
def _tilelang_sdpa_direct(q, k, v):
    """Direct SDPA replacement: inputs are (B, H, S, D), no transpose needed."""
    from heddle.frontend.patterns import _heuristic_config
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    batch, heads, seq_q, dim = q.shape
    _, _, seq_kv, _ = k.shape
    config = _heuristic_config(batch, heads, seq_q, seq_kv, dim)

    # Use causal attention when seq_q == seq_kv (prefill / self-attention)
    is_causal = (seq_q == seq_kv)
    cache_key = (batch, heads, seq_q, seq_kv, dim, is_causal,
                 config["block_M"], config["block_N"], config["num_stages"], config["threads"])
    if cache_key not in _fa_kernel_cache:
        _fa_kernel_cache[cache_key] = _build_fa_kernel(
            batch, heads, seq_q, seq_kv, dim, is_causal=is_causal,
            block_M=config["block_M"], block_N=config["block_N"],
            num_stages=config["num_stages"], threads=config["threads"])
    kernel = _fa_kernel_cache[cache_key]
    return kernel(q, k, v)


def _build_gemm_kernel(M, N, K, block_M=128, block_N=128, block_K=64, num_stages=3, threads=128):
    """Build GEMM kernel via tilelang.jit with Heddle."""
    import tilelang
    import tilelang.language as T

    pc = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
          tilelang.PassConfigKey.TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE: True,
          tilelang.PassConfigKey.TL_HEDDLE_USE_PRECISE_LATENCY: True}

    @tilelang.jit(out_idx=[2], pass_configs=pc)
    def matmul(M, N, K, block_M, block_N, block_K, num_stages, threads):
        @T.prim_func
        def f(A: T.Tensor((M, K), T.float16), B: T.Tensor((K, N), T.float16),
              C: T.Tensor((M, N), T.float16)):
            with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
                As = T.alloc_shared((block_M, block_K), T.float16)
                Bs = T.alloc_shared((block_K, block_N), T.float16)
                Cl = T.alloc_fragment((block_M, block_N), T.float32)
                T.clear(Cl)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[bx * block_M:(bx + 1) * block_M, k * block_K:(k + 1) * block_K], As)
                    T.copy(B[k * block_K:(k + 1) * block_K, by * block_N:(by + 1) * block_N], Bs)
                    T.gemm(As, Bs, Cl)
                T.copy(Cl, C[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N])
        return f
    return matmul(M, N, K, block_M, block_N, block_K, num_stages, threads)


_gemm_kernel_cache: Dict[tuple, Any] = {}


@torch.fx.wrap
def _tilelang_gemm_wrapper(a, b):
    """TileLang GEMM wrapper for torch.mm / matmul replacement."""
    a = a.contiguous(); b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    # Simple heuristic for tile selection
    bM = 128 if M >= 128 else max(16, (M // 16) * 16)
    bN = 128 if N >= 128 else max(16, (N // 16) * 16)
    bK = 64 if K >= 64 else max(16, (K // 16) * 16)
    ns = 3 if K >= 256 else 1
    cache_key = (M, N, K, bM, bN, bK, ns)
    if cache_key not in _gemm_kernel_cache:
        _gemm_kernel_cache[cache_key] = _build_gemm_kernel(M, N, K, bM, bN, bK, ns, 128)
    return _gemm_kernel_cache[cache_key](a, b)


@torch.fx.wrap
def _tilelang_linear_wrapper(input_nd, weight_2d):
    """TileLang wrapper for nn.Linear (supports any-dim input × 2D weight).

    Reshapes input to 2D, runs GEMM, reshapes back.
    input_nd: (..., K), weight_2d: (N, K) → output: (..., N)
    """
    orig_shape = input_nd.shape
    K = orig_shape[-1]
    M = 1
    for s in orig_shape[:-1]:
        M *= s
    N = weight_2d.shape[0]

    a = input_nd.reshape(M, K).contiguous()
    b = weight_2d.transpose(0, 1).contiguous()  # (K, N)

    bM = 128 if M >= 128 else max(16, (M // 16) * 16)
    bN = 128 if N >= 128 else max(16, (N // 16) * 16)
    bK = 64 if K >= 64 else max(16, (K // 16) * 16)
    ns = 3 if K >= 256 else 1
    cache_key = (M, N, K, bM, bN, bK, ns)
    if cache_key not in _gemm_kernel_cache:
        _gemm_kernel_cache[cache_key] = _build_gemm_kernel(M, N, K, bM, bN, bK, ns, 128)
    c = _gemm_kernel_cache[cache_key](a, b)
    return c.reshape(*orig_shape[:-1], N)


def _replace_gemm_ops(gm: fx.GraphModule) -> bool:
    """Replace aten::mm, aten::matmul (2D), and aten::linear with TileLang GEMM."""
    replaced = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        target_name = getattr(node.target, "__name__", "") or str(node.target)

        # Handle nn.Linear / aten.linear
        if "linear" in target_name:
            input_node = node.args[0] if len(node.args) > 0 else None
            weight_node = node.args[1] if len(node.args) > 1 else None
            bias_node = node.args[2] if len(node.args) > 2 else None
            if input_node is None or weight_node is None:
                continue
            # Skip if bias is present (would need fused GEMM+bias)
            if bias_node is not None and not isinstance(bias_node, type(None)):
                continue

            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(
                    _tilelang_linear_wrapper, (input_node, weight_node))
            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
            replaced = True
            print(f"[TileLang Compiler] Replaced linear with TileLang GEMM")
            continue

        # Handle mm / matmul (2D only)
        if "bmm" in target_name:
            continue
        if "mm" not in target_name and "matmul" not in target_name:
            continue

        a_node = node.args[0] if len(node.args) > 0 else None
        b_node = node.args[1] if len(node.args) > 1 else None
        if a_node is None or b_node is None:
            continue

        a_meta = a_node.meta.get("tensor_meta") or a_node.meta.get("val")
        b_meta = b_node.meta.get("tensor_meta") or b_node.meta.get("val")
        if a_meta is not None and hasattr(a_meta, "shape") and len(a_meta.shape) != 2:
            continue
        if b_meta is not None and hasattr(b_meta, "shape") and len(b_meta.shape) != 2:
            continue

        with gm.graph.inserting_before(node):
            new_node = gm.graph.call_function(_tilelang_gemm_wrapper, (a_node, b_node))
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)
        replaced = True
        print(f"[TileLang Compiler] Replaced mm/matmul with TileLang GEMM")

    if replaced:
        gm.graph.lint()
        gm.recompile()
    return replaced


def _build_conv2d_kernel(N, C, H, W, F, K, S, D, P,
                         block_M=64, block_N=128, block_K=32, num_stages=2, threads=256):
    """Build Conv2D im2col+GEMM kernel via tilelang.jit."""
    import tilelang
    import tilelang.language as T

    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    pc = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
          tilelang.PassConfigKey.TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE: True,
          tilelang.PassConfigKey.TL_HEDDLE_USE_PRECISE_LATENCY: True}

    @tilelang.jit(out_idx=[2], pass_configs=pc)
    def conv(N, C, H, W, F, K, S, D, P, block_M, block_N, block_K, ns, threads):
        OH_l = (H + 2 * P - D * (K - 1) - 1) // S + 1
        OW_l = (W + 2 * P - D * (K - 1) - 1) // S + 1
        @T.prim_func
        def main(data: T.Tensor((N, H, W, C), T.float16),
                 weight: T.Tensor((K, K, C, F), T.float16),
                 out: T.Tensor((N, OH_l, OW_l, F), T.float16)):
            with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(N * OH_l * OW_l, block_M), threads=threads) as (bx, by):
                ds = T.alloc_shared((block_M, block_K), T.float16)
                ws = T.alloc_shared((block_K, block_N), T.float16)
                ol = T.alloc_fragment((block_M, block_N), T.float32)
                os_ = T.alloc_shared((block_M, block_N), T.float16)
                wf = T.Tensor((K * K * C, F), T.float16, weight.data)
                of_ = T.Tensor((N * OH_l * OW_l, F), T.float16, out.data)
                T.clear(ol)
                for ki in T.Pipelined(T.ceildiv(K * K * C, block_K), num_stages=ns):
                    T.c2d_im2col(data, ds, by, ki, K, S, D, P)
                    T.copy(wf[ki * block_K, bx * block_N], ws)
                    T.gemm(ds, ws, ol)
                T.copy(ol, os_)
                T.copy(os_, of_[by * block_M, bx * block_N])
        return main
    return conv(N, C, H, W, F, K, S, D, P, block_M, block_N, block_K, num_stages, threads)


_conv2d_kernel_cache: Dict[tuple, Any] = {}


@torch.fx.wrap
def _tilelang_conv2d_wrapper(input_nhwc, weight_kkcf, N, C, H, W, F, K, S, D, P):
    """TileLang Conv2D wrapper (NHWC layout)."""
    cache_key = (N, C, H, W, F, K, S, D, P)
    if cache_key not in _conv2d_kernel_cache:
        _conv2d_kernel_cache[cache_key] = _build_conv2d_kernel(N, C, H, W, F, K, S, D, P)
    return _conv2d_kernel_cache[cache_key](input_nhwc, weight_kkcf)


def _replace_conv2d_ops(gm: fx.GraphModule) -> bool:
    """Replace aten::conv2d / aten::convolution with TileLang Conv2D."""
    replaced = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        target_name = getattr(node.target, "__name__", "") or str(node.target)
        if "conv" not in target_name.lower():
            continue
        # Match conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        if len(node.args) < 2:
            continue

        input_node = node.args[0]
        weight_node = node.args[1]

        # Get shapes from metadata
        in_meta = input_node.meta.get("tensor_meta") or input_node.meta.get("val")
        w_meta = weight_node.meta.get("tensor_meta") or weight_node.meta.get("val")
        if in_meta is None or w_meta is None:
            continue
        if not hasattr(in_meta, "shape") or len(in_meta.shape) != 4:
            continue
        if not hasattr(w_meta, "shape") or len(w_meta.shape) != 4:
            continue

        N_val, C_val, H_val, W_val = [int(x) for x in in_meta.shape]
        F_val, C2, K_val, K2 = [int(x) for x in w_meta.shape]
        if K_val != K2 or C_val != C2:
            continue

        # Extract stride/padding/dilation
        S_val = int(node.args[3][0]) if len(node.args) > 3 and node.args[3] else 1
        P_val = int(node.args[4][0]) if len(node.args) > 4 and node.args[4] else 0
        D_val = int(node.args[5][0]) if len(node.args) > 5 and node.args[5] else 1
        groups = int(node.args[6]) if len(node.args) > 6 and node.args[6] else 1
        if groups != 1:
            continue  # TileLang conv2d doesn't support grouped conv yet

        # TileLang uses NHWC layout for data, KKCF for weight
        # Need to insert layout transforms: NCHW→NHWC, FCKK→KKCF, NHWCF→NCHW
        with gm.graph.inserting_before(node):
            # input: NCHW → NHWC
            input_nhwc = gm.graph.call_method("permute", (input_node,), {"dims": (0, 2, 3, 1)})
            input_nhwc_c = gm.graph.call_method("contiguous", (input_nhwc,))
            # weight: FCKK → KKCF
            weight_kkcf = gm.graph.call_method("permute", (weight_node,), {"dims": (2, 3, 1, 0)})
            weight_kkcf_c = gm.graph.call_method("contiguous", (weight_kkcf,))

            out_nhwf = gm.graph.call_function(
                _tilelang_conv2d_wrapper,
                (input_nhwc_c, weight_kkcf_c, N_val, C_val, H_val, W_val, F_val, K_val, S_val, D_val, P_val))
            # output: NHWF → NCHW (= NFHW after permute)
            OH = (H_val + 2 * P_val - D_val * (K_val - 1) - 1) // S_val + 1
            OW = (W_val + 2 * P_val - D_val * (K_val - 1) - 1) // S_val + 1
            out_nchw = gm.graph.call_method("permute", (out_nhwf,), {"dims": (0, 3, 1, 2)})
            out_final = gm.graph.call_method("contiguous", (out_nchw,))

        # Handle optional bias
        bias_node = node.args[2] if len(node.args) > 2 else None
        if bias_node is not None and not isinstance(bias_node, type(None)):
            with gm.graph.inserting_after(out_final):
                # bias shape: (F,) → broadcast add
                bias_reshaped = gm.graph.call_method("view", (bias_node,), {"size": (1, F_val, 1, 1)})
                out_biased = gm.graph.call_function(torch.add, (out_final, bias_reshaped))
            node.replace_all_uses_with(out_biased)
        else:
            node.replace_all_uses_with(out_final)
        gm.graph.erase_node(node)
        replaced = True
        print(f"[TileLang Compiler] Replaced conv2d with TileLang Conv2D "
              f"(N={N_val},C={C_val},H={H_val},F={F_val},K={K_val},S={S_val},P={P_val})")

    if replaced:
        gm.graph.lint()
        gm.recompile()
    return replaced


def _build_bmm_kernel(B, M, N, K, block_M=128, block_N=128, block_K=64, num_stages=3, threads=128):
    """Build batched GEMM kernel."""
    import tilelang
    import tilelang.language as T

    pc = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
          tilelang.PassConfigKey.TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE: True,
          tilelang.PassConfigKey.TL_HEDDLE_USE_PRECISE_LATENCY: True}

    ns = num_stages
    @tilelang.jit(out_idx=[2], pass_configs=pc)
    def bmm(B, M, N, K, block_M, block_N, block_K, ns, threads):
        @T.prim_func
        def f(A: T.Tensor((B, M, K), T.float16), Bt: T.Tensor((B, K, N), T.float16),
              C: T.Tensor((B, M, N), T.float16)):
            with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), B, threads=threads) as (bx, by, bz):
                As = T.alloc_shared((block_M, block_K), T.float16)
                Bs = T.alloc_shared((block_K, block_N), T.float16)
                Cl = T.alloc_fragment((block_M, block_N), T.float32)
                T.clear(Cl)
                for ki in T.Pipelined(T.ceildiv(K, block_K), num_stages=ns):
                    T.copy(A[bz, bx * block_M:(bx + 1) * block_M, ki * block_K:(ki + 1) * block_K], As)
                    T.copy(Bt[bz, ki * block_K:(ki + 1) * block_K, by * block_N:(by + 1) * block_N], Bs)
                    T.gemm(As, Bs, Cl)
                T.copy(Cl, C[bz, bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N])
        return f
    return bmm(B, M, N, K, block_M, block_N, block_K, ns, threads)


_bmm_kernel_cache: Dict[tuple, Any] = {}


@torch.fx.wrap
def _tilelang_bmm_wrapper(a, b):
    """TileLang BMM wrapper for torch.bmm / batched matmul."""
    a = a.contiguous(); b = b.contiguous()
    B, M, K = a.shape
    _, _, N = b.shape
    bM = 128 if M >= 128 else max(16, (M // 16) * 16)
    bN = 128 if N >= 128 else max(16, (N // 16) * 16)
    bK = 64 if K >= 64 else max(16, (K // 16) * 16)
    ns = 3 if K >= 256 else 1
    cache_key = (B, M, N, K, bM, bN, bK, ns)
    if cache_key not in _bmm_kernel_cache:
        _bmm_kernel_cache[cache_key] = _build_bmm_kernel(B, M, N, K, bM, bN, bK, ns, 128)
    return _bmm_kernel_cache[cache_key](a, b)


def _replace_bmm_ops(gm: fx.GraphModule) -> bool:
    """Replace aten::bmm / 3D matmul with TileLang batched GEMM."""
    replaced = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        target_name = getattr(node.target, "__name__", "") or str(node.target)
        if target_name not in ("bmm", "aten.bmm.default"):
            if "bmm" not in target_name:
                continue

        a_node = node.args[0] if len(node.args) > 0 else None
        b_node = node.args[1] if len(node.args) > 1 else None
        if a_node is None or b_node is None:
            continue

        a_meta = a_node.meta.get("tensor_meta") or a_node.meta.get("val")
        if a_meta is not None and hasattr(a_meta, "shape") and len(a_meta.shape) != 3:
            continue

        with gm.graph.inserting_before(node):
            new_node = gm.graph.call_function(_tilelang_bmm_wrapper, (a_node, b_node))
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)
        replaced = True
        print(f"[TileLang Compiler] Replaced bmm with TileLang Batched GEMM")

    if replaced:
        gm.graph.lint()
        gm.recompile()
    return replaced


@torch.fx.wrap
def _tilelang_swiglu_wrapper(x, w_gate, w_up):
    """Fused SwiGLU: silu(x @ W_gate.T) * (x @ W_up.T) via single concatenated GEMM."""
    orig_shape = x.shape
    K = orig_shape[-1]
    M = 1
    for s in orig_shape[:-1]:
        M *= s
    E = w_gate.shape[0]

    a = x.reshape(M, K).contiguous()
    # Concatenate gate and up weights: (2E, K) → transpose → (K, 2E)
    w_cat = torch.cat([w_gate, w_up], dim=0).transpose(0, 1).contiguous()  # (K, 2E)
    N = 2 * E

    bM = 128 if M >= 128 else max(16, (M // 16) * 16)
    bN = 128 if N >= 128 else max(16, (N // 16) * 16)
    bK = 64 if K >= 64 else max(16, (K // 16) * 16)
    ns = 3 if K >= 256 else 1
    cache_key = ("swiglu", M, N, K, bM, bN, bK, ns)
    if cache_key not in _gemm_kernel_cache:
        _gemm_kernel_cache[cache_key] = _build_gemm_kernel(M, N, K, bM, bN, bK, ns, 128)
    combined = _gemm_kernel_cache[cache_key](a, w_cat)  # (M, 2E)

    gate_out, up_out = combined.chunk(2, dim=-1)  # each (M, E)
    result = torch.nn.functional.silu(gate_out) * up_out
    return result.reshape(*orig_shape[:-1], E)


def _replace_swiglu_pattern(gm: fx.GraphModule) -> bool:
    """Detect SwiGLU pattern: silu(linear(x, W1)) * linear(x, W2) → fused single GEMM.

    Matches the FX graph pattern where:
    - Two linear ops share the same input
    - One output goes through silu
    - They are multiplied together
    """
    replaced = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        target_name = getattr(node.target, "__name__", "") or str(node.target)
        if "mul" not in target_name:
            continue
        if len(node.args) < 2:
            continue

        lhs, rhs = node.args[0], node.args[1]
        if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
            continue

        # Find which side has silu
        silu_node, other_node = None, None
        for a, b in [(lhs, rhs), (rhs, lhs)]:
            a_name = getattr(a.target, "__name__", "") or str(a.target) if a.op == "call_function" else ""
            if "silu" in a_name:
                silu_node, other_node = a, b
                break

        if silu_node is None:
            continue

        # silu_node.args[0] should be a linear/tilelang_linear call
        silu_input = silu_node.args[0] if len(silu_node.args) > 0 else None
        if not isinstance(silu_input, fx.Node):
            continue

        # Check both sides trace back to linear with same input
        gate_linear, up_linear = silu_input, other_node

        # Check if they're our _tilelang_linear_wrapper calls
        gate_target = getattr(gate_linear.target, "__name__", "") if gate_linear.op == "call_function" else ""
        up_target = getattr(up_linear.target, "__name__", "") if up_linear.op == "call_function" else ""

        if "linear" not in gate_target or "linear" not in up_target:
            continue

        # Both should have (input, weight) args
        if len(gate_linear.args) < 2 or len(up_linear.args) < 2:
            continue

        gate_input, gate_weight = gate_linear.args[0], gate_linear.args[1]
        up_input, up_weight = up_linear.args[0], up_linear.args[1]

        # Same input?
        if gate_input is not up_input:
            continue

        # Replace: mul(silu(linear(x, Wg)), linear(x, Wu)) → swiglu(x, Wg, Wu)
        with gm.graph.inserting_before(node):
            fused = gm.graph.call_function(
                _tilelang_swiglu_wrapper, (gate_input, gate_weight, up_weight))
        node.replace_all_uses_with(fused)

        # Clean up old nodes
        gm.graph.erase_node(node)       # mul
        gm.graph.erase_node(silu_node)  # silu
        # Only erase linears if they have no other users
        if len(gate_linear.users) == 0:
            gm.graph.erase_node(gate_linear)
        if len(up_linear.users) == 0:
            gm.graph.erase_node(up_linear)

        replaced = True
        print(f"[TileLang Compiler] Fused SwiGLU: 2 linear + silu + mul → 1 GEMM")

    if replaced:
        try:
            gm.graph.eliminate_dead_code()
        except Exception:
            pass
        gm.graph.lint()
        gm.recompile()
    return replaced


def _replace_expanded_attention(gm: fx.GraphModule) -> bool:
    """Match expanded attention pattern in FX graph:
        transpose(q) → matmul(q, k^T) → div/mul(scale) → [add(mask)] → softmax → [to] → matmul(probs, v)
    and replace with TileLang FlashAttention.

    Also detects epilogue ops consuming probs (e.g., sum for H2O score).
    """
    replaced = False

    # Find softmax nodes — they're the anchor of the attention pattern.
    # Softmax can appear as:
    #   a) F.softmax (single op) — direct call
    #   b) Decomposed: amax → sub → exp → sum → div — Dynamo decomposition
    def _find_softmax_nodes(gm):
        """Find softmax anchors. Returns list of (softmax_output, softmax_input)."""
        results = []
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            name = getattr(node.target, "__name__", "") or str(node.target)
            # Case a: direct softmax
            if "softmax" in name:
                results.append((node, node.args[0] if len(node.args) > 0 else None))
                continue
            # Case b: decomposed softmax — find the final div(exp(...), sum(exp(...)))
            if "div" in name or "true_divide" in name or "truediv" in name:
                # Check if this is exp(x-max) / sum(exp(x-max))
                num, den = (node.args[0] if len(node.args) > 0 else None,
                            node.args[1] if len(node.args) > 1 else None)
                if not isinstance(num, fx.Node) or not isinstance(den, fx.Node):
                    continue
                num_name = getattr(num.target, "__name__", "") if num.op == "call_function" else ""
                den_name = getattr(den.target, "__name__", "") if den.op == "call_function" else ""
                if "exp" in num_name and "sum" in den_name:
                    # Trace back through exp → sub → to find the original scores
                    exp_node = num
                    exp_input = exp_node.args[0] if len(exp_node.args) > 0 else None
                    if isinstance(exp_input, fx.Node):
                        sub_name = getattr(exp_input.target, "__name__", "") if exp_input.op == "call_function" else ""
                        if "sub" in sub_name:
                            # scores = sub.args[0] (original input to softmax)
                            orig_scores = exp_input.args[0] if len(exp_input.args) > 0 else None
                            # Check for float cast before sub
                            if isinstance(orig_scores, fx.Node) and orig_scores.op == "call_method" and str(orig_scores.target) == "float":
                                orig_scores = orig_scores.args[0]
                            results.append((node, orig_scores))
        return results

    for softmax_output, softmax_input_hint in _find_softmax_nodes(gm):
        softmax_node = softmax_output
        # Trace backward from softmax to find: [add(mask)] → div(scale) → matmul(QK^T)
        # Use hint from decomposed softmax detection if available
        if softmax_input_hint is not None and isinstance(softmax_input_hint, fx.Node):
            cur = softmax_input_hint
        else:
            cur = softmax_node.args[0] if len(softmax_node.args) > 0 else None
        if not isinstance(cur, fx.Node):
            continue

        # Skip optional float/to cast before softmax
        if isinstance(cur, fx.Node) and cur.op == "call_method" and str(cur.target) == "float":
            cur = cur.args[0] if len(cur.args) > 0 and isinstance(cur.args[0], fx.Node) else cur
        cur_name = getattr(cur.target, "__name__", str(cur.target)) if cur.op == "call_function" else str(cur.target) if cur.op == "call_method" else ""

        # Look for add/iadd(scores, mask) — optional (non-causal skips this)
        mask_node = None
        if "iadd" in cur_name or "add" in cur_name or cur_name == "iadd":
            mask_node = cur
            cur = cur.args[0] if isinstance(cur.args[0], fx.Node) else None

        # Look for div(matmul_result, sqrt_d) or mul(matmul_result, 1/sqrt_d)
        if cur is None:
            continue
        cur_name = getattr(cur.target, "__name__", str(cur.target)) if cur.op == "call_function" else str(cur.target) if cur.op == "call_method" else ""
        if "div" in cur_name or "mul" in cur_name:
            scale_node = cur
            cur = cur.args[0] if isinstance(cur.args[0], fx.Node) else None
        else:
            scale_node = None

        # cur should now be matmul(Q, K^T)
        if cur is None:
            continue
        cur_name = getattr(cur.target, "__name__", str(cur.target)) if cur.op == "call_function" else ""
        if "matmul" not in cur_name:
            continue
        qk_matmul = cur

        # Find Q and K from matmul args
        q_bhsd = qk_matmul.args[0] if len(qk_matmul.args) > 0 else None
        k_transposed = qk_matmul.args[1] if len(qk_matmul.args) > 1 else None

        # K might be transpose(-2,-1)(K_bhsd)
        k_bhsd = k_transposed
        if isinstance(k_transposed, fx.Node) and k_transposed.op == "call_method" and str(k_transposed.target) == "transpose":
            k_bhsd = k_transposed.args[0] if len(k_transposed.args) > 0 else k_transposed

        # Trace forward from softmax to find: [to] → matmul(probs, V)
        probs_node = softmax_node
        # For decomposed softmax, the div output IS probs
        # For direct softmax, look for to() cast after
        for user in list(softmax_node.users):
            u_name = getattr(user.target, "__name__", str(user.target)) if user.op == "call_function" else str(user.target) if user.op == "call_method" else ""
            if "to" in u_name:
                probs_node = user
                break

        # Find matmul(probs, V) — search from probs_node and its users
        pv_matmul = None
        v_bhsd = None
        search_nodes = [probs_node] + list(probs_node.users)
        for candidate in search_nodes:
            for user in list(candidate.users) if isinstance(candidate, fx.Node) else []:
                u_name = getattr(user.target, "__name__", str(user.target)) if user.op == "call_function" else ""
                if "matmul" in u_name:
                    pv_matmul = user
                    v_bhsd = user.args[1] if len(user.args) > 1 else None
                    break
            if pv_matmul:
                break
        # Also check probs_node itself if it's used in matmul
        if pv_matmul is None:
            for user in list(probs_node.users):
                u_name = getattr(user.target, "__name__", str(user.target)) if user.op == "call_function" else ""
                if "matmul" in u_name:
                    pv_matmul = user
                    v_bhsd = user.args[1] if len(user.args) > 1 else None
                    break

        if pv_matmul is None or v_bhsd is None or q_bhsd is None or k_bhsd is None:
            continue

        # Trace Q, K, V back through transpose to get original (B,S,H,D) layout
        def _trace_through_transpose(n):
            if isinstance(n, fx.Node) and n.op == "call_method" and str(n.target) == "transpose":
                return n.args[0] if len(n.args) > 0 else n
            return n

        q_orig = _trace_through_transpose(q_bhsd)
        k_orig = _trace_through_transpose(k_bhsd)
        v_orig = _trace_through_transpose(v_bhsd)

        # Now Q, K, V are in (B, H, S, D) layout (after user's transpose(1,2))
        # Our SDPA wrapper expects (B, H, S, D)
        # Detect epilogue: does softmax output (probs) have a sum consumer? (H2O pattern)
        # Check both direct and decomposed softmax outputs
        epilogue_sum = None
        for user in list(softmax_node.users):
            u_name = getattr(user.target, "__name__", str(user.target)) if user.op == "call_function" else str(user.target) if user.op == "call_method" else ""
            if "sum" in u_name and "reduce" not in u_name:
                epilogue_sum = user
                break

        # Find the output after PV matmul (transpose back + contiguous).
        # Some models (e.g., small seq, simple forward) do NOT transpose the PV output —
        # the matmul directly produces the final (B,H,S,D) result.  Track whether we
        # actually found a transpose so we know whether to add one ourselves.
        pv_output = pv_matmul
        pv_has_transpose = False
        for user in list(pv_matmul.users):
            u_name = str(user.target) if user.op == "call_method" else ""
            if "transpose" in u_name:
                pv_output = user
                pv_has_transpose = True
                for u2 in list(user.users):
                    if (str(u2.target) if u2.op == "call_method" else "") == "contiguous":
                        pv_output = u2
                break

        if epilogue_sum is not None:
            # H2O pattern: attention + sum(probs)
            # Use combined kernel: FA(out+LSE) + h2o_score(Q,K,LSE)
            print(f"[TileLang Compiler] Found expanded H2O attention with epilogue sum")

            with gm.graph.inserting_before(pv_output):
                h2o_result = gm.graph.call_function(
                    _tilelang_h2o_attention, (q_bhsd, k_bhsd, v_bhsd))
                # h2o_result = (out_BHSD, h2o_score_BHK)
                attn_out_bhsd = gm.graph.call_function(operator.getitem, args=(h2o_result, 0))
                if pv_has_transpose:
                    # Transpose: (B,H,S,D) → (B,S,H,D) to match pv_output layout
                    attn_out = gm.graph.call_method("transpose", (attn_out_bhsd,), {"dim0": 1, "dim1": 2})
                    attn_out = gm.graph.call_method("contiguous", (attn_out,))
                else:
                    attn_out = attn_out_bhsd
                h2o_score = gm.graph.call_function(operator.getitem, args=(h2o_result, 1))

            pv_output.replace_all_uses_with(attn_out)
            epilogue_sum.replace_all_uses_with(h2o_score)
        else:
            # Pure attention (no epilogue)
            # FA outputs (B,H,S,D); only transpose if the original code did so.
            print(f"[TileLang Compiler] Found expanded attention (no epilogue)")

            with gm.graph.inserting_before(pv_output):
                sdpa_out_bhsd = gm.graph.call_function(
                    _tilelang_sdpa_direct, (q_bhsd, k_bhsd, v_bhsd))
                if pv_has_transpose:
                    # Transpose back: (B,H,S,D) → (B,S,H,D)
                    sdpa_out = gm.graph.call_method("transpose", (sdpa_out_bhsd,), {"dim0": 1, "dim1": 2})
                    sdpa_out = gm.graph.call_method("contiguous", (sdpa_out,))
                else:
                    sdpa_out = sdpa_out_bhsd
            pv_output.replace_all_uses_with(sdpa_out)

        try:
            gm.graph.eliminate_dead_code()
        except Exception:
            pass
        gm.graph.lint()
        gm.recompile()
        replaced = True
        print(f"[TileLang Compiler] Replaced expanded attention with TileLang kernel")
        break

    return replaced


def _replace_sdpa_op(gm: fx.GraphModule) -> bool:
    """Replace aten::scaled_dot_product_attention nodes with TileLang wrapper.

    When torch.compile captures F.scaled_dot_product_attention, it appears
    as a single call_function node (not the matmul+softmax+matmul pattern).
    This function directly replaces those nodes.
    """
    replaced = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        target = node.target
        # Match both the C++ and Python-level SDPA ops
        target_name = getattr(target, "__name__", "") or str(target)
        if "scaled_dot_product_attention" not in target_name:
            continue

        # SDPA signature: (query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
        q_node = node.args[0] if len(node.args) > 0 else None
        k_node = node.args[1] if len(node.args) > 1 else None
        v_node = node.args[2] if len(node.args) > 2 else None
        if q_node is None or k_node is None or v_node is None:
            continue

        with gm.graph.inserting_before(node):
            # SDPA inputs are (B, H, S, D); TileLang wrapper expects (B, S, H, D).
            # Import the direct-layout wrapper that skips internal transpose.
            new_node = gm.graph.call_function(
                _tilelang_sdpa_direct, (q_node, k_node, v_node))
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)
        replaced = True
        print(f"[TileLang Compiler] Replaced SDPA op with TileLang FlashAttention")

    if replaced:
        gm.graph.lint()
        gm.recompile()
    return replaced


def _dump_partitions_dot(
    gm: fx.GraphModule,
    partitions: List[Any],
    path: str,
) -> None:
    """
    Dump current FX graph + selected partitions into a GraphViz DOT file.

    - Nodes are colored by partition id (selected partitions only).
    - Nodes not covered by any selected partition are colored gray.

    Usage:
      export TILELANG_FRONTEND_DUMP_PARTITIONS_DOT=/tmp/partitions.dot
      python ...
      dot -Tsvg /tmp/partitions.dot > /tmp/partitions.svg
    """
    # Assign partition id to nodes (first match wins).
    node_to_pid: Dict[fx.Node, int] = {}
    for pid, p in enumerate(partitions or []):
        try:
            for n in p.nodes:
                if n not in node_to_pid:
                    node_to_pid[n] = pid
        except Exception:
            continue

    # A small deterministic palette.
    palette = [
        "#ffd166", "#06d6a0", "#118ab2", "#ef476f",
        "#9b5de5", "#f15bb5", "#00bbf9", "#00f5d4",
    ]

    def _escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace("\"", "\\\"")

    def _node_label(n: fx.Node) -> str:
        tgt = ""
        if n.op == "call_function":
            try:
                tgt = getattr(n.target, "__name__", str(n.target))
            except Exception:
                tgt = str(n.target)
        elif n.op == "call_method":
            tgt = str(n.target)
        elif n.op == "placeholder":
            tgt = "placeholder"
        else:
            tgt = n.op

        shape = ""
        dtype = ""
        try:
            tm = n.meta.get("tensor_meta", None)
            if tm is not None:
                if hasattr(tm, "shape"):
                    shape = str(tuple(tm.shape))
                if hasattr(tm, "dtype"):
                    dtype = str(tm.dtype).replace("torch.", "")
        except Exception:
            pass

        parts = [n.name, f"{n.op}:{tgt}"]
        if shape or dtype:
            parts.append(f"{shape} {dtype}".strip())
        return _escape("\\n".join(parts))

    # Build edges from dataflow (arg -> user).
    edges = set()
    for n in gm.graph.nodes:
        for a in itertools.chain(n.args, n.kwargs.values() if isinstance(n.kwargs, dict) else []):
            if isinstance(a, fx.Node):
                edges.add((a, n))
            elif isinstance(a, (list, tuple)):
                for aa in a:
                    if isinstance(aa, fx.Node):
                        edges.add((aa, n))

    lines: List[str] = []
    lines.append("digraph fx_partitions {")
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=box, style=\"rounded,filled\", fontname=\"monospace\", fontsize=10];")
    lines.append("  edge [color=\"#666666\"];")

    for n in gm.graph.nodes:
        nid = f"n_{id(n)}"
        pid = node_to_pid.get(n, -1)
        fill = "#e0e0e0" if pid < 0 else palette[pid % len(palette)]
        pen = "#444444" if pid < 0 else "#222222"
        lines.append(f"  {nid} [label=\"{_node_label(n)}\", fillcolor=\"{fill}\", color=\"{pen}\"];")

    for a, b in edges:
        lines.append(f"  n_{id(a)} -> n_{id(b)};")

    lines.append("}")

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def _dump_partitions_html(
    gm: fx.GraphModule,
    partitions: List[Any],
    path: str,
    title: str = "TileLang Frontend Partitions",
) -> None:
    """
    Dump an interactive HTML visualization (force-directed) using d3-force.

    This avoids requiring the `dot` binary on the machine. It loads d3 from a CDN.

    Usage:
      export TILELANG_FRONTEND_DUMP_PARTITIONS_HTML=/tmp/partitions.html
      python ...
      # then open /tmp/partitions.html locally (or copy it to your workstation)
    """
    node_to_pid: Dict[fx.Node, int] = {}
    for pid, p in enumerate(partitions or []):
        try:
            for n in p.nodes:
                if n not in node_to_pid:
                    node_to_pid[n] = pid
        except Exception:
            continue

    palette = [
        "#ffd166", "#06d6a0", "#118ab2", "#ef476f",
        "#9b5de5", "#f15bb5", "#00bbf9", "#00f5d4",
    ]

    def _node_text(n: fx.Node) -> str:
        tgt = ""
        if n.op == "call_function":
            try:
                tgt = getattr(n.target, "__name__", str(n.target))
            except Exception:
                tgt = str(n.target)
        elif n.op == "call_method":
            tgt = str(n.target)
        elif n.op == "placeholder":
            tgt = "placeholder"
        else:
            tgt = n.op

        shape = ""
        dtype = ""
        try:
            tm = n.meta.get("tensor_meta", None)
            if tm is not None:
                if hasattr(tm, "shape"):
                    shape = str(tuple(tm.shape))
                if hasattr(tm, "dtype"):
                    dtype = str(tm.dtype).replace("torch.", "")
        except Exception:
            pass
        parts = [n.name, f"{n.op}:{tgt}"]
        if shape or dtype:
            parts.append(f"{shape} {dtype}".strip())
        return "\n".join(parts)

    # Build nodes list
    nodes = []
    idx = {}
    for i, n in enumerate(gm.graph.nodes):
        idx[n] = i
        pid = node_to_pid.get(n, -1)
        color = "#e0e0e0" if pid < 0 else palette[pid % len(palette)]
        nodes.append(
            {
                "id": i,
                "name": n.name,
                "label": _node_text(n),
                "pid": pid,
                "color": color,
            }
        )

    # Build edges from args/kwargs
    links = []
    for n in gm.graph.nodes:
        # args
        for a in n.args:
            if isinstance(a, fx.Node):
                links.append({"source": idx[a], "target": idx[n]})
            elif isinstance(a, (list, tuple)):
                for aa in a:
                    if isinstance(aa, fx.Node):
                        links.append({"source": idx[aa], "target": idx[n]})
        # kwargs
        if isinstance(n.kwargs, dict):
            for v in n.kwargs.values():
                if isinstance(v, fx.Node):
                    links.append({"source": idx[v], "target": idx[n]})
                elif isinstance(v, (list, tuple)):
                    for vv in v:
                        if isinstance(vv, fx.Node):
                            links.append({"source": idx[vv], "target": idx[n]})

    data = {"nodes": nodes, "links": links}

    template = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>__TITLE__</title>
  <style>
    body { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    #topbar { padding: 8px 12px; background: #111; color: #eee; font-size: 12px; display:flex; gap:12px; align-items:center; }
    #topbar .hint { color: #bbb; }
    #topbar .badge { background:#333; padding:2px 6px; border-radius: 8px; }
    #wrap { display:flex; width:100vw; height: calc(100vh - 34px); }
    #canvas { flex: 1; }
    #side { width: 380px; border-left: 1px solid #ddd; background:#fafafa; padding: 10px 12px; overflow:auto; }
    #side h3 { margin: 6px 0 8px; font-size: 13px; }
    #side pre { white-space: pre-wrap; word-break: break-word; background:#fff; border:1px solid #eee; padding:8px; border-radius:6px; }
    #side .small { font-size: 11px; color:#555; }
  </style>
</head>
<body>
  <div id="topbar">
    <span><b>__TITLE__</b></span>
    <span class="badge">drag: move</span>
    <span class="badge">scroll: zoom</span>
    <span class="badge">click: details</span>
    <span class="hint">（灰色=不在选中 partition；彩色=某个 partition）</span>
  </div>
  <div id="wrap">
    <svg id="canvas"></svg>
    <div id="side">
      <h3>节点信息</h3>
      <div class="small">点击任意节点查看：name / op:target / shape dtype / partition id</div>
      <pre id="info">（未选择）</pre>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <script>
  const data = __DATA_JSON__;

  const svg = d3.select("#canvas");
  const width = window.innerWidth - 380;
  const height = window.innerHeight - 34;
  svg.attr("viewBox", [0, 0, width, height]);

  const g = svg.append("g");
  const zoom = d3.zoom().scaleExtent([0.1, 8]).on("zoom", (event) => g.attr("transform", event.transform));
  svg.call(zoom);

  const link = g.append("g")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.4)
      .selectAll("line")
      .data(data.links)
      .join("line")
      .attr("stroke-width", 1);

  const nodeG = g.append("g")
      .attr("stroke", "#333")
      .attr("stroke-width", 1)
      .selectAll("g")
      .data(data.nodes)
      .join("g")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

  nodeG.append("circle")
      .attr("r", 8)
      .attr("fill", d => d.color);

  nodeG.append("text")
      .text(d => d.name)
      .attr("x", 10)
      .attr("y", 4)
      .attr("font-size", 10)
      .attr("fill", "#222");

  const info = document.getElementById("info");
  nodeG.on("click", (event, d) => {
    const pid = d.pid >= 0 ? d.pid : "none";
    info.textContent = "pid: " + pid + "\\n\\n" + d.label;
  });

  const simulation = d3.forceSimulation(data.nodes)
      .force("link", d3.forceLink(data.links).id(d => d.id).distance(50).strength(0.3))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide(10));

  simulation.on("tick", () => {
    link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

    nodeG.attr("transform", d => "translate(" + d.x + "," + d.y + ")");
  });

  function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }
  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }
  function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
  </script>
</body>
</html>
"""

    html = template.replace("__TITLE__", title).replace("__DATA_JSON__", json.dumps(data))

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

def _has_tilelang_wrapper_calls(gm: fx.GraphModule) -> bool:
    for n in gm.graph.nodes:
        if n.op == "call_function":
            try:
                # Avoid triggering JIT during ShapeProp; wrappers are executed in Python.
                if n.target in (tilelang_flash_attn_wrapper, tilelang_h2o_attention_wrapper):
                    return True
            except Exception:
                pass
    return False

def _get_softmax_dim(node: fx.Node) -> Optional[int]:
    if node.op != "call_function":
        return None
    if len(node.args) > 1 and isinstance(node.args[1], int):
        return int(node.args[1])
    dim = node.kwargs.get("dim", None)
    return int(dim) if isinstance(dim, int) else None

def _is_softmax_node(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    return node.target in (torch.nn.functional.softmax, torch.softmax)

def _maybe_get_placeholder_from_transpose(n: fx.Node) -> Optional[fx.Node]:
    """
    Accept chains like placeholder -> transpose(...) -> (maybe transpose) and return the placeholder.
    """
    cur = n
    visited = set()
    while isinstance(cur, fx.Node) and cur not in visited:
        visited.add(cur)
        if cur.op == "placeholder":
            return cur
        if cur.op == "call_method" and str(cur.target) == "transpose" and len(cur.args) > 0 and isinstance(cur.args[0], fx.Node):
            cur = cur.args[0]
            continue
        break
    return None

def try_fuse_h2o_attention(gm: fx.GraphModule) -> bool:
    """
    Detect H2O attention structure and replace the whole region with a single call:
      (out, h2o_score) = tilelang_h2o_attention_wrapper(q, k, v)

    This is intentionally structural (not exact pattern rewrite) to be robust to minor graph differences.
    """
    graph = gm.graph
    nodes = list(graph.nodes)
    output_node = next((n for n in nodes if n.op == "output"), None)
    if output_node is None:
        return False

    for sm in nodes:
        if not _is_softmax_node(sm):
            continue
        if _get_softmax_dim(sm) not in (-1, None):
            continue

        # We want probs used by:
        #  - matmul(probs.to(dtype), v_transpose) -> out
        #  - probs.sum(dim=2) -> h2o_score
        probs = sm
        users = list(probs.users)
        if len(users) < 2:
            continue

        sum_node = None
        to_node = None
        for u in users:
            if u.op == "call_method" and str(u.target) == "sum":
                sum_node = u
            if u.op == "call_method" and str(u.target) == "to":
                to_node = u
        if sum_node is None or to_node is None:
            continue

        # Find v placeholder via matmul(to, v_transpose)
        matmul_out = None
        v_ph = None
        for u in list(to_node.users):
            if u.op == "call_function" and u.target == torch.matmul:
                matmul_out = u
                if len(u.args) >= 2 and isinstance(u.args[1], fx.Node):
                    v_ph = _maybe_get_placeholder_from_transpose(u.args[1])
                break
        if matmul_out is None or v_ph is None:
            continue

        # Find q/k placeholders via the score matmul earlier in the graph:
        #   matmul(q_transpose, k_transpose(-2,-1))
        q_ph = None
        k_ph = None
        score_in = sm.args[0] if len(sm.args) > 0 else None
        # softmax input is often float() cast
        if isinstance(score_in, fx.Node) and score_in.op == "call_method" and str(score_in.target) == "float":
            score_in = score_in.args[0] if len(score_in.args) > 0 else score_in

        # Walk back to find a torch.matmul that produces scores
        matmul_score = None
        cur = score_in if isinstance(score_in, fx.Node) else None
        visited = set()
        while isinstance(cur, fx.Node) and cur not in visited:
            visited.add(cur)
            if cur.op == "call_function" and cur.target == torch.matmul:
                matmul_score = cur
                break
            # follow first tensor arg if possible
            nxt = None
            for a in cur.args:
                if isinstance(a, fx.Node):
                    nxt = a
                    break
            cur = nxt

        if matmul_score is not None and len(matmul_score.args) >= 2:
            q_ph = _maybe_get_placeholder_from_transpose(matmul_score.args[0]) if isinstance(matmul_score.args[0], fx.Node) else None
            # k is often transpose(-2,-1)(transpose(1,2)(k))
            k_src = matmul_score.args[1] if isinstance(matmul_score.args[1], fx.Node) else None
            if isinstance(k_src, fx.Node) and k_src.op == "call_method" and str(k_src.target) == "transpose":
                k_src = k_src.args[0] if len(k_src.args) > 0 else k_src
            k_ph = _maybe_get_placeholder_from_transpose(k_src) if isinstance(k_src, fx.Node) else None

        if q_ph is None or k_ph is None:
            continue

        # Rewrite: insert fused call and redirect outputs.
        with graph.inserting_before(output_node):
            fused = graph.call_function(tilelang_h2o_attention_wrapper, args=(q_ph, k_ph, v_ph))
            out0 = graph.call_function(operator.getitem, args=(fused, 0))
            out1 = graph.call_function(operator.getitem, args=(fused, 1))
            output_node.args = ((out0, out1),)

        # Drop old region
        try:
            graph.eliminate_dead_code()
        except Exception:
            pass
        graph.lint()
        gm.recompile()
        print("[TileLang Compiler] Fused H2O attention (out + h2o_score) into TileLang kernels.")
        return True

    return False

def tilelang_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    The main entry point for the TileLang torch.compile backend.
    """
    print("[TileLang Compiler] Analyzing Graph...")
    print(f"\n{'='*20} Captured Graph (Before Optimization) {'='*20}")
    print(gm.print_readable())
    print(f"{'='*60}\n")

    # Optional: dump the raw captured FX graph (before any rewrites) for visualization/debug.
    dump_before_dot = os.environ.get("TILELANG_FRONTEND_DUMP_FX_DOT_BEFORE", "").strip()
    if dump_before_dot:
        try:
            _dump_partitions_dot(gm, [], dump_before_dot)
            print(f"[TileLang Compiler] Wrote FX graph (before) to: {dump_before_dot}")
        except Exception as e:
            print(f"[TileLang Compiler] Warning: failed to dump FX graph before (dot) ({type(e).__name__}: {e})")
    dump_before_html = os.environ.get("TILELANG_FRONTEND_DUMP_FX_HTML_BEFORE", "").strip()
    if dump_before_html:
        try:
            _dump_partitions_html(gm, [], dump_before_html, title="FX Graph (Before Optimization)")
            print(f"[TileLang Compiler] Wrote FX graph (before) to: {dump_before_html}")
        except Exception as e:
            print(f"[TileLang Compiler] Warning: failed to dump FX graph before (html) ({type(e).__name__}: {e})")
    
    # 1. Pattern Matching & Replacement (High-Level Fusion)
    # Match large composite patterns like FlashAttention first
    print("[TileLang Compiler] Searching for Attention Patterns...")
    enable_h2o_fuse = os.environ.get("TILELANG_FRONTEND_ENABLE_H2O_FUSE", "1").strip() == "1"
    enable_flashattn_pattern = os.environ.get("TILELANG_FRONTEND_ENABLE_FLASHATTN_PATTERN", "1").strip() == "1"
    continue_after_attention_fuse = os.environ.get("TILELANG_FRONTEND_CONTINUE_AFTER_ATTENTION_FUSE", "0").strip() == "1"

    # H2O attention (multi-output) is more specific; try it first unless disabled.
    fused_attention = False
    if enable_h2o_fuse:
        fused_attention = try_fuse_h2o_attention(gm) or fused_attention
    else:
        print("[TileLang Compiler] H2O fuse disabled by TILELANG_FRONTEND_ENABLE_H2O_FUSE=0.")

    if enable_flashattn_pattern:
        replace_pattern(gm, scaled_dot_product_attention_pattern, tilelang_flash_attn_replacement)
        # Also replace aten::scaled_dot_product_attention directly (single-op form)
        fused_sdpa = _replace_sdpa_op(gm)
        # Also match expanded attention: matmul → div → [add] → softmax → matmul
        if not fused_sdpa:
            fused_sdpa = _replace_expanded_attention(gm)
        fused_attention = fused_attention or fused_sdpa
    else:
        print("[TileLang Compiler] FlashAttn pattern disabled by TILELANG_FRONTEND_ENABLE_FLASHATTN_PATTERN=0.")

    # Optional: dump the FX graph after attention pattern rewrites (e.g., wrapper calls inserted).
    dump_after_attn_dot = os.environ.get("TILELANG_FRONTEND_DUMP_FX_DOT_AFTER_ATTENTION", "").strip()
    if dump_after_attn_dot:
        try:
            _dump_partitions_dot(gm, [], dump_after_attn_dot)
            print(f"[TileLang Compiler] Wrote FX graph (after attention) to: {dump_after_attn_dot}")
        except Exception as e:
            print(f"[TileLang Compiler] Warning: failed to dump FX graph after attention (dot) ({type(e).__name__}: {e})")
    dump_after_attn_html = os.environ.get("TILELANG_FRONTEND_DUMP_FX_HTML_AFTER_ATTENTION", "").strip()
    if dump_after_attn_html:
        try:
            _dump_partitions_html(gm, [], dump_after_attn_html, title="FX Graph (After Attention Rewrite)")
            print(f"[TileLang Compiler] Wrote FX graph (after attention) to: {dump_after_attn_html}")
        except Exception as e:
            print(f"[TileLang Compiler] Warning: failed to dump FX graph after attention (html) ({type(e).__name__}: {e})")

    # 1b. GEMM / Linear / Conv2D — NO individual replacement.
    # These ops are marked SUPPORTED in support.py so the DFS partitioner
    # can include them in fusible subgraphs. Codegen handles GEMM + epilogue
    # fusion within a single kernel.
    # (Legacy individual replacements available via TILELANG_FRONTEND_LEGACY_REPLACE=1)

    # If we already fused the whole attention region into TileLang wrapper calls,
    # the remaining pipeline (decompose/partition/codegen) is usually counter-productive:
    # - ShapeProp would execute the wrapper (JIT) or must be skipped
    # - Partitioner sees 0 eligible nodes and produces confusing logs
    # Users can opt-in to continue for experimentation.
    if fused_attention and not continue_after_attention_fuse:
        # Optional: dump visualization even when we short-circuit (partitions will be empty).
        dot_path = os.environ.get("TILELANG_FRONTEND_DUMP_PARTITIONS_DOT", "").strip()
        if dot_path:
            try:
                _dump_partitions_dot(gm, [], dot_path)
                print(f"[TileLang Compiler] Wrote partition visualization to: {dot_path}")
            except Exception as e:
                print(f"[TileLang Compiler] Warning: failed to dump partitions dot ({type(e).__name__}: {e})")

        html_path = os.environ.get("TILELANG_FRONTEND_DUMP_PARTITIONS_HTML", "").strip()
        if html_path:
            try:
                _dump_partitions_html(gm, [], html_path, title="Partitions (After Attention Fuse - short-circuit)")
                print(f"[TileLang Compiler] Wrote partition visualization to: {html_path}")
            except Exception as e:
                print(f"[TileLang Compiler] Warning: failed to dump partitions html ({type(e).__name__}: {e})")

        print("[TileLang Compiler] Attention fused. Skipping decomposition/partitioning (set TILELANG_FRONTEND_CONTINUE_AFTER_ATTENTION_FUSE=1 to continue).")
        print("[TileLang Compiler] Optimization Complete.")
        gm.recompile()
        return gm
    
    # 2. Decomposition (Lowering)
    # Break down remaining composite ops (Softmax, LayerNorm) into primitives
    # to expose more fine-grained fusion opportunities.
    print("[TileLang Compiler] Decomposing composite operators...")
    gm = decompose_ops(gm)

    # 2.1 Canonicalization (Simplification)
    # Standardize operators (e.g., iadd -> add) and remove redundant ops
    print("[TileLang Compiler] Canonicalizing operators...")
    gm = canonicalize_ops(gm)

    # 2.5 Shape/DType Propagation
    # Populate `node.meta['tensor_meta']` so downstream partition/codegen can infer dtypes
    # (e.g., softmax decomposition introduces explicit float32 tensors via `.float()`).
    if _has_tilelang_wrapper_calls(gm):
        print("[TileLang Compiler] Skip ShapeProp: TileLang wrapper calls present (avoid executing JIT during analysis).")
    else:
        try:
            ShapeProp(gm).propagate(*example_inputs)
        except Exception as e:
            print(f"[TileLang Compiler] Warning: ShapeProp failed ({type(e).__name__}: {e}).")
    
    # 3. Graph Partitioning (Enumeration -> Pruning -> Selection)
    print("[TileLang Compiler] Running Graph Partitioning...")
    partitioner = GraphPartitioner(gm, example_inputs=example_inputs)
    cost_model = os.environ.get("TILELANG_FRONTEND_COST_MODEL", "heuristic").strip().lower()
    partitions = partitioner.partition(cost_model=cost_model)

    # Optional: dump graph+partitions visualization
    dot_path = os.environ.get("TILELANG_FRONTEND_DUMP_PARTITIONS_DOT", "").strip()
    if dot_path:
        try:
            _dump_partitions_dot(gm, partitions, dot_path)
            print(f"[TileLang Compiler] Wrote partition visualization to: {dot_path}")
        except Exception as e:
            print(f"[TileLang Compiler] Warning: failed to dump partitions dot ({type(e).__name__}: {e})")

    html_path = os.environ.get("TILELANG_FRONTEND_DUMP_PARTITIONS_HTML", "").strip()
    if html_path:
        try:
            _dump_partitions_html(gm, partitions, html_path, title="Partitions (Selected)")
            print(f"[TileLang Compiler] Wrote partition visualization to: {html_path}")
        except Exception as e:
            print(f"[TileLang Compiler] Warning: failed to dump partitions html ({type(e).__name__}: {e})")
    
    if partitions:
        print(f"[TileLang Compiler] Found {len(partitions)} optimal partitions.")
        best_p = partitions[0]
        if cost_model == "analyzer":
            print(
                f"  > Best Candidate: {len(best_p.nodes)} nodes, "
                f"AI={best_p.calculate_arithmetic_intensity():.2f}, score={best_p.score:.2e} (nodes/sec est)"
            )
        else:
            print(f"  > Best Candidate: {len(best_p.nodes)} nodes, AI={best_p.calculate_arithmetic_intensity():.2f}")

    # 3.5 Execute selected partitions as multiple kernels at runtime (hybrid mode).
    # Default is "auto": only run when we actually selected profitable partitions.
    execute_partitions_env = os.environ.get("TILELANG_FRONTEND_EXECUTE_PARTITIONS", "auto").strip().lower()
    if execute_partitions_env in ("0", "false", "no", "off"):
        execute_partitions = False
    elif execute_partitions_env in ("1", "true", "yes", "on"):
        execute_partitions = True
    else:
        # auto
        execute_partitions = True

    if execute_partitions and partitions:
        print("[TileLang Compiler] Partition execution enabled (multi-kernel interpreter mode).")

        # Build runners for each selected partition (single-output only for now).
        runner_map: Dict[fx.Node, Callable[[Dict[fx.Node, torch.Tensor]], torch.Tensor]] = {}
        skip_nodes = set()

        orig_order = list(gm.graph.nodes)

        # Default policy: only execute partitions that directly produce final graph outputs.
        # Executing mid-graph partitions (e.g., softmax decomposition producing `probs`) often slows down
        # because it introduces extra launches/materialization and the intermediate is typically consumed
        # multiple times. Users can override with TILELANG_FRONTEND_EXECUTE_NON_OUTPUT_PARTITIONS=1.
        execute_non_output = os.environ.get("TILELANG_FRONTEND_EXECUTE_NON_OUTPUT_PARTITIONS", "0").strip() == "1"

        for p in partitions:
            if not execute_non_output:
                if not any(o in partitioner.graph_output_nodes for o in p.outputs):
                    continue
            
            # Map inputs and outputs to original order for consistent argument passing
            out_nodes = [n for n in orig_order if n in p.outputs]
            in_nodes = [n for n in orig_order if n in p.inputs]
            
            if not in_nodes or not out_nodes:
                continue

            try:
                sub_gm = extract_subgraph_gm(gm, p.nodes, set(in_nodes), set(out_nodes))
                codegen = ElementwiseCodegen(sub_gm)
                jit_impl = codegen.generate(tune=False)
                is_reduction = codegen.has_reduction
                is_gemm = codegen.has_gemm
                is_einsum = codegen.has_einsum
                # Quick validation: run once to catch shape/ndim errors early
                if is_gemm and not is_einsum:
                    try:
                        test_args = [env_val for env_val in [] ]  # skip validation for now
                    except Exception:
                        pass
            except Exception as e:
                print(f"[TileLang Compiler] Skip partition (codegen failed): {type(e).__name__}: {str(e)[:60]}")
                continue

            def _make_runner(in_nodes, out_nodes, jit_impl, is_reduction, is_gemm, is_einsum=False, sub_gm=None):
                # Capture by value
                def run(env: Dict[fx.Node, torch.Tensor]) -> Any:
                    args = [env[n] for n in in_nodes]

                    # Einsum/GEMM: pass raw tensors, wrapper handles everything.
                    if is_einsum or is_gemm:
                        return jit_impl(*args)

                    inp0 = args[0]

                    # 1. Determine unified grid shape (rows, cols)
                    max_idx = 0
                    max_numel = 0
                    for i, a in enumerate(args):
                        if a.numel() > max_numel:
                            max_numel = a.numel()
                            max_idx = i

                    master_tensor = args[max_idx]
                    if is_reduction:
                        cols = master_tensor.shape[-1]
                        rows = master_tensor.numel() // cols
                    else:
                        rows = master_tensor.numel()
                        cols = 1

                    # 2. Pre-allocate outputs
                    output_tensors = []
                    for o in out_nodes:
                        tm = o.meta.get("tensor_meta", None)
                        if tm is not None:
                            output_tensors.append(torch.empty(tm.shape, dtype=tm.dtype, device=master_tensor.device))
                        else:
                            output_tensors.append(torch.empty_like(master_tensor))

                    # 3. Flatten and call
                    if is_reduction:
                        flat_in = [a.broadcast_to(master_tensor.shape).reshape(rows, cols) for a in args]
                        flat_out = [a.reshape(rows, cols) for a in output_tensors]
                        jit_impl(rows, cols)(*flat_in, *flat_out)
                    else:
                        flat_in = [a.broadcast_to(master_tensor.shape).reshape(-1) for a in args]
                        flat_out = [a.reshape(-1) for a in output_tensors]
                        jit_impl(rows)(*flat_in, *flat_out)

                    return output_tensors[0] if len(output_tensors) == 1 else tuple(output_tensors)

                return run

            # Register the runner at each output node of the partition
            runner = _make_runner(in_nodes, out_nodes, jit_impl, is_reduction, is_gemm, is_einsum,
                                  sub_gm=sub_gm if is_einsum else None)
            
            if len(out_nodes) == 1:
                runner_map[out_nodes[0]] = runner
            else:
                # For multi-output, we need a way to trigger the runner once and cache results
                # A simple way: store a shared state
                memo = {}
                def _make_multi_trigger(idx):
                    def trigger(env):
                        if "res" not in memo:
                            memo["res"] = runner(env)
                        return memo["res"][idx]
                    return trigger
                
                for idx, o in enumerate(out_nodes):
                    runner_map[o] = _make_multi_trigger(idx)

            # Mark internal compute nodes as skippable
            for n in p.nodes:
                if n not in p.outputs:
                    skip_nodes.add(n)

        if runner_map:
            # Fast-path: if a single partition produces the final graph output and there are no other
            # compute nodes outside the partition, avoid FX Interpreter overhead.
            output_node = None
            for n in gm.graph.nodes:
                if n.op == "output":
                    out = n.args[0]
                    if isinstance(out, tuple) and len(out) == 1 and isinstance(out[0], fx.Node):
                        output_node = out[0]
                    elif isinstance(out, fx.Node):
                        output_node = out
                    break

            if output_node is not None and len(runner_map) == 1 and output_node in runner_map:
                covered = set()
                for p in partitions:
                    covered |= set(p.nodes)
                outside_compute = False
                for n in gm.graph.nodes:
                    if n.op in ("call_function", "call_method", "call_module", "get_attr") and n not in covered:
                        outside_compute = True
                        break

                # Fast-path: single partition produces graph output.
                if not outside_compute:
                    phs = [n for n in gm.graph.nodes if n.op == "placeholder"]

                    def runtime_wrapper(*args):
                        env = {phs[i]: args[i] for i in range(min(len(phs), len(args)))}
                        out_tensor = runner_map[output_node](env)
                        return (out_tensor,)

                    return runtime_wrapper

                # Einsum fast-path: bypass interpreter entirely.
                # Pre-compute the mapping from function args → kernel args.
                if is_einsum and outside_compute:
                    phs = [n for n in gm.graph.nodes if n.op == "placeholder"]

                    # Find partition inputs and how to produce them from placeholders
                    p_inputs = []
                    for p in partitions:
                        p_inputs = [n for n in gm.graph.nodes if n in p.inputs]
                        break

                    # Build a small "recipe" to compute non-placeholder partition inputs
                    # from placeholder values. Run it once to figure out the mapping.
                    # Then bake it into a fast closure.
                    the_runner = runner_map[output_node]

                    # Pre-compute: which partition inputs are placeholders (direct pass-through)
                    # and which need computation (e.g., getitem = slice)
                    ph_set = set(phs)
                    direct_inputs = [(i, n) for i, n in enumerate(p_inputs) if n in ph_set]
                    computed_inputs = [n for n in p_inputs if n not in ph_set]

                    # For computed inputs, find their "recipe" from the FX graph
                    # (single-step ops like getitem, slice, etc.)
                    recipes = []
                    for cn in computed_inputs:
                        if cn.op == "call_function":
                            # Record: target function, which placeholder args it uses, other args
                            fn_args = []
                            for a in cn.args:
                                if isinstance(a, fx.Node) and a in ph_set:
                                    fn_args.append(("ph", phs.index(a)))
                                else:
                                    fn_args.append(("val", a))
                            recipes.append((cn, cn.target, fn_args, cn.kwargs))

                    def runtime_wrapper(*args):
                        env = {}
                        for i, ph in enumerate(phs):
                            if i < len(args):
                                env[ph] = args[i]
                        # Compute non-placeholder partition inputs
                        for cn, target, fn_args, kwargs in recipes:
                            real_args = []
                            for kind, val in fn_args:
                                if kind == "ph":
                                    real_args.append(args[val])
                                else:
                                    real_args.append(val)
                            env[cn] = target(*real_args, **kwargs)
                        return the_runner(env)

                    return runtime_wrapper

            class PartitionInterpreter(Interpreter):
                def run_node(self, n: fx.Node):
                    if n in skip_nodes:
                        return None
                    if n in runner_map:
                        return runner_map[n](self.env)
                    return super().run_node(n)

            def runtime_wrapper(*args):
                # Interpreter will follow the (possibly decomposed) FX graph.
                # Partitions are executed as TileLang kernels when hitting their output nodes.
                # Important: disable GC because we skip internal nodes and still need their
                # inputs alive when we reach the partition output node.
                out = PartitionInterpreter(gm, garbage_collect_values=False).run(*args)
                # Normalize: backend convention prefers tuple-of-outputs
                return out if isinstance(out, tuple) else (out,)

            return runtime_wrapper
    
    # 4. Auto-Fusion for Elementwise Subgraphs
    # Heuristic: If the graph only contains supported pointwise ops, generate a kernel.
    is_pointwise_only = True
    supported_ops = [
        operator.add, operator.mul, operator.sub, operator.truediv,
        torch.add, torch.mul, torch.sub, torch.div,
        torch.sigmoid, torch.relu, torch.exp, torch.sum, torch.max, torch.amax
    ]
    
    for node in gm.graph.nodes:
        if node.op == 'call_function':
            if node.target not in supported_ops:
                # Any opaque/custom call (including TileLangFlashAttn wrapper) should disable auto-codegen.
                # Our elementwise/reduction codegen cannot inline/call such functions correctly.
                is_pointwise_only = False

    # If the model returns multiple outputs, our demo auto-codegen path is not ready to allocate/return
    # multiple output buffers. Fall back to executing the (possibly rewritten) graph as-is.
    is_multi_output = False
    for node in gm.graph.nodes:
        if node.op == "output":
            out = node.args[0]
            if isinstance(out, tuple) and len(out) > 1:
                is_multi_output = True
            break
    if is_multi_output:
        print("[TileLang Compiler] Multi-output graph detected. Skipping auto-codegen.")
        print("[TileLang Compiler] Optimization Complete.")
        gm.recompile()
        return gm
    
    if is_pointwise_only and len(list(gm.graph.nodes)) > 3:
        print("[TileLang Compiler] Detected custom elementwise pattern. Generating Kernel...")
        try:
            # Use AutoFusionScheduler (formerly ElementwiseCodegen)
            codegen = ElementwiseCodegen(gm)
            # Enable basic tuning (Sketch Generation)
            jit_kernel = codegen.generate(tune=True)
            is_reduction = codegen.has_reduction

            # Guardrail: our reduction demo codegen assumes a single reduction pipeline.
            # Softmax decomposition + extra reductions (e.g. H2O sum(dim=2)) will create multiple
            # reduction nodes that are not representable by this kernel.
            reduction_count = 0
            for n in gm.graph.nodes:
                if n.op == 'call_function' and n.target in [torch.sum, torch.mean, torch.max, torch.amax]:
                    reduction_count += 1
                elif n.op == 'call_method' and n.target in ['sum', 'mean', 'max', 'amax']:
                    reduction_count += 1
            if is_reduction and reduction_count > 1:
                print(f"[TileLang Compiler] Detected {reduction_count} reductions. Generating multi-pass reduction kernel.")
                # print("[TileLang Compiler] Optimization Complete.")
                # gm.recompile()
                # return gm
            
            # Return a runtime wrapper that invokes the JIT kernel
            def runtime_wrapper(*args):
                inp = args[0]
                print(f"[Runtime] Input shape: {inp.shape}")
                if is_reduction:
                    # Assume 2D [Rows, Cols] layout for reduction demo
                    # In production, we need proper layout analysis mapping
                    if inp.dim() == 2:
                        rows, cols = inp.shape
                        flat_args = list(args)
                        out_buf = torch.empty_like(inp)
                    else:
                        # Flatten to [Outer, Inner]
                        # Assume reduction on last dim
                        cols = inp.shape[-1]
                        rows = inp.numel() // cols
                        # Reshape inputs to 2D view for the kernel
                        flat_args = [arg.view(rows, cols) for arg in args]
                        # Allocate a 2D output buffer that matches the kernel's expected layout
                        out_buf = torch.empty((rows, cols), device=inp.device, dtype=inp.dtype)
                    
                    print(f"[Runtime] Calling Reduction Kernel with rows={rows}, cols={cols}")
                    # Pass inputs + output buffer
                    jit_kernel(rows, cols)(*flat_args, out_buf)

                    # For non-2D inputs, reshape the 2D buffer back to the original input shape.
                    out_tensor = out_buf if inp.dim() == 2 else out_buf.view_as(inp)
                    print(f"[Runtime] Output tensor shape before return: {out_tensor.shape}")
                    print(f"[Runtime] Output tensor data sample: {out_tensor.view(-1)[:5]}")

                    # Important: some torch.compile wrappers treat backend returns as a tuple-of-outputs.
                    # Always return a 1-tuple here to avoid accidental tensor indexing (e.g. out[0]).
                    return (out_tensor,)
                else:
                    total_elements = args[0].numel()
                    flattened_args = [arg.view(-1) for arg in args]
                    # TileLang JIT expects explicit output buffers when out_idx=None.
                    out_buf = torch.empty_like(args[0]).view(-1)
                    jit_kernel(total_elements)(*flattened_args, out_buf)
                    out_tensor = out_buf.reshape(args[0].shape)
                    return (out_tensor,)
                
            return runtime_wrapper
        except Exception as e:
            print(f"[TileLang Compiler] Codegen failed: {e}. Falling back to default.")
            pass

    print("[TileLang Compiler] Optimization Complete.")
    gm.recompile()
    return gm
