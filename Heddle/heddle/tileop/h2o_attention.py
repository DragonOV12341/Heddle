# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import os


def get_configs():
    iter_params = dict(block_M=[128], block_N=[128], num_stages=[2], threads=[256])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        # Default to precision-first for correctness checks; can enable via env for perf.
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: os.environ.get("TILELANG_H2O_ENABLE_FAST_MATH", "0").strip() == "1",
        # Workaround: avoid warp-specialization pipeline for now (can trigger TVM internal errors).
        "tl.disable_warp_specialized": True,
        "tl.disable_tma_lower": True,
    },
)
def h2o_score(batch, heads, seq_q, seq_kv, dim, is_causal, block_M=64, block_N=64, num_stages=1, threads=128):
    """
    Compute H2O score:
      h2o_score[b,h,k] = sum_q softmax(score(q,k))  (sum over query axis)

    We assume standard attention score:
      score(q,k) = dot(q, k) / sqrt(dim) + causal_mask

    Inputs (layout is B,H,S,D):
      - Q:   [batch, heads, seq_q,  dim]  fp16
      - K:   [batch, heads, seq_kv, dim]  fp16
      - LSE: [batch, heads, seq_q,  1]    fp32, log2(sum(exp2(score*scale)))
             where scale = log2(e)/sqrt(dim)

    Output:
      - Out: [batch, heads, seq_kv]       fp32
    """
    # Use a more accurate log2(e) constant to reduce numerical drift vs PyTorch exp().
    scale = (1.0 / dim) ** 0.5 * 1.4426950408889634  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    k_shape = [batch, heads, seq_kv, dim]
    lse_shape = [batch, heads, seq_q, 1]
    out_shape = [batch, heads, seq_kv]

    dtype = "float16"
    accum_dtype = "float"

    past_len = seq_kv - seq_q
    assert past_len >= 0, "seq_kv must be greater than or equal to seq_q"

    @T.macro
    def MMA(
        Q: T.Tensor(q_shape, dtype),
        Q_shared: T.SharedBuffer([block_M, dim], dtype),
        K: T.Tensor(k_shape, dtype),
        K_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        q_blk: T.int32,
        k_blk: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(Q[bz, by, q_blk * block_M : (q_blk + 1) * block_M, :], Q_shared)
        T.copy(K[bz, by, k_blk * block_N : (k_blk + 1) * block_N, :], K_shared)
        # Mask / OOB init
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                q_idx = q_blk * block_M + i + past_len
                k_idx = k_blk * block_N + j
                acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
        else:
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(k_blk * block_N + j >= seq_kv, -T.infinity(acc_s.dtype), 0)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        LSE: T.Tensor(lse_shape, "float"),
        Out: T.Tensor(out_shape, "float"),
    ):
        with T.Kernel(T.ceildiv(seq_kv, block_N), heads, batch, threads=threads) as (bk, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            col_sum = T.alloc_fragment([block_N], accum_dtype)
            lse_row = T.alloc_fragment([block_M], accum_dtype)
            tmp_col = T.alloc_fragment([block_N], accum_dtype)

            T.fill(col_sum, 0)

            # q blocks range: for causal, only q >= k (+past).
            # conservative lower bound:
            q_start = 0
            if is_causal:
                q_start = T.max(0, (bk * block_N - past_len) // block_M)

            for bq in T.Pipelined(T.ceildiv(seq_q, block_M) - q_start, num_stages=num_stages):
                q_blk = bq + q_start

                # Load LSE for this q block (log2 domain)
                for i in T.Parallel(block_M):
                    q_idx = q_blk * block_M + i
                    lse_row[i] = T.if_then_else(q_idx < seq_q, LSE[bz, by, q_idx, 0], T.infinity(accum_dtype))

                MMA(Q, Q_shared, K, K_shared, acc_s, q_blk, bk, by, bz)

                # Convert scores to probs in a stable way using LSE:
                #   p = exp2(score*scale - lse)
                for i, j in T.Parallel(block_M, block_N):
                    q_idx = q_blk * block_M + i
                    k_idx = bk * block_N + j
                    # OOB: q beyond seq_q, k beyond seq_kv => contribute 0
                    in_range = (q_idx < seq_q) & (k_idx < seq_kv)
                    acc_s[i, j] = T.if_then_else(in_range, T.exp2(acc_s[i, j] * scale - lse_row[i]), 0)

                # Reduce across q (rows) to get partial column sums for this k block.
                T.fill(tmp_col, 0)
                T.reduce_sum(acc_s, tmp_col, dim=0, clear=False)
                for j in T.Parallel(block_N):
                    col_sum[j] += tmp_col[j]

            # Store
            for j in T.Parallel(block_N):
                k_idx = bk * block_N + j
                if k_idx < seq_kv:
                    Out[bz, by, k_idx] = col_sum[j]

    return main


