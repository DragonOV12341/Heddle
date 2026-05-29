# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.fx as fx
import tilelang
from heddle.tileop.flash_attention import flashattn, flashattn_lse
from heddle.tileop.h2o_attention import h2o_score
import math
import os

from heddle.frontend.automixed_decision import (
    AutoMixedPolicy,
    Candidate,
    StaticEstimate,
    decide_with_fallback,
)

# --- Auto-Mixed Decision Logic ---


def _heuristic_config(B, H, seq_q, seq_kv, D, is_causal: bool = False):
    """
    AutoMixed decision: static prune (SMEM) + Top-K ptxas feedback + deterministic fallback chain.
    Returns: dict(block_M, block_N, num_stages, threads)
    """
    # Enable policy:
    # - TILELANG_AUTOMIXED_ENABLE=never|always|auto
    #   auto: enable only for "high-risk" shapes (default)
    enable_mode = os.environ.get("TILELANG_AUTOMIXED_ENABLE", "auto").strip().lower()
    if enable_mode not in ("auto", "always", "never"):
        enable_mode = "auto"

    enable_ptxas_feedback = os.environ.get("TILELANG_AUTOMIXED_PTXAS_FEEDBACK", "0").strip() == "1"
    regs_hard = int(os.environ.get("TILELANG_AUTOMIXED_PTXAS_SOFT_MAX_REGS", "240").strip() or "240")
    require_no_spill = os.environ.get("TILELANG_AUTOMIXED_PTXAS_REQUIRE_NO_SPILL", "1").strip() == "1"
    try_topk = int(os.environ.get("TILELANG_AUTOMIXED_PTXAS_TRY_TOPK", "3").strip() or "3")
    log = os.environ.get("TILELANG_AUTOMIXED_LOG", "0").strip() == "1"
    trace_path = os.environ.get("TILELANG_AUTOMIXED_TRACE_PATH", "").strip() or None

    # If user explicitly disables, return a stable default.
    if enable_mode == "never":
        return {"block_M": 128, "block_N": 128, "num_stages": 2, "threads": 256}

    # "auto" gating: only enable for shapes likely to hit reg/occupancy cliffs (docs/design.md).
    if enable_mode == "auto":
        try:
            cc_major, _ = torch.cuda.get_device_capability()
        except Exception:
            cc_major = 0
        is_hopper_like = cc_major >= 9
        high_risk = bool(int(D) >= 128 and int(seq_q) >= 512)
        if not (is_hopper_like and high_risk):
            return {"block_M": 128, "block_N": 128, "num_stages": 2, "threads": 256}

    policy = AutoMixedPolicy(
        enable_ptxas_feedback=bool(enable_ptxas_feedback),
        regs_hard=int(regs_hard),
        require_no_spill=bool(require_no_spill),
        try_topk=int(try_topk),
        log=bool(log),
        trace_path=trace_path,
    )

    cand_list: list[Candidate] = [
        Candidate(128, 128, 2, 256),
        Candidate(128, 128, 2, 128),
        Candidate(128, 64, 3, 128),
        Candidate(64, 128, 3, 128),
        Candidate(64, 64, 4, 128),
        Candidate(64, 64, 2, 128),
    ]

    def static_estimator(c: Candidate) -> StaticEstimate:
        # IMPORTANT: shared allocations are not multiplied by num_stages for this kernel shape.
        # Estimate: Q/K/V/O shared buffers in fp16.
        smem_est = (2 * int(c.block_M) + 2 * int(c.block_N)) * int(D) * 2
        return StaticEstimate(smem_est_bytes=int(smem_est), note="Q/K/V/O shared (fp16), no stage-mult")

    def build_fn_of_candidate(c: Candidate):
        def _build():
            # Temporarily enable verbose ptxas output for this compilation.
            old_pass_configs = getattr(flashattn, "pass_configs", None)
            try:
                pc = dict(old_pass_configs or {})
                pc[tilelang.PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT] = True
                flashattn.pass_configs = pc
                return flashattn(
                    int(B),
                    int(H),
                    int(seq_q),
                    int(seq_kv),
                    int(D),
                    bool(is_causal),
                    block_M=int(c.block_M),
                    block_N=int(c.block_N),
                    num_stages=int(c.num_stages),
                    threads=int(c.threads),
                )
            finally:
                try:
                    flashattn.pass_configs = old_pass_configs
                except Exception:
                    pass

        return _build

    def fallback_fn(c: Candidate) -> list[Candidate]:
        out: list[Candidate] = []
        # 1) Reduce stages to shrink pipeline live-ranges.
        if int(c.num_stages) > 1:
            out.append(Candidate(c.block_M, c.block_N, max(1, int(c.num_stages) - 1), c.threads))
        # 2) Shrink tile.
        if int(c.block_M) > 64 or int(c.block_N) > 64:
            out.append(Candidate(min(64, int(c.block_M)), min(64, int(c.block_N)), c.num_stages, c.threads))
        # 3) Increase threads to reduce per-thread reg pressure.
        if int(c.threads) == 128:
            out.append(Candidate(c.block_M, c.block_N, c.num_stages, 256))
        return out

    shape = {"B": int(B), "H": int(H), "Tq": int(seq_q), "Tkv": int(seq_kv), "D": int(D), "is_causal": bool(is_causal)}
    return decide_with_fallback(
        op="flashattn_tileop",
        shape=shape,
        candidates=cand_list,
        static_estimator=static_estimator,
        build_fn_of_candidate=build_fn_of_candidate,
        policy=policy,
        fallback_fn=fallback_fn,
    )

# --- Flash Attention Pattern ---

def scaled_dot_product_attention_pattern(q, k, v, scale):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    k_t = k.transpose(-2, -1)
    scores = torch.matmul(q, k_t)
    scores = scores * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    output = output.transpose(1, 2)
    return output

class TileLangFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        # Runtime Layout Transform
        # Use contiguous() without explicit check to avoid control flow in tracer
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Capture shapes
        batch, seq_q, heads, dim = q.shape
        _, seq_kv, _, _ = k.shape

        # Layout Optimization: (B, S, H, D) -> (B, H, S, D)
        q_device = q.transpose(1, 2).contiguous()
        k_device = k.transpose(1, 2).contiguous()
        v_device = v.transpose(1, 2).contiguous()

        # [Auto-Mixed] Dynamic Configuration
        config = _heuristic_config(batch, heads, seq_q, seq_kv, dim)
        # Optional: print chosen config once per process for debugging/repro
        if os.environ.get("TILELANG_AUTOMIXED_LOG", "0").strip() == "1":
            printed = getattr(TileLangFlashAttnFunc, "_printed_config", False)
            if not printed:
                TileLangFlashAttnFunc._printed_config = True
                print(f"[AutoMixed] flashattn config: B={batch},H={heads},T={seq_q},D={dim} -> {config}")

        # Invoke TileLang Kernel
        # Note: Parameters are hardcoded for demonstration. 
        # In production, these should be autotuned or heuristics-based.
        output = flashattn(
            batch, heads, seq_q, seq_kv, dim, 
            is_causal=False,
            block_M=config["block_M"], 
            block_N=config["block_N"], 
            num_stages=config["num_stages"], 
            threads=config["threads"]
        )(q_device, k_device, v_device)

        # Transpose back
        output = output.transpose(1, 2).contiguous()
        return output

    @staticmethod
    def symbolic(g, q, k, v):
        return g.op("TileLangFlashAttn", q, k, v)

@torch.fx.wrap
def tilelang_flash_attn_wrapper(q, k, v):
    return TileLangFlashAttnFunc.apply(q, k, v)

def tilelang_flash_attn_replacement(q, k, v, scale):
    # Note: TileLang's `flashattn` kernel already applies the standard attention scaling
    # (1 / sqrt(dim)) internally. For the common PyTorch pattern where `scale` equals
    # 1 / sqrt(head_dim), we can ignore `scale` here to match semantics.
    #
    # If you need arbitrary `scale`, we should plumb a ratio into the kernel or pre-scale
    # with (scale / (1/sqrt(dim))) using shape-aware analysis (not implemented yet).
    return tilelang_flash_attn_wrapper(q, k, v)


# --- H2O Attention (out + h2o_score) ---
#
# H2O requires an additional output:
#   h2o_score[b, h, k] = sum_q softmax(score(q,k))
#
# A good implementation should NOT materialize full probs; instead it should reuse
# per-row LSE/denom produced by a FlashAttention-like kernel.

class TileLangH2OFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        # Expect (B, S, H, D) inputs.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        batch, seq_q, heads, dim = q.shape
        _, seq_kv, _, _ = k.shape

        # Layout: (B, S, H, D) -> (B, H, S, D)
        q_device = q.transpose(1, 2).contiguous()
        k_device = k.transpose(1, 2).contiguous()
        v_device = v.transpose(1, 2).contiguous()

        # 1) FlashAttention-style kernel produces (LSE, Out) in (B, H, S, D)
        # Note: use causal attention (H2O uses upper-triangular mask).
        lse, out_bhsd = flashattn_lse(
            batch, heads, seq_q, seq_kv, dim,
            is_causal=True,
            block_M=128, block_N=128, num_stages=2, threads=256,
        )(q_device, k_device, v_device)

        # 2) Compute h2o_score via col-sum using (Q, K, LSE)
        score_bhs = h2o_score(
            batch, heads, seq_q, seq_kv, dim,
            is_causal=True,
            block_M=128, block_N=128, num_stages=2, threads=256,
        )(q_device, k_device, lse)

        # Back to (B, S, H, D)
        out_bshd = out_bhsd.transpose(1, 2).contiguous()
        return out_bshd, score_bhs

    @staticmethod
    def symbolic(g, q, k, v):
        return g.op("TileLangH2O", q, k, v)


@torch.fx.wrap
def tilelang_h2o_attention_wrapper(q, k, v):
    return TileLangH2OFunc.apply(q, k, v)

