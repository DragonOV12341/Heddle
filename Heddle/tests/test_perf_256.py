#!/usr/bin/env python3
"""Heddle performance smoke test.

Verifies that kernels compiled via heddle.init() + pass_configs
produce correct results and expected performance levels.

Usage:
    CUDA_VISIBLE_DEVICES=0 TILELANG_DISABLE_CACHE=1 python heddle/tests/test_perf.py
"""
import os, sys, time
os.environ["TILELANG_DISABLE_CACHE"] = "1"

# Ensure heddle package is found (dev environment workaround)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import heddle
heddle.init()

import torch
import tilelang
import tilelang.language as T
from tilelang.transform import PassConfigKey


def bench(fn, warmup=10, repeat=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def test_gemm():
    M = N = K = 8192
    print(f"\n=== GEMM {M}x{N}x{K} ===")
    bM, bN, bK, stages, threads = 128, 128, 64, 3, 128

    for mode_name, pc in [
        ("Baseline", {}),
        ("Heddle", {
            PassConfigKey.TL_ENABLE_FAST_MATH: True,
            PassConfigKey.TL_ENABLE_AUTO_TL_PIPELINE_SMT: True,
            PassConfigKey.TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE: True,
            PassConfigKey.TL_HEDDLE_USE_PRECISE_LATENCY: True,
            PassConfigKey.TL_HEDDLE_PC_TOTAL_NUM_WARPS: 2,
        }),
    ]:
        @tilelang.jit(out_idx=[2], pass_configs=pc)
        def matmul(M, N, K, bM, bN, bK, stages, threads):
            @T.prim_func
            def gemm(A: T.Tensor((M, K), T.float16),
                     B: T.Tensor((K, N), T.float16),
                     C: T.Tensor((M, N), T.float16)):
                with T.Kernel(T.ceildiv(M, bM), T.ceildiv(N, bN), threads=threads) as (bx, by):
                    As = T.alloc_shared((bM, bK), T.float16)
                    Bs = T.alloc_shared((bK, bN), T.float16)
                    Cl = T.alloc_fragment((bM, bN), T.float32)
                    T.clear(Cl)
                    for k in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                        T.copy(A[bx*bM:(bx+1)*bM, k*bK:(k+1)*bK], As)
                        T.copy(B[k*bK:(k+1)*bK, by*bN:(by+1)*bN], Bs)
                        T.gemm(As, Bs, Cl)
                    T.copy(Cl, C[bx*bM:(bx+1)*bM, by*bN:(by+1)*bN])
            return gemm

        kernel = matmul(M, N, K, bM, bN, bK, stages, threads)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)

        # Correctness
        C_ours = kernel(A, B)
        C_ref = A @ B
        err = (C_ours.float() - C_ref.float()).abs().max().item()
        ok = err < 1.0

        # Performance
        ms = bench(lambda: kernel(A, B))
        tflops = 2 * M * N * K / ms / 1e9

        print(f"  {mode_name:<12}: {tflops:>6.0f} TFLOPS  {ms:.3f} ms  err={err:.4f}  {'OK' if ok else 'FAIL'}")


def test_fa_fwd():
    B, H, Tseq, D = 1, 32, 4096, 128
    print(f"\n=== FlashAttention FWD ({B=} {H=} {Tseq=} {D=}) ===")
    bM, bN, stages, threads = 128, 64, 1, 256
    # bM, bN, stages, threads = 128, 64, 3, 128
    scale = (1.0 / D) ** 0.5 * 1.44269504
    shape = [B, Tseq, H, D]
    flops = 4.0 * B * H * Tseq * Tseq * D
    
    Q = torch.randn(B, Tseq, H, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, Tseq, H, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, Tseq, H, D, device="cuda", dtype=torch.float16)
    
    def inner_baseline(Q, K, V):
        q = Q.permute(0, 2, 1, 3).float()
        k = K.permute(0, 2, 1, 3).float()
        v = V.permute(0, 2, 1, 3).float()
        scores = torch.matmul(q, k.transpose(-2, -1)) * ((1.0 / D) ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v).permute(0, 2, 1, 3).to(Q.dtype)
    
    rets = []
    for mode_name, pc in [
        ("Baseline", {}),
        ("Heddle", {
            PassConfigKey.TL_ENABLE_FAST_MATH: True,
            PassConfigKey.TL_ENABLE_AUTO_TL_PIPELINE_SMT: True,
            PassConfigKey.TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE: True,
            PassConfigKey.TL_HEDDLE_USE_PHASE_B: True,
            PassConfigKey.TL_HEDDLE_USE_PRECISE_LATENCY: True,
            PassConfigKey.TL_HEDDLE_USE_ALAP_PRIORITY: True,
            PassConfigKey.TL_HEDDLE_PC_TOTAL_NUM_WARPS: 2,
        }),
    ]:
        try:
            @tilelang.jit(out_idx=[3], target="cuda -arch=sm_90", pass_configs=pc)
            def kern(B, H, Tseq, D, bM, bN, stages, threads, scale):
                @T.prim_func
                def main(Q: T.Tensor(shape, T.float16), K_: T.Tensor(shape, T.float16),
                         V: T.Tensor(shape, T.float16), O: T.Tensor(shape, T.float16)):
                    with T.Kernel(T.ceildiv(Tseq, bM), H, B, threads=threads) as (bx, by, bz):
                        Qs = T.alloc_shared([bM, D], T.float16)
                        Ks = T.alloc_shared([bN, D], T.float16)
                        Vs = T.alloc_shared([bN, D], T.float16)
                        acc_s = T.alloc_fragment([bM, bN], T.float32)
                        acc_s_c = T.alloc_shared([bM, bN], T.float16)
                        acc_o = T.alloc_fragment([bM, D], T.float32)
                        sm = T.alloc_fragment([bM], T.float32)
                        smp = T.alloc_fragment([bM], T.float32)
                        ss = T.alloc_fragment([bM], T.float32)
                        ssum = T.alloc_fragment([bM], T.float32)
                        ls = T.alloc_fragment([bM], T.float32)
                        ss_shared = T.alloc_shared([bM], T.float32)
                        ls_shared = T.alloc_shared([bM], T.float32)
                        T.copy(Q[bz, bx*bM:(bx+1)*bM, by, :], Qs)
                        T.fill(acc_o, 0); T.fill(ls, 0); T.fill(sm, -T.infinity(T.float32))
                        for k in T.Pipelined(T.ceildiv(Tseq, bN), num_stages=stages):
                            T.copy(K_[bz, k*bN:(k+1)*bN, by, :], Ks)
                            # smp : sm_prev 旧的 QK max值
                            T.copy(sm, smp); T.fill(sm, -T.infinity(T.float32)); T.clear(acc_s)
                            # sm 存放 QK 的 新的max, acc_s = QK
                            T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                            T.reduce_max(acc_s, sm, dim=1, clear=False)
                            for i in T.Parallel(bM):
                                # 取最大
                                sm[i] = T.max(sm[i], smp[i])
                            for i in T.Parallel(bM):
                                # 根据新旧max 计算缩放因子 ss[i]
                                ss[i] = T.exp2(smp[i] * scale - sm[i] * scale)
                                ss_shared[i] = ss[i]
                            for i, j in T.Parallel(bM, bN):
                                # acc_s = exp((QK - 行max)*scale)
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                            # X方向reduce sum , 存入 ssum
                            T.reduce_sum(acc_s, ssum, dim=1)
                            for i, j in T.Parallel(bM, D):
                                acc_o[i, j] *= ss_shared[i]
                            for i in T.Parallel(bM):
                                # ssum 累加，放进 ls（分母） （考虑每轮做缩放 ）
                                ls[i] = ls[i] * ss[i] + ssum[i]
                            # 类型cast f32->f16
                            T.copy(acc_s, acc_s_c)
                            T.copy(V[bz, k*bN:(k+1)*bN, by, :], Vs)
                            # acc_o += acc_s_c @ Vs
                            T.gemm(acc_s_c, Vs, acc_o)
                        for i in T.Parallel(bM):
                            ls_shared[i] = ls[i]
                        for i, j in T.Parallel(bM, D):
                            # acc_o 除以分母
                            acc_o[i, j] /= ls_shared[i]
                        T.copy(acc_o, O[bz, bx*bM:(bx+1)*bM, by, :])
                return main

            compiled = kern(B, H, Tseq, D, bM, bN, stages, threads, scale)

            ms = bench(lambda: compiled(Q, K, V))
            tflops = flops / ms / 1e9
            print(f"  {mode_name:<12}: {tflops:>6.0f} TFLOPS  {ms:.3f} ms")
            rets.append( compiled(Q,K,V) )
        except Exception as e:
            print(f"  {mode_name:<12}: FAIL — {str(e)[:]}")
    assert len(rets) == 2
    baseline = inner_baseline(Q,K,V)
    isEqual0 = torch.allclose(rets[0], baseline, atol=1e-2, rtol=1e-2)
    isEqual1 = torch.allclose(rets[1], baseline, atol=1e-2, rtol=1e-2)
    print(f"{isEqual0=}  {isEqual1=}")
    
if __name__ == "__main__":
    # test_gemm()
    test_fa_fwd()
    print("\nDone.")
