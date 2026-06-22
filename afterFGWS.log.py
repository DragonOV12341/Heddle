# [d] ---- after ProducerConsumerWarpSpecialized ---
from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K__handle: T.handle, V_handle: T.handle, O_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90a", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        Q = T.match_buffer(Q_handle, (1, 4096, 32, 128), "float16", strides=(16777216, 4096, 128, 1))
        K_ = T.match_buffer(K__handle, (1, 4096, 32, 128), "float16", strides=(16777216, 4096, 128, 1))
        V = T.match_buffer(V_handle, (1, 4096, 32, 128), "float16", strides=(16777216, 4096, 128, 1))
        O = T.match_buffer(O_handle, (1, 4096, 32, 128), "float16", strides=(16777216, 4096, 128, 1))
        with T.block("root"):
            T.reads()
            T.writes()
            Qs = T.Buffer((2, 16, 512), "float16", scope="shared.dyn")
            Ks = T.Buffer((2, 8, 512), "float16", scope="shared.dyn")
            smp = T.Buffer((4,), scope="local")
            sm = T.Buffer((4,), scope="local")
            ss = T.Buffer((4,), scope="local")
            ssum = T.Buffer((4,), scope="local")
            acc_s = T.Buffer((64,), scope="local")
            acc_s_c = T.Buffer((64,), "float16", scope="local")
            Vs = T.Buffer((2, 8, 512), "float16", scope="shared.dyn")
            ls = T.Buffer((4,), scope="local")
            acc_o = T.Buffer((128,), scope="local")
            T.block_attr({"layout_map": {Qs: metadata["tl.Layout"][0], Ks: metadata["tl.Layout"][1], smp: metadata["tl.Fragment"][0], sm: metadata["tl.Fragment"][1], ss: metadata["tl.Fragment"][2], ssum: metadata["tl.Fragment"][3], acc_s: metadata["tl.Fragment"][4], acc_s_c: metadata["tl.Fragment"][5], Vs: metadata["tl.Layout"][2], ls: metadata["tl.Fragment"][6], acc_o: metadata["tl.Fragment"][7]}})
            bx = T.launch_thread("blockIdx.x", 32)
            by = T.launch_thread("blockIdx.y", 32)
            bz = T.launch_thread("blockIdx.z", 1)
            tx = T.launch_thread("threadIdx.x", 384)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            with T.block("tilelang_root"):
                T.reads(Q[0, bx * 128, by, 0], K_[0, 0:4033, by, 0], V[0, 0:4033, by, 0], O[0, bx * 128, by, 0])
                T.writes()
                T.block_attr({"layout_map": {Qs: metadata["tl.Layout"][0], Ks: metadata["tl.Layout"][1], smp: metadata["tl.Fragment"][0], sm: metadata["tl.Fragment"][1], ss: metadata["tl.Fragment"][2], ssum: metadata["tl.Fragment"][3], acc_s: metadata["tl.Fragment"][4], acc_s_c: metadata["tl.Fragment"][5], Vs: metadata["tl.Layout"][2], ls: metadata["tl.Fragment"][6], acc_o: metadata["tl.Fragment"][7]}})
                Qs = T.alloc_buffer((2, 16, 512), "float16", data=Qs.data, scope="shared.dyn")
                Ks_1 = T.alloc_buffer((2, 2, 8, 512), "float16", data=Ks.data, scope="shared.dyn")
                Vs_1 = T.alloc_buffer((2, 2, 8, 512), "float16", data=Vs.data, scope="shared.dyn")
                acc_s = T.alloc_buffer((64,), data=acc_s.data, scope="local")
                acc_s_c = T.alloc_buffer((64,), "float16", data=acc_s_c.data, scope="local")
                acc_o = T.alloc_buffer((128,), data=acc_o.data, scope="local")
                sm = T.alloc_buffer((4,), data=sm.data, scope="local")
                smp = T.alloc_buffer((4,), data=smp.data, scope="local")
                ss = T.alloc_buffer((4,), data=ss.data, scope="local")
                ssum = T.alloc_buffer((4,), data=ssum.data, scope="local")
                ls = T.alloc_buffer((4,), data=ls.data, scope="local")
                T.create_list_of_mbarrier(1, 1, 1, 1, 128, 128, 128, 128, 1)
                T.attr([128, 256], "kWarpSpecializationScope", 0)
                if tx >= 256:  # PRODUCER
                    if T.tl_shuffle_elect(128):
                        T.mbarrier_expect_tx(T.get_mbarrier(8), 32768)
                        for i in T.unroll(2):
                            T.tma_load(T.create_tma_descriptor(6, 4, Q.data, 128, 32, 4096, 1, T.int64(2), T.int64(256), T.int64(8192), T.int64(33554432), 64, 1, 128, 1, 1, 1, 1, 1, 0, 3, 2, 0), T.get_mbarrier(8), T.tvm_access_ptr(T.type_annotation("float16"), Qs.data, i * 8192, 8192, 2), i * 64, by, bx * 128, 0, 0)
                    if T.tl_shuffle_elect(128):
                        T.ptx_arrive_barrier(T.get_mbarrier(8))  # commit async_copy Q [8]
                    for k in T.serial(64, annotations={"tl_finegrainedws_barrier_hints": "Qs:wait=2,arrive=3;Ks:wait=2,arrive=3;Vs:wait=11,arrive=12", "tl_finegrainedws_warp_assigns": "s0:0,s12:0,s9:1,s13:0,s10:0,s14:0,s11:0,s5:1,s3:0,s1:0,s2:0,s4:0,s6:1,s7:0,s8:0", "tl_pcws_barrier_hints": "Qs:wait=2,arrive=3;Ks:wait=2,arrive=3;Vs:wait=11,arrive=12", "tl_pcws_warp_assigns": "s0:0,s12:0,s9:1,s13:0,s10:0,s14:0,s11:0,s5:1,s3:0,s1:0,s2:0,s4:0,s6:1,s7:0,s8:0"}):
                        T.mbarrier_wait_parity(T.get_mbarrier(4 + k % 2), T.bitwise_xor(k // 2 % 2, 1))  # wait bar[4,5]  即 等gemmOK完成
                        if T.tl_shuffle_elect(128):
                            T.mbarrier_expect_tx(T.get_mbarrier(k % 2), 16384)
                            for i in T.unroll(2):
                                T.tma_load(T.create_tma_descriptor(6, 4, K_.data, 128, 32, 4096, 1, T.int64(2), T.int64(256), T.int64(8192), T.int64(33554432), 64, 1, 64, 1, 1, 1, 1, 1, 0, 3, 2, 0), T.get_mbarrier(k % 2), T.tvm_access_ptr(T.type_annotation("float16"), Ks.data, i * 4096 + k % 2 * 8192, 4096, 2), i * 64, by, k * 64, 0, 0)
                        if T.tl_shuffle_elect(128):
                            T.ptx_arrive_barrier(T.get_mbarrier(k % 2))  # commit async_copy K [0,1]
                        T.mbarrier_wait_parity(T.get_mbarrier(6 + k % 2), T.bitwise_xor(k // 2 % 2, 1))  # wait bar[6,7]  等
                        if T.tl_shuffle_elect(128):
                            T.mbarrier_expect_tx(T.get_mbarrier(2 + k % 2), 16384)
                            for i in T.unroll(2):
                                T.tma_load(T.create_tma_descriptor(6, 4, V.data, 128, 32, 4096, 1, T.int64(2), T.int64(256), T.int64(8192), T.int64(33554432), 64, 1, 64, 1, 1, 1, 1, 1, 0, 3, 2, 0), T.get_mbarrier(2 + k % 2), T.tvm_access_ptr(T.type_annotation("float16"), Vs.data, i * 4096 + k % 2 * 8192, 4096, 2), i * 64, by, k * 64, 0, 0)
                        if T.tl_shuffle_elect(128):
                            T.ptx_arrive_barrier(T.get_mbarrier(2 + k % 2))  # commit async_copy V [2,3]
                else:  # CONSUMER 
                    for i in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        for vec in T.vectorized(4):
                            acc_o[i * 4 + vec] = T.float32(0.0)
                    for i in T.vectorized(4):
                        ls[i] = T.float32(0.0)
                    for i in T.vectorized(4):
                        sm[i] = T.infinity("float32") * T.float32(-1.0)
                    T.mbarrier_wait_parity(T.get_mbarrier(8), 0)  # wait Q copy ok
                    if tx < 128:  # wg0
                        for k in T.serial(64, annotations={"tl_finegrainedws_barrier_hints": "Qs:wait=2,arrive=3;Ks:wait=2,arrive=3;Vs:wait=11,arrive=12", "tl_finegrainedws_warp_assigns": "s0:0,s12:0,s9:1,s13:0,s10:0,s14:0,s11:0,s5:1,s3:0,s1:0,s2:0,s4:0,s6:1,s7:0,s8:0", "tl_pcws_barrier_hints": "Qs:wait=2,arrive=3;Ks:wait=2,arrive=3;Vs:wait=11,arrive=12", "tl_pcws_warp_assigns": "s0:0,s12:0,s9:1,s13:0,s10:0,s14:0,s11:0,s5:1,s3:0,s1:0,s2:0,s4:0,s6:1,s7:0,s8:0"}):
                            for i in T.vectorized(4):
                                smp[i] = sm[i]
                            for i in T.vectorized(4):
                                sm[i] = T.infinity("float32") * T.float32(-1.0)
                            T.mbarrier_wait_parity(T.get_mbarrier(k % 2), k // 2 % 2)  # wait bar[0,1] , 即 wait K copyOK
                            for i in T.unroll(16, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                for vec in T.vectorized(4):
                                    acc_s[i * 4 + vec] = T.float32(0.0)
                            with T.block("_gemm_ssr"):
                                T.reads()
                                T.writes()
                                desc_a = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                                desc_b = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                                T.initialize_wgmma_descriptor(desc_a[0], T.tvm_access_ptr(T.type_annotation("float16"), Qs.data, 0, 16384, 1), 1, 1, 64)
                                T.initialize_wgmma_descriptor(desc_b[0], T.tvm_access_ptr(T.type_annotation("float16"), Ks.data, k % 2 * 8192, 8192, 1), 1, 1, 64)
                                T.warpgroup_fence_operand("float32", acc_s.data, 0, 64)
                                T.warpgroup_arrive()
                                for warp_j in T.unroll(1, annotations={"pragma_unroll_explicit": False}):
                                    for i in T.unroll(2, annotations={"pragma_unroll_explicit": False}):
                                        for ki in T.unroll(8, annotations={"pragma_unroll_explicit": False}):
                                            T.ptx_wgmma_ss("m64n64k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a.data, T.shift_right(ki // 4 * 16384 + i * 8192 + ki % 4 * 32, 4), desc_b.data, T.shift_right(ki // 4 * 8192 + ki % 4 * 32, 4), acc_s.data, i * 32, 1, 1, 1)
                                T.warpgroup_commit_batch()
                                T.warpgroup_wait(0)
                                T.warpgroup_fence_operand("float32", acc_s.data, 0, 64)
                            T.ptx_arrive_barrier(T.get_mbarrier(4 + k % 2))  # commit wgmmaQK [4,5]
                            # softmax
                            with T.allocate([4], "float32", "local") as sm_clear:
                                for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                    sm_clear_1 = T.Buffer((4,), data=sm_clear, scope="local")
                                    sm_clear_1[i] = T.float32("-inf")
                                    for rv in T.unroll(16, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                        sm_clear_1[i] = T.max(sm_clear_1[i], acc_s[i // 2 * 32 + rv % 8 * 4 + i % 2 * 2 + rv // 8])
                                    sm_clear_1[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run", sm_clear_1[i])
                                    sm[i] = T.max(sm[i], sm_clear_1[i])
                            for i in T.unroll(64, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                acc_s[i] = T.exp2(acc_s[i] * T.float32(0.1275174307460247) - sm[i // 32 * 2 + i % 4 // 2] * T.float32(0.1275174307460247))
                            for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                ssum[i] = T.float32(0.0)
                                for rv in T.unroll(16, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                    ssum[i] = ssum[i] + acc_s[i // 2 * 32 + rv % 8 * 4 + i % 2 * 2 + rv // 8]
                                ssum[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run", ssum[i])
                            for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                ls[i] = ls[i] * ss[i] + ssum[i]
                            T.mbarrier_wait_parity(T.get_mbarrier(2 + k % 2), k // 2 % 2)  # wait bar[2,3]  等V copyOK
                            for i in T.unroll(16, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                for vec in T.vectorized(4):
                                    acc_s_c[i * 4 + vec] = T.Cast("float16", acc_s[i % 4 // 2 * 32 + i // 4 * 8 + i % 2 * 4 + vec])
                            with T.block("_gemm_rsr"):
                                T.reads()
                                T.writes()
                                desc_b = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                                T.initialize_wgmma_descriptor(desc_b[0], T.tvm_access_ptr(T.type_annotation("float16"), Vs.data, k % 2 * 8192, 8192, 1), 1, 512, 64)
                                T.warpgroup_fence_operand("float16", acc_s_c.data, 0, 32)
                                T.warpgroup_fence_operand("float32", acc_o.data, 0, 128)
                                T.warpgroup_arrive()
                                for warp_j in T.unroll(1, annotations={"pragma_unroll_explicit": False}):
                                    for i in T.unroll(2, annotations={"pragma_unroll_explicit": False}):
                                        for ki in T.unroll(4, annotations={"pragma_unroll_explicit": False}):
                                            T.ptx_wgmma_rs("m64n128k16", T.bool(False), "fp16", "fp16", "fp32", acc_s_c.data, ki * 16 + i * 8, desc_b.data, T.shift_right(ki * 2048, 4), acc_o.data, i * 64, 1, 1, 1)
                                T.warpgroup_commit_batch()
                                T.warpgroup_wait(0)
                                T.warpgroup_fence_operand("float32", acc_o.data, 0, 128)
                                T.warpgroup_fence_operand("float16", acc_s_c.data, 0, 32)
                            T.ptx_arrive_barrier(T.get_mbarrier(6 + k % 2))  # commit gemmPV [6,7]
                    else:  # wg1 
                        for k in T.serial(64, annotations={"tl_finegrainedws_barrier_hints": "Qs:wait=2,arrive=3;Ks:wait=2,arrive=3;Vs:wait=11,arrive=12", "tl_finegrainedws_warp_assigns": "s0:0,s12:0,s9:1,s13:0,s10:0,s14:0,s11:0,s5:1,s3:0,s1:0,s2:0,s4:0,s6:1,s7:0,s8:0", "tl_pcws_barrier_hints": "Qs:wait=2,arrive=3;Ks:wait=2,arrive=3;Vs:wait=11,arrive=12", "tl_pcws_warp_assigns": "s0:0,s12:0,s9:1,s13:0,s10:0,s14:0,s11:0,s5:1,s3:0,s1:0,s2:0,s4:0,s6:1,s7:0,s8:0"}):
                            for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                sm[i] = T.max(sm[i], smp[i])
                            for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                ss[i] = T.exp2(smp[i] * T.float32(0.1275174307460247) - sm[i] * T.float32(0.1275174307460247))
                            for i in T.unroll(128, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                acc_o[i] = acc_o[i] * ss[i // 64 * 2 + i % 4 // 2]
                    # end mainloop
                    for i in T.unroll(128, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        acc_o[i] = acc_o[i] / ls[i // 64 * 2 + i % 4 // 2]
                    for i in T.unroll(64, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        O_local_cast = T.decl_buffer((2,), "float16", scope="local")
                        for vec in T.vectorized(2):
                            O_local_cast[vec] = T.Cast("float16", acc_o[i * 2 + vec])
                        for vec_copy in T.vectorized(2):
                            O[0, bx * 128 + i // 32 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4, by, i % 32 // 2 * 8 + tx % 4 * 2 + vec_copy] = O_local_cast[vec_copy]

# Metadata omitted. Use show_meta=True in script() method to show it. 
# ----------