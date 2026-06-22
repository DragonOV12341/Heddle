# -*- coding: utf-8 -*-
# 注释版说明：这段 TIR/TileLang 代码实现的是类似 FlashAttention forward 的一个 kernel。
# 总体计算：对每个 batch/head/block，计算 O = softmax(Q @ K^T / sqrt(d)) @ V。
# 核心硬件路径：TMA 负责 global->shared 异步搬运，mbarrier 负责等待 TMA 完成，WGMMA 负责 warp-group 矩阵乘。
# 注意：这里 metadata['tl.Layout']/metadata['tl.Fragment'] 在原始脚本末尾被省略，因此本文件主要用于阅读理解。
# [d] ---- func before SMT ----- # 
# 导入 TVM TensorIR 脚本 DSL，下面的 T.* 都是 TIR/TileLang 层面的 IR 构造或 intrinsic。
from tvm.script import tir as T
# 声明一个 TIR PrimFunc；编译器会把它当作底层 CUDA kernel 生成。
@T.prim_func
# main 的四个 handle 分别对应 Q/K/V/O 的原始指针。
def main(Q_handle: T.handle, K__handle: T.handle, V_handle: T.handle, O_handle: T.handle):
    # 设置编译目标：NVIDIA sm_90a，也就是 H100/Hopper 架构；warp size=32，block 最大线程数=1024。
    T.func_attr({"target": T.target({"arch": "sm_90a", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    # 把 Q_handle 绑定成 4D buffer：形状 [B=1, Seq=4096, Head=32, Dim=128]，元素 fp16。
    Q = T.match_buffer(Q_handle, (1, 4096, 32, 128), "float16", strides=(16777216, 4096, 128, 1))
    # 把 K_handle 绑定成同样形状的 K buffer。这里变量名 K_ 是为了避开潜在命名冲突。
    K_ = T.match_buffer(K__handle, (1, 4096, 32, 128), "float16", strides=(16777216, 4096, 128, 1))
    # 把 V_handle 绑定成同样形状的 V buffer。
    V = T.match_buffer(V_handle, (1, 4096, 32, 128), "float16", strides=(16777216, 4096, 128, 1))
    # 把 O_handle 绑定成输出 O buffer，形状同 Q/V。
    O = T.match_buffer(O_handle, (1, 4096, 32, 128), "float16", strides=(16777216, 4096, 128, 1))
    # 创建 5 个 mbarrier：barrier0 给 Q 的 TMA，barrier1/2 给双缓冲 K，barrier3/4 给双缓冲 V。
    T.create_list_of_mbarrier(1, 1, 1, 1, 1)
    # root block：TIR 顶层 block，主要用来承载 buffer、thread binding、layout metadata。
    with T.block("root"):
        # 声明 root block 读集合。这里为空，实际读写在内层 tilelang_root 里描述。
        T.reads()
        # 声明 root block 写集合。这里为空。
        T.writes()
        # Qs：Q 的 shared memory staging buffer。大小 2*16*512 fp16 = 32768 bytes，对应一个 128x128 Q tile 的搬运空间。
        Qs = T.Buffer((2, 16, 512), "float16", scope="shared.dyn")
        # Ks：K 的 shared memory staging buffer。这里后面会以双缓冲形式重解释为 Ks_1。
        Ks = T.Buffer((2, 8, 512), "float16", scope="shared.dyn")
        # smp：previous softmax max，保存上一轮 K-block 处理完后的每行最大值 m_old。
        smp = T.Buffer((4,), scope="local")
        # sm：current softmax max，保存当前合并后的每行最大值 m_new。
        sm = T.Buffer((4,), scope="local")
        # ss：softmax 重标定因子，等价于 exp(m_old - m_new)，用于缩放旧的 acc_o 和 ls。
        ss = T.Buffer((4,), scope="local")
        # ssum：当前 K-block 上 exp(score-m_new) 的行和。
        ssum = T.Buffer((4,), scope="local")
        # acc_s：score/probability accumulator，先存 QK^T 的 fp32 分数，随后原地变成 exp 后的 softmax 分子。
        acc_s = T.Buffer((64,), scope="local")
        # acc_s_c：把 acc_s 转为 fp16，作为第二个 WGMMA 的 A 操作数，即 P 矩阵。
        acc_s_c = T.Buffer((64,), "float16", scope="local")
        # Vs：V 的 shared memory staging buffer。后面会以双缓冲形式重解释为 Vs_1。
        Vs = T.Buffer((2, 8, 512), "float16", scope="shared.dyn")
        # ls：online softmax 的累计归一化分母 l。
        ls = T.Buffer((4,), scope="local")
        # acc_o：输出 accumulator，fp32 累计 softmax(QK^T) @ V 的结果。
        acc_o = T.Buffer((128,), scope="local")
        # layout_map：告诉 TileLang 每个 buffer/fragment 的物理 layout，影响 shared memory swizzle、fragment 分布和 WGMMA 读写方式。
        T.block_attr({"layout_map": {Qs: metadata["tl.Layout"][0], Ks: metadata["tl.Layout"][1], smp: metadata["tl.Fragment"][0], sm: metadata["tl.Fragment"][1], ss: metadata["tl.Fragment"][2], ssum: metadata["tl.Fragment"][3], acc_s: metadata["tl.Fragment"][4], acc_s_c: metadata["tl.Fragment"][5], Vs: metadata["tl.Layout"][2], ls: metadata["tl.Fragment"][6], acc_o: metadata["tl.Fragment"][7]}})
        # blockIdx.x：序列维度上的 query block id。每个 block 处理 128 个 query token。
        bx = T.launch_thread("blockIdx.x", 32)
        # blockIdx.y：head id。这里一个 CUDA block 处理一个 attention head。
        by = T.launch_thread("blockIdx.y", 32)
        # blockIdx.z：batch 或额外网格维度，这里只有 1。
        bz = T.launch_thread("blockIdx.z", 1)
        # threadIdx.x：每个 CTA 使用 128 个线程，也就是 4 个 warp = 1 个 warpgroup，正好服务 WGMMA。
        tx = T.launch_thread("threadIdx.x", 128)
        # threadIdx.y：未使用的线程维度。
        ty = T.launch_thread("threadIdx.y", 1)
        # threadIdx.z：未使用的线程维度。
        tz = T.launch_thread("threadIdx.z", 1)
        # tilelang_root：真正的计算 block，从这里开始是 kernel 主体。
        with T.block("tilelang_root"):
            # 声明本 block 理论读到的 global memory 范围：当前 Q block、所有历史/可见 K/V block，以及输出 O 的位置。
            T.reads(Q[0, bx * 128, by, 0], K_[0, 0:4033, by, 0], V[0, 0:4033, by, 0], O[0, bx * 128, by, 0])
            # 声明写集合。这里为空是因为后续写回通过具体 store 表达。
            T.writes()
            # 再次绑定 layout metadata，保证内层 block 的 buffer layout 信息完整。
            T.block_attr({"layout_map": {Qs: metadata["tl.Layout"][0], Ks: metadata["tl.Layout"][1], smp: metadata["tl.Fragment"][0], sm: metadata["tl.Fragment"][1], ss: metadata["tl.Fragment"][2], ssum: metadata["tl.Fragment"][3], acc_s: metadata["tl.Fragment"][4], acc_s_c: metadata["tl.Fragment"][5], Vs: metadata["tl.Layout"][2], ls: metadata["tl.Fragment"][6], acc_o: metadata["tl.Fragment"][7]}})
            # 为 Qs 分配/绑定 shared.dyn buffer；data=Qs.data 表示沿用前面声明的共享内存区域。
            Qs = T.alloc_buffer((2, 16, 512), "float16", data=Qs.data, scope="shared.dyn")
            # 把 Ks 重解释为 4D 双缓冲：[stage=2, inner=2, tile_maybe=8, 512]。k%2 会在两个 stage 间 ping-pong。
            Ks_1 = T.alloc_buffer((2, 2, 8, 512), "float16", data=Ks.data, scope="shared.dyn")
            # 把 Vs 重解释为 4D 双缓冲，结构类似 Ks。
            Vs_1 = T.alloc_buffer((2, 2, 8, 512), "float16", data=Vs.data, scope="shared.dyn")
            # 局部寄存器/fragment：acc_s 保存 QK^T 或 softmax 分子。
            acc_s = T.alloc_buffer((64,), data=acc_s.data, scope="local")
            # 局部 fp16 fragment：acc_s_c 给 WGMMA rs 路径使用。
            acc_s_c = T.alloc_buffer((64,), "float16", data=acc_s_c.data, scope="local")
            # 局部 fp32 fragment：acc_o 保存最终输出的累加结果。
            acc_o = T.alloc_buffer((128,), data=acc_o.data, scope="local")
            # 局部 fragment：sm 当前 max。
            sm = T.alloc_buffer((4,), data=sm.data, scope="local")
            # 局部 fragment：smp 上一轮 max。
            smp = T.alloc_buffer((4,), data=smp.data, scope="local")
            # 局部 fragment：ss 重标定 scale。
            ss = T.alloc_buffer((4,), data=ss.data, scope="local")
            # 局部 fragment：ssum 当前 block 的 softmax 分母贡献。
            ssum = T.alloc_buffer((4,), data=ssum.data, scope="local")
            # 局部 fragment：ls 累计 softmax 分母。
            ls = T.alloc_buffer((4,), data=ls.data, scope="local")
            # 给 Qs.data 标注：接下来这段是 TMA 写 shared buffer 的区域。
            with T.attr(Qs.data, "tl.tma_copy_write_buffer", 1):
                if tx == 0:
                    # thread0 设置 barrier0 期望接收 32768 bytes，也就是 Q tile 的 TMA 搬运字节数。
                    T.mbarrier_expect_tx(T.get_mbarrier(0), 32768)
                if tx == 0:
                    # unroll(2)：把 Q tile 拆成两个 TMA transaction/子块加载。
                    for i in T.unroll(2):
                        # tma_load：用 TMA 从 global Q 搬到 shared Qs。
                        # create_tma_descriptor 描述 global tensor 的 rank、stride、box shape 等；tvm_access_ptr 指向 shared memory 目标地址。
                        # 坐标里的 by 是 head，bx*128 是当前 query block 起点，i*64 表示分两块搬。
                        T.tma_load(T.create_tma_descriptor(6, 4, Q.data, 128, 32, 4096, 1, T.int64(2), T.int64(256), T.int64(8192), T.int64(33554432), 64, 1, 128, 1, 1, 1, 1, 1, 0, 3, 2, 0), T.get_mbarrier(0), T.tvm_access_ptr(T.type_annotation("float16"), Qs.data, i * 8192, 8192, 2), i * 64, by, bx * 128, 0, 0)
                if tx == 0:
                    # TMA 发起后 arrive barrier，表示 producer 已经提交这组异步 copy。
                    T.ptx_arrive_barrier(T.get_mbarrier(0))
            # 等待 barrier0 的 parity=0，确保 Qs 中的 Q tile 已经可被 WGMMA 读取。
            T.mbarrier_wait_parity(T.get_mbarrier(0), 0)
            # 初始化输出 accumulator acc_o。unroll(32)*vectorized(4)=128 个 fp32 元素。
            for i in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                # vectorized(4)：向量化写 4 个连续 fragment 元素。
                for vec in T.vectorized(4):
                    # acc_o 清零；后面会不断累加 P@V。
                    acc_o[i * 4 + vec] = T.float32(0.0)
            # 初始化 online softmax 分母 ls。
            for i in T.vectorized(4):
                # ls 初始为 0。
                ls[i] = T.float32(0.0)
            # 初始化 online softmax 最大值 sm。
            for i in T.vectorized(4):
                # sm 初始为 -inf，用于后续 max(score)。
                sm[i] = T.infinity("float32") * T.float32(-1.0)
            # 主循环 k：遍历 4096/64=64 个 K/V block；每轮处理 64 个 key/value token。num_stages=2 表示软件流水/双缓冲意图。
            for k in T.serial(64, annotations={"num_stages": 2}):
                # 标注 Ks.data 是 TMA 写 shared buffer 的目的区域。
                with T.attr(Ks.data, "tl.tma_copy_write_buffer", 1):
                    if tx == 0:
                        # 设置 K 的 TMA barrier：k%2+1 在 barrier1 和 barrier2 间切换，对应 K 的双缓冲 stage。期望 16384 bytes。
                        T.mbarrier_expect_tx(T.get_mbarrier(k % 2 + 1), 16384)
                    if tx == 0:
                        # K tile 也拆成两个 TMA 子块搬。
                        for i in T.unroll(2):
                            # 从 global K_ 搬当前 k-block 到 shared Ks。
                            # k*64 是 key 序列起点；k%2*8192 选择双缓冲 stage；i*4096 选择子块偏移。
                            T.tma_load(T.create_tma_descriptor(6, 4, K_.data, 128, 32, 4096, 1, T.int64(2), T.int64(256), T.int64(8192), T.int64(33554432), 64, 1, 64, 1, 1, 1, 1, 1, 0, 3, 2, 0), T.get_mbarrier(k % 2 + 1), T.tvm_access_ptr(T.type_annotation("float16"), Ks.data, i * 4096 + k % 2 * 8192, 4096, 2), i * 64, by, k * 64, 0, 0)
                    if tx == 0:
                        # K 的 TMA producer arrive，表示该轮 K copy 已提交。
                        T.ptx_arrive_barrier(T.get_mbarrier(k % 2 + 1))
                # 等待当前 K stage 的 TMA 完成；parity 用 k%4//2 在重复使用同一个 mbarrier 时区分轮次。
                T.mbarrier_wait_parity(T.get_mbarrier(k % 2 + 1), k % 4 // 2)
                # 保存上一轮 softmax max。
                for i in T.vectorized(4):
                    # smp = sm，也就是 m_old。
                    smp[i] = sm[i]
                # 准备重新计算当前 block 的局部 max。
                for i in T.vectorized(4):
                    # sm 先重置为 -inf，后面会用当前 QK^T block 的最大值更新。
                    sm[i] = T.infinity("float32") * T.float32(-1.0)
                # 清零 score accumulator acc_s；每轮 K-block 都要重新计算 QK^T。
                for i in T.unroll(16, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    for vec in T.vectorized(4):
                        # acc_s 清零。
                        acc_s[i * 4 + vec] = T.float32(0.0)
                # _gemm_ssr：shared-shared-register 的 WGMMA，计算 score = Qs @ Ks^T，结果进 acc_s。
                with T.block("_gemm_ssr"):
                    # 这个内层 block 的显式 read 集合省略，由 WGMMA intrinsic 隐式读。
                    T.reads()
                    # 这个内层 block 的显式 write 集合省略，由 WGMMA intrinsic 隐式写 acc_s。
                    T.writes()
                    # desc_a：WGMMA A 操作数描述符，指向 shared memory 中的 Qs。
                    desc_a = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                    # desc_b：WGMMA B 操作数描述符，指向 shared memory 中的 Ks。
                    desc_b = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                    # 初始化 Qs 的 WGMMA descriptor；descriptor 编码 shared 地址、stride/layout、矩阵 tile 信息。
                    T.initialize_wgmma_descriptor(desc_a[0], T.tvm_access_ptr(T.type_annotation("float16"), Qs.data, 0, 16384, 1), 1, 1, 64)
                    # 初始化 Ks 的 WGMMA descriptor；k%2*8192 选择当前 K 双缓冲 stage。
                    T.initialize_wgmma_descriptor(desc_b[0], T.tvm_access_ptr(T.type_annotation("float16"), Ks.data, k % 2 * 8192, 8192, 1), 1, 1, 64)
                    # warpgroup_fence_operand：确保 acc_s 作为 WGMMA 输出 operand 的读写顺序正确，避免编译器/硬件重排。
                    T.warpgroup_fence_operand("float32", acc_s.data, 0, 64)
                    # warpgroup_arrive：当前 4 个 warp 组成的 warpgroup 到达 WGMMA 发射点。
                    T.warpgroup_arrive()
                    # warp_j 只有 1，保留循环结构用于统一生成多 warpgroup 代码。
                    for warp_j in T.unroll(1, annotations={"pragma_unroll_explicit": False}):
                        # i 遍历两个 m64 子 tile，对应 128 行 query 被拆成两个 64 行 WGMMA。
                        for i in T.unroll(2, annotations={"pragma_unroll_explicit": False}):
                            # ki 遍历 K 维上的 8 个 16-wide chunk：8*16=128，对应 head_dim=128。
                            for ki in T.unroll(8, annotations={"pragma_unroll_explicit": False}):
                                # ptx_wgmma_ss：Hopper WGMMA 指令，s/s 表示 A、B 都来自 shared memory。
                                # m64n64k16：一次做 64x64 输出 tile、K=16；fp16*fp16 累加到 fp32 acc_s。
                                # desc_a/desc_b 的 shift_right(...,4) 是 WGMMA descriptor 内部以 16B 对齐粒度编码地址偏移。
                                T.ptx_wgmma_ss("m64n64k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a.data, T.shift_right(ki // 4 * 16384 + i * 8192 + ki % 4 * 32, 4), desc_b.data, T.shift_right(ki // 4 * 8192 + ki % 4 * 32, 4), acc_s.data, i * 32, 1, 1, 1)
                    # commit_batch：提交刚刚发射的一批 WGMMA 指令。
                    T.warpgroup_commit_batch()
                    # warpgroup_wait(0)：等待所有已提交 WGMMA batch 完成，确保 acc_s 可读。
                    T.warpgroup_wait(0)
                    # WGMMA 完成后再次 fence acc_s，确保后续标量 softmax 逻辑看到正确结果。
                    T.warpgroup_fence_operand("float32", acc_s.data, 0, 64)
                # 分配 4 个 fp32 local 临时值，用于每个 fragment row-group 的局部 max reduce。
                with T.allocate([4], "float32", "local") as sm_clear:
                    # 遍历 4 个 softmax row-group。这里 fragment layout 把 128 行折叠成 4 组统计量。
                    for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        # 把 allocate 出来的指针包装成 T.Buffer，方便用下标访问。
                        sm_clear_1 = T.Buffer((4,), data=sm_clear, scope="local")
                        # 局部 max 初始化为 -inf。
                        sm_clear_1[i] = T.float32("-inf")
                        # 遍历当前 group 内的 16 个 score 元素，做局部最大值归约。
                        for rv in T.unroll(16, annotations={"pragma_unroll_explicit": T.bool(False)}):
                            # 从 acc_s 的 fragment layout 中取 score，更新局部 max。索引表达式是在解 WGMMA fragment 排布。
                            sm_clear_1[i] = T.max(sm_clear_1[i], acc_s[i // 2 * 32 + rv % 8 * 4 + i % 2 * 2 + rv // 8])
                        # 跨 128 线程做 AllReduce Max，得到整个 CTA/warpgroup 对应 row-group 的最大 score。
                        sm_clear_1[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run", sm_clear_1[i])
                        # sm 更新为当前 K-block 的局部最大值。
                        sm[i] = T.max(sm[i], sm_clear_1[i])
                # 把当前 block max 和历史 max 合并。
                for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    # sm = max(m_block, m_old)，这是 online softmax 的数值稳定关键。
                    sm[i] = T.max(sm[i], smp[i])
                # 计算旧累计结果的重标定系数。
                for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    # ss = exp2((m_old - m_new) * log2(e) / sqrt(128))。
                    # 0.1275174307460247 ≈ log2(e) / sqrt(128)，用 exp2 实现 softmax scaling。
                    ss[i] = T.exp2(smp[i] * T.float32(0.1275174307460247) - sm[i] * T.float32(0.1275174307460247))
                # 把 score 转换成 softmax 分子。
                for i in T.unroll(64, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    # acc_s = exp2((score - m_new) * log2(e)/sqrt(128))；此时 acc_s 不再是 score，而是概率分子 P。
                    acc_s[i] = T.exp2(acc_s[i] * T.float32(0.1275174307460247) - sm[i // 32 * 2 + i % 4 // 2] * T.float32(0.1275174307460247))
                # 计算当前 K-block 的 softmax 分母贡献。
                for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    # 当前 row-group 的分母贡献先清零。
                    ssum[i] = T.float32(0.0)
                    # 遍历 16 个概率分子元素并求和。
                    for rv in T.unroll(16, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        # 按 fragment layout 从 acc_s 取值累加到 ssum。
                        ssum[i] = ssum[i] + acc_s[i // 2 * 32 + rv % 8 * 4 + i % 2 * 2 + rv // 8]
                    # 跨线程 AllReduce Sum，得到完整 row-group 的 sum(exp(score-m_new))。
                    ssum[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run", ssum[i])
                # 旧输出累加值 acc_o 需要按照 m_old->m_new 的变化重新缩放。
                for i in T.unroll(128, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    # acc_o *= ss，对应 online softmax 公式里的旧贡献重标定。
                    acc_o[i] = acc_o[i] * ss[i // 64 * 2 + i % 4 // 2]
                # 更新累计分母 l。
                for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    # ls = ls_old * ss + ssum，也就是 online softmax 的分母更新。
                    ls[i] = ls[i] * ss[i] + ssum[i]
                # 准备把 fp32 概率分子 acc_s 转成 fp16，作为后续 P@V 的 WGMMA 输入。
                for i in T.unroll(16, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    for vec in T.vectorized(4):
                        # acc_s_c = fp16(acc_s)，并按 WGMMA rs 所需的 fragment 排布重排。
                        acc_s_c[i * 4 + vec] = T.Cast("float16", acc_s[i % 4 // 2 * 32 + i // 4 * 8 + i % 2 * 4 + vec])
                # 标注 Vs.data 是 TMA 写 shared buffer 的目的区域。
                with T.attr(Vs.data, "tl.tma_copy_write_buffer", 1):
                    if tx == 0:
                        # 设置 V 的 TMA barrier：barrier3/4 双缓冲，期望 16384 bytes。
                        T.mbarrier_expect_tx(T.get_mbarrier(k % 2 + 3), 16384)
                    if tx == 0:
                        # V tile 拆成两个 TMA 子块搬运。
                        for i in T.unroll(2):
                            # 从 global V 搬当前 k-block 到 shared Vs。
                            # 坐标与 K 类似：head=by，value 序列起点=k*64，stage=k%2。
                            T.tma_load(T.create_tma_descriptor(6, 4, V.data, 128, 32, 4096, 1, T.int64(2), T.int64(256), T.int64(8192), T.int64(33554432), 64, 1, 64, 1, 1, 1, 1, 1, 0, 3, 2, 0), T.get_mbarrier(k % 2 + 3), T.tvm_access_ptr(T.type_annotation("float16"), Vs.data, i * 4096 + k % 2 * 8192, 4096, 2), i * 64, by, k * 64, 0, 0)
                    if tx == 0:
                        # V 的 TMA producer arrive。
                        T.ptx_arrive_barrier(T.get_mbarrier(k % 2 + 3))
                # 等待当前 V stage 的 TMA 完成，确保 Vs 可被 WGMMA 读取。
                T.mbarrier_wait_parity(T.get_mbarrier(k % 2 + 3), k % 4 // 2)
                # _gemm_rsr：register-shared-register 的 WGMMA，计算 acc_o += P @ V。
                with T.block("_gemm_rsr"):
                    # 显式 read 集合省略，由 WGMMA intrinsic 隐式读 acc_s_c 和 Vs。
                    T.reads()
                    # 显式 write 集合省略，由 WGMMA intrinsic 隐式更新 acc_o。
                    T.writes()
                    # desc_b：V 在 shared memory 中的 WGMMA descriptor。A 操作数 P 来自寄存器 acc_s_c。
                    desc_b = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                    # 初始化 V 的 WGMMA descriptor；stride 参数和前面的 K 不同，因为这里做的是 P(64x64) @ V(64x128)。
                    T.initialize_wgmma_descriptor(desc_b[0], T.tvm_access_ptr(T.type_annotation("float16"), Vs.data, k % 2 * 8192, 8192, 1), 1, 512, 64)
                    # fence acc_s_c：确保寄存器中的 fp16 P fragment 已准备好给 WGMMA 读取。
                    T.warpgroup_fence_operand("float16", acc_s_c.data, 0, 32)
                    # fence acc_o：确保 WGMMA 对输出 accumulator 的累加顺序正确。
                    T.warpgroup_fence_operand("float32", acc_o.data, 0, 128)
                    # warpgroup_arrive：4 个 warp 同步进入第二个 WGMMA 发射段。
                    T.warpgroup_arrive()
                    # warp_j 仍为占位循环。
                    for warp_j in T.unroll(1, annotations={"pragma_unroll_explicit": False}):
                        # i 遍历两个 64-row 子 tile。
                        for i in T.unroll(2, annotations={"pragma_unroll_explicit": False}):
                            # ki 遍历 4 个 K=16 chunk：4*16=64，对应当前 K-block 的 64 个 token。
                            for ki in T.unroll(4, annotations={"pragma_unroll_explicit": False}):
                                # ptx_wgmma_rs：r/s 表示 A 来自 register fragment，B 来自 shared memory。
                                # m64n128k16：一次产生 64x128 输出 tile 的一部分；fp16*fp16 累加到 fp32 acc_o。
                                T.ptx_wgmma_rs("m64n128k16", T.bool(False), "fp16", "fp16", "fp32", acc_s_c.data, ki * 16 + i * 8, desc_b.data, T.shift_right(ki * 2048, 4), acc_o.data, i * 64, 1, 1, 1)
                    # 提交 P@V 的 WGMMA batch。
                    T.warpgroup_commit_batch()
                    # 等待 P@V WGMMA 完成。
                    T.warpgroup_wait(0)
                    # WGMMA 完成后 fence acc_o，后续可能继续缩放或最终写回。
                    T.warpgroup_fence_operand("float32", acc_o.data, 0, 128)
                    # fence acc_s_c，结束该 register operand 的使用区间。
                    T.warpgroup_fence_operand("float16", acc_s_c.data, 0, 32)
            # end for K 循环
            # 所有 K/V block 处理完后，进行最终 softmax 归一化。
            for i in T.unroll(128, annotations={"pragma_unroll_explicit": T.bool(False)}):
                # acc_o /= ls，把累计的 softmax 分子加权和除以累计分母，得到真正的 attention 输出。
                acc_o[i] = acc_o[i] / ls[i // 64 * 2 + i % 4 // 2]
            # 写回 O：遍历 acc_o fragment 中的 64 组，每组转成 2 个 fp16 元素。
            for i in T.unroll(64, annotations={"pragma_unroll_explicit": T.bool(False)}):
                # 声明一个 2 元素 fp16 local buffer，作为 fp32->fp16 cast 后的临时写回向量。
                O_local_cast = T.decl_buffer((2,), "float16", scope="local")
                # 向量化处理两个连续输出元素。
                for vec in T.vectorized(2):
                    # 把 fp32 acc_o cast 成 fp16。
                    O_local_cast[vec] = T.Cast("float16", acc_o[i * 2 + vec])
                # 向量化 store 两个 fp16 到 global O。
                for vec_copy in T.vectorized(2):
                    # 把 fragment 中的元素映射回 O[batch, seq, head, dim]。
                    # seq 索引由 bx*128、warp/lane id、fragment i 共同决定；dim 索引由 i 和 lane 内偏移决定。
                    O[0, bx * 128 + i // 32 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4, by, i % 32 // 2 * 8 + tx % 4 * 2 + vec_copy] = O_local_cast[vec_copy]

# 原始脚本省略了 metadata；如果需要完整可编译 IR，需要用 script(show_meta=True) 导出 metadata。
# Metadata omitted. Use show_meta=True in script() method to show it.
