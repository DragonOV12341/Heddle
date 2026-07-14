# TileLang 在 Heddle SMT 方案后的代码生成过程

本文按当前仓库代码整理：Heddle SMT 不直接生成 CUDA，而是在 TileLang lowering 中提前改写 pipeline loop，并把调度结果写成 loop annotation；后续 TileLang 的 `ProducerConsumerWarpSpecialized` / `FineGrainedWS` pass 读取这些 annotation，完成 producer/consumer warp-specialized TIR 重写；最后由 TileLang/TVM 的 CUDA codegen 把重写后的 device TIR 变成 CUDA source 或编译产物。

## 1. 总体链路

主要路径如下：

```text
TileLang Python kernel
  -> tilelang.jit.JITKernel._compile()
  -> tilelang.lower()
  -> LowerAndLegalize()
  -> OptimizeForTarget()
       -> LowerSharedTmem()
       -> IfStmtBinding()
       -> MultiVersionBuffer()
            -> HeddleConsumerSchedule()   # Heddle monkey patch 插入
            -> original MultiVersionBuffer()
       -> LowerSharedBarrier()
       -> ProducerConsumerWarpSpecialized()
            -> FineGrainedWSRewriter::Substitute()
       -> FuseMBarrierArriveExpectTx()
       -> LowerOpaqueBlock()
       -> RewriteWgmmaSync()
       -> FlattenBuffer / StorageRewrite / Unroll / Simplify ...
  -> split host/device module
  -> device_codegen_without_compile() 或 device_codegen()
  -> target.build.tilelang_cuda_without_compile / target.build.tilelang_cuda
  -> CUDA source / compiled module
```

对应源码：

- `tilelang/tilelang/jit/kernel.py`：`JITKernel._compile()` 在 `PassContext(config=pass_configs)` 内调用 `tilelang.lower()`。
- `tilelang/tilelang/engine/lower.py`：`lower()` 先 `LowerAndLegalize()`，再 `OptimizeForTarget()`，然后拆出 host/device module 并进入 device codegen。
- `tilelang/tilelang/engine/phase.py`：`OptimizeForTarget()` 中的 TMA/WS 主 pass 顺序。
- `Heddle/heddle/_monkey_patch.py`：当前不是替换整个 `OptimizeForTarget()`，而是把 `HeddleConsumerSchedule()` 包到 `MultiVersionBuffer()` 前面。
- `Heddle/heddle/transform/heddle_consumer_schedule.py`：Heddle SMT 分析、重排和 annotation 注入。
- `tilelang/src/transform/finegrained_ws.cc`：`ProducerConsumerWarpSpecialized()` 的实际 C++ lowering 实现。

## 2. Heddle 如何插入 TileLang lowering

`Heddle/heddle/_monkey_patch.py::_patch_transform_init()` 会保存原始 `tilelang.transform.MultiVersionBuffer`，然后把它替换为：

```python
tvm.transform.Sequential([
    HeddleConsumerSchedule(),
    transform_mod._old_mvb(),
])
```

因此当前 Heddle pass 的实际位置是：

```text
LowerSharedTmem
IfStmtBinding
HeddleConsumerSchedule
MultiVersionBuffer
LowerSharedBarrier
ProducerConsumerWarpSpecialized
...
```

这点和旧注释里“wrap PCWS”略有差异：现在 live code 是放在 MVB 前，目的也是代码注释里写的“排除用户 numstage 对 IR 结构的影响”。

启用条件在 `HeddleConsumerSchedule()` pass 里：

- `tl.enable_heddle_consumer_schedule=True` 时运行。
- 其他相关 config 包括：
  - `tl.heddle_use_precise_latency`
  - `tl.heddle_use_alap_priority`
  - `tl.heddle_buffer_span_aware`
  - `tl.heddle_relax_producer_boundary`
  - `tl.heddle_use_phase_b`
  - `tl.heddle_consumer_num_warps`

## 3. Heddle pass 的输入 IR

Heddle pass 处理的是带 pipeline annotation 的 TIR loop。它会在 `_transform_pipeline_loop()` 中找到 pipeline loop，把 loop body 展开为 statement 列表，然后用 `_build_stmt_infos()` / `_detect_op_latency_and_resource()` 等 helper 提取：

- 每个 stmt 的原始 id：通常表示为 `s0`, `s1`, ...
- producer / consumer 分类。
- op latency、resource 类型、issue cycles。
- buffer read/write 区间。
- barrier/TMA/WGMMA 信息。
- 原始 `num_stages` 和 `threadIdx.x` extent。

这里的 stmt id 很重要：后续 `tl_pcws_warp_assigns` 仍然使用原始 flattened pipeline stmt id，而不是重排后的紧凑序号。

## 4. SMT 求解阶段

当前 joint SMT 入口是：

```python
_solve_smt_joint_optimize(
    op_deps,
    infos,
    all_indices,
    mod_sched_plan,
    kernel_num_threads=func_num_threads,
)
```

它的输入是 naive modulo schedule 的候选 plan：

- `M`：原始 op 到 modulo slot/start time 的映射。
- `I`：initiation interval。
- `L`：窗口长度。
- `heddle_expect_consumer_warps`：用户期望的 consumer warp 数。
- `heddle_original_num_stages`：原始 pipeline stage 数。

`_solve_smt_joint_optimize()` 会把 `_StmtInfo` 转成 `HeddleScheduler` 的 `OpNode` / `OutputValue`，然后调用 joint scheduler 同时求：

- `schedule`：每个 child op 的时间。
- `warp_assign`：每个 child op 的 logical warp id。
- FU capacity / issue resource。
- memory capacity 的固定解复核。
- `reg_peak` / `smem_peak` / lifetime 信息。

需要注意 warp 数的单位：

- `mod_sched_plan['num_warps'] = expect_consumer_warps` 先记录 consumer-only warp 数。
- 进入 SMT 时，代码使用 `num_warps = consumer_warps + 4`，因为 producer TMA warpgroup 也要在 SMT 域里保留。
- solver 输出的 `warp_assign` 是 raw logical warp id，例如 warp 0..11。
- 后面写给 FineGrainedWS 的不是 raw warp id，而是 compact consumer warpgroup id。

求解结果会整理成 `optimized`：

```python
{
    "I": base_I,
    "L": optimized_L,
    "M": optimized_M,
    "expanded_M": expanded_M,
    "warp_assign": collapsed_stmt_warp_assign,
    "expanded_warp_assign": child_level_warp_assign,
    "reg_peak": ...,
    "smem_peak": ...,
    "variable_lifetimes": ...,
    "ordering": ...
}
```

## 5. SMT 结果如何写回 TIR

joint SMT 成功后，`_joint_result_to_phase_b_result()` 会把 `optimized` 转成 Phase B 结果，供 `_transform_pipeline_loop()` 继续使用。

写回分三类。

### 5.1 重排 loop body

如果 `phase_b_order` 或完整 stmt order 改变，Heddle 会调用：

- `_planned_reordered_stmt_order()`
- `_reorder_loop_body()`
- `_rewrap_body()`

把 consumer statement 按 SMT/Phase B 的顺序重排。PCWS 后续看到的就是已经重排过的 loop body。

### 5.2 写 pipeline 调度 annotation

如果有 joint SMT source，会调用 `_build_joint_pipeline_annotations()` 注入 `tl_pipeline_*` 一类 annotation，用于把 joint schedule 的 stage/order 信息传给后续 pass。

如果 joint 推导出的 stage 数不同，还会直接改：

```python
new_annotations["num_stages"] = tvm.tir.IntImm("int32", phase_b_num_stages)
```

### 5.3 写 PCWS/FineGrainedWS annotation

Heddle 用 `_set_ws_annotation()` 同时写两套 key：

- 主 key：`tl_pcws_*`
- 兼容 key：`tl_finegrainedws_*`

重要 annotation：

- `tl_pcws_barrier_hints`
  - 格式：`buffer:wait=W,arrive=A;...`
  - 用来提示 PCWS forward-wait / backpressure-arrive 位置。
- `tl_pcws_stage_offsets`
  - 格式：`stmt_idx=offset,...`
  - 用来表达跨 stage 调整。
- `tl_pcws_dual_consumer`
  - 自动启用 dual-consumer split。
- `tl_pcws_dual_consumer_split`
  - 指定 dual-consumer 的 split stmt。
- `tl_pcws_three_role`
  - 检测到 TMA reduce-add 时启用 third role。
- `tl_pcws_warp_assigns`
  - 格式：`s0:0,s1:1,s2:0,...`
  - per-op warpgroup dispatch 的核心输入。

`tl_pcws_warp_assigns` 的生成有一个关键转换：

```text
solver raw logical warp id
  -> raw warpgroup = raw_warp // 4
  -> compact consumer warpgroup id = 0,1,...
  -> annotation: s{original_stmt_id}:{compact_wg}
```

因此，文档/日志中看到的 `---warp_assign={'s6__u1': 3, ...}` 和最终 TIR annotation `s6:0` / `s6:1` 不是同一个单位。

## 6. PCWS/FineGrainedWS 如何消费 annotation

TileLang Python 侧 `tilelang.transform.ProducerConsumerWarpSpecialized()` 只是 FFI 包装；C++ 侧当前实现是：

```cpp
tvm::transform::Pass ProducerConsumerWarpSpecialized() {
  return MakeFineGrainedWarpSpecializedPass("tl.ProducerConsumerWarpSpecialized");
}
```

也就是说 PCWS 实际进入 `FineGrainedWSRewriter::Substitute()`。

在 `finegrained_ws.cc` 中，rewriter 会：

1. 找到 `threadIdx.x` 的 `AttrStmt`，记录原始 thread extent。
2. 找到带 `num_stages` 的 pipeline loop。
3. 从 loop annotation 读取：
   - `tl_finegrainedws_barrier_hints`
   - `tl_finegrainedws_stage_offsets`
   - `tl_finegrainedws_dual_consumer`
   - `tl_finegrainedws_dual_consumer_split`
   - `tl_finegrainedws_three_role`
   - `tl_finegrainedws_warp_assigns`
4. 如果没有 `tl_finegrainedws_warp_assigns`，再 fallback 读取：
   - `tl_pcws_warp_assigns`
5. 展开 pipeline loop，提取 producer blocks 和 consumer compute stmts。
6. 根据 barrier hint 重建 mbarrier wait/arrive。
7. 根据 warp assignment 做 per-op dispatch 或 fallback 到 standard two-role / dual-consumer。

## 7. per-op warp dispatch 的代码生成形态

当 `warp_assigns_map_` 非空且覆盖所有 consumer stmt 时，会启用 Plan B per-op dispatch。

C++ 侧先把每个 compute stmt 映射到 warpgroup：

```cpp
original_stmt_index = extractor.compute_stmt_indices[ci];
wg = warp_assigns_map_[original_stmt_index];
```

这里再次强调：查表 key 是原始 flattened stmt id。`compute_stmts` 已经移除了 async producer/wait pair，因此不能用 `ci` 直接查 `s{ci}`。

随后 FineGrainedWS 构造每个 WG 的 loop body：

```text
WG0 body: assigned to group 0 的 stmts
WG1 body: assigned to group 1 的 stmts
...
common stmt: wg=-1 时复制到所有 WG
producer loop: 放到 consumer thread extent 之后
```

线程布局为：

```text
WG0:      threadIdx.x in [0, 128)
WG1:      threadIdx.x in [128, 256)
...
WG(N-1):  threadIdx.x in [(N-1)*128, N*128)
Producer: threadIdx.x in [N*128, (N+1)*128)
```

生成的 TIR 结构大致是：

```text
if threadIdx.x >= total_consumer_threads:
    producer_loop(threadIdx.x - total_consumer_threads)
else:
    if threadIdx.x < 128:
        WG0_consumer_loop(threadIdx.x)
    elif threadIdx.x < 256:
        WG1_consumer_loop(threadIdx.x - 128)
    ...
```

同时：

- WG0 会把原 consumer barrier thread count rewrite 到 128。
- WG1.. 会做 `threadIdx.x - w*128` 替换，并对 barrier id 做 offset。
- producer loop 会把 `threadIdx.x` 减去 `total_consumer_threads`。
- 最终 `ws_consumer_thread_extent = num_warp_groups * 128`。
- block 总 thread extent 会变成 consumer WGs 加 producer WG。

如果 `num_warp_groups < 2`，会 fallback 到 standard two-role。

## 8. 标准 two-role / dual-consumer fallback

当没有完整 `tl_pcws_warp_assigns`，或者只有一个 consumer WG 时，FineGrainedWS 会走 fallback：

- standard two-role：
  - consumer 使用原 consumer thread extent。
  - producer 使用额外 producer thread extent，通常是 128。
  - 最终 thread extent = consumer + producer。
- dual-consumer：
  - 根据 split index 把 consumer stmts 分成两个结构化 group。
  - 更偏固定 pattern，不是 SMT per-op 任意分配。
- three-role：
  - 用于 TMA reduce-add 等场景，把额外 writer role 抽出来。

这也是为什么日志里要区分：

- `FineGrainedWS: parsed ... per-op warp assignments`
- `FineGrainedWS per-op dispatch: N warp groups`
- `FineGrainedWS per-op dispatch: only 1 warp group, falling back`

只有看到 per-op dispatch 的多 WG 日志，才说明 SMT 的 per-op warp map 真正在 PCWS/FineGrainedWS 代码生成中生效。

## 9. 进入 CUDA codegen

PCWS/FineGrainedWS 完成后，`OptimizeForTarget()` 继续执行：

- `FuseMBarrierArriveExpectTx()`
- `LowerOpaqueBlock()`
- `RewriteWgmmaSync()` on Hopper
- `FlattenBuffer()`
- `StorageRewrite()`
- `UnrollLoop()`
- `Simplify()`
- `LowerDeviceStorageAccessInfo()`
- `LowerIntrin()`
- `HoistBroadcastValues()`

之后 `lower()` 分离 host/device module：

```python
host_mod = tir.transform.Filter(_is_host_call)(mod)
device_mod = tir.transform.Filter(_is_device_call)(mod)
```

device 侧：

- `enable_device_compile=False` 时调用 `device_codegen_without_compile()`。
  - CUDA target 会走 `target.build.tilelang_cuda_without_compile`。
  - 返回的 `CompiledArtifact.src` 是 `codegen_mod.inspect_source()`。
- `enable_device_compile=True` 时调用 `device_codegen()`。
  - CUDA target 会走 `target.build.tilelang_cuda`。
  - 通常 JIT 的 `tvm_ffi` backend 会启用 host/device compile。

因此最终 CUDA 中看到的 warp-specialized 结构，并不是 SMT 直接打印出来的，而是：

```text
SMT result
  -> TIR loop reorder + tl_pcws_* annotations
  -> FineGrainedWS producer/consumer split + per-WG dispatch TIR
  -> TileLang CUDA codegen
```

## 10. 调试时建议看的关键输出

Python/Heddle 侧：

- `[Heddle] Pass entry: enable=True`
- `----start _solve_smt_joint_optimize`
- `Phase B (CP-SAT) succeeded`
- `---warp_assign=...`
- `---optimized_L=...`
- `---optimized_M=...`
- `[Heddle] Using joint SMT schedule for PCWS lowering`
- `[Heddle] Injected joint SMT tl_pipeline_* annotations`
- `[Heddle] Injected per-op warp assigns: s...`

C++/TileLang 侧：

- `FineGrainedWS: parsed ... per-op warp assignments from annotation`
- `FineGrainedWS: parsed ... per-op warp assignments from pcws annotation`
- `FineGrainedWS per-op dispatch: N warp groups`
- `FineGrainedWS per-op dispatch: only 1 warp group, falling back`
- `FineGrainedWS dual-consumer: split at stmt ...`

CUDA/TIR 侧：

- `__launch_bounds__(...)` 是否反映 consumer WGs + producer WG。
- `threadIdx.x >= total_consumer_threads` 是否出现 producer branch。
- consumer branch 是否按 `threadIdx.x < 128`, `< 256`, ... 分 WG。
- WGMMA/TMA/barrier 是否仍位于预期 role。

## 11. 最容易混淆的几点

1. `T.Kernel(threads=N)` 不等于最终 CUDA block 总线程数。
   在 WS 路径里，FineGrainedWS 会在 consumer extent 之外加 producer extent；per-op dispatch 时还可能把 consumer extent 扩成 `num_warp_groups * 128`。

2. SMT 的 `warp_assign` 是 raw logical warp id。
   写入 `tl_pcws_warp_assigns` 前会先做 `warp // 4`，再 compact 成 consumer WG id。

3. `tl_pcws_warp_assigns` 的 key 必须是原始 flattened stmt id。
   C++ lookup 使用 `extractor.compute_stmt_indices[ci]` 回到原始 stmt id 后查表。

4. Heddle 当前插入点是 `MultiVersionBuffer()` 前。
   不是整替 `OptimizeForTarget()`，也不是当前 live code 中直接 wrap `ProducerConsumerWarpSpecialized()`。

5. 看到 Python SMT 成功不代表 CUDA per-op dispatch 一定成功。
   还需要确认 C++ 日志里真的进入 `FineGrainedWS per-op dispatch: N warp groups`，以及生成 CUDA 的 thread dispatch 结构正确。

