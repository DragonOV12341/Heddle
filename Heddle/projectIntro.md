# Heddle 项目简介

下面为 `Heddle/Heddle` 目录下主要文件与子目录的简要说明，方便快速定位代码职责。

## 顶层文件
- `README.md`：项目概览、快速上手、性能对比与包结构说明。
- `README.zh.md`：中文说明（翻译）。
- `install.sh`：自动化安装脚本；可选地把 `patches/cpp/` 的 C++ 补丁应用到 TileLang 源码并重建。
- `pyproject.toml`：打包与依赖元数据。
- `patches/`：可选的 C++ 替换/补丁，`patches/cpp/` 为对 TileLang v0.1.8 的完整后端替代实现（参见 `patches/cpp/README.md`）。
- `tests/`：测试与基准脚本（FlashAttention、分区/编译/基准流水线样例）。

## Python 包：`heddle/`

- 包入口：`heddle/__init__.py` — 暴露 `init()`，运行后会应用运行时 monkey-patch 将 Heddle 功能整合进 TileLang（调用 `heddle._monkey_patch.apply_all_patches()`）。
- 运行时补丁：`heddle/_monkey_patch.py` —
  - 扩展 `PassConfigKey` 以接受 Heddle 的自定义 pass 配置键；
  - 修补 TVM `PassContext` 以在 Python 侧透传/提取 Heddle 配置；
  - 替换 `tilelang.engine.phase.OptimizeForTarget` 在合适时插入 `HeddleConsumerSchedule`、persistent kernel 等变换；
  - 把 Heddle 的 pass 构造器注入到 `tilelang.transform` 命名空间。

### 子包职责（关键模块）

- `frontend/`：编译器前端与 PyTorch 集成（`torch.compile` / `torch.fx`），负责把高阶图转换为 TileLang kernel 模板或通过自动 codegen 生成 TIR。主要文件：`backend.py`、`patterns.py`、`codegen.py`。

- `scheduler/`：调度器与搜索逻辑，包含 CP-SAT 与 SMT 求解器实现与桥接代码：
  - `cp_sat.py`：基于 OR-Tools CP-SAT 的联合分区 + 调度求解器（UnifiedScheduler 等），用于联合决定分区、consumer 排序与寄存器活跃峰值等。
  - `cpsat_bridge.py`：将 CP-SAT 输出转换为 PCWS/Heddle consumer ordering，并提供基准辅助函数。
  - `cpsat_pipeline.py`：CP-SAT → 编译 → 基准的一站式流水线，用于自动搜索/评估候选配置。
  - `smt.py`：（SMT 求解器实现，供 transform 使用）

- `transform/`：Heddle 的 TileLang Python pass 实现：
  - `heddle_consumer_schedule.py`：在 PCWS 之前运行的 SMT 消费者重排 pass（ALAP/slack、buffer-span-aware 优化、阶段/屏障提示等）。
  - `auto_tl_pipeline_smt.py`：AutoPiped / pipeline SMT 相关辅助与推断逻辑。
  - `persistent_kernel.py`：把普通 kernel 注释/转换为 persistent kernel 的 pass（动态 tile 调度 wrapper 所需元数据）。

- `tileop/`：tile-op 级模板与示例（对高性能算子如 FlashAttention 的 tile-level 实现）：
  - `flash_attention.py`、`h2o_attention.py`：针对 FA/H2O 的 TileLang 模板与 autotune 注释。

- `tools/`：实用工具与性能反馈桥接：
  - `ptxas.py`：解析 `ptxas --verbose` 输出、提取 regs/spill/smem 等信息并提供轻量 occupancy 估算，用于 AutoMixed/调度回退或决策闭环。

## C++ 补丁（`patches/cpp/`）
- 提供对 TileLang v0.1.8 的完整后端替换，实现 FineGrainedWS、双消费者、三角色 WS、屏障提示、async WGMMA 等。详见 `patches/cpp/README.md`。该路径需要把补丁复制到 TileLang 源码并重建。

## 快速使用说明
- 在使用端代码中调用：

  ```py
  import heddle
  heddle.init()  # 运行时打补丁并启用 Heddle pass
  ```

- 通过 `tilelang.transform.PassConfigKey` 设置开关（示例见 `README.md` 中的 `pass_configs`）：开启 AutoPiped、Heddle consumer scheduling、Phase B，或开启精确延迟/ALAP 优先级等。

## 建议的下一步
- 如果需要，我可以把每个子目录内的关键实现文件再逐个列出并摘录文件头的 docstring /注释，生成更详细的 API 索引。

## 子目录关键文件（逐项）

### `heddle/frontend/`
- `backend.py`：TileLang ↔ PyTorch frontend 的桥接与编译后端入口。
- `patterns.py`：算子模式识别（FlashAttention/GEMM 等）并替换为高性能 tile-op 模板。
- `codegen.py`：Elementwise / 子图的自动 codegen，生成可编译的 TileLang TIR。
- `automixed_codegen.py`、`automixed_decision.py`：AutoMixed 的决策与 codegen 支撑（资源感知的融合/划分）。
- `partitioner.py`、`subgraph.py`：子图划分与分区逻辑。

### `heddle/scheduler/`
- `cp_sat.py`：UnifiedScheduler（OR-Tools CP-SAT）——联合分区与调度求解器及数据类型定义（`KernelSpec`/`OpSpec`）。
- `cpsat_bridge.py`：将 CP-SAT 输出转换为 PCWS consumer ordering，并包含基准工具（`benchmark_partition`）。
- `cpsat_pipeline.py`：端到端搜索/编译/基准流水线（`run_pipeline`、`AutoSearch` 风格逻辑）。
- `smt.py`：基于 Z3/SMT 的调度求解器实现（供 transform 中 SMT 路径使用）。

### `heddle/transform/`
- `heddle_consumer_schedule.py`：SMT-based consumer reordering pass，核心调度策略与 barrier hint 协议。
- `auto_tl_pipeline_smt.py`：AutoPiped 的 SMT 辅助函数、override 解析与语义 stmt→stage 计算。
- `persistent_kernel.py`：persistent kernel 注释器（生成持久化执行所需元数据）。

### `heddle/tileop/`
- `flash_attention.py`：FlashAttention 的 TileLang 模板与 `@autotune` 注解示例。
- `h2o_attention.py`：H2O 风格 attention 的模板实现。

### `heddle/tools/`
- `ptxas.py`：ptxas 日志解析、`PtxasStats` 与 `OccupancyProxy`，用于提取 regs/spill/smem 并估算占用率。

### `patches/cpp/`
- `README.md`：如何把 C++ 补丁应用到 TileLang v0.1.8 的说明。
- `src/`：包含 C++ 源文件（如 `transform/finegrained_ws.cc`、`op/builtin.*` 等）用于替换 TileLang 后端实现。

### `tests/`
- `test_perf.py`、`batchGEMM.py` 等：性能/基准脚本与样例，便于复现实验表格中的结果。

（已将以上列表写入此文件）
