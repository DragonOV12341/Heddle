# Heddle 项目总结

## 项目定位

Heddle 是一个面向 NVIDIA Hopper (H100+) GPU 的**编译器插件**，为 [TileLang](https://github.com/tile-ai/tilelang) 提供自动异步流水线调度优化。它以 Python 运行时 monkey-patch 的方式注入 TileLang，无需修改 TileLang 源码即可生效。

项目属于学术研究性质（附带论文 `陈茜-论文.pdf`），目标是在 GPU kernel 编译阶段自动完成流水线标注、资源感知融合和调度优化，从而提升算子（特别是 FlashAttention）的执行性能。

## 核心架构

Heddle 由三个编译器模块组成：

| 模块 | 功能 | 关键技术 |
|------|------|---------|
| **AutoPiped** | 自动流水线标注（group/order/stage） | SMT 求解器推断最优流水线分组与阶段 |
| **AutoMixed** | 资源感知的图级融合与 tiling 决策 | 基于 ptxas 反馈的 occupancy 估算 |
| **Heddle Scheduler** | 联合 SWP+WS 调度 | CP-SAT (OR-Tools) 约束求解，支持 FineGrainedWS、PCWS、双消费者、TRWS |

## 工作原理

1. 用户调用 `heddle.init()` → 触发 `_monkey_patch.py`
2. Monkey-patch 做三件事：
   - 扩展 `PassConfigKey` 枚举，注入 30+ 个 Heddle 配置键
   - 拦截 `TVM PassContext`，将 Heddle 配置剥离后存入线程本地存储（避免 TVM C++ 侧报错）
   - 替换 `OptimizeForTarget` 编译阶段，在 warp-specialization 之前插入 `HeddleConsumerSchedule` pass
3. 用户正常使用 `@tilelang.jit(pass_configs={...})` 编写 kernel，Heddle pass 自动生效

## 目录结构

```
Heddle/
├── Heddle/                      # 项目主体
│   ├── heddle/                  # Python 包
│   │   ├── __init__.py          # 入口，暴露 init()
│   │   ├── _monkey_patch.py     # 运行时补丁（核心集成机制）
│   │   ├── scheduler/           # 调度器
│   │   │   ├── cp_sat.py        # CP-SAT 联合分区+调度求解器（UnifiedScheduler）
│   │   │   ├── cpsat_bridge.py  # CP-SAT 输出 → PCWS consumer ordering 转换
│   │   │   ├── cpsat_pipeline.py# 搜索→编译→基准 端到端流水线
│   │   │   └── smt.py           # Z3/SMT 求解器实现
│   │   ├── transform/           # 编译 pass
│   │   │   ├── heddle_consumer_schedule.py  # SMT consumer 重排 pass
│   │   │   ├── auto_tl_pipeline_smt.py      # AutoPiped SMT 辅助
│   │   │   └── persistent_kernel.py         # persistent kernel 注释器
│   │   ├── frontend/            # PyTorch 集成 (torch.compile/torch.fx)
│   │   │   ├── backend.py       # TileLang ↔ PyTorch 桥接
│   │   │   ├── patterns.py      # 算子模式识别 (FA/GEMM)
│   │   │   ├── codegen.py       # 自动 codegen
│   │   │   ├── automixed_*.py   # AutoMixed 决策与代码生成
│   │   │   └── partitioner.py   # 子图划分
│   │   ├── tileop/              # 算子模板
│   │   │   ├── flash_attention.py  # FlashAttention TileLang 模板
│   │   │   └── h2o_attention.py    # H2O Attention 模板
│   │   └── tools/
│   │       └── ptxas.py         # ptxas 日志解析与 occupancy 估算
│   ├── patches/cpp/             # C++ 后端补丁（仅用于旧版 TileLang v0.1.8）
│   ├── tests/                   # 性能测试（GEMM、FlashAttention）
│   ├── pyproject.toml           # 包配置（依赖：ortools>=9.7, 可选 torch>=2.4）
│   └── install.sh               # 安装脚本
├── TODO.md                      # 待办事项
└── 陈茜-论文.pdf                # 相关论文
```

## 性能表现

在 NVIDIA H100 上的 FP16 TFLOPS 测试结果：

- **FlashAttention 前向**：相比原版 TileLang 几何平均提升 **7.5%**（最高 14%）
- **FlashAttention 反向**（Heddle+3Role）：相比 TileLang WS 几何平均提升 **25%**

增益主要来自：
- 前向：SMT consumer 重排 + CP-SAT 联合调度
- 反向：拆分 dKdV+dQ 设计 + 三角色 warp specialization

## 技术栈与依赖

- **语言**：Python（主体）+ C++（可选补丁）
- **核心依赖**：`ortools>=9.7`（Google OR-Tools，提供 CP-SAT 求解器）
- **可选依赖**：`torch>=2.4`（PyTorch 前端集成）
- **运行环境**：Python >= 3.9, CUDA >= 12.0, NVIDIA H100 (Hopper) 或更高
- **上游依赖**：TileLang 0.1.8+（基于 TVM 的 GPU kernel DSL）

## 关键设计决策

1. **Monkey-patch 而非 fork**：不修改 TileLang 源码，通过运行时补丁注入，保持与上游的兼容性
2. **双求解器策略**：SMT (Z3) 用于 consumer 重排，CP-SAT (OR-Tools) 用于联合分区调度
3. **两条后端路径**：Python-only 路径（兼容主分支 PCWS）和 C++ 替换路径（旧版 FineGrainedWS，仅限 v0.1.8）
4. **线程本地配置存储**：将 Heddle 配置从 TVM PassContext 中剥离，避免 C++ 侧未注册导致的错误
