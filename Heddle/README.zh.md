# Heddle

用于在 NVIDIA Hopper GPU 上为 TileLang 提供自动异步流水线调度。

Heddle 由三个编译器模块组成：
- **AutoPiped** — 自动的流水线注释（分组/顺序/阶段）
- **AutoMixed** — 考虑资源的图级融合与划块决策
- **Heddle 调度器** — 基于 CP-SAT 的联合 SWP+WS 调度，支持 FineGrainedWS、PCWS、双消费者和 TRWS

## 快速开始

```bash
# 从源码安装上游 TileLang。
git clone https://github.com/tile-ai/tilelang.git
cd tilelang
git submodule update --init --depth=1 3rdparty/tvm 3rdparty/cutlass 3rdparty/composable_kernel
pip install -e . --no-build-isolation

# 安装 Heddle。
cd ../Heddle
pip install -e .

# 在你的代码中使用：
import heddle
heddle.init()  # 在运行时打补丁 TileLang
```

## PCWS 集成

Heddle 现在使用来自 TileLang 主分支的 PCWS 组件。这使得 Heddle 保持与上游 TileLang 当前的基于 tile-op 的 PCWS 实现对齐，同时仍允许 Heddle 的 Python pass 在 PCWS 运行之前执行，并提供 AutoPiped、消费者调度、CP-SAT 阶段 B、双消费者和 TRWS 的决策。在本 README 使用的 H100 基准集中，与历史的 Heddle FineGrainedWS 路径相比，测得的性能差异约在 2% 左右。

`patches/cpp/` 目录是面向旧版的完整后端替换路径（遗留），针对 TileLang v0.1.8。它实现了 FineGrainedWS 加上 Heddle 特定的扩展，但在不移植的情况下不应直接应用于 TileLang 主分支，因为 TileLang 主分支使用不同的基于 tile-op 的 PCWS 架构。

对于旧版 TileLang v0.1.8 的实验，C++ 替换路径为：

```bash
bash install.sh --with-cpp /path/to/tilelang/source
cd /path/to/tilelang/source && pip install -e . --no-build-isolation
```

详见 `patches/cpp/README.md`。

## 使用方法

```python
import heddle
heddle.init()

import tilelang
from tilelang.transform import PassConfigKey

pass_configs = {
    PassConfigKey.TL_ENABLE_FAST_MATH: True,
    PassConfigKey.TL_ENABLE_AUTO_TL_PIPELINE_SMT: True,       # AutoPiped
    PassConfigKey.TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE: True,    # Heddle 调度
    PassConfigKey.TL_HEDDLE_USE_PHASE_B: True,                 # CP-SAT 联合求解
    PassConfigKey.TL_HEDDLE_USE_PRECISE_LATENCY: True,         # 架构感知延迟
    PassConfigKey.TL_HEDDLE_USE_ALAP_PRIORITY: True,           # ALAP 优先级
}

@tilelang.jit(out_idx=[...], pass_configs=pass_configs)
def my_kernel(...):
    ...
```

## 性能快照

下面所有数值均为在 NVIDIA H100（CUDA 后端）上测得的 FP16 TFLOPS。

### FlashAttention 前向

2026-05-19 的同一会话复测比较了当前 TileLang 与 Heddle 提供者在 MHA 形状上的表现：

| Shape | TileLang | Heddle | Heddle / TileLang |
|-------|----------|--------|-------------------|
| B4 H32 D128 T1024 | 448.4 | 443.9 | 0.99x |
| B4 H32 D128 T2048 | 530.7 | 557.6 | 1.05x |
| B4 H32 D128 T4096 | 534.2 | 586.6 | 1.10x |
| B4 H32 D128 T8192 | 539.6 | 598.6 | 1.11x |
| B1 H32 D128 T4096 | 555.9 | 596.9 | 1.07x |
| B1 H32 D128 T8192 | 543.2 | 618.6 | 1.14x |
| B1 H32 D128 T16384 | 544.2 | 582.0 | 1.07x |

几何平均：**1.075x**（相对于 TileLang）。

### FlashAttention 反向

已发布的 Heddle BWD 结果使用原始的拆分实现：dKdV 和 dQ 为独立内核，按顺序计时，并采用标准的 5-GEMM FlashAttention 反向 FLOP 统计。这个结果适用于论文表格；它与 2026-05-19 当前提供者的复测不可直接比较，后者使用了不同的 BWD 路径。

| Shape | TileLang WS | Heddle | Heddle+3Role | 3Role / TileLang WS |
|-------|-------------|--------|--------------|---------------------|
| B4 H32 D128 T1024 | 324.0 | 323.5 | 403.7 | 1.25x |
| B4 H32 D128 T2048 | 349.4 | 349.1 | 441.8 | 1.26x |
| B4 H32 D128 T4096 | 358.5 | 359.0 | 434.2 | 1.21x |
| B4 H32 D128 T8192 | 358.0 | 358.0 | 442.3 | 1.24x |
| B1 H32 D128 T4096 | 361.1 | 360.8 | 479.1 | 1.33x |
| B1 H32 D128 T8192 | 364.8 | 365.9 | 445.1 | 1.22x |

几何平均：普通 Heddle 相对于 TileLang WS 是 **1.00x**，而 Heddle+3Role 相对于 TileLang WS 为 **1.25x**。增益来自于拆分的 dKdV+dQ 设计和三角色 WS，而非仅仅是消费者重排序。

## 包结构

```
heddle/
  heddle/                    # Python 包
    scheduler/               # CP-SAT 和 SMT 调度
    transform/               # Heddle 消费者调度、AutoPiped
    frontend/                # AutoMixed：图级决策
    tools/                   # ptxas 反馈工具
    tileop/                  # FA/attention 操作模板
    _monkey_patch.py         # 用于修补原生 TileLang 的运行时补丁
  patches/cpp/               # C++ 替换文件（需要重建）
  install.sh                 # 自动化安装脚本
```

## 兼容性

- TileLang 0.1.8+；已在上游 `tile-ai/tilelang` 提交 `7bf8de1` 下验证
- Python >= 3.9
- NVIDIA H100（Hopper）或更高
- CUDA >= 12.0
