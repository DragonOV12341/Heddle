# Heddle

Automated async pipeline scheduling for TileLang on NVIDIA Hopper GPUs.

Heddle consists of three compiler modules:
- **AutoPiped** — automatic pipeline annotation (group/order/stage)
- **AutoMixed** — resource-aware graph-level fusion and tiling decisions
- **Heddle scheduler** — CP-SAT-based joint SWP+WS scheduling with FineGrainedWS, PCWS, dual-consumer, and TRWS

## Quick start

```bash
# Install upstream TileLang from source.
git clone https://github.com/tile-ai/tilelang.git
cd tilelang
git submodule update --init --depth=1 3rdparty/tvm 3rdparty/cutlass 3rdparty/composable_kernel
pip install -e . --no-build-isolation

# Install Heddle.
cd ../Heddle
pip install -e .

# In your code:
import heddle
heddle.init()  # patches TileLang at runtime
```

## PCWS integration

Heddle now uses the PCWS component from the TileLang main branch. This keeps
Heddle aligned with upstream TileLang's current tile-op-level PCWS
implementation while still allowing Heddle's Python passes to run before PCWS
and provide AutoPiped, consumer scheduling, CP-SAT Phase B, dual-consumer, and
TRWS decisions. In the H100 benchmark set used for this README, the measured
performance difference from the historical Heddle FineGrainedWS path is within about
2%.

The `patches/cpp/` directory is a legacy full-backend replacement path for
TileLang v0.1.8. It implements FineGrainedWS plus Heddle-specific extensions, but it
should not be applied directly to TileLang main without a port because TileLang
main uses a different tile-op-level PCWS architecture.

For legacy TileLang v0.1.8 experiments, the C++ replacement path is:

```bash
bash install.sh --with-cpp /path/to/tilelang/source
cd /path/to/tilelang/source && pip install -e . --no-build-isolation
```

See `patches/cpp/README.md` for details.

## Usage

```python
import heddle
heddle.init()

import tilelang
from tilelang.transform import PassConfigKey

pass_configs = {
    PassConfigKey.TL_ENABLE_FAST_MATH: True,
    PassConfigKey.TL_ENABLE_AUTO_TL_PIPELINE_SMT: True,       # AutoPiped
    PassConfigKey.TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE: True,    # Heddle scheduling
    PassConfigKey.TL_HEDDLE_USE_PHASE_B: True,                 # CP-SAT joint solve
    PassConfigKey.TL_HEDDLE_USE_PRECISE_LATENCY: True,         # arch-aware latency
    PassConfigKey.TL_HEDDLE_USE_ALAP_PRIORITY: True,           # ALAP priority
}

@tilelang.jit(out_idx=[...], pass_configs=pass_configs)
def my_kernel(...):
    ...
```

## Performance snapshot

All numbers below are FP16 TFLOPS on NVIDIA H100 with the CUDA backend.

### FlashAttention forward

The 2026-05-19 same-session retest compares the current TileLang and Heddle
providers on MHA shapes:

| Shape | TileLang | Heddle | Heddle / TileLang |
|-------|----------|--------|-------------------|
| B4 H32 D128 T1024 | 448.4 | 443.9 | 0.99x |
| B4 H32 D128 T2048 | 530.7 | 557.6 | 1.05x |
| B4 H32 D128 T4096 | 534.2 | 586.6 | 1.10x |
| B4 H32 D128 T8192 | 539.6 | 598.6 | 1.11x |
| B1 H32 D128 T4096 | 555.9 | 596.9 | 1.07x |
| B1 H32 D128 T8192 | 543.2 | 618.6 | 1.14x |
| B1 H32 D128 T16384 | 544.2 | 582.0 | 1.07x |

Geometric mean: **1.075x** over TileLang.

### FlashAttention backward

The published Heddle BWD result uses the original split implementation: dKdV
and dQ are separate kernels, timed sequentially, with the standard 5-GEMM
FlashAttention backward FLOP accounting. This is the result to use for thesis
tables; it is not comparable to the 2026-05-19 current-provider retest, which
exercises a different BWD path.

| Shape | TileLang WS | Heddle | Heddle+3Role | 3Role / TileLang WS |
|-------|-------------|--------|--------------|---------------------|
| B4 H32 D128 T1024 | 324.0 | 323.5 | 403.7 | 1.25x |
| B4 H32 D128 T2048 | 349.4 | 349.1 | 441.8 | 1.26x |
| B4 H32 D128 T4096 | 358.5 | 359.0 | 434.2 | 1.21x |
| B4 H32 D128 T8192 | 358.0 | 358.0 | 442.3 | 1.24x |
| B1 H32 D128 T4096 | 361.1 | 360.8 | 479.1 | 1.33x |
| B1 H32 D128 T8192 | 364.8 | 365.9 | 445.1 | 1.22x |

Geometric mean: ordinary Heddle is **1.00x** over TileLang WS, while
Heddle+3Role is **1.25x** over TileLang WS. The gain comes from the split
dKdV+dQ design and three-role WS, not from consumer reordering alone.

## Package structure

```
heddle/
  heddle/                    # Python package
    scheduler/               # CP-SAT and SMT scheduling
    transform/               # Heddle consumer schedule, AutoPiped
    frontend/                # AutoMixed: graph-level decisions
    tools/                   # ptxas feedback utilities
    tileop/                  # FA/attention op templates
    _monkey_patch.py         # Runtime patches for stock TileLang
  patches/cpp/               # C++ replacement files (require rebuild)
  install.sh                 # Automated installer
```

## Compatibility

- TileLang 0.1.8+; verified with upstream `tile-ai/tilelang` commit `7bf8de1`
- Python >= 3.9
- NVIDIA H100 (Hopper) or later
- CUDA >= 12.0
