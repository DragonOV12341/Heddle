# C++ Patches for Heddle

These are **full replacement files** for the TileLang C++ backend.
They implement FineGrainedWS, dual-consumer, three-role WS, barrier hints,
async WGMMA, and early release optimizations.

## Files

| File | Description |
|---|---|
| `src/transform/finegrained_ws.cc` | Legacy FineGrainedWS implementation with Heddle WS features; registers `FineGrainedWarpSpecialized` as the primary pass |
| `src/op/builtin.h` | Pass config key declarations (adds Heddle keys) |
| `src/op/builtin.cc` | Pass config key registrations |
| `src/transform/fuse_mbarrier_arrive_expect_tx.cc` | mbarrier arrive+expect_tx fusion pass |
| `src/transform/lower_ptx_async_copy.cc` | PTX async copy lowering |
| `src/transform/optimize_cp_async_sync.cc` | cp.async sync optimization |

## How to apply

1. Clone TileLang source and checkout v0.1.8:
   ```bash
   git clone https://github.com/tile-ai/tilelang.git
   cd tilelang && git checkout v0.1.8
   ```

2. Replace files:
   ```bash
   # Remove TileLang 0.1.8's stock warp-specialization source to avoid duplicate pass registration.
   rm -f src/transform/producer_consumer_ws.cc
   cp patches/cpp/src/transform/finegrained_ws.cc src/transform/
   cp patches/cpp/src/op/builtin.h src/op/
   cp patches/cpp/src/op/builtin.cc src/op/
   # Copy additional files if they don't exist in stock
   cp patches/cpp/src/transform/fuse_mbarrier_arrive_expect_tx.cc src/transform/
   cp patches/cpp/src/transform/lower_ptx_async_copy.cc src/transform/
   cp patches/cpp/src/transform/optimize_cp_async_sync.cc src/transform/
   ```

3. Rebuild:
   ```bash
   pip install -e . --no-build-isolation
   pip install apache-tvm-ffi==0.1.8.post2
   ```

## Note

The Heddle FineGrainedWS patch operates on **lowered IR** (post-`LowerTileOp`).
This is architecturally different from upstream main's tile-op-level
warp-specialization backend.
`FineGrainedWarpSpecialized` is the primary pass name; the older
`ProducerConsumerWarpSpecialized` entry point is kept only as a compatibility
alias for TileLang 0.1.8 pipelines.
These patches target TileLang 0.1.8 specifically.
