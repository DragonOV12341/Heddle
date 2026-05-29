#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Heddle installer ==="

# 1. Install Python package
echo "[1/3] Installing heddle Python package..."
pip install -e "$SCRIPT_DIR" --quiet

# 2. Verify TileLang is available
python -c "import tilelang; print(f'  TileLang found: {tilelang.__file__}')" || {
    echo "ERROR: TileLang not installed. Install TileLang from https://github.com/tile-ai/tilelang first."
    exit 1
}

# 3. Smoke test
echo "[2/3] Verifying heddle.init()..."
python -c "
import heddle
heddle.init()
from tilelang.transform import PassConfigKey
assert hasattr(PassConfigKey, 'TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE'), 'PassConfigKey not extended'
print('  Python patches OK')
"

echo "[3/3] C++ patches..."
if [ "${1:-}" = "--with-cpp" ]; then
    TILELANG_SRC="${2:?Usage: install.sh --with-cpp /path/to/tilelang/source}"
    echo "  Applying C++ patches to $TILELANG_SRC"
    rm -f "$TILELANG_SRC/src/transform/producer_consumer_ws.cc"
    cp "$SCRIPT_DIR/patches/cpp/src/transform/finegrained_ws.cc" "$TILELANG_SRC/src/transform/"
    cp "$SCRIPT_DIR/patches/cpp/src/op/builtin.h" "$TILELANG_SRC/src/op/"
    cp "$SCRIPT_DIR/patches/cpp/src/op/builtin.cc" "$TILELANG_SRC/src/op/"
    for f in fuse_mbarrier_arrive_expect_tx.cc lower_ptx_async_copy.cc optimize_cp_async_sync.cc ptx_async_copy_injector.h; do
        test -f "$SCRIPT_DIR/patches/cpp/src/transform/$f" && \
            cp "$SCRIPT_DIR/patches/cpp/src/transform/$f" "$TILELANG_SRC/src/transform/"
    done
    test -d "$SCRIPT_DIR/patches/cpp/src/transform/common" && \
        cp -r "$SCRIPT_DIR/patches/cpp/src/transform/common/"* "$TILELANG_SRC/src/transform/common/" 2>/dev/null || true
    echo "  C++ files copied. Rebuild TileLang:"
    echo "    cd $TILELANG_SRC && pip install -e . --no-build-isolation"
else
    echo "  Skipped (use --with-cpp /path/to/tilelang/source to apply)"
    echo "  Without C++ patches: Python scheduling works, but legacy FineGrainedWS splitting requires patched libtl"
fi

echo ""
echo "=== Done ==="
