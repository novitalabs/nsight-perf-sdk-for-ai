#!/usr/bin/env bash
# Build bench/replay.cpp against the SDK headers patched by nvtrace-sys's build script, then run it.
#   bash bench/replay.sh DIR SHARDS [SERIAL=0] [REPEAT=3] [MEM=0]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
SDK=${NVTRACE_SDK_DIR:?set NVTRACE_SDK_DIR to the Nsight Perf SDK checkout}
TARGET=${CARGO_TARGET_DIR:-$ROOT/target}
INC=$(ls -dt "$TARGET"/release/build/nvtrace-sys*/out/nvtrace-client-include "$TARGET"/release/build/nvtrace-sys*/*/out/nvtrace-client-include 2>/dev/null | head -1 || true)
BIN=${REPLAY_BIN:-/tmp/gt-bench/replay}
if [[ ! -x $BIN || $HERE/replay.cpp -nt $BIN || $ROOT/crates/nvtrace-sys/shim/nvtrace_shim.cpp -nt $BIN || $ROOT/crates/nvtrace-sys/shim/gt_memstore.h -nt $BIN ]]; then
  g++ -O2 -g -std=c++17 -fno-strict-aliasing -DNVTRACE_ENABLE_CXA_DEMANGLE=1 \
    -I "$ROOT/crates/nvtrace-sys/shim" -I "$INC" -I "$SDK/nvperf-core/include" -I "$SDK/nvperf-core/include/linux-desktop-a64" \
    -I "$SDK/build/_deps/flatbuffers-src/include" "$HERE/replay.cpp" -o "$BIN" -ldl -lpthread
fi
exec "$BIN" "$@"
