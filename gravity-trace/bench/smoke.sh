#!/usr/bin/env bash
# Nsight Perf smoke test: an already-running global service + a streaming consumer while the SDK's CUDA
# benchmark (and optionally a CUDA-graph probe) runs on GPU 0. Prints the export summary.
set -uo pipefail
B=${NPS_BIN:-$NVTRACE_SDK_DIR/build/bin}
OUT=${1:-/tmp/gt-bench/smoke}
DUR=${DUR:-15000}
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$OUT"
"$B/nvtrace_streaming_consumer" --output-dir "$OUT" --duration-ms "$DUR" > consumer.log 2>&1 &
cpid=$!
sleep 2
CUDA_VISIBLE_DEVICES=0 "$B/BenchmarkNvTraceCuda" --nvtrace=false \
  --benchmark_filter='EndToEnd|KernelAdd|AsyncGraphKernel_Empty|AsyncDenseKernel_Empty' \
  --benchmark_min_time=0.5 > bench.log 2>&1
echo "bench rc=$?"
wait $cpid; echo "consumer rc=$?"
db=$(ls -d "$OUT"/*/ | head -1)
du -sh "$db"; ls "$db" | head
"$B/nvtrace_perfetto_trace_exporter" --help 2>&1 | head -30
