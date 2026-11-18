#!/usr/bin/env bash
# Which processes does a running nvidia-nsight-perf service trace? (service already started on SVC_GPU)
#   case served:    workload on the service's GPU, default env
#   case unserved:  workload on another GPU, default env
#   case disabled:  workload on the service's GPU with CUDA_NSIGHT_PERF_FORCE_DISABLE_CUDA_TRACING=1
# Each case gets its own streaming-consumer capture, imported with `gt import`.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=${NPS_BIN:-$NVTRACE_SDK_DIR/build/bin}
GT=${GT_BIN:-${CARGO_TARGET_DIR:-$HERE/../target}/release/gt}
SVC_GPU=${SVC_GPU:-2}
OTHER_GPU=${OTHER_GPU:-0}
OUT=${1:-/tmp/gt-bench/iso}
rm -rf "$OUT"; mkdir -p "$OUT"
run_case() {
  local name=$1 gpu=$2; shift 2
  mkdir -p "$OUT/$name"
  CUDA_VISIBLE_DEVICES=$SVC_GPU "$B/nvtrace_streaming_consumer" --output-dir "$OUT/$name" --duration-ms 8000 --ignore gpu_pm > "$OUT/$name/consumer.log" 2>&1 &
  local c=$!
  sleep 1.5
  env CUDA_VISIBLE_DEVICES=$gpu "$@" "$B/BenchmarkNvTraceCuda" --nvtrace=false \
    --benchmark_filter='KernelAdd|AsyncGraphKernel_Empty' --benchmark_min_time=0.5s --benchmark_format=json \
    > "$OUT/$name/bench.json" 2> "$OUT/$name/bench.err"
  wait $c
  local db; db=$(ls -d "$OUT/$name"/*/ | head -1)
  LD_LIBRARY_PATH=$B "$GT" import "$db" -o "$OUT/$name/imp" --no-chrome --ignore gpu_pm --json > "$OUT/$name/import.json" 2> "$OUT/$name/import.err"
  python3 - "$OUT/$name" <<'PY'
import json, sys
d = sys.argv[1]
b = json.load(open(f"{d}/bench.json"))
t = {x["name"].split("/")[1]: round(x["real_time"], 1) for x in b["benchmarks"]}
try:
    r = json.load(open(f"{d}/import.json")); s = r["stats"]
    tr = dict(procs=[p[1] for p in r["processes"]], kernels=s["kernels"], graph=s["graph_kernels"], api=s["driver_api"])
except Exception as e:
    tr = f"no import: {e}"
print(d.rsplit("/", 1)[1], "ns/iter", t, "traced", tr)
PY
}
run_case served "$SVC_GPU"
run_case unserved "$OTHER_GPU"
run_case disabled "$SVC_GPU" CUDA_NSIGHT_PERF_FORCE_DISABLE_CUDA_TRACING=1 CUDA_NSIGHT_PERF_FORCE_DISABLE_NVTX_TRACING=1
