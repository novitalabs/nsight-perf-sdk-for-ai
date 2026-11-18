#!/usr/bin/env bash
# Completeness of the Nsight Perf live consumer (`gt exporter`) against a load with known kernel counts.
#   NPROC=4 bash bench/livediag.sh      # one graphload.py per GPU 0..NPROC-1, service on GPUs 0-3
# Output in $OUT: diag.jsonl (one line per snapshot), metrics.txt (final counters), load*.log (ground truth).
set -uo pipefail
NPROC=${NPROC:-4}
SECS=${SECS:-60}
NODES=${NODES:-300}
EAGER=${EAGER:-20}
SLEEP_US=${SLEEP_US:-0}
SYNC_EVERY=${SYNC_EVERY:-8}
TOUCH_ALL=${TOUCH_ALL:-0}   # 1: every load sees all 4 GPUs and touches each (tensor-parallel-like), runs on its own
DATA_RATE=${DATA_RATE:-64}
NVP_CONFIG=${NVP_CONFIG:-}   # service YAML (e.g. a larger GpuHwTrace.recordBufferSize)
SNAP_MS=${SNAP_MS:-5000}
OUT=${OUT:-/tmp/gt-bench/nvp-livediag/p${NPROC}}
HERE=$(cd "$(dirname "$0")" && pwd)
GT_BUILD=${GT_BUILD:-${CARGO_TARGET_DIR:-$HERE/../target}/release/gt}
NPS_BIN=${NPS_BIN:-$NVTRACE_SDK_DIR/build/bin}
PY=${PY:-python3}
PORT=${PORT:-9475}

mkdir -p "$OUT"
rm -f "$OUT"/diag.jsonl "$OUT"/*.log "$OUT"/metrics.txt
rm -rf "$OUT/live"
cp "$GT_BUILD" "$OUT/gt"   # a stable copy: a rebuild must not replace the running binary
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_NSIGHT_PERF_FORCE_DISABLE_CUDA_TRACING CUDA_NSIGHT_PERF_FORCE_DISABLE_NVTX_TRACING

CUDA_VISIBLE_DEVICES=0,1,2,3 LD_LIBRARY_PATH=${HOSTDRV:-} setsid "$NPS_BIN/nvidia-nsight-perf" ${NVP_CONFIG:+--config "$NVP_CONFIG"} > "$OUT/service.log" 2>&1 &
S=$!
CUDA_VISIBLE_DEVICES=0,1,2,3 "$NPS_BIN/nvtrace_poll_until_service_ready" --timeout-sec 120 > "$OUT/poll.log" 2>&1 || { echo "service not ready"; kill -TERM $S; exit 1; }
CUDA_VISIBLE_DEVICES=0,1,2,3 LD_LIBRARY_PATH=$NPS_BIN setsid "$OUT/gt" exporter --listen "127.0.0.1:$PORT" --snapshot-ms "$SNAP_MS" \
  --work-dir "$OUT/live" --data-rate-mib "$DATA_RATE" --diag "$OUT/diag.jsonl" > "$OUT/exporter.log" 2>&1 &
C=$!
sleep 8
pids=()
for g in $(seq 0 $((NPROC - 1))); do
  if [[ $TOUCH_ALL == 1 ]]; then vis=0,1,2,3; dev=$g; extra=--touch-all; else vis=$g; dev=0; extra=; fi
  CUDA_VISIBLE_DEVICES=$vis $PY "$HERE/graphload.py" --nodes "$NODES" --eager "$EAGER" --secs "$SECS" --sleep-us "$SLEEP_US" \
    --sync-every "$SYNC_EVERY" --device "$dev" $extra > "$OUT/load$g.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
sleep 25   # three more snapshots: pending rows complete or expire
curl -s "127.0.0.1:$PORT/metrics" | grep -E '^nvtrace_(snap|rows|activ|pend|lost|orphan|compl)' > "$OUT/metrics.txt"
kill -INT $C; wait $C 2>/dev/null
kill -TERM $S; wait $S 2>/dev/null
cat "$OUT/metrics.txt"
tail -q -n1 "$OUT"/load*.log
