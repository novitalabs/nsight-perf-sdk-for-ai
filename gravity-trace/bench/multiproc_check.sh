#!/usr/bin/env bash
# Several CUDA processes time-sliced on one GPU: 4 graphload.py loads on GPU 0 (2 of them PID 1 in PID namespaces of
# their own, like containers) under `gt export`, then each host PID's kernels in the persisted trace against what it
# launched (multiproc_cmp.py): graph and eager must both be +0, and `ns_pid` the PID in the load's own namespace.
#   bash bench/multiproc_check.sh               # service on GPUs 0,1; needs root (unshare) and the host PID namespace
#   NS_WRAP= bash bench/multiproc_check.sh      # all 4 loads in gt's PID namespace
# Output in $OUT: traces/, pids.txt (host pid + NSpid per load), load-*.log, metrics.txt, summary.txt.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/tmp/gt-bench/multiproc-check}
GT=${GT_BUILD:-${CARGO_TARGET_DIR:-$HERE/../target}/release/gt}
NPS_BIN=${NPS_BIN:-$NVTRACE_SDK_DIR/build/bin}
PY=${PY:-python3}
PORT=${PORT:-19465}
SECS=${SECS:-20}

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$GT" "$OUT/gt"   # a stable copy: a rebuild must not replace the running binary
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_NSIGHT_PERF_FORCE_DISABLE_CUDA_TRACING CUDA_NSIGHT_PERF_FORCE_DISABLE_NVTX_TRACING

CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=${HOSTDRV:-} setsid "$NPS_BIN/nvidia-nsight-perf" > "$OUT/service.log" 2>&1 &
S=$!
CUDA_VISIBLE_DEVICES=0,1 "$NPS_BIN/nvtrace_poll_until_service_ready" --timeout-sec 120 > "$OUT/poll.log" 2>&1 || { echo "service not ready"; kill -TERM $S; exit 1; }
cat > "$OUT/config.toml" <<TOML
[daemon]
listen = "127.0.0.1:$PORT"
pid_file = "$OUT/gt.pid"
log_file = "$OUT/gt.log"
captures_dir = "$OUT/captures"
[collect]
work_dir = "$OUT/live"
[persist]
dir = "$OUT/traces"
rotate_s = 20
[limits]
max_cpus = 4
TOML
LD_LIBRARY_PATH=$NPS_BIN "$OUT/gt" export $((SECS + 30))s -c "$OUT/config.toml" > "$OUT/out.txt" 2> "$OUT/err.txt" &
G=$!
sleep 5
LOADS=()
for n in 300 301; do
  CUDA_VISIBLE_DEVICES=0 $PY "$HERE/graphload.py" --nodes $n --eager 20 --secs "$SECS" --sleep-us 200 > "$OUT/load-$n.log" 2>&1 &
  LOADS+=($!)
done
for n in 302 303; do
  ${NS_WRAP-unshare --pid --fork --mount-proc --} env CUDA_VISIBLE_DEVICES=0 \
    $PY "$HERE/graphload.py" --nodes $n --eager 20 --secs "$SECS" --sleep-us 200 > "$OUT/load-$n.log" 2>&1 &
  LOADS+=($!)
done
sleep $((SECS * 3 / 4))
# The python processes (under unshare, the child of the forked init), their host pid and NSpid.
for n in 300 301 302 303; do
  for p in $(pgrep -f "graphload.py --nodes $n "); do
    [[ $(cat /proc/$p/comm) == python* ]] && echo "load $n host pid $p $(grep NSpid /proc/$p/status | tr '\t' ' ')"
  done
done | tee "$OUT/pids.txt"
curl -s "127.0.0.1:$PORT/metrics" > "$OUT/metrics.txt"
wait "${LOADS[@]}"
wait $G; echo "gt rc=$?"
kill -TERM $S; wait $S 2>/dev/null
$PY "$HERE/multiproc_cmp.py" "$OUT" | tee "$OUT/summary.txt"
