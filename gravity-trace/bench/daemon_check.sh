#!/usr/bin/env bash
# gt export-daemon end to end: config.toml -> background daemon (CPU limit, persisted trace rotating every 20 s) on a
# synthetic CUDA-graph load with known kernel counts (graphload.py: every replay = 300 graph-node kernels + 20 eager
# kernels), `gt dump` while it runs (asks the daemon to close its open file) and after `stop`, checked against the
# load's own counts.
#   bash bench/daemon_check.sh            # service on GPUs 0,1; load on GPU 0
# Output in $OUT: config.toml, gt.log, traces/ (persisted), dump-*/, status.txt, metrics.txt, summary.json.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/tmp/gt-bench/daemon-check}
GT=${GT_BUILD:-${CARGO_TARGET_DIR:-$HERE/../target}/release/gt}
NPS_BIN=${NPS_BIN:-$NVTRACE_SDK_DIR/build/bin}
PY=${PY:-python3}
PORT=${PORT:-9477}
SECS=${SECS:-75}
CPUS=${CPUS:-4}

summarize() {
$PY - "$OUT" <<'PY'
import glob, json, os, re, sys
import pyarrow.compute as pc
import pyarrow.parquet as pq
out = sys.argv[1]
load = json.loads(open(f"{out}/load.log").read().strip().splitlines()[-1])
res = {"load": load}
names = sorted(os.listdir(f"{out}/traces"))
span = re.compile(r"^(\d+)-(\d+)\.(pm\.)?parquet$")
res["persisted"] = [{"name": n, "bytes": os.path.getsize(f"{out}/traces/{n}"),
                     "rows": pq.ParquetFile(f"{out}/traces/{n}").metadata.num_rows,
                     "row_groups": pq.ParquetFile(f"{out}/traces/{n}").metadata.num_row_groups} for n in names]
res["bad_names"] = [n for n in names if not span.match(n)]
for n in names:
    m = span.match(n)
    if m and not m.group(3):
        t = pq.read_table(f"{out}/traces/{n}", columns=["start_ns", "end_ns"])
        lo, hi = pc.min(t["start_ns"]).as_py(), pc.max(t["end_ns"]).as_py()
        assert (lo, hi) == (int(m.group(1)), int(m.group(2))), (n, lo, hi)
for name in ("dump-live", "dump-load"):
    try:
        res[name] = json.load(open(f"{out}/{name}.json"))
    except ValueError:
        res[name] = open(f"{out}/{name}.json").read()[:300]
d = res["dump-load"]
if isinstance(d, dict):
    t = pq.read_table(d["output"])
    k = t.filter(pc.and_(pc.equal(t["kind"].cast("string"), "kernel"), pc.equal(t["pid"], load["pid"])))
    graph = k.filter(pc.is_valid(k["graph_id"])).num_rows
    res["check"] = {"rows": t.num_rows, "graph_kernels": graph, "eager_kernels": k.num_rows - graph,
                    "expected_graph": load["graph_kernels"], "expected_eager": load["eager_kernels"],
                    "min_start_ok": pc.min(t["start_ns"]).as_py() >= load["start_utc_ns"],
                    "max_start_ok": pc.max(t["start_ns"]).as_py() < load["end_utc_ns"],
                    "duplicates": t.num_rows - t.unify_dictionaries().group_by(["pid", "correlation_id", "start_ns", "kind"]).aggregate([]).num_rows}
res["perfetto"] = [(os.path.basename(f), os.path.getsize(f)) for f in glob.glob(f"{out}/dump-perfetto/*")]
json.dump(res, open(f"{out}/summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "persisted"}, indent=1))
print(f"{len(res['persisted'])} persisted files")
PY
}
if [[ -n "${SUMMARY_ONLY:-}" ]]; then summarize; exit; fi

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$GT" "$OUT/gt"   # a stable copy: a rebuild must not replace the running binary
GT="$OUT/gt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_NSIGHT_PERF_FORCE_DISABLE_CUDA_TRACING CUDA_NSIGHT_PERF_FORCE_DISABLE_NVTX_TRACING
log() { echo "=== $(date +%T) $*"; }

cat > "$OUT/config.toml" <<EOF
[daemon]
listen = "127.0.0.1:$PORT"
pid_file = "$OUT/run/gt.pid"
log_file = "$OUT/gt.log"
captures_dir = "$OUT/captures"

[collect]
work_dir = "$OUT/live"

[persist]
dir = "$OUT/traces"
rotate_s = 20
max_buffer_mib = ${BUF_MIB:-1024}
row_group_rows = 262144

[limits]
max_cpus = $CPUS
EOF

CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=${HOSTDRV:-} setsid "$NPS_BIN/nvidia-nsight-perf" > "$OUT/service.log" 2>&1 &
S=$!
CUDA_VISIBLE_DEVICES=0,1 "$NPS_BIN/nvtrace_poll_until_service_ready" --timeout-sec 120 > "$OUT/poll.log" 2>&1 || { echo "service not ready"; kill -TERM $S; exit 1; }

export CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$NPS_BIN GT_CONFIG="$OUT/config.toml"
log "gt export-daemon start"
"$GT" export-daemon start; echo "rc=$?"
log "second start (must refuse)"
"$GT" export-daemon start; echo "rc=$? (expected 1)"
"$GT" export-daemon status | tee "$OUT/status.txt"
PID=$(cat "$OUT/run/gt.pid")
grep -E 'Cpus_allowed_list|Threads' "/proc/$PID/status" | tee -a "$OUT/status.txt"
for t in /proc/$PID/task/*; do grep Cpus_allowed_list "$t/status"; done | sort | uniq -c | tee -a "$OUT/status.txt"

CUDA_VISIBLE_DEVICES=0 $PY "$HERE/graphload.py" --nodes 300 --eager 20 --secs "$SECS" --sleep-us 200 > "$OUT/load.log" 2>&1 &
L=$!
sleep 45
ps -o pid,nlwp,pcpu,rss,etime -p "$PID" | tee -a "$OUT/status.txt"
log "gt dump parquet now-30s now (while running: the daemon closes its open file first)"
"$GT" dump parquet now-30s now -o "$OUT/dump-live" --json > "$OUT/dump-live.json"; echo "rc=$?"
log "gt dump perfetto now-20s +2s"
"$GT" dump perfetto now-20s +2s -o "$OUT/dump-perfetto"; echo "rc=$?"
curl -s "127.0.0.1:$PORT/metrics" | grep -E '^nvtrace_(persist_|snapshots_total|activities_total|lost_activities_total)' > "$OUT/metrics.txt"
wait $L
sleep 12   # the load's last activities reach a snapshot
log "gt export-daemon stop"
"$GT" export-daemon stop; echo "rc=$?"
"$GT" export-daemon status; echo "status rc=$? (expected 1)"
"$GT" export-daemon stop; echo "stop again rc=$?"
ls "$OUT/run" 2>/dev/null; ls -la "$OUT/traces" | tail -n +2 > "$OUT/traces.txt"
kill -TERM $S; wait $S 2>/dev/null

LOAD=$(tail -1 "$OUT/load.log")
W0=$($PY -c "import json,sys; print(json.loads(sys.argv[1])['start_utc_ns'])" "$LOAD")
W1=$($PY -c "import json,sys; print(json.loads(sys.argv[1])['end_utc_ns'])" "$LOAD")
log "gt dump parquet <load start> <load end> (after stop)"
"$GT" dump parquet "$W0" "$W1" -o "$OUT/dump-load" --json > "$OUT/dump-load.json"; echo "rc=$?"
"$GT" dump parquet 2000-01-01T00:00:00Z +1m -o "$OUT/dump-none"; echo "empty range rc=$? (expected 1)"

summarize
