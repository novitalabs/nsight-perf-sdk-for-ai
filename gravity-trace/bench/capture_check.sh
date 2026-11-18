#!/usr/bin/env bash
# Live capture (gt capture / gt exporter POST /capture, straight from in-memory snapshots) against the SDK's
# streaming consumer + gt import running at the same time, on a synthetic CUDA-graph load with known kernel counts
# (graphload.py: every replay = 300 graph-node kernels + 20 eager kernels). Both paths are compared on the capture's
# window. Perfetto conversion is skipped (3.8 M events take ~100 s to convert, the load would be over).
#   bash bench/capture_check.sh                 # service on GPUs 0,1; load on GPU 0
# Output in $OUT: standalone/, exporter/, tsf/ + imp/, load.log, summary.json (rows per kind, kernels per replay).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/tmp/gt-bench/capture-check}
GT=${GT_BUILD:-${CARGO_TARGET_DIR:-$HERE/../target}/release/gt}
NPS_BIN=${NPS_BIN:-$NVTRACE_SDK_DIR/build/bin}
PY=${PY:-python3}
PORT=${PORT:-9476}
SECS=${SECS:-90}
DUR=${DUR:-5s}

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$GT" "$OUT/gt"   # a stable copy: a rebuild must not replace the running binary
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_NSIGHT_PERF_FORCE_DISABLE_CUDA_TRACING CUDA_NSIGHT_PERF_FORCE_DISABLE_NVTX_TRACING
log() { echo "=== $(date +%T) $*"; }

CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=${HOSTDRV:-} setsid "$NPS_BIN/nvidia-nsight-perf" > "$OUT/service.log" 2>&1 &
S=$!
CUDA_VISIBLE_DEVICES=0,1 "$NPS_BIN/nvtrace_poll_until_service_ready" --timeout-sec 120 > "$OUT/poll.log" 2>&1 || { echo "service not ready"; kill -TERM $S; exit 1; }
CUDA_VISIBLE_DEVICES=0 $PY "$HERE/graphload.py" --nodes 300 --eager 20 --secs "$SECS" --sleep-us 200 > "$OUT/load.log" 2>&1 &
L=$!
sleep 10

log "nvtrace_streaming_consumer (15 s, the old path) and, inside it, gt capture -d $DUR"
CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$NPS_BIN "$NPS_BIN/nvtrace_streaming_consumer" --output-dir "$OUT/tsf" --duration-ms 15000 > "$OUT/consumer.log" 2>&1 &
N=$!
sleep 3
t0=$(date +%s%N)
CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$NPS_BIN "$OUT/gt" capture -d "$DUR" -o "$OUT/standalone" --work-dir "$OUT/live1" --no-chrome --json \
  > "$OUT/standalone.json" 2> "$OUT/standalone.err"
echo "rc=$? wall $(( ($(date +%s%N) - t0) / 1000000 )) ms"
wait $N
DB=$(dirname "$(ls "$OUT"/tsf/*/*.nvtsf 2>/dev/null | head -1)")
t0=$(date +%s%N)
LD_LIBRARY_PATH=$NPS_BIN "$OUT/gt" import "$DB" -o "$OUT/imp" --no-chrome --json > "$OUT/import.json" 2> "$OUT/import.err"
echo "import rc=$? wall $(( ($(date +%s%N) - t0) / 1000000 )) ms"

log "gt exporter + POST /capture?duration=$DUR"
CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$NPS_BIN setsid "$OUT/gt" exporter --listen "127.0.0.1:$PORT" --work-dir "$OUT/live2" \
  --captures-dir "$OUT/exporter" > "$OUT/exporter.log" 2>&1 &
C=$!
sleep 8
t0=$(date +%s%N)
curl -s -X POST "127.0.0.1:$PORT/capture?duration=$DUR&chrome=false" > "$OUT/exporter.json"
echo "rc=$? wall $(( ($(date +%s%N) - t0) / 1000000 )) ms"
curl -s "127.0.0.1:$PORT/metrics" | grep -E '^nvtrace_(snapshots_total|activities_total|lost_activities_total|snapshot_storage_in_memory)' > "$OUT/metrics.txt"
kill -INT $C; wait $C 2>/dev/null


wait $L
kill -TERM $S; wait $S 2>/dev/null
du -sb "$OUT/tsf" "$OUT/standalone" "$OUT/exporter" "$OUT/imp" 2>/dev/null

$PY - "$OUT" <<'PY'
import glob, json, os, re, sys
import pyarrow.compute as pc
import pyarrow.parquet as pq
out = sys.argv[1]
load = json.loads(open(f"{out}/load.log").read().strip().splitlines()[-1])
res = {"load": load, "replays_per_s": load["replays"] / ((load["end_utc_ns"] - load["start_utc_ns"]) / 1e9)}

def counts(t, w0, w1):
    t = t.filter(pc.and_(pc.greater_equal(t["start_ns"], w0), pc.less(t["start_ns"], w1)))
    k = t.filter(pc.and_(pc.equal(t["kind"].cast("string"), "kernel"), pc.equal(t["pid"], load["pid"])))
    graph = k.filter(pc.is_valid(k["graph_id"])).num_rows
    kinds = {r["values"]: r["counts"] for r in pc.value_counts(t["kind"].cast("string")).to_pylist()}
    return {"rows": t.num_rows, "kinds": kinds, "graph_kernels": graph, "eager_kernels": k.num_rows - graph,
            "graph_per_eager": round(graph / (k.num_rows - graph), 4) if k.num_rows > graph else None, "replays": graph / 300}

def table(name):
    f = [f for f in sorted(glob.glob(f"{out}/{name}/*.parquet")) if re.fullmatch(r"\d+-\d+\.parquet|trace-nvp-.*\.parquet", os.path.basename(f))]
    return (pq.read_table(f[0]), os.path.getsize(f[0])) if f else (None, 0)

for name, rep in (("standalone", "standalone.json"), ("exporter", "exporter.json")):
    try:
        r = json.load(open(f"{out}/{rep}"))
    except ValueError:
        res[name] = open(f"{out}/{rep}").read()[:300]
        continue
    t, size = table(name)
    r.pop("processes", None)
    res[name] = {"report": r, "bytes": size, **(counts(t, r["start_utc_ns"], r["end_utc_ns"]) if t is not None else {})}
    if name == "standalone":
        w = (r["start_utc_ns"], r["end_utc_ns"])
imp, size = table("imp")
if imp is not None:
    rep = json.load(open(f"{out}/import.json"))
    res["import_same_window"] = {"bytes_whole_file": size, "tsf_bytes": sum(os.path.getsize(f) for f in glob.glob(f"{out}/tsf/*/*")),
                                 "observation_s": (rep["observation_end_utc_ns"] - rep["observation_start_utc_ns"]) / 1e9,
                                 "decode_s": rep["decode_s"], **counts(imp, *w)}
json.dump(res, open(f"{out}/summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))
PY
