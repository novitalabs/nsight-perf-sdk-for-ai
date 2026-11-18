#!/usr/bin/env bash
# gt profile --flightsql end to end, with the network traffic of every second: a synthetic CUDA-graph load with known
# kernel counts (graphload.py: every replay = 300 graph-node kernels + 20 eager kernels) is profiled while gt streams
# every activity to a Flight SQL server (examples/flightsql_sink.rs: every DoPut -> Parquet), once per encoding
# variant. Per second it records
#   - gt:     Arrow IPC bytes gt handed to gRPC (the profile report's traffic.per_second),
#   - wire:   bytes on the pod's loopback interface (/proc/net/dev; the pod has its own network namespace, so this is
#             the gt -> sink TCP traffic with TCP/IP headers and the ACKs back),
#   - server: Flight message bytes the sink received,
# plus gt's CPU time, and checks the received rows against the load's own counts.
#   bash bench/flightsql_check.sh                                  # variants: plain dict dict-zstd plain-cap
#   VARIANTS="dict-zstd" SECS=60 bash bench/flightsql_check.sh
# Output in $OUT/<variant>/: config.toml, gt.log, profile.json, sink/ (Parquet), sink.json, server-traffic.jsonl,
# wire.txt, load.log; $OUT/summary.json.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/tmp/gt-bench/flightsql-check}
GT=${GT_BUILD:-${CARGO_TARGET_DIR:-$HERE/../target}/release/gt}
SINK=${SINK_BUILD:-${CARGO_TARGET_DIR:-$HERE/../target}/release/examples/flightsql_sink}
NPS_BIN=${NPS_BIN:-$NVTRACE_SDK_DIR/build/bin}
PY=${PY:-python3}
SECS=${SECS:-45}
CPUS=${CPUS:-4}
SPORT=${SPORT:-50551}
CAP_MIB=${CAP_MIB:-250}
VARIANTS=${VARIANTS:-plain dict dict-zstd plain-cap}

variant() {
  case $1 in
    plain) printf 'dictionaries = false\ncompression = "none"\n' ;;
    dict) printf 'dictionaries = true\ncompression = "none"\n' ;;
    dict-zstd) printf 'dictionaries = true\ncompression = "zstd"\n' ;;
    dict-lz4) printf 'dictionaries = true\ncompression = "lz4"\n' ;;
    plain-cap) printf 'dictionaries = false\ncompression = "none"\nmax_send_mib_s = %s\n' "$CAP_MIB" ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}

summarize() {
$PY - "$OUT" $VARIANTS <<'PY'
import glob, json, os, socket, sys
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
out, variants = sys.argv[1], sys.argv[2:]
MiB = 1 << 20

def dist(xs):
    xs = sorted(xs)
    n = len(xs)
    q = lambda p: xs[round((n - 1) * p)] if n else 0
    return {"seconds": n, "total": sum(xs), "mean": sum(xs) / n if n else 0, "p50": q(0.5), "p90": q(0.9), "max": xs[-1] if n else 0,
            "active": sum(1 for x in xs if x > 0)}

res = {}
for v in variants:
    d = f"{out}/{v}"
    r = {"variant": v}
    try:
        load = json.loads(open(f"{d}/load.log").read().strip().splitlines()[-1])
        prof = json.load(open(f"{d}/profile.json"))
    except (OSError, ValueError, IndexError) as e:
        res[v] = {"error": f"{type(e).__name__}: {e}"}
        continue
    f = prof["flightsql"]
    if not f:
        res[v] = {"error": "gt did not stream (no flightsql report)"}
        continue
    r["profile"] = {k: prof.get(k) for k in ("seconds", "activities", "snapshots", "lost_activities", "snapshot_stall_seconds", "drain_seconds", "cpus")}
    r["stream"] = {k: f[k] for k in ("rows", "server_rows", "pm_samples", "dropped_rows", "puts", "errors", "retries", "bytes", "bytes_per_row", "put_seconds")}
    r["gt"] = dist(f["traffic"]["per_second"])
    # wire: loopback bytes per second (rx counts each byte once), gt CPU
    rows = [l.split() for l in open(f"{d}/wire.txt") if len(l.split()) == 4]
    rows = [(float(t), int(b), int(c), int(m)) for t, b, c, m in rows]
    wire = [max(0, b1 - b0) / (t1 - t0) for (t0, b0, _, _), (t1, b1, _, _) in zip(rows, rows[1:]) if t1 > t0]
    r["wire"] = dist([round(w) for w in wire])
    r["wire"]["total"] = rows[-1][1] - rows[0][1] if rows else 0
    hz = os.sysconf("SC_CLK_TCK")
    if len(rows) > 1:
        cpu_s = (rows[-1][2] - rows[0][2]) / hz
        r["gt_cpu"] = {"cpu_seconds": cpu_s, "cores_mean": cpu_s / (rows[-1][0] - rows[0][0]), "rss_max_mib": max(x[3] for x in rows) / 1024}
    srv = [json.loads(l)["bytes"] for l in open(f"{d}/server-traffic.jsonl") if l.strip()]
    r["server"] = dist(srv)
    try:
        r["sink"] = json.load(open(f"{d}/sink.json"))
    except ValueError:
        r["sink"] = open(f"{d}/sink.json").read()[:300]
    # what arrived, against the load's own counts
    t = ds.dataset(f"{d}/sink/gt_trace", format="parquet").to_table(columns=["host", "kind", "pid", "graph_id", "start_ns", "correlation_id"])
    t = t.unify_dictionaries() if hasattr(t, "unify_dictionaries") else t
    kind = pc.cast(t["kind"], pa.string())
    k = t.filter(pc.and_(pc.equal(kind, "kernel"), pc.equal(t["pid"], load["pid"])))
    k = k.filter(pc.and_(pc.greater_equal(k["start_ns"], load["start_utc_ns"]), pc.less(k["start_ns"], load["end_utc_ns"])))
    graph = k.filter(pc.is_valid(k["graph_id"])).num_rows
    hosts = pc.unique(pc.cast(t["host"], pa.string())).to_pylist()
    keyed = pa.table({"pid": t["pid"], "c": t["correlation_id"], "s": t["start_ns"], "k": kind})
    r["check"] = {"rows": t.num_rows, "graph_kernels": graph, "eager_kernels": k.num_rows - graph,
                  "expected_graph": load["graph_kernels"], "expected_eager": load["eager_kernels"],
                  "hosts": hosts, "host_ok": hosts == [socket.gethostname()],
                  "duplicates": t.num_rows - keyed.group_by(["pid", "c", "s", "k"]).aggregate([]).num_rows,
                  "rows_match_gt": t.num_rows == f["rows"]}
    r["gt_bytes_equal_server"] = f["bytes"] == (r["sink"].get("bytes") if isinstance(r["sink"], dict) else None)
    res[v] = r
json.dump(res, open(f"{out}/summary.json", "w"), indent=1)
m = lambda b: f"{b / MiB:7.1f}"
print(f"{'variant':10} {'rows':>10} {'drop':>5} {'B/row':>6} | {'gt MiB/s: mean':>14} {'p50':>7} {'p90':>7} {'max':>7} | {'wire mean':>9} {'max':>7} {'wire/gt':>7} | {'cpu':>5} | check")
for v, r in res.items():
    if "error" in r:
        print(f"{v:10} {r['error']}")
        continue
    s, g, w, c = r["stream"], r["gt"], r["wire"], r["check"]
    ok = (c["graph_kernels"], c["eager_kernels"]) == (c["expected_graph"], c["expected_eager"]) and c["duplicates"] == 0 and c["host_ok"] and c["rows_match_gt"]
    print(f"{v:10} {s['rows']:>10} {s['dropped_rows']:>5} {s['bytes_per_row']:6.1f} | {m(g['mean']):>14} {m(g['p50'])} {m(g['p90'])} {m(g['max'])} | "
          f"{m(w['mean']):>9} {m(w['max'])} {w['total'] / max(1, s['bytes']):7.3f} | {r.get('gt_cpu', {}).get('cores_mean', 0):5.2f} | "
          f"{'OK' if ok else 'MISMATCH'} graph {c['graph_kernels']}/{c['expected_graph']} eager {c['eager_kernels']}/{c['expected_eager']} dup {c['duplicates']}")
    st = r["sink"]["tables"].get("gt_trace", {}) if isinstance(r["sink"], dict) else {}
    p = r["profile"]
    print(f"{'':10} DoPuts {s['puts']} in {s['put_seconds']:.1f} s (sink: receive {st.get('receive_s', 0):.1f} s, write {st.get('write_s', 0):.1f} s); "
          f"service lost {p.get('lost_activities')}, stall {p.get('snapshot_stall_seconds') or 0:.1f} s; drained {p['drain_seconds']:.1f} s")
PY
}
if [[ -n "${SUMMARY_ONLY:-}" ]]; then summarize; exit; fi

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$GT" "$OUT/gt"; cp "$SINK" "$OUT/flightsql_sink"   # stable copies: a rebuild must not replace running binaries
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_NSIGHT_PERF_FORCE_DISABLE_CUDA_TRACING CUDA_NSIGHT_PERF_FORCE_DISABLE_NVTX_TRACING
log() { echo "=== $(date +%T) $*"; }
lo_bytes() { awk '$1 == "lo:" { print $2 }' /proc/net/dev; }
# CPU seconds of every thread of a process, by thread name: "name seconds" lines, busiest first
threads() { for t in /proc/$1/task/*; do s=$(cat "$t/stat" 2>/dev/null) || continue; c=${s#*(}; c=${c%%)*}; r=${s##*) }
  echo "${c// /_} $(echo "$r" | awk -v hz="$(getconf CLK_TCK)" '{ printf "%.2f", ($12 + $13) / hz }')"; done |
  awk '{ s[$1] += $2 } END { for (k in s) printf "%-18s %8.2f\n", k, s[k] }' | sort -k2 -nr; }

CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=${HOSTDRV:-} setsid "$NPS_BIN/nvidia-nsight-perf" > "$OUT/service.log" 2>&1 &
S=$!
CUDA_VISIBLE_DEVICES=0,1 "$NPS_BIN/nvtrace_poll_until_service_ready" --timeout-sec 120 > "$OUT/poll.log" 2>&1 || { echo "service not ready"; kill -TERM $S; exit 1; }

for v in $VARIANTS; do
  D="$OUT/$v"; mkdir -p "$D"
  { printf '[collect]\nwork_dir = "%s/live"\ndiag_file = "%s/diag.jsonl"\n\n[flightsql]\nenabled = true\nendpoint = "grpc://127.0.0.1:%s"\n' "$D" "$D" "$SPORT"; variant "$v"
    printf '\n[limits]\nmax_cpus = %s\n' "$CPUS"; } > "$D/config.toml" || exit 1
  log "$v: sink on :$SPORT"
  "$OUT/flightsql_sink" --listen "127.0.0.1:$SPORT" --out "$D/sink" --traffic "$D/server-traffic.jsonl" > "$D/sink.json" 2> "$D/sink.log" &
  K=$!
  sleep 1
  # long enough for the load's warm-up (torch import, graph capture), its SECS and the last snapshot after it
  log "$v: gt profile -d $((SECS + 25))s"
  CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$NPS_BIN "$OUT/gt" profile -d "$((SECS + 25))s" -c "$D/config.toml" --json > "$D/profile.json" 2> "$D/gt.log" &
  G=$!
  sleep 3   # streaming before the load starts
  CUDA_VISIBLE_DEVICES=0 $PY "$HERE/graphload.py" --nodes 300 --eager 20 --secs "$SECS" --sleep-us 200 > "$D/load.log" 2>&1 &
  L=$!
  # every second: time, loopback bytes, gt CPU ticks (utime + stime), gt RSS (KiB)
  ticks=0; rss=0
  while kill -0 $G 2>/dev/null; do
    st=$(cat /proc/$G/stat 2>/dev/null) || break
    ticks=$(echo "$st" | awk '{ print $14 + $15 }')
    rss=$(awk '/^VmRSS:/ { print $2 }' /proc/$G/status 2>/dev/null || echo "$rss")
    echo "$(date +%s.%N) $(lo_bytes) $ticks ${rss:-0}" >> "$D/wire.txt"
    threads $G > "$D/gt-threads.txt"; threads $K > "$D/sink-threads.txt"   # the last one before gt exits remains
    sleep 1
  done
  wait $G; echo "gt rc=$?"
  echo "$(date +%s.%N) $(lo_bytes) $ticks ${rss:-0}" >> "$D/wire.txt"   # the drain after the last sample
  wait $L
  kill -TERM $K; wait $K
  tail -3 "$D/gt.log"
  echo "  gt threads (CPU s):"; head -6 "$D/gt-threads.txt" | sed 's/^/    /'
  echo "  sink threads (CPU s):"; head -4 "$D/sink-threads.txt" | sed 's/^/    /'
done
kill -TERM $S; wait $S 2>/dev/null

summarize
