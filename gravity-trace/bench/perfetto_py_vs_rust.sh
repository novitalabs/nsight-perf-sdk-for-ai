#!/usr/bin/env bash
# Parquet -> Perfetto JSON (.json.gz): `gt dump perfetto` (Rust) against bench/parquet2perfetto.py (Python, naive and
# vectorized) on one trace file and its PM file: wall / CPU time, peak RSS, output size, and whether the outputs hold
# the same events (perfetto_cmp.py).
#   bash bench/perfetto_py_vs_rust.sh TRACE.parquet   # a trace file of `gt dump parquet` / a capture; results in $OUT
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${1:?usage: bash bench/perfetto_py_vs_rust.sh TRACE.parquet}
OUT=${OUT:-/tmp/gt-bench/perfetto-py-vs-rust}
GT=${GT_BUILD:-${CARGO_TARGET_DIR:-$HERE/../target}/release/gt}
PY=${PY:-python3}
RUNS=${RUNS:-3}
rm -rf "$OUT"; mkdir -p "$OUT/in"
cp "$SRC" "$OUT/in/"; PM=${SRC%.parquet}.pm.parquet; [[ -f $PM ]] && cp "$PM" "$OUT/in/"
cp "$GT" "$OUT/gt"
IN_TRACE=$OUT/in/$(basename "$SRC"); IN_PM=$OUT/in/$(basename "$PM")
NAME=$(basename "$SRC" .parquet); START=$(( ${NAME%-*} - 1000000000 )); END=$(( ${NAME#*-} + 1000000000 ))
$PY -c "import pyarrow.parquet as pq; m=pq.read_metadata('$IN_TRACE'); print(f'input: {m.num_rows} activities, {m.num_row_groups} row groups')" | tee "$OUT/summary.txt"
ls -la "$OUT/in" | tail -n +2 >> "$OUT/summary.txt"

# one measured run: wall, user+sys CPU and peak RSS of the command (a fresh parent per run: ru_maxrss of its child)
measure() {  # label out cmd...
    local label=$1 out=$2; shift 2
    $PY - "$label" "$out" "$@" <<'PY' | tee -a "$OUT/runs.jsonl"
import json, os, resource, subprocess, sys, time
label, out, cmd = sys.argv[1], sys.argv[2], sys.argv[3:]
t = time.perf_counter()
r = subprocess.run(cmd, capture_output=True, text=True)
wall = time.perf_counter() - t
u = resource.getrusage(resource.RUSAGE_CHILDREN)
phases = {}
for line in r.stderr.splitlines():
    if line.startswith("{"):
        phases = json.loads(line)
print(json.dumps({"run": label, "rc": r.returncode, "wall_s": round(wall, 2), "cpu_s": round(u.ru_utime + u.ru_stime, 2),
                  "max_rss_mib": round(u.ru_maxrss / 1024), "out_mib": round(os.path.getsize(out) / 2**20, 1) if os.path.exists(out) else None,
                  "phases": phases}))
if r.returncode:
    print(r.stderr[-2000:], file=sys.stderr)
PY
}

for i in $(seq "$RUNS"); do
    measure rust "$OUT/rust.json.gz" "$OUT/gt" dump perfetto "$START" "$END" --dir "$OUT/in" -o "$OUT/rust.json.gz" --no-flush
done
for i in $(seq "$RUNS"); do
    measure py-vector "$OUT/py-vector.json.gz" $PY "$HERE/parquet2perfetto.py" --impl vector --pm "$IN_PM" -o "$OUT/py-vector.json.gz" "$IN_TRACE"
done
measure py-naive "$OUT/py-naive.json.gz" $PY "$HERE/parquet2perfetto.py" --impl naive --pm "$IN_PM" -o "$OUT/py-naive.json.gz" "$IN_TRACE"

for impl in py-vector py-naive; do
    echo "rust vs $impl:" | tee -a "$OUT/summary.txt"
    $PY "$HERE/perfetto_cmp.py" "$OUT/rust.json.gz" "$OUT/$impl.json.gz" | tee -a "$OUT/summary.txt"
done
