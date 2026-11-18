#!/usr/bin/env bash
# Does a CUDA process that started BEFORE nvidia-nsight-perf get traced once the service is up?
#   A: torch kernel loop started first (no service yet)
#   B: same loop started after the service is ready (control)
# then one 5 s streaming capture, decoded with `gt import`; kernels counted per pid.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
GPU=${GPU:-0}
NPS_BIN=${NPS_BIN:-$NVTRACE_SDK_DIR/build/bin}
GT=${GT_BUILD:-${CARGO_TARGET_DIR:-$HERE/../target}/release/gt}
OUT=${OUT:-/tmp/gt-bench/nvp-order}
PY=${PY:-python3}
rm -rf "$OUT"; mkdir -p "$OUT"
loop='import os,time,torch
x=torch.ones(1<<20,device="cuda"); t=time.time()
print("pid",os.getpid(),flush=True)
while time.time()-t<float(os.environ["SECS"]):
    for _ in range(100): x.mul_(1.0000001)
    torch.cuda.synchronize(); time.sleep(0.01)'
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID
SECS=70 $PY -c "$loop" > "$OUT/a.log" 2>&1 &
A=$!
sleep 15                                   # A is fully initialised and launching kernels before the service exists
LD_LIBRARY_PATH=${HOSTDRV:-} setsid "$NPS_BIN/nvidia-nsight-perf" > "$OUT/service.log" 2>&1 &
S=$!
"$NPS_BIN/nvtrace_poll_until_service_ready" --timeout-sec 60 > "$OUT/poll.log" 2>&1; echo "poll rc=$?"
SECS=40 $PY -c "$loop" > "$OUT/b.log" 2>&1 &
B=$!
sleep 15
LD_LIBRARY_PATH=$NPS_BIN "$NPS_BIN/nvtrace_streaming_consumer" --output-dir "$OUT/cap" --duration-ms 5000 > "$OUT/consumer.log" 2>&1
DB=$(ls -d "$OUT"/cap/*/ | head -1)
LD_LIBRARY_PATH=$NPS_BIN "$GT" import "$DB" -o "$OUT/imp" --no-chrome --json > "$OUT/import.json" 2>"$OUT/import.err"
wait $A $B
kill -TERM $S; wait $S 2>/dev/null
echo "A=$A ($(head -1 "$OUT/a.log")) B=$B ($(head -1 "$OUT/b.log"))"
$PY - "$OUT" "$A" "$B" <<'EOF'
import glob, json, os, re, sys, collections
import pyarrow.parquet as pq
out, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = json.load(open(f"{out}/import.json"))
print("processes in capture:", r.get("processes"))
tf = [f for f in sorted(glob.glob(f"{out}/imp/*.parquet")) if re.fullmatch(r"\d+-\d+\.parquet|trace-nvp-.*\.parquet", os.path.basename(f))]
t = pq.read_table(tf[0], columns=["kind", "pid"]).to_pydict()
c = collections.Counter((p, k) for k, p in zip(t["kind"], t["pid"]))
for (p, k), n in sorted(c.items()):
    print(f"pid {p} ({'A: started before service' if p == a else 'B: started after' if p == b else '?'}) {k}: {n}")
EOF
