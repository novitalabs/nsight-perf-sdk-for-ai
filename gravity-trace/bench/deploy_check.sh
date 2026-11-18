#!/usr/bin/env bash
# What a container deployment of the Nsight Perf service, gt and the workloads needs (docs/deployment.md), checked in
# one pod. Each part runs the way it would in a container of its own: own PID, IPC and mount namespaces, its own
# /dev/shm and /tmp, its own user and capabilities. Only the socket directory /run/nvidia/nsight-perf and /dev/nvidia*
# are shared, as a hostPath mount and the GPU runtime share them between containers.
#
#   service    root in the pod's (= host) PID namespace, Docker's default capabilities + SYS_ADMIN (SYS_PTRACE
#              dropped: SVC_DROP), the host-matched libcuda ($HOSTDRV); GPUs 0,1
#   gt-a       uid 65534, no capabilities, own namespaces, sockets read-only -> `gt export`, persisted trace
#   gt-nodev   the same without /dev/nvidia*                             -> sees no GPU
#   load-a     uid 1000, no capabilities, own namespaces, sockets read-only, GPU 0 -> traced
#   load-b     the same on GPU 1 with /run/nvidia hidden                 -> not traced
#   bash bench/deploy_check.sh       # in a pod with hostPID and CAP_SYS_ADMIN (unshare, setpriv); GPUs 0,1 idle
#   SVC_DROP=-sys_admin,-sys_ptrace bash bench/deploy_check.sh   # the service without SYS_ADMIN (no GPU data when
#                                    # the driver restricts profiling: RmProfilingAdminOnly: 1)
#   HOSTDRV= bash bench/deploy_check.sh    # the service on the libcuda the image resolves (CUDA 13.5 compat here)
#   SUMMARY_ONLY=1 bash bench/deploy_check.sh
# Output in $OUT: summary.json (counts per GPU / pid / graph-node or eager kernel against the loads' own), logs.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/tmp/gt-bench/deploy-check}
GT=${GT_BUILD:-${CARGO_TARGET_DIR:-$HERE/../target}/release/gt}
NPS_BIN=${NPS_BIN:-$NVTRACE_SDK_DIR/build/bin}
PY=${PY:-python3}
HOSTDRV=${HOSTDRV-}   # a directory of symlinks to the host driver's libcuda; "" = the one the image resolves
SECS=${SECS:-20}
SVC_DROP=${SVC_DROP--sys_ptrace}

summarize() {
$PY - "$OUT" <<'PY'
import glob, json, sys
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
out = sys.argv[1]
res = {}
for n in ("load-a", "load-b"):
    text = open(f"{out}/{n}.log").read().strip()
    try:
        res[n] = json.loads(text.splitlines()[-1])
    except (ValueError, IndexError):
        res[n] = {"error": text[-400:]}
for g in ("gt-a", "gt-nodev"):
    r = {"stdout": open(f"{out}/{g}/out.txt").read().strip(), "stderr": open(f"{out}/{g}/err.txt").read().strip().splitlines()[-3:]}
    files = sorted(f for f in glob.glob(f"{out}/{g}/traces/*.parquet") if not f.endswith(".pm.parquet"))
    r["pm_rows"] = sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob(f"{out}/{g}/traces/*.pm.parquet"))
    r["rows"], parts = 0, []
    for f in files:
        t = pq.read_table(f, columns=["kind", "pid", "device_id", "graph_id"])
        r["rows"] += t.num_rows
        k = t.filter(pc.equal(t["kind"].cast(pa.string()), "kernel"))
        parts.append(np.stack([pc.fill_null(k["device_id"].cast(pa.int64()), -1).to_numpy().astype(np.int64), pc.fill_null(k["pid"].cast(pa.int64()), -1).to_numpy().astype(np.int64),
                               pc.is_valid(k["graph_id"]).to_numpy(zero_copy_only=False).astype(np.int64)], axis=1))
    if parts:
        keys, counts = np.unique(np.concatenate(parts), axis=0, return_counts=True)
        r["kernels"] = [{"device_id": int(d), "pid": int(p), "graph": bool(gr), "count": int(c)} for (d, p, gr), c in zip(keys, counts)]
    res[g] = r
def kernels(g, device, graph):
    return sum(k["count"] for k in res[g].get("kernels", []) if k["device_id"] == device and k["graph"] == graph)
res["check"] = {
    "load_a_graph_kernels_traced": f'{kernels("gt-a", 0, True)} of {res["load-a"].get("graph_kernels")}',
    "load_a_eager_kernels_traced": f'{kernels("gt-a", 0, False)} (the load counts {res["load-a"].get("eager_kernels")} in its timed loop, plus its warm-up)',
    "load_b_kernels_traced": kernels("gt-a", 1, True) + kernels("gt-a", 1, False),
    "gt_nodev_kernels": sum(k["count"] for k in res["gt-nodev"].get("kernels", [])),
    "pm_rows": {g: res[g]["pm_rows"] for g in ("gt-a", "gt-nodev")},
}
json.dump(res, open(f"{out}/summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))
PY
}
if [[ -n "${SUMMARY_ONLY:-}" ]]; then summarize; exit; fi

rm -rf "$OUT"; mkdir -p "$OUT"; chmod 755 "$OUT"
cp "$GT" "$HERE/graphload.py" "$OUT/"; chmod 755 "$OUT/gt"; chmod 644 "$OUT/graphload.py"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_NSIGHT_PERF_FORCE_DISABLE_CUDA_TRACING CUDA_NSIGHT_PERF_FORCE_DISABLE_NVTX_TRACING
log() { echo "=== $(date +%T) $*"; }
status() { grep -E '^(Uid|NSpid|CapEff)' "/proc/$1/status" | tr '\t' ' ' | paste -sd' '; }
# host pids of the processes named $1 whose command line contains $2 (not the unshare wrappers around them)
pids() { for p in $(pgrep -f -- "$2"); do [[ $(cat "/proc/$p/comm" 2>/dev/null) == "$1"* ]] && echo "$p"; done; }

# ctr 'PRE' SETPRIV_ARGS... -- CMD...: CMD as in a container of its own: new PID / IPC / mount namespaces, fresh
# /dev/shm and /tmp, PRE (shell, as root) for more mounts, then setpriv to the user and capabilities.
ctr() {
  local pre=$1; shift
  unshare --pid --fork --mount-proc --ipc --mount --propagation private -- \
    bash -c 'mount -t tmpfs tmpfs /dev/shm && mount -t tmpfs tmpfs /tmp && eval "$0" && exec setpriv "$@"' "$pre" "$@"
}
NOBODY=(--reuid=65534 --regid=65534 --clear-groups --inh-caps=-all --bounding-set=-all)
LOADER=(--reuid=1000 --regid=1000 --clear-groups --inh-caps=-all --bounding-set=-all)
# the socket directories mounted read-only, as the manifests in docs/deployment.md mount them
RO='for d in /run/nvidia/nsight-perf /run/nvidia/nvtx; do mount --bind $d $d && mount -o remount,bind,ro $d; done'
NODEV='mount -t tmpfs -o mode=755 tmpfs /dev && for n in null:3 zero:5 random:8 urandom:9; do mknod -m 666 /dev/${n%:*} c 1 ${n#*:}; done && mkdir /dev/shm && mount -t tmpfs tmpfs /dev/shm'

# svc LOG DROP: the service as root without the capabilities in DROP (setpriv syntax, "" = all of the pod's)
svc() {
  if [[ -n $2 ]]; then
    CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$HOSTDRV setsid setpriv --inh-caps="$2" --bounding-set="$2" -- "$NPS_BIN/nvidia-nsight-perf" > "$1" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$HOSTDRV setsid "$NPS_BIN/nvidia-nsight-perf" > "$1" 2>&1 &
  fi
  S=$!
}

log "service: root, host PID namespace, without ${SVC_DROP:-nothing}"
svc "$OUT/service.log" "$SVC_DROP"
CUDA_VISIBLE_DEVICES=0,1 "$NPS_BIN/nvtrace_poll_until_service_ready" --timeout-sec 120 > "$OUT/poll.log" 2>&1; echo "ready rc=$?"
echo "service $(status $S)" | tee "$OUT/status.txt"
ls -la /run/nvidia/nsight-perf/*/ > "$OUT/sockets.txt"

for g in gt-a gt-nodev; do
  mkdir -p "$OUT/$g"
  printf '[daemon]\nlisten = ""\npid_file = "%s/gt.pid"\nlog_file = "%s/gt.log"\ncaptures_dir = "%s/captures"\n[collect]\nwork_dir = "%s/live"\n[persist]\ndir = "%s/traces"\n[limits]\nmax_cpus = 4\n' \
    "$OUT/$g" "$OUT/$g" "$OUT/$g" "$OUT/$g" "$OUT/$g" > "$OUT/$g/config.toml"
  chown -R 65534:65534 "$OUT/$g"
done
log "gt-a: gt export as uid 65534, no capabilities, own namespaces; gt-nodev: the same without /dev/nvidia*"
ctr "$RO" "${NOBODY[@]}" -- env LD_LIBRARY_PATH="$NPS_BIN" "$OUT/gt" export $((SECS + 40))s -c "$OUT/gt-a/config.toml" > "$OUT/gt-a/out.txt" 2> "$OUT/gt-a/err.txt" &
GA=$!
ctr "$NODEV" "${NOBODY[@]}" -- env LD_LIBRARY_PATH="$NPS_BIN" "$OUT/gt" export $((SECS + 20))s -c "$OUT/gt-nodev/config.toml" > "$OUT/gt-nodev/out.txt" 2> "$OUT/gt-nodev/err.txt" &
GN=$!
sleep 5
log "load-a: uid 1000, GPU 0; load-b: uid 1000, GPU 1, /run/nvidia hidden"
ctr "$RO" "${LOADER[@]}" -- env CUDA_VISIBLE_DEVICES=0 HOME=/tmp "$PY" "$OUT/graphload.py" --nodes 300 --eager 20 --secs "$SECS" --sleep-us 200 > "$OUT/load-a.log" 2>&1 &
LA=$!
ctr 'mount -t tmpfs tmpfs /run/nvidia' "${LOADER[@]}" -- env CUDA_VISIBLE_DEVICES=1 HOME=/tmp "$PY" "$OUT/graphload.py" --nodes 301 --eager 20 --secs "$SECS" --sleep-us 200 > "$OUT/load-b.log" 2>&1 &
LB=$!
sleep 12
for p in $(pids gt "gt-a/config.toml"); do echo "gt-a $(status "$p")"; done | tee -a "$OUT/status.txt"
for p in $(pids gt "gt-nodev/config.toml"); do echo "gt-nodev $(status "$p")"; done | tee -a "$OUT/status.txt"
for n in 300 301; do
  for p in $(pids python "graphload.py --nodes $n "); do
    echo "load --nodes $n $(status "$p")"
    grep -E ' /run/nvidia/(nsight-perf|nvtx) ' "/proc/$p/mountinfo" | awk '{print "  mount " $5 " " $6}'
    awk '$6 ~ /^\/dev\/|memfd:|^\/run\// {print "  mapped " $6}' "/proc/$p/maps" | sort | uniq -c
  done
done | tee -a "$OUT/status.txt"
wait $LA; wait $LB
wait $GA; echo "gt-a rc=$?"
wait $GN; echo "gt-nodev rc=$?"
kill -TERM $S; wait $S 2>/dev/null

summarize
