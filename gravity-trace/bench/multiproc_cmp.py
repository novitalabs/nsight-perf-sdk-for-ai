"""Kernels per host pid in a multiproc_check.sh run against what each graphload.py launched.

graph: the graph-node kernels of the replays; eager: the eager kernels plus the warm-up (--nodes kernels and the two
fills of torch.ones). Every load must be +0 / +0, with ns_pid its pid in its own namespace.
"""
import collections
import glob
import json
import re
import sys

import pyarrow.parquet as pq

out = sys.argv[1]
pids = {}  # nodes -> (host pid, pid in its own namespace)
for line in open(f"{out}/pids.txt"):
    m = re.match(r"load (\d+) host pid (\d+) NSpid: ([\d ]+)", line)
    if m:
        pids[int(m.group(1))] = (int(m.group(2)), int(m.group(3).split()[-1]))
loads = {}
for f in glob.glob(f"{out}/load-*.log"):
    n = int(f.rsplit("-", 1)[1].split(".")[0])
    loads[n] = json.loads(open(f).read().strip().splitlines()[-1])

kernels = collections.Counter()
ns = collections.defaultdict(set)
for f in glob.glob(f"{out}/traces/*.parquet"):
    if f.endswith(".pm.parquet"):
        continue
    t = pq.read_table(f, columns=["kind", "pid", "ns_pid", "graph_id"]).to_pandas()
    for p, g in t.groupby("pid"):
        ns[p] |= set(g.ns_pid.dropna().astype(int).unique())
    k = t[t.kind.astype(str) == "kernel"]
    for (p, graph), c in k.groupby([k.pid, k.graph_id.notna()]).size().items():
        kernels[(p, graph)] += c

ok = True
for n in sorted(loads):
    host, own = pids[n]
    d = loads[n]
    dg = kernels[(host, True)] - d["graph_kernels"]
    de = kernels[(host, False)] - d["eager_kernels"] - (n + 2)
    good = dg == 0 and de == 0 and ns[host] == {own}
    ok &= good
    print(f"nodes {n}: pid {host} ns_pid {sorted(int(x) for x in ns[host])} (expected {own}): graph {dg:+d}, eager {de:+d}{'' if good else '  MISMATCH'}")
others = sorted({p for p, _ in kernels} - {h for h, _ in pids.values()})
if others:  # other CUDA processes on the GPUs (informational: the loads' counts are exact or not)
    print("kernels of other pids:", {p: kernels[(p, True)] + kernels[(p, False)] for p in others})
for line in open(f"{out}/metrics.txt"):
    if line.startswith("nvtrace_gpu_kernels_total{"):
        print("  " + re.sub(r'pod="[^"]*",host="[^"]*",', "", line.strip()))
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
