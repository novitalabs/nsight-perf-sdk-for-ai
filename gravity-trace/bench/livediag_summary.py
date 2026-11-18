"""Summarize `gt exporter --diag` output against graphload.py ground truth.

    python -I bench/livediag_summary.py /tmp/gt-bench/nvp-livediag/p4
"""
import glob
import json
import sys

d = sys.argv[1]
rows = [json.loads(x) for x in open(f"{d}/diag.jsonl") if x.strip()]
truth = []
for f in sorted(glob.glob(f"{d}/load*.log")):
    for line in open(f):
        if line.startswith("{"):
            truth.append(json.loads(line))
S = lambda k: sum(r.get(k, 0) for r in rows)  # noqa: E731
gk = sum(t["graph_kernels"] for t in truth)
ek = sum(t["eager_kernels"] for t in truth)
print(f"snapshots {len(rows)}  numbers {rows[0]['snapshot'] if rows else '-'}..{rows[-1]['snapshot'] if rows else '-'}")
gaps = [b["snapshot"] - a["snapshot"] for a, b in zip(rows, rows[1:]) if b["snapshot"] - a["snapshot"] != 1]
print(f"snapshot number gaps: {gaps[:10]}")
win = [(r["window_end_utc_ns"] - r["window_start_utc_ns"]) / 1e9 for r in rows]
ovl = [(a["window_end_utc_ns"] - b["window_start_utc_ns"]) / 1e6 for a, b in zip(rows, rows[1:])]
print(f"window s: min {min(win):.3f} max {max(win):.3f}; overlap with previous ms: min {min(ovl or [0]):.1f} max {max(ovl or [0]):.1f}")
print(f"history_truncated {sum(r['history_truncated'] for r in rows)}  backpressured {S('backpressured_records')}  "
      f"attach_incomplete {sum(not r['attach_complete'] for r in rows)}  dropped_attach {S('dropped_attach_records')}")
print(f"truth: graph kernels {gk:,}  eager kernels {ek:,}  (processes {len(truth)})")
print(f"node rows {S('node_rows'):,}  incomplete {S('node_rows_incomplete'):,}  before_window {S('node_rows_before_window'):,}  "
      f"seen_before {S('node_rows_seen_before'):,} (incomplete {S('node_rows_incomplete_seen_before'):,})")
print(f"kern rows {S('kern_rows'):,}  incomplete {S('kern_rows_incomplete'):,}  seen_before {S('kern_rows_seen_before'):,}")
print(f"upd node {S('upd_node'):,}  orphan nokey {S('upd_node_orphan_nokey'):,}  nomatch {S('upd_node_orphan_nomatch'):,}  no_start {S('upd_node_no_start'):,}")
print(f"upd kern {S('upd_kern'):,}  orphan {S('upd_kern_orphan'):,};  upd cmd {S('upd_cmd'):,} orphan {S('upd_cmd_orphan'):,};  "
      f"upd glaunch {S('upd_glaunch'):,} orphan {S('upd_glaunch_orphan'):,}")
print(f"lost node {S('lost_node'):,}  kern {S('lost_kern'):,}  cmd {S('lost_cmd'):,}")
gh = [sum(r["node_incomplete_gap"][i] for r in rows) for i in range(5)]
lh = [sum(r["lost_node_gap"][i] for r in rows) for i in range(5)]
ah = [sum(r["lost_node_age_snapshots"][i] for r in rows) for i in range(4)]
print(f"incomplete node gap (<1ms,<10ms,<100ms,<1s,>=1s): {gh}")
print(f"lost node gap: {lh}   lost age (3,4,5,>=6 snapshots): {ah}")
# Complete kernels the consumer mapped, per pid (cumulative counters in metrics.txt) vs the loads' ground truth.
import os, re  # noqa: E401,E402
got = {}
if os.path.exists(f"{d}/metrics.txt"):
    for line in open(f"{d}/metrics.txt"):
        m = re.match(r'nvtrace_activity_count_total\{.*pid="(\d+)".*kind="kernel".*\} (\d+)', line)
        if m:
            got[int(m.group(1))] = got.get(int(m.group(1)), 0) + int(m.group(2))
for tr in truth:
    want = tr["graph_kernels"] + tr["eager_kernels"]
    have = got.get(tr["pid"], 0)
    print(f"pid {tr['pid']} (GPU {tr.get('device', '?')}): launched {want:,} kernels, live mapped {have:,} ({have / want * 100 if want else 0:.2f}%)")
print(f"decode ms: mean {sum(r['decode_ms'] for r in rows) / max(len(rows), 1):.0f} max {max(r['decode_ms'] for r in rows):.0f}")
print("per snapshot: snap  win_s  node_rows  incompl  seen_bef  upd_node  orph_nokey  orph_nomatch  lost_node  backp  trunc")
for r in rows:
    if r["node_rows"] or r["lost_node"] or r["upd_node"]:
        print(f"  {r['snapshot']:>4} {(r['window_end_utc_ns'] - r['window_start_utc_ns']) / 1e9:6.2f} {r['node_rows']:>10,} {r['node_rows_incomplete']:>8,} "
              f"{r['node_rows_seen_before']:>9,} {r['upd_node']:>9,} {r['upd_node_orphan_nokey']:>10,} {r['upd_node_orphan_nomatch']:>12,} "
              f"{r['lost_node']:>9,} {r['backpressured_records']:>6} {int(r['history_truncated']):>5}")
