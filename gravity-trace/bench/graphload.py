"""Synthetic CUDA-graph load with known counts, for checking the Nsight Perf live consumer's completeness.

Replays one graph of --nodes tiny kernels plus --eager eager kernels per iteration for --secs seconds and prints
how many graph-node and eager kernels it launched (ground truth for `gt exporter --diag`).
"""
import argparse
import json
import os
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--nodes", type=int, default=300)
p.add_argument("--eager", type=int, default=20)
p.add_argument("--secs", type=float, default=60)
p.add_argument("--sync-every", type=int, default=8)
p.add_argument("--sleep-us", type=int, default=0, help="pause after each iteration (rate control)")
p.add_argument("--device", type=int, default=0)
p.add_argument("--touch-all", action="store_true", help="also create a context on every visible GPU (like TP workers / NCCL)")
a = p.parse_args()

torch.cuda.set_device(a.device)
if a.touch_all:
    for d in range(torch.cuda.device_count()):
        torch.ones(1, device=f"cuda:{d}")
    torch.cuda.synchronize()

x = torch.ones(1024, device="cuda")
y = torch.ones(1024, device="cuda")
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):  # warm up outside the graph (allocator, kernels loaded)
    for _ in range(a.nodes):
        x.mul_(1.0000001)
torch.cuda.current_stream().wait_stream(side)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(a.nodes):
        x.mul_(1.0000001)
torch.cuda.synchronize()

t0_ns = time.time_ns()
t0 = time.time()
n = 0
while time.time() - t0 < a.secs:
    g.replay()
    for _ in range(a.eager):
        y.add_(1e-7)
    n += 1
    if n % a.sync_every == 0:
        torch.cuda.synchronize()
    if a.sleep_us:
        time.sleep(a.sleep_us / 1e6)
torch.cuda.synchronize()
print(json.dumps({"pid": os.getpid(), "device": a.device, "replays": n, "graph_kernels": n * a.nodes, "eager_kernels": n * a.eager,
                  "start_utc_ns": t0_ns, "end_utc_ns": time.time_ns()}), flush=True)
