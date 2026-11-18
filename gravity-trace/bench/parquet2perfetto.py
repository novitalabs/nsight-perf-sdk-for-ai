"""Trace Parquet (+ .pm.parquet) -> Perfetto / Chrome trace JSON (.json.gz) in Python, with the layout of gt's Rust
converter (gravity-trace-core chrome.rs `convert_from`): the same tracks, events, args, ac2g flows and PM counters, one
event per line, gzip level 1. For timing against `gt dump perfetto` (bench/perfetto_py_vs_rust.sh).

    --impl naive   row by row: a dict per event, json.dumps (how such a script is usually written)
    --impl vector  column-wise: the JSON text of every event is built by pyarrow.compute (Arrow's C++ kernels) and
                   written from the Arrow buffer, no Python object per row

Names are demangled with c++filt, once per distinct name. Prints the phase times as one JSON line on stderr.
"""
import argparse
import gzip
import json
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

GPU_PID_BASE = 1 << 23
PM_PID_BASE = GPU_PID_BASE + (1 << 20)
PROCESSES_META = b"gravity-trace.processes"
T = {}


def phase(name, t0):
    T[name] = round(time.perf_counter() - t0, 3)
    return time.perf_counter()


def demangle(names):
    """c++filt over the distinct names (one process)."""
    names = list(names)
    mangled = [n for n in names if n.startswith("_Z")]
    out = dict(zip(names, names))
    if mangled:
        r = subprocess.run(["c++filt"], input="\n".join(mangled) + "\n", capture_output=True, text=True, check=True)
        out.update(zip(mangled, r.stdout.splitlines()))
    return out


def processes(files):
    """pid -> track label, from the files' footers ("title (pid N)", else "exe (pid N)")."""
    labels = {}
    for f in files:
        md = pq.read_metadata(f).metadata or {}
        for p in json.loads(md.get(PROCESSES_META, b"[]")):
            labels[p["pid"]] = f"{p.get('title') or p.get('exe') or ''} (pid {p['pid']})"
    return labels


def header(base):
    return ('{"schemaVersion":1,"traceName":"gravity-trace","displayTimeUnit":"ms","baseTimeNanoseconds":%d,'
            '"deviceProperties":[],"traceEvents":[' % base)


def meta_events(base_procs, labels, streams, threads, multi):
    """The metadata events, in gt's order."""
    ev = []
    devices = sorted({d for d, _, _ in streams})
    for d in devices:
        ev.append({"name": "process_name", "ph": "M", "pid": GPU_PID_BASE + d, "args": {"name": f"GPU {d}"}})
        ev.append({"name": "process_sort_index", "ph": "M", "pid": GPU_PID_BASE + d, "args": {"sort_index": 1000 + d}})
    index = {p: i for i, p in enumerate(base_procs)}
    for d, p, s in sorted(streams):
        name = f"{labels.get(p, f'process (pid {p})')} stream {s}" if multi else f"stream {s}"
        tid = (index[p] << 32) | s if multi else s
        ev.append({"name": "thread_name", "ph": "M", "pid": GPU_PID_BASE + d, "tid": tid, "args": {"name": name}})
    for p in base_procs:
        ev.append({"name": "process_name", "ph": "M", "pid": p, "args": {"name": labels.get(p, f"process (pid {p})")}})
    for p, t in sorted(threads):
        ev.append({"name": "thread_name", "ph": "M", "pid": p, "tid": t, "args": {"name": f"thread {t}"}})
    return ev


# ---------------------------------------------------------------------------------------------------------------
# naive: rows as Python dicts


def naive(files, pms, out):
    t = time.perf_counter()
    labels = processes(files)
    tables = [pq.read_table(f) for f in files]
    t = phase("read", t)
    base, procs, streams, threads, api, names = None, set(), set(), set(), {}, set()
    for tab in tables:
        for b in tab.to_batches():
            for r in b.to_pylist():
                base = r["start_ns"] if base is None else min(base, r["start_ns"])
                procs.add(r["pid"])
                names.add(r["name"])
                if r["device_id"] is not None:
                    streams.add((r["device_id"], r["pid"], r["stream_id"] or 0))
                else:
                    tid = r["tid"] or 0
                    threads.add((r["pid"], tid))
                    api[(r["pid"], r["correlation_id"])] = (tid, r["start_ns"])
    base = base or 0
    procs = sorted(procs)
    multi = len(procs) > 1
    index = {p: i for i, p in enumerate(procs)}
    dem = demangle(names)
    t = phase("plan", t)
    dumps = json.JSONEncoder(separators=(",", ":")).encode
    gpu_cat = {"kernel": "kernel", "memcpy": "gpu_memcpy", "memset": "gpu_memset"}
    flowed, flows = set(), 0
    with gzip.open(out, "wt", compresslevel=1, encoding="utf-8", newline="") as f:
        f.write(header(base))
        first = True

        def emit(ev):
            nonlocal first
            f.write(("\n" if first else ",\n") + dumps(ev))
            first = False

        for ev in meta_events(procs, labels, streams, threads, multi):
            emit(ev)
        for tab in tables:
            for b in tab.to_batches():
                for r in b.to_pylist():
                    kind, dev = r["kind"], r["device_id"]
                    args = {}
                    if r["correlation_id"]:
                        args["correlation"] = r["correlation_id"]
                    if dev is None:
                        cat = "cuda_driver" if kind == "driver" else "cuda_runtime"
                        pid, tid = r["pid"], r["tid"] or 0
                    else:
                        cat = gpu_cat.get(kind, kind)
                        stream = r["stream_id"] or 0
                        pid, tid = GPU_PID_BASE + dev, ((index[r["pid"]] << 32) | stream if multi else stream)
                        args["device"] = dev
                        args["stream"] = stream
                        if r["graph_id"]:
                            args["graph id"] = r["graph_id"]
                            args["graph node id"] = r["graph_node_id"] or 0
                        if r["channel_id"] is not None:
                            args["channel"] = r["channel_id"]
                        if kind == "kernel":
                            if r["grid_x"] is not None:
                                args["grid"] = [r["grid_x"], r["grid_y"], r["grid_z"]]
                                args["block"] = [r["block_x"] or 0, r["block_y"] or 0, r["block_z"] or 0]
                                args["registers per thread"] = r["registers_per_thread"] or 0
                                args["shared memory"] = (r["static_shared_mem"] or 0) + (r["dynamic_shared_mem"] or 0)
                            if r["device_launched"]:
                                args["device launched"] = True
                        if r["bytes"] is not None:
                            args["bytes"] = r["bytes"]
                            if r["end_ns"] > r["start_ns"]:
                                args["memory bandwidth (GB/s)"] = r["bytes"] / (r["end_ns"] - r["start_ns"])
                    emit({"ph": "X", "cat": cat, "name": dem[r["name"]], "pid": pid, "tid": tid,
                          "ts": (r["start_ns"] - base) / 1000, "dur": (r["end_ns"] - r["start_ns"]) / 1000, "args": args})
                    corr = r["correlation_id"]
                    if dev is not None and corr and (r["pid"], corr) not in flowed:
                        flowed.add((r["pid"], corr))
                        a = api.get((r["pid"], corr))
                        if a:
                            flows += 1
                            emit({"ph": "s", "id": flows, "pid": r["pid"], "tid": a[0], "ts": (a[1] - base) / 1000,
                                  "cat": "ac2g", "name": "ac2g"})
                            emit({"ph": "f", "id": flows, "pid": pid, "tid": tid, "ts": (r["start_ns"] - base) / 1000,
                                  "cat": "ac2g", "name": "ac2g", "bp": "e"})
        counters = {}
        for p in pms:
            for b in pq.read_table(p).to_batches():
                for r in b.to_pylist():
                    pid = counters.get(r["gpu_uuid"])
                    if pid is None:
                        pid = counters[r["gpu_uuid"]] = PM_PID_BASE + len(counters)
                        emit({"name": "process_name", "ph": "M", "pid": pid, "args": {"name": f"{r['gpu_uuid']} PM counters"}})
                        emit({"name": "process_sort_index", "ph": "M", "pid": pid, "args": {"sort_index": 2000 + pid - PM_PID_BASE}})
                    emit({"ph": "C", "name": r["metric"], "pid": pid, "ts": (r["end_ns"] - base) / 1000, "args": {"value": r["value"]}})
        f.write("\n]}\n")
        t = phase("events+gzip", t)


# ---------------------------------------------------------------------------------------------------------------
# vector: the JSON text built column-wise by Arrow

NULL = pa.scalar(None, pa.string())


def s(a):
    return pc.cast(a, pa.string())


def cat(*parts):
    return pc.binary_join_element_wise(*parts, "")


def us(ns):
    """int64 ns -> "µs.fff" (exact, like gt)."""
    ns = pc.cast(ns, pa.int64())
    whole = pc.divide(ns, 1000)
    frac = pc.subtract(ns, pc.multiply(whole, 1000))
    return cat(s(whole), ".", pc.utf8_lpad(s(frac), width=3, padding="0"))


def json_strings(values):
    """A string array's values as JSON string literals."""
    return pa.array([json.dumps(v) for v in values.to_pylist()], pa.string())


def write_lines(f, events, first):
    """Write a string array of events, ",\\n"-separated, straight from its Arrow buffers."""
    if len(events) == 0:
        return first
    events = cat(",\n", events)
    events = events.combine_chunks() if isinstance(events, pa.ChunkedArray) else events
    offs = events.buffers()[1]
    o = np.frombuffer(offs, dtype=np.int32, count=len(events) + 1, offset=events.offset * 4)
    data = memoryview(events.buffers()[2])[o[0]:o[-1]]
    f.write(data[1:] if first else data)  # the first event without the comma
    return False


def vector(files, pms, out):
    t = time.perf_counter()
    labels = processes(files)
    tab = pa.concat_tables([pq.read_table(f) for f in files]).unify_dictionaries().combine_chunks()
    n = tab.num_rows
    tab = tab.append_column("row", pa.array(range(n), pa.int64()))
    t = phase("read", t)

    gpu = pc.is_valid(tab["device_id"])
    base = pc.min(tab["start_ns"]).as_py() or 0
    procs = sorted(pc.unique(tab["pid"]).to_pylist())
    multi = len(procs) > 1
    g = tab.filter(gpu)
    h = tab.filter(pc.invert(gpu))
    streams = set(zip(*[c.to_pylist() for c in pa.table({"d": g["device_id"], "p": g["pid"], "s": pc.fill_null(g["stream_id"], 0)})
                        .group_by(["d", "p", "s"]).aggregate([]).columns]))
    threads = set(zip(*[c.to_pylist() for c in pa.table({"p": h["pid"], "t": pc.fill_null(h["tid"], 0)})
                        .group_by(["p", "t"]).aggregate([]).columns]))
    t = phase("plan", t)

    # names and categories per dictionary entry
    name = tab["name"].chunk(0)
    dem = demangle(name.dictionary.to_pylist())
    name_json = pc.take(json_strings(pa.array([dem[v] for v in name.dictionary.to_pylist()])), name.indices)
    kind = s(tab["kind"])
    gpu_cat = pc.if_else(pc.equal(kind, "kernel"), "kernel",
                         pc.if_else(pc.equal(kind, "memcpy"), "gpu_memcpy", pc.if_else(pc.equal(kind, "memset"), "gpu_memset", kind)))
    cat_ = pc.if_else(gpu, gpu_cat, pc.if_else(pc.equal(kind, "driver"), "cuda_driver", "cuda_runtime"))
    stream = pc.fill_null(tab["stream_id"], 0)
    pidx = pc.cast(pc.index_in(tab["pid"], value_set=pa.array(procs, pa.uint32())), pa.uint64())
    gtid = pc.add(pc.multiply(pidx, pa.scalar(1 << 32, pa.uint64())), pc.cast(stream, pa.uint64())) if multi else pc.cast(stream, pa.uint64())
    pid = pc.if_else(gpu, pc.add(pc.cast(tab["device_id"], pa.uint64()), pa.scalar(GPU_PID_BASE, pa.uint64())), pc.cast(tab["pid"], pa.uint64()))
    tid = pc.if_else(gpu, gtid, pc.fill_null(tab["tid"], pa.scalar(0, pa.uint64())))
    start, end = tab["start_ns"], tab["end_ns"]
    corr = tab["correlation_id"]

    def frag(cond, text):
        """`,text` where cond holds, else "" (no nulls: pyarrow's null_handling="skip" join lost rows here)."""
        return pc.fill_null(pc.if_else(cond, cat(",", text), NULL), "")

    is_kernel = pc.and_(gpu, pc.equal(kind, "kernel"))
    has_grid = pc.and_(is_kernel, pc.is_valid(tab["grid_x"]))
    graph = pc.and_(gpu, pc.not_equal(pc.fill_null(tab["graph_id"], 0), 0))
    nbytes = tab["bytes"]
    dur = pc.subtract(end, start)
    z = lambda c: s(pc.fill_null(tab[c], 0))
    triple = lambda a, b, c: cat("[", z(a), ",", z(b), ",", z(c), "]")
    shared = s(pc.add(pc.cast(pc.fill_null(tab["static_shared_mem"], 0), pa.int64()), pc.cast(pc.fill_null(tab["dynamic_shared_mem"], 0), pa.int64())))
    args = cat(
        frag(pc.not_equal(corr, 0), cat('"correlation":', s(corr))),
        frag(gpu, cat('"device":', s(tab["device_id"]), ',"stream":', s(stream))),
        frag(graph, cat('"graph id":', s(tab["graph_id"]), ',"graph node id":', z("graph_node_id"))),
        frag(pc.and_(gpu, pc.is_valid(tab["channel_id"])), cat('"channel":', s(tab["channel_id"]))),
        frag(has_grid, cat('"grid":', triple("grid_x", "grid_y", "grid_z"), ',"block":', triple("block_x", "block_y", "block_z"),
                           ',"registers per thread":', z("registers_per_thread"), ',"shared memory":', shared)),
        frag(pc.and_(is_kernel, pc.fill_null(tab["device_launched"], False)), '"device launched":true'),
        frag(pc.and_(gpu, pc.is_valid(nbytes)), cat('"bytes":', s(nbytes))),
        frag(pc.and_(pc.and_(gpu, pc.is_valid(nbytes)), pc.greater(dur, 0)),
             cat('"memory bandwidth (GB/s)":', s(pc.divide(pc.cast(nbytes, pa.float64()), pc.cast(dur, pa.float64()))))),
    )
    args = pc.utf8_slice_codeunits(args, 1)  # the first fragment's comma
    events = cat('{"ph":"X","cat":"', cat_, '","name":', name_json, ',"pid":', s(pid), ',"tid":', s(tid),
                 ',"ts":', us(pc.subtract(start, base)), ',"dur":', us(dur), ',"args":{', args, "}}")
    t = phase("events", t)

    # flows: the first GPU activity of each (pid, correlation) and the API call of that correlation (the last one)
    gk = pa.table({"pid": tab["pid"], "corr": corr, "row": tab["row"]}).filter(pc.and_(gpu, pc.not_equal(corr, 0)))
    first = gk.group_by(["pid", "corr"]).aggregate([("row", "min")]).rename_columns(["pid", "corr", "grow"])
    hk = pa.table({"pid": tab["pid"], "corr": corr, "row": tab["row"]}).filter(pc.invert(gpu))
    last = hk.group_by(["pid", "corr"]).aggregate([("row", "max")]).rename_columns(["pid", "corr", "arow"])
    fl = first.join(last, ["pid", "corr"], join_type="inner").sort_by("grow")
    gr, ar = fl["grow"], fl["arow"]
    ids = s(pa.array(range(1, fl.num_rows + 1), pa.int64()))
    fs = cat('{"ph":"s","id":', ids, ',"pid":', s(pc.take(tab["pid"], ar)), ',"tid":', s(pc.take(tid, ar)),
             ',"ts":', us(pc.subtract(pc.take(start, ar), base)), ',"cat":"ac2g","name":"ac2g"}')
    ff = cat('{"ph":"f","id":', ids, ',"pid":', s(pc.take(pid, gr)), ',"tid":', s(pc.take(tid, gr)),
             ',"ts":', us(pc.subtract(pc.take(start, gr), base)), ',"cat":"ac2g","name":"ac2g","bp":"e"}')
    t = phase("flows", t)

    with gzip.open(out, "wb", compresslevel=1) as f:
        f.write(header(base).encode())
        dumps = json.JSONEncoder(separators=(",", ":")).encode
        meta = [dumps(e) for e in meta_events(procs, labels, streams, threads, multi)]
        f.write(("\n" + ",\n".join(meta)).encode())
        first = not meta
        for part in (events, fs, ff):
            first = write_lines(f, part, first)
        for p in pms:
            pm = pq.read_table(p).combine_chunks()
            uuids = s(pm["gpu_uuid"])
            order = list(dict.fromkeys(uuids.to_pylist()))
            cpid = pa.array([PM_PID_BASE + i for i in range(len(order))], pa.uint64())
            pmeta = []
            for i, u in enumerate(order):
                pmeta.append(dumps({"name": "process_name", "ph": "M", "pid": PM_PID_BASE + i, "args": {"name": f"{u} PM counters"}}))
                pmeta.append(dumps({"name": "process_sort_index", "ph": "M", "pid": PM_PID_BASE + i, "args": {"sort_index": 2000 + i}}))
            f.write((",\n" + ",\n".join(pmeta)).encode())
            metric = pm["metric"].chunk(0) if isinstance(pm["metric"], pa.ChunkedArray) else pm["metric"]
            if pa.types.is_dictionary(metric.type):
                mj = pc.take(json_strings(metric.dictionary), metric.indices)
            else:
                mj = json_strings(metric)
            ce = cat('{"ph":"C","name":', mj, ',"pid":', s(pc.take(cpid, pc.index_in(uuids, value_set=pa.array(order)))),
                     ',"ts":', us(pc.subtract(pm["end_ns"], base)), ',"args":{"value":', s(pm["value"]), "}}")
            write_lines(f, ce, False)
        f.write(b"\n]}\n")
    t = phase("write+gzip", t)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--impl", choices=["naive", "vector"], required=True)
    a.add_argument("--pm", action="append", default=[], help=".pm.parquet file (repeatable)")
    a.add_argument("-o", "--out", required=True)
    a.add_argument("files", nargs="+")
    o = a.parse_args()
    t0 = time.perf_counter()
    (naive if o.impl == "naive" else vector)(o.files, o.pm, o.out)
    T["total"] = round(time.perf_counter() - t0, 3)
    print(json.dumps({"impl": o.impl, **T}), file=sys.stderr)


if __name__ == "__main__":
    main()
