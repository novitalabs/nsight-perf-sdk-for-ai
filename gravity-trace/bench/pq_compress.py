#!/usr/bin/env python3
"""How file length, row groups, zstd level, row order and encodings change the size of gt trace Parquet.

    python -I bench/pq_compress.py SAMPLE.parquet [--json OUT.json]

SAMPLE is a gt trace Parquet (e.g. `gt import` of a DeepSeek-V4 capture). Longer files are simulated by
repeating the sample with shifted timestamps (a decode step replays the same graph, so names / launch dims repeat
anyway; only timestamps change). Test files go to $TMPDIR (default /tmp/pqtest) and are removed.
"""
import argparse
import json
import os
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("sample")
ap.add_argument("--json")
a = ap.parse_args()

out = os.environ.get("TMPDIR", "/tmp/pqtest")
os.makedirs(out, exist_ok=True)
base = pq.read_table(a.sample)
n0 = base.num_rows
shift = pc.max(base["end_ns"]).as_py() - pc.min(base["start_ns"]).as_py() + 1_000_000
RG256, RG1M, RG4M = 256 * 1024, 1024 * 1024, 4 * 1024 * 1024
ZSTD = lambda level: dict(compression="zstd", compression_level=level)  # noqa: E731
DELTA_COLS = ("start_ns", "end_ns", "correlation_id")
STREAM_KEYS = [("pid", "ascending"), ("device_id", "ascending"), ("stream_id", "ascending"), ("start_ns", "ascending")]
results = []


def tiled(k):
    parts = []
    for i in range(k):
        t = base
        for col in ("start_ns", "end_ns"):
            t = t.set_column(t.schema.get_field_index(col), col, pc.add(t[col], pa.scalar(i * shift, t[col].type)))
        parts.append(t)
    return pa.concat_tables(parts)


def sort_per_group(t, rows):
    """What a streaming writer does: buffer one row group, sort it, write it."""
    return pa.concat_tables([t.slice(i, rows).sort_by(STREAM_KEYS) for i in range(0, t.num_rows, rows)])


def delta_opts(t, plain=()):
    """Timestamps / correlation ids (and `plain` columns) without dictionary; the timestamps delta-encoded."""
    keep = set(DELTA_COLS) | set(plain)
    return dict(use_dictionary=[c for c in t.column_names if c not in keep],
                column_encoding={c: "DELTA_BINARY_PACKED" for c in DELTA_COLS})


def write(group, label, t, **kw):
    path = f"{out}/test.parquet"
    t0 = time.time()
    pq.write_table(t, path, **kw)
    dt = time.time() - t0
    size = os.path.getsize(path)
    os.remove(path)
    r = {"group": group, "label": label, "rows": t.num_rows, "bytes_per_row": size / t.num_rows, "write_mrows_s": t.num_rows / dt / 1e6}
    results.append(r)
    print(f"{label:<52} rows {t.num_rows:>10,}  {r['bytes_per_row']:6.2f} B/row  write {r['write_mrows_s']:5.2f} M rows/s")
    return r


print(f"sample: {n0:,} rows from {os.path.basename(a.sample)} ({os.path.getsize(a.sample) / n0:.2f} B/row as written)")
t10, t30 = tiled(10), tiled(30)
print("-- file length (zstd 3, 256k-row groups, rows as exported)")
write("length", "1x (a ~3 s capture)", base, row_group_size=RG256, **ZSTD(3))
write("length", "10x", t10, row_group_size=RG256, **ZSTD(3))
write("length", "30x", t30, row_group_size=RG256, **ZSTD(3))
print("-- row group size (30x, zstd 3)")
write("rowgroup", "row groups 1M (1024 x 1024)", t30, row_group_size=RG1M, **ZSTD(3))
write("rowgroup", "row groups 4M", t30, row_group_size=RG4M, **ZSTD(3))
print("-- zstd level (30x, 256k)")
write("zstd", "zstd 9", t30, row_group_size=RG256, **ZSTD(9))
write("zstd", "zstd 19", t30, row_group_size=RG256, **ZSTD(19))
print("-- row order / columns / encodings (30x, zstd 3)")
s30 = t30.sort_by(STREAM_KEYS)
write("encode", "whole file sorted by pid, GPU, stream, time (256k)", s30, row_group_size=RG256, **ZSTD(3))
s30d = s30.drop_columns(["dur_ns"])
write("encode", "  + without dur_ns (256k)", s30d, row_group_size=RG256, **ZSTD(3))
write("encode", "  + delta-encoded timestamps (256k)", s30d, row_group_size=RG256, **delta_opts(s30d), **ZSTD(3))
write("encode", "  + row groups 1M", s30d, row_group_size=RG1M, **delta_opts(s30d), **ZSTD(3))
print("-- what a streaming writer can do: 1M-row groups, each sorted on its own, zstd 3")
d30 = t30.drop_columns(["dur_ns"])
write("stream", "unsorted, without dur_ns, delta timestamps", d30, row_group_size=RG1M, **delta_opts(d30), **ZSTD(3))
g30d = sort_per_group(d30, RG1M)
write("stream", "each row group sorted, without dur_ns, delta timestamps", g30d, row_group_size=RG1M, **delta_opts(g30d), **ZSTD(3))
g30 = sort_per_group(t30, RG1M)
write("stream", "each row group sorted, dur_ns kept (no dictionary), delta timestamps", g30, row_group_size=RG1M, **delta_opts(g30, plain=("dur_ns",)), **ZSTD(3))
write("stream", "the same, dur_ns delta-encoded too", g30, row_group_size=RG1M, use_dictionary=[c for c in g30.column_names if c not in DELTA_COLS + ("dur_ns",)],
      column_encoding={c: "DELTA_BINARY_PACKED" for c in DELTA_COLS + ("dur_ns",)}, **ZSTD(3))

print("-- where the bytes go: column, compressed bytes/row")
cols = {}
for label, t, kw in (("dict", s30d, dict(row_group_size=RG256, **ZSTD(3))),
                     ("delta", g30, dict(row_group_size=RG1M, **delta_opts(g30, plain=("dur_ns",)), **ZSTD(3)))):
    path = f"{out}/cols.parquet"
    pq.write_table(t, path, **kw)
    md = pq.ParquetFile(path).metadata
    acc = {}
    for rg in range(md.num_row_groups):
        for ci in range(md.num_columns):
            c = md.row_group(rg).column(ci)
            acc[c.path_in_schema] = acc.get(c.path_in_schema, 0) + c.total_compressed_size
    os.remove(path)
    cols[label] = {k: v / t.num_rows for k, v in acc.items()}
for name in sorted(cols["delta"], key=lambda k: -cols["delta"][k])[:10]:
    print(f"  {name:<22} dictionary {cols['dict'].get(name, 0):5.2f}   delta {cols['delta'][name]:5.2f}")

if a.json:
    with open(a.json, "w") as f:
        json.dump({"sample": os.path.basename(a.sample), "sample_rows": n0, "sample_bytes_per_row": os.path.getsize(a.sample) / n0,
                   "results": results, "columns": cols}, f, ensure_ascii=False, indent=1)
    print("wrote", a.json)
