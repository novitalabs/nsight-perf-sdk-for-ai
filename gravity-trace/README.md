# gravity-trace — NVIDIA Nsight Perf SDK for AI as a GPU trace source (Rust)

`gt` reads what the Nsight Perf SDK for AI (NvTrace) records — libcuda's built-in CUDA API trace, the GPU hardware
trace and PM counters, collected out of process by the `nvidia-nsight-perf` service — and turns it into a flat
per-activity trace (kernels, CUDA graph node kernels, memcpy / memset, driver / runtime API calls): Parquet, Perfetto
JSON, a Flight SQL stream and Prometheus metrics.

```sh
gt export-daemon start          # the service (config.toml): /metrics, captures, every activity persisted
gt export-daemon status | stop | restart
gt export 1h                    # the same in the foreground for 1 h (or until Ctrl-C / stop): `successful` or the error
gt dump parquet now-10m now     # a time range of the persisted trace -> <start>-<end>.parquet (+ .pm.parquet)
gt dump perfetto 10:30 +2s      #   ... or -> <start>-<end>.json.gz for https://ui.perfetto.dev
gt profile -d 10m --flightsql grpc://db:50051   # collect 10 min, streaming every activity to a Flight SQL server
curl -X POST ':9465/capture?duration=5s'   # a capture from the live snapshots -> trace / PM Parquet + Perfetto
gt exporter --listen :9465      # the pipeline in the foreground, configured by flags, nothing persisted
gt capture -d 5s -o out/        # one capture without a running service
gt import <tsf-dir> -o out/     # an existing TSF database (nvtrace_streaming_consumer) -> the same files
gt tables <tsf-dir> -o out/     # every NvTrace table and deferred update -> one Parquet file each (lossless)
gt schema                       # the SDK's table schema
```

Contents: [Architecture](#architecture) · [Service](#service-gt-export-daemon-gt-export-and-gt-dump) ·
[Output](#output) · [Perfetto](#perfetto) · [Processes](#processes) ·
[Streaming](#streaming-gt-profile-and-flight-sql) · [Deployment](#deployment) · [Build](#build) ·
[Layout](#layout) · [Development](#development) · [Benchmarks](#benchmarks-bench)

## Architecture

![gravity-trace architecture: libcuda and the GPU hardware trace feed the nvidia-nsight-perf service; one gt pipeline keeps snapshot data blocks in memory and both serves metrics and writes captures](docs/architecture.svg)

- **Nothing is injected.** libcuda (≥ R615) writes driver API records into shared-memory rings; the
  `nvidia-nsight-perf` service collects them together with the GPU hardware trace, GR-context switches and PM
  counters, and keeps a rolling history. It runs as root in the host PID namespace, one per node.
- **gt runs the SDK's ContinuousSnapshotExporter in process** (through `nvtrace-sys`). Its data blocks, which the SDK
  would write to preallocated files, stay in anonymous memory (build-time patch, `--storage memory`, the default).
  Every 5 s it seals one of two history banks and exports it in one shard per GPU: each shard builds the SDK data
  model, runs the TableExporter, and `mapper.rs` turns the rows into activities.
- **One merge thread** receives every snapshot's activities and always aggregates them per kernel for `/metrics`.
  Under `gt export-daemon` it hands every activity to the persist thread, which writes the rotating trace (Parquet,
  which `gt dump` reads, or Perfetto JSON: `[persist] format`), or, with `[flightsql]` / `gt profile --flightsql`, to
  the Flight SQL sender, which streams them to a server instead of the disk. While a capture is open it also hands the
  activities that start in the window to a writer thread (trace / PM Parquet + Perfetto JSON). The capture closes
  once a snapshot after the window has been merged (GPU times can arrive one snapshot late), so `gt capture -d 5s`
  returns after about 15 s.
- **Exporting and capturing are the same pass**: there is no TSF database on disk. Measured with
  `bench/capture_check.sh` (synthetic CUDA-graph load, GPU cluster): `gt capture -d 5s` and the SDK's
  `nvtrace_streaming_consumer` + `gt import`, running at the same time, gave the same 3,867,452 activities in the
  window, row for row (durations within 1 ns; the absolute times differ by ≤ 29 ns because each consumer converts the
  clocks itself). The consumer wrote 1.7 GB of TSF for its 17 s and the import took 12 s; the live capture wrote
  3.1 MB of Parquet 0.35 s after the window closed. `gt import` / `gt tables` remain for TSF directories that already
  exist.
- **Storage**: every file gt writes and reads back goes through [object_store](https://docs.rs/object_store)
  (`store.rs`): a local directory, or `s3://bucket/prefix` with the `s3` feature ([Build](#build)). The only exception is TSF
  input: the SDK reads those directories itself.

## Service: `gt export-daemon`, `gt export` and `gt dump`

```sh
gt export-daemon start  [-c config.toml]   # validates the config, starts `gt export-daemon run` in a new session,
                                           # returns once it serves (or prints the end of its log if it exits)
gt export-daemon status                    # pid, CPUs, HTTP address, persisted files and the time they cover
gt export-daemon stop                      # SIGTERM; waits until the open files are finished (stop_timeout_s)
gt export-daemon run    [-c config.toml]   # the daemon in the foreground (systemd, containers)
gt export-daemon config [-c config.toml]   # the effective configuration (file + environment)
gt export [DURATION]    [-c config.toml]   # blocking: the daemon in this terminal for DURATION; prints
          [--flightsql URL] [--table T] [--listen ADDR|""]   # `successful`, or fails with the error
```

- **`gt export [DURATION]`** is `gt export-daemon start` in the foreground: the same configuration, HTTP API
  (`daemon.listen`, or `--listen`; `""` = none), persisted trace or Flight SQL stream, CPU limits, and pid file. It
  blocks for DURATION (`30s`, `10m`, `2h`; else `collect.duration_s`; else until Ctrl-C / SIGTERM), then finishes the
  open files / sends what is left. No report: it prints `successful` on stdout (exit status 0), or fails with the
  error on stderr (exit status 1): activities dropped, the Flight SQL server unreachable, a crash, ... Its progress
  lines (config, target, stop) go to stderr. Since it holds the daemon's pid file it never runs next to the daemon,
  `gt export-daemon status` shows it, `gt export-daemon stop` ends it, and `gt dump` asks it to close its open file
  like the daemon. `gt profile` is the variant for measurements: no pid file, HTTP only with `--listen`, a duration is
  required.
- **Background process**: `start` re-runs the binary detached (`setsid`, stdin closed, output appended to
  `log_file`). The pid file is also an `flock`: one daemon per pid file, released by the kernel even after a crash
  (a stale file never blocks a start), and `stop` checks `/proc/<pid>/cmdline` before signalling.

### Configuration

`config.toml`, from `--config` / `$GT_CONFIG`, else `/etc/gravity-trace/config.toml`, else the built-in defaults.
[`config.example.toml`](config.example.toml) documents every key with its default (a unit test keeps the two equal);
unknown keys are errors.

**Environment variables override the file**: `GT_<SECTION>_<KEY>` sets `<key>` of `[<section>]`. Precedence:
built-in defaults < `config.toml` < environment < the command-line flags of `gt export` / `gt profile`
(`--flightsql`, `--table`, `--listen`, DURATION). Values are plain text of the key's type; a variable naming no key
is an error, like a typo in the file. `gt export-daemon config` prints the result and the variables applied, and the
daemon logs them at startup. Every command reads the configuration itself, so `status`, `stop` and `dump` need the
same variables as `start` when they change `pid_file`, `listen` or `persist.dir`.

```sh
GT_PERSIST_DIR=s3://bucket/traces GT_LIMITS_MAX_CPUS=4 gt export-daemon start
GT_FLIGHTSQL_ENABLED=true GT_FLIGHTSQL_ENDPOINT=grpc://db:50051 GT_FLIGHTSQL_PASSWORD=... gt export 2h
GT_COLLECT_IGNORE=gpu_pm,nvtx                       # list: a,b or a TOML array ["gpu_pm", "nvtx"]
GT_FLIGHTSQL_HEADERS="database=traces,tenant=ml"    # table: k=v,k2=v2 or a TOML inline table { database = "traces" }
GT_PERSIST_ENABLED=off                              # bool: true / false, 1 / 0, yes / no, on / off
GT_DAEMON_LISTEN=                                   # "" is a value (no HTTP server); GT_COLLECT_DIAG_FILE= unsets it
```

| key | what |
|---|---|
| `[collect] duration_s` | collection time; after it the daemon finishes its files and exits (0 = until stopped) |
| `[collect] snapshot_ms`, `storage`, `data_rate_mib`, ... | the `gt exporter` flags of the same names |
| `[persist] enabled` | write every activity to `persist.dir` (else only `/metrics` and captures) |
| `[persist] format` | `parquet` (default; `gt dump` reads it) or `perfetto` (`.json.gz` files for ui.perfetto.dev) |
| `[persist] rotate_s`, `max_disk_gib` | new file every N s; oldest files deleted beyond the limit |
| `[persist] max_buffer_mib` | write buffer between the merge thread and the disk writer |
| `[persist] row_group_rows`, `zstd_level` | Parquet row group size (each group is sorted in memory) and compression |
| `[persist] gzip_level` | gzip level of the persisted Perfetto files (default 1: it has to keep up, see [Perfetto](#perfetto)) |
| `[persist] pm` | PM samples too (`.pm.parquet` files, or counter tracks in the Perfetto files) |
| `[flightsql] ...` | the stream: [Streaming](#streaming-gt-profile-and-flight-sql) |
| `[limits] max_cpus`, `cpu_list`, `nice` | CPU affinity of every thread (set before any thread starts) and priority |
| `[daemon] listen`, `pid_file`, `log_file`, `captures_dir` | HTTP address ("" = none), lock / pid, log, capture output |

### Persisted trace and `gt dump`

- **Files**: `<start_ns>-<end_ns>.parquet` (UTC ns of the first start and the last end in the file) and
  `<start_ns>-<end_ns>.pm.parquet`. A file is closed at the first snapshot after `rotate_s`, on
  `POST /persist/flush`, and when the daemon stops. GPU times can arrive one snapshot late, so neighbouring files
  overlap by up to a snapshot interval.
- **The write buffer is bounded**: when the writer falls behind, whole 64K-row chunks are dropped and counted
  (`nvtrace_persist_dropped_rows_total`) instead of slowing the snapshots down. It must hold one snapshot: an
  activity takes 168 bytes, so 1 GiB is 6.4 M activities, i.e. 5 s at 1.3 M/s.
- **`format = "perfetto"`** writes `<start_ns>-<end_ns>.json.gz` instead, one per rotation: the activities, the PM
  counters and the launch -> kernel flows, as they arrive (no sorting, no second pass; the event order follows the
  snapshots). Same rotation, buffer and disk limit. `gt dump` reads Parquet only and says so when a range is covered
  only by `.json.gz`. Sizes, speed and UI limits: [Perfetto](#perfetto).
- **`gt dump parquet|perfetto START END`** selects files by the span in their names, row groups by their `start_ns`
  statistics (the others are not fetched), and rows by `start_ns` in `[START, END)`. It writes one sorted trace file
  `<START>-<END>.parquet` (+ `.pm.parquet`), or Perfetto JSON (`.json.gz`, the PM samples as counter tracks),
  converted from the filtered batches without an intermediate file. `--no-pm` leaves the PM samples out. Times can
  be `now-10m`, epoch s / ms / us / ns, `2026-11-16 10:00` (local), `2026-11-16T10:00:00Z`, `10:00:00+08:00`
  (today), and the end also `+5m`. When the range reaches past the newest file and the daemon is running, the daemon
  is asked to close its open file first; the last snapshot interval is not merged yet.

Measured with `bench/daemon_check.sh` (GPU cluster, service on 2 GPUs, synthetic CUDA-graph load on one GPU at ~770 K
activities/s for 75 s, `max_cpus = 4`, `rotate_s = 20`):

| | |
|---|---|
| start / stop | 0.1 s / 0.5 s (stop includes finishing the open file) |
| CPU | all 12 threads on the last 4 CPUs; 0.75 cores on average; RSS 2.5 GB |
| completeness | 57,273,517 activities persisted, 0 dropped, 0 duplicates; graph-node and eager kernels equal the load's own counts (50,368,800 / 3,357,920) |
| write buffer 512 MiB | 1,703,936 activities dropped (26 chunks), exactly the graph-kernel deficit: a 5 s snapshot (3.9 M activities) did not fit |
| `gt dump parquet now-30s now` (running) | 5.4 s incl. closing the open file: 18.9 M activities, 74 row groups read, 39 skipped |
| `gt dump parquet` of the whole load | 16.4 s, 57.3 M activities -> 46 MB |
| `gt dump perfetto` of 1 s | 742 K activities -> 13 MB `.json.gz` in 0.6 s (gzip level 1 at the time; 19.8 s before the one-pass JSON writer, same events) |

## Output

Every file is named by the UTC time range it covers, in nanoseconds: a capture's or a dump's window, else the span of
its rows (`gt import`, the persisted trace). Files are uploaded under a hidden temporary name (`.gt-<pid>-<n>.*`) and
renamed once complete.

- **trace** (`<start>-<end>.parquet`): `kind, name, start_ns, end_ns, pid, tid, device_id, context_id, stream_id,
  correlation_id, graph_id, graph_node_id, grid_*, block_*, registers_per_thread, static/dynamic_shared_mem, bytes,
  channel_id, device_launched, ns_pid` (duration = `end_ns - start_ns`; `pid` is the host PID, `ns_pid` the PID in
  the process's own namespace: [Processes](#processes)). Timestamps are UTC ns. Row groups of 1M rows, each sorted by
  `pid, device_id, stream_id, start_ns`; `start_ns`, `end_ns`, `correlation_id` delta-encoded, the rest
  dictionary-encoded, zstd 3: about 4.2 bytes per row on DeepSeek-V4 (`bench/pq_compress.py`).
- **pm** (`<start>-<end>.pm.parquet`): `gpu_uuid, end_ns, metric, value` (PM counter samples).
- **Perfetto** (`<start>-<end>.json.gz`): [Perfetto](#perfetto). Captures write one next to their Parquet files
  (`--no-chrome`, or `chrome=false` on `POST /capture`, skips it).
- **Footers** carry `gravity-trace.host`, `gravity-trace.processes` (JSON `[{pid, ns_pid, exe, title}]`, which
  `gt dump perfetto` names the tracks by), `gravity-trace.source` and `gravity-trace.tsf_dir` (import),
  `capture_start_utc_ns` / `capture_end_utc_ns` (live capture: the activities that start in that window) or
  `dump_start_utc_ns` / `dump_end_utc_ns` (dump).
- **Prometheus** (`gt exporter`, `gt export-daemon`): per-process series with the
  process labels `pid, ns_pid, exe, title` (`nvtrace_activity_count_total`, `nvtrace_activity_seconds_total`,
  `nvtrace_gpu_kernel_busy_seconds_total`, p50 / p99, ...), snapshot health
  (`nvtrace_snapshot_stall_seconds_total`, `nvtrace_lost_activities_total`,
  `nvtrace_snapshot_history_truncated_total`, ...), `nvtrace_gpu_metric` (PM counter means); with streaming on
  `nvtrace_flight_*` (rows, dropped rows, DoPuts, errors, retries, bytes, queue bytes, bytes of the last second),
  with persisting on `nvtrace_persist_*` (rows, dropped rows, files, bytes, queue bytes, disk bytes, newest end).
- **Diagnostics**: `--diag FILE` (`collect.diag_file`) appends one JSON line per snapshot.

### Perfetto

Perfetto / Chrome trace JSON, gzip, for https://ui.perfetto.dev: one process per GPU with a track per CUDA stream,
one process per traced host process with a track per thread (API calls), `ac2g` flow arrows from each launch to its
GPU work (correlation id), and a counter process per GPU for the PM samples. Times are relative to the earliest
event (`baseTimeNanoseconds` holds the absolute UTC time), exact to the nanosecond. Kernel names are demangled.

- **Written in one go** (`gt dump perfetto`, captures, `gt import`): read Parquet, convert, gzip level 6, one thread,
  streaming row group by row group. 3 s of vLLM DeepSeek-V4.1-Flash TP 4 (2.44 M activities, 1.24 GB of JSON):
  53 MiB in 7.1 s on the GPU cluster; at gzip level 1 it was 126 MiB in 2.9 s (52 MiB of RSS), same events.
- **Persisted** (`[persist] format = "perfetto"`): compressed while the trace is collected, on the persist thread, at
  `[persist] gzip_level` (default 1). The level has to keep up with the trace: DeepSeek-V4 TP 4 under load makes
  ~400 MB/s of JSON; one CPU core of a node gzips ~660 MB/s at level 1, ~480 at level 2, ~330 at level 3 (half the size of
  level 1). Measured at level 1 with ~650 K activities/s on one GPU (`rotate_s = 10`, `max_cpus = 4`, 45 s): 18 MiB
  per million activities (Parquet: 0.75 MiB), gt CPU 21.3-22.2 s against 18.9-19.1 s with Parquet, every graph-node
  and eager kernel of the load in the files, every flow start matched by its end. vLLM DeepSeek-V4.1-Flash TP 4
  under `vllm bench serve`: 11 M activities and 572 MiB per 15 s file.
- **gzip levels** (flate2's default backend, miniz_oxide; DeepSeek-V4 JSON on one laptop core): level 1 9.3x at
  691 MB/s, level 3 18.9x at 413 MB/s, level 6 22.0x at 226 MB/s; levels 7-9 are no smaller. The zlib-rs backend is
  8% smaller at level 1 but larger than miniz_oxide at levels 3-6, so gt keeps miniz_oxide.
- **The UI** parses JSON in the browser and holds a few million events (~2 GB of memory): a 15 s DeepSeek-V4 file is
  more than it holds. Persist with `rotate_s = 2`, or persist Parquet and `gt dump perfetto` the seconds of interest.
- **Rust or Python?** `bench/parquet2perfetto.py` converts with the same layout in Python, and
  `bench/perfetto_py_vs_rust.sh` checks event by event that the outputs agree (only the demangler's spelling of some
  template names differs). Same 3 s of DeepSeek-V4 on the GPU cluster, all three at gzip level 1 (gt's level before it moved to
  6): gt 2.9 s and 52 MiB of RSS; Python built column-wise with pyarrow.compute 7.3 s (2.5x) and 5.5-6.0 GiB (the
  whole window in memory); Python row by row with `json.dumps` 39.3 s (13x) and 675 MiB.

### Processes

- **Identity.** `pid` is the host PID (the SDK's `process_id_host`, what `nvidia-smi` and the node's `ps` show);
  `ns_pid` is the PID in the process's own namespace (1 for the main process of most containers). Processes of
  different containers stay apart even when both are PID 1 inside.
- **Names.** A process is named after its title when it rewrote its command line (`setproctitle`): vLLM's
  `VLLM::Worker_TP0` ... `VLLM::Worker_TP3`, `VLLM::EngineCore`, else after its executable (`python3.12` for every
  vLLM process). The title is `cmdline` in the host's procfs (`[collect] host_proc` / `--host-proc`, default
  `/proc`, which is the host's when gt runs with hostPID; [deployment](docs/deployment.md)), used only when `NSpid`
  there is (host PID, ..., `ns_pid`). Perfetto names a process `VLLM::Worker_TP0 (pid 1937258)` and its GPU streams
  `VLLM::Worker_TP0 (pid 1937258) stream 13` (`stream 13` in a converted file with one process). A process that takes
  its title after its first snapshot is renamed. `/metrics` has `exe` and `title` labels.
- **Several processes on one GPU** (time-sliced; MPS not tried). The SDK attributes GPU work by the GR-context
  switches of the GPU; three build-time patches fix it for more than one process (`graph-node-per-context`,
  `context-lead-in`, `grctx-carry` under [Build](#build)). Without them the export crashed, and the kernels of each
  snapshot's first milliseconds went to one process. Measured on one GPU with 4 CUDA-graph + eager loads
  time-sliced on it, 2 of them PID 1 in their own namespaces (`bench/multiproc_check.sh`): each host PID's kernel
  count in the trace equals what it launched, graph and eager, in three runs. vLLM, DeepSeek-V4.1-Flash, TP 4
  (`vllm bench serve`, 240 requests): the tracks are `VLLM::EngineCore`, `VLLM::Worker_TP0..3` and the API server,
  and GPU n's kernels are all under `VLLM::Worker_TPn`, about 2.5 M per rank per 15 s.

## Streaming: `gt profile` and Flight SQL

```sh
gt profile -d 10m --flightsql grpc://db:50051 [--table gt_trace] [-c config.toml] [--json]
gt profile -d 2h                 # [flightsql] of config.toml (enabled = true), else persisted like the daemon ([persist])
```

`gt profile` collects for the given time in the foreground, then exits: Ctrl-C / SIGTERM stop it early. Meanwhile
every merged activity (and PM sample) is streamed to an Arrow Flight SQL server, and nothing is written to disk.
`gt export-daemon` and `gt export` do the same with `[flightsql] enabled = true` (or `gt export --flightsql URL`). CPU
limits and collection settings come from `config.toml` (and `GT_*` variables) as for the daemon.

- **Protocol**: Flight SQL bulk ingestion, `CommandStatementIngest` in a DoPut, into `table` (activities) and
  `pm_table` (PM samples). Tables are created when missing (`create`) and appended to. There is one DoPut per
  snapshot, at most `put_rows` rows each. Authentication is the Flight handshake (`username` / `password`, or
  `$GT_FLIGHTSQL_PASSWORD`, like any key), or a bearer `token` (`$GT_FLIGHTSQL_TOKEN`). Extra gRPC `headers`, ingest
  `options`, and `grpc+tls://` with the web PKI roots or `tls_ca_file` are supported.
- **Rows**: the trace Parquet columns with `host` (this host's name) and any constant `columns` in front, so one table
  can hold every node. `dictionaries = true` sends `kind` / `name` / `host` as Arrow dictionaries instead of strings.
  `compression` compresses the Arrow IPC bodies with zstd or lz4. The server has to accept either.
- **Never in the way of the snapshots**: the merge thread hands activities to the sender through the same bounded
  queue as the persisted trace (`max_buffer_mib`). When the server is slower than the trace, whole chunks are
  dropped and counted. A failed DoPut is retried `retries` times on a new connection, then its rows are dropped and
  counted. A retry after a failure the server never reported can ingest rows twice. `max_send_mib_s` caps the rate.
  The cap must stay above the trace's data rate, or the buffer fills and chunks are dropped.
- **Traffic**: gt counts every Arrow IPC message as gRPC takes it and samples the count every second. The
  `gt profile` report gives the mean, p50 / p90 / p99 and max bytes per second (`--json`: every second), and the
  daemon exports `nvtrace_flight_bytes_total` (`rate()` is the traffic) and `nvtrace_flight_last_second_bytes`.
  `gt profile` exits with status 1 when rows were dropped. Its report also shows the activities the service lost
  before gt saw them.

Measured with `bench/flightsql_check.sh` (GPU cluster, the synthetic CUDA-graph load of `daemon_check.sh`, ~770 K
activities/s for 45 s, `max_cpus = 4`; the server is `examples/flightsql_sink.rs` on the same pod, writing every DoPut
to Parquet). Traffic is per second, over the 70 s profile:

| encoding | bytes per row | mean | p50 | p90 | max | gt CPU |
|---|---|---|---|---|---|---|
| strings, uncompressed (default) | 254 | 120 MiB/s | 0 | 539 MiB/s | 565 MiB/s | 0.53 cores |
| `dictionaries = true` | 113 | 55 MiB/s | 0 | 230 MiB/s | 304 MiB/s | 0.51 cores |
| `dictionaries = true`, `compression = "zstd"` | 4.4 | 2.2 MiB/s | 0 | 8.0 MiB/s | 9.3 MiB/s | 0.52 cores |
| strings, `max_send_mib_s = 250` | 254 | 124 MiB/s | 163 MiB/s | 228 MiB/s | 251 MiB/s | 0.64 cores |

- ~34 M activities and 1.5 M PM samples per run, in 51 DoPuts. The stream dropped nothing and duplicated nothing; the
  server's rows are gt's rows. Graph-node and eager kernels equal the load's own counts. Twice, one run came up 2
  or 4 graph-node kernels short; gt had never received them, so they were lost before the stream (service / SDK).
  The bytes gt counts equal the bytes the server received, to the byte. The loopback interface carried 0.5% more
  (TCP/IP headers and ACKs). After the stop, the rest went out in 0.4 s.
- The synthetic load replays one graph, so it compresses unusually well. Real workloads take more than 4.4 bytes per
  row (5.8 on DeepSeek-V4, see below).
- The traffic is bursty: each 5 s snapshot (~3.8 M activities) goes out as fast as the server takes it, then the
  stream is idle until the next one. `max_send_mib_s` spreads the bursts.
- Server side, `TCP_NODELAY` matters. The first version of the test server accepted connections without it. Its
  small HTTP/2 WINDOW_UPDATE frames then waited for gt's delayed ACK while gt waited for the window, and DoPuts
  crawled at 30-40 MiB/s, dropping half the trace (`bench/flightsql_check.sh` reports the server's receive and write
  time per DoPut). gRPC servers set it by default.

### Bytes per row: the stream (Arrow IPC) against Parquet

Is it worth sending compressed Parquet instead of the Arrow IPC stream? Both encodings were measured on the same
activities: 9.1 M from the synthetic load above, and 2.6 M from three DeepSeek-V4 captures (`gt import`). The
"encode" column is CPU per core, for encoding only.

| encoding | synthetic | DeepSeek-V4 | encode |
|---|---|---|---|
| IPC, strings, uncompressed (default) | 245.8 B | 245.1 B | 27-44 M rows/s |
| IPC, dictionaries | 111.1 B | 111.3 B | 115 M rows/s |
| IPC, dictionaries + lz4 | 10.5 B | 12.3 B | 23-26 M rows/s |
| **IPC, dictionaries + zstd** (what the stream sends) | **4.47 B** | **5.82 B** | 14-17 M rows/s |
| IPC, dictionaries + zstd, 64K-row batches in one message (no 2 MiB split) | 4.30 B | 5.54 B | 11-13 M rows/s |
| IPC, dictionaries + zstd, 1M-row batches | 4.18 B | 5.34 B | 10-12 M rows/s |
| IPC, dictionaries + zstd, 1M-row batches sorted by pid, GPU, stream, start | 4.23 B | 5.36 B | 3-5 M rows/s |
| Parquet, unsorted, dictionaries, zstd 3 | 6.63 B | 8.29 B | 4.2-4.4 M rows/s |
| **Parquet, gt trace layout** (sorted 1M-row groups, delta-encoded times, zstd 3) | **0.86 B** | **4.22 B** | 2.5-3.9 M rows/s |
| Parquet, gt trace layout with 64K-row groups | 0.96 B | 4.38 B | 4.2-5.4 M rows/s |

- What makes the trace Parquet small is the delta encoding of the sorted timestamps (DELTA_BINARY_PACKED), not the
  sorting itself. Sorting barely changes IPC + zstd, and unsorted Parquet is larger than IPC + zstd.
- On real workloads the gap is small: 4.22 against 5.82 bytes per row on DeepSeek-V4, i.e. Parquet is 28% smaller.
  Its encoding costs 4-6 times the CPU, and it buffers a row group to sort it. The synthetic load shows 5x only because
  one graph replayed at a steady pace gives near-constant timestamp deltas; real timestamps jitter, and neither
  encoding can remove that information.
- Bigger IPC batches save another ~8% (5.82 -> 5.34), but a message would then be tens of MiB. Many gRPC servers
  accept 4 MiB by default, which is why the encoder splits batches into 2 MiB messages.
- Flight SQL ingestion takes Arrow IPC only. Parquet goes through object storage instead: `persist.dir =
  "s3://bucket/prefix"` (`--features s3`) uploads the rotating trace files as they are written. The latency is then
  `rotate_s` instead of one snapshot. DuckDB, Doris, StarRocks and ClickHouse read Parquet from S3.

Reproduce with `GT_REPLAY_DIR=<dir of trace Parquet> GT_REPLAY_FILES=10 cargo test --release --features monitor
flight::tests::encodings -- --ignored --nocapture`. Two related ignored tests are also in `flight.rs`:
`flight::tests::throughput` (encode / decode / end-to-end time per encoding) and `flight::tests::replay` (real rows
through gRPC into a Parquet-writing server).

## Deployment

[`docs/deployment.md`](docs/deployment.md) covers running the Nsight Perf service, gt and the workloads in Docker and
Kubernetes (manifests included) and what each needs. It was verified with `bench/deploy_check.sh` on a GPU cluster.

- **The service** runs as root in the host PID namespace, with `CAP_SYS_ADMIN` (or `CAP_PERFMON`) and the host's
  libcuda.
- **gt and the workloads** need no root, no capability and no host PID namespace. They only need
  `/run/nvidia/nsight-perf` mounted read-only, and gt also needs the GPUs.
- **The silent failure.** Without the GPU profiling permission every check reports success, but the trace holds only
  the API calls.

## Build

```sh
export NVTRACE_SDK_DIR=/path/to/nsight-perf-sdk-for-ai   # cloned --recursive, configured with CMake (fetches FlatBuffers)
cargo build --release --features monitor                 # -> target/release/gt, every command
cargo build --release                                    # gt dump only: any OS, no SDK
```

`gt` has three cargo features:

| feature | default | what |
|---|---|---|
| `cli` | on | `gt dump`: a time range of a persisted trace (directory or object store) -> Parquet / Perfetto. Plain Rust: builds on Linux and macOS, no SDK. |
| `monitor` | off | everything that collects: `export-daemon`, `export`, `profile`, `exporter`, `capture`, `import`, `tables`, `schema` (adds nvtrace-sys, Arrow Flight / tonic, axum). Runs on Linux (CPU limits, procfs) with the SDK. |
| `s3` | off | `s3://bucket/prefix` for `persist.dir` and `gt dump --dir / --out` (credentials from the `AWS_*` variables). |

`--features monitor` builds both (`cli` stays on); `--no-default-features --features monitor` leaves `gt dump` out.
A collecting host needs `monitor`. A machine that only reads traces (a laptop next to a shared `persist.dir`) needs the default build.
Without `NVTRACE_SDK_DIR` a `monitor` build still compiles, macOS included (for `cargo check` / `cargo test`), and every
collecting command then fails with "gt was built without the Nsight Perf SDK".

Rust 1.88 or newer (edition 2024). PM-metric evaluation loads `libnvperf_grfx_host.so` from the SDK's `build/bin`:
run `gt` with that directory on `LD_LIBRARY_PATH`.

**Crashes** come with their stack (`crates/gravity-trace/src/crash.rs`). Panics unwind (`panic = "unwind"`): the
hook prints the thread, file:line, the message and the stack backtrace from the frame that panicked, always (no
`RUST_BACKTRACE` needed). A panic of any thread stops a running `gt export` / daemon / `gt profile`, which then fail
with it (a pipeline thread that panicked is joined and reported, not ignored). Fatal signals of native code (SIGSEGV,
SIGBUS, SIGILL, SIGFPE, SIGABRT, e.g. in the SDK) print the native stack (the SDK's C++ functions by name) and a
ready-to-run `addr2line -fCipe <gt> 0x...` line for the frames in gt, then the process dies of the signal as before
(core dump). For file:line the release build keeps line tables (`debug = "line-tables-only"`, the C++ shim `-g1`),
zlib-compressed by the linker (`crates/gravity-trace/build.rs`): `gt` is 54 MB instead of 31 MB stripped (154 MB
uncompressed).

`crates/nvtrace-sys/build.rs` copies the SDK client headers into `$OUT_DIR` and patches the copy; a patch whose
anchor is gone (the SDK changed) is skipped with a cargo warning:

| patch | why |
|---|---|
| `grctx-resident-fallback` | with no GR-context event in a snapshot window the SDK gives a GPU's hardware events to the *last* process holding that GPU; under tensor parallelism 3/4 of the GPUs went to the wrong process |
| `log-no-sink`, `gridnullend-log-level` | the SDK formatted ~920k log messages per DeepSeek-V4 snapshot with nothing to write them to (2/3 of the build time, and a glibc lock that serialized the export shards) |
| `memstore-*` | keep the continuous-snapshot data blocks in memory (`shim/gt_memstore.h`) instead of rewriting preallocated files (`--storage memory`, the default) |
| `graph-node-per-context` | CUDA graph node starts and ends were paired through one map per GPU keyed by the hardware correlation id, which every process counts from the same start: with several processes on a GPU an end found another process's start (SIGSEGV in the export, or a wrong end time). Keyed by (process, correlation id) |
| `context-lead-in` | a context's first hardware events are stamped up to ~2 µs before its GR-context load; they fell into the switch gap and were dropped (~0.1% of the kernels with 4 processes on a GPU). Up to 5 µs before a load, inside the gap, they belong to the context loaded |
| `grctx-carry` | the GR-context records of a window's first 10-110 ms are in the previous snapshot, so the SDK made the first context in the window resident from its start: those milliseconds went to one process. The contexts of the previous window (at most 1 s old) are carried over |

## Layout

| path | what |
|---|---|
| `crates/nvtrace-sys` | C++ shim over the SDK's `TableExporter` / `ContinuousSnapshotExporter` (built against `$NVTRACE_SDK_DIR`), Rust `Sink` API; patches the SDK client headers at build time |
| `crates/gravity-trace-core` | activity model, log-linear histogram, kernel aggregator (exact busy-time interval union), trace Parquet writer (sorted, delta-encoded row groups), Perfetto converter, metric structs |
| `crates/gravity-trace` | the `gt` binary (features `cli` / `monitor`, see [Build](#build)): NvTrace rows -> activities (`mapper.rs`), the live pipeline (`live.rs`), captures, `export-daemon` (`daemon.rs`, `config.rs`: file + `GT_*` environment), `export` / `profile` / `export-daemon run` in the foreground (`export.rs`), the persisted trace (`persist.rs`; its files and the pid file, shared with `dump`: `files.rs`, `pidfile.rs`), the Flight SQL stream (`flight.rs`), `dump`, object store IO (`store.rs`), `import`, `tables`, Prometheus rendering, process titles (`proctitle.rs`); `examples/flightsql_sink.rs`: a Flight SQL ingest server for the bench |
| `config.example.toml` | every `gt export-daemon` setting with its default |
| `bench` | Nsight Perf benchmarks and analysis (see [Benchmarks](#benchmarks-bench)) |
| `docs/architecture.svg` | the diagram above (light and dark colors) |
| `docs/deployment.md` | deployment in Docker / Kubernetes: permissions of the service, gt and the workloads, manifests, checks |

Coding rules: no `unwrap()` (`clippy::unwrap_used = "deny"`). Errors use snafu, and invariants use `expect("why")`.
Concurrency uses tokio. SDK callbacks into Rust catch panics (`panic = "unwind"`). The code is hand-wrapped at about
120 columns and not rustfmt-formatted.

## Development

```sh
cargo clippy --workspace --all-targets --features monitor && cargo test --workspace --features monitor   # NVTRACE_SDK_DIR set
```

The default build (`gt dump` only) also builds and tests on macOS: `cargo clippy --all-targets && cargo test`.
`cargo test --features monitor` runs there as well, without the SDK.

Environment variables for experiments:

| variable | effect |
|---|---|
| `GT_SAVE_DIR`, `GT_SAVE_MIN_BLOCKS` | copy the data blocks of the first snapshot with at least N blocks (default 40) to the directory, for `bench/replay.sh` |
| `GT_SHARDS_SERIAL=1` | run the export shards one after another (measures their work without contention) |
| `GT_LOG_VOLUME=<n>` | raise the SDK's log volume (with `NVTRACE_LOG_ENABLE_STDERR=1`) |

## Benchmarks (`bench`)

Most checks start their own `nvidia-nsight-perf` (from `$NVTRACE_SDK_DIR/build/bin`, or `NPS_BIN`) on idle GPUs and
compare gt against loads with known counts (`bench/graphload.py`, torch). Common variables: `NVTRACE_SDK_DIR`,
`CARGO_TARGET_DIR` (gt from `$CARGO_TARGET_DIR/release`, default `target/`), `PY` (a Python with torch, default
`python3`), `OUT` (default under `/tmp/gt-bench`), `HOSTDRV` (a directory of symlinks to the host driver's libcuda,
put on the service's `LD_LIBRARY_PATH`; see [`docs/deployment.md`](docs/deployment.md)).

```sh
bash bench/order.sh          # are processes started before / after the service recorded?
NPROC=4 bash bench/livediag.sh   # live completeness against a load with known kernel counts
bash bench/capture_check.sh      # gt capture / POST /capture vs streaming consumer + gt import on the same window
bash bench/multiproc_check.sh    # 4 processes time-sliced on GPU 0 (2 in PID namespaces of their own): kernels per host pid exact
bash bench/daemon_check.sh       # gt export-daemon + gt dump against a load with known kernel counts (BUF_MIB, CPUS)
bash bench/flightsql_check.sh    # gt profile --flightsql: rows vs the load's counts, network traffic per second (VARIANTS)
bash bench/deploy_check.sh       # service / gt / loads as separate containers: users, capabilities, mounts (docs/deployment.md)
bash bench/perfetto_py_vs_rust.sh TRACE.parquet   # Parquet -> Perfetto: gt vs Python (pyarrow / row by row), same events
GT_REPLAY_DIR=... cargo test --release --features monitor flight::tests::encodings -- --ignored --nocapture   # IPC vs Parquet bytes per row
```
