//! `gt` — gravity-trace: the NVIDIA Nsight Perf SDK for AI (NvTrace) as a GPU trace source.
//!
//! ```text
//! gt export-daemon start|stop|status   # the service: config.toml, background process, /metrics, persisted trace
//! gt export 1h                         # the same in the foreground, blocking: `successful` or the error
//! gt dump parquet|perfetto START END   # a time range of the persisted trace -> <start>-<end>.parquet / .json.gz
//! gt profile -d 10m --flightsql grpc://host:port   # collect 10 min, streaming every activity to a Flight SQL server
//! gt exporter --listen :9465           # the same pipeline in the foreground, configured by flags, nothing persisted
//! gt capture -d 5s -o out/             # one capture from live snapshots, no exporter needed
//! gt import <tsf-dir> -o out/          # an existing TSF database -> the same files
//! gt tables <tsf-dir> -o out/          # every NvTrace table -> one Parquet file each (lossless)
//! gt schema                            # the SDK's table schema
//! ```
//!
//! `export-daemon`, `exporter` and `capture` talk to the running `nvidia-nsight-perf` service themselves: snapshot
//! data blocks stay in memory and files are written straight from them. `import` / `tables` read TSF directories
//! written by the SDK's `nvtrace_streaming_consumer --output-dir ...`. Every trace file is named by the UTC time range
//! it covers: `<start_ns>-<end_ns>.parquet`.
//!
//! Two cargo features: `cli` (default) is `gt dump`, which reads a persisted trace on any OS; `monitor` is everything
//! that collects (`export-daemon`, `export`, `profile`, `exporter`, `capture`, `import`, `tables`, `schema`): Linux,
//! built against the Nsight Perf SDK. `cargo build --release --features monitor` builds both.
// With one feature only, part of the shared code (configuration keys, error variants, time parsing) has no user;
// dead code is checked in the build with both.
#![cfg_attr(not(all(feature = "cli", feature = "monitor")), allow(dead_code))]

#[cfg(not(any(feature = "cli", feature = "monitor")))]
compile_error!("gt needs the `cli` feature (gt dump, the default) and/or `monitor` (collecting)");

mod config;
mod crash;
#[cfg(feature = "cli")]
mod dump;
mod error;
mod files;
mod pidfile;
mod store;
mod timefmt;
mod writer;
#[cfg(feature = "monitor")]
mod capture;
#[cfg(feature = "monitor")]
mod daemon;
#[cfg(feature = "monitor")]
mod export;
#[cfg(feature = "monitor")]
mod feed;
#[cfg(feature = "monitor")]
mod flight;
#[cfg(all(test, feature = "monitor"))]
mod flight_sink;
#[cfg(feature = "monitor")]
mod import;
#[cfg(feature = "monitor")]
mod live;
#[cfg(feature = "monitor")]
mod mapper;
#[cfg(feature = "monitor")]
mod persist;
#[cfg(feature = "monitor")]
mod proctitle;
#[cfg(feature = "monitor")]
mod prom;
#[cfg(feature = "monitor")]
mod tables;

use crate::error::{BadDurationSnafu, Result, RuntimeSnafu};
#[cfg(feature = "monitor")]
use crate::error::{BadIgnoreSnafu, JoinSnafu};
#[cfg(feature = "monitor")]
use clap::Args;
use clap::{Parser, Subcommand};
use snafu::ResultExt;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "gt", version, about = "gravity-trace: Nsight Perf SDK for AI captures and live snapshots -> Parquet, Perfetto, Prometheus")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[cfg(feature = "monitor")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Storage {
    Memory,
    Disk,
}

#[derive(Subcommand)]
enum Cmd {
    /// The live pipeline as a background service, configured by config.toml: Prometheus /metrics, captures, and
    /// every activity persisted to <start>-<end>.parquet files (read back by `gt dump`).
    #[cfg(feature = "monitor")]
    #[command(name = "export-daemon")]
    ExportDaemon {
        /// Configuration file [default: /etc/gravity-trace/config.toml if it exists, else built-in defaults; see
        /// config.example.toml].
        #[arg(short, long, env = "GT_CONFIG", global = true)]
        config: Option<PathBuf>,
        #[command(subcommand)]
        action: daemon::Action,
    },
    /// The export daemon in the foreground, blocking for DURATION (else collect.duration_s; without either until
    /// Ctrl-C / SIGTERM). The same config.toml, HTTP API (daemon.listen), persisted trace or Flight SQL stream, CPU
    /// limits and pid file as `gt export-daemon start`: it does not run next to the daemon, and `gt dump` and
    /// `gt export-daemon status|stop` work on it. Prints `successful` at the end, else fails with the error (exit
    /// status 1): activities dropped, a crash (with its stack backtrace), ...
    #[cfg(feature = "monitor")]
    Export {
        /// How long, e.g. 30s, 10m, 2h [default: collect.duration_s; 0 = until stopped].
        #[arg(value_parser = parse_duration)]
        duration: Option<Duration>,
        /// Stream to this Arrow Flight SQL server instead of persisting: grpc://host:port or grpc+tls://host:port
        /// (turns [flightsql] on).
        #[arg(long)]
        flightsql: Option<String>,
        /// Target table of the activities [default: flightsql.table].
        #[arg(long)]
        table: Option<String>,
        /// HTTP address of /metrics, /capture, /persist/flush [default: daemon.listen]; "" = no HTTP server.
        #[arg(long)]
        listen: Option<String>,
        /// Configuration file [default: /etc/gravity-trace/config.toml if it exists, else built-in defaults].
        #[arg(short, long, env = "GT_CONFIG")]
        config: Option<PathBuf>,
    },
    /// Every persisted activity that starts in [START, END) -> one trace Parquet file (+ PM samples) or one
    /// gzip-compressed Perfetto JSON file, named <start>-<end>.parquet / .json.gz (UTC ns).
    #[cfg(feature = "cli")]
    Dump {
        format: dump::Format,
        /// now-10m | epoch s / ms / us / ns (1763287200, 1763287200123, ...) | "2026-11-16 10:00" (local time) |
        /// 2026-11-16T10:00:00Z | 10:00:00+08:00 (today).
        #[arg(allow_hyphen_values = true)]
        start: String,
        /// Same forms (exclusive), or +5m: the start plus a duration.
        #[arg(allow_hyphen_values = true)]
        end: String,
        /// Output directory or object store URL, or the output file in it (`.parquet`, `.json.gz`).
        #[arg(short, long, default_value = ".")]
        out: String,
        /// Persisted trace directory or object store URL [default: persist.dir of the configuration].
        #[arg(long)]
        dir: Option<String>,
        /// Configuration file (for persist.dir and the daemon's address).
        #[arg(short, long, env = "GT_CONFIG")]
        config: Option<PathBuf>,
        /// Leave the PM samples out (the `.pm.parquet` of a Parquet dump, the counter tracks of a Perfetto one).
        #[arg(long)]
        no_pm: bool,
        /// Do not ask a running daemon to close its open file first.
        #[arg(long)]
        no_flush: bool,
        /// Print the dump report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Collect for DURATION, then exit. Meanwhile every activity is streamed to an Arrow Flight SQL server
    /// (--flightsql, or [flightsql] of the configuration; nothing is written to disk then), else persisted like
    /// `gt export-daemon` does ([persist]). Collection settings and CPU limits come from config.toml. Ends with a
    /// report: activities, rows the server took, network traffic per second.
    #[cfg(feature = "monitor")]
    Profile {
        /// How long, e.g. 30s, 10m, 2h [default: collect.duration_s of the configuration].
        #[arg(short, long, value_parser = parse_duration)]
        duration: Option<Duration>,
        /// Arrow Flight SQL server to stream to: grpc://host:port or grpc+tls://host:port (turns [flightsql] on).
        #[arg(long)]
        flightsql: Option<String>,
        /// Target table of the activities [default: flightsql.table].
        #[arg(long)]
        table: Option<String>,
        /// Also serve GET /metrics and POST /capture on this address while profiling.
        #[arg(long)]
        listen: Option<String>,
        /// Configuration file [default: /etc/gravity-trace/config.toml if it exists, else built-in defaults].
        #[arg(short, long, env = "GT_CONFIG")]
        config: Option<PathBuf>,
        /// Print the report as JSON (with the bytes sent in every second).
        #[arg(long)]
        json: bool,
    },
    /// Decode a TSF database (nvtrace_streaming_consumer output) into trace Parquet (+ PM metrics, Perfetto JSON).
    #[cfg(feature = "monitor")]
    Import {
        tsf_dir: PathBuf,
        /// Output directory or object store URL.
        #[arg(short, long, default_value = "/tmp/gravity-trace/import")]
        out_dir: String,
        /// Skip the Perfetto/Chrome JSON conversion.
        #[arg(long)]
        no_chrome: bool,
        /// Providers to skip while decoding: gpu_pm,gpu_hwt,cuda_api,nvtx.
        #[arg(long, default_value = "")]
        ignore: String,
        /// Observation window (UTC ms); 0 = whole database.
        #[arg(long, default_value_t = 0)]
        start_ms: u64,
        #[arg(long, default_value_t = 0)]
        end_ms: u64,
        /// Print the import report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Dump every NvTrace table and deferred update to its own Parquet file (lossless).
    #[cfg(feature = "monitor")]
    Tables {
        tsf_dir: PathBuf,
        /// Output directory or object store URL.
        #[arg(short, long, default_value = "/tmp/gravity-trace/tables")]
        out_dir: String,
        #[arg(long, default_value = "")]
        ignore: String,
    },
    /// Live: continuous snapshots of the nvidia-nsight-perf service -> Prometheus /metrics; `POST /capture` records
    /// a window of them into Parquet.
    #[cfg(feature = "monitor")]
    Exporter {
        #[arg(long, default_value = "0.0.0.0:9465")]
        listen: String,
        /// Kernels per process and kind exported individually (rest folded into `<other>`).
        #[arg(long, default_value_t = 50)]
        top: usize,
        /// Append one JSON line of diagnostics per snapshot (manifest, export queue / build / decode times, rows,
        /// updates, orphans, losses, CUDA graph completeness) to FILE.
        #[arg(long)]
        diag: Option<PathBuf>,
        /// Where `POST /capture` writes trace / PM Parquet and Perfetto JSON (a directory or an object store URL).
        #[arg(long, default_value = "/tmp/gravity-trace/captures")]
        captures_dir: String,
        #[command(flatten)]
        live: LiveArgs,
    },
    /// Record every activity of the next DURATION from live snapshots (in memory, no TSF on disk) into trace / PM
    /// Parquet (+ Perfetto JSON). Returns once a snapshot after the window arrived (up to two snapshot intervals later).
    #[cfg(feature = "monitor")]
    Capture {
        /// How long to record, e.g. 500ms, 5s, 1m.
        #[arg(short, long, default_value = "5s", value_parser = parse_duration)]
        duration: Duration,
        /// Output directory or object store URL.
        #[arg(short, long, default_value = "/tmp/gravity-trace/captures")]
        out_dir: String,
        /// Skip the Perfetto/Chrome JSON conversion.
        #[arg(long)]
        no_chrome: bool,
        /// Print the capture report as JSON.
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        live: LiveArgs,
    },
    #[cfg(feature = "monitor")]
    /// Print the SDK's table schema.
    Schema,
}

#[cfg(feature = "monitor")]
/// How the SDK takes snapshots (shared by `exporter` and `capture`).
#[derive(Args)]
struct LiveArgs {
    /// Snapshot cadence (the SDK recommends >= 5000 ms).
    #[arg(long, default_value_t = 5000)]
    snapshot_ms: u64,
    /// Scratch directory for snapshot data blocks (--storage disk); with --storage memory only an empty session
    /// directory is created there.
    #[arg(long, default_value = "/tmp/gravity-trace/live")]
    work_dir: PathBuf,
    /// Where the SDK keeps the snapshot data blocks between consuming and exporting them. `memory`: anonymous
    /// memory, released after each export (needs the nvtrace-sys patch of the SDK headers, else falls back to disk).
    /// `disk`: the SDK's own preallocated files in --work-dir, rewritten continuously (DeepSeek-V4 ~32 MB/s).
    #[arg(long, value_enum, default_value_t = Storage::Memory)]
    storage: Storage,
    /// Provisioned trace data rate (MiB/s) used to size the snapshot storage: two history banks of
    /// rate x snapshot interval in 10 MiB data blocks (~10.3 MiB per MiB/s at 5 s, plus --max-attach-mib),
    /// preallocated in --work-dir with --storage disk, reserved (resident only while holding data) in memory. A bank that fills early only cuts the snapshot short
    /// (nvtrace_snapshot_history_truncated_total). Data is lost when the previous snapshot is still being exported
    /// by then: the SDK stops reading the service until it gets that bank back
    /// (nvtrace_snapshot_stall_seconds_total). DeepSeek-V4 TP=4 decode c1 (~270 MiB/s of trace, ~1.3 M graph node
    /// launches/s) fills a 128 MiB/s bank in 2.6 s, and its export (one shard per GPU) is done well before the next
    /// cutoff (smallest margin 3.5 s); at 64 the bank fills in 1.3 s and the margin shrinks to ~0.2 s.
    #[arg(long, default_value_t = 128)]
    data_rate_mib: u64,
    /// Upper bound of the per-snapshot attachment data (module / function / CUDA graph definitions of every
    /// traced process). Too small and snapshots lose definitions: 4 DeepSeek-V4 TP ranks need more than 100 MiB.
    #[arg(long, default_value_t = 1024)]
    max_attach_mib: u64,
    /// Providers to skip: gpu_pm,gpu_hwt,cuda_api,nvtx.
    #[arg(long, default_value = "")]
    ignore: String,
    /// Export every snapshot in this many parallel shards, each owning whole GPUs (data model build, table
    /// export and mapping run per shard; the SDK otherwise does all of it on one thread). 0 = one per GPU
    /// (at most 8); 1 = the SDK's single-threaded export.
    #[arg(long, default_value_t = 0)]
    shards: u32,
    /// The procfs of the host's PID namespace, for process titles (vLLM's VLLM::Worker_TP0): /proc in the host's
    /// namespace, else the host's /proc mounted read-only.
    #[arg(long, default_value = "/proc")]
    host_proc: PathBuf,
}

#[cfg(feature = "monitor")]
impl LiveArgs {
    fn options(&self) -> Result<nvtrace_sys::LiveOptions> {
        Ok(nvtrace_sys::LiveOptions {
            work_dir: self.work_dir.clone(),
            snapshot_ms: self.snapshot_ms,
            data_rate_mib_per_s: self.data_rate_mib,
            max_attach_mib: self.max_attach_mib,
            shards: self.shards,
            in_memory: self.storage == Storage::Memory,
            ignore: parse_ignore(&self.ignore)?,
            ..Default::default()
        })
    }
}

/// `500ms`, `5s`, `1m`, `2h`, `1d` (a bare number is seconds).
pub(crate) fn parse_duration(s: &str) -> Result<Duration> {
    let t = s.trim();
    let (num, unit) = match t.find(|c: char| c.is_ascii_alphabetic()) {
        Some(i) => (&t[..i], &t[i..]),
        None => (t, "s"),
    };
    let v: f64 = num
        .trim()
        .parse()
        .ok()
        .filter(|v: &f64| v.is_finite() && *v > 0.0)
        .ok_or_else(|| BadDurationSnafu { input: s.to_string() }.build())?;
    let secs = match unit {
        "ms" => v / 1e3,
        "s" => v,
        "m" | "min" => v * 60.0,
        "h" => v * 3600.0,
        "d" => v * 86_400.0,
        _ => return BadDurationSnafu { input: s.to_string() }.fail(),
    };
    Ok(Duration::from_secs_f64(secs))
}

#[cfg(feature = "monitor")]
fn parse_ignore(s: &str) -> Result<nvtrace_sys::Ignore> {
    nvtrace_sys::Ignore::parse(s).ok_or_else(|| BadIgnoreSnafu { input: s.to_string() }.build())
}

/// This host's name (process and file labels).
pub(crate) fn host() -> String {
    hostname::get().map(|h| h.to_string_lossy().into_owned()).unwrap_or_default()
}

/// A multi-threaded async runtime with `workers` worker threads.
pub(crate) fn runtime(workers: usize) -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread().worker_threads(workers).enable_all().build().context(RuntimeSnafu)
}

#[snafu::report]
fn main() -> Result<()> {
    crash::install();
    match Cli::parse().cmd {
        // Before any runtime exists: `run` limits the CPUs every later thread may use.
        #[cfg(feature = "monitor")]
        Cmd::ExportDaemon { config, action } => daemon::main(action, config.as_deref()),
        #[cfg(feature = "monitor")]
        Cmd::Export { duration, flightsql, table, listen, config } => {
            export::main(export::Mode::Export, export::Overrides { duration, flightsql, table, listen }, config.as_deref(), false)
        }
        #[cfg(feature = "monitor")]
        Cmd::Profile { duration, flightsql, table, listen, config, json } => {
            export::main(export::Mode::Profile, export::Overrides { duration, flightsql, table, listen }, config.as_deref(), json)
        }
        cmd => runtime(4)?.block_on(run(cmd)),
    }
}

async fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        #[cfg(feature = "monitor")]
        Cmd::ExportDaemon { .. } | Cmd::Export { .. } | Cmd::Profile { .. } => Ok(()), // handled in main
        #[cfg(feature = "cli")]
        Cmd::Dump { format, start, end, out, dir, config, no_pm, no_flush, json } => {
            let a = dump::DumpArgs { format, start, end, dir, out, no_pm, no_flush, json };
            dump::dump_cmd(a, config.as_deref()).await
        }
        #[cfg(feature = "monitor")]
        Cmd::Import { tsf_dir, out_dir, no_chrome, ignore, start_ms, end_ms, json } => {
            let o = import::ImportOptions {
                start_utc_ns: start_ms * 1_000_000,
                end_utc_ns: end_ms * 1_000_000,
                ignore: parse_ignore(&ignore)?,
                chrome: !no_chrome,
            };
            import::import_cmd(tsf_dir, out_dir, o, json).await
        }
        #[cfg(feature = "monitor")]
        Cmd::Tables { tsf_dir, out_dir, ignore } => {
            let ig = parse_ignore(&ignore)?;
            let files = tokio::task::spawn_blocking(move || tables::dump_tables(&tsf_dir, &out_dir, ig)).await.context(JoinSnafu)??;
            for (f, n) in files {
                println!("{n:>12} rows  {f}");
            }
            Ok(())
        }
        #[cfg(feature = "monitor")]
        Cmd::Exporter { listen, top, diag, captures_dir, live } => {
            let pipeline = live::PipelineConfig { top, live: live.options()?, diag, host_proc: live.host_proc.clone(), persist: None, flight: None };
            live::exporter(live::ExporterConfig { listen, pipeline, captures_dir }).await
        }
        #[cfg(feature = "monitor")]
        Cmd::Capture { duration, out_dir, no_chrome, json, live } => {
            let cfg = live::PipelineConfig { top: 50, live: live.options()?, diag: None, host_proc: live.host_proc.clone(), persist: None, flight: None };
            live::capture_cmd(cfg, duration, out_dir, !no_chrome, json).await
        }
        #[cfg(feature = "monitor")]
        Cmd::Schema => tables::print_schema(),
    }
}
