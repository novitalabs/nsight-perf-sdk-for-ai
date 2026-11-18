//! `gt` errors (snafu).
use snafu::Snafu;
use std::path::PathBuf;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("{source}"))]
    Core { source: gravity_trace_core::error::Error },

    #[snafu(display("background task failed"))]
    Join { source: tokio::task::JoinError },

    #[snafu(display("cannot listen on {addr}"))]
    Listen { addr: String, source: std::io::Error },

    #[snafu(display("arrow error while {context}"))]
    Arrow { context: String, source: arrow_schema::ArrowError },

    #[cfg(feature = "monitor")]
    #[snafu(display("{source}"))]
    Nvtrace { source: nvtrace_sys::Error },

    #[snafu(display("the Nsight Perf SDK schema has no {table}{}{column}", if column.is_empty() { "" } else { "." }))]
    NvtraceSchema { table: String, column: String },

    #[snafu(display("cannot open {}", path.display()))]
    OpenFile { path: PathBuf, source: std::io::Error },

    #[snafu(display("invalid --ignore list `{input}` (gpu_pm,gpu_hwt,cuda_api,nvtx)"))]
    BadIgnore { input: String },

    #[snafu(display("cannot start thread {name}"))]
    SpawnThread { name: String, source: std::io::Error },

    #[snafu(display("a capture is already running"))]
    CaptureBusy,

    #[snafu(display("the capture was aborted: the live pipeline stopped"))]
    CaptureAborted,

    #[snafu(display("invalid duration `{input}` (examples: 5s, 500ms, 1m, 2h)"))]
    BadDuration { input: String },

    #[snafu(display("cannot create directory {}", path.display()))]
    CreateDir { path: PathBuf, source: std::io::Error },

    #[snafu(display("{context}"))]
    Io { context: String, source: std::io::Error },

    #[snafu(display("cannot read the configuration {}", path.display()))]
    ReadConfig { path: PathBuf, source: std::io::Error },

    #[snafu(display("{}: {message}", path.display()))]
    ParseConfig { path: PathBuf, message: String },

    #[snafu(display("{}: {message}", path.display()))]
    BadConfig { path: PathBuf, message: String },

    #[snafu(display("environment variable {var}: {message}"))]
    BadEnv { var: String, message: String },

    #[snafu(display("cannot start the async runtime"))]
    Runtime { source: std::io::Error },

    #[snafu(display("object store: {what}"))]
    Store { what: String, source: object_store::Error },

    #[snafu(display("cannot read {what}"))]
    ReadParquet { what: String, source: parquet::errors::ParquetError },

    #[snafu(display("invalid store location `{url}`: {message}"))]
    StoreUrl { url: String, message: String },

    #[snafu(display("persisting is off (persist.enabled = false)"))]
    PersistOff,

    #[snafu(display("the persist writer stopped"))]
    PersistStopped,

    #[snafu(display("gt export-daemon or gt export is already running (pid {pid} holds {})", pid_file.display()))]
    AlreadyRunning { pid: String, pid_file: PathBuf },

    #[snafu(display("gt export-daemon is not running ({})", pid_file.display()))]
    NotRunning { pid_file: PathBuf },

    #[snafu(display("pid file {}", path.display()))]
    PidFile { path: PathBuf, source: std::io::Error },

    #[snafu(display("cannot start the daemon process"))]
    SpawnDaemon { source: std::io::Error },

    #[snafu(display("the daemon exited during startup ({status}); last lines of {}:\n{tail}", log.display()))]
    DaemonExited { status: String, log: PathBuf, tail: String },

    #[snafu(display("the daemon (pid {pid}) is not ready after {secs} s; see {}", log.display()))]
    DaemonNotReady { pid: u32, secs: u64, log: PathBuf },

    #[snafu(display("pid {pid} is not gt export-daemon / gt export (cmdline `{cmdline}`); remove {} if it is stale", pid_file.display()))]
    NotOurProcess { pid: u32, cmdline: String, pid_file: PathBuf },

    #[snafu(display("cannot signal pid {pid}"))]
    Kill { pid: u32, source: std::io::Error },

    #[snafu(display("gt (pid {pid}) still running after {secs} s (`kill -KILL {pid}` ends it without closing its files)"))]
    StopTimeout { pid: u32, secs: u64 },

    #[snafu(display("cannot set the CPU affinity"))]
    Affinity { source: std::io::Error },

    #[snafu(display("limits: none of the CPUs {wanted} is available to this process (allowed: {allowed})"))]
    NoCpus { wanted: String, allowed: String },

    #[snafu(display("invalid time `{input}` (examples: now-10m, 1763287200, 1763287200123, \"2026-11-16 10:00\", 2026-11-16T10:00:00Z, +5m)"))]
    BadTime { input: String },

    #[snafu(display("the range is empty: {start} is not before {end}"))]
    EmptyRange { start: String, end: String },

    #[snafu(display("no trace between {start} and {end} in {} ({have})", dir.display()))]
    NoData { dir: PathBuf, start: String, end: String, have: String },

    #[snafu(display(
        "{} holds Perfetto files (persist.format = \"perfetto\"), not the Parquet trace gt dump reads: open the \
         <start>-<end>.json.gz files of the range in https://ui.perfetto.dev",
        dir.display()
    ))]
    PerfettoFiles { dir: PathBuf },

    #[snafu(display("flightsql endpoint `{endpoint}`: {message}"))]
    FlightEndpoint { endpoint: String, message: String },

    #[cfg(feature = "monitor")]
    #[snafu(display("cannot connect to the Flight SQL server {endpoint}"))]
    FlightConnect { endpoint: String, source: tonic::transport::Error },

    #[cfg(feature = "monitor")]
    #[snafu(display("Flight SQL handshake with {endpoint} failed (flightsql.username / password)"))]
    FlightAuth { endpoint: String, source: arrow_flight::error::FlightError },

    #[snafu(display("gt profile needs a duration: --duration 10m, or collect.duration_s in the configuration"))]
    NoDuration,

    #[snafu(display("{what}: {dropped} activities were not delivered"))]
    Incomplete { what: String, dropped: u64 },

    #[snafu(display("{what} (stack backtrace above)"))]
    Panicked { what: String },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// `error: cause: cause ...` on one line (for log files).
pub fn chain(e: &dyn std::error::Error) -> String {
    let mut s = e.to_string();
    let mut cur = e.source();
    while let Some(c) = cur {
        s.push_str(": ");
        s.push_str(&c.to_string());
        cur = c.source();
    }
    s
}
