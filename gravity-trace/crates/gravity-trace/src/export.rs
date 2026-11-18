//! The live pipeline in the foreground until a duration, Ctrl-C or SIGTERM, then a report of what was collected and
//! written / sent. Three commands run it:
//!
//! - `gt export [DURATION]`: the export daemon in the foreground, blocking. The same configuration, HTTP API
//!   (`daemon.listen`), persisted trace or Flight SQL stream, CPU limits and pid file as `gt export-daemon start`: it
//!   never runs next to the daemon, and `gt dump` / `gt export-daemon status|stop` find it. Runs for DURATION, else
//!   `collect.duration_s`, else until stopped. No report: it prints `successful`, or fails with the error (exit
//!   status 1) when activities were dropped or a thread panicked.
//! - `gt profile -d DURATION`: a measurement: no pid file, HTTP only with `--listen`, a duration is required; the
//!   report goes to stdout (`--json`).
//! - `gt export-daemon run` (what `start` launches): timestamped lines and the report go to the daemon's log, dropped
//!   activities are only logged.
//!
//! Every activity is streamed to an Arrow Flight SQL server (`--flightsql URL` or `[flightsql]`; nothing is written to
//! disk then), else persisted (`[persist]`). On the stop the snapshots end and what is left is sent / written. A panic
//! of any thread stops it too ([`crate::crash`]), and the command fails.
use crate::config::{self, Config, Loaded};
use crate::daemon::{self, PidLock};
use crate::error::{BadConfigSnafu, IncompleteSnafu, ListenSnafu, NoDurationSnafu, PanickedSnafu, Result};
use crate::flight::{FlightOptions, FlightReport};
use crate::live;
use crate::persist::PersistOptions;
use crate::timefmt::{fmt_span, fmt_utc, now_ns};
use snafu::ResultExt;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `gt export`.
    Export,
    /// `gt profile`.
    Profile,
    /// `gt export-daemon run`.
    Daemon,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Export => "gt export",
            Mode::Profile => "gt profile",
            Mode::Daemon => "gt export-daemon",
        }
    }
}

/// Command-line flags, over the configuration (and the environment in it).
#[derive(Default)]
pub struct Overrides {
    pub duration: Option<Duration>,
    pub flightsql: Option<String>,
    pub table: Option<String>,
    /// "" = no HTTP server.
    pub listen: Option<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct PersistReport {
    pub dir: String,
    /// "parquet" or "perfetto".
    pub format: &'static str,
    pub rows: u64,
    pub files: u64,
    pub bytes: u64,
    pub dropped_rows: u64,
}

#[derive(Debug, serde::Serialize)]
pub struct Report {
    pub start_utc_ns: i64,
    pub end_utc_ns: i64,
    pub seconds: f64,
    pub stopped_by: String,
    /// CPUs the process was limited to.
    pub cpus: Option<String>,
    /// The pipeline: activities merged (and handed to the stream / the persisted trace), and lost on the way from
    /// the service.
    #[serde(flatten)]
    pub totals: live::Totals,
    pub flightsql: Option<FlightReport>,
    pub persist: Option<PersistReport>,
    /// Seconds from the stop until everything was sent / written.
    pub drain_seconds: f64,
}

/// A timestamped line on stderr (the daemon's log).
fn say(mode: Mode, msg: impl std::fmt::Display) {
    eprintln!("{} {}: {msg}", fmt_utc(now_ns()), mode.name());
}

pub fn main(mode: Mode, o: Overrides, config_path: Option<&Path>, json: bool) -> Result<()> {
    run(mode, config::load(config_path)?, o, json)
}

/// Runs before any async runtime exists: the CPU limits apply to every thread.
pub fn run(mode: Mode, loaded: Loaded, o: Overrides, json: bool) -> Result<()> {
    let source = loaded.source();
    let Loaded { config: mut c, path, .. } = loaded;
    if let Some(e) = &o.flightsql {
        c.flightsql.enabled = true;
        c.flightsql.endpoint = e.clone();
    }
    if let Some(t) = &o.table {
        c.flightsql.table = t.clone();
    }
    if c.flightsql.enabled {
        let path = path.unwrap_or_else(|| PathBuf::from("--flightsql"));
        c.validate_flightsql().map_err(|message| BadConfigSnafu { path, message }.build())?;
    }
    let duration = o.duration.or((c.collect.duration_s > 0).then(|| Duration::from_secs(c.collect.duration_s)));
    if mode == Mode::Profile && duration.is_none() {
        return NoDurationSnafu.fail();
    }
    let listen = match (o.listen, mode) {
        (Some(l), _) => l,
        (None, Mode::Profile) => String::new(),
        (None, _) => c.daemon.listen.clone(),
    };
    // The pid file is also the lock: one exporter per pid file, whether a daemon or in the foreground.
    let lock = match mode {
        Mode::Profile => None,
        Mode::Export | Mode::Daemon => Some(PidLock::acquire(&c.daemon.pid_file)?),
    };
    // Before the async runtime and the SDK create their threads: they inherit affinity and nice.
    let cpus = daemon::apply_limits(&c.limits)?;
    let cpu_list = cpus.map(|_| daemon::cpu_list_string(&daemon::current_affinity().unwrap_or_default()));
    say(
        mode,
        format_args!(
            "gt {} starting: pid {}, config {source}, cpus {}",
            gravity_trace_core::VERSION,
            std::process::id(),
            match (cpus, &cpu_list) {
                (Some(n), Some(l)) => format!("{n} ({l})"),
                _ => "unlimited".to_string(),
            }
        ),
    );
    let rt = crate::runtime(cpus.map_or(4, |n| n.clamp(1, 4)))?;
    let r = rt.block_on(serve(mode, c, Run { listen, duration, lock, cpus: cpu_list, json }));
    if mode == Mode::Daemon
        && let Err(e) = &r
    {
        say(mode, format_args!("failed: {}", crate::error::chain(e)));
    }
    r
}

struct Run {
    listen: String,
    duration: Option<Duration>,
    lock: Option<PidLock>,
    cpus: Option<String>,
    json: bool,
}

async fn serve(mode: Mode, c: Config, run: Run) -> Result<()> {
    let Run { listen, duration, mut lock, cpus, json } = run;
    // Bind first: a busy port fails before the SDK is touched.
    let listener = match listen.as_str() {
        "" => None,
        addr => Some(tokio::net::TcpListener::bind(addr).await.context(ListenSnafu { addr: addr.to_string() })?),
    };
    let flight = c.flightsql.enabled.then(|| FlightOptions::from_config(&c.flightsql));
    let persist = (!c.flightsql.enabled && c.persist.enabled).then(|| PersistOptions::from_config(&c.persist));
    let target = match (&flight, &persist) {
        (Some(f), _) => format!(", streaming to {}", f.endpoint),
        (None, Some(p)) => format!(", persisting to {} ({})", p.location, p.format.as_str()),
        (None, None) => ", nothing streamed or persisted".to_string(),
    };
    let start = now_ns();
    let pipeline = live::start(live::PipelineConfig {
        top: c.collect.top,
        live: c.live_options(),
        diag: c.collect.diag_file.clone(),
        host_proc: c.collect.host_proc.clone(),
        persist,
        flight,
    })
    .await?;
    if let Some(l) = &mut lock {
        l.ready()?;
    }
    say(
        mode,
        format_args!(
            "collecting {}{target}{}",
            match (duration, mode) {
                (Some(d), _) => format!("for {}", fmt_span(d.as_nanos() as i64)),
                (None, Mode::Daemon) => "until `gt export-daemon stop`".to_string(),
                (None, _) => "until Ctrl-C / SIGTERM".to_string(),
            },
            if listen.is_empty() { String::new() } else { format!("; http://{listen}/metrics") }
        ),
    );
    let stop = daemon::wait_for_stop(duration);
    let (reason, served) = match listener {
        Some(l) => {
            let reason = std::sync::Arc::new(std::sync::OnceLock::new());
            let r = reason.clone();
            let served = axum::serve(l, live::router(pipeline.handle(), c.daemon.captures_dir.clone()))
                .with_graceful_shutdown(async move {
                    let _ = r.set(stop.await);
                })
                .await
                .context(ListenSnafu { addr: listen.clone() });
            (reason.get().copied().unwrap_or("HTTP server failed"), served)
        }
        None => (stop.await, Ok(())),
    };
    let end = now_ns();
    say(mode, format_args!("stopping ({reason}): {} ...", if c.flightsql.enabled { "sending what is left" } else { "finishing the open files" }));
    let h = pipeline.handle();
    pipeline.stop().await?;
    // Everything is written / sent: `gt export-daemon stop` waits for the pid file to go.
    drop(lock);
    served?;
    let f = &c.flightsql;
    let r = Report {
        start_utc_ns: start,
        end_utc_ns: end,
        seconds: (end - start) as f64 / 1e9,
        stopped_by: reason.to_string(),
        cpus,
        totals: h.totals(),
        flightsql: h.flight_stats().map(|s| s.report(&f.endpoint, &f.table, &f.pm_table)),
        persist: h.persist_stats().map(|s| PersistReport {
            dir: c.persist.dir.clone(),
            format: c.persist.format.as_str(),
            rows: s.rows.load(Ordering::Relaxed),
            files: s.files.load(Ordering::Relaxed),
            bytes: s.bytes.load(Ordering::Relaxed),
            dropped_rows: s.queue.dropped_rows.load(Ordering::Relaxed),
        }),
        drain_seconds: (now_ns() - end) as f64 / 1e9,
    };
    match mode {
        Mode::Daemon => render(&r).lines().for_each(|l| say(mode, l)),
        Mode::Profile if json => println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default()),
        Mode::Profile => print!("{}", render(&r)),
        Mode::Export => {}
    }
    if let Some(what) = crate::crash::first_panic() {
        return PanickedSnafu { what }.fail();
    }
    let dropped = [
        r.flightsql.as_ref().map(|f| (format!("streaming to {}", f.endpoint), f.dropped_rows)),
        r.persist.as_ref().map(|p| (format!("persisting to {}", p.dir), p.dropped_rows)),
    ];
    for (what, dropped) in dropped.into_iter().flatten().filter(|(_, n)| *n > 0) {
        if mode == Mode::Daemon {
            say(mode, format_args!("{what}: {dropped} activities were not delivered"));
        } else {
            return IncompleteSnafu { what, dropped }.fail();
        }
    }
    if mode == Mode::Export {
        println!("successful");
    }
    Ok(())
}

/// `412.3 MiB`.
pub fn fmt_bytes(b: f64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < U.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 { format!("{v:.0} B") } else { format!("{v:.1} {}", U[i]) }
}

/// The report as text lines.
fn render(r: &Report) -> String {
    let mut s = String::new();
    let t = &r.totals;
    let _ = writeln!(
        s,
        "collected {} .. {} ({:.1} s, {}): {} activities in {} snapshots{}",
        fmt_utc(r.start_utc_ns),
        fmt_utc(r.end_utc_ns),
        r.seconds,
        r.stopped_by,
        t.activities,
        t.snapshots,
        r.cpus.as_ref().map_or(String::new(), |c| format!(" (CPUs {c})"))
    );
    if t.lost_activities > 0 || t.snapshot_stall_seconds > 0.0 {
        let _ = writeln!(
            s,
            "  service    {} activities lost before gt (GPU times never arrived), snapshots stalled {:.1} s",
            t.lost_activities, t.snapshot_stall_seconds
        );
    }
    if let Some(f) = &r.flightsql {
        let _ = writeln!(
            s,
            "  flightsql  {} -> {}{}: {} activities{} in {} DoPuts; {} dropped, {} errors, {} retries",
            f.endpoint,
            f.table,
            if f.pm_table.is_empty() { String::new() } else { format!(", {}", f.pm_table) },
            f.rows,
            if f.pm_table.is_empty() { String::new() } else { format!(" + {} PM samples", f.pm_samples) },
            f.puts,
            f.dropped_rows,
            f.errors,
            f.retries
        );
        let t = &f.traffic;
        let rate = |b: f64| format!("{}/s", fmt_bytes(b));
        let _ = writeln!(
            s,
            "  traffic    {} sent ({:.1} B per row) over {} s: mean {}, p50 {}, p90 {}, p99 {}, max {} (data in {} of the seconds)",
            fmt_bytes(t.total_bytes as f64),
            f.bytes_per_row,
            t.seconds,
            rate(t.mean_bytes_per_s),
            rate(t.p50_bytes_per_s as f64),
            rate(t.p90_bytes_per_s as f64),
            rate(t.p99_bytes_per_s as f64),
            rate(t.max_bytes_per_s as f64),
            t.active_seconds
        );
    }
    if let Some(p) = &r.persist {
        let _ = writeln!(
            s,
            "  persisted  {} ({}): {} activities in {} files ({}), {} dropped",
            p.dir,
            p.format,
            p.rows,
            p.files,
            fmt_bytes(p.bytes as f64),
            p.dropped_rows
        );
    }
    let _ = writeln!(s, "  drained in {:.1} s after the stop", r.drain_seconds);
    s
}
