//! `gt export-daemon start|stop|restart|status|run|config`: the live pipeline as a background service, configured by
//! `config.toml` ([`crate::config`]).
//!
//! - `start` validates the configuration, re-runs this binary as `gt export-daemon run` in a new session (stdin
//!   closed, stdout / stderr appended to `daemon.log_file`) and returns once the daemon is serving, or shows the end of
//!   the log when it exits during startup.
//! - `run` is the daemon itself (also usable in the foreground under systemd or in a container; [`crate::export`]):
//!   CPU limits first (before any thread exists, so every thread inherits them), then the pipeline with the persisted
//!   trace (or the Flight SQL stream, `[flightsql]`), the HTTP API, and the pid file. It ends on SIGTERM / SIGINT or
//!   after `collect.duration_s`, finishing the open files / sending what is left. `gt export [DURATION]` is the same
//!   in the foreground of a terminal, with the same pid file: `stop` and `status` work on it too.
//! - The pid file is also the daemon's lock ([`crate::pidfile`]).
use crate::config::{self, Config, Loaded};
use crate::error::{
    AffinitySnafu, AlreadyRunningSnafu, CreateDirSnafu, DaemonExitedSnafu, DaemonNotReadySnafu, IoSnafu, KillSnafu,
    NoCpusSnafu, NotOurProcessSnafu, Result, SpawnDaemonSnafu, StopTimeoutSnafu,
};
pub(crate) use crate::pidfile::PidLock;
use crate::pidfile::{State, state};
use crate::export;
use crate::timefmt::{fmt_span, fmt_utc, now_ns};
use clap::Subcommand;
use snafu::ResultExt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Subcommand, Clone, Copy, Debug)]
pub enum Action {
    /// Start the daemon in the background (logs to daemon.log_file); returns once it serves.
    Start,
    /// Stop the daemon: SIGTERM, then wait until it has finished its files and exited.
    Stop,
    /// Stop (if running), then start.
    Restart,
    /// Is the daemon running? Pid, HTTP address, persisted trace on disk.
    Status,
    /// Run the daemon in the foreground (what `start` launches; for systemd or containers).
    Run,
    /// Print the effective configuration (built-in defaults + the config file + GT_<SECTION>_<KEY> environment
    /// variables) as TOML.
    Config,
}

/// Runs before any async runtime exists (`run` applies CPU limits first).
pub fn main(action: Action, config: Option<&Path>) -> Result<()> {
    let loaded = config::load(config)?;
    match action {
        Action::Start => start(&loaded),
        Action::Stop => stop(&loaded.config),
        Action::Restart => {
            stop(&loaded.config)?;
            start(&loaded)
        }
        Action::Status => status(&loaded),
        Action::Run => export::run(export::Mode::Daemon, loaded, export::Overrides::default(), false),
        Action::Config => {
            match &loaded.path {
                Some(p) => println!("# {}", p.display()),
                None => println!("# built-in defaults ({} does not exist)", config::DEFAULT_PATH),
            }
            if !loaded.env.is_empty() {
                println!("# overridden by the environment: {}", loaded.env.join(", "));
            }
            print!("{}", loaded.config.to_toml());
            Ok(())
        }
    }
}

fn log(msg: impl std::fmt::Display) {
    eprintln!("{} gt export-daemon: {msg}", fmt_utc(now_ns()));
}

fn start(loaded: &Loaded) -> Result<()> {
    let c = &loaded.config;
    let pid_file = &c.daemon.pid_file;
    match state(pid_file)? {
        State::Stopped => {}
        State::Starting => return AlreadyRunningSnafu { pid: "starting".to_string(), pid_file: pid_file.clone() }.fail(),
        State::Running(pid) => return AlreadyRunningSnafu { pid: pid.to_string(), pid_file: pid_file.clone() }.fail(),
    }
    let log_path = &c.daemon.log_file;
    if let Some(d) = log_path.parent() {
        std::fs::create_dir_all(d).context(CreateDirSnafu { path: d.to_path_buf() })?;
    }
    let out = OpenOptions::new().create(true).append(true).open(log_path).context(IoSnafu { context: format!("cannot open {}", log_path.display()) })?;
    let log_from = out.metadata().map(|m| m.len()).unwrap_or(0);
    let err = out.try_clone().context(IoSnafu { context: format!("cannot open {}", log_path.display()) })?;
    let exe = std::env::current_exe().context(IoSnafu { context: "cannot find the gt binary".to_string() })?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("export-daemon");
    if let Some(p) = &loaded.path {
        cmd.arg("--config").arg(p);
    }
    cmd.arg("run").stdin(std::process::Stdio::null()).stdout(out).stderr(err);
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: setsid is async-signal-safe; it detaches the child from this terminal and process group, so the
        // shell's hangup / Ctrl-C does not reach it.
        unsafe {
            cmd.pre_exec(|| if libc::setsid() < 0 { Err(std::io::Error::last_os_error()) } else { Ok(()) });
        }
    }
    let mut child = cmd.spawn().context(SpawnDaemonSnafu)?;
    let pid = child.id();
    let t0 = Instant::now();
    const READY_WAIT: Duration = Duration::from_secs(120);
    loop {
        if let Some(st) = child.try_wait().context(SpawnDaemonSnafu)? {
            return DaemonExitedSnafu { status: st.to_string(), log: log_path.clone(), tail: log_tail(log_path, log_from, 30) }.fail();
        }
        if state(pid_file)? == State::Running(pid) {
            break;
        }
        if t0.elapsed() > READY_WAIT {
            return DaemonNotReadySnafu { pid, secs: READY_WAIT.as_secs(), log: log_path.clone() }.fail();
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    println!("gt export-daemon started: pid {pid} ({:.1}s)", t0.elapsed().as_secs_f64());
    describe(loaded);
    Ok(())
}

/// Lines written to `path` after byte `from`, at most `n`.
fn log_tail(path: &Path, from: u64, n: usize) -> String {
    let mut s = String::new();
    if let Ok(mut f) = File::open(path) {
        let _ = f.seek(SeekFrom::Start(from));
        let _ = f.read_to_string(&mut s);
    }
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

fn describe(loaded: &Loaded) {
    let c = &loaded.config;
    println!("  config   {}", loaded.source());
    if !c.daemon.listen.is_empty() {
        println!("  http     http://{}/metrics  (POST /capture, POST /persist/flush)", c.daemon.listen);
    }
    if c.flightsql.enabled {
        let f = &c.flightsql;
        println!(
            "  stream   {} -> {}{} (Flight SQL ingest; nothing persisted)",
            f.endpoint,
            f.table,
            if f.pm_table.is_empty() { String::new() } else { format!(", {}", f.pm_table) }
        );
    } else if c.persist.enabled {
        let files = list_persisted(&c.persist.dir);
        let bytes: u64 = files.iter().map(|f| f.bytes).sum();
        let traces: Vec<_> = files.iter().filter(|f| f.kind != crate::files::FileKind::Pm).collect();
        let span = match (traces.first(), traces.iter().map(|f| f.span.1).max()) {
            (Some(f), Some(e)) => format!(", {} .. {}", fmt_utc(f.span.0), fmt_utc(e)),
            _ => String::new(),
        };
        println!(
            "  traces   {}: {} {} files, {:.1} MiB{span} (new file every {} s)",
            c.persist.dir,
            traces.len(),
            c.persist.format.as_str(),
            bytes as f64 / (1 << 20) as f64,
            c.persist.rotate_s
        );
    } else {
        println!("  traces   not persisted (persist.enabled = false)");
    }
    if c.collect.duration_s > 0 {
        println!("  stops    after {} s of collection", c.collect.duration_s);
    }
    println!("  log      {}", c.daemon.log_file.display());
}

/// The persisted files (empty when the store cannot be listed); `start` / `status` run without an async runtime.
fn list_persisted(location: &str) -> Vec<crate::files::StoredFile> {
    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(_) => return Vec::new(),
    };
    crate::store::Store::open(location).map_or_else(|_| Vec::new(), |s| rt.block_on(crate::files::list(&s)).unwrap_or_default())
}

/// The arguments of a process (empty when it is gone, or a zombie).
fn cmdline(pid: u32) -> Vec<String> {
    let b = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
    let mut args: Vec<String> = b.split(|&c| c == 0).map(|a| String::from_utf8_lossy(a).into_owned()).collect();
    if args.last().is_some_and(String::is_empty) {
        args.pop(); // after the terminating NUL
    }
    args
}

/// `gt export-daemon` / `gt export` by the subcommand (`None`: another program, e.g. a recycled pid).
fn exporter_kind<S: AsRef<str>>(args: &[S]) -> Option<&'static str> {
    match args.get(1).map(AsRef::as_ref) {
        Some("export-daemon") => Some("gt export-daemon"),
        Some("export") => Some("gt export"),
        _ => None,
    }
}

fn stop(c: &Config) -> Result<()> {
    let pid_file = &c.daemon.pid_file;
    let t0 = Instant::now();
    let pid = loop {
        match state(pid_file)? {
            State::Stopped => {
                println!("gt export-daemon is not running ({})", pid_file.display());
                return Ok(());
            }
            State::Running(pid) => break pid,
            // the pid appears once startup is done
            State::Starting if t0.elapsed() < Duration::from_secs(c.daemon.stop_timeout_s) => {
                std::thread::sleep(Duration::from_millis(200));
            }
            State::Starting => return StopTimeoutSnafu { pid: 0u32, secs: c.daemon.stop_timeout_s }.fail(),
        }
    };
    let cmd = cmdline(pid);
    let Some(kind) = exporter_kind(&cmd) else {
        return NotOurProcessSnafu { pid, cmdline: cmd.join(" "), pid_file: pid_file.clone() }.fail();
    };
    // SAFETY: kill(2) with a pid read from the locked pid file and checked above.
    if unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) } != 0 {
        return Err(std::io::Error::last_os_error()).context(KillSnafu { pid });
    }
    println!("stopping {kind} (pid {pid}): {} ...", if c.flightsql.enabled { "sending what is left" } else { "finishing the open files" });
    let t0 = Instant::now();
    while state(pid_file)? != State::Stopped {
        if t0.elapsed() > Duration::from_secs(c.daemon.stop_timeout_s) {
            return StopTimeoutSnafu { pid, secs: c.daemon.stop_timeout_s }.fail();
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    // The lock is released when the process is gone; wait for /proc too (bounded).
    while Path::new(&format!("/proc/{pid}")).exists() && t0.elapsed() < Duration::from_secs(c.daemon.stop_timeout_s) {
        if cmdline(pid).is_empty() {
            break; // zombie, waiting for its reaper
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    println!("stopped ({:.1}s)", t0.elapsed().as_secs_f64());
    Ok(())
}

fn status(loaded: &Loaded) -> Result<()> {
    let c = &loaded.config;
    match state(&c.daemon.pid_file)? {
        State::Stopped => crate::error::NotRunningSnafu { pid_file: c.daemon.pid_file.clone() }.fail(),
        State::Starting => {
            println!("gt export-daemon is starting ({})", c.daemon.pid_file.display());
            Ok(())
        }
        State::Running(pid) => {
            let since = std::fs::metadata(&c.daemon.pid_file)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as i64);
            let kind = exporter_kind(&cmdline(pid)).unwrap_or("gt export-daemon");
            match since {
                Some(s) => println!("{kind} running: pid {pid}, serving since {} ({})", fmt_utc(s), fmt_span(now_ns() - s)),
                None => println!("{kind} running: pid {pid}"),
            }
            if let Some(cpus) = std::fs::read_to_string(format!("/proc/{pid}/status"))
                .ok()
                .and_then(|s| s.lines().find(|l| l.starts_with("Cpus_allowed_list:")).and_then(|l| l.split_whitespace().next_back().map(str::to_string)))
            {
                println!("  cpus     {cpus}");
            }
            describe(loaded);
            Ok(())
        }
    }
}

/// CPU affinity and nice of every thread of this process; the number of CPUs it may use when limited.
pub(crate) fn apply_limits(l: &config::Limits) -> Result<Option<usize>> {
    if l.nice != 0 {
        // Per thread on Linux: every existing thread (normally just this one); later threads inherit it.
        for tid in tasks() {
            // SAFETY: setpriority on a thread id of this process.
            if unsafe { libc::setpriority(libc::PRIO_PROCESS, tid as libc::id_t, l.nice) } != 0 {
                log(format_args!("cannot set nice {}: {}", l.nice, std::io::Error::last_os_error()));
                break;
            }
        }
    }
    if l.max_cpus == 0 && l.cpu_list.is_empty() {
        return Ok(None);
    }
    let allowed = current_affinity().context(AffinitySnafu)?;
    let mut cpus: Vec<usize> = if l.cpu_list.is_empty() {
        allowed.clone()
    } else {
        let wanted = config::parse_cpu_list(&l.cpu_list).unwrap_or_default();
        wanted.into_iter().filter(|c| allowed.contains(c)).collect()
    };
    if l.max_cpus > 0 && cpus.len() > l.max_cpus as usize {
        // The last ones: CPU 0 and its neighbours tend to carry interrupts and system services.
        cpus.drain(..cpus.len() - l.max_cpus as usize);
    }
    if cpus.is_empty() {
        return NoCpusSnafu { wanted: l.cpu_list.clone(), allowed: cpu_list_string(&allowed) }.fail();
    }
    set_affinity(&cpus).context(AffinitySnafu)?;
    Ok(Some(cpus.len()))
}

/// Pins every thread of this process to `cpus`.
#[cfg(target_os = "linux")]
fn set_affinity(cpus: &[usize]) -> std::io::Result<()> {
    // SAFETY: an all-zero cpu_set_t is the empty set.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &c in cpus {
        // SAFETY: c < 4096 (parse_cpu_list / sched_getaffinity), CPU_SET ignores CPUs beyond the set size.
        unsafe { libc::CPU_SET(c, &mut set) };
    }
    for tid in tasks() {
        // SAFETY: `set` is initialised; the size is that of the struct.
        if unsafe { libc::sched_setaffinity(tid as libc::pid_t, std::mem::size_of::<libc::cpu_set_t>(), &set) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

/// CPU limits (`limits.max_cpus`, `limits.cpu_list`) are Linux only.
#[cfg(not(target_os = "linux"))]
fn set_affinity(_cpus: &[usize]) -> std::io::Result<()> {
    Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "CPU affinity is only supported on Linux"))
}

/// Thread ids of this process.
fn tasks() -> Vec<u32> {
    let mut out: Vec<u32> = std::fs::read_dir("/proc/self/task")
        .map(|rd| rd.flatten().filter_map(|e| e.file_name().to_str()?.parse().ok()).collect())
        .unwrap_or_default();
    if out.is_empty() {
        out.push(0); // the calling thread
    }
    out
}

/// The CPUs this thread may run on.
#[cfg(target_os = "linux")]
pub(crate) fn current_affinity() -> std::io::Result<Vec<usize>> {
    // SAFETY: an all-zero cpu_set_t is the empty set.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: `set` is a valid out-buffer of the given size.
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let n = 8 * std::mem::size_of::<libc::cpu_set_t>();
    // SAFETY: c < the set size.
    Ok((0..n).filter(|&c| unsafe { libc::CPU_ISSET(c, &set) }).collect())
}

/// Every CPU (no affinity outside Linux).
#[cfg(not(target_os = "linux"))]
pub(crate) fn current_affinity() -> std::io::Result<Vec<usize>> {
    Ok((0..std::thread::available_parallelism()?.get()).collect())
}

pub(crate) fn cpu_list_string(cpus: &[usize]) -> String {
    cpus.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",")
}

/// SIGTERM, SIGINT, the end of the collection time, or a panic of any thread. SIGHUP is logged and ignored.
pub(crate) async fn wait_for_stop(limit: Option<Duration>) -> &'static str {
    use tokio::signal::unix::{SignalKind, signal};
    let deadline = sleep_or_forever(limit);
    tokio::pin!(deadline);
    let (mut term, mut hup) = match (signal(SignalKind::terminate()), signal(SignalKind::hangup())) {
        (Ok(t), Ok(h)) => (t, h),
        _ => {
            log("cannot install signal handlers; stopping only on Ctrl-C or the collection time");
            return tokio::select! {
                _ = tokio::signal::ctrl_c() => "SIGINT",
                _ = &mut deadline => "collection time reached",
                _ = crate::crash::panicked() => "a thread panicked",
            };
        }
    };
    let panicked = crate::crash::panicked();
    tokio::pin!(panicked);
    loop {
        tokio::select! {
            _ = term.recv() => return "SIGTERM",
            _ = tokio::signal::ctrl_c() => return "SIGINT",
            _ = &mut deadline => return "collection time reached",
            _ = &mut panicked => return "a thread panicked",
            _ = hup.recv() => log("SIGHUP ignored (a restart reloads the configuration)"),
        }
    }
}

async fn sleep_or_forever(limit: Option<Duration>) {
    match limit {
        Some(d) => tokio::time::sleep(d).await,
        None => std::future::pending::<()>().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exporter_command_lines() {
        assert_eq!(exporter_kind(&["/usr/bin/gt", "export-daemon", "--config", "/etc/gt.toml", "run"]), Some("gt export-daemon"));
        assert_eq!(exporter_kind(&["/opt/my tools/gt", "export", "10m", "--json"]), Some("gt export"));
        assert_eq!(exporter_kind(&["gt", "exporter", "--listen", ":9465"]), None);
        assert_eq!(exporter_kind(&["python", "x.py", "export"]), None);
        assert_eq!(exporter_kind(&["export"]), None);
        assert_eq!(exporter_kind::<&str>(&[]), None);
        #[cfg(target_os = "linux")] // /proc/<pid>/cmdline
        assert_eq!(cmdline(std::process::id()).first().map(String::is_empty), Some(false));
    }

}
