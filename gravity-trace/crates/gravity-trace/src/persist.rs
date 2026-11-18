//! The daemon's persisted trace: every merged activity, written by its own thread (`gt-persist`) to objects in one
//! store location, a local directory or an object store URL ([`crate::store`]), in one of two formats
//! (`persist.format`):
//! - `parquet`: `<start>-<end>.parquet` (PM samples to `<start>-<end>.pm.parquet`), read back by `gt dump`;
//! - `perfetto`: `<start>-<end>.json.gz`, Perfetto / Chrome trace JSON written while the activities arrive, with the
//!   PM samples as counter tracks ([`crate::writer::PerfettoWriter`]).
//!
//! - The merge thread hands activities over in chunks through a queue bounded in bytes (`max_buffer_mib`,
//!   [`crate::feed`]): when the writer falls behind, chunks are dropped and counted instead of slowing the merge thread
//!   (and with it the SDK's snapshots) down.
//! - A file is closed and renamed to the span of its rows (smallest start, largest end, UTC ns) at the first snapshot
//!   boundary after `rotate_s`, on `POST /persist/flush` and when the daemon stops. Activities complete late (their GPU
//!   times arrive with a later snapshot), so neighbouring files may overlap by a snapshot interval.
//! - Beyond `max_disk_gib` the oldest files are deleted.
use crate::error::{PersistOffSnafu, PersistStoppedSnafu, Result, SpawnThreadSnafu};
use crate::config::PersistFormat;
use crate::feed::{Feed, Msg, Names, Queue};
use crate::mapper::{Activity, PmSample};
use crate::files::{FileKind, list};
use crate::store::{Store, remove_stale_temporaries};
use crate::timefmt::fmt_utc;
use crate::writer::{CHROME_EXT, PerfettoWriter, TRACE_EXT, TraceWriter, Written, parse_span, write_pm};
use gravity_trace_core::model::ProcessInfo;
use gravity_trace_core::parquet_io::WriteOptions;
use snafu::ResultExt;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};
use tokio::runtime::Handle;
use tokio::sync::oneshot;

#[derive(Clone, Debug)]
pub struct PersistOptions {
    /// A directory or an object store URL.
    pub location: String,
    pub format: PersistFormat,
    pub rotate: Duration,
    pub max_buffer_bytes: usize,
    pub row_group_rows: usize,
    pub zstd_level: i32,
    /// gzip level of Perfetto files.
    pub gzip_level: u32,
    /// 0 = no limit.
    pub max_disk_bytes: u64,
    pub pm: bool,
}

impl PersistOptions {
    pub fn from_config(p: &crate::config::Persist) -> Self {
        Self {
            location: p.dir.clone(),
            format: p.format,
            rotate: Duration::from_secs(p.rotate_s),
            max_buffer_bytes: (p.max_buffer_mib as usize).saturating_mul(1 << 20),
            row_group_rows: p.row_group_rows,
            zstd_level: p.zstd_level,
            gzip_level: p.gzip_level,
            max_disk_bytes: p.max_disk_gib.saturating_mul(1 << 30),
            pm: p.pm,
        }
    }
}

/// Counters behind the `nvtrace_persist_*` series.
#[derive(Debug, Default)]
pub struct PersistStats {
    /// Activities in closed trace files.
    pub rows: AtomicU64,
    /// The write buffer (and the activities dropped: buffer full, or a file write failed).
    pub queue: Arc<Queue>,
    pub pm_samples: AtomicU64,
    pub files: AtomicU64,
    pub bytes: AtomicU64,
    pub errors: AtomicU64,
    pub deleted_files: AtomicU64,
    pub disk_bytes: AtomicU64,
    /// End of the newest closed trace file (UTC ns).
    pub newest_end_ns: AtomicI64,
}

/// What `POST /persist/flush` answers.
#[derive(Debug, Default, serde::Serialize)]
pub struct FlushReport {
    /// The trace file closed: `.parquet` or `.json.gz` (Perfetto, with the PM samples in it).
    pub trace: Option<String>,
    pub pm: Option<String>,
    pub rows: u64,
    /// End of the newest trace file on disk after the flush (UTC ns).
    pub newest_end_ns: i64,
}

/// What the HTTP handlers and `/metrics` use.
#[derive(Clone)]
pub struct Control {
    tx: mpsc::Sender<Msg>,
    pub stats: Arc<PersistStats>,
}

impl Control {
    /// Close the open file now (everything merged so far is then on disk).
    pub async fn flush(&self) -> Result<FlushReport> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Msg::Flush(reply)).map_err(|_| PersistStoppedSnafu.build())?;
        rx.await.map_err(|_| PersistStoppedSnafu.build())?
    }
}

/// `Control::flush` of an optional writer.
pub async fn flush(c: Option<&Control>) -> Result<FlushReport> {
    match c {
        Some(c) => c.flush().await,
        None => PersistOffSnafu.fail(),
    }
}

/// Start the writer thread; `rt` drives the object store (callable from async code: the thread does the IO).
pub fn start(opts: PersistOptions, rt: &Handle) -> Result<(Feed, Control, std::thread::JoinHandle<()>)> {
    let store = Store::open(&opts.location)?;
    let stats = Arc::new(PersistStats::default());
    let (tx, rx) = mpsc::channel();
    let feed = Feed::new(tx.clone(), stats.queue.clone(), opts.max_buffer_bytes, opts.pm);
    let control = Control { tx, stats: stats.clone() };
    let mut w = Writer { opts, store, rt: rt.clone(), cur: None, stats, names: Default::default() };
    let thread = std::thread::Builder::new()
        .name("gt-persist".into())
        .spawn(move || w.run(rx))
        .context(SpawnThreadSnafu { name: "gt-persist" })?;
    Ok((feed, control, thread))
}

/// The open file.
enum Open {
    /// The trace, and the PM samples for the `.pm.parquet` file written next to it at the rotation.
    Parquet { trace: Box<TraceWriter>, pm: Vec<PmSample> },
    Perfetto(Box<PerfettoWriter>),
}

impl Open {
    fn rows(&self) -> u64 {
        match self {
            Open::Parquet { trace, .. } => trace.rows(),
            Open::Perfetto(p) => p.rows(),
        }
    }

    fn abandon(self) {
        match self {
            Open::Parquet { trace, .. } => trace.abandon(),
            Open::Perfetto(p) => p.abandon(),
        }
    }
}

struct Current {
    file: Open,
    opened: Instant,
}

struct Writer {
    opts: PersistOptions,
    store: Store,
    rt: Handle,
    cur: Option<Current>,
    stats: Arc<PersistStats>,
    /// Every process and GPU name so far (a new Perfetto file starts with them).
    names: (BTreeMap<u32, ProcessInfo>, BTreeMap<[u8; 16], String>),
}

impl Writer {
    fn run(&mut self, rx: mpsc::Receiver<Msg>) {
        remove_stale_temporaries(&self.store);
        match self.rt.block_on(list(&self.store)) {
            Ok(files) => {
                self.stats.disk_bytes.store(files.iter().map(|f| f.bytes).sum(), Ordering::Relaxed);
                if let Some(e) = files.iter().filter(|f| f.kind != FileKind::Pm).map(|f| f.span.1).max() {
                    self.stats.newest_end_ns.store(e, Ordering::Relaxed);
                }
            }
            Err(e) => eprintln!("gt: persist: {}", crate::error::chain(&e)),
        }
        loop {
            match rx.recv_timeout(Duration::from_secs(1)) {
                Ok(Msg::Rows(rows)) => {
                    self.write(&rows);
                    self.stats.queue.taken(rows.len());
                }
                Ok(Msg::Pm(s)) => {
                    self.stats.pm_samples.fetch_add(s.len() as u64, Ordering::Relaxed);
                    self.pm(s);
                }
                Ok(Msg::Names(n)) => {
                    self.names.0.extend(n.procs.iter().map(|p| (p.pid, p.clone())));
                    self.names.1.extend(n.gpus.iter().cloned());
                    if let Some(Current { file: Open::Perfetto(p), .. }) = self.cur.as_mut()
                        && let Err(e) = p.names(&n)
                    {
                        let lost = self.open_rows();
                        self.fail(e, lost);
                    }
                }
                Ok(Msg::Snapshot) => {
                    if let Some(Current { file: Open::Perfetto(p), .. }) = self.cur.as_mut() {
                        p.snapshot();
                    }
                    if self.cur.as_ref().is_some_and(|c| c.opened.elapsed() >= self.opts.rotate) {
                        let _ = self.rotate();
                    }
                }
                Ok(Msg::Flush(reply)) => {
                    let _ = reply.send(self.rotate());
                }
                Ok(Msg::Close) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                    let _ = self.rotate();
                    return;
                }
                // No snapshot for a long time (the pipeline is stuck): do not keep a file open forever.
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if self.cur.as_ref().is_some_and(|c| c.opened.elapsed() >= 2 * self.opts.rotate) {
                        let _ = self.rotate();
                    }
                }
            }
        }
    }

    fn current(&mut self) -> Result<&mut Current> {
        if self.cur.is_none() {
            let (o, store, rt) = (&self.opts, &self.store, &self.rt);
            let file = match o.format {
                PersistFormat::Parquet => {
                    let mut wo = WriteOptions { row_group_rows: o.row_group_rows, zstd_level: o.zstd_level, ..WriteOptions::trace() };
                    wo.metadata = BTreeMap::from([
                        ("source".to_string(), "nsight-perf-sdk-for-ai".to_string()),
                        ("capture".to_string(), "gt export-daemon".to_string()),
                    ]);
                    Open::Parquet { trace: Box::new(TraceWriter::with_options(store, rt, wo)), pm: Vec::new() }
                }
                PersistFormat::Perfetto => {
                    let names = Names {
                        procs: self.names.0.values().cloned().collect(),
                        gpus: self.names.1.iter().map(|(u, d)| (*u, d.clone())).collect(),
                    };
                    Open::Perfetto(Box::new(PerfettoWriter::new(store, rt, &names, o.gzip_level)?))
                }
            };
            self.cur = Some(Current { file, opened: Instant::now() });
        }
        Ok(self.cur.as_mut().expect("opened above"))
    }

    fn write(&mut self, rows: &[Activity]) {
        let before = self.open_rows();
        let r = self.current().and_then(|cur| match &mut cur.file {
            Open::Parquet { trace, .. } => rows.iter().try_for_each(|a| trace.push(a)),
            Open::Perfetto(p) => p.push(rows),
        });
        if let Err(e) = r {
            self.fail(e, before + rows.len() as u64);
        }
    }

    fn pm(&mut self, s: Vec<PmSample>) {
        let r = self.current().and_then(|cur| match &mut cur.file {
            Open::Parquet { pm, .. } => {
                pm.extend(s);
                Ok(())
            }
            Open::Perfetto(p) => p.pm(&s),
        });
        if let Err(e) = r {
            let lost = self.open_rows();
            self.fail(e, lost);
        }
    }

    fn open_rows(&self) -> u64 {
        self.cur.as_ref().map_or(0, |c| c.file.rows())
    }

    /// A write failed (e.g. disk full): give the open file up (`lost` activities with it) and start over with the
    /// next rows.
    fn fail(&mut self, e: crate::error::Error, lost: u64) {
        eprintln!("gt: persist: {e}; dropping the open file");
        self.stats.errors.fetch_add(1, Ordering::Relaxed);
        self.stats.queue.dropped(lost);
        if let Some(c) = self.cur.take() {
            c.file.abandon();
        }
    }

    /// Close the open file (if any) under its time span (with its PM samples), apply the disk limit.
    fn rotate(&mut self) -> Result<FlushReport> {
        let Some(cur) = self.cur.take() else {
            return Ok(FlushReport { newest_end_ns: self.stats.newest_end_ns.load(Ordering::Relaxed), ..Default::default() });
        };
        let rows = cur.file.rows();
        let (trace, pm, ext) = match cur.file {
            Open::Parquet { mut trace, pm } => {
                trace.set_processes(self.names.0.values().cloned().collect());
                let t = trace.close(None);
                let p = if self.opts.pm { write_pm(&self.store, &self.rt, &pm, None) } else { Ok(None) };
                (t, p, TRACE_EXT)
            }
            Open::Perfetto(p) => (p.close(), Ok(None), CHROME_EXT),
        };
        let mut report = FlushReport { rows, ..Default::default() };
        for (r, is_trace) in [(trace, true), (pm, false)] {
            match r {
                Ok(Some(Written { name, shown, bytes })) => {
                    self.stats.files.fetch_add(1, Ordering::Relaxed);
                    self.stats.bytes.fetch_add(bytes, Ordering::Relaxed);
                    self.stats.disk_bytes.fetch_add(bytes, Ordering::Relaxed);
                    if is_trace {
                        self.stats.rows.fetch_add(rows, Ordering::Relaxed);
                        if let Some((s, e)) = parse_span(&name, ext) {
                            self.stats.newest_end_ns.fetch_max(e, Ordering::Relaxed);
                            eprintln!(
                                "gt: persist: {name} ({rows} rows, {:.1} MiB, {} .. {})",
                                bytes as f64 / (1 << 20) as f64,
                                fmt_utc(s),
                                fmt_utc(e)
                            );
                        }
                        report.trace = Some(shown);
                    } else {
                        report.pm = Some(shown);
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("gt: persist: {e}");
                    self.stats.errors.fetch_add(1, Ordering::Relaxed);
                    if is_trace {
                        self.stats.queue.dropped(rows);
                    }
                }
            }
        }
        self.enforce_disk_limit();
        report.newest_end_ns = self.stats.newest_end_ns.load(Ordering::Relaxed);
        Ok(report)
    }

    fn enforce_disk_limit(&mut self) {
        let Ok(mut files) = self.rt.block_on(list(&self.store)) else { return };
        let mut total: u64 = files.iter().map(|f| f.bytes).sum();
        if self.opts.max_disk_bytes > 0 && total > self.opts.max_disk_bytes {
            files.sort_by_key(|f| f.span);
            // never the newest file
            for f in &files[..files.len().saturating_sub(1)] {
                if total <= self.opts.max_disk_bytes {
                    break;
                }
                if self.rt.block_on(self.store.delete(&f.name)).is_ok() {
                    total -= f.bytes;
                    self.stats.deleted_files.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        self.stats.disk_bytes.store(total, Ordering::Relaxed);
    }
}

/// `GET /metrics` lines of the persisted trace.
pub fn render(o: &mut String, pod: &str, s: &PersistStats) {
    use std::fmt::Write as _;
    let mut g = |name: &str, kind: &str, help: &str, v: String| {
        let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name}{{pod=\"{pod}\"}} {v}");
    };
    let l = |a: &AtomicU64| a.load(Ordering::Relaxed).to_string();
    g("nvtrace_persist_rows_total", "counter", "Activities in closed persisted trace files.", l(&s.rows));
    g(
        "nvtrace_persist_dropped_rows_total",
        "counter",
        "Activities not persisted: the write buffer (persist.max_buffer_mib) was full, or a file write failed.",
        l(&s.queue.dropped_rows),
    );
    g("nvtrace_persist_pm_samples_total", "counter", "PM samples handed to the persisted trace.", l(&s.pm_samples));
    g("nvtrace_persist_files_total", "counter", "Persisted files closed.", l(&s.files));
    g("nvtrace_persist_bytes_total", "counter", "Bytes of persisted files closed.", l(&s.bytes));
    g("nvtrace_persist_errors_total", "counter", "Persisted-file write errors.", l(&s.errors));
    g("nvtrace_persist_deleted_files_total", "counter", "Oldest files deleted for persist.max_disk_gib.", l(&s.deleted_files));
    g("nvtrace_persist_queue_bytes", "gauge", "Activities waiting for the disk writer (approximate bytes).", s.queue.bytes.load(Ordering::Relaxed).to_string());
    g("nvtrace_persist_disk_bytes", "gauge", "Bytes of persisted files in persist.dir.", l(&s.disk_bytes));
    g(
        "nvtrace_persist_newest_end_seconds",
        "gauge",
        "End of the newest persisted trace file (Unix time).",
        format!("{:.3}", s.newest_end_ns.load(Ordering::Relaxed) as f64 / 1e9),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use gravity_trace_core::model::EventKind;

    fn act(start: i64) -> Activity {
        Activity {
            kind: EventKind::Kernel,
            name: Arc::from("k"),
            start_ns: start,
            end_ns: start + 10,
            pid: 1,
            ns_pid: 1,
            tid: None,
            device: Some(0),
            context: None,
            stream: Some(7),
            correlation: 1,
            graph_id: None,
            graph_node_id: None,
            grid: None,
            block: None,
            shmem: None,
            bytes: None,
        }
    }

    fn opts(dir: &std::path::Path) -> PersistOptions {
        PersistOptions {
            location: dir.display().to_string(),
            format: PersistFormat::Parquet,
            rotate: Duration::from_secs(3600),
            max_buffer_bytes: 1 << 30,
            row_group_rows: 1024,
            zstd_level: 3,
            gzip_level: 1,
            max_disk_bytes: 0,
            pm: true,
        }
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build().expect("runtime")
    }

    fn names(rt: &tokio::runtime::Runtime, dir: &std::path::Path) -> Vec<String> {
        let store = Store::open(&dir.display().to_string()).expect("store");
        rt.block_on(list(&store)).expect("list").into_iter().map(|f| f.name).collect()
    }

    #[test]
    fn files_are_named_by_their_span_and_rotate_on_flush() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rt = runtime();
        let (mut feed, control, thread) = start(opts(dir.path()), rt.handle()).expect("start");
        for t in [5_000, 1_000, 3_000] {
            feed.push(act(t));
        }
        feed.snapshot_done();
        let r = rt.block_on(control.flush()).expect("flush");
        assert_eq!(r.rows, 3);
        assert_eq!(r.newest_end_ns, 5_010);
        assert_eq!(r.trace, Some(format!("{}/1000-5010.parquet", dir.path().display())));
        let r = rt.block_on(control.flush()).expect("flush with nothing open");
        assert!(r.trace.is_none());
        feed.push(act(9_000));
        feed.finish();
        thread.join().expect("writer thread");
        assert_eq!(names(&rt, dir.path()), ["1000-5010.parquet", "9000-9010.parquet"], "closed on stop");
        let all: Vec<_> = std::fs::read_dir(dir.path()).expect("ls").flatten().map(|e| e.file_name()).collect();
        assert_eq!(all.len(), 2, "no temporaries left: {all:?}");
        assert_eq!(control.stats.rows.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn a_full_buffer_drops_chunks_and_counts_them() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rt = runtime();
        let mut o = opts(dir.path());
        o.max_buffer_bytes = crate::feed::ROW_BYTES; // not even one chunk fits
        let (mut feed, control, thread) = start(o, rt.handle()).expect("start");
        for t in 0..10 {
            feed.push(act(t));
        }
        feed.finish();
        thread.join().expect("writer thread");
        assert_eq!(control.stats.queue.dropped_rows.load(Ordering::Relaxed), 10);
        assert_eq!(control.stats.rows.load(Ordering::Relaxed), 0);
        assert!(names(&rt, dir.path()).is_empty());
    }

    #[test]
    fn the_oldest_files_go_beyond_the_disk_limit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rt = runtime();
        let mut o = opts(dir.path());
        o.max_disk_bytes = 1; // keeps only the newest file
        let (mut feed, control, thread) = start(o, rt.handle()).expect("start");
        for t in [1_000, 2_000, 3_000] {
            feed.push(act(t));
            feed.snapshot_done();
            rt.block_on(control.flush()).expect("flush");
        }
        feed.finish();
        thread.join().expect("writer thread");
        assert_eq!(names(&rt, dir.path()), ["3000-3010.parquet"]);
        assert_eq!(control.stats.deleted_files.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn perfetto_files_hold_the_activities_and_the_pm_samples() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rt = runtime();
        let mut o = opts(dir.path());
        o.format = PersistFormat::Perfetto;
        let (mut feed, control, thread) = start(o, rt.handle()).expect("start");
        let procs = vec![ProcessInfo { pid: 1, ns_pid: 1, exe: "/bin/app".to_string(), title: String::new() }];
        feed.names(&Names { procs, gpus: vec![([7; 16], "NVIDIA Test GPU".to_string())] });
        let api = Activity { kind: EventKind::Driver, name: Arc::from("cuLaunchKernel"), tid: Some(5), device: None, stream: None, ..act(900) };
        feed.push(act(1_000));
        feed.push(api);
        feed.push(act(3_000));
        feed.push_pm(&[PmSample { gpu: [7; 16], end_utc_ns: 2_000, metric: Arc::from("sm__busy"), value: 0.5 }]);
        feed.snapshot_done();
        let r = rt.block_on(control.flush()).expect("flush");
        assert_eq!((r.rows, r.pm.as_deref()), (3, None), "the PM samples are in the same file");
        assert_eq!(r.trace, Some(format!("{}/900-3010.json.gz", dir.path().display())));
        feed.finish();
        thread.join().expect("writer thread");
        let store = Store::open(&dir.path().display().to_string()).expect("store");
        let files = rt.block_on(list(&store)).expect("list");
        assert_eq!(files.iter().map(|f| (f.name.as_str(), f.kind)).collect::<Vec<_>>(), [("900-3010.json.gz", FileKind::Perfetto)]);
        let gz = std::fs::read(dir.path().join("900-3010.json.gz")).expect("read");
        let doc: serde_json::Value = serde_json::from_reader(flate2::read::GzDecoder::new(&gz[..])).expect("JSON");
        let ev = doc["traceEvents"].as_array().expect("events");
        let count = |ph: &str| ev.iter().filter(|e| e["ph"] == ph).count();
        assert_eq!((count("X"), count("C"), count("s"), count("f")), (3, 1, 1, 1), "3 activities, a counter, one launch arrow");
        let named = |n: &str| ev.iter().any(|e| e["ph"] == "M" && e["args"]["name"] == n);
        assert!(named("/bin/app (pid 1)"), "process named after its executable: {ev:?}");
        assert!(named("NVIDIA Test GPU GPU-07070707-0707-0707-0707-070707070707 PM counters"), "{ev:?}");
        assert_eq!(control.stats.rows.load(Ordering::Relaxed), 3);
    }
}
