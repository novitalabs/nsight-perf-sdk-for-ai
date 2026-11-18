//! The live pipeline: continuous snapshots of the nvidia-nsight-perf service (data blocks kept in memory), exported in
//! one shard per GPU, mapped to activities and merged on one thread. Every merged snapshot updates the per-kernel
//! metrics (`gt exporter`, Prometheus `nvtrace_*`), feeds the persisted trace (`gt export-daemon`, [`crate::persist`])
//! or the Flight SQL stream (`gt profile --flightsql`, [`crate::flight`]) and, while a capture is open, its Parquet
//! files (`gt capture`, `POST /capture`). Nothing goes through a TSF database on disk.
use crate::capture::{CaptureReport, CaptureRequest, Recorder, refuse};
use crate::error::{
    CaptureAbortedSnafu, CaptureBusySnafu, JoinSnafu, ListenSnafu, NvtraceSnafu, OpenFileSnafu, PanickedSnafu, Result,
    SpawnThreadSnafu,
};
use crate::feed::{self, Feed};
use crate::flight::{self, FlightOptions, FlightStats};
use crate::host;
use crate::mapper::{self, Activity, MapStats, Mapper, PmSample};
use crate::persist::{self, FlushReport, PersistOptions, PersistStats};
use crate::prom;
use gravity_trace_core::agg::Aggregator;
use gravity_trace_core::metrics::Metrics;
use gravity_trace_core::model::{EventKind, ProcessInfo};
use nvtrace_sys::{self as nvt, Event, Sink};
use snafu::ResultExt;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

/// What a scrape renders: per-process kernel metrics plus NvTrace-specific series, published at every snapshot.
#[derive(Default)]
struct Published {
    procs: Vec<Metrics>,
    pm_last: Vec<(String, String, String, f64)>, // gpu uuid, device name, metric, value (mean of the last snapshot)
    snapshots: u64,
    events: u64,
    pending: usize,
    lost: u64,
    last_window_ns: u64,
    sink_seconds_total: f64,
    stats: mapper::MapStats,
    history_truncated: u64,
    backpressured_records: u64,
    dropped_attach_records: u64,
    stall_ns: u64,
}

/// What one export shard hands to the coordinator after its part of a snapshot.
struct ShardOut {
    acts: Vec<Activity>,
    procs: Vec<ProcessInfo>,
    gpus: Vec<([u8; 16], String)>,
    pm: Vec<PmSample>,
    pending: usize,
    stats: MapStats,
    /// The snapshot manifest's flags (the same in every shard).
    history_truncated: bool,
    backpressured_records: u64,
    dropped_attach_records: u64,
    diag: Option<serde_json::Value>,
}

/// One export shard: maps the rows of its GPUs (and, for shard 0, the CPU API calls) on its own thread.
struct ShardSink {
    mapper: Mapper,
    acts: Vec<Activity>,
    t_begin: Option<Instant>,
    slot: Arc<Mutex<Option<ShardOut>>>,
}

impl Sink for ShardSink {
    fn event(&mut self, ev: Event<'_>) -> bool {
        let (begin, commit) = (matches!(ev, Event::Begin(_)), matches!(ev, Event::Commit(_)));
        if begin {
            self.t_begin = Some(Instant::now());
        }
        let acts = &mut self.acts;
        self.mapper.on_event(ev, &mut |a| acts.push(a));
        if commit {
            let m = &mut self.mapper;
            m.diag.decode_ms = self.t_begin.take().map(|t| t.elapsed().as_secs_f64() * 1e3).unwrap_or(0.0);
            let out = ShardOut {
                acts: std::mem::take(&mut self.acts),
                procs: m.procs.values().cloned().collect(),
                gpus: m.gpus.iter().map(|(u, (_, d))| (*u, d.clone())).collect(),
                pm: std::mem::take(&mut m.pm),
                pending: m.pending(),
                stats: m.stats.clone(),
                history_truncated: m.diag.history_truncated,
                backpressured_records: m.diag.backpressured_records,
                dropped_attach_records: m.diag.dropped_attach_records,
                diag: if m.diag_on { serde_json::to_value(&m.diag).ok() } else { None },
            };
            *self.slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(out);
        }
        true
    }
}

/// One snapshot's shard outputs, handed from the SDK's export thread to the merge thread.
struct MergeJob {
    window: (u64, u64),
    /// Indexed by shard; `None` for shards that did not run.
    outs: Vec<Option<ShardOut>>,
    stall_ns: u64,
    slack_ms: Option<f64>,
}

/// Runs on the SDK's export thread once every shard of a snapshot committed: only collects the shard outputs and
/// queues them, so the SDK gets its history bank back right away (it cannot switch banks, and stops reading the
/// service, until this callback returns). Merging and publishing happen on the merge thread ([`LiveSink`]).
struct DoneSink {
    slots: Vec<Arc<Mutex<Option<ShardOut>>>>,
    tx: mpsc::SyncSender<MergeJob>,
    /// Wall clock (UTC ns) when the previous snapshot's export finished, i.e. when the SDK got its history bank back.
    last_done_utc_ns: u64,
    /// Set before `Live::stop`: the final snapshot is sealed by the stop, not by a bank switch, so it cannot stall.
    stopping: Arc<AtomicBool>,
}

impl Sink for DoneSink {
    fn event(&mut self, ev: Event<'_>) -> bool {
        let Event::Done(w) = ev else { return true };
        let cutoff = w.manifest.map_or(w.end_utc_ns, |m| m.cutoff_utc_ns);
        let counted = self.last_done_utc_ns != 0 && !self.stopping.load(Ordering::Relaxed);
        let job = MergeJob {
            window: (w.start_utc_ns, w.end_utc_ns),
            outs: self.slots.iter().map(|s| s.lock().unwrap_or_else(|e| e.into_inner()).take()).collect(),
            stall_ns: if counted { stall_ns(self.last_done_utc_ns, cutoff) } else { 0 },
            // Signed margin: how long before this cutoff the previous export finished (negative = the SDK stalled).
            slack_ms: counted.then(|| (cutoff as f64 - self.last_done_utc_ns as f64) / 1e6),
        };
        // Blocks only when the merge thread is several snapshots behind (then the SDK should slow down too).
        if self.tx.send(job).is_err() {
            eprintln!("gt: the merge thread is gone; snapshot dropped");
        }
        self.last_done_utc_ns = utc_now_ns();
        true
    }
}

/// Merges the shards of every snapshot and publishes, on its own thread (fed by [`DoneSink`]).
struct LiveSink {
    /// Latest cumulative counters of every shard (a shard that did not run keeps its last values).
    last: Vec<Option<(MapStats, usize)>>,
    procs: BTreeMap<u32, ProcessInfo>,
    gpus: BTreeMap<[u8; 16], String>,
    aggs: BTreeMap<u32, Aggregator>,
    busy: BTreeMap<(u32, u32), Vec<(i64, i64)>>,
    events: u64,
    top: usize,
    out: Arc<Mutex<Published>>,
    sink_s: f64,
    window: (u64, u64),
    diag: Option<std::io::BufWriter<std::fs::File>>,
    /// Capture requests; polled before every merge (a snapshot merged after the request holds every later row).
    captures: mpsc::Receiver<CaptureRequest>,
    recorder: Option<Recorder>,
    /// Every activity goes to these writers: the persisted trace, the Flight SQL stream.
    feeds: Vec<Feed>,
    /// Drives the object store of capture files.
    rt: tokio::runtime::Handle,
    host_proc: PathBuf,
}

/// Time the SDK stopped reading the service before this snapshot's successor could start.
///
/// The SDK keeps two history banks and switches at every snapshot cutoff; the switch needs the other bank back, i.e.
/// the previous snapshot fully exported. Until then `ContinuousSnapshotExporter::ConsumeData` returns without reading
/// anything (so the manifest's backpressure counter stays 0), and trace produced meanwhile survives only while the
/// service's ring buffers hold it. The cutoff is the last read before the stall, so the stall is the part of the
/// previous export that ran past it.
fn stall_ns(prev_done_utc_ns: u64, cutoff_utc_ns: u64) -> u64 {
    if prev_done_utc_ns == 0 { 0 } else { prev_done_utc_ns.saturating_sub(cutoff_utc_ns) }
}

fn utc_now_ns() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

impl LiveSink {
    fn record(&mut self, a: &Activity) {
        self.events += 1;
        if !a.kind.is_gpu() {
            return;
        }
        let dev = a.device.unwrap_or(0);
        self.aggs.entry(a.pid).or_insert_with(|| Aggregator::new(a.start_ns)).record(a.kind, dev, &a.name, a.start_ns, a.end_ns);
        if a.kind == EventKind::Kernel {
            self.busy.entry((a.pid, dev)).or_default().push((a.start_ns, a.end_ns));
        }
    }

    fn publish(&mut self, pm: Vec<PmSample>, flags: (bool, u64, u64), stall_ns: u64) {
        for ((pid, dev), ivs) in std::mem::take(&mut self.busy) {
            if let Some(a) = self.aggs.get_mut(&pid) {
                let mut ivs = ivs;
                a.add_busy(dev, &mut ivs);
            }
        }
        let now = self.window.1 as i64;
        let host = host();
        let mut procs = Vec::new();
        for (&pid, a) in self.aggs.iter_mut() {
            a.close_window(now);
            let (kernels, devices) = a.metrics(self.top, 96);
            let p = self.procs.get(&pid).cloned().unwrap_or(ProcessInfo { pid, ns_pid: pid, ..Default::default() });
            procs.push(Metrics { pid, ns_pid: p.ns_pid, host: host.clone(), exe: p.exe, title: p.title, enabled: true, records: 0, dropped: 0, kernels, devices });
        }
        let mut means: BTreeMap<([u8; 16], Arc<str>), (f64, u64)> = BTreeMap::new();
        for s in pm {
            let e = means.entry((s.gpu, s.metric)).or_default();
            e.0 += s.value;
            e.1 += 1;
        }
        let pm_last: Vec<_> = means
            .into_iter()
            .map(|((g, m), (s, n))| (nvt::uuid_string(&g), self.gpus.get(&g).cloned().unwrap_or_default(), m.to_string(), s / n as f64))
            .collect();
        let mut stats = MapStats::default();
        let mut pending = 0;
        for (st, pe) in self.last.iter().flatten() {
            stats.add(st);
            pending += pe;
        }
        let mut p = self.out.lock().unwrap_or_else(|e| e.into_inner());
        p.procs = procs;
        if !pm_last.is_empty() {
            p.pm_last = pm_last;
        }
        p.snapshots += 1;
        p.events = self.events;
        p.pending = pending;
        p.lost = stats.lost_incomplete;
        p.last_window_ns = self.window.1.saturating_sub(self.window.0);
        p.sink_seconds_total = self.sink_s;
        p.stats = stats;
        p.history_truncated += u64::from(flags.0);
        p.backpressured_records += flags.1;
        p.dropped_attach_records += flags.2;
        p.stall_ns += stall_ns;
    }

    fn write_diag(&mut self, line: &serde_json::Value) {
        use std::io::Write as _;
        let Some(w) = self.diag.as_mut() else { return };
        if writeln!(w, "{line}").and_then(|()| w.flush()).is_err() {
            eprintln!("gt: cannot write the diagnostics file; diagnostics off");
            self.diag = None;
        }
    }
}

impl LiveSink {
    fn merge(&mut self, job: MergeJob) {
        let t0 = Instant::now();
        self.window = job.window;
        self.take_captures();
        let mut pm = Vec::new();
        let mut diags = Vec::new();
        let mut flags = None;
        for (i, o) in job.outs.into_iter().enumerate() {
            let Some(o) = o else { continue };
            // Names first: the writers name tracks when their first rows arrive.
            let gpus = o.gpus.into_iter().filter(|(u, d)| self.gpus.get(u) != Some(d)).collect();
            let names = feed::Names { procs: self.new_processes(o.procs), gpus };
            self.send_names(names);
            for a in o.acts {
                self.record(&a);
                if let Some(r) = self.recorder.as_mut() {
                    r.offer(&a);
                }
                feed::offer(&mut self.feeds, a);
            }
            if let Some(r) = self.recorder.as_mut() {
                r.offer_pm(&o.pm);
            }
            for f in &mut self.feeds {
                f.push_pm(&o.pm);
            }
            pm.extend(o.pm);
            flags.get_or_insert((o.history_truncated, o.backpressured_records, o.dropped_attach_records));
            if let Some(d) = o.diag {
                diags.push(d);
            }
            if let Some(l) = self.last.get_mut(i) {
                *l = Some((o.stats, o.pending));
            }
        }
        // Titles change while a process runs (vLLM names a worker `VLLM::Worker`, then `VLLM::Worker_TP0`).
        let retitled = self.retitle();
        self.send_names(feed::Names { procs: retitled, gpus: Vec::new() });
        self.publish(pm, flags.unwrap_or_default(), job.stall_ns);
        for f in &mut self.feeds {
            f.snapshot_done();
        }
        if self.recorder.as_mut().is_some_and(|r| r.snapshot_done(job.window.0, job.window.1)) {
            self.close_capture(false);
        }
        if !diags.is_empty() {
            let mut line = merge_diag(diags);
            if let Some(o) = line.as_object_mut() {
                o.insert("merge_ms".into(), serde_json::json!(t0.elapsed().as_secs_f64() * 1e3));
                o.insert("stall_ms".into(), serde_json::json!(job.stall_ns as f64 / 1e6));
                o.insert("slack_ms".into(), serde_json::json!(job.slack_ms));
            }
            self.write_diag(&line);
        }
        self.sink_s += t0.elapsed().as_secs_f64();
    }

    /// The processes of a shard that are new or changed, with their titles (kept from before, else read now).
    fn new_processes(&self, procs: Vec<ProcessInfo>) -> Vec<ProcessInfo> {
        let mut out = Vec::new();
        for mut p in procs {
            match self.procs.get(&p.pid) {
                Some(old) => p.title.clone_from(&old.title),
                None => p.title = crate::proctitle::title(&self.host_proc, p.pid, p.ns_pid).unwrap_or_default(),
            }
            if self.procs.get(&p.pid) != Some(&p) {
                out.push(p);
            }
        }
        out
    }

    /// Re-read the titles of the known processes; the ones that changed (a process that is gone keeps its title).
    fn retitle(&mut self) -> Vec<ProcessInfo> {
        let mut changed = Vec::new();
        for p in self.procs.values_mut() {
            if let Some(t) = crate::proctitle::title(&self.host_proc, p.pid, p.ns_pid)
                && t != p.title
            {
                p.title = t;
                changed.push(p.clone());
            }
        }
        changed
    }

    fn send_names(&mut self, names: feed::Names) {
        for f in &mut self.feeds {
            f.names(&names);
        }
        self.procs.extend(names.procs.into_iter().map(|p| (p.pid, p)));
        self.gpus.extend(names.gpus);
    }

    /// Start the capture requested since the last merge (one at a time).
    fn take_captures(&mut self) {
        while let Ok(req) = self.captures.try_recv() {
            if self.recorder.is_some() {
                refuse(req, CaptureBusySnafu.build());
            } else {
                self.recorder = Recorder::start(req, &self.rt);
            }
        }
    }

    fn close_capture(&mut self, truncated: bool) {
        let Some(r) = self.recorder.take() else { return };
        let pending = self.last.iter().flatten().map(|(_, p)| p).sum();
        r.close(pending, self.procs.values().cloned().collect(), truncated);
    }

    /// The pipeline stopped: close the open capture, finish the writers (persisted file, Flight SQL stream), refuse
    /// the captures that never started.
    fn finish(&mut self) {
        self.close_capture(true);
        for f in self.feeds.drain(..) {
            f.finish();
        }
        while let Ok(req) = self.captures.try_recv() {
            refuse(req, CaptureAbortedSnafu.build());
        }
    }
}

/// One diagnostics line per snapshot from the shards' lines: counters are summed (per name for the table / update
/// lists, per bucket for histograms), the manifest comes from shard 0, timings are the slowest shard's.
fn merge_diag(shards: Vec<serde_json::Value>) -> serde_json::Value {
    use serde_json::Value as J;
    const FIRST: &[&str] = &[
        "snapshot",
        "window_start_utc_ns",
        "window_end_utc_ns",
        "requested_begin_utc_ns",
        "cutoff_utc_ns",
        "actual_begin_utc_ns",
        "actual_end_utc_ns",
        "backpressured_records",
        "history_truncated",
        "attach_complete",
        "dropped_attach_records",
        "data_blocks",
        "queue_ms",
    ];
    const MAX: &[&str] = &["build_ms", "decode_ms", "lag_ms"];
    let n = shards.len();
    let mut it = shards.into_iter();
    let Some(mut acc) = it.next() else { return J::Null };
    let mut builds = vec![acc.get("build_ms").cloned().unwrap_or(J::Null)];
    for d in it {
        builds.push(d.get("build_ms").cloned().unwrap_or(J::Null));
        let (Some(a), J::Object(b)) = (acc.as_object_mut(), d) else { continue };
        for (k, v) in b {
            if FIRST.contains(&k.as_str()) {
                continue;
            }
            let Some(cur) = a.get_mut(&k) else {
                a.insert(k, v);
                continue;
            };
            if MAX.contains(&k.as_str()) {
                if v.as_f64().unwrap_or(0.0) > cur.as_f64().unwrap_or(0.0) {
                    *cur = v;
                }
                continue;
            }
            match (cur, v) {
                (J::Number(x), J::Number(y)) => {
                    *x = match (x.as_u64(), y.as_u64()) {
                        (Some(p), Some(q)) => (p + q).into(),
                        _ => serde_json::Number::from_f64(x.as_f64().unwrap_or(0.0) + y.as_f64().unwrap_or(0.0)).unwrap_or_else(|| 0.into()),
                    };
                }
                (J::Array(xs), J::Array(ys)) if k == "graph_short_top" => {
                    xs.extend(ys);
                    let short = |e: &J| {
                        let g = |i: usize| e.get(i).and_then(J::as_u64).unwrap_or(0);
                        (g(2) * g(3)).saturating_sub(g(4))
                    };
                    xs.sort_by_key(|e| std::cmp::Reverse(short(e)));
                    xs.truncate(8);
                }
                (J::Array(xs), J::Array(ys)) if xs.first().is_some_and(J::is_array) || ys.first().is_some_and(J::is_array) => {
                    // [[name, count], ...]
                    for y in ys {
                        let (Some(name), Some(c)) = (y.get(0).cloned(), y.get(1).and_then(J::as_u64)) else { continue };
                        match xs.iter_mut().find(|x| x.get(0) == Some(&name)) {
                            Some(x) => {
                                if let Some(slot) = x.get_mut(1) {
                                    *slot = (slot.as_u64().unwrap_or(0) + c).into();
                                }
                            }
                            None => xs.push(y),
                        }
                    }
                }
                (J::Array(xs), J::Array(ys)) => {
                    for (x, y) in xs.iter_mut().zip(ys) {
                        if let (Some(p), Some(q)) = (x.as_u64(), y.as_u64()) {
                            *x = (p + q).into();
                        }
                    }
                }
                (J::Bool(x), J::Bool(y)) => *x |= y,
                _ => {}
            }
        }
    }
    if let Some(a) = acc.as_object_mut() {
        a.insert("shards".into(), n.into());
        a.insert("shard_build_ms".into(), J::Array(builds));
    }
    acc
}

/// Shard sinks created for `--shards 0` (one shard per GPU partition, at most this many).
const MAX_AUTO_SHARDS: usize = 8;

pub struct PipelineConfig {
    pub top: usize,
    pub live: nvt::LiveOptions,
    pub diag: Option<PathBuf>,
    /// The host's procfs, for process titles ([`crate::proctitle`]).
    pub host_proc: PathBuf,
    /// Write every activity to disk ([`crate::persist`]).
    pub persist: Option<PersistOptions>,
    /// Stream every activity to a Flight SQL server ([`crate::flight`]).
    pub flight: Option<FlightOptions>,
}

/// [`Handle::totals`].
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Totals {
    /// Activities merged (and handed to the stream / the persisted trace).
    pub activities: u64,
    pub snapshots: u64,
    /// Activities whose GPU times never arrived (nvtrace_lost_activities_total).
    pub lost_activities: u64,
    /// Time the SDK stopped reading the service while a snapshot was still exported (nvtrace_snapshot_stall_seconds_total).
    pub snapshot_stall_seconds: f64,
    pub history_truncated: u64,
}

/// What the HTTP handlers and `gt capture` use while the pipeline runs.
#[derive(Clone)]
pub struct Handle {
    out: Arc<Mutex<Published>>,
    live: Arc<Mutex<Option<nvt::Live>>>,
    captures: mpsc::Sender<CaptureRequest>,
    persist: Option<persist::Control>,
    flight: Option<Arc<FlightStats>>,
}

/// A running live pipeline (SDK snapshots, export shards, merge thread, disk writer / Flight SQL sender).
pub struct Pipeline {
    handle: Handle,
    merger: std::thread::JoinHandle<()>,
    /// The writer threads behind the feeds, by name.
    writers: Vec<(&'static str, std::thread::JoinHandle<()>)>,
    stopping: Arc<AtomicBool>,
}

impl Handle {
    /// Record every activity that starts in the next `duration` into trace / PM Parquet (+ Perfetto JSON) in `out` (a
    /// directory or an object store URL). Returns once a snapshot after the window has been merged and the files are
    /// written.
    pub async fn capture(&self, duration: std::time::Duration, out: String, chrome: bool) -> Result<CaptureReport> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let req = CaptureRequest { start_utc_ns: utc_now_ns(), duration_ns: duration.as_nanos() as u64, out, chrome, reply };
        self.captures.send(req).map_err(|_| CaptureAbortedSnafu.build())?;
        rx.await.map_err(|_| CaptureAbortedSnafu.build())?
    }

    /// Prometheus exposition of the latest published snapshot.
    pub fn metrics(&self, pod: &str) -> String {
        let stats = self.live.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|l| l.stats()).unwrap_or_default();
        let p = self.out.lock().unwrap_or_else(|e| e.into_inner());
        let mut o = render(pod, &p, &stats);
        if let Some(c) = &self.persist {
            persist::render(&mut o, &prom::esc(pod), &c.stats);
        }
        if let Some(s) = &self.flight {
            flight::render(&mut o, &prom::esc(pod), s);
        }
        o
    }

    /// What the pipeline merged and lost so far.
    pub fn totals(&self) -> Totals {
        let p = self.out.lock().unwrap_or_else(|e| e.into_inner());
        Totals {
            activities: p.events,
            snapshots: p.snapshots,
            lost_activities: p.lost,
            snapshot_stall_seconds: p.stall_ns as f64 / 1e9,
            history_truncated: p.history_truncated,
        }
    }

    pub fn persist_stats(&self) -> Option<Arc<PersistStats>> {
        self.persist.as_ref().map(|c| c.stats.clone())
    }

    pub fn flight_stats(&self) -> Option<Arc<FlightStats>> {
        self.flight.clone()
    }

    /// Close the open persisted file now, so everything merged so far can be dumped.
    pub async fn flush_persist(&self) -> Result<FlushReport> {
        persist::flush(self.persist.as_ref()).await
    }
}

/// Connect to the service and start taking snapshots.
pub async fn start(cfg: PipelineConfig) -> Result<Pipeline> {
    let schema = nvt::schema().context(NvtraceSnafu)?;
    let diag = match &cfg.diag {
        Some(p) => Some(std::io::BufWriter::new(
            std::fs::OpenOptions::new().create(true).append(true).open(p).context(OpenFileSnafu { path: p.clone() })?,
        )),
        None => None,
    };
    // One sink per possible shard; with --shards 0 the shim uses one per GPU partition, up to this many.
    let n_shards = if cfg.live.shards == 0 { MAX_AUTO_SHARDS } else { cfg.live.shards as usize };
    let mut slots = Vec::with_capacity(n_shards);
    let mut shards: Vec<Box<dyn Sink + Send>> = Vec::with_capacity(n_shards);
    for _ in 0..n_shards {
        let mut mapper = Mapper::new(schema)?;
        mapper.keep_pm = true;
        mapper.diag_on = cfg.diag.is_some();
        let slot = Arc::new(Mutex::new(None));
        slots.push(slot.clone());
        shards.push(Box::new(ShardSink { mapper, acts: Vec::new(), t_begin: None, slot }));
    }
    let out = Arc::new(Mutex::new(Published::default()));
    let (cap_tx, cap_rx) = mpsc::channel();
    let rt = tokio::runtime::Handle::current();
    let (mut feeds, mut writers) = (Vec::new(), Vec::new());
    // The Flight SQL server first: when it cannot be reached, nothing has started yet.
    let flight_stats = match cfg.flight.clone() {
        Some(o) => {
            let (f, s, t) = flight::start(o).await?;
            feeds.push(f);
            writers.push(("flightsql", t));
            Some(s)
        }
        None => None,
    };
    let control = match cfg.persist.clone() {
        Some(o) => {
            let (f, c, t) = persist::start(o, &rt)?;
            feeds.push(f);
            writers.push(("persist", t));
            Some(c)
        }
        None => None,
    };
    let mut sink = LiveSink {
        last: (0..n_shards).map(|_| None).collect(),
        procs: BTreeMap::new(),
        gpus: BTreeMap::new(),
        aggs: BTreeMap::new(),
        busy: BTreeMap::new(),
        events: 0,
        top: cfg.top,
        out: out.clone(),
        sink_s: 0.0,
        window: (0, 0),
        diag,
        captures: cap_rx,
        recorder: None,
        feeds,
        rt,
        host_proc: cfg.host_proc.clone(),
    };
    let (tx, rx) = mpsc::sync_channel::<MergeJob>(4);
    let merger = std::thread::Builder::new()
        .name("gt-merge".into())
        .spawn(move || {
            for job in rx {
                sink.merge(job);
            }
            sink.finish();
        })
        .context(SpawnThreadSnafu { name: "gt-merge" })?;
    let stopping = Arc::new(AtomicBool::new(false));
    let done = DoneSink { slots, tx, last_done_utc_ns: 0, stopping: stopping.clone() };
    let live_opts = cfg.live.clone();
    let live = match tokio::task::spawn_blocking(move || nvt::Live::start(&live_opts, shards, Box::new(done))).await.context(JoinSnafu)? {
        Ok(l) => l,
        Err(e) => {
            // The Done sink went away with the failed start: the merge thread ends and finishes the writers.
            let _ = tokio::task::spawn_blocking(move || {
                let _ = merger.join();
                for (_, t) in writers {
                    let _ = t.join();
                }
            })
            .await;
            return Err(e).context(NvtraceSnafu);
        }
    };
    let in_memory = live.stats().in_memory;
    if cfg.live.in_memory && !in_memory {
        eprintln!("gt: in-memory snapshot storage unavailable (the SDK patch did not apply), using files");
    }
    eprintln!(
        "gt live capture started: snapshot every {} ms, data blocks {}",
        cfg.live.snapshot_ms,
        if in_memory { "in memory".to_string() } else { format!("in {}", cfg.live.work_dir.display()) }
    );
    if let Some(p) = &cfg.persist {
        eprintln!(
            "gt: persisting every activity to {} as {} (new file every {} s, write buffer {} MiB = {:.1} M activities)",
            p.location,
            p.format.as_str(),
            p.rotate.as_secs(),
            p.max_buffer_bytes >> 20,
            (p.max_buffer_bytes / feed::ROW_BYTES) as f64 / 1e6
        );
    }
    if let Some(f) = &cfg.flight {
        eprintln!(
            "gt: streaming every activity to {} (Flight SQL ingest into {}{}; dictionaries {}, compression {:?}, send buffer {} MiB)",
            f.endpoint,
            f.table,
            if f.pm_table.is_empty() { String::new() } else { format!(", PM samples into {}", f.pm_table) },
            if f.dictionaries { "kept" } else { "as strings" },
            f.compression,
            f.max_buffer_bytes >> 20
        );
    }
    let handle = Handle { out, live: Arc::new(Mutex::new(Some(live))), captures: cap_tx, persist: control, flight: flight_stats };
    Ok(Pipeline { handle, merger, writers, stopping })
}

impl Pipeline {
    pub fn handle(&self) -> Handle {
        self.handle.clone()
    }

    /// Stop the SDK capture and wait for the merge thread (which closes an open capture as truncated) and the writers
    /// (the disk writer closes its file, the Flight SQL sender sends what is left).
    pub async fn stop(self) -> Result<()> {
        let Pipeline { handle, merger, writers, stopping } = self;
        stopping.store(true, Ordering::Relaxed);
        let taken = handle.live.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(l) = taken {
            // Dropping the capture after the stop frees the Done sink, which closes the channel and ends the merge thread.
            tokio::task::spawn_blocking(move || l.stop()).await.context(JoinSnafu)?.context(NvtraceSnafu)?;
        }
        drop(handle);
        // Every thread is joined (the writers still finish their files / send what they have), then the first panic
        // is the error.
        let mut panicked = None;
        if let Err(p) = tokio::task::spawn_blocking(move || merger.join()).await.context(JoinSnafu)? {
            panicked = Some(format!("the merge thread panicked: {}", crate::crash::payload_message(&*p)));
        }
        for (name, t) in writers {
            if let Err(p) = tokio::task::spawn_blocking(move || t.join()).await.context(JoinSnafu)? {
                panicked.get_or_insert_with(|| format!("the {name} thread panicked: {}", crate::crash::payload_message(&*p)));
            }
        }
        match panicked {
            Some(what) => PanickedSnafu { what }.fail(),
            None => Ok(()),
        }
    }
}

pub struct ExporterConfig {
    pub listen: String,
    pub pipeline: PipelineConfig,
    /// Where `POST /capture` writes its files (a directory or an object store URL).
    pub captures_dir: String,
}

#[derive(serde::Deserialize)]
struct CaptureQuery {
    /// e.g. `5s`, `500ms` (default 5s).
    duration: Option<String>,
    /// Also write Perfetto JSON (default true).
    chrome: Option<bool>,
}

/// The HTTP API of `gt exporter` and `gt export-daemon`: `GET /metrics`, `POST /capture?duration=5s`,
/// `POST /persist/flush`.
pub fn router(h: Handle, captures_dir: String) -> axum::Router {
    use axum::Router;
    use axum::extract::{Query, State};
    use axum::http::{StatusCode, header};
    use axum::response::IntoResponse;
    use axum::routing::{get, post};

    #[derive(Clone)]
    struct Ctx {
        h: Handle,
        pod: String,
        captures_dir: String,
    }
    async fn metrics(State(c): State<Ctx>) -> impl IntoResponse {
        ([(header::CONTENT_TYPE, prom::CONTENT_TYPE)], c.h.metrics(&c.pod))
    }
    async fn capture(State(c): State<Ctx>, Query(q): Query<CaptureQuery>) -> axum::response::Response {
        let duration = match crate::parse_duration(q.duration.as_deref().unwrap_or("5s")) {
            Ok(d) => d,
            Err(e) => return (StatusCode::BAD_REQUEST, format!("{e}\n")).into_response(),
        };
        match c.h.capture(duration, c.captures_dir.clone(), q.chrome.unwrap_or(true)).await {
            Ok(r) => axum::Json(r).into_response(),
            Err(e @ crate::error::Error::CaptureBusy) => (StatusCode::CONFLICT, format!("{e}\n")).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}\n")).into_response(),
        }
    }
    async fn flush(State(c): State<Ctx>) -> axum::response::Response {
        match c.h.flush_persist().await {
            Ok(r) => axum::Json(r).into_response(),
            Err(e @ crate::error::Error::PersistOff) => (StatusCode::NOT_FOUND, format!("{e}\n")).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}\n")).into_response(),
        }
    }
    let ctx = Ctx { h, pod: prom::pod_name(), captures_dir };
    Router::new()
        .route("/metrics", get(metrics))
        .route("/capture", post(capture))
        .route("/persist/flush", post(flush))
        .route("/healthz", get(|| async { "ok\n" }))
        .route("/", get(|| async { "gt: GET /metrics, POST /capture?duration=5s, POST /persist/flush\n" }))
        .with_state(ctx)
}

/// `gt exporter`: the pipeline plus the HTTP API ([`router`]) until Ctrl-C.
pub async fn exporter(cfg: ExporterConfig) -> Result<()> {
    let pipeline = start(cfg.pipeline).await?;
    let app = router(pipeline.handle(), cfg.captures_dir);
    let listener = tokio::net::TcpListener::bind(&cfg.listen).await.context(ListenSnafu { addr: cfg.listen.clone() })?;
    eprintln!("gt exporter listening on http://{}/metrics", cfg.listen);
    let served = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context(ListenSnafu { addr: cfg.listen });
    pipeline.stop().await?;
    served
}

/// `gt capture`: start the pipeline, record one window, stop.
pub async fn capture_cmd(cfg: PipelineConfig, duration: std::time::Duration, out: String, chrome: bool, json: bool) -> Result<()> {
    let pipeline = start(cfg).await?;
    let r = pipeline.handle().capture(duration, out, chrome).await;
    pipeline.stop().await?;
    let r = r?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
        return Ok(());
    }
    println!(
        "captured {:.3}s: {} activities, {} PM samples over {} snapshots{}",
        (r.end_utc_ns - r.start_utc_ns) as f64 / 1e9,
        r.rows,
        r.pm_samples,
        r.snapshots,
        if r.truncated { " (truncated: stopped before the window was complete)" } else { "" }
    );
    for p in &r.processes {
        println!("  process {}", p.describe());
    }
    for f in [&r.trace, &r.pm].into_iter().flatten() {
        println!("wrote {f}");
    }
    if let Some(f) = &r.chrome {
        println!("wrote {f} — open in https://ui.perfetto.dev");
    }
    Ok(())
}

fn render(pod: &str, p: &Published, s: &nvt::LiveStats) -> String {
    use std::fmt::Write as _;
    let mut o = prom::render("nvtrace", pod, &p.procs, 0);
    let pod = prom::esc(pod);
    let mut g = |name: &str, kind: &str, help: &str, v: String| {
        let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name}{{pod=\"{pod}\"}} {v}");
    };
    g("nvtrace_snapshots_total", "counter", "Continuous snapshots exported.", s.snapshots_ok.to_string());
    g("nvtrace_snapshots_failed_total", "counter", "Snapshots whose export failed.", s.snapshots_failed.to_string());
    g("nvtrace_rows_total", "counter", "Table rows decoded from snapshots.", s.rows.to_string());
    g("nvtrace_snapshot_export_seconds", "gauge", "Decode + row mapping time of the last snapshot.", format!("{:.6}", s.last_export_ns as f64 / 1e9));
    g("nvtrace_snapshot_export_seconds_total", "counter", "Decode + row mapping time of all snapshots.", format!("{:.6}", s.total_export_ns as f64 / 1e9));
    g("nvtrace_activities_total", "counter", "Activities (kernels, copies, API calls) mapped.", p.events.to_string());
    g("nvtrace_pending_activities", "gauge", "Activities waiting for GPU times from a later snapshot.", p.pending.to_string());
    g("nvtrace_lost_activities_total", "counter", "Activities whose GPU times never arrived.", p.lost.to_string());
    g("nvtrace_lost_kernels_total", "counter", "Eager kernels whose GPU times never arrived.", p.stats.lost_kernels.to_string());
    g("nvtrace_lost_graph_kernels_total", "counter", "Graph node kernels whose GPU end never arrived.", p.stats.lost_graph_kernels.to_string());
    g("nvtrace_lost_gpu_commands_total", "counter", "Memcpy / memset commands whose GPU times never arrived.", p.stats.lost_gpu_commands.to_string());
    g("nvtrace_orphan_updates_total", "counter", "Deferred GPU-time updates that matched no pending row.", p.stats.orphan_updates.to_string());
    g("nvtrace_completed_by_update_total", "counter", "Rows completed by a later snapshot's update.", p.stats.completed_by_update.to_string());
    g(
        "nvtrace_snapshot_history_truncated_total",
        "counter",
        "Snapshots cut before the snapshot interval because a history bank filled (not data loss by itself).",
        p.history_truncated.to_string(),
    );
    g("nvtrace_snapshot_backpressured_records_total", "counter", "Records met while the SDK's latency buffer was full.", p.backpressured_records.to_string());
    g(
        "nvtrace_snapshot_stall_seconds_total",
        "counter",
        "Time the SDK stopped reading the service because the previous snapshot was still being exported; trace produced \
         meanwhile is lost once the service's ring buffers wrap.",
        format!("{:.3}", p.stall_ns as f64 / 1e9),
    );
    g("nvtrace_snapshot_dropped_attach_records_total", "counter", "Attachment records the SDK dropped.", p.dropped_attach_records.to_string());
    g("nvtrace_snapshot_window_seconds", "gauge", "Observation window of the last snapshot.", format!("{:.3}", p.last_window_ns as f64 / 1e9));
    g("nvtrace_snapshot_storage_in_memory", "gauge", "1: snapshot data blocks kept in memory, 0: in work-dir files.", u8::from(s.in_memory).to_string());
    let _ = writeln!(o, "# HELP nvtrace_gpu_metric GPU PM metric, mean over the last snapshot.\n# TYPE nvtrace_gpu_metric gauge");
    for (gpu, dev, m, v) in &p.pm_last {
        let _ = writeln!(
            o,
            "nvtrace_gpu_metric{{pod=\"{pod}\",gpu=\"{gpu}\",device_name=\"{}\",metric=\"{}\"}} {v}",
            prom::esc(dev),
            prom::esc(m)
        );
    }
    o
}

#[cfg(test)]
mod stall_tests {
    use super::stall_ns;

    #[test]
    fn stall_is_the_previous_export_past_the_cutoff() {
        assert_eq!(stall_ns(0, 5_000), 0, "no previous snapshot");
        assert_eq!(stall_ns(4_000, 5_000), 0, "previous export finished before the cutoff");
        assert_eq!(stall_ns(6_500, 5_000), 1_500);
    }
}
