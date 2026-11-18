//! `gt import`: a TSF database written by the SDK's `nvtrace_streaming_consumer` -> trace Parquet, PM-metric
//! Parquet and, optionally, Perfetto JSON. Live captures (`gt capture`, `POST /capture`) produce the same files
//! straight from in-memory snapshots; this path is for TSF directories that already exist.
use crate::error::{JoinSnafu, NvtraceSnafu, Result};
use crate::mapper::{Activity, MapStats, Mapper};
use crate::store::Store;
use crate::writer::{TraceWriter, write_chrome, write_pm};
use gravity_trace_core::model::ProcessInfo;
use nvtrace_sys::{self as nvt, Event, Sink};
use snafu::ResultExt;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub struct ImportOptions {
    pub start_utc_ns: u64,
    pub end_utc_ns: u64,
    pub ignore: nvt::Ignore,
    pub chrome: bool,
}

#[derive(Debug, serde::Serialize)]
pub struct ImportReport {
    pub source: String,
    pub trace: Option<String>,
    pub pm: Option<String>,
    pub chrome: Option<String>,
    pub events: u64,
    pub observation_start_utc_ns: u64,
    pub observation_end_utc_ns: u64,
    pub decode_s: f64,
    pub processes: Vec<ProcessInfo>,
    pub gpus: Vec<(String, String, String)>,
    pub stats: MapStats,
    /// Mean of every PM metric per GPU over the capture.
    pub pm_means: Vec<(String, String, f64, u64)>,
}

struct ImportSink {
    mapper: Mapper,
    out: TraceWriter,
    err: Option<crate::error::Error>,
}

impl ImportSink {
    fn push(&mut self, a: &Activity) {
        if self.err.is_none()
            && let Err(e) = self.out.push(a)
        {
            self.err = Some(e);
        }
    }
}

impl Sink for ImportSink {
    fn event(&mut self, ev: Event<'_>) -> bool {
        let mut buf = Vec::new();
        self.mapper.on_event(ev, &mut |a| buf.push(a));
        for a in &buf {
            self.push(a);
        }
        self.err.is_none()
    }
}

/// TSF database -> trace Parquet, PM metrics Parquet, optional Perfetto JSON in `out` (a directory or an object store
/// URL). The SDK reads the TSF directory itself, so that stays a local directory. Runs on a blocking thread of the
/// async runtime (the object store is driven through it).
pub fn import(dir: &Path, out: &str, o: &ImportOptions) -> Result<ImportReport> {
    let rt = tokio::runtime::Handle::current();
    let store = Store::open(out)?;
    let schema = nvt::schema().context(NvtraceSnafu)?;
    let mut mapper = Mapper::new(schema)?;
    mapper.keep_pm = true;
    let meta = BTreeMap::from([("source".to_string(), "nsight-perf-sdk-for-ai".to_string()), ("tsf_dir".to_string(), dir.display().to_string())]);
    let mut sink = ImportSink { mapper, out: TraceWriter::new(&store, &rt, meta), err: None };
    let t0 = Instant::now();
    let eo = nvt::ExportOptions { start_utc_ns: o.start_utc_ns, end_utc_ns: o.end_utc_ns, ignore: o.ignore };
    let st = nvt::export_dir(dir, &eo, &mut sink);
    if let Some(e) = sink.err.take() {
        return Err(e);
    }
    let st = st.context(NvtraceSnafu)?;
    let mut tail = Vec::new();
    sink.mapper.finish(&mut |a| tail.push(a));
    for a in &tail {
        sink.push(a);
    }
    if let Some(e) = sink.err.take() {
        return Err(e);
    }
    let ImportSink { mapper: m, mut out, .. } = sink;
    out.set_processes(m.procs.values().cloned().collect());
    let events = out.rows();
    let decode_s = t0.elapsed().as_secs_f64();
    // Named by the span of the rows: `<first start>-<last end>.parquet` (UTC ns).
    let trace = out.close(None)?;
    let pm = write_pm(&store, &rt, &m.pm, None)?.map(|w| w.shown);
    let chrome = match (&trace, o.chrome) {
        (Some(t), true) => Some(write_chrome(&store, &rt, t)?.shown),
        _ => None,
    };
    let mut means: BTreeMap<([u8; 16], Arc<str>), (f64, u64)> = BTreeMap::new();
    for s in &m.pm {
        let e = means.entry((s.gpu, s.metric.clone())).or_default();
        e.0 += s.value;
        e.1 += 1;
    }
    Ok(ImportReport {
        source: dir.display().to_string(),
        trace: trace.map(|w| w.shown),
        pm,
        chrome,
        events,
        observation_start_utc_ns: st.observation_start_utc_ns,
        observation_end_utc_ns: st.observation_end_utc_ns,
        decode_s,
        processes: m.procs.values().cloned().collect(),
        gpus: m.gpus.iter().map(|(u, (c, d))| (nvt::uuid_string(u), c.clone(), d.clone())).collect(),
        stats: m.stats.clone(),
        pm_means: means.into_iter().map(|((g, k), (s, n))| (nvt::uuid_string(&g), k.to_string(), s / n as f64, n)).collect(),
    })
}

/// `gt import`: run [`import`] on a blocking thread and print the report.
pub async fn import_cmd(dir: PathBuf, out: String, o: ImportOptions, json: bool) -> Result<()> {
    let r = tokio::task::spawn_blocking(move || import(&dir, &out, &o)).await.context(JoinSnafu)??;
    if json {
        println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
        return Ok(());
    }
    println!(
        "decoded {} in {:.1}s: {} activities over {:.3}s of observation",
        r.source,
        r.decode_s,
        r.events,
        (r.observation_end_utc_ns.saturating_sub(r.observation_start_utc_ns)) as f64 / 1e9
    );
    let s = &r.stats;
    println!(
        "  kernels {} (+{} in CUDA graphs), memcpy {}, memset {}, driver API {}, runtime API {}",
        s.kernels, s.graph_kernels, s.memcpy, s.memset, s.driver_api, s.runtime_api
    );
    println!(
        "  unnamed kernels {}, graph kernels without launch {}, completed by update {}, orphan updates {}, lost (no GPU time) {}",
        s.unnamed_kernels, s.graph_kernels_without_launch, s.completed_by_update, s.orphan_updates, s.lost_incomplete
    );
    for p in &r.processes {
        println!("  process {}", p.describe());
    }
    for (u, chip, dev) in &r.gpus {
        println!("  gpu {u} {chip} {dev}");
    }
    for (g, m, v, n) in &r.pm_means {
        println!("  pm {g} {m:<70} mean {v:>12.3} ({n} samples)");
    }
    if let Some(t) = &r.trace {
        println!("wrote {t}");
    }
    if let Some(t) = &r.pm {
        println!("wrote {t}");
    }
    if let Some(t) = &r.chrome {
        println!("wrote {t} — open in https://ui.perfetto.dev");
    }
    Ok(())
}
