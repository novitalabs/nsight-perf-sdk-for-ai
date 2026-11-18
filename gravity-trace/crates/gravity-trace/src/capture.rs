//! Live captures: every activity of a time window, written to trace / PM Parquet straight from the in-memory
//! snapshots of the export pipeline. There is no TSF database on disk: the merge thread, which already sees every
//! mapped activity of every snapshot, hands the ones that start inside the window to a writer thread, and the writer
//! thread does the Parquet work (sorting, encoding, zstd) so a capture never delays the metrics.
//!
//! A row and its GPU times can arrive in different snapshots, so a capture stays open until a snapshot that starts
//! after the window has been merged: by then the GPU times of the window's last activities have arrived.
use crate::error::{CaptureAbortedSnafu, Result, SpawnThreadSnafu};
use crate::mapper::{Activity, PmSample};
use crate::store::Store;
use crate::writer::{TraceWriter, write_chrome, write_pm};
use gravity_trace_core::model::ProcessInfo;
use snafu::ResultExt;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;
use tokio::runtime::Handle;
use tokio::sync::oneshot;

/// A capture request handed to the merge thread.
pub struct CaptureRequest {
    pub start_utc_ns: u64,
    pub duration_ns: u64,
    /// Directory or object store URL of the files.
    pub out: String,
    /// Also write Perfetto / Chrome trace JSON.
    pub chrome: bool,
    pub reply: oneshot::Sender<Result<CaptureReport>>,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct CaptureReport {
    pub start_utc_ns: u64,
    pub end_utc_ns: u64,
    pub trace: Option<String>,
    pub pm: Option<String>,
    pub chrome: Option<String>,
    /// Activities that start inside the window.
    pub rows: u64,
    pub pm_samples: u64,
    /// Snapshots merged while the capture was open.
    pub snapshots: u64,
    /// Activities (of every process, not only the window's) still waiting for GPU times when the capture closed.
    pub pending_at_close: usize,
    /// The pipeline stopped before a snapshot after the window arrived: the last activities may be missing.
    pub truncated: bool,
    pub processes: Vec<ProcessInfo>,
    /// Time from the close of the window's last snapshot to the files being written.
    pub write_s: f64,
}

enum ToWriter {
    Rows(Vec<Activity>),
    Pm(Vec<PmSample>),
    Close { snapshots: u64, pending: usize, processes: Vec<ProcessInfo>, truncated: bool },
}

/// The merge thread's side of a running capture.
pub struct Recorder {
    start: i64,
    end: i64,
    tx: mpsc::Sender<ToWriter>,
    rows: Vec<Activity>,
    snapshots: u64,
    /// Merged snapshots that reach past the end of the window.
    past_end: u32,
}

impl Recorder {
    /// Start the writer thread (`rt` drives the object store); on failure the request is answered with the error and
    /// `None` is returned.
    pub fn start(req: CaptureRequest, rt: &Handle) -> Option<Recorder> {
        let (start, end) = (req.start_utc_ns, req.start_utc_ns.saturating_add(req.duration_ns));
        let (tx, rx) = mpsc::channel();
        let job = WriterJob { start, end, out: req.out, chrome: req.chrome, rt: rt.clone() };
        // Shared so the request can still be answered when the thread cannot be started.
        let reply = Arc::new(Mutex::new(Some(req.reply)));
        let in_thread = reply.clone();
        let spawned = std::thread::Builder::new().name("gt-capture".into()).spawn(move || {
            // A panic is the answer to the request (the hook printed its stack), not a request left hanging.
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run(rx))).unwrap_or_else(|p| {
                crate::error::PanickedSnafu { what: format!("the capture writer panicked: {}", crate::crash::payload_message(&*p)) }.fail()
            });
            if let Some(tx) = take(&in_thread) {
                let _ = tx.send(r);
            }
        });
        match spawned.context(SpawnThreadSnafu { name: "gt-capture" }) {
            Ok(_) => Some(Recorder { start: start as i64, end: end as i64, tx, rows: Vec::new(), snapshots: 0, past_end: 0 }),
            Err(e) => {
                if let Some(tx) = take(&reply) {
                    let _ = tx.send(Err(e));
                }
                None
            }
        }
    }

    /// Keep (a copy of) `a` if it starts inside the window.
    pub fn offer(&mut self, a: &Activity) {
        if a.start_ns >= self.start && a.start_ns < self.end {
            self.rows.push(a.clone());
        }
    }

    pub fn offer_pm(&mut self, pm: &[PmSample]) {
        let keep: Vec<PmSample> =
            pm.iter().filter(|s| (s.end_utc_ns as i64) >= self.start && (s.end_utc_ns as i64) < self.end).cloned().collect();
        if !keep.is_empty() {
            let _ = self.tx.send(ToWriter::Pm(keep));
        }
    }

    /// After a merged snapshot `[window_start, window_end]`: hand its rows to the writer. True once the capture is
    /// complete (a snapshot that starts after the window, or two snapshots that reach past it, have been merged).
    pub fn snapshot_done(&mut self, window_start: u64, window_end: u64) -> bool {
        self.snapshots += 1;
        let rows = std::mem::take(&mut self.rows);
        if !rows.is_empty() {
            let _ = self.tx.send(ToWriter::Rows(rows));
        }
        if window_end as i64 >= self.end {
            self.past_end += 1;
        }
        window_start as i64 >= self.end || self.past_end >= 2
    }

    /// Let the writer thread finish the files and answer the request.
    pub fn close(self, pending: usize, processes: Vec<ProcessInfo>, truncated: bool) {
        let rows = self.rows;
        if !rows.is_empty() {
            let _ = self.tx.send(ToWriter::Rows(rows));
        }
        let _ = self.tx.send(ToWriter::Close { snapshots: self.snapshots, pending, processes, truncated });
    }
}

fn take<T>(m: &Mutex<Option<T>>) -> Option<T> {
    m.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// Answer a request that never started (another capture is running, or the pipeline stopped).
pub fn refuse(req: CaptureRequest, e: crate::error::Error) {
    let _ = req.reply.send(Err(e));
}

struct WriterJob {
    start: u64,
    end: u64,
    out: String,
    chrome: bool,
    rt: Handle,
}

impl WriterJob {
    fn run(self, rx: mpsc::Receiver<ToWriter>) -> Result<CaptureReport> {
        let meta = BTreeMap::from([
            ("source".to_string(), "nsight-perf-sdk-for-ai".to_string()),
            ("capture".to_string(), "live snapshots".to_string()),
            ("capture_start_utc_ns".to_string(), self.start.to_string()),
            ("capture_end_utc_ns".to_string(), self.end.to_string()),
        ]);
        let store = Store::open(&self.out)?;
        let mut out = TraceWriter::new(&store, &self.rt, meta);
        let mut pm = Vec::new();
        let mut close = None;
        for m in rx.iter() {
            match m {
                ToWriter::Rows(rows) => {
                    for a in &rows {
                        out.push(a)?;
                    }
                }
                ToWriter::Pm(s) => pm.extend(s),
                ToWriter::Close { snapshots, pending, processes, truncated } => {
                    close = Some((snapshots, pending, processes, truncated));
                    break;
                }
            }
        }
        // The merge thread went away without closing: the pipeline stopped.
        let (snapshots, pending_at_close, processes, truncated) = close.unwrap_or((0, 0, Vec::new(), true));
        if snapshots == 0 && out.rows() == 0 && truncated {
            return CaptureAbortedSnafu.fail();
        }
        let t0 = Instant::now();
        let rows = out.rows();
        // Named by the window: `<start>-<end>.parquet` (UTC ns).
        let window = Some((self.start as i64, self.end as i64));
        out.set_processes(processes.clone());
        let trace = out.close(window)?;
        let pm_file = write_pm(&store, &self.rt, &pm, window)?.map(|w| w.shown);
        let chrome = match (&trace, self.chrome) {
            (Some(t), true) => Some(write_chrome(&store, &self.rt, t)?.shown),
            _ => None,
        };
        Ok(CaptureReport {
            start_utc_ns: self.start,
            end_utc_ns: self.end,
            trace: trace.map(|w| w.shown),
            pm: pm_file,
            chrome,
            rows,
            pm_samples: pm.len() as u64,
            snapshots,
            pending_at_close,
            truncated,
            processes,
            write_s: t0.elapsed().as_secs_f64(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gravity_trace_core::model::EventKind;
    use gravity_trace_core::parquet_io::open_reader;

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

    #[test]
    fn keeps_the_window_and_waits_for_the_next_snapshot() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build().expect("runtime");
        let (reply, rx) = oneshot::channel();
        let out = dir.path().display().to_string();
        let req = CaptureRequest { start_utc_ns: 1_000, duration_ns: 1_000, out, chrome: true, reply };
        let mut r = Recorder::start(req, rt.handle()).expect("writer thread");
        for t in [500, 1_000, 1_500, 1_999, 2_000, 2_500] {
            r.offer(&act(t));
        }
        assert!(!r.snapshot_done(0, 1_800), "the window is still open");
        // A row of the window completed by the next snapshot (its GPU times arrived late).
        r.offer(&act(1_900));
        assert!(!r.snapshot_done(1_800, 3_000), "this snapshot started inside the window");
        assert!(r.snapshot_done(3_000, 4_000), "a snapshot after the window closes the capture");
        r.close(0, vec![ProcessInfo { pid: 1, ns_pid: 1, exe: "proc".to_string(), title: String::new() }], false);
        let rep = rx.blocking_recv().expect("reply").expect("capture");
        assert_eq!(rep.rows, 4, "1000, 1500, 1999 and the late 1900");
        assert_eq!(rep.snapshots, 3);
        assert!(!rep.truncated);
        let trace = std::path::PathBuf::from(rep.trace.expect("trace file"));
        assert_eq!(trace.file_name().and_then(|n| n.to_str()), Some("1000-2000.parquet"), "named by the window");
        assert_eq!(rep.chrome, Some(format!("{}/1000-2000.json.gz", dir.path().display())));
        let (_, meta, batches) = open_reader(&trace).expect("read back");
        assert_eq!(meta.get("capture_start_utc_ns").map(String::as_str), Some("1000"));
        let n: usize = batches.map(|b| b.expect("batch").num_rows()).sum();
        assert_eq!(n, 4);
    }

    #[test]
    fn an_empty_stopped_capture_is_aborted() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build().expect("runtime");
        let (reply, rx) = oneshot::channel();
        let req = CaptureRequest { start_utc_ns: 0, duration_ns: 1, out: dir.path().display().to_string(), chrome: false, reply };
        let r = Recorder::start(req, rt.handle()).expect("writer thread");
        drop(r); // the merge thread went away without closing
        assert!(matches!(rx.blocking_recv().expect("reply"), Err(crate::error::Error::CaptureAborted)));
    }
}
