//! The merge thread's hand-off to a writer thread (the persisted trace [`crate::persist`], the Flight SQL stream
//! [`crate::flight`]): activities in chunks through a queue bounded in bytes. When the writer falls behind, whole
//! chunks are dropped and counted instead of slowing the merge thread (and with it the SDK's snapshots) down.
use crate::error::Result;
use crate::mapper::{Activity, PmSample};
use crate::persist::FlushReport;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use tokio::sync::oneshot;

/// Activities per chunk handed to the writer.
pub const CHUNK_ROWS: usize = 64 * 1024;
/// Queue accounting per activity (the struct; names are shared).
pub const ROW_BYTES: usize = std::mem::size_of::<Activity>();

/// The queue's counters, shared by the merge thread and the writer.
#[derive(Debug, Default)]
pub struct Queue {
    /// Activities waiting for the writer (approximate bytes).
    pub bytes: AtomicUsize,
    /// Activities given up: the queue was full, or the writer could not write them.
    pub dropped_rows: AtomicU64,
}

impl Queue {
    /// The writer took `rows` activities off the queue.
    pub fn taken(&self, rows: usize) {
        self.bytes.fetch_sub(rows * ROW_BYTES, Ordering::Relaxed);
    }

    pub fn dropped(&self, rows: u64) {
        self.dropped_rows.fetch_add(rows, Ordering::Relaxed);
    }
}

/// Processes (pids and executables) and device names of GPUs (by UUID), as the merge thread learns them.
#[derive(Clone, Debug, Default)]
pub struct Names {
    pub procs: Vec<gravity_trace_core::model::ProcessInfo>,
    pub gpus: Vec<([u8; 16], String)>,
}

pub enum Msg {
    Rows(Vec<Activity>),
    Pm(Vec<PmSample>),
    /// New or changed names, before the rows that use them (the Perfetto writer names its tracks).
    Names(Names),
    /// A snapshot has been merged.
    Snapshot,
    /// Close the open file now (persisted trace only).
    Flush(oneshot::Sender<Result<FlushReport>>),
    Close,
}

/// The merge thread's side.
pub struct Feed {
    tx: mpsc::Sender<Msg>,
    chunk: Vec<Activity>,
    queue: Arc<Queue>,
    max_bytes: usize,
    pm: bool,
}

impl Feed {
    /// `pm`: also hand over PM samples.
    pub fn new(tx: mpsc::Sender<Msg>, queue: Arc<Queue>, max_bytes: usize, pm: bool) -> Feed {
        Feed { tx, chunk: Vec::with_capacity(CHUNK_ROWS), queue, max_bytes, pm }
    }

    pub fn push(&mut self, a: Activity) {
        self.chunk.push(a);
        if self.chunk.len() >= CHUNK_ROWS {
            self.send_chunk();
        }
    }

    fn send_chunk(&mut self) {
        if self.chunk.is_empty() {
            return;
        }
        let rows = std::mem::replace(&mut self.chunk, Vec::with_capacity(CHUNK_ROWS));
        let (n, bytes) = (rows.len() as u64, rows.len() * ROW_BYTES);
        if self.queue.bytes.load(Ordering::Relaxed) + bytes > self.max_bytes {
            self.queue.dropped(n);
            return;
        }
        self.queue.bytes.fetch_add(bytes, Ordering::Relaxed);
        if self.tx.send(Msg::Rows(rows)).is_err() {
            self.queue.bytes.fetch_sub(bytes, Ordering::Relaxed);
            self.queue.dropped(n);
        }
    }

    pub fn push_pm(&mut self, pm: &[PmSample]) {
        if self.pm && !pm.is_empty() {
            let _ = self.tx.send(Msg::Pm(pm.to_vec()));
        }
    }

    pub fn names(&mut self, n: &Names) {
        if !n.procs.is_empty() || !n.gpus.is_empty() {
            let _ = self.tx.send(Msg::Names(n.clone()));
        }
    }

    pub fn snapshot_done(&mut self) {
        self.send_chunk();
        let _ = self.tx.send(Msg::Snapshot);
    }

    /// The pipeline stopped: the writer finishes and exits.
    pub fn finish(mut self) {
        self.send_chunk();
        let _ = self.tx.send(Msg::Close);
    }
}

/// Hand `a` to every feed (cloned for all but the last).
pub fn offer(feeds: &mut [Feed], a: Activity) {
    if let Some((last, rest)) = feeds.split_last_mut() {
        for f in rest {
            f.push(a.clone());
        }
        last.push(a);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_documented_activity_size() {
        // README, config.example.toml and config.rs size the buffers with it.
        assert_eq!(super::ROW_BYTES, 168, "update the documented 168 bytes per activity");
    }
}
