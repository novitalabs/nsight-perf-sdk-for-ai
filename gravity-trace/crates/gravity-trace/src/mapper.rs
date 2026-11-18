//! NvTrace rows -> gravity-trace activities.
//!
//! The NVIDIA Nsight Perf SDK for AI exports a capture as relational tables (`NvTraceTableSchema.h`). This
//! module joins them back into a flat activity model ([`gravity_trace_core::model::TraceRow`]), so the trace Parquet,
//! Perfetto and Prometheus output does not depend on the SDK's table layout:
//!
//! | kind     | NvTrace source                                                                          |
//! |----------|-----------------------------------------------------------------------------------------|
//! | kernel   | `cuda_immediate_kernel_launches` (eager) and `cuda_graph_node_launches` (CUDA graph)    |
//! |          |   name: `cuda_functions.name_mangled`, graph stream / correlation: the graph launch     |
//! | memcpy   | `cuda_gpu_commands` with a `cuda_gpu_memcpy_details` row                               |
//! | memset   | `cuda_gpu_commands` with a `cuda_gpu_memset_details` row                               |
//! | driver   | `cuda_driver_api_calls` (thread: `thread_event_threads.thread_id_native`)               |
//! | runtime  | `cuda_runtime_api_calls` (only with the service's `traceRuntimeApiCalls: true`)         |
//!
//! Correlation ids are the driver API correlation ids (truncated to u32 like CUPTI's), so CPU->GPU flows link
//! `cuLaunchKernel` / `cuGraphLaunch` to their kernels. GPU timestamps are UTC ns, the same clock as CUPTI's.
//!
//! In live mode a row and its GPU times can land in different snapshots, in either order: the launch record comes
//! from the process (CUDA API trace), the GPU times from the GPU hardware trace, and the two streams reach the
//! service with different delays. GPU times that come after their row arrive as a table *update*; rows wait in a
//! pending set for them. GPU times that come before their row (the SDK found no launch for them in that snapshot)
//! arrive as an update too and wait in an *early* set for the row. Both sets are dropped after
//! [`PENDING_SNAPSHOTS`] snapshots: rows as lost, updates as orphans.
use crate::error::{NvtraceSchemaSnafu, Result};
use gravity_trace_core::model::{EventKind, ProcessInfo, TraceRow};
use nvtrace_sys::{Event, Schema, Table, Update, Value};
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Snapshots an incomplete row waits for its update before it is counted as lost.
pub const PENDING_SNAPSHOTS: u64 = 3;
/// Graph launches kept per executable across snapshots (to place node launches whose launch came earlier).
const TIMELINE_KEEP: usize = 4096;

type Uuid = [u8; 16];

/// Owned activity (a `TraceRow` with owned name).
#[derive(Clone, Debug)]
pub struct Activity {
    pub kind: EventKind,
    pub name: Arc<str>,
    pub start_ns: i64,
    pub end_ns: i64,
    /// Host pid and the pid in the process's own namespace ([`ProcessInfo`]).
    pub pid: u32,
    pub ns_pid: u32,
    pub tid: Option<u64>,
    pub device: Option<u32>,
    pub context: Option<u32>,
    pub stream: Option<u32>,
    pub correlation: u32,
    pub graph_id: Option<u32>,
    pub graph_node_id: Option<u64>,
    pub grid: Option<[i32; 3]>,
    pub block: Option<[i32; 3]>,
    pub shmem: Option<i32>,
    pub bytes: Option<u64>,
}

impl Activity {
    pub fn row(&self) -> TraceRow<'_> {
        TraceRow {
            kind: Some(self.kind),
            name: &self.name,
            start_ns: self.start_ns,
            end_ns: self.end_ns,
            pid: self.pid,
            ns_pid: Some(self.ns_pid),
            tid: self.tid,
            device_id: self.device,
            context_id: self.context,
            stream_id: self.stream,
            correlation_id: self.correlation,
            graph_id: self.graph_id,
            graph_node_id: self.graph_node_id,
            grid: self.grid,
            block: self.block,
            dynamic_shared_mem: self.shmem,
            bytes: self.bytes,
            device_launched: self.kind.is_gpu().then_some(false),
            ..Default::default()
        }
    }
}

/// Column indexes of one table, resolved by name.
struct Cols<const N: usize> {
    table: usize,
    idx: [usize; N],
}

impl<const N: usize> Cols<N> {
    fn new(schema: &Schema, table: &str, names: [&str; N]) -> Result<Self> {
        let (ti, t) = schema
            .table(table)
            .ok_or_else(|| NvtraceSchemaSnafu { table: table.to_string(), column: String::new() }.build())?;
        let mut idx = [0usize; N];
        for (i, n) in names.iter().enumerate() {
            idx[i] = col(t, n)?;
        }
        Ok(Self { table: ti, idx })
    }
}

fn col(t: &Table, name: &str) -> Result<usize> {
    t.col(name).ok_or_else(|| NvtraceSchemaSnafu { table: t.name.clone(), column: name.to_string() }.build())
}

/// Positions inside an update's value list: (process_id_unique, gpu_uuid, key, [latest bound], update values...).
struct UpdCols {
    index: usize,
    puid: usize,
    gpu: usize,
    key: usize,
    latest: Option<usize>,
    values: usize,
}

impl UpdCols {
    fn new(schema: &Schema, name: &str, key: &str) -> Result<Self> {
        let (index, u): (usize, &Update) = schema
            .update(name)
            .ok_or_else(|| NvtraceSchemaSnafu { table: format!("update {name}"), column: String::new() }.build())?;
        let pos = |c: &str| {
            u.match_columns
                .iter()
                .position(|m| m == c)
                .ok_or_else(|| NvtraceSchemaSnafu { table: format!("update {name}"), column: c.to_string() }.build())
        };
        let n = u.match_columns.len();
        Ok(Self {
            index,
            puid: pos("process_id_unique").or_else(|_| pos("gpu_uuid"))?,
            gpu: pos("gpu_uuid")?,
            key: pos(key)?,
            latest: u.latest_column.as_ref().map(|_| n),
            values: n + usize::from(u.latest_column.is_some()),
        })
    }
}

fn u64_at(v: &[Value<'_>], i: usize) -> Option<u64> {
    v.get(i).and_then(Value::u64)
}
fn i64_at(v: &[Value<'_>], i: usize) -> Option<i64> {
    v.get(i).and_then(Value::i64)
}
fn str_at<'a>(v: &[Value<'a>], i: usize) -> Option<&'a str> {
    v.get(i).and_then(Value::str)
}
fn uuid_at(v: &[Value<'_>], i: usize) -> Uuid {
    v.get(i).and_then(Value::uuid).unwrap_or([0; 16])
}
fn ts(v: Option<u64>) -> Option<i64> {
    v.and_then(|t| i64::try_from(t).ok()).filter(|&t| t > 0)
}
fn dims(v: &[Value<'_>], i: [usize; 3]) -> Option<[i32; 3]> {
    let g = |k: usize| u64_at(v, i[k]).map(|x| i32::try_from(x).unwrap_or(i32::MAX));
    Some([g(0)?, g(1)?, g(2)?])
}
fn low32(v: Option<u64>) -> Option<u32> {
    v.map(|x| x as u32)
}

#[derive(Clone, Debug)]
struct Pending {
    act: Activity,
    snapshot: u64,
    /// Graph node launches: GPU start in the GPU clock domain (update bound).
    gpu_ns: u64,
    /// How long before the end of its snapshot's window the activity started (diagnostics).
    gap_ns: u64,
}

/// Buckets of "started this long before the snapshot cutoff": < 1 ms, < 10 ms, < 100 ms, < 1 s, >= 1 s.
pub type GapHist = [u64; 5];

fn gap_bucket(h: &mut GapHist, gap_ns: u64) {
    let i = match gap_ns {
        0..1_000_000 => 0,
        1_000_000..10_000_000 => 1,
        10_000_000..100_000_000 => 2,
        100_000_000..1_000_000_000 => 3,
        _ => 4,
    };
    h[i] += 1;
}

/// Per-snapshot diagnostics (`gt exporter --diag FILE` writes one JSON line per snapshot).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SnapDiag {
    pub snapshot: u64,
    pub window_start_utc_ns: u64,
    pub window_end_utc_ns: u64,
    pub requested_begin_utc_ns: u64,
    pub cutoff_utc_ns: u64,
    pub actual_begin_utc_ns: u64,
    pub actual_end_utc_ns: u64,
    pub backpressured_records: u64,
    pub history_truncated: bool,
    pub attach_complete: bool,
    pub dropped_attach_records: u64,
    pub data_blocks: u64,
    pub kern_rows: u64,
    pub kern_rows_incomplete: u64,
    pub kern_rows_seen_before: u64,
    pub node_rows: u64,
    pub node_rows_incomplete: u64,
    pub node_rows_before_window: u64,
    pub node_rows_seen_before: u64,
    pub node_rows_incomplete_seen_before: u64,
    pub node_incomplete_gap: GapHist,
    pub cmd_rows: u64,
    pub glaunch_rows: u64,
    pub upd_kern: u64,
    pub upd_kern_orphan: u64,
    pub upd_node: u64,
    pub upd_node_orphan_nokey: u64,
    pub upd_node_orphan_nomatch: u64,
    pub upd_node_no_start: u64,
    pub upd_cmd: u64,
    pub upd_cmd_orphan: u64,
    pub upd_glaunch: u64,
    pub upd_glaunch_orphan: u64,
    /// Rows completed by an update that had arrived in an earlier snapshot (GPU times before the launch record).
    pub early_kern_used: u64,
    pub early_node_used: u64,
    pub early_cmd_used: u64,
    pub early_glaunch_used: u64,
    /// Early updates still unmatched after PENDING_SNAPSHOTS snapshots (true orphans).
    pub early_expired: u64,
    pub early_kern: u64,
    pub early_node: u64,
    pub early_cmd: u64,
    pub node_rows_without_launch: u64,
    /// Graph completeness, cumulative since start (diag only). Per (process, executable): launches L, unique
    /// node-launch rows N, distinct kernel nodes ever seen D. With nothing lost N = D * L for every executable
    /// (up to launches whose nodes straddle the newest cutoff).
    pub graph_launches: u64,
    pub graph_node_rows: u64,
    pub graph_node_expected: u64,
    /// Executables with N < D * L, and the largest shortfalls (pid, exec, launches, distinct nodes, rows).
    pub graph_execs_short: u64,
    pub graph_short_top: Vec<(u32, u64, u64, u64, u64)>,
    pub lost_kern: u64,
    pub lost_node: u64,
    pub lost_cmd: u64,
    pub lost_node_gap: GapHist,
    /// Lost graph kernels by expiry age: 3, 4, 5, >= 6 snapshots after their row.
    pub lost_node_age_snapshots: [u64; 4],
    pub pending_kern: u64,
    pub pending_node: u64,
    pub pending_cmd: u64,
    pub decode_ms: f64,
    /// Wall clock at the snapshot's begin marker minus its window end: how far the export runs behind (the SDK
    /// builds the data model before the marker, so this includes the build).
    pub lag_ms: f64,
    /// Export-thread queueing after the cutoff (snapshot handed to us minus window end).
    pub queue_ms: f64,
    /// SDK data-model build of this snapshot (part of lag_ms).
    pub build_ms: f64,
    /// Rows per table and updates per kind in this snapshot (non-zero only).
    pub tables: Vec<(String, u64)>,
    pub updates: Vec<(String, u64)>,
}

fn key_hash<T: Hash>(t: T) -> u64 {
    let mut h = FxHasher::default();
    t.hash(&mut h);
    h.finish()
}

#[derive(Clone, Debug)]
struct Cmd {
    act: Activity,
    kind: Option<EventKind>,
    end_ref: u64,
    snapshot: u64,
}

/// An update that arrived before its row.
#[derive(Clone, Copy, Debug)]
struct Early<T> {
    val: T,
    snapshot: u64,
}

/// Early end of a graph node launch: `bound` is the GPU-clock end (the update's "latest" column).
#[derive(Clone, Copy, Debug)]
struct EarlyEnd {
    bound: u64,
    end_ns: i64,
    snapshot: u64,
}

#[derive(Clone, Copy, Debug)]
struct GraphLaunch {
    exec: u64,
    stream: Option<u32>,
    context: Option<u32>,
    correlation: u32,
    gpu_ns: Option<u64>,
}

/// One evaluated GPU PM metric sample (`gpu_metrics`).
#[derive(Clone, Debug)]
pub struct PmSample {
    pub gpu: Uuid,
    pub end_utc_ns: u64,
    pub metric: Arc<str>,
    pub value: f64,
}


/// Counters for validation and the import report.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct MapStats {
    pub rows_per_table: Vec<(String, u64)>,
    pub updates_per_kind: Vec<(String, u64)>,
    pub kernels: u64,
    pub graph_kernels: u64,
    pub memcpy: u64,
    pub memset: u64,
    pub driver_api: u64,
    pub runtime_api: u64,
    pub other_gpu_commands: u64,
    pub unnamed_kernels: u64,
    pub graph_kernels_without_launch: u64,
    pub completed_by_update: u64,
    /// Rows completed by an update that arrived before them.
    pub completed_by_early_update: u64,
    /// Updates that never matched a row (counted when they expire, or at the end of the input).
    pub orphan_updates: u64,
    pub lost_incomplete: u64,
    pub lost_kernels: u64,
    pub lost_graph_kernels: u64,
    pub lost_gpu_commands: u64,
    pub snapshots: u64,
}

impl MapStats {
    /// Add another shard's counters (`snapshots` is shared, not summed).
    pub fn add(&mut self, o: &MapStats) {
        for (dst, src) in [(&mut self.rows_per_table, &o.rows_per_table), (&mut self.updates_per_kind, &o.updates_per_kind)] {
            for (k, v) in src {
                match dst.iter_mut().find(|(n, _)| n == k) {
                    Some(e) => e.1 += v,
                    None => dst.push((k.clone(), *v)),
                }
            }
        }
        self.kernels += o.kernels;
        self.graph_kernels += o.graph_kernels;
        self.memcpy += o.memcpy;
        self.memset += o.memset;
        self.driver_api += o.driver_api;
        self.runtime_api += o.runtime_api;
        self.other_gpu_commands += o.other_gpu_commands;
        self.unnamed_kernels += o.unnamed_kernels;
        self.graph_kernels_without_launch += o.graph_kernels_without_launch;
        self.completed_by_update += o.completed_by_update;
        self.completed_by_early_update += o.completed_by_early_update;
        self.orphan_updates += o.orphan_updates;
        self.lost_incomplete += o.lost_incomplete;
        self.lost_kernels += o.lost_kernels;
        self.lost_graph_kernels += o.lost_graph_kernels;
        self.lost_gpu_commands += o.lost_gpu_commands;
        self.snapshots = self.snapshots.max(o.snapshots);
    }
}

pub struct Mapper {
    procs_c: Cols<4>,
    threads_c: Cols<3>,
    funcs_c: Cols<3>,
    gnodes_c: Cols<4>,
    drv_c: Cols<6>,
    rt_c: Cols<6>,
    glaunch_c: Cols<8>,
    kern_c: Cols<17>,
    node_c: Cols<14>,
    cmd_c: Cols<12>,
    memcpy_c: Cols<5>,
    memset_c: Cols<4>,
    part_c: Cols<3>,
    metric_c: Cols<4>,
    u_kern: UpdCols,
    u_cmd_start: UpdCols,
    u_cmd_end: UpdCols,
    u_glaunch: UpdCols,
    u_node: UpdCols,

    pub procs: FxHashMap<u64, ProcessInfo>,
    threads: FxHashMap<(u64, u64), u32>,
    funcs: FxHashMap<(u64, u64), Arc<str>>,
    gnodes: FxHashMap<(u64, u64), (u64, u64)>, // (puid, node corr) -> (exec id, cugraph node id)
    glaunch: FxHashMap<(u64, Uuid, u32), GraphLaunch>,
    timelines: FxHashMap<(u64, u64), Vec<(u64, GraphLaunch)>>, // (puid, exec) -> sorted by gpu ns
    pending_kern: FxHashMap<(u64, Uuid, u32), Pending>,
    pending_node: FxHashMap<(u64, Uuid, u64), Vec<Pending>>,
    cmds: FxHashMap<(u64, Uuid, u64), Cmd>,
    cmd_end: FxHashMap<(u64, Uuid, u64), u64>,
    early_kern: FxHashMap<(u64, Uuid, u32), Early<(i64, i64)>>,
    early_node: FxHashMap<(u64, Uuid, u64), Vec<EarlyEnd>>,
    early_cmd_start: FxHashMap<(u64, Uuid, u64), Early<i64>>,
    early_cmd_end: FxHashMap<(u64, Uuid, u64), Early<i64>>,
    early_glaunch: FxHashMap<(u64, Uuid, u32), Early<u64>>,
    names: FxHashMap<String, Arc<str>>,
    pub gpus: FxHashMap<Uuid, (String, String)>,
    pub pm: Vec<PmSample>,
    pub keep_pm: bool,
    rows: Vec<u64>,
    upds: Vec<u64>,
    snap_rows: Vec<u64>,
    snap_upds: Vec<u64>,
    table_names: Vec<String>,
    update_names: Vec<String>,
    pub stats: MapStats,
    snapshot: u64,
    unknown: Arc<str>,
    /// Collect [`SnapDiag`] (costs two hash sets of row keys).
    pub diag_on: bool,
    pub diag: SnapDiag,
    window: (u64, u64),
    seen: [FxHashSet<u64>; 3],
    gexec: FxHashMap<(u64, u64), GraphExecCount>,
    /// Sharded live export: GPUs whose hardware data this mapper's shard decodes (None: all), and whether it
    /// handles the CPU-side API calls.
    owned: Option<FxHashSet<Uuid>>,
    cpu_rows: bool,
    gpu_col_t: Vec<Option<usize>>,
    /// CPU-only tables (API calls) and their process column.
    cpu_t: Vec<Option<usize>>,
    gpu_col_u: Vec<Option<usize>>,
    /// Processes whose CPU-side rows only this shard gets.
    excl: FxHashSet<u64>,
}

/// Per graph executable (diag only), see [`SnapDiag::graph_node_expected`].
#[derive(Default)]
struct GraphExecCount {
    launches: u64,
    node_rows: u64,
    nodes: FxHashSet<u64>,
}

impl Mapper {
    pub fn new(s: &Schema) -> Result<Self> {
        let mut m = Self {
            procs_c: Cols::new(s, "processes", ["process_id_unique", "process_id_namespace", "path", "process_id_host"])?,
            threads_c: Cols::new(s, "thread_event_threads", ["process_id_unique", "thread_id_unique", "thread_id_native"])?,
            funcs_c: Cols::new(s, "cuda_functions", ["process_id_unique", "cufunction_id", "name_mangled"])?,
            gnodes_c: Cols::new(s, "cuda_graph_nodes", ["process_id_unique", "graph_node_correlation_id", "cugraph_exec_id", "cugraph_node_id"])?,
            drv_c: Cols::new(
                s,
                "cuda_driver_api_calls",
                ["process_id_unique", "thread_id_unique", "driver_api_correlation_id", "start_utc_ns", "end_utc_ns", "driver_api_function_name"],
            )?,
            rt_c: Cols::new(
                s,
                "cuda_runtime_api_calls",
                ["process_id_unique", "thread_id_unique", "runtime_api_correlation_id", "start_utc_ns", "end_utc_ns", "runtime_api_function_name"],
            )?,
            glaunch_c: Cols::new(
                s,
                "cuda_graph_launches",
                [
                    "process_id_unique",
                    "gpu_uuid",
                    "hw_correlation_id",
                    "custream_id",
                    "cucontext_id",
                    "cugraph_exec_id",
                    "driver_api_correlation_id",
                    "timestamp_gpu_ns",
                ],
            )?,
            kern_c: Cols::new(
                s,
                "cuda_immediate_kernel_launches",
                [
                    "process_id_unique",
                    "gpu_uuid",
                    "cudevice_ordinal",
                    "hw_correlation_id",
                    "driver_api_correlation_id",
                    "custream_id",
                    "cucontext_id",
                    "cufunction_id",
                    "grid_dim_x",
                    "grid_dim_y",
                    "grid_dim_z",
                    "block_dim_x",
                    "block_dim_y",
                    "block_dim_z",
                    "shmem_per_block_bytes_total",
                    "gpu_start_utc_ns",
                    "gpu_end_utc_ns",
                ],
            )?,
            node_c: Cols::new(
                s,
                "cuda_graph_node_launches",
                [
                    "process_id_unique",
                    "gpu_uuid",
                    "cudevice_ordinal",
                    "graph_node_correlation_id",
                    "start_timestamp_gpu_ns",
                    "cufunction_id",
                    "grid_dim_x",
                    "grid_dim_y",
                    "grid_dim_z",
                    "block_dim_x",
                    "block_dim_y",
                    "block_dim_z",
                    "gpu_start_utc_ns",
                    "gpu_end_utc_ns",
                ],
            )?,
            cmd_c: Cols::new(
                s,
                "cuda_gpu_commands",
                [
                    "process_id_unique",
                    "gpu_uuid",
                    "cudevice_ordinal",
                    "start_gpu_timestamp_report_ref",
                    "end_gpu_timestamp_report_ref",
                    "custream_id",
                    "cucontext_id",
                    "gpu_start_utc_ns",
                    "gpu_end_utc_ns",
                    "graph_node_correlation_id",
                    "driver_api_correlation_id",
                    "name_display",
                ],
            )?,
            memcpy_c: Cols::new(
                s,
                "cuda_gpu_memcpy_details",
                ["process_id_unique", "gpu_uuid", "start_gpu_timestamp_report_ref", "byte_count", "copy_direction"],
            )?,
            memset_c: Cols::new(s, "cuda_gpu_memset_details", ["process_id_unique", "gpu_uuid", "start_gpu_timestamp_report_ref", "byte_count"])?,
            part_c: Cols::new(s, "gpu_partitions", ["gpu_uuid", "chip_name", "device_name"])?,
            metric_c: Cols::new(s, "gpu_metrics", ["gpu_uuid", "gpu_end_utc_ns", "metric_name", "metric_value"])?,
            u_kern: UpdCols::new(s, "cuda_immediate_kernel_launch_gpu_times", "hw_correlation_id")?,
            u_cmd_start: UpdCols::new(s, "cuda_gpu_command_gpu_start", "start_gpu_timestamp_report_ref")?,
            u_cmd_end: UpdCols::new(s, "cuda_gpu_command_gpu_end", "end_gpu_timestamp_report_ref")?,
            u_glaunch: UpdCols::new(s, "cuda_graph_launch_timestamp", "hw_correlation_id")?,
            u_node: UpdCols::new(s, "cuda_graph_node_launch_gpu_end", "graph_node_correlation_id")?,
            procs: FxHashMap::default(),
            threads: FxHashMap::default(),
            funcs: FxHashMap::default(),
            gnodes: FxHashMap::default(),
            glaunch: FxHashMap::default(),
            timelines: FxHashMap::default(),
            pending_kern: FxHashMap::default(),
            pending_node: FxHashMap::default(),
            cmds: FxHashMap::default(),
            cmd_end: FxHashMap::default(),
            early_kern: FxHashMap::default(),
            early_node: FxHashMap::default(),
            early_cmd_start: FxHashMap::default(),
            early_cmd_end: FxHashMap::default(),
            early_glaunch: FxHashMap::default(),
            names: FxHashMap::default(),
            gpus: FxHashMap::default(),
            pm: Vec::new(),
            keep_pm: true,
            rows: vec![0; s.tables.len()],
            upds: vec![0; s.updates.len()],
            snap_rows: vec![0; s.tables.len()],
            snap_upds: vec![0; s.updates.len()],
            table_names: s.tables.iter().map(|t| t.name.clone()).collect(),
            update_names: s.updates.iter().map(|u| u.name.clone()).collect(),
            stats: MapStats::default(),
            snapshot: 0,
            unknown: Arc::from("<unknown>"),
            diag_on: false,
            diag: SnapDiag::default(),
            window: (0, 0),
            seen: Default::default(),
            gexec: FxHashMap::default(),
            owned: None,
            cpu_rows: true,
            gpu_col_t: Vec::new(),
            cpu_t: Vec::new(),
            gpu_col_u: Vec::new(),
            excl: FxHashSet::default(),
        };
        m.init_routing(s.tables.len(), s.updates.len());
        Ok(m)
    }

    /// Which tables / updates belong to one GPU (and in which column), and which are CPU-only API calls.
    fn init_routing(&mut self, tables: usize, updates: usize) {
        self.gpu_col_t = vec![None; tables];
        self.cpu_t = vec![None; tables];
        for (t, c) in [
            (self.kern_c.table, self.kern_c.idx[1]),
            (self.node_c.table, self.node_c.idx[1]),
            (self.glaunch_c.table, self.glaunch_c.idx[1]),
            (self.cmd_c.table, self.cmd_c.idx[1]),
            (self.memcpy_c.table, self.memcpy_c.idx[1]),
            (self.memset_c.table, self.memset_c.idx[1]),
            (self.metric_c.table, self.metric_c.idx[0]),
        ] {
            if let Some(x) = self.gpu_col_t.get_mut(t) {
                *x = Some(c);
            }
        }
        for (t, c) in [(self.drv_c.table, self.drv_c.idx[0]), (self.rt_c.table, self.rt_c.idx[0])] {
            if let Some(x) = self.cpu_t.get_mut(t) {
                *x = Some(c);
            }
        }
        self.gpu_col_u = vec![None; updates];
        for u in [&self.u_kern, &self.u_cmd_start, &self.u_cmd_end, &self.u_glaunch, &self.u_node] {
            if let Some(x) = self.gpu_col_u.get_mut(u.index) {
                *x = Some(u.gpu);
            }
        }
    }

    /// Sharded live export: a row of another shard's GPU arrives here without GPU times and must be ignored; API
    /// calls belong to the shard that owns their process exclusively, else to shard 0. Dimension tables are
    /// processed everywhere but counted once. Returns (process, count).
    fn route(&self, gpu_col: Option<usize>, cpu_col: Option<usize>, gpu_side: bool, v: &[Value<'_>]) -> (bool, bool) {
        let Some(owned) = &self.owned else { return (true, true) };
        if let Some(c) = gpu_col {
            let g = uuid_at(v, c);
            let mine = if g == [0u8; 16] { self.cpu_rows } else { owned.contains(&g) };
            return (mine, mine);
        }
        if let Some(c) = cpu_col {
            let mine = self.cpu_rows || u64_at(v, c).is_some_and(|p| self.excl.contains(&p));
            return (mine, mine);
        }
        if gpu_side {
            return (self.cpu_rows, self.cpu_rows);
        }
        (true, self.cpu_rows)
    }

    fn intern(&mut self, s: &str) -> Arc<str> {
        if let Some(a) = self.names.get(s) {
            return a.clone();
        }
        let a: Arc<str> = Arc::from(s);
        self.names.insert(s.to_string(), a.clone());
        a
    }

    /// The host pid of a process (its unique id when the process row is missing).
    fn pid(&self, puid: u64) -> u32 {
        self.procs.get(&puid).map(|p| p.pid).unwrap_or(puid as u32)
    }

    /// The pid in the process's own PID namespace.
    fn ns_pid(&self, puid: u64) -> u32 {
        self.procs.get(&puid).map(|p| p.ns_pid).unwrap_or(puid as u32)
    }

    /// Feed one exporter event; completed activities go to `out`.
    pub fn on_event(&mut self, ev: Event<'_>, out: &mut dyn FnMut(Activity)) {
        match ev {
            Event::Begin(w) => {
                self.snapshot = w.snapshot;
                self.owned = w.shard.map(|sh| sh.gpus().iter().copied().collect());
                self.excl = w.shard.map(|sh| sh.processes().iter().copied().collect()).unwrap_or_default();
                self.cpu_rows = w.shard.is_none_or(|sh| sh.index == 0);
                self.window = (w.start_utc_ns, w.end_utc_ns);
                let m = w.manifest.unwrap_or_default();
                self.diag = SnapDiag {
                    snapshot: w.snapshot,
                    window_start_utc_ns: w.start_utc_ns,
                    window_end_utc_ns: w.end_utc_ns,
                    requested_begin_utc_ns: m.requested_begin_utc_ns,
                    cutoff_utc_ns: m.cutoff_utc_ns,
                    actual_begin_utc_ns: m.actual_begin_utc_ns,
                    actual_end_utc_ns: m.actual_end_utc_ns,
                    backpressured_records: m.backpressured_records,
                    history_truncated: m.history_truncated,
                    attach_complete: m.attach_complete,
                    dropped_attach_records: m.dropped_attach_records,
                    data_blocks: m.data_blocks,
                    lag_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| (d.as_nanos() as f64 - w.end_utc_ns as f64) / 1e6)
                        .unwrap_or(0.0),
                    queue_ms: if m.export_entry_utc_ns > 0 {
                        (m.export_entry_utc_ns as f64 - w.end_utc_ns as f64) / 1e6
                    } else {
                        0.0
                    },
                    build_ms: m.build_ns as f64 / 1e6,
                    ..Default::default()
                };
                self.snap_rows.iter_mut().for_each(|c| *c = 0);
                self.snap_upds.iter_mut().for_each(|c| *c = 0);
                if self.diag_on {
                    // seen[0]: this snapshot, seen[1..]: the two before it.
                    self.seen.rotate_right(1);
                    self.seen[0].clear();
                }
            }
            Event::Row { table, values } => {
                let (keep, count) =
                    self.route(self.gpu_col_t.get(table).copied().flatten(), self.cpu_t.get(table).copied().flatten(), false, values);
                if count {
                    if let Some(c) = self.rows.get_mut(table) {
                        *c += 1;
                    }
                    if let Some(c) = self.snap_rows.get_mut(table) {
                        *c += 1;
                    }
                }
                if keep {
                    self.on_row(table, values, out);
                }
            }
            Event::Update { update, values } => {
                let gpu_col = self.gpu_col_u.get(update).copied().flatten();
                // Updates the mapper does not use are GPU-side too: count them once.
                let (keep, count) = self.route(gpu_col, None, gpu_col.is_none(), values);
                if count {
                    if let Some(c) = self.upds.get_mut(update) {
                        *c += 1;
                    }
                    if let Some(c) = self.snap_upds.get_mut(update) {
                        *c += 1;
                    }
                }
                if keep {
                    self.on_update(update, values, out);
                }
            }
            Event::Done(_) => {}
            Event::Commit(_) => {
                self.stats.snapshots += 1;
                self.flush_cmds(out, false);
                self.expire();
                self.diag.pending_kern = self.pending_kern.len() as u64;
                self.diag.pending_node = self.pending_node.values().map(|l| l.len() as u64).sum();
                self.diag.pending_cmd = self.cmds.len() as u64;
                self.diag.early_kern = self.early_kern.len() as u64;
                self.diag.early_node = self.early_node.values().map(|l| l.len() as u64).sum();
                self.diag.early_cmd = (self.early_cmd_start.len() + self.early_cmd_end.len()) as u64;
                if self.diag_on {
                    let nz = |names: &[String], c: &[u64]| names.iter().cloned().zip(c.iter().copied()).filter(|(_, n)| *n > 0).collect();
                    self.diag.tables = nz(&self.table_names, &self.snap_rows);
                    self.diag.updates = nz(&self.update_names, &self.snap_upds);
                    self.graph_completeness();
                }
            }
        }
    }

    fn on_row(&mut self, t: usize, v: &[Value<'_>], out: &mut dyn FnMut(Activity)) {
        if t == self.procs_c.table {
            let c = &self.procs_c.idx;
            let ns_pid = i64_at(v, c[1]).and_then(|p| u32::try_from(p).ok()).unwrap_or(0);
            // The host pid is 0 when the service could not tell (no SO_PEERCRED): the namespace's is all there is.
            let pid = i64_at(v, c[3]).and_then(|p| u32::try_from(p).ok()).filter(|&p| p != 0).unwrap_or(ns_pid);
            let exe = str_at(v, c[2]).map(|p| p.rsplit('/').next().unwrap_or(p).to_string()).unwrap_or_default();
            if let Some(puid) = u64_at(v, c[0]) {
                self.procs.insert(puid, ProcessInfo { pid, ns_pid, exe, title: String::new() });
            }
        } else if t == self.threads_c.table {
            let c = &self.threads_c.idx;
            if let (Some(p), Some(tu), Some(n)) = (u64_at(v, c[0]), u64_at(v, c[1]), u64_at(v, c[2])) {
                self.threads.insert((p, tu), n as u32);
            }
        } else if t == self.funcs_c.table {
            let c = &self.funcs_c.idx;
            if let (Some(p), Some(f), Some(n)) = (u64_at(v, c[0]), u64_at(v, c[1]), str_at(v, c[2])) {
                let n = self.intern(n);
                self.funcs.insert((p, f), n);
            }
        } else if t == self.gnodes_c.table {
            let c = &self.gnodes_c.idx;
            if let (Some(p), Some(nc), Some(e)) = (u64_at(v, c[0]), u64_at(v, c[1]), u64_at(v, c[2])) {
                self.gnodes.insert((p, nc), (e, u64_at(v, c[3]).unwrap_or(nc)));
            }
        } else if t == self.drv_c.table || t == self.rt_c.table {
            let (c, kind) = if t == self.drv_c.table { (self.drv_c.idx, EventKind::Driver) } else { (self.rt_c.idx, EventKind::Runtime) };
            let (Some(s), Some(e)) = (ts(u64_at(v, c[3])), ts(u64_at(v, c[4]))) else { return };
            let puid = u64_at(v, c[0]).unwrap_or(0);
            let tid = u64_at(v, c[1]).and_then(|tu| self.threads.get(&(puid, tu)).copied()).map(u64::from);
            let name = self.intern(str_at(v, c[5]).unwrap_or("?"));
            match kind {
                EventKind::Driver => self.stats.driver_api += 1,
                _ => self.stats.runtime_api += 1,
            }
            out(Activity {
                kind,
                name,
                start_ns: s,
                end_ns: e,
                pid: self.pid(puid),
                ns_pid: self.ns_pid(puid),
                tid,
                device: None,
                context: None,
                stream: None,
                correlation: u64_at(v, c[2]).unwrap_or(0) as u32,
                graph_id: None,
                graph_node_id: None,
                grid: None,
                block: None,
                shmem: None,
                bytes: None,
            });
        } else if t == self.glaunch_c.table {
            let c = &self.glaunch_c.idx;
            let (Some(p), Some(hw), Some(exec)) = (u64_at(v, c[0]), u64_at(v, c[2]), u64_at(v, c[5])) else { return };
            let g = GraphLaunch {
                exec,
                stream: low32(u64_at(v, c[3])),
                context: low32(u64_at(v, c[4])),
                correlation: u64_at(v, c[6]).unwrap_or(0) as u32,
                gpu_ns: u64_at(v, c[7]),
            };
            let gpu = uuid_at(v, c[1]);
            self.diag.glaunch_rows += 1;
            if self.diag_on && self.seen_row(key_hash((2u8, p, gpu, hw))) {
                self.gexec.entry((p, exec)).or_default().launches += 1;
            }
            let mut g = g;
            if g.gpu_ns.is_none()
                && let Some(e) = self.early_glaunch.remove(&(p, gpu, hw as u32))
            {
                g.gpu_ns = Some(e.val);
                self.diag.early_glaunch_used += 1;
            }
            self.glaunch.insert((p, gpu, hw as u32), g);
            if let Some(ns) = g.gpu_ns {
                self.add_timeline(p, g, ns);
            }
        } else if t == self.kern_c.table {
            self.on_kernel(v, out);
        } else if t == self.node_c.table {
            self.on_node(v, out);
        } else if t == self.cmd_c.table {
            let c = self.cmd_c.idx;
            let (Some(p), Some(sr)) = (u64_at(v, c[0]), u64_at(v, c[3])) else { return };
            let gpu = uuid_at(v, c[1]);
            let er = u64_at(v, c[4]).unwrap_or(0);
            let name = self.intern(str_at(v, c[11]).unwrap_or("?"));
            let lname = name.to_ascii_lowercase();
            let kind = if lname.contains("memcpy") {
                Some(EventKind::Memcpy)
            } else if lname.contains("memset") {
                Some(EventKind::Memset)
            } else {
                None
            };
            let act = Activity {
                kind: kind.unwrap_or(EventKind::Memcpy),
                name,
                start_ns: ts(u64_at(v, c[7])).unwrap_or(0),
                end_ns: ts(u64_at(v, c[8])).unwrap_or(0),
                pid: self.pid(p),
                ns_pid: self.ns_pid(p),
                tid: None,
                device: low32(u64_at(v, c[2])),
                context: low32(u64_at(v, c[6])),
                stream: low32(u64_at(v, c[5])),
                correlation: u64_at(v, c[10]).unwrap_or(0) as u32,
                graph_id: None,
                graph_node_id: u64_at(v, c[9]),
                grid: None,
                block: None,
                shmem: None,
                bytes: None,
            };
            self.diag.cmd_rows += 1;
            let mut act = act;
            let mut used = false;
            if act.start_ns == 0
                && let Some(e) = self.early_cmd_start.remove(&(p, gpu, sr))
            {
                act.start_ns = e.val;
                used = true;
            }
            if act.end_ns == 0
                && let Some(e) = self.early_cmd_end.remove(&(p, gpu, er))
            {
                act.end_ns = e.val;
                used = true;
            }
            if used {
                self.diag.early_cmd_used += 1;
                self.stats.completed_by_early_update += u64::from(act.start_ns > 0 && act.end_ns > 0);
            }
            self.cmd_end.insert((p, gpu, er), sr);
            self.cmds.insert((p, gpu, sr), Cmd { act, kind, end_ref: er, snapshot: self.snapshot });
        } else if t == self.memcpy_c.table || t == self.memset_c.table {
            let is_cpy = t == self.memcpy_c.table;
            let (p, g, r, b) = if is_cpy {
                let c = &self.memcpy_c.idx;
                (u64_at(v, c[0]), uuid_at(v, c[1]), u64_at(v, c[2]), u64_at(v, c[3]))
            } else {
                let c = &self.memset_c.idx;
                (u64_at(v, c[0]), uuid_at(v, c[1]), u64_at(v, c[2]), u64_at(v, c[3]))
            };
            let dir = if is_cpy { str_at(v, self.memcpy_c.idx[4]).map(str::to_string) } else { None };
            let (Some(p), Some(r)) = (p, r) else { return };
            let name = dir.map(|d| self.intern(&format!("Memcpy {d}")));
            if let Some(cmd) = self.cmds.get_mut(&(p, g, r)) {
                cmd.kind = Some(if is_cpy { EventKind::Memcpy } else { EventKind::Memset });
                cmd.act.bytes = b;
                if let Some(n) = name {
                    cmd.act.name = n;
                }
            }
        } else if t == self.part_c.table {
            let c = &self.part_c.idx;
            let gpu = uuid_at(v, c[0]);
            self.gpus.insert(gpu, (str_at(v, c[1]).unwrap_or("").to_string(), str_at(v, c[2]).unwrap_or("").to_string()));
        } else if t == self.metric_c.table && self.keep_pm {
            let c = self.metric_c.idx;
            if let (Some(end), Some(m), Some(val)) = (u64_at(v, c[1]), str_at(v, c[2]), v.get(c[3]).and_then(Value::f64)) {
                let metric = self.intern(m);
                self.pm.push(PmSample { gpu: uuid_at(v, c[0]), end_utc_ns: end, metric, value: val });
            }
        }
    }

    fn add_timeline(&mut self, puid: u64, g: GraphLaunch, ns: u64) {
        let tl = self.timelines.entry((puid, g.exec)).or_default();
        let pos = tl.partition_point(|(t, _)| *t <= ns);
        tl.insert(pos, (ns, g));
    }

    fn kernel_name(&mut self, puid: u64, f: Option<u64>) -> Arc<str> {
        match f.and_then(|f| self.funcs.get(&(puid, f))) {
            Some(n) => n.clone(),
            None => {
                self.stats.unnamed_kernels += 1;
                self.unknown.clone()
            }
        }
    }

    fn on_kernel(&mut self, v: &[Value<'_>], out: &mut dyn FnMut(Activity)) {
        let c = self.kern_c.idx;
        let Some(p) = u64_at(v, c[0]) else { return };
        let gpu = uuid_at(v, c[1]);
        let name = self.kernel_name(p, u64_at(v, c[7]));
        let act = Activity {
            kind: EventKind::Kernel,
            name,
            start_ns: ts(u64_at(v, c[15])).unwrap_or(0),
            end_ns: ts(u64_at(v, c[16])).unwrap_or(0),
            pid: self.pid(p),
            ns_pid: self.ns_pid(p),
            tid: None,
            device: low32(u64_at(v, c[2])),
            context: low32(u64_at(v, c[6])),
            stream: low32(u64_at(v, c[5])),
            correlation: u64_at(v, c[4]).unwrap_or(0) as u32,
            graph_id: None,
            graph_node_id: None,
            grid: dims(v, [c[8], c[9], c[10]]),
            block: dims(v, [c[11], c[12], c[13]]),
            shmem: u64_at(v, c[14]).map(|x| x as i32),
            bytes: None,
        };
        let hw = u64_at(v, c[3]).unwrap_or(0) as u32;
        self.diag.kern_rows += 1;
        if self.diag_on && !self.seen_row(key_hash((0u8, p, gpu, hw))) {
            self.diag.kern_rows_seen_before += 1;
        }
        let mut act = act;
        if (act.start_ns == 0 || act.end_ns == 0)
            && let Some(e) = self.early_kern.remove(&(p, gpu, hw))
        {
            if act.start_ns == 0 {
                act.start_ns = e.val.0;
            }
            if act.end_ns == 0 {
                act.end_ns = e.val.1;
            }
            self.diag.early_kern_used += 1;
            self.stats.completed_by_early_update += u64::from(act.start_ns > 0 && act.end_ns > 0);
        }
        if act.start_ns > 0 && act.end_ns > 0 {
            self.stats.kernels += 1;
            out(act);
        } else {
            self.diag.kern_rows_incomplete += 1;
            let gap_ns = self.gap(act.start_ns);
            self.pending_kern.insert((p, gpu, hw), Pending { act, snapshot: self.snapshot, gpu_ns: 0, gap_ns });
        }
    }

    fn on_node(&mut self, v: &[Value<'_>], out: &mut dyn FnMut(Activity)) {
        let c = self.node_c.idx;
        let (Some(p), Some(nc)) = (u64_at(v, c[0]), u64_at(v, c[3])) else { return };
        let gpu = uuid_at(v, c[1]);
        let start_gpu = u64_at(v, c[4]).unwrap_or(0);
        let name = self.kernel_name(p, u64_at(v, c[5]));
        let (exec, node) = self.gnodes.get(&(p, nc)).copied().unwrap_or((0, nc));
        // The launch that ran this node: the latest launch of its executable at or before the node's GPU start.
        let launch = self.timelines.get(&(p, exec)).and_then(|tl| {
            let i = tl.partition_point(|(t, _)| *t <= start_gpu);
            i.checked_sub(1).map(|i| tl[i].1)
        });
        if launch.is_none() {
            self.stats.graph_kernels_without_launch += 1;
            self.diag.node_rows_without_launch += 1;
        }
        let mut act = Activity {
            kind: EventKind::Kernel,
            name,
            start_ns: ts(u64_at(v, c[12])).unwrap_or(0),
            end_ns: ts(u64_at(v, c[13])).unwrap_or(0),
            pid: self.pid(p),
            ns_pid: self.ns_pid(p),
            tid: None,
            device: low32(u64_at(v, c[2])),
            context: launch.and_then(|l| l.context),
            stream: launch.and_then(|l| l.stream),
            correlation: launch.map(|l| l.correlation).unwrap_or(0),
            graph_id: (exec != 0).then_some(exec as u32),
            graph_node_id: Some(node),
            grid: dims(v, [c[6], c[7], c[8]]),
            block: dims(v, [c[9], c[10], c[11]]),
            shmem: None,
            bytes: None,
        };
        if act.end_ns == 0 {
            // An end that arrived before this row: the first early end at or after this launch's GPU start.
            if let Some(list) = self.early_node.get_mut(&(p, gpu, nc)) {
                let best = list.iter().enumerate().filter(|(_, e)| e.bound >= start_gpu).min_by_key(|(_, e)| e.bound).map(|(i, _)| i);
                if let Some(i) = best {
                    act.end_ns = list.swap_remove(i).end_ns;
                    self.diag.early_node_used += 1;
                    self.stats.completed_by_early_update += u64::from(act.start_ns > 0);
                }
                if list.is_empty() {
                    self.early_node.remove(&(p, gpu, nc));
                }
            }
        }
        self.diag.node_rows += 1;
        if act.start_ns > 0 && (act.start_ns as u64) < self.window.0 {
            self.diag.node_rows_before_window += 1;
        }
        let complete = act.start_ns > 0 && act.end_ns > 0;
        if self.diag_on {
            if self.seen_row(key_hash((1u8, p, gpu, nc, start_gpu))) {
                if exec != 0 {
                    let g = self.gexec.entry((p, exec)).or_default();
                    g.node_rows += 1;
                    g.nodes.insert(node);
                }
            } else {
                self.diag.node_rows_seen_before += 1;
                self.diag.node_rows_incomplete_seen_before += u64::from(!complete);
            }
        }
        if complete {
            self.stats.graph_kernels += 1;
            out(act);
        } else {
            self.diag.node_rows_incomplete += 1;
            let gap_ns = self.gap(act.start_ns);
            gap_bucket(&mut self.diag.node_incomplete_gap, gap_ns);
            self.pending_node.entry((p, gpu, nc)).or_default().push(Pending { act, snapshot: self.snapshot, gpu_ns: start_gpu, gap_ns });
        }
    }

    fn on_update(&mut self, u: usize, v: &[Value<'_>], out: &mut dyn FnMut(Activity)) {
        let key3 = |c: &UpdCols| (u64_at(v, c.puid).unwrap_or(0), uuid_at(v, c.gpu), u64_at(v, c.key).unwrap_or(0));
        if u == self.u_kern.index {
            let (p, g, k) = key3(&self.u_kern);
            self.diag.upd_kern += 1;
            let vi = self.u_kern.values;
            match self.pending_kern.remove(&(p, g, k as u32)) {
                Some(mut pe) => {
                    pe.act.start_ns = ts(u64_at(v, vi)).unwrap_or(pe.act.start_ns);
                    pe.act.end_ns = ts(u64_at(v, vi + 1)).unwrap_or(pe.act.end_ns);
                    if pe.act.start_ns > 0 && pe.act.end_ns > 0 {
                        self.stats.kernels += 1;
                        self.stats.completed_by_update += 1;
                        out(pe.act);
                    } else {
                        self.pending_kern.insert((p, g, k as u32), pe);
                    }
                }
                None => {
                    // GPU times before the launch row: keep them (merged: start and end can come separately).
                    self.diag.upd_kern_orphan += 1;
                    let (s0, e0) = (ts(u64_at(v, vi)).unwrap_or(0), ts(u64_at(v, vi + 1)).unwrap_or(0));
                    let e = self.early_kern.entry((p, g, k as u32)).or_insert(Early { val: (0, 0), snapshot: self.snapshot });
                    if s0 > 0 {
                        e.val.0 = s0;
                    }
                    if e0 > 0 {
                        e.val.1 = e0;
                    }
                }
            }
        } else if u == self.u_node.index {
            let (p, g, k) = key3(&self.u_node);
            let bound = self.u_node.latest.and_then(|i| u64_at(v, i)).unwrap_or(u64::MAX);
            let end = ts(u64_at(v, self.u_node.values));
            self.diag.upd_node += 1;
            let early = |m: &mut Self| {
                if let Some(e) = end {
                    m.early_node.entry((p, g, k)).or_default().push(EarlyEnd { bound, end_ns: e, snapshot: m.snapshot });
                }
            };
            let Some(list) = self.pending_node.get_mut(&(p, g, k)) else {
                self.diag.upd_node_orphan_nokey += 1;
                early(self);
                return;
            };
            let best = list.iter().enumerate().filter(|(_, pe)| pe.gpu_ns <= bound).max_by_key(|(_, pe)| pe.gpu_ns).map(|(i, _)| i);
            match (best, end) {
                (Some(i), Some(e)) => {
                    let mut pe = list.swap_remove(i);
                    if list.is_empty() {
                        self.pending_node.remove(&(p, g, k));
                    }
                    pe.act.end_ns = e;
                    if pe.act.start_ns > 0 {
                        self.stats.graph_kernels += 1;
                        self.stats.completed_by_update += 1;
                        out(pe.act);
                    } else {
                        self.diag.upd_node_no_start += 1;
                    }
                }
                _ => {
                    self.diag.upd_node_orphan_nomatch += 1;
                    early(self);
                }
            }
        } else if u == self.u_cmd_start.index || u == self.u_cmd_end.index {
            let is_start = u == self.u_cmd_start.index;
            let c = if is_start { &self.u_cmd_start } else { &self.u_cmd_end };
            let (p, g, r) = key3(c);
            self.diag.upd_cmd += 1;
            let val = ts(u64_at(v, c.values));
            let sr = if is_start { Some(r) } else { self.cmd_end.get(&(p, g, r)).copied() };
            match sr.and_then(|sr| self.cmds.get_mut(&(p, g, sr))) {
                Some(cmd) => {
                    if let Some(t) = val {
                        if is_start {
                            cmd.act.start_ns = t;
                        } else {
                            cmd.act.end_ns = t;
                        }
                        self.stats.completed_by_update += u64::from(cmd.act.start_ns > 0 && cmd.act.end_ns > 0);
                    }
                }
                None => {
                    self.diag.upd_cmd_orphan += 1;
                    if let Some(t) = val {
                        let m = if is_start { &mut self.early_cmd_start } else { &mut self.early_cmd_end };
                        m.insert((p, g, r), Early { val: t, snapshot: self.snapshot });
                    }
                }
            }
        } else if u == self.u_glaunch.index {
            let (p, g, hw) = key3(&self.u_glaunch);
            self.diag.upd_glaunch += 1;
            let ns = u64_at(v, self.u_glaunch.values);
            if let (Some(l), Some(ns)) = (self.glaunch.get_mut(&(p, g, hw as u32)), ns) {
                l.gpu_ns = Some(ns);
                let l = *l;
                self.add_timeline(p, l, ns);
            } else {
                self.diag.upd_glaunch_orphan += 1;
                if let Some(ns) = ns {
                    self.early_glaunch.insert((p, g, hw as u32), Early { val: ns, snapshot: self.snapshot });
                }
            }
        }
    }

    /// Diagnostics: record a row key for this snapshot; false if one of the two previous snapshots had it.
    /// Cumulative graph completeness into the snapshot diagnostics.
    fn graph_completeness(&mut self) {
        let (mut l, mut n, mut e, mut short) = (0, 0, 0, Vec::new());
        for (&(p, exec), g) in &self.gexec {
            let want = g.nodes.len() as u64 * g.launches;
            l += g.launches;
            n += g.node_rows;
            e += want;
            if g.node_rows < want {
                short.push((self.procs.get(&p).map(|x| x.pid).unwrap_or(0), exec, g.launches, g.nodes.len() as u64, g.node_rows));
            }
        }
        short.sort_by_key(|&(_, _, l, d, r)| std::cmp::Reverse(d * l - r));
        self.diag.graph_launches = l;
        self.diag.graph_node_rows = n;
        self.diag.graph_node_expected = e;
        self.diag.graph_execs_short = short.len() as u64;
        short.truncate(8);
        self.diag.graph_short_top = short;
    }

    fn seen_row(&mut self, k: u64) -> bool {
        let before = self.seen[1].contains(&k) || self.seen[2].contains(&k);
        self.seen[0].insert(k);
        !before
    }

    /// How long before the end of the current window an activity started.
    fn gap(&self, start_ns: i64) -> u64 {
        u64::try_from(start_ns).ok().filter(|&s| s > 0).map(|s| self.window.1.saturating_sub(s)).unwrap_or(u64::MAX)
    }

    /// Emit GPU commands that are complete (kind known, both times known). With `all`, give up on the rest.
    fn flush_cmds(&mut self, out: &mut dyn FnMut(Activity), all: bool) {
        let snap = self.snapshot;
        let mut done = Vec::new();
        for (k, c) in self.cmds.iter() {
            let complete = c.act.start_ns > 0 && c.act.end_ns > 0;
            if c.kind.is_none() || complete || all || snap.saturating_sub(c.snapshot) >= PENDING_SNAPSHOTS {
                done.push(*k);
            }
        }
        for k in done {
            let Some(mut c) = self.cmds.remove(&k) else { continue };
            self.cmd_end.remove(&(k.0, k.1, c.end_ref));
            let Some(kind) = c.kind else {
                self.stats.other_gpu_commands += 1;
                continue;
            };
            if c.act.start_ns > 0 && c.act.end_ns > 0 {
                c.act.kind = kind;
                match kind {
                    EventKind::Memset => self.stats.memset += 1,
                    _ => self.stats.memcpy += 1,
                }
                out(c.act);
            } else {
                self.stats.lost_incomplete += 1;
                self.stats.lost_gpu_commands += 1;
                self.diag.lost_cmd += 1;
            }
        }
    }

    fn expire(&mut self) {
        // A snapshot exports all its graph launches before its graph node launches, so launch timelines are only
        // trimmed between snapshots; later snapshots need just the most recent launches of each executable.
        for tl in self.timelines.values_mut() {
            if tl.len() > TIMELINE_KEEP {
                tl.drain(..tl.len() - TIMELINE_KEEP);
            }
        }
        let snap = self.snapshot;
        let old = |s: u64| snap.saturating_sub(s) >= PENDING_SNAPSHOTS;
        let before = self.pending_kern.len();
        self.pending_kern.retain(|_, p| !old(p.snapshot));
        let lost_k = (before - self.pending_kern.len()) as u64;
        let (mut lost_n, mut gaps, mut ages) = (0u64, GapHist::default(), [0u64; 4]);
        for list in self.pending_node.values_mut() {
            list.retain(|p| {
                if !old(p.snapshot) {
                    return true;
                }
                lost_n += 1;
                gap_bucket(&mut gaps, p.gap_ns);
                // Expiry age in snapshots, from PENDING_SNAPSHOTS up: a gap in the SDK's numbering shows here.
                let age = snap.saturating_sub(p.snapshot).saturating_sub(PENDING_SNAPSHOTS);
                ages[usize::try_from(age).unwrap_or(3).min(3)] += 1;
                false
            });
        }
        self.pending_node.retain(|_, l| !l.is_empty());
        let mut orphans = 0u64;
        let n0 = self.early_kern.len() + self.early_cmd_start.len() + self.early_cmd_end.len() + self.early_glaunch.len();
        self.early_kern.retain(|_, e| !old(e.snapshot));
        self.early_cmd_start.retain(|_, e| !old(e.snapshot));
        self.early_cmd_end.retain(|_, e| !old(e.snapshot));
        self.early_glaunch.retain(|_, e| !old(e.snapshot));
        orphans += (n0 - (self.early_kern.len() + self.early_cmd_start.len() + self.early_cmd_end.len() + self.early_glaunch.len())) as u64;
        for list in self.early_node.values_mut() {
            let n = list.len();
            list.retain(|e| !old(e.snapshot));
            orphans += (n - list.len()) as u64;
        }
        self.early_node.retain(|_, l| !l.is_empty());
        self.stats.orphan_updates += orphans;
        self.diag.early_expired = orphans;
        self.stats.lost_incomplete += lost_k + lost_n;
        self.stats.lost_kernels += lost_k;
        self.stats.lost_graph_kernels += lost_n;
        self.diag.lost_kern = lost_k;
        self.diag.lost_node = lost_n;
        self.diag.lost_node_gap = gaps;
        self.diag.lost_node_age_snapshots = ages;
    }

    /// End of input: emit what can be emitted, count the rest as lost, fill the per-table counters.
    pub fn finish(&mut self, out: &mut dyn FnMut(Activity)) {
        self.flush_cmds(out, true);
        let (k, n) = (self.pending_kern.len() as u64, self.pending_node.values().map(|l| l.len() as u64).sum::<u64>());
        self.stats.lost_incomplete += k + n;
        self.stats.lost_kernels += k;
        self.stats.lost_graph_kernels += n;
        self.pending_kern.clear();
        self.pending_node.clear();
        self.stats.orphan_updates += (self.early_kern.len()
            + self.early_cmd_start.len()
            + self.early_cmd_end.len()
            + self.early_glaunch.len()
            + self.early_node.values().map(Vec::len).sum::<usize>()) as u64;
        self.early_kern.clear();
        self.early_node.clear();
        self.early_cmd_start.clear();
        self.early_cmd_end.clear();
        self.early_glaunch.clear();
        self.snapshot_counters();
    }

    pub fn snapshot_counters(&mut self) {
        self.stats.rows_per_table =
            self.table_names.iter().cloned().zip(self.rows.iter().copied()).filter(|(_, n)| *n > 0).collect();
        self.stats.updates_per_kind =
            self.update_names.iter().cloned().zip(self.upds.iter().copied()).filter(|(_, n)| *n > 0).collect();
    }

    pub fn pending(&self) -> usize {
        self.pending_kern.len() + self.pending_node.values().map(Vec::len).sum::<usize>() + self.cmds.len()
    }
}

#[cfg(test)]
mod tests {
    //! Rows and their deferred GPU times in both arrival orders (live snapshots deliver either first).
    use super::*;
    use nvtrace_sys::{Column, ColumnRole, ColumnType, Manifest, Shard, Window};

    const TABLES: &[(&str, &[&str])] = &[
        ("processes", &["process_id_unique", "process_id_namespace", "path", "process_id_host"]),
        ("thread_event_threads", &["process_id_unique", "thread_id_unique", "thread_id_native"]),
        ("cuda_functions", &["process_id_unique", "cufunction_id", "name_mangled"]),
        ("cuda_graph_nodes", &["process_id_unique", "graph_node_correlation_id", "cugraph_exec_id", "cugraph_node_id"]),
        (
            "cuda_driver_api_calls",
            &["process_id_unique", "thread_id_unique", "driver_api_correlation_id", "start_utc_ns", "end_utc_ns", "driver_api_function_name"],
        ),
        (
            "cuda_runtime_api_calls",
            &["process_id_unique", "thread_id_unique", "runtime_api_correlation_id", "start_utc_ns", "end_utc_ns", "runtime_api_function_name"],
        ),
        (
            "cuda_graph_launches",
            &[
                "process_id_unique",
                "gpu_uuid",
                "hw_correlation_id",
                "custream_id",
                "cucontext_id",
                "cugraph_exec_id",
                "driver_api_correlation_id",
                "timestamp_gpu_ns",
            ],
        ),
        (
            "cuda_immediate_kernel_launches",
            &[
                "process_id_unique",
                "gpu_uuid",
                "cudevice_ordinal",
                "hw_correlation_id",
                "driver_api_correlation_id",
                "custream_id",
                "cucontext_id",
                "cufunction_id",
                "grid_dim_x",
                "grid_dim_y",
                "grid_dim_z",
                "block_dim_x",
                "block_dim_y",
                "block_dim_z",
                "shmem_per_block_bytes_total",
                "gpu_start_utc_ns",
                "gpu_end_utc_ns",
            ],
        ),
        (
            "cuda_graph_node_launches",
            &[
                "process_id_unique",
                "gpu_uuid",
                "cudevice_ordinal",
                "graph_node_correlation_id",
                "start_timestamp_gpu_ns",
                "cufunction_id",
                "grid_dim_x",
                "grid_dim_y",
                "grid_dim_z",
                "block_dim_x",
                "block_dim_y",
                "block_dim_z",
                "gpu_start_utc_ns",
                "gpu_end_utc_ns",
            ],
        ),
        (
            "cuda_gpu_commands",
            &[
                "process_id_unique",
                "gpu_uuid",
                "cudevice_ordinal",
                "start_gpu_timestamp_report_ref",
                "end_gpu_timestamp_report_ref",
                "custream_id",
                "cucontext_id",
                "gpu_start_utc_ns",
                "gpu_end_utc_ns",
                "graph_node_correlation_id",
                "driver_api_correlation_id",
                "name_display",
            ],
        ),
        ("cuda_gpu_memcpy_details", &["process_id_unique", "gpu_uuid", "start_gpu_timestamp_report_ref", "byte_count", "copy_direction"]),
        ("cuda_gpu_memset_details", &["process_id_unique", "gpu_uuid", "start_gpu_timestamp_report_ref", "byte_count"]),
        ("gpu_partitions", &["gpu_uuid", "chip_name", "device_name"]),
        ("gpu_metrics", &["gpu_uuid", "gpu_end_utc_ns", "metric_name", "metric_value"]),
    ];
    /// (update, table, match columns, latest column, update columns)
    type UpdateDef = (&'static str, &'static str, &'static [&'static str], Option<&'static str>, &'static [&'static str]);
    const UPDATES: &[UpdateDef] = &[
        (
            "cuda_immediate_kernel_launch_gpu_times",
            "cuda_immediate_kernel_launches",
            &["process_id_unique", "gpu_uuid", "hw_correlation_id"],
            None,
            &["gpu_start_utc_ns", "gpu_end_utc_ns"],
        ),
        (
            "cuda_gpu_command_gpu_start",
            "cuda_gpu_commands",
            &["process_id_unique", "gpu_uuid", "start_gpu_timestamp_report_ref"],
            None,
            &["gpu_start_utc_ns"],
        ),
        (
            "cuda_gpu_command_gpu_end",
            "cuda_gpu_commands",
            &["process_id_unique", "gpu_uuid", "end_gpu_timestamp_report_ref"],
            None,
            &["gpu_end_utc_ns"],
        ),
        ("cuda_graph_launch_timestamp", "cuda_graph_launches", &["process_id_unique", "gpu_uuid", "hw_correlation_id"], None, &["timestamp_gpu_ns"]),
        (
            "cuda_graph_node_launch_gpu_end",
            "cuda_graph_node_launches",
            &["process_id_unique", "gpu_uuid", "graph_node_correlation_id"],
            Some("start_timestamp_gpu_ns"),
            &["gpu_end_utc_ns"],
        ),
    ];
    const GPU: Uuid = [7; 16];

    fn schema() -> Schema {
        let tables: Vec<Table> = TABLES
            .iter()
            .map(|(n, cols)| Table {
                name: (*n).to_string(),
                columns: cols.iter().map(|c| Column { name: (*c).to_string(), ty: ColumnType::UInt64, role: ColumnRole::Nullable }).collect(),
            })
            .collect();
        let updates = UPDATES
            .iter()
            .map(|(n, t, m, l, u)| Update {
                name: (*n).to_string(),
                table: tables.iter().position(|x| x.name == *t).unwrap_or(usize::MAX),
                match_columns: m.iter().map(|c| (*c).to_string()).collect(),
                latest_column: l.map(str::to_string),
                update_columns: u.iter().map(|c| (*c).to_string()).collect(),
            })
            .collect();
        Schema { version: 0, tables, updates }
    }

    /// Feeds snapshots into a mapper and collects what it emits.
    struct Run {
        s: Schema,
        m: Mapper,
        out: Vec<Activity>,
        snap: u64,
        shard: Option<Shard>,
    }

    impl Run {
        fn new() -> Self {
            let s = schema();
            let m = Mapper::new(&s).expect("test schema has every table the mapper needs");
            Self { s, m, out: Vec::new(), snap: 0, shard: None }
        }
        fn ev(&mut self, ev: Event<'_>) {
            let out = &mut self.out;
            self.m.on_event(ev, &mut |a| out.push(a));
        }
        fn begin(&mut self) {
            self.snap += 1;
            let w = Window { snapshot: self.snap, start_utc_ns: 0, end_utc_ns: 0, manifest: Some(Manifest::default()), shard: self.shard };
            self.ev(Event::Begin(w));
        }
        fn commit(&mut self) {
            let w = Window { snapshot: self.snap, start_utc_ns: 0, end_utc_ns: 0, manifest: None, shard: None };
            self.ev(Event::Commit(w));
        }
        fn snapshot(&mut self, f: impl FnOnce(&mut Self)) {
            self.begin();
            f(self);
            self.commit();
        }
        fn row(&mut self, table: &str, kv: &[(&str, Value<'static>)]) {
            let (ti, t) = self.s.table(table).expect("table in the test schema");
            let mut v = vec![Value::Null; t.columns.len()];
            for (k, x) in kv {
                v[t.col(k).expect("column in the test schema")] = *x;
            }
            self.ev(Event::Row { table: ti, values: &v });
        }
        fn update(&mut self, name: &str, kv: &[(&str, Value<'static>)]) {
            let (ui, u) = self.s.update(name).expect("update in the test schema");
            let names: Vec<&str> =
                u.match_columns.iter().chain(u.latest_column.iter()).chain(u.update_columns.iter()).map(String::as_str).collect();
            let v: Vec<Value<'static>> =
                names.iter().map(|n| kv.iter().find(|(k, _)| k == n).map(|(_, x)| *x).unwrap_or(Value::Null)).collect();
            self.ev(Event::Update { update: ui, values: &v });
        }
        fn kernel_row(&mut self, hw: u64, start: Option<u64>, end: Option<u64>) {
            let opt = |x: Option<u64>| x.map(Value::U64).unwrap_or(Value::Null);
            self.row(
                "cuda_immediate_kernel_launches",
                &[
                    ("process_id_unique", Value::U64(1)),
                    ("gpu_uuid", Value::Uuid(GPU)),
                    ("hw_correlation_id", Value::U64(hw)),
                    ("gpu_start_utc_ns", opt(start)),
                    ("gpu_end_utc_ns", opt(end)),
                ],
            );
        }
        fn kernel_times(&mut self, hw: u64, start: Option<u64>, end: Option<u64>) {
            let opt = |x: Option<u64>| x.map(Value::U64).unwrap_or(Value::Null);
            self.update(
                "cuda_immediate_kernel_launch_gpu_times",
                &[
                    ("process_id_unique", Value::U64(1)),
                    ("gpu_uuid", Value::Uuid(GPU)),
                    ("hw_correlation_id", Value::U64(hw)),
                    ("gpu_start_utc_ns", opt(start)),
                    ("gpu_end_utc_ns", opt(end)),
                ],
            );
        }
        fn node_row(&mut self, nc: u64, start_gpu: u64, start: u64) {
            self.row(
                "cuda_graph_node_launches",
                &[
                    ("process_id_unique", Value::U64(1)),
                    ("gpu_uuid", Value::Uuid(GPU)),
                    ("graph_node_correlation_id", Value::U64(nc)),
                    ("start_timestamp_gpu_ns", Value::U64(start_gpu)),
                    ("gpu_start_utc_ns", Value::U64(start)),
                ],
            );
        }
        fn node_end(&mut self, nc: u64, end_gpu: u64, end: u64) {
            self.update(
                "cuda_graph_node_launch_gpu_end",
                &[
                    ("process_id_unique", Value::U64(1)),
                    ("gpu_uuid", Value::Uuid(GPU)),
                    ("graph_node_correlation_id", Value::U64(nc)),
                    ("start_timestamp_gpu_ns", Value::U64(end_gpu)),
                    ("gpu_end_utc_ns", Value::U64(end)),
                ],
            );
        }
        fn times(&self) -> Vec<(i64, i64)> {
            self.out.iter().map(|a| (a.start_ns, a.end_ns)).collect()
        }
    }

    #[test]
    fn kernel_times_after_the_row() {
        let mut r = Run::new();
        r.snapshot(|r| r.kernel_row(7, Some(100), None));
        r.snapshot(|r| r.kernel_times(7, Some(100), Some(200)));
        assert_eq!(r.times(), vec![(100, 200)]);
        assert_eq!((r.m.stats.lost_incomplete, r.m.stats.orphan_updates), (0, 0));
    }

    #[test]
    fn kernel_times_before_the_row() {
        let mut r = Run::new();
        r.snapshot(|r| r.kernel_times(7, Some(100), Some(200)));
        r.snapshot(|r| r.kernel_row(7, None, None));
        assert_eq!(r.times(), vec![(100, 200)]);
        assert_eq!((r.m.stats.lost_incomplete, r.m.stats.orphan_updates, r.m.stats.completed_by_early_update), (0, 0, 1));
    }

    #[test]
    fn kernel_start_and_end_in_separate_snapshots_before_the_row() {
        let mut r = Run::new();
        r.snapshot(|r| r.kernel_times(7, Some(100), None));
        r.snapshot(|r| r.kernel_times(7, None, Some(200)));
        r.snapshot(|r| r.kernel_row(7, None, None));
        assert_eq!(r.times(), vec![(100, 200)]);
        assert_eq!(r.m.stats.lost_incomplete, 0);
    }

    #[test]
    fn graph_node_end_before_the_row_picks_the_first_end_after_its_start() {
        let mut r = Run::new();
        // Two ends of node 5 arrive early; the row (GPU start 900) belongs to the end at 1000, not the one at 500.
        r.snapshot(|r| {
            r.node_end(5, 500, 2500);
            r.node_end(5, 1000, 3000);
        });
        r.snapshot(|r| r.node_row(5, 900, 2900));
        assert_eq!(r.times(), vec![(2900, 3000)]);
        for _ in 0..PENDING_SNAPSHOTS {
            r.snapshot(|_| {});
        }
        // The unmatched end at 500 expires as an orphan; nothing was lost.
        assert_eq!((r.m.stats.lost_incomplete, r.m.stats.orphan_updates), (0, 1));
    }

    #[test]
    fn graph_node_end_after_the_row() {
        let mut r = Run::new();
        r.snapshot(|r| r.node_row(5, 900, 2900));
        r.snapshot(|r| r.node_end(5, 1000, 3000));
        assert_eq!(r.times(), vec![(2900, 3000)]);
        assert_eq!(r.m.stats.lost_incomplete, 0);
    }

    #[test]
    fn unmatched_rows_and_updates_expire() {
        let mut r = Run::new();
        r.snapshot(|r| {
            r.kernel_row(1, None, None);
            r.kernel_times(2, Some(10), Some(20));
        });
        for _ in 0..PENDING_SNAPSHOTS {
            r.snapshot(|_| {});
        }
        assert!(r.out.is_empty());
        assert_eq!((r.m.stats.lost_incomplete, r.m.stats.orphan_updates), (1, 1));
    }

    #[test]
    fn graph_completeness_counts_missing_node_launches() {
        let mut r = Run::new();
        r.m.diag_on = true;
        let p = ("process_id_unique", Value::U64(1));
        let launch = |r: &mut Run, hw: u64, t: u64| {
            r.row(
                "cuda_graph_launches",
                &[p, ("gpu_uuid", Value::Uuid(GPU)), ("hw_correlation_id", Value::U64(hw)), ("cugraph_exec_id", Value::U64(9)), ("timestamp_gpu_ns", Value::U64(t))],
            );
        };
        r.snapshot(|r| {
            for (nc, node) in [(5, 1), (6, 2)] {
                r.row("cuda_graph_nodes", &[p, ("graph_node_correlation_id", Value::U64(nc)), ("cugraph_exec_id", Value::U64(9)), ("cugraph_node_id", Value::U64(node))]);
            }
            launch(r, 1, 90);
            r.node_row(5, 100, 1100);
            r.node_row(6, 110, 1110);
        });
        // The second launch lost node 6; repeating a row from the previous snapshot must not count twice.
        r.snapshot(|r| {
            launch(r, 2, 190);
            r.node_row(6, 110, 1110);
            r.node_row(5, 200, 1200);
        });
        let d = &r.m.diag;
        assert_eq!((d.graph_launches, d.graph_node_rows, d.graph_node_expected, d.graph_execs_short), (2, 3, 4, 1));
    }

    #[test]
    fn shard_ignores_other_gpus_and_cpu_rows() {
        const OTHER: [u8; 16] = [9; 16];
        let mut r = Run::new();
        r.shard = Some(Shard::new(1, 2, &[GPU], &[]));
        r.snapshot(|r| {
            r.kernel_row(7, Some(100), Some(200));
            r.row(
                "cuda_immediate_kernel_launches",
                &[("process_id_unique", Value::U64(1)), ("gpu_uuid", Value::Uuid(OTHER)), ("hw_correlation_id", Value::U64(8))],
            );
            r.row("cuda_driver_api_calls", &[("process_id_unique", Value::U64(1)), ("start_utc_ns", Value::U64(5)), ("end_utc_ns", Value::U64(6))]);
        });
        for _ in 0..PENDING_SNAPSHOTS {
            r.snapshot(|_| {});
        }
        // Only the kernel on this shard's GPU; the other GPU's row (no GPU times here) is neither mapped nor lost.
        assert_eq!(r.times(), vec![(100, 200)]);
        assert_eq!((r.m.stats.lost_incomplete, r.m.stats.driver_api), (0, 0));
        // A process this shard owns exclusively: its API calls are mapped here.
        r.shard = Some(Shard::new(1, 2, &[GPU], &[1]));
        r.snapshot(|r| {
            r.row("cuda_driver_api_calls", &[("process_id_unique", Value::U64(1)), ("start_utc_ns", Value::U64(5)), ("end_utc_ns", Value::U64(6))])
        });
        assert_eq!(r.m.stats.driver_api, 1);
    }

    #[test]
    fn memcpy_times_before_the_command_row() {
        let mut r = Run::new();
        let pg = [("process_id_unique", Value::U64(1)), ("gpu_uuid", Value::Uuid(GPU))];
        r.snapshot(|r| {
            r.update("cuda_gpu_command_gpu_start", &[pg[0], pg[1], ("start_gpu_timestamp_report_ref", Value::U64(11)), ("gpu_start_utc_ns", Value::U64(500))]);
            r.update("cuda_gpu_command_gpu_end", &[pg[0], pg[1], ("end_gpu_timestamp_report_ref", Value::U64(12)), ("gpu_end_utc_ns", Value::U64(600))]);
        });
        r.snapshot(|r| {
            r.row(
                "cuda_gpu_commands",
                &[
                    pg[0],
                    pg[1],
                    ("start_gpu_timestamp_report_ref", Value::U64(11)),
                    ("end_gpu_timestamp_report_ref", Value::U64(12)),
                    ("name_display", Value::Str("Memcpy HtoD")),
                ],
            );
            r.row("cuda_gpu_memcpy_details", &[pg[0], pg[1], ("start_gpu_timestamp_report_ref", Value::U64(11)), ("byte_count", Value::U64(64))]);
        });
        assert_eq!(r.times(), vec![(500, 600)]);
        assert_eq!(r.out[0].bytes, Some(64));
        assert_eq!((r.m.stats.lost_incomplete, r.m.stats.orphan_updates), (0, 0));
    }
}
