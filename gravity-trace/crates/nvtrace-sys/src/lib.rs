//! Bindings to the NVIDIA Nsight Perf SDK for AI (NvTrace) table exporter.
//!
//! The SDK decodes a Trace Stream Database (TSF, written by `nvtrace_streaming_consumer` or kept by a
//! continuous snapshot capture) into a data model, and its `TableExporter` turns that model into
//! database-agnostic rows (schema: `NvTraceTableSchema.h`, mirrored at runtime by [`schema()`]).
//! A small C++ shim (`shim/nvtrace_shim.cpp`) forwards every row to a Rust [`Sink`]:
//!
//! - [`export_dir`]: one pass over an on-disk TSF database;
//! - [`Live`]: continuous snapshots of the running `nvidia-nsight-perf` service; each completed snapshot is
//!   delivered on the SDK's export thread.
//!
//! Rows of one snapshot arrive between [`Event::Begin`] and [`Event::Commit`], tables in the exporter's
//! order (processes and threads, CUDA objects, API calls, graph launches, kernels, GPU commands, NVTX, GPU
//! partitions, GR contexts, PM samples). [`Event::Update`]s fill in values (usually GPU end times) of a row
//! that was emitted in an earlier snapshot.
// Built without the SDK (NVTRACE_SDK_DIR unset, e.g. on macOS) every entry point fails at run time and the FFI
// plumbing behind them has no user.
#![cfg_attr(not(nvtrace_sdk), allow(dead_code, unused_imports))]
use snafu::Snafu;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[allow(non_camel_case_types, dead_code)]
mod ffi {
    use std::ffi::{c_char, c_int, c_void};

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct nvt_str {
        pub ptr: *const c_char,
        pub len: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub union nvt_union {
        pub u: u64,
        pub i: i64,
        pub f: f64,
        pub s: nvt_str,
        pub uuid: [u8; 16],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct nvt_value {
        pub tag: u8,
        pub pad: [u8; 7],
        pub v: nvt_union,
    }

    pub type nvt_callback =
        unsafe extern "C" fn(user: *mut c_void, event: u32, index: u32, values: *const nvt_value, count: usize) -> c_int;

    #[repr(C)]
    #[derive(Default, Clone, Copy, Debug)]
    pub struct nvt_export_stats {
        pub rows: u64,
        pub updates: u64,
        pub observation_start_utc_ns: u64,
        pub observation_end_utc_ns: u64,
        pub data_blocks: u64,
    }

    #[repr(C)]
    pub struct nvt_live_options {
        pub work_dir: *const c_char,
        pub snapshot_ms: u64,
        pub data_rate_mib_per_s: u64,
        pub data_block_size_mib: u32,
        pub latency_blocks: u32,
        pub max_attach_mib: u64,
        pub max_retained_shm_mib: u64,
        pub ignore_mask: u32,
        pub shards: u32,
        pub in_memory: u32,
    }

    #[repr(C)]
    #[derive(Default, Clone, Copy, Debug)]
    pub struct nvt_live_stats {
        pub snapshots_ok: u64,
        pub snapshots_failed: u64,
        pub rows: u64,
        pub updates: u64,
        pub last_export_ns: u64,
        pub total_export_ns: u64,
        pub in_memory: u32,
    }

    #[repr(C)]
    pub struct nvt_live {
        _opaque: [u8; 0],
    }

    #[cfg(nvtrace_sdk)]
    unsafe extern "C" {
        pub fn nvt_schema_version() -> i32;
        pub fn nvt_table_count() -> usize;
        pub fn nvt_table_name(t: usize) -> *const c_char;
        pub fn nvt_table_column_count(t: usize) -> usize;
        pub fn nvt_table_column_name(t: usize, c: usize) -> *const c_char;
        pub fn nvt_table_column_type(t: usize, c: usize) -> u8;
        pub fn nvt_table_column_role(t: usize, c: usize) -> u8;
        pub fn nvt_update_count() -> usize;
        pub fn nvt_update_name(u: usize) -> *const c_char;
        pub fn nvt_update_table(u: usize) -> usize;
        pub fn nvt_update_match_count(u: usize) -> usize;
        pub fn nvt_update_match_column(u: usize, i: usize) -> *const c_char;
        pub fn nvt_update_latest_column(u: usize) -> *const c_char;
        pub fn nvt_update_value_count(u: usize) -> usize;
        pub fn nvt_update_value_column(u: usize, i: usize) -> *const c_char;
        pub fn nvt_export_tsf_dir(
            dir: *const c_char,
            start_utc_ns: u64,
            end_utc_ns: u64,
            ignore_mask: u32,
            cb: nvt_callback,
            user: *mut c_void,
            out_stats: *mut nvt_export_stats,
            err: *mut c_char,
            err_len: usize,
        ) -> c_int;
        pub fn nvt_live_start(
            opts: *const nvt_live_options,
            cb: nvt_callback,
            shard_users: *const *mut c_void,
            shard_user_count: u32,
            done_user: *mut c_void,
            err: *mut c_char,
            err_len: usize,
        ) -> *mut nvt_live;
        pub fn nvt_live_stats_get(live: *const nvt_live, out: *mut nvt_live_stats);
        pub fn nvt_live_stop(live: *mut nvt_live, err: *mut c_char, err_len: usize) -> c_int;
    }
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("gt was built without the Nsight Perf SDK (set NVTRACE_SDK_DIR when building)"))]
    Unavailable,
    #[snafu(display("path {} is not valid UTF-8 / contains NUL", path.display()))]
    BadPath { path: PathBuf },
    #[snafu(display("NvTrace export of {} failed (code {code}): {message}", path.display()))]
    Export { path: PathBuf, code: i32, message: String },
    #[snafu(display("NvTrace live capture: {message}"))]
    Live { message: String },
    #[snafu(display("a sink callback panicked"))]
    SinkPanicked,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// `nv::trace::TableColumnType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnType {
    Bool,
    Int8,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Double,
    String,
    Uuid,
    Unknown(u8),
}

impl ColumnType {
    fn from_raw(v: u8) -> Self {
        use ColumnType::*;
        [Bool, Int8, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Double, String, Uuid].get(v as usize).copied().unwrap_or(Unknown(v))
    }
}

/// `nv::trace::TableColumnRole`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnRole {
    Required,
    Nullable,
    PrimaryKey,
    Unknown(u8),
}

#[derive(Clone, Debug)]
pub struct Column {
    pub name: String,
    pub ty: ColumnType,
    pub role: ColumnRole,
}

#[derive(Clone, Debug)]
pub struct Table {
    pub name: String,
    pub columns: Vec<Column>,
}

impl Table {
    /// Index of a column by name.
    pub fn col(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }
}

/// A deferred update: rows of `table` matching `match_columns` (and, with `latest_column`, the greatest value
/// of that column not above the bound) get `update_columns` set.
#[derive(Clone, Debug)]
pub struct Update {
    pub name: String,
    pub table: usize,
    pub match_columns: Vec<String>,
    pub latest_column: Option<String>,
    pub update_columns: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Schema {
    pub version: i32,
    pub tables: Vec<Table>,
    pub updates: Vec<Update>,
}

impl Schema {
    pub fn table(&self, name: &str) -> Option<(usize, &Table)> {
        self.tables.iter().enumerate().find(|(_, t)| t.name == name)
    }

    pub fn update(&self, name: &str) -> Option<(usize, &Update)> {
        self.updates.iter().enumerate().find(|(_, u)| u.name == name)
    }
}

#[cfg(nvtrace_sdk)]
fn cstr(p: *const c_char) -> Option<String> {
    // SAFETY: the shim returns pointers to static NUL-terminated schema strings, or NULL.
    (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

/// The SDK's table schema (static; built once).
pub fn schema() -> Result<&'static Schema> {
    static S: OnceLock<Option<Schema>> = OnceLock::new();
    S.get_or_init(load_schema).as_ref().ok_or(Error::Unavailable)
}

#[cfg(not(nvtrace_sdk))]
fn load_schema() -> Option<Schema> {
    None
}

#[cfg(nvtrace_sdk)]
fn load_schema() -> Option<Schema> {
    use ffi::*;
    // SAFETY: plain getters over the SDK's static schema tables; indices are in range.
    unsafe {
        let tables = (0..nvt_table_count())
            .map(|t| Table {
                name: cstr(nvt_table_name(t)).unwrap_or_default(),
                columns: (0..nvt_table_column_count(t))
                    .map(|c| Column {
                        name: cstr(nvt_table_column_name(t, c)).unwrap_or_default(),
                        ty: ColumnType::from_raw(nvt_table_column_type(t, c)),
                        role: match nvt_table_column_role(t, c) {
                            0 => ColumnRole::Required,
                            1 => ColumnRole::Nullable,
                            2 => ColumnRole::PrimaryKey,
                            r => ColumnRole::Unknown(r),
                        },
                    })
                    .collect(),
            })
            .collect();
        let updates = (0..nvt_update_count())
            .map(|u| Update {
                name: cstr(nvt_update_name(u)).unwrap_or_default(),
                table: nvt_update_table(u),
                match_columns: (0..nvt_update_match_count(u)).filter_map(|i| cstr(nvt_update_match_column(u, i))).collect(),
                latest_column: cstr(nvt_update_latest_column(u)),
                update_columns: (0..nvt_update_value_count(u)).filter_map(|i| cstr(nvt_update_value_column(u, i))).collect(),
            })
            .collect();
        Some(Schema { version: nvt_schema_version(), tables, updates })
    }
}

/// One cell. Integer widths are collapsed (the schema keeps the declared type).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value<'a> {
    Null,
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    Str(&'a str),
    Uuid([u8; 16]),
}

impl<'a> Value<'a> {
    pub fn u64(&self) -> Option<u64> {
        match *self {
            Value::U64(v) => Some(v),
            Value::I64(v) => u64::try_from(v).ok(),
            Value::Bool(b) => Some(u64::from(b)),
            _ => None,
        }
    }

    pub fn i64(&self) -> Option<i64> {
        match *self {
            Value::I64(v) => Some(v),
            Value::U64(v) => i64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn f64(&self) -> Option<f64> {
        match *self {
            Value::F64(v) => Some(v),
            Value::U64(v) => Some(v as f64),
            Value::I64(v) => Some(v as f64),
            _ => None,
        }
    }

    pub fn str(&self) -> Option<&'a str> {
        match *self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn uuid(&self) -> Option<[u8; 16]> {
        match *self {
            Value::Uuid(u) => Some(u),
            _ => None,
        }
    }

    pub fn bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            Value::U64(v) => Some(v != 0),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// SAFETY: `v` comes from the shim; string pointers are valid for the callback's duration.
    unsafe fn from_ffi(v: &ffi::nvt_value) -> Value<'a> {
        // SAFETY: the tag says which union member the shim wrote.
        unsafe {
            match v.tag {
                1 => Value::Bool(v.v.u != 0),
                2..=4 => Value::I64(v.v.i),
                5..=8 => Value::U64(v.v.u),
                9 => Value::F64(v.v.f),
                10 => {
                    let s = v.v.s;
                    if s.ptr.is_null() {
                        Value::Str("")
                    } else {
                        let b = std::slice::from_raw_parts(s.ptr.cast::<u8>(), s.len);
                        Value::Str(std::str::from_utf8(b).unwrap_or("<non-utf8>"))
                    }
                }
                11 => Value::Uuid(v.v.uuid),
                _ => Value::Null,
            }
        }
    }
}

/// Snapshot window markers carry `[snapshot_number, observation_start_utc_ns, observation_end_utc_ns]`; the begin
/// marker of a live snapshot also carries its manifest and its shard.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub snapshot: u64,
    pub start_utc_ns: u64,
    pub end_utc_ns: u64,
    pub manifest: Option<Manifest>,
    pub shard: Option<Shard>,
}

/// Most GPUs / processes one shard reports as its own.
pub const MAX_SHARD_GPUS: usize = 16;
pub const MAX_SHARD_PROCESSES: usize = 64;

/// The part of a live snapshot one export shard decoded: the hardware data of `gpus`, and the CPU-side data of the
/// processes using them (`processes` reach only this shard) plus of processes not yet seen using any GPU.
#[derive(Clone, Copy, Debug)]
pub struct Shard {
    pub index: u32,
    pub count: u32,
    gpus: [[u8; 16]; MAX_SHARD_GPUS],
    n_gpus: usize,
    processes: [u64; MAX_SHARD_PROCESSES],
    n_processes: usize,
}

impl Shard {
    pub fn new(index: u32, count: u32, gpus: &[[u8; 16]], processes: &[u64]) -> Self {
        let mut s = Shard {
            index,
            count,
            gpus: [[0; 16]; MAX_SHARD_GPUS],
            n_gpus: 0,
            processes: [0; MAX_SHARD_PROCESSES],
            n_processes: 0,
        };
        for g in gpus.iter().take(MAX_SHARD_GPUS) {
            s.gpus[s.n_gpus] = *g;
            s.n_gpus += 1;
        }
        for p in processes.iter().take(MAX_SHARD_PROCESSES) {
            s.processes[s.n_processes] = *p;
            s.n_processes += 1;
        }
        s
    }

    /// GPUs whose hardware trace this shard decoded.
    pub fn gpus(&self) -> &[[u8; 16]] {
        &self.gpus[..self.n_gpus]
    }

    /// `process_id_unique` of the processes whose CPU-side rows reach only this shard.
    pub fn processes(&self) -> &[u64] {
        &self.processes[..self.n_processes]
    }
}

/// `nv::trace::ContinuousSnapshotManifest` of a live snapshot (what the SDK reports about its own completeness).
#[derive(Clone, Copy, Debug, Default)]
pub struct Manifest {
    pub requested_begin_utc_ns: u64,
    pub cutoff_utc_ns: u64,
    pub actual_begin_utc_ns: u64,
    pub actual_end_utc_ns: u64,
    /// Records met while every latency-buffer slot was busy (kept unless the provider ring lapped them).
    pub backpressured_records: u64,
    /// The window is shorter than the history duration (snapshot interval): the bank filled and the SDK cut the
    /// snapshot early. Consecutive windows still abut, so this is not data loss by itself.
    pub history_truncated: bool,
    pub attach_complete: bool,
    pub dropped_attach_records: u64,
    pub data_blocks: u64,
    /// Wall clock when the SDK's export thread handed this snapshot to us (it queues behind the previous one).
    pub export_entry_utc_ns: u64,
    /// Time the SDK took to build the data model from the snapshot's data blocks.
    pub build_ns: u64,
}

#[derive(Debug)]
pub enum Event<'a> {
    Begin(Window),
    Row { table: usize, values: &'a [Value<'a>] },
    Update { update: usize, values: &'a [Value<'a>] },
    Commit(Window),
    /// Live: every shard of the snapshot committed (sent to the coordinating sink only).
    Done(Window),
}

/// Receives the exported rows. Return `false` to abort.
pub trait Sink {
    fn event(&mut self, ev: Event<'_>) -> bool;
}

struct Tramp<'s> {
    sink: &'s mut dyn Sink,
    buf: Vec<Value<'static>>,
    panicked: bool,
}

/// C callback: converts the values and calls the sink; a panic is caught (it must not unwind into C++).
unsafe extern "C" fn trampoline(user: *mut c_void, event: u32, index: u32, values: *const ffi::nvt_value, count: usize) -> c_int {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: `user` is the `Tramp` passed to the shim, alive for the whole export / live capture.
        let t = unsafe { &mut *user.cast::<Tramp<'_>>() };
        let raw: &[ffi::nvt_value] =
            if values.is_null() || count == 0 { &[] } else { unsafe { std::slice::from_raw_parts(values, count) } };
        t.buf.clear();
        // SAFETY: values (and the strings they point to) are valid until this callback returns; `buf` is cleared
        // before the next call, so the 'static lifetime never escapes it.
        t.buf.extend(raw.iter().map(|v| unsafe { Value::from_ffi(v) }));
        let at = |i: usize| t.buf.get(i).and_then(Value::u64).unwrap_or(0);
        let window = || Window {
            snapshot: at(0),
            start_utc_ns: at(1),
            end_utc_ns: at(2),
            shard: (t.buf.len() >= 16).then(|| {
                let gpus: Vec<[u8; 16]> = t.buf[16..].iter().filter_map(Value::uuid).collect();
                let procs: Vec<u64> = t.buf[16..].iter().filter(|v| matches!(v, Value::U64(_))).filter_map(Value::u64).collect();
                Shard::new(at(14) as u32, at(15) as u32, &gpus, &procs)
            }),
            manifest: (t.buf.len() >= 12).then(|| Manifest {
                requested_begin_utc_ns: at(3),
                cutoff_utc_ns: at(4),
                actual_begin_utc_ns: at(5),
                actual_end_utc_ns: at(6),
                backpressured_records: at(7),
                history_truncated: at(8) != 0,
                attach_complete: at(9) != 0,
                dropped_attach_records: at(10),
                data_blocks: at(11),
                export_entry_utc_ns: at(12),
                build_ns: at(13),
            }),
        };
        let ev = match event {
            0 => Event::Row { table: index as usize, values: &t.buf },
            1 => Event::Update { update: index as usize, values: &t.buf },
            2 => Event::Begin(window()),
            3 => Event::Commit(window()),
            _ => Event::Done(window()),
        };
        t.sink.event(ev)
    }));
    match r {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(_) => {
            // SAFETY: as above.
            unsafe { (*user.cast::<Tramp<'_>>()).panicked = true };
            1
        }
    }
}

/// Provider types to skip while decoding.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ignore {
    pub gpu_pm: bool,
    pub gpu_hwt: bool,
    pub cuda_api: bool,
    pub nvtx: bool,
}

impl Ignore {
    pub fn mask(self) -> u32 {
        u32::from(self.gpu_pm) | (u32::from(self.gpu_hwt) << 1) | (u32::from(self.cuda_api) << 2) | (u32::from(self.nvtx) << 3)
    }

    /// Parse `gpu_pm,gpu_hwt,cuda_api,nvtx` (the SDK tools' `--ignore` syntax).
    pub fn parse(s: &str) -> Option<Self> {
        let mut i = Ignore::default();
        for p in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            match p {
                "gpu_pm" => i.gpu_pm = true,
                "gpu_hwt" => i.gpu_hwt = true,
                "cuda_api" => i.cuda_api = true,
                "nvtx" => i.nvtx = true,
                _ => return None,
            }
        }
        Some(i)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ExportOptions {
    /// Observation window bounds (UTC ns); 0 = unbounded.
    pub start_utc_ns: u64,
    pub end_utc_ns: u64,
    pub ignore: Ignore,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ExportStats {
    pub rows: u64,
    pub updates: u64,
    pub observation_start_utc_ns: u64,
    pub observation_end_utc_ns: u64,
}

fn c_path(p: &Path) -> Result<std::ffi::CString> {
    p.to_str().and_then(|s| std::ffi::CString::new(s).ok()).ok_or_else(|| Error::BadPath { path: p.to_path_buf() })
}

fn err_string(buf: &[u8]) -> String {
    let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

/// Decode the TSF database in `dir` (a `nvtrace_streaming_consumer` output directory) into `sink`.
#[cfg(nvtrace_sdk)]
pub fn export_dir(dir: &Path, opts: &ExportOptions, sink: &mut dyn Sink) -> Result<ExportStats> {
    let cdir = c_path(dir)?;
    let mut t = Tramp { sink, buf: Vec::with_capacity(32), panicked: false };
    let mut st = ffi::nvt_export_stats::default();
    let mut err = [0u8; 512];
    // SAFETY: all pointers outlive the call; the trampoline matches nvt_callback.
    let rc = unsafe {
        ffi::nvt_export_tsf_dir(
            cdir.as_ptr(),
            opts.start_utc_ns,
            opts.end_utc_ns,
            opts.ignore.mask(),
            trampoline,
            (&raw mut t).cast::<c_void>(),
            &raw mut st,
            err.as_mut_ptr().cast::<c_char>(),
            err.len(),
        )
    };
    if t.panicked {
        return Err(Error::SinkPanicked);
    }
    if rc != 0 {
        return Err(Error::Export { path: dir.to_path_buf(), code: rc, message: err_string(&err) });
    }
    Ok(ExportStats {
        rows: st.rows,
        updates: st.updates,
        observation_start_utc_ns: st.observation_start_utc_ns,
        observation_end_utc_ns: st.observation_end_utc_ns,
    })
}

#[cfg(not(nvtrace_sdk))]
pub fn export_dir(_dir: &Path, _opts: &ExportOptions, _sink: &mut dyn Sink) -> Result<ExportStats> {
    Err(Error::Unavailable)
}

/// Continuous snapshot capture settings (the SDK's `nvtrace_continuous_export` options).
#[derive(Clone, Debug)]
pub struct LiveOptions {
    pub work_dir: PathBuf,
    pub snapshot_ms: u64,
    pub data_rate_mib_per_s: u64,
    pub data_block_size_mib: u32,
    pub latency_blocks: u32,
    pub max_attach_mib: u64,
    pub max_retained_shm_mib: u64,
    pub ignore: Ignore,
    /// Parallel export shards (each owns whole GPUs); 0 = one per GPU partition, capped by the sinks given.
    pub shards: u32,
    /// Keep the snapshot data blocks in memory (anonymous pages, released after each export) instead of files in
    /// `work_dir`. Falls back to files when the SDK patch did not apply ([`LiveStats::in_memory`] tells).
    pub in_memory: bool,
}

impl Default for LiveOptions {
    fn default() -> Self {
        Self {
            work_dir: PathBuf::from("/tmp/gravity-trace/live"),
            snapshot_ms: 5000,
            data_rate_mib_per_s: 128,
            data_block_size_mib: 10,
            latency_blocks: 8,
            max_attach_mib: 1024,
            max_retained_shm_mib: 2048,
            ignore: Ignore::default(),
            shards: 0,
            in_memory: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LiveStats {
    pub snapshots_ok: u64,
    pub snapshots_failed: u64,
    pub rows: u64,
    pub updates: u64,
    pub last_export_ns: u64,
    pub total_export_ns: u64,
    /// The data blocks are kept in memory (not in work_dir files).
    pub in_memory: bool,
}

/// A running continuous snapshot capture. Each snapshot is exported by up to `shards.len()` shard sinks in
/// parallel (one thread each); then the `done` sink gets [`Event::Done`] on the SDK's export thread.
pub struct Live {
    raw: *mut ffi::nvt_live,
    // Boxed so their addresses are stable for the shim; dropped after the capture stopped.
    tramps: Vec<*mut Tramp<'static>>,
    sinks: Vec<*mut (dyn Sink + Send)>,
}

// SAFETY: the handle is only used from one owner; the shim's exporter is internally synchronized.
unsafe impl Send for Live {}

impl Live {
    #[cfg(nvtrace_sdk)]
    pub fn start(opts: &LiveOptions, shards: Vec<Box<dyn Sink + Send>>, done: Box<dyn Sink + Send>) -> Result<Live> {
        if shards.is_empty() {
            return Err(Error::Live { message: "at least one shard sink is required".to_string() });
        }
        std::fs::create_dir_all(&opts.work_dir).map_err(|e| Error::Live { message: format!("{}: {e}", opts.work_dir.display()) })?;
        let wd = c_path(&opts.work_dir)?;
        let mut live = Live { raw: std::ptr::null_mut(), tramps: Vec::new(), sinks: Vec::new() };
        for sink in shards.into_iter().chain(std::iter::once(done)) {
            let sink: *mut (dyn Sink + Send) = Box::into_raw(sink);
            live.sinks.push(sink);
            // SAFETY: `sink` stays allocated until `Live` is dropped, after the capture stopped.
            live.tramps.push(Box::into_raw(Box::new(Tramp { sink: unsafe { &mut *sink }, buf: Vec::with_capacity(32), panicked: false })));
        }
        let n = live.tramps.len() - 1;
        let users: Vec<*mut c_void> = live.tramps[..n].iter().map(|t| t.cast::<c_void>()).collect();
        let o = ffi::nvt_live_options {
            work_dir: wd.as_ptr(),
            snapshot_ms: opts.snapshot_ms,
            data_rate_mib_per_s: opts.data_rate_mib_per_s,
            data_block_size_mib: opts.data_block_size_mib,
            latency_blocks: opts.latency_blocks,
            max_attach_mib: opts.max_attach_mib,
            max_retained_shm_mib: opts.max_retained_shm_mib,
            ignore_mask: opts.ignore.mask(),
            shards: opts.shards,
            in_memory: u32::from(opts.in_memory),
        };
        let mut err = [0u8; 512];
        // SAFETY: options are read during the call; the trampoline contexts outlive the capture (freed in Drop after
        // stop). The shim copies the `users` array.
        let raw = unsafe {
            ffi::nvt_live_start(
                &raw const o,
                trampoline,
                users.as_ptr(),
                n as u32,
                live.tramps[n].cast::<c_void>(),
                err.as_mut_ptr().cast::<c_char>(),
                err.len(),
            )
        };
        if raw.is_null() {
            return Err(Error::Live { message: err_string(&err) });
        }
        live.raw = raw;
        Ok(live)
    }

    #[cfg(not(nvtrace_sdk))]
    pub fn start(_opts: &LiveOptions, _shards: Vec<Box<dyn Sink + Send>>, _done: Box<dyn Sink + Send>) -> Result<Live> {
        Err(Error::Unavailable)
    }

    pub fn stats(&self) -> LiveStats {
        #[cfg(nvtrace_sdk)]
        {
            let mut s = ffi::nvt_live_stats::default();
            // SAFETY: `raw` is a live handle until stop.
            unsafe { ffi::nvt_live_stats_get(self.raw, &raw mut s) };
            LiveStats {
                snapshots_ok: s.snapshots_ok,
                snapshots_failed: s.snapshots_failed,
                rows: s.rows,
                updates: s.updates,
                last_export_ns: s.last_export_ns,
                total_export_ns: s.total_export_ns,
                in_memory: s.in_memory != 0,
            }
        }
        #[cfg(not(nvtrace_sdk))]
        LiveStats::default()
    }

    /// Stop the capture; the final partial snapshot is exported through the sink before this returns.
    pub fn stop(mut self) -> Result<()> {
        self.stop_inner()
    }

    fn stop_inner(&mut self) -> Result<()> {
        if self.raw.is_null() {
            return Ok(());
        }
        let raw = std::mem::replace(&mut self.raw, std::ptr::null_mut());
        #[cfg(nvtrace_sdk)]
        {
            let mut err = [0u8; 512];
            // SAFETY: `raw` came from nvt_live_start and is freed by nvt_live_stop exactly once.
            let rc = unsafe { ffi::nvt_live_stop(raw, err.as_mut_ptr().cast::<c_char>(), err.len()) };
            if rc != 0 {
                return Err(Error::Live { message: err_string(&err) });
            }
        }
        #[cfg(not(nvtrace_sdk))]
        let _ = raw;
        Ok(())
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        let _ = self.stop_inner();
        // SAFETY: the capture has stopped (or never started), nothing references the trampolines or the sinks.
        unsafe {
            for t in self.tramps.drain(..) {
                drop(Box::from_raw(t));
            }
            for s in self.sinks.drain(..) {
                drop(Box::from_raw(s));
            }
        }
    }
}

/// Format a GPU UUID like `nvidia-smi` (`GPU-xxxxxxxx-xxxx-...`).
pub fn uuid_string(u: &[u8; 16]) -> String {
    let h: String = u.iter().map(|b| format!("{b:02x}")).collect();
    format!("GPU-{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignore_mask_roundtrip() {
        let i = Ignore::parse("gpu_pm,nvtx").expect("valid list");
        assert_eq!(i.mask(), 0b1001);
        assert!(Ignore::parse("bogus").is_none());
    }

    #[test]
    fn uuid_format() {
        let u = [0x12u8; 16];
        assert_eq!(uuid_string(&u), "GPU-12121212-1212-1212-1212-121212121212");
    }

    #[test]
    fn value_layout_matches_c() {
        assert_eq!(std::mem::size_of::<ffi::nvt_value>(), 24);
        assert_eq!(std::mem::align_of::<ffi::nvt_value>(), 8);
    }
}
