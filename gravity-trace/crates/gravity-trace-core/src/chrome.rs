//! Trace Parquet -> Chrome Trace Event Format (JSON), the format produced by the PyTorch
//! profiler (Kineto). Open the output in <https://ui.perfetto.dev>, `chrome://tracing`
//! or TensorBoard's trace viewer.
//!
//! Layout (same conventions as Kineto):
//! - one "process" per GPU (`GPU <n>`), one "thread" per CUDA stream;
//! - one process per traced host process with one thread per OS thread for API calls;
//! - `ac2g` flow arrows from a runtime API call to the GPU work it launched (correlation id).
//!
//! Timestamps are written relative to the earliest event (`baseTimeNanoseconds` holds the
//! absolute UTC time) to keep sub-microsecond precision in JSON doubles.
//!
//! Two writers share the event encoding (hand-written JSON: a converter writes millions of events, and building a
//! `serde_json::Value` for each, unbuffered into the gzip encoder, took ~27 µs per event):
//! - [`convert_from`]: trace Parquet batches, read twice (plan the tracks, then the events): captures, `gt dump`;
//! - [`StreamWriter`]: activities as they arrive, in one pass: the daemon's persisted trace in Perfetto format.
use crate::error::{ArrowSnafu, CreateFileSnafu, JsonSnafu, NoInputsSnafu, Result, WriteOutputSnafu, WrongTableSnafu};
use crate::model::{ProcessInfo, TraceRow};
use crate::names::demangle;
use crate::parquet_io::open_reader;
use arrow_array::{
    Array, BooleanArray, Int32Array, Int64Array, RecordBatch, StringArray, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::DataType;
use flate2::Compression;
use flate2::write::GzEncoder;
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::{Value, json};
use snafu::{OptionExt, ResultExt, ensure};
use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// GPU tracks use pids above Linux' pid_max (2^22) so they never collide with real pids.
pub const GPU_PID_BASE: u64 = 1 << 23;

#[derive(Clone, Debug)]
pub struct ConvertOptions {
    pub demangle: bool,
    pub flows: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self { demangle: true, flows: true }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConvertStats {
    pub files: usize,
    pub events: u64,
    pub flows: u64,
    pub base_ns: i64,
}

/// Typed view over the columns of one trace batch.
struct Cols {
    kind: StringArray,
    name: StringArray,
    start: Int64Array,
    end: Int64Array,
    pid: UInt32Array,
    tid: UInt64Array,
    device: UInt32Array,
    stream: UInt32Array,
    corr: UInt32Array,
    graph: UInt32Array,
    graph_node: UInt64Array,
    grid: [Int32Array; 3],
    block: [Int32Array; 3],
    regs: UInt16Array,
    static_shm: Int32Array,
    dynamic_shm: Int32Array,
    bytes: UInt64Array,
    channel: UInt32Array,
    device_launched: BooleanArray,
    /// Absent in files written before the column existed.
    ns_pid: Option<UInt32Array>,
}

fn col<T: Array + Clone + 'static>(b: &RecordBatch, name: &'static str, dt: DataType, path: &Path) -> Result<T> {
    let c = b
        .column_by_name(name)
        .context(WrongTableSnafu { path: path.to_path_buf(), expected: "trace", column: name })?;
    let c = if c.data_type() == &dt {
        c.clone()
    } else {
        arrow_cast::cast(c, &dt).context(ArrowSnafu { context: format!("casting column {name}") })?
    };
    c.as_any()
        .downcast_ref::<T>()
        .cloned()
        .context(WrongTableSnafu { path: path.to_path_buf(), expected: "trace", column: name })
}

impl Cols {
    fn new(b: &RecordBatch, p: &Path) -> Result<Self> {
        use DataType::*;
        Ok(Self {
            kind: col(b, "kind", Utf8, p)?,
            name: col(b, "name", Utf8, p)?,
            start: col(b, "start_ns", Int64, p)?,
            end: col(b, "end_ns", Int64, p)?,
            pid: col(b, "pid", UInt32, p)?,
            tid: col(b, "tid", UInt64, p)?,
            device: col(b, "device_id", UInt32, p)?,
            stream: col(b, "stream_id", UInt32, p)?,
            corr: col(b, "correlation_id", UInt32, p)?,
            graph: col(b, "graph_id", UInt32, p)?,
            graph_node: col(b, "graph_node_id", UInt64, p)?,
            grid: [col(b, "grid_x", Int32, p)?, col(b, "grid_y", Int32, p)?, col(b, "grid_z", Int32, p)?],
            block: [col(b, "block_x", Int32, p)?, col(b, "block_y", Int32, p)?, col(b, "block_z", Int32, p)?],
            regs: col(b, "registers_per_thread", UInt16, p)?,
            static_shm: col(b, "static_shared_mem", Int32, p)?,
            dynamic_shm: col(b, "dynamic_shared_mem", Int32, p)?,
            bytes: col(b, "bytes", UInt64, p)?,
            channel: col(b, "channel_id", UInt32, p)?,
            device_launched: col(b, "device_launched", Boolean, p)?,
            ns_pid: b.column_by_name("ns_pid").and_then(|c| c.as_any().downcast_ref::<UInt32Array>()).cloned(),
        })
    }
}

impl Cols {
    /// Row `i` (`kind` stays a string: [`push_activity`] takes it as written).
    fn row(&self, i: usize) -> TraceRow<'_> {
        let grid_ok = self.grid[0].is_valid(i);
        TraceRow {
            kind: None,
            name: self.name.value(i),
            start_ns: self.start.value(i),
            end_ns: self.end.value(i),
            pid: self.pid.value(i),
            tid: self.tid.is_valid(i).then_some(self.tid.value(i)),
            device_id: self.device.is_valid(i).then_some(self.device.value(i)),
            context_id: None,
            stream_id: self.stream.is_valid(i).then_some(self.stream.value(i)),
            correlation_id: self.corr.value(i),
            graph_id: self.graph.is_valid(i).then_some(self.graph.value(i)),
            graph_node_id: self.graph_node.is_valid(i).then_some(self.graph_node.value(i)),
            grid: grid_ok.then_some([self.grid[0].value(i), self.grid[1].value(i), self.grid[2].value(i)]),
            block: grid_ok.then_some([self.block[0].value(i), self.block[1].value(i), self.block[2].value(i)]),
            registers_per_thread: self.regs.is_valid(i).then_some(self.regs.value(i)),
            static_shared_mem: self.static_shm.is_valid(i).then_some(self.static_shm.value(i)),
            dynamic_shared_mem: self.dynamic_shm.is_valid(i).then_some(self.dynamic_shm.value(i)),
            bytes: self.bytes.is_valid(i).then_some(self.bytes.value(i)),
            channel_id: self.channel.is_valid(i).then_some(self.channel.value(i)),
            device_launched: self.device_launched.is_valid(i).then_some(self.device_launched.value(i)),
            ns_pid: self.ns_pid(i),
        }
    }

    fn ns_pid(&self, i: usize) -> Option<u32> {
        self.ns_pid.as_ref().and_then(|c| c.is_valid(i).then(|| c.value(i)))
    }
}

fn opt<A: Array>(a: &A, i: usize) -> bool {
    a.is_valid(i)
}

// ---------------------------------------------------------------------------------------------------------------
// Event encoding

fn push_u64(b: &mut Vec<u8>, mut v: u64) {
    let mut d = [0u8; 20];
    let mut i = d.len();
    loop {
        i -= 1;
        d[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    b.extend_from_slice(&d[i..]);
}

fn push_i64(b: &mut Vec<u8>, v: i64) {
    if v < 0 {
        b.push(b'-');
    }
    push_u64(b, v.unsigned_abs());
}

/// Nanoseconds as microseconds with three decimals: exact, where a double would round the sub-µs digits of a long
/// trace.
fn push_us(b: &mut Vec<u8>, ns: i64) {
    if ns < 0 {
        b.push(b'-');
    }
    let a = ns.unsigned_abs();
    push_u64(b, a / 1000);
    let f = a % 1000;
    b.extend_from_slice(&[b'.', b'0' + (f / 100) as u8, b'0' + (f / 10 % 10) as u8, b'0' + (f % 10) as u8]);
}

/// A finite double (JSON has no NaN / infinity: 0 instead).
fn push_f64(b: &mut Vec<u8>, v: f64) {
    use std::io::Write as _;
    if v.is_finite() {
        let _ = write!(b, "{v}");
    } else {
        b.push(b'0');
    }
}

/// A JSON string literal.
fn push_str(b: &mut Vec<u8>, s: &str) {
    b.push(b'"');
    for &c in s.as_bytes() {
        match c {
            b'"' => b.extend_from_slice(b"\\\""),
            b'\\' => b.extend_from_slice(b"\\\\"),
            b'\n' => b.extend_from_slice(b"\\n"),
            b'\r' => b.extend_from_slice(b"\\r"),
            b'\t' => b.extend_from_slice(b"\\t"),
            0..=0x1f => {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                b.extend_from_slice(&[b'\\', b'u', b'0', b'0', HEX[(c >> 4) as usize], HEX[(c & 15) as usize]]);
            }
            _ => b.push(c),
        }
    }
    b.push(b'"');
}

/// Activity names as JSON string literals, demangled once per distinct name.
struct NameCache {
    demangle: bool,
    map: FxHashMap<Box<str>, Box<[u8]>>,
}

impl NameCache {
    fn new(demangle: bool) -> Self {
        Self { demangle, map: FxHashMap::default() }
    }

    fn get(&mut self, raw: &str) -> &[u8] {
        if !self.map.contains_key(raw) {
            let mut b = Vec::with_capacity(raw.len() + 2);
            if self.demangle {
                push_str(&mut b, &demangle(raw));
            } else {
                push_str(&mut b, raw);
            }
            self.map.insert(raw.into(), b.into_boxed_slice());
        }
        self.map.get(raw).map_or(&b"\"\""[..], |v| &v[..])
    }
}

/// One complete event (`"X"`) for an activity on track (`pid`, `tid`), times relative to `base`. `kind` is the trace's
/// `kind` column; GPU activities are the ones with a device.
fn push_activity(b: &mut Vec<u8>, kind: &str, r: &TraceRow<'_>, name: &[u8], pid: u64, tid: u64, base: i64) {
    let cat: &[u8] = match (r.device_id.is_some(), kind) {
        (true, "kernel") => b"kernel",
        (true, "memcpy") => b"gpu_memcpy",
        (true, "memset") => b"gpu_memset",
        (true, other) => other.as_bytes(),
        (false, "driver") => b"cuda_driver",
        (false, _) => b"cuda_runtime",
    };
    b.extend_from_slice(b"{\"ph\":\"X\",\"cat\":\"");
    b.extend_from_slice(cat);
    b.extend_from_slice(b"\",\"name\":");
    b.extend_from_slice(name);
    b.extend_from_slice(b",\"pid\":");
    push_u64(b, pid);
    b.extend_from_slice(b",\"tid\":");
    push_u64(b, tid);
    b.extend_from_slice(b",\"ts\":");
    push_us(b, r.start_ns - base);
    b.extend_from_slice(b",\"dur\":");
    push_us(b, r.end_ns - r.start_ns);
    b.extend_from_slice(b",\"args\":{");
    let mut first = true;
    let mut key = |b: &mut Vec<u8>, k: &str| {
        if !first {
            b.push(b',');
        }
        first = false;
        push_str(b, k);
        b.push(b':');
    };
    if r.correlation_id != 0 {
        key(b, "correlation");
        push_u64(b, r.correlation_id.into());
    }
    if let Some(device) = r.device_id {
        key(b, "device");
        push_u64(b, device.into());
        key(b, "stream");
        push_u64(b, r.stream_id.unwrap_or(0).into());
        if let Some(g) = r.graph_id.filter(|&g| g != 0) {
            key(b, "graph id");
            push_u64(b, g.into());
            key(b, "graph node id");
            push_u64(b, r.graph_node_id.unwrap_or(0));
        }
        if let Some(c) = r.channel_id {
            key(b, "channel");
            push_u64(b, c.into());
        }
        if kind == "kernel" {
            if let Some(g) = r.grid {
                let triple = |b: &mut Vec<u8>, v: [i32; 3]| {
                    b.push(b'[');
                    for (i, x) in v.into_iter().enumerate() {
                        if i > 0 {
                            b.push(b',');
                        }
                        push_i64(b, x.into());
                    }
                    b.push(b']');
                };
                key(b, "grid");
                triple(b, g);
                key(b, "block");
                triple(b, r.block.unwrap_or_default());
                key(b, "registers per thread");
                push_u64(b, r.registers_per_thread.unwrap_or(0).into());
                key(b, "shared memory");
                push_i64(b, i64::from(r.static_shared_mem.unwrap_or(0)) + i64::from(r.dynamic_shared_mem.unwrap_or(0)));
            }
            if r.device_launched == Some(true) {
                key(b, "device launched");
                b.extend_from_slice(b"true");
            }
        }
        if let Some(bytes) = r.bytes {
            key(b, "bytes");
            push_u64(b, bytes);
            if r.end_ns > r.start_ns {
                key(b, "memory bandwidth (GB/s)");
                push_f64(b, bytes as f64 / (r.end_ns - r.start_ns) as f64);
            }
        }
    }
    b.extend_from_slice(b"}}");
}

/// The start (`ph` `s`, on the API call) or end (`f`, on the GPU work) of an `ac2g` flow arrow. The id is numeric and
/// unique in the file: Perfetto rejects non-numeric flow v1 ids, and a small integer stays exact in JS viewers.
fn push_flow(b: &mut Vec<u8>, end: bool, id: u64, pid: u64, tid: u64, ts: i64) {
    b.extend_from_slice(if end { b"{\"ph\":\"f\",\"id\":" } else { b"{\"ph\":\"s\",\"id\":" });
    push_u64(b, id);
    b.extend_from_slice(b",\"pid\":");
    push_u64(b, pid);
    b.extend_from_slice(b",\"tid\":");
    push_u64(b, tid);
    b.extend_from_slice(b",\"ts\":");
    push_us(b, ts);
    b.extend_from_slice(if end { &b",\"cat\":\"ac2g\",\"name\":\"ac2g\",\"bp\":\"e\"}"[..] } else { &b",\"cat\":\"ac2g\",\"name\":\"ac2g\"}"[..] });
}

/// A metadata event (`ph` `M`): `process_name` / `thread_name` (a string) or `process_sort_index` (a number).
fn push_meta(b: &mut Vec<u8>, what: &str, pid: u64, tid: Option<u64>, name: Option<&str>, sort_index: u64) {
    b.extend_from_slice(b"{\"name\":");
    push_str(b, what);
    b.extend_from_slice(b",\"ph\":\"M\",\"pid\":");
    push_u64(b, pid);
    if let Some(t) = tid {
        b.extend_from_slice(b",\"tid\":");
        push_u64(b, t);
    }
    match name {
        Some(n) => {
            b.extend_from_slice(b",\"args\":{\"name\":");
            push_str(b, n);
        }
        None => {
            b.extend_from_slice(b",\"args\":{\"sort_index\":");
            push_u64(b, sort_index);
        }
    }
    b.extend_from_slice(b"}}");
}

/// What a [`BatchSource`] hands every batch to: (label for errors, footer metadata of its file, batch).
pub type BatchFn<'a> = dyn FnMut(&Path, &BTreeMap<String, String>, &RecordBatch) -> Result<()> + 'a;

/// Where the converter reads trace batches from. It reads them twice: once to plan the tracks, once for the events.
pub trait BatchSource {
    /// Hand every batch to `f`, in the same order every time, with a label for errors (the file) and the footer
    /// metadata of its file.
    fn for_each(&mut self, f: &mut BatchFn<'_>) -> Result<()>;
}

/// Local trace Parquet files.
pub struct Files<'a>(pub &'a [PathBuf]);

impl BatchSource for Files<'_> {
    fn for_each(&mut self, f: &mut BatchFn<'_>) -> Result<()> {
        for path in self.0 {
            let (_, meta, reader) = open_reader(path)?;
            for batch in reader {
                let batch = batch.context(ArrowSnafu { context: format!("reading {}", path.display()) })?;
                f(path, &meta, &batch)?;
            }
        }
        Ok(())
    }
}

fn for_each_batch(src: &mut dyn BatchSource, mut f: impl FnMut(&Path, &BTreeMap<String, String>, &Cols) -> Result<()>) -> Result<()> {
    src.for_each(&mut |path, meta, batch| {
        let cols = Cols::new(batch, path)?;
        f(path, meta, &cols)
    })
}

/// Summary of the trace needed before events can be written.
#[derive(Default)]
struct Plan {
    base_ns: i64,
    procs: BTreeMap<u32, ProcessInfo>,
    gpu_streams: BTreeMap<(u32, u32, u32), ()>, // (device, pid, stream)
    host_threads: BTreeMap<(u32, u64), ()>,
    /// (pid, correlation) -> (tid, api start ns)
    api: FxHashMap<(u32, u32), (u64, i64)>,
}

/// Footer metadata key of trace files: the JSON list of their [`ProcessInfo`]s.
pub const PROCESSES_META: &str = "processes";

fn plan(src: &mut dyn BatchSource) -> Result<Plan> {
    let mut p = Plan { base_ns: i64::MAX, ..Default::default() };
    // The processes the files describe (a later file's entry wins: titles appear while a process runs).
    let mut described: BTreeMap<u32, ProcessInfo> = BTreeMap::new();
    let mut last_path = PathBuf::new();
    for_each_batch(src, |path, meta, c| {
        if path != last_path {
            last_path = path.to_path_buf();
            let list: Vec<ProcessInfo> = meta.get(PROCESSES_META).and_then(|j| serde_json::from_str(j).ok()).unwrap_or_default();
            described.extend(list.into_iter().map(|q| (q.pid, q)));
        }
        for i in 0..c.start.len() {
            p.base_ns = p.base_ns.min(c.start.value(i));
            let pid = c.pid.value(i);
            p.procs.entry(pid).or_insert_with(|| ProcessInfo {
                pid,
                ns_pid: c.ns_pid(i).unwrap_or(pid),
                exe: meta.get("exe").cloned().unwrap_or_else(|| "process".into()),
                title: String::new(),
            });
            if opt(&c.device, i) {
                let stream = if opt(&c.stream, i) { c.stream.value(i) } else { 0 };
                p.gpu_streams.insert((c.device.value(i), pid, stream), ());
            } else {
                let tid = if opt(&c.tid, i) { c.tid.value(i) } else { 0 };
                p.host_threads.insert((pid, tid), ());
                p.api.insert((pid, c.corr.value(i)), (tid, c.start.value(i)));
            }
        }
        Ok(())
    })?;
    if p.base_ns == i64::MAX {
        p.base_ns = 0;
    }
    for (pid, q) in p.procs.iter_mut() {
        if let Some(d) = described.remove(pid) {
            *q = d;
        }
    }
    Ok(p)
}

struct Emitter<'w, W: Write> {
    out: &'w mut W,
    first: bool,
    events: u64,
}

impl<W: Write> Emitter<'_, W> {
    /// An event already encoded.
    fn raw(&mut self, ev: &[u8]) -> Result<()> {
        let sep: &[u8] = if self.first { b"\n" } else { b",\n" };
        self.first = false;
        self.out.write_all(sep).context(WriteOutputSnafu { path: PathBuf::from("<trace>") })?;
        self.out.write_all(ev).context(WriteOutputSnafu { path: PathBuf::from("<trace>") })?;
        self.events += 1;
        Ok(())
    }

    fn emit(&mut self, v: &Value) -> Result<()> {
        let sep: &[u8] = if self.first { b"\n" } else { b",\n" };
        self.first = false;
        self.out.write_all(sep).context(WriteOutputSnafu { path: PathBuf::from("<trace>") })?;
        serde_json::to_writer(&mut self.out, v).context(JsonSnafu)?;
        self.events += 1;
        Ok(())
    }
}

/// Convert trace Parquet files into one Chrome trace JSON document written to `out`.
pub fn convert<W: Write>(files: &[PathBuf], out: &mut W, opts: &ConvertOptions) -> Result<ConvertStats> {
    ensure!(!files.is_empty(), NoInputsSnafu);
    let mut stats = convert_from(&mut Files(files), None, out, opts)?;
    stats.files = files.len();
    Ok(stats)
}

/// Convert the trace batches of `src` (and PM samples from `pm`, when given) into one Chrome trace JSON document
/// written to `out`. PM samples become counter tracks per GPU UUID (`gpc__cycles_elapsed.avg.per_second`, …).
pub fn convert_from<W: Write>(src: &mut dyn BatchSource, pm: Option<&mut dyn BatchSource>, out: &mut W, opts: &ConvertOptions) -> Result<ConvertStats> {
    let plan = plan(src)?;
    let multi_proc = plan.procs.len() > 1;
    let gpu_pid = |device: u32| GPU_PID_BASE + device as u64;
    let proc_index: FxHashMap<u32, u64> = plan.procs.keys().enumerate().map(|(i, &p)| (p, i as u64)).collect();
    let gpu_tid = |pid: u32, stream: u32| -> u64 {
        if multi_proc { (proc_index.get(&pid).copied().unwrap_or(0) << 32) | stream as u64 } else { stream as u64 }
    };

    // Many small writes: buffered before they reach the (gzip) writer.
    let mut out = std::io::BufWriter::with_capacity(1 << 16, out);
    let mut em = Emitter { out: &mut out, first: true, events: 0 };
    let header = format!(
        "{{\"schemaVersion\":1,\"traceName\":\"gravity-trace\",\"displayTimeUnit\":\"ms\",\"baseTimeNanoseconds\":{},\"deviceProperties\":[],\"traceEvents\":[",
        plan.base_ns
    );
    em.out.write_all(header.as_bytes()).context(WriteOutputSnafu { path: PathBuf::from("<trace>") })?;

    // Track metadata.
    let devices: FxHashSet<u32> = plan.gpu_streams.keys().map(|k| k.0).collect();
    let mut devices: Vec<u32> = devices.into_iter().collect();
    devices.sort_unstable();
    for d in &devices {
        em.emit(&json!({"name":"process_name","ph":"M","pid":gpu_pid(*d),"args":{"name":format!("GPU {d}")}}))?;
        em.emit(&json!({"name":"process_sort_index","ph":"M","pid":gpu_pid(*d),"args":{"sort_index":1000 + d}}))?;
    }
    for (device, pid, stream) in plan.gpu_streams.keys() {
        let name = match plan.procs.get(pid) {
            Some(p) if multi_proc => format!("{} stream {stream}", p.label()),
            _ => format!("stream {stream}"),
        };
        em.emit(&json!({"name":"thread_name","ph":"M","pid":gpu_pid(*device),"tid":gpu_tid(*pid, *stream),"args":{"name":name}}))?;
    }
    for (pid, p) in &plan.procs {
        em.emit(&json!({"name":"process_name","ph":"M","pid":pid,"args":{"name":p.label()}}))?;
    }
    for (pid, tid) in plan.host_threads.keys() {
        em.emit(&json!({"name":"thread_name","ph":"M","pid":pid,"tid":tid,"args":{"name":format!("thread {tid}")}}))?;
    }

    let mut names = NameCache::new(opts.demangle);
    let mut flowed: FxHashSet<(u32, u32)> = FxHashSet::default();
    let mut flows = 0u64;
    let base = plan.base_ns;
    let mut ev = Vec::with_capacity(512);
    for_each_batch(src, |_, _, c| {
        for i in 0..c.start.len() {
            let kind = c.kind.value(i);
            let r = c.row(i);
            let name = names.get(r.name);
            ev.clear();
            let Some(device) = r.device_id else {
                push_activity(&mut ev, kind, &r, name, r.pid.into(), r.tid.unwrap_or(0), base);
                em.raw(&ev)?;
                continue;
            };
            let (gp, gt) = (gpu_pid(device), gpu_tid(r.pid, r.stream_id.unwrap_or(0)));
            push_activity(&mut ev, kind, &r, name, gp, gt, base);
            em.raw(&ev)?;
            let corr = r.correlation_id;
            if opts.flows
                && corr != 0
                && flowed.insert((r.pid, corr))
                && let Some(&(tid, api_start)) = plan.api.get(&(r.pid, corr))
            {
                flows += 1;
                ev.clear();
                push_flow(&mut ev, false, flows, r.pid.into(), tid, api_start - base);
                em.raw(&ev)?;
                ev.clear();
                push_flow(&mut ev, true, flows, gp, gt, r.start_ns - base);
                em.raw(&ev)?;
            }
        }
        Ok(())
    })?;

    // PM counter tracks: one counter process per GPU UUID, samples as `ph:C` events.
    if let Some(pm) = pm {
        let mut counters_map: FxHashMap<String, u64> = FxHashMap::default();
        pm.for_each(&mut |_, _, batch| {
            let Some(uuid_col) = batch.column_by_name("gpu_uuid") else { return Ok(()) };
            let Some(end_col)  = batch.column_by_name("end_ns")   else { return Ok(()) };
            let Some(val_col)  = batch.column_by_name("value")    else { return Ok(()) };
            use arrow_array::{Array, cast::AsArray, types::{Int64Type, Float64Type}};
            let uuids  = uuid_col.as_string::<i32>();
            let ends   = end_col.as_primitive::<Int64Type>();
            let values = val_col.as_primitive::<Float64Type>();
            // metric column may be plain Utf8 or Dict<Int32, Utf8>
            let metric_at = |i: usize| -> &str {
                if let Some(a) = batch.column_by_name("metric") {
                    if let Some(s) = a.as_any().downcast_ref::<arrow_array::StringArray>() {
                        return if s.is_valid(i) { s.value(i) } else { "" };
                    }
                    if let Some(d) = a.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::Int32Type>>()
                        && d.is_valid(i)
                        && let Some(v) = d.values().as_any().downcast_ref::<arrow_array::StringArray>()
                    {
                        return v.value(d.keys().value(i) as usize);
                    }
                }
                ""
            };
            for i in 0..batch.num_rows() {
                let uuid   = uuids.value(i);
                let ts_ns  = ends.value(i);
                let metric = metric_at(i);
                let value  = values.value(i);
                let pid = match counters_map.get(uuid) {
                    Some(&p) => p,
                    None => {
                        let p = PM_PID_BASE + counters_map.len() as u64;
                        counters_map.insert(uuid.to_string(), p);
                        em.emit(&json!({"name":"process_name","ph":"M","pid":p,
                            "args":{"name":format!("{uuid} PM counters")}}))?;
                        em.emit(&json!({"name":"process_sort_index","ph":"M","pid":p,
                            "args":{"sort_index":2000u64 + p - PM_PID_BASE}}))?;
                        p
                    }
                };
                ev.clear();
                ev.extend_from_slice(b"{\"ph\":\"C\",\"name\":");
                push_str(&mut ev, metric);
                ev.extend_from_slice(b",\"pid\":");
                push_u64(&mut ev, pid);
                ev.extend_from_slice(b",\"ts\":");
                push_us(&mut ev, ts_ns - base);
                ev.extend_from_slice(b",\"args\":{\"value\":");
                push_f64(&mut ev, value);
                ev.extend_from_slice(b"}}");
                em.raw(&ev)?;
            }
            Ok(())
        })?;
    }

    em.out.write_all(b"\n]}\n").context(WriteOutputSnafu { path: PathBuf::from("<trace>") })?;
    em.out.flush().context(WriteOutputSnafu { path: PathBuf::from("<trace>") })?;
    Ok(ConvertStats { files: 0, events: em.events, flows, base_ns: base })
}

/// gzip level of the Perfetto files written in one go (captures, `gt dump`, `gt import`). On DeepSeek-V4 trace JSON
/// (miniz_oxide, flate2's default backend) level 6 is 2.4x smaller than level 1 and the most this backend gets
/// (7-9 are no smaller), at a third of level 1's speed. The persisted trace's level is `[persist] gzip_level`.
pub const GZIP_LEVEL: u32 = 6;

/// Convert into a file; gzip-compressed when the name ends with `.gz` (Perfetto opens it directly).
pub fn convert_to_path(files: &[PathBuf], out: &Path, opts: &ConvertOptions) -> Result<ConvertStats> {
    let file = File::create(out).context(CreateFileSnafu { path: out.to_path_buf() })?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    let stats = if out.extension().is_some_and(|e| e == "gz") {
        let mut gz = GzEncoder::new(&mut w, Compression::new(GZIP_LEVEL));
        let stats = convert(files, &mut gz, opts)?;
        gz.finish().context(WriteOutputSnafu { path: out.to_path_buf() })?;
        stats
    } else {
        convert(files, &mut w, opts)?
    };
    w.flush().context(WriteOutputSnafu { path: out.to_path_buf() })?;
    Ok(stats)
}

// ---------------------------------------------------------------------------------------------------------------
// Streaming

/// Process ids of the PM-counter tracks (one process per physical GPU), above the GPU tracks.
pub const PM_PID_BASE: u64 = GPU_PID_BASE + (1 << 20);
/// [`StreamWriter`] times are relative to its first item minus this: later activities may start earlier (GPU times
/// arrive a snapshot or more after the API call that launched the work).
pub const STREAM_BASE_MARGIN_NS: i64 = 60_000_000_000;
/// Snapshots an API call stays available for the flow arrow to its GPU work ([`StreamWriter::snapshot`]).
pub const FLOW_GENERATIONS: usize = 4;

/// One Chrome trace JSON document written while the activities arrive, in one pass (the daemon's persisted trace with
/// `persist.format = "perfetto"`). Without the look-ahead of [`convert_from`]:
/// - times are relative to the first item minus [`STREAM_BASE_MARGIN_NS`] (`baseTimeNanoseconds`);
/// - tracks are named when they first appear: a process by [`StreamWriter::set_process`] (else `pid <n>`), a GPU
///   stream's thread is `<process index> << 32 | stream` (every file, one or many processes);
/// - an API call's flow arrow goes to the first GPU activity of its correlation id that arrives within
///   [`FLOW_GENERATIONS`] snapshots of the call. The caller hands the API calls of a chunk to
///   [`StreamWriter::note_api`] before it pushes the chunk (GPU work can come before its launch in a chunk); an API
///   call that was not noted gets no arrow;
/// - PM samples become counter tracks of a process per GPU ([`StreamWriter::counter`]).
pub struct StreamWriter<W: Write> {
    out: std::io::BufWriter<W>,
    ev: Vec<u8>,
    base: Option<i64>,
    first: bool,
    names: NameCache,
    flows_on: bool,
    /// pid -> process index (GPU thread ids).
    proc_index: FxHashMap<u32, u64>,
    proc_names: FxHashMap<u32, ProcessInfo>,
    /// Host processes whose name was written, and the name.
    named: FxHashMap<u32, String>,
    gpus: FxHashSet<u32>,
    streams: FxHashSet<(u32, u32, u32)>,
    threads: FxHashSet<(u32, u64)>,
    /// GPU (as the caller names it) -> counter process id.
    counters: FxHashMap<String, u64>,
    counter_names: FxHashMap<String, String>,
    /// (pid, correlation) -> (tid, start) of recent API calls, newest generation first.
    api: VecDeque<FxHashMap<(u32, u32), (u64, i64)>>,
    stats: ConvertStats,
}

impl<W: Write> StreamWriter<W> {
    pub fn new(out: W, opts: &ConvertOptions) -> Self {
        Self {
            out: std::io::BufWriter::with_capacity(1 << 16, out),
            ev: Vec::with_capacity(512),
            base: None,
            first: true,
            names: NameCache::new(opts.demangle),
            flows_on: opts.flows,
            proc_index: FxHashMap::default(),
            proc_names: FxHashMap::default(),
            named: FxHashMap::default(),
            gpus: FxHashSet::default(),
            streams: FxHashSet::default(),
            threads: FxHashSet::default(),
            counters: FxHashMap::default(),
            counter_names: FxHashMap::default(),
            api: VecDeque::from([FxHashMap::default()]),
            stats: ConvertStats::default(),
        }
    }

    /// A process, for its tracks' names (written when the process first appears, or now if it did: a process that
    /// gives itself a title is renamed, with its GPU stream tracks).
    pub fn set_process(&mut self, p: &ProcessInfo) -> Result<()> {
        let label = p.label();
        let renamed = self.proc_names.insert(p.pid, p.clone()).is_none_or(|old| old.label() != label);
        if self.named.get(&p.pid).is_some_and(|n| *n != label) {
            self.name_process(p.pid, None)?;
        }
        if renamed {
            let mut streams: Vec<(u32, u32)> = self.streams.iter().filter(|s| s.1 == p.pid).map(|s| (s.0, s.2)).collect();
            streams.sort_unstable();
            for (device, stream) in streams {
                self.name_stream(device, p.pid, stream)?;
            }
        }
        Ok(())
    }

    /// The name of a GPU's counter process (e.g. the device name), for counters that appear later.
    pub fn set_gpu_name(&mut self, gpu: &str, name: &str) {
        self.counter_names.insert(gpu.to_string(), name.to_string());
    }

    /// Remember an API call for the flow arrow to its GPU work.
    pub fn note_api(&mut self, r: &TraceRow<'_>) {
        if self.flows_on && r.device_id.is_none() && r.correlation_id != 0
            && let Some(g) = self.api.front_mut()
        {
            g.insert((r.pid, r.correlation_id), (r.tid.unwrap_or(0), r.start_ns));
        }
    }

    /// A snapshot boundary: API calls older than [`FLOW_GENERATIONS`] snapshots are forgotten.
    pub fn snapshot(&mut self) {
        self.api.push_front(FxHashMap::default());
        self.api.truncate(FLOW_GENERATIONS);
    }

    /// One activity; `kind` as in the trace's `kind` column.
    pub fn push(&mut self, kind: &str, r: &TraceRow<'_>) -> Result<()> {
        let base = self.start(r.start_ns)?;
        let Some(device) = r.device_id else {
            let tid = r.tid.unwrap_or(0);
            if !self.named.contains_key(&r.pid) {
                self.name_process(r.pid, r.ns_pid)?;
            }
            if self.threads.insert((r.pid, tid)) {
                self.meta("thread_name", r.pid.into(), Some(tid), Some(&format!("thread {tid}")), 0)?;
            }
            let name = self.names.get(r.name);
            self.ev.clear();
            push_activity(&mut self.ev, kind, r, name, r.pid.into(), tid, base);
            return self.write_ev();
        };
        let stream = r.stream_id.unwrap_or(0);
        let gp = GPU_PID_BASE + u64::from(device);
        let next = self.proc_index.len() as u64;
        let gt = (*self.proc_index.entry(r.pid).or_insert(next) << 32) | u64::from(stream);
        if self.gpus.insert(device) {
            self.meta("process_name", gp, None, Some(&format!("GPU {device}")), 0)?;
            self.meta("process_sort_index", gp, None, None, 1000 + u64::from(device))?;
        }
        if self.streams.insert((device, r.pid, stream)) {
            self.name_stream(device, r.pid, stream)?;
        }
        let name = self.names.get(r.name);
        self.ev.clear();
        push_activity(&mut self.ev, kind, r, name, gp, gt, base);
        self.write_ev()?;
        let api = if self.flows_on && r.correlation_id != 0 {
            self.api.iter_mut().find_map(|g| g.remove(&(r.pid, r.correlation_id)))
        } else {
            None
        };
        if let Some((tid, api_start)) = api {
            self.stats.flows += 1;
            let id = self.stats.flows;
            self.ev.clear();
            push_flow(&mut self.ev, false, id, r.pid.into(), tid, api_start - base);
            self.write_ev()?;
            self.ev.clear();
            push_flow(&mut self.ev, true, id, gp, gt, r.start_ns - base);
            self.write_ev()?;
        }
        Ok(())
    }

    /// One sample of a counter (`metric`) of a GPU, on that GPU's counter process.
    pub fn counter(&mut self, gpu: &str, metric: &str, ts_ns: i64, value: f64) -> Result<()> {
        let base = self.start(ts_ns)?;
        let pid = match self.counters.get(gpu) {
            Some(&p) => p,
            None => {
                let p = PM_PID_BASE + self.counters.len() as u64;
                self.counters.insert(gpu.to_string(), p);
                let label = match self.counter_names.get(gpu) {
                    Some(n) if !n.is_empty() => format!("{n} {gpu} PM counters"),
                    _ => format!("{gpu} PM counters"),
                };
                self.meta("process_name", p, None, Some(&label), 0)?;
                self.meta("process_sort_index", p, None, None, 2000 + p - PM_PID_BASE)?;
                p
            }
        };
        let b = &mut self.ev;
        b.clear();
        b.extend_from_slice(b"{\"ph\":\"C\",\"name\":");
        push_str(b, metric);
        b.extend_from_slice(b",\"pid\":");
        push_u64(b, pid);
        b.extend_from_slice(b",\"ts\":");
        push_us(b, ts_ns - base);
        b.extend_from_slice(b",\"args\":{\"value\":");
        push_f64(b, value);
        b.extend_from_slice(b"}}");
        self.write_ev()
    }

    /// Activities and counter samples written so far.
    pub fn events(&self) -> u64 {
        self.stats.events
    }

    /// The writer underneath (e.g. to abort an upload).
    pub fn get_mut(&mut self) -> &mut W {
        self.out.get_mut()
    }

    /// Close the document; the writer underneath (flushed) and what was written.
    pub fn finish(mut self) -> Result<(W, ConvertStats)> {
        self.start(0)?;
        let path = || PathBuf::from("<trace>");
        self.out.write_all(b"\n]}\n").context(WriteOutputSnafu { path: path() })?;
        self.out.flush().context(WriteOutputSnafu { path: path() })?;
        let out = self.out.into_inner().map_err(|e| e.into_error()).context(WriteOutputSnafu { path: path() })?;
        Ok((out, self.stats))
    }

    /// The header, before the first event; the time base.
    fn start(&mut self, ts_ns: i64) -> Result<i64> {
        if let Some(b) = self.base {
            return Ok(b);
        }
        let base = ts_ns.saturating_sub(STREAM_BASE_MARGIN_NS).max(0);
        self.base = Some(base);
        self.stats.base_ns = base;
        let header = format!(
            "{{\"schemaVersion\":1,\"traceName\":\"gravity-trace\",\"displayTimeUnit\":\"ms\",\"baseTimeNanoseconds\":{base},\"deviceProperties\":[],\"traceEvents\":["
        );
        self.out.write_all(header.as_bytes()).context(WriteOutputSnafu { path: PathBuf::from("<trace>") })?;
        Ok(base)
    }

    fn name_stream(&mut self, device: u32, pid: u32, stream: u32) -> Result<()> {
        let gt = (self.proc_index.get(&pid).copied().unwrap_or(0) << 32) | u64::from(stream);
        let p = self.proc_names.get(&pid).map_or_else(|| format!("pid {pid}"), ProcessInfo::label);
        self.meta("thread_name", GPU_PID_BASE + u64::from(device), Some(gt), Some(&format!("{p} stream {stream}")), 0)
    }

    /// `ns_pid`: of the row that made the process appear, when it has no [`ProcessInfo`].
    fn name_process(&mut self, pid: u32, ns_pid: Option<u32>) -> Result<()> {
        let name = match self.proc_names.get(&pid) {
            Some(p) => p.label(),
            None => ProcessInfo { pid, ns_pid: ns_pid.unwrap_or(pid), ..Default::default() }.label(),
        };
        self.meta("process_name", pid.into(), None, Some(&name), 0)?;
        self.named.insert(pid, name);
        Ok(())
    }

    fn meta(&mut self, what: &str, pid: u64, tid: Option<u64>, name: Option<&str>, sort_index: u64) -> Result<()> {
        self.ev.clear();
        push_meta(&mut self.ev, what, pid, tid, name, sort_index);
        self.write_ev()?;
        self.stats.events -= 1; // metadata is not an event of the trace
        Ok(())
    }

    fn write_ev(&mut self) -> Result<()> {
        let sep: &[u8] = if self.first { b"\n" } else { b",\n" };
        self.first = false;
        let path = || PathBuf::from("<trace>");
        self.out.write_all(sep).context(WriteOutputSnafu { path: path() })?;
        self.out.write_all(&self.ev).context(WriteOutputSnafu { path: path() })?;
        self.stats.events += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::EventKind;

    fn row(kind: EventKind, start: i64, corr: u32) -> TraceRow<'static> {
        let gpu = kind.is_gpu();
        TraceRow {
            kind: Some(kind),
            name: if gpu { "_Z4tinyPfi" } else { "cuLaunchKernel" },
            start_ns: start,
            end_ns: start + 1_500,
            pid: 42,
            tid: (!gpu).then_some(7),
            device_id: gpu.then_some(1),
            stream_id: gpu.then_some(13),
            correlation_id: corr,
            grid: gpu.then_some([2, 1, 1]),
            block: gpu.then_some([128, 1, 1]),
            ..Default::default()
        }
    }

    fn events(doc: &[u8]) -> Vec<Value> {
        let v: Value = serde_json::from_slice(doc).expect("valid JSON");
        v["traceEvents"].as_array().expect("traceEvents").clone()
    }

    #[test]
    fn numbers_and_strings() {
        let mut b = Vec::new();
        push_us(&mut b, 1_234_567);
        b.push(b' ');
        push_us(&mut b, -5);
        b.push(b' ');
        push_str(&mut b, "a\"b\\c\nd\u{1}é");
        assert_eq!(String::from_utf8(b).expect("utf8"), "1234.567 -0.005 \"a\\\"b\\\\c\\nd\\u0001é\"");
        let s: String = serde_json::from_str("\"a\\\"b\\\\c\\nd\\u0001é\"").expect("parses back");
        assert_eq!(s, "a\"b\\c\nd\u{1}é");
    }

    #[test]
    fn a_stream_of_activities_and_counters() {
        let mut w = StreamWriter::new(Vec::new(), &ConvertOptions::default());
        w.set_process(&ProcessInfo { pid: 42, ns_pid: 7, exe: "python3".into(), title: String::new() }).expect("name");
        w.set_process(&ProcessInfo { pid: 42, ns_pid: 7, exe: "python3".into(), title: "VLLM::Worker_TP0".into() }).expect("titled before it appears");
        w.set_gpu_name("GPU-abc", "NVIDIA Test GPU");
        let start = 100_000_000_000;
        // the kernel arrives before its launch in the chunk: note the chunk's API calls first
        let (api, kernel) = (row(EventKind::Driver, start, 9), row(EventKind::Kernel, start + 2_000, 9));
        w.note_api(&api);
        w.push("kernel", &kernel).expect("kernel");
        w.push("driver", &api).expect("api");
        w.snapshot();
        w.note_api(&row(EventKind::Driver, start + 4_000, 10));
        w.push("kernel", &row(EventKind::Kernel, start + 5_000, 9)).expect("same correlation: no second arrow");
        w.counter("GPU-abc", "sm__throughput.avg.pct_of_peak_sustained_elapsed", start + 3_000, 12.5).expect("counter");
        assert_eq!(w.events(), 6, "3 activities, 2 flow events, 1 counter sample");
        let (out, stats) = w.finish().expect("finish");
        // the base is the first item written (the kernel) minus the margin: the launch before it stays positive
        assert_eq!((stats.flows, stats.base_ns), (1, start + 2_000 - STREAM_BASE_MARGIN_NS));
        let ev = events(&out);
        let ph = |p: &str| ev.iter().filter(|e| e["ph"] == p).collect::<Vec<_>>();
        let x = ph("X");
        assert_eq!(x.len(), 3);
        let k = &x[0];
        assert_eq!(k["name"], "tiny(float*, int)", "demangled");
        assert_eq!((k["cat"].as_str(), k["pid"].as_u64(), k["tid"].as_u64()), (Some("kernel"), Some(GPU_PID_BASE + 1), Some(13)));
        assert_eq!(k["ts"].as_f64(), Some(60_000_000.0), "µs after the base");
        assert_eq!(k["dur"].as_f64(), Some(1.5));
        assert_eq!(k["args"]["grid"], json!([2, 1, 1]));
        assert_eq!(k["args"]["stream"], 13);
        assert_eq!((x[1]["cat"].as_str(), x[1]["pid"].as_u64(), x[1]["tid"].as_u64()), (Some("cuda_driver"), Some(42), Some(7)));
        let (s, f) = (ph("s"), ph("f"));
        assert_eq!((s.len(), f.len()), (1, 1));
        assert_eq!((s[0]["id"].clone(), s[0]["pid"].clone(), s[0]["ts"].as_f64()), (f[0]["id"].clone(), json!(42), Some(59_999_998.0)));
        let c = ph("C");
        assert_eq!((c[0]["pid"].as_u64(), c[0]["args"]["value"].as_f64()), (Some(PM_PID_BASE), Some(12.5)));
        let names: Vec<String> = ph("M").iter().filter(|m| m["name"] != "process_sort_index").map(|m| m["args"]["name"].as_str().unwrap_or("").to_string()).collect();
        assert_eq!(names, ["GPU 1", "VLLM::Worker_TP0 (pid 42) stream 13", "VLLM::Worker_TP0 (pid 42)", "thread 7", "NVIDIA Test GPU GPU-abc PM counters"]);
    }

    #[test]
    fn a_process_that_titles_itself_later_is_renamed() {
        let mut w = StreamWriter::new(Vec::new(), &ConvertOptions::default());
        w.push("kernel", &row(EventKind::Kernel, 100_000_000_000, 9)).expect("kernel of a process not described yet");
        let p = ProcessInfo { pid: 42, ns_pid: 1, exe: "python3.12".into(), title: "VLLM::Worker".into() };
        w.set_process(&p).expect("described");
        w.push("driver", &row(EventKind::Driver, 100_000_001_000, 10)).expect("api");
        w.set_process(&ProcessInfo { title: "VLLM::Worker_TP1".into(), ..p.clone() }).expect("titled");
        w.set_process(&ProcessInfo { title: "VLLM::Worker_TP1".into(), ..p }).expect("unchanged: nothing written");
        let (out, _) = w.finish().expect("finish");
        let names: Vec<String> = events(&out)
            .iter()
            .filter(|m| m["ph"] == "M" && m["name"] != "process_sort_index")
            .map(|m| m["args"]["name"].as_str().unwrap_or("").to_string())
            .collect();
        assert_eq!(
            names,
            [
                "GPU 1",
                "pid 42 stream 13",
                "VLLM::Worker (pid 42) stream 13",
                "VLLM::Worker (pid 42)",
                "thread 7",
                "VLLM::Worker_TP1 (pid 42)",
                "VLLM::Worker_TP1 (pid 42) stream 13",
            ],
            "the last name of a track wins"
        );
    }

    #[test]
    fn an_empty_stream_is_a_valid_document() {
        let (out, stats) = StreamWriter::new(Vec::new(), &ConvertOptions::default()).finish().expect("finish");
        assert!(events(&out).is_empty());
        assert_eq!(stats.events, 0);
    }
}
