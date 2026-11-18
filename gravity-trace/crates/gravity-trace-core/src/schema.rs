//! Arrow schemas and batch builders for the two Parquet tables:
//!
//! - **trace**: one row per activity (kernel / memcpy / memset / runtime / driver API call),
//!   written by `gt import`. The duration is `end_ns - start_ns` (not stored: a third
//!   timestamp column costs ~40% more bytes per row).
//! - **stats**: one row per (window, device, kind, name), produced by the aggregator at every
//!   window close (`gt` only exports them as metrics, it does not write this table). Rows with `kind = "kernel_busy"` carry the per-device union of
//!   kernel execution intervals (`total_ns`) for that window.
use crate::model::TraceRow;
use arrow_array::RecordBatch;
use arrow_array::builder::{
    BooleanBuilder, Int32Builder, Int64Builder, StringDictionaryBuilder, UInt16Builder, UInt32Builder,
    UInt64Builder,
};
use arrow_array::types::Int32Type;
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use std::sync::{Arc, OnceLock};

fn dict() -> DataType {
    DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
}

pub fn trace_schema() -> SchemaRef {
    static S: OnceLock<SchemaRef> = OnceLock::new();
    S.get_or_init(|| {
        Arc::new(Schema::new(vec![
            Field::new("kind", dict(), false),
            Field::new("name", dict(), false),
            Field::new("start_ns", DataType::Int64, false),
            Field::new("end_ns", DataType::Int64, false),
            Field::new("pid", DataType::UInt32, false),
            Field::new("tid", DataType::UInt64, true),
            Field::new("device_id", DataType::UInt32, true),
            Field::new("context_id", DataType::UInt32, true),
            Field::new("stream_id", DataType::UInt32, true),
            Field::new("correlation_id", DataType::UInt32, false),
            Field::new("graph_id", DataType::UInt32, true),
            Field::new("graph_node_id", DataType::UInt64, true),
            Field::new("grid_x", DataType::Int32, true),
            Field::new("grid_y", DataType::Int32, true),
            Field::new("grid_z", DataType::Int32, true),
            Field::new("block_x", DataType::Int32, true),
            Field::new("block_y", DataType::Int32, true),
            Field::new("block_z", DataType::Int32, true),
            Field::new("registers_per_thread", DataType::UInt16, true),
            Field::new("static_shared_mem", DataType::Int32, true),
            Field::new("dynamic_shared_mem", DataType::Int32, true),
            Field::new("bytes", DataType::UInt64, true),
            Field::new("channel_id", DataType::UInt32, true),
            Field::new("device_launched", DataType::Boolean, true),
            // Last: added after the others (pid = the host's since then); null in older files.
            Field::new("ns_pid", DataType::UInt32, true),
        ]))
    })
    .clone()
}

pub fn stats_schema() -> SchemaRef {
    static S: OnceLock<SchemaRef> = OnceLock::new();
    S.get_or_init(|| {
        Arc::new(Schema::new(vec![
            Field::new("window_start_ns", DataType::Int64, false),
            Field::new("window_end_ns", DataType::Int64, false),
            Field::new("pid", DataType::UInt32, false),
            Field::new("device_id", DataType::UInt32, false),
            Field::new("kind", dict(), false),
            Field::new("name", dict(), false),
            Field::new("count", DataType::UInt64, false),
            Field::new("total_ns", DataType::UInt64, false),
            Field::new("min_ns", DataType::UInt64, false),
            Field::new("max_ns", DataType::UInt64, false),
            Field::new("p50_ns", DataType::UInt64, false),
            Field::new("p90_ns", DataType::UInt64, false),
            Field::new("p99_ns", DataType::UInt64, false),
        ]))
    })
    .clone()
}

/// Column-wise accumulator for trace rows.
pub struct TraceBatchBuilder {
    kind: StringDictionaryBuilder<Int32Type>,
    name: StringDictionaryBuilder<Int32Type>,
    start: Int64Builder,
    end: Int64Builder,
    pid: UInt32Builder,
    tid: UInt64Builder,
    device: UInt32Builder,
    context: UInt32Builder,
    stream: UInt32Builder,
    correlation: UInt32Builder,
    graph: UInt32Builder,
    graph_node: UInt64Builder,
    grid: [Int32Builder; 3],
    block: [Int32Builder; 3],
    regs: UInt16Builder,
    static_shm: Int32Builder,
    dynamic_shm: Int32Builder,
    bytes: UInt64Builder,
    channel: UInt32Builder,
    device_launched: BooleanBuilder,
    ns_pid: UInt32Builder,
    len: usize,
}

impl Default for TraceBatchBuilder {
    fn default() -> Self {
        Self {
            kind: StringDictionaryBuilder::new(),
            name: StringDictionaryBuilder::new(),
            start: Int64Builder::new(),
            end: Int64Builder::new(),
            pid: UInt32Builder::new(),
            tid: UInt64Builder::new(),
            device: UInt32Builder::new(),
            context: UInt32Builder::new(),
            stream: UInt32Builder::new(),
            correlation: UInt32Builder::new(),
            graph: UInt32Builder::new(),
            graph_node: UInt64Builder::new(),
            grid: [Int32Builder::new(), Int32Builder::new(), Int32Builder::new()],
            block: [Int32Builder::new(), Int32Builder::new(), Int32Builder::new()],
            regs: UInt16Builder::new(),
            static_shm: Int32Builder::new(),
            dynamic_shm: Int32Builder::new(),
            bytes: UInt64Builder::new(),
            channel: UInt32Builder::new(),
            device_launched: BooleanBuilder::new(),
            ns_pid: UInt32Builder::new(),
            len: 0,
        }
    }
}

impl TraceBatchBuilder {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(&mut self, r: &TraceRow<'_>) {
        self.kind.append_value(r.kind.map(|k| k.as_str()).unwrap_or("unknown"));
        self.name.append_value(r.name);
        self.start.append_value(r.start_ns);
        self.end.append_value(r.end_ns);
        self.pid.append_value(r.pid);
        self.tid.append_option(r.tid);
        self.device.append_option(r.device_id);
        self.context.append_option(r.context_id);
        self.stream.append_option(r.stream_id);
        self.correlation.append_value(r.correlation_id);
        self.graph.append_option(r.graph_id);
        self.graph_node.append_option(r.graph_node_id);
        for i in 0..3 {
            self.grid[i].append_option(r.grid.map(|g| g[i]));
            self.block[i].append_option(r.block.map(|b| b[i]));
        }
        self.regs.append_option(r.registers_per_thread);
        self.static_shm.append_option(r.static_shared_mem);
        self.dynamic_shm.append_option(r.dynamic_shared_mem);
        self.bytes.append_option(r.bytes);
        self.channel.append_option(r.channel_id);
        self.device_launched.append_option(r.device_launched);
        self.ns_pid.append_option(r.ns_pid);
        self.len += 1;
    }

    /// Emit the accumulated rows as a batch and reset the builder.
    pub fn finish(&mut self) -> Result<RecordBatch, ArrowError> {
        self.len = 0;
        let [gx, gy, gz] = &mut self.grid;
        let [bx, by, bz] = &mut self.block;
        RecordBatch::try_new(
            trace_schema(),
            vec![
                Arc::new(self.kind.finish()),
                Arc::new(self.name.finish()),
                Arc::new(self.start.finish()),
                Arc::new(self.end.finish()),
                Arc::new(self.pid.finish()),
                Arc::new(self.tid.finish()),
                Arc::new(self.device.finish()),
                Arc::new(self.context.finish()),
                Arc::new(self.stream.finish()),
                Arc::new(self.correlation.finish()),
                Arc::new(self.graph.finish()),
                Arc::new(self.graph_node.finish()),
                Arc::new(gx.finish()),
                Arc::new(gy.finish()),
                Arc::new(gz.finish()),
                Arc::new(bx.finish()),
                Arc::new(by.finish()),
                Arc::new(bz.finish()),
                Arc::new(self.regs.finish()),
                Arc::new(self.static_shm.finish()),
                Arc::new(self.dynamic_shm.finish()),
                Arc::new(self.bytes.finish()),
                Arc::new(self.channel.finish()),
                Arc::new(self.device_launched.finish()),
                Arc::new(self.ns_pid.finish()),
            ],
        )
    }
}

/// One aggregated statistics row.
#[derive(Clone, Debug)]
pub struct StatRow {
    pub window_start_ns: i64,
    pub window_end_ns: i64,
    pub device_id: u32,
    pub kind: &'static str,
    pub name: Arc<str>,
    pub count: u64,
    pub total_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub p50_ns: u64,
    pub p90_ns: u64,
    pub p99_ns: u64,
}

pub fn stats_batch(pid: u32, rows: &[StatRow]) -> Result<RecordBatch, ArrowError> {
    let mut ws = Int64Builder::with_capacity(rows.len());
    let mut we = Int64Builder::with_capacity(rows.len());
    let mut p = UInt32Builder::with_capacity(rows.len());
    let mut dev = UInt32Builder::with_capacity(rows.len());
    let mut kind = StringDictionaryBuilder::<Int32Type>::new();
    let mut name = StringDictionaryBuilder::<Int32Type>::new();
    let mut cols: [UInt64Builder; 7] = std::array::from_fn(|_| UInt64Builder::with_capacity(rows.len()));
    for r in rows {
        ws.append_value(r.window_start_ns);
        we.append_value(r.window_end_ns);
        p.append_value(pid);
        dev.append_value(r.device_id);
        kind.append_value(r.kind);
        name.append_value(&*r.name);
        for (b, v) in cols.iter_mut().zip([r.count, r.total_ns, r.min_ns, r.max_ns, r.p50_ns, r.p90_ns, r.p99_ns]) {
            b.append_value(v);
        }
    }
    let mut arrays: Vec<arrow_array::ArrayRef> = vec![
        Arc::new(ws.finish()),
        Arc::new(we.finish()),
        Arc::new(p.finish()),
        Arc::new(dev.finish()),
        Arc::new(kind.finish()),
        Arc::new(name.finish()),
    ];
    arrays.extend(cols.iter_mut().map(|b| Arc::new(b.finish()) as arrow_array::ArrayRef));
    RecordBatch::try_new(stats_schema(), arrays)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::EventKind;

    #[test]
    fn trace_builder_roundtrip_shape() {
        let mut b = TraceBatchBuilder::default();
        b.push(&TraceRow {
            kind: Some(EventKind::Kernel),
            name: "k",
            start_ns: 10,
            end_ns: 25,
            grid: Some([1, 2, 3]),
            ..Default::default()
        });
        b.push(&TraceRow { kind: Some(EventKind::Runtime), name: "cudaLaunchKernel", ..Default::default() });
        let batch = b.finish().expect("valid batch");
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), trace_schema().fields().len());
        assert!(b.is_empty());
    }
}
