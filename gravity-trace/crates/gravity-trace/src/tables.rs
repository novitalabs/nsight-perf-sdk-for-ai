//! `gt tables`: dump every NvTrace table (and deferred update) to its own Parquet file, columns and types
//! exactly as in the SDK schema. This is the lossless export; `gt import` is the activity-model view.
use crate::error::{ArrowSnafu, CoreSnafu, NvtraceSnafu, Result};
use arrow_array::ArrayRef;
use arrow_array::builder::{
    BooleanBuilder, Float64Builder, Int8Builder, Int32Builder, Int64Builder, StringBuilder, UInt8Builder, UInt16Builder,
    UInt32Builder, UInt64Builder,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use crate::store::{ObjectWriter, Store};
use gravity_trace_core::parquet_io::{TableWriter, WriteOptions};
use nvtrace_sys::{self as nvt, ColumnRole, ColumnType, Event, Sink, Value};
use snafu::ResultExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const BATCH_ROWS: usize = 64 * 1024;

enum ColB {
    Bool(BooleanBuilder),
    I8(Int8Builder),
    I32(Int32Builder),
    I64(Int64Builder),
    U8(UInt8Builder),
    U16(UInt16Builder),
    U32(UInt32Builder),
    U64(UInt64Builder),
    F64(Float64Builder),
    Str(StringBuilder),
}

impl ColB {
    fn new(t: ColumnType) -> (Self, DataType) {
        match t {
            ColumnType::Bool => (ColB::Bool(BooleanBuilder::new()), DataType::Boolean),
            ColumnType::Int8 => (ColB::I8(Int8Builder::new()), DataType::Int8),
            ColumnType::Int32 => (ColB::I32(Int32Builder::new()), DataType::Int32),
            ColumnType::Int64 => (ColB::I64(Int64Builder::new()), DataType::Int64),
            ColumnType::UInt8 => (ColB::U8(UInt8Builder::new()), DataType::UInt8),
            ColumnType::UInt16 => (ColB::U16(UInt16Builder::new()), DataType::UInt16),
            ColumnType::UInt32 => (ColB::U32(UInt32Builder::new()), DataType::UInt32),
            ColumnType::UInt64 => (ColB::U64(UInt64Builder::new()), DataType::UInt64),
            ColumnType::Double => (ColB::F64(Float64Builder::new()), DataType::Float64),
            // UUIDs are written in nvidia-smi form (GPU-xxxxxxxx-...).
            ColumnType::String | ColumnType::Uuid | ColumnType::Unknown(_) => (ColB::Str(StringBuilder::new()), DataType::Utf8),
        }
    }

    fn push(&mut self, v: Option<&Value<'_>>) {
        let v = v.copied().unwrap_or(Value::Null);
        match self {
            ColB::Bool(b) => b.append_option(v.bool()),
            ColB::I8(b) => b.append_option(v.i64().and_then(|x| i8::try_from(x).ok())),
            ColB::I32(b) => b.append_option(v.i64().and_then(|x| i32::try_from(x).ok())),
            ColB::I64(b) => b.append_option(v.i64()),
            ColB::U8(b) => b.append_option(v.u64().and_then(|x| u8::try_from(x).ok())),
            ColB::U16(b) => b.append_option(v.u64().and_then(|x| u16::try_from(x).ok())),
            ColB::U32(b) => b.append_option(v.u64().and_then(|x| u32::try_from(x).ok())),
            ColB::U64(b) => b.append_option(v.u64()),
            ColB::F64(b) => b.append_option(v.f64()),
            ColB::Str(b) => match v {
                Value::Str(s) => b.append_value(s),
                Value::Uuid(u) => b.append_value(nvt::uuid_string(&u)),
                Value::Null => b.append_null(),
                other => b.append_value(format!("{other:?}")),
            },
        }
    }

    fn finish(&mut self) -> ArrayRef {
        match self {
            ColB::Bool(b) => Arc::new(b.finish()),
            ColB::I8(b) => Arc::new(b.finish()),
            ColB::I32(b) => Arc::new(b.finish()),
            ColB::I64(b) => Arc::new(b.finish()),
            ColB::U8(b) => Arc::new(b.finish()),
            ColB::U16(b) => Arc::new(b.finish()),
            ColB::U32(b) => Arc::new(b.finish()),
            ColB::U64(b) => Arc::new(b.finish()),
            ColB::F64(b) => Arc::new(b.finish()),
            ColB::Str(b) => Arc::new(b.finish()),
        }
    }
}

struct Writer {
    name: String,
    store: Store,
    rt: tokio::runtime::Handle,
    schema: SchemaRef,
    cols: Vec<ColB>,
    len: usize,
    file: Option<TableWriter<ObjectWriter>>,
    rows: u64,
}

impl Writer {
    fn new(store: &Store, rt: &tokio::runtime::Handle, name: String, cols: &[(String, ColumnType, bool)]) -> Self {
        let mut fields = Vec::new();
        let mut b = Vec::new();
        for (name, ty, nullable) in cols {
            let (c, dt) = ColB::new(*ty);
            b.push(c);
            fields.push(Field::new(name, dt, *nullable));
        }
        let schema = Arc::new(ArrowSchema::new(fields));
        Self { name, store: store.clone(), rt: rt.clone(), schema, cols: b, len: 0, file: None, rows: 0 }
    }

    fn push(&mut self, v: &[Value<'_>]) -> Result<()> {
        for (i, c) in self.cols.iter_mut().enumerate() {
            c.push(v.get(i));
        }
        self.len += 1;
        self.rows += 1;
        if self.len >= BATCH_ROWS {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }
        self.len = 0;
        let arrays: Vec<ArrayRef> = self.cols.iter_mut().map(ColB::finish).collect();
        let batch = arrow_array::RecordBatch::try_new(self.schema.clone(), arrays)
            .context(ArrowSnafu { context: format!("building {}", self.name) })?;
        if self.file.is_none() {
            let sink = self.store.writer(&self.rt, ".parquet");
            let label = PathBuf::from(self.store.show(&self.name));
            self.file = Some(TableWriter::new(sink, label, self.schema.clone(), &WriteOptions::default()).context(CoreSnafu)?);
        }
        if let Some(f) = self.file.as_mut() {
            f.write(&batch).context(CoreSnafu)?;
        }
        Ok(())
    }
}

struct TableSink {
    tables: Vec<Writer>,
    updates: Vec<Writer>,
    err: Option<crate::error::Error>,
}

impl Sink for TableSink {
    fn event(&mut self, ev: Event<'_>) -> bool {
        let r = match ev {
            Event::Row { table, values } => self.tables.get_mut(table).map(|w| w.push(values)).unwrap_or(Ok(())),
            Event::Update { update, values } => self.updates.get_mut(update).map(|w| w.push(values)).unwrap_or(Ok(())),
            _ => Ok(()),
        };
        if let Err(e) = r {
            self.err = Some(e);
            return false;
        }
        true
    }
}

/// Returns (file, rows) per non-empty table / update. `out` is a directory or an object store URL; runs on a blocking
/// thread of the async runtime.
pub fn dump_tables(dir: &Path, out: &str, ignore: nvt::Ignore) -> Result<Vec<(String, u64)>> {
    let rt = tokio::runtime::Handle::current();
    let store = Store::open(out)?;
    let s = nvt::schema().context(NvtraceSnafu)?;
    let tables = s
        .tables
        .iter()
        .map(|t| {
            let cols: Vec<_> = t.columns.iter().map(|c| (c.name.clone(), c.ty, c.role == ColumnRole::Nullable)).collect();
            Writer::new(&store, &rt, format!("{}.parquet", t.name), &cols)
        })
        .collect();
    let updates = s
        .updates
        .iter()
        .map(|u| {
            let t = s.tables.get(u.table);
            let ty = |n: &str| t.and_then(|t| t.col(n)).and_then(|i| t.map(|t| t.columns[i].ty)).unwrap_or(ColumnType::UInt64);
            let mut cols: Vec<_> = u.match_columns.iter().map(|n| (n.clone(), ty(n), true)).collect();
            if let Some(l) = &u.latest_column {
                cols.push((format!("{l}_bound"), ty(l), true));
            }
            cols.extend(u.update_columns.iter().map(|n| (n.clone(), ty(n), true)));
            Writer::new(&store, &rt, format!("update.{}.parquet", u.name), &cols)
        })
        .collect();
    let mut sink = TableSink { tables, updates, err: None };
    let r = nvt::export_dir(dir, &nvt::ExportOptions { ignore, ..Default::default() }, &mut sink);
    if let Some(e) = sink.err.take() {
        return Err(e);
    }
    r.context(NvtraceSnafu)?;
    let mut done = Vec::new();
    for mut w in sink.tables.into_iter().chain(sink.updates) {
        w.flush()?;
        if let Some(f) = w.file.take() {
            let sink = f.finish().context(CoreSnafu)?;
            done.push((sink.finish(&w.name)?, w.rows));
        }
    }
    Ok(done)
}

/// Print the SDK schema (tables, columns, deferred updates).
pub fn print_schema() -> Result<()> {
    let s = nvt::schema().context(NvtraceSnafu)?;
    println!("NvTrace table schema v{} ({} tables, {} updates)", s.version, s.tables.len(), s.updates.len());
    for (i, t) in s.tables.iter().enumerate() {
        println!("{i:>2} {}", t.name);
        for c in &t.columns {
            println!("     {:<44} {:?} {:?}", c.name, c.ty, c.role);
        }
    }
    for u in &s.updates {
        println!(
            "update {} -> {}: match [{}]{} set [{}]",
            u.name,
            s.tables.get(u.table).map(|t| t.name.as_str()).unwrap_or("?"),
            u.match_columns.join(", "),
            u.latest_column.as_ref().map(|l| format!(" latest {l}")).unwrap_or_default(),
            u.update_columns.join(", ")
        );
    }
    Ok(())
}
