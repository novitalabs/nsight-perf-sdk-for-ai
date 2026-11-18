//! Parquet IO with zstd compression.
//!
//! Files are written as `<name>.parquet.inprogress` and atomically renamed on close, so
//! readers never see a file without a footer.
//!
//! Trace files ([`WriteOptions::trace`]) buffer each row group, sort it by process, GPU, stream
//! and start time, and delta-encode the timestamps: sorted, consecutive timestamps differ by
//! little, while a dictionary cannot help with values that are all distinct. On a DeepSeek-V4
//! capture that is about 4.2 bytes per row instead of 9.8 (bench/pq_compress.py). How long a
//! file is does not change the ratio: Parquet compresses each row group and column on its own.
use crate::error::{
    ArrowSnafu, CreateDirSnafu, CreateFileSnafu, OpenFileSnafu, ParquetReadSnafu, ParquetWriteSnafu, ReadDirSnafu,
    RenameSnafu, Result, WrongTableSnafu, ZstdLevelSnafu,
};
use arrow_array::RecordBatch;
use arrow_ord::sort::{SortColumn, lexsort_to_indices};
use arrow_schema::SchemaRef;
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::metadata::{KeyValue, ParquetMetaData};
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;
use snafu::{OptionExt, ResultExt};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const INPROGRESS_SUFFIX: &str = ".inprogress";
/// Prefix of the footer key-value metadata written from [`WriteOptions::metadata`].
pub const META_PREFIX: &str = "gravity-trace.";

/// Rows per row group of a trace file. With delta-encoded timestamps the group size barely changes the
/// size (4.24 vs 4.21 bytes per row at 256k vs 1M rows); 1M rows keeps the per-group overhead small.
pub const TRACE_ROW_GROUP_ROWS: usize = 1024 * 1024;
/// Sort keys of a trace row group: rows of one stream end up next to each other, in time order.
pub const TRACE_SORT_BY: [&str; 4] = ["pid", "device_id", "stream_id", "start_ns"];
/// Trace columns written with DELTA_BINARY_PACKED instead of a dictionary.
pub const TRACE_DELTA_COLUMNS: [&str; 3] = ["start_ns", "end_ns", "correlation_id"];
/// A sorted row group is handed to the Parquet writer in slices of this many rows (bounds the extra copy).
const SORTED_WRITE_ROWS: usize = 64 * 1024;

#[derive(Clone, Debug)]
pub struct WriteOptions {
    pub zstd_level: i32,
    pub row_group_rows: usize,
    /// Sort the rows of every row group by these columns (ascending, nulls first) before writing it. The rows of
    /// the group in progress are buffered in memory until it is full or the file is closed.
    pub sort_by: Vec<String>,
    /// Integer columns written with DELTA_BINARY_PACKED instead of a dictionary (values that are mostly distinct
    /// but close to their neighbours once sorted, e.g. timestamps).
    pub delta_columns: Vec<String>,
    /// Stored in the Parquet footer under `gravity-trace.<key>` (source, TSF directory, ...).
    pub metadata: BTreeMap<String, String>,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            zstd_level: 3,
            row_group_rows: 256 * 1024,
            sort_by: Vec::new(),
            delta_columns: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }
}

impl WriteOptions {
    /// Options for trace files (`schema::trace_schema`): 1M-row groups, each sorted by pid, GPU, stream and start
    /// time; timestamps and correlation ids delta-encoded, everything else dictionary-encoded; zstd 3.
    pub fn trace() -> Self {
        Self {
            row_group_rows: TRACE_ROW_GROUP_ROWS,
            sort_by: TRACE_SORT_BY.iter().map(|c| c.to_string()).collect(),
            delta_columns: TRACE_DELTA_COLUMNS.iter().map(|c| c.to_string()).collect(),
            ..Self::default()
        }
    }

    pub fn properties(&self) -> Result<WriterProperties> {
        let level = ZstdLevel::try_new(self.zstd_level).context(ZstdLevelSnafu { level: self.zstd_level })?;
        let kv: Vec<KeyValue> =
            self.metadata.iter().map(|(k, v)| KeyValue::new(format!("{META_PREFIX}{k}"), v.clone())).collect();
        let mut b = WriterProperties::builder()
            .set_compression(Compression::ZSTD(level))
            .set_dictionary_enabled(true)
            .set_max_row_group_row_count(Some(self.row_group_rows))
            .set_created_by(format!("gravity-trace {}", crate::VERSION))
            .set_key_value_metadata(Some(kv));
        for c in &self.delta_columns {
            let col = ColumnPath::from(c.as_str());
            b = b.set_column_dictionary_enabled(col.clone(), false).set_column_encoding(col, Encoding::DELTA_BINARY_PACKED);
        }
        Ok(b.build())
    }
}

/// The row group in progress of a file whose groups are sorted.
struct GroupSorter {
    keys: Vec<usize>,
    group_rows: usize,
    pending: Vec<RecordBatch>,
    pending_rows: usize,
}

impl GroupSorter {
    /// Sort the buffered rows and write them as one row group.
    fn flush<W: Write + Send>(&mut self, writer: &mut ArrowWriter<W>, name: &Path) -> Result<()> {
        let Some(first) = self.pending.first() else { return Ok(()) };
        let all = concat_batches(&first.schema(), &self.pending)
            .context(ArrowSnafu { context: "concatenating a row group".to_string() })?;
        self.pending.clear();
        self.pending_rows = 0;
        let cols: Vec<SortColumn> =
            self.keys.iter().map(|&i| SortColumn { values: all.column(i).clone(), options: None }).collect();
        let order = lexsort_to_indices(&cols, None).context(ArrowSnafu { context: "sorting a row group".to_string() })?;
        drop(cols);
        let mut at = 0;
        while at < order.len() {
            let n = SORTED_WRITE_ROWS.min(order.len() - at);
            let part = take_record_batch(&all, &order.slice(at, n))
                .context(ArrowSnafu { context: "reordering a row group".to_string() })?;
            writer.write(&part).context(ParquetWriteSnafu { path: name.to_path_buf() })?;
            at += n;
        }
        // end the group here even if it is not full (the last group of the file)
        writer.flush().context(ParquetWriteSnafu { path: name.to_path_buf() })
    }
}

/// A Parquet table written to any byte sink (a local file, an object store upload, memory), with the row groups
/// sorted when [`WriteOptions::sort_by`] says so.
pub struct TableWriter<W: Write + Send> {
    writer: ArrowWriter<W>,
    /// For error messages.
    name: PathBuf,
    rows: usize,
    sorter: Option<GroupSorter>,
}

impl<W: Write + Send> TableWriter<W> {
    pub fn new(sink: W, name: PathBuf, schema: SchemaRef, opts: &WriteOptions) -> Result<Self> {
        let keys = opts
            .sort_by
            .iter()
            .map(|c| schema.index_of(c).context(ArrowSnafu { context: format!("looking up sort column {c}") }))
            .collect::<Result<Vec<_>>>()?;
        let writer = ArrowWriter::try_new(sink, schema, Some(opts.properties()?)).context(ParquetWriteSnafu { path: name.clone() })?;
        let sorter = (!keys.is_empty()).then(|| GroupSorter {
            keys,
            group_rows: opts.row_group_rows.max(1),
            pending: Vec::new(),
            pending_rows: 0,
        });
        Ok(Self { writer, name, rows: 0, sorter })
    }

    pub fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        self.rows += batch.num_rows();
        let Self { writer, name, sorter, .. } = self;
        let Some(s) = sorter.as_mut() else {
            return writer.write(batch).context(ParquetWriteSnafu { path: name.clone() });
        };
        let mut at = 0;
        while at < batch.num_rows() {
            let n = (s.group_rows - s.pending_rows).min(batch.num_rows() - at);
            s.pending.push(batch.slice(at, n));
            s.pending_rows += n;
            at += n;
            if s.pending_rows >= s.group_rows {
                s.flush(writer, name)?;
            }
        }
        Ok(())
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Add footer metadata (`key` without the prefix) known only once the rows are written.
    pub fn append_metadata(&mut self, key: &str, value: String) {
        self.writer.append_key_value_metadata(KeyValue::new(format!("{META_PREFIX}{key}"), value));
    }

    /// The sink (e.g. to abort an upload before dropping the writer).
    pub fn sink_mut(&mut self) -> &mut W {
        self.writer.inner_mut()
    }

    /// Write the last row group and the footer; the sink back.
    pub fn finish(self) -> Result<W> {
        let Self { mut writer, name, sorter, .. } = self;
        if let Some(mut s) = sorter {
            s.flush(&mut writer, &name)?;
        }
        writer.into_inner().context(ParquetWriteSnafu { path: name })
    }
}

/// A single local Parquet file being written (`<name>.inprogress` until it is complete).
pub struct ParquetFile {
    table: TableWriter<File>,
    tmp: PathBuf,
    path: PathBuf,
}

impl ParquetFile {
    pub fn create(path: PathBuf, schema: SchemaRef, opts: &WriteOptions) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).context(CreateDirSnafu { path: dir.to_path_buf() })?;
        }
        let mut tmp = path.clone().into_os_string();
        tmp.push(INPROGRESS_SUFFIX);
        let tmp = PathBuf::from(tmp);
        let file = File::create(&tmp).context(CreateFileSnafu { path: tmp.clone() })?;
        let table = TableWriter::new(file, tmp.clone(), schema, opts)?;
        Ok(Self { table, tmp, path })
    }

    pub fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        self.table.write(batch)
    }

    pub fn rows(&self) -> usize {
        self.table.rows()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write the footer and publish the file under its final name.
    pub fn close(self) -> Result<PathBuf> {
        let Self { table, tmp, path } = self;
        table.finish()?;
        std::fs::rename(&tmp, &path).context(RenameSnafu { from: tmp, to: path.clone() })?;
        Ok(path)
    }
}

/// Row groups of a file that may hold rows whose Int64 `column` lies in `[lo, hi)` (by their min / max statistics;
/// groups without statistics are kept), and how many were skipped.
pub fn row_groups_in_range(md: &ParquetMetaData, column: &str, lo: i64, hi: i64) -> Option<(Vec<usize>, usize)> {
    use parquet::file::statistics::Statistics;
    let ci = md.file_metadata().schema_descr().columns().iter().position(|c| c.name() == column)?;
    let mut keep = Vec::new();
    let mut skipped = 0;
    for g in 0..md.num_row_groups() {
        let hit = match md.row_group(g).column(ci).statistics() {
            Some(Statistics::Int64(s)) => match (s.min_opt(), s.max_opt()) {
                (Some(&min), Some(&max)) => max >= lo && min < hi,
                _ => true,
            },
            _ => true,
        };
        if hit {
            keep.push(g);
        } else {
            skipped += 1;
        }
    }
    Some((keep, skipped))
}

/// The rows of `batch` whose Int64 `column` lies in `[lo, hi)`.
pub fn filter_range(batch: &RecordBatch, column: &str, lo: i64, hi: i64, path: &Path) -> Result<RecordBatch> {
    use arrow_array::{Array, BooleanArray, Int64Array};
    let c = batch
        .column_by_name(column)
        .context(WrongTableSnafu { path: path.to_path_buf(), expected: "time-ordered", column })?;
    let c = arrow_cast::cast(c, &arrow_schema::DataType::Int64).context(ArrowSnafu { context: format!("casting {column} to Int64") })?;
    let c = c
        .as_any()
        .downcast_ref::<Int64Array>()
        .context(WrongTableSnafu { path: path.to_path_buf(), expected: "time-ordered", column })?;
    let mask: BooleanArray = (0..c.len()).map(|i| Some(c.is_valid(i) && (lo..hi).contains(&c.value(i)))).collect();
    arrow_select::filter::filter_record_batch(batch, &mask).context(ArrowSnafu { context: format!("filtering {}", path.display()) })
}

/// `batch` with the columns (by name) and types of `schema`: files written by other versions may order or type
/// columns differently.
pub fn conform(batch: &RecordBatch, schema: &SchemaRef, path: &Path) -> Result<RecordBatch> {
    use arrow_array::Array;
    if batch.schema().fields() == schema.fields() {
        return RecordBatch::try_new(schema.clone(), batch.columns().to_vec())
            .context(ArrowSnafu { context: format!("re-labelling a batch of {}", path.display()) });
    }
    let mut cols = Vec::with_capacity(schema.fields().len());
    for f in schema.fields() {
        let c = match batch.column_by_name(f.name()) {
            Some(c) if c.data_type() == f.data_type() => c.clone(),
            Some(c) => arrow_cast::cast(c, f.data_type()).context(ArrowSnafu { context: format!("casting column {}", f.name()) })?,
            None if f.is_nullable() => arrow_array::new_null_array(f.data_type(), batch.num_rows()),
            None => {
                return WrongTableSnafu { path: path.to_path_buf(), expected: "matching", column: f.name().clone() }.fail();
            }
        };
        cols.push(c);
    }
    RecordBatch::try_new(schema.clone(), cols).context(ArrowSnafu { context: format!("conforming a batch of {}", path.display()) })
}

pub fn unix_millis() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

/// Expand files / directories into the list of finished `.parquet` files (sorted).
pub fn collect_parquet_files(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for input in inputs {
        if input.is_dir() {
            for entry in std::fs::read_dir(input).context(ReadDirSnafu { path: input.clone() })? {
                let entry = entry.context(ReadDirSnafu { path: input.clone() })?;
                let p = entry.path();
                if p.extension().is_some_and(|e| e == "parquet") {
                    out.push(p);
                }
            }
        } else {
            out.push(input.clone());
        }
    }
    out.sort();
    Ok(out)
}

/// The `gravity-trace.*` footer metadata of a file (keys without the prefix).
pub fn footer_meta(md: &ParquetMetaData) -> BTreeMap<String, String> {
    md.file_metadata()
        .key_value_metadata()
        .map(|kvs| {
            kvs.iter()
                .filter_map(|kv| {
                    let k = kv.key.strip_prefix(META_PREFIX)?;
                    Some((k.to_string(), kv.value.clone().unwrap_or_default()))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Open a Parquet file and return its schema, footer metadata and a batch reader.
pub fn open_reader(
    path: &Path,
) -> Result<(SchemaRef, BTreeMap<String, String>, parquet::arrow::arrow_reader::ParquetRecordBatchReader)> {
    let file = File::open(path).context(OpenFileSnafu { path: path.to_path_buf() })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).context(ParquetReadSnafu { path: path.to_path_buf() })?;
    let schema = builder.schema().clone();
    let meta = footer_meta(builder.metadata());
    let reader = builder.with_batch_size(64 * 1024).build().context(ParquetReadSnafu { path: path.to_path_buf() })?;
    Ok((schema, meta, reader))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{EventKind, TraceRow};
    use crate::schema::{TraceBatchBuilder, trace_schema};
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int64Type, UInt32Type};
    use parquet::file::reader::{FileReader, SerializedFileReader};

    #[test]
    fn trace_groups_are_sorted_and_delta_encoded() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("trace.parquet");
        let opts = WriteOptions { row_group_rows: 5, ..WriteOptions::trace() };
        let mut f = ParquetFile::create(path.clone(), trace_schema(), &opts).expect("create");
        // 12 rows in two batches, deliberately out of order: (pid, stream) interleaved, time descending
        for chunk in [0..7i64, 7..12] {
            let mut b = TraceBatchBuilder::default();
            for i in chunk {
                b.push(&TraceRow {
                    kind: Some(EventKind::Kernel),
                    name: "k",
                    start_ns: 1_000 - i * 10,
                    end_ns: 1_005 - i * 10,
                    pid: 2 - (i % 2) as u32,
                    device_id: Some(0),
                    stream_id: Some(7 + (i % 3) as u32),
                    correlation_id: i as u32,
                    ..Default::default()
                });
            }
            f.write(&b.finish().expect("batch")).expect("write");
        }
        assert_eq!(f.rows(), 12);
        let out = f.close().expect("close");

        let reader = SerializedFileReader::new(File::open(&out).expect("open")).expect("reader");
        let md = reader.metadata();
        let groups: Vec<i64> = (0..md.num_row_groups()).map(|g| md.row_group(g).num_rows()).collect();
        assert_eq!(groups, vec![5, 5, 2]);
        let schema = trace_schema();
        for name in TRACE_DELTA_COLUMNS {
            let ci = schema.index_of(name).expect("column");
            let enc = md.row_group(0).column(ci).encodings().collect::<Vec<_>>();
            assert!(enc.contains(&Encoding::DELTA_BINARY_PACKED), "{name}: {enc:?}");
        }

        let (_, _, batches) = open_reader(&out).expect("read back");
        let mut rows = Vec::new();
        for b in batches {
            let b = b.expect("batch");
            let pid = b.column_by_name("pid").expect("pid").as_primitive::<UInt32Type>().clone();
            let stream = b.column_by_name("stream_id").expect("stream").as_primitive::<UInt32Type>().clone();
            let start = b.column_by_name("start_ns").expect("start").as_primitive::<Int64Type>().clone();
            let end = b.column_by_name("end_ns").expect("end").as_primitive::<Int64Type>().clone();
            for i in 0..b.num_rows() {
                rows.push((pid.value(i), stream.value(i), start.value(i), end.value(i)));
            }
        }
        assert_eq!(rows.len(), 12);
        for (g, w) in [(0usize, 5usize), (5, 5), (10, 2)] {
            let grp = &rows[g..g + w];
            assert!(grp.windows(2).all(|x| (x[0].0, x[0].1, x[0].2) <= (x[1].0, x[1].1, x[1].2)), "group at {g}: {grp:?}");
        }
        assert!(rows.iter().all(|r| r.3 - r.2 == 5), "end - start preserved");
    }

    #[test]
    fn unsorted_files_keep_their_row_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("plain.parquet");
        let mut f = ParquetFile::create(path, trace_schema(), &WriteOptions::default()).expect("create");
        let mut b = TraceBatchBuilder::default();
        for i in 0..4i64 {
            b.push(&TraceRow { kind: Some(EventKind::Kernel), name: "k", start_ns: 100 - i, end_ns: 101 - i, ..Default::default() });
        }
        f.write(&b.finish().expect("batch")).expect("write");
        let out = f.close().expect("close");
        let (_, _, batches) = open_reader(&out).expect("read back");
        let starts: Vec<i64> = batches
            .flat_map(|b| b.expect("batch").column_by_name("start_ns").expect("start").as_primitive::<Int64Type>().values().to_vec())
            .collect();
        assert_eq!(starts, vec![100, 99, 98, 97]);
    }
}
