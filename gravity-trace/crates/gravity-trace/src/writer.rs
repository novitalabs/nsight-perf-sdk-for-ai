//! Trace / PM-metric Parquet and Perfetto output, shared by `gt import`, live captures, the daemon's persisted trace
//! and `gt dump`. Everything is written through a [`Store`] (object_store: a local directory or S3).
//!
//! Every file is named by the UTC time range it covers, in nanoseconds since the Unix epoch:
//! `<start>-<end>.parquet` (trace), `<start>-<end>.pm.parquet` (PM samples), `<start>-<end>.json.gz` (Perfetto).
//! The range is the one asked for (a capture's or a dump's window) or else the span of the rows (smallest start,
//! largest end). Files are uploaded under a hidden temporary name and renamed once complete.
//!
//! `gt dump` uses the names, the PM schema and the Perfetto conversion; the writers of activities and PM samples
//! (live captures, the persisted trace, `gt import`) come with the `monitor` feature.
use crate::error::{CoreSnafu, IoSnafu, Result};
use crate::store::{Batches, Store};
use arrow_schema::SchemaRef;
use gravity_trace_core::chrome::ConvertOptions;
use snafu::ResultExt;
use std::sync::{Arc, OnceLock};
use tokio::runtime::Handle;
#[cfg(feature = "monitor")]
use {
    crate::error::ArrowSnafu,
    crate::feed::Names,
    crate::mapper::{Activity, PmSample},
    crate::store::{Object, ObjectWriter},
    gravity_trace_core::chrome::{PROCESSES_META, StreamWriter},
    gravity_trace_core::model::ProcessInfo,
    gravity_trace_core::parquet_io::{TableWriter, WriteOptions},
    gravity_trace_core::schema::{TraceBatchBuilder, trace_schema},
    nvtrace_sys as nvt,
    std::collections::BTreeMap,
    std::path::PathBuf,
};

#[cfg(feature = "monitor")]
/// Rows buffered before they are handed to the Parquet writer as one Arrow batch.
const BATCH_ROWS: usize = 64 * 1024;

pub const TRACE_EXT: &str = ".parquet";
pub const PM_EXT: &str = ".pm.parquet";
pub const CHROME_EXT: &str = ".json.gz";
/// Prefix of the temporary names files are uploaded under (`.gt-<pid>-<n>.parquet`).
pub const TMP_PREFIX: &str = ".gt-";

/// `<start>-<end><ext>`.
pub fn span_name(span: (i64, i64), ext: &str) -> String {
    format!("{}-{}{ext}", span.0.max(0), span.1.max(0))
}

/// The range of a `<start>-<end><ext>` file name.
pub fn parse_span(name: &str, ext: &str) -> Option<(i64, i64)> {
    let (a, b) = name.strip_suffix(ext)?.split_once('-')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit());
    if !digits(a) || !digits(b) {
        return None;
    }
    Some((a.parse().ok()?, b.parse().ok()?))
}

#[cfg(feature = "monitor")]
fn widen(span: &mut Option<(i64, i64)>, start: i64, end: i64) {
    let end = end.max(start);
    *span = Some(match *span {
        Some((s, e)) => (s.min(start), e.max(end)),
        None => (start, end),
    });
}

/// A file written to a store.
#[derive(Clone, Debug)]
pub struct Written {
    pub name: String,
    /// Where it is (a local path or a URL).
    pub shown: String,
    pub bytes: u64,
}

#[cfg(feature = "monitor")]
/// Writes activities to one trace Parquet file (`WriteOptions::trace`: sorted, delta-encoded row groups). The file is
/// created on the first row, so an empty capture leaves no file behind.
pub struct TraceWriter {
    builder: TraceBatchBuilder,
    table: Option<TableWriter<ObjectWriter>>,
    store: Store,
    rt: Handle,
    opts: WriteOptions,
    rows: u64,
    span: Option<(i64, i64)>,
    /// Written to the footer ([`PROCESSES_META`]) when the file is closed.
    processes: Vec<ProcessInfo>,
}

#[cfg(feature = "monitor")]
impl TraceWriter {
    /// A trace file in `store` with the default trace layout and `metadata` in the footer.
    pub fn new(store: &Store, rt: &Handle, metadata: BTreeMap<String, String>) -> Self {
        Self::with_options(store, rt, WriteOptions { metadata, ..WriteOptions::trace() })
    }

    pub fn with_options(store: &Store, rt: &Handle, mut opts: WriteOptions) -> Self {
        opts.metadata.entry("host".to_string()).or_insert_with(crate::host);
        Self {
            builder: TraceBatchBuilder::default(),
            table: None,
            store: store.clone(),
            rt: rt.clone(),
            opts,
            rows: 0,
            span: None,
            processes: Vec::new(),
        }
    }

    /// The processes of the file (their names for `gt dump perfetto`), recorded in its footer.
    pub fn set_processes(&mut self, processes: Vec<ProcessInfo>) {
        self.processes = processes;
    }

    pub fn push(&mut self, a: &Activity) -> Result<()> {
        self.builder.push(&a.row());
        self.rows += 1;
        widen(&mut self.span, a.start_ns, a.end_ns);
        if self.builder.len() >= BATCH_ROWS {
            self.flush()?;
        }
        Ok(())
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    fn flush(&mut self) -> Result<()> {
        if self.builder.is_empty() {
            return Ok(());
        }
        let batch = self.builder.finish().context(ArrowSnafu { context: "building a trace batch".to_string() })?;
        let table = match self.table.as_mut() {
            Some(t) => t,
            None => {
                let sink = self.store.writer(&self.rt, TRACE_EXT);
                let t = TableWriter::new(sink, PathBuf::from(self.store.location()), trace_schema(), &self.opts).context(CoreSnafu)?;
                self.table.insert(t)
            }
        };
        table.write(&batch).context(CoreSnafu)
    }

    /// Drop the file being written (after a write error): nothing is published.
    pub fn abandon(mut self) {
        if let Some(mut t) = self.table.take() {
            t.sink_mut().abort();
        }
    }

    /// Write the remaining rows and the footer and publish the file as `<start>-<end>.parquet`: `window` if given,
    /// else the span of the rows. `None` when no row was written.
    pub fn close(mut self, window: Option<(i64, i64)>) -> Result<Option<Written>> {
        self.flush()?;
        let (Some(mut t), Some(span)) = (self.table.take(), window.or(self.span)) else { return Ok(None) };
        if !self.processes.is_empty() {
            // a list of plain structs: serializing cannot fail
            t.append_metadata(PROCESSES_META, serde_json::to_string(&self.processes).unwrap_or_default());
        }
        let sink = t.finish().context(CoreSnafu)?;
        let (name, bytes) = (span_name(span, TRACE_EXT), sink.bytes());
        let shown = sink.finish(&name)?;
        Ok(Some(Written { name, shown, bytes }))
    }
}

#[cfg(feature = "monitor")]
/// Activities and PM samples written as Perfetto / Chrome trace JSON while they arrive, one gzip file published as
/// `<start>-<end>.json.gz` (the span of its contents): the persisted trace with `persist.format = "perfetto"`
/// ([`gravity_trace_core::chrome::StreamWriter`]).
pub struct PerfettoWriter {
    w: StreamWriter<flate2::write::GzEncoder<ObjectWriter>>,
    /// GPU UUID -> how the counter tracks name it.
    uuids: rustc_hash::FxHashMap<[u8; 16], String>,
    rows: u64,
    span: Option<(i64, i64)>,
}

#[cfg(feature = "monitor")]
impl PerfettoWriter {
    /// A file in `store` that names its tracks after `names` (and later [`PerfettoWriter::names`]), gzip level
    /// `gzip_level` (1-9: it is compressed while the trace is collected, so it has to keep up).
    pub fn new(store: &Store, rt: &Handle, names: &Names, gzip_level: u32) -> Result<Self> {
        let gz = flate2::write::GzEncoder::new(store.writer(rt, CHROME_EXT), flate2::Compression::new(gzip_level));
        let mut p = Self { w: StreamWriter::new(gz, &ConvertOptions::default()), uuids: Default::default(), rows: 0, span: None };
        p.names(names)?;
        Ok(p)
    }

    pub fn names(&mut self, n: &Names) -> Result<()> {
        for p in &n.procs {
            self.w.set_process(p).context(CoreSnafu)?;
        }
        for (uuid, device) in &n.gpus {
            let u = self.uuid(uuid).to_string();
            self.w.set_gpu_name(&u, device);
        }
        Ok(())
    }

    fn uuid(&mut self, u: &[u8; 16]) -> &str {
        self.uuids.entry(*u).or_insert_with(|| nvt::uuid_string(u))
    }

    /// A chunk of activities (its API calls are noted first: GPU work may come before its launch in the chunk).
    pub fn push(&mut self, rows: &[Activity]) -> Result<()> {
        for a in rows.iter().filter(|a| !a.kind.is_gpu()) {
            self.w.note_api(&a.row());
        }
        for a in rows {
            self.w.push(a.kind.as_str(), &a.row()).context(CoreSnafu)?;
            self.rows += 1;
            widen(&mut self.span, a.start_ns, a.end_ns);
        }
        Ok(())
    }

    /// PM samples, as counter tracks.
    pub fn pm(&mut self, samples: &[PmSample]) -> Result<()> {
        for s in samples {
            let u = self.uuid(&s.gpu).to_string();
            self.w.counter(&u, &s.metric, s.end_utc_ns as i64, s.value).context(CoreSnafu)?;
            widen(&mut self.span, s.end_utc_ns as i64, s.end_utc_ns as i64);
        }
        Ok(())
    }

    /// A snapshot boundary (API calls age out of the flow arrows).
    pub fn snapshot(&mut self) {
        self.w.snapshot();
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Drop the file being written (after a write error): nothing is published.
    pub fn abandon(mut self) {
        self.w.get_mut().get_mut().abort();
    }

    /// Finish the document and publish it under the span of its contents; `None` when nothing was written.
    pub fn close(mut self) -> Result<Option<Written>> {
        let Some(span) = self.span else {
            self.w.get_mut().get_mut().abort();
            return Ok(None);
        };
        let name = span_name(span, CHROME_EXT);
        let (gz, _) = self.w.finish().context(CoreSnafu)?;
        let sink = gz.finish().context(IoSnafu { context: format!("compressing {name}") })?;
        let bytes = sink.bytes();
        let shown = sink.finish(&name)?;
        Ok(Some(Written { name, shown, bytes }))
    }
}

/// Schema of PM-metric files: `gpu_uuid, end_ns, metric, value`.
pub fn pm_schema() -> SchemaRef {
    use arrow_schema::{DataType, Field, Schema};
    static S: OnceLock<SchemaRef> = OnceLock::new();
    S.get_or_init(|| {
        Arc::new(Schema::new(vec![
            Field::new("gpu_uuid", DataType::Utf8, false),
            Field::new("end_ns", DataType::Int64, false),
            Field::new("metric", DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)), false),
            Field::new("value", DataType::Float64, false),
        ]))
    })
    .clone()
}

#[cfg(feature = "monitor")]
/// PM metric samples as one [`pm_schema`] batch, with the span of their times; `None` when there are none.
pub fn pm_batch(samples: &[PmSample]) -> Result<Option<(arrow_array::RecordBatch, (i64, i64))>> {
    use arrow_array::builder::{Float64Builder, Int64Builder, StringBuilder, StringDictionaryBuilder};
    use arrow_array::types::Int32Type;
    let mut span = None;
    let (mut g, mut t, mut m, mut v) =
        (StringBuilder::new(), Int64Builder::new(), StringDictionaryBuilder::<Int32Type>::new(), Float64Builder::new());
    for s in samples {
        g.append_value(nvt::uuid_string(&s.gpu));
        t.append_value(s.end_utc_ns as i64);
        m.append_value(&*s.metric);
        v.append_value(s.value);
        widen(&mut span, s.end_utc_ns as i64, s.end_utc_ns as i64);
    }
    let Some(span) = span else { return Ok(None) };
    let batch = arrow_array::RecordBatch::try_new(
        pm_schema(),
        vec![Arc::new(g.finish()), Arc::new(t.finish()), Arc::new(m.finish()), Arc::new(v.finish())],
    )
    .context(ArrowSnafu { context: "building the PM batch".to_string() })?;
    Ok(Some((batch, span)))
}

#[cfg(feature = "monitor")]
/// PM metric samples -> `<start>-<end>.pm.parquet` (`window`, else the span of the sample times); `None` when there
/// are no samples.
pub fn write_pm(store: &Store, rt: &Handle, samples: &[PmSample], window: Option<(i64, i64)>) -> Result<Option<Written>> {
    let Some((batch, span)) = pm_batch(samples)? else { return Ok(None) };
    let span = window.unwrap_or(span);
    let opts = WriteOptions { metadata: BTreeMap::from([("host".to_string(), crate::host())]), ..WriteOptions::default() };
    let mut table =
        TableWriter::new(store.writer(rt, PM_EXT), PathBuf::from(store.location()), pm_schema(), &opts).context(CoreSnafu)?;
    table.write(&batch).context(CoreSnafu)?;
    let sink = table.finish().context(CoreSnafu)?;
    let (name, bytes) = (span_name(span, PM_EXT), sink.bytes());
    let shown = sink.finish(&name)?;
    Ok(Some(Written { name, shown, bytes }))
}

/// Perfetto / Chrome trace JSON (gzip, [`gravity_trace_core::chrome::GZIP_LEVEL`]) of the batches of `src` (and PM
/// samples from `pm` when given), published in `store` as `name`.
pub fn write_chrome_from(store: &Store, rt: &Handle, src: &mut Batches<'_>, pm: Option<&mut Batches<'_>>, name: &str) -> Result<Written> {
    use gravity_trace_core::chrome::convert_from;
    use std::io::Write as _;
    let mut gz = flate2::write::GzEncoder::new(store.writer(rt, CHROME_EXT), flate2::Compression::new(gravity_trace_core::chrome::GZIP_LEVEL));
    convert_from(src, pm.map(|b| b as &mut dyn gravity_trace_core::chrome::BatchSource), &mut gz, &ConvertOptions::default()).context(CoreSnafu)?;
    let mut sink = gz.finish().context(IoSnafu { context: format!("compressing {}", store.show(name)) })?;
    sink.flush().context(IoSnafu { context: format!("writing {}", store.show(name)) })?;
    let bytes = sink.bytes();
    let shown = sink.finish(name)?;
    Ok(Written { name: name.to_string(), shown, bytes })
}

#[cfg(feature = "monitor")]
/// Perfetto JSON of a trace file just written to `store`, next to it as `<start>-<end>.json.gz` (read back through
/// the store).
pub fn write_chrome(store: &Store, rt: &Handle, trace: &Written) -> Result<Written> {
    let stem = trace.name.strip_suffix(TRACE_EXT).unwrap_or(&trace.name);
    let objects = [Object { name: trace.name.clone(), size: trace.bytes }];
    let mut src = Batches { store, rt, objects: &objects, filter: None, schema: None, scan: Default::default() };
    write_chrome_from(store, rt, &mut src, None, &format!("{stem}{CHROME_EXT}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_names_round_trip() {
        let name = span_name((1_763_280_000_123_456_789, 1_763_280_060_000_000_000), TRACE_EXT);
        assert_eq!(name, "1763280000123456789-1763280060000000000.parquet");
        assert_eq!(parse_span(&name, TRACE_EXT), Some((1_763_280_000_123_456_789, 1_763_280_060_000_000_000)));
        assert_eq!(parse_span("1-2.pm.parquet", TRACE_EXT), None, "PM files are not trace files");
        assert_eq!(parse_span("1-2.pm.parquet", PM_EXT), Some((1, 2)));
        assert_eq!(parse_span(".gt-7-0.parquet", TRACE_EXT), None);
        assert_eq!(parse_span("trace-nvp-host-17.parquet", TRACE_EXT), None);
    }
}
