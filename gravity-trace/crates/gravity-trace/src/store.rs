//! Where gt writes files and reads them back: an [`object_store`] location, so the same code writes to a local
//! directory today and to S3 (`s3://bucket/prefix`, built with `--features s3`) later.
//!
//! A file is uploaded while it is written ([`ObjectWriter`]: 8 MiB parts of a multipart upload, a single PUT when
//! smaller) under a hidden temporary name, and renamed to its final name once complete, because names such as
//! `<start>-<end>.parquet` are only known at the end. On a local directory the rename is a `rename(2)`; on S3 a
//! server-side copy. Readers never see a partial file: they only list names they can parse.
//!
//! Reading goes through the store as well: Parquet footers first, then only the column chunks of the row groups a
//! reader needs ([`Store::open_parquet`], [`Batches`]).
//!
//! The parts of gt that write and read files run on plain threads (merge / capture / persist / blocking tasks), so the
//! writer drives the store's async API with [`Handle::block_on`]; it must not be used from an async task.
use crate::error::{CoreSnafu, CreateDirSnafu, ReadParquetSnafu, Result, StoreSnafu, StoreUrlSnafu};
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use bytes::Bytes;
use futures::StreamExt;
use gravity_trace_core::parquet_io::{conform, filter_range, footer_meta, row_groups_in_range};
use object_store::path::Path as ObjPath;
use object_store::{MultipartUpload, ObjectStore, ObjectStoreExt, PutPayload};
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::{AsyncFileReader, MetadataSuffixFetch, ParquetRecordBatchStream, ParquetRecordBatchStreamBuilder};
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use snafu::ResultExt;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::runtime::Handle;

/// Bytes per multipart part (S3 needs at least 5 MiB for every part but the last).
const PART_BYTES: usize = 8 << 20;

/// An object store location: a directory, or a URL.
#[derive(Clone, Debug)]
pub struct Store {
    os: Arc<dyn ObjectStore>,
    prefix: ObjPath,
    /// How the location is shown: the directory, or the URL.
    shown: String,
    /// The directory, for a local store.
    local: Option<PathBuf>,
}

/// An object in a store.
#[derive(Clone, Debug)]
pub struct Object {
    pub name: String,
    pub size: u64,
}

impl Store {
    /// `/some/dir` or `relative/dir` (created if missing), `file:///some/dir`, `memory://`, `s3://bucket/prefix`
    /// (needs the `s3` feature; credentials and region from the usual `AWS_*` variables).
    pub fn open(location: &str) -> Result<Store> {
        if !location.contains("://") {
            return Self::local(Path::new(location));
        }
        let url = url::Url::parse(location).map_err(|e| StoreUrlSnafu { url: location.to_string(), message: e.to_string() }.build())?;
        match url.scheme() {
            "file" => {
                let dir = url
                    .to_file_path()
                    .map_err(|()| StoreUrlSnafu { url: location.to_string(), message: "not a local path".to_string() }.build())?;
                Self::local(&dir)
            }
            #[cfg(feature = "s3")]
            "s3" | "s3a" => {
                let os = object_store::aws::AmazonS3Builder::from_env()
                    .with_url(location)
                    .build()
                    .context(StoreSnafu { what: format!("opening {location}") })?;
                let prefix = ObjPath::parse(url.path().trim_matches('/'))
                    .map_err(|e| StoreUrlSnafu { url: location.to_string(), message: e.to_string() }.build())?;
                Ok(Store { os: Arc::new(os), prefix, shown: location.trim_end_matches('/').to_string(), local: None })
            }
            _ => {
                let (os, prefix) = object_store::parse_url(&url).map_err(|e| {
                    let hint = if url.scheme().starts_with("s3") { " (build gt with `--features s3`)" } else { "" };
                    StoreUrlSnafu { url: location.to_string(), message: format!("{e}{hint}") }.build()
                })?;
                Ok(Store { os: Arc::from(os), prefix, shown: location.trim_end_matches('/').to_string(), local: None })
            }
        }
    }

    fn local(dir: &Path) -> Result<Store> {
        std::fs::create_dir_all(dir).context(CreateDirSnafu { path: dir.to_path_buf() })?;
        let os = object_store::local::LocalFileSystem::new_with_prefix(dir).context(StoreSnafu { what: format!("opening {}", dir.display()) })?;
        let abs = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
        Ok(Store { os: Arc::new(os), prefix: ObjPath::default(), shown: dir.display().to_string(), local: Some(abs) })
    }

    pub fn path(&self, name: &str) -> ObjPath {
        self.prefix.clone().join(name)
    }

    /// How `name` is shown: a local path, or a URL.
    pub fn show(&self, name: &str) -> String {
        format!("{}/{name}", self.shown.trim_end_matches('/'))
    }

    pub fn location(&self) -> &str {
        &self.shown
    }

    /// Every object directly under the location.
    pub async fn list(&self) -> Result<Vec<Object>> {
        let what = || format!("listing {}", self.shown);
        let prefix = (!self.prefix.as_ref().is_empty()).then_some(&self.prefix);
        let r = self.os.list_with_delimiter(prefix).await.context(StoreSnafu { what: what() })?;
        Ok(r.objects
            .into_iter()
            .filter_map(|m| Some(Object { name: m.location.filename()?.to_string(), size: m.size }))
            .collect())
    }

    pub async fn delete(&self, name: &str) -> Result<()> {
        self.os.delete(&self.path(name)).await.context(StoreSnafu { what: format!("deleting {}", self.show(name)) })
    }

    /// A file written while it is produced, published as `final_name` by [`ObjectWriter::finish`].
    pub fn writer(&self, rt: &Handle, ext: &str) -> ObjectWriter {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let tmp = format!("{}{}-{}{ext}", crate::writer::TMP_PREFIX, std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed));
        ObjectWriter { store: self.clone(), rt: rt.clone(), tmp, upload: None, buf: Vec::new(), bytes: 0 }
    }

    /// Open `obj` as Parquet; `pick` chooses the row groups to read from its footer (`None`: all of them).
    pub fn open_parquet(&self, rt: &Handle, obj: &Object, pick: impl FnOnce(&ParquetMetaData) -> Option<Vec<usize>>) -> Result<ParquetReader> {
        let shown = self.show(&obj.name);
        let reader = ObjectReader { os: self.os.clone(), path: self.path(&obj.name) };
        let b = rt.block_on(ParquetRecordBatchStreamBuilder::new(reader)).context(ReadParquetSnafu { what: shown.clone() })?;
        let meta = footer_meta(b.metadata());
        let b = match pick(b.metadata()) {
            Some(groups) => b.with_row_groups(groups),
            None => b,
        };
        let stream = b.with_batch_size(64 * 1024).build().context(ReadParquetSnafu { what: shown.clone() })?;
        Ok(ParquetReader { meta, stream, rt: rt.clone(), shown })
    }

}

/// Parquet's async reader on an object store: the footer by a suffix request, then the byte ranges of the chosen
/// column chunks (as in parquet's `examples/object_store.rs`).
#[derive(Clone, Debug)]
struct ObjectReader {
    os: Arc<dyn ObjectStore>,
    path: ObjPath,
}

fn parquet_err(e: object_store::Error) -> parquet::errors::ParquetError {
    parquet::errors::ParquetError::External(Box::new(e))
}

impl AsyncFileReader for ObjectReader {
    fn get_bytes(&mut self, range: std::ops::Range<u64>) -> futures::future::BoxFuture<'_, parquet::errors::Result<Bytes>> {
        Box::pin(async move { self.os.get_range(&self.path, range).await.map_err(parquet_err) })
    }

    fn get_byte_ranges(&mut self, ranges: Vec<std::ops::Range<u64>>) -> futures::future::BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        Box::pin(async move { self.os.get_ranges(&self.path, &ranges).await.map_err(parquet_err) })
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> futures::future::BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        Box::pin(async move {
            let md = ParquetMetaDataReader::new().with_arrow_reader_options(options).load_via_suffix_and_finish(self).await?;
            Ok(Arc::new(md))
        })
    }
}

impl MetadataSuffixFetch for &mut ObjectReader {
    fn fetch_suffix(&mut self, suffix: usize) -> futures::future::BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let options = object_store::GetOptions { range: Some(object_store::GetRange::Suffix(suffix as u64)), ..Default::default() };
        Box::pin(async move {
            let r = self.os.get_opts(&self.path, options).await.map_err(parquet_err)?;
            r.bytes().await.map_err(parquet_err)
        })
    }
}

/// A Parquet object being read through the store.
pub struct ParquetReader {
    /// `gravity-trace.*` footer metadata.
    pub meta: BTreeMap<String, String>,
    stream: ParquetRecordBatchStream<ObjectReader>,
    rt: Handle,
    shown: String,
}

impl ParquetReader {
    pub fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        match self.rt.block_on(self.stream.next()) {
            None => Ok(None),
            Some(b) => b.map(Some).context(ReadParquetSnafu { what: self.shown.clone() }),
        }
    }
}

/// What [`Batches`] read in its last pass.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Scan {
    pub files: usize,
    pub row_groups: usize,
    /// Row groups whose statistics lie outside the range (not fetched).
    pub row_groups_skipped: usize,
    pub rows: u64,
}

/// The batches of Parquet objects in a store, optionally only the rows whose Int64 `filter.0` lies in
/// `[filter.1, filter.2)` (row groups outside it are not fetched), conformed to `schema`: what `gt dump` copies and the
/// Perfetto converter reads ([`gravity_trace_core::chrome::BatchSource`]).
pub struct Batches<'a> {
    pub store: &'a Store,
    pub rt: &'a Handle,
    pub objects: &'a [Object],
    pub filter: Option<(&'static str, i64, i64)>,
    pub schema: Option<SchemaRef>,
    pub scan: Scan,
}

impl Batches<'_> {
    pub fn each(&mut self, mut f: impl FnMut(&std::path::Path, &BTreeMap<String, String>, RecordBatch) -> Result<()>) -> Result<()> {
        let mut scan = Scan::default();
        let filter = self.filter;
        for o in self.objects {
            let label = PathBuf::from(self.store.show(&o.name));
            let (mut groups, mut skipped) = (0, 0);
            let mut r = self.store.open_parquet(self.rt, o, |md| {
                let all = || ((0..md.num_row_groups()).collect::<Vec<_>>(), 0);
                let (keep, sk) = match filter {
                    Some((c, lo, hi)) => row_groups_in_range(md, c, lo, hi).unwrap_or_else(all),
                    None => all(),
                };
                (groups, skipped) = (keep.len(), sk);
                Some(keep)
            })?;
            scan.files += 1;
            scan.row_groups += groups;
            scan.row_groups_skipped += skipped;
            while let Some(b) = r.next_batch()? {
                let b = match filter {
                    Some((c, lo, hi)) => filter_range(&b, c, lo, hi, &label).context(CoreSnafu)?,
                    None => b,
                };
                if b.num_rows() == 0 {
                    continue;
                }
                let b = match &self.schema {
                    Some(s) => conform(&b, s, &label).context(CoreSnafu)?,
                    None => b,
                };
                scan.rows += b.num_rows() as u64;
                f(&label, &r.meta, b)?;
            }
        }
        self.scan = scan;
        Ok(())
    }
}

impl gravity_trace_core::chrome::BatchSource for Batches<'_> {
    fn for_each(&mut self, f: &mut gravity_trace_core::chrome::BatchFn<'_>) -> gravity_trace_core::error::Result<()> {
        match self.each(|p, m, b| f(p, m, &b).context(CoreSnafu)) {
            Ok(()) => Ok(()),
            Err(crate::error::Error::Core { source }) => Err(source),
            Err(e) => Err(gravity_trace_core::error::Error::Input { what: "reading trace Parquet from the store".to_string(), source: Box::new(e) }),
        }
    }
}

/// A file being uploaded (see the module docs); also a [`std::io::Write`] for the Parquet / JSON writers.
pub struct ObjectWriter {
    store: Store,
    rt: Handle,
    tmp: String,
    upload: Option<Box<dyn MultipartUpload>>,
    buf: Vec<u8>,
    bytes: u64,
}

fn io_err(e: object_store::Error) -> std::io::Error {
    std::io::Error::other(e)
}

impl ObjectWriter {
    fn send_part(&mut self) -> std::io::Result<()> {
        let part = std::mem::replace(&mut self.buf, Vec::with_capacity(PART_BYTES));
        let (store, rt) = (&self.store, &self.rt);
        let upload = match self.upload.as_mut() {
            Some(u) => u,
            None => self.upload.insert(rt.block_on(store.os.put_multipart(&store.path(&self.tmp))).map_err(io_err)?),
        };
        rt.block_on(upload.put_part(PutPayload::from(part))).map_err(io_err)
    }

    /// Bytes written so far.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Complete the upload and publish it as `name`; the location shown for it.
    pub fn finish(mut self, name: &str) -> Result<String> {
        let what = format!("writing {}", self.store.show(name));
        let r: std::result::Result<(), object_store::Error> = (|| {
            match self.upload.take() {
                Some(mut u) => {
                    if !self.buf.is_empty() {
                        let last = PutPayload::from(std::mem::take(&mut self.buf));
                        self.rt.block_on(u.put_part(last))?;
                    }
                    self.rt.block_on(u.complete())?;
                }
                None => {
                    let body = PutPayload::from(Bytes::from(std::mem::take(&mut self.buf)));
                    self.rt.block_on(self.store.os.put(&self.store.path(&self.tmp), body))?;
                }
            }
            self.rt.block_on(self.store.os.rename(&self.store.path(&self.tmp), &self.store.path(name)))
        })();
        r.context(StoreSnafu { what })?;
        Ok(self.store.show(name))
    }

    /// Drop the upload (after a write error): nothing is published, later writes go nowhere.
    pub fn abort(&mut self) {
        if let Some(mut u) = self.upload.take() {
            let _ = self.rt.block_on(u.abort());
        }
        self.buf = Vec::new();
    }
}

impl Write for ObjectWriter {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(b);
        self.bytes += b.len() as u64;
        if self.buf.len() >= PART_BYTES {
            self.send_part()?;
        }
        Ok(b.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Remove temporaries of processes that are gone (a crash or kill -9 while a file was written), local stores only:
/// on S3, abandoned multipart uploads are cleaned up by a bucket lifecycle rule.
pub fn remove_stale_temporaries(store: &Store) {
    let Some(dir) = &store.local else { return };
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(pid) = name.strip_prefix(crate::writer::TMP_PREFIX).and_then(|n| n.split('-').next()) else { continue };
        if pid.parse::<u32>().is_ok_and(|p| p != std::process::id() && !Path::new(&format!("/proc/{p}")).exists()) {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_and_multipart_files_are_published_under_their_final_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build().expect("runtime");
        let store = Store::open(&dir.path().display().to_string()).expect("store");
        let h = rt.handle().clone();
        let mut small = store.writer(&h, ".bin");
        small.write_all(b"hello").expect("write");
        assert_eq!(small.finish("small.bin").expect("finish"), format!("{}/small.bin", dir.path().display()));
        let mut big = store.writer(&h, ".bin");
        let chunk = vec![7u8; 3 << 20];
        for _ in 0..5 {
            big.write_all(&chunk).expect("write"); // 15 MiB: one part sent while writing, the rest at finish
        }
        big.finish("big.bin").expect("finish");
        let mut aborted = store.writer(&h, ".bin");
        aborted.write_all(&vec![1u8; PART_BYTES + 1]).expect("write");
        aborted.abort();
        let mut names: Vec<(String, u64)> = rt.block_on(store.list()).expect("list").into_iter().map(|o| (o.name, o.size)).collect();
        names.sort();
        assert_eq!(names, [("big.bin".to_string(), 15 << 20), ("small.bin".to_string(), 5)], "no temporaries visible");
        assert_eq!(std::fs::read(dir.path().join("small.bin")).expect("read"), b"hello");
        rt.block_on(store.delete("small.bin")).expect("delete");
        assert!(!dir.path().join("small.bin").exists());
    }

    #[test]
    fn locations() {
        let dir = tempfile::tempdir().expect("tempdir");
        let url = format!("file://{}/sub", dir.path().display());
        let s = Store::open(&url).expect("file url");
        assert!(dir.path().join("sub").is_dir(), "created");
        assert!(s.local.is_some());
        assert!(Store::open("memory://").is_ok());
        let e = Store::open("s3://bucket/prefix");
        if cfg!(not(feature = "s3")) {
            assert!(e.is_err_and(|e| e.to_string().contains("--features s3")));
        }
    }
}
