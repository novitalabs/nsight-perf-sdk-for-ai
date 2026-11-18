//! The files of a persisted trace (`persist.dir`): `<start>-<end>.parquet` activities, `.pm.parquet` PM samples and
//! `.json.gz` Perfetto JSON, found by their names. Read by `gt dump` and written by the daemon ([`crate::persist`]).
use crate::error::Result;
use crate::store::{Object, Store};
use crate::writer::{CHROME_EXT, PM_EXT, TRACE_EXT, parse_span};

/// What a persisted file holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FileKind {
    /// `.parquet`: activities.
    Trace,
    /// `.pm.parquet`: PM samples.
    Pm,
    /// `.json.gz`: Perfetto JSON (activities and PM samples).
    Perfetto,
}

/// A persisted file.
#[derive(Clone, Debug)]
pub struct StoredFile {
    pub name: String,
    pub span: (i64, i64),
    pub bytes: u64,
    pub kind: FileKind,
}

impl StoredFile {
    pub fn object(&self) -> Object {
        Object { name: self.name.clone(), size: self.bytes }
    }
}

/// Every `<start>-<end>.parquet` / `.pm.parquet` / `.json.gz` object of the store, sorted by span.
pub async fn list(store: &Store) -> Result<Vec<StoredFile>> {
    let mut out = Vec::new();
    for o in store.list().await? {
        let kind = [(PM_EXT, FileKind::Pm), (TRACE_EXT, FileKind::Trace), (CHROME_EXT, FileKind::Perfetto)]
            .into_iter()
            .find_map(|(ext, k)| parse_span(&o.name, ext).map(|s| (s, k)));
        let Some((span, kind)) = kind else { continue };
        out.push(StoredFile { name: o.name, span, bytes: o.size, kind });
    }
    out.sort_by_key(|f| (f.span, f.kind));
    Ok(out)
}

