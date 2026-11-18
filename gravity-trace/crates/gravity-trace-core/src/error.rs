//! Crate-wide error type (snafu).
use snafu::Snafu;
use std::path::PathBuf;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("failed to create directory {}", path.display()))]
    CreateDir { path: PathBuf, source: std::io::Error },

    #[snafu(display("failed to create file {}", path.display()))]
    CreateFile { path: PathBuf, source: std::io::Error },

    #[snafu(display("failed to open {}", path.display()))]
    OpenFile { path: PathBuf, source: std::io::Error },

    #[snafu(display("failed to read directory {}", path.display()))]
    ReadDir { path: PathBuf, source: std::io::Error },

    #[snafu(display("failed to rename {} -> {}", from.display(), to.display()))]
    Rename { from: PathBuf, to: PathBuf, source: std::io::Error },

    #[snafu(display("failed to write {}", path.display()))]
    WriteOutput { path: PathBuf, source: std::io::Error },

    #[snafu(display("invalid zstd level {level}"))]
    ZstdLevel { level: i32, source: parquet::errors::ParquetError },

    #[snafu(display("parquet write to {} failed", path.display()))]
    ParquetWrite { path: PathBuf, source: parquet::errors::ParquetError },

    #[snafu(display("parquet read from {} failed", path.display()))]
    ParquetRead { path: PathBuf, source: parquet::errors::ParquetError },

    #[snafu(display("arrow error while {context}"))]
    Arrow { context: String, source: arrow_schema::ArrowError },

    #[snafu(display("{} is not a {expected} table (missing column `{column}`)", path.display()))]
    WrongTable { path: PathBuf, expected: &'static str, column: String },

    #[snafu(display("no parquet files found in the given inputs"))]
    NoInputs,

    #[snafu(display("{what}"))]
    Input { what: String, source: Box<dyn std::error::Error + Send + Sync> },

    #[snafu(display("json encoding failed"))]
    Json { source: serde_json::Error },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
