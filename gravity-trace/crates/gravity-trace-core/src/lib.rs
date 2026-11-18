//! Data model and output formats of gravity-trace (`gt`), independent of the Nsight Perf SDK.
//!
//! - [`model`]: event kinds and the row type NvTrace rows are mapped to
//! - [`hist`]: fixed-size log-linear latency histogram
//! - [`agg`]: per-window and cumulative kernel statistics (+ approximate GPU busy time)
//! - [`schema`] / [`parquet_io`]: Arrow schemas and zstd Parquet writers (trace files sorted per row group)
//! - [`chrome`]: Parquet trace -> Chrome Trace Event JSON (Perfetto / torch profiler format)
//! - [`metrics`]: per-process kernel / GPU metrics behind the Prometheus exporter
pub mod agg;
pub mod chrome;
pub mod error;
pub mod hist;
pub mod metrics;
pub mod model;
pub mod names;
pub mod parquet_io;
pub mod schema;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
