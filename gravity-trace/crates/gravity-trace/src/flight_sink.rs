//! A minimal Arrow Flight SQL server for bulk ingestion (`CommandStatementIngest`): keeps every ingested batch in
//! memory, or writes each DoPut to `<out>/<table>/<n>.parquet`. Not part of gt: the tests of [`crate::flight`] use it,
//! and `examples/flightsql_sink.rs` (the receiving end of `bench/flightsql_check.sh`) includes this file as a module,
//! so it only uses external crates.
#![allow(dead_code)]

use arrow_array::RecordBatch;
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_server::FlightServiceServer;
use arrow_flight::sql::server::{FlightSqlService, PeekableFlightDataStream};
use arrow_flight::sql::{CommandStatementIngest, SqlInfo};
use arrow_flight::{FlightData, HandshakeRequest, HandshakeResponse};
use futures::{Stream, TryStreamExt};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tonic::{Request, Response, Status, Streaming};

/// The bearer token the handshake hands out.
pub const TOKEN: &str = "gt-test-token";

#[derive(Default)]
pub struct Table {
    pub rows: u64,
    pub puts: u64,
    /// Bytes of the Flight messages received for this table.
    pub bytes: u64,
    /// The batches (without `out`).
    pub batches: Vec<RecordBatch>,
    /// Files written (with `out`).
    pub files: Vec<PathBuf>,
    /// Time receiving and decoding the DoPuts, and writing their Parquet files (seconds).
    pub receive_s: f64,
    pub write_s: f64,
}

#[derive(Default)]
pub struct State {
    pub tables: BTreeMap<String, Table>,
    /// Every ingest command received.
    pub commands: Vec<CommandStatementIngest>,
    /// The gRPC headers of every DoPut.
    pub headers: Vec<BTreeMap<String, String>>,
    /// Fail this many DoPuts (UNAVAILABLE) before accepting any.
    pub fail_puts: u32,
}

#[derive(Clone)]
pub struct Sink {
    pub state: Arc<Mutex<State>>,
    /// Bytes of every Flight message received (IPC header + body + app metadata + descriptor).
    pub bytes: Arc<AtomicU64>,
    pub out: Option<PathBuf>,
    /// (user, password) the handshake expects; then every DoPut needs the bearer token.
    pub auth: Option<(String, String)>,
}

impl Sink {
    pub fn new(out: Option<PathBuf>, auth: Option<(String, String)>) -> Sink {
        Sink { state: Arc::default(), bytes: Arc::default(), out, auth }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Bytes of one Flight message, as gt counts them.
pub fn message_bytes(fd: &FlightData) -> u64 {
    let desc = fd.flight_descriptor.as_ref().map_or(0, |d| d.cmd.len() + d.path.iter().map(String::len).sum::<usize>());
    (fd.data_header.len() + fd.data_body.len() + fd.app_metadata.len() + desc) as u64
}

fn base64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in b.chunks(3) {
        let n = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            s.push(if i <= c.len() { char::from(T[(n >> (18 - 6 * i)) as usize & 63]) } else { '=' });
        }
    }
    s
}

type HandshakeStream = Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>;

#[tonic::async_trait]
impl FlightSqlService for Sink {
    type FlightService = Sink;

    async fn do_handshake(&self, request: Request<Streaming<HandshakeRequest>>) -> Result<Response<HandshakeStream>, Status> {
        let Some((user, password)) = &self.auth else { return Err(Status::unimplemented("no authentication configured")) };
        let got = request.metadata().get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("");
        if got != format!("Basic {}", base64(format!("{user}:{password}").as_bytes())) {
            return Err(Status::unauthenticated("bad credentials"));
        }
        let out: HandshakeStream = Box::pin(futures::stream::iter([Ok(HandshakeResponse { protocol_version: 0, payload: Default::default() })]));
        let mut resp = Response::new(out);
        resp.metadata_mut().insert("authorization", format!("Bearer {TOKEN}").parse().map_err(|_| Status::internal("token header"))?);
        Ok(resp)
    }

    async fn do_put_statement_ingest(&self, cmd: CommandStatementIngest, request: Request<PeekableFlightDataStream>) -> Result<i64, Status> {
        if self.auth.is_some() && request.metadata().get("authorization").and_then(|v| v.to_str().ok()) != Some(&format!("Bearer {TOKEN}")) {
            return Err(Status::unauthenticated("no bearer token"));
        }
        {
            let mut st = self.lock();
            st.commands.push(cmd.clone());
            let headers = request.metadata().iter().filter_map(|kv| match kv {
                tonic::metadata::KeyAndValueRef::Ascii(k, v) => Some((k.to_string(), v.to_str().unwrap_or("").to_string())),
                tonic::metadata::KeyAndValueRef::Binary(..) => None,
            });
            st.headers.push(headers.collect());
            if st.fail_puts > 0 {
                st.fail_puts -= 1;
                return Err(Status::unavailable("injected failure"));
            }
        }
        let (total, put) = (self.bytes.clone(), Arc::new(AtomicU64::new(0)));
        let counted = put.clone();
        let data = request
            .into_inner()
            .map_ok(move |fd| {
                let n = message_bytes(&fd);
                total.fetch_add(n, Ordering::Relaxed);
                counted.fetch_add(n, Ordering::Relaxed);
                fd
            })
            .map_err(FlightError::from);
        let t0 = std::time::Instant::now();
        let batches: Vec<RecordBatch> =
            FlightRecordBatchStream::new_from_flight_data(data).try_collect().await.map_err(|e| Status::invalid_argument(e.to_string()))?;
        let receive_s = t0.elapsed().as_secs_f64();
        let rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
        let seq = {
            let mut st = self.lock();
            let t = st.tables.entry(cmd.table.clone()).or_default();
            t.rows += rows;
            t.puts += 1;
            t.bytes += put.load(Ordering::Relaxed);
            t.receive_s += receive_s;
            if self.out.is_none() {
                t.batches.extend(batches.iter().cloned());
            }
            t.puts
        };
        if let (Some(dir), Some(first)) = (&self.out, batches.first()) {
            let path = dir.join(&cmd.table).join(format!("{seq:06}.parquet"));
            let schema = first.schema();
            let p = path.clone();
            let t0 = std::time::Instant::now();
            tokio::task::spawn_blocking(move || -> Result<(), String> {
                if let Some(d) = p.parent() {
                    std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
                }
                let f = std::fs::File::create(&p).map_err(|e| e.to_string())?;
                let props = parquet::file::properties::WriterProperties::builder()
                    .set_compression(parquet::basic::Compression::ZSTD(parquet::basic::ZstdLevel::default()))
                    .build();
                let mut w = parquet::arrow::ArrowWriter::try_new(f, schema, Some(props)).map_err(|e| e.to_string())?;
                for b in &batches {
                    w.write(b).map_err(|e| e.to_string())?;
                }
                w.close().map_err(|e| e.to_string())?;
                Ok(())
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(|e| Status::internal(format!("writing {}: {e}", path.display())))?;
            let mut st = self.lock();
            let t = st.tables.entry(cmd.table.clone()).or_default();
            t.files.push(path);
            t.write_s += t0.elapsed().as_secs_f64();
        }
        Ok(rows as i64)
    }

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}

/// Serve `sink` on `listener` until `shutdown` completes. TCP_NODELAY on every connection (as gRPC servers set it):
/// without it the server's small HTTP/2 WINDOW_UPDATE frames wait for the client's delayed ACK (Nagle), while the
/// client waits for the window, and a DoPut crawls at about window / 40 ms.
pub async fn serve(
    listener: tokio::net::TcpListener,
    sink: Sink,
    shutdown: impl std::future::Future<Output = ()> + Send,
) -> Result<(), tonic::transport::Error> {
    tonic::transport::Server::builder()
        .http2_adaptive_window(Some(true))
        .add_service(FlightServiceServer::new(sink).max_decoding_message_size(64 << 20))
        .serve_with_incoming_shutdown(tonic::transport::server::TcpIncoming::from(listener).with_nodelay(Some(true)), shutdown)
        .await
}
