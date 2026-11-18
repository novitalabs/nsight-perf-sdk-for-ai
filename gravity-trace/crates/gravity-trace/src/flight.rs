//! Streams every merged activity (and PM sample) to an Arrow Flight SQL server while gt collects (`gt profile
//! --flightsql`, `[flightsql]` of `gt export-daemon`): bulk ingestion, i.e. `CommandStatementIngest` in a DoPut, into
//! `table` (activities) and `pm_table` (PM samples), created by the server when missing and appended to.
//!
//! - Rows are the trace Parquet schema with `host` (and the constant `columns`) in front, so one table can hold many
//!   nodes. Dictionary columns (`kind`, `name`, ...) go out as plain strings unless `dictionaries` is set, Arrow IPC
//!   bodies uncompressed unless `compression` says otherwise.
//! - One DoPut per merged snapshot (at most `put_rows` rows each), on its own thread (`gt-flight`). The merge thread
//!   hands activities over through the same bounded queue as the persisted trace ([`crate::feed`]): when the server
//!   is slower than the trace, chunks are dropped and counted, the snapshots never wait.
//! - A failed DoPut is retried `retries` times on a new connection, then its rows are dropped and counted. A retry
//!   after a failure the server did not report (e.g. the connection broke after it committed) ingests rows twice.
//! - `max_send_mib_s` caps the sending rate (each snapshot is otherwise sent as fast as the network takes it).
//! - Every Arrow IPC message is counted as it is handed to gRPC (the bytes on the wire minus gRPC / HTTP/2 framing,
//!   < 0.1%), and the count is sampled every second: `nvtrace_flight_bytes_total` and the per-second traffic of the
//!   `gt profile` report.
use crate::config;
use crate::error::{ArrowSnafu, FlightAuthSnafu, FlightConnectSnafu, FlightEndpointSnafu, Result, SpawnThreadSnafu};
use crate::feed::{Feed, Msg, Queue};
use crate::mapper::{Activity, PmSample};
use crate::writer::pm_batch;
use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, DictionaryArray, Int32Array, RecordBatch, StringArray};
use arrow_flight::encode::{DictionaryHandling, FlightDataEncoderBuilder};
use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::sql::client::FlightSqlServiceClient;
use arrow_flight::sql::{CommandStatementIngest, DoPutUpdateResult, ProstMessageExt, TableDefinitionOptions, TableExistsOption, TableNotExistOption};
use arrow_flight::{FlightData, FlightDescriptor};
use arrow_ipc::writer::IpcWriteOptions;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use gravity_trace_core::schema::{TraceBatchBuilder, trace_schema};
use prost::Message;
use snafu::ResultExt;
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};
use tokio::runtime::Handle;
use tonic::metadata::{AsciiMetadataKey, MetadataMap, MetadataValue};
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};

/// Per-second samples kept (a day).
const MAX_SECONDS: usize = 86_400;

pub use crate::config::Compression;

#[derive(Clone, Debug)]
pub struct FlightOptions {
    /// `grpc://host:port`, `grpc+tls://host:port`, `http://...`, `https://...`, or `host:port`.
    pub endpoint: String,
    pub table: String,
    /// "" = PM samples are not sent.
    pub pm_table: String,
    pub catalog: String,
    pub db_schema: String,
    /// Let the server create missing tables (else ingesting into one fails).
    pub create: bool,
    pub username: String,
    pub password: String,
    pub token: String,
    /// gRPC headers on every call.
    pub headers: BTreeMap<String, String>,
    /// `CommandStatementIngest.options` (server specific).
    pub ingest_options: BTreeMap<String, String>,
    /// Constant columns in front of every row, `host` first.
    pub columns: Vec<(String, String)>,
    pub tls_ca_file: Option<PathBuf>,
    pub batch_rows: usize,
    pub put_rows: usize,
    pub max_buffer_bytes: usize,
    /// Bytes per second; 0 = no cap.
    pub max_send_bytes_s: u64,
    pub dictionaries: bool,
    pub compression: Compression,
    pub timeout: Duration,
    pub retries: u32,
}

impl FlightOptions {
    /// From the configuration (the environment is already in it: `$GT_FLIGHTSQL_PASSWORD`, ...).
    pub fn from_config(f: &config::FlightSql) -> FlightOptions {
        let mut columns = vec![("host".to_string(), f.columns.get("host").cloned().unwrap_or_else(crate::host))];
        columns.extend(f.columns.iter().filter(|(k, _)| *k != "host").map(|(k, v)| (k.clone(), v.clone())));
        FlightOptions {
            endpoint: f.endpoint.clone(),
            table: f.table.clone(),
            pm_table: f.pm_table.clone(),
            catalog: f.catalog.clone(),
            db_schema: f.db_schema.clone(),
            create: f.create,
            username: f.username.clone(),
            password: f.password.clone(),
            token: f.token.clone(),
            headers: f.headers.clone(),
            ingest_options: f.options.clone(),
            columns,
            tls_ca_file: (!f.tls_ca_file.is_empty()).then(|| PathBuf::from(&f.tls_ca_file)),
            batch_rows: f.batch_rows,
            put_rows: f.put_rows,
            max_buffer_bytes: (f.max_buffer_mib as usize).saturating_mul(1 << 20),
            max_send_bytes_s: f.max_send_mib_s.saturating_mul(1 << 20),
            dictionaries: f.dictionaries,
            compression: f.compression,
            timeout: Duration::from_secs(f.timeout_s),
            retries: f.retries,
        }
    }

    /// The endpoint as an `http(s)://` URI, and whether it uses TLS.
    pub fn uri(&self) -> Result<(String, bool)> {
        let e = self.endpoint.trim();
        let schemes = [("grpc+tls://", true), ("grpc+tcp://", false), ("grpc://", false), ("https://", true), ("http://", false)];
        let (rest, tls) = match schemes.iter().find_map(|(p, tls)| e.strip_prefix(p).map(|r| (r, *tls))) {
            Some(x) => x,
            None if !e.contains("://") => (e, false),
            None => {
                return FlightEndpointSnafu { endpoint: e.to_string(), message: "use grpc://, grpc+tls://, http:// or https://".to_string() }
                    .fail();
            }
        };
        let rest = rest.trim_end_matches('/');
        if rest.is_empty() {
            return FlightEndpointSnafu { endpoint: e.to_string(), message: "no host".to_string() }.fail();
        }
        Ok((format!("{}://{rest}", if tls { "https" } else { "http" }), tls))
    }
}

/// Counters behind the `nvtrace_flight_*` series and the `gt profile` report.
#[derive(Debug, Default)]
pub struct FlightStats {
    /// The send buffer (and the activities dropped: buffer full, or every attempt of their DoPut failed).
    pub queue: Arc<Queue>,
    /// Activities in successful DoPuts.
    pub rows: AtomicU64,
    pub pm_samples: AtomicU64,
    pub dropped_pm_samples: AtomicU64,
    /// Rows the server reported ingesting (`DoPutUpdateResult.record_count`, when it reports them).
    pub server_rows: AtomicU64,
    /// Successful DoPuts.
    pub puts: AtomicU64,
    /// Failed attempts (connections and DoPuts).
    pub errors: AtomicU64,
    pub retries: AtomicU64,
    /// Arrow IPC bytes handed to gRPC (failed attempts included).
    pub bytes: AtomicU64,
    /// Time spent in DoPuts (ns).
    pub put_ns: AtomicU64,
    /// Bytes sent in each second since the stream started (the last day).
    pub per_second: Mutex<VecDeque<u64>>,
    /// The sender thread has finished.
    pub closed: AtomicBool,
}

impl FlightStats {
    pub fn seconds(&self) -> Vec<u64> {
        self.per_second.lock().unwrap_or_else(|e| e.into_inner()).iter().copied().collect()
    }
}

/// Per-second traffic: what `gt profile` reports.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Traffic {
    pub seconds: usize,
    pub total_bytes: u64,
    pub mean_bytes_per_s: f64,
    pub p50_bytes_per_s: u64,
    pub p90_bytes_per_s: u64,
    pub p99_bytes_per_s: u64,
    pub max_bytes_per_s: u64,
    /// Seconds in which something was sent.
    pub active_seconds: usize,
    /// Bytes sent in each second.
    pub per_second: Vec<u64>,
}

impl Traffic {
    pub fn of(per_second: Vec<u64>) -> Traffic {
        let n = per_second.len();
        let total: u64 = per_second.iter().sum();
        let mut sorted = per_second.clone();
        sorted.sort_unstable();
        let q = |p: f64| if n == 0 { 0 } else { sorted[((n - 1) as f64 * p).round() as usize] };
        Traffic {
            seconds: n,
            total_bytes: total,
            mean_bytes_per_s: if n == 0 { 0.0 } else { total as f64 / n as f64 },
            p50_bytes_per_s: q(0.5),
            p90_bytes_per_s: q(0.9),
            p99_bytes_per_s: q(0.99),
            max_bytes_per_s: sorted.last().copied().unwrap_or(0),
            active_seconds: per_second.iter().filter(|&&b| b > 0).count(),
            per_second,
        }
    }
}

/// A connection with its call metadata (custom headers and the bearer token).
struct Client {
    inner: FlightServiceClient<Channel>,
    metadata: MetadataMap,
}

fn header(k: &str, v: &str, endpoint: &str) -> Result<(AsciiMetadataKey, MetadataValue<tonic::metadata::Ascii>)> {
    let bad = |what: &str| FlightEndpointSnafu { endpoint: endpoint.to_string(), message: format!("header `{k}`: {what}") }.build();
    let key = AsciiMetadataKey::from_bytes(k.to_ascii_lowercase().as_bytes()).map_err(|_| bad("not a valid header name"))?;
    let val = v.parse().map_err(|_| bad("not a valid header value"))?;
    Ok((key, val))
}

/// Connect (and authenticate): the first connection fails `gt profile` / the daemon before anything starts.
async fn connect(o: &FlightOptions) -> Result<Client> {
    let (uri, tls) = o.uri()?;
    let bad = |m: String| FlightEndpointSnafu { endpoint: o.endpoint.clone(), message: m }.build();
    let mut ep = Endpoint::from_shared(uri.clone())
        .map_err(|e| bad(e.to_string()))?
        .connect_timeout(Duration::from_secs(10))
        .timeout(o.timeout)
        .tcp_nodelay(true)
        .http2_adaptive_window(true);
    if tls {
        let mut cfg = ClientTlsConfig::new().with_webpki_roots();
        if let Some(ca) = &o.tls_ca_file {
            let pem = std::fs::read(ca).map_err(|e| bad(format!("cannot read {}: {e}", ca.display())))?;
            cfg = cfg.ca_certificate(Certificate::from_pem(pem));
        }
        ep = ep.tls_config(cfg).map_err(|e| bad(e.to_string()))?;
    }
    let channel = ep.connect().await.context(FlightConnectSnafu { endpoint: o.endpoint.clone() })?;
    let mut metadata = MetadataMap::new();
    for (k, v) in &o.headers {
        let (k, v) = header(k, v, &o.endpoint)?;
        metadata.insert(k, v);
    }
    let token = if !o.token.is_empty() {
        Some(o.token.clone())
    } else if !o.username.is_empty() {
        let mut sql = FlightSqlServiceClient::new(channel.clone());
        for (k, v) in &o.headers {
            sql.set_header(k.to_ascii_lowercase(), v);
        }
        sql.handshake(&o.username, &o.password).await.context(FlightAuthSnafu { endpoint: o.endpoint.clone() })?;
        sql.token().cloned()
    } else {
        None
    };
    if let Some(t) = token {
        let (k, v) = header("authorization", &format!("Bearer {t}"), &o.endpoint)?;
        metadata.insert(k, v);
    }
    // Large messages: the encoder splits batches into messages of at most 2 MiB.
    let inner = FlightServiceClient::new(channel).max_decoding_message_size(64 << 20).max_encoding_message_size(64 << 20);
    Ok(Client { inner, metadata })
}

impl Client {
    /// One DoPut of `batches` under `cmd`; the record count the server reports (-1 when it does not).
    async fn put(
        &mut self,
        cmd: &CommandStatementIngest,
        batches: &[RecordBatch],
        o: &FlightOptions,
        stats: &Arc<FlightStats>,
    ) -> std::result::Result<i64, String> {
        let descriptor = FlightDescriptor::new_cmd(cmd.as_any().encode_to_vec());
        let ipc = IpcWriteOptions::default()
            .try_with_compression(match o.compression {
                Compression::None => None,
                Compression::Zstd => Some(arrow_ipc::CompressionType::ZSTD),
                Compression::Lz4 => Some(arrow_ipc::CompressionType::LZ4_FRAME),
            })
            .map_err(|e| format!("IPC options: {e}"))?;
        let input = futures::stream::iter(batches.to_vec()).map(Ok);
        let encoded = FlightDataEncoderBuilder::new()
            .with_options(ipc)
            .with_dictionary_handling(if o.dictionaries { DictionaryHandling::Resend } else { DictionaryHandling::Hydrate })
            .with_flight_descriptor(Some(descriptor))
            .build(input);
        // Count (and pace) every message as gRPC takes it; an encoding error ends the stream and fails the put.
        let failed: Arc<Mutex<Option<String>>> = Arc::default();
        let (fail, st, rate, t0) = (failed.clone(), stats.clone(), o.max_send_bytes_s, Instant::now());
        let sent = Arc::new(AtomicU64::new(0));
        let data = encoded
            .then(move |r| {
                let (fail, st, sent) = (fail.clone(), st.clone(), sent.clone());
                async move {
                    match r {
                        Ok(fd) => {
                            let n = message_bytes(&fd);
                            st.bytes.fetch_add(n, Ordering::Relaxed);
                            let total = sent.fetch_add(n, Ordering::Relaxed) + n;
                            if rate > 0
                                && let Some(wait) = Duration::from_secs_f64(total as f64 / rate as f64).checked_sub(t0.elapsed())
                            {
                                tokio::time::sleep(wait).await;
                            }
                            Some(fd)
                        }
                        Err(e) => {
                            *fail.lock().unwrap_or_else(|p| p.into_inner()) = Some(e.to_string());
                            None
                        }
                    }
                }
            })
            .take_while(|fd| futures::future::ready(fd.is_some()))
            .filter_map(futures::future::ready);
        let mut req = tonic::Request::new(data);
        *req.metadata_mut() = self.metadata.clone();
        let resp = self.inner.do_put(req).await;
        if let Some(e) = failed.lock().unwrap_or_else(|p| p.into_inner()).take() {
            return Err(format!("encoding: {e}"));
        }
        let mut stream = resp.map_err(|s| status_text(&s))?.into_inner();
        let mut count = -1;
        while let Some(r) = stream.message().await.map_err(|s| status_text(&s))? {
            if count < 0
                && let Ok(u) = DoPutUpdateResult::decode(&*r.app_metadata)
            {
                count = u.record_count;
            }
        }
        Ok(count)
    }
}

/// Bytes of one Flight message (IPC header + body + app metadata + descriptor).
fn message_bytes(fd: &FlightData) -> u64 {
    let desc = fd.flight_descriptor.as_ref().map_or(0, |d| d.cmd.len() + d.path.iter().map(String::len).sum::<usize>());
    (fd.data_header.len() + fd.data_body.len() + fd.app_metadata.len() + desc) as u64
}

fn status_text(s: &tonic::Status) -> String {
    format!("{:?}: {}", s.code(), s.message())
}

/// The outgoing schemas: the constant columns (dictionary-encoded, one value) in front of the trace / PM columns.
struct Shaper {
    trace: SchemaRef,
    pm: SchemaRef,
    columns: Vec<(String, String)>,
}

impl Shaper {
    fn new(columns: Vec<(String, String)>) -> Shaper {
        let dict = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let consts: Vec<Field> = columns.iter().map(|(k, _)| Field::new(k, dict.clone(), false)).collect();
        let with = |s: SchemaRef| -> SchemaRef {
            Arc::new(Schema::new(consts.iter().cloned().chain(s.fields().iter().map(|f| f.as_ref().clone())).collect::<Vec<_>>()))
        };
        Shaper { trace: with(trace_schema()), pm: with(crate::writer::pm_schema()), columns }
    }

    fn shape(&self, b: &RecordBatch, schema: &SchemaRef) -> Result<RecordBatch> {
        let n = b.num_rows();
        let ctx = || ArrowSnafu { context: "adding the constant columns".to_string() };
        let mut cols: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
        for (_, v) in &self.columns {
            let values: ArrayRef = Arc::new(StringArray::from(vec![v.as_str()]));
            cols.push(Arc::new(DictionaryArray::<Int32Type>::try_new(Int32Array::from(vec![0; n]), values).context(ctx())?));
        }
        cols.extend(b.columns().iter().cloned());
        RecordBatch::try_new(schema.clone(), cols).context(ctx())
    }
}

/// The `gt-flight` thread.
struct Sender {
    o: FlightOptions,
    rt: Handle,
    client: Option<Client>,
    stats: Arc<FlightStats>,
    shaper: Shaper,
    builder: TraceBatchBuilder,
    /// Trace batches of the next DoPut.
    pending: Vec<RecordBatch>,
    pending_rows: usize,
    pm: Vec<PmSample>,
    trace_cmd: CommandStatementIngest,
    pm_cmd: Option<CommandStatementIngest>,
}

fn log(msg: impl std::fmt::Display) {
    eprintln!("gt: flightsql: {msg}");
}

impl Sender {
    fn run(mut self, rx: mpsc::Receiver<Msg>) {
        for msg in rx {
            match msg {
                Msg::Rows(rows) => {
                    self.add(&rows);
                    self.stats.queue.taken(rows.len());
                    if self.pending_rows >= self.o.put_rows {
                        self.send_trace();
                    }
                }
                Msg::Pm(s) => {
                    if self.pm_cmd.is_some() {
                        self.pm.extend(s);
                    }
                }
                Msg::Names(_) => {}
                Msg::Snapshot => {
                    self.send_trace();
                    self.send_pm();
                }
                Msg::Flush(reply) => {
                    let _ = reply.send(crate::error::PersistOffSnafu.fail());
                }
                Msg::Close => break,
            }
        }
        self.send_trace();
        self.send_pm();
        self.stats.closed.store(true, Ordering::Release);
    }

    fn add(&mut self, rows: &[Activity]) {
        for a in rows {
            self.builder.push(&a.row());
            self.pending_rows += 1;
            if self.builder.len() >= self.o.batch_rows {
                self.cut();
            }
        }
    }

    /// The builder's rows -> one pending batch.
    fn cut(&mut self) {
        let n = self.builder.len();
        if n == 0 {
            return;
        }
        let b = self.builder.finish().context(ArrowSnafu { context: "building a trace batch".to_string() });
        match b.and_then(|b| self.shaper.shape(&b, &self.shaper.trace)) {
            Ok(b) => self.pending.push(b),
            Err(e) => {
                log(format_args!("{}; {n} activities dropped", crate::error::chain(&e)));
                self.pending_rows -= n;
                self.stats.queue.dropped(n as u64);
            }
        }
    }

    fn send_trace(&mut self) {
        self.cut();
        let batches = std::mem::take(&mut self.pending);
        self.pending_rows = 0;
        let rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
        if rows == 0 {
            return;
        }
        let cmd = self.trace_cmd.clone();
        if self.send(&cmd, &batches, rows) {
            self.stats.rows.fetch_add(rows, Ordering::Relaxed);
        } else {
            self.stats.queue.dropped(rows);
        }
    }

    fn send_pm(&mut self) {
        let Some(cmd) = self.pm_cmd.clone() else { return };
        let samples = std::mem::take(&mut self.pm);
        let n = samples.len() as u64;
        let batches = match pm_batch(&samples).and_then(|b| b.map(|(b, _)| self.shaper.shape(&b, &self.shaper.pm)).transpose()) {
            Ok(Some(b)) => (0..b.num_rows()).step_by(self.o.batch_rows).map(|i| b.slice(i, self.o.batch_rows.min(b.num_rows() - i))).collect::<Vec<_>>(),
            Ok(None) => return,
            Err(e) => {
                log(format_args!("{}; {n} PM samples dropped", crate::error::chain(&e)));
                self.stats.dropped_pm_samples.fetch_add(n, Ordering::Relaxed);
                return;
            }
        };
        if self.send(&cmd, &batches, n) {
            self.stats.pm_samples.fetch_add(n, Ordering::Relaxed);
        } else {
            self.stats.dropped_pm_samples.fetch_add(n, Ordering::Relaxed);
        }
    }

    /// One DoPut, retried on a new connection; false when every attempt failed.
    fn send(&mut self, cmd: &CommandStatementIngest, batches: &[RecordBatch], rows: u64) -> bool {
        let t0 = Instant::now();
        let mut ok = false;
        for attempt in 0..=self.o.retries {
            if attempt > 0 {
                self.stats.retries.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(500 << (attempt - 1).min(4)));
            }
            if self.client.is_none() {
                match self.rt.block_on(connect(&self.o)) {
                    Ok(c) => self.client = Some(c),
                    Err(e) => {
                        self.stats.errors.fetch_add(1, Ordering::Relaxed);
                        log(format_args!("{} (attempt {}/{})", crate::error::chain(&e), attempt + 1, self.o.retries + 1));
                        continue;
                    }
                }
            }
            let Some(c) = self.client.as_mut() else { continue };
            match self.rt.block_on(c.put(cmd, batches, &self.o, &self.stats)) {
                Ok(n) => {
                    self.stats.puts.fetch_add(1, Ordering::Relaxed);
                    if n >= 0 {
                        self.stats.server_rows.fetch_add(n as u64, Ordering::Relaxed);
                    }
                    ok = true;
                    break;
                }
                Err(e) => {
                    self.stats.errors.fetch_add(1, Ordering::Relaxed);
                    log(format_args!("DoPut of {rows} rows into {} failed (attempt {}/{}): {e}", cmd.table, attempt + 1, self.o.retries + 1));
                    self.client = None;
                }
            }
        }
        self.stats.put_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !ok {
            log(format_args!("giving up on {rows} rows for {}", cmd.table));
        }
        ok
    }
}

/// Connect (fails here when the server cannot be reached or refuses the credentials), then start the sender thread
/// and the per-second sampler.
pub async fn start(o: FlightOptions) -> Result<(Feed, Arc<FlightStats>, std::thread::JoinHandle<()>)> {
    let client = connect(&o).await?;
    let stats = Arc::new(FlightStats::default());
    let (tx, rx) = mpsc::channel();
    let feed = Feed::new(tx, stats.queue.clone(), o.max_buffer_bytes, !o.pm_table.is_empty());
    let rt = Handle::current();
    rt.spawn(sample(stats.clone()));
    let ingest = |table: &str| CommandStatementIngest {
        table_definition_options: Some(TableDefinitionOptions {
            if_not_exist: if o.create { TableNotExistOption::Create } else { TableNotExistOption::Fail } as i32,
            if_exists: TableExistsOption::Append as i32,
        }),
        table: table.to_string(),
        schema: (!o.db_schema.is_empty()).then(|| o.db_schema.clone()),
        catalog: (!o.catalog.is_empty()).then(|| o.catalog.clone()),
        temporary: false,
        transaction_id: None,
        options: o.ingest_options.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
    };
    let (trace_cmd, pm_cmd) = (ingest(&o.table), (!o.pm_table.is_empty()).then(|| ingest(&o.pm_table)));
    let sender = Sender {
        shaper: Shaper::new(o.columns.clone()),
        o,
        rt,
        client: Some(client),
        stats: stats.clone(),
        builder: TraceBatchBuilder::default(),
        pending: Vec::new(),
        pending_rows: 0,
        pm: Vec::new(),
        trace_cmd,
        pm_cmd,
    };
    let thread = std::thread::Builder::new()
        .name("gt-flight".into())
        .spawn(move || sender.run(rx))
        .context(SpawnThreadSnafu { name: "gt-flight" })?;
    Ok((feed, stats, thread))
}

/// Bytes sent in every second, until the sender has finished.
async fn sample(stats: Arc<FlightStats>) {
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    tick.tick().await;
    let mut last = stats.bytes.load(Ordering::Relaxed);
    loop {
        tick.tick().await;
        let closed = stats.closed.load(Ordering::Acquire);
        let now = stats.bytes.load(Ordering::Relaxed);
        {
            let mut q = stats.per_second.lock().unwrap_or_else(|e| e.into_inner());
            q.push_back(now - last);
            if q.len() > MAX_SECONDS {
                q.pop_front();
            }
        }
        last = now;
        if closed {
            return;
        }
    }
}

/// What `gt profile` reports about the stream.
#[derive(Debug, serde::Serialize)]
pub struct FlightReport {
    pub endpoint: String,
    pub table: String,
    pub pm_table: String,
    pub rows: u64,
    /// Rows the server reported ingesting (0 when it does not report them).
    pub server_rows: u64,
    pub pm_samples: u64,
    pub dropped_rows: u64,
    pub dropped_pm_samples: u64,
    pub puts: u64,
    pub errors: u64,
    pub retries: u64,
    pub bytes: u64,
    pub bytes_per_row: f64,
    pub put_seconds: f64,
    pub traffic: Traffic,
}

impl FlightStats {
    /// The report; the bytes sent after the last full second count as one more second.
    pub fn report(&self, endpoint: &str, table: &str, pm_table: &str) -> FlightReport {
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let mut seconds = self.seconds();
        let rest = l(&self.bytes).saturating_sub(seconds.iter().sum());
        if rest > 0 {
            seconds.push(rest);
        }
        let (rows, pm, bytes) = (l(&self.rows), l(&self.pm_samples), l(&self.bytes));
        FlightReport {
            endpoint: endpoint.to_string(),
            table: table.to_string(),
            pm_table: pm_table.to_string(),
            rows,
            server_rows: l(&self.server_rows),
            pm_samples: pm,
            dropped_rows: l(&self.queue.dropped_rows),
            dropped_pm_samples: l(&self.dropped_pm_samples),
            puts: l(&self.puts),
            errors: l(&self.errors),
            retries: l(&self.retries),
            bytes,
            bytes_per_row: if rows + pm == 0 { 0.0 } else { bytes as f64 / (rows + pm) as f64 },
            put_seconds: l(&self.put_ns) as f64 / 1e9,
            traffic: Traffic::of(seconds),
        }
    }
}

/// `GET /metrics` lines of the Flight SQL stream.
pub fn render(o: &mut String, pod: &str, s: &FlightStats) {
    use std::fmt::Write as _;
    let mut g = |name: &str, kind: &str, help: &str, v: String| {
        let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name}{{pod=\"{pod}\"}} {v}");
    };
    let l = |a: &AtomicU64| a.load(Ordering::Relaxed).to_string();
    g("nvtrace_flight_rows_total", "counter", "Activities sent to the Flight SQL server (successful DoPuts).", l(&s.rows));
    g(
        "nvtrace_flight_dropped_rows_total",
        "counter",
        "Activities not sent: the send buffer (flightsql.max_buffer_mib) was full, or every attempt of their DoPut failed.",
        l(&s.queue.dropped_rows),
    );
    g("nvtrace_flight_pm_samples_total", "counter", "PM samples sent to the Flight SQL server.", l(&s.pm_samples));
    g("nvtrace_flight_dropped_pm_samples_total", "counter", "PM samples whose DoPut failed.", l(&s.dropped_pm_samples));
    g("nvtrace_flight_puts_total", "counter", "Successful DoPuts.", l(&s.puts));
    g("nvtrace_flight_errors_total", "counter", "Failed connection and DoPut attempts.", l(&s.errors));
    g("nvtrace_flight_retries_total", "counter", "DoPut retries.", l(&s.retries));
    g("nvtrace_flight_bytes_total", "counter", "Arrow IPC bytes sent (rate() of it is the stream's network traffic).", l(&s.bytes));
    g("nvtrace_flight_queue_bytes", "gauge", "Activities waiting for the sender (approximate bytes).", s.queue.bytes.load(Ordering::Relaxed).to_string());
    let last = s.per_second.lock().unwrap_or_else(|e| e.into_inner()).back().copied().unwrap_or(0);
    g("nvtrace_flight_last_second_bytes", "gauge", "Bytes sent in the last full second.", last.to_string());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flight_sink::{Sink, State};
    use arrow_array::cast::AsArray;
    use gravity_trace_core::model::EventKind;

    fn act(i: i64) -> Activity {
        Activity {
            kind: EventKind::Kernel,
            name: Arc::from(format!("kernel_{}", i % 7).as_str()),
            start_ns: 1_000_000 + i * 1000,
            end_ns: 1_000_000 + i * 1000 + 500,
            pid: 42,
            ns_pid: 42,
            tid: None,
            device: Some((i % 4) as u32),
            context: None,
            stream: Some(7),
            correlation: i as u32,
            graph_id: None,
            graph_node_id: None,
            grid: None,
            block: None,
            shmem: None,
            bytes: None,
        }
    }

    fn pm(i: u64) -> PmSample {
        PmSample { gpu: [7; 16], end_utc_ns: 2_000_000 + i, metric: Arc::from("gr__cycles_active.avg.pct_of_peak_sustained_elapsed"), value: i as f64 }
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().expect("runtime")
    }

    /// A sink server on a free port; its endpoint.
    fn server(rt: &tokio::runtime::Runtime, sink: Sink) -> String {
        let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).expect("bind");
        let addr = listener.local_addr().expect("addr");
        rt.spawn(crate::flight_sink::serve(listener, sink, std::future::pending()));
        format!("grpc://{addr}")
    }

    fn options(endpoint: String) -> FlightOptions {
        let mut o = FlightOptions::from_config(&config::FlightSql { endpoint, ..Default::default() });
        o.batch_rows = 1024;
        o.put_rows = 4096;
        o.columns = vec![("host".into(), "node-a".into()), ("cluster".into(), "c1".into())];
        o
    }

    fn strings(b: &RecordBatch, col: &str) -> Vec<String> {
        let a = arrow_cast::cast(b.column_by_name(col).expect("column"), &DataType::Utf8).expect("cast");
        a.as_string::<i32>().iter().map(|v| v.unwrap_or_default().to_string()).collect()
    }

    #[test]
    fn activities_and_pm_samples_reach_the_server() {
        let rt = runtime();
        let sink = Sink::new(None, Some(("gt".into(), "secret".into())));
        let state = sink.state.clone();
        let received = sink.bytes.clone();
        let mut o = options(server(&rt, sink));
        o.username = "gt".into();
        o.password = "secret".into();
        o.headers = BTreeMap::from([("x-gt-test".to_string(), "yes".to_string())]);
        let (mut feed, stats, thread) = rt.block_on(start(o)).expect("start");
        for i in 0..10_000 {
            feed.push(act(i));
        }
        feed.push_pm(&(0..3000).map(pm).collect::<Vec<_>>());
        feed.snapshot_done();
        for i in 10_000..10_005 {
            feed.push(act(i));
        }
        feed.finish();
        thread.join().expect("sender thread");
        let st: std::sync::MutexGuard<'_, State> = state.lock().expect("state");
        let t = &st.tables["gt_trace"];
        assert_eq!((t.rows, t.puts), (10_005, 2), "one DoPut per snapshot, one at the end");
        assert_eq!(st.tables["gt_pm"].rows, 3000);
        let b = &t.batches[0];
        let names: Vec<_> = b.schema().fields().iter().take(4).map(|f| f.name().clone()).collect();
        assert_eq!(names, ["host", "cluster", "kind", "name"]);
        assert_eq!(b.schema().field_with_name("name").expect("name").data_type(), &DataType::Utf8, "dictionaries hydrated");
        assert!(strings(b, "host").iter().all(|h| h == "node-a"));
        assert!(strings(b, "cluster").iter().all(|h| h == "c1"));
        assert_eq!(strings(b, "name")[..3], ["kernel_0", "kernel_1", "kernel_2"]);
        let cmd = &st.commands[0];
        assert_eq!(cmd.table, "gt_trace");
        let tdo = cmd.table_definition_options.expect("table options");
        assert_eq!((tdo.if_not_exist, tdo.if_exists), (TableNotExistOption::Create as i32, TableExistsOption::Append as i32));
        assert!(st.headers.iter().all(|h| h.get("x-gt-test").map(String::as_str) == Some("yes")), "custom header on every call");
        drop(st);
        let r = stats.report("e", "gt_trace", "gt_pm");
        assert_eq!((r.rows, r.server_rows, r.pm_samples, r.dropped_rows, r.errors, r.puts), (10_005, 13_005, 3000, 0, 0, 3));
        assert_eq!(r.bytes, received.load(Ordering::Relaxed), "the bytes gt counts are the bytes the server receives");
        assert_eq!(r.traffic.total_bytes, r.bytes);
        assert!(r.bytes > 0);
    }

    #[test]
    fn dictionaries_and_compression_shrink_the_stream() {
        let rt = runtime();
        let mut sizes = Vec::new();
        for (dictionaries, compression) in [(false, Compression::None), (true, Compression::None), (true, Compression::Zstd), (false, Compression::Lz4)] {
            let sink = Sink::new(None, None);
            let state = sink.state.clone();
            let mut o = options(server(&rt, sink));
            o.dictionaries = dictionaries;
            o.compression = compression;
            let (mut feed, stats, thread) = rt.block_on(start(o)).expect("start");
            for i in 0..20_000 {
                feed.push(act(i));
            }
            feed.finish();
            thread.join().expect("sender thread");
            let st = state.lock().expect("state");
            let t = &st.tables["gt_trace"];
            assert_eq!(t.rows, 20_000, "{dictionaries} {compression:?}");
            let name_type = t.batches[0].schema().field_with_name("name").expect("name").data_type().clone();
            assert_eq!(matches!(name_type, DataType::Dictionary(..)), dictionaries);
            sizes.push(stats.bytes.load(Ordering::Relaxed));
        }
        assert!(sizes[1] < sizes[0], "dictionaries: {sizes:?}");
        assert!(sizes[2] < sizes[1], "zstd: {sizes:?}");
        assert!(sizes[3] < sizes[0], "lz4: {sizes:?}");
    }

    #[test]
    fn failed_puts_are_retried_then_dropped() {
        let rt = runtime();
        let sink = Sink::new(None, None);
        let state = sink.state.clone();
        let mut o = options(server(&rt, sink));
        o.retries = 2;
        state.lock().expect("state").fail_puts = 2;
        let (mut feed, stats, thread) = rt.block_on(start(o)).expect("start");
        for i in 0..100 {
            feed.push(act(i));
        }
        feed.snapshot_done();
        // the next DoPut fails every attempt
        while stats.puts.load(Ordering::Relaxed) == 0 {
            std::thread::sleep(Duration::from_millis(10));
        }
        state.lock().expect("state").fail_puts = 3;
        for i in 0..50 {
            feed.push(act(i));
        }
        feed.finish();
        thread.join().expect("sender thread");
        let r = stats.report("e", "gt_trace", "");
        assert_eq!((r.rows, r.dropped_rows, r.retries, r.errors, r.puts), (100, 50, 4, 5, 1));
        assert_eq!(state.lock().expect("state").tables["gt_trace"].rows, 100);
    }

    #[test]
    fn endpoints() {
        let uri = |e: &str| options(e.to_string()).uri().map_err(|e| e.to_string());
        assert_eq!(uri("grpc://h:1"), Ok(("http://h:1".into(), false)));
        assert_eq!(uri("grpc+tls://h:1/"), Ok(("https://h:1".into(), true)));
        assert_eq!(uri("h:1"), Ok(("http://h:1".into(), false)));
        assert_eq!(uri("https://h:1"), Ok(("https://h:1".into(), true)));
        assert!(uri("s3://b").is_err());
        let rt = runtime();
        let e = rt.block_on(start(options("grpc://127.0.0.1:1".into())));
        assert!(matches!(e, Err(crate::error::Error::FlightConnect { .. })), "nothing listening: fails at start");
    }

    /// Throughput per encoding: encoding alone, and through gRPC into the in-memory sink.
    /// `cargo test --release --features monitor flight::tests::throughput -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn throughput() {
        let rt = runtime();
        let long = |i: i64| -> Activity {
            let mut a = act(i);
            a.name = Arc::from(format!("void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul>>_{}", i % 300).as_str());
            a.graph_id = Some(3);
            a.graph_node_id = Some(i as u64);
            a
        };
        let n = 1_000_000;
        for (dictionaries, compression) in [(false, Compression::None), (true, Compression::None), (true, Compression::Zstd), (true, Compression::Lz4), (false, Compression::Zstd)] {
            // encoding alone
            let shaper = Shaper::new(vec![("host".into(), "node-a".into())]);
            let mut builder = TraceBatchBuilder::default();
            let mut batches = Vec::new();
            for i in 0..n {
                builder.push(&long(i).row());
                if builder.len() == 65536 {
                    batches.push(shaper.shape(&builder.finish().expect("batch"), &shaper.trace).expect("shape"));
                }
            }
            let ipc = IpcWriteOptions::default()
                .try_with_compression(match compression {
                    Compression::None => None,
                    Compression::Zstd => Some(arrow_ipc::CompressionType::ZSTD),
                    Compression::Lz4 => Some(arrow_ipc::CompressionType::LZ4_FRAME),
                })
                .expect("ipc");
            let t0 = Instant::now();
            let enc = FlightDataEncoderBuilder::new()
                .with_options(ipc)
                .with_dictionary_handling(if dictionaries { DictionaryHandling::Resend } else { DictionaryHandling::Hydrate })
                .build(futures::stream::iter(batches.clone()).map(Ok));
            let msgs: Vec<FlightData> = rt.block_on(enc.map(|r| r.expect("encode")).collect());
            let encode_s = t0.elapsed().as_secs_f64();
            let bytes: u64 = msgs.iter().map(message_bytes).sum();
            // decoding alone
            let t0 = Instant::now();
            let dec = arrow_flight::decode::FlightRecordBatchStream::new_from_flight_data(futures::stream::iter(msgs.clone()).map(Ok));
            let rows: usize = rt.block_on(dec.map(|b| b.expect("decode").num_rows()).collect::<Vec<_>>()).iter().sum();
            let decode_s = t0.elapsed().as_secs_f64();
            // end to end (GT_SINK_PARQUET=1: the sink writes Parquet, like examples/flightsql_sink)
            let dir = tempfile::tempdir().expect("tempdir");
            let sink = Sink::new(std::env::var_os("GT_SINK_PARQUET").map(|_| dir.path().to_path_buf()), None);
            let mut o = options(server(&rt, sink));
            o.dictionaries = dictionaries;
            o.compression = compression;
            o.columns = vec![("host".into(), "node-a".into())];
            o.batch_rows = 65536;
            o.put_rows = 1 << 20;
            let (mut feed, stats, thread) = rt.block_on(start(o)).expect("start");
            let t0 = Instant::now();
            for i in 0..n {
                feed.push(long(i));
            }
            feed.finish();
            thread.join().expect("sender");
            let total_s = t0.elapsed().as_secs_f64();
            println!(
                "dictionaries {dictionaries:5} {compression:5?}: {:6.1} B/row, {} messages; encode {encode_s:.2} s, decode {decode_s:.2} s ({rows} rows); \
                 end to end {total_s:.2} s ({:.0} k rows/s, put {:.2} s)",
                bytes as f64 / n as f64,
                msgs.len(),
                n as f64 / total_s / 1e3,
                stats.put_ns.load(Ordering::Relaxed) as f64 / 1e9
            );
        }
    }

    /// Real activities (Parquet files a sink wrote, `GT_REPLAY_DIR`) through each encoding: where the time goes.
    /// `GT_REPLAY_DIR=/tmp/gt-bench/flightsql-check/plain/sink/gt_trace cargo test --release --features monitor flight::tests::replay -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn replay() {
        let Some(dir) = std::env::var_os("GT_REPLAY_DIR") else { return };
        let mut files: Vec<_> = std::fs::read_dir(dir).expect("dir").flatten().map(|e| e.path()).collect();
        files.sort();
        let dict = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let mut batches = Vec::new();
        for f in files.iter().take(3) {
            let r = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(f).expect("open"))
                .expect("parquet")
                .with_batch_size(65536)
                .build()
                .expect("reader");
            for b in r {
                let b = b.expect("batch");
                let fields: Vec<Field> = b.schema().fields().iter().map(|f| match f.data_type() {
                    DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Field::new(f.name(), dict.clone(), f.is_nullable()),
                    _ => f.as_ref().clone(),
                }).collect();
                let cols: Vec<ArrayRef> = b.columns().iter().zip(&fields).map(|(c, f)| arrow_cast::cast(c, f.data_type()).expect("cast")).collect();
                batches.push(RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).expect("batch"));
            }
        }
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        let distinct = arrow_array::cast::AsArray::as_dictionary::<Int32Type>(batches[0].column_by_name("name").expect("name")).values().len();
        println!("{rows} real rows in {} batches ({distinct} names in the first)", batches.len());
        let rt = runtime();
        for (dictionaries, compression) in [(false, Compression::None), (true, Compression::None), (true, Compression::Zstd)] {
            let ipc = IpcWriteOptions::default()
                .try_with_compression((compression == Compression::Zstd).then_some(arrow_ipc::CompressionType::ZSTD))
                .expect("ipc");
            let t0 = Instant::now();
            let enc = FlightDataEncoderBuilder::new()
                .with_options(ipc)
                .with_dictionary_handling(if dictionaries { DictionaryHandling::Resend } else { DictionaryHandling::Hydrate })
                .build(futures::stream::iter(batches.clone()).map(Ok));
            let msgs: Vec<FlightData> = rt.block_on(enc.map(|r| r.expect("encode")).collect());
            let encode_s = t0.elapsed().as_secs_f64();
            let bytes: u64 = msgs.iter().map(message_bytes).sum();
            let t0 = Instant::now();
            let dec = arrow_flight::decode::FlightRecordBatchStream::new_from_flight_data(futures::stream::iter(msgs.clone()).map(Ok));
            let decoded: Vec<RecordBatch> = rt.block_on(dec.map(|b| b.expect("decode")).collect());
            let decode_s = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            let tmp = tempfile::tempdir().expect("tempdir");
            let mut w = parquet::arrow::ArrowWriter::try_new(std::fs::File::create(tmp.path().join("x.parquet")).expect("create"), decoded[0].schema(), None).expect("writer");
            for b in &decoded {
                w.write(b).expect("write");
            }
            w.close().expect("close");
            let parquet_s = t0.elapsed().as_secs_f64();
            // through gRPC into a sink that writes Parquet, one DoPut per 1M rows
            let sink = Sink::new(Some(tmp.path().join("sink")), None);
            let mut o = options(server(&rt, sink));
            o.dictionaries = dictionaries;
            o.compression = compression;
            let stats = Arc::new(FlightStats::default());
            let mut c = rt.block_on(connect(&o)).expect("connect");
            let cmd = CommandStatementIngest { table: "t".into(), ..Default::default() };
            let t0 = Instant::now();
            for chunk in batches.chunks(16) {
                rt.block_on(c.put(&cmd, chunk, &o, &stats)).expect("put");
            }
            let put_s = t0.elapsed().as_secs_f64();
            println!(
                "dictionaries {dictionaries:5} {compression:5?}: {:6.1} B/row, {} messages | encode {encode_s:.2} s, decode {decode_s:.2} s, \
                 parquet {parquet_s:.2} s | DoPuts {put_s:.2} s = {:.0} k rows/s, {:.0} MiB/s",
                bytes as f64 / rows as f64,
                msgs.len(),
                rows as f64 / put_s / 1e3,
                bytes as f64 / put_s / (1 << 20) as f64
            );
        }
    }

    /// Bytes per row of the same real activities: Arrow IPC as the stream sends it (batch size, message size,
    /// sorting, compression) against gt's trace Parquet (sorted 1M-row groups, delta-encoded times, zstd 3).
    /// `GT_REPLAY_DIR=<sink>/gt_trace GT_REPLAY_FILES=10 cargo test --release --features monitor flight::tests::encodings -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn encodings() {
        use arrow_ipc::CompressionType as C;
        use gravity_trace_core::parquet_io::{TableWriter, WriteOptions, conform};
        let Some(dir) = std::env::var_os("GT_REPLAY_DIR") else { return };
        let n_files: usize = std::env::var("GT_REPLAY_FILES").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
        let mut files: Vec<_> = std::fs::read_dir(dir).expect("dir").flatten().map(|e| e.path()).collect();
        files.sort();
        let schema = trace_schema();
        let mut batches = Vec::new();
        for f in files.iter().take(n_files) {
            let r = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(f).expect("open"))
                .expect("parquet")
                .with_batch_size(65536)
                .build()
                .expect("reader");
            for b in r {
                batches.push(conform(&b.expect("batch"), &schema, f).expect("conform"));
            }
        }
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        println!("{rows} activities from {} files", n_files.min(files.len()));
        let sort = |b: &RecordBatch| -> RecordBatch {
            let keys: Vec<arrow_ord::sort::SortColumn> = ["pid", "device_id", "stream_id", "start_ns"]
                .iter()
                .map(|c| arrow_ord::sort::SortColumn { values: b.column_by_name(c).expect("key").clone(), options: None })
                .collect();
            let idx = arrow_ord::sort::lexsort_to_indices(&keys, None).expect("sort");
            arrow_select::take::take_record_batch(b, &idx).expect("take")
        };
        let rt = runtime();
        let report = |name: &str, bytes: u64, secs: f64| {
            println!(
                "{name:58} {:6.2} B/row {:8.1} MB  {secs:6.2} s ({:5.1} M rows/s on one core)",
                bytes as f64 / rows as f64,
                bytes as f64 / 1e6,
                rows as f64 / secs / 1e6
            );
        };
        let ipc = |name: &str, batch_rows: usize, max_msg: usize, sorted: bool, dictionaries: bool, c: Option<arrow_ipc::CompressionType>| {
            let groups: Vec<RecordBatch> =
                batches.chunks((batch_rows / 65536).max(1)).map(|g| arrow_select::concat::concat_batches(&schema, g).expect("concat")).collect();
            let t0 = Instant::now();
            let groups: Vec<RecordBatch> = if sorted { groups.iter().map(sort).collect() } else { groups };
            let enc = FlightDataEncoderBuilder::new()
                .with_options(IpcWriteOptions::default().try_with_compression(c).expect("ipc"))
                .with_dictionary_handling(if dictionaries { DictionaryHandling::Resend } else { DictionaryHandling::Hydrate })
                .with_max_flight_data_size(max_msg)
                .build(futures::stream::iter(groups).map(Ok));
            let bytes = rt.block_on(enc.fold(0u64, |n, r| async move { n + message_bytes(&r.expect("encode")) }));
            report(name, bytes, t0.elapsed().as_secs_f64());
        };
        let pq = |name: &str, opts: WriteOptions| {
            let t0 = Instant::now();
            let mut w = TableWriter::new(Vec::new(), PathBuf::from("memory"), schema.clone(), &opts).expect("writer");
            for b in &batches {
                w.write(b).expect("write");
            }
            let bytes = w.finish().expect("finish").len() as u64;
            report(name, bytes, t0.elapsed().as_secs_f64());
        };
        let (k64, m2, m64) = (65536, 2 << 20, 64 << 20);
        ipc("IPC strings, uncompressed (default)", k64, m2, false, false, None);
        ipc("IPC dictionaries, uncompressed", k64, m2, false, true, None);
        ipc("IPC dictionaries + lz4", k64, m2, false, true, Some(C::LZ4_FRAME));
        ipc("IPC dictionaries + zstd (what was measured)", k64, m2, false, true, Some(C::ZSTD));
        ipc("IPC dictionaries + zstd, 64K batches unsplit", k64, m64, false, true, Some(C::ZSTD));
        ipc("IPC dictionaries + zstd, 1M-row batches", 1 << 20, m64, false, true, Some(C::ZSTD));
        ipc("IPC dictionaries + zstd, 64K batches sorted", k64, m64, true, true, Some(C::ZSTD));
        ipc("IPC dictionaries + zstd, 1M-row batches sorted", 1 << 20, m64, true, true, Some(C::ZSTD));
        pq("Parquet, unsorted, dictionaries, zstd 3", WriteOptions::default());
        pq("Parquet, gt trace layout (sorted 1M groups, delta, zstd 3)", WriteOptions::trace());
        pq("Parquet, gt trace layout with 64K-row groups", WriteOptions { row_group_rows: k64, ..WriteOptions::trace() });
    }

    #[test]
    fn traffic_percentiles() {
        let t = Traffic::of(vec![0, 0, 10, 0, 30]);
        assert_eq!((t.seconds, t.total_bytes, t.max_bytes_per_s, t.p50_bytes_per_s, t.active_seconds), (5, 40, 30, 0, 2));
        assert!((t.mean_bytes_per_s - 8.0).abs() < 1e-9);
    }
}
