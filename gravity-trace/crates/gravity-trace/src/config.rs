//! `config.toml` of `gt export-daemon` / `gt export` / `gt profile` (and the defaults of `gt dump`). Every key is
//! optional; see `config.example.toml` at the repository root for the documented defaults.
//!
//! Looked up at `--config PATH` / `$GT_CONFIG` (must exist), else `/etc/gravity-trace/config.toml` (built-in defaults
//! when that does not exist either). Unknown keys are errors, so a typo does not silently fall back to a default.
//!
//! Environment variables override the file: `GT_<SECTION>_<KEY>` sets `<key>` of `[<section>]`, e.g.
//! `GT_PERSIST_DIR=s3://bucket/traces`, `GT_LIMITS_MAX_CPUS=4`, `GT_FLIGHTSQL_PASSWORD=...`. Precedence: built-in
//! defaults < the file < the environment (< command-line flags of `gt export` / `gt profile`). The value is the
//! plain text of the key's type: strings as they are, numbers, booleans (`true`/`false`, `1`/`0`, `yes`/`no`,
//! `on`/`off`), lists as `a,b` or a TOML array, tables as `k=v,k2=v2` or a TOML inline table; an empty
//! `GT_COLLECT_DIAG_FILE` unsets it. A `GT_<SECTION>_...` variable that names no key is an error, like a typo in the
//! file.
use crate::error::{BadConfigSnafu, BadEnvSnafu, ParseConfigSnafu, ReadConfigSnafu, Result};
use serde::{Deserialize, Serialize};
use snafu::ResultExt;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const DEFAULT_PATH: &str = "/etc/gravity-trace/config.toml";

#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub daemon: Daemon,
    pub collect: Collect,
    pub persist: Persist,
    pub flightsql: FlightSql,
    pub limits: Limits,
}

/// The background process.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Daemon {
    /// HTTP address of `GET /metrics`, `POST /capture`, `POST /persist/flush`; "" = no HTTP server.
    pub listen: String,
    /// Holds the pid of the running daemon (and its lock: one daemon per pid file).
    pub pid_file: PathBuf,
    /// stdout / stderr of the background process (appended).
    pub log_file: PathBuf,
    /// Where `POST /capture` writes its files: a directory or an object store URL.
    pub captures_dir: String,
    /// How long `gt export-daemon stop` waits for the daemon to finish its files and exit.
    pub stop_timeout_s: u64,
}

impl Default for Daemon {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:9465".into(),
            pid_file: "/run/gravity-trace/gt.pid".into(),
            log_file: "/var/log/gravity-trace/gt.log".into(),
            captures_dir: "/var/lib/gravity-trace/captures".into(),
            stop_timeout_s: 60,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Storage {
    Memory,
    Disk,
}

/// What is collected and how the SDK takes snapshots (the `gt exporter` flags of the same names).
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Collect {
    /// How long to collect, in seconds; the daemon finishes its files and exits afterwards. 0 = until stopped.
    pub duration_s: u64,
    /// Snapshot cadence (the SDK recommends >= 5000 ms). Also the delay before activities reach the trace files.
    pub snapshot_ms: u64,
    /// Where the SDK keeps snapshot data blocks until they are exported: "memory" or "disk" (files in `work_dir`).
    pub storage: Storage,
    pub work_dir: PathBuf,
    /// Provisioned trace data rate (MiB/s) that sizes the snapshot storage.
    pub data_rate_mib: u64,
    /// Upper bound of the per-snapshot attachment data (module / function / CUDA graph definitions).
    pub max_attach_mib: u64,
    /// Parallel export shards, each owning whole GPUs; 0 = one per GPU (at most 8).
    pub shards: u32,
    /// Providers not to decode: "gpu_pm", "gpu_hwt", "cuda_api", "nvtx".
    pub ignore: Vec<String>,
    /// Kernels per process and kind exported individually in /metrics (the rest folded into `<other>`).
    pub top: usize,
    /// Append one JSON line of diagnostics per snapshot to this file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diag_file: Option<PathBuf>,
    /// The procfs of the host's PID namespace, for process titles (vLLM's `VLLM::Worker_TP0`) by host pid.
    pub host_proc: PathBuf,
}

impl Default for Collect {
    fn default() -> Self {
        Self {
            duration_s: 0,
            snapshot_ms: 5000,
            storage: Storage::Memory,
            work_dir: "/var/lib/gravity-trace/live".into(),
            data_rate_mib: 128,
            max_attach_mib: 1024,
            shards: 0,
            ignore: Vec::new(),
            top: 50,
            diag_file: None,
            host_proc: "/proc".into(),
        }
    }
}

/// What the persisted trace is written as.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PersistFormat {
    /// `<start>-<end>.parquet` trace + `<start>-<end>.pm.parquet` PM samples (what `gt dump` reads).
    #[default]
    Parquet,
    /// `<start>-<end>.json.gz`: Perfetto / Chrome trace JSON (activities and PM counters), for ui.perfetto.dev.
    Perfetto,
}

impl PersistFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            PersistFormat::Parquet => "parquet",
            PersistFormat::Perfetto => "perfetto",
        }
    }
}

/// The trace written to disk continuously (read back by `gt dump`).
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Persist {
    /// Write every activity to `dir`. Off: only /metrics and on-demand captures.
    pub enabled: bool,
    /// Parquet (`gt dump` reads it) or Perfetto JSON, one or the other.
    pub format: PersistFormat,
    /// `<start>-<end>.parquet` trace files and `<start>-<end>.pm.parquet` PM samples, or `<start>-<end>.json.gz`
    /// Perfetto files (UTC ns): a directory or an object store URL (`s3://bucket/prefix` with the `s3` build feature).
    pub dir: String,
    /// Start a new file every this many seconds (at the next snapshot boundary).
    pub rotate_s: u64,
    /// Activities queued between the merge thread and the disk writer, at most (MiB, 168 bytes per activity: it
    /// must hold one snapshot).
    /// Beyond it whole chunks are dropped and counted (nvtrace_persist_dropped_rows_total) rather than slowing the
    /// snapshots down. Separately, the writer holds one row group (`row_group_rows`) in memory to sort it.
    pub max_buffer_mib: u64,
    /// Rows per Parquet row group (sorted by pid, GPU, stream, start time inside the group).
    pub row_group_rows: usize,
    /// zstd level of the Parquet pages (1-22).
    pub zstd_level: i32,
    /// gzip level of the Perfetto files (1-9). They are compressed on one thread while the trace is collected:
    /// higher levels are smaller but must keep up with the trace (level 1 ~660 MB/s of JSON on one CPU core of a node, level 3
    /// ~330 MB/s; DeepSeek-V4 TP 4 under load makes ~400 MB/s).
    pub gzip_level: u32,
    /// Delete the oldest files while `dir` holds more than this (GiB); 0 = never delete.
    pub max_disk_gib: u64,
    /// Also write PM metric samples (parquet: `.pm.parquet` files; perfetto: counter tracks in the same file).
    pub pm: bool,
}

impl Default for Persist {
    fn default() -> Self {
        Self {
            enabled: true,
            format: PersistFormat::Parquet,
            dir: "/var/lib/gravity-trace/traces".into(),
            rotate_s: 60,
            max_buffer_mib: 1024,
            row_group_rows: 1024 * 1024,
            zstd_level: 3,
            gzip_level: 1,
            max_disk_gib: 64,
            pm: true,
        }
    }
}

/// Compression of the Arrow IPC bodies of the Flight SQL stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Compression {
    None,
    Zstd,
    Lz4,
}

/// The providers `[collect] ignore` may name (the SDK tools' `--ignore` list).
pub const PROVIDERS: [&str; 4] = ["gpu_pm", "gpu_hwt", "cuda_api", "nvtx"];

/// Every activity (and PM sample) streamed to an Arrow Flight SQL server while collecting (bulk ingestion). On:
/// nothing is persisted to disk (`persist` is ignored).
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlightSql {
    pub enabled: bool,
    /// `grpc://host:port` or `grpc+tls://host:port` (also `http://`, `https://`).
    pub endpoint: String,
    /// Target table of the activities; of the PM samples ("" = not sent).
    pub table: String,
    pub pm_table: String,
    /// "" = the server's default.
    pub catalog: String,
    pub db_schema: String,
    /// Let the server create missing tables (else ingesting into a missing table fails); rows are appended.
    pub create: bool,
    /// Flight handshake (basic auth, the server answers with a bearer token), or a bearer token. Keep the secrets
    /// out of the file with `$GT_FLIGHTSQL_PASSWORD` / `$GT_FLIGHTSQL_TOKEN`.
    pub username: String,
    pub password: String,
    pub token: String,
    /// PEM CA certificate for `grpc+tls` ("" = the built-in web PKI roots).
    pub tls_ca_file: String,
    /// Rows per Arrow record batch.
    pub batch_rows: usize,
    /// At most this many activities per DoPut (one DoPut per snapshot otherwise).
    pub put_rows: usize,
    /// Activities queued for the sender, at most (MiB, 168 bytes per activity: it must hold one snapshot); beyond
    /// it chunks are dropped and counted (nvtrace_flight_dropped_rows_total).
    pub max_buffer_mib: u64,
    /// Cap of the sending rate (MiB/s); 0 = as fast as the network takes it.
    pub max_send_mib_s: u64,
    /// Send dictionary-encoded columns (kind, name, ...) as dictionaries (smaller; the server must accept them);
    /// false = plain strings.
    pub dictionaries: bool,
    /// Arrow IPC body compression: "none", "zstd", "lz4" (the server must support it).
    pub compression: Compression,
    /// Per-request timeout (seconds); a failed DoPut is retried this many times on a new connection, then its rows
    /// are dropped and counted.
    pub timeout_s: u64,
    pub retries: u32,
    /// gRPC headers on every call.
    pub headers: BTreeMap<String, String>,
    /// Server-specific ingestion options (`CommandStatementIngest.options`).
    pub options: BTreeMap<String, String>,
    /// Constant columns added to every row besides `host` (this host's name unless set here).
    pub columns: BTreeMap<String, String>,
}

impl Default for FlightSql {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: String::new(),
            table: "gt_trace".into(),
            pm_table: "gt_pm".into(),
            catalog: String::new(),
            db_schema: String::new(),
            create: true,
            username: String::new(),
            password: String::new(),
            token: String::new(),
            tls_ca_file: String::new(),
            batch_rows: 65536,
            put_rows: 1 << 20,
            max_buffer_mib: 1024,
            max_send_mib_s: 0,
            dictionaries: false,
            compression: Compression::None,
            timeout_s: 60,
            retries: 3,
            headers: BTreeMap::new(),
            options: BTreeMap::new(),
            columns: BTreeMap::new(),
        }
    }
}

/// CPU use of the whole process (SDK export threads, merge, disk writer, HTTP), set before any thread starts.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    /// At most this many CPU cores (CPU affinity of every thread: the last N CPUs of `cpu_list`, or of the CPUs
    /// the process may use). 0 = no limit.
    pub max_cpus: u32,
    /// CPUs to run on, e.g. "64-71" or "0,2,4-7"; "" = every CPU the process may use.
    pub cpu_list: String,
    /// Scheduling priority (nice, -20..19; below 0 needs CAP_SYS_NICE).
    pub nice: i32,
}

/// A configuration, the file it came from (`None`: built-in defaults) and the environment variables that overrode it.
pub struct Loaded {
    pub config: Config,
    pub path: Option<PathBuf>,
    pub env: Vec<String>,
}

impl Loaded {
    /// `/etc/gravity-trace/config.toml + GT_PERSIST_DIR, GT_LIMITS_MAX_CPUS` (for logs).
    pub fn source(&self) -> String {
        let file = self.path.as_ref().map_or("built-in defaults".to_string(), |p| p.display().to_string());
        if self.env.is_empty() { file } else { format!("{file} + {}", self.env.join(", ")) }
    }
}

/// `explicit` (`--config` / `$GT_CONFIG`) must exist; without it the default path is used when it exists. Then the
/// `GT_<SECTION>_<KEY>` environment variables.
pub fn load(explicit: Option<&Path>) -> Result<Loaded> {
    let env = std::env::vars_os().filter_map(|(k, v)| Some((k.into_string().ok()?, v.to_string_lossy().into_owned())));
    load_with_env(explicit, env)
}

fn load_with_env(explicit: Option<&Path>, env: impl IntoIterator<Item = (String, String)>) -> Result<Loaded> {
    let path = match explicit {
        Some(p) => Some(p.to_path_buf()),
        None => Some(PathBuf::from(DEFAULT_PATH)).filter(|p| p.exists()),
    };
    let path = path.map(|p| std::path::absolute(&p).unwrap_or(p));
    let text = match &path {
        Some(p) => std::fs::read_to_string(p).context(ReadConfigSnafu { path: p.clone() })?,
        None => String::new(),
    };
    let shown = path.clone().unwrap_or_else(|| PathBuf::from("built-in defaults"));
    let (config, env) = parse_with_env(&text, env).map_err(|e| match e {
        ConfigError::Toml(source) => ParseConfigSnafu { path: shown, message: source.to_string() }.build(),
        ConfigError::Invalid(message) => BadConfigSnafu { path: shown, message }.build(),
        ConfigError::Env { var, message } => BadEnvSnafu { var, message }.build(),
    })?;
    Ok(Loaded { config, path, env })
}

#[derive(Debug)]
pub enum ConfigError {
    Toml(toml::de::Error),
    Invalid(String),
    /// An environment variable that names no key, or whose value does not fit the key.
    Env { var: String, message: String },
}

/// The file alone (no environment).
#[cfg(test)]
pub fn parse(text: &str) -> std::result::Result<Config, ConfigError> {
    parse_with_env(text, []).map(|(c, _)| c)
}

/// Sections of the configuration: `GT_<SECTION>_<KEY>`.
const SECTIONS: [&str; 5] = ["daemon", "collect", "persist", "flightsql", "limits"];
/// Keys without a value by default (`None`), all of them strings when set; an empty variable unsets them.
const OPTIONAL: [(&str, &str); 1] = [("collect", "diag_file")];

/// `GT_PERSIST_MAX_DISK_GIB` -> ("persist", "max_disk_gib"); `None` for other variables (`GT_CONFIG`, `PATH`, ...).
fn env_key(var: &str) -> Option<(&'static str, String)> {
    let rest = var.strip_prefix("GT_")?;
    SECTIONS.iter().find_map(|s| {
        let key = rest.strip_prefix(s.to_ascii_uppercase().as_str())?.strip_prefix('_')?;
        Some((*s, key.to_ascii_lowercase()))
    })
}

/// The file, then the `GT_<SECTION>_<KEY>` variables among `env` on top of it; with the variables applied (sorted).
pub fn parse_with_env(
    text: &str,
    env: impl IntoIterator<Item = (String, String)>,
) -> std::result::Result<(Config, Vec<String>), ConfigError> {
    // The file on its own first: its errors point into the file.
    let file: Config = toml::from_str(text).map_err(ConfigError::Toml)?;
    let mut vars: Vec<(String, &'static str, String, String)> =
        env.into_iter().filter_map(|(var, v)| env_key(&var).map(|(s, k)| (var, s, k, v))).collect();
    if vars.is_empty() {
        file.validate().map_err(ConfigError::Invalid)?;
        return Ok((file, Vec::new()));
    }
    vars.sort();
    let defaults = toml::Value::try_from(Config::default()).map_err(|e| ConfigError::Invalid(format!("cannot render the defaults: {e}")))?;
    let mut table = toml::Value::Table(toml::from_str(text).map_err(ConfigError::Toml)?);
    let mut applied = Vec::new();
    for (var, section, key, raw) in vars {
        let env_err = |message: String| ConfigError::Env { var: var.clone(), message };
        let optional = OPTIONAL.contains(&(section, key.as_str()));
        let like = match defaults.get(section).and_then(|s| s.get(&key)) {
            Some(v) => v.clone(),
            None if optional => toml::Value::String(String::new()),
            None => return Err(env_err(format!("[{section}] has no key `{key}`"))),
        };
        let value = env_value(&raw, &like, optional).map_err(|m| env_err(format!("{section}.{key}: {m}")))?;
        let Some(t) = table.as_table_mut() else { return Err(env_err("the configuration is not a table".into())) };
        let sec = t.entry(section).or_insert_with(|| toml::Value::Table(toml::Table::new()));
        let Some(sec) = sec.as_table_mut() else { return Err(env_err(format!("[{section}] is not a table"))) };
        match value {
            Some(v) => sec.insert(key.clone(), v),
            None => sec.remove(&key),
        };
        // Each variable on its own, so that a bad value (an unknown enum variant, a negative count) names it.
        table.clone().try_into::<Config>().map_err(|e| env_err(format!("{section}.{key}: {}", e.message())))?;
        applied.push(var);
    }
    let c: Config = table.try_into().map_err(ConfigError::Toml)?;
    c.validate().map_err(|m| ConfigError::Invalid(format!("{m} (with {} from the environment)", applied.join(", "))))?;
    Ok((c, applied))
}

/// The text of an environment variable as a value of the same TOML type as `like` (the key's default); `None`:
/// unset an optional key.
fn env_value(raw: &str, like: &toml::Value, optional: bool) -> std::result::Result<Option<toml::Value>, String> {
    use toml::Value as V;
    let toml_value = |s: &str| -> std::result::Result<V, String> {
        let mut t: toml::Table = toml::from_str(&format!("v = {s}")).map_err(|e| format!("`{s}` is not a TOML value: {}", e.message()))?;
        t.remove("v").ok_or_else(|| format!("`{s}` is not a TOML value"))
    };
    let t = raw.trim();
    Ok(Some(match like {
        V::String(_) if optional && raw.is_empty() => return Ok(None),
        V::String(_) => V::String(raw.to_string()),
        V::Integer(_) => V::Integer(t.parse().map_err(|_| format!("`{raw}` is not an integer"))?),
        V::Float(_) => V::Float(t.parse().map_err(|_| format!("`{raw}` is not a number"))?),
        V::Boolean(_) => V::Boolean(match t.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => true,
            "false" | "0" | "no" | "off" => false,
            _ => return Err(format!("`{raw}` is not a boolean (true / false)")),
        }),
        V::Array(_) if t.starts_with('[') => toml_value(t)?,
        V::Array(_) => V::Array(t.split(',').map(str::trim).filter(|x| !x.is_empty()).map(|x| V::String(x.to_string())).collect()),
        V::Table(_) if t.starts_with('{') => toml_value(t)?,
        V::Table(_) => {
            let mut m = toml::Table::new();
            for kv in t.split(',').map(str::trim).filter(|x| !x.is_empty()) {
                let (k, v) = kv.split_once('=').ok_or_else(|| format!("`{kv}` is not key=value"))?;
                m.insert(k.trim().to_string(), V::String(v.trim().to_string()));
            }
            V::Table(m)
        }
        V::Datetime(_) => toml_value(t)?,
    }))
}

impl Config {
    fn validate(&self) -> std::result::Result<(), String> {
        let (c, p, l) = (&self.collect, &self.persist, &self.limits);
        let check = |ok: bool, msg: &str| if ok { Ok(()) } else { Err(msg.to_string()) };
        check(c.snapshot_ms >= 500, "collect.snapshot_ms must be at least 500")?;
        check(c.data_rate_mib > 0, "collect.data_rate_mib must be positive")?;
        let known = c.ignore.join(",").split(',').map(str::trim).filter(|p| !p.is_empty()).all(|p| PROVIDERS.contains(&p));
        check(known, "collect.ignore: use gpu_pm, gpu_hwt, cuda_api, nvtx")?;
        check(p.rotate_s >= 1, "persist.rotate_s must be at least 1")?;
        check(p.max_buffer_mib >= 1, "persist.max_buffer_mib must be at least 1")?;
        check(p.row_group_rows >= 1024, "persist.row_group_rows must be at least 1024")?;
        check((1..=22).contains(&p.zstd_level), "persist.zstd_level must be 1-22")?;
        check((1..=9).contains(&p.gzip_level), "persist.gzip_level must be 1-9")?;
        check((-20..=19).contains(&l.nice), "limits.nice must be -20..19")?;
        if !l.cpu_list.is_empty() {
            parse_cpu_list(&l.cpu_list).map_err(|e| format!("limits.cpu_list: {e}"))?;
        }
        if self.flightsql.enabled {
            self.validate_flightsql()?;
        }
        Ok(())
    }

    /// The `[flightsql]` keys (checked when it is enabled, also after `gt profile --flightsql`).
    pub fn validate_flightsql(&self) -> std::result::Result<(), String> {
        let f = &self.flightsql;
        let check = |ok: bool, msg: &str| if ok { Ok(()) } else { Err(msg.to_string()) };
        check(!f.endpoint.trim().is_empty(), "flightsql.endpoint is required (grpc://host:port)")?;
        #[cfg(feature = "monitor")]
        crate::flight::FlightOptions::from_config(f).uri().map_err(|e| format!("flightsql.endpoint: {e}"))?;
        check(!f.table.is_empty(), "flightsql.table must not be empty")?;
        check(f.pm_table != f.table, "flightsql.pm_table must differ from flightsql.table")?;
        check(f.batch_rows >= 1024, "flightsql.batch_rows must be at least 1024")?;
        check(f.put_rows >= f.batch_rows, "flightsql.put_rows must be at least flightsql.batch_rows")?;
        check(f.max_buffer_mib >= 1, "flightsql.max_buffer_mib must be at least 1")?;
        check(f.timeout_s >= 1, "flightsql.timeout_s must be at least 1")?;
        check(f.retries <= 20, "flightsql.retries must be at most 20")?;
        let taken: Vec<String> = gravity_trace_core::schema::trace_schema()
            .fields()
            .iter()
            .chain(crate::writer::pm_schema().fields().iter())
            .map(|f| f.name().clone())
            .collect();
        for k in f.columns.keys() {
            check(!k.is_empty() && !taken.contains(k), &format!("flightsql.columns: `{k}` is a trace / PM column"))?;
        }
        Ok(())
    }

    /// The SDK snapshot options.
    #[cfg(feature = "monitor")]
    pub fn live_options(&self) -> nvtrace_sys::LiveOptions {
        let c = &self.collect;
        nvtrace_sys::LiveOptions {
            work_dir: c.work_dir.clone(),
            snapshot_ms: c.snapshot_ms,
            data_rate_mib_per_s: c.data_rate_mib,
            max_attach_mib: c.max_attach_mib,
            shards: c.shards,
            in_memory: c.storage == Storage::Memory,
            // validated
            ignore: nvtrace_sys::Ignore::parse(&c.ignore.join(",")).unwrap_or_default(),
            ..Default::default()
        }
    }

    /// Effective configuration as TOML (`gt export-daemon config`), secrets masked.
    pub fn to_toml(&self) -> String {
        let mut c = self.clone();
        for s in [&mut c.flightsql.password, &mut c.flightsql.token] {
            if !s.is_empty() {
                *s = "***".into();
            }
        }
        toml::to_string_pretty(&c).unwrap_or_else(|e| format!("# cannot render the configuration: {e}\n"))
    }
}

/// `0,2,4-7` -> [0, 2, 4, 5, 6, 7] (sorted, deduplicated).
pub fn parse_cpu_list(s: &str) -> std::result::Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let num = |x: &str| x.trim().parse::<usize>().map_err(|_| format!("`{part}` is not a CPU number or range"));
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (num(a)?, num(b)?);
                if a > b || b >= 4096 {
                    return Err(format!("bad CPU range `{part}`"));
                }
                out.extend(a..=b);
            }
            None => out.push(num(part)?),
        }
    }
    if out.is_empty() {
        return Err("no CPUs".into());
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_example_file_is_the_defaults() {
        let text = include_str!("../../../config.example.toml");
        let c = parse(text).expect("config.example.toml parses");
        assert_eq!(c, Config::default(), "config.example.toml documents the defaults");
    }

    #[test]
    fn partial_files_keep_the_other_defaults_and_typos_fail() {
        let c = parse("[persist]\nenabled = false\nrow_group_rows = 65536\n[limits]\nmax_cpus = 4\n").expect("valid");
        assert!(!c.persist.enabled);
        assert_eq!(c.persist.row_group_rows, 65536);
        assert_eq!(c.persist.rotate_s, 60);
        assert_eq!(c.limits.max_cpus, 4);
        assert_eq!(c.collect.snapshot_ms, 5000);
        assert!(matches!(parse("[persist]\nrotate = 5\n"), Err(ConfigError::Toml(_))), "unknown key");
        assert!(matches!(parse("[persist]\nzstd_level = 40\n"), Err(ConfigError::Invalid(_))));
        assert!(matches!(parse("[persist]\ngzip_level = 0\n"), Err(ConfigError::Invalid(_))));
        assert!(matches!(parse("[collect]\nignore = [\"gpu_pmm\"]\n"), Err(ConfigError::Invalid(_))));
        assert!(matches!(parse("[collect]\nstorage = \"tape\"\n"), Err(ConfigError::Toml(_))));
    }

    #[test]
    fn flightsql_section() {
        assert!(matches!(parse("[flightsql]\nenabled = true\n"), Err(ConfigError::Invalid(m)) if m.contains("endpoint")));
        #[cfg(feature = "monitor")] // the endpoint's URI is checked by the Flight SQL client
        assert!(matches!(parse("[flightsql]\nenabled = true\nendpoint = \"s3://x\"\n"), Err(ConfigError::Invalid(_))));
        assert!(matches!(parse("[flightsql]\nenabled = true\nendpoint = \"h:1\"\ncolumns = { pid = \"x\" }\n"), Err(ConfigError::Invalid(_))));
        assert!(matches!(parse("[flightsql]\ncompression = \"gzip\"\n"), Err(ConfigError::Toml(_))));
        let c = parse(
            "[flightsql]\nenabled = true\nendpoint = \"grpc+tls://db:443\"\ncompression = \"zstd\"\npassword = \"pw\"\n\
             headers = { database = \"traces\" }\ncolumns = { cluster = \"gpu-cluster-1\" }\n",
        )
        .expect("valid");
        assert_eq!(c.flightsql.compression, Compression::Zstd);
        assert_eq!(c.flightsql.headers["database"], "traces");
        let shown = c.to_toml();
        assert!(shown.contains("password = \"***\"") && !shown.contains("\"pw\""), "{shown}");
        assert_eq!(parse(&shown).expect("renders and parses back").flightsql.columns, c.flightsql.columns);
    }

    fn env(vars: &[(&str, &str)]) -> Vec<(String, String)> {
        vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn the_environment_overrides_the_file() {
        let file = "[persist]\ndir = \"/data/traces\"\nrotate_s = 30\n[limits]\nmax_cpus = 2\n[flightsql]\nheaders = { a = \"1\" }\n";
        let (c, applied) = parse_with_env(
            file,
            env(&[
                ("GT_LIMITS_MAX_CPUS", "4"),
                ("GT_PERSIST_DIR", "s3://bucket/traces"),
                ("GT_PERSIST_ENABLED", "off"),
                ("GT_PERSIST_FORMAT", "perfetto"),
                ("GT_COLLECT_IGNORE", "gpu_pm, nvtx"),
                ("GT_COLLECT_DIAG_FILE", "/tmp/diag.jsonl"),
                ("GT_COLLECT_STORAGE", "disk"),
                ("GT_DAEMON_LISTEN", ""),
                ("GT_FLIGHTSQL_HEADERS", "database=traces, x = y"),
                ("GT_FLIGHTSQL_COLUMNS", "{ cluster = \"gpu-cluster-1\" }"),
                ("GT_FLIGHTSQL_PASSWORD", "pw"),
                ("GT_FLIGHTSQL_COMPRESSION", "zstd"),
                ("GT_LIMITS_NICE", "-5"),
                // not configuration keys
                ("GT_CONFIG", "/x.toml"),
                ("GT_SAVE_DIR", "/tmp"),
                ("PATH", "/bin"),
            ]),
        )
        .expect("valid");
        assert_eq!(c.limits.max_cpus, 4, "the environment wins over the file");
        assert_eq!(c.limits.nice, -5);
        assert_eq!(c.persist.dir, "s3://bucket/traces");
        assert!(!c.persist.enabled);
        assert_eq!(c.persist.format, PersistFormat::Perfetto);
        assert_eq!(c.persist.rotate_s, 30, "keys the environment does not name keep the file's value");
        assert_eq!(c.collect.ignore, ["gpu_pm", "nvtx"]);
        assert_eq!(c.collect.diag_file.as_deref(), Some(Path::new("/tmp/diag.jsonl")));
        assert_eq!(c.collect.storage, Storage::Disk);
        assert_eq!(c.daemon.listen, "", "an empty string is a value");
        assert_eq!(c.flightsql.headers, BTreeMap::from([("database".into(), "traces".into()), ("x".into(), "y".into())]), "replaces the table");
        assert_eq!(c.flightsql.columns["cluster"], "gpu-cluster-1");
        assert_eq!(c.flightsql.password, "pw");
        assert_eq!(c.flightsql.compression, Compression::Zstd);
        assert_eq!(applied.len(), 13);
        assert!(applied.windows(2).all(|w| w[0] < w[1]), "sorted: {applied:?}");

        let (c, applied) = parse_with_env("[collect]\ndiag_file = \"/d.jsonl\"\n", env(&[("GT_COLLECT_DIAG_FILE", "")])).expect("valid");
        assert_eq!(c.collect.diag_file, None, "an empty optional key is unset");
        assert_eq!(applied, ["GT_COLLECT_DIAG_FILE"]);
        let (c, _) = parse_with_env("", env(&[("GT_COLLECT_IGNORE", "[\"cuda_api\"]"), ("GT_PERSIST_PM", "0")])).expect("TOML forms");
        assert_eq!((c.collect.ignore, c.persist.pm), (vec!["cuda_api".to_string()], false));
        let (c, applied) = parse_with_env("[limits]\nmax_cpus = 2\n", env(&[("GT_SHARDS_SERIAL", "1")])).expect("valid");
        assert_eq!((c.limits.max_cpus, applied.len()), (2, 0));
    }

    #[test]
    fn bad_environment_variables_are_errors() {
        let err = |vars: &[(&str, &str)]| parse_with_env("", env(vars)).expect_err("must fail");
        let named = |e: ConfigError, want: &str| match e {
            ConfigError::Env { var, message } => assert_eq!(var, want, "{message}"),
            other => panic!("{want}: {other:?}"),
        };
        named(err(&[("GT_PERSIST_ROTATE", "5")]), "GT_PERSIST_ROTATE"); // no such key
        named(err(&[("GT_LIMITS_MAX_CPUS", "four")]), "GT_LIMITS_MAX_CPUS");
        named(err(&[("GT_LIMITS_MAX_CPUS", "-1")]), "GT_LIMITS_MAX_CPUS"); // u32
        named(err(&[("GT_PERSIST_ENABLED", "maybe")]), "GT_PERSIST_ENABLED");
        named(err(&[("GT_COLLECT_STORAGE", "tape")]), "GT_COLLECT_STORAGE");
        named(err(&[("GT_PERSIST_FORMAT", "json")]), "GT_PERSIST_FORMAT"); // parquet or perfetto
        named(err(&[("GT_FLIGHTSQL_HEADERS", "database")]), "GT_FLIGHTSQL_HEADERS");
        named(err(&[("GT_DAEMON_PID_FILE", "/x"), ("GT_COLLECT_IGNORE", "[1")]), "GT_COLLECT_IGNORE");
        // valid on its own, not as a whole: the message names the variables
        assert!(matches!(err(&[("GT_PERSIST_ZSTD_LEVEL", "40")]), ConfigError::Invalid(m) if m.contains("GT_PERSIST_ZSTD_LEVEL")));
        // the file's own errors come first and point into the file
        assert!(matches!(parse_with_env("[persist]\nrotate = 5\n", env(&[("GT_PERSIST_DIR", "/x")])), Err(ConfigError::Toml(_))));
    }

    #[test]
    fn load_applies_the_environment() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[persist]\ndir = \"/data/traces\"\n").expect("write");
        let l = load_with_env(Some(&path), env(&[("GT_PERSIST_DIR", "/elsewhere")])).expect("load");
        assert_eq!((l.config.persist.dir.as_str(), l.env.as_slice()), ("/elsewhere", ["GT_PERSIST_DIR".to_string()].as_slice()));
        assert_eq!(l.source(), format!("{} + GT_PERSIST_DIR", path.display()));
        let e = load_with_env(Some(&path), env(&[("GT_PERSIST_ZSTD", "1")])).err().map(|e| e.to_string());
        assert_eq!(e.as_deref(), Some("environment variable GT_PERSIST_ZSTD: [persist] has no key `zstd`"));
    }

    #[test]
    fn cpu_lists() {
        assert_eq!(parse_cpu_list("0,2,4-7"), Ok(vec![0, 2, 4, 5, 6, 7]));
        assert_eq!(parse_cpu_list(" 3 , 1-2,2 "), Ok(vec![1, 2, 3]));
        assert!(parse_cpu_list("7-4").is_err());
        assert!(parse_cpu_list("a").is_err());
        assert!(parse_cpu_list(",").is_err());
    }
}
