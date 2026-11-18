//! The receiving end of `bench/flightsql_check.sh`: an Arrow Flight SQL server that accepts bulk ingestion
//! (`CommandStatementIngest`) and writes every DoPut to `<out>/<table>/<n>.parquet`, sampling the bytes it receives
//! every second.
//!
//! ```text
//! flightsql_sink --listen 127.0.0.1:50051 --out /tmp/sink [--user gt --password pw] [--traffic traffic.jsonl]
//! ```
//! Runs until SIGINT / SIGTERM, then prints a JSON summary (rows, DoPuts, bytes per table).
#[path = "../src/flight_sink.rs"]
mod flight_sink;

use std::io::Write as _;
use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (mut listen, mut out, mut user, mut password, mut traffic) = ("127.0.0.1:50051".to_string(), None, None, None, None);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--listen" => listen = val()?,
            "--out" => out = Some(std::path::PathBuf::from(val()?)),
            "--user" => user = Some(val()?),
            "--password" => password = Some(val()?),
            "--traffic" => traffic = Some(val()?),
            _ => return Err(format!("unknown argument {a}").into()),
        }
    }
    let auth = user.map(|u| (u, password.unwrap_or_default()));
    let sink = flight_sink::Sink::new(out, auth);
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(&listen).await?;
        eprintln!("flightsql_sink listening on {}", listener.local_addr()?);
        if let Some(path) = traffic {
            let bytes = sink.bytes.clone();
            let mut f = std::fs::File::create(path)?;
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(Duration::from_secs(1));
                let mut last = 0;
                loop {
                    tick.tick().await;
                    let now = bytes.load(Ordering::Relaxed);
                    let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
                    let _ = writeln!(f, "{{\"t\": {t:.3}, \"bytes\": {}}}", now - last);
                    last = now;
                }
            });
        }
        let shutdown = async {
            let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
        };
        flight_sink::serve(listener, sink.clone(), shutdown).await?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;
    let st = sink.state.lock().unwrap_or_else(|e| e.into_inner());
    let tables: serde_json::Map<String, serde_json::Value> = st
        .tables
        .iter()
        .map(|(k, t)| {
            let v = serde_json::json!({"rows": t.rows, "puts": t.puts, "bytes": t.bytes, "files": t.files.len(),
                "receive_s": t.receive_s, "write_s": t.write_s});
            (k.clone(), v)
        })
        .collect();
    println!("{}", serde_json::json!({"tables": tables, "bytes": sink.bytes.load(Ordering::Relaxed), "puts_failed": st.fail_puts}));
    Ok(())
}
