//! Prometheus text exposition of the per-process kernel metrics (`nvtrace_*` series).
use gravity_trace_core::metrics::{KernelMetric, Metrics};
use std::fmt::Write as _;

pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// The `pod` label: $HOSTNAME (the pod name in Kubernetes), else the host name.
pub fn pod_name() -> String {
    std::env::var("HOSTNAME").ok().unwrap_or_else(crate::host)
}

/// Escape a label value.
pub fn esc(v: &str) -> String {
    let mut s = String::with_capacity(v.len());
    for c in v.chars() {
        match c {
            '\\' => s.push_str("\\\\"),
            '"' => s.push_str("\\\""),
            '\n' => s.push_str("\\n"),
            c => s.push(c),
        }
    }
    s
}

fn header(out: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
}

/// Process, per-GPU and per-kernel series of `procs`, every metric name prefixed with `prefix`.
pub fn render(prefix: &str, pod: &str, procs: &[Metrics], failed: usize) -> String {
    let mut o = String::with_capacity(16 * 1024);
    let base = |m: &Metrics| {
        format!(r#"pod="{}",host="{}",pid="{}",ns_pid="{}",exe="{}",title="{}""#, esc(pod), esc(&m.host), m.pid, m.ns_pid, esc(&m.exe), esc(&m.title))
    };

    header(&mut o, &format!("{prefix}_processes"), "gauge", "Traced processes seen in the snapshots so far.");
    let _ = writeln!(o, "{prefix}_processes{{pod=\"{}\"}} {}", esc(pod), procs.len());
    header(&mut o, &format!("{prefix}_scrape_errors"), "gauge", "Sources that failed to answer this scrape.");
    let _ = writeln!(o, "{prefix}_scrape_errors{{pod=\"{}\"}} {failed}", esc(pod));

    header(&mut o, &format!("{prefix}_enabled"), "gauge", "1 if GPU times come from hardware timestamps.");
    for m in procs {
        let _ = writeln!(o, "{prefix}_enabled{{{}}} {}", base(m), u8::from(m.enabled));
    }
    header(&mut o, &format!("{prefix}_records_total"), "counter", "Activity records processed.");
    for m in procs {
        let _ = writeln!(o, "{prefix}_records_total{{{}}} {}", base(m), m.records);
    }
    header(&mut o, &format!("{prefix}_records_dropped_total"), "counter", "Activity records dropped by the source.");
    for m in procs {
        let _ = writeln!(o, "{prefix}_records_dropped_total{{{}}} {}", base(m), m.dropped);
    }

    header(&mut o, &format!("{prefix}_gpu_kernel_busy_seconds_total"), "counter", "Union of kernel execution intervals per GPU.");
    for m in procs {
        for d in &m.devices {
            let _ = writeln!(o, "{prefix}_gpu_kernel_busy_seconds_total{{{},device=\"{}\"}} {:.9}", base(m), d.device, d.kernel_busy_ns as f64 / 1e9);
        }
    }
    header(&mut o, &format!("{prefix}_gpu_kernels_total"), "counter", "Kernels executed per GPU.");
    for m in procs {
        for d in &m.devices {
            let _ = writeln!(o, "{prefix}_gpu_kernels_total{{{},device=\"{}\"}} {}", base(m), d.device, d.kernels);
        }
    }

    let series = |o: &mut String, name: &str, kind: &str, help: &str, f: &dyn Fn(&KernelMetric) -> String| {
        header(o, name, kind, help);
        for m in procs {
            for k in &m.kernels {
                let _ = writeln!(
                    o,
                    "{name}{{{},device=\"{}\",kind=\"{}\",name=\"{}\"}} {}",
                    base(m),
                    k.device,
                    k.kind.as_str(),
                    esc(&k.name),
                    f(k)
                );
            }
        }
    };
    series(&mut o, &format!("{prefix}_activity_seconds_total"), "counter", "Cumulative GPU time per kernel/memcpy/memset name.", &|k| {
        format!("{:.9}", k.total_ns as f64 / 1e9)
    });
    series(&mut o, &format!("{prefix}_activity_count_total"), "counter", "Executions per kernel/memcpy/memset name.", &|k| k.count.to_string());
    series(&mut o, &format!("{prefix}_activity_p50_seconds"), "gauge", "Median duration in the last stats window.", &|k| {
        format!("{:.9}", k.last_p50_ns as f64 / 1e9)
    });
    series(&mut o, &format!("{prefix}_activity_p99_seconds"), "gauge", "p99 duration in the last stats window.", &|k| {
        format!("{:.9}", k.last_p99_ns as f64 / 1e9)
    });
    o
}
