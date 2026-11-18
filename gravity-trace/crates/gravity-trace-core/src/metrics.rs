//! Per-process metrics rendered by the Prometheus exporter: cumulative per-kernel counters and per-GPU busy time.
use crate::model::EventKind;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub pid: u32,
    /// The pid in the process's own PID namespace and its title (see [`crate::model::ProcessInfo`]).
    pub ns_pid: u32,
    pub title: String,
    pub host: String,
    pub exe: String,
    /// Exported as `<prefix>_enabled` (always 1 for NvTrace).
    pub enabled: bool,
    pub records: u64,
    pub dropped: u64,
    pub kernels: Vec<KernelMetric>,
    pub devices: Vec<DeviceMetric>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelMetric {
    pub kind: EventKind,
    pub device: u32,
    pub name: String,
    pub count: u64,
    pub total_ns: u64,
    /// Quantiles of the last closed stats window (0 if not seen in it).
    pub last_p50_ns: u64,
    pub last_p99_ns: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceMetric {
    pub device: u32,
    /// Union of kernel intervals since start (ns of GPU time covered by at least one kernel).
    pub kernel_busy_ns: u64,
    pub kernels: u64,
    /// Latest kernel end timestamp merged so far (ns). A live busy ratio is
    /// `Δkernel_busy_ns / Δbusy_horizon_ns`. Using GPU time as the denominator removes the
    /// jitter of record delivery (flush batches), which a wall-clock denominator would include.
    #[serde(default)]
    pub busy_horizon_ns: i64,
}
