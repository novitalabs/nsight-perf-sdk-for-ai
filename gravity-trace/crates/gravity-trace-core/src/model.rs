//! Event kinds and the row type that NvTrace rows are mapped to before they go into Parquet.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Kernel,
    Memcpy,
    Memset,
    Runtime,
    Driver,
}

impl EventKind {
    pub const ALL: [EventKind; 5] =
        [EventKind::Kernel, EventKind::Memcpy, EventKind::Memset, EventKind::Runtime, EventKind::Driver];

    pub fn as_str(self) -> &'static str {
        match self {
            EventKind::Kernel => "kernel",
            EventKind::Memcpy => "memcpy",
            EventKind::Memset => "memset",
            EventKind::Runtime => "runtime",
            EventKind::Driver => "driver",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.as_str() == s)
    }

    /// GPU-side activity (has device/stream) vs host-side API call.
    pub fn is_gpu(self) -> bool {
        matches!(self, EventKind::Kernel | EventKind::Memcpy | EventKind::Memset)
    }
}

/// One traced activity. GPU-only fields are `None` for API records and vice versa.
/// Timestamps are UTC ns (NvTrace GPU and CPU timestamps share that clock).
#[derive(Clone, Debug, Default)]
pub struct TraceRow<'a> {
    pub kind: Option<EventKind>,
    pub name: &'a str,
    pub start_ns: i64,
    pub end_ns: i64,
    /// In the service's PID namespace (the host's): unique on the node, what `nvidia-smi` shows.
    pub pid: u32,
    /// In the process's own PID namespace (`getpid()`, e.g. 1 for a container's first process); `None` in files
    /// written before the column existed.
    pub ns_pid: Option<u32>,
    pub tid: Option<u64>,
    pub device_id: Option<u32>,
    pub context_id: Option<u32>,
    pub stream_id: Option<u32>,
    pub correlation_id: u32,
    pub graph_id: Option<u32>,
    pub graph_node_id: Option<u64>,
    pub grid: Option<[i32; 3]>,
    pub block: Option<[i32; 3]>,
    pub registers_per_thread: Option<u16>,
    pub static_shared_mem: Option<i32>,
    pub dynamic_shared_mem: Option<i32>,
    pub bytes: Option<u64>,
    pub channel_id: Option<u32>,
    pub device_launched: Option<bool>,
}

/// A traced process. `pid` is its pid in the service's PID namespace (the host's: the Nsight Perf service runs there),
/// unique on the node and what `nvidia-smi` and `ps` on the host show; `ns_pid` its pid in its own PID namespace
/// (`getpid()`: 1 for a container's first process). Processes in different containers can have the same `ns_pid`,
/// never the same `pid`. Equal outside containers.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessInfo {
    pub pid: u32,
    pub ns_pid: u32,
    /// File name of the executable.
    pub exe: String,
    /// The process's own name for itself, when it rewrote its command line (`setproctitle`): vLLM's
    /// `VLLM::Worker_TP0`, `VLLM::EngineCore`. Empty when there is none, or gt cannot see the host's procfs.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
}

impl ProcessInfo {
    /// The process's name: its title, else its executable.
    pub fn name(&self) -> &str {
        if self.title.is_empty() { &self.exe } else { &self.title }
    }

    /// Track name: the name and the host pid, `VLLM::Worker_TP0 (pid 4711)`, `python3 (pid 4711)`; `pid 4711`
    /// without a name.
    pub fn label(&self) -> String {
        match self.name() {
            "" => format!("pid {}", self.pid),
            n => format!("{n} (pid {})", self.pid),
        }
    }

    /// For people: `4711 python3`, `4711 python3 VLLM::Worker_TP0 (pid 1 in its namespace)`.
    pub fn describe(&self) -> String {
        let title = if self.title.is_empty() { String::new() } else { format!(" {}", self.title) };
        let ns = if self.ns_pid != self.pid { format!(" (pid {} in its namespace)", self.ns_pid) } else { String::new() };
        format!("{} {}{title}{ns}", self.pid, self.exe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_labels() {
        let p = |pid, ns_pid, exe: &str| ProcessInfo { pid, ns_pid, exe: exe.into(), title: String::new() };
        assert_eq!(p(4711, 1, "python3").label(), "python3 (pid 4711)", "the host pid");
        assert_eq!(p(4711, 1, "").label(), "pid 4711");
        let worker = ProcessInfo { title: "VLLM::Worker_TP0".into(), ..p(4711, 1, "python3.12") };
        assert_eq!(worker.label(), "VLLM::Worker_TP0 (pid 4711)", "the title before the executable");
        assert_eq!(p(4711, 4711, "python3").describe(), "4711 python3");
        assert_eq!(worker.describe(), "4711 python3.12 VLLM::Worker_TP0 (pid 1 in its namespace)");
    }
}
