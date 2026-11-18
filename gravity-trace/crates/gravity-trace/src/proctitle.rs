//! Process titles: the name a process gives itself by rewriting its command line (`setproctitle`), e.g. vLLM's
//! `VLLM::Worker_TP0`, `VLLM::EngineCore`. The SDK records only the executable (`python3.12` for every vLLM process),
//! so the title is read from the procfs of the host's PID namespace by host pid: `/proc` when gt runs there
//! (`hostPID`, `--pid=host`, bare metal), else the host's `/proc` mounted read-only (`[collect] host_proc`).
//!
//! A process is only named when the procfs entry is the process the SDK reported: its `NSpid` must list the host pid
//! first and the pid in the process's own namespace last (a procfs of another PID namespace, or a reused pid, does
//! not match). `cmdline` and `status` are readable by any user.
use std::path::Path;

/// The longest title kept (a rewritten command line can be padded to the size of the original one).
const MAX_TITLE: usize = 128;

/// The title of the process with host pid `pid` (`ns_pid` in its own PID namespace) under `proc_dir`: its command
/// line when that is a single argument that is not a path (a rewritten one; a normal command line is the program
/// and its arguments). `None` when the process is gone, the procfs is not the host's, or there is no title.
pub fn title(proc_dir: &Path, pid: u32, ns_pid: u32) -> Option<String> {
    let dir = proc_dir.join(pid.to_string());
    let status = std::fs::read_to_string(dir.join("status")).ok()?;
    if !same_process(&status, pid, ns_pid) {
        return None;
    }
    from_cmdline(&std::fs::read(dir.join("cmdline")).ok()?)
}

/// `NSpid:` lists the pid in each PID namespace from the procfs's down to the process's own.
fn same_process(status: &str, pid: u32, ns_pid: u32) -> bool {
    let Some(line) = status.lines().find_map(|l| l.strip_prefix("NSpid:")) else {
        return pid == ns_pid; // a kernel without NSpid (< 4.1): no namespaces to tell apart
    };
    let ids: Vec<u32> = line.split_whitespace().filter_map(|s| s.parse().ok()).collect();
    ids.first() == Some(&pid) && ids.last() == Some(&ns_pid)
}

fn from_cmdline(raw: &[u8]) -> Option<String> {
    let mut args = raw.split(|&b| b == 0).filter(|a| !a.is_empty());
    let first = args.next()?;
    if args.next().is_some() {
        return None;
    }
    let t = String::from_utf8_lossy(first);
    let t = t.trim();
    (!t.is_empty() && !t.contains('/')).then(|| t.chars().take(MAX_TITLE).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn titles_are_rewritten_command_lines() {
        assert_eq!(from_cmdline(b"VLLM::Worker_TP0\0\0\0\0").as_deref(), Some("VLLM::Worker_TP0"));
        assert_eq!(from_cmdline(b"VLLM::EngineCore   \0").as_deref(), Some("VLLM::EngineCore"), "padding trimmed");
        assert_eq!(from_cmdline(b"postgres: writer process\0").as_deref(), Some("postgres: writer process"));
        assert_eq!(from_cmdline(b"/usr/bin/python3\0-m\0vllm.entrypoints.cli.main\0serve\0"), None, "a command line");
        assert_eq!(from_cmdline(b"/usr/bin/python3\0"), None, "a path");
        assert_eq!(from_cmdline(b""), None, "a kernel thread / zombie");
    }

    #[test]
    fn the_procfs_entry_must_be_the_reported_process() {
        let status = "Name:\tVLLM::Worker_T\nNSpid:\t4711\t1\nNSpgid:\t4711\t1\n";
        assert!(same_process(status, 4711, 1), "host pid first, own namespace last");
        assert!(!same_process(status, 4711, 4711), "another process in the host namespace");
        assert!(!same_process("NSpid:\t4711\n", 4711, 1), "a procfs of another namespace");
        assert!(same_process("NSpid:\t4711\n", 4711, 4711));
    }

    #[test]
    fn titles_of_running_processes() {
        let me = std::process::id();
        assert_eq!(title(Path::new("/proc"), me, me), None, "the test binary has a normal command line");
        assert_eq!(title(Path::new("/proc"), me, me + 1), None, "wrong namespace pid");
        assert_eq!(title(Path::new("/nonexistent"), me, me), None);
    }
}
