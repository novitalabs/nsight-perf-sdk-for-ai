//! The daemon's pid file, which is also its lock (`flock`): held for the daemon's (or `gt export`'s) lifetime and
//! released by the kernel even after a crash, so a stale file never blocks a start, `stop` never signals a recycled
//! pid, and `gt dump` knows whether a daemon is there to close its open file.
use crate::error::{AlreadyRunningSnafu, CreateDirSnafu, PidFileSnafu, Result};
use snafu::ResultExt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

/// Seen through the pid file.
#[derive(Debug, PartialEq)]
pub(crate) enum State {
    Stopped,
    /// Locked, the pid not written yet.
    Starting,
    Running(u32),
}

pub(crate) fn state(pid_file: &Path) -> Result<State> {
    let mut f = match File::open(pid_file) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(State::Stopped),
        Err(e) => return Err(e).context(PidFileSnafu { path: pid_file.to_path_buf() }),
    };
    // A shared lock is refused while the daemon holds its exclusive one.
    // SAFETY: flock on a file descriptor owned by `f`.
    if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) } == 0 {
        return Ok(State::Stopped); // released when `f` is closed
    }
    let mut s = String::new();
    f.read_to_string(&mut s).context(PidFileSnafu { path: pid_file.to_path_buf() })?;
    Ok(s.trim().parse().map_or(State::Starting, State::Running))
}

/// Pid of the daemon serving with this pid file.
pub(crate) fn running_pid(pid_file: &Path) -> Option<u32> {
    match state(pid_file) {
        Ok(State::Running(pid)) => Some(pid),
        _ => None,
    }
}

/// The daemon's (or `gt export`'s) hold on its pid file.
pub(crate) struct PidLock {
    file: File,
    path: PathBuf,
}

impl PidLock {
    pub(crate) fn acquire(path: &Path) -> Result<PidLock> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d).context(CreateDirSnafu { path: d.to_path_buf() })?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .context(PidFileSnafu { path: path.to_path_buf() })?;
        // SAFETY: flock on a file descriptor owned by `file`.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let pid = std::fs::read_to_string(path).unwrap_or_default().trim().to_string();
            return AlreadyRunningSnafu { pid: if pid.is_empty() { "starting".to_string() } else { pid }, pid_file: path.to_path_buf() }
                .fail();
        }
        // Whatever an earlier, crashed daemon left there is not our pid.
        file.set_len(0).context(PidFileSnafu { path: path.to_path_buf() })?;
        Ok(PidLock { file, path: path.to_path_buf() })
    }

    /// Publish the pid: the daemon is serving.
    pub(crate) fn ready(&mut self) -> Result<()> {
        let ctx = PidFileSnafu { path: self.path.clone() };
        self.file.set_len(0).context(ctx.clone())?;
        self.file.seek(SeekFrom::Start(0)).context(ctx.clone())?;
        writeln!(self.file, "{}", std::process::id()).context(ctx.clone())?;
        self.file.sync_all().context(ctx)
    }
}

impl Drop for PidLock {
    fn drop(&mut self) {
        // The lock goes with the descriptor right after.
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pid_file_is_a_lock() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("run/gt.pid");
        assert_eq!(state(&path).expect("state"), State::Stopped, "no file");
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(&path, "12345\n").expect("stale pid file");
        assert_eq!(state(&path).expect("state"), State::Stopped, "a stale file without a lock is not a daemon");
        let mut lock = PidLock::acquire(&path).expect("lock");
        assert_eq!(state(&path).expect("state"), State::Starting, "locked, the stale pid is gone");
        assert!(matches!(PidLock::acquire(&path), Err(crate::error::Error::AlreadyRunning { .. })));
        lock.ready().expect("ready");
        assert_eq!(state(&path).expect("state"), State::Running(std::process::id()));
        drop(lock);
        assert_eq!(state(&path).expect("state"), State::Stopped);
        assert!(!path.exists());
    }
}
