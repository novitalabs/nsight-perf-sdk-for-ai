//! Crashes with a stack trace.
//!
//! Panics unwind (`panic = "unwind"`, also through the SDK's callbacks, which catch them). The hook prints the thread,
//! where, the message and the stack, always (not only with `RUST_BACKTRACE`; the release build keeps line tables for
//! file:line), and records the first panic: a running export then stops (a pipeline thread that panicked moves no data
//! any more) and the command fails instead of reporting success.
//!
//! Fatal signals of native code (the SDK is C++: SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT) print the native stack:
//! glibc `backtrace_symbols_fd` (the names of exported symbols, i.e. the SDK's C++ functions; `gt(+0xoffset)` for
//! frames in gt), and an `addr2line` command that resolves the gt frames to function and file:line (the release
//! build keeps line tables). Only async-signal-safe calls, on the thread's alternate signal stack when it has one
//! (small: symbolizing in the handler would overflow it). Then the previous handler takes over (Rust's stack
//! overflow report, else the default action: core dump).
use crate::timefmt::{fmt_utc, now_ns};
use std::io::Write;
use std::sync::OnceLock;
use std::time::Duration;

static FIRST_PANIC: OnceLock<String> = OnceLock::new();

/// At the start of `main`, before any thread exists.
pub fn install() {
    std::panic::set_hook(Box::new(|info| {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("<unnamed>");
        let at = info.location().map_or(String::new(), |l| format!(" at {}:{}:{}", l.file(), l.line(), l.column()));
        let msg = payload_message(info.payload());
        let _ = FIRST_PANIC.set(format!("thread '{name}' panicked{at}: {msg}"));
        let bt = from_frame_after(&std::backtrace::Backtrace::force_capture().to_string(), PANIC_MACHINERY);
        // Not eprintln!: a panic while panicking (stderr closed) would abort.
        let _ = writeln!(std::io::stderr().lock(), "{} gt: thread '{name}' panicked{at}:\n{msg}\nstack backtrace:\n{bt}", fmt_utc(now_ns()));
    }));
    native::install();
}

/// Frames of the hook and the panic machinery, above the frame that panicked.
const PANIC_MACHINERY: &[&str] = &[
    "std::panicking::",
    "core::panicking::",
    "rust_begin_unwind",
    "core::result::unwrap_failed",
    "core::option::expect_failed",
    "core::option::unwrap_failed",
];

/// `std::backtrace::Backtrace` text from the frame after the last frame whose name contains one of `markers` (all of
/// it when none does). Frames start with `  <n>: <name>`, followed by `at <file>:<line>` lines.
fn from_frame_after(bt: &str, markers: &[&str]) -> String {
    let lines: Vec<&str> = bt.lines().collect();
    let is_frame = |l: &str| l.trim_start().split_once(": ").is_some_and(|(n, _)| !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()));
    let Some(last) = lines.iter().rposition(|l| is_frame(l) && markers.iter().any(|m| l.contains(m))) else {
        return bt.to_string();
    };
    let start = lines[last + 1..].iter().position(|l| is_frame(l)).map_or(lines.len(), |i| last + 1 + i);
    lines[start..].join("\n")
}

/// `thread 'gt-merge' panicked at src/live.rs:12:5: ...`, once any thread panicked.
pub fn first_panic() -> Option<&'static str> {
    FIRST_PANIC.get().map(String::as_str)
}

/// Resolves once any thread has panicked.
pub async fn panicked() {
    let mut tick = tokio::time::interval(Duration::from_millis(200));
    while first_panic().is_none() {
        tick.tick().await;
    }
}

/// The message of a panic (`panic!("...")`, `expect("...")`).
pub fn payload_message(p: &(dyn std::any::Any + Send)) -> String {
    match (p.downcast_ref::<&str>(), p.downcast_ref::<String>()) {
        (Some(s), _) => (*s).to_string(),
        (None, Some(s)) => s.clone(),
        (None, None) => "(no message)".to_string(),
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod native {
    use std::sync::OnceLock;

    const SIGNALS: [libc::c_int; 5] = [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGFPE, libc::SIGABRT];
    /// The handlers before ours (Rust's own SIGSEGV / SIGBUS stack overflow handler).
    static PREVIOUS: OnceLock<Vec<(libc::c_int, libc::sigaction)>> = OnceLock::new();
    /// Where the gt binary is mapped (start, end) and its path.
    static EXE: OnceLock<(usize, usize, std::ffi::CString)> = OnceLock::new();

    /// The address range of this binary's mappings and its path, from /proc/self/maps.
    fn exe_mapping() -> Option<(usize, usize, std::ffi::CString)> {
        use std::os::unix::ffi::OsStringExt;
        let exe = std::fs::read_link("/proc/self/exe").ok()?;
        let name = exe.to_str()?.to_string();
        let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
        let mut range: Option<(usize, usize)> = None;
        for line in maps.lines() {
            // range perms offset dev inode   path (padded; " (deleted)" once the file was replaced)
            let mut f = line.splitn(6, ' ');
            let (Some(r), Some(path)) = (f.next(), f.nth(4)) else { continue };
            if path.trim_start() != name {
                continue;
            }
            let Some((a, b)) = r.split_once('-') else { continue };
            let (Ok(a), Ok(b)) = (usize::from_str_radix(a, 16), usize::from_str_radix(b, 16)) else { continue };
            range = Some(range.map_or((a, b), |(x, y)| (x.min(a), y.max(b))));
        }
        let (start, end) = range?;
        Some((start, end, std::ffi::CString::new(exe.into_os_string().into_vec()).ok()?))
    }

    // glibc <execinfo.h>
    unsafe extern "C" {
        fn backtrace(buffer: *mut *mut libc::c_void, size: libc::c_int) -> libc::c_int;
        fn backtrace_symbols_fd(buffer: *const *mut libc::c_void, size: libc::c_int, fd: libc::c_int);
    }

    pub fn install() {
        // The first backtrace() loads the unwinder (dlopen, malloc), which a signal handler must not do.
        let mut warm = [std::ptr::null_mut(); 4];
        // SAFETY: the buffer has room for 4 frames.
        unsafe { backtrace(warm.as_mut_ptr(), 4) };
        let mut previous = Vec::new();
        for sig in SIGNALS {
            // SAFETY: an all-zero sigaction is a valid out-parameter.
            let mut old: libc::sigaction = unsafe { std::mem::zeroed() };
            // SAFETY: query only (null new action).
            if unsafe { libc::sigaction(sig, std::ptr::null(), &mut old) } == 0 {
                previous.push((sig, old));
            }
        }
        if PREVIOUS.set(previous).is_err() {
            return; // installed already
        }
        if let Some(m) = exe_mapping() {
            let _ = EXE.set(m);
        }
        let handler: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) = on_fatal;
        for (sig, _) in PREVIOUS.get().map(Vec::as_slice).unwrap_or_default() {
            // SAFETY: as above.
            let mut sa: libc::sigaction = unsafe { std::mem::zeroed() };
            sa.sa_sigaction = handler as libc::sighandler_t;
            // On the alternate stack where the thread has one (Rust threads): a stack overflow is a SIGSEGV too.
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
            // SAFETY: `sa` is initialised; the mask is emptied in place.
            unsafe {
                libc::sigemptyset(&mut sa.sa_mask);
                libc::sigaction(*sig, &sa, std::ptr::null_mut());
            }
        }
    }

    /// Text built on the stack: a signal handler must not allocate.
    struct Line {
        b: [u8; 1024],
        n: usize,
    }

    impl Line {
        fn push(&mut self, s: &[u8]) {
            let k = s.len().min(self.b.len() - self.n);
            self.b[self.n..self.n + k].copy_from_slice(&s[..k]);
            self.n += k;
        }

        fn hex(&mut self, mut v: usize) {
            let mut d = [0u8; 16];
            let mut i = d.len();
            loop {
                i -= 1;
                d[i] = b"0123456789abcdef"[v & 15];
                v >>= 4;
                if v == 0 {
                    break;
                }
            }
            self.push(b"0x");
            self.push(&d[i..]);
        }
    }

    extern "C" fn on_fatal(sig: libc::c_int, info: *mut libc::siginfo_t, _ctx: *mut libc::c_void) {
        // Async-signal-safe calls only: write(2), prctl(2), sigaction(2), raise(3), and backtrace (loaded already).
        let mut l = Line { b: [0; 1024], n: 0 };
        l.push(b"gt: fatal signal ");
        l.push(match sig {
            libc::SIGSEGV => b"SIGSEGV",
            libc::SIGBUS => b"SIGBUS",
            libc::SIGILL => b"SIGILL",
            libc::SIGFPE => b"SIGFPE",
            libc::SIGABRT => b"SIGABRT",
            _ => b"?",
        });
        // SAFETY: the kernel passes a valid siginfo with SA_SIGINFO.
        let (code, addr) = unsafe { ((*info).si_code, (*info).si_addr() as usize) };
        if sig != libc::SIGABRT && code > 0 {
            l.push(b" at address ");
            l.hex(addr);
        }
        let mut name = [0u8; 17];
        // SAFETY: PR_GET_NAME writes at most 16 bytes.
        unsafe { libc::prctl(libc::PR_GET_NAME, name.as_mut_ptr()) };
        let len = name.iter().position(|&c| c == 0).unwrap_or(16);
        l.push(b" in thread '");
        l.push(&name[..len]);
        l.push(b"'; native stack:\n");
        let mut frames = [std::ptr::null_mut(); 64];
        // SAFETY: plain writes of initialised buffers to stderr; `frames` has room for 64 entries.
        let n = unsafe {
            libc::write(2, l.b.as_ptr().cast(), l.n);
            let n = backtrace(frames.as_mut_ptr(), 64);
            backtrace_symbols_fd(frames.as_ptr(), n, 2);
            n.max(0) as usize
        };
        // The gt frames for addr2line: offsets in the binary, return addresses minus one (the calling instruction).
        if let Some((start, end, path)) = EXE.get() {
            l.n = 0;
            l.push(b"gt frames: addr2line -fCipe ");
            l.push(path.as_bytes());
            for &f in &frames[..n.min(frames.len())] {
                let a = f as usize;
                if a > *start && a < *end {
                    l.push(b" ");
                    l.hex(a - start - 1);
                }
            }
            l.push(b"\n");
            // SAFETY: as above.
            unsafe { libc::write(2, l.b.as_ptr().cast(), l.n) };
        }
        // The previous handler takes over: a fault re-executes the faulting instruction on return and arrives there;
        // a signal sent by kill / raise / abort is raised again (pending until this handler returns).
        if let Some((_, old)) = PREVIOUS.get().and_then(|p| p.iter().find(|(s, _)| *s == sig)) {
            // SAFETY: restoring an action read by sigaction(2).
            unsafe { libc::sigaction(sig, old, std::ptr::null_mut()) };
        }
        if code <= 0 || sig == libc::SIGABRT {
            // SAFETY: raise(3) is async-signal-safe.
            unsafe { libc::raise(sig) };
        }
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
mod native {
    pub fn install() {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_messages() {
        let p = std::panic::catch_unwind(|| panic!("static")).expect_err("panics");
        assert_eq!(payload_message(&*p), "static");
        let n = 3;
        let p = std::panic::catch_unwind(|| panic!("formatted {n}")).expect_err("panics");
        assert_eq!(payload_message(&*p), "formatted 3");
        let p = std::panic::catch_unwind(|| std::panic::panic_any(7u8)).expect_err("panics");
        assert_eq!(payload_message(&*p), "(no message)");
    }

    #[test]
    fn backtraces_start_at_the_frame_that_panicked() {
        let bt = "   0: std::backtrace::Backtrace::create\n             at /rustc/x/backtrace.rs:331:13\n   1: core::panicking::panic_fmt\n\
                  \x20  2: gt::live::merge\n             at ./src/live.rs:12:5\n  10: std::thread::spawn\n";
        assert_eq!(from_frame_after(bt, PANIC_MACHINERY), "   2: gt::live::merge\n             at ./src/live.rs:12:5\n  10: std::thread::spawn");
        assert_eq!(from_frame_after("   0: main\n", PANIC_MACHINERY), "   0: main\n", "no marker: everything");
    }

    /// Runs `test` again in a child process with `GT_CRASH_CHILD` set (the hook is process-wide, the crash kills the
    /// process): its exit status and stderr.
    fn child(test: &str) -> (std::process::ExitStatus, String) {
        let exe = std::env::current_exe().expect("test binary");
        let out = std::process::Command::new(exe)
            .args([test, "--exact", "--nocapture", "--test-threads", "1"])
            .env("GT_CRASH_CHILD", "1")
            .output()
            .expect("run the test binary");
        (out.status, String::from_utf8_lossy(&out.stderr).into_owned())
    }

    #[test]
    fn a_panic_prints_its_stack_and_is_recorded() {
        if std::env::var_os("GT_CRASH_CHILD").is_some() {
            install();
            let t = std::thread::Builder::new().name("gt-merge".into()).spawn(|| panic!("injected")).expect("spawn");
            let p = t.join().expect_err("the thread panicked");
            assert_eq!(payload_message(&*p), "injected");
            let first = first_panic().expect("recorded");
            assert!(first.starts_with("thread 'gt-merge' panicked at ") && first.ends_with(": injected"), "{first}");
            return;
        }
        let (status, err) = child("crash::tests::a_panic_prints_its_stack_and_is_recorded");
        assert!(status.success(), "{status}: {err}");
        assert!(err.contains("gt: thread 'gt-merge' panicked at crates/gravity-trace/src/crash.rs:"), "{err}");
        assert!(err.contains("stack backtrace:"), "{err}");
        assert!(err.contains("a_panic_prints_its_stack_and_is_recorded"), "a frame of the test: {err}");
        assert!(err.contains("crash.rs:"), "file:line in the frames: {err}");
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn a_native_crash_prints_its_stack() {
        use std::os::unix::process::ExitStatusExt;
        if std::env::var_os("GT_CRASH_CHILD").is_some() {
            install();
            // SAFETY: none: a deliberate fault inside libc, like a crash in the SDK (black_box: strlen is pure, an
            // unused call would be optimised away).
            std::hint::black_box(unsafe { libc::strlen(std::hint::black_box(8 as *const libc::c_char)) });
            return;
        }
        let (status, err) = child("crash::tests::a_native_crash_prints_its_stack");
        assert_eq!(status.signal(), Some(libc::SIGSEGV), "dies of the signal (core dump), as without the handler: {err}");
        assert!(err.contains("gt: fatal signal SIGSEGV at address 0x8 in thread '"), "{err}");
        assert!(err.contains("native stack"), "{err}");
        let frames = err.lines().filter(|l| l.contains("[0x")).count();
        assert!(frames >= 3, "{frames} frames: {err}");
        let cmd = err.lines().find_map(|l| l.strip_prefix("gt frames: addr2line -fCipe ")).unwrap_or_default();
        let exe = std::env::current_exe().expect("test binary").display().to_string();
        assert!(cmd.starts_with(&format!("{exe} 0x")), "{err}");
        // the first gt frame after the fault is the test calling strlen: its offset is what glibc printed, minus one
        let printed: Vec<usize> = err
            .lines()
            .filter_map(|l| l.split_once("(+0x").filter(|(p, _)| p.ends_with(exe.rsplit('/').next().unwrap_or("?"))))
            .filter_map(|(_, rest)| usize::from_str_radix(rest.split(')').next().unwrap_or(""), 16).ok())
            .collect();
        let ours: Vec<usize> = cmd.split(' ').skip(1).filter_map(|h| usize::from_str_radix(h.trim_start_matches("0x"), 16).ok()).collect();
        assert_eq!(ours, printed.iter().map(|o| o - 1).collect::<Vec<_>>(), "{err}");
    }
}
