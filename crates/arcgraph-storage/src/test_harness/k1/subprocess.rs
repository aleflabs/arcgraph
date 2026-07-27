//! SIGKILL subprocess crash harness for K-1.
//!
//! ## Why subprocess (vs. in-thread fault injection)?
//!
//! Phase 5.5 (`tests/phase_5_5_torture.rs`) injects WAL faults via
//! `WalWriter::shutdown()` + `recover_from_wal` cycle in a single
//! process. That is the GRACEFUL teardown path — Drop runs, the
//! WalWriter's background fsync drains, async cleanup completes.
//! Real production crashes are NOT graceful: a crashing process
//! gets SIGKILL'd by the kernel (OOM-killer, SIGSEGV-with-core-
//! dumps-disabled, parent-process kill-9, hardware reset). No Drop
//! runs. No async cleanup. No panic handlers fire.
//!
//! K-1's load-bearing crash test is the SIGKILL case. Recovery from
//! pure WAL replay (no graceful teardown) is the contract that
//! ADR-031 §R3 commit-bundle atomicity + ADR-034 D-1 strict
//! durability rest on. If recovery only works after graceful
//! teardown, the project's durability claims are over-stated.
//!
//! ## Subprocess strategy
//!
//! Per spec D3, parent forks child via `Command::spawn(current_exe)`
//! with an env-var-driven workload selector. macOS lacks `fork()` on
//! multi-threaded processes (rust-lang/rust#80265); Linux requires
//! careful WAL handle teardown across the fork. The portable
//! solution is `Command::spawn`-based subprocess execution: the
//! child re-execs the test binary with `K1_SUBPROCESS_WORKLOAD=<id>`
//! set; the test binary's `main` checks the env var and routes to
//! the workload entry point.
//!
//! ### Workload entry-point convention
//!
//! Test binaries that opt into K-1 subprocess crash testing call
//! [`maybe_dispatch_subprocess_workload`] at the top of `main()`
//! (or from a `#[ctor]` if available). If the env var is set, the
//! entry point runs the workload-to-completion (or until SIGKILL
//! arrives) and exits — `main()` never returns control. If the env
//! var is absent, the function returns and `main()` proceeds with
//! the parent-side test logic.
//!
//! `tests/k1_smoke_30s.rs` calls this dispatcher from a
//! `#[test]` body before the parent-side workload runs (so the
//! parent test process is also a candidate for spawning child
//! subprocesses). The subprocess workload itself is registered via
//! [`SubprocessWorkloadRegistry::register`] before the test body
//! runs.
//!
//! ## SIGKILL semantics
//!
//! [`ChildHandle::kill_with_sigkill`] sends `SIGKILL` (signal 9) via
//! [`std::process::Child::kill`], which the standard library
//! documents as "equivalent to sending a SIGKILL on Unix platforms"
//! (Rust std docs `std::process::Child::kill`). Windows is not
//! supported at v1.0 (per `docs/testing-strategy.md` §5 — Windows is
//! explicitly out of v1.0 scope). On Unix the SIGKILL delivery is
//! synchronous: the kernel marks the process as terminated; the
//! next scheduler tick teardowns the address space. `waitpid`
//! returns with `WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL`
//! shortly after.
//!
//! No FFI dep — std-only. Avoids adding `nix` or `libc` as a direct
//! dep on the storage crate (Prime Directive #1: every dep is a
//! contract surface).
//!
//! ## Recovery verification
//!
//! After SIGKILL, the parent restarts the child with a `recover`
//! workload variant that:
//!
//! 1. Opens the same WAL directory the killed child wrote to.
//! 2. Calls `recover_from_wal` against a fresh `TxnManager`.
//! 3. Reads back every committed record observable post-recovery.
//! 4. Writes the read-back state to a parent-readable file (or
//!    pipes it back) so the parent's oracle can compare against
//!    the pre-crash committed state recorded by the killed child
//!    before it died.
//!
//! The pre-crash committed state is captured by the child writing
//! a "ledger" file inside the workload — every successful commit
//! appends a `(tenant, id, label, a, b, tier)` tuple to the ledger.
//! The parent reads the ledger after SIGKILL + restart, then
//! compares against the recovered store's read-back state. The
//! oracle in [`super::oracle`] performs the comparison.
//!
//! ## What this module is NOT
//!
//! - It is NOT a `fork()` wrapper. macOS rules that out for
//!   multi-threaded test binaries; Linux works but isn't portable.
//! - It is NOT a Jepsen DSL replacement. K-1 is harness scaffolding;
//!   v1.1 may add a Jepsen DSL wrapper on top.
//! - It is NOT a process-supervisor. We don't restart on every crash;
//!   we restart on the explicit `kill_with_sigkill_and_recover` call.
//!
//! ## End-to-end exercise
//!
//! Per codex M3 retro Finding HIGH-1 (PR #176 review): pre-fix this
//! module shipped ~565 LOC with NO end-to-end exercise — the
//! registered workloads in `tests/k1_smoke_30s.rs::subprocess_smoke`
//! just `thread::sleep`. Post-fix, `tests/k1_subprocess_smoke.rs`
//! (gated `K1_SUBPROCESS_SMOKE=1`) opens a real WAL stack in the
//! child, commits N rows under `crate::wal::DurabilityTier::Strict`,
//! records each to [`PreCrashLedger`], gets SIGKILL'd by the parent's
//! crash window, and the parent restarts + replays WAL + asserts
//! T1-Strict + 1:1 unique:total invariants via [`super::oracle::
//! verify_post_recovery_invariants`].

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Env var the parent sets to route a child subprocess to a
/// registered workload. Absent in the parent.
pub const SUBPROCESS_WORKLOAD_ENV: &str = "K1_SUBPROCESS_WORKLOAD";

/// Env var carrying the workload's parameter (typically a path to
/// the WAL directory the workload writes into). The workload reads
/// this when it routes off `SUBPROCESS_WORKLOAD_ENV`.
pub const SUBPROCESS_WORKLOAD_ARG_ENV: &str = "K1_SUBPROCESS_WORKLOAD_ARG";

/// Process-exit code the child writes when its workload completes
/// successfully WITHOUT being SIGKILL'd. The parent uses this to
/// distinguish "workload finished before crash window expired" from
/// "workload was SIGKILL'd by the parent" (`status.signal() == 9`).
pub const WORKLOAD_CLEAN_EXIT_CODE: i32 = 42;

/// Workload signature: `fn(arg: &str) -> i32` (process exit code).
pub type WorkloadFn = fn(arg: &str) -> i32;

/// Registry of workload names → entry points. Test binaries call
/// [`SubprocessWorkloadRegistry::register`] before executing the
/// `#[test]` body; child subprocesses look up by name in their
/// dispatcher.
pub struct SubprocessWorkloadRegistry {
    map: Mutex<HashMap<String, WorkloadFn>>,
}

impl SubprocessWorkloadRegistry {
    fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
        }
    }

    /// Register a workload by name. Called by test binaries that
    /// participate in subprocess crash testing.
    pub fn register(name: impl Into<String>, f: WorkloadFn) {
        let n = name.into();
        let r = global();
        r.map
            .lock()
            .expect("workload registry poisoned")
            .insert(n, f);
    }

    fn lookup(&self, name: &str) -> Option<WorkloadFn> {
        self.map
            .lock()
            .expect("workload registry poisoned")
            .get(name)
            .copied()
    }
}

fn global() -> &'static SubprocessWorkloadRegistry {
    static REG: OnceLock<SubprocessWorkloadRegistry> = OnceLock::new();
    REG.get_or_init(SubprocessWorkloadRegistry::new)
}

/// Top-of-`main` dispatcher. If the child env vars are set, runs
/// the registered workload to completion and exits the process.
/// Otherwise returns so parent-side test logic proceeds.
///
/// Test binaries call this from a helper invoked at the top of
/// every test body that may spawn child subprocesses (or from
/// `#[ctor]` / a process-init hook if available). Calling it
/// multiple times is safe — once the env var is consumed, subsequent
/// calls no-op because the child has already exited.
pub fn maybe_dispatch_subprocess_workload() {
    let Ok(name) = std::env::var(SUBPROCESS_WORKLOAD_ENV) else {
        return;
    };
    let arg = std::env::var(SUBPROCESS_WORKLOAD_ARG_ENV).unwrap_or_default();
    let workload = match global().lookup(&name) {
        Some(w) => w,
        None => {
            eprintln!(
                "k1 subprocess: workload `{name}` not registered; \
                 test binary forgot to call SubprocessWorkloadRegistry::register"
            );
            std::process::exit(127);
        }
    };
    let code = workload(&arg);
    std::process::exit(code);
}

/// Handle to a spawned child subprocess running a K-1 workload.
pub struct ChildHandle {
    child: Child,
    pid: u32,
    /// Spawn time — used to enforce the crash window.
    spawned_at: Instant,
    /// Workload name (for error messages).
    workload: String,
    /// Env var arg passed to the child.
    arg: String,
}

impl ChildHandle {
    pub fn pid(&self) -> u32 {
        self.pid
    }

    pub fn workload(&self) -> &str {
        &self.workload
    }

    pub fn arg(&self) -> &str {
        &self.arg
    }

    pub fn spawned_at(&self) -> Instant {
        self.spawned_at
    }

    /// Send SIGKILL. Returns `Ok(())` if the kill landed; `Err` if
    /// it failed (typically because the process already exited). The
    /// caller MUST follow up with [`Self::wait`] to reap the zombie.
    ///
    /// `Child::kill` is "equivalent to sending a SIGKILL on Unix
    /// platforms" per the Rust std docs; on Windows it terminates
    /// the process via `TerminateProcess` (Windows is out of v1.0
    /// scope per `docs/testing-strategy.md` §5; the call still
    /// works as a "force-terminate" but the SIGKILL semantics rules
    /// don't apply).
    pub fn kill_with_sigkill(&mut self) -> std::io::Result<()> {
        self.child.kill()
    }

    /// Reap the child. Returns the exit status (which encodes the
    /// SIGKILL signal on Unix per `WIFSIGNALED` / `WTERMSIG`).
    pub fn wait(mut self) -> std::io::Result<ExitStatus> {
        self.child.wait()
    }

    /// Convenience: send SIGKILL, then reap. Returns the exit status.
    pub fn kill_and_wait(mut self) -> std::io::Result<ExitStatus> {
        let _ = self.kill_with_sigkill();
        self.child.wait()
    }
}

/// Crash-window record. Surfaces what the harness fired so the
/// caller / oracle can validate the crash actually landed (vs. the
/// workload exiting cleanly before the window).
#[derive(Debug, Clone)]
pub struct CrashRecord {
    pub workload: String,
    pub arg: String,
    pub pid: u32,
    /// Wall-clock duration between spawn and SIGKILL.
    pub elapsed_to_kill: Duration,
    /// `true` iff the kill syscall succeeded (i.e., the child was
    /// still alive and got the signal).
    pub kill_succeeded: bool,
    /// Exit status as returned by `waitpid`. On Unix, this carries
    /// the signal-exit info via `ExitStatus::signal()`.
    pub exit_status: ExitStatus,
}

impl CrashRecord {
    /// True iff the exit reflects a SIGKILL (signal 9 on Unix).
    /// Returns `false` on non-Unix targets (Windows is out of v1.0
    /// scope per `docs/testing-strategy.md` §5).
    #[cfg(unix)]
    pub fn was_sigkilled(&self) -> bool {
        use std::os::unix::process::ExitStatusExt;
        // SIGKILL is signal 9 on every Unix platform Rust supports
        // (POSIX-mandated; macOS / Linux / *BSD all share it).
        self.exit_status.signal() == Some(9)
    }

    #[cfg(not(unix))]
    pub fn was_sigkilled(&self) -> bool {
        false
    }

    /// True iff the workload exited cleanly before the crash window
    /// expired (status code matches `WORKLOAD_CLEAN_EXIT_CODE`). A
    /// clean exit during a campaign that EXPECTS a crash is a
    /// harness regression — increase the workload duration or
    /// shorten the crash window.
    pub fn exited_cleanly(&self) -> bool {
        self.exit_status.code() == Some(WORKLOAD_CLEAN_EXIT_CODE)
    }
}

/// Build a `Command` rooted at the current test binary that re-execs
/// itself with the workload env vars set. This is the cross-platform
/// substitute for `fork()`.
///
/// `arg` is the workload's parameter (typically a WAL directory path
/// the workload writes into). The path is serialized through the
/// env var so it must be valid UTF-8; non-UTF-8 paths panic via
/// `arg.to_string_lossy()` (acceptable on the macOS/Linux
/// development hosts; v1.1 may switch to a base64-encoded byte path
/// for max portability).
pub fn build_workload_command(workload: &str, arg: &Path) -> Command {
    let exe = std::env::current_exe().expect("current_exe() — test binary must be discoverable");
    let mut cmd = Command::new(exe);
    cmd.env(SUBPROCESS_WORKLOAD_ENV, workload);
    let mut arg_os = OsString::new();
    arg_os.push(arg);
    cmd.env(SUBPROCESS_WORKLOAD_ARG_ENV, arg_os);
    // Inherit stdio so the workload's eprintln logs surface in the
    // test output. v1.1 may add piped-stdout for structured ack
    // back to the parent.
    cmd.stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    cmd
}

/// Build a child command that runs exactly one non-ignored dispatcher
/// test, with no child libtest output inherited by the parent.
///
/// Re-executing an integration-test binary without a filter lets libtest
/// schedule unrelated tests alongside the dispatcher. Under a slow CI
/// runner, the parent can then SIGKILL the process while libtest is
/// emitting another test's status, making the intentional child exit
/// look like a top-level test failure. An exact, single-threaded filter
/// makes the dispatcher the child's only test, while null stdio keeps
/// its intentionally incomplete libtest line out of the parent's output.
///
/// The dispatcher test must remain non-ignored: the child does not pass
/// `--include-ignored`, and the workload must run before
/// [`maybe_dispatch_subprocess_workload`] exits the process.
pub fn build_workload_command_for_dispatcher(
    workload: &str,
    arg: &Path,
    dispatcher_test: &str,
) -> Command {
    let mut cmd = build_workload_command(workload, arg);
    cmd.arg("--exact")
        .arg(dispatcher_test)
        .arg("--test-threads=1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd
}

fn spawn_workload_command(
    mut cmd: Command,
    workload: &str,
    arg: &Path,
) -> std::io::Result<ChildHandle> {
    let child = cmd.spawn()?;
    let pid = child.id();
    Ok(ChildHandle {
        child,
        pid,
        spawned_at: Instant::now(),
        workload: workload.to_string(),
        arg: arg.to_string_lossy().to_string(),
    })
}

/// Spawn a child subprocess running `workload` with `arg`. The
/// caller chooses the crash window (delay until SIGKILL); the parent
/// must invoke [`ChildHandle::kill_with_sigkill`] (or
/// [`run_with_crash_window`] convenience) to actually fire SIGKILL.
pub fn fork_child_with_workload(workload: &str, arg: &Path) -> std::io::Result<ChildHandle> {
    let cmd = build_workload_command(workload, arg);
    spawn_workload_command(cmd, workload, arg)
}

/// Spawn a child whose libtest harness can execute only
/// `dispatcher_test`. This is the deterministic form for subprocess
/// chaos tests that provide a dedicated non-ignored router.
pub fn fork_child_with_workload_via_dispatcher(
    workload: &str,
    arg: &Path,
    dispatcher_test: &str,
) -> std::io::Result<ChildHandle> {
    let cmd = build_workload_command_for_dispatcher(workload, arg, dispatcher_test);
    spawn_workload_command(cmd, workload, arg)
}

fn run_child_with_crash_window(
    mut handle: ChildHandle,
    crash_after: Duration,
) -> std::io::Result<CrashRecord> {
    let pid = handle.pid;
    let workload_name = handle.workload.clone();
    let arg_str = handle.arg.clone();

    // Sleep until the crash window. We do not poll the child during
    // the window — the child is meant to be doing work, and polling
    // adds noise. If the child exits early, `wait()` below returns
    // immediately and the record reflects the clean exit.
    std::thread::sleep(crash_after);

    let kill_result = handle.kill_with_sigkill();
    let kill_succeeded = kill_result.is_ok();
    let elapsed_to_kill = handle.spawned_at.elapsed();

    let status = handle.child.wait()?;
    Ok(CrashRecord {
        workload: workload_name,
        arg: arg_str,
        pid,
        elapsed_to_kill,
        kill_succeeded,
        exit_status: status,
    })
}

/// Convenience: fork a child running `workload`, sleep for
/// `crash_after`, then SIGKILL it and reap. Returns a
/// [`CrashRecord`] capturing the firing.
///
/// If the child exits cleanly before `crash_after` elapses, the
/// SIGKILL syscall fails (process already gone); the record's
/// `kill_succeeded` is `false` and `exited_cleanly()` is `true`. The
/// caller should treat that case as "the crash window was longer
/// than the workload duration" and tune accordingly.
pub fn run_with_crash_window(
    workload: &str,
    arg: &Path,
    crash_after: Duration,
) -> std::io::Result<CrashRecord> {
    let handle = fork_child_with_workload(workload, arg)?;
    run_child_with_crash_window(handle, crash_after)
}

/// Deterministic counterpart to [`run_with_crash_window`]: the child
/// executes only `dispatcher_test`, so slow-runner scheduling cannot
/// start unrelated libtest bodies or bleed their status into the
/// parent's output while the intentional SIGKILL lands.
pub fn run_with_crash_window_via_dispatcher(
    workload: &str,
    arg: &Path,
    dispatcher_test: &str,
    crash_after: Duration,
) -> std::io::Result<CrashRecord> {
    let handle = fork_child_with_workload_via_dispatcher(workload, arg, dispatcher_test)?;
    run_child_with_crash_window(handle, crash_after)
}

/// Re-execute `workload` in the SAME process (no fork). Used by the
/// harness's "recover and verify" step: after SIGKILL'ing the
/// workload subprocess, the parent calls a recover-workload in-
/// process to assemble the post-recovery state.
///
/// This is NOT a subprocess; it's a direct call to a registered
/// workload. The naming mirrors the subprocess API for symmetry.
pub fn run_workload_in_process(workload: &str, arg: &Path) -> std::io::Result<i32> {
    let f = global().lookup(workload).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("workload `{workload}` not registered"),
        )
    })?;
    Ok(f(&arg.to_string_lossy()))
}

/// Pre-crash ledger: the workload writes one line per successful
/// commit to a file (or set of per-tenant files) the parent reads
/// after SIGKILL + restart. The oracle uses the ledger to enforce
/// the recovery contract.
///
/// Why a file (not a pipe / IPC)? Because SIGKILL'd processes don't
/// flush any buffered IPC; the ledger MUST be on disk + fsync'd at
/// every append (otherwise the parent reads a truncated ledger and
/// thinks pre-crash commits never happened, masking real bugs).
///
/// The ledger format is one CSV line per commit:
/// `tenant_raw,node_id_raw,label,a,b,tier`
/// where `tier` is `1` for `Strict` and `3` for `Periodic{rpo_ms}`.
///
/// The struct holds either:
///
/// - a single `Arc<Mutex<File>>` (legacy K-1a single-file mode — every
///   tenant's commits go to one shared CSV); or
/// - a `<workdir>/<tenant_raw>.csv` directory layout with one file per
///   tenant, lazily opened (K-1b canonical multi-tenant mode — issue
///   #214). Per-tenant files prevent fault-induced cross-pollution
///   masquerading as a recovery oracle pass: a torn trailing row in
///   tenant A's CSV cannot bleed into tenant B's CSV because they
///   are physically separate files.
///
/// Each `record` call fsyncs the targeted file — slow but correct.
/// K-1 smoke run rates (~100 commits/sec) make this acceptable;
/// K-3 multi-hour campaigns may batch fsyncs at the expense of
/// widening the recovery uncertainty window (still ADR-034-compliant
/// since the rpo_ms tolerance accommodates).
pub struct PreCrashLedger {
    layout: LedgerLayout,
}

/// Internal storage layout for [`PreCrashLedger`]. Variants chosen at
/// construction time and never re-key: a `SingleFile` ledger never
/// migrates to per-tenant and vice versa.
enum LedgerLayout {
    /// Legacy K-1a single-file mode. One CSV at `path`; every tenant's
    /// rows interleave into this file. K-1a smokes use this.
    SingleFile {
        file: Arc<Mutex<std::fs::File>>,
        path: PathBuf,
    },
    /// K-1b canonical per-tenant directory mode. Each tenant's rows go
    /// to `<workdir>/<tenant_raw>.csv`. Files are opened lazily on the
    /// first `record(tenant_raw, ...)` call for that tenant; the
    /// `HashMap` caches the open handle for subsequent appends.
    PerTenantDir {
        workdir: PathBuf,
        files: Mutex<HashMap<u64, Arc<Mutex<std::fs::File>>>>,
    },
}

impl PreCrashLedger {
    /// Legacy K-1a constructor: single CSV at `path`. The file is
    /// truncated + opened for read/write so a fresh test always sees
    /// an empty ledger.
    pub fn create(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let p = path.as_ref().to_path_buf();
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&p)?;
        Ok(Self {
            layout: LedgerLayout::SingleFile {
                file: Arc::new(Mutex::new(file)),
                path: p,
            },
        })
    }

    /// Legacy K-1a constructor: open an existing single CSV at `path`
    /// for append. Used post-restart to continue logging.
    pub fn open_existing(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let p = path.as_ref().to_path_buf();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .append(true)
            .open(&p)?;
        Ok(Self {
            layout: LedgerLayout::SingleFile {
                file: Arc::new(Mutex::new(file)),
                path: p,
            },
        })
    }

    /// K-1b canonical: open a per-tenant directory ledger at `workdir`.
    /// Each tenant's rows land in `<workdir>/<tenant_raw>.csv`; files
    /// are opened lazily on first `record`. The directory is created
    /// if absent. Stale CSVs in `workdir` (from prior runs) are NOT
    /// truncated — the caller is responsible for using a fresh
    /// directory (e.g., a `tempfile::TempDir` per test).
    ///
    /// Per issue #214 + the K-1b spec, per-tenant separation prevents
    /// fault-induced cross-pollution: a torn trailing row in tenant A's
    /// CSV cannot bleed into tenant B's recovery oracle input because
    /// the two are physically separate files.
    pub fn create_in_dir(workdir: impl AsRef<Path>) -> std::io::Result<Self> {
        let dir = workdir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            layout: LedgerLayout::PerTenantDir {
                workdir: dir,
                files: Mutex::new(HashMap::new()),
            },
        })
    }

    /// Legacy single-file path accessor. Returns `Some(path)` when the
    /// ledger is in `SingleFile` mode, `None` in `PerTenantDir` mode.
    pub fn path(&self) -> Option<&Path> {
        match &self.layout {
            LedgerLayout::SingleFile { path, .. } => Some(path),
            LedgerLayout::PerTenantDir { .. } => None,
        }
    }

    /// Per-tenant workdir accessor. Returns `Some(workdir)` when the
    /// ledger is in `PerTenantDir` mode, `None` in `SingleFile` mode.
    pub fn workdir(&self) -> Option<&Path> {
        match &self.layout {
            LedgerLayout::SingleFile { .. } => None,
            LedgerLayout::PerTenantDir { workdir, .. } => Some(workdir),
        }
    }

    /// Append one commit to the ledger and fsync. `tier` is `1` for
    /// Strict, `3` for Periodic.
    ///
    /// In `SingleFile` mode (K-1a) every tenant's row goes to the
    /// shared file; in `PerTenantDir` mode (K-1b) the row is routed
    /// to `<workdir>/<tenant_raw>.csv` (lazily opened on first use
    /// for that tenant).
    ///
    /// 7-arg surface mirrors the [`LedgerRecord`] field shape; bundling
    /// into a struct adds ceremony at every call site without clarity
    /// gain (the workload calls this once per commit on the hot path).
    /// Same allow as `crate::crud::CrudOracle::record` and the
    /// snapshot-flush helpers in `vector_store::snapshot`.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        tenant_raw: u64,
        node_id_raw: u64,
        label: u32,
        a: u32,
        b: u32,
        tier: u8,
    ) -> std::io::Result<()> {
        use std::io::Write as _;
        let line = format!("{tenant_raw},{node_id_raw},{label},{a},{b},{tier}\n");
        let file_arc = self.acquire_tenant_file(tenant_raw)?;
        let mut g = file_arc.lock().expect("ledger file poisoned");
        g.write_all(line.as_bytes())?;
        g.sync_data()?;
        Ok(())
    }

    /// Resolve the file handle for `tenant_raw`. In `SingleFile` mode
    /// the same handle is returned regardless of tenant. In
    /// `PerTenantDir` mode the handle for `tenant_raw` is opened
    /// lazily on first call and cached; subsequent calls hit the
    /// cache.
    fn acquire_tenant_file(&self, tenant_raw: u64) -> std::io::Result<Arc<Mutex<std::fs::File>>> {
        match &self.layout {
            LedgerLayout::SingleFile { file, .. } => Ok(Arc::clone(file)),
            LedgerLayout::PerTenantDir { workdir, files } => {
                let mut g = files.lock().expect("ledger files map poisoned");
                if let Some(existing) = g.get(&tenant_raw) {
                    return Ok(Arc::clone(existing));
                }
                let path = workdir.join(format!("{tenant_raw}.csv"));
                let f = std::fs::OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .read(true)
                    .open(&path)?;
                let arc = Arc::new(Mutex::new(f));
                g.insert(tenant_raw, Arc::clone(&arc));
                Ok(arc)
            }
        }
    }

    /// Read every committed record. Used by the parent post-restart
    /// to assemble the pre-crash ground truth.
    ///
    /// ## Truncated trailing row tolerance (codex B-2)
    ///
    /// `record()` writes a single CSV line + `sync_data()`. SIGKILL
    /// during a partial `write_all` (or before `sync_data` returns)
    /// can leave a partial trailing row. Pre-codex this rejected ANY
    /// `parts.len() != 6` row including a torn trailing record —
    /// bricking the entire ledger replay on the canonical Jepsen-
    /// style harness failure mode the ledger is meant to record.
    ///
    /// Post-fix:
    /// - A malformed line that is **NOT** the last line in the file
    ///   is real corruption and returns `Err(InvalidData)`.
    /// - A malformed line that **IS** the last line in the file is
    ///   treated as a torn-write trailing row: `tracing::warn!` +
    ///   skip + return the prefix successfully.
    ///
    /// This mirrors the LogSegment recovery primitive in Kafka /
    /// etcd's `wal::decoder::nextValid`. The K-2/K-3 path may
    /// upgrade to length-prefixed binary records + CRC (codex Option
    /// (c)); Option (a) is sufficient for K-1a.
    pub fn read_all(path: impl AsRef<Path>) -> std::io::Result<Vec<LedgerRecord>> {
        read_csv_with_torn_tail_tolerance(path.as_ref())
    }

    /// K-1b: read a single tenant's records from a per-tenant
    /// directory. Returns `Ok(empty Vec)` if the tenant has no CSV
    /// (i.e., the tenant never committed during the workload). This
    /// makes "did tenant T commit anything?" expressible without an
    /// extra `exists()` probe.
    ///
    /// Torn-tail tolerance is preserved per-file (the file is
    /// processed by the same private helper as `read_all`).
    pub fn read_for(
        workdir: impl AsRef<Path>,
        tenant_raw: u64,
    ) -> std::io::Result<Vec<LedgerRecord>> {
        let path = workdir.as_ref().join(format!("{tenant_raw}.csv"));
        match std::fs::metadata(&path) {
            Ok(_) => read_csv_with_torn_tail_tolerance(&path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    /// K-1b: read every tenant's CSV in a per-tenant directory and
    /// return them keyed by tenant raw id. Files whose names do not
    /// parse as `<u64>.csv` are ignored (defensive against e.g. a
    /// stray `.DS_Store` on macOS or hidden temporary files).
    ///
    /// Torn-tail tolerance is preserved per-file (the same private
    /// helper as `read_all` is used).
    pub fn read_all_per_tenant(
        workdir: impl AsRef<Path>,
    ) -> std::io::Result<HashMap<u64, Vec<LedgerRecord>>> {
        let dir_ref = workdir.as_ref();
        let mut out: HashMap<u64, Vec<LedgerRecord>> = HashMap::new();
        let read_dir = match std::fs::read_dir(dir_ref) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(e),
        };
        for entry in read_dir {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
                continue;
            };
            if ext != "csv" {
                continue;
            }
            let Ok(tenant_raw) = stem.parse::<u64>() else {
                continue;
            };
            let rows = read_csv_with_torn_tail_tolerance(&path)?;
            out.insert(tenant_raw, rows);
        }
        Ok(out)
    }
}

/// Parse a single CSV file into [`LedgerRecord`]s with torn-trailing-
/// row tolerance. Shared between [`PreCrashLedger::read_all`] (legacy
/// single-file mode) + [`PreCrashLedger::read_for`] +
/// [`PreCrashLedger::read_all_per_tenant`] (K-1b per-tenant mode) so
/// both surfaces share the codex B-2 trailing-row contract verbatim.
///
/// ## Truncated trailing row tolerance (codex B-2)
///
/// `record()` writes a single CSV line + `sync_data()`. SIGKILL during
/// a partial `write_all` (or before `sync_data` returns) can leave a
/// partial trailing row.
///
/// - A malformed line that is **NOT** the last line in the file is
///   real corruption and returns `Err(InvalidData)`.
/// - A malformed line that **IS** the last line in the file is treated
///   as a torn-write trailing row: `tracing::warn!` + skip + return
///   the prefix successfully.
fn read_csv_with_torn_tail_tolerance(path_ref: &Path) -> std::io::Result<Vec<LedgerRecord>> {
    let bytes = std::fs::read(path_ref)?;
    let s = std::str::from_utf8(&bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    // Capture every non-empty line (preserving order) so we can
    // identify "is this the last line?" before deciding whether
    // a malformed row is a torn trailing record (tolerated) or
    // mid-file corruption (rejected). `str::lines()` yields each
    // line exactly once with the trailing newline stripped; an
    // unterminated trailing line still appears as the last
    // element, which is the torn-write case we want to detect.
    let lines: Vec<&str> = s.lines().collect();
    let total_lines = lines.len();
    let mut out = Vec::with_capacity(total_lines);

    for (idx, line) in lines.iter().enumerate() {
        let is_last = idx + 1 == total_lines;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 6 {
            if is_last {
                tracing::warn!(
                    path = %path_ref.display(),
                    line_number = idx + 1,
                    line_content = %line,
                    reason = "truncated trailing row",
                    "PreCrashLedger: skipping torn trailing record \
                     (SIGKILL mid-write); returning prefix"
                );
                break;
            }
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "ledger line {} malformed (mid-file corruption, not torn \
                     trailing row): `{line}`",
                    idx + 1
                ),
            ));
        }
        // Field-parse failures are mid-row corruption: same
        // is_last/torn-tolerance treatment so a SIGKILL-during-
        // numeric-write also tolerates the trailing row instead
        // of bricking replay.
        let parsed = (|| -> std::io::Result<LedgerRecord> {
            Ok(LedgerRecord {
                tenant_raw: parts[0].parse().map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e}"))
                })?,
                node_id_raw: parts[1].parse().map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e}"))
                })?,
                label: parts[2].parse().map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e}"))
                })?,
                a: parts[3].parse().map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e}"))
                })?,
                b: parts[4].parse().map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e}"))
                })?,
                tier: parts[5].parse().map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e}"))
                })?,
            })
        })();
        match parsed {
            Ok(rec) => out.push(rec),
            Err(e) => {
                if is_last {
                    tracing::warn!(
                        path = %path_ref.display(),
                        line_number = idx + 1,
                        line_content = %line,
                        reason = "trailing row field parse failure",
                        error = %e,
                        "PreCrashLedger: skipping torn trailing record \
                         (SIGKILL mid-numeric-write); returning prefix"
                    );
                    break;
                }
                return Err(e);
            }
        }
    }
    Ok(out)
}

/// One row in the pre-crash ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerRecord {
    pub tenant_raw: u64,
    pub node_id_raw: u64,
    pub label: u32,
    pub a: u32,
    pub b: u32,
    /// 1 = Strict; 3 = Periodic.
    pub tier: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn registry_round_trip() {
        fn workload_a(_arg: &str) -> i32 {
            WORKLOAD_CLEAN_EXIT_CODE
        }
        SubprocessWorkloadRegistry::register("test-registry-round-trip", workload_a);
        assert!(global().lookup("test-registry-round-trip").is_some());
        assert!(global().lookup("nonexistent").is_none());
    }

    #[test]
    fn deterministic_dispatch_command_filters_exactly_one_router() {
        let tmp = TempDir::new().unwrap();
        let cmd = build_workload_command_for_dispatcher(
            "test-deterministic-dispatch",
            tmp.path(),
            "aaaa_subprocess_dispatcher_router",
        );
        let args: Vec<_> = cmd.get_args().map(|arg| arg.to_owned()).collect();
        assert_eq!(
            args,
            [
                OsString::from("--exact"),
                OsString::from("aaaa_subprocess_dispatcher_router"),
                OsString::from("--test-threads=1"),
            ],
            "child libtest must run only the dedicated non-ignored router"
        );
    }

    #[test]
    fn ledger_round_trip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("ledger.csv");
        let ledger = PreCrashLedger::create(&path).unwrap();
        ledger.record(0, 1, 100, 11, 22, 1).unwrap();
        ledger.record(1001, 2, 200, 33, 44, 3).unwrap();
        drop(ledger);

        let read = PreCrashLedger::read_all(&path).unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].tenant_raw, 0);
        assert_eq!(read[0].node_id_raw, 1);
        assert_eq!(read[0].label, 100);
        assert_eq!(read[0].a, 11);
        assert_eq!(read[0].b, 22);
        assert_eq!(read[0].tier, 1);
        assert_eq!(read[1].tenant_raw, 1001);
        assert_eq!(read[1].tier, 3);
    }

    #[test]
    fn ledger_rejects_malformed_lines() {
        // Pre-codex B-2: ANY malformed line bricked the ledger. Post-
        // fix: a malformed line that is the SOLE line in the file is
        // a torn trailing record and is tolerated (returns Ok(empty)
        // with a tracing::warn). To pin "real corruption fails," put
        // a malformed line BEFORE a valid trailing line.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bad.csv");
        std::fs::write(&path, "1,2,3\n0,1,100,11,22,1\n").unwrap();
        let err = PreCrashLedger::read_all(&path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn ledger_tolerates_truncated_trailing_row() {
        // Codex B-2 pin: SIGKILL during write_all (or before
        // sync_data) leaves a partial trailing CSV row. read_all must
        // return the prefix successfully (with a tracing::warn) — NOT
        // brick the entire ledger replay.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("ledger.csv");
        let ledger = PreCrashLedger::create(&path).unwrap();
        // Write 5 valid rows.
        for i in 0..5 {
            ledger.record(0, i, 100, 11, 22, 1).unwrap();
        }
        drop(ledger);
        // Append a torn trailing row (truncated mid-CSV; missing
        // 3 of 6 fields).
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(b"99,42,").unwrap();
        // No trailing newline — this is what SIGKILL-during-write
        // looks like on disk: partial bytes, no terminating \n.
        drop(f);

        let read = PreCrashLedger::read_all(&path).unwrap();
        assert_eq!(
            read.len(),
            5,
            "expected 5 prefix rows; got {} (torn trailing row should NOT brick)",
            read.len()
        );
        for (i, rec) in read.iter().enumerate() {
            assert_eq!(rec.tenant_raw, 0);
            assert_eq!(rec.node_id_raw, i as u64);
            assert_eq!(rec.label, 100);
            assert_eq!(rec.a, 11);
            assert_eq!(rec.b, 22);
            assert_eq!(rec.tier, 1);
        }
    }

    #[test]
    fn ledger_rejects_malformed_middle_row() {
        // Codex B-2 pin: a malformed row in the MIDDLE of the ledger
        // is real corruption — NOT a SIGKILL torn trailing record.
        // Bracket pattern: 3 valid rows + 1 malformed row + 2 valid
        // rows. read_all must error (InvalidData) on the middle
        // corruption, not silently skip it.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("ledger.csv");
        let mut buf = String::new();
        for i in 0..3 {
            buf.push_str(&format!("{},{},100,11,22,1\n", 0, i));
        }
        buf.push_str("garbage,row,here\n"); // malformed middle row (3 parts)
        for i in 3..5 {
            buf.push_str(&format!("{},{},100,11,22,1\n", 0, i));
        }
        std::fs::write(&path, buf).unwrap();

        let err = PreCrashLedger::read_all(&path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let msg = format!("{err}");
        assert!(
            msg.contains("mid-file corruption"),
            "error message must distinguish mid-file corruption from torn trailing row; \
             got: {msg}"
        );
    }

    // ── K-1b per-tenant directory mode (issue #214) ──────────────────

    #[test]
    fn ledger_per_tenant_dir_round_trip_writes_separate_files() {
        // K-1b pin: PerTenantDir mode opens one CSV per tenant on first
        // record(); subsequent records for the same tenant append.
        // Reading back via read_for(tenant) returns ONLY that tenant's
        // rows; reading via read_all_per_tenant returns the
        // tenant-keyed map.
        let tmp = TempDir::new().unwrap();
        let workdir = tmp.path().join("pre_crash_ledger");
        let ledger = PreCrashLedger::create_in_dir(&workdir).unwrap();

        // Two tenants, three commits each, interleaved.
        for i in 0..3u64 {
            ledger.record(1, i, 100 + i as u32, 11, 22, 1).unwrap();
            ledger.record(1001, i, 200 + i as u32, 33, 44, 3).unwrap();
        }
        drop(ledger);

        // Per-tenant files exist with the expected names.
        assert!(workdir.join("1.csv").exists(), "tenant 1 CSV missing");
        assert!(workdir.join("1001.csv").exists(), "tenant 1001 CSV missing");
        assert!(
            !workdir.join("999.csv").exists(),
            "no commits for tenant 999 — file must NOT have been created"
        );

        // read_for surfaces only that tenant's rows.
        let t1 = PreCrashLedger::read_for(&workdir, 1).unwrap();
        assert_eq!(t1.len(), 3);
        assert!(
            t1.iter().all(|r| r.tenant_raw == 1),
            "read_for(1) leaked tenant 1001's rows: {t1:?}"
        );
        let t1001 = PreCrashLedger::read_for(&workdir, 1001).unwrap();
        assert_eq!(t1001.len(), 3);
        assert!(
            t1001.iter().all(|r| r.tenant_raw == 1001),
            "read_for(1001) leaked tenant 1's rows: {t1001:?}"
        );

        // Tier round-trips: tenant 1 wrote tier=1, tenant 1001 wrote
        // tier=3.
        assert!(t1.iter().all(|r| r.tier == 1));
        assert!(t1001.iter().all(|r| r.tier == 3));

        // read_all_per_tenant returns the keyed map.
        let all = PreCrashLedger::read_all_per_tenant(&workdir).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all.get(&1).map(|v| v.len()), Some(3));
        assert_eq!(all.get(&1001).map(|v| v.len()), Some(3));
    }

    #[test]
    fn ledger_per_tenant_dir_read_for_absent_tenant_is_empty_ok() {
        // K-1b: read_for() on a tenant that never recorded must return
        // Ok(empty Vec) — NOT an io::Error::NotFound. This makes "did
        // tenant T commit anything?" expressible without an extra
        // exists() probe at the call site.
        let tmp = TempDir::new().unwrap();
        let workdir = tmp.path().join("pre_crash_ledger");
        let ledger = PreCrashLedger::create_in_dir(&workdir).unwrap();
        ledger.record(1, 0, 100, 11, 22, 1).unwrap();
        drop(ledger);

        let absent = PreCrashLedger::read_for(&workdir, 9999).unwrap();
        assert!(
            absent.is_empty(),
            "read_for(absent_tenant) must return empty Vec, not error"
        );
    }

    #[test]
    fn ledger_per_tenant_dir_torn_tail_isolated_per_tenant() {
        // K-1b cross-pollution pin: a torn trailing row in tenant A's
        // CSV must NOT affect tenant B's CSV. Per-tenant separation is
        // the structural fix — physically separate files cannot bleed
        // into each other under SIGKILL-during-write.
        let tmp = TempDir::new().unwrap();
        let workdir = tmp.path().join("pre_crash_ledger");
        let ledger = PreCrashLedger::create_in_dir(&workdir).unwrap();

        for i in 0..3u64 {
            ledger.record(1, i, 100 + i as u32, 11, 22, 1).unwrap();
            ledger.record(2, i, 200 + i as u32, 33, 44, 1).unwrap();
        }
        drop(ledger);

        // Append a torn trailing row to ONLY tenant 1's CSV.
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(workdir.join("1.csv"))
            .unwrap();
        f.write_all(b"99,42,").unwrap();
        drop(f);

        // Tenant 1 reads back its prefix (3 rows; torn tail tolerated).
        let t1 = PreCrashLedger::read_for(&workdir, 1).unwrap();
        assert_eq!(
            t1.len(),
            3,
            "tenant 1 should read 3 prefix rows past torn tail; got {}",
            t1.len()
        );
        // Tenant 2 reads back ALL its rows — torn tail in tenant 1
        // CANNOT affect tenant 2's file.
        let t2 = PreCrashLedger::read_for(&workdir, 2).unwrap();
        assert_eq!(
            t2.len(),
            3,
            "tenant 2's CSV is physically separate; got {}",
            t2.len()
        );
    }

    #[test]
    fn ledger_per_tenant_dir_ignores_non_csv_files() {
        // Defensive: read_all_per_tenant ignores non-`<u64>.csv` files
        // (e.g., a stray .DS_Store on macOS, or a hidden TempDir file).
        let tmp = TempDir::new().unwrap();
        let workdir = tmp.path().join("pre_crash_ledger");
        let ledger = PreCrashLedger::create_in_dir(&workdir).unwrap();
        ledger.record(1, 0, 100, 11, 22, 1).unwrap();
        drop(ledger);

        // Plant decoys: non-CSV file + non-numeric stem.
        std::fs::write(workdir.join(".DS_Store"), b"junk").unwrap();
        std::fs::write(workdir.join("README.csv"), b"junk\n").unwrap();
        std::fs::write(workdir.join("not-a-tenant.csv"), b"x,y,z\n").unwrap();

        let all = PreCrashLedger::read_all_per_tenant(&workdir).unwrap();
        assert_eq!(
            all.len(),
            1,
            "decoy files must be ignored; only tenant 1 should appear"
        );
        assert_eq!(all.get(&1).map(|v| v.len()), Some(1));
    }

    #[test]
    fn ledger_path_accessor_returns_some_only_in_single_file_mode() {
        // K-1b backward-compat pin: path() returns Some for legacy
        // SingleFile mode, None for PerTenantDir mode. Mirror for
        // workdir().
        let tmp = TempDir::new().unwrap();
        let single = PreCrashLedger::create(tmp.path().join("ledger.csv")).unwrap();
        assert!(single.path().is_some());
        assert!(single.workdir().is_none());

        let dir = PreCrashLedger::create_in_dir(tmp.path().join("dir")).unwrap();
        assert!(dir.path().is_none());
        assert!(dir.workdir().is_some());
    }
}
