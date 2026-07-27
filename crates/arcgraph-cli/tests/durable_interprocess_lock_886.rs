//! #886 — a durable `--data` store is single-process: an EXCLUSIVE inter-process
//! advisory lock is taken at durable bootstrap (before `pages.db` / the WAL are
//! opened). This file is the fault-injection suite for that fix (data-loss /
//! durability surface → ≥1 test per failure mode;
//! `feedback_load_bearing_pr_requires_fault_injection_tests`).
//!
//! Background: before this fix, two `arcgraph serve --data <SAMEDIR>` processes
//! both opened the same store, interleaved their WAL appends, and bricked it on
//! the next restart (`WalCorruption … crc mismatch` at `Lsn(0)` — unrecoverable,
//! losing acknowledged Strict-tier commits, violating the ADR-183 guarantee).
//! It is reachable through the documented CLI (`--http` `conflicts_with`
//! `--bolt`, so serving both protocols on one durable store means two processes).
//!
//! Tests (oracles):
//! 1. [`second_durable_opener_in_process_is_refused_886`] (REQUIRED, RED-on-
//!    revert) — a second durable bootstrap on the same dir returns `Err` (the
//!    lock-held error), and the FIRST store is STILL fully usable (write+read a
//!    node through it). With the lock removed the second bootstrap SUCCEEDS
//!    (today's #886 bug) → the `expect_err` panics.
//! 2. [`subprocess::two_serve_processes_on_same_dir_second_refused_886`] — the
//!    STRONGEST oracle (matches the issue repro exactly): a REAL second
//!    `arcgraph serve` process on the same dir exits non-zero with the lock
//!    error on stderr. RED-on-revert: without the lock it binds its listener and
//!    serves forever → it never exits → the deadline fails the test.
//! 3. [`subprocess::crash_releases_data_dir_lock_886`] (REQUIRED) — `kill -9` the
//!    lock holder, then a fresh open on the same dir SUCCEEDS: the OS released
//!    the dead process's advisory lock, so a crashed process does NOT brick the
//!    dir (no stale-lock cleanup needed — no new failure mode introduced).
//! 4. [`durability_guard_field_order_pins_writer_before_lock_drop_886`] —
//!    source-order pin for the Rust field-declaration drop-order invariant:
//!    `writer` MUST drop before `lock`, otherwise a second opener can acquire the
//!    dir while the first writer is still flushing (#886 shutdown window).
//! 5. [`two_in_memory_bootstraps_do_not_conflict_886`] — `--in-memory` has no
//!    shared on-disk state → no lock → two in-memory bootstraps coexist.
//!
//! These drive the production `arcgraph_cli::bootstrap::bootstrap_storage_backend`
//! surface + the real `arcgraph` binary — no mocks where a real path is feasible.

use std::path::Path;
use std::sync::Arc;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::data_lock::DataDirLock;
use arcgraph_core::{LabelId, PartitionId, TenantId};
use arcgraph_mcp::storage::StorageBackend;
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, read_node_with_store};
use tempfile::TempDir;

/// A durable bootstrap mode rooted at `data_dir`.
fn durable(data_dir: &Path) -> BootstrapMode {
    BootstrapMode::Durable {
        data_dir: data_dir.to_path_buf(),
    }
}

/// The shared per-tenant `CrudStore` via the production router surface.
fn crud_for(backend: &StorageBackend, tenant: TenantId) -> Arc<CrudStore> {
    backend
        .router()
        .route(tenant, PartitionId::ZERO)
        .expect("route tenant")
        .crud()
        .clone()
}

// ─────────────────────────────────────────────────────────────────────
// 1. Second durable opener (same process) is refused; the first store
//    stays fully usable. REQUIRED, RED-on-revert.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn second_durable_opener_in_process_is_refused_886() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // First opener holds the durability guard (and the data-dir lock). flock
    // treats two open descriptions as independent even in one process, so this
    // exercises the SAME exclusion a second OS process would hit (the
    // subprocess test below is the cross-process confirmation).
    let (backend1, guard1) =
        bootstrap_storage_backend(&durable(&data_dir)).expect("first durable bootstrap");
    assert!(
        guard1.is_durable(),
        "first opener owns the durable substrate"
    );
    assert_eq!(
        guard1.data_dir_lock_path(),
        Some(data_dir.join("LOCK").as_path()),
        "the durable guard must report the inter-process lock it holds (#886)",
    );

    // Second opener on the SAME dir is REFUSED with the lock error, BEFORE it
    // opens pages.db / replays the WAL. RED-on-revert: without the §1 lock the
    // second bootstrap SUCCEEDS (today's #886 bug — two writers onto one store,
    // WAL corruption) and this `expect_err` panics.
    // `.err()` (not `expect_err`) so we don't require `Debug` on the Ok tuple
    // (which holds the non-`Debug` `DurabilityGuard`).
    let err = bootstrap_storage_backend(&durable(&data_dir))
        .err()
        .expect("second durable bootstrap on the same dir MUST be refused (#886)");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("already in use"),
        "refusal must say the dir is already in use; got: {msg}"
    );
    assert!(
        msg.contains(&data_dir.display().to_string()),
        "refusal must name the data dir; got: {msg}"
    );

    // The FIRST store is STILL fully usable after the second was refused: write
    // + read a node through it (the refusal did not disturb the holder — the 2nd
    // attempt bailed before touching any shared state).
    let crud = crud_for(&backend1, TenantId::DEFAULT);
    let mut tx = backend1.txn_manager().begin(TenantId::DEFAULT);
    let id = create_node(
        &crud,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::InlineU32Pair(11, 22),
    )
    .expect("create_node on the holder");
    commit(tx, &crud).expect("commit on the holder (still usable after 2nd opener refused)");

    let tx = backend1.txn_manager().begin(TenantId::DEFAULT);
    let rec = read_node_with_store(&crud, &tx, id)
        .expect("read node")
        .expect("the holder's store is still fully usable after the 2nd opener was refused");
    assert_eq!(rec.inline_u32a, 11);
    assert_eq!(rec.inline_u32b, 22);
}

#[test]
fn durability_guard_field_order_pins_writer_before_lock_drop_886() {
    // #929: this is intentionally a source-order oracle. Rust drops struct
    // fields in declaration order, so a reorder of the actual DurabilityGuard
    // fields is the behavioral regression. RED-on-revert/reorder: placing
    // `lock` before `writer` makes `lock_pos < writer_pos` and fails here,
    // catching the #886 shutdown-window hazard before a second opener can race a
    // still-flushing WAL writer.
    let source = include_str!("../src/bootstrap.rs");
    let start = source
        .find("pub struct DurabilityGuard")
        .expect("DurabilityGuard declaration present");
    let rest = &source[start..];
    let end = rest
        .find("impl DurabilityGuard")
        .expect("DurabilityGuard impl follows declaration");
    let declaration = &rest[..end];
    let writer_pos = declaration
        .find("writer: Option<WalWriter>")
        .expect("DurabilityGuard.writer field present");
    let lock_pos = declaration
        .find("lock: Option<DataDirLock>")
        .expect("DurabilityGuard.lock field present");

    assert!(
        writer_pos < lock_pos,
        "DROP ORDER LOAD-BEARING (#886): DurabilityGuard.writer must be declared \
         before lock so the WAL drains/fsyncs/joins before the data-dir lock releases"
    );
    assert!(
        declaration.contains("DROP ORDER LOAD-BEARING — writer MUST precede lock; do not reorder"),
        "DurabilityGuard fields must carry the explicit #886 drop-order warning"
    );
}

// ─────────────────────────────────────────────────────────────────────
// 5. --in-memory never locks: two in-memory bootstraps coexist.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn two_in_memory_bootstraps_do_not_conflict_886() {
    // No shared on-disk state → no lock → no false "in use" refusal between two
    // concurrent in-memory bootstraps.
    let (_b1, g1) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("first in-memory bootstrap");
    let (_b2, g2) = bootstrap_storage_backend(&BootstrapMode::InMemory)
        .expect("second in-memory bootstrap must NOT be refused (no on-disk lock)");
    assert!(!g1.is_durable());
    assert!(!g2.is_durable());
    assert_eq!(
        g1.data_dir_lock_path(),
        None,
        "--in-memory mode holds NO data-dir lock"
    );
    assert_eq!(g2.data_dir_lock_path(), None);
}

// ─────────────────────────────────────────────────────────────────────
// Real-process oracles (unix): a SECOND `arcgraph serve` is refused, and a
// crashed holder's lock is auto-released by the OS. Gated to unix — the SIGKILL
// + advisory-lock-on-death semantics are the unix `flock` path; the windows
// `share_mode(0)` path is best-effort and not exercised on the CI platform (see
// PR Risks). The cross-platform lock LOGIC is covered by the in-process tests
// above + the `data_lock` unit tests.
// ─────────────────────────────────────────────────────────────────────

#[cfg(unix)]
mod subprocess {
    use super::*;
    use std::io::Read;
    use std::net::TcpStream;
    use std::process::{Child, Command, ExitStatus, Stdio};
    use std::time::{Duration, Instant};

    /// Budget for a `serve` child to bootstrap + bind its listener.
    const STARTUP_BUDGET: Duration = Duration::from_secs(30);

    /// Grab a free loopback TCP port (bind `:0`, read it, drop). Standard
    /// subprocess-port idiom; a tiny TOCTOU window is acceptable for a hermetic
    /// test.
    fn free_port() -> u16 {
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind :0 for free port");
        let p = l.local_addr().expect("local_addr").port();
        drop(l);
        p
    }

    /// A spawned `arcgraph serve --data <dir> --bolt 127.0.0.1:<port>` child
    /// (durable). stderr is piped (read after exit). Admin/metrics/audit are
    /// disabled so the data-dir LOCK is the ONLY shared contention between two
    /// `serve` processes — no port / CWD-file collision confounds the oracle.
    struct Serve {
        child: Child,
    }

    impl Serve {
        fn spawn_durable_bolt(bin: &str, data_dir: &Path, port: u16) -> Self {
            let child = Command::new(bin)
                .args([
                    "serve",
                    "--data",
                    data_dir.to_str().expect("utf-8 data dir"),
                    "--bolt",
                    &format!("127.0.0.1:{port}"),
                    "--admin-http",
                    "",
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .expect("spawn arcgraph serve --data --bolt");
            Serve { child }
        }

        /// Poll until the bolt port accepts ⟹ bootstrap done ⟹ lock held. The
        /// lock is taken DURING bootstrap, before the listener binds, so an
        /// accepting port proves the lock is held. Panics (with captured
        /// stderr) on early child exit / budget elapsed.
        fn await_bolt_up(&mut self, port: u16) {
            let deadline = Instant::now() + STARTUP_BUDGET;
            loop {
                if let Some(status) = self.child.try_wait().expect("try_wait") {
                    let se = self.read_stderr();
                    panic!(
                        "serve exited early (status {status}) before binding bolt :{port}\nstderr:\n{se}"
                    );
                }
                if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                    return;
                }
                if Instant::now() >= deadline {
                    panic!("serve did not bind bolt :{port} within {STARTUP_BUDGET:?}");
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }

        /// Wait up to `budget` for the child to exit on its own. `Some(status)`
        /// if it exited, `None` if still running at the deadline.
        fn wait_for_exit(&mut self, budget: Duration) -> Option<ExitStatus> {
            let deadline = Instant::now() + budget;
            loop {
                if let Some(status) = self.child.try_wait().expect("try_wait") {
                    return Some(status);
                }
                if Instant::now() >= deadline {
                    return None;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }

        /// SIGKILL the child and reap it. After this returns the kernel has torn
        /// the process down (all fds closed → the advisory `flock` released), so
        /// the dir is immediately re-openable. (std `Child::kill` is SIGKILL on
        /// unix.)
        fn sigkill_and_reap(&mut self) {
            self.child.kill().expect("SIGKILL serve holder");
            self.child.wait().expect("reap killed holder");
        }

        /// Read the child's stderr to EOF. MUST be called only after the child
        /// has exited / been killed (else it blocks until EOF).
        fn read_stderr(&mut self) -> String {
            let mut s = String::new();
            if let Some(mut e) = self.child.stderr.take() {
                let _ = e.read_to_string(&mut s);
            }
            s
        }

        /// Best-effort cleanup.
        fn kill(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    impl Drop for Serve {
        fn drop(&mut self) {
            // Never leak a serve process out of a test.
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    // ── 2. Two real `serve` processes; the second is refused. ──────────

    #[test]
    fn two_serve_processes_on_same_dir_second_refused_886() {
        let bin = env!("CARGO_BIN_EXE_arcgraph");
        let tmp = TempDir::new().expect("tempdir");
        let data_dir = tmp.path().join("db");

        // Process A: durable serve; acquires the lock during bootstrap, binds.
        let port_a = free_port();
        let mut a = Serve::spawn_durable_bolt(bin, &data_dir, port_a);
        a.await_bolt_up(port_a); // A now holds the data-dir lock.

        // Process B: a SECOND `arcgraph serve` on the SAME dir (different bolt
        // port → the data-dir lock is the only shared contention). It MUST fail
        // fast at durable bootstrap (lock held) and exit non-zero, NEVER binding
        // its listener. RED-on-revert: without the lock B bootstraps a 2nd
        // writer, binds :port_b, and serves forever → it never exits → the
        // deadline below yields `None` → the test fails.
        let port_b = free_port();
        let mut b = Serve::spawn_durable_bolt(bin, &data_dir, port_b);
        let b_status = b.wait_for_exit(STARTUP_BUDGET);
        if b_status.is_none() {
            // #930: when the lock regresses, B keeps serving forever. Kill it
            // before reading stderr, because read_stderr() waits for EOF.
            b.kill();
        }
        let b_stderr = b.read_stderr();

        // Clean up A regardless of B's outcome.
        a.kill();

        let status = b_status.unwrap_or_else(|| {
            panic!(
                "2nd `arcgraph serve` on the same --data dir did NOT exit within {STARTUP_BUDGET:?} \
                 — it bound a listener instead of being refused (RED-on-revert: the #886 lock is \
                 missing).\nstderr:\n{b_stderr}"
            )
        });
        assert!(
            !status.success(),
            "the 2nd serve on the same --data dir MUST exit non-zero (lock held); got {status:?}\nstderr:\n{b_stderr}"
        );
        assert!(
            b_stderr.contains("already in use"),
            "the 2nd serve must explain the dir is already in use; stderr:\n{b_stderr}"
        );
        assert!(
            b_stderr.contains(&data_dir.display().to_string()),
            "the 2nd serve must name the data dir; stderr:\n{b_stderr}"
        );
    }

    // ── 3. A crashed (kill -9) holder's lock is auto-released. ──────────

    #[test]
    fn crash_releases_data_dir_lock_886() {
        let bin = env!("CARGO_BIN_EXE_arcgraph");
        let tmp = TempDir::new().expect("tempdir");
        let data_dir = tmp.path().join("db");

        // Holder process: durable serve acquires the lock during bootstrap.
        let port = free_port();
        let mut holder = Serve::spawn_durable_bolt(bin, &data_dir, port);
        holder.await_bolt_up(port); // lock is held now.

        // SIGKILL the holder — an ungraceful crash: no `Drop`, no graceful WAL
        // drain. The OS releases the holder's advisory lock on process death.
        holder.sigkill_and_reap();

        // The dir is now re-openable with NO manual stale-lock cleanup:
        // (a) the pure lock primitive re-acquires (targeted oracle: the OS
        //     auto-released the dead holder's lock — no stale-lock bricking),…
        {
            let relock = DataDirLock::acquire(&data_dir).expect(
                "a crashed holder's advisory lock MUST be auto-released by the OS — \
                 the dir must NOT be bricked for the next opener (#886)",
            );
            assert_eq!(relock.path(), data_dir.join("LOCK").as_path());
            drop(relock); // release before the end-to-end bootstrap re-locks.
        }
        // (b) …and a full durable bootstrap succeeds end-to-end (recovers the
        //     crashed holder's fsync'd catalog genesis + re-locks the dir) — the
        //     "a subsequent process can open" guarantee.
        let (_backend, guard) = bootstrap_storage_backend(&durable(&data_dir)).expect(
            "a fresh durable bootstrap after a kill -9 crash MUST succeed (no stale-lock bricking)",
        );
        assert!(guard.is_durable());
        assert_eq!(
            guard.data_dir_lock_path(),
            Some(data_dir.join("LOCK").as_path()),
        );
    }
}
