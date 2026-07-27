//! Exclusive advisory inter-process lock on a durable data directory.
//!
//! # Why (issue #886, ADR-183 Strict-tier)
//!
//! ArcGraph's durable storage engine is **single-writer by design** (the
//! buffer-pool + WAL architecture, design-v2 §3): exactly one process may own a
//! `<data_dir>` at a time. Nothing enforced that across *processes*, so two
//! `arcgraph serve --data <SAMEDIR>` invocations both opened the same store,
//! interleaved their WAL appends, and bricked it on the next restart
//! (`WalCorruption { … crc mismatch }` at `Lsn(0)` — unrecoverable, losing
//! acknowledged Strict-tier (fsync-before-ack) commits, violating the ADR-183
//! "acked commits survive restart" guarantee). It is reachable through the
//! documented CLI: `--http` `conflicts_with` `--bolt`, so an operator who wants
//! both protocols on one durable store has no single-process option and reaches
//! for two `serve` processes (#886).
//!
//! This module adds the missing exclusion: a durable bootstrap takes an
//! **exclusive, non-blocking advisory lock** on `<data_dir>/LOCK` before it
//! opens `pages.db` / the WAL (`crate::bootstrap::build_durable` §1). A second
//! opener fails fast with an actionable error instead of silently corrupting the
//! store — matching Neo4j ("The database is already in use… Store lock"),
//! PostgreSQL, LMDB, and RocksDB.
//!
//! # Mechanism + crash safety
//!
//! - **unix:** `flock(2)` with `LOCK_EX | LOCK_NB` over an owned `File` fd
//!   (`libc::flock`). The lock is *advisory* and tied to the open file
//!   description; the kernel releases it when the fd is closed — on
//!   `DataDirLock` `Drop` (clean exit) **and** on process death, including
//!   `SIGKILL` / OOM / power loss. So a crashed holder never bricks the dir for
//!   the next opener: there is no on-disk "locked" flag to reap, no stale-lock
//!   failure mode (proven by `crash_releases_data_dir_lock_886` in
//!   `tests/durable_interprocess_lock_886.rs`).
//! - **windows:** the lockfile is opened with `share_mode(0)` (deny all
//!   sharing), so the OS refuses any second open until this handle closes (on
//!   `Drop` or process death) — the same crash-release semantics, std-only with
//!   no FFI. Best effort: not exercised on the macOS/Linux CI platforms (see the
//!   PR's Risks section).
//!
//! Both paths are **non-blocking** (`LOCK_NB` / a single open attempt): a held
//! lock returns immediately as "in use", never hangs the second opener.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// Lockfile name created inside `<data_dir>` to host the advisory lock. Lives at
/// the data-dir root (NOT inside `<data_dir>/wal`), so it never perturbs the
/// `build_durable` "is the WAL pre-existing?" restart heuristic.
pub const LOCK_FILE: &str = "LOCK";

/// An exclusive advisory lock held on a durable `<data_dir>` for the lifetime of
/// this process.
///
/// Acquired by [`DataDirLock::acquire`] at durable bootstrap (before the WAL /
/// `pages.db` are opened) and stored on the [`crate::bootstrap::DurabilityGuard`]
/// so it is held for the whole server loop and released — by the OS — when that
/// guard drops or the process dies.
///
/// The lock is held purely by keeping the underlying lockfile [`File`] open;
/// there is no explicit unlock call, the clean-path release IS the `Drop` of the
/// owned handle.
#[derive(Debug)]
#[must_use = "dropping the DataDirLock releases the inter-process lock on the data dir"]
pub struct DataDirLock {
    /// The open lockfile handle. Its *only* purpose is to own the OS lock for
    /// the lifetime of this value: on unix it holds the `flock`; on windows it
    /// holds the exclusive `share_mode(0)` open. Closing it (on `Drop` or
    /// process death) releases the lock. Never read — held solely for its
    /// `Drop` / fd ownership.
    _file: File,
    /// `<data_dir>/LOCK`, retained for diagnostics + the [`Debug`] impl.
    path: PathBuf,
}

impl DataDirLock {
    /// Take an exclusive, non-blocking advisory lock on `<data_dir>/LOCK`.
    ///
    /// `data_dir` MUST already exist — durable bootstrap `create_dir_all`s it
    /// immediately before calling this (the lockfile is created inside it).
    ///
    /// # Errors
    ///
    /// - **Lock held** (the actionable #886 case): another `arcgraph serve`
    ///   process already owns this data dir. The error names the dir, explains
    ///   that a durable store is single-process, and that a second process would
    ///   corrupt the WAL (#886) — so the operator stops the other process or
    ///   picks a different `--data` dir. The caller must NOT proceed to WAL
    ///   replay or bind a listener.
    /// - **I/O error** creating/opening the lockfile (permissions, `ENOSPC`, …)
    ///   or an unexpected `flock` errno.
    pub fn acquire(data_dir: &Path) -> Result<Self> {
        let path = data_dir.join(LOCK_FILE);
        match try_open_and_lock(&path)? {
            LockOutcome::Acquired(file) => Ok(Self { _file: file, path }),
            LockOutcome::Held => bail!(
                "data dir {dir} is already in use by another `arcgraph serve` process \
                 (the advisory lock on {lock} is held).\n\
                 A durable ArcGraph data dir is single-process: a second `serve --data {dir}` \
                 would interleave WAL appends and corrupt the store on the next restart \
                 (unrecoverable — acknowledged commits are lost; issue #886).\n\
                 Stop the other process, or start this one with a different --data dir.",
                dir = data_dir.display(),
                lock = path.display(),
            ),
        }
    }

    /// The lockfile path (`<data_dir>/LOCK`).
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Outcome of one open-and-lock attempt on the lockfile.
enum LockOutcome {
    /// Lock acquired; hold this [`File`] (do not drop it) to keep the lock.
    Acquired(File),
    /// Another process already holds the lock — fail fast with a friendly error.
    Held,
}

/// Open `<data_dir>/LOCK` read-write (creating it) and take an exclusive,
/// non-blocking advisory lock on it.
///
/// Returns [`LockOutcome::Acquired`] (caller keeps the `File`),
/// [`LockOutcome::Held`] when another process owns it, or `Err` on an
/// unexpected I/O error.
#[cfg(unix)]
fn try_open_and_lock(path: &Path) -> Result<LockOutcome> {
    use std::os::unix::io::AsRawFd;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        // The lockfile is a pure flock anchor — never truncate it (it may carry
        // a previous owner's bytes; we neither read nor rewrite its content).
        .truncate(false)
        .open(path)
        .with_context(|| format!("open data-dir lockfile {}", path.display()))?;

    let fd = file.as_raw_fd();
    // SAFETY: `fd` is a valid, open file descriptor owned by `file`, which is
    // borrowed (and so kept alive) for the whole duration of this call. `flock`
    // is an advisory whole-file lock that takes only the integer fd + a flags
    // bitset; it neither dereferences a pointer nor retains the fd past the
    // call, so no aliasing, lifetime, or memory-safety invariant is at stake
    // beyond the fd's validity, which `file` guarantees here. `LOCK_EX | LOCK_NB`
    // requests an exclusive lock and returns immediately with errno
    // `EWOULDBLOCK` (rather than blocking) when another open file description
    // already holds it. The kernel releases the lock when the fd is closed — on
    // `DataDirLock::Drop` and on process death — so it cannot leak.
    let rc = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if rc == 0 {
        return Ok(LockOutcome::Acquired(file));
    }

    let err = std::io::Error::last_os_error();
    match err.raw_os_error() {
        // EWOULDBLOCK: the file is locked and LOCK_NB was selected (flock(2)).
        Some(libc::EWOULDBLOCK) => Ok(LockOutcome::Held),
        _ => Err(anyhow::Error::new(err)
            .context(format!("flock(LOCK_EX|LOCK_NB) on {}", path.display()))),
    }
}

/// Windows counterpart: a `share_mode(0)` (deny-all-sharing) open IS the lock —
/// the OS refuses a second open of the same path until this handle closes (on
/// `Drop` or process death). std-only, no FFI; a second opener's `open` fails
/// with `ERROR_SHARING_VIOLATION`, which we translate to [`LockOutcome::Held`].
#[cfg(windows)]
fn try_open_and_lock(path: &Path) -> Result<LockOutcome> {
    use std::os::windows::fs::OpenOptionsExt;

    /// `ERROR_SHARING_VIOLATION` — another process has the file open with a
    /// sharing mode that forbids our open (Win32 system error 32).
    const ERROR_SHARING_VIOLATION: i32 = 32;

    match OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        // The lockfile is a pure sharing anchor — never truncate it.
        .truncate(false)
        .share_mode(0)
        .open(path)
    {
        Ok(file) => Ok(LockOutcome::Acquired(file)),
        Err(e) if e.raw_os_error() == Some(ERROR_SHARING_VIOLATION) => Ok(LockOutcome::Held),
        Err(e) => Err(anyhow::Error::new(e)
            .context(format!("open+lock data-dir lockfile {}", path.display()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn acquire_creates_lockfile_and_reports_its_path() {
        let tmp = TempDir::new().expect("tempdir");
        let lock = DataDirLock::acquire(tmp.path()).expect("first acquire");
        assert!(
            tmp.path().join(LOCK_FILE).exists(),
            "acquire must create <data_dir>/LOCK"
        );
        assert_eq!(lock.path(), tmp.path().join(LOCK_FILE));
        drop(lock);
    }

    #[test]
    fn second_acquire_same_dir_is_refused() {
        // The core inter-process exclusion (#886), exercised in-process: flock
        // treats two separate open descriptions as independent even within one
        // process, so the 2nd acquire on a held dir is denied — exactly as a 2nd
        // process would be. RED-on-revert: a no-op lock lets the 2nd acquire
        // succeed and this expect_err panics.
        let tmp = TempDir::new().expect("tempdir");
        let _held = DataDirLock::acquire(tmp.path()).expect("first acquire");
        let err = DataDirLock::acquire(tmp.path())
            .expect_err("second acquire on a held dir MUST be refused");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("already in use"),
            "refusal must say the dir is already in use; got: {msg}"
        );
        assert!(
            msg.contains(&tmp.path().display().to_string()),
            "refusal must name the data dir; got: {msg}"
        );
        assert!(
            msg.contains("886"),
            "refusal should cite the root-cause issue #886; got: {msg}"
        );
    }

    #[test]
    fn acquire_succeeds_again_after_drop() {
        // Clean release on Drop: a fresh acquire after the first is dropped
        // succeeds (no stale-lock bricking on the graceful path).
        let tmp = TempDir::new().expect("tempdir");
        let first = DataDirLock::acquire(tmp.path()).expect("first acquire");
        drop(first);
        let _second = DataDirLock::acquire(tmp.path())
            .expect("acquire after drop must succeed (lock released on Drop)");
    }

    #[test]
    fn different_dirs_do_not_conflict() {
        let a = TempDir::new().expect("tempdir a");
        let b = TempDir::new().expect("tempdir b");
        let _la = DataDirLock::acquire(a.path()).expect("acquire dir a");
        let _lb =
            DataDirLock::acquire(b.path()).expect("acquire dir b — different dir, no conflict");
    }
}
