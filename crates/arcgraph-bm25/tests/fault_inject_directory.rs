//! Test-helper `Directory` wrapper that injects errors at configurable
//! Tantivy directory I/O points (W9a M3.b N1 — issue #224).
//!
//! Used by `tests/eviction_recovery.rs::permit_returned_to_pool_on_tantivy_rollback_failure`
//! (the rollback-path F1 regression pin) so the test exercises the
//! leak shape empirically instead of decorating: codex round-2 review
//! N1 found that the prior `chmod 0o500`-on-tenant-directory injection
//! does not reliably surface `Err` on Tantivy's `IndexWriter::rollback()`
//! path because rollback's I/O is mostly directory-read (`atomic_read`
//! of `meta.json`), not write. A wrapping `Directory` returns `Err`
//! synchronously on the next read/write of any kind regardless of
//! filesystem state, which is what the regression pin actually needs.
//!
//! # How injection works
//!
//! `FaultInjectDirectory<D>` wraps any `D: Directory + Clone` and
//! exposes two `Arc<AtomicBool>` flags. When set, the corresponding
//! flag intercepts a specific subset of directory operations:
//!
//! - `inject_rollback_err` — `atomic_read` returns `Err`. Tantivy's
//!   `IndexWriter::rollback()` constructs a fresh `IndexWriter` via
//!   `IndexWriter::new(...)?`, which calls `index.load_metas()?.opstamp`,
//!   which calls `directory.atomic_read(&META_FILEPATH)?`. Failing
//!   `atomic_read` therefore deterministically fails `rollback()` —
//!   regardless of whether the underlying filesystem is writable.
//! - `inject_commit_err` — `open_write` returns `Err`. Commit's
//!   segment-file publish goes through `directory.open_write(&seg_path)`
//!   for each new segment file. Failing `open_write` fails the commit
//!   path. (Symmetric to `inject_rollback_err` even though the commit
//!   path is already load-bearing via the chmod test on the tenant
//!   dir; this flag is here for symmetry and future commit-path
//!   regression pins.)
//!
//! Flags are `Arc<AtomicBool>` because Tantivy's `Directory::box_clone`
//! produces independent clones of the wrapper struct via the
//! `DirectoryClone` blanket impl — sharing the flag arc across clones
//! lets the test toggle injection on the original handle and have it
//! affect every Tantivy-owned clone of the directory.
//!
//! # Why a custom `Directory` instead of more `chmod` tricks
//!
//! Tantivy `IndexWriter::rollback()` body (tantivy 0.26.1
//! `src/indexer/index_writer.rs:564-595`):
//!
//! ```text
//! self.segment_updater.kill();
//! ...
//! let new_index_writer = IndexWriter::new(&self.index, ..., directory_lock)?;
//! ```
//!
//! `IndexWriter::new` calls `index.load_metas()?` (index.rs:520) which
//! calls `directory.atomic_read(META_FILEPATH)`. So rollback I/O is a
//! *read*, not a *write*, and `chmod 0o500` (write-deny) on the
//! directory does NOT surface `Err`. A wrapping `Directory` is the
//! correct injection seam — flagged in the issue body as the "most
//! flexible; cross-platform" approach.
//!
//! # Cross-platform
//!
//! Unlike the `chmod`-based path (`#[cfg(unix)]`), this wrapper is
//! cross-platform: it is pure Rust on top of the Tantivy `Directory`
//! trait. The rollback regression pin no longer needs `#[cfg(unix)]`.

#![allow(dead_code)]

use std::fmt;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tantivy::directory::error::{DeleteError, LockError, OpenReadError, OpenWriteError};
use tantivy::directory::{
    Directory, DirectoryLock, FileHandle, FileSlice, Lock, WatchCallback, WatchHandle, WritePtr,
};

/// Atomic flags driving fault-injection points in
/// [`FaultInjectDirectory`]. Held inside an [`Arc`] so toggling on the
/// test-side handle is visible to every Tantivy-owned clone of the
/// directory.
#[derive(Default, Debug)]
pub struct FaultInjectFlags {
    /// When `true`, [`Directory::atomic_read`] returns
    /// [`OpenReadError::IoError`] regardless of filesystem state. Used
    /// to fail the rollback path (Tantivy's `IndexWriter::rollback()`
    /// reads `meta.json` via `atomic_read` inside `IndexWriter::new`).
    pub inject_rollback_err: AtomicBool,
    /// When `true`, [`Directory::open_write`] returns
    /// [`OpenWriteError::IoError`]. Used to fail the commit-path
    /// segment-file write. (Symmetric to `inject_rollback_err`; not
    /// load-bearing for the W9a M3.b N1 fix but here for symmetry.)
    pub inject_commit_err: AtomicBool,
}

impl FaultInjectFlags {
    /// Construct flags with both injectors disabled.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Whether `inject_rollback_err` is set.
    #[must_use]
    pub fn rollback_active(&self) -> bool {
        self.inject_rollback_err.load(Ordering::Acquire)
    }

    /// Whether `inject_commit_err` is set.
    #[must_use]
    pub fn commit_active(&self) -> bool {
        self.inject_commit_err.load(Ordering::Acquire)
    }

    /// Activate (or deactivate) the rollback-path injector.
    pub fn set_rollback_err(&self, on: bool) {
        self.inject_rollback_err.store(on, Ordering::Release);
    }

    /// Activate (or deactivate) the commit-path injector.
    pub fn set_commit_err(&self, on: bool) {
        self.inject_commit_err.store(on, Ordering::Release);
    }
}

/// `Directory` wrapper that delegates to an inner `D: Directory + Clone`
/// and intercepts a subset of operations to return `Err` when the
/// corresponding [`FaultInjectFlags`] is set.
///
/// Construction wraps a real Tantivy directory (typically a
/// `MmapDirectory` for the per-tenant dir, or a `RamDirectory` for
/// pure-memory tests). Tantivy clones `D` internally via
/// [`tantivy::directory::DirectoryClone::box_clone`]; the flag arc is
/// shared across clones so the test can toggle injection on the
/// original handle and affect every Tantivy-owned clone.
pub struct FaultInjectDirectory<D: Directory + Clone> {
    inner: D,
    flags: Arc<FaultInjectFlags>,
}

impl<D: Directory + Clone> FaultInjectDirectory<D> {
    /// Wrap `inner` with a fresh [`FaultInjectFlags`] (both injectors
    /// off). Use [`Self::flags`] to retrieve a clone of the
    /// `Arc<FaultInjectFlags>` for test-side toggling.
    #[must_use]
    pub fn new(inner: D) -> Self {
        Self {
            inner,
            flags: FaultInjectFlags::new(),
        }
    }

    /// Wrap `inner` and share `flags` with an existing
    /// [`FaultInjectFlags`]. Useful when constructing multiple wrapped
    /// directories that should respond to the same toggle (e.g., a
    /// per-tenant factory called repeatedly by `Bm25Service::handle`).
    #[must_use]
    pub fn with_flags(inner: D, flags: Arc<FaultInjectFlags>) -> Self {
        Self { inner, flags }
    }

    /// Clone of the shared flags. Cheap (just an `Arc::clone`).
    #[must_use]
    pub fn flags(&self) -> Arc<FaultInjectFlags> {
        Arc::clone(&self.flags)
    }
}

impl<D: Directory + Clone> Clone for FaultInjectDirectory<D> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            flags: Arc::clone(&self.flags),
        }
    }
}

impl<D: Directory + Clone> fmt::Debug for FaultInjectDirectory<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FaultInjectDirectory")
            .field("inner", &self.inner)
            .field("inject_rollback_err", &self.flags.rollback_active())
            .field("inject_commit_err", &self.flags.commit_active())
            .finish()
    }
}

impl<D: Directory + Clone> Directory for FaultInjectDirectory<D> {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        self.inner.get_file_handle(path)
    }

    fn open_read(&self, path: &Path) -> Result<FileSlice, OpenReadError> {
        self.inner.open_read(path)
    }

    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        self.inner.delete(path)
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        self.inner.exists(path)
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        if self.flags.commit_active() {
            return Err(OpenWriteError::wrap_io_error(
                io::Error::other("FaultInjectDirectory: injected commit-path open_write error"),
                path.to_path_buf(),
            ));
        }
        self.inner.open_write(path)
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        if self.flags.rollback_active() {
            return Err(OpenReadError::wrap_io_error(
                io::Error::other("FaultInjectDirectory: injected rollback-path atomic_read error"),
                path.to_path_buf(),
            ));
        }
        self.inner.atomic_read(path)
    }

    fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        if self.flags.commit_active() {
            return Err(io::Error::other(
                "FaultInjectDirectory: injected commit-path atomic_write error",
            ));
        }
        self.inner.atomic_write(path, data)
    }

    fn sync_directory(&self) -> io::Result<()> {
        self.inner.sync_directory()
    }

    fn acquire_lock(&self, lock: &Lock) -> Result<DirectoryLock, LockError> {
        self.inner.acquire_lock(lock)
    }

    fn watch(&self, watch_callback: WatchCallback) -> tantivy::Result<WatchHandle> {
        self.inner.watch(watch_callback)
    }
}

#[cfg(test)]
mod self_tests {
    //! Self-tests for the test helper. These run as part of the
    //! standalone `fault_inject_directory` integration-test binary
    //! (cargo treats `tests/foo.rs` as its own test crate); the
    //! `permit_returned_to_pool_on_tantivy_rollback_failure` test in
    //! `eviction_recovery.rs` includes this same source via
    //! `#[path = "fault_inject_directory.rs"] mod ...`, so the helper
    //! is exercised in both contexts.

    use super::*;
    use tantivy::directory::RamDirectory;

    #[test]
    fn flags_default_to_off_and_round_trip_set() {
        let flags = FaultInjectFlags::new();
        assert!(!flags.rollback_active());
        assert!(!flags.commit_active());
        flags.set_rollback_err(true);
        flags.set_commit_err(true);
        assert!(flags.rollback_active());
        assert!(flags.commit_active());
        flags.set_rollback_err(false);
        flags.set_commit_err(false);
        assert!(!flags.rollback_active());
        assert!(!flags.commit_active());
    }

    #[test]
    fn cloning_preserves_flag_arc_so_toggle_propagates_to_clone() {
        // Tantivy clones the directory internally via DirectoryClone;
        // the flag toggle must affect every clone, not just the
        // original. Pin so a regression that switches `Arc<AtomicBool>`
        // for plain `AtomicBool` (which would be by-value-cloned and
        // diverge) surfaces here.
        let inner = RamDirectory::create();
        let d = FaultInjectDirectory::new(inner);
        let cloned = d.clone();
        d.flags().set_rollback_err(true);
        assert!(
            cloned.flags().rollback_active(),
            "PIN: a flag toggle on the original wrapper must be visible \
             to its Tantivy-owned clones (shared Arc<AtomicBool>)"
        );
    }

    #[test]
    fn atomic_read_returns_err_when_rollback_flag_set() {
        let inner = RamDirectory::create();
        let d = FaultInjectDirectory::new(inner);
        // Sentinel write so the read has data to return on the
        // success path.
        d.atomic_write(Path::new("sentinel.json"), b"{\"x\":1}")
            .expect("seed write while flags off must succeed");
        let read_ok = d.atomic_read(Path::new("sentinel.json"));
        assert!(
            read_ok.is_ok(),
            "PIN: atomic_read with flags off must delegate to inner"
        );
        d.flags().set_rollback_err(true);
        let read_err = d.atomic_read(Path::new("sentinel.json"));
        assert!(
            read_err.is_err(),
            "PIN: atomic_read with inject_rollback_err set MUST return Err \
             — this is what makes Tantivy's IndexWriter::rollback() \
             surface Err deterministically"
        );
    }

    #[test]
    fn open_write_returns_err_when_commit_flag_set() {
        let inner = RamDirectory::create();
        let d = FaultInjectDirectory::new(inner);
        let ok = d.open_write(Path::new("seg.dat"));
        assert!(
            ok.is_ok(),
            "PIN: open_write with flags off must delegate to inner"
        );
        d.flags().set_commit_err(true);
        let err = d.open_write(Path::new("seg2.dat"));
        assert!(
            err.is_err(),
            "PIN: open_write with inject_commit_err set MUST return Err"
        );
    }
}
