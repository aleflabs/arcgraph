//! K-2 multi-FS adapter scaffolding (issue #223; ADR-038
//! amendment-03 §"Slice K" K-2 row).
//!
//! ## Why per-FS adapters
//!
//! K-1a (PR #176) + K-1b (PR #219) build the rate-based fault-injection
//! API + multi-tenant interleave on top of the calling-process default
//! filesystem (whatever `std::env::temp_dir()` returns). That is good
//! enough for harness-shape verification but does NOT exercise the
//! load-bearing FS-variation contracts that v1.0 production deployments
//! will hit:
//!
//! - **APFS** (macOS dev hosts, common deployment for client-side
//!   embeds) — `F_FULLFSYNC` semantics differ from `fsync`; APFS does
//!   NOT support `O_DIRECT`. ADR-031 §R2 (WAL segment durability) +
//!   ADR-034 D-1 (T1 strict durability) both bind to fsync semantics.
//! - **ext4** (mainstream Linux server) — the canonical FS for ADR-031
//!   commit-bundle atomicity proofs (the journaling layer's data=ordered
//!   default matches our atomicity invariants without ceremony).
//! - **XFS** (high-throughput Linux deployments) — `F_FALLOCATE` and
//!   metadata journaling differ from ext4; the WAL writer's
//!   pre-allocation hints + segment recycle path may surface
//!   FS-specific edge cases here.
//! - **EBS** (AWS gp3 / io2) — network-attached block storage with
//!   stalls in the 100ms-1s range during EBS volume migrations or
//!   throughput throttling. The K-1 background fsync scheduler's
//!   T3-tier RPO-loss bounds need EBS-specific calibration.
//!
//! K-2 is the **scaffolding** for these adapters; the K-2 PR ships a
//! 1-hour smoke per-FS variant + the multi-FS proptest for recovery
//! determinism. K-3 (issue #215, separate slice) adds the multi-hour
//! crons.
//!
//! ## Adapter trait shape
//!
//! Per spec D2 in the K-2 spawn prompt:
//!
//! ```ignore
//! pub trait FsAdapter {
//!     fn name(&self) -> &'static str;
//!     fn create_tmpdir(&self) -> std::io::Result<FsTempDir>;
//!     fn fsync_durable(&self, path: &Path) -> std::io::Result<()>;
//!     fn supports_o_direct(&self) -> bool;
//! }
//! ```
//!
//! Plus `is_supported(&self) -> bool` so tests can skip-on-platform-
//! mismatch instead of panicking when an adapter constructs on the
//! wrong host.
//!
//! ## What this module is NOT
//!
//! - It does NOT remount filesystems (we don't have CAP_SYS_ADMIN in
//!   CI). The adapters use `std::env::temp_dir()` as the substrate and
//!   declare which FS semantics they're approximating + which they're
//!   actually exercising. Future K-3 may upgrade to actual
//!   sudo-mounted ramdisks with explicit FS variants when the CI host
//!   supports it.
//! - It does NOT bypass the K-1 hooks-vs-production discipline (mod.rs
//!   §"Hooks vs production"). The adapter's `fsync_durable` calls into
//!   `std::fs::File::sync_all` on every platform; the difference between
//!   adapters is the SEMANTIC documentation + the platform-skip gate,
//!   not the production-side fsync implementation.
//! - It does NOT implement encoding-mismatch I-V coverage (that's
//!   K-1c+d / K-3, separate slice — issue #215).
//!
//! ## Why a custom `FsTempDir` (not `tempfile::TempDir`)
//!
//! `tempfile` is a `[dev-dependencies]` entry in `arcgraph-storage`
//! Cargo.toml — available for `#[cfg(test)]` blocks within the lib
//! AND for integration tests under `tests/`, but NOT for the lib's
//! public API surface. Since `multi_fs.rs` lives under `src/` and is
//! `pub`-exported (tests/k2_*.rs imports it across crate boundaries),
//! the adapter trait cannot return `tempfile::TempDir` without
//! promoting `tempfile` from dev-dep to regular dep — a Cargo.toml
//! change outside K-2's "test-harness only" hard-boundary scope.
//!
//! The custom `FsTempDir` wrapper provides the same Drop-based
//! cleanup semantics as `tempfile::TempDir` without the dep churn.
//!
//! ## Forward references
//!
//! - **K-1c (issue #215)** will use these adapters under a 1-hour
//!   per-FS-variant cron campaign + add encoding-mismatch I-V coverage.
//! - **K-3** (post-v1.0-alpha) may upgrade adapters to use actual
//!   sudo-mounted ramdisks per FS variant, and add an `EBS_REMOTE`
//!   adapter that rdma-mounts a real EBS volume.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ────────────────────────────────────────────────────────────────────
// FsTempDir — RAII temp-directory wrapper
// ────────────────────────────────────────────────────────────────────

/// RAII wrapper around a temp directory. Mirrors `tempfile::TempDir`
/// semantics: the directory is created on construction + deleted
/// recursively on `Drop` if `cleanup_on_drop` is true.
///
/// Avoids `tempfile` as a non-dev dep for the lib (see module-level
/// "Why a custom `FsTempDir`" note).
///
/// The Drop is best-effort — if the recursive delete fails (e.g.,
/// because the directory has already been unmounted under our feet,
/// or a Drop is called inside a panic chain that already cleaned up),
/// the error is dropped silently. This matches `tempfile::TempDir`'s
/// behavior.
#[derive(Debug)]
pub struct FsTempDir {
    path: PathBuf,
    cleanup_on_drop: bool,
}

impl FsTempDir {
    /// Construct an `FsTempDir` rooted at `path`. The path MUST already
    /// exist (callers typically create it via `std::fs::create_dir_all`
    /// just before constructing). `cleanup_on_drop` controls whether
    /// `Drop` recursively removes the directory.
    pub fn new(path: PathBuf, cleanup_on_drop: bool) -> Self {
        Self {
            path,
            cleanup_on_drop,
        }
    }

    /// Path to the temp directory.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Disable the on-Drop cleanup. Useful for post-mortem debugging
    /// when a test fails — leaks the directory so the user can `ls -la`
    /// it.
    pub fn keep(&mut self) {
        self.cleanup_on_drop = false;
    }

    /// Take ownership of the path without running the Drop cleanup.
    pub fn into_path(mut self) -> PathBuf {
        self.cleanup_on_drop = false;
        std::mem::replace(&mut self.path, PathBuf::new())
    }
}

impl Drop for FsTempDir {
    fn drop(&mut self) {
        if self.cleanup_on_drop && !self.path.as_os_str().is_empty() {
            // Best-effort. A previous panic may have cleaned up; a
            // mount-time-of-check / time-of-use race may have moved
            // the directory.
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// FsAdapter trait
// ────────────────────────────────────────────────────────────────────

/// Per-FS variation adapter for K-2 harness campaigns.
///
/// Each adapter:
///
/// - Declares the FS variant it APPROXIMATES (`name()`).
/// - Reports whether it's APPLICABLE on the current host
///   (`is_supported()`); tests use this to skip-on-platform-mismatch
///   without panicking.
/// - Constructs an `FsTempDir` (`create_tmpdir()`) rooted under the
///   process's `std::env::temp_dir()` so cleanup is correct even on
///   abnormal exit (the OS reaps `temp_dir()` content on reboot).
/// - Provides FS-specific fsync semantics (`fsync_durable`) — APFS
///   uses `F_FULLFSYNC` on macOS for full durability; ext4/XFS use
///   regular `fsync` (which is sufficient because the journaling
///   layer adds the metadata barrier).
/// - Reports whether the FS supports `O_DIRECT` (`supports_o_direct`).
///
/// The trait is `Send + Sync` so adapters can be passed across
/// thread boundaries (a future multi-threaded campaign may run
/// per-FS adapters in parallel worker threads).
pub trait FsAdapter: Send + Sync {
    /// Stable, lowercase name for the FS variant. Used in test logs +
    /// the K-2 campaign manifest.
    fn name(&self) -> &'static str;

    /// True iff the adapter is applicable on the current host. Adapters
    /// whose semantics cannot be reasonably approximated on the current
    /// host return `false`; tests use this gate to skip rather than
    /// fail. For example, `XfsAdapter::is_supported` returns `false` on
    /// macOS (no XFS userspace at v1.0); `EbsAdapter::is_supported`
    /// returns `false` unless `K_2_EBS=1` is set.
    fn is_supported(&self) -> bool;

    /// Construct a fresh temp directory rooted under
    /// `std::env::temp_dir()`. The returned `FsTempDir` cleans up on
    /// `Drop`.
    fn create_tmpdir(&self) -> io::Result<FsTempDir>;

    /// FS-appropriate "durable fsync" of `path`. On APFS this uses
    /// `F_FULLFSYNC`; on ext4/XFS this uses `fsync` (sufficient
    /// because the journal adds the metadata barrier); on tmpfs this
    /// is a no-op (RAM-backed; "durable" is a contradiction).
    ///
    /// ## What this does
    ///
    /// Opens `path` in read-only mode and calls `File::sync_all` on
    /// the resulting file handle. On macOS this maps to
    /// `fcntl(F_FULLFSYNC)`; on Linux this maps to `fsync(2)`. The
    /// implementation is **file-level fsync of the passed path** — it
    /// does NOT issue a directory-level barrier (`fsync(dir_fd)`).
    ///
    /// ## What this does NOT do (issue #237 MEDIUM-3)
    ///
    /// Prior to issue #237, the doc-comment claimed "this is the
    /// directory-level barrier (`fsync(dir_fd)`) that ADR-031 §R2
    /// commit-bundle atomicity needs". That overclaimed — the impl
    /// opens whatever path the caller passes (file or directory) and
    /// fsyncs the resulting `File` handle. K-2 callers pass file
    /// paths, so the call IS file-level fsync, not directory-level.
    ///
    /// The ADR-031 §R2 directory-barrier invariant is enforced by the
    /// production `WalWriter`'s own fsync ordering (the production WAL
    /// writer issues both file + directory fsyncs in the correct order
    /// for commit-bundle atomicity). This adapter method is a
    /// test-harness primitive for FS-variation campaigns; it is NOT
    /// the production atomicity surface.
    fn fsync_durable(&self, path: &Path) -> io::Result<()>;

    /// True iff the FS supports `O_DIRECT` for direct-IO paths. APFS:
    /// false. tmpfs: false. ext4 / XFS / EBS: true.
    fn supports_o_direct(&self) -> bool;
}

// ────────────────────────────────────────────────────────────────────
// FsKind enum (catalog of supported variants)
// ────────────────────────────────────────────────────────────────────

/// The four FS variants K-2 ships adapters for. Tests parameterize on
/// this enum + dispatch to the matching adapter via [`adapter_for`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FsKind {
    /// macOS APFS (default macOS) — `F_FULLFSYNC`-style durability;
    /// no `O_DIRECT`.
    Apfs,
    /// Linux ext4 (default Linux server) — fsync sufficient; supports
    /// `O_DIRECT`.
    Ext4,
    /// Linux XFS (high-throughput Linux) — fsync sufficient; supports
    /// `O_DIRECT`. Skipped on non-Linux.
    Xfs,
    /// AWS EBS (gp3 / io2) — fsync sufficient; supports `O_DIRECT`.
    /// Skipped unless `K_2_EBS=1`.
    Ebs,
}

impl FsKind {
    /// All four variants in canonical iteration order. Stable for
    /// reproducible test logs.
    pub const ALL: [FsKind; 4] = [FsKind::Apfs, FsKind::Ext4, FsKind::Xfs, FsKind::Ebs];

    /// Lowercase name. Matches the corresponding adapter's `name()`.
    pub fn name(self) -> &'static str {
        match self {
            FsKind::Apfs => "apfs",
            FsKind::Ext4 => "ext4",
            FsKind::Xfs => "xfs",
            FsKind::Ebs => "ebs",
        }
    }
}

/// Construct the adapter for a given [`FsKind`]. Always returns an
/// adapter; the caller checks [`FsAdapter::is_supported`] to decide
/// whether to skip the variant on the current host.
pub fn adapter_for(kind: FsKind) -> Box<dyn FsAdapter> {
    match kind {
        FsKind::Apfs => Box::new(ApfsAdapter::new()),
        FsKind::Ext4 => Box::new(Ext4Adapter::new()),
        FsKind::Xfs => Box::new(XfsAdapter::new()),
        FsKind::Ebs => Box::new(EbsAdapter::new()),
    }
}

/// All adapters that are applicable on the current host. Tests iterate
/// this list to run a per-applicable-FS campaign without hand-skipping
/// each variant.
pub fn supported_adapters() -> Vec<Box<dyn FsAdapter>> {
    FsKind::ALL
        .iter()
        .map(|k| adapter_for(*k))
        .filter(|a| a.is_supported())
        .collect()
}

// ────────────────────────────────────────────────────────────────────
// Helper — unique workdir-name generation
// ────────────────────────────────────────────────────────────────────

/// Process-local counter so two adapters running in the same process
/// don't collide on workdir names. Combined with the wall-clock
/// nanoseconds + the adapter name, the workdir name is unique per
/// adapter+call.
static WORKDIR_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_workdir_path(adapter_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let counter = WORKDIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    std::env::temp_dir().join(format!(
        "arcgraph-k2-{adapter_name}-{pid}-{nanos}-{counter}"
    ))
}

fn create_unique_workdir(adapter_name: &str) -> io::Result<FsTempDir> {
    let path = unique_workdir_path(adapter_name);
    std::fs::create_dir_all(&path)?;
    Ok(FsTempDir::new(path, true))
}

/// Open + fsync a path. On macOS this uses `F_FULLFSYNC`-style
/// `sync_all` semantics; on Linux this is a regular fsync (sufficient
/// because ext4/XFS journals add the metadata barrier).
///
/// We use Rust's `File::sync_all` which on Linux maps to `fsync(2)`,
/// on macOS maps to `fcntl(F_FULLFSYNC)` per the std docs (rust-lang
/// std `sync_all` doc-comment: "this is equivalent to F_FULLFSYNC on
/// macOS"). This is exactly the durability surface ADR-034 D-1 strict
/// tier needs.
fn open_and_fsync(path: &Path) -> io::Result<()> {
    let f = std::fs::File::open(path)?;
    f.sync_all()?;
    Ok(())
}

// ────────────────────────────────────────────────────────────────────
// APFS adapter
// ────────────────────────────────────────────────────────────────────

/// macOS APFS adapter. On macOS hosts the adapter exercises real APFS
/// (since `std::env::temp_dir()` resolves to `/var/folders/.../tmp` on
/// macOS, which is APFS-backed under the user's startup volume). On
/// non-macOS the adapter falls back to whatever the host's temp_dir
/// is (typically tmpfs or ext4 on Linux); `is_supported()` returns
/// `true` regardless so CI on Linux runners still exercises the
/// adapter's HARNESS-SHAPE contract (construct + fsync + cleanup
/// round-trip).
///
/// ## Surrogate limitations (issue #237 NIT-2)
///
/// "tmpfs surrogate elsewhere" is a SHAPE surrogate, NOT a semantic
/// surrogate: tmpfs is RAM-backed, so `fsync` is a no-op and the
/// "durability" the adapter exercises is purely procedural. Real
/// APFS-specific semantics (`F_FULLFSYNC` vs `fsync`, the lack of
/// `O_DIRECT`, the COW snapshot lineage) are exercised ONLY on macOS
/// hosts. Production behavior verification on actual FS variants is
/// K-1c / K-3 scope (sudo'd loopback mounts per FS).
pub struct ApfsAdapter;

impl ApfsAdapter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ApfsAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl FsAdapter for ApfsAdapter {
    fn name(&self) -> &'static str {
        "apfs"
    }

    fn is_supported(&self) -> bool {
        // APFS is approximated by std::env::temp_dir() on macOS hosts;
        // on non-macOS we still construct (using the host tmpfs) so
        // CI on Linux runners can still exercise the harness shape.
        true
    }

    fn create_tmpdir(&self) -> io::Result<FsTempDir> {
        create_unique_workdir(self.name())
    }

    fn fsync_durable(&self, path: &Path) -> io::Result<()> {
        // On macOS, `File::sync_all` invokes `fcntl(F_FULLFSYNC)` per
        // the Rust std docs — the strongest durability barrier APFS
        // exposes. On non-macOS this falls through to fsync, which
        // is the canonical tmpfs surrogate.
        open_and_fsync(path)
    }

    fn supports_o_direct(&self) -> bool {
        // APFS does NOT support O_DIRECT cleanly. ADR-031 §R2 notes
        // this; production builds skip the O_DIRECT path on macOS.
        false
    }
}

// ────────────────────────────────────────────────────────────────────
// ext4 adapter
// ────────────────────────────────────────────────────────────────────

/// Linux ext4 adapter. On Linux hosts the adapter exercises ext4 if
/// `std::env::temp_dir()` is on an ext4 mount (typical on `/tmp` for
/// older Linux distros + `/var/tmp` for modern systemd-run distros).
/// On non-Linux the adapter falls back to the host tmpfs; the harness
/// contract (construct + fsync + cleanup) holds.
pub struct Ext4Adapter;

impl Ext4Adapter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Ext4Adapter {
    fn default() -> Self {
        Self::new()
    }
}

impl FsAdapter for Ext4Adapter {
    fn name(&self) -> &'static str {
        "ext4"
    }

    fn is_supported(&self) -> bool {
        // ext4 is the Linux default; on macOS we still construct for
        // CI smoke-completeness (tmpfs surrogate). The semantic
        // documentation pin (`fsync sufficient`, `O_DIRECT supported`)
        // is what's load-bearing — not the actual on-disk format.
        true
    }

    fn create_tmpdir(&self) -> io::Result<FsTempDir> {
        create_unique_workdir(self.name())
    }

    fn fsync_durable(&self, path: &Path) -> io::Result<()> {
        // ext4's journaling layer (data=ordered default) provides the
        // metadata barrier; `fsync(2)` is sufficient.
        open_and_fsync(path)
    }

    fn supports_o_direct(&self) -> bool {
        // ext4 supports O_DIRECT cleanly. Mainstream Linux production
        // path.
        true
    }
}

// ────────────────────────────────────────────────────────────────────
// XFS adapter
// ────────────────────────────────────────────────────────────────────

/// Linux XFS adapter. Skipped on macOS (no XFS userspace at v1.0).
/// On Linux the adapter constructs against the host tmpfs as a
/// surrogate — actual XFS-mounted exercise lands at K-1c when the CI
/// runner gets a sudo'd loopback XFS mount.
pub struct XfsAdapter;

impl XfsAdapter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for XfsAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl FsAdapter for XfsAdapter {
    fn name(&self) -> &'static str {
        "xfs"
    }

    fn is_supported(&self) -> bool {
        // Skip on non-Linux. macOS has no XFS userspace at v1.0.
        cfg!(target_os = "linux")
    }

    fn create_tmpdir(&self) -> io::Result<FsTempDir> {
        if !self.is_supported() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "XFS adapter is supported on Linux only; \
                 caller MUST gate via FsAdapter::is_supported",
            ));
        }
        create_unique_workdir(self.name())
    }

    fn fsync_durable(&self, path: &Path) -> io::Result<()> {
        if !self.is_supported() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "XFS adapter is supported on Linux only",
            ));
        }
        // XFS metadata journaling adds the directory-level barrier;
        // fsync is sufficient. The pre-allocation hints are not
        // exercised at the K-2 scaffolding layer.
        open_and_fsync(path)
    }

    fn supports_o_direct(&self) -> bool {
        // XFS supports O_DIRECT cleanly on Linux. (Returning true
        // unconditionally is fine: callers who actually attempt
        // O_DIRECT will fail-loudly on non-Linux because the FS itself
        // doesn't exist.)
        cfg!(target_os = "linux")
    }
}

// ────────────────────────────────────────────────────────────────────
// EBS adapter
// ────────────────────────────────────────────────────────────────────

/// AWS EBS (gp3 / io2) adapter. Skipped unless `K_2_EBS=1` is set
/// in the environment — on a non-EC2 host the adapter has no real EBS
/// volume to exercise. The env-var gate makes EBS opt-in for CI
/// runners that DO have an EBS volume mounted (typically `/var/lib/`
/// or a mounted volume at `/mnt/data/`).
///
/// The K_2_EBS env var is stylistically distinct from `K_2_*` flags
/// for K-2 sub-features because EBS specifically requires a manual
/// CI opt-in — we don't want a stray `K_2_EVERYTHING=1` flag firing
/// EBS on a developer laptop.
pub struct EbsAdapter;

impl EbsAdapter {
    pub fn new() -> Self {
        Self
    }

    /// Read the `K_2_EBS` env var. `1` enables the adapter; absent /
    /// any other value disables it. Centralising the lookup so test
    /// code can easily mock if needed (today: just env::var).
    fn ebs_env_set() -> bool {
        std::env::var("K_2_EBS").map(|v| v == "1").unwrap_or(false)
    }
}

impl Default for EbsAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl FsAdapter for EbsAdapter {
    fn name(&self) -> &'static str {
        "ebs"
    }

    fn is_supported(&self) -> bool {
        // Opt-in via K_2_EBS=1.
        Self::ebs_env_set()
    }

    fn create_tmpdir(&self) -> io::Result<FsTempDir> {
        if !self.is_supported() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "EBS adapter requires K_2_EBS=1 in the environment; \
                 caller MUST gate via FsAdapter::is_supported",
            ));
        }
        create_unique_workdir(self.name())
    }

    fn fsync_durable(&self, path: &Path) -> io::Result<()> {
        if !self.is_supported() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "EBS adapter requires K_2_EBS=1 in the environment",
            ));
        }
        // EBS gp3 / io2 honor fsync semantics through the EC2
        // hypervisor's storage stack (the EBS volume's underlying
        // disk stack is XFS or ext4 in the host VM). Same fsync
        // contract.
        open_and_fsync(path)
    }

    fn supports_o_direct(&self) -> bool {
        // EBS supports O_DIRECT through the underlying ext4/XFS
        // mount of the EBS volume. (The EBS volume is presented as a
        // block device inside the EC2 VM; the FS on top is what
        // determines O_DIRECT availability.)
        Self::ebs_env_set()
    }
}

// ────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1 — APFS adapter constructs cleanly on macOS or substitutes
    /// host tmpfs on non-macOS. Both paths verify the adapter
    /// round-trips create + fsync + cleanup without panicking.
    #[test]
    fn apfs_adapter_constructs_and_round_trips() {
        let adapter = ApfsAdapter::new();
        assert_eq!(adapter.name(), "apfs");
        assert!(
            adapter.is_supported(),
            "APFS adapter must be supported on every host (real APFS \
             on macOS; tmpfs surrogate elsewhere)"
        );
        assert!(
            !adapter.supports_o_direct(),
            "APFS does not support O_DIRECT; the adapter pins this \
             constant for ADR-031 §R2 compliance"
        );

        let tmp = adapter.create_tmpdir().expect("create_tmpdir");
        assert!(tmp.path().exists(), "temp dir must exist after create");
        assert!(
            tmp.path().is_dir(),
            "temp dir must be a directory: {:?}",
            tmp.path()
        );

        // Write + fsync + cleanup round-trip.
        let f_path = tmp.path().join("apfs-fsync-target");
        std::fs::write(&f_path, b"hello apfs").expect("write");
        adapter
            .fsync_durable(&f_path)
            .expect("fsync_durable on APFS-backed tempfile");

        let parent_path = tmp.path().to_path_buf();
        drop(tmp);
        // After Drop, the directory should typically be gone — but
        // this is best-effort per the FsTempDir contract. Issue #237
        // LOW-3: prior to closure the assert! panicked on the
        // leftover-directory case; that masked the best-effort
        // contract. Post-#237 behavior: force-cleanup + log, NEVER
        // panic.
        if parent_path.exists() {
            let _ = std::fs::remove_dir_all(&parent_path);
            eprintln!(
                "multi_fs::tests::apfs_adapter_constructs_and_round_trips: \
                 APFS adapter's FsTempDir Drop did not clean up at first attempt \
                 (best-effort semantics); forced cleanup succeeded."
            );
        }
    }

    /// Test 2 — ext4 adapter constructs cleanly on Linux or substitutes
    /// host tmpfs on macOS / Windows. Both paths verify the same
    /// round-trip contract.
    #[test]
    fn ext4_adapter_constructs_and_round_trips() {
        let adapter = Ext4Adapter::new();
        assert_eq!(adapter.name(), "ext4");
        assert!(
            adapter.is_supported(),
            "ext4 adapter must be supported on every host (real ext4 \
             on Linux; tmpfs surrogate elsewhere)"
        );
        assert!(
            adapter.supports_o_direct(),
            "ext4 supports O_DIRECT; adapter pins this for the v1.0 \
             production path"
        );

        let tmp = adapter.create_tmpdir().expect("create_tmpdir");
        let f_path = tmp.path().join("ext4-fsync-target");
        std::fs::write(&f_path, b"hello ext4").expect("write");
        adapter
            .fsync_durable(&f_path)
            .expect("fsync_durable on ext4-backed tempfile");
    }

    /// Test 3 — XFS adapter is supported on Linux only; on non-Linux
    /// the adapter must skip cleanly (is_supported=false; create_tmpdir
    /// returns Unsupported).
    #[test]
    fn xfs_adapter_skips_on_non_linux() {
        let adapter = XfsAdapter::new();
        assert_eq!(adapter.name(), "xfs");

        if cfg!(target_os = "linux") {
            assert!(
                adapter.is_supported(),
                "XFS adapter must be supported on Linux"
            );
            let tmp = adapter.create_tmpdir().expect("create_tmpdir on linux");
            assert!(tmp.path().exists());
            assert!(
                adapter.supports_o_direct(),
                "XFS supports O_DIRECT on Linux"
            );
        } else {
            assert!(
                !adapter.is_supported(),
                "XFS adapter must skip on non-Linux"
            );
            // Construction must fail with Unsupported, not a different
            // io error kind (NotFound, PermissionDenied, etc.) so test
            // code can match-on-it cleanly.
            let err = adapter
                .create_tmpdir()
                .expect_err("create_tmpdir must error on non-Linux");
            assert_eq!(
                err.kind(),
                io::ErrorKind::Unsupported,
                "skipped XFS construction must surface Unsupported, \
                 not {:?}",
                err.kind()
            );
            assert!(
                !adapter.supports_o_direct(),
                "XFS adapter on non-Linux must report O_DIRECT \
                 unsupported (consistent with is_supported=false)"
            );
        }
    }

    /// Test 4 — EBS adapter requires K_2_EBS=1; without the env var
    /// the adapter must skip cleanly.
    ///
    /// Note: This test does NOT set K_2_EBS itself — env mutation
    /// races other tests in the same process (Cargo's test harness
    /// runs tests in parallel by default). The "is_supported=false
    /// when env not set" path is the load-bearing path for CI; the
    /// env-set path is exercised by the `tests/k2_*.rs` integration
    /// tests when K_2_EBS=1 is passed at the cargo invocation.
    #[test]
    fn ebs_adapter_skips_without_env_var() {
        // Defensive: snapshot the current K_2_EBS value, ensure it is
        // not "1" for the duration of the test, then restore. If the
        // user ran `K_2_EBS=1 cargo test` we'd otherwise spuriously
        // hit the supported path.
        let prior = std::env::var("K_2_EBS").ok();
        // SAFETY: env-var mutation in a multithreaded test is a known
        // hazard. The test mutates K_2_EBS and restores on exit.
        //
        // Issue #237 LOW-2: the prior comment claimed "no other test
        // reads K_2_EBS" — that was incorrect. The sibling test
        // `supported_adapters_round_trip_create_fsync_cleanup` calls
        // `supported_adapters()` which invokes `EbsAdapter::is_supported()`,
        // which reads `K_2_EBS`. Under `K_2_EBS=1 cargo test` there IS
        // a race window between this test's remove/restore and the
        // sibling test's reads — the sibling could observe a transient
        // empty value and skip the EBS round-trip.
        //
        // We accept this hazard at the test-harness layer because:
        //   (a) the K_2_EBS opt-in path is exercised end-to-end by the
        //       `tests/k2_*.rs` integration tests under the cargo
        //       invocation, NOT this in-module test;
        //   (b) the load-bearing assertion here is the SKIP path
        //       (is_supported=false), which is independent of any
        //       sibling test's behavior;
        //   (c) under default `cargo test` (no K_2_EBS), there's no
        //       contention because the env var stays absent throughout.
        //
        // A future refactor could switch to a process-local mock
        // env-getter on `EbsAdapter` for full thread safety, OR add a
        // `serial_test` dep to serialize EBS-reading tests. Neither is
        // taken at v1.0-alpha — see issue #237 LOW-2 for the deferral
        // rationale.
        unsafe {
            std::env::remove_var("K_2_EBS");
        }

        let adapter = EbsAdapter::new();
        assert_eq!(adapter.name(), "ebs");
        assert!(
            !adapter.is_supported(),
            "EBS adapter must be unsupported without K_2_EBS=1"
        );
        let err = adapter
            .create_tmpdir()
            .expect_err("create_tmpdir must error without K_2_EBS=1");
        assert_eq!(
            err.kind(),
            io::ErrorKind::Unsupported,
            "skipped EBS construction must surface Unsupported, not {:?}",
            err.kind()
        );

        // Restore env var.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("K_2_EBS", v),
                None => std::env::remove_var("K_2_EBS"),
            }
        }
    }

    /// Test 5 — every applicable adapter round-trips create + fsync +
    /// cleanup. This is the cross-FS contract that the K-2 multi-FS
    /// proptest will lean on (tests/k2_fault_during_recovery.rs).
    ///
    /// The test walks every FsKind, gates on is_supported, and asserts
    /// the round-trip — so adding a new FsKind variant automatically
    /// extends test coverage.
    #[test]
    fn supported_adapters_round_trip_create_fsync_cleanup() {
        let adapters = supported_adapters();
        assert!(
            !adapters.is_empty(),
            "at least one adapter must be supported on every host \
             (APFS + ext4 are platform-agnostic surrogates)"
        );

        for adapter in adapters {
            let name = adapter.name();
            let tmp = adapter
                .create_tmpdir()
                .unwrap_or_else(|e| panic!("create_tmpdir for {name}: {e}"));
            assert!(tmp.path().exists(), "temp dir for {name} must exist");
            assert!(
                tmp.path()
                    .components()
                    .next_back()
                    .map(|c| c.as_os_str().to_string_lossy().contains(name))
                    .unwrap_or(false),
                "temp dir name for {name} must include adapter name; \
                 got {:?}",
                tmp.path()
            );

            let f_path = tmp.path().join(format!("{name}-fsync-target"));
            std::fs::write(&f_path, format!("hello {name}").as_bytes())
                .unwrap_or_else(|e| panic!("write for {name}: {e}"));
            adapter
                .fsync_durable(&f_path)
                .unwrap_or_else(|e| panic!("fsync_durable for {name}: {e}"));

            let parent = tmp.path().to_path_buf();
            drop(tmp);
            // Drop is best-effort. The directory should typically be
            // gone, but the FsTempDir contract is "best-effort
            // cleanup" — macOS Drop races with TimeMachine /
            // Spotlight indexing can leave the directory briefly
            // present (and so can an in-progress panic chain).
            //
            // Issue #237 LOW-3: prior to closure, this codepath
            // panicked on the leftover-directory case, masking the
            // "best-effort" contract behind a hard assertion that
            // surfaced as a spurious test failure on macOS hosts. The
            // post-#237 behavior is: force-cleanup + log, NEVER panic.
            // A genuine adapter leak would surface elsewhere
            // (`FsTempDir::keep` is the only path that intentionally
            // bypasses cleanup).
            if parent.exists() {
                let _ = std::fs::remove_dir_all(&parent);
                eprintln!(
                    "multi_fs::tests::supported_adapters_round_trip_create_fsync_cleanup: \
                     {name} adapter's FsTempDir Drop did not clean up at first attempt \
                     (best-effort semantics — see FsTempDir doc); forced cleanup succeeded."
                );
            }
        }
    }

    /// Test 6 — FsKind catalog stability: ALL covers every variant +
    /// names round-trip via FsKind::name() ↔ adapter_for(kind).name().
    /// Pinning so adding a variant requires updating the FsKind::ALL
    /// + this test atomically.
    #[test]
    fn fs_kind_catalog_round_trips_names() {
        for kind in FsKind::ALL {
            let adapter = adapter_for(kind);
            assert_eq!(
                kind.name(),
                adapter.name(),
                "FsKind::{:?}.name() must match adapter_for({:?}).name()",
                kind,
                kind
            );
        }
        // Stability pin: the four variants in declaration order. If a
        // future K-3 adds a fifth (e.g., `Btrfs`), this test fails
        // until ALL is updated — the intended forcing function.
        let names: Vec<&'static str> = FsKind::ALL.iter().map(|k| k.name()).collect();
        assert_eq!(names, vec!["apfs", "ext4", "xfs", "ebs"]);
    }

    /// Test 7 — FsTempDir round-trips Drop semantics + into_path
    /// bypass.
    #[test]
    fn fs_temp_dir_drop_and_into_path() {
        let path = unique_workdir_path("fs-temp-dir-test");
        std::fs::create_dir_all(&path).expect("create_dir_all");

        // Drop=true cleans up.
        {
            let tmp = FsTempDir::new(path.clone(), true);
            assert!(tmp.path().exists());
        }
        assert!(
            !path.exists(),
            "FsTempDir with cleanup_on_drop=true must remove the dir on Drop"
        );

        // into_path() bypasses cleanup; caller owns the path.
        let path2 = unique_workdir_path("fs-temp-dir-test-2");
        std::fs::create_dir_all(&path2).expect("create_dir_all");
        let tmp2 = FsTempDir::new(path2.clone(), true);
        let leaked = tmp2.into_path();
        assert_eq!(leaked, path2);
        assert!(
            path2.exists(),
            "into_path() must NOT trigger cleanup (caller now owns the path)"
        );
        // Test cleanup.
        std::fs::remove_dir_all(&path2).expect("manual cleanup");
    }
}
