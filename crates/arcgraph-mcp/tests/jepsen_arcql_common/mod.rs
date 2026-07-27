//! W27-ν / ADR-163 — Jepsen-style invariant harness for the ArcQL
//! write-op surface (shared test module).
//!
//! Extends the ADR-047 founding bank-transfer Jepsen harness to the
//! ArcQL write-op family (CREATE / DELETE / SET / MERGE) introduced by
//! W26-θ Phase 1-5. Per `feedback_jepsen_isolation_discipline.md`
//! ("any v1.1+ work that adds a new transactional surface gets a
//! Jepsen-style test alongside, not just per-invariant proptests"),
//! the ArcQL write-op surface is a new transactional surface and
//! therefore gets a history-based isolation harness here.
//!
//! # Why this lives in `arcgraph-mcp/tests`, not `arcgraph-test-harness`
//!
//! The W27-ν spawn brief named `arcgraph-test-harness::jepsen::arcql`
//! as the home. That location is not buildable: the Jepsen + K-1
//! primitives live in **`arcgraph_storage::test_harness::{jepsen,k1}`**
//! (a `pub` module reachable cross-crate), and the *real* ArcQL→MVCC
//! execution surface is
//! [`arcgraph_mcp::storage::CrudExecutorSubstrate`] — the production
//! [`arcgraph_query::ExecutorSubstrate`] impl that drives each ArcQL
//! write-op through `begin → arcgraph_storage::crud::* → crud::commit`
//! against the real MVCC kernel. `arcgraph-test-harness` deliberately
//! does NOT depend on `arcgraph-storage` (its manifest defers that to
//! M5-08) and does not host `CrudExecutorSubstrate`. `arcgraph-mcp` is
//! the only crate where the full authentic path
//! (`Pipeline::build` → executor operators → `CrudExecutorSubstrate`
//! → real MVCC) is available, so the harness lives here. See ADR-163
//! §"Module placement" for the full rationale.
//!
//! # Two-tier design (both tiers exercise REAL MVCC; no stub)
//!
//! 1. **Rigorous history tier (crud-level).** The workloads drive the
//!    exact `begin → crud::create_node / delete_node_with_store →
//!    crud::commit` sequence that
//!    [`CrudExecutorSubstrate::create_node`] /
//!    [`CrudExecutorSubstrate::delete_node`] perform internally, but
//!    observe the commit LSN (which the [`ExecutorSubstrate`] trait
//!    discards). With the start-LSN (`tx.snapshot()`) and the commit
//!    LSN in hand, each op is recorded into the reused
//!    [`arcgraph_storage::test_harness::jepsen::history::OperationHistory`]
//!    and the [`checker`] runs Adya-2000-style snapshot-isolation
//!    anomaly detection over the *whole* history.
//! 2. **End-to-end executor tier.** A handful of transit pins drive
//!    the literal ArcQL operator surface
//!    (`parse → bind → type-check → lower → Pipeline::build →
//!    next_batch`) against a shared [`CrudExecutorSubstrate`] under
//!    concurrency, proving the operators preserve the coarse
//!    visibility / atomicity / dirty-read-freedom that the rigorous
//!    tier verifies at the layer below. The executor path cannot
//!    surface commit LSNs (the trait returns `NodeId` / `()`), so the
//!    rigorous checker runs on tier 1; closing that observability gap
//!    is forward-deferred (ADR-163 §"Forward-deferred").
//!
//! # Determinism contract (REQUIRED reading before editing)
//!
//! Jepsen workloads are **deliberately non-deterministic**. The
//! **CHECKER PREDICATE is the oracle**, NOT a binary-equal reference
//! snapshot. Per `feedback_determinism_oracle_concurrency_tests.md` +
//! `feedback_jepsen_isolation_discipline.md` §"Determinism contract":
//! the workload generators take an explicit seed so the per-client
//! `(op, target)` *generator* sequence is reproducible, but the
//! *interleaving* of those ops across worker threads is intentionally
//! racy so legitimate-but-rare orderings surface anomalies. A future
//! maintainer MUST NOT switch this to reference-equality — it would
//! false-fail under legitimate interleaving variance. When the checker
//! flags an anomaly, the printed history (sorted by commit LSN) is the
//! load-bearing reproduction artifact.
//!
//! # SI anomaly coverage (Bailis 2014 §3 / Adya 2000 §4)
//!
//! | Class | Workload | What the checker asserts |
//! |---|---|---|
//! | G0 (dirty write) | [`WorkloadKind::G0DirtyWrite`] — concurrent DELETE of shared nodes | each node is atomically present-or-tombstoned; no torn record |
//! | G1a (aborted read) | [`WorkloadKind::G1aAbortedRead`] — CREATE with injected aborts | no committed MATCH observes an aborted op's node |
//! | G1b (intermediate read) | [`WorkloadKind::G1bIntermediate`] — multi-write txns | no MATCH observes a partial subset of a multi-write tx |
//! | G1c (circular info flow) | [`WorkloadKind::G1cReadThenWrite`] — read-modify-write | the ww∪wr dependency graph is acyclic |
//! | G2-item (write skew) | [`WorkloadKind::G2WriteSkew`] — read-count-then-create | write skew is **permitted** under SI (reported as witness, NOT a violation) per Adya 2000 §4.3 |
//! | steady-state | [`WorkloadKind::SteadyState`] — CREATE + MATCH | every MATCH read-set == committed prefix visible at its snapshot |
//!
//! **G2-item is the cite-correctness anchor:** snapshot isolation does
//! NOT forbid write skew (that is the defining gap between SI and
//! serializability — Adya 2000 §4.3, Berenson 1995 A5B). The G2
//! workload therefore proves the surface *is* SI (write skew is
//! observable) while exhibiting no G1c, rather than asserting write
//! skew is prevented.
//!
//! # Fault injection
//!
//! - In-process abort injection (steady-state, always on via
//!   [`WorkloadKind::G1aAbortedRead`]): a configurable fraction of
//!   write ops are deliberately aborted (transaction dropped without
//!   commit, mirroring [`CrudExecutorSubstrate`]'s `discard_pending`
//!   error path). The G1a predicate proves the aborted writes are
//!   never observed — a genuine fault-injection regression test per
//!   `feedback_load_bearing_pr_requires_fault_injection_tests.md`.
//! - SIGKILL-during-commit subprocess + recovery variant
//!   (`JEPSEN_SIGKILL=1`): forward-deferred, NOT stubbed. The ADR-047
//!   founding bank-transfer harness deliberately deferred its SIGKILL
//!   variant rather than ship a no-op env-gated path (PR #344 R1 F-M3:
//!   an env-gate that constructs a fault context but never injects +
//!   asserts the steady-state property is a positive assertion on a
//!   no-op path). The same posture applies here: the SIGKILL variant
//!   activates alongside a durable-storage fixture (the v1.0-α
//!   `CrudExecutorSubstrate` fixture uses `InMemoryPageIo`, so there
//!   is no cross-process recovery to verify yet). See ADR-163
//!   §"Forward-deferred".
//!
//! # ADR provenance
//! - **ADR-163** (this harness) — extends ADR-047 to the ArcQL surface.
//! - **ADR-047** — founding bank-transfer Jepsen + K-1 primitive reuse.
//! - **ADR-016** — per-query consistency; **ADR-018** — MVCC.
//! - **ADR-031** — CommitBundle; **ADR-033** — rollback.
//! - **ADR-147..151** — W26-θ ArcQL write-op family under test.

// This module is `#[path]`-included by multiple integration-test
// binaries (read-side + write-side skeleton). Each binary uses a
// different subset, so per-binary dead-code is expected and benign.
#![allow(dead_code)]
// The op-recorder helpers thread the shared MVCC handles (client_id,
// op_id, mgr, crud, tenant, [label/threshold], history) explicitly —
// intrinsic to the Jepsen workload shape; a context struct would not
// improve a test harness's readability here.
#![allow(clippy::too_many_arguments)]

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};
use arcgraph_mcp::storage::CrudExecutorSubstrate;
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{self, CrudStore, PropertyData, crud_allocator_seed_handle};
use arcgraph_storage::io::{InMemoryPageIo, PosixPageIo};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::test_harness::jepsen::history::{OpBuilder, OperationHistory, RecordedOp};
use arcgraph_storage::transaction::{Transaction, TxnManager};
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
    RecordPageStoreHandle, WalConfig, WalWriter, recover_from_wal,
};
use bytes::Bytes;
use tempfile::TempDir;

pub mod checker;

/// Reserved history key marking a MATCH (read-only scan) op so the
/// checker can distinguish "a MATCH that observed zero nodes" from a
/// write-only op (both would otherwise record empty read/write sets).
/// Node ids are allocated from 1 upward and never reach `u64::MAX`, so
/// there is no collision with a real node key.
pub const SCAN_SENTINEL_KEY: u64 = u64::MAX;

/// Default workload seed. Distinguishable in logs ("JEPSARC1").
pub const DEFAULT_SEED: u64 = 0x4A45_5053_4152_4331;

/// `client_id` reserved for the serial seed phase. Sorts after every
/// worker (which use 0..clients) so seed ops are distinguishable in a
/// printed history.
pub const SEED_CLIENT_ID: u32 = u32::MAX;

/// Value marker for "this node exists / is visible". The exact bytes
/// are irrelevant — the checker reasons over node *identity* (key)
/// presence, not value — but a stable non-empty marker keeps recorded
/// reads/writes symmetric with the bank-transfer harness shape.
///
/// INVARIANT (load-bearing for the counter / node-identity partition):
/// this marker MUST NOT be exactly 8 bytes, or the checker's
/// [`checker::decode_counter`] would misclassify a node-identity write
/// as a counter write (an 8-byte big-endian counter payload), silently
/// moving it out of the G1c/write-skew predicates into the lost-update
/// predicate. The `const` assertion below pins the invariant at compile
/// time so a future edit to the marker shape cannot break the partition.
#[must_use]
pub fn present_marker() -> Bytes {
    // 1-byte marker; see the INVARIANT above (must never be 8 bytes).
    const PRESENT_MARKER: &[u8] = &[1u8];
    const _: () = assert!(
        PRESENT_MARKER.len() != 8,
        "present_marker() must not be 8 bytes — would alias the 8-byte counter payload \
         discriminant in checker::decode_counter and corrupt the counter/node-identity partition"
    );
    Bytes::from_static(PRESENT_MARKER)
}

// ─────────────────────────────────────────────────────────────────────
// Deterministic RNG (XorShift64) — zero new dep, seedable, matches the
// bank-transfer harness convention (`arcgraph-storage` does not depend
// on `rand`; neither does this harness need to).
// ─────────────────────────────────────────────────────────────────────

/// Tiny deterministic PRNG. Identical algorithm to the bank-transfer
/// workload's generator so the harnesses share their reproducibility
/// story.
#[derive(Debug, Clone)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Construct from a seed. A zero seed is remapped to a non-zero
    /// constant (XorShift64 is degenerate at 0).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// Next pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform in `[0, n)` (n > 0). Modulo bias is negligible for the
    /// small `n` this harness uses.
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next_u64() % n }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Fixture
// ─────────────────────────────────────────────────────────────────────

/// A fully-wired ArcQL→MVCC fixture: a `MultiTenantRouter` over a real
/// `CrudStore` + `TxnManager` + `InternTable`, plus the production
/// [`CrudExecutorSubstrate`] bound over them. The same `crud` + `mgr`
/// back both the crud-tier workloads (tier 1) and the substrate-driven
/// executor pins (tier 2), so a node created by either path is visible
/// to the other.
///
/// Construction mirrors [`CrudExecutorSubstrate`]'s own unit-test
/// fixture (`crates/arcgraph-mcp/src/storage/substrate.rs` `fixture()`).
pub struct JepsenArcqlFixture {
    pub substrate: CrudExecutorSubstrate,
    pub crud: Arc<CrudStore>,
    pub mgr: Arc<TxnManager>,
    pub router: Arc<MultiTenantRouter>,
    pub intern: Arc<InternTable>,
    pub tenant: TenantId,
}

impl JepsenArcqlFixture {
    /// Build a fresh fixture for `TenantId::DEFAULT`. Every call yields
    /// an independent MVCC kernel (clean node-id space).
    #[must_use]
    pub fn new() -> Self {
        let io = Arc::new(InMemoryPageIo::new());
        // 64 frames: node records live in the MVCC version chains
        // (CrudStore::new() leaves the primary index unset), so the
        // buffer pool is exercised only by catalog bootstrap; 64 is
        // ample headroom over the 8 the substrate unit test uses.
        let pool = BufferPool::new(64, io);
        let mgr = Arc::new(TxnManager::new());
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap catalog");
        let crud = Arc::new(CrudStore::new());
        let router = Arc::new(MultiTenantRouter::new(
            Arc::clone(&catalog),
            Arc::clone(&crud),
            None,
        ));
        let intern = Arc::new(InternTable::new());
        let substrate =
            CrudExecutorSubstrate::new(Arc::clone(&router), Arc::clone(&mgr), Arc::clone(&intern));
        Self {
            substrate,
            crud,
            mgr,
            router,
            intern,
            tenant: TenantId::DEFAULT,
        }
    }
}

impl Default for JepsenArcqlFixture {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Durable fixture (S7d-1 / ADR-182) — PosixPageIo + WAL + recover()
// ─────────────────────────────────────────────────────────────────────

/// Default catalog buffer-pool frame count for the durable fixture.
///
/// Matches the ADR-183 durable-bootstrap `POOL_FRAMES = 256`
/// (`crates/arcgraph-cli/src/bootstrap.rs:122`) so the durable Jepsen
/// fixture exercises the same catalog-pool sizing the production
/// `bootstrap_storage_backend` path does. The catalog page is the only
/// page the pool currently touches at v1.0 (`catalog.rs` `_pool` is
/// otherwise unused), so 256 is ample headroom either way.
pub const DURABLE_POOL_FRAMES: usize = 256;

/// File name of the [`PosixPageIo`]-backed page store inside the
/// fixture's data dir. Mirrors ADR-183's `PAGES_FILE = "pages.db"`
/// (`bootstrap.rs:125`).
pub const DURABLE_PAGES_FILE: &str = "pages.db";

/// WAL subdirectory name inside the fixture's data dir. Mirrors
/// ADR-183's `WAL_SUBDIR = "wal"` (`bootstrap.rs:128`).
pub const DURABLE_WAL_SUBDIR: &str = "wal";

/// A **durable** ArcQL→MVCC Jepsen fixture: a real on-disk
/// [`PosixPageIo`] over `<data_dir>/pages.db` + a real [`WalWriter`]
/// over `<data_dir>/wal` at [`DurabilityTier::Strict`]
/// (fsync-before-ack, the ADR-034 default tier), wired through the SAME
/// [`MultiTenantRouter`] + [`CrudExecutorSubstrate`] the in-memory
/// [`JepsenArcqlFixture`] uses — so ArcQL CREATE/MERGE/DELETE ops route
/// to durable storage and a crash leaves recoverable state.
///
/// # Why this exists (S7d-1 / ADR-182, #555)
///
/// The in-memory [`JepsenArcqlFixture`] is hard-wired to
/// [`InMemoryPageIo`] + `CrudStore::new()` (no WAL): a SIGKILL'd
/// subprocess loses ALL state, so a crash-atomicity checker over it
/// would be vacuous (PE-532 §1; the prior block). ADR-183 (#665)
/// retired that block by shipping a fully-wired durable substrate
/// (`PosixPageIo` + `WalWriter` + `recover_from_wal`) that is reachable
/// at the `CrudStore`/ArcQL level. This fixture is the FIRST consumer of
/// that durable surface in the `arcgraph-mcp` Jepsen test tree — it is
/// the foundation the S7d-2/3/4 follow-up slices (SIGKILL-during-MERGE
/// workload, recovery-reconciliation predicate, non-vacuity self-test)
/// build on.
///
/// # Construction mirrors the production durable stack
///
/// [`Self::build`] is a faithful port of ADR-183's `build_durable`
/// (`crates/arcgraph-cli/src/bootstrap.rs:339`) and the chaos-tests
/// durable crash-fixture precedent
/// into the `arcgraph-mcp` test surface — every underlying API
/// (`PosixPageIo`, `WalWriter`, `recover_from_wal`,
/// [`PageStoreTarget`], `CrudStore::new_with_index`,
/// `crud_allocator_seed_handle`) is `pub` from `arcgraph-storage` and
/// already consumed by this module, so this is a copy-the-pattern, not a
/// new-cross-crate-API task (the chaos-tests fixture lives in a
/// different crate and cannot be `use`d here — PE-555 §6 cross-crate
/// note).
///
/// # Determinism + hermeticity (ADR-160 §D-4)
///
/// Each fixture owns a [`TempDir`] (auto-cleaned on `Drop`); the WAL +
/// page store live under it. There is NO clock/rng read outside the
/// explicit op inputs — the Jepsen *workload* generators take an
/// explicit seed (see the module determinism contract), and the fixture
/// itself is fully deterministic given its inputs.
///
/// # Lifecycle
///
/// ```text
/// build(dir) ──CREATE/MERGE/DELETE via ArcQL──▶ commit (fsync, Strict)
///       │                                              │
///   shutdown the live WAL writer (or drop the stack)   │
///       │                                              ▼
///       └───────────────── recover(dir) ──────▶ re-open pages.db +
///                                                recover_from_wal into a
///                                                fresh CrudStore that
///                                                reads post-recovery
///                                                committed state.
/// ```
///
/// The `data_dir` MUST outlive both the live stack and its recovered
/// successor — hold the owning [`DurableJepsenWorkspace`] across the
/// whole build → crash → recover cycle.
pub struct DurableJepsenArcqlFixture {
    pub substrate: CrudExecutorSubstrate,
    pub crud: Arc<CrudStore>,
    pub mgr: Arc<TxnManager>,
    pub router: Arc<MultiTenantRouter>,
    pub intern: Arc<InternTable>,
    pub catalog: Arc<SystemCatalog>,
    pub tenant: TenantId,
    /// The live WAL writer. Held as `Option` so [`Self::shutdown_wal`]
    /// can drain + drop it (the graceful in-process crash proxy) while
    /// the rest of the stack stays alive. `recover()` re-spawns a fresh
    /// writer over the same dir.
    writer: Option<WalWriter>,
    /// The allocator backing this fixture's `CrudStore`. Recovery wires
    /// it as the [`AllocatorSeedHandle`] so v4-bundle `allocator_advances`
    /// replay (issue #129 P0) restores per-tenant id high-water marks.
    #[allow(dead_code)]
    allocator: Arc<PageAllocator>,
    /// The primary index. Held so it outlives the store/writer; recovery
    /// attaches its page-store handle to the replay target.
    #[allow(dead_code)]
    primary: Arc<PrimaryIndex>,
}

/// Owns the durable fixture's data directory ([`TempDir`]) + the
/// `wal/` + `pages.db` paths under it. Hold it alive across the
/// build → crash → recover cycle; `Drop` cleans the whole tree.
///
/// Mirrors the chaos-tests `ChaosWorkspace`
/// used by the storage recovery tests.
pub struct DurableJepsenWorkspace {
    _root: TempDir,
    data_dir: std::path::PathBuf,
}

impl DurableJepsenWorkspace {
    /// Create a fresh hermetic workspace with the `wal/` subdir created
    /// (the page file is created lazily by [`PosixPageIo::open_or_create`]
    /// at `build`/`recover` time).
    #[must_use]
    pub fn new() -> Self {
        let root = TempDir::new().expect("durable jepsen workspace tempdir");
        let data_dir = root.path().to_path_buf();
        std::fs::create_dir_all(data_dir.join(DURABLE_WAL_SUBDIR)).expect("mkdir wal");
        Self {
            _root: root,
            data_dir,
        }
    }

    /// The data directory the fixture's page store + WAL live under.
    #[must_use]
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}

impl Default for DurableJepsenWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Assemble a fresh durable stack (`PosixPageIo` + `WalWriter` +
/// fully-wired `CrudStore`) over `data_dir`, optionally running WAL
/// recovery first. Shared by [`DurableJepsenArcqlFixture::build`] (no
/// recovery — fresh dir) and [`DurableJepsenArcqlFixture::recover`]
/// (recover the WAL into the fresh MVCC store).
///
/// Faithful port of ADR-183 `build_durable` (`bootstrap.rs:339`)
/// steps §2–§9. The only knob is `do_recover`: `build` skips replay
/// (nothing is on disk yet); `recover` runs `recover_from_wal` +
/// `rebuild_all_tenant_stats` (the server-restart path).
///
/// Returns the assembled fixture plus the recovery watermark
/// (`recover_from_wal`'s `applied_commit_lsn` — the
/// `committed_fsync_watermark` boundary per ADR-034 §Slice-B): `Some(lsn)`
/// on the recover path, `None` on the build path (nothing was replayed).
/// [`DurableJepsenArcqlFixture::recover_with_watermark`] (S7d-2 / ADR-182
/// §2.2) surfaces it so the live SIGKILL-during-MERGE reconciliation can
/// pass the REAL recovered watermark to the predicate.
fn assemble_durable_stack(
    data_dir: &Path,
    do_recover: bool,
) -> (DurableJepsenArcqlFixture, Option<Lsn>) {
    let wal_dir = data_dir.join(DURABLE_WAL_SUBDIR);
    std::fs::create_dir_all(&wal_dir).expect("mkdir wal");
    let pages_path = data_dir.join(DURABLE_PAGES_FILE);

    // §2. File-backed page IO. PD#2: `PosixPageIo` is std `File`
    //     read/write + `sync_data()` (fdatasync / F_FULLFSYNC) — NOT
    //     mmap (`io.rs:190-192`). `open_or_create` is restart-safe.
    let io =
        Arc::new(PosixPageIo::open_or_create(&pages_path).expect("open durable jepsen page store"));
    let pool = BufferPool::new(DURABLE_POOL_FRAMES, io);

    // §3. WAL writer over `<data_dir>/wal`. Default `WalConfig` matches
    //     the ADR-183 bootstrap (`WalConfig::new` defaults: 1 ms group
    //     commit, 16-batch). The owning writer is stored on the fixture;
    //     its handle threads into the txn manager + crud store.
    let writer = WalWriter::spawn(WalConfig::new(&wal_dir)).expect("spawn durable jepsen WAL");
    let handle = writer.handle();

    // §4. Txn manager (WAL-backed) + catalog. Bootstrap BEFORE recover
    //     (canonical order). The catalog registers DEFAULT at
    //     `DurabilityTier::Strict` (ADR-034 default → fsync-before-ack);
    //     `set_durability_lookup` makes commits read that tier, so every
    //     CREATE/MERGE/DELETE on DEFAULT acks only after its WAL fsync.
    let mut mgr_inner = TxnManager::with_wal(handle.clone());
    let catalog = Arc::new(SystemCatalog::new());
    catalog
        .bootstrap(&pool, &mgr_inner)
        .expect("bootstrap catalog");
    mgr_inner.set_durability_lookup(catalog.clone());
    let mgr = Arc::new(mgr_inner);

    // §5. PrimaryIndex + CrudStore wired with the WAL (`Some(handle)`).
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(
            Arc::clone(&mgr),
            Arc::clone(&allocator),
            Some(handle.clone()),
        )
        .expect("durable jepsen primary index"),
    );
    let crud = Arc::new(CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        Arc::clone(&allocator),
    ));

    // §6–§8. On the recover path only: ADR-183 R1 fully-wired
    //         `PageStoreTarget` → `recover_from_wal` → M4-41 cold-start
    //         stats rebuild. On the build path the dir is fresh (nothing
    //         to replay), so we skip straight to the router.
    let mut applied_commit_lsn: Option<Lsn> = None;
    if do_recover {
        let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
            Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
        let records_handle: Arc<dyn RecordPageStoreHandle> = Arc::clone(
            crud.records()
                .expect("CrudStore::new_with_index exposes a record store"),
        )
            as Arc<dyn RecordPageStoreHandle>;
        let blob_handle: Arc<dyn BlobStoreHandle> =
            Arc::clone(crud.blob_store()) as Arc<dyn BlobStoreHandle>;
        let allocator_seed: Arc<dyn AllocatorSeedHandle> =
            crud_allocator_seed_handle(Arc::clone(&crud), Arc::clone(&allocator));
        let target = PageStoreTarget::primary_only(primary_handle)
            .with_record_store(records_handle)
            .with_blob_store(blob_handle)
            .with_allocator_seed(allocator_seed);
        let report = recover_from_wal(&wal_dir, Arc::clone(&mgr), target, None)
            .expect("durable jepsen WAL recovery");
        // The recovery watermark: exactly the commits at or below this LSN
        // survived (ADR-034 §Slice-B). S7d-2 passes it to the
        // recovery-reconciliation predicate as the acked-durable boundary.
        applied_commit_lsn = Some(report.applied_commit_lsn);
        let rebuild = arcgraph_storage::recovery::rebuild_all_tenant_stats(
            report.applied_commit_lsn,
            &mgr,
            &crud,
        );
        if !rebuild.failed.is_empty() {
            tracing::error!(
                target: "jepsen_arcql_durable",
                failed = ?rebuild.failed,
                "rebuild_all_tenant_stats reported per-tenant failures during durable jepsen recover"
            );
        }
    }

    // §9. Router + intern + substrate.
    let router = Arc::new(MultiTenantRouter::new(
        Arc::clone(&catalog),
        Arc::clone(&crud),
        None,
    ));
    let intern = Arc::new(InternTable::new());
    let substrate =
        CrudExecutorSubstrate::new(Arc::clone(&router), Arc::clone(&mgr), Arc::clone(&intern));

    let fixture = DurableJepsenArcqlFixture {
        substrate,
        crud,
        mgr,
        router,
        intern,
        catalog,
        tenant: TenantId::DEFAULT,
        writer: Some(writer),
        allocator,
        primary,
    };
    (fixture, applied_commit_lsn)
}

impl DurableJepsenArcqlFixture {
    /// Build a fresh durable fixture rooted at `data_dir`. The page
    /// store + WAL are created (empty); no recovery runs. Every CREATE /
    /// MERGE / DELETE driven through [`Self::substrate`] (or the
    /// crud-tier helpers) commits at [`DurabilityTier::Strict`]
    /// (fsync-before-ack), so the post-commit on-disk state is durable
    /// across a crash.
    #[must_use]
    pub fn build(data_dir: &Path) -> Self {
        assemble_durable_stack(data_dir, /* do_recover = */ false).0
    }

    /// Recover a fixture from the durable state at `data_dir`: re-open
    /// the same `pages.db`, re-spawn a WAL writer over the same `wal/`
    /// dir, and `recover_from_wal` into a FRESH `CrudStore` + MVCC store
    /// (then run the M4-41 cold-start stats rebuild). The returned
    /// fixture reads the post-recovery committed state — exactly the
    /// state a server restart would expose.
    ///
    /// This is the server-restart / crash-recovery path: it mirrors
    /// ADR-183 `build_durable` recover (`bootstrap.rs:454`) and the
    /// chaos-tests `ChaosStack::recover`
    /// used by the storage recovery tests. Call it after
    /// dropping (or [`Self::shutdown_wal`]-ing) the live fixture so the
    /// WAL is closed for re-open.
    #[must_use]
    pub fn recover(data_dir: &Path) -> Self {
        assemble_durable_stack(data_dir, /* do_recover = */ true).0
    }

    /// Recover from `data_dir` AND return the recovery **watermark** —
    /// `recover_from_wal`'s `applied_commit_lsn`, i.e. the
    /// `committed_fsync_watermark` boundary (ADR-034 §Slice-B): exactly the
    /// commits at or below it survived recovery. S7d-2 / ADR-182 §2.2: the
    /// live SIGKILL-during-MERGE reconciliation feeds this REAL boundary to
    /// [`checker::ArcqlSiChecker::reconcile_arcql_pending_with_recovery`] as
    /// the acked-durable threshold (an op `Committed{lsn ≤ watermark}` MUST
    /// be fully present; an op past the watermark may be absent). Same
    /// re-open path as [`Self::recover`]; only the watermark is additionally
    /// surfaced.
    #[must_use]
    pub fn recover_with_watermark(data_dir: &Path) -> (Self, Lsn) {
        let (fixture, applied) = assemble_durable_stack(data_dir, /* do_recover = */ true);
        (
            fixture,
            applied.expect("recover path always yields an applied_commit_lsn watermark"),
        )
    }

    /// Shut down ONLY this fixture's WAL writer (drain + drop), leaving
    /// `crud` / `mgr` / `catalog` alive. This is the graceful in-process
    /// crash proxy: the writer's `Drop` drains the fsync queue, so an
    /// acked Strict commit is on disk before [`Self::recover`] re-opens
    /// the dir. (A true SIGKILL — "no Drop runs" — is the S7d-2 subprocess
    /// follow-up; for the S7d-1 smoke, graceful shutdown is the
    /// K-1-accepted recovery proxy, mirroring `ChaosStack::shutdown_wal`.)
    /// Idempotent.
    pub fn shutdown_wal(&mut self) {
        if let Some(w) = self.writer.take() {
            let _ = w.shutdown();
        }
    }

    /// The crud store (write-op entry surface).
    #[must_use]
    pub fn crud(&self) -> &Arc<CrudStore> {
        &self.crud
    }

    /// The transaction manager.
    #[must_use]
    pub fn mgr(&self) -> &Arc<TxnManager> {
        &self.mgr
    }
}

// ─────────────────────────────────────────────────────────────────────
// ArcQL operation encoder
// ─────────────────────────────────────────────────────────────────────

/// The ArcQL operations the harness encodes onto the MVCC kernel. Each
/// variant names the ArcQL surface it stands for + the
/// [`CrudExecutorSubstrate`] method it lowers through. Per ADR-163 D-1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArcqlOp {
    /// `CREATE (:Label)` → [`CrudExecutorSubstrate::create_node`] →
    /// `begin → crud::create_node → crud::commit`.
    CreateNode { label: LabelId },
    /// `CREATE (:Label), (:Label)` in one statement-scoped tx (the
    /// W26-θ Phase 5 batched shape) → two `crud::create_node` calls
    /// under one transaction. Used to exercise G1b.
    CreateNodes { label: LabelId, count: u32 },
    /// `MATCH (n) DELETE n` → [`CrudExecutorSubstrate::delete_node`] →
    /// `begin → crud::delete_node_with_store → crud::commit`.
    DeleteNode { node: NodeId },
    /// `MATCH (n) RETURN n` → [`CrudExecutorSubstrate::scan_nodes`] →
    /// `begin → read_node(1..=high_water) → drop`.
    Match,
    /// Read-modify-write: `MATCH (n) ... CREATE (:Label)` in one tx
    /// (the scan and the create share a snapshot). Exercises wr/rw
    /// dependency edges for G1c / G2.
    ReadThenCreate { label: LabelId },
    /// Deliberately-aborted `CREATE` (fault injection): allocate +
    /// stage the node then drop the tx without committing. The node id
    /// is burned but never becomes visible.
    AbortedCreateNode { label: LabelId },
}

// ─────────────────────────────────────────────────────────────────────
// Op recorders (crud-tier; observe real LSNs)
// ─────────────────────────────────────────────────────────────────────

/// Enumerate the node ids visible to `tx`'s snapshot, mirroring
/// [`CrudExecutorSubstrate::scan_nodes`]'s `1..=high_water` walk +
/// `crud::read_node` MVCC visibility filter.
fn visible_node_ids(crud: &CrudStore, tx: &Transaction<'_>, tenant: TenantId) -> Vec<u64> {
    let high_water = crud.node_high_water(tenant);
    let mut out = Vec::new();
    for raw in 1..=high_water {
        match crud::read_node(tx, NodeId::new(raw)) {
            Ok(Some(_)) => out.push(raw),
            Ok(None) => {}
            // A torn / unreadable record at a committed snapshot is a
            // G0-class corruption — surface it as a poisoned read so
            // the checker fails loud rather than silently dropping it.
            Err(_) => out.push(POISONED_READ_MARKER),
        }
    }
    out
}

/// Sentinel pushed into a MATCH's observed set when `read_node`
/// returns a decode error (torn write). The checker treats its
/// presence as a hard G0 violation.
pub const POISONED_READ_MARKER: u64 = u64::MAX - 1;

/// Record a `CREATE (:label)` op. One tx: begin → create_node →
/// commit. Returns the new node id on commit.
fn record_create_node(
    client_id: u32,
    op_id: u64,
    mgr: &TxnManager,
    crud: &CrudStore,
    tenant: TenantId,
    label: LabelId,
    history: &OperationHistory,
) -> Option<NodeId> {
    let mut tx = mgr.begin(tenant);
    let start = tx.snapshot();
    let mut builder = OpBuilder::new(client_id, op_id, tenant, start);
    let node_id = match crud::create_node(crud, &mut tx, tenant, label, &PropertyData::Empty) {
        Ok(id) => id,
        Err(_) => {
            crud.discard_pending(tx.id());
            crud.discard_pending_installs(tx.id());
            history.push(builder.into_aborted());
            return None;
        }
    };
    builder.intend_write(node_id.raw(), Some(present_marker()));
    match crud::commit(tx, crud) {
        Ok(lsn) => {
            history.push(builder.into_committed(lsn));
            Some(node_id)
        }
        Err(_) => {
            history.push(builder.into_aborted());
            None
        }
    }
}

/// Record a batched `CREATE (:label), (:label), ...` op: `count`
/// nodes created under ONE transaction (the W26-θ Phase 5 statement-
/// scoped shape). All nodes share one commit LSN, so a MATCH must
/// observe all-or-none — the G1b intermediate-read invariant.
fn record_create_nodes(
    client_id: u32,
    op_id: u64,
    mgr: &TxnManager,
    crud: &CrudStore,
    tenant: TenantId,
    label: LabelId,
    count: u32,
    history: &OperationHistory,
) {
    let mut tx = mgr.begin(tenant);
    let start = tx.snapshot();
    let mut builder = OpBuilder::new(client_id, op_id, tenant, start);
    for _ in 0..count {
        match crud::create_node(crud, &mut tx, tenant, label, &PropertyData::Empty) {
            Ok(id) => builder.intend_write(id.raw(), Some(present_marker())),
            Err(_) => {
                crud.discard_pending(tx.id());
                crud.discard_pending_installs(tx.id());
                history.push(builder.into_aborted());
                return;
            }
        }
    }
    match crud::commit(tx, crud) {
        Ok(lsn) => history.push(builder.into_committed(lsn)),
        Err(_) => history.push(builder.into_aborted()),
    }
}

/// Record a `MATCH (n) DELETE n` op. One tx: begin →
/// delete_node_with_store → commit. The written value is `None`
/// (tombstone).
fn record_delete_node(
    client_id: u32,
    op_id: u64,
    mgr: &TxnManager,
    crud: &CrudStore,
    tenant: TenantId,
    node: NodeId,
    history: &OperationHistory,
) {
    let mut tx = mgr.begin(tenant);
    let start = tx.snapshot();
    let mut builder = OpBuilder::new(client_id, op_id, tenant, start);
    if crud::delete_node_with_store(crud, &mut tx, node).is_err() {
        crud.discard_pending(tx.id());
        crud.discard_pending_installs(tx.id());
        history.push(builder.into_aborted());
        return;
    }
    builder.intend_write(node.raw(), None);
    match crud::commit(tx, crud) {
        Ok(lsn) => history.push(builder.into_committed(lsn)),
        Err(_) => history.push(builder.into_aborted()),
    }
}

/// Record a `MATCH (n) RETURN n` op. Read-only snapshot scan. Records
/// the scan sentinel + one observed read per visible node.
fn record_match(
    client_id: u32,
    op_id: u64,
    mgr: &TxnManager,
    crud: &CrudStore,
    tenant: TenantId,
    history: &OperationHistory,
) {
    let tx = mgr.begin(tenant);
    let start = tx.snapshot();
    let mut builder = OpBuilder::new(client_id, op_id, tenant, start);
    builder.observe_read(SCAN_SENTINEL_KEY, Some(present_marker()));
    for raw in visible_node_ids(crud, &tx, tenant) {
        builder.observe_read(raw, Some(present_marker()));
    }
    // A read-only op has no commit LSN of its own; its logical
    // timestamp is its snapshot. Recording it as committed-at-snapshot
    // keeps the `Committed ⇒ commit_lsn.is_some()` invariant the
    // reused history primitive documents, and the checker identifies
    // read ops by the scan sentinel, not by the LSN.
    history.push(builder.into_committed(start));
}

/// Record a read-modify-write op in ONE transaction: scan the visible
/// set (recording reads), then create a new node (recording the
/// write), then commit. This is the op shape that produces wr / rw
/// dependency edges, exercising G1c and G2.
///
/// When `gate_threshold` is `Some(t)`, the create is performed only if
/// the observed node count is `< t` — the canonical write-skew shape
/// (two concurrent ops both read count `t-1`, both create, total `t+1`).
fn record_read_then_create(
    client_id: u32,
    op_id: u64,
    mgr: &TxnManager,
    crud: &CrudStore,
    tenant: TenantId,
    label: LabelId,
    gate_threshold: Option<u64>,
    history: &OperationHistory,
) {
    let mut tx = mgr.begin(tenant);
    let start = tx.snapshot();
    let mut builder = OpBuilder::new(client_id, op_id, tenant, start);
    builder.observe_read(SCAN_SENTINEL_KEY, Some(present_marker()));
    let visible = visible_node_ids(crud, &tx, tenant);
    for raw in &visible {
        builder.observe_read(*raw, Some(present_marker()));
    }
    let gate_open = match gate_threshold {
        Some(t) => (visible.len() as u64) < t,
        None => true,
    };
    if gate_open {
        match crud::create_node(crud, &mut tx, tenant, label, &PropertyData::Empty) {
            Ok(id) => builder.intend_write(id.raw(), Some(present_marker())),
            Err(_) => {
                crud.discard_pending(tx.id());
                crud.discard_pending_installs(tx.id());
                history.push(builder.into_aborted());
                return;
            }
        }
    }
    match crud::commit(tx, crud) {
        Ok(lsn) => history.push(builder.into_committed(lsn)),
        Err(_) => history.push(builder.into_aborted()),
    }
}

/// Record a deliberately-aborted CREATE (fault injection). It allocates
/// and stages the node, then drops the tx without committing, mirroring
/// [`CrudExecutorSubstrate`]'s `discard_pending` error path. The
/// burned node id is recorded as an aborted write so the checker can
/// prove it is never observed (G1a).
fn record_aborted_create(
    client_id: u32,
    op_id: u64,
    mgr: &TxnManager,
    crud: &CrudStore,
    tenant: TenantId,
    label: LabelId,
    history: &OperationHistory,
) {
    let mut tx = mgr.begin(tenant);
    let start = tx.snapshot();
    let mut builder = OpBuilder::new(client_id, op_id, tenant, start);
    if let Ok(id) = crud::create_node(crud, &mut tx, tenant, label, &PropertyData::Empty) {
        builder.intend_write(id.raw(), Some(present_marker()));
    }
    // Genuine abort: discard staged work, drop the tx (Drop aborts the
    // MVCC transaction). NO commit — the node never becomes visible.
    crud.discard_pending(tx.id());
    crud.discard_pending_installs(tx.id());
    drop(tx);
    history.push(builder.into_aborted());
}

// ─────────────────────────────────────────────────────────────────────
// Workloads
// ─────────────────────────────────────────────────────────────────────

/// Which anomaly class a workload run targets. See the module-level
/// coverage table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadKind {
    /// CREATE + MATCH interleaved (snapshot-read consistency).
    SteadyState,
    /// Concurrent DELETE of shared seeded nodes (dirty write).
    G0DirtyWrite,
    /// CREATE with a fraction of ops aborted (aborted read).
    G1aAbortedRead,
    /// Multi-write txns + MATCH (intermediate read).
    G1bIntermediate,
    /// Read-modify-write + MATCH (circular information flow).
    G1cReadThenWrite,
    /// Read-count-then-create + MATCH (write skew; SI-permitted).
    G2WriteSkew,
}

/// Workload configuration. All fields public so a test can override
/// individual knobs.
#[derive(Debug, Clone, Copy)]
pub struct WorkloadConfig {
    pub clients: u32,
    pub ops_per_client: u64,
    pub seed: u64,
    pub tenant: TenantId,
    /// Nodes seeded serially before the workload (for DELETE-heavy /
    /// read-heavy classes). Ignored by CREATE-only classes.
    pub seed_nodes: u32,
    /// Write-skew gate threshold (G2 only).
    pub skew_threshold: u64,
    /// 1-in-N write ops aborted (G1a only; 0 = never).
    pub abort_one_in: u32,
    /// Nodes created per batched op (G1b only).
    pub batch_size: u32,
    /// Label stamped on created nodes.
    pub label: LabelId,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            clients: 4,
            ops_per_client: 50,
            seed: DEFAULT_SEED,
            tenant: TenantId::DEFAULT,
            seed_nodes: 20,
            skew_threshold: 8,
            abort_one_in: 4,
            batch_size: 3,
            label: LabelId::new(1),
        }
    }
}

/// Seed `cfg.seed_nodes` nodes serially before the workload, recording
/// each as a committed CREATE under [`SEED_CLIENT_ID`]. Returns the
/// ids created. The seed phase is the equivalent of Bailis 2014's
/// "initial state setup".
pub fn seed_nodes(
    mgr: &TxnManager,
    crud: &CrudStore,
    cfg: &WorkloadConfig,
    history: &OperationHistory,
) -> Vec<NodeId> {
    let mut ids = Vec::with_capacity(cfg.seed_nodes as usize);
    for i in 0..cfg.seed_nodes {
        if let Some(id) = record_create_node(
            SEED_CLIENT_ID,
            u64::from(i),
            mgr,
            crud,
            cfg.tenant,
            cfg.label,
            history,
        ) {
            ids.push(id);
        }
    }
    ids
}

/// Run a workload of `kind` to completion across `cfg.clients` worker
/// threads, populating `history`. Blocks until every worker joins.
///
/// The interleaving across threads is intentionally non-deterministic
/// (see the module-level determinism contract); only the per-client
/// generator sequence is seed-reproducible.
pub fn run_workload(
    fixture: &JepsenArcqlFixture,
    kind: WorkloadKind,
    cfg: WorkloadConfig,
    history: Arc<OperationHistory>,
) {
    // Seeding (DELETE / read-heavy classes need a non-empty graph).
    let seeded: Vec<u64> = if matches!(
        kind,
        WorkloadKind::G0DirtyWrite | WorkloadKind::G1cReadThenWrite | WorkloadKind::G2WriteSkew
    ) {
        seed_nodes(&fixture.mgr, &fixture.crud, &cfg, &history)
            .into_iter()
            .map(|n| n.raw())
            .collect()
    } else {
        Vec::new()
    };
    let seeded = Arc::new(seeded);

    let global_op = Arc::new(AtomicU64::new(0));
    let handles: Vec<_> = (0..cfg.clients)
        .map(|client_id| {
            let mgr = Arc::clone(&fixture.mgr);
            let crud = Arc::clone(&fixture.crud);
            let history = Arc::clone(&history);
            let global_op = Arc::clone(&global_op);
            let seeded = Arc::clone(&seeded);
            // Per-client seed derived via the xxHash64 avalanche
            // multiplier so clients are independent yet the whole run
            // is reproducible from `cfg.seed`.
            let client_seed = cfg
                .seed
                .wrapping_add(u64::from(client_id).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            thread::Builder::new()
                .name(format!("jepsen-arcql-{kind:?}-client-{client_id}"))
                .spawn(move || {
                    run_client(
                        client_id,
                        client_seed,
                        kind,
                        &cfg,
                        &mgr,
                        &crud,
                        &history,
                        &global_op,
                        &seeded,
                    );
                })
                .expect("spawn jepsen-arcql worker thread")
        })
        .collect();

    for h in handles {
        h.join().expect("jepsen-arcql worker thread panicked");
    }
}

#[allow(clippy::too_many_arguments)]
fn run_client(
    client_id: u32,
    seed: u64,
    kind: WorkloadKind,
    cfg: &WorkloadConfig,
    mgr: &TxnManager,
    crud: &CrudStore,
    history: &OperationHistory,
    global_op: &AtomicU64,
    seeded: &[u64],
) {
    let mut rng = XorShift64::new(seed);
    for _ in 0..cfg.ops_per_client {
        let op_id = global_op.fetch_add(1, Ordering::Relaxed);
        match kind {
            WorkloadKind::SteadyState => {
                // ~50/50 CREATE vs MATCH.
                if rng.below(2) == 0 {
                    record_create_node(client_id, op_id, mgr, crud, cfg.tenant, cfg.label, history);
                } else {
                    record_match(client_id, op_id, mgr, crud, cfg.tenant, history);
                }
            }
            WorkloadKind::G0DirtyWrite => {
                // Delete a random seeded node (shared across clients →
                // WW conflict) or MATCH.
                if rng.below(3) == 0 || seeded.is_empty() {
                    record_match(client_id, op_id, mgr, crud, cfg.tenant, history);
                } else {
                    let idx = rng.below(seeded.len() as u64) as usize;
                    record_delete_node(
                        client_id,
                        op_id,
                        mgr,
                        crud,
                        cfg.tenant,
                        NodeId::new(seeded[idx]),
                        history,
                    );
                }
            }
            WorkloadKind::G1aAbortedRead => {
                // A fraction of write ops are aborted; the rest commit;
                // MATCH observes. The checker proves aborted writes are
                // never observed.
                let roll = rng.below(3);
                if roll == 0 {
                    record_match(client_id, op_id, mgr, crud, cfg.tenant, history);
                } else if cfg.abort_one_in > 0 && rng.below(u64::from(cfg.abort_one_in)) == 0 {
                    record_aborted_create(
                        client_id, op_id, mgr, crud, cfg.tenant, cfg.label, history,
                    );
                } else {
                    record_create_node(client_id, op_id, mgr, crud, cfg.tenant, cfg.label, history);
                }
            }
            WorkloadKind::G1bIntermediate => {
                if rng.below(2) == 0 {
                    record_create_nodes(
                        client_id,
                        op_id,
                        mgr,
                        crud,
                        cfg.tenant,
                        cfg.label,
                        cfg.batch_size.max(2),
                        history,
                    );
                } else {
                    record_match(client_id, op_id, mgr, crud, cfg.tenant, history);
                }
            }
            WorkloadKind::G1cReadThenWrite => {
                if rng.below(3) == 0 {
                    record_match(client_id, op_id, mgr, crud, cfg.tenant, history);
                } else {
                    record_read_then_create(
                        client_id, op_id, mgr, crud, cfg.tenant, cfg.label, None, history,
                    );
                }
            }
            WorkloadKind::G2WriteSkew => {
                if rng.below(3) == 0 {
                    record_match(client_id, op_id, mgr, crud, cfg.tenant, history);
                } else {
                    record_read_then_create(
                        client_id,
                        op_id,
                        mgr,
                        crud,
                        cfg.tenant,
                        cfg.label,
                        Some(cfg.skew_threshold),
                        history,
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Live cross-check helper
// ─────────────────────────────────────────────────────────────────────

/// Read the live visible node count at the latest snapshot. Used by
/// tests to cross-check the recorded history against the actual store
/// state (catches "the history says one thing, the store says
/// another"), mirroring the bank-transfer test's `live_sum` check.
#[must_use]
pub fn live_visible_count(fixture: &JepsenArcqlFixture) -> usize {
    let tx = fixture.mgr.begin(fixture.tenant);
    visible_node_ids(&fixture.crud, &tx, fixture.tenant)
        .into_iter()
        .filter(|&n| n != POISONED_READ_MARKER)
        .count()
}

/// Durable-fixture analog of [`live_visible_count`]: the live visible
/// node count at the latest snapshot of a [`DurableJepsenArcqlFixture`]
/// (used by the S7d-1 recover() smoke to cross-check the recovered store
/// state against the executor-driven MATCH count).
#[must_use]
pub fn live_visible_count_durable(fixture: &DurableJepsenArcqlFixture) -> usize {
    let tx = fixture.mgr.begin(fixture.tenant);
    visible_node_ids(&fixture.crud, &tx, fixture.tenant)
        .into_iter()
        .filter(|&n| n != POISONED_READ_MARKER)
        .count()
}

/// Count committed vs aborted ops in a drained history (telemetry +
/// commit-rate-floor assertions).
#[must_use]
pub fn outcome_counts(ops: &[RecordedOp]) -> (usize, usize) {
    use arcgraph_storage::test_harness::jepsen::history::OpOutcome;
    let committed = ops
        .iter()
        .filter(|o| matches!(o.outcome, OpOutcome::Committed))
        .count();
    let aborted = ops
        .iter()
        .filter(|o| matches!(o.outcome, OpOutcome::Aborted))
        .count();
    (committed, aborted)
}

/// True iff a recorded op is a MATCH (carries the scan sentinel).
#[must_use]
pub fn is_match_op(op: &RecordedOp) -> bool {
    op.reads.iter().any(|r| r.key == SCAN_SENTINEL_KEY)
}

/// Build a `PartitionId::ZERO` execution context for executor transit
/// pins.
#[must_use]
pub fn exec_partition() -> PartitionId {
    PartitionId::ZERO
}

// ─────────────────────────────────────────────────────────────────────
// W27-ν-2 write-side property round-trip helpers (ADR-163 §FD-1)
//
// These thread a single integer `counter` property through the real
// ADR-152 property-bag persistence path (`properties_to_property_data`
// → `crud::*` → `record_property_bag`), so the lost-update workload
// exercises the SAME blob round-trip the substrate `create_node` /
// `set_node` use — not a private side-channel. The crud tier observes
// real commit LSNs (which the `ExecutorSubstrate` trait discards), so
// the recorded history is checkable by `ArcqlSiChecker::check`.
// ─────────────────────────────────────────────────────────────────────

/// Property key for the lost-update counter workload.
pub const COUNTER_PROP: &str = "counter";

/// Encode a single `{counter: value}` bag into a [`PropertyData`] via
/// the production ADR-152 §D-1 helper (canonical JSON → `Blob`). This is
/// the exact serializer the substrate `create_node` / `set_node` use, so
/// the crud-tier counter workload rides the real persistence path.
#[must_use]
pub fn counter_property_data(value: i64) -> PropertyData {
    use arcgraph_query::executor::value::Value;
    arcgraph_mcp::storage::property_payload::properties_to_property_data(&[(
        COUNTER_PROP.to_string(),
        Value::Integer(value),
    )])
}

/// Read the `counter` property of `node` at `tx`'s snapshot via the
/// real ADR-152 §D-3 decode path. Returns `None` when the node is not
/// visible or carries no `counter` key.
#[must_use]
pub fn read_counter(
    crud: &CrudStore,
    intern: &InternTable,
    tx: &Transaction<'_>,
    tenant: TenantId,
    node: NodeId,
) -> Option<i64> {
    use arcgraph_query::executor::value::Value;
    let rec = match crud::read_node(tx, node) {
        Ok(Some(r)) => r,
        _ => return None,
    };
    // v2 M2 checked read (typed-or-legacy dispatch); a decode fault
    // in a Jepsen workload is a hard harness failure, never a silent
    // None (the loud-corruption contract).
    let bag = arcgraph_mcp::storage::property_payload::record_property_bag_checked(
        &rec,
        crud.blob_store(),
        intern,
        tenant,
    )
    .expect("jepsen read_counter: property payload decode must succeed");
    match bag.get(COUNTER_PROP) {
        Some(Value::Integer(v)) => Some(*v),
        _ => None,
    }
}

/// Seed a single counter node initialized to `seed`, committed serially
/// under [`SEED_CLIENT_ID`]. Records the CREATE into `history` with the
/// seed value tagged via [`checker::encode_counter`] so the lost-update
/// predicate can recover the seed. Returns the new node id.
#[must_use]
pub fn seed_counter_node(
    mgr: &TxnManager,
    crud: &CrudStore,
    tenant: TenantId,
    seed: i64,
    history: &OperationHistory,
) -> NodeId {
    let mut tx = mgr.begin(tenant);
    let start = tx.snapshot();
    let mut builder = OpBuilder::new(SEED_CLIENT_ID, 0, tenant, start);
    let node = crud::create_node(
        crud,
        &mut tx,
        tenant,
        LabelId::new(1),
        &counter_property_data(seed),
    )
    .expect("seed counter create");
    builder.intend_write(node.raw(), Some(checker::encode_counter(seed as u64)));
    let lsn = crud::commit(tx, crud).expect("seed counter commit");
    history.push(builder.into_committed(lsn));
    node
}

/// One read-modify-write increment on `node`'s `counter` property in a
/// SINGLE transaction (begin → read counter → write counter+1 →
/// commit). This is the genuine atomic-RMW shape: the read and the
/// write share one transaction, so a concurrent writer that commits
/// first makes THIS commit lose the OCC WW race
/// ([`arcgraph_storage::crud::commit`] → `MvccConflict`). On a
/// committed increment the new value is recorded via
/// [`checker::encode_counter`]; on an aborted (OCC-lost) increment the
/// op is recorded aborted and the value is NOT applied.
///
/// Returns `true` iff the increment committed.
#[allow(clippy::too_many_arguments)] // jepsen worker context set (v2 M2 adds the intern table)
pub fn record_counter_increment(
    client_id: u32,
    op_id: u64,
    mgr: &TxnManager,
    crud: &CrudStore,
    intern: &InternTable,
    tenant: TenantId,
    node: NodeId,
    history: &OperationHistory,
) -> bool {
    let mut tx = mgr.begin(tenant);
    let start = tx.snapshot();
    let mut builder = OpBuilder::new(client_id, op_id, tenant, start);

    // READ (in-tx): the current counter value at this tx's snapshot.
    let current = match read_counter(crud, intern, &tx, tenant, node) {
        Some(v) => v,
        None => {
            crud.discard_pending(tx.id());
            crud.discard_pending_installs(tx.id());
            history.push(builder.into_aborted());
            return false;
        }
    };
    builder.observe_read(node.raw(), Some(checker::encode_counter(current as u64)));

    // MODIFY + WRITE (same tx): counter := current + 1.
    let next = current + 1;
    if crud::update_node(crud, &mut tx, node, &counter_property_data(next)).is_err() {
        crud.discard_pending(tx.id());
        crud.discard_pending_installs(tx.id());
        history.push(builder.into_aborted());
        return false;
    }
    builder.intend_write(node.raw(), Some(checker::encode_counter(next as u64)));

    // COMMIT: OCC rejects the second of two writers that read the same
    // version (the lost-update defence).
    match crud::commit(tx, crud) {
        Ok(lsn) => {
            history.push(builder.into_committed(lsn));
            true
        }
        Err(_) => {
            history.push(builder.into_aborted());
            false
        }
    }
}

/// Run `clients` worker threads, each attempting `increments_per_client`
/// counter RMW increments on the shared `node`, populating `history`.
/// Returns when every worker joins. The interleaving is intentionally
/// racy (the determinism contract) so OCC conflicts arise naturally;
/// the lost-update INVARIANT (final == seed + committed count) holds
/// over any interleaving.
pub fn run_counter_workload(
    fixture: &JepsenArcqlFixture,
    node: NodeId,
    clients: u32,
    increments_per_client: u64,
    history: Arc<OperationHistory>,
) {
    let global_op = Arc::new(AtomicU64::new(1));
    let handles: Vec<_> = (0..clients)
        .map(|client_id| {
            let mgr = Arc::clone(&fixture.mgr);
            let crud = Arc::clone(&fixture.crud);
            let intern = Arc::clone(&fixture.intern);
            let history = Arc::clone(&history);
            let global_op = Arc::clone(&global_op);
            let tenant = fixture.tenant;
            thread::Builder::new()
                .name(format!("jepsen-arcql-counter-client-{client_id}"))
                .spawn(move || {
                    for _ in 0..increments_per_client {
                        let op_id = global_op.fetch_add(1, Ordering::Relaxed);
                        // Retry the RMW until it commits, so every
                        // intended increment is eventually APPLIED — the
                        // workload models "N increments must all land",
                        // and OCC losers retry rather than vanish. Each
                        // attempt (committed or aborted) is recorded, so
                        // the history shows the real abort rate.
                        loop {
                            if record_counter_increment(
                                client_id, op_id, &mgr, &crud, &intern, tenant, node, &history,
                            ) {
                                break;
                            }
                        }
                    }
                })
                .expect("spawn jepsen-arcql counter worker")
        })
        .collect();
    for h in handles {
        h.join().expect("counter worker panicked");
    }
}

/// Count COMMITTED counter increments recorded in a drained history
/// (the seed CREATE under [`SEED_CLIENT_ID`] is excluded). Used by the
/// live test to cross-check the scanned final value against the number
/// of increments that actually committed.
#[must_use]
pub fn committed_increment_count(ops: &[RecordedOp]) -> u64 {
    use arcgraph_storage::test_harness::jepsen::history::OpOutcome;
    ops.iter()
        .filter(|o| o.outcome == OpOutcome::Committed && o.client_id != SEED_CLIENT_ID)
        .filter(|o| o.writes.iter().any(|w| w.value.is_some()))
        .count() as u64
}
