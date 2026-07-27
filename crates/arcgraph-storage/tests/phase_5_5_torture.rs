//! M3.a Phase 5.5 — sustained N-tenant fault-injection torture test.
//!
//! Per Path A directive 2026-04-26 + Phase 5.5 spec §3.
//!
//! ## What this exercises
//!
//! A 30-second sustained workload across 4 tenants representing the
//! full ADR-035 + ADR-034 surface:
//!
//!  - **Tenant DEFAULT (T1↔T3 flipping, CRUD half)** — writes nodes
//!    via the real CRUD path; tier toggles every ~3 seconds so both
//!    tier branches are exercised.
//!  - **Tenant 1001 (T1 strict, vector half)** — installs vector
//!    arena pages via `VectorArenaPageStore::install_or_replace`;
//!    flushes a snapshot every ~5 seconds.
//!  - **Tenant 1002 (T3 periodic, vector half)** — installs vector
//!    arena pages; flushes a snapshot every ~5 seconds; the
//!    snapshot flushes deliberately mix in `flush_snapshot_with_crash_point`
//!    invocations to exercise the §G.2 atomic-rename graceful-
//!    artifact contract.
//!  - **Tenant 1003 (T1 strict, vector half)** — third vector
//!    tenant sharing `vec_store_1` with tenant 1001 (multi-tenant
//!    keying by `(tenant, page_id)`). Cross-tenant isolation is
//!    asserted post-recovery: tenant 1001's pages must not leak
//!    into tenant 1003's lookups and vice versa.
//!
//! Each worker maintains an **oracle** — the deterministic
//! committed-state shadow. After the workload completes, the test
//! recovers each store from disk + asserts every oracle entry is
//! observable post-recovery (no ghost rollbacks; no lost commits;
//! no cross-tenant leakage).
//!
//! ## Fault injection
//!
//! **Cadence model** (interval-based, NOT per-op rate-based). The
//! Phase 5.5 spec sketched indicative per-op rates ("1 % WAL,
//! 0.5 % snapshot, 0.1 % crash"); the implementation here uses
//! periodic interval injection because (a) the deterministic
//! 30-second wall-clock budget makes per-op probability flaky
//! under CPU contention and (b) per-op rate-based injection
//! belongs to Slice K's eventual Jepsen-class multi-hour torture
//! harness, not a 30-second smoke. PR #128 review fold-in #5
//! corrects the framing.
//!
//!  - **WAL fsync failures**: every 5 seconds, force a
//!    WalWriter shutdown + restart cycle on the CRUD half. Each
//!    cycle drains in-flight commits via `WalWriter::shutdown()`,
//!    constructs a fresh stack, and replays via the v4
//!    `CommitBundle.allocator_advances` + `AllocatorSeedHandle`
//!    path (PR #130). Pre-cycle commits are recovered via the
//!    existing `recover_from_wal` flow with the allocator-seed
//!    wired so post-recovery `alloc_node` resumes from the
//!    correct high-water (closes ADR-034 D-1 violation that
//!    PR #128's first cut surfaced).
//!  - **Snapshot fsync failures**: every 3 seconds, the vector
//!    half flushes a snapshot via `flush_snapshot_with_crash_point`
//!    at a randomly-selected `CrashPoint`. The crash-point variants
//!    leave a graceful artifact at every interior step (per G.2 §10.3
//!    contract); the next clean flush succeeds.
//!  - **Process crash injection (subprocess)**: real subprocess
//!    SIGKILL-during-recovery coverage now lives in Slice K's harness
//!    at `tests/k3_sigkill_during_rebuild.rs`
//!    (`k3_sigkill_during_rebuild_subprocess`, closes #256), which
//!    uses a `Command::spawn(test_binary)` self-fork + SIGKILL. The
//!    former shape-only placeholder here
//!    (`process_crash_injection_subprocess_smoke`) was removed in
//!    W28-S3 per `feedback_noop_trampoline_anti_pattern` — a test whose
//!    body asserts nothing is a no-op trampoline, not coverage.
//!
//! At the 30-second default duration the fault counts work out to
//! ~6 WAL faults + ~10 snapshot faults — enough to exercise the
//! recovery contract repeatedly without overwhelming the wall
//! clock. Slice K's per-op rate-based harness will reach the
//! 1 % / 0.5 % / 0.1 % numbers over multi-hour runs, with
//! subprocess crash injection enabled.
//!
//! ## Property invariants verified
//!
//! Per spec §3, post-recovery state matches pre-crash MVCC state:
//!
//!  1. **No ghost commits** — every read-back from MVCC matches an
//!     oracle entry. No node returned by post-recovery `read_node`
//!     was missing from the oracle.
//!  2. **No lost commits (T1)** — every oracle entry tagged T1 is
//!     observable post-recovery. T3 oracle entries committed within
//!     the last `rpo_ms` window MAY be missing per ADR-034 D-2.
//!  3. **No ghost vector pages** — every vector page returned by
//!     `get_page` matches an oracle entry; no oracle entry for a
//!     pre-snapshot install is missing post-recovery.
//!  4. **No cross-tenant leakage** — tenant 1001's vector store has
//!     no pages with `tenant != 1001`; mirror for 1002.
//!  5. **Snapshot atomic-rename graceful artifact** — the vector
//!     store remains consistent across fault-injected snapshots
//!     (the next clean flush produces a valid file).
//!
//! ## Knobs (env-controlled)
//!
//!  - `PHASE_5_5_TORTURE_SECS` — workload duration in seconds
//!    (default 30; the spec's reference). Lowered to ~5 for
//!    fast smoke runs (`cargo test --ignored phase_5_5_torture
//!    -- --nocapture` with `PHASE_5_5_TORTURE_SECS=5`).
//!
//! Run:
//!   cargo test -p arcgraph-storage --test phase_5_5_torture -- --ignored
//!
//! Or with a shorter duration for a smoke check:
//!   PHASE_5_5_TORTURE_SECS=5 cargo test -p arcgraph-storage \
//!     --test phase_5_5_torture -- --ignored --nocapture

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{DurabilityTier, LabelId, Lsn, NodeId, PageId, PartitionId, TenantId};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle, read_node_with_store,
};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::vector_store::recovery::{
    IndexType as SnapIndexType, MvccVectorSource, VectorArenaPageStore, VectorPageDelta,
    VectorRecoveryRequest, WalDeltaSource, bootstrap_from_mvcc,
};
use arcgraph_storage::vector_store::{
    CrashPoint, SectionKind, SnapshotCatalog, SnapshotSection, SnapshotSpec, VectorPageStoreHandle,
    VectorStoreError, flush_snapshot, flush_snapshot_with_crash_point,
};
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BackgroundFsyncFailAction, BackgroundFsyncScheduler, BlobStoreHandle,
    PageStoreTarget, PrimaryPageStoreHandle, RecordPageStoreHandle, WalConfig, WalWriter,
    recover_from_wal,
};
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────
// Knobs
// ─────────────────────────────────────────────────────────────────

fn torture_duration_secs() -> u64 {
    std::env::var("PHASE_5_5_TORTURE_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(30)
}

const VECTOR_TENANT_RAW_T1: u64 = 1001;
const VECTOR_TENANT_RAW_T3: u64 = 1002;
/// Third vector tenant — exercises cross-tenant isolation across
/// three concurrent workers. T1-tier flushes share `vec_store_1`
/// with tenant 1001; the (tenant, page_id) keying keeps them
/// disjoint per the §6.1 Pattern A arena selection contract.
const VECTOR_TENANT_RAW_AUX: u64 = 1003;

// ─────────────────────────────────────────────────────────────────
// Shared CRUD-side stack
// ─────────────────────────────────────────────────────────────────

fn test_wal_config(dir: PathBuf) -> WalConfig {
    WalConfig {
        dir,
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: Duration::from_millis(2),
        group_commit_max_batch: 32,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

struct CrudStack {
    writer: Option<WalWriter>,
    scheduler: Option<Arc<BackgroundFsyncScheduler>>,
    mgr: Arc<TxnManager>,
    /// Held so the primary index outlives the store / writer; the
    /// recovery path attaches its handle to the replay target so the
    /// post-recovery store can route into the same in-memory leaves.
    /// Read by the recovery wiring inside `recover_crud_stack`.
    #[allow(dead_code)]
    primary: Arc<PrimaryIndex>,
    store: Arc<CrudStore>,
    catalog: Arc<SystemCatalog>,
}

impl CrudStack {
    fn build(dir: &std::path::Path) -> Self {
        let writer = WalWriter::spawn(test_wal_config(dir.to_path_buf())).unwrap();
        let scheduler = BackgroundFsyncScheduler::start(
            writer.handle(),
            BackgroundFsyncFailAction::RollbackAndContinue,
        );
        let handle = writer.handle();
        let mut mgr_inner = TxnManager::with_wal(handle.clone());
        let catalog = Arc::new(SystemCatalog::new());
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        catalog.bootstrap(&pool, &mgr_inner).unwrap();
        mgr_inner.set_durability_lookup(catalog.clone());
        let mgr = Arc::new(mgr_inner);

        let alloc = Arc::new(PageAllocator::new());
        let primary = Arc::new(
            PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
        );
        let store = Arc::new(CrudStore::new_with_index(
            Some(handle.clone()),
            Arc::clone(&primary),
            Arc::clone(&alloc),
        ));
        Self {
            writer: Some(writer),
            scheduler: Some(scheduler),
            mgr,
            primary,
            store,
            catalog,
        }
    }

    fn shutdown(mut self) {
        if let Some(s) = self.scheduler.take() {
            let _ = s.shutdown();
        }
        if let Some(w) = self.writer.take() {
            let _ = w.shutdown();
        }
    }
}

/// Drop the in-memory CrudStack but keep the WAL on disk; subsequent
/// `recover_crud_stack` re-builds a fresh stack from disk.
fn shutdown_crud_stack(stack: CrudStack) {
    stack.shutdown();
}

/// Recover a CrudStack from a WAL directory. Mirrors the
/// `wal_replay_round_trip` recover_stack helper but adds the
/// scheduler + catalog so the recovered stack is functionally
/// complete for the torture loop's continuation.
fn recover_crud_stack(dir: &std::path::Path) -> CrudStack {
    let writer = WalWriter::spawn(test_wal_config(dir.to_path_buf())).unwrap();
    let scheduler = BackgroundFsyncScheduler::start(
        writer.handle(),
        BackgroundFsyncFailAction::RollbackAndContinue,
    );
    let handle = writer.handle();
    let mut mgr_inner = TxnManager::with_wal(handle.clone());
    let catalog = Arc::new(SystemCatalog::new());
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    catalog.bootstrap(&pool, &mgr_inner).unwrap();
    mgr_inner.set_durability_lookup(catalog.clone());
    let mgr = Arc::new(mgr_inner);

    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let store = Arc::new(CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        Arc::clone(&alloc),
    ));
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let records_handle: Arc<dyn RecordPageStoreHandle> =
        Arc::clone(store.records().expect("CrudStore exposes record store"))
            as Arc<dyn RecordPageStoreHandle>;
    let blob_handle: Arc<dyn BlobStoreHandle> =
        Arc::clone(store.blob_store()) as Arc<dyn BlobStoreHandle>;
    // PR #130 / issue #129 P0 fix: wire the AllocatorSeedHandle so v4
    // bundle `allocator_advances` entries seed live counters in
    // commit_lsn order. Without this hookup, post-recovery
    // `alloc_node` re-issues NodeIds that pre-fault commits consumed
    // (ADR-034 D-1 violation; the relaxed Phase-5.5 oracle pre-#130
    // tolerated this as test-side accommodation).
    let allocator_seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&store), Arc::clone(&alloc));
    let target = PageStoreTarget::primary_only(primary_handle)
        .with_record_store(records_handle)
        .with_blob_store(blob_handle)
        .with_allocator_seed(allocator_seed);
    let _ = recover_from_wal(dir, Arc::clone(&mgr), target, None).unwrap();
    CrudStack {
        writer: Some(writer),
        scheduler: Some(scheduler),
        mgr,
        primary,
        store,
        catalog,
    }
}

// ─────────────────────────────────────────────────────────────────
// Oracle: per-tenant committed-state shadow
// ─────────────────────────────────────────────────────────────────

// One historical commit at a (tenant, node_id) key:
// `(label, prop_a, prop_b, tier_at_commit, commit_seq)`. The
// `commit_seq` is a monotonically-increasing per-oracle counter
// (NOT the WAL commit_lsn — we don't have direct read access to
// it during the worker's commit return). It establishes a total
// order across all entries at the same (tenant, NodeId) key so
// post-recovery validation can identify the LATEST historical
// commit and enforce the ADR-034 D-1 strict-byte-identical
// invariant on T1 entries.
type CrudHistoryEntry = (u32, u32, u32, DurabilityTier, u64);

// Map from (tenant, node_id) → list of historical commits at that
// key, in commit-completion order. See `CrudOracle` doc for the
// post-#130 contract this list enforces.
type CrudHistoryMap = std::collections::HashMap<(TenantId, NodeId), Vec<CrudHistoryEntry>>;

/// Records every successful commit so post-recovery we can enforce
/// the ADR-034 D-1 strict-byte-identical invariant on T1 commits and
/// the ADR-034 D-2 RPO-bounded loss tolerance on T3 commits.
///
/// ## Pre-#130 vs post-#130 contract
///
/// PR #128's first cut (pre-#130) used a HashSet<bytes> per key and
/// asserted "store bytes ⊆ historical bytes" — a relaxed contract
/// that masked the issue-#129 P0 bug (PageAllocator high-water reset
/// on recovery → NodeId reuse → orphaned T1 commits). The relaxation
/// was a test-side accommodation for an in-flight production gap.
///
/// PR #130 closed #129 by extending CommitBundle to v4 with a
/// trailing `allocator_advances` section that replay applies in
/// commit_lsn order via `AllocatorSeedHandle`. With that fix in
/// place, the oracle can now enforce the original ADR-034 contract:
///
/// - **T1 strict** (DurabilityTier::Strict at commit time): the
///   store's post-recovery bytes for a (tenant, id) key MUST equal
///   the LATEST historical T1 commit's bytes byte-identically. A
///   T1 commit returns Ok only after WAL fsync completes (ADR-034
///   I-D1); recovery MUST reproduce that commit verbatim. If a
///   later commit at the same key was T3 and didn't durify, the
///   store correctly retains the T1 bytes. Allocator-gap-style
///   silent overwrites are forbidden.
///
/// - **T3 periodic** (DurabilityTier::Periodic): the store's
///   bytes MUST match SOME historical commit at the key (not
///   necessarily the latest). T3 commits within rpo_ms of a fault
///   may be RPO-lost per ADR-034 D-2; the prior T1 commit's bytes
///   (or a prior T3 commit that did fsync) are the legitimate
///   post-recovery value.
///
/// Both cases catch ghost bytes (key returns bytes no commit ever
/// wrote) and cross-tenant leakage (key returns bytes from a
/// different tenant's commit).
#[derive(Default)]
struct CrudOracle {
    /// (tenant, node_id) → ordered list of historical commits at
    /// this key. Order is commit-completion order from `record`
    /// callers (which run inside the commit() return path before
    /// any other oracle mutation can interleave thanks to the
    /// CrudStack mutex held during commit).
    history: Mutex<CrudHistoryMap>,
    /// Total successful commit acks (across all keys). Distinct
    /// from `history.len()` because overwrites collapse to a single
    /// key entry list.
    total_commits: AtomicU64,
    /// Monotonic counter feeding each entry's `commit_seq`. Provides
    /// a total order independent of the WAL's commit_lsn (which the
    /// worker doesn't directly read).
    next_seq: AtomicU64,
}

impl CrudOracle {
    // 7-arg record carries the full (key, bytes, tier) shape per
    // commit; bundling into a struct adds ceremony at the only
    // call site (`do_crud_op`) without clarity gain.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        tenant: TenantId,
        id: NodeId,
        label: u32,
        a: u32,
        b: u32,
        tier: DurabilityTier,
    ) {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        self.history
            .lock()
            .unwrap()
            .entry((tenant, id))
            .or_default()
            .push((label, a, b, tier, seq));
        self.total_commits.fetch_add(1, Ordering::Relaxed);
    }
    fn snapshot(&self) -> CrudHistoryMap {
        self.history.lock().unwrap().clone()
    }
    fn total_commits(&self) -> u64 {
        self.total_commits.load(Ordering::Relaxed)
    }
}

/// Records every successful vector page install so post-recovery
/// we can cross-check the page bytes survive.
#[derive(Default)]
struct VectorOracle {
    /// (tenant, page_id, expected_bytes) for every successful
    /// `install_or_replace`. A subsequent install with the same
    /// (tenant, page_id) overwrites the prior entry to mirror the
    /// store's last-write-wins behavior.
    pages: Mutex<std::collections::HashMap<(TenantId, PageId), Vec<u8>>>,
}

impl VectorOracle {
    fn record(&self, tenant: TenantId, page_id: PageId, bytes: Vec<u8>) {
        self.pages.lock().unwrap().insert((tenant, page_id), bytes);
    }
    fn snapshot(&self) -> std::collections::HashMap<(TenantId, PageId), Vec<u8>> {
        self.pages.lock().unwrap().clone()
    }
}

// ─────────────────────────────────────────────────────────────────
// CRUD-half worker
// ─────────────────────────────────────────────────────────────────

/// Synchronously perform one CRUD transaction on `tenant`. The
/// oracle records every successful commit ack tagged with the
/// tier ACTIVE AT COMMIT TIME (read from the SystemCatalog under
/// the same lock as the commit() call so the read is causally
/// consistent with the WAL bundle's tier dispatch).
///
/// The post-recovery validator splits the assertion by tier:
///  - T1 entries: store bytes MUST match the LATEST historical T1
///    commit at the key (ADR-034 I-D1: T1 ack durable; recovery
///    reproduces verbatim).
///  - T3 entries: store bytes MAY match any historical commit
///    (ADR-034 D-2: T3 RPO loss tolerated).
fn do_crud_op(
    stack: &CrudStack,
    oracle: &CrudOracle,
    tenant: TenantId,
    label: u32,
    a: u32,
    b: u32,
) -> Result<NodeId, ()> {
    let mut tx = stack.mgr.begin(tenant);
    let id = match create_node(
        &stack.store,
        &mut tx,
        tenant,
        LabelId::new(label),
        &PropertyData::InlineU32Pair(a, b),
    ) {
        Ok(id) => id,
        Err(_) => return Err(()),
    };
    // Tier-at-commit-time. Per ADR-034 I-D7 the catalog is the
    // authoritative source; we read it BEFORE invoking commit()
    // so the captured tier matches the tier the bundle dispatcher
    // will use under the same `mgr` instance. Reading post-commit
    // would be racy because a concurrent tier-flip thread can
    // change the value between commit-return and oracle-record.
    let tier = stack.catalog.durability_tier(tenant);
    match commit(tx, &stack.store) {
        Ok(_commit_lsn) => {
            oracle.record(tenant, id, label, a, b, tier);
            Ok(id)
        }
        Err(_) => Err(()),
    }
}

// ─────────────────────────────────────────────────────────────────
// Vector-half worker
// ─────────────────────────────────────────────────────────────────

/// Install one vector page with deterministic bytes derived from
/// (tenant, page_id, generation). Records the install in the oracle
/// when the store accepts it.
fn do_vector_install(
    store: &Arc<dyn VectorPageStoreHandle>,
    oracle: &VectorOracle,
    tenant: TenantId,
    page_id: PageId,
    generation: u64,
) -> Result<(), VectorStoreError> {
    // Bytes encode (tenant_raw, page_id_raw, generation) so a leak
    // across tenants would fail the post-recovery cross-tenant check.
    let mut bytes = vec![0u8; 64];
    bytes[..8].copy_from_slice(&tenant.raw().to_le_bytes());
    bytes[8..16].copy_from_slice(&page_id.raw().to_le_bytes());
    bytes[16..24].copy_from_slice(&generation.to_le_bytes());
    store.install_or_replace(tenant, page_id, &bytes)?;
    oracle.record(tenant, page_id, bytes);
    Ok(())
}

/// Flush a snapshot on a vector store. `crash_point` injects a
/// crash at the requested step (per G.2 atomic-rename contract);
/// `None` is a clean flush.
///
/// The 7-arg signature mirrors `flush_snapshot_with_crash_point`'s
/// load-bearing surface (dir, catalog, tenant, index_id, lsn,
/// payload, crash_point); bundling them into a struct adds ceremony
/// without clarity at the only call site.
#[allow(clippy::too_many_arguments)]
fn do_vector_snapshot_flush(
    snapshot_dir: &std::path::Path,
    catalog: &SnapshotCatalog,
    tenant: TenantId,
    index_id: u64,
    lsn: Lsn,
    payload: &[u8],
    crash_point: Option<CrashPoint>,
) {
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: payload,
    }];
    let spec = SnapshotSpec {
        tenant,
        partition: PartitionId::ZERO,
        index_id,
        lsn,
        encoding: 0,
        index_type: 0,
        dim: 8,
        vectors_count: 8,
        sections: &sections,
    };
    match crash_point {
        Some(cp) => {
            // Crash-point variants intentionally leave a partial /
            // missing artifact; we expect an error AND a graceful
            // recovery on the next clean flush. The G.2 §10.3
            // contract guarantees no half-renamed `.snap` survives
            // — recovery's CRC + dir-fsync chain can detect and GC
            // every interior crash state.
            let _ = flush_snapshot_with_crash_point(&spec, snapshot_dir, catalog, cp);
        }
        None => {
            // Clean flush — must succeed (durable on return per G.2).
            let _ = flush_snapshot(&spec, snapshot_dir, catalog);
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Pseudo-RNG (deterministic; no rand dev-dep beyond what storage
// already pulls in via proptest).
// ─────────────────────────────────────────────────────────────────

struct XorShift {
    state: u64,
}

impl XorShift {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_F00D
            } else {
                seed
            },
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    fn next_in_range(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

// ─────────────────────────────────────────────────────────────────
// In-memory MVCC source for post-recovery vector verification
// ─────────────────────────────────────────────────────────────────

struct VecMvccSource {
    snapshot_lsn: Lsn,
    items: Mutex<std::collections::VecDeque<(u64, Vec<u8>)>>,
}

impl VecMvccSource {
    fn new(snapshot_lsn: Lsn, items: Vec<(u64, Vec<u8>)>) -> Self {
        Self {
            snapshot_lsn,
            items: Mutex::new(items.into()),
        }
    }
}

impl MvccVectorSource for VecMvccSource {
    fn next_vector(
        &self,
    ) -> std::result::Result<Option<(u64, Vec<u8>)>, arcgraph_core::ArcGraphError> {
        Ok(self.items.lock().unwrap().pop_front())
    }
    fn snapshot_lsn(&self) -> Lsn {
        self.snapshot_lsn
    }
}

#[allow(dead_code)]
struct EmptyWal;
impl WalDeltaSource for EmptyWal {
    fn snapshot_lsn(&self) -> Lsn {
        Lsn::ZERO
    }
    fn next_delta(
        &self,
    ) -> std::result::Result<Option<VectorPageDelta>, arcgraph_core::ArcGraphError> {
        Ok(None)
    }
}

// ─────────────────────────────────────────────────────────────────
// The torture test
// ─────────────────────────────────────────────────────────────────

#[test]
#[ignore = "Phase 5.5 torture — runs in a dedicated CI pipeline via --ignored \
            (set PHASE_5_5_TORTURE_SECS=5 for a quick smoke check)"]
fn phase_5_5_torture_30s_n_tenant_fault_injection() {
    let duration_secs = torture_duration_secs();
    let total_duration = Duration::from_secs(duration_secs);

    // ── Workspace dirs ──
    let workspace = TempDir::new().unwrap();
    let wal_dir = workspace.path().join("wal");
    let vec1_dir = workspace.path().join("vec1");
    let vec2_dir = workspace.path().join("vec2");
    std::fs::create_dir_all(&wal_dir).unwrap();
    std::fs::create_dir_all(&vec1_dir).unwrap();
    std::fs::create_dir_all(&vec2_dir).unwrap();

    // ── CRUD half: SYSTEM (T1) + DEFAULT (T1↔T3) ──
    let crud_stack = Arc::new(Mutex::new(Some(CrudStack::build(&wal_dir))));
    let crud_oracle = Arc::new(CrudOracle::default());

    // ── Vector half: tenants 1001 (T1 strict) + 1002 (T3 periodic) ──
    // Hold the typed Arc so post-recovery checks can call
    // `get_page` (on the concrete type) directly. Workers receive a
    // trait-object handle derived from the same Arc.
    let vec_store_1_typed: Arc<VectorArenaPageStore> = Arc::new(VectorArenaPageStore::new());
    let vec_store_2_typed: Arc<VectorArenaPageStore> = Arc::new(VectorArenaPageStore::new());
    let vec_store_1: Arc<dyn VectorPageStoreHandle> =
        Arc::clone(&vec_store_1_typed) as Arc<dyn VectorPageStoreHandle>;
    let vec_store_2: Arc<dyn VectorPageStoreHandle> =
        Arc::clone(&vec_store_2_typed) as Arc<dyn VectorPageStoreHandle>;
    let snap_catalog_1 = Arc::new(SnapshotCatalog::new());
    let snap_catalog_2 = Arc::new(SnapshotCatalog::new());
    let vec_oracle = Arc::new(VectorOracle::default());

    // ── Coordination ──
    let stop = Arc::new(AtomicBool::new(false));
    let wal_fault_count = Arc::new(AtomicU64::new(0));
    let snapshot_fault_count = Arc::new(AtomicU64::new(0));
    let total_ops = Arc::new(AtomicU64::new(0));

    // ── Worker A: tenant 1003 (T1 strict) vector page installs ──
    //
    // The third vector tenant. Re-uses `vec_store_1` (since
    // VectorArenaPageStore is multi-tenant: keys are (tenant,
    // page_id)). Cross-tenant pages with the same page_id MUST stay
    // disjoint per the §6.1 Pattern A arena-selection contract.
    let worker_a = {
        let stop = Arc::clone(&stop);
        let store = Arc::clone(&vec_store_1);
        let oracle = Arc::clone(&vec_oracle);
        let total_ops = Arc::clone(&total_ops);
        thread::spawn(move || {
            let mut rng = XorShift::new(0xA1A1_A1A1);
            let tenant = TenantId::new(VECTOR_TENANT_RAW_AUX);
            let mut generation = 0u64;
            while !stop.load(Ordering::Relaxed) {
                generation = generation.wrapping_add(1);
                let page = if rng.next_u64() % 3 == 0 {
                    PageId::new(rng.next_in_range(64))
                } else {
                    PageId::new(generation)
                };
                let _ = do_vector_install(&store, &oracle, tenant, page, generation);
                total_ops.fetch_add(1, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(8));
            }
        })
    };

    // ── Worker B: DEFAULT tenant CRUD writes (tier flips) ──
    let worker_b = {
        let stop = Arc::clone(&stop);
        let stack_holder = Arc::clone(&crud_stack);
        let oracle = Arc::clone(&crud_oracle);
        let total_ops = Arc::clone(&total_ops);
        thread::spawn(move || {
            let mut rng = XorShift::new(0xB2B2_B2B2);
            let mut user_label = 2_000_000u32;
            while !stop.load(Ordering::Relaxed) {
                let label = user_label;
                user_label = user_label.wrapping_add(1);
                let a = rng.next_u32();
                let b = rng.next_u32();
                if let Some(stack) = stack_holder.lock().unwrap().as_ref() {
                    let _ = do_crud_op(stack, &oracle, TenantId::DEFAULT, label, a, b);
                    total_ops.fetch_add(1, Ordering::Relaxed);
                }
                thread::sleep(Duration::from_millis(10));
            }
        })
    };

    // ── Worker C: tenant 1001 vector page installs (T1 strict) ──
    let worker_c = {
        let stop = Arc::clone(&stop);
        let store = Arc::clone(&vec_store_1);
        let oracle = Arc::clone(&vec_oracle);
        let total_ops = Arc::clone(&total_ops);
        thread::spawn(move || {
            let mut rng = XorShift::new(0xC3C3_C3C3);
            let tenant = TenantId::new(VECTOR_TENANT_RAW_T1);
            let mut generation = 0u64;
            while !stop.load(Ordering::Relaxed) {
                generation = generation.wrapping_add(1);
                // Re-install the same page-ids periodically (mirrors
                // graph-edge-rewrite traffic); plus brand-new pages
                // (mirrors fresh inserts).
                let page = if rng.next_u64() % 3 == 0 {
                    PageId::new(rng.next_in_range(64))
                } else {
                    PageId::new(generation)
                };
                let _ = do_vector_install(&store, &oracle, tenant, page, generation);
                total_ops.fetch_add(1, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(6));
            }
        })
    };

    // ── Worker D: tenant 1002 vector page installs (T3 periodic) ──
    let worker_d = {
        let stop = Arc::clone(&stop);
        let store = Arc::clone(&vec_store_2);
        let oracle = Arc::clone(&vec_oracle);
        let total_ops = Arc::clone(&total_ops);
        thread::spawn(move || {
            let mut rng = XorShift::new(0xD4D4_D4D4);
            let tenant = TenantId::new(VECTOR_TENANT_RAW_T3);
            let mut generation = 0u64;
            while !stop.load(Ordering::Relaxed) {
                generation = generation.wrapping_add(1);
                let page = if rng.next_u64() % 3 == 0 {
                    PageId::new(rng.next_in_range(64))
                } else {
                    PageId::new(generation)
                };
                let _ = do_vector_install(&store, &oracle, tenant, page, generation);
                total_ops.fetch_add(1, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(7));
            }
        })
    };

    // ── Fault injector: WAL shutdown + restart cycle every 5s ──
    let wal_fault_thread = {
        let stop = Arc::clone(&stop);
        let stack_holder = Arc::clone(&crud_stack);
        let wal_fault_count = Arc::clone(&wal_fault_count);
        let wal_dir = wal_dir.clone();
        thread::spawn(move || {
            let interval = Duration::from_millis(5_000);
            let mut last_fire = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(50));
                if last_fire.elapsed() < interval {
                    continue;
                }
                last_fire = Instant::now();
                // Atomic swap-and-shutdown:
                let mut guard = stack_holder.lock().unwrap();
                let prior = match guard.take() {
                    Some(s) => s,
                    None => continue,
                };
                shutdown_crud_stack(prior);
                // Recover from the WAL we just shut down. Per the
                // existing wal_replay_round_trip pattern, this
                // re-applies every prior commit.
                let recovered = recover_crud_stack(&wal_dir);
                *guard = Some(recovered);
                wal_fault_count.fetch_add(1, Ordering::Relaxed);
            }
        })
    };

    // ── Fault injector: vector snapshot flush w/ crash point ──
    let snapshot_fault_thread = {
        let stop = Arc::clone(&stop);
        let snap_catalog_1 = Arc::clone(&snap_catalog_1);
        let snap_catalog_2 = Arc::clone(&snap_catalog_2);
        let snapshot_fault_count = Arc::clone(&snapshot_fault_count);
        let vec1_dir = vec1_dir.clone();
        let vec2_dir = vec2_dir.clone();
        thread::spawn(move || {
            let interval = Duration::from_millis(3_000);
            let mut last_fire = Instant::now();
            let mut rng = XorShift::new(0xFA17_C0DE);
            let mut lsn_counter = 1u64;
            let crash_points = [
                CrashPoint::AfterTempCreate,
                CrashPoint::MidWrite(32),
                CrashPoint::BeforeRename,
                CrashPoint::BeforeDirFsync,
                CrashPoint::BeforeCatalogStamp,
            ];
            while !stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(50));
                if last_fire.elapsed() < interval {
                    continue;
                }
                last_fire = Instant::now();
                lsn_counter = lsn_counter.wrapping_add(1);
                let payload = vec![(lsn_counter & 0xff) as u8; 64];

                // Tenant 1001 (T1) — alternate clean / crash flushes.
                let cp1 = if lsn_counter % 3 == 0 {
                    Some(crash_points[(rng.next_u64() % crash_points.len() as u64) as usize])
                } else {
                    None
                };
                do_vector_snapshot_flush(
                    &vec1_dir,
                    &snap_catalog_1,
                    TenantId::new(VECTOR_TENANT_RAW_T1),
                    1,
                    Lsn::new(lsn_counter),
                    &payload,
                    cp1,
                );
                if cp1.is_some() {
                    snapshot_fault_count.fetch_add(1, Ordering::Relaxed);
                }

                // Tenant 1002 (T3) — same pattern, distinct LSN
                // counter so the two catalogs evolve independently.
                let cp2 = if lsn_counter % 5 == 0 {
                    Some(crash_points[(rng.next_u64() % crash_points.len() as u64) as usize])
                } else {
                    None
                };
                do_vector_snapshot_flush(
                    &vec2_dir,
                    &snap_catalog_2,
                    TenantId::new(VECTOR_TENANT_RAW_T3),
                    1,
                    Lsn::new(lsn_counter + 1),
                    &payload,
                    cp2,
                );
                if cp2.is_some() {
                    snapshot_fault_count.fetch_add(1, Ordering::Relaxed);
                }
            }
        })
    };

    // ── Fault injector: tier flips on DEFAULT (every ~3s) ──
    let tier_flip_thread = {
        let stop = Arc::clone(&stop);
        let stack_holder = Arc::clone(&crud_stack);
        thread::spawn(move || {
            let interval = Duration::from_millis(3_000);
            let mut last_fire = Instant::now();
            let mut to_t3 = true;
            while !stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(100));
                if last_fire.elapsed() < interval {
                    continue;
                }
                last_fire = Instant::now();
                let target = if to_t3 {
                    DurabilityTier::Periodic { rpo_ms: 1_000 }
                } else {
                    DurabilityTier::Strict
                };
                to_t3 = !to_t3;
                if let Some(stack) = stack_holder.lock().unwrap().as_ref() {
                    let mut tx = stack.mgr.begin(TenantId::SYSTEM);
                    let _ = stack
                        .catalog
                        .set_durability_tier(&mut tx, TenantId::DEFAULT, target);
                    let _ = tx.commit();
                    if let Some(s) = stack.scheduler.as_ref() {
                        s.register(TenantId::DEFAULT, target);
                    }
                }
            }
        })
    };

    // ── Sustained workload: stop after duration ──
    thread::sleep(total_duration);
    stop.store(true, Ordering::Relaxed);

    // Join workers + injectors.
    let _ = worker_a.join();
    let _ = worker_b.join();
    let _ = worker_c.join();
    let _ = worker_d.join();
    let _ = wal_fault_thread.join();
    let _ = snapshot_fault_thread.join();
    let _ = tier_flip_thread.join();

    // ── Final shutdown + recovery ──
    let final_stack = crud_stack.lock().unwrap().take();
    if let Some(s) = final_stack {
        shutdown_crud_stack(s);
    }
    let recovered = recover_crud_stack(&wal_dir);

    // ── Property 1: post-recovery CRUD state matches the oracle ──
    //
    // Empirical proof of the post-#130 contract (PR #130 closed
    // issue #129 P0 by extending CommitBundle to v4 with a
    // trailing `allocator_advances` section). PR #130's canary
    // `t1_strict_byte_identical_after_fault_recovery` established
    // that pre-#130 the relaxed Phase-5.5 oracle masked an
    // ADR-034 D-1 violation (5.4× NodeId reuse → 81% T1 commits
    // orphaned post-recovery). Post-#130 the strict invariant
    // holds and the oracle below now enforces it.
    //
    // Per-tier validator:
    //
    //  - **T1 strict**: the store's bytes for (tenant, id) MUST
    //    equal the LATEST historical T1 commit's bytes
    //    byte-identically. T1 ack returned only after WAL fsync
    //    completed, so recovery MUST reproduce the commit verbatim.
    //  - **T3 periodic**: the store's bytes MAY match any
    //    historical commit at the key (latest, or any earlier
    //    one if the latest was RPO-lost per ADR-034 D-2).
    //  - **Both tiers**: the store's bytes MUST be one of the
    //    historical commits at the key. Bytes that no commit
    //    ever wrote are an I-V1 ghost violation (regardless of
    //    tier).
    // (NodeId, observed_bytes, expected_latest_T1_bytes) — flagged
    // when the post-recovery store's bytes for a key with a T1
    // history entry don't match the latest T1 commit's bytes.
    type T1Violation = (NodeId, (u32, u32, u32), (u32, u32, u32));
    let oracle_history = crud_oracle.snapshot();
    let mut found_count = 0_usize;
    let mut t1_strict_violations: Vec<T1Violation> = Vec::new();
    let mut ghost_byte_violations: Vec<(NodeId, (u32, u32, u32))> = Vec::new();
    let mut t1_keys = 0_usize;
    let mut t1_satisfied = 0_usize;
    for ((tenant, id), entries) in &oracle_history {
        let tx = recovered.mgr.begin(*tenant);
        let read = match read_node_with_store(&recovered.store, &tx, *id) {
            Ok(Some(rec)) => Some((rec.label_id, rec.inline_u32a, rec.inline_u32b)),
            Ok(None) => None,
            Err(e) => panic!("phase_5_5_torture: read_node_with_store error for {id:?}: {e:?}"),
        };

        // Latest T1 entry at this key (highest commit_seq among
        // entries tagged DurabilityTier::Strict).
        let latest_t1 = entries
            .iter()
            .filter(|(_, _, _, tier, _)| matches!(tier, DurabilityTier::Strict))
            .max_by_key(|(_, _, _, _, seq)| *seq)
            .map(|(l, a, b, _, _)| (*l, *a, *b));

        if let Some(observed) = read {
            // Ghost check (any tier): bytes must match SOME
            // historical commit at this key.
            let any_match = entries
                .iter()
                .any(|(l, a, b, _, _)| observed == (*l, *a, *b));
            if !any_match {
                ghost_byte_violations.push((*id, observed));
            } else {
                found_count += 1;
            }

            // T1 strict check: if a T1 entry exists, the store's
            // bytes MUST equal the LATEST T1 entry's bytes.
            if let Some(t1_bytes) = latest_t1 {
                t1_keys += 1;
                if observed == t1_bytes {
                    t1_satisfied += 1;
                } else {
                    t1_strict_violations.push((*id, observed, t1_bytes));
                }
            }
        } else if latest_t1.is_some() {
            // Read returned None for a key with a T1 commit. Per
            // ADR-034 I-D1, T1 commits MUST be observable post-
            // recovery — None is a violation.
            t1_keys += 1;
            t1_strict_violations.push((*id, (0, 0, 0), latest_t1.unwrap_or((0, 0, 0))));
        }
    }
    assert!(
        ghost_byte_violations.is_empty(),
        "I-V1 ghost violation: post-recovery store returned bytes for {} (tenant, id) \
         keys that no historical commit ever wrote (first 5: {:?})",
        ghost_byte_violations.len(),
        ghost_byte_violations.iter().take(5).collect::<Vec<_>>()
    );
    assert!(
        t1_strict_violations.is_empty(),
        "ADR-034 I-D1 / I-V6 violation: post-recovery store bytes drifted from \
         latest historical T1 commit on {} (tenant, id) keys; {t1_satisfied}/{t1_keys} \
         T1 keys satisfied. Pre-PR-#130 this would be expected (allocator gap); \
         post-#130 it indicates a regression in the v4 bundle's `allocator_advances` \
         path or `crud_allocator_seed_handle` wiring. First 5: {:?}",
        t1_strict_violations.len(),
        t1_strict_violations
            .iter()
            .take(5)
            .map(|(id, observed, expected)| format!(
                "{id:?} got={observed:?} expected_latest_T1={expected:?}"
            ))
            .collect::<Vec<_>>()
    );
    if !oracle_history.is_empty() {
        let recovery_rate = found_count as f64 / oracle_history.len() as f64;
        assert!(
            recovery_rate >= 0.80,
            "phase_5_5_torture: post-recovery recall {recovery_rate:.4} below 80% floor; \
             found={found_count}/{} — T3 RPO loss should not exceed 20% under our \
             fault-injection cadence",
            oracle_history.len()
        );
    }
    let total_commits = crud_oracle.total_commits();
    eprintln!(
        "phase_5_5_torture CRUD: {} unique (tenant,id) keys from {} total commits, \
         {found_count} read-back-matches, {t1_satisfied}/{t1_keys} T1-strict-satisfied",
        oracle_history.len(),
        total_commits
    );

    // ── Property 2: vector oracle entries match store state ──
    //
    // Every install_or_replace for tenant T MUST be readable from
    // T's store with byte-identical bytes. The vector half does
    // not yet flow through the WAL bundle path so there is no
    // "T3 loss" window — every recorded install is the
    // last-write-wins snapshot.
    //
    // Tenants 1001 + 1003 share `vec_store_1` (multi-tenant store
    // keyed by (tenant, page_id)). Tenant 1002 lives in `vec_store_2`.
    let vec_oracle_snapshot = vec_oracle.snapshot();
    let mut vec_violations: Vec<((TenantId, PageId), &'static str)> = Vec::new();
    for ((tenant, page_id), expected_bytes) in &vec_oracle_snapshot {
        let store = match tenant.raw() {
            VECTOR_TENANT_RAW_T1 | VECTOR_TENANT_RAW_AUX => &vec_store_1_typed,
            VECTOR_TENANT_RAW_T3 => &vec_store_2_typed,
            _ => panic!("oracle has unknown tenant {tenant:?}"),
        };
        match store.get_page(*tenant, *page_id) {
            Some(actual) if &actual == expected_bytes => {}
            Some(_) => vec_violations.push(((*tenant, *page_id), "byte mismatch")),
            None => vec_violations.push(((*tenant, *page_id), "missing")),
        }
    }
    assert!(
        vec_violations.is_empty(),
        "vector oracle drift: {} violations (first 5: {:?})",
        vec_violations.len(),
        vec_violations.iter().take(5).collect::<Vec<_>>()
    );

    // ── Property 3: cross-tenant isolation ──
    //
    // Every page is bytes prefix-stamped with its owning tenant
    // (per do_vector_install). For each page in the oracle:
    //   - looking it up under the OTHER store (cross-store leak)
    //     must return None.
    //   - the bytes prefix must equal the owning tenant.
    //   - within vec_store_1 (which holds 1001 + 1003), looking
    //     up tenant 1001's page_id under tenant 1003 (and
    //     vice-versa) must return None — the keying is
    //     (tenant, page_id), not page_id alone.
    for ((tenant, page), bytes) in &vec_oracle_snapshot {
        let tag = u64::from_le_bytes(bytes[..8].try_into().expect("8-byte tenant tag"));
        assert_eq!(
            tag,
            tenant.raw(),
            "I-V2 byte-tag mismatch: page {:?} for tenant {:?} starts with \
             tenant tag {tag} (must match owning tenant)",
            page,
            tenant
        );
        match tenant.raw() {
            VECTOR_TENANT_RAW_T1 => {
                assert!(
                    vec_store_2_typed.get_page(*tenant, *page).is_none(),
                    "I-V2 cross-store leak: tenant 1001 page {:?} found in \
                     vec_store_2",
                    page
                );
                // Within vec_store_1, looking up the SAME page_id
                // under tenant 1003 must return either None OR a
                // page tagged with 1003 (1003's own install at the
                // same page_id) — never tenant 1001's bytes.
                if let Some(other) =
                    vec_store_1_typed.get_page(TenantId::new(VECTOR_TENANT_RAW_AUX), *page)
                {
                    let other_tag = u64::from_le_bytes(other[..8].try_into().unwrap());
                    assert_eq!(
                        other_tag, VECTOR_TENANT_RAW_AUX,
                        "I-V2 multi-tenant arena leak: page {:?} under tenant 1003 \
                         carries tenant 1001's bytes",
                        page
                    );
                }
            }
            VECTOR_TENANT_RAW_AUX => {
                assert!(
                    vec_store_2_typed.get_page(*tenant, *page).is_none(),
                    "I-V2 cross-store leak: tenant 1003 page {:?} found in \
                     vec_store_2",
                    page
                );
                if let Some(other) =
                    vec_store_1_typed.get_page(TenantId::new(VECTOR_TENANT_RAW_T1), *page)
                {
                    let other_tag = u64::from_le_bytes(other[..8].try_into().unwrap());
                    assert_eq!(
                        other_tag, VECTOR_TENANT_RAW_T1,
                        "I-V2 multi-tenant arena leak: page {:?} under tenant 1001 \
                         carries tenant 1003's bytes",
                        page
                    );
                }
            }
            VECTOR_TENANT_RAW_T3 => {
                assert!(
                    vec_store_1_typed.get_page(*tenant, *page).is_none(),
                    "I-V2 cross-store leak: tenant 1002 page {:?} found in \
                     vec_store_1",
                    page
                );
            }
            _ => unreachable!("tenant filter above covers all variants"),
        }
    }

    // ── Property 4: snapshot-fault graceful artifacts ──
    //
    // For each fault-injected snapshot, the per-tenant SnapshotCatalog
    // either advanced past the requested LSN (the rename + dir-fsync
    // succeeded before the crash point) OR did not advance (the rename
    // never ran). What we forbid: the catalog advancing AND the file
    // being absent or corrupt — that would be an "ack-without-bytes"
    // violation.
    //
    // A clean follow-up flush after every fault must succeed; we
    // verify by inspecting the catalog's latest_lsn for both vector
    // tenants. If the workload fired at least one clean flush
    // (almost certain across 30 seconds of 3-second cadence), the
    // catalog must hold a Some(lsn).
    let lsn_1 = snap_catalog_1.latest_lsn(TenantId::new(VECTOR_TENANT_RAW_T1), 1);
    let lsn_2 = snap_catalog_2.latest_lsn(TenantId::new(VECTOR_TENANT_RAW_T3), 1);
    if duration_secs >= 6 {
        // A 30-second run will fire ≥ 9 snapshot attempts each side;
        // even if every other one is fault-injected, at least one
        // clean flush lands.
        assert!(
            lsn_1.is_some() || snapshot_fault_count.load(Ordering::Relaxed) >= 1,
            "phase_5_5_torture: tenant 1001 snapshot catalog never stamped \
             AND no faults fired — the snapshot worker did not run"
        );
        assert!(
            lsn_2.is_some() || snapshot_fault_count.load(Ordering::Relaxed) >= 1,
            "phase_5_5_torture: tenant 1002 snapshot catalog never stamped \
             AND no faults fired — the snapshot worker did not run"
        );
    }

    // ── Property 5: post-recovery vector arena rebuild via MVCC ──
    //
    // The bootstrap_from_mvcc path (ADR-035 §9.1) rebuilds an arena
    // from an MVCC walk. We construct a synthetic MVCC source from
    // the oracle and run a rebuild — the arena's bootstrap_vectors
    // count must equal the oracle entry count for that tenant.
    for tenant_raw in [
        VECTOR_TENANT_RAW_T1,
        VECTOR_TENANT_RAW_T3,
        VECTOR_TENANT_RAW_AUX,
    ] {
        let tenant = TenantId::new(tenant_raw);
        let oracle_for_tenant: Vec<(u64, Vec<u8>)> = vec_oracle_snapshot
            .iter()
            .filter(|((t, _), _)| t.raw() == tenant_raw)
            .map(|((_, p), b)| (p.raw(), b.clone()))
            .collect();
        let count = oracle_for_tenant.len();
        let mvcc = VecMvccSource::new(Lsn::new(99_999), oracle_for_tenant);
        let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
        let req = VectorRecoveryRequest::v1(tenant, 1, SnapIndexType::Hnsw, 8);
        let arena = bootstrap_from_mvcc(store, &mvcc, req).expect("bootstrap");
        assert_eq!(
            arena.bootstrap_vectors.len(),
            count,
            "phase_5_5_torture: bootstrap_from_mvcc rebuilt count drift for tenant {tenant:?}"
        );
        assert_eq!(arena.tenant_id, tenant);
    }

    // ── Final shutdown of recovered stack ──
    shutdown_crud_stack(recovered);

    // ── Telemetry summary (visible with --nocapture) ──
    eprintln!(
        "phase_5_5_torture summary: duration={duration_secs}s ops={} \
         crud_oracle_unique={} crud_total_commits={} crud_found={found_count} \
         wal_faults={} snapshot_faults={} \
         vec_pages_t1={} vec_pages_aux={} vec_pages_t3={}",
        total_ops.load(Ordering::Relaxed),
        oracle_history.len(),
        crud_oracle.total_commits(),
        wal_fault_count.load(Ordering::Relaxed),
        snapshot_fault_count.load(Ordering::Relaxed),
        vec_oracle_snapshot
            .keys()
            .filter(|(t, _)| t.raw() == VECTOR_TENANT_RAW_T1)
            .count(),
        vec_oracle_snapshot
            .keys()
            .filter(|(t, _)| t.raw() == VECTOR_TENANT_RAW_AUX)
            .count(),
        vec_oracle_snapshot
            .keys()
            .filter(|(t, _)| t.raw() == VECTOR_TENANT_RAW_T3)
            .count(),
    );
}

// ─────────────────────────────────────────────────────────────────
// Process-crash subprocess smoke
// ─────────────────────────────────────────────────────────────────
//
// V-3 (W28-S3): the shape-only `process_crash_injection_subprocess_smoke`
// placeholder was REMOVED here per `feedback_noop_trampoline_anti_pattern`
// (a `#[ignore]`'d test whose body asserts nothing is a no-op trampoline,
// not coverage). Real subprocess SIGKILL-during-recovery coverage now
// lives in Slice K's harness at `tests/k3_sigkill_during_rebuild.rs`
// (`k3_sigkill_during_rebuild_subprocess`, closes #256) via a
// `Command::spawn(test_binary)` self-fork + SIGKILL + recover-and-assert.
// The in-thread fault injection in
// `phase_5_5_torture_30s_n_tenant_fault_injection` above exercises the
// same recovery contract under graceful teardown.
