//! #849 STEP-0 — durable-ingest throughput + WAL-amplification MEASUREMENT
//! harness + B1 root-cause REGRESSION GUARDS.
//!
//! This file carries two kinds of test:
//!
//! 1. **Always-on assertion guards** (run in the default gauntlet, no
//!    env flag):
//!    - [`node_vs_edge_have_no_structural_penalty_849`] — the B1
//!      strong oracle. It locks in the measured finding that the durable
//!      *node* ingest path is NOT structurally heavier than the *edge*
//!      path at the same batch size (the "37×" was a batch-size /
//!      loader artifact, NOT a per-node penalty). RED-on-revert if a
//!      future change adds per-node WAL/fsync/page work that edges skip.
//!    - [`durable_node_ingest_recovers_exact_849`] — the durability
//!      determinism guard: N nodes ingested durably → process-equivalent
//!      restart (`recover_from_wal`) → every node recovers with its
//!      EXACT label + properties (no silent loss, no swap).
//!
//! 2. **The `#[ignore]` measurement sweep**
//!    ([`step0_durable_ingest_batch_sweep_849`]) — prints a structured
//!    throughput / WAL-amplification report across batch sizes. It is
//!    `#[ignore]`'d (heavy: ~30 s) AND panic-by-default env-gated
//!    (W25-MFI-2; `feedback_test_env_gate_panic_by_default`): when
//!    invoked via `--ignored` it PANICS unless `ARC849_REPRO=1` is set
//!    (or `ARCGRAPH_ARC849_REPRO_SKIP_OK=1` opts into a soft-skip).
//!
//! ```text
//! ARC849_REPRO=1 cargo test -p arcgraph-storage \
//!     --test durable_ingest_throughput_849 -- --ignored --nocapture
//! ```
//!
//! # What it measures (the make-or-break STEP-0)
//!
//! It replicates the EXACT per-record sequence the durable `graph.ingest`
//! path runs (`StorageIngestProvider::ingest`,
//! `crates/arcgraph-mcp/src/storage/adapters.rs:931`):
//!
//! - node: `intern_label_logged` (WAL-logs the label iff `was_new`) +
//!   `crud::create_node` (MVCC write + buffered primary-index install)
//! - rel : `intern_type_logged` + `crud::create_rel` (MVCC write +
//!   buffered TEL append + buffered primary-index install)
//! - one `crud::commit` per BATCH (= one `CommitBundle` = one fsync
//!   cohort under `DurabilityTier::Strict`, ADR-031)
//!
//! over a REAL durable stack (`WalWriter` + on-disk WAL + `PosixPageIo`
//! semantics) wired exactly like the CLI `build_durable`
//! (`crates/arcgraph-cli/src/bootstrap.rs:340`): primary index + record
//! store + blob store + intern table, NO secondary index (the durable
//! bootstrap wires none). web-Google–shaped fixture: every node shares
//! ONE label, every edge shares ONE rel-type, NO properties — so the
//! intern WAL-log is paid exactly once per phase (NOT per record), which
//! the report verifies.
//!
//! CDC sink is intentionally NOT wired here: it stages an in-memory event
//! per record SYMMETRICALLY for nodes and rels, so it cannot create a
//! node-vs-edge asymmetry; omitting it isolates the storage primitives.
//!
//! # Metrics emitted per (phase, batch_size)
//!
//! - records/s (the B1 number)
//! - WAL bytes written per record — both apparent (sum of file lengths)
//!   and on-disk blocks (`st_blocks * 512`), to settle B3(a)
//!   real-amplification vs segment-pre-alloc-artifact
//! - fsync count (one `observe_wal_fsync_ms` per `fire()`)
//! - coarse per-stage wall-clock split (intern / create / commit)

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use arcgraph_core::{LabelId, NodeId, TenantId, TypeId};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, create_rel, crud_allocator_seed_handle,
    read_node_with_store,
};
use arcgraph_storage::intern::{InternTable, intern_label_logged, intern_type_logged};
use arcgraph_storage::metrics::{MetricsSink, QueryPlanType, StoragePageKind, WalWriteOutcome};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
    RecordPageStoreHandle, WalConfig, WalWriter, recover_from_wal,
};
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────────
// Fsync-counting metrics sink
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct FsyncCounter {
    fsyncs: AtomicU64,
    wal_writes: AtomicU64,
}

impl MetricsSink for FsyncCounter {
    fn record_wal_write(&self, _outcome: WalWriteOutcome) {
        self.wal_writes.fetch_add(1, Ordering::Relaxed);
    }
    fn observe_wal_fsync_ms(&self, _duration_ms: f64) {
        self.fsyncs.fetch_add(1, Ordering::Relaxed);
    }
    fn record_storage_page(&self, _kind: StoragePageKind) {}
    fn record_hot_vertex_warning(&self, _tenant: TenantId) {}
    fn record_query_plan_choice(&self, _plan_type: QueryPlanType) {}
}

// ─────────────────────────────────────────────────────────────────────
// Durable stack — mirrors CLI build_durable wiring (no secondary index)
// ─────────────────────────────────────────────────────────────────────

struct DurableStack {
    writer: Option<WalWriter>,
    crud: Arc<CrudStore>,
    txn_manager: Arc<TxnManager>,
    intern: Arc<InternTable>,
    sink: Arc<FsyncCounter>,
    wal_dir: std::path::PathBuf,
}

impl DurableStack {
    fn build(data_dir: &Path) -> Self {
        let wal_dir = data_dir.join("wal");
        std::fs::create_dir_all(&wal_dir).unwrap();
        let sink = Arc::new(FsyncCounter::default());

        // Default 64 MiB segments (WalConfig::new), group-commit window
        // 2 ms / batch 4 — identical to the CLI durable default
        // (writer.rs:105). A single-threaded loader never fills a
        // group-commit batch, so every commit is its own fsync cohort.
        let config =
            WalConfig::new(&wal_dir).with_metrics_sink(Arc::clone(&sink) as Arc<dyn MetricsSink>);
        let writer = WalWriter::spawn(config).unwrap();
        let handle = writer.handle();

        // No durability lookup → DurabilityTier::Strict (fsync-before-ack).
        let txn_manager = Arc::new(TxnManager::with_wal(handle.clone()));
        let allocator = Arc::new(PageAllocator::new());
        let primary = Arc::new(
            PrimaryIndex::new(
                Arc::clone(&txn_manager),
                Arc::clone(&allocator),
                Some(handle.clone()),
            )
            .unwrap(),
        );
        let crud = Arc::new(CrudStore::new_with_index(
            Some(handle.clone()),
            Arc::clone(&primary),
            Arc::clone(&allocator),
        ));
        let intern = Arc::new(InternTable::new());

        Self {
            writer: Some(writer),
            crud,
            txn_manager,
            intern,
            sink,
            wal_dir,
        }
    }

    fn wal_bytes_apparent(&self) -> u64 {
        wal_dir_apparent(&self.wal_dir)
    }
    fn wal_bytes_on_disk(&self) -> u64 {
        wal_dir_on_disk_blocks(&self.wal_dir)
    }
    fn fsyncs(&self) -> u64 {
        self.sink.fsyncs.load(Ordering::Relaxed)
    }
}

impl Drop for DurableStack {
    fn drop(&mut self) {
        if let Some(w) = self.writer.take() {
            let _ = w.shutdown();
        }
    }
}

/// Apparent WAL size = sum of file lengths (the "logical" byte count;
/// equals written bytes because segments are append-written with NO
/// pre-allocation / `set_len` — see `wal/segment.rs::SegmentWriter`).
fn wal_dir_apparent(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(read) = std::fs::read_dir(dir) {
        for entry in read.flatten() {
            if let Ok(md) = entry.metadata() {
                if md.is_file() {
                    total += md.len();
                }
            }
        }
    }
    total
}

/// On-disk WAL size = sum of allocated blocks (`st_blocks * 512`). If
/// this is ~equal to the apparent size, the WAL is NOT sparse / NOT
/// pre-allocated and the byte count is REAL written data. If it greatly
/// exceeds apparent, segments are pre-sized (the artifact hypothesis).
fn wal_dir_on_disk_blocks(dir: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    let mut total = 0u64;
    if let Ok(read) = std::fs::read_dir(dir) {
        for entry in read.flatten() {
            if let Ok(md) = entry.metadata() {
                if md.is_file() {
                    total += md.blocks() * 512;
                }
            }
        }
    }
    total
}

// ─────────────────────────────────────────────────────────────────────
// Phase measurement
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct PhaseReport {
    records: u64,
    batch: usize,
    wall_s: f64,
    per_s: f64,
    wal_bytes_apparent: u64,
    wal_bytes_on_disk: u64,
    bytes_per_rec_apparent: f64,
    bytes_per_rec_on_disk: f64,
    fsyncs: u64,
    fsync_per_rec: f64,
    intern_s: f64,
    create_s: f64,
    commit_s: f64,
}

impl PhaseReport {
    fn print(&self, label: &str) {
        println!(
            "  {label:<6} batch={batch:<5} {records:>8} recs  {per_s:>11.1} rec/s  \
             wall={wall_s:>7.3}s | WAL/rec: apparent={bpa:>8.1}B  on_disk={bpd:>8.1}B  \
             | fsyncs={fsyncs:>7} ({fpr:.3}/rec) | WALtot app={wta}/disk={wtd} \
             | stage% intern={ip:>4.1} create={cp:>4.1} commit={cm:>4.1}",
            label = label,
            batch = self.batch,
            records = self.records,
            per_s = self.per_s,
            wall_s = self.wall_s,
            bpa = self.bytes_per_rec_apparent,
            bpd = self.bytes_per_rec_on_disk,
            wta = self.wal_bytes_apparent,
            wtd = self.wal_bytes_on_disk,
            fsyncs = self.fsyncs,
            fpr = self.fsync_per_rec,
            ip = 100.0 * self.intern_s / self.wall_s.max(1e-9),
            cp = 100.0 * self.create_s / self.wall_s.max(1e-9),
            cm = 100.0 * self.commit_s / self.wall_s.max(1e-9),
        );
    }
}

const TENANT: TenantId = TenantId::DEFAULT;
const NODE_LABEL: &str = "Page";
const REL_TYPE: &str = "LINKS";

/// Load `n` nodes in batches of `batch`. Returns the per-phase report and
/// the allocated node ids (used as edge endpoints in the rel phase).
fn run_node_phase(
    stack: &DurableStack,
    n: u64,
    batch: usize,
) -> (PhaseReport, Vec<arcgraph_core::NodeId>) {
    let wal_before_app = stack.wal_bytes_apparent();
    let wal_before_disk = stack.wal_bytes_on_disk();
    let fsync_before = stack.fsyncs();

    let mut node_ids = Vec::with_capacity(n as usize);
    let (mut intern_s, mut create_s, mut commit_s) = (0.0f64, 0.0f64, 0.0f64);

    let t0 = Instant::now();
    let mut done = 0u64;
    while done < n {
        let this = std::cmp::min(batch as u64, n - done);
        let mut tx = stack.txn_manager.begin(TENANT);
        for _ in 0..this {
            let ti = Instant::now();
            let label: LabelId =
                intern_label_logged(&stack.intern, stack.crud.wal(), TENANT, NODE_LABEL).unwrap();
            intern_s += ti.elapsed().as_secs_f64();

            let tc = Instant::now();
            let nid =
                create_node(&stack.crud, &mut tx, TENANT, label, &PropertyData::Empty).unwrap();
            create_s += tc.elapsed().as_secs_f64();
            node_ids.push(nid);
        }
        let tk = Instant::now();
        commit(tx, &stack.crud).unwrap();
        commit_s += tk.elapsed().as_secs_f64();
        done += this;
    }
    let wall_s = t0.elapsed().as_secs_f64();

    let wal_app = stack.wal_bytes_apparent() - wal_before_app;
    let wal_disk = stack.wal_bytes_on_disk().saturating_sub(wal_before_disk);
    let fsyncs = stack.fsyncs() - fsync_before;
    (
        PhaseReport {
            records: n,
            batch,
            wall_s,
            per_s: n as f64 / wall_s,
            wal_bytes_apparent: wal_app,
            wal_bytes_on_disk: wal_disk,
            bytes_per_rec_apparent: wal_app as f64 / n as f64,
            bytes_per_rec_on_disk: wal_disk as f64 / n as f64,
            fsyncs,
            fsync_per_rec: fsyncs as f64 / n as f64,
            intern_s,
            create_s,
            commit_s,
        },
        node_ids,
    )
}

/// Load `n` edges in batches of `batch` between already-committed nodes.
fn run_edge_phase(
    stack: &DurableStack,
    n: u64,
    batch: usize,
    nodes: &[arcgraph_core::NodeId],
) -> PhaseReport {
    assert!(nodes.len() >= 2, "need ≥2 nodes to wire edges");
    let wal_before_app = stack.wal_bytes_apparent();
    let wal_before_disk = stack.wal_bytes_on_disk();
    let fsync_before = stack.fsyncs();
    let (mut intern_s, mut create_s, mut commit_s) = (0.0f64, 0.0f64, 0.0f64);

    let t0 = Instant::now();
    let mut done = 0u64;
    let mut idx: u64 = 0;
    // Deterministic endpoint walk: src = i % N, dst = (i*2654435761+1) % N
    // (a cheap LCG-ish spread so edges are NOT all on one src chain —
    // mirrors web-Google's scattered adjacency).
    while done < n {
        let this = std::cmp::min(batch as u64, n - done);
        let mut tx = stack.txn_manager.begin(TENANT);
        for _ in 0..this {
            let src = nodes[(idx as usize) % nodes.len()];
            let dst =
                nodes[((idx.wrapping_mul(2_654_435_761).wrapping_add(1)) as usize) % nodes.len()];
            idx += 1;

            let ti = Instant::now();
            let ty: TypeId =
                intern_type_logged(&stack.intern, stack.crud.wal(), TENANT, REL_TYPE).unwrap();
            intern_s += ti.elapsed().as_secs_f64();

            let tc = Instant::now();
            create_rel(
                &stack.crud,
                &mut tx,
                TENANT,
                src,
                dst,
                ty,
                &PropertyData::Empty,
            )
            .unwrap();
            create_s += tc.elapsed().as_secs_f64();
        }
        let tk = Instant::now();
        commit(tx, &stack.crud).unwrap();
        commit_s += tk.elapsed().as_secs_f64();
        done += this;
    }
    let wall_s = t0.elapsed().as_secs_f64();

    let wal_app = stack.wal_bytes_apparent() - wal_before_app;
    let wal_disk = stack.wal_bytes_on_disk().saturating_sub(wal_before_disk);
    let fsyncs = stack.fsyncs() - fsync_before;
    PhaseReport {
        records: n,
        batch,
        wall_s,
        per_s: n as f64 / wall_s,
        wal_bytes_apparent: wal_app,
        wal_bytes_on_disk: wal_disk,
        bytes_per_rec_apparent: wal_app as f64 / n as f64,
        bytes_per_rec_on_disk: wal_disk as f64 / n as f64,
        fsyncs,
        fsync_per_rec: fsyncs as f64 / n as f64,
        intern_s,
        create_s,
        commit_s,
    }
}

fn n_env(default: u64) -> u64 {
    std::env::var("ARC849_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "heavy measurement harness; ARC849_REPRO=1 + --ignored --nocapture"]
fn step0_durable_ingest_batch_sweep_849() {
    repro_gate_or_panic();
    let n = n_env(20_000);
    println!("\n#849 STEP-0 — durable ingest throughput + WAL amplification");
    println!(
        "  fixture: {n} nodes / {n} edges, label=\"{NODE_LABEL}\", rel=\"{REL_TYPE}\", no props, Strict tier"
    );
    println!("  (one ingest call == one commit == one CommitBundle == one fsync cohort)\n");

    for &batch in &[1usize, 32, 256, 1000] {
        // Cap record count for the fsync-bound small batches so the
        // sweep stays minutes, not hours (one fsync per commit per
        // record at batch=1). WAL bytes/record is N-independent.
        let phase_n = if batch < 64 { n.min(2_000) } else { n };
        // Fresh durable store per batch size so WAL deltas are clean.
        let tmp = TempDir::new().unwrap();
        let stack = DurableStack::build(tmp.path());
        let (node_rep, nodes) = run_node_phase(&stack, phase_n, batch);
        let edge_rep = run_edge_phase(&stack, phase_n, batch, &nodes);
        node_rep.print("NODES");
        edge_rep.print("EDGES");
        let ratio = edge_rep.per_s / node_rep.per_s.max(1e-9);
        println!(
            "  → edge/node throughput ratio = {ratio:.2}×   (finding claims ~37×)   \
             node WAL/rec apparent={:.0}B (~{:.1} pages @8KiB)\n",
            node_rep.bytes_per_rec_apparent,
            node_rep.bytes_per_rec_apparent / 8192.0,
        );
    }
    println!(
        "  NOTE: apparent≈on_disk ⇒ WAL bytes are REAL written data (segments are not pre-allocated)."
    );
}

// ─────────────────────────────────────────────────────────────────────
// B1 root-cause regression GUARDS (always-on assertion tests)
// ─────────────────────────────────────────────────────────────────────

const REPRO_RUN_ENV: &str = "ARC849_REPRO";
const REPRO_SKIP_OK_ENV: &str = "ARCGRAPH_ARC849_REPRO_SKIP_OK";

/// Panic-by-default gate for the heavy `#[ignore]` measurement sweep
/// (W25-MFI-2; `feedback_test_env_gate_panic_by_default`). The sweep is
/// `#[ignore]`'d (off the default gauntlet); when a `--ignored` runner
/// invokes it, `ARC849_REPRO=1` must be set or it PANICS — never a
/// silent soft-skip. `ARCGRAPH_ARC849_REPRO_SKIP_OK=1` opts into a
/// loud-skip for hostile / CI hosts that run `--ignored` broadly.
fn repro_gate_or_panic() {
    let run = std::env::var(REPRO_RUN_ENV)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if run {
        return;
    }
    if std::env::var(REPRO_SKIP_OK_ENV).is_ok() {
        eprintln!(
            "step0_durable_ingest_batch_sweep_849: SKIPPING (opt-out via \
             {REPRO_SKIP_OK_ENV}=1) — set {REPRO_RUN_ENV}=1 to run the sweep"
        );
        return;
    }
    panic!(
        "step0_durable_ingest_batch_sweep_849: required run-flag \
         {REPRO_RUN_ENV}=1 not set. This heavy (~30 s) measurement sweep \
         is `#[ignore]`'d; when invoked via `--ignored`, {REPRO_RUN_ENV}=1 \
         must be set so it actually runs. Set {REPRO_RUN_ENV}=1 to run, or \
         {REPRO_SKIP_OK_ENV}=1 to opt into a soft-skip (hostile/CI envs \
         only). Soft-skipping silently after a `--ignored` bypass is the \
         W12δ HIGH-1 bug class (feedback_test_env_gate_panic_by_default)."
    );
}

/// **B1 strong oracle — no per-node storage penalty.**
///
/// Re-derives (deterministically, not via timing) the #849-B1 verdict:
/// at the SAME batch size the durable node-ingest path costs NO MORE
/// fsyncs and NO MORE WAL bytes than the edge path. The headline "152
/// nodes/s vs 5,672 edges/s (37×)" was two points on ONE
/// batch→throughput curve (152/s≈batch1, 5,672/s≈batch≈64) produced by a
/// loader that committed nodes ~singly and edges in batches — NOT a
/// per-node penalty. `create_rel` is in fact structurally HEAVIER than
/// `create_node` (it stages an extra TEL append), so node cost ≤ edge
/// cost at every batch.
///
/// RED-on-revert: if a future change makes the node path emit a per-node
/// WAL record / page snapshot / fsync that the edge path skips (e.g. a
/// per-node secondary-index write), `fsyncs(node) > fsyncs(edge)` or
/// `wal_bytes(node) > wal_bytes(edge)` and this test fails.
///
/// The deterministic counters (fsync count, WAL bytes) are the
/// load-bearing oracle; the throughput band is a coarse 100×-slack
/// backstop that stays robust under parallel-test contention (both
/// phases share the same noise in one thread) yet still fails on a real
/// 37× regression.
#[test]
fn node_vs_edge_have_no_structural_penalty_849() {
    // 256 records / batch 16 = 16 commits per phase → counters are small
    // + deterministic and the whole test is well under a second.
    let n: u64 = 256;
    let batch = 16usize;
    let tmp = TempDir::new().unwrap();
    let stack = DurableStack::build(tmp.path());
    let (node_rep, nodes) = run_node_phase(&stack, n, batch);
    let edge_rep = run_edge_phase(&stack, n, batch, &nodes);
    let commits = n.div_ceil(batch as u64);

    // 1) fsyncs identical node-vs-edge (the anti-37× oracle): both pay
    //    one fsync per commit cohort + exactly one intern log.
    assert_eq!(
        node_rep.fsyncs, edge_rep.fsyncs,
        "node and edge must fsync identically at batch={batch} \
         (node={}, edge={}); a per-node fsync would diverge here",
        node_rep.fsyncs, edge_rep.fsyncs,
    );
    // 2) ~1 fsync per COMMIT, never ~1 per RECORD (per-node amplification
    //    would be ≈ `n` fsyncs, not ≈ `commits`).
    assert!(
        node_rep.fsyncs <= commits + 2,
        "node fsyncs ({}) must track commits ({commits}), not records ({n})",
        node_rep.fsyncs,
    );
    // 3) node WAL bytes ≤ edge WAL bytes at the same batch (edge is
    //    heavier — the TEL append). Deterministic for this fixture.
    assert!(
        node_rep.wal_bytes_apparent <= edge_rep.wal_bytes_apparent,
        "node WAL bytes ({}) must not exceed edge WAL bytes ({}) at \
         batch={batch} — the node path must not be heavier than the edge path",
        node_rep.wal_bytes_apparent,
        edge_rep.wal_bytes_apparent,
    );
    // 4) Coarse throughput backstop: node/edge within one order of
    //    magnitude. A genuine 37× node penalty → ratio ≈ 0.027 → fails
    //    the 0.1 floor. The 100× slack keeps it non-flaky under load.
    let ratio = node_rep.per_s / edge_rep.per_s.max(1e-9);
    assert!(
        (0.1..=10.0).contains(&ratio),
        "node/edge throughput ratio {ratio:.3} outside [0.1, 10.0] at \
         batch={batch} (node={:.0}/s, edge={:.0}/s) — a 37× penalty would fail",
        node_rep.per_s,
        edge_rep.per_s,
    );
}

/// Recover a durable node store from `data_dir/wal` with a FULL
/// `PageStoreTarget` (record + blob + allocator-seed), mirroring the CLI
/// `build_durable` recovery wiring (`crates/arcgraph-cli/src/bootstrap.rs`).
/// Returns the recovered store + txn manager + the live writer (kept
/// alive for the read-back; the caller shuts it down).
fn recover_node_store(data_dir: &Path) -> (Arc<CrudStore>, Arc<TxnManager>, WalWriter) {
    let wal_dir = data_dir.join("wal");
    let writer = WalWriter::spawn(WalConfig::new(&wal_dir)).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(
            Arc::clone(&mgr),
            Arc::clone(&allocator),
            Some(handle.clone()),
        )
        .unwrap(),
    );
    let store = Arc::new(CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        Arc::clone(&allocator),
    ));
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let records_handle: Arc<dyn RecordPageStoreHandle> =
        Arc::clone(store.records().expect("record store wired")) as Arc<dyn RecordPageStoreHandle>;
    let blob_handle: Arc<dyn BlobStoreHandle> =
        Arc::clone(store.blob_store()) as Arc<dyn BlobStoreHandle>;
    let allocator_seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&store), Arc::clone(&allocator));
    let target = PageStoreTarget::primary_only(primary_handle)
        .with_record_store(records_handle)
        .with_blob_store(blob_handle)
        .with_allocator_seed(allocator_seed);
    recover_from_wal(&wal_dir, Arc::clone(&mgr), target, None).expect("recover_from_wal");
    (store, mgr, writer)
}

/// **Durability determinism guard (crash-atomicity, no silent loss).**
///
/// Ingests N nodes, each with a DISTINCT label + DISTINCT inline
/// property pair, through the durable Strict-tier stack; shuts the writer
/// down (flushing the WAL the way a clean process exit does); then
/// recovers into a fresh stack via `recover_from_wal` and asserts every
/// node returns with its EXACT label + properties. A dropped / swapped /
/// torn record fails the per-id equality — a strictly stronger oracle
/// than a recovered-count (`feedback_review_oracle_relaxations`). Mirrors
/// the K-1 durable round-trip discipline for the specific #849-B1 node
/// ingest path.
#[test]
fn durable_node_ingest_recovers_exact_849() {
    let tmp = TempDir::new().unwrap();
    let n: u64 = 50;
    let batch = 8usize; // spans multiple CommitBundles (coalescing path)

    // (node_id, label, prop_a, prop_b) — distinct per node so a swap or
    // partial loss cannot pass.
    let mut expected: Vec<(NodeId, u32, u32, u32)> = Vec::with_capacity(n as usize);
    {
        let stack = DurableStack::build(tmp.path());
        let mut i: u64 = 0;
        while i < n {
            let this = std::cmp::min(batch as u64, n - i);
            let mut tx = stack.txn_manager.begin(TENANT);
            for _ in 0..this {
                let label = 7_000 + i as u32;
                let a = 1_000_000 + i as u32;
                let b = 2_000_000 + i as u32;
                let id = create_node(
                    &stack.crud,
                    &mut tx,
                    TENANT,
                    LabelId::new(label),
                    &PropertyData::InlineU32Pair(a, b),
                )
                .unwrap();
                expected.push((id, label, a, b));
                i += 1;
            }
            commit(tx, &stack.crud).unwrap();
        }
        // `stack` drops here → WalWriter::shutdown flushes + fsyncs the
        // tail: the clean-process-exit durability boundary.
    }

    // Restart: recover from the on-disk WAL into a fresh stack.
    let (store, mgr, writer) = recover_node_store(tmp.path());
    let tx = mgr.begin(TENANT);
    for (id, label, a, b) in &expected {
        let rec = read_node_with_store(&store, &tx, *id)
            .expect("read_node_with_store")
            .unwrap_or_else(|| panic!("node {id:?} LOST after restart — silent data loss"));
        assert_eq!(
            rec.label_id, *label,
            "label mismatch after restart for {id:?}: got {}, want {label}",
            rec.label_id,
        );
        assert_eq!(
            (rec.inline_u32a, rec.inline_u32b),
            (*a, *b),
            "property mismatch after restart for {id:?}",
        );
    }
    let _ = writer.shutdown();
}
