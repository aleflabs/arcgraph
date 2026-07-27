//! M2-E5 — WAL recovery replay correctness + N-2 multi-page crash
//! atomicity fault injection.
//!
//! Contract (from M2.e prompt §M2-E5): 100 rounds of
//! (seed-child ⇒ SIGKILL ⇒ replay-child) where each round writes a
//! diverse WAL workload (Commit / InternString / PutBlob / IndexPage)
//! and every round exercises at least one of the two known multi-
//! page atomicity hazards — B-tree split paired emits and the
//! overflow-successor sequence. Gate: "100 rounds ran; each
//! divergence filed. Only harness-crashes fail the gate." Divergence
//! is EXPECTED at pre-redo-undo-protocol state per M2-34 follow-up;
//! the harness characterizes the failure shape rather than asserting
//! post-recovery equality.
//!
//! ## Why this is in `arcgraph-index/tests/`
//!
//! The hazard-site drivers use `SecondaryIndex::insert` directly to
//! force B-tree splits (unique-key bulk inserts) and overflow chains
//! (duplicate-key bulk inserts). Those live in `arcgraph-index`.
//! The baseline workload uses `CrudStore`, `BlobStore`, and
//! `InternTable`, which live in `arcgraph-storage` — and
//! `arcgraph-index` already depends on `arcgraph-storage`, so
//! pulling everything into the `arcgraph-index` test crate is the
//! only path that respects the existing bounded-context graph
//! (putting this under `arcgraph-storage/tests/` would require
//! `arcgraph-storage` to dev-depend on `arcgraph-index`, reversing
//! the established dep edge).
//!
//! ## Runtime gating
//!
//! `#[ignore]` by default. Run explicitly:
//!
//!   cargo test -p arcgraph-index --release \
//!     --test m2e_wal_recovery_replay -- --ignored --nocapture
//!
//! ### Environment overrides
//!
//!   M2E_E5_ROUNDS       (default 100)
//!   M2E_E5_KILL_MIN_MS  (default 50)
//!   M2E_E5_KILL_MAX_MS  (default 150)
//!
//! ## Self-reexec pattern (deviation from the M2.e prompt example)
//!
//! The M2.e prompt's example spawns `cargo run --release --bin
//! m2e_replay_worker`, but HARD BOUNDARIES forbid new binary
//! targets in Cargo.toml. Instead, the parent harness spawns
//! `std::env::current_exe()` — i.e., this same test binary — with
//! `M2E_E5_CHILD={seed|replay}` in the env. The `#[test]` fn
//! branches on that env var at entry: if set, run worker-mode and
//! `std::process::exit(0)` before falling into the parent loop.
//!
//! Functionally equivalent to the prompt's example:
//!   - Separate OS process (verified by different PIDs).
//!   - Separate virtual memory (verified by fresh stack
//!     construction on each spawn).
//!   - WAL lives on disk in a shared tempdir.
//!   - `child.kill()` sends SIGKILL on Unix, matching the spec's
//!     out-of-process kill requirement.
//!
//! ## Hazard-site round distribution (100 rounds)
//!
//! Round index mod 3 selects the hazard:
//!
//!   - 0 → `basic` — clean run-to-completion; validates the
//!     "restart without panic on clean WAL" path.
//!   - 1 → `split` — bulk secondary-index inserts on unique keys,
//!     SIGKILL during the paired-emit window.
//!   - 2 → `overflow` — bulk duplicate-key inserts on a single key,
//!     SIGKILL during the overflow-successor sequence.
//!
//! Split and overflow rounds park the child indefinitely after
//! staging the hazard workload so the SIGKILL lands mid-sequence.

use std::collections::BTreeMap;
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{LabelId, NodeId, StringId, TenantId, TypeId};
use arcgraph_index::{PropertyValue, SecondaryIndex, SecondaryKey};
use arcgraph_storage::blob::BlobStore;
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, create_rel};
use arcgraph_storage::intern::{InternTable, intern_logged};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{WalConfig, WalRecordType, WalRecoveryReader, WalWriter};

// ─── env helpers ──────────────────────────────────────────────────

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default)
}

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn hex_encode(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

// ─── snapshot format ──────────────────────────────────────────────
//
// Each line is one logical row. Deterministic ordering via BTreeMap
// iteration. Equality check is line-by-line. Avoids serde dep.
//
// Line formats:
//   NODE\t<tenant>\t<node_id>\t<bytes_hex>
//   REL\t<tenant>\t<rel_id>\t<bytes_hex>
//   INTERN\t<tenant>\t<string_id>\t<name>
//   BLOB\t<tenant>\t<head_page>\t<slot>\t<bytes_hex>
//   SK\t<key_hex>\t<node_id_csv>
//   META\t<label>\t<value>

#[derive(Default)]
struct Snapshot {
    nodes: BTreeMap<(u64, u64), Vec<u8>>,
    /// Key: (tenant_raw, rel_id); value: record bytes.
    rels: BTreeMap<(u64, u64), Vec<u8>>,
    /// Key: (tenant_raw, string_id); value: interned name.
    interned: BTreeMap<(u64, u32), String>,
    /// Key: (tenant_raw, head_page, slot); value: blob bytes.
    blobs: BTreeMap<(u64, u64, u16), Vec<u8>>,
    /// Key: encoded SecondaryKey bytes; value: sorted NodeId raws.
    secondary: BTreeMap<Vec<u8>, Vec<u64>>,
    /// Misc counters/markers.
    meta: BTreeMap<String, String>,
}

impl Snapshot {
    fn write_to(&self, path: &Path) -> std::io::Result<()> {
        let mut out = String::new();
        for ((t, id), v) in &self.nodes {
            out.push_str(&format!("NODE\t{t}\t{id}\t{}\n", hex_encode(v)));
        }
        for ((t, id), v) in &self.rels {
            out.push_str(&format!("REL\t{t}\t{id}\t{}\n", hex_encode(v)));
        }
        for ((t, id), name) in &self.interned {
            out.push_str(&format!("INTERN\t{t}\t{id}\t{name}\n"));
        }
        for ((t, head, slot), v) in &self.blobs {
            out.push_str(&format!("BLOB\t{t}\t{head}\t{slot}\t{}\n", hex_encode(v)));
        }
        for (key_bytes, node_ids) in &self.secondary {
            let csv = node_ids
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(",");
            out.push_str(&format!("SK\t{}\t{csv}\n", hex_encode(key_bytes)));
        }
        for (k, v) in &self.meta {
            out.push_str(&format!("META\t{k}\t{v}\n"));
        }
        std::fs::write(path, out)
    }

    fn total_rows(&self) -> usize {
        self.nodes.len()
            + self.rels.len()
            + self.interned.len()
            + self.blobs.len()
            + self.secondary.len()
    }
}

// ─── child: seed phase ────────────────────────────────────────────

/// Baseline workload applied in every seed round before hazard-
/// specific extensions. Covers every WAL record variant the
/// contract names except `TelAppend` (not yet emitted by the
/// current CRUD path — marked "future" in design-v2 §4.2).
fn run_seed_baseline(
    txn_mgr: &Arc<TxnManager>,
    store: &CrudStore,
    intern: &InternTable,
    blob: &BlobStore,
    wal_handle: &arcgraph_storage::wal::WalHandle,
    snap: &mut Snapshot,
) {
    // 1. ≥ 100 Commit records: 10 txns × 10 nodes each (100 creates,
    //    10 commits — commits carry 10 writes each).
    //
    //    We actually want ≥ 100 COMMIT records per the spec —
    //    easiest path: commit once per node so we get 100 Commit
    //    records plus 100 PutNode records.
    let mut node_seed_ids: Vec<(TenantId, NodeId)> = Vec::with_capacity(100);
    for i in 0..100u64 {
        let tenant = TenantId::new(1 + (i % 5)); // 5 tenants
        let mut tx = txn_mgr.begin(tenant);
        let label = LabelId::new((i % 8) as u32);
        let id = create_node(store, &mut tx, tenant, label, &PropertyData::Empty)
            .expect("seed create_node");
        commit(tx, store).expect("seed commit");
        node_seed_ids.push((tenant, id));
    }

    // 2. ≥ 10 rels so we get PutRel WAL records too. Cross-tenant
    //    pairs within the same tenant (MVCC rels are tenant-scoped).
    for i in 0..12u64 {
        let (tenant, src) = node_seed_ids[(i as usize) % node_seed_ids.len()];
        let (_, dst) = node_seed_ids[((i + 1) as usize) % node_seed_ids.len()];
        // Only create if both are in the same tenant; otherwise skip.
        let same_tenant_dst = node_seed_ids
            .iter()
            .find(|(t, d)| *t == tenant && *d != src)
            .copied();
        if let Some((_, dst2)) = same_tenant_dst {
            let mut tx = txn_mgr.begin(tenant);
            create_rel(
                store,
                &mut tx,
                tenant,
                src,
                dst2,
                TypeId::new((i % 4) as u32),
                &PropertyData::Empty,
            )
            .expect("seed create_rel");
            commit(tx, store).expect("seed commit rel");
        } else {
            let _ = dst;
        }
    }

    // 3. ≥ 50 InternString records across ≥ 5 tenants. Mix of
    //    distinct names per tenant.
    for tenant_idx in 0..5u64 {
        let tenant = TenantId::new(100 + tenant_idx);
        for name_idx in 0..12u64 {
            let name = format!("t{tenant_idx}_prop_{name_idx}");
            let id = intern_logged(intern, wal_handle, tenant, &name).expect("seed intern_logged");
            snap.interned.insert((tenant.raw(), id.raw()), name);
        }
    }

    // 4. ≥ 10 blobs: 6 single-page (< page size) + 4 multi-page.
    for i in 0..10u64 {
        let tenant = TenantId::new(200 + (i % 3));
        let is_multi_page = i >= 6;
        let size = if is_multi_page {
            // Multi-page: 16 KiB so we span ≥ 2 pages (page size is
            // 8 KiB in the blob layout).
            16 * 1024
        } else {
            // Single-page: < 1 KiB.
            512 + (i as usize) * 64
        };
        let bytes: Vec<u8> = (0..size as u64).map(|j| ((j ^ i) & 0xFF) as u8).collect();
        let bref = blob
            .put_logged(wal_handle, tenant, &bytes)
            .expect("seed put_logged");
        snap.blobs
            .insert((tenant.raw(), bref.page_id, bref.slot_id), bytes.clone());
    }

    // 5. IndexPage WAL records are emitted as a byproduct of step 1
    //    (PrimaryIndex page-image writes) and will also be driven by
    //    the hazard-specific phases below. Nothing extra needed here.

    // Snapshot node records via crud read path (reads from the slotted
    // record store; primary-index page resolution).
    for (tenant, id) in &node_seed_ids {
        // Fresh read tx per id so snapshot LSN is current.
        let reader = txn_mgr.begin(*tenant);
        match arcgraph_storage::crud::read_node_with_store(store, &reader, *id) {
            Ok(Some(rec)) => {
                snap.nodes
                    .insert((tenant.raw(), id.raw()), rec.to_bytes().to_vec());
            }
            _ => { /* tombstone or missing — skip (shouldn't happen in seed baseline) */ }
        }
        reader.abort();
    }
    // Rels: iterate the rel high-water-mark per tenant and try each id.
    for (tenant, _id) in &node_seed_ids {
        let reader = txn_mgr.begin(*tenant);
        let hw = store.rel_high_water(*tenant);
        for rid in 1..=hw {
            let rel_id = arcgraph_core::RelId::new(rid);
            if let Ok(Some(rec)) =
                arcgraph_storage::crud::read_rel_with_store(store, &reader, rel_id)
            {
                snap.rels
                    .insert((tenant.raw(), rid), rec.to_bytes().to_vec());
            }
        }
        reader.abort();
    }
    snap.meta.insert(
        "baseline_seed_nodes".into(),
        format!("{}", node_seed_ids.len()),
    );
    snap.meta
        .insert("baseline_tenants".into(), "1,2,3,4,5".into());
}

/// Drive the B-tree split hazard: bulk unique-key inserts on the
/// secondary index so leaves split, each split emitting a paired
/// (new-right-leaf, updated-left-leaf) pair of `IndexPage` WAL
/// records. The child then parks — parent SIGKILLs mid-sequence.
fn run_seed_hazard_split(secondary: &Arc<SecondaryIndex>, snap: &mut Snapshot, n_inserts: u64) {
    let tenant = TenantId::new(777);
    let label = LabelId::new(42);
    let property_key = StringId::new(1); // Fictional interned prop name.
    for i in 0..n_inserts {
        // Unique key per insert → triggers splits once leaf fills up.
        let key = SecondaryKey::new(tenant, label, property_key, PropertyValue::U64(i + 1));
        let node = NodeId::new(i + 1);
        secondary.insert(key, node).expect("hazard-split insert");
        // Capture into snapshot map so we can compare post-recovery.
        let mut key_bytes = vec![0u8; SecondaryKey::SIZE];
        key.encode_into(&mut key_bytes).expect("key encode");
        let entry = snap.secondary.entry(key_bytes).or_default();
        entry.push(node.raw());
    }
    snap.meta.insert("hazard".into(), "split".into());
    snap.meta
        .insert("hazard_split_inserts".into(), n_inserts.to_string());
}

/// Drive the overflow-chain hazard: bulk duplicate-key inserts on a
/// single SecondaryKey so the inline slot fills up and the overflow-
/// successor sequence fires (emit overflow page, install, update
/// predecessor `next` pointer). The child parks — parent SIGKILLs
/// mid-sequence.
fn run_seed_hazard_overflow(secondary: &Arc<SecondaryIndex>, snap: &mut Snapshot, n_dups: u64) {
    let tenant = TenantId::new(888);
    let label = LabelId::new(99);
    let property_key = StringId::new(2);
    let key = SecondaryKey::new(tenant, label, property_key, PropertyValue::U64(123456));
    for i in 0..n_dups {
        let node = NodeId::new(1_000_000 + i);
        secondary.insert(key, node).expect("hazard-overflow insert");
        let mut key_bytes = vec![0u8; SecondaryKey::SIZE];
        key.encode_into(&mut key_bytes).expect("key encode");
        let entry = snap.secondary.entry(key_bytes).or_default();
        entry.push(node.raw());
    }
    snap.meta.insert("hazard".into(), "overflow".into());
    snap.meta
        .insert("hazard_overflow_dups".into(), n_dups.to_string());
}

fn child_run_seed(wal_dir: PathBuf, snap_path: PathBuf, round: u64, hazard: &str) -> ! {
    // ─── bring up full stack with WAL ──────────────────────────────
    let wal_config = WalConfig {
        dir: wal_dir,
        segment_size_bytes: 64 * 1024 * 1024, // 64 MiB default
        group_commit_window: Duration::from_millis(1),
        group_commit_max_batch: 16,
        metrics_sink: None,
        encryption: None,

        inflight_budget_bytes: None,
    };
    let wal_writer = WalWriter::spawn(wal_config).expect("seed-child: wal spawn");
    let wal_handle = wal_writer.handle();

    let txn_mgr = Arc::new(TxnManager::with_wal(wal_handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(
            Arc::clone(&txn_mgr),
            Arc::clone(&alloc),
            Some(wal_handle.clone()),
        )
        .expect("seed-child: PrimaryIndex"),
    );
    let secondary = Arc::new(
        SecondaryIndex::new(
            Arc::clone(&txn_mgr),
            Arc::clone(&alloc),
            Some(wal_handle.clone()),
        )
        .expect("seed-child: SecondaryIndex"),
    );
    let secondary_as_handle: Arc<dyn arcgraph_storage::secondary_handle::SecondaryIndexHandle> =
        Arc::clone(&secondary) as _;
    let store = CrudStore::new_with_indices(
        Some(wal_handle.clone()),
        Arc::clone(&primary),
        Some(secondary_as_handle),
        Arc::clone(&alloc),
    );
    let intern = InternTable::new();
    let blob = BlobStore::new();

    let mut snap = Snapshot::default();
    snap.meta.insert("round".into(), round.to_string());
    snap.meta.insert("hazard_requested".into(), hazard.into());

    // ─── baseline workload ─────────────────────────────────────────
    run_seed_baseline(&txn_mgr, &store, &intern, &blob, &wal_handle, &mut snap);

    // ─── hazard-specific phase ─────────────────────────────────────
    match hazard {
        "basic" => {
            snap.meta.insert("hazard".into(), "basic".into());
        }
        "split" => {
            // Leaves hold ~LEAF_CAPACITY entries; push enough inserts
            // to force ≥ 5 splits so we have a strong chance of
            // landing SIGKILL mid-paired-emit.
            run_seed_hazard_split(&secondary, &mut snap, 400);
        }
        "overflow" => {
            // 5 dups stack into inline. The 5th (or beyond) triggers
            // overflow-successor. Push 20 to force multiple successor
            // allocations — more hazard opportunities per round.
            run_seed_hazard_overflow(&secondary, &mut snap, 20);
        }
        other => panic!("unknown M2E_E5_HAZARD: {other}"),
    }

    // ─── persist + flush ──────────────────────────────────────────
    snap.write_to(&snap_path).expect("seed-child: write snap");
    wal_handle.flush().expect("seed-child: wal flush");

    // Signal readiness to parent.
    println!("READY:{round}:{hazard}:rows={}", snap.total_rows());
    std::io::stdout().flush().ok();

    if hazard == "basic" {
        // Clean exit for the basic round.
        // Shutting down the WAL here ensures no in-flight batch is
        // lost when we exit — the basic round is the only one that
        // gets to flush-then-exit cleanly.
        drop(store);
        drop(primary);
        drop(secondary);
        drop(txn_mgr);
        drop(wal_handle);
        let _ = wal_writer.shutdown();
        println!("DONE:{round}");
        std::io::stdout().flush().ok();
        std::process::exit(0);
    }

    // Hazard rounds: park indefinitely. The parent SIGKILLs mid-
    // sequence. Keep the WAL handle open so in-flight batches remain
    // up to whatever fsync boundary the kill catches.
    //
    // We drive additional hazard work in a loop to maximize the
    // probability of SIGKILL landing INSIDE a multi-page sequence,
    // not just after one completed. Each iteration does one more
    // split / overflow insert, respectively.
    let mut spin = 0u64;
    let tenant_split = TenantId::new(779);
    let label_split = LabelId::new(43);
    let pk_split = StringId::new(3);
    let tenant_overflow = TenantId::new(890);
    let label_overflow = LabelId::new(100);
    let pk_overflow = StringId::new(4);
    let key_overflow = SecondaryKey::new(
        tenant_overflow,
        label_overflow,
        pk_overflow,
        PropertyValue::U64(0xDEAD_BEEF),
    );
    loop {
        spin += 1;
        match hazard {
            "split" => {
                let k = SecondaryKey::new(
                    tenant_split,
                    label_split,
                    pk_split,
                    PropertyValue::U64(10_000 + spin),
                );
                let _ = secondary.insert(k, NodeId::new(2_000_000 + spin));
            }
            "overflow" => {
                let _ = secondary.insert(key_overflow, NodeId::new(3_000_000 + spin));
            }
            _ => unreachable!(),
        }
        // No sleep — we want the tightest possible inner loop so the
        // SIGKILL from parent catches us in the middle of a multi-
        // page sequence.
        if spin > 1_000_000 {
            // Safety valve: if the parent somehow hasn't killed us
            // after 1 M iterations (~unreachable), exit so we don't
            // loop forever.
            std::process::exit(77);
        }
    }
}

// ─── child: replay phase ──────────────────────────────────────────

fn child_run_replay(wal_dir: PathBuf, snap_path: PathBuf) -> ! {
    // 1. Open the WAL directory for replay.
    let reader = WalRecoveryReader::open(&wal_dir).expect("replay-child: open WAL dir");

    // 2. Iterate every record, keeping a histogram by type.
    let mut variant_counts: BTreeMap<u8, u64> = BTreeMap::new();
    let mut last_lsn: u64 = 0;
    let mut torn_tail: Option<String> = None;
    let mut error: Option<String> = None;

    // Consume the iterator.
    let mut reader_iter = reader;
    loop {
        match reader_iter.next() {
            Some(Ok(rec)) => {
                *variant_counts.entry(rec.record_type as u8).or_insert(0) += 1;
                if rec.lsn.raw() > last_lsn {
                    last_lsn = rec.lsn.raw();
                }
            }
            Some(Err(e)) => {
                error = Some(format!("{e:?}"));
                break;
            }
            None => break,
        }
    }
    if let Some(t) = reader_iter.torn_tail() {
        torn_tail = Some(format!("segment={} offset={}", t.segment, t.offset));
    }

    // 3. Build a fresh (unpopulated) stack on top of the SAME wal_dir
    //    so we can observe state-after-restart. Replay is NOT wired
    //    for IndexPage / InternString / PutBlob at M2.e start, so the
    //    fresh stack is empty by design — we're asserting "process
    //    constructs without panic on a populated WAL dir."
    let wal_config = WalConfig {
        dir: wal_dir.clone(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: Duration::from_millis(1),
        group_commit_max_batch: 16,
        metrics_sink: None,
        encryption: None,

        inflight_budget_bytes: None,
    };
    // spawn_from advances LSN counter past the pre-kill tail.
    let wal_writer = WalWriter::spawn_from(wal_config, arcgraph_core::Lsn::new(last_lsn))
        .expect("replay-child: wal spawn_from");
    let wal_handle = wal_writer.handle();
    let txn_mgr = Arc::new(TxnManager::with_wal(wal_handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(
            Arc::clone(&txn_mgr),
            Arc::clone(&alloc),
            Some(wal_handle.clone()),
        )
        .expect("replay-child: PrimaryIndex"),
    );
    let secondary = Arc::new(
        SecondaryIndex::new(
            Arc::clone(&txn_mgr),
            Arc::clone(&alloc),
            Some(wal_handle.clone()),
        )
        .expect("replay-child: SecondaryIndex"),
    );
    let secondary_as_handle: Arc<dyn arcgraph_storage::secondary_handle::SecondaryIndexHandle> =
        Arc::clone(&secondary) as _;
    let _store = CrudStore::new_with_indices(
        Some(wal_handle.clone()),
        Arc::clone(&primary),
        Some(secondary_as_handle),
        Arc::clone(&alloc),
    );
    let _intern = InternTable::new();
    let _blob = BlobStore::new();

    // 4. Build an empty snapshot (fresh stack has no data) + record
    //    WAL metadata so the parent can classify divergence shape.
    let mut snap = Snapshot::default();
    snap.meta
        .insert("replay_last_lsn".into(), last_lsn.to_string());
    snap.meta.insert(
        "replay_torn_tail".into(),
        torn_tail.unwrap_or_else(|| "none".into()),
    );
    snap.meta.insert(
        "replay_error".into(),
        error.unwrap_or_else(|| "none".into()),
    );
    for (ty, count) in &variant_counts {
        let ty_name = match WalRecordType::from_byte(*ty) {
            Ok(v) => format!("{v:?}"),
            Err(_) => format!("unknown-0x{ty:02x}"),
        };
        snap.meta
            .insert(format!("replay_wal_count_{ty_name}"), count.to_string());
    }
    // Record that the fresh stack constructed without panic.
    snap.meta
        .insert("replay_stack_constructed".into(), "true".into());

    snap.write_to(&snap_path).expect("replay-child: write snap");

    // Clean shutdown of the fresh WAL writer we spawned (purely so we
    // don't leak a thread; nothing was appended).
    drop(txn_mgr);
    drop(wal_handle);
    let _ = wal_writer.shutdown();

    println!("DONE:replay");
    std::io::stdout().flush().ok();
    std::process::exit(0);
}

// ─── parent: round loop ───────────────────────────────────────────

#[derive(Default, Debug)]
struct DivergenceTally {
    rounds_clean: u64,
    rounds_diverged: u64,
    diff_shapes: BTreeMap<String, u64>,
}

fn run_parent(n_rounds: u64, kill_min_ms: u64, kill_max_ms: u64) {
    eprintln!("M2-E5 harness: {n_rounds} rounds  kill window {kill_min_ms}..{kill_max_ms} ms");
    let mut basic = DivergenceTally::default();
    let mut split = DivergenceTally::default();
    let mut overflow = DivergenceTally::default();
    let mut rng_state: u64 = 0xD00D_BEEF_F00D_F00D;

    for round in 0..n_rounds {
        let hazard = match round % 3 {
            0 => "basic",
            1 => "split",
            _ => "overflow",
        };
        let wal_dir = tempfile::tempdir().expect("tempdir wal");
        let snap_pre = tempfile::NamedTempFile::new().expect("tempfile pre");
        let snap_post = tempfile::NamedTempFile::new().expect("tempfile post");

        // ─── spawn seed child ──────────────────────────────────────
        let exe = env::current_exe().expect("current_exe");
        let mut seed_cmd = Command::new(&exe);
        seed_cmd
            .arg("--ignored")
            .arg("--exact")
            .arg("wal_recovery_replay_100_rounds")
            .arg("--nocapture")
            .env("M2E_E5_CHILD", "seed")
            .env("M2E_E5_WAL_DIR", wal_dir.path())
            .env("M2E_E5_SNAP_PATH", snap_pre.path())
            .env("M2E_E5_ROUND", round.to_string())
            .env("M2E_E5_HAZARD", hazard)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut seed = seed_cmd.spawn().expect("spawn seed child");

        // Wait for the READY line.
        let stdout = seed.stdout.take().expect("seed stdout piped");
        let mut reader = BufReader::new(stdout);
        let mut ready_seen = false;
        let mut done_seen = false;
        let mut ready_rows: u64 = 0;
        let mut line = String::new();
        loop {
            line.clear();
            let n = reader.read_line(&mut line).unwrap_or(0);
            if n == 0 {
                break;
            }
            let l = line.trim_end_matches('\n');
            if let Some(rest) = l.strip_prefix("READY:") {
                // Format: <round>:<hazard>:rows=<n>
                ready_seen = true;
                let parts: Vec<&str> = rest.split(':').collect();
                if parts.len() >= 3
                    && let Some(rows_str) = parts[2].strip_prefix("rows=")
                    && let Ok(v) = rows_str.parse::<u64>()
                {
                    ready_rows = v;
                }
                if hazard != "basic" {
                    // Hazard rounds park after READY; stop reading.
                    break;
                }
            } else if l.starts_with("DONE:") {
                done_seen = true;
                break;
            }
        }
        assert!(
            ready_seen,
            "round {round} ({hazard}): seed child did not emit READY"
        );

        // ─── hazard rounds: SIGKILL mid-sequence ───────────────────
        if hazard == "basic" {
            // Wait for DONE / clean exit.
            let status = seed.wait().expect("seed wait");
            assert!(
                status.success(),
                "round {round} basic: seed child exited non-zero: {status:?}"
            );
            assert!(done_seen, "round {round} basic: DONE not observed");
        } else {
            let delay_ms = kill_min_ms + (xorshift64(&mut rng_state) % (kill_max_ms - kill_min_ms));
            std::thread::sleep(Duration::from_millis(delay_ms));
            seed.kill().expect("kill seed child");
            // Reap the zombie. Status reflects the signal.
            let _status = seed.wait().expect("seed wait after kill");
        }

        // ─── spawn replay child on same wal_dir ────────────────────
        let mut replay_cmd = Command::new(&exe);
        replay_cmd
            .arg("--ignored")
            .arg("--exact")
            .arg("wal_recovery_replay_100_rounds")
            .arg("--nocapture")
            .env("M2E_E5_CHILD", "replay")
            .env("M2E_E5_WAL_DIR", wal_dir.path())
            .env("M2E_E5_SNAP_PATH", snap_post.path())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut replay = replay_cmd.spawn().expect("spawn replay child");
        let status = replay.wait().expect("replay wait");
        assert!(
            status.success(),
            "round {round} ({hazard}): replay child crashed — status {status:?}. \
             This is a harness-level failure (the 'restart without panic' invariant is broken)."
        );

        // ─── compare snapshots ─────────────────────────────────────
        let pre_text = std::fs::read_to_string(snap_pre.path()).unwrap_or_default();
        let post_text = std::fs::read_to_string(snap_post.path()).unwrap_or_default();

        let tally: &mut DivergenceTally = match hazard {
            "basic" => &mut basic,
            "split" => &mut split,
            "overflow" => &mut overflow,
            _ => unreachable!(),
        };

        if pre_text == post_text {
            tally.rounds_clean += 1;
        } else {
            tally.rounds_diverged += 1;
            // Dump both for post-mortem.
            let diff_dir = PathBuf::from(format!(
                "target/m2e-replay-diff/round_{:03}_{}",
                round, hazard
            ));
            std::fs::create_dir_all(&diff_dir).ok();
            let _ = std::fs::write(diff_dir.join("pre.tsv"), &pre_text);
            let _ = std::fs::write(diff_dir.join("post.tsv"), &post_text);
            // Classify the divergence shape.
            let shape = classify_divergence(&pre_text, &post_text);
            *tally.diff_shapes.entry(shape).or_insert(0) += 1;
        }

        if round % 10 == 9 {
            eprintln!(
                "  [round {round:>3}] hazard={hazard} ready_rows={ready_rows} \
                 clean(basic/split/overflow)={}/{}/{} diverged={}/{}/{}",
                basic.rounds_clean,
                split.rounds_clean,
                overflow.rounds_clean,
                basic.rounds_diverged,
                split.rounds_diverged,
                overflow.rounds_diverged,
            );
        }
    }

    // ─── report ────────────────────────────────────────────────────
    eprintln!();
    eprintln!("═══════════════════════ M2-E5 RESULTS ═══════════════════════");
    eprintln!("  rounds total:        {n_rounds}");
    for (hazard_name, tally) in [
        ("basic", &basic),
        ("split", &split),
        ("overflow", &overflow),
    ]
    .iter()
    {
        let total = tally.rounds_clean + tally.rounds_diverged;
        eprintln!("  hazard '{hazard_name}':");
        eprintln!("    rounds:            {total}");
        eprintln!("    clean match:       {}", tally.rounds_clean);
        eprintln!("    diverged:          {}", tally.rounds_diverged);
        for (shape, count) in &tally.diff_shapes {
            eprintln!("      shape {shape}: {count} rounds");
        }
    }
    eprintln!();
    eprintln!("  Divergence is EXPECTED at pre-redo-undo-protocol state");
    eprintln!("  (per M2-34 follow-up). IndexPage / InternString / PutBlob");
    eprintln!("  apply-on-replay is not wired at M2.e start. The 100-round");
    eprintln!("  gate is 'harness ran; shapes filed' — not equality.");
    eprintln!();
    eprintln!("  Diff dumps: target/m2e-replay-diff/round_NNN_<hazard>/");
    eprintln!("  File per-hazard issue comments on #37 with the observed shape.");
    eprintln!("══════════════════════════════════════════════════════════════");
}

/// Summarize a divergence as one of a small set of shapes. Returns
/// the shape identifier for tally keying.
fn classify_divergence(pre: &str, post: &str) -> String {
    let pre_rows = pre.lines().count();
    let post_rows = post.lines().count();
    // Exclude META lines from the "data" row count.
    let pre_data = pre.lines().filter(|l| !l.starts_with("META\t")).count();
    let post_data = post.lines().filter(|l| !l.starts_with("META\t")).count();
    if post_data == 0 && pre_data > 0 {
        format!("empty-post  (pre_data={pre_data} post_data=0)")
    } else if pre_data > post_data {
        format!("silent-drop (pre_data={pre_data} post_data={post_data})")
    } else if post_data > pre_data {
        format!("post>pre    (pre_data={pre_data} post_data={post_data})")
    } else {
        format!(
            "same-count-diff (rows={pre_rows} meta-lines-match? {})",
            pre_rows == post_rows
        )
    }
}

// ─── entry point ──────────────────────────────────────────────────

#[test]
#[ignore = "100-round WAL recovery replay with SIGKILL fault injection; run with --ignored per M2-E5 contract"]
fn wal_recovery_replay_100_rounds() {
    if let Ok(mode) = env::var("M2E_E5_CHILD") {
        let wal_dir = PathBuf::from(env::var("M2E_E5_WAL_DIR").expect("M2E_E5_WAL_DIR"));
        let snap_path = PathBuf::from(env::var("M2E_E5_SNAP_PATH").expect("M2E_E5_SNAP_PATH"));
        match mode.as_str() {
            "seed" => {
                let round: u64 = env::var("M2E_E5_ROUND")
                    .expect("M2E_E5_ROUND")
                    .parse()
                    .expect("M2E_E5_ROUND parse");
                let hazard = env::var("M2E_E5_HAZARD").unwrap_or_else(|_| "basic".into());
                child_run_seed(wal_dir, snap_path, round, &hazard);
            }
            "replay" => {
                child_run_replay(wal_dir, snap_path);
            }
            other => panic!("unknown M2E_E5_CHILD mode: {other}"),
        }
    }
    // Parent mode.
    let n_rounds = env_u64("M2E_E5_ROUNDS", 100);
    let kill_min = env_u64("M2E_E5_KILL_MIN_MS", 50);
    let kill_max = env_u64("M2E_E5_KILL_MAX_MS", 150).max(kill_min + 1);
    run_parent(n_rounds, kill_min, kill_max);
}

// ─── unit smoke: classify_divergence ──────────────────────────────
//
// Runs on every cargo test so we catch regressions in the
// shape-classifier without invoking the 100-round harness.

#[test]
fn classify_empty_post_diagnoses_full_wipe() {
    let pre = "NODE\t1\t1\tde\nMETA\thazard\tbasic\n";
    let post = "META\treplay_stack_constructed\ttrue\n";
    let shape = classify_divergence(pre, post);
    assert!(
        shape.starts_with("empty-post"),
        "expected empty-post, got {shape}"
    );
}

#[test]
fn classify_same_data_same_shape_is_silent_drop_or_same() {
    let pre = "NODE\t1\t1\tde\nNODE\t1\t2\tad\n";
    let post = "NODE\t1\t1\tde\n";
    let shape = classify_divergence(pre, post);
    assert!(
        shape.starts_with("silent-drop"),
        "expected silent-drop, got {shape}"
    );
}

#[test]
fn snapshot_roundtrip_writes_sorted_rows() {
    let mut snap = Snapshot::default();
    snap.nodes.insert((1, 3), vec![0xDE, 0xAD]);
    snap.nodes.insert((1, 1), vec![0xBE, 0xEF]);
    snap.meta.insert("hazard".into(), "basic".into());
    let f = tempfile::NamedTempFile::new().unwrap();
    snap.write_to(f.path()).unwrap();
    let text = std::fs::read_to_string(f.path()).unwrap();
    // BTreeMap sort → NodeId 1 before NodeId 3.
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "NODE\t1\t1\tbeef");
    assert_eq!(lines[1], "NODE\t1\t3\tdead");
    assert_eq!(lines.last().copied(), Some("META\thazard\tbasic"));
}
