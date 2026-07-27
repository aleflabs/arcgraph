//! v2 M1 migrate-on-open — the DEC-4 → slotted re-encode sweep
//! (ADR-230 M1; design `m1-m2-m4-m5-impl-designs.md` §0.1/§0.2/§M1.4;
//! `v2-build-plan.md` §2 M1 EXIT gate 4).
//!
//! # What this does
//!
//! A pre-M1 (`data_dir_version` stamp `1`) store encodes EVERY property
//! bag as a dedicated DEC-4 page chain. This sweep walks the live
//! records, and for each whose `property_ref` is a chained SMALL bag
//! (≤ [`PROP_BAG_MAX_BYTES`]): re-writes the SAME bag bytes through the
//! normal update path — which, at M1, packs them into shared slotted
//! pages — then reclaims the dead chain pages. Large bags stay chained
//! (the design §M1.2 overflow tail). Bag BYTES are untouched (JSON in,
//! JSON out — M1 is packing only), so correctness reduces to "every
//! live bag round-trips", provable per record.
//!
//! # How §0.2's contract maps onto this engine (the adaptation)
//!
//! The design's §0.2 mechanics (`props.store.migrating` file +
//! `migration.progress` marker) presuppose a standalone on-disk props
//! store. In THIS engine blob pages are VIRTUAL — the durable carriers
//! are the WAL (`CommitBundle` staged pages) and the ADR-229 checkpoint;
//! there is no props store file to shadow-copy. The §0.2 invariant
//! ("atomic-or-RESUMABLE, old data intact on crash") is met by riding
//! the engine's OWN atomicity machinery instead of inventing a parallel
//! one:
//!
//! - **Forward-only, idempotent per unit:** the unit is one record's
//!   `property_ref`. A migrated ref has `slot_id >= 1` and is skipped on
//!   re-run; an unmigrated ref is re-encoded. Re-running the sweep on
//!   any prefix of prior progress converges.
//! - **Never mutates the source:** the re-encode writes NEW slotted
//!   pages at NEW page ids + NEW MVCC record versions through
//!   [`crate::crud::update_node`]/[`update_rel`] — the same tested
//!   write path every production update uses (MVCC + record page +
//!   index + WAL bundle, all-or-nothing per batch commit). Old chain
//!   pages and old versions are untouched until AFTER the batch's
//!   commit is durable (commit-then-remove; see "chain reclaim" below).
//! - **Crash mid-sweep:** the store re-opens and recovery replays
//!   checkpoint + WAL exactly as for any crash — committed batches are
//!   present (slotted), uncommitted ones absent (still chained), every
//!   bag readable in both forms (the M1 read path dispatches per ref).
//!   The sweep resumes and completes. Never a torn state, because no
//!   state exists outside the engine's own crash-safe commits.
//! - **Single commit point:** the caller (bootstrap) rewrites the
//!   MANIFEST `props_store_format: "slotted-v1-migrating"` →
//!   `"slotted-v1"` (crash-atomic rename, [`crate::manifest`]) only
//!   after the sweep returns Ok — the §0.2 MANIFEST-swap commit point.
//!
//! # Chain reclaim (commit-then-remove)
//!
//! After a batch's commit is durable, the superseded chains' pages are
//! removed from the blob store (`remove_uncommitted_chain` — the
//! mechanism is "walk + remove", its name predates this second caller).
//! Removal strictly AFTER the commit fsync means a crash between the
//! two leaves ≤ one batch of dead-but-present chain pages; they are
//! unreferenced (harmless), and the sweep-end checkpoint the bootstrap
//! fires drops them from the durable image. The inverse order would be
//! data LOSS on crash (bag deleted before its replacement is durable) —
//! never do that.
//!
//! # MVCC / visibility
//!
//! The sweep runs at boot, under the data-dir `LOCK`, before the server
//! accepts connections: there are no concurrent snapshots. Each batch
//! is a normal transaction; the new version supersedes the old with
//! identical visible property bytes (`update_node` preserves the label;
//! bag bytes are byte-identical by construction). CDC: the plain
//! `update_node`/`update_rel` wrappers stage empty-diff events into the
//! subscriber-less boot-time broker — dropped on flush, never observed.
//!
//! # Memory (OOM-guardrail conformance)
//!
//! The sweep never materializes the candidate set: it streams record
//! pages one at a time (≤ 119 records each), accumulates at most
//! `batch_size` (id, kind) pairs + one bag at a time, and commits.
//! Peak = O(batch_size × avg-bag) ≈ 512 × 150 B ≈ 77 KB. The one
//! whole-store collection is `RecordPageStore::iter_pages`'s page-id +
//! latch-handle Vec (16 B/page — the same pre-existing shape
//! `bootstrap_primary_index` uses on every durable boot).
//!
//! # Budget (PD#5)
//!
//! Per migrated record: one chain read (~1 page re-fault worst case) +
//! one slotted append (+CRC) + its share of a batch commit fsync +
//! chain-page removals. Dominated by the batch fsync ⟹ throughput ≈
//! batch ingest speed (thousands/s), one-shot at first M1 boot.

use thiserror::Error;

use arcgraph_core::TenantId;
use arcgraph_core::record::PageType;

use crate::crud::{
    self, CrudError, CrudStore, PropertyData, decode_node_bytes, decode_rel_bytes, node_mvcc_key,
    rel_mvcc_key, update_node, update_rel,
};
use crate::property::BlobRef;
use crate::records::{PROP_BAG_MAX_BYTES, SlottedPageRef};
use crate::transaction::TxnManager;

/// Records (nodes + rels) re-encoded per migration transaction. Sized
/// so a batch's WAL bundle stays small (~512 bags ≈ 4-5 slotted pages
/// ≈ 40 KB of staged images) while amortizing the commit fsync.
pub const M1_MIGRATE_BATCH_SIZE: usize = 512;

/// Environment variable (TEST-ONLY, crash-injection) — when set to an
/// integer N, the sweep calls `std::process::abort()` immediately after
/// its N-th batch commit returns (before the batch's chain reclaim).
/// This is the §0.2 crash-during-migration gate's kill point: the
/// hardest window (commit durable, chains not yet reclaimed, MANIFEST
/// not yet flipped). Production never sets it.
pub const ENV_M1_MIGRATE_CRASH_AFTER_BATCHES: &str = "ARCGRAPH_M1_MIGRATE_CRASH_AFTER_BATCHES";

/// Sweep options.
#[derive(Debug, Clone)]
pub struct M1MigrateOptions {
    /// Records per migration transaction.
    pub batch_size: usize,
    /// TEST-ONLY crash injection: abort the process after this many
    /// batch commits (see [`ENV_M1_MIGRATE_CRASH_AFTER_BATCHES`]).
    pub crash_after_batches: Option<u64>,
}

impl Default for M1MigrateOptions {
    fn default() -> Self {
        Self {
            batch_size: M1_MIGRATE_BATCH_SIZE,
            crash_after_batches: None,
        }
    }
}

impl M1MigrateOptions {
    /// Default options + the crash-injection env hook (test-only).
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            crash_after_batches: std::env::var(ENV_M1_MIGRATE_CRASH_AFTER_BATCHES)
                .ok()
                .and_then(|s| s.parse::<u64>().ok()),
            ..Self::default()
        }
    }
}

/// What the sweep did — logged by the bootstrap + asserted by the
/// migration gates.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct M1MigrateReport {
    /// Node records re-encoded chained → slotted.
    pub nodes_migrated: u64,
    /// Rel records re-encoded chained → slotted.
    pub rels_migrated: u64,
    /// Superseded chain HEAD pages reclaimed (chains may span multiple
    /// pages; this counts chains, not pages).
    pub chains_removed: u64,
    /// Records whose ref was already slotted (`slot_id >= 1`) — the
    /// idempotent-resume skip.
    pub already_slotted: u64,
    /// Records whose bag exceeds [`PROP_BAG_MAX_BYTES`] — kept chained
    /// by design (the §M1.2 overflow tail).
    pub kept_chained_large: u64,
    /// Migration transactions committed.
    pub batches_committed: u64,
}

impl M1MigrateReport {
    /// True iff the sweep found nothing left to convert (a fully
    /// migrated store — the idempotent re-run shape).
    #[must_use]
    pub fn was_noop(&self) -> bool {
        self.nodes_migrated == 0 && self.rels_migrated == 0
    }
}

/// Faults surfaced by the sweep. Any error leaves the store fully
/// readable at its current (mixed but valid) state; the MANIFEST is
/// not flipped, so the next open resumes.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum M1MigrateError {
    /// A record update or batch commit failed.
    #[error("m1 migrate: crud failure: {0}")]
    Crud(#[from] CrudError),
    /// A chained bag could not be read back (corrupt source chain) —
    /// fail LOUD; migrating past it would silently drop the bag.
    #[error("m1 migrate: chained bag read failed: {0}")]
    Blob(#[from] crate::blob::BlobError),
}

/// One candidate record discovered on the page walk.
#[derive(Debug, Clone, Copy)]
enum Candidate {
    Node(u64),
    Rel(u64),
}

// ─────────────────────────────────────────────────────────────────────
// v2 M2 — the JSON → typed-block migrate-on-open sweep (design §M2.6,
// `data_dir_version` 3 → 4)
// ─────────────────────────────────────────────────────────────────────

/// Records re-encoded per M2 migration transaction (same sizing
/// rationale as [`M1_MIGRATE_BATCH_SIZE`]).
pub const M2_MIGRATE_BATCH_SIZE: usize = 512;

/// TEST-ONLY crash injection for the M2 sweep — the §0.2
/// crash-during-migration gate's kill point, mirroring
/// [`ENV_M1_MIGRATE_CRASH_AFTER_BATCHES`]: abort AFTER the N-th batch
/// commit (batch durable, superseded chains unreclaimed, MANIFEST
/// still `typed-v1-migrating` — the hardest window).
pub const ENV_M2_MIGRATE_CRASH_AFTER_BATCHES: &str = "ARCGRAPH_M2_MIGRATE_CRASH_AFTER_BATCHES";

/// The injected JSON → typed re-encoder (the mcp bridge owns the JSON
/// contract + the Value mapping — ADR-089 §D-1: storage NEVER parses
/// JSON; the bootstrap injects
/// `arcgraph_mcp::storage::property_payload::reencode_json_bag_to_typed`).
/// `Ok(None)` = the bag re-encodes to EMPTY (a degenerate `{}` source).
pub type M2ReencodeFn<'a> =
    dyn Fn(TenantId, &[u8]) -> Result<Option<crate::prop_block::TypedBagParts>, String> + 'a;

/// M2 sweep options.
#[derive(Debug, Clone)]
pub struct M2MigrateOptions {
    /// Records per migration transaction.
    pub batch_size: usize,
    /// TEST-ONLY crash injection (see
    /// [`ENV_M2_MIGRATE_CRASH_AFTER_BATCHES`]).
    pub crash_after_batches: Option<u64>,
}

impl Default for M2MigrateOptions {
    fn default() -> Self {
        Self {
            batch_size: M2_MIGRATE_BATCH_SIZE,
            crash_after_batches: None,
        }
    }
}

impl M2MigrateOptions {
    /// Default options + the crash-injection env hook (test-only).
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            crash_after_batches: std::env::var(ENV_M2_MIGRATE_CRASH_AFTER_BATCHES)
                .ok()
                .and_then(|s| s.parse::<u64>().ok()),
            ..Self::default()
        }
    }
}

/// What the M2 sweep did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct M2MigrateReport {
    /// Node records re-encoded JSON → typed.
    pub nodes_migrated: u64,
    /// Rel records re-encoded JSON → typed.
    pub rels_migrated: u64,
    /// Superseded DEC-4 chain heads reclaimed (chained-JSON sources;
    /// slotted-JSON sources are superseded MVCC-style — their slots
    /// reclaim with their versions, no explicit removal).
    pub chains_removed: u64,
    /// Records whose payload was ALREADY a typed block — the
    /// idempotent-resume skip.
    pub already_typed: u64,
    /// Records whose payload is NEITHER a typed block NOR a JSON
    /// object — opaque crud-grain bytes an embedder wrote directly
    /// (`PropertyData::Blob` accepts arbitrary bytes; ArcGraph is
    /// embeddable-or-server). These are PRESERVED byte-identical and
    /// skipped with a loud warn: the M2 props contract covers the
    /// MCP-written JSON bag class; bricking an embedder's store at
    /// boot over payloads the sweep does not own would be an
    /// availability loss, not a correctness win. (Such payloads were
    /// never meaningfully readable through the mcp bag path — pre-M2
    /// they warn-degraded to an empty bag there; post-M2 that path
    /// rejects them loudly — while the crud-grain read that wrote
    /// them is untouched either way.)
    pub skipped_opaque: u64,
    /// Migration transactions committed.
    pub batches_committed: u64,
}

impl M2MigrateReport {
    /// True iff nothing was left to convert (the idempotent re-run
    /// shape).
    #[must_use]
    pub fn was_noop(&self) -> bool {
        self.nodes_migrated == 0 && self.rels_migrated == 0
    }
}

/// Faults surfaced by the M2 sweep. Any error leaves the store fully
/// readable at its current (mixed but valid) state; the MANIFEST is
/// not flipped, so the next open resumes.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum M2MigrateError {
    /// A record update or batch commit failed.
    #[error("m2 migrate: crud failure: {0}")]
    Crud(#[from] CrudError),
    /// A source bag could not be read back — fail LOUD (migrating past
    /// it would silently drop the bag; the M1 sweep's posture).
    #[error("m2 migrate: source bag read failed: {0}")]
    Blob(#[from] crate::blob::BlobError),
    /// The injected re-encoder rejected a source bag (malformed JSON =
    /// corrupt source) — fail LOUD.
    #[error("m2 migrate: JSON → typed re-encode failed: {0}")]
    Reencode(String),
}

/// Run the M2 (JSON → typed block) migrate-on-open sweep. Candidates
/// are EVERY live record version whose `property_ref` resolves to a
/// payload beginning `b'{'` — BOTH the M1 slotted-JSON class
/// (`slot_id >= 1`) and the DEC-4 chained-JSON class (`slot_id == 0`,
/// the M1 sweep's kept-chained-large leftovers and any pre-M1
/// stragglers). The §0.2 contract mapping is IDENTICAL to the M1
/// sweep's (module docs above): forward-only + idempotent per record
/// (a typed payload is skipped on re-run), never mutates the source
/// (new versions through the production `update_node`/`update_rel`
/// path), crash-mid-sweep resumes, and the caller's MANIFEST rewrite
/// (`typed-v1-migrating` → `typed-v1`) is the single commit point.
pub fn run_m2_migrate_on_open(
    mgr: &TxnManager,
    store: &CrudStore,
    reencode: &M2ReencodeFn<'_>,
    opts: &M2MigrateOptions,
) -> Result<M2MigrateReport, M2MigrateError> {
    let mut report = M2MigrateReport::default();
    let Some(records) = store.records() else {
        return Ok(report);
    };
    let batch_size = opts.batch_size.max(1);

    let mut batch: Vec<Candidate> = Vec::with_capacity(batch_size);
    let mut batch_tenant: Option<TenantId> = None;

    for (_page_id, latch) in records.iter_pages() {
        let candidates_in_page: Vec<(TenantId, Candidate)> = {
            let g = latch.read();
            let Ok(page) = SlottedPageRef::open(g.as_ref().as_ref()) else {
                continue;
            };
            let hdr = page.header();
            let tenant = TenantId::new(hdr.tenant_id);
            match PageType::from_byte(hdr.page_type) {
                // ANY blob-referencing record is a candidate; the
                // payload classification (typed vs JSON) happens under
                // the batch txn where the bag is read once anyway.
                Ok(PageType::Node) => page
                    .iter_nodes()
                    .filter(|(_, rec)| BlobRef::decode(rec.property_ref).is_some())
                    .map(|(_, rec)| (tenant, Candidate::Node(rec.id)))
                    .collect(),
                Ok(PageType::Rel) => page
                    .iter_rels()
                    .filter(|(_, rec)| BlobRef::decode(rec.property_ref).is_some())
                    .map(|(_, rec)| (tenant, Candidate::Rel(rec.id)))
                    .collect(),
                _ => continue,
            }
        };
        for (tenant, cand) in candidates_in_page {
            if batch_tenant != Some(tenant) && !batch.is_empty() {
                flush_m2_batch(
                    mgr,
                    store,
                    reencode,
                    batch_tenant.expect("non-empty batch has a tenant"),
                    &mut batch,
                    opts,
                    &mut report,
                )?;
            }
            batch_tenant = Some(tenant);
            batch.push(cand);
            if batch.len() >= batch_size {
                flush_m2_batch(mgr, store, reencode, tenant, &mut batch, opts, &mut report)?;
            }
        }
    }
    if let Some(tenant) = batch_tenant {
        if !batch.is_empty() {
            flush_m2_batch(mgr, store, reencode, tenant, &mut batch, opts, &mut report)?;
        }
    }

    tracing::info!(
        target: "arcgraph_storage::migrate",
        nodes = report.nodes_migrated,
        rels = report.rels_migrated,
        chains_removed = report.chains_removed,
        already_typed = report.already_typed,
        batches = report.batches_committed,
        "v2 M2 migrate-on-open sweep complete (JSON → typed-block re-encode)",
    );
    Ok(report)
}

/// Re-encode one batch of M2 candidates in a single transaction,
/// commit, then reclaim superseded CHAINED sources (commit-then-remove
/// — the M1 sweep's crash analysis applies verbatim).
#[allow(clippy::too_many_arguments)] // the M1 flush signature + the injected re-encoder
fn flush_m2_batch(
    mgr: &TxnManager,
    store: &CrudStore,
    reencode: &M2ReencodeFn<'_>,
    tenant: TenantId,
    batch: &mut Vec<Candidate>,
    opts: &M2MigrateOptions,
    report: &mut M2MigrateReport,
) -> Result<(), M2MigrateError> {
    let mut tx = mgr.begin(tenant);
    // Chained-JSON heads superseded by THIS batch — removed only after
    // its commit is durable.
    let mut superseded_chains: Vec<u64> = Vec::new();
    let mut nodes = 0u64;
    let mut rels = 0u64;

    for cand in batch.drain(..) {
        let (kind, id, bytes_opt) = match cand {
            Candidate::Node(id) => {
                let Some(bytes) = tx.read(node_mvcc_key(arcgraph_core::NodeId::new(id))) else {
                    continue; // deleted since the walk
                };
                ("node", id, Some(bytes))
            }
            Candidate::Rel(id) => {
                let Some(bytes) = tx.read(rel_mvcc_key(arcgraph_core::RelId::new(id))) else {
                    continue;
                };
                ("rel", id, Some(bytes))
            }
        };
        let bytes = bytes_opt.expect("populated above");
        let (property_ref, is_node) = match kind {
            "node" => (decode_node_bytes(bytes.as_ref())?.property_ref, true),
            _ => (decode_rel_bytes(bytes.as_ref())?.property_ref, false),
        };
        let Some(bref) = BlobRef::decode(property_ref) else {
            continue; // became inline since the walk
        };
        let bag = store.blob_store().get(tenant, bref)?;
        match bag.first() {
            Some(&crate::prop_block::PROP_BLOCK_DISCRIMINANT) => {
                report.already_typed += 1;
                continue; // idempotent resume: already typed
            }
            Some(&b'{') => { /* the JSON class — re-encode below */ }
            // Opaque crud-grain bytes (embedder-written) — preserved
            // byte-identical, skipped LOUDLY (see the report field's
            // rustdoc for the full disposition rationale).
            first => {
                tracing::warn!(
                    target: "arcgraph_storage::migrate",
                    kind,
                    id,
                    first_byte = ?first,
                    "M2 sweep: non-JSON, non-typed opaque payload preserved as-is \
                     (embedder-written crud-grain bytes; outside the MCP props contract)",
                );
                report.skipped_opaque += 1;
                continue;
            }
        }
        let parts = reencode(tenant, &bag).map_err(M2MigrateError::Reencode)?;
        let props = match parts {
            Some(p) => PropertyData::TypedBlock(p),
            // A degenerate `{}` source re-encodes to Empty.
            None => PropertyData::Empty,
        };
        if is_node {
            update_node(store, &mut tx, arcgraph_core::NodeId::new(id), &props)?;
            nodes += 1;
        } else {
            update_rel(store, &mut tx, arcgraph_core::RelId::new(id), &props)?;
            rels += 1;
        }
        if bref.slot_id == 0 {
            superseded_chains.push(bref.page_id);
        }
    }

    if nodes == 0 && rels == 0 {
        tx.abort();
        return Ok(());
    }

    crud::commit(tx, store)?;
    report.nodes_migrated += nodes;
    report.rels_migrated += rels;
    report.batches_committed += 1;

    // §0.2 crash-injection kill point (TEST-ONLY) — commit durable,
    // chains unreclaimed, MANIFEST not yet flipped.
    if let Some(n) = opts.crash_after_batches {
        if report.batches_committed >= n {
            tracing::warn!(
                target: "arcgraph_storage::migrate",
                batches = report.batches_committed,
                "M2 migrate crash injection firing (test hook) — aborting process",
            );
            std::process::abort();
        }
    }

    // Commit-then-remove (chained-JSON sources only; the inverse order
    // would be data loss on crash — see the M1 module docs).
    for head in superseded_chains {
        store.blob_store().remove_uncommitted_chain(tenant, head)?;
        report.chains_removed += 1;
    }
    Ok(())
}

/// Run the M1 migrate-on-open sweep. See the module docs for the
/// contract; the caller (bootstrap) owns the MANIFEST/VERSION stamping
/// around it.
pub fn run_m1_migrate_on_open(
    mgr: &TxnManager,
    store: &CrudStore,
    opts: &M1MigrateOptions,
) -> Result<M1MigrateReport, M1MigrateError> {
    let mut report = M1MigrateReport::default();
    let Some(records) = store.records() else {
        // No record page store (an in-memory/no-dual-write build) —
        // nothing durable to migrate.
        return Ok(report);
    };
    let batch_size = opts.batch_size.max(1);

    // Stream: page-by-page candidate discovery → per-tenant batches.
    // The page list is the same O(pages) handle Vec the
    // `bootstrap_primary_index` boot walk already takes; record bytes
    // are examined one page at a time under its read latch.
    let mut batch: Vec<Candidate> = Vec::with_capacity(batch_size);
    let mut batch_tenant: Option<TenantId> = None;

    for (_page_id, latch) in records.iter_pages() {
        let candidates_in_page: Vec<(TenantId, Candidate)> = {
            let g = latch.read();
            let Ok(page) = SlottedPageRef::open(g.as_ref().as_ref()) else {
                // Unreadable page: recovery would have failed on it
                // already; mirror `bootstrap_primary_index` and skip.
                continue;
            };
            let hdr = page.header();
            let tenant = TenantId::new(hdr.tenant_id);
            match PageType::from_byte(hdr.page_type) {
                Ok(PageType::Node) => page
                    .iter_nodes()
                    .filter(|(_, rec)| is_chained_overflow(rec.property_ref))
                    .map(|(_, rec)| (tenant, Candidate::Node(rec.id)))
                    .collect(),
                Ok(PageType::Rel) => page
                    .iter_rels()
                    .filter(|(_, rec)| is_chained_overflow(rec.property_ref))
                    .map(|(_, rec)| (tenant, Candidate::Rel(rec.id)))
                    .collect(),
                _ => continue,
            }
            // latch guard drops here — never held across a commit.
        };
        for (tenant, cand) in candidates_in_page {
            if batch_tenant != Some(tenant) && !batch.is_empty() {
                // Tenant boundary: flush the previous tenant's batch.
                flush_batch(
                    mgr,
                    store,
                    batch_tenant.expect("non-empty batch has a tenant"),
                    &mut batch,
                    opts,
                    &mut report,
                )?;
            }
            batch_tenant = Some(tenant);
            batch.push(cand);
            if batch.len() >= batch_size {
                flush_batch(mgr, store, tenant, &mut batch, opts, &mut report)?;
            }
        }
    }
    if let Some(tenant) = batch_tenant {
        if !batch.is_empty() {
            flush_batch(mgr, store, tenant, &mut batch, opts, &mut report)?;
        }
    }

    tracing::info!(
        target: "arcgraph_storage::migrate",
        nodes = report.nodes_migrated,
        rels = report.rels_migrated,
        chains_removed = report.chains_removed,
        already_slotted = report.already_slotted,
        kept_chained_large = report.kept_chained_large,
        batches = report.batches_committed,
        "v2 M1 migrate-on-open sweep complete (DEC-4 → slotted re-encode)",
    );
    Ok(report)
}

/// True iff `property_ref` is an overflow ref pointing at a DEC-4
/// chain (`slot_id == 0` — the pre-M1 encoding for every chained bag).
fn is_chained_overflow(property_ref: u64) -> bool {
    matches!(BlobRef::decode(property_ref), Some(r) if r.slot_id == 0)
}

/// Re-encode one batch of candidates in a single transaction, commit,
/// then reclaim the superseded chains (commit-then-remove). See the
/// module docs for the crash analysis of each step.
fn flush_batch(
    mgr: &TxnManager,
    store: &CrudStore,
    tenant: TenantId,
    batch: &mut Vec<Candidate>,
    opts: &M1MigrateOptions,
    report: &mut M1MigrateReport,
) -> Result<(), M1MigrateError> {
    let mut tx = mgr.begin(tenant);
    // Chains superseded by THIS batch — removed only after its commit
    // is durable.
    let mut superseded: Vec<u64> = Vec::new();
    let mut nodes = 0u64;
    let mut rels = 0u64;

    for cand in batch.drain(..) {
        match cand {
            Candidate::Node(id) => {
                // Re-check liveness + shape under THIS txn's snapshot
                // (the page walk is advisory; MVCC is authoritative).
                let Some(bytes) = tx.read(node_mvcc_key(arcgraph_core::NodeId::new(id))) else {
                    continue; // deleted since the walk — nothing to do
                };
                let rec = decode_node_bytes(bytes.as_ref())?;
                let Some(bref) = BlobRef::decode(rec.property_ref) else {
                    continue; // inline since the walk
                };
                if bref.slot_id != 0 {
                    report.already_slotted += 1;
                    continue; // idempotent resume: already packed
                }
                let bag = store.blob_store().get(tenant, bref)?;
                if bag.len() > PROP_BAG_MAX_BYTES {
                    report.kept_chained_large += 1;
                    continue; // §M1.2 overflow tail: stays chained
                }
                update_node(
                    store,
                    &mut tx,
                    arcgraph_core::NodeId::new(id),
                    &PropertyData::Blob(bag.to_vec()),
                )?;
                superseded.push(bref.page_id);
                nodes += 1;
            }
            Candidate::Rel(id) => {
                let Some(bytes) = tx.read(rel_mvcc_key(arcgraph_core::RelId::new(id))) else {
                    continue;
                };
                let rec = decode_rel_bytes(bytes.as_ref())?;
                let Some(bref) = BlobRef::decode(rec.property_ref) else {
                    continue;
                };
                if bref.slot_id != 0 {
                    report.already_slotted += 1;
                    continue;
                }
                let bag = store.blob_store().get(tenant, bref)?;
                if bag.len() > PROP_BAG_MAX_BYTES {
                    report.kept_chained_large += 1;
                    continue;
                }
                update_rel(
                    store,
                    &mut tx,
                    arcgraph_core::RelId::new(id),
                    &PropertyData::Blob(bag.to_vec()),
                )?;
                superseded.push(bref.page_id);
                rels += 1;
            }
        }
    }

    if superseded.is_empty() {
        // Nothing re-encoded (all skipped) — abort the empty txn.
        tx.abort();
        return Ok(());
    }

    // The batch is all-or-nothing: one CommitBundle, one fsync. At
    // boot there are no concurrent writers, so a commit failure is a
    // hard fault (disk/WAL) — surface it; the store stays at its
    // current valid mixed state and the next open resumes.
    crud::commit(tx, store)?;
    report.nodes_migrated += nodes;
    report.rels_migrated += rels;
    report.batches_committed += 1;

    // §0.2 crash-injection kill point (TEST-ONLY): commit durable,
    // chains not yet reclaimed, MANIFEST not yet flipped — the
    // hardest window. See ENV_M1_MIGRATE_CRASH_AFTER_BATCHES.
    if let Some(n) = opts.crash_after_batches {
        if report.batches_committed >= n {
            tracing::warn!(
                target: "arcgraph_storage::migrate",
                batches = report.batches_committed,
                "M1 migrate crash injection firing (test hook) — aborting process",
            );
            std::process::abort();
        }
    }

    // Commit-then-remove: reclaim the superseded chains ONLY now that
    // the replacing versions are durable. A crash before/among these
    // removals leaks ≤ one batch of unreferenced chain pages (bounded,
    // harmless — dropped from the durable image by the sweep-end
    // checkpoint the bootstrap fires).
    for head in superseded {
        store.blob_store().remove_uncommitted_chain(tenant, head)?;
        report.chains_removed += 1;
    }
    Ok(())
}
