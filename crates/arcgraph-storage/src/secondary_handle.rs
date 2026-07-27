//! Façade trait for secondary-index integration with the CRUD
//! dual-write drain (M2-34).
//!
//! The concrete B-tree implementation lives in `arcgraph-index`, which
//! depends on `arcgraph-storage`. To keep the dependency graph acyclic
//! (`-index → -storage`, never the other way — see
//! `docs/bounded-contexts.md`), `arcgraph-storage::crud` references
//! secondary indices only through this trait; `arcgraph-index` supplies
//! the concrete `SecondaryIndex` and implements
//! [`SecondaryIndexHandle`] on top of it.
//!
//! Per ADR-023 (the read-accelerator contract), the drain's invocations
//! here are best-effort — install failures log a warning and do not
//! fail the MVCC commit. The trait's `Result` type gives tests a way
//! to observe failures without forcing the drain to handle them.

use std::fmt;

use arcgraph_core::{LabelId, NodeId, PageId, StringId, TenantId};
use thiserror::Error;

use crate::mutation_log::{PageBuf, TxnMutationLog};
use crate::wal::bundle::StagedEmit;

/// Property-value variants supported by the secondary index in M2.d.
/// Mirror of `arcgraph_index::PropertyValue` — defined here so
/// `arcgraph-storage::crud` can construct values without depending on
/// `arcgraph-index`. The index crate provides an
/// `From<SecondaryIndexValue>` conversion into its internal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SecondaryIndexValue {
    /// 32-bit unsigned integer.
    U32(u32),
    /// 64-bit unsigned integer. Values exceeding `(1 << 56) - 1`
    /// cannot be encoded on-disk (DEC-19); the index returns a
    /// [`SecondaryIndexHandleError`] at insert time if so.
    U64(u64),
    /// Interned string-id value (categorical / enum-style property) —
    /// the M2.d storage-internal positional-field index.
    StringId(StringId),
    /// **RC-4 (#1366)** — a 56-bit hash of a UTF-8 string value's
    /// bytes, the KEYING for user-visible string property values.
    /// Keying by hash (rather than interned `StringId`) avoids
    /// intern-memory blowup + the read-path intern-mutation hazard;
    /// hash collisions are absorbed by the mandatory
    /// candidate-then-verify recheck (ADR-023). The MCP layer computes
    /// the hash via `arcgraph_index::hash_str_56` and feeds it here as
    /// an already-masked `< 2^56` value; the index maps this into
    /// `PropertyValue::StrHash`.
    StrHash(u64),
}

/// **RC-2 (secondary property index, #1366)** — declared-index
/// lifecycle state, the state-machine half of write-follows-declare.
///
/// # Write-follows-declare (the false-negative this closes)
///
/// A declared index is backfilled by scanning MVCC-visible nodes once,
/// during which its state is `Building`. If synchronous commit-path
/// maintenance were gated on `Online` only, a node created or updated
/// *after* the backfill snapshot but *before* the `Online` flip would
/// be silently absent from the index — a permanent false negative the
/// moment the planner starts choosing the index. So maintenance MUST
/// apply in `Building` as well as `Online`: backfill covers everything
/// visible at its snapshot, concurrent commit-path maintenance covers
/// everything after, and the overlap collapses via idempotent insert.
///
/// # Crash safety
///
/// The `Online` flip is NOT co-committed in one shared `CommitBundle`
/// with the backfill tail. Backfill emits N independent, each
/// durable-on-return per-insert bundles; the `Online` flip is a
/// discrete SYSTEM tx committed AFTER the whole backfill loop. The
/// "no durable `Online` without a complete tail" guarantee therefore
/// rests on THREE things, not a shared bundle: (1) append-all-then-flip
/// ordering (every insert precedes the flip), (2) synchronous-durable
/// inserts (each insert is durable before the flip is even begun), and
/// (3) recovery forcing any non-`Online` state back to `Building`. A
/// crash mid-backfill recovers as `Building` (never a partial `Online`);
/// the planner ignores a `Building` index and backfill restarts.
///
/// # Phase-0 note
///
/// The planner-selection and catalog/DDL that consume this state land
/// in Phase 1. Phase 0 lands the state machine + the maintenance rule
/// ([`Self::maintenance_active`]) so the rule is in force before the
/// index is ever query-enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexState {
    /// Backfill in progress. The planner does NOT choose the index, but
    /// synchronous commit-path maintenance DOES apply (write-follows-
    /// declare). A crash in this state recovers as `Building`.
    Building,
    /// Backfill complete and the `Online` flip (a discrete SYSTEM tx
    /// committed AFTER all backfill inserts are durable) has landed. The
    /// planner may now choose the index; maintenance continues to
    /// apply. (Durable-`Online`-implies-complete-tail rests on
    /// append-all-then-flip ordering + recover-forces-`Building`, not a
    /// shared `CommitBundle` — see the type-level `Crash safety` note.)
    Online,
}

impl IndexState {
    /// **RC-2 write-follows-declare rule.** Whether synchronous
    /// commit-path maintenance applies for an index in this state.
    ///
    /// TRUE for BOTH `Building` and `Online`. Gating this on `Online`
    /// only is the exact false-negative regression RC-2 exists to
    /// prevent — the crash-consistency / sensitivity test drives a node
    /// written while `Building` and asserts it is found once `Online`.
    #[must_use]
    #[inline]
    pub fn maintenance_active(self) -> bool {
        matches!(self, IndexState::Building | IndexState::Online)
    }

    /// Whether the planner may choose an index in this state. Only
    /// `Online` (Phase-1 planner consumer). A `Building` index is
    /// ignored by planning while its tail is still being backfilled.
    #[must_use]
    #[inline]
    pub fn planner_visible(self) -> bool {
        matches!(self, IndexState::Online)
    }
}

/// Error surface for trait-level secondary-index calls.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SecondaryIndexHandleError {
    /// Any failure in the concrete backend — the index crate provides
    /// a `.to_string()` of its local error type. The string is
    /// intentionally open-ended so the trait boundary never forces
    /// error-taxonomy coupling between storage and index.
    #[error("secondary index backend: {0}")]
    Backend(String),
}

/// Handle used by the CRUD dual-write drain to publish property
/// changes to a secondary index.
///
/// The implementation in `arcgraph-index::SecondaryIndex` maps the
/// raw `(tenant, label, property_key, value)` tuple into a
/// `SecondaryKey` and calls `insert` / `remove` accordingly. Every
/// method may fail with [`SecondaryIndexHandleError`]; the drain logs
/// and continues (per ADR-023). Tests that need to observe failures
/// can inspect the returned `Result` directly.
pub trait SecondaryIndexHandle: Send + Sync + fmt::Debug {
    /// Insert `(tenant, label, property_key, value) → node` into the
    /// secondary index.
    fn insert_property(
        &self,
        tenant: TenantId,
        label: LabelId,
        property_key: StringId,
        value: SecondaryIndexValue,
        node: NodeId,
    ) -> Result<(), SecondaryIndexHandleError>;

    /// Remove `node` from `(tenant, label, property_key, value)`'s
    /// entry. Returns `true` if the NodeId was found and zeroed,
    /// `false` otherwise.
    fn remove_property(
        &self,
        tenant: TenantId,
        label: LabelId,
        property_key: StringId,
        value: SecondaryIndexValue,
        node: NodeId,
    ) -> Result<bool, SecondaryIndexHandleError>;

    /// Bundle-aware insert (ADR-031). Performs the index mutation +
    /// in-memory install under the backend's write_gate + per-page
    /// latch and returns the staged `IndexPage` byte snapshots to
    /// the caller. The caller (typically `crud::commit`'s builder
    /// closure) folds these into a `CommitBundle` WAL record so each
    /// commit pays exactly one fire.
    ///
    /// **Z-1 F-1 (rollback-drain, #1366).** The `log` argument carries
    /// the enclosing transaction's [`TxnMutationLog`]. The backend
    /// records every secondary page it *installs fresh* (splits /
    /// overflow / grow-root) and captures the pre-mutation bytes of
    /// every page it edits *in place* under `PageStoreKind::Secondary`,
    /// so the Z-1 (b) rollback closure can drain them on WAL fsync
    /// failure — closing the F-1 gap where aborted-insert secondary
    /// pages leaked. Root-pointer changes from `grow_root` are recorded
    /// under [`IndexHandle::SECONDARY`](crate::mutation_log::IndexHandle::SECONDARY).
    // Seven params: the index key tuple (tenant/label/property_key/
    // value/node) plus the Z-1 F-1 rollback `log`. The tuple is the key
    // shape; splitting it would obscure the call site.
    #[allow(clippy::too_many_arguments)]
    fn insert_property_deferred(
        &self,
        tenant: TenantId,
        label: LabelId,
        property_key: StringId,
        value: SecondaryIndexValue,
        node: NodeId,
        log: &mut TxnMutationLog,
    ) -> Result<Vec<StagedEmit>, SecondaryIndexHandleError>;

    /// Bundle-aware remove (ADR-031). See
    /// [`Self::insert_property_deferred`]. Returns the staged emits
    /// regardless of whether a matching NodeId was found; the
    /// resulting bundle carries the mutated-leaf/overflow-page
    /// snapshots (potentially zero-length if nothing changed).
    ///
    /// **RC-1 note (#1366).** Under insert-only commit-path
    /// maintenance the property index's live drain never calls this
    /// method for old-value removals — those are enqueued as deferred
    /// removals ([`crate::crud::CrudStore`] snapshot-horizon queue) and
    /// applied only after `oldest_active_snapshot()` passes their LSN.
    /// This method remains on the trait for the deferred-application
    /// path and for standalone/test callers; it takes `log` for
    /// rollback symmetry with the insert path.
    #[allow(clippy::too_many_arguments)]
    fn remove_property_deferred(
        &self,
        tenant: TenantId,
        label: LabelId,
        property_key: StringId,
        value: SecondaryIndexValue,
        node: NodeId,
        log: &mut TxnMutationLog,
    ) -> Result<Vec<StagedEmit>, SecondaryIndexHandleError>;

    /// Drain the backend's stashed `grow_root` root-pointer update
    /// (if any) into a SYSTEM-tenant MVCC commit. MUST be called
    /// OUTSIDE any enclosing `Transaction::commit_with_bundle`
    /// builder. The `crud::commit` wrapper calls this on every
    /// configured secondary after the outer commit returns.
    fn persist_pending_root_update(&self) -> Result<(), SecondaryIndexHandleError>;

    // ─── Z-1 F-1 rollback dispatch (published across the storage↔index
    //     boundary per PD#7; the concrete `SecondaryPageStore` lives in
    //     `arcgraph-index`) ────────────────────────────────────────────

    /// Remove a newly-installed secondary page from the backend's page
    /// store (Z-1 (b) rollback, `TxnMutationLog::new_pages` drain).
    /// Mirrors `PrimaryPageStore::remove_page`. Idempotent: removing an
    /// absent page is a no-op.
    fn rollback_remove_page(&self, page_id: PageId);

    /// Restore a secondary page's pre-mutation bytes (Z-1 (b) rollback,
    /// `TxnMutationLog::page_mutations` drain). Mirrors
    /// `PrimaryPageStore::restore_page_bytes`. Best-effort: a missing
    /// page surfaces as a warning at the call site, never a panic.
    fn rollback_restore_page(
        &self,
        page_id: PageId,
        pre_bytes: &PageBuf,
    ) -> Result<(), SecondaryIndexHandleError>;

    /// Restore the backend's cached root pointer to `old_root_id`
    /// (Z-1 (b) rollback, `TxnMutationLog::root_changes` drain for
    /// [`IndexHandle::SECONDARY`](crate::mutation_log::IndexHandle::SECONDARY)).
    /// Mirrors `PrimaryIndex::restore_root_cache`. Also clears any
    /// pending (undurified) grow-root stash so the aborted new root is
    /// not persisted post-rollback.
    fn rollback_restore_root(&self, old_root_id: PageId);

    // ─── RC-2 write-follows-declare state machine (#1366) ────────────

    /// Current [`IndexState`] of this declared index.
    ///
    /// Default: [`IndexState::Online`]. A backend with no backfill
    /// lifecycle (the Phase-0 always-on secondary) is `Online` by
    /// construction — there is nothing to backfill, so maintenance is
    /// already complete. Backends that model a `CREATE INDEX` backfill
    /// (Phase 1) override this to return `Building` until the `Online`
    /// flip lands (a discrete SYSTEM tx committed after the whole
    /// backfill tail is durable — append-all-then-flip, not a shared
    /// bundle).
    fn index_state(&self) -> IndexState {
        IndexState::Online
    }

    /// Transition this index to `state`. Default is a no-op for
    /// backends with no lifecycle. Phase-1 backends override to flip
    /// `Building → Online` durably (a discrete commit AFTER the backfill
    /// tail is durable; append-all-then-flip ordering — NOT the same
    /// `CommitBundle` as the tail).
    fn set_index_state(&self, state: IndexState) {
        let _ = state;
    }

    /// **RC-2 write-follows-declare rule.** Whether the commit drain
    /// should apply synchronous maintenance for this index right now.
    ///
    /// The default delegates to [`IndexState::maintenance_active`],
    /// which is TRUE for BOTH `Building` and `Online`. Gating this on
    /// `Online` only reintroduces the false-negative RC-2 closes: a
    /// node written while `Building` would be absent once the index
    /// goes `Online`.
    fn maintenance_active(&self) -> bool {
        self.index_state().maintenance_active()
    }
}
