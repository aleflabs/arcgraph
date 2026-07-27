//! Substrate-access adapter for the M4-61 / M4-62 executor.
//!
//! [`ExecutorSubstrate`] is consumer-defined HERE in `arcgraph-query`
//! — parallel to the long-standing
//! [`crate::semantic::CatalogProvider`] pattern. The trait is the
//! seam between the planner-output layer (LogicalPlan + cost
//! annotations) and the storage / index layer (CRUD scans, HNSW
//! vector search, BM25 text search, community lookup).
//!
//! # Why not in `arcgraph-storage`?
//!
//! Same three reasons as `CatalogProvider`:
//!
//! 1. **Cyclic-dependency avoidance.** Storage already consumes
//!    query types in v1.1+ pathways (e.g., a `BoundAst` cached
//!    alongside MVCC versions); declaring the trait in storage would
//!    invert the `query → storage` edge of `docs/bounded-contexts.md`.
//! 2. **Test ergonomics.** [`StubExecutorSubstrate`] lives next to
//!    the trait — tests stub the substrate without pulling in
//!    `arcgraph-storage`'s buffer-pool / WAL machinery.
//! 3. **Bounded-context discipline.** `arcgraph-query` depends only
//!    on `arcgraph-core` for type primitives. Production wiring at
//!    M4-08+ (when `arcgraph-storage::router::TenantHandle` is
//!    bound to the executor) provides an `ExecutorSubstrate` impl
//!    on TenantHandle at composition time.
//!
//! # Why a trait at all?
//!
//! The 3-strike "no traits without ≥2 consumers" rule is
//! satisfied at this slice: the trait has FIVE in-slice consumers
//! ([`crate::executor::ops::scan::ScanOp`] /
//! [`crate::executor::ops::expand::ExpandOp`] /
//! [`crate::executor::ops::rank_by_hybrid::RankByHybridOp`]'s vector
//! sub-call / its BM25 sub-call / its community-lookup sub-call). The
//! trait pattern follows the established
//! [`crate::semantic::CatalogProvider`] precedent — at NO point in this
//! repo's history has the executor-substrate seam been a CONCRETE
//! struct.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.f** — primary M4-61 / M4-62 cite.
//! - **ADR-038 amendment-03 §TIER-2-c** — RANK BY HYBRID 3-substrate
//!   composition.
//! - **ADR-037 §D-1** — `TenantHandle` per-tenant substrate
//!   composition (the production binding [`ExecutorSubstrate`] will
//!   delegate to at M4-08+).
//! - **ADR-035** — vector substrate.
//! - **ADR-039** — BM25 substrate.
//! - **ADR-040** — community substrate.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::{LabelId, Lsn, NodeId, RelId, TenantId, TypeId};

use crate::executor::context::ExecutionContext;
use crate::executor::value::{NodeView, RelView, Value};
use crate::logical_plan::{CountStoreSource, Direction};

/// **ADR-197 — published seam for a Bolt explicit-transaction held
/// transaction.**
///
/// `arcgraph-query` carries the held transaction opaquely behind this
/// trait so the production held-transaction type
/// (`arcgraph_storage::transaction::OwnedTxn`) is NOT a production
/// dependency of `arcgraph-query` — preserving the
/// `query → storage` bounded-context edge direction (bounded-context policy:
/// "Do not reach across crate boundaries except through published
/// traits"; `arcgraph-storage` is a `[dev-dependencies]`-only edge of
/// `arcgraph-query` at v1.0-α).
///
/// The Bolt handler (in `arcgraph-mcp`, which depends on BOTH crates)
/// installs the concrete `OwnedTxn` (boxed as `dyn HeldTxnHandle`)
/// onto the [`ExecutionContext`] before EXECUTE; the production
/// substrate impl (`arcgraph_mcp::storage::bolt`'s
/// `CrudExecutorSubstrate`) downcasts it back via [`Self::as_any_mut`]
/// to stage CRUD writes into the held transaction.
///
/// The trait is deliberately minimal — it carries no transaction
/// semantics itself (the substrate owns the staging logic); it is a
/// type-erasure carrier + a downcast seam.
pub trait HeldTxnHandle: std::any::Any + Send + std::fmt::Debug {
    /// Downcast seam — the production substrate recovers the concrete
    /// `OwnedTxn` to stage CRUD ops into the held transaction.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// ADR-197-amendment-01 D-5 — the transaction's pinned snapshot
    /// LSN, fixed at BEGIN and constant for the transaction's
    /// lifetime (valid even after the handle is finalized — impls
    /// capture it at construction). [`ExecutionContext::with_held_txn`]
    /// seeds the context's snapshot from this so `ensure_snapshot_lsn`
    /// REPORTS the LSN explicit-mode reads actually observe; the
    /// visibility itself is enforced by reading through the held
    /// transaction (amendment D-1), not by this accessor.
    fn snapshot_lsn(&self) -> Lsn;
}

/// **NN-4 (#1384) — published seam for the MERGE get-or-create critical
/// section.**
///
/// A `MergeGuard` is an opaque, RAII serialization handle: while it is
/// held, no other MERGE on the SAME merge key (tenant + label + property
/// set) may execute its match→create→commit span. The loser BLOCKS on
/// [`ExecutorSubstrate::merge_guard`] until the winner's create has
/// committed + the winner's guard has dropped; it then pins its snapshot +
/// re-probes the match branch (which now sees the winner's committed node)
/// and takes the match branch — clean get-or-create uniqueness.
///
/// # Where the guard is acquired (NN-4 re-spin, Fix 1)
///
/// The guard is acquired by the QUERY DRIVER
/// ([`crate::materialize::materialize`] /
/// [`crate::executor::execute_with_context`], via
/// `crate::executor::ops::acquire_merge_guards`) BEFORE the statement's
/// read snapshot is pinned — i.e. before `begin_statement` on the D-2
/// auto-commit path — and STASHED on the
/// [`crate::executor::ExecutionContext`] until AFTER `commit_statement` /
/// `rollback_statement`. Acquiring before the snapshot pin is load-bearing:
/// under D-2 `begin_statement` installs a `BoltHeldTxn` whose pinned
/// snapshot the match probe reads at, so a guard taken only INSIDE
/// `MergeOp::next_batch` (after the pin) would let the loser observe a
/// stale pre-commit snapshot and still double-create.
///
/// # Why this seam (mirrors [`HeldTxnHandle`])
///
/// The serialization primitive lives in the PRODUCTION substrate
/// (`arcgraph_mcp::storage::substrate::CrudExecutorSubstrate`), which is
/// `Arc`-shared across concurrent executor sessions — so all racers
/// observe the SAME per-`(tenant, key)` lock table. `arcgraph-query`
/// carries the guard opaquely behind this trait so the concrete lock
/// type is NOT a production dependency of `arcgraph-query`, preserving
/// the `query → storage` bounded-context edge (bounded-context policy) — exactly
/// the [`HeldTxnHandle`] pattern. The trait is deliberately empty: it
/// carries NO methods; releasing the lock is `Drop`, so the guard only
/// needs to be held (bound to a local / stashed on the context) for the
/// critical-section span and dropped at scope exit.
///
/// # Lock order (no deadlock — NN-4 §Risks; re-verified after Fix 1)
///
/// The merge-key lock is acquired by the query driver BEFORE any snapshot
/// pin / scan / create, and is the STRICTLY-OUTER lock: no commit path
/// (`crud::commit` → the MVCC `commit_gate` / `write_gate`) ever acquires a
/// merge-key lock, so the acquisition order is always
/// `merge_key_lock → commit_gate` and never the reverse — even though the
/// guard is now held ACROSS `commit_statement` (Fix 1), the commit takes
/// `commit_gate` while the outer merge-key lock is held; the reverse edge
/// does not exist, including the `fire_actions` SET paths (`set_node` /
/// `set_rel`, whose commit likewise never names the merge table). A
/// concurrent commit therefore cannot invert against a held merge-key
/// lock. Two merges on DIFFERENT keys never share a lock (distinct map
/// entries), so they run fully concurrently; a single statement acquiring
/// TWO keys (path-shape MERGE) takes them in canonical TOTAL ORDER (sorted)
/// so two path-MERGEs naming the same keys in opposite order cannot invert
/// against each other either.
///
/// # Not `Send` (thread-affinity)
///
/// The guard is deliberately NOT required to be `Send`: the production
/// implementation wraps a `parking_lot::ArcMutexGuard`, whose default
/// `RawMutex` requires the SAME thread that locked to unlock (guard
/// thread-affinity). The NN-4 re-spin (Fix 1) STASHES the guard on the
/// [`crate::executor::ExecutionContext`] so it SPANS the statement commit
/// rather than dropping at `next_batch` return — but the WHOLE statement
/// (match probe → stage create → `commit_statement` → drain-and-drop the
/// guard) still runs SYNCHRONOUSLY on ONE thread: the guard is created,
/// stashed, and dropped on the same acquiring thread; it never crosses a
/// thread boundary and never straddles an `.await`. A `Send` bound would
/// force the production `send_guard` parking_lot feature for no benefit,
/// and `ExecutionContext` is likewise never moved across threads (it is
/// held by shared `&` reference within a single synchronous drive).
///
/// # `Debug` supertrait
///
/// [`Debug`](std::fmt::Debug) is a supertrait so the guard can be a field
/// of the `#[derive(Debug)]` [`crate::executor::ExecutionContext`] (same
/// pattern as [`HeldTxnHandle`]). The production `CrudMergeGuard` provides
/// a manual `Debug` impl (its inner `ArcMutexGuard` is not `Debug`).
///
/// [`MergeOp::next_batch`]: crate::executor::ops::MergeOp::next_batch
pub trait MergeGuard: std::fmt::Debug {}

/// Substrate-access adapter consumed by the executor.
///
/// Implementations live outside the executor module:
/// - **Tests** — [`StubExecutorSubstrate`] is a fluent in-memory
///   fixture mirroring the [`crate::semantic::StubCatalogProvider`]
///   pattern.
/// - **Production** — `arcgraph_storage::router::TenantHandle`
///   gains an [`ExecutorSubstrate`] impl at the M4-08 wiring layer.
///
/// All methods take `&self` so the substrate can be shared across
/// concurrent operators (each operator clones cheaply or borrows).
/// `Send + Sync` lets the executor pass the substrate across
/// thread-pool boundaries when M4-64a parallel-execution lights.
///
/// ## Snapshot contract
///
/// A finite `read_lsn` is an exact request, not a lower bound and not
/// permission to ratchet forward. An implementation that cannot serve
/// that exact snapshot must return
/// [`SubstrateAccessError::SnapshotUnavailable`] with the requested and
/// available LSNs. [`Lsn::MAX`] is the explicit read-latest sentinel;
/// production implementations resolve it to the snapshot actually used.
pub trait ExecutorSubstrate: Send + Sync {
    /// O(1) tenant-wide count from the counts store. This is intentionally
    /// total-only: filtered counts must use the normal scan path.
    fn count_store(
        &self,
        tenant: TenantId,
        source: CountStoreSource,
    ) -> Result<u64, SubstrateAccessError> {
        let _ = tenant;
        let _ = source;
        Err(SubstrateAccessError::IndexUnavailable(
            "counts-store".into(),
        ))
    }

    /// Sequential scan over the tenant's nodes, optionally filtered
    /// by `label`. Returns ALL matching nodes (the executor decides
    /// when to stop pulling). Order is implementation-defined; tests
    /// SHOULD use deterministic ordering for reproducibility.
    ///
    /// `read_lsn` is the MVCC visibility key (per ADR-041 §D-4).
    /// In-memory fixtures have one timeless state, so every LSN names
    /// that same state; production implementations follow the trait-level
    /// exact-or-error snapshot contract.
    fn scan_nodes(
        &self,
        tenant: TenantId,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError>;

    /// Transaction-aware node scan. Substrates that support a held
    /// explicit transaction override this to read through `ctx`
    /// (visibility = `snapshot(tx) ⊎ write_set(tx)` per
    /// ADR-197-amendment-01 D-1).
    ///
    /// **Default fails LOUD inside an explicit transaction**
    /// (amendment D-4): a substrate that has not implemented held-txn
    /// reads must surface [`SubstrateAccessError::HeldTxnReadsUnsupported`]
    /// rather than silently serving committed-read isolation — silent
    /// degradation is the #822 bug shape. Without a held transaction
    /// the default delegates to the plain committed read, unchanged.
    fn scan_nodes_with_context(
        &self,
        ctx: &ExecutionContext,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        if ctx.has_held_txn() {
            return Err(SubstrateAccessError::HeldTxnReadsUnsupported(
                "scan_nodes".into(),
            ));
        }
        self.scan_nodes(ctx.tenant(), label, read_lsn)
    }

    /// **v2 M2 (design §M2.3) — projection-pushdown scan.** Like
    /// [`Self::scan_nodes_with_context`], but the returned
    /// [`BoundNode`]s' property bags MAY be restricted to
    /// `projected_properties` (the plan-time-derived complete
    /// consumption set for the scan variable — the pipeline pushes it
    /// ONLY when the whole plan provably consumes nothing else, so a
    /// restricted bag is observation-equivalent to the full one).
    ///
    /// The DEFAULT ignores the projection and delegates to the full
    /// scan — over-fetching is always correct (the safe polarity);
    /// substrates with a typed zero-decode read (the production
    /// `CrudExecutorSubstrate`) override this to materialize only the
    /// projected key_ids (`PropBlockView` touches nothing else — the
    /// M2 "cost of a read is O(|projection|), not O(|bag|)" contract).
    fn scan_nodes_projected_with_context(
        &self,
        ctx: &ExecutionContext,
        label: Option<LabelId>,
        read_lsn: Lsn,
        projected_properties: &[String],
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let _ = projected_properties;
        self.scan_nodes_with_context(ctx, label, read_lsn)
    }

    /// One-hop neighbor enumeration.
    ///
    /// For node `from`, return all neighbors reachable via a relationship
    /// of `rel_type` (or any rel-type if `None`) traversed in `direction`.
    /// Each entry carries the matched relationship + the destination
    /// node so the operator can populate both the relationship-binding
    /// (when `LogicalExpand::rel_var` is `Some`) and the destination
    /// node binding.
    fn expand(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError>;

    /// Transaction-aware one-hop expansion. Substrates that support a
    /// held explicit transaction override this to read through `ctx`
    /// (visibility = `snapshot(tx) ⊎ write_set(tx)` per
    /// ADR-197-amendment-01 D-1).
    ///
    /// **Default fails LOUD inside an explicit transaction**
    /// (amendment D-4) — see [`Self::scan_nodes_with_context`].
    fn expand_with_context(
        &self,
        ctx: &ExecutionContext,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        if ctx.has_held_txn() {
            return Err(SubstrateAccessError::HeldTxnReadsUnsupported(
                "expand".into(),
            ));
        }
        self.expand(ctx.tenant(), from, rel_type, direction, read_lsn)
    }

    /// Point-read a single node (id + label + properties) for path
    /// materialization (#965 / ADR-211). Default `Ok(None)` means "not
    /// supported"; callers MUST degrade gracefully to an id-only view.
    fn node_by_id_with_context(
        &self,
        ctx: &ExecutionContext,
        id: NodeId,
    ) -> Result<Option<BoundNode>, SubstrateAccessError> {
        let _ = (ctx, id);
        Ok(None)
    }

    /// **#1366 (Phase 2) — the indexed point-lookup seam.** Resolve the
    /// declared secondary index on `(label, property)`, do the B+tree
    /// candidate lookup for `value`, then **MVCC-verify each candidate**:
    /// hydrate it through the read snapshot (the `node_by_id_with_context`
    /// path) and re-check that it (a) still carries `label` and (b) still
    /// has `property == value`. Return ONLY the verified, label-and-
    /// property-matching nodes, **deduplicated by NodeId** (a candidate
    /// slot that appears twice from the Phase-1 insert-only + backfill
    /// overlap yields ONE row). The index NEVER determines visibility —
    /// a stale / ghost / snapshot-invisible candidate is DROPPED (ADR-023
    /// candidate-then-verify), never surfaced and never an error.
    ///
    /// The caller ([`crate::executor::ops::PropertyIndexScanOp`]) has
    /// ALREADY confirmed (at plan time) the index is `Online` (RC-6);
    /// production impls MUST re-gate on `planner_visible()` at the lookup
    /// entry so a Building index never serves query rows even on a direct
    /// call. `read_lsn` mirrors [`Self::scan_nodes`]'s MVCC key.
    ///
    /// # Bounded contexts (PD#7)
    ///
    /// MCP owns the typed lookup + verify; the storage / index crates
    /// stay JSON-opaque behind the published `PropertyIndexManager`
    /// candidate API. The query crate hands over a typed
    /// [`crate::executor::value::Value`] and receives typed
    /// [`BoundNode`]s — no cross-crate reach-through into the B+tree.
    ///
    /// # Default impl
    ///
    /// `Ok(Vec::new())` — read-only fixtures have no property-index
    /// backend, so the planner never routes to this op on them. The
    /// production `CrudExecutorSubstrate` and the test
    /// `StubExecutorSubstrate` override it.
    fn property_index_lookup_with_context(
        &self,
        ctx: &ExecutionContext,
        label: LabelId,
        property: &str,
        value: &Value,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let _ = (ctx, label, property, value, read_lsn);
        Ok(Vec::new())
    }

    /// **#1366 (Phase 2) — the op's index-vs-scan-fallback gate.** Whether
    /// a resolved runtime lookup [`Value`] has a CANONICAL INDEX KEY, i.e.
    /// whether [`Self::property_index_lookup_with_context`] can answer for
    /// it via the B+tree (`canonical_key_for(value).is_some()` on the MCP
    /// seam).
    ///
    /// This is the fix for the REJECT-class silent-wrong-results bug
    /// (#1415): a `$param` is admitted to `PropertyIndexScan` UNCONDITIONALLY
    /// at plan time (its runtime type is unknown until it binds), but a
    /// value with NO canonical key — a fractional / out-of-i64-range
    /// `Float`, a NEGATIVE `Integer`, a `List` / `Map` — makes
    /// `lookup_candidates` return an EMPTY vec. Treating that empty as "no
    /// matches" would silently drop rows a full scan's `Filter(prop = v)`
    /// would keep (`values_equal_3vl` still matches `10.5`, `[1,2]`, `-5`,
    /// …). So the op MUST consult this predicate BEFORE the index lookup:
    ///
    /// - `true`  ⇒ the value is keyable (String / in-range non-negative
    ///   Integer / Boolean / INTEGRAL in-range Float that coerces to the
    ///   int bucket) — use the fast index path.
    /// - `false` ⇒ NO canonical key — the op falls back to a Scan+Filter
    ///   over the label with the equality as the filter, returning the
    ///   SAME rows the un-routed full-scan path would (never the empty
    ///   index result).
    ///
    /// # Bounded contexts (PD#7)
    ///
    /// MCP owns the typed key logic (`canonical_key_for`); the query crate
    /// only asks "is this keyable?" through this seam and never reaches
    /// into the storage/index B+tree. The default is `false` (read-only
    /// fixtures have no property-index backend, so the op scans — which is
    /// correct, never a wrong result). The production
    /// `CrudExecutorSubstrate` and the test `StubExecutorSubstrate`
    /// override it to mirror `canonical_key_for`.
    fn value_is_indexable(&self, value: &Value) -> bool {
        let _ = value;
        false
    }

    /// Owned streaming cursor over one-hop expansion.
    ///
    /// `Send + 'static` lets an executor operator store the cursor in
    /// its own state across `next_batch` calls. The default delegates
    /// to [`Self::expand`], so existing substrates stay correct while
    /// production substrates can override this with a real stream.
    ///
    /// Order contract: `LeftToRight` and `RightToLeft` produce the same
    /// sequence as [`Self::expand`]. `Undirected` is multiset-equal to
    /// [`Self::expand`] and yields self-loops once, but the substrate
    /// contract does not pin a specific undirected order.
    fn expand_cursor(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<BoundEdgeCursor, SubstrateAccessError> {
        Ok(Box::new(
            self.expand(tenant, from, rel_type, direction, read_lsn)?
                .into_iter()
                .map(Ok),
        ))
    }

    /// Transaction-aware streaming expansion.
    ///
    /// Held-transaction streaming is forward-pinned. Delegating to the
    /// materialized held-transaction read keeps visibility correct
    /// (ADR-197-amendment-01 D-1): it degrades streaming only, never
    /// correctness.
    fn expand_cursor_with_context(
        &self,
        ctx: &ExecutionContext,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<BoundEdgeCursor, SubstrateAccessError> {
        if ctx.has_held_txn() {
            return Ok(Box::new(
                self.expand_with_context(ctx, from, rel_type, direction, read_lsn)?
                    .into_iter()
                    .map(Ok),
            ));
        }
        self.expand_cursor(ctx.tenant(), from, rel_type, direction, read_lsn)
    }

    /// Vector ANN top-K retrieval. `query_vec` is the query vector
    /// (typically resolved from a `$qv` parameter at expression-eval
    /// time); `k` is the top-K cap. Returns ranked hits in score-
    /// descending order. v1.0-alpha stub substrates ignore the score's
    /// distance-vs-similarity convention; production wiring follows
    /// ADR-035 D-2 (similarity-ascending if cosine, distance-ascending
    /// if L2).
    ///
    /// **Explicit-transaction visibility (ADR-197-amendment-01 D-3):**
    /// index-backed reads serve **index-state-at-read** (current
    /// committed index state), NOT the held transaction's pinned
    /// snapshot, inside a Bolt explicit transaction at v1.0-α — staged
    /// writes are reachable through the D-1 graph reads
    /// (`scan_nodes_with_context` / `expand_with_context`) but absent
    /// from index-backed results until COMMIT, and post-BEGIN external
    /// commits MAY be visible through the index. This divergence is a
    /// documented contract (mirroring the served-HNSW build-time
    /// posture and Neo4j's own in-tx index-visibility caveats); the
    /// staged-overlay + in-tx hydration surface is the amendment's S2.
    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError>;

    /// BM25 text-match top-K retrieval. `query_text` is the query
    /// string; `k` is the top-K cap. Returns ranked hits in BM25-score-
    /// descending order.
    ///
    /// **Explicit-transaction visibility (ADR-197-amendment-01 D-3):**
    /// index-backed reads serve **index-state-at-read** (current
    /// committed index state), NOT the held transaction's pinned
    /// snapshot, inside a Bolt explicit transaction at v1.0-α — staged
    /// writes are reachable through the D-1 graph reads
    /// (`scan_nodes_with_context` / `expand_with_context`) but absent
    /// from index-backed results until COMMIT, and post-BEGIN external
    /// commits MAY be visible through the index. This divergence is a
    /// documented contract (mirroring the served-HNSW build-time
    /// posture and Neo4j's own in-tx index-visibility caveats); the
    /// staged-overlay + in-tx hydration surface is the amendment's S2.
    fn bm25_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_text: &str,
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError>;

    /// Community-membership lookup. Returns the nodes belonging to
    /// `community_id` (per ADR-040 D-3 keying).
    ///
    /// # FORWARD-PIN: M4-62b LogicalCommunityLookup
    ///
    /// W11Z fix-up LOW-2 acknowledgement (PR #268 retro): no in-slice
    /// operator consumes this method at v1.0-alpha. The
    /// [`crate::executor::ops::rank_by_hybrid::RankByHybridOp`]
    /// composes only VECTOR + TEXT substrates;
    /// `LogicalCommunityLookup` is the forward-deferred consumer
    /// (M4-62b). The method shape may change at first consumer — e.g.,
    /// `community_id: CommunityId` newtype instead of `i64`, or a
    /// streaming cursor instead of `Vec<BoundNode>` — per
    /// `feedback_avoid_speculative_scaffolding.md`. The trait surface
    /// itself is defensible (parallel to `CatalogProvider`); the
    /// unconsumed METHOD is over-shipped and the forward-pin lets the
    /// next consuming slice re-shape without a v1.0-alpha breakage.
    ///
    /// **Explicit-transaction visibility (ADR-197-amendment-01 D-3):**
    /// index-backed reads serve **index-state-at-read** (current
    /// committed index state), NOT the held transaction's pinned
    /// snapshot, inside a Bolt explicit transaction at v1.0-α — staged
    /// writes are reachable through the D-1 graph reads
    /// (`scan_nodes_with_context` / `expand_with_context`) but absent
    /// from index-backed results until COMMIT, and post-BEGIN external
    /// commits MAY be visible through the index. This divergence is a
    /// documented contract (mirroring the served-HNSW build-time
    /// posture and Neo4j's own in-tx index-visibility caveats); the
    /// staged-overlay + in-tx hydration surface is the amendment's S2.
    fn community_members(
        &self,
        tenant: TenantId,
        community_id: i64,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError>;

    /// Legacy tenant-free substrate-availability mirror of
    /// [`crate::semantic::CatalogProvider::has_vector_index`].
    ///
    /// Do not use this to gate tenant-scoped execution: availability
    /// can differ by tenant. Call [`Self::vector_search`] with the
    /// query tenant and propagate its structured error instead.
    fn has_vector_substrate(&self) -> bool {
        false
    }

    /// Legacy tenant-free substrate-availability mirror of
    /// [`crate::semantic::CatalogProvider::has_bm25_index`].
    ///
    /// Do not use this to gate tenant-scoped execution; call
    /// [`Self::bm25_search`] with the query tenant instead.
    fn has_bm25_substrate(&self) -> bool {
        false
    }

    /// Substrate-availability mirror of
    /// [`crate::semantic::CatalogProvider::has_community_index`].
    fn has_community_substrate(&self) -> bool {
        false
    }

    /// **D-2 (ADR-147 §D-8 / W26-θ Phase 5) — begin a statement-scoped
    /// autocommit transaction.**
    ///
    /// Called by `crate::materialize` at the START of an AUTO-COMMIT
    /// write statement (see [`crate::logical_plan::LogicalPlan::writes`]),
    /// BEFORE the pipeline drives its first batch. A supporting substrate
    /// opens ONE owned transaction and installs it as a held txn on `ctx`
    /// (via [`ExecutionContext::install_held_txn`]) so every write op of
    /// the statement STAGES into it (the mature ADR-197 held-txn / EXPLICIT
    /// path) instead of begin→op→commit per op. The commit happens ONCE at
    /// [`Self::commit_statement`]; a mid-statement error routes to
    /// [`Self::rollback_statement`], discarding the WHOLE spine.
    ///
    /// # Idempotence / no-op contract
    ///
    /// The default impl is a no-op (`Ok(())`): read-only substrates + the
    /// v1.0-α stub keep the byte-for-byte one-call-one-tx behavior. The
    /// materialize caller ONLY invokes this when the plan writes AND no
    /// held txn is already installed (Bolt BEGIN…COMMIT explicit mode owns
    /// its own lifetime — D-2 must NOT nest a statement txn inside it).
    ///
    /// Returns `Err(_)` if the transaction cannot be opened; the caller
    /// then aborts the statement without driving any op.
    fn begin_statement(&self, _ctx: &ExecutionContext) -> Result<(), SubstrateAccessError> {
        Ok(())
    }

    /// **D-2 — commit the statement-scoped autocommit transaction ONCE.**
    ///
    /// Called by `crate::materialize` after a write statement's pipeline
    /// drains WITHOUT error. A supporting substrate reclaims the held txn
    /// installed by [`Self::begin_statement`] and drains it through the
    /// FULL commit machinery — one WAL CommitBundle / one fsync, the
    /// HNSW/BM25 maintenance hooks fired ONCE (issue #963; queued while the
    /// spine staged), and CDC observing ONE commit (not one per op). This
    /// is the single-durable-commit / atomicity guarantee.
    ///
    /// The default impl is a no-op (`Ok(())`) for substrates that did not
    /// begin a statement txn (read-only / stub). On commit failure the
    /// error surfaces so materialize propagates it; the substrate discards
    /// pending side effects on that path (no partial spine leaks).
    fn commit_statement(&self, _ctx: &ExecutionContext) -> Result<(), SubstrateAccessError> {
        Ok(())
    }

    /// **D-2 — roll back the statement-scoped autocommit transaction.**
    ///
    /// Called by `crate::materialize` when a write statement's pipeline
    /// (or its commit) FAILS. A supporting substrate reclaims the held txn
    /// and aborts it — discarding EVERY staged write of the spine so a
    /// 2-node-1-rel statement that fails on the last op leaves NEITHER node
    /// committed (mirrors the ADR-197 Bolt ROLLBACK / RESET abort path).
    /// The queued HNSW / BM25 hooks are dropped with the handle (no
    /// index maintenance fires for an aborted statement).
    ///
    /// Infallible (best-effort) — the default is a no-op; a supporting
    /// substrate's abort cannot fail (the `OwnedTxn::abort` / `Drop` path
    /// always discards).
    fn rollback_statement(&self, _ctx: &ExecutionContext) {}

    /// **ADR-147 W26-θ Phase 1.** Create a fresh node carrying
    /// `label` (optional; the substrate intern-tables the name to
    /// a `LabelId` if needed per ADR-147 §D-7) and `properties` (a
    /// flat key/Value slice — Phase 1 stores via
    /// `PropertyData::Empty` at this adapter layer, which means the
    /// slice's contents are
    /// IGNORED at storage but the parameter shape is preserved for
    /// the v1.2 wire-through).
    ///
    /// Returns the assigned `NodeId` on success. Production
    /// implementations open a per-tenant `Transaction` per ADR-031
    /// (CommitBundle) + ADR-033 (rollback) and commit at this call
    /// boundary (one-call-one-tx at Phase 1
    /// per ADR-147 §D-8); the implementation either fully commits or
    /// returns `Err(_)` with no partial side-effect.
    ///
    /// # Default impl
    ///
    /// Returns `IndexUnavailable("write-op create_node unavailable
    /// on this substrate")` so existing substrate impls (read-only
    /// test fixtures) compile unchanged. Substrates that support
    /// CREATE override this method.
    fn create_node(
        &self,
        _tenant: TenantId,
        _label: Option<&str>,
        _properties: &[(String, Value)],
        _ctx: &ExecutionContext,
    ) -> Result<NodeId, SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable(
            "write-op create_node unavailable on this substrate".into(),
        ))
    }

    /// **ADR-148 W26-θ Phase 2.** Create a fresh relationship between
    /// `source` + `target` carrying `label` (mandatory; the substrate
    /// intern-tables the name to a `TypeId` if needed per ADR-148 §D-7)
    /// and `properties` (a flat key/Value slice — Phase 2 stores via
    /// `PropertyData::Empty` at this adapter layer).
    ///
    /// Returns the assigned `RelId` on success. Production
    /// implementations open a per-tenant `Transaction` per ADR-031
    /// (CommitBundle) + ADR-033 (rollback) and commit at this call
    /// boundary (one-call-one-tx at Phase 2 per ADR-148 §D-8); the
    /// implementation either fully commits or returns `Err(_)` with
    /// no partial side-effect.
    ///
    /// # Default impl
    ///
    /// Returns `IndexUnavailable("write-op create_rel unavailable
    /// on this substrate")` so existing substrate impls (read-only
    /// test fixtures) compile unchanged. Substrates that support
    /// CREATE-rel override this method.
    // ADR-197 added the `_ctx` param (explicit-tx threading) — the arg
    // set (tenant + source + target + label + properties + ctx) is
    // intrinsic to a relationship create; collapsing into a struct would
    // obscure the call sites for no benefit.
    #[allow(clippy::too_many_arguments)]
    fn create_rel(
        &self,
        _tenant: TenantId,
        _source: NodeId,
        _target: NodeId,
        _label: &str,
        _properties: &[(String, Value)],
        _ctx: &ExecutionContext,
    ) -> Result<RelId, SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable(
            "write-op create_rel unavailable on this substrate".into(),
        ))
    }

    /// **NN-4 (#1384).** Acquire the get-or-create serialization guard
    /// for a MERGE on `key` under `tenant`. `key` is the execute-time
    /// canonical rendering of the merge pattern's unique identity
    /// (label + property set — see
    /// `crate::executor::ops::acquire_merge_guards`); it is opaque to the
    /// substrate (an injection-safe string).
    ///
    /// The returned guard, while held, EXCLUDES any other MERGE on the
    /// SAME `(tenant, key)` from running its match→create→commit span. The
    /// loser BLOCKS here until the winner's create commits and the winner's
    /// guard drops, then it pins its snapshot + re-probes the match branch
    /// (which now sees the winner's committed node — SI made it invisible
    /// before the commit) and takes the match branch. This closes the
    /// concurrent-double-create hole: the query driver otherwise holds no
    /// lock across match→create→commit, so under snapshot isolation both
    /// racers see 0 match rows and both create (the OCC commit-check only
    /// iterates write keys = disjoint new node ids = no overlap = both
    /// commit). The driver acquires this guard BEFORE pinning the read
    /// snapshot (before `begin_statement`) and drops it AFTER
    /// `commit_statement` — see `crate::executor::ops::acquire_merge_guards`.
    ///
    /// # Default impl
    ///
    /// Returns `Ok(None)` — the substrate provides NO serialization. This
    /// is correct for the single-threaded / read-only test fixtures
    /// ([`StubExecutorSubstrate`]): with no concurrent racer there is no
    /// race to serialize, and the driver treats `None` as "run without the
    /// critical section" (byte-identical to the pre-NN-4 behavior).
    /// Production substrates that serve concurrent sessions override this
    /// to return a real [`MergeGuard`].
    ///
    /// [`MergeOp`]: crate::executor::ops::MergeOp
    /// [`MergeOp::next_batch`]: crate::executor::ops::MergeOp::next_batch
    fn merge_guard(
        &self,
        _tenant: TenantId,
        _key: &str,
    ) -> Result<Option<Box<dyn MergeGuard>>, SubstrateAccessError> {
        Ok(None)
    }

    /// **ADR-149 W26-θ Phase 3.** Tombstone the node identified by
    /// `node`. When `detach = true`, the substrate tombstones every
    /// rel attached to `node` FIRST (within the same per-tenant
    /// `Transaction`), then tombstones the node itself. When
    /// `detach = false` AND the node has attached rels, the
    /// substrate returns
    /// `Err(SubstrateAccessError::Io("relationships attached"))` —
    /// the executor maps this to the openCypher v9 §6
    /// "relationships attached" runtime error (a dedicated
    /// `RelationshipsAttached` variant lights at v1.1+ per the
    /// 7-slice 3-strike rule).
    ///
    /// Returns `Ok(())` on success (MVCC tombstone committed).
    /// Production implementations open a per-tenant `Transaction`
    /// per ADR-031 (CommitBundle) + ADR-033 (rollback) and commit
    /// at this call boundary (one-call-one-tx at Phase 3 per
    /// ADR-149 §D-8).
    ///
    /// # Default impl
    ///
    /// Returns `IndexUnavailable("write-op delete_node unavailable
    /// on this substrate")` so existing substrate impls (read-only
    /// test fixtures) compile unchanged. Substrates that support
    /// DELETE override this method.
    fn delete_node(
        &self,
        _tenant: TenantId,
        _node: NodeId,
        _detach: bool,
        _ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable(
            "write-op delete_node unavailable on this substrate".into(),
        ))
    }

    /// **ADR-149 W26-θ Phase 3.** Tombstone the relationship
    /// identified by `rel`. Per ADR-018 the tombstone is a pure
    /// MVCC version-chain operation; no page is touched, pre-delete
    /// snapshots continue to read the prior version through MVCC.
    ///
    /// Returns `Ok(())` on success.
    ///
    /// # Default impl
    ///
    /// Returns `IndexUnavailable("write-op delete_rel unavailable
    /// on this substrate")` so existing substrate impls (read-only
    /// test fixtures) compile unchanged.
    fn delete_rel(
        &self,
        _tenant: TenantId,
        _rel: RelId,
        _ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable(
            "write-op delete_rel unavailable on this substrate".into(),
        ))
    }

    /// **ADR-150 W26-θ Phase 4.** Apply a [`SetNodeMutation`] to the
    /// node identified by `node`. Production implementations route
    /// PropertyAssign, PropertyReplace, and PropertyMerge through
    /// `arcgraph_storage::crud::update_node` per ADR-150 §D-7 (using
    /// `PropertyData::Empty` at v1.0-α); LabelAdd surfaces
    /// `IndexUnavailable("...forward-pinned to v1.1...")` per ADR-150
    /// §D-9 because the storage `update_node` primitive preserves
    /// `label_id` immutably per `crud.rs:3754` "PR #170 reviewer
    /// Finding 4".
    ///
    /// Returns `Ok(())` on success. Production implementations open a
    /// per-tenant `Transaction` per ADR-031 + ADR-033 and commit at
    /// this call boundary (one-call-one-tx at Phase 4 per ADR-150
    /// §D-8).
    ///
    /// # Default impl
    ///
    /// Returns `IndexUnavailable("write-op set_node unavailable on
    /// this substrate")` so existing substrate impls compile
    /// unchanged.
    fn set_node(
        &self,
        _tenant: TenantId,
        _node: NodeId,
        _mutation: &SetNodeMutation,
        _ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable(
            "write-op set_node unavailable on this substrate".into(),
        ))
    }

    /// **ADR-150 W26-θ Phase 4.** Apply a [`SetRelMutation`] to the
    /// relationship identified by `rel`. Production implementations
    /// route property mutations through
    /// `arcgraph_storage::crud::update_rel`.
    ///
    /// # Default impl
    ///
    /// Returns `IndexUnavailable("write-op set_rel unavailable on
    /// this substrate")`.
    fn set_rel(
        &self,
        _tenant: TenantId,
        _rel: RelId,
        _mutation: &SetRelMutation,
        _ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable(
            "write-op set_rel unavailable on this substrate".into(),
        ))
    }

    /// **ADR-150 W26-θ Phase 4.** Apply a [`RemoveNodeMutation`] to
    /// the node identified by `node`. Production implementations route
    /// `Property` through `arcgraph_storage::crud::update_node` (per
    /// ADR-150 §D-7 PropertyData::Empty v1.0-α posture); `LabelRemove`
    /// surfaces `IndexUnavailable("...forward-pinned to v1.1...")`
    /// per ADR-150 §D-9.
    ///
    /// # Default impl
    ///
    /// Returns `IndexUnavailable("write-op remove_node unavailable on
    /// this substrate")`.
    fn remove_node(
        &self,
        _tenant: TenantId,
        _node: NodeId,
        _mutation: &RemoveNodeMutation,
        _ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable(
            "write-op remove_node unavailable on this substrate".into(),
        ))
    }

    /// **ADR-150 W26-θ Phase 4.** Apply a [`RemoveRelMutation`] to
    /// the relationship identified by `rel`. Production
    /// implementations route `Property` through
    /// `arcgraph_storage::crud::update_rel`.
    ///
    /// # Default impl
    ///
    /// Returns `IndexUnavailable("write-op remove_rel unavailable on
    /// this substrate")`.
    fn remove_rel(
        &self,
        _tenant: TenantId,
        _rel: RelId,
        _mutation: &RemoveRelMutation,
        _ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable(
            "write-op remove_rel unavailable on this substrate".into(),
        ))
    }

    /// **#830 / ADR-200.** Register `entry` in the per-tenant
    /// vector-index catalog (the `CREATE VECTOR INDEX` accept-and-register
    /// path). This is a METADATA write ONLY — the served HNSW BUILD is
    /// auto-on-ingest (#765 PART-1), so registering an entry does NOT
    /// trigger a heavyweight build.
    ///
    /// `IF NOT EXISTS` semantics (a common Neo4j-compatible client
    /// form):
    /// - name absent  → insert + return [`VectorIndexRegistration::Created`].
    /// - name present + `if_not_exists` → idempotent no-op + return
    ///   [`VectorIndexRegistration::AlreadyExists`] (the pre-existing
    ///   entry is retained unchanged).
    /// - name present + NOT `if_not_exists` → return
    ///   [`SubstrateAccessError::IndexAlreadyExists`].
    ///
    /// The check-and-insert is performed atomically under the substrate's
    /// own lock (no TOCTOU race for concurrent same-name creates).
    ///
    /// # Default impl
    ///
    /// Returns `IndexUnavailable` so existing read-only substrate impls
    /// compile unchanged; the production [`crate::executor::ExecutorSubstrate`]
    /// binding (`arcgraph_mcp`'s `CrudExecutorSubstrate`) and the test
    /// [`StubExecutorSubstrate`] override it.
    fn register_vector_index(
        &self,
        _tenant: TenantId,
        _entry: VectorIndexCatalogEntry,
        _if_not_exists: bool,
    ) -> Result<VectorIndexRegistration, SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable(
            "vector-index catalog register unavailable on this substrate".into(),
        ))
    }

    /// **#830 / ADR-200.** List the registered vector indexes for
    /// `tenant` — the `SHOW VECTOR INDEXES` read path. Order is
    /// implementation-defined; the production + stub impls return
    /// registration order for determinism.
    ///
    /// # Default impl
    ///
    /// Returns an empty `Vec` (no catalog on read-only fixtures); an
    /// empty catalog legitimately yields zero `SHOW VECTOR INDEXES` rows.
    fn list_vector_indexes(&self, _tenant: TenantId) -> Vec<VectorIndexCatalogEntry> {
        Vec::new()
    }

    /// **#830 / ADR-200.** Resolve `name → catalog entry` for `tenant`
    /// — the TRUTHFUL `db.index.vector.queryNodes(name, …)`
    /// name→property resolution. Returns `None` when no entry matches,
    /// at which point the caller falls back to the served-convention
    /// property (`embedding`) for back-compat with the pre-catalog
    /// advisory-name behavior (#861).
    ///
    /// # Default impl
    ///
    /// A linear filter over [`Self::list_vector_indexes`] (a tenant has
    /// O(1) named vector indexes at v1.0-α — the langchain happy path
    /// registers exactly one). Production impls MAY override for an
    /// indexed lookup.
    fn resolve_vector_index(
        &self,
        tenant: TenantId,
        name: &str,
    ) -> Option<VectorIndexCatalogEntry> {
        self.list_vector_indexes(tenant)
            .into_iter()
            .find(|e| e.name == name)
    }

    /// **#1366 (task #248, Phase 1).** `CREATE INDEX <name> [IF NOT
    /// EXISTS] FOR (var:Label) ON (var.prop)` — the user-visible
    /// property index accept-register-AND-backfill path.
    ///
    /// Unlike the vector-index register (metadata-only), this is a
    /// heavyweight op: the production impl registers the index in the
    /// durable property-index catalog as `Building`, backfills the
    /// MVCC-visible nodes once (extract the declared property, compute
    /// the canonical key, insert into the secondary B+tree), then flips
    /// `Online` co-committed with the final backfill watermark. `IF NOT
    /// EXISTS` is idempotent (a re-create is a no-op, no re-backfill).
    ///
    /// # Default impl
    ///
    /// Returns `IndexUnavailable` so read-only fixtures compile; the
    /// production `CrudExecutorSubstrate` and the test
    /// `StubExecutorSubstrate` override it.
    fn create_property_index(
        &self,
        _tenant: TenantId,
        _name: &str,
        _if_not_exists: bool,
        _label: &str,
        _property: &str,
    ) -> Result<PropertyIndexRegistration, SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable(
            "property-index CREATE unavailable on this substrate".into(),
        ))
    }
}

/// **#1366 (task #248).** The outcome of a
/// [`ExecutorSubstrate::create_property_index`] call — distinguishes a
/// fresh register+backfill from an `IF NOT EXISTS` idempotent no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyIndexRegistration {
    /// A new index was registered and backfilled (now `Online`).
    Created,
    /// An index of the same name already existed; the `IF NOT EXISTS`
    /// create was an idempotent no-op (no re-backfill).
    AlreadyExists,
}

// =====================================================================
// ADR-150 W26-θ Phase 4 — SET / REMOVE mutation enums
// =====================================================================

/// The four bound SET-node mutation shapes per ADR-150 §D-7.
///
/// `Value`s in `PropertyAssign` / `PropertyReplace` / `PropertyMerge`
/// are runtime literals materialized by the executor at first-batch
/// time (per ADR-150 §D-4 inherited from ADR-147 §D-4 literal-only
/// narrowing).
#[derive(Debug, Clone, PartialEq)]
pub enum SetNodeMutation {
    /// `SET n.prop = value` — per-key write.
    PropertyAssign { name: String, value: Value },
    /// `SET n = {k: v, ...}` — full bag overwrite per ADR-150 §D-1.
    PropertyReplace(Vec<(String, Value)>),
    /// `SET n += {k: v, ...}` — additive merge per ADR-150 §D-1.
    PropertyMerge(Vec<(String, Value)>),
    /// `SET n:L1:L2` — multi-label add (Node-only per ADR-150 §D-4;
    /// the production substrate at v1.0-α surfaces
    /// `IndexUnavailable` per ADR-150 §D-9 forward-pin to v1.1).
    LabelAdd(Vec<String>),
}

/// The three bound SET-rel mutation shapes per ADR-150 §D-7. Rels do
/// not carry labels at v1.0-α per ADR-150 §D-4 so the LabelAdd variant
/// is absent from this enum.
#[derive(Debug, Clone, PartialEq)]
pub enum SetRelMutation {
    PropertyAssign { name: String, value: Value },
    PropertyReplace(Vec<(String, Value)>),
    PropertyMerge(Vec<(String, Value)>),
}

/// The two bound REMOVE-node mutation shapes per ADR-150 §D-7.
#[derive(Debug, Clone, PartialEq)]
pub enum RemoveNodeMutation {
    /// `REMOVE n.prop` — per-key clear.
    Property(String),
    /// `REMOVE n:L1:L2` — multi-label remove (Node-only per ADR-150
    /// §D-4; the production substrate at v1.0-α surfaces
    /// `IndexUnavailable` per ADR-150 §D-9).
    LabelRemove(Vec<String>),
}

/// The single bound REMOVE-rel mutation shape per ADR-150 §D-7. Rels
/// do not carry labels at v1.0-α so LabelRemove is absent.
#[derive(Debug, Clone, PartialEq)]
pub enum RemoveRelMutation {
    Property(String),
}

/// One node returned from a scan / community lookup.
///
/// Owned struct (not a reference) so the executor can re-use the
/// stub substrate's fixture data across batches without lifetime
/// gymnastics. v1.0-alpha clones are cheap (the fixture row count is
/// bounded by the test harness); production wiring at M4-08+ may
/// switch to `Cow<'_, NodeView>` for zero-copy CRUD reads if
/// profiling motivates.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundNode {
    /// The node value (id + label + properties).
    pub node: NodeView,
}

/// One edge returned from an expand call.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundEdge {
    /// The matched relationship.
    pub rel: RelView,
    /// The destination node (the FAR end of the traversal — opposite
    /// of `LogicalExpand::from`).
    pub dst: NodeView,
}

/// Owned streaming cursor over one-hop expansion.
pub type BoundEdgeCursor =
    Box<dyn Iterator<Item = Result<BoundEdge, SubstrateAccessError>> + Send + 'static>;

/// One ranked hit returned from a vector / BM25 retrieval.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedHit {
    /// The matched node.
    pub node: NodeView,
    /// The substrate's score for this hit. Distance-vs-similarity
    /// convention is substrate-defined; the
    /// [`crate::executor::ops::rank_by_hybrid::RankByHybridOp`] only
    /// reads the rank ORDER (RRF fusion is rank-based), not the
    /// absolute score.
    pub score: f64,
}

/// Substrate-access error.
///
/// `#[non_exhaustive]` under the code-quality policy — `SubstrateAccessError` is a
/// public enum returned through `ExecutorSubstrate` impls, and future
/// substrate variants (e.g. M4-08+ production routing) MUST be able to
/// land additive variants without breaking downstream pattern matches.
///
/// `Eq` is derived alongside `PartialEq` so the W11Z fix-up MED-2
/// translation in `ExplainError` can derive `Eq` end-to-end. Every
/// variant carries only `Eq`-bearing payloads.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SubstrateAccessError {
    /// Tenant identity not known to this substrate. Should be rare —
    /// the catalog already gated the tenant ID — but the production
    /// binding may surface this on a stale handle.
    #[error("tenant {0:?} unknown to substrate")]
    TenantUnknown(TenantId),

    /// The requested substrate (vector / bm25 / community) is not
    /// attached for this tenant. Should be rare — M4-23 cross-
    /// substrate validation gates this — but the executor surfaces it
    /// defensively.
    #[error("substrate `{0}` unavailable for tenant")]
    IndexUnavailable(String),

    /// A finite snapshot was requested from a substrate that can only
    /// open a transaction at `available`. Returning current data here
    /// would falsely report that `requested` was honoured, so production
    /// substrates fail closed with both LSNs. `Lsn::MAX` is the
    /// read-latest sentinel and does not produce this error.
    #[error(
        "requested snapshot LSN {requested:?} is unavailable; \
         available snapshot LSN is {available:?}"
    )]
    SnapshotUnavailable {
        /// Exact finite LSN requested by the caller.
        requested: Lsn,
        /// Snapshot the substrate was able to open.
        available: Lsn,
    },

    /// Generic I/O / WAL failure during substrate access.
    /// v1.0-alpha stub substrates never surface this; production
    /// wiring at M4-08+ may.
    #[error("substrate I/O error: {0}")]
    Io(String),

    /// **v2 M2 (ADR-230 row M2, design §M2.2).** A record's typed
    /// property payload is structurally corrupt (unknown block
    /// version/type tag, out-of-range offsets, an unresolvable
    /// interned key, an unknown payload discriminant, …). LOUD by
    /// design: the pre-M2 JSON path silently degraded a corrupt bag to
    /// an empty one — a wrong-read; the typed path rejects instead.
    /// Distinct from [`Self::Io`] so operators and the MCP boundary
    /// can distinguish data corruption from transient I/O.
    #[error("corrupt property payload: {0}")]
    CorruptPropertyPayload(String),

    /// **ADR-197-amendment-01 D-4.** A read was attempted inside a
    /// Bolt explicit transaction against a substrate that has not
    /// implemented held-transaction reads (the `_with_context`
    /// override). Surfaced by the trait DEFAULTS so a non-overriding
    /// substrate fails loud instead of silently serving committed-read
    /// isolation (wrong RYW + wrong snapshot — the #822 bug shape).
    /// Payload = the read surface name (`"scan_nodes"` / `"expand"`).
    #[error(
        "held-transaction reads unsupported by this substrate for `{0}` \
         (ADR-197-amendment-01 D-4: override `{0}_with_context` to read \
         through the held transaction)"
    )]
    HeldTxnReadsUnsupported(String),

    /// A `query_vec` carries a dimension that does not match the derived
    /// index's established dimension (a single-dimension-per-index
    /// substrate). Distinct from [`Self::Io`] so the MCP boundary maps it
    /// to a precise client-facing `invalid params` error (#786) instead of
    /// the cryptic `-32006 execution eval` catch-all the generic `Io`
    /// bucket renders.
    #[error(
        "query_vec dimension {query_dim} does not match index dimension \
         {index_dim} for property `{property}`"
    )]
    DimensionMismatch {
        /// The vector property whose established index dimension was violated.
        property: String,
        /// The dimension of the offending query vector.
        query_dim: usize,
        /// The index's established dimension (inferred from the first
        /// embedding-bearing node).
        index_dim: usize,
    },

    /// **#830 / ADR-200.** A `CREATE VECTOR INDEX <name>` (WITHOUT
    /// `IF NOT EXISTS`) named an index already present in the per-tenant
    /// vector-index catalog. Distinct from [`Self::Io`] so the MCP
    /// boundary maps it to a precise client-facing `invalid params`
    /// (`-32602`) error carrying the index name — mirroring the #786
    /// [`Self::DimensionMismatch`] decision and Neo4j's
    /// `EquivalentSchemaRuleAlreadyExists` (a `ClientError`) — instead of
    /// the cryptic `-32006 execution eval` catch-all the generic `Io`
    /// bucket renders. (See the `substrate_to_mcp` dedicated arm +
    /// `substrate_index_already_exists_maps_to_minus_32602_invalid_params`
    /// boundary test in `arcgraph-mcp`.) Compatible vector clients
    /// commonly emit `IF NOT EXISTS`, so this surfaces only on the
    /// raw-Cypher / negative path.
    #[error("a vector index named `{name}` already exists")]
    IndexAlreadyExists {
        /// The conflicting index name.
        name: String,
    },

    /// **#907.** A write-write **MVCC conflict** — the optimistic-
    /// concurrency loser whose commit lost the OCC validation race
    /// (`arcgraph_storage::crud::commit` →
    /// `CrudError::Mvcc(ArcGraphError::MvccConflict)`). This is a
    /// *logical* serialization conflict and a NORMAL, expected outcome of
    /// optimistic concurrency under write contention — NOT a *physical*
    /// I/O fault — and the transaction may succeed if retried.
    ///
    /// Distinct from [`Self::Io`] so the public boundary can classify it
    /// as a **retriable transient transaction error** (the Bolt boundary
    /// maps it to `Neo.TransientError.Transaction.*`, which Neo4j drivers
    /// AUTO-RETRY under `session.execute_write` / `execute_read`) instead
    /// of flattening it into the generic `Io` bucket. Before #907 the
    /// conflict rode `Io` → the Bolt boundary emitted a *fatal*
    /// `Neo.DatabaseError.General.UnknownError` (non-retriable) AND leaked
    /// the storage-layer wrapping ("substrate I/O error: write commit
    /// failed: MVCC commit failed") to the client, breaking the standard
    /// optimistic-concurrency retry pattern.
    ///
    /// Mirrors the [`Self::DimensionMismatch`] / [`Self::IndexAlreadyExists`]
    /// precedent in this enum: a precise typed variant the boundary maps
    /// deliberately, never the cryptic catch-all. The `target` is carried
    /// for server-side diagnostics; client-facing renderers surface a
    /// clean "retry the transaction" message WITHOUT the internal layering.
    #[error("transaction conflict on {target}; retry the transaction")]
    Conflict {
        /// The contention point reported by the MVCC kernel, carried
        /// verbatim from [`arcgraph_core::ArcGraphError::MvccConflict`]'s
        /// `target` (e.g. an internal `key:N` version-store key). For
        /// diagnostics / structured logging only — the Bolt FAILURE
        /// message deliberately does NOT echo it (no internal leak, #907).
        target: String,
    },
}

// =====================================================================
// ADR-200 — minimal vector-index catalog (the #830 D2/D3 half of
// ADR-198 §OQ-7). `CREATE VECTOR INDEX` registers an entry; `SHOW
// VECTOR INDEXES` reflects it; `db.index.vector.queryNodes(name, …)`
// resolves `name → property` truthfully against it (falling back to
// the served-convention property when no entry matches).
// =====================================================================

/// **#830 / ADR-198 §OQ-7 / ADR-200.** A registered vector-index
/// catalog entry — the minimal per-tenant metadata a `CREATE VECTOR
/// INDEX` registers.
///
/// The served HNSW index BUILD is auto-on-ingest (#765 PART-1 — the
/// `SubstrateSearchProvider` builds the per-tenant index from every
/// node carrying the vector property); this catalog entry is pure
/// METADATA (name → label/property/dims/similarity). It carries NO
/// index data — registering an entry does NOT trigger a build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorIndexCatalogEntry {
    /// The registered index name (a `$param` name is resolved to its
    /// string value BEFORE registration).
    pub name: String,
    /// The node label the index is `FOR (var:Label)` (e.g. `CzChunk`).
    pub label: String,
    /// The indexed vector property `ON var.property` (e.g. `embedding`).
    pub property: String,
    /// The configured dimension from `OPTIONS` `vector.dimensions`.
    /// `None` when `OPTIONS` omitted it (the served index infers the
    /// dimension from the first embedding-bearing node per #786).
    pub dimensions: Option<u32>,
    /// The similarity function from `OPTIONS` `vector.similarity_function`
    /// (`cosine` / `euclidean`). `None` when `OPTIONS` omitted it.
    pub similarity_function: Option<String>,
}

/// **ADR-200.** The outcome of a [`ExecutorSubstrate::register_vector_index`]
/// call — distinguishes a fresh insert from an `IF NOT EXISTS`
/// idempotent no-op (so the executor can stay silent / observably log
/// without erroring on a duplicate create-if-not-exists).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorIndexRegistration {
    /// A new catalog entry was inserted.
    Created,
    /// An entry of the same name already existed; the `IF NOT EXISTS`
    /// create was an idempotent no-op (the pre-existing entry is
    /// retained unchanged).
    AlreadyExists,
}

// =====================================================================
// StubExecutorSubstrate — in-memory fixture for tests.
// =====================================================================

/// In-memory [`ExecutorSubstrate`] impl for tests.
///
/// Mirrors the [`crate::semantic::StubCatalogProvider`] pattern: a
/// fluent builder (`with_node`, `with_edge`, `with_vector_hit`,
/// `with_bm25_hit`, `with_community_membership`) populates an
/// in-memory store; the trait methods read deterministically from it.
///
/// # Determinism
///
/// All accessors return owned `Vec`s in INSERTION order. Tests that
/// need a different traversal order can re-order their fixture
/// inserts. Production substrates have no such guarantee — operator
/// tests that assert a specific row order MUST use the stub or
/// build their own deterministic mock.
///
/// # Per-tenant isolation
///
/// The fixture is keyed by `(TenantId, NodeId)` / `(TenantId, RelId)`
/// internally so a multi-tenant test can populate two tenants in the
/// same stub and verify isolation per ADR-037 D-1.
#[derive(Debug, Clone, Default)]
pub struct StubExecutorSubstrate {
    /// Per-tenant node store keyed by NodeId.
    nodes: HashMap<TenantId, HashMap<NodeId, NodeView>>,
    /// Per-tenant edge store keyed by RelId.
    edges: HashMap<TenantId, HashMap<RelId, RelView>>,
    /// Per-tenant adjacency: node → list of (rel_id, dst_id, dir).
    /// `dir` is the direction of the rel from the FROM-side
    /// node's perspective (LeftToRight = outbound from key).
    adjacency: HashMap<TenantId, HashMap<NodeId, Vec<AdjacencyEntry>>>,
    /// Per-tenant pre-baked vector top-K. Keyed by `(property,
    /// query-id)` where query-id is a deterministic stand-in for the
    /// full query vector (tests register a hit list against a
    /// caller-supplied tag).
    vector_hits: HashMap<(TenantId, String), Vec<RankedHit>>,
    /// Per-tenant pre-baked BM25 top-K. Keyed by `(property,
    /// query-text)`.
    bm25_hits: HashMap<(TenantId, String, String), Vec<RankedHit>>,
    /// Per-tenant community-membership lookup keyed by community-id.
    community_members: HashMap<(TenantId, i64), Vec<NodeView>>,
    /// Substrate-availability flags (mirror the
    /// `CatalogProvider::has_*_index` shape).
    has_vector: bool,
    has_bm25: bool,
    has_community: bool,
    /// Optional pre-baked tenant-wide counts-store totals. When absent,
    /// the stub derives totals from its in-memory rows.
    count_store_totals: HashMap<(TenantId, CountStoreSource), u64>,
    /// **ADR-147 W26-θ Phase 1.** Created-node bookkeeping shared
    /// across `Clone`s so a test that constructs the stub, executes
    /// a `CREATE` query, then runs a `MATCH` query observes the
    /// freshly-CREATEd node. The two sub-stores are wrapped in a
    /// single `Arc<std::sync::Mutex<_>>` for thread-safe interior
    /// mutability; the trait impl's `&self` receiver requires this.
    create_state: std::sync::Arc<std::sync::Mutex<CreateState>>,
    /// **#1366 (Phase 2).** Declared property indexes as
    /// `(tenant, label, property)` — the set the stub's
    /// [`Self::property_index_lookup_with_context`] recognizes as an
    /// index it can serve. A lookup on a `(label, property)` NOT in this
    /// set returns empty (mirrors "no declared index").
    property_index_declared: std::collections::HashSet<(TenantId, LabelId, String)>,
    /// **#1366 (Phase 2).** Pre-baked B+tree CANDIDATE slots keyed by
    /// `(tenant, label, property, canonical-value-key)`. A test seeds the
    /// candidate NodeIds a real B+tree WOULD return — INCLUDING stale /
    /// duplicate / snapshot-invisible ids — so the executor op's
    /// candidate-then-verify + dedup path is exercised end-to-end: the
    /// stub seam hydrates each candidate via
    /// [`Self::node_by_id_with_context`], re-checks label + property,
    /// drops the ones that fail, and dedups by NodeId.
    property_index_candidates: HashMap<(TenantId, LabelId, String, String), Vec<NodeId>>,
}

/// Interior state shared across `StubExecutorSubstrate::clone()`
/// instances so CREATE writes are visible to subsequent MATCH reads
/// across clones. The `tombstoned_*` sets ADR-149 Phase 3 additions
/// are used to filter pre-built node / edge entries that have been
/// tombstoned by an in-flight DELETE.
#[derive(Debug, Default)]
struct CreateState {
    /// Per-tenant CREATE-introduced nodes. The trait impl unions
    /// these with the pre-built `nodes` field at `scan_nodes` time.
    nodes: HashMap<TenantId, Vec<NodeView>>,
    /// Per-tenant interned label-name → LabelId map. Phase 1 the
    /// stub allocates fresh ids per (tenant, name); a real production
    /// substrate routes through `arcgraph_storage::InternTable`.
    labels: HashMap<(TenantId, String), LabelId>,
    /// Monotonic per-stub NodeId allocator (one counter across all
    /// tenants — matches the stub's existing `with_node`
    /// convention where the test caller-supplied id space avoids
    /// per-tenant overlap; CREATE allocates above the high-water
    /// mark to avoid collisions with caller-supplied fixtures).
    next_node: AtomicU64,
    /// Monotonic per-stub LabelId allocator. Reserves the first
    /// 1024 ids for caller-supplied test fixtures so CREATE-side
    /// allocations don't collide.
    next_label: AtomicU64,
    /// ADR-148 W26-θ Phase 2 — per-tenant CREATE-introduced edges
    /// (full RelView) for subsequent `expand` round-trip.
    edges: HashMap<TenantId, Vec<RelView>>,
    /// Per-tenant CREATE-introduced adjacency: source NodeId → list
    /// of (rel_id, dst, direction). Mirrors the pre-built
    /// `adjacency` field convention.
    adjacency: HashMap<TenantId, HashMap<NodeId, Vec<AdjacencyEntry>>>,
    /// Per-tenant interned rel-type-name → TypeId map (Phase 2;
    /// parallel to `labels`).
    rel_types: HashMap<(TenantId, String), TypeId>,
    /// Monotonic per-stub RelId allocator (Phase 2). Above 2^32
    /// boundary so caller-supplied fixture rel ids stay disjoint.
    next_rel: AtomicU64,
    /// Monotonic per-stub TypeId allocator (Phase 2). Reserves the
    /// first 1024 ids for caller-supplied fixtures.
    next_rel_type: AtomicU64,
    /// ADR-149 W26-θ Phase 3 — per-tenant set of tombstoned node ids.
    /// Used by `scan_nodes` to filter pre-built nodes that the
    /// in-flight DELETE has tombstoned. Mirrors the storage layer's
    /// MVCC tombstone semantic per ADR-018 (the version-chain at the
    /// stub level is degenerate; the set IS the visibility filter).
    tombstoned_nodes: HashMap<TenantId, std::collections::HashSet<NodeId>>,
    /// ADR-149 W26-θ Phase 3 — per-tenant set of tombstoned rel ids.
    /// Used by `expand` to filter pre-built rels that the in-flight
    /// DELETE has tombstoned.
    tombstoned_rels: HashMap<TenantId, std::collections::HashSet<RelId>>,
    /// ADR-150 W26-θ Phase 4 — per-(tenant, NodeId) property bag.
    /// Tracks the post-SET / post-REMOVE property state for nodes so
    /// tests can verify end-to-end mutation. Production substrates
    /// route through `arcgraph_storage::crud::update_node` instead.
    node_properties: HashMap<(TenantId, NodeId), HashMap<String, Value>>,
    /// ADR-150 W26-θ Phase 4 — per-(tenant, RelId) property bag.
    /// Tracks the post-SET / post-REMOVE property state for rels.
    rel_properties: HashMap<(TenantId, RelId), HashMap<String, Value>>,
    /// ADR-150 W26-θ Phase 4 — per-(tenant, NodeId) ADDITIONAL labels.
    /// The pre-built / CREATE-d NodeView carries `Option<LabelId>` for
    /// its primary label; this sidecar tracks any additional labels
    /// added via `SET n:L1:L2` (stored as label NAMES; the multi-label
    /// NodeView shape is forward-pinned to v1.1 per ADR-150 §D-9).
    additional_labels: HashMap<(TenantId, NodeId), Vec<String>>,
    /// **#830 / ADR-200.** Per-tenant vector-index catalog (registration
    /// order). `CREATE VECTOR INDEX` appends; `SHOW VECTOR INDEXES`
    /// reads; `db.index.vector.queryNodes` resolves name→property
    /// against it. Interior-mutable like the other CreateState sidecars
    /// so a test that CREATEs an index then SHOWs / queries it across
    /// stub clones observes the registration.
    vector_indexes: HashMap<TenantId, Vec<VectorIndexCatalogEntry>>,
    /// **#1366 (task #248, Phase 1).** Per-tenant property-index names
    /// (registration order). `CREATE INDEX` appends the name; the stub
    /// backfill is a no-op (no fixture nodes to scan). Enables the
    /// executor-op tests to exercise the register + IF NOT EXISTS
    /// idempotency contract without the storage backend.
    property_indexes: HashMap<TenantId, Vec<String>>,
}

/// ADR-152 §D-3 helper — snapshot of the per-tenant SET/REMOVE
/// sidecars used by [`StubExecutorSubstrate::expand`] to override
/// the pre-built / CREATE-time property bags on emitted node + rel
/// views. Factored out per `clippy::type_complexity` reviewer
/// guidance + R1 readability pin.
struct ExpandOverrideSnapshot {
    tombstoned_rels: std::collections::HashSet<RelId>,
    node_overrides: HashMap<NodeId, HashMap<String, Value>>,
    rel_overrides: HashMap<RelId, HashMap<String, Value>>,
}

impl ExpandOverrideSnapshot {
    fn capture(stub: &StubExecutorSubstrate, tenant: TenantId) -> Self {
        let Some(state) = stub.create_state.lock().ok() else {
            return Self {
                tombstoned_rels: std::collections::HashSet::new(),
                node_overrides: HashMap::new(),
                rel_overrides: HashMap::new(),
            };
        };
        let tombstoned_rels = state
            .tombstoned_rels
            .get(&tenant)
            .cloned()
            .unwrap_or_default();
        let mut node_overrides = HashMap::new();
        for ((t, n), bag) in &state.node_properties {
            if *t == tenant {
                node_overrides.insert(*n, bag.clone());
            }
        }
        let mut rel_overrides = HashMap::new();
        for ((t, r), bag) in &state.rel_properties {
            if *t == tenant {
                rel_overrides.insert(*r, bag.clone());
            }
        }
        Self {
            tombstoned_rels,
            node_overrides,
            rel_overrides,
        }
    }
}

#[derive(Debug, Clone)]
struct AdjacencyEntry {
    rel_id: RelId,
    dst: NodeId,
    /// Direction relative to the source node (the map key). Used to
    /// match `Direction::LeftToRight` / `RightToLeft` / `Undirected`
    /// at scan time per the LogicalExpand contract.
    direction: Direction,
}

/// **#1366 (Phase 2).** A stable canonical string key for a lookup
/// value, used ONLY by the stub's in-memory candidate map. Mirrors the
/// production `canonical_key_for` (RC-4 / RC-5) EXACTLY in the
/// keyable-vs-unkeyable partition — distinct keyable values map to
/// distinct keys, and every UNKEYABLE value maps to the SAME unfindable
/// sentinel — but stays a debug-string (the stub has no B+tree).
///
/// # Fidelity (#1415)
///
/// A stub that keyed unsupported values by their debug-string (the old
/// `unsupported:{other:?}`) would let a SEEDED List/Map candidate be
/// FOUND — masking the production bug where `lookup_candidates` returns
/// EMPTY for a `None` canonical key. To keep a stub-level test honest,
/// this fn mirrors `canonical_key_for`'s `Some`/`None` partition:
///
/// - **String / Boolean** → keyable (stable per-value key).
/// - **Integer** → keyable ONLY when non-negative (the `u56` slot has no
///   sign bit; a negative i64 is `None` in production → unkeyable here).
/// - **Float** → keyable ONLY when integral + in the i64 range, and it
///   keys THROUGH THE INTEGER BUCKET so `10.0` collides with stored int
///   `10` (production coerces the integral float to the int key). A
///   fractional / out-of-range float is `None` → unkeyable.
/// - **Everything else** (`List` / `Map` / `Node` / `Null` / …) →
///   unkeyable.
///
/// Every unkeyable value returns the SAME sentinel; because a candidate
/// is only ever seeded under a keyable value's key
/// ([`StubExecutorSubstrate::with_property_index_candidate`] routes
/// through this same fn), a lookup for an unkeyable value NEVER finds a
/// candidate — matching production's empty lookup. (In practice the op
/// short-circuits unkeyable values to a Scan+Filter fallback via
/// [`ExecutorSubstrate::value_is_indexable`] before it ever reaches the
/// candidate map; the fidelity here is defense-in-depth so a stub-level
/// candidate test cannot pass while production is wrong.)
fn stub_value_key(value: &Value) -> String {
    /// The single sentinel every UNKEYABLE value maps to — no candidate
    /// is ever seeded under it, so a lookup for any unkeyable value is
    /// empty (production's `canonical_key_for` `None` behaviour).
    const UNKEYABLE: &str = "unkeyable";
    match value {
        Value::String(s) => format!("s:{s}"),
        Value::Boolean(b) => format!("b:{b}"),
        // Negative integers are unsupported in production (u56 slot has no
        // sign bit) → unkeyable. Non-negative → the integer bucket.
        Value::Integer(i) if *i >= 0 => format!("i:{i}"),
        // RC-5: only an INTEGRAL, in-i64-range float is index-eligible,
        // and it coerces to the SAME integer bucket key so `10.0` finds
        // stored int `10`. Fractional / non-finite / out-of-range → the
        // unkeyable sentinel (Float dropped as an indexed value type).
        Value::Float(f)
            if f.fract() == 0.0
                && *f >= i64::MIN as f64
                && *f <= i64::MAX as f64
                && (*f as i64) as f64 == *f =>
        {
            format!("i:{}", *f as i64)
        }
        // Negative integer / fractional-or-OOR float / List / Map / Node /
        // Null / … → the single unfindable sentinel.
        _ => UNKEYABLE.to_string(),
    }
}

/// **#1366 (Phase 2).** Whether a hydrated node still carries `label` as
/// its primary label AND `property == value` under the engine `=`
/// coercion. The candidate-then-verify recheck the executor op relies on;
/// a candidate that fails EITHER leg is dropped.
///
/// # Coercion fidelity (#1415)
///
/// Property equality mirrors production `index_value_eq_coerced`: the ONE
/// coercion engine `=` applies across the RC-supported set is numeric
/// (`Integer` ⇄ `Float`), so `10.0 = 10` (an integral-float lookup
/// against a stored int) matches. Every other pair is same-variant
/// `PartialEq`, so an integer candidate keyed under a string value (a
/// hash-collision analog) still fails the recheck. The prior version used
/// bare `PartialEq`, which would DROP an integral-float lookup against a
/// stored int — a stub divergence from production.
fn stub_node_matches(node: &NodeView, label: LabelId, property: &str, value: &Value) -> bool {
    if node.label != Some(label) {
        return false;
    }
    let Some(stored) = node.properties.get(property) else {
        return false;
    };
    match (stored, value) {
        // Numeric coercion, mirroring engine `=` (NN-4): an Integer lookup
        // matches a stored Float of equal magnitude and vice versa.
        (Value::Integer(a), Value::Float(b)) | (Value::Float(b), Value::Integer(a)) => {
            (*a as f64) == *b
        }
        (a, b) => a == b,
    }
}

impl StubExecutorSubstrate {
    /// Construct an empty stub. All `has_*` flags default to `false`;
    /// callers explicitly opt in via the fluent builders.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the per-tenant store. The node's `id` is the key.
    #[must_use]
    pub fn with_node(mut self, tenant: TenantId, node: NodeView) -> Self {
        self.nodes.entry(tenant).or_default().insert(node.id, node);
        self
    }

    /// Add a relationship to the per-tenant store + adjacency. The
    /// `rel.from` / `rel.to` endpoints are the adjacency keys.
    #[must_use]
    pub fn with_edge(mut self, tenant: TenantId, rel: RelView) -> Self {
        let from = rel.from;
        let to = rel.to;
        let rel_id = rel.id;
        self.edges.entry(tenant).or_default().insert(rel_id, rel);
        // Outbound edge from `from`.
        self.adjacency
            .entry(tenant)
            .or_default()
            .entry(from)
            .or_default()
            .push(AdjacencyEntry {
                rel_id,
                dst: to,
                direction: Direction::LeftToRight,
            });
        // Inbound edge from `to`'s perspective.
        self.adjacency
            .entry(tenant)
            .or_default()
            .entry(to)
            .or_default()
            .push(AdjacencyEntry {
                rel_id,
                dst: from,
                direction: Direction::RightToLeft,
            });
        self
    }

    /// Pre-bake a vector top-K result for a `(property, query-tag)`
    /// pair. The `query_tag` is the deterministic stand-in for the
    /// query vector; tests pass the same tag to
    /// [`Self::vector_search`] (via the operator's resolved query
    /// expression, which the stub maps deterministically — see
    /// [`Self::vector_search_tag_for`]).
    ///
    /// At v1.0-alpha the tag is the literal first f32's bit pattern,
    /// rendered as a hex string. This is stub-friendly: a test that
    /// registers `(prop, "0x40490fdb", hits)` (≈ π) and queries with
    /// `[3.14159, ...]` matches deterministically.
    #[must_use]
    pub fn with_vector_hit(
        mut self,
        tenant: TenantId,
        property: impl Into<String>,
        query_tag: impl Into<String>,
        hits: Vec<RankedHit>,
    ) -> Self {
        let key = (tenant, property.into());
        // The tag is part of the property key for deterministic
        // multi-query fixtures; HashMap key is `(tenant, property +
        // "@" + tag)` for collision avoidance.
        let entry_key = (key.0, format!("{}@{}", key.1, query_tag.into()));
        self.vector_hits.insert(entry_key, hits);
        self
    }

    /// Compute the deterministic stub-side tag for a vector query.
    ///
    /// Used by tests: register hits with the same tag, then query.
    /// The first `f32` in the query vector (rendered as a hex string)
    /// is the canonical tag.
    #[must_use]
    pub fn vector_search_tag_for(query: &[f32]) -> String {
        match query.first().copied() {
            Some(f) => format!("0x{:08x}", f.to_bits()),
            None => "<empty>".into(),
        }
    }

    /// Pre-bake a BM25 top-K result for a `(property, query-text)`
    /// pair.
    #[must_use]
    pub fn with_bm25_hit(
        mut self,
        tenant: TenantId,
        property: impl Into<String>,
        query_text: impl Into<String>,
        hits: Vec<RankedHit>,
    ) -> Self {
        self.bm25_hits
            .insert((tenant, property.into(), query_text.into()), hits);
        self
    }

    /// Pre-bake a community-members list for a `community_id`.
    #[must_use]
    pub fn with_community_membership(
        mut self,
        tenant: TenantId,
        community_id: i64,
        members: Vec<NodeView>,
    ) -> Self {
        self.community_members
            .insert((tenant, community_id), members);
        self
    }

    /// Mark the vector substrate as attached.
    #[must_use]
    pub fn with_vector_substrate(mut self) -> Self {
        self.has_vector = true;
        self
    }

    /// Mark the BM25 substrate as attached.
    #[must_use]
    pub fn with_bm25_substrate(mut self) -> Self {
        self.has_bm25 = true;
        self
    }

    /// Mark the community substrate as attached.
    #[must_use]
    pub fn with_community_substrate(mut self) -> Self {
        self.has_community = true;
        self
    }

    /// **#1366 (Phase 2).** Declare a property index on
    /// `(tenant, label, property)` for the stub — the set the
    /// [`Self::property_index_lookup_with_context`] seam recognizes as an
    /// index it can serve candidates for. Pair with
    /// [`Self::with_property_index_candidate`] to seed the B+tree slots.
    #[must_use]
    pub fn with_property_index(mut self, tenant: TenantId, label: LabelId, property: &str) -> Self {
        self.property_index_declared
            .insert((tenant, label, property.to_string()));
        self
    }

    /// **#1366 (Phase 2).** Seed the B+tree CANDIDATE NodeIds a lookup on
    /// `(tenant, label, property = value)` returns — the raw slots BEFORE
    /// MVCC verify. Callers deliberately include stale / duplicate /
    /// snapshot-invisible ids to prove the executor op's
    /// candidate-then-verify + dedup drops them. Appends (multiple calls
    /// accumulate), so a duplicate id can be seeded twice to exercise the
    /// dedup path.
    #[must_use]
    pub fn with_property_index_candidate(
        mut self,
        tenant: TenantId,
        label: LabelId,
        property: &str,
        value: &Value,
        candidate: NodeId,
    ) -> Self {
        let key = (tenant, label, property.to_string(), stub_value_key(value));
        self.property_index_candidates
            .entry(key)
            .or_default()
            .push(candidate);
        self
    }

    /// Pre-bake a tenant-wide counts-store total.
    #[must_use]
    pub fn with_count_store_total(
        mut self,
        tenant: TenantId,
        source: CountStoreSource,
        count: u64,
    ) -> Self {
        self.count_store_totals.insert((tenant, source), count);
        self
    }

    /// ADR-150 W26-θ Phase 4 test accessor — snapshot the node's
    /// post-SET property bag (or `None` if no SET / REMOVE has touched
    /// the node).
    #[must_use]
    pub fn node_properties(
        &self,
        tenant: TenantId,
        node: NodeId,
    ) -> Option<HashMap<String, Value>> {
        self.create_state
            .lock()
            .ok()
            .and_then(|state| state.node_properties.get(&(tenant, node)).cloned())
    }

    /// ADR-150 W26-θ Phase 4 test accessor — snapshot the rel's
    /// post-SET property bag.
    #[must_use]
    pub fn rel_properties(&self, tenant: TenantId, rel: RelId) -> Option<HashMap<String, Value>> {
        self.create_state
            .lock()
            .ok()
            .and_then(|state| state.rel_properties.get(&(tenant, rel)).cloned())
    }

    /// ADR-150 W26-θ Phase 4 test accessor — snapshot the node's
    /// additional (SET-added) label sidecar.
    #[must_use]
    pub fn additional_labels(&self, tenant: TenantId, node: NodeId) -> Vec<String> {
        self.create_state
            .lock()
            .ok()
            .and_then(|state| state.additional_labels.get(&(tenant, node)).cloned())
            .unwrap_or_default()
    }
}

impl ExecutorSubstrate for StubExecutorSubstrate {
    // ADR-197-amendment-01 D-4 — CONSCIOUS overrides of the loud
    // defaults: for the stub, plain delegation is EXACT (not a
    // degradation). Stub writes mutate the in-memory fixture
    // immediately (create/set/delete are visible to the very next
    // read regardless of any held transaction), so committed-read ≡
    // read-your-writes here and there is no snapshot to pin. Engine
    // tests that drive `execute_in_txn` against the stub therefore
    // keep working, while a substrate with REAL deferred staging
    // (the production CrudExecutorSubstrate) must route through the
    // held transaction or fail loud via the trait default.
    fn scan_nodes_with_context(
        &self,
        ctx: &ExecutionContext,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        self.scan_nodes(ctx.tenant(), label, read_lsn)
    }

    fn expand_with_context(
        &self,
        ctx: &ExecutionContext,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        self.expand(ctx.tenant(), from, rel_type, direction, read_lsn)
    }

    fn node_by_id_with_context(
        &self,
        ctx: &ExecutionContext,
        id: NodeId,
    ) -> Result<Option<BoundNode>, SubstrateAccessError> {
        let tenant = ctx.tenant();
        let (tombstoned, prop_override) = {
            let state_guard = self.create_state.lock().ok();
            match state_guard {
                Some(state) => (
                    state
                        .tombstoned_nodes
                        .get(&tenant)
                        .map(|set| set.contains(&id))
                        .unwrap_or(false),
                    state.node_properties.get(&(tenant, id)).cloned(),
                ),
                None => (false, None),
            }
        };
        if tombstoned {
            return Ok(None);
        }

        let mut node = self
            .nodes
            .get(&tenant)
            .and_then(|nodes| nodes.get(&id))
            .cloned()
            .or_else(|| {
                self.create_state.lock().ok().and_then(|state| {
                    state
                        .nodes
                        .get(&tenant)
                        .and_then(|nodes| nodes.iter().find(|n| n.id == id).cloned())
                })
            });

        if let (Some(n), Some(bag)) = (&mut node, prop_override) {
            n.properties.clear();
            for (k, v) in bag {
                n.properties.insert(k, v);
            }
        }

        Ok(node.map(|node| BoundNode { node }))
    }

    fn property_index_lookup_with_context(
        &self,
        ctx: &ExecutionContext,
        label: LabelId,
        property: &str,
        value: &Value,
        _read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let tenant = ctx.tenant();
        // No declared index on this (label, property) ⇒ nothing to serve.
        if !self
            .property_index_declared
            .contains(&(tenant, label, property.to_string()))
        {
            return Ok(Vec::new());
        }
        // The raw B+tree candidate slots the test seeded (stale / dup /
        // invisible ids included). An absent key ⇒ no candidates ⇒ empty.
        let key = (tenant, label, property.to_string(), stub_value_key(value));
        let candidates = match self.property_index_candidates.get(&key) {
            Some(c) => c,
            None => return Ok(Vec::new()),
        };
        // Candidate-then-verify + dedup by NodeId: hydrate each candidate
        // through the SAME snapshot the scan path uses
        // (`node_by_id_with_context` — which honors tombstones + property
        // overrides), re-check label + property equality, drop failures,
        // and emit each surviving NodeId exactly once.
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for &cand in candidates {
            if !seen.insert(cand) {
                // Duplicate candidate slot already emitted → one row.
                continue;
            }
            let Some(bn) = self.node_by_id_with_context(ctx, cand)? else {
                // Stale / tombstoned / invisible candidate → dropped.
                continue;
            };
            if stub_node_matches(&bn.node, label, property, value) {
                out.push(bn);
            }
        }
        Ok(out)
    }

    /// **#1366 (Phase 2) — the op's index-vs-scan-fallback gate (#1415).**
    /// Mirror production `canonical_key_for`'s keyable partition EXACTLY:
    /// a value is index-keyable iff `stub_value_key` returns a per-value
    /// key (not the unkeyable sentinel). The stub keys keyable values
    /// deterministically and every unkeyable value to one unfindable
    /// sentinel — so `value_is_indexable == (key != sentinel)`.
    fn value_is_indexable(&self, value: &Value) -> bool {
        // Keep in lockstep with `stub_value_key`'s partition; the sentinel
        // is the only string it emits for an unkeyable value.
        stub_value_key(value) != "unkeyable"
    }

    fn count_store(
        &self,
        tenant: TenantId,
        source: CountStoreSource,
    ) -> Result<u64, SubstrateAccessError> {
        if let Some(count) = self.count_store_totals.get(&(tenant, source)) {
            return Ok(*count);
        }
        match source {
            CountStoreSource::Nodes => Ok(self.scan_nodes(tenant, None, Lsn::MAX)?.len() as u64),
            // F1 (#1356 §F1): the in-memory stub holds no `CatalogStats`
            // handle, so a seeded total (above) OR a correct filtered scan
            // serves the per-label count. The O(1) win is the production
            // catalog path (`CrudExecutorSubstrate::count_store`); the stub
            // only owes a CORRECT answer.
            CountStoreSource::NodesWithLabel(label) => {
                Ok(self.scan_nodes(tenant, Some(label), Lsn::MAX)?.len() as u64)
            }
            CountStoreSource::Relationships => {
                let prebuilt = self.edges.get(&tenant).map(|m| m.len()).unwrap_or(0);
                let created = self
                    .create_state
                    .lock()
                    .ok()
                    .and_then(|state| state.edges.get(&tenant).map(Vec::len))
                    .unwrap_or(0);
                Ok((prebuilt + created) as u64)
            }
            // F1: mirror the `Relationships` fallback but filter to the
            // requested rel-type. `RelView::rel_type` is an
            // `Option<TypeId>`; a bare (type-free) edge never matches a
            // typed count.
            CountStoreSource::RelsWithType(rel_type) => {
                let prebuilt = self
                    .edges
                    .get(&tenant)
                    .map(|m| m.values().filter(|e| e.rel_type == Some(rel_type)).count())
                    .unwrap_or(0);
                let created = self
                    .create_state
                    .lock()
                    .ok()
                    .and_then(|state| {
                        state
                            .edges
                            .get(&tenant)
                            .map(|v| v.iter().filter(|e| e.rel_type == Some(rel_type)).count())
                    })
                    .unwrap_or(0);
                Ok((prebuilt + created) as u64)
            }
        }
    }

    fn scan_nodes(
        &self,
        tenant: TenantId,
        label: Option<LabelId>,
        _read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        // ADR-149 W26-θ Phase 3: snapshot the tombstone set so we
        // can filter pre-built nodes that an in-flight DELETE has
        // tombstoned.
        //
        // ADR-152 W27-α §D-3: also snapshot the per-(tenant, NodeId)
        // node_properties sidecar so post-SET / post-REMOVE bags
        // override the NodeView's pre-built / CREATE-time bag.
        let (tombstoned, prop_overrides): (
            std::collections::HashSet<NodeId>,
            HashMap<NodeId, HashMap<String, Value>>,
        ) = {
            let state_guard = self.create_state.lock().ok();
            match state_guard {
                Some(state) => {
                    let tomb = state
                        .tombstoned_nodes
                        .get(&tenant)
                        .cloned()
                        .unwrap_or_default();
                    let mut overrides = HashMap::new();
                    for ((t, n), bag) in &state.node_properties {
                        if *t == tenant {
                            overrides.insert(*n, bag.clone());
                        }
                    }
                    (tomb, overrides)
                }
                None => (std::collections::HashSet::new(), HashMap::new()),
            }
        };
        let mut out: Vec<BoundNode> = Vec::new();
        // Pre-built (fluent-builder) nodes — filter out tombstones.
        if let Some(store) = self.nodes.get(&tenant) {
            out.extend(
                store
                    .values()
                    .filter(|n| !tombstoned.contains(&n.id))
                    .filter(|n| match label {
                        Some(l) => n.label == Some(l),
                        None => true,
                    })
                    .cloned()
                    .map(|mut n| {
                        // ADR-152 §D-3 — merge post-SET bag overrides.
                        if let Some(bag) = prop_overrides.get(&n.id) {
                            n.properties.clear();
                            for (k, v) in bag {
                                n.properties.insert(k.clone(), v.clone());
                            }
                        }
                        BoundNode { node: n }
                    }),
            );
        }
        // ADR-147 W26-θ Phase 1: union CREATE-introduced nodes so a
        // `CREATE → MATCH` test on the same stub observes the
        // freshly-allocated nodes.
        if let Ok(state) = self.create_state.lock() {
            if let Some(created) = state.nodes.get(&tenant) {
                out.extend(
                    created
                        .iter()
                        // CREATE-state nodes are already pruned by
                        // `delete_node` removing them from the vec;
                        // the tombstone filter is the secondary
                        // defense against re-adds.
                        .filter(|n| !tombstoned.contains(&n.id))
                        .filter(|n| match label {
                            Some(l) => n.label == Some(l),
                            None => true,
                        })
                        .cloned()
                        .map(|mut n| {
                            // ADR-152 §D-3 — merge post-SET overrides
                            // for CREATE-introduced nodes. The
                            // CREATE-time bag is already on the
                            // NodeView (per `create_node` ADR-152
                            // §D-1 wire); the override REPLACES it
                            // when SET / REMOVE has subsequently
                            // touched the node.
                            if let Some(bag) = prop_overrides.get(&n.id) {
                                n.properties.clear();
                                for (k, v) in bag {
                                    n.properties.insert(k.clone(), v.clone());
                                }
                            }
                            BoundNode { node: n }
                        }),
                );
            }
        }
        // Deterministic traversal order: ascending by NodeId. Tests
        // assert specific row orderings; HashMap iteration order is
        // randomized so we sort.
        out.sort_by_key(|b| b.node.id.raw());
        Ok(out)
    }

    // ADR-148 W26-θ Phase 2: in-memory CREATE-rel bookkeeping.
    //
    // Allocates a fresh RelId (above 2^32 boundary), interns the
    // rel-type-name via the in-stub rel-type table (above 1024
    // boundary), appends to the CreateState's per-tenant created-
    // edges vec + adjacency, and returns the new id. Properties
    // are IGNORED at v1.0-α per ADR-147 §"Forward-deferred" inherited.
    #[allow(clippy::too_many_arguments)] // ADR-197 _ctx param; see trait def
    fn create_rel(
        &self,
        tenant: TenantId,
        source: NodeId,
        target: NodeId,
        label: &str,
        properties: &[(String, Value)],
        _ctx: &ExecutionContext,
    ) -> Result<RelId, SubstrateAccessError> {
        let mut state = self
            .create_state
            .lock()
            .map_err(|e| SubstrateAccessError::Io(format!("stub create_rel lock poisoned: {e}")))?;
        // Allocate RelId above 2^32 boundary so test fixtures' caller-
        // supplied ids stay disjoint.
        let raw = state.next_rel.fetch_add(1, Ordering::SeqCst);
        let raw = if raw == 0 { (1u64 << 32) + 1 } else { raw };
        if raw == (1u64 << 32) + 1 {
            state.next_rel.store((1u64 << 32) + 2, Ordering::SeqCst);
        }
        let rel_id = RelId::new(raw);
        // Intern rel-type name.
        let key = (tenant, label.to_string());
        let type_id = if let Some(id) = state.rel_types.get(&key) {
            *id
        } else {
            let raw = state.next_rel_type.fetch_add(1, Ordering::SeqCst);
            let raw = if raw < 1024 { 1024 } else { raw };
            if raw == 1024 {
                state.next_rel_type.store(1025, Ordering::SeqCst);
            }
            let id = TypeId::new(raw as u32);
            state.rel_types.insert(key, id);
            id
        };
        // ADR-152 §D-1 — persist the rel property bag in the stub's
        // `rel_properties` sidecar AND populate the RelView so
        // `expand` returns the bag.
        let mut rel = RelView::new(rel_id, source, target, Some(type_id));
        // #871 — carry the rel-type NAME on the stub-stored view (the
        // create call supplies it verbatim), mirroring production
        // `expand` which reverse-resolves it via the intern table.
        rel.rel_type_name = Some(label.to_string());
        if !properties.is_empty() {
            let mut bag = std::collections::HashMap::with_capacity(properties.len());
            for (k, v) in properties {
                rel.properties.insert(k.clone(), v.clone());
                bag.insert(k.clone(), v.clone());
            }
            state.rel_properties.insert((tenant, rel_id), bag);
        }
        state.edges.entry(tenant).or_default().push(rel);
        // Outbound entry from source.
        state
            .adjacency
            .entry(tenant)
            .or_default()
            .entry(source)
            .or_default()
            .push(AdjacencyEntry {
                rel_id,
                dst: target,
                direction: Direction::LeftToRight,
            });
        // Inbound entry from target's perspective.
        state
            .adjacency
            .entry(tenant)
            .or_default()
            .entry(target)
            .or_default()
            .push(AdjacencyEntry {
                rel_id,
                dst: source,
                direction: Direction::RightToLeft,
            });
        Ok(rel_id)
    }

    // ADR-147 W26-θ Phase 1: in-memory CREATE bookkeeping.
    //
    // Allocates a fresh NodeId (above the 2^32 boundary so caller-
    // supplied test fixtures don't collide), interns the label name
    // via the in-stub label table (also allocating above 1024 so
    // caller-supplied LabelIds don't collide), appends to the
    // CreateState's per-tenant created-nodes vec, and returns the
    // new id. Properties are IGNORED at v1.0-α per ADR-147 §"Forward-
    // deferred" → property-bag strict-schema typing — the property
    // slice's shape is preserved for the v1.2 wire-through.
    fn create_node(
        &self,
        tenant: TenantId,
        label: Option<&str>,
        properties: &[(String, Value)],
        _ctx: &ExecutionContext,
    ) -> Result<NodeId, SubstrateAccessError> {
        let mut state = self.create_state.lock().map_err(|e| {
            SubstrateAccessError::Io(format!("stub create_node lock poisoned: {e}"))
        })?;
        // Allocate the NodeId above the 2^32 boundary so test
        // fixtures' caller-supplied ids stay disjoint. Initial value
        // is 1<<32 on first call; subsequent calls increment.
        let raw = state.next_node.fetch_add(1, Ordering::SeqCst);
        let raw = if raw == 0 { (1u64 << 32) + 1 } else { raw };
        // Bump the counter to the next free slot (defensive against
        // re-entry — the AtomicU64 already advances, but the
        // initial-zero branch needs a one-time hoist).
        if raw == (1u64 << 32) + 1 {
            state.next_node.store((1u64 << 32) + 2, Ordering::SeqCst);
        }
        let node_id = NodeId::new(raw);
        // Intern the label name into the stub's per-tenant label
        // table. Re-using a previously-seen name returns the same id;
        // a fresh name allocates above the 1024 boundary.
        let label_id = label.map(|name| {
            let key = (tenant, name.to_string());
            if let Some(id) = state.labels.get(&key) {
                return *id;
            }
            let raw = state.next_label.fetch_add(1, Ordering::SeqCst);
            let raw = if raw < 1024 { 1024 } else { raw };
            if raw == 1024 {
                state.next_label.store(1025, Ordering::SeqCst);
            }
            let id = LabelId::new(raw as u32);
            state.labels.insert(key, id);
            id
        });
        // ADR-152 §D-1 — persist the literal property bag in the
        // stub's `node_properties` sidecar AND populate the
        // NodeView so scan_nodes returns the bag.
        let mut node = NodeView::new(node_id, label_id);
        // #871 — carry the label NAME on the stub-stored view so a
        // `CREATE → MATCH … RETURN labels(n)` / `RETURN n` exercise on
        // the stub mirrors production (where `scan`/`expand` reverse-
        // resolve the name via the intern table). The stub holds the
        // verbatim name from the create call.
        node.label_name = label.map(str::to_string);
        if !properties.is_empty() {
            let mut bag = std::collections::HashMap::with_capacity(properties.len());
            for (k, v) in properties {
                node.properties.insert(k.clone(), v.clone());
                bag.insert(k.clone(), v.clone());
            }
            state.node_properties.insert((tenant, node_id), bag);
        }
        state.nodes.entry(tenant).or_default().push(node);
        Ok(node_id)
    }

    fn expand(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        _read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        // ADR-149 W26-θ Phase 3: snapshot tombstoned rel set so we
        // filter pre-built rels that an in-flight DELETE removed.
        //
        // ADR-152 W27-α §D-3: also snapshot the per-(tenant, NodeId|RelId)
        // property bag sidecars so post-SET / post-REMOVE bags override
        // the pre-built / CREATE-time bags on the emitted views.
        let overrides = ExpandOverrideSnapshot::capture(self, tenant);
        let tombstoned_rels = overrides.tombstoned_rels;
        let node_overrides = overrides.node_overrides;
        let rel_overrides = overrides.rel_overrides;
        let apply_node_override = |mut n: NodeView| -> NodeView {
            if let Some(bag) = node_overrides.get(&n.id) {
                n.properties.clear();
                for (k, v) in bag {
                    n.properties.insert(k.clone(), v.clone());
                }
            }
            n
        };
        let apply_rel_override = |mut r: RelView| -> RelView {
            if let Some(bag) = rel_overrides.get(&r.id) {
                r.properties.clear();
                for (k, v) in bag {
                    r.properties.insert(k.clone(), v.clone());
                }
            }
            r
        };
        let mut out: Vec<BoundEdge> = Vec::new();
        // Pre-built (fluent-builder) adjacency.
        if let Some(adj) = self.adjacency.get(&tenant).and_then(|m| m.get(&from)) {
            let edges = self.edges.get(&tenant);
            let nodes = self.nodes.get(&tenant);
            for entry in adj {
                if tombstoned_rels.contains(&entry.rel_id) {
                    continue;
                }
                let dir_match =
                    matches!(direction, Direction::Undirected) || direction == entry.direction;
                if !dir_match {
                    continue;
                }
                let rel = match edges.and_then(|m| m.get(&entry.rel_id)) {
                    Some(r) => apply_rel_override(r.clone()),
                    None => continue,
                };
                if let Some(ty) = rel_type {
                    if rel.rel_type != Some(ty) {
                        continue;
                    }
                }
                let dst = match nodes.and_then(|m| m.get(&entry.dst)) {
                    Some(n) => apply_node_override(n.clone()),
                    None => continue,
                };
                out.push(BoundEdge { rel, dst });
            }
        }
        // ADR-148 W26-θ Phase 2 — union CREATE-introduced adjacency
        // so a `CREATE-rel → MATCH` test on the same stub observes
        // the freshly-CREATEd rel.
        if let Ok(state) = self.create_state.lock() {
            if let Some(adj) = state.adjacency.get(&tenant).and_then(|m| m.get(&from)) {
                let created_edges = state.edges.get(&tenant);
                let created_nodes = state.nodes.get(&tenant);
                for entry in adj {
                    let dir_match =
                        matches!(direction, Direction::Undirected) || direction == entry.direction;
                    if !dir_match {
                        continue;
                    }
                    // Find the RelView from the CreateState's edge
                    // store first; fall back to the pre-built store.
                    let rel = created_edges
                        .and_then(|edges| edges.iter().find(|r| r.id == entry.rel_id).cloned())
                        .or_else(|| {
                            self.edges
                                .get(&tenant)
                                .and_then(|m| m.get(&entry.rel_id))
                                .cloned()
                        });
                    let Some(rel) = rel else {
                        continue;
                    };
                    let rel = apply_rel_override(rel);
                    if let Some(ty) = rel_type {
                        if rel.rel_type != Some(ty) {
                            continue;
                        }
                    }
                    // Find the destination node from CreateState first
                    // (newly-CREATEd) or pre-built nodes (fixture).
                    let dst = created_nodes
                        .and_then(|nodes| nodes.iter().find(|n| n.id == entry.dst).cloned())
                        .or_else(|| {
                            self.nodes
                                .get(&tenant)
                                .and_then(|m| m.get(&entry.dst))
                                .cloned()
                        });
                    let Some(dst) = dst else {
                        continue;
                    };
                    let dst = apply_node_override(dst);
                    out.push(BoundEdge { rel, dst });
                }
            }
        }
        // Deterministic order by rel-id ascending.
        out.sort_by_key(|e| e.rel.id.raw());
        Ok(out)
    }

    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        _read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        if !self.has_vector {
            return Err(SubstrateAccessError::IndexUnavailable("vector".into()));
        }
        let tag = Self::vector_search_tag_for(query_vec);
        let key = (tenant, format!("{}@{}", property, tag));
        let hits = self.vector_hits.get(&key).cloned().unwrap_or_default();
        Ok(hits.into_iter().take(k as usize).collect())
    }

    fn bm25_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_text: &str,
        k: u64,
        _read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        if !self.has_bm25 {
            return Err(SubstrateAccessError::IndexUnavailable("bm25".into()));
        }
        let key = (tenant, property.to_owned(), query_text.to_owned());
        let hits = self.bm25_hits.get(&key).cloned().unwrap_or_default();
        Ok(hits.into_iter().take(k as usize).collect())
    }

    fn community_members(
        &self,
        tenant: TenantId,
        community_id: i64,
        _read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        if !self.has_community {
            return Err(SubstrateAccessError::IndexUnavailable("community".into()));
        }
        let key = (tenant, community_id);
        let members = self
            .community_members
            .get(&key)
            .cloned()
            .unwrap_or_default();
        Ok(members.into_iter().map(|n| BoundNode { node: n }).collect())
    }

    fn has_vector_substrate(&self) -> bool {
        self.has_vector
    }

    fn has_bm25_substrate(&self) -> bool {
        self.has_bm25
    }

    fn has_community_substrate(&self) -> bool {
        self.has_community
    }

    // ADR-149 W26-θ Phase 3: in-memory DELETE bookkeeping for nodes.
    //
    // Walks BOTH the pre-built `nodes` store + the CreateState's
    // per-tenant `nodes` vec; removes the matching entry from
    // whichever one contains it. When `detach = true`, also walks the
    // pre-built adjacency + CreateState adjacency and tombstones each
    // attached rel BEFORE removing the node. When `detach = false`
    // AND the node has any attached rels, returns an Io error with
    // the "relationships attached" message (the executor maps this
    // to ExecutionError::Substrate per ADR-149 §D-7).
    fn delete_node(
        &self,
        tenant: TenantId,
        node: NodeId,
        detach: bool,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        // Collect rels attached to this node by walking BOTH the
        // pre-built and CreateState adjacency tables.
        let pre_attached: Vec<RelId> = self
            .adjacency
            .get(&tenant)
            .and_then(|m| m.get(&node))
            .map(|adj| adj.iter().map(|e| e.rel_id).collect())
            .unwrap_or_default();
        let state_attached: Vec<RelId> = self
            .create_state
            .lock()
            .ok()
            .and_then(|state| {
                state
                    .adjacency
                    .get(&tenant)
                    .and_then(|m| m.get(&node))
                    .map(|adj| adj.iter().map(|e| e.rel_id).collect())
            })
            .unwrap_or_default();
        let mut all_attached: Vec<RelId> = pre_attached;
        for r in state_attached {
            if !all_attached.contains(&r) {
                all_attached.push(r);
            }
        }
        if !all_attached.is_empty() {
            if !detach {
                return Err(SubstrateAccessError::Io(
                    "delete_node: node has relationships attached; use DETACH DELETE".into(),
                ));
            }
            // DETACH=true: tombstone each attached rel first.
            for rel_id in &all_attached {
                self.delete_rel(tenant, *rel_id, ctx)?;
            }
        }
        // Now remove the node itself. The stub's pre-built `nodes`
        // field is read via `&self` so we cannot mutate it through an
        // immutable receiver; the CreateState bookkeeping is interior-
        // mutable via Mutex. We remove from CreateState; pre-built
        // tests should set up the node via `with_node` (we cannot
        // remove from that path since `nodes` is owned by the stub
        // struct directly). To support pre-baked-node delete the
        // stub state tracks a separate per-tenant "deleted" set; we
        // filter at scan_nodes time.
        let mut state = self.create_state.lock().map_err(|e| {
            SubstrateAccessError::Io(format!("stub delete_node lock poisoned: {e}"))
        })?;
        // Remove from CreateState's per-tenant `nodes` if present.
        if let Some(v) = state.nodes.get_mut(&tenant) {
            v.retain(|n| n.id != node);
        }
        // Record the tombstone for the pre-built `nodes` filter.
        state
            .tombstoned_nodes
            .entry(tenant)
            .or_default()
            .insert(node);
        Ok(())
    }

    // ADR-149 W26-θ Phase 3: in-memory DELETE bookkeeping for rels.
    //
    // Removes the matching rel from the CreateState's per-tenant
    // edges vec + adjacency; records the tombstone in
    // `tombstoned_rels` so the pre-built `edges` store's contribution
    // to `expand` is filtered.
    fn delete_rel(
        &self,
        tenant: TenantId,
        rel: RelId,
        _ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        let mut state = self
            .create_state
            .lock()
            .map_err(|e| SubstrateAccessError::Io(format!("stub delete_rel lock poisoned: {e}")))?;
        // Remove from CreateState's per-tenant edges + adjacency.
        if let Some(v) = state.edges.get_mut(&tenant) {
            v.retain(|r| r.id != rel);
        }
        if let Some(adj_map) = state.adjacency.get_mut(&tenant) {
            for entries in adj_map.values_mut() {
                entries.retain(|e| e.rel_id != rel);
            }
        }
        state.tombstoned_rels.entry(tenant).or_default().insert(rel);
        Ok(())
    }

    // ADR-150 W26-θ Phase 4 — in-memory SET-node bookkeeping.
    //
    // PropertyAssign / PropertyReplace / PropertyMerge update the
    // per-(tenant, node) property bag in the CreateState sidecar:
    // - Assign overwrites the single key;
    // - Replace clears the existing bag and inserts the new entries;
    // - Merge keeps existing entries that aren't in the new map and
    //   overwrites entries that are.
    //
    // LabelAdd appends the new labels to the per-(tenant, node)
    // additional-labels sidecar (the primary NodeView label is
    // preserved unchanged at v1.0-α; the sidecar tracks the additive
    // labels per ADR-150 §D-9 forward-pin to v1.1 multi-label
    // NodeView).
    fn set_node(
        &self,
        tenant: TenantId,
        node: NodeId,
        mutation: &SetNodeMutation,
        _ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        let mut state = self
            .create_state
            .lock()
            .map_err(|e| SubstrateAccessError::Io(format!("stub set_node lock poisoned: {e}")))?;
        match mutation {
            SetNodeMutation::PropertyAssign { name, value } => {
                state
                    .node_properties
                    .entry((tenant, node))
                    .or_default()
                    .insert(name.clone(), value.clone());
            }
            SetNodeMutation::PropertyReplace(entries) => {
                let bag = state.node_properties.entry((tenant, node)).or_default();
                bag.clear();
                for (k, v) in entries {
                    bag.insert(k.clone(), v.clone());
                }
            }
            SetNodeMutation::PropertyMerge(entries) => {
                let bag = state.node_properties.entry((tenant, node)).or_default();
                for (k, v) in entries {
                    bag.insert(k.clone(), v.clone());
                }
            }
            SetNodeMutation::LabelAdd(labels) => {
                let sidecar = state.additional_labels.entry((tenant, node)).or_default();
                for l in labels {
                    if !sidecar.contains(l) {
                        sidecar.push(l.clone());
                    }
                }
            }
        }
        Ok(())
    }

    // ADR-150 W26-θ Phase 4 — in-memory SET-rel bookkeeping.
    fn set_rel(
        &self,
        tenant: TenantId,
        rel: RelId,
        mutation: &SetRelMutation,
        _ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        let mut state = self
            .create_state
            .lock()
            .map_err(|e| SubstrateAccessError::Io(format!("stub set_rel lock poisoned: {e}")))?;
        match mutation {
            SetRelMutation::PropertyAssign { name, value } => {
                state
                    .rel_properties
                    .entry((tenant, rel))
                    .or_default()
                    .insert(name.clone(), value.clone());
            }
            SetRelMutation::PropertyReplace(entries) => {
                let bag = state.rel_properties.entry((tenant, rel)).or_default();
                bag.clear();
                for (k, v) in entries {
                    bag.insert(k.clone(), v.clone());
                }
            }
            SetRelMutation::PropertyMerge(entries) => {
                let bag = state.rel_properties.entry((tenant, rel)).or_default();
                for (k, v) in entries {
                    bag.insert(k.clone(), v.clone());
                }
            }
        }
        Ok(())
    }

    // ADR-150 W26-θ Phase 4 — in-memory REMOVE-node bookkeeping.
    fn remove_node(
        &self,
        tenant: TenantId,
        node: NodeId,
        mutation: &RemoveNodeMutation,
        _ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        let mut state = self.create_state.lock().map_err(|e| {
            SubstrateAccessError::Io(format!("stub remove_node lock poisoned: {e}"))
        })?;
        match mutation {
            RemoveNodeMutation::Property(name) => {
                if let Some(bag) = state.node_properties.get_mut(&(tenant, node)) {
                    bag.remove(name);
                }
            }
            RemoveNodeMutation::LabelRemove(labels) => {
                if let Some(sidecar) = state.additional_labels.get_mut(&(tenant, node)) {
                    sidecar.retain(|l| !labels.contains(l));
                }
            }
        }
        Ok(())
    }

    // ADR-150 W26-θ Phase 4 — in-memory REMOVE-rel bookkeeping.
    fn remove_rel(
        &self,
        tenant: TenantId,
        rel: RelId,
        mutation: &RemoveRelMutation,
        _ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        let mut state = self
            .create_state
            .lock()
            .map_err(|e| SubstrateAccessError::Io(format!("stub remove_rel lock poisoned: {e}")))?;
        match mutation {
            RemoveRelMutation::Property(name) => {
                if let Some(bag) = state.rel_properties.get_mut(&(tenant, rel)) {
                    bag.remove(name);
                }
            }
        }
        Ok(())
    }

    // #830 / ADR-200 — in-memory vector-index catalog bookkeeping.
    fn register_vector_index(
        &self,
        tenant: TenantId,
        entry: VectorIndexCatalogEntry,
        if_not_exists: bool,
    ) -> Result<VectorIndexRegistration, SubstrateAccessError> {
        let mut state = self.create_state.lock().map_err(|e| {
            SubstrateAccessError::Io(format!("stub register_vector_index lock poisoned: {e}"))
        })?;
        let bucket = state.vector_indexes.entry(tenant).or_default();
        if bucket.iter().any(|e| e.name == entry.name) {
            if if_not_exists {
                return Ok(VectorIndexRegistration::AlreadyExists);
            }
            return Err(SubstrateAccessError::IndexAlreadyExists { name: entry.name });
        }
        bucket.push(entry);
        Ok(VectorIndexRegistration::Created)
    }

    fn list_vector_indexes(&self, tenant: TenantId) -> Vec<VectorIndexCatalogEntry> {
        self.create_state
            .lock()
            .ok()
            .and_then(|state| state.vector_indexes.get(&tenant).cloned())
            .unwrap_or_default()
    }

    // #1366 (task #248) — in-memory property-index registration. The
    // stub backfill is a no-op (no fixture nodes); this exercises the
    // register + IF NOT EXISTS idempotency contract.
    fn create_property_index(
        &self,
        tenant: TenantId,
        name: &str,
        if_not_exists: bool,
        _label: &str,
        _property: &str,
    ) -> Result<PropertyIndexRegistration, SubstrateAccessError> {
        let mut state = self.create_state.lock().map_err(|e| {
            SubstrateAccessError::Io(format!("stub create_property_index lock poisoned: {e}"))
        })?;
        let bucket = state.property_indexes.entry(tenant).or_default();
        if bucket.iter().any(|n| n == name) {
            if if_not_exists {
                return Ok(PropertyIndexRegistration::AlreadyExists);
            }
            return Err(SubstrateAccessError::IndexAlreadyExists {
                name: name.to_string(),
            });
        }
        bucket.push(name.to_string());
        Ok(PropertyIndexRegistration::Created)
    }
}

// Compatibility helper: tests importing `from_value_strict` etc.
// reach for a `Value` constructor. Keep `Value` re-export under
// `crate::executor::value::Value` and let clippy + tests see it
// through the prelude.
#[allow(dead_code)]
fn _value_use_pin(_v: &Value) {
    // Compile-time pin: `Value` must remain in scope for the trait
    // signatures (BoundNode / BoundEdge embed it via NodeView /
    // RelView).
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── ADR-197-amendment-01 D-4 — loud defaults ──────────────

    /// Fake held-txn handle for default-behavior pins.
    #[derive(Debug)]
    struct FakeHeld;
    impl HeldTxnHandle for FakeHeld {
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn snapshot_lsn(&self) -> Lsn {
            Lsn::new(7)
        }
    }

    /// Minimal substrate that implements ONLY the required reads and
    /// does NOT override the `_with_context` variants — the class the
    /// D-4 loud default protects against.
    #[derive(Debug)]
    struct NoTxnReadsSubstrate;
    impl ExecutorSubstrate for NoTxnReadsSubstrate {
        fn scan_nodes(
            &self,
            _tenant: TenantId,
            _label: Option<LabelId>,
            _read_lsn: Lsn,
        ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
            Ok(Vec::new())
        }
        fn expand(
            &self,
            _tenant: TenantId,
            _from: NodeId,
            _rel_type: Option<TypeId>,
            _direction: Direction,
            _read_lsn: Lsn,
        ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
            Ok(Vec::new())
        }
        fn vector_search(
            &self,
            _tenant: TenantId,
            _property: &str,
            _query_vec: &[f32],
            _k: u64,
            _read_lsn: Lsn,
        ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
            Err(SubstrateAccessError::IndexUnavailable("vector".into()))
        }
        fn bm25_search(
            &self,
            _tenant: TenantId,
            _property: &str,
            _query_text: &str,
            _k: u64,
            _read_lsn: Lsn,
        ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
            Err(SubstrateAccessError::IndexUnavailable("bm25".into()))
        }
        fn community_members(
            &self,
            _tenant: TenantId,
            _community_id: i64,
            _read_lsn: Lsn,
        ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
            Err(SubstrateAccessError::IndexUnavailable("community".into()))
        }
    }

    /// D-4 pin: inside an explicit transaction, a substrate without
    /// held-txn read support fails LOUD (typed error) — never silently
    /// serves committed-read isolation (the #822 bug shape).
    #[test]
    fn default_context_reads_fail_loud_inside_held_txn() {
        let s = NoTxnReadsSubstrate;
        let ctx = crate::executor::ExecutionContext::new(
            TenantId::DEFAULT,
            arcgraph_core::PartitionId::ZERO,
        )
        .with_held_txn(Box::new(FakeHeld));

        let scan = s.scan_nodes_with_context(&ctx, None, Lsn::MAX);
        assert!(
            matches!(scan, Err(SubstrateAccessError::HeldTxnReadsUnsupported(ref w)) if w == "scan_nodes"),
            "scan default must fail loud in explicit mode; got {scan:?}"
        );
        let exp =
            s.expand_with_context(&ctx, NodeId::new(1), None, Direction::LeftToRight, Lsn::MAX);
        assert!(
            matches!(exp, Err(SubstrateAccessError::HeldTxnReadsUnsupported(ref w)) if w == "expand"),
            "expand default must fail loud in explicit mode; got {exp:?}"
        );
    }

    /// D-4 counter-pin: WITHOUT a held transaction the defaults
    /// delegate to the plain committed reads (auto-commit unchanged).
    #[test]
    fn default_context_reads_delegate_without_held_txn() {
        let s = NoTxnReadsSubstrate;
        let ctx = crate::executor::ExecutionContext::new(
            TenantId::DEFAULT,
            arcgraph_core::PartitionId::ZERO,
        );
        assert!(s.scan_nodes_with_context(&ctx, None, Lsn::MAX).is_ok());
        assert!(
            s.expand_with_context(&ctx, NodeId::new(1), None, Direction::LeftToRight, Lsn::MAX)
                .is_ok()
        );
    }

    /// D-4 stub pin: the stub's CONSCIOUS overrides keep working under
    /// a held transaction (for the stub, plain delegation is exact —
    /// its writes are immediately visible, so committed-read ≡ RYW).
    #[test]
    fn stub_overrides_serve_reads_under_held_txn() {
        let stub = StubExecutorSubstrate::new().with_node(
            TenantId::DEFAULT,
            crate::executor::value::NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
        );
        let ctx = crate::executor::ExecutionContext::new(
            TenantId::DEFAULT,
            arcgraph_core::PartitionId::ZERO,
        )
        .with_held_txn(Box::new(FakeHeld));
        let nodes = stub
            .scan_nodes_with_context(&ctx, None, Lsn::MAX)
            .expect("stub override serves reads under a held txn");
        assert_eq!(nodes.len(), 1);
    }

    fn _assert_dyn_executor_substrate(_: &dyn ExecutorSubstrate) {}

    #[test]
    fn default_expand_cursor_materializes_and_trait_remains_dyn_safe() {
        #[derive(Debug)]
        struct MinimalCursorSubstrate;
        impl ExecutorSubstrate for MinimalCursorSubstrate {
            fn scan_nodes(
                &self,
                _tenant: TenantId,
                _label: Option<LabelId>,
                _read_lsn: Lsn,
            ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
                Ok(Vec::new())
            }

            fn expand(
                &self,
                _tenant: TenantId,
                from: NodeId,
                rel_type: Option<TypeId>,
                _direction: Direction,
                _read_lsn: Lsn,
            ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
                Ok(vec![BoundEdge {
                    rel: RelView::new(RelId::new(99), from, NodeId::new(2), rel_type),
                    dst: NodeView::new(NodeId::new(2), Some(LabelId::new(1))),
                }])
            }

            fn vector_search(
                &self,
                _tenant: TenantId,
                _property: &str,
                _query_vec: &[f32],
                _k: u64,
                _read_lsn: Lsn,
            ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
                Err(SubstrateAccessError::IndexUnavailable("vector".into()))
            }

            fn bm25_search(
                &self,
                _tenant: TenantId,
                _property: &str,
                _query_text: &str,
                _k: u64,
                _read_lsn: Lsn,
            ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
                Err(SubstrateAccessError::IndexUnavailable("bm25".into()))
            }

            fn community_members(
                &self,
                _tenant: TenantId,
                _community_id: i64,
                _read_lsn: Lsn,
            ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
                Err(SubstrateAccessError::IndexUnavailable("community".into()))
            }
        }

        let s = MinimalCursorSubstrate;
        _assert_dyn_executor_substrate(&s);
        let rows: Vec<BoundEdge> = s
            .expand_cursor(
                TenantId::DEFAULT,
                NodeId::new(1),
                Some(TypeId::new(7)),
                Direction::LeftToRight,
                Lsn::MAX,
            )
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].rel.id, RelId::new(99));
        assert_eq!(rows[0].dst.id, NodeId::new(2));
    }

    fn alice() -> NodeView {
        NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
            .with_property("name", Value::String("Alice".into()))
    }

    fn bob() -> NodeView {
        NodeView::new(NodeId::new(2), Some(LabelId::new(1)))
            .with_property("name", Value::String("Bob".into()))
    }

    #[test]
    fn scan_nodes_filters_by_label_and_sorts_deterministically() {
        let sub = StubExecutorSubstrate::new()
            .with_node(TenantId::DEFAULT, alice())
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(3), Some(LabelId::new(2))),
            )
            .with_node(TenantId::DEFAULT, bob());
        let l1 = sub
            .scan_nodes(TenantId::DEFAULT, Some(LabelId::new(1)), Lsn::MAX)
            .expect("scan");
        // Two L1 nodes — Alice (id=1), Bob (id=2). Order by id asc.
        assert_eq!(l1.len(), 2);
        assert_eq!(l1[0].node.id, NodeId::new(1));
        assert_eq!(l1[1].node.id, NodeId::new(2));

        // No-label scan returns all 3.
        let all = sub
            .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
            .expect("scan all");
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn scan_nodes_isolates_tenants() {
        let other = TenantId::new(42);
        let sub = StubExecutorSubstrate::new()
            .with_node(TenantId::DEFAULT, alice())
            .with_node(other, bob());
        let def = sub.scan_nodes(TenantId::DEFAULT, None, Lsn::MAX).unwrap();
        let oth = sub.scan_nodes(other, None, Lsn::MAX).unwrap();
        assert_eq!(def.len(), 1);
        assert_eq!(oth.len(), 1);
        assert_eq!(def[0].node.id, NodeId::new(1));
        assert_eq!(oth[0].node.id, NodeId::new(2));
    }

    #[test]
    fn expand_filters_by_rel_type_and_direction() {
        let knows = TypeId::new(1);
        let likes = TypeId::new(2);
        let sub = StubExecutorSubstrate::new()
            .with_node(TenantId::DEFAULT, alice())
            .with_node(TenantId::DEFAULT, bob())
            .with_edge(
                TenantId::DEFAULT,
                RelView::new(RelId::new(10), NodeId::new(1), NodeId::new(2), Some(knows)),
            )
            .with_edge(
                TenantId::DEFAULT,
                RelView::new(RelId::new(11), NodeId::new(1), NodeId::new(2), Some(likes)),
            );
        // From Alice via KNOWS LeftToRight (outbound) → Bob.
        let kr = sub
            .expand(
                TenantId::DEFAULT,
                NodeId::new(1),
                Some(knows),
                Direction::LeftToRight,
                Lsn::MAX,
            )
            .unwrap();
        assert_eq!(kr.len(), 1);
        assert_eq!(kr[0].rel.rel_type, Some(knows));
        assert_eq!(kr[0].dst.id, NodeId::new(2));
        // Undirected admits both stored directions; from Alice we have
        // 1 outbound KNOWS, so still 1 hit (the inbound stamp would
        // require Bob → ... via KNOWS — none).
        let und = sub
            .expand(
                TenantId::DEFAULT,
                NodeId::new(1),
                Some(knows),
                Direction::Undirected,
                Lsn::MAX,
            )
            .unwrap();
        assert_eq!(und.len(), 1);
        // No rel-type filter: 2 outbound from Alice (KNOWS + LIKES).
        let any = sub
            .expand(
                TenantId::DEFAULT,
                NodeId::new(1),
                None,
                Direction::LeftToRight,
                Lsn::MAX,
            )
            .unwrap();
        assert_eq!(any.len(), 2);
    }

    #[test]
    fn vector_search_returns_unavailable_when_substrate_off() {
        let sub = StubExecutorSubstrate::new();
        let r = sub
            .vector_search(TenantId::DEFAULT, "embedding", &[0.0], 5, Lsn::MAX)
            .expect_err("unavailable");
        assert_eq!(r, SubstrateAccessError::IndexUnavailable("vector".into()));
    }

    #[test]
    fn vector_search_round_trips_pre_baked_hits() {
        // Test-only sentinel; using `1.5` (NOT π) so the
        // `clippy::approx_constant` heuristic doesn't fire.
        let qv = [1.5_f32, 0.0, 1.0];
        let tag = StubExecutorSubstrate::vector_search_tag_for(&qv);
        let hits = vec![
            RankedHit {
                node: alice(),
                score: 0.99,
            },
            RankedHit {
                node: bob(),
                score: 0.42,
            },
        ];
        let sub = StubExecutorSubstrate::new()
            .with_vector_substrate()
            .with_vector_hit(TenantId::DEFAULT, "embedding", &tag, hits.clone());
        let r = sub
            .vector_search(TenantId::DEFAULT, "embedding", &qv, 10, Lsn::MAX)
            .unwrap();
        assert_eq!(r, hits);
        // Top-K cap.
        let r1 = sub
            .vector_search(TenantId::DEFAULT, "embedding", &qv, 1, Lsn::MAX)
            .unwrap();
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0], hits[0]);
    }

    #[test]
    fn bm25_search_round_trips_pre_baked_hits() {
        let hits = vec![RankedHit {
            node: alice(),
            score: 7.5,
        }];
        let sub = StubExecutorSubstrate::new()
            .with_bm25_substrate()
            .with_bm25_hit(TenantId::DEFAULT, "content", "alice doc", hits.clone());
        let r = sub
            .bm25_search(TenantId::DEFAULT, "content", "alice doc", 10, Lsn::MAX)
            .unwrap();
        assert_eq!(r, hits);
    }

    #[test]
    fn community_members_round_trips_pre_baked_membership() {
        let sub = StubExecutorSubstrate::new()
            .with_community_substrate()
            .with_community_membership(TenantId::DEFAULT, 7, vec![alice(), bob()]);
        let r = sub
            .community_members(TenantId::DEFAULT, 7, Lsn::MAX)
            .unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].node.id, NodeId::new(1));
        assert_eq!(r[1].node.id, NodeId::new(2));

        // Empty community.
        let nope = sub
            .community_members(TenantId::DEFAULT, 99, Lsn::MAX)
            .unwrap();
        assert!(nope.is_empty());
    }

    #[test]
    fn substrate_flags_compose_independently() {
        let sub = StubExecutorSubstrate::new()
            .with_vector_substrate()
            .with_community_substrate();
        assert!(sub.has_vector_substrate());
        assert!(!sub.has_bm25_substrate());
        assert!(sub.has_community_substrate());
    }
}
