//! Counting decorator wrapping any [`ExecutorSubstrate`] to aggregate
//! write-effect counters per ADR-153 §D-2 (W27-β).
//!
//! The MCP-side raw_query adapter constructs a [`CountingSubstrate`]
//! around the production [`crate::storage::substrate::CrudExecutorSubstrate`]
//! per `graph.raw_query` invocation, runs the executor pipeline against
//! the decorated substrate, then drains the counters into a
//! [`crate::tools::raw_query::WriteSummary`] for the response envelope.
//!
//! # Why decorator (not "modify each op")?
//!
//! The 5 W26-θ executor ops (CreateNode/CreateRel/Delete/Set/Remove/
//! Merge per ADR-147..151) each invoke exactly one substrate write
//! method per logical write. Wrapping the substrate at the trait
//! boundary lets the count flow through every op WITHOUT touching any
//! op or trait shape — the production wiring continues to bind to
//! `&dyn ExecutorSubstrate` and gets accurate counters transparently.
//!
//! # Counter rules (ADR-153 §D-2)
//!
//! - `nodes_created` — increments on every successful `create_node`.
//! - `rels_created` — increments on every successful `create_rel`.
//! - `nodes_deleted` — increments on every successful `delete_node`
//!   (whether `detach = true` or `false`).
//! - `rels_deleted` — increments on every successful `delete_rel`. A
//!   `DETACH DELETE` cascade in the underlying substrate emits N
//!   `delete_rel` calls + 1 `delete_node` call → the cascade surfaces
//!   correctly: `rels_deleted = N`, `nodes_deleted = 1`.
//! - `properties_set` — sums `SetNodeMutation` / `SetRelMutation`
//!   shapes:
//!   - `PropertyAssign{..}` → +1
//!   - `PropertyReplace(entries)` → +entries.len()
//!   - `PropertyMerge(entries)` → +entries.len()
//!
//!   `LabelAdd(labels)` does NOT increment `properties_set`; it
//!   increments `labels_added` instead.
//! - `properties_removed` — sums `RemoveNodeMutation::Property` and
//!   `RemoveRelMutation::Property` → +1 each.
//! - `labels_added` — sums `SetNodeMutation::LabelAdd(labels)` →
//!   +labels.len() per call.
//! - `labels_removed` — sums `RemoveNodeMutation::LabelRemove(labels)`
//!   → +labels.len() per call.
//!
//! Counters increment EVEN WHEN the inner call returns
//! `Err(IndexUnavailable)` for the v1.0-α LabelAdd/LabelRemove path
//! per ADR-150 §D-9? NO — we increment ONLY on `Ok(_)` return per
//! ADR-153 §D-2 ("counters reflect committed effects only"). A
//! substrate that surfaces `IndexUnavailable` for label-add does NOT
//! advance `labels_added`.
//!
//! # Thread-safety
//!
//! The counters are atomic so the substrate stays `Sync` (every
//! `ExecutorSubstrate` impl is `Send + Sync` per the trait bound). At
//! v1.0-α the executor pipeline runs single-threaded per query, but
//! the atomic shape forward-binds the M4-64a parallel-execution work
//! without churn.
//!
//! # ADR provenance
//! - **ADR-153 §D-2** — WriteSummary 8-counter shape + commit-only rule.
//! - **ADR-147..151 §"Counting semantics"** — per-clause counting
//!   rules each W26-θ ADR ratified; ADR-153 composes them.
//! - **ADR-031 + ADR-033** — per-tenant Transaction commit-or-rollback
//!   (counter semantics align with the all-or-nothing guarantee — a
//!   per-call `Err(_)` does NOT advance counters).
//! - **Bounded contexts** — the decorator lives in `arcgraph-mcp::storage`
//!   adjacent to
//!   `CrudExecutorSubstrate`; the inner substrate trait is consumed
//!   verbatim through the existing `arcgraph-query` re-export).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::{LabelId, Lsn, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::executor::substrate::{
    RemoveNodeMutation, RemoveRelMutation, SetNodeMutation, SetRelMutation,
};
use arcgraph_query::executor::value::Value;
use arcgraph_query::executor::{
    BoundEdge, BoundEdgeCursor, BoundNode, ExecutionContext, ExecutorSubstrate, RankedHit,
    SubstrateAccessError,
};
use arcgraph_query::logical_plan::{CountStoreSource, Direction};

use crate::tools::raw_query::WriteSummary;

/// Shared atomic counters consumed by [`CountingSubstrate`] +
/// drained into a [`WriteSummary`] at request-end.
#[derive(Debug, Default)]
pub struct WriteCounters {
    nodes_created: AtomicU64,
    nodes_deleted: AtomicU64,
    rels_created: AtomicU64,
    rels_deleted: AtomicU64,
    properties_set: AtomicU64,
    properties_removed: AtomicU64,
    labels_added: AtomicU64,
    labels_removed: AtomicU64,
}

impl WriteCounters {
    /// Construct a fresh zero-state counter bag.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot the counters into a [`WriteSummary`]. Uses
    /// `Ordering::Relaxed` because the executor pipeline runs
    /// single-threaded per query at v1.0-α (M4-64a parallel-execution
    /// adds a barrier at materialize-end before the snapshot).
    #[must_use]
    pub fn snapshot(&self) -> WriteSummary {
        WriteSummary {
            nodes_created: self.nodes_created.load(Ordering::Relaxed),
            nodes_deleted: self.nodes_deleted.load(Ordering::Relaxed),
            rels_created: self.rels_created.load(Ordering::Relaxed),
            rels_deleted: self.rels_deleted.load(Ordering::Relaxed),
            properties_set: self.properties_set.load(Ordering::Relaxed),
            properties_removed: self.properties_removed.load(Ordering::Relaxed),
            labels_added: self.labels_added.load(Ordering::Relaxed),
            labels_removed: self.labels_removed.load(Ordering::Relaxed),
        }
    }
}

/// [`ExecutorSubstrate`] decorator that counts successful write calls.
///
/// Reads pass through untouched; writes increment the shared
/// [`WriteCounters`] on `Ok(_)` return only. The decorator is
/// `&self`-only so it satisfies the trait's `Send + Sync` bound; the
/// counter Arc is shared with the raw_query adapter that constructed
/// the decorator.
pub struct CountingSubstrate<S: ExecutorSubstrate> {
    inner: S,
    counters: Arc<WriteCounters>,
}

impl<S: ExecutorSubstrate> CountingSubstrate<S> {
    /// Wrap `inner` with a fresh counter bag. Returns the wrapped
    /// substrate + the shared counter Arc (caller reads counters via
    /// the Arc after the executor pipeline drains).
    pub fn new(inner: S) -> (Self, Arc<WriteCounters>) {
        let counters = Arc::new(WriteCounters::new());
        let wrapped = Self {
            inner,
            counters: Arc::clone(&counters),
        };
        (wrapped, counters)
    }

    /// Borrow the inner substrate. Used by tests / debug paths.
    pub fn inner(&self) -> &S {
        &self.inner
    }
}

impl<S: ExecutorSubstrate> ExecutorSubstrate for CountingSubstrate<S> {
    // ─────────────────────────────────────────────────────────────────
    // Read-side pass-through — no counter touches.
    // ─────────────────────────────────────────────────────────────────

    fn scan_nodes(
        &self,
        tenant: TenantId,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        self.inner.scan_nodes(tenant, label, read_lsn)
    }

    fn count_store(
        &self,
        tenant: TenantId,
        source: CountStoreSource,
    ) -> Result<u64, SubstrateAccessError> {
        self.inner.count_store(tenant, source)
    }

    fn expand(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        self.inner
            .expand(tenant, from, rel_type, direction, read_lsn)
    }

    fn node_by_id_with_context(
        &self,
        ctx: &ExecutionContext,
        id: NodeId,
    ) -> Result<Option<BoundNode>, SubstrateAccessError> {
        self.inner.node_by_id_with_context(ctx, id)
    }

    // #1366 (Phase 2) / #1415 — the indexed point-lookup seam + its
    // index-vs-scan-fallback gate MUST forward to the inner production
    // substrate. Without the `property_index_lookup_with_context` forward
    // the default (`Ok(vec![])`) would make EVERY indexed lookup empty
    // through this decorator (defeating the #1366 820× point-lookup fix);
    // without the `value_is_indexable` forward the default (`false`) would
    // force even a keyable value down the op's Scan+Filter fallback
    // (correct, but a silent perf regression). Forwarding both keeps the
    // index fast-path AND the correct-for-unkeyable-values behaviour.

    fn property_index_lookup_with_context(
        &self,
        ctx: &ExecutionContext,
        label: LabelId,
        property: &str,
        value: &Value,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        self.inner
            .property_index_lookup_with_context(ctx, label, property, value, read_lsn)
    }

    fn value_is_indexable(&self, value: &Value) -> bool {
        self.inner.value_is_indexable(value)
    }

    // D-2 — the statement-scoped autocommit txn installs a held txn on
    // `ctx` for a write statement (e.g. `MATCH … CREATE …`), so the
    // transaction-aware read methods MUST route through the inner
    // production substrate's held-txn read path — the DEFAULT trait impls
    // fail LOUD (`HeldTxnReadsUnsupported`) once a held txn is present,
    // which would break read-your-writes for a MATCH-then-write spine run
    // through the raw_query CountingSubstrate. Pre-D-2 this decorator only
    // ever saw the AUTO-COMMIT no-held-txn path, so these forwards were
    // absent; D-2 lights the held-txn read path here.

    fn scan_nodes_with_context(
        &self,
        ctx: &ExecutionContext,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        self.inner.scan_nodes_with_context(ctx, label, read_lsn)
    }

    fn expand_with_context(
        &self,
        ctx: &ExecutionContext,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        self.inner
            .expand_with_context(ctx, from, rel_type, direction, read_lsn)
    }

    fn expand_cursor_with_context(
        &self,
        ctx: &ExecutionContext,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<BoundEdgeCursor, SubstrateAccessError> {
        self.inner
            .expand_cursor_with_context(ctx, from, rel_type, direction, read_lsn)
    }

    // D-2 — statement-scoped autocommit txn lifecycle: forward to the
    // inner production substrate so the ONE-txn-per-statement begin /
    // commit-once / rollback actually opens on the real store (the default
    // trait impls are NO-OPs → the create ops would fall back to per-op
    // auto-commit, defeating D-2's 3-commits→1 + atomicity for the
    // raw_query write path).

    fn begin_statement(&self, ctx: &ExecutionContext) -> Result<(), SubstrateAccessError> {
        self.inner.begin_statement(ctx)
    }

    fn commit_statement(&self, ctx: &ExecutionContext) -> Result<(), SubstrateAccessError> {
        self.inner.commit_statement(ctx)
    }

    fn rollback_statement(&self, ctx: &ExecutionContext) {
        self.inner.rollback_statement(ctx);
    }

    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        self.inner
            .vector_search(tenant, property, query_vec, k, read_lsn)
    }

    fn bm25_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_text: &str,
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        self.inner
            .bm25_search(tenant, property, query_text, k, read_lsn)
    }

    fn community_members(
        &self,
        tenant: TenantId,
        community_id: i64,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        self.inner.community_members(tenant, community_id, read_lsn)
    }

    fn has_vector_substrate(&self) -> bool {
        self.inner.has_vector_substrate()
    }

    fn has_bm25_substrate(&self) -> bool {
        self.inner.has_bm25_substrate()
    }

    fn has_community_substrate(&self) -> bool {
        self.inner.has_community_substrate()
    }

    // ─────────────────────────────────────────────────────────────────
    // Write-side counting — increment ONLY on Ok(_) per ADR-153 §D-2.
    // ─────────────────────────────────────────────────────────────────

    fn create_node(
        &self,
        tenant: TenantId,
        label: Option<&str>,
        properties: &[(String, Value)],
        ctx: &ExecutionContext,
    ) -> Result<NodeId, SubstrateAccessError> {
        let out = self.inner.create_node(tenant, label, properties, ctx);
        if out.is_ok() {
            self.counters.nodes_created.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    fn create_rel(
        &self,
        tenant: TenantId,
        source: NodeId,
        target: NodeId,
        label: &str,
        properties: &[(String, Value)],
        ctx: &ExecutionContext,
    ) -> Result<RelId, SubstrateAccessError> {
        let out = self
            .inner
            .create_rel(tenant, source, target, label, properties, ctx);
        if out.is_ok() {
            self.counters.rels_created.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    fn delete_node(
        &self,
        tenant: TenantId,
        node: NodeId,
        detach: bool,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        let out = self.inner.delete_node(tenant, node, detach, ctx);
        if out.is_ok() {
            self.counters.nodes_deleted.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    fn delete_rel(
        &self,
        tenant: TenantId,
        rel: RelId,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        let out = self.inner.delete_rel(tenant, rel, ctx);
        if out.is_ok() {
            self.counters.rels_deleted.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    fn set_node(
        &self,
        tenant: TenantId,
        node: NodeId,
        mutation: &SetNodeMutation,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        let out = self.inner.set_node(tenant, node, mutation, ctx);
        if out.is_ok() {
            match mutation {
                SetNodeMutation::PropertyAssign { .. } => {
                    self.counters.properties_set.fetch_add(1, Ordering::Relaxed);
                }
                SetNodeMutation::PropertyReplace(entries)
                | SetNodeMutation::PropertyMerge(entries) => {
                    self.counters
                        .properties_set
                        .fetch_add(entries.len() as u64, Ordering::Relaxed);
                }
                SetNodeMutation::LabelAdd(labels) => {
                    self.counters
                        .labels_added
                        .fetch_add(labels.len() as u64, Ordering::Relaxed);
                }
            }
        }
        out
    }

    fn set_rel(
        &self,
        tenant: TenantId,
        rel: RelId,
        mutation: &SetRelMutation,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        let out = self.inner.set_rel(tenant, rel, mutation, ctx);
        if out.is_ok() {
            match mutation {
                SetRelMutation::PropertyAssign { .. } => {
                    self.counters.properties_set.fetch_add(1, Ordering::Relaxed);
                }
                SetRelMutation::PropertyReplace(entries)
                | SetRelMutation::PropertyMerge(entries) => {
                    self.counters
                        .properties_set
                        .fetch_add(entries.len() as u64, Ordering::Relaxed);
                }
            }
        }
        out
    }

    fn remove_node(
        &self,
        tenant: TenantId,
        node: NodeId,
        mutation: &RemoveNodeMutation,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        let out = self.inner.remove_node(tenant, node, mutation, ctx);
        if out.is_ok() {
            match mutation {
                RemoveNodeMutation::Property(_) => {
                    self.counters
                        .properties_removed
                        .fetch_add(1, Ordering::Relaxed);
                }
                RemoveNodeMutation::LabelRemove(labels) => {
                    self.counters
                        .labels_removed
                        .fetch_add(labels.len() as u64, Ordering::Relaxed);
                }
            }
        }
        out
    }

    fn remove_rel(
        &self,
        tenant: TenantId,
        rel: RelId,
        mutation: &RemoveRelMutation,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        let out = self.inner.remove_rel(tenant, rel, mutation, ctx);
        if out.is_ok() {
            match mutation {
                RemoveRelMutation::Property(_) => {
                    self.counters
                        .properties_removed
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        out
    }

    // #830 / ADR-200 — transparent delegation of the vector-index
    // catalog methods (metadata-only; not counted — registering an
    // index is a DDL, not a row write).
    fn register_vector_index(
        &self,
        tenant: TenantId,
        entry: arcgraph_query::executor::substrate::VectorIndexCatalogEntry,
        if_not_exists: bool,
    ) -> Result<arcgraph_query::executor::substrate::VectorIndexRegistration, SubstrateAccessError>
    {
        self.inner
            .register_vector_index(tenant, entry, if_not_exists)
    }

    fn list_vector_indexes(
        &self,
        tenant: TenantId,
    ) -> Vec<arcgraph_query::executor::substrate::VectorIndexCatalogEntry> {
        self.inner.list_vector_indexes(tenant)
    }

    fn resolve_vector_index(
        &self,
        tenant: TenantId,
        name: &str,
    ) -> Option<arcgraph_query::executor::substrate::VectorIndexCatalogEntry> {
        self.inner.resolve_vector_index(tenant, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcgraph_core::PartitionId;
    use arcgraph_query::executor::StubExecutorSubstrate;
    use arcgraph_query::executor::value::NodeView;

    /// ADR-197 — a fresh auto-commit `ExecutionContext` for the
    /// CountingSubstrate forwarding tests (no held tx; the stub inner
    /// ignores `ctx` for its in-memory bookkeeping).
    fn tctx() -> ExecutionContext {
        ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
    }

    #[test]
    fn create_node_increments_nodes_created_counter_on_ok() {
        let inner = StubExecutorSubstrate::new();
        let (sub, counters) = CountingSubstrate::new(inner);
        let _id = sub
            .create_node(TenantId::DEFAULT, Some("User"), &[], &tctx())
            .expect("ok");
        assert_eq!(counters.snapshot().nodes_created, 1);
        assert_eq!(counters.snapshot().total(), 1);
        // Other counters stay zero.
        assert_eq!(counters.snapshot().rels_created, 0);
        assert_eq!(counters.snapshot().nodes_deleted, 0);
    }

    #[test]
    fn create_rel_increments_rels_created_counter_on_ok() {
        let inner = StubExecutorSubstrate::new();
        let (sub, counters) = CountingSubstrate::new(inner);
        let a = sub
            .create_node(TenantId::DEFAULT, Some("U"), &[], &tctx())
            .expect("a");
        let b = sub
            .create_node(TenantId::DEFAULT, Some("U"), &[], &tctx())
            .expect("b");
        let _r = sub
            .create_rel(TenantId::DEFAULT, a, b, "KNOWS", &[], &tctx())
            .expect("rel ok");
        let snap = counters.snapshot();
        assert_eq!(snap.nodes_created, 2);
        assert_eq!(snap.rels_created, 1);
    }

    #[test]
    fn delete_node_and_rel_increment_per_call() {
        let inner = StubExecutorSubstrate::new();
        let (sub, counters) = CountingSubstrate::new(inner);
        let a = sub
            .create_node(TenantId::DEFAULT, Some("U"), &[], &tctx())
            .expect("a");
        sub.delete_node(TenantId::DEFAULT, a, false, &tctx())
            .expect("del");
        let snap = counters.snapshot();
        assert_eq!(snap.nodes_created, 1);
        assert_eq!(snap.nodes_deleted, 1);
    }

    #[test]
    fn set_node_property_assign_increments_one() {
        let inner = StubExecutorSubstrate::new();
        let (sub, counters) = CountingSubstrate::new(inner);
        let id = sub
            .create_node(TenantId::DEFAULT, Some("U"), &[], &tctx())
            .expect("a");
        let m = SetNodeMutation::PropertyAssign {
            name: "x".into(),
            value: Value::Integer(42),
        };
        sub.set_node(TenantId::DEFAULT, id, &m, &tctx())
            .expect("set");
        assert_eq!(counters.snapshot().properties_set, 1);
    }

    #[test]
    fn set_node_property_replace_increments_by_entries_len() {
        let inner = StubExecutorSubstrate::new();
        let (sub, counters) = CountingSubstrate::new(inner);
        let id = sub
            .create_node(TenantId::DEFAULT, Some("U"), &[], &tctx())
            .expect("a");
        let m = SetNodeMutation::PropertyReplace(vec![
            ("a".into(), Value::Integer(1)),
            ("b".into(), Value::Integer(2)),
            ("c".into(), Value::Integer(3)),
        ]);
        sub.set_node(TenantId::DEFAULT, id, &m, &tctx())
            .expect("set");
        assert_eq!(counters.snapshot().properties_set, 3);
    }

    #[test]
    fn set_node_label_add_increments_labels_added_not_properties() {
        let inner = StubExecutorSubstrate::new();
        let (sub, counters) = CountingSubstrate::new(inner);
        let id = sub
            .create_node(TenantId::DEFAULT, Some("U"), &[], &tctx())
            .expect("a");
        let m = SetNodeMutation::LabelAdd(vec!["L1".into(), "L2".into()]);
        sub.set_node(TenantId::DEFAULT, id, &m, &tctx())
            .expect("set");
        let snap = counters.snapshot();
        assert_eq!(snap.labels_added, 2);
        assert_eq!(snap.properties_set, 0, "labels DON'T tick properties");
    }

    #[test]
    fn remove_node_property_increments_properties_removed_by_one() {
        let inner = StubExecutorSubstrate::new();
        let (sub, counters) = CountingSubstrate::new(inner);
        let id = sub
            .create_node(TenantId::DEFAULT, Some("U"), &[], &tctx())
            .expect("a");
        let m = RemoveNodeMutation::Property("x".into());
        sub.remove_node(TenantId::DEFAULT, id, &m, &tctx())
            .expect("rm");
        assert_eq!(counters.snapshot().properties_removed, 1);
    }

    #[test]
    fn remove_node_label_remove_increments_labels_removed_by_count() {
        let inner = StubExecutorSubstrate::new();
        let (sub, counters) = CountingSubstrate::new(inner);
        let id = sub
            .create_node(TenantId::DEFAULT, Some("U"), &[], &tctx())
            .expect("a");
        let m = RemoveNodeMutation::LabelRemove(vec!["L1".into(), "L2".into(), "L3".into()]);
        sub.remove_node(TenantId::DEFAULT, id, &m, &tctx())
            .expect("rm");
        assert_eq!(counters.snapshot().labels_removed, 3);
    }

    #[test]
    fn err_path_does_not_increment_counters() {
        // The stub substrate's default-trait impl for delete_node on a
        // node that has NEVER been created returns Ok per the
        // tombstone-set semantics — we can't easily test the
        // Err branch on the stub. Instead, build a manual "always-err"
        // substrate and verify counters stay zero.
        #[derive(Default)]
        struct ErrSubstrate;
        impl ExecutorSubstrate for ErrSubstrate {
            fn scan_nodes(
                &self,
                _t: TenantId,
                _l: Option<LabelId>,
                _r: Lsn,
            ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
                Ok(vec![])
            }
            fn expand(
                &self,
                _t: TenantId,
                _f: NodeId,
                _rt: Option<TypeId>,
                _d: Direction,
                _r: Lsn,
            ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
                Ok(vec![])
            }
            fn vector_search(
                &self,
                _t: TenantId,
                _p: &str,
                _q: &[f32],
                _k: u64,
                _r: Lsn,
            ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
                Ok(vec![])
            }
            fn bm25_search(
                &self,
                _t: TenantId,
                _p: &str,
                _q: &str,
                _k: u64,
                _r: Lsn,
            ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
                Ok(vec![])
            }
            fn community_members(
                &self,
                _t: TenantId,
                _c: i64,
                _r: Lsn,
            ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
                Ok(vec![])
            }
            fn create_node(
                &self,
                _t: TenantId,
                _l: Option<&str>,
                _p: &[(String, Value)],
                _ctx: &ExecutionContext,
            ) -> Result<NodeId, SubstrateAccessError> {
                Err(SubstrateAccessError::Io("create_node forced err".into()))
            }
        }
        let (sub, counters) = CountingSubstrate::new(ErrSubstrate);
        let err = sub.create_node(TenantId::DEFAULT, Some("X"), &[], &tctx());
        assert!(err.is_err());
        // ADR-153 §D-2: counters reflect committed effects only.
        assert!(counters.snapshot().is_empty());
        // Used to suppress unused-var warning on the NodeView import.
        let _ = NodeView::new(NodeId::new(1), None);
    }

    #[test]
    fn pure_read_traffic_leaves_all_counters_zero() {
        // Pin: scans/expand/vector/bm25/community pass-throughs must
        // NOT bump any counter. The decorator must be invisible on the
        // read path.
        let inner = StubExecutorSubstrate::new()
            .with_vector_substrate()
            .with_bm25_substrate()
            .with_community_substrate()
            .with_node(TenantId::DEFAULT, NodeView::new(NodeId::new(1), None));
        let (sub, counters) = CountingSubstrate::new(inner);
        let _ = sub.scan_nodes(TenantId::DEFAULT, None, Lsn::MAX).unwrap();
        let _ = sub.expand(
            TenantId::DEFAULT,
            NodeId::new(1),
            None,
            Direction::Undirected,
            Lsn::MAX,
        );
        let _ = sub.vector_search(TenantId::DEFAULT, "p", &[0.0], 10, Lsn::MAX);
        let _ = sub.bm25_search(TenantId::DEFAULT, "p", "q", 10, Lsn::MAX);
        let _ = sub.community_members(TenantId::DEFAULT, 7, Lsn::MAX);
        let _ = sub.has_vector_substrate();
        let _ = sub.has_bm25_substrate();
        let _ = sub.has_community_substrate();
        assert!(counters.snapshot().is_empty());
    }
}
