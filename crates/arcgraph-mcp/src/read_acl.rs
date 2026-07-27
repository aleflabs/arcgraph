//! Shared ADR-212 read-authorization choke point.
//!
//! Every principal-scoped graph read enters through [`authorize_read`].
//! The returned [`ReadAccess`] is the only production seam that calls
//! [`EffectivePermissions::is_visible`]. This keeps the fail-closed scope
//! gate, permission-index resolution, and node-visibility predicate in one
//! place for MCP tools and non-MCP query transports alike.

use std::sync::Arc;

use arcgraph_core::{LabelId, Lsn, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::executor::substrate::{
    BoundEdge, BoundEdgeCursor, BoundNode, ExecutorSubstrate, MergeGuard,
    PropertyIndexRegistration, RankedHit, RemoveNodeMutation, RemoveRelMutation, SetNodeMutation,
    SetRelMutation, SubstrateAccessError, VectorIndexCatalogEntry, VectorIndexRegistration,
};
use arcgraph_query::executor::{ExecutionContext, Value};
use arcgraph_query::logical_plan::{CountStoreSource, Direction};
use arcgraph_storage::permissions::{EffectivePermissions, PermissionIndex};

use crate::{MCPError, SessionScope};

/// Stable error slug for an unavailable ADR-212 permission index.
pub(crate) const PERMISSION_INDEX_SLUG: &str = "permissions";

/// Resolve the authorization context for one graph-read statement.
///
/// This is the single shared choke point for principal-aware reads:
///
/// 1. an absent principal is admitted only for explicit Power scope;
/// 2. an empty principal is rejected;
/// 3. a principal requires a real per-tenant [`PermissionIndex`]; and
/// 4. the resulting [`ReadAccess::allows`] predicate is the sole
///    production call to [`EffectivePermissions::is_visible`].
///
/// `permission_index` is lazy so the Power-scoped SYSTEM-TRUSTED path does
/// not probe storage. Provider errors propagate without being weakened.
pub(crate) fn authorize_read(
    surface: &'static str,
    principal: Option<&str>,
    session_scope: SessionScope,
    permission_index: impl FnOnce() -> Result<Option<Arc<PermissionIndex>>, MCPError>,
) -> Result<ReadAccess, MCPError> {
    let Some(principal) = principal else {
        if session_scope.admits_power() {
            return Ok(ReadAccess(ReadAccessKind::SystemTrusted));
        }
        return Err(MCPError::Forbidden {
            required_scope: SessionScope::Power.slug(),
        });
    };
    if principal.is_empty() {
        return Err(MCPError::InvalidParams(format!(
            "{surface}: principal must be non-empty when present"
        )));
    }
    let Some(index) = permission_index()? else {
        return Err(MCPError::IndexUnavailable(format!(
            "{PERMISSION_INDEX_SLUG}: {surface} exposes no permission index; \
             principal-scoped read refused (fail-closed, ADR-212)"
        )));
    };
    Ok(ReadAccess(ReadAccessKind::Principal(
        index.effective(principal),
    )))
}

/// Resolved, statement-scoped visibility decision returned by
/// [`authorize_read`]. The tuple field and inner enum stay private so callers
/// cannot construct any access context without going through the shared gate.
#[derive(Clone)]
pub(crate) struct ReadAccess(ReadAccessKind);

#[derive(Clone)]
enum ReadAccessKind {
    SystemTrusted,
    Principal(Arc<EffectivePermissions>),
}

impl ReadAccess {
    /// Whether a node may participate in this statement's result universe.
    ///
    /// This is intentionally the only non-test `is_visible` invocation in
    /// `arcgraph-mcp`: neutralizing it disables every read surface together,
    /// which makes the shared-seam RED gates class-wide rather than per-copy.
    #[inline]
    pub(crate) fn allows(&self, node: NodeId) -> bool {
        match &self.0 {
            ReadAccessKind::SystemTrusted => true,
            ReadAccessKind::Principal(permissions) => permissions.is_visible(node),
        }
    }

    #[inline]
    pub(crate) fn is_system_trusted(&self) -> bool {
        matches!(self.0, ReadAccessKind::SystemTrusted)
    }

    /// Defense-in-depth check for materialized rows. Executor substrate
    /// filtering already prevents scalar projections and aggregates over
    /// denied nodes; this recursive check additionally prevents entity/path
    /// values introduced by a write-return arm from bypassing final assembly.
    pub(crate) fn allows_row(&self, row: &[Value]) -> bool {
        row.iter().all(|value| self.allows_value(value))
    }

    fn allows_value(&self, value: &Value) -> bool {
        match value {
            Value::Node(node) => self.allows(node.id),
            Value::Relationship(rel) => self.allows(rel.from) && self.allows(rel.to),
            Value::List(values) => values.iter().all(|value| self.allows_value(value)),
            Value::Map(values) => values.values().all(|value| self.allows_value(value)),
            Value::Path(path) => {
                self.allows(path.start.id)
                    && path.segments.iter().all(|segment| {
                        self.allows(segment.rel.from)
                            && self.allows(segment.rel.to)
                            && self.allows(segment.end.id)
                    })
            }
            Value::Null
            | Value::Boolean(_)
            | Value::Integer(_)
            | Value::Float(_)
            | Value::String(_)
            | Value::Temporal(_)
            | Value::LocalDateTime(_)
            | Value::Date(_)
            | Value::Duration(_)
            | Value::Decimal(_) => true,
        }
    }
}

/// ADR-212 executor decorator used by Bolt.
///
/// Filtering at the substrate boundary, before operators, is required for
/// scalar projections, predicates, aggregates, ORDER/LIMIT, and paths to be
/// computed over the authorized universe. Final row filtering alone cannot
/// recover provenance after `RETURN n.secret` or `count(n)` becomes a scalar.
pub(crate) struct PermissionEnforcedSubstrate<S> {
    inner: S,
    access: ReadAccess,
}

impl<S> PermissionEnforcedSubstrate<S> {
    pub(crate) fn new(inner: S, access: ReadAccess) -> Self {
        Self { inner, access }
    }

    fn retain_nodes(&self, nodes: &mut Vec<BoundNode>) {
        nodes.retain(|node| self.access.allows(node.node.id));
    }

    fn retain_edges(&self, from: NodeId, edges: &mut Vec<BoundEdge>) {
        if !self.access.allows(from) {
            edges.clear();
            return;
        }
        edges.retain(|edge| {
            self.access.allows(edge.dst.id)
                && self.access.allows(edge.rel.from)
                && self.access.allows(edge.rel.to)
        });
    }

    fn retain_hits(&self, hits: &mut Vec<RankedHit>) {
        hits.retain(|hit| self.access.allows(hit.node.id));
    }
}

impl<S: ExecutorSubstrate> ExecutorSubstrate for PermissionEnforcedSubstrate<S> {
    fn count_store(
        &self,
        tenant: TenantId,
        source: CountStoreSource,
    ) -> Result<u64, SubstrateAccessError> {
        if self.access.is_system_trusted() {
            return self.inner.count_store(tenant, source);
        }

        // A total from the O(1) count store is an existence oracle. Until the
        // planner exposes a decorator-aware rewrite toggle, compute the exact
        // principal count from filtered scans. This is O(V + E), matching the
        // normal aggregate fallback and allocating at most the scan result.
        match source {
            CountStoreSource::Nodes => {
                let mut nodes = self.inner.scan_nodes(tenant, None, Lsn::MAX)?;
                self.retain_nodes(&mut nodes);
                Ok(nodes.len() as u64)
            }
            CountStoreSource::NodesWithLabel(label) => {
                let mut nodes = self.inner.scan_nodes(tenant, Some(label), Lsn::MAX)?;
                self.retain_nodes(&mut nodes);
                Ok(nodes.len() as u64)
            }
            CountStoreSource::Relationships | CountStoreSource::RelsWithType(_) => {
                let rel_type = match source {
                    CountStoreSource::RelsWithType(rel_type) => Some(rel_type),
                    _ => None,
                };
                let mut nodes = self.inner.scan_nodes(tenant, None, Lsn::MAX)?;
                self.retain_nodes(&mut nodes);
                let mut count = 0u64;
                for node in nodes {
                    let mut edges = self.inner.expand(
                        tenant,
                        node.node.id,
                        rel_type,
                        Direction::LeftToRight,
                        Lsn::MAX,
                    )?;
                    self.retain_edges(node.node.id, &mut edges);
                    count = count.saturating_add(edges.len() as u64);
                }
                Ok(count)
            }
        }
    }

    fn scan_nodes(
        &self,
        tenant: TenantId,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let mut nodes = self.inner.scan_nodes(tenant, label, read_lsn)?;
        self.retain_nodes(&mut nodes);
        Ok(nodes)
    }

    fn scan_nodes_with_context(
        &self,
        ctx: &ExecutionContext,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let mut nodes = self.inner.scan_nodes_with_context(ctx, label, read_lsn)?;
        self.retain_nodes(&mut nodes);
        Ok(nodes)
    }

    fn scan_nodes_projected_with_context(
        &self,
        ctx: &ExecutionContext,
        label: Option<LabelId>,
        read_lsn: Lsn,
        projected_properties: &[String],
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let mut nodes = self.inner.scan_nodes_projected_with_context(
            ctx,
            label,
            read_lsn,
            projected_properties,
        )?;
        self.retain_nodes(&mut nodes);
        Ok(nodes)
    }

    fn expand(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        let mut edges = self
            .inner
            .expand(tenant, from, rel_type, direction, read_lsn)?;
        self.retain_edges(from, &mut edges);
        Ok(edges)
    }

    fn expand_with_context(
        &self,
        ctx: &ExecutionContext,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        let mut edges = self
            .inner
            .expand_with_context(ctx, from, rel_type, direction, read_lsn)?;
        self.retain_edges(from, &mut edges);
        Ok(edges)
    }

    fn node_by_id_with_context(
        &self,
        ctx: &ExecutionContext,
        id: NodeId,
    ) -> Result<Option<BoundNode>, SubstrateAccessError> {
        if !self.access.allows(id) {
            return Ok(None);
        }
        self.inner.node_by_id_with_context(ctx, id)
    }

    fn property_index_lookup_with_context(
        &self,
        ctx: &ExecutionContext,
        label: LabelId,
        property: &str,
        value: &Value,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let mut nodes = self
            .inner
            .property_index_lookup_with_context(ctx, label, property, value, read_lsn)?;
        self.retain_nodes(&mut nodes);
        Ok(nodes)
    }

    fn value_is_indexable(&self, value: &Value) -> bool {
        self.inner.value_is_indexable(value)
    }

    fn expand_cursor(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<BoundEdgeCursor, SubstrateAccessError> {
        if !self.access.allows(from) {
            return Ok(Box::new(std::iter::empty()));
        }
        let access = self.access.clone();
        let cursor = self
            .inner
            .expand_cursor(tenant, from, rel_type, direction, read_lsn)?;
        Ok(Box::new(cursor.filter_map(move |edge| match edge {
            Ok(edge)
                if access.allows(edge.dst.id)
                    && access.allows(edge.rel.from)
                    && access.allows(edge.rel.to) =>
            {
                Some(Ok(edge))
            }
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        })))
    }

    fn expand_cursor_with_context(
        &self,
        ctx: &ExecutionContext,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<BoundEdgeCursor, SubstrateAccessError> {
        if !self.access.allows(from) {
            return Ok(Box::new(std::iter::empty()));
        }
        let access = self.access.clone();
        let cursor = self
            .inner
            .expand_cursor_with_context(ctx, from, rel_type, direction, read_lsn)?;
        Ok(Box::new(cursor.filter_map(move |edge| match edge {
            Ok(edge)
                if access.allows(edge.dst.id)
                    && access.allows(edge.rel.from)
                    && access.allows(edge.rel.to) =>
            {
                Some(Ok(edge))
            }
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        })))
    }

    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        let mut hits = self
            .inner
            .vector_search(tenant, property, query_vec, k, read_lsn)?;
        self.retain_hits(&mut hits);
        Ok(hits)
    }

    fn bm25_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_text: &str,
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        let mut hits = self
            .inner
            .bm25_search(tenant, property, query_text, k, read_lsn)?;
        self.retain_hits(&mut hits);
        Ok(hits)
    }

    fn community_members(
        &self,
        tenant: TenantId,
        community_id: i64,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let mut nodes = self
            .inner
            .community_members(tenant, community_id, read_lsn)?;
        self.retain_nodes(&mut nodes);
        Ok(nodes)
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

    fn begin_statement(&self, ctx: &ExecutionContext) -> Result<(), SubstrateAccessError> {
        self.inner.begin_statement(ctx)
    }

    fn commit_statement(&self, ctx: &ExecutionContext) -> Result<(), SubstrateAccessError> {
        self.inner.commit_statement(ctx)
    }

    fn rollback_statement(&self, ctx: &ExecutionContext) {
        self.inner.rollback_statement(ctx);
    }

    fn create_node(
        &self,
        tenant: TenantId,
        label: Option<&str>,
        properties: &[(String, Value)],
        ctx: &ExecutionContext,
    ) -> Result<NodeId, SubstrateAccessError> {
        self.inner.create_node(tenant, label, properties, ctx)
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
        self.inner
            .create_rel(tenant, source, target, label, properties, ctx)
    }

    fn merge_guard(
        &self,
        tenant: TenantId,
        key: &str,
    ) -> Result<Option<Box<dyn MergeGuard>>, SubstrateAccessError> {
        self.inner.merge_guard(tenant, key)
    }

    fn delete_node(
        &self,
        tenant: TenantId,
        node: NodeId,
        detach: bool,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        self.inner.delete_node(tenant, node, detach, ctx)
    }

    fn delete_rel(
        &self,
        tenant: TenantId,
        rel: RelId,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        self.inner.delete_rel(tenant, rel, ctx)
    }

    fn set_node(
        &self,
        tenant: TenantId,
        node: NodeId,
        mutation: &SetNodeMutation,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        self.inner.set_node(tenant, node, mutation, ctx)
    }

    fn set_rel(
        &self,
        tenant: TenantId,
        rel: RelId,
        mutation: &SetRelMutation,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        self.inner.set_rel(tenant, rel, mutation, ctx)
    }

    fn remove_node(
        &self,
        tenant: TenantId,
        node: NodeId,
        mutation: &RemoveNodeMutation,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        self.inner.remove_node(tenant, node, mutation, ctx)
    }

    fn remove_rel(
        &self,
        tenant: TenantId,
        rel: RelId,
        mutation: &RemoveRelMutation,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        self.inner.remove_rel(tenant, rel, mutation, ctx)
    }

    fn register_vector_index(
        &self,
        tenant: TenantId,
        entry: VectorIndexCatalogEntry,
        if_not_exists: bool,
    ) -> Result<VectorIndexRegistration, SubstrateAccessError> {
        self.inner
            .register_vector_index(tenant, entry, if_not_exists)
    }

    fn list_vector_indexes(&self, tenant: TenantId) -> Vec<VectorIndexCatalogEntry> {
        self.inner.list_vector_indexes(tenant)
    }

    fn resolve_vector_index(
        &self,
        tenant: TenantId,
        name: &str,
    ) -> Option<VectorIndexCatalogEntry> {
        self.inner.resolve_vector_index(tenant, name)
    }

    fn create_property_index(
        &self,
        tenant: TenantId,
        name: &str,
        if_not_exists: bool,
        label: &str,
        property: &str,
    ) -> Result<PropertyIndexRegistration, SubstrateAccessError> {
        self.inner
            .create_property_index(tenant, name, if_not_exists, label, property)
    }
}
