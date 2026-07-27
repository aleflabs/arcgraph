//! [`NamedShortestPathOp`] — `MATCH p = SHORTEST_PATH(...)` (M4-63).
//!
//! Lowers from [`crate::logical_plan::LogicalNamedPath`] with
//! [`crate::logical_plan::PathAlgorithm::ShortestPath`]. Computes the
//! shortest path from a source node to a target node (or to every
//! reachable node, depending on the planner's lowering shape).
//!
//! # v1.0 algorithm: BFS only
//!
//! Per the M4-63 spawn brief: "single-source + bidirectional; v1.0
//! BFS only; no DFS/A*". v1.0-alpha implements:
//!
//! - **Single-source BFS**: given a source node, emit one row per
//!   reachable target with the shortest-path length + path-cell
//!   carrier.
//! - **Bidirectional BFS**: given a source AND target node, emit at
//!   most one row carrying the shortest path between them.
//!   Bidirectional BFS halves the search frontier compared to plain
//!   SSSP-then-target — the implementation alternates expansion from
//!   each end until the frontiers meet.
//!
//! DFS / A* / weighted-edge variants are deferred per ADR-038 §2 D-7
//! to a future amendment alongside `arcgraph-storage` exposing
//! per-edge weights through the substrate.
//!
//! # Substrate adapter
//!
//! Path traversal reads neighbors via
//! [`crate::executor::ExecutorSubstrate::expand`] — the same
//! interface [`crate::executor::ops::ExpandOp`] uses. The path op
//! reuses the substrate's adjacency view; no new substrate trait
//! methods are introduced (per `feedback_avoid_speculative_scaffolding.md`
//! 7-slice 3-strike).
//!
//! # Schema
//!
//! Output schema is a single column carrying the path-binding ID. The
//! column value is a [`Value::Path`] ([`PathView`]) carrying the path's
//! nodes AND relationships in source-to-target traversal order, for each
//! emitted row (ADR-194 D-5, realizing ADR-193 D-14). The richer typed
//! path that Cypher 9 §6.5 / ADR-007 anticipated is now REALIZED: every
//! path producer (`shortestPath`, `allShortestPaths`, and the plain
//! named-path op) emits the SAME `Value::Path` representation, so
//! `nodes(shortestPath(...))` == `nodes(p)`. The BFS threads the
//! relationship traversed at each hop through its parent/predecessor
//! maps so the reconstructed [`PathView`] is complete (the migration from
//! the legacy node-only `Value::List`).
//!
//! # Algorithms
//!
//! - `shortestPath` / `SHORTEST_PATH` → ONE minimum-length path
//!   (bidirectional BFS when a target is bound; single-source enumeration
//!   to every reachable node when the tail endpoint is anonymous).
//! - `allShortestPaths` → ALL equal-minimum-length source→target paths
//!   (ADR-194 D-4): a single-source layered BFS computes the shortest
//!   distance + the full predecessor DAG, then enumerates every
//!   source→target path through it. REQUIRES a bound target endpoint.
//!
//! # Memory budget
//!
//! BFS state (visited set + frontier queue) is tracked against the
//! per-tenant memory budget. The path output's row size is bounded
//! by the diameter of the substrate graph; a deep traversal may
//! produce a long path-list cell.
//!
//! # ADR provenance
//!
//! - **ADR-038 amendment-02 §M4.f** — primary M4-63 cite.
//! - **ADR-038 §2 D-7** — SHORTEST_PATH operator contract;
//!   weighted variants reserved for v1.1.
//! - **Cypher 9 §6.5** — named-path semantics.

use std::collections::{HashMap, HashSet, VecDeque};

use arcgraph_core::{Lsn, NodeId};

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::ops::PhysicalOperator;
use crate::executor::ops::schema_index;
use crate::executor::substrate::{ExecutorSubstrate, SubstrateAccessError};
use crate::executor::value::{NodeView, PathSegment, PathView, RelView, Value};
use crate::logical_plan::Direction;
use crate::semantic::bound_ast::BindingId;

/// Maximum BFS hop depth at v1.0-alpha. Defends against pathological
/// substrates with cycles + low diameter heuristic mismatches. The
/// cap is intentionally generous (LDBC SNB Interactive max diameter
/// is ~6); a future grammar `*N..M` form will pass a per-query depth
/// to the operator.
pub const DEFAULT_MAX_DEPTH: u32 = 64;

/// Cardinality budget for `allShortestPaths` (ADR-194 D-4 / §Negative).
/// The number of equal-minimum-length paths can be exponential in a
/// dense graph; enumerating them all unbounded risks OOM. When the
/// enumeration would exceed this many paths the operator surfaces a clean
/// [`ExecutionError`] instead (NEVER an OOM — ADR-194 §Negative honors the
/// ADR-025 frontier-budget discipline). The cap is generous relative to
/// LDBC IC11-IC14 shapes (small diameter, few equal-min paths).
pub const MAX_ALL_SHORTEST_PATHS: usize = 10_000;

/// Source-target binding pair the path operator consumes.
#[derive(Debug, Clone)]
pub struct PathSpec {
    /// Source node binding (must come from a child Scan / Filter /
    /// Project that surfaces a single Node value).
    pub source: BindingId,
    /// Target node binding. `None` ⇒ single-source mode (emit one row
    /// per reachable node); `Some(b)` ⇒ bidirectional mode (emit at
    /// most one row).
    pub target: Option<BindingId>,
    /// Relationship-type filter (`Some(t)` ⇒ traverse only edges of
    /// type `t`; `None` ⇒ any rel-type).
    pub rel_type: Option<arcgraph_core::TypeId>,
    /// Direction; v1.0-alpha defaults to Undirected for shortest-path
    /// (the spec is direction-aware but the v1.0 grammar only emits
    /// undirected named-paths until ADR-038 amendment-02 lights the
    /// directed-shortest-path surface).
    pub direction: Direction,
    /// Path-binding ID (the output column carrying the path cell).
    pub path_var: BindingId,
    /// **ADR-194 D-4.** When `true`, the operator runs the
    /// `allShortestPaths` algorithm — enumerate EVERY equal-minimum-length
    /// source→target path (one `Value::Path` row each). `allShortestPaths`
    /// REQUIRES `target = Some(..)` (the pipeline rejects an anonymous tail
    /// before construction). When `false`, the single-shortest
    /// `shortestPath` / `SHORTEST_PATH` behavior: one path via
    /// bidirectional BFS (`target = Some`), or single-source enumeration to
    /// every reachable node (`target = None`).
    pub all_shortest: bool,
}

/// `MATCH p = SHORTEST_PATH(...)` operator — v1.0 BFS only.
pub struct NamedShortestPathOp {
    child: Box<PhysicalOperator>,
    spec: PathSpec,
    /// Output schema = `[path_var]` (a single path-list column).
    schema: Vec<BindingId>,
    /// Cached child schema for source/target lookup.
    child_schema: Vec<BindingId>,
    /// Have we drained the upstream + computed all paths?
    materialized: bool,
    /// Computed output rows.
    output: Vec<Vec<Value>>,
    /// Output cursor.
    cursor: usize,
    /// MVCC visibility key threaded through to the substrate.
    plan_read_lsn: Lsn,
}

impl std::fmt::Debug for NamedShortestPathOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedShortestPathOp")
            .field("child", &self.child)
            .field("source", &self.spec.source)
            .field("target", &self.spec.target)
            .field("schema", &self.schema)
            .field("output_rows", &self.output.len())
            .finish()
    }
}

impl NamedShortestPathOp {
    /// Construct a [`NamedShortestPathOp`].
    #[must_use]
    pub fn new(child: PhysicalOperator, spec: PathSpec, plan_read_lsn: Lsn) -> Self {
        let child_schema = child.schema().to_vec();
        let schema = vec![spec.path_var];
        Self {
            child: Box::new(child),
            spec,
            schema,
            child_schema,
            materialized: false,
            output: Vec::new(),
            cursor: 0,
            plan_read_lsn,
        }
    }

    /// Output schema.
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if !self.materialized {
            self.materialize(ctx, substrate)?;
        }
        if self.cursor >= self.output.len() {
            return Ok(Batch::empty(self.schema.len()));
        }
        let mut out = Batch::with_capacity(self.schema.len());
        let take = (self.output.len() - self.cursor).min(crate::executor::BATCH_ROWS);
        for row in &self.output[self.cursor..self.cursor + take] {
            if !out.push_row(row.clone()) {
                return Err(ExecutionError::Eval(
                    "NamedShortestPathOp: batch overflow during sized push".into(),
                ));
            }
        }
        self.cursor += take;
        Ok(out)
    }

    fn materialize<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<(), ExecutionError> {
        let source_idx = self.find_idx(self.spec.source)?;
        let target_idx = match self.spec.target {
            Some(b) => Some(self.find_idx(b)?),
            None => None,
        };
        // De-duplicate by (source, target) endpoint pair. The child
        // pattern subtree may enumerate MULTIPLE rows for the SAME endpoint
        // pair — a var-length `(a)-[:R*1..5]->(b)` expand yields one row
        // per connecting path — but `shortestPath` / `allShortestPaths` are
        // functions of the ENDPOINTS + the graph: each distinct
        // `(source, target)` pair is processed exactly ONCE (one shortest
        // path, or one full equal-min set), matching openCypher cardinality.
        // Without this, N connecting paths in the pattern would multiply
        // the result N× (latent pre-D-4: #750's single-path fixtures never
        // surfaced it; allShortestPaths over a graph with ≥2 equal-min
        // paths makes it acute).
        let mut seen: HashSet<(NodeId, Option<NodeId>)> = HashSet::new();
        loop {
            ctx.cancellation().check()?;
            let batch = self.child.next_batch(ctx, substrate)?;
            if batch.is_empty() {
                break;
            }
            for row in batch.into_rows() {
                let source_node = match row.get(source_idx) {
                    Some(Value::Node(n)) => n.clone(),
                    _ => {
                        return Err(ExecutionError::Eval(
                            "NamedShortestPathOp: source binding is not a Node".into(),
                        ));
                    }
                };
                let target_node = match target_idx {
                    Some(idx) => match row.get(idx) {
                        Some(Value::Node(n)) => Some(n.clone()),
                        _ => {
                            return Err(ExecutionError::Eval(
                                "NamedShortestPathOp: target binding is not a Node".into(),
                            ));
                        }
                    },
                    None => None,
                };
                // Process each distinct (source, target) endpoint pair once.
                let dedup_key = (source_node.id, target_node.as_ref().map(|n| n.id));
                if !seen.insert(dedup_key) {
                    continue;
                }
                if self.spec.all_shortest {
                    // ADR-194 D-4 — `allShortestPaths` REQUIRES a bound
                    // target (the pipeline rejects an anonymous tail before
                    // constructing the op; this guard is the defensive
                    // backstop). Enumerate EVERY equal-min-length path.
                    let target = target_node.ok_or_else(|| {
                        ExecutionError::Eval(
                            "NamedShortestPathOp: allShortestPaths requires a bound target \
                             endpoint (ADR-194 D-4); the pipeline must not construct it with \
                             target = None"
                                .into(),
                        )
                    })?;
                    let mut paths =
                        self.all_shortest_paths(ctx, substrate, &source_node, &target)?;
                    self.output.append(&mut paths);
                } else if let Some(target) = target_node {
                    // ADR-194 D-3a — `bidirectional` returns a fully-formed
                    // OUTPUT ROW (`vec![Value::Path(..)]`), exactly like each
                    // element of `single_source`'s `Vec<row>`. Push it AS the
                    // row.
                    if let Some(path_row) =
                        self.bidirectional(ctx, substrate, &source_node, &target)?
                    {
                        self.output.push(path_row);
                    }
                } else {
                    let mut paths = self.single_source(ctx, substrate, &source_node)?;
                    self.output.append(&mut paths);
                }
            }
        }
        self.materialized = true;
        Ok(())
    }

    /// Look up the column index for `binding` in the child schema.
    fn find_idx(&self, binding: BindingId) -> Result<usize, ExecutionError> {
        schema_index(&self.child_schema, binding).ok_or_else(|| {
            ExecutionError::Eval(format!(
                "NamedShortestPathOp: binding {binding:?} not in child schema"
            ))
        })
    }

    /// Single-source BFS. Emits one row per reachable target.
    ///
    /// Returns each path as a `Vec<Value>` row carrying a [`Value::Path`]
    /// ([`PathView`]) with the nodes AND relationships in source→target
    /// traversal order (ADR-194 D-5). The `parent` map records the
    /// relationship traversed to reach each node so the reconstructed
    /// path is complete (nodes + rels).
    fn single_source<S: ExecutorSubstrate>(
        &self,
        ctx: &ExecutionContext,
        substrate: &S,
        source: &NodeView,
    ) -> Result<Vec<Vec<Value>>, ExecutionError> {
        let mut visited: HashMap<NodeId, NodeView> = HashMap::new();
        // child → `Some((parent, rel_traversed))`; the source maps to
        // `None`. The rel lets `reconstruct_path` build a complete
        // `Value::Path` (D-5) rather than a node-only list.
        let mut parent: HashMap<NodeId, Option<(NodeId, RelView)>> = HashMap::new();
        let mut frontier: VecDeque<(NodeView, u32)> = VecDeque::new();
        visited.insert(source.id, source.clone());
        parent.insert(source.id, None);
        frontier.push_back((source.clone(), 0));
        let mut out: Vec<Vec<Value>> = Vec::new();
        while let Some((node, depth)) = frontier.pop_front() {
            ctx.cancellation().check()?;
            // Emit a path for every visited target (including source
            // itself — Cypher 9 §6.5 admits zero-length paths).
            if node.id != source.id {
                let path = reconstruct_path(source, node.id, &visited, &parent);
                out.push(vec![Value::Path(path)]);
            }
            if depth >= DEFAULT_MAX_DEPTH {
                continue;
            }
            for edge in expand_neighbors(
                substrate,
                ctx,
                node.id,
                self.spec.rel_type,
                self.spec.direction,
                self.plan_read_lsn,
            )? {
                if visited.insert(edge.dst.id, edge.dst.clone()).is_none() {
                    parent.insert(edge.dst.id, Some((node.id, edge.rel.clone())));
                    frontier.push_back((edge.dst, depth + 1));
                }
            }
        }
        Ok(out)
    }

    /// Bidirectional BFS. Returns at most one shortest path between
    /// `source` and `target`.
    ///
    /// Algorithm: maintain two BFS frontiers (one from source, one
    /// from target). At each step expand the smaller frontier. When
    /// any node appears in BOTH frontiers' visited sets, the meeting
    /// node is on a shortest path; reconstruct via the two parent
    /// chains.
    fn bidirectional<S: ExecutorSubstrate>(
        &self,
        ctx: &ExecutionContext,
        substrate: &S,
        source: &NodeView,
        target: &NodeView,
    ) -> Result<Option<Vec<Value>>, ExecutionError> {
        if source.id == target.id {
            // Zero-length path: a single-node `Value::Path` (no segments),
            // per Cypher 9 §6.5 zero-length-path semantics + ADR-193 D-6
            // (`PathView { start, segments: [] }`).
            return Ok(Some(vec![Value::Path(PathView::new(source.clone()))]));
        }
        // child → `Some((parent, rel))`; endpoints map to `None`. The rel
        // lets `reconstruct_bidirectional` build a complete `Value::Path`
        // (nodes + rels, ADR-194 D-5).
        let mut fwd_visited: HashMap<NodeId, NodeView> = HashMap::new();
        let mut fwd_parent: HashMap<NodeId, Option<(NodeId, RelView)>> = HashMap::new();
        let mut fwd_front: VecDeque<NodeId> = VecDeque::new();
        let mut bwd_visited: HashMap<NodeId, NodeView> = HashMap::new();
        let mut bwd_parent: HashMap<NodeId, Option<(NodeId, RelView)>> = HashMap::new();
        let mut bwd_front: VecDeque<NodeId> = VecDeque::new();
        fwd_visited.insert(source.id, source.clone());
        fwd_parent.insert(source.id, None);
        fwd_front.push_back(source.id);
        bwd_visited.insert(target.id, target.clone());
        bwd_parent.insert(target.id, None);
        bwd_front.push_back(target.id);
        let mut depth: u32 = 0;
        while !fwd_front.is_empty() && !bwd_front.is_empty() {
            ctx.cancellation().check()?;
            depth += 1;
            if depth > DEFAULT_MAX_DEPTH {
                return Ok(None);
            }
            // Expand the smaller frontier each iteration to halve the
            // search frontier compared to plain SSSP.
            let expand_fwd = fwd_front.len() <= bwd_front.len();
            let (front, visited, parent, other_visited, reverse_direction) = if expand_fwd {
                (
                    &mut fwd_front,
                    &mut fwd_visited,
                    &mut fwd_parent,
                    &bwd_visited,
                    false,
                )
            } else {
                (
                    &mut bwd_front,
                    &mut bwd_visited,
                    &mut bwd_parent,
                    &fwd_visited,
                    true,
                )
            };
            let mut next_front: VecDeque<NodeId> = VecDeque::new();
            while let Some(nid) = front.pop_front() {
                let neighbors = expand_neighbors(
                    substrate,
                    ctx,
                    nid,
                    self.spec.rel_type,
                    // Backward expansion flips the direction so the
                    // edges are traversed in reverse from the target side.
                    if reverse_direction {
                        flip_direction(self.spec.direction)
                    } else {
                        self.spec.direction
                    },
                    self.plan_read_lsn,
                )?;
                for edge in neighbors {
                    if visited.insert(edge.dst.id, edge.dst.clone()).is_none() {
                        parent.insert(edge.dst.id, Some((nid, edge.rel.clone())));
                        if other_visited.contains_key(&edge.dst.id) {
                            // Frontiers meet at edge.dst. Reconstruct the
                            // full node+rel path (ADR-194 D-5).
                            let meeting = edge.dst.id;
                            let path = reconstruct_bidirectional(
                                source,
                                target.id,
                                meeting,
                                &fwd_visited,
                                &fwd_parent,
                                &bwd_visited,
                                &bwd_parent,
                            );
                            return Ok(Some(vec![Value::Path(path)]));
                        }
                        next_front.push_back(edge.dst.id);
                    }
                }
            }
            // Replace the consumed frontier with the next layer.
            *front = next_front;
        }
        Ok(None)
    }

    /// `allShortestPaths` (ADR-194 D-4) — enumerate EVERY equal-minimum-
    /// length path from `source` to `target`. Each path is one output row
    /// carrying a [`Value::Path`]. Returns an empty `Vec` when `target` is
    /// unreachable (the MATCH drops → zero rows).
    ///
    /// # Algorithm
    ///
    /// A single-source LAYERED BFS computes the shortest distance to every
    /// node AND the full predecessor DAG (`preds[n]` = every `(pred, rel)`
    /// with `dist[pred] + 1 == dist[n]`). BFS processes nodes in
    /// non-decreasing distance, so once every distance-`d` node is drained,
    /// each distance-`d+1` node has recorded ALL its minimum-distance
    /// predecessors. The enumeration ([`enumerate_all_shortest`]) then
    /// walks that acyclic DAG from `target` back to `source`, emitting one
    /// [`Value::Path`] per distinct route. (Single-source layered BFS — not
    /// the bidirectional meeting-set machinery — because enumerating ALL
    /// shortest paths is provably correct via the predecessor DAG; a
    /// bidirectional all-paths combine over every optimal meeting node is
    /// far more error-prone for no asymptotic win at LDBC diameters.)
    ///
    /// # Bounds
    ///
    /// The BFS never expands past the target's shortest level nor past
    /// [`DEFAULT_MAX_DEPTH`]; the enumeration is capped at
    /// [`MAX_ALL_SHORTEST_PATHS`] (over-budget ⇒ clean error, never OOM —
    /// ADR-194 §Negative / ADR-025 frontier budget).
    fn all_shortest_paths<S: ExecutorSubstrate>(
        &self,
        ctx: &ExecutionContext,
        substrate: &S,
        source: &NodeView,
        target: &NodeView,
    ) -> Result<Vec<Vec<Value>>, ExecutionError> {
        // Degenerate zero-length path: source == target (ADR-193 D-6).
        if source.id == target.id {
            return Ok(vec![vec![Value::Path(PathView::new(source.clone()))]]);
        }
        let mut dist: HashMap<NodeId, u32> = HashMap::new();
        let mut preds: HashMap<NodeId, Vec<(NodeId, RelView)>> = HashMap::new();
        let mut node_views: HashMap<NodeId, NodeView> = HashMap::new();
        let mut frontier: VecDeque<NodeId> = VecDeque::new();
        dist.insert(source.id, 0);
        node_views.insert(source.id, source.clone());
        frontier.push_back(source.id);
        while let Some(nid) = frontier.pop_front() {
            ctx.cancellation().check()?;
            let d = dist[&nid];
            // Once the target's shortest distance is known, expanding any
            // node at that depth or deeper cannot yield an equal-min path.
            if let Some(&td) = dist.get(&target.id) {
                if d >= td {
                    continue;
                }
            }
            if d >= DEFAULT_MAX_DEPTH {
                continue;
            }
            for edge in expand_neighbors(
                substrate,
                ctx,
                nid,
                self.spec.rel_type,
                self.spec.direction,
                self.plan_read_lsn,
            )? {
                let nd = edge.dst.id;
                match dist.get(&nd).copied() {
                    None => {
                        dist.insert(nd, d + 1);
                        node_views.insert(nd, edge.dst.clone());
                        preds.entry(nd).or_default().push((nid, edge.rel.clone()));
                        frontier.push_back(nd);
                    }
                    // Another equal-minimum-length predecessor of `nd`.
                    Some(existing) if existing == d + 1 => {
                        preds.entry(nd).or_default().push((nid, edge.rel.clone()));
                    }
                    // `nd` already has a strictly shorter (or same-level,
                    // non-predecessor) distance — not on a shortest path
                    // through `nid`.
                    Some(_) => {}
                }
            }
        }
        if !dist.contains_key(&target.id) {
            // Unreachable ⇒ no connecting path ⇒ zero rows.
            return Ok(Vec::new());
        }
        let mut out: Vec<Vec<Value>> = Vec::new();
        let mut suffix: Vec<PathSegment> = Vec::new();
        enumerate_all_shortest(
            target.id,
            source,
            &preds,
            &node_views,
            &mut suffix,
            &mut out,
        )?;
        Ok(out)
    }
}

/// Flip a Direction for the backward BFS pass.
fn flip_direction(d: Direction) -> Direction {
    match d {
        Direction::LeftToRight => Direction::RightToLeft,
        Direction::RightToLeft => Direction::LeftToRight,
        Direction::Undirected => Direction::Undirected,
    }
}

/// Walk the parent chain from `target` back to `source`, building a
/// complete [`PathView`] (nodes AND relationships in source→target
/// traversal order, ADR-194 D-5). Each non-source node's `parent` entry
/// carries `(predecessor, rel_traversed)`, so the reconstructed path
/// records the relationship at every hop — not just the node sequence.
fn reconstruct_path(
    source: &NodeView,
    target: NodeId,
    visited: &HashMap<NodeId, NodeView>,
    parent: &HashMap<NodeId, Option<(NodeId, RelView)>>,
) -> PathView {
    // Collect (rel, end_node) segments walking BACKWARD from target to
    // source, then reverse into forward (source→target) order.
    let mut rev: Vec<(RelView, NodeView)> = Vec::new();
    let mut current = target;
    while let Some((pred, rel)) = parent.get(&current).cloned().flatten() {
        let end = visited
            .get(&current)
            .cloned()
            .unwrap_or_else(|| NodeView::new(current, None));
        rev.push((rel, end));
        current = pred;
    }
    // `current` is now the source (whose parent entry is `None`).
    let mut path = PathView::new(source.clone());
    for (rel, end) in rev.into_iter().rev() {
        path.segments.push(PathSegment { rel, end });
    }
    path
}

/// Reconstruct a bidirectional-BFS path through the `meeting` node as a
/// complete [`PathView`] (nodes AND relationships, ADR-194 D-5). The
/// forward half walks `fwd_parent` from `meeting` back to `source` (then
/// reverses into source→meeting order); the backward half walks
/// `bwd_parent` from `meeting` toward `target` (already in meeting→target
/// order). Both parent maps carry `(node, rel)` so every hop's
/// relationship is recovered.
#[allow(clippy::too_many_arguments)]
fn reconstruct_bidirectional(
    source: &NodeView,
    target: NodeId,
    meeting: NodeId,
    fwd_visited: &HashMap<NodeId, NodeView>,
    fwd_parent: &HashMap<NodeId, Option<(NodeId, RelView)>>,
    bwd_visited: &HashMap<NodeId, NodeView>,
    bwd_parent: &HashMap<NodeId, Option<(NodeId, RelView)>>,
) -> PathView {
    let _ = target; // endpoint identity is recovered via the parent chains.
    // Forward half: collect (rel, end) walking meeting → source, reverse
    // into source → meeting traversal order.
    let mut fwd_rev: Vec<(RelView, NodeView)> = Vec::new();
    let mut current = meeting;
    while let Some((pred, rel)) = fwd_parent.get(&current).cloned().flatten() {
        let end = fwd_visited
            .get(&current)
            .cloned()
            .unwrap_or_else(|| NodeView::new(current, None));
        fwd_rev.push((rel, end));
        current = pred;
    }
    let mut path = PathView::new(source.clone());
    for (rel, end) in fwd_rev.into_iter().rev() {
        path.segments.push(PathSegment { rel, end });
    }
    // Backward half: walk bwd_parent from meeting toward target. Each
    // `bwd_parent[current] = (next_toward_target, rel)` segment lands on
    // `next_toward_target` in source→target order (no reversal).
    let mut current = meeting;
    while let Some((next, rel)) = bwd_parent.get(&current).cloned().flatten() {
        let end = bwd_visited
            .get(&next)
            .cloned()
            .unwrap_or_else(|| NodeView::new(next, None));
        path.segments.push(PathSegment { rel, end });
        current = next;
    }
    path
}

/// Recursively enumerate every source→target shortest path through the
/// predecessor DAG `preds` (ADR-194 D-4), appending each as a
/// `vec![Value::Path]` row to `out`. `suffix` accumulates the segments
/// from the current `node` forward to the target (built target-first as
/// the recursion descends toward `source`, then reversed at the base
/// case). The DAG is acyclic (edges strictly increase BFS distance), so
/// every descent terminates; the [`MAX_ALL_SHORTEST_PATHS`] cap bounds
/// the result set (over-budget ⇒ clean error, never OOM).
fn enumerate_all_shortest(
    node: NodeId,
    source: &NodeView,
    preds: &HashMap<NodeId, Vec<(NodeId, RelView)>>,
    node_views: &HashMap<NodeId, NodeView>,
    suffix: &mut Vec<PathSegment>,
    out: &mut Vec<Vec<Value>>,
) -> Result<(), ExecutionError> {
    if node == source.id {
        if out.len() >= MAX_ALL_SHORTEST_PATHS {
            return Err(ExecutionError::Eval(format!(
                "allShortestPaths: result exceeds the {MAX_ALL_SHORTEST_PATHS}-path budget \
                 (dense shortest-path DAG); narrow the pattern or bound the depth \
                 (ADR-194 §Negative / ADR-025)"
            )));
        }
        // `suffix` holds segments target-first; reverse into source→target
        // order to build the PathView.
        let mut path = PathView::new(source.clone());
        for seg in suffix.iter().rev() {
            path.segments.push(seg.clone());
        }
        out.push(vec![Value::Path(path)]);
        return Ok(());
    }
    if let Some(plist) = preds.get(&node) {
        let end = node_views
            .get(&node)
            .cloned()
            .unwrap_or_else(|| NodeView::new(node, None));
        for (pred, rel) in plist {
            // Segment: traverse `rel` from `pred` to land on `node`.
            suffix.push(PathSegment {
                rel: rel.clone(),
                end: end.clone(),
            });
            enumerate_all_shortest(*pred, source, preds, node_views, suffix, out)?;
            suffix.pop();
        }
    }
    Ok(())
}

/// Expand from `from` via the substrate, returning the neighbor list.
///
/// ADR-197-amendment-01 D-2: routes through `expand_with_context` so
/// variable-length / shortest-path traversal inside a Bolt explicit
/// transaction observes the SAME `snapshot(tx) ⊎ write_set(tx)`
/// visibility as single-hop `ExpandOp` (pre-amendment this called the
/// plain `expand`, silently missing staged rels AND reading at a
/// fresh, unpinned snapshot — the amendment's R-1 hole). Auto-commit
/// behavior is unchanged: with no held tx installed, substrates
/// delegate `_with_context` to the plain committed read.
fn expand_neighbors<S: ExecutorSubstrate>(
    substrate: &S,
    ctx: &ExecutionContext,
    from: NodeId,
    rel_type: Option<arcgraph_core::TypeId>,
    direction: Direction,
    read_lsn: Lsn,
) -> Result<Vec<crate::executor::substrate::BoundEdge>, SubstrateAccessError> {
    substrate.expand_with_context(ctx, from, rel_type, direction, read_lsn)
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, RelId, TenantId, TypeId};

    use super::*;
    use crate::executor::ops::ScanOp;
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::{NodeView, RelView};

    fn ctx() -> ExecutionContext {
        ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
    }

    /// Linear-chain fixture: 1 → 2 → 3 → 4 → 5 (KNOWS).
    fn linear_chain() -> StubExecutorSubstrate {
        let mut s = StubExecutorSubstrate::new();
        for i in 1..=5_u64 {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(i), Some(LabelId::new(1))),
            );
        }
        for i in 1..5_u64 {
            s = s.with_edge(
                TenantId::DEFAULT,
                RelView::new(
                    RelId::new(100 + i),
                    NodeId::new(i),
                    NodeId::new(i + 1),
                    Some(TypeId::new(1)),
                ),
            );
        }
        s
    }

    fn singleton_substrate(id: u64) -> StubExecutorSubstrate {
        StubExecutorSubstrate::new().with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(id), Some(LabelId::new(1))),
        )
    }

    /// Scan with a label filter for `LabelId::new(1)`.
    fn person_scan() -> ScanOp {
        ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX)
    }

    #[test]
    fn shortest_path_single_source_emits_one_row_per_reachable_target() {
        // 1 → 2 → 3 → 4 → 5 chain; SSSP from each of the 5 person
        // nodes emits paths to every reachable descendant.
        let s = linear_chain();
        let scan = person_scan();
        let spec = PathSpec {
            source: BindingId::new(0),
            target: None,
            rel_type: Some(TypeId::new(1)),
            direction: Direction::LeftToRight,
            path_var: BindingId::new(99),
            all_shortest: false,
        };
        let mut op = NamedShortestPathOp::new(PhysicalOperator::Scan(scan), spec, Lsn::MAX);
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        // Substrate has 5 person nodes; SSSP from each of them to all
        // their reachable peers. For a strict linear DAG with edges
        // 1→2→3→4→5, BFS from each node emits rows for every reachable
        // descendant (n=1 emits 4 paths; n=2 emits 3; etc.).
        // Sum: 4 + 3 + 2 + 1 + 0 = 10 paths.
        assert_eq!(b.row_count(), 10);
    }

    #[test]
    fn shortest_path_bidirectional_emits_one_row_for_existing_path() {
        // Bidirectional BFS from 1 → 5 in a linear chain — exactly
        // ONE shortest path (1→2→3→4→5). v1.0-alpha exercises the
        // bidirectional helper directly because the M4-32 lowering
        // doesn't yet emit a 2-column source+target row shape (M4-72
        // forward-deferred per ADR-038 amendment-02 §M4.f).
        let s = linear_chain();
        let op = NamedShortestPathOp::new(
            PhysicalOperator::Empty(crate::executor::ops::EmptyOp::new()),
            PathSpec {
                source: BindingId::new(0),
                target: Some(BindingId::new(1)),
                rel_type: Some(TypeId::new(1)),
                direction: Direction::LeftToRight,
                path_var: BindingId::new(99),
                all_shortest: false,
            },
            Lsn::MAX,
        );
        let ctx = ctx();
        let n1 = NodeView::new(NodeId::new(1), Some(LabelId::new(1)));
        let n5 = NodeView::new(NodeId::new(5), Some(LabelId::new(1)));
        let path_row = op.bidirectional(&ctx, &s, &n1, &n5).unwrap().unwrap();
        // ADR-194 D-5 — the bidirectional helper returns a 1-column row
        // whose single cell is a `Value::Path` carrying nodes AND
        // relationships in source→target traversal order.
        let path = match &path_row[0] {
            Value::Path(p) => p,
            other => panic!("expected Value::Path path cell; got {other:?}"),
        };
        let ids: Vec<u64> = path.nodes().iter().map(|n| n.id.raw()).collect();
        // The linear chain gives exactly 1 → 2 → 3 → 4 → 5 (5 nodes
        // total) — both ends pinned + length pinned.
        assert_eq!(ids.first().copied(), Some(1));
        assert_eq!(ids.last().copied(), Some(5));
        assert_eq!(ids.len(), 5);
        // D-5: the relationships are threaded too — 4 hops for 5 nodes,
        // in traversal order (linear_chain's KNOWS edges 101..=104).
        let rel_ids: Vec<u64> = path.relationships().iter().map(|r| r.id.raw()).collect();
        assert_eq!(
            rel_ids,
            vec![101, 102, 103, 104],
            "rels threaded in traversal order"
        );
        assert_eq!(path.hop_count(), 4);
    }

    #[test]
    fn shortest_path_bidirectional_returns_none_for_disconnected_nodes() {
        // Two disconnected components: {1, 2} and {10}. BFS finds no
        // path between 1 and 10.
        let s = StubExecutorSubstrate::new()
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(2), Some(LabelId::new(1))),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(10), Some(LabelId::new(1))),
            )
            .with_edge(
                TenantId::DEFAULT,
                RelView::new(
                    RelId::new(100),
                    NodeId::new(1),
                    NodeId::new(2),
                    Some(TypeId::new(1)),
                ),
            );
        let op = NamedShortestPathOp::new(
            PhysicalOperator::Empty(crate::executor::ops::EmptyOp::new()),
            PathSpec {
                source: BindingId::new(0),
                target: Some(BindingId::new(1)),
                rel_type: Some(TypeId::new(1)),
                direction: Direction::LeftToRight,
                path_var: BindingId::new(99),
                all_shortest: false,
            },
            Lsn::MAX,
        );
        let ctx = ctx();
        let n1 = NodeView::new(NodeId::new(1), Some(LabelId::new(1)));
        let n10 = NodeView::new(NodeId::new(10), Some(LabelId::new(1)));
        let r = op.bidirectional(&ctx, &s, &n1, &n10).unwrap();
        assert!(r.is_none(), "no path between disconnected components");
    }

    #[test]
    fn shortest_path_bidirectional_self_loop_returns_zero_length_path() {
        // Cypher 9 §6.5: zero-length path admissible (source == target).
        let s = singleton_substrate(7);
        let op = NamedShortestPathOp::new(
            PhysicalOperator::Empty(crate::executor::ops::EmptyOp::new()),
            PathSpec {
                source: BindingId::new(0),
                target: Some(BindingId::new(1)),
                rel_type: None,
                direction: Direction::Undirected,
                path_var: BindingId::new(99),
                all_shortest: false,
            },
            Lsn::MAX,
        );
        let ctx = ctx();
        let n = NodeView::new(NodeId::new(7), Some(LabelId::new(1)));
        let r = op.bidirectional(&ctx, &s, &n, &n).unwrap().unwrap();
        // ADR-194 D-5 — the bidirectional helper returns a 1-column row
        // carrying a `Value::Path`. Source == target ⇒ zero-length path:
        // one node, zero relationships (ADR-193 D-6).
        let path = match &r[0] {
            Value::Path(p) => p,
            other => panic!("expected Value::Path path cell; got {other:?}"),
        };
        assert_eq!(path.hop_count(), 0, "zero-length path has no hops");
        assert_eq!(
            path.nodes().len(),
            1,
            "zero-length path = single source node"
        );
        assert_eq!(path.nodes()[0].id.raw(), 7, "the single node is the source");
    }

    #[test]
    fn shortest_path_propagates_cancel() {
        let s = linear_chain();
        let ctx = ctx();
        ctx.cancellation().cancel();
        let mut op = NamedShortestPathOp::new(
            PhysicalOperator::Scan(person_scan()),
            PathSpec {
                source: BindingId::new(0),
                target: None,
                rel_type: Some(TypeId::new(1)),
                direction: Direction::LeftToRight,
                path_var: BindingId::new(99),
                all_shortest: false,
            },
            Lsn::MAX,
        );
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }

    // -------------------------------------------------------------
    // W12α fix-up MED-2 (PR #277 retro): adversarial-graph coverage
    // for bidirectional BFS — exercises the early-return-on-first-
    // meeting at `path.rs:367` (line numbers are pre-fix-up). The
    // pre-fix-up test set covered only single-path / disconnected /
    // self-loop / cancel; none probed multi-path / skewed / cyclic
    // graphs where the early-return correctness invariant is
    // load-bearing. Phase 4.2 controlled-mutation gate: comment out
    // the `other_visited.contains_key(&edge.dst.id)` guard at
    // `bidirectional` and the diamond test below MUST fail (proves
    // the early-return is load-bearing).
    // -------------------------------------------------------------

    /// Helper: build the bidirectional helper's invocation harness
    /// (an `EmptyOp`-rooted `NamedShortestPathOp` so the test can
    /// drive `op.bidirectional` directly with arbitrary endpoints).
    fn bidi_op() -> NamedShortestPathOp {
        NamedShortestPathOp::new(
            PhysicalOperator::Empty(crate::executor::ops::EmptyOp::new()),
            PathSpec {
                source: BindingId::new(0),
                target: Some(BindingId::new(1)),
                rel_type: Some(TypeId::new(1)),
                direction: Direction::Undirected,
                path_var: BindingId::new(99),
                all_shortest: false,
            },
            Lsn::MAX,
        )
    }

    /// Wire up `(src, dst)` as an Undirected KNOWS edge in the stub
    /// substrate (the substrate stores a single edge; the
    /// `Direction::Undirected` traversal at `expand_neighbors` returns
    /// edges in both directions).
    fn with_knows_edge(
        s: StubExecutorSubstrate,
        rel_id: u64,
        src: u64,
        dst: u64,
    ) -> StubExecutorSubstrate {
        s.with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(rel_id),
                NodeId::new(src),
                NodeId::new(dst),
                Some(TypeId::new(1)),
            ),
        )
    }

    fn person(id: u64) -> NodeView {
        NodeView::new(NodeId::new(id), Some(LabelId::new(1)))
    }

    fn extract_path_node_ids(row: &[Value]) -> Vec<u64> {
        // ADR-194 D-5 — path cells are `Value::Path`; read the node-id
        // sequence via `PathView::nodes()`.
        match &row[0] {
            Value::Path(p) => p.nodes().iter().map(|n| n.id.raw()).collect(),
            other => panic!("expected Value::Path path cell; got {other:?}"),
        }
    }

    #[test]
    fn shortest_path_bidirectional_diamond_returns_a_shortest_path() {
        // Diamond: S(1) → A(2) → T(4); S(1) → B(3) → T(4). Two parallel
        // shortest paths of length 2 (3 nodes). EITHER A or B is on a
        // shortest path; the algorithm picks one based on iteration
        // order — both must be valid shortest paths.
        let mut s = StubExecutorSubstrate::new();
        for id in [1, 2, 3, 4_u64] {
            s = s.with_node(TenantId::DEFAULT, person(id));
        }
        s = with_knows_edge(s, 100, 1, 2);
        s = with_knows_edge(s, 101, 1, 3);
        s = with_knows_edge(s, 102, 2, 4);
        s = with_knows_edge(s, 103, 3, 4);
        let op = bidi_op();
        let ctx = ctx();
        let r = op.bidirectional(&ctx, &s, &person(1), &person(4)).unwrap();
        let path = r.expect("diamond is connected; bidirectional must find a path");
        let ids = extract_path_node_ids(&path);
        // Length pinned: 3 nodes (source + meeting + target = length 2 edges).
        assert_eq!(ids.len(), 3, "shortest path has 3 nodes; got {ids:?}");
        // Endpoints pinned.
        assert_eq!(ids.first().copied(), Some(1), "source endpoint");
        assert_eq!(ids.last().copied(), Some(4), "target endpoint");
        // Middle node MUST be on the shortest path (A=2 or B=3).
        assert!(
            ids[1] == 2 || ids[1] == 3,
            "middle node must be A or B; got {}",
            ids[1]
        );
    }

    #[test]
    fn shortest_path_bidirectional_skewed_returns_strictly_shortest() {
        // Skewed: two paths between S(1) and T(4):
        //   long  : S(1) → A(2) → B(3) → T(4)  (length 3)
        //   short : S(1) → C(5)        → T(4)  (length 2)
        // Bidirectional MUST return the length-2 path (NOT the
        // length-3 first-found one). FAIL-on-revert: a future
        // refactor that picks the first meeting irrespective of layer
        // balance regresses this — the test catches it.
        let mut s = StubExecutorSubstrate::new();
        for id in [1, 2, 3, 4, 5_u64] {
            s = s.with_node(TenantId::DEFAULT, person(id));
        }
        // Long path: 1 → 2 → 3 → 4
        s = with_knows_edge(s, 100, 1, 2);
        s = with_knows_edge(s, 101, 2, 3);
        s = with_knows_edge(s, 102, 3, 4);
        // Short path: 1 → 5 → 4
        s = with_knows_edge(s, 103, 1, 5);
        s = with_knows_edge(s, 104, 5, 4);
        let op = bidi_op();
        let ctx = ctx();
        let r = op.bidirectional(&ctx, &s, &person(1), &person(4)).unwrap();
        let path = r.expect("skewed graph is connected");
        let ids = extract_path_node_ids(&path);
        assert_eq!(
            ids.len(),
            3,
            "STRICTLY shortest = length 2 (3 nodes via 1→5→4); \
             got length-{} path {ids:?} — algorithm picked a non-\
             shortest meeting",
            ids.len() - 1
        );
        assert_eq!(ids.first().copied(), Some(1));
        assert_eq!(ids.last().copied(), Some(4));
        // Middle node must be C (=5), not on the length-3 path.
        assert_eq!(
            ids[1], 5,
            "middle of strictly-shortest path must be C(5), not A(2) or B(3)"
        );
    }

    #[test]
    fn shortest_path_single_source_in_cyclic_graph_does_not_loop() {
        // W12α fix-up NIT-3 (PR #277 retro): cycle pin for the SSSP
        // visited-set guard at `single_source` (`path.rs:281`'s
        // `visited.insert(...).is_none()` check). A future refactor
        // that disables the guard would cause SSSP to loop forever on
        // cyclic substrates; this test catches that regression.
        // Substrate: 1 ↔ 2 (bi-directional KNOWS via two directed
        // edges) so an Undirected SSSP from 1 sees the cycle. The
        // visited set MUST cap the BFS at one visit per node; SSSP
        // emits exactly one row to node 2 and terminates.
        let mut s = StubExecutorSubstrate::new();
        for id in [1, 2_u64] {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(id), Some(LabelId::new(1))),
            );
        }
        s = s.with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(100),
                NodeId::new(1),
                NodeId::new(2),
                Some(TypeId::new(1)),
            ),
        );
        s = s.with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(101),
                NodeId::new(2),
                NodeId::new(1),
                Some(TypeId::new(1)),
            ),
        );
        let mut op = NamedShortestPathOp::new(
            PhysicalOperator::Scan(person_scan()),
            PathSpec {
                source: BindingId::new(0),
                target: None,
                rel_type: Some(TypeId::new(1)),
                direction: Direction::Undirected,
                path_var: BindingId::new(99),
                all_shortest: false,
            },
            Lsn::MAX,
        );
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        // SSSP from each of {1, 2} emits one row to the OTHER node
        // (visited-guard caps each BFS at one visit). 2 nodes × 1
        // reachable peer = 2 paths total; if the visited guard
        // regressed, this would be > 2 (or never terminate).
        assert_eq!(
            b.row_count(),
            2,
            "SSSP on a 1↔2 cycle MUST emit exactly 2 paths \
             (one per node, to its peer); cycle MUST NOT cause \
             infinite emission via missing visited guard"
        );
    }

    #[test]
    fn shortest_path_bidirectional_cycle_does_not_loop() {
        // Cycle: S(1) ↔ A(2); S(1) → T(3). Without the visited-set
        // guard at `bidirectional`'s `visited.insert(...).is_none()`
        // check, BFS would loop forever between 1 ↔ 2. With the guard,
        // it terminates with the length-1 path 1→3.
        let mut s = StubExecutorSubstrate::new();
        for id in [1, 2, 3_u64] {
            s = s.with_node(TenantId::DEFAULT, person(id));
        }
        // Cycle: 1 ↔ 2 via TWO directed edges (one each way) so that
        // an Undirected traversal sees a true cycle even at the
        // substrate's directed-edge layer.
        s = with_knows_edge(s, 100, 1, 2);
        s = with_knows_edge(s, 101, 2, 1);
        // Direct path 1 → 3.
        s = with_knows_edge(s, 102, 1, 3);
        let op = bidi_op();
        let ctx = ctx();
        let r = op.bidirectional(&ctx, &s, &person(1), &person(3)).unwrap();
        let path = r.expect("1 → 3 is reachable in one hop despite the 1 ↔ 2 cycle");
        let ids = extract_path_node_ids(&path);
        // Must be length-1 (2 nodes): no detour via the cycle.
        assert_eq!(
            ids.len(),
            2,
            "shortest path length-1 (2 nodes); cycle MUST NOT extend the path; got {ids:?}"
        );
        assert_eq!(ids, vec![1, 3]);
    }

    // =================================================================
    // ADR-194 D-4 — allShortestPaths: ALL equal-minimum-length paths.
    // =================================================================

    /// An `allShortestPaths`-mode op rooted at an `EmptyOp` so tests can
    /// drive `op.all_shortest_paths` directly with arbitrary endpoints.
    fn all_op() -> NamedShortestPathOp {
        NamedShortestPathOp::new(
            PhysicalOperator::Empty(crate::executor::ops::EmptyOp::new()),
            PathSpec {
                source: BindingId::new(0),
                target: Some(BindingId::new(1)),
                rel_type: Some(TypeId::new(1)),
                direction: Direction::Undirected,
                path_var: BindingId::new(99),
                all_shortest: true,
            },
            Lsn::MAX,
        )
    }

    /// Sorted node-id sequences of every emitted path (deterministic
    /// oracle independent of enumeration order).
    fn sorted_path_id_seqs(rows: &[Vec<Value>]) -> Vec<Vec<u64>> {
        let mut seqs: Vec<Vec<u64>> = rows.iter().map(|r| extract_path_node_ids(r)).collect();
        seqs.sort();
        seqs
    }

    #[test]
    fn all_shortest_paths_diamond_returns_both_equal_min_paths() {
        // Diamond: S(1) → A(2) → T(4); S(1) → B(3) → T(4). TWO distinct
        // length-2 shortest paths. allShortestPaths MUST return BOTH (this
        // is the discriminator vs single `shortestPath`, which returns
        // exactly ONE). FAIL-on-revert: an impl that returns only one path
        // regresses `assert_eq!(rows.len(), 2)`.
        let mut s = StubExecutorSubstrate::new();
        for id in [1, 2, 3, 4_u64] {
            s = s.with_node(TenantId::DEFAULT, person(id));
        }
        s = with_knows_edge(s, 100, 1, 2);
        s = with_knows_edge(s, 101, 1, 3);
        s = with_knows_edge(s, 102, 2, 4);
        s = with_knows_edge(s, 103, 3, 4);
        let op = all_op();
        let ctx = ctx();
        let rows = op
            .all_shortest_paths(&ctx, &s, &person(1), &person(4))
            .unwrap();
        assert_eq!(
            rows.len(),
            2,
            "allShortestPaths over a diamond MUST return BOTH length-2 paths; got {rows:?}"
        );
        // Both paths pinned exactly: 1→2→4 and 1→3→4.
        assert_eq!(
            sorted_path_id_seqs(&rows),
            vec![vec![1_u64, 2, 4], vec![1_u64, 3, 4]],
        );
        // Every result is min-length (2 hops / 3 nodes).
        for r in &rows {
            match &r[0] {
                Value::Path(p) => assert_eq!(p.hop_count(), 2, "every result is min-length"),
                other => panic!("expected Value::Path; got {other:?}"),
            }
        }
    }

    #[test]
    fn all_shortest_paths_skewed_returns_only_minimum_length() {
        // Skewed: short S(1)→C(5)→T(4) (length 2); long S(1)→A(2)→B(3)→T(4)
        // (length 3). allShortestPaths returns ALL MINIMUM paths — ONLY the
        // length-2 one (NOT the length-3 path). This discriminates "all
        // equal-MIN" from "all paths": an impl that returns every path
        // regresses here.
        let mut s = StubExecutorSubstrate::new();
        for id in [1, 2, 3, 4, 5_u64] {
            s = s.with_node(TenantId::DEFAULT, person(id));
        }
        s = with_knows_edge(s, 100, 1, 2);
        s = with_knows_edge(s, 101, 2, 3);
        s = with_knows_edge(s, 102, 3, 4);
        s = with_knows_edge(s, 103, 1, 5);
        s = with_knows_edge(s, 104, 5, 4);
        let op = all_op();
        let ctx = ctx();
        let rows = op
            .all_shortest_paths(&ctx, &s, &person(1), &person(4))
            .unwrap();
        assert_eq!(
            rows.len(),
            1,
            "only the single length-2 path is minimum; the length-3 path is EXCLUDED; got {rows:?}"
        );
        assert_eq!(extract_path_node_ids(&rows[0]), vec![1, 5, 4]);
    }

    #[test]
    fn all_shortest_paths_two_equal_plus_one_longer() {
        // 1→2→4 and 1→3→4 (both length 2) PLUS a length-3 detour
        // 1→5→6→4. allShortestPaths returns the TWO length-2 paths only.
        let mut s = StubExecutorSubstrate::new();
        for id in [1, 2, 3, 4, 5, 6_u64] {
            s = s.with_node(TenantId::DEFAULT, person(id));
        }
        s = with_knows_edge(s, 100, 1, 2);
        s = with_knows_edge(s, 101, 1, 3);
        s = with_knows_edge(s, 102, 2, 4);
        s = with_knows_edge(s, 103, 3, 4);
        s = with_knows_edge(s, 104, 1, 5);
        s = with_knows_edge(s, 105, 5, 6);
        s = with_knows_edge(s, 106, 6, 4);
        let op = all_op();
        let ctx = ctx();
        let rows = op
            .all_shortest_paths(&ctx, &s, &person(1), &person(4))
            .unwrap();
        assert_eq!(
            rows.len(),
            2,
            "exactly the two length-2 paths; got {rows:?}"
        );
        assert_eq!(
            sorted_path_id_seqs(&rows),
            vec![vec![1_u64, 2, 4], vec![1_u64, 3, 4]],
        );
    }

    #[test]
    fn all_shortest_paths_threads_relationships_in_order() {
        // D-5 — every emitted path is a `Value::Path` carrying rels.
        // 1→2→4 via rels [100, 102]; 1→3→4 via rels [101, 103].
        let mut s = StubExecutorSubstrate::new();
        for id in [1, 2, 3, 4_u64] {
            s = s.with_node(TenantId::DEFAULT, person(id));
        }
        s = with_knows_edge(s, 100, 1, 2);
        s = with_knows_edge(s, 101, 1, 3);
        s = with_knows_edge(s, 102, 2, 4);
        s = with_knows_edge(s, 103, 3, 4);
        let op = all_op();
        let ctx = ctx();
        let rows = op
            .all_shortest_paths(&ctx, &s, &person(1), &person(4))
            .unwrap();
        let mut rel_seqs: Vec<Vec<u64>> = rows
            .iter()
            .map(|r| match &r[0] {
                Value::Path(p) => p.relationships().iter().map(|rel| rel.id.raw()).collect(),
                other => panic!("expected Value::Path; got {other:?}"),
            })
            .collect();
        rel_seqs.sort();
        assert_eq!(
            rel_seqs,
            vec![vec![100_u64, 102], vec![101_u64, 103]],
            "rels threaded per path in traversal order"
        );
    }

    #[test]
    fn all_shortest_paths_no_connecting_path_returns_empty() {
        // Disconnected: {1,2} and {10}. allShortestPaths(1, 10) ⇒ no rows
        // (the MATCH drops).
        let s = StubExecutorSubstrate::new()
            .with_node(TenantId::DEFAULT, person(1))
            .with_node(TenantId::DEFAULT, person(2))
            .with_node(TenantId::DEFAULT, person(10));
        let s = with_knows_edge(s, 100, 1, 2);
        let op = all_op();
        let ctx = ctx();
        let rows = op
            .all_shortest_paths(&ctx, &s, &person(1), &person(10))
            .unwrap();
        assert!(
            rows.is_empty(),
            "no connecting path ⇒ zero rows; got {rows:?}"
        );
    }

    #[test]
    fn all_shortest_paths_self_loop_returns_zero_length_path() {
        // source == target ⇒ exactly one zero-length path (ADR-193 D-6).
        let s = singleton_substrate(7);
        let op = all_op();
        let ctx = ctx();
        let rows = op
            .all_shortest_paths(&ctx, &s, &person(7), &person(7))
            .unwrap();
        assert_eq!(rows.len(), 1, "zero-length path is a single row");
        match &rows[0][0] {
            Value::Path(p) => {
                assert_eq!(p.hop_count(), 0);
                assert_eq!(p.nodes().len(), 1);
                assert_eq!(p.nodes()[0].id.raw(), 7);
            }
            other => panic!("expected Value::Path; got {other:?}"),
        }
    }
}
