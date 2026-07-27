//! M6.2 OOC-4 spillable expand-frontier release gates.
//!
//! The correctness oracle is an independent queue plus per-path relationship
//! set BFS. The bounded gates use a substrate whose owned cursor yields one
//! edge at a time, so an eager adjacency materialization cannot hide behind
//! the frontier probe.

#![cfg(feature = "fault-injection")]

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fs;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, RelId, TenantId, TypeId};
use arcgraph_query::ast::LengthRange;
use arcgraph_query::executor::ops::{
    ExpandOp, ExpandSpillProbe, ExpandSpillTarget, OptionalExpandOp, PhysicalOperator, ScanOp,
};
use arcgraph_query::executor::value::{NodeView, RelView};
use arcgraph_query::executor::{
    BATCH_ROWS, BoundEdge, BoundEdgeCursor, BoundNode, ExecutionContext, ExecutionError,
    ExecutorSpillError, ExecutorSpillFailureKind, ExecutorSubstrate, MemoryBudget, RankedHit,
    SubstrateAccessError, Value, estimate_row_bytes,
};
use arcgraph_query::logical_plan::Direction;
use arcgraph_query::semantic::bound_ast::BindingId;
use arcgraph_storage::spill::{
    SpillEncryptionPolicy, SpillManager, SpillManagerConfig, SpillQuery, SpillQueryConfig,
    SpillRejectReason,
};

const ROOT_LABEL: LabelId = LabelId::new(701);
const OTHER_LABEL: LabelId = LabelId::new(702);
const EDGE_TYPE: TypeId = TypeId::new(703);
const SOURCE: BindingId = BindingId::new(704);
const REL: BindingId = BindingId::new(705);
const DESTINATION: BindingId = BindingId::new(706);
const OPTIONAL_RIGHT: BindingId = BindingId::new(707);
const GENEROUS_SPILL_QUOTA_BYTES: u64 = 512 * 1024 * 1024;
const TERMINATION_TIMEOUT: Duration = Duration::from_secs(15);

static NEXT_SCRATCH_ID: AtomicU64 = AtomicU64::new(1);

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(gate: &str) -> Self {
        let unique = NEXT_SCRATCH_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "arcgraph-m6-expand-{gate}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create expand scratch root");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone, Copy, Debug)]
struct OracleEdge {
    rel_id: u64,
    destination: u64,
}

#[derive(Debug, Default)]
struct CursorMetrics {
    scan_calls: AtomicUsize,
    eager_expand_calls: AtomicUsize,
    eager_expand_peak_rows: AtomicUsize,
    cursor_opens: AtomicUsize,
    cursor_rows_yielded: AtomicUsize,
    max_live_cursor_rows: AtomicUsize,
}

struct GraphState {
    nodes: BTreeMap<u64, NodeView>,
    adjacency: BTreeMap<u64, Arc<Vec<BoundEdge>>>,
    metrics: Arc<CursorMetrics>,
}

#[derive(Clone)]
struct StreamingSubstrate(Arc<GraphState>);

impl StreamingSubstrate {
    fn metrics(&self) -> &CursorMetrics {
        &self.0.metrics
    }

    fn other_label_count(&self) -> usize {
        self.0
            .nodes
            .values()
            .filter(|node| node.label == Some(OTHER_LABEL))
            .count()
    }

    fn validate_request(
        tenant: TenantId,
        direction: Direction,
    ) -> Result<(), SubstrateAccessError> {
        if tenant != TenantId::DEFAULT {
            return Err(SubstrateAccessError::TenantUnknown(tenant));
        }
        if direction != Direction::LeftToRight {
            return Err(SubstrateAccessError::Io(
                "M6 expand gate substrate supports LeftToRight only".to_owned(),
            ));
        }
        Ok(())
    }
}

impl ExecutorSubstrate for StreamingSubstrate {
    fn scan_nodes(
        &self,
        tenant: TenantId,
        label: Option<LabelId>,
        _read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        if tenant != TenantId::DEFAULT {
            return Err(SubstrateAccessError::TenantUnknown(tenant));
        }
        self.0.metrics.scan_calls.fetch_add(1, Ordering::Relaxed);
        Ok(self
            .0
            .nodes
            .values()
            .filter(|node| label.is_none_or(|wanted| node.label == Some(wanted)))
            .cloned()
            .map(|node| BoundNode { node })
            .collect())
    }

    fn expand(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        _read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        Self::validate_request(tenant, direction)?;
        self.0
            .metrics
            .eager_expand_calls
            .fetch_add(1, Ordering::Relaxed);
        let rows = self
            .0
            .adjacency
            .get(&from.raw())
            .into_iter()
            .flat_map(|edges| edges.iter())
            .filter(|edge| rel_type.is_none_or(|wanted| edge.rel.rel_type == Some(wanted)))
            .cloned()
            .collect::<Vec<_>>();
        self.0
            .metrics
            .eager_expand_peak_rows
            .fetch_max(rows.len(), Ordering::Relaxed);
        Ok(rows)
    }

    fn expand_cursor(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        _read_lsn: Lsn,
    ) -> Result<BoundEdgeCursor, SubstrateAccessError> {
        Self::validate_request(tenant, direction)?;
        self.0.metrics.cursor_opens.fetch_add(1, Ordering::Relaxed);
        let edges = self
            .0
            .adjacency
            .get(&from.raw())
            .cloned()
            .unwrap_or_else(|| Arc::new(Vec::new()));
        Ok(Box::new(OwnedAdjacencyCursor {
            edges,
            position: 0,
            rel_type,
            metrics: Arc::clone(&self.0.metrics),
        }))
    }

    fn vector_search(
        &self,
        _tenant: TenantId,
        _property: &str,
        _query_vec: &[f32],
        _k: u64,
        _read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable("vector".to_owned()))
    }

    fn bm25_search(
        &self,
        _tenant: TenantId,
        _property: &str,
        _query_text: &str,
        _k: u64,
        _read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable("bm25".to_owned()))
    }

    fn community_members(
        &self,
        _tenant: TenantId,
        _community_id: i64,
        _read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable(
            "community".to_owned(),
        ))
    }
}

/// Cursor state is an index into substrate-owned adjacency. It never clones an
/// adjacency vector; exactly one BoundEdge is cloned for each `next()` call.
struct OwnedAdjacencyCursor {
    edges: Arc<Vec<BoundEdge>>,
    position: usize,
    rel_type: Option<TypeId>,
    metrics: Arc<CursorMetrics>,
}

impl Iterator for OwnedAdjacencyCursor {
    type Item = Result<BoundEdge, SubstrateAccessError>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(edge) = self.edges.get(self.position) {
            self.position += 1;
            if self
                .rel_type
                .is_some_and(|wanted| edge.rel.rel_type != Some(wanted))
            {
                continue;
            }
            self.metrics
                .max_live_cursor_rows
                .fetch_max(1, Ordering::Relaxed);
            self.metrics
                .cursor_rows_yielded
                .fetch_add(1, Ordering::Relaxed);
            return Some(Ok(edge.clone()));
        }
        None
    }
}

struct Fixture {
    substrate: StreamingSubstrate,
    oracle_adjacency: BTreeMap<u64, Vec<OracleEdge>>,
    root_degree: usize,
    max_single_hop_row_allocation: u64,
    max_depth_one_frontier_allocation: u64,
    disconnected: [u64; 2],
}

fn add_edge(
    adjacency: &mut BTreeMap<u64, Vec<BoundEdge>>,
    oracle: &mut BTreeMap<u64, Vec<OracleEdge>>,
    nodes: &BTreeMap<u64, NodeView>,
    rel_id: u64,
    from: u64,
    to: u64,
) {
    let rel = RelView::new(
        RelId::new(rel_id),
        NodeId::new(from),
        NodeId::new(to),
        Some(EDGE_TYPE),
    );
    let destination = nodes
        .get(&to)
        .cloned()
        .expect("fixture edge destination exists");
    adjacency.entry(from).or_default().push(BoundEdge {
        rel,
        dst: destination,
    });
    oracle.entry(from).or_default().push(OracleEdge {
        rel_id,
        destination: to,
    });
}

fn fanout_fixture(fanout: usize) -> Fixture {
    let root = NodeView::new(NodeId::new(1), Some(ROOT_LABEL));
    let mut nodes = BTreeMap::from([(1_u64, root.clone())]);
    for index in 0..fanout {
        let id = 2_u64 + index as u64;
        nodes.insert(id, NodeView::new(NodeId::new(id), Some(OTHER_LABEL)));
    }
    let second_level = fanout / 2;
    for group in 0..second_level {
        let id = 2_u64 + fanout as u64 + group as u64;
        nodes.insert(id, NodeView::new(NodeId::new(id), Some(OTHER_LABEL)));
    }
    let disconnected = [9_000_000_001_u64, 9_000_000_002_u64];
    for id in disconnected {
        nodes.insert(id, NodeView::new(NodeId::new(id), Some(OTHER_LABEL)));
    }

    let mut adjacency: BTreeMap<u64, Vec<BoundEdge>> = BTreeMap::new();
    let mut oracle_adjacency: BTreeMap<u64, Vec<OracleEdge>> = BTreeMap::new();
    let mut next_rel = 10_000_u64;

    for index in 0..fanout {
        let leaf = 2_u64 + index as u64;
        add_edge(
            &mut adjacency,
            &mut oracle_adjacency,
            &nodes,
            next_rel,
            1,
            leaf,
        );
        next_rel += 1;
    }

    // Adjacent leaves converge on the same second-level node. Existing Expand
    // semantics retain both distinct relationship paths.
    for index in 0..fanout {
        let leaf = 2_u64 + index as u64;
        let child = 2_u64 + fanout as u64 + (index / 2) as u64;
        add_edge(
            &mut adjacency,
            &mut oracle_adjacency,
            &nodes,
            next_rel,
            leaf,
            child,
        );
        next_rel += 1;
    }

    // Distinct back-edge relationships form real cycles without changing the
    // per-path relationship-uniqueness contract.
    for index in (0..fanout).step_by(257) {
        let leaf = 2_u64 + index as u64;
        add_edge(
            &mut adjacency,
            &mut oracle_adjacency,
            &nodes,
            next_rel,
            leaf,
            1,
        );
        next_rel += 1;
    }

    // A self-loop makes the per-path HashSet load-bearing: at depth two the
    // same relationship must be pruned, while every other root edge remains
    // a valid distinct path after the loop.
    add_edge(
        &mut adjacency,
        &mut oracle_adjacency,
        &nodes,
        next_rel,
        1,
        1,
    );
    next_rel += 1;

    // Disconnected component: present in the substrate and oracle graph, but
    // unreachable from the sole ROOT_LABEL scan row.
    add_edge(
        &mut adjacency,
        &mut oracle_adjacency,
        &nodes,
        next_rel,
        disconnected[0],
        disconnected[1],
    );

    for edges in adjacency.values_mut() {
        edges.sort_by_key(|edge| edge.rel.id.raw());
    }
    for edges in oracle_adjacency.values_mut() {
        edges.sort_by_key(|edge| edge.rel_id);
    }

    let root_edges = adjacency.get(&1).expect("root has fixture edges");
    let max_single_hop_row_allocation = root_edges
        .iter()
        .map(|edge| {
            let row = vec![
                Value::Node(root.clone()),
                Value::Relationship(edge.rel.clone()),
                Value::Node(edge.dst.clone()),
            ];
            (estimate_row_bytes(&row).saturating_add(size_of::<Vec<Value>>())) as u64
        })
        .max()
        .expect("root fanout is non-empty");
    let max_depth_one_frontier_allocation = root_edges
        .iter()
        .map(|edge| {
            let row = vec![
                Value::Node(edge.dst.clone()),
                Value::List(vec![Value::Relationship(edge.rel.clone())]),
                Value::Integer(1),
            ];
            (estimate_row_bytes(&row).saturating_add(size_of::<Vec<Value>>())) as u64
        })
        .max()
        .expect("root frontier is non-empty");
    let root_degree = root_edges.len();
    let metrics = Arc::new(CursorMetrics::default());
    let adjacency = adjacency
        .into_iter()
        .map(|(node, edges)| (node, Arc::new(edges)))
        .collect();
    let substrate = StreamingSubstrate(Arc::new(GraphState {
        nodes,
        adjacency,
        metrics,
    }));

    Fixture {
        substrate,
        oracle_adjacency,
        root_degree,
        max_single_hop_row_allocation,
        max_depth_one_frontier_allocation,
        disconnected,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExpandSignature {
    source: u64,
    rels: Vec<u64>,
    destination: u64,
}

struct OracleState {
    node: u64,
    depth: u32,
    rels: Vec<u64>,
    used_rels: HashSet<u64>,
}

/// Independent in-memory BFS oracle for the shipped Expand contract: FIFO
/// path states, per-path relationship uniqueness, and node revisits allowed.
fn independent_expand_bfs(
    adjacency: &BTreeMap<u64, Vec<OracleEdge>>,
    min_depth: u32,
    max_depth: u32,
) -> Vec<ExpandSignature> {
    let mut queue = VecDeque::from([OracleState {
        node: 1,
        depth: 0,
        rels: Vec::new(),
        used_rels: HashSet::new(),
    }]);
    let mut output = Vec::new();
    while let Some(state) = queue.pop_front() {
        if state.depth >= max_depth {
            continue;
        }
        if let Some(edges) = adjacency.get(&state.node) {
            for edge in edges {
                if state.used_rels.contains(&edge.rel_id) {
                    continue;
                }
                let mut rels = state.rels.clone();
                rels.push(edge.rel_id);
                let mut used_rels = state.used_rels.clone();
                used_rels.insert(edge.rel_id);
                let depth = state.depth + 1;
                if depth >= min_depth {
                    output.push(ExpandSignature {
                        source: 1,
                        rels: rels.clone(),
                        destination: edge.destination,
                    });
                }
                queue.push_back(OracleState {
                    node: edge.destination,
                    depth,
                    rels,
                    used_rels,
                });
            }
        }
    }
    output
}

fn encrypted_query(
    manager: &SpillManager,
    query_id: u64,
    executor_budget_bytes: u64,
    quota_bytes: u64,
) -> SpillQuery {
    let mut config = SpillQueryConfig::new(TenantId::DEFAULT, query_id, 0, executor_budget_bytes);
    config.spill_quota_bytes = Some(quota_bytes);
    config.encryption = SpillEncryptionPolicy {
        tenant_encryption_enabled: false,
        force_encryption: true,
    };
    manager
        .begin_query(config)
        .expect("begin encrypted expand spill query")
}

fn build_expand(
    query: SpillQuery,
    probe: ExpandSpillProbe,
    length_range: Option<LengthRange>,
) -> ExpandOp {
    let target = ExpandSpillTarget::new(query).with_probe(probe);
    ExpandOp::new(
        PhysicalOperator::Scan(ScanOp::new(SOURCE, Some(ROOT_LABEL), Lsn::MAX)),
        SOURCE,
        Some(REL),
        DESTINATION,
        Some(EDGE_TYPE),
        Direction::LeftToRight,
        length_range,
        Lsn::MAX,
    )
    .expect("build ExpandOp")
    .with_spillover_target(Some(target))
    .expect("attach expand spill target")
}

fn configured_context(budget_bytes: u64) -> ExecutionContext {
    ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO).with_budget(
        MemoryBudget::with_per_tenant_cap(TenantId::DEFAULT, budget_bytes),
    )
}

fn signature(row: &[Value]) -> ExpandSignature {
    let source = match row.first() {
        Some(Value::Node(node)) => node.id.raw(),
        other => panic!("expand row has no source Node: {other:?}"),
    };
    let rels = match row.get(1) {
        Some(Value::Relationship(rel)) => vec![rel.id.raw()],
        Some(Value::List(values)) => values
            .iter()
            .map(|value| match value {
                Value::Relationship(rel) => rel.id.raw(),
                other => panic!("var-length rel list has non-rel value: {other:?}"),
            })
            .collect(),
        other => panic!("expand row has no relationship binding: {other:?}"),
    };
    let destination = match row.get(2) {
        Some(Value::Node(node)) => node.id.raw(),
        other => panic!("expand row has no destination Node: {other:?}"),
    };
    ExpandSignature {
        source,
        rels,
        destination,
    }
}

fn drain_expand(
    op: &mut ExpandOp,
    ctx: &ExecutionContext,
    substrate: &StreamingSubstrate,
) -> Result<Vec<ExpandSignature>, ExecutionError> {
    let mut output = Vec::new();
    loop {
        let batch = op.next_batch(ctx, substrate)?;
        if batch.is_empty() {
            break;
        }
        output.extend(batch.rows().iter().map(|row| signature(row)));
    }
    Ok(output)
}

fn drain_expand_count(
    op: &mut ExpandOp,
    ctx: &ExecutionContext,
    substrate: &StreamingSubstrate,
) -> Result<usize, ExecutionError> {
    let mut count = 0_usize;
    loop {
        let batch = op.next_batch(ctx, substrate)?;
        if batch.is_empty() {
            break;
        }
        for row in batch.rows() {
            let row_signature = signature(row);
            assert_eq!(row_signature.source, 1);
            assert_eq!(row_signature.rels.len(), 1);
            count = count.saturating_add(1);
        }
    }
    Ok(count)
}

fn assert_bounded_single_hop(
    fixture: &Fixture,
    stats: arcgraph_query::executor::ops::ExpandSpillStats,
    budget_bytes: u64,
) {
    let one_batch_slack = fixture
        .max_single_hop_row_allocation
        .saturating_mul(BATCH_ROWS as u64);
    let hard_bound = budget_bytes.saturating_add(one_batch_slack);
    let logical_frontier_bytes = fixture
        .max_single_hop_row_allocation
        .saturating_mul(fixture.root_degree as u64);
    assert!(
        logical_frontier_bytes > hard_bound.saturating_mul(4),
        "logical supernode frontier is not materially larger than the hard bound: logical={logical_frontier_bytes}, bound={hard_bound}"
    );
    assert!(
        stats.peak_batch_slack_bytes <= one_batch_slack,
        "reader+writer staging exceeded one measured BATCH_ROWS slack: {stats:?}, slack={one_batch_slack}"
    );
    assert!(
        stats.peak_batch_slack_bytes <= stats.batch_slack_limit_bytes,
        "published queue slack limit regressed below observed occupancy: {stats:?}"
    );
    assert!(
        stats.peak_frontier_bytes <= hard_bound,
        "measured FIFO exceeded budget + one fixed measured batch: {stats:?}, bound={hard_bound}"
    );
    assert!(
        stats.rows_spilled > fixture.root_degree as u64 / 2,
        "fixture did not spill most of the supernode frontier: {stats:?}"
    );
    assert_eq!(
        stats.rows_rehydrated, stats.rows_spilled,
        "every spilled row must be restored exactly once: {stats:?}"
    );
    assert!(
        stats.runs_created > 0,
        "no spill run was drained: {stats:?}"
    );

    let metrics = fixture.substrate.metrics();
    assert_eq!(
        metrics.eager_expand_calls.load(Ordering::Relaxed),
        0,
        "single-hop gate fell back to eager expand"
    );
    assert_eq!(
        metrics.eager_expand_peak_rows.load(Ordering::Relaxed),
        0,
        "cursor gate materialized an eager adjacency Vec"
    );
    assert_eq!(
        metrics.cursor_opens.load(Ordering::Relaxed),
        1,
        "single root should open one cursor"
    );
    assert_eq!(
        metrics.max_live_cursor_rows.load(Ordering::Relaxed),
        1,
        "owned cursor buffered more than one edge"
    );
    assert_eq!(
        metrics.cursor_rows_yielded.load(Ordering::Relaxed),
        fixture.root_degree,
        "cursor did not stream the complete root fanout"
    );
}

fn assert_bounded_depth_two(
    fixture: &Fixture,
    stats: arcgraph_query::executor::ops::ExpandSpillStats,
    budget_bytes: u64,
) {
    // The production FIFO stores only depth-one states for a max-depth-two
    // traversal. This bound is computed from an independently constructed
    // maximum-sized depth-one state, not from the observed peak under test.
    let one_batch_slack = fixture
        .max_depth_one_frontier_allocation
        .saturating_mul(BATCH_ROWS as u64);
    let hard_bound = budget_bytes.saturating_add(one_batch_slack);
    let logical_frontier_bytes = fixture
        .max_depth_one_frontier_allocation
        .saturating_mul(fixture.root_degree as u64);
    assert!(
        logical_frontier_bytes > hard_bound.saturating_mul(4),
        "logical depth-one frontier is not materially larger than the hard bound: logical={logical_frontier_bytes}, bound={hard_bound}"
    );
    assert!(
        stats.peak_batch_slack_bytes <= one_batch_slack,
        "depth-two reader+writer staging exceeded one fixed measured batch: {stats:?}, slack={one_batch_slack}"
    );
    assert!(
        stats.peak_batch_slack_bytes <= stats.batch_slack_limit_bytes,
        "published depth-two slack limit regressed below observed occupancy: {stats:?}"
    );
    assert!(
        stats.peak_frontier_bytes <= hard_bound,
        "depth-two FIFO exceeded budget + one fixed measured batch: {stats:?}, bound={hard_bound}"
    );
    assert!(
        stats.rows_spilled > fixture.root_degree as u64 / 2,
        "depth-two supernode did not spill most of its frontier: {stats:?}"
    );
    assert_eq!(
        stats.rows_rehydrated, stats.rows_spilled,
        "every spilled depth-one state must be restored exactly once: {stats:?}"
    );
    assert!(
        stats.runs_created > 0,
        "depth-two traversal never drained a spill run: {stats:?}"
    );

    let root_edges = fixture
        .oracle_adjacency
        .get(&1)
        .expect("fixture root adjacency exists");
    let expected_cursor_opens = 1_u64.saturating_add(root_edges.len() as u64);
    let expected_rows_yielded = root_edges
        .iter()
        .fold(root_edges.len() as u64, |total, edge| {
            total.saturating_add(
                fixture
                    .oracle_adjacency
                    .get(&edge.destination)
                    .map_or(0, Vec::len) as u64,
            )
        });
    let metrics = fixture.substrate.metrics();
    assert_eq!(
        metrics.eager_expand_calls.load(Ordering::Relaxed),
        0,
        "depth-two gate fell back to eager adjacency materialization"
    );
    assert_eq!(
        metrics.eager_expand_peak_rows.load(Ordering::Relaxed),
        0,
        "depth-two gate materialized an eager adjacency Vec"
    );
    assert_eq!(
        metrics.cursor_opens.load(Ordering::Relaxed) as u64,
        expected_cursor_opens,
        "depth-two BFS opened the wrong number of level cursors"
    );
    assert_eq!(
        metrics.max_live_cursor_rows.load(Ordering::Relaxed),
        1,
        "depth-two BFS cursor buffered more than one edge"
    );
    assert_eq!(
        metrics.cursor_rows_yielded.load(Ordering::Relaxed) as u64,
        expected_rows_yielded,
        "depth-two BFS did not stream every expected adjacency row"
    );
}

#[test]
fn m6_expand_correctness_vs_inmemory_oracle() {
    let scratch = ScratchDir::new("correctness");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 0, false)
            .expect("create spill manager");
    let fixture = fanout_fixture(BATCH_ROWS + 700);
    let oracle = independent_expand_bfs(&fixture.oracle_adjacency, 1, 2);
    assert!(
        oracle
            .iter()
            .all(|row| !fixture.disconnected.contains(&row.destination)),
        "independent oracle reached disconnected component"
    );

    let budget_bytes = 4 * 1024;
    let ctx = configured_context(budget_bytes);
    let probe = ExpandSpillProbe::new();
    let query = encrypted_query(
        &manager,
        0xE4A0_0001,
        budget_bytes,
        GENEROUS_SPILL_QUOTA_BYTES,
    );
    let mut expand = build_expand(
        query,
        probe.clone(),
        Some(LengthRange::Cypher {
            min: 1,
            max: Some(2),
        }),
    );
    let actual = drain_expand(&mut expand, &ctx, &fixture.substrate)
        .expect("forced-spill variable-length expand succeeds");

    assert_eq!(
        actual, oracle,
        "spilled Expand order/multiplicity differs from independent per-path BFS"
    );
    let stats = probe.snapshot();
    assert!(
        stats.peak_resident_frontier_bytes > 0,
        "FIFO inversion control needs a resident prefix: {stats:?}"
    );
    assert!(
        stats.rows_spilled > 0,
        "fixture did not force spill: {stats:?}"
    );
    assert!(stats.runs_created > 0, "no FIFO run was sealed: {stats:?}");
    assert_eq!(stats.rows_rehydrated, stats.rows_spilled, "{stats:?}");
}

#[test]
fn m6_expand_bounded_peak_rss() {
    let scratch = ScratchDir::new("bounded-peak");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 0, false)
            .expect("create spill manager");
    let fixture = fanout_fixture(12_000);
    let budget_bytes = 4 * 1024;
    let ctx = configured_context(budget_bytes);
    let probe = ExpandSpillProbe::new();
    let query = encrypted_query(
        &manager,
        0xE4A0_0002,
        budget_bytes,
        GENEROUS_SPILL_QUOTA_BYTES,
    );
    let mut expand = build_expand(query, probe.clone(), None);
    let count = drain_expand_count(&mut expand, &ctx, &fixture.substrate)
        .expect("bounded single-hop expand succeeds");
    assert_eq!(count, fixture.root_degree, "single-hop expand lost rows");
    assert_bounded_single_hop(&fixture, probe.snapshot(), budget_bytes);
}

#[test]
fn m6_expand_supernode_terminates() {
    let scratch = ScratchDir::new("supernode");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 0, false)
            .expect("create spill manager");
    let fixture = fanout_fixture(16_000);
    let budget_bytes = 2 * 1024;
    let probe = ExpandSpillProbe::new();
    let query = encrypted_query(
        &manager,
        0xE4A0_0003,
        budget_bytes,
        GENEROUS_SPILL_QUOTA_BYTES,
    );
    let mut expand = build_expand(
        query,
        probe.clone(),
        Some(LengthRange::Cypher {
            min: 1,
            max: Some(2),
        }),
    );
    let oracle = independent_expand_bfs(&fixture.oracle_adjacency, 1, 2);
    assert!(
        oracle
            .iter()
            .any(|row| row.rels.len() == 2 && row.destination == 1),
        "fixture cycle is not exercised at depth two"
    );
    assert!(
        oracle.len() > fixture.root_degree.saturating_mul(2),
        "fixture does not exercise a second BFS level"
    );
    let substrate = fixture.substrate.clone();
    let (sender, receiver) = mpsc::sync_channel(1);
    let worker = std::thread::spawn(move || {
        // ExecutionContext intentionally stays thread-local: its held-merge-
        // guard slot is not Send, while the operator and substrate are.
        let ctx = configured_context(budget_bytes);
        let result = drain_expand(&mut expand, &ctx, &substrate);
        let _ = sender.send(result);
    });
    let result = match receiver.recv_timeout(TERMINATION_TIMEOUT) {
        Ok(result) => result,
        Err(error) => {
            drop(worker);
            panic!(
                "cyclic supernode expand did not terminate within {TERMINATION_TIMEOUT:?}: {error}"
            );
        }
    };
    worker.join().expect("supernode expand worker panicked");
    let actual = result.expect("cyclic supernode expand succeeds");
    assert_eq!(
        actual, oracle,
        "cyclic spilled supernode differs from independent ordered BFS"
    );
    assert_bounded_depth_two(&fixture, probe.snapshot(), budget_bytes);
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ScratchEntryCounts {
    files: u64,
    directories: u64,
}

fn scratch_entry_counts(root: &Path) -> ScratchEntryCounts {
    let Ok(entries) = fs::read_dir(root) else {
        return ScratchEntryCounts::default();
    };
    let mut counts = ScratchEntryCounts::default();
    for entry in entries {
        let entry = entry.expect("read expand scratch entry");
        if entry
            .file_type()
            .expect("read expand scratch file type")
            .is_dir()
        {
            counts.directories = counts.directories.saturating_add(1);
            let descendants = scratch_entry_counts(&entry.path());
            counts.files = counts.files.saturating_add(descendants.files);
            counts.directories = counts.directories.saturating_add(descendants.directories);
        } else {
            counts.files = counts.files.saturating_add(1);
        }
    }
    counts
}

#[test]
fn m6_expand_quota_abort_clean() {
    let scratch = ScratchDir::new("quota-abort");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 8, false)
            .expect("create retaining spill manager");
    let allocation_unit = manager
        .volume_space()
        .expect("measure scratch allocation unit")
        .allocation_unit_bytes;
    let quota_bytes = allocation_unit.saturating_mul(5);
    let budget_bytes = 1;
    let ctx = configured_context(budget_bytes);
    let fixture = fanout_fixture(6_000);
    let query = encrypted_query(&manager, 0xE4A0_0004, budget_bytes, quota_bytes);
    let key_probe = query
        .key_zeroize_probe_for_test()
        .expect("forced encryption owns an ephemeral spill key");
    assert!(!key_probe.is_zeroized(), "live query key starts non-zero");
    let mut expand = build_expand(query, ExpandSpillProbe::new(), None);

    let error = expand
        .next_batch(&ctx, &fixture.substrate)
        .expect_err("expand frontier must breach tiny tenant quota");
    match error {
        ExecutionError::Spill(ExecutorSpillError::ResourceExhausted {
            reason: SpillRejectReason::TenantQuota,
            requested_bytes,
            spilled_bytes,
            limit_bytes,
            ..
        }) => {
            assert!(requested_bytes > 0, "quota reject reports measured delta");
            assert!(spilled_bytes > 0, "a frontier frame was charged first");
            assert_eq!(limit_bytes, quota_bytes);
        }
        other => panic!("expected typed TenantQuota executor error, got {other:?}"),
    }
    assert_eq!(
        manager.spilled_bytes(TenantId::DEFAULT),
        0,
        "abort must drop every frontier run/query quota guard"
    );
    assert!(key_probe.is_zeroized(), "abort must zeroize the query key");
    let retained = scratch_entry_counts(manager.spill_root());
    assert!(retained.files >= 1, "retention hook exposed no run file");
    assert!(
        retained.directories >= 1,
        "retention hook exposed no scratch directory"
    );
    let sweep = manager
        .periodic_sweep()
        .expect("sweep aborted expand scratch");
    assert!(sweep.removed_files >= 1, "sweep removed no retained run");
    assert!(
        sweep.removed_directories >= 1,
        "sweep removed no orphan directory"
    );
    assert_eq!(
        scratch_entry_counts(manager.spill_root()),
        ScratchEntryCounts::default(),
        "quota abort left orphan expand scratch after periodic sweep"
    );
}

#[test]
fn expand_spill_requires_configured_budget_before_pull() {
    let scratch = ScratchDir::new("precondition");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 0, false)
            .expect("create spill manager");
    let fixture = fanout_fixture(8);
    let query = encrypted_query(&manager, 0xE4A0_0005, 4 * 1024, GENEROUS_SPILL_QUOTA_BYTES);
    let key_probe = query
        .key_zeroize_probe_for_test()
        .expect("forced encryption owns an ephemeral spill key");
    let mut expand = build_expand(query, ExpandSpillProbe::new(), None);
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

    let error = expand
        .next_batch(&ctx, &fixture.substrate)
        .expect_err("uncapped spillable expand must reject before pulling input");
    assert!(
        matches!(
            error,
            ExecutionError::Spill(ExecutorSpillError::Failure {
                kind: ExecutorSpillFailureKind::InvalidConfig,
                ..
            })
        ),
        "uncapped precondition returned the wrong typed error: {error:?}"
    );
    assert_eq!(
        fixture
            .substrate
            .metrics()
            .scan_calls
            .load(Ordering::Relaxed),
        0,
        "precondition must reject before child scan"
    );
    assert!(
        key_probe.is_zeroized(),
        "precondition abort must zeroize key"
    );
}

#[test]
fn optional_expand_forced_spill_symmetry() {
    let scratch = ScratchDir::new("optional-symmetry");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 0, false)
            .expect("create spill manager");
    let fixture = fanout_fixture(BATCH_ROWS + 500);
    let expected = fixture.substrate.other_label_count();
    let budget_bytes = 4 * 1024;
    let ctx = configured_context(budget_bytes);
    let probe = ExpandSpillProbe::new();
    let query = encrypted_query(
        &manager,
        0xE4A0_0006,
        budget_bytes,
        GENEROUS_SPILL_QUOTA_BYTES,
    );
    let target = ExpandSpillTarget::new(query).with_probe(probe.clone());
    let mut optional = OptionalExpandOp::new(
        PhysicalOperator::Scan(ScanOp::new(SOURCE, Some(ROOT_LABEL), Lsn::MAX)),
        vec![OPTIONAL_RIGHT],
        |_| PhysicalOperator::Scan(ScanOp::new(OPTIONAL_RIGHT, Some(OTHER_LABEL), Lsn::MAX)),
    )
    .with_spillover_target(Some(target))
    .expect("attach OptionalExpand spill target");

    let mut count = 0_usize;
    loop {
        let batch = optional
            .next_batch(&ctx, &fixture.substrate)
            .expect("forced-spill OptionalExpand succeeds");
        if batch.is_empty() {
            break;
        }
        count = count.saturating_add(batch.row_count());
    }
    assert_eq!(count, expected, "OptionalExpand lost right-side rows");
    let stats = probe.snapshot();
    assert!(
        stats.rows_spilled > 0,
        "OptionalExpand did not spill: {stats:?}"
    );
    assert_eq!(stats.rows_rehydrated, stats.rows_spilled, "{stats:?}");
}
