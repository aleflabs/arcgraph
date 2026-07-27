//! **ADR-194 D-3a (#619 part-C precursor)** — source→target binding
//! lowering for named shortest-paths, END-TO-END.
//!
//! ## The bug D-3a fixes
//!
//! Before D-3a, `lower_named_path` captured only the path variable +
//! algorithm — never the pattern's tail endpoint — so
//! `LogicalNamedPath` had no `target` and the pipeline hardcoded
//! `PathSpec.target = None`. The `NamedShortestPathOp` therefore ALWAYS
//! ran `single_source()` (one row per reachable node) and its
//! `bidirectional()` source→target BFS was dead code. Concretely,
//! `MATCH p = SHORTEST_PATH((a:X)-[:R*]->(b:Y)) RETURN p` returned ONE
//! ROW PER REACHABLE NODE instead of the single shortest `a→b` path.
//!
//! ## What D-3a lands
//!
//! `lower_named_path` now captures the pattern tail-endpoint node
//! binding into `LogicalNamedPath.target`; `pipeline.rs` threads it into
//! `PathSpec.target`; the executor's existing `bidirectional()` lights
//! up automatically (`PathSpec.target = Some(b)`).
//!
//! ## Oracle (ADR-194 test 0)
//!
//! `test0_*` is the DISCRIMINATING test: over a fixture where `a`
//! reaches `b` (2 hops) AND reaches a non-target node `c`, the query
//! returns EXACTLY ONE ROW (the `a→…→b` path) — NOT one row per
//! reachable node. On the pre-D-3a `target: None` substrate this test
//! is RED (it returns 3 rows). It is therefore the durable fail-on-old
//! guard: reverting the target threading re-breaks `assert_eq!(rows, 1)`.
//!
//! ## Scope note (D-5 migrated)
//!
//! D-3a (#750) shipped the source→target binding; ADR-194 D-5 (this slice,
//! #619) migrated the shortest-path executor's output from a node-only
//! `Value::List` to a full `Value::Path` (nodes AND relationships). The
//! `p` column here is now a `Value::Path`; `path_node_ids` reads the node
//! sequence via `PathView::nodes()` — a strictly stronger check than
//! `nodes(p)` (it pins the whole sequence, not just the projection).
//! `nodes(p)`/`relationships(p)`/`length(p)` over a migrated shortestPath
//! result are exercised in the dedicated D-5 e2e file
//! (`shortestpath_allshortestpaths_e2e.rs`).
//!
//! All oracles are STRONG `==` over the result rows.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

// `with_labels` assigns LabelIds monotonically from 1 in iteration
// order; `with_rel_types` likewise from 1.
const LABEL_X: u32 = 1; // source label
const LABEL_M: u32 = 2; // intermediate label (NOT a target candidate)
const LABEL_Y: u32 = 3; // target label
const LABEL_C: u32 = 4; // an extra reachable node's label (NOT a target)
const TYPE_R: u32 = 1;

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["X", "M", "Y", "C"])
        .with_rel_types(["R"])
}

fn node_l(id: u64, label: u32) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(label)))
}

fn edge(id: u64, from: u64, to: u64) -> RelView {
    RelView::new(
        RelId::new(id),
        NodeId::new(from),
        NodeId::new(to),
        Some(TypeId::new(TYPE_R)),
    )
}

/// Execute through the FULL [`QueryEngine`] pipeline (parse → bind →
/// type-check → cross-substrate → lower → enumerate/plan → execute) —
/// the production path, which exercises the D-3a lowering capture, the
/// enumeration-rewrite target preservation, AND the pipeline
/// wire-through together. Returns the result rows; expects success.
fn run(query: &str, s: &StubExecutorSubstrate, c: &StubCatalogProvider) -> Vec<Vec<Value>> {
    QueryEngine::new(c)
        .execute(query, s)
        .expect("execute")
        .rows()
        .to_vec()
}

/// Read a shortest-path row's path cell (`Value::Path`, ADR-194 D-5) as
/// the node-id sequence in source→target order. Pins the WHOLE path, not
/// just endpoints. (Pre-D-5 this read a node-only `Value::List`; the cell
/// is now a `Value::Path` carrying nodes AND relationships — the node
/// sequence is read via `PathView::nodes()`.)
fn path_node_ids(row: &[Value]) -> Vec<u64> {
    match &row[0] {
        Value::Path(p) => p.nodes().iter().map(|n| n.id.raw()).collect(),
        other => panic!("expected Value::Path path cell, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------

/// The discriminating fixture:
/// ```text
///   a(1:X) ─R─▶ m(2:M) ─R─▶ b(3:Y)   the shortest a→b path (2 hops)
///   a(1:X) ─R─▶ c(4:C)               an extra reachable node (NOT :Y)
/// ```
/// `a` reaches `m`, `b`, and `c` — three nodes. The OLD single-source
/// behavior emits a row per reachable node (3 rows); the FIXED
/// bidirectional behavior emits exactly the one `a→…→b` path (1 row).
fn a_to_b_plus_extra() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node_l(1, LABEL_X))
        .with_node(TenantId::DEFAULT, node_l(2, LABEL_M))
        .with_node(TenantId::DEFAULT, node_l(3, LABEL_Y))
        .with_node(TenantId::DEFAULT, node_l(4, LABEL_C))
        .with_edge(TenantId::DEFAULT, edge(100, 1, 2)) // a → m
        .with_edge(TenantId::DEFAULT, edge(101, 2, 3)) // m → b
        .with_edge(TenantId::DEFAULT, edge(102, 1, 4)) // a → c
}

// =====================================================================
// Test 0 (ADR-194) — TARGET-BINDING: one a→b path, NOT one per node.
// This is ALSO the ADR-133 §D-4 Query-class active-verification recipe
// (exact-row oracle over the LDBC-IC11-shaped shortestPath pattern).
// =====================================================================

#[test]
fn test0_shortest_path_returns_single_a_to_b_path_not_per_reachable_node() {
    let rows = run(
        "MATCH p = SHORTEST_PATH((a:X)-[:R*1..5]->(b:Y)) RETURN p",
        &a_to_b_plus_extra(),
        &cat(),
    );

    // The headline D-3a oracle: EXACTLY ONE ROW. On the pre-D-3a
    // `target: None` substrate this is 3 (single-source emits a row to
    // m, b, and c). Reverting the target threading re-breaks this.
    assert_eq!(
        rows.len(),
        1,
        "shortestPath((a:X)..(b:Y)) MUST return the single a→b path, \
         NOT one row per reachable node; got {} rows: {rows:?}",
        rows.len()
    );

    // Strong oracle: the exact node sequence is a(1) → m(2) → b(3).
    // Endpoints pinned (a and b), length pinned (2 hops / 3 nodes), and
    // the non-target node c(4) is NOT present.
    let ids = path_node_ids(&rows[0]);
    assert_eq!(
        ids,
        vec![1, 2, 3],
        "the single shortest a→b path is a(1)→m(2)→b(3); got {ids:?}"
    );
}

// =====================================================================
// Per-(a,b)-pair semantics — multiple target endpoints each yield ONE
// shortest path (the correct openCypher shape, vs single-source's
// per-reachable-node rows).
// =====================================================================

#[test]
fn shortest_path_emits_one_shortest_path_per_target_endpoint() {
    // a(1:X) ─R─▶ b1(3:Y)             [direct: 1 hop]
    // a(1:X) ─R─▶ m(2:M) ─R─▶ b2(5:Y) [via m: 2 hops]
    let s = StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node_l(1, LABEL_X))
        .with_node(TenantId::DEFAULT, node_l(2, LABEL_M))
        .with_node(TenantId::DEFAULT, node_l(3, LABEL_Y))
        .with_node(TenantId::DEFAULT, node_l(5, LABEL_Y))
        .with_edge(TenantId::DEFAULT, edge(100, 1, 3)) // a → b1
        .with_edge(TenantId::DEFAULT, edge(101, 1, 2)) // a → m
        .with_edge(TenantId::DEFAULT, edge(102, 2, 5)); // m → b2

    let rows = run(
        "MATCH p = SHORTEST_PATH((a:X)-[:R*1..5]->(b:Y)) RETURN p",
        &s,
        &cat(),
    );

    // Two target endpoints (b1=3, b2=5) ⇒ exactly two rows, one
    // shortest path each.
    assert_eq!(
        rows.len(),
        2,
        "two :Y endpoints ⇒ two shortest paths; got {rows:?}"
    );

    let mut paths: Vec<Vec<u64>> = rows.iter().map(|r| path_node_ids(r)).collect();
    paths.sort();
    let mut expected = vec![vec![1_u64, 3], vec![1_u64, 2, 5]];
    expected.sort();
    assert_eq!(
        paths, expected,
        "expected the a→b1 (1 hop) and a→m→b2 (2 hop) shortest paths; got {paths:?}"
    );
}

// =====================================================================
// No-path case — a and b not connected ⇒ zero rows (the MATCH drops).
// =====================================================================

#[test]
fn shortest_path_with_no_connecting_path_yields_zero_rows() {
    // a(1:X) ─R─▶ c(4:C); b(3:Y) is isolated ⇒ a never reaches a :Y node.
    let s = StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node_l(1, LABEL_X))
        .with_node(TenantId::DEFAULT, node_l(3, LABEL_Y))
        .with_node(TenantId::DEFAULT, node_l(4, LABEL_C))
        .with_edge(TenantId::DEFAULT, edge(100, 1, 4)); // a → c only

    let rows = run(
        "MATCH p = SHORTEST_PATH((a:X)-[:R*1..5]->(b:Y)) RETURN p",
        &s,
        &cat(),
    );
    assert_eq!(
        rows.len(),
        0,
        "no a→b connecting path ⇒ zero rows (MATCH drops); got {rows:?}"
    );
}

// =====================================================================
// Single-hop fixed-length pattern (no var-length) — target binding is
// captured regardless of the relationship quantifier.
// =====================================================================

#[test]
fn shortest_path_single_hop_fixed_length_returns_one_row() {
    // a(1:X) ─R─▶ b(3:Y); a(1:X) ─R─▶ c(4:C).
    let s = StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node_l(1, LABEL_X))
        .with_node(TenantId::DEFAULT, node_l(3, LABEL_Y))
        .with_node(TenantId::DEFAULT, node_l(4, LABEL_C))
        .with_edge(TenantId::DEFAULT, edge(100, 1, 3)) // a → b
        .with_edge(TenantId::DEFAULT, edge(101, 1, 4)); // a → c

    let rows = run(
        "MATCH p = SHORTEST_PATH((a:X)-[:R]->(b:Y)) RETURN p",
        &s,
        &cat(),
    );
    assert_eq!(
        rows.len(),
        1,
        "one a→b edge ⇒ one shortest path; got {rows:?}"
    );
    assert_eq!(path_node_ids(&rows[0]), vec![1, 3], "a(1)→b(3)");
}

// =====================================================================
// Lowering-level pin (D-3a) — `lower_named_path` captures BOTH the head
// (source) and tail (target) endpoint bindings, and they are distinct.
// This is independent of the executor: it proves the binding capture
// directly, so it stays green even if the executor representation
// changes (e.g. the D-5 `Value::Path` migration).
// =====================================================================

#[test]
fn lowering_captures_distinct_source_and_target_bindings() {
    use arcgraph_query::logical_plan::{
        LogicalNamedPath, LogicalPlan, LogicalPlanLoweringVisitor, PathAlgorithm,
    };
    use arcgraph_query::parse;
    use arcgraph_query::semantic::{BindingVisitor, CrossSubstrateValidator, TypeCheckVisitor};

    fn find_named_path(p: &LogicalPlan) -> Option<&LogicalNamedPath> {
        match p {
            LogicalPlan::NamedPath(np) => Some(np),
            LogicalPlan::Project(pr) => find_named_path(&pr.input),
            LogicalPlan::Filter(f) => find_named_path(&f.input),
            LogicalPlan::Sort(s) => find_named_path(&s.input),
            LogicalPlan::Distinct(d) => find_named_path(&d.input),
            _ => None,
        }
    }

    let input = "MATCH p = SHORTEST_PATH((a:X)-[:R]->(b:Y)) RETURN p";
    let c = cat();
    let stmt = parse(input).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, input, &c).expect("bind");
    TypeCheckVisitor::check(&mut bound, &c).expect("type-check");
    CrossSubstrateValidator::validate(&bound, &c).expect("validate");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");

    let np = find_named_path(&plan).expect("NamedPath present");
    assert_eq!(np.algorithm, PathAlgorithm::ShortestPath);
    // D-3a: both endpoints captured. Pre-D-3a `target` did not exist
    // (the pipeline hardcoded `PathSpec.target = None`).
    let source = np.source.expect("D-3a captures the head (source) binding");
    let target = np.target.expect("D-3a captures the tail (target) binding");
    // The head (a) and tail (b) of a 2-endpoint pattern are DISTINCT
    // bindings — the precondition for bidirectional source→target BFS.
    assert_ne!(
        source, target,
        "distinct head/tail bindings expected; got source={source:?} target={target:?}"
    );
}
