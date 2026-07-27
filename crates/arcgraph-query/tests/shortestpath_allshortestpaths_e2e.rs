//! **ADR-194 (#619)** — canonical `shortestPath()` / `allShortestPaths()`
//! + the `Value::List`→`Value::Path` migration, END-TO-END.
//!
//! Exercises the FULL pipeline (parse → bind → type-check →
//! cross-substrate → lower → enumerate/plan → execute) for the ADR-194
//! consumer slice, mirroring the `value_path_e2e.rs` (#737) and
//! `shortest_path_target_binding_e2e.rs` (#750) e2e shapes. This file
//! realizes ADR-194's "Active verification" test plan tests 1-7:
//!
//! 1. canonical camelCase `shortestPath()` parses + executes → ONE
//!    `Value::Path` (min-length, src→dst).
//! 2. the `SHORTEST_PATH` macro still parses + executes (D-3 back-compat)
//!    and now returns a `Value::Path` (D-5 migration), NOT `Value::List`.
//! 3. `shortestPath()` == `SHORTEST_PATH()` (D-3 alias — byte-identical
//!    result rows).
//! 4. `allShortestPaths()` returns ALL equal-minimum-length paths (the
//!    headline new capability): a graph with three length-2 shortest
//!    paths → 3 `Value::Path` rows each `length(p) == 2`; a
//!    single-shortest graph → 1 row.
//! 5. `nodes(p)` / `relationships(p)` / `length(p)` over a migrated
//!    `shortestPath` result work identically to a plain named path
//!    (the D-5 single-representation consistency).
//! 6. `Value::List`→`Value::Path` migration REGRESSION: `RETURN
//!    shortestPath(...)` (and `allShortestPaths(...)`) is a `Value::Path`
//!    — assert the TYPE, not just non-error. FAILS on any producer that
//!    still emits a node-only `Value::List`.
//! 7. `allShortestPaths` orderability (ADR-193 D-11 / ADR-194 D-6):
//!    the multi-path result is a deterministic TOTAL order under the
//!    path comparator (`PathView::cmp_paths`, the same arm `ORDER BY`
//!    routes through `compare_orderability`); distinct paths never
//!    collide.
//!
//! All oracles are STRONG `==` over the result rows / exact `Value::Path`
//! shapes.
//!
//! ## Note on test 7's oracle
//!
//! Full-pipeline `RETURN p ORDER BY p` is a PRE-EXISTING executor gap
//! (documented in `value_path_e2e.rs` §"ORDER BY <projected-var>": it
//! fails with "binding … missing from row schema" for ANY key type, not
//! just paths). Test 7 therefore asserts orderability via
//! `PathView::cmp_paths` directly over the real `allShortestPaths`
//! results — the EXACT comparator the sort op invokes through
//! `compare_orderability`'s `Value::Path` arm — so it is a faithful,
//! robust proof of D-6 independent of that orthogonal pipeline gap.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, PathView, RelView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

// `with_labels` assigns LabelIds monotonically from 1 in iteration order;
// `with_rel_types` likewise from 1.
const LABEL_X: u32 = 1; // source label
const LABEL_M: u32 = 2; // intermediate label
const LABEL_Y: u32 = 3; // target label
const LABEL_C: u32 = 4; // an extra reachable non-target label
const TYPE_R: u32 = 1;

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["X", "M", "Y", "C"])
        .with_rel_types(["R"])
        .with_properties(["name"])
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

/// Execute through the FULL [`QueryEngine`] pipeline. Returns result rows.
fn run(query: &str, s: &StubExecutorSubstrate, c: &StubCatalogProvider) -> Vec<Vec<Value>> {
    QueryEngine::new(c)
        .execute(query, s)
        .expect("execute")
        .rows()
        .to_vec()
}

/// Read a path cell's node-id sequence via `PathView::nodes()`
/// (ADR-194 D-5 — the cell is a `Value::Path`). Panics on a non-path
/// cell, so it doubles as the D-5 migration type-assertion.
fn path_node_ids(cell: &Value) -> Vec<u64> {
    match cell {
        Value::Path(p) => p.nodes().iter().map(|n| n.id.raw()).collect(),
        other => panic!("expected Value::Path cell, got {other:?}"),
    }
}

/// Borrow the [`PathView`] out of a path cell (panics on non-path).
fn as_path(cell: &Value) -> &PathView {
    match cell {
        Value::Path(p) => p,
        other => panic!("expected Value::Path cell, got {other:?}"),
    }
}

/// Read a `List(Node)` cell (e.g. `nodes(p)`) as the node-id sequence.
fn list_node_ids(cell: &Value) -> Vec<u64> {
    match cell {
        Value::List(xs) => xs
            .iter()
            .map(|x| match x {
                Value::Node(n) => n.id.raw(),
                other => panic!("expected Value::Node in list, got {other:?}"),
            })
            .collect(),
        other => panic!("expected Value::List cell, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------

/// One shortest path + an extra reachable non-target node:
/// ```text
///   a(1:X) ─R─▶ m(2:M) ─R─▶ b(3:Y)   the shortest a→b path (2 hops)
///   a(1:X) ─R─▶ c(4:C)               extra reachable node (NOT :Y)
/// ```
fn one_target_two_hop() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node_l(1, LABEL_X))
        .with_node(TenantId::DEFAULT, node_l(2, LABEL_M))
        .with_node(TenantId::DEFAULT, node_l(3, LABEL_Y))
        .with_node(TenantId::DEFAULT, node_l(4, LABEL_C))
        .with_edge(TenantId::DEFAULT, edge(100, 1, 2)) // a → m
        .with_edge(TenantId::DEFAULT, edge(101, 2, 3)) // m → b
        .with_edge(TenantId::DEFAULT, edge(102, 1, 4)) // a → c
}

/// THREE distinct length-2 shortest paths a→b, plus a length-3 detour
/// (the discriminator: `allShortestPaths` returns the THREE length-2
/// paths, NOT the detour and NOT just one):
/// ```text
///   a(1:X) ─R─▶ m1(2:M) ─R─▶ b(3:Y)
///   a(1:X) ─R─▶ m2(4:M) ─R─▶ b(3:Y)
///   a(1:X) ─R─▶ m3(5:M) ─R─▶ b(3:Y)
///   a(1:X) ─R─▶ d1(6:C) ─R─▶ d2(7:C) ─R─▶ b(3:Y)   (length-3 detour)
/// ```
fn three_equal_min() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node_l(1, LABEL_X))
        .with_node(TenantId::DEFAULT, node_l(2, LABEL_M))
        .with_node(TenantId::DEFAULT, node_l(3, LABEL_Y))
        .with_node(TenantId::DEFAULT, node_l(4, LABEL_M))
        .with_node(TenantId::DEFAULT, node_l(5, LABEL_M))
        .with_node(TenantId::DEFAULT, node_l(6, LABEL_C))
        .with_node(TenantId::DEFAULT, node_l(7, LABEL_C))
        .with_edge(TenantId::DEFAULT, edge(100, 1, 2)) // a → m1
        .with_edge(TenantId::DEFAULT, edge(101, 2, 3)) // m1 → b
        .with_edge(TenantId::DEFAULT, edge(102, 1, 4)) // a → m2
        .with_edge(TenantId::DEFAULT, edge(103, 4, 3)) // m2 → b
        .with_edge(TenantId::DEFAULT, edge(104, 1, 5)) // a → m3
        .with_edge(TenantId::DEFAULT, edge(105, 5, 3)) // m3 → b
        .with_edge(TenantId::DEFAULT, edge(106, 1, 6)) // a → d1 (detour)
        .with_edge(TenantId::DEFAULT, edge(107, 6, 7)) // d1 → d2
        .with_edge(TenantId::DEFAULT, edge(108, 7, 3)) // d2 → b
}

/// Exactly one shortest path: a(1:X) ─R─▶ b(3:Y).
fn single_min() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node_l(1, LABEL_X))
        .with_node(TenantId::DEFAULT, node_l(3, LABEL_Y))
        .with_edge(TenantId::DEFAULT, edge(100, 1, 3)) // a → b
}

// =====================================================================
// Test 1 — canonical camelCase `shortestPath()` → ONE Value::Path.
// =====================================================================

#[test]
fn test1_canonical_shortest_path_parses_and_executes_one_path() {
    let rows = run(
        "MATCH p = shortestPath((a:X)-[:R*1..5]->(b:Y)) RETURN p",
        &one_target_two_hop(),
        &cat(),
    );
    assert_eq!(
        rows.len(),
        1,
        "canonical shortestPath((a:X)..(b:Y)) returns the SINGLE a→b path, \
         not one-row-per-reachable-node; got {rows:?}"
    );
    // Exact node sequence a(1) → m(2) → b(3); src/dst pinned, length pinned.
    assert_eq!(path_node_ids(&rows[0][0]), vec![1, 2, 3]);
    // And it carries relationships (D-5): 2 hops, rels [100, 101].
    let p = as_path(&rows[0][0]);
    assert_eq!(p.hop_count(), 2, "2-hop min path");
    let rel_ids: Vec<u64> = p.relationships().iter().map(|r| r.id.raw()).collect();
    assert_eq!(rel_ids, vec![100, 101], "rels threaded in traversal order");
}

// =====================================================================
// Test 2 — `SHORTEST_PATH` macro back-compat (D-3) → Value::Path (D-5).
// =====================================================================

#[test]
fn test2_macro_back_compat_now_returns_value_path() {
    let rows = run(
        "MATCH p = SHORTEST_PATH((a:X)-[:R*1..5]->(b:Y)) RETURN p",
        &one_target_two_hop(),
        &cat(),
    );
    assert_eq!(rows.len(), 1, "macro still parses + executes; got {rows:?}");
    // The migration: the macro's output is now a `Value::Path` (NOT the
    // legacy node-only `Value::List`). Assert the TYPE explicitly.
    assert!(
        matches!(&rows[0][0], Value::Path(_)),
        "SHORTEST_PATH macro must now emit Value::Path (D-5 migration), got {:?}",
        rows[0][0]
    );
    assert_eq!(path_node_ids(&rows[0][0]), vec![1, 2, 3]);
}

// =====================================================================
// Test 3 — `shortestPath()` == `SHORTEST_PATH()` (D-3 alias equality).
// =====================================================================

#[test]
fn test3_camel_case_and_macro_produce_identical_results() {
    let s = one_target_two_hop();
    let c = cat();
    let camel = run(
        "MATCH p = shortestPath((a:X)-[:R*1..5]->(b:Y)) RETURN p",
        &s,
        &c,
    );
    let macro_rows = run(
        "MATCH p = SHORTEST_PATH((a:X)-[:R*1..5]->(b:Y)) RETURN p",
        &s,
        &c,
    );
    // Two spellings of the SAME single-shortest algorithm ⇒ byte-identical
    // result rows (same Value::Path, nodes + rels).
    assert_eq!(
        camel, macro_rows,
        "shortestPath and SHORTEST_PATH must produce identical results (D-3 alias)"
    );
}

// =====================================================================
// Test 4 — `allShortestPaths()` returns ALL equal-min paths (headline).
// =====================================================================

#[test]
fn test4_all_shortest_paths_returns_all_equal_min_paths() {
    // (4a) three distinct length-2 paths + a length-3 detour → exactly 3
    // rows, each length 2 (the detour is EXCLUDED; the "all" capability is
    // proven only because there are ≥2 equal-min paths).
    let rows = run(
        "MATCH p = allShortestPaths((a:X)-[:R*1..5]->(b:Y)) RETURN p",
        &three_equal_min(),
        &cat(),
    );
    assert_eq!(
        rows.len(),
        3,
        "three equal-min length-2 paths ⇒ THREE rows (NOT one, NOT the detour); got {rows:?}"
    );
    // Each result is a min-length (2-hop) Value::Path.
    for r in &rows {
        let p = as_path(&r[0]);
        assert_eq!(
            p.hop_count(),
            2,
            "every allShortestPaths result is min-length"
        );
    }
    // The three middle nodes are exactly {2, 4, 5}; all endpoints a(1)→b(3).
    let mut seqs: Vec<Vec<u64>> = rows.iter().map(|r| path_node_ids(&r[0])).collect();
    seqs.sort();
    assert_eq!(
        seqs,
        vec![vec![1, 2, 3], vec![1, 4, 3], vec![1, 5, 3]],
        "the three length-2 paths via m1(2)/m2(4)/m3(5)"
    );

    // (4b) a single-shortest-path graph → exactly ONE row.
    let one = run(
        "MATCH p = allShortestPaths((a:X)-[:R*1..5]->(b:Y)) RETURN p",
        &single_min(),
        &cat(),
    );
    assert_eq!(one.len(), 1, "single shortest path ⇒ one row; got {one:?}");
    assert_eq!(path_node_ids(&one[0][0]), vec![1, 3]);
}

// =====================================================================
// Test 5 — nodes(p)/relationships(p)/length(p) over a migrated path.
// =====================================================================

#[test]
fn test5_path_functions_work_on_migrated_shortest_path() {
    let rows = run(
        "MATCH p = shortestPath((a:X)-[:R*1..5]->(b:Y)) \
         RETURN nodes(p), relationships(p), length(p)",
        &one_target_two_hop(),
        &cat(),
    );
    assert_eq!(rows.len(), 1);
    // nodes(p) = [a(1), m(2), b(3)]; relationships(p) = 2 rels; length = 2.
    assert_eq!(
        list_node_ids(&rows[0][0]),
        vec![1, 2, 3],
        "nodes(p) traversal order"
    );
    match &rows[0][1] {
        Value::List(rels) => {
            let rel_ids: Vec<u64> = rels
                .iter()
                .map(|r| match r {
                    Value::Relationship(rv) => rv.id.raw(),
                    other => panic!("expected Value::Relationship, got {other:?}"),
                })
                .collect();
            assert_eq!(rel_ids, vec![100, 101], "relationships(p) traversal order");
        }
        other => panic!("relationships(p) must be a List, got {other:?}"),
    }
    assert_eq!(rows[0][2], Value::Integer(2), "length(p) = hop count = 2");

    // Shape equivalence with a PLAIN named path over the SAME a→m→b edges:
    // `nodes(shortestPath(...))` == `nodes(p)` for the plain path.
    let plain = run(
        "MATCH p = (a:X)-[:R*2..2]->(b:Y) RETURN nodes(p)",
        &one_target_two_hop(),
        &cat(),
    );
    assert_eq!(plain.len(), 1);
    assert_eq!(
        list_node_ids(&plain[0][0]),
        list_node_ids(&rows[0][0]),
        "nodes(shortestPath(...)) == nodes(plain path) — D-5 single-representation"
    );
}

// =====================================================================
// Test 6 — Value::List→Value::Path migration REGRESSION (assert TYPE).
// =====================================================================

#[test]
fn test6_no_path_producer_emits_value_list() {
    // shortestPath → Value::Path (NOT Value::List).
    let sp = run(
        "MATCH p = shortestPath((a:X)-[:R*1..5]->(b:Y)) RETURN p",
        &one_target_two_hop(),
        &cat(),
    );
    assert!(
        matches!(&sp[0][0], Value::Path(_)),
        "shortestPath result MUST be Value::Path, got {:?}",
        sp[0][0]
    );
    assert!(
        !matches!(&sp[0][0], Value::List(_)),
        "shortestPath MUST NOT emit a node-only Value::List (D-5 regression)"
    );

    // allShortestPaths → every row a Value::Path.
    let asp = run(
        "MATCH p = allShortestPaths((a:X)-[:R*1..5]->(b:Y)) RETURN p",
        &three_equal_min(),
        &cat(),
    );
    assert!(!asp.is_empty());
    for r in &asp {
        assert!(
            matches!(&r[0], Value::Path(_)),
            "allShortestPaths result MUST be Value::Path, got {:?}",
            r[0]
        );
    }

    // Single-source mode (anonymous tail) also emits Value::Path.
    let ss = run(
        "MATCH p = shortestPath((a:X)-[:R*1..5]->()) RETURN p",
        &one_target_two_hop(),
        &cat(),
    );
    assert!(!ss.is_empty(), "single-source enumeration emits rows");
    for r in &ss {
        assert!(
            matches!(&r[0], Value::Path(_)),
            "single-source shortestPath result MUST be Value::Path, got {:?}",
            r[0]
        );
    }
}

// =====================================================================
// Test 7 — allShortestPaths orderability (ADR-193 D-11 / ADR-194 D-6).
// =====================================================================

#[test]
fn test7_all_shortest_paths_orderable_and_distinct() {
    let rows = run(
        "MATCH p = allShortestPaths((a:X)-[:R*1..5]->(b:Y)) RETURN p",
        &three_equal_min(),
        &cat(),
    );
    assert_eq!(rows.len(), 3, "three equal-min paths");

    // Extract the three PathViews and sort with the SAME comparator the
    // sort op routes through (`compare_orderability`'s Value::Path arm
    // calls `PathView::cmp_paths`).
    let mut paths: Vec<PathView> = rows.iter().map(|r| as_path(&r[0]).clone()).collect();
    paths.sort_by(|a, b| a.cmp_paths(b));

    // Deterministic total order: node-id sequence ascending →
    // [1,2,3] < [1,4,3] < [1,5,3].
    let ordered: Vec<Vec<u64>> = paths
        .iter()
        .map(|p| p.nodes().iter().map(|n| n.id.raw()).collect())
        .collect();
    assert_eq!(
        ordered,
        vec![vec![1, 2, 3], vec![1, 4, 3], vec![1, 5, 3]],
        "ORDER BY p over the 3-path result is deterministic (ADR-193 D-11)"
    );

    // Distinct paths NEVER collide: every pairwise compare is non-Equal.
    for i in 0..paths.len() {
        for j in 0..paths.len() {
            let ord = paths[i].cmp_paths(&paths[j]);
            if i == j {
                assert_eq!(ord, std::cmp::Ordering::Equal, "a path equals itself");
            } else {
                assert_ne!(
                    ord,
                    std::cmp::Ordering::Equal,
                    "distinct shortest paths MUST NOT collide under ORDER BY (D-6/D-11)"
                );
            }
        }
    }

    // Every result path has the same (minimum) length — D-6.
    assert!(
        paths.iter().all(|p| p.hop_count() == 2),
        "all allShortestPaths results share the min hop-count (D-6)"
    );
}
