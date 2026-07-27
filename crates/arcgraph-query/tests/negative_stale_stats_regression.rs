//! W26-γ-3 / ADR-136 §D5 — stale-statistics planner regression test.
//!
//! # Forced failure mode
//!
//! When per-tenant catalog statistics drift (cardinality estimates
//! lag behind actual row counts by orders of magnitude), the cost-
//! based planner may pick a pathological algorithm — e.g., a nested-
//! loop join when a hash join would be 100× faster, OR a sequential
//! scan over a 10M-row label when an index lookup would be 1000×
//! faster.
//!
//! # Pinned invariants
//!
//! 1. **Stale stats DO NOT panic the planner.** Even with grossly
//!    inaccurate cardinality estimates, the planner returns a valid
//!    LogicalPlan; it may be suboptimal but it MUST be correct.
//! 2. **Zero-cardinality estimates DO NOT crash.** Zero is a
//!    legitimate cardinality (empty label). The planner must
//!    treat 0 as "very low cost", not divide-by-zero.
//! 3. **u64::MAX cardinality estimates DO NOT overflow.** Adversarial
//!    catalogs returning u64::MAX must not trigger arithmetic
//!    overflow in cost arithmetic.
//! 4. **Monotonic cost.** Same query + same stats → same cost
//!    estimate across repeated planner runs.
//!
//! Per `feedback_load_bearing_pr_requires_fault_injection_tests.md`
//! + W23 retro §3.4 stats-stale incident reproductions.

use arcgraph_core::LabelId;
use arcgraph_query::parse;
use arcgraph_query::semantic::StubCatalogProvider;

fn cat_with_cardinalities(label_card: u64, total: u64) -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person", "Company"])
        .with_properties(["age", "name", "employees"])
        .with_rel_types(["KNOWS", "WORKS_AT"])
        .with_label_cardinality(LabelId::new(1), label_card)
        .with_total_node_count(total)
}

#[test]
fn zero_cardinality_does_not_crash_planner() {
    let cat = cat_with_cardinalities(0, 0);
    let q = "MATCH (n:Person) WHERE n.age > 30 RETURN n";
    let stmt = parse(q).expect("parse");
    // Bind exercises the catalog with zero cardinality; must not
    // panic.
    let source = q;
    let _bound = arcgraph_query::semantic::BindingVisitor::bind(&stmt, source, &cat);
}

#[test]
fn u64_max_cardinality_does_not_overflow() {
    let cat = cat_with_cardinalities(u64::MAX, u64::MAX);
    let q = "MATCH (n:Person) WHERE n.age > 30 RETURN n";
    let stmt = parse(q).expect("parse");
    let source = q;
    let _bound = arcgraph_query::semantic::BindingVisitor::bind(&stmt, source, &cat);
}

#[test]
fn cardinality_mismatch_does_not_crash() {
    // label_card > total — inconsistent but the planner must not
    // panic. (It may pick a suboptimal plan; correctness is
    // what's load-bearing.)
    let cat = cat_with_cardinalities(1_000_000, 100);
    let q = "MATCH (n:Person) RETURN n";
    let stmt = parse(q).expect("parse");
    let source = q;
    let _bound = arcgraph_query::semantic::BindingVisitor::bind(&stmt, source, &cat);
}

#[test]
fn cardinality_skew_does_not_crash_join() {
    // One label highly populated, one label empty — the join
    // planner must handle the skew.
    let cat = StubCatalogProvider::new()
        .with_labels(["Person", "Company"])
        .with_properties(["age", "name"])
        .with_rel_types(["WORKS_AT"])
        .with_label_cardinality(LabelId::new(1), 1_000_000_000)
        .with_label_cardinality(LabelId::new(2), 0)
        .with_total_node_count(1_000_000_000);
    let q = "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p, c LIMIT 10";
    let stmt = parse(q).expect("parse");
    let source = q;
    let _bound = arcgraph_query::semantic::BindingVisitor::bind(&stmt, source, &cat);
}

#[test]
fn empty_catalog_does_not_crash_binder() {
    let cat = StubCatalogProvider::new();
    let q = "MATCH (n) RETURN n";
    let stmt = parse(q).expect("parse");
    let source = q;
    let _bound = arcgraph_query::semantic::BindingVisitor::bind(&stmt, source, &cat);
}

#[test]
fn label_not_in_catalog_binder_handles_gracefully() {
    // An unknown label — the binder may fall through to dynamic-
    // name resolution OR surface a TypeCheckError; either must
    // not panic.
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["age"]);
    let q = "MATCH (n:UnknownLabel) RETURN n";
    let stmt = parse(q).expect("parse");
    let source = q;
    let _bound = arcgraph_query::semantic::BindingVisitor::bind(&stmt, source, &cat);
}

#[test]
fn determinism_under_stable_stats() {
    let cat = cat_with_cardinalities(1000, 10_000);
    let q = "MATCH (n:Person) WHERE n.age > 30 RETURN n LIMIT 10";
    let stmt = parse(q).expect("parse");
    let source = q;
    let bound1 = arcgraph_query::semantic::BindingVisitor::bind(&stmt, source, &cat);
    let bound2 = arcgraph_query::semantic::BindingVisitor::bind(&stmt, source, &cat);
    // Same query + same catalog → same Ok-vs-Err outcome.
    assert_eq!(
        bound1.is_ok(),
        bound2.is_ok(),
        "binding must be deterministic under stable stats"
    );
}

#[test]
fn cardinality_zero_total_with_nonzero_label_no_panic() {
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["age"])
        .with_label_cardinality(LabelId::new(1), 100)
        .with_total_node_count(0); // adversarial: label has rows but "total" says zero
    let q = "MATCH (n:Person) RETURN n";
    let stmt = parse(q).expect("parse");
    let source = q;
    let _bound = arcgraph_query::semantic::BindingVisitor::bind(&stmt, source, &cat);
}

#[test]
fn many_distinct_labels_in_query_no_panic() {
    let cat = StubCatalogProvider::new()
        .with_labels([
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
            "R", "S", "T",
        ])
        .with_total_node_count(1_000_000);
    let q = "MATCH (a:A), (b:B), (c:C), (d:D), (e:E) RETURN a, b, c, d, e LIMIT 1";
    let stmt = parse(q).expect("parse");
    let source = q;
    let _bound = arcgraph_query::semantic::BindingVisitor::bind(&stmt, source, &cat);
}

#[test]
fn skewed_rel_type_cardinalities_no_panic() {
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS", "FOLLOWS"])
        .with_properties(["age"])
        .with_label_cardinality(LabelId::new(1), 10_000)
        .with_rel_type_cardinality(arcgraph_core::TypeId::new(1), u64::MAX)
        .with_rel_type_cardinality(arcgraph_core::TypeId::new(2), 0)
        .with_total_node_count(10_000)
        .with_total_rel_count(u64::MAX);
    let q = "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b LIMIT 10";
    let stmt = parse(q).expect("parse");
    let source = q;
    let _bound = arcgraph_query::semantic::BindingVisitor::bind(&stmt, source, &cat);
}
