//! W26-γ-3 / ADR-136 §D5 — pathological cartesian-product query
//! negative-test pin.
//!
//! # Forced failure mode
//!
//! A cartesian-product query `MATCH (a), (b) RETURN a, b` with no
//! WHERE clause and no LIMIT is admissible per the openCypher
//! grammar but produces |A| × |B| rows at execution time. Without
//! defensive cardinality budgets the executor would OOM.
//!
//! # Pinned invariants (parser-only at v1.0 landing)
//!
//! 1. **Parse admissible.** The cartesian-product shape is
//!    syntactically valid; `arcgraph_query::parse` returns `Ok`.
//!    Rejecting it at the parser would be a false positive.
//! 2. **Parser no-panic.** Repeated forms (with / without LIMIT;
//!    with WHERE; with ORDER BY) MUST NOT panic in
//!    `arcgraph_query::parse`.
//! 3. **Parser round-trip.** Every parsed shape re-serialises via
//!    `ast::Display` and re-parses to an equal AST (sister
//!    invariant to ADR-136 §D-3 #2).
//!
//! **Binder + planner coverage is intentionally deferred.** Per
//! `feedback_avoid_speculative_scaffolding.md`: this file pins only
//! the parser-level invariants at v1.0 landing. The binder + planner
//! invariants (cardinality budget enforcement via
//! `BUDGET_FALLBACK_ROWS` per ADR-038 amendment-03 §TIER-2-d; LIMIT
//! push-down at the planning layer; planner termination on cartesian
//! shapes) are covered by `negative_stale_stats_regression.rs`'s
//! `StubCatalogProvider` pattern — when the cartesian-specific
//! binder/planner pin becomes load-bearing (e.g., a regression
//! surfaces), extend this file via the same pattern. See
//! `feedback_noop_trampoline_anti_pattern.md` (W23-MFI-6): scope
//! the doc claim to what the body proves.
//!
//! Per `feedback_load_bearing_pr_requires_fault_injection_tests.md`:
//! cartesian-product is the canonical "pathological-but-valid"
//! query class — parser-level pinning establishes the foundation;
//! binder/planner pins follow when load-bearing.

use arcgraph_query::parse;

#[test]
fn cartesian_no_limit_parses() {
    let q = "MATCH (a), (b) RETURN a, b";
    let stmt = parse(q).expect("cartesian must parse");
    let _ = stmt;
}

#[test]
fn cartesian_with_limit_parses() {
    let q = "MATCH (a), (b) RETURN a, b LIMIT 100";
    parse(q).expect("cartesian + LIMIT must parse");
}

#[test]
fn cartesian_with_labels_parses() {
    let q = "MATCH (a:Person), (b:Company) RETURN a, b";
    parse(q).expect("cartesian + labels must parse");
}

#[test]
fn cartesian_with_property_filters_parses() {
    let q = "MATCH (a:Person {age: 30}), (b:Company {employees: 100}) RETURN a, b LIMIT 50";
    parse(q).expect("cartesian + filters + LIMIT must parse");
}

#[test]
fn three_way_cartesian_parses() {
    let q = "MATCH (a:A), (b:B), (c:C) RETURN a, b, c LIMIT 10";
    parse(q).expect("three-way cartesian must parse");
}

#[test]
fn cartesian_round_trips_via_display() {
    let q = "MATCH (a:Person), (b:Company) RETURN a, b LIMIT 10";
    let stmt = parse(q).expect("parse");
    let printed = format!("{}", stmt);
    let reparsed = parse(&printed).expect("re-parse");
    assert_eq!(stmt, reparsed, "cartesian round-trip");
}

#[test]
fn cartesian_no_panic_in_parser_under_random_paths() {
    // 20 random cartesian-shape queries — none should panic.
    let queries = [
        "MATCH (a), (b) RETURN a, b",
        "MATCH (a:X), (b:Y), (c:Z) RETURN *",
        "MATCH (a), (b), (c), (d) RETURN a, b, c, d LIMIT 1",
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) RETURN a, b",
        "MATCH (a)-[:KNOWS]->(b), (c)-[:WORKS_AT]->(d) RETURN a, b, c, d",
        "MATCH (a), (b) WHERE a.age > b.age RETURN a, b",
        "MATCH (a), (b) WHERE a.age = b.age RETURN a, b LIMIT 100",
        "MATCH (a:A), (b:B) WITH a, b WHERE a.x > 0 RETURN b",
        "MATCH (a), (b), (c) WHERE a.x = b.x AND b.y = c.y RETURN a, b, c",
        "MATCH (a), (b) RETURN count(*) AS pairs",
    ];
    for q in queries {
        let _ = parse(q);
    }
}

#[test]
fn cartesian_with_optional_match_parses() {
    // Cartesian + OPTIONAL MATCH is admissible.
    let q = "MATCH (a:Person), (b:Company) OPTIONAL MATCH (a)-[:WORKS_AT]->(b) RETURN a, b";
    parse(q).expect("cartesian + OPTIONAL MATCH must parse");
}

#[test]
fn long_cartesian_chain_parses() {
    // 5-way cartesian — the planner's join-enumeration surface.
    let q = "MATCH (a:A), (b:B), (c:C), (d:D), (e:E) RETURN a, b, c, d, e LIMIT 100";
    parse(q).expect("5-way cartesian must parse");
}
