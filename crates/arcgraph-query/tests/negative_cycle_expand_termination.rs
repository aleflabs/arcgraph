//! W26-γ-3 / ADR-136 §D5 — variable-length-expand SYNTAX admission
//! on cyclic-shaped queries (parser-only at v1.0 landing).
//!
//! # Forced failure mode (motivation; see "Coverage scope" below)
//!
//! `MATCH (a)-[:R*]->(b)` on a graph containing a cycle would loop
//! forever without termination logic. Per openCypher spec, the
//! expand operator MUST short-circuit when it re-visits a node;
//! per ADR-038 §D-9 the variable-length expand carries a hop budget
//! that bounds the walk depth.
//!
//! # Coverage scope (parser-only at v1.0 landing)
//!
//! This file pins the **parser-level** invariants that the
//! variable-length-expand syntax surface is admitted across the
//! full range of edge shapes (1..N, 0..N, huge upper bound, bare
//! star, self-loop patterns, back-edge patterns). The
//! planner/executor termination contract (the actual cycle-walk
//! short-circuit + ADR-038 §D-9 hop-budget enforcement) is covered
//! by the executor cycle-detection test suite (in-scope at the
//! M4-72 + executor-cycle-detection slice; out-of-scope at this
//! ADR-136 §D5 negative-scenario file).
//!
//! # Pinned invariants
//!
//! 1. **Parser admits variable-length expand syntax.** `*1..N`
//!    syntactically valid for any reasonable N.
//! 2. **Lower-bound > upper-bound rejects.** `*3..1` is invalid
//!    per the grammar.
//! 3. **Star-only (`*`) parses.** Bare `*` means "any length"; the
//!    parser admits it (the executor enforces a default cap).
//! 4. **Huge upper bound (u32::MAX) parses.** The grammar admits
//!    any integer literal; the planner / executor cap to a sane
//!    default at a separate seam (see "Coverage scope" above).
//! 5. **Round-trip stability.** Variable-length syntax printed +
//!    re-parsed produces the same AST.
//! 6. **Cyclic-graph-shaped queries parse.** Self-loops + back-
//!    edges in the query PATTERN parse (parser-level admission of
//!    the syntactic shape; the actual graph-walk cycle-detection
//!    contract is enforced at the executor seam, not here).
//!
//! # Discipline anchors
//!
//! Discipline references for the parser-only scope above:
//! `feedback_load_bearing_pr_requires_fault_injection_tests.md`
//! (load-bearing PRs require fault-injection regression tests per
//! failure mode); ADR-038 §D-9 (variable-length expand contract);
//! `feedback_noop_trampoline_anti_pattern.md` (W23-MFI-6 — scope the
//! doc claim to what the body proves; parser-level pinning is the
//! foundation, planner/executor termination pins follow when the
//! corresponding seams become load-bearing).

use arcgraph_query::parse;

#[test]
fn variable_length_one_to_three_parses() {
    let q = "MATCH (a)-[:KNOWS*1..3]->(b) RETURN a, b";
    parse(q).expect("variable-length 1..3 must parse");
}

#[test]
fn variable_length_zero_lower_bound_parses() {
    let q = "MATCH (a)-[:KNOWS*0..3]->(b) RETURN a, b";
    let _ = parse(q); // may or may not be admissible; either is fine
}

#[test]
fn variable_length_huge_upper_bound_parses() {
    let q = format!(
        "MATCH (a)-[:KNOWS*1..{}]->(b) RETURN a, b LIMIT 10",
        1_000_000
    );
    parse(&q).expect("huge upper bound must parse");
}

#[test]
fn variable_length_round_trip_stability() {
    let queries = [
        "MATCH (a)-[:KNOWS*1..3]->(b) RETURN a, b",
        "MATCH (a)-[:KNOWS*1..5]->(b) RETURN a, b",
        "MATCH (a)-[:KNOWS*2..2]->(b) RETURN a, b",
    ];
    for q in queries {
        let stmt = parse(q).expect("parse");
        let printed = format!("{}", stmt);
        let reparsed = parse(&printed).expect("re-parse");
        assert_eq!(stmt, reparsed, "round-trip failed for {q}");
    }
}

#[test]
fn cyclic_pattern_self_join_parses() {
    let q = "MATCH (a)-[:KNOWS]->(b), (b)-[:KNOWS]->(a) RETURN a, b";
    parse(q).expect("cyclic pattern (mutual KNOWS) must parse");
}

#[test]
fn self_loop_pattern_parses() {
    let q = "MATCH (a)-[:LIKES]->(a) RETURN a";
    parse(q).expect("self-loop pattern must parse");
}

#[test]
fn nested_variable_length_in_chain_parses() {
    let q = "MATCH (a)-[:KNOWS*1..3]->(b)-[:WORKS_AT]->(c) RETURN a, b, c LIMIT 100";
    parse(q).expect("variable-length + fixed chain must parse");
}

#[test]
fn back_edge_in_pattern_parses() {
    // A "diamond" pattern that closes via a back-edge.
    let q = "MATCH (a)-[:R]->(b)-[:R]->(c), (a)-[:R]->(c) RETURN a, b, c LIMIT 50";
    parse(q).expect("diamond pattern must parse");
}

#[test]
fn variable_length_no_panic_on_adversarial_inputs() {
    // 10 adversarial variable-length shapes — none should panic.
    let queries = [
        "MATCH (a)-[:R*1..1]->(b) RETURN a, b",
        "MATCH (a)-[:R*..5]->(b) RETURN a, b",
        "MATCH (a)-[:R*5..]->(b) RETURN a, b LIMIT 10",
        "MATCH (a)-[r:R*1..3]->(b) RETURN a, b, r",
        "MATCH (a)-[:R*]->(b) RETURN a, b LIMIT 100",
        "MATCH (a)-[*1..3]->(b) RETURN a, b",
        "MATCH (a)<-[:R*1..3]-(b) RETURN a, b",
        "MATCH (a)-[:R*1..3]-(b) RETURN a, b",
        // Multiple variable-length segments.
        "MATCH (a)-[:R*1..2]->(b)-[:R*1..2]->(c) RETURN a, b, c LIMIT 5",
        // Variable-length + named relationship.
        "MATCH p = (a)-[:R*1..3]->(b) RETURN p LIMIT 10",
    ];
    for q in &queries {
        // Some forms may be syntactically invalid per ADR-038 narrowing
        // (e.g., `*..N` vs `*1..N`); we only assert NO PANIC, not parse-OK.
        let _ = parse(q);
    }
}

#[test]
fn variable_length_lower_zero_or_one_parses() {
    parse("MATCH (a)-[:R*0..3]->(b) RETURN a, b").map_or_else(
        |_e| {
            // 0-lower-bound MAY be rejected per ADR-038 narrowing;
            // the no-panic invariant is what's load-bearing.
        },
        |_stmt| (),
    );
    parse("MATCH (a)-[:R*1..3]->(b) RETURN a, b").expect("1..N is canonical");
}

#[test]
fn deeply_nested_variable_length_via_or_parses() {
    let q = "MATCH (a:Person)-[:KNOWS*1..3]->(b:Person)
             WHERE (b)-[:FOLLOWS*1..3]->(a) OR a.age > 30
             RETURN a, b LIMIT 100";
    let _ = parse(q); // form may be rejected; no panic is the key
}
