//! GA-rand (#618) — active end-to-end verification for the `rand()`
//! builtin.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every assertion below runs a real ArcQL query through the FULL
//! `QueryEngine` pipeline (parse → bind → type-check → cross-substrate →
//! lower → enumerate → execute) and asserts a property of the result —
//! NOT merely "no error".
//!
//! # Why a property oracle, not a value oracle
//!
//! `rand()` is non-deterministic (openCypher v9 §3: "Returns a random
//! floating point number in the range from 0 (inclusive) to 1
//! (exclusive); i.e. [0,1). The numbers returned follow an approximate
//! uniform distribution."), so there is no golden VALUE to compare. The
//! oracles here are therefore (a) the RANGE `[0, 1)`, (b) range
//! invariants expressed in ArcQL itself, and (c) the random-INDEPENDENT
//! quantifier invariants that are exactly what the openCypher
//! `Quantifier9`..`Quantifier12` TCK scenarios assert — they consume
//! `rand()` only via `[y IN list WHERE rand() > 0.5 | y]` to build an
//! arbitrary sublist, then assert a result that holds for ANY sublist.
//! The stability test re-executes such an invariant several times so a
//! genuine random-independence (not a lucky seed) is what is proven.

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::{StubExecutorSubstrate, Value};
use arcgraph_query::semantic::StubCatalogProvider;

/// Execute `query` through the full pipeline against an empty stub
/// substrate (these queries operate on list literals + `range`, no
/// graph rows needed) and return the result rows.
fn engine_run(query: &str) -> Vec<Vec<Value>> {
    let s = StubExecutorSubstrate::new();
    let c = StubCatalogProvider::new();
    let engine = QueryEngine::new(&c);
    engine
        .execute(query, &s)
        .expect("query should execute")
        .rows
}

/// `true` iff `query` is rejected (parse / bind / type-check / exec).
fn errs(query: &str) -> bool {
    let s = StubExecutorSubstrate::new();
    let c = StubCatalogProvider::new();
    let engine = QueryEngine::new(&c);
    engine.execute(query, &s).is_err()
}

#[test]
fn rand_returns_float_in_unit_interval() {
    // RANGE oracle: every `rand()` draw over 200 seeded rows is a `Float`
    // in `[0, 1)`.
    let rows = engine_run("UNWIND range(1, 200) AS i RETURN rand() AS r");
    assert_eq!(rows.len(), 200, "range(1, 200) seeds 200 rows (inclusive)");
    for row in &rows {
        match &row[0] {
            Value::Float(f) => {
                assert!((0.0..1.0).contains(f), "rand() must be in [0, 1), got {f}");
            }
            other => panic!("rand() must return Float, got {other:?}"),
        }
    }
}

#[test]
fn rand_range_invariant_expressed_in_arcql() {
    // The range invariant expressed in ArcQL itself: `rand() >= 0 AND
    // rand() < 1` is always `true` (two independent draws, each in
    // range). Every one of the 200 rows must be `true`.
    let rows = engine_run("UNWIND range(1, 200) AS i RETURN (rand() >= 0.0 AND rand() < 1.0) AS r");
    assert_eq!(rows.len(), 200);
    for row in &rows {
        assert_eq!(
            row[0],
            Value::Boolean(true),
            "rand() >= 0 AND rand() < 1 must hold on every draw"
        );
    }
}

#[test]
fn rand_quantifier_invariant_is_random_independent() {
    // The load-bearing oracle — the `Quantifier9`[5]-style invariant
    // `none(x IN list WHERE P) = (size([x IN list WHERE P | x]) = 0)`
    // over a RANDOM sublist `[y IN inputList WHERE rand() > 0.5 | y]`. It
    // holds for ANY sublist, so each of the 50 random trials must yield
    // `true`. This is precisely why registering `rand()` closes the
    // Quantifier9..12 scenarios WITHOUT a value oracle.
    let q = r#"
        WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS inputList
        UNWIND range(1, 50) AS trial
        WITH [ y IN inputList WHERE rand() > 0.5 | y] AS list
        RETURN none(x IN list WHERE x > 100)
             = (size([x IN list WHERE x > 100 | x]) = 0) AS result
    "#;
    let rows = engine_run(q);
    assert_eq!(rows.len(), 50, "50 random trials");
    for row in &rows {
        assert_eq!(
            row[0],
            Value::Boolean(true),
            "none == size-filter-is-zero must hold for ANY random sublist"
        );
    }
}

#[test]
fn rand_quantifier_invariant_stable_across_repeated_runs() {
    // STABILITY oracle (mirrors the per-PR ratchet stability re-run at
    // the unit-of-work level): re-execute a random-independent invariant
    // several times. Each run draws fresh entropy-seeded randomness; the
    // result must be all-`true` on EVERY run — proving genuine random
    // independence, not a single lucky seed. This is the `Quantifier9`[1]
    // invariant verbatim ("none is always true if the predicate is
    // statically false and the list is not empty") over the same mixed
    // input list the TCK scenario uses.
    let q = r#"
        WITH [1, null, true, 4.5, 'abc', false, '', [234, false], {a: null, b: true}, {}, [], [null]] AS inputList
        UNWIND range(1, 20) AS trial
        WITH [ y IN inputList WHERE rand() > 0.5 | y] AS list
        WITH list WHERE size(list) > 0
        RETURN none(x IN list WHERE false) AS result
    "#;
    for run_ix in 0..8 {
        let rows = engine_run(q);
        for row in &rows {
            assert_eq!(
                row[0],
                Value::Boolean(true),
                "none(x IN nonempty-list WHERE false) must be true on every run (run {run_ix})"
            );
        }
    }
}

#[test]
fn rand_arity_enforced_end_to_end() {
    // `rand()` is nullary — `rand(1)` is rejected at type-check (arity),
    // and the zero-arg form is accepted. This pins the registry arity
    // end-to-end (mirrors `e`/`pi`).
    assert!(
        errs("RETURN rand(1) AS r"),
        "rand(1) must be rejected (nullary)"
    );
    assert!(!errs("RETURN rand() AS r"), "rand() must be accepted");
}
