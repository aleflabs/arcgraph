//! **ADR-038 amendment-12 (#796 companion)** — openCypher v9 §6.4
//! implicit-grouping-key validation (`AmbiguousAggregationExpression`),
//! END-TO-END through `QueryEngine::execute`.
//!
//! The NON-aggregating projections form the implicit grouping key; within an
//! AGGREGATING projection every variable/property reference OUTSIDE an
//! aggregate-function argument must itself be a grouping key. A COMPLEX
//! grouping key (`a + b`) does not make its leaves grouping keys.
//!
//! Golden oracle = vendored TCK `clauses/return/Return6.feature` [18]/[19]
//! (pass) vs [20]/[21] (fail) + `clauses/with/With6.feature` [6]/[7] (pass)
//! vs [8]/[9] (fail). This validation is the mandatory companion to the
//! permissive-binding lane: it catches the "Fail if …" scenarios that were
//! previously MASKED by the `UnknownLabel` error (so removing that error does
//! not regress them).

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::semantic::StubCatalogProvider;

/// Execute against a bare catalog + empty substrate (labels are permissive
/// now, so `:Person` binds to empty). Returns Ok/Err of the FULL pipeline.
fn run(cypher: &str) -> Result<usize, String> {
    let cat = StubCatalogProvider::new();
    let s = StubExecutorSubstrate::new();
    QueryEngine::new(&cat)
        .execute(cypher, &s)
        .map(|r| r.rows.len())
        .map_err(|e| format!("{e:?}"))
}

/// Assert the query COMPILES + runs (no aggregation-ambiguity error).
fn assert_ok(cypher: &str) {
    match run(cypher) {
        Ok(_) => {}
        Err(e) => panic!("expected OK for `{cypher}`, got error: {e}"),
    }
}

/// Assert the query is REJECTED with `AmbiguousAggregationExpression` at
/// compile time (the specific variant, not merely "an error").
fn assert_ambiguous(cypher: &str) {
    match run(cypher) {
        Ok(n) => panic!("expected AmbiguousAggregationExpression for `{cypher}`, got Ok({n} rows)"),
        Err(e) => assert!(
            e.contains("AmbiguousAggregation"),
            "expected AmbiguousAggregationExpression for `{cypher}`, got a DIFFERENT error: {e}"
        ),
    }
}

// =====================================================================
// PASS — a SIMPLE grouping key referenced inside an aggregating projection
// is fine (Return6 [18]/[19]; With6 [6]/[7]).
// =====================================================================

#[test]
fn simple_grouping_key_reference_is_ok() {
    // [19] me.age is a (simple) grouping key, referenced inside the aggregate.
    assert_ok("MATCH (me:Person)--(you:Person) RETURN me.age, me.age + count(you.age)");
    // [18] aliased simple grouping key.
    assert_ok(
        "MATCH (me:Person)--(you:Person) WITH me.age AS age, you RETURN age, age + count(you.age)",
    );
    // [6] grouping key referenced inside a map-literal aggregating projection.
    assert_ok(
        "MATCH (a {name:'x'})<-[:FATHER]-(child) RETURN a.name, {foo: a.name='Andres', kids: collect(child.name)}",
    );
    // WITH-aggregation analogue — the simple grouping key `me.x` (projection 1)
    // is referenced inside the aggregating projection 2 (both aliased, as WITH
    // requires for non-variable projections). `me.x` matches the grouping key.
    assert_ok("MATCH (me:Person)--(you:Person) WITH me.x AS x, me.x + count(you) AS agg RETURN *");
}

#[test]
fn plain_and_grouped_aggregation_is_ok() {
    assert_ok("MATCH (n) RETURN count(*)");
    assert_ok("MATCH (n) RETURN n.x, count(*)"); // group by n.x
    assert_ok("MATCH (n) RETURN n.x AS k, count(*)"); // aliased grouping key
}

// =====================================================================
// FAIL — AmbiguousAggregationExpression (Return6 [20]/[21]; With6 [8]/[9]).
// =====================================================================

#[test]
fn no_grouping_key_is_ambiguous() {
    // [20] me.age is not a grouping key (the only projection is aggregating).
    assert_ambiguous("MATCH (me:Person)--(you:Person) RETURN me.age + count(you.age)");
    // [8] WITH analogue.
    assert_ambiguous(
        "MATCH (me:Person)--(you:Person) WITH me.age + count(you.age) AS agg RETURN *",
    );
}

#[test]
fn complex_grouping_key_recomputed_is_ambiguous() {
    // [21] `me.age + you.age` is a complex grouping key — recomputing it inside
    // the aggregating projection (rather than aliasing + referencing) is
    // ambiguous: its leaves `me.age`/`you.age` are not individually grouping keys.
    assert_ambiguous(
        "MATCH (me:Person)--(you:Person) RETURN me.age + you.age, me.age + you.age + count(*)",
    );
    // [9] WITH analogue (aliased grp, still recomputed not referenced).
    assert_ambiguous(
        "MATCH (me:Person)--(you:Person) WITH me.age + you.age AS grp, me.age + you.age + count(*) AS agg RETURN *",
    );
}

// =====================================================================
// Boundary — aliasing the complex key and REFERENCING the alias is the
// openCypher-correct way, and must be accepted.
// =====================================================================

#[test]
fn aliased_complex_key_referenced_is_ok() {
    // Reference the alias `grp` (a SIMPLE variable) — not a recomputation.
    assert_ok(
        "MATCH (me:Person)--(you:Person) WITH me.age + you.age AS grp RETURN grp, grp + count(*)",
    );
}
