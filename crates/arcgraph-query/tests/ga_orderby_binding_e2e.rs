//! **#618 (GA Lane A)** — `ORDER BY` resolves against the RETURN
//! projection OUTPUT, END-TO-END.
//!
//! # The bug this pins (founding failure, #618)
//!
//! `UNWIND [1,3,2] AS ints RETURN ints ORDER BY ints` errored at runtime
//! with `binding BindingId(0) missing from row schema`, and
//! `RETURN ints AS x ORDER BY x` errored at bind with `undeclared
//! variable x`. The RETURN projection mints a FRESH `output_id` per item
//! (the #746 binder↔`ProjectOp` contract), but ORDER BY resolved its
//! variable refs in the PRE-projection scope → the original source id,
//! NOT the projected `output_id`. The lowered plan is
//! `Sort[key=src_id]( Project[emits output_id]( … ) )`; the Sort runs
//! over the Project's OUTPUT schema (which carries `output_id`), so a key
//! of `src_id` is "missing from row schema".
//!
//! The fix (mirroring `bind_with_clause`'s #746 back-patch): the binder
//! pushes a projection-output scope mapping each RETURN output NAME
//! (alias OR passthrough variable) → its `output_id`, so ORDER BY (a
//! standalone `Clause::TailOrderBy` the parser emits after RETURN)
//! resolves to the id the `ProjectOp` emits.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every assertion drives a REAL ArcQL query through the FULL pipeline
//! (`QueryEngine::execute`: parse → bind → type-check → cross-substrate →
//! lower → execute) — the EXACT path the TCK conformance ratchet
//! (`arcgraph-tck/tests/full_eligible_conformance.rs`) uses — and asserts
//! the EXACT row sequence, not merely "no error". The oracle is the
//! openCypher `clauses/return-orderby` ordering semantics.

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::semantic::StubCatalogProvider;

/// Execute `cypher` through the full engine against an EMPTY substrate
/// and return all result rows (row-major).
fn run(cypher: &str) -> Vec<Vec<Value>> {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    engine.execute(cypher, &substrate).expect("execute").rows
}

/// Execute `cypher`, assert every row is single-column, return the
/// column-0 cells in row order (the ORDER BY result sequence).
fn col0(cypher: &str) -> Vec<Value> {
    run(cypher)
        .into_iter()
        .map(|mut row| {
            assert_eq!(row.len(), 1, "expected single-column rows for `{cypher}`");
            row.remove(0)
        })
        .collect()
}

fn ints(ns: &[i64]) -> Vec<Value> {
    ns.iter().map(|n| Value::Integer(*n)).collect()
}

fn list(values: Vec<Value>) -> Value {
    Value::List(values)
}

// =====================================================================
// PART A — passthrough variable: `RETURN ints ORDER BY ints`.
// The founding failure (`binding BindingId(0) missing from row schema`).
// =====================================================================

#[test]
fn orderby_passthrough_ascending() {
    // ASC is the default direction.
    assert_eq!(
        col0("UNWIND [1, 3, 2] AS ints RETURN ints ORDER BY ints"),
        ints(&[1, 2, 3]),
    );
}

#[test]
fn orderby_passthrough_descending() {
    assert_eq!(
        col0("UNWIND [1, 3, 2] AS ints RETURN ints ORDER BY ints DESC"),
        ints(&[3, 2, 1]),
    );
}

#[test]
fn orderby_lists_ascending_matches_opencypher_corpus() {
    assert_eq!(
        col0(
            "UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists \
             RETURN lists ORDER BY lists"
        ),
        vec![
            list(vec![]),
            list(vec![Value::String("a".into())]),
            list(vec![Value::String("a".into()), Value::Integer(1)]),
            list(vec![Value::Integer(1)]),
            list(vec![Value::Integer(1), Value::String("a".into())]),
            list(vec![Value::Integer(1), Value::Null]),
            list(vec![Value::Null, Value::Integer(1)]),
            list(vec![Value::Null, Value::Integer(2)]),
        ],
    );
}

#[test]
fn orderby_lists_descending_matches_opencypher_corpus() {
    assert_eq!(
        col0(
            "UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists \
             RETURN lists ORDER BY lists DESC"
        ),
        vec![
            list(vec![Value::Null, Value::Integer(2)]),
            list(vec![Value::Null, Value::Integer(1)]),
            list(vec![Value::Integer(1), Value::Null]),
            list(vec![Value::Integer(1), Value::String("a".into())]),
            list(vec![Value::Integer(1)]),
            list(vec![Value::String("a".into()), Value::Integer(1)]),
            list(vec![Value::String("a".into())]),
            list(vec![]),
        ],
    );
}

#[test]
fn with_orderby_lists_descending_limit_matches_opencypher_corpus() {
    assert_eq!(
        col0(
            "UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists \
             WITH lists ORDER BY lists DESC LIMIT 4 RETURN lists"
        ),
        vec![
            list(vec![Value::Null, Value::Integer(2)]),
            list(vec![Value::Null, Value::Integer(1)]),
            list(vec![Value::Integer(1), Value::Null]),
            list(vec![Value::Integer(1), Value::String("a".into())]),
        ],
    );
}

// =====================================================================
// PART B — order by the ALIAS: `RETURN n AS x ORDER BY x`.
// The `undeclared variable x` failure (alias not in pre-projection scope).
// =====================================================================

#[test]
fn orderby_alias_ascending() {
    assert_eq!(
        col0("UNWIND [3, 1, 2] AS n RETURN n AS x ORDER BY x"),
        ints(&[1, 2, 3]),
    );
}

#[test]
fn orderby_alias_descending() {
    assert_eq!(
        col0("UNWIND [3, 1, 2] AS n RETURN n AS x ORDER BY x DESC"),
        ints(&[3, 2, 1]),
    );
}

// =====================================================================
// PART C — order by an ALIASED EXPRESSION (property access).
// `RETURN m.a AS v ORDER BY v` → the alias maps to the projected value.
// =====================================================================

#[test]
fn orderby_aliased_property_expression() {
    assert_eq!(
        col0("UNWIND [{a: 3}, {a: 1}, {a: 2}] AS m RETURN m.a AS v ORDER BY v"),
        ints(&[1, 2, 3]),
    );
}

// =====================================================================
// PART D — NO binder regression on the projection / aggregate path
// (the #749 binder↔ProjectOp contract is load-bearing).
// =====================================================================

#[test]
fn plain_projection_without_orderby_unchanged() {
    // A bare RETURN with no ORDER BY still projects correctly (the
    // projection-output scope the fix pushes is inert when no ORDER BY /
    // SKIP / LIMIT references it).
    assert_eq!(col0("RETURN 1 AS c"), ints(&[1]));
    assert_eq!(
        col0("UNWIND [10, 20, 30] AS x RETURN x"),
        ints(&[10, 20, 30]),
    );
}

#[test]
fn aggregate_projection_unchanged() {
    // `count(x)` lowers to `Project(Aggregate(..))`; the #746 contract
    // (output_id agreement) must stay intact under the ORDER BY fix.
    assert_eq!(
        col0("UNWIND [1, 2, 3, 4] AS x RETURN count(x) AS c"),
        ints(&[4])
    );
}

#[test]
fn aggregate_with_orderby_on_alias() {
    // `RETURN count(x) AS c ORDER BY c` — order by the aggregate's
    // OUTPUT alias. Single group → single row; the point is it resolves
    // + executes (no "missing from row schema").
    assert_eq!(
        col0("UNWIND [7, 7, 7] AS x RETURN count(x) AS c ORDER BY c"),
        ints(&[3]),
    );
}

// =====================================================================
// PART E — ORDER BY + SKIP / LIMIT compose over the projection output.
// =====================================================================

#[test]
fn orderby_with_limit() {
    // ORDER BY composes with a (literal) LIMIT over the projection
    // output. NB: SKIP / dynamic-LIMIT execution is a separate
    // `NotImplemented` surface (ADR-038 §2 D-28 → M4-72) — out of scope
    // for the ORDER BY binding fix, so not pinned here.
    assert_eq!(
        col0("UNWIND [5, 1, 4, 2, 3] AS n RETURN n AS x ORDER BY x LIMIT 3"),
        ints(&[1, 2, 3]),
    );
    // Passthrough variant of the same.
    assert_eq!(
        col0("UNWIND [5, 1, 4, 2, 3] AS n RETURN n ORDER BY n LIMIT 2"),
        ints(&[1, 2]),
    );
}

// =====================================================================
// PART F — #836 (CZ customer-found, HIGH) — RETURN-clause ORDER BY by a
// PROJECTED EXPRESSION (not an alias, not a bare passthrough variable).
//
// `RETURN m.a ORDER BY m.a` errored at runtime with
// `binding BindingId(0) missing from row schema`. PART C above shows the
// ALIASED form (`RETURN m.a AS v ORDER BY v`) already works; the gap is
// ordering by the SAME unaliased expression that was projected.
//
// Root cause (binder): `return_output_name` only registered an output
// NAME for an alias or a bare identifier passthrough, so an unaliased
// expression projection (`RETURN m.a`) put NOTHING in the projection-
// output scope. ORDER BY `m.a` then bound `m` to its PRE-projection id,
// which `Project` drops → "missing from row schema" over the Sort's
// (post-Project) row schema. The fix matches the ORDER BY key
// STRUCTURALLY against the RETURN projection's AST expressions and
// resolves it to the projected column's `output_id` (openCypher v9 §6.6:
// the sort sees the projection OUTPUT — an unaliased expression behaves
// like ordering by its implicit column).
//
// RED-on-revert: every assertion below errors with
// `Eval("binding BindingId(N) missing from row schema")` on unfixed main.
// =====================================================================

#[test]
fn cz836_orderby_projected_expression_ascending() {
    // The canonical #836 repro (UNWIND-map analogue of `RETURN p.name
    // ORDER BY p.name`): unaliased projected expression, ORDER BY the
    // SAME expression. ASC default.
    assert_eq!(
        col0("UNWIND [{a: 3}, {a: 1}, {a: 2}] AS m RETURN m.a ORDER BY m.a"),
        ints(&[1, 2, 3]),
    );
}

#[test]
fn cz836_orderby_projected_expression_descending() {
    assert_eq!(
        col0("UNWIND [{a: 3}, {a: 1}, {a: 2}] AS m RETURN m.a ORDER BY m.a DESC"),
        ints(&[3, 2, 1]),
    );
}

#[test]
fn cz836_orderby_projected_expression_with_ties_is_stable() {
    // Ties (two rows with a==2) preserve insertion order (stable sort);
    // the distinct value (1) sorts first. Oracle asserts EXACT sequence.
    assert_eq!(
        col0("UNWIND [{a: 2}, {a: 1}, {a: 2}] AS m RETURN m.a ORDER BY m.a"),
        ints(&[1, 2, 2]),
    );
}

#[test]
fn cz836_orderby_multi_key_two_projected_expressions() {
    // Multi-key `RETURN a.x, a.y ORDER BY a.x, a.y` — both keys are
    // unaliased projected expressions. Primary key x asc, secondary key
    // y asc. Rows: (x,y) sorted lexicographically → (1,5),(1,9),(2,1).
    let rows = run("UNWIND [{x: 2, y: 1}, {x: 1, y: 9}, {x: 1, y: 5}] AS a \
         RETURN a.x, a.y ORDER BY a.x, a.y");
    let pairs: Vec<(Value, Value)> = rows
        .into_iter()
        .map(|r| {
            assert_eq!(r.len(), 2, "expected two columns");
            (r[0].clone(), r[1].clone())
        })
        .collect();
    assert_eq!(
        pairs,
        vec![
            (Value::Integer(1), Value::Integer(5)),
            (Value::Integer(1), Value::Integer(9)),
            (Value::Integer(2), Value::Integer(1)),
        ],
    );
}

#[test]
fn cz836_orderby_mixed_alias_and_projected_expression() {
    // Mixed: one column aliased (`m.a AS n`), one unaliased (`m.b`);
    // ORDER BY the unaliased projected expression `m.b`. The alias path
    // (PART C) and the #836 expression path coexist. Sort by b asc →
    // rows ordered (a,b): (30,1),(10,2),(20,3) → columns [n=a, b].
    let rows = run("UNWIND [{a: 10, b: 2}, {a: 30, b: 1}, {a: 20, b: 3}] AS m \
         RETURN m.a AS n, m.b ORDER BY m.b");
    let pairs: Vec<(Value, Value)> = rows
        .into_iter()
        .map(|r| {
            assert_eq!(r.len(), 2, "expected two columns");
            (r[0].clone(), r[1].clone())
        })
        .collect();
    assert_eq!(
        pairs,
        vec![
            (Value::Integer(30), Value::Integer(1)),
            (Value::Integer(10), Value::Integer(2)),
            (Value::Integer(20), Value::Integer(3)),
        ],
    );
}

#[test]
fn cz836_orderby_projected_expression_with_limit() {
    // #836 expression key composes with a literal LIMIT over the
    // projection output.
    assert_eq!(
        col0(
            "UNWIND [{a: 5}, {a: 1}, {a: 4}, {a: 2}, {a: 3}] AS m RETURN m.a ORDER BY m.a LIMIT 3"
        ),
        ints(&[1, 2, 3]),
    );
}
