//! **#1056 / #990** — dynamic map subscript `map['key']` END-TO-END
//! verification + the #618 projection-output-type registration keystone.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every assertion drives a REAL ArcQL query through the FULL pipeline
//! (`QueryEngine::execute`: parse → bind → type-check → cross-substrate →
//! lower → execute) against a fresh empty `StubExecutorSubstrate` — the
//! EXACT path the TCK conformance ratchet
//! (`arcgraph-tck/tests/full_eligible_conformance.rs`) uses — and asserts
//! the returned row equals the **openCypher-golden** value from the
//! vendored TCK feature files
//! (`crates/arcgraph-tck/tck/features/expressions/map/Map2.feature` and
//! `.../graph/Graph6.feature`, `.../map/Map1.feature`), NOT merely "no
//! error".
//!
//! # The three coupled changes (all in THIS slice)
//!
//! 1. **type-check** admits a `Map` base × `String` index in the
//!    `Subscript` arm (`check_subscript_base` + `check_string_index`),
//!    keeping the `List` base × `Integer` index path intact.
//! 2. **eval** handles `Value::Map` × `Value::String` (case-sensitive key
//!    lookup; missing key ⇒ `null`), keeping `list[int]` intact.
//! 3. **projection-output-type registration** (#618 re-land): `WITH
//!    <expr> AS n` registers `n`'s CONCRETE type so a downstream non-map /
//!    non-entity property access (`WITH 123 AS n RETURN n.x`) rejects at
//!    COMPILE time (`Graph6` [9] / `Map1` [6], WrongErrorPhase → correct
//!    phase). Re-landing this is SAFE only because steps 1+2 make a `Map`
//!    base type-check (the prior #618 revert tension).

use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::semantic::{ArcQLError, StubCatalogProvider, TypeCheckError};
use arcgraph_query::{ExplainError, QueryEngine};
use std::collections::BTreeMap;

/// Execute `cypher` through the full engine against a fresh EMPTY
/// substrate and return all result rows.
fn run(cypher: &str) -> Vec<Vec<Value>> {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    engine.execute(cypher, &substrate).expect("execute").rows
}

/// Execute `cypher`, assert exactly one row + one column, return the cell.
fn cell(cypher: &str) -> Value {
    let rows = run(cypher);
    assert_eq!(rows.len(), 1, "expected exactly one row for `{cypher}`");
    assert_eq!(
        rows[0].len(),
        1,
        "expected exactly one column for `{cypher}`"
    );
    rows[0][0].clone()
}

/// Bind + type-check `query`; return the `ArcQLError` on COMPILE-time
/// rejection. Panics if it returned rows OR errored at a NON-compile
/// phase (a runtime `ExecutionEval` is the WrongErrorPhase the #618
/// keystone closes — so a test calling this asserts the correct compile
/// phase).
fn reject_at_compile(query: &str) -> ArcQLError {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    match engine.execute(query, &substrate) {
        Ok(res) => panic!(
            "expected COMPILE-time rejection for `{query}`, but it returned {} row(s)",
            res.rows.len()
        ),
        Err(ExplainError::ArcQL(e @ ArcQLError::TypeCheck(_))) => e,
        Err(ExplainError::ArcQL(other)) => panic!(
            "expected a TypeCheck (compile) error for `{query}`, got a different ArcQLError: {other}"
        ),
        Err(ExplainError::ExecutionEval(msg)) => panic!(
            "WRONG PHASE for `{query}`: expected COMPILE-time TypeCheck error, got RUNTIME eval error: {msg}"
        ),
        Err(other) => panic!("expected compile-time TypeCheck error for `{query}`, got: {other}"),
    }
}

fn map_of(pairs: &[(&str, &str)]) -> Value {
    let mut m = BTreeMap::new();
    for (k, v) in pairs {
        m.insert((*k).to_string(), Value::String((*v).to_string()));
    }
    Value::Map(m)
}

// =====================================================================
// STEP 1+2 — Map × String dynamic value access (Map2 [5] + semantics)
// =====================================================================

#[test]
fn map_subscript_hit_returns_value() {
    // `{name:'a'}['name']` ⇒ 'a'.
    assert_eq!(
        cell("WITH {name: 'a'} AS m RETURN m['name'] AS r"),
        Value::String("a".into())
    );
}

#[test]
fn map_subscript_direct_literal_base() {
    // A map LITERAL subscripted directly (no WITH binding) — the
    // `TypeInfo::Map` base flows straight from the MapLiteral.
    assert_eq!(
        cell("RETURN {name: 'a'}['name'] AS r"),
        Value::String("a".into())
    );
}

#[test]
fn map_subscript_is_case_sensitive() {
    // Map2 [5] — the load-bearing case-sensitivity oracle. `map['name']`
    // ≠ `map['Name']` ≠ `map['nAMe']`.
    let setup = "WITH {name: 'Mats', Name: 'Pontus'} AS map RETURN map[";
    assert_eq!(
        cell(&format!("{setup}'name'] AS r")),
        Value::String("Mats".into())
    );
    assert_eq!(
        cell(&format!("{setup}'Name'] AS r")),
        Value::String("Pontus".into())
    );
    // A case-mismatched key is a MISS ⇒ null (NOT the other-cased value).
    assert_eq!(cell(&format!("{setup}'nAMe'] AS r")), Value::Null);
}

#[test]
fn map_subscript_missing_key_is_null() {
    // openCypher dynamic value access: a missing key ⇒ null (NOT an
    // error), mirroring out-of-range list-index ⇒ null.
    assert_eq!(
        cell("WITH {name: 'Mats', nome: 'Pontus'} AS m RETURN m['null'] AS r"),
        Value::Null
    );
    assert_eq!(
        cell("WITH {name: 'a'} AS m RETURN m['missing'] AS r"),
        Value::Null
    );
}

#[test]
fn map_subscript_null_index_is_null() {
    // Map2 [4] equivalent — `map[null]` ⇒ null (3VL short-circuit on a
    // null index, even with a concrete map base).
    assert_eq!(
        cell("WITH {name: 'Mats'} AS m, null AS i RETURN m[i] AS r"),
        Value::Null
    );
}

#[test]
fn map_subscript_int_index_rejects_at_compile() {
    // A STATICALLY-concrete Integer index into a STATICALLY-concrete Map
    // base is a compile-time TypeMismatch (InvalidArgumentType analog).
    let err = reject_at_compile("WITH {name: 'a'} AS m RETURN m[0] AS r");
    assert!(
        matches!(
            err,
            ArcQLError::TypeCheck(TypeCheckError::TypeMismatch { .. })
        ),
        "expected TypeMismatch for int-index-into-map, got {err:?}"
    );
    // Float index too.
    let err = reject_at_compile("WITH {name: 'a'} AS m RETURN m[1.5] AS r");
    assert!(matches!(
        err,
        ArcQLError::TypeCheck(TypeCheckError::TypeMismatch { .. })
    ));
}

// =====================================================================
// ZERO-REGRESSION GUARD — Map2 [3]/[4] (passed BEFORE this slice; the
// #618 revert was triggered when they regressed). They MUST still pass.
// =====================================================================

#[test]
fn map2_3_null_base_is_null() {
    // Map2 [3] — `null[idx]` ⇒ null (null base short-circuit). MUST stay
    // green after the projection-type registration re-land.
    assert_eq!(
        cell("WITH null AS expr, 'x' AS idx RETURN expr[idx] AS value"),
        Value::Null
    );
}

#[test]
fn map2_4_null_index_is_null() {
    // Map2 [4] — `map[null]` ⇒ null. MUST stay green: the projected
    // `{name:'Mats'}` now types as `Map`, and the map-subscript check
    // admits it (the pre-slice `check_list_operand` would have
    // over-rejected once the projection type went concrete — that was the
    // revert).
    assert_eq!(
        cell("WITH {name: 'Mats'} AS expr, null AS idx RETURN expr[idx] AS value"),
        Value::Null
    );
}

// =====================================================================
// STEP 3 — projection-output-type registration → COMPILE-phase property
// access rejection on a non-entity / non-map base (Graph6 [9], Map1 [6]).
// =====================================================================

#[test]
fn property_access_on_projected_scalar_rejects_at_compile() {
    // Graph6 [9] / Map1 [6] — `WITH 123 AS n RETURN n.num` rejects at
    // COMPILE time (PropertyAccessOnNonEntity), NOT at runtime
    // (WrongErrorPhase). Driven by the projection-output-type
    // registration: `n` now types as `Integer`, which
    // `is_definitely_non_entity_non_map` rejects.
    for exp in ["123", "42.45", "true", "false", "'string'", "[123, true]"] {
        let q = format!("WITH {exp} AS nonMap RETURN nonMap.num AS r");
        let err = reject_at_compile(&q);
        assert!(
            matches!(
                err,
                ArcQLError::TypeCheck(TypeCheckError::PropertyAccessOnNonEntity { .. })
            ),
            "expected PropertyAccessOnNonEntity at compile time for `{q}`, got {err:?}"
        );
    }
}

#[test]
fn property_access_on_projected_map_is_admitted() {
    // The complement: a projected MAP base admits property access (the
    // registration must NOT over-reject the map case). `m.existing` over
    // a `Map`-typed `m` resolves at runtime; `m.missing` ⇒ null. (Map1
    // [1] static map field access.)
    assert_eq!(
        cell("WITH {existing: 42, notMissing: null} AS m RETURN m.existing AS r"),
        Value::Integer(42)
    );
    assert_eq!(
        cell("WITH {existing: 42} AS m RETURN m.missing AS r"),
        Value::Null
    );
}

// =====================================================================
// NO LIST-SUBSCRIPT REGRESSION — the `list[int]` path stays intact.
// =====================================================================

#[test]
fn list_subscript_still_works() {
    assert_eq!(cell("RETURN [10, 20, 30][1] AS r"), Value::Integer(20));
    assert_eq!(cell("RETURN [10, 20, 30][-1] AS r"), Value::Integer(30));
    assert_eq!(cell("RETURN [10, 20, 30][9] AS r"), Value::Null);
    // List slice unaffected.
    assert_eq!(
        cell("RETURN [10, 20, 30][0..2] AS r"),
        Value::List(vec![Value::Integer(10), Value::Integer(20)])
    );
}

#[test]
fn list_subscript_string_index_still_rejects_at_compile() {
    // A String index into a statically-concrete LIST base must STILL be a
    // compile-time TypeMismatch (the dual-dispatch must not relax the
    // list path to accept a string index).
    let err = reject_at_compile("WITH [10, 20] AS l RETURN l['x'] AS r");
    assert!(
        matches!(
            err,
            ArcQLError::TypeCheck(TypeCheckError::TypeMismatch { .. })
        ),
        "expected TypeMismatch for string-index-into-list, got {err:?}"
    );
}

// =====================================================================
// Sanity: the constructed map value matches the engine's row value.
// =====================================================================

#[test]
fn projected_map_value_roundtrips() {
    // `WITH {a:'x', b:'y'} AS m RETURN m` ⇒ the map value itself.
    assert_eq!(
        cell("WITH {a: 'x', b: 'y'} AS m RETURN m AS r"),
        map_of(&[("a", "x"), ("b", "y")])
    );
}
