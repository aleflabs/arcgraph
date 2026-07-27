//! ADR-152-amendment-02 (W28) — composite `List`-literal property-value
//! smoke test (`feat/composite-list-literals`).
//!
//! Pre-amendment, `CREATE (n:User {tags: ["a","b","c"]})` rejected at the
//! executor's write-op `literal_to_value` helper (ADR-152 §"Forward-
//! deferred" #5). Post-amendment-02 §D-1 a `List`-of-scalars literal (and
//! nested lists thereof) materializes to `Value::List` and round-trips
//! through the property-bag path.
//!
//! Strong oracles: every test asserts the ROUND-TRIPPED list value
//! (`==`), not "no error". The full query pipeline (parse → bind →
//! type-check → cross-substrate → lower → execute) runs against a single
//! `StubExecutorSubstrate` instance per the
//! `create_then_match_by_property_smoke.rs` harness shape.
//!
//! Scope pin (amendment-02 §D-2): `Map` remains deferred (no `Value::Map`
//! variant); the test below pins that an honest forward-pin (rejected,
//! not silently coerced).

use arcgraph_core::{LabelId, PartitionId, TenantId};
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::{ExecutionContext, value::Value};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{Statement, executor::Pipeline, parse};

/// The first LabelId the `StubExecutorSubstrate` allocates — pre-bound
/// into the catalog so the MATCH-lowered Scan emits the SAME LabelId the
/// substrate's `create_node` assigns.
const STUB_FIRST_LABEL_ID: u32 = 1024;

fn lower(query: &str) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    let inner = match stmt {
        Statement::Read(_) => stmt,
        other => panic!("expected Read statement, got {other:?}"),
    };
    let cat = StubCatalogProvider::new().with_label_id("User", LabelId::new(STUB_FIRST_LABEL_ID));
    let mut bound = BindingVisitor::bind(&inner, query, &cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

/// Like [`lower`] but returns a `Result` so a deferred-composite query
/// (e.g. a `Map` literal) can be asserted as rejected WHEREVER it is
/// rejected (parse / bind / type-check / lower) without panicking.
fn try_lower(query: &str) -> Result<LogicalPlan, String> {
    let stmt = parse(query).map_err(|e| format!("parse: {e:?}"))?;
    let cat = StubCatalogProvider::new().with_label_id("User", LabelId::new(STUB_FIRST_LABEL_ID));
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).map_err(|e| format!("bind: {e:?}"))?;
    TypeCheckVisitor::check(&mut bound, &cat).map_err(|e| format!("type-check: {e:?}"))?;
    CrossSubstrateValidator::validate(&bound, &cat).map_err(|e| format!("cross: {e:?}"))?;
    LogicalPlanLoweringVisitor::lower(&bound).map_err(|e| format!("lower: {e:?}"))
}

fn execute(
    plan: &LogicalPlan,
    substrate: &StubExecutorSubstrate,
    ctx: &ExecutionContext,
) -> Vec<Vec<Value>> {
    let mut op = Pipeline::build(plan).expect("pipeline build OK");
    let mut out: Vec<Vec<Value>> = Vec::new();
    loop {
        let b = op.next_batch(ctx, substrate).expect("batch OK");
        if b.is_empty() {
            break;
        }
        for i in 0..b.row_count() {
            out.push(b.row(i).to_vec());
        }
    }
    out
}

/// Like [`execute`] but returns the executor error rather than
/// panicking — used to assert the `Map` forward-pin rejects at execute.
fn execute_result(
    plan: &LogicalPlan,
    substrate: &StubExecutorSubstrate,
    ctx: &ExecutionContext,
) -> Result<Vec<Vec<Value>>, String> {
    let mut op = Pipeline::build(plan).map_err(|e| format!("build: {e:?}"))?;
    let mut out: Vec<Vec<Value>> = Vec::new();
    loop {
        let b = op
            .next_batch(ctx, substrate)
            .map_err(|e| format!("{e:?}"))?;
        if b.is_empty() {
            break;
        }
        for i in 0..b.row_count() {
            out.push(b.row(i).to_vec());
        }
    }
    Ok(out)
}

fn fresh_ctx() -> ExecutionContext {
    ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
}

#[test]
fn create_list_then_match_returns_list_exact() {
    // ADR-152-amendment-02 oracle #1: CREATE {tags:["a","b","c"]} →
    // MATCH … RETURN n.tags returns ["a","b","c"] EXACTLY.
    let substrate = StubExecutorSubstrate::new();
    let _ = execute(
        &lower(r#"CREATE (n:User {tags: ["a", "b", "c"]}) RETURN n"#),
        &substrate,
        &fresh_ctx(),
    );

    let rows = execute(
        &lower("MATCH (n:User) RETURN n.tags"),
        &substrate,
        &fresh_ctx(),
    );
    assert_eq!(rows.len(), 1, "MATCH observes the created node");
    assert_eq!(
        rows[0][0],
        Value::List(vec![
            Value::String("a".into()),
            Value::String("b".into()),
            Value::String("c".into()),
        ]),
        "n.tags round-trips as the exact list of strings"
    );
}

#[test]
fn nested_list_round_trips_exact() {
    // Oracle #2: nested list [[1,2],[3]] round-trips exactly.
    let substrate = StubExecutorSubstrate::new();
    let _ = execute(
        &lower("CREATE (n:User {matrix: [[1, 2], [3]]}) RETURN n"),
        &substrate,
        &fresh_ctx(),
    );

    let rows = execute(
        &lower("MATCH (n:User) RETURN n.matrix"),
        &substrate,
        &fresh_ctx(),
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0][0],
        Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(3)]),
        ]),
        "nested list round-trips with structure + element types preserved"
    );
}

#[test]
fn heterogeneous_list_round_trips_exact() {
    // Oracle #3: heterogeneous [1, "x", true] preserved (Cypher 9 §3.5).
    let substrate = StubExecutorSubstrate::new();
    let _ = execute(
        &lower(r#"CREATE (n:User {mixed: [1, "x", true]}) RETURN n"#),
        &substrate,
        &fresh_ctx(),
    );

    let rows = execute(
        &lower("MATCH (n:User) RETURN n.mixed"),
        &substrate,
        &fresh_ctx(),
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0][0],
        Value::List(vec![
            Value::Integer(1),
            Value::String("x".into()),
            Value::Boolean(true),
        ]),
        "heterogeneous list preserves each element's runtime type"
    );
}

#[test]
fn empty_list_round_trips_as_empty_list_not_null() {
    // Oracle #4 (documented behavior, amendment-02 §D-4): an empty-list
    // VALUE persists + reads back as an empty list — distinct from an
    // ABSENT property (which projects as Value::Null).
    let substrate = StubExecutorSubstrate::new();
    let _ = execute(
        &lower("CREATE (n:User {tags: []}) RETURN n"),
        &substrate,
        &fresh_ctx(),
    );

    let rows = execute(
        &lower("MATCH (n:User) RETURN n.tags"),
        &substrate,
        &fresh_ctx(),
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0][0],
        Value::List(vec![]),
        "empty-list property value round-trips as an empty list (NOT Null / absent)"
    );

    // Contrast pin: an absent property projects as Null.
    let absent = execute(
        &lower("MATCH (n:User) RETURN n.never_set"),
        &substrate,
        &fresh_ctx(),
    );
    assert_eq!(absent.len(), 1);
    assert_eq!(
        absent[0][0],
        Value::Null,
        "an absent property is Null — distinguishable from an empty-list value"
    );
}

#[test]
fn set_list_property_then_match_returns_list() {
    // Oracle #5: SET shares the write-op gate (amendment-02 §D-1) — SET a
    // list property, then read it back exactly.
    let substrate = StubExecutorSubstrate::new();
    let _ = execute(
        &lower("CREATE (n:User {id: 1}) RETURN n"),
        &substrate,
        &fresh_ctx(),
    );

    // SET n.tags = ["x", "y"] on the (single) User node.
    let _ = execute(
        &lower(r#"MATCH (n:User) SET n.tags = ["x", "y"]"#),
        &substrate,
        &fresh_ctx(),
    );

    let rows = execute(
        &lower("MATCH (n:User) RETURN n.tags"),
        &substrate,
        &fresh_ctx(),
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0][0],
        Value::List(vec![Value::String("x".into()), Value::String("y".into())]),
        "SET-applied list property round-trips exactly"
    );
}

#[test]
fn return_after_create_carries_the_list() {
    // The CreateNodeOp's emitted NodeView carries the materialized list
    // so RETURN-after-CREATE shows it (no separate MATCH needed).
    let substrate = StubExecutorSubstrate::new();
    let rows = execute(
        &lower(r#"CREATE (n:User {tags: ["a", "b"]}) RETURN n.tags"#),
        &substrate,
        &fresh_ctx(),
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0][0],
        Value::List(vec![Value::String("a".into()), Value::String("b".into())]),
        "RETURN-after-CREATE reads the materialized list from the emitted NodeView"
    );
}

#[test]
fn map_property_value_still_deferred() {
    // Oracle #6 (honest forward-pin, amendment-02 §D-2): `Map` has no
    // `Value::Map` variant; a map-literal property value is rejected
    // (NOT silently coerced). Robust to WHERE it is rejected: if it
    // lowers, it MUST reject at execute.
    let substrate = StubExecutorSubstrate::new();
    let query = "CREATE (n:User {m: {k: 1}}) RETURN n";
    match try_lower(query) {
        Err(_) => {
            // Rejected before execute (parse / bind / type-check / lower)
            // — still a valid "Map deferred" pin.
        }
        Ok(plan) => {
            let r = execute_result(&plan, &substrate, &fresh_ctx());
            assert!(
                r.is_err(),
                "Map-literal property value must be rejected at execute (deferred per \
                 amendment-02 §D-2); got Ok({r:?})"
            );
        }
    }

    // No node was persisted (the rejected CREATE wrote nothing the
    // round-trip can observe).
    let rows = execute(&lower("MATCH (n:User) RETURN n"), &substrate, &fresh_ctx());
    assert_eq!(
        rows.len(),
        0,
        "a rejected Map-literal CREATE persists no node"
    );
}

#[test]
fn negative_number_in_property_value_round_trips() {
    // #870 — a negative numeric value parses as `UnaryOp { Neg, <numeric
    // literal> }`, NOT a bare `Literal`, in EVERY position. It IS a numeric
    // constant; CREATE/SET now ACCEPT + persist it (was rejected — the #870
    // bug this test previously pinned as a "documented boundary"). Both the
    // scalar (`{x: -1}`) and list-element (`{v: [-1]}`) forms round-trip to
    // their EXACT negative value.

    // (a) Scalar negative — accepted (type-check) + round-trips to -1.
    let substrate_a = StubExecutorSubstrate::new();
    let _ = execute(
        &lower("CREATE (n:User {x: -1}) RETURN n"),
        &substrate_a,
        &fresh_ctx(),
    );
    let rows_a = execute(
        &lower("MATCH (n:User) RETURN n.x"),
        &substrate_a,
        &fresh_ctx(),
    );
    assert_eq!(rows_a.len(), 1);
    assert_eq!(
        rows_a[0][0],
        Value::Integer(-1),
        "a negative scalar property value round-trips to -1 (#870)"
    );

    // (b) Negative element inside a list — accepted + round-trips to [-1].
    let substrate_b = StubExecutorSubstrate::new();
    let _ = execute(
        &lower("CREATE (n:User {v: [-1]}) RETURN n"),
        &substrate_b,
        &fresh_ctx(),
    );
    let rows_b = execute(
        &lower("MATCH (n:User) RETURN n.v"),
        &substrate_b,
        &fresh_ctx(),
    );
    assert_eq!(rows_b.len(), 1);
    assert_eq!(
        rows_b[0][0],
        Value::List(vec![Value::Integer(-1)]),
        "a negative list-element property value round-trips to [-1] (#870)"
    );

    // (c) The positive counterpart still round-trips — regression guard.
    let substrate_c = StubExecutorSubstrate::new();
    let _ = execute(
        &lower("CREATE (n:User {v: [1]}) RETURN n"),
        &substrate_c,
        &fresh_ctx(),
    );
    let rows_c = execute(
        &lower("MATCH (n:User) RETURN n.v"),
        &substrate_c,
        &fresh_ctx(),
    );
    assert_eq!(rows_c.len(), 1);
    assert_eq!(
        rows_c[0][0],
        Value::List(vec![Value::Integer(1)]),
        "the positive-integer list counterpart still round-trips"
    );
}

#[test]
fn merge_with_list_property_round_trips_exact() {
    // N-3 (amendment-02 §D-1): close the write-op family DIRECTLY. A MERGE
    // pattern's inline `List` property bag — `MERGE (n:User {tags:["a","b"]})`
    // — on an empty store fires the create-branch (Node-shape → a
    // `CreateNodeOp` sub-pipeline per ADR-151 §D-6 lowering), which
    // materializes the inline list through the SAME shared
    // `literal_lift::literal_value` gate that CREATE node uses
    // (`create_node.rs::literal_to_value` delegates to it per §D-1). Prior
    // MERGE list coverage was only TRANSITIVE — via ON CREATE / ON MATCH SET
    // firing the SetOp; this pins the MERGE pattern-bag path itself. Strong
    // oracle: the round-tripped list `==`.
    let substrate = StubExecutorSubstrate::new();

    // MERGE parses as a `Statement::Read` (like the other write ops — cf.
    // `merge_by_property_idempotent_smoke.rs`), so the file's `lower` helper
    // (with its Read-guard) applies unchanged.
    let _ = execute(
        &lower(r#"MERGE (n:User {tags: ["a", "b"]})"#),
        &substrate,
        &fresh_ctx(),
    );

    let rows = execute(
        &lower("MATCH (n:User) RETURN n.tags"),
        &substrate,
        &fresh_ctx(),
    );
    assert_eq!(
        rows.len(),
        1,
        "MERGE create-branch persisted exactly one node"
    );
    assert_eq!(
        rows[0][0],
        Value::List(vec![Value::String("a".into()), Value::String("b".into())]),
        "MERGE-created node's inline list property round-trips exactly"
    );
}
