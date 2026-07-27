//! ADR-147-amendment-03 (D-1) — non-literal CREATE property values.
//! **The 51× UNWIND-ingest lever.**
//!
//! The 18-row adversarial test matrix from the ultracode brief §5. Each
//! row is a RED-on-revert pin tied to an adversarial verdict. The live
//! CREATE executor is `CreateSpineOp` (`create_spine.rs`), NOT the
//! dead `create_node.rs::do_create` — every executor row here drives the
//! spine path (all CREATE/UNWIND-CREATE queries lower to `CreateSpineOp`
//! per `pipeline.rs:112/480`).
//!
//! The HARD GATES (brief §5): T1 (rejected→accepted + round-trip read-
//! back flip), T9/T10/T16 (the runtime map-fence — 0 nodes persisted on a
//! map-via-param), T15 (held-txn atomicity — lives in the arcgraph-cli
//! served e2e where real Bolt BEGIN/COMMIT exists). T3/T4 flip existing
//! rejection contracts to success; T5/T6 stay green (SET rejected).
//!
//! # Strong oracles
//!
//! Every executor row asserts the ROUND-TRIPPED value (via
//! MATCH-then-RETURN over a single `StubExecutorSubstrate`), not merely
//! "no error" — proving the value survives the JSON-blob property-bag
//! path (`create_then_match_by_property_smoke.rs` harness shape).

use std::collections::{BTreeMap, HashMap};

use arcgraph_core::{LabelId, PartitionId, TenantId};
use arcgraph_query::executor::eval::Parameters;
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::{ExecutionContext, value::Value};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{Statement, executor::Pipeline, parse};

/// The first LabelId the `StubExecutorSubstrate` allocates — pre-bound
/// into the catalog so the MATCH-lowered Scan emits the SAME LabelId the
/// substrate's `create_node` assigns (mirrors
/// `composite_list_literal_smoke.rs`).
const STUB_FIRST_LABEL_ID: u32 = 1024;

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new().with_label_id("User", LabelId::new(STUB_FIRST_LABEL_ID))
}

/// Parse → bind → type-check → cross-substrate → lower. Panics at the
/// first failing stage (used when the query MUST be admitted).
fn lower(query: &str) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    let inner = match stmt {
        Statement::Read(_) => stmt,
        other => panic!("expected Read statement, got {other:?}"),
    };
    let cat = catalog();
    let mut bound = BindingVisitor::bind(&inner, query, &cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

/// Whether type-check ADMITS the query (true) or rejects it (false).
fn type_checks(query: &str) -> Result<(), String> {
    let stmt = parse(query).map_err(|e| format!("parse: {e:?}"))?;
    let cat = catalog();
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).map_err(|e| format!("bind: {e:?}"))?;
    TypeCheckVisitor::check(&mut bound, &cat).map_err(|e| format!("type-check: {e:?}"))
}

fn fresh_ctx() -> ExecutionContext {
    ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
}

/// Execute a plan against a substrate with a parameter bag, collecting
/// output rows. Uses `build_with_parameters` so `$param` binds.
fn execute_params(
    plan: &LogicalPlan,
    substrate: &StubExecutorSubstrate,
    ctx: &ExecutionContext,
    params: &Parameters,
) -> Result<Vec<Vec<Value>>, String> {
    let mut op =
        Pipeline::build_with_parameters(plan, params).map_err(|e| format!("build: {e:?}"))?;
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

/// Lower + execute a query with parameters against a substrate, returning
/// the output rows or the first-stage error string.
fn run_params(
    query: &str,
    substrate: &StubExecutorSubstrate,
    params: &Parameters,
) -> Result<Vec<Vec<Value>>, String> {
    let stmt = parse(query).map_err(|e| format!("parse: {e:?}"))?;
    let cat = catalog();
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).map_err(|e| format!("bind: {e:?}"))?;
    TypeCheckVisitor::check(&mut bound, &cat).map_err(|e| format!("type-check: {e:?}"))?;
    CrossSubstrateValidator::validate(&bound, &cat).map_err(|e| format!("cross: {e:?}"))?;
    let plan = LogicalPlanLoweringVisitor::lower(&bound).map_err(|e| format!("lower: {e:?}"))?;
    execute_params(&plan, substrate, &fresh_ctx(), params)
}

/// Count nodes currently persisted for the default tenant/User label by
/// running a MATCH scan (the read side that observes committed nodes).
fn count_nodes(substrate: &StubExecutorSubstrate) -> usize {
    execute_params(
        &lower("MATCH (n:User) RETURN n"),
        substrate,
        &fresh_ctx(),
        &Parameters::new(),
    )
    .expect("scan OK")
    .len()
}

/// Build a parameter bag (`Parameters = HashMap`).
fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

/// Build a `Value::Map` payload (`BTreeMap` — the map value representation).
fn vmap(pairs: &[(&str, Value)]) -> Value {
    Value::Map(
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect::<BTreeMap<String, Value>>(),
    )
}

// =====================================================================
// T1 — the core lever: rejected→accepted + storage round-trip read-back
// =====================================================================

#[test]
fn t1_unwind_rows_create_property_round_trips() {
    // `UNWIND $rows AS r CREATE (n {name: r.name})` with
    // $rows=[{name:'a'},{name:'b'}] → 2 nodes; RETURN n.name reads back
    // 'a','b' (not NULL/empty) — proves storage round-trip via the same
    // JSON-blob path. HARD GATE (CORRECTNESS: storage round-trip).
    let substrate = StubExecutorSubstrate::new();
    let rows_param = Value::List(vec![
        vmap(&[("name", Value::String("a".into()))]),
        vmap(&[("name", Value::String("b".into()))]),
    ]);
    let bag = params(&[("rows", rows_param)]);

    let created = run_params(
        "UNWIND $rows AS r CREATE (n:User {name: r.name}) RETURN n",
        &substrate,
        &bag,
    )
    .expect("UNWIND … CREATE admitted + executed");
    assert_eq!(created.len(), 2, "2 nodes created from a 2-element $rows");

    // Round-trip read-back through MATCH (the storage-decode path).
    let read = execute_params(
        &lower("MATCH (n:User) RETURN n.name"),
        &substrate,
        &fresh_ctx(),
        &Parameters::new(),
    )
    .expect("MATCH read-back OK");
    let mut names: Vec<String> = read
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.clone(),
            other => panic!("expected String name, got {other:?}"),
        })
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec!["a".to_string(), "b".to_string()],
        "n.name round-trips"
    );
}

// =====================================================================
// T2 — leaf-param path (no UNWIND)
// =====================================================================

#[test]
fn t2_leaf_param_property_round_trips() {
    // `CREATE (n {id: $p})`, $p=7 → 1 node, n.id==7 (Parameter resolves
    // without an upstream row).
    let substrate = StubExecutorSubstrate::new();
    let bag = params(&[("p", Value::Integer(7))]);
    let created = run_params("CREATE (n:User {id: $p}) RETURN n", &substrate, &bag)
        .expect("leaf CREATE with $p admitted + executed");
    assert_eq!(created.len(), 1);

    let read = execute_params(
        &lower("MATCH (n:User) RETURN n.id"),
        &substrate,
        &fresh_ctx(),
        &Parameters::new(),
    )
    .expect("read-back OK");
    assert_eq!(read.len(), 1);
    assert_eq!(read[0][0], Value::Integer(7), "n.id == 7 round-trips");
}

// =====================================================================
// T3/T4 — committed-contract flips live in
// bolt_param_binding_e2e.rs (T3) + create_node_smoke.rs /
// create_rel_smoke.rs (T4). Here we pin the type-check ADMITS side.
// =====================================================================

#[test]
fn t4_type_check_admits_param_property() {
    // FLIP of the Phase 1 rejection contract (SURFACE verdict).
    type_checks("CREATE (n:User {v: $v}) RETURN n").expect("param property admitted at type-check");
}

// =====================================================================
// T5/T6 — SET / MERGE stay literal-only (CREATE-only scope)
// =====================================================================

#[test]
fn t5_set_property_param_stays_rejected() {
    // `SET n.x=$p` rejected at type-check (SET executor is const-only).
    let err =
        type_checks("MATCH (n:User) SET n.x = $p").expect_err("SET non-literal must stay rejected");
    assert!(
        err.contains("SetPropertyValueNotLiteral"),
        "SET rejection cites the SET literal-only variant; got {err}"
    );
}

#[test]
fn t6_merge_pattern_and_on_create_set_param_stay_rejected() {
    // MERGE pattern property value stays literal-only.
    let pat_err = type_checks("MERGE (m:User {x: $p})")
        .expect_err("MERGE pattern non-literal must stay rejected (CREATE-only scope)");
    assert!(
        pat_err.contains("CreatePropertyValueNotLiteral"),
        "MERGE pattern rejection cites the create-property variant; got {pat_err}"
    );

    // ON CREATE SET action stays literal-only.
    let set_err = type_checks("MERGE (m:User {x: 1}) ON CREATE SET m.y = $p")
        .expect_err("MERGE ON CREATE SET non-literal must stay rejected");
    assert!(
        set_err.contains("SetPropertyValueNotLiteral"),
        "ON CREATE SET rejection cites the SET variant; got {set_err}"
    );
}

// =====================================================================
// T7/T8 — null-absence semantics (openCypher: null property is absent)
// =====================================================================

#[test]
fn t7_missing_map_key_yields_absent_property() {
    // `UNWIND [{a:1}] AS r CREATE (n {x: r.b})` (missing key) → node has
    // NO property `x` (absent, not `x:null`).
    let substrate = StubExecutorSubstrate::new();
    let rows = Value::List(vec![vmap(&[("a", Value::Integer(1))])]);
    let bag = params(&[("rows", rows)]);
    let created = run_params(
        "UNWIND $rows AS r CREATE (n:User {x: r.b}) RETURN n",
        &substrate,
        &bag,
    )
    .expect("executed");
    assert_eq!(created.len(), 1, "1 node created");

    // The node's bag has no `x` key (absent, not stored-null).
    let read = execute_params(
        &lower("MATCH (n:User) RETURN n.x"),
        &substrate,
        &fresh_ctx(),
        &Parameters::new(),
    )
    .expect("read-back OK");
    assert_eq!(read.len(), 1);
    assert_eq!(
        read[0][0],
        Value::Null,
        "absent property reads as null (3VL)"
    );

    // Prove ABSENCE (not a stored null) via the property-bag accessor.
    let bag = single_node_bag(&substrate);
    assert!(
        !bag.contains_key("x"),
        "a null-valued property is ABSENT from the stored bag, not stored-as-null; bag={bag:?}"
    );
}

#[test]
fn t8_ragged_rows_leave_absent_property_per_row() {
    // `UNWIND [{a:1},{b:2}] AS r CREATE (n {a: r.a})` (ragged) → row-2
    // node has no property `a` (its `r.a` is a missing key → absent).
    let substrate = StubExecutorSubstrate::new();
    let rows = Value::List(vec![
        vmap(&[("a", Value::Integer(1))]),
        vmap(&[("b", Value::Integer(2))]),
    ]);
    let bag = params(&[("rows", rows)]);
    let created = run_params(
        "UNWIND $rows AS r CREATE (n:User {a: r.a}) RETURN n",
        &substrate,
        &bag,
    )
    .expect("executed");
    assert_eq!(created.len(), 2, "2 nodes created");

    let read = execute_params(
        &lower("MATCH (n:User) RETURN n.a"),
        &substrate,
        &fresh_ctx(),
        &Parameters::new(),
    )
    .expect("read-back OK");
    let mut got: Vec<Value> = read.into_iter().map(|r| r[0].clone()).collect();
    got.sort_by_key(|v| matches!(v, Value::Null));
    assert_eq!(
        got,
        vec![Value::Integer(1), Value::Null],
        "row-1 stores a=1; row-2 leaves `a` absent (its r.a is a missing key)"
    );
}

// =====================================================================
// T9/T10 — the RUNTIME MAP FENCE (the load-bearing guard). HARD GATES.
// =====================================================================

#[test]
fn t9_map_via_param_rejected_zero_nodes() {
    // `CREATE (n {m: $p})`, $p bound to a MAP → clean typed error, 0
    // nodes persisted. AST-shape admits `$p`; the runtime value-type gate
    // rejects the map BEFORE the substrate write. HARD GATE.
    let substrate = StubExecutorSubstrate::new();
    let map_val = vmap(&[("k", Value::Integer(1))]);
    let bag = params(&[("p", map_val)]);
    let err = run_params("CREATE (n:User {m: $p}) RETURN n", &substrate, &bag)
        .expect_err("a map-via-param CREATE property must be a typed execution error");
    assert!(
        err.contains("Map") || err.contains("map"),
        "error names the map fence; got {err}"
    );
    assert_eq!(
        count_nodes(&substrate),
        0,
        "0 nodes persisted when the property value is a runtime map (no silent corruption)"
    );
}

#[test]
fn t10_nested_map_via_row_ref_rejected_zero_nodes() {
    // `CREATE (n {addr: r.profile})` where the unwound element's value is
    // a nested map → typed error, 0 nodes. The map arrives via a row-ref,
    // not a literal — only the value-type gate catches it. HARD GATE.
    let substrate = StubExecutorSubstrate::new();
    let rows = Value::List(vec![vmap(&[(
        "profile",
        vmap(&[("city", Value::String("NYC".into()))]),
    )])]);
    let bag = params(&[("rows", rows)]);
    let err = run_params(
        "UNWIND $rows AS r CREATE (n:User {addr: r.profile}) RETURN n",
        &substrate,
        &bag,
    )
    .expect_err("a nested-map row value must be a typed execution error");
    assert!(
        err.contains("Map") || err.contains("map"),
        "error names the map fence; got {err}"
    );
    assert_eq!(
        count_nodes(&substrate),
        0,
        "0 nodes persisted on a nested-map row leak"
    );
}

// =====================================================================
// T11/T12/T13 — the SAFETY fences (nesting bypass, DoS, determinism)
// =====================================================================

#[test]
fn t11_nested_function_call_rejected_at_type_check() {
    // `[randomUUID()]` — a FunctionCall hidden in a list element must be
    // rejected at TYPE-CHECK (the classifier recurses), never OOM /
    // nondeterministic write.
    type_checks("CREATE (n:User {x: [randomUUID()]}) RETURN n")
        .expect_err("a function call nested in a list must be rejected at type-check (T11)");
}

#[test]
fn t12_range_dos_rejected_at_type_check() {
    // `[range(1, 1000000000000)]` — rejected at type-check (FunctionCall
    // arm), so the read-path range() cap never even runs; never OOM.
    type_checks("CREATE (n:User {x: [range(1,1000000000000)]}) RETURN n")
        .expect_err("range() in a CREATE property must be rejected at type-check (T12)");
}

#[test]
fn t13_timestamp_rejected_at_type_check() {
    // `timestamp()` — rejected at type-check (determinism fence).
    type_checks("CREATE (n:User {t: timestamp()}) RETURN n")
        .expect_err("timestamp() in a CREATE property must be rejected at type-check (T13)");
}

// =====================================================================
// T14 — regression: existing literal-only CREATE still works unchanged
// =====================================================================

#[test]
fn t14_literal_only_create_unchanged() {
    // `CREATE (n {x: 1}), (m {y: 2})` still succeeds — the const fast
    // path is untouched.
    let substrate = StubExecutorSubstrate::new();
    let created = run_params(
        "CREATE (n:User {x: 1}), (m:User {y: 2}) RETURN n, m",
        &substrate,
        &Parameters::new(),
    )
    .expect("literal-only CREATE unchanged");
    assert_eq!(created.len(), 1, "one row binding both created nodes");
    assert_eq!(count_nodes(&substrate), 2, "two nodes created");
}

// =====================================================================
// N1 — EXECUTING BinaryOp round-trips (ultracode verify §3 N1). The
// newly-admitted `evaluate` arithmetic/comparison/logical spine had ZERO
// executing test (the only prior BinaryOp test, `create_node_smoke.rs:
// 271-284`, asserted `check(...).is_ok()` ONLY — never lowered/executed/
// read back). Each test below drives the value through the runtime
// evaluator and asserts the COMPUTED result — a regression in
// `evaluate`/`arithmetic`/`apply_binop` producing a wrong scalar/type
// would now flip these RED.
//
// GRAMMAR NOTE: `property_map`/`prop_entry` are compound-atomic (`${…}`),
// so pest's implicit whitespace is suppressed INSIDE a property value.
// The no-space operator forms (`$a*$b`, `$a>$b`, `$a+$a`) parse; but the
// keyword logical operators `AND`/`OR`/`XOR` REQUIRE surrounding
// whitespace to separate the keyword from its operands, and the
// compound-atomic grammar suppresses that whitespace — so `$a AND $b`
// does NOT parse as a CREATE property value (the same pre-existing
// limitation `create_node_smoke.rs:276` documents for `$a + 1`). The
// arithmetic + comparison round-trips therefore drive the FULL CREATE
// path (parse→type-check→lower→evaluate→materialize→store→read-back);
// the logical `And` + `Xor` round-trips drive the SAME runtime evaluator
// (`evaluate`→`apply_binop`) via a `RETURN $a AND $b` query (where the
// grammar DOES admit whitespaced keyword operators) — a real EXECUTION
// test of the newly-admitted 3VL arms, plus a store round-trip of the
// computed Boolean through the CREATE property path.
// =====================================================================

/// Read back the single created node's property `key` as a `Value`.
fn read_single_prop(substrate: &StubExecutorSubstrate, key: &str) -> Value {
    let read = execute_params(
        &lower(&format!("MATCH (n:User) RETURN n.{key}")),
        substrate,
        &fresh_ctx(),
        &Parameters::new(),
    )
    .expect("read-back OK");
    assert_eq!(read.len(), 1, "expected exactly one created node");
    read[0][0].clone()
}

/// Execute `RETURN $a <op> $b AS r` through the FULL query pipeline
/// (parse→bind→type-check→lower→evaluate) and return the single computed
/// cell — a real execution of the `apply_binop` arm named by `op_kw`.
fn eval_logical_return(op_kw: &str, a: Value, b: Value) -> Value {
    let substrate = StubExecutorSubstrate::new();
    let bag = params(&[("a", a), ("b", b)]);
    let rows = run_params(&format!("RETURN $a {op_kw} $b AS r"), &substrate, &bag)
        .unwrap_or_else(|e| panic!("RETURN $a {op_kw} $b executed: {e}"));
    assert_eq!(rows.len(), 1, "one result row");
    rows[0][0].clone()
}

#[test]
fn n1_arithmetic_binaryop_stores_computed_scalar() {
    // `{total: $a*$b}` with $a=6, $b=7 → the EVALUATED product 42 is
    // stored (numeric arithmetic through `apply_binop::Mul`). Full CREATE
    // path: parse→type-check→lower→evaluate→materialize→store→read-back.
    let substrate = StubExecutorSubstrate::new();
    let bag = params(&[("a", Value::Integer(6)), ("b", Value::Integer(7))]);
    let created = run_params("CREATE (n:User {total: $a*$b}) RETURN n", &substrate, &bag)
        .expect("arithmetic BinaryOp CREATE property admitted + executed");
    assert_eq!(created.len(), 1);
    assert_eq!(
        read_single_prop(&substrate, "total"),
        Value::Integer(42),
        "$a*$b evaluates to 42 and round-trips through storage"
    );
}

#[test]
fn n1_comparison_binaryop_stores_boolean() {
    // `{gt: $a>$b}` with $a=6, $b=7 → stored Boolean(false) (comparison
    // through `apply_binop::Gt` → `compare_op`). Full CREATE path.
    let substrate = StubExecutorSubstrate::new();
    let bag = params(&[("a", Value::Integer(6)), ("b", Value::Integer(7))]);
    run_params("CREATE (n:User {gt: $a>$b}) RETURN n", &substrate, &bag)
        .expect("comparison BinaryOp CREATE property admitted + executed");
    assert_eq!(
        read_single_prop(&substrate, "gt"),
        Value::Boolean(false),
        "6>7 evaluates to Boolean(false) and round-trips through storage"
    );
}

#[test]
fn n1_logical_and_binaryop_evaluates_and_stores_boolean() {
    // `$a AND $b` with $a=true, $b=false → Boolean(false) (the 3VL `And`
    // arm of `apply_binop`). Executed through the full query pipeline via
    // `RETURN` (grammar admits whitespaced keyword ops there); the
    // computed Boolean then round-trips through the CREATE store path.
    let out = eval_logical_return("AND", Value::Boolean(true), Value::Boolean(false));
    assert_eq!(
        out,
        Value::Boolean(false),
        "true AND false evaluates to Boolean(false) via the runtime evaluator"
    );
    let substrate = StubExecutorSubstrate::new();
    let bag = params(&[("b", out.clone())]);
    run_params("CREATE (n:User {both: $b}) RETURN n", &substrate, &bag).expect("store OK");
    assert_eq!(
        read_single_prop(&substrate, "both"),
        out,
        "the AND-computed Boolean round-trips through storage"
    );
}

#[test]
fn n1_xor_binaryop_evaluates_and_stores_boolean() {
    // `$a XOR $b` with $a=true, $b=false → Boolean(true) (the 3VL `Xor`
    // arm of `apply_binop`; the ultracode noted Xor is benign). Same
    // pipeline as AND — execute the arm through `RETURN`, then store the
    // computed Boolean via the CREATE property path.
    let out = eval_logical_return("XOR", Value::Boolean(true), Value::Boolean(false));
    assert_eq!(
        out,
        Value::Boolean(true),
        "true XOR false evaluates to Boolean(true) via the runtime evaluator"
    );
    let substrate = StubExecutorSubstrate::new();
    let bag = params(&[("x", out.clone())]);
    run_params("CREATE (n:User {x: $x}) RETURN n", &substrate, &bag).expect("store OK");
    assert_eq!(
        read_single_prop(&substrate, "x"),
        out,
        "the XOR-computed Boolean round-trips through storage"
    );
}

// =====================================================================
// B1 — write-path OOM DoS: the admitted `BinOp::Add` concat amplifier.
// `is_whitelisted_binop` admits `Add`, and the predicate recurses both
// operands, so a bracketed doubling tree `{x: (($a+$a)+($a+$a))+…}` with
// `$a` a list/string TYPE-CHECKS. WITHOUT the per-op cap inside
// `eval::add_or_concat`, `evaluate` materializes every intermediate up to
// ~2^depth → OOM; the result-only `MAX_CREATE_PROP_LIST_LEN` cap gates
// the RESULT not the intermediates. Each test asserts a CLEAN typed error
// + 0 nodes (NOT OOM). Kept fast/deterministic by sizing the base param
// so the FIRST over-cap concat trips at the length-CHECK before the
// allocation (no exponential blowup runs). The per-op cap's genuine
// RED-on-revert amplification-kill is unit-tested at a SMALL cap in
// `eval.rs::tests` (`checked_concat_doubling_tree_dies_at_first_over_cap_
// node` / `_flat_chain_over_cap_errors`), where deleting the check makes
// the tree build 2^depth elements instead of erroring.
// =====================================================================

/// The eval-layer per-op list cap (`eval::MAX_CONCAT_LIST_LEN` == 1M).
/// Sized just over half so a single `$a + $a` list concat crosses it.
const OVER_HALF_LIST_CAP: usize = 600_000;

/// A string just over half the eval-layer string byte cap
/// (`eval::MAX_CONCAT_STRING_BYTES` == 16 MiB), so one `$s + $s` string
/// concat crosses it. There is NO result-level string cap, so this test
/// is genuinely RED-on-revert at the integration layer: without the
/// per-op string cap the concat succeeds and stores a ~17 MiB string.
const OVER_HALF_STRING_CAP_BYTES: usize = 9 * 1024 * 1024;

#[test]
fn b1_flat_chain_list_concat_over_cap_clean_error_zero_nodes() {
    // A flat `$a + $a` where `$a` is a 600K-element list → 1.2M > 1M cap.
    // WITH the fix: the per-op length-check errors at the concat BEFORE
    // allocating the 1.2M list. Clean typed error, 0 nodes persisted.
    let substrate = StubExecutorSubstrate::new();
    let big = Value::List(vec![Value::Integer(0); OVER_HALF_LIST_CAP]);
    let bag = params(&[("a", big)]);
    let err = run_params("CREATE (n:User {x: $a+$a}) RETURN n", &substrate, &bag)
        .expect_err("an over-cap list concat must be a clean typed error, not OOM");
    assert!(
        err.contains("exceeding cap") && err.contains("concatenation"),
        "error names the concat cap (ADR-147-amendment-03 §B1); got {err}"
    );
    assert_eq!(
        count_nodes(&substrate),
        0,
        "0 nodes persisted when the concat is over-cap (no partial write)"
    );
}

#[test]
fn b1_nested_doubling_tree_list_dies_at_first_over_cap_node_zero_nodes() {
    // A depth-2 doubling tree `(($a+$a)+($a+$a))` with `$a` a 600K list.
    // The INNERMOST `$a+$a` is already 1.2M > 1M, so evaluation errors at
    // the FIRST fold — the outer folds (which WITHOUT the cap would reach
    // 2.4M then 4.8M …) never run. Clean typed error, 0 nodes. This is the
    // shape that WITHOUT the per-op cap amplifies ~2^depth and OOMs.
    let substrate = StubExecutorSubstrate::new();
    let big = Value::List(vec![Value::Integer(0); OVER_HALF_LIST_CAP]);
    let bag = params(&[("a", big)]);
    let err = run_params(
        "CREATE (n:User {x: (($a+$a)+($a+$a))}) RETURN n",
        &substrate,
        &bag,
    )
    .expect_err("a doubling-tree concat must die at the first over-cap node, not OOM");
    assert!(
        err.contains("exceeding cap") && err.contains("concatenation"),
        "error names the concat cap; got {err}"
    );
    assert_eq!(
        count_nodes(&substrate),
        0,
        "0 nodes on the doubling-tree reject"
    );
}

#[test]
fn b1_string_concat_over_cap_clean_error_zero_nodes() {
    // A `$s + $s` where `$s` is a 9 MiB string → 18 MiB > 16 MiB cap. NO
    // result-level string cap exists, so the per-op string byte cap is the
    // SOLE backstop → genuinely RED-on-revert at this layer (delete the
    // check ⇒ this stores an 18 MiB string instead of erroring). WITH the
    // fix: clean typed error at the length-check before the push_str.
    let substrate = StubExecutorSubstrate::new();
    let big = Value::String("x".repeat(OVER_HALF_STRING_CAP_BYTES));
    let bag = params(&[("s", big)]);
    let err = run_params("CREATE (n:User {s: $s+$s}) RETURN n", &substrate, &bag)
        .expect_err("an over-cap string concat must be a clean typed error, not OOM");
    assert!(
        err.contains("exceeding cap") && err.contains("string"),
        "error names the string byte cap; got {err}"
    );
    assert_eq!(
        count_nodes(&substrate),
        0,
        "0 nodes persisted on the over-cap string concat"
    );
}

// =====================================================================
// T16 — classifier/evaluator drift cross-check: for the admitted AST
// shapes, feed CONSTRUCTED runtime Values (map-via-param, list-of-map)
// through evaluate→materialize→store and assert clean-error OR byte-
// identical read-back. Variant-level assertions alone are insufficient.
// HARD GATE (the runtime-fence cross-check).
// =====================================================================

#[test]
fn t16_runtime_value_cross_check_clean_error_or_byte_identical() {
    // (a) A scalar-via-param round-trips BYTE-IDENTICAL.
    {
        let substrate = StubExecutorSubstrate::new();
        let bag = params(&[("p", Value::String("héllo".into()))]);
        run_params("CREATE (n:User {s: $p}) RETURN n", &substrate, &bag).expect("scalar OK");
        let read = execute_params(
            &lower("MATCH (n:User) RETURN n.s"),
            &substrate,
            &fresh_ctx(),
            &Parameters::new(),
        )
        .expect("read OK");
        assert_eq!(
            read[0][0],
            Value::String("héllo".into()),
            "scalar byte-identical"
        );
    }
    // (b) A list-of-scalars-via-param round-trips BYTE-IDENTICAL.
    {
        let substrate = StubExecutorSubstrate::new();
        let list = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]);
        let bag = params(&[("p", list.clone())]);
        run_params("CREATE (n:User {xs: $p}) RETURN n", &substrate, &bag).expect("list OK");
        let read = execute_params(
            &lower("MATCH (n:User) RETURN n.xs"),
            &substrate,
            &fresh_ctx(),
            &Parameters::new(),
        )
        .expect("read OK");
        assert_eq!(read[0][0], list, "list-of-scalars byte-identical");
    }
    // (c) A map-via-param → CLEAN ERROR, 0 nodes (already T9, re-pinned
    //     here as the composite-result arm of the cross-check).
    {
        let substrate = StubExecutorSubstrate::new();
        let bag = params(&[("p", vmap(&[("k", Value::Integer(1))]))]);
        run_params("CREATE (n:User {m: $p}) RETURN n", &substrate, &bag)
            .expect_err("map-via-param → clean error");
        assert_eq!(
            count_nodes(&substrate),
            0,
            "no partial node on the composite-result reject"
        );
    }
    // (d) A list-CONTAINING-a-map-via-param → CLEAN ERROR, 0 nodes.
    {
        let substrate = StubExecutorSubstrate::new();
        let bad = Value::List(vec![Value::Integer(1), vmap(&[("k", Value::Integer(2))])]);
        let bag = params(&[("p", bad)]);
        run_params("CREATE (n:User {xs: $p}) RETURN n", &substrate, &bag)
            .expect_err("list-of-map-via-param → clean error");
        assert_eq!(
            count_nodes(&substrate),
            0,
            "no partial node on a list-of-map reject"
        );
    }
}

// =====================================================================
// T17 — property/drift pin: an admitted-shape whose runtime result is a
// scalar/list STORES; whose runtime result is composite → typed error.
// (A focused, deterministic stand-in for the brief's proptest — the
// classifier is exhaustively covered by unit tests in type_check.rs.)
// =====================================================================

#[test]
fn t17_admitted_shape_result_type_decides_store_vs_error() {
    let substrate = StubExecutorSubstrate::new();

    // Same admitted AST shape (`$p`), two runtime results:
    // scalar → stores; map → typed error, no node.
    let ok = run_params(
        "CREATE (n:User {v: $p}) RETURN n",
        &substrate,
        &params(&[("p", Value::Integer(42))]),
    );
    assert!(ok.is_ok(), "admitted shape + scalar result stores");
    assert_eq!(count_nodes(&substrate), 1);

    let bad = run_params(
        "CREATE (n:User {v: $p}) RETURN n",
        &substrate,
        &params(&[("p", vmap(&[("k", Value::Integer(1))]))]),
    );
    assert!(
        bad.is_err(),
        "admitted shape + composite result → typed error"
    );
    assert_eq!(
        count_nodes(&substrate),
        1,
        "no additional node from the rejected composite"
    );
}

// =====================================================================
// Helpers
// =====================================================================

/// Return the property bag of the single created node (asserts exactly
/// one node exists across the label space).
fn single_node_bag(substrate: &StubExecutorSubstrate) -> std::collections::HashMap<String, Value> {
    // The StubExecutorSubstrate assigns node ids deterministically; the
    // first created node under the User label is id `(FIRST_LABEL << 32) + 1`
    // per the stub's allocation. Rather than couple to that arithmetic, we
    // scan the stub's recorded bags via the public accessor over the
    // created node's id, discovered by MATCH.
    let ctx = fresh_ctx();
    let nodes = execute_params(
        &lower("MATCH (n:User) RETURN n"),
        substrate,
        &ctx,
        &Parameters::new(),
    )
    .expect("scan OK");
    assert_eq!(
        nodes.len(),
        1,
        "expected exactly one node for the bag probe"
    );
    match &nodes[0][0] {
        Value::Node(view) => substrate
            .node_properties(TenantId::DEFAULT, view.id)
            .unwrap_or_default(),
        other => panic!("expected a Node cell, got {other:?}"),
    }
}
