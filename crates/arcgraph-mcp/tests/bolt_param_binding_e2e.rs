//! **#797 (HIGH, customer-zero) — Bolt RUN `$param` binding end-to-end.**
//!
//! The neo4j-universal idiom `session.run("… $p …", {p: value})` reaches
//! the production [`StorageBoltHandler::run`] / `run_in_txn` path with a
//! populated `parameters` map. Pre-#797 the handler under-scored that map
//! away (`_parameters`) and called the no-params engine entry point, so
//! `$p` reached the executor UNBOUND → `DatabaseError "internal: missing
//! parameter: $p"`. This pins that the wire parameter map is now
//! converted (PackValue → ExecValue) and threaded into the query engine
//! so `$p` resolves at runtime.
//!
//! # The DISCRIMINATING oracle (RED-without-the-fix)
//!
//! On `origin/main` every case below raised `missing parameter` (the
//! handler dropped the bag); `bolt.run(...).expect(...)` PANICS → the
//! test FAILS. With the fix the runtime substitution binds the value and
//! the exact records / fields assert. (Verified RED-on-revert in the PR.)
//!
//! # Why in-process (no socket)
//!
//! Hermetic ADR-133 §D-4 "Driver"-class active-verification gate: the
//! SAME production `StorageBoltHandler` + `QueryEngine` +
//! `CrudExecutorSubstrate` the wire server drives, minus the TCP/
//! PackStream frame layer (which is exercised by the wire-level
//! `arcgraph-cli` driver-compat subprocess test). The oracles are
//! data-free (parameter echo / arithmetic / UNWIND / WHERE) so they pin
//! the binding mechanism without depending on the v1.0-α property
//! round-trip posture.
//!
//! # Scope boundary (ADR-147 Phase 2 read-side + UNWIND)
//!
//! Read-side (`WHERE`/`RETURN`/arithmetic/function-arg) + UNWIND `$data`
//! bind via the runtime substitution. Write-side property values
//! (`CREATE (n {v: $v})`) stay rejected per ADR-147 §D-4 / Forward-
//! deferred ("requires parameter-eval at CreateNodeOp time", v1.1) — the
//! last test pins that the forward-pin is HONORED, not silently lifted.

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::SessionScope;
use arcgraph_mcp::storage::{StorageBackend, StorageBoltHandler};
use arcgraph_mcp::transport::bolt::{
    BoltError, BoltQueryHandler, BoltSessionAuth, PackValue, TAG_NODE,
};
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;

/// Fresh in-memory backend with PrimaryIndex wired (mirrors
/// `return_alias_columns_wire_e2e::fresh_backend`) so the few
/// data-touching paths behave like production; the param oracles
/// themselves are data-free.
fn fresh_backend() -> StorageBackend {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(64, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("catalog bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&allocator), None).expect("PrimaryIndex"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    StorageBackend::new(router, mgr, intern)
}

fn handler() -> (StorageBoltHandler, BoltSessionAuth) {
    let bolt = StorageBoltHandler::new(fresh_backend());
    let session = BoltSessionAuth::new(TenantId::DEFAULT, None, SessionScope::Power);
    (bolt, session)
}

fn params(pairs: &[(&str, PackValue)]) -> BTreeMap<String, PackValue> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

/// `RETURN $x AS y` with `{x: 42}` → exactly `[[42]]`, column `y`.
#[test]
fn scalar_param_in_return_binds() {
    let (bolt, tenant) = handler();
    let out = bolt
        .run(
            &tenant,
            "RETURN $x AS y",
            &params(&[("x", PackValue::Integer(42))]),
        )
        .expect("RETURN $x bound (RED on origin/main: `missing parameter: $x`)");
    assert_eq!(out.fields, vec!["y".to_string()], "alias column");
    assert_eq!(
        out.records,
        vec![vec![PackValue::Integer(42)]],
        "row carries the bound parameter value"
    );
}

/// Parameters in arithmetic: `RETURN $x + $y AS z` with `{x:40, y:2}`.
#[test]
fn params_in_arithmetic_bind() {
    let (bolt, tenant) = handler();
    let out = bolt
        .run(
            &tenant,
            "RETURN $x + $y AS z",
            &params(&[("x", PackValue::Integer(40)), ("y", PackValue::Integer(2))]),
        )
        .expect("arithmetic over params");
    assert_eq!(out.records, vec![vec![PackValue::Integer(42)]]);
}

/// String parameter: `RETURN $s AS y` with `{s: "hello"}`.
#[test]
fn string_param_binds() {
    let (bolt, tenant) = handler();
    let out = bolt
        .run(
            &tenant,
            "RETURN $s AS y",
            &params(&[("s", PackValue::String("hello".into()))]),
        )
        .expect("string param");
    assert_eq!(out.records, vec![vec![PackValue::String("hello".into())]]);
}

/// **The langchain ingest mechanism** — `UNWIND $data AS row` binds a
/// LIST parameter and fans it out. `UNWIND $xs AS x RETURN x` with
/// `{xs:[1,2,3]}` → `[[1],[2],[3]]`.
#[test]
fn list_param_feeds_unwind() {
    let (bolt, tenant) = handler();
    let xs = PackValue::List(vec![
        PackValue::Integer(1),
        PackValue::Integer(2),
        PackValue::Integer(3),
    ]);
    let out = bolt
        .run(&tenant, "UNWIND $xs AS x RETURN x", &params(&[("xs", xs)]))
        .expect("UNWIND $xs bound (the #830/#865 langchain $data path)");
    assert_eq!(
        out.records,
        vec![
            vec![PackValue::Integer(1)],
            vec![PackValue::Integer(2)],
            vec![PackValue::Integer(3)],
        ]
    );
}

/// Parameter in a WHERE predicate (post-UNWIND so the oracle is
/// data-free): `UNWIND $xs AS x WITH x WHERE x >= $lo RETURN x` with
/// `{xs:[1,2,3], lo:2}` → `[[2],[3]]`.
#[test]
fn param_in_where_predicate_binds() {
    let (bolt, tenant) = handler();
    let xs = PackValue::List(vec![
        PackValue::Integer(1),
        PackValue::Integer(2),
        PackValue::Integer(3),
    ]);
    let out = bolt
        .run(
            &tenant,
            "UNWIND $xs AS x WITH x WHERE x >= $lo RETURN x",
            &params(&[("xs", xs), ("lo", PackValue::Integer(2))]),
        )
        .expect("WHERE over a param");
    assert_eq!(
        out.records,
        vec![vec![PackValue::Integer(2)], vec![PackValue::Integer(3)]]
    );
}

/// A referenced-but-unbound `$missing` is a CLIENT fault →
/// [`BoltError::ParameterMissing`] (`Neo.ClientError.Statement.
/// ParameterMissing`), NEVER a panic, a silent NULL, or the pre-#797
/// `Internal`/`DatabaseError` bucket.
#[test]
fn missing_parameter_is_a_clean_client_error() {
    let (bolt, tenant) = handler();
    let err = bolt
        .run(&tenant, "RETURN $missing AS y", &BTreeMap::new())
        .expect_err("unbound $missing must error, not return NULL");
    assert!(
        matches!(err, BoltError::ParameterMissing(_)),
        "expected ParameterMissing, got {err:?}"
    );
    assert_eq!(
        err.neo4j_code(),
        "Neo.ClientError.Statement.ParameterMissing"
    );
}

/// A graph-entity parameter VALUE (a Node struct — the `exec_to_pack`
/// OUTPUT shape fed back in as an INPUT) is rejected as an invalid
/// parameter shape (`Neo.ClientError.Statement.TypeError`), even when
/// the statement doesn't reference it.
#[test]
fn graph_entity_parameter_is_rejected() {
    let (bolt, tenant) = handler();
    let node = PackValue::Struct {
        tag: TAG_NODE,
        fields: vec![
            PackValue::Integer(1),
            PackValue::List(vec![]),
            PackValue::Map(BTreeMap::new()),
            PackValue::String("4::1".into()),
        ],
    };
    let err = bolt
        .run(&tenant, "RETURN 1 AS y", &params(&[("n", node)]))
        .expect_err("Node-shaped param must be rejected");
    assert!(
        matches!(err, BoltError::InvalidParameter(_)),
        "expected InvalidParameter, got {err:?}"
    );
    assert_eq!(err.neo4j_code(), "Neo.ClientError.Statement.TypeError");
}

/// **T3 (ADR-147-amendment-03, D-1) — FLIP.** Write-side parameter-typed
/// CREATE property values are now ADMITTED and STORED. The live
/// `CreateSpineOp` executor `evaluate`s `$v` against the parameter bag,
/// value-gates the result, and persists it. Was
/// `write_side_create_param_property_stays_rejected_per_adr147` (which
/// expect_err'd a `BoltError::Syntax` — RED on revert).
#[test]
fn write_side_create_param_property_is_accepted_and_round_trips() {
    let (bolt, tenant) = handler();
    // CREATE with a $param property now SUCCEEDS.
    bolt.run(
        &tenant,
        "CREATE (n:P {v: $v})",
        &params(&[("v", PackValue::Integer(2))]),
    )
    .expect("CREATE property param admitted + stored (amendment-03)");

    // Round-trip: MATCH reads back the stored value (proves it persisted
    // through the JSON-blob property-bag path, not a silent no-op).
    let out = bolt
        .run(&tenant, "MATCH (n:P) RETURN n.v AS v", &params(&[]))
        .expect("MATCH read-back OK");
    assert_eq!(out.fields, vec!["v".to_string()], "alias column");
    assert_eq!(
        out.records,
        vec![vec![PackValue::Integer(2)]],
        "n.v round-trips as the bound parameter value"
    );
}

/// **T-scope (ADR-147-amendment-03, D-1) — scope airtightness + the
/// "literal"-message pin.** A MAP-LITERAL CREATE property value STAYS
/// rejected at type-check (openCypher forbids map property values;
/// ADR-191 D-11). Amendment-03 lifted params / row-refs / bounded exprs,
/// NOT maps. The rejection message MUST keep the word "literal" (the
/// client-facing contract this harness pins) — replaces the old
/// param-rejection pin, preserving that assertion. (`{a:1}` no inner
/// space — the compound-atomic prop grammar suppresses implicit
/// whitespace, a pre-existing constraint out of D-1 scope.)
#[test]
fn write_side_create_map_literal_property_stays_rejected_with_literal_message() {
    let (bolt, tenant) = handler();
    let err = bolt
        .run(&tenant, "CREATE (n:P {m: {a:1}})", &params(&[]))
        .expect_err("map-literal CREATE property stays rejected (amendment-03)");
    // Type-check rejection → Bolt Syntax. Crucially NOT a silent success.
    assert!(
        matches!(err, BoltError::Syntax(_)),
        "expected a type-check Syntax rejection, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("literal") || msg.contains("CreatePropertyValueNotLiteral"),
        "rejection should cite the literal-only-plus gate (keeps the word 'literal'); got: {msg}"
    );
}

/// **T9 (ADR-147-amendment-03, D-1) — the runtime map fence, over Bolt.**
/// A `$param` bound to a MAP value passes the AST-shape check but is
/// rejected at the executor value-type gate BEFORE the write — 0 nodes
/// persisted. This is the load-bearing guard (a naive evaluate-swap
/// without the gate would corrupt live data).
#[test]
fn write_side_create_map_via_param_rejected_no_partial_node() {
    let (bolt, tenant) = handler();
    let map_param = PackValue::Map(
        [("k".to_string(), PackValue::Integer(1))]
            .into_iter()
            .collect(),
    );
    let err = bolt
        .run(
            &tenant,
            "CREATE (n:P {m: $m})",
            &params(&[("m", map_param)]),
        )
        .expect_err("a map-via-param CREATE property must be rejected at the value gate");
    // A runtime execution error surfaces as a client-class Bolt error.
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("map") || msg.contains("property value"),
        "error names the map/property fence; got: {msg}"
    );
    // 0 nodes persisted (no silent corruption).
    let out = bolt
        .run(&tenant, "MATCH (n:P) RETURN n", &params(&[]))
        .expect("MATCH scan OK");
    assert!(
        out.records.is_empty(),
        "0 nodes persisted when the CREATE property is a runtime map; got {} rows",
        out.records.len()
    );
}
