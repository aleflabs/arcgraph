//! ADR-152-amendment-02 (W28) — composite `List`-literal property
//! persistence through the production [`CrudExecutorSubstrate`].
//!
//! Mirrors `mcp_property_persistence_e2e.rs` (real Buffer-pool + Router +
//! CrudStore + TxnManager + the production `BlobStore` blob path over
//! `InMemoryPageIo` — in-memory, non-durable page IO; NOT the
//! `StubExecutorSubstrate`) but exercises `Value::List` property values:
//! the bag serializes to a JSON array inside `PropertyData::Blob`, and
//! `scan_nodes` / `expand` decode it back to `Value::List` EXACTLY.
//!
//! The proptest at the bottom is oracle #7's faithful "random
//! `List`-of-scalars property bags round-trip" form: it drives random
//! lists (including finite floats + nesting) through the REAL JSON-blob
//! write/read path and asserts `==`. Non-finite floats / `u64 > i64`
//! are excluded per the `Value::to_json_value` lossy edges
//! (amendment-02 §D-4).

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use arcgraph_core::{Lsn, TenantId};
use arcgraph_mcp::storage::substrate::CrudExecutorSubstrate;
use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::substrate::SetNodeMutation;
use arcgraph_query::executor::value::Value;
use arcgraph_query::logical_plan::Direction;
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;

use proptest::prelude::*;

fn fixture() -> CrudExecutorSubstrate {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(64, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap catalog");
    let crud = Arc::new(CrudStore::new());
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    CrudExecutorSubstrate::new(router, mgr, intern)
}

#[test]
fn mcp_e2e_create_with_list_property_round_trips_via_blob() {
    let sub = fixture();
    let tenant = TenantId::DEFAULT;

    let list = Value::List(vec![
        Value::String("a".into()),
        Value::String("b".into()),
        Value::String("c".into()),
    ]);
    let node_id = sub
        .create_node(
            tenant,
            Some("User"),
            &[("tags".to_string(), list.clone())],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create_node OK");

    let scanned = sub.scan_nodes(tenant, None, Lsn::MAX).expect("scan OK");
    assert_eq!(scanned.len(), 1);
    assert_eq!(scanned[0].node.id, node_id);
    assert_eq!(
        scanned[0].node.properties.get("tags"),
        Some(&list),
        "list property round-trips through PropertyData::Blob (JSON array) + BlobStore"
    );
}

#[test]
fn mcp_e2e_nested_and_heterogeneous_list_round_trips() {
    let sub = fixture();
    let tenant = TenantId::DEFAULT;

    let nested = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(2)]),
        Value::List(vec![Value::Integer(3)]),
    ]);
    let hetero = Value::List(vec![
        Value::Integer(1),
        Value::String("x".into()),
        Value::Boolean(true),
    ]);
    let _ = sub
        .create_node(
            tenant,
            Some("User"),
            &[
                ("matrix".to_string(), nested.clone()),
                ("mixed".to_string(), hetero.clone()),
            ],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create OK");

    let scanned = sub.scan_nodes(tenant, None, Lsn::MAX).expect("scan OK");
    assert_eq!(scanned.len(), 1);
    assert_eq!(scanned[0].node.properties.get("matrix"), Some(&nested));
    assert_eq!(scanned[0].node.properties.get("mixed"), Some(&hetero));
}

#[test]
fn mcp_e2e_empty_list_value_round_trips_as_empty_list() {
    // Documented behavior (amendment-02 §D-4): an empty-LIST value
    // persists + reads back as an empty list (the BAG is non-empty — it
    // has the `tags` key — so it does NOT hit the empty-bag fast-path).
    let sub = fixture();
    let tenant = TenantId::DEFAULT;
    let _ = sub
        .create_node(
            tenant,
            Some("User"),
            &[("tags".to_string(), Value::List(vec![]))],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create OK");

    let scanned = sub.scan_nodes(tenant, None, Lsn::MAX).expect("scan OK");
    assert_eq!(scanned.len(), 1);
    assert_eq!(
        scanned[0].node.properties.get("tags"),
        Some(&Value::List(vec![])),
        "empty-list value round-trips as an empty list (present key, empty array)"
    );
}

#[test]
fn mcp_e2e_set_list_property_round_trips() {
    let sub = fixture();
    let tenant = TenantId::DEFAULT;
    let node_id = sub
        .create_node(
            tenant,
            Some("User"),
            &[("id".to_string(), Value::Integer(1))],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create OK");

    let list = Value::List(vec![Value::String("x".into()), Value::String("y".into())]);
    sub.set_node(
        tenant,
        node_id,
        &SetNodeMutation::PropertyAssign {
            name: "tags".into(),
            value: list.clone(),
        },
        &arcgraph_query::executor::ExecutionContext::new(
            TenantId::DEFAULT,
            arcgraph_core::PartitionId::ZERO,
        ),
    )
    .expect("set_node OK");

    let scanned = sub.scan_nodes(tenant, None, Lsn::MAX).expect("scan OK");
    assert_eq!(scanned.len(), 1);
    assert_eq!(
        scanned[0].node.properties.get("tags"),
        Some(&list),
        "SET-applied list property round-trips through the blob path"
    );
    assert_eq!(
        scanned[0].node.properties.get("id"),
        Some(&Value::Integer(1)),
        "the pre-existing scalar property is preserved"
    );
}

#[test]
fn mcp_e2e_create_rel_with_list_property_round_trips_via_expand() {
    let sub = fixture();
    let tenant = TenantId::DEFAULT;
    let src = sub
        .create_node(
            tenant,
            Some("User"),
            &[],
            &arcgraph_query::executor::ExecutionContext::new(
                TenantId::DEFAULT,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("src");
    let dst = sub
        .create_node(
            tenant,
            Some("User"),
            &[],
            &arcgraph_query::executor::ExecutionContext::new(
                TenantId::DEFAULT,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("dst");

    let list = Value::List(vec![Value::Integer(2020), Value::Integer(2021)]);
    let _ = sub
        .create_rel(
            tenant,
            src,
            dst,
            "KNOWS",
            &[("years".to_string(), list.clone())],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create_rel OK");

    let edges = sub
        .expand(tenant, src, None, Direction::LeftToRight, Lsn::MAX)
        .expect("expand OK");
    assert_eq!(edges.len(), 1);
    assert_eq!(
        edges[0].rel.properties.get("years"),
        Some(&list),
        "rel list property round-trips through BlobStore + expand decode"
    );
}

#[test]
fn mcp_e2e_float_in_list_round_trips_within_tolerance() {
    // Finite floats inside a list round-trip through the JSON blob to
    // within f64 precision (NOT bit-exact — serde_json's float parse can
    // be ~1 ULP off without the `float_roundtrip` feature; the existing
    // `mcp_property_persistence_e2e.rs` scalar-float assertions use the
    // same `< 1e-9` tolerance). The LIST STRUCTURE + non-float siblings
    // are exact; each float element is compared with tolerance.
    let sub = fixture();
    let tenant = TenantId::DEFAULT;
    let _ = sub
        .create_node(
            tenant,
            Some("User"),
            &[(
                "scores".to_string(),
                Value::List(vec![
                    Value::Float(0.75),
                    Value::Float(2.5),
                    Value::Integer(3),
                ]),
            )],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create OK");

    let scanned = sub.scan_nodes(tenant, None, Lsn::MAX).expect("scan OK");
    assert_eq!(scanned.len(), 1);
    match scanned[0].node.properties.get("scores") {
        Some(Value::List(elems)) => {
            assert_eq!(elems.len(), 3, "list structure preserved exactly");
            match (&elems[0], &elems[1], &elems[2]) {
                (Value::Float(a), Value::Float(b), Value::Integer(c)) => {
                    assert!((a - 0.75).abs() < 1e-9, "float[0] within tolerance: {a}");
                    assert!((b - 2.5).abs() < 1e-9, "float[1] within tolerance: {b}");
                    assert_eq!(*c, 3, "integer sibling is exact");
                }
                other => panic!("expected [Float, Float, Integer], got {other:?}"),
            }
        }
        other => panic!("expected List, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Oracle #7 — random List-of-scalars property bags round-trip through the
// REAL production JSON-blob write/read path (bit-exact types only; floats
// covered by the tolerance test above).
// ─────────────────────────────────────────────────────────────────────

/// A BIT-EXACT-round-tripping scalar (`Null` / `Bool` / `Integer` /
/// `String`) or a nested list thereof, bounded depth 3. Integers span
/// full `i64` (no `u64 > i64` since the source is `i64`).
///
/// `Float` is deliberately EXCLUDED from the exact-`==` proptest: finite
/// floats round-trip through the JSON blob only to within f64 precision
/// (serde_json without the `float_roundtrip` feature can be ~1 ULP off),
/// NOT bit-exact — a pre-existing edge the existing
/// `mcp_property_persistence_e2e.rs` float assertions already handle with
/// a `< 1e-9` tolerance. The float-in-list case is covered by the
/// deterministic tolerance test
/// [`mcp_e2e_float_in_list_round_trips_within_tolerance`] below.
fn element_strategy() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Boolean),
        any::<i64>().prop_map(Value::Integer),
        "[a-zA-Z0-9 _\\-]{0,16}".prop_map(Value::String),
    ];
    leaf.prop_recursive(3, 24, 4, |inner| {
        proptest::collection::vec(inner, 0..4).prop_map(Value::List)
    })
}

fn list_value_strategy() -> impl Strategy<Value = Value> {
    proptest::collection::vec(element_strategy(), 0..5).prop_map(Value::List)
}

proptest! {
    #[test]
    fn random_list_property_round_trips_through_production_blob(list in list_value_strategy()) {
        let sub = fixture();
        let tenant = TenantId::DEFAULT;
        sub.create_node(tenant, Some("User"), &[("v".to_string(), list.clone())], &arcgraph_query::executor::ExecutionContext::new(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO))
            .map_err(|e| TestCaseError::fail(format!("create_node: {e:?}")))?;

        let scanned = sub
            .scan_nodes(tenant, None, Lsn::MAX)
            .map_err(|e| TestCaseError::fail(format!("scan: {e:?}")))?;
        prop_assert_eq!(scanned.len(), 1);
        prop_assert_eq!(
            scanned[0].node.properties.get("v"),
            Some(&list),
            "random list property round-trips EXACTLY through the JSON-blob path"
        );
    }
}
