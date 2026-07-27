//! ADR-152 W27-α — end-to-end property-bag persistence through the
//! production [`CrudExecutorSubstrate`].
//!
//! Closes the audit-identified semantic narrowing at 2026-05-27 by
//! exercising the CREATE-then-MATCH-by-property round-trip against
//! the real Buffer-pool + Router + CrudStore + TxnManager + BlobStore
//! stack (NOT the in-memory StubExecutorSubstrate).
//!
//! Walks:
//!
//! 1. `fixture()` builds a production-shaped substrate (catalog
//!    bootstrapped, CrudStore + TxnManager + InternTable wired
//!    through a fresh MultiTenantRouter).
//! 2. Round-trip 1: CREATE with literal property bag → scan_nodes
//!    returns NodeView carrying every persisted key/value.
//! 3. Round-trip 2: CREATE then SET property → scan_nodes returns
//!    the merged bag.
//! 4. Round-trip 3: CREATE then REMOVE property → scan_nodes
//!    returns the bag minus the removed key.
//! 5. Round-trip 4: rel-side — create_rel with properties → expand
//!    returns RelView carrying the persisted bag.
//!
//! The substrate's `create_node` / `set_node` / `remove_node` route
//! through `crud::create_node` / `crud::update_node` per ADR-031 +
//! ADR-033; blob serialization committed via the `BlobStore` chain.

use std::sync::Arc;

use arcgraph_core::{Lsn, TenantId};
use arcgraph_mcp::storage::substrate::CrudExecutorSubstrate;
use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::substrate::{
    RemoveNodeMutation, RemoveRelMutation, SetNodeMutation, SetRelMutation,
};
use arcgraph_query::executor::value::Value;
use arcgraph_query::logical_plan::Direction;
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;

/// Production-shaped substrate fixture. Mirrors the
/// `mcp_create_node_e2e.rs` fixture but adds nothing — the BlobStore
/// is internal to `CrudStore`.
fn fixture() -> CrudExecutorSubstrate {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap catalog");
    let crud = Arc::new(CrudStore::new());
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    CrudExecutorSubstrate::new(router, mgr, intern)
}

#[test]
fn mcp_e2e_create_then_scan_round_trips_property_bag() {
    // The audit's smoking-gun case translated to the production
    // substrate trait surface. CREATE persists; scan reads back.
    let sub = fixture();
    let tenant = TenantId::DEFAULT;

    let props = vec![
        ("id".to_string(), Value::Integer(42)),
        ("name".to_string(), Value::String("Alice".into())),
        ("flag".to_string(), Value::Boolean(true)),
    ];
    let node_id = sub
        .create_node(
            tenant,
            Some("User"),
            &props,
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create_node OK");

    let scanned = sub
        .scan_nodes(tenant, None, Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(scanned.len(), 1, "post-CREATE: 1 node observed");
    assert_eq!(
        scanned[0].node.id, node_id,
        "scanned node carries the assigned id"
    );
    assert_eq!(
        scanned[0].node.properties.get("id"),
        Some(&Value::Integer(42)),
        "property `id` round-trips through PropertyData::Blob + BlobStore"
    );
    assert_eq!(
        scanned[0].node.properties.get("name"),
        Some(&Value::String("Alice".into())),
        "property `name` round-trips"
    );
    assert_eq!(
        scanned[0].node.properties.get("flag"),
        Some(&Value::Boolean(true)),
        "property `flag` round-trips"
    );
}

#[test]
fn mcp_e2e_node_by_id_point_read_hydrates_label_and_properties() {
    let sub = fixture();
    let tenant = TenantId::DEFAULT;
    let ctx =
        arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO);
    let props = vec![
        ("id".to_string(), Value::String("a".into())),
        ("rank".to_string(), Value::Integer(7)),
    ];
    let node_id = sub
        .create_node(tenant, Some("PointRead"), &props, &ctx)
        .expect("create_node OK");

    let hydrated = sub
        .node_by_id_with_context(&ctx, node_id)
        .expect("node_by_id_with_context OK")
        .expect("created node is visible to point-read");

    assert_eq!(hydrated.node.id, node_id);
    assert_eq!(
        hydrated.node.label_name.as_deref(),
        Some("PointRead"),
        "point-read must reverse-resolve label names like scan_nodes"
    );
    assert_eq!(
        hydrated.node.properties.get("id"),
        Some(&Value::String("a".into()))
    );
    assert_eq!(
        hydrated.node.properties.get("rank"),
        Some(&Value::Integer(7))
    );
}

#[test]
fn mcp_e2e_create_empty_bag_routes_to_property_data_empty_fast_path() {
    // An empty property slice should NOT publish a blob (BlobStore
    // rejects zero-length payloads per BlobError::Empty). The substrate
    // routes to `PropertyData::Empty` via the helper's empty-fast-path
    // per ADR-152 §D-1.
    let sub = fixture();
    let tenant = TenantId::DEFAULT;

    let _ = sub
        .create_node(
            tenant,
            Some("User"),
            &[],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create_node empty OK");

    let scanned = sub.scan_nodes(tenant, None, Lsn::MAX).expect("scan");
    assert_eq!(scanned.len(), 1);
    assert!(
        scanned[0].node.properties.is_empty(),
        "empty CREATE bag round-trips to empty NodeView.properties"
    );
}

#[test]
fn mcp_e2e_set_property_assign_merges_with_existing_bag() {
    // ADR-152 §D-2: SET PropertyAssign reads the current bag, inserts
    // the new key, writes back. The other keys are preserved.
    let sub = fixture();
    let tenant = TenantId::DEFAULT;

    let node_id = sub
        .create_node(
            tenant,
            Some("User"),
            &[("id".to_string(), Value::Integer(42))],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create OK");

    // SET name="Alice".
    sub.set_node(
        tenant,
        node_id,
        &SetNodeMutation::PropertyAssign {
            name: "name".into(),
            value: Value::String("Alice".into()),
        },
        &arcgraph_query::executor::ExecutionContext::new(
            TenantId::DEFAULT,
            arcgraph_core::PartitionId::ZERO,
        ),
    )
    .expect("set_node OK");

    let scanned = sub.scan_nodes(tenant, None, Lsn::MAX).expect("scan");
    assert_eq!(scanned.len(), 1);
    assert_eq!(
        scanned[0].node.properties.get("id"),
        Some(&Value::Integer(42)),
        "pre-SET `id` is preserved"
    );
    assert_eq!(
        scanned[0].node.properties.get("name"),
        Some(&Value::String("Alice".into())),
        "post-SET `name` is observable"
    );
}

#[test]
fn mcp_e2e_set_property_replace_overwrites_full_bag() {
    let sub = fixture();
    let tenant = TenantId::DEFAULT;
    let node_id = sub
        .create_node(
            tenant,
            Some("User"),
            &[
                ("id".to_string(), Value::Integer(42)),
                ("legacy".to_string(), Value::String("old".into())),
            ],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create OK");

    // SET n = {name: "Alice"} — replace semantic per ADR-150 §D-1.
    sub.set_node(
        tenant,
        node_id,
        &SetNodeMutation::PropertyReplace(vec![(
            "name".to_string(),
            Value::String("Alice".into()),
        )]),
        &arcgraph_query::executor::ExecutionContext::new(
            TenantId::DEFAULT,
            arcgraph_core::PartitionId::ZERO,
        ),
    )
    .expect("set_node replace OK");

    let scanned = sub.scan_nodes(tenant, None, Lsn::MAX).expect("scan");
    assert_eq!(scanned.len(), 1);
    let bag = &scanned[0].node.properties;
    assert_eq!(bag.len(), 1, "replace overwrites the full bag");
    assert_eq!(bag.get("name"), Some(&Value::String("Alice".into())));
    assert!(
        !bag.contains_key("id"),
        "PropertyReplace clears keys not in the new entries"
    );
    assert!(!bag.contains_key("legacy"));
}

#[test]
fn mcp_e2e_set_property_merge_additive() {
    let sub = fixture();
    let tenant = TenantId::DEFAULT;
    let node_id = sub
        .create_node(
            tenant,
            Some("User"),
            &[("id".to_string(), Value::Integer(42))],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create OK");

    // SET n += {name: "Alice", id: 99} — additive merge per ADR-150 §D-1.
    sub.set_node(
        tenant,
        node_id,
        &SetNodeMutation::PropertyMerge(vec![
            ("name".to_string(), Value::String("Alice".into())),
            ("id".to_string(), Value::Integer(99)),
        ]),
        &arcgraph_query::executor::ExecutionContext::new(
            TenantId::DEFAULT,
            arcgraph_core::PartitionId::ZERO,
        ),
    )
    .expect("set_node merge OK");

    let scanned = sub.scan_nodes(tenant, None, Lsn::MAX).expect("scan");
    assert_eq!(scanned.len(), 1);
    let bag = &scanned[0].node.properties;
    assert_eq!(
        bag.get("id"),
        Some(&Value::Integer(99)),
        "merge OVERWRITES same-key entries"
    );
    assert_eq!(
        bag.get("name"),
        Some(&Value::String("Alice".into())),
        "merge inserts new key"
    );
}

#[test]
fn mcp_e2e_remove_property_drops_key() {
    let sub = fixture();
    let tenant = TenantId::DEFAULT;
    let node_id = sub
        .create_node(
            tenant,
            Some("User"),
            &[
                ("id".to_string(), Value::Integer(42)),
                ("name".to_string(), Value::String("Alice".into())),
            ],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create OK");

    sub.remove_node(
        tenant,
        node_id,
        &RemoveNodeMutation::Property("name".to_string()),
        &arcgraph_query::executor::ExecutionContext::new(
            TenantId::DEFAULT,
            arcgraph_core::PartitionId::ZERO,
        ),
    )
    .expect("remove_node OK");

    let scanned = sub.scan_nodes(tenant, None, Lsn::MAX).expect("scan");
    assert_eq!(scanned.len(), 1);
    let bag = &scanned[0].node.properties;
    assert!(!bag.contains_key("name"), "removed key is absent");
    assert_eq!(
        bag.get("id"),
        Some(&Value::Integer(42)),
        "other keys are preserved"
    );
}

#[test]
fn mcp_e2e_create_rel_round_trips_property_bag_via_expand() {
    let sub = fixture();
    let tenant = TenantId::DEFAULT;

    let src = sub
        .create_node(
            tenant,
            Some("User"),
            &[],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create src");
    let dst = sub
        .create_node(
            tenant,
            Some("User"),
            &[],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create dst");

    let _rel = sub
        .create_rel(
            tenant,
            src,
            dst,
            "KNOWS",
            &[
                ("since".to_string(), Value::Integer(2020)),
                ("weight".to_string(), Value::Float(0.75)),
            ],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create_rel OK");

    let edges = sub
        .expand(tenant, src, None, Direction::LeftToRight, Lsn::MAX)
        .expect("expand OK");
    assert_eq!(edges.len(), 1, "single rel observed");
    let bag = &edges[0].rel.properties;
    assert_eq!(
        bag.get("since"),
        Some(&Value::Integer(2020)),
        "rel property `since` round-trips through BlobStore"
    );
    // Float round-trip: bag.get returns Value::Float; compare via
    // as_f64 to be NaN-tolerant.
    match bag.get("weight") {
        Some(Value::Float(f)) => assert!((f - 0.75).abs() < 1e-9),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn mcp_e2e_set_rel_property_round_trip() {
    let sub = fixture();
    let tenant = TenantId::DEFAULT;
    let src = sub
        .create_node(
            tenant,
            Some("User"),
            &[],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create src");
    let dst = sub
        .create_node(
            tenant,
            Some("User"),
            &[],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create dst");
    let rel_id = sub
        .create_rel(
            tenant,
            src,
            dst,
            "KNOWS",
            &[],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create_rel OK");

    sub.set_rel(
        tenant,
        rel_id,
        &SetRelMutation::PropertyAssign {
            name: "weight".into(),
            value: Value::Float(0.5),
        },
        &arcgraph_query::executor::ExecutionContext::new(
            TenantId::DEFAULT,
            arcgraph_core::PartitionId::ZERO,
        ),
    )
    .expect("set_rel OK");

    let edges = sub
        .expand(tenant, src, None, Direction::LeftToRight, Lsn::MAX)
        .expect("expand OK");
    assert_eq!(edges.len(), 1);
    match edges[0].rel.properties.get("weight") {
        Some(Value::Float(f)) => assert!((f - 0.5).abs() < 1e-9),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn mcp_e2e_remove_rel_property_round_trip() {
    let sub = fixture();
    let tenant = TenantId::DEFAULT;
    let src = sub
        .create_node(
            tenant,
            Some("User"),
            &[],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create src");
    let dst = sub
        .create_node(
            tenant,
            Some("User"),
            &[],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create dst");
    let rel_id = sub
        .create_rel(
            tenant,
            src,
            dst,
            "KNOWS",
            &[
                ("since".to_string(), Value::Integer(2020)),
                ("weight".to_string(), Value::Float(0.75)),
            ],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create_rel OK");

    sub.remove_rel(
        tenant,
        rel_id,
        &RemoveRelMutation::Property("weight".to_string()),
        &arcgraph_query::executor::ExecutionContext::new(
            TenantId::DEFAULT,
            arcgraph_core::PartitionId::ZERO,
        ),
    )
    .expect("remove_rel OK");

    let edges = sub
        .expand(tenant, src, None, Direction::LeftToRight, Lsn::MAX)
        .expect("expand OK");
    assert_eq!(edges.len(), 1);
    let bag = &edges[0].rel.properties;
    assert!(
        !bag.contains_key("weight"),
        "removed rel property is absent"
    );
    assert_eq!(
        bag.get("since"),
        Some(&Value::Integer(2020)),
        "other rel properties preserved"
    );
}
