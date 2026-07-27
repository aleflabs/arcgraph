//! #1488 RC-blocker gate — principal ACL enforcement on graph traversal.
//!
//! Production-shaped chain:
//! standard MCP `tools/call` → Dispatcher → StorageNeighborhoodExplorer /
//! StorageNodeInspector → routed TenantHandle::permissions() →
//! EffectivePermissions::is_visible.
//!
//! The served session is bound to the non-DEFAULT SYSTEM tenant because the
//! v1 catalog's production tenant lifecycle currently exposes exactly DEFAULT
//! plus the SYSTEM carve-out. A DEFAULT-tenant canary is committed through the
//! same production backend so the gate also exercises two tenant identities.
//!
//! RED-on-revert anchors:
//! - remove the per-neighbor `retain_visible_reachable` call while retaining
//!   seed gating: `gate_fix_1488_*filters_denied*` returns the denied secret;
//! - replace the transport-threaded read scope with Power (the pre-fix
//!   principal-less SYSTEM-TRUSTED behavior): `gate_fix_1488_*minus_32008*`
//!   returns a non--32008 result.

use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::storage::{
    StorageBackend, StorageHybridSearcher, StorageIngestProvider, StorageNeighborhoodExplorer,
    StorageNodeInspector, StorageRawQueryExecutor, StorageSchemaProvider,
};
use arcgraph_mcp::tools::explore::Neighborhood;
use arcgraph_mcp::tools::inspect::NodeInspection;
use arcgraph_mcp::transport::handle_raw_envelope;
use arcgraph_mcp::{Dispatcher, SessionScope};
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use serde_json::{Value, json};

type ProductionDispatcher = Dispatcher<
    StorageSchemaProvider,
    StorageNodeInspector,
    StorageNeighborhoodExplorer,
    StorageHybridSearcher,
    StorageIngestProvider,
    StorageRawQueryExecutor,
>;

fn fresh_backend() -> StorageBackend {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(64, io);
    let txn_manager = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog
        .bootstrap(&pool, &txn_manager)
        .expect("catalog bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&txn_manager), Arc::clone(&allocator), None)
            .expect("primary index"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let router = Arc::new(MultiTenantRouter::new(catalog, crud, None));
    StorageBackend::new(router, txn_manager, Arc::new(InternTable::new()))
}

fn dispatcher(
    backend: &StorageBackend,
    tenant: TenantId,
    scope: SessionScope,
) -> ProductionDispatcher {
    Dispatcher::with_session_scope(
        tenant,
        scope,
        Arc::new(StorageSchemaProvider::new(backend.clone())),
        Arc::new(StorageNodeInspector::new(backend.clone())),
        Arc::new(StorageNeighborhoodExplorer::new(backend.clone())),
        Arc::new(StorageHybridSearcher::new(backend.clone())),
        Arc::new(StorageIngestProvider::new(backend.clone())),
        Arc::new(StorageRawQueryExecutor::new(backend.clone())),
    )
}

fn tools_call(dispatcher: &ProductionDispatcher, name: &str, arguments: Value) -> Value {
    handle_raw_envelope(
        dispatcher,
        json!({
            "jsonrpc": "2.0",
            "id": 1488,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments}
        }),
    )
    .expect("request, not notification")
}

fn successful_tool_result(
    dispatcher: &ProductionDispatcher,
    name: &str,
    arguments: Value,
) -> Value {
    let response = tools_call(dispatcher, name, arguments);
    assert!(
        response["error"].is_null(),
        "tools/call JSON-RPC failure: {response}"
    );
    assert_eq!(response["result"]["isError"], false, "{response}");
    serde_json::from_str(
        response["result"]["content"][0]["text"]
            .as_str()
            .expect("tool result text"),
    )
    .expect("inner tool result JSON")
}

fn ingest_acl_fixture(backend: &StorageBackend) -> (u64, u64) {
    let tenant = TenantId::SYSTEM;
    assert_ne!(tenant, TenantId::DEFAULT, "gate must bind non-DEFAULT");
    let power = dispatcher(backend, tenant, SessionScope::Power);
    let result = successful_tool_result(
        &power,
        "graph.ingest",
        json!({
            "tenant_id": tenant.raw(),
            "nodes": [
                {
                    "external_id": "fix-1488-seed",
                    "label": "Document",
                    "properties": {"body": "SEED_VISIBLE_1488"}
                },
                {
                    "external_id": "fix-1488-denied",
                    "label": "Document",
                    "properties": {"body": "DENIED_SECRET_1488"}
                }
            ],
            "relationships": [{
                "external_id": "fix-1488-adjacent",
                "from_external_id": "fix-1488-seed",
                "to_external_id": "fix-1488-denied",
                "rel_type": "LINKS_TO",
                "properties": {}
            }],
            "acl_grants": [
                {"external_id": "fix-1488-seed", "read_principals": ["alice"]},
                {"external_id": "fix-1488-denied", "read_principals": ["bob"]}
            ],
            "format": "json"
        }),
    );
    let summary: Value = serde_json::from_str(result["body"].as_str().expect("ingest body"))
        .expect("ingest summary");
    assert_eq!(summary["failed_count"], 0, "fixture ingest: {summary}");
    let seed = summary["records"][0]["internal_id"]
        .as_u64()
        .expect("seed id");
    let denied = summary["records"][1]["internal_id"]
        .as_u64()
        .expect("denied id");

    // A second real tenant identity on the same backend. Its canary must
    // never appear in the SYSTEM-tenant response.
    let default_power = dispatcher(backend, TenantId::DEFAULT, SessionScope::Power);
    let default_result = successful_tool_result(
        &default_power,
        "graph.ingest",
        json!({
            "tenant_id": TenantId::DEFAULT.raw(),
            "nodes": [{
                "external_id": "fix-1488-default-canary",
                "label": "Document",
                "properties": {"body": "DEFAULT_TENANT_CANARY_1488"}
            }],
            "acl_grants": [{
                "external_id": "fix-1488-default-canary",
                "read_principals": ["alice"]
            }],
            "format": "json"
        }),
    );
    let default_summary: Value = serde_json::from_str(
        default_result["body"]
            .as_str()
            .expect("default ingest body"),
    )
    .expect("default ingest summary");
    assert_eq!(default_summary["failed_count"], 0);

    (seed, denied)
}

#[test]
fn gate_fix_1488_mcp_explore_filters_denied_neighbor_content_nondefault_tenant() {
    let backend = fresh_backend();
    let (seed, denied) = ingest_acl_fixture(&backend);
    let read = dispatcher(&backend, TenantId::SYSTEM, SessionScope::Read);

    let result = successful_tool_result(
        &read,
        "graph.explore",
        json!({
            "tenant_id": TenantId::SYSTEM.raw(),
            "seed": seed,
            "max_depth": 1,
            "format": "json",
            "principal": "alice"
        }),
    );
    let body = result["body"].as_str().expect("explore body");
    let neighborhood: Neighborhood = serde_json::from_str(body).expect("neighborhood");
    assert_eq!(
        neighborhood
            .nodes
            .iter()
            .map(|node| node.id)
            .collect::<Vec<_>>(),
        vec![seed],
        "only the authorized seed may be returned: {body}"
    );
    assert!(
        neighborhood.edges.is_empty(),
        "denied incident edge omitted"
    );
    assert!(
        neighborhood.nodes.iter().all(|node| node.id != denied),
        "denied node id must not be returned"
    );
    assert!(body.contains("SEED_VISIBLE_1488"), "positive control");
    assert!(!body.contains("DENIED_SECRET_1488"), "denied content leak");
    assert!(
        !body.contains("DEFAULT_TENANT_CANARY_1488"),
        "cross-tenant canary leak"
    );

    // Class-complete same-class hardening: graph.inspect must not expose
    // the denied adjacent node through its neighbor metadata either.
    let inspect_result = successful_tool_result(
        &read,
        "graph.inspect",
        json!({
            "tenant_id": TenantId::SYSTEM.raw(),
            "node_id": seed,
            "format": "json",
            "principal": "alice"
        }),
    );
    let inspection: NodeInspection =
        serde_json::from_str(inspect_result["body"].as_str().expect("inspect body"))
            .expect("inspection");
    assert!(inspection.neighbors.is_empty(), "denied neighbor omitted");
}

#[test]
fn gate_fix_1488_mcp_explore_missing_principal_fails_closed_minus_32008_nondefault_tenant() {
    let backend = fresh_backend();
    let (seed, _) = ingest_acl_fixture(&backend);
    let read = dispatcher(&backend, TenantId::SYSTEM, SessionScope::Read);
    let response = tools_call(
        &read,
        "graph.explore",
        json!({
            "tenant_id": TenantId::SYSTEM.raw(),
            "seed": seed,
            "max_depth": 1,
            "format": "json"
        }),
    );
    assert!(response["error"].is_null(), "standard tools/call envelope");
    assert_eq!(response["result"]["isError"], true, "{response}");
    let error: Value = serde_json::from_str(
        response["result"]["content"][0]["text"]
            .as_str()
            .expect("tool error text"),
    )
    .expect("inner tool error JSON");
    assert_eq!(error["code"], -32008, "must fail closed: {error}");
    assert_eq!(error["data"]["required_scope"], "arcgraph.power");
}
