//! #1023 — production bootstrap must attach BM25 for served text search.
//!
//! This is intentionally distinct from the lower-level served BM25 tests:
//! it uses the same `bootstrap_storage_backend(BootstrapMode::InMemory)`
//! entry point as the shipped binaries, then drives served ingest and
//! `graph.search`. No test-local `Bm25Service` or router BM25 wiring is
//! installed here.

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::vector_search::HnswVectorSearchProvider;
use arcgraph_core::TenantId;
use arcgraph_mcp::SessionScope;
use arcgraph_mcp::storage::{
    StorageHybridSearcher, StorageIngestProvider, SubstrateSearchProvider,
};
use arcgraph_mcp::tools::ResponseFormat;
use arcgraph_mcp::tools::ingest::{IngestBatch, IngestProvider, IngestRecordOutcome, NodeIngest};
use arcgraph_mcp::tools::search::{SearchRequest, search_tool};
use arcgraph_query::CancellationToken;

fn text_props(text: &str) -> BTreeMap<String, serde_json::Value> {
    let mut props = BTreeMap::new();
    props.insert(
        "text".to_string(),
        serde_json::Value::String(text.to_string()),
    );
    props
}

#[test]
fn shipped_bootstrap_serves_text_search_after_ingest_1023() {
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    let summary = ingest
        .ingest(
            tenant,
            IngestBatch {
                nodes: vec![
                    NodeIngest {
                        external_id: Some("alpha-doc".to_string()),
                        label: "Doc".to_string(),
                        properties: text_props("alpha arcgraph bootstrap needle"),
                    },
                    NodeIngest {
                        external_id: Some("beta-doc".to_string()),
                        label: "Doc".to_string(),
                        properties: text_props("beta unrelated haystack"),
                    },
                ],
                relationships: vec![],
                acl_grants: vec![],
            },
        )
        .expect("served ingest");
    assert_eq!(summary.failed_count, 0, "ingest must have no failures");

    let alpha_id = summary
        .records
        .iter()
        .find_map(|rec| match rec {
            IngestRecordOutcome::Inserted {
                internal_id,
                external_id: Some(external_id),
            } if external_id == "alpha-doc" => Some(*internal_id),
            _ => None,
        })
        .expect("alpha-doc inserted");

    let provider: Arc<dyn SubstrateSearchProvider> =
        Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let searcher = StorageHybridSearcher::new(backend.clone()).with_search_provider(provider);
    let token = CancellationToken::new();

    let resp = search_tool(
        &searcher,
        tenant,
        SessionScope::Power,
        &token,
        SearchRequest {
            tenant_id: tenant.raw(),
            query: "bootstrap needle".to_string(),
            query_vec: None,
            k: Some(1),
            label_filter: None,
            ef_search: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        },
    )
    .expect("graph.search text-only must not return -32004 IndexUnavailable(\"bm25\")");

    let body: serde_json::Value = serde_json::from_str(resp["body"].as_str().expect("body string"))
        .expect("parse graph.search body");
    let hits = body["hits"].as_array().expect("hits array");
    assert_eq!(
        hits.first().and_then(|h| h["node_id"].as_u64()),
        Some(alpha_id),
        "production bootstrap BM25 attach must make ingested text searchable; hits={hits:?}",
    );
}
