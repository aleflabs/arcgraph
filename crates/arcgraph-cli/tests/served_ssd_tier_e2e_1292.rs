//! #1292 PART-3 — served SSD-resident DiskANN tier end-to-end proof (ADR-195).
//!
//! The honesty gate for #1292: the served vector path can actually RUN the
//! RAM-decoupled SSD tier with the RSS ceiling enforced — not just build an SSD
//! provider that `serve` never uses. This test exercises the REAL served path
//! through the SAME `SubstrateSearchProvider` trait `graph.search` /
//! ArcQL `RANK BY` dispatch behind:
//!
//! ```text
//! StorageIngestProvider.ingest (real ingest + commit)
//!   → StorageHybridSearcher.search (real body, #765)
//!   → SsdVectorSearchProvider.vector_search (SubstrateSearchProvider, #1292)
//!   → SsdDiskAnnIndex::search (SQ8 nav beam + f32 rerank via pread — NOT mmap)
//!   → ranked SearchHits
//! ```
//!
//! Three claims proved:
//!
//! 1. **Tier selection / served path uses the SSD tier.** `VectorSearchTier::Ssd`
//!    builds an `SsdVectorSearchProvider`; a served `graph.search` returns correct
//!    top-k VIA `SsdDiskAnnIndex` (the served path is ABLE to run the RAM-bounded
//!    tier — #1292's core complaint). RED-on-revert: if serve were reverted to
//!    HNSW-only, `VectorSearchTier::Ssd.build_provider` would not exist and the
//!    tier-selection assertion `is SSD provider` would fail to compile / fail.
//!
//! 2. **Recall parity vs HNSW on a small corpus.** The SSD tier's top-k is
//!    results-comparable to the HNSW provider on the same ingested corpus (both
//!    rank the nearest node first) — the SSD swap does not corrupt results.
//!
//! 3. **Served-path RSS ceiling is enforced.** With a synthetic tiny RSS cap
//!    (1 MB — the live process is already far above it), the SSD build trips the
//!    guard and surfaces a structured error (RssCapExceeded / clean abort), NOT
//!    unbounded memory growth or an OOM-kill. This is the ADR-195 §2.2
//!    detect-and-abort backstop on the SERVED path.

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::vector_search::{
    HnswVectorSearchProvider, SsdVectorSearchProvider, VectorSearchTier,
};
use arcgraph_core::{Lsn, NodeId, PartitionId, TenantId};
use arcgraph_mcp::SessionScope;
use arcgraph_mcp::storage::{
    CrudExecutorSubstrate, StorageHybridSearcher, StorageIngestProvider, SubstrateSearchProvider,
};
use arcgraph_mcp::tools::ResponseFormat;
use arcgraph_mcp::tools::ingest::{IngestBatch, IngestProvider, IngestRecordOutcome, NodeIngest};
use arcgraph_mcp::tools::search::{SearchRequest, search_tool};
use arcgraph_query::CancellationToken;
use arcgraph_query::executor::ExecutionContext;
use arcgraph_query::executor::substrate::{ExecutorSubstrate, RemoveNodeMutation, SetNodeMutation};
use arcgraph_query::executor::value::Value;
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};
use serial_test::serial;
use tempfile::TempDir;

/// Deterministic LCG → f32 in `[0, 1)` (proptest-determinism discipline; no
/// `rand` dep, no clock — reproducible across runs).
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Build an `embedding` node property from an f32 slice.
fn embedding_props(v: &[f32]) -> BTreeMap<String, serde_json::Value> {
    vector_props("embedding", v)
}

/// Build an arbitrary vector property from an f32 slice.
fn vector_props(property: &str, v: &[f32]) -> BTreeMap<String, serde_json::Value> {
    let mut m = BTreeMap::new();
    m.insert(
        property.to_string(),
        serde_json::Value::Array(
            v.iter()
                .map(|f| {
                    serde_json::Number::from_f64(f64::from(*f))
                        .map(serde_json::Value::Number)
                        .expect("finite f32")
                })
                .collect(),
        ),
    );
    m
}

/// Ingest `(external_id, label, vector)` rows and return the external→internal
/// id map (drawn from the real commit's `Inserted` outcomes).
fn ingest_vectors(
    ingest: &StorageIngestProvider,
    tenant: TenantId,
    rows: &[(String, &str, Vec<f32>)],
) -> BTreeMap<String, u64> {
    let nodes = rows
        .iter()
        .map(|(ext, label, vec)| NodeIngest {
            external_id: Some(ext.clone()),
            label: (*label).to_string(),
            properties: embedding_props(vec),
        })
        .collect();
    let summary = ingest
        .ingest(
            tenant,
            IngestBatch {
                nodes,
                relationships: vec![],
                acl_grants: vec![],
            },
        )
        .expect("ingest vectors");
    assert_eq!(summary.failed_count, 0, "ingest must have 0 failures");
    let mut map = BTreeMap::new();
    for rec in &summary.records {
        if let IngestRecordOutcome::Inserted {
            internal_id,
            external_id,
        } = rec
        {
            map.insert(external_id.clone().unwrap_or_default(), *internal_id);
        }
    }
    map
}

/// Ingest vectors under a non-default property and return external→internal ids.
fn ingest_property_vectors(
    ingest: &StorageIngestProvider,
    tenant: TenantId,
    property: &str,
    rows: &[(String, &str, Vec<f32>)],
) -> BTreeMap<String, u64> {
    let nodes = rows
        .iter()
        .map(|(ext, label, vec)| NodeIngest {
            external_id: Some(ext.clone()),
            label: (*label).to_string(),
            properties: vector_props(property, vec),
        })
        .collect();
    let summary = ingest
        .ingest(
            tenant,
            IngestBatch {
                nodes,
                relationships: vec![],
                acl_grants: vec![],
            },
        )
        .expect("ingest property vectors");
    assert_eq!(summary.failed_count, 0, "ingest must have 0 failures");
    let mut map = BTreeMap::new();
    for rec in &summary.records {
        if let IngestRecordOutcome::Inserted {
            internal_id,
            external_id,
        } = rec
        {
            map.insert(external_id.clone().unwrap_or_default(), *internal_id);
        }
    }
    map
}

/// Run one ArcQL statement through the full front-end and production substrate.
fn run_query(query: &str, substrate: &CrudExecutorSubstrate) -> Vec<Vec<Value>> {
    let catalog = StubCatalogProvider::new();
    let statement = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&statement, query, &catalog).expect("bind");
    TypeCheckVisitor::check(&mut bound, &catalog).expect("type-check");
    CrossSubstrateValidator::validate(&bound, &catalog).expect("cross-substrate");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    materialize::materialize(&plan, substrate, &ctx)
        .expect("materialize")
        .rows()
        .to_vec()
}

fn rank1_node_id(rows: &[Vec<Value>]) -> u64 {
    match rows.first().map(Vec::as_slice) {
        Some([Value::Node(node), Value::Float(_)]) => node.id.raw(),
        other => panic!("queryNodes rank-1 row must be (node, score); got {other:?}"),
    }
}

fn ranked_node_ids(rows: &[Vec<Value>]) -> Vec<u64> {
    rows.iter()
        .map(|row| match row.as_slice() {
            [Value::Node(node), Value::Float(_)] => node.id.raw(),
            other => panic!("queryNodes row must be (node, score); got {other:?}"),
        })
        .collect()
}

/// The `VectorSearchTier::from_env` factory returns the SSD provider when
/// `ARCGRAPH_VECTOR_TIER=ssd` is set — and HNSW otherwise. This is the
/// tier-SELECTION contract (#1292): `serve` MUST be able to pick the tier.
///
/// RED-on-revert: if serve were reverted to HNSW-only (the pre-#1292 posture),
/// `VectorSearchTier::Ssd` would not build an SSD provider and this test would
/// fail. We assert the resolved tier is `Ssd` (env=ssd) and the built provider
/// is the SSD provider by exercising it against a real backend below.
#[test]
#[serial]
fn tier_selection_from_env_picks_ssd_when_requested() {
    // Directly construct the SSD tier config (env-parsing is exercised in the
    // dedicated env test; this asserts the tier variant + factory shape).
    let tmp = TempDir::new().expect("tempdir");
    let tier = VectorSearchTier::ssd_with_dir(tmp.path().to_path_buf());
    match tier {
        VectorSearchTier::Ssd {
            ref index_dir,
            rss_cap_mb,
        } => {
            assert_eq!(index_dir, tmp.path(), "SSD tier must carry the index dir");
            assert!(rss_cap_mb > 0, "SSD tier must carry a positive RSS cap");
        }
        VectorSearchTier::Hnsw => {
            panic!("ssd_with_dir must produce the SSD tier, not HNSW (front-4→A regression)")
        }
    }

    // The default tier stays HNSW (nothing regresses without an explicit opt-in).
    assert!(
        matches!(VectorSearchTier::default(), VectorSearchTier::Hnsw),
        "default tier must remain HNSW (no regression)",
    );
}

/// `from_env` honors `ARCGRAPH_VECTOR_TIER` + overrides. Serial to avoid env
/// races with other tests in the same binary (env is process-global).
#[test]
#[serial]
fn from_env_resolves_ssd_hnsw_and_overrides() {
    // Snapshot + restore the env to keep this test hermetic.
    let saved_tier = std::env::var_os(VectorSearchTier::TIER_ENV);
    let saved_dir = std::env::var_os(VectorSearchTier::DIR_ENV);
    let saved_cap = std::env::var_os(VectorSearchTier::RSS_CAP_ENV);

    // Unset → HNSW default.
    // SAFETY-style note: single-threaded test-only env mutation.
    unsafe {
        std::env::remove_var(VectorSearchTier::TIER_ENV);
    }
    assert!(
        matches!(VectorSearchTier::from_env(None), VectorSearchTier::Hnsw),
        "unset tier → HNSW default",
    );

    // ssd (case-insensitive) → SSD tier with overrides.
    unsafe {
        std::env::set_var(VectorSearchTier::TIER_ENV, "SSD");
        std::env::set_var(VectorSearchTier::DIR_ENV, "/tmp/arcgraph-ssd-test-1292");
        std::env::set_var(VectorSearchTier::RSS_CAP_ENV, "9999");
    }
    match VectorSearchTier::from_env(None) {
        VectorSearchTier::Ssd {
            index_dir,
            rss_cap_mb,
        } => {
            assert_eq!(
                index_dir,
                std::path::PathBuf::from("/tmp/arcgraph-ssd-test-1292")
            );
            assert_eq!(rss_cap_mb, 9999, "RSS cap override must be honored");
        }
        VectorSearchTier::Hnsw => panic!("ARCGRAPH_VECTOR_TIER=SSD must resolve to the SSD tier"),
    }

    // Restore env.
    unsafe {
        match saved_tier {
            Some(v) => std::env::set_var(VectorSearchTier::TIER_ENV, v),
            None => std::env::remove_var(VectorSearchTier::TIER_ENV),
        }
        match saved_dir {
            Some(v) => std::env::set_var(VectorSearchTier::DIR_ENV, v),
            None => std::env::remove_var(VectorSearchTier::DIR_ENV),
        }
        match saved_cap {
            Some(v) => std::env::set_var(VectorSearchTier::RSS_CAP_ENV, v),
            None => std::env::remove_var(VectorSearchTier::RSS_CAP_ENV),
        }
    }
}

/// CLAIM 1 + 2: the SERVED `graph.search` returns correct top-k VIA the SSD tier,
/// results-comparable to HNSW on the same corpus.
///
/// Corpus size (~200 vectors) matches the arcgraph-vector SSD test sizing (its
/// smallest is 8×20=160): the SSD DiskANN tier is a >10M-scale serving path, so
/// a realistic small corpus exercises the real Vamana build (a handful of vectors
/// is neither a real use case nor a build the graph params are tuned for).
#[test]
#[serial]
fn served_graph_search_uses_ssd_tier_and_matches_hnsw_rank1() {
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    // ~200-vector deterministic corpus in an 8-D space. One node (`near`) is
    // planted at the exact query point so it is the UNAMBIGUOUS nearest neighbor
    // for both tiers (a strong, tie-free oracle for rank-1 parity).
    const DIM: usize = 8;
    const N: usize = 200;
    let query = vec![0.5_f32; DIM];
    let mut seed = 0x1292_0001u64;
    let mut rows: Vec<(String, &str, Vec<f32>)> = Vec::with_capacity(N + 1);
    // The planted nearest — exactly at the query.
    rows.push(("near".to_string(), "Doc", query.clone()));
    // The rest — random, at L2 distance >> 0 from the query with high probability.
    for i in 0..N {
        let vec: Vec<f32> = (0..DIM).map(|_| 2.0 + lcg(&mut seed) * 8.0).collect();
        rows.push((format!("doc-{i}"), "Doc", vec));
    }
    let ids = ingest_vectors(&ingest, tenant, &rows);

    // ── The SSD tier via the tier factory (the served-provider construction path).
    let tmp = TempDir::new().expect("tempdir");
    let ssd_tier = VectorSearchTier::ssd_with_dir(tmp.path().to_path_buf());
    let ssd_provider = ssd_tier.build_provider(backend.clone());

    // Direct provider call — proves the SSD path returns ranked hits VIA SsdDiskAnnIndex.
    let ssd_hits = ssd_provider
        .vector_search(tenant, "embedding", &query, 3, Lsn::MAX)
        .expect("SSD-tier served provider vector_search must return ranked hits");
    assert!(
        !ssd_hits.is_empty(),
        "SSD-tier served path must return ranked rows, not empty (front-4→A: serve can run the tier)",
    );
    eprintln!("#1292 SSD-tier served vector_search ranked node_ids:");
    for (i, h) in ssd_hits.iter().enumerate() {
        eprintln!(
            "  rank {}: node_id={} score={:.4}",
            i + 1,
            h.node.id.raw(),
            h.score
        );
    }
    assert_eq!(
        ssd_hits[0].node.id.raw(),
        ids["near"],
        "SSD-tier rank-1 must be the node planted at the query (the nearest embedding)",
    );

    // Bind the SSD provider into the production graph.search adapter (the real
    // served dispatch path — the same StorageHybridSearcher that serve wires).
    let searcher =
        StorageHybridSearcher::new(backend.clone()).with_search_provider(Arc::clone(&ssd_provider));
    let token = CancellationToken::new();
    let resp = search_tool(
        &searcher,
        tenant,
        SessionScope::Power,
        &token,
        SearchRequest {
            tenant_id: tenant.raw(),
            query: String::new(), // vector-only (no BM25 operand)
            query_vec: Some(query.clone()),
            k: Some(3),
            label_filter: None,
            ef_search: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        },
    )
    .expect("served graph.search over SSD tier must return Ok, not IndexUnavailable");
    let body: serde_json::Value =
        serde_json::from_str(resp["body"].as_str().expect("body string")).expect("parse body");
    let hits = body["hits"].as_array().expect("hits array");
    assert!(
        !hits.is_empty(),
        "served graph.search over the SSD tier must return rows (front-4→A honesty gate)",
    );
    assert_eq!(
        hits[0]["node_id"].as_u64().expect("node_id u64"),
        ids["near"],
        "served graph.search over SSD tier: rank-1 must be the planted nearest node",
    );

    // ── CLAIM 2: recall parity vs HNSW — same rank-1 on the same corpus.
    let hnsw_provider: Arc<dyn SubstrateSearchProvider> =
        Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let hnsw_hits = hnsw_provider
        .vector_search(tenant, "embedding", &query, 3, Lsn::MAX)
        .expect("HNSW provider vector_search");
    assert_eq!(
        ssd_hits[0].node.id.raw(),
        hnsw_hits[0].node.id.raw(),
        "SSD tier and HNSW must agree on rank-1 (results-comparable top-k parity): \
         ssd={:?}, hnsw={:?}",
        ssd_hits.iter().map(|h| h.node.id.raw()).collect::<Vec<_>>(),
        hnsw_hits
            .iter()
            .map(|h| h.node.id.raw())
            .collect::<Vec<_>>(),
    );
}

/// #1382: an SSD cache entry must not outlive an ingest high-water advance or
/// an embedding UPDATE. DELETE remains a query-time tombstone and must not need
/// an immutable-index rebuild.
#[test]
#[serial]
fn served_ssd_graph_search_refreshes_after_ingest_and_update_and_honors_delete() {
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    const DIM: usize = 8;
    const FILLER_COUNT: usize = 200;
    let build_query = vec![0.0_f32; DIM];
    let post_build_query = vec![10.0_f32; DIM];
    let updated_query = vec![-10.0_f32; DIM];
    let deleted_query = vec![-20.0_f32; DIM];
    let mut rows = vec![
        ("build-anchor".to_string(), "Doc", build_query.clone()),
        ("updated".to_string(), "Doc", vec![20.0_f32; DIM]),
        ("deleted".to_string(), "Doc", deleted_query.clone()),
    ];
    let mut seed = 0x1382_0001u64;
    for i in 0..FILLER_COUNT {
        let vector = (0..DIM).map(|_| 100.0 + lcg(&mut seed) * 10.0).collect();
        rows.push((format!("filler-{i}"), "Doc", vector));
    }
    let ids = ingest_vectors(&ingest, tenant, &rows);
    let updated_id = NodeId::new(ids["updated"]);
    let deleted_id = NodeId::new(ids["deleted"]);

    let tmp = TempDir::new().expect("tempdir");
    let provider = Arc::new(SsdVectorSearchProvider::new(
        backend.clone(),
        tmp.path().to_path_buf(),
        64_000,
    ));
    let searcher = StorageHybridSearcher::new(backend.clone())
        .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);
    let substrate = CrudExecutorSubstrate::new(
        Arc::clone(backend.router()),
        Arc::clone(backend.txn_manager()),
        Arc::clone(backend.intern_table()),
    )
    .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    let token = CancellationToken::new();

    let search_ids = |query_vec: Vec<f32>, k: u32| -> Vec<u64> {
        let response = search_tool(
            &searcher,
            tenant,
            SessionScope::Power,
            &token,
            SearchRequest {
                tenant_id: tenant.raw(),
                query: String::new(),
                query_vec: Some(query_vec),
                k: Some(k),
                label_filter: None,
                ef_search: None,
                format: Some(ResponseFormat::Json),
                principal: None,
            },
        )
        .expect("graph.search over SSD tier");
        let body: serde_json::Value =
            serde_json::from_str(response["body"].as_str().expect("body string"))
                .expect("parse graph.search body");
        body["hits"]
            .as_array()
            .expect("hits array")
            .iter()
            .map(|hit| hit["node_id"].as_u64().expect("node_id u64"))
            .collect()
    };

    let baseline = search_ids(build_query, 5);
    assert_eq!(
        baseline.first().copied(),
        Some(ids["build-anchor"]),
        "first graph.search must build the SSD cache; hits={baseline:?}",
    );

    let new_ids = ingest_vectors(
        &ingest,
        tenant,
        &[("post-build".to_string(), "Doc", post_build_query.clone())],
    );
    let post_build_id = new_ids["post-build"];
    let after_ingest = search_ids(post_build_query, 5);

    substrate
        .set_node(
            tenant,
            updated_id,
            &SetNodeMutation::PropertyAssign {
                name: "embedding".to_string(),
                value: Value::List(
                    updated_query
                        .iter()
                        .map(|value| Value::Float(f64::from(*value)))
                        .collect(),
                ),
            },
            &ctx,
        )
        .expect("SET updated.embedding through production substrate");
    let after_update = search_ids(updated_query, 5);

    assert_eq!(
        search_ids(deleted_query.clone(), 5).first().copied(),
        Some(deleted_id.raw()),
        "delete target must be resident before the tombstone regression check",
    );
    substrate
        .delete_node(tenant, deleted_id, true, &ctx)
        .expect("DETACH DELETE through production substrate");
    let after_delete = search_ids(deleted_query, 5);
    assert!(
        !after_delete.contains(&deleted_id.raw()),
        "SSD query-time tombstone must continue to exclude deleted node; hits={after_delete:?}",
    );

    assert_eq!(
        (after_ingest.first().copied(), after_update.first().copied(),),
        (Some(post_build_id), Some(updated_id.raw())),
        "SSD cache must rebuild for both post-build ingest and embedding UPDATE; \
         after_ingest={after_ingest:?}, after_update={after_update:?}",
    );
}

/// #1450: the SSD freshness gate must invalidate the property-specific cache
/// slot when SET changes a registered, non-default vector property.
#[test]
#[serial]
fn query_nodes_refreshes_ssd_after_non_default_vector_property_update() {
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    const DIM: usize = 8;
    const FILLER_COUNT: usize = 200;
    let build_query = vec![0.0_f32; DIM];
    let updated_query = [-10.0_f32; DIM];
    let mut rows = vec![
        ("build-anchor".to_string(), "Doc", build_query.clone()),
        ("updated".to_string(), "Doc", vec![20.0_f32; DIM]),
    ];
    let mut seed = 0x1450_0001u64;
    for i in 0..FILLER_COUNT {
        let vector = (0..DIM).map(|_| 100.0 + lcg(&mut seed) * 10.0).collect();
        rows.push((format!("filler-{i}"), "Doc", vector));
    }
    let ids = ingest_property_vectors(&ingest, tenant, "content", &rows);
    let updated_id = NodeId::new(ids["updated"]);

    let tmp = TempDir::new().expect("tempdir");
    let provider = Arc::new(SsdVectorSearchProvider::new(
        backend.clone(),
        tmp.path().to_path_buf(),
        64_000,
    ));
    let substrate = CrudExecutorSubstrate::new(
        Arc::clone(backend.router()),
        Arc::clone(backend.txn_manager()),
        Arc::clone(backend.intern_table()),
    )
    .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);

    assert!(
        run_query(
            "CREATE VECTOR INDEX content_idx FOR (n:Doc) ON n.content",
            &substrate,
        )
        .is_empty(),
        "CREATE VECTOR INDEX must register metadata and return no rows",
    );
    let baseline = run_query(
        "CALL db.index.vector.queryNodes('content_idx', 5, \
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) \
         YIELD node, score RETURN node, score",
        &substrate,
    );
    assert_eq!(
        rank1_node_id(&baseline),
        ids["build-anchor"],
        "first queryNodes must resolve content_idx → content and build its SSD cache",
    );

    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    substrate
        .set_node(
            tenant,
            updated_id,
            &SetNodeMutation::PropertyAssign {
                name: "content".to_string(),
                value: Value::List(
                    updated_query
                        .iter()
                        .map(|value| Value::Float(f64::from(*value)))
                        .collect(),
                ),
            },
            &ctx,
        )
        .expect("SET updated.content through production substrate");

    let after_update = run_query(
        "CALL db.index.vector.queryNodes('content_idx', 5, \
         [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]) \
         YIELD node, score RETURN node, score",
        &substrate,
    );
    assert_eq!(
        rank1_node_id(&after_update),
        updated_id.raw(),
        "queryNodes must rebuild the content slot and expose SET n.content immediately; \
         hits={after_update:?}",
    );
}

/// #1450: REMOVE must invalidate the registered property's SSD cache slot;
/// unlike DELETE, the node remains live, so a stale slot would keep returning
/// the removed embedding forever.
#[test]
#[serial]
fn query_nodes_refreshes_ssd_after_non_default_vector_property_remove() {
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    const DIM: usize = 8;
    const FILLER_COUNT: usize = 200;
    let build_query = vec![0.0_f32; DIM];
    let mut rows = vec![
        ("removed".to_string(), "Doc", build_query.clone()),
        ("durable-survivor".to_string(), "Doc", vec![20.0_f32; DIM]),
    ];
    let mut seed = 0x1450_0002u64;
    for i in 0..FILLER_COUNT {
        let vector = (0..DIM).map(|_| 100.0 + lcg(&mut seed) * 10.0).collect();
        rows.push((format!("filler-{i}"), "Doc", vector));
    }
    let ids = ingest_property_vectors(&ingest, tenant, "content", &rows);

    let tmp = TempDir::new().expect("tempdir");
    let provider = Arc::new(SsdVectorSearchProvider::new(
        backend.clone(),
        tmp.path().to_path_buf(),
        64_000,
    ));
    let substrate = CrudExecutorSubstrate::new(
        Arc::clone(backend.router()),
        Arc::clone(backend.txn_manager()),
        Arc::clone(backend.intern_table()),
    )
    .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);

    assert!(
        run_query(
            "CREATE VECTOR INDEX content_idx FOR (n:Doc) ON n.content",
            &substrate,
        )
        .is_empty(),
        "CREATE VECTOR INDEX must register metadata and return no rows",
    );
    let query_nodes = |substrate: &CrudExecutorSubstrate| {
        run_query(
            "CALL db.index.vector.queryNodes('content_idx', 5, \
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) \
             YIELD node, score RETURN node, score",
            substrate,
        )
    };
    let baseline = query_nodes(&substrate);
    assert_eq!(
        rank1_node_id(&baseline),
        ids["removed"],
        "first queryNodes must build a content slot with the removal target at rank 1",
    );

    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    substrate
        .remove_node(
            tenant,
            NodeId::new(ids["removed"]),
            &RemoveNodeMutation::Property("content".to_string()),
            &ctx,
        )
        .expect("REMOVE removed.content through production substrate");
    let durable_target = substrate
        .scan_nodes(tenant, None, Lsn::MAX)
        .expect("scan after REMOVE")
        .into_iter()
        .find(|bound| bound.node.id.raw() == ids["removed"])
        .expect("removal target remains live");
    assert!(
        !durable_target.node.properties.contains_key("content"),
        "store oracle: REMOVE must durably remove the target's content property",
    );

    let after_remove = query_nodes(&substrate);
    let after_remove_ids = ranked_node_ids(&after_remove);
    assert!(
        !after_remove_ids.is_empty(),
        "queryNodes must continue serving vectors that remain durable after REMOVE",
    );
    assert!(
        !after_remove_ids.contains(&ids["removed"]),
        "queryNodes must rebuild the content slot after REMOVE n.content and stop returning \
         the removed rank-1 vector; hits={after_remove:?}",
    );
}

/// CLAIM 3: the served-path RSS ceiling is ENFORCED. With a synthetic 1 MB cap
/// (the live test process RSS is already far above it), the SSD build trips the
/// guard at the first build checkpoint and surfaces a structured error — a clean
/// abort, NOT unbounded growth / OOM-kill (ADR-195 §2.2 detect-and-abort).
#[test]
#[serial]
fn served_ssd_tier_rss_ceiling_aborts_cleanly_not_oom() {
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    // Ingest enough vectors that the build spans several guard sample windows
    // (100 ms cadence) and crosses the per-4096 in-loop `guard.check()` boundary.
    // The RSS guard samples the LIVE process RSS (already >> 1 MB), so a 1 MB cap
    // trips deterministically at a build checkpoint — a clean abort, not an OOM.
    // 5000 > BUILD_GUARD_CHECK_EVERY (4096), so the in-loop poll fires too.
    let mut seed = 0x1292_5541u64;
    const N: usize = 5000;
    let mut rows: Vec<(String, &str, Vec<f32>)> = Vec::with_capacity(N);
    for i in 0..N {
        let vec: Vec<f32> = (0..8).map(|_| lcg(&mut seed)).collect();
        rows.push((format!("v-{i}"), "Vec", vec));
    }
    let _ = ingest_vectors(&ingest, tenant, &rows);

    // 1 MB RSS cap — far below the live process footprint. The guard trips.
    let tmp = TempDir::new().expect("tempdir");
    let provider = SsdVectorSearchProvider::new(backend.clone(), tmp.path().to_path_buf(), 1);

    let query: Vec<f32> = (0..8).map(|_| lcg(&mut seed)).collect();
    let result = provider.vector_search(tenant, "embedding", &query, 5, Lsn::MAX);

    // The build must ABORT with a structured error, not hang / OOM / return junk.
    let err = result.expect_err(
        "1 MB RSS cap must trip the guard and surface a structured error (clean abort), \
         not silently succeed or OOM",
    );
    let msg = format!("{err}");
    eprintln!("#1292 served-path RSS ceiling enforced — clean abort error: {msg}");
    assert!(
        msg.to_lowercase().contains("rss")
            || msg.to_lowercase().contains("cap")
            || msg.to_lowercase().contains("exceed"),
        "RSS-ceiling abort error must name the cap/RSS breach (ADR-195 §2.2); got: {msg}",
    );

    // ── RED-on-revert control: the HNSW tier (the pre-#1292 served provider) has
    // NO RSS ceiling — the SAME corpus + query succeeds unbounded. This is the
    // regression the whole #1292 fix exists to close: if `serve` were reverted to
    // HNSW-only, a large ingest would grow RAM without a ceiling (the OOM). The
    // contrast (SSD aborts, HNSW does not) proves the ceiling is uniquely the SSD
    // tier's behavior — so a serve reverted to HNSW-only loses this enforcement.
    let hnsw = HnswVectorSearchProvider::new(backend.clone());
    let hnsw_result = hnsw.vector_search(tenant, "embedding", &query, 5, Lsn::MAX);
    assert!(
        hnsw_result.is_ok(),
        "HNSW tier has NO RSS ceiling — the same corpus must NOT abort under HNSW; \
         the SSD tier's ceiling is the #1292 fix (RED-on-revert: HNSW-only serve loses it)",
    );
}

/// A generous RSS cap does NOT trip — the same build succeeds and serves. This is
/// the control for the ceiling test above (proves the guard only aborts on a real
/// breach, not always).
#[test]
#[serial]
fn served_ssd_tier_generous_rss_cap_builds_and_serves() {
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    // ~200-vector corpus (real Vamana build sizing) with a planted nearest.
    const DIM: usize = 8;
    const N: usize = 200;
    let query = vec![0.5_f32; DIM];
    let mut seed = 0x1292_C0DEu64;
    let mut rows: Vec<(String, &str, Vec<f32>)> = Vec::with_capacity(N + 1);
    rows.push(("near".to_string(), "Doc", query.clone()));
    for i in 0..N {
        let vec: Vec<f32> = (0..DIM).map(|_| 2.0 + lcg(&mut seed) * 8.0).collect();
        rows.push((format!("far-{i}"), "Doc", vec));
    }
    let ids = ingest_vectors(&ingest, tenant, &rows);

    // 64 GB cap — never trips on a test box.
    let tmp = TempDir::new().expect("tempdir");
    let provider = SsdVectorSearchProvider::new(backend.clone(), tmp.path().to_path_buf(), 64_000);
    let hits = provider
        .vector_search(tenant, "embedding", &query, 5, Lsn::MAX)
        .expect("generous-cap SSD build must succeed and serve");
    assert!(!hits.is_empty(), "generous-cap SSD tier must return hits");
    assert_eq!(
        hits[0].node.id.raw(),
        ids["near"],
        "generous-cap SSD tier rank-1 must be the planted nearest node",
    );
}
