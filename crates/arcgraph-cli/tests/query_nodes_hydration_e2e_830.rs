//! #830 D4 residual — `db.index.vector.queryNodes` stored-property + label
//! hydration, end-to-end over the REAL served stack.
//!
//! ## The bug (Customer-Zero #830, 2026-06-08)
//!
//! ```text
//! CALL db.index.vector.queryNodes('vector', 2, [0.95,0.05,0,0]) YIELD node, score
//! RETURN node.id AS id, node.text AS text, node{.*} AS meta, score
//! -> [[None, None, {}, 0.995], [None, None, {}, 0.356]]
//! ```
//!
//! The `queryNodes`-returned node came back with the CORRECT id/score/order
//! but an EMPTY property bag (`node{.*}` → `{}`, `node.text` → `None`) and an
//! unresolved label name (`node.labels` → `['LabelId(1)']`). langchain's
//! `Neo4jVector` RAG therefore retrieved documents with no content/metadata.
//! The control `MATCH (a:Doc) RETURN a` returned the SAME node WITH its full
//! property bag + the real label name — so hydration worked on the MATCH path
//! but not on the `queryNodes` path.
//!
//! ## Root cause
//!
//! The served HNSW provider (`arcgraph_cli::vector_search`) builds each
//! `RankedHit` from only the resident `(node_id, label)` sidecar, never
//! reading the record store — so `RankedHit.node` is an empty
//! `NodeView::new(id, label)`. The fix (PR for #830 D4) re-hydrates each hit's
//! node by id inside `CrudExecutorSubstrate::vector_search`, through the SAME
//! idiom the MATCH path uses, so a `queryNodes` node is shaped IDENTICALLY to
//! a MATCH node.
//!
//! ## Why this lives in `arcgraph-cli` tests
//!
//! It is the only bounded context that can wire BOTH the production
//! `CrudExecutorSubstrate` (arcgraph-mcp) AND the production
//! `HnswVectorSearchProvider` (arcgraph-cli) over one populated, committed
//! store and drive the FULL arcgraph-query executor against them.

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::vector_search::HnswVectorSearchProvider;
use arcgraph_core::{Lsn, PartitionId, TenantId};
use arcgraph_mcp::storage::{
    CrudExecutorSubstrate, StorageBackend, StorageIngestProvider, SubstrateSearchProvider,
};
use arcgraph_mcp::tools::ingest::{IngestBatch, IngestProvider, IngestRecordOutcome, NodeIngest};
use arcgraph_query::executor::ExecutionContext;
use arcgraph_query::executor::substrate::ExecutorSubstrate;
use arcgraph_query::executor::value::Value;

/// One `(external_id, label, text, embedding)` document to ingest with a FULL
/// property bag (`id`, `text`, `embedding`) — the shape the #830 customer used.
struct Doc {
    external_id: &'static str,
    label: &'static str,
    text: &'static str,
    embedding: Vec<f32>,
}

/// Build the `{id, text, embedding}` property bag from a [`Doc`] — the
/// embedding stored as a JSON array (the served vector property).
fn doc_props(d: &Doc) -> BTreeMap<String, serde_json::Value> {
    let mut m = BTreeMap::new();
    m.insert(
        "id".to_string(),
        serde_json::Value::String(d.external_id.to_string()),
    );
    m.insert(
        "text".to_string(),
        serde_json::Value::String(d.text.to_string()),
    );
    m.insert(
        "embedding".to_string(),
        serde_json::Value::Array(
            d.embedding
                .iter()
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

/// Ingest the docs through the REAL `StorageIngestProvider` (commit lands them
/// in the record store + the HNSW-derived index) and return external→internal
/// id map drawn from the commit's `Inserted` outcomes.
fn ingest_docs(
    ingest: &StorageIngestProvider,
    tenant: TenantId,
    docs: &[Doc],
) -> BTreeMap<String, u64> {
    let nodes = docs
        .iter()
        .map(|d| NodeIngest {
            external_id: Some(d.external_id.to_string()),
            label: d.label.to_string(),
            properties: doc_props(d),
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
        .expect("ingest docs");
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

/// Build a production `CrudExecutorSubstrate` over a bootstrapped backend with
/// the served `HnswVectorSearchProvider` attached — the exact surface
/// `CALL db.index.vector.queryNodes(...)` drives.
fn substrate_with_served_provider(
    backend: &StorageBackend,
    provider: Arc<dyn SubstrateSearchProvider>,
) -> CrudExecutorSubstrate {
    CrudExecutorSubstrate::new(
        Arc::clone(backend.router()),
        Arc::clone(backend.txn_manager()),
        Arc::clone(backend.intern_table()),
    )
    .with_search_provider(provider)
}

/// The three deterministic docs every test below shares: a 4-D plane where the
/// query `[0.9, 0.1, 0, 0]` ranks doc-1 first, doc-2 second, doc-3 third.
fn corpus() -> Vec<Doc> {
    vec![
        Doc {
            external_id: "doc-1",
            label: "Doc",
            text: "hello world",
            embedding: vec![1.0, 0.0, 0.0, 0.0],
        },
        Doc {
            external_id: "doc-2",
            label: "Doc",
            text: "goodbye moon",
            embedding: vec![0.0, 1.0, 0.0, 0.0],
        },
        Doc {
            external_id: "doc-3",
            label: "Doc",
            text: "lorem ipsum",
            embedding: vec![0.0, 0.0, 1.0, 0.0],
        },
    ]
}

const QUERY: [f32; 4] = [0.9, 0.1, 0.0, 0.0];

#[test]
fn query_nodes_full_executor_hydrates_stored_props_and_label_830() {
    // ── The Customer-Zero #830 D4 oracle: drive
    // `CALL db.index.vector.queryNodes('docs', 2, [query]) YIELD node, score
    //  RETURN node, score` through the FULL arcgraph-query executor (parse →
    // bind → type-check → cross-substrate → lower → materialize) against the
    // REAL CrudExecutorSubstrate + REAL HnswVectorSearchProvider, and assert
    // each returned node carries its FULL stored property bag + the resolved
    // label NAME (not the empty `{}` / `LabelId(1)` the bug returned).
    use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
    use arcgraph_query::semantic::{
        BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
    };
    use arcgraph_query::{materialize, parse};

    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;

    let ingest = StorageIngestProvider::new(backend.clone());
    let docs = corpus();
    let ids = ingest_docs(&ingest, tenant, &docs);

    let provider: Arc<dyn SubstrateSearchProvider> =
        Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let sub = substrate_with_served_provider(&backend, provider);

    // Build the plan via the real front-end (advisory index name 'docs' →
    // the served "embedding" vector property at v1.0-α).
    let q = "CALL db.index.vector.queryNodes('docs', 2, [0.9, 0.1, 0.0, 0.0]) \
             YIELD node, score RETURN node, score";
    let cat = StubCatalogProvider::new();
    let stmt = parse(q).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, q, &cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");

    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    let result = materialize::materialize(&plan, &sub, &ctx).expect("materialize");
    let rows = result.rows().to_vec();

    // k=2 → exactly the two nearest docs, closest-first.
    assert_eq!(rows.len(), 2, "k=2 → two ranked rows");

    // ── Rank-1 node MUST be doc-1, fully hydrated.
    let (node0, score0) = match rows[0].as_slice() {
        [Value::Node(n), Value::Float(s)] => (n, *s),
        other => panic!("row[0] must be (node, score); got {other:?}"),
    };
    assert_eq!(
        node0.id.raw(),
        ids["doc-1"],
        "rank-1 must be doc-1 (the nearest embedding)",
    );
    // THE #830 D4 FIX: stored properties are present (not the empty `{}` the
    // bug returned). A node that comes back without `text` IS the bug.
    assert_eq!(
        node0.properties.get("text"),
        Some(&Value::String("hello world".to_string())),
        "node.text must hydrate to the stored value (the #830 D4 defect: it was None/{{}}); \
         got properties = {:?}",
        node0.properties,
    );
    assert_eq!(
        node0.properties.get("id"),
        Some(&Value::String("doc-1".to_string())),
        "node.id property must hydrate to the stored value",
    );
    assert!(
        node0.properties.contains_key("embedding"),
        "the full stored bag hydrates (id, text, embedding) — matching the MATCH-path \
         control `node{{.*}}`; got keys = {:?}",
        node0.properties.keys().collect::<Vec<_>>(),
    );
    // THE #871 half: the label NAME resolves (not the opaque `LabelId(1)`).
    assert_eq!(
        node0.label_name.as_deref(),
        Some("Doc"),
        "node.label_name must reverse-resolve to \"Doc\", not leak LabelId(1)",
    );

    // ── Rank-2 node MUST be doc-2, also fully hydrated, ranked BELOW doc-1.
    let (node1, score1) = match rows[1].as_slice() {
        [Value::Node(n), Value::Float(s)] => (n, *s),
        other => panic!("row[1] must be (node, score); got {other:?}"),
    };
    assert_eq!(
        node1.id.raw(),
        ids["doc-2"],
        "rank-2 must be doc-2 (the second-nearest)",
    );
    assert_eq!(
        node1.properties.get("text"),
        Some(&Value::String("goodbye moon".to_string())),
        "rank-2 node.text must hydrate too",
    );
    assert_eq!(node1.label_name.as_deref(), Some("Doc"));

    // ── Score/order preserved: strictly closest-first (doc-1 nearer than doc-2).
    assert!(
        score0 > score1,
        "scores must be descending (doc-1 nearer than doc-2): {score0} vs {score1}",
    );

    eprintln!(
        "#830 D4 queryNodes (full executor): \
         rank1 id={} text={:?} label={:?} score={score0:.4}; \
         rank2 id={} text={:?} score={score1:.4}",
        node0.id.raw(),
        node0.properties.get("text"),
        node0.label_name,
        node1.id.raw(),
        node1.properties.get("text"),
    );
}

#[test]
fn substrate_vector_search_hydration_preserves_id_score_order_830() {
    // ── Strong-oracle invariant pin (substrate boundary): the served provider
    // and CrudExecutorSubstrate materialize the same hydrated NodeViews without
    // changing id, score, rank order, or count. Proven hit-for-hit over the
    // SAME provider instance + same query.
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;

    let ingest = StorageIngestProvider::new(backend.clone());
    let docs = corpus();
    let ids = ingest_docs(&ingest, tenant, &docs);

    let provider = Arc::new(HnswVectorSearchProvider::new(backend.clone()));

    // BASELINE: the raw provider, straight from the served HNSW. #986 moved
    // the shared hydration helper into the provider too, so these NodeViews
    // must already carry stored props + label names.
    let baseline = SubstrateSearchProvider::vector_search(
        provider.as_ref(),
        tenant,
        "embedding",
        &QUERY,
        3,
        Lsn::MAX,
    )
    .expect("baseline provider search");
    assert_eq!(baseline.len(), 3, "3-doc corpus, k=3 → 3 hits");
    for (i, h) in baseline.iter().enumerate() {
        assert!(
            !h.node.properties.is_empty(),
            "served provider must hydrate stored property bags (the #986 drift fix); \
             got {:?}",
            h.node.properties,
        );
        assert_eq!(
            h.node.properties.get("text"),
            Some(&Value::String(
                docs[i_to_doc(h.node.id.raw(), &ids)].text.to_string()
            )),
            "baseline hit[{i}] node.text must hydrate to the stored value",
        );
        assert_eq!(
            h.node.label_name.as_deref(),
            Some("Doc"),
            "baseline hit[{i}] label_name must reverse-resolve to \"Doc\"",
        );
    }

    // HYDRATED: the substrate over the SAME provider Arc.
    let sub = substrate_with_served_provider(
        &backend,
        Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>,
    );
    let hydrated = ExecutorSubstrate::vector_search(&sub, tenant, "embedding", &QUERY, 3, Lsn::MAX)
        .expect("hydrated substrate search");

    // ── INVARIANT: same count, same id, same score, same order.
    assert_eq!(
        hydrated.len(),
        baseline.len(),
        "hydration must not change the hit count",
    );
    for (i, (b, h)) in baseline.iter().zip(&hydrated).enumerate() {
        assert_eq!(
            h.node.id, b.node.id,
            "hit[{i}] id + rank order must be byte-identical pre/post hydration",
        );
        assert_eq!(
            h.score.to_bits(),
            b.score.to_bits(),
            "hit[{i}] score must be byte-identical pre/post hydration",
        );
        assert_eq!(
            h.node.label, b.node.label,
            "hit[{i}] label id must be unchanged",
        );
        assert_eq!(
            h.node.label_name, b.node.label_name,
            "hit[{i}] label name must match the provider-hydrated row",
        );
        // ── …and hydration is still present at the substrate boundary.
        assert!(
            !h.node.properties.is_empty(),
            "hit[{i}] node must be hydrated with its stored property bag (the #830 D4 fix); \
             still empty == the bug",
        );
        assert_eq!(
            h.node.properties.get("text"),
            Some(&Value::String(
                docs[i_to_doc(b.node.id.raw(), &ids)].text.to_string()
            )),
            "hit[{i}] node.text must hydrate to the stored value",
        );
        assert_eq!(
            h.node.label_name.as_deref(),
            Some("Doc"),
            "hit[{i}] label_name must reverse-resolve to \"Doc\"",
        );
    }

    eprintln!(
        "#830 D4/#986 invariant: count {}→{}, ids {:?} (order preserved), props + \
         label_name hydrated in provider and substrate",
        baseline.len(),
        hydrated.len(),
        hydrated.iter().map(|h| h.node.id.raw()).collect::<Vec<_>>(),
    );
}

/// Map an internal node id back to its `corpus()` index via the ingest map, so
/// the per-hit `text` assertion above pins the RIGHT doc's text regardless of
/// the provider's rank order.
fn i_to_doc(internal_id: u64, ids: &BTreeMap<String, u64>) -> usize {
    let ext = ids
        .iter()
        .find(|(_, v)| **v == internal_id)
        .map(|(k, _)| k.as_str())
        .unwrap_or_else(|| panic!("internal id {internal_id} not in ingest map"));
    match ext {
        "doc-1" => 0,
        "doc-2" => 1,
        "doc-3" => 2,
        other => panic!("unexpected external id {other}"),
    }
}
