//! W13δ M5-01 integration tests — end-to-end stdio MCP transport.
//!
//! These tests bootstrap an in-memory pipe pair, drive the
//! [`arcgraph_mcp::serve_stdio`] loop, and verify the canonical
//! request → response flow. Subprocess-based end-to-end tests are
//! gated `#[ignore]` per the W13δ spawn prompt's empirical-gauntlet
//! step ("Subprocess-based end-to-end MCP test (mark `#[ignore]` if
//! it can't run on bare `cargo test`)").

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::tools::explore::{Neighborhood, NeighborhoodEdge, NeighborhoodNode};
use arcgraph_mcp::tools::ingest::{
    IngestBatch, IngestProvider, IngestRecordOutcome, IngestSummary,
};
use arcgraph_mcp::tools::inspect::{NeighborDirection, NeighborInfo, NodeInspection};
use arcgraph_mcp::tools::schema::{
    GraphSchema, IndexDescriptor, IndexKind, LabelInfo, PropertyDescriptor, RelTypeInfo,
};
use arcgraph_mcp::tools::search::{AvailableSubstrates, SearchHit};
use arcgraph_mcp::{
    Dispatcher, HybridSearcher, MCPError, NeighborhoodExplorer, NodeInspector, SchemaProvider,
    serve_stdio,
};
use arcgraph_query::CancellationToken;
use arcgraph_query::cancel::CancellationRegistry;
use serde_json::{Value, json};
use tokio::io::BufReader;

// ─────────────────────────────────────────────────────────────────────
// Shared fixtures
// ─────────────────────────────────────────────────────────────────────

struct FxSchema {
    tenant: TenantId,
}

impl SchemaProvider for FxSchema {
    fn schema(&self, tenant: TenantId) -> Result<GraphSchema, MCPError> {
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(GraphSchema {
            tenant_id: tenant.raw(),
            labels: vec![
                LabelInfo {
                    name: "Person".into(),
                    cardinality: Some(100),
                    properties: vec![
                        PropertyDescriptor {
                            name: "name".into(),
                            kind: "STRING".into(),
                        },
                        PropertyDescriptor {
                            name: "age".into(),
                            kind: "INTEGER".into(),
                        },
                    ],
                },
                LabelInfo {
                    name: "Doc".into(),
                    cardinality: Some(50),
                    properties: vec![],
                },
            ],
            rel_types: vec![RelTypeInfo {
                name: "KNOWS".into(),
                cardinality: Some(250),
            }],
            indexes: vec![
                IndexDescriptor {
                    kind: IndexKind::Vector,
                    available: true,
                },
                IndexDescriptor {
                    kind: IndexKind::Bm25,
                    available: false,
                },
            ],
            total_node_count: Some(150),
            total_rel_count: Some(250),
        })
    }
}

struct FxInspect {
    tenant: TenantId,
}

impl NodeInspector for FxInspect {
    fn inspect(&self, tenant: TenantId, node_id: u64) -> Result<NodeInspection, MCPError> {
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        if node_id != 1 {
            return Err(MCPError::QueryError(format!("node {node_id} not found")));
        }
        let mut props: BTreeMap<String, Value> = BTreeMap::new();
        props.insert("name".into(), json!("Alice"));
        props.insert("age".into(), json!(30));
        Ok(NodeInspection {
            id: 1,
            label: Some("Person".into()),
            properties: props,
            neighbors: vec![NeighborInfo {
                node_id: 2,
                label: Some("Person".into()),
                rel_type: Some("KNOWS".into()),
                direction: NeighborDirection::Out,
            }],
        })
    }
}

struct FxExplore {
    tenant: TenantId,
}

impl NeighborhoodExplorer for FxExplore {
    fn explore(
        &self,
        tenant: TenantId,
        seed: u64,
        max_depth: u32,
        _rel_filter: Option<&[String]>,
        _direction: arcgraph_mcp::tools::explore::ExploreDirection,
        cancel: &CancellationToken,
    ) -> Result<Neighborhood, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        let mut props_seed: BTreeMap<String, Value> = BTreeMap::new();
        props_seed.insert("name".into(), json!("Alice"));
        let mut props_nbr: BTreeMap<String, Value> = BTreeMap::new();
        props_nbr.insert("name".into(), json!("Bob"));
        let mut nodes = vec![
            NeighborhoodNode {
                id: seed,
                label: Some("Person".into()),
                depth: 0,
                properties: props_seed,
            },
            NeighborhoodNode {
                id: seed + 1,
                label: Some("Person".into()),
                depth: 1,
                properties: props_nbr,
            },
        ];
        nodes.retain(|n| n.depth <= max_depth);
        let mut edges = vec![NeighborhoodEdge {
            from: seed,
            to: seed + 1,
            rel_type: Some("KNOWS".into()),
            direction: NeighborDirection::Out,
        }];
        let allowed: std::collections::HashSet<u64> = nodes.iter().map(|n| n.id).collect();
        edges.retain(|e| allowed.contains(&e.from) && allowed.contains(&e.to));
        Ok(Neighborhood {
            seed,
            max_depth,
            truncated: false,
            nodes,
            edges,
        })
    }
}

struct FxIngest {
    tenant: TenantId,
}

impl IngestProvider for FxIngest {
    fn ingest(&self, tenant: TenantId, batch: IngestBatch) -> Result<IngestSummary, MCPError> {
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        let mut records = Vec::new();
        let mut next_id = 100u64;
        let count = batch.nodes.len() + batch.relationships.len();
        for n in batch.nodes {
            records.push(IngestRecordOutcome::Inserted {
                internal_id: next_id,
                external_id: n.external_id,
            });
            next_id += 1;
        }
        for r in batch.relationships {
            records.push(IngestRecordOutcome::Inserted {
                internal_id: next_id,
                external_id: r.external_id,
            });
            next_id += 1;
        }
        Ok(IngestSummary {
            records,
            inserted_count: count as u64,
            failed_count: 0,
            commit_lsn: Some(next_id - 1),
            dropped_acl_grants: Vec::new(),
        })
    }
}

struct FxSearch {
    tenant: TenantId,
}

impl HybridSearcher for FxSearch {
    fn available_substrates(
        &self,
        tenant: TenantId,
        cancel: &CancellationToken,
    ) -> Result<AvailableSubstrates, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(AvailableSubstrates {
            vector: true,
            bm25: true,
        })
    }

    fn search(
        &self,
        tenant: TenantId,
        query_text: &str,
        _query_vec: Option<&[f32]>,
        k: u32,
        cancel: &CancellationToken,
    ) -> Result<Vec<SearchHit>, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        let base = vec![
            SearchHit {
                node_id: 1,
                label: Some("Document".into()),
                score: 0.95,
            },
            SearchHit {
                node_id: 2,
                label: Some("Document".into()),
                score: 0.81,
            },
        ];
        let mut hits = if query_text.is_empty() {
            base.into_iter().take(1).collect::<Vec<_>>()
        } else {
            base
        };
        hits.truncate(k as usize);
        Ok(hits)
    }
}

/// Stub raw-query executor (W16ζ M5-11) — stdio integ tests do not
/// exercise raw_query; the impl exists only to satisfy the W16ζ-merged
/// dispatcher's `RawQueryExecutor` generic.
struct FxRawQuery {
    tenant: TenantId,
}
impl arcgraph_mcp::tools::raw_query::RawQueryExecutor for FxRawQuery {
    fn execute(
        &self,
        tenant: TenantId,
        _query: &str,
        _max_rows: u32,
        _cancel: &CancellationToken,
    ) -> Result<arcgraph_mcp::tools::raw_query::RawQueryRows, MCPError> {
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub raw_query not exercised by stdio integ tests".into(),
        ))
    }
}

fn dispatcher(
    tenant: u64,
) -> Dispatcher<FxSchema, FxInspect, FxExplore, FxSearch, FxIngest, FxRawQuery> {
    let t = TenantId::new(tenant);
    Dispatcher::new(
        t,
        Arc::new(FxSchema { tenant: t }),
        Arc::new(FxInspect { tenant: t }),
        Arc::new(FxExplore { tenant: t }),
        Arc::new(FxSearch { tenant: t }),
        Arc::new(FxIngest { tenant: t }),
        Arc::new(FxRawQuery { tenant: t }),
    )
}

fn frame(payload: &str) -> Vec<u8> {
    let mut out = format!("Content-Length: {}\r\n\r\n", payload.len()).into_bytes();
    out.extend_from_slice(payload.as_bytes());
    out
}

async fn parse_responses(buf: &[u8]) -> Vec<Value> {
    let mut r = BufReader::new(buf);
    let mut out = Vec::new();
    while let Some(v) = arcgraph_mcp::read_message(&mut r).await.expect("ok") {
        out.push(v);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// 1. End-to-end stdio: framed request → framed response
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_end_to_end_stdio_schema_request_returns_framed_response() {
    let d = dispatcher(7);
    let cr = CancellationRegistry::new();
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"graph.schema","params":{"tenant_id":7,"format":"json"}}"#;
    let input = frame(req);
    let mut output: Vec<u8> = Vec::new();
    let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = async move {
        let _ = rx.await;
    };
    let stats = serve_stdio(
        std::sync::Arc::new(d),
        &cr,
        &input[..],
        &mut output,
        shutdown,
        None,
    )
    .await
    .expect("stdio loop ok");
    assert_eq!(stats.messages_in, 1);
    assert_eq!(stats.messages_out, 1);

    let responses = parse_responses(&output).await;
    assert_eq!(responses.len(), 1);
    let resp = &responses[0];
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    assert_eq!(resp["result"]["format"], "json");

    // Decode the inner body and verify the schema shape.
    let body_str = resp["result"]["body"].as_str().expect("body string");
    let body: GraphSchema = serde_json::from_str(body_str).expect("body parses as GraphSchema");
    assert_eq!(body.tenant_id, 7);
    assert!(body.labels.iter().any(|l| l.name == "Person"));
    assert!(body.rel_types.iter().any(|r| r.name == "KNOWS"));
    assert_eq!(body.total_node_count, Some(150));
}

// ─────────────────────────────────────────────────────────────────────
// 2. graph.schema returns the expected types per the M5-04 contract
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_schema_returns_expected_property_types() {
    let d = dispatcher(7);
    let cr = CancellationRegistry::new();
    let req = r#"{"jsonrpc":"2.0","id":42,"method":"graph.schema","params":{"tenant_id":7,"format":"json"}}"#;
    let input = frame(req);
    let mut output: Vec<u8> = Vec::new();
    let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
    serve_stdio(
        std::sync::Arc::new(d),
        &cr,
        &input[..],
        &mut output,
        async move {
            let _ = rx.await;
        },
        None,
    )
    .await
    .unwrap();

    let responses = parse_responses(&output).await;
    let body_str = responses[0]["result"]["body"]
        .as_str()
        .expect("body string");
    let body: GraphSchema = serde_json::from_str(body_str).unwrap();

    // M5-04 contract: per-label property descriptors carry name + kind.
    let person = body
        .labels
        .iter()
        .find(|l| l.name == "Person")
        .expect("Person label present");
    let name_prop = person
        .properties
        .iter()
        .find(|p| p.name == "name")
        .expect("name property");
    assert_eq!(name_prop.kind, "STRING");
    let age_prop = person
        .properties
        .iter()
        .find(|p| p.name == "age")
        .expect("age property");
    assert_eq!(age_prop.kind, "INTEGER");

    // Index descriptors round-trip with `available` per ADR-038 §"substrate-
    // availability flags" (consumed at bind time at M4-21, surfaced here at M5-04).
    let vector_idx = body
        .indexes
        .iter()
        .find(|i| matches!(i.kind, IndexKind::Vector))
        .expect("vector index descriptor");
    assert!(vector_idx.available);
    let bm25_idx = body
        .indexes
        .iter()
        .find(|i| matches!(i.kind, IndexKind::Bm25))
        .expect("bm25 index descriptor");
    assert!(!bm25_idx.available, "bm25 still building");
}

// ─────────────────────────────────────────────────────────────────────
// 3. graph.inspect cross-tenant rejection at the M5-05 boundary
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_inspect_cross_tenant_returns_unauthorized() {
    // Session bound to tenant 7; the request body asks for tenant 99.
    // MUST surface MCPError::Unauthorized (-32002) without ever
    // calling the inspector — pinned by the FxInspect.tenant != 99
    // (so a leak past the guard would surface TenantUnknown -32003,
    // which the assertion would catch).
    let d = dispatcher(7);
    let cr = CancellationRegistry::new();
    let req = r#"{"jsonrpc":"2.0","id":7,"method":"graph.inspect","params":{"tenant_id":99,"node_id":1}}"#;
    let input = frame(req);
    let mut output: Vec<u8> = Vec::new();
    let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
    serve_stdio(
        std::sync::Arc::new(d),
        &cr,
        &input[..],
        &mut output,
        async move {
            let _ = rx.await;
        },
        None,
    )
    .await
    .unwrap();

    let responses = parse_responses(&output).await;
    let resp = &responses[0];
    assert_eq!(resp["error"]["code"], -32002, "unauthorized code");
    assert!(
        resp.get("result").is_none(),
        "error envelope must not carry result"
    );
}

// ─────────────────────────────────────────────────────────────────────
// 4. Cancel-via-MCP fires the QueryEngine cancellation registry
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_cancel_via_shutdown_fires_inflight_query_token() {
    // Pin: when the SIGTERM-style shutdown signal fires while a
    // query is registered in the registry, `serve_stdio` invokes
    // `cancel_all` so the in-flight query surfaces Cancelled at the
    // next batch boundary. This is the M5↔M4 contract surface for
    // `QueryEngine::cancel` per ADR-038 amendment-03.
    let d = dispatcher(7);
    let cr = CancellationRegistry::new();
    // Pre-register a fake in-flight query.
    let qid = arcgraph_query::QueryId::new();
    let token = cr.register(qid);
    assert!(!token.is_cancelled(), "fresh token");

    // Drive the loop with a never-ending reader and fire the
    // shutdown after a short delay.
    let (read_tx, read_rx) = tokio::sync::mpsc::unbounded_channel::<u8>();
    drop(read_tx); // closing makes reader yield Pending forever (channel closed).
    let reader = ChannelReader { rx: read_rx };
    let mut output: Vec<u8> = Vec::new();
    let (sig_tx, sig_rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = async move {
        let _ = sig_rx.await;
    };
    let signal_task = async move {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let _ = sig_tx.send(());
    };
    let serve = serve_stdio(
        std::sync::Arc::new(d),
        &cr,
        reader,
        &mut output,
        shutdown,
        None,
    );
    let (stats, _) = tokio::join!(serve, signal_task);
    let stats = stats.expect("ok");
    assert_eq!(
        stats.exit_reason,
        arcgraph_mcp::ExitReason::ShutdownSignal,
        "shutdown drain executed"
    );
    assert_eq!(stats.in_flight_cancelled, 1, "registered token was fired");
    // The token bound to the QueryId tripped — this is the
    // QueryEngine::cancel surface in action.
    assert!(token.is_cancelled());
}

/// Async reader that yields Pending forever once the mpsc channel is
/// closed (the test pattern lets us "block" the reader so the
/// shutdown branch of the select! dominates).
struct ChannelReader {
    rx: tokio::sync::mpsc::UnboundedReceiver<u8>,
}

impl tokio::io::AsyncRead for ChannelReader {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        match self.rx.poll_recv(cx) {
            std::task::Poll::Ready(Some(byte)) => {
                buf.put_slice(&[byte]);
                std::task::Poll::Ready(Ok(()))
            }
            // Channel closed — pretend the read is pending so the
            // shutdown branch of the caller's select! has a chance.
            std::task::Poll::Ready(None) => std::task::Poll::Pending,
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// 5. Subprocess-based end-to-end MCP test
//
// Now lives at `crates/arcgraph-cli/tests/mcp_stdio_subprocess.rs` —
// the `arcgraph-cli` test crate is the only place
// `env!("CARGO_BIN_EXE_arcgraph-mcp-stdio")` resolves (Cargo only
// provides binary-path env vars for tests in the same package as the
// binary). The library-side integ tests above already exercise the
// in-memory pipe path; the subprocess test pins the binary's
// SIGTERM-aware run-loop.
// ─────────────────────────────────────────────────────────────────────
