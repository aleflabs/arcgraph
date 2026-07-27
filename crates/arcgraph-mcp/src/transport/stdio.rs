//! W13δ M5-01 — stdio MCP transport.
//!
//! Reads Content-Length-framed JSON-RPC envelopes from stdin and
//! writes responses to stdout. Per the Anthropic MCP spec
//! (2025-11-25) stdio is the canonical local-server transport; the
//! parent process spawns the server, exchanges messages over the
//! stdio pipe, and shuts the server down via SIGTERM.
//!
//! # Concurrency
//!
//! The stdio loop is **strictly sequential** at v1.0-alpha: one
//! request in, one response out. This matches the LSP / MCP shape
//! and avoids the head-of-line blocking + ordering complexity of an
//! interleaved transport. M5-02 streamable-HTTP forward will lift
//! this to per-connection concurrency; M5-13 Bolt is also one-at-a-
//! time per connection per the protocol spec.
//!
//! # SIGTERM graceful shutdown
//!
//! Per amendment-03 §TIER-1 GAP C ("graceful drain at shutdown"), a
//! SIGTERM signal:
//! 1. Stops the accept loop (no new requests dispatched).
//! 2. Cancels the cancellation registry (every in-flight query
//!    surfaces `MCPError::Cancelled` at the next batch boundary).
//! 3. Flushes pending stdout writes.
//!
//! v1.0-alpha sequential dispatch means there's at most ONE in-
//! flight request at a time — the SIGTERM handler still cancels via
//! the registry so future M5-02 / M5-13 transports inherit the same
//! shape without re-implementing the drain.
//!
//! # ADR provenance
//! - **ADR-004 §"Tier 1 (agent-facing, default)"** — MCP tool catalog
//!   the dispatcher serves over this transport.
//! - **design-v2 §9 (Agent-Native MCP Interface)** — transport
//!   layering: stdio for local servers; streamable-HTTP at M5-02; Bolt
//!   at v1.1.
//! - **ADR-038 amendment-03 §TIER-2-c** — per-request tracing span
//!   tagged with `request_id`, `method`, `tenant_id`.
//! - **ADR-038 amendment-03 §TIER-1 GAP C** — SIGTERM graceful drain
//!   discipline (the M4-92 cancellation registry's `cancel_all`).

use std::sync::Arc;

use arcgraph_query::cancel::CancellationRegistry;
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncWrite, BufReader};

use crate::error::MCPError;
use crate::jsonrpc::{
    STDIO_MAX_MESSAGE_BYTES, StdioFramingMode, read_stdio_message_with_cap, write_stdio_message,
};
use crate::tools::explore::NeighborhoodExplorer;
use crate::tools::ingest::IngestProvider;
use crate::tools::inspect::NodeInspector;
use crate::tools::schema::SchemaProvider;
use crate::tools::search::HybridSearcher;
use crate::transport::bulkhead::{BulkheadOutcome, DispatchBulkhead};
use crate::transport::metrics::{ConnectionTransport, MetricsRegistry, ToolInvocationStatus};
use crate::transport::{Dispatcher, handle_raw_envelope, op_class_for_method};

/// Run the stdio MCP server until EOF on stdin OR until
/// `shutdown_signal` resolves. Returns when the loop exits cleanly
/// (peer closed stdin OR shutdown signaled); errors only on a
/// non-recoverable I/O fault on stdout.
///
/// # Parameters
///
/// - `dispatcher` — the per-session dispatcher composing the
///   tenant-bound [`SchemaProvider`] + [`NodeInspector`].
/// - `cancel_registry` — fires every in-flight query token on
///   shutdown via [`CancellationRegistry::cancel_all`]. v1.0-alpha
///   stdio dispatches sequentially so at most one entry is live;
///   future M5-02 streamable-HTTP transports run with concurrent
///   in-flight requests where the registry shape matters more.
/// - `reader` / `writer` — the async stdin / stdout handles. Tests
///   pass in-memory pipes; production callers pass
///   [`tokio::io::stdin`] / [`tokio::io::stdout`].
/// - `shutdown_signal` — a future that resolves on SIGTERM (or any
///   other shutdown condition the caller provides). Tests use a
///   `tokio::sync::oneshot::Receiver`; production callers use
///   [`tokio::signal::ctrl_c`] / a `signal::unix::SignalKind::terminate`
///   stream.
/// - `metrics` — W16γ M6-07 optional metrics registry. When
///   `Some`, the function emits `arcgraph_active_connections{transport="stdio"}`
///   as a 0→1 gauge over the session lifetime (mirrors the HTTP
///   `ActiveConnGuard` RAII pattern at `http.rs:823`). Stdio is
///   single-session per `serve_stdio` invocation (the v1.0-α
///   sequential dispatch shape), so the gauge oscillates between 0
///   (idle / pre-start) and 1 (in-session). `None` skips emission.
///
/// # Errors
///
/// Returns [`MCPError::InternalError`] only on a stdout write
/// failure (the parent process is gone — there's no recovery). All
/// other faults route through the JSON-RPC error envelope and are
/// emitted to the peer; the loop continues.
pub async fn serve_stdio<Rd, W, Sig, S, I, E, H, G, R>(
    dispatcher: Arc<Dispatcher<S, I, E, H, G, R>>,
    cancel_registry: &CancellationRegistry,
    reader: Rd,
    writer: W,
    shutdown_signal: Sig,
    metrics: Option<Arc<MetricsRegistry>>,
) -> Result<ServeStats, MCPError>
where
    Rd: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
    Sig: std::future::Future<Output = ()> + Send,
    S: SchemaProvider + Send + Sync + 'static,
    I: NodeInspector + Send + Sync + 'static,
    E: NeighborhoodExplorer + Send + Sync + 'static,
    H: HybridSearcher + Send + Sync + 'static,
    G: IngestProvider + Send + Sync + 'static,
    R: crate::tools::raw_query::RawQueryExecutor + Send + Sync + 'static,
{
    tracing::info!(
        target: "arcgraph_mcp::stdio",
        tenant_id = dispatcher.session_tenant.raw(),
        "stdio MCP transport starting",
    );

    // W16γ M6-07 — increment active_connections{transport="stdio"} at
    // session start; the StdioActiveConnGuard decrements on Drop
    // (function-scope guard fires on any return: clean EOF, shutdown,
    // I/O error). Mirror of `http.rs::ActiveConnGuard` at line 823.
    if let Some(m) = metrics.as_ref() {
        m.set_active_connections(ConnectionTransport::Stdio, 1);
    }
    let _active_guard = StdioActiveConnGuard {
        metrics: metrics.clone(),
    };

    let mut reader = BufReader::new(reader);
    let mut writer = writer;
    let mut stats = ServeStats::default();
    let mut framing_mode: Option<StdioFramingMode> = None;

    // AHP-1 (ADR-225 §3) — stdio is single-session + strictly sequential
    // (≤ 1 in-flight dispatch), so the bulkhead's semaphore is trivially
    // satisfied; its value here is moving the blocking dispatch OFF the
    // reactor via `spawn_blocking` so a durable write no longer delays the
    // SIGTERM drain / other runtime tasks. No per-request deadline is
    // threaded through stdio, so dispatches run to completion (`None`).
    let bulkhead = DispatchBulkhead::with_default_cap();

    tokio::pin!(shutdown_signal);

    loop {
        tokio::select! {
            // Branch 1: read the next framed message.
            //
            // The select! is cancellation-safe by virtue of
            // BufReader: a partial header read does not corrupt the
            // stream because the next iteration calls `read_line`
            // which is BufReader-cancellation-safe (per tokio docs:
            // "AsyncBufReadExt::read_line is cancellation safe").
            // The read_exact on the body, however, is NOT
            // cancellation-safe; we mitigate by handling shutdown at
            // the message boundary only — once a header read kicks
            // off a body read, we let it complete before checking
            // shutdown. This is fine for v1.0-alpha sequential
            // dispatch; in-flight body reads complete in O(MB / GB-
            // per-second) ≈ ms.
            // #818 — the stdio peer is the trusted local MCP host, and
            // `graph.ingest` is bulk-data; frame at STDIO_MAX_MESSAGE_BYTES
            // (512 MiB) rather than the 16 MiB untrusted-network cap, so a
            // real-scale single-batch ingest is not silently rejected.
            msg = read_stdio_message_with_cap(&mut reader, STDIO_MAX_MESSAGE_BYTES, &mut framing_mode) => {
                match msg {
                    Ok(Some(envelope)) => {
                        stats.messages_in += 1;
                        // W28 #588 — capture dispatch metadata BEFORE the
                        // `envelope` Value is moved into `handle_raw_envelope`,
                        // so the post-dispatch metric record can label by
                        // method + op-class. We only allocate when a metrics
                        // registry is wired; the `None` path is the legacy
                        // zero-overhead default (mirror of the http.rs
                        // `policy.metrics.is_some()` gate at the W15γ M6-06
                        // dispatch site, transport/http.rs:1456).
                        let dispatch_meta = metrics.as_ref().map(|m| {
                            let method = envelope
                                .get("method")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();
                            (m, method, std::time::Instant::now())
                        });
                        // AHP-1 (ADR-225 §3) — off-reactor dispatch behind
                        // the bulkhead. Capture the request id BEFORE the
                        // `envelope` `Value` is moved into the blocking
                        // closure so a panicked dispatch can still echo it.
                        let envelope_id = envelope.get("id").cloned().unwrap_or(Value::Null);
                        let disp = Arc::clone(&dispatcher);
                        let response = match bulkhead
                            .run(None, move || handle_raw_envelope(disp.as_ref(), envelope))
                            .await
                        {
                            BulkheadOutcome::Completed(r) => r,
                            // No deadline is passed for stdio, so `TimedOut`
                            // is unreachable; fold it in with the panic path
                            // defensively — emit a -32603 envelope + keep the
                            // loop alive (a single bad dispatch must not tear
                            // down the session).
                            BulkheadOutcome::Panicked | BulkheadOutcome::TimedOut => {
                                tracing::error!(
                                    target: "arcgraph_mcp::stdio",
                                    "stdio dispatch task panicked; emitting -32603 internal-error envelope",
                                );
                                let env = crate::jsonrpc::JsonRpcErrorResponse::from_mcp(
                                    envelope_id,
                                    &MCPError::InternalError("dispatch task panicked".to_string()),
                                );
                                Some(serde_json::to_value(env).unwrap_or(Value::Null))
                            }
                        };
                        // W28 #588 — record the MCP tool invocation (counter
                        // `arcgraph_mcp_tool_invocations`) + read/write latency
                        // (histogram `arcgraph_{read,write}_latency_ms`) for the
                        // stdio transport. Direct mirror of the http.rs
                        // `record_dispatch` site (transport/http.rs:1456-1469):
                        // op-class is derived from the method name via the
                        // shared `op_class_for_method`, and status is inferred
                        // from the envelope shape — an `error` member ⇒ Error,
                        // else Ok (the notification `None`-response case is by
                        // definition NOT a JSON-RPC error, so it tags Ok).
                        //
                        // This closes the W16γ M6-07 forward-pin (#588): before
                        // this, stdio threaded the registry for the
                        // `active_connections` gauge ONLY (see the session-start
                        // `set_active_connections` above), so the §10.2 tool /
                        // latency producers saw no data through the production
                        // `arcgraph` binary (which uses stdio + bolt; the
                        // record_dispatch producer lived only on the
                        // forward-pinned HTTP transport). Cite: design-v2 §10.2
                        // (`arcgraph_mcp_tool_invocations` + `arcgraph_{read,write}_latency_ms`).
                        if let Some((m, method, start)) = dispatch_meta {
                            let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
                            let op_class = op_class_for_method(&method);
                            let status = match response.as_ref() {
                                Some(env) if env.get("error").is_some() => {
                                    ToolInvocationStatus::Error
                                }
                                _ => ToolInvocationStatus::Ok,
                            };
                            m.record_dispatch(
                                dispatcher.session_tenant,
                                &method,
                                op_class,
                                status,
                                elapsed_ms,
                            );
                        }
                        if let Some(response) = response {
                            stats.messages_out += 1;
                            let mode = framing_mode.unwrap_or(StdioFramingMode::ContentLength);
                            if let Err(e) = write_stdio_message(&mut writer, &response, mode).await {
                                tracing::error!(
                                    target: "arcgraph_mcp::stdio",
                                    error = %e,
                                    "stdout write failed; shutting down stdio loop",
                                );
                                return Err(e);
                            }
                        }
                    }
                    Ok(None) => {
                        // EOF on stdin → peer closed cleanly.
                        tracing::info!(
                            target: "arcgraph_mcp::stdio",
                            "stdin EOF; stdio MCP transport exiting cleanly",
                        );
                        stats.exit_reason = ExitReason::PeerClosed;
                        return Ok(stats);
                    }
                    Err(e) => {
                        // Framing-layer fault → emit a -32700 parse-
                        // error envelope per JSON-RPC §5.1.
                        stats.parse_errors += 1;
                        // #818 — LOG the fault (the prior code logged only on
                        // a subsequent stdout-write failure, so an over-cap
                        // `graph.ingest` rejection was "no error and no log"
                        // from the operator's side). A WARN here makes every
                        // framing fault (oversized frame, malformed header,
                        // bad JSON) visible server-side.
                        tracing::warn!(
                            target: "arcgraph_mcp::stdio",
                            error = %e,
                            parse_errors = stats.parse_errors,
                            "stdio framing fault; emitting -32700 parse-error envelope",
                        );
                        let envelope = error_envelope_for_unknown_id(&e);
                        let mode = framing_mode.unwrap_or(StdioFramingMode::ContentLength);
                        if let Err(write_err) =
                            write_stdio_message(&mut writer, &envelope, mode).await
                        {
                            tracing::error!(
                                target: "arcgraph_mcp::stdio",
                                error = %write_err,
                                "stdout write failed during parse-error response",
                            );
                            return Err(write_err);
                        }
                        // Continue the loop — a single bad envelope
                        // shouldn't tear down the session. (The peer
                        // can recover by sending a fresh framed
                        // message.)
                    }
                }
            }

            // Branch 2: shutdown signal fired.
            _ = &mut shutdown_signal => {
                tracing::info!(
                    target: "arcgraph_mcp::stdio",
                    in_flight = cancel_registry.len(),
                    "shutdown signal received; cancelling in-flight queries",
                );
                let fired = cancel_registry.cancel_all();
                stats.in_flight_cancelled = fired;
                stats.exit_reason = ExitReason::ShutdownSignal;
                // Flush pending stdout writes. Errors here are best-
                // effort — we're shutting down anyway.
                if let Err(e) = tokio::io::AsyncWriteExt::flush(&mut writer).await {
                    tracing::warn!(
                        target: "arcgraph_mcp::stdio",
                        error = %e,
                        "stdout flush during shutdown failed",
                    );
                }
                return Ok(stats);
            }
        }
    }
}

/// Per-loop telemetry returned by [`serve_stdio`]. Useful for tests
/// + future structured-log emission at session-end.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServeStats {
    /// Number of inbound framed messages successfully read.
    pub messages_in: u64,
    /// Number of outbound response envelopes successfully written.
    pub messages_out: u64,
    /// Number of inbound messages that failed framing/parsing.
    pub parse_errors: u64,
    /// Number of cancellation tokens fired during shutdown drain.
    pub in_flight_cancelled: usize,
    /// Why the loop exited.
    pub exit_reason: ExitReason,
}

/// Reason the stdio loop exited.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExitReason {
    /// Loop has not yet exited (default for in-progress stats).
    #[default]
    InProgress,
    /// Peer closed stdin cleanly (EOF).
    PeerClosed,
    /// Shutdown signal fired (SIGTERM / ctrl-c / explicit).
    ShutdownSignal,
}

fn error_envelope_for_unknown_id(err: &MCPError) -> Value {
    // The request envelope was unparseable, so we don't have an `id`
    // to echo. Per JSON-RPC §5.1 the `id` MUST be `null` when the
    // originating request is unknown.
    let env = crate::jsonrpc::JsonRpcErrorResponse::from_mcp(Value::Null, err);
    serde_json::to_value(env).unwrap_or(Value::Null)
}

/// W16γ M6-07 — RAII guard decrementing
/// `arcgraph_active_connections{transport="stdio"}` to 0 when the
/// `serve_stdio` future drops.
///
/// Mirror of `http.rs:823 ActiveConnGuard`. Stdio is single-session
/// per invocation (v1.0-α sequential dispatch), so this guard
/// resets the gauge to 0 unconditionally (a multi-session future
/// would track running session count instead).
struct StdioActiveConnGuard {
    metrics: Option<Arc<MetricsRegistry>>,
}

impl Drop for StdioActiveConnGuard {
    fn drop(&mut self) {
        if let Some(m) = self.metrics.as_ref() {
            m.set_active_connections(ConnectionTransport::Stdio, 0);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Convenience: bind a SIGTERM future via tokio::signal.
// ─────────────────────────────────────────────────────────────────────

/// Build a future that resolves on SIGTERM (Unix) or Ctrl-C (cross-
/// platform fallback). Production callers use this; tests use a
/// `tokio::sync::oneshot::Receiver` they fire explicitly.
///
/// On Unix: listens for `SignalKind::terminate` AND `Ctrl-C`.
/// On non-Unix: listens for `Ctrl-C` only (production targets are
/// Linux per design-v2 §JD-2; macOS dev / Windows fallback).
pub async fn shutdown_on_term() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    target: "arcgraph_mcp::stdio",
                    error = %e,
                    "could not register SIGTERM handler; falling back to Ctrl-C only",
                );
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = term.recv() => {
                tracing::info!(
                    target: "arcgraph_mcp::stdio",
                    "SIGTERM received",
                );
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!(
                    target: "arcgraph_mcp::stdio",
                    "Ctrl-C received",
                );
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::explore::{Neighborhood, NeighborhoodEdge, NeighborhoodNode};
    use crate::tools::ingest::{IngestBatch, IngestRecordOutcome, IngestSummary};
    use crate::tools::inspect::{NeighborDirection, NeighborInfo, NodeInspection};
    use crate::tools::schema::{GraphSchema, IndexDescriptor, IndexKind, LabelInfo, RelTypeInfo};
    use crate::tools::search::{AvailableSubstrates, SearchHit};
    use arcgraph_core::TenantId;
    use arcgraph_query::CancellationToken;
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use tokio::io::AsyncWriteExt;

    struct StubSchema(TenantId);
    impl SchemaProvider for StubSchema {
        fn schema(&self, tenant: TenantId) -> Result<GraphSchema, MCPError> {
            if tenant != self.0 {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            Ok(GraphSchema {
                tenant_id: tenant.raw(),
                labels: vec![LabelInfo {
                    name: "Person".into(),
                    cardinality: None,
                    properties: vec![],
                }],
                rel_types: vec![RelTypeInfo {
                    name: "KNOWS".into(),
                    cardinality: None,
                }],
                indexes: vec![IndexDescriptor {
                    kind: IndexKind::Bm25,
                    available: true,
                }],
                total_node_count: None,
                total_rel_count: None,
            })
        }
    }

    struct StubInspect(TenantId);
    impl NodeInspector for StubInspect {
        fn inspect(&self, tenant: TenantId, node_id: u64) -> Result<NodeInspection, MCPError> {
            if tenant != self.0 {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            let mut props = BTreeMap::new();
            props.insert("name".into(), json!("Alice"));
            Ok(NodeInspection {
                id: node_id,
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

    struct StubExplore(TenantId);
    impl NeighborhoodExplorer for StubExplore {
        fn explore(
            &self,
            tenant: TenantId,
            seed: u64,
            max_depth: u32,
            _rel_filter: Option<&[String]>,
            _direction: crate::tools::explore::ExploreDirection,
            cancel: &CancellationToken,
        ) -> Result<Neighborhood, MCPError> {
            if cancel.is_cancelled() {
                return Err(MCPError::Cancelled);
            }
            if tenant != self.0 {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            Ok(Neighborhood {
                seed,
                max_depth,
                truncated: false,
                nodes: vec![NeighborhoodNode {
                    id: seed,
                    label: Some("Person".into()),
                    depth: 0,
                    properties: BTreeMap::new(),
                }],
                edges: vec![NeighborhoodEdge {
                    from: seed,
                    to: seed + 1,
                    rel_type: Some("KNOWS".into()),
                    direction: NeighborDirection::Out,
                }],
            })
        }
    }

    struct StubSearch(TenantId);
    impl HybridSearcher for StubSearch {
        fn available_substrates(
            &self,
            tenant: TenantId,
            cancel: &CancellationToken,
        ) -> Result<AvailableSubstrates, MCPError> {
            if cancel.is_cancelled() {
                return Err(MCPError::Cancelled);
            }
            if tenant != self.0 {
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
            _q: &str,
            _v: Option<&[f32]>,
            k: u32,
            cancel: &CancellationToken,
        ) -> Result<Vec<SearchHit>, MCPError> {
            if cancel.is_cancelled() {
                return Err(MCPError::Cancelled);
            }
            if tenant != self.0 {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            let mut h = vec![SearchHit {
                node_id: 1,
                label: Some("Doc".into()),
                score: 1.0,
            }];
            h.truncate(k as usize);
            Ok(h)
        }
    }

    struct StubIngest(TenantId);
    impl IngestProvider for StubIngest {
        fn ingest(&self, tenant: TenantId, _batch: IngestBatch) -> Result<IngestSummary, MCPError> {
            if tenant != self.0 {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            Ok(IngestSummary {
                records: vec![IngestRecordOutcome::Inserted {
                    internal_id: 42,
                    external_id: None,
                }],
                inserted_count: 1,
                failed_count: 0,
                commit_lsn: Some(42),
                dropped_acl_grants: Vec::new(),
            })
        }
    }

    /// Tiny raw-query stub (W16ζ M5-11). stdio integ tests do not
    /// exercise raw_query directly; the stub exists only to satisfy
    /// the W16ζ-merged `Dispatcher`'s `RawQueryExecutor` generic.
    struct StubRawQuery(TenantId);
    impl crate::tools::raw_query::RawQueryExecutor for StubRawQuery {
        fn execute(
            &self,
            tenant: TenantId,
            _query: &str,
            _max_rows: u32,
            _cancel: &CancellationToken,
        ) -> Result<crate::tools::raw_query::RawQueryRows, MCPError> {
            if tenant != self.0 {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            Err(MCPError::InternalError(
                "stub raw_query not exercised by stdio unit tests".into(),
            ))
        }
    }

    fn dispatcher(
        t: u64,
    ) -> Arc<Dispatcher<StubSchema, StubInspect, StubExplore, StubSearch, StubIngest, StubRawQuery>>
    {
        let tid = TenantId::new(t);
        // AHP-1 — `serve_stdio` now takes `Arc<Dispatcher>` (the
        // `spawn_blocking` bulkhead needs an owned `'static` dispatcher).
        Arc::new(Dispatcher::new(
            tid,
            Arc::new(StubSchema(tid)),
            Arc::new(StubInspect(tid)),
            Arc::new(StubExplore(tid)),
            Arc::new(StubSearch(tid)),
            Arc::new(StubIngest(tid)),
            Arc::new(StubRawQuery(tid)),
        ))
    }

    fn frame(payload: &str) -> Vec<u8> {
        let mut out = format!("Content-Length: {}\r\n\r\n", payload.len()).into_bytes();
        out.extend_from_slice(payload.as_bytes());
        out
    }

    #[tokio::test]
    async fn serve_stdio_processes_one_request_then_eofs() {
        let d = dispatcher(7);
        let cr = CancellationRegistry::new();
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"graph.schema","params":{"tenant_id":7}}"#;
        let input = frame(req);
        let mut output: Vec<u8> = Vec::new();
        // Shutdown future: never fires (peer EOF triggers exit first).
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        let stats = serve_stdio(d, &cr, &input[..], &mut output, shutdown, None)
            .await
            .expect("serve_stdio ok");
        assert_eq!(stats.messages_in, 1);
        assert_eq!(stats.messages_out, 1);
        assert_eq!(stats.exit_reason, ExitReason::PeerClosed);
        // Output is a Content-Length-framed success envelope.
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("Content-Length"));
        assert!(s.contains("\"id\":1"));
        assert!(s.contains("\"result\""));
    }

    #[tokio::test]
    async fn serve_stdio_emits_parse_error_envelope_for_malformed_input() {
        let d = dispatcher(7);
        let cr = CancellationRegistry::new();
        // Body with malformed JSON.
        let bad = "Content-Length: 5\r\n\r\n{bad}";
        let mut output: Vec<u8> = Vec::new();
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        let stats = serve_stdio(d, &cr, bad.as_bytes(), &mut output, shutdown, None)
            .await
            .expect("loop exits on EOF after error reply");
        assert!(stats.parse_errors >= 1);
        let s = String::from_utf8(output).unwrap();
        // Parse-error code -32700 appears in the response body.
        assert!(s.contains("-32700"));
    }

    #[tokio::test]
    async fn serve_stdio_returns_on_shutdown_signal() {
        // No input — the loop blocks on read. Then shutdown fires
        // and the loop returns ShutdownSignal.
        let d = dispatcher(7);
        let cr = CancellationRegistry::new();
        // Pretend a query is in-flight by registering a token.
        let qid = arcgraph_query::QueryId::new();
        let _t = cr.register(qid);
        assert_eq!(cr.len(), 1);

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        // Use an empty read source that blocks forever; tokio's
        // `tokio::io::empty()` returns AsyncRead that always reports
        // EOF, which would short-circuit the loop. Instead use a
        // pipe whose write side we never close.
        let (_tx_pipe, rx_pipe) = tokio::sync::mpsc::unbounded_channel::<u8>();
        let reader = AsyncReadFromMpsc {
            rx: rx_pipe,
            buf: Vec::new(),
        };
        let mut output: Vec<u8> = Vec::new();
        let serve = serve_stdio(d, &cr, reader, &mut output, shutdown, None);
        // Fire the shutdown after a tiny delay to let serve_stdio
        // actually start the read.
        let stats_handle = tokio::spawn(async move {
            // Borrow workaround: can't move serve into spawn while
            // d, cr, output are referenced. Run inline instead.
        });
        drop(stats_handle);
        // Simpler: race the serve future with a fire-the-signal
        // future in the same task.
        let signal_task = async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let _ = tx.send(());
        };
        let (stats, _) = tokio::join!(serve, signal_task);
        let stats = stats.expect("ok");
        assert_eq!(stats.exit_reason, ExitReason::ShutdownSignal);
        // SIGTERM drain: cancel_all was called → the registered
        // token tripped.
        assert_eq!(stats.in_flight_cancelled, 1);
    }

    /// Helper: a trivial AsyncRead that reads from an mpsc receiver
    /// (so tests can dripfeed bytes without closing the stream).
    struct AsyncReadFromMpsc {
        rx: tokio::sync::mpsc::UnboundedReceiver<u8>,
        buf: Vec<u8>,
    }
    impl AsyncRead for AsyncReadFromMpsc {
        fn poll_read(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            // Drain any buffered bytes into `buf`.
            if !self.buf.is_empty() {
                let n = std::cmp::min(buf.remaining(), self.buf.len());
                buf.put_slice(&self.buf[..n]);
                self.buf.drain(..n);
                return std::task::Poll::Ready(Ok(()));
            }
            // Try to receive more bytes from the mpsc.
            match self.rx.poll_recv(cx) {
                std::task::Poll::Ready(Some(byte)) => {
                    buf.put_slice(&[byte]);
                    std::task::Poll::Ready(Ok(()))
                }
                std::task::Poll::Ready(None) => std::task::Poll::Ready(Ok(())),
                std::task::Poll::Pending => std::task::Poll::Pending,
            }
        }
    }

    #[tokio::test]
    async fn serve_stdio_handles_two_requests_in_sequence() {
        let d = dispatcher(7);
        let cr = CancellationRegistry::new();
        let req1 = r#"{"jsonrpc":"2.0","id":1,"method":"graph.schema","params":{"tenant_id":7}}"#;
        let req2 = r#"{"jsonrpc":"2.0","id":2,"method":"graph.inspect","params":{"tenant_id":7,"node_id":1}}"#;
        let mut input = frame(req1);
        input.extend_from_slice(&frame(req2));
        let mut output: Vec<u8> = Vec::new();
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        let stats = serve_stdio(d, &cr, &input[..], &mut output, shutdown, None)
            .await
            .expect("ok");
        assert_eq!(stats.messages_in, 2);
        assert_eq!(stats.messages_out, 2);
        let s = String::from_utf8(output).unwrap();
        // Both ids appear in the output stream.
        assert!(s.contains("\"id\":1"), "id 1 in response");
        assert!(s.contains("\"id\":2"), "id 2 in response");
        // graph.inspect response carries the Alice property.
        assert!(s.contains("Alice"));
    }

    #[tokio::test]
    async fn serve_stdio_unauthorized_response_for_cross_tenant_request() {
        // Session bound to tenant 7; request asks for tenant 8.
        let d = dispatcher(7);
        let cr = CancellationRegistry::new();
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"graph.schema","params":{"tenant_id":8}}"#;
        let input = frame(req);
        let mut output: Vec<u8> = Vec::new();
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        serve_stdio(d, &cr, &input[..], &mut output, shutdown, None)
            .await
            .unwrap();
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("-32002"), "unauthorized code emitted");
    }

    #[tokio::test]
    async fn write_message_via_stdio_loop_round_trips() {
        // Roundtrip pin: an envelope written by the serve_stdio loop
        // is parseable by jsonrpc::read_message at the reader end.
        let d = dispatcher(7);
        let cr = CancellationRegistry::new();
        let req = r#"{"jsonrpc":"2.0","id":99,"method":"graph.schema","params":{"tenant_id":7}}"#;
        let input = frame(req);
        let mut output: Vec<u8> = Vec::new();
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        serve_stdio(d, &cr, &input[..], &mut output, shutdown, None)
            .await
            .unwrap();
        // Re-parse the response.
        let mut r = BufReader::new(&output[..]);
        let parsed = crate::jsonrpc::read_message(&mut r).await.unwrap().unwrap();
        assert_eq!(parsed["id"], 99);
        assert!(parsed.get("result").is_some());
    }

    #[tokio::test]
    async fn serve_stdio_flushes_writer_on_shutdown() {
        // Pin: shutdown path calls flush on the writer. We use a
        // wrapper writer that records whether flush was called.
        struct FlushTracker {
            inner: Vec<u8>,
            flushed: std::sync::Arc<std::sync::atomic::AtomicBool>,
        }
        impl AsyncWrite for FlushTracker {
            fn poll_write(
                mut self: std::pin::Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
                buf: &[u8],
            ) -> std::task::Poll<std::io::Result<usize>> {
                self.inner.extend_from_slice(buf);
                std::task::Poll::Ready(Ok(buf.len()))
            }
            fn poll_flush(
                self: std::pin::Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<std::io::Result<()>> {
                self.flushed
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                std::task::Poll::Ready(Ok(()))
            }
            fn poll_shutdown(
                self: std::pin::Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<std::io::Result<()>> {
                std::task::Poll::Ready(Ok(()))
            }
        }
        let flushed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let writer = FlushTracker {
            inner: Vec::new(),
            flushed: flushed.clone(),
        };
        let d = dispatcher(7);
        let cr = CancellationRegistry::new();
        let (_tx_pipe, rx_pipe) = tokio::sync::mpsc::unbounded_channel::<u8>();
        let reader = AsyncReadFromMpsc {
            rx: rx_pipe,
            buf: Vec::new(),
        };
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        let signal_task = async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let _ = tx.send(());
        };
        // Suppress unused-mut warning on writer by binding through a
        // local mut.
        let mut writer = writer;
        let _ = AsyncWriteExt::write_all(&mut writer, b"").await;
        let serve = serve_stdio(d, &cr, reader, writer, shutdown, None);
        let (_stats, _) = tokio::join!(serve, signal_task);
        assert!(
            flushed.load(std::sync::atomic::Ordering::SeqCst),
            "shutdown path must flush stdout"
        );
    }

    /// W16γ M6-07 — pin: `serve_stdio` with a metrics registry sets
    /// `arcgraph_active_connections{transport="stdio"}` to 1 during
    /// the session and back to 0 after the future returns. Mirror
    /// of the HTTP transport's `ActiveConnGuard` discipline.
    #[tokio::test]
    async fn serve_stdio_emits_active_connections_gauge_via_raii_guard() {
        let metrics = MetricsRegistry::shared().expect("metrics init");
        let d = dispatcher(7);
        let cr = CancellationRegistry::new();
        // Empty input → loop immediately observes EOF and returns.
        let input: &[u8] = &[];
        let mut output: Vec<u8> = Vec::new();
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        let stats = serve_stdio(d, &cr, input, &mut output, shutdown, Some(metrics.clone()))
            .await
            .unwrap();
        assert_eq!(stats.exit_reason, ExitReason::PeerClosed);
        // After return, the gauge must be 0 (RAII guard fired on
        // function-scope drop).
        let text = String::from_utf8(metrics.gather_text().expect("gather")).expect("utf-8");
        assert!(
            text.contains(r#"arcgraph_active_connections{transport="stdio"} 0"#),
            "post-serve_stdio, stdio gauge must be 0; text was:\n{text}"
        );
        // The gauge must have been observed at least once (the set
        // to 1 at session start; the set to 0 at drop). The text
        // exposition shows the most recent value (0); the metric
        // exists in the registry.
    }

    /// W28 #588 — strong-oracle pin: `serve_stdio` with a metrics
    /// registry records the §10.2 tool-invocation counter + read/write
    /// latency histograms PER DISPATCHED REQUEST (not merely the
    /// `active_connections` gauge). This is the load-bearing producer
    /// wire that lets the production `arcgraph` binary's stdio transport
    /// surface real operator metrics at `/metrics`. A registry wired to
    /// nothing — the pre-#588 state, where stdio threaded the registry
    /// for the connection gauge only — would pass an endpoint-liveness
    /// test but FAIL these VALUE assertions (per `feedback_review_
    /// oracle_relaxations.md`: a green test that can't fail on its bug
    /// is worse than no test).
    #[tokio::test]
    async fn serve_stdio_records_dispatch_metrics_per_invocation() {
        let metrics = MetricsRegistry::shared().expect("metrics init");
        let d = dispatcher(7);
        let cr = CancellationRegistry::new();
        // Two real requests for the session tenant, one per op-class:
        //   - graph.schema  → READ-class  (op_class_for_method).
        //   - graph.ingest  → WRITE-class (op_class_for_method).
        // Concatenated framed; EOF after the second drives a clean
        // PeerClosed exit. The graph.ingest record-count is asserted
        // status-agnostically (the histogram `_count` increments whether
        // the dispatch returns ok OR an error envelope — record_dispatch
        // fires on every dispatched envelope, mirroring http.rs).
        let req_read =
            r#"{"jsonrpc":"2.0","id":1,"method":"graph.schema","params":{"tenant_id":7}}"#;
        let req_write =
            r#"{"jsonrpc":"2.0","id":2,"method":"graph.ingest","params":{"tenant_id":7}}"#;
        let mut input = frame(req_read);
        input.extend_from_slice(&frame(req_write));
        let mut output: Vec<u8> = Vec::new();
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        let stats = serve_stdio(
            d,
            &cr,
            &input[..],
            &mut output,
            shutdown,
            Some(metrics.clone()),
        )
        .await
        .unwrap();
        assert_eq!(stats.messages_in, 2, "exactly two inbound messages");
        assert_eq!(stats.exit_reason, ExitReason::PeerClosed);

        let text = String::from_utf8(metrics.gather_text().expect("gather")).expect("utf-8");
        // Counter moved to EXACTLY 1 for (tenant=7, tool=graph.schema,
        // status=ok) — proves record_dispatch fired through the stdio
        // loop, not just the connection gauge.
        assert!(
            text.contains(
                r#"arcgraph_mcp_tool_invocations{status="ok",tenant="7",tool="graph.schema"} 1"#
            ),
            "stdio dispatch must increment the tool-invocation counter to 1; text was:\n{text}"
        );
        // graph.schema is READ-class → observation lands in the READ
        // histogram, NOT the write histogram (op-class routing oracle).
        assert!(
            text.contains(r#"arcgraph_read_latency_ms_count{tenant="7",tool="graph.schema"} 1"#),
            "read-class dispatch must observe the read-latency histogram; text was:\n{text}"
        );
        assert!(
            !text.contains(r#"arcgraph_write_latency_ms_count{tenant="7",tool="graph.schema"}"#),
            "read-class dispatch must NOT touch the write-latency histogram; text was:\n{text}"
        );
        // graph.ingest is WRITE-class → observation lands in the WRITE
        // histogram (status-agnostic _count oracle).
        assert!(
            text.contains(r#"arcgraph_write_latency_ms_count{tenant="7",tool="graph.ingest"} 1"#),
            "write-class dispatch must observe the write-latency histogram; text was:\n{text}"
        );
        assert!(
            !text.contains(r#"arcgraph_read_latency_ms_count{tenant="7",tool="graph.ingest"}"#),
            "write-class dispatch must NOT touch the read-latency histogram; text was:\n{text}"
        );
    }
}
