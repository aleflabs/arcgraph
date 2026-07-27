//! W28 #588 (M6-08+ operator-metrics closure) — the production
//! `arcgraph` binary's Prometheus `/metrics` scrape listener.
//!
//! # What this closes
//!
//! `arcgraph_mcp::transport::metrics::MetricsRegistry` (W15γ M6-06)
//! composes the design-v2 §10.2 observability inventory and the HTTP
//! MCP transport (`transport/http.rs`) already mounts `GET /metrics`
//! on it. BUT the production `arcgraph serve` binary historically ran
//! only the **stdio** and **bolt** transports and never instantiated a
//! registry — so an operator running the production binary saw zero
//! metric data. The `metrics` containerPort was therefore removed
//! from `deploy/k8s/` until the dedicated `--metrics-http <addr>`
//! listener shipped. This module is that listener.
//!
//! (The HTTP MCP transport itself landed in #761 slice 1 —
//! `bin/arcgraph.rs::run_serve_http` now runs a live HTTPS transport,
//! no longer a `bail!` stub. It deliberately does NOT mount `/metrics`
//! on its public HTTPS data port; the scrape surface stays on this
//! dedicated `--metrics-http` loopback listener to keep scrape traffic
//! isolated from public data traffic.)
//!
//! # Why a SEPARATE listener (not co-located on the admin `/livez`
//! `/readyz` server)
//!
//! The `--metrics-http` listener binds its **own** axum server on its
//! own address, distinct from BOTH the MCP transport port and the admin
//! HTTP port (8090, `ops::admin_http`). This is the v1.0 posture. It
//! applies the same port-isolation rationale that put
//! `/livez`+`/readyz` on a separate port from the MCP transport:
//!
//! 1. **Network-policy granularity.** Probe traffic
//!    (`/livez`+`/readyz`) originates from the kubelet (host
//!    network namespace / control plane); scrape traffic (`/metrics`)
//!    originates from Prometheus (the `monitoring` namespace, typically
//!    via a `ServiceMonitor`). These are DIFFERENT origins. A separate
//!    L4 port lets a `NetworkPolicy` express "allow `monitoring` →
//!    `:9090`" distinctly from "allow `kube-system` → `:8090`" — the
//!    L4-vs-L7 benefit of distinct probe and scrape ports.
//! 2. **Failure isolation.** An operator most needs to scrape
//!    `/metrics` during an incident — possibly while a readiness
//!    handler is itself wedged. Decoupling the scrape listener from
//!    the probe listener keeps their fates independent.
//! 3. **design-v2 §10.2 documents the scrape endpoint on its own port
//!    (9090 by default)** — co-locating on the admin `:8090` would
//!    contradict the design doc's documented default and the existing
//!    dedicated `--metrics-http <addr>` flag (distinct from
//!    `--admin-http`).
//! 4. **Admin-port co-location remains a v1.1 consideration.**
//!    Shipping a separate listener now preserves the v1.0 isolation
//!    model without pre-empting that deferred question.
//!
//! # Runtime discipline (design-v2 §4.1)
//!
//! design-v2 §4.1 places the "Metrics exporter, log shipper" on the
//! **Tokio background pool**, NOT the Monoio thread-per-core hot path.
//! The scrape listener is operational/background (infrequent GET
//! requests, no latency SLO), so it is spawned via `tokio::spawn` from
//! the binary's `#[tokio::main(flavor = "multi_thread")]` runtime —
//! never on the Monoio executor that owns query/storage I/O.
//!
//! # Loopback-default + graceful degradation
//!
//! - **Loopback-default (W14 retro IR L1-HIGH-4).** `validate` rejects
//!   any non-loopback bind unless the operator opts in via
//!   `allow_remote_bind` — mirror of `ops::admin_http`'s
//!   `--allow-remote-admin-bind`. Production Kubernetes scraping
//!   (`0.0.0.0:9090`) requires the opt-in; localhost-only operators pay
//!   nothing. A non-loopback-without-opt-in bind is an operator
//!   *config* error and fails LOUD synchronously at startup (security).
//! - **Observability must not cascade into unavailability.** A runtime
//!   bind failure (e.g.
//!   `EADDRINUSE`) surfaces as a recoverable [`MetricsHttpError`] that
//!   `bin/arcgraph.rs::run_serve` LOGS — the main MCP server keeps
//!   running. The scrape endpoint degrading never takes the database
//!   down.

use std::net::SocketAddr;
use std::sync::Arc;

use arcgraph_mcp::{CONTENT_TYPE_PROMETHEUS_TEXT, MetricsRegistry, PATH_METRICS};
use axum::Router;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::get;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;

/// Configuration for the `--metrics-http` Prometheus scrape listener.
///
/// `#[serde(deny_unknown_fields)]` ensures a misspelled key rejects rather than silently
/// degrading. Forward-binds the M6-08+ `--config <toml>` schema landing.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsHttpServerConfig {
    /// Bind address. design-v2 §10.2 default scrape port is 9090.
    pub bind: SocketAddr,
    /// Whether to permit binding to a non-loopback address.
    ///
    /// Loopback-default per W14 retro IR L1-HIGH-4 — mirror of
    /// `ops::admin_http`'s `--allow-remote-admin-bind`. Kubernetes
    /// `ServiceMonitor` scraping binds `0.0.0.0:9090` and needs this
    /// opt-in; localhost-only operators leave it `false`.
    #[serde(default)]
    pub allow_remote_bind: bool,
}

impl Default for MetricsHttpServerConfig {
    fn default() -> Self {
        // design-v2 §10.2: "Prometheus scrape endpoint on port 9090 by
        // default." Loopback per W14 retro IR L1-HIGH-4.
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], 9090)),
            allow_remote_bind: false,
        }
    }
}

/// Errors emitted by the metrics HTTP server.
///
/// `#[non_exhaustive]` under the strict public-contract policy.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MetricsHttpError {
    /// Bind address rejected (non-loopback without `allow_remote_bind`).
    /// A startup *config* error — surfaced LOUD per the loopback-default
    /// security discipline (W14 retro IR L1-HIGH-4).
    #[error(
        "metrics HTTP server refused to bind {bind}: non-loopback without `allow_remote_bind` \
         (W14 retro IR L1-HIGH-4 loopback-default; pass --allow-remote-metrics-bind to opt in)"
    )]
    BindAddrForbidden { bind: SocketAddr },
    /// TCP listener bind failed (e.g. `EADDRINUSE`). A RUNTIME
    /// observability failure — `run_serve` logs this and keeps the main
    /// MCP server running (ADR-093 §Decision item 2: observability
    /// degradation must not cascade into unavailability).
    #[error("metrics HTTP server bind {bind} failed: {source}")]
    Bind {
        bind: SocketAddr,
        #[source]
        source: std::io::Error,
    },
    /// `axum::serve` returned with an error.
    #[error("metrics HTTP server serve loop returned an error: {source}")]
    Serve {
        #[source]
        source: std::io::Error,
    },
}

impl MetricsHttpServerConfig {
    /// Validate the loopback-default policy. Returns
    /// [`MetricsHttpError::BindAddrForbidden`] if `bind` is non-loopback
    /// and `allow_remote_bind` is false.
    ///
    /// Mirror of [`crate::ops::admin_http::AdminHttpServerConfig::validate`]
    /// — identical loopback-default discipline (W14 retro IR L1-HIGH-4)
    /// applied to the scrape surface.
    pub fn validate(&self) -> Result<(), MetricsHttpError> {
        if self.allow_remote_bind {
            return Ok(());
        }
        if self.bind.ip().is_loopback() {
            return Ok(());
        }
        Err(MetricsHttpError::BindAddrForbidden { bind: self.bind })
    }
}

/// State shared with the `/metrics` handler — the per-process
/// [`MetricsRegistry`] (cheap `Arc`-share; the prometheus internals are
/// `Arc`-backed so all clones observe the same counters).
#[derive(Clone)]
struct MetricsState {
    registry: Arc<MetricsRegistry>,
}

/// `GET /metrics` handler — emits the Prometheus text-exposition format.
///
/// Returns 200 + `CONTENT_TYPE_PROMETHEUS_TEXT` with the gathered
/// registry on success. The encode error path is unreachable on the
/// in-memory writer (`MetricsRegistry::gather_text` doc), but we map it
/// to a 500 + structured log rather than panicking — a wedged scrape
/// must never take the process down (ADR-093 §Decision item 2).
async fn metrics_handler(State(state): State<MetricsState>) -> impl IntoResponse {
    match state.registry.gather_text() {
        Ok(body) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, CONTENT_TYPE_PROMETHEUS_TEXT)],
            body,
        )
            .into_response(),
        Err(e) => {
            tracing::error!(
                target: "arcgraph_cli::ops::metrics_http",
                error = %e,
                "metrics text encoding failed; returning 500",
            );
            (StatusCode::INTERNAL_SERVER_ERROR, "metrics encode error\n").into_response()
        }
    }
}

/// Build the metrics HTTP router (`GET /metrics` only).
///
/// Public for tests that exercise the route over a real socket. The
/// path is [`arcgraph_mcp::PATH_METRICS`] — reusing the constant keeps
/// the scrape path byte-identical to the HTTP MCP transport's mount and
/// the Grafana / Prometheus / `ServiceMonitor` artifacts in `deploy/`.
pub fn build_router(registry: Arc<MetricsRegistry>) -> Router {
    let state = MetricsState { registry };
    Router::new()
        .route(PATH_METRICS, get(metrics_handler))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
}

/// Bind + serve the metrics HTTP router until `shutdown` resolves
/// (typically the same SIGTERM-aware future the MCP transport uses).
///
/// Spawned on the Tokio background runtime per design-v2 §4.1 (the
/// metrics exporter is operational/background, NOT a Monoio hot-path
/// surface).
///
/// # Errors
///
/// - [`MetricsHttpError::BindAddrForbidden`] — non-loopback bind without
///   `allow_remote_bind` (LOUD config error; caller bails at startup).
/// - [`MetricsHttpError::Bind`] — TCP bind syscall failed (e.g.
///   `EADDRINUSE`); the caller LOGS this and keeps serving (graceful
///   degradation — ADR-093 §Decision item 2).
/// - [`MetricsHttpError::Serve`] — axum serve loop returned an error.
pub async fn serve_metrics_http<F>(
    config: MetricsHttpServerConfig,
    registry: Arc<MetricsRegistry>,
    shutdown: F,
) -> Result<(), MetricsHttpError>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    config.validate()?;
    let router = build_router(registry);
    let listener =
        TcpListener::bind(config.bind)
            .await
            .map_err(|source| MetricsHttpError::Bind {
                bind: config.bind,
                source,
            })?;
    tracing::info!(
        target: "arcgraph_cli::ops::metrics_http",
        bind = %config.bind,
        path = PATH_METRICS,
        "metrics HTTP server listening (Prometheus scrape endpoint; design-v2 §10.2)",
    );
    axum::serve(listener, router.into_make_service())
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|source| MetricsHttpError::Serve { source })?;
    tracing::info!(
        target: "arcgraph_cli::ops::metrics_http",
        bind = %config.bind,
        "metrics HTTP server exiting cleanly",
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Tests — loopback-default policy, strong-oracle scrape values,
// graceful bind-conflict degradation.
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    use arcgraph_core::ids::TenantId;
    use arcgraph_mcp::{OpClass, ToolInvocationStatus};

    // ── Loopback-default validate() policy (mirror admin_http) ──────

    #[test]
    fn default_config_binds_loopback_9090_and_validates() {
        let cfg = MetricsHttpServerConfig::default();
        assert_eq!(cfg.bind, SocketAddr::from(([127, 0, 0, 1], 9090)));
        assert!(!cfg.allow_remote_bind);
        cfg.validate().expect("default loopback:9090 must validate");
    }

    #[test]
    fn non_loopback_bind_without_opt_in_rejects() {
        let cfg = MetricsHttpServerConfig {
            bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 9090),
            allow_remote_bind: false,
        };
        match cfg.validate() {
            Err(MetricsHttpError::BindAddrForbidden { bind }) => {
                assert_eq!(bind, cfg.bind);
            }
            other => panic!("expected BindAddrForbidden, got {other:?}"),
        }
    }

    #[test]
    fn non_loopback_bind_with_opt_in_passes() {
        let cfg = MetricsHttpServerConfig {
            bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 9090),
            allow_remote_bind: true,
        };
        cfg.validate().expect("0.0.0.0 with opt-in must validate");
    }

    #[test]
    fn ipv4_and_ipv6_loopback_bind_pass_without_opt_in() {
        let v4 = MetricsHttpServerConfig {
            bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9090),
            allow_remote_bind: false,
        };
        v4.validate().expect("v4 loopback OK");
        let v6 = MetricsHttpServerConfig {
            bind: SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 9090),
            allow_remote_bind: false,
        };
        v6.validate().expect("v6 loopback OK");
    }

    #[test]
    fn deserialize_rejects_unknown_fields() {
        // Strict mode: misspellings reject at startup.
        let json = r#"{ "bind": "127.0.0.1:9090", "allow_remote_bnd": true }"#;
        let err =
            serde_json::from_str::<MetricsHttpServerConfig>(json).expect_err("unknown rejects");
        assert!(
            err.to_string().contains("unknown field"),
            "deny_unknown_fields must fire: {err}"
        );
    }

    // ── Real-socket scrape helpers (mirror admin_http; no client dep) ─

    async fn spawn_test_server(
        registry: Arc<MetricsRegistry>,
    ) -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .expect("bind ephemeral");
        let bound = listener.local_addr().expect("local_addr");
        let router = build_router(registry);
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        tokio::spawn(async move {
            let _ = axum::serve(listener, router.into_make_service())
                .with_graceful_shutdown(shutdown)
                .await;
        });
        for _ in 0..50 {
            if tokio::net::TcpStream::connect(bound).await.is_ok() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        (bound, tx)
    }

    async fn http_get(addr: SocketAddr, path: &str) -> (u16, String) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let req = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
        stream.write_all(req.as_bytes()).await.expect("write");
        stream.flush().await.expect("flush");
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await.expect("read");
        let text = String::from_utf8_lossy(&buf).to_string();
        let status = text
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);
        let body_start = text.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
        (status, text[body_start..].to_string())
    }

    /// STRONG ORACLE — the load-bearing test. A registry with a RECORDED
    /// observation must scrape with that observation's VALUE present in
    /// the Prometheus body, with the Prometheus content-type. This is
    /// the assertion that fails if `/metrics` is wired to a registry
    /// that sees no data — the exact "registry wired to nothing" trap
    /// the spawn prompt + `feedback_review_oracle_relaxations.md` warn
    /// against. We assert the VALUE moved, NOT merely that GET returns
    /// 200.
    #[tokio::test]
    async fn metrics_endpoint_serves_recorded_values_not_just_200() {
        let registry = MetricsRegistry::shared().expect("registry");
        // Record a deterministic write dispatch on tenant 42.
        registry.record_dispatch(
            TenantId::new(42),
            "graph.ingest",
            OpClass::Write,
            ToolInvocationStatus::Ok,
            3.0,
        );
        let (addr, _shutdown) = spawn_test_server(registry).await;

        let (status, body) = http_get(addr, "/metrics").await;
        assert_eq!(status, 200, "GET /metrics must be 200; body:\n{body}");
        // VALUE oracle: the exact counter cell must be present at 1.
        assert!(
            body.contains(
                r#"arcgraph_mcp_tool_invocations{status="ok",tenant="42",tool="graph.ingest"} 1"#
            ),
            "scrape must carry the recorded counter VALUE (=1), not an empty registry; body:\n{body}"
        );
        // Write-class observation routed to the write-latency histogram.
        assert!(
            body.contains(r#"arcgraph_write_latency_ms_count{tenant="42",tool="graph.ingest"} 1"#),
            "scrape must carry the write-latency histogram count; body:\n{body}"
        );
        // Prometheus text-exposition framing (HELP/TYPE comment lines).
        assert!(
            body.contains("# TYPE arcgraph_mcp_tool_invocations counter"),
            "scrape must be Prometheus text-exposition format; body:\n{body}"
        );
    }

    /// Content-type carries the Prometheus version qualifier so the
    /// scraper can branch on protocol version (drift here silently
    /// breaks strict scrapers).
    #[tokio::test]
    async fn metrics_endpoint_emits_prometheus_content_type() {
        let registry = MetricsRegistry::shared().expect("registry");
        let (addr, _shutdown) = spawn_test_server(registry).await;
        // Read the full raw response (headers included).
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
        stream
            .write_all(b"GET /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .await
            .expect("write");
        stream.flush().await.expect("flush");
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await.expect("read");
        let raw = String::from_utf8_lossy(&buf).to_lowercase();
        assert!(
            raw.contains("content-type: text/plain; version=0.0.4"),
            "must emit the Prometheus text content-type; raw head:\n{}",
            &raw[..raw.len().min(400)]
        );
    }

    /// FAULT INJECTION — bind conflict (`EADDRINUSE`) degrades
    /// gracefully: `serve_metrics_http` returns a recoverable
    /// [`MetricsHttpError::Bind`] (it does NOT panic, does NOT hang).
    /// `bin/arcgraph.rs::run_serve` logs this and keeps the main MCP
    /// server running — observability degradation must not cascade into
    /// unavailability (ADR-093 §Decision item 2). This is the
    /// load-bearing failure-mode regression test for this surface.
    #[tokio::test]
    async fn bind_conflict_surfaces_recoverable_error_no_panic() {
        // Occupy an ephemeral loopback port.
        let squatter = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .expect("squatter bind");
        let occupied = squatter.local_addr().expect("addr");

        let registry = MetricsRegistry::shared().expect("registry");
        let cfg = MetricsHttpServerConfig {
            bind: occupied,
            allow_remote_bind: false,
        };
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        let result = serve_metrics_http(cfg, registry, shutdown).await;
        match result {
            Err(MetricsHttpError::Bind { bind, .. }) => assert_eq!(bind, occupied),
            other => panic!("bind conflict must surface as recoverable Bind error, got {other:?}"),
        }
    }

    /// FAULT INJECTION — the LOUD path: a non-loopback bind without the
    /// opt-in is refused BEFORE any socket is bound, so the operator
    /// gets an immediate actionable config error (no silent half-open
    /// metrics surface). `serve_metrics_http` returns `BindAddrForbidden`
    /// without touching the network.
    #[tokio::test]
    async fn non_loopback_without_opt_in_refuses_to_bind() {
        let registry = MetricsRegistry::shared().expect("registry");
        let cfg = MetricsHttpServerConfig {
            bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 9090),
            allow_remote_bind: false,
        };
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        let result = serve_metrics_http(cfg, registry, shutdown).await;
        assert!(
            matches!(result, Err(MetricsHttpError::BindAddrForbidden { .. })),
            "non-loopback without opt-in must refuse to bind, got {result:?}"
        );
    }
}
