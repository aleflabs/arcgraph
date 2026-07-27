//! W24-OPS-α — axum-based admin HTTP server exposing `/livez` + `/readyz`.
//!
//! # Endpoints (v1.0-GA contract per ADR-093 §Decision)
//!
//! - `GET /livez` — always returns `200 OK` with body `{"status":"alive"}`
//!   as long as the process is alive enough to accept the request.
//!   The Kubernetes liveness-probe consumer: a non-200 here means the
//!   process is wedged and the kubelet restarts the pod.
//!
//! - `GET /readyz` — returns `200 OK` with body `{"status":"ready"}`
//!   when ALL registered components are in [`ComponentState::Ready`];
//!   `503 Service Unavailable` with body
//!   `{"status":"not_ready","components":{...}}` listing each
//!   component's state otherwise. The Kubernetes readiness-probe
//!   consumer: a 503 here means the pod is alive but should NOT
//!   receive traffic yet (startup-incomplete or transient component
//!   failure during graceful drain).
//!
//! # Why `/livez` + `/readyz` distinction (vs. the MCP `/healthz`)
//!
//! The MCP HTTP transport's `/healthz` (per
//! `crates/arcgraph-mcp/src/transport/http.rs`) is a TLS-handshake-
//! success probe: it returns 200 as soon as the server can accept a
//! request. That conflates "process alive" with "ready to serve
//! traffic." Production deployments need the distinction because:
//!
//! - At cold-start, the process is alive (passes `/livez`) but WAL
//!   replay hasn't completed (fails `/readyz`).
//! - During graceful drain, the process is alive (passes `/livez`)
//!   but new traffic should be steered to other replicas
//!   (fails `/readyz`) so in-flight queries complete cleanly.
//! - During transient backend hiccups (storage backend slow, index
//!   rebuild in progress), the process is alive (passes `/livez`)
//!   but should drop out of the LB target pool (fails `/readyz`)
//!   until the backend recovers.
//!
//! Kubernetes documents this distinction at
//! <https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/>
//! and the operator-facing v1.0-GA Kubernetes manifests in
//! `deploy/k8s/statefulset.yaml` wire to these two distinct endpoints.
//!
//! # Loopback-default bind (W14 retro IR L1-HIGH-4)
//!
//! [`AdminHttpServerConfig::validate`] rejects any non-loopback bind
//! unless `allow_remote_bind` is `true`. The operator-opt-in flag
//! (`--allow-remote-admin-bind`) is already wired (W24-OPS-α).
//! Loopback-default applies to admin endpoints just as it does to the
//! Bolt transport (per `BoltServerConfig::validate`) — admin endpoints
//! reveal startup status + future cert-rotation state, both of which
//! are sensitive enough to deserve the default-private posture.
//!
//! # Why axum (and not hyper-direct like the MCP transport)
//!
//! The MCP HTTP transport at `crates/arcgraph-mcp/src/transport/http.rs`
//! deliberately chose hyper-direct because its surface is exactly one
//! POST `/mcp` + one GET `/healthz` and the per-connection TLS
//! handshake + CancellationToken plumbing benefits from explicit
//! control over the accept loop. The admin HTTP server is the
//! opposite case: small surface today (2 endpoints), broad extension
//! shape over the v1.0-GA → v1.1 → v1.2 trajectory (cert rotation
//! trigger; config reload; future per-tenant admin verbs). axum's
//! typed router + tower middleware composition is the right fit for
//! that extension shape; the per-PR comparison is documented in the
//! ADR-093 §Alternatives table.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use arcgraph_core::cost_telemetry::{CostSnapshot, PerTenantCostRegistry};
use arcgraph_core::ids::TenantId;
use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use parking_lot::RwLock;
use serde::Serialize;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;

/// State of a single readiness-tracked component.
///
/// `Initializing` is the cold-start state; the component moves to
/// `Ready` once its bootstrap completes successfully, and to
/// `Failed` if bootstrap fails (with a short reason string for the
/// /readyz JSON body). `Draining` is reserved for v1.1's graceful
/// shutdown surface — at v1.0-GA it is unused but pre-bound to keep
/// the JSON wire shape forward-compatible.
///
/// `#[non_exhaustive]` under the strict public-contract policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[non_exhaustive]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ComponentState {
    /// Bootstrap in flight; /readyz returns 503 if any component
    /// is in this state.
    Initializing,
    /// Component is ready to serve traffic.
    Ready,
    /// Component is shutting down (graceful drain). v1.1+ forward-
    /// pin: drops the component out of /readyz=200 calculation so
    /// the pod leaves the LB target pool before connections terminate.
    Draining,
    /// Component bootstrap failed; /readyz returns 503 and the
    /// JSON body includes the reason.
    Failed {
        /// Operator-readable reason string (one sentence, no PII).
        reason: String,
    },
}

impl ComponentState {
    /// Returns `true` iff this state contributes to /readyz=200.
    /// Only [`ComponentState::Ready`] qualifies; every other state
    /// (including [`ComponentState::Draining`]) takes /readyz to 503.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(self, ComponentState::Ready)
    }
}

/// Snapshot of every registered component's state. Returned by the
/// /readyz handler in the JSON body when /readyz=503.
#[derive(Debug, Clone, Serialize)]
pub struct ReadinessGateSnapshot {
    /// Map from component name (e.g. `"storage"`, `"wal"`, `"index"`)
    /// to its current state.
    pub components: HashMap<String, ComponentState>,
}

impl ReadinessGateSnapshot {
    /// Returns `true` iff every component is ready.
    #[must_use]
    pub fn all_ready(&self) -> bool {
        !self.components.is_empty() && self.components.values().all(ComponentState::is_ready)
    }
}

/// Process-global readiness gate. Components register at startup
/// (default state `Initializing`) and flip to `Ready` once their
/// bootstrap completes. `/readyz` reads this gate.
///
/// The gate is `Send + Sync + Clone` — every clone shares the same
/// underlying `Arc<RwLock<...>>` so multiple wiring sites (the
/// storage bootstrap, the WAL replay loop, the index loader) can
/// hold their own clone and update state in parallel.
///
/// # Component registration contract
///
/// At v1.0-GA the production wiring registers three components:
/// `storage`, `wal`, `index`. New components added at v1.1+ must
/// register at process start (NOT lazily) so a /readyz probe before
/// any component reports state cannot accidentally succeed.
#[derive(Debug, Clone, Default)]
pub struct ReadinessGate {
    inner: Arc<RwLock<HashMap<String, ComponentState>>>,
}

impl ReadinessGate {
    /// Construct an empty gate. /readyz will return 503 with
    /// `{"status":"not_ready","components":{}}` until at least one
    /// component registers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a component with the gate; initial state is
    /// [`ComponentState::Initializing`]. Idempotent: re-registering
    /// an existing component overwrites its state.
    pub fn register(&self, component: impl Into<String>) {
        self.inner
            .write()
            .insert(component.into(), ComponentState::Initializing);
    }

    /// Mark a component as [`ComponentState::Ready`]. Returns `true`
    /// if the component was previously registered (typical), `false`
    /// if this is the first time the component is being mentioned
    /// (operationally still OK — the gate auto-registers — but a
    /// hint of a bootstrap-order bug worth investigating).
    pub fn mark_ready(&self, component: impl Into<String>) -> bool {
        let name = component.into();
        let mut map = self.inner.write();
        let existed = map.contains_key(&name);
        map.insert(name, ComponentState::Ready);
        existed
    }

    /// Mark a component as [`ComponentState::Failed`] with the given
    /// reason. Returns the same boolean as [`Self::mark_ready`].
    pub fn mark_failed(&self, component: impl Into<String>, reason: impl Into<String>) -> bool {
        let name = component.into();
        let mut map = self.inner.write();
        let existed = map.contains_key(&name);
        map.insert(
            name,
            ComponentState::Failed {
                reason: reason.into(),
            },
        );
        existed
    }

    /// Mark a component as [`ComponentState::Draining`] (v1.1+
    /// graceful-shutdown surface).
    pub fn mark_draining(&self, component: impl Into<String>) -> bool {
        let name = component.into();
        let mut map = self.inner.write();
        let existed = map.contains_key(&name);
        map.insert(name, ComponentState::Draining);
        existed
    }

    /// Snapshot the entire gate.
    #[must_use]
    pub fn snapshot(&self) -> ReadinessGateSnapshot {
        ReadinessGateSnapshot {
            components: self.inner.read().clone(),
        }
    }
}

/// Configuration for the admin HTTP server.
///
/// `#[serde(deny_unknown_fields)]` ensures misspelled config keys reject at startup
/// rather than silently degrading. The struct is forward-pinned for
/// the M6-08+ TOML config schema landing.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdminHttpServerConfig {
    /// Bind address. Default: `127.0.0.1:8090`.
    pub bind: SocketAddr,
    /// Whether to permit binding to a non-loopback address.
    /// Loopback-default per W14 retro IR L1-HIGH-4. The operator
    /// opt-in (`--allow-remote-admin-bind`) is already wired (W24-OPS-α).
    #[serde(default)]
    pub allow_remote_bind: bool,
}

impl Default for AdminHttpServerConfig {
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], 8090)),
            allow_remote_bind: false,
        }
    }
}

/// Errors emitted by the admin HTTP server.
///
/// `#[non_exhaustive]` under the strict public-contract policy.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AdminHttpError {
    /// Bind address rejected (non-loopback without `allow_remote_bind`).
    #[error(
        "admin HTTP server refused to bind {bind}: non-loopback without `allow_remote_bind` \
         (W14 retro IR L1-HIGH-4 loopback-default; pass `--allow-remote-admin-bind` to opt in)"
    )]
    BindAddrForbidden { bind: SocketAddr },
    /// TCP listener bind failed.
    #[error("admin HTTP server bind {bind} failed: {source}")]
    Bind {
        bind: SocketAddr,
        #[source]
        source: std::io::Error,
    },
    /// axum::serve returned with an error.
    #[error("admin HTTP server serve loop returned an error: {source}")]
    Serve {
        #[source]
        source: std::io::Error,
    },
}

impl AdminHttpServerConfig {
    /// Validate the loopback-default policy. Returns
    /// [`AdminHttpError::BindAddrForbidden`] if `bind` is non-loopback
    /// and `allow_remote_bind` is false.
    pub fn validate(&self) -> Result<(), AdminHttpError> {
        if self.allow_remote_bind {
            return Ok(());
        }
        let ip = self.bind.ip();
        if ip.is_loopback() {
            return Ok(());
        }
        Err(AdminHttpError::BindAddrForbidden { bind: self.bind })
    }
}

/// Server state shared with axum handlers.
///
/// `cost` is optional: deployments that do NOT wire a registry still
/// expose `/livez` + `/readyz`; the cost endpoints return 503 in that
/// case (the registry is the "component" for cost telemetry).
#[derive(Clone)]
struct AdminState {
    gate: ReadinessGate,
    cost: Option<PerTenantCostRegistry>,
}

/// `GET /livez` handler — always returns 200.
async fn livez_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status": "alive"})))
}

/// `GET /readyz` handler — 200 if all components ready, 503 otherwise.
async fn readyz_handler(State(state): State<AdminState>) -> impl IntoResponse {
    let snap = state.gate.snapshot();
    if snap.all_ready() {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ready",
                "components": snap.components,
            })),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "not_ready",
                "components": snap.components,
            })),
        )
    }
}

/// Wire shape for a single tenant's cost line in the `/cost` JSON
/// response. Field names are part of the v1.0-GA operator surface per
/// ADR-093-amendment-01 §D-6 — do NOT rename without an amendment.
#[derive(Debug, Serialize)]
struct TenantCostLine {
    tenant_id: u64,
    cpu_ms: u64,
    mem_mb_peak: u64,
    bytes_read: u64,
    bytes_written: u64,
}

impl TenantCostLine {
    fn from_snapshot(tenant: TenantId, snap: CostSnapshot) -> Self {
        Self {
            tenant_id: tenant.raw(),
            cpu_ms: snap.cpu_ms,
            mem_mb_peak: snap.mem_mb_peak,
            bytes_read: snap.bytes_read,
            bytes_written: snap.bytes_written,
        }
    }
}

/// `GET /cost` handler — all-tenant cost snapshot. Returns 503 if no
/// cost registry is wired into the admin server (a deployment that
/// scopes admin endpoints to liveness/readiness only).
async fn cost_all_handler(State(state): State<AdminState>) -> impl IntoResponse {
    let Some(registry) = state.cost.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "cost_telemetry_not_wired",
                "reason": "admin server constructed without a per-tenant cost registry",
            })),
        );
    };
    let snaps = registry.snapshot_all();
    let mut lines: Vec<TenantCostLine> = snaps
        .into_iter()
        .map(|(tenant, snap)| TenantCostLine::from_snapshot(tenant, snap))
        .collect();
    // Deterministic order for operator-readable output + reproducible
    // test assertions; sort by tenant_id ascending.
    lines.sort_by_key(|line| line.tenant_id);
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "tenants": lines,
            "tenant_count": lines.len(),
        })),
    )
}

/// `GET /cost/:tenant_id` handler — single-tenant cost snapshot.
/// Returns 404 if the tenant has no recorded activity; 503 if no
/// cost registry is wired.
async fn cost_single_handler(
    State(state): State<AdminState>,
    Path(tenant_id_raw): Path<u64>,
) -> impl IntoResponse {
    let Some(registry) = state.cost.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "cost_telemetry_not_wired",
            })),
        );
    };
    let tenant = TenantId::new(tenant_id_raw);
    match registry.snapshot(tenant) {
        Some(snap) => (
            StatusCode::OK,
            Json(serde_json::json!(TenantCostLine::from_snapshot(
                tenant, snap
            ))),
        ),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "status": "no_recorded_activity",
                "tenant_id": tenant_id_raw,
            })),
        ),
    }
}

/// Build the admin HTTP router with /livez + /readyz only (no cost
/// telemetry endpoint surface). Pre-W25-OPS-PROD entrypoint preserved
/// for callers that don't wire a cost registry.
pub fn build_router(gate: ReadinessGate) -> axum::Router {
    build_router_with_cost(gate, None)
}

/// Build the admin HTTP router with optional cost-telemetry surface.
/// When `cost` is `Some`, `/cost` + `/cost/:tenant_id` are wired;
/// otherwise those routes return 503.
///
/// Public for tests that want to exercise the router via
/// `axum::http::Request` without spinning a TCP socket.
pub fn build_router_with_cost(
    gate: ReadinessGate,
    cost: Option<PerTenantCostRegistry>,
) -> axum::Router {
    let state = AdminState { gate, cost };
    axum::Router::new()
        .route("/livez", get(livez_handler))
        .route("/readyz", get(readyz_handler))
        .route("/cost", get(cost_all_handler))
        .route("/cost/{tenant_id}", get(cost_single_handler))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
}

/// Bind + serve the admin HTTP router. Returns when `shutdown` resolves
/// (typically wired to the same SIGTERM-aware future the MCP transport
/// uses).
///
/// # Errors
///
/// - [`AdminHttpError::BindAddrForbidden`] — non-loopback bind without
///   `allow_remote_bind`.
/// - [`AdminHttpError::Bind`] — TCP listener bind syscall failed.
/// - [`AdminHttpError::Serve`] — axum serve loop returned an error.
pub async fn serve_admin_http<F>(
    config: AdminHttpServerConfig,
    gate: ReadinessGate,
    shutdown: F,
) -> Result<(), AdminHttpError>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    serve_admin_http_with_cost(config, gate, None, shutdown).await
}

/// W25-OPS-PROD: bind + serve the admin HTTP router with optional
/// per-tenant cost-telemetry surface. Convenience wrapper that
/// preserves the W24-OPS-α `serve_admin_http` entrypoint for callers
/// that don't wire cost.
pub async fn serve_admin_http_with_cost<F>(
    config: AdminHttpServerConfig,
    gate: ReadinessGate,
    cost: Option<PerTenantCostRegistry>,
    shutdown: F,
) -> Result<(), AdminHttpError>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    config.validate()?;
    let router = build_router_with_cost(gate, cost);
    let listener = TcpListener::bind(config.bind)
        .await
        .map_err(|source| AdminHttpError::Bind {
            bind: config.bind,
            source,
        })?;
    tracing::info!(
        target: "arcgraph_cli::ops::admin_http",
        bind = %config.bind,
        "admin HTTP server listening on /livez + /readyz",
    );
    axum::serve(listener, router.into_make_service())
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|source| AdminHttpError::Serve { source })?;
    tracing::info!(
        target: "arcgraph_cli::ops::admin_http",
        bind = %config.bind,
        "admin HTTP server exiting cleanly",
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Tests — readiness gate semantics + bind policy + router shape.
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    // ── ReadinessGate state machine ───────────────────────────────

    #[test]
    fn empty_gate_is_not_ready() {
        let gate = ReadinessGate::new();
        let snap = gate.snapshot();
        assert!(!snap.all_ready());
        assert!(snap.components.is_empty());
    }

    #[test]
    fn register_then_mark_ready_flips_to_ready() {
        let gate = ReadinessGate::new();
        gate.register("storage");
        let snap = gate.snapshot();
        assert!(!snap.all_ready());
        assert_eq!(
            snap.components.get("storage"),
            Some(&ComponentState::Initializing)
        );
        assert!(gate.mark_ready("storage"));
        let snap = gate.snapshot();
        assert!(snap.all_ready());
    }

    #[test]
    fn mark_failed_keeps_gate_not_ready_with_reason() {
        let gate = ReadinessGate::new();
        gate.register("storage");
        gate.mark_failed("storage", "buffer pool init: ENOSPC");
        let snap = gate.snapshot();
        assert!(!snap.all_ready());
        match snap.components.get("storage").expect("present") {
            ComponentState::Failed { reason } => {
                assert!(reason.contains("ENOSPC"));
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn draining_drops_out_of_ready_calculation() {
        let gate = ReadinessGate::new();
        gate.register("storage");
        gate.register("wal");
        gate.mark_ready("storage");
        gate.mark_ready("wal");
        assert!(gate.snapshot().all_ready());
        gate.mark_draining("wal");
        let snap = gate.snapshot();
        assert!(!snap.all_ready(), "draining must drop from /readyz=200");
    }

    #[test]
    fn mark_ready_first_time_returns_false_but_still_promotes() {
        // Auto-register on mark_ready is operationally OK (no
        // bootstrap-order bug if components mark themselves), but
        // signals back to the caller via the bool return.
        let gate = ReadinessGate::new();
        assert!(!gate.mark_ready("storage"), "auto-register signals false");
        assert!(gate.snapshot().all_ready());
    }

    #[test]
    fn all_ready_requires_at_least_one_component() {
        let gate = ReadinessGate::new();
        // Empty gate is NOT ready — protects against a /readyz probe
        // accidentally succeeding before any component reports state.
        assert!(!gate.snapshot().all_ready());
    }

    // ── ComponentState helpers ────────────────────────────────────

    #[test]
    fn component_state_is_ready_only_for_ready_variant() {
        assert!(ComponentState::Ready.is_ready());
        assert!(!ComponentState::Initializing.is_ready());
        assert!(!ComponentState::Draining.is_ready());
        assert!(!ComponentState::Failed { reason: "x".into() }.is_ready());
    }

    // ── AdminHttpServerConfig validate (loopback-default) ─────────

    #[test]
    fn default_config_binds_loopback_8090_and_validates() {
        let cfg = AdminHttpServerConfig::default();
        assert_eq!(cfg.bind.ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(cfg.bind.port(), 8090);
        assert!(!cfg.allow_remote_bind);
        cfg.validate().expect("default loopback OK");
    }

    #[test]
    fn non_loopback_bind_without_opt_in_rejects() {
        let cfg = AdminHttpServerConfig {
            bind: SocketAddr::from(([0, 0, 0, 0], 8090)),
            allow_remote_bind: false,
        };
        match cfg.validate() {
            Err(AdminHttpError::BindAddrForbidden { bind }) => {
                assert_eq!(bind, cfg.bind);
            }
            other => panic!("expected BindAddrForbidden, got {other:?}"),
        }
    }

    #[test]
    fn non_loopback_bind_with_opt_in_passes() {
        let cfg = AdminHttpServerConfig {
            bind: SocketAddr::from(([0, 0, 0, 0], 8090)),
            allow_remote_bind: true,
        };
        cfg.validate().expect("opt-in OK");
    }

    #[test]
    fn ipv6_loopback_bind_passes() {
        let cfg = AdminHttpServerConfig {
            bind: "[::1]:8090".parse().expect("parse"),
            allow_remote_bind: false,
        };
        cfg.validate().expect("v6 loopback OK");
    }

    // ── Deserialize strict-mode (deny_unknown_fields) ─────────────

    #[test]
    fn deserialize_rejects_unknown_fields() {
        // Strict mode: misspellings reject at startup.
        let json = r#"{ "bind": "127.0.0.1:8090", "allow_remote_bnd": true }"#;
        let err = serde_json::from_str::<AdminHttpServerConfig>(json).expect_err("unknown rejects");
        assert!(
            err.to_string().contains("unknown field"),
            "deny_unknown_fields fired: {err}"
        );
    }

    // ── Router smoke — /livez + /readyz over a real TCP socket ───

    /// Spawn the admin server on an ephemeral loopback port. Returns
    /// the bound `SocketAddr` + a shutdown trigger.
    async fn spawn_test_server(
        gate: ReadinessGate,
    ) -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
        let cfg = AdminHttpServerConfig {
            bind: SocketAddr::from(([127, 0, 0, 1], 0)),
            allow_remote_bind: false,
        };
        cfg.validate().expect("validate");
        let listener = TcpListener::bind(cfg.bind).await.expect("bind");
        let bound = listener.local_addr().expect("local_addr");
        let router = build_router(gate);
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        tokio::spawn(async move {
            let _ = axum::serve(listener, router.into_make_service())
                .with_graceful_shutdown(shutdown)
                .await;
        });
        // Wait for the server to be ready by polling /livez.
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
        let req = format!(
            "GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n",
            addr = addr,
            path = path,
        );
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

    #[tokio::test]
    async fn livez_returns_200_with_alive_body() {
        let gate = ReadinessGate::new();
        let (addr, _shutdown) = spawn_test_server(gate).await;
        let (status, body) = http_get(addr, "/livez").await;
        assert_eq!(status, 200, "/livez always 200; body: {body}");
        assert!(body.contains("alive"), "alive body: {body}");
    }

    #[tokio::test]
    async fn readyz_returns_503_when_no_components_ready() {
        let gate = ReadinessGate::new();
        let (addr, _shutdown) = spawn_test_server(gate).await;
        let (status, body) = http_get(addr, "/readyz").await;
        assert_eq!(status, 503, "no components → 503; body: {body}");
        assert!(body.contains("not_ready"), "body: {body}");
    }

    #[tokio::test]
    async fn readyz_returns_200_when_all_components_ready() {
        let gate = ReadinessGate::new();
        gate.register("storage");
        gate.register("wal");
        gate.register("index");
        gate.mark_ready("storage");
        gate.mark_ready("wal");
        gate.mark_ready("index");
        let (addr, _shutdown) = spawn_test_server(gate).await;
        let (status, body) = http_get(addr, "/readyz").await;
        assert_eq!(status, 200, "all ready → 200; body: {body}");
        assert!(body.contains("\"status\":\"ready\""), "body: {body}");
    }

    #[tokio::test]
    async fn readyz_returns_503_with_reason_when_one_component_failed() {
        // Fault-injection regression test per
        // `feedback_load_bearing_pr_requires_fault_injection_tests.md`:
        // a Failed component must surface 503 + the reason string in
        // the body so an operator's kubectl-describe-pod debug loop
        // gets actionable failure info.
        let gate = ReadinessGate::new();
        gate.register("storage");
        gate.register("wal");
        gate.mark_ready("storage");
        gate.mark_failed("wal", "fsync EIO on segment 042");
        let (addr, _shutdown) = spawn_test_server(gate).await;
        let (status, body) = http_get(addr, "/readyz").await;
        assert_eq!(status, 503, "Failed component → 503");
        assert!(body.contains("not_ready"), "status field: {body}");
        assert!(body.contains("EIO"), "reason in body: {body}");
        assert!(body.contains("042"), "reason in body: {body}");
    }

    #[tokio::test]
    async fn readyz_flips_503_to_200_when_late_component_reports_ready() {
        // Fault-injection regression test: cold-start with WAL replay
        // takes seconds; /readyz should reflect the late mark_ready
        // without any state-machine staleness.
        let gate = ReadinessGate::new();
        gate.register("storage");
        gate.register("wal");
        gate.mark_ready("storage");
        let (addr, shutdown) = spawn_test_server(gate.clone()).await;
        let (status, _) = http_get(addr, "/readyz").await;
        assert_eq!(status, 503, "wal still Initializing → 503");
        gate.mark_ready("wal");
        let (status, body) = http_get(addr, "/readyz").await;
        assert_eq!(status, 200, "late mark_ready promotes to 200; body: {body}");
        let _ = shutdown.send(());
    }

    #[tokio::test]
    async fn readyz_returns_503_when_one_component_draining() {
        // Fault-injection: graceful shutdown wires mark_draining; the
        // /readyz handler must drop the pod from the LB target pool
        // BEFORE in-flight connections terminate.
        let gate = ReadinessGate::new();
        gate.register("storage");
        gate.mark_ready("storage");
        let (addr, _shutdown) = spawn_test_server(gate.clone()).await;
        let (status, _) = http_get(addr, "/readyz").await;
        assert_eq!(status, 200);
        gate.mark_draining("storage");
        let (status, body) = http_get(addr, "/readyz").await;
        assert_eq!(status, 503, "draining drops /readyz; body: {body}");
        assert!(body.contains("draining"), "draining state in body: {body}");
    }

    // ── W25-OPS-PROD: cost-endpoint surface ───────────────────────

    /// Spawn the admin server with cost telemetry wired.
    async fn spawn_test_server_with_cost(
        gate: ReadinessGate,
        cost: PerTenantCostRegistry,
    ) -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
        let cfg = AdminHttpServerConfig {
            bind: SocketAddr::from(([127, 0, 0, 1], 0)),
            allow_remote_bind: false,
        };
        cfg.validate().expect("validate");
        let listener = TcpListener::bind(cfg.bind).await.expect("bind");
        let bound = listener.local_addr().expect("local_addr");
        let router = build_router_with_cost(gate, Some(cost));
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

    #[tokio::test]
    async fn cost_all_returns_503_when_no_registry_wired() {
        let gate = ReadinessGate::new();
        gate.register("storage");
        gate.mark_ready("storage");
        let (addr, _shutdown) = spawn_test_server(gate).await;
        let (status, body) = http_get(addr, "/cost").await;
        assert_eq!(status, 503, "no registry → 503; body: {body}");
        assert!(body.contains("cost_telemetry_not_wired"));
    }

    #[tokio::test]
    async fn cost_all_returns_200_with_empty_list_when_no_tenants_recorded() {
        let gate = ReadinessGate::new();
        gate.register("storage");
        gate.mark_ready("storage");
        let registry = PerTenantCostRegistry::new();
        let (addr, _shutdown) = spawn_test_server_with_cost(gate, registry).await;
        let (status, body) = http_get(addr, "/cost").await;
        assert_eq!(status, 200);
        assert!(body.contains("\"tenant_count\":0"), "body: {body}");
        assert!(body.contains("\"tenants\":[]"), "body: {body}");
    }

    #[tokio::test]
    async fn cost_all_returns_recorded_tenants_in_sorted_order() {
        let gate = ReadinessGate::new();
        gate.register("storage");
        gate.mark_ready("storage");
        let registry = PerTenantCostRegistry::new();
        registry.get_or_init(TenantId::new(2)).record_cpu_ms(200);
        registry.get_or_init(TenantId::new(1)).record_cpu_ms(100);
        let (addr, _shutdown) = spawn_test_server_with_cost(gate, registry).await;
        let (status, body) = http_get(addr, "/cost").await;
        assert_eq!(status, 200);
        // Tenant 1 should appear before tenant 2 (sorted by tenant_id).
        let idx_t1 = body.find("\"tenant_id\":1").expect("t1");
        let idx_t2 = body.find("\"tenant_id\":2").expect("t2");
        assert!(idx_t1 < idx_t2, "deterministic order; body: {body}");
        assert!(body.contains("\"tenant_count\":2"));
    }

    #[tokio::test]
    async fn cost_single_returns_404_when_tenant_unknown() {
        let gate = ReadinessGate::new();
        gate.register("storage");
        gate.mark_ready("storage");
        let registry = PerTenantCostRegistry::new();
        let (addr, _shutdown) = spawn_test_server_with_cost(gate, registry).await;
        let (status, body) = http_get(addr, "/cost/999").await;
        assert_eq!(status, 404, "unknown tenant → 404; body: {body}");
        assert!(body.contains("no_recorded_activity"));
    }

    #[tokio::test]
    async fn cost_single_returns_200_with_recorded_observations() {
        let gate = ReadinessGate::new();
        gate.register("storage");
        gate.mark_ready("storage");
        let registry = PerTenantCostRegistry::new();
        let acc = registry.get_or_init(TenantId::new(42));
        acc.record_cpu_ms(1500);
        acc.observe_mem_mb(256);
        acc.record_bytes_read(1024 * 1024);
        acc.record_bytes_written(512 * 1024);
        let (addr, _shutdown) = spawn_test_server_with_cost(gate, registry).await;
        let (status, body) = http_get(addr, "/cost/42").await;
        assert_eq!(status, 200, "body: {body}");
        assert!(body.contains("\"tenant_id\":42"));
        assert!(body.contains("\"cpu_ms\":1500"));
        assert!(body.contains("\"mem_mb_peak\":256"));
        assert!(body.contains("\"bytes_read\":1048576"));
        assert!(body.contains("\"bytes_written\":524288"));
    }

    #[tokio::test]
    async fn cost_single_returns_503_when_no_registry_wired() {
        let gate = ReadinessGate::new();
        gate.register("storage");
        gate.mark_ready("storage");
        let (addr, _shutdown) = spawn_test_server(gate).await;
        let (status, body) = http_get(addr, "/cost/1").await;
        assert_eq!(status, 503);
        assert!(body.contains("cost_telemetry_not_wired"));
    }

    #[tokio::test]
    async fn cost_endpoints_do_not_break_existing_livez_readyz() {
        // Regression: the cost endpoint additions must not perturb
        // the W24-OPS-α /livez + /readyz behavior on a deployment
        // that DOES wire the cost registry.
        let gate = ReadinessGate::new();
        gate.register("storage");
        gate.mark_ready("storage");
        let registry = PerTenantCostRegistry::new();
        let (addr, _shutdown) = spawn_test_server_with_cost(gate, registry).await;
        let (livez_status, livez_body) = http_get(addr, "/livez").await;
        assert_eq!(livez_status, 200);
        assert!(livez_body.contains("alive"));
        let (readyz_status, readyz_body) = http_get(addr, "/readyz").await;
        assert_eq!(readyz_status, 200);
        assert!(readyz_body.contains("ready"));
    }
}
