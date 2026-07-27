//! W24-OPS-α — Operations hardening surface for v1.0-GA.
//!
//! This module hosts the operator-facing observability + administration
//! primitives required by the v1.0-GA deployment contract:
//!
//! - [`admin_http`] — axum-based admin HTTP server exposing
//!   `/livez` (process-alive only, always 200) and `/readyz`
//!   (storage + WAL + index loaded → 200; startup-incomplete → 503
//!   with a JSON body listing not-ready components). Loopback-default
//!   bind per W14 retro IR L1-HIGH-4; the operator-opt-in flag for
//!   non-loopback binds (`--allow-remote-admin-bind`) is already wired
//!   (W24-OPS-α), piped through to
//!   `AdminHttpServerConfig::allow_remote_bind`.
//! - [`tracing_init`] — `tracing_subscriber::registry` init with an
//!   `env-filter` layer (`RUST_LOG` honored) + `fmt` layer
//!   (structured stderr lines) + optional OTLP-gRPC export layer
//!   gated by `ARCGRAPH_OTLP_ENDPOINT`.
//!
//! These primitives compose into the `arcgraph serve` binary's
//! initialization sequence (see `bin/arcgraph.rs::main`):
//!
//! ```text
//!   1. tracing_init::init(...)        ← stderr + optional OTLP
//!   2. ReadinessGate::new()            ← components track Ready-ness
//!   3. bootstrap_storage_backend()     ← mark "storage" ready on Ok
//!   4. admin_http::serve(gate, ...)    ← bind /livez + /readyz
//!   5. serve_stdio() / serve_bolt()    ← the production MCP loop
//! ```
//!
//! The `admin_http` server runs on a separate port from the MCP
//! transport (default `127.0.0.1:8090`) so an operator can probe
//! readiness independent of which MCP transport (stdio / HTTP / Bolt)
//! the deployment is using. The MCP HTTP transport's `/healthz`
//! endpoint (per `crates/arcgraph-mcp/src/transport/http.rs`) is a
//! TLS-handshake-success probe — it does NOT distinguish process-
//! alive (`/livez`) from request-ready (`/readyz`), so the admin
//! server is the canonical operator surface for the lifecycle
//! distinction the v1.0-GA Kubernetes manifests in `deploy/k8s/`
//! consume.
//!
//! # Why a new module (not extension of `arcgraph_mcp::transport`)
//!
//! The MCP transport surface is bounded to the JSON-RPC over HTTP/2 +
//! TLS wire shape with TOON/YAML/JSON serializers. The admin server's
//! surface is distinct: operator-facing health probes + future
//! cert-rotation triggers + future config-reload endpoints. Keeping
//! them in separate crates lets the bounded contexts stay narrow
//! (bounded-context policy) and lets `arcgraph-mcp` evolve its transport without
//! pulling axum into the dispatch path.

#![allow(clippy::module_inception)]

pub mod admin_http;
pub mod backup;
/// #1291 — served-binary default per-tenant memory cap resolution
/// (`ARCGRAPH_TENANT_MEMORY_CAP_BYTES`, default 1 GiB, `0` disables).
pub mod memory_cap;
pub mod metrics_http;
pub mod tracing_init;

pub use admin_http::{
    AdminHttpServerConfig, ComponentState, ReadinessGate, ReadinessGateSnapshot,
    build_router_with_cost, serve_admin_http, serve_admin_http_with_cost,
};
pub use memory_cap::{
    DEFAULT_PER_TENANT_MEMORY_CAP_BYTES, ENV_TENANT_MEMORY_CAP_BYTES, resolve_per_tenant_memory_cap,
};
pub use metrics_http::{MetricsHttpError, MetricsHttpServerConfig, serve_metrics_http};
pub use tracing_init::{TracingConfig, TracingGuard, init_tracing};
