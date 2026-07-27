//! Engine bootstrap glue (M3.d-3 — closes F-2 SUBSTANTIVE per
//! ADR-040 amendment-04 §D-3 + amendment-05 §D-5).
//!
//! This module wires the workspace-wide production composition that
//! ADR-040 §D-7 + ADR-037 §D-1 commit to but that prior slices
//! deliberately deferred:
//!
//! - **[`graph_adapter::CrudStoreGraphAdapter`]** — materialises a
//!   per-tenant [`arcgraph_community::Graph`] from a
//!   [`crate::crud::CrudStore`] at the txn manager's current visible
//!   snapshot.
//! - **[`refresh_hook::ProductionRefreshHook`]** — the v1.0
//!   production [`arcgraph_community::RefreshHook`] impl. Closes the
//!   "0 prod impls" residual concern from F-2 research §1.2 by
//!   converting it to "1 prod impl" while consuming the
//!   [`crate::crud::CrudStore`] + the shared
//!   [`arcgraph_community::SharedBTreeIndexProvider`] (PR #218).
//! - **[`bootstrap::bootstrap_engine`]** — the orchestration entry
//!   point. Wires [`crate::router::MultiTenantRouter`] +
//!   [`arcgraph_community::CommunityRefreshScheduler`] +
//!   [`refresh_hook::ProductionRefreshHook`] together and returns the
//!   composed [`bootstrap::EngineHandles`].
//!
//! ## Why this lives in `arcgraph-storage`
//!
//! `arcgraph-storage` already depends on `arcgraph-community`
//! (`MultiTenantRouter` consumes [`arcgraph_community::CommunityIndexProvider`]
//! per ADR-040 §D-3). Putting the engine glue here adds zero new
//! dependency edges in `docs/bounded-contexts.md`. The alternative —
//! a NEW `arcgraph-engine` crate — adds workspace burden without
//! payoff at v1.0; if a future MCP / CLI surface needs richer engine
//! orchestration the module can be promoted to a dedicated crate at
//! that point.
//!
//! ## v1.0 production posture: per-tick re-materialisation (amendment-05)
//!
//! Per ADR-040 amendment-05 (Wave 9b Slice 4) the
//! [`arcgraph_community::RefreshHook`] trait surface returns owned
//! `Arc<Graph>` + `Arc<BTreeMembershipIndex>`. Production hooks
//! re-materialise per-tenant Graphs from `CrudStore` on each
//! scheduler tick — the canonical reset (per ADR-040 §D-7) runs
//! against the current substrate state, not a frozen-at-bootstrap
//! snapshot.
//!
//! [`bootstrap::bootstrap_engine`] does NOT pre-materialise per-
//! tenant Graphs at boot time. The
//! [`refresh_hook::ProductionRefreshHook`] is constructed empty;
//! tenants are registered via
//! [`refresh_hook::ProductionRefreshHook::register_tenant`]; the
//! first scheduler tick for each tenant materialises its Graph
//! afresh.
//!
//! ### Boot-time validation (opt-in)
//!
//! Operators who want eager materialisation (so first-tick errors
//! surface at boot rather than silently soft-skipping) can call
//! [`refresh_hook::ProductionRefreshHook::warm_up`] for each
//! registered tenant in their startup sequence. The default
//! [`bootstrap::bootstrap_engine`] does NOT call `warm_up` — boot
//! must remain robust to a single corrupted-tenant scenario.
//!
//! ## Forward link: pre-amendment-05 history (FROZEN-GRAPH retired)
//!
//! Pre-amendment-05 (PR #235), this module shipped a v1.0
//! "frozen-graph" workaround: per-tenant Graphs were materialised at
//! engine bootstrap and frozen into a `HashMap<TenantId, Box<Graph>>`
//! field on `ProductionRefreshHook`. The frozen-graph posture
//! honoured the determinism property of ADR-040 §D-7 ("daily refresh
//! is the canonical reset") but NOT the substrate-freshness property
//! — long-uptime continuous-ingest deployments saw their canonical
//! reset diverge from live data until operator restart.
//!
//! Amendment-05 retires the frozen-graph posture entirely. The
//! `ProductionRefreshHookBuilder` type that pre-materialised Graphs
//! at construction is gone; the simpler
//! [`refresh_hook::ProductionRefreshHook::new`] constructor takes
//! its place.
//!
//! ## v1.1+ forward references
//!
//! Per amendment-05 §5:
//!
//! - **Cross-node refresh coordination** — v1.1 partitioned
//!   deployments may need a coordinator that decides which node
//!   materialises which (tenant, partition) per refresh tick. Trait
//!   surface (`Arc<Graph>` + `Arc<BTreeMembershipIndex>`) is
//!   forward-compatible.
//! - **Streaming materialisation** — v1.0 builds the full CSR Graph
//!   in-memory; large tenants (>10M edges) may benefit from a
//!   streaming materialiser that pipelines TEL scans → CSR build.
//!   Tracked at `docs/roadmap.md` M5+.
//! - **Catalog-change-driven refresh** — v1.0 fires daily refresh
//!   on fixed cadence; v1.1+ may surface a "refresh-on-N-batches"
//!   hook for high-ingest tenants. Trait surface forward-compatible.
//!
//! ## Memory budget (v1.0 envelope)
//!
//! Per `arcgraph-community/src/graph.rs:40-57`, the CSR `Graph` is
//! `8n + 16m` bytes (offsets `u32 × (n+1)` + neighbours `u32 × 2m`
//! + weights `f32 × 2m` + degrees `f32 × n`). Concretely at n=1M:
//!
//! - sparse (avg degree 2):    n=1M → ~24 MiB
//! - moderate (avg degree 10): n=1M → ~88 MiB
//! - typical KG (avg deg 20):  n=1M → ~168 MiB
//! - dense KG (avg deg 100):   n=1M → ~808 MiB
//!
//! Post-amendment-05, the production hook holds an `Arc<Graph>` in
//! its diagnostic cache per registered tenant from the most recent
//! `resolve()` call onward — the resident memory is the same as the
//! pre-amendment-05 frozen-graph posture (one Graph per tenant) plus
//! a one-pointer Arc indirection. The transient overhead during a
//! tick is one additional `Arc<Graph>` clone (~5 ns) live for the
//! duration of `GveLeiden::run` — negligible vs. the run itself.
//!
//! Operators with multi-tenant deployments should size against the
//! upper density end. v1.1 may stream materialisation to bound
//! resident memory (per `docs/roadmap.md` M5+).

pub mod bootstrap;
pub mod graph_adapter;
pub mod refresh_hook;

pub use bootstrap::{EngineConfig, EngineError, EngineHandles, bootstrap_engine};
pub use graph_adapter::{CrudStoreGraphAdapter, GraphAdapterError};
pub use refresh_hook::{ProductionRefreshHook, ProductionRefreshHookError};
