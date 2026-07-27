//! Community detection for ArcGraph.
//!
//! Per ADR-040 (advanced from v1.1 to v1.0 per owner directive
//! 2026-04-27 ratifying ADR-036 §D-6), this crate owns:
//!
//! - GVE-Leiden static algorithm (Sahu ICPP 2024) — implemented
//!   in `leiden_static` module (M3.d-1 task #4).
//! - DF Leiden incremental algorithm (Sahu arxiv 2024) —
//!   implemented in `leiden_incremental` module (M3.d-2).
//! - B-tree membership index `(tenant_id, community_id, level,
//!   node_id)` plus reverse `(tenant_id, node_id, level) →
//!   community_id` (M3.d-1 task #4).
//! - [`CommunityIndexProvider`] — factory trait that the
//!   storage-layer `MultiTenantRouter` consults to materialise a
//!   per-tenant [`CommunityIndexHandle`] on `route()` (M3.d-1 #2).
//! - Background daily-refresh scheduler (M3.d-2).
//!
//! See [`CommunityIndexHandle`] for the public API surface.
//!
//! ## What this crate is NOT
//!
//! Membership-index page persistence, WAL integration, and
//! catalog DDL parsing all live elsewhere. This crate exposes a
//! [`MembershipIndex`] trait that the storage / router layer
//! satisfies; it does not own bytes-on-disk or the
//! `DEFINE INDEX … USING COMMUNITY` parser surface.
//!
//! `partition_id` is currently required to be [`PartitionId::ZERO`].
//!
//! [`PartitionId::ZERO`]: arcgraph_core::PartitionId::ZERO

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![recursion_limit = "256"]

pub mod error;
pub mod graph;
pub mod handle;
pub mod ids;
pub mod index;
pub mod leiden_incremental;
pub mod leiden_static;
pub mod membership_index;
pub mod provider;
pub mod scheduler;

pub use error::CommunityError;
pub use graph::Graph;
pub use handle::CommunityIndexHandle;
pub use ids::{CommunityId, CommunityIndexId, Level};
pub use index::MembershipIndex;
pub use leiden_incremental::{EdgeUpdate, IncrementalResult, LeidenIncremental};
pub use leiden_static::{GveLeiden, LeidenParams, LeidenResult, modularity};
pub use membership_index::BTreeMembershipIndex;
pub use provider::{CommunityIndexProvider, SharedBTreeIndexProvider};
pub use scheduler::{
    CommunityRefreshScheduler, OwnedRefreshInputs, RefreshHook, RefreshObserver, SchedulerConfig,
    SchedulerHealth,
};

/// Canonical `Result` alias for the crate. The error is the
/// codec-local [`CommunityError`]; per the workspace pattern in
/// `docs/codec-error-translation.md`, this error is translated to
/// `arcgraph_core::ArcGraphError` only at the public crate
/// boundary that the engine wires together (which is not this
/// crate; see `arcgraph-storage` for the analogue surface).
pub type Result<T> = std::result::Result<T, CommunityError>;
