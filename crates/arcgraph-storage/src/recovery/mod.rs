//! Cold-start derivative-substrate rebuild module.
//!
//! This module homes the post-recovery rebuild paths for substrates that
//! are DERIVED from the recovered MVCC primary store and do NOT replay
//! from the WAL directly:
//!
//! - [`stats_rebuild`] — per-tenant `CatalogStats` cardinality counters
//!   (M4-41, ADR-038 amendment-06 §D-25.1).
//! - [`tel_rebuild`] — per-tenant TEL **adjacency** chains that
//!   `scan_out` / `scan_in` walk for `MATCH ()-[r]->()` traversal (P0
//!   #780; the TEL is not in the CommitBundle — issue #20 — so it must
//!   be rebuilt from the recovered rel records here).
//! - [`index_rebuild`] — per-`(tenant, kind, id)` PRIMARY + per-node
//!   SECONDARY index reconciliation-from-MVCC (#1380). The live commit's
//!   dual-write index install DEGRADES on failure (warn-and-continue per
//!   ADR-023), so a record can be MVCC-committed (scan-visible) yet
//!   missing from the id/label lookup indices — and, before this pass,
//!   that split-brain survived restart. This heals it from the
//!   authoritative recovered MVCC records, mirroring the stats/TEL
//!   rebuilds.
//!
//! All three are intentionally SEPARATE from [`crate::wal::replay`] (the
//! canonical primary-store WAL replay path); per amendment-06 §Context
//! the "derivatives-rebuild-separately" architectural posture forbids
//! `ReplayExecutor` from calling per-tenant derivative hooks (locked
//! invariant I-Q17). This module is the dedicated home for
//! derivative-substrate rebuild paths invoked AFTER `recover_from_wal`
//! returns, but BEFORE query serving.
//!
//! ## Why a dedicated module rather than `wal::recovery`
//!
//! `wal::recovery` is the WAL primary-store replay path (ADR-032
//! `ReplayExecutor::run`). The derivative rebuilds do NOT live there
//! because the recovery shape is fundamentally different (per
//! amendment-06 §Context):
//!
//! | Path | Source of truth | Driver |
//! |------|-----------------|--------|
//! | `wal::recovery` | WAL records | `ReplayExecutor` walks WAL by LSN |
//! | `recovery::stats_rebuild` | Recovered MVCC chains | Per-tenant scan at recovered LSN |
//! | `recovery::tel_rebuild` | Recovered MVCC chains | Per-tenant scan at recovered LSN |
//! | `recovery::index_rebuild` | Recovered MVCC chains | Per-tenant scan at recovered LSN |
//!
//! Mixing them would (per amendment-06 §Context) re-introduce the
//! double-counting risk that locked invariant I-Q17 prevents at v1.0
//! GA option-(b) promotion.

pub mod index_rebuild;
pub mod stats_rebuild;
pub mod tel_rebuild;

pub use index_rebuild::{
    IndexRebuildOutcome, IndexRebuildReport, rebuild_all_tenant_index, rebuild_index_for_tenant,
};
pub use stats_rebuild::{
    RebuildReport, TenantRebuildOutcome, rebuild_all_tenant_stats, rebuild_catalog_stats_for_tenant,
};
pub use tel_rebuild::{
    AdjacencyRebuildOutcome, AdjacencyRebuildReport, rebuild_adjacency_for_tenant,
    rebuild_all_tenant_adjacency,
};
