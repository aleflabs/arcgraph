//! DiskANN / Vamana vector index — Slice D in-memory baseline.
//!
//! Per ADR-035 §3.1 + §5.3 + the Subramanya et al. NeurIPS 2019
//! Vamana paper. This module ships the **in-memory** build + beam
//! search + delta-segment streaming-insert lookaside; persistence
//! (Slice G), filter-aware build/search (Slice F.3), and rescore
//! wiring (Slice E.3) layer on top.
//!
//! ## Module layout
//!
//! - [`graph`] — [`DiskAnnGraph`] data structure + per-label
//!   index hooks (Filtered-DiskANN scaffolding shared with
//!   [`filtered`]).
//! - [`build`] — Vamana α-pruning bulk build (Algorithm 1).
//! - [`search`] — beam search (Algorithm 2).
//! - [`stream`] — delta-segment lookaside per ADR-035 §5.3.1
//!   B-3 resolution; preserves the I-V7 T1 read-your-writes
//!   invariant.
//! - [`filtered`] — Slice F.3 label-aware Vamana α-prune +
//!   filter-aware beam search per Gollapudi et al. WWW 2023.
//!   Layered on top of the Slice D scaffolding without
//!   reshaping the base build / search / stream paths.
//!
//! ## Tuning defaults (ADR-035 §5.3 / handoff prompt)
//!
//! Defaults live on [`DiskAnnParams`] and are wired from the
//! planner at index-create time. The slice-D defaults are:
//!
//! | Parameter            | Default | Source                    |
//! |----------------------|---------|---------------------------|
//! | `r` (max degree)     | 70      | Slice D handoff prompt    |
//! | `alpha`              | 1.2     | ADR-035 §5.3 / Vamana §3  |
//! | `l_construction`     | 100     | Vamana §3 / ADR-035 §3.1  |
//! | `l_search_default`   | 100     | ADR-035 §3.1              |
//! | `delta_max_size`     | 1000    | ADR-035 §5.3.1            |
//! | `delta_brute_thresh` | 128     | Slice D handoff prompt    |
//! | `medoid_sample_size` | 1000    | Microsoft DiskANN ref     |
//!
//! ## What this module is NOT
//!
//! - Persistence — the snapshot byte format (ARCV §10.3) is
//!   produced by Slice G.2 from a built graph; the snapshot
//!   path is wired by `arcgraph_storage::VectorPageStore` at
//!   Slice G.1, not by this module.
//! - Persistence of the Filtered-DiskANN per-label entry-point
//!   cache — Slice G.2 owns the snapshot byte format; the
//!   in-memory cache populated by
//!   [`DiskAnnGraph::build_filtered`] is rebuilt on replay.
//! - Rescore — search returns the kernel's distance directly;
//!   the rescore pipeline against a full-precision section is
//!   Slice E.3.
//! - `pread` for cold pages — Slice G owns the on-disk
//!   contiguous Vamana edge layout. (#808 shipped `PosixPageIo`
//!   on `pread`, NOT mmap, per Prime-Directive #2 "no mmap in the
//!   hot path"; the prior "mmap / pread" wording is corrected here.)

pub mod build;
pub mod filtered;
pub mod graph;
pub mod persist;
pub mod rss_guard;
pub mod search;
pub mod ssd;
pub mod stream;

pub use graph::{DiskAnnGraph, DiskAnnLabelId, DiskAnnParams};
pub use rss_guard::RssGuard;
pub use ssd::{RecordLayout, SsdDiskAnnIndex};

// The canonical `Filter` enum lives in [`crate::query`] per
// ADR-035 amendment-03 (issue #127). Re-exported here so existing
// call sites that resolve against `arcgraph_vector::diskann::Filter`
// keep compiling — the canonical resolution path is
// `arcgraph_vector::Filter`.
pub use crate::query::Filter;
