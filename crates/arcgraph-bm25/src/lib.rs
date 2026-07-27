//! BM25 text search via Tantivy — the 8th bounded context for
//! ArcGraph v1.0 (ADR-036 §D-2 + ADR-039).
//!
//! # What this crate is
//!
//! - The Tantivy schema (`segment.rs`) and MVCC visibility filter
//!   (`mvcc.rs`) for v1.0 BM25.
//! - The per-tenant search-side handle (`handle.rs`:
//!   [`Bm25IndexHandle`]) — top-K search, filtered search, upsert,
//!   delete.
//! - The workspace-level service (`service.rs`: [`Bm25Service`]) —
//!   per-tenant directory layout under `<data_dir>/bm25/`, lazy
//!   `open_or_create` on first touch, append-only handle cache.
//! - The commit-side trait impl (`store.rs`):
//!   [`Bm25Service`] impls
//!   [`arcgraph_storage::mutation_log::Bm25IndexStoreHandle`] so the
//!   kernel commit closure can dispatch `commit_pending` /
//!   `rollback_pending` per tenant without taking a Tantivy
//!   dependency.
//!
//! # What this crate is NOT
//!
//! - **No DDL.** `CREATE TEXT INDEX ON <Label>(<property>)` and
//!   per-language analyzers are M7 / v1.1 scope per ADR-036 §D-2.
//! - **Snapshot visibility is storage-owned.** BM25 filters on the
//!   transaction's MVCC window and does not reconstruct historical state.
//! - **No CommitBundle codec bump.** Per ADR-039 §D-5, BM25 docs
//!   live in Tantivy's own commit log; the bundle does NOT carry
//!   BM25 bytes.
//!
//! # Apache-2.0 license chain
//!
//! Tantivy is dual-licensed `MIT OR Apache-2.0`; ArcGraph adopts
//! under the Apache-2.0 arm for license-chain consistency
//! (workspace Apache-2.0 licensing policy).

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![recursion_limit = "256"]

pub mod error;
pub mod eviction;
pub mod handle;
pub mod mvcc;
pub mod pool;
pub mod segment;
pub mod service;
pub mod store;

pub use error::Bm25Error;
pub use eviction::{IDLE_EVICTION_COMMIT_THRESHOLD, IDLE_EVICTION_WALL_CLOCK_THRESHOLD_SECS};
pub use handle::{Bm25IndexHandle, Filter, IndexId};
pub use mvcc::build_visibility_filter;
pub use pool::{WRITER_ACQUIRE_BLOCK_TIMEOUT, WRITER_POOL_SIZE};
pub use segment::Bm25Schema;
#[doc(hidden)]
pub use service::Bm25DirectoryFactory;
pub use service::Bm25Service;
