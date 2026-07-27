//! Vector index engine for ArcGraph (v1.0 hybrid HNSW + DiskANN).
//!
//! Per ADR-035, this crate owns the vector retrieval surface:
//!
//! - `VectorIndexHandle` — tenant-scoped entry point keyed by
//!   `(TenantId, PartitionId, IndexId)`. `PartitionId` is always
//!   [`PartitionId::ZERO`].
//! - `DistanceKernel` — SIMD-backed distance trait. v1.0 implementors
//!   wrap [`simsimd`] (Slice B); v1.0 covers L2 / IP / Cosine on
//!   F32 / F16 / SQ8 plus Hamming on Binary.
//! - `FilteredVectorIndex` — object-safe runtime-polymorphism trait
//!   for filtered search dispatch (Slice F.4). Both HNSW
//!   (`FilteredHnsw`) and DiskANN (`DiskAnnGraph`) implement it.
//! - Quantizer state (`QuantizerState`) — SQ8 default, binary
//!   opt-in per ADR-035 D-4. Trained per (tenant, index) on the
//!   Tokio background pool.
//!
//! ## What this crate is NOT
//!
//! Page persistence, WAL integration, CommitBundle v3 staging, and
//! Z-1 (b) rollback wiring all live in `arcgraph-storage`'s
//! `VectorPageStore` (Slice G of M3.a). This crate consumes that
//! interface; it does not own bytes-on-disk.
//!
//! [`PartitionId::ZERO`]: arcgraph_core::PartitionId::ZERO

// `unsafe` is denied by default; the only crate-local allowance
// is the f16 byte-slice cast in `distance.rs` (simsimd's f16 is
// not `bytemuck::Pod`). Every `unsafe { … }` block in this crate
// MUST carry a `// SAFETY:` comment under the code-quality policy.
#![deny(unsafe_code)]
#![recursion_limit = "256"]

pub mod arena;
pub mod diskann;
pub mod dispatcher;
pub mod distance;
pub mod encoding;
pub mod error;
pub mod handle;
pub mod hnsw;
pub mod ids;
pub mod quantizer;
pub mod query;

pub use arena::{ArenaLabelsRef, ArenaSliceRef, VectorArena, VectorArenaRegistry};
pub use dispatcher::{
    BackendKind, BackendSet, DispatchPreference, FilteredVectorIndex, dispatch_preference,
};
pub use distance::DistanceKernel;
pub use encoding::{Encoding, IndexType, Metric};
pub use error::VectorIndexError;
pub use handle::VectorIndexHandle;
pub use ids::{IndexId, VectorId};
pub use quantizer::{QuantizerState, RaBitQParams, Sq8Params};
pub use query::{Filter, PropertyKey, PropertyValue};

/// Canonical `Result` alias for the crate. The error is the
/// codec-local [`VectorIndexError`]; per the workspace pattern in
/// `docs/codec-error-translation.md`, this error is translated to
/// `arcgraph_core::ArcGraphError` only at the public crate
/// boundary that the engine wires together (which is not this
/// crate; see `arcgraph-storage::CrudError` for the analogue).
pub type Result<T> = std::result::Result<T, VectorIndexError>;
