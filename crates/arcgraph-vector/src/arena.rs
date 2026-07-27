//! Per-tenant `VectorArena` + `VectorArenaRegistry` (Slice F.1).
//!
//! Per ADR-035 §6.1, every vector index lives in a per-tenant
//! arena keyed by [`crate::VectorIndexHandle`] (which carries
//! `(TenantId, PartitionId, IndexId)`; v1.0 invariant
//! `partition_id == PartitionId::ZERO` per ADR-035 D-7). Tenant
//! isolation is structural: the registry's lookup is `O(1)` and
//! cross-tenant queries are *unrepresentable* — there is no path
//! from one arena to another.
//!
//! ## Two arenas, one struct (per ADR-035 §3.3 D-4 + §3.5)
//!
//! Each [`VectorArena`] owns at most two byte stores:
//!
//! 1. **Quantized arena** (`VectorArena::quantized`) — the
//!    kernel's primary storage. Contents follow the arena's
//!    [`crate::Encoding`]:
//!    - [`Encoding::F32`]: raw little-endian `f32` bytes
//!      (4 bytes/dim).
//!    - [`Encoding::Sq8`]: `i8`-native quantized bytes
//!      (1 byte/dim) per #116 closure — the codec
//!      ([`crate::quantizer::Sq8Codebook::encode`]) emits `i8`
//!      directly so the byte width matches what
//!      [`crate::distance::L2Sq8`] reads via
//!      `bytemuck::cast_slice::<u8, i8>`. No per-read
//!      translation.
//!    - [`Encoding::Binary`]: 1-bit-per-dim packed bytes,
//!      padded to 128-byte cache-line alignment per ADR-035
//!      §S-1. The arena stores the aligned bytes directly so
//!      the [`crate::distance::HammingBinary`] kernel reads
//!      cache-aligned slots.
//!    - [`Encoding::RaBitQ`]: ADR-209 standalone codec payloads;
//!      rejected by this index-side arena until slice 2 wires a
//!      prepared-query path.
//!    - [`Encoding::F16`]: reserved for v1.1; not accepted at
//!      v1.0 (returns `VectorIndexError::UnsupportedFlags`).
//!
//! 2. **Rescore arena** (`VectorArena::rescore`) — only present
//!    when [`crate::QuantizerState::default_recommends_rescore`]
//!    holds (i.e., the arena is `Sq8`, `Binary`, or `RaBitQ`).
//!    Stores the
//!    raw `f32` bytes of every inserted vector so
//!    `search_with_rescore` (Slices E.2/E.3) can re-rank the
//!    primary top-`(rescore_factor × K)` against full precision
//!    and recover recall@10 ≥ 0.95 per ADR-035 AC-1a.
//!
//! When the encoding is `F32` and the quantizer is `None`, the
//! rescore arena is `None` because the primary arena already
//! holds full-precision bytes.
//!
//! ## Filter-aware groundwork (per ADR-035 §3.4 / Slice F.3)
//!
//! `VectorArena::labels` holds the per-vector
//! [`arcgraph_core::LabelId`] set populated at insert time so
//! Filtered-DiskANN search (Slice F.3) can apply the
//! payload-aware traversal *without* a separate label index
//! table. F.1 ships only the storage; F.3 plugs in the
//! pruning logic.
//!
//! ## Local partition invariant
//!
//! [`VectorArenaRegistry`] keys on the full
//! [`crate::VectorIndexHandle`] tuple. At v1.0 every handle has
//! `partition_id == PartitionId::ZERO`, asserted by the
//! Slice A regression test
//! `vector_index_partition_id_always_zero_at_v1` and re-asserted
//! at the arena level by `arena_partition_id_always_zero_at_v1`
//! in `crates/arcgraph-vector/tests/arena.rs`.
//!
//! ## Concurrency
//!
//! - Both byte arenas and the label index use [`DashMap`] for
//!   sharded lock-free reads. Insert is `O(1)` lock-free per
//!   shard; concurrent reads of distinct vectors do not
//!   contend.
//! - The registry itself is a `DashMap<VectorIndexHandle,
//!   Arc<VectorArena>>`; the inner arenas are `Arc`-shared so
//!   callers can hold a strong reference across an HNSW or
//!   DiskANN search without holding the registry's read
//!   guard.
//! - Per-arena state changes that span more than one DashMap
//!   shard (e.g., the quantizer state transition during the
//!   OQ-V4 re-encode pass per ADR-035 §5.2 step 6) are F.5/G.4
//!   territory; F.1 ships the static post-construction state.
//!
//! ## Reference-returning getters
//!
//! [`VectorArena::get_primary`], [`VectorArena::get_rescore`],
//! and [`VectorArena::labels_for`] return guard wrappers
//! ([`ArenaSliceRef`] / [`ArenaLabelsRef`]) that `Deref` to
//! `[u8]` / `[LabelId]`. The DashMap `Ref` is held internally
//! for the lifetime of the wrapper so the slice is valid; the
//! caller treats the result as `Option<&[u8]>` /
//! `Option<&[LabelId]>` ergonomically (`bytes.len()`,
//! `&bytes[..]`, etc.).

use std::sync::Arc;

use arcgraph_core::LabelId;
use dashmap::DashMap;

use crate::quantizer::auto_quantizer_for_collection;
use crate::{
    Encoding, IndexType, QuantizerState, VectorId, VectorIndexError, VectorIndexHandle,
    quantizer::Sq8Codebook,
};

// ─── reference wrappers ──────────────────────────────────────────

/// A read guard over a primary / rescore byte slot in a
/// [`VectorArena`]. Derefs to `[u8]` so callers treat it like
/// the conceptual `&[u8]` return.
///
/// The wrapper holds a [`DashMap`] read guard internally; the
/// returned slice is valid for the wrapper's lifetime. Drop the
/// wrapper to release the read lock on the underlying shard.
pub struct ArenaSliceRef<'a> {
    inner: dashmap::mapref::one::Ref<'a, VectorId, Box<[u8]>>,
}

impl std::ops::Deref for ArenaSliceRef<'_> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.inner.value().as_ref()
    }
}

/// A read guard over a [`LabelId`] vector in a [`VectorArena`].
/// Derefs to `[LabelId]` so callers treat it like
/// `Option<&[LabelId]>` ergonomically.
pub struct ArenaLabelsRef<'a> {
    inner: dashmap::mapref::one::Ref<'a, VectorId, Vec<LabelId>>,
}

impl std::ops::Deref for ArenaLabelsRef<'_> {
    type Target = [LabelId];

    #[inline]
    fn deref(&self) -> &[LabelId] {
        self.inner.value().as_slice()
    }
}

// ─── VectorArena ─────────────────────────────────────────────────

/// Per-tenant vector arena: kernel-native primary storage, an
/// optional full-precision rescore mirror, and a per-vector
/// label index for filter-aware search.
///
/// See module docs for the full design (ADR-035 §3.5 + §6.1).
pub struct VectorArena {
    handle: VectorIndexHandle,
    encoding: Encoding,
    index_type: IndexType,
    quantizer: QuantizerState,
    dim: usize,
    /// Quantized arena (primary kernel storage).
    /// Per ADR-035 §3.3 + #116: stored bytes use the kernel's
    /// native byte width (i8 for SQ8 + simsimd L2Sq8 path).
    quantized: DashMap<VectorId, Box<[u8]>>,
    /// Full-precision arena for rescore (only present if
    /// `quantizer != QuantizerState::None`). F32 for HNSW /
    /// DiskANN rescore path per ADR-035 AC-1a.
    rescore: Option<DashMap<VectorId, Box<[u8]>>>,
    /// Per-vector-id label index (Filtered-DiskANN groundwork
    /// for Slice F.3; populated on insert).
    labels: DashMap<VectorId, Vec<LabelId>>,
}

impl VectorArena {
    /// Construct an empty arena for the given handle / shape.
    ///
    /// The rescore arena is allocated when the quantizer is
    /// non-`None` (Sq8 or Binary); otherwise it is `None` (F32
    /// arenas serve their own primary as the rescore source).
    ///
    /// # Panics
    ///
    /// - `dim == 0`. Per [`crate::hnsw::HnswGraph::new`] and
    ///   [`crate::diskann::DiskAnnGraph::new`], a zero-dim arena
    ///   has no insertable vectors; fail loud at construction.
    /// - `quantizer = Sq8/RaBitQ { params }` with
    ///   `params.dim() != dim`. The codebook must match the
    ///   arena's dim.
    #[must_use]
    pub fn new(
        handle: VectorIndexHandle,
        encoding: Encoding,
        index_type: IndexType,
        quantizer: QuantizerState,
        dim: usize,
    ) -> Self {
        assert!(dim > 0, "VectorArena dim must be > 0");
        if let QuantizerState::Sq8 { params } = &quantizer {
            assert_eq!(
                params.dim(),
                dim,
                "VectorArena dim ({dim}) does not match Sq8 codebook dim ({})",
                params.dim(),
            );
        }
        if let QuantizerState::RaBitQ { params } = &quantizer {
            assert_eq!(
                params.dim(),
                dim,
                "VectorArena dim ({dim}) does not match RaBitQ codebook dim ({})",
                params.dim(),
            );
        }
        let rescore = if quantizer.default_recommends_rescore() {
            Some(DashMap::new())
        } else {
            None
        };
        Self {
            handle,
            encoding,
            index_type,
            quantizer,
            dim,
            quantized: DashMap::new(),
            rescore,
            labels: DashMap::new(),
        }
    }

    /// Insert a vector into the arena.
    ///
    /// `full_precision_bytes` is the raw f32 byte slice of the
    /// vector (4 bytes per dim, little-endian — what
    /// [`bytemuck::cast_slice::<f32, u8>`] would produce). The
    /// arena owns the encoding step:
    ///
    /// - F32 / no quantizer: store the f32 bytes verbatim in
    ///   the primary arena. No rescore arena allocated.
    /// - Sq8: decode the f32 input, encode through the trained
    ///   [`Sq8Codebook`], cast the resulting `[i8]` to `[u8]`
    ///   (zero-copy bit reinterpretation), store in the primary
    ///   arena. Store the original f32 bytes in the rescore
    ///   arena.
    /// - Binary: encode via [`crate::quantizer::binary_encode_aligned`]
    ///   (sign function + 128-byte cache-line padding per
    ///   ADR-035 §S-1), store in the primary arena. Store the
    ///   original f32 bytes in the rescore arena.
    /// - F16: returns `UnsupportedFlags` — F16 is a v1.1 halfvec
    ///   compatibility option (ADR-035 §3.2) not auto-selected
    ///   or insert-supported at v1.0.
    ///
    /// `labels`, when `Some`, populates [`VectorArena::labels_for`]
    /// for Filtered-DiskANN (Slice F.3). When `None` the entry
    /// is omitted (callers wanting "no labels" pass `Some(&[])`
    /// to record an empty list explicitly).
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::DimensionMismatch`] when
    ///   `full_precision_bytes.len() != dim * 4`.
    /// - [`VectorIndexError::UnsupportedFlags`] for `F16` at v1.0.
    /// - [`VectorIndexError::Rebuilding`] when `encoding == Sq8`
    ///   but the quantizer is `None` (the codebook has not been
    ///   trained yet — production callers wait for the OQ-V4
    ///   re-encode pass to fire).
    pub fn insert(
        &self,
        id: VectorId,
        full_precision_bytes: &[u8],
        labels: Option<&[LabelId]>,
    ) -> Result<(), VectorIndexError> {
        // Validate the input is a well-formed f32 buffer matching
        // the arena's dim.
        let expected_bytes = self.dim * std::mem::size_of::<f32>();
        if full_precision_bytes.len() != expected_bytes {
            return Err(VectorIndexError::DimensionMismatch {
                expected: expected_bytes,
                got: full_precision_bytes.len(),
            });
        }

        // Decode the f32 view once; every encoding except F32
        // consumes a typed view of the input.
        let f32_view: &[f32] = bytemuck::cast_slice(full_precision_bytes);

        // Compute the bytes to store in the primary arena.
        let primary: Box<[u8]> = match (self.encoding, &self.quantizer) {
            (Encoding::F32, _) => full_precision_bytes.to_vec().into_boxed_slice(),
            (Encoding::Sq8, QuantizerState::Sq8 { params }) => {
                let codebook = Sq8Codebook::from_params(params.clone());
                let i8_vec = codebook.encode(f32_view)?;
                // Zero-copy bit reinterpretation: i8 and u8 share
                // a Pod layout; the kernel reads the same bytes
                // back via `cast_slice::<u8, i8>`.
                let bytes: &[u8] = bytemuck::cast_slice(&i8_vec);
                bytes.to_vec().into_boxed_slice()
            }
            (Encoding::Sq8, QuantizerState::None) => {
                // Catalog flagged Sq8 but the trainer has not
                // fired yet; the OQ-V4 re-encode pass (§5.2 step
                // 6) bridges this. Surface a retryable error so
                // the caller backs off and retries.
                return Err(VectorIndexError::Rebuilding {
                    tenant: self.handle.tenant_id,
                    index: self.handle.index_id,
                    kind: self.index_type,
                });
            }
            (Encoding::Sq8, QuantizerState::Binary) => {
                // Catalog mismatch — Sq8 encoding paired with a
                // binary quantizer state. Reject as
                // UnsupportedFlags rather than silently using one
                // path or the other.
                return Err(VectorIndexError::UnsupportedFlags {
                    encoding: self.encoding,
                    metric: crate::Metric::L2,
                });
            }
            (Encoding::Sq8, QuantizerState::RaBitQ { .. }) => {
                return Err(VectorIndexError::UnsupportedFlags {
                    encoding: self.encoding,
                    metric: crate::Metric::L2,
                });
            }
            (Encoding::Binary, _) => {
                // Binary is the deterministic sign function — no
                // codebook required, regardless of the quantizer
                // state. The aligned variant pads to 128-byte
                // cache-line per ADR-035 §S-1.
                let packed = crate::quantizer::binary_encode_aligned(f32_view);
                packed.into_boxed_slice()
            }
            (Encoding::RaBitQ, _) => {
                // TODO(#758): HNSW-arena RaBitQ needs its own asymmetric-seam
                // design (OQ-4-adjacent); the SSD tier (ADR-209 D-3) is the
                // slice-2/3 consumer.
                return Err(VectorIndexError::UnsupportedFlags {
                    encoding: Encoding::RaBitQ,
                    metric: crate::Metric::L2,
                });
            }
            (Encoding::F16, _) => {
                // v1.1 halfvec compatibility option per ADR-035
                // §3.2; not insert-supported at v1.0.
                return Err(VectorIndexError::UnsupportedFlags {
                    encoding: Encoding::F16,
                    metric: crate::Metric::L2,
                });
            }
        };

        self.quantized.insert(id, primary);

        // Populate the rescore arena when present. Quantized
        // arenas (Sq8 / Binary) need full-precision lookup;
        // F32-no-quantizer arenas serve their primary as the
        // rescore source so `rescore` is `None`.
        if let Some(rescore) = &self.rescore {
            rescore.insert(id, full_precision_bytes.to_vec().into_boxed_slice());
        }

        if let Some(label_ids) = labels {
            self.labels.insert(id, label_ids.to_vec());
        }

        Ok(())
    }

    /// Read-guard the primary kernel bytes for `id`. Returns
    /// `None` when the arena does not hold `id`.
    ///
    /// The guard derefs to `&[u8]`; callers treat it as the
    /// conceptual `Option<&[u8]>` return.
    #[must_use]
    pub fn get_primary(&self, id: VectorId) -> Option<ArenaSliceRef<'_>> {
        self.quantized.get(&id).map(|inner| ArenaSliceRef { inner })
    }

    /// Read-guard the full-precision (f32) bytes for `id`.
    ///
    /// Returns `None` when (a) the arena has no rescore arena
    /// (encoding `F32` + no quantizer), or (b) `id` has no
    /// rescore entry (race against deletion or pre-insert state).
    #[must_use]
    pub fn get_rescore(&self, id: VectorId) -> Option<ArenaSliceRef<'_>> {
        self.rescore
            .as_ref()
            .and_then(|r| r.get(&id))
            .map(|inner| ArenaSliceRef { inner })
    }

    /// Read-guard the per-vector label set for `id`. Returns
    /// `None` when no labels were recorded at insert time.
    #[must_use]
    pub fn labels_for(&self, id: VectorId) -> Option<ArenaLabelsRef<'_>> {
        self.labels.get(&id).map(|inner| ArenaLabelsRef { inner })
    }

    /// Number of vectors currently held in the primary arena.
    /// Used by I-V1 sanity checks (per ADR-035 invariant tests)
    /// and by the auto-quantize trigger threshold (§5.2 step 6).
    #[must_use]
    pub fn vectors_count(&self) -> usize {
        self.quantized.len()
    }

    /// Whether the rescore arena is allocated.
    ///
    /// `true` when [`QuantizerState::default_recommends_rescore`]
    /// holds at construction time. Arenas constructed with
    /// `QuantizerState::None` + `Encoding::F32` have no rescore
    /// arena because the primary already holds full precision.
    #[inline]
    #[must_use]
    pub fn has_rescore(&self) -> bool {
        self.rescore.is_some()
    }

    /// The handle this arena was constructed for.
    #[inline]
    #[must_use]
    pub const fn handle(&self) -> &VectorIndexHandle {
        &self.handle
    }

    /// Encoding of the primary arena (the v1.0 enum tag from
    /// the catalog at `DEFINE INDEX` DDL time).
    #[inline]
    #[must_use]
    pub const fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Index algorithm of the arena (HNSW or DiskANN).
    #[inline]
    #[must_use]
    pub const fn index_type(&self) -> IndexType {
        self.index_type
    }

    /// Vector dimension this arena was constructed for.
    #[inline]
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Auto-quantize hint per ADR-035 D-4 / Q3 ratification:
    /// returns `Some(Encoding::Sq8)` when `n_vectors >= 10 M`.
    ///
    /// **F.1 just exposes the API**: production wiring of the
    /// size-aware dispatch (catalog → trainer trigger → arena
    /// quantizer-state transition) lives in Slice F.5 / G.4.
    /// This helper is the API surface those slices consume.
    #[inline]
    #[must_use]
    pub const fn auto_quantize_target(n_vectors: usize) -> Option<Encoding> {
        auto_quantizer_for_collection(n_vectors)
    }
}

// ─── VectorArenaRegistry ─────────────────────────────────────────

/// Per-tenant arena registry. Routes
/// [`crate::VectorIndexHandle`] tuples to the right
/// [`VectorArena`] in `O(1)`.
///
/// Registry keys use `(TenantId, PartitionId, IndexId)`, with
/// `PartitionId::ZERO` enforced by the public handle constructor.
#[derive(Default)]
pub struct VectorArenaRegistry {
    arenas: DashMap<VectorIndexHandle, Arc<VectorArena>>,
}

impl VectorArenaRegistry {
    /// An empty registry. Callers populate via
    /// [`Self::create_arena`].
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up the arena for `handle`. Returns `None` when the
    /// arena has not been created (DDL race) or has been dropped
    /// out from under a stale handle. Callers translate to
    /// [`VectorIndexError::ArenaNotFound`] at the public API
    /// boundary.
    #[must_use]
    pub fn for_tenant_index(&self, handle: VectorIndexHandle) -> Option<Arc<VectorArena>> {
        self.arenas.get(&handle).map(|r| Arc::clone(r.value()))
    }

    /// Create an arena for `handle`. Returns the freshly
    /// constructed `Arc<VectorArena>` (also retained inside the
    /// registry).
    ///
    /// If an arena already exists for `handle`, it is replaced —
    /// callers that need "create-only" semantics check
    /// [`Self::for_tenant_index`] first. The registry intentionally
    /// allows replacement so the OQ-V4 re-encode pass (§5.2 step
    /// 6) can swap in a freshly-trained arena atomically when the
    /// time comes.
    pub fn create_arena(
        &self,
        handle: VectorIndexHandle,
        encoding: Encoding,
        index_type: IndexType,
        quantizer: QuantizerState,
        dim: usize,
    ) -> Arc<VectorArena> {
        let arena = Arc::new(VectorArena::new(
            handle, encoding, index_type, quantizer, dim,
        ));
        self.arenas.insert(handle, Arc::clone(&arena));
        arena
    }

    /// Number of arenas currently registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.arenas.len()
    }

    /// Whether the registry holds zero arenas.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.arenas.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{IndexId, Sq8Params};
    use arcgraph_core::{PartitionId, TenantId};

    fn handle_for(tenant: u64, idx: u64) -> VectorIndexHandle {
        VectorIndexHandle::for_tenant(TenantId::new(tenant), IndexId::new(idx))
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(v).to_vec()
    }

    #[test]
    fn empty_arena_has_zero_vectors() {
        let arena = VectorArena::new(
            handle_for(1, 1),
            Encoding::F32,
            IndexType::Hnsw,
            QuantizerState::None,
            4,
        );
        assert_eq!(arena.vectors_count(), 0);
        assert!(!arena.has_rescore());
    }

    #[test]
    fn sq8_arena_has_rescore_arena() {
        let params = Sq8Params::try_new(vec![1.0; 4], vec![0.0; 4]).unwrap();
        let arena = VectorArena::new(
            handle_for(1, 1),
            Encoding::Sq8,
            IndexType::Hnsw,
            QuantizerState::Sq8 { params },
            4,
        );
        assert!(arena.has_rescore());
    }

    #[test]
    fn binary_arena_has_rescore_arena() {
        let arena = VectorArena::new(
            handle_for(1, 1),
            Encoding::Binary,
            IndexType::Hnsw,
            QuantizerState::Binary,
            8,
        );
        assert!(arena.has_rescore());
    }

    #[test]
    fn f32_no_quantizer_arena_has_no_rescore() {
        let arena = VectorArena::new(
            handle_for(1, 1),
            Encoding::F32,
            IndexType::Hnsw,
            QuantizerState::None,
            4,
        );
        assert!(!arena.has_rescore());
    }

    #[test]
    fn insert_f32_round_trips_through_primary() {
        let arena = VectorArena::new(
            handle_for(1, 1),
            Encoding::F32,
            IndexType::Hnsw,
            QuantizerState::None,
            3,
        );
        let v = vec![1.0_f32, -2.0, 3.5];
        arena
            .insert(VectorId::new(7), &f32_bytes(&v), None)
            .unwrap();
        assert_eq!(arena.vectors_count(), 1);
        let bytes = arena.get_primary(VectorId::new(7)).unwrap();
        assert_eq!(&*bytes, f32_bytes(&v).as_slice());
        // F32-no-quantizer path: the rescore arena is None.
        assert!(arena.get_rescore(VectorId::new(7)).is_none());
    }

    #[test]
    fn insert_rejects_dim_mismatch() {
        let arena = VectorArena::new(
            handle_for(1, 1),
            Encoding::F32,
            IndexType::Hnsw,
            QuantizerState::None,
            3,
        );
        // Pass 4 floats (16 bytes) into a dim=3 arena (12 bytes).
        let bad = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let err = arena
            .insert(VectorId::new(0), &bad, None)
            .expect_err("must reject dim mismatch");
        assert!(
            matches!(
                err,
                VectorIndexError::DimensionMismatch {
                    expected: 12,
                    got: 16
                }
            ),
            "got: {err:?}"
        );
    }

    #[test]
    fn insert_f16_returns_unsupported_flags() {
        let arena = VectorArena::new(
            handle_for(1, 1),
            Encoding::F16,
            IndexType::Hnsw,
            QuantizerState::None,
            4,
        );
        let bytes = f32_bytes(&[0.0; 4]);
        let err = arena.insert(VectorId::new(0), &bytes, None).unwrap_err();
        assert!(
            matches!(
                err,
                VectorIndexError::UnsupportedFlags {
                    encoding: Encoding::F16,
                    ..
                }
            ),
            "got: {err:?}"
        );
    }

    #[test]
    fn insert_sq8_with_no_quantizer_state_is_rebuilding() {
        // Catalog flagged Sq8 but the trainer has not fired yet;
        // the §5.2 step 6 re-encode pass bridges this.
        let arena = VectorArena::new(
            handle_for(1, 1),
            Encoding::Sq8,
            IndexType::Hnsw,
            QuantizerState::None,
            4,
        );
        let bytes = f32_bytes(&[0.0; 4]);
        let err = arena.insert(VectorId::new(0), &bytes, None).unwrap_err();
        assert!(
            matches!(err, VectorIndexError::Rebuilding { .. }),
            "got: {err:?}"
        );
    }

    #[test]
    fn registry_create_then_lookup() {
        let registry = VectorArenaRegistry::new();
        let h = handle_for(1, 7);
        let arena =
            registry.create_arena(h, Encoding::F32, IndexType::Hnsw, QuantizerState::None, 4);
        let looked = registry.for_tenant_index(h).unwrap();
        // Same Arc payload — pointer-equal.
        assert!(Arc::ptr_eq(&arena, &looked));
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
    }

    #[test]
    fn registry_lookup_missing_returns_none() {
        let registry = VectorArenaRegistry::new();
        assert!(registry.for_tenant_index(handle_for(1, 999)).is_none());
    }

    #[test]
    fn auto_quantize_target_returns_sq8_at_threshold() {
        assert_eq!(VectorArena::auto_quantize_target(9_999_999), None);
        assert_eq!(
            VectorArena::auto_quantize_target(10_000_000),
            Some(Encoding::Sq8)
        );
    }

    #[test]
    fn arena_handle_partition_id_is_zero_at_v1() {
        let arena = VectorArena::new(
            handle_for(1, 1),
            Encoding::F32,
            IndexType::Hnsw,
            QuantizerState::None,
            4,
        );
        assert_eq!(arena.handle().partition(), PartitionId::ZERO);
        assert!(arena.handle().is_v1_local());
    }
}
