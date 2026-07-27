//! SSD-resident DiskANN serving tier (ADR-195).
//!
//! Composes EXISTING mmap-free primitives into a bounded-RAM ANN index that
//! serves a >10M-vector corpus on a 19 GB box (ADR-189 §B GA-block):
//!
//! - **On disk** (`arcgraph_storage::PosixPageIo`, pread/pwrite — **NOT mmap**,
//!   PD#2 / design-v2 §3.4): full-precision f32 vectors, co-located per node in
//!   an 8 KiB page store. The rerank reads them through a bounded
//!   `arcgraph_storage::BufferPool`.
//! - **In RAM** (bounded): an SQ8-compressed navigation graph
//!   ([`DiskAnnGraph`] under [`Encoding::Sq8`]) — vectors + adjacency. Building
//!   under SQ8 makes the in-RAM `vectors` array 4× smaller than f32 AND gives
//!   the beam-search phase-1 the same distance space the graph was optimised
//!   for.
//! - **Search**: 2-phase. Phase-1 = SQ8-distance beam-search in RAM (candidate
//!   generation). Phase-2 = exact f32 rerank of the top candidates, read from
//!   the page store through the `BufferPool`.
//! - **RSS guard** ([`super::rss_guard::RssGuard`]): detect-and-abort backstop.
//!
//! ## Back-of-envelope budget (PD#5; ADR-195 §1/§2 — re-derived from code)
//!
//! At the GA-gate point `N = 10M`, `dim = 768`, `R = 128`
//! (`params_for_dim(768)` — the V-1 #740 measured curve; the 128-d defaults are
//! graph-starved at 768d), `PAGE_SIZE = 8192` (`arcgraph_core::PAGE_SIZE`):
//!
//! - **RAM (serving):** SQ8 nav `10M × 768 × 1 B = 7.68 GB` + adjacency
//!   `10M × 128 × 4 B = 5.12 GB` + bounded `BufferPool` (rerank cache) + id
//!   maps ≈ **13–14 GB** → RIGHT at the 14 GB cap. This is the RC-1 GO/NO-GO
//!   checkpoint (genuinely tight); PQ-nav (`10M × 96 B = 0.96 GB`) is the
//!   pre-committed fallback (ADR-195 §2.1).
//! - **RAM (build):** the f32 corpus is NEVER all-resident — it is streamed to
//!   disk in bounded batches (ADR-195 §3, RC-3); the build holds only the SQ8
//!   nav + adjacency + one batch (≈ same 13–14 GB peak). The guard backstops a
//!   spike.
//! - **Disk:** record = `align8(8 + 768·4 + 4 + 128·4) = 3600 B`;
//!   `records_per_page = 8192 / 3600 = 2`; `pages = ceil(10M / 2) = 5M`;
//!   `disk = 5M × 8192 = 40.96 GB` (raw f32 = 30.7 GB; ~12% page-tail padding).
//!   At the ADR-189 §B `NVMe ≤ 40 GB` budget. **1 record/page would be 80 GB**
//!   (blowing the budget) — packing 2/page is required.

use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use arcgraph_core::{PAGE_SIZE, PageId};
use arcgraph_storage::{BufferPool, PageIo, PosixPageIo};

use super::graph::{DiskAnnGraph, DiskAnnParams};
use super::persist;
use super::rss_guard::RssGuard;
use crate::distance::{DistanceKernel, L2F32, L2RaBitQSym, L2Sq8};
use crate::quantizer::{RaBitQCodebook, RaBitQParams, Sq8Codebook, Sq8Params};
use crate::{Encoding, IndexId, Metric, Result, VectorId, VectorIndexError};

/// Bytes for the per-record node id (`u64`).
const NODE_ID_BYTES: usize = 8;
/// Bytes for the per-record degree (`u32`).
const DEGREE_BYTES: usize = 4;
/// Bytes per adjacency entry (`u32` neighbor slot).
const ADJ_ELEM_BYTES: usize = 4;
/// How often (in vectors processed) the build polls the RSS guard.
const BUILD_GUARD_CHECK_EVERY: usize = 4096;
/// Default phase-1 over-fetch factor for the 2-phase rerank (ADR-035 AC-1a's
/// `rescore_factor = 5×` convention: phase-1 returns `5·k` SQ8 candidates that
/// phase-2 reranks by exact f32 distance to the final top-`k`).
pub const DEFAULT_RERANK_FACTOR: usize = 5;

/// Magic for the [`SsdDiskAnnIndex`] nav sidecar file header (`ArcGraph SSD
/// Index 1`).
const SSD_NAV_MAGIC: &[u8; 8] = b"AGSSIX\x01\x00";
/// Current nav sidecar file format version.
const SSD_NAV_VERSION: u32 = 2;
const SSD_NAV_VERSION_SQ8: u32 = 1;
const SSD_NAV_KIND_SQ8: u8 = 2;
const SSD_NAV_KIND_RABITQ: u8 = 4;

/// Fixed-size on-disk record geometry for the co-located (vector + adjacency)
/// page store (ADR-195 §1 / DiskANN NeurIPS 2019 §4.4).
///
/// One node-record never straddles a page boundary (so a single `read_page`
/// returns the whole node — the §4.4 single-fault property). `record_size` is
/// 8-byte-aligned and `records_per_page = PAGE_SIZE / record_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordLayout {
    /// Vector dimensionality.
    pub dim: usize,
    /// Maximum out-degree stored on disk (`R`).
    pub r_max: usize,
    /// `dim · 4` — bytes of the f32 vector payload.
    pub vec_bytes: usize,
    /// `r_max · 4` — bytes reserved for adjacency.
    pub adj_bytes: usize,
    /// 8-byte-aligned size of one node-record.
    pub record_size: usize,
    /// Records packed into one [`PAGE_SIZE`] page (`≥ 1`).
    pub records_per_page: usize,
    /// Byte offset of the f32 vector within a record.
    pub vec_offset: usize,
    /// Byte offset of the degree `u32` within a record.
    pub degree_offset: usize,
    /// Byte offset of the adjacency array within a record.
    pub adj_offset: usize,
}

impl RecordLayout {
    /// Compute the layout for `(dim, r_max)`. Returns an error if a single
    /// record would not fit in one page (the single-fault property requires
    /// `record_size ≤ PAGE_SIZE`).
    pub fn new(dim: usize, r_max: usize) -> Result<Self> {
        let vec_bytes = dim * 4;
        let adj_bytes = r_max * ADJ_ELEM_BYTES;
        let raw = NODE_ID_BYTES + vec_bytes + DEGREE_BYTES + adj_bytes;
        let record_size = raw.div_ceil(8) * 8; // align up to 8 bytes
        if record_size > PAGE_SIZE {
            return Err(VectorIndexError::IrrecoverableLoss {
                index: IndexId::ZERO,
                reason: format!(
                    "SSD DiskANN node-record ({record_size} B for dim={dim}, R={r_max}) \
                     exceeds PAGE_SIZE ({PAGE_SIZE} B): a node must fit in one page \
                     (ADR-195 §1 single-fault property)"
                ),
            });
        }
        let records_per_page = PAGE_SIZE / record_size; // ≥ 1 (checked above)
        Ok(Self {
            dim,
            r_max,
            vec_bytes,
            adj_bytes,
            record_size,
            records_per_page,
            vec_offset: NODE_ID_BYTES,
            degree_offset: NODE_ID_BYTES + vec_bytes,
            adj_offset: NODE_ID_BYTES + vec_bytes + DEGREE_BYTES,
        })
    }

    /// Page holding `slot`'s record.
    #[inline]
    #[must_use]
    pub fn page_of(&self, slot: u32) -> PageId {
        PageId::new(slot as u64 / self.records_per_page as u64)
    }

    /// Byte offset of `slot`'s record within its page.
    #[inline]
    #[must_use]
    pub fn offset_in_page(&self, slot: u32) -> usize {
        (slot as usize % self.records_per_page) * self.record_size
    }

    /// Total pages needed to store `n` records.
    #[inline]
    #[must_use]
    pub fn pages_for(&self, n: usize) -> usize {
        n.div_ceil(self.records_per_page)
    }

    /// On-disk byte footprint for `n` records (page-granular).
    #[inline]
    #[must_use]
    pub fn disk_bytes(&self, n: usize) -> u64 {
        self.pages_for(n) as u64 * PAGE_SIZE as u64
    }
}

/// Build-time configuration for [`SsdDiskAnnIndex::build`] (grouped so the build
/// entry point stays at a small, idiomatic argument count).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsdBuildConfig {
    /// Vector dimensionality.
    pub dim: usize,
    /// Distance metric (v1.0 SSD tier: [`Metric::L2`] only — the GA gate).
    pub metric: Metric,
    /// Vamana tuning params. **Must** be the dim-scaled params at 768d
    /// (`R = 128 / L_construction = 200`, the V-1 #740 measured curve) — the
    /// 128-d defaults are graph-starved at 768d (ADR-195 §2.1).
    pub params: DiskAnnParams,
    /// `BufferPool` frame count — the rerank cache RAM bound (ADR-195 §2).
    pub pool_frames: usize,
    /// Phase-1 over-fetch factor (see [`DEFAULT_RERANK_FACTOR`]).
    pub rerank_factor: usize,
    /// When `Some(batch)`, build the in-RAM nav graph with the rayon-parallel
    /// Vamana refinement (`DiskAnnGraph::build_owned_parallel`, ADR-195 §3 /
    /// #112) at the given batch size — required for the 10M build to be
    /// iterable. `None` = the deterministic single-threaded build.
    pub parallel_build_batch: Option<usize>,
}

/// Navigation quantizer for the in-RAM graph.
///
/// The variant is the encoding selection. `SsdBuildConfig` intentionally carries
/// no separate encoding field, so config and codebook cannot disagree.
#[derive(Debug, Clone, PartialEq)]
pub enum NavQuantizer {
    Sq8(Sq8Codebook),
    RaBitQ(RaBitQCodebook),
}

impl NavQuantizer {
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        match self {
            Self::Sq8(cb) => cb.dim(),
            Self::RaBitQ(cb) => cb.dim(),
        }
    }

    #[inline]
    #[must_use]
    pub const fn encoding(&self) -> Encoding {
        match self {
            Self::Sq8(_) => Encoding::Sq8,
            Self::RaBitQ(_) => Encoding::RaBitQ,
        }
    }
}

/// An SSD-resident DiskANN index: bounded-RAM SQ8 nav graph + on-disk f32 store.
///
/// Build with [`SsdDiskAnnIndex::build`]; query with
/// [`SsdDiskAnnIndex::search`]. The index keeps the f32 page store + its
/// `BufferPool` alive for the rerank read path.
pub struct SsdDiskAnnIndex {
    /// Navigation graph (vectors + adjacency) — in RAM.
    graph: DiskAnnGraph,
    /// Codebook to encode/prepare incoming queries into the nav space.
    nav: NavQuantizer,
    /// Bounded page cache over the f32 store (the rerank read path; its
    /// `frame_count` is the RAM-bounding knob, ADR-195 §2).
    pool: BufferPool,
    /// The underlying f32 page store (shared with `pool`).
    io: Arc<dyn PageIo>,
    /// On-disk record geometry.
    layout: RecordLayout,
    /// Vector dimensionality.
    dim: usize,
    /// Live vector count.
    n: usize,
    /// Phase-1 over-fetch factor for the rerank.
    rerank_factor: usize,
}

impl std::fmt::Debug for SsdDiskAnnIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `BufferPool` / `Arc<dyn PageIo>` are not `Debug`; surface the
        // serving-relevant scalars instead.
        f.debug_struct("SsdDiskAnnIndex")
            .field("n", &self.n)
            .field("dim", &self.dim)
            .field("record_size", &self.layout.record_size)
            .field("records_per_page", &self.layout.records_per_page)
            .field("rerank_factor", &self.rerank_factor)
            .field("disk_bytes", &self.disk_bytes())
            .finish()
    }
}

impl SsdDiskAnnIndex {
    /// Build the index from a (lazily-iterated) f32 vector source, streaming
    /// the f32 corpus to `path` in bounded batches while accumulating only the
    /// compressed nav copy in RAM (ADR-195 §3 — the bounded/disk-spilling build).
    ///
    /// `nav` is pre-trained by the caller on a representative sample
    /// (`Sq8Trainer`). `pool_frames` bounds the rerank cache RAM. `guard` is
    /// polled every `BUILD_GUARD_CHECK_EVERY` vectors for a clean abort.
    ///
    /// **Memory contract:** for the f32 corpus to stay off the heap, `vectors`
    /// MUST be a *lazy* iterator (a generator), not a fully-materialised `Vec`
    /// at 10M scale. Each `(VectorId, Vec<f32>)` is written to disk + encoded to
    /// nav bytes, then dropped; only the compressed accumulation grows.
    ///
    /// # Errors
    /// - [`VectorIndexError::DimensionMismatch`] if any vector's length ≠ `dim`
    ///   (or ≠ the codebook dim).
    /// - [`VectorIndexError::RssCapExceeded`] if the guard trips mid-build.
    /// - [`VectorIndexError::UnsupportedFlags`] for a non-L2 metric (v1.0 gate
    ///   is L2; IP/Cosine on the SSD tier are a follow-up).
    pub fn build<I>(
        path: &Path,
        config: &SsdBuildConfig,
        nav: NavQuantizer,
        vectors: I,
        guard: &RssGuard,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = (VectorId, Vec<f32>)>,
    {
        let &SsdBuildConfig {
            dim,
            metric,
            params,
            pool_frames,
            rerank_factor,
            parallel_build_batch,
        } = config;
        if metric != Metric::L2 {
            // The Vamana α-prune + the SQ8/f32 L2 kernels are L2-correct; the
            // GA gate is L2. IP/Cosine on the SSD tier is a documented
            // follow-up, mirroring DiskAnnGraph::new's IP rejection (#109).
            return Err(VectorIndexError::UnsupportedFlags {
                encoding: nav.encoding(),
                metric,
            });
        }
        if nav.dim() != dim {
            return Err(VectorIndexError::DimensionMismatch {
                expected: dim,
                got: nav.dim(),
            });
        }
        let layout = RecordLayout::new(dim, params.r as usize)?;
        let rerank_factor = rerank_factor.max(1);

        // The f32 page store (pread/pwrite — NOT mmap). Held as Arc<dyn PageIo>
        // so the BufferPool can share the exact same file for rerank reads.
        let io: Arc<dyn PageIo> = Arc::new(PosixPageIo::create(path).map_err(map_io_err)?);

        // Bounded-batch write + compressed-nav accumulation.
        let mut nav_accum: Vec<(VectorId, Vec<u8>)> = Vec::new();
        let mut page_buf = [0u8; PAGE_SIZE];
        let mut cur_page: u64 = 0;
        let mut have_page = false;
        let mut slot: u32 = 0;

        for (id, vec) in vectors {
            if vec.len() != dim {
                return Err(VectorIndexError::DimensionMismatch {
                    expected: dim,
                    got: vec.len(),
                });
            }
            let page = slot as u64 / layout.records_per_page as u64;
            if have_page && page != cur_page {
                io.write_page(PageId::new(cur_page), &page_buf)
                    .map_err(map_io_err)?;
                page_buf = [0u8; PAGE_SIZE];
            }
            cur_page = page;
            have_page = true;

            let off = layout.offset_in_page(slot);
            // node id
            page_buf[off..off + NODE_ID_BYTES]
                .copy_from_slice(&(u64::from(id.raw())).to_le_bytes());
            // f32 vector (LE bytes — exactly what the L2F32 kernel reads)
            write_f32_le(
                &mut page_buf[off + layout.vec_offset..off + layout.degree_offset],
                &vec,
            );
            // degree + adjacency left zero here; filled by finalize_adjacency.

            let encoded = match &nav {
                NavQuantizer::Sq8(codebook) => {
                    // SQ8 nav copy (i8 reinterpreted as the u8 the graph stores;
                    // the L2Sq8 kernel reads them back as i8).
                    sq8_i8_to_u8(&codebook.encode(&vec)?)
                }
                NavQuantizer::RaBitQ(codebook) => codebook.encode_aligned(&vec)?,
            };
            nav_accum.push((id, encoded));

            slot += 1;
            // f32 `vec` drops here — never accumulated.

            if slot as usize % BUILD_GUARD_CHECK_EVERY == 0 {
                guard.check()?;
            }
        }

        // Flush the final (partial) page.
        if have_page {
            io.write_page(PageId::new(cur_page), &page_buf)
                .map_err(map_io_err)?;
        }
        io.flush().map_err(map_io_err)?;
        guard.check()?;

        let n = slot as usize;

        // Build the in-RAM nav graph. `build_owned` MOVES the nav bytes into
        // the graph (no 2× copy at 10M; ADR-195 §3).
        let (encoding, kernel): (Encoding, Box<dyn DistanceKernel>) = match &nav {
            NavQuantizer::Sq8(_) => (Encoding::Sq8, Box::new(L2Sq8)),
            NavQuantizer::RaBitQ(_) => (Encoding::RaBitQ, Box::new(L2RaBitQSym::new(dim))),
        };
        let mut graph = DiskAnnGraph::new(params, encoding, metric, kernel)?;
        match parallel_build_batch {
            Some(batch) => graph.build_owned_parallel(nav_accum, batch)?,
            None => graph.build_owned(nav_accum)?,
        }
        guard.check()?;

        let pool = BufferPool::new(pool_frames.max(1), Arc::clone(&io));

        Ok(Self {
            graph,
            nav,
            pool,
            io,
            layout,
            dim,
            n,
            rerank_factor,
        })
    }

    /// Write each node's degree + adjacency into its on-disk record (a post-build
    /// pass — adjacency is unknown until the Vamana passes complete). This
    /// completes the durable §4.4 co-located format; the serving path reads
    /// adjacency from the in-RAM graph, so callers that only need recall/latency
    /// may skip this (it is a read-modify-write over every page).
    ///
    /// # Errors
    /// Propagates page I/O errors.
    pub fn finalize_adjacency(&self) -> Result<()> {
        let rpp = self.layout.records_per_page;
        let pages = self.layout.pages_for(self.n);
        let mut buf = [0u8; PAGE_SIZE];
        for page in 0..pages {
            let page_id = PageId::new(page as u64);
            self.io.read_page(page_id, &mut buf).map_err(map_io_err)?;
            for r in 0..rpp {
                let slot = page * rpp + r;
                if slot >= self.n {
                    break;
                }
                let off = r * self.layout.record_size;
                let neighbors = &self.graph.neighbors[slot];
                let degree = neighbors.len().min(self.layout.r_max) as u32;
                buf[off + self.layout.degree_offset..off + self.layout.adj_offset]
                    .copy_from_slice(&degree.to_le_bytes());
                for (i, &nb) in neighbors.iter().take(self.layout.r_max).enumerate() {
                    let a = off + self.layout.adj_offset + i * ADJ_ELEM_BYTES;
                    buf[a..a + ADJ_ELEM_BYTES].copy_from_slice(&nb.to_le_bytes());
                }
            }
            self.io.write_page(page_id, &buf).map_err(map_io_err)?;
        }
        self.io.flush().map_err(map_io_err)?;
        Ok(())
    }

    /// 2-phase top-`k` search at the index's DEFAULT search params (ADR-195 §2 /
    /// DiskANN §4.4): phase-1 beam width `max(rerank_factor·k, l_search_default)`
    /// and phase-2 rerank depth `rerank_factor·k`. A thin wrapper over
    /// [`SsdDiskAnnIndex::search_with_params`].
    ///
    /// # Errors
    /// - [`VectorIndexError::DimensionMismatch`] if `query.len() != dim`.
    /// - page I/O errors (mapped to [`VectorIndexError::IrrecoverableLoss`]).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>> {
        let rerank_k = (k * self.rerank_factor).max(k);
        let l_search = rerank_k.max(self.graph.params().l_search_default as usize);
        self.search_with_params(query, k, l_search, rerank_k)
    }

    /// 2-phase top-`k` search with EXPLICIT phase-1 beam width (`l_search`) and
    /// phase-2 rerank depth (`rerank_k`) — the knobs the ADR-189 §B recall/P95
    /// search-param sweep varies on a single built index WITHOUT a rebuild (the
    /// V-1 #740 `L_SEARCH` finding).
    ///
    /// Phase-1: encode the query to SQ8 and beam-search the in-RAM nav graph
    /// with beam width `l_search`. Phase-2: read each of the top `rerank_k`
    /// candidates' full f32 vector from the page store **through the
    /// `BufferPool`** (`pread`, NOT mmap), compute the exact L2 distance, and
    /// return the closest `k` as `(VectorId, exact_distance)` ascending.
    ///
    /// `rerank_k` is floored at `k` (never rerank fewer than the result count)
    /// and `l_search` is floored at `rerank_k` (the beam must surface at least
    /// the rerank set). The default [`SsdDiskAnnIndex::search`] is therefore
    /// exactly `search_with_params(q, k, rerank_k.max(l_search_default),
    /// rerank_k)`.
    ///
    /// # Errors
    /// - [`VectorIndexError::DimensionMismatch`] if `query.len() != dim`.
    /// - page I/O errors (mapped to [`VectorIndexError::IrrecoverableLoss`]).
    pub fn search_with_params(
        &self,
        query: &[f32],
        k: usize,
        l_search: usize,
        rerank_k: usize,
    ) -> Result<Vec<(VectorId, f32)>> {
        if query.len() != self.dim {
            return Err(VectorIndexError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 || self.n == 0 {
            return Ok(Vec::new());
        }
        let Some(entry) = self.graph.entry_point else {
            return Ok(Vec::new());
        };
        let rerank_k = rerank_k.max(k);
        let l_search = l_search.max(rerank_k).max(1);

        // Phase 1: compressed-nav beam-search in RAM at the requested beam
        // width. RaBitQ pays one D x D query rotation (~0.1-0.5 ms scalar at
        // 768d, ~3 KB transient) and then evaluates candidates from payloads.
        let visit = match &self.nav {
            NavQuantizer::Sq8(codebook) => {
                let q_i8 = codebook.encode(query)?;
                let q_sq8 = sq8_i8_to_u8(&q_i8);
                self.graph.greedy_visit_from(&q_sq8, entry, l_search)
            }
            NavQuantizer::RaBitQ(codebook) => {
                let prepared = codebook.prepare_query(query)?;
                self.graph.greedy_visit_rabitq(&prepared, entry, l_search)
            }
        };

        // Phase 2: exact f32 rerank, reading the full vectors from disk through
        // the bounded BufferPool.
        let q_f32 = f32_to_le_vec(query);
        let mut reranked: Vec<(VectorId, f32)> =
            Vec::with_capacity(rerank_k.min(visit.visited.len()));
        for &(slot, _) in visit.visited.iter().take(rerank_k) {
            let page_id = self.layout.page_of(slot);
            let off = self.layout.offset_in_page(slot) + self.layout.vec_offset;
            let guard = self.pool.pin_read(page_id).map_err(map_io_err)?;
            let page = guard.as_bytes();
            let dist = L2F32.distance(&page[off..off + self.layout.vec_bytes], &q_f32);
            // guard (pin) drops at end of iteration
            reranked.push((self.graph.ids[slot as usize], dist));
        }
        reranked.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.raw().cmp(&b.0.raw())));
        reranked.truncate(k);
        Ok(reranked)
    }

    /// Live vector count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// `true` if the index holds no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Vector dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The on-disk record geometry (for reporting / the disk budget).
    #[must_use]
    pub fn layout(&self) -> RecordLayout {
        self.layout
    }

    /// On-disk byte footprint of the f32 store (page-granular).
    #[must_use]
    pub fn disk_bytes(&self) -> u64 {
        self.layout.disk_bytes(self.n)
    }

    /// Borrow the in-RAM nav graph (diagnostics / tests).
    #[must_use]
    pub fn graph(&self) -> &DiskAnnGraph {
        &self.graph
    }

    /// Persist the in-RAM nav graph + codebook + serving meta to a `nav`
    /// sidecar so the index can be [`SsdDiskAnnIndex::open`]'d later WITHOUT
    /// re-running the (multi-hour at 10M) Vamana build. The f32 page store at
    /// the build `path` is the durable other half of the pair and is reopened
    /// in place by [`SsdDiskAnnIndex::open`]. mmap-free (PD#2) — buffered
    /// `Write` only.
    ///
    /// This is the enabler of the ADR-189 §B "reload + search-param sweep"
    /// instrument: build once → `save_nav` → re-measure recall/P95 across many
    /// `(l_search, rerank_k)` configs in fresh processes that each `open` in
    /// minutes rather than rebuilding.
    ///
    /// # Errors
    /// - [`VectorIndexError::IrrecoverableLoss`] if the graph carries
    ///   delta/tombstone state (the v1 nav format is bulk-only) or on write
    ///   error.
    pub fn save_nav(&self, nav_path: &Path) -> Result<()> {
        let f = std::fs::File::create(nav_path).map_err(persist::io_loss)?;
        let mut w = std::io::BufWriter::new(f);
        // File header.
        w.write_all(SSD_NAV_MAGIC).map_err(persist::io_loss)?;
        let version = match self.nav {
            NavQuantizer::Sq8(_) => SSD_NAV_VERSION_SQ8,
            NavQuantizer::RaBitQ(_) => SSD_NAV_VERSION,
        };
        persist::write_u32(&mut w, version)?;
        // Serving meta needed to reconstruct the index shell.
        persist::write_u64(&mut w, self.dim as u64)?;
        persist::write_u64(&mut w, self.n as u64)?;
        persist::write_u64(&mut w, self.rerank_factor as u64)?;
        persist::write_u64(&mut w, self.layout.r_max as u64)?;
        if version == SSD_NAV_VERSION {
            persist::write_u8(&mut w, nav_kind(&self.nav))?;
        }
        match &self.nav {
            NavQuantizer::Sq8(codebook) => {
                // Codebook (Sq8Params: per-dim scale + bias). This body is
                // byte-identical to v1 for Sq8 writers.
                let params = codebook.params();
                persist::write_u64(&mut w, params.scale.len() as u64)?;
                for &s in &params.scale {
                    persist::write_f32(&mut w, s)?;
                }
                for &b in &params.bias {
                    persist::write_f32(&mut w, b)?;
                }
            }
            NavQuantizer::RaBitQ(codebook) => {
                let params = codebook.params();
                persist::write_u64(&mut w, params.dim as u64)?;
                for &c in &params.centroid {
                    persist::write_f32(&mut w, c)?;
                }
                for &r in &params.rotation {
                    persist::write_f32(&mut w, r)?;
                }
            }
        }
        // Nav graph section (DiskAnnGraph::serialize_nav).
        self.graph.serialize_nav(&mut w)?;
        w.flush().map_err(persist::io_loss)?;
        Ok(())
    }

    /// Reopen a previously [`SsdDiskAnnIndex::save_nav`]'d index: reconstruct the
    /// in-RAM SQ8 nav graph + codebook from `nav_path` and reopen the f32 page
    /// store at `f32_path` in place (`PosixPageIo::open`, pread — NOT mmap,
    /// PD#2). No Vamana build runs; the wall-clock cost is the sidecar read +
    /// the in-RAM graph allocation.
    ///
    /// `pool_frames` sizes the rerank `BufferPool` (the RAM bound, ADR-195 §2);
    /// `guard` is polled once after the nav graph is resident.
    ///
    /// # Errors
    /// - [`VectorIndexError::IrrecoverableLoss`] on a bad/short sidecar, a
    ///   codebook/graph/store geometry mismatch, or page-store I/O failure.
    /// - [`VectorIndexError::RssCapExceeded`] if the guard trips.
    pub fn open(
        f32_path: &Path,
        nav_path: &Path,
        pool_frames: usize,
        guard: &RssGuard,
    ) -> Result<Self> {
        let f = std::fs::File::open(nav_path).map_err(persist::io_loss)?;
        let mut r = std::io::BufReader::new(f);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).map_err(persist::io_loss)?;
        if &magic != SSD_NAV_MAGIC {
            return Err(persist::fmt_loss("bad SSD nav sidecar magic"));
        }
        let version = persist::read_u32(&mut r)?;
        if !matches!(version, SSD_NAV_VERSION_SQ8 | SSD_NAV_VERSION) {
            return Err(persist::fmt_loss(format!(
                "unsupported SSD nav sidecar version {version} (expected 1 or {SSD_NAV_VERSION})"
            )));
        }
        let dim = persist::read_u64(&mut r)? as usize;
        let n = persist::read_u64(&mut r)? as usize;
        let rerank_factor = (persist::read_u64(&mut r)? as usize).max(1);
        let r_max = persist::read_u64(&mut r)? as usize;

        let kind = if version == SSD_NAV_VERSION_SQ8 {
            SSD_NAV_KIND_SQ8
        } else {
            persist::read_u8(&mut r)?
        };
        let nav = read_nav_quantizer(&mut r, kind, dim)?;
        let kernel: Box<dyn DistanceKernel> = match &nav {
            NavQuantizer::Sq8(_) => Box::new(L2Sq8),
            NavQuantizer::RaBitQ(_) => Box::new(L2RaBitQSym::new(dim)),
        };
        let graph = DiskAnnGraph::deserialize_nav(&mut r, kernel)?;
        if graph.encoding() != nav.encoding() || graph.metric() != Metric::L2 {
            return Err(persist::fmt_loss(format!(
                "nav kind {:?} does not match graph encoding {:?} / metric {:?}",
                nav.encoding(),
                graph.encoding(),
                graph.metric()
            )));
        }
        if graph.main_len() != n {
            return Err(persist::fmt_loss(format!(
                "nav graph node count {} != meta n {n}",
                graph.main_len()
            )));
        }

        // Reopen the f32 page store in place + sanity-check its geometry against
        // the stored (dim, R, n) — a too-small store would read garbage / panic.
        let layout = RecordLayout::new(dim, r_max)?;
        let posix = PosixPageIo::open(f32_path).map_err(map_io_err)?;
        let expected = layout.disk_bytes(n);
        let actual = posix.file_len().map_err(map_io_err)?;
        if actual < expected {
            return Err(persist::fmt_loss(format!(
                "f32 store {} too small: {actual} B < expected {expected} B \
                 (n={n}, dim={dim}, R={r_max})",
                f32_path.display()
            )));
        }
        let io: Arc<dyn PageIo> = Arc::new(posix);
        let pool = BufferPool::new(pool_frames.max(1), Arc::clone(&io));
        guard.check()?;

        Ok(Self {
            graph,
            nav,
            pool,
            io,
            layout,
            dim,
            n,
            rerank_factor,
        })
    }
}

/// Reinterpret an SQ8 `i8` code vector as the `u8` bytes the graph stores. The
/// [`L2Sq8`] kernel reads them back as `i8` (per #116: the codec emits `i8`
/// directly, the kernel's signed-SIMD path reads `i8`).
#[inline]
fn sq8_i8_to_u8(q: &[i8]) -> Vec<u8> {
    q.iter().map(|&b| b as u8).collect()
}

#[inline]
const fn nav_kind(nav: &NavQuantizer) -> u8 {
    match nav {
        NavQuantizer::Sq8(_) => SSD_NAV_KIND_SQ8,
        NavQuantizer::RaBitQ(_) => SSD_NAV_KIND_RABITQ,
    }
}

fn read_nav_quantizer(r: &mut dyn Read, kind: u8, dim: usize) -> Result<NavQuantizer> {
    match kind {
        SSD_NAV_KIND_SQ8 => {
            let cb_dim = persist::read_u64(r)? as usize;
            if cb_dim != dim {
                return Err(persist::fmt_loss(format!(
                    "codebook dim {cb_dim} != index dim {dim}"
                )));
            }
            let mut scale = Vec::with_capacity(cb_dim.min(1 << 20));
            for _ in 0..cb_dim {
                scale.push(persist::read_f32(r)?);
            }
            let mut bias = Vec::with_capacity(cb_dim.min(1 << 20));
            for _ in 0..cb_dim {
                bias.push(persist::read_f32(r)?);
            }
            Ok(NavQuantizer::Sq8(Sq8Codebook::from_params(
                Sq8Params::try_new(scale, bias)?,
            )))
        }
        SSD_NAV_KIND_RABITQ => {
            let cb_dim = persist::read_u64(r)? as usize;
            if cb_dim != dim {
                return Err(persist::fmt_loss(format!(
                    "RaBitQ codebook dim {cb_dim} != index dim {dim}"
                )));
            }
            let mut centroid = Vec::with_capacity(cb_dim.min(1 << 20));
            for _ in 0..cb_dim {
                centroid.push(persist::read_f32(r)?);
            }
            let rot_len = cb_dim
                .checked_mul(cb_dim)
                .ok_or_else(|| persist::fmt_loss("RaBitQ rotation dimension overflow"))?;
            let mut rotation = Vec::with_capacity(rot_len.min(1 << 20));
            for _ in 0..rot_len {
                rotation.push(persist::read_f32(r)?);
            }
            Ok(NavQuantizer::RaBitQ(RaBitQCodebook::from_params(
                RaBitQParams::try_new(cb_dim, centroid, rotation)?,
            )))
        }
        other => Err(persist::fmt_loss(format!(
            "unknown SSD nav quantizer tag {other}"
        ))),
    }
}

/// f32 slice → little-endian byte vec (the layout the `L2F32` kernel decodes;
/// LE-only targets, byte-identical to `bytemuck::cast_slice`, matching the V-1
/// recall oracle's encoder).
#[inline]
fn f32_to_le_vec(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Write an f32 slice as little-endian bytes into `dst` (`dst.len()` must be
/// `v.len() * 4`).
#[inline]
fn write_f32_le(dst: &mut [u8], v: &[f32]) {
    debug_assert_eq!(dst.len(), v.len() * 4);
    for (i, &x) in v.iter().enumerate() {
        dst[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
    }
}

/// Map an `arcgraph_storage` (workspace) error from the page store into the
/// codec-local [`VectorIndexError`] at this consuming boundary (per
/// `docs/codec-error-translation.md`).
fn map_io_err(e: arcgraph_core::ArcGraphError) -> VectorIndexError {
    VectorIndexError::IrrecoverableLoss {
        index: IndexId::ZERO,
        reason: format!("SSD DiskANN page store I/O failed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantizer::{RaBitQTrainer, Sq8Trainer};
    use std::collections::HashSet;

    /// Deterministic xorshift32 (no `rand` dep; mirrors the build/bench PRNG).
    struct Xs32(u32);
    impl Xs32 {
        fn new(s: u32) -> Self {
            Self(if s == 0 { 0xDEAD_BEEF } else { s })
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }
        fn gauss(&mut self) -> f32 {
            let u1 = (self.next_u32() as f32 / u32::MAX as f32).max(1e-10);
            let u2 = self.next_u32() as f32 / u32::MAX as f32;
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        }
        fn signed(&mut self) -> f32 {
            (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// `(corpus points, cluster centers)`.
    type ClusterCorpus = (Vec<(VectorId, Vec<f32>)>, Vec<Vec<f32>>);

    /// Gaussian-cluster corpus (the ADR-189 §9 / V-1 shape, scaled down).
    fn cluster_corpus(
        seed: u32,
        clusters: usize,
        per: usize,
        dim: usize,
        sigma: f32,
    ) -> ClusterCorpus {
        let mut rng = Xs32::new(seed);
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|_| (0..dim).map(|_| rng.signed()).collect())
            .collect();
        let mut out = Vec::with_capacity(clusters * per);
        let mut id = 0u32;
        for c in &centers {
            for _ in 0..per {
                let v: Vec<f32> = c.iter().map(|&cc| cc + rng.gauss() * sigma).collect();
                out.push((VectorId::new(id), v));
                id += 1;
            }
        }
        (out, centers)
    }

    fn brute_force_top_k(data: &[(VectorId, Vec<f32>)], q: &[f32], k: usize) -> Vec<VectorId> {
        let qb = f32_to_le_vec(q);
        let mut all: Vec<(VectorId, f32)> = data
            .iter()
            .map(|(id, v)| (*id, L2F32.distance(&f32_to_le_vec(v), &qb)))
            .collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1));
        all.into_iter().take(k).map(|(id, _)| id).collect()
    }

    fn recall_at_10_with<F>(
        idx: &SsdDiskAnnIndex,
        corpus: &[(VectorId, Vec<f32>)],
        queries: &[Vec<f32>],
        mut search: F,
    ) -> f64
    where
        F: FnMut(&SsdDiskAnnIndex, &[f32]) -> Vec<(VectorId, f32)>,
    {
        let mut hits = 0usize;
        for q in queries {
            let gt: HashSet<u32> = brute_force_top_k(corpus, q, 10)
                .into_iter()
                .map(|id| id.raw())
                .collect();
            let got = search(idx, q);
            hits += got.iter().filter(|(id, _)| gt.contains(&id.raw())).count();
        }
        hits as f64 / (queries.len() * 10) as f64
    }

    fn train_codebook(data: &[(VectorId, Vec<f32>)]) -> Sq8Codebook {
        let refs: Vec<&[f32]> = data.iter().map(|(_, v)| v.as_slice()).collect();
        Sq8Trainer.train(&refs).expect("train sq8")
    }

    fn train_rabitq(data: &[(VectorId, Vec<f32>)]) -> RaBitQCodebook {
        let refs: Vec<&[f32]> = data.iter().map(|(_, v)| v.as_slice()).collect();
        RaBitQTrainer
            .train(&refs, 0x7580_0002)
            .expect("train rabitq")
    }

    /// Test helper: build an SSD index with L2 + default params + the given
    /// pool frames (groups the `SsdBuildConfig` so call sites stay terse).
    fn build_small(
        path: &std::path::Path,
        dim: usize,
        pool_frames: usize,
        codebook: Sq8Codebook,
        vectors: Vec<(VectorId, Vec<f32>)>,
        guard: &RssGuard,
    ) -> Result<SsdDiskAnnIndex> {
        build_small_nav(
            path,
            dim,
            pool_frames,
            NavQuantizer::Sq8(codebook),
            vectors,
            guard,
        )
    }

    fn build_small_nav(
        path: &std::path::Path,
        dim: usize,
        pool_frames: usize,
        nav: NavQuantizer,
        vectors: Vec<(VectorId, Vec<f32>)>,
        guard: &RssGuard,
    ) -> Result<SsdDiskAnnIndex> {
        let cfg = SsdBuildConfig {
            dim,
            metric: Metric::L2,
            params: DiskAnnParams::default(),
            pool_frames,
            rerank_factor: DEFAULT_RERANK_FACTOR,
            parallel_build_batch: None,
        };
        SsdDiskAnnIndex::build(path, &cfg, nav, vectors, guard)
    }

    fn assert_ssd_sidecar_version(path: &std::path::Path, version: u32) {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[..SSD_NAV_MAGIC.len()], SSD_NAV_MAGIC);
        assert_eq!(
            u32::from_le_bytes(
                bytes[SSD_NAV_MAGIC.len()..SSD_NAV_MAGIC.len() + 4]
                    .try_into()
                    .unwrap()
            ),
            version
        );
    }

    #[test]
    fn record_layout_768_r128_packs_two_per_page() {
        let l = RecordLayout::new(768, 128).unwrap();
        assert_eq!(l.vec_bytes, 768 * 4);
        // 8 + 3072 + 4 + 512 = 3596 → align8 → 3600.
        assert_eq!(l.record_size, 3600);
        assert_eq!(l.records_per_page, 2);
        // Offsets co-locate vector then adjacency (the §4.4 layout).
        assert_eq!(l.vec_offset, 8);
        assert_eq!(l.degree_offset, 8 + 3072);
        assert_eq!(l.adj_offset, 8 + 3072 + 4);
        // 10M nodes ≈ 40 GB (the ADR-189 §B budget; 1/page would be 80 GB).
        assert_eq!(l.pages_for(10_000_000), 5_000_000);
        assert_eq!(l.disk_bytes(10_000_000), 5_000_000u64 * 8192);
    }

    #[test]
    fn record_layout_rejects_oversize_node() {
        // dim so large that one record exceeds a page → single-fault property
        // cannot hold.
        let err = RecordLayout::new(4000, 128).unwrap_err();
        assert!(matches!(err, VectorIndexError::IrrecoverableLoss { .. }));
    }

    #[test]
    fn build_and_search_recall_vs_brute_force() {
        // Small in-distribution recall check: the SSD 2-phase search must match
        // the exhaustive brute-force oracle at recall@10 ≥ 0.95 (the gate's
        // shape, at unit scale). dim=64 with the default params is well-posed.
        let dim = 64;
        let (corpus, centers) = cluster_corpus(7, 20, 100, dim, 0.03); // 2000 pts
        let codebook = train_codebook(&corpus);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let idx = build_small(tmp.path(), dim, 512, codebook, corpus.clone(), &guard).unwrap();
        assert_eq!(idx.len(), 2000);

        // In-distribution queries from the same generative process.
        let mut rng = Xs32::new(999);
        let k = 10;
        let n_q = 50;
        let mut hits = 0usize;
        for _ in 0..n_q {
            let c = &centers[(rng.next_u32() as usize) % centers.len()];
            let q: Vec<f32> = c.iter().map(|&cc| cc + rng.gauss() * 0.03).collect();
            let gt: HashSet<u32> = brute_force_top_k(&corpus, &q, k)
                .into_iter()
                .map(|v| v.raw())
                .collect();
            let got = idx.search(&q, k).unwrap();
            for (id, _) in &got {
                if gt.contains(&id.raw()) {
                    hits += 1;
                }
            }
        }
        let recall = hits as f64 / (k * n_q) as f64;
        assert!(recall >= 0.95, "SSD recall@10 = {recall:.4} < 0.95");
    }

    #[test]
    fn rabitq_build_and_search_recall_vs_brute_force() {
        let dim = 128;
        let (corpus, centers) = cluster_corpus(758, 20, 100, dim, 0.02);
        let codebook = train_rabitq(&corpus);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let idx = build_small_nav(
            tmp.path(),
            dim,
            512,
            NavQuantizer::RaBitQ(codebook),
            corpus.clone(),
            &guard,
        )
        .unwrap();
        assert_eq!(idx.graph().encoding(), Encoding::RaBitQ);
        assert_eq!(
            idx.graph().bytes_per_vector().unwrap(),
            Encoding::RaBitQ.bytes_per_vector_aligned(dim)
        );

        let mut rng = Xs32::new(9758);
        let k = 10;
        let n_q = 40;
        let mut hits = 0usize;
        for _ in 0..n_q {
            let c = &centers[(rng.next_u32() as usize) % centers.len()];
            let q: Vec<f32> = c.iter().map(|&cc| cc + rng.gauss() * 0.02).collect();
            let gt: HashSet<u32> = brute_force_top_k(&corpus, &q, k)
                .into_iter()
                .map(|v| v.raw())
                .collect();
            let got = idx.search(&q, k).unwrap();
            for (id, _) in &got {
                if gt.contains(&id.raw()) {
                    hits += 1;
                }
            }
        }
        let recall = hits as f64 / (k * n_q) as f64;
        eprintln!("W3_RABITQ_DEFAULT_SEARCH recall@10={recall:.4} pin=0.50");
        assert!(recall >= 0.50, "RaBitQ SSD recall@10 = {recall:.4} < 0.50");
    }

    #[test]
    fn rabitq_recall_tracks_sq8_at_equal_rerank() {
        let dim = 128;
        let (corpus, centers) = cluster_corpus(0xA209_0004, 16, 60, dim, 0.025);
        let guard = RssGuard::disabled();
        let sq8 = train_codebook(&corpus);
        let rabitq = train_rabitq(&corpus);
        let sq8_store = tempfile::NamedTempFile::new().unwrap();
        let rab_store = tempfile::NamedTempFile::new().unwrap();
        let sq8_idx = build_small_nav(
            sq8_store.path(),
            dim,
            512,
            NavQuantizer::Sq8(sq8),
            corpus.clone(),
            &guard,
        )
        .unwrap();
        let rab_idx = build_small_nav(
            rab_store.path(),
            dim,
            512,
            NavQuantizer::RaBitQ(rabitq),
            corpus.clone(),
            &guard,
        )
        .unwrap();

        let queries: Vec<Vec<f32>> = centers.iter().take(16).cloned().collect();
        let sq8_recall = recall_at_10_with(&sq8_idx, &corpus, &queries, |idx, q| {
            idx.search_with_params(q, 10, 400, 100).unwrap()
        });
        let rabitq_recall = recall_at_10_with(&rab_idx, &corpus, &queries, |idx, q| {
            idx.search_with_params(q, 10, 400, 100).unwrap()
        });
        eprintln!(
            "W4_RABITQ_SQ8_PARITY sq8={sq8_recall:.4} rabitq={rabitq_recall:.4} beam=400 rerank=100 band=0.02"
        );
        assert!(
            sq8_recall >= 0.95,
            "W4 fixture invalid: SQ8 recall@10 {sq8_recall:.4} < 0.95"
        );
        assert!(
            rabitq_recall + 0.02 >= sq8_recall,
            "W4 RaBitQ recall@10 {rabitq_recall:.4} trails SQ8 {sq8_recall:.4} by > 0.02"
        );
    }

    #[test]
    fn rabitq_self_query_exact_rerank_and_prepared_seam() {
        let dim = 128;
        let (corpus, _) = cluster_corpus(759, 12, 60, dim, 0.02);
        let codebook = train_rabitq(&corpus);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let idx = build_small_nav(
            tmp.path(),
            dim,
            256,
            NavQuantizer::RaBitQ(codebook.clone()),
            corpus.clone(),
            &guard,
        )
        .unwrap();

        let query = &corpus[123].1;
        let got = idx.search_with_params(query, 1, 500, 300).unwrap();
        assert_eq!(got[0].0, corpus[123].0);
        assert!(got[0].1 < 1e-5, "exact rerank distance {}", got[0].1);

        let prepared = codebook.prepare_query(query).unwrap();
        let entry = idx.graph.entry_point.unwrap();
        let visit = idx.graph.greedy_visit_rabitq(&prepared, entry, 50);
        let non_self = visit
            .visited
            .iter()
            .find(|(slot, _)| idx.graph.ids[*slot as usize] != corpus[123].0)
            .copied()
            .unwrap();
        let external =
            crate::quantizer::estimate_l2_sq(&prepared, idx.graph.vector_bytes(non_self.0));
        assert_eq!(non_self.1, external);
    }

    #[test]
    fn search_distances_are_exact_f32_not_sq8() {
        // The returned distances must be the EXACT f32 rerank distances (read
        // from disk), not the SQ8 phase-1 keys. Verify the top-1 distance
        // equals the brute-force exact L2 to that id.
        let dim = 32;
        let (corpus, _) = cluster_corpus(3, 10, 50, dim, 0.05);
        let codebook = train_codebook(&corpus);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let idx = build_small(tmp.path(), dim, 256, codebook, corpus.clone(), &guard).unwrap();
        let q = corpus[123].1.clone();
        let got = idx.search(&q, 1).unwrap();
        assert_eq!(got.len(), 1);
        let (top_id, top_dist) = got[0];
        // Exact distance from the page-store f32 to the query.
        let exact = L2F32.distance(
            &f32_to_le_vec(&corpus.iter().find(|(id, _)| *id == top_id).unwrap().1),
            &f32_to_le_vec(&q),
        );
        assert!(
            (top_dist - exact).abs() < 1e-3,
            "rerank dist {top_dist} != exact {exact}"
        );
        // Querying a corpus member should return itself as top-1 with ~0 dist.
        assert!(
            top_dist < 1e-2,
            "self-query top-1 distance {top_dist} not ~0"
        );
    }

    #[test]
    fn finalize_adjacency_roundtrips_vector_and_adjacency_on_disk() {
        // The durable §4.4 format: after finalize, a raw page read returns each
        // node's f32 vector AND its adjacency (degree + neighbor slots).
        let dim = 16;
        let (corpus, _) = cluster_corpus(5, 8, 20, dim, 0.05);
        let codebook = train_codebook(&corpus);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let idx = build_small(tmp.path(), dim, 128, codebook, corpus.clone(), &guard).unwrap();
        idx.finalize_adjacency().unwrap();

        let layout = idx.layout();
        let mut buf = [0u8; PAGE_SIZE];
        // Check slot 5: its on-disk vector + adjacency match the in-RAM graph.
        let slot = 5u32;
        idx.io.read_page(layout.page_of(slot), &mut buf).unwrap();
        let off = layout.offset_in_page(slot);
        // node id
        let id = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
        assert_eq!(id, u64::from(corpus[slot as usize].0.raw()));
        // f32 vector
        for d in 0..dim {
            let b = off + layout.vec_offset + d * 4;
            let x = f32::from_le_bytes(buf[b..b + 4].try_into().unwrap());
            assert!((x - corpus[slot as usize].1[d]).abs() < 1e-6);
        }
        // degree + adjacency match the graph
        let deg = u32::from_le_bytes(
            buf[off + layout.degree_offset..off + layout.adj_offset]
                .try_into()
                .unwrap(),
        ) as usize;
        assert_eq!(
            deg,
            idx.graph().neighbors[slot as usize].len().min(layout.r_max)
        );
        for (i, &nb) in idx.graph().neighbors[slot as usize]
            .iter()
            .take(deg)
            .enumerate()
        {
            let a = off + layout.adj_offset + i * 4;
            let on_disk = u32::from_le_bytes(buf[a..a + 4].try_into().unwrap());
            assert_eq!(on_disk, nb);
        }
    }

    #[test]
    fn build_aborts_cleanly_when_rss_guard_trips() {
        // Fault injection (ADR-195 §2.2): arm the guard with a 0 MB cap so it
        // trips against the live process; the build must return a CLEAN
        // RssCapExceeded at a checkpoint — not OOM-kill. We use a corpus large
        // enough to cross a BUILD_GUARD_CHECK_EVERY boundary.
        let dim = 8;
        let (corpus, _) = cluster_corpus(1, 50, 200, dim, 0.05); // 10_000 > 4096
        let codebook = train_codebook(&corpus);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::spawn(0, std::time::Duration::from_millis(5));
        // Give the sampler a moment to latch before the build reaches a check.
        std::thread::sleep(std::time::Duration::from_millis(40));
        let res = build_small(tmp.path(), dim, 64, codebook, corpus, &guard);
        match res {
            Err(VectorIndexError::RssCapExceeded { cap_mb, .. }) => assert_eq!(cap_mb, 0),
            other => panic!("expected clean RssCapExceeded abort, got {other:?}"),
        }
        // Process still alive — fail-CLEAN proven.
    }

    #[test]
    fn empty_build_is_empty_and_search_returns_nothing() {
        let dim = 8;
        let codebook = {
            // Train on a throwaway sample so the codebook is valid.
            let sample = vec![(VectorId::new(0), vec![0.1f32; dim])];
            train_codebook(&sample)
        };
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let idx = build_small(tmp.path(), dim, 16, codebook, Vec::new(), &guard).unwrap();
        assert!(idx.is_empty());
        assert!(idx.search(&vec![0.0; dim], 10).unwrap().is_empty());
    }

    #[test]
    fn search_rejects_dim_mismatch() {
        let dim = 8;
        let (corpus, _) = cluster_corpus(2, 4, 10, dim, 0.05);
        let codebook = train_codebook(&corpus);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let idx = build_small(tmp.path(), dim, 64, codebook, corpus, &guard).unwrap();
        let err = idx.search(&vec![0.0; dim + 1], 5).unwrap_err();
        assert!(matches!(err, VectorIndexError::DimensionMismatch { .. }));
    }

    #[test]
    fn search_with_params_matches_default_search() {
        // Refactor-safety oracle: the default `search(q, k)` must be EXACTLY
        // `search_with_params(q, k, rerank_k.max(l_search_default), rerank_k)`
        // with `rerank_k = k·rerank_factor` — bit-identical ids AND distances.
        let dim = 48;
        let (corpus, centers) = cluster_corpus(11, 24, 80, dim, 0.03); // ~1920 pts
        let codebook = train_codebook(&corpus);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let idx = build_small(tmp.path(), dim, 512, codebook, corpus, &guard).unwrap();

        let l_default = idx.graph().params().l_search_default as usize;
        let mut rng = Xs32::new(4242);
        for _ in 0..40 {
            let c = &centers[(rng.next_u32() as usize) % centers.len()];
            let q: Vec<f32> = c.iter().map(|&cc| cc + rng.gauss() * 0.03).collect();
            for k in [1usize, 5, 10] {
                let rerank_k = (k * DEFAULT_RERANK_FACTOR).max(k);
                let l_search = rerank_k.max(l_default);
                let a = idx.search(&q, k).unwrap();
                let b = idx.search_with_params(&q, k, l_search, rerank_k).unwrap();
                assert_eq!(a, b, "default search != explicit-param search (k={k})");
            }
        }
    }

    #[test]
    fn save_open_roundtrip_returns_byte_identical_search() {
        // The determinism-equality oracle for persistence (doctrine "strong
        // oracles"): a reopened index must return BYTE-IDENTICAL results
        // (VectorIds AND exact f32 distances) to the original, across many
        // queries × several (l_search, rerank_k) configs. A weaker recall-parity
        // check could pass on a subtly-wrong reconstruction; bit-equality cannot.
        let dim = 56;
        let (corpus, centers) = cluster_corpus(21, 30, 90, dim, 0.025); // ~2700 pts
        let codebook = train_codebook(&corpus);
        let store = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let orig = build_small(store.path(), dim, 512, codebook, corpus, &guard).unwrap();

        let nav = tempfile::NamedTempFile::new().unwrap();
        orig.save_nav(nav.path()).unwrap();
        assert_ssd_sidecar_version(nav.path(), SSD_NAV_VERSION_SQ8);
        let reopened = SsdDiskAnnIndex::open(store.path(), nav.path(), 512, &guard).unwrap();

        assert_eq!(orig.len(), reopened.len());
        assert_eq!(orig.dim(), reopened.dim());
        assert_eq!(orig.layout(), reopened.layout());
        assert_eq!(
            orig.graph().params().l_search_default,
            reopened.graph().params().l_search_default
        );

        let mut rng = Xs32::new(7);
        let configs = [(10usize, 10usize), (50, 20), (100, 50), (200, 100)];
        for _ in 0..60 {
            let c = &centers[(rng.next_u32() as usize) % centers.len()];
            let q: Vec<f32> = c.iter().map(|&cc| cc + rng.gauss() * 0.025).collect();
            for &(ls, rk) in &configs {
                let a = orig.search_with_params(&q, 10, ls, rk).unwrap();
                let b = reopened.search_with_params(&q, 10, ls, rk).unwrap();
                assert_eq!(
                    a, b,
                    "reopened search != original (l_search={ls}, rerank_k={rk})"
                );
            }
            // The default path too.
            assert_eq!(
                orig.search(&q, 10).unwrap(),
                reopened.search(&q, 10).unwrap()
            );
        }
    }

    #[test]
    fn rabitq_save_open_roundtrip_returns_byte_identical_search() {
        let dim = 56;
        let (corpus, centers) = cluster_corpus(2109, 30, 90, dim, 0.025);
        let codebook = train_rabitq(&corpus);
        let store = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let orig = build_small_nav(
            store.path(),
            dim,
            512,
            NavQuantizer::RaBitQ(codebook),
            corpus,
            &guard,
        )
        .unwrap();

        let nav = tempfile::NamedTempFile::new().unwrap();
        orig.save_nav(nav.path()).unwrap();
        assert_ssd_sidecar_version(nav.path(), SSD_NAV_VERSION);
        let reopened = SsdDiskAnnIndex::open(store.path(), nav.path(), 512, &guard).unwrap();
        assert_eq!(reopened.graph().encoding(), Encoding::RaBitQ);

        let mut rng = Xs32::new(77);
        let configs = [(10usize, 10usize), (50, 20), (100, 50)];
        for _ in 0..40 {
            let c = &centers[(rng.next_u32() as usize) % centers.len()];
            let q: Vec<f32> = c.iter().map(|&cc| cc + rng.gauss() * 0.025).collect();
            for &(ls, rk) in &configs {
                let a = orig.search_with_params(&q, 10, ls, rk).unwrap();
                let b = reopened.search_with_params(&q, 10, ls, rk).unwrap();
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn rabitq_open_rejects_corrupt_rotation() {
        let dim = 16;
        let (corpus, _) = cluster_corpus(901, 8, 20, dim, 0.04);
        let codebook = train_rabitq(&corpus);
        let store = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let idx = build_small_nav(
            store.path(),
            dim,
            128,
            NavQuantizer::RaBitQ(codebook),
            corpus,
            &guard,
        )
        .unwrap();
        let nav = tempfile::NamedTempFile::new().unwrap();
        idx.save_nav(nav.path()).unwrap();

        let mut bytes = std::fs::read(nav.path()).unwrap();
        let rotation_off = SSD_NAV_MAGIC.len() + 4 + 8 + 8 + 8 + 8 + 1 + 8 + dim * 4;
        bytes[rotation_off..rotation_off + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(nav.path(), bytes).unwrap();
        assert!(SsdDiskAnnIndex::open(store.path(), nav.path(), 128, &guard).is_err());
    }

    #[test]
    fn rabitq_open_rejects_nav_kind_graph_encoding_mismatch() {
        let dim = 16;
        let (corpus, _) = cluster_corpus(902, 8, 20, dim, 0.04);
        let codebook = train_rabitq(&corpus);
        let store = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let idx = build_small_nav(
            store.path(),
            dim,
            128,
            NavQuantizer::RaBitQ(codebook),
            corpus,
            &guard,
        )
        .unwrap();
        let nav = tempfile::NamedTempFile::new().unwrap();
        idx.save_nav(nav.path()).unwrap();

        let mut bytes = std::fs::read(nav.path()).unwrap();
        let graph_magic = bytes
            .windows(persist::NAV_MAGIC.len())
            .position(|w| w == persist::NAV_MAGIC)
            .unwrap();
        let graph_encoding_tag = graph_magic + persist::NAV_MAGIC.len() + 4;
        bytes[graph_encoding_tag] = SSD_NAV_KIND_SQ8;
        std::fs::write(nav.path(), bytes).unwrap();
        assert!(SsdDiskAnnIndex::open(store.path(), nav.path(), 128, &guard).is_err());
    }

    #[test]
    #[ignore = "RaBitQ/SQ8 nav parity instrument; opt in with ARCGRAPH_RABITQ_NAV_BENCH_OK=1"]
    fn rabitq_nav_parity() {
        if std::env::var("ARCGRAPH_RABITQ_NAV_BENCH_OK").as_deref() != Ok("1") {
            panic!("ARCGRAPH_RABITQ_NAV_BENCH_OK must be 1 for rabitq_nav_parity (heavy, opt-in)");
        }
        let dim = std::env::var("ARCGRAPH_RABITQ_NAV_DIM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
        let clusters = std::env::var("ARCGRAPH_RABITQ_NAV_CLUSTERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(40);
        let per = std::env::var("ARCGRAPH_RABITQ_NAV_PER_CLUSTER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let (corpus, centers) = cluster_corpus(0x758, clusters, per, dim, 0.025);
        let guard = RssGuard::disabled();
        let sq8 = train_codebook(&corpus);
        let rabitq = train_rabitq(&corpus);
        let sq8_store = tempfile::NamedTempFile::new().unwrap();
        let rab_store = tempfile::NamedTempFile::new().unwrap();
        let t = std::time::Instant::now();
        let sq8_idx = build_small_nav(
            sq8_store.path(),
            dim,
            1024,
            NavQuantizer::Sq8(sq8),
            corpus.clone(),
            &guard,
        )
        .unwrap();
        let sq8_secs = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let rab_idx = build_small_nav(
            rab_store.path(),
            dim,
            1024,
            NavQuantizer::RaBitQ(rabitq),
            corpus.clone(),
            &guard,
        )
        .unwrap();
        let rab_secs = t.elapsed().as_secs_f64();
        let queries: Vec<Vec<f32>> = centers.iter().take(20).cloned().collect();
        let gt: Vec<HashSet<u32>> = queries
            .iter()
            .map(|q| {
                brute_force_top_k(&corpus, q, 10)
                    .into_iter()
                    .map(|id| id.raw())
                    .collect()
            })
            .collect();
        eprintln!(
            "[RABITQ_NAV] N={} dim={dim} sq8_bpv={} rabitq_bpv={} sq8_build={sq8_secs:.2}s rabitq_build={rab_secs:.2}s",
            corpus.len(),
            sq8_idx.graph().bytes_per_vector().unwrap(),
            rab_idx.graph().bytes_per_vector().unwrap()
        );
        eprintln!("encoding,l_search,rerank_k,recall@10");
        for &(l, rk) in &[(400usize, 100usize), (800, 200)] {
            for (label, idx) in [("sq8", &sq8_idx), ("rabitq", &rab_idx)] {
                let mut hits = 0usize;
                for (q, gt_set) in queries.iter().zip(&gt) {
                    let got = idx.search_with_params(q, 10, l, rk).unwrap();
                    hits += got
                        .iter()
                        .filter(|(id, _)| gt_set.contains(&id.raw()))
                        .count();
                }
                let recall = hits as f64 / (queries.len() * 10) as f64;
                eprintln!("{label},{l},{rk},{recall:.4}");
            }
        }
    }

    #[test]
    fn open_rejects_bad_magic() {
        let store = tempfile::NamedTempFile::new().unwrap();
        let nav = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            nav.path(),
            b"NOTAMAGICnav-bytes-that-are-not-a-valid-sidecar",
        )
        .unwrap();
        let guard = RssGuard::disabled();
        let err = SsdDiskAnnIndex::open(store.path(), nav.path(), 16, &guard).unwrap_err();
        assert!(matches!(err, VectorIndexError::IrrecoverableLoss { .. }));
    }

    #[test]
    fn open_rejects_truncated_sidecar() {
        // A sidecar truncated mid-graph must fail-clean (read_exact EOF), not
        // panic or silently load a partial graph.
        let dim = 16;
        let (corpus, _) = cluster_corpus(9, 10, 40, dim, 0.04);
        let codebook = train_codebook(&corpus);
        let store = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let idx = build_small(store.path(), dim, 128, codebook, corpus, &guard).unwrap();
        let nav = tempfile::NamedTempFile::new().unwrap();
        idx.save_nav(nav.path()).unwrap();

        let full = std::fs::read(nav.path()).unwrap();
        std::fs::write(nav.path(), &full[..full.len() / 2]).unwrap();
        let err = SsdDiskAnnIndex::open(store.path(), nav.path(), 128, &guard).unwrap_err();
        assert!(matches!(err, VectorIndexError::IrrecoverableLoss { .. }));
    }

    #[test]
    fn rabitq_open_rejects_truncated_v2_sidecar() {
        let dim = 16;
        let (corpus, _) = cluster_corpus(0xA209_0005, 8, 30, dim, 0.04);
        let codebook = train_rabitq(&corpus);
        let store = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let idx = build_small_nav(
            store.path(),
            dim,
            128,
            NavQuantizer::RaBitQ(codebook),
            corpus,
            &guard,
        )
        .unwrap();
        let nav = tempfile::NamedTempFile::new().unwrap();
        idx.save_nav(nav.path()).unwrap();
        assert_ssd_sidecar_version(nav.path(), SSD_NAV_VERSION);

        let full = std::fs::read(nav.path()).unwrap();
        std::fs::write(nav.path(), &full[..full.len() / 2]).unwrap();
        let err = SsdDiskAnnIndex::open(store.path(), nav.path(), 128, &guard).unwrap_err();
        assert!(matches!(err, VectorIndexError::IrrecoverableLoss { .. }));
    }

    #[test]
    fn open_rejects_too_small_f32_store() {
        // The nav sidecar is valid but the f32 store path points at a file too
        // small for the (dim, R, n) geometry → reject rather than read garbage.
        let dim = 16;
        let (corpus, _) = cluster_corpus(8, 12, 50, dim, 0.04);
        let codebook = train_codebook(&corpus);
        let store = tempfile::NamedTempFile::new().unwrap();
        let guard = RssGuard::disabled();
        let idx = build_small(store.path(), dim, 128, codebook, corpus, &guard).unwrap();
        let nav = tempfile::NamedTempFile::new().unwrap();
        idx.save_nav(nav.path()).unwrap();

        // An empty store file → file_len 0 < expected.
        let tiny_store = tempfile::NamedTempFile::new().unwrap();
        let err = SsdDiskAnnIndex::open(tiny_store.path(), nav.path(), 128, &guard).unwrap_err();
        assert!(matches!(err, VectorIndexError::IrrecoverableLoss { .. }));
    }
}
