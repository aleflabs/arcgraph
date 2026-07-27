//! Persistence for the SSD-resident DiskANN navigation graph (ADR-195 §3 /
//! the v1.0-α 10M GA-gate "reload + search-param sweep" instrument).
//!
//! ## Why this exists
//!
//! [`super::ssd::SsdDiskAnnIndex::build`] streams the f32 corpus to the on-disk
//! page store but keeps the EXPENSIVE artifact — the SQ8 Vamana navigation graph
//! (`vectors + neighbors + entry_point`, the multi-hour `build_owned_parallel`
//! refinement, build.rs: "@100K×768 ≈ 481 s → hours at 10M") — in RAM ONLY.
//! Before this module there was no `open`/`load` path, so every serving process
//! re-ran the full Vamana build (the 10M GA-gate's ~day cost) even when the
//! on-disk f32 store already existed. The ADR-189 §B search-param sweep needs to
//! re-measure recall/P95 across many `(l_search, rerank_k)` configs WITHOUT
//! paying that build per fresh process.
//!
//! This module adds a compact, mmap-free (PD#2 — plain buffered `Read`/`Write`,
//! not `memmap2`) sidecar format for the in-RAM nav graph + its codebook so a
//! built index can be [`SsdDiskAnnIndex::save_nav`]'d once and
//! [`SsdDiskAnnIndex::open`]'d in minutes (bounded by the sidecar read, not the
//! Vamana build).
//!
//! [`SsdDiskAnnIndex::save_nav`]: super::ssd::SsdDiskAnnIndex::save_nav
//! [`SsdDiskAnnIndex::open`]: super::ssd::SsdDiskAnnIndex::open
//!
//! ## Scope (nav format v1)
//!
//! Serializes the BULK-built main graph only: ids, SQ8 vectors, Vamana
//! adjacency, entry point, params, encoding/metric. The delta-segment,
//! tombstones, per-label entry cache, and MVCC LSN windows are NOT serialized —
//! the SSD GA build path (`build_owned_parallel`) produces none of them (delta /
//! tombstones empty; the LSN arrays are the always-visible `(ZERO, MAX)`
//! defaults that `DiskAnnGraph::allocate_slot` re-creates on load).
//! `DiskAnnGraph::serialize_nav` REJECTS a graph that carries a delta-segment
//! or tombstones rather than silently dropping them (no-op-trampoline / silent
//! data-loss guard).
//!
//! ## Back-of-envelope (PD#5)
//!
//! At the 10M GA point (`dim=768`, `R=128`, SQ8 nav): payload ≈ ids
//! `10M·4 B = 40 MB` + SQ8 vectors `10M·768 B = 7.68 GB` + adjacency
//! `10M·(4 + ≤128·4) B ≈ 5.16 GB` ≈ **12.9 GB** sidecar — read back at SSD
//! bandwidth (~2–4 GB/s) in **single-digit minutes**, vs the ~day Vamana
//! rebuild. The f32 page store (the 40.96 GB `.bin`) is reopened in place via
//! `PosixPageIo::open` — no copy, no re-stream.

use std::io::{Read, Write};

use super::graph::{DiskAnnGraph, DiskAnnParams};
use crate::distance::DistanceKernel;
use crate::{Encoding, IndexId, Metric, Result, VectorId, VectorIndexError};

/// Magic for the nav-graph section of the sidecar (`ArcGraph SSD Nav Graph 1`).
pub(crate) const NAV_MAGIC: &[u8; 8] = b"AGSSNG\x01\x00";
/// Nav-graph section format version.
pub(crate) const NAV_VERSION: u32 = 1;

/// A defensive pre-allocation cap: never `Vec::with_capacity(n)` with an
/// untrusted `n` straight from the file header (a corrupt/lying length must hit
/// EOF on `read_exact`, not OOM the pre-alloc). Real 10M nav has n ≤ 10M.
const PREALLOC_CAP: usize = 1 << 20;

// ── error helpers ────────────────────────────────────────────────────────────

/// Map a raw `std::io::Error` from the sidecar stream into the codec-local
/// [`VectorIndexError`] at this consuming boundary (per
/// `docs/codec-error-translation.md`).
pub(crate) fn io_loss(e: std::io::Error) -> VectorIndexError {
    VectorIndexError::IrrecoverableLoss {
        index: IndexId::ZERO,
        reason: format!("SSD DiskANN nav sidecar I/O failed: {e}"),
    }
}

/// A format/validation error in the sidecar (bad magic, truncation, an
/// out-of-range slot reference, …).
pub(crate) fn fmt_loss(reason: impl Into<String>) -> VectorIndexError {
    VectorIndexError::IrrecoverableLoss {
        index: IndexId::ZERO,
        reason: format!("SSD DiskANN nav sidecar format error: {}", reason.into()),
    }
}

// ── little-endian scalar read/write over trait objects ───────────────────────

pub(crate) fn write_u8(w: &mut dyn Write, v: u8) -> Result<()> {
    w.write_all(&[v]).map_err(io_loss)
}
pub(crate) fn write_u32(w: &mut dyn Write, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(io_loss)
}
pub(crate) fn write_u64(w: &mut dyn Write, v: u64) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(io_loss)
}
pub(crate) fn write_i64(w: &mut dyn Write, v: i64) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(io_loss)
}
pub(crate) fn write_f32(w: &mut dyn Write, v: f32) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(io_loss)
}

pub(crate) fn read_u8(r: &mut dyn Read) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(io_loss)?;
    Ok(b[0])
}
pub(crate) fn read_u32(r: &mut dyn Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(io_loss)?;
    Ok(u32::from_le_bytes(b))
}
pub(crate) fn read_u64(r: &mut dyn Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(io_loss)?;
    Ok(u64::from_le_bytes(b))
}
pub(crate) fn read_i64(r: &mut dyn Read) -> Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(io_loss)?;
    Ok(i64::from_le_bytes(b))
}
pub(crate) fn read_f32(r: &mut dyn Read) -> Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(io_loss)?;
    Ok(f32::from_le_bytes(b))
}

// ── encoding / metric discriminant codecs ────────────────────────────────────
//
// Explicit u8 tags (NOT `as u8` / serde) so the on-disk format is stable and
// independent of source-order. Unknown tags error rather than silently
// mis-decode.

fn encoding_to_u8(e: Encoding) -> u8 {
    match e {
        Encoding::F32 => 0,
        Encoding::F16 => 1,
        Encoding::Sq8 => 2,
        Encoding::Binary => 3,
        // TODO(#758): tag reserved for ADR-209 slice 2 nav
        // producers; slice 1 round-trips but emits no RaBitQ nav.
        Encoding::RaBitQ => 4,
    }
}
fn encoding_from_u8(v: u8) -> Result<Encoding> {
    Ok(match v {
        0 => Encoding::F32,
        1 => Encoding::F16,
        2 => Encoding::Sq8,
        3 => Encoding::Binary,
        // TODO(#758): tag reserved for ADR-209 slice 2 nav
        // producers; slice 1 round-trips but emits no RaBitQ nav.
        4 => Encoding::RaBitQ,
        other => return Err(fmt_loss(format!("unknown Encoding tag {other}"))),
    })
}
fn metric_to_u8(m: Metric) -> u8 {
    match m {
        Metric::L2 => 0,
        Metric::Ip => 1,
        Metric::Cosine => 2,
        Metric::Hamming => 3,
    }
}
fn metric_from_u8(v: u8) -> Result<Metric> {
    Ok(match v {
        0 => Metric::L2,
        1 => Metric::Ip,
        2 => Metric::Cosine,
        3 => Metric::Hamming,
        other => return Err(fmt_loss(format!("unknown Metric tag {other}"))),
    })
}

impl DiskAnnGraph {
    /// Serialize the BULK-built main nav graph into `w` (the nav-graph section
    /// of the [`SsdDiskAnnIndex`] sidecar).
    ///
    /// [`SsdDiskAnnIndex`]: super::ssd::SsdDiskAnnIndex
    ///
    /// # Errors
    /// - [`VectorIndexError::IrrecoverableLoss`] if the graph carries a
    ///   delta-segment or any tombstone (the v1 format supports only the
    ///   bulk-built main graph — refuse rather than silently drop state), or on
    ///   a write error.
    pub(crate) fn serialize_nav(&self, w: &mut dyn Write) -> Result<()> {
        // Scope guard: v1 supports the bulk-built main graph ONLY. The SSD GA
        // build path produces neither a delta-segment nor tombstones; refuse to
        // silently drop them if a future caller hands us a streamed graph.
        if self.delta_len() != 0 {
            return Err(fmt_loss(format!(
                "serialize_nav: graph has a non-empty delta-segment (len {}); \
                 the v1 nav format supports the bulk-built main graph only",
                self.delta_len()
            )));
        }
        if self.live_tombstone_count() != 0 {
            return Err(fmt_loss(format!(
                "serialize_nav: graph has {} tombstone(s); the v1 nav format \
                 supports the bulk-built main graph only",
                self.live_tombstone_count()
            )));
        }

        let n = self.main_len();
        // bytes_per_vector is None only for a never-ingested (empty) graph.
        let bpv = self.bytes_per_vector.unwrap_or(0);

        w.write_all(NAV_MAGIC).map_err(io_loss)?;
        write_u32(w, NAV_VERSION)?;
        write_u8(w, encoding_to_u8(self.encoding))?;
        write_u8(w, metric_to_u8(self.metric))?;

        // Tuning params (so a reload reproduces l_search_default etc.).
        let p = self.params();
        write_u32(w, p.r)?;
        write_f32(w, p.alpha)?;
        write_u32(w, p.l_construction)?;
        write_u32(w, p.l_search_default)?;
        write_u32(w, p.delta_max_size)?;
        write_u32(w, p.delta_brute_thresh)?;
        write_u32(w, p.medoid_sample_size)?;

        write_u64(w, bpv as u64)?;
        write_u64(w, n as u64)?;
        write_i64(w, self.entry_point.map_or(-1, i64::from))?;

        // ids
        for id in &self.ids {
            write_u32(w, id.raw())?;
        }
        // SQ8 nav vectors (each exactly bpv bytes). The on-disk format remains
        // slot-contiguous even though the in-memory store is one flat arena.
        for slot in 0..n {
            let v = self.vector_bytes(slot as u32);
            debug_assert_eq!(v.len(), bpv, "nav vector width != bytes_per_vector");
            w.write_all(v).map_err(io_loss)?;
        }
        // adjacency (degree-prefixed slot lists)
        for nbrs in &self.neighbors {
            write_u32(
                w,
                u32::try_from(nbrs.len()).map_err(|_| fmt_loss("degree exceeds u32"))?,
            )?;
            for &nb in nbrs {
                write_u32(w, nb)?;
            }
        }
        Ok(())
    }

    /// Reconstruct a bulk nav graph from a [`serialize_nav`] stream, using the
    /// caller-supplied `kernel` (whose `encoding`/`metric` MUST match the stored
    /// ones — validated by [`DiskAnnGraph::new`]).
    ///
    /// [`serialize_nav`]: DiskAnnGraph::serialize_nav
    ///
    /// # Errors
    /// - [`VectorIndexError::IrrecoverableLoss`] on bad magic / unsupported
    ///   version / truncation / an out-of-range slot reference.
    /// - [`VectorIndexError::UnsupportedFlags`] if the kernel does not match the
    ///   stored `(encoding, metric)`.
    pub(crate) fn deserialize_nav(
        r: &mut dyn Read,
        kernel: Box<dyn DistanceKernel>,
    ) -> Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).map_err(io_loss)?;
        if &magic != NAV_MAGIC {
            return Err(fmt_loss("bad nav-graph magic"));
        }
        let version = read_u32(r)?;
        if version != NAV_VERSION {
            return Err(fmt_loss(format!(
                "unsupported nav-graph version {version} (expected {NAV_VERSION})"
            )));
        }
        let encoding = encoding_from_u8(read_u8(r)?)?;
        let metric = metric_from_u8(read_u8(r)?)?;

        let params = DiskAnnParams {
            r: read_u32(r)?,
            alpha: read_f32(r)?,
            l_construction: read_u32(r)?,
            l_search_default: read_u32(r)?,
            delta_max_size: read_u32(r)?,
            delta_brute_thresh: read_u32(r)?,
            medoid_sample_size: read_u32(r)?,
        };

        let bpv = read_u64(r)? as usize;
        let n = read_u64(r)? as usize;
        let entry_raw = read_i64(r)?;
        let entry_point = if entry_raw < 0 {
            None
        } else {
            Some(u32::try_from(entry_raw).map_err(|_| fmt_loss("entry_point out of u32 range"))?)
        };

        // Empty graph: nothing more to read.
        if n == 0 {
            return DiskAnnGraph::new(params, encoding, metric, kernel);
        }
        if bpv == 0 {
            return Err(fmt_loss("non-empty graph with zero byte-width"));
        }

        // ids
        let mut ids = Vec::with_capacity(n.min(PREALLOC_CAP));
        for _ in 0..n {
            ids.push(VectorId::new(read_u32(r)?));
        }
        // vectors (read_exact per slot → a lying `n` hits EOF, never OOMs)
        let mut entries: Vec<(VectorId, Vec<u8>)> = Vec::with_capacity(n.min(PREALLOC_CAP));
        for id in ids {
            let mut buf = vec![0u8; bpv];
            r.read_exact(&mut buf).map_err(io_loss)?;
            entries.push((id, buf));
        }
        // adjacency
        let mut neighbors: Vec<Vec<u32>> = Vec::with_capacity(n.min(PREALLOC_CAP));
        for _ in 0..n {
            let deg = read_u32(r)? as usize;
            let mut nbrs = Vec::with_capacity(deg.min(PREALLOC_CAP));
            for _ in 0..deg {
                nbrs.push(read_u32(r)?);
            }
            neighbors.push(nbrs);
        }

        // `new` validates params + that the kernel matches (encoding, metric).
        let mut graph = DiskAnnGraph::new(params, encoding, metric, kernel)?;
        graph.load_bulk(entries, neighbors, entry_point)?;
        Ok(graph)
    }

    /// Populate a freshly-constructed (empty) graph with a bulk set of
    /// `(id, encoded-bytes)` entries + Vamana adjacency + entry point, REUSING
    /// the tested [`DiskAnnGraph::allocate_slot`] primitive so the MVCC LSN
    /// arrays + `id_to_slot` grow in lockstep exactly as a real build does. No
    /// Vamana refinement runs — the neighbors are taken verbatim from the
    /// serialized graph.
    ///
    /// Precondition: `self` is empty (just `new`'d). The serialized graph is
    /// delta-/tombstone-free by [`serialize_nav`]'s guard, so no delta or
    /// tombstone state is established here.
    ///
    /// [`serialize_nav`]: DiskAnnGraph::serialize_nav
    ///
    /// # Errors
    /// - [`VectorIndexError::IrrecoverableLoss`] on a length mismatch, a
    ///   duplicate id, or an out-of-range neighbor / entry-point slot.
    /// - [`VectorIndexError::DimensionMismatch`] if the entries' byte widths
    ///   disagree.
    pub(crate) fn load_bulk(
        &mut self,
        entries: Vec<(VectorId, Vec<u8>)>,
        neighbors: Vec<Vec<u32>>,
        entry_point: Option<u32>,
    ) -> Result<()> {
        debug_assert!(
            self.main_len() == 0,
            "load_bulk must be called on a freshly-constructed graph"
        );
        let n = entries.len();
        if neighbors.len() != n {
            return Err(fmt_loss(format!(
                "load_bulk: neighbors len {} != ids len {n}",
                neighbors.len()
            )));
        }
        // Bound-check every neighbor + the entry point against `n` BEFORE
        // installing (a single out-of-range slot would panic the search path).
        for (slot, nbrs) in neighbors.iter().enumerate() {
            for &nb in nbrs {
                if nb as usize >= n {
                    return Err(fmt_loss(format!(
                        "load_bulk: neighbor {nb} of slot {slot} out of range (n={n})"
                    )));
                }
            }
        }
        if let Some(ep) = entry_point
            && ep as usize >= n
        {
            return Err(fmt_loss(format!(
                "load_bulk: entry_point {ep} out of range (n={n})"
            )));
        }

        for (id, bytes) in entries {
            self.check_or_set_byte_width(bytes.len())?;
            if self.id_to_slot.contains_key(&id) {
                return Err(fmt_loss(format!(
                    "load_bulk: duplicate VectorId {} in serialized graph",
                    id.raw()
                )));
            }
            // allocate_slot pushes an empty neighbor list + the (ZERO, MAX) LSN
            // window; we overwrite neighbors wholesale below.
            let _slot = self.allocate_slot(id, bytes);
        }
        self.neighbors = neighbors;
        self.entry_point = entry_point;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{L2F32, L2RaBitQSym};
    use crate::quantizer::{RaBitQCodebook, RaBitQParams};

    fn f32_le(v: &[f32]) -> Vec<u8> {
        let mut o = Vec::with_capacity(v.len() * 4);
        for &x in v {
            o.extend_from_slice(&x.to_le_bytes());
        }
        o
    }

    /// A tiny F32 graph (the nav format is encoding-agnostic; F32 keeps the test
    /// payloads legible). Returns a built, non-empty graph.
    fn tiny_f32_graph(n: usize, dim: usize) -> DiskAnnGraph {
        let kernel: Box<dyn DistanceKernel> = Box::new(L2F32);
        let mut g =
            DiskAnnGraph::new(DiskAnnParams::default(), Encoding::F32, Metric::L2, kernel).unwrap();
        let entries: Vec<(VectorId, Vec<u8>)> = (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|d| (i * 7 + d) as f32 * 0.01).collect();
                (VectorId::new(i as u32), f32_le(&v))
            })
            .collect();
        g.build_owned(entries).unwrap();
        g
    }

    #[test]
    fn rabitq_encoding_tag_round_trips_without_producer() {
        assert_eq!(encoding_to_u8(Encoding::RaBitQ), 4);
        assert_eq!(encoding_from_u8(4).unwrap(), Encoding::RaBitQ);
    }

    #[test]
    fn nav_roundtrip_reproduces_every_field() {
        let g = tiny_f32_graph(64, 12);
        let mut buf = Vec::new();
        g.serialize_nav(&mut buf).unwrap();

        let kernel: Box<dyn DistanceKernel> = Box::new(L2F32);
        let mut cur = std::io::Cursor::new(buf);
        let g2 = DiskAnnGraph::deserialize_nav(&mut cur, kernel).unwrap();

        assert_eq!(g2.ids, g.ids);
        assert_eq!(g2.vectors, g.vectors);
        assert_eq!(g2.neighbors, g.neighbors);
        assert_eq!(g2.entry_point, g.entry_point);
        assert_eq!(g2.params(), g.params());
        assert_eq!(g2.bytes_per_vector, g.bytes_per_vector);
        assert_eq!(g2.encoding, g.encoding);
        assert_eq!(g2.metric, g.metric);
        // The reverse map must be rebuilt consistently.
        assert_eq!(g2.id_to_slot, g.id_to_slot);
    }

    #[test]
    fn rabitq_nav_roundtrip_reproduces_tag_and_aligned_width() {
        let dim = 16;
        let cb = identity_rabitq(dim);
        let entries: Vec<(VectorId, Vec<u8>)> = (0..32)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|d| (i as f32 * 0.03) + (d as f32 * 0.01))
                    .collect();
                (VectorId::new(i), cb.encode_aligned(&v).unwrap())
            })
            .collect();
        let mut g = DiskAnnGraph::new(
            DiskAnnParams::default(),
            Encoding::RaBitQ,
            Metric::L2,
            Box::new(L2RaBitQSym::new(dim)),
        )
        .unwrap();
        g.build_owned(entries).unwrap();
        assert_eq!(
            g.bytes_per_vector(),
            Some(Encoding::RaBitQ.bytes_per_vector_aligned(dim))
        );

        let mut buf = Vec::new();
        g.serialize_nav(&mut buf).unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let g2 = DiskAnnGraph::deserialize_nav(&mut cur, Box::new(L2RaBitQSym::new(dim))).unwrap();
        assert_eq!(g2.encoding, Encoding::RaBitQ);
        assert_eq!(g2.metric, Metric::L2);
        assert_eq!(g2.ids, g.ids);
        assert_eq!(g2.vectors, g.vectors);
        assert_eq!(g2.neighbors, g.neighbors);
        assert_eq!(g2.bytes_per_vector, g.bytes_per_vector);
    }

    #[test]
    fn empty_graph_roundtrips() {
        let kernel: Box<dyn DistanceKernel> = Box::new(L2F32);
        let g =
            DiskAnnGraph::new(DiskAnnParams::default(), Encoding::F32, Metric::L2, kernel).unwrap();
        let mut buf = Vec::new();
        g.serialize_nav(&mut buf).unwrap();
        let kernel2: Box<dyn DistanceKernel> = Box::new(L2F32);
        let mut cur = std::io::Cursor::new(buf);
        let g2 = DiskAnnGraph::deserialize_nav(&mut cur, kernel2).unwrap();
        assert_eq!(g2.main_len(), 0);
        assert_eq!(g2.entry_point, None);
    }

    #[test]
    fn serialize_rejects_tombstoned_graph() {
        // The v1 nav format is bulk-only; a graph carrying a tombstone must be
        // refused, not silently serialized without the tombstone (data loss).
        let mut g = tiny_f32_graph(16, 8);
        g.set_tombstone(0);
        let mut buf = Vec::new();
        let err = g.serialize_nav(&mut buf).unwrap_err();
        assert!(matches!(err, VectorIndexError::IrrecoverableLoss { .. }));
    }

    #[test]
    fn deserialize_rejects_bad_magic() {
        let mut cur = std::io::Cursor::new(b"BADMAGIC\x00\x00\x00\x00".to_vec());
        let kernel: Box<dyn DistanceKernel> = Box::new(L2F32);
        let err = DiskAnnGraph::deserialize_nav(&mut cur, kernel).unwrap_err();
        assert!(matches!(err, VectorIndexError::IrrecoverableLoss { .. }));
    }

    #[test]
    fn load_bulk_rejects_out_of_range_neighbor() {
        let kernel: Box<dyn DistanceKernel> = Box::new(L2F32);
        let mut g =
            DiskAnnGraph::new(DiskAnnParams::default(), Encoding::F32, Metric::L2, kernel).unwrap();
        let entries = vec![
            (VectorId::new(0), f32_le(&[0.0, 1.0])),
            (VectorId::new(1), f32_le(&[1.0, 0.0])),
        ];
        // neighbor 5 is out of range for n=2.
        let neighbors = vec![vec![1u32], vec![5u32]];
        let err = g.load_bulk(entries, neighbors, Some(0)).unwrap_err();
        assert!(matches!(err, VectorIndexError::IrrecoverableLoss { .. }));
    }

    #[test]
    fn load_bulk_rejects_out_of_range_entry_point() {
        let kernel: Box<dyn DistanceKernel> = Box::new(L2F32);
        let mut g =
            DiskAnnGraph::new(DiskAnnParams::default(), Encoding::F32, Metric::L2, kernel).unwrap();
        let entries = vec![(VectorId::new(0), f32_le(&[0.0, 1.0]))];
        let neighbors = vec![vec![]];
        let err = g.load_bulk(entries, neighbors, Some(9)).unwrap_err();
        assert!(matches!(err, VectorIndexError::IrrecoverableLoss { .. }));
    }

    #[test]
    fn load_bulk_rejects_duplicate_id() {
        let kernel: Box<dyn DistanceKernel> = Box::new(L2F32);
        let mut g =
            DiskAnnGraph::new(DiskAnnParams::default(), Encoding::F32, Metric::L2, kernel).unwrap();
        let entries = vec![
            (VectorId::new(3), f32_le(&[0.0, 1.0])),
            (VectorId::new(3), f32_le(&[1.0, 0.0])),
        ];
        let neighbors = vec![vec![], vec![]];
        let err = g.load_bulk(entries, neighbors, None).unwrap_err();
        assert!(matches!(err, VectorIndexError::IrrecoverableLoss { .. }));
    }

    fn identity_rabitq(dim: usize) -> RaBitQCodebook {
        let mut rotation = vec![0.0; dim * dim];
        for d in 0..dim {
            rotation[d * dim + d] = 1.0;
        }
        RaBitQCodebook::from_params(RaBitQParams::try_new(dim, vec![0.0; dim], rotation).unwrap())
    }
}
