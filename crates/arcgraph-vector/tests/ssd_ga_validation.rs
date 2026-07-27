//! The ADR-189 §B / ADR-195 §6 **>10M-vector DiskANN GA-gate** validation
//! driver, as an env-gated out-of-band test.
//!
//! Binary gate (Director E-1): **recall@10 ≥ 0.95 ∧ P95 ≤ 15 ms ∧ peak-RSS <
//! cap** at the configured scale, dim=768, REAL measured against an EXHAUSTIVE
//! brute-force oracle. HONESTY GATE (ADR-195 §6): real numbers or a real
//! blocker — never a faked/hardcoded pass.
//!
//! ## Why this is the 10M instrument (run at a smaller scale in-session)
//!
//! The corpus is generated **random-access** (each `vector(i)` is a pure
//! function of `(seed, i)` via precomputed centers + a per-vector noise PRNG),
//! so the f32 corpus is NEVER all-resident — the build streams it to the
//! `PosixPageIo` page store, and the brute-force GT regenerates each vector on
//! demand. This is the SAME code path at 100K and at 10M; only the compute
//! window differs (the 10M brute-force GT is O(10M·768·Q) ≈ multi-hour, hence
//! the content-hashed `.gt` cache per RC-5). Running it at 100K in-session
//! produces a REAL measured number on the identical instrument.
//!
//! ## Gates / env
//!
//! - `ARCGRAPH_VECTOR_GA_BENCH_OK=1` — REQUIRED opt-in (panic-by-default skip
//!   per W25-MFI-2; this is RAM/CPU-heavy, NEVER in CI).
//! - `ARCGRAPH_VECTOR_GA_CLUSTERS` (default 1000) × `ARCGRAPH_VECTOR_GA_POINTS`
//!   (default 100) = N. For the 10M gate: `CLUSTERS=100000 POINTS=100`.
//! - `ARCGRAPH_VECTOR_GA_DIM` (default 768).
//! - `ARCGRAPH_VECTOR_GA_QUERIES` (default 1000).
//! - `ARCGRAPH_VECTOR_GA_BUILD_BATCH` (default 4096) — rayon parallel-build batch.
//! - `ARCGRAPH_VECTOR_RSS_CAP_MB` (default 14000) — the RSS GUARD cap (the
//!   fail-clean abort threshold). Raise it above the gate to let a run COMPLETE
//!   + measure recall/latency/true-RSS even when over the 14 GB target.
//! - `ARCGRAPH_VECTOR_GA_RSS_GATE_MB` (default 14000) — the RSS GATE the assert
//!   checks (the E-1 binary-gate target), independent of the guard cap.
//! - `ARCGRAPH_VECTOR_GA_GATE` (default "1") — when "1", assert the binary gate;
//!   when "0", measure + report only (for a sub-gate scale sweep).
//! - `ARCGRAPH_VECTOR_GA_ENCODING` (`sq8` default, or `rabitq`) — nav encoding.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;

use arcgraph_vector::diskann::DiskAnnParams;
use arcgraph_vector::diskann::rss_guard::{DEFAULT_RSS_CAP_MB, RssGuard};
use arcgraph_vector::diskann::ssd::{
    DEFAULT_RERANK_FACTOR, NavQuantizer, SsdBuildConfig, SsdDiskAnnIndex,
};
use arcgraph_vector::distance::{DistanceKernel, L2F32};
use arcgraph_vector::{Metric, VectorId};

const GA_ENV: &str = "ARCGRAPH_VECTOR_GA_BENCH_OK";
const SEED: u32 = 0x5EED_1234;
const SIGMA: f32 = 0.02;
/// Beam width — wide enough for recall@10 ≥ 0.95 with margin at 768d (matches
/// the V-1 #740 `L_SEARCH` finding).
const RERANK_FACTOR: usize = DEFAULT_RERANK_FACTOR;
const RABITQ_GA_SEED: u64 = 0x7580_2090_0000_0002;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GaEncoding {
    Sq8,
    RaBitQ,
}

impl GaEncoding {
    fn from_env() -> Self {
        match std::env::var("ARCGRAPH_VECTOR_GA_ENCODING")
            .unwrap_or_else(|_| "sq8".to_string())
            .as_str()
        {
            "sq8" => Self::Sq8,
            "rabitq" => Self::RaBitQ,
            other => {
                panic!("unknown ARCGRAPH_VECTOR_GA_ENCODING={other:?}; allowed values: sq8, rabitq")
            }
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Sq8 => "sq8",
            Self::RaBitQ => "rabitq",
        }
    }
}

fn train_nav_quantizer(encoding: GaEncoding, train_refs: &[&[f32]]) -> NavQuantizer {
    match encoding {
        GaEncoding::Sq8 => NavQuantizer::Sq8(
            arcgraph_vector::quantizer::Sq8Trainer
                .train(train_refs)
                .expect("train sq8"),
        ),
        GaEncoding::RaBitQ => NavQuantizer::RaBitQ(
            arcgraph_vector::quantizer::RaBitQTrainer
                .train(train_refs, RABITQ_GA_SEED)
                .expect("train rabitq"),
        ),
    }
}

// ─── random-access deterministic corpus (bounded RAM) ────────────────────────

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

/// Precomputed cluster centers (the only all-resident corpus structure — at the
/// 10M gate `clusters=100000 × dim=768 × 4 B = 307 MB`, well within the cap).
struct Corpus {
    centers: Vec<Vec<f32>>,
    points_per: usize,
    dim: usize,
    n: usize,
}

impl Corpus {
    fn new(clusters: usize, points_per: usize, dim: usize) -> Self {
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|c| {
                let mut rng = Xs32::new(SEED ^ 0x000C_0DE5u32.wrapping_mul(c as u32 + 1));
                (0..dim).map(|_| rng.signed()).collect()
            })
            .collect();
        Self {
            centers,
            points_per,
            dim,
            n: clusters * points_per,
        }
    }

    /// Random-access regeneration of corpus vector `i` (pure fn of `(SEED, i)`).
    fn vector(&self, i: usize) -> Vec<f32> {
        let center = &self.centers[i / self.points_per];
        let mut rng = Xs32::new(SEED ^ 0x5151_0000u32.wrapping_add(i as u32 + 1));
        center.iter().map(|&cc| cc + rng.gauss() * SIGMA).collect()
    }

    /// Regenerate corpus vector `i` directly as little-endian bytes into a
    /// REUSED buffer (no per-vector allocation) — the GT hot path. The bytes are
    /// exactly what the `L2F32` kernel reads, keeping the oracle byte-identical
    /// to the index's distance.
    fn vector_into_le(&self, i: usize, buf: &mut [u8]) {
        debug_assert_eq!(buf.len(), self.dim * 4);
        let center = &self.centers[i / self.points_per];
        let mut rng = Xs32::new(SEED ^ 0x5151_0000u32.wrapping_add(i as u32 + 1));
        for d in 0..self.dim {
            let x = center[d] + rng.gauss() * SIGMA;
            buf[d * 4..d * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
    }

    /// An in-distribution query (same generative process, distinct seed stream).
    fn query(&self, q: usize) -> Vec<f32> {
        let mut sel = Xs32::new(0xA11C_E000u32.wrapping_add(q as u32 + 1));
        let center = &self.centers[(sel.next_u32() as usize) % self.centers.len()];
        let mut rng = Xs32::new(0xB0B0_0000u32.wrapping_add(q as u32 + 1));
        center.iter().map(|&cc| cc + rng.gauss() * SIGMA).collect()
    }
}

fn f32_le(v: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 4);
    for &x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Content hash of the GT-defining inputs (RC-5): a stale `.gt` false-pass is
/// impossible — the hash keys (seed, dim, clusters, points_per, query-set, k).
/// `clusters` and `points_per` are keyed SEPARATELY, not via `N = clusters ×
/// points_per`: the corpus is a function of BOTH factors (`vector(i)` selects
/// `centers[i / points_per]`, and the centers themselves are derived per
/// cluster), so two different factorizations at equal `N` (e.g. 1000×100 vs
/// 2000×50) yield DIFFERENT corpora. Keying on `N` alone would let those collide
/// and read each other's stale `.gt` → a false recall pass; keying on the
/// factors makes the cache content-correct.
fn gt_hash(dim: usize, clusters: usize, points_per: usize, n_queries: usize, k: usize) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (
        SEED,
        dim,
        clusters,
        points_per,
        n_queries,
        k,
        SIGMA.to_bits(),
    )
        .hash(&mut h);
    h.finish()
}

/// Insert `(d, id)` into an ascending-by-distance top-`k` list in place.
fn insert_topk(list: &mut Vec<(f32, u32)>, d: f32, id: u32, k: usize) {
    if list.len() < k {
        list.push((d, id));
        if list.len() == k {
            list.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
    } else if d < list[k - 1].0 {
        list[k - 1] = (d, id);
        let mut j = k - 1;
        while j > 0 && list[j].0 < list[j - 1].0 {
            list.swap(j, j - 1);
            j -= 1;
        }
    }
}

/// Exhaustive brute-force top-k ground truth — TRANSPOSED: each corpus vector is
/// regenerated ONCE and compared to ALL queries (N regens, not N·Q; the regen
/// is the expensive part, the L2 is cheap SIMD). Parallel fold over corpus
/// indices + reduce of per-query top-k partials; bounded RAM. Cached to a
/// content-hashed `.gt` file (RC-5). This is the strong oracle (exhaustive,
/// kernel-identical to the index) the recall@10 gate is measured against.
fn brute_force_gt(corpus: &Corpus, queries: &[Vec<f32>], k: usize) -> Vec<HashSet<u32>> {
    // Key on the corpus FACTORS (clusters, points_per), not the product N — two
    // factorizations at equal N produce different corpora (see `gt_hash`).
    let hash = gt_hash(
        corpus.dim,
        corpus.centers.len(),
        corpus.points_per,
        queries.len(),
        k,
    );
    let path = std::env::temp_dir().join(format!("arcgraph_ssd_ga_{hash:016x}.gt"));

    if let Ok(mut f) = std::fs::File::open(&path) {
        let mut buf = Vec::new();
        if f.read_to_end(&mut buf).is_ok() && buf.len() >= 8 {
            let stored = u64::from_le_bytes(buf[..8].try_into().unwrap());
            if stored == hash {
                eprintln!("[GA] GT cache HIT ({})", path.display());
                let mut out = Vec::with_capacity(queries.len());
                let mut off = 8;
                for _ in 0..queries.len() {
                    let mut set = HashSet::with_capacity(k);
                    for _ in 0..k {
                        let id = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
                        set.insert(id);
                        off += 4;
                    }
                    out.push(set);
                }
                return out;
            }
        }
        eprintln!("[GA] GT cache hash MISMATCH — recomputing (RC-5)");
    }

    let q_bytes: Vec<Vec<u8>> = queries.iter().map(|q| f32_le(q)).collect();
    let nq = queries.len();
    let dim_bytes = corpus.dim * 4;

    let new_partials = || -> Vec<Vec<(f32, u32)>> { vec![Vec::with_capacity(k + 1); nq] };
    let merged: Vec<Vec<(f32, u32)>> = (0..corpus.n)
        .into_par_iter()
        .fold(
            || (new_partials(), vec![0u8; dim_bytes]),
            |(mut acc, mut vbuf), i| {
                // Regenerate corpus vector `i` ONCE, score against all queries.
                corpus.vector_into_le(i, &mut vbuf);
                for q in 0..nq {
                    let d = L2F32.distance(&vbuf, &q_bytes[q]);
                    insert_topk(&mut acc[q], d, i as u32, k);
                }
                (acc, vbuf)
            },
        )
        .map(|(acc, _)| acc)
        .reduce(new_partials, |mut a, b| {
            // Per-job corpus ranges are disjoint → no duplicate ids across a/b.
            for q in 0..nq {
                for (d, id) in b[q].iter().copied() {
                    insert_topk(&mut a[q], d, id, k);
                }
            }
            a
        });
    let gt: Vec<Vec<u32>> = merged
        .into_iter()
        .map(|v| v.into_iter().map(|(_, id)| id).collect())
        .collect();

    // Persist (content-hash header + flat u32 ids).
    if let Ok(mut f) = std::fs::File::create(&path) {
        let mut buf = Vec::with_capacity(8 + queries.len() * k * 4);
        buf.extend_from_slice(&hash.to_le_bytes());
        for set in &gt {
            for &id in set {
                buf.extend_from_slice(&id.to_le_bytes());
            }
        }
        let _ = f.write_all(&buf);
        eprintln!("[GA] GT cached to {}", path.display());
    }

    gt.into_iter().map(|v| v.into_iter().collect()).collect()
}

#[test]
#[ignore = "GA-gate: heavy (RAM+CPU+disk); opt-in via ARCGRAPH_VECTOR_GA_BENCH_OK=1 + --ignored"]
fn ssd_diskann_ga_gate() {
    // Panic-by-default env-gate (W25-MFI-2): NEVER silently skip.
    if std::env::var(GA_ENV).as_deref() != Ok("1") {
        panic!(
            "{GA_ENV} not set to 1 — the >10M GA-gate validation is opt-in (heavy; \
             NEVER in CI). Set {GA_ENV}=1 (+ scale envs) to run. See module docs."
        );
    }

    let clusters = env_usize("ARCGRAPH_VECTOR_GA_CLUSTERS", 1000);
    let points_per = env_usize("ARCGRAPH_VECTOR_GA_POINTS", 100);
    let dim = env_usize("ARCGRAPH_VECTOR_GA_DIM", 768);
    let n_queries = env_usize("ARCGRAPH_VECTOR_GA_QUERIES", 1000);
    let build_batch = env_usize("ARCGRAPH_VECTOR_GA_BUILD_BATCH", 4096);
    // The RSS GUARD cap (abort threshold) is distinct from the RSS GATE
    // threshold (the E-1 binary-gate target, 14 GB). Raising the guard cap above
    // the gate lets a run COMPLETE and MEASURE recall/latency/true-RSS even when
    // it exceeds the 14 GB target (the gate assert still checks the real target);
    // a guard-cap breach is the fail-clean backstop against runaway RAM.
    let cap_mb = env_usize("ARCGRAPH_VECTOR_RSS_CAP_MB", DEFAULT_RSS_CAP_MB as usize) as u64;
    let rss_gate_mb = env_usize("ARCGRAPH_VECTOR_GA_RSS_GATE_MB", 14000) as u64;
    let enforce_gate = std::env::var("ARCGRAPH_VECTOR_GA_GATE").as_deref() != Ok("0");
    let encoding = GaEncoding::from_env();
    let k = 10;
    let n = clusters * points_per;

    eprintln!(
        "[GA] N={n} (clusters={clusters}×{points_per}) dim={dim} queries={n_queries} \
         encoding={} build_batch={build_batch} rss_cap_mb={cap_mb} gate={enforce_gate}",
        encoding.label()
    );

    // Dim-scaled params (V-1 #740: 128-d defaults are graph-starved at 768d).
    let params = if dim > 256 {
        DiskAnnParams {
            r: 128,
            l_construction: 200,
            ..DiskAnnParams::default()
        }
    } else {
        DiskAnnParams::default()
    };

    let corpus = Corpus::new(clusters, points_per, dim);

    // Train nav quantizer on a reservoir-ish sample (every stride-th vector, ≤ 100K).
    let n_train = n.min(100_000);
    let stride = (n / n_train).max(1);
    let train_storage: Vec<Vec<f32>> = (0..n).step_by(stride).map(|i| corpus.vector(i)).collect();
    let train_refs: Vec<&[f32]> = train_storage.iter().map(Vec::as_slice).collect();
    let nav = train_nav_quantizer(encoding, &train_refs);
    drop(train_storage);

    // Arm the RSS guard for the WHOLE run (build + serving).
    let guard = RssGuard::spawn(cap_mb, std::time::Duration::from_millis(500));

    // ── Build (bounded/disk-spilling, parallel) ──
    let store = tempfile::NamedTempFile::new().unwrap();
    let cfg = SsdBuildConfig {
        dim,
        metric: Metric::L2,
        params,
        // Bound the rerank cache: a few thousand 8 KiB frames is plenty for the
        // rerank working set and keeps cache RAM small.
        pool_frames: 8192,
        rerank_factor: RERANK_FACTOR,
        parallel_build_batch: Some(build_batch),
    };
    let t_build = Instant::now();
    // Lazy iterator — the f32 corpus is NEVER all-resident.
    let vectors = (0..n).map(|i| (VectorId::new(i as u32), corpus.vector(i)));
    let idx = match SsdDiskAnnIndex::build(store.path(), &cfg, nav, vectors, &guard) {
        Ok(idx) => idx,
        Err(e) => {
            // Honest blocker (e.g., RssCapExceeded) — report + fail loudly.
            panic!(
                "[GA] BUILD FAILED (honest blocker): {e}\n[GA] peak-RSS during build = {} MB (cap {cap_mb})",
                guard.peak_mb()
            );
        }
    };
    let build_secs = t_build.elapsed().as_secs_f64();
    eprintln!(
        "[GA] build OK: {n} vectors in {build_secs:.1}s; disk={} GB; rss_peak={} MB",
        idx.disk_bytes() as f64 / 1e9,
        guard.peak_mb()
    );

    // ── Ground truth (exhaustive, content-hashed, parallel) ──
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|q| corpus.query(q)).collect();
    let t_gt = Instant::now();
    let gt = brute_force_gt(&corpus, &queries, k);
    eprintln!("[GA] GT computed in {:.1}s", t_gt.elapsed().as_secs_f64());

    // ── Serving: recall@10 + latency, RSS-guarded ──
    let mut hits = 0usize;
    let mut latencies_us: Vec<u128> = Vec::with_capacity(n_queries);
    for (qi, q) in queries.iter().enumerate() {
        guard.check().expect("RSS cap exceeded during serving");
        let t = Instant::now();
        let res = idx.search(q, k).expect("search");
        latencies_us.push(t.elapsed().as_micros());
        for (id, _) in &res {
            if gt[qi].contains(&id.raw()) {
                hits += 1;
            }
        }
    }
    let recall = hits as f64 / (k * n_queries) as f64;
    latencies_us.sort_unstable();
    let p50_ms = latencies_us[n_queries / 2] as f64 / 1000.0;
    let p95_ms = latencies_us[(n_queries as f64 * 0.95) as usize] as f64 / 1000.0;
    let peak_rss_mb = guard.peak_mb();

    eprintln!("════════════════════ ADR-189 §B 10M GA-GATE RESULT ════════════════════");
    eprintln!(
        "  N={n}  dim={dim}  encoding={}  R={}/L={}",
        encoding.label(),
        params.r,
        params.l_construction
    );
    eprintln!("  recall@10 = {recall:.4}   (gate ≥ 0.95)");
    eprintln!("  P50 = {p50_ms:.3} ms   P95 = {p95_ms:.3} ms   (gate P95 ≤ 15 ms)");
    eprintln!("  peak-RSS = {peak_rss_mb} MB   (gate < {rss_gate_mb} MB; guard cap {cap_mb} MB)");
    eprintln!(
        "  disk = {:.2} GB   build = {build_secs:.1}s",
        idx.disk_bytes() as f64 / 1e9
    );
    eprintln!("═══════════════════════════════════════════════════════════════════════");

    if enforce_gate {
        // HONESTY GATE: real measured asserts. A measured FAIL here is a real
        // finding (the correct outcome), NOT something to paper over. The RSS
        // assert is against the E-1 14 GB GATE (rss_gate_mb), independent of the
        // guard cap — so a cap-raised run that completes still fails honestly if
        // it exceeds the 14 GB target (the RC-1 → PQ-nav finding).
        assert!(recall >= 0.95, "GA-gate FAIL: recall@10 {recall:.4} < 0.95");
        assert!(p95_ms <= 15.0, "GA-gate FAIL: P95 {p95_ms:.3} ms > 15 ms");
        assert!(
            peak_rss_mb < rss_gate_mb,
            "GA-gate FAIL: peak-RSS {peak_rss_mb} MB ≥ E-1 gate {rss_gate_mb} MB \
             (RC-1: SQ8-nav over 14 GB target → escalate to PQ-nav per ADR-195 §2.1)"
        );
    }
}

// ─── ADR-189 §B reload + search-param sweep (the recall/P95-recovery instrument) ─

/// Parse a comma-separated `usize` list env (e.g. "400,800,1600"); empty/unset →
/// `default`.
fn env_list_usize(key: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(key) {
        Ok(s) if !s.trim().is_empty() => s
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect(),
        _ => default.to_vec(),
    }
}

/// Measure recall@10 + P50/P95 of one `cfg = (l_search, rerank_k)` config over
/// all queries, against the prebuilt ground truth. SEARCH-ONLY — no rebuild.
fn measure_config(
    idx: &SsdDiskAnnIndex,
    queries: &[Vec<f32>],
    gt: &[HashSet<u32>],
    k: usize,
    cfg: (usize, usize),
    guard: &RssGuard,
) -> (f64, f64, f64) {
    let (l_search, rerank_k) = cfg;
    let mut hits = 0usize;
    let mut lat_us: Vec<u128> = Vec::with_capacity(queries.len());
    for (qi, q) in queries.iter().enumerate() {
        guard.check().expect("RSS cap exceeded during sweep");
        let t = Instant::now();
        let res = idx
            .search_with_params(q, k, l_search, rerank_k)
            .expect("search_with_params");
        lat_us.push(t.elapsed().as_micros());
        for (id, _) in &res {
            if gt[qi].contains(&id.raw()) {
                hits += 1;
            }
        }
    }
    lat_us.sort_unstable();
    let recall = hits as f64 / (k * queries.len()) as f64;
    let p50 = lat_us[queries.len() / 2] as f64 / 1000.0;
    let p95 =
        lat_us[((queries.len() as f64 * 0.95) as usize).min(queries.len() - 1)] as f64 / 1000.0;
    (recall, p50, p95)
}

/// The PHASE-1 search-param sweep (ADR-189 §B recall/P95 recovery — the CHEAP
/// lever). RELOADS a prebuilt index when a nav sidecar is provided (no Vamana
/// rebuild) or BUILDS once (optionally saving a sidecar for the next process),
/// then sweeps `L_search × rerank_k` SEARCH-ONLY against the EXISTING
/// content-hashed GT and prints the real recall/P95 frontier.
///
/// ## Env (in addition to the `ssd_diskann_ga_gate` scale envs)
///
/// - `ARCGRAPH_VECTOR_GA_INDEX_PATH` — the f32 page store `.bin`. If set, BUILD
///   writes here (so it persists for reload) instead of a tempfile; RELOAD reads
///   here. Pair with the preserved 10M store.
/// - `ARCGRAPH_VECTOR_GA_NAV_PATH` — the nav sidecar (`save_nav` / `open`). If it
///   EXISTS (with an index-path that exists) the run RELOADS; otherwise the run
///   BUILDS and, when `ARCGRAPH_VECTOR_GA_SAVE_NAV=1`, writes the sidecar here.
/// - `ARCGRAPH_VECTOR_GA_LSEARCH_SET` (default `400,800,1600,3200`) — beam grid.
/// - `ARCGRAPH_VECTOR_GA_RERANK_SET`  (default `100,200,400`)        — rerank grid.
/// - `ARCGRAPH_VECTOR_GA_SWEEP_ASSERT` (default `0`) — when `1`, FAIL the test if
///   NO config reaches recall@10 ≥ 0.95 ∧ P95 ≤ 15 ms (a measured "no config
///   clears the bar" is an honest NEGATIVE result, not a test failure by
///   default — it justifies PHASE 2 RaBitQ-nav #758).
#[test]
#[ignore = "GA-sweep: heavy (RAM+CPU+disk); opt-in via ARCGRAPH_VECTOR_GA_BENCH_OK=1 + --ignored"]
fn ssd_diskann_ga_sweep() {
    // Panic-by-default env-gate (W25-MFI-2): NEVER silently skip.
    if std::env::var(GA_ENV).as_deref() != Ok("1") {
        panic!(
            "{GA_ENV} not set to 1 — the reload + search-param sweep is opt-in (heavy; \
             NEVER in CI). Set {GA_ENV}=1 (+ scale/path envs) to run. See module docs."
        );
    }

    let clusters = env_usize("ARCGRAPH_VECTOR_GA_CLUSTERS", 1000);
    let points_per = env_usize("ARCGRAPH_VECTOR_GA_POINTS", 100);
    let dim = env_usize("ARCGRAPH_VECTOR_GA_DIM", 768);
    let n_queries = env_usize("ARCGRAPH_VECTOR_GA_QUERIES", 1000);
    let build_batch = env_usize("ARCGRAPH_VECTOR_GA_BUILD_BATCH", 4096);
    let cap_mb = env_usize("ARCGRAPH_VECTOR_RSS_CAP_MB", DEFAULT_RSS_CAP_MB as usize) as u64;
    let encoding = GaEncoding::from_env();
    let k = 10;
    let n = clusters * points_per;
    let pool_frames = 8192;

    let l_set = env_list_usize("ARCGRAPH_VECTOR_GA_LSEARCH_SET", &[400, 800, 1600, 3200]);
    let rerank_set = env_list_usize("ARCGRAPH_VECTOR_GA_RERANK_SET", &[100, 200, 400]);
    let sweep_assert = std::env::var("ARCGRAPH_VECTOR_GA_SWEEP_ASSERT").as_deref() == Ok("1");

    let index_path = std::env::var("ARCGRAPH_VECTOR_GA_INDEX_PATH").ok();
    let nav_path = std::env::var("ARCGRAPH_VECTOR_GA_NAV_PATH").ok();

    eprintln!(
        "[SWEEP] N={n} (clusters={clusters}×{points_per}) dim={dim} queries={n_queries} \
         encoding={} L_set={l_set:?} rerank_set={rerank_set:?}",
        encoding.label()
    );

    let corpus = Corpus::new(clusters, points_per, dim);
    let guard = RssGuard::spawn(cap_mb, std::time::Duration::from_millis(500));

    // ── Acquire the index: RELOAD (no rebuild) or BUILD (once). ──
    let reload = match (&index_path, &nav_path) {
        (Some(ip), Some(np)) => Path::new(ip).exists() && Path::new(np).exists(),
        _ => false,
    };

    // The tempfile (if we build without an explicit persistent index path) must
    // outlive the index — bind it here.
    let mut _store_tmp: Option<tempfile::NamedTempFile> = None;

    let idx = if reload {
        let ip = index_path.as_ref().unwrap();
        let np = nav_path.as_ref().unwrap();
        eprintln!("[SWEEP] RELOAD: f32 store {ip} + nav sidecar {np} (no Vamana rebuild)");
        let t = Instant::now();
        let idx = SsdDiskAnnIndex::open(Path::new(ip), Path::new(np), pool_frames, &guard)
            .expect("reload (open) prebuilt index");
        eprintln!(
            "[SWEEP] RELOAD OK: {} vectors in {:.1}s (sidecar read, NOT a rebuild)",
            idx.len(),
            t.elapsed().as_secs_f64()
        );
        idx
    } else {
        // Dim-scaled params (V-1 #740) — identical to the gate's build.
        let params = if dim > 256 {
            DiskAnnParams {
                r: 128,
                l_construction: 200,
                ..DiskAnnParams::default()
            }
        } else {
            DiskAnnParams::default()
        };
        let n_train = n.min(100_000);
        let stride = (n / n_train).max(1);
        let train_storage: Vec<Vec<f32>> =
            (0..n).step_by(stride).map(|i| corpus.vector(i)).collect();
        let train_refs: Vec<&[f32]> = train_storage.iter().map(Vec::as_slice).collect();
        let nav = train_nav_quantizer(encoding, &train_refs);
        drop(train_storage);

        let cfg = SsdBuildConfig {
            dim,
            metric: Metric::L2,
            params,
            pool_frames,
            rerank_factor: RERANK_FACTOR,
            parallel_build_batch: Some(build_batch),
        };
        // Build to the persistent index path if given (so a sidecar pairing is
        // reloadable next run), else to a held tempfile.
        let store_path = if let Some(ip) = &index_path {
            std::path::PathBuf::from(ip)
        } else {
            let tmp = tempfile::NamedTempFile::new().unwrap();
            let p = tmp.path().to_path_buf();
            _store_tmp = Some(tmp);
            p
        };
        eprintln!(
            "[SWEEP] BUILD: {n} vectors → {} (Vamana refine; one-time)",
            store_path.display()
        );
        let t = Instant::now();
        let vectors = (0..n).map(|i| (VectorId::new(i as u32), corpus.vector(i)));
        let idx =
            SsdDiskAnnIndex::build(&store_path, &cfg, nav, vectors, &guard).unwrap_or_else(|e| {
                panic!(
                    "[SWEEP] BUILD FAILED (honest blocker): {e}; peak-RSS {} MB (cap {cap_mb})",
                    guard.peak_mb()
                )
            });
        eprintln!(
            "[SWEEP] BUILD OK: {n} vectors in {:.1}s; disk={:.2} GB; rss_peak={} MB",
            t.elapsed().as_secs_f64(),
            idx.disk_bytes() as f64 / 1e9,
            guard.peak_mb()
        );
        // Optionally mint a nav sidecar so the NEXT process reloads in minutes.
        if std::env::var("ARCGRAPH_VECTOR_GA_SAVE_NAV").as_deref() == Ok("1") {
            if let Some(np) = &nav_path {
                let t = Instant::now();
                idx.save_nav(Path::new(np)).expect("save_nav");
                eprintln!(
                    "[SWEEP] nav sidecar SAVED → {np} in {:.1}s (reload-ready)",
                    t.elapsed().as_secs_f64()
                );
            } else {
                eprintln!(
                    "[SWEEP] ARCGRAPH_VECTOR_GA_SAVE_NAV=1 but no ARCGRAPH_VECTOR_GA_NAV_PATH — skipping save"
                );
            }
        }
        idx
    };

    // ── Ground truth (exhaustive, content-hashed — reused across configs). ──
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|q| corpus.query(q)).collect();
    let t_gt = Instant::now();
    let gt = brute_force_gt(&corpus, &queries, k);
    eprintln!("[SWEEP] GT ready in {:.1}s", t_gt.elapsed().as_secs_f64());

    // ── The sweep: L_search × rerank_k, SEARCH-ONLY. ──
    eprintln!(
        "════════════════ ADR-189 §B SEARCH-PARAM SWEEP FRONTIER (N={n}, dim={dim}, encoding={}) ════════════════",
        encoding.label()
    );
    eprintln!(
        "  {:>9} {:>9} | {:>10} {:>9} {:>9} | gate(recall≥0.95 ∧ P95≤15ms)",
        "L_search", "rerank_k", "recall@10", "P50 ms", "P95 ms"
    );
    let mut best: Option<(usize, usize, f64, f64)> = None; // (L, rk, recall, p95) meeting gate, min P95
    let mut frontier: Vec<(usize, usize, f64, f64, f64)> = Vec::new();
    for &l_req in &l_set {
        for &rerank_k in &rerank_set {
            // The EFFECTIVE knobs `search_with_params` actually applies (rerank
            // floored at k; beam floored at the rerank set) — print these, not
            // the requested values, so a degenerate `l_req < rerank_k` row is
            // never mis-reported (HONESTY GATE).
            let eff_rerank = rerank_k.max(k);
            let l_search = l_req.max(eff_rerank).max(1);
            let (recall, p50, p95) =
                measure_config(&idx, &queries, &gt, k, (l_search, eff_rerank), &guard);
            let meets = recall >= 0.95 && p95 <= 15.0;
            eprintln!(
                "  {l_search:>9} {eff_rerank:>9} | {recall:>10.4} {p50:>9.3} {p95:>9.3} | {}",
                if meets { "✓ MEETS" } else { "·" }
            );
            frontier.push((l_search, eff_rerank, recall, p50, p95));
            let improves = match best {
                Some((_, _, _, bp95)) => p95 < bp95,
                None => true,
            };
            if meets && improves {
                best = Some((l_search, eff_rerank, recall, p95));
            }
        }
    }
    eprintln!(
        "═════════════════════════════════════════════════════════════════════════════════════════"
    );
    eprintln!(
        "  peak-RSS during sweep = {} MB (guard cap {cap_mb} MB)",
        guard.peak_mb()
    );

    match best {
        Some((l, rk, recall, p95)) => {
            eprintln!(
                "[SWEEP] RESULT: GATE MET at L_search={l}, rerank_k={rk} → recall@10={recall:.4}, P95={p95:.3} ms. \
                 encoding={}. Re-run `ssd_diskann_ga_gate` (or this config) to CONFIRM the §B gate green.",
                encoding.label()
            );
        }
        None => {
            let best_recall = frontier.iter().map(|f| f.2).fold(0.0_f64, f64::max);
            eprintln!(
                "[SWEEP] RESULT: NO config clears recall@10 ≥ 0.95 ∧ P95 ≤ 15 ms on the {} nav \
                 (best recall on this frontier = {best_recall:.4}). If this is the 10M run, the SQ8 nav \
                 has hit its recall/latency ceiling → PHASE 2 RaBitQ-nav (#758) is the real fix.",
                encoding.label()
            );
        }
    }

    if sweep_assert {
        assert!(
            best.is_some(),
            "GA-sweep ASSERT: no (L_search, rerank_k) config reached recall@10 ≥ 0.95 ∧ P95 ≤ 15 ms \
             (SQ8 nav ceiling at N={n}; honest negative → PHASE 2 RaBitQ #758)"
        );
    }
}
