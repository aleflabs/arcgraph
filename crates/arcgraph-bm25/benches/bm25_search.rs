//! M3.b BM25 perf-gate bench (ADR-039 §"Performance budget" + ADR-036
//! §D-24).
//!
//! Validates the v1.0 contract:
//!
//! | Workload                                   | P95 budget |
//! |--------------------------------------------|-----------:|
//! | BM25 top-10 at 1 M docs                    | ≤ 20 ms    |
//! | Filter overhead (`Filter::Any`)            | < 10 %     |
//! | Visibility-filter overhead (read-LSN mid)  | < 5 % of base |
//!
//! The 100 M-doc workload is correctly deferred to scale-validation
//! issue #76 per ADR-039 §"Performance budget"; this bench is the
//! 1 M-doc gate only.
//!
//! ## Shape
//!
//! Three Criterion groups, each `bench_function`-ing one
//! `Bm25IndexHandle::search` call per iteration. The 1 M-doc corpus
//! is built exactly once per process via [`OnceLock`] and the
//! resulting `Bm25Service` + `Bm25IndexHandle` are shared across all
//! groups. Build phase is OUTSIDE the Criterion measurement window
//! (Criterion measures only the closure body; the `OnceLock`
//! initialiser runs on first dereference, before any
//! `bench.iter(|| …)` cycle).
//!
//! - `bm25_top10_1m_docs`              — baseline `search(query, 10, Lsn::MAX)`.
//! - `bm25_top10_1m_docs_filter_any`   — `filtered_search(query, 10, &Filter::Any, Lsn::MAX)`.
//! - `bm25_top10_1m_docs_visibility`   — `search(query, 10, Lsn::new(500_000))`
//!   to exercise the visibility filter when it actually filters
//!   (~half the corpus is excluded because each doc carries a
//!   synthetic `commit_lsn` equal to its build-order index 0..1_000_000).
//!
//! Each group queries five rotating strings drawn from the
//! deterministic vocabulary so the branch predictor sees a mix of
//! selectivities (single-token vs phrase queries).
//!
//! ## Running
//!
//! ```bash
//! cargo bench -p arcgraph-bm25 --bench bm25_search
//! ```
//!
//! Wall-clock build takes 30 s – 3 min depending on dev hardware.
//! Criterion HTML reports land under `target/criterion/`.
//!
//! ## Acceptance
//!
//! Per ADR-039 §"Performance budget" + ADR-036 §D-24, the P95 of
//! `bm25_top10_1m_docs` MUST be ≤ 20 ms. ADR-035 amendment-04
//! precedent (§D-3 / §D-4) ratifies recording the empirical
//! posture in the commit message when dev hardware cannot reach the
//! production target — the hard gate is enforced at scale-validation
//! issue #76, not in dev CI.

use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use arcgraph_bm25::{Bm25IndexHandle, Bm25Service, Filter, IndexId};
use arcgraph_core::{Lsn, NodeId, TenantId};
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use criterion::{Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Deterministic vocabulary + corpus generator
// ---------------------------------------------------------------------------

/// Seed for the corpus PRNG. Pinned at 42 so the bench corpus is
/// byte-for-byte reproducible across runs and machines (the
/// Criterion sampling is the only source of variance).
const CORPUS_SEED: u64 = 42;

/// Number of docs in the bench corpus. Pinned to 1 M per ADR-039
/// §"Performance budget"; lowering this number invalidates the
/// budget gate and is forbidden by the slice prompt.
const CORPUS_SIZE: usize = 1_000_000;

/// Per-batch commit interval. 50 k docs per `IndexWriter::commit()`
/// gives a reasonable trade-off between per-commit overhead and
/// segment count for the M3 dev hardware envelope (M-series MBP).
const COMMIT_BATCH_SIZE: usize = 50_000;

/// Min / max words per doc. Each doc samples uniformly from this
/// range; matches the OPEN-Q-6 envelope in ADR-039 ("~50-word
/// average for Wikipedia-shaped corpora").
const DOC_MIN_WORDS: usize = 50;
const DOC_MAX_WORDS: usize = 200;

/// Lorem-ipsum-shaped vocabulary. ~500 distinct tokens — the
/// query-selectivity middle ground:
/// - Common tokens (e.g. `"lorem"`, `"in"`) hit O(N/k) docs and
///   dominate top-K with score-tail saturation.
/// - Rare tokens (e.g. `"machinatio"`, `"phantasma"`) hit O(N/V)
///   docs where V is the vocab size; the top-K is sharply
///   ranked.
///
/// 500 tokens is large enough that the top-10 is meaningful (we
/// rarely exhaust matching docs at 1 M) but small enough that the
/// inverted-index posting lists are dense — a worst-case-friendly
/// shape for the budget gate.
#[rustfmt::skip]
const LOREM_VOCAB: &[&str] = &[
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
    "sed", "do", "eiusmod", "tempor", "incididunt", "labore", "et", "dolore",
    "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis", "nostrud",
    "exercitation", "ullamco", "laboris", "nisi", "aliquip", "ex", "ea", "commodo",
    "consequat", "duis", "aute", "irure", "in", "reprehenderit", "voluptate",
    "velit", "esse", "cillum", "eu", "fugiat", "nulla", "pariatur", "excepteur",
    "sint", "occaecat", "cupidatat", "non", "proident", "sunt", "culpa", "qui",
    "officia", "deserunt", "mollit", "anim", "id", "est", "laborum", "vitae",
    "natus", "atque", "vero", "eos", "accusamus", "iusto", "odio", "dignissimos",
    "ducimus", "blanditiis", "praesentium", "voluptatum", "deleniti", "atque",
    "corrupti", "quos", "dolores", "quas", "molestias", "excepturi", "occaecati",
    "cupiditate", "similique", "tenetur", "sapiente", "delectus", "reiciendis",
    "voluptatibus", "maiores", "alias", "consequatur", "perferendis", "doloribus",
    "asperiores", "repellat", "aliquid", "earum", "rerum", "necessitatibus",
    "saepe", "eveniet", "voluptates", "repudiandae", "molestiae", "recusandae",
    "itaque", "neque", "porro", "quisquam", "dolorem", "magnam", "aliquam",
    "quaerat", "etiam", "tempora", "incidunt", "labore", "magna", "aliquam",
    "minima", "voluptatibus", "consectetur", "fermentum", "scelerisque", "fringilla",
    "pellentesque", "habitasse", "platea", "dictumst", "vestibulum", "ante",
    "primis", "faucibus", "orci", "luctus", "ultrices", "posuere", "cubilia",
    "curae", "phasellus", "tristique", "neque", "lacus", "tincidunt", "egestas",
    "vitae", "pellentesque", "sagittis", "feugiat", "magna", "fusce", "ornare",
    "metus", "sollicitudin", "varius", "malesuada", "fames", "turpis", "egestas",
    "morbi", "tristique", "senectus", "netus", "malesuada", "fames", "ipsum",
    "donec", "ultricies", "tellus", "ut", "tempor", "vestibulum", "ipsum",
    "primis", "in", "faucibus", "orci", "luctus", "et", "ultrices", "posuere",
    "cubilia", "curae", "vivamus", "vestibulum", "ipsum", "primis", "facere",
    "facilisis", "lectus", "iaculis", "interdum", "scelerisque", "purus",
    "machinatio", "phantasma", "obscurus", "veritas", "fortuna", "augurium",
    "imperium", "magisterium", "templum", "altare", "sacramentum", "mysterium",
    "fundamentum", "argumentum", "instrumentum", "monumentum", "ornamentum",
    "documentum", "elementum", "incrementum", "supplementum", "experimentum",
    "investigatio", "ratio", "actio", "natio", "operatio", "creatio", "destructio",
    "constructio", "instructio", "obstructio", "productio", "reductio", "deductio",
    "introductio", "conclusio", "fusio", "confusio", "diffusio", "infusio",
    "clavis", "navis", "avis", "civis", "finis", "ignis", "panis", "canis",
    "amicus", "inimicus", "antiquus", "nuntius", "studium", "principium",
    "exitus", "aditus", "fructus", "spiritus", "manus", "domus", "passus",
    "currus", "metus", "exitus", "casus", "sensus", "visus", "tactus",
    "auditus", "gustus", "olfactus", "luxus", "pluvius", "ventus", "tonitrus",
    "fulgur", "stella", "luna", "sol", "mundus", "terra", "aqua", "ignis",
    "aer", "lapis", "ferrum", "aurum", "argentum", "electrum", "plumbum",
    "stannum", "cuprum", "marmor", "saxum", "silex", "arena", "limus",
    "viridis", "rubeus", "caeruleus", "purpureus", "niger", "albus", "flavus",
    "magnus", "parvus", "longus", "brevis", "altus", "humilis", "latus",
    "angustus", "fortis", "debilis", "celer", "tardus", "novus", "vetus",
    "frigidus", "calidus", "siccus", "humidus", "asper", "lenis", "mollis",
    "durus", "levis", "gravis", "facilis", "difficilis", "verus", "falsus",
    "iustus", "iniustus", "bonus", "malus", "pulcher", "turpis", "felix",
    "miser", "carus", "hostilis", "amabilis", "odiosus", "doctus", "indoctus",
    "sapiens", "stultus", "audax", "timidus", "fidus", "infidus", "liber",
    "servus", "potens", "impotens", "saevus", "mitis", "dives", "pauper",
    "rex", "regina", "princeps", "consul", "senator", "miles", "civis",
    "imperator", "tribunus", "quaestor", "praetor", "legatus", "centurio",
    "auriga", "gladiator", "histrio", "pictor", "sculptor", "poeta", "scriptor",
    "orator", "philosophus", "magister", "discipulus", "medicus", "advocatus",
    "judex", "vates", "sacerdos", "augur", "haruspex", "pontifex", "vestal",
    "victoria", "gloria", "fama", "honor", "virtus", "pietas", "fides", "spes",
    "caritas", "pax", "bellum", "discordia", "concordia", "libertas", "servitus",
    "potestas", "dignitas", "felicitas", "calamitas", "fortitudo", "fortuna",
    "fatum", "destinatio", "providentia", "necessitas", "casus", "occasio",
    "tempus", "aeternitas", "memoria", "obvilio", "lacrima", "risus", "iocus",
    "labor", "otium", "negotium", "officium", "munus", "donum", "praemium",
    "poena", "merces", "stipendium", "tributum", "portus", "porta", "via",
    "iter", "patria", "exilium", "domus", "templum", "forum", "circus", "arena",
];

// ---------------------------------------------------------------------------
// Query strings — pinned per ADR-039 §"Performance budget" reproducibility
// ---------------------------------------------------------------------------

/// Five representative query strings drawn from the vocabulary. The
/// mix is deliberate:
/// - `"lorem ipsum"` — two-token phrase, very common (high-DF
///   posting lists, expensive scoring).
/// - `"voluptate"` — single-token, mid-frequency.
/// - `"consequat duis"` — two-token, mid-frequency phrase.
/// - `"exercitation ullamco"` — two-token, low-frequency.
/// - `"magna aliqua"` — two-token phrase.
///
/// The bench rotates through these via a counter so both the
/// branch predictor and Tantivy's posting-list cache see a realistic
/// query distribution rather than a degenerate one-query loop.
const QUERY_STRINGS: &[&str] = &[
    "lorem ipsum",
    "voluptate",
    "consequat duis",
    "exercitation ullamco",
    "magna aliqua",
];

/// Generate one synthetic doc body. Words drawn uniformly from
/// [`LOREM_VOCAB`]; word count uniform in
/// `[DOC_MIN_WORDS, DOC_MAX_WORDS)`. Output is a single-line
/// space-separated string suitable for the v1.0 `body` TEXT field.
fn generate_doc(rng: &mut StdRng) -> String {
    let n = rng.random_range(DOC_MIN_WORDS..DOC_MAX_WORDS);
    let mut buf = String::with_capacity(n * 8);
    for i in 0..n {
        if i > 0 {
            buf.push(' ');
        }
        let idx = rng.random_range(0..LOREM_VOCAB.len());
        buf.push_str(LOREM_VOCAB[idx]);
    }
    buf
}

// ---------------------------------------------------------------------------
// Corpus fixture (built ONCE per process via OnceLock)
// ---------------------------------------------------------------------------

/// The 1 M-doc Tantivy corpus. Hangs onto the [`TempDir`] guard so
/// the on-disk segments survive for the lifetime of the bench
/// process. All Criterion groups share the same [`Bm25IndexHandle`].
struct CorpusFixture {
    /// `TempDir` guard — held so its `Drop` runs at process exit, not
    /// before. The leading underscore appeases `dead_code` because
    /// the field is intentionally only consulted via its `Drop`.
    _tmp: TempDir,
    /// Workspace-level service. Held so the trait-object dispatch
    /// for `commit_pending` is reachable from build phase.
    _service: Arc<Bm25Service>,
    /// Per-tenant search-side handle. The hot loop calls
    /// `search` / `filtered_search` against this directly.
    handle: Arc<Bm25IndexHandle>,
    /// Wall-clock build duration (informational; logged at first
    /// dereference for the commit-message ratification trail).
    build_duration: Duration,
}

/// Process-global corpus. Initialised on first dereference; every
/// subsequent call returns a borrow of the same fixture.
static CORPUS: OnceLock<CorpusFixture> = OnceLock::new();

/// Lazy 1 M-doc corpus accessor. Builds the corpus once per process;
/// later calls are O(1).
///
/// The build phase:
/// 1. Allocate a [`TempDir`] under the OS tempdir.
/// 2. `Bm25Service::new(tmp.path())`.
/// 3. `service.handle(DEFAULT, DEFAULT_BM25)`.
/// 4. For each batch of [`COMMIT_BATCH_SIZE`] docs:
///    a. `upsert_document(NodeId(i), generate_doc(&mut rng), Lsn::new(i as u64))`.
///    b. After [`COMMIT_BATCH_SIZE`] upserts, dispatch
///    `commit_pending(DEFAULT)` via the trait-object impl on
///    `Bm25Service` so the segment is flushed and the reader
///    reloads.
///
/// The MVCC pinning is load-bearing: each doc is upserted with
/// `commit_lsn = i`, so the visibility-filter group can pick a
/// `read_lsn = 500_000` and exercise the filter on roughly half the
/// corpus.
fn corpus() -> &'static CorpusFixture {
    CORPUS.get_or_init(|| {
        let build_start = Instant::now();
        let tmp = TempDir::new().expect("tempdir for bm25 bench corpus");
        let data_dir: PathBuf = tmp.path().to_path_buf();
        let service = Bm25Service::new(data_dir);

        let handle = service
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("open default bm25 handle");

        // Trait-object dispatch for commit_pending — matches the
        // commit pipeline shape in `arcgraph-storage::crud` (ADR-039
        // §D-5). We hold a separate Arc<dyn Bm25IndexStoreHandle>
        // pointer because `service.commit_pending` is the trait
        // method, not an inherent one.
        let store_handle: Arc<dyn Bm25IndexStoreHandle> = Arc::clone(&service) as _;

        let mut rng = StdRng::seed_from_u64(CORPUS_SEED);

        // Build phase: 1M docs in 50k-doc batches with one
        // commit_pending per batch. The progress prints flush at
        // every batch boundary so a stalled bench is diagnosable.
        let mut docs_in_batch = 0_usize;
        let total_batches = CORPUS_SIZE.div_ceil(COMMIT_BATCH_SIZE);
        let mut batch_index = 0_usize;
        eprintln!(
            "[bm25_search bench] building {} doc corpus in {} batches of {} \
             (seed={}; vocab={}; words/doc=[{}, {}))",
            CORPUS_SIZE,
            total_batches,
            COMMIT_BATCH_SIZE,
            CORPUS_SEED,
            LOREM_VOCAB.len(),
            DOC_MIN_WORDS,
            DOC_MAX_WORDS,
        );
        for i in 0..CORPUS_SIZE {
            let body = generate_doc(&mut rng);
            // commit_lsn = i so the visibility-filter group can pick
            // read_lsn = 500_000 and exercise the filter on roughly
            // half the corpus. NodeId(i as u64) avoids the i==0
            // sentinel collision (NodeId::ZERO is unused at v1.0).
            handle
                .upsert_document(NodeId::new(i as u64 + 1), &body, Lsn::new(i as u64))
                .expect("upsert during corpus build");

            docs_in_batch += 1;
            if docs_in_batch >= COMMIT_BATCH_SIZE || i + 1 == CORPUS_SIZE {
                // Dispatch through the trait object so the path
                // exercised here matches the real commit pipeline.
                store_handle
                    .commit_pending(TenantId::DEFAULT)
                    .expect("commit_pending during corpus build");
                batch_index += 1;
                eprintln!(
                    "[bm25_search bench] batch {batch_index}/{total_batches} \
                     committed (cumulative docs = {})",
                    i + 1,
                );
                docs_in_batch = 0;
            }
        }

        let build_duration = build_start.elapsed();
        eprintln!(
            "[bm25_search bench] corpus build complete: {} docs in {:.2?} \
             ({} batches, {} docs/sec)",
            CORPUS_SIZE,
            build_duration,
            total_batches,
            (CORPUS_SIZE as f64 / build_duration.as_secs_f64()) as u64,
        );

        CorpusFixture {
            _tmp: tmp,
            _service: service,
            handle,
            build_duration,
        }
    })
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

/// Group 1 — baseline `search(query, 10, Lsn::MAX)` at 1 M docs.
///
/// This is THE budget gate per ADR-039 §"Performance budget":
/// P95 ≤ 20 ms.
///
/// `Lsn::MAX` selects every doc through the visibility filter (every
/// doc has `commit_lsn ≤ MAX` and `expired_lsn = MAX`), so this
/// measures the pure search-side cost without the visibility filter
/// actually filtering anything. The visibility filter's compose-cost
/// is still included because the filter is always built; only its
/// posting-list traversal is short-circuited by Tantivy when both
/// bounds are saturated.
fn bench_top10_1m_docs(c: &mut Criterion) {
    let fx = corpus();
    eprintln!(
        "[bm25_search bench] Group 1 starting against {}-doc corpus \
         (build = {:.2?})",
        CORPUS_SIZE, fx.build_duration,
    );

    let mut group = c.benchmark_group("bm25_top10_1m_docs");
    // Sample size + measurement time per the slice prompt: ≥ 50
    // samples, ≥ 10 s measurement window. This is the budget gate
    // — we want a tight P95 estimate so the commit message can
    // ratify the empirical posture honestly.
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    let mut tick: usize = 0;
    group.bench_function("baseline_search", |b| {
        b.iter(|| {
            let q = QUERY_STRINGS[tick % QUERY_STRINGS.len()];
            tick = tick.wrapping_add(1);
            let hits = fx
                .handle
                .search(black_box(q), 10, Lsn::MAX)
                .expect("search");
            black_box(hits);
        });
    });
    group.finish();
}

/// Group 2 — `filtered_search(query, 10, &Filter::Any, Lsn::MAX)` at
/// 1 M docs.
///
/// Validates the ADR-039 §"Performance budget" filter-overhead row
/// (< 10 % vs Group 1). At v1.0 only `Filter::Any` is supported;
/// internally `filtered_search` short-circuits to `search` (per
/// `handle.rs` `match filter { Filter::Any => self.search(...) }`),
/// so the overhead measured here is the cost of one variant-match
/// branch. The overhead ratio is computed offline by comparing P95s
/// across the two groups (Criterion does not natively express
/// cross-group ratios).
fn bench_top10_1m_docs_filter_any(c: &mut Criterion) {
    let fx = corpus();
    eprintln!("[bm25_search bench] Group 2 starting (Filter::Any short-circuit overhead)");

    let mut group = c.benchmark_group("bm25_top10_1m_docs_filter_any");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    let filter = Filter::Any;
    let mut tick: usize = 0;
    group.bench_function("filtered_any_search", |b| {
        b.iter(|| {
            let q = QUERY_STRINGS[tick % QUERY_STRINGS.len()];
            tick = tick.wrapping_add(1);
            let hits = fx
                .handle
                .filtered_search(black_box(q), 10, black_box(&filter), Lsn::MAX)
                .expect("filtered_search");
            black_box(hits);
        });
    });
    group.finish();
}

/// Group 3 — `search(query, 10, Lsn::new(500_000))` at 1 M docs.
///
/// The visibility filter actually filters here: every doc with
/// `commit_lsn > 500_000` (~half the corpus) is excluded. This
/// measures the cost of the visibility-filter range-traversal under
/// realistic snapshot semantics; the result feeds the
/// "Visibility-filter overhead < 5 % of base" row of ADR-039
/// §"Performance budget".
///
/// The overhead vs Group 1 is computed offline by comparing P95s.
/// ADR-039 §D-3 calls out that v1.0's `expired_lsn = MAX` means the
/// upper-bound clause is trivially true; only the
/// `commit_lsn ∈ [0, read_lsn]` clause does real work.
fn bench_top10_1m_docs_visibility(c: &mut Criterion) {
    let fx = corpus();
    eprintln!(
        "[bm25_search bench] Group 3 starting (visibility-filter at \
         read_lsn = 500_000 / mid-corpus)"
    );

    let mut group = c.benchmark_group("bm25_top10_1m_docs_visibility");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    // Mid-corpus read_lsn so the visibility filter excludes ~half
    // of the corpus rather than degenerating to "all visible" (top
    // half of CORPUS_SIZE LSN range is invisible).
    let read_lsn = Lsn::new((CORPUS_SIZE / 2) as u64);
    let mut tick: usize = 0;
    group.bench_function("visibility_filter_mid_lsn", |b| {
        b.iter(|| {
            let q = QUERY_STRINGS[tick % QUERY_STRINGS.len()];
            tick = tick.wrapping_add(1);
            let hits = fx
                .handle
                .search(black_box(q), 10, read_lsn)
                .expect("search at mid-LSN");
            black_box(hits);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_top10_1m_docs,
    bench_top10_1m_docs_filter_any,
    bench_top10_1m_docs_visibility,
);
criterion_main!(benches);
