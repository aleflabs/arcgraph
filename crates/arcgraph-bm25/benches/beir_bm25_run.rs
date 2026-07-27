//! BEIR BM25-alone IR driver (#1266 — FTS Tier-1 benchmark leg).
//!
//! This is a *driver*, not a Criterion micro-bench: it reads a real BEIR
//! dataset (`corpus.jsonl` + `queries.jsonl`), indexes the corpus through the
//! exact `arcgraph-bm25` crate-direct surface the M3.b `bm25_search` template
//! uses (`Bm25Service::new` → `handle` → batched `upsert_document` +
//! `commit_pending` → `search`), runs each query, and emits a **TREC run
//! file** (`query_id Q0 doc_id rank score BM25`) for evaluation with
//! standard IR tooling.
//!
//! ## Scope: BM25-alone ONLY
//!
//! This driver measures the sparse BM25 index in isolation.
//!
//! ## Why a `harness = false` bench and not a `[[bin]]`
//!
//! Placing this in `benches/` reuses the crate's existing dev-dependency
//! envelope (`tempfile`, `criterion`) and mirrors the `bm25_search.rs`
//! template exactly — the BM25 surface it drives is dev-dep-only
//! (`arcgraph-storage::mutation_log::Bm25IndexStoreHandle`). `harness =
//! false` + a `main()` gives us a plain runnable driver with no Criterion
//! sampling loop (we want ONE pass over the corpus, not statistical
//! resampling). It is NEVER run in CI (no dataset is vendored — dependency and artifact policy).
//!
//! ## Inputs (all via env vars — no dataset vendored, dependency and artifact policy)
//!
//! | Env var                    | Meaning                                    | Default            |
//! |----------------------------|--------------------------------------------|--------------------|
//! | `BEIR_CORPUS`   (required) | path to `corpus.jsonl`                     | —                  |
//! | `BEIR_QUERIES`  (required) | path to `queries.jsonl`                    | —                  |
//! | `BEIR_RUN_OUT`  (required) | path to write the TREC run file            | —                  |
//! | `BEIR_K`                   | top-k retrieved per query                  | `100`              |
//! | `BEIR_COMMIT_BATCH`        | docs per `commit_pending` flush            | `50000`            |
//! | `BEIR_STATS_OUT`           | path to write a JSON stats sidecar         | (skipped if unset) |
//!
//! `corpus.jsonl`  lines: `{"_id": "...", "title": "...", "text": "..."}`
//! `queries.jsonl` lines: `{"_id": "...", "text": "..."}`
//!
//! The `body` indexed per doc is `title + " " + text` (the BEIR-canonical
//! concatenation; `pyserini`/`beir` build the same field). The string `_id`
//! is mapped to a dense `NodeId(i + 1)` (i is the 0-based line index; +1
//! avoids the `NodeId::ZERO` sentinel per the template) and the inverse
//! `NodeId → _id` map is retained to translate hits back to BEIR doc-ids for
//! the TREC run.
//!
//! ## Run
//!
//! ```bash
//! BEIR_CORPUS=~/beir-data/scifact/corpus.jsonl \
//! BEIR_QUERIES=~/beir-data/scifact/queries.jsonl \
//! BEIR_RUN_OUT=/tmp/scifact.run \
//! BEIR_STATS_OUT=/tmp/scifact.stats.json \
//!   cargo bench -p arcgraph-bm25 --bench beir_bm25_run
//! ```
//!
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arcgraph_bm25::{Bm25Service, IndexId};
use arcgraph_core::{Lsn, NodeId, TenantId};
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use serde_json::Value;
use tempfile::TempDir;

/// Read-snapshot LSN for the query phase: "see every committed doc".
///
/// Every doc is committed at `commit_lsn = i + 1` (`i` = 0-based line index),
/// so any `read_lsn >= n_docs` admits the whole corpus through the MVCC
/// visibility filter. We use `u64::MAX - 1` rather than `Lsn::MAX`
/// (`u64::MAX`) deliberately: `arcgraph_bm25::build_visibility_filter` carries
/// a `debug_assert!(read != u64::MAX)` guarding a `saturating_add(1)` boundary
/// (ADR-039 §D-3) that fires in DEBUG builds. `cargo bench` compiles in
/// release (assert elided), but pinning to `MAX - 1` keeps the driver correct
/// under a debug build too, and `MAX - 1` still exceeds every real corpus LSN
/// by ~19 orders of magnitude, so it sees everything.
const READ_ALL_LSN: Lsn = Lsn::new(u64::MAX - 1);

/// Read a required path env var or abort with a clear diagnostic. This driver
/// is opt-in (no dataset in-repo); a missing var means the operator did not
/// point it at a downloaded BEIR dataset, which is a usage error, not a bug.
fn require_env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| {
        eprintln!(
            "[beir_bm25_run] required env var {key} is not set.\n\
             This driver reads a downloaded BEIR dataset; nothing is vendored.\n\
             Example:\n  \
             BEIR_CORPUS=~/beir-data/scifact/corpus.jsonl \\\n  \
             BEIR_QUERIES=~/beir-data/scifact/queries.jsonl \\\n  \
             BEIR_RUN_OUT=/tmp/scifact.run \\\n    \
             cargo bench -p arcgraph-bm25 --bench beir_bm25_run"
        );
        std::process::exit(2);
    })
}

/// Parse an optional `usize` env var, falling back to `default` on unset /
/// malformed input.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

/// Extract a string field from a JSON object, defaulting to `""` for
/// missing / non-string values (BEIR docs occasionally omit `title`).
fn str_field<'a>(obj: &'a Value, key: &str) -> &'a str {
    obj.get(key).and_then(Value::as_str).unwrap_or("")
}

fn main() {
    let corpus_path = PathBuf::from(require_env("BEIR_CORPUS"));
    let queries_path = PathBuf::from(require_env("BEIR_QUERIES"));
    let run_out_path = PathBuf::from(require_env("BEIR_RUN_OUT"));
    let k = env_usize("BEIR_K", 100);
    let commit_batch = env_usize("BEIR_COMMIT_BATCH", 50_000).max(1);
    let stats_out = std::env::var("BEIR_STATS_OUT").ok();

    eprintln!(
        "[beir_bm25_run] corpus={} queries={} run_out={} k={} commit_batch={}",
        corpus_path.display(),
        queries_path.display(),
        run_out_path.display(),
        k,
        commit_batch,
    );

    // ---------------------------------------------------------------
    // 1. Build the BM25 index over the BEIR corpus.
    //    EXACT template loop: Bm25Service::new → handle → batched
    //    upsert_document + commit_pending.
    // ---------------------------------------------------------------
    let tmp = TempDir::new().expect("tempdir for beir bm25 index");
    let service = Bm25Service::new(tmp.path().to_path_buf());
    let handle = service
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("open default bm25 handle");
    // Trait-object dispatch for commit_pending (mirrors bm25_search.rs).
    let store_handle: Arc<dyn Bm25IndexStoreHandle> = Arc::clone(&service) as _;

    let corpus_file = File::open(&corpus_path)
        .unwrap_or_else(|e| panic!("open corpus {}: {e}", corpus_path.display()));
    let corpus_reader = BufReader::new(corpus_file);

    // NodeId(i+1) ↔ BEIR doc-id (`_id`). Vec index i ⇒ NodeId(i+1).
    let mut node_to_docid: Vec<String> = Vec::new();

    let build_start = Instant::now();
    let mut docs_in_batch = 0_usize;
    let mut n_docs = 0_usize;
    let mut skipped = 0_usize;

    for (lineno, line) in corpus_reader.lines().enumerate() {
        let line = line.unwrap_or_else(|e| panic!("read corpus line {lineno}: {e}"));
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let obj: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[beir_bm25_run] WARN skipping malformed corpus line {lineno}: {e}");
                skipped += 1;
                continue;
            }
        };
        let doc_id = str_field(&obj, "_id");
        if doc_id.is_empty() {
            eprintln!("[beir_bm25_run] WARN skipping corpus line {lineno} with empty _id");
            skipped += 1;
            continue;
        }
        // BEIR-canonical body = title + " " + text.
        let title = str_field(&obj, "title");
        let text = str_field(&obj, "text");
        let body = if title.is_empty() {
            text.to_owned()
        } else {
            format!("{title} {text}")
        };

        // i = current index into node_to_docid; NodeId = i + 1.
        let i = node_to_docid.len();
        node_to_docid.push(doc_id.to_owned());
        // commit_lsn = i + 1 (monotone, << u64::MAX). read_lsn =
        // READ_ALL_LSN (u64::MAX - 1) at query time makes every doc visible.
        handle
            .upsert_document(NodeId::new(i as u64 + 1), &body, Lsn::new(i as u64 + 1))
            .expect("upsert_document during corpus build");

        n_docs += 1;
        docs_in_batch += 1;
        if docs_in_batch >= commit_batch {
            store_handle
                .commit_pending(TenantId::DEFAULT)
                .expect("commit_pending during corpus build");
            eprintln!("[beir_bm25_run]   committed batch (cumulative docs = {n_docs})");
            docs_in_batch = 0;
        }
    }
    // Flush the tail batch.
    if docs_in_batch > 0 {
        store_handle
            .commit_pending(TenantId::DEFAULT)
            .expect("final commit_pending");
    }
    let build_secs = build_start.elapsed().as_secs_f64();
    eprintln!(
        "[beir_bm25_run] indexed {n_docs} docs ({skipped} skipped) in {build_secs:.2}s \
         ({:.0} docs/sec)",
        if build_secs > 0.0 {
            n_docs as f64 / build_secs
        } else {
            0.0
        },
    );

    // ---------------------------------------------------------------
    // 2. Run every query → search(query, k, READ_ALL_LSN) → TREC run.
    // ---------------------------------------------------------------
    let queries_file = File::open(&queries_path)
        .unwrap_or_else(|e| panic!("open queries {}: {e}", queries_path.display()));
    let queries_reader = BufReader::new(queries_file);

    let run_file = File::create(&run_out_path)
        .unwrap_or_else(|e| panic!("create run_out {}: {e}", run_out_path.display()));
    let mut run_writer = BufWriter::new(run_file);

    let query_start = Instant::now();
    let mut n_queries = 0_usize;
    let mut n_empty_result = 0_usize;
    let mut q_skipped = 0_usize;

    for (lineno, line) in queries_reader.lines().enumerate() {
        let line = line.unwrap_or_else(|e| panic!("read query line {lineno}: {e}"));
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let obj: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[beir_bm25_run] WARN skipping malformed query line {lineno}: {e}");
                q_skipped += 1;
                continue;
            }
        };
        let query_id = str_field(&obj, "_id");
        let query_text = str_field(&obj, "text");
        if query_id.is_empty() {
            eprintln!("[beir_bm25_run] WARN skipping query line {lineno} with empty _id");
            q_skipped += 1;
            continue;
        }

        let hits = handle
            .search(query_text, k, READ_ALL_LSN)
            .expect("search during query phase");
        if hits.is_empty() {
            n_empty_result += 1;
        }
        // TREC run format: `query_id Q0 doc_id rank score run_tag`.
        // rank is 1-based, score descending (already sorted by search).
        for (rank, (node_id, score)) in hits.iter().enumerate() {
            let idx = node_id.raw() as usize - 1; // inverse of NodeId(i+1)
            let doc_id = node_to_docid
                .get(idx)
                .map(String::as_str)
                .unwrap_or("__UNKNOWN__");
            writeln!(
                run_writer,
                "{query_id} Q0 {doc_id} {} {score} BM25",
                rank + 1,
            )
            .expect("write run line");
        }
        n_queries += 1;
    }
    run_writer.flush().expect("flush run file");
    let query_secs = query_start.elapsed().as_secs_f64();
    eprintln!(
        "[beir_bm25_run] ran {n_queries} queries ({q_skipped} skipped, \
         {n_empty_result} returned 0 hits) in {query_secs:.2}s \
         ({:.0} q/sec); run written to {}",
        if query_secs > 0.0 {
            n_queries as f64 / query_secs
        } else {
            0.0
        },
        run_out_path.display(),
    );

    // ---------------------------------------------------------------
    // 3. Optional JSON stats sidecar (for the results-table assembler).
    // ---------------------------------------------------------------
    if let Some(stats_path) = stats_out {
        let stats = serde_json::json!({
            "corpus_path": corpus_path.display().to_string(),
            "queries_path": queries_path.display().to_string(),
            "n_docs": n_docs,
            "n_docs_skipped": skipped,
            "n_queries": n_queries,
            "n_queries_skipped": q_skipped,
            "n_queries_empty_result": n_empty_result,
            "k": k,
            "commit_batch": commit_batch,
            "index_build_secs": build_secs,
            "query_phase_secs": query_secs,
        });
        let f = File::create(&stats_path)
            .unwrap_or_else(|e| panic!("create stats_out {stats_path}: {e}"));
        let mut w = BufWriter::new(f);
        serde_json::to_writer_pretty(&mut w, &stats).expect("write stats json");
        w.flush().expect("flush stats");
        eprintln!("[beir_bm25_run] stats written to {stats_path}");
    }
}
