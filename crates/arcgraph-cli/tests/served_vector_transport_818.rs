//! #818 — served vector search over the STDIO TRANSPORT (the real served path).
//!
//! ## What this guards
//!
//! #818: a single-batch `graph.ingest` whose Content-Length-framed JSON-RPC
//! request exceeds the (old, shared) 16 MiB `MAX_MESSAGE_BYTES` cap was
//! SILENTLY rejected by the stdio framer — so the lazily-built served HNSW
//! had nothing to build and `graph.search` returned EMPTY (recall 0) for any
//! tenant above ~6 300 × 128-d vectors (16 MiB ÷ ~2 665 bytes/node). No
//! error surfaced to the customer harness (which ignores ingest-response
//! errors) and no server log fired. The fix (`arcgraph_mcp::jsonrpc`) frames
//! the *trusted-local* stdio transport at `STDIO_MAX_MESSAGE_BYTES` (512 MiB)
//! so bulk ingest works at real scale, drains+logs genuinely-over-cap frames
//! (never silent / never desyncing), and keeps the untrusted-network HTTP cap
//! at 16 MiB.
//!
//! ## Why a SUBPROCESS test (not the in-process `served_vector_search_e2e.rs`)
//!
//! The bug lives in the stdio TRANSPORT (the framer's per-message cap). The
//! in-process provider has no such cap, so an in-process test is GREEN before
//! AND after the fix — useless as a #818 regression (doctrine §3: a green
//! test that can't fail on its bug is worse than no test). This test drives
//! the REAL `arcgraph-mcp-stdio` binary (the same `serve_stdio` + framer that
//! `arcgraph serve --stdio-mcp --in-memory` uses) over framed stdio, so it is
//! RED before the fix and GREEN after.
//!
//! ## Test tiers
//!
//! - [`transport_large_ingest_frame_accepted_above_16mib_cap`] — NORMAL SUITE,
//!   FAST. Ingests N=7 000 (≈18 MiB) and N=20 000 (≈51 MiB) single-batch over
//!   the transport and asserts an EXACT `inserted_count == N`. No `graph.search`
//!   → no (lazy) HNSW build → seconds, not minutes. This is the load-bearing
//!   #818 oracle: RED before the fix (the frame is rejected → 0 records).
//! - [`transport_recall_ladder_above_cliff`] — GATED (`#[ignore]` + opt-in
//!   `ARCGRAPH_VECTOR_SCALE_LADDER=1`; release recommended). Full end-to-end
//!   `graph.search` recall@10 ≥ 0.95 vs a brute-force oracle at 2k/5k/7k/20k
//!   (+100k when `ARCGRAPH_VECTOR_SCALE_100K=1`). Heavy: the HNSW build at
//!   M=32/ef=200 is ~minutes in debug for a past-cliff corpus, so it is opt-in
//!   and PANICS-by-default if invoked without the flag (no silent skip, per
//!   `feedback_test_env_gate_panic_by_default`). Doubles as the ADR-133
//!   Index-class active-verification recipe.

use std::collections::HashSet;
use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::timeout;

/// Deterministic LCG → f32 in `[0, 1)` (matches `served_vector_search_e2e.rs`):
/// no `rand` dep, no clock, reproducible across runs.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Generate `n` deterministic `dim`-d f32 vectors from `seed`.
fn gen_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    (0..n)
        .map(|_| (0..dim).map(|_| lcg(&mut s)).collect())
        .collect()
}

/// Build a `graph.ingest` JSON-RPC request for `vectors` as the `embedding`
/// node property, external_id `v-{i}`. This is the exact wire shape the
/// customer-zero harness sends (`{"embedding": [f64, ...]}`), so the framed
/// request reproduces the #818 frame size.
fn ingest_request(tenant: u64, vectors: &[Vec<f32>]) -> Value {
    let nodes: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let emb: Vec<Value> = v.iter().map(|f| json!(f64::from(*f))).collect();
            json!({
                "external_id": format!("v-{i}"),
                "label": "Vec",
                "properties": { "embedding": emb },
            })
        })
        .collect();
    json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "graph.ingest",
        "params": {
            "tenant_id": tenant,
            "nodes": nodes,
            "relationships": [],
            "format": "json",
        }
    })
}

/// Render a JSON value as a Content-Length-framed envelope (matches
/// `arcgraph_mcp::jsonrpc::write_message`).
fn frame_request(req: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(req).expect("request serializes");
    let mut out = Vec::with_capacity(body.len() + 32);
    out.extend_from_slice(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes());
    out.extend_from_slice(&body);
    out
}

/// Read one Content-Length-framed JSON-RPC envelope (mirror of
/// `arcgraph_mcp::jsonrpc::read_message`'s wire format — reproduced inline so
/// a framer regression surfaces here, not as a silent passthrough).
async fn read_framed_response<R>(reader: &mut R) -> std::io::Result<Value>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    let mut content_length: Option<usize> = None;
    let mut line = String::new();
    loop {
        line.clear();
        let n = tokio::io::AsyncBufReadExt::read_line(reader, &mut line).await?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "EOF before headers complete",
            ));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some(value) = trimmed.strip_prefix("Content-Length:") {
            content_length = Some(value.trim().parse::<usize>().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad len: {e}"))
            })?);
        }
    }
    let len = content_length.ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "no Content-Length header")
    })?;
    let mut body = vec![0u8; len];
    AsyncReadExt::read_exact(reader, &mut body).await?;
    serde_json::from_slice(&body)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad JSON: {e}")))
}

/// Spawn the production stdio binary (default ephemeral in-memory substrate —
/// the same wiring as `arcgraph serve --stdio-mcp --in-memory`).
fn spawn_server() -> Child {
    Command::new(env!("CARGO_BIN_EXE_arcgraph-mcp-stdio"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn arcgraph-mcp-stdio")
}

/// Parse the `{format, body}` JSON-RPC result into the tool's body object.
fn tool_body(resp: &Value) -> Value {
    let body = resp
        .get("result")
        .and_then(|r| r.get("body"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("response has no result.body: {resp}"));
    serde_json::from_str(body).expect("tool body parses as JSON")
}

// ─────────────────────────────────────────────────────────────────────
// Tier 1 — NORMAL SUITE: large ingest frame is accepted over the transport.
// ─────────────────────────────────────────────────────────────────────

/// #818 load-bearing regression: a single-batch `graph.ingest` whose framed
/// request exceeds the old 16 MiB cap must now be ACCEPTED over the stdio
/// transport (all N nodes ingested), not silently rejected.
///
/// Ingest-only (no `graph.search`) so the lazy HNSW build is never triggered
/// → fast. RED before the fix: the framer rejected the >16 MiB frame with a
/// `-32700` envelope carrying no `result` → `inserted_count` is absent / 0.
#[tokio::test]
async fn transport_large_ingest_frame_accepted_above_16mib_cap() {
    // 7 000 × 128-d ≈ 18 MiB (just past the old cap; the #818 cliff) and
    // 20 000 × 128-d ≈ 51 MiB (well past) — both previously rejected.
    for &n in &[7_000usize, 20_000usize] {
        let vectors = gen_vectors(n, 128, 0x5765_7635);
        let req = ingest_request(1, &vectors);
        let framed = frame_request(&req);
        let frame_mib = framed.len() as f64 / 1024.0 / 1024.0;
        assert!(
            framed.len() > 16 * 1024 * 1024,
            "N={n} frame is {frame_mib:.2} MiB — must exceed the 16 MiB cap to test #818",
        );

        let mut child = spawn_server();
        let mut stdin = child.stdin.take().expect("stdin");
        let mut reader = BufReader::new(child.stdout.take().expect("stdout"));

        // Generous budget: the child parses a ~18–51 MiB frame + creates N
        // nodes (no HNSW build — that's lazy on first search).
        let budget = Duration::from_secs(120);
        let resp = timeout(budget, async {
            stdin.write_all(&framed).await.expect("write ingest frame");
            stdin.flush().await.expect("flush ingest frame");
            read_framed_response(&mut reader)
                .await
                .expect("framed ingest response")
        })
        .await
        .unwrap_or_else(|_| panic!("N={n}: ingest did not complete within {budget:?}"));

        // STRONG oracle: the response is a SUCCESS envelope (no error) and the
        // ingest summary reports EXACTLY N inserted. Before the fix this is a
        // `-32700` error envelope with no `result` → the body parse / count
        // assertion fails (RED).
        assert!(
            resp.get("error").is_none(),
            "N={n} ({frame_mib:.2} MiB frame): graph.ingest returned an error \
             envelope — the frame was rejected (the #818 cliff). resp={resp}",
        );
        let body = tool_body(&resp);
        assert_eq!(
            body["inserted_count"].as_u64(),
            Some(n as u64),
            "N={n} ({frame_mib:.2} MiB frame): inserted_count must equal N \
             (got {:?}) — the whole single-batch frame must be accepted, not \
             silently dropped above ~6 300 vectors (#818). body={body}",
            body["inserted_count"],
        );
        assert_eq!(
            body["failed_count"].as_u64(),
            Some(0),
            "N={n}: no record may fail. body={body}",
        );

        drop(stdin); // EOF → clean shutdown.
        let status = timeout(Duration::from_secs(30), child.wait())
            .await
            .expect("child exits within timeout")
            .expect("wait ok");
        assert!(
            status.success(),
            "N={n}: binary exited non-zero: {status:?}"
        );

        eprintln!("[#818] transport ingest N={n} frame={frame_mib:.2}MiB -> inserted_count={n} ✓");
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tier 2 — GATED: full end-to-end recall@10 ladder over the transport.
// ─────────────────────────────────────────────────────────────────────

/// Send one `graph.search` and return the hit node-ids (internal ids).
async fn search_over_transport<R, W>(
    stdin: &mut W,
    reader: &mut BufReader<R>,
    tenant: u64,
    query: &[f32],
    k: u64,
    id: u64,
) -> Vec<u64>
where
    R: tokio::io::AsyncRead + Unpin,
    W: AsyncWriteExt + Unpin,
{
    let qv: Vec<Value> = query.iter().map(|f| json!(f64::from(*f))).collect();
    let req = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "graph.search",
        "params": { "tenant_id": tenant, "query": "", "query_vec": qv, "k": k, "format": "json" }
    });
    let framed = frame_request(&req);
    stdin.write_all(&framed).await.expect("write search");
    stdin.flush().await.expect("flush search");
    let resp = read_framed_response(reader).await.expect("search response");
    assert!(resp.get("error").is_none(), "graph.search error: {resp}");
    let body = tool_body(&resp);
    body["hits"]
        .as_array()
        .expect("hits array")
        .iter()
        .map(|h| h["node_id"].as_u64().expect("node_id u64"))
        .collect()
}

/// GATED full recall ladder — opt-in via `ARCGRAPH_VECTOR_SCALE_LADDER=1`
/// (release recommended; the HNSW build is heavy in debug). PANICS by default
/// when invoked (e.g. via `--ignored`) without the flag — never a silent skip.
#[tokio::test]
#[ignore = "heavy (HNSW build at scale); opt-in via ARCGRAPH_VECTOR_SCALE_LADDER=1"]
async fn transport_recall_ladder_above_cliff() {
    assert_eq!(
        std::env::var("ARCGRAPH_VECTOR_SCALE_LADDER").as_deref(),
        Ok("1"),
        "this heavy opt-in test must be enabled with ARCGRAPH_VECTOR_SCALE_LADDER=1 \
         (release recommended). It is NEVER silently skipped (panic-by-default per \
         feedback_test_env_gate_panic_by_default).",
    );

    const DIM: usize = 128;
    const K: u64 = 10;
    const QUERIES: usize = 30;
    const FLOOR: f64 = 0.95;

    // The #818 cliff is between 6k and 7k; 7k/20k are the previously-EMPTY
    // (recall 0) cases. 2k/5k are below the cliff (worked before) and pin that
    // the fix is non-regressive there. 100k is the scalability floor (opt-in).
    let mut scales = vec![2_000usize, 5_000, 7_000, 20_000];
    if std::env::var("ARCGRAPH_VECTOR_SCALE_100K").as_deref() == Ok("1") {
        scales.push(100_000);
    }

    for n in scales {
        let vectors = gen_vectors(n, DIM, 0x5765_7635);
        let req = ingest_request(1, &vectors);
        let framed = frame_request(&req);
        let frame_mib = framed.len() as f64 / 1024.0 / 1024.0;

        let mut child = spawn_server();
        let mut stdin = child.stdin.take().expect("stdin");
        let mut reader = BufReader::new(child.stdout.take().expect("stdout"));

        // Ingest (budget scales with N: ~minutes for a 100k debug build via the
        // first search below; the ingest itself is fast).
        let ingest_resp = timeout(Duration::from_secs(600), async {
            stdin.write_all(&framed).await.expect("write ingest");
            stdin.flush().await.expect("flush ingest");
            read_framed_response(&mut reader)
                .await
                .expect("ingest response")
        })
        .await
        .unwrap_or_else(|_| panic!("N={n}: ingest timed out"));
        assert!(
            ingest_resp.get("error").is_none(),
            "N={n}: ingest error: {ingest_resp}"
        );
        let ib = tool_body(&ingest_resp);
        assert_eq!(
            ib["inserted_count"].as_u64(),
            Some(n as u64),
            "N={n}: inserted_count"
        );

        // internal_id -> vector index, from the ingest records.
        let mut id2idx: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        for rec in ib["records"].as_array().expect("records") {
            if rec["status"] == "inserted" {
                let ext = rec["external_id"].as_str().unwrap_or("");
                if let Some(idx) = ext.strip_prefix("v-").and_then(|s| s.parse::<usize>().ok()) {
                    id2idx.insert(rec["internal_id"].as_u64().expect("internal_id"), idx);
                }
            }
        }
        assert_eq!(
            id2idx.len(),
            n,
            "N={n}: every inserted record maps to a vector"
        );

        // Deterministic queries drawn from the data MANIFOLD: a corpus vector
        // + tiny noise (±0.001). This matches the customer-zero harness
        // (`cz_vthresh.py`: `train[i] + N(0, 0.001)`) and the ann-benchmarks
        // recall methodology (queries on the data distribution, each with a
        // well-defined true-NN set) — i.e. the EXACT measurement the issue
        // reported (recall 1.000 below the cliff, 0.000 above). The #818 fix
        // must restore it. recall@K vs an EXACT brute-force L2 oracle.
        //
        // NB (honesty): random-uniform queries in empty 128-d space are a
        // different, pathological worst-case (the 10-NN are far + arbitrary)
        // that conflates ANN recall-vs-`ef_search` tuning with the cliff; on
        // uniform-synthetic data those measure ~0.91 at N=20k / `ef_search`=128
        // — a recall-vs-N property, NOT the #818 cliff. See the PR report.
        let l2 =
            |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };
        let mut qseed = 0xC0FFEE_u64;
        let queries: Vec<Vec<f32>> = (0..QUERIES)
            .map(|_| {
                let idx = (lcg(&mut qseed) * n as f32) as usize % n;
                vectors[idx]
                    .iter()
                    .map(|&c| c + (lcg(&mut qseed) - 0.5) * 0.002)
                    .collect()
            })
            .collect();
        let mut total = 0.0_f64;
        // First search pays the lazy HNSW build (the dominant cost at scale);
        // generous budget for a 100k debug build.
        for (qi, q) in queries.iter().enumerate() {
            let hit_ids = timeout(
                Duration::from_secs(900),
                search_over_transport(&mut stdin, &mut reader, 1, q, K, 100 + qi as u64),
            )
            .await
            .unwrap_or_else(|_| panic!("N={n} q{qi}: search timed out"));
            assert_eq!(hit_ids.len(), K as usize, "N={n} q{qi}: must return K hits");
            let hit_idx: HashSet<usize> = hit_ids
                .iter()
                .filter_map(|id| id2idx.get(id).copied())
                .collect();

            let mut exact: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2(q, v)))
                .collect();
            exact.sort_by(|a, b| a.1.total_cmp(&b.1));
            let exact_idx: HashSet<usize> =
                exact.iter().take(K as usize).map(|(i, _)| *i).collect();

            total += exact_idx.intersection(&hit_idx).count() as f64 / K as f64;
        }
        let recall = total / QUERIES as f64;
        eprintln!(
            "[#818] transport recall@{K} N={n} frame={frame_mib:.1}MiB queries={QUERIES}: {recall:.4} (floor {FLOOR})"
        );
        assert!(
            recall >= FLOOR,
            "N={n} ({frame_mib:.1} MiB frame): served recall@{K} {recall:.4} < floor {FLOOR} \
             — this is the #818 cliff (was 0.000 above ~6 300 vectors)",
        );

        drop(stdin);
        let _ = timeout(Duration::from_secs(60), child.wait()).await;
    }
}
