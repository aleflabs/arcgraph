//! **#833 (P0, CZ Stage-3 of #818)** — the MCP **stdio read surface**
//! silently returned EMPTY for every read after ~100 reads/process.
//!
//! ## The bug (MEASURED root cause)
//!
//! `graph.search` + `graph.raw_query` are BOTH `OpClass::Read`
//! (`arcgraph_mcp::transport::op_class_for_method`), so they SHARE one
//! `(tenant, Read)` token bucket in the W14γ M5-12 per-tenant
//! rate-limiter — capacity **100**, refill ≈ 1.667/s. The stdio binary
//! wired that limiter onto the LOCAL stdio dispatcher. An agent driving
//! more than 100 reads/min (a recall sweep, multi-hop exploration, batch
//! inspection) had every read past the ~100th rejected with `-32007`,
//! which agent clients (langchain, the #818 `cz_sift_recall` harness)
//! coerce to an EMPTY result-set — silently-wrong answers. This is WHY
//! #818's served-vector recall pinned at **0.50** (the harness issues
//! 200 queries → the first ~100 return hits, the rest are rejected →
//! recall = 100/200). A probe proved it is NOT an MVCC snapshot leak
//! (`TxnManager::active_count()` stayed flat at 0 across 250 reads — the
//! per-read borrowed `Transaction` is dropped per call).
//!
//! The fix: the TRUSTED-LOCAL stdio surface runs UNTHROTTLED
//! (`Dispatcher::with_session_scope`, no rate-limiter) — mirroring the
//! same trusted-local-vs-untrusted-network split #818 already applies to
//! the frame cap (512 MiB stdio vs 16 MiB network). The limiter
//! primitive stays intact for the network HTTP/Bolt surfaces.
//!
//! ## Why a SUBPROCESS test (RED-on-revert)
//!
//! Same rationale as `served_vector_transport_818.rs`: the rate-limiter
//! is wired in the BINARY's dispatcher construction, so only a test that
//! drives the REAL `arcgraph-mcp-stdio` binary over the REAL stdio
//! transport is RED before the fix and GREEN after. An in-process test
//! that builds its own dispatcher cannot catch a revert of the binary
//! wiring. (Dispatcher-level properties — active_count-stays-bounded +
//! exhaustion-errors-not-empties — are pinned in
//! `crates/arcgraph-mcp/tests/cz833_stdio_read_no_leak_e2e.rs`.)
//!
//! ## Why #818's own test did not catch this
//!
//! `served_vector_transport_818.rs::transport_recall_ladder_above_cliff`
//! issues only `QUERIES = 30` searches per scale — UNDER the 100-token
//! read cap — so it never tripped the limiter. This test drives **250**
//! reads over ONE process, exactly the regime CZ's 200-query harness
//! hits.
//!
//! ADR-133 §D-4 "MCP"/"Driver" active-verification: real binary + real
//! stdio transport + 250-read roundtrip against pinned oracles.

use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::timeout;

/// Deterministic LCG → f32 in `[0, 1)` (matches the #818 harness).
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32) / ((1u64 << 24) as f32)
}

fn gen_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    (0..n)
        .map(|_| (0..dim).map(|_| lcg(&mut s)).collect())
        .collect()
}

/// `graph.ingest` request: `n` `Vec`-labelled nodes carrying an
/// `embedding` property (so the lazy served-HNSW has vectors to build
/// from + `graph.search` returns hits).
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
        "params": { "tenant_id": tenant, "nodes": nodes, "relationships": [], "format": "json" }
    })
}

fn frame_request(req: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(req).expect("request serializes");
    let mut out = Vec::with_capacity(body.len() + 32);
    out.extend_from_slice(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes());
    out.extend_from_slice(&body);
    out
}

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

/// Spawn the production stdio binary (default ephemeral in-memory — the
/// same wiring as `arcgraph serve --stdio-mcp --in-memory`).
fn spawn_server() -> Child {
    Command::new(env!("CARGO_BIN_EXE_arcgraph-mcp-stdio"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn arcgraph-mcp-stdio")
}

fn tool_body(resp: &Value) -> Value {
    let body = resp
        .get("result")
        .and_then(|r| r.get("body"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("response has no result.body: {resp}"));
    serde_json::from_str(body).expect("tool body parses as JSON")
}

/// One framed request → one framed response.
async fn roundtrip<R, W>(stdin: &mut W, reader: &mut BufReader<R>, req: &Value) -> Value
where
    R: tokio::io::AsyncRead + Unpin,
    W: AsyncWriteExt + Unpin,
{
    let framed = frame_request(req);
    stdin.write_all(&framed).await.expect("write request");
    stdin.flush().await.expect("flush request");
    read_framed_response(reader).await.expect("framed response")
}

fn search_request(tenant: u64, query: &[f32], k: u64, id: u64) -> Value {
    let qv: Vec<Value> = query.iter().map(|f| json!(f64::from(*f))).collect();
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "graph.search",
        "params": { "tenant_id": tenant, "query": "", "query_vec": qv, "k": k, "format": "json" }
    })
}

fn raw_query_request(tenant: u64, query: &str, id: u64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "graph.raw_query",
        "params": { "tenant_id": tenant, "query": query, "max_rows": 100, "format": "json" }
    })
}

// ─────────────────────────────────────────────────────────────────────
// THE #833 regression: 250 interleaved reads over the real transport.
// ─────────────────────────────────────────────────────────────────────

/// Drive **250** sequential reads over ONE stdio process, interleaving
/// `graph.search` (odd ids) and `graph.raw_query` (even ids). EVERY read
/// — most importantly reads 101..250 — must return a SUCCESS envelope
/// with the CORRECT non-empty result.
///
/// RED before the fix: reads past the ~100-token shared read bucket
/// return `-32007` (rate-limited) → the search/raw_query assertions fail.
/// GREEN after: the trusted-local stdio dispatcher is unthrottled.
#[tokio::test]
async fn transport_250_interleaved_reads_unthrottled_after_833_fix() {
    const TENANT: u64 = 1;
    const N: usize = 20;
    const DIM: usize = 16;
    const K: u64 = 5;
    const READS: u64 = 250;

    let vectors = gen_vectors(N, DIM, 0x5765_7635);

    let mut child = spawn_server();
    let mut stdin = child.stdin.take().expect("stdin");
    let mut reader = BufReader::new(child.stdout.take().expect("stdout"));

    let outcome = timeout(Duration::from_secs(120), async {
        // ── Seed: one bulk ingest of N Vec nodes with embeddings.
        //    `graph.ingest` is OpClass::Write — it does NOT touch the
        //    read bucket, so all 100 read tokens (pre-fix) would be
        //    available to the read loop below. ──
        let ingest = roundtrip(&mut stdin, &mut reader, &ingest_request(TENANT, &vectors)).await;
        assert!(ingest.get("error").is_none(), "ingest error: {ingest}");
        let ib = tool_body(&ingest);
        assert_eq!(
            ib["inserted_count"].as_u64(),
            Some(N as u64),
            "ingest must insert all {N} nodes: {ib}"
        );

        // ── 250 interleaved reads over the SAME process/session. ──
        let raw_q = "MATCH (n:Vec) RETURN count(*) AS c";
        let mut searches = 0u64;
        let mut raw_queries = 0u64;
        for id in 1..=READS {
            if id % 2 == 1 {
                // graph.search — query the first ingested vector; expect K hits.
                let resp = roundtrip(
                    &mut stdin,
                    &mut reader,
                    &search_request(TENANT, &vectors[0], K, id),
                )
                .await;
                assert!(
                    resp.get("error").is_none(),
                    "read #{id} (graph.search): rejected — the #833 ~100-read \
                     silent cap. resp={resp}"
                );
                let body = tool_body(&resp);
                let hits = body["hits"].as_array().expect("hits array");
                assert_eq!(
                    hits.len(),
                    K as usize,
                    "read #{id} (graph.search): must return {K} hits (reads \
                     101..250 were the silently-EMPTY zone in #833). body={body}"
                );
                searches += 1;
            } else {
                // graph.raw_query — COUNT over the Vec label; expect N.
                let resp = roundtrip(
                    &mut stdin,
                    &mut reader,
                    &raw_query_request(TENANT, raw_q, id),
                )
                .await;
                assert!(
                    resp.get("error").is_none(),
                    "read #{id} (graph.raw_query): rejected — the #833 ~100-read \
                     silent cap. resp={resp}"
                );
                let body = tool_body(&resp);
                let count = body["rows"][0][0].as_u64().unwrap_or_else(|| {
                    panic!("read #{id} (graph.raw_query): no count row. body={body}")
                });
                assert_eq!(
                    count, N as u64,
                    "read #{id} (graph.raw_query): COUNT must be {N} (a \
                     silently-empty read would render 0 / no rows). body={body}"
                );
                raw_queries += 1;
            }
        }
        (searches, raw_queries)
    })
    .await
    .expect("250-read loop completed within budget");

    let (searches, raw_queries) = outcome;
    assert_eq!(searches + raw_queries, READS, "every read accounted for");
    eprintln!(
        "[#833] 250 interleaved reads OVER STDIO TRANSPORT all correct \
         ({searches} graph.search + {raw_queries} graph.raw_query) — the \
         ~100-read silent cap is GONE ✓"
    );

    drop(stdin); // EOF → clean shutdown.
    let status = timeout(Duration::from_secs(30), child.wait())
        .await
        .expect("child exits within timeout")
        .expect("wait ok");
    assert!(status.success(), "binary exited non-zero: {status:?}");
}
