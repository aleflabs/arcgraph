//! **#1186 (MUST-LLM-04)** — the per-tenant token-bucket `RateLimiter`
//! is shipped + unit-tested but was NOT wired into the served CLI
//! dispatcher, so NO per-tenant rate cap was enforceable on the real
//! served surface (DoS / noisy-neighbor exposure in multi-tenant
//! deployments).
//!
//! ## Root cause (code-verified)
//!
//! Both CLI dispatcher construction sites used
//! `Dispatcher::with_session_scope(...)` (which sets `rate_limiter:
//! None`) instead of `with_session_scope_and_rate_limiter(...)`. The
//! dispatch-time gate (`transport/mod.rs` `if let Some(limiter) =
//! self.rate_limiter` → `try_consume` → `-32007`) is a no-op when the
//! limiter is `None`.
//!
//! ## Default-OFF, flag-gated (`--rate-limit`)
//!
//! The fix wires the limiter behind a `--rate-limit` flag defaulting
//! **OFF**, NOT default-ON. Default-ON would re-introduce the **#833**
//! regression: the trusted-local stdio surface is the PRIMARY
//! agent-native read workload (a single agent issuing >100 sequential
//! reads/min — a recall sweep, multi-hop exploration — would have every
//! read past the ~100th rejected with `-32007`, which agent clients
//! coerce to an EMPTY result-set → silently-wrong answers; this is the
//! exact regime `cz833_stdio_read_lifecycle_e2e.rs` pins as MUST-stay-
//! unthrottled at the binary default). `--rate-limit` lets a
//! multi-tenant network operator opt INTO the cap (satisfying the
//! MUST-LLM-04 AC: "a customer can observe a per-tenant rate cap")
//! WITHOUT regressing the trusted-local default.
//!
//! ## Why a SUBPROCESS test (RED-on-revert)
//!
//! Same rationale as `cz833_stdio_read_lifecycle_e2e.rs`: the limiter
//! is wired in the BINARY's dispatcher construction, so only a test
//! that drives the REAL `arcgraph-mcp-stdio` binary over the REAL stdio
//! transport — WITH `--rate-limit` — is RED before the wiring and GREEN
//! after. An in-process test that builds its own dispatcher cannot
//! catch a revert of the binary wiring.
//!
//! ## The oracle
//!
//! Burst **150** `graph.schema` reads on tenant 1 over ONE
//! `--rate-limit` process. The read bucket capacity is 100 (refill ≈
//! 1.667/s, so a fast burst refills < 1 token over its sub-second
//! duration). Reads 1..~100 succeed; reads ~101..150 reject with
//! `-32007` (`CODE_RATE_LIMITED`). RED before the fix: ALL 150 succeed
//! (the limiter is `None`). GREEN after: ≥ 40 of the 150 reject with
//! `-32007` (a conservative lower bound — the exact split is
//! ~100/50 modulo the < 1 token refilled mid-burst).
//!
//! ADR-004 amendment-02 (rate-limit defaults) + ADR-133 §D-4 "MCP"
//! active-verification: real binary + real stdio transport.

use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::timeout;

/// `-32007` per `arcgraph_mcp::CODE_RATE_LIMITED`.
const CODE_RATE_LIMITED: i64 = -32007;

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

/// Spawn the production stdio binary WITH `--rate-limit` (opt-in
/// per-tenant cap) + `--in-memory`.
fn spawn_server_rate_limited() -> Child {
    Command::new(env!("CARGO_BIN_EXE_arcgraph-mcp-stdio"))
        .arg("--rate-limit")
        .arg("--in-memory")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn arcgraph-mcp-stdio --rate-limit")
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

/// `graph.schema` for tenant `t` with a unique json-rpc `id`.
/// `graph.schema` is `OpClass::Read` so it draws on the 100-token read
/// bucket — no ingest needed (the schema of the empty default tenant is
/// a valid SUCCESS envelope until the bucket exhausts).
fn schema_request(tenant: u64, id: u64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "graph.schema",
        "params": { "tenant_id": tenant, "format": "json" }
    })
}

/// Extract the JSON-RPC error code, if the envelope is an error.
fn error_code(resp: &Value) -> Option<i64> {
    resp.get("error")
        .and_then(|e| e.get("code"))
        .and_then(Value::as_i64)
}

// ─────────────────────────────────────────────────────────────────────
// THE #1186 wiring proof: 150 graph.schema reads over the real transport
// WITH --rate-limit; reads beyond the 100-token read bucket reject -32007.
// ─────────────────────────────────────────────────────────────────────

/// Burst **150** sequential `graph.schema` reads on tenant 1 over ONE
/// `--rate-limit` stdio process. Reads 1..~100 succeed; reads ~101..150
/// reject with `-32007` (`CODE_RATE_LIMITED`).
///
/// RED before the fix: the binary constructs the dispatcher via
/// `with_session_scope` (`rate_limiter: None`), so ALL 150 succeed and
/// the `rate_limited >= 40` assertion fails.
/// GREEN after: `with_session_scope_and_rate_limiter(.., RateLimiter::new())`
/// is wired behind `--rate-limit`, so the cap engages.
#[tokio::test]
async fn served_rate_limit_caps_read_burst_at_100_when_flag_set() {
    const TENANT: u64 = 1;
    const READS: u64 = 150;

    let mut child = spawn_server_rate_limited();
    let mut stdin = child.stdin.take().expect("stdin");
    let mut reader = BufReader::new(child.stdout.take().expect("stdout"));

    let (ok, rate_limited) = timeout(Duration::from_secs(120), async {
        let mut ok = 0u64;
        let mut rate_limited = 0u64;
        for id in 1..=READS {
            let resp = roundtrip(&mut stdin, &mut reader, &schema_request(TENANT, id)).await;
            match error_code(&resp) {
                None => ok += 1,
                Some(CODE_RATE_LIMITED) => rate_limited += 1,
                Some(other) => panic!("read #{id}: unexpected error code {other}. resp={resp}"),
            }
        }
        (ok, rate_limited)
    })
    .await
    .expect("150-read loop completed within budget");

    eprintln!(
        "[#1186] 150 graph.schema reads OVER STDIO with --rate-limit: \
         OK={ok}  rate_limited(-32007)={rate_limited}"
    );

    assert_eq!(ok + rate_limited, READS, "every read accounted for");
    // The first ~100 reads draw down the 100-token bucket; the
    // remainder reject. A conservative lower bound (≥40) absorbs the
    // < 1 token of mid-burst refill without flaking, while still being
    // RED when the limiter is `None` (rate_limited == 0).
    assert!(
        rate_limited >= 40,
        "expected the 100-token read cap to reject ≥40 of the 150 reads; \
         got rate_limited={rate_limited} (OK={ok}). When --rate-limit is \
         NOT wired, rate_limited==0 → this is the #1186 RED-on-revert oracle."
    );
    // Sanity: the cap is at 100, so a healthy chunk of the early reads
    // MUST succeed (the cap is a throttle, not a full block).
    assert!(
        ok >= 90,
        "expected ≥90 of the first ~100 reads to succeed before the bucket \
         exhausts; got OK={ok}"
    );

    drop(stdin); // EOF → clean shutdown.
    let status = timeout(Duration::from_secs(30), child.wait())
        .await
        .expect("child exits within timeout")
        .expect("wait ok");
    assert!(status.success(), "binary exited non-zero: {status:?}");
}

/// Control: WITHOUT `--rate-limit` (the default), the same 150-read
/// burst is UNTHROTTLED — every read succeeds. Pins the default-OFF
/// posture (the #833 trusted-local protection) so a future "make it
/// default-ON" change is RED here, not just silently regressing #833.
#[tokio::test]
async fn served_default_no_rate_limit_burst_unthrottled() {
    const TENANT: u64 = 1;
    const READS: u64 = 150;

    let mut child = Command::new(env!("CARGO_BIN_EXE_arcgraph-mcp-stdio"))
        .arg("--in-memory")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn arcgraph-mcp-stdio --in-memory");
    let mut stdin = child.stdin.take().expect("stdin");
    let mut reader = BufReader::new(child.stdout.take().expect("stdout"));

    let rate_limited = timeout(Duration::from_secs(120), async {
        let mut rate_limited = 0u64;
        for id in 1..=READS {
            let resp = roundtrip(&mut stdin, &mut reader, &schema_request(TENANT, id)).await;
            if error_code(&resp) == Some(CODE_RATE_LIMITED) {
                rate_limited += 1;
            }
        }
        rate_limited
    })
    .await
    .expect("150-read loop completed within budget");

    assert_eq!(
        rate_limited, 0,
        "default (no --rate-limit) MUST stay unthrottled (the #833 \
         trusted-local protection); got rate_limited={rate_limited}"
    );

    drop(stdin);
    let status = timeout(Duration::from_secs(30), child.wait())
        .await
        .expect("child exits within timeout")
        .expect("wait ok");
    assert!(status.success(), "binary exited non-zero: {status:?}");
}
