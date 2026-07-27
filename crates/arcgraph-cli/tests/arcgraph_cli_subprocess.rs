//! W15α M6-02 — subprocess integration tests for the `arcgraph`
//! umbrella CLI binary.
//!
//! Each test spawns the built `arcgraph` binary as a subprocess via
//! the cargo-set `CARGO_BIN_EXE_arcgraph` environment variable, runs
//! a subcommand, and verifies (a) exit code + (b) stdout/stderr
//! shape.
//!
//! The full Bolt protocol end-to-end test (HELLO → RUN → DISCARD)
//! lives at `crates/arcgraph-mcp/tests/...` — this test pins only
//! that the listener binds and exits cleanly on shutdown, which is
//! the "start server, run query, stop" exit criterion from
//! `docs/roadmap.md` M6-02.
//!
//! # Why these live here
//!
//! `env!("CARGO_BIN_EXE_arcgraph")` only resolves for tests in the
//! same package as the `arcgraph` bin target (same cargo convention
//! the W13δ `arcgraph-mcp-stdio` subprocess test uses).

use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;

const SUBPROCESS_TIMEOUT: Duration = Duration::from_secs(30);

// ─────────────────────────────────────────────────────────────────────
// `arcgraph serve --stdio-mcp` — exit clean on stdin EOF
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn serve_stdio_mcp_handles_graph_schema_then_exits_on_eof() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");

    // W28 / ADR-183 — `serve` refuses to start without an explicit storage
    // mode; this subprocess smoke is ephemeral, so it passes `--in-memory`.
    let mut child = Command::new(bin)
        .args(["serve", "--stdio-mcp", "--in-memory"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn arcgraph serve --stdio-mcp --in-memory");

    let mut stdin = child.stdin.take().expect("stdin piped");
    let stdout = child.stdout.take().expect("stdout piped");

    // Send one graph.schema request — same wire shape as the
    // standalone arcgraph-mcp-stdio binary test.
    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "graph.schema",
        "params": { "tenant_id": 1 }
    });
    let body = serde_json::to_vec(&request).expect("serialize request");
    let mut framed = Vec::with_capacity(body.len() + 32);
    framed.extend_from_slice(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes());
    framed.extend_from_slice(&body);

    timeout(SUBPROCESS_TIMEOUT, async {
        stdin.write_all(&framed).await.expect("write framed");
        stdin.flush().await.expect("flush");
    })
    .await
    .expect("write completed in time");

    // Read the framed response.
    let mut reader = BufReader::new(stdout);
    let response = timeout(SUBPROCESS_TIMEOUT, read_framed(&mut reader))
        .await
        .expect("response in time")
        .expect("framed response");

    assert_eq!(response.get("id").and_then(Value::as_i64), Some(1));
    assert!(
        response.get("error").is_none(),
        "expected success envelope: {response:?}"
    );

    // Close stdin — binary exits via PeerClosed.
    drop(stdin);

    let exit = timeout(SUBPROCESS_TIMEOUT, child.wait())
        .await
        .expect("exit in time")
        .expect("wait");
    assert!(exit.success(), "arcgraph serve --stdio-mcp exit: {exit:?}");
}

/// Read one Content-Length-framed JSON-RPC envelope. Mirrors the W13δ
/// `mcp_stdio_subprocess.rs` framer; kept inline to avoid leaking a
/// helper into the test crate's surface.
async fn read_framed<R>(reader: &mut R) -> std::io::Result<Value>
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
                "EOF before headers",
            ));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some(v) = trimmed.strip_prefix("Content-Length:") {
            content_length = Some(v.trim().parse().map_err(|e: std::num::ParseIntError| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad len: {e}"))
            })?);
        }
    }
    let len = content_length
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no Content-Length"))?;
    let mut body = vec![0u8; len];
    AsyncReadExt::read_exact(reader, &mut body).await?;
    serde_json::from_slice(&body)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad JSON: {e}")))
}

// ─────────────────────────────────────────────────────────────────────
// `arcgraph check` — empty-tenant smoke
// ─────────────────────────────────────────────────────────────────────

#[test]
fn check_no_data_prints_status_ok() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let out = std::process::Command::new(bin)
        .arg("check")
        .output()
        .expect("spawn arcgraph check");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "arcgraph check exit: {:?}\nstdout: {stdout}\nstderr: {stderr}",
        out.status
    );
    assert!(
        stdout.contains("status:   ok"),
        "expected 'status: ok' in stdout: {stdout}"
    );
    assert!(
        stdout.contains("empty-tenant"),
        "expected empty-tenant note: {stdout}"
    );
}

#[test]
fn check_with_existing_data_dir_succeeds() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let tmp = tempfile::tempdir().expect("create tmp dir");
    let out = std::process::Command::new(bin)
        .args(["check", "--data"])
        .arg(tmp.path())
        .output()
        .expect("spawn arcgraph check --data");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "arcgraph check --data exit: {:?}\nstdout: {stdout}\nstderr: {stderr}",
        out.status
    );
    assert!(
        stdout.contains(&format!("data-dir: {}", tmp.path().display())),
        "expected data-dir line in stdout: {stdout}"
    );
    assert!(
        stdout.contains("status:   ok (directory exists; no committed store found)"),
        "expected uninitialized-directory status in stdout: {stdout}"
    );
    assert!(
        !stdout.contains("status:   ok (empty-tenant"),
        "--data must not report the no-data empty-tenant status: {stdout}"
    );
}

#[test]
fn check_with_missing_data_dir_rejects() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let out = std::process::Command::new(bin)
        .args(["check", "--data", "/tmp/arcgraph-cli-test-nonexistent-xyz"])
        .output()
        .expect("spawn arcgraph check --data");
    assert!(
        !out.status.success(),
        "missing data dir should produce non-zero exit"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("does not exist"),
        "expected error message in stderr: {stderr}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// `arcgraph dump` — empty-tenant export envelope (JSON / TOON / Cypher)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn dump_json_prints_valid_envelope() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let out = std::process::Command::new(bin)
        .args(["dump", "--format", "json"])
        .output()
        .expect("spawn arcgraph dump --format json");
    assert!(out.status.success(), "arcgraph dump exit: {:?}", out.status);

    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: Value = serde_json::from_str(&stdout).expect("dump output is valid JSON");
    assert_eq!(parsed["format"], "json");
    assert_eq!(parsed["tenant_id"], 1);
    assert!(parsed["nodes"].as_array().expect("nodes array").is_empty());
    assert!(
        parsed["relationships"]
            .as_array()
            .expect("relationships array")
            .is_empty()
    );
}

#[test]
fn dump_toon_prints_toon_header() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let out = std::process::Command::new(bin)
        .args(["dump", "--format", "toon", "--tenant", "42"])
        .output()
        .expect("spawn arcgraph dump --format toon --tenant 42");
    assert!(out.status.success(), "dump toon exit: {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("# arcgraph dump format=toon"),
        "toon header missing: {stdout}"
    );
    assert!(
        stdout.contains("tenant_id: 42"),
        "tenant id missing: {stdout}"
    );
}

#[test]
fn dump_cypher_prints_cypher_comment_header() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let out = std::process::Command::new(bin)
        .args(["dump", "--format", "cypher"])
        .output()
        .expect("spawn arcgraph dump --format cypher");
    assert!(out.status.success(), "dump cypher exit: {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("// arcgraph dump format=cypher"),
        "cypher header missing: {stdout}"
    );
    assert!(
        stdout.contains("// tenant_id: 1"),
        "default tenant id missing: {stdout}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// #866 — DATA-SAFETY: `arcgraph dump --data <dir>` must FAIL LOUD, not
// emit a false-empty backup + exit 0, while the storage-rooted export is
// an unwired stub. This is the binary-level reproduction of the issue.
// ─────────────────────────────────────────────────────────────────────

/// The exact issue-#866 repro: `arcgraph dump --data <dir> > backup &&
/// echo OK` printed `OK` for a 0-record (comment-only) restore artifact
/// over a populated durable store. The pre-fix stub returned exit 0; the
/// fix makes `--data` refuse loudly.
///
/// RED-on-revert oracle: with the old `Ok(())` stub this asserts a clean
/// exit + empty output, so `!success()` fails — the false success returns.
/// The guard fires before any store is opened, so it is population-
/// independent (an existing empty `--data` dir hits the exact code path);
/// we therefore do not need to seed the store to reproduce the footgun.
#[test]
fn dump_with_data_refuses_false_empty_backup_866() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let dir = tempfile::tempdir().expect("tempdir");
    let out = std::process::Command::new(bin)
        .args(["dump", "--data"])
        .arg(dir.path())
        .args(["--format", "cypher"])
        .output()
        .expect("spawn arcgraph dump --data <dir> --format cypher");

    // (1) Non-zero exit — `dump --data <dir> && echo OK` can NEVER print OK.
    assert!(
        !out.status.success(),
        "arcgraph dump --data over the unwired stub MUST refuse (non-zero exit), \
         not emit a false-empty backup + exit 0 (#866); status: {:?}",
        out.status
    );

    // (2) Actionable stderr message naming the footgun + the issue.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("storage-rooted export is NOT yet implemented"),
        "stderr must explain the unwired stub: {stderr}"
    );
    assert!(
        stderr.contains("FALSE backup") && stderr.contains("#866"),
        "stderr must name the false-backup footgun + cite #866: {stderr}"
    );

    // (3) No partial/false backup leaked to stdout (the redirect target is
    //     empty — 0 CREATE statements — so nothing masquerades as a backup).
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.trim().is_empty() && !stdout.contains("CREATE"),
        "no backup artifact must be written to stdout on refusal: {stdout}"
    );
}

/// Format-independence at the binary level: `--data` + `--format json`
/// also refuses (the guard is upstream of the format dispatch).
#[test]
fn dump_with_data_refuses_json_format_866() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let dir = tempfile::tempdir().expect("tempdir");
    let out = std::process::Command::new(bin)
        .args(["dump", "--data"])
        .arg(dir.path())
        .args(["--format", "json"])
        .output()
        .expect("spawn arcgraph dump --data <dir> --format json");
    assert!(
        !out.status.success(),
        "dump --data --format json must refuse (#866); status: {:?}",
        out.status
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("#866"), "stderr must cite #866: {stderr}");
}

/// Honest-empty case (`--data` UNSET): no durable store is opened, so the
/// empty envelope is honest. We KEEP success + the stdout envelope (wire
/// shape preserved), but a prominent stderr WARNING must flag that this is
/// a stub, not a storage-rooted backup — so it is never mistaken for one.
#[test]
fn dump_without_data_warns_but_emits_envelope_866() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let out = std::process::Command::new(bin)
        .args(["dump", "--format", "json"])
        .output()
        .expect("spawn arcgraph dump --format json");
    assert!(
        out.status.success(),
        "dump without --data stays honest-empty success: {:?}",
        out.status
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: Value = serde_json::from_str(&stdout).expect("dump output is valid JSON");
    assert!(
        parsed["nodes"].as_array().expect("nodes array").is_empty(),
        "no --data ⇒ honest empty envelope: {stdout}"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("WARNING") && stderr.contains("STUB") && stderr.contains("#866"),
        "stderr must warn this is a stub, not a backup: {stderr}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// `arcgraph --help` smoke — verifies clap renders the surface
// ─────────────────────────────────────────────────────────────────────

#[test]
fn root_help_lists_all_subcommands() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let out = std::process::Command::new(bin)
        .arg("--help")
        .output()
        .expect("spawn arcgraph --help");
    assert!(
        out.status.success(),
        "arcgraph --help exit: {:?}",
        out.status
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    for sub in ["serve", "check", "dump"] {
        assert!(
            stdout.contains(sub),
            "--help missing subcommand {sub}: {stdout}"
        );
    }
}

#[test]
fn serve_http_without_cert_key_rejects_cleanly() {
    // #761 slice 1 — the M6-08+ "HTTPS/TLS wiring deferred" runtime
    // bail is GONE: `serve --http` now wires live TLS. `--http` WITHOUT
    // `--tls-cert`/`--tls-key` is a clean clap parse rejection (server-
    // side TLS is mandatory), NOT a panic. The full TLS roundtrip +
    // ADR-183 + loopback-default fault e2e live in
    // `tests/serve_http_tls_e2e.rs`.
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let out = std::process::Command::new(bin)
        .args(["serve", "--http", "127.0.0.1:18443"])
        .stdin(Stdio::null())
        .output()
        .expect("spawn arcgraph serve --http");
    assert!(
        !out.status.success(),
        "arcgraph serve --http without cert/key must reject"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("tls-cert") && stderr.contains("tls-key"),
        "clap must name the required cert+key flags: {stderr}"
    );
}
