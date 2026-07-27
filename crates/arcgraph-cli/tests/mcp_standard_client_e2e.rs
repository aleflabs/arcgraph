//! #846 / MUST-UI-04 — standard MCP client lifecycle over stdio.
//!
//! This test speaks newline-delimited JSON, matching standard MCP SDK stdio
//! clients. The older subprocess tests keep pinning legacy Content-Length
//! framing; this file proves the same production binary now supports both.

use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::time::timeout;

const SUBPROCESS_TIMEOUT: Duration = Duration::from_secs(30);
const NO_RESPONSE_TIMEOUT: Duration = Duration::from_millis(250);

fn spawn_stdio() -> (Child, ChildStdin, BufReader<tokio::process::ChildStdout>) {
    let bin = env!("CARGO_BIN_EXE_arcgraph-mcp-stdio");
    let mut child = Command::new(bin)
        .arg("--in-memory")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn arcgraph-mcp-stdio --in-memory");
    let stdin = child.stdin.take().expect("stdin piped");
    let stdout = BufReader::new(child.stdout.take().expect("stdout piped"));
    (child, stdin, stdout)
}

async fn write_line(stdin: &mut ChildStdin, req: &Value) {
    let mut body = serde_json::to_vec(req).expect("request serializes");
    body.push(b'\n');
    stdin.write_all(&body).await.expect("write request");
    stdin.flush().await.expect("flush request");
}

async fn read_line_response<R>(reader: &mut R) -> Value
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    let mut line = String::new();
    timeout(SUBPROCESS_TIMEOUT, reader.read_line(&mut line))
        .await
        .expect("response within timeout")
        .expect("read response line");
    assert!(!line.is_empty(), "EOF before response");
    serde_json::from_str(line.trim_end_matches(['\r', '\n'])).expect("response JSON")
}

fn content_length_frame(req: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(req).expect("request serializes");
    let mut out = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
    out.extend_from_slice(&body);
    out
}

async fn read_content_length_response<R>(reader: &mut R) -> Value
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    let mut content_length = None;
    let mut line = String::new();
    loop {
        line.clear();
        timeout(SUBPROCESS_TIMEOUT, reader.read_line(&mut line))
            .await
            .expect("header within timeout")
            .expect("read header");
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some(value) = trimmed.strip_prefix("Content-Length:") {
            content_length = Some(value.trim().parse::<usize>().expect("content length"));
        }
    }
    let len = content_length.expect("content length header");
    let mut body = vec![0u8; len];
    timeout(SUBPROCESS_TIMEOUT, reader.read_exact(&mut body))
        .await
        .expect("body within timeout")
        .expect("read body");
    serde_json::from_slice(&body).expect("body JSON")
}

async fn shutdown(mut child: Child, stdin: ChildStdin) {
    drop(stdin);
    let status = timeout(SUBPROCESS_TIMEOUT, child.wait())
        .await
        .expect("child exits")
        .expect("wait succeeds");
    assert!(status.success(), "child exited non-zero: {status:?}");
}

#[tokio::test]
async fn standard_mcp_newline_client_can_initialize_discover_and_call_tools() {
    let (child, mut stdin, mut stdout) = spawn_stdio();

    write_line(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "arcgraph-test", "version": "0"}
            }
        }),
    )
    .await;
    let init = read_line_response(&mut stdout).await;
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(
        init["result"]["capabilities"]["tools"]["listChanged"],
        false
    );
    assert_eq!(init["result"]["serverInfo"]["name"], "arcgraph");

    write_line(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }),
    )
    .await;
    let mut no_response = String::new();
    assert!(
        timeout(NO_RESPONSE_TIMEOUT, stdout.read_line(&mut no_response))
            .await
            .is_err(),
        "notifications/initialized must not produce a response: {no_response:?}"
    );

    write_line(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }),
    )
    .await;
    let listed = read_line_response(&mut stdout).await;
    let tools = listed["result"]["tools"].as_array().expect("tools array");
    let names = tools
        .iter()
        .map(|tool| tool["name"].as_str().expect("tool name"))
        .collect::<Vec<_>>();
    // Order matches `Dispatcher::wired_methods()`. This is the public
    // bare-database catalog pin.
    assert_eq!(
        names,
        vec![
            "graph.schema",
            "graph.inspect",
            "graph.explore",
            "graph.search",
            "graph.ingest",
            "graph.raw_query",
        ]
    );
    // ADR-004 10-tool cap: the advertised catalog must stay within 10.
    assert!(
        names.len() <= 10,
        "advertised catalog exceeds ADR-004 10-tool cap: {} tools {names:?}",
        names.len()
    );
    for tool in tools {
        assert!(
            !tool["description"].as_str().unwrap_or_default().is_empty(),
            "description is populated for {tool:?}"
        );
        assert_eq!(tool["inputSchema"]["type"], "object");
    }
    let raw_query = tools
        .iter()
        .find(|tool| tool["name"] == "graph.raw_query")
        .expect("raw_query listed");
    assert!(
        raw_query["inputSchema"]["required"]
            .as_array()
            .expect("required array")
            .iter()
            .any(|value| value == "query")
    );

    write_line(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "graph.schema",
                "arguments": {"tenant_id": 1, "format": "json"}
            }
        }),
    )
    .await;
    let schema_call = read_line_response(&mut stdout).await;
    assert_eq!(schema_call["result"]["content"][0]["type"], "text");
    let schema_text = schema_call["result"]["content"][0]["text"]
        .as_str()
        .expect("text content");
    serde_json::from_str::<Value>(schema_text).expect("tool result text is JSON");
    assert_eq!(schema_call["result"]["isError"], false);

    write_line(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "graph.nope",
                "arguments": {}
            }
        }),
    )
    .await;
    let unknown = read_line_response(&mut stdout).await;
    assert_eq!(unknown["error"]["code"], -32602);
    assert!(unknown["result"].is_null());

    write_line(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "graph.raw_query",
                "arguments": {
                    "tenant_id": 1,
                    "query": "MATCH (n RETURN n",
                    "format": "json"
                }
            }
        }),
    )
    .await;
    let raw_query_fault = read_line_response(&mut stdout).await;
    assert_eq!(raw_query_fault["result"]["isError"], true);
    assert_eq!(raw_query_fault["result"]["content"][0]["type"], "text");
    let error_text = raw_query_fault["result"]["content"][0]["text"]
        .as_str()
        .expect("error text");
    let error_value: Value = serde_json::from_str(error_text).expect("error text JSON");
    assert!(error_value["code"].as_i64().is_some(), "tool error object");

    shutdown(child, stdin).await;
}

#[tokio::test]
async fn legacy_content_length_initialize_still_works() {
    let (child, mut stdin, mut stdout) = spawn_stdio();
    let req = json!({
        "jsonrpc": "2.0",
        "id": "legacy-init",
        "method": "initialize",
        "params": {"protocolVersion": "2025-03-26"}
    });
    stdin
        .write_all(&content_length_frame(&req))
        .await
        .expect("write framed initialize");
    stdin.flush().await.expect("flush framed initialize");

    let response = read_content_length_response(&mut stdout).await;
    assert_eq!(response["id"], "legacy-init");
    assert_eq!(response["result"]["protocolVersion"], "2025-03-26");
    assert_eq!(response["result"]["serverInfo"]["name"], "arcgraph");

    shutdown(child, stdin).await;
}
