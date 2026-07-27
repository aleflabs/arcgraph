//! #1641 / #1643 release-blocker regression — served `RANK BY HYBRID`
//! must agree with `graph.search` over both MCP and Bolt.
//!
//! This is deliberately a subprocess + durable-store test. It starts the
//! shipped `arcgraph` binary as an MCP stdio server against a fresh real data
//! directory, ingests text and embeddings, proves `graph.schema` and
//! `graph.search`, and executes the two-operand hybrid query through
//! `graph.raw_query`. After a clean MCP shutdown, it reopens the same directory
//! through the shipped Bolt server and executes the identical query over the
//! Bolt 5.0 wire.
//!
//! The fixture is deliberately discriminating: vector ranks are A,B,C while
//! BM25 ranks are B,C,A. Returning vector/ingest order instead of performing
//! RRF therefore produces A,B,C and fails. `graph.search` is the correctness
//! oracle (not either transport), and the query's explicit score binding makes
//! both exact fused scores and order observable outside the engine.

use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr, TcpListener as StdTcpListener};
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use arcgraph_mcp::transport::bolt::{
    ClientMessage, MAGIC_PREAMBLE, PackValue, SERVER_ACCEPT_V5_0, decode, encode_client,
    message::{TAG_RECORD, TAG_SUCCESS},
    read_chunked_message, write_chunked_message,
};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::timeout;

const PROCESS_TIMEOUT: Duration = Duration::from_secs(30);
const WIRE_TIMEOUT: Duration = Duration::from_secs(10);

const QUERY: &str = "MATCH (d:Document) \
RANK BY HYBRID(\
VECTOR(d.embedding, [1.0, 0.0, 0.0], K = 3), \
TEXT(d.text, 'compiler', K = 3)) AS fusion_score \
WITH FUSION = RRF(k = 60) \
RETURN d.external_id AS external_id, fusion_score AS score";

struct ProcessGuard(Option<Child>);

impl ProcessGuard {
    fn new(child: Child) -> Self {
        Self(Some(child))
    }

    fn child_mut(&mut self) -> &mut Child {
        self.0.as_mut().expect("process still present")
    }

    fn take(&mut self) -> Child {
        self.0.take().expect("process still present")
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        if let Some(child) = self.0.as_mut() {
            let _ = child.start_kill();
        }
    }
}

struct McpSession {
    process: ProcessGuard,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl McpSession {
    async fn start(data_dir: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_arcgraph"))
            .args([
                "serve",
                "--stdio-mcp",
                "--data",
                data_dir.to_str().expect("UTF-8 data path"),
                "--admin-http",
                "",
                "--metrics-http",
                "",
                "--drain-grace-seconds",
                "0",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn served MCP subprocess");
        let stdin = child.stdin.take().expect("MCP stdin");
        let stdout = BufReader::new(child.stdout.take().expect("MCP stdout"));
        let mut session = Self {
            process: ProcessGuard::new(child),
            stdin: Some(stdin),
            stdout,
            next_id: 1,
        };
        let initialized = session
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "fix-1641-regression",
                        "version": "1"
                    }
                }),
            )
            .await;
        assert_eq!(
            initialized["result"]["protocolVersion"], "2025-06-18",
            "MCP initialize: {initialized}"
        );
        session.notify("notifications/initialized", json!({})).await;
        session
    }

    async fn write_message(&mut self, message: &Value) {
        let mut encoded = serde_json::to_vec(message).expect("MCP request serializes");
        encoded.push(b'\n');
        let stdin = self.stdin.as_mut().expect("MCP stdin open");
        timeout(PROCESS_TIMEOUT, stdin.write_all(&encoded))
            .await
            .expect("MCP request write timeout")
            .expect("write MCP request");
        timeout(PROCESS_TIMEOUT, stdin.flush())
            .await
            .expect("MCP request flush timeout")
            .expect("flush MCP request");
    }

    async fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.write_message(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        }))
        .await;

        let mut line = String::new();
        timeout(PROCESS_TIMEOUT, self.stdout.read_line(&mut line))
            .await
            .expect("MCP response timeout")
            .expect("read MCP response");
        assert!(
            !line.is_empty(),
            "MCP server closed before replying to {method}"
        );
        let response: Value = serde_json::from_str(line.trim_end()).expect("MCP response is JSON");
        assert_eq!(response["id"], id, "MCP response id: {response}");
        response
    }

    async fn notify(&mut self, method: &str, params: Value) {
        self.write_message(&json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }))
        .await;
    }

    async fn tool_raw(&mut self, name: &str, arguments: Value) -> Value {
        self.request(
            "tools/call",
            json!({
                "name": name,
                "arguments": arguments
            }),
        )
        .await
    }

    async fn tool(&mut self, name: &str, arguments: Value) -> Value {
        let response = self.tool_raw(name, arguments).await;
        tool_body(&response, name).unwrap_or_else(|error| panic!("{error}"))
    }

    async fn clean_close(mut self) {
        drop(self.stdin.take());
        let mut child = self.process.take();
        let status = timeout(PROCESS_TIMEOUT, child.wait())
            .await
            .expect("MCP subprocess exit timeout")
            .expect("wait for MCP subprocess");
        assert!(status.success(), "MCP subprocess exited {status}");
    }
}

fn tool_body(response: &Value, name: &str) -> Result<Value, String> {
    if !response["error"].is_null() {
        return Err(format!("{name} JSON-RPC error: {response}"));
    }
    if response["result"]["isError"] != false {
        return Err(format!("{name} tool error: {response}"));
    }
    let text = response["result"]["content"][0]["text"]
        .as_str()
        .ok_or_else(|| format!("{name} missing MCP text content: {response}"))?;
    let rendered: Value = serde_json::from_str(text)
        .map_err(|error| format!("{name} rendered envelope is not JSON: {error}; {text}"))?;
    let body = rendered["body"]
        .as_str()
        .ok_or_else(|| format!("{name} rendered envelope missing body: {rendered}"))?;
    serde_json::from_str(body).map_err(|error| format!("{name} body is not JSON: {error}; {body}"))
}

fn pick_loopback_port() -> u16 {
    let listener =
        StdTcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind ephemeral loopback port");
    listener.local_addr().expect("loopback address").port()
}

async fn start_bolt(data_dir: &Path) -> (ProcessGuard, TcpStream) {
    let port = pick_loopback_port();
    let address = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let child = Command::new(env!("CARGO_BIN_EXE_arcgraph"))
        .args([
            "serve",
            "--bolt",
            &address.to_string(),
            "--data",
            data_dir.to_str().expect("UTF-8 data path"),
            "--admin-http",
            "",
            "--metrics-http",
            "",
            "--drain-grace-seconds",
            "0",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn served Bolt subprocess");
    let mut process = ProcessGuard::new(child);
    let deadline = Instant::now() + PROCESS_TIMEOUT;
    loop {
        if let Some(status) = process
            .child_mut()
            .try_wait()
            .expect("poll Bolt subprocess")
        {
            panic!("Bolt subprocess exited before readiness: {status}");
        }
        match TcpStream::connect(address).await {
            Ok(stream) => return (process, stream),
            Err(error) if Instant::now() < deadline => {
                let _ = error;
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            Err(error) => panic!("Bolt readiness timed out: {error}"),
        }
    }
}

async fn stop_bolt(mut process: ProcessGuard) {
    let mut child = process.take();
    #[cfg(unix)]
    {
        let pid = child.id().expect("Bolt subprocess id");
        // SAFETY: `pid` is the live child process returned by `tokio::process`;
        // SIGTERM requests the server's installed graceful-shutdown path.
        let result = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
        assert_eq!(result, 0, "send SIGTERM to Bolt subprocess");
    }
    #[cfg(not(unix))]
    child.start_kill().expect("stop Bolt subprocess");

    let status = timeout(PROCESS_TIMEOUT, child.wait())
        .await
        .expect("Bolt subprocess exit timeout")
        .expect("wait for Bolt subprocess");
    #[cfg(unix)]
    assert!(
        status.success(),
        "Bolt subprocess must exit cleanly after SIGTERM: {status}"
    );
}

async fn bolt_send(stream: &mut TcpStream, message: &ClientMessage) -> PackValue {
    let mut encoded = Vec::new();
    encode_client(&mut encoded, message).expect("encode Bolt client message");
    timeout(WIRE_TIMEOUT, write_chunked_message(stream, &encoded))
        .await
        .expect("Bolt write timeout")
        .expect("write Bolt message");
    let payload = timeout(WIRE_TIMEOUT, read_chunked_message(stream))
        .await
        .expect("Bolt read timeout")
        .expect("read Bolt message")
        .expect("Bolt server closed before replying");
    decode(&payload, 0).expect("decode Bolt response").0
}

async fn bolt_handshake_and_hello(stream: &mut TcpStream) {
    let mut handshake = Vec::with_capacity(20);
    handshake.extend_from_slice(&MAGIC_PREAMBLE);
    handshake.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]);
    handshake.extend_from_slice(&[0; 12]);
    timeout(WIRE_TIMEOUT, stream.write_all(&handshake))
        .await
        .expect("Bolt handshake write timeout")
        .expect("write Bolt handshake");
    let mut accepted = [0_u8; 4];
    timeout(WIRE_TIMEOUT, stream.read_exact(&mut accepted))
        .await
        .expect("Bolt handshake read timeout")
        .expect("read Bolt handshake");
    assert_eq!(accepted, SERVER_ACCEPT_V5_0);

    let hello = bolt_send(
        stream,
        &ClientMessage::Hello {
            user_agent: Some("fix-1641-regression/1".into()),
            scheme: Some("basic".into()),
            principal: Some("neo4j".into()),
            credentials: Some("test-only".into()),
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .await;
    assert!(
        matches!(
            hello,
            PackValue::Struct {
                tag: TAG_SUCCESS,
                ..
            }
        ),
        "Bolt HELLO failed: {hello:?}"
    );
}

async fn bolt_run_and_pull(
    stream: &mut TcpStream,
    query: &str,
) -> Result<Vec<Vec<PackValue>>, String> {
    let run = bolt_send(
        stream,
        &ClientMessage::Run {
            query: query.into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .await;
    if !matches!(
        run,
        PackValue::Struct {
            tag: TAG_SUCCESS,
            ..
        }
    ) {
        return Err(format!("Bolt RUN failed: {run:?}"));
    }

    let mut encoded = Vec::new();
    encode_client(&mut encoded, &ClientMessage::Pull { n: -1, qid: None })
        .expect("encode Bolt PULL");
    timeout(WIRE_TIMEOUT, write_chunked_message(stream, &encoded))
        .await
        .map_err(|_| "Bolt PULL write timed out".to_string())?
        .map_err(|error| format!("Bolt PULL write failed: {error}"))?;

    let mut rows = Vec::new();
    loop {
        let payload = timeout(WIRE_TIMEOUT, read_chunked_message(stream))
            .await
            .map_err(|_| "Bolt PULL read timed out".to_string())?
            .map_err(|error| format!("Bolt PULL read failed: {error}"))?
            .ok_or_else(|| "Bolt server closed during PULL".to_string())?;
        match decode(&payload, 0)
            .map_err(|error| format!("decode Bolt PULL response: {error}"))?
            .0
        {
            PackValue::Struct {
                tag: TAG_RECORD,
                mut fields,
            } => match fields.pop() {
                Some(PackValue::List(row)) if fields.is_empty() => rows.push(row),
                other => {
                    return Err(format!(
                        "malformed Bolt RECORD: {other:?}; remainder={fields:?}"
                    ));
                }
            },
            PackValue::Struct {
                tag: TAG_SUCCESS, ..
            } => return Ok(rows),
            other => return Err(format!("Bolt PULL failed: {other:?}")),
        }
    }
}

fn bolt_rows_as_json(rows: &[Vec<PackValue>]) -> Value {
    Value::Array(
        rows.iter()
            .map(|row| {
                Value::Array(
                    row.iter()
                        .map(|cell| match cell {
                            PackValue::String(value) => Value::String(value.clone()),
                            PackValue::Float(value) => json!(value),
                            other => panic!("unexpected Bolt result cell: {other:?}"),
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

fn substrate_flags(schema: &Value) -> BTreeMap<String, bool> {
    schema["indexes"]
        .as_array()
        .expect("schema indexes")
        .iter()
        .map(|entry| {
            (
                entry["kind"].as_str().expect("index kind").to_string(),
                entry["available"].as_bool().expect("index availability"),
            )
        })
        .collect()
}

fn search_hit_ids(search: &Value) -> Vec<u64> {
    search["hits"]
        .as_array()
        .expect("graph.search hits")
        .iter()
        .map(|hit| hit["node_id"].as_u64().expect("graph.search node_id"))
        .collect()
}

fn graph_search_rows(search: &Value) -> Value {
    Value::Array(
        search["hits"]
            .as_array()
            .expect("graph.search hits")
            .iter()
            .map(|hit| {
                let external_id = match hit["node_id"].as_u64().expect("graph.search node_id") {
                    1 => "doc-vector-first",
                    2 => "doc-bm25-first",
                    3 => "doc-middle",
                    other => panic!("unexpected graph.search node id {other}"),
                };
                json!([
                    external_id,
                    hit["score"].as_f64().expect("graph.search score")
                ])
            })
            .collect(),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn served_hybrid_fusion_matches_graph_search_over_mcp_and_bolt_1643() {
    let data_dir = tempfile::tempdir().expect("durable data directory");
    let b_score = 1.0_f64 / 62.0 + 1.0_f64 / 61.0;
    let a_score = 1.0_f64 / 61.0 + 1.0_f64 / 63.0;
    let c_score = 1.0_f64 / 63.0 + 1.0_f64 / 62.0;

    let mut mcp = McpSession::start(data_dir.path()).await;
    let ingest = mcp
        .tool(
            "graph.ingest",
            json!({
                "tenant_id": 1,
                "format": "json",
                "nodes": [
                    {
                        "external_id": "doc-vector-first",
                        "label": "Document",
                        "properties": {
                            "external_id": "doc-vector-first",
                            "text": "compiler architecture notes with many unrelated background words about gardening databases networks storage testing releases and operations",
                            "embedding": [1.0, 0.0, 0.0]
                        }
                    },
                    {
                        "external_id": "doc-bm25-first",
                        "label": "Document",
                        "properties": {
                            "external_id": "doc-bm25-first",
                            "text": "compiler compiler compiler compiler compiler compiler compiler compiler",
                            "embedding": [0.8, 0.6, 0.0]
                        }
                    },
                    {
                        "external_id": "doc-middle",
                        "label": "Document",
                        "properties": {
                            "external_id": "doc-middle",
                            "text": "compiler compiler guide",
                            "embedding": [0.0, 1.0, 0.0]
                        }
                    }
                ],
                "relationships": [],
                "acl_grants": [
                    {
                        "external_id": "doc-vector-first",
                        "read_principals": ["neo4j"]
                    },
                    {
                        "external_id": "doc-bm25-first",
                        "read_principals": ["neo4j"]
                    },
                    {
                        "external_id": "doc-middle",
                        "read_principals": ["neo4j"]
                    }
                ]
            }),
        )
        .await;
    assert_eq!(ingest["inserted_count"], 3, "served ingest: {ingest}");
    assert_eq!(ingest["failed_count"], 0, "served ingest: {ingest}");

    let schema = mcp
        .tool("graph.schema", json!({"tenant_id": 1, "format": "json"}))
        .await;
    assert_eq!(
        substrate_flags(&schema),
        BTreeMap::from([("bm25".into(), true), ("vector".into(), true)]),
        "schema must advertise both substrates: {schema}"
    );

    let vector_leg = mcp
        .tool(
            "graph.search",
            json!({
                "tenant_id": 1,
                "principal": "neo4j",
                "query": "",
                "query_vec": [1.0, 0.0, 0.0],
                "k": 3,
                "label_filter": ["Document"],
                "format": "json"
            }),
        )
        .await;
    assert_eq!(
        search_hit_ids(&vector_leg),
        vec![1, 2, 3],
        "vector leg must rank A,B,C: {vector_leg}"
    );

    let bm25_leg = mcp
        .tool(
            "graph.search",
            json!({
                "tenant_id": 1,
                "principal": "neo4j",
                "query": "compiler",
                "k": 3,
                "label_filter": ["Document"],
                "format": "json"
            }),
        )
        .await;
    assert_eq!(
        search_hit_ids(&bm25_leg),
        vec![2, 3, 1],
        "BM25 leg must oppose vector order as B,C,A: {bm25_leg}"
    );

    let search = mcp
        .tool(
            "graph.search",
            json!({
                "tenant_id": 1,
                "principal": "neo4j",
                "query": "compiler",
                "query_vec": [1.0, 0.0, 0.0],
                "k": 3,
                "label_filter": ["Document"],
                "format": "json"
            }),
        )
        .await;
    assert_eq!(search["k"], 3, "graph.search k: {search}");
    assert_eq!(
        search["hits"],
        json!([
            {
                "node_id": 2,
                "label": "Document",
                "score": b_score
            },
            {
                "node_id": 1,
                "label": "Document",
                "score": a_score
            },
            {
                "node_id": 3,
                "label": "Document",
                "score": c_score
            }
        ]),
        "graph.search oracle must return exact B,A,C RRF scores: {search}"
    );
    let oracle_rows = graph_search_rows(&search);

    let mcp_raw_response = mcp
        .tool_raw(
            "graph.raw_query",
            json!({
                "tenant_id": 1,
                "format": "json",
                "query": QUERY,
                "max_rows": 100
            }),
        )
        .await;
    let mcp_result = tool_body(&mcp_raw_response, "graph.raw_query");
    mcp.clean_close().await;

    let (bolt_process, mut bolt) = start_bolt(data_dir.path()).await;
    bolt_handshake_and_hello(&mut bolt).await;
    let bolt_result = bolt_run_and_pull(&mut bolt, QUERY)
        .await
        .map(|rows| bolt_rows_as_json(&rows));
    drop(bolt);
    stop_bolt(bolt_process).await;

    let (mcp_body, bolt_rows) = match (mcp_result, bolt_result) {
        (Ok(mcp_body), Ok(bolt_rows)) => (mcp_body, bolt_rows),
        (mcp, bolt) => {
            panic!(
                "#1643 served hybrid transport failure\n\
                 MCP graph.raw_query raw: {mcp:#?}\n\
                 Bolt RUN/PULL raw: {bolt:#?}"
            );
        }
    };

    let expected_rows = json!([
        ["doc-bm25-first", b_score],
        ["doc-vector-first", a_score],
        ["doc-middle", c_score]
    ]);
    assert_eq!(
        oracle_rows, expected_rows,
        "graph.search oracle must expose the exact expected rows"
    );
    assert_eq!(
        mcp_body["columns"],
        json!(["external_id", "score"]),
        "MCP columns: {mcp_body}"
    );
    assert_eq!(
        mcp_body["rows"], oracle_rows,
        "MCP must match graph.search exact scored rows: {mcp_body}"
    );
    assert_eq!(
        bolt_rows, oracle_rows,
        "Bolt must match graph.search exact scored rows"
    );
    assert_eq!(
        bolt_rows, mcp_body["rows"],
        "MCP graph.raw_query and Bolt must agree exactly"
    );
}
