//! [`Neo4jOracleExecutor`] — minimal Bolt 5.0 client that runs queries
//! against a `neo4j-community:5` Docker container.
//!
//! # Wire scope
//!
//! Implements the minimal client surface for read-only Cypher diffing:
//!
//! - Bolt 5.0 handshake (preamble + 4 version proposals).
//! - HELLO message with `BASIC` auth (`{principal: "neo4j", credentials: <password>}`).
//! - RUN + PULL ALL.
//! - Decodes SUCCESS / RECORD / FAILURE frames.
//!
//! # NOT in scope (forward-pinned to v1.1)
//!
//! - LOGOFF (we close the TCP socket).
//! - Routing / cluster membership.
//! - Date / DateTime / Duration / Point PackStream struct decoders
//!   (the harness's curated queries return scalar values).
//! - TLS — the harness assumes the Docker neo4j is on localhost and
//!   speaks plain Bolt. Production deployments use Bolt+SSC / Bolt+TLS.
//!
//! # Env-gating
//!
//! The constructor [`Neo4jOracleExecutor::connect_localhost`] returns a
//! structured [`ExecutorError::OracleUnavailable`] if the TCP connect
//! fails. The dual-execute test in `tests/dual_execute.rs` panics on
//! that error by default per `feedback_test_env_gate_panic_by_default.md`;
//! the opt-out is `ARCGRAPH_TCK_SKIP_OK=1`.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use arcgraph_mcp::transport::bolt::packstream::{PackValue, decode, encode};

use super::{ExecutorError, RowSet, TckExecutor};

/// Default Bolt port for a local `neo4j-community:5` Docker container.
pub const DEFAULT_BOLT_ADDR: &str = "127.0.0.1:7687";

/// Default credentials for the W18δ harness Docker neo4j (must match
/// the `NEO4J_AUTH` env passed to `docker run`).
pub const DEFAULT_USERNAME: &str = "neo4j";
pub const DEFAULT_PASSWORD: &str = "arcgraph-tck";

/// Read / write socket timeout. A misbehaving / overloaded Docker
/// container shouldn't hang the harness more than this.
pub const SOCKET_TIMEOUT_SECS: u64 = 10;

/// Bolt protocol minor versions this client requests in handshake. We
/// ask for Bolt 5.0 only — the W18δ harness pins to one wire version
/// to keep the encode/decode surface narrow.
const BOLT_5_0_VERSION_PROPOSAL: u32 = 0x00_00_00_05;
const BOLT_NULL_PROPOSAL: u32 = 0x00_00_00_00;
const BOLT_HANDSHAKE_PREAMBLE: u32 = 0x60_60_b0_17;

/// Bolt message tags (subset).
const MSG_HELLO: u8 = 0x01;
const MSG_RUN: u8 = 0x10;
const MSG_PULL: u8 = 0x3F;
const MSG_SUCCESS: u8 = 0x70;
const MSG_RECORD: u8 = 0x71;
const MSG_FAILURE: u8 = 0x7F;

/// Neo4j-Docker-backed TCK oracle executor.
pub struct Neo4jOracleExecutor {
    addr: String,
    username: String,
    password: String,
}

impl std::fmt::Debug for Neo4jOracleExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Neo4jOracleExecutor")
            .field("addr", &self.addr)
            .finish_non_exhaustive()
    }
}

impl Neo4jOracleExecutor {
    /// Connect to the default-localhost Docker neo4j. Returns
    /// [`ExecutorError::OracleUnavailable`] if the TCP connect fails
    /// (Docker container not running, port not exposed, etc.).
    pub fn connect_localhost() -> Result<Self, ExecutorError> {
        Self::probe(DEFAULT_BOLT_ADDR, DEFAULT_USERNAME, DEFAULT_PASSWORD)
    }

    /// Custom address / credentials variant. Tests pin this; the
    /// default-localhost variant is what the dual-execute harness
    /// calls.
    pub fn probe(addr: &str, username: &str, password: &str) -> Result<Self, ExecutorError> {
        // Quick connect probe (≤ 1 s) so a missing Docker container
        // doesn't hang the suite. We do NOT carry the stream
        // long-lived — every `execute` opens its own connection,
        // mirroring the v1.1 connection-per-query posture (no pooling
        // at the W18δ skeleton).
        let _stream = TcpStream::connect_timeout(
            &addr.parse().map_err(|e| {
                ExecutorError::OracleUnavailable(format!("invalid addr `{addr}`: {e}"))
            })?,
            Duration::from_secs(1),
        )
        .map_err(|e| {
            ExecutorError::OracleUnavailable(format!(
                "connect to {addr} failed: {e} \
                 (start the Docker neo4j with `docker run --rm -p 7687:7687 \
                 -e NEO4J_AUTH=neo4j/arcgraph-tck neo4j:5`)"
            ))
        })?;
        Ok(Self {
            addr: addr.to_string(),
            username: username.to_string(),
            password: password.to_string(),
        })
    }
}

impl TckExecutor for Neo4jOracleExecutor {
    fn name(&self) -> &'static str {
        "Neo4jOracle"
    }

    fn execute(&self, cypher: &str) -> Result<RowSet, ExecutorError> {
        let mut stream = TcpStream::connect_timeout(
            &self.addr.parse().map_err(|e| {
                ExecutorError::OracleUnavailable(format!("invalid addr `{}`: {e}", self.addr))
            })?,
            Duration::from_secs(SOCKET_TIMEOUT_SECS),
        )
        .map_err(|e| ExecutorError::OracleUnavailable(format!("connect failed: {e}")))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(SOCKET_TIMEOUT_SECS)))
            .map_err(|e| ExecutorError::Io(e.to_string()))?;
        stream
            .set_write_timeout(Some(Duration::from_secs(SOCKET_TIMEOUT_SECS)))
            .map_err(|e| ExecutorError::Io(e.to_string()))?;

        // ─── handshake ──────────────────────────────────────────────
        let mut handshake = Vec::with_capacity(20);
        handshake.extend_from_slice(&BOLT_HANDSHAKE_PREAMBLE.to_be_bytes());
        handshake.extend_from_slice(&BOLT_5_0_VERSION_PROPOSAL.to_be_bytes());
        handshake.extend_from_slice(&BOLT_NULL_PROPOSAL.to_be_bytes());
        handshake.extend_from_slice(&BOLT_NULL_PROPOSAL.to_be_bytes());
        handshake.extend_from_slice(&BOLT_NULL_PROPOSAL.to_be_bytes());
        stream
            .write_all(&handshake)
            .map_err(|e| ExecutorError::Io(format!("handshake write: {e}")))?;
        let mut version_buf = [0u8; 4];
        stream
            .read_exact(&mut version_buf)
            .map_err(|e| ExecutorError::Io(format!("handshake read: {e}")))?;
        if u32::from_be_bytes(version_buf) != BOLT_5_0_VERSION_PROPOSAL {
            return Err(ExecutorError::OracleUnavailable(format!(
                "server selected an unexpected Bolt version: {:?}",
                version_buf
            )));
        }

        // ─── HELLO ──────────────────────────────────────────────────
        let mut hello_fields: BTreeMap<String, PackValue> = BTreeMap::new();
        hello_fields.insert(
            "user_agent".to_string(),
            PackValue::String("arcgraph-tck/0.1".to_string()),
        );
        hello_fields.insert("scheme".to_string(), PackValue::String("basic".into()));
        hello_fields.insert(
            "principal".to_string(),
            PackValue::String(self.username.clone()),
        );
        hello_fields.insert(
            "credentials".to_string(),
            PackValue::String(self.password.clone()),
        );
        send_msg(&mut stream, MSG_HELLO, vec![PackValue::Map(hello_fields)])?;
        expect_success(&mut stream)?;

        // ─── RUN + PULL ─────────────────────────────────────────────
        send_msg(
            &mut stream,
            MSG_RUN,
            vec![
                PackValue::String(cypher.to_string()),
                PackValue::Map(BTreeMap::new()), // parameters
                PackValue::Map(BTreeMap::new()), // metadata
            ],
        )?;
        let run_success = read_summary(&mut stream)?;
        let columns = extract_columns(&run_success);

        let mut pull_fields: BTreeMap<String, PackValue> = BTreeMap::new();
        pull_fields.insert("n".to_string(), PackValue::Integer(-1));
        send_msg(&mut stream, MSG_PULL, vec![PackValue::Map(pull_fields)])?;

        let mut rows: Vec<Vec<String>> = Vec::new();
        loop {
            let (tag, mut fields) = read_message(&mut stream)?;
            match tag {
                MSG_RECORD => {
                    // RECORD's single field is a List of column values.
                    let row = match fields.pop() {
                        Some(PackValue::List(list)) => {
                            list.into_iter().map(packvalue_to_string).collect()
                        }
                        Some(other) => {
                            return Err(ExecutorError::Execution(format!(
                                "Bolt RECORD body unexpected shape: {other:?}"
                            )));
                        }
                        None => {
                            return Err(ExecutorError::Execution("Bolt RECORD body empty".into()));
                        }
                    };
                    rows.push(row);
                }
                MSG_SUCCESS => break, // PULL completed
                MSG_FAILURE => {
                    let msg = extract_failure_msg(&fields);
                    return Err(ExecutorError::Execution(format!(
                        "Bolt FAILURE during PULL: {msg}"
                    )));
                }
                other => {
                    return Err(ExecutorError::Execution(format!(
                        "unexpected Bolt message tag during PULL: 0x{other:02x}"
                    )));
                }
            }
        }

        Ok(RowSet { columns, rows })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Wire helpers
// ─────────────────────────────────────────────────────────────────────

/// Write `tag` + `fields` as a Bolt struct, chunked + terminated.
///
/// Uses `PackValue::Struct` so the encoding goes through the canonical
/// codec (single-source-of-truth for the marker byte + field count).
fn send_msg(stream: &mut TcpStream, tag: u8, fields: Vec<PackValue>) -> Result<(), ExecutorError> {
    let n = fields.len();
    if n > 15 {
        return Err(ExecutorError::Execution(format!(
            "TINY_STRUCT only supports up to 15 fields; got {n}"
        )));
    }
    let body_value = PackValue::Struct { tag, fields };
    let mut body: Vec<u8> = Vec::new();
    encode(&mut body, &body_value)
        .map_err(|e| ExecutorError::Execution(format!("PackStream encode failed: {e}")))?;
    // Chunking: split body into ≤65535-byte chunks; each prefixed with
    // u16-BE length. Trailing 2-byte zero terminator marks message end.
    let mut chunked: Vec<u8> = Vec::with_capacity(body.len() + 4);
    for chunk in body.chunks(65535) {
        let len = chunk.len() as u16;
        chunked.extend_from_slice(&len.to_be_bytes());
        chunked.extend_from_slice(chunk);
    }
    chunked.extend_from_slice(&[0u8, 0u8]);
    stream
        .write_all(&chunked)
        .map_err(|e| ExecutorError::Io(format!("send_msg write: {e}")))?;
    Ok(())
}

/// Read one whole Bolt message (de-chunked + decoded). Returns
/// `(struct_tag, fields)` for a TINY_STRUCT body.
fn read_message(stream: &mut TcpStream) -> Result<(u8, Vec<PackValue>), ExecutorError> {
    let mut body: Vec<u8> = Vec::new();
    loop {
        let mut len_buf = [0u8; 2];
        stream
            .read_exact(&mut len_buf)
            .map_err(|e| ExecutorError::Io(format!("read chunk length: {e}")))?;
        let len = u16::from_be_bytes(len_buf);
        if len == 0 {
            break;
        }
        let mut chunk = vec![0u8; len as usize];
        stream
            .read_exact(&mut chunk)
            .map_err(|e| ExecutorError::Io(format!("read chunk body: {e}")))?;
        body.extend_from_slice(&chunk);
    }
    if body.is_empty() {
        return Err(ExecutorError::Execution("empty Bolt message".into()));
    }
    let (value, _consumed) = decode(&body, 0)
        .map_err(|e| ExecutorError::Execution(format!("PackStream decode failed: {e}")))?;
    match value {
        PackValue::Struct { tag, fields } => Ok((tag, fields)),
        other => Err(ExecutorError::Execution(format!(
            "expected Bolt message struct; got {other:?}"
        ))),
    }
}

/// Read a single SUCCESS / FAILURE summary frame. Used after HELLO +
/// RUN where only one summary is expected before stream divergence.
fn read_summary(stream: &mut TcpStream) -> Result<Vec<PackValue>, ExecutorError> {
    let (tag, fields) = read_message(stream)?;
    match tag {
        MSG_SUCCESS => Ok(fields),
        MSG_FAILURE => {
            let msg = extract_failure_msg(&fields);
            Err(ExecutorError::Execution(format!("Bolt FAILURE: {msg}")))
        }
        other => Err(ExecutorError::Execution(format!(
            "unexpected Bolt summary tag: 0x{other:02x}"
        ))),
    }
}

fn expect_success(stream: &mut TcpStream) -> Result<(), ExecutorError> {
    let _ = read_summary(stream)?;
    Ok(())
}

fn extract_columns(fields: &[PackValue]) -> Option<Vec<String>> {
    if let Some(PackValue::Map(map)) = fields.first() {
        if let Some(PackValue::List(list)) = map.get("fields") {
            return Some(
                list.iter()
                    .map(|v| match v {
                        PackValue::String(s) => s.clone(),
                        other => format!("{other:?}"),
                    })
                    .collect(),
            );
        }
    }
    None
}

fn extract_failure_msg(fields: &[PackValue]) -> String {
    if let Some(PackValue::Map(map)) = fields.first() {
        if let Some(PackValue::String(m)) = map.get("message") {
            return m.clone();
        }
    }
    "<no message>".to_string()
}

fn packvalue_to_string(v: PackValue) -> String {
    match v {
        PackValue::Null => "NULL".to_string(),
        PackValue::Boolean(b) => b.to_string(),
        PackValue::Integer(i) => i.to_string(),
        PackValue::Float(f) => format!("{f:?}"),
        PackValue::String(s) => s,
        PackValue::Bytes(b) => format!("Bytes({} bytes)", b.len()),
        PackValue::List(list) => {
            let parts: Vec<String> = list.into_iter().map(packvalue_to_string).collect();
            format!("[{}]", parts.join(","))
        }
        PackValue::Map(map) => {
            let parts: Vec<String> = map
                .into_iter()
                .map(|(k, v)| format!("{k}={}", packvalue_to_string(v)))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        PackValue::Struct { tag, fields } => {
            let parts: Vec<String> = fields.into_iter().map(packvalue_to_string).collect();
            format!("Struct(0x{tag:02x}, [{}])", parts.join(","))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_columns_handles_missing_metadata() {
        assert!(extract_columns(&[]).is_none());
    }

    #[test]
    fn extract_columns_reads_from_map_fields_key() {
        let mut map = BTreeMap::new();
        map.insert(
            "fields".to_string(),
            PackValue::List(vec![
                PackValue::String("a".into()),
                PackValue::String("b".into()),
            ]),
        );
        let cols = extract_columns(&[PackValue::Map(map)]).expect("cols");
        assert_eq!(cols, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn extract_failure_msg_default() {
        assert_eq!(extract_failure_msg(&[]), "<no message>");
    }

    #[test]
    fn packvalue_stringification() {
        assert_eq!(packvalue_to_string(PackValue::Null), "NULL");
        assert_eq!(packvalue_to_string(PackValue::Integer(42)), "42");
        assert_eq!(
            packvalue_to_string(PackValue::String("hi".to_string())),
            "hi"
        );
        assert_eq!(
            packvalue_to_string(PackValue::List(vec![
                PackValue::Integer(1),
                PackValue::String("a".into()),
            ])),
            "[1,a]"
        );
    }

    #[test]
    fn connect_localhost_returns_oracle_unavailable_when_no_neo4j() {
        // CI default + dev machine: no neo4j on 7687.
        let result = Neo4jOracleExecutor::connect_localhost();
        match result {
            Err(ExecutorError::OracleUnavailable(_)) => { /* expected */ }
            Ok(_) => {
                eprintln!(
                    "a Bolt server is running on {DEFAULT_BOLT_ADDR}; \
                          this test passes (no assertion on the success path)"
                );
            }
            Err(other) => panic!("expected OracleUnavailable, got {other:?}"),
        }
    }
}
