//! W14δ M5-13 — Bolt 5.0 message types + encode / decode.
//!
//! Each Bolt message is a PackStream Struct: a 1-byte marker
//! `0xB<arity>`, a 1-byte tag identifying the message, and `arity`
//! positional fields. The chunk-framing layer ([`super::chunking`])
//! handles the split into 0xFFFF-byte chunks.
//!
//! # v1.0-α message catalog
//!
//! | Tag    | Direction | Name      | Fields                                         |
//! |--------|-----------|-----------|------------------------------------------------|
//! | `0x01` | C → S     | HELLO     | `[extra: Map]`                                 |
//! | `0x02` | C → S     | GOODBYE   | none                                           |
//! | `0x0F` | C → S     | RESET     | none                                           |
//! | `0x10` | C → S     | RUN       | `[query: String, params: Map, extra: Map]`     |
//! | `0x11` | C → S     | BEGIN     | `[extra: Map]`                                 |
//! | `0x12` | C → S     | COMMIT    | none                                           |
//! | `0x13` | C → S     | ROLLBACK  | none                                           |
//! | `0x2F` | C → S     | DISCARD   | `[extra: Map]`                                 |
//! | `0x3F` | C → S     | PULL      | `[extra: Map]`                                 |
//! | `0x70` | S → C     | SUCCESS   | `[metadata: Map]`                              |
//! | `0x71` | S → C     | RECORD    | `[fields: List]`                               |
//! | `0x7E` | S → C     | IGNORED   | none                                           |
//! | `0x7F` | S → C     | FAILURE   | `[metadata: Map { code, message }]`            |
//!
//! BEGIN / COMMIT / ROLLBACK (explicit transactions) land at ADR-197
//! (the langchain-neo4j managed-transaction drop-in). Still
//! out-of-scope: TELEMETRY (5.4+), ROUTE (cluster mode), LOGON /
//! LOGOFF (5.1+).

use std::collections::BTreeMap;

use super::error::BoltError;
use super::packstream::{self, PackValue};

/// Tag byte for HELLO (`0x01`).
pub const TAG_HELLO: u8 = 0x01;
/// Tag byte for GOODBYE (`0x02`).
pub const TAG_GOODBYE: u8 = 0x02;
/// Tag byte for RESET (`0x0F`).
pub const TAG_RESET: u8 = 0x0F;
/// Tag byte for RUN (`0x10`).
pub const TAG_RUN: u8 = 0x10;
/// Tag byte for BEGIN (`0x11`) — ADR-197 explicit-transaction open.
pub const TAG_BEGIN: u8 = 0x11;
/// Tag byte for COMMIT (`0x12`) — ADR-197 explicit-transaction commit.
pub const TAG_COMMIT: u8 = 0x12;
/// Tag byte for ROLLBACK (`0x13`) — ADR-197 explicit-transaction abort.
pub const TAG_ROLLBACK: u8 = 0x13;
/// Tag byte for DISCARD (`0x2F`).
pub const TAG_DISCARD: u8 = 0x2F;
/// Tag byte for PULL (`0x3F`).
pub const TAG_PULL: u8 = 0x3F;
/// Tag byte for SUCCESS (`0x70`).
pub const TAG_SUCCESS: u8 = 0x70;
/// Tag byte for RECORD (`0x71`).
pub const TAG_RECORD: u8 = 0x71;
/// Tag byte for IGNORED (`0x7E`).
pub const TAG_IGNORED: u8 = 0x7E;
/// Tag byte for FAILURE (`0x7F`).
pub const TAG_FAILURE: u8 = 0x7F;

/// Inbound (client → server) Bolt message variants the v1.0-α server
/// admits.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientMessage {
    /// `HELLO` — opening handshake-tail message carrying the auth
    /// scheme + principal/credentials + user_agent. The peer expects
    /// either SUCCESS (with a server-side connection metadata map) or
    /// FAILURE.
    Hello {
        /// User-agent string the client identifies itself with.
        user_agent: Option<String>,
        /// Auth scheme — at v1.0-α we accept "none" (no auth) and
        /// "basic" (any non-empty principal).
        scheme: Option<String>,
        /// Principal / username. v1.0-α: required + non-empty for
        /// "basic" auth, ignored for "none".
        principal: Option<String>,
        /// Credentials — at v1.0-α we don't validate the contents, but
        /// "basic" auth requires the field to be present + non-empty.
        credentials: Option<String>,
        /// Routing context — Bolt 5.0 routing parameters (forward-pin
        /// for v1.1 cluster mode; v1.0-α ignores).
        routing: Option<BTreeMap<String, PackValue>>,
        /// Catch-all for additional `extra` map entries the client
        /// includes (e.g., `bolt_agent`, `notifications_minimum_severity`).
        /// v1.0-α accepts and ignores; future Bolt slices may inspect.
        extras: BTreeMap<String, PackValue>,
    },
    /// `GOODBYE` — peer asked to close the connection cleanly.
    Goodbye,
    /// `RESET` — peer asked to reset the session: cancel any in-flight
    /// query + return to the READY state. Server replies with SUCCESS.
    Reset,
    /// `RUN` — peer submitted a Cypher statement.
    Run {
        /// Cypher source string. The server pipes this to the ArcQL
        /// parser ([`arcgraph_query::parse_multi`]); statements outside
        /// the supported subset surface as FAILURE.
        query: String,
        /// Statement parameters (Cypher `$paramName`). Restricted to
        /// the JSON-translatable subset (no Bytes / Struct).
        parameters: BTreeMap<String, PackValue>,
        /// `extra` map. v1.0-α uses this for `tx_timeout` /
        /// `tx_metadata` / `db` / `imp_user`. Most fields are
        /// forward-pinned (read but ignored); the `db` field IS
        /// honored to scope tenant context.
        extra: BTreeMap<String, PackValue>,
    },
    /// `DISCARD` — drop the next `n` records from the active result
    /// stream without materializing them to the peer.
    Discard {
        /// Number of records to drop. `-1` = drop all remaining.
        n: i64,
        /// Optional run-id (Bolt 5.0 supports multi-statement
        /// streams; v1.0-α accepts but ignores qid since RUN is
        /// auto-commit).
        qid: Option<i64>,
    },
    /// `PULL` — request the next `n` records from the active result
    /// stream.
    Pull {
        /// Number of records to pull. `-1` = pull all remaining.
        n: i64,
        /// Optional run-id (see [`ClientMessage::Discard`]).
        qid: Option<i64>,
    },
    /// `BEGIN` — ADR-197: open an explicit transaction. The peer's
    /// managed-transaction API (`driver.execute_query` /
    /// `session.execute_read|write`) sends this before the first RUN.
    /// Subsequent RUNs stage into the one held transaction until
    /// COMMIT / ROLLBACK.
    Begin {
        /// The `extra` map. ADR-197 honors `mode` (`"r"`/`"w"`) +
        /// `db`; `tx_timeout`, `tx_metadata`, `bookmarks` are accepted
        /// and (at v1.0-α) not acted on — but are NOT rejected (the
        /// neo4j driver always sends `bookmarks: []`).
        extra: BTreeMap<String, PackValue>,
    },
    /// `COMMIT` — ADR-197: commit the open explicit transaction. The
    /// server replies SUCCESS `{bookmark}`.
    Commit,
    /// `ROLLBACK` — ADR-197: abort the open explicit transaction. All
    /// writes staged since BEGIN are discarded.
    Rollback,
}

/// Outbound (server → client) Bolt message variants the v1.0-α
/// server emits.
#[derive(Debug, Clone, PartialEq)]
pub enum ServerMessage {
    /// `SUCCESS` — the prior client message succeeded; metadata
    /// carries the per-message tail (e.g., RUN reply with field
    /// names; PULL reply with summary stats).
    Success(BTreeMap<String, PackValue>),
    /// `RECORD` — one row of result data; the field list is
    /// positionally aligned with the SUCCESS metadata's `fields`
    /// list from the corresponding RUN reply.
    Record(Vec<PackValue>),
    /// `IGNORED` — the prior client message was not processed
    /// because the connection was in a FAILED state. Used for
    /// drain semantics: client can keep sending RUN / PULL until it
    /// emits RESET; server keeps replying IGNORED.
    Ignored,
    /// `FAILURE` — the prior client message failed; metadata
    /// carries `code` (Neo4j status) + `message` (human-readable).
    Failure(BTreeMap<String, PackValue>),
}

impl ServerMessage {
    /// Construct a SUCCESS for HELLO acceptance. v1.0-α metadata:
    /// - `connection_id`: server-issued opaque connection id.
    /// - `server`: server name + version.
    /// - `hints`: empty map (forward-pin).
    pub fn hello_success(connection_id: impl Into<String>) -> ServerMessage {
        let mut meta = BTreeMap::new();
        meta.insert(
            "connection_id".into(),
            PackValue::String(connection_id.into()),
        );
        meta.insert("server".into(), PackValue::String("ArcGraph/0.0.0".into()));
        meta.insert("hints".into(), PackValue::Map(BTreeMap::new()));
        ServerMessage::Success(meta)
    }

    /// Construct a SUCCESS for RUN acceptance. v1.0-α metadata:
    /// - `fields`: list of column names from the projection.
    /// - `qid`: optional run-id (omitted at v1.0-α — auto-commit).
    pub fn run_success(fields: Vec<String>) -> ServerMessage {
        let mut meta = BTreeMap::new();
        meta.insert(
            "fields".into(),
            PackValue::List(fields.into_iter().map(PackValue::String).collect()),
        );
        ServerMessage::Success(meta)
    }

    /// Construct a SUCCESS for PULL completion. v1.0-α metadata:
    /// - `has_more`: bool — `true` if PULL emitted exactly `n` rows
    ///   and the stream has more remaining.
    /// - `type`: `"r"` (read) at v1.0-α (no DDL / write paths yet).
    pub fn pull_success(has_more: bool) -> ServerMessage {
        let mut meta = BTreeMap::new();
        meta.insert("has_more".into(), PackValue::Boolean(has_more));
        if !has_more {
            meta.insert("type".into(), PackValue::String("r".into()));
        }
        ServerMessage::Success(meta)
    }

    /// Construct a SUCCESS for RESET completion. The metadata is
    /// empty per the spec (RESET is a control message; no payload).
    pub fn reset_success() -> ServerMessage {
        ServerMessage::Success(BTreeMap::new())
    }

    /// ADR-197 — SUCCESS for BEGIN. Empty metadata at v1.0-α (the
    /// neo4j driver does not require a `tx_id` in the BEGIN reply).
    pub fn begin_success() -> ServerMessage {
        ServerMessage::Success(BTreeMap::new())
    }

    /// ADR-197 — SUCCESS for COMMIT. Carries `bookmark` — an opaque
    /// monotonic token the driver records for causal consistency. At
    /// v1.0-α the token is `arcgraph:<commit-lsn>` (monotonic, opaque
    /// to the driver); cluster-mode causal chaining is forward-debt.
    pub fn commit_success(bookmark: impl Into<String>) -> ServerMessage {
        let mut meta = BTreeMap::new();
        meta.insert("bookmark".into(), PackValue::String(bookmark.into()));
        ServerMessage::Success(meta)
    }

    /// ADR-197 — SUCCESS for ROLLBACK. Empty metadata per the spec.
    pub fn rollback_success() -> ServerMessage {
        ServerMessage::Success(BTreeMap::new())
    }

    /// Construct a FAILURE message from a `code` + `message` pair.
    /// Maps directly to the Bolt FAILURE metadata shape.
    pub fn failure(code: impl Into<String>, message: impl Into<String>) -> ServerMessage {
        let mut meta = BTreeMap::new();
        meta.insert("code".into(), PackValue::String(code.into()));
        meta.insert("message".into(), PackValue::String(message.into()));
        ServerMessage::Failure(meta)
    }

    /// Convenience: build a FAILURE from a [`BoltError`] using its
    /// canonical Neo4j code mapping.
    pub fn failure_from_error(err: &BoltError) -> ServerMessage {
        ServerMessage::failure(err.neo4j_code(), err.message())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Encode / Decode
// ─────────────────────────────────────────────────────────────────────

/// Encode a [`ServerMessage`] as a PackStream Struct, writing the
/// PackStream-encoded bytes into `out`. Caller is responsible for
/// the chunk-framing layer ([`super::chunking::write_chunked_message`]).
pub fn encode_server(out: &mut Vec<u8>, msg: &ServerMessage) -> Result<(), BoltError> {
    let pack = match msg {
        ServerMessage::Success(meta) => PackValue::Struct {
            tag: TAG_SUCCESS,
            fields: vec![map_to_pack(meta)],
        },
        ServerMessage::Record(fields) => PackValue::Struct {
            tag: TAG_RECORD,
            fields: vec![PackValue::List(fields.clone())],
        },
        ServerMessage::Ignored => PackValue::Struct {
            tag: TAG_IGNORED,
            fields: vec![],
        },
        ServerMessage::Failure(meta) => PackValue::Struct {
            tag: TAG_FAILURE,
            fields: vec![map_to_pack(meta)],
        },
    };
    packstream::encode(out, &pack).map_err(BoltError::from)?;
    Ok(())
}

/// Decode a [`ClientMessage`] from a chunk-dechunked PackStream
/// payload. Returns [`BoltError::Framing`] on tag-not-recognized
/// (so the caller can either FAILURE + RESET or close).
pub fn decode_client(payload: &[u8]) -> Result<ClientMessage, BoltError> {
    let (value, consumed) = packstream::decode(payload, 0).map_err(BoltError::from)?;
    if consumed != payload.len() {
        return Err(BoltError::Framing(format!(
            "trailing bytes after PackStream value ({} of {} consumed)",
            consumed,
            payload.len()
        )));
    }
    let (tag, fields) = match value {
        PackValue::Struct { tag, fields } => (tag, fields),
        other => {
            return Err(BoltError::Framing(format!(
                "expected struct at message root, got {other:?}"
            )));
        }
    };
    match tag {
        TAG_HELLO => decode_hello(fields),
        TAG_GOODBYE => Ok(ClientMessage::Goodbye),
        TAG_RESET => Ok(ClientMessage::Reset),
        TAG_RUN => decode_run(fields),
        TAG_BEGIN => decode_begin(fields),
        TAG_COMMIT => decode_zero_field(fields, "COMMIT", ClientMessage::Commit),
        TAG_ROLLBACK => decode_zero_field(fields, "ROLLBACK", ClientMessage::Rollback),
        TAG_DISCARD => decode_pull_or_discard(fields, /*is_discard=*/ true),
        TAG_PULL => decode_pull_or_discard(fields, /*is_discard=*/ false),
        other => Err(BoltError::Framing(format!(
            "unsupported message tag 0x{other:02X}"
        ))),
    }
}

/// ADR-197 — decode BEGIN `[extra: Map]`. The `extra` map carries
/// `mode`/`db`/`tx_timeout`/`tx_metadata`/`bookmarks`; we keep the
/// whole map and let the handler pick out what it honors.
fn decode_begin(fields: Vec<PackValue>) -> Result<ClientMessage, BoltError> {
    if fields.len() != 1 {
        return Err(BoltError::Framing(format!(
            "BEGIN expects 1 field (extra map), got {}",
            fields.len()
        )));
    }
    let extra = match fields.into_iter().next().expect("len-checked") {
        PackValue::Map(m) => m,
        PackValue::Null => BTreeMap::new(),
        other => {
            return Err(BoltError::Framing(format!(
                "BEGIN extra must be a Map, got {other:?}"
            )));
        }
    };
    Ok(ClientMessage::Begin { extra })
}

/// ADR-197 — decode a zero-field control message (COMMIT / ROLLBACK).
fn decode_zero_field(
    fields: Vec<PackValue>,
    name: &str,
    msg: ClientMessage,
) -> Result<ClientMessage, BoltError> {
    if !fields.is_empty() {
        return Err(BoltError::Framing(format!(
            "{name} expects 0 fields, got {}",
            fields.len()
        )));
    }
    Ok(msg)
}

fn decode_hello(fields: Vec<PackValue>) -> Result<ClientMessage, BoltError> {
    if fields.len() != 1 {
        return Err(BoltError::Framing(format!(
            "HELLO expects 1 field (extra map), got {}",
            fields.len()
        )));
    }
    let mut extras = match fields.into_iter().next().expect("len-checked") {
        PackValue::Map(m) => m,
        other => {
            return Err(BoltError::Framing(format!(
                "HELLO extra must be a Map, got {other:?}"
            )));
        }
    };
    let user_agent = take_string(&mut extras, "user_agent");
    let scheme = take_string(&mut extras, "scheme");
    let principal = take_string(&mut extras, "principal");
    let credentials = take_string(&mut extras, "credentials");
    let routing = match extras.remove("routing") {
        Some(PackValue::Map(m)) => Some(m),
        Some(PackValue::Null) => None,
        Some(other) => {
            return Err(BoltError::Framing(format!(
                "HELLO routing must be Map or Null, got {other:?}"
            )));
        }
        None => None,
    };
    Ok(ClientMessage::Hello {
        user_agent,
        scheme,
        principal,
        credentials,
        routing,
        extras,
    })
}

fn decode_run(fields: Vec<PackValue>) -> Result<ClientMessage, BoltError> {
    if fields.len() != 3 {
        return Err(BoltError::Framing(format!(
            "RUN expects 3 fields (query, params, extra), got {}",
            fields.len()
        )));
    }
    let mut iter = fields.into_iter();
    let query = match iter.next().expect("len-checked") {
        PackValue::String(s) => s,
        other => {
            return Err(BoltError::Framing(format!(
                "RUN query must be String, got {other:?}"
            )));
        }
    };
    let parameters = match iter.next().expect("len-checked") {
        PackValue::Map(m) => m,
        PackValue::Null => BTreeMap::new(),
        other => {
            return Err(BoltError::Framing(format!(
                "RUN parameters must be Map, got {other:?}"
            )));
        }
    };
    let extra = match iter.next().expect("len-checked") {
        PackValue::Map(m) => m,
        PackValue::Null => BTreeMap::new(),
        other => {
            return Err(BoltError::Framing(format!(
                "RUN extra must be Map, got {other:?}"
            )));
        }
    };
    Ok(ClientMessage::Run {
        query,
        parameters,
        extra,
    })
}

fn decode_pull_or_discard(
    fields: Vec<PackValue>,
    is_discard: bool,
) -> Result<ClientMessage, BoltError> {
    if fields.len() != 1 {
        return Err(BoltError::Framing(format!(
            "{} expects 1 field (extra map), got {}",
            if is_discard { "DISCARD" } else { "PULL" },
            fields.len()
        )));
    }
    let mut extra = match fields.into_iter().next().expect("len-checked") {
        PackValue::Map(m) => m,
        other => {
            return Err(BoltError::Framing(format!(
                "{} extra must be a Map, got {other:?}",
                if is_discard { "DISCARD" } else { "PULL" }
            )));
        }
    };
    let n = match extra.remove("n") {
        Some(PackValue::Integer(i)) => i,
        Some(other) => {
            return Err(BoltError::Framing(format!(
                "PULL/DISCARD `n` must be Integer, got {other:?}"
            )));
        }
        None => -1, // default: pull/discard all
    };
    let qid = match extra.remove("qid") {
        Some(PackValue::Integer(i)) => Some(i),
        Some(PackValue::Null) => None,
        Some(other) => {
            return Err(BoltError::Framing(format!(
                "PULL/DISCARD `qid` must be Integer or Null, got {other:?}"
            )));
        }
        None => None,
    };
    Ok(if is_discard {
        ClientMessage::Discard { n, qid }
    } else {
        ClientMessage::Pull { n, qid }
    })
}

fn take_string(map: &mut BTreeMap<String, PackValue>, key: &str) -> Option<String> {
    match map.remove(key) {
        Some(PackValue::String(s)) => Some(s),
        Some(PackValue::Null) => None,
        Some(other) => {
            // The HELLO/RUN keys we care about should be strings;
            // anything else gets dropped silently. The framing layer
            // catches truly malformed inputs before this point.
            tracing::debug!(target: "arcgraph_mcp::bolt::message",
                key, "expected string, got {other:?}; discarding");
            None
        }
        None => None,
    }
}

fn map_to_pack(m: &BTreeMap<String, PackValue>) -> PackValue {
    PackValue::Map(m.clone())
}

/// Convenience for tests + integration drivers: encode a
/// [`ClientMessage`] as a PackStream payload (caller chunks it).
pub fn encode_client(out: &mut Vec<u8>, msg: &ClientMessage) -> Result<(), BoltError> {
    let pack = match msg {
        ClientMessage::Hello {
            user_agent,
            scheme,
            principal,
            credentials,
            routing,
            extras,
        } => {
            let mut extra = extras.clone();
            if let Some(ua) = user_agent {
                extra.insert("user_agent".into(), PackValue::String(ua.clone()));
            }
            if let Some(s) = scheme {
                extra.insert("scheme".into(), PackValue::String(s.clone()));
            }
            if let Some(p) = principal {
                extra.insert("principal".into(), PackValue::String(p.clone()));
            }
            if let Some(c) = credentials {
                extra.insert("credentials".into(), PackValue::String(c.clone()));
            }
            if let Some(r) = routing {
                extra.insert("routing".into(), PackValue::Map(r.clone()));
            }
            PackValue::Struct {
                tag: TAG_HELLO,
                fields: vec![PackValue::Map(extra)],
            }
        }
        ClientMessage::Goodbye => PackValue::Struct {
            tag: TAG_GOODBYE,
            fields: vec![],
        },
        ClientMessage::Reset => PackValue::Struct {
            tag: TAG_RESET,
            fields: vec![],
        },
        ClientMessage::Run {
            query,
            parameters,
            extra,
        } => PackValue::Struct {
            tag: TAG_RUN,
            fields: vec![
                PackValue::String(query.clone()),
                PackValue::Map(parameters.clone()),
                PackValue::Map(extra.clone()),
            ],
        },
        ClientMessage::Pull { n, qid } => PackValue::Struct {
            tag: TAG_PULL,
            fields: vec![pull_or_discard_extra(*n, *qid)],
        },
        ClientMessage::Discard { n, qid } => PackValue::Struct {
            tag: TAG_DISCARD,
            fields: vec![pull_or_discard_extra(*n, *qid)],
        },
        ClientMessage::Begin { extra } => PackValue::Struct {
            tag: TAG_BEGIN,
            fields: vec![PackValue::Map(extra.clone())],
        },
        ClientMessage::Commit => PackValue::Struct {
            tag: TAG_COMMIT,
            fields: vec![],
        },
        ClientMessage::Rollback => PackValue::Struct {
            tag: TAG_ROLLBACK,
            fields: vec![],
        },
    };
    packstream::encode(out, &pack).map_err(BoltError::from)?;
    Ok(())
}

fn pull_or_discard_extra(n: i64, qid: Option<i64>) -> PackValue {
    let mut m = BTreeMap::new();
    m.insert("n".into(), PackValue::Integer(n));
    if let Some(q) = qid {
        m.insert("qid".into(), PackValue::Integer(q));
    }
    PackValue::Map(m)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_dec_client(msg: &ClientMessage) -> ClientMessage {
        let mut buf = Vec::new();
        encode_client(&mut buf, msg).expect("encode ok");
        decode_client(&buf).expect("decode ok")
    }

    fn enc_server(msg: &ServerMessage) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_server(&mut buf, msg).expect("encode ok");
        buf
    }

    #[test]
    fn hello_roundtrips() {
        let h = ClientMessage::Hello {
            user_agent: Some("test/1.0".into()),
            scheme: Some("basic".into()),
            principal: Some("alice".into()),
            credentials: Some("****".into()),
            routing: None,
            extras: BTreeMap::new(),
        };
        match enc_dec_client(&h) {
            ClientMessage::Hello {
                user_agent,
                scheme,
                principal,
                ..
            } => {
                assert_eq!(user_agent.as_deref(), Some("test/1.0"));
                assert_eq!(scheme.as_deref(), Some("basic"));
                assert_eq!(principal.as_deref(), Some("alice"));
            }
            other => panic!("expected Hello, got {other:?}"),
        }
    }

    #[test]
    fn run_roundtrips_with_parameters() {
        let mut params = BTreeMap::new();
        params.insert("name".into(), PackValue::String("Alice".into()));
        let r = ClientMessage::Run {
            query: "MATCH (n:Person {name: $name}) RETURN n".into(),
            parameters: params,
            extra: BTreeMap::new(),
        };
        let got = enc_dec_client(&r);
        assert_eq!(got, r);
    }

    #[test]
    fn pull_with_default_n_decodes_negative_one() {
        // Spec: PULL with no `n` field defaults to -1 (pull all).
        let mut buf = Vec::new();
        encode_client(&mut buf, &ClientMessage::Pull { n: -1, qid: None }).unwrap();
        let got = decode_client(&buf).unwrap();
        assert_eq!(got, ClientMessage::Pull { n: -1, qid: None });
    }

    #[test]
    fn pull_with_n_and_qid_roundtrips() {
        let p = ClientMessage::Pull {
            n: 100,
            qid: Some(7),
        };
        assert_eq!(enc_dec_client(&p), p);
    }

    #[test]
    fn goodbye_and_reset_roundtrip() {
        assert_eq!(
            enc_dec_client(&ClientMessage::Goodbye),
            ClientMessage::Goodbye
        );
        assert_eq!(enc_dec_client(&ClientMessage::Reset), ClientMessage::Reset);
    }

    #[test]
    fn server_record_encoded_as_struct_with_one_list_field() {
        let rec = ServerMessage::Record(vec![PackValue::Integer(1), PackValue::Integer(2)]);
        let buf = enc_server(&rec);
        // First byte is TINY_STRUCT(1) | tag-marker. With arity 1:
        //   0xB1, 0x71, then a list of 2 integers.
        assert_eq!(buf[0], 0xB1);
        assert_eq!(buf[1], TAG_RECORD);
    }

    #[test]
    fn server_failure_carries_neo4j_code_and_message() {
        let err = BoltError::Syntax("expected RETURN at line 1".into());
        let msg = ServerMessage::failure_from_error(&err);
        let ServerMessage::Failure(meta) = &msg else {
            panic!("expected Failure");
        };
        assert_eq!(
            meta.get("code"),
            Some(&PackValue::String(
                "Neo.ClientError.Statement.SyntaxError".into()
            ))
        );
        match meta.get("message") {
            Some(PackValue::String(s)) => assert!(s.contains("expected RETURN")),
            _ => panic!("missing message"),
        }
    }

    #[test]
    fn server_hello_success_carries_connection_id_and_server() {
        let s = ServerMessage::hello_success("conn-123");
        let ServerMessage::Success(meta) = &s else {
            panic!("expected Success");
        };
        assert_eq!(
            meta.get("connection_id"),
            Some(&PackValue::String("conn-123".into()))
        );
        assert!(matches!(meta.get("server"), Some(PackValue::String(_))));
    }

    #[test]
    fn run_success_lists_field_names() {
        let s = ServerMessage::run_success(vec!["n".into(), "m".into()]);
        let ServerMessage::Success(meta) = &s else {
            panic!("expected Success");
        };
        let fields = meta
            .get("fields")
            .and_then(|v| {
                if let PackValue::List(l) = v {
                    Some(l)
                } else {
                    None
                }
            })
            .expect("fields present");
        assert_eq!(fields.len(), 2);
    }

    #[test]
    fn pull_success_carries_has_more_flag() {
        let s = ServerMessage::pull_success(true);
        let ServerMessage::Success(meta) = &s else {
            panic!("expected Success");
        };
        assert_eq!(meta.get("has_more"), Some(&PackValue::Boolean(true)));
        // has_more=true → no `type` field (spec: only the terminal
        // SUCCESS carries `type`).
        assert!(meta.get("type").is_none());
        let s2 = ServerMessage::pull_success(false);
        let ServerMessage::Success(meta2) = &s2 else {
            panic!();
        };
        assert_eq!(meta2.get("has_more"), Some(&PackValue::Boolean(false)));
        assert!(meta2.get("type").is_some(), "terminal SUCCESS carries type");
    }

    #[test]
    fn decode_rejects_unknown_message_tag() {
        // Hand-craft a TINY_STRUCT(0)+unknown_tag.
        let bytes = [0xB0, 0x55];
        let err = decode_client(&bytes).unwrap_err();
        assert!(matches!(err, BoltError::Framing(_)));
    }

    #[test]
    fn decode_rejects_run_with_wrong_arity() {
        // A RUN with only 2 fields instead of 3.
        let bytes = [0xB2, TAG_RUN, 0x80, 0xA0]; // empty string + empty map
        let err = decode_client(&bytes).unwrap_err();
        assert!(matches!(err, BoltError::Framing(_)));
    }
}
