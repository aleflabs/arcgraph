//! W14δ M5-13 — Bolt 5.0 protocol scaffold for Neo4j-driver compat.
//!
//! ArcGraph's Bolt server speaks the Bolt 5.0 protocol so that the
//! official Neo4j Python driver (and any other Bolt 5.0 capable
//! client — `neo4rs`, `neo4j-go-driver`, `neo4j-javascript-driver`)
//! can connect, authenticate, and run queries against ArcGraph using
//! the openCypher subset that the W13γ M4-83 multi-statement parser
//! admits.
//!
//! # Module layout
//!
//! - [`packstream`] — Bolt's MessagePack-like binary value codec.
//! - [`chunking`] — 0xFFFF-byte chunk framing on top of TCP.
//! - [`handshake`] — 4-byte magic preamble + Bolt-version negotiation.
//! - [`message`] — Bolt 5.0 [`message::ClientMessage`] /
//!   [`message::ServerMessage`] taxonomies + encode/decode.
//! - [`state`] — connection state machine (Initial → Ready →
//!   Streaming, plus Failed / Closed terminals).
//! - [`error`] — Bolt-side error taxonomy + Neo4j-status-code mapping.
//! - [`value`] — bridge between [`arcgraph_query::executor::Value`]
//!   and [`packstream::PackValue`].
//! - [`handler`] — `BoltQueryHandler` adapter trait + a
//!   [`handler::StubBoltHandler`] for tests.
//! - [`server`] — TCP listener + per-connection task loop.
//!
//! # v1.0-α scope
//!
//! Per the spawn prompt's "Hard boundaries" section:
//!
//! - Bolt 5.0 ONLY (not 4.4 / 5.1 / 5.2 / …).
//! - No routing protocol (cluster mode is v1.1+).
//! - TLS via the W13ε resolver is OPTIONAL but ENFORCED for
//!   non-loopback binds (W15δ Bolt-TLS-wire — see
//!   [`server::BoltServerConfig::with_tls`] /
//!   [`server::BoltServerConfig::validate`]). Plain TCP is admitted
//!   only for loopback dev / test configurations.
//! - Embedded/dev authentication accepts any non-empty principal for
//!   `basic`; `none` may complete HELLO but the storage handler refuses a
//!   principal-less content `RUN`. Production can require OAuth bearer auth.
//!
//! # ADR provenance
//!
//! - **design-v2 §16.3** — "Bolt protocol (openCypher driver
//!   compatibility)" listed as an M4-5 deliverable; this slice
//!   delivers the v1.0-α scaffold per ADR-038 amendment-03 §M5↔M4.
//! - **ADR-094 (W24-DRIVERS-α)** — ratifies the Bolt 5.0-only
//!   commitment, Bolt 4.4 deferral to v1.1+, and GQL 2024
//!   partial-conformance commitment. The version stability table in
//!   ADR-094 D-1 is the binding-through-v1.2-GA reference for
//!   downstream driver authors. Companion artifacts:
//!   `docs/conformance/gql-2024-conformance-matrix.md` and
//!   `crates/arcgraph-cli/tests/driver_compat_bolt_v5.rs` subprocess
//!   wire-conformance pin.
//! - **dependency and artifact policy** — the 10-MCP-tool cap is preserved (Bolt is a
//!   transport, not an MCP tool — it does not count against the
//!   cap; re-affirmed by ADR-094 D-5).
//! - **TLS reuse** — the same hot-reload resolver instance powers both
//!   the HTTP/TLS and Bolt transports.

pub mod auth;
pub mod chunking;
pub mod error;
pub mod handler;
pub mod handshake;
pub mod message;
pub mod packstream;
pub mod server;
pub mod state;
pub mod value;

// ─────────────────────────────────────────────────────────────────────
// Public re-exports — top-level surface for downstream consumers.
// ─────────────────────────────────────────────────────────────────────

pub use auth::{BoltOAuthValidator, tenant_id_for_suffix, tenant_id_from_claims};
pub use chunking::{
    MAX_BOLT_MESSAGE_BYTES, MAX_CHUNK_LEN, read_chunked_message, write_chunked_message,
};
pub use error::BoltError;
pub use handler::{BoltQueryHandler, BoltSessionAuth, RunOutcome, StubBoltHandler, StubFault};
pub use handshake::{
    BoltVersion, MAGIC_PREAMBLE, SERVER_ACCEPT_V5_0, SERVER_REJECT, perform_handshake,
};
pub use message::{
    ClientMessage, ServerMessage, TAG_DISCARD, TAG_FAILURE, TAG_GOODBYE, TAG_HELLO, TAG_IGNORED,
    TAG_PULL, TAG_RECORD, TAG_RESET, TAG_RUN, TAG_SUCCESS, decode_client, encode_client,
    encode_server,
};
pub use packstream::{
    MAX_PACKSTREAM_DEPTH, PackError, PackValue, TAG_NODE, TAG_PATH, TAG_RELATIONSHIP,
    TAG_UNBOUND_RELATIONSHIP, decode, encode,
};
pub use server::{
    BoltServeStats, BoltServerConfig, ConnTaskOutcome, handle_pair, serve_bolt_inner,
    serve_bolt_inner_with_tls, serve_bolt_listener,
};
pub use state::{ConnFsm, ConnState, HandlerOutcome, Transition};
pub use value::{
    ParamError, exec_to_pack, exec_to_pack_with_tenant, pack_params_to_exec, pack_to_exec,
};
