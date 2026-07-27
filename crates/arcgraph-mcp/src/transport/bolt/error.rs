//! W14δ M5-13 — Bolt-protocol error taxonomy + Neo4j FAILURE-code mapping.
//!
//! Bolt FAILURE messages carry a structured `code` field formatted as
//! `Neo.<class>.<category>.<title>` per the Neo4j status-codes
//! catalog. Drivers (including the official Python `neo4j` package)
//! pattern-match on this code to decide whether the error is
//! retryable, whether it indicates a transient issue, or whether it
//! reflects a client bug.
//!
//! v1.0-α surfaces a minimal-but-faithful subset:
//!
//! | ArcGraph fault                       | Neo4j code                                       |
//! |--------------------------------------|--------------------------------------------------|
//! | ParseError / ArcQLError              | `Neo.ClientError.Statement.SyntaxError`          |
//! | ResourceExhausted                    | `Neo.TransientError.General.OutOfMemoryError`    |
//! | MissingParameter (#797)              | `Neo.ClientError.Statement.ParameterMissing`     |
//! | Invalid parameter shape (#797)       | `Neo.ClientError.Statement.TypeError`            |
//! | NotImplemented                       | `Neo.ClientError.Statement.NotImplemented`       |
//! | ExecutionError::Cancelled            | `Neo.ClientError.Transaction.Terminated`         |
//! | MVCC write-write conflict (#907)     | `Neo.TransientError.Transaction.DeadlockDetected`|
//! | TenantUnknown                        | `Neo.ClientError.Database.DatabaseNotFound`      |
//! | IndexUnavailable                     | `Neo.ClientError.Schema.IndexNotFound`           |
//! | Authentication (empty principal)     | `Neo.ClientError.Security.Unauthorized`          |
//! | Internal eval / I/O fault            | `Neo.DatabaseError.General.UnknownError`         |
//! | Protocol violation (e.g., RUN before HELLO) | `Neo.ClientError.Request.Invalid`         |
//!
//! The mapping table is the public contract surface. Future Bolt
//! slices (M5-14+) MUST keep these mappings stable so existing
//! drivers don't regress.
//!
//! **#907 — retriable transient class.** A write-write MVCC conflict is a
//! NORMAL optimistic-concurrency outcome, not a fault: it maps to a
//! `Neo.TransientError.Transaction.*` code so a Neo4j driver's managed
//! transaction (`session.execute_write` / `execute_read`) AUTO-RETRIES it
//! (the conflict becomes invisible to the app, exactly as on Neo4j). The
//! retriable signal is the `TransientError` CLASS: the official drivers
//! retry every `Neo.TransientError.*` EXCEPT
//! `Neo.TransientError.Transaction.Terminated` and
//! `…Transaction.LockClientStopped` (which signal an explicitly-killed
//! transaction, not a transient conflict). We therefore use
//! `Neo.TransientError.Transaction.DeadlockDetected` — the
//! ecosystem-canonical retriable write-conflict code that all drivers
//! auto-retry — and deliberately NOT `Terminated` / `LockClientStopped`,
//! even though #907's prose names the latter two: those are precisely the
//! two transient codes drivers do NOT retry, so they would not fix
//! `execute_write`. (ArcGraph commits optimistically rather than via lock
//! waits, so there is no literal lock-cycle "deadlock"; `DeadlockDetected`
//! is used as Neo4j's standard retriable *write-conflict* signal — the
//! purely-OCC analog `…Transaction.Outdated` is also retriable but far
//! less recognized by the driver/tooling ecosystem.)

use std::net::SocketAddr;

use thiserror::Error;

use super::packstream::PackError;

/// Codec-local error type for the Bolt transport surface. Distinct
/// from [`crate::error::MCPError`] so the Bolt server can pattern-
/// match on Bolt-specific failures (handshake rejection, framing
/// fault, protocol violation) without forcing them through the
/// MCP-Tier-1 taxonomy. The public boundary translates via
/// [`Self::neo4j_code`] / [`Self::message`] when emitting a Bolt
/// FAILURE message.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BoltError {
    /// Handshake failed (bad magic, no shared protocol version, peer
    /// closed mid-handshake). Caller closes the TCP connection.
    #[error("handshake rejected: {0}")]
    HandshakeRejected(String),
    /// Peer sent a chunk header / message body whose structure the
    /// codec rejected at the framing layer (e.g., chunk length 0 in
    /// the middle of a message, length overflow). The server replies
    /// with a Bolt FAILURE message and transitions the FSM to Failed;
    /// the client must send RESET to recover (per Bolt §"Server
    /// Lifecycle" IGNORED-after-FAILURE semantics, as enforced by
    /// [`super::state::ConnFsm::admit`]). The chunking layer's
    /// 0x0000-terminator framing means a single corrupted message
    /// does NOT poison subsequent messages — the connection stays
    /// open and the next 0x0000-terminated payload is re-decoded
    /// from scratch.
    #[error("framing fault: {0}")]
    Framing(String),
    /// Underlying TCP / I/O fault.
    #[error("io: {0}")]
    Io(String),
    /// PackStream codec rejected a value (unknown marker, truncated
    /// body, invalid UTF-8, unsupported struct tag, non-string map
    /// key).
    #[error("packstream: {0}")]
    Pack(#[from] PackError),
    /// Peer sent a Bolt message whose total dechunked size (sum of
    /// all chunk bodies before the `0x0000` terminator) exceeds
    /// [`super::chunking::MAX_BOLT_MESSAGE_BYTES`]. Per-chunk length is
    /// already capped at `0xFFFF` by the wire format, but without a
    /// total-size cap an attacker can stream N×64KiB chunks unbounded
    /// → OOM. Surfaces from
    /// [`super::chunking::read_chunked_message`]. Maps to
    /// `Neo.ClientError.Request.InvalidFormat` (framing-class fault).
    #[error("bolt message too large: {bytes} > {max}")]
    MessageTooLarge {
        /// Accumulated dechunked body length at the moment the cap
        /// was exceeded (i.e., the running total after appending the
        /// chunk whose body would have crossed the line).
        bytes: usize,
        /// The cap that was exceeded — kept in the variant so call-site
        /// logs / FAILURE messages report the boundary the operator
        /// would need to raise.
        max: usize,
    },
    /// Peer sent a message in a state that does not admit it (e.g.,
    /// RUN before HELLO; PULL with no active autocommit / explicit tx).
    /// Maps to `Neo.ClientError.Request.Invalid`.
    #[error("protocol violation: {0}")]
    ProtocolViolation(String),
    /// HELLO had an empty / missing `principal` field — v1.0-α auth
    /// rejects. Maps to `Neo.ClientError.Security.Unauthorized`.
    #[error("authentication failed: {0}")]
    Unauthorized(String),
    /// RUN's Cypher fell outside the openCypher subset the v1.0-α
    /// ArcQL parser admits. Maps to
    /// `Neo.ClientError.Statement.SyntaxError`.
    #[error("syntax error: {0}")]
    Syntax(String),
    /// **#797** — the statement referenced a `$name` parameter that the
    /// RUN message did not bind. A CLIENT fault: maps to
    /// `Neo.ClientError.Statement.ParameterMissing` (the neo4j-idiomatic
    /// code drivers recognize), NOT the `Neo.DatabaseError` server-fault
    /// bucket the pre-#797 `Internal("missing parameter")` rendered.
    #[error("parameter missing: {0}")]
    ParameterMissing(String),
    /// **#797** — a RUN `parameters` value was not an admissible
    /// parameter shape (a Node / Relationship / Path PackStream struct,
    /// or raw Bytes). A CLIENT fault: maps to
    /// `Neo.ClientError.Statement.TypeError`.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    /// Statement asked for a feature ArcGraph hasn't shipped yet. Maps to
    /// `Neo.ClientError.Statement.NotImplemented`.
    #[error("not implemented: {0}")]
    NotImplemented(String),
    /// Query execution exceeded a configured resource budget. Maps to
    /// Neo4j's transient resource class, not statement syntax.
    #[error("resource exhausted: {0}")]
    ResourceExhausted(String),
    /// Cancellation surfaced from the executor mid-RUN. Maps to
    /// `Neo.ClientError.Transaction.Terminated`.
    #[error("cancelled")]
    Cancelled,
    /// Tenant-scoped access faulted (tenant unknown to the catalog,
    /// substrate index unavailable). Maps to
    /// `Neo.ClientError.Database.*`.
    #[error("substrate: {0}")]
    Substrate(String),
    /// **#907.** A write-write **MVCC conflict** — the optimistic-
    /// concurrency loser whose commit lost the OCC validation race
    /// (threaded as the typed
    /// [`arcgraph_query::executor::SubstrateAccessError::Conflict`] through
    /// the auto-commit RUN path, or surfaced from the explicit-tx COMMIT).
    /// A NORMAL outcome under write contention, NOT a server fault: maps
    /// to the **retriable** `Neo.TransientError.Transaction.DeadlockDetected`,
    /// which the official Neo4j drivers AUTO-RETRY under
    /// `session.execute_write` / `execute_read`. Before #907 this
    /// flattened to [`Self::Internal`] →
    /// `Neo.DatabaseError.General.UnknownError` (FATAL, non-retriable) AND
    /// leaked the storage-layer wrapping ("substrate I/O error: write
    /// commit failed: MVCC commit failed") to the client — breaking the
    /// standard optimistic-concurrency retry pattern.
    #[error(
        "the transaction could not complete due to a concurrent write conflict; retry the transaction"
    )]
    TransientConflict {
        /// Contention point carried verbatim from the MVCC kernel (e.g.
        /// an internal `key:N` version-store key). For diagnostics /
        /// tracing only — deliberately NOT echoed into the user-facing
        /// FAILURE `message` field (no internal-layer leak, #907).
        target: String,
    },
    /// Catch-all for runtime / internal faults. Maps to
    /// `Neo.DatabaseError.General.UnknownError`.
    #[error("internal: {0}")]
    Internal(String),
    /// [`super::server::BoltServerConfig::bind`] resolves to a
    /// non-loopback address (e.g. `0.0.0.0:7687`, public IP) but
    /// `allow_remote_bind` is `false`. Shares the variant NAME with
    /// the HTTP transport's
    /// [`crate::transport::http::TransportError::BindAddrForbidden`]
    /// (design-v2 §9.4 line 668: "Bind 127.0.0.1 for local MCP
    /// servers") so a single ops-grep matches both transports. Surfaces
    /// at startup from
    /// [`super::server::BoltServerConfig::validate`] before
    /// `TcpListener::bind` so misconfiguration is loud, not silent.
    /// W14-retro IR L1-HIGH-4.
    #[error("bolt bind to non-loopback {addr} forbidden without allow_remote_bind")]
    BindAddrForbidden {
        /// The non-loopback `bind` value that triggered the refusal.
        addr: SocketAddr,
    },
}

impl BoltError {
    /// Neo4j status code string for a Bolt FAILURE message body. Per
    /// the spec the field is named `code` and follows the
    /// `Neo.<class>.<category>.<title>` shape.
    #[must_use]
    pub fn neo4j_code(&self) -> &'static str {
        match self {
            BoltError::HandshakeRejected(_) => "Neo.ClientError.Request.InvalidFormat",
            BoltError::Framing(_) => "Neo.ClientError.Request.InvalidFormat",
            BoltError::Io(_) => "Neo.DatabaseError.General.UnknownError",
            BoltError::Pack(_) => "Neo.ClientError.Request.InvalidFormat",
            BoltError::MessageTooLarge { .. } => "Neo.ClientError.Request.InvalidFormat",
            BoltError::ProtocolViolation(_) => "Neo.ClientError.Request.Invalid",
            BoltError::Unauthorized(_) => "Neo.ClientError.Security.Unauthorized",
            BoltError::Syntax(_) => "Neo.ClientError.Statement.SyntaxError",
            BoltError::ParameterMissing(_) => "Neo.ClientError.Statement.ParameterMissing",
            BoltError::InvalidParameter(_) => "Neo.ClientError.Statement.TypeError",
            BoltError::NotImplemented(_) => "Neo.ClientError.Statement.NotImplemented",
            BoltError::ResourceExhausted(_) => "Neo.TransientError.General.OutOfMemoryError",
            BoltError::Cancelled => "Neo.ClientError.Transaction.Terminated",
            BoltError::Substrate(_) => "Neo.ClientError.Database.DatabaseNotFound",
            // #907 — write-write MVCC conflict is RETRIABLE: the
            // `TransientError` class is what makes drivers auto-retry
            // `session.execute_write`. NOT `Terminated`/`LockClientStopped`
            // (the two transient codes drivers do NOT retry).
            BoltError::TransientConflict { .. } => {
                "Neo.TransientError.Transaction.DeadlockDetected"
            }
            BoltError::Internal(_) => "Neo.DatabaseError.General.UnknownError",
            // Startup-class fault — never reaches a client over Bolt
            // (validate() runs BEFORE TcpListener::bind, so the variant
            // is constructed AT STARTUP and propagates out via the
            // `serve_bolt_listener` Result; no Bolt FAILURE frame is
            // ever built from it). Mark `unreachable!()` per W14-retro
            // IR R1 NIT-2 to make the dead-code semantic explicit and
            // panic loudly if a future surface accidentally wires
            // BindAddrForbidden into a wire-frame path.
            BoltError::BindAddrForbidden { .. } => {
                unreachable!("BindAddrForbidden is startup-only; never reaches FAILURE wire")
            }
        }
    }

    /// Human-readable message for the FAILURE `message` field.
    /// Drivers surface this to application code so it should be
    /// useful but compact.
    #[must_use]
    pub fn message(&self) -> String {
        // `Display`'s output already includes the variant slot; for the
        // FAILURE field we strip the redundant prefix that the variant
        // adds when it's load-bearing for log triage but noisy for
        // user-facing rendering.
        self.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syntax_error_maps_to_neo_client_statement_syntaxerror() {
        let e = BoltError::Syntax("expected RETURN".into());
        assert_eq!(e.neo4j_code(), "Neo.ClientError.Statement.SyntaxError");
    }

    #[test]
    fn unauthorized_maps_to_neo_client_security_unauthorized() {
        let e = BoltError::Unauthorized("empty principal".into());
        assert_eq!(e.neo4j_code(), "Neo.ClientError.Security.Unauthorized");
    }

    #[test]
    fn cancelled_maps_to_neo_client_transaction_terminated() {
        let e = BoltError::Cancelled;
        assert_eq!(e.neo4j_code(), "Neo.ClientError.Transaction.Terminated");
    }

    #[test]
    fn protocol_violation_maps_to_neo_client_request_invalid() {
        let e = BoltError::ProtocolViolation("RUN before HELLO".into());
        assert_eq!(e.neo4j_code(), "Neo.ClientError.Request.Invalid");
    }

    #[test]
    fn parameter_missing_maps_to_neo_client_statement_parametermissing() {
        // #797 — a missing `$param` is a CLIENT fault (ParameterMissing),
        // NOT the `Neo.DatabaseError.General.UnknownError` the pre-fix
        // `Internal("missing parameter")` rendered.
        let e = BoltError::ParameterMissing("$id".into());
        assert_eq!(e.neo4j_code(), "Neo.ClientError.Statement.ParameterMissing");
    }

    #[test]
    fn invalid_parameter_maps_to_neo_client_statement_typeerror() {
        // #797 — a Node/Relationship/Path or Bytes param is a CLIENT
        // type fault.
        let e = BoltError::InvalidParameter("parameter `n`: graph entity".into());
        assert_eq!(e.neo4j_code(), "Neo.ClientError.Statement.TypeError");
    }

    #[test]
    fn resource_exhausted_maps_to_transient_resource_error() {
        let e = BoltError::ResourceExhausted("HashJoinOp build-side".into());
        let code = e.neo4j_code();
        assert_eq!(code, "Neo.TransientError.General.OutOfMemoryError");
        assert!(!code.starts_with("Neo.ClientError.Statement."));
    }

    #[test]
    fn message_field_round_trips_through_display() {
        let e = BoltError::Syntax("expected RETURN".into());
        assert!(e.message().contains("expected RETURN"));
    }

    // ── #907 — MVCC write-write conflict → retriable TransientError ──

    /// The chosen code must be in the `TransientError` CLASS — that is
    /// the contract that makes a Neo4j driver's managed transaction
    /// auto-retry it. (Was `Neo.DatabaseError.General.UnknownError`,
    /// fatal/non-retriable — the #907 defect.)
    #[test]
    fn mvcc_conflict_maps_to_retriable_transient_error() {
        let e = BoltError::TransientConflict {
            target: "key:6404".into(),
        };
        let code = e.neo4j_code();
        assert!(
            code.starts_with("Neo.TransientError."),
            "MVCC conflict must be a retriable TransientError; got {code}"
        );
        assert_eq!(code, "Neo.TransientError.Transaction.DeadlockDetected");
        // RED-on-revert: if TransientConflict mapped to
        // `Neo.DatabaseError.*` (the pre-#907 behavior), both asserts fail.
        assert!(!code.starts_with("Neo.DatabaseError."));
        assert!(!code.starts_with("Neo.ClientError."));
    }

    /// Driver-retry oracle: the official Neo4j drivers retry every
    /// `Neo.TransientError.*` EXCEPT `…Transaction.Terminated` and
    /// `…Transaction.LockClientStopped`. Pin that our code is NOT one of
    /// those two — so `session.execute_write` actually auto-retries it.
    #[test]
    fn mvcc_conflict_code_is_in_driver_auto_retry_set() {
        let code = BoltError::TransientConflict {
            target: "key:1".into(),
        }
        .neo4j_code();
        assert!(code.starts_with("Neo.TransientError."));
        assert_ne!(
            code, "Neo.TransientError.Transaction.Terminated",
            "Terminated is NOT auto-retried by drivers"
        );
        assert_ne!(
            code, "Neo.TransientError.Transaction.LockClientStopped",
            "LockClientStopped is NOT auto-retried by drivers"
        );
    }

    /// The FAILURE `message` must read as a clean "retry the transaction"
    /// and must NOT leak the storage-layer wrapping (the #907 symptom:
    /// "internal: substrate: substrate I/O error: write commit failed:
    /// MVCC commit failed: …") nor the raw internal conflict key.
    #[test]
    fn mvcc_conflict_message_does_not_leak_internals() {
        let msg = BoltError::TransientConflict {
            target: "key:6404".into(),
        }
        .message();
        for leak in [
            "substrate",
            "internal",
            "MVCC commit failed",
            "I/O",
            "i/o",
            "key:6404",
        ] {
            assert!(
                !msg.contains(leak),
                "FAILURE message must not leak {leak:?}; got {msg:?}"
            );
        }
        assert!(
            msg.contains("retry"),
            "message should advise the client to retry; got {msg:?}"
        );
    }

    /// No over-broadening: a GENUINE I/O fault (not an MVCC conflict)
    /// still maps to the fatal `Neo.DatabaseError.General.UnknownError`.
    /// Only the logical conflict was reclassified as transient.
    #[test]
    fn genuine_io_fault_still_maps_to_fatal_database_error() {
        assert_eq!(
            BoltError::Io("disk write failed".into()).neo4j_code(),
            "Neo.DatabaseError.General.UnknownError"
        );
        assert_eq!(
            BoltError::Internal("eval panic".into()).neo4j_code(),
            "Neo.DatabaseError.General.UnknownError"
        );
    }
}
