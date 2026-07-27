//! W14δ M5-13 — Bolt 5.0 connection state machine.
//! ADR-197 (#802) — explicit-transaction states (TxReady / TxStreaming)
//! for BEGIN / COMMIT / ROLLBACK.
//!
//! Per the Bolt §"Server Lifecycle" spec a connection moves through
//! a deterministic state graph. The auto-commit subset is the base
//! diagram below; ADR-197 adds the explicit-transaction lane
//! (`Ready --BEGIN--> TxReady --RUN--> TxStreaming --PULL(done)-->
//! TxReady --COMMIT/ROLLBACK--> Ready`):
//!
//! ```text
//!     Initial
//!        │  HELLO
//!        ▼
//!   Authenticated
//!        │  (immediate, v1.0-α has no LOGON step)
//!        ▼
//!     Ready  ◄────────────────────┐
//!        │  RUN                   │ RESET / SUCCESS-after-PULL(has_more=false)
//!        ▼                        │
//!  Streaming  ── PULL (more) ─────┘
//!        │  PULL (final) / DISCARD
//!        ▼
//!     Ready
//!
//!  Any error from any state → Failed
//!  Failed: every C→S except RESET / GOODBYE → IGNORED
//!  RESET (from any state) → Ready
//!  GOODBYE (from any state) → Closed
//! ```
//!
//! # `Authenticated` collapse at v1.0-α
//!
//! The diagram preserves the spec-shape intermediate `Authenticated`
//! state for reader-mental-model alignment, but v1.0-α's
//! [`ConnState`] enum elides it — `Initial → Authenticated → Ready`
//! collapses into a single `Initial → Ready` transition on HELLO
//! success because v1.0-α exposes neither LOGON / LOGOFF (Bolt 5.1+,
//! lighting at v1.1) nor delegated-auth (Bolt 5.5+). When LOGON
//! support lands the enum gains an `Authenticated` variant additively
//! (the existing variants stay stable; no breaking change for
//! downstream drivers).
//!
//! # Why the FSM is its own type
//!
//! Per ADR-038 amendment-03 §M5↔M4 contract surface the executor
//! exposes EVALUATION-ORDER invariants (RUN before PULL) at the
//! request boundary; surfacing those at the Bolt-server-side state
//! transition layer means the listener loop does not have to
//! re-derive them. The FSM is intentionally tiny + total: every
//! message in every state has a defined transition.

use super::error::BoltError;
use super::message::ClientMessage;

/// Connection state per Bolt 5.0 §"Server Lifecycle". Covers BOTH the
/// auto-commit subset (`Ready` / `Streaming`) AND the ADR-197 (#802)
/// explicit-transaction lane (`TxReady` / `TxStreaming`) for
/// BEGIN / COMMIT / ROLLBACK.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnState {
    /// Connection just accepted; HANDSHAKE complete, awaiting HELLO.
    Initial,
    /// HELLO succeeded; idle awaiting RUN (auto-commit) or BEGIN.
    Ready,
    /// Auto-commit RUN in flight; awaiting PULL / DISCARD to drain
    /// results.
    Streaming,
    /// **ADR-197** — an explicit transaction is open (BEGIN seen),
    /// idle awaiting RUN (which stages into the held tx) or
    /// COMMIT / ROLLBACK.
    TxReady,
    /// **ADR-197** — RUN-in-explicit-transaction in flight; awaiting
    /// PULL / DISCARD. On PULL-done the connection returns to
    /// [`Self::TxReady`] (the tx stays open for more RUNs), NOT to
    /// [`Self::Ready`].
    TxStreaming,
    /// A failure was observed; client must RESET to recover.
    Failed,
    /// Peer sent GOODBYE; loop should exit.
    Closed,
}

/// Outcome of the FSM admitting an inbound message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    /// Server should process the message (run handler, emit
    /// response). Caller threads the result back via
    /// [`ConnFsm::commit_result`].
    Process,
    /// Server should reply with IGNORED. The connection is in
    /// `Failed` and the only recovery is RESET.
    Ignore,
    /// Server should reply with FAILURE describing the protocol
    /// violation. The state moves to `Failed`.
    ProtocolViolation(&'static str),
    /// Server should run the message AND immediately close after
    /// emitting the reply. Used for GOODBYE.
    ProcessThenClose,
}

/// The post-handler outcome the server reports back to the FSM via
/// [`ConnFsm::commit_result`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandlerOutcome {
    /// The handler succeeded. State advances per the state graph
    /// (e.g., RUN → Streaming, PULL with `has_more=false` → Ready).
    Success,
    /// The handler emitted FAILURE. Connection moves to Failed.
    Failure,
    /// The handler reported "still have rows". For PULL this means
    /// `has_more=true` → stay in Streaming; for other messages this
    /// is identical to `Success`.
    HasMore,
}

/// Bolt connection state machine. One instance per connection.
#[derive(Debug)]
pub struct ConnFsm {
    state: ConnState,
}

impl Default for ConnFsm {
    fn default() -> Self {
        Self::new()
    }
}

impl ConnFsm {
    /// Construct a fresh FSM in the [`ConnState::Initial`] state.
    /// Caller drives the handshake out-of-band before the first
    /// `admit` call (handshake itself is NOT modeled here — the
    /// server does the magic-preamble dance before allocating the
    /// FSM).
    pub fn new() -> Self {
        Self {
            state: ConnState::Initial,
        }
    }

    /// Current state — exposed for tests + diagnostics.
    pub fn state(&self) -> ConnState {
        self.state
    }

    /// Decide whether the inbound message is admissible in the
    /// current state. Returns the transition the server should take
    /// (run handler, ignore, etc.) but does NOT advance state — the
    /// caller threads the handler outcome back via
    /// [`ConnFsm::commit_result`] which is what actually advances state.
    pub fn admit(&self, msg: &ClientMessage) -> Transition {
        // GOODBYE / RESET are admissible from any non-Closed state.
        match msg {
            ClientMessage::Goodbye => return Transition::ProcessThenClose,
            ClientMessage::Reset => return Transition::Process,
            _ => {}
        }
        match (self.state, msg) {
            // Initial: only HELLO admissible.
            (ConnState::Initial, ClientMessage::Hello { .. }) => Transition::Process,
            (ConnState::Initial, _) => Transition::ProtocolViolation("expected HELLO first"),
            // Ready: RUN (auto-commit) admissible; BEGIN opens an
            // explicit tx; PULL / DISCARD / COMMIT / ROLLBACK are
            // protocol violations (no active stream / no open tx).
            (ConnState::Ready, ClientMessage::Run { .. }) => Transition::Process,
            (ConnState::Ready, ClientMessage::Begin { .. }) => Transition::Process,
            (ConnState::Ready, ClientMessage::Pull { .. } | ClientMessage::Discard { .. }) => {
                Transition::ProtocolViolation("PULL/DISCARD without active RUN")
            }
            (ConnState::Ready, ClientMessage::Commit | ClientMessage::Rollback) => {
                Transition::ProtocolViolation("COMMIT/ROLLBACK without an open transaction")
            }
            (ConnState::Ready, ClientMessage::Hello { .. }) => {
                Transition::ProtocolViolation("duplicate HELLO")
            }
            // Streaming (auto-commit): PULL / DISCARD admissible; RUN is
            // a protocol violation (auto-commit single-stream).
            (ConnState::Streaming, ClientMessage::Pull { .. } | ClientMessage::Discard { .. }) => {
                Transition::Process
            }
            (ConnState::Streaming, ClientMessage::Run { .. }) => {
                Transition::ProtocolViolation("RUN while streaming previous result")
            }
            (ConnState::Streaming, ClientMessage::Begin { .. }) => {
                Transition::ProtocolViolation("BEGIN while streaming previous result")
            }
            (ConnState::Streaming, ClientMessage::Commit | ClientMessage::Rollback) => {
                Transition::ProtocolViolation("COMMIT/ROLLBACK while streaming (no open tx)")
            }
            (ConnState::Streaming, ClientMessage::Hello { .. }) => {
                Transition::ProtocolViolation("duplicate HELLO")
            }
            // ── ADR-197 explicit-transaction states ──
            // TxReady (tx open, idle): RUN stages into the held tx;
            // COMMIT / ROLLBACK finalize it; BEGIN-in-tx + PULL/DISCARD
            // (no active stream) are violations.
            (ConnState::TxReady, ClientMessage::Run { .. }) => Transition::Process,
            (ConnState::TxReady, ClientMessage::Commit | ClientMessage::Rollback) => {
                Transition::Process
            }
            (ConnState::TxReady, ClientMessage::Begin { .. }) => {
                Transition::ProtocolViolation("BEGIN while a transaction is already open")
            }
            (ConnState::TxReady, ClientMessage::Pull { .. } | ClientMessage::Discard { .. }) => {
                Transition::ProtocolViolation("PULL/DISCARD without active RUN in transaction")
            }
            (ConnState::TxReady, ClientMessage::Hello { .. }) => {
                Transition::ProtocolViolation("duplicate HELLO")
            }
            // TxStreaming (RUN-in-tx in flight): PULL / DISCARD drain;
            // a subsequent RUN / BEGIN / COMMIT / ROLLBACK must wait for
            // the stream to drain.
            (
                ConnState::TxStreaming,
                ClientMessage::Pull { .. } | ClientMessage::Discard { .. },
            ) => Transition::Process,
            (ConnState::TxStreaming, ClientMessage::Run { .. }) => {
                Transition::ProtocolViolation("RUN while streaming previous result in transaction")
            }
            (ConnState::TxStreaming, ClientMessage::Begin { .. }) => {
                Transition::ProtocolViolation("BEGIN while streaming previous result")
            }
            (ConnState::TxStreaming, ClientMessage::Commit | ClientMessage::Rollback) => {
                Transition::ProtocolViolation(
                    "COMMIT/ROLLBACK while streaming (drain the stream first)",
                )
            }
            (ConnState::TxStreaming, ClientMessage::Hello { .. }) => {
                Transition::ProtocolViolation("duplicate HELLO")
            }
            // Failed: every non-RESET / non-GOODBYE message → IGNORED.
            (ConnState::Failed, _) => Transition::Ignore,
            // Closed: no message admissible (caller should have
            // exited the loop).
            (ConnState::Closed, _) => Transition::ProtocolViolation("connection closed"),
            // RESET / GOODBYE were dispatched at the top of the
            // function with `return`; the compiler doesn't see those
            // as exhaustive across (state × msg) so we restate the
            // catch-all explicitly. Reaching this arm is a bug.
            (_, ClientMessage::Reset | ClientMessage::Goodbye) => {
                debug_assert!(
                    false,
                    "RESET/GOODBYE should be handled in the top-level match"
                );
                Transition::Process
            }
        }
    }

    /// Commit the handler's outcome back into the state machine.
    /// The `msg` argument is the message that was handled (so we
    /// know whether RUN→Streaming or PULL→Ready).
    pub fn commit_result(
        &mut self,
        msg: &ClientMessage,
        outcome: HandlerOutcome,
    ) -> Result<(), BoltError> {
        // RESET always returns to Ready, regardless of outcome (a
        // RESET that fails is a server bug, not a client bug —
        // RESET semantics are total).
        if matches!(msg, ClientMessage::Reset) {
            self.state = ConnState::Ready;
            return Ok(());
        }
        // GOODBYE is the only message that drives to Closed.
        if matches!(msg, ClientMessage::Goodbye) {
            self.state = ConnState::Closed;
            return Ok(());
        }
        match outcome {
            HandlerOutcome::Failure => {
                self.state = ConnState::Failed;
                return Ok(());
            }
            HandlerOutcome::Success | HandlerOutcome::HasMore => {}
        }
        // Per-message success transitions.
        self.state = match (self.state, msg, outcome) {
            (ConnState::Initial, ClientMessage::Hello { .. }, _) => ConnState::Ready,
            (ConnState::Ready, ClientMessage::Run { .. }, _) => ConnState::Streaming,
            (
                ConnState::Streaming,
                ClientMessage::Pull { .. } | ClientMessage::Discard { .. },
                HandlerOutcome::HasMore,
            ) => ConnState::Streaming,
            (
                ConnState::Streaming,
                ClientMessage::Pull { .. } | ClientMessage::Discard { .. },
                HandlerOutcome::Success,
            ) => ConnState::Ready,
            // ── ADR-197 explicit-transaction transitions ──
            // BEGIN (from Ready) → TxReady (tx open).
            (ConnState::Ready, ClientMessage::Begin { .. }, _) => ConnState::TxReady,
            // RUN-in-tx (from TxReady) → TxStreaming (stream the result).
            (ConnState::TxReady, ClientMessage::Run { .. }, _) => ConnState::TxStreaming,
            // PULL/DISCARD in tx with more rows → stay TxStreaming.
            (
                ConnState::TxStreaming,
                ClientMessage::Pull { .. } | ClientMessage::Discard { .. },
                HandlerOutcome::HasMore,
            ) => ConnState::TxStreaming,
            // PULL/DISCARD in tx, stream drained → back to TxReady (the
            // tx STAYS OPEN for more RUNs / COMMIT — NOT to Ready).
            (
                ConnState::TxStreaming,
                ClientMessage::Pull { .. } | ClientMessage::Discard { .. },
                HandlerOutcome::Success,
            ) => ConnState::TxReady,
            // COMMIT / ROLLBACK (from TxReady) → Ready (tx finalized).
            (ConnState::TxReady, ClientMessage::Commit | ClientMessage::Rollback, _) => {
                ConnState::Ready
            }
            (state, _, _) => state, // no transition (RESET/GOODBYE handled above; idle)
        };
        Ok(())
    }

    /// Drive the FSM to `Failed` after emitting a FAILURE the server
    /// has already serialized to the wire. Called for any error that
    /// should transition the FSM to Failed: spec-shaped protocol
    /// violation (valid message in invalid state, e.g., RUN before
    /// HELLO), but also codec-layer faults — decode rejection
    /// (Pack/Framing) at `super::server::handle_pair_inner`'s
    /// message-loop preamble surfaces here too, because from the
    /// FSM's perspective the post-FAILURE state is identical
    /// regardless of which layer rejected the inbound bytes.
    pub fn record_violation(&mut self) {
        self.state = ConnState::Failed;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn hello() -> ClientMessage {
        ClientMessage::Hello {
            user_agent: Some("t/1".into()),
            scheme: Some("none".into()),
            principal: None,
            credentials: None,
            routing: None,
            extras: BTreeMap::new(),
        }
    }
    fn run() -> ClientMessage {
        ClientMessage::Run {
            query: "RETURN 1".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        }
    }
    fn pull() -> ClientMessage {
        ClientMessage::Pull { n: -1, qid: None }
    }

    #[test]
    fn happy_path_initial_to_ready_to_streaming_to_ready() {
        let mut fsm = ConnFsm::new();
        // Initial → admit HELLO → Process.
        assert_eq!(fsm.admit(&hello()), Transition::Process);
        fsm.commit_result(&hello(), HandlerOutcome::Success)
            .unwrap();
        assert_eq!(fsm.state(), ConnState::Ready);
        // Ready → admit RUN → Process.
        assert_eq!(fsm.admit(&run()), Transition::Process);
        fsm.commit_result(&run(), HandlerOutcome::Success).unwrap();
        assert_eq!(fsm.state(), ConnState::Streaming);
        // Streaming → admit PULL → Process.
        assert_eq!(fsm.admit(&pull()), Transition::Process);
        // PULL with has_more=false (Success outcome) → Ready.
        fsm.commit_result(&pull(), HandlerOutcome::Success).unwrap();
        assert_eq!(fsm.state(), ConnState::Ready);
    }

    #[test]
    fn pull_with_has_more_stays_in_streaming() {
        let mut fsm = ConnFsm::new();
        fsm.commit_result(&hello(), HandlerOutcome::Success)
            .unwrap();
        fsm.commit_result(&run(), HandlerOutcome::Success).unwrap();
        assert_eq!(fsm.state(), ConnState::Streaming);
        fsm.commit_result(&pull(), HandlerOutcome::HasMore).unwrap();
        assert_eq!(fsm.state(), ConnState::Streaming);
    }

    #[test]
    fn run_before_hello_is_protocol_violation() {
        let fsm = ConnFsm::new();
        match fsm.admit(&run()) {
            Transition::ProtocolViolation(_) => {}
            other => panic!("expected violation, got {other:?}"),
        }
    }

    #[test]
    fn pull_in_ready_is_protocol_violation() {
        let mut fsm = ConnFsm::new();
        fsm.commit_result(&hello(), HandlerOutcome::Success)
            .unwrap();
        match fsm.admit(&pull()) {
            Transition::ProtocolViolation(_) => {}
            other => panic!("expected violation, got {other:?}"),
        }
    }

    #[test]
    fn failure_routes_to_failed_then_resets_to_ready() {
        let mut fsm = ConnFsm::new();
        fsm.commit_result(&hello(), HandlerOutcome::Success)
            .unwrap();
        fsm.commit_result(&run(), HandlerOutcome::Failure).unwrap();
        assert_eq!(fsm.state(), ConnState::Failed);
        // Any non-RESET/GOODBYE message → IGNORED.
        assert_eq!(fsm.admit(&run()), Transition::Ignore);
        assert_eq!(fsm.admit(&pull()), Transition::Ignore);
        // RESET → Process; outcome → Ready.
        assert_eq!(fsm.admit(&ClientMessage::Reset), Transition::Process);
        fsm.commit_result(&ClientMessage::Reset, HandlerOutcome::Success)
            .unwrap();
        assert_eq!(fsm.state(), ConnState::Ready);
    }

    #[test]
    fn goodbye_drives_to_closed() {
        let mut fsm = ConnFsm::new();
        fsm.commit_result(&hello(), HandlerOutcome::Success)
            .unwrap();
        assert_eq!(
            fsm.admit(&ClientMessage::Goodbye),
            Transition::ProcessThenClose
        );
        fsm.commit_result(&ClientMessage::Goodbye, HandlerOutcome::Success)
            .unwrap();
        assert_eq!(fsm.state(), ConnState::Closed);
    }

    #[test]
    fn reset_admissible_from_any_state() {
        // Initial → RESET → Process → Ready.
        let mut f = ConnFsm::new();
        assert_eq!(f.admit(&ClientMessage::Reset), Transition::Process);
        f.commit_result(&ClientMessage::Reset, HandlerOutcome::Success)
            .unwrap();
        assert_eq!(f.state(), ConnState::Ready);
        // Streaming → RESET → Ready.
        let mut f = ConnFsm::new();
        f.commit_result(&hello(), HandlerOutcome::Success).unwrap();
        f.commit_result(&run(), HandlerOutcome::Success).unwrap();
        assert_eq!(f.state(), ConnState::Streaming);
        f.commit_result(&ClientMessage::Reset, HandlerOutcome::Success)
            .unwrap();
        assert_eq!(f.state(), ConnState::Ready);
    }

    #[test]
    fn duplicate_hello_is_protocol_violation() {
        let mut fsm = ConnFsm::new();
        fsm.commit_result(&hello(), HandlerOutcome::Success)
            .unwrap();
        match fsm.admit(&hello()) {
            Transition::ProtocolViolation(_) => {}
            other => panic!("expected violation, got {other:?}"),
        }
    }

    // ── ADR-197 explicit-transaction FSM transitions ──

    fn begin() -> ClientMessage {
        ClientMessage::Begin {
            extra: BTreeMap::new(),
        }
    }

    /// Drive a fresh FSM to `Ready` (post-HELLO).
    fn ready_fsm() -> ConnFsm {
        let mut fsm = ConnFsm::new();
        fsm.commit_result(&hello(), HandlerOutcome::Success)
            .unwrap();
        fsm
    }

    #[test]
    fn explicit_tx_happy_path_begin_run_pull_commit() {
        let mut fsm = ready_fsm();
        // Ready --BEGIN--> TxReady.
        assert_eq!(fsm.admit(&begin()), Transition::Process);
        fsm.commit_result(&begin(), HandlerOutcome::Success)
            .unwrap();
        assert_eq!(fsm.state(), ConnState::TxReady);
        // TxReady --RUN--> TxStreaming.
        assert_eq!(fsm.admit(&run()), Transition::Process);
        fsm.commit_result(&run(), HandlerOutcome::Success).unwrap();
        assert_eq!(fsm.state(), ConnState::TxStreaming);
        // TxStreaming --PULL(done)--> TxReady (tx STAYS open).
        assert_eq!(fsm.admit(&pull()), Transition::Process);
        fsm.commit_result(&pull(), HandlerOutcome::Success).unwrap();
        assert_eq!(fsm.state(), ConnState::TxReady);
        // TxReady --COMMIT--> Ready.
        assert_eq!(fsm.admit(&ClientMessage::Commit), Transition::Process);
        fsm.commit_result(&ClientMessage::Commit, HandlerOutcome::Success)
            .unwrap();
        assert_eq!(fsm.state(), ConnState::Ready);
    }

    #[test]
    fn explicit_tx_multi_statement_run_run_commit() {
        // TxReady → RUN → (PULL done → TxReady) → RUN again → COMMIT.
        let mut fsm = ready_fsm();
        fsm.commit_result(&begin(), HandlerOutcome::Success)
            .unwrap();
        fsm.commit_result(&run(), HandlerOutcome::Success).unwrap();
        fsm.commit_result(&pull(), HandlerOutcome::Success).unwrap();
        assert_eq!(fsm.state(), ConnState::TxReady);
        // Second RUN in the SAME tx is admissible (multi-statement).
        assert_eq!(fsm.admit(&run()), Transition::Process);
        fsm.commit_result(&run(), HandlerOutcome::Success).unwrap();
        assert_eq!(fsm.state(), ConnState::TxStreaming);
    }

    #[test]
    fn explicit_tx_rollback_returns_to_ready() {
        let mut fsm = ready_fsm();
        fsm.commit_result(&begin(), HandlerOutcome::Success)
            .unwrap();
        assert_eq!(fsm.admit(&ClientMessage::Rollback), Transition::Process);
        fsm.commit_result(&ClientMessage::Rollback, HandlerOutcome::Success)
            .unwrap();
        assert_eq!(fsm.state(), ConnState::Ready);
    }

    #[test]
    fn pull_with_has_more_stays_in_tx_streaming() {
        let mut fsm = ready_fsm();
        fsm.commit_result(&begin(), HandlerOutcome::Success)
            .unwrap();
        fsm.commit_result(&run(), HandlerOutcome::Success).unwrap();
        assert_eq!(fsm.state(), ConnState::TxStreaming);
        fsm.commit_result(&pull(), HandlerOutcome::HasMore).unwrap();
        assert_eq!(fsm.state(), ConnState::TxStreaming);
    }

    #[test]
    fn commit_without_open_tx_is_protocol_violation() {
        // Ready (no BEGIN) --COMMIT--> violation (not a panic).
        let fsm = ready_fsm();
        match fsm.admit(&ClientMessage::Commit) {
            Transition::ProtocolViolation(_) => {}
            other => panic!("expected violation, got {other:?}"),
        }
        match fsm.admit(&ClientMessage::Rollback) {
            Transition::ProtocolViolation(_) => {}
            other => panic!("expected violation, got {other:?}"),
        }
    }

    #[test]
    fn begin_while_in_tx_is_protocol_violation() {
        let mut fsm = ready_fsm();
        fsm.commit_result(&begin(), HandlerOutcome::Success)
            .unwrap();
        assert_eq!(fsm.state(), ConnState::TxReady);
        match fsm.admit(&begin()) {
            Transition::ProtocolViolation(_) => {}
            other => panic!("expected violation, got {other:?}"),
        }
    }

    #[test]
    fn pull_in_tx_ready_without_run_is_protocol_violation() {
        let mut fsm = ready_fsm();
        fsm.commit_result(&begin(), HandlerOutcome::Success)
            .unwrap();
        match fsm.admit(&pull()) {
            Transition::ProtocolViolation(_) => {}
            other => panic!("expected violation, got {other:?}"),
        }
    }

    #[test]
    fn reset_aborts_open_tx_returns_to_ready() {
        // RESET admissible from TxReady → Ready (the handler aborts the
        // held tx; the FSM just returns to Ready).
        let mut fsm = ready_fsm();
        fsm.commit_result(&begin(), HandlerOutcome::Success)
            .unwrap();
        assert_eq!(fsm.admit(&ClientMessage::Reset), Transition::Process);
        fsm.commit_result(&ClientMessage::Reset, HandlerOutcome::Success)
            .unwrap();
        assert_eq!(fsm.state(), ConnState::Ready);
    }
}
