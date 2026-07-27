//! Jepsen-style MVCC isolation test scaffolding (ADR-047).
//!
//! ## Why this module exists
//!
//! ArcGraph's MVCC kernel (`crate::transaction`) claims **snapshot
//! isolation** per `transaction.rs` invariants 1-8 + ADR-016
//! §"Context". The claim is defended today by
//! per-invariant unit + proptest coverage in
//! `tests/mvcc_*.rs` — invariants 1 (snapshot iso), 2 (no lost
//! updates), 3 (RYW), 4 (WW conflict), 5 (GC safety), 6 (begin/gc
//! serialization), 7 (commit atomicity), 8 (pipelining) per the
//! `transaction.rs` module rustdoc.
//!
//! What those tests do NOT cover: a recorded multi-client history is
//! **as a whole** SI-legal. The Adya 2000 taxonomy (G1a/G1b/G1c/G2-item/
//! G-single) gives the rigorous "what SI forbids" definitions, and
//! the canonical way to test against them is to record an interleaved
//! history of operations + verify a checker predicate over the whole
//! history — the Jepsen / Elle approach (Kingsbury & Alvaro VLDB 2020).
//!
//! This module is the **foundation** of that approach, sized for the
//! v0.1.0-alpha.0+1 release cycle and explicitly scoped to:
//!
//! 1. Recording operation histories (start-LSN, commit-LSN, reads,
//!    writes, outcome) from concurrent worker threads.
//! 2. A canonical **bank-transfer** workload (Bailis et al. 2014
//!    §3.1) whose sum-invariant catches G1a/G1b, subset-of-G1c, and
//!    commit-atomicity torn writes (write skew proper is a v1.1
//!    follow-up; see ADR-047 §"Consequences" for the not-caught list).
//! 3. A sum-invariant + per-op-visibility **checker**.
//! 4. **Fault-injection** seams that consume the K-1 primitives
//!    ([`super::k1::injection`] + [`super::k1::subprocess`]) so the
//!    same workload runs under SIGKILL-during-commit and verifies
//!    recovery preserves SI.
//!
//! What this module deliberately does NOT cover (v1.1 follow-ups,
//! per ADR-047 §"Open questions"):
//!
//! - List-append workload + Elle cycle detection.
//! - Range-scan / phantom-read invariants.
//! - Multi-tenant interleaved workloads.
//! - Read-only-anomaly catalogue (Bailis Hermitage queries).
//!
//! ## Submodules
//!
//! - [`history`] — `Op`, `OpResult`, `OperationHistory`,
//!   `RecordedOp` types. The history is a flat
//!   `Vec<RecordedOp>` that is `Send + Sync` so worker threads
//!   append to a `Mutex`-wrapped collector.
//! - [`workload`] — `BankTransferWorkload`. Runs N clients × M ops
//!   each; each op picks two random accounts and transfers a
//!   random amount inside one transaction. Conflicts retry up to
//!   `MAX_RETRIES` before being recorded as `OpResult::Aborted`.
//! - [`checker`] — `SnapshotIsolationChecker`. Verifies the
//!   sum-invariant (Bailis §3.1) and per-op visibility
//!   consistency. Returns a `CheckerVerdict` with structured
//!   failure context if any anomaly is detected.
//! - [`fault_injection`] — `CommitSigkillHook` and `FsyncStallHook`,
//!   thin adapters over [`super::k1::injection::InjectionConfig`]
//!   and `super::k1::subprocess::SubprocessHandle`. The Jepsen
//!   module does NOT define new fault primitives; it composes K-1.
//!
//! ## Why `pub` (not `#[cfg(test)]`)?
//!
//! Same reasoning as [`super::k1`]: the harness is consumed from
//! `tests/*.rs` integration tests, which compile against the crate's
//! library target. `#[cfg(test)]` is invisible from there.
//!
//! ## Bounded-context discipline
//!
//! - No new crate (`arcgraph-jepsen`) — v1.1 adds no new bounded contexts.
//! - No new dependencies — proptest, rand, bytes, parking_lot,
//!   crossbeam-channel are already in the storage crate.
//! - The harness lives **inside** `arcgraph-storage` because every
//!   contract it tests (snapshot isolation, MVCC commit pipeline,
//!   WAL durability) is a storage contract.
//!
//! ## Determinism contract
//!
//! Workload generators take an explicit RNG seed; the same seed
//! produces the same `(client_id, op_id, src_account, dst_account,
//! amount)` sequence across runs. The *interleaving* of those ops
//! across worker threads is non-deterministic by design (Jepsen
//! relies on randomised concurrency to surface anomalies). When the
//! checker flags an anomaly, the printed history is the
//! load-bearing reproduction artifact — re-running with the same
//! seed produces a *similar* (but not byte-equal) history, which is
//! enough to debug because the recorded LSNs anchor the failure to
//! a specific MVCC commit ordering.
//!
//! This is intentionally different from
//! `feedback_determinism_oracle_concurrency_tests.md`'s "binary-equal
//! reference snapshot" oracle: that pattern fits algorithms whose
//! output is deterministic; Jepsen workloads are deliberately
//! non-deterministic so the *checker predicate* is the oracle.

pub mod checker;
pub mod fault_injection;
pub mod history;
pub mod workload;
