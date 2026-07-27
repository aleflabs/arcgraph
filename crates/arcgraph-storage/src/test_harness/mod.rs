//! Test-harness scaffolding for adversarial hardening campaigns.
//!
//! This module is **load-bearing for tests only**. Production code paths do
//! not call into anything under `test_harness::`. The module exists to
//! collect cross-cutting fault-injection / recovery / oracle helpers that
//! several integration tests share (and that would otherwise be copy-
//! pasted across `tests/*.rs` files).
//!
//! ## Submodules
//!
//! - [`k1`] — Slice K K-1 pre-v1.0-alpha hardening harness scaffolding
//!   (per ADR-038 amendment-03 §"Slice K"). Per-op rate-based fault
//!   injection at the WAL / snapshot seams + SIGKILL subprocess crash
//!   harness + recovery oracle. Consumed by:
//!     - `tests/k1_smoke_30s.rs` (CI-gating 30 s smoke run)
//!     - `tests/k1_extended_smoke_5min.rs` (`K1_EXTENDED_SMOKE=1`
//!       opt-in 5 min extended smoke)
//! - [`jepsen`] — Jepsen-style MVCC isolation test scaffolding
//!   (per ADR-047). History recorder + bank-transfer workload +
//!   snapshot-isolation checker + fault-injection adapter over
//!   the K-1 primitives. Consumed by:
//!     - `tests/jepsen_bank_transfer_snapshot.rs` (steady-state
//!       4-client × 100-op SI sanity test; the `JEPSEN_SIGKILL=1`
//!       opt-in variant is deferred to v1.1 alongside list-append +
//!       Elle — see ADR-047 §"Open questions").
//!
//! ## Why ship it as a public module instead of `#[cfg(test)]`?
//!
//! - The harness is consumed from `tests/*.rs` (integration tests),
//!   which compile against the crate's library target — `#[cfg(test)]`
//!   modules are not visible from there.
//! - The harness MAY be consumed by future per-crate fault-injection
//!   campaigns at K-2 / K-3 (multi-FS variation, encoding-mismatch
//!   I-V coverage, 10 K-cycle long-running campaigns) — keeping it
//!   `pub` avoids a `#[cfg(test)]` → `pub` migration mid-campaign.
//! - Production code paths do NOT depend on it; it's strictly
//!   reachable from test binaries (`tests/k1_smoke_30s.rs`,
//!   `tests/k1_extended_smoke_5min.rs`, and any future subprocess
//!   workload entry points the harness self-execs into).
//!
//! ## Bounded-context discipline
//!
//! The harness lives inside `arcgraph-storage` because every fault
//! seam it injects against is a storage seam (WAL fsync, snapshot
//! install, commit pipeline). Cross-substrate harness coverage at
//! K-2 / K-3 (multi-tenant query / vector / community paths) calls
//! into this module from `tests/*.rs` files in other crates via the
//! storage crate's public `test_harness::k1` API.

pub mod jepsen;
pub mod k1;
