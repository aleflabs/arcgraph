//! Slice K K-1 — Pre-v1.0-alpha hardening harness scaffolding.
//!
//! Per ADR-038 amendment-03 §"Slice K" (renamed from "Jepsen-class
//! harness" per amendment-03 Structural-4). Slice K covers Aphyr's
//! 4 Jepsen criteria (a) and partial (b); v1.1 may pursue the full
//! 4-criteria certification per the same Structural-4 fix.
//!
//! ## Scope
//!
//! K-1 is **harness scaffolding**, not the multi-hour campaign:
//!
//! - [`injection`] — per-op rate-based fault-injection API.
//!   `InjectionConfig` carries per-seam rates (1 % WAL fsync /
//!   0.5 % snapshot install / 0.1 % process crash by default);
//!   `InjectionDecisionRng` is a deterministic `XorShift` so a
//!   given `(config, op_count, seed)` triple produces a reproducible
//!   fault sequence.
//! - [`subprocess`] — SIGKILL fork harness. Spawns the test binary
//!   with an env-var-driven workload selector; parent monitors child;
//!   child runs workload; parent issues SIGKILL after a configured
//!   crash window; parent restarts child + verifies recovery state
//!   matches pre-crash committed state.
//! - [`oracle`] — recovery validation oracle. Checks post-recovery
//!   invariants: 1:1 unique:total CRUD invariant (Phase 5.5 baseline),
//!   T1-strict-satisfied count, catalog-stats consistency (per M4-41
//!   stats infrastructure; per PR #170 reviewer Finding 1 this is
//!   where stats persistence/recovery becomes load-bearing).
//!   K-1b extends with cross-tenant invariants
//!   (`verify_cross_tenant_invariants`) — see issue #214.
//! - [`multi_tenant`] — K-1b multi-tenant workload generator
//!   (issue #214). Interleaves N tenants' commits with per-tenant
//!   `InjectionConfig`; pairs with the K-1b per-tenant
//!   `PreCrashLedger` directory mode in [`subprocess`].
//! - [`encoding_mismatch`] — K-1c+d encoding-mismatch I-V coverage
//!   (issue #215). Five deterministic primitives that mutate a
//!   `RecoveredState` to simulate each of the five encoding
//!   surfaces (record format / key format / MVCC chain layout /
//!   WAL commit-bundle atomicity / cross-tenant serialization);
//!   each maps to a SPECIFIC `OracleViolation` variant per the
//!   K-1d row's "per-class violation reporting wired through
//!   `OracleViolation`" exit criterion.
//!
//! ## Hooks vs. production
//!
//! The injection seams in this module are **orthogonal to production
//! correctness**. The harness:
//!
//! - DOES NOT modify the production commit pipeline in `crud::commit`.
//! - DOES NOT modify `WalWriter` / `WalHandle` / `BackgroundFsyncScheduler`.
//! - DOES NOT modify snapshot flush / install in `vector_store::snapshot`.
//!
//! Instead, the harness drives faults from the test side by:
//!
//! - Tearing down + re-spawning [`crate::wal::WalWriter`] mid-workload
//!   (mirrors the Phase 5.5 30 s torture pattern in
//!   `tests/phase_5_5_torture.rs`).
//! - Calling [`crate::vector_store::flush_snapshot_with_crash_point`]
//!   with a chosen [`crate::vector_store::CrashPoint`].
//! - Forking the test binary + sending SIGKILL via `libc::kill` (or
//!   the equivalent platform call).
//! - Restarting via `Command::spawn(env::current_exe())` with an
//!   env-var-driven workload re-entry point.
//!
//! No production source edit is required to support K-1. Future
//! K-2 / K-3 may surface the need for additional production hooks
//! (e.g., a `WalHandle::install_failure_injector(...)` method); those
//! land via separate ADR + slice when needed.
//!
//! ## Forward-references
//!
//! - **K-2** — multi-FS variation (APFS / ext4 / XFS / EBS) +
//!   concurrent snapshot-during-recovery. K-2 ships [`multi_fs`]
//!   adapters + the `tests/k2_fault_during_recovery.rs` proptest +
//!   the `tests/k2_concurrent_snapshot_recovery.rs` integration test
//!   (issue #223; ADR-038 amendment-03 §"Slice K" K-2 row).
//! - **K-3** — 10 K-cycle long-running campaigns + encoding-mismatch
//!   I-V coverage. Builds on K-2's multi-FS substrate.
//! - **v1.1 Jepsen-class certification** — open-source Jepsen DSL
//!   wrapper; published consistency-model spec coverage; checksum-
//!   pinned harness commit + dataset + report. Out of v1.0 scope.

pub mod encoding_mismatch;
pub mod injection;
pub mod multi_fs;
pub mod multi_tenant;
pub mod oracle;
pub mod subprocess;
