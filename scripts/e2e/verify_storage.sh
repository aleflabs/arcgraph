#!/usr/bin/env bash
#
# Active end-to-end verification — STORAGE class recipe
# ------------------------------------------------------
# Per ADR-133 §D-4 (storage class row) + SPAWN_PROMPT_PREAMBLE
# addendum 22 + feedback_active_verification_per_pr.md (W26-MFI-4).
#
# This recipe runs the K-1 30s smoke harness against the worktree
# HEAD. The harness:
#
#   1. Builds a CrudStore + BufferPool + WAL stack.
#   2. Drives ~30s of CRUD + restart cycles with per-op rate-based
#      fault injection at the WAL boundary.
#   3. After each WAL teardown + recover_from_wal pass, the K-1
#      oracle verifies the 5 K-1 invariants:
#        - 1:1 unique:total CRUD invariant
#        - T1-strict-satisfied (Phase 5.5 baseline)
#        - Pre-crash ledger reconciliation
#        - Post-recovery MVCC stats rebuild correctness
#        - Per-tenant fault isolation (no cross-tenant contamination)
#   4. The test asserts the oracle passed; exits non-zero on
#      violation.
#
# Wall budget (ADR-133 §D-4): ~30s (controllable via K1_SMOKE_SECS
# env var; default 30s honored by the test harness).
#
# Pattern: ADR-133 §D-1 Pattern A (load-fixture + run + compare-vs-
# reference; reference is the K-1 invariant oracle).
#
# Opt-out: storage-class PRs may NOT skip this recipe per ADR-133
# §D-4 table (skip-eligible = NO).
#
# Usage:
#
#   # Default (30s K-1 smoke):
#   scripts/e2e/verify_storage.sh
#
#   # Quick dry-run (5s K-1 smoke; for active-verification gate
#   # dogfooding only — NOT acceptable as PR-open recipe output):
#   K1_SMOKE_SECS=5 scripts/e2e/verify_storage.sh
#
# Exit codes:
#   0       — K-1 smoke passed; all 5 invariants survived.
#   non-0   — K-1 smoke failed OR cargo invocation failed.
#
# Output:
#   stdout — verbatim `cargo test --nocapture` output (includes
#            K-1 oracle progress + invariant-check lines).
#   stderr — same; combined for log capture.
#
# Per the W12 retro exit-code-capture mandate
# (feedback_pre_rebase_post_rebase_gauntlet_drift.md): capture this
# script's exit with `> /tmp/<wave>_e2e_storage.log 2>&1; echo
# "exit_e2e_storage=$?"`. NEVER pipe through `tee` / `tail` — those
# mask the upstream exit.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────
# Discovery: where is the K-1 30s smoke test?
# ─────────────────────────────────────────────────────────────────
TEST_CRATE="arcgraph-storage"
TEST_BIN="k1_smoke_30s"
TEST_FN="k1_smoke_30s_per_op_injection_oracle"

# K-1 smoke duration (default 30s; configurable via K1_SMOKE_SECS).
SMOKE_SECS="${K1_SMOKE_SECS:-30}"

echo "[verify_storage] active-verification (storage class) — ADR-133 §D-4"
echo "[verify_storage] crate=${TEST_CRATE} test=${TEST_BIN} fn=${TEST_FN}"
echo "[verify_storage] K1_SMOKE_SECS=${SMOKE_SECS} (default 30)"
echo "[verify_storage] wall-budget per ADR-133 §D-4: ~30s (build + smoke)"
echo ""

# ─────────────────────────────────────────────────────────────────
# Recipe execution
# ─────────────────────────────────────────────────────────────────
# Per ADR-133 §D-2 hermetic-mode mandate: tempdir fixtures only;
# no live-deploy dependencies. The K-1 30s smoke is structurally
# hermetic (TempDir + InMemoryPageIo + in-process WAL).

K1_SMOKE_SECS="${SMOKE_SECS}" \
    cargo test \
        -p "${TEST_CRATE}" \
        --test "${TEST_BIN}" \
        --release \
        "${TEST_FN}" \
        -- --nocapture
RC=$?

# ─────────────────────────────────────────────────────────────────
# Result summary
# ─────────────────────────────────────────────────────────────────
echo ""
if [ "${RC}" -eq 0 ]; then
    echo "[verify_storage] STORAGE recipe PASSED (5 K-1 invariants survived ${SMOKE_SECS}s smoke)"
else
    echo "[verify_storage] STORAGE recipe FAILED (cargo test exit=${RC})"
fi

# Re-raise the exact cargo exit code (NEVER swallow it; per W12
# retro exit-code mandate).
exit "${RC}"
