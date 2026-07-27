#!/usr/bin/env bash
#
# Active end-to-end verification — INDEX class recipe
# ---------------------------------------------------
# Per ADR-133 §D-4 (Index class row) + SPAWN_PROMPT_PREAMBLE
# addendum 22 + feedback_active_verification_per_pr.md.
#
# This recipe runs the DiskANN/Vamana 10K recall test against the
# worktree HEAD. The test:
#
#   1. Builds a 10 K-vector synthetic SIFT-class corpus (128-dim,
#      100 clusters × 100, σ=0.03 — deterministic XorShift32).
#   2. Builds a DiskAnnGraph (Vamana α-prune, R=70 / α=1.2 / L=100).
#   3. Runs 200 queries (half perturbed dataset points, half uniform
#      OOD) at l_search=128.
#   4. Computes recall@10 against an EXHAUSTIVE brute-force top-k
#      oracle (full linear scan, NaN-safe total_cmp) and asserts
#      recall@10 ≥ 0.95.
#
# This is the ADR-133 §D-4 "Index" row recipe ("10K vector insert +
# 1K queries; recall ≥ 0.90 vs brute-force oracle") at a STRICTER
# floor (≥ 0.95 ≫ the ≥ 0.90 §D-4 minimum — exceed-the-bar
# discipline) and is the proxy-scale companion to the ADR-189 §5.2
# row-1 GA gate (recall@10 ≥ 0.95). The binding 10M validation is the
# V-2 SSD-resident driver (ADR-195), not this proxy recipe.
#
# Pattern: ADR-133 §D-1 Pattern A (load-fixture + run + compare-vs-
# reference; reference = the exhaustive brute-force recall oracle).
#
# Opt-out: index-class PRs may NOT skip this recipe.
#
# Usage:
#   scripts/e2e/verify_index.sh
#
# Exit codes:
#   0       — recall test passed (recall@10 ≥ 0.95 vs brute-force).
#   non-0   — recall test failed OR cargo invocation failed.
#
# Output:
#   stdout/stderr — verbatim `cargo test --nocapture` output (includes
#                   the build-ms + measured recall@10 line).
#
# Per the W12 retro exit-code-capture mandate
# (feedback_pre_rebase_post_rebase_gauntlet_drift.md): capture this
# script's exit with `> /tmp/<wave>_e2e_index.log 2>&1; echo
# "exit_e2e_index=$?"`. NEVER pipe through `tee` / `tail` — those mask
# the upstream exit.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────
# Discovery: the DiskANN 10K recall test.
# ─────────────────────────────────────────────────────────────────
TEST_CRATE="arcgraph-vector"
TEST_BIN="diskann"
TEST_FN="diskann_build_search_recall_sift_subset"

echo "[verify_index] active-verification (index class) — ADR-133 §D-4"
echo "[verify_index] crate=${TEST_CRATE} test=${TEST_BIN} fn=${TEST_FN}"
echo "[verify_index] fixture=10K synthetic cluster (128-dim); oracle=exhaustive brute-force"
echo "[verify_index] asserts recall@10 >= 0.95 (ADR-189 §5.2 row 1 proxy; ADR-133 §D-4 Index ≥0.90 floor)"
echo ""

# ─────────────────────────────────────────────────────────────────
# Recipe execution — hermetic (in-process, deterministic synthetic
# corpus; no live-deploy / network / disk fixtures).
# ─────────────────────────────────────────────────────────────────
# Capture the exit WITHOUT letting `set -e` abort first: a bare
# `cargo test; RC=$?` would abort at the failing cargo line under
# `set -e`, leaving the FAILED branch + `exit "${RC}"` unreachable
# (no-op trampoline). `|| RC=$?` consumes the failure so RC is real and
# the summary + re-raise below actually fire on failure.
RC=0
cargo test \
    -p "${TEST_CRATE}" \
    --release \
    --test "${TEST_BIN}" \
    "${TEST_FN}" \
    -- --nocapture || RC=$?

# ─────────────────────────────────────────────────────────────────
# Result summary
# ─────────────────────────────────────────────────────────────────
echo ""
if [ "${RC}" -eq 0 ]; then
    echo "[verify_index] INDEX recipe PASSED (recall@10 >= 0.95 vs exhaustive brute-force oracle)"
else
    echo "[verify_index] INDEX recipe FAILED (cargo test exit=${RC})"
fi

# Re-raise the exact cargo exit code (NEVER swallow it; per W12 retro
# exit-code mandate).
exit "${RC}"
