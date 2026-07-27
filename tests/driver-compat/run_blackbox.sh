#!/usr/bin/env bash
# W26-γ-3 / ADR-136 §D4 — blackbox driver suite orchestrator.
#
# Spawns `arcgraph serve --bolt --in-memory` (ephemeral; W28 / ADR-183),
# runs all Python driver tests against it, tears the server down. A
# tempdir holds the server log only. Hermetic tests run unconditionally;
# live-server tests run only if the arcgraph binary is on PATH.
#
# Usage:
#   bash tests/driver-compat/run_blackbox.sh
#
# Exit codes:
#   0 — all hermetic tests passed (live tests may have SKIPPED).
#   1 — at least one test failed.
#   2 — environment setup error (Python missing, etc).
#
# Per ADR-136 §D4 driver-suite blackbox deliverable + workspace-level
# driver-compat README.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="${SCRIPT_DIR}/python"

# ─── 1. Sanity check Python ───────────────────────────────────────────

if ! command -v python3 >/dev/null 2>&1; then
    echo "[run_blackbox] FAIL: python3 not on PATH" >&2
    exit 2
fi

PYTHON_VERSION="$(python3 --version 2>&1 | awk '{print $2}')"
echo "[run_blackbox] Python: ${PYTHON_VERSION}"

# ─── 2. Check Python deps (don't auto-install) ────────────────────────

MISSING_DEPS=0
if ! python3 -c 'import neo4j' 2>/dev/null; then
    echo "[run_blackbox] WARN: 'neo4j' package not installed; Bolt smoke will SKIP" >&2
    MISSING_DEPS=$((MISSING_DEPS + 1))
fi

# ─── 3. Run hermetic Python tests (no server needed) ─────────────────

PASS=0
FAIL=0
SKIP=0

echo "[run_blackbox] === HERMETIC TESTS ==="

# MCP TOON parser tests — always hermetic mode
echo "[run_blackbox] running test_mcp_toon_round_trip.py (hermetic)"
if python3 "${PYTHON_DIR}/test_mcp_toon_round_trip.py" >&2; then
    PASS=$((PASS + 1))
else
    echo "[run_blackbox] FAIL: test_mcp_toon_round_trip.py" >&2
    FAIL=$((FAIL + 1))
fi

# gRPC streaming tests — all skipped at v1.0-α (forward-pin)
echo "[run_blackbox] running test_grpc_streaming.py (skipped at v1.0-α)"
if python3 "${PYTHON_DIR}/test_grpc_streaming.py" >&2; then
    SKIP=$((SKIP + 1))
fi

# ─── 4. Live server tests (gated on binary availability) ─────────────

if [ "${ARCGRAPH_BOLT_SKIP_OK:-0}" = "1" ]; then
    echo "[run_blackbox] ARCGRAPH_BOLT_SKIP_OK=1; skipping live Bolt tests"
    SKIP=$((SKIP + 1))
elif ! command -v arcgraph >/dev/null 2>&1; then
    echo "[run_blackbox] WARN: 'arcgraph' binary not on PATH; live tests SKIP"
    SKIP=$((SKIP + 1))
elif [ ${MISSING_DEPS} -gt 0 ]; then
    echo "[run_blackbox] WARN: Python deps missing; live tests SKIP"
    SKIP=$((SKIP + 1))
else
    echo "[run_blackbox] === LIVE TESTS ==="
    TEMP_DIR="$(mktemp -d)"
    trap 'rm -rf "${TEMP_DIR}"' EXIT
    echo "[run_blackbox] tempdir: ${TEMP_DIR}"

    # Spawn server in background.
    # W28 / ADR-183: `serve` refuses to start without an explicit storage
    # mode. This blackbox is a hermetic driver-compat smoke (spawn → drive
    # Bolt → tear down; no restart-survival assertion), so it passes
    # `--in-memory` (the ephemeral, non-durable store). The prior
    # `--data-dir "${TEMP_DIR}"` was a no-op typo — the flag is `--data`,
    # not `--data-dir`, so it never parsed; `--in-memory` is the correct
    # flag for an ephemeral smoke per ADR-183 §Policy. (server.log still
    # lands in the tempdir, which the EXIT trap cleans up.)
    arcgraph serve --bolt 127.0.0.1:17687 --in-memory \
        >"${TEMP_DIR}/server.log" 2>&1 &
    SERVER_PID=$!
    echo "[run_blackbox] spawned arcgraph serve at pid ${SERVER_PID}"

    # Trap to ensure server is killed on exit.
    trap 'kill ${SERVER_PID} 2>/dev/null || true; rm -rf "${TEMP_DIR}"' EXIT

    # Wait for server to come up (max 10s).
    for _ in $(seq 1 10); do
        if nc -z 127.0.0.1 17687 2>/dev/null; then
            break
        fi
        sleep 1
    done

    if ! nc -z 127.0.0.1 17687 2>/dev/null; then
        echo "[run_blackbox] FAIL: arcgraph server failed to start" >&2
        cat "${TEMP_DIR}/server.log" >&2
        FAIL=$((FAIL + 1))
    else
        echo "[run_blackbox] arcgraph server ready on 127.0.0.1:17687"
        export ARCGRAPH_BOLT_URI="bolt://127.0.0.1:17687"

        # Run smoke + adversarial suite.
        echo "[run_blackbox] running smoke.py (live)"
        if python3 "${PYTHON_DIR}/smoke.py" >&2; then
            PASS=$((PASS + 1))
        else
            echo "[run_blackbox] FAIL: smoke.py" >&2
            FAIL=$((FAIL + 1))
        fi

        echo "[run_blackbox] running test_bolt_v5_smoke.py (live)"
        if python3 "${PYTHON_DIR}/test_bolt_v5_smoke.py" >&2; then
            PASS=$((PASS + 1))
        else
            echo "[run_blackbox] FAIL: test_bolt_v5_smoke.py" >&2
            FAIL=$((FAIL + 1))
        fi
    fi

    # Tear down server.
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
    echo "[run_blackbox] server torn down"
fi

# ─── 5. Summary ──────────────────────────────────────────────────────

echo ""
echo "[run_blackbox] === SUMMARY ==="
echo "[run_blackbox] PASS:  ${PASS}"
echo "[run_blackbox] FAIL:  ${FAIL}"
echo "[run_blackbox] SKIP:  ${SKIP}"

if [ ${FAIL} -gt 0 ]; then
    echo "[run_blackbox] OVERALL: FAIL"
    exit 1
fi

echo "[run_blackbox] OVERALL: PASS"
exit 0
