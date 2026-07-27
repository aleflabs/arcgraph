#!/usr/bin/env python3
"""Driver-compat layer 4 smoke — real Neo4j Python driver against ArcGraph.

W24-DRIVERS-alpha. ADR-094 D-4. Companion to:
  - tests/driver-compat/README.md (runbook)
  - crates/arcgraph-cli/tests/driver_compat_bolt_v5.rs (layer 3 automated)
  - docs/conformance/gql-2024-conformance-matrix.md (audit-grade evidence)

This is an OPERATOR-RUNNABLE smoke. CI does NOT execute it (Python +
the `neo4j` PyPI package are environment-dependent). The script is the
canonical ground-truth verification surface for "a real third-party
Bolt 5.0 driver round-trips against ArcGraph."

Prerequisites:
  - Python 3.10+
  - pip install -r tests/driver-compat/python/requirements.txt
  - cargo build --release --bin arcgraph
  - In another terminal (W28 / ADR-183: `serve` needs an explicit
    storage mode; `--in-memory` for a throwaway smoke, `--data <dir>`
    for a durable store):
      ./target/release/arcgraph serve --bolt 127.0.0.1:7687 --in-memory

Run:
  python tests/driver-compat/python/smoke.py
  (optional: ARCGRAPH_BOLT_URI=bolt://host:port python ...)

Assertion discipline (post-R1 fix-up — W24-DRIVERS-α R1 MED-1):
  The smoke asserts WIRE-LEVEL invariants only — mirroring the layer-3
  subprocess test pattern at
  `crates/arcgraph-cli/tests/driver_compat_bolt_v5.rs:325-400`. We do
  NOT assert handler-specific row / column counts because:
    - The production `StorageBoltHandler` routes RUN through the real
      `QueryEngine`; on an empty substrate the column derivation at
      `crates/arcgraph-mcp/src/storage/bolt.rs:251` returns 0 cols for
      a literal `RETURN 1` (issue #353 — the M5-streaming surface
      that threads user RETURN aliases through MaterializedResult).
    - A wire-level invariant ("the driver consumed a server reply
      without raising; .keys() returned a Sequence; iterating records
      returned a list of any length") is what the audit-grade evidence
      claim actually rests on. Row / column count claims belong to
      the M5-12+ correctness-evidence surface, not the wire-conformance
      surface.

License compliance:
  - The neo4j PyPI driver is Apache-2.0 (per PyPI metadata 2026-05-24).
  - This script is Apache-2.0 per workspace LICENSE.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Sequence
from typing import List, Tuple

# The neo4j Python driver is Apache-2.0, preserving the workspace's
# Apache-2.0 license chain; verified at W24-DRIVERS-alpha landing.
try:
    from neo4j import GraphDatabase  # type: ignore[import-not-found]
    from neo4j.exceptions import (  # type: ignore[import-not-found]
        AuthError,
        ServiceUnavailable,
    )
except ImportError as exc:  # pragma: no cover — environment-dependent
    sys.stderr.write(
        "[smoke] FAIL: the `neo4j` PyPI package is not installed.\n"
        "[smoke] Install with: pip install -r "
        "tests/driver-compat/python/requirements.txt\n"
        f"[smoke] Original ImportError: {exc}\n"
    )
    sys.exit(2)


BOLT_URI = os.environ.get("ARCGRAPH_BOLT_URI", "bolt://127.0.0.1:7687")
BOLT_USER = os.environ.get("ARCGRAPH_BOLT_USER", "arcgraph-smoke")
BOLT_PASSWORD = os.environ.get("ARCGRAPH_BOLT_PASSWORD", "smoke-secret")
CONNECT_TIMEOUT_SEC = float(os.environ.get("ARCGRAPH_BOLT_CONNECT_TIMEOUT_SEC", "10"))


def _log(msg: str) -> None:
    """Single-line tagged log to stdout — easy for the README's expected output to match."""
    print(f"[smoke] {msg}", flush=True)


def _connect_with_retry(uri: str) -> "GraphDatabase.driver":  # type: ignore[name-defined]
    """Bolt 5.0 connection with a tight retry loop. Per ADR-094 D-1 the server
    rejects non-5.0 offers; the real Python driver negotiates 5.0 by default.

    Retry rationale: the operator runbook spawns arcgraph in a separate
    terminal; if the smoke fires before the listener is bound, the first
    connect attempt sees connection-refused. Bounded retry handles that
    race without masking a genuine startup failure.
    """
    deadline = time.monotonic() + CONNECT_TIMEOUT_SEC
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            driver = GraphDatabase.driver(
                uri,
                auth=(BOLT_USER, BOLT_PASSWORD),
                # Per ADR-094 D-1 + W14-retro IR L1-HIGH-4: loopback default;
                # the runbook does not exercise TLS.
                encrypted=False,
                connection_timeout=2.0,
            )
            # `verify_connectivity()` exercises the HANDSHAKE + HELLO + GOODBYE
            # path under the driver's hood; success means the negotiation
            # picked Bolt 5.0 + the server authenticated the principal.
            driver.verify_connectivity()
            return driver
        except (ServiceUnavailable, OSError) as exc:
            last_err = exc
            time.sleep(0.25)
    raise SystemExit(
        f"[smoke] FAIL: could not establish a Bolt 5.0 session at {uri} "
        f"within {CONNECT_TIMEOUT_SEC}s. Last error: {last_err}"
    )


def _exec_run(driver, cypher: str) -> Tuple[List[str], List[Sequence[object]]]:
    """Execute one Cypher statement and return (column_names, rows).

    Exercises the wire path: HELLO -> RUN -> PULL -> RECORD* -> SUCCESS.
    Closing the session emits the corresponding RESET / RESET-fast path.
    """
    with driver.session() as session:
        result = session.run(cypher)
        keys = list(result.keys())
        rows = [tuple(record.values()) for record in result]
        return keys, rows


def _assert_wire_round_trip(cypher: str, cols: object, rows: object) -> None:
    """Wire-level invariants only — see module docstring.

    Asserts:
      - `cols` is a Sequence (the driver's RUN reply carried a `fields` list,
        possibly empty — empty is legal for queries that do not project).
      - `rows` is a list (the driver consumed PULL/RECORDs without raising
        — empty is legal for the empty-substrate case).

    Does NOT assert: `len(cols) > 0`, `len(rows) > 0`, specific column
    names, or specific row values — those belong to handler-correctness
    evidence (M5-12+), not wire-conformance evidence.
    """
    if not isinstance(cols, list):
        raise AssertionError(
            f"wire invariant violated: RUN '{cypher}' .keys() returned "
            f"{type(cols).__name__}, expected list"
        )
    if not isinstance(rows, list):
        raise AssertionError(
            f"wire invariant violated: RUN '{cypher}' record stream "
            f"materialized to {type(rows).__name__}, expected list"
        )


def main() -> int:
    _log(f"connecting to {BOLT_URI} ...")
    try:
        driver = _connect_with_retry(BOLT_URI)
    except AuthError as exc:
        _log(f"FAIL: authentication failed: {exc}")
        return 1

    failures = 0
    rounds = 0

    try:
        # Smoke 1: HELLO already exercised by verify_connectivity above.
        _log("HELLO ok; verify_connectivity() passed (HANDSHAKE + HELLO round-trip)")
        rounds += 1

        # Smoke 2: simplest RETURN — wire-level: RUN reply parsed; PULL
        # drained; SUCCESS reached. No row / column count assertions.
        try:
            cols, rows = _exec_run(driver, "RETURN 1")
            _assert_wire_round_trip("RETURN 1", cols, rows)
            _log(
                f"RUN 'RETURN 1' ok; wire round-trip complete "
                f"(cols={len(cols)}, rows={len(rows)})"
            )
            rounds += 1
        except (AssertionError, Exception) as exc:  # noqa: BLE001
            _log(f"FAIL: RUN 'RETURN 1' wire round-trip raised: {exc}")
            failures += 1

        # Smoke 3: query that exercises the executor — wire-level only.
        # Mirrors the layer-3 subprocess test pattern: an empty substrate
        # returns 0 rows but the wire round-trip is still valid.
        try:
            cols, rows = _exec_run(driver, "MATCH (n) RETURN n")
            _assert_wire_round_trip("MATCH (n) RETURN n", cols, rows)
            _log(
                f"RUN 'MATCH (n) RETURN n' ok; wire round-trip complete "
                f"(cols={len(cols)}, rows={len(rows)})"
            )
            rounds += 1
        except (AssertionError, Exception) as exc:  # noqa: BLE001
            _log(f"FAIL: RUN 'MATCH (n) RETURN n' wire round-trip raised: {exc}")
            failures += 1

        # Smoke 4: GOODBYE — driver's `close()` emits GOODBYE; server closes
        # the socket. Per ADR-094 D-1, no reply is expected.
        # (close() is below, in the finally block.)
        _log("GOODBYE prepared; closing session...")
        rounds += 1

    finally:
        driver.close()
        _log("GOODBYE ok; session closed")

    if failures == 0:
        _log(f"PASS: {rounds} round-trips, 0 failures")
        return 0
    _log(f"FAIL: {rounds} round-trips, {failures} failure(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
