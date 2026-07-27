#!/usr/bin/env python3
"""W26-γ-3 / ADR-136 — gRPC streaming + cancellation tests.

# Status (v1.0-α / W26-γ-3)

ArcGraph does NOT ship a native gRPC server at v1.0-α; the MCP
protocol surface is JSON-RPC over stdio/HTTP + Bolt 5.0 over TCP.

This file is a SCAFFOLD / PLACEHOLDER for the v1.1+ gRPC service
landing per ADR's forward-pin "M5 gRPC service ADR" (candidate slot
ADR-070..ADR-075 + ADR-077..ADR-078).

The tests below are written as `unittest.skip` so they:
  - Are checked-in to lock the test plan for v1.1.
  - Document the gRPC + streaming + cancellation invariants ArcGraph
    will need to pin.
  - Run as "SKIPPED" in the blackbox suite, so the runner produces
    a green report.

When the gRPC service lands (v1.1), this file's skip decorators are
removed and the test bodies wire up against the real surface.

Per `feedback_avoid_speculative_scaffolding.md`: the SCAFFOLD pattern
is acceptable when the consuming surface is forward-pinned via an
ADR + slot reservation; the scaffold documents intent without
shipping production code ahead of a consumer.
"""

from __future__ import annotations

import os
import sys
import unittest


GRPC_AVAILABLE = False
try:
    import grpc  # noqa: F401

    GRPC_AVAILABLE = True
except ImportError:
    pass


GRPC_TARGET = os.environ.get("ARCGRAPH_GRPC_TARGET", "localhost:50051")


@unittest.skipUnless(GRPC_AVAILABLE, "grpcio not installed")
@unittest.skip("gRPC service forward-pinned to v1.1 (ADR-070..078)")
class GrpcStreamingTests(unittest.TestCase):
    """Test plan for the v1.1+ gRPC streaming surface."""

    def test_unary_call_round_trip(self):
        """Unary call returns a structured response."""
        # When v1.1 lands: open channel, call SchemaInfo, assert response shape.
        raise NotImplementedError("v1.1 forward-pin")

    def test_server_streaming_round_trip(self):
        """Server-streaming RPC delivers N messages in order."""
        # When v1.1 lands: call GraphExplore server-stream, assert N messages.
        raise NotImplementedError("v1.1 forward-pin")

    def test_streaming_cancellation_clean_close(self):
        """Cancellation mid-stream closes the channel cleanly."""
        # When v1.1 lands: start GraphExplore stream, cancel after N messages,
        # assert no leaked server-side resources.
        raise NotImplementedError("v1.1 forward-pin")

    def test_goaway_recovery(self):
        """Server-initiated GOAWAY → client reconnects automatically."""
        # When v1.1 lands: trigger server graceful shutdown mid-stream,
        # assert client receives GOAWAY + reconnects + completes the
        # interrupted call on the new channel.
        raise NotImplementedError("v1.1 forward-pin")

    def test_deadline_exceeded(self):
        """Per-call deadline trips at the configured budget."""
        # When v1.1 lands: call with deadline=100ms, assert DEADLINE_EXCEEDED
        # status code.
        raise NotImplementedError("v1.1 forward-pin")


def main() -> int:
    """Run the test class — all tests will SKIP at v1.0-α (intended)."""
    # Suppress the noisy unittest output; print a summary line so the
    # blackbox runner sees the file as exercised.
    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stderr)
    suite = unittest.TestLoader().loadTestsFromTestCase(GrpcStreamingTests)
    result = runner.run(suite)
    print(
        f"[test_grpc_streaming] {result.testsRun} tests; {len(result.skipped)} skipped (v1.0-α forward-pin)",
        file=sys.stderr,
    )
    return 0 if (not result.failures and not result.errors) else 1


if __name__ == "__main__":
    sys.exit(main())
