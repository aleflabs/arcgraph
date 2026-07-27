#!/usr/bin/env python3
"""Reusable MCP stdio driver for arcgraph-mcp-stdio.

Drives the ArcGraph MCP server (Content-Length-framed JSON-RPC over
stdin/stdout) through a SINGLE session: writes all requests up-front,
closes stdin (clean PeerClosed shutdown), reads every framed response,
and reports per-call wall-clock latency.

Because the server keeps ONE shared storage backend for the process
lifetime, all requests in a batch share state — so a graph.ingest
followed by graph.raw_query MATCH in the same batch sees the ingested
data.

Usage:
    mcp_driver.py <binary> [--flag ...] -- <requests.json>

  <requests.json> is a JSON array of JSON-RPC request objects.
Prints a JSON report to stdout:
    {"responses": [...], "latencies_ms": [...], "exit_code": N, "stderr": "..."}
"""
import json
import subprocess
import sys
import time


def frame(payload_bytes: bytes) -> bytes:
    header = f"Content-Length: {len(payload_bytes)}\r\n\r\n".encode("ascii")
    return header + payload_bytes


def read_framed(buf: bytes):
    """Yield (json_value) for each Content-Length-framed message in buf."""
    i = 0
    n = len(buf)
    while i < n:
        # find header terminator
        term = buf.find(b"\r\n\r\n", i)
        if term == -1:
            break
        header = buf[i:term].decode("ascii", errors="replace")
        clen = None
        for line in header.split("\r\n"):
            if line.lower().startswith("content-length:"):
                clen = int(line.split(":", 1)[1].strip())
        if clen is None:
            break
        body_start = term + 4
        body = buf[body_start:body_start + clen]
        if len(body) < clen:
            break
        yield json.loads(body.decode("utf-8"))
        i = body_start + clen


def main():
    args = sys.argv[1:]
    if "--" not in args:
        print("usage: mcp_driver.py <binary> [flags...] -- <requests.json>", file=sys.stderr)
        sys.exit(2)
    split = args.index("--")
    cmd = args[:split]
    requests_file = args[split + 1]

    with open(requests_file) as f:
        requests = json.load(f)

    # Build the full framed input stream (all requests up front).
    input_bytes = b""
    for req in requests:
        input_bytes += frame(json.dumps(req).encode("utf-8"))

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = proc.communicate(input=input_bytes, timeout=60)
    total_ms = (time.monotonic() - t0) * 1000.0

    responses = list(read_framed(out))

    report = {
        "n_requests": len(requests),
        "n_responses": len(responses),
        "responses": responses,
        "total_wall_ms": round(total_ms, 2),
        "exit_code": proc.returncode,
        "stderr": err.decode("utf-8", errors="replace"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
