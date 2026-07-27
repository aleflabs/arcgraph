#!/usr/bin/env python3
"""#765/#779 MCP-stdio vector-serve SMOKE — does graph.search KNN SERVE in-session?

The directive's STEP 2: confirm a single `graph.search` KNN query returns ranked
results (not -32004 IndexUnavailable) now that #765 PART-1 / #779 landed. Boots
the real arcgraph-mcp-stdio binary, ingests a handful of Doc nodes carrying the
`embedding` property via graph.ingest (in-session, live catalog), then:

  [A] graph.search {query_vec, k}                 -> the dedicated KNN tool
  [B] graph.raw_query MATCH .. RANK BY HYBRID(VECTOR(..)) -> the ArcQL surface

so we see EXACTLY which served surfaces work, verbatim.

Usage:
    vector_smoke_mcp.py --bin target/release/arcgraph-mcp-stdio --dim 8
"""
import argparse
import json
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_mcp import frame, FramedReader  # noqa: E402


def rpc(proc, reader, rid, method, params):
    req = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
    proc.stdin.write(frame(json.dumps(req).encode()))
    proc.stdin.flush()
    return reader.read_message()


def vec_literal(v):
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--tenant", type=int, default=1)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    proc = subprocess.Popen([args.bin, "--in-memory"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    reader = FramedReader(proc.stdout)
    d = args.dim
    base = [
        ("doc-0", [0.0] * d),
        ("doc-1", [1.0] + [0.0] * (d - 1)),
        ("doc-2", [0.0, 1.0] + [0.0] * (d - 2)),
        ("doc-3", [10.0] * d),
        ("doc-4", [5.0] * d),
    ]
    try:
        # ingest in-session
        nodes = [{"external_id": ext, "label": "Doc",
                  "properties": {"embedding": v}} for (ext, v) in base]
        r = rpc(proc, reader, 1, "graph.ingest",
                {"tenant_id": args.tenant, "nodes": nodes, "relationships": []})
        print(f"[ingest] {json.dumps(r.get('result') or r.get('error'))[:200]}")

        # [A] graph.search (the dedicated KNN tool) — query near origin -> doc-0 #1
        qv = [0.1] * d
        r = rpc(proc, reader, 2, "graph.search",
                {"tenant_id": args.tenant, "query": "", "query_vec": qv, "k": args.k,
                 "format": "json"})
        if r.get("error"):
            print(f"[A] graph.search ERROR: {json.dumps(r['error'])[:300]}")
        else:
            res = r["result"]
            print(f"[A] graph.search RAW result: {json.dumps(res)[:500]}")
            body = res.get("body") if isinstance(res, dict) else None
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except Exception:
                    pass
            if isinstance(body, dict) and "hits" in body:
                hits = body["hits"]
                print(f"[A] graph.search SERVED: hits={len(hits)}")
                for h in hits:
                    print(f"      {h}")

        # [B] graph.raw_query ArcQL RANK BY HYBRID(VECTOR(..)) — inlined vector literal
        cy = (f"MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, {vec_literal(qv)}, K = {args.k})) "
              f"WITH FUSION = RRF(k = 60) RETURN n")
        r = rpc(proc, reader, 3, "graph.raw_query", {"tenant_id": args.tenant, "query": cy})
        if r.get("error"):
            print(f"[B] graph.raw_query RANK BY ERROR: {json.dumps(r['error'])[:300]}")
        else:
            print(f"[B] graph.raw_query RANK BY SERVED: {json.dumps(r['result'])[:300]}")
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        tail = (proc.stderr.read() or b"").decode(errors="replace")[-400:]
        if tail.strip():
            print(f"[stderr tail] {tail}")


if __name__ == "__main__":
    main()
