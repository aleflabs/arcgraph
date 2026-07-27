#!/usr/bin/env python3
"""Durable bulk loader + write-path timer for the ArcGraph MCP-stdio surface.

Boots `arcgraph-mcp-stdio --data <dir>` (or `--in-memory`), streams a generated
`graph.ingest` request file SYNCHRONOUSLY (one framed request, drain one framed
response) so each ingest cohort is timed individually, and verifies the
post-ingest count queries against the dataset metadata (the write-side oracle).

Why synchronous (not the all-up-front mcp_driver): we want the per-cohort wall
time as the durable write-path number, and the all-up-front driver hard-codes a
60s communicate() timeout that the (fsync-bound) durable node path blows past.

Usage:
    ingest_load.py --bin target/release/arcgraph-mcp-stdio --data /tmp/ag_data \
        --ingest /tmp/ag_perf_ingest.json --meta /tmp/ag_perf_meta.json \
        --out /tmp/ag_ingest_timing.json
"""
import argparse
import json
import re
import subprocess
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_mcp import frame, FramedReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--data")
    ap.add_argument("--in-memory", action="store_true")
    ap.add_argument("--ingest", required=True)
    ap.add_argument("--meta")
    ap.add_argument("--out", default="/tmp/ag_ingest_timing.json")
    args = ap.parse_args()

    reqs = json.load(open(args.ingest))
    cmd = [args.bin] + (["--in-memory"] if args.in_memory else ["--data", args.data])
    print(f"boot: {' '.join(cmd)}  ({len(reqs)} requests)", flush=True)

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0)
    rd = FramedReader(proc.stdout)

    timing = []
    t_all = time.perf_counter()
    nodes_total = edges_total = 0
    for r in reqs:
        t0 = time.perf_counter()
        proc.stdin.write(frame(json.dumps(r).encode()))
        proc.stdin.flush()
        resp = rd.read_message()
        dt = (time.perf_counter() - t0) * 1000.0
        method = r["method"]
        info = {}
        res = resp.get("result", {})
        body = res.get("body", "") if isinstance(res, dict) else json.dumps(res)
        if method == "graph.ingest":
            m = re.search(r'"inserted_count":(\d+)', body)
            f = re.search(r'"failed_count":(\d+)', body)
            n_nodes = len(r["params"].get("nodes", []))
            n_rels = len(r["params"].get("relationships", []))
            nodes_total += n_nodes
            edges_total += n_rels
            info = {"inserted": int(m.group(1)) if m else None,
                    "failed": int(f.group(1)) if f else None,
                    "nodes_in_batch": n_nodes, "rels_in_batch": n_rels,
                    "ms_per_node": round(dt / n_nodes, 3) if n_nodes else None,
                    "ms_per_rel": round(dt / n_rels, 3) if n_rels else None}
        elif method == "graph.raw_query":
            info = {"body": body[:200]}
        timing.append({"method": method, "ms": round(dt, 2), **info})
        if method == "graph.ingest":
            note = f"inserted={info['inserted']} ({info['ms_per_node']} ms/node, {info['ms_per_rel']} ms/rel)"
        else:
            note = str(info)
        print(f"  {method:16} {dt:9.1f} ms  {note}", flush=True)

    total_ms = (time.perf_counter() - t_all) * 1000.0
    proc.stdin.close()
    try:
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
    stderr = (proc.stderr.read() or b"").decode(errors="replace")

    out = {"data": args.data, "in_memory": args.in_memory,
           "total_ms": round(total_ms, 1),
           "nodes_total": nodes_total, "edges_total": edges_total,
           "node_ingest_rps": round(nodes_total / (total_ms / 1000.0), 1) if total_ms else None,
           "timing": timing, "exit_code": proc.returncode,
           "stderr_tail": stderr[-600:]}
    if args.meta:
        out["meta"] = json.load(open(args.meta))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nTOTAL {total_ms/1000:.1f}s  nodes={nodes_total} edges={edges_total}  exit={proc.returncode}  -> {args.out}")


if __name__ == "__main__":
    main()
