#!/usr/bin/env python3
"""ingest_load_streaming.py — like ingest_load.py but STREAMS the ingest JSON.

Fixes the ~20GB python RSS from `json.load(whole 4.3GB file)` that OOM-killed the
10M ingest (combined with the mcp-stdio server's own ~21GB -> 40G cgroup OOM). This
version reads the top-level JSON array one element at a time (ijson) and feeds each
framed request to the mcp-stdio server synchronously, so the CLIENT stays tiny (~26MB)
and only the SERVER's real ingest RSS is measured. This is the isolation that proves
the ~39GB OOM (issue #1404) is the ArcGraph SERVER, not the harness.

Usage:
  ingest_load_streaming.py --bin <mcp-stdio> --data <dir> --ingest <big.json> \
      --out <result.json>
"""
import argparse
import json
import subprocess
import sys
import time

import ijson  # streaming JSON

sys.path.insert(0, "/home/ubuntu/ab-program")  # adjust to the harness dir on the host
from bench_mcp import frame, FramedReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--data")
    ap.add_argument("--ingest", required=True)
    ap.add_argument("--out", default="/tmp/ag_stream_ingest.json")
    args = ap.parse_args()

    cmd = [args.bin]
    if args.data:
        cmd += ["--data", args.data]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0)
    reader = FramedReader(proc.stdout)

    t0 = time.perf_counter()
    n_req, n_nodes, n_rels, errors = 0, 0, 0, 0
    with open(args.ingest, "rb") as f:
        for req in ijson.items(f, "item"):   # stream each top-level array element
            proc.stdin.write(frame(json.dumps(req).encode()))
            proc.stdin.flush()
            resp = reader.read_message()      # returns a parsed dict (matches ingest_load.py)
            n_req += 1
            try:
                if isinstance(resp, dict) and "error" in resp:
                    errors += 1
                params = req.get("params", {})
                n_nodes += len(params.get("nodes", []) or [])
                n_rels += len(params.get("relationships", []) or [])
            except Exception:
                errors += 1
            if n_req % 200 == 0:
                print(f"  {n_req} reqs, {n_nodes} nodes, {n_rels} rels, "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
    dt = time.perf_counter() - t0
    try:
        proc.stdin.close()
        proc.terminate()
    except Exception:
        pass

    out = {"engine": "arcgraph-stream", "requests": n_req, "nodes": n_nodes,
           "rels": n_rels, "secs": round(dt, 1),
           "nodes_per_s": round(n_nodes / dt) if dt else None,
           "errors": errors}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"DONE: {n_nodes} nodes + {n_rels} rels in {dt:.1f}s "
          f"= {n_nodes/dt:.0f} nodes/s, {errors} errors")


if __name__ == "__main__":
    main()
