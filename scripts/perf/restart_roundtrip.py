#!/usr/bin/env python3
"""Durability restart round-trip for the ArcGraph durable (`--data`) path.

Full cycle on the MCP-stdio surface:
  1. Boot process A on a fresh `--data <dir>`; ingest a small graph; verify the
     in-session counts (write-side oracle).
  2. Close A's stdin → clean process exit (commits fsync'd to the WAL per the
     Strict durability tier).
  3. Boot process B on the SAME `--data <dir>` → WAL recovery runs on startup.
  4. Probe B: does the data survive the restart? Report node count, relationship
     count, and label-qualified count separately — they recover differently.

Also `--probe-existing <dir>`: boot a fresh process on a pre-existing durable
dir and run the same recovery probe (used to characterize recovery at 10k scale
on the already-ingested benchmark dir).

Usage:
    restart_roundtrip.py --bin target/release/arcgraph-mcp-stdio --dir /tmp/ag_rt \
        --nodes 300 [--tenant 1]
    restart_roundtrip.py --bin ... --probe-existing /tmp/ag_perf_data --tenant 1
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_mcp import frame, FramedReader  # noqa: E402

PROBES = [
    ("count_nodes",        "MATCH (n) RETURN count(n)"),
    ("count_rels_anytype", "MATCH ()-[r]->() RETURN count(r)"),
    ("count_user_label",   "MATCH (n:User) RETURN count(n)"),
    ("count_knows_rel",    "MATCH ()-[r:KNOWS]->() RETURN count(r)"),
    ("node_with_props",    "MATCH (n {name:'user_1'}) RETURN n.age"),
]


def session(bin_path, dir_, requests):
    """Run a list of JSON-RPC requests synchronously in one process; return responses."""
    proc = subprocess.Popen([bin_path, "--data", dir_], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    rd = FramedReader(proc.stdout)
    resps = []
    for r in requests:
        proc.stdin.write(frame(json.dumps(r).encode()))
        proc.stdin.flush()
        resps.append(rd.read_message())
    proc.stdin.close()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    stderr = (proc.stderr.read() or b"").decode(errors="replace")
    return resps, proc.returncode, stderr


def body_of(resp):
    res = resp.get("result", {})
    if isinstance(res, dict) and "body" in res:
        try:
            return json.loads(res["body"])
        except Exception:
            return res["body"]
    return resp.get("error", res)


def first_cell(resp):
    b = body_of(resp)
    if isinstance(b, dict) and b.get("rows"):
        return b["rows"][0][0]
    if isinstance(b, dict) and "code" in b:
        return f"ERR:{b.get('message', b)[:60]}"
    return b


def probe_session(bin_path, dir_, tenant):
    reqs = [{"jsonrpc": "2.0", "id": i + 1, "method": "graph.raw_query",
             "params": {"tenant_id": tenant, "query": q}} for i, (_n, q) in enumerate(PROBES)]
    resps, code, stderr = session(bin_path, dir_, reqs)
    result = {}
    for (name, _q), resp in zip(PROBES, resps):
        result[name] = first_cell(resp)
    return result, code, stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--dir")
    ap.add_argument("--nodes", type=int, default=300)
    ap.add_argument("--tenant", type=int, default=1)
    ap.add_argument("--probe-existing")
    ap.add_argument("--out", default="/tmp/ag_restart_results.json")
    args = ap.parse_args()
    out = {}

    if args.probe_existing:
        print(f"=== probe-existing durable dir: {args.probe_existing} (restart recovery) ===")
        res, code, stderr = probe_session(args.bin, args.probe_existing, args.tenant)
        out["probe_existing"] = {"dir": args.probe_existing, "after_restart": res, "exit": code}
        for k, v in res.items():
            print(f"  {k:20} -> {v}")
        recov = [l for l in stderr.splitlines() if "recover" in l.lower() or "replay" in l.lower()
                 or "rebuild" in l.lower()]
        out["probe_existing"]["recovery_log"] = recov[-6:]
        print("  recovery log:", *recov[-4:], sep="\n    ")

    if args.dir:
        sys.path.insert(0, __file__.rsplit("/", 1)[0])
        from gen_dataset import build_requests
        os.system(f"rm -rf {args.dir} && mkdir -p {args.dir}")

        # PHASE 1: ingest in process A + in-session verify
        reqs, n_edges = build_requests(args.nodes, args.tenant)
        print(f"=== phase 1: ingest {args.nodes} nodes / {n_edges} edges (process A) ===")
        t0 = time.perf_counter()
        resps_a, code_a, _ = session(args.bin, args.dir, reqs)
        ingest_s = time.perf_counter() - t0
        in_session = {"count_user": first_cell(resps_a[-2]), "expand_count": first_cell(resps_a[-1])}
        print(f"  in-session (process A, pre-restart): {in_session}  ({ingest_s:.1f}s, exit={code_a})")

        # PHASE 2: process A already exited (clean) → reboot process B (recovery)
        print(f"=== phase 2: reboot (process B) on same dir → WAL recovery ===")
        res_b, code_b, stderr_b = probe_session(args.bin, args.dir, args.tenant)
        for k, v in res_b.items():
            print(f"  {k:20} -> {v}")
        out["roundtrip"] = {"dir": args.dir, "nodes": args.nodes, "edges": n_edges,
                            "ingest_s": round(ingest_s, 1),
                            "in_session_prerestart": in_session,
                            "after_restart": res_b, "exit_a": code_a, "exit_b": code_b}
        recov = [l for l in stderr_b.splitlines() if "recover" in l.lower() or "replay" in l.lower()
                 or "rebuild" in l.lower()]
        out["roundtrip"]["recovery_log"] = recov[-6:]

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
