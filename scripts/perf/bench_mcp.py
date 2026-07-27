#!/usr/bin/env python3
"""RPS / latency benchmark for the ArcGraph MCP-stdio surface.

`arcgraph-mcp-stdio` is NOT a concurrent network daemon — it is a single
process speaking Content-Length-framed JSON-RPC over one stdin/stdout pipe,
keeping ONE shared storage backend for its lifetime. So the meaningful
measurements are:

  1. Synchronous round-trip latency (depth=1): write one framed request, flush,
     read exactly one framed response, timing each. This is the per-query wire
     latency a sequential in-process client sees. P50/P95/P99/max.
  2. Pipelined throughput at increasing in-flight depth D ∈ {1,8,64,512}: write
     D framed requests, then drain D framed responses; repeat. Throughput =
     completed / wall. This is the stdio analog of "concurrency" — how far the
     single request/response loop can be filled before the server saturates.

Boots the binary against a durable `--data <dir>` so reads hit the real
WAL/page substrate (the same dir the dataset was ingested into).

Usage:
    bench_mcp.py --bin target/release/arcgraph-mcp-stdio --data /tmp/ag_data \
        --tenant 1 --duration 20 --depths 1,8,64,512 --out /tmp/ag_mcp_results.json
"""
import argparse
import json
import statistics
import subprocess
import time

# Same bounded read workload as the Bolt bench (MCP raw_query shape).
WORKLOAD = [
    ("count_user",      "MATCH (n:User) RETURN count(n)"),
    ("count_all",       "MATCH (n) RETURN count(n)"),
    ("expand_count",    "MATCH (a:User)-[:KNOWS]->(b:User) RETURN count(b)"),
    ("point_lookup",    "MATCH (n:User {name:'user_1000'}) RETURN n.age"),
    ("anchored_expand", "MATCH (a:User {name:'user_100'})-[:KNOWS]->(b) RETURN b.name"),
    ("compute_floor",   "RETURN 1 IN [1,2,3]"),
]


def frame(payload: bytes) -> bytes:
    return f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload


def make_req(rid, tenant, query):
    return {"jsonrpc": "2.0", "id": rid, "method": "graph.raw_query",
            "params": {"tenant_id": tenant, "query": query}}


class FramedReader:
    """Reads Content-Length-framed messages from a binary stream incrementally."""

    def __init__(self, stream):
        self.s = stream
        self.buf = b""

    def _fill_to(self, n):
        while len(self.buf) < n:
            chunk = self.s.read(max(n - len(self.buf), 4096))
            if not chunk:
                raise EOFError("stream closed mid-frame")
            self.buf += chunk

    def read_message(self):
        # read header up to \r\n\r\n
        while b"\r\n\r\n" not in self.buf:
            chunk = self.s.read(4096)
            if not chunk:
                raise EOFError("stream closed before header")
            self.buf += chunk
        term = self.buf.find(b"\r\n\r\n")
        header = self.buf[:term].decode("ascii", errors="replace")
        clen = None
        for line in header.split("\r\n"):
            if line.lower().startswith("content-length:"):
                clen = int(line.split(":", 1)[1].strip())
        if clen is None:
            raise ValueError(f"no content-length in header: {header!r}")
        body_start = term + 4
        self._fill_to(body_start + clen)
        body = self.buf[body_start:body_start + clen]
        self.buf = self.buf[body_start + clen:]
        return json.loads(body.decode("utf-8"))


def pctl(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _r(x):
    return round(x, 3) if x is not None else None


def boot(bin_path, data_dir, in_memory=False):
    cmd = [bin_path] + (["--in-memory"] if in_memory else ["--data", data_dir])
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0)
    return proc, FramedReader(proc.stdout)


def ingest_in_session(proc, reader, ingest_file):
    """Stream a graph.ingest request file in THIS session (keeps the catalog live
    — avoids the WAL-recovery catalog-loss path). Returns post-ingest diagnostics."""
    reqs = json.load(open(ingest_file))
    diag = {}
    for r in reqs:
        proc.stdin.write(frame(json.dumps(r).encode()))
        proc.stdin.flush()
        resp = reader.read_message()
        if r["method"] == "graph.raw_query":
            body = resp.get("result", {}).get("body", "")
            diag[r["params"]["query"][:40]] = body[:120]
    return diag


def sync_latency(proc, reader, tenant, duration, warmup):
    """depth=1 synchronous round-trip latency over the workload."""
    seq = WORKLOAD
    rid = 0
    # warmup
    end = time.monotonic() + warmup
    while time.monotonic() < end:
        name, q = seq[rid % len(seq)]
        rid += 1
        proc.stdin.write(frame(json.dumps(make_req(rid, tenant, q)).encode()))
        proc.stdin.flush()
        reader.read_message()
    # measure
    lat_by_type, lat_all = {n: [] for (n, _q) in seq}, []
    t_start = time.monotonic()
    end = t_start + duration
    while time.monotonic() < end:
        name, q = seq[rid % len(seq)]
        rid += 1
        t0 = time.perf_counter()
        proc.stdin.write(frame(json.dumps(make_req(rid, tenant, q)).encode()))
        proc.stdin.flush()
        reader.read_message()
        dt = (time.perf_counter() - t0) * 1000.0
        lat_by_type[name].append(dt)
        lat_all.append(dt)
    wall = time.monotonic() - t_start
    return {
        "completed": len(lat_all),
        "duration_s": round(wall, 3),
        "rps_sync": round(len(lat_all) / wall, 1) if wall > 0 else 0.0,
        "lat_ms": {"p50": _r(pctl(lat_all, 50)), "p95": _r(pctl(lat_all, 95)),
                   "p99": _r(pctl(lat_all, 99)), "max": _r(max(lat_all) if lat_all else None),
                   "mean": _r(statistics.fmean(lat_all) if lat_all else None)},
        "by_type": {k: {"n": len(v), "p50": _r(pctl(v, 50)), "p99": _r(pctl(v, 99))}
                    for k, v in sorted(lat_by_type.items())},
    }


def pipelined_throughput(bin_path, data, in_memory, ingest_file, tenant, depth, n):
    """Pipelining probe in a FRESH session (avoids any shared-buffer state):
    write `n` requests up front, then drain `n` responses, verifying each is a
    real (non-error) result. Throughput = n / drain_wall. Compared against the
    sync rate to test whether the serial stdio dispatch yields any pipelining
    speedup (it does not — single shared backend, in-order processing)."""
    proc, reader = boot(bin_path, data, in_memory)
    if ingest_file:
        ingest_in_session(proc, reader, ingest_file)
    seq = WORKLOAD
    # write n requests (chunked writes so we never out-write the OS pipe + then
    # block before any drain — interleave a partial drain every `depth`).
    real = err = 0
    t_start = time.monotonic()
    inflight = 0
    sent = 0
    while sent < n:
        burst = min(depth, n - sent)
        for _ in range(burst):
            sent += 1
            _name, q = seq[sent % len(seq)]
            proc.stdin.write(frame(json.dumps(make_req(sent, tenant, q)).encode()))
        proc.stdin.flush()
        inflight += burst
        # drain down to keep in-flight <= depth (bounded so pipes never deadlock)
        while inflight > 0:
            resp = reader.read_message()
            inflight -= 1
            if resp.get("error"):
                err += 1
            else:
                real += 1
    wall = time.monotonic() - t_start
    proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    return {"depth": depth, "n": n, "real": real, "err": err,
            "duration_s": round(wall, 3),
            "rps_pipelined": round(n / wall, 1) if wall > 0 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--data", default="")
    ap.add_argument("--in-memory", action="store_true")
    ap.add_argument("--ingest-file", default="",
                    help="ingest this graph.ingest request file IN-SESSION before benching "
                         "(keeps the label/rel-type catalog live; avoids WAL-recovery loss)")
    ap.add_argument("--tenant", type=int, default=1)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=4.0)
    ap.add_argument("--depths", default="1,8,64")
    ap.add_argument("--pipe-n", type=int, default=300,
                    help="total queries per pipelining-probe session")
    ap.add_argument("--out", default="/tmp/ag_mcp_results.json")
    args = ap.parse_args()

    results = {"data": args.data, "in_memory": args.in_memory, "tenant": args.tenant,
               "ingest_file": args.ingest_file, "workload": WORKLOAD, "duration_s": args.duration}

    # 1. synchronous latency
    proc, reader = boot(args.bin, args.data, args.in_memory)
    try:
        if args.ingest_file:
            results["ingest_diag"] = ingest_in_session(proc, reader, args.ingest_file)
            print(f"  in-session ingest diag: {results['ingest_diag']}", flush=True)
        results["sync"] = sync_latency(proc, reader, args.tenant, args.duration, args.warmup)
        print(f"  sync depth=1  RPS={results['sync']['rps_sync']:>9.1f}  "
              f"P50={results['sync']['lat_ms']['p50']}  P95={results['sync']['lat_ms']['p95']}  "
              f"P99={results['sync']['lat_ms']['p99']}  max={results['sync']['lat_ms']['max']}", flush=True)
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        results["stderr_tail"] = (proc.stderr.read() or b"").decode(errors="replace")[-800:]

    # 2. pipelining probe — FRESH session per depth (no shared-buffer state);
    #    tests whether depth-deep pipelining beats the serial sync rate.
    results["pipelined"] = []
    n = args.pipe_n
    for d in [int(x) for x in args.depths.split(",")]:
        r = pipelined_throughput(args.bin, args.data, args.in_memory, args.ingest_file,
                                 args.tenant, d, n)
        results["pipelined"].append(r)
        print(f"  pipe depth={d:>4}  RPS={r['rps_pipelined']:>9.1f}  "
              f"real={r['real']} err={r['err']}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}  (exit={proc.returncode})")


if __name__ == "__main__":
    main()
