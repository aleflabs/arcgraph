#!/usr/bin/env python3
"""Vector KNN harness over the served `graph.search` MCP surface (post #765/#779).

#765 PART-1 / #779 bound the served `HnswVectorSearchProvider`. EMPIRICALLY at
this tip (verify, don't trust the commit message):

  * `graph.search` (the dedicated MCP KNN tool)  -> SERVES ranked KNN. ✓
  * ArcQL `RANK BY HYBRID(VECTOR(..))` (graph.raw_query AND Bolt) -> REJECTED at
    bind time: `-32005 cross-substrate error: substrate 'vector' not attached`
    — the semantic catalog `build_catalog_for_tenant` (adapters.rs) never sets
    `has_vector_index`, so `CrossSubstrateValidator` fails the gate. (FINDING; the
    un-throttled concurrent surface for KNN RPS is therefore blocked — route to
    mgr-vector.)  This harness PROBES both so the gap is recorded verbatim.

So the served KNN surface is `graph.search` over MCP-stdio, which is SERIAL +
read-rate-limited (OpClass::Read: 100-token burst, ~1.667/s refill — finding G5).
The honest, measurable vector numbers on this surface:

  * recall@k vs an EXACT brute-force L2 oracle (numpy) over the same vectors.
  * single-client KNN latency P50/P95/P99 (within the read burst).
  * cold (first query == lazy HNSW build over all N nodes) vs warm latency.
  * burst throughput: KNN queries served before the rate limiter trips, and the
    RPS over that burst — the practical agent-facing ceiling.

A low recall or a low burst-RPS is the finding. The vectors are deterministic
seeded numpy Gaussians, L2-normalized; the engine casts the stored f64 property
-> f32, and the oracle runs on the identical float32 matrix.

Usage:
    bench_vector.py --bin target/release/arcgraph-mcp-stdio --in-memory \
        --n 5000 --dim 128 --k 10 --queries 90 --out /tmp/ag_vector.json
"""
import argparse
import json
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_mcp import frame, FramedReader, pctl, _r  # noqa: E402

# graph.ingest enforces MAX_MESSAGE_BYTES (16 MiB). Each node carries `dim`
# floats (~18 bytes each as JSON). Size the chunk to stay well under the cap
# (~6 MB target) so a high-dim (768) embedding batch never overflows.
def chunk_for_dim(dim):
    return max(1, min(2000, 6_000_000 // max(1, dim * 18)))


def gen_matrix(n, dim, seed):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


def boot(bin_path, data, in_memory):
    cmd = [bin_path] + (["--in-memory"] if in_memory else ["--data", data])
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0)
    return proc, FramedReader(proc.stdout)


def rpc(proc, reader, rid, method, params):
    req = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
    proc.stdin.write(frame(json.dumps(req).encode()))
    proc.stdin.flush()
    return reader.read_message()


def ingest(proc, reader, tenant, mat, rid0):
    """Ingest N Doc nodes (embedding property) in-session, chunked. Returns
    (internal_id -> vector_index) map drawn from the real Inserted outcomes."""
    n = mat.shape[0]
    rid = rid0
    id_map = {}
    chunk = chunk_for_dim(mat.shape[1])
    for start in range(0, n, chunk):
        nodes = []
        for i in range(start, min(start + chunk, n)):
            nodes.append({"external_id": f"v{i}", "label": "Doc",
                          "properties": {"embedding": [float(x) for x in mat[i]]}})
        # graph.ingest is OpClass::Write (cap 10, refill 0.167/s); many chunks
        # (high-dim small chunks) trip the write limiter — retry through -32007
        # backoff (ingest timing is irrelevant to the KNN measurement).
        r = None
        for _attempt in range(60):
            rid += 1
            r = rpc(proc, reader, rid, "graph.ingest",
                    {"tenant_id": tenant, "nodes": nodes, "relationships": []})
            err = r.get("error")
            if err and err.get("code") == -32007:
                ra = (err.get("data") or {})
                time.sleep(max(ra.get("retry_after_ms", 1200) / 1000.0
                               if isinstance(ra, dict) else 1.2, 0.5))
                continue
            break
        if r.get("error") or "result" not in r:
            raise RuntimeError(f"graph.ingest failed at chunk start={start} "
                               f"(chunk={chunk}, dim={mat.shape[1]}): {json.dumps(r.get('error'))[:200]}")
        body = r["result"].get("body")
        if isinstance(body, str):
            body = json.loads(body)
        for rec in body.get("records", []):
            if rec.get("status") == "inserted":
                ext = rec["external_id"]
                id_map[int(rec["internal_id"])] = int(ext[1:])  # "v123" -> 123
    return id_map, rid


def search(proc, reader, rid, tenant, qv, k):
    return rpc(proc, reader, rid, "graph.search",
               {"tenant_id": tenant, "query": "", "query_vec": [float(x) for x in qv],
                "k": k, "format": "json"})


def parse_hits(resp):
    """-> (hits|None, error|None). hits = list of {node_id,label,score}."""
    if resp.get("error"):
        return None, resp["error"]
    body = resp.get("result", {}).get("body")
    if isinstance(body, str):
        body = json.loads(body)
    return body.get("hits", []), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--data", default="")
    ap.add_argument("--in-memory", action="store_true")
    ap.add_argument("--tenant", type=int, default=1)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=765)
    ap.add_argument("--queries", type=int, default=90,
                    help="recall queries (keep <~95 to stay within the read burst)")
    ap.add_argument("--out", default="/tmp/ag_vector.json")
    args = ap.parse_args()

    print(f"gen {args.n}x{args.dim} vectors (seed={args.seed})...", flush=True)
    mat = gen_matrix(args.n, args.dim, args.seed)
    qrng = np.random.default_rng(args.seed + 1)
    qraw = qrng.standard_normal((args.queries, args.dim)).astype(np.float32)
    qraw /= np.linalg.norm(qraw, axis=1, keepdims=True)

    out = {"n": args.n, "dim": args.dim, "k": args.k, "seed": args.seed,
           "surface": "graph.search (MCP-stdio)", "tenant": args.tenant,
           "in_memory": args.in_memory}

    proc, reader = boot(args.bin, args.data, args.in_memory)
    try:
        # ── ingest
        t0 = time.perf_counter()
        id_map, rid = ingest(proc, reader, args.tenant, mat, rid0=0)
        out["ingest"] = {"requested": args.n, "inserted": len(id_map),
                         "wall_s": round(time.perf_counter() - t0, 2)}
        print(f"  ingest: {out['ingest']}", flush=True)

        # ── PROBE A: graph.search serves?  PROBE B: raw_query RANK BY gate.
        qv0 = mat[0]  # query == doc-0 -> its own internal id must rank #1
        rid += 1
        t = time.perf_counter()
        respA = search(proc, reader, rid, args.tenant, qv0, args.k)
        cold_ms = (time.perf_counter() - t) * 1000.0  # FIRST query == lazy HNSW build
        hitsA, errA = parse_hits(respA)
        served = hitsA is not None and len(hitsA) > 0
        top_internal = hitsA[0]["node_id"] if served else None
        out["probe_graph_search"] = {
            "served": served, "rows": len(hitsA) if hitsA else 0,
            "cold_first_query_ms": round(cold_ms, 2),
            "rank1_is_query_node": served and id_map.get(int(top_internal)) == 0,
            "error": errA}
        print(f"  [A] graph.search served={served} rows={len(hitsA) if hitsA else 0} "
              f"cold(build+query)={cold_ms:.1f}ms rank1_vid={id_map.get(int(top_internal)) if served else None}",
              flush=True)

        rank = (f"MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, "
                f"[{','.join(repr(float(x)) for x in qv0)}], K = {args.k})) "
                f"WITH FUSION = RRF(k = 60) RETURN n")
        rid += 1
        respB = rpc(proc, reader, rid, "graph.raw_query", {"tenant_id": args.tenant, "query": rank})
        out["probe_arcql_rankby"] = {
            "served": respB.get("error") is None,
            "error": respB.get("error")}
        print(f"  [B] graph.raw_query RANK BY served={respB.get('error') is None} "
              f"err={json.dumps(respB.get('error'))[:160] if respB.get('error') else None}", flush=True)

        if not served:
            print("  -> graph.search did NOT serve; stopping vector run (finding).", flush=True)
            out["stopped"] = "graph.search did not serve"
            json.dump(out, open(args.out, "w"), indent=2)
            return

        # ── warm latency + burst throughput: drive KNN back-to-back until the
        #    read limiter trips (-32007). Measures the practical agent ceiling.
        lat, rejected_at, served_ok = [], None, 0
        t_burst = time.perf_counter()
        for i in range(200):
            rid += 1
            qv = qraw[i % args.queries]
            t = time.perf_counter()
            resp = search(proc, reader, rid, args.tenant, qv, args.k)
            dt = (time.perf_counter() - t) * 1000.0
            err = resp.get("error")
            if err and err.get("code") == -32007:
                rejected_at = i
                break
            if err:
                break
            served_ok += 1
            lat.append(dt)
        burst_wall = time.perf_counter() - t_burst
        out["burst"] = {
            "served_before_throttle": served_ok,
            "rejected_at_query": rejected_at,
            "burst_rps": round(served_ok / burst_wall, 1) if burst_wall > 0 else 0.0,
            "warm_lat_ms": {"p50": _r(pctl(lat, 50)), "p95": _r(pctl(lat, 95)),
                            "p99": _r(pctl(lat, 99)), "max": _r(max(lat) if lat else None),
                            "mean": _r(statistics.fmean(lat) if lat else None)}}
        print(f"  burst: served {served_ok} KNN before throttle (rejected_at={rejected_at}); "
              f"burst_rps={out['burst']['burst_rps']}  warm P50={out['burst']['warm_lat_ms']['p50']} "
              f"P95={out['burst']['warm_lat_ms']['p95']} P99={out['burst']['warm_lat_ms']['p99']}", flush=True)

        # ── recall@k vs exact brute-force oracle. Handle -32007 with backoff so
        #    recall (timing-insensitive) can use all `--queries` queries.
        print(f"  recall@{args.k} over {args.queries} queries vs brute-force oracle...", flush=True)
        recalls, missed = [], []
        for qi in range(args.queries):
            qv = qraw[qi]
            d = np.sum((mat - qv) ** 2, axis=1)
            exact = set(int(x) for x in np.argsort(d)[:args.k])
            got = None
            for _attempt in range(40):
                rid += 1
                resp = search(proc, reader, rid, args.tenant, qv, args.k)
                err = resp.get("error")
                if err and err.get("code") == -32007:
                    time.sleep(max((err.get("data", {}) or {}).get("retry_after_ms", 600) / 1000.0
                                   if isinstance(err.get("data"), dict) else 0.6, 0.6))
                    continue
                hits, e2 = parse_hits(resp)
                if e2:
                    missed.append(f"q{qi}: {json.dumps(e2)[:80]}")
                    got = set()
                else:
                    got = set(id_map.get(int(h["node_id"]), -1) for h in hits)
                break
            if got is None:
                got = set()
            ov = len(exact & got) / args.k
            recalls.append(ov)
            if ov < 1.0 and len(missed) < 3:
                missed.append(f"q{qi}: exact={sorted(exact)[:6]}.. got={sorted(x for x in got if x>=0)[:6]}..")
        out["recall"] = {
            "queries": args.queries, "k": args.k,
            "recall_at_k": round(statistics.fmean(recalls), 4) if recalls else None,
            "min_q": round(min(recalls), 3) if recalls else None,
            "perfect_q": sum(1 for x in recalls if x == 1.0),
            "examples": missed[:3]}
        print(f"  recall@{args.k}={out['recall']['recall_at_k']} min_q={out['recall']['min_q']} "
              f"perfect={out['recall']['perfect_q']}/{args.queries}", flush=True)
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        out["stderr_tail"] = (proc.stderr.read() or b"").decode(errors="replace")[-600:]

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
