#!/usr/bin/env python3
"""Vector KNN (HNSW) RPS / latency / recall harness over the Bolt 5.0 surface.

The headline concurrent vector benchmark. Bolt is the only un-throttled surface
(MCP-stdio rate-limits reads to 100-burst / 1.67-per-sec — finding G5), so it is
where sustained KNN RPS is meaningful — directly comparable to the ~475 RPS
graph-read knee in the sibling graph/Cypher report.

Served KNN path (post #765 PART-1 / #779): the Bolt daemon binds the served
`HnswVectorSearchProvider`, so a Cypher

    MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, $q, K = 10))
                  WITH FUSION = RRF(k = 60) RETURN n

builds a per-tenant ephemeral HNSW from the tenant's durable nodes' `embedding`
property (L2 / squared-euclidean kernel) and returns ranked rows.

Honesty-first methodology (mirrors bench_bolt.py):
  * One server lifetime: load all vectors over Bolt (inline-literal CREATE — the
    only working write form, G4), THEN sweep. No writes during the read window,
    so the derived HNSW is built once (first query) + cached (high-water stable);
    a concurrent write would invalidate+rebuild it every query and tank RPS.
  * Closed-loop: C workers issue KNN queries back-to-back, fully draining each
    result, for a fixed window. RPS = completed / wall; latency is per-query,
    client-side. Warmup window discarded (HNSW build + page cache warm).
  * recall@k: HNSW top-k (over the wire) vs an EXACT brute-force L2 oracle
    (numpy) over the SAME float32 vectors. A recall miss is a real finding
    (HNSW ef/M params or substrate quality).

Vectors: deterministic seeded numpy Gaussian, L2-normalized (realistic embedding
shape; unit-norm makes L2 rank order match cosine). The engine casts the stored
f64 property -> f32; the oracle runs on the identical float32 matrix, so the
brute-force ground truth matches what the index actually indexed.

Usage:
    bench_vector_bolt.py --addr 127.0.0.1:7687 --user neo4j --password perftest \
        --n 5000 --dim 128 --k 10 --writers 8 \
        --concurrency 1,2,4,8,16 --duration 20 --warmup 4 \
        --queries 200 --query-mode param --out /tmp/ag_vector_bolt.json
"""
import argparse
import json
import statistics
import threading
import time

import numpy as np
from neo4j import GraphDatabase


# ── deterministic dataset (seeded; loader + oracle regenerate the identical matrix)
#
# NON-NEGATIVE on purpose: the Bolt inline-literal CREATE write path rejects a
# list property containing NEGATIVE floats — `CREATE (n:Doc {embedding:[-0.1,..]})`
# fails `CreateNodeOp: property embedding is not a literal` because `-0.1` parses
# as a unary-minus EXPRESSION, not a literal (the CREATE path requires literal
# property values). MCP `graph.ingest` (bench_vector.py) has no such limit and
# uses full Gaussians. We use uniform [0,1) here so the corpus actually loads over
# Bolt and the harness reaches the real query-side blocker (the G6 RANK BY gate).
def gen_matrix(n, dim, seed):
    rng = np.random.default_rng(seed)
    m = rng.random((n, dim), dtype=np.float32)     # uniform [0,1) — non-negative
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


def vec_literal(v):
    # full-precision repr so the stored f64 -> f32 == the oracle's float32 value
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


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


def knn_cypher(k, query_mode, qv=None):
    """Build the KNN Cypher. query_mode: 'param' -> $q bind; 'literal' -> inline."""
    qexpr = "$q" if query_mode == "param" else vec_literal(qv)
    return (f"MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, {qexpr}, K = {k})) "
            f"WITH FUSION = RRF(k = 60) RETURN n")


def row_vid(rec):
    """Extract the vid from a `RETURN n` record (positional; alias-on-wire is G/#353)."""
    node = rec.values()[0]
    try:
        return int(node.get("vid"))
    except Exception:
        # node may be a dict-like; fall back
        return int(node["vid"])


# ─────────────────────────────── load ───────────────────────────────
def load_writer(driver, mat, idxs, done, errors):
    with driver.session() as s:
        for i in idxs:
            q = f"CREATE (n:Doc {{vid:{i}, embedding:{vec_literal(mat[i])}}})"
            try:
                s.run(q).consume()
                done[0] += 1
            except Exception as e:
                errors.append(f"{type(e).__name__}: {str(e)[:140]}")


def load(driver, mat, writers):
    n = mat.shape[0]
    shards = [[] for _ in range(writers)]
    for i in range(n):
        shards[i % writers].append(i)
    done, errors = [0], []
    t0 = time.perf_counter()
    ts = [threading.Thread(target=load_writer, args=(driver, mat, shards[w], done, errors))
          for w in range(writers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    # NOTE: a fresh-session `MATCH (n:Doc)` can fail `unknown label Doc` after a
    # concurrent multi-writer load — a cross-session label-resolution issue
    # (G1-class catalog finding), distinct from the single-session smoke where it
    # resolves. Capture it as a finding rather than crashing.
    cnt, count_err = None, None
    try:
        with driver.session() as s:
            cnt = s.run("MATCH (n:Doc) RETURN count(n)").single().values()[0]
    except Exception as e:
        count_err = f"{type(e).__name__}: {str(e)[:160]}"
    return {"requested": n, "created": done[0], "verified_count": cnt,
            "verify_count_error": count_err,
            "load_s": round(dt, 2), "nodes_per_s": round(done[0] / dt, 1) if dt else 0.0,
            "writers": writers, "errors": len(errors), "first_errors": errors[:3]}


# ─────────────────────────────── smoke ──────────────────────────────
def smoke(driver, mat, k, query_mode):
    """One KNN query; confirm it SERVES (ranked rows, not IndexUnavailable/empty).
    Catches the query error (e.g. the G6 `-32005 substrate vector not attached`
    bind-gate, or `unknown label`) so the harness records the finding + stops."""
    qv = mat[0].tolist()  # query == an in-set vector -> its own vid must rank #1
    try:
        with driver.session() as s:
            if query_mode == "param":
                res = s.run(knn_cypher(k, "param"), q=[float(x) for x in qv])
            else:
                res = s.run(knn_cypher(k, "literal", qv))
            rows = [r.values() for r in res]
    except Exception as e:
        return {"served": False, "rows": 0, "top_vids": [], "self_is_rank1": False,
                "error": f"{type(e).__name__}: {str(e)[:240]}"}
    vids = []
    for r in rows:
        try:
            vids.append(int(r[0].get("vid")))
        except Exception:
            vids.append(r[0])
    return {"served": len(rows) > 0, "rows": len(rows), "top_vids": vids[:k],
            "self_is_rank1": (len(vids) > 0 and vids[0] == 0), "error": None}


# ─────────────────────────────── sweep ──────────────────────────────
def worker(driver, queries, k, query_mode, stop_at, lat_all, counts, errors):
    local_lat, local_count, local_err = [], 0, 0
    nq = len(queries)
    with driver.session() as session:
        i = 0
        while time.monotonic() < stop_at:
            qv = queries[i % nq]
            i += 1
            t0 = time.perf_counter()
            try:
                if query_mode == "param":
                    res = session.run(knn_cypher(k, "param"), q=qv)
                else:
                    res = session.run(knn_cypher(k, "literal", qv))
                rows = 0
                for _ in res:
                    rows += 1
                res.consume()
            except Exception:
                local_err += 1
                continue
            local_lat.append((time.perf_counter() - t0) * 1000.0)
            local_count += 1
    with counts["lock"]:
        lat_all.extend(local_lat)
        counts["total"] += local_count
        errors[0] += local_err


def run_level(driver, queries, k, query_mode, concurrency, duration, warmup):
    if warmup > 0:
        stop = time.monotonic() + warmup
        lat, counts, errors = [], {"total": 0, "lock": threading.Lock()}, [0]
        ts = [threading.Thread(target=worker,
                               args=(driver, queries, k, query_mode, stop, lat, counts, errors))
              for _ in range(concurrency)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    lat, counts, errors = [], {"total": 0, "lock": threading.Lock()}, [0]
    t_start = time.monotonic()
    stop = t_start + duration
    ts = [threading.Thread(target=worker,
                           args=(driver, queries, k, query_mode, stop, lat, counts, errors))
          for _ in range(concurrency)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.monotonic() - t_start
    rps = counts["total"] / wall if wall > 0 else 0.0
    return {"concurrency": concurrency, "duration_s": round(wall, 3),
            "completed": counts["total"], "errors": errors[0], "rps": round(rps, 1),
            "lat_ms": {"p50": _r(pctl(lat, 50)), "p95": _r(pctl(lat, 95)),
                       "p99": _r(pctl(lat, 99)), "max": _r(max(lat) if lat else None),
                       "mean": _r(statistics.fmean(lat) if lat else None)}}


# ─────────────────────────────── recall ─────────────────────────────
def recall(driver, mat, query_vecs, k, query_mode):
    """recall@k of the served HNSW vs exact brute-force L2 over the same vectors."""
    # exact oracle: top-k by L2 (squared) for each query
    found_total, missed_examples = 0.0, []
    per_q = []
    with driver.session() as s:
        for qi, qv in enumerate(query_vecs):
            d = np.sum((mat - qv) ** 2, axis=1)         # squared L2 to every node
            exact = set(int(x) for x in np.argsort(d)[:k])
            try:
                if query_mode == "param":
                    res = s.run(knn_cypher(k, "param"), q=[float(x) for x in qv])
                else:
                    res = s.run(knn_cypher(k, "literal", qv))
                got = set()
                for r in res:
                    try:
                        got.add(row_vid(r))
                    except Exception:
                        pass
            except Exception as e:
                missed_examples.append(f"q{qi}: {type(e).__name__}: {str(e)[:100]}")
                per_q.append(0.0)
                continue
            ov = len(exact & got) / k
            per_q.append(ov)
            found_total += ov
            if ov < 1.0 and len(missed_examples) < 3:
                missed_examples.append(f"q{qi}: exact={sorted(exact)} got={sorted(got)}")
    mean = found_total / len(query_vecs) if query_vecs else 0.0
    return {"queries": len(query_vecs), "k": k, "recall_at_k": round(mean, 4),
            "min_q": round(min(per_q), 3) if per_q else None,
            "perfect_q": sum(1 for x in per_q if x == 1.0),
            "examples": missed_examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="perftest")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=765)
    ap.add_argument("--writers", type=int, default=8)
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=4.0)
    ap.add_argument("--queries", type=int, default=200, help="query vectors for recall + sweep cycle")
    ap.add_argument("--query-mode", choices=["param", "literal"], default="param")
    ap.add_argument("--skip-load", action="store_true", help="vectors already loaded this lifetime")
    ap.add_argument("--out", default="/tmp/ag_vector_bolt.json")
    args = ap.parse_args()

    print(f"gen {args.n} x {args.dim} vectors (seed={args.seed}) ...", flush=True)
    mat = gen_matrix(args.n, args.dim, args.seed)
    # out-of-set query vectors (fresh seed) — honest ANN, not a trivial self-match.
    # Non-negative (uniform) to match the corpus + the Bolt literal constraint.
    qrng = np.random.default_rng(args.seed + 1)
    qraw = qrng.random((args.queries, args.dim), dtype=np.float32)
    qraw /= np.linalg.norm(qraw, axis=1, keepdims=True)
    query_vecs = [qraw[i] for i in range(args.queries)]
    query_lists = [[float(x) for x in qv] for qv in query_vecs]

    levels = [int(x) for x in args.concurrency.split(",")]
    driver = GraphDatabase.driver(f"bolt://{args.addr}", auth=(args.user, args.password),
                                  max_connection_pool_size=max(levels + [args.writers]) + 4)
    driver.verify_connectivity()
    print(f"connected bolt://{args.addr}", flush=True)

    out = {"addr": args.addr, "n": args.n, "dim": args.dim, "k": args.k, "seed": args.seed,
           "query_mode": args.query_mode, "duration_s": args.duration, "warmup_s": args.warmup}

    if not args.skip_load:
        print(f"loading {args.n} Doc nodes over Bolt ({args.writers} writers) ...", flush=True)
        out["load"] = load(driver, mat, args.writers)
        print(f"  load: {out['load']}", flush=True)
        if out["load"]["verified_count"] == 0:
            print("  -> 0 Doc nodes loaded; KNN cannot serve. Stop (finding).", flush=True)
            driver.close()
            json.dump(out, open(args.out, "w"), indent=2)
            return

    # smoke: prove it SERVES before sweeping
    out["smoke"] = smoke(driver, mat, args.k, args.query_mode)
    print(f"  smoke: served={out['smoke']['served']} rows={out['smoke']['rows']} "
          f"self_is_rank1={out['smoke']['self_is_rank1']} top={out['smoke']['top_vids']}", flush=True)
    if not out["smoke"]["served"]:
        print("  -> graph.search KNN did NOT serve over Bolt (finding). Stop.", flush=True)
        driver.close()
        json.dump(out, open(args.out, "w"), indent=2)
        return

    # recall@k (exact oracle)
    print(f"recall@{args.k} over {args.queries} queries vs brute-force oracle ...", flush=True)
    out["recall"] = recall(driver, mat, query_vecs, args.k, args.query_mode)
    print(f"  recall@{args.k} = {out['recall']['recall_at_k']}  "
          f"min_q={out['recall']['min_q']}  perfect={out['recall']['perfect_q']}/{args.queries}",
          flush=True)

    # concurrency sweep (RPS / latency)
    print("concurrency sweep (KNN RPS):", flush=True)
    out["levels"] = []
    for c in levels:
        r = run_level(driver, query_lists, args.k, args.query_mode, c, args.duration, args.warmup)
        out["levels"].append(r)
        print(f"  C={c:>3}  RPS={r['rps']:>8.1f}  P50={r['lat_ms']['p50']}  "
              f"P95={r['lat_ms']['p95']}  P99={r['lat_ms']['p99']}  max={r['lat_ms']['max']}  "
              f"completed={r['completed']}  errors={r['errors']}", flush=True)

    driver.close()
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
