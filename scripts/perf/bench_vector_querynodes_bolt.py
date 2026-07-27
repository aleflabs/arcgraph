#!/usr/bin/env python3
"""Vector KNN (HNSW) RPS / latency / recall over the Bolt 5.0 surface — via the
`CALL db.index.vector.queryNodes(indexName, k, queryVector)` procedure.

WHY THIS SIBLING TO `bench_vector_bolt.py` EXISTS (post-#789 finding):
  #789 (7c781442) attached the vector index to the ArcQL *catalog* so the served
  concurrent `RANK BY HYBRID(VECTOR(..), TEXT(..))` path passes BIND-time
  cross-substrate validation (closing G6). But two further facts, empirically
  re-derived on this branch, keep `bench_vector_bolt.py`'s RANK-BY query from
  running as an un-throttled vector-only KNN RPS lane:

    1. `RANK BY HYBRID(...)` is a HYBRID surface BY DESIGN (ADR-038 §2 D-3):
       it requires BOTH a `VECTOR(...)` AND a `TEXT(...)` operand. A vector-only
       `HYBRID(VECTOR(..))` is rejected at bind time with
       `CrossSubstrateError::HybridMissingOperand{kind:"TEXT"}` — so it is not a
       pure-vector-KNN query at all. (`bench_vector_bolt.py`'s query is vector-only.)
    2. Even a FULL hybrid `HYBRID(VECTOR(..), TEXT(..))` — which now binds — fails
       at EXECUTION time: `RankByHybridOp::fuse` runs a defense-in-depth
       `substrate.has_vector_substrate()` pre-gate, and the production
       `CrudExecutorSubstrate::has_vector_substrate()` returns a hardcoded `false`
       (availability is per-tenant, resolved INSIDE `vector_search`). So the served
       hybrid path returns `DatabaseError: substrate 'vector' unavailable for tenant`
       (a NEW execution-time gate, distinct from G6's bind-time gate).

  The `CALL db.index.vector.queryNodes` procedure (the langchain-neo4j `Neo4jVector`
  per-query surface, #830 D4) DELIBERATELY does NOT pre-gate on
  `has_vector_substrate()` (procedure_call.rs:425-439) — it calls
  `ExecutorSubstrate::vector_search` directly, which resolves availability per-tenant
  via the router's vector handle + the bound `SubstrateSearchProvider`. On the Bolt
  serve path both ARE wired (`arcgraph.rs:2087` binds the provider; `build_durable`
  attaches `.vector(store)`), so `queryNodes` SERVES real HNSW KNN over the
  un-throttled Bolt surface — the ONLY concurrent vector-KNN RPS lane that runs today.

  This harness measures THAT surface. It is directly comparable to the ~960 RPS
  Bolt graph-read baseline (#1238) and to `bench_vector_bolt.py`'s intended methodology.

Honesty-first methodology (mirrors bench_vector_bolt.py):
  * One server lifetime: load all vectors over Bolt (inline-literal CREATE — G4/G8),
    THEN sweep. No writes during the read window (the derived HNSW is built once on
    first query, then cached; a concurrent write invalidates+rebuilds it every query).
  * Closed-loop: C workers issue `queryNodes` KNN queries back-to-back, draining each
    result, for a fixed window. RPS = completed / wall; latency per-query, client-side.
    Warmup window discarded (HNSW build + page-cache warm).
  * recall@k: HNSW top-k (over the wire) vs an EXACT brute-force L2 oracle (numpy) over
    the SAME float32 vectors. A recall miss is a real finding. RPS is reported AT this
    stated recall (never bare RPS).

Vectors: deterministic seeded numpy uniform-[0,1), L2-normalized (non-negative to
satisfy the Bolt inline-literal CREATE constraint, G8). Queries are out-of-set (fresh
seed) — honest ANN, not a trivial self-match.

Usage:
    bench_vector_querynodes_bolt.py --addr 127.0.0.1:7687 --user neo4j --password perftest \
        --n 5000 --dim 128 --k 10 --writers 8 \
        --concurrency 1,2,4,8,16 --duration 15 --warmup 4 \
        --queries 200 --out /tmp/ag_vector_qn_bolt.json
"""
import argparse
import json
import statistics
import threading
import time

import numpy as np
from neo4j import GraphDatabase

# Advisory index name — `queryNodes` resolves it to the served vector property
# (`embedding`) when no `CREATE VECTOR INDEX` catalog entry matches (#830/ADR-200
# fallback). Any string works; we use the langchain-conventional name.
INDEX_NAME = "doc_embedding"


def gen_matrix(n, dim, seed):
    rng = np.random.default_rng(seed)
    m = rng.random((n, dim), dtype=np.float32)  # uniform [0,1) — non-negative (G8)
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


def knn_cypher(k):
    """The served concurrent KNN surface: langchain-neo4j `Neo4jVector` procedure."""
    return (f"CALL db.index.vector.queryNodes('{INDEX_NAME}', {k}, $q) "
            f"YIELD node, score RETURN node.vid AS vid, score")


# load
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


# smoke
def smoke(driver, mat, k):
    """One KNN query; confirm it SERVES (ranked rows). Query == an in-set vector
    -> its own vid must rank #1 (self-match sanity)."""
    qv = [float(x) for x in mat[0].tolist()]
    try:
        with driver.session() as s:
            res = s.run(knn_cypher(k), q=qv)
            rows = [r.values() for r in res]  # (vid, score)
    except Exception as e:
        return {"served": False, "rows": 0, "top_vids": [],
                "self_is_rank1": False, "error": f"{type(e).__name__}: {str(e)[:240]}"}
    vids = [int(r[0]) if r[0] is not None else None for r in rows]
    return {"served": len(rows) > 0, "rows": len(rows), "top_vids": vids[:k],
            "self_is_rank1": (len(vids) > 0 and vids[0] == 0), "error": None}


# sweep
def worker(driver, queries, k, stop_at, lat_all, counts, errors, err_msgs):
    local_lat, local_count, local_err = [], 0, 0
    nq = len(queries)
    with driver.session() as session:
        i = 0
        while time.monotonic() < stop_at:
            qv = queries[i % nq]
            i += 1
            t0 = time.perf_counter()
            try:
                res = session.run(knn_cypher(k), q=qv)
                for _ in res:
                    pass
                res.consume()
            except Exception as e:
                local_err += 1
                if len(err_msgs) < 3:
                    err_msgs.append(f"{type(e).__name__}: {str(e)[:120]}")
                continue
            local_lat.append((time.perf_counter() - t0) * 1000.0)
            local_count += 1
    with counts["lock"]:
        lat_all.extend(local_lat)
        counts["total"] += local_count
        errors[0] += local_err


def run_level(driver, queries, k, concurrency, duration, warmup):
    if warmup > 0:
        stop = time.monotonic() + warmup
        lat, counts, errors, em = [], {"total": 0, "lock": threading.Lock()}, [0], []
        ts = [threading.Thread(target=worker,
                               args=(driver, queries, k, stop, lat, counts, errors, em))
              for _ in range(concurrency)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    lat, counts, errors, em = [], {"total": 0, "lock": threading.Lock()}, [0], []
    t_start = time.monotonic()
    stop = t_start + duration
    ts = [threading.Thread(target=worker,
                           args=(driver, queries, k, stop, lat, counts, errors, em))
          for _ in range(concurrency)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.monotonic() - t_start
    rps = counts["total"] / wall if wall > 0 else 0.0
    return {"concurrency": concurrency, "duration_s": round(wall, 3),
            "completed": counts["total"], "errors": errors[0], "rps": round(rps, 1),
            "error_examples": em[:3],
            "lat_ms": {"p50": _r(pctl(lat, 50)), "p95": _r(pctl(lat, 95)),
                       "p99": _r(pctl(lat, 99)), "max": _r(max(lat) if lat else None),
                       "mean": _r(statistics.fmean(lat) if lat else None)}}


# recall
def recall(driver, mat, query_vecs, k):
    """recall@k of the served HNSW (via queryNodes) vs exact brute-force L2."""
    found_total, missed_examples, per_q = 0.0, [], []
    with driver.session() as s:
        for qi, qv in enumerate(query_vecs):
            d = np.sum((mat - qv) ** 2, axis=1)  # squared L2 to every node
            exact = set(int(x) for x in np.argsort(d)[:k])
            try:
                res = s.run(knn_cypher(k), q=[float(x) for x in qv])
                got = set()
                for r in res:
                    try:
                        got.add(int(r.values()[0]))
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
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--warmup", type=float, default=4.0)
    ap.add_argument("--queries", type=int, default=200, help="query vectors for recall + sweep cycle")
    ap.add_argument("--skip-load", action="store_true", help="vectors already loaded this lifetime")
    ap.add_argument("--out", default="/tmp/ag_vector_qn_bolt.json")
    args = ap.parse_args()

    print(f"gen {args.n} x {args.dim} vectors (seed={args.seed}) ...", flush=True)
    mat = gen_matrix(args.n, args.dim, args.seed)
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
           "surface": "CALL db.index.vector.queryNodes (Bolt, un-throttled)",
           "duration_s": args.duration, "warmup_s": args.warmup}

    if not args.skip_load:
        print(f"loading {args.n} Doc nodes over Bolt ({args.writers} writers) ...", flush=True)
        out["load"] = load(driver, mat, args.writers)
        print(f"  load: {out['load']}", flush=True)
        if not out["load"]["verified_count"]:
            print("  -> 0 Doc nodes loaded; KNN cannot serve. Stop (finding).", flush=True)
            driver.close()
            json.dump(out, open(args.out, "w"), indent=2)
            return

    out["smoke"] = smoke(driver, mat, args.k)
    print(f"  smoke: served={out['smoke']['served']} rows={out['smoke']['rows']} "
          f"self_is_rank1={out['smoke']['self_is_rank1']} top={out['smoke']['top_vids']} "
          f"err={out['smoke']['error']}", flush=True)
    if not out["smoke"]["served"]:
        print("  -> queryNodes KNN did NOT serve over Bolt (finding). Stop.", flush=True)
        driver.close()
        json.dump(out, open(args.out, "w"), indent=2)
        return

    print(f"recall@{args.k} over {args.queries} queries vs brute-force oracle ...", flush=True)
    out["recall"] = recall(driver, mat, query_vecs, args.k)
    print(f"  recall@{args.k} = {out['recall']['recall_at_k']}  "
          f"min_q={out['recall']['min_q']}  perfect={out['recall']['perfect_q']}/{args.queries}",
          flush=True)

    print("concurrency sweep (KNN RPS):", flush=True)
    out["levels"] = []
    for c in levels:
        r = run_level(driver, query_lists, args.k, c, args.duration, args.warmup)
        out["levels"].append(r)
        print(f"  C={c:>3}  RPS={r['rps']:>8.1f}  P50={r['lat_ms']['p50']}  "
              f"P95={r['lat_ms']['p95']}  P99={r['lat_ms']['p99']}  max={r['lat_ms']['max']}  "
              f"completed={r['completed']}  errors={r['errors']}", flush=True)

    driver.close()
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
