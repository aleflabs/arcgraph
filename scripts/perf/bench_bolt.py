#!/usr/bin/env python3
"""Closed-loop RPS / latency benchmark for the ArcGraph Bolt 5.0 surface.

Drives a running `arcgraph serve --bolt <addr> --data <dir>` daemon with the
official neo4j Python driver (Apache-2.0). Measures, at the client wire, the
sustained throughput and per-query latency (P50/P95/P99/max) under a concurrency
sweep, plus a cold-vs-warm split.

Methodology (honesty-first):
  * Closed-loop: each of C worker threads issues queries back-to-back (no think
    time), fully draining each result, for a fixed measurement window. RPS is
    aggregate completed-queries / wall-time; latency is per-query, client-side.
  * Bounded result sets: the default workload returns 1..~4 rows per query
    (count aggregations + a point-anchored expansion + a pure-compute floor),
    so neither the server's nor the driver's result-streaming cost distorts the
    execution+wire latency we are trying to isolate. (A separate large-result
    probe characterizes row-transfer cost.)
  * Warmup window discarded before the measured window (plan/page cache warm).
  * The very first query of a fresh connection is timed separately as "cold".
  * AVOIDS known-broken constructs (ORDER BY, RETURN ... AS alias read-by-name,
    shortestPath+WHERE, multi-col CREATE-RETURN) — those are routed bugs, not
    perf signal.

The client is the measuring instrument, not the system under test; client CPU
is sampled so client-bound saturation is reported honestly rather than as
server throughput.

Usage:
    bench_bolt.py --addr 127.0.0.1:7687 --user neo4j --password x \
        --concurrency 1,2,4,8,16,32 --duration 20 --warmup 4 \
        --out /tmp/ag_bolt_results.json
"""
import argparse
import json
import statistics
import threading
import time

from neo4j import GraphDatabase

# ── Read workload: (name, cypher, weight). All bounded result sets, all proven
#    to execute on the Bolt surface per the server-e2e report. Read positionally
#    (col_0) — we never read-by-alias (the #353 alias-loss gap).
WORKLOAD = [
    ("count_user",      "MATCH (n:User) RETURN count(n)",                                  3),
    ("count_all",       "MATCH (n) RETURN count(n)",                                       1),
    ("expand_count",    "MATCH (a:User)-[:KNOWS]->(b:User) RETURN count(b)",               2),
    ("point_lookup",    "MATCH (n:User {name:'user_1000'}) RETURN n.age",                  3),
    ("anchored_expand", "MATCH (a:User {name:'user_100'})-[:KNOWS]->(b) RETURN b.name",    2),
    ("compute_floor",   "RETURN 1 IN [1,2,3]",                                             1),
]


def expand_workload(workload):
    seq = []
    for (name, q, w) in workload:
        seq.extend([(name, q)] * w)
    return seq


def run_query(session, q):
    # Fully drain so the server does all the work and serializes the result.
    res = session.run(q)
    rows = 0
    for _ in res:
        rows += 1
    res.consume()
    return rows


def worker(driver, seq, stop_at, lat_by_type, lat_all, counts, idx, errors):
    # Each worker owns its own session (a Bolt connection from the pool).
    local_lat = {name: [] for (name, _q) in seq}
    local_all = []
    local_count = 0
    local_err = 0
    with driver.session() as session:
        i = 0
        n = len(seq)
        while time.monotonic() < stop_at:
            name, q = seq[i % n]
            i += 1
            t0 = time.perf_counter()
            try:
                run_query(session, q)
            except Exception:
                local_err += 1
                continue
            dt = (time.perf_counter() - t0) * 1000.0
            local_lat[name].append(dt)
            local_all.append(dt)
            local_count += 1
    # merge under lock
    with counts["lock"]:
        for k, v in local_lat.items():
            lat_by_type.setdefault(k, []).extend(v)
        lat_all.extend(local_all)
        counts["total"] += local_count
        errors[0] += local_err


def pctl(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def measure_cold(driver, seq):
    """First query on a brand-new connection (cold path)."""
    name, q = seq[0]
    with driver.session() as session:
        t0 = time.perf_counter()
        run_query(session, q)
        return {"query": name, "cold_ms": round((time.perf_counter() - t0) * 1000.0, 3)}


def run_level(driver, seq, concurrency, duration, warmup):
    # Warmup (discarded).
    if warmup > 0:
        stop = time.monotonic() + warmup
        lat_by_type, lat_all = {}, []
        counts = {"total": 0, "lock": threading.Lock()}
        errors = [0]
        ts = [threading.Thread(target=worker, args=(driver, seq, stop, lat_by_type,
                                                     lat_all, counts, k, errors))
              for k in range(concurrency)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    # Measured window.
    lat_by_type, lat_all = {}, []
    counts = {"total": 0, "lock": threading.Lock()}
    errors = [0]
    t_start = time.monotonic()
    stop = t_start + duration
    ts = [threading.Thread(target=worker, args=(driver, seq, stop, lat_by_type,
                                                lat_all, counts, k, errors))
          for k in range(concurrency)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.monotonic() - t_start

    rps = counts["total"] / wall if wall > 0 else 0.0
    out = {
        "concurrency": concurrency,
        "duration_s": round(wall, 3),
        "completed": counts["total"],
        "errors": errors[0],
        "rps": round(rps, 1),
        "lat_ms": {
            "p50": _r(pctl(lat_all, 50)),
            "p95": _r(pctl(lat_all, 95)),
            "p99": _r(pctl(lat_all, 99)),
            "max": _r(max(lat_all) if lat_all else None),
            "mean": _r(statistics.fmean(lat_all) if lat_all else None),
        },
        "by_type": {k: {"n": len(v), "p50": _r(pctl(v, 50)), "p99": _r(pctl(v, 99))}
                    for k, v in sorted(lat_by_type.items())},
    }
    return out


def _r(x):
    return round(x, 3) if x is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="perftest")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=4.0)
    ap.add_argument("--out", default="/tmp/ag_bolt_results.json")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    seq = expand_workload(WORKLOAD)
    uri = f"bolt://{args.addr}"

    # max_connection_pool_size must cover the highest concurrency level.
    driver = GraphDatabase.driver(uri, auth=(args.user, args.password),
                                  max_connection_pool_size=max(levels) + 4)
    driver.verify_connectivity()

    results = {"addr": args.addr, "workload": [(n, q) for (n, q, _w) in WORKLOAD],
               "duration_s": args.duration, "warmup_s": args.warmup, "levels": []}
    results["cold"] = measure_cold(driver, seq)

    for c in levels:
        r = run_level(driver, seq, c, args.duration, args.warmup)
        results["levels"].append(r)
        print(f"  C={c:>3}  RPS={r['rps']:>9.1f}  "
              f"P50={r['lat_ms']['p50']}  P95={r['lat_ms']['p95']}  "
              f"P99={r['lat_ms']['p99']}  max={r['lat_ms']['max']}  "
              f"completed={r['completed']}  errors={r['errors']}", flush=True)

    driver.close()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
