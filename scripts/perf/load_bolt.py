#!/usr/bin/env python3
"""Build a live read-benchmark dataset over the Bolt surface.

The ArcGraph Cypher executor's write path (at this commit) does NOT support the
efficient bulk-load forms over Bolt — UNWIND/parameterized CREATE fails a
type-check, and `MATCH (a),(b) CREATE (a)-[r]->(b)` (relating two existing
nodes) parse-errors. The one rel-creating form that works is the inline-path
literal CREATE: `CREATE (a:User {..})-[:KNOWS]->(b:User {..})` (without a
multi-column RETURN, which hits the #353-class gap). Each such statement creates
2 fresh nodes + 1 rel.

So the live Bolt dataset is `pairs` disjoint 2-node/1-rel components:
  user_(2k) -[:KNOWS]-> user_(2k+1),  for k in [0, pairs).
=> 2*pairs `User` nodes, `pairs` `KNOWS` rels. Disjoint-pair topology (documented
honestly): sufficient for label-scan / point-lookup / 1-hop traversal / compute
read benchmarking; it is not a connected social graph (the executor write-path
limits prevent that over Bolt without graph.ingest, an MCP-only tool).

Writes are fanned across `--writers` concurrent Bolt connections to amortize the
per-write fsync wait on the durable substrate.

Usage:
    load_bolt.py --addr 127.0.0.1:7687 --user neo4j --password x \
        --pairs 1000 --writers 8
"""
import argparse
import threading
import time

from neo4j import GraphDatabase


def writer(driver, ks, errors, done):
    with driver.session() as s:
        for k in ks:
            a, b = 2 * k, 2 * k + 1
            q = (f"CREATE (a:User {{id:{a}, name:'user_{a}', age:{18 + a % 60}}})"
                 f"-[:KNOWS]->(b:User {{id:{b}, name:'user_{b}', age:{18 + b % 60}}})")
            try:
                s.run(q).consume()
                done[0] += 1
            except Exception as e:
                errors.append(str(e)[:120])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="perftest")
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--writers", type=int, default=8)
    args = ap.parse_args()

    driver = GraphDatabase.driver(f"bolt://{args.addr}", auth=(args.user, args.password),
                                  max_connection_pool_size=args.writers + 4)
    driver.verify_connectivity()

    # shard pair indices across writers
    shards = [[] for _ in range(args.writers)]
    for k in range(args.pairs):
        shards[k % args.writers].append(k)
    errors, done = [], [0]
    t0 = time.perf_counter()
    ts = [threading.Thread(target=writer, args=(driver, shards[i], errors, done))
          for i in range(args.writers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0

    with driver.session() as s:
        nuser = s.run("MATCH (n:User) RETURN count(n)").single().values()[0]
        nrel = s.run("MATCH (a:User)-[:KNOWS]->(b:User) RETURN count(b)").single().values()[0]
    driver.close()

    print(f"loaded {done[0]}/{args.pairs} pairs in {dt:.1f}s "
          f"({args.writers} writers, {2*done[0]/dt:.0f} nodes/s)")
    print(f"verify: User nodes={nuser}  KNOWS rels={nrel}  errors={len(errors)}")
    if errors:
        print("first errors:", errors[:3])


if __name__ == "__main__":
    main()
