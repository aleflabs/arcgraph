#!/usr/bin/env python3
"""#765/#779 vector-serve SMOKE — empirically confirm graph.search KNN SERVES over Bolt.

Validates the three building blocks the Bolt KNN RPS/recall harness rests on,
each printed verbatim so a reviewer can see exactly what the server returned:

  1. Inline-literal CREATE with a LIST embedding property
     (`CREATE (n:Doc {vid:0, embedding:[..]})`) — does the Bolt write path
     (which rejects param/UNWIND CREATE, finding G4) accept a list literal?
  2. The canonical ArcQL KNN read shape
     `MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, $q, K = 5))
      WITH FUSION = RRF(k = 60) RETURN n` — does it execute end-to-end now that
     #765 PART-1 / #779 bound the served HNSW provider (was -32004 IndexUnavailable)?
  3. $q passed as a Bolt list PARAMETER vs inlined as a literal — which binds?

A low/blocked result here is the finding — print it precisely and stop.

Usage:
    vector_smoke.py --addr 127.0.0.1:7687 --user neo4j --password perftest --dim 8
"""
import argparse

from neo4j import GraphDatabase


def vec_literal(v):
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="perftest")
    ap.add_argument("--dim", type=int, default=8)
    args = ap.parse_args()

    driver = GraphDatabase.driver(f"bolt://{args.addr}", auth=(args.user, args.password))
    driver.verify_connectivity()
    print(f"connected to bolt://{args.addr}")

    # 5 deterministic 2-D-ish vectors in a `dim`-space: doc-0 near origin, rest farther.
    base = [
        [0.0] * args.dim,
        [1.0] + [0.0] * (args.dim - 1),
        [0.0, 1.0] + [0.0] * (args.dim - 2),
        [10.0] * args.dim,
        [5.0] * args.dim,
    ]

    with driver.session() as s:
        # ── Block 1: inline-literal CREATE with a LIST embedding property.
        created, create_err = 0, None
        for i, v in enumerate(base):
            q = f"CREATE (n:Doc {{vid:{i}, embedding:{vec_literal(v)}}})"
            try:
                s.run(q).consume()
                created += 1
            except Exception as e:
                create_err = f"{type(e).__name__}: {str(e)[:200]}"
                break
        print(f"\n[1] inline-literal CREATE(list embedding): created={created}/{len(base)} "
              f"err={create_err}")
        if created == 0:
            print("    -> list-literal embedding CREATE BLOCKED over Bolt (finding). Stop.")
            driver.close()
            return

        # sanity: count Doc nodes
        try:
            n = s.run("MATCH (n:Doc) RETURN count(n)").single().values()[0]
            print(f"    MATCH (n:Doc) RETURN count(n) = {n}")
        except Exception as e:
            print(f"    count(Doc) err: {type(e).__name__}: {str(e)[:200]}")

        # query vector near the origin → expect vid 0 nearest.
        qv = [0.1] * args.dim
        k = 5
        rank = ("RANK BY HYBRID(VECTOR(n.embedding, {q}, K = %d)) "
                "WITH FUSION = RRF(k = 60)" % k)

        # ── Block 2: $q as a Bolt list PARAMETER, RETURN n.
        for label, ret in [("RETURN n", "RETURN n"), ("RETURN n.vid", "RETURN n.vid")]:
            cy = f"MATCH (n:Doc) {rank.format(q='$q')} {ret}"
            try:
                res = s.run(cy, q=[float(x) for x in qv])
                rows = [r.values() for r in res]
                print(f"\n[2] PARAM $q, {label}: ok rows={len(rows)}")
                for r in rows:
                    print(f"      {r}")
            except Exception as e:
                print(f"\n[2] PARAM $q, {label}: ERR {type(e).__name__}: {str(e)[:240]}")

        # ── Block 3: inlined literal vector (fallback if param binding fails).
        cy = f"MATCH (n:Doc) {rank.format(q=vec_literal(qv))} RETURN n"
        try:
            res = s.run(cy)
            rows = [r.values() for r in res]
            print(f"\n[3] INLINED literal vector, RETURN n: ok rows={len(rows)}")
            for r in rows:
                print(f"      {r}")
        except Exception as e:
            print(f"\n[3] INLINED literal vector: ERR {type(e).__name__}: {str(e)[:240]}")

    driver.close()
    print("\nsmoke done")


if __name__ == "__main__":
    main()
