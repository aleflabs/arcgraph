#!/usr/bin/env python3
"""Stage-3 attack on the #1387 fix: concurrent MERGE on the SAME key must create EXACTLY ONE node.

Original NN-4 defect (#1384): two concurrent `MERGE (u:User {email:X})` both took the
create branch (SI snapshot invisibility + OCC only conflicts on same-record-id) → 2 nodes.
Fix (#1387): a per-(tenant, key) serialization guard acquired by the query driver serializes
the match→create window → get-or-create uniqueness holds under concurrency.

This attack fires N concurrent Bolt connections, each running `MERGE (u:User {email:$e})`
against a SHARED key, released together by a barrier to maximize the race window, then
asserts EXACTLY ONE node exists. Repeat over T trials. PASS = every trial yields exactly 1.

Usage:
  attack_1387_concurrent_merge.py --addr <the-oci-vm>:7687 --user neo4j --password x \
      --threads 8 --trials 50
"""
import argparse
import threading

from neo4j import GraphDatabase


def merge_worker(driver, email, barrier, errors):
    # NOTE: ArcGraph's MERGE type-checks a parameterized prop ($e) as an error but
    # accepts an inline string literal — so we inline the (safe, generated) email.
    q = f'MERGE (u:User {{email: "{email}"}}) RETURN u'
    try:
        with driver.session() as s:
            barrier.wait()  # release all threads together — maximize the match→create race
            s.run(q).consume()
    except Exception as ex:
        errors.append(f"{type(ex).__name__}: {str(ex)[:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="x")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    driver = GraphDatabase.driver(f"bolt://{args.addr}", auth=(args.user, args.password),
                                  max_connection_pool_size=args.threads + 4)
    driver.verify_connectivity()

    dupes, oks, trial_errors = 0, 0, []
    counts = []
    for t in range(args.trials):
        email = f"race{t}atxio"  # alnum only — avoid any tokenizer edge on @/.
        # clean slate for this key (inline literal — MERGE/MATCH param props type-check-error on ArcGraph)
        with driver.session() as s:
            s.run(f'MATCH (u:User {{email: "{email}"}}) DETACH DELETE u').consume()
        barrier = threading.Barrier(args.threads)
        errs = []
        ths = [threading.Thread(target=merge_worker, args=(driver, email, barrier, errs))
               for _ in range(args.threads)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        with driver.session() as s:
            n = s.run(f'MATCH (u:User {{email: "{email}"}}) RETURN count(u)').single().value()
        counts.append(n)
        if n == 1:
            oks += 1
        else:
            dupes += 1
        if errs:
            trial_errors.extend(errs[:2])

    driver.close()
    verdict = "PASS" if dupes == 0 else "FAIL"
    print(f"#1387 concurrent-MERGE attack: {args.threads} threads × {args.trials} trials")
    print(f"  trials yielding exactly 1 node: {oks}/{args.trials}")
    print(f"  trials with DUPLICATE (>1 or 0) nodes: {dupes}  (counts seen: {sorted(set(counts))})")
    print(f"  driver errors: {len(trial_errors)}  {trial_errors[:3]}")
    print(f"  VERDICT: {verdict} — get-or-create uniqueness {'HOLDS' if dupes==0 else 'VIOLATED'} under concurrency")


if __name__ == "__main__":
    main()
