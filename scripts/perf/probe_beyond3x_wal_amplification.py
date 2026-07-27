#!/usr/bin/env python3
"""Probe post-D-1/D-2 batch ingest rate and WAL bytes per node."""

import argparse
import os
import time

from neo4j import GraphDatabase


def dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            try:
                total += os.path.getsize(os.path.join(root, filename))
            except OSError:
                pass
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="127.0.0.1:7797")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="x")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--nodes", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=5_000)
    parser.add_argument("--inline-nodes", type=int, default=2_000)
    args = parser.parse_args()

    active_data_dir = args.data_dir
    current = os.path.join(args.data_dir, "CURRENT")
    if os.path.isfile(current):
        with open(current, encoding="utf-8") as pointer:
            generation = pointer.read().strip()
        if generation:
            active_data_dir = os.path.join(args.data_dir, generation)

    wal_dir = None
    for candidate in ("wal", "WAL"):
        path = os.path.join(active_data_dir, candidate)
        if os.path.isdir(path):
            wal_dir = path
            break
    measured_dir = wal_dir or active_data_dir
    print(f"[probe] measuring bytes under: {measured_dir}")

    driver = GraphDatabase.driver(
        f"bolt://{args.addr}", auth=(args.user, args.password)
    )
    with driver.session() as session:
        try:
            session.run(
                "UNWIND $rows AS row CREATE (n:Probe {id: row.id, name: row.name})",
                rows=[{"id": -1, "name": "smoke"}],
            ).consume()
            print("[probe] D-1 UNWIND+param CREATE: ACCEPTED")
        except Exception as error:  # noqa: BLE001
            print(f"[probe] D-1 UNWIND+param CREATE: REJECTED — {error}")
            print("[probe] aborting (D-1 not in this build)")
            return

        before = dir_bytes(measured_dir)
        start = time.monotonic()
        done = 0
        while done < args.nodes:
            count = min(args.batch, args.nodes - done)
            rows = [
                {
                    "id": done + offset,
                    "name": f"user_{done + offset}",
                    "age": (done + offset) % 90,
                }
                for offset in range(count)
            ]
            session.run(
                "UNWIND $rows AS row "
                "CREATE (n:Probe {id: row.id, name: row.name, age: row.age})",
                rows=rows,
            ).consume()
            done += count
        elapsed = time.monotonic() - start
        after = dir_bytes(measured_dir)
        delta = after - before
        print(
            f"[probe] UNWIND ingest: {done} nodes in {elapsed:.2f}s = "
            f"{done / elapsed:,.0f} nodes/s (batch={args.batch})"
        )
        print(
            f"[probe] WAL delta: {delta:,} B = {delta / done:,.0f} B/node "
            "(vs ~150-250 B/node logical floor)"
        )

        start = time.monotonic()
        for index in range(args.inline_nodes):
            session.run(
                "CREATE (n:ProbeInline "
                f"{{id: {1_000_000 + index}, name: 'inline_{index}', age: {index % 90}}})"
            ).consume()
        inline_elapsed = time.monotonic() - start
        print(
            f"[probe] inline-literal autocommit: {args.inline_nodes} nodes in "
            f"{inline_elapsed:.2f}s = {args.inline_nodes / inline_elapsed:,.0f} nodes/s"
        )
    driver.close()


if __name__ == "__main__":
    main()
