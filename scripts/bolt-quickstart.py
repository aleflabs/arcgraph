#!/usr/bin/env python3
"""Verify ArcGraph Bolt reads with the validated Neo4j Python driver."""

import json
import sys

try:
    import neo4j
except ImportError as error:
    raise SystemExit(
        "bolt-quickstart: FAIL install neo4j==6.2.0 in the active environment"
    ) from error


EXPECTED_DRIVER = "6.2.0"
EXPECTED_ROWS = [["Ada Lovelace", 1952, "Grace Hopper"]]
QUERY = (
    "MATCH (a:Person)-[r:KNOWS]->(b:Person) "
    "RETURN a.name, r.since, b.name ORDER BY a.name"
)


def compact(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def main():
    if neo4j.__version__ != EXPECTED_DRIVER:
        raise SystemExit(
            "bolt-quickstart: FAIL expected neo4j {}, got {}".format(
                EXPECTED_DRIVER, neo4j.__version__
            )
        )

    print("bolt-quickstart: DRIVER neo4j={}".format(neo4j.__version__))
    driver = neo4j.GraphDatabase.driver(
        "bolt://127.0.0.1:7687",
        auth=neo4j.basic_auth("neo4j", "local-dev-password"),
        encrypted=False,
        connection_timeout=5.0,
    )
    try:
        driver.verify_connectivity()
        print("bolt-quickstart: CONNECT bolt=5.0 principal=neo4j")
        with driver.session() as session:
            rows = [list(record.values()) for record in session.run(QUERY)]
    finally:
        driver.close()

    if rows != EXPECTED_ROWS:
        raise SystemExit(
            "bolt-quickstart: FAIL expected rows={}, got rows={}".format(
                compact(EXPECTED_ROWS), compact(rows)
            )
        )
    print("bolt-quickstart: ROWS {}".format(compact(rows)))
    print("bolt-quickstart: PASS properties and relationship survived Bolt")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("bolt-quickstart: FAIL {}".format(error), file=sys.stderr)
        sys.exit(1)
