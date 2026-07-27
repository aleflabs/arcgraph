#!/usr/bin/env python3
"""Generate a deterministic synthetic social graph for the ArcGraph perf harness.

Emits a chunked `graph.ingest` JSON-RPC request file (one MCP session ingests
the whole dataset into a durable `--data` dir) plus a metadata sidecar with the
expected row counts used as the read-side oracle.

Dataset shape (deterministic — no RNG, fully reproducible):
  * N `User` nodes, external_id `u<i>`, properties {name:"user_<i>", age, city}.
  * Out-degree 4 per node (ring + 3 fixed-stride shortcuts) → 4*N `KNOWS` edges.
    Strides {1, 7, 53, 311} are coprime-ish to typical N so the graph is
    well-connected and pattern-expansion touches a spread of pages.

Chunking: each `graph.ingest` request stays far below MAX_MESSAGE_BYTES (16 MiB);
nodes in chunks of NODE_CHUNK, rels in chunks of REL_CHUNK. All requests go in a
single JSON array (one MCP session shares one storage backend, so a later
`graph.raw_query count(n)` in the same array sees the ingested data).

Usage:
    gen_dataset.py --nodes 10000 --out-prefix /tmp/ag_perf [--tenant 1]
Writes:
    <out-prefix>_ingest.json   # JSON-RPC array: [schema, ingest..., counts]
    <out-prefix>_meta.json     # {n_nodes, n_edges, strides, ...} oracle
"""
import argparse
import json

NODE_CHUNK = 5000          # ~5000 node records per ingest call (well under 16 MiB)
REL_CHUNK = 10000          # ~10000 rel records per ingest call
STRIDES = [1, 7, 53, 311]  # out-edges per node → out-degree == len(STRIDES)
CITIES = 16                # city_0 .. city_15 (low-cardinality property)


def build_requests(n_nodes: int, tenant: int):
    n_edges = n_nodes * len(STRIDES)
    reqs = []
    rid = 0

    def nextid():
        nonlocal rid
        rid += 1
        return rid

    # 0. schema (liveness + tenant confirm) up front
    reqs.append({"jsonrpc": "2.0", "id": nextid(), "method": "graph.schema",
                 "params": {"tenant_id": tenant}})

    # 1. node chunks
    for start in range(0, n_nodes, NODE_CHUNK):
        nodes = []
        for i in range(start, min(start + NODE_CHUNK, n_nodes)):
            nodes.append({
                "external_id": f"u{i}",
                "label": "User",
                "properties": {
                    "name": f"user_{i}",
                    "age": 18 + (i % 60),
                    "city": f"city_{i % CITIES}",
                },
            })
        reqs.append({"jsonrpc": "2.0", "id": nextid(), "method": "graph.ingest",
                     "params": {"tenant_id": tenant, "nodes": nodes, "relationships": []}})

    # 2. rel chunks (ring + shortcuts). Generate the full edge list, then chunk.
    edges = []
    for i in range(n_nodes):
        for s in STRIDES:
            j = (i + s) % n_nodes
            edges.append((i, j))
    for start in range(0, len(edges), REL_CHUNK):
        rels = []
        for (i, j) in edges[start:start + REL_CHUNK]:
            rels.append({
                "from_external_id": f"u{i}",
                "to_external_id": f"u{j}",
                "rel_type": "KNOWS",
            })
        reqs.append({"jsonrpc": "2.0", "id": nextid(), "method": "graph.ingest",
                     "params": {"tenant_id": tenant, "nodes": [], "relationships": rels}})

    # 3. post-ingest verification counts (read-side oracle, same session)
    reqs.append({"jsonrpc": "2.0", "id": nextid(), "method": "graph.raw_query",
                 "params": {"tenant_id": tenant, "query": "MATCH (n:User) RETURN count(n)"}})
    reqs.append({"jsonrpc": "2.0", "id": nextid(), "method": "graph.raw_query",
                 "params": {"tenant_id": tenant,
                            "query": "MATCH (a:User)-[:KNOWS]->(b:User) RETURN count(b)"}})
    return reqs, n_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=10000)
    ap.add_argument("--tenant", type=int, default=1)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    reqs, n_edges = build_requests(args.nodes, args.tenant)
    ingest_path = f"{args.out_prefix}_ingest.json"
    meta_path = f"{args.out_prefix}_meta.json"
    with open(ingest_path, "w") as f:
        json.dump(reqs, f)
    meta = {
        "n_nodes": args.nodes,
        "n_edges": n_edges,
        "strides": STRIDES,
        "tenant": args.tenant,
        "n_ingest_requests": sum(1 for r in reqs if r["method"] == "graph.ingest"),
        "node_chunk": NODE_CHUNK,
        "rel_chunk": REL_CHUNK,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({"ingest_file": ingest_path, "meta_file": meta_path, **meta}, indent=2))


if __name__ == "__main__":
    main()
