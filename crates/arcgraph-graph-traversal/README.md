# `arcgraph-graph-traversal`

Bounded, storage-independent traversal algorithms for the ArcGraph
`v0.1.0-beta` bare database engine.

## Public surface

- bounded k-hop expansion with cost budgets, limit pushdown, a per-hop
  frontier cap, and deterministic sampling;
- bidirectional unweighted shortest path;
- Yen k-shortest loopless paths using hop count;
- deterministic local top-k merging;
- `EdgeSource`, the adapter interface used to supply visible neighbors.

`GraphTraversalHandle` binds these algorithms to an `EdgeSource`. The caller's
adapter owns tenant and snapshot selection; this crate performs no I/O, starts
no runtime, and does not implement weighted paths.
