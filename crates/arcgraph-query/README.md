# `arcgraph-query`

ArcQL parsing, semantic analysis, planning, and execution for the ArcGraph
`v0.1.0-beta` bare database engine.

## Responsibility

- PEG grammar and typed AST;
- variable binding, function validation, and type checking;
- logical-plan lowering and cost-based plan selection;
- batched physical execution, projection, aggregation, sorting, joins,
  traversal, and write operators;
- `EXPLAIN`/`PROFILE`, cancellation, budgets, result materialization, and
  adaptive feedback.

The executor uses provider interfaces for graph records and attached indexes.
It does not open storage files or implement network protocols.

The supported statements, clauses, expressions, and functions are listed in
[`docs/arcql-reference.md`](../../docs/arcql-reference.md). In particular,
`FOR VALID_TIME` and `AS OF` are not in this distribution's grammar. Some
reserved forms parse but return `not implemented`; the reference calls those
out explicitly.
