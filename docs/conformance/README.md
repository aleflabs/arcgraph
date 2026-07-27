# Query-conformance material

ArcGraph vendors 220 openCypher TCK feature files at the upstream pin recorded
in [`../../crates/arcgraph-tck/tck/PROVENANCE.md`](../../crates/arcgraph-tck/tck/PROVENANCE.md).
The fixtures exercise parser, planner, executor, and row-diff infrastructure.

Fixture presence is not a conformance claim. The previous static scorecard
classified write scenarios as “not applicable” even though this beta executes
write clauses, so that matrix has been removed rather than relabeled.

The normative public contract for this build is
[`../arcql-reference.md`](../arcql-reference.md). It distinguishes executable
syntax, parsed-but-unimplemented syntax, and syntax absent from the grammar.
