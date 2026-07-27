# ADR-018: Current MVCC record visibility

- **Status:** Accepted for `v0.1.0-beta`

Node, relationship, and adjacency records carry the LSN fields needed for
current transaction-snapshot visibility. Storage applies those fields before
returning a hydrated record, and every index hit is subject to the same check.

This is not a bi-temporal contract. The public ArcQL grammar has no
`FOR VALID_TIME` or `AS OF` qualifier, and the server does not expose a
historical-snapshot query API. The exact record predicates are documented in
[`../records-semantics.md`](../records-semantics.md).
