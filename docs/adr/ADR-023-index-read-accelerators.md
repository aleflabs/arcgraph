# ADR-023: Indexes are read accelerators

- **Status:** Accepted for `v0.1.0-beta`

Primary, secondary-property, vector, BM25, and community indexes narrow
candidate sets. They are not visibility authorities.

Before a served read returns an indexed candidate, the database hydrates it
through tenant-scoped storage and re-applies transaction-snapshot and ACL
visibility. A missing or stale index entry can affect which fast path is used;
it must not make a deleted, unauthorized, or cross-tenant record visible.

Durable graph records and WAL recovery remain the source of truth.
