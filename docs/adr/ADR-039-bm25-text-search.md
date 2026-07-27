# ADR-039: Tenant-local BM25 text search

- **Status:** Accepted for `v0.1.0-beta`

The retained BM25 crate uses Tantivy indexes derived from committed node
properties. The served convention indexes the string property `text`; there is
no text-index DDL and no text-generation service.

Candidates are tenant-local and are rechecked through storage and principal
ACL visibility. Request shapes and verified output are in
[`../search.md`](../search.md).
