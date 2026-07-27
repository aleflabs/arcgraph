//! Tantivy schema for v1.0 BM25 (ADR-039 §D-2).
//!
//! The schema is **fixed at v1.0**: four fields (`node_id`,
//! `commit_lsn`, `expired_lsn`, `body`) shared across every per-tenant
//! `tantivy::Index`. Per-property and per-language indexes are M7 /
//! v1.1 scope per ADR-036 §D-2; the v1.0 schema is a single `body`
//! TEXT field plus the three MVCC fast-fields.
//!
//! # MVCC schema invariants (ADR-039 §D-3)
//!
//! - Every produced live doc has `expired_lsn == Lsn::MAX`. The
//!   second clause of the visibility filter (`expired_lsn >
//!   read_lsn`) is therefore true for a visible live document.
//! - At most one live doc exists per `node_id`. The upsert path is
//!   `delete_term + add_document`.

use tantivy::schema::{FAST, Field, INDEXED, STORED, Schema, SchemaBuilder, TEXT};

/// The v1.0 BM25 schema (ADR-039 §D-2). Constructed via
/// [`Self::build`]; rebuilt each `Bm25Service::new()` so the field
/// handles are fresh per service instance — Tantivy's `Field`
/// identifiers are positional within a `SchemaBuilder` and reusing a
/// schema across `Index::open_or_create_in_dir` calls is safe so long
/// as every index in the workspace uses the SAME schema.
#[derive(Debug, Clone)]
pub struct Bm25Schema {
    /// Underlying Tantivy schema. Held alongside the field handles so
    /// the `IndexWriter` / `IndexReader` constructors can request it.
    pub schema: Schema,

    /// Primary-key field (`u64`; `FAST | INDEXED | STORED`). The
    /// `INDEXED` flag is load-bearing — it enables `delete_term` for
    /// the upsert hard-delete path (ADR-039 §D-3). `STORED` lets the
    /// search retriever pull the `node_id` back from the matched
    /// doc; `FAST` is reserved for v1.1 sort-by-id. Mirrors the
    /// `(node_id_u64)` shape consumed by the F.4 dispatcher.
    pub node_id: Field,

    /// MVCC visibility lower bound (`u64`; `FAST | INDEXED | STORED`).
    /// Queryable via `RangeQuery`; the `FAST` flag enables
    /// constant-time per-doc reads inside the visibility filter
    /// (ADR-039 §D-3).
    pub commit_lsn: Field,

    /// MVCC visibility upper bound (`u64`; `FAST | INDEXED | STORED`).
    /// **At v1.0 every doc has `expired_lsn == Lsn::MAX = u64::MAX`**
    /// (ADR-039 §D-2 invariant). The `RangeQuery` clause that filters
    /// on `expired_lsn > read_lsn` is therefore trivially true at
    /// v1.0; the clause still composes for v1.1 forward-compat.
    pub expired_lsn: Field,

    /// Free-text body (`TEXT | STORED`, default tokenizer). v1.0
    /// supports a single body field per doc; per-property indexes
    /// land at M7 / v1.1.
    pub body: Field,
}

impl Bm25Schema {
    /// Build the v1.0 schema. Calls
    /// [`SchemaBuilder::add_u64_field`] / `add_text_field` with the
    /// fixed flag set documented per-field above.
    ///
    /// The schema is identical across every tenant so callers can
    /// build it once at `Bm25Service::new` time and reuse the
    /// resulting `Bm25Schema` for every per-tenant index.
    #[must_use]
    pub fn build() -> Self {
        let mut builder = SchemaBuilder::new();
        // ADR-039 §D-2: u64 fast+indexed+stored on all three MVCC /
        // PK fields. `INDEXED` is required for `delete_term` /
        // RangeQuery; `FAST` enables per-doc reads in the visibility
        // filter; `STORED` lets `search` round-trip the node_id back
        // out of matched docs.
        let node_id = builder.add_u64_field("node_id", FAST | INDEXED | STORED);
        let commit_lsn = builder.add_u64_field("commit_lsn", FAST | INDEXED | STORED);
        let expired_lsn = builder.add_u64_field("expired_lsn", FAST | INDEXED | STORED);
        // ADR-039 §D-2: TEXT triggers default tokenizer; STORED
        // lets re-rankers pull body bytes back out of matches.
        let body = builder.add_text_field("body", TEXT | STORED);
        let schema = builder.build();
        Self {
            schema,
            node_id,
            commit_lsn,
            expired_lsn,
            body,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schema must always have exactly four fields with the
    /// names rustdoc'd above. A field added or removed without an
    /// ADR-039 amendment surfaces here.
    #[test]
    fn schema_has_four_v1_fields() {
        let s = Bm25Schema::build();
        // Tantivy `Schema` doesn't expose a `len()` directly; iterate
        // over `fields()` to count.
        let count = s.schema.fields().count();
        assert_eq!(count, 4, "v1.0 schema must have exactly 4 fields");
    }

    #[test]
    fn schema_field_names_are_pinned() {
        let s = Bm25Schema::build();
        let names: Vec<String> = s
            .schema
            .fields()
            .map(|(_, entry)| entry.name().to_string())
            .collect();
        assert!(names.iter().any(|n| n == "node_id"), "{names:?}");
        assert!(names.iter().any(|n| n == "commit_lsn"), "{names:?}");
        assert!(names.iter().any(|n| n == "expired_lsn"), "{names:?}");
        assert!(names.iter().any(|n| n == "body"), "{names:?}");
    }

    #[test]
    fn schema_field_handles_are_distinct() {
        let s = Bm25Schema::build();
        // Field equality is by inner u32; the four handles must be
        // pairwise distinct so a `delete_term(node_id)` is not
        // accidentally a `delete_term(commit_lsn)`.
        assert_ne!(s.node_id, s.commit_lsn);
        assert_ne!(s.node_id, s.expired_lsn);
        assert_ne!(s.node_id, s.body);
        assert_ne!(s.commit_lsn, s.expired_lsn);
        assert_ne!(s.commit_lsn, s.body);
        assert_ne!(s.expired_lsn, s.body);
    }

    /// Two builds of the same schema produce structurally equal
    /// schemas. Pin: `Bm25Service::new` rebuilds per service; the
    /// resulting per-tenant `Index`es must use the same schema for
    /// cross-tenant Tantivy compatibility.
    #[test]
    fn rebuild_produces_equal_schema() {
        let a = Bm25Schema::build();
        let b = Bm25Schema::build();
        // Field handles are positional and stable across calls.
        assert_eq!(a.node_id, b.node_id);
        assert_eq!(a.commit_lsn, b.commit_lsn);
        assert_eq!(a.expired_lsn, b.expired_lsn);
        assert_eq!(a.body, b.body);
    }
}
