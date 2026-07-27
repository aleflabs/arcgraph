//! Error taxonomy for the BM25 search-side surface (ADR-039 §D-8).
//!
//! `Bm25Error` is the handle-side error returned from
//! [`crate::Bm25IndexHandle`] methods (search, upsert, delete) and the
//! service-side [`crate::Bm25Service::handle`] opener. The
//! commit-pipeline error [`arcgraph_storage::mutation_log::Bm25StoreError`]
//! is intentionally separate (lives in `arcgraph-storage`) so the
//! kernel commit closure can hold the trait object without taking a
//! tantivy dependency.
//!
//! # `From` impls
//!
//! - `From<tantivy::TantivyError>` translates Tantivy errors into the
//!   `Tantivy { message }` variant. The original error type is
//!   eagerly stringified; consumers above the storage layer should
//!   not need to match on Tantivy's internal taxonomy.
//! - `From<std::io::Error>` translates filesystem errors raised by
//!   the per-tenant directory open path into the `Io { message }`
//!   variant.

use thiserror::Error;

/// Failure modes for the BM25 search-side handle (ADR-039 §D-8).
///
/// Reserved for v1.1+ (annotated; not added at v1.0):
/// - `TimeTravelNotSupportedAtV1` — when ADR-007's lift to time-
///   travel queries lands.
/// - `PerPropertyIndexNotSupportedAtV1` — when M7 / v1.1 DDL adds
///   `CREATE TEXT INDEX ON <Label>(<property>)`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Bm25Error {
    /// A `tantivy::TantivyError` surfaced from the underlying engine.
    /// Original error is rendered to a string at translation time.
    #[error("tantivy error: {message}")]
    Tantivy {
        /// Rendered `tantivy::TantivyError` text.
        message: String,
    },

    /// User query failed to parse against Tantivy's QueryParser. v1.0
    /// uses the default English-leaning parser; per-language parsers
    /// are M7 / v1.1 scope per ADR-039 §D-2.
    #[error("query parse error: {message}")]
    QueryParse {
        /// Rendered parser error text.
        message: String,
    },

    /// `filtered_search` was called with a `Filter` variant that v1.0
    /// BM25 does not support. v1.0 supports only `Filter::Any`
    /// (effectively unfiltered); other variants require label /
    /// tenant FAST fields not present in the v1.0 schema (ADR-039
    /// §D-2). Carried as a `String` so the variant taxonomy is
    /// future-proof against ADR-035 amendment-04 evolution.
    #[error("filter not supported by BM25 at v1.0: {variant}")]
    FilterNotSupported {
        /// `format!("{:?}", filter)` of the rejected variant.
        variant: String,
    },

    /// A document violated the v1.0 schema invariant — e.g.
    /// `expired_lsn != Lsn::MAX` on a freshly-upserted doc. Surfaces
    /// only for in-process invariants; on-disk segments produced by
    /// future v1.1+ writers may legitimately carry non-MAX
    /// `expired_lsn`, and v1.1's Bm25Error taxonomy lifts this
    /// variant to `SchemaVersionMismatch` per ADR-039 §D-3.
    #[error("schema violation: {detail}")]
    SchemaViolation {
        /// Human-readable schema invariant violation summary.
        detail: String,
    },

    /// Filesystem error opening / creating a per-tenant Tantivy
    /// directory. Surfaces from `Bm25Service::handle` on first-touch
    /// directory creation. Subsequent calls hit the in-memory cache
    /// and do not surface this variant.
    #[error("io error opening tantivy index: {message}")]
    Io {
        /// Rendered `std::io::Error` text.
        message: String,
    },
}

impl From<tantivy::TantivyError> for Bm25Error {
    fn from(e: tantivy::TantivyError) -> Self {
        Self::Tantivy {
            message: e.to_string(),
        }
    }
}

impl From<std::io::Error> for Bm25Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io {
            message: e.to_string(),
        }
    }
}

impl From<tantivy::directory::error::OpenDirectoryError> for Bm25Error {
    fn from(e: tantivy::directory::error::OpenDirectoryError) -> Self {
        // OpenDirectoryError is the path-not-found / non-dir / I/O
        // failure family for `MmapDirectory::open`. Surface as `Io`
        // so consumers don't need to special-case the directory open
        // path.
        Self::Io {
            message: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tantivy_error_renders_message() {
        let err = Bm25Error::Tantivy {
            message: "writer poisoned".into(),
        };
        let rendered = err.to_string();
        assert!(rendered.contains("writer poisoned"), "{rendered}");
        assert!(rendered.contains("tantivy"), "{rendered}");
    }

    #[test]
    fn filter_not_supported_renders_variant() {
        let err = Bm25Error::FilterNotSupported {
            variant: "Label(Person)".into(),
        };
        let rendered = err.to_string();
        assert!(rendered.contains("Label(Person)"), "{rendered}");
        assert!(rendered.contains("v1.0"), "{rendered}");
    }

    #[test]
    fn from_io_error_routes_to_io_variant() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing dir");
        let err: Bm25Error = io_err.into();
        match err {
            Bm25Error::Io { message } => {
                assert!(message.contains("missing dir"), "{message}");
            }
            other => panic!("expected Io variant, got {other:?}"),
        }
    }

    #[test]
    fn schema_violation_carries_detail() {
        let err = Bm25Error::SchemaViolation {
            detail: "expired_lsn != MAX".into(),
        };
        let rendered = err.to_string();
        assert!(rendered.contains("expired_lsn"), "{rendered}");
    }

    #[test]
    fn query_parse_renders_message() {
        let err = Bm25Error::QueryParse {
            message: "unbalanced bracket".into(),
        };
        let rendered = err.to_string();
        assert!(rendered.contains("unbalanced bracket"), "{rendered}");
    }
}
