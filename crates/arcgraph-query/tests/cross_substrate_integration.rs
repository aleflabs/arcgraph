//! Integration tests for M4-23 (M4-02c) cross-substrate validation.
//!
//! These tests exercise the public API end-to-end:
//!     `parse → BindingVisitor::bind → TypeCheckVisitor::check
//!     → CrossSubstrateValidator::validate`.
//!
//! # Substrate-availability rejection pins (4)
//!
//! Per ADR-038 amendment-03 §M4-23 row, four pins assert that the
//! validator rejects substrate-bearing surfaces when the per-tenant
//! catalog reports the substrate is not attached:
//!
//! 1. VECTOR clause without `vector_index` → SubstrateUnavailable(Vector).
//! 2. TEXT clause without `bm25_index` → SubstrateUnavailable(Bm25).
//! 3. `community(...)` call without `community_index` →
//!    SubstrateUnavailable(Community).
//! 4. Three-substrate combo (VECTOR + TEXT + community) without ANY
//!    substrates → at least one SubstrateUnavailable for each kind.
//!
//! # RANK BY HYBRID + WITH FUSION semantic-shape pins
//!
//! - HYBRID(VECTOR, VECTOR) → HybridMissingOperand("TEXT").
//! - HYBRID(TEXT, TEXT) → HybridMissingOperand("VECTOR").
//! - VECTOR(field, query) without K → HybridMissingK.
//! - WITH FUSION = RRF without k → ParseError (parser-level rejection;
//!   the bound form cannot represent missing-k).
//! - Full hybrid query with all substrates available → validates clean.
//!
//! # ADR provenance
//! - ADR-038 §2 D-23 (the M4-23 contract).
//! - ADR-038 §2 D-3 (RANK BY HYBRID shape).
//! - ADR-038 §2 D-9 (RRF k requirement).
//! - ADR-038 amendment-03 §M4-23 row (test-artifact pinning).

use arcgraph_query::parse;
use arcgraph_query::semantic::{
    ArcQLError, BindingVisitor, CrossSubstrateError, CrossSubstrateValidator, StubCatalogProvider,
    SubstrateKind, TypeCheckVisitor,
};

/// Run the full pipeline: `parse → bind → type-check → validate`.
/// Returns `Err(Vec<ArcQLError>)` when validation fails (panics on
/// parse / bind / type-check failure since those are out of scope
/// for M4-23 pins).
fn validate(input: &str, cat: &StubCatalogProvider) -> Result<(), Vec<ArcQLError>> {
    let stmt = parse(input).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, input, cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, cat).expect("type-check");
    CrossSubstrateValidator::validate(&bound, cat)
}

/// A catalog with all three substrates attached + the labels /
/// rel-types / properties used across the test inputs.
fn cat_full() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person", "Doc"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age", "name", "embedding", "content"])
        .with_vector_index()
        .with_bm25_index()
        .with_community_index()
}

/// A catalog with NO substrates attached (the substrate-rejection
/// pin baseline).
fn cat_bare() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person", "Doc"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age", "name", "embedding", "content"])
}

// =====================================================================
// Substrate-availability rejection pins (4 — per amendment-03)
// =====================================================================

#[test]
fn vector_clause_without_vector_index_is_rejected() {
    let input = "MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, $q, K = 20), TEXT(n.content, \"x\", K = 20)) WITH FUSION = RRF(k = 60) RETURN n";
    // BM25 + community attached, but NO vector index.
    let cat = StubCatalogProvider::new()
        .with_labels(["Doc"])
        .with_properties(["embedding", "content"])
        .with_bm25_index()
        .with_community_index();
    let errs = validate(input, &cat).expect_err("vector substrate missing");
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ArcQLError::CrossSubstrate(CrossSubstrateError::SubstrateUnavailable {
                kind: SubstrateKind::Vector,
                ..
            })
        )),
        "expected SubstrateUnavailable(Vector), got {errs:?}"
    );
}

#[test]
fn text_clause_without_bm25_index_is_rejected() {
    let input = "MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, $q, K = 20), TEXT(n.content, \"x\", K = 20)) WITH FUSION = RRF(k = 60) RETURN n";
    // Vector + community attached, but NO BM25 index.
    let cat = StubCatalogProvider::new()
        .with_labels(["Doc"])
        .with_properties(["embedding", "content"])
        .with_vector_index()
        .with_community_index();
    let errs = validate(input, &cat).expect_err("bm25 substrate missing");
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ArcQLError::CrossSubstrate(CrossSubstrateError::SubstrateUnavailable {
                kind: SubstrateKind::Bm25,
                ..
            })
        )),
        "expected SubstrateUnavailable(Bm25), got {errs:?}"
    );
}

#[test]
fn community_function_without_community_index_is_rejected() {
    let input = "MATCH (n:Person) RETURN community(n)";
    // Vector + BM25 attached, but NO community index.
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_vector_index()
        .with_bm25_index();
    let errs = validate(input, &cat).expect_err("community substrate missing");
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ArcQLError::CrossSubstrate(CrossSubstrateError::SubstrateUnavailable {
                kind: SubstrateKind::Community,
                ..
            })
        )),
        "expected SubstrateUnavailable(Community), got {errs:?}"
    );
}

#[test]
fn three_substrate_combo_without_any_substrates_yields_three_unavailables() {
    // A query that touches all three substrates: vector via VECTOR(...),
    // bm25 via TEXT(...), community via the IN COMMUNITY predicate.
    let input = concat!(
        "MATCH (n:Doc) WHERE n IN COMMUNITY($cid) ",
        "RANK BY HYBRID(VECTOR(n.embedding, $q, K = 20), TEXT(n.content, \"x\", K = 20)) ",
        "WITH FUSION = RRF(k = 60) ",
        "RETURN n"
    );
    let errs = validate(input, &cat_bare()).expect_err("all three substrates missing");

    let kinds: std::collections::HashSet<SubstrateKind> = errs
        .iter()
        .filter_map(|e| match e {
            ArcQLError::CrossSubstrate(CrossSubstrateError::SubstrateUnavailable {
                kind, ..
            }) => Some(*kind),
            _ => None,
        })
        .collect();

    assert!(
        kinds.contains(&SubstrateKind::Vector),
        "expected SubstrateUnavailable(Vector) in {errs:?}"
    );
    assert!(
        kinds.contains(&SubstrateKind::Bm25),
        "expected SubstrateUnavailable(Bm25) in {errs:?}"
    );
    assert!(
        kinds.contains(&SubstrateKind::Community),
        "expected SubstrateUnavailable(Community) in {errs:?}"
    );
}

// =====================================================================
// RANK BY HYBRID + WITH FUSION semantic-shape pins
// =====================================================================

#[test]
fn hybrid_with_two_vector_operands_rejects_missing_text() {
    let input = concat!(
        "MATCH (n:Doc) RANK BY HYBRID(",
        "VECTOR(n.embedding, $q1, K = 20), ",
        "VECTOR(n.embedding, $q2, K = 20)",
        ") WITH FUSION = RRF(k = 60) RETURN n"
    );
    let errs = validate(input, &cat_full()).expect_err("missing TEXT operand");
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ArcQLError::CrossSubstrate(CrossSubstrateError::HybridMissingOperand {
                kind: "TEXT",
                ..
            })
        )),
        "expected HybridMissingOperand(TEXT), got {errs:?}"
    );
    assert!(
        !errs.iter().any(|e| matches!(
            e,
            ArcQLError::CrossSubstrate(CrossSubstrateError::HybridMissingOperand {
                kind: "VECTOR",
                ..
            })
        )),
        "should NOT report missing VECTOR when two are present"
    );
}

#[test]
fn hybrid_with_two_text_operands_rejects_missing_vector() {
    let input = concat!(
        "MATCH (n:Doc) RANK BY HYBRID(",
        "TEXT(n.content, \"a\", K = 20), ",
        "TEXT(n.content, \"b\", K = 20)",
        ") WITH FUSION = RRF(k = 60) RETURN n"
    );
    let errs = validate(input, &cat_full()).expect_err("missing VECTOR operand");
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ArcQLError::CrossSubstrate(CrossSubstrateError::HybridMissingOperand {
                kind: "VECTOR",
                ..
            })
        )),
        "expected HybridMissingOperand(VECTOR), got {errs:?}"
    );
}

#[test]
fn vector_operand_without_k_is_rejected() {
    // VECTOR(field, query) with NO `K = ...`.
    let input = concat!(
        "MATCH (n:Doc) RANK BY HYBRID(",
        "VECTOR(n.embedding, $q), ",
        "TEXT(n.content, \"x\", K = 20)",
        ") WITH FUSION = RRF(k = 60) RETURN n"
    );
    let errs = validate(input, &cat_full()).expect_err("missing K on VECTOR");
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ArcQLError::CrossSubstrate(CrossSubstrateError::HybridMissingK { .. })
        )),
        "expected HybridMissingK, got {errs:?}"
    );
}

#[test]
fn fusion_rrf_without_k_is_rejected_at_parse_time() {
    // The public grammar requires `k = N` to be the first RRF argument,
    // so a weights-only form fails directly in pest before AST
    // construction. The bound form
    // `BoundFusion::Rrf { k: i64, .. }` cannot represent missing-k,
    // so `CrossSubstrateError::FusionMissingK` is defensive — it
    // exists in the taxonomy for any programmatic constructor of
    // `BoundFusion::Rrf` that bypasses the parser. This pin
    // documents that the parser rejection is the load-bearing
    // contract end-to-end.
    let input = concat!(
        "MATCH (n:Doc) RANK BY HYBRID(",
        "VECTOR(n.embedding, $q, K = 20), ",
        "TEXT(n.content, \"x\", K = 20)",
        ") WITH FUSION = RRF(weights = [0.7 vector, 0.3 text]) RETURN n"
    );
    let err = parse(input).expect_err("RRF without k must fail at parse time");
    let s = format!("{err}");
    assert!(
        s.contains("expected kw_k"),
        "expected missing-K error, got: {s}"
    );
}

#[test]
fn full_hybrid_query_with_all_substrates_validates_clean() {
    let input = concat!(
        "MATCH (n:Doc) WHERE n IN COMMUNITY($cid) ",
        "RANK BY HYBRID(VECTOR(n.embedding, $q, K = 20), TEXT(n.content, \"x\", K = 20)) ",
        "WITH FUSION = RRF(k = 60) ",
        "RETURN n LIMIT 10"
    );
    validate(input, &cat_full())
        .expect("full hybrid query with all substrates available should validate clean");
}
