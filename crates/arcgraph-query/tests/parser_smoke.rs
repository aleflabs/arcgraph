//! ArcQL parser smoke tests — pin every M4-01 production.
//!
//! # Coverage discipline
//!
//! Each test pins one or more of the productions enumerated in
//! ADR-038 §2 D-1..D-10. The locked test names called out in
//! ADR-038 §3.4 ("LOCKED: per-clause regression test obligations")
//! that are PARSER-side at v1.0 appear here verbatim:
//!
//! - `parser_accepts_quantified_path_pattern_curly_at_v1_0`
//!
//! The matching `executor_returns_not_implemented_*` triplet from
//! §3.4 is M4-02 territory and is NOT pinned here.

use arcgraph_query::{
    BinOp, Clause, Expression, Fusion, LengthRange, Literal, MatchBody, NamedPathKind, ParseError,
    ProjectionKind, RankArg, Ranker, RelDirection, Statement, parse,
};

// =====================================================================
// 1. openCypher v1.0 baseline (ADR-038 D-1)
// =====================================================================

#[test]
fn parser_accepts_bare_match_return() {
    let q = parse("MATCH (n) RETURN n").expect("parse");
    let Statement::Read(r) = q else {
        panic!("expected Statement::Read");
    };
    assert_eq!(r.clauses.len(), 2);
    assert!(matches!(r.clauses[0], Clause::Match(_)));
    assert!(matches!(r.clauses[1], Clause::Return(_)));
}

#[test]
fn parser_accepts_match_with_label_props_where() {
    let q = parse("MATCH (n:Person {name: $name}) WHERE n.age > 30 RETURN n.name AS person")
        .expect("parse");
    let Statement::Read(r) = q else {
        panic!("expected Statement::Read");
    };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!("expected MATCH first");
    };
    let MatchBody::Patterns(ps) = &m.body else {
        panic!("expected pattern list");
    };
    assert_eq!(ps[0].head.var.as_deref(), Some("n"));
    assert_eq!(ps[0].head.labels, vec!["Person"]);
    assert!(ps[0].head.properties.is_some());
    assert!(m.where_clause.is_some());
}

#[test]
fn parser_accepts_directed_relationship_with_length_range_star_1_3() {
    let q = parse("MATCH (a)-[r:KNOWS*1..3]->(b) WHERE a.id = $id RETURN b").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let MatchBody::Patterns(ps) = &m.body else {
        panic!()
    };
    let (rel, _) = &ps[0].tail[0];
    assert_eq!(rel.direction, RelDirection::LeftToRight);
    assert_eq!(rel.var.as_deref(), Some("r"));
    assert_eq!(rel.rel_types, vec!["KNOWS"]);
    assert!(matches!(
        rel.length,
        Some(LengthRange::Cypher {
            min: 1,
            max: Some(3)
        })
    ));
}

fn first_relationship_length(query: &str) -> Option<LengthRange> {
    let q = parse(query).expect("parse");
    let Statement::Read(r) = q else {
        panic!("expected Statement::Read");
    };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!("expected MATCH first");
    };
    let MatchBody::Patterns(ps) = &m.body else {
        panic!("expected pattern list");
    };
    let (rel, _) = &ps[0].tail[0];
    rel.length.clone()
}

#[test]
fn parser_pins_cypher_length_range_forms() {
    assert_eq!(
        first_relationship_length("MATCH (a)-[:R*2]->(b) RETURN b"),
        Some(LengthRange::Cypher {
            min: 2,
            max: Some(2)
        })
    );
    assert_eq!(
        first_relationship_length("MATCH (a)-[:R*1]->(b) RETURN b"),
        Some(LengthRange::Cypher {
            min: 1,
            max: Some(1)
        })
    );
    assert_eq!(
        first_relationship_length("MATCH (a)-[:R*2..]->(b) RETURN b"),
        Some(LengthRange::Cypher { min: 2, max: None })
    );
    assert_eq!(
        first_relationship_length("MATCH (a)-[:R*2..5]->(b) RETURN b"),
        Some(LengthRange::Cypher {
            min: 2,
            max: Some(5)
        })
    );
    assert_eq!(
        first_relationship_length("MATCH (a)-[:R*]->(b) RETURN b"),
        Some(LengthRange::Unbounded)
    );
}

#[test]
fn parser_accepts_undirected_relationship() {
    let q = parse("MATCH (a)-[:KNOWS]-(b) RETURN a, b").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let MatchBody::Patterns(ps) = &m.body else {
        panic!()
    };
    let (rel, _) = &ps[0].tail[0];
    assert_eq!(rel.direction, RelDirection::Undirected);
}

#[test]
fn parser_accepts_right_to_left_relationship() {
    let q = parse("MATCH (a)<-[:KNOWS]-(b) RETURN a").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let MatchBody::Patterns(ps) = &m.body else {
        panic!()
    };
    let (rel, _) = &ps[0].tail[0];
    assert_eq!(rel.direction, RelDirection::RightToLeft);
}

#[test]
fn parser_accepts_with_chain_then_return() {
    // ORDER BY / SKIP / LIMIT after RETURN are emitted as
    // standalone `Clause::TailOrderBy / TailSkip / TailLimit`
    // by the v1.0 parser; the M4-02 semantic analyzer folds them
    // back into the surrounding ReturnClause. M4-01 (this slice)
    // pins only the syntactic shape.
    use arcgraph_query::Clause;
    let q = parse(
        "MATCH (n:Doc) WITH n.id AS id, n.score AS score \
         WHERE score > 0 RETURN id ORDER BY score DESC LIMIT 10 SKIP 5",
    )
    .expect("parse");
    let Statement::Read(r) = q else { panic!() };
    assert!(matches!(r.clauses[0], Clause::Match(_)));
    assert!(matches!(r.clauses[1], Clause::With(_)));
    assert!(matches!(r.clauses[2], Clause::Return(_)));
    // The standalone tail clauses appear after the RETURN.
    let has_order_by = r
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::TailOrderBy(_)));
    let has_limit = r.clauses.iter().any(|c| matches!(c, Clause::TailLimit(_)));
    let has_skip = r.clauses.iter().any(|c| matches!(c, Clause::TailSkip(_)));
    assert!(has_order_by, "expected TailOrderBy");
    assert!(has_limit, "expected TailLimit");
    assert!(has_skip, "expected TailSkip");
}

#[test]
fn parser_accepts_unwind_clause() {
    let q = parse("UNWIND $list AS x RETURN x").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Unwind(u) = &r.clauses[0] else {
        panic!()
    };
    assert_eq!(u.var, "x");
}

#[test]
fn parser_accepts_return_distinct_with_alias() {
    let q = parse("MATCH (n) RETURN DISTINCT n.name AS who LIMIT 5").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Return(rc) = &r.clauses[1] else {
        panic!()
    };
    assert!(rc.distinct);
    assert_eq!(rc.items[0].alias.as_deref(), Some("who"));
}

#[test]
fn parser_accepts_return_wildcard() {
    let q = parse("MATCH (n) RETURN *").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Return(rc) = &r.clauses[1] else {
        panic!()
    };
    assert!(matches!(rc.items[0].kind, ProjectionKind::Wildcard));
}

#[test]
fn parser_accepts_parameter_binding() {
    let q = parse("MATCH (n) WHERE n.id = $id RETURN n").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let where_expr = m.where_clause.as_ref().expect("WHERE present");
    // The expression is `n.id = $id` — after the parser, it's a
    // BinaryOp(Eq, PropertyAccess, Parameter).
    let Expression::BinaryOp { op, rhs, .. } = where_expr else {
        panic!("expected BinaryOp at WHERE root, got {where_expr:?}");
    };
    assert_eq!(*op, BinOp::Eq);
    assert!(matches!(rhs.as_ref(), Expression::Parameter(p) if p == "id"));
}

#[test]
fn parser_accepts_logical_and_or_not_in_where() {
    let q = parse("MATCH (n) WHERE NOT (n.x = 1) AND (n.y > 2 OR n.z IS NOT NULL) RETURN n")
        .expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    assert!(m.where_clause.is_some());
}

#[test]
fn parser_accepts_in_op_with_list_literal() {
    let q = parse("MATCH (n) WHERE n.id IN [1, 2, 3] RETURN n").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let where_expr = m.where_clause.as_ref().unwrap();
    assert!(matches!(where_expr, Expression::In { .. }));
}

#[test]
fn parser_accepts_string_literal_with_escapes() {
    let q = parse(r#"MATCH (n) WHERE n.text = "hello \"world\"\n" RETURN n"#).expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let where_expr = m.where_clause.as_ref().unwrap();
    let Expression::BinaryOp { rhs, .. } = where_expr else {
        panic!()
    };
    let Expression::Literal(Literal::String(s)) = rhs.as_ref() else {
        panic!()
    };
    assert_eq!(s, "hello \"world\"\n");
}

#[test]
fn parser_accepts_single_quoted_string_literal() {
    let q = parse("MATCH (n) WHERE n.x = 'abc' RETURN n").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let where_expr = m.where_clause.as_ref().unwrap();
    let Expression::BinaryOp { rhs, .. } = where_expr else {
        panic!()
    };
    assert!(matches!(rhs.as_ref(), Expression::Literal(Literal::String(s)) if s == "abc"));
}

#[test]
fn parser_accepts_float_literal() {
    let q = parse("MATCH (n) WHERE n.score = 3.14 RETURN n").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let where_expr = m.where_clause.as_ref().unwrap();
    let Expression::BinaryOp { rhs, .. } = where_expr else {
        panic!()
    };
    assert!(matches!(
        rhs.as_ref(),
        Expression::Literal(Literal::Float(_))
    ));
}

#[test]
fn parser_accepts_boolean_and_null_literals() {
    parse("MATCH (n) WHERE n.flag = TRUE AND n.other IS NULL RETURN n").expect("TRUE + IS NULL");
    parse("MATCH (n) WHERE n.flag = FALSE RETURN n").expect("FALSE");
}

#[test]
fn parser_accepts_map_literal_in_property_clause() {
    let q = parse("MATCH (n {a: 1, b: 'x'}) RETURN n").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let MatchBody::Patterns(ps) = &m.body else {
        panic!()
    };
    let pm = ps[0].head.properties.as_ref().expect("property map");
    assert_eq!(pm.entries.len(), 2);
}

#[test]
fn parser_accepts_multi_label_node() {
    let q = parse("MATCH (n:Person:Employee) RETURN n").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let MatchBody::Patterns(ps) = &m.body else {
        panic!()
    };
    assert_eq!(ps[0].head.labels, vec!["Person", "Employee"]);
}

#[test]
fn parser_accepts_multi_pattern_match() {
    let q = parse("MATCH (a)-[:KNOWS]->(b), (c) RETURN a, b, c").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let MatchBody::Patterns(ps) = &m.body else {
        panic!()
    };
    assert_eq!(ps.len(), 2);
}

// =====================================================================
// 3. ADR-038 D-3 — RANK BY HYBRID + WITH FUSION = RRF (LIT v1.0)
// =====================================================================

#[test]
fn parser_accepts_rank_by_hybrid_with_rrf_at_v1_0() {
    let q = parse(
        "MATCH (n:Doc) WHERE n.tenant_id = $tenant \
         RANK BY HYBRID( \
           VECTOR(n.embedding, $qvec, K = 20), \
           TEXT(n.content, $qtext, K = 20) \
         ) WITH FUSION = RRF(k = 60) LIMIT 10",
    )
    .expect("parse");
    let Statement::Read(r) = q else { panic!() };
    // Find the RankBy clause.
    let rank = r
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::RankBy(b) => Some(b),
            _ => None,
        })
        .expect("RankBy clause");
    let Ranker::Hybrid(args) = &rank.ranker;
    assert_eq!(args.len(), 2);
    assert!(matches!(args[0], RankArg::Vector { .. }));
    assert!(matches!(args[1], RankArg::Text { .. }));
    // Find the WithFusion clause.
    let fusion = r
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::WithFusion(f) => Some(f),
            _ => None,
        })
        .expect("WithFusion clause");
    let Fusion::Rrf { k } = &fusion.fusion;
    assert_eq!(*k, 60);
}

#[test]
fn parser_exposes_hybrid_fusion_score_with_as_binding() {
    let q = parse(
        "MATCH (n:Doc) \
         RANK BY HYBRID(\
           VECTOR(n.embedding, $qvec, K = 20), \
           TEXT(n.content, $qtext, K = 20)\
         ) AS fusion_score \
         WITH FUSION = RRF(k = 60) \
         RETURN n, fusion_score",
    )
    .expect("parse score binding");
    let Statement::Read(r) = q else { panic!() };
    let rank = r
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::RankBy(rank) => Some(rank),
            _ => None,
        })
        .expect("RankBy clause");
    assert_eq!(rank.score_alias.as_deref(), Some("fusion_score"));
    assert_eq!(
        rank.to_string(),
        "RANK BY HYBRID(VECTOR(n.embedding, $qvec, K = 20), TEXT(n.content, $qtext, K = 20)) AS fusion_score"
    );
}

#[test]
fn model_specific_rank_extensions_are_not_public_syntax() {
    for query in [
        "MODEL SEARCH $query",
        "MATCH (n) RANK BY MODEL('external') RETURN n",
        "MATCH (n) WITH FUSION = MODEL('external') RETURN n",
    ] {
        assert!(parse(query).is_err(), "unexpectedly parsed `{query}`");
    }
}

// =====================================================================
// 4. ADR-038 D-4 — community functions (LIT v1.0)
// =====================================================================

#[test]
fn parser_accepts_community_function_call_at_v1_0() {
    let q =
        parse("MATCH (n) WHERE n.community = community($seed) RETURN n LIMIT 100").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let where_expr = m.where_clause.as_ref().unwrap();
    let Expression::BinaryOp { rhs, .. } = where_expr else {
        panic!()
    };
    assert!(matches!(rhs.as_ref(), Expression::FunctionCall { name, .. } if name == "community"));
}

#[test]
fn parser_accepts_community_rank_by_seeds_function_call() {
    parse(
        "MATCH (n) WHERE community_id(n) IN community_rank_by_seeds($seeds, 0, 5) \
         RETURN n",
    )
    .expect("parse");
}

#[test]
fn parser_accepts_community_members_function_call() {
    parse("MATCH (n) WHERE n.id IN community_members($cid, 0) RETURN n").expect("parse");
}

/// ADR-038 amendment-01 alternate surface: `n IN COMMUNITY(...)`.
/// This is a non-canonical surface explicitly added per the M4-01
/// task brief; documented in
/// `docs/adr/amendments/ADR-038-amendment-01-m4-01-surface-alignment.md`.
#[test]
fn parser_accepts_in_community_predicate_per_amendment_01() {
    let q = parse("MATCH (n) WHERE n IN COMMUNITY($cid) RETURN n").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let where_expr = m.where_clause.as_ref().unwrap();
    assert!(matches!(where_expr, Expression::InCommunity { .. }));
}

// =====================================================================
// 5. ADR-038 D-5 — vector NEAR (LIT v1.0)
// =====================================================================

#[test]
fn parser_accepts_near_with_vector_index_at_v1_0() {
    let q = parse(
        "MATCH (n:Doc) WHERE n.embedding NEAR $q VECTOR_INDEX hnsw_idx \
         RETURN n LIMIT 10",
    )
    .expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let where_expr = m.where_clause.as_ref().unwrap();
    let Expression::Near { vector_index, .. } = where_expr else {
        panic!("expected Near, got {where_expr:?}");
    };
    assert_eq!(vector_index.as_deref(), Some("hnsw_idx"));
}

#[test]
fn parser_accepts_near_without_vector_index() {
    parse("MATCH (n) WHERE n.embedding NEAR $q RETURN n").expect("NEAR no idx");
}

// =====================================================================
// 6. ADR-038 D-6 — BM25 MATCH operator (LIT v1.0)
// =====================================================================

#[test]
fn parser_accepts_text_match_operator_at_v1_0() {
    let q = parse("MATCH (n:Doc) WHERE n.text MATCH $q RETURN n LIMIT 10").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let where_expr = m.where_clause.as_ref().unwrap();
    assert!(matches!(where_expr, Expression::TextMatch { .. }));
}

#[test]
fn parser_disambiguates_match_keyword_from_match_operator() {
    // Per ADR-038 D-6 — clause-MATCH only matches at statement
    // position; operator-MATCH only matches inside WHERE. This
    // query has BOTH; if the disambiguation were broken, the
    // grammar would either reject the input or mis-route the
    // operator MATCH as a clause MATCH.
    parse(
        "MATCH (n:Doc) WHERE n.body MATCH 'service unavailable' \
         AND n.severity = 'P0' RETURN n",
    )
    .expect("parse");
}

// =====================================================================
// 7. SHORTEST_PATH
// =====================================================================

#[test]
fn parser_accepts_shortest_path_at_v1_0() {
    let q = parse(
        "MATCH p = SHORTEST_PATH((a:Service {name: $a})-[:CALLS*1..3]-(b:Service {name: $b})) \
         RETURN p LIMIT 5",
    )
    .expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let MatchBody::NamedPath(np) = &m.body else {
        panic!()
    };
    assert_eq!(np.var, "p");
    assert!(matches!(np.kind, NamedPathKind::ShortestPath(_)));
}

/// LOCKED test name from ADR-038 §3.4.
#[test]
fn parser_accepts_quantified_path_pattern_curly_at_v1_0() {
    let q = parse("MATCH (a:Person)-[:KNOWS]->{1,3}(b:Person) RETURN b").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let MatchBody::Patterns(ps) = &m.body else {
        panic!()
    };
    let (rel, _) = &ps[0].tail[0];
    assert!(matches!(
        rel.length,
        Some(LengthRange::Quantified {
            min: 1,
            max: Some(3)
        })
    ));
}

// =====================================================================
// 10. Error cases — informative ParseError with span
// =====================================================================

#[test]
fn parser_rejects_malformed_property_map_with_pest_error() {
    let r = parse("MATCH (n {prop:})");
    let Err(e) = r else {
        panic!("expected error for unclosed prop value")
    };
    match e {
        ParseError::Pest { span, .. }
        | ParseError::AstConstruction {
            span: Some(span), ..
        } => {
            // Span MUST be a sensible position (not 0:0).
            assert!(
                span.start_line >= 1 && span.start_col >= 1,
                "span {span} should point past start"
            );
        }
        ParseError::AstConstruction { span: None, .. } => {
            panic!("expected span on parse error")
        }
        // A shallow malformed prop-map cannot exceed the nesting-depth
        // cap; if it somehow does, that is a regression in this test.
        ParseError::ExpressionTooDeep { .. } => {
            panic!("unexpected depth-cap error for a shallow malformed prop-map")
        }
    }
}

#[test]
fn parser_rejects_unclosed_string_literal_with_pest_error() {
    let r = parse("MATCH (n) RETURN \"unclosed");
    assert!(matches!(r, Err(ParseError::Pest { .. })));
}

#[test]
fn parser_rejects_invalid_relationship_direction() {
    // `<->` is not a valid relationship arrow in openCypher; the
    // grammar should reject.
    let r = parse("MATCH (a)<->(b) RETURN a");
    assert!(r.is_err(), "expected error for `<->`, got {r:?}");
}

#[test]
fn parser_rejects_empty_input() {
    let r = parse("");
    assert!(matches!(r, Err(ParseError::Pest { .. })));
}

#[test]
fn parser_rejects_garbage_input() {
    let r = parse("@@@ not a query @@@");
    assert!(matches!(r, Err(ParseError::Pest { .. })));
}

// =====================================================================
// 11. Round-trip smoke (the big proptest is in grammar_proptest.rs;
//     these are deterministic landmark cases)
// =====================================================================

#[test]
fn round_trip_minimal_match_return() {
    let q = parse("MATCH (n) RETURN n").unwrap();
    let printed = format!("{q}");
    let q2 = parse(&printed).expect("re-parse");
    assert_eq!(q, q2, "round-trip drift: `{printed}`");
}

#[test]
fn round_trip_hybrid_with_fusion() {
    let src = "MATCH (n:Doc) WHERE n.tenant_id = $tenant \
               RANK BY HYBRID(VECTOR(n.embedding, $qvec, K = 20), TEXT(n.content, $qtext, K = 20)) \
               WITH FUSION = RRF(k = 60) LIMIT 10";
    let q = parse(src).unwrap();
    let printed = format!("{q}");
    let q2 = parse(&printed).unwrap_or_else(|e| panic!("re-parse `{printed}` failed: {e}"));
    assert_eq!(q, q2);
}

#[test]
fn round_trip_shortest_path_named() {
    let src = "MATCH p = SHORTEST_PATH((a)-[:CALLS*1..3]-(b)) RETURN p LIMIT 5";
    let q = parse(src).unwrap();
    let printed = format!("{q}");
    let q2 = parse(&printed).expect("re-parse");
    assert_eq!(q, q2);
}

// =====================================================================
// 12. Trailing semicolon, whitespace, and comments
// =====================================================================

#[test]
fn parser_accepts_trailing_semicolon() {
    parse("MATCH (n) RETURN n;").expect("trailing `;`");
}

#[test]
fn parser_accepts_line_comment() {
    parse("-- this is a comment\nMATCH (n) RETURN n").expect("line comment");
}

#[test]
fn parser_accepts_block_comment() {
    parse("/* block */ MATCH (n) /* inline */ RETURN n /* tail */").expect("block comment");
}

#[test]
fn parser_is_keyword_case_insensitive() {
    parse("match (n) where n.x = 1 return n").expect("lowercase keywords");
    parse("MaTcH (n) ReTuRn n").expect("mixed-case keywords");
}

#[test]
fn parser_is_identifier_case_sensitive() {
    let q1 = parse("MATCH (n) RETURN n.Foo").unwrap();
    let q2 = parse("MATCH (n) RETURN n.foo").unwrap();
    assert_ne!(q1, q2, "identifiers must be case-sensitive");
}

// =====================================================================
// 13. PR #154 reviewer Finding 1 / Fix A.3 — Cypher-compatible
//     identifier handling.
//
//     Two complementary behaviors:
//       (a) lowercase property names that share spelling with a
//           reserved keyword (e.g., `n.match`) MUST parse, because
//           the keyword exclusion rule is now case-sensitive on the
//           uppercase form.
//       (b) any identifier (including the uppercase keyword form)
//           can be backtick-escaped to bypass the keyword exclusion
//           entirely — Cypher's canonical escape hatch for
//           `n.\`MATCH\`` / `n.\`order by\`` / etc.
//
//     Cypher convention is UPPERCASE keywords + lowercase property
//     names, so case-insensitive exclusion at identifier position
//     would have broken parity with Neo4j's openCypher accept set.
//     The combination of (a) + (b) restores full coverage.
// =====================================================================

// ----- (a) lowercase property names that match keywords -----------

#[test]
fn parser_accepts_lowercase_keyword_property_match() {
    parse("MATCH (n) WHERE n.match = 1 RETURN n").expect("n.match");
}

#[test]
fn parser_accepts_lowercase_keyword_property_order() {
    parse("MATCH (n) WHERE n.order = 1 RETURN n.order").expect("n.order");
}

#[test]
fn parser_accepts_lowercase_keyword_property_return() {
    parse("MATCH (n) WHERE n.return = 1 RETURN n.return").expect("n.return");
}

#[test]
fn parser_accepts_lowercase_keyword_property_where() {
    parse("MATCH (n) WHERE n.where = 1 RETURN n").expect("n.where");
}

#[test]
fn parser_accepts_lowercase_keyword_property_with() {
    parse("MATCH (n) WHERE n.with = 1 RETURN n.with").expect("n.with");
}

#[test]
fn parser_accepts_lowercase_keyword_property_null() {
    parse("MATCH (n) WHERE n.null = 1 RETURN n").expect("n.null");
}

#[test]
fn parser_accepts_lowercase_keyword_property_distinct() {
    parse("MATCH (n) WHERE n.distinct = 1 RETURN n.distinct").expect("n.distinct");
}

#[test]
fn parser_accepts_lowercase_keyword_property_skip() {
    parse("MATCH (n) WHERE n.skip = 1 RETURN n.skip").expect("n.skip");
}

#[test]
fn parser_accepts_lowercase_keyword_property_limit() {
    parse("MATCH (n) WHERE n.limit = 1 RETURN n.limit").expect("n.limit");
}

#[test]
fn parser_accepts_lowercase_keyword_property_true() {
    parse("MATCH (n) WHERE n.true = 1 RETURN n").expect("n.true");
}

#[test]
fn parser_accepts_lowercase_keyword_property_false() {
    parse("MATCH (n) WHERE n.false = 1 RETURN n").expect("n.false");
}

#[test]
fn parser_accepts_lowercase_keyword_property_from() {
    parse("MATCH (n) WHERE n.from = 1 RETURN n.from").expect("n.from");
}

#[test]
fn parser_accepts_lowercase_keyword_property_near() {
    parse("MATCH (n) WHERE n.near = 1 RETURN n").expect("n.near");
}

#[test]
fn parser_accepts_lowercase_keyword_node_var_match() {
    // Node variable named `match` (lowercase) — bare-form admissible.
    let q = parse("MATCH (match) RETURN match").expect("var=match");
    let Statement::Read(r) = q else {
        panic!("expected Statement::Read")
    };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!("expected MATCH first")
    };
    let MatchBody::Patterns(ps) = &m.body else {
        panic!("expected pattern list")
    };
    assert_eq!(ps[0].head.var.as_deref(), Some("match"));
}

#[test]
fn parser_accepts_lowercase_keyword_node_var_order() {
    let q = parse("MATCH (order) RETURN order").expect("var=order");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let MatchBody::Patterns(ps) = &m.body else {
        panic!()
    };
    assert_eq!(ps[0].head.var.as_deref(), Some("order"));
}

// ----- (b) backtick-escape — Cypher canonical escape hatch --------

#[test]
fn parser_accepts_backtick_escaped_uppercase_keyword_property() {
    // Backtick lets the user escape the uppercase keyword form too.
    let q = parse("MATCH (n) WHERE n.`MATCH` = 1 RETURN n.`MATCH`").expect("backtick MATCH");
    let Statement::Read(r) = q else {
        panic!("expected Statement::Read")
    };
    // The RETURN clause's projection should carry the inner text
    // without backticks (the AST stores the canonical identifier
    // string; backtick is purely a parser-internal escape).
    let Clause::Return(rc) = r.clauses.last().expect("return tail") else {
        panic!("expected RETURN at tail")
    };
    let ProjectionKind::Expr(Expression::PropertyAccess { path, .. }) = &rc.items[0].kind else {
        panic!("expected PropertyAccess")
    };
    assert_eq!(path, &vec!["MATCH".to_string()]);
}

#[test]
fn parser_accepts_backtick_escaped_multi_word_property() {
    // Spaces are admissible inside backticks per Cypher convention.
    parse("MATCH (n) WHERE n.`order by` = 1 RETURN n").expect("backtick `order by`");
}

#[test]
fn parser_accepts_backtick_escaped_special_chars_property() {
    // Even punctuation that would otherwise fragment the lexer is
    // admissible inside backticks (anything except a literal `).
    parse("MATCH (n) WHERE n.`MATCH (n)` = 1 RETURN n").expect("backtick punctuation");
}

// ----- (b.5) backtick-control-char rejection (codex M4-01 retro F-03) -----
//
// The backtick-escape is the canonical Cypher 9 §2.4 escape hatch for
// arbitrary identifier text, but admitting embedded newline / NUL
// bytes is an operational hazard: such identifiers can corrupt log
// lines, JSON serializers, FFI boundaries (NUL truncation), and
// catalog keys. Pre-fix grammar (`backtick_ident = ${ "`" ~ ( !"`" ~
// ANY )+ ~ "`" }`) admitted any non-backtick byte; the post-fix form
// excludes \n, \r, and \0. Codex M4-01 retro F-03 (MEDIUM, 2026-05-03).

#[test]
fn parser_rejects_backtick_with_newline() {
    let r = parse("MATCH (n) WHERE n.`hello\nworld` = 1 RETURN n");
    assert!(
        r.is_err(),
        "embedded newline inside backtick must be rejected (codex F-03), got {r:?}"
    );
}

#[test]
fn parser_rejects_backtick_with_carriage_return() {
    let r = parse("MATCH (n) WHERE n.`hello\rworld` = 1 RETURN n");
    assert!(
        r.is_err(),
        "embedded carriage return inside backtick must be rejected (codex F-03), got {r:?}"
    );
}

#[test]
fn parser_rejects_backtick_with_null_byte() {
    let r = parse("MATCH (n) WHERE n.`hello\0world` = 1 RETURN n");
    assert!(
        r.is_err(),
        "embedded NUL byte inside backtick must be rejected (codex F-03), got {r:?}"
    );
}

#[test]
fn parser_rejects_backtick_with_unicode_line_separator() {
    // U+2028 (LINE SEPARATOR) is a real line terminator in JS / JSON
    // contexts (ECMA-262 §11.3); admitting it inside identifier text
    // creates the same FFI / log-line corruption hazard as `\n`.
    // Codex M4-01 retro F-03 partial-closure follow-up (2026-05-03).
    let r = parse("MATCH (n) WHERE n.`hello\u{2028}world` = 1 RETURN n");
    assert!(
        r.is_err(),
        "embedded U+2028 LINE SEPARATOR inside backtick must be rejected (codex F-03), got {r:?}"
    );
}

#[test]
fn parser_rejects_backtick_with_unicode_paragraph_separator() {
    // U+2029 (PARAGRAPH SEPARATOR) — same rationale as U+2028: real
    // line terminator in JS / JSON contexts. Codex M4-01 retro F-03
    // partial-closure follow-up (2026-05-03).
    let r = parse("MATCH (n) WHERE n.`hello\u{2029}world` = 1 RETURN n");
    assert!(
        r.is_err(),
        "embedded U+2029 PARAGRAPH SEPARATOR inside backtick must be rejected (codex F-03), got {r:?}"
    );
}

#[test]
fn parser_admits_backtick_with_cjk() {
    // Positive pin: legitimate non-ASCII identifiers (Cypher 9 §2.4
    // explicitly admits arbitrary Unicode) must remain admissible.
    // The F-03 tightening rejects only the three control bytes.
    let q = parse("MATCH (n) WHERE n.`日本` = 1 RETURN n").expect("CJK identifier");
    let Statement::Read(r) = q else { panic!() };
    let Clause::Match(m) = &r.clauses[0] else {
        panic!()
    };
    let where_expr = m.where_clause.as_ref().expect("WHERE present");
    // Property access path should carry the CJK identifier verbatim.
    let Expression::BinaryOp { lhs, .. } = where_expr else {
        panic!("expected BinaryOp at WHERE root")
    };
    let Expression::PropertyAccess { path, .. } = lhs.as_ref() else {
        panic!("expected PropertyAccess on lhs")
    };
    assert_eq!(path, &vec!["日本".to_string()]);
}

// ----- (c) negative pins — uppercase keyword identifier rejection ----
//
// Codex M4-01 retro F-06 (MEDIUM, 2026-05-03). The (a) block above
// has 13 positive pins for `n.match` / `n.order` etc. + 3 backtick
// positive pins for `` n.`MATCH` ``. Zero NEGATIVE pins for
// uppercase rejection — a one-line revert to `^"MATCH"` in
// grammar.pest would silently regress behavior in the OPPOSITE
// direction (rejecting `n.match` again) without any test failure
// catching it.
//
// These pins lock the rejection behavior for `n.MATCH` (uppercase
// keyword as property without backtick) and `MATCH` as a variable
// name. Any future relaxation to admit either case-insensitively
// requires an ADR amendment (see ADR-038 amendment-04 / codex F-04
// for the Cypher 9 §2.5 conformance divergence documentation).

#[test]
fn parser_rejects_uppercase_keyword_property_without_backtick() {
    // Cypher 9 §2.5 admits `n.MATCH` since keywords are case-
    // insensitive AND ArcQL excludes them only at uppercase form;
    // ArcQL therefore requires backticks for the uppercase keyword
    // form. Reverting to case-insensitive `^"MATCH"` exclusion would
    // make this case PARSE again (regressing in the opposite
    // direction — rejecting `n.match`).
    let r = parse("MATCH (n) RETURN n.MATCH");
    assert!(
        r.is_err(),
        "uppercase keyword `MATCH` as property without backtick must be rejected (codex F-06), got {r:?}"
    );
}

#[test]
fn parser_rejects_uppercase_keyword_as_variable_name() {
    // Same rationale: ArcQL's case-sensitive uppercase keyword
    // exclusion rejects `MATCH` at variable position. Users who need
    // the uppercase form must backtick-escape.
    let r = parse("MATCH (MATCH) RETURN MATCH");
    assert!(
        r.is_err(),
        "uppercase keyword `MATCH` as variable name must be rejected (codex F-06), got {r:?}"
    );
}

// ----- (d) positive pin — mid-case keyword admission (Cypher 9 §2.5 divergence) ----
//
// Codex M4-01 retro F-06 partial-closure follow-up (2026-05-03). The
// (a) + (c) pins above lock the lowercase-admit + uppercase-reject
// behaviors, but ArcQL's keyword exclusion is case-SENSITIVE on the
// uppercase form (per Fix A.2 / PR #154 reviewer Finding 1). That
// means mid-case forms (`Match`, `MaTcH`) — which Cypher 9 §2.5
// would treat as keywords (case-insensitive) — are admitted as
// IDENTIFIERS in ArcQL. This is a documented divergence per ADR-038
// amendment-04 §D-X.1; v1.1 conformance fix tracked as issue #189.
// Without a positive pin, the divergence could silently disappear
// (e.g., a future revert to `^"MATCH"` would re-enable rejection of
// mid-case identifiers), and downstream layers that rely on the
// admission would break without test failure.

#[test]
fn parser_admits_midcase_keyword_as_variable_per_cypher9_2_5_divergence() {
    // ArcQL v1.0 admits mid-case keyword identifiers (`Match`, `MaTcH`)
    // per ADR-038 amendment-04 §D-X.1 — documented Cypher 9 §2.5
    // divergence. v1.1 conformance fix tracked as issue #189. This pin
    // proves the divergence STILL HOLDS at parse-time so downstream
    // type-check / projection layers can rely on the parser-level
    // admission (and v1.1 conformance work knows what to flip).
    let r = parse("MATCH (Match) RETURN Match");
    assert!(
        r.is_ok(),
        "expected mid-case keyword `Match` to parse as variable per ADR-038 amendment-04 §D-X.1, got {r:?}"
    );
}

// =====================================================================
// 14. Hybrid text/vector search syntax smoke pin.
// =====================================================================

#[test]
fn smoke_hybrid_rank_query_v1_0() {
    let query = r#"
        MATCH (n:Document) WHERE n.tenant_id = $tenant
        RANK BY HYBRID(
          VECTOR(n.embedding, $qvec, K = 20),
          TEXT(n.content, $qtext, K = 20)
        ) WITH FUSION = RRF(k = 60)
        LIMIT 10
    "#;

    let parsed = parse(query).expect("hybrid rank query must parse");
    // Light shape assertion: top-level Statement::Read with a
    // MATCH first, RANK BY HYBRID + WITH FUSION = RRF in the
    // middle, and a tail LIMIT. The load-bearing claim is "this
    // parses end-to-end"; the AST shape assertion documents the
    // expected clause sequencing without over-pinning the
    // ProjectionItem / RankArg internal layout.
    let Statement::Read(r) = parsed else {
        panic!("expected Statement::Read for hybrid rank query")
    };
    assert!(
        matches!(r.clauses[0], Clause::Match(_)),
        "first clause should be MATCH (anchor)"
    );
    assert!(
        r.clauses.iter().any(|c| matches!(c, Clause::RankBy(_))),
        "expected RANK BY HYBRID clause"
    );
    assert!(
        r.clauses.iter().any(|c| matches!(c, Clause::WithFusion(_))),
        "expected WITH FUSION = RRF clause"
    );
    assert!(
        r.clauses.iter().any(|c| matches!(c, Clause::TailLimit(_))),
        "expected trailing LIMIT clause"
    );
}

// =====================================================================
// 13. OPTIONAL MATCH (added M4-22 per ADR-006 amendment-01 +
//     ADR-038 amendment-03 §TIER-1 GAP D)
// =====================================================================

#[test]
fn parser_accepts_optional_match_simple() {
    let q = parse("OPTIONAL MATCH (n:Person) RETURN n").expect("parse");
    let Statement::Read(r) = q else {
        panic!("expected Statement::Read");
    };
    assert_eq!(r.clauses.len(), 2);
    assert!(
        matches!(r.clauses[0], Clause::OptionalMatch(_)),
        "expected OPTIONAL MATCH first"
    );
    assert!(matches!(r.clauses[1], Clause::Return(_)));
}

#[test]
fn parser_accepts_match_then_optional_match() {
    let q = parse("MATCH (a) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a, b").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    assert!(matches!(r.clauses[0], Clause::Match(_)));
    assert!(matches!(r.clauses[1], Clause::OptionalMatch(_)));
    assert!(matches!(r.clauses[2], Clause::Return(_)));
}

#[test]
fn parser_accepts_optional_match_with_where() {
    let q = parse("OPTIONAL MATCH (n) WHERE n.age > 30 RETURN n").expect("parse");
    let Statement::Read(r) = q else { panic!() };
    let Clause::OptionalMatch(m) = &r.clauses[0] else {
        panic!("expected OPTIONAL MATCH first");
    };
    assert!(m.where_clause.is_some(), "WHERE in OPTIONAL MATCH");
}

#[test]
fn parser_accepts_ldbc_is6_shape_optional_has_creator() {
    // LDBC SNB Interactive Short IS6-shape representative — match
    // a message, OPTIONAL MATCH its creator (which may be a deleted /
    // anonymous node, hence OPTIONAL).
    let src = "MATCH (m:Message {id: $messageId}) \
               OPTIONAL MATCH (m)-[:HAS_CREATOR]->(p:Person) \
               RETURN m, p";
    let q = parse(src).expect("parse");
    let Statement::Read(r) = q else { panic!() };
    assert!(matches!(r.clauses[0], Clause::Match(_)));
    assert!(matches!(r.clauses[1], Clause::OptionalMatch(_)));
    assert!(matches!(r.clauses[2], Clause::Return(_)));
}

#[test]
fn round_trip_optional_match_preserves_keyword() {
    let src = "OPTIONAL MATCH (n:Person) RETURN n";
    let q = parse(src).unwrap();
    let printed = format!("{q}");
    assert!(printed.contains("OPTIONAL"), "{printed:?}");
    let q2 = parse(&printed).expect("re-parse");
    assert_eq!(q, q2);
}
