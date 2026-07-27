//! Curated subset of openCypher TCK-shaped queries for the W18δ
//! dual-execute smoke (addendum item 4 — "≥10 from each of {MATCH-
//! WHERE-RETURN, OPTIONAL MATCH, ORDER BY + LIMIT, 3VL NULL semantics,
//! aggregation}").
//!
//! Each query is a self-contained Cypher string against the fixture
//! seeded by [`crate::executor::ArcGraphExecutor::new`] +
//! [`crate::executor::Neo4jOracleExecutor`]'s setup (the env-gated
//! Docker neo4j is seeded by the dual-execute test before each run).
//!
//! # v1.0-alpha capability envelope
//!
//! Most queries pass on Neo4j; many surface a `PlanBuild` / `Execution`
//! error on ArcGraph (W17α executor capability gaps documented in
//! `docs/migration/from-neo4j.md` §"v1.0-alpha capability gaps"). The
//! dual-execute test labels each per-query outcome:
//!
//! - **Diff** — both executors returned row-sets that match.
//! - **Diverged** — both executors returned row-sets that DIFFER →
//!   ArcGraph-side gap or executor bug.
//! - **ArcGraphOnly** — ArcGraph returned rows but oracle errored.
//! - **OracleOnly** — oracle returned rows but ArcGraph errored.
//! - **BothErrored** — neither executor produced rows; counted but
//!   not asserted.
//!
//! The harness reports per-category pass/diverge/error counts so a CI
//! regression in any category surfaces at-a-glance.

/// Category taxonomy. Each value names one of the W18δ-addendum
/// "≥10 each from" buckets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CuratedCategory {
    /// MATCH-WHERE-RETURN (core read shape).
    MatchWhereReturn,
    /// OPTIONAL MATCH (left-outer-join semantics).
    OptionalMatch,
    /// ORDER BY + LIMIT (sort + paginate).
    OrderByLimit,
    /// 3VL NULL semantics (IS NULL / IS NOT NULL / null-propagating
    /// comparisons).
    NullSemantics,
    /// Aggregation (COUNT, SUM, AVG, MIN, MAX, COLLECT).
    Aggregation,
}

impl CuratedCategory {
    /// Short label for log output.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            CuratedCategory::MatchWhereReturn => "match_where_return",
            CuratedCategory::OptionalMatch => "optional_match",
            CuratedCategory::OrderByLimit => "order_by_limit",
            CuratedCategory::NullSemantics => "null_semantics",
            CuratedCategory::Aggregation => "aggregation",
        }
    }
}

/// One curated query.
#[derive(Debug, Clone, Copy)]
pub struct CuratedQuery {
    /// Stable short name. Used as the per-query report key.
    pub name: &'static str,
    /// Which W18δ-addendum bucket this query belongs to.
    pub category: CuratedCategory,
    /// The cypher text.
    pub cypher: &'static str,
    /// Whether row order is load-bearing (true → ORDER BY queries).
    pub ordered: bool,
}

/// Per-category minimum query count (W18δ addendum item 4: "≥10 from
/// each of ...").
pub const CURATED_PER_CATEGORY: usize = 10;

// ─────────────────────────────────────────────────────────────────────
// MATCH-WHERE-RETURN — 10
// ─────────────────────────────────────────────────────────────────────

macro_rules! q {
    ($name:expr, $cat:expr, $cypher:expr, $ord:expr) => {
        CuratedQuery {
            name: $name,
            category: $cat,
            cypher: $cypher,
            ordered: $ord,
        }
    };
}

const MATCH_WHERE_RETURN: [CuratedQuery; 10] = [
    q!(
        "mwr_01_scan_label",
        CuratedCategory::MatchWhereReturn,
        "MATCH (n:Person) RETURN n.name",
        false
    ),
    q!(
        "mwr_02_filter_eq",
        CuratedCategory::MatchWhereReturn,
        "MATCH (n:Person) WHERE n.age = 30 RETURN n.name",
        false
    ),
    q!(
        "mwr_03_filter_gt",
        CuratedCategory::MatchWhereReturn,
        "MATCH (n:Person) WHERE n.age > 30 RETURN n.name",
        false
    ),
    q!(
        "mwr_04_filter_lt",
        CuratedCategory::MatchWhereReturn,
        "MATCH (n:Person) WHERE n.age < 40 RETURN n.name",
        false
    ),
    q!(
        "mwr_05_filter_ne",
        CuratedCategory::MatchWhereReturn,
        "MATCH (n:Person) WHERE n.age <> 30 RETURN n.name",
        false
    ),
    q!(
        "mwr_06_filter_and",
        CuratedCategory::MatchWhereReturn,
        "MATCH (n:Person) WHERE n.age > 25 AND n.age < 45 RETURN n.name",
        false
    ),
    q!(
        "mwr_07_filter_or",
        CuratedCategory::MatchWhereReturn,
        "MATCH (n:Person) WHERE n.age = 25 OR n.age = 45 RETURN n.name",
        false
    ),
    q!(
        "mwr_08_filter_not",
        CuratedCategory::MatchWhereReturn,
        "MATCH (n:Person) WHERE NOT n.age = 30 RETURN n.name",
        false
    ),
    q!(
        "mwr_09_filter_string_eq",
        CuratedCategory::MatchWhereReturn,
        "MATCH (n:Person) WHERE n.name = 'Alice' RETURN n.age",
        false
    ),
    q!(
        "mwr_10_two_labels",
        CuratedCategory::MatchWhereReturn,
        "MATCH (n:Person), (m:Doc) RETURN n.name, m.title",
        false
    ),
];

// ─────────────────────────────────────────────────────────────────────
// OPTIONAL MATCH — 10
// ─────────────────────────────────────────────────────────────────────

const OPTIONAL_MATCH: [CuratedQuery; 10] = [
    q!(
        "opt_01_basic",
        CuratedCategory::OptionalMatch,
        "MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(m) RETURN n.name, m.name",
        false
    ),
    q!(
        "opt_02_chained",
        CuratedCategory::OptionalMatch,
        "MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(m) OPTIONAL MATCH (m)-[:KNOWS]->(o) RETURN n.name, m.name, o.name",
        false
    ),
    q!(
        "opt_03_with_filter",
        CuratedCategory::OptionalMatch,
        "MATCH (n:Person) WHERE n.age > 25 OPTIONAL MATCH (n)-[:KNOWS]->(m) RETURN n.name, m.name",
        false
    ),
    q!(
        "opt_04_after_match",
        CuratedCategory::OptionalMatch,
        "MATCH (a:Person)-[:KNOWS]->(b:Person) OPTIONAL MATCH (b)-[:KNOWS]->(c:Person) RETURN a.name, b.name, c.name",
        false
    ),
    q!(
        "opt_05_label_filter",
        CuratedCategory::OptionalMatch,
        "MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(m:Person) RETURN n.name, m.name",
        false
    ),
    q!(
        "opt_06_returns_null",
        CuratedCategory::OptionalMatch,
        "MATCH (n:Doc) OPTIONAL MATCH (n)-[:KNOWS]->(m) RETURN n.title, m.name",
        false
    ),
    q!(
        "opt_07_multi_relations",
        CuratedCategory::OptionalMatch,
        "MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(m:Person) OPTIONAL MATCH (m)-[:KNOWS]->(o:Person) RETURN n.name, m.name, o.name",
        false
    ),
    q!(
        "opt_08_with_where",
        CuratedCategory::OptionalMatch,
        "MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(m) WHERE m.age > 35 RETURN n.name, m.name",
        false
    ),
    q!(
        "opt_09_no_relation",
        CuratedCategory::OptionalMatch,
        "OPTIONAL MATCH (n:NoSuchLabel) RETURN n",
        false
    ),
    q!(
        "opt_10_property_access",
        CuratedCategory::OptionalMatch,
        "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.age",
        false
    ),
];

// ─────────────────────────────────────────────────────────────────────
// ORDER BY + LIMIT — 10
// ─────────────────────────────────────────────────────────────────────

const ORDER_BY_LIMIT: [CuratedQuery; 10] = [
    q!(
        "obl_01_basic_asc",
        CuratedCategory::OrderByLimit,
        "MATCH (n:Person) RETURN n.name ORDER BY n.age ASC",
        true
    ),
    q!(
        "obl_02_basic_desc",
        CuratedCategory::OrderByLimit,
        "MATCH (n:Person) RETURN n.name ORDER BY n.age DESC",
        true
    ),
    q!(
        "obl_03_limit_1",
        CuratedCategory::OrderByLimit,
        "MATCH (n:Person) RETURN n.name ORDER BY n.age ASC LIMIT 1",
        true
    ),
    q!(
        "obl_04_limit_3",
        CuratedCategory::OrderByLimit,
        "MATCH (n:Person) RETURN n.name ORDER BY n.age DESC LIMIT 3",
        true
    ),
    q!(
        "obl_05_skip_then_limit",
        CuratedCategory::OrderByLimit,
        "MATCH (n:Person) RETURN n.name ORDER BY n.age ASC SKIP 1 LIMIT 2",
        true
    ),
    q!(
        "obl_06_order_by_string",
        CuratedCategory::OrderByLimit,
        "MATCH (n:Person) RETURN n.name ORDER BY n.name ASC",
        true
    ),
    q!(
        "obl_07_order_by_multi",
        CuratedCategory::OrderByLimit,
        "MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age ASC, n.name DESC",
        true
    ),
    q!(
        "obl_08_limit_large",
        CuratedCategory::OrderByLimit,
        "MATCH (n:Person) RETURN n.name ORDER BY n.age ASC LIMIT 100",
        true
    ),
    q!(
        "obl_09_order_with_where",
        CuratedCategory::OrderByLimit,
        "MATCH (n:Person) WHERE n.age > 25 RETURN n.name ORDER BY n.age DESC",
        true
    ),
    q!(
        "obl_10_limit_zero",
        CuratedCategory::OrderByLimit,
        "MATCH (n:Person) RETURN n.name ORDER BY n.age ASC LIMIT 0",
        true
    ),
];

// ─────────────────────────────────────────────────────────────────────
// 3VL NULL semantics — 10
// ─────────────────────────────────────────────────────────────────────

const NULL_SEMANTICS: [CuratedQuery; 10] = [
    q!(
        "null_01_is_null",
        CuratedCategory::NullSemantics,
        "MATCH (n:Doc) WHERE n.age IS NULL RETURN n.title",
        false
    ),
    q!(
        "null_02_is_not_null",
        CuratedCategory::NullSemantics,
        "MATCH (n:Person) WHERE n.age IS NOT NULL RETURN n.name",
        false
    ),
    q!(
        "null_03_eq_null",
        CuratedCategory::NullSemantics,
        "MATCH (n:Doc) WHERE n.age = NULL RETURN n.title",
        false
    ),
    q!(
        "null_04_ne_null",
        CuratedCategory::NullSemantics,
        "MATCH (n:Person) WHERE n.age <> NULL RETURN n.name",
        false
    ),
    q!(
        "null_05_null_arith",
        CuratedCategory::NullSemantics,
        "MATCH (n:Doc) RETURN n.age + 1",
        false
    ),
    q!(
        "null_06_null_in_and",
        CuratedCategory::NullSemantics,
        "MATCH (n:Person) WHERE n.age > 0 AND n.bogus IS NULL RETURN n.name",
        false
    ),
    q!(
        "null_07_null_in_or",
        CuratedCategory::NullSemantics,
        "MATCH (n:Person) WHERE n.age > 100 OR n.bogus IS NULL RETURN n.name",
        false
    ),
    q!(
        "null_08_null_in_not",
        CuratedCategory::NullSemantics,
        "MATCH (n:Person) WHERE NOT (n.bogus IS NULL) RETURN n.name",
        false
    ),
    q!(
        "null_09_coalesce",
        CuratedCategory::NullSemantics,
        "MATCH (n:Person) RETURN coalesce(n.bogus, 'unknown')",
        false
    ),
    q!(
        "null_10_return_null",
        CuratedCategory::NullSemantics,
        "MATCH (n:Doc) RETURN n.title, n.bogus_field",
        false
    ),
];

// ─────────────────────────────────────────────────────────────────────
// Aggregation — 10
// ─────────────────────────────────────────────────────────────────────

const AGGREGATION: [CuratedQuery; 10] = [
    q!(
        "agg_01_count_star",
        CuratedCategory::Aggregation,
        "MATCH (n:Person) RETURN count(*)",
        false
    ),
    q!(
        "agg_02_count_var",
        CuratedCategory::Aggregation,
        "MATCH (n:Person) RETURN count(n)",
        false
    ),
    q!(
        "agg_03_count_prop",
        CuratedCategory::Aggregation,
        "MATCH (n:Person) RETURN count(n.name)",
        false
    ),
    q!(
        "agg_04_min",
        CuratedCategory::Aggregation,
        "MATCH (n:Person) RETURN min(n.age)",
        false
    ),
    q!(
        "agg_05_max",
        CuratedCategory::Aggregation,
        "MATCH (n:Person) RETURN max(n.age)",
        false
    ),
    q!(
        "agg_06_sum",
        CuratedCategory::Aggregation,
        "MATCH (n:Person) RETURN sum(n.age)",
        false
    ),
    q!(
        "agg_07_avg",
        CuratedCategory::Aggregation,
        "MATCH (n:Person) RETURN avg(n.age)",
        false
    ),
    q!(
        "agg_08_collect",
        CuratedCategory::Aggregation,
        "MATCH (n:Person) RETURN collect(n.name)",
        false
    ),
    q!(
        "agg_09_count_distinct",
        CuratedCategory::Aggregation,
        "MATCH (n:Person) RETURN count(DISTINCT n.age)",
        false
    ),
    q!(
        "agg_10_group_by",
        CuratedCategory::Aggregation,
        "MATCH (n:Person) RETURN n.age, count(n)",
        false
    ),
];

/// All curated queries in canonical order: MWR → OPT → OBL → NULL → AGG.
/// Exactly 50 entries (5 categories × 10).
pub const ALL_CURATED_QUERIES: [CuratedQuery; 50] = {
    let mut out = [CuratedQuery {
        name: "_unset",
        category: CuratedCategory::MatchWhereReturn,
        cypher: "",
        ordered: false,
    }; 50];
    let mut i = 0;
    let mut j = 0;
    while j < 10 {
        out[i] = MATCH_WHERE_RETURN[j];
        i += 1;
        j += 1;
    }
    let mut j = 0;
    while j < 10 {
        out[i] = OPTIONAL_MATCH[j];
        i += 1;
        j += 1;
    }
    let mut j = 0;
    while j < 10 {
        out[i] = ORDER_BY_LIMIT[j];
        i += 1;
        j += 1;
    }
    let mut j = 0;
    while j < 10 {
        out[i] = NULL_SEMANTICS[j];
        i += 1;
        j += 1;
    }
    let mut j = 0;
    while j < 10 {
        out[i] = AGGREGATION[j];
        i += 1;
        j += 1;
    }
    out
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_is_exactly_fifty_queries() {
        assert_eq!(ALL_CURATED_QUERIES.len(), 50);
    }

    #[test]
    fn each_category_has_ten_entries() {
        for cat in [
            CuratedCategory::MatchWhereReturn,
            CuratedCategory::OptionalMatch,
            CuratedCategory::OrderByLimit,
            CuratedCategory::NullSemantics,
            CuratedCategory::Aggregation,
        ] {
            let count = ALL_CURATED_QUERIES
                .iter()
                .filter(|q| q.category == cat)
                .count();
            assert_eq!(
                count,
                CURATED_PER_CATEGORY,
                "category {} should have ≥10 entries",
                cat.name()
            );
        }
    }

    #[test]
    fn every_query_has_a_unique_name() {
        let mut names: Vec<&str> = ALL_CURATED_QUERIES.iter().map(|q| q.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ALL_CURATED_QUERIES.len());
    }

    #[test]
    fn every_query_has_a_non_empty_cypher() {
        for q in ALL_CURATED_QUERIES.iter() {
            assert!(
                !q.cypher.trim().is_empty(),
                "query {} has empty cypher",
                q.name
            );
        }
    }

    #[test]
    fn order_by_queries_carry_ordered_flag() {
        for q in ALL_CURATED_QUERIES
            .iter()
            .filter(|q| q.category == CuratedCategory::OrderByLimit)
        {
            assert!(
                q.ordered,
                "ORDER BY query {} must carry ordered=true",
                q.name
            );
        }
    }

    #[test]
    fn non_order_queries_do_not_assert_ordered() {
        for q in ALL_CURATED_QUERIES
            .iter()
            .filter(|q| q.category != CuratedCategory::OrderByLimit)
        {
            assert!(
                !q.ordered,
                "non-ORDER-BY query {} must carry ordered=false",
                q.name
            );
        }
    }
}
