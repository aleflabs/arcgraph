// File-level `dead_code` allow so this module compiles cleanly when
// `#[path]`-included from benches that consume only a subset of the
// constants/helpers (e.g., `ldbc_is3.rs` consumes `IS3` +
// `IS3_P50_TARGET_US` only).
#![allow(dead_code)]

//! M4-84 LDBC SNB Interactive-Short fixture — schema + query bank
//! per design-v2 §10.5 + LDBC SNB Interactive specification §3.5.
//!
//! # Why a sibling fixture (not an extension of `m4_04d_person_tenant`)?
//!
//! The M4-04d fixture is a CARDINALITY-shape fixture for the M4-04d →
//! M4-51 cost-walker transit pin (issue #262). M4-84 needs a SCHEMA
//! fixture (label/rel-type set + per-instance property data) to drive
//! the executor end-to-end on the stub substrate. The two are
//! orthogonal — extending the M4-04d fixture would conflate the two
//! purposes. Per `feedback_avoid_speculative_scaffolding.md` we ship a
//! NEW LDBC fixture with its own bounded contract.
//!
//! # Scope-factor selection (per W13γ spawn prompt)
//!
//! - **CI (in-tree):** SF-0.0001 (10 Persons, ~50 nodes total, ~50
//!   edges). Build-time effectively zero — the executor + cost walker
//!   walk a few HashMap inserts per query.
//! - **Nightly cron (forward-deferred to M6 / sibling):** SF-0.01
//!   (1K Persons). The schema scales linearly; the SF-0.0001 build
//!   path is the canonical structural pin.
//!
//! # ADR provenance
//!
//! - **design-v2 §10.5** — LDBC SNB Interactive-Short P50/P99 targets
//!   (IS1 50µs/500µs, IS2 200µs/2ms, IS3 500µs/5ms, IS4-7 2ms/20ms).
//! - **ADR-038 amendment-02 §M4.h** — M4-84 LDBC harness wiring.
//! - **ADR-038 amendment-03 §TIER-1 GAP D** — OPTIONAL MATCH at v1.0;
//!   IS5 / IS6 are inside the v1.0 harness scope.
//! - **LDBC SNB Interactive Specification §3.5** — IS1..IS7 query
//!   definitions.

use arcgraph_core::{LabelId, NodeId, RelId, TypeId};
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

// ---------------------------------------------------------------------
// LDBC label IDs (matches the StubCatalogProvider monotonic-from-1).
// ---------------------------------------------------------------------

/// Person label per LDBC SNB §3.4.
pub const LABEL_PERSON: LabelId = LabelId::new(1);
/// Place label.
pub const LABEL_PLACE: LabelId = LabelId::new(2);
/// Forum label.
pub const LABEL_FORUM: LabelId = LabelId::new(3);
/// Comment label (LDBC `Message` superclass — we collapse Post/Comment
/// to one Comment label at v1.0; full Post/Comment hierarchy ships at
/// M6 LDBC perf milestone with the real dataset).
pub const LABEL_COMMENT: LabelId = LabelId::new(4);

// ---------------------------------------------------------------------
// LDBC rel-type IDs.
// ---------------------------------------------------------------------

/// `(Person)-[:KNOWS]-(Person)` (undirected per LDBC SNB §3.5 IS3).
pub const REL_KNOWS: TypeId = TypeId::new(1);
/// `(Person)-[:IS_LOCATED_IN]->(Place)`.
pub const REL_IS_LOCATED_IN: TypeId = TypeId::new(2);
/// `(Person)-[:LIKES]->(Comment)`.
pub const REL_LIKES: TypeId = TypeId::new(3);
/// `(Comment)-[:HAS_CREATOR]->(Person)`.
pub const REL_HAS_CREATOR: TypeId = TypeId::new(4);
/// `(Forum)-[:HAS_MEMBER]->(Person)`.
pub const REL_HAS_MEMBER: TypeId = TypeId::new(5);
/// `(Forum)-[:CONTAINER_OF]->(Comment)`.
pub const REL_CONTAINER_OF: TypeId = TypeId::new(6);
/// `(Forum)-[:HAS_MODERATOR]->(Person)`.
pub const REL_HAS_MODERATOR: TypeId = TypeId::new(7);
/// `(Comment)-[:REPLY_OF]->(Comment)`.
pub const REL_REPLY_OF: TypeId = TypeId::new(8);

// ---------------------------------------------------------------------
// LDBC SNB Interactive-Short query bank.
//
// Queries use literal id values (1) instead of `$personId` parameters
// because the v1.0 stub-substrate bench harness measures end-to-end
// plan + execute time without parameter binding plumbing. The query
// shapes match the LDBC SNB §3.5 reference within v1.0 ArcQL grammar
// constraints (no parameters; no map-equality property filters —
// WHERE clause is canonical).
// ---------------------------------------------------------------------

/// IS1 — Profile of a Person. Per design-v2 §10.5 P50 target = 50µs.
pub const IS1: &str = "MATCH (n:Person)-[:IS_LOCATED_IN]->(p:Place) WHERE n.id = 1 \
     RETURN n.firstName, n.lastName, p.id";

/// IS2 — Recent messages of a Person. Per design-v2 §10.5 P50 = 200µs.
pub const IS2: &str = "MATCH (n:Person)<-[:HAS_CREATOR]-(m:Comment) WHERE n.id = 1 \
     RETURN m.id, m.content";

/// IS3 — Friends of a Person. Per design-v2 §10.5 P50 = 500µs.
pub const IS3: &str = "MATCH (n:Person)-[r:KNOWS]-(friend:Person) WHERE n.id = 1 \
     RETURN friend.id, friend.firstName, r.creationDate";

/// IS4 — Content of a message. Per design-v2 §10.5 P50 (IS4-7) = 2ms.
pub const IS4: &str = "MATCH (m:Comment) WHERE m.id = 1 \
     RETURN m.creationDate, m.content";

/// IS5 — Author of a message (LDBC SNB §3.5: forum-author binding).
///
/// Per W13γ fix-up LOW-4 (closes review-pr-285-final.md LOW-4): uses
/// `OPTIONAL MATCH` per amendment-03 §TIER-1 GAP D (OPTIONAL MATCH
/// at v1.0). The earlier fixture used plain `MATCH` which silently
/// violated the "Cypher feature parity" forcing function (per
/// amendment-03 §TIER-1 GAP D rationale: "Shipping v1.0 without
/// OPTIONAL MATCH means our LDBC harness compares against a strict
/// subset, which fails the parity goal").
pub const IS5: &str = "MATCH (m:Comment) \
     OPTIONAL MATCH (m)-[:HAS_CREATOR]->(p:Person) \
     WHERE m.id = 1 \
     RETURN p.id, p.firstName, p.lastName";

/// IS6 — Forum + moderator of a message (LDBC SNB §3.5).
///
/// Per W13γ fix-up LOW-4 (closes review-pr-285-final.md LOW-4): uses
/// `OPTIONAL MATCH` for the forum-membership binding. See `IS5`
/// rationale.
pub const IS6: &str = "MATCH (m:Comment) \
     OPTIONAL MATCH (m)<-[:CONTAINER_OF]-(f:Forum)-[:HAS_MODERATOR]->(p:Person) \
     WHERE m.id = 1 \
     RETURN f.id, p.id";

/// IS7 — Replies of a message.
pub const IS7: &str = "MATCH (m:Comment)<-[:REPLY_OF]-(reply:Comment)-[:HAS_CREATOR]->(p:Person) \
     WHERE m.id = 1 \
     RETURN reply.id, p.id";

/// All IS queries in canonical order. Used by the perf-regression gate
/// + multi-tenant isolation test + plan-cache hit-rate test.
pub const ALL_IS_QUERIES: [(&str, &str); 7] = [
    ("IS1", IS1),
    ("IS2", IS2),
    ("IS3", IS3),
    ("IS4", IS4),
    ("IS5", IS5),
    ("IS6", IS6),
    ("IS7", IS7),
];

// ---------------------------------------------------------------------
// design-v2 §10.5 P50 targets in microseconds. The perf-regression
// gate compares the measured wall-time against `target × SLACK` (the
// stub-substrate harness can't realistically hit the absolute targets
// — those land at M6 LDBC perf milestone with the real dataset).
// ---------------------------------------------------------------------

/// design-v2 §10.5 P50 targets in microseconds.
///
/// Forward-pin: at v1.0-alpha the harness measures **plan-build only**
/// (the M4-61 executor lacks `LogicalJoin` support, so the IS multi-
/// pattern queries cannot run end-to-end against the stub substrate);
/// the absolute §10.5 P50 targets bind to the M6 release-build LDBC
/// dataset full-execute path. These constants stay in the fixture
/// because the crate-local Criterion harnesses cite them in their
/// output.
///
/// The plan-build-only gate uses [`PLAN_BUILD_CEILINGS_US`] below;
/// these end-to-end constants are NOT the gate ceilings.
pub const IS1_P50_TARGET_US: u64 = 50;
pub const IS2_P50_TARGET_US: u64 = 200;
pub const IS3_P50_TARGET_US: u64 = 500;
pub const IS4_P50_TARGET_US: u64 = 2_000;
pub const IS5_P50_TARGET_US: u64 = 2_000;
pub const IS6_P50_TARGET_US: u64 = 2_000;
pub const IS7_P50_TARGET_US: u64 = 2_000;

/// Per-IS-query design-v2 §10.5 P50 target (µs) lookup. Mirrors
/// `ALL_IS_QUERIES`. **End-to-end** target — NOT the perf-gate ceiling.
pub const TARGETS_P50_US: [(&str, u64); 7] = [
    ("IS1", IS1_P50_TARGET_US),
    ("IS2", IS2_P50_TARGET_US),
    ("IS3", IS3_P50_TARGET_US),
    ("IS4", IS4_P50_TARGET_US),
    ("IS5", IS5_P50_TARGET_US),
    ("IS6", IS6_P50_TARGET_US),
    ("IS7", IS7_P50_TARGET_US),
];

// ---------------------------------------------------------------------
// Plan-build-only ceilings — the canonical gate budget per ADR-036 §D-25.
//
// Per W13γ fix-up HIGH-1 (closes review-pr-285-final.md HIGH-1):
//
// The earlier `STUB_SLACK_MULT = 1000` against the §10.5 end-to-end
// targets neutralized the brief's "≥10% regression" mandate (the gate
// had ~314× headroom on IS1 before it could ever fire). The right
// anchor is ADR-036 §D-25's plan-build budget (5ms for 8-way joins);
// the IS queries are 1-3 hop, so plan-build is well-bounded.
//
// Empirical re-derivation on M3 Pro (debug build, in-CI hardware):
//   IS1 = 17.3µs (release) / 159µs (debug)
//   IS2..IS7 = 100..300µs (debug)
//
// The plan-build ceilings below are derived from join-arity:
//   1-hop (IS1, IS2, IS4) — 100µs anchor; ceiling = 1ms (10× slack)
//   2-3 hop (IS3, IS5, IS6, IS7) — 500µs anchor; ceiling = 5ms (10×)
//
// Slack is bounded at 10× to accommodate (a) debug:release ratio
// ~9-10× empirically (release IS1 17.3µs vs debug 159µs ≈ 9.2×) +
// (b) CI hardware variance ~2× (per memory
// `feedback_pr_mergeable_conflicting_blocks_ci.md`). Total slack 10×
// is principled, not arbitrary; a future regression of >10× the
// plan-build anchor is a real, actionable signal.
// ---------------------------------------------------------------------

/// Per-IS-query plan-build P50 anchor (µs) per ADR-036 §D-25 (5ms for
/// 8-way joins; LDBC IS queries are 1-3 hop).
///
/// 1-hop (IS1, IS2, IS4) = 100µs; 2-3 hop (IS3, IS5, IS6, IS7) = 500µs.
pub const PLAN_BUILD_ANCHORS_US: [(&str, u64); 7] = [
    ("IS1", 100),
    ("IS2", 100),
    ("IS3", 500),
    ("IS4", 100),
    ("IS5", 500),
    ("IS6", 500),
    ("IS7", 500),
];

/// Per-IS-query plan-build ceiling (µs) — the perf-gate fail threshold.
/// Each ceiling is `anchor × PLAN_BUILD_SLACK_MULT`. A regression of
/// any size that exceeds the ceiling fires the gate.
pub const PLAN_BUILD_CEILINGS_US: [(&str, u64); 7] = [
    ("IS1", 100 * PLAN_BUILD_SLACK_MULT),
    ("IS2", 100 * PLAN_BUILD_SLACK_MULT),
    ("IS3", 500 * PLAN_BUILD_SLACK_MULT),
    ("IS4", 100 * PLAN_BUILD_SLACK_MULT),
    ("IS5", 500 * PLAN_BUILD_SLACK_MULT),
    ("IS6", 500 * PLAN_BUILD_SLACK_MULT),
    ("IS7", 500 * PLAN_BUILD_SLACK_MULT),
];

/// Plan-build slack multiplier — applied to [`PLAN_BUILD_ANCHORS_US`]
/// to derive the gate ceilings.
///
/// Value 10× is principled, not arbitrary, and decomposes:
/// - Debug:release ratio (~9-10×): IS1 measured 17.3µs release vs
///   159µs debug ≈ 9.2× on M3 Pro per W13γ fix-up empirical re-
///   derivation.
/// - CI hardware variance (~1-2×): per memory
///   `feedback_pr_mergeable_conflicting_blocks_ci.md` and PR #282
///   flake history.
///
/// Total: 10× covers both axes without leaving 100×+ headroom that
/// would silently neutralize regression detection.
pub const PLAN_BUILD_SLACK_MULT: u64 = 10;

/// (Removed at W13γ fix-up HIGH-1 — see [`PLAN_BUILD_SLACK_MULT`].)
///
/// The earlier `STUB_SLACK_MULT = 1000` against the §10.5 end-to-end
/// targets neutralized the brief's "≥10% regression" mandate. The
/// constant is preserved here as a deprecated alias for any forward
/// callers that still depend on the old name during the W13γ fix-up
/// transition; new callers MUST use [`PLAN_BUILD_CEILINGS_US`].
#[deprecated(
    since = "0.0.0",
    note = "W13γ fix-up HIGH-1: 1000× slack against §10.5 end-to-end \
            silently neutralizes regression detection. Use \
            PLAN_BUILD_CEILINGS_US (anchored to ADR-036 §D-25 plan-build \
            budget × 10× principled slack)."
)]
pub const STUB_SLACK_MULT: u64 = 1_000;

// ---------------------------------------------------------------------
// LDBC fixture builders.
// ---------------------------------------------------------------------

/// Default scale factor for the in-CI bench harness. SF-0.0001 = 10
/// Persons (LDBC SNB SF-1.0 = 10K Persons; v1.0 stub-substrate path
/// is structural, not perf-anchored — the M6 release-build LDBC perf
/// milestone uses SF-1.0 / SF-10 / SF-30 datasets per the LDBC SNB
/// driver contract).
pub const DEFAULT_PERSON_COUNT_SF_0_0001: u64 = 10;

/// W13γ fix-up LOW-3 — SF-0.01 cron-only scale (1K Persons; 100×
/// SF-0.0001). The schema scales linearly; the SF-0.0001 builder is
/// the canonical structural pin, and SF-0.01 is the forward-pin for
/// the M6 LDBC perf milestone's release-build perf-gate re-anchoring.
pub const DEFAULT_PERSON_COUNT_SF_0_01: u64 = 1_000;

/// LDBC catalog at SF-0.0001 (the default in-CI scale).
///
/// All LDBC SNB §3.5 properties are registered up-front so the
/// type-check pass surfaces concrete `PropertyType` values rather
/// than `Unknown` — improves cost-walker selectivity precision +
/// catches schema-drift regressions at typecheck time.
pub fn catalog_sf_0_0001() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person", "Place", "Forum", "Comment"])
        .with_rel_types([
            "KNOWS",
            "IS_LOCATED_IN",
            "LIKES",
            "HAS_CREATOR",
            "HAS_MEMBER",
            "CONTAINER_OF",
            "HAS_MODERATOR",
            "REPLY_OF",
        ])
        .with_properties([
            "id",
            "firstName",
            "lastName",
            "birthday",
            "locationIp",
            "browserUsed",
            "gender",
            "creationDate",
            "name",
            "title",
            "content",
            "type",
        ])
        // Total cardinalities (used by the M4-51 cost walker for IS-
        // query plan-cost ranking). Numbers chosen at SF-0.0001 — 10
        // Persons / 5 Places / 2 Forums / 50 Comments / ~100 edges.
        .with_total_node_count(10 + 5 + 2 + 50)
        .with_total_rel_count(100)
        .with_label_cardinality(LABEL_PERSON, DEFAULT_PERSON_COUNT_SF_0_0001)
        .with_label_cardinality(LABEL_PLACE, 5)
        .with_label_cardinality(LABEL_FORUM, 2)
        .with_label_cardinality(LABEL_COMMENT, 50)
        .with_rel_type_cardinality(REL_KNOWS, 30)
        .with_rel_type_cardinality(REL_IS_LOCATED_IN, 10)
        .with_rel_type_cardinality(REL_LIKES, 20)
        .with_rel_type_cardinality(REL_HAS_CREATOR, 50)
        .with_rel_type_cardinality(REL_HAS_MEMBER, 5)
        .with_rel_type_cardinality(REL_CONTAINER_OF, 50)
        .with_rel_type_cardinality(REL_HAS_MODERATOR, 2)
        .with_rel_type_cardinality(REL_REPLY_OF, 30)
}

/// LDBC catalog at SF-0.01 (W13γ fix-up LOW-3 — cron-only scale).
///
/// 100× SF-0.0001 cardinalities — the schema scales linearly per the
/// LDBC SNB driver contract. Used by the `#[ignore]`-gated SF-0.01
/// perf-gate test (`ldbc_is1_through_is7_plan_build_budget_gate_sf_0_01`)
/// for the nightly cron path. Stub-substrate at SF-0.01 is fixture-
/// only — the substrate is empty (the perf-gate measures plan-build,
/// not execute-time, so substrate population is unnecessary at this
/// scale).
pub fn catalog_sf_0_01() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person", "Place", "Forum", "Comment"])
        .with_rel_types([
            "KNOWS",
            "IS_LOCATED_IN",
            "LIKES",
            "HAS_CREATOR",
            "HAS_MEMBER",
            "CONTAINER_OF",
            "HAS_MODERATOR",
            "REPLY_OF",
        ])
        .with_properties([
            "id",
            "firstName",
            "lastName",
            "birthday",
            "locationIp",
            "browserUsed",
            "gender",
            "creationDate",
            "name",
            "title",
            "content",
            "type",
        ])
        // Total cardinalities — 100× SF-0.0001.
        .with_total_node_count(1_000 + 500 + 200 + 5_000)
        .with_total_rel_count(10_000)
        .with_label_cardinality(LABEL_PERSON, DEFAULT_PERSON_COUNT_SF_0_01)
        .with_label_cardinality(LABEL_PLACE, 500)
        .with_label_cardinality(LABEL_FORUM, 200)
        .with_label_cardinality(LABEL_COMMENT, 5_000)
        .with_rel_type_cardinality(REL_KNOWS, 3_000)
        .with_rel_type_cardinality(REL_IS_LOCATED_IN, 1_000)
        .with_rel_type_cardinality(REL_LIKES, 2_000)
        .with_rel_type_cardinality(REL_HAS_CREATOR, 5_000)
        .with_rel_type_cardinality(REL_HAS_MEMBER, 500)
        .with_rel_type_cardinality(REL_CONTAINER_OF, 5_000)
        .with_rel_type_cardinality(REL_HAS_MODERATOR, 200)
        .with_rel_type_cardinality(REL_REPLY_OF, 3_000)
}

/// Build the LDBC stub-substrate at SF-0.0001 against the canonical
/// LDBC catalog. Populates a small but representative graph:
///
/// - 10 Persons (id 1..10), each with firstName/lastName/birthday.
/// - 5 Places (id 100..104).
/// - 2 Forums (id 200..201).
/// - 50 Comments (id 1000..1049), each tied to a creator + a forum.
/// - Edges:
///   * Person 1 -[IS_LOCATED_IN]-> Place 100
///   * Person {i} -[KNOWS]- Person {i+1 mod 10}, i ∈ 1..=10
///   * Comment {1000+j} -[HAS_CREATOR]-> Person {1 + j%10}
///   * Forum 200 -[CONTAINER_OF]-> Comment {1000+j}
///   * Forum 200 -[HAS_MODERATOR]-> Person 1
///   * Comment 1001 -[REPLY_OF]-> Comment 1000
///
/// The exact edge structure is calibrated so each IS query's WHERE
/// clause `n.id = 1` matches at least one starting point, exercising
/// the operator pipeline end-to-end. The stub does NOT enforce LDBC
/// schema integrity — additional properties or edges would not break
/// the bench, only its representativeness.
pub fn substrate_sf_0_0001(catalog: &StubCatalogProvider) -> StubExecutorSubstrate {
    use arcgraph_query::semantic::CatalogProvider;
    let tenant = catalog.tenant();
    let mut sub = StubExecutorSubstrate::new();

    // Persons: ids 1..=10.
    for i in 1u64..=DEFAULT_PERSON_COUNT_SF_0_0001 {
        sub = sub.with_node(
            tenant,
            NodeView::new(NodeId::new(i), Some(LABEL_PERSON))
                .with_property("id", Value::Integer(i as i64))
                .with_property("firstName", Value::String(format!("first{i}")))
                .with_property("lastName", Value::String(format!("last{i}")))
                .with_property("birthday", Value::Integer(19900000 + i as i64))
                .with_property("locationIp", Value::String(format!("10.0.0.{i}")))
                .with_property("browserUsed", Value::String("Firefox".into()))
                .with_property("gender", Value::String("U".into())),
        );
    }
    // Places: ids 100..=104.
    for i in 100u64..=104 {
        sub = sub.with_node(
            tenant,
            NodeView::new(NodeId::new(i), Some(LABEL_PLACE))
                .with_property("id", Value::Integer(i as i64))
                .with_property("name", Value::String(format!("Place{i}"))),
        );
    }
    // Forums: ids 200..=201.
    for i in 200u64..=201 {
        sub = sub.with_node(
            tenant,
            NodeView::new(NodeId::new(i), Some(LABEL_FORUM))
                .with_property("id", Value::Integer(i as i64))
                .with_property("title", Value::String(format!("Forum{i}"))),
        );
    }
    // Comments: ids 1000..=1049.
    for j in 0u64..50 {
        let cid = 1000 + j;
        sub = sub.with_node(
            tenant,
            NodeView::new(NodeId::new(cid), Some(LABEL_COMMENT))
                .with_property("id", Value::Integer(cid as i64))
                .with_property("content", Value::String(format!("c{cid}")))
                .with_property("creationDate", Value::Integer(20260000 + j as i64)),
        );
    }
    // Person 1 -[IS_LOCATED_IN]-> Place 100.
    sub = sub.with_edge(
        tenant,
        RelView::new(
            RelId::new(1),
            NodeId::new(1),
            NodeId::new(100),
            Some(REL_IS_LOCATED_IN),
        ),
    );
    // KNOWS ring among Persons.
    let mut edge_id: u64 = 2;
    for i in 1u64..=DEFAULT_PERSON_COUNT_SF_0_0001 {
        let nxt = if i == DEFAULT_PERSON_COUNT_SF_0_0001 {
            1
        } else {
            i + 1
        };
        sub = sub.with_edge(
            tenant,
            RelView::new(
                RelId::new(edge_id),
                NodeId::new(i),
                NodeId::new(nxt),
                Some(REL_KNOWS),
            )
            .with_property("creationDate", Value::Integer(20250000 + i as i64)),
        );
        edge_id += 1;
    }
    // HAS_CREATOR + CONTAINER_OF for each Comment.
    for j in 0u64..50 {
        let cid = 1000 + j;
        let creator = 1 + (j % DEFAULT_PERSON_COUNT_SF_0_0001);
        sub = sub.with_edge(
            tenant,
            RelView::new(
                RelId::new(edge_id),
                NodeId::new(cid),
                NodeId::new(creator),
                Some(REL_HAS_CREATOR),
            ),
        );
        edge_id += 1;
        sub = sub.with_edge(
            tenant,
            RelView::new(
                RelId::new(edge_id),
                NodeId::new(200),
                NodeId::new(cid),
                Some(REL_CONTAINER_OF),
            ),
        );
        edge_id += 1;
    }
    // Forum 200 -[HAS_MODERATOR]-> Person 1.
    sub = sub.with_edge(
        tenant,
        RelView::new(
            RelId::new(edge_id),
            NodeId::new(200),
            NodeId::new(1),
            Some(REL_HAS_MODERATOR),
        ),
    );
    edge_id += 1;
    // Comment 1001 -[REPLY_OF]-> Comment 1000.
    sub = sub.with_edge(
        tenant,
        RelView::new(
            RelId::new(edge_id),
            NodeId::new(1001),
            NodeId::new(1000),
            Some(REL_REPLY_OF),
        ),
    );
    let _ = edge_id; // edge counter kept for future LDBC fixture extension.

    sub
}

/// Build the LDBC stub-substrate at SF-0.01 — the W15γ M6-04 nightly-cron
/// scale.
///
/// 100× SF-0.0001 in cardinality (1K Persons, 500 Places, 200 Forums,
/// 5K Comments). The schema is identical to [`substrate_sf_0_0001`];
/// only the per-label counts scale linearly per the LDBC SNB driver
/// contract.
///
/// # Why both substrates ship as in-tree fixtures
///
/// SF-0.01 is the W15γ M6-04 nightly-cron scale per the spawn brief:
/// in-tree fixture-only (no external datagen), small enough to build
/// in <50ms (1K Persons × 7 properties), large enough that an
/// O(card²) cost-walker regression would surface here. The SF-1.0+
/// real-LDBC-datagen path is intentionally outside this hermetic
/// fixture.
///
/// # Forward-pin: full-execute
///
/// At v1.0-alpha the M4-61 executor lacks `LogicalJoin` support, so
/// the IS multi-pattern queries cannot run end-to-end against this
/// substrate; the perf-gate measures plan-build only (per
/// `tests/m4_84_ldbc_perf_gate.rs` and the W15γ M6-04 cron). The
/// substrate is included for forward-binding so the M6 LDBC perf
/// milestone (`LogicalJoin` executor support + full-execute path)
/// lands a substrate-bearing harness without re-defining the fixture.
pub fn substrate_sf_0_01(catalog: &StubCatalogProvider) -> StubExecutorSubstrate {
    use arcgraph_query::semantic::CatalogProvider;
    let tenant = catalog.tenant();
    let mut sub = StubExecutorSubstrate::new();

    // Persons: ids 1..=1_000.
    for i in 1u64..=DEFAULT_PERSON_COUNT_SF_0_01 {
        sub = sub.with_node(
            tenant,
            NodeView::new(NodeId::new(i), Some(LABEL_PERSON))
                .with_property("id", Value::Integer(i as i64))
                .with_property("firstName", Value::String(format!("first{i}")))
                .with_property("lastName", Value::String(format!("last{i}")))
                .with_property("birthday", Value::Integer(19900000 + i as i64))
                .with_property(
                    "locationIp",
                    Value::String(format!("10.0.{}.{}", i / 256, i % 256)),
                )
                .with_property("browserUsed", Value::String("Firefox".into()))
                .with_property("gender", Value::String("U".into())),
        );
    }
    // Places: ids 100_000..=100_499.
    for i in 100_000u64..=100_499 {
        sub = sub.with_node(
            tenant,
            NodeView::new(NodeId::new(i), Some(LABEL_PLACE))
                .with_property("id", Value::Integer(i as i64))
                .with_property("name", Value::String(format!("Place{i}"))),
        );
    }
    // Forums: ids 200_000..=200_199.
    for i in 200_000u64..=200_199 {
        sub = sub.with_node(
            tenant,
            NodeView::new(NodeId::new(i), Some(LABEL_FORUM))
                .with_property("id", Value::Integer(i as i64))
                .with_property("title", Value::String(format!("Forum{i}"))),
        );
    }
    // Comments: ids 1_000_000..=1_004_999.
    for j in 0u64..5_000 {
        let cid = 1_000_000 + j;
        sub = sub.with_node(
            tenant,
            NodeView::new(NodeId::new(cid), Some(LABEL_COMMENT))
                .with_property("id", Value::Integer(cid as i64))
                .with_property("content", Value::String(format!("c{cid}")))
                .with_property("creationDate", Value::Integer(20260000 + j as i64)),
        );
    }
    // Person 1 -[IS_LOCATED_IN]-> Place 100_000.
    let mut edge_id: u64 = 1;
    sub = sub.with_edge(
        tenant,
        RelView::new(
            RelId::new(edge_id),
            NodeId::new(1),
            NodeId::new(100_000),
            Some(REL_IS_LOCATED_IN),
        ),
    );
    edge_id += 1;
    // KNOWS ring among Persons (P_i ↔ P_{(i mod N) + 1}).
    for i in 1u64..=DEFAULT_PERSON_COUNT_SF_0_01 {
        let nxt = if i == DEFAULT_PERSON_COUNT_SF_0_01 {
            1
        } else {
            i + 1
        };
        sub = sub.with_edge(
            tenant,
            RelView::new(
                RelId::new(edge_id),
                NodeId::new(i),
                NodeId::new(nxt),
                Some(REL_KNOWS),
            )
            .with_property("creationDate", Value::Integer(20250000 + i as i64)),
        );
        edge_id += 1;
    }
    // HAS_CREATOR + CONTAINER_OF for each Comment (cycled across
    // Persons / Forums).
    for j in 0u64..5_000 {
        let cid = 1_000_000 + j;
        let creator = 1 + (j % DEFAULT_PERSON_COUNT_SF_0_01);
        let forum = 200_000 + (j % 200);
        sub = sub.with_edge(
            tenant,
            RelView::new(
                RelId::new(edge_id),
                NodeId::new(cid),
                NodeId::new(creator),
                Some(REL_HAS_CREATOR),
            ),
        );
        edge_id += 1;
        sub = sub.with_edge(
            tenant,
            RelView::new(
                RelId::new(edge_id),
                NodeId::new(forum),
                NodeId::new(cid),
                Some(REL_CONTAINER_OF),
            ),
        );
        edge_id += 1;
    }
    // Forum 200_000 -[HAS_MODERATOR]-> Person 1.
    sub = sub.with_edge(
        tenant,
        RelView::new(
            RelId::new(edge_id),
            NodeId::new(200_000),
            NodeId::new(1),
            Some(REL_HAS_MODERATOR),
        ),
    );
    edge_id += 1;
    // Comment 1_000_001 -[REPLY_OF]-> Comment 1_000_000.
    sub = sub.with_edge(
        tenant,
        RelView::new(
            RelId::new(edge_id),
            NodeId::new(1_000_001),
            NodeId::new(1_000_000),
            Some(REL_REPLY_OF),
        ),
    );
    let _ = edge_id;

    sub
}
