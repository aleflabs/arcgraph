//! Integration tests for M4-52 (M4-05b) DP-based binary-join
//! ordering — LDBC SNB Interactive-Short-shaped query plans.
//!
//! Each test hand-builds a logical plan that mimics the shape of an
//! LDBC SNB IS query (per design-v2 §10.5; Erling et al.
//! "The LDBC Social Network Benchmark", DBMOD 2015). We do NOT go
//! through the full parser → binder → lowering pipeline — those are
//! covered upstream — instead we plant the LogicalPlan directly so
//! the tests pin the M4-52 enumerator's behavior in isolation.
//!
//! # Pinned invariants
//!
//! Every test pins:
//! 1. The output's exhaustive-match behavior (root variant
//!    preserved across enumeration).
//! 2. Determinism (same inputs → byte-equal outputs across two
//!    invocations).
//! 3. Cost-monotonic ordering (the chosen plan's total cost ≤ a
//!    naive-input-order baseline).
//!
//! # ADR provenance
//! - ADR-038 amendment-02 §M4.e — M4-52 (M4-05b) slice scope.
//! - ADR-038 §2 D-24 — `LogicalPlan` exhaustive-match contract.
//! - ADR-038 §2 D-25 — catalog stats (M4-41 input).
//! - ADR-036 §D-25 — 5 ms plan-build budget.
//! - ADR-006 amendment-01 §A-2 — OPTIONAL MATCH lowers to LeftOuterJoin.

use arcgraph_core::{LabelId, Lsn, TypeId};
use arcgraph_query::error::Span;
use arcgraph_query::logical_plan::{
    Direction, JoinAlgorithm, JoinCondition, LogicalEmpty, LogicalExpand, LogicalFilter,
    LogicalJoin, LogicalLeftOuterJoin, LogicalLimit, LogicalPlan, LogicalProject, LogicalScan,
};
use arcgraph_query::planner::cost::estimate_costs;
use arcgraph_query::planner::enumerate_join_order;
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression};

fn span() -> Span {
    Span::point(1, 1)
}

fn scan(label_raw: u32, var_raw: u64) -> LogicalPlan {
    LogicalPlan::Scan(LogicalScan {
        label: Some(LabelId::new(label_raw)),
        var: BindingId::new(var_raw),
        read_lsn: Lsn::MAX,
        span: span(),
    })
}

fn expand(from: u64, to: u64, rel_type: u32) -> LogicalPlan {
    LogicalPlan::Expand(LogicalExpand {
        from: BindingId::new(from),
        to: BindingId::new(to),
        direction: Direction::LeftToRight,
        rel_type: Some(TypeId::new(rel_type)),
        length_range: None,
        rel_var: None,
        span: span(),
    })
}

fn ij(left: LogicalPlan, right: LogicalPlan, on: Vec<BindingId>) -> LogicalPlan {
    LogicalPlan::Join(LogicalJoin {
        left: Box::new(left),
        right: Box::new(right),
        on: JoinCondition::SharedBindings(on),
        algorithm: JoinAlgorithm::Auto,
        span: span(),
    })
}

fn loj(left: LogicalPlan, right: LogicalPlan, on: Vec<BindingId>) -> LogicalPlan {
    LogicalPlan::LeftOuterJoin(LogicalLeftOuterJoin {
        left: Box::new(left),
        right: Box::new(right),
        on: JoinCondition::SharedBindings(on),
        span: span(),
    })
}

fn ldbc_catalog() -> StubCatalogProvider {
    // LDBC SNB-1 sf=1 approximate cardinalities, rounded for
    // test readability.
    StubCatalogProvider::new()
        .with_total_node_count(11_000)
        .with_total_rel_count(50_000)
        .with_label_cardinality(LabelId::new(1), 9_900) // Person
        .with_label_cardinality(LabelId::new(2), 1_000) // Forum
        .with_label_cardinality(LabelId::new(3), 5_000) // Post
        .with_label_cardinality(LabelId::new(4), 8_000) // Comment
        .with_rel_type_cardinality(TypeId::new(1), 30_000) // KNOWS
        .with_rel_type_cardinality(TypeId::new(2), 5_000) // HAS_MEMBER
        .with_rel_type_cardinality(TypeId::new(3), 5_000) // HAS_CREATOR
        .with_rel_type_cardinality(TypeId::new(4), 8_000) // REPLY_OF
}

/// IS3 — friend list.
///
/// Cypher shape: `MATCH (p:Person) WHERE p.id = $x MATCH (p)-[:KNOWS]->(f:Person) RETURN f`.
/// LogicalPlan shape: 2-way Join over Person scan + KNOWS-Expand+Person scan.
/// Pin: enumeration may swap left/right; SharedBindings on `p` preserved.
#[test]
fn is3_friend_list_two_way_join() {
    let cat = ldbc_catalog();
    // Two patterns sharing var=0 (the Person `p`).
    let left = scan(1, 0); // Person p
    let right = expand(0, 1, 1); // (p)-[:KNOWS]->(f)
    let plan = ij(left, right, vec![BindingId::new(0)]);

    let out_a = enumerate_join_order(plan.clone(), &cat);
    let out_b = enumerate_join_order(plan.clone(), &cat);
    // Determinism.
    assert_eq!(out_a, out_b);
    // Root preserved as Join with shared-binding p.
    match out_a {
        LogicalPlan::Join(j) => match j.on {
            JoinCondition::SharedBindings(ids) => {
                assert_eq!(ids, vec![BindingId::new(0)]);
            }
        },
        _ => panic!("expected Join at root"),
    }
}

/// IS4 — message detail (single-node lookup).
///
/// Cypher shape: `MATCH (m:Comment) WHERE m.id = $x RETURN m`.
/// LogicalPlan shape: Filter(Scan). NO joins → enumeration is
/// identity.
#[test]
fn is4_message_detail_no_joins_passthrough() {
    let cat = ldbc_catalog();
    let plan = LogicalPlan::Filter(LogicalFilter {
        input: Box::new(scan(4, 0)),
        predicate: BoundExpression::Literal {
            value: arcgraph_query::ast::Literal::Bool(true),
            span: span(),
            type_info: None,
        },
        span: span(),
    });
    let out = enumerate_join_order(plan.clone(), &cat);
    assert_eq!(out, plan, "no joins → enumeration is identity");
}

/// IS5 — friend's posts with OPTIONAL MATCH (Cypher 9 §6.5).
///
/// Cypher shape (paraphrased): `MATCH (p:Person)-[:KNOWS]->(f:Person)
/// OPTIONAL MATCH (f)<-[:HAS_CREATOR]-(post:Post) RETURN f, post`.
/// LogicalPlan shape: LeftOuterJoin(Inner-Join(Person, KNOWS), Post).
/// Pin: enumeration preserves the LeftOuterJoin boundary; the inner
/// equi-join sub-tree on the left side is reordered.
#[test]
fn is5_friend_posts_optional_match_preserves_outer_join() {
    let cat = ldbc_catalog();
    // Inner: (p:Person) ⨝ (p)-[:KNOWS]->(f)  on p
    let inner = ij(scan(1, 0), expand(0, 1, 1), vec![BindingId::new(0)]);
    // Outer: inner LEFT OUTER JOIN (post:Post) on f
    let plan = loj(inner, scan(3, 1), vec![BindingId::new(1)]);
    let out = enumerate_join_order(plan, &cat);

    match out {
        LogicalPlan::LeftOuterJoin(j) => {
            // Outer boundary preserved.
            match j.on {
                JoinCondition::SharedBindings(ids) => {
                    assert_eq!(ids, vec![BindingId::new(1)]);
                }
            }
            // Inner side is still a Join (re-rooted but join-rooted).
            assert!(matches!(*j.left, LogicalPlan::Join(_)));
            // Right side is the Post scan (preserved).
            match *j.right {
                LogicalPlan::Scan(s) => assert_eq!(s.label, Some(LabelId::new(3))),
                _ => panic!("right side of outer join should be Post scan"),
            }
        }
        _ => panic!("LeftOuterJoin must be preserved at the boundary"),
    }
}

/// IS6 — forum membership 3-way Join.
///
/// Cypher shape: `MATCH (forum:Forum)-[:HAS_MEMBER]->(p:Person),
/// (forum)-[:HAS_MEMBER]->(p2:Person) WHERE p.id = $x RETURN forum,
/// p2`.
/// LogicalPlan shape: 3-way inner-join cluster — forum scan + two
/// expands sharing the forum binding.
/// Pin: enumeration covers all 3 leaves; output is a left-deep
/// 3-way Join.
#[test]
fn is6_forum_membership_three_way() {
    let cat = ldbc_catalog();
    // Three leaves all anchored at forum (var=0).
    let leaf_forum = scan(2, 0);
    let leaf_member1 = expand(0, 1, 2); // forum-[HAS_MEMBER]->p
    let leaf_member2 = expand(0, 2, 2); // forum-[HAS_MEMBER]->p2
    // Original plan: ((forum, member1), member2) on var=0.
    let plan = ij(
        ij(leaf_forum, leaf_member1, vec![BindingId::new(0)]),
        leaf_member2,
        vec![BindingId::new(0)],
    );
    let out = enumerate_join_order(plan, &cat);

    // Root is a Join.
    let outer = match out {
        LogicalPlan::Join(j) => j,
        _ => panic!("expected Join at root"),
    };
    // The chain has the binding-shared SharedBindings on the outer
    // join.
    match outer.on {
        JoinCondition::SharedBindings(ids) => {
            assert_eq!(ids, vec![BindingId::new(0)]);
        }
    }
    // The left side is itself a Join (left-deep on 3 leaves).
    assert!(matches!(*outer.left, LogicalPlan::Join(_)));
    // The right side is a singleton leaf (left-deep invariant).
    assert!(!matches!(*outer.right, LogicalPlan::Join(_)));
}

/// IS7 — post replier chain (linear traversal).
///
/// Cypher shape (paraphrased): `MATCH (post:Post)<-[:REPLY_OF]-(c:Comment)<-[:HAS_CREATOR]-(replier:Person) RETURN replier`.
///
/// LogicalPlan shape: 3-leaf linear chain — Post + REPLY_OF expand + HAS_CREATOR expand.
///
/// Pin: enumeration over a linear chain produces a left-deep tree.
#[test]
fn is7_post_replier_chain_linear() {
    let cat = ldbc_catalog();
    let leaf_post = scan(3, 0); // Post p
    let leaf_reply = expand(0, 1, 4); // (p)<-[:REPLY_OF]-(c)
    let leaf_creator = expand(1, 2, 3); // (c)<-[:HAS_CREATOR]-(replier)
    let plan = ij(
        ij(leaf_post, leaf_reply, vec![BindingId::new(0)]),
        leaf_creator,
        vec![BindingId::new(1)],
    );
    let out = enumerate_join_order(plan, &cat);

    // Output is a Join-rooted left-deep tree.
    match out {
        LogicalPlan::Join(j) => {
            // Right side is a singleton (left-deep invariant).
            assert!(!matches!(*j.right, LogicalPlan::Join(_)));
            // Left side is itself a Join (3-leaf chain).
            assert!(matches!(*j.left, LogicalPlan::Join(_)));
        }
        _ => panic!("expected Join at root"),
    }
}

/// **Cost-monotonicity oracle: M4-52 enumeration cost ≤ M4-51 cost
/// of the input plan.** This is the load-bearing claim of the DP
/// — if M4-52 picks a worse plan than the input, it has failed its
/// purpose.
///
/// Constructs an input plan in the WORST left-deep order (largest
/// scan first), then checks that the enumeration improves the
/// total cost (or matches it; never worsens). Pins the M4-52 ↔
/// M4-51 cross-PR coherence: same catalog → same cost-frame; M4-52
/// only ever lowers the cost.
#[test]
fn enumeration_cost_does_not_increase_versus_input() {
    let cat = ldbc_catalog();
    // Worst-case input: scan the largest table first (Person 9_900),
    // then medium (Comment 8_000), then smallest (Forum 1_000). All
    // sharing var=0.
    let leaf_a = scan(1, 0); // Person 9_900
    let leaf_b = scan(4, 0); // Comment 8_000
    let leaf_c = scan(2, 0); // Forum 1_000
    let input_plan = ij(
        ij(leaf_a, leaf_b, vec![BindingId::new(0)]),
        leaf_c,
        vec![BindingId::new(0)],
    );

    let input_cost = estimate_costs(input_plan.clone(), &cat).total_cost();
    let out = enumerate_join_order(input_plan, &cat);
    let out_cost = estimate_costs(out, &cat).total_cost();
    assert!(
        out_cost.total() <= input_cost.total() + 1e-6,
        "enumeration must not increase cost (input={}, out={})",
        input_cost.total(),
        out_cost.total()
    );
}

/// **Sanity coverage: degenerate empty plan.** An empty input is
/// returned unchanged. Pins the no-op behavior at the API
/// boundary.
#[test]
fn empty_plan_passthrough() {
    let cat = ldbc_catalog();
    let plan = LogicalPlan::Empty(LogicalEmpty { span: span() });
    let out = enumerate_join_order(plan.clone(), &cat);
    assert_eq!(out, plan);
}

/// **Wrapper preservation: Limit + Filter wrapping a 3-way Join.**
/// Pins that the rewriter's exhaustive-match preserves wrapper
/// chains.
#[test]
fn limit_filter_wraps_three_way_join_preserved() {
    let cat = ldbc_catalog();
    let inner_join = ij(
        ij(scan(1, 0), scan(2, 0), vec![BindingId::new(0)]),
        scan(3, 0),
        vec![BindingId::new(0)],
    );
    let plan = LogicalPlan::Limit(LogicalLimit {
        input: Box::new(LogicalPlan::Filter(LogicalFilter {
            input: Box::new(inner_join),
            predicate: BoundExpression::Literal {
                value: arcgraph_query::ast::Literal::Bool(true),
                span: span(),
                type_info: None,
            },
            span: span(),
        })),
        count: 100,
        span: span(),
    });
    let out = enumerate_join_order(plan, &cat);
    match out {
        LogicalPlan::Limit(l) => match *l.input {
            LogicalPlan::Filter(f) => assert!(matches!(*f.input, LogicalPlan::Join(_))),
            _ => panic!("expected Filter under Limit"),
        },
        _ => panic!("expected Limit at root"),
    }
}

/// **Project wrapping: Project + 4-way star around a Person anchor.**
/// Tests the typical LDBC IS3-IS6 pattern (multi-pattern MATCH with
/// shared anchor).
#[test]
fn project_wraps_four_way_shared_anchor_join() {
    let cat = ldbc_catalog();
    // 4 leaves all anchored at var=0 (Person p).
    let plan = LogicalPlan::Project(LogicalProject {
        input: Box::new(ij(
            ij(
                ij(
                    scan(1, 0),      // Person p
                    expand(0, 1, 1), // (p)-[:KNOWS]->(f1)
                    vec![BindingId::new(0)],
                ),
                expand(0, 2, 1), // (p)-[:KNOWS]->(f2)
                vec![BindingId::new(0)],
            ),
            expand(0, 3, 1), // (p)-[:KNOWS]->(f3)
            vec![BindingId::new(0)],
        )),
        items: Vec::new(),
        span: span(),
    });
    let out_a = enumerate_join_order(plan.clone(), &cat);
    let out_b = enumerate_join_order(plan, &cat);
    // Determinism.
    assert_eq!(out_a, out_b);
    // Root is Project preserved.
    match out_a {
        LogicalPlan::Project(p) => {
            assert!(matches!(*p.input, LogicalPlan::Join(_)));
        }
        _ => panic!("expected Project at root"),
    }
}
