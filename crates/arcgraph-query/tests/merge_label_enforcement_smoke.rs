//! ADR-152-amendment-01 W27 — MERGE match-branch LABEL enforcement.
//!
//! Lifts ADR-152 §"Forward-deferred" #8 (and closes the live
//! correctness bug the W27-ζ audit flagged as O-3 + ADR-152 Risk #5):
//! the MERGE match-branch now enforces the pattern's node / path-source
//! LABEL, not just its property literals.
//!
//! Mechanism (no v1.2 catalog index needed — issue #351 is a PERF
//! optimization, not a correctness prerequisite):
//!   * §D-1 — the binder resolves the match-side label NAME →
//!     `Option<LabelId>` at bind time (None-tolerant `lookup_label`).
//!   * §D-2 — `Some(id)` lowers to `Scan{label: Some(id)}` (MATCH's
//!     proven path) + the existing property-filter wrap.
//!   * §D-3 — label-present-but-un-interned lowers to
//!     `LogicalPlan::Empty` (O(1) EOS), so the create-branch fires.
//!
//! Each test asserts a DISTINCT branch with a STRONG oracle — the
//! created node's **label** (via `scan_nodes(Some(label_id))`) and its
//! **property bag**, never a bare node-count. Tests 1, 2, and 6 FAIL at
//! HEAD before this amendment (the match-branch cross-matched a
//! different label).
//!
//! ## Stub label-id allocation (deterministic — pinned here)
//!
//! `StubExecutorSubstrate::create_node` interns label NAMES in
//! creation order starting at `1024` (`substrate.rs` `next_label`).
//! Each test documents the resulting id per label inline; the MERGE
//! bind catalog uses the SAME id the substrate assigns so the
//! lowered `Scan{Some(id)}` targets the right label.

use arcgraph_core::{LabelId, Lsn, PartitionId, TenantId};
use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::executor::{ExecutionContext, Pipeline};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{Statement, parse};

const TENANT: TenantId = TenantId::DEFAULT;

/// Walk Parse → Bind → TypeCheck → CrossSubstrate → Lower against a
/// catalog whose interned label set is given by `labels` (name → id).
///
/// Modelling the interned set EXPLICITLY is the point: the MERGE
/// match-branch's label enforcement is decided at BIND time from
/// `lookup_label`, so a label absent from `labels` is "not yet
/// interned" — exactly what the production per-statement
/// `build_catalog_for_tenant` would report when no live node carries
/// that label.
fn lower_with_labels(query: &str, labels: &[(&str, u32)]) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    match &stmt {
        Statement::Read(_) => {}
        other => panic!("expected Read statement, got {other:?}"),
    }
    let mut cat = StubCatalogProvider::new();
    for (name, id) in labels {
        cat = cat.with_label_id(*name, LabelId::new(*id));
    }
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

/// Drive a lowered plan to EOS against the substrate, returning the
/// emitted rows (write-ops emit zero or one binding row + EOS).
fn run(plan: &LogicalPlan, substrate: &StubExecutorSubstrate) -> Vec<Vec<Value>> {
    let ctx = ExecutionContext::new(TENANT, PartitionId::ZERO);
    let mut op = Pipeline::build(plan).expect("pipeline build OK");
    let mut out = Vec::new();
    loop {
        let b = op.next_batch(&ctx, substrate).expect("batch OK");
        if b.is_empty() {
            break;
        }
        for i in 0..b.row_count() {
            out.push(b.row(i).to_vec());
        }
    }
    out
}

/// STRONG oracle helper — the live nodes carrying `label_id`.
fn nodes_with_label(substrate: &StubExecutorSubstrate, label_id: u32) -> Vec<NodeView> {
    substrate
        .scan_nodes(TENANT, Some(LabelId::new(label_id)), Lsn::MAX)
        .expect("scan_nodes OK")
        .into_iter()
        .map(|bn| bn.node)
        .collect()
}

/// STRONG oracle helper — all live nodes (label-agnostic).
fn all_nodes(substrate: &StubExecutorSubstrate) -> Vec<NodeView> {
    substrate
        .scan_nodes(TENANT, None, Lsn::MAX)
        .expect("scan_nodes OK")
        .into_iter()
        .map(|bn| bn.node)
        .collect()
}

fn id_prop(node: &NodeView) -> Option<i64> {
    match node.properties.get("id") {
        Some(Value::Integer(n)) => Some(*n),
        _ => None,
    }
}

// =====================================================================
// Test 1 — cross-label property (ADR-152 Risk #5). FAILS at HEAD.
// =====================================================================

#[test]
fn cross_label_property_merge_creates_new_user_not_matches_account() {
    // `CREATE (:Account {id:42})` then `MERGE (n:User {id:42})` MUST
    // create a NEW :User — the existing :Account {id:42} has the same
    // property but a DIFFERENT label, so the match-branch must NOT
    // fire. At HEAD (pre-amendment) the match-branch was
    // `Scan{None}` + property-filter(id=42) → cross-matched the
    // :Account → no :User created (the bug).
    const ACCOUNT: u32 = 1024; // interned first by CREATE
    const USER: u32 = 1025; // minted second by the MERGE create-branch

    let substrate = StubExecutorSubstrate::new();
    run(
        &lower_with_labels("CREATE (n:Account {id: 42})", &[]),
        &substrate,
    );
    // MERGE bound with User NOT in the catalog (faithful: no live :User
    // exists yet → not interned) → §D-3 LogicalEmpty → create fires.
    run(
        &lower_with_labels("MERGE (n:User {id: 42})", &[("Account", ACCOUNT)]),
        &substrate,
    );

    assert_eq!(all_nodes(&substrate).len(), 2, "a 2nd node was created");
    let users = nodes_with_label(&substrate, USER);
    assert_eq!(users.len(), 1, "exactly one :User node exists");
    assert_eq!(
        id_prop(&users[0]),
        Some(42),
        "the created :User carries the MERGE property bag {{id:42}}"
    );
    let accounts = nodes_with_label(&substrate, ACCOUNT);
    assert_eq!(accounts.len(), 1, ":Account is untouched (not matched)");
    assert_eq!(id_prop(&accounts[0]), Some(42));
}

// =====================================================================
// Test 2 — bare heterogeneous (audit O-3). FAILS at HEAD.
// =====================================================================

#[test]
fn bare_label_merge_on_heterogeneous_graph_creates_user() {
    // `CREATE (:Article)` then bare `MERGE (n:User)` MUST create a
    // :User. At HEAD bare `MERGE (n:User)` lowered to `Scan{None}`
    // with NO property filter → matched the existence of ANY node
    // (the :Article) → never created a :User even though zero :Users
    // exist (the most severe form of the bug).
    const ARTICLE: u32 = 1024;
    const USER: u32 = 1025;

    let substrate = StubExecutorSubstrate::new();
    run(&lower_with_labels("CREATE (n:Article)", &[]), &substrate);
    run(
        &lower_with_labels("MERGE (n:User)", &[("Article", ARTICLE)]),
        &substrate,
    );

    assert_eq!(all_nodes(&substrate).len(), 2, "the :User was created");
    assert_eq!(
        nodes_with_label(&substrate, USER).len(),
        1,
        "exactly one :User node (bare MERGE enforced its label)"
    );
    assert_eq!(
        nodes_with_label(&substrate, ARTICLE).len(),
        1,
        ":Article is untouched"
    );
}

// =====================================================================
// Test 3 — Some(id) match positive (no spurious create).
// =====================================================================

#[test]
fn interned_label_merge_matches_existing_node_no_second_create() {
    // Two statements: `CREATE (:User)` then bare `MERGE (n:User)`.
    // The MERGE's label IS interned (the prior CREATE minted it) →
    // §D-2 `Scan{Some(id)}` finds the live :User → match-branch fires
    // → NO second node created.
    const USER: u32 = 1024;

    let substrate = StubExecutorSubstrate::new();
    run(&lower_with_labels("CREATE (n:User)", &[]), &substrate);
    // Bind with User interned (mirrors production's catalog rebuild
    // after the CREATE commits + the plan-cache `commits_observed`
    // watermark re-bind — see amendment §"Cross-statement is safe").
    run(
        &lower_with_labels("MERGE (n:User)", &[("User", USER)]),
        &substrate,
    );

    assert_eq!(
        all_nodes(&substrate).len(),
        1,
        "MERGE matched the existing :User — no 2nd create"
    );
    assert_eq!(nodes_with_label(&substrate, USER).len(), 1);
}

// =====================================================================
// Test 4 — interned-but-all-deleted (Some(id) scan empty → create).
// =====================================================================

#[test]
fn interned_label_all_deleted_merge_recreates() {
    // `CREATE (n:User {id:42}) DELETE n` interns :User then tombstones
    // the only instance. A later `MERGE (n:User {id:42})` MUST create:
    // the label is interned (→ §D-2 `Scan{Some(id)}`, NOT §D-3 Empty),
    // but the scan returns zero LIVE nodes (tombstone filter) → the
    // create-branch fires.
    const USER: u32 = 1024;

    let substrate = StubExecutorSubstrate::new();
    run(
        &lower_with_labels("CREATE (n:User {id: 42}) DELETE n", &[]),
        &substrate,
    );
    assert_eq!(
        all_nodes(&substrate).len(),
        0,
        "precondition: the only :User was deleted"
    );

    run(
        &lower_with_labels("MERGE (n:User {id: 42})", &[("User", USER)]),
        &substrate,
    );

    let users = nodes_with_label(&substrate, USER);
    assert_eq!(
        users.len(),
        1,
        "MERGE re-created the :User (scan was empty)"
    );
    assert_eq!(id_prop(&users[0]), Some(42));
    assert_eq!(all_nodes(&substrate).len(), 1);
}

// =====================================================================
// Test 5 — F2/D-6 intra-statement residual (EMPIRICAL PIN).
// =====================================================================

#[test]
fn intra_statement_merge_residual_v1_0_alpha_pin() {
    // ADR-152-amendment-01 §D-6 — PIN the actual v1.0-α intra-statement
    // behavior (no env-gate, no skip; documents reality).
    //
    // The pre-code design review (F2) HYPOTHESIZED that the bind-time-
    // baked match decision would regress `CREATE (:User) MERGE (n:User)`
    // from 1 node (HEAD) to 2 (amendment): both clauses bind with
    // `lookup_label("User") = None` → §D-3 Empty → a 2nd :User.
    //
    // EMPIRICALLY THAT REGRESSION DOES NOT MANIFEST (verified here).
    // The ADR-151 statement-composition narrowing makes `lower_merge`
    // DISCARD its `prev` sub-plan, so a leading write-clause before a
    // terminal MERGE is dropped from the executed plan entirely — the
    // MERGE runs against the PRE-statement committed substrate state,
    // which is exactly what its bind-time catalog reflects. Bind-time
    // and execute-time are therefore CONSISTENT, and both scenarios
    // below yield exactly 1 node — matching openCypher's desired count
    // (though by a different provenance: the MERGE create-branch makes
    // the node; the discarded CREATE / first-MERGE never runs).
    //
    // Forward-pin: full multi-write-clause same-statement composition
    // (where the leading CREATE/MERGE SHOULD be visible to a later
    // MERGE) lands with the ADR-151 v1.1 statement-scoped MATCH→MERGE
    // composition. CROSS-statement is already SAFE (test 3 + the
    // `commits_observed` plan-cache watermark re-bind).

    // Scenario A — `CREATE (:User) MERGE (n:User)`: leading CREATE
    // discarded → only the MERGE runs → match-branch Empty → 1 :User.
    let sub_a = StubExecutorSubstrate::new();
    run(
        &lower_with_labels("CREATE (:User) MERGE (n:User)", &[]),
        &sub_a,
    );
    assert_eq!(
        all_nodes(&sub_a).len(),
        1,
        "CREATE (:User) MERGE (n:User) yields 1 node at v1.0-α (leading \
         CREATE discarded by ADR-151 prev-narrowing; no F2 1->2 \
         regression). If this changes, revisit the §D-6 forward-pin."
    );
    assert_eq!(
        nodes_with_label(&sub_a, 1024).len(),
        1,
        "the 1 node is :User"
    );

    // Scenario B — `MERGE (a:New) MERGE (b:New)`: first MERGE discarded
    // → only the last MERGE runs → 1 :New.
    let sub_b = StubExecutorSubstrate::new();
    run(
        &lower_with_labels("MERGE (a:New) MERGE (b:New)", &[]),
        &sub_b,
    );
    assert_eq!(
        all_nodes(&sub_b).len(),
        1,
        "MERGE (a:New) MERGE (b:New) yields 1 node at v1.0-α (first MERGE \
         discarded by ADR-151 prev-narrowing)"
    );
}

// =====================================================================
// Test 6 — property AND label BOTH enforced. FAILS at HEAD.
// =====================================================================

#[test]
fn merge_enforces_label_and_property_together() {
    // Pre-state: `:Account {id:42}` (right prop, wrong label) and
    // `:User {id:99}` (right label, wrong prop). `MERGE (n:User {id:42})`
    // must match NEITHER → create a fresh `:User {id:42}`.
    //
    // This DISTINGUISHES "both enforced" from either partial behavior:
    //   * label-only (no prop filter): would match :User {id:99} → no create.
    //   * property-only (HEAD behavior): would match :Account {id:42} → no create.
    //   * BOTH (correct): matches neither → CREATE.
    // Only the correct behavior yields a 3rd node, so node-count alone
    // discriminates — and we additionally assert per-label populations.
    const ACCOUNT: u32 = 1024; // CREATE'd first
    const USER: u32 = 1025; // CREATE'd second

    let substrate = StubExecutorSubstrate::new();
    run(
        &lower_with_labels("CREATE (n:Account {id: 42})", &[]),
        &substrate,
    );
    run(
        &lower_with_labels("CREATE (n:User {id: 99})", &[]),
        &substrate,
    );
    // MERGE bound with User interned → §D-2 Scan{Some(USER)} + filter(id=42).
    run(
        &lower_with_labels(
            "MERGE (n:User {id: 42})",
            &[("User", USER), ("Account", ACCOUNT)],
        ),
        &substrate,
    );

    assert_eq!(
        all_nodes(&substrate).len(),
        3,
        "label+property both enforced: matched neither the wrong-label \
         :Account {{id:42}} nor the wrong-property :User {{id:99}} → created"
    );
    let users = nodes_with_label(&substrate, USER);
    assert_eq!(
        users.len(),
        2,
        "two :User nodes: {{id:99}} + the new {{id:42}}"
    );
    let mut user_ids: Vec<i64> = users.iter().filter_map(id_prop).collect();
    user_ids.sort_unstable();
    assert_eq!(user_ids, vec![42, 99], ":User population is {{42, 99}}");
    let accounts = nodes_with_label(&substrate, ACCOUNT);
    assert_eq!(
        accounts.len(),
        1,
        ":Account {{id:42}} untouched (wrong label)"
    );
    assert_eq!(id_prop(&accounts[0]), Some(42));
}
