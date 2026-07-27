//! ADR-149 W26-θ Phase 3 — DELETE proptest.
//!
//! Random labels + DETACH-flag combinations should:
//! 1. Parse cleanly.
//! 2. Round-trip through Display.
//! 3. Bind + type-check + cross-substrate validate.
//! 4. Lower to a plan containing a `LogicalPlan::Delete`.
//! 5. Execute CREATE-then-DELETE end-to-end against
//!    `StubExecutorSubstrate` — post-delete `scan_nodes` reports 0
//!    visible CREATE-introduced nodes.

use arcgraph_core::{PartitionId, TenantId};

use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::ExecutionContext;
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{executor::Pipeline, parse};
use proptest::prelude::*;

fn label_strategy() -> impl Strategy<Value = String> {
    // The canonical reserved-word set lives in grammar.pest; keep this
    // self-maintaining by generating labels that cannot equal a bare
    // keyword.
    "[A-Z][A-Za-z0-9_]{0,8}".prop_map(|s| format!("L_{s}"))
}

/// Walk Parse → Bind → TypeCheck → CrossSubstrate → Lower.
fn lower(query: &str) -> Result<LogicalPlan, String> {
    let stmt = parse(query).map_err(|e| format!("parse: {e:?}"))?;
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).map_err(|e| format!("bind: {e:?}"))?;
    TypeCheckVisitor::check(&mut bound, &cat).map_err(|e| format!("typecheck: {e:?}"))?;
    CrossSubstrateValidator::validate(&bound, &cat)
        .map_err(|e| format!("cross-substrate: {e:?}"))?;
    LogicalPlanLoweringVisitor::lower(&bound).map_err(|e| format!("lower: {e:?}"))
}

fn has_delete(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Delete(_) => true,
        LogicalPlan::Filter(f) => has_delete(&f.input),
        LogicalPlan::Project(p) => has_delete(&p.input),
        LogicalPlan::Limit(l) => has_delete(&l.input),
        LogicalPlan::Skip(s) => has_delete(&s.input),
        LogicalPlan::DynamicLimit(d) => has_delete(&d.input),
        LogicalPlan::Sort(s) => has_delete(&s.input),
        LogicalPlan::Distinct(d) => has_delete(&d.input),
        LogicalPlan::Unwind(u) => has_delete(&u.input),
        LogicalPlan::ProcedureCall(p) => has_delete(&p.input),
        LogicalPlan::Aggregate(a) => has_delete(&a.input),
        LogicalPlan::CommunityLookup(c) => has_delete(&c.input),
        LogicalPlan::NamedPath(np) => has_delete(&np.input),
        LogicalPlan::Join(j) => has_delete(&j.left) || has_delete(&j.right),
        LogicalPlan::LeftOuterJoin(j) => has_delete(&j.left) || has_delete(&j.right),
        LogicalPlan::Fusion(f) => f.inputs.iter().any(|inp| has_delete(inp)),
        LogicalPlan::Union(u) => u.arms.iter().any(has_delete),
        LogicalPlan::CreateRel(c) => has_delete(&c.source_plan) || has_delete(&c.target_plan),
        LogicalPlan::Set(s) => has_delete(&s.input),
        LogicalPlan::Remove(r) => has_delete(&r.input),
        LogicalPlan::Merge(m) => has_delete(&m.match_branch) || has_delete(&m.create_branch),
        LogicalPlan::Scan(_)
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Expand(_)
        | LogicalPlan::Empty(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::CreateNode(_)
        | LogicalPlan::CreateVectorIndex(_)
        | LogicalPlan::CreatePropertyIndex(_)
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => false,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// ADR-149 §D-1 / §D-2: random CREATE-then-DELETE queries parse +
    /// round-trip through Display.
    #[test]
    fn delete_random_label_parses_and_roundtrips(label in label_strategy()) {
        let q = format!("CREATE (n:{label}) DELETE n");
        let parsed = parse(&q).expect("parse OK");
        let printed = format!("{parsed}");
        let re_parsed = parse(&printed).expect("re-parse OK");
        prop_assert_eq!(parsed, re_parsed, "Display round-trips");
    }

    /// ADR-149 §D-1 / §D-2: DETACH DELETE forms also round-trip.
    #[test]
    fn detach_delete_random_label_parses_and_roundtrips(label in label_strategy()) {
        let q = format!("CREATE (n:{label}) DETACH DELETE n");
        let parsed = parse(&q).expect("parse OK");
        let printed = format!("{parsed}");
        let re_parsed = parse(&printed).expect("re-parse OK");
        prop_assert_eq!(parsed, re_parsed, "Display round-trips with DETACH");
        prop_assert!(printed.contains("DETACH"), "DETACH prefix preserved");
    }

    /// End-to-end: each CREATE-then-DELETE pair tombstones the
    /// just-created node. Post-delete `scan_nodes` reports 0.
    #[test]
    fn create_then_delete_round_trip_tombstones_node(label in label_strategy()) {
        let q = format!("CREATE (n:{label}) DELETE n");
        let plan = lower(&q).expect("lower OK");
        prop_assert!(has_delete(&plan), "Delete present");
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = Pipeline::build(&plan).expect("pipeline build OK");
        loop {
            let b = op.next_batch(&ctx, &s).expect("batch OK");
            if b.is_empty() {
                break;
            }
        }
        let nodes = s
            .scan_nodes(TenantId::DEFAULT, None, arcgraph_core::Lsn::MAX)
            .expect("scan_nodes OK");
        prop_assert_eq!(nodes.len(), 0, "node tombstoned post-DELETE");
    }

    /// DETACH variant lowers with detach=true; bare form lowers with
    /// detach=false. The grammar's `detach?` production is structural.
    #[test]
    fn delete_detach_flag_threads_through_lower(label in label_strategy()) {
        let bare_q = format!("CREATE (n:{label}) DELETE n");
        let detach_q = format!("CREATE (n:{label}) DETACH DELETE n");
        let bare_plan = lower(&bare_q).expect("bare lower OK");
        let detach_plan = lower(&detach_q).expect("detach lower OK");
        prop_assert!(!plan_detach(&bare_plan), "bare = detach false");
        prop_assert!(plan_detach(&detach_plan), "detach = detach true");
    }
}

fn plan_detach(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Delete(d) => d.detach,
        LogicalPlan::Filter(f) => plan_detach(&f.input),
        LogicalPlan::Project(p) => plan_detach(&p.input),
        LogicalPlan::Limit(l) => plan_detach(&l.input),
        LogicalPlan::Skip(s) => plan_detach(&s.input),
        LogicalPlan::DynamicLimit(d) => plan_detach(&d.input),
        LogicalPlan::Sort(s) => plan_detach(&s.input),
        LogicalPlan::Distinct(d) => plan_detach(&d.input),
        LogicalPlan::Unwind(u) => plan_detach(&u.input),
        LogicalPlan::ProcedureCall(p) => plan_detach(&p.input),
        LogicalPlan::Aggregate(a) => plan_detach(&a.input),
        LogicalPlan::CommunityLookup(c) => plan_detach(&c.input),
        LogicalPlan::NamedPath(np) => plan_detach(&np.input),
        _ => false,
    }
}
