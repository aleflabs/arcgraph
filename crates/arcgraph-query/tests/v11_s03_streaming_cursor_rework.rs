//! V11-S-03 R1 rework oracles: result-side OOM defense plus
//! streaming variable-length correctness pins.

use std::collections::HashSet;

use arcgraph_core::{LabelId, NodeId, PartitionId, RelId, TenantId, TypeId};
use arcgraph_query::executor::value::{NodeView, RelView};
use arcgraph_query::executor::{
    ExecutionContext, ExecutionError, MemoryBudget, StubExecutorSubstrate, Value,
    execute_with_context,
};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    ArcQLError, BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider,
    TypeCheckVisitor,
};
use arcgraph_query::{QueryEngine, StreamingCursor, parse};

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Start", "N"])
        .with_rel_types(["R"])
        .with_properties(["name"])
}

fn lower(query: &str, catalog: &StubCatalogProvider) -> LogicalPlan {
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, catalog).expect("bind");
    TypeCheckVisitor::check(&mut bound, catalog).expect("type-check");
    CrossSubstrateValidator::validate(&bound, catalog).expect("cross-substrate");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
}

fn node(id: u64, label: u32) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(label)))
        .with_property("name", Value::String(format!("n{id}")))
}

fn edge(rel: u64, src: u64, dst: u64) -> RelView {
    RelView::new(
        RelId::new(rel),
        NodeId::new(src),
        NodeId::new(dst),
        Some(TypeId::new(1)),
    )
}

fn add_three_hop_pyramid(
    mut s: StubExecutorSubstrate,
    root: u64,
    layer_base: u64,
    rel_start: u64,
    fanout: u64,
) -> StubExecutorSubstrate {
    s = s.with_node(TenantId::DEFAULT, node(root, 1));
    for id in layer_base..layer_base + fanout {
        s = s.with_node(TenantId::DEFAULT, node(id, 2));
    }
    for id in layer_base + 1_000..layer_base + 1_000 + fanout {
        s = s.with_node(TenantId::DEFAULT, node(id, 2));
    }
    for id in layer_base + 2_000..layer_base + 2_000 + fanout {
        s = s.with_node(TenantId::DEFAULT, node(id, 2));
    }

    let mut rel_id = rel_start;
    for dst in layer_base..layer_base + fanout {
        s = s.with_edge(TenantId::DEFAULT, edge(rel_id, root, dst));
        rel_id += 1;
    }
    for src in layer_base..layer_base + fanout {
        for dst in layer_base + 1_000..layer_base + 1_000 + fanout {
            s = s.with_edge(TenantId::DEFAULT, edge(rel_id, src, dst));
            rel_id += 1;
        }
    }
    for src in layer_base + 1_000..layer_base + 1_000 + fanout {
        for dst in layer_base + 2_000..layer_base + 2_000 + fanout {
            s = s.with_edge(TenantId::DEFAULT, edge(rel_id, src, dst));
            rel_id += 1;
        }
    }
    s
}

/// #980 NIT-1 (was: V11-S-03 result-side OOM oracle) — the eager
/// result-Vec path in `execute_with_context` no longer imposes the OLD
/// fixed 131 072-row clip on the UNCAPPED budget path. A 60³ = 216 000
/// row result (above the former `BUDGET_FALLBACK_ROWS` ceiling) now
/// MATERIALIZES IN FULL on the uncapped (default) path — the fixed
/// row-count valve was a GA-blocker (#980), not a correctness floor.
///
/// The genuine result-side OOM defense lives on the per-tenant BYTE cap
/// (see `q3_byte_capped_tenant_still_oom_defends_at_result_tail`); the
/// uncapped path is an explicit "no memory limit" choice guarded only by
/// the far-larger `UNCAPPED_RUNAWAY_GUARD_ROWS`.
#[test]
fn q3_uncapped_result_tail_materializes_full_past_old_ceiling() {
    const FANOUT: u64 = 60; // 60^3 = 216_000, above the old 131 072 valve.
    let s = add_three_hop_pyramid(StubExecutorSubstrate::new(), 1, 10_000, 1, FANOUT);
    let cat = catalog();
    let plan = lower("MATCH (a:Start)-[:R*3..3]->(b) RETURN b", &cat);
    let ctx = ExecutionContext::new(cat.tenant(), PartitionId::ZERO);
    assert!(
        !ctx.budget().has_cap(ctx.tenant()),
        "this oracle pins the UNCAPPED path"
    );

    let rows = execute_with_context(&plan, &s, &ctx)
        .expect("uncapped result tail must materialize, not hit the old 131K ceiling");
    let expected = (FANOUT * FANOUT * FANOUT) as usize; // 216_000
    assert_eq!(
        rows.len(),
        expected,
        "every 3-hop result row must materialize past the old ceiling"
    );
}

/// #980 NIT-1 (consistency) — the eager result-Vec path is now
/// `has_cap`-aware (mirroring the 6 operator-level fixes), so a BUDGETED
/// tenant is likewise NOT clipped at the old fixed 131 072-row boundary:
/// the byte budget governs instead. Here a generous byte cap admits the
/// full 216 000-row result, proving the row-count valve no longer fires
/// on the capped path.
///
/// NOTE — honesty pin: `execute_with_context`'s eager result-Vec is the
/// public `execute()` / PROFILE surface, NOT the streaming-cursor
/// `materialize` path that the user-facing MCP `raw_query` + Bolt RUN
/// route through. The byte budget is enforced by the OPERATOR layer
/// (e.g. spillover reservations), not by this drain loop; this loop's
/// only job after #980 is to grow with the actual cardinality (uncapped)
/// without imposing the obsolete fixed row ceiling. We therefore assert
/// the capped tenant materializes in full under a generous cap rather
/// than over-claiming a byte trip this drain loop does not itself
/// perform.
#[test]
fn q3_byte_capped_tenant_not_clipped_at_old_row_boundary() {
    const FANOUT: u64 = 60; // 60^3 = 216_000 rows, above the old valve.
    let s = add_three_hop_pyramid(StubExecutorSubstrate::new(), 1, 10_000, 1, FANOUT);
    let cat = catalog();
    let plan = lower("MATCH (a:Start)-[:R*3..3]->(b) RETURN b", &cat);
    // A generous 8 GiB cap that comfortably admits the full result — the
    // point is that the ROW-count clip no longer fires for a capped
    // tenant at the old 131 072 boundary.
    let budget = MemoryBudget::with_per_tenant_cap(cat.tenant(), 8 * 1024 * 1024 * 1024);
    let ctx = ExecutionContext::new(cat.tenant(), PartitionId::ZERO).with_budget(budget);
    assert!(
        ctx.budget().has_cap(ctx.tenant()),
        "this oracle pins the CAPPED path"
    );

    let rows = execute_with_context(&plan, &s, &ctx)
        .expect("capped tenant within byte budget must not be clipped at the old row boundary");
    let expected = (FANOUT * FANOUT * FANOUT) as usize; // 216_000
    assert_eq!(rows.len(), expected);
}

fn q2_fixture() -> (StubExecutorSubstrate, Vec<(u64, u64, u64)>) {
    let edges = vec![
        (10, 1, 2),
        (11, 1, 3),
        (20, 2, 4),
        (21, 3, 4),
        (30, 4, 5),
        (40, 5, 2),
    ];
    let mut s = StubExecutorSubstrate::new().with_node(TenantId::DEFAULT, node(1, 1));
    for id in 2..=5 {
        s = s.with_node(TenantId::DEFAULT, node(id, 2));
    }
    for (rel, src, dst) in &edges {
        s = s.with_edge(TenantId::DEFAULT, edge(*rel, *src, *dst));
    }
    (s, edges)
}

fn independent_unbounded_to_ids(edges: &[(u64, u64, u64)], start: u64) -> Vec<u64> {
    fn dfs(
        edges: &[(u64, u64, u64)],
        node: u64,
        depth: u32,
        visited: &mut HashSet<u64>,
        out: &mut Vec<u64>,
    ) {
        if depth == 5 {
            return;
        }
        for (rel, src, dst) in edges {
            if *src != node || visited.contains(rel) {
                continue;
            }
            visited.insert(*rel);
            out.push(*dst);
            dfs(edges, *dst, depth + 1, visited, out);
            visited.remove(rel);
        }
    }

    let mut out = Vec::new();
    dfs(edges, start, 0, &mut HashSet::new(), &mut out);
    out.sort_unstable();
    out
}

#[test]
fn q2_streaming_unbounded_parity_with_independent_edge_unique_enumerator() {
    let (s, edges) = q2_fixture();
    let cat = catalog();
    let rows = QueryEngine::new(&cat)
        .execute("MATCH (a:Start)-[:R*]->(b) RETURN b", &s)
        .expect("execute unbounded cycle+diamond fixture");
    let mut got: Vec<u64> = rows
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::Node(n) => n.id.raw(),
            other => panic!("expected node result, got {other:?}"),
        })
        .collect();
    got.sort_unstable();

    assert_eq!(got, independent_unbounded_to_ids(&edges, 1));
}

#[test]
fn q4_streaming_unbounded_ceiling_five_errors_structured() {
    let mut s = StubExecutorSubstrate::new().with_node(TenantId::DEFAULT, node(1, 1));
    for id in 2..=7 {
        s = s.with_node(TenantId::DEFAULT, node(id, 2));
    }
    for id in 1..=6 {
        s = s.with_edge(TenantId::DEFAULT, edge(id * 10, id, id + 1));
    }
    let cat = catalog();
    let plan = lower("MATCH (a:Start)-[:R*]->(b) RETURN b", &cat);
    let ctx = ExecutionContext::new(cat.tenant(), PartitionId::ZERO);
    let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open cursor");

    match cursor.next_batch() {
        Err(ExecutionError::Plan(ArcQLError::ResourceExhausted { feature, .. })) => assert!(
            feature.contains("depth cap"),
            "Q4 must surface the depth-cap ResourceExhausted, got {feature}"
        ),
        other => panic!("expected structured depth-cap error, got {other:?}"),
    }
}
