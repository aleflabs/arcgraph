//! #965 — var-length named paths must hydrate intermediate nodes.
//!
//! `PlainPathOp` used to synthesize intermediate nodes from relationship
//! endpoints as id-only `NodeView`s. `id(n)` stayed correct while
//! properties and labels silently disappeared from `nodes(p)` and
//! `RETURN p`.

use arcgraph_core::{LabelId, Lsn, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::substrate::{
    BoundEdge, BoundNode, ExecutorSubstrate, RankedHit, SubstrateAccessError,
};
use arcgraph_query::executor::value::{NodeView, PathView, RelView, Value};
use arcgraph_query::logical_plan::Direction;
use arcgraph_query::semantic::StubCatalogProvider;

const P_LABEL: u32 = 1;
const R_TYPE: u32 = 1;

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["P"])
        .with_rel_types(["R"])
        .with_properties(["id", "rank"])
}

fn node(raw: u64, id: &str, rank: i64) -> NodeView {
    NodeView::new(NodeId::new(raw), Some(LabelId::new(P_LABEL)))
        .with_label_name("P")
        .with_property("id", Value::String(id.to_string()))
        .with_property("rank", Value::Integer(rank))
}

fn rel(raw: u64, from: u64, to: u64) -> RelView {
    RelView::new(
        RelId::new(raw),
        NodeId::new(from),
        NodeId::new(to),
        Some(TypeId::new(R_TYPE)),
    )
}

fn chain_graph() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node(1, "a", 10))
        .with_node(TenantId::DEFAULT, node(2, "b", 20))
        .with_node(TenantId::DEFAULT, node(3, "c", 30))
        .with_node(TenantId::DEFAULT, node(4, "d", 40))
        .with_edge(TenantId::DEFAULT, rel(100, 1, 2))
        .with_edge(TenantId::DEFAULT, rel(101, 2, 3))
        .with_edge(TenantId::DEFAULT, rel(102, 3, 4))
}

fn shared_intermediate_graph() -> StubExecutorSubstrate {
    chain_graph()
        .with_node(TenantId::DEFAULT, node(5, "e", 50))
        .with_edge(TenantId::DEFAULT, rel(103, 3, 5))
}

fn run<S: ExecutorSubstrate>(query: &str, substrate: &S) -> Vec<Vec<Value>> {
    QueryEngine::new(&catalog())
        .execute(query, substrate)
        .unwrap_or_else(|e| panic!("execute must not error for `{query}`: {e:?}"))
        .rows()
        .to_vec()
}

fn only_row(rows: Vec<Vec<Value>>) -> Vec<Value> {
    match rows.as_slice() {
        [row] => row.clone(),
        other => panic!("expected one row, got {other:?}"),
    }
}

fn strings(cell: &Value) -> Vec<String> {
    match cell {
        Value::List(xs) => xs
            .iter()
            .map(|x| match x {
                Value::String(s) => s.clone(),
                Value::Null => "<null>".to_string(),
                other => panic!("expected string/null list item, got {other:?}"),
            })
            .collect(),
        other => panic!("expected list, got {other:?}"),
    }
}

fn ints(cell: &Value) -> Vec<i64> {
    match cell {
        Value::List(xs) => xs
            .iter()
            .map(|x| match x {
                Value::Integer(n) => *n,
                Value::Null => -1,
                other => panic!("expected integer/null list item, got {other:?}"),
            })
            .collect(),
        other => panic!("expected list, got {other:?}"),
    }
}

fn label_lists(cell: &Value) -> Vec<Vec<String>> {
    match cell {
        Value::List(xs) => xs
            .iter()
            .map(|x| match x {
                Value::List(labels) => labels
                    .iter()
                    .map(|label| match label {
                        Value::String(s) => s.clone(),
                        other => panic!("expected label string, got {other:?}"),
                    })
                    .collect(),
                other => panic!("expected nested label list, got {other:?}"),
            })
            .collect(),
        other => panic!("expected list, got {other:?}"),
    }
}

fn as_path(cell: &Value) -> &PathView {
    match cell {
        Value::Path(p) => p,
        other => panic!("expected path, got {other:?}"),
    }
}

fn path_ids(path: &PathView) -> Vec<String> {
    path.nodes()
        .iter()
        .map(|n| match n.properties.get("id") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => panic!("expected string id property, got {other:?}"),
            None => "<null>".to_string(),
        })
        .collect()
}

fn path_ranks(path: &PathView) -> Vec<i64> {
    path.nodes()
        .iter()
        .map(|n| match n.properties.get("rank") {
            Some(Value::Integer(rank)) => *rank,
            Some(other) => panic!("expected integer rank property, got {other:?}"),
            None => -1,
        })
        .collect()
}

fn path_labels(path: &PathView) -> Vec<Vec<String>> {
    path.nodes()
        .iter()
        .map(|n| n.label_name.iter().cloned().collect())
        .collect()
}

#[test]
fn varlen_named_path_hydrates_intermediate_nodes_in_nodes_projection() {
    let row = only_row(run(
        "MATCH p=(:P {id:'a'})-[:R*1..5]->(:P {id:'d'}) \
         RETURN [n IN nodes(p) | n.id], [n IN nodes(p) | labels(n)], \
                [n IN nodes(p) | n.rank]",
        &chain_graph(),
    ));

    assert_eq!(strings(&row[0]), ["a", "b", "c", "d"]);
    assert_eq!(
        label_lists(&row[1]),
        vec![
            vec!["P".to_string()],
            vec!["P".to_string()],
            vec!["P".to_string()],
            vec!["P".to_string()],
        ]
    );
    assert_eq!(ints(&row[2]), [10, 20, 30, 40]);
}

#[test]
fn fixed_length_named_path_regression_stays_hydrated() {
    let row = only_row(run(
        "MATCH p=(:P {id:'a'})-[:R]->(:P)-[:R]->(:P)-[:R]->(:P {id:'d'}) \
         RETURN [n IN nodes(p) | n.id], [n IN nodes(p) | labels(n)], \
                [n IN nodes(p) | n.rank]",
        &chain_graph(),
    ));

    assert_eq!(strings(&row[0]), ["a", "b", "c", "d"]);
    assert_eq!(
        label_lists(&row[1]),
        vec![
            vec!["P".to_string()],
            vec!["P".to_string()],
            vec!["P".to_string()],
            vec!["P".to_string()],
        ]
    );
    assert_eq!(ints(&row[2]), [10, 20, 30, 40]);
}

#[test]
fn start_and_end_bindings_remain_hydrated() {
    let row = only_row(run(
        "MATCH p=(s:P {id:'a'})-[:R*1..5]->(e:P {id:'d'}) \
         RETURN s.id, labels(s), s.rank, e.id, labels(e), e.rank",
        &chain_graph(),
    ));

    assert_eq!(row[0], Value::String("a".to_string()));
    assert_eq!(
        label_lists(&Value::List(vec![row[1].clone()])),
        vec![vec!["P".to_string()]]
    );
    assert_eq!(row[2], Value::Integer(10));
    assert_eq!(row[3], Value::String("d".to_string()));
    assert_eq!(
        label_lists(&Value::List(vec![row[4].clone()])),
        vec![vec!["P".to_string()]]
    );
    assert_eq!(row[5], Value::Integer(40));
}

#[test]
fn return_whole_path_carries_hydrated_intermediate_nodes() {
    let row = only_row(run(
        "MATCH p=(:P {id:'a'})-[:R*1..5]->(:P {id:'d'}) RETURN p",
        &chain_graph(),
    ));
    let path = as_path(&row[0]);

    assert_eq!(path_ids(path), ["a", "b", "c", "d"]);
    assert_eq!(path_labels(path), vec![vec!["P".to_string()]; 4]);
    assert_eq!(path_ranks(path), [10, 20, 30, 40]);
}

#[test]
fn memo_dedup_hydrates_shared_intermediates_across_paths() {
    let rows = run(
        "MATCH p=(:P {id:'a'})-[:R*1..5]->(z:P) \
         WHERE z.id IN ['d', 'e'] \
         RETURN [n IN nodes(p) | n.id], [n IN nodes(p) | n.rank] \
         ORDER BY z.id",
        &shared_intermediate_graph(),
    );

    assert_eq!(rows.len(), 2);
    assert_eq!(strings(&rows[0][0]), ["a", "b", "c", "d"]);
    assert_eq!(ints(&rows[0][1]), [10, 20, 30, 40]);
    assert_eq!(strings(&rows[1][0]), ["a", "b", "c", "e"]);
    assert_eq!(ints(&rows[1][1]), [10, 20, 30, 50]);
}

#[derive(Debug, Clone)]
struct DefaultPointReadSubstrate(StubExecutorSubstrate);

impl ExecutorSubstrate for DefaultPointReadSubstrate {
    fn scan_nodes(
        &self,
        tenant: TenantId,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        self.0.scan_nodes(tenant, label, read_lsn)
    }

    fn expand(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        self.0.expand(tenant, from, rel_type, direction, read_lsn)
    }

    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        self.0
            .vector_search(tenant, property, query_vec, k, read_lsn)
    }

    fn bm25_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_text: &str,
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        self.0
            .bm25_search(tenant, property, query_text, k, read_lsn)
    }

    fn community_members(
        &self,
        tenant: TenantId,
        community_id: i64,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        self.0.community_members(tenant, community_id, read_lsn)
    }
}

#[test]
fn default_point_read_substrate_gracefully_degrades_to_id_only_intermediates() {
    let substrate = DefaultPointReadSubstrate(chain_graph());
    let row = only_row(run(
        "MATCH p=(:P {id:'a'})-[:R*1..5]->(:P {id:'d'}) \
         RETURN [n IN nodes(p) | n.id], [n IN nodes(p) | labels(n)], \
                [n IN nodes(p) | n.rank]",
        &substrate,
    ));

    assert_eq!(strings(&row[0]), ["a", "<null>", "<null>", "d"]);
    assert_eq!(
        label_lists(&row[1]),
        vec![vec!["P".to_string()], vec![], vec![], vec!["P".to_string()]]
    );
    assert_eq!(ints(&row[2]), [10, -1, -1, 40]);
}
