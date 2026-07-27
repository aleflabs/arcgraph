//! [`ArcGraphExecutor`] — runs queries against
//! [`arcgraph_query::QueryEngine`] using a [`StubExecutorSubstrate`]
//! seeded with a minimal fixture.
//!
//! # Why a stub substrate (not the storage one)
//!
//! The TCK harness consumes self-contained Gherkin features that bring
//! their own setup data (`having executed: """CREATE..."""`). The W17α
//! storage executor substrate is read-only against the seeded data set,
//! and `CREATE` is not yet supported at v1.0-alpha (ADR-006 amendment-01).
//! So this executor uses the in-memory `StubExecutorSubstrate` + a fixed
//! micro-fixture, which lets MATCH-shape features close end-to-end at
//! v1.0-alpha without depending on the CREATE-on-storage path.
//!
//! When the M4-61b / M5-08 CREATE landing flips, this executor swaps to
//! the storage substrate without changing the [`TckExecutor`] surface.

use std::sync::Arc;

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

use super::{ExecutorError, RowSet, TckExecutor};

/// In-process ArcGraph TCK executor.
pub struct ArcGraphExecutor {
    catalog: Arc<StubCatalogProvider>,
    substrate: Arc<StubExecutorSubstrate>,
}

impl std::fmt::Debug for ArcGraphExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArcGraphExecutor").finish_non_exhaustive()
    }
}

impl ArcGraphExecutor {
    /// Build the executor with the canonical TCK micro-fixture:
    ///
    /// - 5 `:Person` nodes named `Alice`..`Eve` with `age` ∈ {25,30,35,40,45}.
    /// - 4 `:KNOWS` edges (Alice→Bob, Bob→Carol, Carol→Dave, Dave→Eve).
    /// - 3 `:Doc` nodes (the OPTIONAL MATCH category exercises them).
    ///
    /// All `lookup_label` / `lookup_rel_type` IDs are deterministic
    /// (Person=1, Doc=2, KNOWS=1) so the differ output is stable across
    /// runs.
    pub fn new() -> Self {
        let catalog = StubCatalogProvider::new()
            .with_labels(["Person", "Doc"])
            .with_rel_types(["KNOWS"])
            .with_properties(["name", "age", "title"]);

        let mut substrate = StubExecutorSubstrate::new();
        let persons = [
            ("Alice", 25i64),
            ("Bob", 30),
            ("Carol", 35),
            ("Dave", 40),
            ("Eve", 45),
        ];
        for (i, (name, age)) in persons.iter().enumerate() {
            let nid = NodeId::new((i + 1) as u64);
            substrate = substrate.with_node(
                TenantId::DEFAULT,
                NodeView::new(nid, Some(LabelId::new(1)))
                    .with_property("name", Value::String((*name).into()))
                    .with_property("age", Value::Integer(*age)),
            );
        }
        for i in 0..4 {
            let from = NodeId::new((i + 1) as u64);
            let to = NodeId::new((i + 2) as u64);
            substrate = substrate.with_edge(
                TenantId::DEFAULT,
                RelView::new(RelId::new((i + 1) as u64), from, to, Some(TypeId::new(1))),
            );
        }
        for i in 0..3 {
            let nid = NodeId::new((6 + i) as u64);
            substrate = substrate.with_node(
                TenantId::DEFAULT,
                NodeView::new(nid, Some(LabelId::new(2)))
                    .with_property("title", Value::String(format!("doc-{i}"))),
            );
        }

        Self {
            catalog: Arc::new(catalog),
            substrate: Arc::new(substrate),
        }
    }
}

impl Default for ArcGraphExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl TckExecutor for ArcGraphExecutor {
    fn name(&self) -> &'static str {
        "ArcGraph"
    }

    fn execute(&self, cypher: &str) -> Result<RowSet, ExecutorError> {
        let engine = QueryEngine::new(self.catalog.as_ref());
        let result = engine
            .execute(cypher, self.substrate.as_ref())
            .map_err(|e| {
                let msg = e.to_string();
                if msg.contains("parse")
                    || msg.contains("binding")
                    || msg.contains("type")
                    || msg.contains("plan")
                {
                    ExecutorError::PlanBuild(msg)
                } else {
                    ExecutorError::Execution(msg)
                }
            })?;
        let rows: Vec<Vec<String>> = result
            .rows()
            .iter()
            .map(|row| row.iter().map(stringify).collect())
            .collect();
        Ok(RowSet {
            columns: None,
            rows,
        })
    }
}

/// Stringify a `Value` for the differ. The serialization is
/// intentionally lossy: it loses type information but is comparable
/// across executor backends.
fn stringify(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => format!("{f:?}"),
        Value::String(s) => s.clone(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_returns_five_persons() {
        let exec = ArcGraphExecutor::new();
        let rs = exec
            .execute("MATCH (n:Person) RETURN n.name")
            .expect("scan must succeed");
        // Stub substrate scan_nodes returns 5 :Person nodes; one
        // string column per row.
        assert_eq!(rs.rows.len(), 5);
    }

    #[test]
    fn fixture_returns_three_docs() {
        let exec = ArcGraphExecutor::new();
        let rs = exec
            .execute("MATCH (n:Doc) RETURN n.title")
            .expect("Doc scan must succeed");
        assert_eq!(rs.rows.len(), 3);
    }
}
