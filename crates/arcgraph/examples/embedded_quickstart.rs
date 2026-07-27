//! Embedded-library quickstart for ArcGraph (M6-01 demo).
//!
//! Demonstrates the minimal loop an embedded caller follows:
//!
//! 1. Open a catalog (this v1.0-alpha example uses the in-memory
//!    [`StubCatalogProvider`] / [`StubExecutorSubstrate`] pair —
//!    storage-backed wiring lands at M4-08+).
//! 2. Ingest ~100 nodes.
//! 3. Run a MATCH query through [`QueryEngine::execute`].
//! 4. Inspect the materialized result (row count + first row).
//!
//! Run with:
//!
//! ```text
//! cargo run -p arcgraph --example embedded_quickstart
//! ```

use arcgraph::core::{LabelId, NodeId, TenantId};
use arcgraph::query::QueryEngine;
use arcgraph::query::executor::StubExecutorSubstrate;
use arcgraph::query::executor::value::{NodeView, Value};
use arcgraph::query::semantic::StubCatalogProvider;

const NODE_COUNT: u64 = 100;

fn main() {
    // 1. Catalog: register the "Person" label + an "age" property so
    //    the binder accepts our query.
    let catalog = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["age"]);

    // 2. Substrate: ingest NODE_COUNT Person nodes with age = i.
    let mut substrate = StubExecutorSubstrate::new();
    for i in 1..=NODE_COUNT {
        substrate = substrate.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(i as i64)),
        );
    }

    // 3. Run a MATCH+WHERE query. Predicate `n.age > 50` keeps
    //    age 51..=100 → 50 rows.
    let engine = QueryEngine::new(&catalog);
    let result = engine
        .execute("MATCH (n:Person) WHERE n.age > 50 RETURN n.age", &substrate)
        .expect("execute MATCH+WHERE");

    // 4. Inspect: print row count + first row + last row.
    println!("ingested {NODE_COUNT} Person nodes");
    println!("MATCH (n:Person) WHERE n.age > 50 RETURN n.age");
    println!("  → {} rows materialized", result.rows().len());
    if let Some(first) = result.rows().first() {
        println!("  → first row: {first:?}");
    }
    if let Some(last) = result.rows().last() {
        println!("  → last row:  {last:?}");
    }

    // The example double-asserts so it doubles as a smoke test (run
    // via the integration test in `tests/embedded_quickstart_example.rs`).
    assert_eq!(result.rows().len(), 50, "predicate keeps ages 51..=100");
    assert!(
        result.truncation.is_none(),
        "no per-tenant memory-budget truncation expected at 100 rows",
    );
}
