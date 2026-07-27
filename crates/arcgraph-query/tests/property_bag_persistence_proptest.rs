//! ADR-152 W27-α — property-bag persistence proptest.
//!
//! Property test: random property bags round-trip through the
//! StubExecutorSubstrate's create_node + scan_nodes path. Each
//! generated bag is materialized, CREATEd into the substrate, then
//! scanned back; the round-trip preserves every key/value pair.
//!
//! This proptest pins the property-bag round-trip semantics at the
//! executor's substrate-trait layer; the integration smoke at
//! `create_then_match_by_property_smoke.rs` separately pins the
//! end-to-end MATCH-by-property predicate filter.

use std::collections::BTreeMap;

use arcgraph_core::{Lsn, NodeId, TenantId};
use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;

use proptest::collection::vec;
use proptest::prelude::*;

/// Generate a single property-bag entry. The key is a non-empty
/// ASCII alpha-numeric identifier; the value is a finite scalar
/// `Value` variant (no NaN / Inf per Value::to_json_value's lossy
/// edge — the proptest excludes them per the W12γ materialize_proptest
/// precedent inherited from the executor value module's doc).
fn scalar_value_strategy() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Boolean),
        any::<i64>().prop_map(Value::Integer),
        // Restrict floats to a finite, magnitude-bounded range so the
        // JSON round-trip is lossless.
        (-1e6f64..1e6f64).prop_map(Value::Float),
        "[a-zA-Z0-9 _\\-]{0,32}".prop_map(Value::String),
    ]
}

fn property_entry_strategy() -> impl Strategy<Value = (String, Value)> {
    (
        "[a-z][a-z0-9_]{0,15}".prop_map(|s| s.to_string()),
        scalar_value_strategy(),
    )
}

fn property_bag_strategy() -> impl Strategy<Value = Vec<(String, Value)>> {
    vec(property_entry_strategy(), 0..6)
}

proptest! {
    /// Round-trip: every CREATE-time bag must be observable via
    /// scan_nodes' returned BoundNode.node.properties.
    #[test]
    fn create_then_scan_preserves_property_bag(bag in property_bag_strategy()) {
        let substrate = StubExecutorSubstrate::new();
        let tenant = TenantId::DEFAULT;
        // The stub's create_node de-duplicates by key (last-wins via
        // `properties.insert`); the round-trip oracle is the
        // last-wins reduction of the input slice.
        let expected: BTreeMap<String, Value> = bag.iter().cloned().collect();

        let _node_id = substrate.create_node(tenant, None, &bag, &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO)).expect("create OK");

        let scanned = substrate.scan_nodes(tenant, None, Lsn::MAX).expect("scan OK");
        prop_assert_eq!(scanned.len(), 1);
        prop_assert_eq!(scanned[0].node.properties.clone(), expected);
    }

    /// Round-trip multiple bags: each CREATE adds a row; scan returns
    /// every row's bag verbatim.
    #[test]
    fn multiple_creates_each_round_trip_their_bag(
        bags in vec(property_bag_strategy(), 0..4),
    ) {
        let substrate = StubExecutorSubstrate::new();
        let tenant = TenantId::DEFAULT;

        let mut expected_bags: Vec<BTreeMap<String, Value>> = Vec::new();
        for bag in &bags {
            substrate.create_node(tenant, None, bag, &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO)).expect("create OK");
            expected_bags.push(bag.iter().cloned().collect());
        }

        let scanned = substrate.scan_nodes(tenant, None, Lsn::MAX).expect("scan OK");
        prop_assert_eq!(scanned.len(), bags.len());

        // Order is ascending by NodeId; CREATE allocates monotonically,
        // so scanned[i].properties == expected_bags[i].
        for (i, n) in scanned.iter().enumerate() {
            prop_assert_eq!(n.node.properties.clone(), expected_bags[i].clone());
        }
    }

    /// Round-trip preserves the rel property bag via create_rel + expand.
    #[test]
    fn create_rel_round_trips_property_bag(rel_bag in property_bag_strategy()) {
        let substrate = StubExecutorSubstrate::new();
        let tenant = TenantId::DEFAULT;

        let src = substrate.create_node(tenant, None, &[], &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO)).expect("create src");
        let dst = substrate.create_node(tenant, None, &[], &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO)).expect("create dst");
        let _rel = substrate
            .create_rel(
                tenant,
                src,
                dst,
                "KNOWS",
                &rel_bag,
                &arcgraph_query::executor::ExecutionContext::new(
                    tenant,
                    arcgraph_core::PartitionId::ZERO,
                ),
            )
            .expect("create rel OK");

        let expected: BTreeMap<String, Value> = rel_bag.iter().cloned().collect();

        let edges = substrate
            .expand(
                tenant,
                src,
                None,
                arcgraph_query::logical_plan::Direction::LeftToRight,
                Lsn::MAX,
            )
            .expect("expand OK");
        prop_assert_eq!(edges.len(), 1);
        prop_assert_eq!(edges[0].rel.properties.clone(), expected);
    }

    /// Empty bag round-trips to empty (no PropertyData::Empty vs Blob
    /// mismatch).
    #[test]
    fn empty_bag_round_trips_as_empty(label_name in "[a-zA-Z][a-zA-Z0-9_]{0,15}") {
        let substrate = StubExecutorSubstrate::new();
        let tenant = TenantId::DEFAULT;

        let _ = substrate
            .create_node(
                tenant,
                Some(&label_name),
                &[],
                &arcgraph_query::executor::ExecutionContext::new(
                    tenant,
                    arcgraph_core::PartitionId::ZERO,
                ),
            )
            .expect("create OK");

        let scanned = substrate.scan_nodes(tenant, None, Lsn::MAX).expect("scan OK");
        prop_assert_eq!(scanned.len(), 1);
        prop_assert!(
            scanned[0].node.properties.is_empty(),
            "empty CREATE bag round-trips to empty NodeView.properties"
        );
    }
}

#[test]
fn manual_round_trip_single_entry() {
    // Sanity check that the proptest's oracle behavior is realized
    // outside the property-test loop (defense against a passing
    // proptest with broken expected-vs-actual semantics).
    let substrate = StubExecutorSubstrate::new();
    let tenant = TenantId::DEFAULT;
    let bag = vec![
        ("id".to_string(), Value::Integer(42)),
        ("name".to_string(), Value::String("Alice".into())),
    ];
    let _node_id: NodeId = substrate
        .create_node(
            tenant,
            None,
            &bag,
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create OK");

    let scanned = substrate
        .scan_nodes(tenant, None, Lsn::MAX)
        .expect("scan OK");
    assert_eq!(scanned.len(), 1);
    assert_eq!(
        scanned[0].node.properties.get("id"),
        Some(&Value::Integer(42))
    );
    assert_eq!(
        scanned[0].node.properties.get("name"),
        Some(&Value::String("Alice".into()))
    );
}
