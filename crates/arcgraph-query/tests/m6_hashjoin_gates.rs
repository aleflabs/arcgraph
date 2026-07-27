//! M6.2 OOC-3 Grace hash-join release/fault-injection gates.
//!
//! The correctness and skew oracles are independent `HashMap` joins. Resource
//! assertions consume measurements taken from the production Grace runtime;
//! no test-side assumed row size is used as the occupancy observation.

#![cfg(feature = "fault-injection")]

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, RelId, TenantId, TypeId};
use arcgraph_query::error::Span;
use arcgraph_query::executor::eval::Parameters;
use arcgraph_query::executor::ops::{
    CorrelationSeedOp, ExpandOp, GraceHashJoinProbe, GraceHashJoinTarget, HashJoinOp,
    PhysicalOperator, ProjectOp, ScanOp, UnwindOp,
};
use arcgraph_query::executor::value::{NodeView, RelView};
use arcgraph_query::executor::{
    BATCH_ROWS, ExecutionContext, ExecutionError, ExecutorSpillError, MemoryBudget,
    StubExecutorSubstrate, Value, estimate_row_bytes,
};
use arcgraph_query::logical_plan::Direction;
use arcgraph_query::semantic::bound_ast::{
    BindingId, BoundExpression, BoundProjectionItem, BoundProjectionKind, BoundPropertyRef,
};
use arcgraph_storage::spill::{
    SpillEncryptionPolicy, SpillManager, SpillManagerConfig, SpillQuery, SpillQueryConfig,
    SpillRejectReason,
};

const KEY: BindingId = BindingId::new(10);
const LEFT_VALUE: BindingId = BindingId::new(11);
const RIGHT_VALUE: BindingId = BindingId::new(12);
const RECORD: BindingId = BindingId::new(9_999);
const NODE_LABEL: LabelId = LabelId::new(101);
const HUB_LABEL: LabelId = LabelId::new(102);
const EDGE_TYPE: TypeId = TypeId::new(103);
const GENEROUS_SPILL_QUOTA_BYTES: u64 = 512 * 1024 * 1024;

static NEXT_SCRATCH_ID: AtomicU64 = AtomicU64::new(1);

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(gate: &str) -> Self {
        let unique = NEXT_SCRATCH_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "arcgraph-m6-hashjoin-{gate}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create Grace hash-join scratch root");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn encrypted_query(
    manager: &SpillManager,
    query_id: u64,
    executor_budget_bytes: u64,
    quota_bytes: u64,
) -> SpillQuery {
    let mut config = SpillQueryConfig::new(TenantId::DEFAULT, query_id, 0, executor_budget_bytes);
    config.spill_quota_bytes = Some(quota_bytes);
    config.encryption = SpillEncryptionPolicy {
        tenant_encryption_enabled: false,
        force_encryption: true,
    };
    manager
        .begin_query(config)
        .expect("begin encrypted Grace hash-join spill query")
}

fn row_map(row: &[Value]) -> Value {
    let mut map = BTreeMap::new();
    for (index, value) in row.iter().enumerate() {
        map.insert(format!("c{index}"), value.clone());
    }
    Value::Map(map)
}

/// Build a public executor pipeline that lazily emits the supplied rows. This
/// is used only for semantic fixtures; the bounded-RSS gate uses ScanOp over a
/// much larger substrate so its input arrives in real BATCH_ROWS chunks.
fn values_operator(rows: &[Vec<Value>], schema: &[BindingId], parameter: &str) -> PhysicalOperator {
    let mut parameters = Parameters::new();
    parameters.insert(
        parameter.to_owned(),
        Value::List(rows.iter().map(|row| row_map(row)).collect()),
    );
    let seed = PhysicalOperator::CorrelationSeed(CorrelationSeedOp::new(Vec::new()));
    let unwind = UnwindOp::new(
        seed,
        BoundExpression::Parameter {
            name: parameter.to_owned(),
            span: Span::point(1, 1),
            type_info: None,
        },
        RECORD,
    )
    .with_parameters(parameters);
    let items = schema
        .iter()
        .copied()
        .enumerate()
        .map(|(index, output_id)| BoundProjectionItem {
            kind: BoundProjectionKind::Expr(BoundExpression::PropertyAccess {
                base: Box::new(BoundExpression::VariableRef {
                    name: "record".to_owned(),
                    binding_id: RECORD,
                    span: Span::point(1, 1),
                    type_info: None,
                }),
                path: vec![BoundPropertyRef {
                    name: format!("c{index}"),
                    property_id: None,
                    span: Span::point(1, 1),
                }],
                span: Span::point(1, 1),
                type_info: None,
            }),
            alias: None,
            output_id: Some(output_id),
            source_text: None,
            span: Span::point(1, 1),
        })
        .collect();
    PhysicalOperator::Project(ProjectOp::new(PhysicalOperator::Unwind(unwind), items))
}

fn estimated_rows_bytes(rows: &[Vec<Value>]) -> u64 {
    rows.iter().fold(0_u64, |total, row| {
        total.saturating_add(u64::try_from(estimate_row_bytes(row)).expect("row estimate fits u64"))
    })
}

fn build_values_join(
    left_rows: &[Vec<Value>],
    right_rows: &[Vec<Value>],
    query: SpillQuery,
    probe: GraceHashJoinProbe,
    max_depth: u8,
) -> HashJoinOp {
    let left = values_operator(left_rows, &[KEY, LEFT_VALUE], "left_rows");
    let right = values_operator(right_rows, &[KEY, RIGHT_VALUE], "right_rows");
    let target = GraceHashJoinTarget::new(query, estimated_rows_bytes(left_rows), 512)
        .with_max_repartition_depth(max_depth)
        .expect("valid test recursion depth")
        .with_probe(probe);
    HashJoinOp::new(left, right, vec![KEY])
        .expect("construct values hash join")
        .with_spillover_target(Some(target))
        .expect("attach Grace spill target")
}

fn drain_join(
    join: &mut HashJoinOp,
    ctx: &ExecutionContext,
    substrate: &StubExecutorSubstrate,
) -> Result<Vec<Vec<Value>>, ExecutionError> {
    let mut rows = Vec::new();
    loop {
        let batch = join.next_batch(ctx, substrate)?;
        if batch.is_empty() {
            break;
        }
        rows.extend(batch.into_rows());
    }
    Ok(rows)
}

/// Independent integer-key HashMap oracle. It deliberately does not call the
/// production fingerprint/partition/join code.
fn integer_oracle(left: &[Vec<Value>], right: &[Vec<Value>]) -> Vec<(i64, i64, i64)> {
    let mut table: HashMap<i64, Vec<i64>> = HashMap::new();
    for row in left {
        if let (Some(Value::Integer(key)), Some(Value::Integer(payload))) =
            (row.first(), row.get(1))
        {
            table.entry(*key).or_default().push(*payload);
        }
    }
    let mut output = Vec::new();
    for row in right {
        let (Some(Value::Integer(key)), Some(Value::Integer(right_payload))) =
            (row.first(), row.get(1))
        else {
            continue;
        };
        if let Some(left_payloads) = table.get(key) {
            output.extend(
                left_payloads
                    .iter()
                    .map(|left_payload| (*key, *left_payload, *right_payload)),
            );
        }
    }
    output.sort_unstable();
    output
}

fn integer_output(rows: &[Vec<Value>]) -> Vec<(i64, i64, i64)> {
    let mut output = rows
        .iter()
        .map(|row| match row.as_slice() {
            [
                Value::Integer(key),
                Value::Integer(left),
                Value::Integer(right),
            ] => (*key, *left, *right),
            other => panic!("unexpected integer join row: {other:?}"),
        })
        .collect::<Vec<_>>();
    output.sort_unstable();
    output
}

#[test]
fn m6_hashjoin_correctness_vs_inmemory_oracle() {
    let scratch = ScratchDir::new("correctness");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 0, false)
            .expect("create spill manager");
    let mut left = (0_i64..1_024)
        .map(|index| vec![Value::Integer(index % 512), Value::Integer(index)])
        .collect::<Vec<_>>();
    let mut right = (0_i64..768)
        .map(|index| {
            vec![
                Value::Integer((index * 17) % 512),
                Value::Integer(10_000 + index),
            ]
        })
        .collect::<Vec<_>>();
    left.push(vec![Value::Null, Value::Integer(-1)]);
    right.push(vec![Value::Null, Value::Integer(-2)]);

    let budget_bytes = 4 * 1024;
    let budget = MemoryBudget::with_per_tenant_cap(TenantId::DEFAULT, budget_bytes);
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO).with_budget(budget);
    let probe = GraceHashJoinProbe::new();
    let query = encrypted_query(
        &manager,
        0xA501_0001,
        budget_bytes,
        GENEROUS_SPILL_QUOTA_BYTES,
    );
    let mut join = build_values_join(&left, &right, query, probe.clone(), 3);
    let actual = drain_join(&mut join, &ctx, &StubExecutorSubstrate::new())
        .expect("partitioned values join succeeds");
    assert_eq!(
        integer_output(&actual),
        integer_oracle(&left, &right),
        "Grace join differs from independent HashMap oracle"
    );
    let stats = probe.snapshot();
    let chosen_partitions =
        u64::try_from(stats.chosen_partitions).expect("partition count fits u64");
    assert!(
        stats.chosen_partitions >= 2,
        "fixture must partition: {stats:?}"
    );
    assert_eq!(
        stats.initial_build_runs_created, chosen_partitions,
        "every configured initial build-root partition must be non-empty and create its root run: {stats:?}"
    );
    assert_eq!(
        stats.initial_probe_partitions_spilled, chosen_partitions,
        "every configured initial probe-root partition must route at least one row: {stats:?}"
    );
    assert!(
        stats.initial_probe_runs_created >= chosen_partitions,
        "each covered initial probe root must seal at least one leaf/block run: {stats:?}"
    );

    // Empty-side semantics are part of this named gate. Each case gets a new
    // epoch because a SpillQuery is single execution/attempt state.
    for (case, empty_left, empty_right) in [
        ("empty-left", true, false),
        ("empty-right", false, true),
        ("both-empty", true, true),
    ] {
        let case_left = if empty_left { &[][..] } else { &left[..] };
        let case_right = if empty_right { &[][..] } else { &right[..] };
        let query = encrypted_query(
            &manager,
            0xA501_0100 + u64::from(empty_left) * 2 + u64::from(empty_right),
            budget_bytes,
            GENEROUS_SPILL_QUOTA_BYTES,
        );
        let mut empty_join =
            build_values_join(case_left, case_right, query, GraceHashJoinProbe::new(), 3);
        let rows = drain_join(&mut empty_join, &ctx, &StubExecutorSubstrate::new())
            .unwrap_or_else(|error| panic!("{case} Grace join failed: {error}"));
        assert!(rows.is_empty(), "{case} must emit no rows");
    }
}

struct NodeFixture {
    substrate: StubExecutorSubstrate,
    estimated_total_bytes: u64,
}

fn node_fixture(count: usize) -> NodeFixture {
    let mut substrate = StubExecutorSubstrate::new();
    let mut estimated_total_bytes = 0_u64;
    for index in 0..count {
        let id = u64::try_from(index + 1).expect("fixture node id fits u64");
        let node = NodeView::new(NodeId::new(id), Some(NODE_LABEL)).with_property(
            "payload",
            Value::String(format!("node-{id:08}-{}", "x".repeat(index % 53))),
        );
        estimated_total_bytes = estimated_total_bytes.saturating_add(
            u64::try_from(estimate_row_bytes(&[Value::Node(node.clone())]))
                .expect("node row estimate fits u64"),
        );
        substrate = substrate.with_node(TenantId::DEFAULT, node);
    }
    NodeFixture {
        substrate,
        estimated_total_bytes,
    }
}

#[test]
fn m6_hashjoin_bounded_peak_rss() {
    let scratch = ScratchDir::new("bounded-peak");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 0, false)
            .expect("create spill manager");
    let fixture = node_fixture(BATCH_ROWS * 16);
    let budget_bytes = 256 * 1024;
    let budget = MemoryBudget::with_per_tenant_cap(TenantId::DEFAULT, budget_bytes);
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO).with_budget(budget);
    let probe = GraceHashJoinProbe::new();
    let query = encrypted_query(
        &manager,
        0xA501_0002,
        budget_bytes,
        GENEROUS_SPILL_QUOTA_BYTES,
    );
    let left = PhysicalOperator::Scan(ScanOp::new(KEY, Some(NODE_LABEL), Lsn::MAX));
    let right = PhysicalOperator::Scan(ScanOp::new(KEY, Some(NODE_LABEL), Lsn::MAX));
    let target = GraceHashJoinTarget::new(
        query,
        fixture.estimated_total_bytes,
        u64::try_from(BATCH_ROWS * 16).expect("cardinality fits u64"),
    )
    .with_probe(probe.clone());
    let mut join = HashJoinOp::new(left, right, vec![KEY])
        .expect("construct scan join")
        .with_spillover_target(Some(target))
        .expect("attach Grace target");
    let output =
        drain_join(&mut join, &ctx, &fixture.substrate).expect("large Grace join succeeds");
    assert_eq!(output.len(), BATCH_ROWS * 16, "unique-key join lost rows");

    let stats = probe.snapshot();
    assert!(
        stats.chosen_partitions >= 2,
        "large fixture must partition: {stats:?}"
    );
    assert!(
        stats.peak_build_table_bytes > 0,
        "gate observed no live build table"
    );
    assert!(
        stats.peak_probe_row_bytes > 0,
        "gate observed no live probe row"
    );
    assert!(
        stats.peak_probe_batch_bytes >= stats.peak_probe_row_bytes,
        "decoded probe-frame occupancy must be measured, not assumed from one row"
    );
    assert!(
        stats.peak_partition_batch_bytes > 0,
        "upstream BATCH_ROWS occupancy must be measured, not assumed"
    );
    // Partitioning and joining are disjoint phases. Account for the larger
    // measured phase-local input slack (an upstream partition batch or one
    // restored probe row), then compare it with the larger observed phase
    // peak. This prevents the partition-batch observation from being merely
    // decorative while retaining the executor-budget + one-live-input bound.
    let measured_input_slack = stats
        .peak_partition_batch_bytes
        .max(stats.peak_probe_batch_bytes);
    let measured_bound = budget_bytes.saturating_add(measured_input_slack);
    let measured_executor_peak = stats
        .peak_partition_batch_bytes
        .max(stats.peak_join_resident_bytes);
    assert!(
        measured_executor_peak <= measured_bound,
        "measured partition/join phase peak exceeded budget + largest measured live-input slack: stats={stats:?}, budget={budget_bytes}, slack={measured_input_slack}"
    );
    assert!(
        fixture.estimated_total_bytes > measured_bound.saturating_mul(6),
        "N must be much larger than the measured resident bound: total={}, bound={measured_bound}",
        fixture.estimated_total_bytes
    );
}

struct HotFixture {
    substrate: StubExecutorSubstrate,
    edge_ids: Vec<u64>,
    estimated_build_bytes: u64,
}

fn hot_fixture(edges: usize) -> HotFixture {
    let hub = NodeView::new(NodeId::new(1), Some(HUB_LABEL));
    let mut substrate = StubExecutorSubstrate::new().with_node(TenantId::DEFAULT, hub.clone());
    let mut edge_ids = Vec::with_capacity(edges);
    let mut estimated_build_bytes = 0_u64;
    for index in 0..edges {
        let target_id = u64::try_from(index + 2).expect("target id fits u64");
        let edge_id = u64::try_from(index + 1_000).expect("edge id fits u64");
        let target = NodeView::new(NodeId::new(target_id), Some(NODE_LABEL));
        let rel = RelView::new(RelId::new(edge_id), hub.id, target.id, Some(EDGE_TYPE));
        estimated_build_bytes = estimated_build_bytes.saturating_add(
            u64::try_from(estimate_row_bytes(&[
                Value::Node(hub.clone()),
                Value::Relationship(rel.clone()),
                Value::Node(target.clone()),
            ]))
            .expect("hot row estimate fits u64"),
        );
        substrate = substrate
            .with_node(TenantId::DEFAULT, target)
            .with_edge(TenantId::DEFAULT, rel);
        edge_ids.push(edge_id);
    }
    HotFixture {
        substrate,
        edge_ids,
        estimated_build_bytes,
    }
}

fn hot_expand(rel_binding: BindingId, target_binding: BindingId) -> PhysicalOperator {
    let scan = PhysicalOperator::Scan(ScanOp::new(KEY, Some(HUB_LABEL), Lsn::MAX));
    PhysicalOperator::Expand(
        ExpandOp::new(
            scan,
            KEY,
            Some(rel_binding),
            target_binding,
            Some(EDGE_TYPE),
            Direction::LeftToRight,
            None,
            Lsn::MAX,
        )
        .expect("construct hot expand"),
    )
}

#[test]
fn m6_hashjoin_skew_recursion_terminates() {
    const HOT_ROWS: usize = 96;
    const COLLISION_ROWS_PER_KEY: i64 = 32;
    const MAX_DEPTH: u8 = 3;

    let scratch = ScratchDir::new("skew");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 0, false)
            .expect("create spill manager");
    let fixture = hot_fixture(HOT_ROWS);
    let budget_bytes = 4 * 1024;
    let budget = MemoryBudget::with_per_tenant_cap(TenantId::DEFAULT, budget_bytes);
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO).with_budget(budget);
    let probe = GraceHashJoinProbe::new();
    let query = encrypted_query(
        &manager,
        0xA501_0003,
        budget_bytes,
        GENEROUS_SPILL_QUOTA_BYTES,
    );
    let left_rel = BindingId::new(20);
    let left_target = BindingId::new(21);
    let right_rel = BindingId::new(22);
    let right_target = BindingId::new(23);
    let target = GraceHashJoinTarget::new(query, fixture.estimated_build_bytes, 1)
        .with_max_repartition_depth(MAX_DEPTH)
        .expect("valid skew depth")
        .with_probe(probe.clone());
    let mut join = HashJoinOp::new(
        hot_expand(left_rel, left_target),
        hot_expand(right_rel, right_target),
        vec![KEY],
    )
    .expect("construct hot join")
    .with_spillover_target(Some(target))
    .expect("attach Grace target");
    let rows = drain_join(&mut join, &ctx, &fixture.substrate).expect("hot-key join terminates");

    // Independent HashMap oracle over the conceptual left/probe inputs.
    let mut oracle_table: HashMap<u64, Vec<u64>> = HashMap::new();
    oracle_table.insert(1, fixture.edge_ids.clone());
    let mut oracle = Vec::with_capacity(HOT_ROWS * HOT_ROWS);
    for right_edge in &fixture.edge_ids {
        if let Some(left_edges) = oracle_table.get(&1) {
            oracle.extend(left_edges.iter().map(|left_edge| (*left_edge, *right_edge)));
        }
    }
    oracle.sort_unstable();
    let mut actual = rows
        .iter()
        .map(|row| match (row.get(1), row.get(3)) {
            (Some(Value::Relationship(left)), Some(Value::Relationship(right))) => {
                (left.id.raw(), right.id.raw())
            }
            other => panic!("unexpected hot join row: {other:?}"),
        })
        .collect::<Vec<_>>();
    actual.sort_unstable();
    assert_eq!(
        actual, oracle,
        "hot-key Grace output differs from HashMap oracle"
    );

    let stats = probe.snapshot();
    assert_eq!(
        stats.max_recursion_depth, MAX_DEPTH,
        "unsplittable hot key must reach, but never exceed, the depth cap: {stats:?}"
    );
    assert!(
        stats.recursive_runs_created > 0,
        "fixture did not recurse: {stats:?}"
    );
    assert!(
        stats.block_fallback_partitions >= 1,
        "hot key must terminate through bounded block fallback: {stats:?}"
    );

    // Reseed control: I:0 and I:10 collide in the initial P=2 bucket,
    // but route to different buckets under the depth-1 seed. The combined
    // root is oversized while each 32-row child fits. Reusing the initial
    // seed therefore reaches the depth cap/fallback and makes these exact
    // depth/run assertions RED, while the production reseed stops at depth 1.
    let mut collision_left = Vec::new();
    for key in [0_i64, 10] {
        collision_left.extend(
            (0..COLLISION_ROWS_PER_KEY)
                .map(|ordinal| vec![Value::Integer(key), Value::Integer(key * 1_000 + ordinal)]),
        );
    }
    let collision_right = vec![
        vec![Value::Integer(0), Value::Integer(20_000)],
        vec![Value::Integer(10), Value::Integer(20_010)],
    ];
    let collision_budget_bytes = 40 * 1024;
    let collision_budget =
        MemoryBudget::with_per_tenant_cap(TenantId::DEFAULT, collision_budget_bytes);
    let collision_ctx =
        ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO).with_budget(collision_budget);
    let collision_probe = GraceHashJoinProbe::new();
    let collision_query = encrypted_query(
        &manager,
        0xA501_0005,
        collision_budget_bytes,
        GENEROUS_SPILL_QUOTA_BYTES,
    );
    // The Grace build allowance is half the configured cap (20 KiB).
    // A 30 KiB estimate therefore selects ceil(30/20)=2 initial roots. The
    // combined collision is larger than 20 KiB under the production table
    // planner, while each reseeded 32-row child is smaller than 20 KiB.
    let collision_target = GraceHashJoinTarget::new(collision_query, 30 * 1024, 2)
        .with_max_repartition_depth(MAX_DEPTH)
        .expect("valid collision depth")
        .with_probe(collision_probe.clone());
    let mut collision_join = HashJoinOp::new(
        values_operator(&collision_left, &[KEY, LEFT_VALUE], "collision_left"),
        values_operator(&collision_right, &[KEY, RIGHT_VALUE], "collision_right"),
        vec![KEY],
    )
    .expect("construct reseed-control join")
    .with_spillover_target(Some(collision_target))
    .expect("attach reseed-control target");
    let collision_actual = drain_join(
        &mut collision_join,
        &collision_ctx,
        &StubExecutorSubstrate::new(),
    )
    .expect("splittable collision join terminates");
    assert_eq!(
        integer_output(&collision_actual),
        integer_oracle(&collision_left, &collision_right),
        "splittable collision output differs from independent HashMap oracle"
    );
    let collision_stats = collision_probe.snapshot();
    assert_eq!(
        collision_stats.chosen_partitions, 2,
        "collision fixture must select P=2: {collision_stats:?}"
    );
    assert_eq!(
        collision_stats.max_recursion_depth, 1,
        "depth-1 reseeding must split the oversized initial collision: {collision_stats:?}"
    );
    assert_eq!(
        collision_stats.recursive_runs_created, 2,
        "depth-1 reseeding must create exactly two fitting child runs: {collision_stats:?}"
    );
    assert_eq!(
        collision_stats.block_fallback_partitions, 0,
        "splittable collision must not reach block fallback: {collision_stats:?}"
    );
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ScratchEntryCounts {
    files: u64,
    directories: u64,
}

fn scratch_entry_counts(root: &Path) -> ScratchEntryCounts {
    let Ok(entries) = fs::read_dir(root) else {
        return ScratchEntryCounts::default();
    };
    let mut counts = ScratchEntryCounts::default();
    for entry in entries {
        let entry = entry.expect("read Grace scratch entry");
        if entry
            .file_type()
            .expect("read Grace scratch file type")
            .is_dir()
        {
            counts.directories = counts.directories.saturating_add(1);
            let descendants = scratch_entry_counts(&entry.path());
            counts.files = counts.files.saturating_add(descendants.files);
            counts.directories = counts.directories.saturating_add(descendants.directories);
        } else {
            counts.files = counts.files.saturating_add(1);
        }
    }
    counts
}

#[test]
fn m6_hashjoin_quota_abort_clean() {
    let scratch = ScratchDir::new("quota-abort");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 8, false)
            .expect("create retaining spill manager");
    let allocation_unit = manager
        .volume_space()
        .expect("measure scratch allocation unit")
        .allocation_unit_bytes;
    let quota_bytes = allocation_unit.saturating_mul(5);
    let budget_bytes = 1;
    let budget = MemoryBudget::with_per_tenant_cap(TenantId::DEFAULT, budget_bytes);
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO).with_budget(budget);
    let fixture = node_fixture(64);
    let query = encrypted_query(&manager, 0xA501_0004, budget_bytes, quota_bytes);
    let key_probe = query
        .key_zeroize_probe_for_test()
        .expect("forced encryption owns an ephemeral spill key");
    assert!(!key_probe.is_zeroized(), "live query key starts non-zero");
    let target = GraceHashJoinTarget::new(query, fixture.estimated_total_bytes, 64)
        .with_probe(GraceHashJoinProbe::new());
    let mut join = HashJoinOp::new(
        PhysicalOperator::Scan(ScanOp::new(KEY, Some(NODE_LABEL), Lsn::MAX)),
        PhysicalOperator::Scan(ScanOp::new(KEY, Some(NODE_LABEL), Lsn::MAX)),
        vec![KEY],
    )
    .expect("construct quota join")
    .with_spillover_target(Some(target))
    .expect("attach quota target");

    let error = join
        .next_batch(&ctx, &fixture.substrate)
        .expect_err("Grace partitions must breach tiny tenant quota");
    match error {
        ExecutionError::Spill(ExecutorSpillError::ResourceExhausted {
            reason: SpillRejectReason::TenantQuota,
            requested_bytes,
            spilled_bytes,
            limit_bytes,
            ..
        }) => {
            assert!(requested_bytes > 0, "quota reject reports measured delta");
            assert!(spilled_bytes > 0, "a run was charged before rejection");
            assert_eq!(limit_bytes, quota_bytes);
        }
        other => panic!("expected typed TenantQuota executor error, got {other:?}"),
    }
    assert_eq!(
        manager.spilled_bytes(TenantId::DEFAULT),
        0,
        "abort must drop every run/query quota guard"
    );
    assert!(
        key_probe.is_zeroized(),
        "quota abort must zeroize query key"
    );
    let retained = scratch_entry_counts(manager.spill_root());
    assert!(retained.files >= 1, "retention hook exposed no run file");
    assert!(
        retained.directories >= 1,
        "retention hook exposed no tenant/epoch scratch directory"
    );
    let sweep = manager
        .periodic_sweep()
        .expect("sweep aborted join scratch");
    assert!(sweep.removed_files >= 1, "sweep removed no retained run");
    assert!(
        sweep.removed_directories >= 1,
        "sweep removed no orphan tenant/epoch directory"
    );
    assert_eq!(
        scratch_entry_counts(manager.spill_root()),
        ScratchEntryCounts::default(),
        "quota abort left file or directory scratch entries after periodic sweep"
    );
}
