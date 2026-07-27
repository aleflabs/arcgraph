//! M6.2 OOC-2 external-merge-sort release gates.
//!
//! These tests intentionally consume the public query/storage seam. The
//! correctness oracle is an independent stable `Vec::sort_by_key`, while the
//! resource assertions observe the external sort's production counters.

#![cfg(feature = "fault-injection")]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};
use arcgraph_query::error::Span;
use arcgraph_query::executor::ops::{
    ExternalSortProbe, PhysicalOperator, ScanOp, SortKey, SortOp, SortSpillTarget,
};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{
    BATCH_ROWS, ExecutionContext, ExecutionError, ExecutorSpillError, MemoryBudget,
    StubExecutorSubstrate, Value, estimate_row_bytes,
};
use arcgraph_query::logical_plan::SortDirection;
use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression, BoundPropertyRef};
use arcgraph_storage::spill::{
    SpillEncryptionPolicy, SpillManager, SpillManagerConfig, SpillQuery, SpillQueryConfig,
    SpillRejectReason,
};

const LABEL: LabelId = LabelId::new(1);
const BINDING: BindingId = BindingId::new(0);
const GENEROUS_SPILL_QUOTA_BYTES: u64 = 512 * 1024 * 1024;

static NEXT_SCRATCH_ID: AtomicU64 = AtomicU64::new(1);

/// An exact, process-unique directory so integration tests can run in
/// parallel without adding a test-only filesystem dependency.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(gate: &str) -> Self {
        let unique = NEXT_SCRATCH_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "arcgraph-m6-extsort-{gate}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create external-sort scratch root");
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct FixtureRecord {
    key: i64,
    input_id: u64,
    payload: String,
}

struct Fixture {
    input: Vec<FixtureRecord>,
    substrate: StubExecutorSubstrate,
    /// The exact estimator inputs used by external run generation:
    /// `estimate_row_bytes(keys) + estimate_row_bytes(row) + ordinal`.
    record_bytes: Vec<u64>,
}

fn fixture(row_count: usize) -> Fixture {
    let tenant = TenantId::DEFAULT;
    let mut substrate = StubExecutorSubstrate::new();
    let mut input = Vec::with_capacity(row_count);
    let mut record_bytes = Vec::with_capacity(row_count);

    for index in 0..row_count {
        let input_id = u64::try_from(index + 1).expect("fixture id fits u64");
        // Deliberately non-monotonic with many duplicates. ScanOp emits the
        // rows by ascending NodeId, so `input_id` is also the stable input
        // ordinal used by the independent oracle.
        let key =
            i64::try_from((input_id * 37 + input_id / 5) % 17).expect("fixture key fits i64") - 8;
        let payload = format!(
            "input-{input_id:05}-{}",
            "x".repeat(usize::try_from(input_id % 29).expect("payload length fits usize"))
        );
        let node = NodeView::new(NodeId::new(input_id), Some(LABEL))
            .with_property("sort_key", Value::Integer(key))
            .with_property("input_id", Value::Integer(input_id as i64))
            .with_property("payload", Value::String(payload.clone()));
        let row = vec![Value::Node(node.clone())];
        let keys = [Value::Integer(key)];
        let bytes = estimate_row_bytes(&keys)
            .saturating_add(estimate_row_bytes(&row))
            .saturating_add(std::mem::size_of::<u64>());
        record_bytes.push(u64::try_from(bytes).expect("record estimate fits u64"));
        substrate = substrate.with_node(tenant, node);
        input.push(FixtureRecord {
            key,
            input_id,
            payload,
        });
    }

    Fixture {
        input,
        substrate,
        record_bytes,
    }
}

fn independent_stable_oracle(input: &[FixtureRecord]) -> Vec<FixtureRecord> {
    let mut oracle = input.to_vec();
    // This is intentionally the standard library's stable in-memory sort,
    // not any comparator or helper from SortOp.
    oracle.sort_by_key(|record| record.key);
    oracle
}

fn assert_oracle_preserves_ties(oracle: &[FixtureRecord]) {
    let mut duplicate_pairs = 0_usize;
    for pair in oracle.windows(2) {
        if pair[0].key == pair[1].key {
            duplicate_pairs += 1;
            assert!(
                pair[0].input_id < pair[1].input_id,
                "stable oracle reordered equal-key input ids: {pair:?}"
            );
        }
    }
    assert!(
        duplicate_pairs > 0,
        "fixture must contain duplicate sort keys"
    );
}

fn sort_key_expression() -> BoundExpression {
    BoundExpression::PropertyAccess {
        base: Box::new(BoundExpression::VariableRef {
            name: "n".to_owned(),
            binding_id: BINDING,
            span: Span::point(1, 1),
            type_info: None,
        }),
        path: vec![BoundPropertyRef {
            name: "sort_key".to_owned(),
            property_id: None,
            span: Span::point(1, 1),
        }],
        span: Span::point(1, 1),
        type_info: None,
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
        .expect("begin encrypted external-sort spill query")
}

fn build_sort(
    query: SpillQuery,
    fan_in: usize,
    probe: ExternalSortProbe,
) -> Result<SortOp, ExecutionError> {
    let target = SortSpillTarget::new(query)
        .with_merge_fan_in(fan_in)?
        .with_probe(probe);
    SortOp::new(
        PhysicalOperator::Scan(ScanOp::new(BINDING, Some(LABEL), Lsn::MAX)),
        vec![SortKey {
            expr: sort_key_expression(),
            direction: SortDirection::Asc,
        }],
    )
    .with_spillover_target(Some(target))
}

fn extract_record(row: &[Value]) -> FixtureRecord {
    let Value::Node(node) = row.first().expect("sort row has node column") else {
        panic!("sort row did not contain a node: {row:?}");
    };
    let Some(Value::Integer(key)) = node.properties.get("sort_key") else {
        panic!("sorted node missing integer sort_key: {node:?}");
    };
    let Some(Value::Integer(input_id)) = node.properties.get("input_id") else {
        panic!("sorted node missing integer input_id: {node:?}");
    };
    let Some(Value::String(payload)) = node.properties.get("payload") else {
        panic!("sorted node missing string payload: {node:?}");
    };
    FixtureRecord {
        key: *key,
        input_id: u64::try_from(*input_id).expect("fixture input id is non-negative"),
        payload: payload.clone(),
    }
}

fn drain_sort(
    sort: &mut SortOp,
    ctx: &ExecutionContext,
    substrate: &StubExecutorSubstrate,
) -> Result<Vec<FixtureRecord>, ExecutionError> {
    let mut output = Vec::new();
    loop {
        let batch = sort.next_batch(ctx, substrate)?;
        if batch.is_empty() {
            break;
        }
        output.extend(batch.rows().iter().map(|row| extract_record(row)));
    }
    Ok(output)
}

fn tiny_budget_for(fixture: &Fixture, records_per_run: u64) -> u64 {
    fixture
        .record_bytes
        .iter()
        .copied()
        .max()
        .expect("non-empty fixture")
        .saturating_mul(records_per_run)
}

fn count_named_files(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .map(|entry| {
            let entry = entry.expect("read external-sort scratch entry");
            if entry
                .file_type()
                .expect("read external-sort scratch file type")
                .is_dir()
            {
                count_named_files(&entry.path())
            } else {
                1
            }
        })
        .sum()
}

fn named_files(path: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(path) else {
        return Vec::new();
    };
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.expect("read external-sort scratch entry");
        if entry
            .file_type()
            .expect("read external-sort scratch file type")
            .is_dir()
        {
            files.extend(named_files(&entry.path()));
        } else {
            files.push(entry.path());
        }
    }
    files
}

#[test]
fn m6_extsort_correctness_vs_inmemory_oracle() {
    let scratch = ScratchDir::new("correctness");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 0, false)
            .expect("create spill manager");
    let fixture = fixture(192);
    let budget_bytes = tiny_budget_for(&fixture, 5);
    let budget = MemoryBudget::with_per_tenant_cap(TenantId::DEFAULT, budget_bytes);
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO).with_budget(budget);
    let probe = ExternalSortProbe::new();
    let query = encrypted_query(
        &manager,
        0xE570_0001,
        budget_bytes,
        GENEROUS_SPILL_QUOTA_BYTES,
    );
    let mut sort = build_sort(query, 16, probe.clone()).expect("build external sort");

    let actual = drain_sort(&mut sort, &ctx, &fixture.substrate).expect("external sort succeeds");
    let oracle = independent_stable_oracle(&fixture.input);
    assert_oracle_preserves_ties(&oracle);
    assert_eq!(
        actual, oracle,
        "encrypted external sort differs from Vec oracle"
    );

    let stats = probe.snapshot();
    assert!(
        stats.initial_runs_created >= 3,
        "fixture must force at least three initial runs, got {stats:?}"
    );
}

#[test]
fn m6_extsort_bounded_peak_rss() {
    let scratch = ScratchDir::new("bounded-peak");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 0, false)
            .expect("create spill manager");
    let fixture = fixture(BATCH_ROWS * 8);
    let budget_bytes = tiny_budget_for(&fixture, 96);
    let total_record_estimate = fixture.record_bytes.iter().copied().sum::<u64>();
    let one_upstream_batch_slack = fixture
        .record_bytes
        .chunks(BATCH_ROWS)
        .map(|batch| batch.iter().copied().sum::<u64>())
        .max()
        .expect("non-empty fixture has an upstream batch");
    let asserted_bound = budget_bytes.saturating_add(one_upstream_batch_slack);
    assert!(
        total_record_estimate > asserted_bound.saturating_mul(6),
        "N must be much larger than the measured budget+batch bound: total={total_record_estimate}, bound={asserted_bound}"
    );

    let budget = MemoryBudget::with_per_tenant_cap(TenantId::DEFAULT, budget_bytes);
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO).with_budget(budget);
    let probe = ExternalSortProbe::new();
    let query = encrypted_query(
        &manager,
        0xE570_0002,
        budget_bytes,
        GENEROUS_SPILL_QUOTA_BYTES,
    );
    let mut sort = build_sort(query, 16, probe.clone()).expect("build external sort");
    let output = drain_sort(&mut sort, &ctx, &fixture.substrate).expect("external sort succeeds");
    assert_eq!(output.len(), fixture.input.len(), "sort lost input rows");

    let stats = probe.snapshot();
    assert!(
        stats.peak_buffer_bytes <= asserted_bound,
        "measured resident buffer exceeded budget + one actual upstream batch: stats={stats:?}, budget={budget_bytes}, slack={one_upstream_batch_slack}"
    );
    assert!(
        stats.initial_runs_created >= 3,
        "bounded-peak gate must actually spill repeatedly: {stats:?}"
    );
}

#[test]
fn m6_extsort_multipass_fanin_bound() {
    const FAN_IN: usize = 3;

    let scratch = ScratchDir::new("multipass");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 0, false)
            .expect("create spill manager");
    let fixture = fixture(240);
    let budget_bytes = tiny_budget_for(&fixture, 4);
    let budget = MemoryBudget::with_per_tenant_cap(TenantId::DEFAULT, budget_bytes);
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO).with_budget(budget);
    let probe = ExternalSortProbe::new();
    let query = encrypted_query(
        &manager,
        0xE570_0003,
        budget_bytes,
        GENEROUS_SPILL_QUOTA_BYTES,
    );
    let mut sort = build_sort(query, FAN_IN, probe.clone()).expect("build external sort");

    let actual = drain_sort(&mut sort, &ctx, &fixture.substrate).expect("external sort succeeds");
    let oracle = independent_stable_oracle(&fixture.input);
    assert_oracle_preserves_ties(&oracle);
    assert_eq!(actual, oracle, "multi-pass merge differs from Vec oracle");

    let stats = probe.snapshot();
    assert!(
        stats.initial_runs_created > FAN_IN as u64,
        "fixture must create more initial runs than fan-in: {stats:?}"
    );
    // Check the safety bound before the strategy counters so an unbounded-
    // reader revert fails on the fd/read-buffer invariant itself.
    assert!(
        stats.max_concurrent_readers <= FAN_IN,
        "opened more than F readers: {stats:?}"
    );
    assert!(
        stats.intermediate_runs_created > 0,
        "more than F runs must produce intermediate runs: {stats:?}"
    );
    assert!(
        stats.merge_passes >= 2,
        "expected a real multi-pass merge: {stats:?}"
    );
    assert!(
        stats.max_live_runs <= FAN_IN,
        "retained more than F live runs: {stats:?}"
    );
}

#[test]
fn m6_extsort_quota_abort_clean() {
    let scratch = ScratchDir::new("quota-abort");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 8, false)
            .expect("create retaining spill manager");
    let allocation_unit = manager
        .volume_space()
        .expect("measure scratch allocation unit")
        .allocation_unit_bytes;
    // One small run consumes two directory units plus one inode and one data
    // block. Five units admit that first retained run but reject the second
    // run's two-unit create reservation.
    let quota_bytes = allocation_unit.saturating_mul(5);
    let fixture = fixture(32);
    let budget_bytes = 1;
    let budget = MemoryBudget::with_per_tenant_cap(TenantId::DEFAULT, budget_bytes);
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO).with_budget(budget);
    let probe = ExternalSortProbe::new();
    let query = encrypted_query(&manager, 0xE570_0004, budget_bytes, quota_bytes);
    let key_probe = query
        .key_zeroize_probe_for_test()
        .expect("forced encryption owns an ephemeral spill key");
    assert!(!key_probe.is_zeroized(), "live query key starts non-zero");
    let mut sort = build_sort(query, 16, probe.clone()).expect("build external sort");

    let error = sort
        .next_batch(&ctx, &fixture.substrate)
        .expect_err("second spill run must breach the tenant quota");
    match error {
        ExecutionError::Spill(ExecutorSpillError::ResourceExhausted {
            reason: SpillRejectReason::TenantQuota,
            requested_bytes,
            spilled_bytes,
            limit_bytes,
            ..
        }) => {
            assert!(requested_bytes > 0, "quota reject reports measured delta");
            assert!(spilled_bytes > 0, "one run was charged before rejection");
            assert_eq!(limit_bytes, quota_bytes);
        }
        other => panic!("expected typed TenantQuota executor error, got {other:?}"),
    }

    let stats = probe.snapshot();
    assert_eq!(
        stats.initial_runs_created, 1,
        "quota must admit exactly one complete run before rejecting: {stats:?}"
    );
    assert!(
        count_named_files(manager.spill_root()) >= 1,
        "retention hook must expose the completed run before sweep"
    );
    assert_eq!(
        manager.spilled_bytes(TenantId::DEFAULT),
        0,
        "abort must immediately drop every run/query quota guard"
    );
    assert!(
        key_probe.is_zeroized(),
        "quota abort must end the spill query and zeroize its key immediately"
    );

    let sweep = manager
        .periodic_sweep()
        .expect("sweep aborted query scratch");
    assert!(
        sweep.removed_files >= 1,
        "sweep must remove retained orphan"
    );
    assert_eq!(
        count_named_files(manager.spill_root()),
        0,
        "quota abort left orphan scratch after periodic sweep"
    );
}

/// Regression for the final-drain cleanup boundary: a reader fault must not
/// return past the runtime-drop arm and leave the query key/readers alive.
#[test]
fn m6_extsort_final_read_error_aborts_clean() {
    let scratch = ScratchDir::new("read-abort");
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(scratch.path()), 8, false)
            .expect("create retaining spill manager");
    let fixture = fixture(BATCH_ROWS * 8);
    let budget_bytes = tiny_budget_for(&fixture, BATCH_ROWS as u64);
    let budget = MemoryBudget::with_per_tenant_cap(TenantId::DEFAULT, budget_bytes);
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO).with_budget(budget);
    let probe = ExternalSortProbe::new();
    let query = encrypted_query(
        &manager,
        0xE570_0005,
        budget_bytes,
        GENEROUS_SPILL_QUOTA_BYTES,
    );
    let key_probe = query
        .key_zeroize_probe_for_test()
        .expect("forced encryption owns an ephemeral spill key");
    let mut sort = build_sort(query, 128, probe).expect("build external sort");

    let first = sort
        .next_batch(&ctx, &fixture.substrate)
        .expect("materialize and start final merge");
    assert_eq!(
        first.row_count(),
        BATCH_ROWS,
        "fixture must leave the final merge live"
    );

    let files = named_files(manager.spill_root());
    assert!(!files.is_empty(), "retention hook must expose live runs");
    for path in &files {
        fs::write(path, []).expect("truncate live retained spill run");
    }

    let failure = loop {
        match sort.next_batch(&ctx, &fixture.substrate) {
            Ok(batch) if !batch.is_empty() => {}
            Ok(_) => panic!("truncated final runs were accepted as clean EOF"),
            Err(error) => break error,
        }
    };
    assert!(
        matches!(
            failure,
            ExecutionError::Spill(ExecutorSpillError::Failure { .. })
        ),
        "reader corruption must stay a typed spill failure: {failure:?}"
    );
    assert!(
        key_probe.is_zeroized(),
        "final-reader error must zeroize the query key before returning"
    );
    assert_eq!(
        manager.spilled_bytes(TenantId::DEFAULT),
        0,
        "final-reader error must release all run accounting"
    );
    assert_eq!(
        sort.next_batch(&ctx, &fixture.substrate)
            .expect_err("a failed final merge is terminal"),
        failure,
        "retry must return the stored typed failure without touching readers"
    );

    manager
        .periodic_sweep()
        .expect("sweep corrupted aborted-query scratch");
    assert_eq!(
        count_named_files(manager.spill_root()),
        0,
        "final-reader abort left retained scratch after sweep"
    );
}
