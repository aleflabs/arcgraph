//! W27-ν / ADR-163 — Jepsen-style ArcQL WRITE-SIDE SI tests
//! (ACTIVE; CI-gating). Activated at W27-ν-2 per ADR-163 §"Forward-
//! deferred" FD-1.
//!
//! # What "write-side" verifies
//!
//! The write-side workloads verify **property persistence round-trips**
//! under concurrency: `CREATE (:L {p: v})` / `SET n.p = v` followed by a
//! `MATCH`-by-property returning the node with `p == v`. The three SI
//! write-op properties under test (per the read-side's Adya 2000 §4 /
//! Bailis 2014 §3 taxonomy, lifted to the property level):
//!
//! - **WAW (write-after-write)** — `CREATE (:Account {balance:100})`
//!   then `SET n.balance = 50`: the surviving property reflects the SET
//!   (version order), and a reader pinned at a snapshot BETWEEN the two
//!   writes still observes `100` (snapshot stability on property writes).
//! - **RYW (read-your-writes)** — `CREATE (:Account {balance:100})` then
//!   `MATCH (n {balance:100})` returns exactly that node; a non-matching
//!   predicate (`balance:999`) returns zero rows. The property is
//!   durable and queryable by value.
//! - **Lost update** — N clients each read-modify-write a shared node's
//!   `counter` property in one transaction. Under SI/OCC the WW conflict
//!   aborts the loser, so the final value reflects EVERY committed
//!   increment, never a torn/lost mix (the property-level analog of the
//!   read-side G0 dirty-write test; the canonical lost-update workload,
//!   Bailis 2014 §3.1). Verified over the recorded history via the
//!   public [`common::checker::ArcqlSiChecker::check`] oracle (the
//!   `ArcqlViolation::LostUpdate` predicate, W27-ν-2 extension).
//!
//! # Activation provenance (ADR-163 §FD-1)
//!
//! These tests were `#[ignore]`'d + `panic!`-guarded at the W27-ν
//! skeleton landing because the property round-trip was not observable:
//! the W26-θ write executors stored `PropertyData::Empty` and
//! [`arcgraph_query::executor::value::NodeView`] carried `(id, label)`
//! only. **ADR-152** (W27-α property-bag persistence, PR **#492**, merge
//! SHA **`4db0946`**) lifted that: `CrudExecutorSubstrate::create_node`
//! now serializes the bag via `properties_to_property_data`
//! (→ `PropertyData::Blob`), and `scan_nodes` decodes it back into
//! `NodeView.properties` via `record_property_bag`
//! (`crates/arcgraph-mcp/src/storage/substrate.rs` §"ADR-152 §D-3").
//! So the property round-trip is now expressible and these assertions
//! are real (no `panic!` guard). See `docs/adr/amendments/
//! ADR-163-amendment-01-writeside-activation.md`.
//!
//! # Determinism contract (REQUIRED reading before editing)
//!
//! Same contract as the read-side: the CHECKER predicate is the oracle,
//! NOT a binary-equal reference snapshot. The lost-update workload's
//! interleaving is intentionally racy so OCC conflicts surface; the
//! lost-update INVARIANT (final == seed + committed increments) holds
//! over any interleaving. See `jepsen_arcql_common/mod.rs` §"Determinism
//! contract". The deterministic WAW snapshot-stability pin uses a
//! constructed interleaving (pinned reader between two writes), so the
//! suite is non-vacuous even when the scheduler serializes the workload.
//!
//! Run:
//!   cargo test -p arcgraph-mcp --test jepsen_arcql_si_write_skeleton

#[path = "jepsen_arcql_common/mod.rs"]
mod common;

use std::sync::Arc;

use arcgraph_core::NodeId;
use arcgraph_query::executor::ExecutorSubstrate;
use arcgraph_query::executor::substrate::SetNodeMutation;
use arcgraph_query::executor::value::Value;
use arcgraph_storage::crud;
use arcgraph_storage::test_harness::jepsen::history::OperationHistory;

use common::checker::ArcqlSiChecker;
use common::{
    COUNTER_PROP, JepsenArcqlFixture, committed_increment_count, read_counter,
    run_counter_workload, seed_counter_node,
};

/// The `Lsn` threaded into `scan_nodes` is ignored by the v1.0-α
/// substrate (the read-snapshot is the per-call `begin`); pass the
/// latest so the call is well-formed.
fn latest_read_lsn(fixture: &JepsenArcqlFixture) -> arcgraph_core::Lsn {
    fixture.mgr.begin(fixture.tenant).snapshot()
}

/// Find the single node whose `counter`/named integer property equals
/// `want` in a substrate scan (MATCH-by-property at the trait surface).
fn scan_count_with_property(fixture: &JepsenArcqlFixture, prop: &str, want: i64) -> usize {
    let read_lsn = latest_read_lsn(fixture);
    let nodes = fixture
        .substrate
        .scan_nodes(fixture.tenant, None, read_lsn)
        .expect("scan_nodes OK");
    nodes
        .iter()
        .filter(|bn| matches!(bn.node.properties.get(prop), Some(Value::Integer(v)) if *v == want))
        .count()
}

// ─────────────────────────────────────────────────────────────────────
// WAW — write-after-write (CREATE then SET) + snapshot stability
// ─────────────────────────────────────────────────────────────────────

/// **Write-after-write (CREATE then SET).** `CREATE (:Account
/// {balance:100})` then `SET n.balance = 50`: the surviving property
/// reflects the SET (write/version ordering), and a reader pinned at a
/// snapshot BETWEEN the two writes still sees `100` (snapshot stability
/// on property writes). Drives the production `CrudExecutorSubstrate`
/// for the round-trip + ordering; uses a crud-tier pinned reader for the
/// snapshot-stability sub-assertion (the `scan_nodes` trait method opens
/// its own per-call snapshot and so cannot pin one).
#[test]
fn arcql_si_waw_write_after_write() {
    let fixture = JepsenArcqlFixture::new();
    let tenant = fixture.tenant;

    // CREATE (:Account {balance:100}) via the production path.
    let node = fixture
        .substrate
        .create_node(
            tenant,
            Some("Account"),
            &[("balance".to_string(), Value::Integer(100))],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create_node OK");

    // Round-trip: a fresh scan observes balance == 100 (and not 50).
    assert_eq!(
        scan_count_with_property(&fixture, "balance", 100),
        1,
        "post-CREATE scan must observe the {{balance:100}} round-trip"
    );

    // Pin a reader snapshot S AFTER the create commits but BEFORE the
    // SET (constructed interleaving for the snapshot-stability proof).
    let pinned_reader = fixture.mgr.begin(tenant);
    let balance_at_s = {
        let rec = crud::read_node(&pinned_reader, node)
            .expect("read_node OK")
            .expect("node visible at S");
        let bag = arcgraph_mcp::storage::property_payload::record_property_bag_checked(
            &rec,
            fixture.crud.blob_store(),
            &fixture.intern,
            tenant,
        )
        .expect("jepsen skeleton: property payload decode must succeed");
        match bag.get("balance") {
            Some(Value::Integer(v)) => *v,
            other => panic!("expected balance:Integer at S, got {other:?}"),
        }
    };
    assert_eq!(balance_at_s, 100, "pinned reader sees the pre-SET value");

    // SET n.balance = 50 (production path; commits a new version).
    fixture
        .substrate
        .set_node(
            tenant,
            node,
            &SetNodeMutation::PropertyAssign {
                name: "balance".to_string(),
                value: Value::Integer(50),
            },
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("set_node OK");

    // WAW ordering: the surviving value reflects the SET.
    assert_eq!(
        scan_count_with_property(&fixture, "balance", 50),
        1,
        "post-SET scan must observe the WAW-ordered value 50"
    );
    assert_eq!(
        scan_count_with_property(&fixture, "balance", 100),
        0,
        "the pre-SET value 100 is superseded — no node still reads 100"
    );

    // Snapshot stability: the pinned reader STILL sees 100 (its snapshot
    // predates the SET commit). This is the property-write analog of the
    // read-side `deterministic_snapshot_read_stability`.
    let balance_still_at_s = {
        let rec = crud::read_node(&pinned_reader, node)
            .expect("read_node OK")
            .expect("node visible at S");
        let bag = arcgraph_mcp::storage::property_payload::record_property_bag_checked(
            &rec,
            fixture.crud.blob_store(),
            &fixture.intern,
            tenant,
        )
        .expect("jepsen skeleton: property payload decode must succeed");
        match bag.get("balance") {
            Some(Value::Integer(v)) => *v,
            other => panic!("expected balance:Integer at S, got {other:?}"),
        }
    };
    assert_eq!(
        balance_still_at_s, 100,
        "reader pinned at S must NOT observe a SET committed after S (snapshot stability)"
    );
    drop(pinned_reader);

    // A fresh reader does see the SET value.
    assert_eq!(
        scan_count_with_property(&fixture, "balance", 50),
        1,
        "a fresh reader observes the committed SET"
    );
}

// ─────────────────────────────────────────────────────────────────────
// RYW — read-your-(property)-writes (CREATE then MATCH-by-property)
// ─────────────────────────────────────────────────────────────────────

/// **Read-your-writes (CREATE then MATCH-by-property).** `CREATE
/// (:Account {balance:100})` then `MATCH (n {balance:100})` returns
/// exactly that node; `MATCH (n {balance:999})` returns zero rows. The
/// property is durable and queryable by value through the production
/// `scan_nodes` → `NodeView.properties` path.
#[test]
fn arcql_si_ryw_read_your_property_writes() {
    let fixture = JepsenArcqlFixture::new();
    let tenant = fixture.tenant;

    let node = fixture
        .substrate
        .create_node(
            tenant,
            Some("Account"),
            &[("balance".to_string(), Value::Integer(100))],
            &arcgraph_query::executor::ExecutionContext::new(
                tenant,
                arcgraph_core::PartitionId::ZERO,
            ),
        )
        .expect("create_node OK");

    // MATCH (n {balance:100}) → exactly one row, and it is `node`.
    let read_lsn = latest_read_lsn(&fixture);
    let matched: Vec<NodeId> = fixture
        .substrate
        .scan_nodes(tenant, None, read_lsn)
        .expect("scan OK")
        .into_iter()
        .filter(|bn| matches!(bn.node.properties.get("balance"), Some(Value::Integer(100))))
        .map(|bn| bn.node.id)
        .collect();
    assert_eq!(
        matched,
        vec![node],
        "RYW: MATCH {{balance:100}} returns exactly the created node"
    );

    // MATCH (n {balance:999}) → zero rows (a value that was never set).
    assert_eq!(
        scan_count_with_property(&fixture, "balance", 999),
        0,
        "RYW: a never-written property value matches nothing"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Lost update — concurrent RMW on a shared counter property
// ─────────────────────────────────────────────────────────────────────

/// **Lost update (concurrent SET on the same node/property).** N clients
/// each read-modify-write a shared node's `counter` in ONE transaction
/// (begin → read → write+1 → commit). Under SI/OCC the WW conflict on
/// the shared property key aborts the loser, who retries — so EVERY
/// intended increment is eventually applied and the final value equals
/// `seed + committed_increments`, never a lost/torn mix. Verified BOTH
/// over the recorded history via the public `ArcqlSiChecker::check`
/// oracle (the `LostUpdate` predicate) AND by a live cross-check of the
/// scanned final value against the committed-increment count. Canonical
/// lost-update workload (Bailis 2014 §3.1).
#[test]
fn arcql_si_lost_update() {
    let fixture = JepsenArcqlFixture::new();
    let tenant = fixture.tenant;
    let seed = 0i64;
    let clients = 4u32;
    let increments_per_client = 25u64;
    let expected_total = u64::from(clients) * increments_per_client;

    let history = Arc::new(OperationHistory::new());
    let node = seed_counter_node(&fixture.mgr, &fixture.crud, tenant, seed, &history);

    run_counter_workload(
        &fixture,
        node,
        clients,
        increments_per_client,
        Arc::clone(&history),
    );

    let ops = history.drain_sorted();

    // (1) The history oracle: no lost update detected (the `LostUpdate`
    //     predicate runs because the counter writes are tagged).
    let verdict = ArcqlSiChecker::new().check(&ops);
    assert!(
        verdict.is_ok(),
        "lost-update workload surfaced an SI violation: {verdict}"
    );
    let summary = verdict.summary().expect("summary on OK path");
    assert_eq!(
        summary.counters_checked, 1,
        "exactly one counter key must have been lost-update-checked (else the predicate was vacuous)"
    );

    // (2) The committed-increment count equals the intended total (every
    //     increment eventually committed via retry — no permanent loss).
    let committed = committed_increment_count(&ops);
    assert_eq!(
        committed, expected_total,
        "all {expected_total} increments must eventually commit (OCC losers retry)"
    );

    // (3) Live cross-check: the scanned final counter == seed + total
    //     (catches "history says one thing, store says another").
    let read_tx = fixture.mgr.begin(tenant);
    let final_value = read_counter(&fixture.crud, &fixture.intern, &read_tx, tenant, node)
        .expect("counter node visible");
    drop(read_tx);
    assert_eq!(
        final_value,
        seed + expected_total as i64,
        "final counter must reflect every committed increment (no lost update)"
    );

    // (4) Cross-check the live value against the same value observed
    //     through the production substrate scan path.
    assert_eq!(
        scan_count_with_property(&fixture, COUNTER_PROP, seed + expected_total as i64),
        1,
        "the production scan path observes the same lost-update-free final value"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Write-side adversarial-oracle self-tests (NON-VACUITY GATE, S7c)
//
// Per `feedback_review_oracle_relaxations.md` + the W27-ν R2 precedent
// (read-side `jepsen_arcql_si_read.rs:692` `mod adversarial_history_
// tests`): the positive assertions above (`verdict.is_ok()` against the
// real kernel) are VACUOUS if the checker were a no-op that always
// returns `Ok`. These self-tests drive the PUBLIC `ArcqlSiChecker::
// check` with hand-built ANOMALOUS histories and assert it REPORTS the
// EXACT violation variant — proving the positive results are load-
// bearing. Mirrors the read-side module's exact-variant `matches!`
// discipline (Tier-B R1).
//
// Non-vacuity of THESE tests was itself proven by mutation: temporarily
// stubbing the relevant checker predicate to a no-op makes tests 1-3
// FAIL; reverting restores green (see the PR's §"Mutation-test").
// ─────────────────────────────────────────────────────────────────────

mod adversarial_history_tests {
    use arcgraph_core::{Lsn, TenantId};
    use arcgraph_storage::test_harness::jepsen::history::OpBuilder;
    use bytes::Bytes;

    use crate::common::SCAN_SENTINEL_KEY;
    use crate::common::checker::{ArcqlSiChecker, ArcqlViolation, encode_counter};

    fn lsn(n: u64) -> Lsn {
        Lsn::new(n)
    }

    fn present() -> Option<Bytes> {
        Some(Bytes::from_static(b"present"))
    }

    /// 1. **Lost update.** Two committed RMW increments on counter key
    ///    `100` whose recorded values are `{seed:0, then 1, then 1}` —
    ///    i.e. BOTH increments committed but the final value reflects
    ///    only ONE (the planted lost update, Bailis 2014 §3.1). The
    ///    checker MUST report `ArcqlViolation::LostUpdate` with both the
    ///    expected (2) and the (short) final (1) value.
    #[test]
    fn check_detects_lost_update() {
        const COUNTER: u64 = 100;

        // Seed CREATE: counter := 0 (committed).
        let mut seed = OpBuilder::new(u32::MAX, 0, TenantId::DEFAULT, lsn(1));
        seed.intend_write(COUNTER, Some(encode_counter(0)));
        let seed_op = seed.into_committed(lsn(5));

        // Increment A: read 0, write 1, commit.
        let mut a = OpBuilder::new(0, 1, TenantId::DEFAULT, lsn(5));
        a.observe_read(COUNTER, Some(encode_counter(0)));
        a.intend_write(COUNTER, Some(encode_counter(1)));
        let op_a = a.into_committed(lsn(10));

        // Increment B: ALSO read 0, ALSO write 1, ALSO commit — the
        // lost update (a correct OCC kernel would have aborted B).
        let mut b = OpBuilder::new(1, 2, TenantId::DEFAULT, lsn(5));
        b.observe_read(COUNTER, Some(encode_counter(0)));
        b.intend_write(COUNTER, Some(encode_counter(1)));
        let op_b = b.into_committed(lsn(20));

        let verdict = ArcqlSiChecker::new().check(&[seed_op, op_a, op_b]);
        assert!(
            !verdict.is_ok(),
            "checker must FAIL on a lost-update history"
        );
        let violations = verdict.violations().expect("violations on the bad path");
        assert!(
            violations.iter().any(|v| matches!(
                v,
                ArcqlViolation::LostUpdate {
                    key: COUNTER,
                    committed_increments: 2,
                    final_value: 1,
                    expected_value: 2,
                }
            )),
            "expected LostUpdate {{key 100, committed 2, final 1, expected 2}}; got {violations:?}"
        );
    }

    /// 2. **Stale property read.** A committed MATCH at snapshot
    ///    `start_lsn = 8` observes counter node `100` — but `100` was
    ///    CREATED (committed) at LSN `10`, i.e. AFTER the reader's
    ///    snapshot. Under SI a reader must not see a write that commits
    ///    after its snapshot (the property-write snapshot-stability
    ///    violation). The checker MUST report `ArcqlViolation::
    ///    SnapshotRead` naming `100` in its `extra` set.
    #[test]
    fn check_detects_stale_property_read() {
        const NODE: u64 = 100;

        // The property write (CREATE counter node 100) commits at 10.
        let mut writer = OpBuilder::new(0, 0, TenantId::DEFAULT, lsn(2));
        writer.intend_write(NODE, present());
        let create_op = writer.into_committed(lsn(10));

        // A MATCH at snapshot 8 (BEFORE 10) observes node 100 — stale.
        let mut reader = OpBuilder::new(1, 1, TenantId::DEFAULT, lsn(8));
        reader.observe_read(SCAN_SENTINEL_KEY, present());
        reader.observe_read(NODE, present());
        let match_op = reader.into_committed(lsn(9));

        let verdict = ArcqlSiChecker::new().check(&[create_op, match_op]);
        assert!(
            !verdict.is_ok(),
            "checker must FAIL on a stale property-read history"
        );
        let violations = verdict.violations().expect("violations on the bad path");
        assert!(
            violations.iter().any(|v| matches!(
                v,
                ArcqlViolation::SnapshotRead { client_id: 1, op_id: 1, extra, .. }
                    if extra.contains(&NODE)
            )),
            "expected SnapshotRead{{client 1, op 1, extra∋100}}; got {violations:?}"
        );
    }

    /// 3. **G1a on an aborted property CREATE.** An aborted `CREATE (:L
    ///    {p:v})` burns node id `77`; a committed MATCH-by-property
    ///    observes it. The checker MUST report `ArcqlViolation::
    ///    AbortedReadObserved{node:77,..}` (write-side analog of the
    ///    read-side `check_detects_g1a_aborted_read`).
    #[test]
    fn check_detects_g1a_on_aborted_property_create() {
        const GHOST: u64 = 77;

        // Aborted CREATE of node 77 (burns the id into the aborted set).
        let mut writer = OpBuilder::new(0, 0, TenantId::DEFAULT, lsn(1));
        writer.intend_write(GHOST, present());
        let aborted = writer.into_aborted();

        // Committed MATCH that illegally observes the burned node.
        let mut reader = OpBuilder::new(2, 9, TenantId::DEFAULT, lsn(10));
        reader.observe_read(SCAN_SENTINEL_KEY, present());
        reader.observe_read(GHOST, present());
        let match_op = reader.into_committed(lsn(11));

        let verdict = ArcqlSiChecker::new().check(&[aborted, match_op]);
        assert!(!verdict.is_ok(), "checker must FAIL on a G1a history");
        let violations = verdict.violations().expect("violations on the bad path");
        assert!(
            violations.iter().any(|v| matches!(
                v,
                ArcqlViolation::AbortedReadObserved {
                    reader_client: 2,
                    reader_op: 9,
                    node: GHOST,
                }
            )),
            "expected AbortedReadObserved {{client 2, op 9, node 77}}; got {violations:?}"
        );
    }

    /// 4. **Negative control — a fully-legal WAW/RYW history passes.**
    ///    Guards the dual vacuity hole: an over-eager checker that fails
    ///    everything. CREATE node 1 (counter:0) → increment to 1 → the
    ///    final value matches `seed + 1`, plus a legal MATCH that reads
    ///    node 1 at a snapshot AFTER its create. `verdict.is_ok()`.
    #[test]
    fn check_passes_legal_serial_write_history() {
        const NODE: u64 = 1;

        // CREATE counter node 1 := 0, commit @ 10.
        let mut create = OpBuilder::new(0, 0, TenantId::DEFAULT, lsn(1));
        create.intend_write(NODE, Some(encode_counter(0)));
        let create_op = create.into_committed(lsn(10));

        // Serial increment: read 0 (@ snapshot 10), write 1, commit @ 20.
        let mut inc = OpBuilder::new(0, 1, TenantId::DEFAULT, lsn(10));
        inc.observe_read(NODE, Some(encode_counter(0)));
        inc.intend_write(NODE, Some(encode_counter(1)));
        let inc_op = inc.into_committed(lsn(20));

        // Legal MATCH @ snapshot 25 (after both commits): observes node 1.
        let mut reader = OpBuilder::new(1, 2, TenantId::DEFAULT, lsn(25));
        reader.observe_read(SCAN_SENTINEL_KEY, present());
        reader.observe_read(NODE, present());
        let match_op = reader.into_committed(lsn(26));

        let verdict = ArcqlSiChecker::new().check(&[create_op, inc_op, match_op]);
        assert!(
            verdict.is_ok(),
            "a fully-legal WAW/RYW history must pass; got {verdict}"
        );
        // The counter WAS checked (non-vacuous OK): seed 0 + 1 increment
        // == final 1, so the predicate ran AND passed (not skipped).
        assert_eq!(
            verdict.summary().expect("summary").counters_checked,
            1,
            "the legal history's counter must have been lost-update-checked"
        );
    }
}
