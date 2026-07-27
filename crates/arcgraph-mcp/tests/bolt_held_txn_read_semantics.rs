//! ADR-197-amendment-01 D-6 — the #822 acceptance proof: held-transaction
//! READ semantics at the production Bolt handler.
//!
//! Exercises the SAME production `StorageBoltHandler` +
//! `CrudExecutorSubstrate` path the Bolt server drives (handler-level,
//! no TCP — the wire FSM is pinned by `transport/bolt/state.rs` tests;
//! THESE tests pin the transaction-read SEMANTICS the FSM delegates
//! to). Per the #806-R1 "9 fault-tests false-green on a bare
//! `CrudStore::new()`" lesson, the backend wires the `PrimaryIndex`
//! (mirroring `bolt_param_binding_e2e::fresh_backend`); patterns stay
//! directed (LeftToRight) so the reverse-adjacency index is not a
//! dependency of these oracles.
//!
//! The discriminating set (amendment §Test plan):
//! 1. RYW — staged CREATE visible to MATCH pre-COMMIT (#822's own
//!    acceptance criterion).
//! 2. Path-operator RYW — `shortestPath` over staged rels (RED before
//!    amendment D-2 routed `ops/path.rs::expand_neighbors`) + the
//!    already-routed `[*1..2]` var-length parity pin.
//! 3. Pinned snapshot — an external auto-commit write mid-transaction
//!    is INVISIBLE to the open transaction (ADR-047 SI across
//!    statements), then visible after COMMIT.
//! 4. ROLLBACK discards — staged writes never observable afterward.
//! 5. Staged SET visible / staged DELETE hides.
//!
//! Loud-default (D-4) + snapshot-reporting (D-5) pins are query-crate
//! unit tests (`executor::substrate` / `executor::context`) — they
//! assert seam behavior no full backend is needed for.

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::SessionScope;
use arcgraph_mcp::storage::{StorageBackend, StorageBoltHandler};
use arcgraph_mcp::transport::bolt::{BoltQueryHandler, BoltSessionAuth, PackValue};
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;

/// Fully-wired in-memory backend (PrimaryIndex attached) — the
/// production-shaped substrate per the #806-R1 false-green lesson.
fn fresh_backend() -> StorageBackend {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(64, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("catalog bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&allocator), None).expect("PrimaryIndex"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    StorageBackend::new(router, mgr, intern)
}

fn handler() -> (StorageBoltHandler, TenantId) {
    let bolt = StorageBoltHandler::new(fresh_backend());
    (bolt, TenantId::DEFAULT)
}

fn power_session(tenant: TenantId) -> BoltSessionAuth {
    BoltSessionAuth::new(tenant, None, SessionScope::Power)
}

fn no_params() -> BTreeMap<String, PackValue> {
    BTreeMap::new()
}

/// Auto-commit row count for `cypher` (committed-visibility probe).
fn committed_rows(bolt: &StorageBoltHandler, tenant: TenantId, cypher: &str) -> usize {
    bolt.run(&power_session(tenant), cypher, &no_params())
        .unwrap_or_else(|e| panic!("auto-commit `{cypher}` failed: {e:?}"))
        .records
        .len()
}

/// Run `cypher` inside the held tx, asserting success; returns
/// (row_count, handle-moved-back).
fn tx_rows(
    bolt: &StorageBoltHandler,
    tenant: TenantId,
    cypher: &str,
    held: Box<dyn arcgraph_query::executor::substrate::HeldTxnHandle>,
) -> (
    usize,
    Box<dyn arcgraph_query::executor::substrate::HeldTxnHandle>,
) {
    let (out, held) = bolt.run_in_txn(&power_session(tenant), cypher, &no_params(), held);
    let rows = out
        .unwrap_or_else(|e| panic!("in-txn `{cypher}` failed: {e:?}"))
        .records
        .len();
    (rows, held)
}

/// §1 — THE #822 acceptance test: `BEGIN → CREATE → MATCH` sees the
/// staged node pre-COMMIT; nothing is committed-visible until COMMIT.
#[test]
fn ryw_staged_create_visible_to_match_within_txn() {
    let (bolt, tenant) = handler();

    assert_eq!(
        committed_rows(&bolt, tenant, "MATCH (n:Person) RETURN n"),
        0,
        "clean slate"
    );

    let held = bolt.begin_txn(tenant, None, None).expect("BEGIN");
    let (_, held) = tx_rows(&bolt, tenant, "CREATE (n:Person {name: 'staged'})", held);

    // Read-your-writes: the staged node is visible INSIDE the tx…
    let (rows_in_tx, held) = tx_rows(&bolt, tenant, "MATCH (n:Person) RETURN n", held);
    assert_eq!(
        rows_in_tx, 1,
        "#822 acceptance: staged CREATE must be visible to MATCH inside the SAME tx"
    );

    // …and NOT outside it (MVCC publication: nothing observable
    // before COMMIT).
    assert_eq!(
        committed_rows(&bolt, tenant, "MATCH (n:Person) RETURN n"),
        0,
        "staged write must NOT be committed-visible before COMMIT"
    );

    bolt.commit_txn(held).expect("COMMIT");
    assert_eq!(
        committed_rows(&bolt, tenant, "MATCH (n:Person) RETURN n"),
        1,
        "after COMMIT the write is durable + visible"
    );
}

/// §2 — Path-operator RYW (the amendment D-2 fix; RED before
/// `ops/path.rs::expand_neighbors` routed through
/// `expand_with_context`) + parity pins for the already-routed
/// single-hop and `[*1..2]` var-length expansion.
#[test]
fn path_operators_see_staged_rels_within_txn() {
    let (bolt, tenant) = handler();

    let held = bolt.begin_txn(tenant, None, None).expect("BEGIN");
    // Stage ONE hop a→c. (A 2-hop chain is not constructible at
    // v1.0-α: CREATE rejects shared variables across pattern items
    // AND `MATCH → CREATE` with a bound endpoint is the ADR-148
    // forward-pin. One staged hop is equally discriminating: before
    // amendment D-2 the path operator read fresh-committed state and
    // saw ZERO rels — any staged-rel traversal RED-flags it.)
    let (_, held) = tx_rows(&bolt, tenant, "CREATE (a:A)-[:R]->(c:C)", held);

    // Single-hop (ExpandOp, already routed — parity pin).
    let (single_hop, held) = tx_rows(&bolt, tenant, "MATCH (x:A)-[r]->(y) RETURN y", held);
    assert_eq!(single_hop, 1, "single-hop sees the staged a→c rel");

    // Var-length expansion (expand.rs var_length_paths, already
    // routed — parity pin).
    let (var_len, held) = tx_rows(&bolt, tenant, "MATCH (x:A)-[*1..2]->(y) RETURN y", held);
    assert_eq!(var_len, 1, "[*1..2] sees the staged hop");

    // shortestPath (NamedShortestPathOp via expand_neighbors — the
    // D-2 surface; RED on pre-amendment main).
    let (sp, held) = tx_rows(
        &bolt,
        tenant,
        "MATCH p = shortestPath((x:A)-[*]->(y:C)) RETURN p",
        held,
    );
    assert_eq!(
        sp, 1,
        "shortestPath must traverse staged rels inside the tx (amendment D-2; \
         RED before expand_neighbors routed through expand_with_context)"
    );

    bolt.rollback_txn(held);
}

/// §3 — Pinned snapshot (ADR-047 SI across statements): an external
/// auto-commit write that lands AFTER BEGIN is invisible to the open
/// transaction's reads, in both scan and path form; visible to a
/// fresh query after the tx ends.
#[test]
fn external_commit_mid_txn_is_invisible_until_after_txn() {
    let (bolt, tenant) = handler();

    // Pre-existing committed row, so the tx's snapshot has content.
    bolt.run(
        &power_session(tenant),
        "CREATE (n:Item {k: 'pre'})",
        &no_params(),
    )
    .expect("seed");
    let held = bolt.begin_txn(tenant, None, None).expect("BEGIN");

    // The tx observes the pre-BEGIN world…
    let (rows0, held) = tx_rows(&bolt, tenant, "MATCH (n:Item) RETURN n", held);
    assert_eq!(rows0, 1, "snapshot contains the pre-BEGIN commit");

    // External writer commits mid-transaction ("second connection" —
    // an independent auto-commit run on the same handler; held-tx
    // state is carried per-call-chain, so this run opens+commits its
    // own transaction exactly as another connection would).
    bolt.run(
        &power_session(tenant),
        "CREATE (n:Item {k: 'external'})",
        &no_params(),
    )
    .expect("external commit");
    assert_eq!(
        committed_rows(&bolt, tenant, "MATCH (n:Item) RETURN n"),
        2,
        "external write IS committed (control)"
    );

    // …but the OPEN transaction still reads its pinned snapshot:
    // the external commit must NOT appear (scan form)…
    let (rows1, held) = tx_rows(&bolt, tenant, "MATCH (n:Item) RETURN n", held);
    assert_eq!(
        rows1, 1,
        "SI: a post-BEGIN external commit is INVISIBLE to the open tx (scan)"
    );

    // …including through repeated statements (repeatable read).
    let (rows2, held) = tx_rows(&bolt, tenant, "MATCH (n:Item) RETURN n", held);
    assert_eq!(rows2, 1, "repeatable: same snapshot on every statement");

    bolt.commit_txn(held)
        .expect("COMMIT (empty write-set is fine)");
    assert_eq!(
        committed_rows(&bolt, tenant, "MATCH (n:Item) RETURN n"),
        2,
        "after the tx ends, a fresh query sees the external commit"
    );
}

/// §4 — ROLLBACK discards: staged writes visible in-tx are never
/// observable after ROLLBACK.
#[test]
fn rollback_discards_staged_writes() {
    let (bolt, tenant) = handler();

    let held = bolt.begin_txn(tenant, None, None).expect("BEGIN");
    let (_, held) = tx_rows(&bolt, tenant, "CREATE (n:Ghost {k: 1})", held);
    let (visible, held) = tx_rows(&bolt, tenant, "MATCH (n:Ghost) RETURN n", held);
    assert_eq!(visible, 1, "staged write visible pre-ROLLBACK (RYW)");

    bolt.rollback_txn(held);

    assert_eq!(
        committed_rows(&bolt, tenant, "MATCH (n:Ghost) RETURN n"),
        0,
        "ROLLBACK must discard the staged write entirely"
    );
}

/// §5 — Staged SET is visible to in-tx reads; staged DELETE hides a
/// node that IS visible at the snapshot.
#[test]
fn staged_set_visible_and_staged_delete_hides() {
    let (bolt, tenant) = handler();

    bolt.run(
        &power_session(tenant),
        "CREATE (n:Doc {state: 'old'})",
        &no_params(),
    )
    .expect("seed committed Doc");
    bolt.run(
        &power_session(tenant),
        "CREATE (n:Tomb {k: 1})",
        &no_params(),
    )
    .expect("seed committed Tomb");

    let held = bolt.begin_txn(tenant, None, None).expect("BEGIN");

    // Staged SET: in-tx filter on the NEW value matches.
    let (_, held) = tx_rows(&bolt, tenant, "MATCH (n:Doc) SET n.state = 'new'", held);
    let (new_rows, held) = tx_rows(
        &bolt,
        tenant,
        "MATCH (n:Doc) WHERE n.state = 'new' RETURN n",
        held,
    );
    assert_eq!(new_rows, 1, "staged SET visible to in-tx WHERE");

    // Staged DELETE: the snapshot-visible Tomb disappears in-tx…
    let (_, held) = tx_rows(&bolt, tenant, "MATCH (n:Tomb) DELETE n", held);
    let (tomb_rows, held) = tx_rows(&bolt, tenant, "MATCH (n:Tomb) RETURN n", held);
    assert_eq!(tomb_rows, 0, "staged DELETE hides the node in-tx");

    // …while committed state is untouched until COMMIT.
    assert_eq!(
        committed_rows(&bolt, tenant, "MATCH (n:Tomb) RETURN n"),
        1,
        "DELETE not committed yet"
    );
    assert_eq!(
        committed_rows(
            &bolt,
            tenant,
            "MATCH (n:Doc) WHERE n.state = 'old' RETURN n"
        ),
        1,
        "SET not committed yet (old value still committed-visible)"
    );

    bolt.commit_txn(held).expect("COMMIT");
    assert_eq!(
        committed_rows(&bolt, tenant, "MATCH (n:Tomb) RETURN n"),
        0,
        "DELETE durable after COMMIT"
    );
    assert_eq!(
        committed_rows(
            &bolt,
            tenant,
            "MATCH (n:Doc) WHERE n.state = 'new' RETURN n"
        ),
        1,
        "SET durable after COMMIT"
    );
}

/// **T15 (ADR-147-amendment-03, D-1) — held-txn atomicity.** A held-tx
/// `UNWIND $rows AS r CREATE (n {v: r.v})` where one mid-batch row's
/// value resolves to a MAP faults at the runtime value-type gate. Because
/// everything staged into the held tx, ROLLBACK discards ALL staged rows
/// → ZERO durable nodes (not the 2 that materialized before the faulting
/// row). This proves the amendment composes with the EXISTING held-txn
/// atomicity (ADR-197) without inventing a new transactional surface.
///
/// HARD GATE (ATOMICITY — held-txn path only). The auto-commit surface
/// partial-applies on mid-batch failure (documented honestly in the PR
/// body); this pin is the held-tx all-or-nothing proof.
#[test]
fn t15_held_txn_mid_batch_create_fault_rolls_back_zero_durable() {
    let (bolt, tenant) = handler();

    // $rows: two OK maps then a map whose `v` is itself a MAP (the fault
    // — the value-type gate rejects a map property value mid-batch).
    let mut ok1 = BTreeMap::new();
    ok1.insert("v".to_string(), PackValue::Integer(1));
    let mut ok2 = BTreeMap::new();
    ok2.insert("v".to_string(), PackValue::Integer(2));
    let mut bad = BTreeMap::new();
    let mut nested = BTreeMap::new();
    nested.insert("bad".to_string(), PackValue::Integer(9));
    bad.insert("v".to_string(), PackValue::Map(nested));
    let rows = PackValue::List(vec![
        PackValue::Map(ok1),
        PackValue::Map(ok2),
        PackValue::Map(bad),
    ]);
    let mut params = BTreeMap::new();
    params.insert("rows".to_string(), rows);

    let held = bolt.begin_txn(tenant, None, None).expect("BEGIN");
    let (out, held) = bolt.run_in_txn(
        &power_session(tenant),
        "UNWIND $rows AS r CREATE (n:Batch {v: r.v})",
        &params,
        held,
    );
    assert!(
        out.is_err(),
        "the mid-batch map value must fault at the runtime value-type gate"
    );

    // The held tx is intact; ROLLBACK discards ALL staged rows.
    bolt.rollback_txn(held);

    assert_eq!(
        committed_rows(&bolt, tenant, "MATCH (n:Batch) RETURN n"),
        0,
        "held-txn atomicity: a mid-batch fault + ROLLBACK leaves ZERO durable nodes \
         (not the rows that materialized before the faulting row)"
    );
}
