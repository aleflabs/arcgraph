//! ADR-152-amendment-01 — MERGE match-branch LABEL enforcement,
//! active end-to-end verification (ADR-133 §D-4 Query class) through
//! the PRODUCTION `Dispatcher::dispatch("graph.raw_query")` path.
//!
//! Unlike the `arcgraph-query` smoke tests (which hand-build the bind
//! catalog), this suite routes every statement through the real
//! JSON-RPC dispatcher → `raw_query_tool` → `StorageRawQueryExecutor`
//! → `CrudExecutorSubstrate`, with the catalog REBUILT per statement
//! from the live backend (`build_catalog_for_tenant`). So "the label is
//! not interned" is a REAL property of the substrate state, not a
//! fixture choice — the most faithful test of the pull-forward.
//!
//! Oracle (STRONG): assert BOTH the `writes.nodes_created` counter AND
//! the resulting per-label populations via label-filtered
//! `MATCH (n:Label) RETURN n` `row_count`. Tests `cross_label_*` and
//! `bare_heterogeneous_*` FAIL at HEAD before the amendment (the
//! match-branch cross-matched a different label → `nodes_created=0`).
//!
//! Audit refs: O-3 (bare heterogeneous), ADR-152 Risk #5 (cross-label
//! property). Harness reused from `raw_query_write_common`.

#![allow(clippy::unwrap_used)]

mod raw_query_write_common;
use raw_query_write_common::{fresh_dispatcher, parse_body, raw_query};

/// `writes.nodes_created` from a write response body.
fn nodes_created(body: &serde_json::Value) -> u64 {
    body["writes"]["nodes_created"]
        .as_u64()
        .unwrap_or_else(|| panic!("writes.nodes_created missing/non-int: {body}"))
}

/// `row_count` of a `MATCH (n:Label) RETURN n` — the count of live
/// nodes carrying that label (the label-enforced read oracle).
fn label_count(d: &raw_query_write_common::TestDispatcher, label: &str) -> u64 {
    let q = format!("MATCH (n:{label}) RETURN n");
    let body = parse_body(&raw_query(d, &q));
    body["row_count"]
        .as_u64()
        .unwrap_or_else(|| panic!("MATCH (n:{label}) row_count missing/non-int: {body}"))
}

#[test]
fn cross_label_property_merge_creates_user_not_matches_account() {
    // ADR-152 Risk #5 — `CREATE (:Account {id:42})` then
    // `MERGE (n:User {id:42})` MUST create a NEW :User. The existing
    // :Account {id:42} has the same property under a DIFFERENT label;
    // the label-enforced match-branch must NOT fire on it.
    // FAILS at HEAD (nodes_created would be 0 — cross-matched Account).
    let d = fresh_dispatcher();

    let acct = parse_body(&raw_query(&d, "CREATE (n:Account {id: 42}) RETURN n"));
    assert_eq!(nodes_created(&acct), 1, "Account created");

    let merge = parse_body(&raw_query(&d, "MERGE (n:User {id: 42})"));
    assert_eq!(
        nodes_created(&merge),
        1,
        "MERGE (n:User {{id:42}}) creates a NEW :User — must NOT \
         cross-match the :Account {{id:42}} (ADR-152 Risk #5)"
    );

    // Label oracle: exactly one :User and one :Account.
    assert_eq!(label_count(&d, "User"), 1, "one :User exists post-MERGE");
    assert_eq!(label_count(&d, "Account"), 1, ":Account untouched");
}

#[test]
fn bare_heterogeneous_merge_creates_user() {
    // Audit O-3 — `CREATE (:Article)` then bare `MERGE (n:User)` MUST
    // create a :User. At HEAD bare MERGE matched the existence of ANY
    // node (the :Article) → never created a :User. FAILS at HEAD.
    let d = fresh_dispatcher();

    assert_eq!(
        nodes_created(&parse_body(&raw_query(&d, "CREATE (n:Article)"))),
        1
    );

    let merge = parse_body(&raw_query(&d, "MERGE (n:User)"));
    assert_eq!(
        nodes_created(&merge),
        1,
        "bare MERGE (n:User) on a heterogeneous graph creates a :User \
         (closes audit O-3); must NOT match the :Article"
    );

    assert_eq!(label_count(&d, "User"), 1, "one :User created");
    assert_eq!(label_count(&d, "Article"), 1, ":Article untouched");
}

#[test]
fn interned_label_merge_matches_existing_no_duplicate() {
    // `CREATE (n:User {id:1})` then `MERGE (n:User {id:1})` — the label
    // is interned (catalog rebuild sees the prior CREATE), so the
    // match-branch lowers to Scan{Some(id)} + property-filter → matches
    // → NO second node. The plan-cache `commits_observed` watermark +
    // per-statement catalog rebuild make this cross-statement-safe.
    let d = fresh_dispatcher();

    assert_eq!(
        nodes_created(&parse_body(&raw_query(
            &d,
            "CREATE (n:User {id: 1}) RETURN n"
        ))),
        1
    );

    let merge = parse_body(&raw_query(&d, "MERGE (n:User {id: 1})"));
    assert_eq!(
        nodes_created(&merge),
        0,
        "MERGE matched the existing :User {{id:1}} — no duplicate"
    );
    assert_eq!(label_count(&d, "User"), 1, "still exactly one :User");
}

#[test]
fn merge_enforces_label_and_property_together() {
    // Pre: :Account {id:42} (right prop, wrong label) + :User {id:99}
    // (right label, wrong prop). `MERGE (n:User {id:42})` matches
    // NEITHER → creates a fresh :User {id:42}. Only "both enforced"
    // yields a create (label-only would match User{99}; property-only —
    // the HEAD behavior — would match Account{42}). FAILS at HEAD.
    let d = fresh_dispatcher();

    assert_eq!(
        nodes_created(&parse_body(&raw_query(&d, "CREATE (n:Account {id: 42})"))),
        1
    );
    assert_eq!(
        nodes_created(&parse_body(&raw_query(&d, "CREATE (n:User {id: 99})"))),
        1
    );

    let merge = parse_body(&raw_query(&d, "MERGE (n:User {id: 42})"));
    assert_eq!(
        nodes_created(&merge),
        1,
        "label+property both enforced: matched neither wrong-label \
         :Account {{id:42}} nor wrong-property :User {{id:99}} → created"
    );

    assert_eq!(
        label_count(&d, "User"),
        2,
        ":User population is {{id:99, id:42}}"
    );
    assert_eq!(
        label_count(&d, "Account"),
        1,
        ":Account {{id:42}} untouched"
    );
}
