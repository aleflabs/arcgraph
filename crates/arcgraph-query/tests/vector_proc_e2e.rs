//! **#830 (D1 + D4)** — `Neo4jVector` search-path procedure END-TO-END
//! tests: `CALL dbms.components()` + `CALL db.index.vector.queryNodes(
//! indexName, k, queryVector) [YIELD node, score]`.
//!
//! These are the in-crate oracle for the langchain-neo4j `Neo4jVector`
//! search surface (the D1 version handshake + the D4 per-query KNN
//! search). They exercise the FULL front-end + executor (parse → bind →
//! type-check → cross-substrate → lower → materialize) with STRONG `==`
//! oracles over the result rows — exact rows, exact score order, exact
//! k-truncation, and the substrate-unavailable FAULT (a structured
//! error, never a silent-empty).
//!
//! The D2/D3 `SHOW VECTOR INDEXES` / `CREATE VECTOR INDEX` DDL is
//! grammar-gated and OUT of this slice (owned by mgr-dev).
//!
//! ## Active-verification linchpin (D1 version gate)
//!
//! [`dbms_components_versions0_clears_langchain_5_23_gate`] re-implements
//! a compatible client's version parse (`versions[0]` → split `-` →
//! split `.` → int tuple → pad to 3)
//! and asserts the result clears BOTH gates the vector surface checks:
//! `has_vector_index_support` (`>= (5, 11, 0)`) and — critically —
//! `is_version_5_23_or_above` (`>= (5, 23, 0)`), which is what routes
//! `db.index.vector.queryNodes` to the SUPPORTED path. This proves the
//! advertised version clears the REAL gate, verified against
//! version_utils.py — not guessed.

use arcgraph_core::{LabelId, NodeId};
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::executor::{
    ExecutionContext, ExecutionError, RankedHit, StubExecutorSubstrate,
};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
}

/// Full front-end → [`LogicalPlan`] (panics on any front-end-stage error).
fn lower(query: &str, c: &StubCatalogProvider) -> LogicalPlan {
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, c).expect("bind");
    TypeCheckVisitor::check(&mut bound, c).expect("type-check");
    CrossSubstrateValidator::validate(&bound, c).expect("cross-substrate");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
}

/// Full pipeline → result rows against a caller-provided substrate
/// (panics on any stage error, incl. execution).
fn run_on(query: &str, c: &StubCatalogProvider, s: &StubExecutorSubstrate) -> Vec<Vec<Value>> {
    let plan = lower(query, c);
    let ctx = ExecutionContext::new(c.tenant(), c.partition());
    materialize::materialize(&plan, s, &ctx)
        .expect("materialize")
        .rows()
        .to_vec()
}

/// Full pipeline expecting an EXECUTION error (panics if it succeeds).
fn exec_err(query: &str, c: &StubCatalogProvider, s: &StubExecutorSubstrate) -> ExecutionError {
    let plan = lower(query, c);
    let ctx = ExecutionContext::new(c.tenant(), c.partition());
    match materialize::materialize(&plan, s, &ctx) {
        Ok(r) => panic!(
            "expected an execution error for `{query}`; got {} row(s)",
            r.rows().len()
        ),
        Err(e) => e,
    }
}

fn node(id: u64, label: u32) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(label)))
}

/// A vector-enabled stub pre-baked with `hits` for `(tenant,
/// "embedding", tag(query_vec))`. The query vector `[1.5, 0.0]` matches
/// the `[1.5, 0.0]` list literal the test queries pass.
fn vector_stub(c: &StubCatalogProvider, hits: Vec<RankedHit>) -> StubExecutorSubstrate {
    let qv = [1.5_f32, 0.0];
    let tag = StubExecutorSubstrate::vector_search_tag_for(&qv);
    StubExecutorSubstrate::new()
        .with_vector_substrate()
        .with_vector_hit(c.tenant(), "embedding", &tag, hits)
}

// =====================================================================
// D1 — dbms.components()
// =====================================================================

#[test]
fn dbms_components_returns_exact_single_version_row() {
    // Standalone `CALL dbms.components()` (no YIELD) — the cz-probe D1
    // shape. Yields all columns: (name, versions, edition). EXACT row.
    let s = StubExecutorSubstrate::new();
    let rows = run_on("CALL dbms.components()", &cat(), &s);
    assert_eq!(rows.len(), 1, "dbms.components → exactly one row");
    assert_eq!(
        rows[0],
        vec![
            Value::String("Neo4j Kernel".to_string()),
            Value::List(vec![Value::String("5.26.0".to_string())]),
            Value::String("community".to_string()),
        ],
        "dbms.components row must be exactly (name, versions, edition)"
    );
}

#[test]
fn dbms_components_yield_subset_projects_versions_and_edition() {
    // langchain reads `records[0]["versions"][0]` + `["edition"]`. Prove
    // an explicit YIELD of just those two columns binds + projects.
    let s = StubExecutorSubstrate::new();
    let rows = run_on(
        "CALL dbms.components() YIELD versions, edition RETURN versions, edition",
        &cat(),
        &s,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0],
        vec![
            Value::List(vec![Value::String("5.26.0".to_string())]),
            Value::String("community".to_string()),
        ]
    );
}

#[test]
fn dbms_components_versions0_clears_langchain_5_23_gate() {
    // THE active-verification linchpin. Re-implement a compatible
    // client's version parse over the EXACT row our proc emits, then
    // assert it clears the two gates the
    // vector surface checks. Oracle = version_utils.py (verified read).
    let s = StubExecutorSubstrate::new();
    let rows = run_on("CALL dbms.components()", &cat(), &s);

    // get_version(): version = records[0]["versions"][0]; edition = ["edition"].
    let versions = match &rows[0][1] {
        Value::List(l) => l,
        other => panic!("versions column must be a List, got {other:?}"),
    };
    let version_str = match &versions[0] {
        Value::String(s) => s.clone(),
        other => panic!("versions[0] must be a String, got {other:?}"),
    };
    let edition = match &rows[0][2] {
        Value::String(s) => s.clone(),
        other => panic!("edition column must be a String, got {other:?}"),
    };

    // version_main, *_ = version.split("-"); tuple(map(int, split(".")));
    // pad to 3 if shorter.
    let version_main = version_str.split('-').next().unwrap();
    let mut version_tuple: Vec<i64> = version_main
        .split('.')
        .map(|p| p.parse::<i64>().expect("version component is an int"))
        .collect();
    while version_tuple.len() < 3 {
        version_tuple.push(0);
    }
    assert_eq!(
        version_tuple,
        vec![5, 26, 0],
        "parsed version tuple must be (5, 26, 0)"
    );

    // has_vector_index_support: version_tuple >= (5, 11, 0).
    assert!(
        version_tuple.as_slice() >= [5, 11, 0].as_slice(),
        "must clear has_vector_index_support gate (>= 5.11.0); got {version_tuple:?}"
    );
    // is_version_5_23_or_above: version_tuple >= (5, 23, 0) — the gate
    // that routes db.index.vector.queryNodes to the SUPPORTED path.
    assert!(
        version_tuple.as_slice() >= [5, 23, 0].as_slice(),
        "must clear is_version_5_23_or_above gate (>= 5.23.0); got {version_tuple:?}"
    );
    // edition == "enterprise" → is_enterprise; we advertise community.
    assert_ne!(edition, "enterprise", "edition must not gate as enterprise");
    // "aura" in version → is_aura; our version string must not.
    assert!(!version_str.contains("aura"), "must not gate as Aura");
}

// =====================================================================
// D4 — db.index.vector.queryNodes(indexName, k, queryVector)
// =====================================================================

#[test]
fn query_nodes_returns_exact_hits_in_score_order() {
    // Pre-bake 3 hits (score-descending). Ask for k=3. Assert EXACTLY
    // those 3 rows, in order, with (node, score) at the right slots —
    // byte-equal to the pre-baked hits (NOT merely "non-empty").
    let c = cat();
    let hits = vec![
        RankedHit {
            node: node(1, 1),
            score: 0.99,
        },
        RankedHit {
            node: node(2, 1),
            score: 0.50,
        },
        RankedHit {
            node: node(3, 1),
            score: 0.10,
        },
    ];
    let s = vector_stub(&c, hits);
    let rows = run_on(
        "CALL db.index.vector.queryNodes('anyname', 3, [1.5, 0.0]) \
         YIELD node, score RETURN node, score",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![
            vec![Value::Node(node(1, 1)), Value::Float(0.99)],
            vec![Value::Node(node(2, 1)), Value::Float(0.50)],
            vec![Value::Node(node(3, 1)), Value::Float(0.10)],
        ],
        "exact (node, score) rows in score-descending order"
    );
}

#[test]
fn query_nodes_standalone_no_yield_returns_all_columns() {
    // The cz-probe D4 shape: standalone `CALL ...queryNodes(...)` with
    // no YIELD → yields all output columns (node, score).
    let c = cat();
    let hits = vec![RankedHit {
        node: node(7, 2),
        score: 0.42,
    }];
    let s = vector_stub(&c, hits);
    let rows = run_on(
        "CALL db.index.vector.queryNodes('cztest', 2, [1.5, 0.0])",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![Value::Node(node(7, 2)), Value::Float(0.42)]],
        "standalone CALL yields all columns (node, score)"
    );
}

#[test]
fn query_nodes_k_truncates_to_requested_top_k() {
    // Pre-bake 5 hits, ask k=3 → exactly the first 3 (the substrate
    // truncates by top-k). Proves k flows into vector_search.
    let c = cat();
    let hits = vec![
        RankedHit {
            node: node(1, 1),
            score: 0.99,
        },
        RankedHit {
            node: node(2, 1),
            score: 0.80,
        },
        RankedHit {
            node: node(3, 1),
            score: 0.60,
        },
        RankedHit {
            node: node(4, 1),
            score: 0.40,
        },
        RankedHit {
            node: node(5, 1),
            score: 0.20,
        },
    ];
    let s = vector_stub(&c, hits);
    let rows = run_on(
        "CALL db.index.vector.queryNodes('any', 3, [1.5, 0.0]) YIELD node, score RETURN node, score",
        &c,
        &s,
    );
    assert_eq!(rows.len(), 3, "k=3 truncates a 5-hit substrate to 3 rows");
    assert_eq!(
        rows,
        vec![
            vec![Value::Node(node(1, 1)), Value::Float(0.99)],
            vec![Value::Node(node(2, 1)), Value::Float(0.80)],
            vec![Value::Node(node(3, 1)), Value::Float(0.60)],
        ],
        "the first 3 hits in score order"
    );
}

#[test]
fn query_nodes_substrate_unavailable_is_structured_error_not_empty() {
    // THE load-bearing fault-injection oracle: vector substrate OFF →
    // a structured SubstrateAccessError (IndexUnavailable), NEVER a
    // silent-empty rowset. langchain must see a real error or real hits.
    let c = cat();
    // A fresh stub has NO vector substrate attached.
    let s = StubExecutorSubstrate::new();
    let e = exec_err(
        "CALL db.index.vector.queryNodes('any', 2, [1.5, 0.0]) YIELD node, score RETURN node, score",
        &c,
        &s,
    );
    match e {
        ExecutionError::Substrate(
            arcgraph_query::executor::SubstrateAccessError::IndexUnavailable(ref what),
        ) => assert_eq!(what, "vector", "must name the unavailable vector substrate"),
        other => panic!("expected Substrate(IndexUnavailable(\"vector\")); got {other:?}"),
    }
}

#[test]
fn query_nodes_wrong_arity_is_clean_error_not_panic() {
    // 2 args (missing queryVector) → clean Eval error at execution.
    let c = cat();
    let s = vector_stub(&c, vec![]);
    let e = exec_err(
        "CALL db.index.vector.queryNodes('any', 3) YIELD node, score RETURN node, score",
        &c,
        &s,
    );
    match e {
        ExecutionError::Eval(ref msg) => assert!(
            msg.contains("expects 3 arguments"),
            "arity error message should be explicit; got: {msg}"
        ),
        other => panic!("expected Eval(arity); got {other:?}"),
    }
}

#[test]
fn query_nodes_non_integer_k_is_clean_error_not_panic() {
    // k as a string literal → clean Eval error at execution (no panic).
    let c = cat();
    let s = vector_stub(&c, vec![]);
    let e = exec_err(
        "CALL db.index.vector.queryNodes('any', 'three', [1.5, 0.0]) YIELD node, score RETURN node, score",
        &c,
        &s,
    );
    match e {
        ExecutionError::Eval(ref msg) => assert!(
            msg.contains("k (arg 2)"),
            "k-type error message should name arg 2; got: {msg}"
        ),
        other => panic!("expected Eval(k type); got {other:?}"),
    }
}

#[test]
fn query_nodes_non_list_query_vector_is_clean_error_not_panic() {
    // queryVector as a scalar → clean Eval error (no panic).
    let c = cat();
    let s = vector_stub(&c, vec![]);
    let e = exec_err(
        "CALL db.index.vector.queryNodes('any', 2, 0.5) YIELD node, score RETURN node, score",
        &c,
        &s,
    );
    match e {
        ExecutionError::Eval(ref msg) => assert!(
            msg.contains("query vector (arg 3)"),
            "query-vector error should name arg 3; got: {msg}"
        ),
        other => panic!("expected Eval(query vector); got {other:?}"),
    }
}

// =====================================================================
// Parse smoke — pins that the call rides the EXISTING grammar (#806).
// =====================================================================

#[test]
fn query_nodes_parses_on_existing_grammar() {
    // No grammar change is owed for D4 — `CALL proc(args) YIELD …
    // RETURN …` parses on the PR #806 grammar. This pins it.
    assert!(
        parse("CALL db.index.vector.queryNodes('i', 2, [0.1, 0.2]) YIELD node, score RETURN node, score")
            .is_ok(),
        "db.index.vector.queryNodes must parse on the existing CALL-proc grammar"
    );
    assert!(
        parse("CALL dbms.components()").is_ok(),
        "dbms.components must parse on the existing CALL-proc grammar"
    );
}

// =====================================================================
// #830 / ADR-200 — CREATE VECTOR INDEX → queryNodes TRUTHFUL
// name→property resolution via the per-tenant catalog.
// =====================================================================

#[test]
fn query_nodes_resolves_to_registered_property_via_catalog() {
    // DISCRIMINATING oracle: register an index whose property ("vec")
    // DIFFERS from the served convention ("embedding"). queryNodes(name)
    // must search the REGISTERED property — proving TRUTHFUL
    // name→property resolution via the #830/ADR-200 catalog, NOT the
    // convention. The hit is baked ONLY against "vec"; if queryNodes
    // wrongly fell back to the convention "embedding", it would find
    // ZERO rows. (Closes R1 #861 Finding #1's residual — the advisory
    // shim becomes a real lookup.)
    let c = cat();
    let qv = [1.5_f32, 0.0];
    let tag = StubExecutorSubstrate::vector_search_tag_for(&qv);
    let hits = vec![RankedHit {
        node: node(7, 1),
        score: 0.9,
    }];
    let s = StubExecutorSubstrate::new()
        .with_vector_substrate()
        .with_vector_hit(c.tenant(), "vec", &tag, hits);
    // Register 'myidx' → property 'vec' (NOT the convention).
    let created = run_on("CREATE VECTOR INDEX myidx FOR (n:Doc) ON n.vec", &c, &s);
    assert_eq!(created.len(), 0, "CREATE VECTOR INDEX returns zero rows");
    // queryNodes('myidx', 1, [1.5, 0.0]) → catalog resolves 'myidx' →
    // 'vec' → finds the hit baked against 'vec'.
    let rows = run_on(
        "CALL db.index.vector.queryNodes('myidx', 1, [1.5, 0.0]) YIELD node, score \
         RETURN node, score",
        &c,
        &s,
    );
    assert_eq!(
        rows.len(),
        1,
        "queryNodes('myidx') must resolve to the REGISTERED property 'vec' (catalog), \
         not the convention 'embedding' — found {} row(s)",
        rows.len()
    );
}

#[test]
fn query_nodes_unregistered_name_falls_back_to_convention() {
    // BACK-COMPAT oracle: with NO registered index, queryNodes(anyName)
    // falls back to the served-convention property ("embedding") — the
    // pre-catalog #861 behavior. The hit is baked against "embedding";
    // an unregistered name resolves there.
    let c = cat();
    let qv = [1.5_f32, 0.0];
    let tag = StubExecutorSubstrate::vector_search_tag_for(&qv);
    let hits = vec![RankedHit {
        node: node(3, 1),
        score: 0.7,
    }];
    let s = StubExecutorSubstrate::new()
        .with_vector_substrate()
        .with_vector_hit(c.tenant(), "embedding", &tag, hits);
    // No CREATE — 'notRegistered' is unknown to the catalog.
    let rows = run_on(
        "CALL db.index.vector.queryNodes('notRegistered', 1, [1.5, 0.0]) YIELD node, score \
         RETURN node, score",
        &c,
        &s,
    );
    assert_eq!(
        rows.len(),
        1,
        "an unregistered index name must fall back to the convention property 'embedding'"
    );
}

#[test]
fn query_nodes_registered_with_convention_property_still_resolves() {
    // The langchain happy-path shape: the index property IS the
    // convention ("embedding"). Registering it + querying it resolves
    // (the catalog lookup returns 'embedding'; same as the fallback, but
    // now via a real entry).
    let c = cat();
    let qv = [1.5_f32, 0.0];
    let tag = StubExecutorSubstrate::vector_search_tag_for(&qv);
    let hits = vec![RankedHit {
        node: node(11, 1),
        score: 0.8,
    }];
    let s = StubExecutorSubstrate::new()
        .with_vector_substrate()
        .with_vector_hit(c.tenant(), "embedding", &tag, hits);
    run_on(
        "CREATE VECTOR INDEX cz806vec IF NOT EXISTS FOR (n:CzChunk) ON n.embedding \
         OPTIONS {indexConfig: {`vector.dimensions`: 16, `vector.similarity_function`: 'cosine'}}",
        &c,
        &s,
    );
    let rows = run_on(
        "CALL db.index.vector.queryNodes('cz806vec', 1, [1.5, 0.0]) YIELD node, score \
         RETURN node, score",
        &c,
        &s,
    );
    assert_eq!(
        rows.len(),
        1,
        "registered convention-property index resolves + finds the hit"
    );
}
