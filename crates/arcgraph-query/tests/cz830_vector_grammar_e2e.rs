//! **#830 (ADR-198 §OQ-7)** — Neo4j-compatible vector-grammar surface
//! END-TO-END tests: `SHOW VECTOR INDEXES [YIELD … WHERE …]` +
//! `CREATE VECTOR INDEX …` + `DROP INDEX … [IF EXISTS]`.
//!
//! This is the mgr-dev half of the ADR-198 §OQ-7 split (grammar +
//! proc-registration surface); the substrate binding (the actual index
//! BUILD + the non-empty SHOW VECTOR INDEXES catalog rows) is the vector
//! track's follow-up. These tests are the in-crate oracle for the
//! Neo4j-compatible vector-client initialization surface that
//! is 100% blocked over Bolt pre-#830 (see the issue: each of these
//! forms PARSE-ERRORED — `SHOW VECTOR INDEXES` → `expected show_kind`;
//! `CREATE VECTOR INDEX …` → `expected create_item`).
//!
//! STRONG ORACLES (doctrine §3): the parser tests assert the EXACT AST
//! shape (not merely "no parse error"); the SHOW execution tests assert
//! row count == 0 over the CORRECT column schema (an empty catalog
//! legitimately yields zero rows — the signal `_fetch_index_infos`
//! reads to then CREATE the index); the CREATE/DROP tests assert a TYPED
//! `NotImplemented` (parsed-but-not-built) — NOT a panic, NOT a parse
//! error, NOT a silent-empty (a no-op trampoline would be a doctrine
//! violation). The query strings are real client wire Cypher, not
//! hand-simplified approximations.

use arcgraph_query::ast::{
    Clause, CreateVectorIndexStatement, DropIndexStatement, IndexDdlStatement, IndexNameRef,
    ShowClause, ShowKind, Statement,
};
use arcgraph_query::executor::ExecutionContext;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::error::{ArcQLError, BindingError};
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
}

/// Full pipeline → result rows (panics on any stage error). SHOW over
/// an EMPTY catalog is substrate-independent, so a fresh empty stub
/// suffices for the empty-rowset oracles.
fn run(query: &str, c: &StubCatalogProvider) -> Vec<Vec<Value>> {
    run_on(query, c, &StubExecutorSubstrate::new())
}

/// Full pipeline → result rows over a CALLER-SUPPLIED substrate, so a
/// `CREATE VECTOR INDEX` and a following `SHOW VECTOR INDEXES` /
/// `queryNodes` share the per-tenant catalog (#830 / ADR-200).
fn run_on(query: &str, c: &StubCatalogProvider, s: &StubExecutorSubstrate) -> Vec<Vec<Value>> {
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, c).expect("bind");
    TypeCheckVisitor::check(&mut bound, c).expect("type-check");
    CrossSubstrateValidator::validate(&bound, c).expect("cross-substrate");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    let ctx = ExecutionContext::new(c.tenant(), c.partition());
    materialize::materialize(&plan, s, &ctx)
        .expect("materialize")
        .rows()
        .to_vec()
}

/// Bind-only — for the tests that assert a BIND error.
fn bind_err(query: &str, c: &StubCatalogProvider) -> Vec<BindingError> {
    let stmt = parse(query).expect("parse");
    match BindingVisitor::bind(&stmt, query, c) {
        Ok(_) => panic!("expected a bind error for: {query}"),
        Err(errs) => errs,
    }
}

/// Parse + bind (must SUCCEED) then assert TYPE-CHECK returns the typed
/// `NotImplemented` — the honest "parsed + bound, build not wired"
/// contract for CREATE / DROP VECTOR INDEX (ADR-198 §OQ-7).
fn typecheck_errs(query: &str, c: &StubCatalogProvider) -> Vec<ArcQLError> {
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, c).expect("bind (parses + binds)");
    match TypeCheckVisitor::check(&mut bound, c) {
        Ok(()) => panic!("expected a type-check NotImplemented for: {query}"),
        Err(errs) => errs,
    }
}

fn first_show_clause(stmt: &Statement) -> &ShowClause {
    match stmt {
        Statement::Read(q) => match &q.clauses[0] {
            Clause::Show(s) => s,
            other => panic!("expected a Show clause, got {other:?}"),
        },
        other => panic!("expected Statement::Read, got {other:?}"),
    }
}

fn create_vector_index(stmt: &Statement) -> &CreateVectorIndexStatement {
    match stmt {
        Statement::IndexDdl(IndexDdlStatement::CreateVector(c)) => c,
        other => panic!("expected Statement::IndexDdl(CreateVector), got {other:?}"),
    }
}

fn drop_index(stmt: &Statement) -> &DropIndexStatement {
    match stmt {
        Statement::IndexDdl(IndexDdlStatement::Drop(d)) => d,
        other => panic!("expected Statement::IndexDdl(Drop), got {other:?}"),
    }
}

// =====================================================================
// Gap 2 — SHOW VECTOR INDEXES (grammar + YIELD/WHERE tail + body).
// =====================================================================

#[test]
fn show_vector_indexes_bare_parses_to_vectorindexes_kind() {
    // Issue #830 raw-Cypher repro #4 — pre-#830 this was
    // `SyntaxError: expected show_kind`. STRONG oracle: exact kind +
    // empty YIELD + no WHERE.
    let stmt = parse("SHOW VECTOR INDEXES").expect("parse");
    let show = first_show_clause(&stmt);
    assert_eq!(show.kind, ShowKind::VectorIndexes);
    assert!(show.yield_items.is_empty(), "bare form → no YIELD items");
    assert!(show.where_clause.is_none(), "bare form → no WHERE");
}

#[test]
fn show_vector_indexes_bare_executes_empty() {
    // Empty vector-index catalog at this layer (build is the vector
    // track's follow-up) → 0 rows. NOT a placeholder no-op: an empty
    // catalog legitimately yields zero rows over the declared columns.
    let rows = run("SHOW VECTOR INDEXES", &cat());
    assert_eq!(
        rows.len(),
        0,
        "no vector indexes exist at the grammar layer → empty rowset"
    );
}

#[test]
fn show_vector_indexes_real_client_query_parses_binds_executes_empty() {
    // The load-bearing compatible-client wire form. The bare-kind form
    // alone does NOT
    // unblock it: it carries a YIELD + WHERE (on `$index_name`) + a
    // RETURN that projects nested map property access
    // (`options.indexConfig.`vector.dimensions``). It must parse + bind +
    // execute WITHOUT error, returning 0 rows (no such index yet → the
    // "create it" signal the client reads).
    let rows = run(
        "SHOW VECTOR INDEXES \
         YIELD name, labelsOrTypes, properties, options \
         WHERE name = $index_name \
         RETURN labelsOrTypes AS labels, properties, \
         options.indexConfig.`vector.dimensions` AS dimensions, \
         options.indexConfig.`vector.filterable_properties` AS filterable_properties",
        &cat(),
    );
    assert_eq!(
        rows.len(),
        0,
        "empty vector-index catalog → 0 rows; the load-bearing proof is \
         that the real YIELD-WHERE-RETURN client query parses + binds + \
         executes WITHOUT error"
    );
}

#[test]
fn show_vector_indexes_yield_where_ast_shape() {
    // STRONG oracle on the YIELD/WHERE tail AST (the real client shape).
    let stmt = parse(
        "SHOW VECTOR INDEXES YIELD name, labelsOrTypes AS labels WHERE name = $index_name RETURN labels",
    )
    .expect("parse");
    // The SHOW clause is clause[0]; RETURN is clause[1].
    let show = match stmt {
        Statement::Read(ref q) => match &q.clauses[0] {
            Clause::Show(s) => s,
            other => panic!("expected Show, got {other:?}"),
        },
        ref other => panic!("expected Read, got {other:?}"),
    };
    assert_eq!(show.kind, ShowKind::VectorIndexes);
    assert_eq!(
        show.yield_items,
        vec![
            ("name".to_string(), None),
            ("labelsOrTypes".to_string(), Some("labels".to_string())),
        ],
        "YIELD items captured with aliases"
    );
    assert!(show.where_clause.is_some(), "WHERE predicate captured");
}

#[test]
fn show_vector_indexes_invalid_yield_column_rejected_at_bind() {
    // DISCRIMINATING oracle: YIELDing a column the SHOW VECTOR INDEXES
    // output does not produce must reject at bind (proves the YIELD
    // resolves against the fixed column set, not a free-for-all).
    let errs = bind_err(
        "SHOW VECTOR INDEXES YIELD notAColumn RETURN notAColumn",
        &cat(),
    );
    assert!(
        errs.iter()
            .any(|e| matches!(e, BindingError::InvalidYieldColumn { .. })),
        "YIELDing a non-output column must be an InvalidYieldColumn bind error; got {errs:?}"
    );
}

#[test]
fn bare_show_kinds_still_parse_unaffected() {
    // Regression: the existing ADR-197 single-word SHOW kinds are
    // unaffected by the VECTOR INDEXES / YIELD-tail additions.
    assert_eq!(
        first_show_clause(&parse("SHOW INDEXES").expect("parse")).kind,
        ShowKind::Indexes
    );
    assert_eq!(
        first_show_clause(&parse("SHOW CONSTRAINTS").expect("parse")).kind,
        ShowKind::Constraints
    );
    assert_eq!(
        first_show_clause(&parse("SHOW DATABASES").expect("parse")).kind,
        ShowKind::Databases
    );
}

// =====================================================================
// Gap 4 — CREATE VECTOR INDEX DDL (grammar + AST + bind + NotImplemented).
// =====================================================================

#[test]
fn create_vector_index_real_client_form_parses_to_ast() {
    // The compatible-client wire form: `$name` (parameter),
    // `IF NOT EXISTS`,
    // `ON n.embedding` (NO parens), OPTIONS with backtick-escaped keys +
    // `toInteger($dimensions)` (function-call) + `$similarity_fn`
    // (parameter) values. Pre-#830: `SyntaxError: expected create_item`.
    let stmt = parse(
        "CREATE VECTOR INDEX $name IF NOT EXISTS FOR (n:Chunk) ON n.embedding \
         OPTIONS { indexConfig: { `vector.dimensions`: toInteger($dimensions), \
         `vector.similarity_function`: $similarity_fn } }",
    )
    .expect("parse");
    let c = create_vector_index(&stmt);
    assert_eq!(
        c.name,
        IndexNameRef::Param("name".to_string()),
        "$name param"
    );
    assert!(c.if_not_exists, "IF NOT EXISTS present");
    assert_eq!(c.pattern_var, "n");
    assert_eq!(c.label, "Chunk");
    assert_eq!(
        c.property, "embedding",
        "ON n.embedding → property `embedding`"
    );
    assert!(c.options.is_some(), "OPTIONS map captured");
}

#[test]
fn create_vector_index_issue_repro_literal_form_parses_to_ast() {
    // Issue #830 raw-Cypher repro #2: a LITERAL index name (`cz`),
    // literal option values (`16`, `'cosine'`).
    let stmt = parse(
        "CREATE VECTOR INDEX cz IF NOT EXISTS FOR (n:CzChunk) ON n.embedding \
         OPTIONS {indexConfig: {`vector.dimensions`: 16, `vector.similarity_function`: 'cosine'}}",
    )
    .expect("parse");
    let c = create_vector_index(&stmt);
    assert_eq!(c.name, IndexNameRef::Literal("cz".to_string()));
    assert!(c.if_not_exists);
    assert_eq!(c.label, "CzChunk");
    assert_eq!(c.property, "embedding");
    assert!(c.options.is_some());
}

#[test]
fn create_vector_index_bare_no_options_no_if_not_exists_parses() {
    // The minimal form: no IF NOT EXISTS, no OPTIONS, parenthesized
    // `ON (n.prop)` (the Neo4j-docs variant). Proves the optional parts
    // are genuinely optional + both ON forms parse.
    let stmt = parse("CREATE VECTOR INDEX myIdx FOR (m:Doc) ON (m.vec)").expect("parse");
    let c = create_vector_index(&stmt);
    assert_eq!(c.name, IndexNameRef::Literal("myIdx".to_string()));
    assert!(!c.if_not_exists, "no IF NOT EXISTS");
    assert_eq!(c.pattern_var, "m");
    assert_eq!(c.label, "Doc");
    assert_eq!(c.property, "vec");
    assert!(c.options.is_none(), "no OPTIONS clause");
}

#[test]
fn create_vector_index_registers_then_show_reflects_it() {
    // #830 / ADR-200: CREATE VECTOR INDEX now ACCEPTS + REGISTERS (it is
    // no longer a typed NotImplemented). A following SHOW VECTOR INDEXES
    // reflects the registered entry over the declared columns — the
    // EXACT-ROW oracle (not merely "non-empty"). Literal name + literal
    // OPTIONS here (the $param wire form is covered by the op's unit
    // tests; the Bolt param-threading gap is tracked separately).
    let c = cat();
    let s = StubExecutorSubstrate::new();
    let created = run_on(
        "CREATE VECTOR INDEX cz806vec IF NOT EXISTS FOR (n:CzChunk) ON n.embedding \
         OPTIONS {indexConfig: {`vector.dimensions`: 16, `vector.similarity_function`: 'cosine'}}",
        &c,
        &s,
    );
    assert_eq!(created.len(), 0, "CREATE VECTOR INDEX returns zero rows");

    // Bare SHOW VECTOR INDEXES → all six declared columns, in order:
    // name, type, entityType, labelsOrTypes, properties, options.
    let rows = run_on("SHOW VECTOR INDEXES", &c, &s);
    assert_eq!(rows.len(), 1, "one registered index → exactly one SHOW row");
    let row = &rows[0];
    assert_eq!(row.len(), 6, "six declared SHOW VECTOR INDEXES columns");
    assert_eq!(row[0], Value::String("cz806vec".into()), "name");
    assert_eq!(row[1], Value::String("VECTOR".into()), "type");
    assert_eq!(row[2], Value::String("NODE".into()), "entityType");
    assert_eq!(
        row[3],
        Value::List(vec![Value::String("CzChunk".into())]),
        "labelsOrTypes = [label]"
    );
    assert_eq!(
        row[4],
        Value::List(vec![Value::String("embedding".into())]),
        "properties = [property]"
    );
    // options.indexConfig.vector.dimensions == 16 — the value langchain's
    // retrieve_existing_index reads to validate its embedding dimension.
    match &row[5] {
        Value::Map(opts) => match opts.get("indexConfig") {
            Some(Value::Map(cfg)) => {
                assert_eq!(
                    cfg.get("vector.dimensions"),
                    Some(&Value::Integer(16)),
                    "options.indexConfig.`vector.dimensions`"
                );
                assert_eq!(
                    cfg.get("vector.similarity_function"),
                    Some(&Value::String("cosine".into())),
                    "options.indexConfig.`vector.similarity_function`"
                );
            }
            other => panic!("options.indexConfig must be a map, got {other:?}"),
        },
        other => panic!("options must be a map, got {other:?}"),
    }
}

#[test]
fn create_vector_index_then_generic_show_indexes_yields_vector_options() {
    // #892: Neo4j's generic SHOW INDEXES is a superset of SHOW VECTOR
    // INDEXES. Compatible clients introspect with this generic form
    // and explicitly YIELD options; both the declared column set and the
    // executor rows must match the vector-specific path.
    let c = cat();
    let s = StubExecutorSubstrate::new();
    let created = run_on(
        "CREATE VECTOR INDEX cz892vec IF NOT EXISTS FOR (n:CzChunk) ON n.embedding \
         OPTIONS {indexConfig: {`vector.dimensions`: 16, `vector.similarity_function`: 'cosine'}}",
        &c,
        &s,
    );
    assert_eq!(created.len(), 0, "CREATE VECTOR INDEX returns zero rows");

    let generic = run_on(
        "SHOW INDEXES \
         YIELD name, type, entityType, labelsOrTypes, properties, options \
         WHERE type = 'VECTOR'",
        &c,
        &s,
    );
    let vector = run_on(
        "SHOW VECTOR INDEXES \
         YIELD name, type, entityType, labelsOrTypes, properties, options \
         WHERE type = 'VECTOR'",
        &c,
        &s,
    );
    assert_eq!(
        generic, vector,
        "generic SHOW INDEXES must expose the same vector row shape as SHOW VECTOR INDEXES"
    );
    assert_eq!(
        generic.len(),
        1,
        "WHERE type = 'VECTOR' must see the registered vector index"
    );
    let row = &generic[0];
    assert_eq!(row[0], Value::String("cz892vec".into()), "name");
    assert_eq!(row[1], Value::String("VECTOR".into()), "type");
    assert_eq!(row[2], Value::String("NODE".into()), "entityType");
    assert_eq!(
        row[3],
        Value::List(vec![Value::String("CzChunk".into())]),
        "labelsOrTypes = [label]"
    );
    assert_eq!(
        row[4],
        Value::List(vec![Value::String("embedding".into())]),
        "properties = [property]"
    );
    match &row[5] {
        Value::Map(opts) => match opts.get("indexConfig") {
            Some(Value::Map(cfg)) => {
                assert_eq!(
                    cfg.get("vector.dimensions"),
                    Some(&Value::Integer(16)),
                    "options.indexConfig.`vector.dimensions`"
                );
                assert_eq!(
                    cfg.get("vector.similarity_function"),
                    Some(&Value::String("cosine".into())),
                    "options.indexConfig.`vector.similarity_function`"
                );
            }
            other => panic!("options.indexConfig must be a map, got {other:?}"),
        },
        other => panic!("options must be a map, got {other:?}"),
    }
}

#[test]
fn create_vector_index_if_not_exists_idempotent_via_pipeline() {
    // IF NOT EXISTS re-create through the full pipeline → no error, no
    // duplicate (SHOW still reflects exactly one row). `from_texts`
    // called twice must work.
    let c = cat();
    let s = StubExecutorSubstrate::new();
    let q = "CREATE VECTOR INDEX cz806vec IF NOT EXISTS FOR (n:CzChunk) ON n.embedding \
             OPTIONS {indexConfig: {`vector.dimensions`: 16, `vector.similarity_function`: 'cosine'}}";
    run_on(q, &c, &s);
    run_on(q, &c, &s); // second IF NOT EXISTS create — idempotent
    let rows = run_on("SHOW VECTOR INDEXES", &c, &s);
    assert_eq!(
        rows.len(),
        1,
        "IF NOT EXISTS re-create must not duplicate the catalog entry"
    );
}

#[test]
fn show_vector_indexes_empty_before_any_create() {
    // Before any CREATE the catalog is empty → zero rows (the "no such
    // index yet" signal the client reads to then CREATE the index).
    let c = cat();
    let s = StubExecutorSubstrate::new();
    let rows = run_on("SHOW VECTOR INDEXES", &c, &s);
    assert_eq!(rows.len(), 0, "empty catalog → zero SHOW rows");
}

// =====================================================================
// Gap 4 — DROP INDEX DDL (the GENERIC Neo4j drop the clients emit).
// =====================================================================

#[test]
fn drop_index_real_client_form_parses_to_ast() {
    // The compatible-client wire form: `DROP INDEX $name IF EXISTS`.
    // Neo4j has NO
    // `DROP VECTOR INDEX` — this generic drop-by-name covers vector
    // indexes (cite-correctness: implement the form the client sends).
    let stmt = parse("DROP INDEX $name IF EXISTS").expect("parse");
    let d = drop_index(&stmt);
    assert_eq!(d.name, IndexNameRef::Param("name".to_string()));
    assert!(d.if_exists, "IF EXISTS present");
}

#[test]
fn drop_index_literal_no_if_exists_parses_to_ast() {
    // `Neo4jVector` (neo4j_vector.py:277) emits `DROP INDEX {name}` —
    // a literal name, no IF EXISTS.
    let stmt = parse("DROP INDEX cz806vec").expect("parse");
    let d = drop_index(&stmt);
    assert_eq!(d.name, IndexNameRef::Literal("cz806vec".to_string()));
    assert!(!d.if_exists, "no IF EXISTS");
}

#[test]
fn drop_index_typecheck_is_notimplemented_not_panic() {
    let errs = typecheck_errs("DROP INDEX cz806vec IF EXISTS", &cat());
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ArcQLError::NotImplemented { feature, .. } if feature.contains("DROP INDEX")
        )),
        "DROP INDEX must surface a typed NotImplemented (lifecycle owned by vector track, \
         ADR-198 §OQ-7); got {errs:?}"
    );
}

// =====================================================================
// Regression — the DDL alternation must NOT hijack graph CREATE.
// =====================================================================

#[test]
fn plain_create_node_still_parses_as_read_not_ddl() {
    // The load-bearing PEG-backtrack proof: `CREATE (n:Foo {…})` fails
    // `create_vector_index_ddl` at the 2nd token (`kw_vector`), so
    // `ddl_statement` as a whole fails and `statement` backtracks to
    // `read_query` → `create_clause`. A regression here would silently
    // break ALL graph writes.
    let stmt = parse("CREATE (n:Foo {x: 1})").expect("parse");
    assert!(
        matches!(stmt, Statement::Read(_)),
        "plain CREATE must remain a Read (graph write), got {stmt:?}"
    );
}

#[test]
fn model_specific_define_index_is_not_public_syntax() {
    assert!(parse("DEFINE INDEX myIdx ON title USING MODEL_BACKEND").is_err());
}
