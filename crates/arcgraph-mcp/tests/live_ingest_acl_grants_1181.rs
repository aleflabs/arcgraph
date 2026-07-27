//! #1181 [MUST-CON-07] — LIVE `graph.ingest` `acl_grants` write-through.
//!
//! A running server must accept a pushed doc carrying ACLs via the live
//! MCP `graph.ingest` surface, and the pushed grants must be ENFORCED:
//! the granted principal sees the doc, an ungranted principal does not.
//! Before #1181, [`IngestRequest`] had `#[serde(deny_unknown_fields)]`
//! and no `acl_grants` field, so `graph.ingest` with `acl_grants`
//! returned `-32602 unknown field acl_grants`; ACL attachment happened
//! only through a separate startup batch path. This suite
//! proves the SAME post-commit `PermissionIndex::apply_doc_acl`
//! write-through now fires on the live push path.
//!
//! # Chain under test (production accessor chain — ADR-212 §D-4 Seam-1)
//!
//! ```text
//! IngestRequest { acl_grants: [...] }            [wire, REAL deny_unknown_fields]
//!   → ingest_tool                                [REAL]
//!   → StorageIngestProvider::ingest              [REAL — commits, THEN apply_doc_acl]
//!   → TenantHandle::permissions()                [REAL, ADR-037-am-02]
//!   → EffectivePermissions::is_visible           [REAL enforcement oracle]
//! ```
//!
//! The enforcement assertions mirror `permission_fidelity.rs` and the
//! `permissions` module's own `is_visible` truth table.
//!
//! # RED-on-revert (§D-8 discipline)
//!
//! Neuter the `apply_doc_acl` call inside
//! `StorageIngestProvider::ingest` (or the `apply_live_acl_grants`
//! helper) and `granted_principal_sees_doc_*` flips RED: alice's doc
//! stays UNCLASSIFIED ⇒ `is_visible` is `false` ⇒ the positive
//! assertion fails. The `null` / absent branches stay GREEN under that
//! neuter (they assert invisibility), so the positive control is the
//! load-bearing half — captured verbatim in the PR body.

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_core::{NodeId, PageId, PartitionId, TenantId};
use arcgraph_mcp::storage::{StorageBackend, StorageIngestProvider};
use arcgraph_mcp::tools::ingest::{
    AclGrant, DroppedAclGrant, IngestBatch, IngestProvider, IngestRecordOutcome, IngestRequest,
    NodeIngest, ingest_tool,
};
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::mutation_log::{Bm25IndexStoreHandle, Bm25StoreError};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::permissions::{PUBLIC_PRINCIPAL, PermissionIndex};
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::vector_store::{VectorPageStoreHandle, VectorStoreError};

const ALICE: &str = "alice";
const BOB: &str = "bob";

fn tenant() -> TenantId {
    // `SystemCatalog::bootstrap` registers exactly the DEFAULT tenant;
    // the router fails-closed on unregistered tenants (ADR-037).
    TenantId::DEFAULT
}

/// No-op vector arena handle (availability gate only).
#[derive(Debug)]
struct NoopVectorStore;

impl VectorPageStoreHandle for NoopVectorStore {
    fn install_or_replace(
        &self,
        _tenant: TenantId,
        _page_id: PageId,
        _bytes: &[u8],
    ) -> Result<(), VectorStoreError> {
        Ok(())
    }
    fn restore_page_bytes(
        &self,
        _tenant: TenantId,
        _page_id: PageId,
        _bytes: &[u8],
    ) -> Result<(), VectorStoreError> {
        Ok(())
    }
}

/// No-op BM25 commit handle (availability gate only).
#[derive(Debug)]
struct NoopBm25Store;

impl Bm25IndexStoreHandle for NoopBm25Store {
    fn commit_pending(&self, _tenant: TenantId) -> Result<(), Bm25StoreError> {
        Ok(())
    }
    fn rollback_pending(&self, _tenant: TenantId) -> Result<(), Bm25StoreError> {
        Ok(())
    }
}

/// `permission_fidelity::fresh_backend` wire-pattern.
fn fresh_backend() -> StorageBackend {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(64, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("catalog bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&allocator), None)
            .expect("PrimaryIndex::new"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let router = Arc::new(MultiTenantRouter::new_with_bm25(
        catalog,
        crud,
        Some(Arc::new(NoopVectorStore)),
        Some(Arc::new(NoopBm25Store)),
    ));
    let intern = Arc::new(InternTable::new());
    StorageBackend::new(router, mgr, intern)
}

/// The REAL per-tenant index, reached through the SAME production
/// accessor chain the dispatcher's searcher resolves via
/// (`router → TenantHandle::permissions()`, ADR-037-amendment-02).
fn permissions_for(backend: &StorageBackend) -> Arc<PermissionIndex> {
    let handle = backend
        .router()
        .route(tenant(), PartitionId::ZERO)
        .expect("route DEFAULT tenant");
    Arc::clone(handle.permissions())
}

fn doc_node(external_id: &str) -> NodeIngest {
    NodeIngest {
        external_id: Some(external_id.to_owned()),
        label: "Document".to_owned(),
        properties: BTreeMap::from([(
            "body".to_owned(),
            serde_json::Value::String(format!("body of {external_id}")),
        )]),
    }
}

/// Drive ONE node through the live provider with the supplied
/// `acl_grants`, returning the committed internal node id.
fn ingest_one(
    provider: &StorageIngestProvider,
    external_id: &str,
    acl_grants: Vec<AclGrant>,
) -> u64 {
    let summary = provider
        .ingest(
            tenant(),
            IngestBatch {
                nodes: vec![doc_node(external_id)],
                relationships: vec![],
                acl_grants,
            },
        )
        .expect("live ingest ok");
    assert_eq!(summary.failed_count, 0, "doc must commit: {summary:?}");
    match &summary.records[0] {
        IngestRecordOutcome::Inserted { internal_id, .. }
        | IngestRecordOutcome::Idempotent { internal_id, .. } => *internal_id,
        other => panic!("doc did not commit: {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────
// The assertion matrix
// ─────────────────────────────────────────────────────────────────────

/// THE headline (#1181): a doc pushed on the LIVE path with
/// `acl_grants: [{external_id, read_principals:["alice"]}]` is enforced
/// — alice sees it, bob does NOT. This is the RED-on-revert load-bearing
/// assertion (neuter `apply_doc_acl` ⇒ alice's positive control fails).
#[test]
fn granted_principal_sees_doc_ungranted_does_not() {
    let backend = fresh_backend();
    let permissions = permissions_for(&backend);
    let provider = StorageIngestProvider::new(backend);

    let node = ingest_one(
        &provider,
        "doc:007",
        vec![AclGrant {
            external_id: "doc:007".to_owned(),
            read_principals: Some(vec![ALICE.to_owned()]),
        }],
    );

    // Positive control (RED-on-revert load-bearing): alice was granted.
    assert!(
        permissions.effective(ALICE).is_visible(NodeId::new(node)),
        "alice was granted read on the LIVE-pushed doc and must see it"
    );
    // Negative: bob was not granted ⇒ invisible.
    assert!(
        !permissions.effective(BOB).is_visible(NodeId::new(node)),
        "bob was NOT granted read and must not see the doc"
    );
}

/// `read_principals: null` ⇒ the doc stays UNCLASSIFIED (skip — do NOT
/// call `apply_doc_acl`) ⇒ invisible to every principal (fail-closed).
/// Pins `read_principals: None` semantics.
#[test]
fn null_read_principals_leaves_doc_unclassified_invisible() {
    let backend = fresh_backend();
    let permissions = permissions_for(&backend);
    let provider = StorageIngestProvider::new(backend);

    let node = ingest_one(
        &provider,
        "doc:008",
        vec![AclGrant {
            external_id: "doc:008".to_owned(),
            read_principals: None,
        }],
    );

    // No ACL entry was written ⇒ UNCLASSIFIED ⇒ deny-all.
    assert_eq!(
        permissions.tagged_docs(),
        0,
        "null read_principals must NOT tag the doc"
    );
    assert!(!permissions.effective(ALICE).is_visible(NodeId::new(node)));
    assert!(
        !permissions
            .effective(PUBLIC_PRINCIPAL)
            .is_visible(NodeId::new(node)),
        "UNCLASSIFIED doc is invisible even to PUBLIC"
    );
}

/// `acl_grants` ABSENT (empty) ⇒ backward-compatible: no ACL applied,
/// the doc behaves exactly as a pre-#1181 ingest (no enforcement entry).
#[test]
fn absent_acl_grants_is_backward_compatible() {
    let backend = fresh_backend();
    let permissions = permissions_for(&backend);
    let provider = StorageIngestProvider::new(backend);

    let node = ingest_one(&provider, "doc:009", vec![]);

    assert_eq!(
        permissions.tagged_docs(),
        0,
        "absent acl_grants must apply no ACL (backward-compatible)"
    );
    assert!(!permissions.effective(ALICE).is_visible(NodeId::new(node)));
}

/// An explicit empty grant list `Some([])` ⇒ grant-to-nobody: the doc
/// IS tagged (distinct from UNCLASSIFIED in provenance) but no principal
/// — not even PUBLIC — can read it. Mirrors the seed path + the
/// `permissions` module's empty-set truth.
#[test]
fn empty_grant_list_tags_doc_granted_to_nobody() {
    let backend = fresh_backend();
    let permissions = permissions_for(&backend);
    let provider = StorageIngestProvider::new(backend);

    let node = ingest_one(
        &provider,
        "doc:010",
        vec![AclGrant {
            external_id: "doc:010".to_owned(),
            read_principals: Some(vec![]),
        }],
    );

    assert_eq!(
        permissions.tagged_docs(),
        1,
        "an explicit empty grant set tags the doc (grant-to-nobody)"
    );
    assert!(!permissions.effective(ALICE).is_visible(NodeId::new(node)));
    assert!(
        !permissions
            .effective(PUBLIC_PRINCIPAL)
            .is_visible(NodeId::new(node))
    );
}

/// PUBLIC grant reaches every principal (sanity that the live path
/// threads the principal set verbatim into the index).
#[test]
fn public_grant_reaches_every_principal() {
    let backend = fresh_backend();
    let permissions = permissions_for(&backend);
    let provider = StorageIngestProvider::new(backend);

    let node = ingest_one(
        &provider,
        "doc:011",
        vec![AclGrant {
            external_id: "doc:011".to_owned(),
            read_principals: Some(vec![PUBLIC_PRINCIPAL.to_owned()]),
        }],
    );

    assert!(permissions.effective(ALICE).is_visible(NodeId::new(node)));
    assert!(permissions.effective(BOB).is_visible(NodeId::new(node)));
    assert!(
        permissions
            .effective("anyone-else")
            .is_visible(NodeId::new(node))
    );
}

/// A grant whose `external_id` did NOT commit (never in `nodes`) is
/// skipped — surfacing nothing fatal; the request still returns Ok and
/// the committed doc that WAS granted is enforced (fail-closed for the
/// unresolved one).
#[test]
fn unresolved_grant_external_id_is_skipped_not_fatal() {
    let backend = fresh_backend();
    let permissions = permissions_for(&backend);
    let provider = StorageIngestProvider::new(backend);

    let summary = provider
        .ingest(
            tenant(),
            IngestBatch {
                nodes: vec![doc_node("doc:012")],
                relationships: vec![],
                acl_grants: vec![
                    AclGrant {
                        external_id: "doc:012".to_owned(),
                        read_principals: Some(vec![ALICE.to_owned()]),
                    },
                    // No node with this external_id was submitted.
                    AclGrant {
                        external_id: "doc:does-not-exist".to_owned(),
                        read_principals: Some(vec![BOB.to_owned()]),
                    },
                ],
            },
        )
        .expect("call returns Ok despite an unresolved grant");
    assert_eq!(summary.failed_count, 0);

    let node = match &summary.records[0] {
        IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
        other => panic!("doc:012 did not commit: {other:?}"),
    };

    // Only the resolved grant was applied.
    assert_eq!(permissions.tagged_docs(), 1, "exactly one doc tagged");
    assert!(permissions.effective(ALICE).is_visible(NodeId::new(node)));
    // bob's grant referenced a doc that never committed ⇒ no widening.
    assert!(!permissions.effective(BOB).is_visible(NodeId::new(node)));

    // #1198 hardening: the dropped grant is now SURFACED (not silently
    // skipped). The caller can see EXACTLY which security-relevant grant
    // did not apply.
    assert_eq!(
        summary.dropped_acl_grants,
        vec![DroppedAclGrant::unresolved("doc:does-not-exist")],
        "the unresolved grant must be surfaced in dropped_acl_grants"
    );
}

/// #1198 [MUST-CON-07 hardening] — THE CZ repro. `graph.ingest` with a
/// node `doc-typo` + an `acl_grant` for `doc-tpyo` (a TYPO) must:
/// (a) still INSERT the node (`inserted_count:1`),
/// (b) still report `failed_count:0` (the node did NOT fail), AND
/// (c) SURFACE the dropped grant in `dropped_acl_grants` rather than
///     silently reporting full success while the grant vanished.
///
/// Before #1198 the response was `{failed_count:0, inserted_count:1}`
/// with NO signal the grant was dropped — the caller had no way to learn
/// the security-relevant grant didn't apply (the intended grantee
/// silently loses access; the doc is left UNCLASSIFIED). This test is the
/// RED-on-revert oracle: revert the `apply_live_acl_grants` dropped-list
/// capture and `dropped_acl_grants` is empty ⇒ the assertion flips RED.
///
/// The skip BEHAVIOR is unchanged (fail-closed): the typo'd grant is
/// still NOT applied (widening on an unresolved external_id would be
/// strictly worse). Only the SILENCE is removed.
#[test]
fn cz_typo_grant_is_surfaced_not_silently_dropped() {
    let backend = fresh_backend();
    let permissions = permissions_for(&backend);
    let provider = StorageIngestProvider::new(backend);

    let summary = provider
        .ingest(
            tenant(),
            IngestBatch {
                // Node committed under external_id "doc-typo".
                nodes: vec![doc_node("doc-typo")],
                relationships: vec![],
                // Grant references "doc-tpyo" — a TYPO that resolves to
                // NO committed node.
                acl_grants: vec![AclGrant {
                    external_id: "doc-tpyo".to_owned(),
                    read_principals: Some(vec![ALICE.to_owned()]),
                }],
            },
        )
        .expect("call returns Ok — the node commits, the grant is dropped");

    // (a) the node still inserted.
    assert_eq!(summary.inserted_count, 1, "the node still commits");
    // (b) the node did NOT fail — failed_count counts failed NODE/REL
    //     inserts, NOT dropped grants (a different axis).
    assert_eq!(summary.failed_count, 0, "the node did not fail");
    // (c) the dropped grant is now SURFACED (the #1198 fix).
    assert_eq!(
        summary.dropped_acl_grants,
        vec![DroppedAclGrant {
            external_id: "doc-tpyo".to_owned(),
            reason: "unresolved".to_owned(),
        }],
        "the typo'd grant must surface in dropped_acl_grants (not silent)"
    );

    // The skip behavior is UNCHANGED (fail-closed): the grant was NOT
    // applied — alice does not gain access (the doc stays UNCLASSIFIED).
    let node = match &summary.records[0] {
        IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
        other => panic!("doc-typo did not commit: {other:?}"),
    };
    assert_eq!(
        permissions.tagged_docs(),
        0,
        "no doc was tagged — the typo'd grant resolved to nothing (fail-closed)"
    );
    assert!(
        !permissions.effective(ALICE).is_visible(NodeId::new(node)),
        "alice gains NO access — the grant did not apply (fail-closed)"
    );
}

/// #1198 — the surfaced drop is greppable in the RENDERED wire response,
/// not just the in-memory summary. Drives the full
/// `IngestRequest → ingest_tool → render_response` seam and asserts the
/// JSON body carries `dropped_acl_grants` with the typo'd external_id +
/// reason. A pre-#1198 silent drop would render NO such field.
#[test]
fn cz_typo_grant_surfaces_in_rendered_wire_body() {
    let backend = fresh_backend();
    let provider = StorageIngestProvider::new(backend);

    let params = serde_json::json!({
        "tenant_id": tenant().raw(),
        "nodes": [{"external_id": "doc-typo", "label": "Document",
                   "properties": {"text": "secret"}}],
        "acl_grants": [{"external_id": "doc-tpyo", "read_principals": [ALICE]}]
    });
    let req: IngestRequest = serde_json::from_value(params).expect("parses");
    let value = ingest_tool(&provider, tenant(), req).expect("ingest_tool ok");
    let body = value["body"].as_str().expect("json body");

    // The dropped grant is visible + greppable in the wire body.
    assert!(
        body.contains("dropped_acl_grants"),
        "wire body must carry the dropped_acl_grants field; body={body}"
    );
    assert!(
        body.contains("doc-tpyo"),
        "wire body must name the dropped external_id; body={body}"
    );
    assert!(
        body.contains("unresolved"),
        "wire body must carry the drop reason; body={body}"
    );
    // The node still succeeded — failed_count stays 0.
    assert!(
        body.contains("\"failed_count\":0"),
        "the node did not fail; body={body}"
    );
    assert!(
        body.contains("\"inserted_count\":1"),
        "the node inserted; body={body}"
    );
}

/// #1198 — backward-compat: when NO grant is dropped, `dropped_acl_grants`
/// is ABSENT from the wire body (`skip_serializing_if = "Vec::is_empty"`),
/// so every pre-#1198 caller / pinned shape is unaffected.
#[test]
fn no_dropped_grant_omits_field_from_wire_body() {
    let backend = fresh_backend();
    let provider = StorageIngestProvider::new(backend);

    let params = serde_json::json!({
        "tenant_id": tenant().raw(),
        "nodes": [{"external_id": "doc:014", "label": "Document"}],
        "acl_grants": [{"external_id": "doc:014", "read_principals": [ALICE]}]
    });
    let req: IngestRequest = serde_json::from_value(params).expect("parses");
    let value = ingest_tool(&provider, tenant(), req).expect("ingest_tool ok");
    let body = value["body"].as_str().expect("json body");

    assert!(
        !body.contains("dropped_acl_grants"),
        "no drop ⇒ field omitted (backward-compatible); body={body}"
    );
}

/// The WIRE path end-to-end: `graph.ingest` with `acl_grants` no longer
/// rejects as `-32602 unknown field` (the #1181 bug) — it parses,
/// commits, and the grant is enforced. Drives the FULL
/// `IngestRequest → ingest_tool → StorageIngestProvider` seam, proving
/// the `deny_unknown_fields` acceptance + the field threading.
#[test]
fn graph_ingest_wire_path_accepts_and_enforces_acl_grants() {
    let backend = fresh_backend();
    let permissions = permissions_for(&backend);
    let provider = StorageIngestProvider::new(backend);

    // The exact wire payload an MCP client pushes. Before #1181 this
    // deserialization (deny_unknown_fields) failed on `acl_grants`.
    let params = serde_json::json!({
        "tenant_id": tenant().raw(),
        "nodes": [{"external_id": "doc:013", "label": "Document",
                   "properties": {"body": "live wire payload"}}],
        "acl_grants": [{"external_id": "doc:013", "read_principals": [ALICE]}]
    });
    let req: IngestRequest =
        serde_json::from_value(params).expect("acl_grants must parse (deny_unknown_fields)");

    let value = ingest_tool(&provider, tenant(), req).expect("ingest_tool ok");
    let body = value["body"].as_str().expect("json body");
    let summary: arcgraph_mcp::tools::ingest::IngestSummary =
        serde_json::from_str(body).expect("summary parses");
    let node = match &summary.records[0] {
        IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
        other => panic!("doc:013 did not commit via wire: {other:?}"),
    };

    assert!(
        permissions.effective(ALICE).is_visible(NodeId::new(node)),
        "wire-pushed acl_grants must be enforced for the granted principal"
    );
    assert!(!permissions.effective(BOB).is_visible(NodeId::new(node)));
}

/// `IngestRequest` still REJECTS a genuinely unknown field (the
/// `deny_unknown_fields` discipline is intact — adding `acl_grants` did
/// not loosen it). A typo'd `acl_grant` (singular) must reject.
#[test]
fn ingest_request_still_rejects_genuinely_unknown_field() {
    let params = serde_json::json!({
        "tenant_id": tenant().raw(),
        "nodes": [],
        "acl_grant": []  // typo (singular)
    });
    let res: Result<IngestRequest, _> = serde_json::from_value(params);
    assert!(
        res.is_err(),
        "a typo'd acl_grant (singular) must still reject"
    );
}
