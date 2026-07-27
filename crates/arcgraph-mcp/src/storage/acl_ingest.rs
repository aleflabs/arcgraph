//! Source-ACL ingest seam — ADR-212 §D-3 stage-1 (MUST-CON-02).
//!
//! **ACLs arrive ALONGSIDE content, through the same ingest path, as
//! graph data** (ADR-212 §D-2(a)): one [`ingest_docs_with_acls`] call
//! writes, through ONE [`IngestProvider`],
//!
//! 1. the content nodes themselves;
//! 2. the `_Acl*` provenance graph data — one
//!    [`ACL_PRINCIPAL_LABEL`] node per distinct grant subject and one
//!    [`ACL_GRANT_TYPE`] edge per `(subject → doc)` read grant
//!    (auditable: *"why can A see X?"* is a 1-hop traversal; the
//!    FR-D-04-class provenance falls out of graph storage for free);
//! 3. the write-through to the derived enforcement plane
//!    ([`PermissionIndex::apply_doc_acl`], ADR-212 §D-2(b)) — in the
//!    SAME flow, so query-time enforcement and graph provenance can
//!    never drift across the call boundary.
//!
//! The caller supplies grants already extracted from source rows. This
//! crate deliberately does not depend on external ingestion adapters.
//!
//! # Fail-closed ordering (ADR-212 §D-5 "direction of failure")
//!
//! The enforcement-index write-through happens **after** the
//! provenance batch commits. A crash between the two leaves committed
//! `_Acl*` provenance with NO enforcement entry — the docs stay
//! UNCLASSIFIED ⇒ invisible to every principal (narrow). The reverse
//! ordering could leave enforcement grants whose provenance is
//! missing; both orderings deny nothing they should grant, but only
//! provenance-first keeps the index derivable from committed graph
//! state (the stage-2 rebuild path's invariant).
//!
//! Per-doc content-record failures (validation, idempotency conflict)
//! skip that doc's grants entirely — a doc that did not commit gets
//! no ACL entry (and no provenance edges), so a partial batch can
//! only UNDER-grant.
//!
//! Per-record failures INSIDE the provenance batch (a `_AclPrincipal`
//! node or `_ACL_GRANT` edge rejected) fail the WHOLE call before the
//! write-through: tolerating them would leave the enforcement index
//! granting access the provenance graph cannot explain. Every doc in
//! the failed call stays UNCLASSIFIED (fail-closed); the idempotency
//! keys make a clean re-sync converge.
//!
//! # External-id requirement
//!
//! Every [`AclDocIngest`] carries a REQUIRED `external_id` (the
//! source system's stable native id, ADR-212 §D-2(a)
//! `source_native_id`): grant edges bind `from_external_id /
//! to_external_id` through the provider's idempotency table, and a
//! real connector always has a native id. Principal provenance nodes
//! are idempotency-keyed as `_acl:principal:<ext_id>` so re-syncs
//! re-use the same node (cross-call dedup additionally short-circuits
//! through [`PermissionIndex::principal_node`]).
//!
//! # Stage-1 posture
//!
//! In-memory write-through index per the `permissions` module docs
//! (restart ⇒ UNCLASSIFIED ⇒ deny-all until re-ingest — fail-closed);
//! rebuild-from-graph + CDC-tail invalidation for non-ingest `_Acl*`
//! mutations is stage-2 scope (ADR-212 §D-7).

use std::collections::{BTreeMap, BTreeSet};

use arcgraph_core::{NodeId, TenantId};
use arcgraph_storage::permissions::{ACL_GRANT_TYPE, ACL_PRINCIPAL_LABEL, PermissionIndex};

use crate::error::MCPError;
use crate::tools::ingest::{
    IngestBatch, IngestProvider, IngestRecordOutcome, NodeIngest, RelIngest,
};

/// Reserved `kind` property value for the [`PUBLIC`] marker's
/// provenance node (`_AclPrincipal {kind: "public"}`).
const PRINCIPAL_KIND_PUBLIC: &str = "public";

/// Reserved `kind` property value for ordinary subjects at stage-1
/// (per-connector identity mapping refines user/group/service at
/// stage-2+; un-expanded groups can only UNDER-grant — ADR-212 §D-1).
const PRINCIPAL_KIND_SUBJECT: &str = "subject";

/// One content document + its extracted read-ACL, submitted together
/// (the ADR-212 §D-3 "content + ACL alongside" contract).
#[derive(Debug, Clone, PartialEq)]
pub struct AclDocIngest {
    /// REQUIRED source-native stable id (idempotency key + grant-edge
    /// anchor). See module docs §"External-id requirement".
    pub external_id: String,
    /// Content node label (e.g. `"Document"`).
    pub label: String,
    /// Content property bag (same shape as [`NodeIngest::properties`]).
    pub properties: BTreeMap<String, serde_json::Value>,
    /// The extracted read grant list:
    /// - `None` — the source supplied NO ACL ⇒ the doc stays
    ///   UNCLASSIFIED ⇒ invisible to every principal (fail-closed);
    /// - `Some(vec![])` — explicitly granted to NOBODY;
    /// - `Some([...])` — principal external ids; may contain
    ///   [`arcgraph_storage::permissions::PUBLIC_PRINCIPAL`].
    pub read_principals: Option<Vec<String>>,
}

/// Summary of one [`ingest_docs_with_acls`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AclIngestSummary {
    /// `(doc external_id, committed internal node id)` for every doc
    /// whose CONTENT record committed (insert or idempotent re-use).
    pub committed_docs: Vec<(String, u64)>,
    /// Docs whose content record FAILED (no ACL entry was applied).
    pub failed_docs: Vec<String>,
    /// Docs that committed AND got an enforcement-index entry.
    pub tagged: usize,
    /// Docs that committed WITHOUT an ACL (UNCLASSIFIED — invisible
    /// under principal-scoped enforcement).
    pub unclassified: usize,
    /// Distinct `_AclPrincipal` provenance nodes written or re-used
    /// by THIS call.
    pub principal_nodes: usize,
    /// `_ACL_GRANT` provenance edges submitted by this call.
    pub grant_edges: usize,
    /// High-watermark commit LSN across the content + provenance
    /// batches (`None` when nothing committed).
    pub commit_lsn: Option<u64>,
}

/// Ingest `docs` (content + ACL together) under `tenant` through
/// `provider`, writing `_Acl*` provenance graph data and the
/// [`PermissionIndex`] write-through in the same flow. See module
/// docs for ordering + fail-closed semantics.
///
/// # Errors
///
/// Propagates request-level [`MCPError`]s from the provider
/// (tenant-unknown, storage-down). Per-RECORD faults do not error the
/// call; they surface in [`AclIngestSummary::failed_docs`] and the
/// affected docs get no grants (fail-closed).
pub fn ingest_docs_with_acls<P: IngestProvider + ?Sized>(
    provider: &P,
    permissions: &PermissionIndex,
    tenant: TenantId,
    docs: Vec<AclDocIngest>,
) -> Result<AclIngestSummary, MCPError> {
    // ── Batch 1: content nodes ──────────────────────────────────────
    let content_batch = IngestBatch {
        nodes: docs
            .iter()
            .map(|d| NodeIngest {
                external_id: Some(d.external_id.clone()),
                label: d.label.clone(),
                properties: d.properties.clone(),
            })
            .collect(),
        relationships: vec![],
        acl_grants: vec![],
    };
    let content_summary = provider.ingest(tenant, content_batch)?;

    let mut committed_docs: Vec<(String, u64)> = Vec::new();
    let mut failed_docs: Vec<String> = Vec::new();
    // records[i] corresponds to docs[i] (provider contract: one
    // outcome per submitted node, in submission order).
    let mut doc_ids: Vec<Option<u64>> = Vec::with_capacity(docs.len());
    for (doc, outcome) in docs.iter().zip(content_summary.records.iter()) {
        match outcome {
            IngestRecordOutcome::Inserted { internal_id, .. }
            | IngestRecordOutcome::Idempotent { internal_id, .. } => {
                committed_docs.push((doc.external_id.clone(), *internal_id));
                doc_ids.push(Some(*internal_id));
            }
            IngestRecordOutcome::Failed { .. } => {
                failed_docs.push(doc.external_id.clone());
                doc_ids.push(None);
            }
        }
    }

    // ── Batch 2: `_Acl*` provenance (principals + grant edges) ──────
    // Distinct grant subjects across every committed, tagged doc.
    let mut subjects: BTreeSet<&str> = BTreeSet::new();
    for (doc, id) in docs.iter().zip(doc_ids.iter()) {
        if id.is_some() {
            if let Some(grants) = &doc.read_principals {
                subjects.extend(grants.iter().map(String::as_str));
            }
        }
    }

    let mut principal_nodes: Vec<NodeIngest> = Vec::new();
    let mut new_subjects: Vec<&str> = Vec::new();
    for subject in &subjects {
        // Cross-call dedup: a subject already recorded this process
        // lifetime re-uses its provenance node (the provider-side
        // idempotency key would catch it too; the short-circuit skips
        // the resubmission entirely).
        if permissions
            .try_principal_node(subject)
            .map_err(|error| {
                MCPError::InternalError(format!(
                    "ACL principal binding lookup failed closed: {error}"
                ))
            })?
            .is_some()
        {
            continue;
        }
        let kind = if *subject == arcgraph_storage::permissions::PUBLIC_PRINCIPAL {
            PRINCIPAL_KIND_PUBLIC
        } else {
            PRINCIPAL_KIND_SUBJECT
        };
        principal_nodes.push(NodeIngest {
            external_id: Some(principal_external_id(subject)),
            label: ACL_PRINCIPAL_LABEL.to_owned(),
            properties: BTreeMap::from([
                (
                    "ext_id".to_owned(),
                    serde_json::Value::String((*subject).to_owned()),
                ),
                (
                    "kind".to_owned(),
                    serde_json::Value::String(kind.to_owned()),
                ),
            ]),
        });
        new_subjects.push(subject);
    }

    let mut grant_edges: Vec<RelIngest> = Vec::new();
    for (doc, id) in docs.iter().zip(doc_ids.iter()) {
        if id.is_none() {
            continue;
        }
        if let Some(grants) = &doc.read_principals {
            for subject in grants {
                grant_edges.push(RelIngest {
                    external_id: Some(format!(
                        "_acl:grant:{subject}:{doc_ext}",
                        doc_ext = doc.external_id
                    )),
                    from_external_id: principal_external_id(subject),
                    to_external_id: doc.external_id.clone(),
                    rel_type: ACL_GRANT_TYPE.to_owned(),
                    properties: BTreeMap::from([(
                        "access".to_owned(),
                        serde_json::Value::String("read".to_owned()),
                    )]),
                });
            }
        }
    }

    let grant_edge_count = grant_edges.len();
    let provenance_summary = if principal_nodes.is_empty() && grant_edges.is_empty() {
        None
    } else {
        Some(provider.ingest(
            tenant,
            IngestBatch {
                nodes: principal_nodes,
                relationships: grant_edges,
                acl_grants: vec![],
            },
        )?)
    };

    // Per-RECORD provenance failures fail the WHOLE call BEFORE the
    // index write-through (module docs §ordering): a tolerated failed
    // `_AclPrincipal` node or `_ACL_GRANT` edge would leave the
    // enforcement index granting access the provenance graph cannot
    // explain — breaking the "index derivable from committed graph
    // state" invariant in the audit dimension. Failing here is
    // fail-closed (every doc in this call stays UNCLASSIFIED ⇒
    // invisible) and the idempotency keys make a clean re-sync
    // converge.
    if let Some(summary) = &provenance_summary {
        if summary.failed_count > 0 {
            let detail: Vec<String> = summary
                .records
                .iter()
                .filter_map(|r| match r {
                    IngestRecordOutcome::Failed { external_id, error } => Some(format!(
                        "{}: {error}",
                        external_id.as_deref().unwrap_or("<no external_id>")
                    )),
                    _ => None,
                })
                .collect();
            return Err(MCPError::ExecutionEval(format!(
                "acl_ingest: {} _Acl* provenance record(s) failed; refusing index \
                 write-through (fail-closed — re-sync to converge): {}",
                summary.failed_count,
                detail.join("; ")
            )));
        }
    }

    // Record the provenance node ids for cross-call dedup. The
    // provenance batch lists the NEW principal nodes first, in
    // submission order (provider contract), then the edges.
    if let Some(summary) = &provenance_summary {
        for (subject, outcome) in new_subjects.iter().zip(summary.records.iter()) {
            if let IngestRecordOutcome::Inserted { internal_id, .. }
            | IngestRecordOutcome::Idempotent { internal_id, .. } = outcome
            {
                permissions.record_principal_node(subject, NodeId::new(*internal_id));
            }
        }
    }

    // ── Write-through: enforcement index LAST (fail-closed; module
    // docs §ordering) ────────────────────────────────────────────────
    let mut tagged = 0usize;
    let mut unclassified = 0usize;
    for (doc, id) in docs.iter().zip(doc_ids.iter()) {
        let Some(internal) = id else { continue };
        match &doc.read_principals {
            Some(grants) => {
                let set: BTreeSet<String> = grants.iter().cloned().collect();
                permissions
                    .apply_doc_acl_checked(NodeId::new(*internal), set)
                    .map_err(|error| {
                        MCPError::InternalError(format!("ACL owner publish failed: {error}"))
                    })?;
                tagged += 1;
            }
            None => unclassified += 1,
        }
    }

    let commit_lsn = match (
        content_summary.commit_lsn,
        provenance_summary.as_ref().and_then(|s| s.commit_lsn),
    ) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    };

    Ok(AclIngestSummary {
        committed_docs,
        failed_docs,
        tagged,
        unclassified,
        principal_nodes: subjects.len(),
        grant_edges: grant_edge_count,
        commit_lsn,
    })
}

/// Idempotency key for a grant subject's `_AclPrincipal` provenance
/// node — namespaced so it can never collide with a content
/// external_id.
fn principal_external_id(subject: &str) -> String {
    format!("_acl:principal:{subject}")
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use arcgraph_storage::permissions::PUBLIC_PRINCIPAL;

    use super::*;
    use crate::tools::ingest::{IngestError, IngestSummary};

    /// Stub provider: assigns sequential ids, records every batch,
    /// optionally fails specific node external_ids or whole calls.
    #[derive(Debug, Default)]
    struct StubProvider {
        batches: Mutex<Vec<IngestBatch>>,
        next_id: Mutex<u64>,
        fail_node_ext_ids: Vec<String>,
        fail_rel_ext_ids: Vec<String>,
        fail_call_index: Option<usize>,
        calls: Mutex<usize>,
    }

    impl IngestProvider for StubProvider {
        fn ingest(&self, _tenant: TenantId, batch: IngestBatch) -> Result<IngestSummary, MCPError> {
            let call_idx = {
                let mut c = self.calls.lock().expect("calls");
                let i = *c;
                *c += 1;
                i
            };
            if self.fail_call_index == Some(call_idx) {
                return Err(MCPError::ExecutionEval("injected provider fault".into()));
            }
            let mut records = Vec::new();
            let mut inserted = 0u64;
            let mut failed = 0u64;
            {
                let mut next = self.next_id.lock().expect("next_id");
                for n in &batch.nodes {
                    let ext = n.external_id.clone();
                    if ext
                        .as_deref()
                        .is_some_and(|e| self.fail_node_ext_ids.iter().any(|f| f == e))
                    {
                        failed += 1;
                        records.push(IngestRecordOutcome::Failed {
                            external_id: ext,
                            error: IngestError::Invalid {
                                detail: "injected per-record fault".into(),
                            },
                        });
                        continue;
                    }
                    *next += 1;
                    inserted += 1;
                    records.push(IngestRecordOutcome::Inserted {
                        internal_id: *next,
                        external_id: ext,
                    });
                }
                for r in &batch.relationships {
                    let ext = r.external_id.clone();
                    if ext
                        .as_deref()
                        .is_some_and(|e| self.fail_rel_ext_ids.iter().any(|f| f == e))
                    {
                        failed += 1;
                        records.push(IngestRecordOutcome::Failed {
                            external_id: ext,
                            error: IngestError::Storage {
                                detail: "injected per-record edge fault".into(),
                            },
                        });
                        continue;
                    }
                    *next += 1;
                    inserted += 1;
                    records.push(IngestRecordOutcome::Inserted {
                        internal_id: *next,
                        external_id: ext,
                    });
                }
            }
            self.batches.lock().expect("batches").push(batch);
            Ok(IngestSummary {
                records,
                inserted_count: inserted,
                failed_count: failed,
                commit_lsn: Some(100 + inserted),
                dropped_acl_grants: Vec::new(),
            })
        }
    }

    fn doc(ext: &str, grants: Option<&[&str]>) -> AclDocIngest {
        AclDocIngest {
            external_id: ext.into(),
            label: "Document".into(),
            properties: BTreeMap::from([(
                "body".to_owned(),
                serde_json::Value::String(format!("body of {ext}")),
            )]),
            read_principals: grants.map(|g| g.iter().map(|s| (*s).to_owned()).collect()),
        }
    }

    #[test]
    fn content_provenance_and_index_land_in_one_flow() {
        let provider = StubProvider::default();
        let perms = PermissionIndex::new();
        let summary = ingest_docs_with_acls(
            &provider,
            &perms,
            TenantId::new(7),
            vec![
                doc("d1", Some(&["alice"])),
                doc("d2", Some(&["bob", PUBLIC_PRINCIPAL])),
                doc("d3", None), // UNCLASSIFIED
            ],
        )
        .expect("ingest ok");

        assert_eq!(summary.committed_docs.len(), 3);
        assert_eq!(summary.tagged, 2);
        assert_eq!(summary.unclassified, 1);
        assert_eq!(summary.principal_nodes, 3, "alice, bob, __public__");
        assert_eq!(summary.grant_edges, 3, "d1←alice, d2←bob, d2←public");
        assert!(summary.commit_lsn.is_some());

        // Enforcement plane is live for the same flow.
        let d1 = summary.committed_docs[0].1;
        let d2 = summary.committed_docs[1].1;
        let d3 = summary.committed_docs[2].1;
        assert!(perms.effective("alice").is_visible(NodeId::new(d1)));
        assert!(!perms.effective("bob").is_visible(NodeId::new(d1)));
        assert!(perms.effective("bob").is_visible(NodeId::new(d2)));
        assert!(
            perms.effective("alice").is_visible(NodeId::new(d2)),
            "PUBLIC grant reaches every principal"
        );
        assert!(
            !perms.effective("alice").is_visible(NodeId::new(d3)),
            "no ACL ⇒ UNCLASSIFIED ⇒ invisible"
        );

        // Provenance batch shape: 3 principal nodes + 3 grant edges,
        // labels/types from the storage-side reserved constants.
        let batches = provider.batches.lock().expect("batches");
        assert_eq!(batches.len(), 2, "content batch + provenance batch");
        assert!(
            batches[1]
                .nodes
                .iter()
                .all(|n| n.label == ACL_PRINCIPAL_LABEL)
        );
        assert!(
            batches[1]
                .relationships
                .iter()
                .all(|r| r.rel_type == ACL_GRANT_TYPE)
        );
        // Grant edges anchor on the namespaced principal key + the
        // doc's own external id.
        assert!(
            batches[1]
                .relationships
                .iter()
                .any(|r| r.from_external_id == "_acl:principal:alice" && r.to_external_id == "d1")
        );
    }

    #[test]
    fn failed_content_record_gets_no_grants() {
        let provider = StubProvider {
            fail_node_ext_ids: vec!["d1".into()],
            ..Default::default()
        };
        let perms = PermissionIndex::new();
        let summary = ingest_docs_with_acls(
            &provider,
            &perms,
            TenantId::new(7),
            vec![doc("d1", Some(&["alice"])), doc("d2", Some(&["bob"]))],
        )
        .expect("call ok despite per-record fault");

        assert_eq!(summary.failed_docs, vec!["d1".to_owned()]);
        assert_eq!(summary.tagged, 1);
        // d1 never reached the index (fail-closed: nothing to widen),
        // and no grant edge references it.
        assert_eq!(perms.tagged_docs(), 1);
        let batches = provider.batches.lock().expect("batches");
        assert!(
            batches[1]
                .relationships
                .iter()
                .all(|r| r.to_external_id != "d1"),
            "no provenance edge for a doc that did not commit"
        );
    }

    #[test]
    fn provenance_batch_failure_leaves_docs_unclassified() {
        // Fail the SECOND provider call (the provenance batch): the
        // write-through must NOT have happened — committed content
        // stays UNCLASSIFIED (invisible), never enforcement-tagged
        // ahead of its provenance (module docs §ordering).
        let provider = StubProvider {
            fail_call_index: Some(1),
            ..Default::default()
        };
        let perms = PermissionIndex::new();
        let err = ingest_docs_with_acls(
            &provider,
            &perms,
            TenantId::new(7),
            vec![doc("d1", Some(&["alice"]))],
        )
        .expect_err("provenance fault propagates");
        assert_eq!(
            err.code(),
            crate::error::MCPError::ExecutionEval(String::new()).code()
        );
        assert_eq!(
            perms.tagged_docs(),
            0,
            "index untouched ⇒ deny-all (narrow)"
        );
        assert!(!perms.effective("alice").is_visible(NodeId::new(1)));
    }

    #[test]
    fn per_record_provenance_failure_refuses_index_write_through() {
        // A failed `_ACL_GRANT` edge inside an otherwise-Ok provenance
        // batch must fail the CALL before the write-through: the index
        // must never grant access the provenance graph cannot explain
        // (module docs §ordering, audit-dimension invariant).
        let provider = StubProvider {
            fail_rel_ext_ids: vec!["_acl:grant:alice:d1".into()],
            ..Default::default()
        };
        let perms = PermissionIndex::new();
        let err = ingest_docs_with_acls(
            &provider,
            &perms,
            TenantId::new(7),
            vec![doc("d1", Some(&["alice"])), doc("d2", Some(&["bob"]))],
        )
        .expect_err("per-record provenance fault fails the call");
        let rendered = format!("{err:?}");
        assert!(
            rendered.contains("_acl:grant:alice:d1"),
            "error names the failed provenance record: {rendered}"
        );
        // NO doc got an enforcement entry — not even d2, whose own
        // records succeeded (whole-call fail-closed; re-sync converges
        // via the idempotency keys).
        assert_eq!(perms.tagged_docs(), 0);
        assert!(!perms.effective("alice").is_visible(NodeId::new(1)));
        assert!(!perms.effective("bob").is_visible(NodeId::new(2)));
    }

    #[test]
    fn principal_nodes_dedupe_within_and_across_calls() {
        let provider = StubProvider::default();
        let perms = PermissionIndex::new();
        // alice grants two docs in one call: ONE principal node.
        ingest_docs_with_acls(
            &provider,
            &perms,
            TenantId::new(7),
            vec![doc("d1", Some(&["alice"])), doc("d2", Some(&["alice"]))],
        )
        .expect("ok");
        {
            let batches = provider.batches.lock().expect("batches");
            assert_eq!(batches[1].nodes.len(), 1, "within-call dedup");
        }
        assert!(
            perms.principal_node("alice").is_some(),
            "dedup map recorded"
        );

        // Second call re-using alice: NO new principal node submitted.
        ingest_docs_with_acls(
            &provider,
            &perms,
            TenantId::new(7),
            vec![doc("d3", Some(&["alice"]))],
        )
        .expect("ok");
        let batches = provider.batches.lock().expect("batches");
        assert_eq!(batches.len(), 4);
        assert!(
            batches[3].nodes.is_empty(),
            "cross-call dedup: alice's provenance node is re-used"
        );
        assert_eq!(
            batches[3].relationships.len(),
            1,
            "the new grant edge still lands"
        );
    }

    #[test]
    fn empty_grant_list_tags_doc_granted_to_nobody() {
        let provider = StubProvider::default();
        let perms = PermissionIndex::new();
        let summary = ingest_docs_with_acls(
            &provider,
            &perms,
            TenantId::new(7),
            vec![doc("d1", Some(&[]))],
        )
        .expect("ok");
        assert_eq!(summary.tagged, 1);
        assert_eq!(summary.grant_edges, 0);
        let d1 = summary.committed_docs[0].1;
        assert!(!perms.effective("alice").is_visible(NodeId::new(d1)));
        assert!(
            !perms
                .effective(PUBLIC_PRINCIPAL)
                .is_visible(NodeId::new(d1))
        );
    }
}
