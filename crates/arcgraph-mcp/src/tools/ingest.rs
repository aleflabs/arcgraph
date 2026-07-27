//! W14γ M5-08 — `graph.ingest` Tier-1 MCP write-side tool.
//!
//! First WRITE-side surface in the ADR-004 Tier-1 catalog. Accepts a
//! batch of node + relationship records and routes each through the
//! [`IngestProvider`] adapter trait, returning a per-record success /
//! failure summary.
//!
//! The v1.0-alpha wire shape (`IngestRequest` carrying `tenant_id` +
//! `nodes` + `relationships` + optional `format`) is canonical per
//! ADR-004 amendment-01 (`docs/adr/amendments/ADR-004-amendment-01-graph-ingest-v1-surface.md`),
//! which supersedes ADR-004 D-1's original `(document, schema_hint)`
//! shape.
//!
//! # Per-record durability (ADR-031 commitment)
//!
//! Per ADR-031 §Decision (single-fire `CommitBundle` per commit +
//! group-commit window), every write goes through full WAL + group
//! commit. The MCP layer does NOT take a "batched-without-fsync"
//! shortcut — each record's WAL append participates in the standard
//! group-commit fsync cohort; the implementer-side
//! [`IngestProvider::ingest`] body MUST honor this contract (the MCP
//! tool surface only verifies it indirectly, via the round-trip tests
//! that prove `graph.ingest` → `graph.inspect` observes the write).
//!
//! # Per-record idempotency
//!
//! Each [`NodeIngest`] / [`RelIngest`] record may carry an
//! `external_id` (`Option<String>`). When `Some`, the implementer
//! SHOULD treat `(tenant_id, external_id)` as the idempotency key:
//! a re-submission of the same external_id with the same payload
//! returns the original [`IngestRecordOutcome::Inserted`] result with
//! the same internal node/rel id; conflicting payloads surface as
//! [`IngestError::IdempotencyConflict`].
//!
//! # Cross-MCP-call reads-after-write
//!
//! Per ADR-038 amendment-03 §TIER-1 GAP E rule 1 (snapshot LSN
//! acquired at execute-time, before the first operator pulls a batch)
//! plus LSN monotonicity (the per-tenant LSN clock advances on every
//! commit and never rewinds), an MCP session that observes a
//! successful `graph.ingest` return — whose [`IngestSummary`] carries
//! `commit_lsn = Some(L)` — is guaranteed that the next read tool
//! (`graph.inspect`, `graph.search`, etc.) acquires a snapshot LSN
//! ≥ `L` and therefore observes the just-ingested records. The
//! ingest provider's contract surfaces `commit_lsn` so routers /
//! callers MAY pin subsequent reads to ≥ this value when an explicit
//! monotone-bind is required.
//!
//! # Cross-tenant guard
//!
//! Same shape as [`crate::tools::schema`] / [`crate::tools::inspect`]:
//! a request whose `tenant_id` differs from the session's bound
//! tenant rejects as [`MCPError::Unauthorized`] BEFORE any
//! [`IngestProvider`] call.
//!
//! # ADR provenance
//!
//! - **ADR-004 amendment-01** — v1.0-alpha wire shape (structured
//!   records, not document-text + entity resolution).
//! - **ADR-004 §"Tier 1 (agent-facing, default)"** — `graph.ingest()`
//!   is the third Tier-1 tool to ship.
//! - **ADR-038 amendment-03 §TIER-1 GAP A** — pinned MCP `graph.ingest`
//!   as the v1.0 data-modification surface ("ArcQL v1.0 admits only
//!   read clauses").
//! - **ADR-038 amendment-03 §TIER-1 GAP E rule 1** — snapshot LSN
//!   acquired at execute-time; combined with LSN monotonicity, this
//!   is the canonical reads-after-write contract across MCP calls.
//! - **ADR-031 §Decision** — single-`CommitBundle`-per-commit +
//!   group-commit window; durability contract every record
//!   participates in.
//! - **ADR-037 D-1** — per-tenant routing inherited via the
//!   cross-tenant guard.

use std::collections::BTreeMap;

use arcgraph_core::TenantId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::error::MCPError;
use crate::tools::ResponseFormat;

// ─────────────────────────────────────────────────────────────────────
// IngestError — codec-local error taxonomy
// ─────────────────────────────────────────────────────────────────────

/// Per-record fault surface returned inside [`IngestRecordOutcome`].
///
/// `#[non_exhaustive]` under the strict public-contract policy: production wiring at
/// M4-08+ may add storage-side variants (e.g., `LabelUnknown`,
/// `RelTypeUnknown`, `BlobOversize`). The current variant set covers
/// the v1.0-alpha stub's needs; downstream pattern-matchers MUST keep
/// a wildcard arm.
#[derive(Debug, Clone, Error, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IngestError {
    /// Record's `external_id` matched a previously-committed record
    /// AND the payload differs. v1.0-alpha implementers MUST surface
    /// this when a re-submission attempts to mutate.
    #[error("idempotency conflict on external_id={external_id:?}")]
    IdempotencyConflict {
        /// The conflicting external_id.
        external_id: String,
    },
    /// Validation rejected the record (e.g., empty label, malformed
    /// property bag, oversized blob payload).
    #[error("invalid record: {detail}")]
    Invalid {
        /// Human-readable detail. The MCP wire surface renders this
        /// inside the per-record outcome envelope.
        detail: String,
    },
    /// Implementer storage call failed (WAL append, page allocator,
    /// substrate I/O). Routed to MCP error code [`crate::CODE_EXECUTION_EVAL`]
    /// at the request envelope level when ALL records fail this way.
    #[error("storage fault: {detail}")]
    Storage {
        /// Implementer-rendered detail.
        detail: String,
    },
    /// Endpoint refused under per-tenant rate-limit (forward path
    /// when an implementer chooses to surface back-pressure on a
    /// per-record basis rather than rejecting the entire batch).
    #[error("rate limited; retry after {retry_after_ms}ms")]
    RateLimited {
        /// Hint to the caller for back-off duration in milliseconds.
        retry_after_ms: u64,
    },
    /// The per-tenant idempotency cache is at capacity and cannot record
    /// a new `external_id → internal_id` binding. Surfaced loudly (never
    /// silently dropped) so the caller sees accurate back-pressure
    /// instead of a divergent duplicate node or a falsely-unresolved edge
    /// (issue #352). The binding map is in-memory and **non-evicting** at
    /// v1.0-alpha — existing bindings and re-ingests of already-known
    /// external_ids still resolve; only NEW distinct external_ids beyond
    /// the per-tenant cap are refused. v1.1 (#352 Part 2) WAL-persists
    /// the binding and removes the cap.
    #[error("idempotency capacity exceeded: {detail}")]
    CapacityExceeded {
        /// Human-readable detail (current cap + remediation).
        detail: String,
    },
}

// ─────────────────────────────────────────────────────────────────────
// Adapter trait
// ─────────────────────────────────────────────────────────────────────

/// Adapter trait read by the [`ingest_tool`] entry point.
///
/// Implementations live OUTSIDE this crate: tests stub it in-line;
/// production wiring at M4-08+ implements it on the storage tenant
/// handle (the `MultiTenantRouter::tenant_handle` surface), composing
/// CRUD-store writes + WAL group commit per ADR-031.
///
/// # Hard contracts (implementor MUST honor)
///
/// 1. **Per-record durability.** Each successful record's bytes MUST
///    have hit `fsync` before [`IngestProvider::ingest`] returns. Per
///    ADR-031 §Decision (group-commit window aggregates concurrent
///    commits into one fsync), multiple records in the same call MAY
///    share an fsync cohort (i.e., one fsync per batch is allowed),
///    but a successful return MUST imply a successful sync.
/// 2. **Idempotency.** When a record carries `external_id = Some(s)`,
///    a re-submission of `(tenant_id, s)` with the SAME payload MUST
///    succeed and return the SAME internal id. A re-submission with a
///    DIFFERENT payload MUST surface [`IngestError::IdempotencyConflict`].
///    Records with `external_id = None` are treated as fresh inserts;
///    de-dupe is the caller's responsibility.
/// 3. **Snapshot-isolation reads-after-write.** The returned
///    [`IngestSummary::commit_lsn`] MUST be ≤ the LSN observed by any
///    subsequent same-session read tool (`graph.inspect`,
///    `graph.search`, etc.). v1.0-alpha implementers can satisfy this
///    by acquiring a single LSN per group-commit cohort and returning
///    the cohort's high-watermark.
/// 4. **Tenant scoping.** Records MUST land under `tenant`; an
///    implementer that observes a cross-tenant write MUST surface
///    [`MCPError::Unauthorized`] (defense-in-depth — the dispatcher's
///    cross-tenant guard already rejects this case before the
///    inspector is called, but the trait body owns the second-layer
///    check per the same M5-05 hard requirement).
/// 5. **Cancellation.** v1.0-alpha implementers MAY ignore
///    cancellation; the W12γ `CancellationRegistry` integration
///    (token-check at record boundary) lands at M4-08+ wiring. A
///    forward-deferred cancellation surface MUST yield
///    [`IngestError::Storage`] with detail "cancelled" rather than
///    silently dropping records.
///
/// # `Send + Sync`
///
/// MCP transport runs on tokio; the inspector must be shareable
/// across awaits.
pub trait IngestProvider: Send + Sync {
    /// Ingest a batch of records under `tenant`.
    ///
    /// Returns a per-record summary. Errors that prevent ANY record
    /// from being processed (e.g., tenant unknown, request-level
    /// rate-limit) surface as `Err(MCPError)`; per-record faults
    /// (validation, idempotency conflict, single-record storage
    /// failure) surface inside [`IngestSummary::records`] as
    /// [`IngestRecordOutcome::Failed`].
    fn ingest(&self, tenant: TenantId, batch: IngestBatch) -> Result<IngestSummary, MCPError>;
}

// ─────────────────────────────────────────────────────────────────────
// Request / response envelopes
// ─────────────────────────────────────────────────────────────────────

/// The body of an `IngestProvider::ingest` call. Constructed from the
/// validated [`IngestRequest`] and passed in unchanged so the
/// implementer sees the same structural shape the wire spoke.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct IngestBatch {
    /// Node records to insert / upsert (idempotency-keyed).
    pub nodes: Vec<NodeIngest>,
    /// Relationship records to insert. Relationships reference nodes
    /// by `from_external_id` / `to_external_id`; if either reference
    /// targets a node in the same batch, the implementer is
    /// responsible for ordering (typically: nodes first, then rels).
    pub relationships: Vec<RelIngest>,
    /// Optional per-doc read-ACL grants, applied via
    /// [`arcgraph_storage::permissions::PermissionIndex::apply_doc_acl`]
    /// AFTER the records commit (ADR-212 §D-4 Seam-1, MUST-CON-07
    /// #1181). An implementer with no `PermissionIndex`
    /// (the in-memory test stubs) ignores this field; the
    /// storage-backed [`crate::storage::StorageIngestProvider`] honors
    /// it. `#[serde(default)]` so it threads through unchanged and
    /// every existing caller / wire payload is unaffected (omitted ⇒
    /// empty ⇒ no ACL applied, behaves exactly as before).
    #[serde(default)]
    pub acl_grants: Vec<AclGrant>,
}

/// One content document's read-ACL grant, carried alongside the
/// records on the LIVE `graph.ingest` push path (#1181, MUST-CON-07).
///
/// Carries a content `external_id` plus an optional read-grant list. The
/// storage-backed [`IngestProvider`] resolves the `external_id` to the
/// node id it just committed in the SAME call, then writes the grant
/// set through
/// [`arcgraph_storage::permissions::PermissionIndex::apply_doc_acl`]
/// (the enforcement plane `graph.search` reads, ADR-212 §D-4 Seam-1).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AclGrant {
    /// The content node's `external_id` (e.g. `"doc:007"`) — resolved
    /// to the internal node id committed in this same ingest call. A
    /// grant whose `external_id` did NOT commit (failed record, or
    /// absent from `nodes`) is skipped (fail-closed: the doc stays
    /// UNCLASSIFIED ⇒ invisible under enforcement).
    pub external_id: String,
    /// Read grant list. `null`/absent ⇒ the doc stays UNCLASSIFIED
    /// (skipped — do NOT call `apply_doc_acl`; fail-closed, invisible
    /// under enforcement); `[]` ⇒ an explicit grant-to-nobody;
    /// otherwise principal external ids (may contain
    /// [`arcgraph_storage::permissions::PUBLIC_PRINCIPAL`]).
    #[serde(default)]
    pub read_principals: Option<Vec<String>>,
}

/// A node record submitted for ingestion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NodeIngest {
    /// Optional client-provided id used as the idempotency key. When
    /// `Some`, the implementer MUST de-dupe on `(tenant_id, external_id)`.
    #[serde(default)]
    pub external_id: Option<String>,
    /// Node label (single-label per ADR-038 §2 D-1 v1.0 grammar).
    pub label: String,
    /// Property bag — keyed by property name; values rendered as
    /// `serde_json::Value` so the implementer can route them to the
    /// existing `arcgraph_storage::property` encoder. `BTreeMap` for
    /// deterministic test diffs.
    #[serde(default)]
    pub properties: BTreeMap<String, serde_json::Value>,
}

/// A relationship record submitted for ingestion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RelIngest {
    /// Optional idempotency key. Same semantics as
    /// [`NodeIngest::external_id`].
    #[serde(default)]
    pub external_id: Option<String>,
    /// `from`-side node external id. The implementer resolves this
    /// to the internal node id via the same idempotency table.
    pub from_external_id: String,
    /// `to`-side node external id.
    pub to_external_id: String,
    /// Relationship type (single-type per ADR-038 §2 D-1).
    pub rel_type: String,
    /// Property bag.
    #[serde(default)]
    pub properties: BTreeMap<String, serde_json::Value>,
}

/// One read-ACL grant that was DROPPED (skipped) during ingest because
/// its content `external_id` did not resolve to a node committed in the
/// same call — surfaced (never silently swallowed) so the caller can
/// detect that a security-relevant grant did NOT apply (issue #1198,
/// MUST-CON-07 hardening).
///
/// The skip itself stays fail-closed (an unresolved `external_id` leaves
/// the doc UNCLASSIFIED ⇒ invisible under enforcement — widening on an
/// unresolved grant would be strictly worse); the only change is making
/// the drop VISIBLE in the response. See [`IngestSummary::dropped_acl_grants`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DroppedAclGrant {
    /// The grant's content `external_id` that did not resolve to a
    /// committed node (e.g. a typo, or the referenced node's insert
    /// failed).
    pub external_id: String,
    /// Why the grant was dropped — `"unresolved"` (the `external_id`
    /// was never submitted, or its node record failed to commit). A
    /// stable string discriminant so callers can branch / alert without
    /// pattern-matching a closed enum.
    pub reason: String,
}

impl DroppedAclGrant {
    /// Reason discriminant for a grant whose `external_id` did not
    /// resolve to a committed node in the same call.
    pub const REASON_UNRESOLVED: &'static str = "unresolved";

    /// Construct a dropped-grant record with the canonical
    /// [`Self::REASON_UNRESOLVED`] reason.
    pub fn unresolved(external_id: impl Into<String>) -> Self {
        Self {
            external_id: external_id.into(),
            reason: Self::REASON_UNRESOLVED.to_owned(),
        }
    }
}

/// Summary returned by [`IngestProvider::ingest`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IngestSummary {
    /// Per-record outcomes — one entry per submitted node, in
    /// submission order, then one entry per relationship.
    pub records: Vec<IngestRecordOutcome>,
    /// Number of records that committed successfully.
    pub inserted_count: u64,
    /// Number of records that failed (validation / idempotency /
    /// storage). Equals `records.len() - inserted_count`.
    pub failed_count: u64,
    /// LSN at which the cohort committed. `None` when no record
    /// committed (every record failed). Forward consumers (router,
    /// next read tool) MAY pin subsequent snapshot reads to ≥ this
    /// LSN to satisfy reads-after-write per the trait contract.
    pub commit_lsn: Option<u64>,
    /// Read-ACL grants that were DROPPED because their content
    /// `external_id` did not resolve to a node committed in this call
    /// (issue #1198, MUST-CON-07 hardening). EMPTY in the common case;
    /// non-empty surfaces a security-relevant skip the caller would
    /// otherwise NOT see (the pre-#1198 silent drop reported
    /// `failed_count:0` / full success while a grant was dropped).
    ///
    /// This is a DIFFERENT axis from `failed_count` (which counts failed
    /// NODE/REL inserts): a dropped grant rides on a SUCCESSFUL node, so
    /// surfacing it separately is cleaner than overloading
    /// `failed_count`. The skip behavior is unchanged (fail-closed); only
    /// the SILENCE is removed.
    ///
    /// `#[serde(default, skip_serializing_if = "Vec::is_empty")]` keeps
    /// the wire shape backward-compatible: ABSENT when no grant dropped
    /// (every existing caller / pinned shape is unaffected), present only
    /// when there is a drop to surface.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped_acl_grants: Vec<DroppedAclGrant>,
}

/// Per-record outcome inside [`IngestSummary::records`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum IngestRecordOutcome {
    /// Record committed to storage. Internal id assigned by the
    /// implementer (node id for [`NodeIngest`], rel id for
    /// [`RelIngest`]).
    Inserted {
        /// Internal id assigned by the implementer.
        internal_id: u64,
        /// Optional echo of the client's external_id.
        external_id: Option<String>,
    },
    /// Idempotent re-submission: a previous call inserted the same
    /// `(tenant_id, external_id)` with the same payload, and this
    /// call returned the original internal id. Distinct from
    /// `Inserted` so callers can observe back-pressure-friendly
    /// retry behavior.
    Idempotent {
        /// Original internal id.
        internal_id: u64,
        /// External id (always Some in this branch; the implementer
        /// can only short-circuit when external_id is provided).
        external_id: String,
    },
    /// Record failed; carries the [`IngestError`] detail.
    Failed {
        /// External id if the client provided one, for client-side
        /// correlation.
        external_id: Option<String>,
        /// The fault.
        error: IngestError,
    },
}

/// Request params for the `graph.ingest` tool.
///
/// `#[serde(deny_unknown_fields)]` under the code-quality policy strict-mode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IngestRequest {
    /// The tenant under which the records land. Cross-tenant requests
    /// reject as [`MCPError::Unauthorized`] before any provider call.
    pub tenant_id: u64,
    /// Node records.
    #[serde(default)]
    pub nodes: Vec<NodeIngest>,
    /// Relationship records.
    #[serde(default)]
    pub relationships: Vec<RelIngest>,
    /// Optional per-doc read-ACL grants, applied via
    /// [`arcgraph_storage::permissions::PermissionIndex::apply_doc_acl`]
    /// after the records commit (ADR-212 §D-4 Seam-1; closes the LIVE half of
    /// MUST-CON-07 #1181). Omitted ⇒ empty ⇒ no ACL applied
    /// (backward-compatible). Threaded into [`IngestBatch::acl_grants`]
    /// so the storage-backed provider can perform the post-commit
    /// write-through without a trait-surface change.
    #[serde(default)]
    pub acl_grants: Option<Vec<AclGrant>>,
    /// Optional render-format hint. Defaults to JSON (per-record
    /// outcomes are heterogeneous; YAML / TOON pivot through Value).
    #[serde(default)]
    pub format: Option<ResponseFormat>,
}

// ─────────────────────────────────────────────────────────────────────
// Tool entry point
// ─────────────────────────────────────────────────────────────────────

/// `graph.ingest` — write-side Tier-1 tool entry point.
///
/// # Cross-tenant guard
///
/// Same shape as [`crate::tools::schema::schema_tool`] /
/// [`crate::tools::inspect::inspect_tool`].
///
/// # Errors
///
/// - [`MCPError::Unauthorized`] — cross-tenant request.
/// - [`MCPError::TenantUnknown`] — provider has no binding for the
///   tenant.
/// - [`MCPError::ExecutionEval`] — substrate fault during the call.
/// - [`MCPError::InternalError`] — serializer encode failure.
///
/// Per-record faults (validation, idempotency, single-record storage)
/// surface inside the returned [`IngestSummary::records`] as
/// [`IngestRecordOutcome::Failed`] — the request-level call still
/// returns `Ok`.
pub fn ingest_tool<P: IngestProvider>(
    provider: &P,
    session_tenant: TenantId,
    req: IngestRequest,
) -> Result<serde_json::Value, MCPError> {
    let request_tenant = TenantId::new(req.tenant_id);
    if request_tenant != session_tenant {
        return Err(MCPError::Unauthorized);
    }
    let batch = IngestBatch {
        nodes: req.nodes,
        relationships: req.relationships,
        // Thread the optional ACL grants through to the provider so the
        // storage-backed [`IngestProvider`] can perform the post-commit
        // `apply_doc_acl` write-through (#1181, MUST-CON-07).
        // `None` ⇒ empty ⇒
        // no ACL applied (backward-compatible).
        acl_grants: req.acl_grants.unwrap_or_default(),
    };
    let summary = provider.ingest(request_tenant, batch)?;
    let format = req.format.unwrap_or(ResponseFormat::Json);
    let value = serde_json::to_value(&summary)
        .map_err(|e| MCPError::InternalError(format!("ingest summary serialize: {e}")))?;
    crate::tools::render_response(format, &value)
}

// ─────────────────────────────────────────────────────────────────────
// Tests — ≥10 unit tests per spawn prompt §Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Stub IngestProvider — keeps an in-memory log of records, simulates
    /// per-record idempotency on `external_id`, and exposes hooks for
    /// the WAL-emission test.
    #[derive(Debug, Default)]
    struct StubIngestProvider {
        bound_tenant: Option<TenantId>,
        next_internal_id: Mutex<u64>,
        /// Per-tenant idempotency table: external_id → (internal_id, payload-hash).
        idem: Mutex<HashMap<(u64, String), (u64, u64)>>,
        /// Records observed (used by WAL-emission test).
        wal_log: Mutex<Vec<String>>,
        /// Number of fsync calls observed (one per ingest cohort per
        /// ADR-031 group-commit).
        fsyncs: Mutex<u64>,
        /// If `true`, every record fails with `Storage("disk full")` —
        /// used by the request-level failure test.
        fail_storage: bool,
        /// If `true`, return `MCPError::TenantUnknown` regardless of
        /// tenant — used by the tenant-routing test.
        force_tenant_unknown: bool,
        /// Cancellation flag — when set, the provider yields every
        /// record as `IngestError::Storage { detail: "cancelled" }`
        /// per the trait hard-contract 5 (W12γ-style cancellation
        /// surface; the W14γ slice forward-defers SIGTERM signal
        /// wiring to M4-08+ but pins the per-record fault shape).
        cancel_flag: Arc<std::sync::atomic::AtomicBool>,
    }

    impl StubIngestProvider {
        fn new(tenant: TenantId) -> Self {
            Self {
                bound_tenant: Some(tenant),
                next_internal_id: Mutex::new(1000),
                ..Default::default()
            }
        }
        fn alloc_id(&self) -> u64 {
            let mut g = self.next_internal_id.lock();
            let id = *g;
            *g += 1;
            id
        }
        fn payload_hash(properties: &BTreeMap<String, serde_json::Value>) -> u64 {
            // Deterministic hash so a re-submit of identical payload
            // matches; intentionally simple (DJB2-ish over the JSON
            // bytes) — production implementer uses the real WAL CRC.
            let s = serde_json::to_string(properties).unwrap_or_default();
            let mut h: u64 = 5381;
            for b in s.bytes() {
                h = h.wrapping_mul(33).wrapping_add(b as u64);
            }
            h
        }
    }

    impl IngestProvider for StubIngestProvider {
        fn ingest(&self, tenant: TenantId, batch: IngestBatch) -> Result<IngestSummary, MCPError> {
            if self.force_tenant_unknown {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            match self.bound_tenant {
                Some(t) if t == tenant => {}
                _ => return Err(MCPError::TenantUnknown(format!("{tenant:?}"))),
            }

            let mut records = Vec::new();
            let mut inserted = 0u64;
            let mut failed = 0u64;

            // Single fsync per cohort per ADR-031 group-commit.
            *self.fsyncs.lock() += 1;
            let mut commit_lsn: Option<u64> = None;

            for n in batch.nodes {
                // Per IngestProvider hard contract 5: cancellation
                // surfaces as `IngestError::Storage { detail: "cancelled" }`
                // per record, not a silent drop. Check the flag at
                // each record boundary (matching the W12γ
                // batch-boundary cancel-token discipline).
                if self.cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                    failed += 1;
                    records.push(IngestRecordOutcome::Failed {
                        external_id: n.external_id,
                        error: IngestError::Storage {
                            detail: "cancelled".into(),
                        },
                    });
                    continue;
                }
                if self.fail_storage {
                    failed += 1;
                    records.push(IngestRecordOutcome::Failed {
                        external_id: n.external_id,
                        error: IngestError::Storage {
                            detail: "disk full".into(),
                        },
                    });
                    continue;
                }
                if n.label.trim().is_empty() {
                    failed += 1;
                    records.push(IngestRecordOutcome::Failed {
                        external_id: n.external_id,
                        error: IngestError::Invalid {
                            detail: "empty label".into(),
                        },
                    });
                    continue;
                }
                let payload_hash = Self::payload_hash(&n.properties);
                if let Some(ext) = n.external_id.clone() {
                    let key = (tenant.raw(), ext.clone());
                    let mut g = self.idem.lock();
                    if let Some((existing_id, existing_hash)) = g.get(&key).copied() {
                        if existing_hash == payload_hash {
                            records.push(IngestRecordOutcome::Idempotent {
                                internal_id: existing_id,
                                external_id: ext,
                            });
                            // Idempotent hits do NOT advance LSN, but we
                            // count them as "successful" (they don't
                            // count against `failed_count`).
                            continue;
                        } else {
                            failed += 1;
                            records.push(IngestRecordOutcome::Failed {
                                external_id: Some(ext.clone()),
                                error: IngestError::IdempotencyConflict { external_id: ext },
                            });
                            continue;
                        }
                    }
                    let id = self.alloc_id();
                    g.insert(key, (id, payload_hash));
                    drop(g);
                    self.wal_log.lock().push(format!("NODE:{}", n.label));
                    let lsn = id; // stub: LSN increments with internal id
                    commit_lsn = Some(commit_lsn.map_or(lsn, |c| c.max(lsn)));
                    inserted += 1;
                    records.push(IngestRecordOutcome::Inserted {
                        internal_id: id,
                        external_id: Some(ext),
                    });
                } else {
                    let id = self.alloc_id();
                    self.wal_log.lock().push(format!("NODE:{}", n.label));
                    let lsn = id;
                    commit_lsn = Some(commit_lsn.map_or(lsn, |c| c.max(lsn)));
                    inserted += 1;
                    records.push(IngestRecordOutcome::Inserted {
                        internal_id: id,
                        external_id: None,
                    });
                }
            }

            for r in batch.relationships {
                // Per IngestProvider hard contract 5: cancellation
                // surfaces as `IngestError::Storage { detail: "cancelled" }`
                // per record (same shape as the node-loop check).
                if self.cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                    failed += 1;
                    records.push(IngestRecordOutcome::Failed {
                        external_id: r.external_id,
                        error: IngestError::Storage {
                            detail: "cancelled".into(),
                        },
                    });
                    continue;
                }
                if self.fail_storage {
                    failed += 1;
                    records.push(IngestRecordOutcome::Failed {
                        external_id: r.external_id,
                        error: IngestError::Storage {
                            detail: "disk full".into(),
                        },
                    });
                    continue;
                }
                if r.rel_type.trim().is_empty() {
                    failed += 1;
                    records.push(IngestRecordOutcome::Failed {
                        external_id: r.external_id,
                        error: IngestError::Invalid {
                            detail: "empty rel_type".into(),
                        },
                    });
                    continue;
                }
                let id = self.alloc_id();
                self.wal_log.lock().push(format!("REL:{}", r.rel_type));
                let lsn = id;
                commit_lsn = Some(commit_lsn.map_or(lsn, |c| c.max(lsn)));
                inserted += 1;
                records.push(IngestRecordOutcome::Inserted {
                    internal_id: id,
                    external_id: r.external_id,
                });
            }

            Ok(IngestSummary {
                records,
                inserted_count: inserted,
                failed_count: failed,
                commit_lsn,
                dropped_acl_grants: Vec::new(),
            })
        }
    }

    fn one_node(label: &str, ext: Option<&str>) -> NodeIngest {
        NodeIngest {
            external_id: ext.map(String::from),
            label: label.into(),
            properties: BTreeMap::new(),
        }
    }

    // ------ Unit tests ------

    #[test]
    fn ingest_tool_rejects_cross_tenant_request_with_unauthorized() {
        // Same M5-05 cross-tenant guard shape: the session is bound to
        // tenant 1, the request asks for tenant 2 → -32002 BEFORE any
        // provider call. We verify "before any provider call" by
        // pointing the provider at NEITHER tenant.
        let p = StubIngestProvider::default();
        let req = IngestRequest {
            tenant_id: 2,
            nodes: vec![one_node("Person", None)],
            relationships: vec![],
            acl_grants: None,
            format: None,
        };
        let err = ingest_tool(&p, TenantId::new(1), req).expect_err("must reject");
        assert_eq!(err.code(), -32002);
        assert!(matches!(err, MCPError::Unauthorized));
    }

    #[test]
    fn ingest_tool_input_validation_rejects_empty_label() {
        // Per-record validation: empty label → IngestError::Invalid in
        // the outcome envelope. Tool-level call still returns Ok (per
        // the M5-08 contract: per-record faults DON'T fail the request
        // envelope).
        let p = StubIngestProvider::new(TenantId::new(7));
        let req = IngestRequest {
            tenant_id: 7,
            nodes: vec![one_node("", Some("client-1"))],
            relationships: vec![],
            acl_grants: None,
            format: Some(ResponseFormat::Json),
        };
        let resp = ingest_tool(&p, TenantId::new(7), req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("invalid"), "body={body}");
        assert!(body.contains("empty label"), "body={body}");
        assert!(body.contains("\"failed_count\":1"), "body={body}");
    }

    #[test]
    fn ingest_tool_per_record_idempotency_returns_same_internal_id() {
        // Per the M5-08 contract: re-submit of (tenant, external_id)
        // with same payload → Idempotent outcome with the SAME
        // internal_id. Pin this hard — implementer impls that
        // re-allocate IDs on retry break reads-after-write.
        let p = StubIngestProvider::new(TenantId::new(7));
        let mut req = IngestRequest {
            tenant_id: 7,
            nodes: vec![one_node("Person", Some("k1"))],
            relationships: vec![],
            acl_grants: None,
            format: None,
        };

        let resp1 = ingest_tool(&p, TenantId::new(7), req.clone()).expect("ok");
        let body1 = resp1["body"].as_str().unwrap();
        // First call: Inserted.
        assert!(body1.contains("\"status\":\"inserted\""));
        // Second call with the same payload.
        req.format = None;
        let resp2 = ingest_tool(&p, TenantId::new(7), req).expect("ok");
        let body2 = resp2["body"].as_str().unwrap();
        assert!(body2.contains("\"status\":\"idempotent\""), "body2={body2}");
    }

    #[test]
    fn ingest_tool_idempotency_conflict_on_different_payload() {
        // Same external_id, different payload → IdempotencyConflict in
        // the outcome envelope. The implementer MUST detect this.
        let p = StubIngestProvider::new(TenantId::new(7));

        // First insert with empty properties.
        let req1 = IngestRequest {
            tenant_id: 7,
            nodes: vec![one_node("Person", Some("k1"))],
            relationships: vec![],
            acl_grants: None,
            format: None,
        };
        let _ = ingest_tool(&p, TenantId::new(7), req1).expect("ok");

        // Second insert with different properties — same external_id.
        let mut props = BTreeMap::new();
        props.insert("name".into(), serde_json::json!("Alice"));
        let req2 = IngestRequest {
            tenant_id: 7,
            nodes: vec![NodeIngest {
                external_id: Some("k1".into()),
                label: "Person".into(),
                properties: props,
            }],
            relationships: vec![],
            acl_grants: None,
            format: Some(ResponseFormat::Json),
        };
        let resp = ingest_tool(&p, TenantId::new(7), req2).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("idempotency_conflict"), "body={body}");
    }

    #[test]
    fn ingest_tool_emits_one_fsync_per_cohort_per_adr031() {
        // ADR-031 §Decision (single CommitBundle per commit, group-
        // commit window): multiple records in a single ingest call
        // MAY share an fsync cohort, but every successful call MUST
        // imply a successful fsync. The stub counts fsyncs; we
        // assert the contract.
        let p = StubIngestProvider::new(TenantId::new(7));
        let req = IngestRequest {
            tenant_id: 7,
            nodes: vec![
                one_node("Person", None),
                one_node("Doc", None),
                one_node("Comment", None),
            ],
            relationships: vec![],
            acl_grants: None,
            format: None,
        };
        let _ = ingest_tool(&p, TenantId::new(7), req).expect("ok");
        assert_eq!(*p.fsyncs.lock(), 1, "exactly one fsync per ingest call");
        // WAL log must contain all 3 records.
        assert_eq!(p.wal_log.lock().len(), 3);
    }

    #[test]
    fn ingest_tool_propagates_tenant_unknown_from_provider() {
        // Even when the dispatcher's cross-tenant guard passes (session
        // tenant matches request tenant), the provider may still
        // surface TenantUnknown if the catalog has no binding for
        // tenant 9. MUST surface as -32003.
        let p = StubIngestProvider {
            force_tenant_unknown: true,
            ..StubIngestProvider::new(TenantId::new(9))
        };
        let req = IngestRequest {
            tenant_id: 9,
            nodes: vec![one_node("Person", None)],
            relationships: vec![],
            acl_grants: None,
            format: None,
        };
        let err = ingest_tool(&p, TenantId::new(9), req).expect_err("must reject");
        assert_eq!(err.code(), -32003);
    }

    #[test]
    fn ingest_tool_default_format_is_json() {
        // Per the M5-08 wire contract: graph.ingest defaults to JSON
        // (heterogeneous outcome shape; YAML / TOON pivot through
        // Value). Pin this so a future M5 sub-slice can't silently
        // change it.
        let p = StubIngestProvider::new(TenantId::new(7));
        let req = IngestRequest {
            tenant_id: 7,
            nodes: vec![one_node("Person", None)],
            relationships: vec![],
            acl_grants: None,
            format: None,
        };
        let resp = ingest_tool(&p, TenantId::new(7), req).expect("ok");
        assert_eq!(resp["format"], "json");
    }

    #[test]
    fn ingest_request_rejects_unknown_field() {
        // code-quality policy strict-mode discipline.
        let v = serde_json::json!({
            "tenant_id": 7,
            "nodes": [],
            "rels": []  // typo of `relationships`
        });
        let res: Result<IngestRequest, _> = serde_json::from_value(v);
        assert!(res.is_err(), "typo must reject");
    }

    #[test]
    fn ingest_summary_round_trips_through_serde_json() {
        // Wire-shape round-trip pin — same discipline as
        // graph_schema_round_trips_through_serde_json. Future
        // refactors can't silently change tags.
        let s = IngestSummary {
            records: vec![
                IngestRecordOutcome::Inserted {
                    internal_id: 1000,
                    external_id: Some("k1".into()),
                },
                IngestRecordOutcome::Idempotent {
                    internal_id: 999,
                    external_id: "k0".into(),
                },
                IngestRecordOutcome::Failed {
                    external_id: Some("k2".into()),
                    error: IngestError::Invalid { detail: "x".into() },
                },
            ],
            inserted_count: 1,
            failed_count: 1,
            commit_lsn: Some(1001),
            dropped_acl_grants: Vec::new(),
        };
        let v = serde_json::to_value(&s).unwrap();
        let s2: IngestSummary = serde_json::from_value(v).unwrap();
        assert_eq!(s, s2);
    }

    #[test]
    fn ingest_tool_commit_lsn_advances_with_each_inserted_record() {
        // Reads-after-write contract: the returned commit_lsn must
        // advance such that subsequent reads at this LSN observe the
        // writes. Pin: commit_lsn is Some after at least one Insert.
        let p = StubIngestProvider::new(TenantId::new(7));
        let req = IngestRequest {
            tenant_id: 7,
            nodes: vec![one_node("Person", None), one_node("Doc", None)],
            relationships: vec![],
            acl_grants: None,
            format: Some(ResponseFormat::Json),
        };
        let resp = ingest_tool(&p, TenantId::new(7), req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        // The body must contain a non-null commit_lsn.
        assert!(body.contains("\"commit_lsn\":1001"), "body={body}");
    }

    #[test]
    fn ingest_tool_simulates_storage_failure_preserves_envelope() {
        // Renamed from the prior misnamed
        // ingest_tool_simulated_sigterm_cancellation_preserves_outcome_envelope.
        // What this test ACTUALLY pins: a per-record `Storage("disk
        // full")` fault surfaces inside the outcome envelope rather
        // than aborting the request. The wire-shape is preserved even
        // when every record fails. SIGTERM coverage is provided by
        // ingest_tool_cancellation_via_cancel_flag_surfaces_outcome_envelope
        // below.
        let p = StubIngestProvider {
            fail_storage: true,
            ..StubIngestProvider::new(TenantId::new(7))
        };
        let req = IngestRequest {
            tenant_id: 7,
            nodes: vec![one_node("Person", None)],
            relationships: vec![],
            acl_grants: None,
            format: Some(ResponseFormat::Json),
        };
        let resp = ingest_tool(&p, TenantId::new(7), req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        // The response envelope reaches the caller even when every
        // record failed; the wire shape is consistent.
        assert!(body.contains("\"failed_count\":1"));
        assert!(body.contains("disk full"));
    }

    #[test]
    fn ingest_tool_cancellation_via_cancel_flag_surfaces_outcome_envelope() {
        // Real exercise of the [`IngestProvider`] hard contract 5:
        // when the provider observes a cancellation signal (W12γ-
        // style: an AtomicBool flag set externally by SIGTERM /
        // engine.cancel_all() / equivalent), per-record output MUST
        // surface as `IngestError::Storage { detail: "cancelled" }`
        // rather than being silently dropped. The outcome envelope is
        // still returned to the caller (caller learns which records
        // committed pre-cancel; here all 3 are cancelled because the
        // flag is set before the first record).
        //
        // The W14γ slice forward-defers SIGTERM signal wiring to
        // M4-08+ (which composes the `CancellationRegistry` /
        // `ExecutionContext::with_cancellation` plumbing per
        // amendment-03 §TIER-1 GAP C); this test pins the per-record
        // fault SHAPE that the wiring must surface.
        use std::sync::atomic::{AtomicBool, Ordering};

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let p = StubIngestProvider {
            cancel_flag: Arc::clone(&cancel_flag),
            ..StubIngestProvider::new(TenantId::new(7))
        };
        // Fire the cancellation BEFORE the ingest call — equivalent
        // to a SIGTERM arriving during request demarshalling but
        // before the first record commits.
        cancel_flag.store(true, Ordering::SeqCst);

        let req = IngestRequest {
            tenant_id: 7,
            nodes: vec![
                one_node("Person", Some("a")),
                one_node("Doc", Some("b")),
                one_node("Comment", Some("c")),
            ],
            relationships: vec![],
            acl_grants: None,
            format: Some(ResponseFormat::Json),
        };
        let resp = ingest_tool(&p, TenantId::new(7), req).expect("envelope returned");
        let body = resp["body"].as_str().unwrap();

        // Every record surfaces as failed with the "cancelled" detail.
        assert!(
            body.contains("\"failed_count\":3"),
            "expected all 3 records failed; body={body}"
        );
        // The cancellation fault uses the canonical IngestError::Storage
        // shape with detail "cancelled" per contract 5.
        assert!(
            body.contains("cancelled"),
            "outcome must carry detail=\"cancelled\"; body={body}"
        );
        assert!(
            body.contains("\"inserted_count\":0"),
            "no records can commit after cancellation; body={body}"
        );
    }

    #[test]
    fn ingest_tool_rels_empty_when_no_relationships_provided() {
        // Default-empty `relationships` defaulting via #[serde(default)]
        // — pin behavior so a JSON-RPC client can omit the slot.
        let p = StubIngestProvider::new(TenantId::new(7));
        let req: IngestRequest = serde_json::from_value(serde_json::json!({
            "tenant_id": 7,
            "nodes": [{"label": "Person"}]
        }))
        .expect("default-empty rels deserialize");
        let resp = ingest_tool(&p, TenantId::new(7), req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("\"inserted_count\":1"));
    }

    #[test]
    fn ingest_provider_is_send_sync() {
        // Compile-time pin: the trait object MUST be Send + Sync so a
        // single Arc<dyn IngestProvider> serves the multi-tenant
        // dispatcher across awaits.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<dyn IngestProvider>>();
    }
}
