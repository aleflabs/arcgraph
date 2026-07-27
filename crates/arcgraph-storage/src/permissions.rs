//! Per-principal source-ACL permission index — ADR-212 stage-1.
//!
//! The derived enforcement plane for permission-aware retrieval
//! (AEB keystone #1093): documents carry source-system ACLs (ingested
//! alongside content at the ADR-050 connector seam); at query time a
//! requesting principal's **effective permission set** filters every
//! candidate. This module is the ADR-212 §D-2(b) *derived index*
//! (interned ACL classes + per-principal allowed-class bitsets); the
//! ACL **source of truth** is tenant-resident graph data (`_AclPrincipal`
//! nodes + `_ACL_GRANT` edges, written by the ingest path — see
//! `arcgraph-mcp::storage::acl_ingest` at stage-1).
//!
//! # Latency / memory budget (ADR-212 §4, Prime Directive 5)
//!
//! - `EffectivePermissions::is_visible`: one `DashMap` read + one bitset
//!   test ≈ 20–60 ns per candidate; a 10k-candidate filter adds
//!   ~0.2–0.6 ms — target < 5 % of engine read-path p95. Pinned by
//!   `benches/permissions_bench.rs`.
//! - `PermissionIndex::effective` (cold): O(classes) set-membership
//!   scans; classes ≪ docs (ACL-class interning — SharePoint/Slack
//!   estates share container-level ACLs). Cached per principal until
//!   the index generation moves.
//! - Memory: doc→class is 1 `u64→u32` entry per ACL-tagged doc
//!   (~12 B + map overhead); class table is append-only and bounded by
//!   the number of DISTINCT grant-sets ever seen (revisit trigger in
//!   ADR-212 §8 if a real estate approaches classes ≈ docs).
//!
//! # Security invariants (ADR-212 §D-4/§D-5 — fail-closed)
//!
//! 1. **Untagged ⇒ invisible.** A node with no `doc→class` mapping is
//!    `UNCLASSIFIED`: `is_visible` returns `false` for EVERY principal.
//!    Content ingested without ACLs is unreachable under enforcement.
//! 2. **Classes are immutable.** A grant change NEVER mutates a class
//!    in place — it interns a (possibly new) class and remaps the doc.
//!    A stale `EffectivePermissions` snapshot can therefore only
//!    UNDER-grant: the doc's old class id no longer resolves from
//!    `doc_class`, and the new class id is absent from the stale
//!    bitset. Staleness narrows; it never widens (`test
//!    stale_snapshot_never_widens_access`).
//! 3. **Statement-granularity freshness.** Consumers resolve
//!    `effective()` once per request/statement (ADR-212 §D-5); a
//!    revocation committed through [`PermissionIndex::apply_doc_acl`] /
//!    [`PermissionIndex::revoke_doc`] bumps the generation and is
//!    honored by the NEXT resolution (in-engine propagation is
//!    sub-millisecond; the ≤ 15 min MUST-GOV-03 budget is spent in
//!    connector sync, measured at stage-3).
//!
//! # Stage-1 posture (documented, ADR-050-precedent)
//!
//! The index is **in-memory and write-through**: the supported ACL
//! mutation path is the ingest seam calling [`PermissionIndex`]
//! directly in the same call-path that writes the `_Acl*` provenance
//! graph data. Restart loses the index ⇒ every doc is UNCLASSIFIED ⇒
//! **deny-all under enforcement** until re-ingest/rebuild — fail-closed
//! in the safe direction (same in-memory-at-alpha posture as ADR-050's
//! `WatermarkStore`/DLQ). Rebuild-from-graph + CDC-tail invalidation
//! (multi-writer paths, e.g. ArcQL mutations of `_Acl*` data) are
//! stage-2 scope per ADR-212 §D-7.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use arcgraph_core::{NodeId, TenantId};
use dashmap::DashMap;

use crate::IdempotencyStore;
use crate::owner_row::{
    ClassOwnerValue, GrantOwnerValue, OWNER_ALLOCATOR_MARKER_ID, OwnerAllocatorMarker,
    OwnerRowClass, OwnerRowError, OwnerRowRegistry,
};
use crate::wal::{AllocatorAdvance, AllocatorKind};

/// Reserved principal granting tenant-wide read: a doc whose grant set
/// contains `PUBLIC_PRINCIPAL` is visible to every resolved principal.
pub const PUBLIC_PRINCIPAL: &str = "__public__";

/// Reserved node label for ingested principal provenance (ADR-212
/// §D-2(a) graph convention; written by the ingest seam, not by this
/// module).
pub const ACL_PRINCIPAL_LABEL: &str = "_AclPrincipal";

/// Reserved relationship type for ingested grant provenance
/// (`(_AclPrincipal)-[:_ACL_GRANT]->(content)`).
pub const ACL_GRANT_TYPE: &str = "_ACL_GRANT";

/// Durable WAL sink for `PermissionIndex` ACL grant/revoke ops
/// (#1221 — ADR-218).
///
/// The boundary contract: [`PermissionIndex`] is the *enforcement* index
/// (an in-memory derived plane). It does NOT know about the WAL / CRUD
/// layer. When a durable backend is in play, it hands the index an
/// `AclWalSink` (implemented in the CRUD/WAL layer) so that every
/// write-through `PermissionIndex::apply_doc_acl` / `revoke_doc`
/// durifies its op into the WAL's `acl_grants` section atomically with a
/// dedicated single-op commit — making the grant present iff its commit
/// is (both-or-neither, ADR-218). On replay the backend re-drives
/// `apply_doc_acl`/`revoke_doc` against a fresh index via the
/// `*_replayed` entry points, which do NOT re-enter the sink (the op is
/// already durable).
///
/// Keeping this a narrow trait (not a `crud` dependency) preserves the
/// bounded-context boundary: `permissions.rs` depends on this trait
/// only; the implementation lives where the WAL does.
pub trait AclWalSink: Send + Sync + std::fmt::Debug {
    /// Durify an `Apply` op (`doc → grants`) — issue a dedicated
    /// single-op v8 commit carrying one `acl_grants` `Apply` entry. On
    /// failure the op is NOT durable; the caller has already applied it
    /// to the in-memory index (the same in-memory-first posture the
    /// pre-ADR-218 index had), so a torn write degrades to "lost on next
    /// restart" — fail-closed (the doc reverts to UNCLASSIFIED), never a
    /// widen.
    fn durify_apply(&self, doc: NodeId, grants: &BTreeSet<String>);

    /// Durify a `Revoke` op (`doc` → UNCLASSIFIED) — issue a dedicated
    /// single-op v8 commit carrying one `acl_grants` `Revoke` entry.
    fn durify_revoke(&self, doc: NodeId);
}

/// Interned id for one DISTINCT grant-set (an "ACL class").
///
/// Docs sharing an identical grant set share a class; per-candidate
/// visibility is a class-bitset test, not a set comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AclClassId(pub u32);

/// Append-only word-bitset over [`AclClassId`] indices.
///
/// Hand-rolled (≈20 lines) instead of pulling a bitmap dependency:
/// class cardinality is small (≪ docs), and a new dependency would
/// require separate review.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClassBitset {
    words: Vec<u64>,
}

impl ClassBitset {
    /// Set bit `idx`.
    pub fn insert(&mut self, idx: u32) {
        let word = (idx / 64) as usize;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1u64 << (idx % 64);
    }

    /// Test bit `idx`.
    #[must_use]
    pub fn contains(&self, idx: u32) -> bool {
        let word = (idx / 64) as usize;
        self.words
            .get(word)
            .is_some_and(|w| w & (1u64 << (idx % 64)) != 0)
    }

    /// Number of set bits (test/diagnostic helper).
    #[must_use]
    pub fn count(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }
}

/// A principal's resolved effective permission set — an immutable
/// snapshot answering [`Self::is_visible`] in O(1).
///
/// Snapshots are cached by [`PermissionIndex::effective`] and
/// invalidated by generation; holding one across index mutations is
/// safe in the narrow direction only (module docs invariant 2).
#[derive(Debug)]
pub struct EffectivePermissions {
    principal: String,
    /// Index generation this snapshot was resolved at.
    generation: u64,
    backing: EffectivePermissionsBacking,
}

const PHYSICAL_PERMISSION_CACHE_CAP: usize = 1_024;

#[derive(Debug)]
struct BoundedPermissionCache {
    entries: HashMap<String, Arc<EffectivePermissions>>,
    order: VecDeque<String>,
    cap: usize,
}

impl Default for BoundedPermissionCache {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            cap: PHYSICAL_PERMISSION_CACHE_CAP,
        }
    }
}

impl BoundedPermissionCache {
    fn get(&mut self, principal: &str, generation: u64) -> Option<Arc<EffectivePermissions>> {
        let hit = self
            .entries
            .get(principal)
            .filter(|snapshot| snapshot.generation == generation)
            .cloned()?;
        self.order.retain(|key| key != principal);
        self.order.push_back(principal.to_owned());
        Some(hit)
    }

    fn insert(&mut self, snapshot: Arc<EffectivePermissions>) {
        let principal = snapshot.principal.clone();
        self.order.retain(|key| key != &principal);
        self.order.push_back(principal.clone());
        self.entries.insert(principal, snapshot);
        while self.entries.len() > self.cap {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
            }
        }
    }
}

#[derive(Debug)]
enum EffectivePermissionsBacking {
    Resident {
        allowed: ClassBitset,
        doc_class: Arc<DashMap<u64, AclClassId>>,
    },
    Physical {
        tenant: TenantId,
        owner: Arc<OwnerRowRegistry>,
    },
}

impl EffectivePermissions {
    /// `true` iff `node` is ACL-tagged AND its class is in this
    /// principal's allowed set. Untagged ⇒ `false` (fail-closed,
    /// invariant 1).
    #[must_use]
    pub fn is_visible(&self, node: NodeId) -> bool {
        self.try_is_visible(node).unwrap_or_else(|error| {
            tracing::error!(%error, "M4 ACL visibility lookup failed closed");
            false
        })
    }

    /// Typed enforcement lookup. The physical path faults the document's
    /// direct grant row and then its immutable class row; it never closes over
    /// a resident doc map.
    pub fn try_is_visible(&self, node: NodeId) -> Result<bool, OwnerRowError> {
        match &self.backing {
            EffectivePermissionsBacking::Resident { allowed, doc_class } => Ok(doc_class
                .get(&node.raw())
                .is_some_and(|class| allowed.contains(class.value().0))),
            EffectivePermissionsBacking::Physical { tenant, owner } => {
                let Some(grant) = owner.read_logical(*tenant, OwnerRowClass::Grant, node.raw())?
                else {
                    return Ok(false);
                };
                let grant = GrantOwnerValue::decode(&grant)?;
                if !grant.active {
                    return Ok(false);
                }
                let Some(class) = owner.read_logical(
                    *tenant,
                    OwnerRowClass::ClassId,
                    u64::from(grant.class_id),
                )?
                else {
                    return Err(OwnerRowError::InvalidEnvelope(format!(
                        "grant for doc {} references missing class {}",
                        node.raw(),
                        grant.class_id
                    )));
                };
                let grants = ClassOwnerValue::decode(&class)?.grants;
                Ok(grants.contains(&self.principal) || grants.contains(PUBLIC_PRINCIPAL))
            }
        }
    }

    /// The principal this snapshot was resolved for.
    #[must_use]
    pub fn principal(&self) -> &str {
        &self.principal
    }

    /// The index generation this snapshot was resolved at.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Per-tenant derived permission index (ADR-212 §D-2(b)).
///
/// One instance per tenant, owned by the routing layer and exposed as
/// [`crate::router::TenantHandle::permissions`] (ADR-037-amendment-02).
/// All maps are tenant-local by construction — the index never sees
/// another tenant's data (ADR-212 §5 Q3).
#[derive(Debug, Default)]
struct ResidentPermissionOwners {
    /// Distinct grant-set → class id (interning map). Append-only.
    class_intern: DashMap<BTreeSet<String>, AclClassId>,
    /// Class id → its immutable grant set.
    class_grants: DashMap<u32, Arc<BTreeSet<String>>>,
    /// ACL-tagged doc → current class.
    doc_class: Arc<DashMap<u64, AclClassId>>,
    /// Pre-M4 unbounded derived cache.
    cache: DashMap<String, Arc<EffectivePermissions>>,
    /// Ingest-side principal-node provenance dedup.
    principal_nodes: DashMap<String, NodeId>,
    /// Pre-M4 logical grant WAL sink.
    wal_sink: parking_lot::RwLock<Option<Arc<dyn AclWalSink>>>,
    /// Pre-M4 capture peak instrumentation.
    capture_peak_in_flight: AtomicU64,
    /// Pre-M4 capture-vs-install exclusion.
    capture_lock: parking_lot::RwLock<()>,
}

#[derive(Debug)]
pub struct PermissionIndex {
    /// M4 authoritative owner. All five legacy maps remain empty when set.
    physical: Option<Arc<OwnerRowRegistry>>,
    /// Pre-M4 owners. `None` after the generation swap: the five scalable
    /// DashMaps and logical WAL sink are not constructed in the physical
    /// process owner at all.
    resident: Option<Box<ResidentPermissionOwners>>,
    physical_tenant: Option<TenantId>,
    physical_idempotency: Option<Arc<IdempotencyStore>>,
    physical_write: parking_lot::Mutex<()>,
    physical_cache: parking_lot::Mutex<BoundedPermissionCache>,
    /// Bumped on every ACL mutation; stale cache entries re-resolve.
    generation: AtomicU64,
    /// #1184 (ADR-226 S1): class-id allocator. Ids are handed out by a
    /// `fetch_add` INSIDE `class_intern`'s vacant-entry closure — never
    /// derived from a `len()` read — so two writers interning DISTINCT
    /// new sets can never alias to one id. Process-local: ids are never
    /// persisted (ADR-218 WAL replay re-interns via the `*_replayed`
    /// entry points), so starting at 0 per process is correct.
    next_class: AtomicU32,
}

impl Default for PermissionIndex {
    fn default() -> Self {
        Self {
            physical: None,
            resident: Some(Box::default()),
            physical_tenant: None,
            physical_idempotency: None,
            physical_write: parking_lot::Mutex::new(()),
            physical_cache: parking_lot::Mutex::new(BoundedPermissionCache::default()),
            generation: AtomicU64::new(0),
            next_class: AtomicU32::new(0),
        }
    }
}

impl PermissionIndex {
    /// Fresh empty index (every doc UNCLASSIFIED ⇒ deny-all under
    /// enforcement).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Open the M4 page-backed permission facade. The class allocator is
    /// seeded from its durable marker using `fetch_max`; all five legacy
    /// `PermissionIndex` maps stay empty.
    pub fn page_backed(
        owner: Arc<OwnerRowRegistry>,
        idempotency: Arc<IdempotencyStore>,
        tenant: TenantId,
    ) -> Result<Self, OwnerRowError> {
        let index = Self {
            physical: Some(Arc::clone(&owner)),
            resident: None,
            physical_tenant: Some(tenant),
            physical_idempotency: Some(idempotency),
            physical_write: parking_lot::Mutex::new(()),
            physical_cache: parking_lot::Mutex::new(BoundedPermissionCache::default()),
            generation: AtomicU64::new(0),
            next_class: AtomicU32::new(0),
        };
        let next_id =
            match owner.read_logical(tenant, OwnerRowClass::ClassId, OWNER_ALLOCATOR_MARKER_ID)? {
                Some(logical) => {
                    let marker = OwnerAllocatorMarker::decode(&logical)?;
                    if marker.kind != AllocatorKind::AclClass.as_byte() {
                        return Err(OwnerRowError::InvalidEnvelope(format!(
                            "ACL class allocator marker has kind {}",
                            marker.kind
                        )));
                    }
                    u32::try_from(marker.high_water)
                        .map_err(|_| {
                            OwnerRowError::InvalidEnvelope(
                                "ACL class allocator marker exceeds u32".to_owned(),
                            )
                        })?
                        .checked_add(1)
                        .ok_or_else(|| {
                            OwnerRowError::InvalidEnvelope(
                                "ACL class allocator marker is exhausted".to_owned(),
                            )
                        })?
                }
                None => 0,
            };
        index.next_class.fetch_max(next_id, Ordering::AcqRel);
        Ok(index)
    }

    /// Recovery seed (ADR-034 D-1, Lemma-I3): advance the ACL class allocator
    /// from a replayed [`AllocatorAdvance`] so a post-crash allocation cannot
    /// reuse an [`AclClassId`] a pre-fault commit already durably consumed.
    ///
    /// Mirrors `InternTable::seed_string_allocator`: the counter is seeded at
    /// bootstrap from the last **checkpointed** allocator marker, so only the
    /// replayed advance can carry it past post-checkpoint commits.
    ///
    /// `new_high_water` is the LAST allocated class id; the next allocation
    /// must return at least `new_high_water + 1`. `fetch_max` keeps it
    /// idempotent under double-replay.
    ///
    /// Tenant-guarded: this index is per-tenant, and an advance for a different
    /// tenant belongs to that tenant's own index. Advances are ignored on the
    /// resident arm, whose class ids are reconstructed rather than durable.
    pub fn seed_class_allocator(&self, tenant: TenantId, new_high_water: u64) {
        let Some(physical_tenant) = self.physical_tenant else {
            return;
        };
        if physical_tenant != tenant {
            return;
        }
        let next = u32::try_from(new_high_water.saturating_add(1)).unwrap_or(u32::MAX);
        self.next_class.fetch_max(next, Ordering::AcqRel);
    }

    /// True only for the post-M4 facade.
    #[must_use]
    pub fn is_page_backed(&self) -> bool {
        self.physical.is_some()
    }

    /// Class-complete resident census for the five legacy maps.
    #[must_use]
    pub fn resident_map_cardinalities(&self) -> [usize; 5] {
        self.resident.as_ref().map_or([0; 5], |resident| {
            [
                resident.class_intern.len(),
                resident.class_grants.len(),
                resident.doc_class.len(),
                resident.cache.len(),
                resident.principal_nodes.len(),
            ]
        })
    }

    /// Structural census: the post-swap owner has no legacy map bundle.
    #[must_use]
    pub fn has_resident_owner_maps(&self) -> bool {
        self.resident.is_some()
    }

    fn resident(&self) -> &ResidentPermissionOwners {
        match self.resident.as_deref() {
            Some(resident) => resident,
            None => unreachable!("resident permission path selected for page-backed facade"),
        }
    }

    /// Bounded physical cache census `(current, hard_cap)`.
    #[must_use]
    pub fn physical_cache_census(&self) -> (usize, usize) {
        let cache = self.physical_cache.lock();
        (cache.entries.len(), cache.cap)
    }

    /// Next durable ACL class id after marker seeding.
    #[doc(hidden)]
    #[must_use]
    pub fn class_allocator_next(&self) -> u32 {
        self.next_class.load(Ordering::Acquire)
    }

    /// Intern `grants` and map `doc` to the resulting class,
    /// replacing any previous mapping. An empty grant set is legal and
    /// means "no principal may read" (distinct from UNCLASSIFIED only
    /// in provenance: the source EXPLICITLY granted nobody).
    ///
    /// This is the write-through entry point for the ingest seam — the
    /// caller writes the `_Acl*` provenance graph data in the same
    /// flow. Bumps the generation (cached snapshots re-resolve).
    ///
    /// #1221 (ADR-218): when a durable [`AclWalSink`] is wired, this
    /// ALSO durifies the op into the WAL's `acl_grants` section so the
    /// grant survives a bare `serve --data` restart. The in-memory apply
    /// happens first (the pre-ADR-218 posture), then the durify — so a
    /// torn WAL write degrades to "lost on next restart" (the doc reverts
    /// to UNCLASSIFIED), never a widen. Replay uses
    /// [`Self::apply_doc_acl_replayed`], which bypasses the sink.
    pub fn apply_doc_acl(&self, doc: NodeId, grants: BTreeSet<String>) {
        self.apply_doc_acl_inner(doc, grants, true);
    }

    /// Typed write-through entry point used by durable serving.
    pub fn apply_doc_acl_checked(
        &self,
        doc: NodeId,
        grants: BTreeSet<String>,
    ) -> Result<(), OwnerRowError> {
        if self.physical.is_some() {
            self.try_apply_doc_acl(doc, grants)
        } else {
            self.apply_doc_acl_inner(doc, grants, true);
            Ok(())
        }
    }

    /// Replay-path apply (#1221 — ADR-218): re-drive an `Apply` op
    /// against this index WITHOUT re-staging it to the WAL (the op is
    /// already durable — re-staging would double-log it on the next
    /// restart). Called by the WAL replay executor in ascending
    /// `commit_lsn` order; last-writer-wins per doc is well-defined.
    pub fn apply_doc_acl_replayed(&self, doc: NodeId, grants: BTreeSet<String>) {
        self.apply_doc_acl_inner(doc, grants, false);
    }

    fn apply_doc_acl_inner(&self, doc: NodeId, grants: BTreeSet<String>, durify: bool) {
        if self.physical.is_some() {
            if durify {
                if let Err(error) = self.try_apply_doc_acl(doc, grants) {
                    tracing::error!(%error, "M4 ACL grant publish failed closed");
                }
            }
            return;
        }
        // #1404 M0.x FIX-D — capture-exclusion read guard (adds/updates a
        // `doc_class` entry → count-changing on a fresh doc; exclude during a
        // capture count+stream). Concurrent with other applies/revokes; blocks
        // only during a capture WRITE guard. Off the enforcement read path.
        let _install = self.resident().capture_lock.read();
        if durify && let Some(sink) = self.resident().wal_sink.read().as_ref() {
            // Durify BEFORE the in-memory mutation consumes `grants` —
            // the sink borrows the set; the index then interns it.
            sink.durify_apply(doc, &grants);
        }
        let class = self.intern_class(grants);
        self.resident().doc_class.insert(doc.raw(), class);
        self.bump();
    }

    /// Typed M4 grant publish. A freshly allocated durable class row, its
    /// allocator marker/advance, and the document's direct grant row share one
    /// v10 WAL commit.
    pub fn try_apply_doc_acl(
        &self,
        doc: NodeId,
        grants: BTreeSet<String>,
    ) -> Result<(), OwnerRowError> {
        let owner = self.physical.as_ref().ok_or_else(|| {
            OwnerRowError::InvalidEnvelope("physical ACL owner is unavailable".to_owned())
        })?;
        let tenant = self.physical_tenant.ok_or_else(|| {
            OwnerRowError::InvalidEnvelope("physical ACL tenant is unavailable".to_owned())
        })?;
        let canonical = ClassOwnerValue { grants };
        let encoded = canonical.encode()?;
        let hash = canonical.hash()?;
        let _write = self.physical_write.lock();
        let existing =
            owner.find_verified(tenant, OwnerRowClass::ClassId, hash, |_id, logical| {
                logical == encoded.as_slice()
            })?;
        let mut rows = Vec::with_capacity(3);
        let mut advances = Vec::with_capacity(1);
        let class_id = match existing {
            Some((id, _)) => u32::try_from(id).map_err(|_| {
                OwnerRowError::InvalidEnvelope("ACL class id exceeds u32".to_owned())
            })?,
            None => {
                let raw = self.next_class.fetch_add(1, Ordering::AcqRel);
                if u64::from(raw) >= OWNER_ALLOCATOR_MARKER_ID {
                    return Err(OwnerRowError::InvalidEnvelope(
                        "ACL class id exceeds direct owner capacity".to_owned(),
                    ));
                }
                rows.push(owner.prepare_indexed_logical_row(
                    tenant,
                    OwnerRowClass::ClassId,
                    u64::from(raw),
                    hash,
                    &encoded,
                )?);
                rows.push(
                    owner.prepare_direct_logical_row(
                        tenant,
                        OwnerRowClass::ClassId,
                        OWNER_ALLOCATOR_MARKER_ID,
                        &OwnerAllocatorMarker {
                            kind: AllocatorKind::AclClass.as_byte(),
                            high_water: u64::from(raw),
                        }
                        .encode(),
                    )?,
                );
                advances.push(AllocatorAdvance {
                    tenant,
                    kind: AllocatorKind::AclClass,
                    new_high_water: u64::from(raw),
                });
                raw
            }
        };
        rows.push(
            owner.prepare_direct_logical_row(
                tenant,
                OwnerRowClass::Grant,
                doc.raw(),
                &GrantOwnerValue {
                    class_id,
                    active: true,
                }
                .encode(),
            )?,
        );
        owner.commit_rows_with_allocator_advances(tenant, rows, advances)?;
        self.bump();
        Ok(())
    }

    /// Remove `doc`'s ACL mapping entirely → UNCLASSIFIED → invisible
    /// to every principal (revocation-by-removal; fail-closed).
    ///
    /// #1221 (ADR-218): durifies a `Revoke` op when a sink is wired (see
    /// [`Self::apply_doc_acl`]). Replay uses
    /// [`Self::revoke_doc_replayed`].
    pub fn revoke_doc(&self, doc: NodeId) {
        self.revoke_doc_inner(doc, true);
    }

    /// Typed revoke used by durable serving.
    pub fn revoke_doc_checked(&self, doc: NodeId) -> Result<(), OwnerRowError> {
        if self.physical.is_some() {
            self.try_revoke_doc(doc)
        } else {
            self.revoke_doc_inner(doc, true);
            Ok(())
        }
    }

    /// Replay-path revoke (#1221 — ADR-218): re-drive a `Revoke` op
    /// without re-staging to the WAL. See [`Self::apply_doc_acl_replayed`].
    pub fn revoke_doc_replayed(&self, doc: NodeId) {
        self.revoke_doc_inner(doc, false);
    }

    fn revoke_doc_inner(&self, doc: NodeId, durify: bool) {
        if self.physical.is_some() {
            if durify && let Err(error) = self.try_revoke_doc(doc) {
                tracing::error!(%error, "M4 ACL revoke publish failed closed");
            }
            return;
        }
        // #1404 M0.x FIX-D — capture-exclusion read guard (removes a
        // `doc_class` entry → count-changing; exclude during a capture
        // count+stream).
        let _install = self.resident().capture_lock.read();
        if durify && let Some(sink) = self.resident().wal_sink.read().as_ref() {
            sink.durify_revoke(doc);
        }
        self.resident().doc_class.remove(&doc.raw());
        self.bump();
    }

    /// Typed M4 revoke. The tombstone is a physical direct row, so restart
    /// never reconstructs it through a retired resident map.
    pub fn try_revoke_doc(&self, doc: NodeId) -> Result<(), OwnerRowError> {
        let owner = self.physical.as_ref().ok_or_else(|| {
            OwnerRowError::InvalidEnvelope("physical ACL owner is unavailable".to_owned())
        })?;
        let tenant = self.physical_tenant.ok_or_else(|| {
            OwnerRowError::InvalidEnvelope("physical ACL tenant is unavailable".to_owned())
        })?;
        let row = owner.prepare_direct_logical_row(
            tenant,
            OwnerRowClass::Grant,
            doc.raw(),
            &GrantOwnerValue {
                class_id: 0,
                active: false,
            }
            .encode(),
        )?;
        owner.commit_row(tenant, row)?;
        self.bump();
        Ok(())
    }

    /// #1221 (ADR-218): wire the durable [`AclWalSink`] so subsequent
    /// write-through `apply_doc_acl` / `revoke_doc` calls durify into the
    /// WAL. Called once at durable-backend bootstrap. Idempotent
    /// (last-write-wins); `None`-by-default keeps the in-memory-only
    /// posture for ephemeral / unit callers.
    pub fn set_wal_sink(&self, sink: Arc<dyn AclWalSink>) {
        if let Some(resident) = self.resident.as_deref() {
            *resident.wal_sink.write() = Some(sink);
        }
    }

    /// `true` iff a durable [`AclWalSink`] is wired (diagnostics/tests).
    #[must_use]
    pub fn has_wal_sink(&self) -> bool {
        self.resident
            .as_deref()
            .is_some_and(|resident| resident.wal_sink.read().is_some())
    }

    /// Resolve (or fetch cached) effective permissions for
    /// `principal`: every class whose grant set contains `principal`
    /// or [`PUBLIC_PRINCIPAL`].
    ///
    /// Per ADR-212 §D-5 consumers call this once per
    /// request/statement; the snapshot is immutable thereafter.
    #[must_use]
    pub fn effective(&self, principal: &str) -> Arc<EffectivePermissions> {
        let current = self.generation.load(Ordering::Acquire);
        if let (Some(owner), Some(tenant)) = (&self.physical, self.physical_tenant) {
            let mut cache = self.physical_cache.lock();
            if let Some(hit) = cache.get(principal, current) {
                return hit;
            }
            let snapshot = Arc::new(EffectivePermissions {
                principal: principal.to_owned(),
                generation: current,
                backing: EffectivePermissionsBacking::Physical {
                    tenant,
                    owner: Arc::clone(owner),
                },
            });
            cache.insert(Arc::clone(&snapshot));
            return snapshot;
        }
        if let Some(hit) = self.resident().cache.get(principal) {
            if hit.generation == current {
                return Arc::clone(&hit);
            }
        }
        let mut allowed = ClassBitset::default();
        for entry in &self.resident().class_grants {
            let grants = entry.value();
            if grants.contains(principal) || grants.contains(PUBLIC_PRINCIPAL) {
                allowed.insert(*entry.key());
            }
        }
        let snapshot = Arc::new(EffectivePermissions {
            principal: principal.to_owned(),
            generation: current,
            backing: EffectivePermissionsBacking::Resident {
                allowed,
                doc_class: Arc::clone(&self.resident().doc_class),
            },
        });
        self.resident()
            .cache
            .insert(principal.to_owned(), Arc::clone(&snapshot));
        snapshot
    }

    /// Ingest-side provenance dedup: the `_AclPrincipal` node id for
    /// `ext_id`, if one was recorded this process lifetime.
    #[must_use]
    pub fn principal_node(&self, ext_id: &str) -> Option<NodeId> {
        self.try_principal_node(ext_id).unwrap_or_else(|error| {
            tracing::error!(%error, "M4 ACL principal owner lookup failed closed");
            None
        })
    }

    /// Typed lookup used by durable ingest so a filesystem failure cannot be
    /// mistaken for an absent principal and create duplicate provenance.
    pub fn try_principal_node(&self, ext_id: &str) -> Result<Option<NodeId>, OwnerRowError> {
        if let (Some(idempotency), Some(tenant)) =
            (&self.physical_idempotency, self.physical_tenant)
        {
            return Ok(idempotency
                .try_get(tenant, 0, &format!("_acl:principal:{ext_id}"))?
                .map(|binding| NodeId::new(binding.internal_id)));
        }
        Ok(self
            .resident()
            .principal_nodes
            .get(ext_id)
            .map(|e| *e.value()))
    }

    /// Record the `_AclPrincipal` provenance node for `ext_id`
    /// (ingest-side dedup).
    pub fn record_principal_node(&self, ext_id: &str, node: NodeId) {
        if self.physical.is_some() {
            // The provenance node already carries `_acl:principal:<ext_id>`
            // through the ordinary idempotency owner in its record commit.
            let _ = (ext_id, node);
            return;
        }
        self.resident()
            .principal_nodes
            .insert(ext_id.to_owned(), node);
    }

    /// Number of ACL-tagged docs (diagnostics/tests).
    #[must_use]
    pub fn tagged_docs(&self) -> usize {
        if self.physical.is_some() {
            return 0;
        }
        self.resident().doc_class.len()
    }

    /// Number of interned classes (diagnostics/tests).
    #[must_use]
    pub fn class_count(&self) -> usize {
        if let (Some(owner), Some(tenant)) = (&self.physical, self.physical_tenant) {
            return owner
                .candidate_count(tenant, OwnerRowClass::ClassId)
                .unwrap_or(0) as usize;
        }
        self.resident().class_grants.len()
    }

    /// The number of `(doc, grant set)` mappings the capture will emit — the
    /// count [`Self::for_each_doc_grant`] enumerates. The ADR-229 producer
    /// writes it as the section header BEFORE streaming the grants (exactly as
    /// the pre-M0.x `iter_doc_grants().len()` did). Cheap: one `doc_class`
    /// DashMap length read.
    ///
    /// This MUST equal the number of callbacks [`Self::for_each_doc_grant`]
    /// fires, or the snapshot's declared count and streamed records diverge.
    /// A doc whose class is (structurally impossibly) missing from
    /// `class_grants` is skipped by BOTH this count and `for_each_doc_grant`,
    /// so they stay in lockstep even in that defensive branch.
    #[must_use]
    pub fn doc_grant_count(&self) -> u64 {
        if self.physical.is_some() {
            return 0;
        }
        // Fast path: every doc in `doc_class` resolves a class (classes are
        // append-only + never removed), so the count is `doc_class.len()`. We
        // still filter defensively for a missing class so the count matches
        // `for_each_doc_grant`'s skip-if-class-missing filter byte-for-byte.
        self.resident()
            .doc_class
            .iter()
            .filter(|e| self.resident().class_grants.contains_key(&e.value().0))
            .count() as u64
    }

    /// ADR-229 checkpoint producer — STREAM every ACL-tagged doc as
    /// `(doc NodeId, grants: &BTreeSet<String>)` through `f`, ONE at a time,
    /// NEVER building a whole-`Vec` and NEVER cloning all grant sets at once.
    /// Resolves each doc's class to its (immutable) grant set and emits a
    /// BORROW of the class's `Arc<BTreeSet<String>>` — the callback serializes
    /// it in place; nothing is retained across iterations. Restore feeds each
    /// mapping back through [`Self::apply_doc_acl_replayed`] (the WAL-replay
    /// entry point that bypasses the durable sink — the grant is already
    /// durable in the checkpoint). A doc whose class is missing from
    /// `class_grants` (structurally impossible — classes are append-only and
    /// never removed) is skipped defensively rather than restored with a widen.
    /// Iteration order is arbitrary; class-ids are process-local and
    /// deliberately NOT captured (restore re-interns). Stops + returns on the
    /// first `f` error (a sink write failure).
    ///
    /// **#1404 M0.x — the freeze-capture bound.** This runs UNDER
    /// `checkpoint_freeze` (`producer.rs:132`). It is the streaming twin of the
    /// (now `#[cfg(test)]`-only) whole-`Vec` `iter_doc_grants`, mirroring the
    /// idempotency-binding / intern streaming: `iter_doc_grants` cloned EVERY
    /// grant set into one `Vec<(NodeId, BTreeSet<String>)>` under the freeze —
    /// O(docs-with-ACLs) whole-in-RAM, the identical sibling. Emitting a borrow
    /// of the interned (`Arc`-shared) grant set keeps the capture's resident
    /// working set to ONE reference at a time. Permission grants are the third
    /// RE-2 owner; they are page-store-backed structurally at M4/M6 (ADR-230
    /// OQ-G), with M0.x streaming as the freeze-capture interim.
    ///
    /// The number of callbacks fired equals [`Self::doc_grant_count`] (same
    /// skip-if-class-missing filter), so the producer can write the count
    /// header first and then stream exactly that many records.
    ///
    /// **#1404 M0.x FIX-D — returns the ACTUAL streamed count** so the producer
    /// can hard-check it against the `doc_grant_count()` header (mismatch →
    /// abort the checkpoint). Under the capture WRITE guard
    /// ([`Self::capture_guard`]) header==streamed deterministically.
    pub fn for_each_doc_grant<F, E>(&self, mut f: F) -> std::result::Result<u64, E>
    where
        F: FnMut(NodeId, &BTreeSet<String>) -> std::result::Result<(), E>,
    {
        if self.physical.is_some() {
            return Ok(0);
        }
        // #1404 M0.x — track the peak grant sets the capture holds
        // SIMULTANEOUSLY, so the permissions capture-peak gate can prove the
        // freeze-capture is bounded. The streaming path holds exactly ONE grant
        // set in-flight (borrow → emit → drop), so the peak stays 1 regardless
        // of the ACL-tagged-doc count. A reverted whole-`Vec` capture would
        // push the peak to N.
        let mut in_flight: u64 = 0;
        let mut peak: u64 = 0;
        let mut streamed: u64 = 0;
        // M5-D3 FIX 4 (#1518 skeptic review) — sorted doc-id capture order.
        // `DashMap` iteration order is nondeterministic, and these grants
        // land byte-for-byte in checkpoint metadata (same INV-M5.24 class
        // as the blob page-image capture, see
        // `blob.rs::sorted_resident_keys`). Only the (8-byte) doc-id KEYS
        // are collected into a `Vec` — the streaming discipline above still
        // holds for the actual grant-set payloads.
        let doc_class = &self.resident().doc_class;
        let mut doc_ids: Vec<u64> = doc_class.iter().map(|e| *e.key()).collect();
        doc_ids.sort_unstable();
        for doc_id in doc_ids {
            let Some(class_entry) = doc_class.get(&doc_id) else {
                continue;
            };
            let doc = NodeId::new(doc_id);
            let class = *class_entry.value();
            drop(class_entry);
            // Resolve the class → its immutable interned grant set. A missing
            // class (structurally impossible) is skipped, exactly as
            // `iter_doc_grants` did — no widen on restore. The `Arc<BTreeSet>`
            // is borrowed for the callback and dropped before the next doc; no
            // whole-`Vec` of cloned grant sets is ever materialized.
            if let Some(g) = self.resident().class_grants.get(&class.0) {
                in_flight += 1;
                peak = peak.max(in_flight);
                f(doc, g.value().as_ref())?;
                in_flight -= 1;
                streamed += 1;
            }
        }
        self.resident()
            .capture_peak_in_flight
            .store(peak, Ordering::Release);
        Ok(streamed)
    }

    /// #1404 M0.x FIX-D — acquire the capture WRITE guard (see the
    /// `capture_lock` field docs). The producer holds it across
    /// `doc_grant_count()` + `for_each_doc_grant()` so no concurrent
    /// `apply_doc_acl`/`revoke_doc` skews header≠stream. The enforcement read
    /// path is untouched.
    pub fn capture_guard(&self) -> Option<parking_lot::RwLockWriteGuard<'_, ()>> {
        self.resident
            .as_deref()
            .map(|resident| resident.capture_lock.write())
    }

    /// #1404 M0.x — the CAPTURE peak-in-flight grant-set count from the last
    /// [`Self::for_each_doc_grant`] pass (the max grant sets the capture held
    /// resident at once). O(1) in the ACL-tagged-doc count for the streaming
    /// path (≤1). Test / observability.
    #[doc(hidden)]
    #[must_use]
    pub fn capture_peak_in_flight(&self) -> u64 {
        self.resident.as_deref().map_or(0, |resident| {
            resident.capture_peak_in_flight.load(Ordering::Acquire)
        })
    }

    /// #1404 M0.x — the whole-`Vec` capture, retained as a `#[cfg(test)]`
    /// ORACLE ONLY (mirroring the M0.5 `append_evicted_supplement` +
    /// `IdempotencyStore::iter_all` whole-`Vec` oracles). The PRODUCTION
    /// capture is [`Self::for_each_doc_grant`], which streams one borrowed
    /// grant set at a time (never a `Vec`, never all clones at once).
    #[cfg(test)]
    #[must_use]
    pub fn iter_doc_grants(&self) -> Vec<(NodeId, BTreeSet<String>)> {
        let mut out = Vec::new();
        let _ = self.for_each_doc_grant::<_, std::convert::Infallible>(|doc, grants| {
            out.push((doc, grants.clone()));
            Ok(())
        });
        out
    }

    /// Budget (PD#5, ADR-226 S1): slow path only — first sighting of a
    /// distinct grant set. One `fetch_add` + one `class_grants` insert
    /// inside the vacant-entry closure + the interning insert itself,
    /// ~200 ns warm. The cache-hit read above and the `is_visible` hot
    /// path are untouched; sustained TPS delta ≈ 0 (interning is
    /// ingest-side, off the read path).
    fn intern_class(&self, grants: BTreeSet<String>) -> AclClassId {
        if let Some(existing) = self.resident().class_intern.get(&grants) {
            return *existing;
        }
        // #1184 (ADR-226 S1, gate CONC-C(a)): the id comes from a
        // dedicated atomic allocator `fetch_add`ed INSIDE the
        // vacant-entry closure — never from a `len()` read taken before
        // the insert. Two writers racing on the SAME new set are
        // arbitrated by `entry` (the closure runs exactly once, under
        // the shard write lock; both observe one id). Two writers
        // interning DISTINCT new sets get distinct ids from the atomic —
        // the old len()-then-insert pattern let both read one `len` and
        // alias two sets to one id, an over-grant (visibility fails
        // open). Mirrors the in-tree allocator at `intern.rs`
        // (`InternTable::intern`).
        //
        // `class_grants` is populated inside the closure, BEFORE the
        // interning entry is published, so a concurrent reader that
        // observes a class id can always resolve its grant set.
        // Lock order is class_intern-shard → class_grants-shard only;
        // no path acquires them in reverse (`effective` iterates
        // `class_grants` without touching `class_intern`).
        let grants = Arc::new(grants);
        *self
            .resident()
            .class_intern
            .entry(grants.as_ref().clone())
            .or_insert_with(|| {
                let raw = self.next_class.fetch_add(1, Ordering::Relaxed);
                assert_ne!(
                    raw,
                    u32::MAX,
                    "more than u32::MAX distinct ACL classes is unreachable at stage-1"
                );
                let id = AclClassId(raw);
                self.resident()
                    .class_grants
                    .insert(id.0, Arc::clone(&grants));
                id
            })
    }

    fn bump(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn untagged_doc_is_invisible_to_everyone() {
        let idx = PermissionIndex::new();
        idx.apply_doc_acl(NodeId::new(1), set(&["alice"]));
        let alice = idx.effective("alice");
        // Node 99 was never tagged: UNCLASSIFIED ⇒ deny (invariant 1).
        assert!(!alice.is_visible(NodeId::new(99)));
        assert!(alice.is_visible(NodeId::new(1)));
    }

    #[test]
    fn direct_grant_and_public_resolve() {
        let idx = PermissionIndex::new();
        idx.apply_doc_acl(NodeId::new(1), set(&["alice"]));
        idx.apply_doc_acl(NodeId::new(2), set(&["bob"]));
        idx.apply_doc_acl(NodeId::new(3), set(&[PUBLIC_PRINCIPAL]));
        idx.apply_doc_acl(NodeId::new(4), set(&["alice", "bob"]));

        let alice = idx.effective("alice");
        assert!(alice.is_visible(NodeId::new(1)));
        assert!(!alice.is_visible(NodeId::new(2)));
        assert!(alice.is_visible(NodeId::new(3)));
        assert!(alice.is_visible(NodeId::new(4)));

        let bob = idx.effective("bob");
        assert!(!bob.is_visible(NodeId::new(1)));
        assert!(bob.is_visible(NodeId::new(2)));
        assert!(bob.is_visible(NodeId::new(3)));
        assert!(bob.is_visible(NodeId::new(4)));

        // A principal the index has never seen still gets PUBLIC docs
        // and nothing else.
        let mallory = idx.effective("mallory");
        assert!(!mallory.is_visible(NodeId::new(1)));
        assert!(!mallory.is_visible(NodeId::new(2)));
        assert!(mallory.is_visible(NodeId::new(3)));
    }

    #[test]
    fn empty_grant_set_denies_all_but_is_tagged() {
        let idx = PermissionIndex::new();
        idx.apply_doc_acl(NodeId::new(7), BTreeSet::new());
        assert_eq!(idx.tagged_docs(), 1);
        assert!(!idx.effective("alice").is_visible(NodeId::new(7)));
        assert!(!idx.effective(PUBLIC_PRINCIPAL).is_visible(NodeId::new(7)));
    }

    #[test]
    fn revocation_applies_at_next_resolution() {
        let idx = PermissionIndex::new();
        idx.apply_doc_acl(NodeId::new(1), set(&["alice", "bob"]));
        assert!(idx.effective("bob").is_visible(NodeId::new(1)));

        // Revoke bob by re-applying the narrowed set.
        idx.apply_doc_acl(NodeId::new(1), set(&["alice"]));
        assert!(!idx.effective("bob").is_visible(NodeId::new(1)));
        assert!(idx.effective("alice").is_visible(NodeId::new(1)));

        // Revoke-by-removal: UNCLASSIFIED ⇒ nobody.
        idx.revoke_doc(NodeId::new(1));
        assert!(!idx.effective("alice").is_visible(NodeId::new(1)));
    }

    #[test]
    fn stale_snapshot_never_widens_access() {
        let idx = PermissionIndex::new();
        idx.apply_doc_acl(NodeId::new(1), set(&["alice", "bob"]));
        let bob_stale = idx.effective("bob");
        assert!(bob_stale.is_visible(NodeId::new(1)));

        // Mutations AFTER bob's snapshot: bob revoked from doc 1; a
        // brand-new doc 2 granted to bob.
        idx.apply_doc_acl(NodeId::new(1), set(&["alice"]));
        idx.apply_doc_acl(NodeId::new(2), set(&["bob"]));

        // Narrow direction: the stale snapshot DENIES doc 1 (its old
        // class id no longer resolves from doc_class — classes are
        // immutable, the doc was remapped) and DENIES doc 2 (class
        // absent from the stale bitset). Staleness can hold a
        // revocation only until re-resolution... but here even the
        // stale snapshot already denies: remap moved the doc to a
        // class the snapshot never allowed.
        assert!(!bob_stale.is_visible(NodeId::new(1)));
        assert!(!bob_stale.is_visible(NodeId::new(2)));

        // Fresh resolution sees the new world.
        let bob_fresh = idx.effective("bob");
        assert!(!bob_fresh.is_visible(NodeId::new(1)));
        assert!(bob_fresh.is_visible(NodeId::new(2)));
    }

    #[test]
    fn class_interning_dedupes_identical_grant_sets() {
        let idx = PermissionIndex::new();
        for n in 0..100 {
            idx.apply_doc_acl(NodeId::new(n), set(&["alice", "team"]));
        }
        for n in 100..150 {
            idx.apply_doc_acl(NodeId::new(n), set(&["bob"]));
        }
        assert_eq!(idx.tagged_docs(), 150);
        assert_eq!(idx.class_count(), 2);
    }

    /// #1184 regression — ADR-226 §3 gate CONC-C(a): 16 writers
    /// interning DISTINCT new grant-sets concurrently must never alias
    /// two sets to one `AclClassId`. RED on revert: the pre-fix
    /// len()-then-insert allocation lets two threads read one `len` and
    /// mint one id for two distinct sets — an over-grant (a principal
    /// from set A becomes visible on set B's docs).
    ///
    /// Sizing: release = 16×64_000 ≈ 1.02M interns (≥ the CONC-C(a)
    /// 10⁶ bar, ~2 s); debug = 16×8_000 (CI-friendly). The rc gate run
    /// is `cargo test -p arcgraph-storage --release intern_storm`.
    #[test]
    fn intern_storm_distinct_sets_never_alias() {
        const THREADS: usize = 16;
        #[cfg(debug_assertions)]
        const PER_THREAD: usize = 8_000;
        #[cfg(not(debug_assertions))]
        const PER_THREAD: usize = 64_000;

        let idx = PermissionIndex::new();
        let ids: Vec<Vec<AclClassId>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..THREADS)
                .map(|t| {
                    let idx = &idx;
                    s.spawn(move || {
                        (0..PER_THREAD)
                            .map(|i| {
                                let grants: BTreeSet<String> =
                                    std::iter::once(format!("grp-{t}-{i}")).collect();
                                idx.intern_class(grants)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // (a) No two distinct sets share an id.
        let mut seen = std::collections::HashSet::with_capacity(THREADS * PER_THREAD);
        for id in ids.iter().flatten() {
            assert!(
                seen.insert(*id),
                "aliased {id:?}: two distinct grant-sets minted one class id (#1184)"
            );
        }
        // (b) Exactly one class per distinct set — no over- or
        // under-allocation.
        assert_eq!(idx.class_count(), THREADS * PER_THREAD);
        // (c) Publish-after-populate: every interned class id resolves
        // to its grant set (no id observable with a missing set).
        for entry in &idx.resident().class_intern {
            assert!(
                idx.resident().class_grants.contains_key(&entry.value().0),
                "class id {:?} published without its grant set",
                entry.value()
            );
        }
    }

    #[test]
    fn cache_hits_until_generation_moves() {
        let idx = PermissionIndex::new();
        idx.apply_doc_acl(NodeId::new(1), set(&["alice"]));
        let a1 = idx.effective("alice");
        let a2 = idx.effective("alice");
        assert!(Arc::ptr_eq(&a1, &a2), "same generation ⇒ cached snapshot");

        idx.apply_doc_acl(NodeId::new(2), set(&["alice"]));
        let a3 = idx.effective("alice");
        assert!(!Arc::ptr_eq(&a1, &a3), "generation bump ⇒ fresh resolution");
        assert!(a3.is_visible(NodeId::new(2)));
    }

    #[test]
    fn bitset_basics() {
        let mut b = ClassBitset::default();
        assert!(!b.contains(0));
        b.insert(0);
        b.insert(63);
        b.insert(64);
        b.insert(200);
        assert!(b.contains(0) && b.contains(63) && b.contains(64) && b.contains(200));
        assert!(!b.contains(1) && !b.contains(199));
        assert_eq!(b.count(), 4);
    }

    #[test]
    fn principal_node_dedup_roundtrip() {
        let idx = PermissionIndex::new();
        assert!(idx.principal_node("alice").is_none());
        idx.record_principal_node("alice", NodeId::new(42));
        assert_eq!(idx.principal_node("alice"), Some(NodeId::new(42)));
    }

    // ─── #1221 (ADR-218) — AclWalSink write-through wiring ────────────

    /// Records every durify call so tests can assert what the index
    /// staged for the WAL.
    #[derive(Debug, Default)]
    struct RecordingSink {
        applies: std::sync::Mutex<Vec<(u64, BTreeSet<String>)>>,
        revokes: std::sync::Mutex<Vec<u64>>,
    }

    impl AclWalSink for RecordingSink {
        fn durify_apply(&self, doc: NodeId, grants: &BTreeSet<String>) {
            self.applies
                .lock()
                .unwrap()
                .push((doc.raw(), grants.clone()));
        }
        fn durify_revoke(&self, doc: NodeId) {
            self.revokes.lock().unwrap().push(doc.raw());
        }
    }

    #[test]
    fn no_sink_by_default_is_in_memory_only() {
        let idx = PermissionIndex::new();
        assert!(!idx.has_wal_sink());
        // Write-through still works (in-memory), it just doesn't durify.
        idx.apply_doc_acl(NodeId::new(1), set(&["alice"]));
        assert!(idx.effective("alice").is_visible(NodeId::new(1)));
    }

    #[test]
    fn write_through_durifies_apply_and_revoke_via_sink() {
        let sink = Arc::new(RecordingSink::default());
        let idx = PermissionIndex::new();
        idx.set_wal_sink(Arc::clone(&sink) as Arc<dyn AclWalSink>);
        assert!(idx.has_wal_sink());

        idx.apply_doc_acl(NodeId::new(1), set(&["alice", "bob"]));
        idx.revoke_doc(NodeId::new(1));

        let applies = sink.applies.lock().unwrap();
        assert_eq!(applies.len(), 1);
        assert_eq!(applies[0].0, 1);
        assert_eq!(applies[0].1, set(&["alice", "bob"]));
        let revokes = sink.revokes.lock().unwrap();
        assert_eq!(*revokes, vec![1u64]);
    }

    #[test]
    fn replayed_entry_points_bypass_the_sink() {
        // The replay entry points re-drive enforcement WITHOUT durifying
        // (the op is already in the WAL — re-staging would double-log it).
        let sink = Arc::new(RecordingSink::default());
        let idx = PermissionIndex::new();
        idx.set_wal_sink(Arc::clone(&sink) as Arc<dyn AclWalSink>);

        idx.apply_doc_acl_replayed(NodeId::new(1), set(&["alice"]));
        idx.revoke_doc_replayed(NodeId::new(2));

        // Enforcement applied in-memory...
        assert!(idx.effective("alice").is_visible(NodeId::new(1)));
        // ...but the sink was NOT touched.
        assert!(sink.applies.lock().unwrap().is_empty());
        assert!(sink.revokes.lock().unwrap().is_empty());
    }

    // ─────────────────────────────────────────────────────────────────
    // #1404 M0.x — freeze-capture streaming (the 3rd RE-2 owner sibling)
    // ─────────────────────────────────────────────────────────────────

    /// `for_each_doc_grant` streams EXACTLY `doc_grant_count()` records, with
    /// the SAME skip-if-class-missing filter — no wire-length drift (the
    /// producer writes the count header before the records, so a drift would
    /// corrupt the snapshot).
    #[test]
    fn for_each_doc_grant_count_matches_streamed_emits() {
        let idx = PermissionIndex::new();
        // Distinct + shared grant sets (interning) across many docs.
        for i in 0..200u64 {
            let grants = if i % 3 == 0 {
                set(&["alice", "bob"])
            } else if i % 3 == 1 {
                set(&["carol"])
            } else {
                set(&[PUBLIC_PRINCIPAL])
            };
            idx.apply_doc_acl(NodeId::new(i + 1), grants);
        }
        let declared = idx.doc_grant_count();
        let mut streamed = 0u64;
        idx.for_each_doc_grant::<_, std::convert::Infallible>(|_, _| {
            streamed += 1;
            Ok(())
        })
        .expect("infallible");
        assert_eq!(
            streamed, declared,
            "doc_grant_count() ({declared}) != streamed emit count ({streamed}) — wire drift",
        );
        assert_eq!(streamed, 200, "every tagged doc must be emitted once");
    }

    /// M5-D3 FIX 4 (#1518 skeptic review) — `for_each_doc_grant`'s capture
    /// order must be a pure function of the resident doc-id key set
    /// (sorted), not `DashMap` iteration order (a function of insertion
    /// history via shard/bucket layout — the same nondeterminism class the
    /// blob page-image capture was pinned for, see
    /// `blob.rs::sorted_resident_keys`). Two independently-built indexes
    /// holding the IDENTICAL doc→grant mappings, applied in DIFFERENT
    /// orders, must stream in IDENTICAL order.
    ///
    /// RED-on-revert: replace the sorted-key capture with a raw
    /// `self.resident().doc_class.iter()` walk (the pre-fix code) — this
    /// test then fails intermittently (most runs, on a large enough doc
    /// set), since two indexes built from the same mappings in different
    /// application order generally land in different DashMap bucket order.
    #[test]
    fn for_each_doc_grant_capture_order_is_sorted_not_insertion_or_shard_order() {
        let forward_idx = PermissionIndex::new();
        let reverse_idx = PermissionIndex::new();
        let docs: Vec<(NodeId, BTreeSet<String>)> = (0..200u64)
            .map(|i| {
                let grants = if i % 3 == 0 {
                    set(&["alice", "bob"])
                } else if i % 3 == 1 {
                    set(&["carol"])
                } else {
                    set(&[PUBLIC_PRINCIPAL])
                };
                (NodeId::new(i + 1), grants)
            })
            .collect();

        for (doc, grants) in &docs {
            forward_idx.apply_doc_acl_replayed(*doc, grants.clone());
        }
        for (doc, grants) in docs.iter().rev() {
            reverse_idx.apply_doc_acl_replayed(*doc, grants.clone());
        }

        let mut forward_stream = Vec::new();
        forward_idx
            .for_each_doc_grant::<_, std::convert::Infallible>(|doc, grants| {
                forward_stream.push((doc, grants.clone()));
                Ok(())
            })
            .unwrap();
        let mut reverse_stream = Vec::new();
        reverse_idx
            .for_each_doc_grant::<_, std::convert::Infallible>(|doc, grants| {
                reverse_stream.push((doc, grants.clone()));
                Ok(())
            })
            .unwrap();

        assert_eq!(
            forward_stream.len(),
            docs.len(),
            "sanity: every doc streamed"
        );
        assert_eq!(
            forward_stream, reverse_stream,
            "DEFECT: for_each_doc_grant's capture order depends on \
             application history (DashMap shard/bucket layout), not a sort \
             — two indexes with identical doc grants streamed in DIFFERENT \
             order"
        );
        let mut sorted = forward_stream.clone();
        sorted.sort_unstable_by_key(|(doc, _)| *doc);
        assert_eq!(
            forward_stream, sorted,
            "capture order must be exactly sorted by doc NodeId"
        );
    }

    /// GATE (permissions capture-peak) — the CAPTURE's peak-resident grant sets
    /// is O(1) in the ACL-tagged-doc count. This kills the 3rd RE-2 whole-in-RAM
    /// sibling: `for_each_doc_grant` must NOT clone all grant sets into a `Vec`
    /// under `checkpoint_freeze`. Measured at 2 sizes (64 vs 1024 tagged docs)
    /// via the PRODUCTION `capture_peak_in_flight` counter; RED-on-revert = the
    /// `#[cfg(test)]` whole-`Vec` `iter_doc_grants` oracle (peak grows ~16×).
    #[test]
    fn gate_permissions_capture_peak_is_o1_in_doc_count() {
        fn tagged(n: u64) -> PermissionIndex {
            let idx = PermissionIndex::new();
            for i in 0..n {
                // Vary grant sets so it isn't a single interned class (still
                // bounded classes ≪ docs, but the CAPTURE walks per-doc).
                let grants = set(&[if i % 2 == 0 { "alice" } else { "bob" }]);
                idx.apply_doc_acl(NodeId::new(i + 1), grants);
            }
            idx
        }

        // STREAMING peak = the store's OWN `capture_peak_in_flight` counter
        // (max grant sets the PRODUCTION capture held at once). Streaming holds
        // ≤1 (borrow → emit → drop), so it stays 1 regardless of N.
        fn streaming_peak(idx: &PermissionIndex) -> u64 {
            let mut count = 0u64;
            idx.for_each_doc_grant::<_, std::convert::Infallible>(|_, _| {
                count += 1;
                Ok(())
            })
            .expect("infallible");
            assert_eq!(count, idx.doc_grant_count());
            idx.capture_peak_in_flight()
        }

        // WHOLE-`Vec` peak (the reverted term) = ALL N grant sets cloned at once.
        fn whole_vec_peak(idx: &PermissionIndex) -> u64 {
            idx.iter_doc_grants().len() as u64
        }

        let small_n = 64u64;
        let large_n = 1024u64; // 16× larger
        let small = tagged(small_n);
        let large = tagged(large_n);
        assert_eq!(small.doc_grant_count(), small_n);
        assert_eq!(large.doc_grant_count(), large_n);

        // ── PRODUCTION streaming capture: FLAT peak (O(1)) across 16× size ──
        let s_small = streaming_peak(&small);
        let s_large = streaming_peak(&large);
        assert_eq!(
            s_small, 1,
            "streaming capture peak must be 1 grant set (O(1)), got {s_small}"
        );
        assert_eq!(
            s_large, 1,
            "streaming capture peak must be 1 grant set (O(1)), got {s_large}"
        );
        let s_ratio = s_large / s_small.max(1);
        assert_eq!(
            s_ratio, 1,
            "streaming peak must be FLAT (ratio 1), got {s_ratio}"
        );

        // ── RED-on-revert: whole-`Vec` capture peak GROWS ~16× with N ──
        let w_small = whole_vec_peak(&small);
        let w_large = whole_vec_peak(&large);
        assert_eq!(w_small, small_n);
        assert_eq!(w_large, large_n);
        let w_ratio = w_large / w_small.max(1);
        assert_eq!(
            w_ratio, 16,
            "whole-Vec capture peak ratio must be ~16× (the reverted unbounded term), got {w_ratio}",
        );

        println!(
            "GATE permissions capture-peak grant-sets — streaming(PROD): n={small_n}→{s_small}, n={large_n}→{s_large} (ratio {s_ratio}×, O(1)); \
             whole-Vec[REVERTED]: n={small_n}→{w_small}, n={large_n}→{w_large} (ratio {w_ratio}×, O(N))",
        );
    }
}
