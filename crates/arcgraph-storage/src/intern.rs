//! M2-32 — label/type interning.
//!
//! Every label and relationship type name in a workload resolves to a
//! [`StringId`] through a per-tenant intern table. The bidirectional
//! map lets `create_node`/`create_rel` callers write numeric
//! [`LabelId`]/[`TypeId`] into records and still recover the original
//! name for audit/query tools.
//!
//! # Tenancy
//!
//! (§A.) Every lookup is keyed by `(TenantId, …)`. Two tenants may use
//! the same name and receive **different** [`StringId`]s; cross-tenant
//! isolation is a hard invariant, verified by the
//! `intern_allocates_new_id_per_tenant` test.
//!
//! # Persistence (§K, ADR-018 addendum; P0 #776)
//!
//! The table is in-memory at runtime, but its bindings are durable.
//! Each intern **without durable proof** (v2 M2 A4 — the
//! durable-logged-set protocol; see [`intern_logged`]) is appended to
//! the WAL via [`intern_logged`] / [`intern_label_logged`] /
//! [`intern_type_logged`] using the
//! [`WalRecordType::InternString`] variant. On restart,
//! [`crate::wal::recover_from_wal`] replays those records back into the
//! served table via [`InternTable::intern_install`] (wired through
//! [`crate::wal::PageStoreTarget::with_intern_table`]), so
//! `graph.schema` shows real label/rel-type names and typed queries
//! resolve after a durable `--data` restart. Before this fix (#776) the
//! recovery replay was a no-op and the production write path never
//! logged interns, so names came back as synthetic `label:N` / `type:N`
//! and typed queries failed -32005.
//!
//! # Budget
//!
//! Fast-path lookup (cache hit): one DashMap read ≈ 50 ns amortized.
//! Slow-path allocation: one DashMap entry-insert + one AtomicU32
//! fetch_add + one reverse-map insert. No global lock — DashMap's
//! sharding serializes per-bucket; different shards proceed in
//! parallel, so a mixed workload across many names scales with
//! shard count.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use arcgraph_core::{ArcGraphError, LabelId, Lsn, Result, StringId, TenantId, TypeId};
use dashmap::{DashMap, DashSet};

use crate::owner_index::str_hash_56;
use crate::owner_row::{
    InternOwnerValue, OWNER_ALLOCATOR_MARKER_ID, OwnerAllocatorMarker, OwnerRowClass,
    OwnerRowError, OwnerRowRegistry,
};
use crate::wal::{AllocatorAdvance, AllocatorKind, WalHandle, WalRecordType};

/// Reserved [`StringId`] sentinel. The per-tenant allocator starts at
/// 1, so an id of `0` never corresponds to an interned name.
pub const STRINGID_SENTINEL: StringId = StringId::ZERO;

/// Per-tenant (String ↔ StringId) table.
#[derive(Debug, Default)]
struct ResidentInternOwners {
    /// Forward map: `(tenant, name)` → [`StringId`].
    forward: DashMap<(TenantId, String), StringId>,
    /// Reverse map: `(tenant, StringId)` → `Arc<String>`.
    reverse: DashMap<(TenantId, StringId), Arc<String>>,
    /// Pre-M4 checkpoint capture exclusion.
    capture_lock: parking_lot::RwLock<()>,
    /// Pre-M4 durable-proof set.
    logged: DashSet<(TenantId, StringId)>,
}

#[derive(Debug)]
pub struct InternTable {
    /// M4 authoritative owner. The legacy maps remain empty when this is set.
    physical: Option<Arc<OwnerRowRegistry>>,
    /// Pre-M4 owner bundle. This is `None` for a page-backed facade, so the
    /// process does not even construct the string-cardinality DashMaps after
    /// the generation swap.
    resident: Option<Box<ResidentInternOwners>>,
    /// Allocation + same-name miss collapse. One bounded mutex replaces
    /// record-cardinality resident maps on the physical path.
    physical_write: parking_lot::Mutex<()>,
    /// Per-tenant allocator. `0` is the sentinel; the first allocated
    /// id is `1`.
    next_id: DashMap<TenantId, AtomicU32>,
}

impl Default for InternTable {
    fn default() -> Self {
        Self {
            physical: None,
            resident: Some(Box::default()),
            physical_write: parking_lot::Mutex::new(()),
            next_id: DashMap::new(),
        }
    }
}

impl InternTable {
    /// Empty table. Per-tenant state is created lazily on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Open the M4 page-backed intern facade and seed each tenant allocator
    /// from its direct-address durable marker using monotone `fetch_max`.
    pub fn page_backed(owner: Arc<OwnerRowRegistry>) -> std::result::Result<Self, OwnerRowError> {
        let table = Self {
            physical: Some(Arc::clone(&owner)),
            resident: None,
            physical_write: parking_lot::Mutex::new(()),
            next_id: DashMap::new(),
        };
        for tenant in owner.tenants() {
            let high_water = match owner.read_logical(
                tenant,
                OwnerRowClass::InternedString,
                OWNER_ALLOCATOR_MARKER_ID,
            )? {
                Some(logical) => {
                    let marker = OwnerAllocatorMarker::decode(&logical)?;
                    if marker.kind != AllocatorKind::InternString.as_byte() {
                        return Err(OwnerRowError::InvalidEnvelope(format!(
                            "intern allocator marker has kind {}",
                            marker.kind
                        )));
                    }
                    u32::try_from(marker.high_water).map_err(|_| {
                        OwnerRowError::InvalidEnvelope(
                            "intern allocator marker exceeds u32".to_owned(),
                        )
                    })?
                }
                None => 0,
            };
            table
                .next_id
                .entry(tenant)
                .or_insert_with(|| AtomicU32::new(0))
                .fetch_max(high_water, Ordering::AcqRel);
        }
        Ok(table)
    }

    /// Recovery seed (ADR-034 D-1, Lemma-I3): advance the per-tenant string
    /// allocator from a replayed [`AllocatorAdvance`] so a post-crash
    /// allocation cannot reuse a [`StringId`] a pre-fault commit already
    /// durably consumed.
    ///
    /// Bootstrap seeds this counter from the `OWNER_ALLOCATOR_MARKER_ID` row
    /// read off the last **checkpointed** page image — i.e. BEFORE WAL replay.
    /// Commits made after that checkpoint advance the durable marker via their
    /// v10 owner-page delta, but the in-RAM counter only learns about them by
    /// replaying their `AllocatorAdvance`. Dropping that advance (the pre-fix
    /// behaviour) left the counter pinned at the checkpoint high-water and
    /// reissued ids across a crash-restart.
    ///
    /// `new_high_water` is the LAST allocated id, and `next_id` holds exactly
    /// that — allocation is `prev = fetch_add(1); id = prev + 1`, so after
    /// handing out id X the counter reads X. Seed with `new_high_water` and NO
    /// `+1`: adding one would skip a `StringId` on every replayed advance.
    ///
    /// This matches [`Self::page_backed`] (which seeds `fetch_max(marker
    /// .high_water)`) and the node/rel allocators
    /// ([`crate::crud::CrudStore::seed_node_from_advance`]). It deliberately
    /// does NOT match [`crate::permissions::PermissionIndex::seed_class_allocator`],
    /// whose counter holds the NEXT id rather than the last one (`raw =
    /// fetch_add(1)` uses `raw` itself as the class id) and therefore does need
    /// the `+1`.
    ///
    /// `fetch_max` makes this idempotent under double-replay and safe against
    /// out-of-order delivery.
    pub fn seed_string_allocator(&self, tenant: TenantId, new_high_water: u64) {
        let last_allocated = u32::try_from(new_high_water).unwrap_or(u32::MAX);
        self.next_id
            .entry(tenant)
            .or_insert_with(|| AtomicU32::new(0))
            .fetch_max(last_allocated, Ordering::AcqRel);
    }

    /// True only for the post-M4 facade.
    #[must_use]
    pub fn is_page_backed(&self) -> bool {
        self.physical.is_some()
    }

    /// Residency-census cardinalities for the two bidirectional maps and the
    /// legacy per-name durable-proof set.
    #[must_use]
    pub fn resident_map_cardinalities(&self) -> [usize; 3] {
        self.resident.as_ref().map_or([0, 0, 0], |resident| {
            [
                resident.forward.len(),
                resident.reverse.len(),
                resident.logged.len(),
            ]
        })
    }

    /// Structural census: page-backed facades do not contain a legacy owner
    /// bundle whose maps could accidentally be repopulated.
    #[must_use]
    pub fn has_resident_owner_maps(&self) -> bool {
        self.resident.is_some()
    }

    fn resident(&self) -> &ResidentInternOwners {
        match self.resident.as_deref() {
            Some(resident) => resident,
            None => unreachable!("resident intern path selected for page-backed facade"),
        }
    }

    /// Intern `s` for `tenant` and report whether the id is freshly
    /// allocated.
    ///
    /// Idempotent: the same `(tenant, s)` always returns the same
    /// [`StringId`]. The `was_new` flag is `true` only for the caller
    /// that actually installed the binding; losers of a race see
    /// `false`.
    ///
    /// (§A) Same string under two tenants gets two **different** ids.
    ///
    /// # Fail-closed
    ///
    /// On the M4 page-backed arm this is a durable publish and can fail (owner
    /// I/O, a forward-index run retired mid-scan, a budget cap). It MUST NOT
    /// invent an id: returning [`STRINGID_SENTINEL`] — the pre-fix behaviour —
    /// hands the caller the *reserved* id as though the string were legitimately
    /// interned, so the binding is written under id 0 and every later resolve of
    /// a real name collides with it. That is silent data corruption, and a
    /// transient forward-index miss was enough to trigger it. Errors propagate.
    pub fn intern_is_new(
        &self,
        tenant: TenantId,
        s: &str,
    ) -> std::result::Result<(StringId, bool), OwnerRowError> {
        if self.physical.is_some() {
            return self.try_intern_physical(tenant, s);
        }
        // Fast path: already interned. Constructing the lookup key
        // requires one String allocation — DashMap's key type is
        // owned, not Borrow-polymorphic. At ~50 ns per allocation and
        // a ~100 M ops/s ceiling that is well inside the 5 K TPS
        // budget.
        if let Some(existing) = self.resident().forward.get(&(tenant, s.to_owned())) {
            return Ok((*existing.value(), false));
        }
        // #1404 M0.x FIX-D — a NEW binding is about to be added; take the
        // capture-exclusion READ guard so this insert can't interleave a
        // checkpoint capture's count+stream (the two-pass skew). Concurrent
        // with other interns; blocks only during a capture WRITE guard. The
        // already-interned fast path above returned WITHOUT the guard (it is
        // count-neutral + the hot lookup case), so the lock is off the read
        // path.
        let _install = self.resident().capture_lock.read();
        // Re-check under the guard: a racing intern of the same key may have
        // installed it while we took the guard (the fast-path check is
        // guard-free), so a second lookup avoids a redundant allocate.
        if let Some(existing) = self.resident().forward.get(&(tenant, s.to_owned())) {
            return Ok((*existing.value(), false));
        }

        // Slow path: entry-guarded insert. The DashMap entry API
        // serializes concurrent writers to the same shard, so racing
        // inserts of the same (tenant, s) collapse onto the first
        // winner.
        let mut was_new = false;
        let id = *self
            .resident()
            .forward
            .entry((tenant, s.to_owned()))
            .or_insert_with(|| {
                was_new = true;
                let allocator = self
                    .next_id
                    .entry(tenant)
                    .or_insert_with(|| AtomicU32::new(0));
                let prev = allocator.fetch_add(1, Ordering::AcqRel);
                // saturating_add keeps the sentinel clear even under
                // pathological overflow; the caller would observe a
                // duplicate id long before reaching u32::MAX here, at
                // which point the intern table is considered
                // exhausted and the caller should error.
                let fresh = StringId::new(prev.saturating_add(1));
                self.resident()
                    .reverse
                    .insert((tenant, fresh), Arc::new(s.to_owned()));
                fresh
            })
            .value();

        Ok((id, was_new))
    }

    /// Intern `s` for `tenant`. Same-tenant, same-string always
    /// returns the same id.
    ///
    /// Fallible for the same reason as [`Self::intern_is_new`]: on the M4
    /// page-backed arm this is a durable publish, and a failure must never be
    /// laundered into the reserved [`STRINGID_SENTINEL`].
    pub fn intern(
        &self,
        tenant: TenantId,
        s: &str,
    ) -> std::result::Result<StringId, OwnerRowError> {
        self.intern_is_new(tenant, s).map(|(id, _)| id)
    }

    /// **RC-3 (secondary-property-index BLOCKING prereq, #1366)** —
    /// non-inserting read-only lookup of `(tenant, s)`.
    ///
    /// Returns `Some(id)` iff `s` was ALREADY interned under `tenant`;
    /// returns `None` **without allocating a fresh id** when it was
    /// not. Unlike [`Self::intern`] / [`Self::intern_is_new`] (which
    /// both insert on a miss), this method never mutates the table.
    ///
    /// # Why this exists
    ///
    /// A property-index *lookup* — `MATCH (n:User {email: "nobody@x"})`
    /// — must map the query string to its `StringId` to build the
    /// index key. Routing that through [`Self::intern`] would
    /// **insert-intern an arbitrary query string on the read path**:
    /// unbounded intern-table growth driven by query traffic, plus a
    /// read path mutating shared durable-adjacent state (the reverse
    /// map is WAL-logged for new interns via [`intern_logged`]). A
    /// `probe` miss is instead a *proof of an empty result set* — no
    /// node has ever carried that value, so no index entry can exist —
    /// and the caller returns zero rows without ever touching the
    /// table.
    ///
    /// Budget: one DashMap read on `forward` (≈ 50 ns amortized, same
    /// as the [`Self::intern_is_new`] fast path), one transient
    /// `String` key allocation. No slow-path, no allocator bump, no
    /// reverse-map write, no WAL append.
    ///
    /// (§A) Tenant-scoped: a value interned under a different tenant is
    /// invisible here, matching [`Self::intern`]'s per-tenant
    /// namespace.
    /// REMOVED (fail-open): the infallible `probe` mapped an owner-store I/O
    /// error to `None`, which a caller reads as "this name was never interned"
    /// — a silently dropped projection or a label/type filter that matches zero
    /// rows. A transient forward-index miss was enough to produce a wrong query
    /// answer. Use [`Self::try_probe`], which propagates.
    ///
    /// (Intentionally left as a doc stub rather than a deprecated shim: there is
    /// no safe infallible value to return, so the API must not offer one.)
    #[doc(hidden)]
    fn _probe_removed_use_try_probe() {}

    /// Resolve `id` to its string. Returns `None` if `id` was never
    /// interned under `tenant` (including the sentinel
    /// [`STRINGID_SENTINEL`]).
    /// Resolve `id` to its string. `None` if `id` was never interned.
    ///
    /// # Known fail-open (tracked in #1493) — reverse *display* lookup only
    ///
    /// On the page-backed arm an owner-store I/O error is still laundered into
    /// `None`, which the callers render as a synthetic `label:N` / `type:N`
    /// name. That is a degraded display value, not a wrong filter and not a
    /// corrupt write — and `None` is also the legitimate "never interned"
    /// answer here, so the two are indistinguishable in this signature.
    ///
    /// The two fail-open paths that could produce a WRONG ANSWER or a CORRUPT
    /// WRITE are fixed: [`Self::intern_is_new`] no longer invents
    /// [`STRINGID_SENTINEL`], and the forward probe is now the fail-closed
    /// [`Self::try_probe`] (a swallowed probe error silently dropped a
    /// projection or made a label filter match zero rows).
    ///
    /// Closing this one means threading `Result` through the label-rendering
    /// closures in `arcgraph-mcp::storage::adapters`; deliberately deferred out
    /// of M4 Slice-3b-2 rather than left silent. Prefer [`Self::try_resolve`]
    /// in new code.
    #[must_use]
    pub fn resolve(&self, tenant: TenantId, id: StringId) -> Option<Arc<String>> {
        if self.physical.is_some() {
            return self.try_resolve(tenant, id).unwrap_or_else(|error| {
                tracing::error!(%error, "M4 intern reverse lookup failed (#1493)");
                None
            });
        }
        self.resident()
            .reverse
            .get(&(tenant, id))
            .map(|e| Arc::clone(e.value()))
    }

    /// Typed exact forward lookup for durable serving.
    pub fn try_probe(
        &self,
        tenant: TenantId,
        s: &str,
    ) -> std::result::Result<Option<StringId>, OwnerRowError> {
        let Some(owner) = self.physical.as_ref() else {
            return Ok(self
                .resident()
                .forward
                .get(&(tenant, s.to_owned()))
                .map(|existing| *existing.value()));
        };
        let found = owner.find_verified(
            tenant,
            OwnerRowClass::InternedString,
            str_hash_56(s),
            |_id, logical| InternOwnerValue::decode(logical).is_ok_and(|value| value.name == s),
        )?;
        found
            .map(|(id, _)| {
                u32::try_from(id)
                    .map(StringId::new)
                    .map_err(|_| OwnerRowError::InvalidEnvelope("StringId exceeds u32".to_owned()))
            })
            .transpose()
    }

    /// Typed direct-address reverse lookup.
    pub fn try_resolve(
        &self,
        tenant: TenantId,
        id: StringId,
    ) -> std::result::Result<Option<Arc<String>>, OwnerRowError> {
        if id == STRINGID_SENTINEL {
            return Ok(None);
        }
        let Some(owner) = self.physical.as_ref() else {
            return Ok(self
                .resident()
                .reverse
                .get(&(tenant, id))
                .map(|entry| Arc::clone(entry.value())));
        };
        let Some(logical) =
            owner.read_logical(tenant, OwnerRowClass::InternedString, u64::from(id.raw()))?
        else {
            return Ok(None);
        };
        Ok(Some(Arc::new(InternOwnerValue::decode(&logical)?.name)))
    }

    fn try_intern_physical(
        &self,
        tenant: TenantId,
        s: &str,
    ) -> std::result::Result<(StringId, bool), OwnerRowError> {
        let owner = self.physical.as_ref().ok_or_else(|| {
            OwnerRowError::InvalidEnvelope("physical intern owner is unavailable".to_owned())
        })?;
        if let Some(id) = self.try_probe(tenant, s)? {
            return Ok((id, false));
        }
        let _write = self.physical_write.lock();
        if let Some(id) = self.try_probe(tenant, s)? {
            return Ok((id, false));
        }
        let previous = self
            .next_id
            .entry(tenant)
            .or_insert_with(|| AtomicU32::new(0))
            .fetch_add(1, Ordering::AcqRel);
        let raw = previous.checked_add(1).ok_or_else(|| {
            OwnerRowError::InvalidEnvelope("StringId allocator exhausted".to_owned())
        })?;
        if u64::from(raw) >= OWNER_ALLOCATOR_MARKER_ID {
            return Err(OwnerRowError::InvalidEnvelope(
                "StringId exceeds direct owner capacity".to_owned(),
            ));
        }
        let id = StringId::new(raw);
        let logical = InternOwnerValue { name: s.to_owned() }.encode()?;
        let row = owner.prepare_indexed_logical_row(
            tenant,
            OwnerRowClass::InternedString,
            u64::from(raw),
            str_hash_56(s),
            &logical,
        )?;
        let marker = owner.prepare_direct_logical_row(
            tenant,
            OwnerRowClass::InternedString,
            OWNER_ALLOCATOR_MARKER_ID,
            &OwnerAllocatorMarker {
                kind: AllocatorKind::InternString.as_byte(),
                high_water: u64::from(raw),
            }
            .encode(),
        )?;
        owner.commit_rows_with_allocator_advances(
            tenant,
            [row, marker],
            [AllocatorAdvance {
                tenant,
                kind: AllocatorKind::InternString,
                new_high_water: u64::from(raw),
            }],
        )?;
        Ok((id, true))
    }

    /// Alias for label interning. The returned [`LabelId`] shares the
    /// [`StringId`] numeric space under `tenant`; `intern_label` and
    /// `intern_type` calls for the same tenant do **not** share an id
    /// namespace with each other per-name, because all three funnel
    /// through the same allocator — the intent is that label and type
    /// names are drawn from disjoint vocabularies at the schema
    /// layer.
    pub fn intern_label(
        &self,
        tenant: TenantId,
        name: &str,
    ) -> std::result::Result<LabelId, OwnerRowError> {
        self.intern(tenant, name).map(|id| LabelId::new(id.raw()))
    }

    /// Alias for type interning. See [`Self::intern_label`] re: the
    /// shared allocator.
    pub fn intern_type(
        &self,
        tenant: TenantId,
        name: &str,
    ) -> std::result::Result<TypeId, OwnerRowError> {
        self.intern(tenant, name).map(|id| TypeId::new(id.raw()))
    }

    /// Installed count for this tenant. Test introspection only.
    #[doc(hidden)]
    #[must_use]
    pub fn len(&self, tenant: TenantId) -> usize {
        self.next_id
            .get(&tenant)
            .map_or(0, |e| e.load(Ordering::Acquire) as usize)
    }

    /// Install an EXACT `(tenant, id) ↔ name` binding (P0 #776 — WAL
    /// replay recovery). Unlike [`Self::intern`], this does **not**
    /// allocate a fresh id; it re-establishes the binding a prior
    /// process committed, replayed from a
    /// [`WalRecordType::InternString`] record by
    /// [`crate::wal::recover_from_wal`].
    ///
    /// The per-tenant allocator is bumped to `max(current, id)` via
    /// [`AtomicU32::fetch_max`] so a future live [`Self::intern`] never
    /// re-hands an id replay already installed. Interns are WAL-logged
    /// in allocation order, so the replay sequence is dense + monotone
    /// and the post-replay allocator equals the recovered high-water
    /// (keeping [`Self::len`] exact); even a (theoretical) sparse
    /// sequence is collision-free because the next allocation is
    /// strictly greater than every installed id.
    ///
    /// Idempotent under double-replay (Lemma I2 parity with the MVCC
    /// chain): re-installing the same `(tenant, id, name)` overwrites
    /// with an identical value and re-applies a no-op `fetch_max`.
    pub fn intern_install(&self, tenant: TenantId, id: StringId, name: &str) {
        if self.physical.is_some() {
            // v10 replay installs the physical delta itself. A logical tail
            // reaching this facade is rejected by the bundle decoder/encoder;
            // do not recreate a resident mirror here.
            return;
        }
        // The sentinel is never handed out by the allocator, so a WAL
        // record carrying it is corruption upstream; skip defensively
        // rather than poisoning the reverse map with an id `resolve`
        // promises never to return.
        if id == STRINGID_SENTINEL {
            return;
        }
        // #1404 M0.x FIX-D — capture-exclusion read guard (adds a binding →
        // count-changing; exclude during a capture count+stream). Replay is
        // single-threaded so uncontended, but kept for symmetry + safety.
        let _install = self.resident().capture_lock.read();
        self.resident()
            .forward
            .insert((tenant, name.to_owned()), id);
        self.resident()
            .reverse
            .insert((tenant, id), Arc::new(name.to_owned()));
        // v2 M2 A4 — a binding arriving through install came FROM a
        // durable source (a WAL `InternString` record being replayed,
        // or a checkpoint section that was fsynced before restore chose
        // it), so it carries durable proof by provenance: mark it in
        // the durable-proof set so post-recovery logged interns of the
        // same name don't re-append it on every first reference.
        self.resident().logged.insert((tenant, id));
        self.next_id
            .entry(tenant)
            .or_insert_with(|| AtomicU32::new(0))
            .fetch_max(id.raw(), Ordering::AcqRel);
    }

    /// **#802 / ADR-197** — enumerate every interned `(StringId, name)`
    /// for `tenant`. Used by the per-call catalog build
    /// (`build_catalog_for_tenant`) so a label/rel-type name that was
    /// interned by a committed `CREATE` is resolvable by a SUBSEQUENT
    /// query's binder even before the catalog-stats snapshot reflects it
    /// — closing the documented catalog-seed gap that made a
    /// label-anchored `MATCH (:Account)` after `CREATE (:Account)`
    /// reject with `UnknownLabel`.
    ///
    /// The intern table draws labels + rel-types from one id space (see
    /// [`Self::intern_label`]), so the caller cannot tell which names
    /// are labels vs rel-types from this alone; the catalog seeds each
    /// name as BOTH (a `MATCH (:RelTypeName)` then scans for a node with
    /// that id and finds none — 0 rows, the correct Cypher result —
    /// rather than an `UnknownLabel` bind error).
    #[must_use]
    pub fn names_for_tenant(&self, tenant: TenantId) -> Vec<(StringId, Arc<String>)> {
        if self.physical.is_some() {
            // The post-M4 callers use exact `probe`; rebuilding a whole
            // string-cardinality Vec would recreate the #1404 RSS term.
            return Vec::new();
        }
        self.resident()
            .reverse
            .iter()
            .filter(|e| e.key().0 == tenant)
            .map(|e| (e.key().1, Arc::clone(e.value())))
            .collect()
    }

    /// The number of `(tenant, StringId, name)` bindings the capture will emit
    /// — the count [`Self::for_each_name`] enumerates. The ADR-229 producer
    /// writes it as the section header BEFORE streaming the names (exactly as
    /// the pre-M0.x `iter_all().len()` did). Cheap: one `reverse` DashMap
    /// length read.
    ///
    /// This MUST equal the number of callbacks [`Self::for_each_name`] fires,
    /// or the snapshot's declared count and streamed records diverge.
    #[must_use]
    pub fn name_count(&self) -> u64 {
        if self.physical.is_some() {
            return 0;
        }
        self.resident().reverse.len() as u64
    }

    /// ADR-229 checkpoint producer — STREAM every `(tenant, StringId, name)`
    /// binding through `f`, ONE at a time, NEVER building a whole-`Vec`.
    /// Restore feeds each back through [`Self::intern_install`] (the same
    /// WAL-replay entry point), so a checkpoint captures the intern table's
    /// full durable state. Iteration order is arbitrary. Stops + returns on the
    /// first `f` error (a sink write failure).
    ///
    /// **#1404 M0.x — the freeze-capture bound.** This runs UNDER
    /// `checkpoint_freeze` (`producer.rs:132`). It is the streaming twin of the
    /// (now `#[cfg(test)]`-only) whole-`Vec` `iter_all`, mirroring the M0.5
    /// snapshot streaming: the `reverse` map is walked emitting each name from
    /// the borrowed entry (no owned `Arc<String>` retained across the callback)
    /// — the same whole-in-RAM sibling the idempotency bindings had. The intern
    /// table has no spill tier (it is a smaller term, page-store-backed
    /// structurally at M4/M6 per ADR-230 OQ-G), so the win here is not
    /// re-collecting the whole reverse map into a `Vec` under the freeze.
    ///
    /// **#1404 M0.x FIX-D — returns the ACTUAL streamed count** so the producer
    /// can hard-check it against the `name_count()` header (mismatch → abort the
    /// checkpoint, never a corrupt-Ok snapshot). Under the capture WRITE guard
    /// ([`Self::capture_guard`]) header==streamed deterministically.
    ///
    /// **M5-D3 FIX 4 (#1518 skeptic review) — sorted `(tenant, StringId)`
    /// capture order.** `DashMap` iteration order is nondeterministic, and
    /// these bindings land byte-for-byte in checkpoint metadata (INV-M5.24:
    /// two loads of identical content must be byte-identical for any
    /// worker count / rerun — the same class the blob page-image capture
    /// was pinned for, see `blob.rs::sorted_resident_keys`). Only the
    /// (small, fixed-size) KEYS are collected into a `Vec` — the streaming
    /// discipline above still holds for the actual name payloads.
    pub fn for_each_name<F, E>(&self, mut f: F) -> std::result::Result<u64, E>
    where
        F: FnMut(TenantId, StringId, &str) -> std::result::Result<(), E>,
    {
        if self.physical.is_some() {
            return Ok(0);
        }
        let reverse = &self.resident().reverse;
        let mut keys: Vec<(TenantId, StringId)> = reverse.iter().map(|e| *e.key()).collect();
        keys.sort_unstable();
        let mut streamed = 0u64;
        for key in keys {
            let Some(entry) = reverse.get(&key) else {
                continue;
            };
            f(key.0, key.1, entry.value().as_str())?;
            streamed += 1;
        }
        Ok(streamed)
    }

    /// #1404 M0.x FIX-D — acquire the capture WRITE guard (see the
    /// `capture_lock` field docs). The producer holds it across `name_count()`
    /// and `for_each_name()` so no concurrent `intern`/`intern_install` skews
    /// header≠stream. The hot lookup path is untouched.
    pub fn capture_guard(&self) -> Option<parking_lot::RwLockWriteGuard<'_, ()>> {
        self.resident
            .as_deref()
            .map(|resident| resident.capture_lock.write())
    }

    /// #1404 M0.x — the whole-`Vec` capture, retained as a `#[cfg(test)]`
    /// ORACLE ONLY. The PRODUCTION capture is [`Self::for_each_name`], which
    /// streams one name at a time (never a `Vec`).
    #[cfg(test)]
    #[must_use]
    pub fn iter_all(&self) -> Vec<(TenantId, StringId, Arc<String>)> {
        self.resident()
            .reverse
            .iter()
            .map(|e| {
                let (tenant, id) = *e.key();
                (tenant, id, Arc::clone(e.value()))
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────
// WAL integration
// ─────────────────────────────────────────────────────────────────────

/// Encode the payload for a [`WalRecordType::InternString`] record.
///
/// Layout (little-endian):
/// ```text
/// 0..4   StringId as u32
/// 4..    UTF-8 name bytes (no inner length prefix; bounded by the
///        outer WalRecord.length field)
/// ```
#[must_use]
pub fn encode_intern_payload(id: StringId, name: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + name.len());
    out.extend_from_slice(&id.raw().to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out
}

/// Decode a [`WalRecordType::InternString`] payload.
///
/// Returns `(StringId, name)`. Rejects short payloads and non-UTF-8
/// name bytes as WAL corruption so recovery stays unambiguous.
pub fn decode_intern_payload(payload: &[u8]) -> Result<(StringId, String)> {
    if payload.len() < 4 {
        return Err(ArcGraphError::WalCorruption {
            lsn: Lsn::ZERO,
            reason: format!(
                "intern payload length {} < 4 bytes for StringId",
                payload.len()
            ),
        });
    }
    let mut id_bytes = [0u8; 4];
    id_bytes.copy_from_slice(&payload[..4]);
    let id = StringId::new(u32::from_le_bytes(id_bytes));
    let name = std::str::from_utf8(&payload[4..]).map_err(|e| ArcGraphError::WalCorruption {
        lsn: Lsn::ZERO,
        reason: format!("intern payload is not valid UTF-8: {e}"),
    })?;
    Ok((id, name.to_owned()))
}

/// Intern `s` for `tenant`, WAL-logging the binding unless the table
/// already holds **durable proof** for it (v2 M2 A4 — the
/// durable-logged-set protocol).
///
/// # Ordering / durability contract
///
/// A caller that receives `Ok(id)` from this function may reference
/// `id` in a subsequent commit: the `InternString` record for the
/// binding is guaranteed to precede that commit in the WAL. Proof:
/// `Ok` is returned only after EITHER
///
/// - this call's own [`WalHandle::append`] returned (append blocks
///   until fsync and assigns the record's LSN before returning, so
///   the caller's later commit append gets a strictly greater LSN), or
/// - the durable-proof set already contained `(tenant, id)` — and
///   membership is inserted only after SOME append returned `Ok` (or
///   by [`InternTable::intern_install`] from an already-durable
///   source). The DashSet shard lock gives the happens-before edge:
///   observing membership ⇒ that append had returned ⇒ this caller's
///   commit append starts later ⇒ higher LSN.
///
/// The pre-A4 gating on `was_new` was WRONG: `intern_is_new`
/// PUBLISHES the binding before any append, so a racing loser saw
/// `was_new == false`, skipped the log, and could commit a block
/// referencing an id with no durable `InternString` record (crash
/// before the winner's append → recovery reconstructs the node but
/// not the binding → the unseeded allocator reuses the id → the old
/// property silently materializes under the WRONG name). Under the
/// durable-logged-set protocol both racers append (an idempotent
/// duplicate — `intern_install` replay overwrites with the identical
/// binding); losers never wait on the winner, so a stalled winner
/// cannot stall the loser (no rendezvous, no timeout policy).
///
/// # Append failure
///
/// An append failure returns `Err` WITHOUT marking durable proof. The
/// binding stays published in-memory (idempotent id assignment for
/// later retries) but remains UNPROVEN: every subsequent logged-path
/// reference re-attempts the append until one succeeds, so no caller
/// can ever reference the id without a durable-proof observation. The
/// failing caller must abort its own commit (all callers propagate
/// this `Err`).
///
/// This also closes the #788 residual: a name published by an
/// UNLOGGED path (e.g. `graph.explore`'s rel-filter phantom-intern,
/// #355) has no durable proof, so the first logged-path reference
/// appends it instead of trusting the in-memory publish.
///
/// Budget: one extra DashSet read per call against an already-proven
/// name (≈ 50 ns amortized, the same class as the `intern_is_new`
/// fast path); the append itself is paid only until the first success
/// per binding.
pub fn intern_logged(
    table: &InternTable,
    wal: &WalHandle,
    tenant: TenantId,
    s: &str,
) -> Result<StringId> {
    if table.is_page_backed() {
        if wal.format_version() != crate::wal::BUNDLE_FORMAT_V10 {
            return Err(ArcGraphError::TransactionAborted {
                reason: format!(
                    "page-backed intern requires v10 WAL, got {}",
                    wal.format_version()
                ),
            });
        }
        return table
            .try_intern_physical(tenant, s)
            .map(|(id, _)| id)
            .map_err(|error| ArcGraphError::TransactionAborted {
                reason: format!("M4 intern owner publish failed: {error}"),
            });
    }
    let (id, _was_new) = table.intern_is_new(tenant, s).map_err(intern_owner_error)?;
    if !table.resident().logged.contains(&(tenant, id)) {
        let payload = encode_intern_payload(id, s);
        wal.append(
            WalRecordType::InternString,
            /* txn_id = */ 0,
            now_millis(),
            tenant,
            payload,
        )?;
        // Only after the fsync-blocking append RETURNED: the set may
        // under-claim durability (bounded duplicate appends under a
        // race), never over-claim it.
        table.resident().logged.insert((tenant, id));
    }
    Ok(id)
}

/// Intern a **label** name, WAL-logging the binding iff it is freshly
/// allocated AND a WAL handle is present (P0 #776 — durable name
/// recovery). The durable write path passes `Some(wal)`; the ephemeral
/// (`--in-memory`) path passes `None` and gets the prior pure in-memory
/// behaviour.
///
/// Ordering / durability (v2 M2 A4): every name referenced through
/// this logged write path carries **durable proof** before the caller
/// may commit — the [`WalRecordType::InternString`] record is appended
/// (fsync-blocking) BEFORE the caller's `crud::commit` runs, so the
/// intern record's LSN is strictly less than the commit LSN, and an
/// append failure propagates so the caller aborts the create rather
/// than committing a record whose name is not durably logged. Gating
/// is on the table's durable-proof set, NOT on the `was_new` latch —
/// see [`intern_logged`] for the race the latch had (a loser could
/// commit a reference before the winner's append) and the unlogged-
/// publish residual (#788 / #355) the set closes.
pub fn intern_label_logged(
    table: &InternTable,
    wal: Option<&WalHandle>,
    tenant: TenantId,
    name: &str,
) -> Result<LabelId> {
    let sid = match wal {
        Some(w) => intern_logged(table, w, tenant, name)?,
        None => table.intern(tenant, name).map_err(intern_owner_error)?,
    };
    Ok(LabelId::new(sid.raw()))
}

/// Intern a **relationship-type** name, WAL-logging iff freshly
/// allocated AND a WAL handle is present. Symmetric to
/// [`intern_label_logged`] (P0 #776). See that function for the
/// durability ordering contract.
pub fn intern_type_logged(
    table: &InternTable,
    wal: Option<&WalHandle>,
    tenant: TenantId,
    name: &str,
) -> Result<TypeId> {
    let sid = match wal {
        Some(w) => intern_logged(table, w, tenant, name)?,
        None => table.intern(tenant, name).map_err(intern_owner_error)?,
    };
    Ok(TypeId::new(sid.raw()))
}

/// Intern a name referenced as a **raw [`StringId`]** (a property KEY,
/// or any id embedded verbatim into durable state), WAL-logging iff a
/// WAL handle is present. Completes the [`intern_label_logged`] /
/// [`intern_type_logged`] family (v2 M2 A4 round-2, #1452): the
/// property-index CATALOG path embeds a `property_key: StringId` into
/// a durable catalog transaction, which needs the same durable-proof
/// ordering as the label leg — the `InternString` append (fsync-
/// blocking) must RETURN before the referencing commit runs, and an
/// append failure must abort that commit. `None` (the `--in-memory`
/// path) falls back to the pure in-memory intern: with no WAL there is
/// no recovery, so no crash-reuse class exists. See
/// [`intern_label_logged`] for the full contract and the `was_new`
/// race the durable-proof set closes.
pub fn intern_string_logged(
    table: &InternTable,
    wal: Option<&WalHandle>,
    tenant: TenantId,
    name: &str,
) -> Result<StringId> {
    match wal {
        Some(w) => intern_logged(table, w, tenant, name),
        None => table.intern(tenant, name).map_err(intern_owner_error),
    }
}

/// Translate a page-backed intern publish/lookup failure into the caller-facing
/// abort. Fail-CLOSED: the commit that needed the id is aborted rather than
/// proceeding with an invented (sentinel) id.
fn intern_owner_error(error: OwnerRowError) -> ArcGraphError {
    ArcGraphError::TransactionAborted {
        reason: format!("M4 intern owner publish failed: {error}"),
    }
}

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::thread;

    use arcgraph_core::TenantId;
    use proptest::prelude::*;

    use super::*;

    /// RE-GATE r2 (3) — a replayed `AllocatorAdvance{InternString}` must land the
    /// allocator on EXACTLY the id it would have had without the crash.
    ///
    /// `next_id` holds the LAST-allocated id (allocation is
    /// `prev = fetch_add(1); id = prev + 1`), and the advance carries
    /// `high_water = id`. Seeding with `high_water + 1` therefore overshoots and
    /// SKIPS one `StringId` per replayed advance. Monotone, so never-reissue and
    /// durability are intact — but it is an id-skip, and it contradicted both the
    /// bootstrap seed (`page_backed`, no `+1`) and the node/rel seeds
    /// (`CrudStore::seed_node_from_advance`, no `+1`).
    ///
    /// The oracle is differential: the same workload, with and without a replayed
    /// advance, must hand out the same next id.
    ///
    /// RED-on-revert: restore `saturating_add(1)` in `seed_string_allocator` —
    /// the replayed side hands out N+2 and this gate fails with "SKIPPED".
    #[test]
    fn gate_replayed_intern_advance_does_not_skip_a_string_id() {
        let tenant = TenantId::new(73); // NON-DEFAULT

        // Baseline: no crash, no replay.
        let baseline = InternTable::new();
        for name in ["alpha", "beta", "gamma"] {
            baseline.intern(tenant, name).unwrap();
        }
        let last_allocated = baseline.intern(tenant, "delta").unwrap().raw();
        let no_crash_next = baseline.intern(tenant, "epsilon").unwrap().raw();
        assert_eq!(
            no_crash_next,
            last_allocated + 1,
            "sanity: ids are dense on the happy path"
        );

        // Replay: a fresh table that learns the same high-water ONLY through the
        // replayed advance — exactly what recovery does (the counter is seeded
        // pre-replay from a stale marker, then advanced by the advance).
        let replayed = InternTable::new();
        replayed.seed_string_allocator(tenant, u64::from(last_allocated));
        let after_replay_next = replayed.intern(tenant, "epsilon").unwrap().raw();

        assert_eq!(
            after_replay_next, no_crash_next,
            "SKIPPED: after replaying an advance with high_water={last_allocated}, the \
             next StringId is {after_replay_next}, but a run with no crash would have \
             handed out {no_crash_next} — the seed overshot by one and burned a live id"
        );

        // The property the skip was masking still holds.
        assert!(
            after_replay_next > last_allocated,
            "REISSUED: replay handed back an already-committed StringId"
        );

        // Idempotent under double-replay.
        replayed.seed_string_allocator(tenant, u64::from(last_allocated));
        replayed.seed_string_allocator(tenant, u64::from(last_allocated));
        assert_eq!(
            replayed.intern(tenant, "zeta").unwrap().raw(),
            after_replay_next + 1,
            "double-replay of the same advance must not move the allocator"
        );
    }

    /// The sibling ACL allocator is deliberately NOT symmetric: `PermissionIndex`
    /// counts the NEXT class id (`raw = fetch_add(1)` uses `raw` itself as the
    /// id), so its `+1` is correct on both of its seed paths. Pinned so a future
    /// "make it consistent with intern" cleanup cannot strip it and start
    /// reissuing ACL class ids.
    #[test]
    fn gate_acl_class_allocator_counts_next_not_last() {
        use crate::permissions::PermissionIndex;

        let index = PermissionIndex::new();
        // Fresh resident index: nothing allocated yet.
        assert_eq!(index.class_allocator_next(), 0);

        // The resident arm has no durable class ids, so a replayed advance is
        // correctly ignored (only the page-backed arm owns durable class ids).
        index.seed_class_allocator(TenantId::new(73), 7);
        assert_eq!(
            index.class_allocator_next(),
            0,
            "the resident arm must ignore replayed AclClass advances"
        );
    }

    #[test]
    fn intern_returns_stable_id_within_tenant() {
        let table = InternTable::new();
        let a1 = table.intern(TenantId::DEFAULT, "Person").unwrap();
        let a2 = table.intern(TenantId::DEFAULT, "Person").unwrap();
        let b = table.intern(TenantId::DEFAULT, "Organization").unwrap();
        assert_eq!(
            a1, a2,
            "repeated intern of the same string yields the same id"
        );
        assert_ne!(a1, b, "distinct strings yield distinct ids");
    }

    /// M5-D3 FIX 4 (#1518 skeptic review) — `for_each_name`'s capture order
    /// must be a pure function of the resident key set (sorted `(tenant,
    /// StringId)`), not `DashMap` iteration order (which is a function of
    /// insertion history via shard/bucket layout — the same nondeterminism
    /// class the blob page-image capture was pinned for, see
    /// `blob.rs::sorted_resident_keys`). Two independently-built tables
    /// holding the IDENTICAL `(tenant, StringId, name)` triples — installed
    /// via `intern_install` in DIFFERENT orders so DashMap shard layout
    /// differs — must stream in IDENTICAL order.
    ///
    /// RED-on-revert: replace the sorted-key capture with a raw
    /// `self.resident().reverse.iter()` walk (the pre-fix code) — this test
    /// then fails intermittently (most runs, on a large enough key set),
    /// since two tables built from the same bindings in different
    /// insertion order generally land in different DashMap bucket order.
    #[test]
    fn for_each_name_capture_order_is_sorted_not_insertion_or_shard_order() {
        let forward_table = InternTable::new();
        let reverse_table = InternTable::new();
        let tenants = [TenantId::new(11), TenantId::new(4), TenantId::new(97)];
        let bindings: Vec<(TenantId, StringId, String)> = tenants
            .iter()
            .flat_map(|tenant| {
                (0..200_u32).map(move |i| {
                    (
                        *tenant,
                        StringId::new(i + 1),
                        format!("name-{tenant:?}-{i:04}"),
                    )
                })
            })
            .collect();

        // Install into `forward_table` in one order...
        for (tenant, id, name) in &bindings {
            forward_table.intern_install(*tenant, *id, name);
        }
        // ...and into `reverse_table` in the REVERSE order (different
        // insertion history -> different DashMap shard/bucket layout), but
        // the EXACT SAME (tenant, StringId, name) triples end up resident.
        for (tenant, id, name) in bindings.iter().rev() {
            reverse_table.intern_install(*tenant, *id, name);
        }

        let mut forward_stream = Vec::new();
        forward_table
            .for_each_name(
                |tenant, id, name| -> std::result::Result<(), std::convert::Infallible> {
                    forward_stream.push((tenant, id, name.to_owned()));
                    Ok(())
                },
            )
            .unwrap();
        let mut reverse_stream = Vec::new();
        reverse_table
            .for_each_name(
                |tenant, id, name| -> std::result::Result<(), std::convert::Infallible> {
                    reverse_stream.push((tenant, id, name.to_owned()));
                    Ok(())
                },
            )
            .unwrap();

        assert_eq!(
            forward_stream.len(),
            bindings.len(),
            "sanity: every binding streamed"
        );
        assert_eq!(
            forward_stream, reverse_stream,
            "DEFECT: for_each_name's capture order depends on insertion \
             history (DashMap shard/bucket layout), not a sort — two tables \
             with identical bindings streamed in DIFFERENT order"
        );
        // And it really is sorted by (tenant, StringId).
        let mut sorted = forward_stream.clone();
        sorted.sort_unstable_by_key(|(tenant, id, _)| (*tenant, *id));
        assert_eq!(
            forward_stream, sorted,
            "capture order must be exactly sorted (tenant, StringId)"
        );
    }

    #[test]
    fn first_id_is_one_sentinel_is_zero() {
        let table = InternTable::new();
        let id = table.intern(TenantId::DEFAULT, "first").unwrap();
        assert_ne!(id, STRINGID_SENTINEL);
        assert_eq!(id.raw(), 1);
    }

    #[test]
    fn intern_is_new_flag_reports_first_insert() {
        let table = InternTable::new();
        let (_id1, new1) = table.intern_is_new(TenantId::DEFAULT, "label").unwrap();
        let (_id2, new2) = table.intern_is_new(TenantId::DEFAULT, "label").unwrap();
        assert!(new1, "first intern is new");
        assert!(!new2, "second intern is not new");
    }

    #[test]
    fn intern_allocates_new_id_per_tenant() {
        // (§A) cross-tenant isolation.
        //
        // The allocator is per-tenant (starting at 1 for each), so the raw numeric
        // id may collide across tenants — e.g. the first-ever name
        // under each tenant is always `StringId(1)`. What this test
        // asserts is semantic isolation, which is what callers
        // actually care about: a lookup keyed by `(tenant, id)`
        // resolves only within the tenant that installed the
        // binding.
        let table = InternTable::new();
        let t1 = TenantId::new(100);
        let t2 = TenantId::new(101);
        let id1 = table.intern(t1, "User").unwrap();
        let id2 = table.intern(t2, "User").unwrap();
        // Per-tenant allocator — both are StringId(1) but that is
        // not the contract; equality is an implementation detail.
        assert_eq!(id1.raw(), 1);
        assert_eq!(id2.raw(), 1);
        // Resolve is tenant-scoped.
        assert_eq!(*table.try_resolve(t1, id1).unwrap().unwrap(), "User");
        assert_eq!(*table.try_resolve(t2, id2).unwrap().unwrap(), "User");
        // A second distinct name under t1 must not accidentally
        // collide with t2's existing "User" binding.
        let id_other = table.intern(t1, "Robot").unwrap();
        assert_ne!(id_other, id1);
    }

    #[test]
    fn per_tenant_namespaces_are_disjoint_under_identical_names() {
        // (§A) Cross-tenant resolve must be strict: the same raw id
        // used in both tenants points to independent bindings.
        let table = InternTable::new();
        let t1 = TenantId::new(200);
        let t2 = TenantId::new(201);
        let _ = table.intern(t1, "A").unwrap();
        let _ = table.intern(t2, "B").unwrap();
        let one = StringId::new(1);
        assert_eq!(&*table.try_resolve(t1, one).unwrap().unwrap(), "A");
        assert_eq!(&*table.try_resolve(t2, one).unwrap().unwrap(), "B");
    }

    #[test]
    fn resolve_roundtrips_for_interned_id() {
        let table = InternTable::new();
        let id = table.intern(TenantId::DEFAULT, "Company").unwrap();
        let got = table.try_resolve(TenantId::DEFAULT, id).unwrap().unwrap();
        assert_eq!(&*got, "Company");
    }

    #[test]
    fn resolve_on_unknown_id_returns_none() {
        let table = InternTable::new();
        let _ = table.intern(TenantId::DEFAULT, "x").unwrap();
        assert!(
            table
                .try_resolve(TenantId::DEFAULT, StringId::new(9999))
                .unwrap()
                .is_none()
        );
        assert!(
            table
                .try_resolve(TenantId::DEFAULT, STRINGID_SENTINEL)
                .unwrap()
                .is_none(),
            "the sentinel is never handed out",
        );
    }

    #[test]
    fn intern_label_and_type_share_allocator() {
        // Interning a name twice via two different aliases surfaces
        // the same underlying id, confirming the allocator is shared.
        let table = InternTable::new();
        let lid = table.intern_label(TenantId::DEFAULT, "Unit").unwrap();
        let tid = table.intern_type(TenantId::DEFAULT, "Unit").unwrap();
        assert_eq!(lid.raw(), tid.raw());
    }

    // ─── RC-3: read-only probe (secondary property index, #1366) ─────

    #[test]
    fn probe_returns_id_for_already_interned_value() {
        let table = InternTable::new();
        let tenant = TenantId::new(7);
        let id = table.intern(tenant, "alice@example.com").unwrap();
        assert_eq!(
            table.try_probe(tenant, "alice@example.com").unwrap(),
            Some(id),
            "probe must return the id of an already-interned value",
        );
    }

    #[test]
    fn probe_is_tenant_scoped() {
        let table = InternTable::new();
        let t1 = TenantId::new(1);
        let t2 = TenantId::new(2);
        let _ = table.intern(t1, "shared").unwrap();
        // A value interned under t1 is invisible when probed under t2.
        assert!(
            table.try_probe(t2, "shared").unwrap().is_none(),
            "probe under a different tenant must miss (per-tenant namespace)",
        );
    }

    /// **RC-3 sensitivity gate — test (d).** A read-side lookup for a
    /// never-seen value MUST NOT grow the intern table. We assert the
    /// EXACT `len(tenant)` is unchanged across N distinct never-seen
    /// probes.
    ///
    /// RED-on-revert: routing the lookup through `intern` /
    /// `intern_is_new` (the only pre-RC-3 string paths) allocates a
    /// fresh id per distinct value, so `len` would grow from `base` to
    /// `base + N`. The parallel `..._red_on_revert` test below pins
    /// that failing behavior so the guard's sensitivity is proven.
    #[test]
    fn probe_does_not_grow_intern_table() {
        let table = InternTable::new();
        let tenant = TenantId::new(500);
        // Seed one real value so `len` starts non-zero (base = 1).
        let _ = table.intern(tenant, "seed@x").unwrap();
        let base = table.len(tenant);
        assert_eq!(base, 1);

        const N: usize = 128;
        for i in 0..N {
            let query = format!("never-seen-{i}@nowhere.test");
            // Probe a value NO node ever wrote — a proof of empty set.
            assert!(
                table.try_probe(tenant, &query).unwrap().is_none(),
                "never-seen value {query} must probe-miss",
            );
        }

        assert_eq!(
            table.len(tenant),
            base,
            "probe of {N} distinct never-seen values must NOT grow the \
             intern table (RC-3: read lookups never insert-intern)",
        );
    }

    /// RED-on-revert control for [`probe_does_not_grow_intern_table`].
    /// This is the behavior a pre-RC-3 implementation (or a regression
    /// that routes lookups through `intern`) would exhibit: the table
    /// GROWS by exactly one id per distinct never-seen value. Encoding
    /// it as an explicit assertion proves the guard above is
    /// sensitive — the two tests cannot both pass under the same code
    /// path.
    #[test]
    fn intern_grows_table_on_never_seen_values_red_on_revert() {
        let table = InternTable::new();
        let tenant = TenantId::new(501);
        let _ = table.intern(tenant, "seed@x").unwrap();
        let base = table.len(tenant);

        const N: usize = 128;
        for i in 0..N {
            let query = format!("never-seen-{i}@nowhere.test");
            // The reverted path: `intern` INSERTS on a miss.
            let _ = table.intern(tenant, &query).unwrap();
        }

        assert_eq!(
            table.len(tenant),
            base + N,
            "the reverted (intern-based) lookup path grows the table by \
             one id per distinct never-seen value — the exact regression \
             RC-3's probe closes",
        );
    }

    // ─── intern_install (P0 #776 WAL replay recovery) ────────────────

    #[test]
    fn intern_install_roundtrips_forward_and_reverse() {
        // Replay installs an EXACT (id, name) pair; both directions must
        // resolve, and a subsequent live intern of the same name returns
        // the SAME id (no re-allocation).
        let table = InternTable::new();
        let tenant = TenantId::new(7);
        table.intern_install(tenant, StringId::new(3), "Account");
        assert_eq!(
            &*table
                .try_resolve(tenant, StringId::new(3))
                .unwrap()
                .unwrap(),
            "Account"
        );
        assert_eq!(
            table.intern(tenant, "Account").unwrap(),
            StringId::new(3),
            "live intern of an installed name returns the installed id",
        );
    }

    #[test]
    fn intern_install_bumps_allocator_so_fresh_ids_never_collide() {
        // After installing id=5, a fresh DISTINCT name must allocate an
        // id strictly greater than 5 — never re-handing an installed id.
        let table = InternTable::new();
        let tenant = TenantId::new(7);
        table.intern_install(tenant, StringId::new(5), "Recovered");
        let fresh = table.intern(tenant, "BrandNew").unwrap();
        assert!(
            fresh.raw() > 5,
            "fresh id {} must exceed the installed high-water 5",
            fresh.raw(),
        );
        assert_ne!(fresh, StringId::new(5));
    }

    #[test]
    fn intern_install_is_idempotent_under_double_replay() {
        // Double-replay (Lemma I2 parity) must be a no-op: same binding,
        // allocator unchanged beyond the installed id.
        let table = InternTable::new();
        let tenant = TenantId::new(7);
        table.intern_install(tenant, StringId::new(2), "Doc");
        table.intern_install(tenant, StringId::new(2), "Doc");
        assert_eq!(
            &*table
                .try_resolve(tenant, StringId::new(2))
                .unwrap()
                .unwrap(),
            "Doc"
        );
        // Next allocation is 3 (one past the installed high-water 2).
        assert_eq!(table.intern(tenant, "Next").unwrap(), StringId::new(3));
    }

    #[test]
    fn intern_install_is_tenant_scoped() {
        // The same raw id under two tenants installs independent bindings.
        let table = InternTable::new();
        let t1 = TenantId::new(1);
        let t2 = TenantId::new(2);
        table.intern_install(t1, StringId::new(1), "Account");
        table.intern_install(t2, StringId::new(1), "Customer");
        assert_eq!(
            &*table.try_resolve(t1, StringId::new(1)).unwrap().unwrap(),
            "Account"
        );
        assert_eq!(
            &*table.try_resolve(t2, StringId::new(1)).unwrap().unwrap(),
            "Customer"
        );
    }

    #[test]
    fn intern_install_skips_sentinel() {
        // A record carrying the sentinel is upstream corruption; install
        // must not poison the reverse map with an id `resolve` promises
        // never to return.
        let table = InternTable::new();
        let tenant = TenantId::new(7);
        table.intern_install(tenant, STRINGID_SENTINEL, "bad");
        assert!(
            table
                .try_resolve(tenant, STRINGID_SENTINEL)
                .unwrap()
                .is_none()
        );
    }

    // ─── v2 M2 A4 — durable-proof set semantics (unit grain; the
    //     WAL-integration legs live in
    //     tests/intern_wal_replay_recovery.rs and the mcp gate
    //     m2_intern_durability_gate.rs) ──────────────────────────────

    #[test]
    fn live_publish_carries_no_durable_proof_but_install_does() {
        let table = InternTable::new();
        let tenant = TenantId::new(7);
        // A live (unlogged) publish must NOT claim durable proof — the
        // A4 race was exactly `was_new == false` being read as proof.
        let live = table.intern(tenant, "live_only").unwrap();
        assert!(
            !table.resident().logged.contains(&(tenant, live)),
            "an in-memory publish must not claim durable proof (A4)",
        );
        // A replay/restore install IS durable by provenance.
        table.intern_install(tenant, StringId::new(9), "replayed");
        assert!(
            table
                .resident()
                .logged
                .contains(&(tenant, StringId::new(9))),
            "an installed binding carries durable proof by provenance",
        );
        // The skipped sentinel install proves nothing.
        table.intern_install(tenant, STRINGID_SENTINEL, "bad");
        assert!(
            !table
                .resident()
                .logged
                .contains(&(tenant, STRINGID_SENTINEL))
        );
    }

    #[test]
    fn intern_payload_roundtrips() {
        let payload = encode_intern_payload(StringId::new(42), "Person");
        let (id, name) = decode_intern_payload(&payload).unwrap();
        assert_eq!(id.raw(), 42);
        assert_eq!(name, "Person");
    }

    #[test]
    fn intern_payload_rejects_truncated() {
        let err = decode_intern_payload(&[0u8; 3]).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    #[test]
    fn intern_payload_rejects_non_utf8() {
        let mut bad = encode_intern_payload(StringId::new(1), "ok");
        bad.push(0xFF);
        bad.push(0xFE);
        let err = decode_intern_payload(&bad).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: if cfg!(debug_assertions) { 500 } else { 5_000 },
            ..ProptestConfig::default()
        })]

        #[test]
        fn intern_payload_any_name_roundtrip(
            raw in any::<u32>(),
            // Keep names short-ish so the test stays fast.
            name in "[a-zA-Z0-9_\\- ]{0,64}",
        ) {
            let bytes = encode_intern_payload(StringId::new(raw), &name);
            let (id, back) = decode_intern_payload(&bytes).unwrap();
            prop_assert_eq!(id.raw(), raw);
            prop_assert_eq!(back, name);
        }
    }

    // ─── Concurrency ──────────────────────────────────────────────────

    #[test]
    fn concurrent_intern_is_race_free() {
        // 8 threads × a pool of 1 000 strings × 50 interns per string
        // (so every string is interned many times by multiple threads).
        // Every resulting id must be stable and identical across
        // threads for the same (tenant, string).
        const THREADS: usize = 8;
        const POOL: usize = 1_000;
        const PASSES: usize = 50;

        let table = Arc::new(InternTable::new());
        let tenant = TenantId::new(500);

        let pool: Vec<String> = (0..POOL).map(|i| format!("label_{i:04}")).collect();
        let pool = Arc::new(pool);

        let mut handles = Vec::with_capacity(THREADS);
        for t in 0..THREADS {
            let table = Arc::clone(&table);
            let pool = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                let mut observed: Vec<StringId> = Vec::with_capacity(POOL);
                for _ in 0..PASSES {
                    for (i, name) in pool.iter().enumerate() {
                        let id = table.intern(tenant, name).unwrap();
                        if observed.len() <= i {
                            observed.push(id);
                        } else {
                            assert_eq!(
                                observed[i], id,
                                "thread {t}: id for {name:?} was not stable",
                            );
                        }
                    }
                }
                observed
            }));
        }

        let first = handles
            .into_iter()
            .map(|h| h.join().expect("thread panicked"))
            .collect::<Vec<_>>();

        // Every thread observed the same sequence.
        for (t, obs) in first.iter().enumerate().skip(1) {
            assert_eq!(obs, &first[0], "thread {t} disagrees on id assignment");
        }

        // Every id is distinct (one per string in the pool).
        let distinct: HashSet<StringId> = first[0].iter().copied().collect();
        assert_eq!(distinct.len(), POOL, "ids collide across distinct names");

        // No id is the sentinel.
        assert!(
            !distinct.contains(&STRINGID_SENTINEL),
            "sentinel must never be handed out",
        );

        // Allocator value equals POOL.
        assert_eq!(table.len(tenant), POOL);
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: if cfg!(debug_assertions) { 16 } else { 64 },
            ..ProptestConfig::default()
        })]

        #[test]
        fn concurrent_intern_matches_sequential_fingerprint(
            // Random list of names, sequential-order insertion defines
            // the "sequential fingerprint"; a concurrent run should
            // yield the same set of ids (not necessarily the same
            // assignment, but the id set and cardinality).
            names in prop::collection::vec("[a-z]{1,8}", 1..32),
        ) {
            let concurrent = InternTable::new();
            let tenant = TenantId::DEFAULT;

            let shared = Arc::new(concurrent);
            let names_arc = Arc::new(names.clone());

            let mut handles = Vec::new();
            for _ in 0..4 {
                let table = Arc::clone(&shared);
                let names = Arc::clone(&names_arc);
                handles.push(thread::spawn(move || {
                    for n in names.iter() {
                        let _ = table.intern(tenant, n).unwrap();
                    }
                }));
            }
            for h in handles { h.join().expect("thread panicked"); }

            // Every unique name has exactly one id; the allocator
            // count equals the distinct-name count.
            let distinct: HashSet<&String> = names_arc.iter().collect();
            prop_assert_eq!(shared.len(tenant), distinct.len());

            // Resolving every interned id yields the original name.
            for n in distinct.iter() {
                let id = shared.intern(tenant, n).unwrap();
                let back = shared.try_resolve(tenant, id).unwrap().unwrap();
                prop_assert_eq!(&*back, *n);
            }
        }
    }
}
