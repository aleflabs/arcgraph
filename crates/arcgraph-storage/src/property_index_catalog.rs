//! **Property-index catalog (#1366, task #248 — Phase 1).**
//!
//! The durable declared-index registry + `Building → Online` state
//! machine for the user-visible secondary node-property index (design
//! `docs/design/property-index-design.md` §Maintenance and crash
//! consistency). It records `(index_name, tenant, label, property, state)`
//! for every `CREATE INDEX … FOR (n:Label) ON (n.prop)`, and it is the
//! authority the planner (Phase 2) consults for `planner_visible`
//! (`Online`) indexes.
//!
//! # Durability model — piggybacks the tenants-table / durability-tier
//! catalog pattern
//!
//! The whole registry is serialized into ONE MVCC value under
//! `TenantId::SYSTEM` at [`PROPERTY_INDEX_CATALOG_KEY`], exactly like
//! `SystemCatalog`'s tenants-table header (`catalog.rs`). Every mutation
//! (`create` / `drop` / `set_state`) is a SYSTEM-tenant MVCC commit that
//! rewrites the whole record list — recovery replays it through the
//! standard MVCC path, no bespoke replay code. At RC scale (a handful of
//! declared indexes) a single serialized list is cheaper than a
//! per-index key namespace and matches the existing catalog precedent.
//!
//! # Building → Online state machine (RC-2, the crash-safety half)
//!
//! - `CREATE INDEX` inserts the record as [`IndexState::Building`]
//!   (planner ignores it; commit-path maintenance still applies —
//!   write-follows-declare, see [`crate::secondary_handle::IndexState`]).
//! - Backfill scans MVCC-visible nodes once, inserts keys into the
//!   [`crate::secondary_handle::SecondaryIndexHandle`], and on completion
//!   flips the record to [`IndexState::Online`]. **The flip is a
//!   DISCRETE SYSTEM tx committed AFTER the whole backfill tail is
//!   durable — it does NOT ride the same `CommitBundle` as the tail.**
//!   (Backfill emits N independent, each-durable-on-return per-insert
//!   bundles; the MCP-layer `create_property_index` commits the flip in
//!   its own tx after the last insert.) The "no durable `Online`
//!   without a complete tail" property therefore rests on
//!   append-all-then-flip ORDERING + synchronous-durable inserts +
//!   recovery forcing non-`Online` → `Building` (below), NOT a shared
//!   bundle. [`PropertyIndexCatalog::set_state_in`] still takes a
//!   caller-held `Transaction` so a FUTURE refactor *could* co-commit
//!   the flip with a watermark, but the Phase-1 caller does not.
//! - **Crash mid-backfill** recovers as `Building` by construction: the
//!   `Online` flip tx never committed (or, if a future co-commit path
//!   is added, its bundle is lost), so recovery reads the last durable
//!   state and `Self::recover` forces any non-`Online` state down to
//!   `Building`. The planner ignores it and backfill restarts from
//!   scratch (idempotent insert reconciles the partial tail).
//!
//! # Bounded contexts (PD#7)
//!
//! This catalog is JSON-opaque: it stores label/property as opaque
//! identifiers (a `LabelId` and the interned property-key `StringId`)
//! plus a human display name. The MCP layer computes typed key deltas
//! and consults the catalog through the published API here; storage
//! never parses a property bag.

use arcgraph_core::{LabelId, Lsn, Result, StringId, TenantId};
use bytes::Bytes;
use parking_lot::RwLock;

use crate::secondary_handle::IndexState;
use crate::transaction::{Transaction, TxnManager};

/// MVCC key inside `TenantId::SYSTEM` for the property-index catalog
/// header (the whole serialized record list). Disjoint from the other
/// SYSTEM catalog key namespaces:
///
/// - `0` — tenants-table header ([`crate::catalog`]).
/// - `2` — secondary-index root pointer (`secondary_btree.rs`).
/// - `0x8000_…` — ADR-034 per-tenant durability tier.
///
/// `0x2000_…` sits in the reserved high-prefix band, disjoint from all
/// of the above (pinned by `tests::catalog_key_is_disjoint`).
pub const PROPERTY_INDEX_CATALOG_KEY: u64 = 0x2000_0000_0000_0000;

/// Serialization format version for the catalog header value. Bump on
/// any wire-shape change; a decoder that sees an unknown version treats
/// the record list as empty (forward-compat with a future variant).
const CATALOG_FORMAT_VERSION: u8 = 1;

/// One declared property index.
///
/// `label` + `property_key` are the storage-opaque identifiers the
/// commit-path maintenance and the (Phase-2) planner match against.
/// `name` + `property_name` are the human-facing display fields for
/// `SHOW INDEXES` / DDL round-trip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyIndexRecord {
    /// Declared index name (the DDL identifier; `SHOW INDEXES` key).
    pub name: String,
    /// Owning tenant.
    pub tenant: TenantId,
    /// The node label the index is `FOR (n:Label)`.
    pub label: LabelId,
    /// Interned property-key id (`ON (n.property)`). The property VALUE
    /// is keyed separately (RC-4 hash); this is the KEY name, which
    /// stays interned.
    pub property_key: StringId,
    /// Human-readable property name (display for `SHOW INDEXES`).
    pub property_name: String,
    /// Lifecycle state (`Building` until backfill completes + the
    /// `Online` flip commits in its own SYSTEM tx after the tail is
    /// durable — append-all-then-flip, not a shared watermark bundle).
    pub state: IndexState,
}

/// Outcome of a [`PropertyIndexCatalog::create_index`] call (the
/// `IF NOT EXISTS` idempotency contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreateOutcome {
    /// The index was newly inserted (as `Building`). The caller MUST
    /// run backfill and flip to `Online`.
    Created,
    /// An index of this name already existed (idempotent `IF NOT
    /// EXISTS` no-op). No backfill needed.
    AlreadyExists,
}

/// Error surface for property-index catalog operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PropertyIndexCatalogError {
    /// A non-`IF NOT EXISTS` `CREATE INDEX` named an existing index.
    #[error("property index '{name}' already exists")]
    AlreadyExists {
        /// The offending index name.
        name: String,
    },
    /// A non-`IF EXISTS` `DROP INDEX` named an absent index.
    #[error("property index '{name}' does not exist")]
    NotFound {
        /// The offending index name.
        name: String,
    },
    /// The underlying MVCC commit failed (e.g. WAL fsync error under T1;
    /// Z-1 (b) rollback restores catalog state).
    #[error("property index catalog commit failed: {0}")]
    Commit(#[from] arcgraph_core::ArcGraphError),
}

/// In-process durable property-index catalog, scoped to
/// `TenantId::SYSTEM`. One instance per `TxnManager`. All mutating
/// methods are serialized under the internal `RwLock`; the durable
/// authority is the MVCC value at [`PROPERTY_INDEX_CATALOG_KEY`].
#[derive(Debug, Default)]
pub struct PropertyIndexCatalog {
    /// In-memory materialization of the record list. The MVCC value is
    /// the durability authority; this cache is rebuilt on `recover` and
    /// kept in sync under the lock by each mutation.
    records: RwLock<Vec<PropertyIndexRecord>>,
}

impl PropertyIndexCatalog {
    /// Construct an empty catalog. Call [`Self::recover`] once at boot
    /// to seed it from durable MVCC state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// **Recovery** — seed the in-memory record list from the durable
    /// MVCC header, then FORCE every non-`Online` state down to
    /// `Building`.
    ///
    /// A record whose last durable state is `Building` means its
    /// backfill never reached the `Online`-flip commit — the planner
    /// must ignore it and backfill must restart. An `Online` record is
    /// complete by ORDERING: the flip tx commits only AFTER every
    /// backfill insert is durable (append-all-then-flip). That runtime
    /// guarantee — a durable `Online` implies a durable tail, i.e. no
    /// STEP4-before-STEP3 reorder — is pinned by the MCP-layer
    /// `backfill_inserts_precede_online_flip` ordering test (the real
    /// RED-on-revert guard: moving the flip before the loop makes an
    /// insert observe `Online`). This force-to-`Building` reconcile is
    /// the layered defense-in-depth belt: any durable state byte that is
    /// not the explicit `Online` completion tag (a would-be future
    /// `Backfilling`/`Flipping` marker) recovers as `Building`. Pinned by
    /// `crash_partial_tail_present_flip_dropped_recovers_building` (a
    /// non-empty partial tail durably present + the flip dropped ⇒
    /// `Building`) and `recover_never_promotes_unrecognized_state_byte`
    /// (an unknown durable state byte ⇒ `Building`, never `Online` — the
    /// decode+reconcile layers together).
    pub fn recover(&self, txn_mgr: &TxnManager, recovered_lsn: Lsn) {
        let txn = txn_mgr.begin(TenantId::SYSTEM);
        let bytes = txn.read(PROPERTY_INDEX_CATALOG_KEY);
        txn.abort();
        let mut records = match bytes {
            Some(b) => decode_catalog(&b),
            None => Vec::new(),
        };
        Self::reconcile_recovered_states(&mut records);
        let _ = recovered_lsn; // reserved for a future watermark check.
        *self.records.write() = records;
    }

    /// **The recover force-to-`Building` reconcile step (#1401 —
    /// property_index_catalog.rs:174-176).** Force every non-`Online`
    /// record down to `Building`: recovery must NEVER trust a partial /
    /// in-progress state. Today, with the `Building`/`Online` two-state
    /// machine and the append-all-then-flip ordering, this is
    /// defense-in-depth (a durably-`Online` record always has a complete
    /// durable tail, so `Online` is trustworthy). Its VALUE is guarding a
    /// future STEP4-before-STEP3 reorder or an added in-progress state:
    /// any state that is NOT the explicit `Online` completion tag
    /// (including a would-be `Backfilling`/`Flipping` marker) recovers as
    /// `Building` and restarts the backfill. Factored out so the crash
    /// test can drive it on a constructed in-progress record (see
    /// `reconcile_forces_non_online_to_building`) — RED-on-revert:
    /// deleting the force leaves the in-progress record un-downgraded.
    fn reconcile_recovered_states(records: &mut [PropertyIndexRecord]) {
        for r in records.iter_mut() {
            if r.state != IndexState::Online {
                r.state = IndexState::Building;
            }
        }
    }

    /// Snapshot the current record list.
    #[must_use]
    pub fn list(&self) -> Vec<PropertyIndexRecord> {
        self.records.read().clone()
    }

    /// Snapshot the record list for one tenant.
    #[must_use]
    pub fn list_for_tenant(&self, tenant: TenantId) -> Vec<PropertyIndexRecord> {
        self.records
            .read()
            .iter()
            .filter(|r| r.tenant == tenant)
            .cloned()
            .collect()
    }

    /// Look up the declared indexes on `(tenant, label)` — the set of
    /// property keys the commit-path maintenance must maintain for a
    /// write to a node of `label`. Returns `(property_key, state)` per
    /// declared index so the caller can gate on `maintenance_active`.
    #[must_use]
    pub fn indexes_on(&self, tenant: TenantId, label: LabelId) -> Vec<(StringId, IndexState)> {
        self.records
            .read()
            .iter()
            .filter(|r| r.tenant == tenant && r.label == label)
            .map(|r| (r.property_key, r.state))
            .collect()
    }

    /// Resolve a declared index by name within a tenant.
    #[must_use]
    pub fn resolve(&self, tenant: TenantId, name: &str) -> Option<PropertyIndexRecord> {
        self.records
            .read()
            .iter()
            .find(|r| r.tenant == tenant && r.name == name)
            .cloned()
    }

    /// **`CREATE INDEX name [IF NOT EXISTS] FOR (n:Label) ON (n.prop)`.**
    ///
    /// Inserts a fresh record as [`IndexState::Building`] and durably
    /// commits the rewritten catalog header under `TenantId::SYSTEM`.
    /// The caller then runs backfill and flips to `Online`.
    ///
    /// `IF NOT EXISTS` semantics:
    /// - name absent → insert + [`CreateOutcome::Created`].
    /// - name present + `if_not_exists` → idempotent no-op +
    ///   [`CreateOutcome::AlreadyExists`] (existing record untouched).
    /// - name present + NOT `if_not_exists` →
    ///   [`PropertyIndexCatalogError::AlreadyExists`].
    ///
    /// The check-and-insert is atomic under the catalog lock (no TOCTOU
    /// for concurrent same-name creates). Uniqueness is by
    /// `(tenant, name)`.
    pub fn create_index(
        &self,
        txn_mgr: &TxnManager,
        record: PropertyIndexRecord,
        if_not_exists: bool,
    ) -> std::result::Result<CreateOutcome, PropertyIndexCatalogError> {
        // Force the inserted state to Building regardless of what the
        // caller passed — a CREATE always starts a fresh backfill.
        let mut record = record;
        record.state = IndexState::Building;

        let mut guard = self.records.write();
        if let Some(existing) = guard
            .iter()
            .find(|r| r.tenant == record.tenant && r.name == record.name)
        {
            if if_not_exists {
                let _ = existing;
                return Ok(CreateOutcome::AlreadyExists);
            }
            return Err(PropertyIndexCatalogError::AlreadyExists { name: record.name });
        }

        // Speculatively append, encode, and commit. On commit failure,
        // roll the in-memory list back (Z-1 (b) convention — the MVCC
        // write was undone; re-align the cache).
        guard.push(record);
        let encoded = encode_catalog(&guard);
        if let Err(e) = commit_catalog(txn_mgr, encoded) {
            guard.pop();
            return Err(e.into());
        }
        Ok(CreateOutcome::Created)
    }

    /// **`DROP INDEX name [IF EXISTS]`.**
    ///
    /// Removes the catalog record + durably commits. (The index PAGES
    /// are drained by the caller's storage teardown; Phase 1's RC scope
    /// tombstones the metadata so the planner and commit-path
    /// maintenance immediately stop consulting it.)
    ///
    /// `IF EXISTS`: absent name → idempotent no-op. Non-`IF EXISTS`
    /// absent name → [`PropertyIndexCatalogError::NotFound`].
    pub fn drop_index(
        &self,
        txn_mgr: &TxnManager,
        tenant: TenantId,
        name: &str,
        if_exists: bool,
    ) -> std::result::Result<bool, PropertyIndexCatalogError> {
        let mut guard = self.records.write();
        let Some(pos) = guard
            .iter()
            .position(|r| r.tenant == tenant && r.name == name)
        else {
            if if_exists {
                return Ok(false);
            }
            return Err(PropertyIndexCatalogError::NotFound {
                name: name.to_string(),
            });
        };
        let removed = guard.remove(pos);
        let encoded = encode_catalog(&guard);
        if let Err(e) = commit_catalog(txn_mgr, encoded) {
            // Restore the removed record on commit failure.
            guard.insert(pos, removed);
            return Err(e.into());
        }
        Ok(true)
    }

    /// **Flip a declared index's lifecycle state, staged into the
    /// caller's held transaction.**
    ///
    /// The caller passes a HELD [`Transaction`] (a `TenantId::SYSTEM`
    /// tx). This method stages the catalog-header rewrite into `tx` but
    /// does NOT commit it — the caller commits `tx`, so the state flip
    /// lands in whatever `CommitBundle` that `tx` carries. In the
    /// Phase-1 `Building → Online` path the caller commits a tx that
    /// carries ONLY the flip (the backfill inserts already committed in
    /// their own N prior bundles) — the flip is NOT co-committed with the
    /// backfill tail; the tail-precedes-`Online` guarantee comes from
    /// append-all-then-flip ordering, not a shared bundle. (The held-tx
    /// signature is retained so a future refactor *could* co-commit the
    /// flip with a watermark write.) If `tx` never commits (crash /
    /// abort), the flip is not durable and recovery reads the prior
    /// `Building` state.
    ///
    /// The in-memory cache is updated eagerly (pre-commit); the caller
    /// MUST call [`Self::revert_state`] on commit failure to re-align it
    /// (mirrors `SystemCatalog::set_durability_tier`'s Z-1 (b) shape).
    ///
    /// Returns the previous state (for the revert path), or `None` if no
    /// record of that name exists in the tenant.
    pub fn set_state_in(
        &self,
        tx: &mut Transaction<'_>,
        tenant: TenantId,
        name: &str,
        new_state: IndexState,
    ) -> Option<IndexState> {
        debug_assert_eq!(
            tx.tenant(),
            TenantId::SYSTEM,
            "property-index catalog writes are under SYSTEM"
        );
        let mut guard = self.records.write();
        let rec = guard
            .iter_mut()
            .find(|r| r.tenant == tenant && r.name == name)?;
        let prev = rec.state;
        rec.state = new_state;
        let encoded = encode_catalog(&guard);
        tx.write(PROPERTY_INDEX_CATALOG_KEY, encoded);
        Some(prev)
    }

    /// Test-only variant of [`Self::create_index`] returning the
    /// commit LSN, so a crash/recovery test can seed a fresh catalog
    /// from the durable state at that LSN.
    #[cfg(test)]
    fn create_index_returning_lsn(
        &self,
        txn_mgr: &TxnManager,
        record: PropertyIndexRecord,
        if_not_exists: bool,
    ) -> std::result::Result<Lsn, PropertyIndexCatalogError> {
        let mut record = record;
        record.state = IndexState::Building;
        let mut guard = self.records.write();
        if guard
            .iter()
            .any(|r| r.tenant == record.tenant && r.name == record.name)
        {
            if if_not_exists {
                // No commit happened; return the current SYSTEM high
                // watermark by doing an empty read-only begin/abort is
                // insufficient — for the test path we only call this on
                // a fresh name, so this branch is unreachable.
                return Err(PropertyIndexCatalogError::AlreadyExists { name: record.name });
            }
            return Err(PropertyIndexCatalogError::AlreadyExists { name: record.name });
        }
        guard.push(record);
        let encoded = encode_catalog(&guard);
        let mut txn = txn_mgr.begin(TenantId::SYSTEM);
        txn.write(PROPERTY_INDEX_CATALOG_KEY, encoded);
        match txn.commit() {
            Ok(lsn) => Ok(lsn),
            Err(e) => {
                guard.pop();
                Err(e.into())
            }
        }
    }

    /// Test-only: durably commit a catalog header carrying `records`,
    /// each with a caller-chosen RAW state byte (bypassing the
    /// `state_byte` mapping). Lets a crash test persist a would-be future
    /// in-progress sentinel (e.g. a `Flipping` marker byte the current
    /// two-state machine does not emit) and assert recovery never brings
    /// it up `Online`. Returns the commit LSN.
    #[cfg(test)]
    fn seed_durable_state_bytes(
        &self,
        txn_mgr: &TxnManager,
        records: &[(PropertyIndexRecord, u8)],
    ) -> Lsn {
        let mut buf = Vec::with_capacity(16 + records.len() * 32);
        buf.push(CATALOG_FORMAT_VERSION);
        buf.extend_from_slice(&(records.len() as u32).to_le_bytes());
        for (r, state_b) in records {
            let name = r.name.as_bytes();
            buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            buf.extend_from_slice(name);
            buf.extend_from_slice(&r.tenant.raw().to_le_bytes());
            buf.extend_from_slice(&r.label.raw().to_le_bytes());
            buf.extend_from_slice(&r.property_key.raw().to_le_bytes());
            let pname = r.property_name.as_bytes();
            buf.extend_from_slice(&(pname.len() as u32).to_le_bytes());
            buf.extend_from_slice(pname);
            buf.push(*state_b);
        }
        let mut txn = txn_mgr.begin(TenantId::SYSTEM);
        txn.write(PROPERTY_INDEX_CATALOG_KEY, Bytes::from(buf));
        txn.commit().expect("seed_durable_state_bytes commit")
    }

    /// Re-align the in-memory cache to `previous_state` after a
    /// [`Self::set_state_in`] whose enclosing `tx.commit()` failed
    /// (the MVCC write was rolled back by Z-1 (b)). No-op if the record
    /// is gone.
    pub fn revert_state(&self, tenant: TenantId, name: &str, previous_state: IndexState) {
        let mut guard = self.records.write();
        if let Some(rec) = guard
            .iter_mut()
            .find(|r| r.tenant == tenant && r.name == name)
        {
            rec.state = previous_state;
        }
    }
}

/// Commit a rewritten catalog-header value as a standalone
/// `TenantId::SYSTEM` MVCC commit (the `create` / `drop` path — these
/// are not co-committed with a data bundle).
fn commit_catalog(txn_mgr: &TxnManager, encoded: Bytes) -> Result<Lsn> {
    let mut txn = txn_mgr.begin(TenantId::SYSTEM);
    txn.write(PROPERTY_INDEX_CATALOG_KEY, encoded);
    txn.commit()
}

// ─────────────────────── wire encoding ───────────────────────
//
// Layout (little-endian):
//   byte 0            : format version
//   bytes 1..5        : record count u32
//   then, per record:
//     u32 name_len, name bytes (utf-8)
//     u64 tenant
//     u32 label
//     u32 property_key (interned StringId)
//     u32 property_name_len, property_name bytes (utf-8)
//     u8  state (0 = Building, 1 = Online)
//
// This is the SYSTEM-tenant MVCC *value* — independent of the ADR-031
// CommitBundle record format, read back through the standard MVCC
// replay path (same posture as `catalog::encode_durability_tier`).

const STATE_BUILDING: u8 = 0;
const STATE_ONLINE: u8 = 1;

fn state_byte(s: IndexState) -> u8 {
    match s {
        IndexState::Building => STATE_BUILDING,
        IndexState::Online => STATE_ONLINE,
    }
}

fn byte_state(b: u8) -> IndexState {
    // Any non-Online byte decodes as Building — recovery never comes up
    // Online on a byte it does not recognize as the explicit Online tag.
    if b == STATE_ONLINE {
        IndexState::Online
    } else {
        IndexState::Building
    }
}

fn encode_catalog(records: &[PropertyIndexRecord]) -> Bytes {
    let mut buf = Vec::with_capacity(16 + records.len() * 32);
    buf.push(CATALOG_FORMAT_VERSION);
    buf.extend_from_slice(&(records.len() as u32).to_le_bytes());
    for r in records {
        let name = r.name.as_bytes();
        buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&r.tenant.raw().to_le_bytes());
        buf.extend_from_slice(&r.label.raw().to_le_bytes());
        buf.extend_from_slice(&r.property_key.raw().to_le_bytes());
        let pname = r.property_name.as_bytes();
        buf.extend_from_slice(&(pname.len() as u32).to_le_bytes());
        buf.extend_from_slice(pname);
        buf.push(state_byte(r.state));
    }
    Bytes::from(buf)
}

/// Decode the catalog header value. Returns an empty list on any
/// malformed / unknown-version input (a corrupt catalog header must
/// never brick boot — the #1386-pattern reconcile is the safety net,
/// and the MVCC/WAL state is the durability authority).
fn decode_catalog(bytes: &[u8]) -> Vec<PropertyIndexRecord> {
    let mut cur = 0usize;
    let Some(&ver) = bytes.first() else {
        return Vec::new();
    };
    if ver != CATALOG_FORMAT_VERSION {
        return Vec::new();
    }
    cur += 1;
    let Some(count) = read_u32(bytes, &mut cur) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let Some(name) = read_str(bytes, &mut cur) else {
            return out;
        };
        let Some(tenant) = read_u64(bytes, &mut cur) else {
            return out;
        };
        let Some(label) = read_u32(bytes, &mut cur) else {
            return out;
        };
        let Some(property_key) = read_u32(bytes, &mut cur) else {
            return out;
        };
        let Some(property_name) = read_str(bytes, &mut cur) else {
            return out;
        };
        let Some(&state_b) = bytes.get(cur) else {
            return out;
        };
        cur += 1;
        out.push(PropertyIndexRecord {
            name,
            tenant: TenantId::new(tenant),
            label: LabelId::new(label),
            property_key: StringId::new(property_key),
            property_name,
            state: byte_state(state_b),
        });
    }
    out
}

fn read_u32(bytes: &[u8], cur: &mut usize) -> Option<u32> {
    let end = *cur + 4;
    let slice = bytes.get(*cur..end)?;
    let arr: [u8; 4] = slice.try_into().ok()?;
    *cur = end;
    Some(u32::from_le_bytes(arr))
}

fn read_u64(bytes: &[u8], cur: &mut usize) -> Option<u64> {
    let end = *cur + 8;
    let slice = bytes.get(*cur..end)?;
    let arr: [u8; 8] = slice.try_into().ok()?;
    *cur = end;
    Some(u64::from_le_bytes(arr))
}

fn read_str(bytes: &[u8], cur: &mut usize) -> Option<String> {
    let len = read_u32(bytes, cur)? as usize;
    let end = *cur + len;
    let slice = bytes.get(*cur..end)?;
    let s = String::from_utf8(slice.to_vec()).ok()?;
    *cur = end;
    Some(s)
}

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(name: &str, state: IndexState) -> PropertyIndexRecord {
        PropertyIndexRecord {
            name: name.to_string(),
            tenant: TenantId::DEFAULT,
            label: LabelId::new(7),
            property_key: StringId::new(11),
            property_name: "email".to_string(),
            state,
        }
    }

    #[test]
    fn catalog_key_is_disjoint() {
        // Disjoint from every other SYSTEM catalog key namespace.
        assert_ne!(PROPERTY_INDEX_CATALOG_KEY, 0); // tenants header
        assert_ne!(PROPERTY_INDEX_CATALOG_KEY, 2); // secondary root
        assert_ne!(PROPERTY_INDEX_CATALOG_KEY, 0x8000_0000_0000_0000); // tier
        assert_eq!(PROPERTY_INDEX_CATALOG_KEY, 0x2000_0000_0000_0000);
    }

    #[test]
    fn encode_decode_round_trip() {
        let recs = vec![
            rec("user_email", IndexState::Building),
            PropertyIndexRecord {
                name: "user_age".to_string(),
                tenant: TenantId::new(42),
                label: LabelId::new(9),
                property_key: StringId::new(13),
                property_name: "age".to_string(),
                state: IndexState::Online,
            },
        ];
        let encoded = encode_catalog(&recs);
        let decoded = decode_catalog(&encoded);
        assert_eq!(decoded, recs);
    }

    #[test]
    fn decode_empty_and_malformed_is_empty() {
        assert!(decode_catalog(&[]).is_empty());
        assert!(decode_catalog(&[99u8]).is_empty()); // unknown version
        // Truncated after count.
        let mut b = vec![CATALOG_FORMAT_VERSION];
        b.extend_from_slice(&5u32.to_le_bytes());
        assert!(decode_catalog(&b).is_empty());
    }

    #[test]
    fn create_inserts_building_and_lists() {
        let mgr = TxnManager::new();
        let cat = PropertyIndexCatalog::new();
        assert_eq!(
            cat.create_index(&mgr, rec("user_email", IndexState::Online), false)
                .unwrap(),
            CreateOutcome::Created
        );
        let listed = cat.list();
        assert_eq!(listed.len(), 1);
        // CREATE always starts Building regardless of the passed state.
        assert_eq!(listed[0].state, IndexState::Building);
        assert_eq!(listed[0].name, "user_email");
    }

    #[test]
    fn if_not_exists_is_idempotent() {
        let mgr = TxnManager::new();
        let cat = PropertyIndexCatalog::new();
        cat.create_index(&mgr, rec("e", IndexState::Building), true)
            .unwrap();
        assert_eq!(
            cat.create_index(&mgr, rec("e", IndexState::Building), true)
                .unwrap(),
            CreateOutcome::AlreadyExists
        );
        assert_eq!(cat.list().len(), 1, "IF NOT EXISTS must not duplicate");
    }

    #[test]
    fn create_without_if_not_exists_on_existing_errors() {
        let mgr = TxnManager::new();
        let cat = PropertyIndexCatalog::new();
        cat.create_index(&mgr, rec("e", IndexState::Building), false)
            .unwrap();
        let err = cat
            .create_index(&mgr, rec("e", IndexState::Building), false)
            .unwrap_err();
        assert!(matches!(err, PropertyIndexCatalogError::AlreadyExists { name } if name == "e"));
    }

    #[test]
    fn drop_removes_and_if_exists_semantics() {
        let mgr = TxnManager::new();
        let cat = PropertyIndexCatalog::new();
        cat.create_index(&mgr, rec("e", IndexState::Building), false)
            .unwrap();
        assert!(cat.drop_index(&mgr, TenantId::DEFAULT, "e", false).unwrap());
        assert!(cat.list().is_empty());
        // IF EXISTS on absent → no-op false.
        assert!(!cat.drop_index(&mgr, TenantId::DEFAULT, "e", true).unwrap());
        // non-IF-EXISTS on absent → NotFound.
        let err = cat
            .drop_index(&mgr, TenantId::DEFAULT, "e", false)
            .unwrap_err();
        assert!(matches!(err, PropertyIndexCatalogError::NotFound { name } if name == "e"));
    }

    #[test]
    fn set_state_online_flip_rides_caller_tx() {
        let mgr = TxnManager::new();
        let cat = PropertyIndexCatalog::new();
        cat.create_index(&mgr, rec("e", IndexState::Building), false)
            .unwrap();
        // Flip in a caller-held SYSTEM tx (the co-commit path).
        let mut tx = mgr.begin(TenantId::SYSTEM);
        let prev = cat
            .set_state_in(&mut tx, TenantId::DEFAULT, "e", IndexState::Online)
            .unwrap();
        assert_eq!(prev, IndexState::Building);
        tx.commit().unwrap();
        // Now durable + in-memory Online.
        assert_eq!(
            cat.resolve(TenantId::DEFAULT, "e").unwrap().state,
            IndexState::Online
        );
    }

    #[test]
    fn crash_mid_backfill_recovers_building_not_online() {
        // RED-on-revert oracle target: if the Online flip is committed
        // WITHOUT its watermark bundle (i.e. the tx is committed but the
        // backfill tail is lost), recovery must NOT come up Online.
        // Here we model "crash before the flip tx commits": the tx is
        // DROPPED (aborted), so the catalog header's last durable state
        // is Building. Recovery reads Building.
        let mgr = TxnManager::new();
        let cat = PropertyIndexCatalog::new();
        let lsn = cat
            .create_index_returning_lsn(&mgr, rec("e", IndexState::Building), false)
            .unwrap();
        // Begin the Online flip but DO NOT commit (simulated crash).
        {
            let mut tx = mgr.begin(TenantId::SYSTEM);
            cat.set_state_in(&mut tx, TenantId::DEFAULT, "e", IndexState::Online);
            // tx dropped here → abort → flip never durable.
        }
        // A fresh catalog recovers from durable MVCC state.
        let recovered = PropertyIndexCatalog::new();
        recovered.recover(&mgr, lsn);
        assert_eq!(
            recovered.resolve(TenantId::DEFAULT, "e").unwrap().state,
            IndexState::Building,
            "crash-mid-backfill must recover Building, never a partial Online"
        );
    }

    #[test]
    fn online_flip_committed_survives_recovery() {
        // Positive control: a PROPERLY co-committed Online flip is
        // durable and recovers Online (so the crash test above is not
        // vacuously always-Building).
        let mgr = TxnManager::new();
        let cat = PropertyIndexCatalog::new();
        cat.create_index(&mgr, rec("e", IndexState::Building), false)
            .unwrap();
        let mut tx = mgr.begin(TenantId::SYSTEM);
        cat.set_state_in(&mut tx, TenantId::DEFAULT, "e", IndexState::Online);
        let lsn = tx.commit().unwrap();
        let recovered = PropertyIndexCatalog::new();
        recovered.recover(&mgr, lsn);
        assert_eq!(
            recovered.resolve(TenantId::DEFAULT, "e").unwrap().state,
            IndexState::Online,
            "a co-committed Online flip must survive recovery"
        );
    }

    #[test]
    fn indexes_on_returns_declared_keys() {
        let mgr = TxnManager::new();
        let cat = PropertyIndexCatalog::new();
        cat.create_index(&mgr, rec("e", IndexState::Building), false)
            .unwrap();
        let on = cat.indexes_on(TenantId::DEFAULT, LabelId::new(7));
        assert_eq!(on, vec![(StringId::new(11), IndexState::Building)]);
        // A different label has no declared index.
        assert!(
            cat.indexes_on(TenantId::DEFAULT, LabelId::new(8))
                .is_empty()
        );
    }

    /// **#1401 — the crash case the existing `crash_mid_backfill` test
    /// does NOT model.** `crash_mid_backfill_recovers_building_not_online`
    /// crashes with an EMPTY tail (nothing was backfilled). This models a
    /// crash mid-backfill with a NON-EMPTY partial tail already durable —
    /// a second declared index (`e2`) is fully `Online`, `e`'s tail is
    /// partially built — then `e`'s flip tx is DROPPED. Recovery must
    /// bring `e` up `Building` (never a partial `Online`) while leaving
    /// the properly-flipped `e2` `Online`. This proves recover is not
    /// vacuously always-`Building` AND that a partial tail present in
    /// durable storage does not tempt a spurious `Online` for the index
    /// whose flip never committed.
    #[test]
    fn crash_partial_tail_present_flip_dropped_recovers_building() {
        let mgr = TxnManager::new();
        let cat = PropertyIndexCatalog::new();
        // e2 is a sibling index that DID complete + flip Online (its tail
        // is a "partial tail present" durable neighbour).
        cat.create_index(&mgr, rec("e2", IndexState::Building), false)
            .unwrap();
        {
            let mut tx = mgr.begin(TenantId::SYSTEM);
            cat.set_state_in(&mut tx, TenantId::DEFAULT, "e2", IndexState::Online);
            tx.commit().unwrap();
        }
        // e is mid-backfill: registered Building (its `create_index`
        // returns the durable lsn), some tail built, but the flip tx is
        // DROPPED (crash before the flip commits).
        let e_building_lsn = cat
            .create_index_returning_lsn(&mgr, rec("e", IndexState::Building), false)
            .unwrap();
        {
            let mut tx = mgr.begin(TenantId::SYSTEM);
            cat.set_state_in(&mut tx, TenantId::DEFAULT, "e", IndexState::Online);
            drop(tx); // abort → e's flip never durable.
        }
        // recover() reads the latest committed catalog value (the
        // `recovered_lsn` arg is reserved); e's last durable state is
        // Building, e2's is the committed Online flip.
        let recovered = PropertyIndexCatalog::new();
        recovered.recover(&mgr, e_building_lsn);
        assert_eq!(
            recovered.resolve(TenantId::DEFAULT, "e").unwrap().state,
            IndexState::Building,
            "e's flip never committed ⇒ recover Building even with a partial tail + a flipped sibling"
        );
        assert_eq!(
            recovered.resolve(TenantId::DEFAULT, "e2").unwrap().state,
            IndexState::Online,
            "e2's flip DID commit ⇒ stays Online (recover is not vacuously always-Building)"
        );
    }

    /// **#1401 — recover NEVER promotes an unrecognized durable state
    /// byte to `Online`.** Models a would-be future in-progress sentinel
    /// (byte 2, e.g. a `Flipping` marker a STEP4-before-STEP3 reorder
    /// might leave) persisted durably. The decode + reconcile layers
    /// together MUST bring it up `Building`, never `Online`. RED-on-revert
    /// oracle: if a future edit made `byte_state` map unknown → `Online`
    /// (a plausible mistake), the `recover`-force loop
    /// ([`PropertyIndexCatalog::reconcile_recovered_states`]) is the
    /// second line that still forces `Building` — deleting it flips this
    /// assertion RED.
    #[test]
    fn recover_never_promotes_unrecognized_state_byte() {
        let mgr = TxnManager::new();
        let cat = PropertyIndexCatalog::new();
        // Persist a record with a raw in-progress sentinel byte (2) that
        // the two-state machine does not emit.
        const FLIPPING_SENTINEL: u8 = 2;
        let lsn = cat
            .seed_durable_state_bytes(&mgr, &[(rec("e", IndexState::Building), FLIPPING_SENTINEL)]);
        let recovered = PropertyIndexCatalog::new();
        recovered.recover(&mgr, lsn);
        assert_eq!(
            recovered.resolve(TenantId::DEFAULT, "e").unwrap().state,
            IndexState::Building,
            "an unrecognized durable state byte must recover Building, never Online"
        );
    }

    /// Direct unit test of the recover reconcile step: it is a total
    /// downgrade of every non-`Online` record to `Building` and a no-op
    /// on `Online`. Pins the contract the two crash tests rely on.
    #[test]
    fn reconcile_recovered_states_downgrades_non_online_only() {
        let mut recs = vec![
            rec("building_stays", IndexState::Building),
            PropertyIndexRecord {
                name: "online_stays".to_string(),
                tenant: TenantId::DEFAULT,
                label: LabelId::new(7),
                property_key: StringId::new(11),
                property_name: "email".to_string(),
                state: IndexState::Online,
            },
        ];
        PropertyIndexCatalog::reconcile_recovered_states(&mut recs);
        assert_eq!(recs[0].state, IndexState::Building);
        assert_eq!(
            recs[1].state,
            IndexState::Online,
            "a complete Online record is never downgraded"
        );
    }
}
