//! **Property-index manager (#1366, task #248 — Phase 1).**
//!
//! The MCP-layer orchestrator for the user-visible secondary
//! node-property index. It is the ONLY place that touches JSON property
//! bags: per Prime Directive 7 it computes TYPED key deltas from the
//! bag and feeds them to storage/index through the published
//! [`arcgraph_index::SecondaryIndex`] + [`arcgraph_storage`] catalog
//! APIs; storage/index stay JSON-opaque.
//!
//! # What lives here (Phase 1 scope)
//!
//! - **`Value → IndexKeyInput`** — the single JSON→typed mapping seam
//!   (the one crossing the storage boundary), fed to
//!   [`arcgraph_index::canonical_row_key`] (RC-4/RC-5).
//! - **CREATE INDEX** — register `Building` in the durable
//!   [`PropertyIndexCatalog`] FIRST (so a concurrent write's maintain
//!   sees a non-empty catalog and is applied — #1401 missed-node fix),
//!   backfill by scanning the MVCC-visible nodes taken AFTER the
//!   register (extract the declared property → canonical key → insert
//!   into the per-index [`SecondaryIndex`]), then flip `Online`. The
//!   flip is a discrete SYSTEM tx AFTER the backfill loop; durability
//!   rests on append-all-then-flip ordering + synchronous-durable
//!   inserts + recover()-forces-non-`Online`→`Building`, NOT a shared
//!   `CommitBundle`.
//! - **maintenance** — `Self::maintain_node` computes declared-index
//!   key deltas for a write (old bag → new bag) and applies them
//!   (insert new; the old value becomes a verify-filtered ghost — RC-1
//!   insert-only posture on the read side). Applies while `Building`
//!   too (write-follows-declare, RC-2).
//! - **lookup** — `Self::lookup_candidates` returns candidate
//!   `NodeId`s (candidate-then-verify is the caller's MVCC hydrate +
//!   property recheck; ADR-023).
//!
//! # What is NOT here (Phase 2)
//!
//! No planner `PropertyIndexScan`, no cost model, no query-enable. The
//! catalog carries `Building`/`Online` so Phase 2 can gate
//! planner-visibility; Phase 1 only maintains the index.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc;

use arcgraph_core::{LabelId, NodeId, StringId, TenantId};
use arcgraph_index::{IndexKeyInput, PropertyValue, SecondaryIndex, SecondaryKey};
use arcgraph_query::executor::value::Value;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::property_index_catalog::{
    CreateOutcome, PropertyIndexCatalog, PropertyIndexCatalogError, PropertyIndexRecord,
};
use arcgraph_storage::secondary_handle::IndexState;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::WalHandle;
use parking_lot::RwLock;

/// Map an MCP property [`Value`] into the JSON-opaque
/// [`IndexKeyInput`] handed across the storage/index boundary. This is
/// the single JSON→typed seam (PD#7).
///
/// - `String` → [`IndexKeyInput::Str`] (RC-4 hash).
/// - `Integer` → [`IndexKeyInput::Int`] (RC-5 coercion / signed-int
///   handling downstream).
/// - `Boolean` → [`IndexKeyInput::Bool`].
/// - `Float` → [`IndexKeyInput::Float`] (integral floats coerce to the
///   integer key; fractional floats are unsupported — Float dropped).
/// - Everything else (`Null`, `List`, `Map`, `Node`, temporals, …) →
///   [`IndexKeyInput::Unsupported`] (absent-from-index path).
#[must_use]
pub fn value_to_index_input(v: &Value) -> IndexKeyInput<'_> {
    match v {
        Value::String(s) => IndexKeyInput::Str(s),
        Value::Integer(i) => IndexKeyInput::Int(*i),
        Value::Boolean(b) => IndexKeyInput::Bool(*b),
        Value::Float(f) => IndexKeyInput::Float(*f),
        // Null / List / Map / Node / Relationship / Path / temporals /
        // duration → not an indexable scalar (RC-5).
        _ => IndexKeyInput::Unsupported,
    }
}

/// Derive the canonical [`PropertyValue`] index key for a property
/// value, or `None` for an unsupported/unrepresentable value (the
/// absent-from-index path). Thin wrapper composing
/// [`value_to_index_input`] with [`arcgraph_index::canonical_row_key`].
#[must_use]
pub fn canonical_key_for(v: &Value) -> Option<PropertyValue> {
    arcgraph_index::canonical_row_key(value_to_index_input(v))
}

/// A single declared property index's runtime handle: the durable state
/// authority is the [`PropertyIndexCatalog`]; the data lives in a
/// dedicated per-index [`SecondaryIndex`] B+tree.
struct DeclaredIndex {
    label: LabelId,
    property_key: StringId,
    property_name: String,
    btree: Arc<SecondaryIndex>,
}

/// An opaque handle returned by [`PropertyIndexManager::register_building`]
/// and consumed by [`PropertyIndexManager::backfill_and_flip`]. It ties
/// the two halves of the split CREATE path together (register FIRST →
/// snapshot under the post-register catalog → backfill+flip) without
/// exposing the private `DeclaredIndex` to the caller. The wrapped
/// handle is already published in `self.indexes`, so a concurrent
/// `maintain_node` that observes the (now non-empty) catalog finds it.
pub struct BackfillHandle {
    declared: Arc<DeclaredIndex>,
    /// The catalog outcome (always `Created` — an `AlreadyExists`
    /// register returns `None`, never a handle), retained so the caller
    /// can propagate the DDL registration result.
    outcome: CreateOutcome,
}

impl std::fmt::Debug for BackfillHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackfillHandle")
            .field("label", &self.declared.label)
            .field("property", &self.declared.property_name)
            .field("outcome", &self.outcome)
            .finish()
    }
}

/// The per-`(tenant, index_name)` runtime B+tree handle map. Factored
/// into a `type` alias to keep [`PropertyIndexManager::indexes`] out of
/// `clippy::type_complexity`.
type IndexMap = HashMap<(TenantId, String), Arc<DeclaredIndex>>;

/// The typed, storage-opaque descriptor of a `CREATE INDEX … FOR
/// (n:Label) ON (n.prop)` — the natural decomposition of the DDL,
/// bundled into one struct so `PropertyIndexManager::create_index`
/// takes two arguments (the spec + the backfill node iterator) instead
/// of eight (`clippy::too_many_arguments`).
#[derive(Debug, Clone, Copy)]
pub struct CreateIndexSpec<'a> {
    /// The declared index name (`SHOW INDEXES` key; `(tenant, name)`
    /// uniqueness).
    pub tenant: TenantId,
    /// The DDL index name.
    pub name: &'a str,
    /// `IF NOT EXISTS` present (idempotent — a re-create is a no-op).
    pub if_not_exists: bool,
    /// The node label the index is `FOR (n:Label)`.
    pub label: LabelId,
    /// Interned property-KEY id (`ON (n.property)`). The property VALUE
    /// is keyed separately (RC-4 hash); this is the KEY name, interned.
    pub property_key: StringId,
    /// The human-readable property name (used to extract the value from
    /// a decoded property bag + for `SHOW INDEXES` display).
    pub property_name: &'a str,
}

/// The MCP-layer property-index manager. Shared (`Arc`) across every
/// `Clone` of the substrate so all Bolt connections observe one
/// per-tenant catalog + one B+tree per declared index.
#[derive(Clone)]
pub struct PropertyIndexManager {
    /// Durable Building→Online catalog (storage layer, MVCC-backed).
    catalog: Arc<PropertyIndexCatalog>,
    /// Per-`(tenant, index_name)` B+tree handles for the data.
    indexes: Arc<RwLock<IndexMap>>,
    txn_mgr: Arc<TxnManager>,
    allocator: Arc<PageAllocator>,
    wal: Option<WalHandle>,
}

impl std::fmt::Debug for PropertyIndexManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PropertyIndexManager")
            .field("declared_indexes", &self.indexes.read().len())
            .finish()
    }
}

/// A single record for `SHOW INDEXES` output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShowIndexRow {
    /// The index name.
    pub name: String,
    /// The label name (display).
    pub label: LabelId,
    /// The property name (display).
    pub property: String,
    /// The lifecycle state (`"BUILDING"` / `"ONLINE"`).
    pub state: &'static str,
}

impl PropertyIndexManager {
    /// Construct a manager over the shared MVCC / allocator / WAL
    /// handles. `recover` should be called once at boot to seed the
    /// catalog from durable state.
    #[must_use]
    pub fn new(
        catalog: Arc<PropertyIndexCatalog>,
        txn_mgr: Arc<TxnManager>,
        allocator: Arc<PageAllocator>,
        wal: Option<WalHandle>,
    ) -> Self {
        Self {
            catalog,
            indexes: Arc::new(RwLock::new(HashMap::new())),
            txn_mgr,
            allocator,
            wal,
        }
    }

    /// The set of declared property keys the maintenance path must
    /// touch for a write to a node of `(tenant, label)`, with each
    /// index's current state (so the caller can gate on
    /// `maintenance_active`).
    #[must_use]
    pub fn indexes_on(&self, tenant: TenantId, label: LabelId) -> Vec<(StringId, IndexState)> {
        self.catalog.indexes_on(tenant, label)
    }

    /// `SHOW INDEXES` — list the declared property indexes for a tenant
    /// with their states.
    #[must_use]
    pub fn show_indexes(&self, tenant: TenantId) -> Vec<ShowIndexRow> {
        self.catalog
            .list_for_tenant(tenant)
            .into_iter()
            .map(|r| ShowIndexRow {
                name: r.name,
                label: r.label,
                property: r.property_name,
                state: match r.state {
                    IndexState::Building => "BUILDING",
                    IndexState::Online => "ONLINE",
                },
            })
            .collect()
    }

    /// Get-or-create the per-index B+tree handle.
    fn btree_for(
        &self,
        tenant: TenantId,
        name: &str,
        label: LabelId,
        property_key: StringId,
        property_name: &str,
    ) -> Result<Arc<DeclaredIndex>, PropertyIndexError> {
        let key = (tenant, name.to_string());
        if let Some(existing) = self.indexes.read().get(&key) {
            return Ok(Arc::clone(existing));
        }
        let mut guard = self.indexes.write();
        // Double-check under the write lock.
        if let Some(existing) = guard.get(&key) {
            return Ok(Arc::clone(existing));
        }
        let btree = Arc::new(
            SecondaryIndex::new(
                Arc::clone(&self.txn_mgr),
                Arc::clone(&self.allocator),
                self.wal.clone(),
            )
            .map_err(|e| PropertyIndexError::Backend(e.to_string()))?,
        );
        let declared = Arc::new(DeclaredIndex {
            label,
            property_key,
            property_name: property_name.to_string(),
            btree,
        });
        guard.insert(key, Arc::clone(&declared));
        Ok(declared)
    }

    /// **STEP 1 of the split CREATE path — register the catalog
    /// `Building` record and publish the runtime handle, BEFORE the
    /// caller takes the backfill snapshot.**
    ///
    /// This is the missed-node W1 race fix (#1401). The old single-shot
    /// `create_index` took the backfill snapshot BEFORE it registered
    /// the catalog record, opening a window `[scan-begin, catalog-commit)`
    /// in which a concurrent writer's data commit was both (a) invisible
    /// to the frozen backfill snapshot AND (b) unmaintained (its
    /// best-effort maintain read an empty catalog and no-op'd) — a
    /// permanently missing entry. Registering FIRST establishes the
    /// invariant: **any concurrent writer is EITHER maintained (catalog
    /// non-empty when its maintain runs) OR captured in the post-register
    /// backfill snapshot — never neither.**
    ///
    /// Returns `Some(handle)` when the index was newly created (the
    /// caller MUST then scan the tenant's nodes under a snapshot taken
    /// AFTER this call returns and pass them to [`Self::backfill_and_flip`]
    /// with the returned handle), or `None` for the `IF NOT EXISTS`
    /// idempotent no-op (no backfill).
    ///
    /// # Handle-init ordering (RC-2 window close, preserved)
    ///
    /// The runtime B+tree handle is published into `self.indexes` BEFORE
    /// the catalog register commit. `maintain_node` gates on the CATALOG
    /// (`indexes_on` non-empty) then iterates `self.indexes` for the
    /// handle; installing the handle first means whenever the catalog
    /// declares the index the handle is already present, so no
    /// Building-window maintain finds a declared-but-handleless index.
    /// The handle carries no durable state (a page-store view over the
    /// shared allocator/WAL), so publishing it before the catalog commit
    /// is safe even if the register then fails: the orphaned handle is
    /// inert (no catalog record ⇒ `maintain_node`/`lookup` never route to
    /// it) and is overwritten by a later same-name create.
    pub fn register_building(
        &self,
        spec: CreateIndexSpec<'_>,
    ) -> Result<Option<BackfillHandle>, PropertyIndexError> {
        let CreateIndexSpec {
            tenant,
            name,
            if_not_exists,
            label,
            property_key,
            property_name,
        } = spec;

        let declared = self.btree_for(tenant, name, label, property_key, property_name)?;

        let record = PropertyIndexRecord {
            name: name.to_string(),
            tenant,
            label,
            property_key,
            property_name: property_name.to_string(),
            state: IndexState::Building,
        };
        let outcome = self
            .catalog
            .create_index(&self.txn_mgr, record, if_not_exists)?;
        if outcome == CreateOutcome::AlreadyExists {
            // Idempotent no-op: the pre-existing index keeps its state
            // and data. No backfill. (The handle we just get-or-created
            // is the pre-existing one — `btree_for` is get-or-create.)
            return Ok(None);
        }

        Ok(Some(BackfillHandle { declared, outcome }))
    }

    /// **STEP 2 of the split CREATE path — backfill the caller's
    /// post-register snapshot into the handle, then flip `Online`.**
    ///
    /// `nodes` is an iterator of `(NodeId, LabelId, &property-bag)` for
    /// the tenant's MVCC-visible nodes, read by the caller (the
    /// substrate) under a snapshot taken AFTER [`Self::register_building`]
    /// returned (which owns JSON decoding). Because the snapshot is taken
    /// post-register, every node the caller could NOT capture (a
    /// concurrent writer that committed after the register) is instead
    /// picked up by that writer's own `maintain_node` (the catalog is now
    /// non-empty). Overlap between "maintained during build" and "in the
    /// backfill snapshot" collapses via candidate-then-verify at read time
    /// (duplicate NodeId slots are verified then ignored — the B+tree insert
    /// is NOT idempotent, it fills successive slots).
    ///
    /// Backfill inserts every declared-property key, then flips `Online`
    /// in a discrete SYSTEM tx. Every backfill insert is already
    /// durable-on-return, so committing the flip AFTER the loop means no
    /// durable `Online` precedes a complete tail (append-all-then-flip
    /// ordering; recover() forces any non-`Online` state back to
    /// `Building`, restarting the backfill — candidate-then-verify at read
    /// time reconciles the partial tail (duplicate NodeId slots re-inserted
    /// by the restart are verified then ignored; the insert is NOT
    /// idempotent).
    pub fn backfill_and_flip<'a>(
        &self,
        handle: &BackfillHandle,
        tenant: TenantId,
        name: &str,
        nodes: impl Iterator<Item = (NodeId, LabelId, &'a BTreeMap<String, Value>)>,
    ) -> Result<CreateOutcome, PropertyIndexError> {
        let declared = &handle.declared;
        let label = declared.label;
        let property_name = declared.property_name.as_str();

        // Backfill into the already-published B+tree.
        let mut backfilled: u64 = 0;
        for (node_id, node_label, bag) in nodes {
            if node_label != label {
                continue;
            }
            let Some(value) = bag.get(property_name) else {
                continue; // node lacks the declared property.
            };
            if let Some(key) = self.insert_one(tenant, declared, value, node_id)? {
                let _ = key;
                backfilled += 1;
            }
            // A value that yields no key (unsupported / negative int) is
            // simply absent from the index — the residual-filter path.
        }

        // Flip Online AFTER the backfill loop. NOTE the durable
        // mechanism is append-all-then-flip ordering + synchronous
        // durable inserts + recover()-forces-non-Online→Building — NOT a
        // single shared CommitBundle (the flip is a discrete SYSTEM tx;
        // the backfill inserts are N independent per-insert bundles that
        // are each durable-on-return). If this flip commit fails, the
        // state stays Building (revert re-aligns the cache) and recovery
        // restarts the backfill.
        let mut tx = self.txn_mgr.begin(TenantId::SYSTEM);
        let prev = self
            .catalog
            .set_state_in(&mut tx, tenant, name, IndexState::Online);
        match tx.commit() {
            Ok(_lsn) => {}
            Err(e) => {
                if let Some(prev_state) = prev {
                    self.catalog.revert_state(tenant, name, prev_state);
                }
                return Err(PropertyIndexError::Backend(format!(
                    "Online flip commit failed (index stays Building; backfill will restart): {e}"
                )));
            }
        }
        let _ = backfilled;
        Ok(handle.outcome)
    }

    /// **`CREATE INDEX name [IF NOT EXISTS] FOR (n:Label) ON (n.prop)` —
    /// eager-snapshot convenience wrapper.**
    ///
    /// Registers, backfills the caller-supplied `nodes`, and flips
    /// `Online` in one call. This is the pre-`#1401` shape and is safe
    /// ONLY when `nodes` was NOT snapshotted before the catalog register
    /// (i.e. there is no concurrent writer, as in the single-threaded
    /// unit tests). Production goes through the split
    /// [`Self::register_building`] → post-register scan →
    /// [`Self::backfill_and_flip`] path (substrate `create_property_index`)
    /// so the backfill snapshot is taken AFTER the register and the
    /// missed-node W1 window is closed.
    ///
    /// **Visibility (#1401 footgun-close):** gated `#[cfg(test)]`. Every
    /// caller (the single-thread unit tests here + the `#[cfg(test)]`
    /// pre-fix RED-on-revert oracles in `substrate.rs`) is test-only; the
    /// production path never touches it. Keeping it out of the non-test
    /// build removes the pre-snapshotted-`nodes` W1 re-introduction footgun.
    #[cfg(test)]
    pub fn create_index<'a>(
        &self,
        spec: CreateIndexSpec<'_>,
        nodes: impl Iterator<Item = (NodeId, LabelId, &'a BTreeMap<String, Value>)>,
    ) -> Result<CreateOutcome, PropertyIndexError> {
        let tenant = spec.tenant;
        let name = spec.name;
        match self.register_building(spec)? {
            None => Ok(CreateOutcome::AlreadyExists),
            Some(handle) => self.backfill_and_flip(&handle, tenant, name, nodes),
        }
    }

    /// Insert one `(value → node)` into the index, returning the
    /// canonical key inserted (or `None` for an unsupported value that
    /// takes the absent-from-index path).
    fn insert_one(
        &self,
        tenant: TenantId,
        declared: &DeclaredIndex,
        value: &Value,
        node: NodeId,
    ) -> Result<Option<PropertyValue>, PropertyIndexError> {
        let Some(pv) = canonical_key_for(value) else {
            // Unsupported value: absent from index + warn (RC-5). The
            // write still succeeds (the caller does not treat this as an
            // error).
            tracing::warn!(
                target: "arcgraph_mcp::property_index",
                node = node.raw(),
                property = %declared.property_name,
                "property-index: unsupported value type; property absent from index (residual \
                 filter retained)"
            );
            return Ok(None);
        };
        let key = SecondaryKey::new(tenant, declared.label, declared.property_key, pv);
        declared
            .btree
            .insert(key, node)
            .map_err(|e| PropertyIndexError::Backend(e.to_string()))?;
        Ok(Some(pv))
    }

    /// **Commit-path maintenance (write-follows-declare, RC-1/RC-2).**
    ///
    /// Given the old and new property bags of `node` (label `label`),
    /// apply the declared-index key deltas: insert the NEW value's key
    /// for every declared, `maintenance_active` index whose property
    /// changed. The OLD value's entry is left as a ghost — the mandatory
    /// candidate-then-verify recheck (ADR-023) filters it for any
    /// snapshot (RC-1 insert-only posture; no eager removal that could
    /// create a snapshot-reader false negative).
    ///
    /// Applies for `Building` AND `Online` indexes (RC-2): a node
    /// written while `Building` must be findable once `Online`.
    pub fn maintain_node(
        &self,
        tenant: TenantId,
        node: NodeId,
        label: LabelId,
        old_bag: Option<&BTreeMap<String, Value>>,
        new_bag: &BTreeMap<String, Value>,
    ) -> Result<(), PropertyIndexError> {
        let declared_keys = self.catalog.indexes_on(tenant, label);
        if declared_keys.is_empty() {
            return Ok(());
        }
        // Resolve each declared index by (tenant, label) to its runtime
        // handle. We match on property_key so undeclared keys are never
        // indexed.
        let handles = self.indexes.read();
        for declared in handles.values() {
            if declared.label != label {
                continue;
            }
            // Only maintain declared indexes (property_key registered
            // in the catalog for this label). An index whose runtime
            // handle exists but whose catalog record was dropped is
            // skipped.
            let Some(&(_, state)) = declared_keys
                .iter()
                .find(|(pk, _)| *pk == declared.property_key)
            else {
                continue;
            };
            if !state.maintenance_active() {
                continue;
            }
            let new_val = new_bag.get(&declared.property_name);
            let old_val = old_bag.and_then(|b| b.get(&declared.property_name));
            // Insert the new value if it changed (or is newly present).
            // Equal values need no work: skipping avoids a redundant B+tree
            // insert (which is NOT idempotent — a re-insert would add a
            // duplicate NodeId slot, tolerated only by candidate-then-verify
            // at read time).
            if new_val != old_val {
                if let Some(v) = new_val {
                    self.insert_one(tenant, declared, v, node)?;
                }
                // The old value's entry is intentionally NOT removed
                // here (RC-1 insert-only). It is a verify-filtered ghost.
            }
        }
        Ok(())
    }

    /// **Lookup** — return candidate `NodeId`s for `(tenant, label,
    /// property = value)`. The caller MUST verify each candidate by
    /// hydrating it through its MVCC snapshot and re-checking label +
    /// property equality (ADR-023 candidate-then-verify) before yielding
    /// — a 56-bit hash collision or a stale/ghost entry is dropped by
    /// that recheck.
    ///
    /// **RC-6 planner-visible gate (#1366 Phase 2).** Candidates are
    /// served ONLY when the matched `(label, property)` index is in the
    /// `Online` state ([`IndexState::planner_visible`]). A `Building`
    /// index — whose backfill tail is incomplete — returns an EMPTY vec
    /// here even though its runtime B+tree handle exists and is being
    /// maintained: serving a Building index for query results risks a
    /// FALSE NEGATIVE (a node written after the backfill snapshot that
    /// the tail has not yet covered would be missed). This is the same
    /// gate the planner applies at plan time
    /// ([`arcgraph_query::semantic::CatalogProvider::online_property_index`]);
    /// enforcing it HERE too is defense-in-depth — a direct
    /// `lookup_candidates` caller can never accidentally read a Building
    /// index.
    ///
    /// Returns an empty vec when the value is unsupported (no key), when
    /// no index is declared on `(tenant, label)` for a matching property,
    /// OR when the matched index is not yet `Online`. NOTE: this NEVER
    /// interns the value (RC-3/RC-4 — strings are hashed in place), so a
    /// never-seen-value lookup does not grow the `InternTable`.
    pub fn lookup_candidates(
        &self,
        tenant: TenantId,
        label: LabelId,
        property_name: &str,
        value: &Value,
    ) -> Result<Vec<NodeId>, PropertyIndexError> {
        let Some(pv) = canonical_key_for(value) else {
            return Ok(Vec::new());
        };
        // The catalog is the state authority (the runtime handle carries
        // no state). Resolve the declared key set for (tenant, label) so
        // we can gate on `planner_visible()`.
        let declared_keys = self.catalog.indexes_on(tenant, label);
        let handles = self.indexes.read();
        for declared in handles.values() {
            if declared.label == label && declared.property_name == property_name {
                // RC-6: only serve an ONLINE index. The catalog record
                // for this index's property_key must be planner-visible;
                // a Building (or missing) record → SKIP this handle and
                // keep looking (a rare second index on the same
                // (label, property) might be Online). Only when NO Online
                // handle matches do we return empty.
                let online = declared_keys
                    .iter()
                    .any(|(pk, state)| *pk == declared.property_key && state.planner_visible());
                if !online {
                    continue;
                }
                let key = SecondaryKey::new(tenant, declared.label, declared.property_key, pv);
                return declared
                    .btree
                    .lookup(key)
                    .map_err(|e| PropertyIndexError::Backend(e.to_string()));
            }
        }
        Ok(Vec::new())
    }

    /// **#1366 (Phase 2) — the RC-6 planner-visible check.** Whether an
    /// **Online** secondary index is declared on `(tenant, label,
    /// property_name)`. Consulted by the query planner (through the
    /// `CatalogProvider::online_property_index` adapter) to decide
    /// whether to route a point lookup to the index.
    ///
    /// Returns `true` ONLY when a declared index on exactly this
    /// `(label, property)` pair is `Online` (`planner_visible()`); a
    /// `Building`, absent, or dropped index returns `false` — the planner
    /// then keeps the full-scan path. Resolves the property NAME → the
    /// runtime handle's `property_key`, then gates on the catalog state
    /// (the state authority), mirroring `lookup_candidates`'s gate.
    #[must_use]
    pub fn has_online_index(&self, tenant: TenantId, label: LabelId, property_name: &str) -> bool {
        let declared_keys = self.catalog.indexes_on(tenant, label);
        if declared_keys.is_empty() {
            return false;
        }
        let handles = self.indexes.read();
        handles.values().any(|declared| {
            declared.label == label
                && declared.property_name == property_name
                && declared_keys
                    .iter()
                    .any(|(pk, state)| *pk == declared.property_key && state.planner_visible())
        })
    }

    /// **`DROP INDEX name [IF EXISTS]`.** Removes the catalog record +
    /// the runtime B+tree handle. Idempotent under `IF EXISTS`.
    pub fn drop_index(
        &self,
        tenant: TenantId,
        name: &str,
        if_exists: bool,
    ) -> Result<bool, PropertyIndexError> {
        let removed = self
            .catalog
            .drop_index(&self.txn_mgr, tenant, name, if_exists)?;
        if removed {
            self.indexes.write().remove(&(tenant, name.to_string()));
        }
        Ok(removed)
    }
}

/// Error surface for the MCP property-index manager.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PropertyIndexError {
    /// The durable catalog rejected the operation (already-exists /
    /// not-found / commit failure).
    #[error(transparent)]
    Catalog(#[from] PropertyIndexCatalogError),
    /// The underlying B+tree backend failed.
    #[error("property-index backend: {0}")]
    Backend(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value_bag(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn mgr() -> PropertyIndexManager {
        let txn_mgr = Arc::new(TxnManager::new());
        let allocator = Arc::new(PageAllocator::new());
        let catalog = Arc::new(PropertyIndexCatalog::new());
        PropertyIndexManager::new(catalog, txn_mgr, allocator, None)
    }

    /// Build a [`CreateIndexSpec`] for the tests (DEFAULT tenant).
    fn spec<'a>(
        name: &'a str,
        if_not_exists: bool,
        label: LabelId,
        pk: StringId,
        property_name: &'a str,
    ) -> CreateIndexSpec<'a> {
        CreateIndexSpec {
            tenant: TenantId::DEFAULT,
            name,
            if_not_exists,
            label,
            property_key: pk,
            property_name,
        }
    }

    /// Empty backfill iterator (no seed nodes).
    fn no_nodes() -> std::iter::Empty<(NodeId, LabelId, &'static BTreeMap<String, Value>)> {
        std::iter::empty()
    }

    /// **#1401 — append-all-then-flip ordering (STEP3-before-STEP4).**
    /// `register_building` leaves the index `Building`; `backfill_and_flip`
    /// flips `Online` only AFTER the backfill loop. Observed at the split
    /// boundary: the state is `Building` between register and flip, and
    /// `Online` after. This is the runtime guard the recover
    /// force-to-`Building` belt (property_index_catalog.rs) documents —
    /// a STEP4-before-STEP3 reorder (flip before the tail) would make the
    /// mid-assert observe `Online` before the inserts, flipping it RED.
    #[test]
    fn backfill_inserts_precede_online_flip() {
        let m = mgr();
        let tenant = TenantId::DEFAULT;
        let label = LabelId::new(3);
        let pk = StringId::new(7);
        let bags: Vec<(NodeId, LabelId, BTreeMap<String, Value>)> = (1..=4)
            .map(|i| {
                (
                    NodeId::new(i),
                    label,
                    value_bag(&[("email", Value::String(format!("u{i}@x.com")))]),
                )
            })
            .collect();

        // STEP 1: register — must be Building, NOT Online (a flip here
        // would be STEP4-before-STEP3).
        let handle = m
            .register_building(spec("e", false, label, pk, "email"))
            .expect("register")
            .expect("Created ⇒ Some(handle)");
        assert_eq!(
            m.show_indexes(tenant)[0].state,
            "BUILDING",
            "register_building must NOT flip Online (append-all-then-flip)"
        );

        // STEP 2: backfill + flip — Online only AFTER the loop.
        let outcome = m
            .backfill_and_flip(
                &handle,
                tenant,
                "e",
                bags.iter().map(|(n, l, b)| (*n, *l, b)),
            )
            .expect("backfill_and_flip");
        assert_eq!(outcome, CreateOutcome::Created);
        assert_eq!(
            m.show_indexes(tenant)[0].state,
            "ONLINE",
            "backfill_and_flip flips Online after the tail is inserted"
        );
        // And every backfilled node is a candidate (the tail is present
        // before Online — not a vacuous flip over an empty index).
        for i in 1..=4u64 {
            let cands = m
                .lookup_candidates(
                    tenant,
                    label,
                    "email",
                    &Value::String(format!("u{i}@x.com")),
                )
                .unwrap();
            assert!(
                cands.contains(&NodeId::new(i)),
                "u{i} in the tail before Online"
            );
        }
    }

    #[test]
    fn value_to_input_mapping() {
        assert_eq!(
            value_to_index_input(&Value::Integer(5)),
            IndexKeyInput::Int(5)
        );
        assert_eq!(
            value_to_index_input(&Value::Boolean(true)),
            IndexKeyInput::Bool(true)
        );
        assert_eq!(
            value_to_index_input(&Value::Null),
            IndexKeyInput::Unsupported
        );
        assert_eq!(
            value_to_index_input(&Value::List(vec![])),
            IndexKeyInput::Unsupported
        );
    }

    #[test]
    fn integral_float_and_int_produce_same_key() {
        // RC-5: n.age = 42.0 must key like stored int 42.
        assert_eq!(
            canonical_key_for(&Value::Float(42.0)),
            canonical_key_for(&Value::Integer(42))
        );
    }

    #[test]
    fn create_backfill_and_lookup_finds_seeded_nodes() {
        let m = mgr();
        let tenant = TenantId::DEFAULT;
        let label = LabelId::new(3);
        let pk = StringId::new(7);
        // Seed N nodes with an "email" property.
        let bags: Vec<(NodeId, LabelId, BTreeMap<String, Value>)> = (1..=5)
            .map(|i| {
                (
                    NodeId::new(i),
                    label,
                    value_bag(&[("email", Value::String(format!("user{i}@x.com")))]),
                )
            })
            .collect();
        let iter = bags.iter().map(|(n, l, b)| (*n, *l, b));
        let outcome = m
            .create_index(spec("email_idx", false, label, pk, "email"), iter)
            .unwrap();
        assert_eq!(outcome, CreateOutcome::Created);
        // Now Online.
        assert_eq!(
            m.show_indexes(tenant)[0].state,
            "ONLINE",
            "backfill-complete flips Online"
        );
        // Every seeded value is findable.
        for i in 1..=5u64 {
            let cands = m
                .lookup_candidates(
                    tenant,
                    label,
                    "email",
                    &Value::String(format!("user{i}@x.com")),
                )
                .unwrap();
            assert!(
                cands.contains(&NodeId::new(i)),
                "backfilled node {i} must be a candidate"
            );
        }
    }

    #[test]
    fn if_not_exists_idempotent_no_dup() {
        let m = mgr();
        let tenant = TenantId::DEFAULT;
        let label = LabelId::new(1);
        let pk = StringId::new(2);
        m.create_index(spec("e", true, label, pk, "email"), no_nodes())
            .unwrap();
        let out2 = m
            .create_index(spec("e", true, label, pk, "email"), no_nodes())
            .unwrap();
        assert_eq!(out2, CreateOutcome::AlreadyExists);
        assert_eq!(m.show_indexes(tenant).len(), 1);
    }

    /// **GENUINE Building-state maintenance test (RC-2).** Unlike a
    /// post-Online write, this exercises `maintain_node` WHILE the
    /// index state is `Building` — by registering the record + handle
    /// directly in `Building` (mirroring what a mid-backfill concurrent
    /// commit sees) WITHOUT running the Online flip, then asserting the
    /// write is applied and found once the flip lands.
    ///
    /// RED-on-revert: gating maintenance on `planner_visible()`
    /// (Online-only) makes the Building-window write ABSENT. The
    /// positive control at the end confirms the value is genuinely
    /// stored (not a vacuous empty-index pass).
    #[test]
    fn maintain_during_building_state_is_applied_rc2() {
        use arcgraph_storage::property_index_catalog::PropertyIndexRecord;

        let m = mgr();
        let tenant = TenantId::DEFAULT;
        let label = LabelId::new(4);
        let pk = StringId::new(9);

        // Publish the runtime handle (as create_index does first), then
        // register the catalog record as Building — but DO NOT flip
        // Online. The index is now in the exact `Building` state a
        // concurrent commit sees mid-backfill.
        let _declared = m
            .btree_for(tenant, "e", label, pk, "email")
            .expect("handle");
        let rec = PropertyIndexRecord {
            name: "e".to_string(),
            tenant,
            label,
            property_key: pk,
            property_name: "email".to_string(),
            state: IndexState::Building,
        };
        m.catalog
            .create_index(&m.txn_mgr, rec, false)
            .expect("register Building");
        // Precondition: state is Building (NOT Online) at maintenance time.
        assert_eq!(
            m.catalog.resolve(tenant, "e").unwrap().state,
            IndexState::Building,
            "index must be Building when maintain_node runs (the RC-2 window)"
        );
        assert!(
            m.show_indexes(tenant)[0].state == "BUILDING",
            "SHOW confirms Building"
        );

        // A write DURING Building is maintained (write-follows-declare).
        let new_bag = value_bag(&[("email", Value::String("during@x.com".into()))]);
        m.maintain_node(tenant, NodeId::new(99), label, None, &new_bag)
            .unwrap();

        // Now flip Online (backfill complete) and assert the
        // Building-window write is findable — the false-negative RC-2
        // closes.
        let mut tx = m.txn_mgr.begin(TenantId::SYSTEM);
        m.catalog
            .set_state_in(&mut tx, tenant, "e", IndexState::Online);
        tx.commit().unwrap();
        assert_eq!(
            m.catalog.resolve(tenant, "e").unwrap().state,
            IndexState::Online
        );

        let cands = m
            .lookup_candidates(
                tenant,
                label,
                "email",
                &Value::String("during@x.com".into()),
            )
            .unwrap();
        assert!(
            cands.contains(&NodeId::new(99)),
            "a node written while Building must be found once Online (RC-2 write-follows-declare)"
        );
    }

    #[test]
    fn negative_int_absent_from_index_write_succeeds() {
        // RC-5: negative int → unsupported → absent, but the maintain
        // call still SUCCEEDS (write not rejected).
        let m = mgr();
        let tenant = TenantId::DEFAULT;
        let label = LabelId::new(5);
        let pk = StringId::new(11);
        m.create_index(spec(" age_idx", false, label, pk, "age"), no_nodes())
            .unwrap();
        let bag = value_bag(&[("age", Value::Integer(-5))]);
        // Must NOT error.
        m.maintain_node(tenant, NodeId::new(1), label, None, &bag)
            .unwrap();
        // And the negative value produces no candidates.
        let cands = m
            .lookup_candidates(tenant, label, "age", &Value::Integer(-5))
            .unwrap();
        assert!(cands.is_empty(), "negative int is absent from the index");
    }

    #[test]
    fn drop_index_removes_it() {
        let m = mgr();
        let tenant = TenantId::DEFAULT;
        let label = LabelId::new(6);
        let pk = StringId::new(13);
        m.create_index(spec("e", false, label, pk, "email"), no_nodes())
            .unwrap();
        assert_eq!(m.show_indexes(tenant).len(), 1);
        assert!(m.drop_index(tenant, "e", false).unwrap());
        assert!(m.show_indexes(tenant).is_empty());
        // IF EXISTS on absent → false, no error.
        assert!(!m.drop_index(tenant, "e", true).unwrap());
    }
}
