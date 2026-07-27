//! ADR-152 W27-α — property-bag persistence helpers for the
//! [`crate::storage::substrate::CrudExecutorSubstrate`].
//!
//! Two helpers bridge the executor's `&[(String, Value)]` /
//! `BTreeMap<String, Value>` property bag and the storage layer's
//! [`arcgraph_storage::crud::PropertyData`] encoding:
//!
//! - [`properties_to_property_data`] — serialize a non-empty bag as
//!   canonical JSON bytes routed into [`PropertyData::Blob`]; empty
//!   bag short-circuits to [`PropertyData::Empty`].
//! - `record_property_bag` / `rel_record_property_bag` — decode
//!   the blob (if any) from a [`NodeRecord`] / [`RelRecord`]'s
//!   `property_ref` slot, fetch the bytes via
//!   [`arcgraph_storage::blob::BlobStore::get`], deserialize JSON
//!   back into a runtime [`BTreeMap<String, Value>`].
//!
//! # ADR provenance
//! - **ADR-152 §D-1** — write-path: `&[(String, Value)]` → `PropertyData::Blob`.
//! - **ADR-152 §D-3** — read-path: `NodeRecord.property_ref` → `BTreeMap<String, Value>`.
//! - **ADR-022** — blob chain durability via `BlobStore::put_logged_and_stage`.
//! - **ADR-018** — MVCC version-chain visibility for the read snapshot.
//!
//! # Round-trip discipline
//!
//! The encoder routes each [`Value`] cell through
//! [`Value::to_json_value`]; the decoder routes JSON back via
//! [`Value::try_from_json_property_value`]. Stored-bag objects decode
//! unconditionally as `Value::Map`: internal Node / Relationship / Path
//! views exist only in query-result rows, so entity-shape detection must
//! not reinterpret customer property objects. Variant-level round-trip
//! is lossless for `Null` / `Boolean` / `Integer` (within i64) /
//! `Float` (finite) / `String` / nested `List` / `Map`; lossy edges per
//! `Value::to_json_value`'s rustdoc (NaN/Inf → JSON null;
//! `u64 > i64::MAX` widens to f64).
//!
//! Composite literal variants (`List` / `Map` / `Temporal` /
//! `LocalDateTime` / `Date` / `Duration` / `Decimal`) inside a
//! property bag are NOT admitted by the executor's literal-only
//! narrowing per ADR-147 §D-4 inherited; they'd reach this helper
//! only on a forward-compat write-path and round-trip via the same
//! [`Value::to_json_value`] / [`Value::try_from_json_property_value`]
//! bridge.

use std::collections::BTreeMap;

use arcgraph_core::{LabelId, NodeId, NodeRecord, RelRecord, TenantId};
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_storage::blob::BlobStore;
use arcgraph_storage::crud::{CrudStore, PropertyData};
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::prop_block::{
    EncodedPropBlock, OverflowView, PROP_BLOCK_DISCRIMINANT, PrimaryLookup, PropBlockBuilder,
    PropBlockView, PropValue, PropValueRef,
};
use arcgraph_storage::property::BlobRef;
use serde_json::{Map as JsonMap, Value as JsonValue};
use thiserror::Error;

/// First byte of every M1-era legacy JSON bag payload: `serde_json`
/// object serialization always begins `b'{'`. The M2 mixed-store
/// payload dispatch keys on this vs [`PROP_BLOCK_DISCRIMINANT`]
/// (design §M2.6 — during the migrate-on-open window both encodings
/// coexist and reads must serve both).
const LEGACY_JSON_DISCRIMINANT: u8 = b'{';

/// Faults surfaced by the v2 M2 typed property-payload bridge.
///
/// LOUD by contract (design §M2.2: "the degrade becomes 'block version
/// unknown → reject' — a corruption signal, not a silent empty bag").
/// The M1 JSON path's warn-degrade-to-empty is exactly what these
/// retire for typed payloads.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PropPayloadError {
    /// The payload bytes violate the typed-block format (or carry a
    /// first byte that is neither a typed block nor a legacy JSON
    /// object), or reference an interned key this store does not know.
    #[error("corrupt property payload on {kind} {rec_id}: {reason}")]
    Corrupt {
        /// `"node"` / `"rel"`.
        kind: &'static str,
        /// Record id (diagnostics).
        rec_id: u64,
        /// What was violated.
        reason: String,
    },
    /// The blob/overflow bytes could not be fetched. Distinct from
    /// `Corrupt`: fetch faults on the MAIN payload keep the ratified
    /// ADR-149/ADR-152 cross-snapshot degrade at the callers that had
    /// it; an OVERFLOW fetch fault inside an otherwise-valid typed
    /// block is corruption-adjacent and stays loud.
    #[error("property payload fetch failed on {kind} {rec_id}: {reason}")]
    Fetch {
        /// `"node"` / `"rel"`.
        kind: &'static str,
        /// Record id (diagnostics).
        rec_id: u64,
        /// Underlying blob-store fault.
        reason: String,
    },
    /// Encode-side fault (typed-block build or overflow staging).
    #[error("typed property encode failed: {reason}")]
    Encode {
        /// What failed.
        reason: String,
    },
}

/// The substrate boundary translation: every typed-payload fault
/// surfaces as the loud
/// `SubstrateAccessError::CorruptPropertyPayload` variant (v2 M2 —
/// the query layer distinguishes data corruption from transient I/O).
/// Codec-local errors translate at the public boundary per
/// `docs/codec-error-translation.md`.
impl From<PropPayloadError> for arcgraph_query::executor::substrate::SubstrateAccessError {
    fn from(e: PropPayloadError) -> Self {
        arcgraph_query::executor::substrate::SubstrateAccessError::CorruptPropertyPayload(
            e.to_string(),
        )
    }
}

/// Serialize a property bag (slice form, executor-side runtime
/// values) into a [`PropertyData`] payload per ADR-152 §D-1.
///
/// Returns [`PropertyData::Empty`] for empty bags so the storage
/// fast-path applies. Non-empty bags serialize as a canonical
/// `serde_json::Map` (key-ordered ascending; BTreeMap iter
/// preserves order on the encode side) and route through
/// [`PropertyData::Blob`]. The serialization uses the
/// already-existing [`Value::to_json_value`] per-cell bridge so the
/// round-trip semantics match the M5↔M4 contract surface per ADR-038
/// amendment-03.
///
/// Duplicate keys in the input slice resolve via **last-wins** —
/// the lowering / type-check passes are expected to reject duplicate
/// property keys upstream, but this fallback prevents a panic in
/// case a future surface admits them.
#[must_use]
pub fn properties_to_property_data(props: &[(String, Value)]) -> PropertyData {
    if props.is_empty() {
        return PropertyData::Empty;
    }
    let mut map = JsonMap::new();
    for (k, v) in props {
        map.insert(k.clone(), v.to_json_value());
    }
    bag_to_property_data(map)
}

/// Serialize a [`BTreeMap`] property bag (executor-side runtime
/// values) into a [`PropertyData`] payload. Mirrors
/// [`properties_to_property_data`] but accepts the post-merge /
/// post-remove bag shape that SET / REMOVE substrate paths route
/// through.
#[must_use]
pub fn property_map_to_property_data(map: &BTreeMap<String, Value>) -> PropertyData {
    if map.is_empty() {
        return PropertyData::Empty;
    }
    let mut json_map = JsonMap::new();
    for (k, v) in map {
        json_map.insert(k.clone(), v.to_json_value());
    }
    bag_to_property_data(json_map)
}

fn bag_to_property_data(map: JsonMap<String, JsonValue>) -> PropertyData {
    let bytes = serde_json::to_vec(&JsonValue::Object(map)).unwrap_or_default();
    if bytes.is_empty() {
        // Defensive: a serialization failure routes back through the
        // empty fast-path rather than panicking. In practice
        // `serde_json::to_vec` over a `JsonMap` cannot fail (no IO).
        return PropertyData::Empty;
    }
    PropertyData::Blob(bytes)
}

// ─────────────────────────────────────────────────────────────────────
// v2 M2 — Value ↔ PropValue mapping (the differential-equality bridge)
// ─────────────────────────────────────────────────────────────────────
//
// THE CONTRACT (build-plan §2 M2 EXIT gate 5): for every projection,
// an M2 typed store must materialize values IDENTICAL to what the M1
// JSON store materializes. The M1 read is the composition
// `try_from_json_value ∘ json-parse ∘ json-serialize ∘ to_json_value`.
// These mappings reproduce that composition exactly:
//
// - Scalars map DIRECTLY (provably identical to the composition —
//   `Integer` round-trips as-is; finite `Float` round-trips bit-exact
//   through ryu; NON-FINITE floats normalize to Null exactly as
//   `to_json_value`'s NaN/±Inf → JSON null does).
// - Everything nested / structural (List, Map, Node, Relationship,
//   Path, the temporal/decimal family) routes THROUGH
//   `Value::to_json_value` and is stored as the serialized JSON bytes
//   (opaque to storage — ADR-089 §D-1), so the read side decodes with
//   the SAME `try_from_json_value` over the SAME bytes ⇒ identical by
//   construction (including the entity-shape detection and the
//   temporal-family's ISO-string materialization).

/// M1 float fidelity: the value an M1 JSON bag read RETURNS for a
/// stored float — i.e. `f` pushed through serde_json's serialize +
/// DEFAULT (non-`float_roundtrip`) parse, which is imprecise by up to
/// ~1 ULP for some doubles.
///
/// The M2 typed store is bit-exact by nature, but M2 is
/// REPRESENTATION-ONLY (design §0.3 consistency-neutral): visible
/// values must not change even by an ULP, even for the better —
/// otherwise (a) the EXIT gate-5 differential ("identical materialized
/// values for EVERY projection" vs an M1 store) fails, (b) a bag read
/// DURING migration returns a different float than the same bag read
/// after it, and (c) M3's differential-replay oracle inherits the ULP
/// noise. The write path therefore stores exactly what M1 would have
/// persisted-and-re-read. (Caught by
/// `m2_write_path_materializes_identically_to_m1` — the oracle
/// working as designed.) Exact floats are a deliberate post-format-
/// chain change with its own gate, not a side effect of M2.
fn m1_float_fidelity(f: f64) -> f64 {
    let printed = serde_json::to_vec(&f).unwrap_or_default();
    serde_json::from_slice::<f64>(&printed).unwrap_or(f)
}

/// Encode-side: one executor [`Value`] → the storage-grain
/// [`PropValue`].
fn value_to_prop_value(v: &Value) -> PropValue {
    match v {
        Value::Null => PropValue::Null,
        Value::Boolean(b) => PropValue::Bool(*b),
        Value::Integer(i) => PropValue::Int(*i),
        // Finite floats store the M1-fidelity value (see
        // `m1_float_fidelity` — the ULP-identical differential pin).
        Value::Float(f) if f.is_finite() => PropValue::Float(m1_float_fidelity(*f)),
        // Non-finite: the M1 bridge stores JSON null (to_json_value).
        Value::Float(_) => PropValue::Null,
        Value::String(s) => PropValue::Str(s.clone()),
        // Structural / temporal family: store the M1 wire form (the
        // nested bytes re-parse through the SAME serde_json default
        // parse M1 uses, so floats inside lists/maps are M1-identical
        // by construction).
        other => json_value_to_prop_value(&other.to_json_value()),
    }
}

/// Migration-side: one raw [`JsonValue`] (from an M1 legacy bag) →
/// [`PropValue`], preserving the `json_number_to_value` decode rules
/// (`as_i64` → Int; `as_u64` overflow → Float widening; else Float).
fn json_value_to_prop_value(jv: &JsonValue) -> PropValue {
    match jv {
        JsonValue::Null => PropValue::Null,
        JsonValue::Bool(b) => PropValue::Bool(*b),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                PropValue::Int(i)
            } else if let Some(u) = n.as_u64() {
                // Mirror `json_number_to_value`: u64 beyond i64 widens
                // to f64 (lossy past 2^53, documented there).
                PropValue::Float(u as f64)
            } else {
                // serde_json's third repr; total by construction.
                PropValue::Float(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        JsonValue::String(s) => PropValue::Str(s.clone()),
        arr @ JsonValue::Array(_) => {
            PropValue::ListOpaque(serde_json::to_vec(arr).unwrap_or_default())
        }
        obj @ JsonValue::Object(_) => {
            PropValue::MapOpaque(serde_json::to_vec(obj).unwrap_or_default())
        }
    }
}

/// Decode-side: one borrowed [`PropValueRef`] → an executor [`Value`].
///
/// Returns `Ok(None)` for the per-entry drop cases the M1 path also
/// drops (a `Value::try_from_json_value` bridge failure on nested
/// bytes — e.g. `NestingTooDeep` — is legitimate data the M1 reader
/// warn-drops per entry; the differential contract pins matching
/// behavior). A serde PARSE failure of nested bytes is different:
/// those bytes were serialized by our own encoder, so a parse failure
/// is corruption → LOUD `Err`.
fn prop_value_ref_to_value(
    v: PropValueRef<'_>,
    key: &str,
    kind: &'static str,
    rec_id: u64,
) -> Result<Option<Value>, PropPayloadError> {
    let nested = |bytes: &[u8], what: &'static str| -> Result<Option<Value>, PropPayloadError> {
        let jv: JsonValue =
            serde_json::from_slice(bytes).map_err(|e| PropPayloadError::Corrupt {
                kind,
                rec_id,
                reason: format!("nested {what} bytes under key `{key}` failed to parse: {e}"),
            })?;
        // #1444 (#1383): stored-bag values decode through the MAP-ONLY
        // property bridge — a nested object is a customer property
        // value, NEVER an entity view.
        match Value::try_from_json_property_value(&jv) {
            Ok(v) => Ok(Some(v)),
            Err(e) => {
                // The M1 per-entry drop posture, byte-for-byte
                // (`decode_blob_bag`'s per-entry warn) — NOT corruption.
                tracing::warn!(
                    kind,
                    rec_id,
                    key = %key,
                    error = %e,
                    "M2 typed property scan: nested value bridge failed; dropping entry \
                     (matches the M1 per-entry decode-drop posture)",
                );
                Ok(None)
            }
        }
    };
    match v {
        PropValueRef::Null => Ok(Some(Value::Null)),
        PropValueRef::Int(i) => Ok(Some(Value::Integer(i))),
        PropValueRef::Float(f) => Ok(Some(Value::Float(f))),
        PropValueRef::Bool(b) => Ok(Some(Value::Boolean(b))),
        PropValueRef::Str(s) => Ok(Some(Value::String(s.to_owned()))),
        PropValueRef::ListOpaque(b) => nested(b, "list"),
        PropValueRef::MapOpaque(b) => nested(b, "map"),
        // No v1.0-α producer exists for these tags (module docs in
        // `prop_block`); encountering one is a format violation.
        PropValueRef::Bytes(_) | PropValueRef::Temporal(_) => Err(PropPayloadError::Corrupt {
            kind,
            rec_id,
            reason: format!(
                "reserved type tag (Bytes/Temporal) under key `{key}` — no producer exists at \
                 this engine version"
            ),
        }),
    }
}

// ─────────────────────────────────────────────────────────────────────
// v2 M2 — typed encode (write path + the migration re-encoder)
// ─────────────────────────────────────────────────────────────────────

/// A built typed block ready for the storage layer's two-phase
/// staging (the storage-grain `TypedBagParts` — defined in
/// `arcgraph_storage::prop_block` so the M2 migrate-on-open sweep and
/// `crud::PropertyData::TypedBlock` can name it without reaching into
/// this crate; PD#7 bounded contexts).
pub use arcgraph_storage::prop_block::TypedBagParts as TypedBagPayload;

/// Build the typed-block payload for a property bag, interning key
/// names through the EXISTING `InternTable` (design §M2.1 — WAL-logged
/// via `intern_logged` when `wal` is present, so a freshly-allocated
/// key_id's `InternString` record lands BEFORE the commit that
/// references it, exactly the label-intern durability ordering).
///
/// Budget (PD#5): O(n log n) builder sort + O(payload bytes); one
/// intern probe/insert per DISTINCT key name; zero JSON encode for
/// scalar values (the M1 JSON tax this replaces).
pub fn build_typed_bag<'a>(
    props: impl IntoIterator<Item = (&'a str, &'a Value)>,
    intern: &InternTable,
    wal: Option<&arcgraph_storage::wal::WalHandle>,
    tenant: TenantId,
) -> Result<Option<TypedBagPayload>, PropPayloadError> {
    let mut builder = PropBlockBuilder::new();
    let mut any = false;
    for (name, value) in props {
        any = true;
        let key_id =
            match wal {
                Some(w) => arcgraph_storage::intern::intern_logged(intern, w, tenant, name)
                    .map_err(|e| PropPayloadError::Encode {
                        reason: format!("interning property key `{name}` failed: {e}"),
                    })?,
                None => intern
                    .intern(tenant, name)
                    .map_err(|e| PropPayloadError::Encode {
                        reason: format!("interning property key `{name}` failed: {e}"),
                    })?,
            };
        builder.put(key_id.raw(), value_to_prop_value(value));
    }
    if !any {
        return Ok(None);
    }
    finish_typed_bag(builder)
}

/// Shared tail of [`build_typed_bag`] / [`reencode_json_bag_to_typed`].
fn finish_typed_bag(
    builder: PropBlockBuilder,
) -> Result<Option<TypedBagPayload>, PropPayloadError> {
    let enc: EncodedPropBlock = builder.build().map_err(|e| PropPayloadError::Encode {
        reason: e.to_string(),
    })?;
    let overflow = enc.overflow_payload().map(<[u8]>::to_vec);
    let block = enc
        .into_block_bytes_deferred()
        .map_err(|e| PropPayloadError::Encode {
            reason: e.to_string(),
        })?;
    Ok(Some(TypedBagPayload { block, overflow }))
}

/// v2 M2 write path (design §M2's "Kill the JSON tax on BOTH paths" —
/// the WRITE half): serialize a slice-form property bag as a TYPED
/// block payload routed into [`PropertyData::TypedBlock`]. The typed
/// twin of [`properties_to_property_data`] (which stays for fixtures
/// + the mixed-store read tests; production write sites use THIS).
///
/// Empty bags short-circuit to [`PropertyData::Empty`] (the storage
/// fast-path, unchanged). Duplicate keys resolve last-wins (the
/// builder's map semantics — the M1 JSON `Map::insert` behavior).
pub fn properties_to_property_data_typed(
    props: &[(String, Value)],
    intern: &InternTable,
    wal: Option<&arcgraph_storage::wal::WalHandle>,
    tenant: TenantId,
) -> Result<PropertyData, PropPayloadError> {
    match build_typed_bag(
        props.iter().map(|(k, v)| (k.as_str(), v)),
        intern,
        wal,
        tenant,
    )? {
        None => Ok(PropertyData::Empty),
        Some(parts) => Ok(PropertyData::TypedBlock(parts)),
    }
}

/// [`BTreeMap`]-form twin of [`properties_to_property_data_typed`]
/// (the post-merge / post-remove SET / REMOVE bag shape).
pub fn property_map_to_property_data_typed(
    map: &BTreeMap<String, Value>,
    intern: &InternTable,
    wal: Option<&arcgraph_storage::wal::WalHandle>,
    tenant: TenantId,
) -> Result<PropertyData, PropPayloadError> {
    match build_typed_bag(
        map.iter().map(|(k, v)| (k.as_str(), v)),
        intern,
        wal,
        tenant,
    )? {
        None => Ok(PropertyData::Empty),
        Some(parts) => Ok(PropertyData::TypedBlock(parts)),
    }
}

/// v2 M2 migrate-on-open re-encoder (design §M2.6): one M1 legacy
/// JSON bag's bytes → the typed payload. The property NAMES in the
/// JSON are interned (WAL-logged) to key_ids; values are typed by
/// their JSON scalar shape via `json_value_to_prop_value`.
///
/// LOUD on malformed source JSON (a candidate bag that does not parse
/// is a corrupt source — migrating past it would silently drop the
/// bag, the exact M1 sweep posture for corrupt chains).
pub fn reencode_json_bag_to_typed(
    bag: &[u8],
    intern: &InternTable,
    wal: Option<&arcgraph_storage::wal::WalHandle>,
    tenant: TenantId,
) -> Result<Option<TypedBagPayload>, PropPayloadError> {
    let json: JsonValue = serde_json::from_slice(bag).map_err(|e| PropPayloadError::Corrupt {
        kind: "bag",
        rec_id: 0,
        reason: format!("M2 migration source bag is not valid JSON: {e}"),
    })?;
    let JsonValue::Object(obj) = json else {
        return Err(PropPayloadError::Corrupt {
            kind: "bag",
            rec_id: 0,
            reason: "M2 migration source bag's top level is not a JSON object".to_string(),
        });
    };
    let mut builder = PropBlockBuilder::new();
    let mut any = false;
    for (name, jv) in &obj {
        any = true;
        let key_id =
            match wal {
                Some(w) => arcgraph_storage::intern::intern_logged(intern, w, tenant, name)
                    .map_err(|e| PropPayloadError::Encode {
                        reason: format!("interning migrated property key `{name}` failed: {e}"),
                    })?,
                None => intern
                    .intern(tenant, name)
                    .map_err(|e| PropPayloadError::Encode {
                        reason: format!("interning migrated property key `{name}` failed: {e}"),
                    })?,
            };
        builder.put(key_id.raw(), json_value_to_prop_value(jv));
    }
    if !any {
        return Ok(None);
    }
    finish_typed_bag(builder)
}

// ─────────────────────────────────────────────────────────────────────
// v2 M2 — typed decode (the zero-decode read path, design §M2.2)
// ─────────────────────────────────────────────────────────────────────

/// A property projection carrying both the originally requested names
/// and their interned key ids — resolved ONCE per operator scan call
/// (never per row, design §M2.3).
#[derive(Debug, Clone, Default)]
pub struct ResolvedProjection {
    /// Names exactly as requested by the query. Legacy JSON bags carry
    /// inline string keys, including keys that were never interned, so
    /// their projection is resolved against this collection.
    requested_names: smallvec::SmallVec<[String; 8]>,
    /// `(property name, key_id)` pairs for requested names that resolve
    /// in the tenant's intern table. Typed blocks store key ids rather
    /// than inline names, so only this collection serves their reads.
    pub entries: smallvec::SmallVec<[(String, u32); 8]>,
}

impl ResolvedProjection {
    /// Resolve requested property names against the tenant's intern
    /// table. O(|names|) probes; call once per scan, not per row.
    ///
    /// # Fail-closed
    ///
    /// Uses the fallible [`InternTable::try_probe`]. The pre-fix code called
    /// the infallible `probe`, which laundered an owner-store I/O error into
    /// `None` — indistinguishable from "this name was never interned", which
    /// this function legitimately treats as "omit the entry". A transient
    /// forward-index miss therefore dropped the property from EVERY result row
    /// of the scan: a silently wrong query answer, with no error surfaced.
    ///
    /// A genuine miss still omits the name from [`Self::entries`], because a
    /// typed block can only store an interned key id. The original name remains
    /// in `requested_names` so legacy JSON bags, whose keys are inline strings,
    /// can still honor the projection. Only a real lookup FAILURE propagates.
    pub fn resolve(
        names: &[String],
        intern: &InternTable,
        tenant: TenantId,
    ) -> Result<Self, arcgraph_storage::owner_row::OwnerRowError> {
        let requested_names = names.iter().cloned().collect();
        let mut entries = smallvec::SmallVec::new();
        for name in names {
            if let Some(id) = intern.try_probe(tenant, name)? {
                entries.push((name.clone(), id.raw()));
            }
        }
        Ok(Self {
            requested_names,
            entries,
        })
    }
}

/// Decode one property payload (typed OR legacy JSON) into the full
/// runtime bag. The mixed-store dispatch (design §M2.6): first byte
/// [`PROP_BLOCK_DISCRIMINANT`] = typed block; `b'{'` = M1 legacy JSON
/// (readable during the migrate-on-open window and for pre-M2 WAL
/// replays); anything else = LOUD corruption.
fn decode_prop_payload(
    bytes: &[u8],
    blobs: &BlobStore,
    intern: &InternTable,
    tenant: TenantId,
    kind: &'static str,
    rec_id: u64,
) -> Result<BTreeMap<String, Value>, PropPayloadError> {
    match bytes.first() {
        Some(&PROP_BLOCK_DISCRIMINANT) => {
            decode_typed_payload(bytes, blobs, intern, tenant, kind, rec_id, None)
        }
        Some(&LEGACY_JSON_DISCRIMINANT) => Ok(decode_legacy_json_bag(bytes, kind, rec_id)),
        Some(&other) => Err(PropPayloadError::Corrupt {
            kind,
            rec_id,
            reason: format!(
                "property payload first byte {other:#04x} is neither a typed block \
                 ({PROP_BLOCK_DISCRIMINANT:#04x}) nor a legacy JSON bag ({LEGACY_JSON_DISCRIMINANT:#04x})"
            ),
        }),
        None => Err(PropPayloadError::Corrupt {
            kind,
            rec_id,
            reason: "empty property payload".to_string(),
        }),
    }
}

/// Decode a typed block, materializing EITHER the full bag
/// (`projection = None`) or only the projected key_ids (design §M2.2
/// `materialize(key_id_set)` — bytes are touched only for requested
/// keys; the overflow payload is fetched lazily, only when a
/// requested/materialized key actually resolves into it).
#[allow(clippy::too_many_arguments)] // decode context: 5 identities + 2 modes — a param struct would just rename them
fn decode_typed_payload(
    bytes: &[u8],
    blobs: &BlobStore,
    intern: &InternTable,
    tenant: TenantId,
    kind: &'static str,
    rec_id: u64,
    projection: Option<&ResolvedProjection>,
) -> Result<BTreeMap<String, Value>, PropPayloadError> {
    let corrupt = |reason: String| PropPayloadError::Corrupt {
        kind,
        rec_id,
        reason,
    };
    let view = PropBlockView::parse(bytes).map_err(|e| corrupt(e.to_string()))?;

    // Lazy overflow fetch state: resolved at most once, only when a
    // materialized key needs it (the §M2.3 laziness contract). The
    // fetch is the M2 zero-copy read (`get_bag` — an Arc-range view
    // for slotted payloads, no byte copy).
    let mut overflow_bytes: Option<arcgraph_storage::blob::BagBytes> = None;
    let fetch_overflow =
        |ob: &mut Option<arcgraph_storage::blob::BagBytes>| -> Result<(), PropPayloadError> {
            if ob.is_some() {
                return Ok(());
            }
            let bref = view
                .overflow_ref()
                .map_err(|e| corrupt(e.to_string()))?
                .ok_or_else(|| corrupt("value locator on a block with no overflow tail".into()))?;
            let fetched = blobs
                .get_bag(tenant, bref)
                .map_err(|e| PropPayloadError::Fetch {
                    kind,
                    rec_id,
                    reason: format!("overflow payload fetch failed: {e}"),
                })?;
            *ob = Some(fetched);
            Ok(())
        };

    let mut out: BTreeMap<String, Value> = BTreeMap::new();

    // Resolve one key_id we already know exists in the primary header.
    let put_primary = |key_id: u32,
                       name: Option<&str>,
                       out: &mut BTreeMap<String, Value>,
                       ob: &mut Option<arcgraph_storage::blob::BagBytes>|
     -> Result<(), PropPayloadError> {
        let looked = view.get(key_id).map_err(|e| corrupt(e.to_string()))?;
        let resolved_name: String = match name {
            Some(n) => n.to_owned(),
            None => resolve_key_name(intern, tenant, key_id, kind, rec_id)?,
        };
        match looked {
            PrimaryLookup::Found(v) => {
                if let Some(val) = prop_value_ref_to_value(v, &resolved_name, kind, rec_id)? {
                    out.insert(resolved_name, val);
                }
            }
            PrimaryLookup::InOverflow { tag, off, len } => {
                fetch_overflow(ob)?;
                let obytes = ob.as_ref().expect("fetched above");
                let oview = OverflowView::parse(obytes, view.max_primary_key_id())
                    .map_err(|e| corrupt(e.to_string()))?;
                let v = oview
                    .resolve_locator(tag, off, len)
                    .map_err(|e| corrupt(e.to_string()))?;
                if let Some(val) = prop_value_ref_to_value(v, &resolved_name, kind, rec_id)? {
                    out.insert(resolved_name, val);
                }
            }
            PrimaryLookup::Absent | PrimaryLookup::MaybeInOverflow => {}
        }
        Ok(())
    };

    match projection {
        None => {
            // Full-bag materialization: every primary key + (iff the
            // block spilled) every xentry.
            let key_ids: Vec<u32> = view.key_ids().collect();
            for key_id in key_ids {
                put_primary(key_id, None, &mut out, &mut overflow_bytes)?;
            }
            if view.has_overflow() {
                fetch_overflow(&mut overflow_bytes)?;
                let obytes = overflow_bytes.as_ref().expect("fetched above");
                let oview = OverflowView::parse(obytes, view.max_primary_key_id())
                    .map_err(|e| corrupt(e.to_string()))?;
                for key_id in oview.key_ids().collect::<Vec<_>>() {
                    let name = resolve_key_name(intern, tenant, key_id, kind, rec_id)?;
                    if let Some(v) = oview.get(key_id).map_err(|e| corrupt(e.to_string()))? {
                        if let Some(val) = prop_value_ref_to_value(v, &name, kind, rec_id)? {
                            out.insert(name, val);
                        }
                    }
                }
            }
        }
        Some(proj) => {
            // Projected materialization: touch ONLY the requested
            // keys' bytes (the M2 EXIT gate-1 zero-decode contract).
            for (name, key_id) in &proj.entries {
                match view.get(*key_id).map_err(|e| corrupt(e.to_string()))? {
                    PrimaryLookup::Found(_) | PrimaryLookup::InOverflow { .. } => {
                        put_primary(*key_id, Some(name.as_str()), &mut out, &mut overflow_bytes)?;
                    }
                    PrimaryLookup::Absent => {}
                    PrimaryLookup::MaybeInOverflow => {
                        // Wide bag: the key may live in the xentries.
                        fetch_overflow(&mut overflow_bytes)?;
                        let obytes = overflow_bytes.as_ref().expect("fetched above");
                        let oview = OverflowView::parse(obytes, view.max_primary_key_id())
                            .map_err(|e| corrupt(e.to_string()))?;
                        if let Some(v) = oview.get(*key_id).map_err(|e| corrupt(e.to_string()))? {
                            if let Some(val) = prop_value_ref_to_value(v, name, kind, rec_id)? {
                                out.insert(name.to_string(), val);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Reverse-resolve one interned property key id. An id the intern
/// table does not know is corruption: the `InternString` WAL record
/// for every key referenced by a committed block landed BEFORE that
/// block's commit (the `intern_logged` ordering), so recovery always
/// rebuilds it first.
fn resolve_key_name(
    intern: &InternTable,
    tenant: TenantId,
    key_id: u32,
    kind: &'static str,
    rec_id: u64,
) -> Result<String, PropPayloadError> {
    intern
        .resolve(tenant, arcgraph_core::StringId::new(key_id))
        .map(|arc| arc.as_ref().clone())
        .ok_or_else(|| PropPayloadError::Corrupt {
            kind,
            rec_id,
            reason: format!("typed block references unknown interned key_id {key_id}"),
        })
}

/// The M1 legacy JSON decode, byte-for-byte the pre-M2 semantics
/// (per-entry bridge failures warn-drop). Serves the migrate-on-open
/// window's mixed-store reads; post-migration stores contain no such
/// payloads.
fn decode_legacy_json_bag(
    bytes: &[u8],
    kind: &'static str,
    rec_id: u64,
) -> BTreeMap<String, Value> {
    let json: JsonValue = match serde_json::from_slice(bytes) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                kind,
                rec_id,
                error = %e,
                "legacy JSON property bag failed to parse; surfacing empty bag \
                 (pre-M2 payload — M1 warn-degrade semantics preserved for it)",
            );
            return BTreeMap::new();
        }
    };
    let JsonValue::Object(obj) = json else {
        tracing::warn!(
            kind,
            rec_id,
            "legacy JSON property bag top-level is not an object; surfacing empty bag",
        );
        return BTreeMap::new();
    };
    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    for (k, v) in obj {
        // #1444 (#1383): the MAP-ONLY property bridge — stored-bag
        // objects are customer values, never entity-detected.
        match Value::try_from_json_property_value(&v) {
            Ok(val) => {
                out.insert(k, val);
            }
            Err(e) => {
                tracing::warn!(
                    kind,
                    rec_id,
                    key = %k,
                    error = %e,
                    "legacy JSON property bag: per-entry decode failed; dropping entry",
                );
            }
        }
    }
    out
}

/// v2 M2 checked node-bag read (design §M2.2): typed payloads decode
/// zero-copy through [`PropBlockView`]; typed-layer corruption is a
/// LOUD `Err`, never a silent empty bag. A missing/unfetchable MAIN
/// payload keeps the ratified ADR-149/ADR-152 cross-snapshot degrade
/// (`Ok` + empty bag + warn) — that class is an MVCC race, not
/// corruption.
pub fn record_property_bag_checked(
    record: &NodeRecord,
    blobs: &BlobStore,
    intern: &InternTable,
    tenant: TenantId,
) -> Result<BTreeMap<String, Value>, PropPayloadError> {
    bag_checked_impl(
        record.property_ref,
        blobs,
        intern,
        tenant,
        "node",
        record.id,
        None,
    )
}

/// v2 M2 checked rel-bag read. Mirror of
/// [`record_property_bag_checked`].
pub fn rel_record_property_bag_checked(
    record: &RelRecord,
    blobs: &BlobStore,
    intern: &InternTable,
    tenant: TenantId,
) -> Result<BTreeMap<String, Value>, PropPayloadError> {
    bag_checked_impl(
        record.property_ref,
        blobs,
        intern,
        tenant,
        "rel",
        record.id,
        None,
    )
}

/// v2 M2 PROJECTED node-bag read (design §M2.3): materializes only
/// the projection's key_ids — the "point read touching K of M
/// properties decodes only K" contract (M2 EXIT gate 1).
pub fn record_property_bag_projected(
    record: &NodeRecord,
    blobs: &BlobStore,
    intern: &InternTable,
    tenant: TenantId,
    projection: &ResolvedProjection,
) -> Result<BTreeMap<String, Value>, PropPayloadError> {
    bag_checked_impl(
        record.property_ref,
        blobs,
        intern,
        tenant,
        "node",
        record.id,
        Some(projection),
    )
}

/// v2 M2 PROJECTED rel-bag read. Mirror of
/// [`record_property_bag_projected`].
pub fn rel_record_property_bag_projected(
    record: &RelRecord,
    blobs: &BlobStore,
    intern: &InternTable,
    tenant: TenantId,
    projection: &ResolvedProjection,
) -> Result<BTreeMap<String, Value>, PropPayloadError> {
    bag_checked_impl(
        record.property_ref,
        blobs,
        intern,
        tenant,
        "rel",
        record.id,
        Some(projection),
    )
}

/// Shared implementation of the checked/projected bag reads.
#[allow(clippy::too_many_arguments)] // mirror of decode_typed_payload's context set
fn bag_checked_impl(
    property_ref: u64,
    blobs: &BlobStore,
    intern: &InternTable,
    tenant: TenantId,
    kind: &'static str,
    rec_id: u64,
    projection: Option<&ResolvedProjection>,
) -> Result<BTreeMap<String, Value>, PropPayloadError> {
    let Some(blob_ref) = BlobRef::decode(property_ref) else {
        return Ok(BTreeMap::new());
    };
    // v2 M2 zero-copy read (design §4.2): a slotted payload is an
    // Arc-range view over the resident page image — no byte copy.
    let bytes = match blobs.get_bag(tenant, blob_ref) {
        Ok(b) => b,
        Err(e) => {
            // Ratified cross-snapshot degrade (ADR-149 §Risks /
            // ADR-152 §Operational): the MAIN payload's fetch fault is
            // an MVCC race class, not corruption — warn + empty, the
            // exact pre-M2 posture.
            tracing::warn!(
                kind,
                rec_id,
                error = %e,
                "property payload fetch failed; surfacing empty bag (cross-snapshot degrade \
                 per ADR-149 §Risks)",
            );
            return Ok(BTreeMap::new());
        }
    };
    match projection {
        None => decode_prop_payload(&bytes, blobs, intern, tenant, kind, rec_id),
        Some(proj) => match bytes.first() {
            Some(&PROP_BLOCK_DISCRIMINANT) => {
                decode_typed_payload(&bytes, blobs, intern, tenant, kind, rec_id, Some(proj))
            }
            // Legacy JSON during the migration window: a projected
            // read still decodes the full bag (the M1 tax — dies with
            // the payload at migration completion), then filters.
            Some(&LEGACY_JSON_DISCRIMINANT) => {
                let mut full = decode_legacy_json_bag(&bytes, kind, rec_id);
                full.retain(|k, _| proj.requested_names.iter().any(|name| name == k));
                Ok(full)
            }
            Some(&other) => Err(PropPayloadError::Corrupt {
                kind,
                rec_id,
                reason: format!(
                    "property payload first byte {other:#04x} is neither a typed block nor a \
                     legacy JSON bag"
                ),
            }),
            None => Err(PropPayloadError::Corrupt {
                kind,
                rec_id,
                reason: "empty property payload".to_string(),
            }),
        },
    }
}

/// Materialize a stored [`NodeRecord`] into the wire-shaped [`NodeView`]:
/// resolve its interned [`LabelId`], decode its persisted property bag
/// (typed OR legacy JSON — the v2 M2 mixed-store dispatch), and
/// reverse-resolve its label name using the caller's catalog lookup.
///
/// # v2 M2 signature deviation (design §M2.2 — flagged for review)
///
/// The design says this helper "keeps its signature (it still returns
/// a `NodeView`)". It still returns a `NodeView` — but literal
/// signature preservation is impossible at M2: decoding a TYPED
/// payload requires the tenant's [`InternTable`] (key_id → name),
/// which the old parameter set could not reach ([`CrudStore`] does not
/// hold the intern table). Deviation, kept minimal: `+intern`
/// parameter, `+Result` (typed-payload corruption is a LOUD reject per
/// the same design section — never a silent empty bag). The rejected
/// alternative — threading an `Arc<InternTable>` into `CrudStore` as
/// set-once state purely to preserve four parameters — was a larger
/// storage-surface change for a shim's sake.
pub fn hydrate_node_view(
    tenant: TenantId,
    crud: &CrudStore,
    intern: &InternTable,
    record: &NodeRecord,
    resolve_label_name: impl FnOnce(u32) -> Option<String>,
) -> Result<NodeView, PropPayloadError> {
    let label = if record.label_id == 0 {
        None
    } else {
        Some(LabelId::new(record.label_id))
    };
    let properties = record_property_bag_checked(record, crud.blob_store(), intern, tenant)?;
    let label_name = resolve_label_name(record.label_id);
    Ok(NodeView {
        id: NodeId::new(record.id),
        label,
        label_name,
        properties,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slice_routes_to_property_data_empty() {
        let pd = properties_to_property_data(&[]);
        assert!(matches!(pd, PropertyData::Empty));
    }

    #[test]
    fn empty_btreemap_routes_to_property_data_empty() {
        let map: BTreeMap<String, Value> = BTreeMap::new();
        let pd = property_map_to_property_data(&map);
        assert!(matches!(pd, PropertyData::Empty));
    }

    #[test]
    fn populated_slice_routes_to_property_data_blob_with_json_payload() {
        let props = vec![
            ("id".to_string(), Value::Integer(42)),
            ("name".to_string(), Value::String("Alice".into())),
        ];
        let pd = properties_to_property_data(&props);
        match pd {
            PropertyData::Blob(bytes) => {
                let json: JsonValue = serde_json::from_slice(&bytes).expect("valid JSON");
                let obj = json.as_object().expect("object");
                assert_eq!(obj.len(), 2);
                assert_eq!(obj.get("id").and_then(JsonValue::as_i64), Some(42));
                assert_eq!(obj.get("name").and_then(JsonValue::as_str), Some("Alice"));
            }
            other => panic!("expected PropertyData::Blob, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_through_blob_store() {
        // Encode → BlobStore::put → BlobStore::get → decode JSON → BTreeMap.
        let tenant = TenantId::DEFAULT;
        let props = vec![
            ("id".to_string(), Value::Integer(42)),
            ("name".to_string(), Value::String("Alice".into())),
            ("active".to_string(), Value::Boolean(true)),
            ("score".to_string(), Value::Float(2.5)),
        ];
        let pd = properties_to_property_data(&props);
        let bytes = match pd {
            PropertyData::Blob(b) => b,
            other => panic!("expected Blob, got {other:?}"),
        };
        // Round-trip the JSON bytes (we don't need to actually go
        // through BlobStore for the round-trip semantic; the JSON
        // bytes ARE what BlobStore stores opaquely).
        let json: JsonValue = serde_json::from_slice(&bytes).expect("valid JSON");
        let obj = json.as_object().expect("object").clone();
        let mut decoded: BTreeMap<String, Value> = BTreeMap::new();
        for (k, v) in obj {
            decoded.insert(k, Value::try_from_json_value(&v).expect("decode"));
        }
        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded.get("id"), Some(&Value::Integer(42)));
        assert_eq!(decoded.get("name"), Some(&Value::String("Alice".into())));
        assert_eq!(decoded.get("active"), Some(&Value::Boolean(true)));
        match decoded.get("score") {
            Some(Value::Float(f)) => assert!((f - 2.5).abs() < 1e-9),
            other => panic!("expected Float, got {other:?}"),
        }
        let _ = tenant;
    }

    /// #1383 — `graph.ingest` accepts nested JSON objects verbatim. Objects
    /// whose keys happen to match an internal entity shape are still customer
    /// property values and must round-trip as maps, even when their fields are
    /// strings rather than internal numeric IDs.
    #[test]
    fn ingest_entity_shaped_json_property_values_round_trip_as_maps() {
        let tenant = TenantId::DEFAULT;
        let mut ingest_properties = BTreeMap::new();
        ingest_properties.insert(
            "edge_descriptor".to_string(),
            serde_json::json!({"id": "x1", "from": "a", "to": "b"}),
        );
        ingest_properties.insert(
            "node_descriptor".to_string(),
            serde_json::json!({"id": "n1", "label": "Person"}),
        );

        let bytes = match crate::storage::adapters::property_data_for_json_map(&ingest_properties) {
            PropertyData::Blob(bytes) => bytes,
            other => panic!("expected ingest property blob, got {other:?}"),
        };
        let blobs = BlobStore::new();
        let blob_ref = blobs
            .put(tenant, &bytes)
            .expect("test property blob should be accepted");
        let mut record = NodeRecord::new(
            arcgraph_core::NodeId::new(1),
            arcgraph_core::LabelId::new(0),
            arcgraph_core::Lsn::new(1),
        );
        arcgraph_storage::property::encode_overflow_node(blob_ref, &mut record);

        // v2 M2: the checked read (the legacy-JSON leg of the
        // mixed-store dispatch — the payload begins `{`).
        let intern = InternTable::new();
        let decoded = record_property_bag_checked(&record, &blobs, &intern, tenant)
            .expect("legacy JSON bag decodes");
        let expect_edge = Value::Map(BTreeMap::from([
            ("from".to_string(), Value::String("a".into())),
            ("id".to_string(), Value::String("x1".into())),
            ("to".to_string(), Value::String("b".into())),
        ]));
        let expect_node = Value::Map(BTreeMap::from([
            ("id".to_string(), Value::String("n1".into())),
            ("label".to_string(), Value::String("Person".into())),
        ]));
        assert_eq!(decoded.get("edge_descriptor"), Some(&expect_edge));
        assert_eq!(decoded.get("node_descriptor"), Some(&expect_node));

        // AND the #1444 ⋈ M2 reconciliation: the SAME entity-shaped
        // values written through the TYPED path (MapOpaque payloads)
        // decode as `Value::Map` through the typed reader too — the
        // map-only stored-bag contract holds on BOTH representations.
        let typed_props = [
            ("edge_descriptor".to_string(), expect_edge.clone()),
            ("node_descriptor".to_string(), expect_node.clone()),
        ];
        let parts = build_typed_bag(
            typed_props.iter().map(|(k, v)| (k.as_str(), v)),
            &intern,
            None,
            tenant,
        )
        .expect("typed encode")
        .expect("non-empty");
        assert!(parts.overflow.is_none(), "small maps stay inline");
        let (tref, _) = blobs.stage_bag(tenant, 7, &parts.block).expect("stage");
        blobs.publish_txn_slotted(7).unwrap();
        let mut typed_rec = NodeRecord::new(
            arcgraph_core::NodeId::new(2),
            arcgraph_core::LabelId::new(0),
            arcgraph_core::Lsn::new(1),
        );
        arcgraph_storage::property::encode_overflow_node(tref, &mut typed_rec);
        let typed_decoded = record_property_bag_checked(&typed_rec, &blobs, &intern, tenant)
            .expect("typed bag decodes");
        assert_eq!(typed_decoded.get("edge_descriptor"), Some(&expect_edge));
        assert_eq!(typed_decoded.get("node_descriptor"), Some(&expect_node));
    }

    /// #1638 — `graph.ingest` writes a legacy JSON bag without interning its
    /// property names. A projected read must therefore match the bag's inline
    /// names, while still returning no property the query did not request.
    #[test]
    fn legacy_json_projection_uses_requested_names_and_remains_narrow() {
        let tenant = TenantId::DEFAULT;
        let intern = InternTable::new();
        let blobs = BlobStore::new();
        let ingest_properties = BTreeMap::from([
            ("age".to_string(), serde_json::json!(37)),
            ("name".to_string(), serde_json::json!("Ada")),
            ("secret".to_string(), serde_json::json!("not projected")),
        ]);
        let bytes = match crate::storage::adapters::property_data_for_json_map(&ingest_properties) {
            PropertyData::Blob(bytes) => bytes,
            other => panic!("expected ingest property blob, got {other:?}"),
        };
        let blob_ref = blobs
            .put(tenant, &bytes)
            .expect("test property blob should be accepted");
        let mut record = NodeRecord::new(
            arcgraph_core::NodeId::new(1),
            arcgraph_core::LabelId::new(1),
            arcgraph_core::Lsn::new(1),
        );
        arcgraph_storage::property::encode_overflow_node(blob_ref, &mut record);

        let projection =
            ResolvedProjection::resolve(&["name".to_string()], &intern, tenant).unwrap();
        assert!(
            projection.entries.is_empty(),
            "the fixture must exercise an uninterned requested name"
        );
        let projected =
            record_property_bag_projected(&record, &blobs, &intern, tenant, &projection)
                .expect("legacy JSON projected read");

        assert_eq!(
            projected,
            BTreeMap::from([("name".to_string(), Value::String("Ada".to_string()))]),
            "the requested uninterned value must survive and age/secret must not leak"
        );
    }

    #[test]
    fn duplicate_key_in_slice_resolves_last_wins() {
        let props = vec![
            ("id".to_string(), Value::Integer(1)),
            ("id".to_string(), Value::Integer(2)),
        ];
        let pd = properties_to_property_data(&props);
        let bytes = match pd {
            PropertyData::Blob(b) => b,
            other => panic!("expected Blob, got {other:?}"),
        };
        let json: JsonValue = serde_json::from_slice(&bytes).expect("valid JSON");
        let obj = json.as_object().expect("object");
        assert_eq!(obj.get("id").and_then(JsonValue::as_i64), Some(2));
    }

    #[test]
    fn record_with_zero_property_ref_returns_empty_bag() {
        // A fresh record (default property_ref = 0) returns empty.
        let rec = NodeRecord::new(
            arcgraph_core::NodeId::new(1),
            arcgraph_core::LabelId::new(0),
            arcgraph_core::Lsn::new(1),
        );
        let blobs = BlobStore::new();
        let intern = InternTable::new();
        let bag = record_property_bag_checked(&rec, &blobs, &intern, TenantId::DEFAULT)
            .expect("zero property_ref is a clean empty bag");
        assert!(bag.is_empty());
    }

    #[test]
    fn record_with_inline_u32_returns_empty_bag() {
        // An inline U32 pair payload does NOT carry key names; helper
        // returns an empty bag per the v1.0-α posture.
        let mut rec = NodeRecord::new(
            arcgraph_core::NodeId::new(1),
            arcgraph_core::LabelId::new(0),
            arcgraph_core::Lsn::new(1),
        );
        arcgraph_storage::property::encode_inline_node(
            arcgraph_storage::property::InlineShape::U32Pair(7, 42),
            &mut rec,
        );
        let blobs = BlobStore::new();
        let intern = InternTable::new();
        let bag = record_property_bag_checked(&rec, &blobs, &intern, TenantId::DEFAULT)
            .expect("inline payload is a clean empty bag");
        assert!(bag.is_empty());
    }

    // ─────────────────────────────────────────────────────────────────
    // v2 M2 — the differential-equality oracle CORE (EXIT gate 5's
    // encode/decode leg): for every bag, the M2 typed store must
    // materialize values IDENTICAL to the M1 JSON store. The full
    // RULE-MT leg (≥8 writers, bounded store + spill + refault, live +
    // post-recovery) lives in `tests/m2_differential_rule_mt.rs`; THIS
    // is the value-domain equivalence that leg builds on.
    // ─────────────────────────────────────────────────────────────────

    use arcgraph_storage::intern::InternTable;
    use arcgraph_storage::prop_block::patch_overflow_tail;
    use proptest::prelude::*;

    /// The M1 materialization: encode via the production JSON path,
    /// decode via the legacy JSON reader (byte-for-byte the pre-M2
    /// `decode_blob_bag` semantics).
    fn m1_materialize(props: &[(String, Value)]) -> BTreeMap<String, Value> {
        match properties_to_property_data(props) {
            PropertyData::Empty => BTreeMap::new(),
            PropertyData::Blob(bytes) => decode_legacy_json_bag(&bytes, "node", 0),
            other => panic!("unexpected PropertyData variant {other:?}"),
        }
    }

    /// The M2 materialization: encode via the typed bridge, stage the
    /// overflow (if any) in a real `BlobStore`, decode via the typed
    /// reader.
    fn m2_materialize(
        props: &[(String, Value)],
        intern: &InternTable,
        blobs: &BlobStore,
    ) -> BTreeMap<String, Value> {
        let tenant = TenantId::DEFAULT;
        let parts = build_typed_bag(
            props.iter().map(|(k, v)| (k.as_str(), v)),
            intern,
            None,
            tenant,
        )
        .expect("typed encode");
        let Some(mut parts) = parts else {
            return BTreeMap::new();
        };
        if let Some(of) = &parts.overflow {
            let (bref, _emits) = blobs.stage_bag(tenant, 1, of).expect("stage overflow");
            blobs.publish_txn_slotted(1).unwrap();
            patch_overflow_tail(&mut parts.block, bref).expect("patch tail");
        }
        decode_prop_payload(&parts.block, blobs, intern, tenant, "node", 0).expect("typed decode")
    }

    /// Value strategy spanning the write-path property domain,
    /// including the lossy edges the M1 bridge defines (non-finite
    /// floats), inline/overflow string boundaries, and nested lists.
    fn value_strategy() -> impl Strategy<Value = Value> {
        let scalar = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::Boolean),
            any::<i64>().prop_map(Value::Integer),
            prop_oneof![
                any::<f64>().prop_filter("finite", |f| f.is_finite()),
                Just(f64::NAN),
                Just(f64::INFINITY),
                Just(f64::NEG_INFINITY),
            ]
            .prop_map(Value::Float),
            proptest::string::string_regex("[a-zA-Z0-9\u{00e9}\u{4e16} ]{0,300}")
                .expect("regex")
                .prop_map(Value::String),
        ];
        scalar.prop_recursive(3, 24, 6, |inner| {
            proptest::collection::vec(inner, 0..6).prop_map(Value::List)
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(
            if cfg!(debug_assertions) { 128 } else { 512 }
        ))]

        /// EXIT gate 5's write-path equivalence: identical materialized
        /// values for every generated bag (0..=90 props spans the
        /// 64-entry primary/overflow boundary), across the full value
        /// domain including non-finite floats and nested lists.
        #[test]
        fn m2_write_path_materializes_identically_to_m1(
            bag in proptest::collection::btree_map(
                proptest::string::string_regex("[a-z_][a-z0-9_]{0,24}").expect("regex"),
                value_strategy(),
                0..90,
            )
        ) {
            let props: Vec<(String, Value)> =
                bag.into_iter().collect();
            let intern = InternTable::new();
            let blobs = BlobStore::new();
            let m1 = m1_materialize(&props);
            let m2 = m2_materialize(&props, &intern, &blobs);
            prop_assert_eq!(m1, m2);
        }

        /// EXIT gate 4's migration equivalence core: an arbitrary M1
        /// legacy JSON bag re-encoded through the migration bridge
        /// materializes identically to reading the JSON directly —
        /// including the JSON-side number-widening rules (u64 beyond
        /// i64 → f64) the migration must preserve.
        #[test]
        fn m2_migration_reencode_materializes_identically_to_m1(
            jbag in proptest::collection::btree_map(
                proptest::string::string_regex("[a-z_][a-z0-9_]{0,24}").expect("regex"),
                proptest::arbitrary::any::<u8>().prop_flat_map(|sel| {
                    match sel % 7 {
                        0 => Just(JsonValue::Null).boxed(),
                        1 => any::<bool>().prop_map(JsonValue::Bool).boxed(),
                        2 => any::<i64>().prop_map(|i| serde_json::json!(i)).boxed(),
                        3 => any::<u64>().prop_map(|u| serde_json::json!(u)).boxed(),
                        4 => any::<f64>()
                            .prop_filter("finite", |f| f.is_finite())
                            .prop_map(|f| serde_json::json!(f))
                            .boxed(),
                        5 => proptest::string::string_regex("[a-zA-Z0-9 ]{0,300}")
                            .expect("regex")
                            .prop_map(JsonValue::String)
                            .boxed(),
                        _ => proptest::collection::vec(
                                prop_oneof![
                                    any::<i64>().prop_map(|i| serde_json::json!(i)),
                                    proptest::string::string_regex("[a-z]{0,40}")
                                        .expect("regex")
                                        .prop_map(JsonValue::String),
                                    Just(JsonValue::Null),
                                ],
                                0..8,
                            )
                            .prop_map(JsonValue::Array)
                            .boxed(),
                    }
                }),
                0..80,
            )
        ) {
            let json_bytes = serde_json::to_vec(&JsonValue::Object(
                jbag.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            ))
            .expect("serialize");

            // M1 read of the legacy bag.
            let m1 = decode_legacy_json_bag(&json_bytes, "node", 0);

            // M2 migration re-encode + typed read.
            let intern = InternTable::new();
            let blobs = BlobStore::new();
            let tenant = TenantId::DEFAULT;
            let parts = reencode_json_bag_to_typed(&json_bytes, &intern, None, tenant)
                .expect("reencode");
            let m2 = match parts {
                None => BTreeMap::new(),
                Some(mut parts) => {
                    if let Some(of) = &parts.overflow {
                        let (bref, _emits) =
                            blobs.stage_bag(tenant, 1, of).expect("stage overflow");
                        blobs.publish_txn_slotted(1).unwrap();
                        patch_overflow_tail(&mut parts.block, bref).expect("patch tail");
                    }
                    decode_prop_payload(&parts.block, &blobs, &intern, tenant, "node", 0)
                        .expect("typed decode")
                }
            };
            prop_assert_eq!(m1, m2);
        }
    }

    #[test]
    fn typed_payload_unknown_discriminant_rejects_loud() {
        let intern = InternTable::new();
        let blobs = BlobStore::new();
        // Neither 0x01 (typed) nor '{' (legacy JSON).
        let err = decode_prop_payload(
            &[0x42, 0, 0, 0],
            &blobs,
            &intern,
            TenantId::DEFAULT,
            "node",
            7,
        )
        .expect_err("must reject");
        assert!(
            matches!(err, PropPayloadError::Corrupt { rec_id: 7, .. }),
            "got {err}"
        );
    }

    #[test]
    fn typed_payload_unknown_key_id_rejects_loud() {
        // A typed block whose key_id the intern table does not know =
        // corruption (the InternString WAL record always precedes the
        // block's commit).
        let intern = InternTable::new();
        let blobs = BlobStore::new();
        let tenant = TenantId::DEFAULT;
        let props = [("k".to_string(), Value::Integer(5))];
        let parts = build_typed_bag(
            props.iter().map(|(k, v)| (k.as_str(), v)),
            &intern,
            None,
            tenant,
        )
        .expect("encode")
        .expect("non-empty");
        // A FRESH intern table (simulating a store that lost the
        // InternString record) must reject, not silently mis-name.
        let fresh = InternTable::new();
        let err = decode_prop_payload(&parts.block, &blobs, &fresh, tenant, "node", 9)
            .expect_err("must reject");
        assert!(matches!(err, PropPayloadError::Corrupt { .. }), "got {err}");
    }

    #[test]
    fn projected_read_touches_only_requested_keys() {
        // The §M2.3 laziness contract, structurally: a projection of
        // inline keys must succeed even when the overflow payload is
        // UNFETCHABLE (poisoned ref) — proof the overflow is not
        // touched unless a requested key resolves into it.
        let intern = InternTable::new();
        let blobs = BlobStore::new();
        let tenant = TenantId::DEFAULT;
        let mut props: Vec<(String, Value)> = vec![
            ("small".to_string(), Value::Integer(1)),
            ("big".to_string(), Value::String("z".repeat(400))),
        ];
        props.sort_by(|a, b| a.0.cmp(&b.0));
        let parts = build_typed_bag(
            props.iter().map(|(k, v)| (k.as_str(), v)),
            &intern,
            None,
            tenant,
        )
        .expect("encode")
        .expect("non-empty");
        assert!(parts.overflow.is_some(), "big string must spill");
        // Deliberately DO NOT stage the overflow; patch a dangling ref.
        let mut block = parts.block;
        patch_overflow_tail(&mut block, BlobRef::new(999_999, 3)).expect("patch");

        // Projecting the INLINE key succeeds — overflow never fetched.
        let proj_small =
            ResolvedProjection::resolve(&["small".to_string()], &intern, tenant).unwrap();
        let got = decode_typed_payload(
            &block,
            &blobs,
            &intern,
            tenant,
            "node",
            0,
            Some(&proj_small),
        )
        .expect("inline projection must not touch the overflow");
        assert_eq!(got.get("small"), Some(&Value::Integer(1)));
        assert_eq!(got.len(), 1);

        // Projecting the SPILLED key surfaces the fetch fault — loud.
        let proj_big = ResolvedProjection::resolve(&["big".to_string()], &intern, tenant).unwrap();
        let err = decode_typed_payload(&block, &blobs, &intern, tenant, "node", 0, Some(&proj_big))
            .expect_err("dangling overflow must surface");
        assert!(matches!(err, PropPayloadError::Fetch { .. }), "got {err}");
    }

    #[test]
    fn fault_missing_old_blob_degrades_to_empty_bag_not_a_failure() {
        // The already-decoded `current_bag` degrades to an empty bag on a
        // missing blob (ADR-152 §Operational #1 — the cross-snapshot fetch
        // race). v2 M2 preserves this ratified degrade for main-payload
        // fetch faults; loud M2 rejects are for payload-decode violations.
        let mut rec = NodeRecord::new(
            arcgraph_core::NodeId::new(1),
            arcgraph_core::LabelId::new(0),
            arcgraph_core::Lsn::new(1),
        );
        // A non-zero overflow property_ref pointing at a slot absent from
        // a fresh BlobStore → `blobs.get` fails → degrade to empty.
        arcgraph_storage::property::encode_overflow_node(BlobRef::new(999_999, 1), &mut rec);
        assert!(
            BlobRef::decode(rec.property_ref).is_some(),
            "fixture must encode an overflow (blob) property_ref",
        );
        let blobs = BlobStore::new();
        let intern = InternTable::new();
        let degraded_old = record_property_bag_checked(&rec, &blobs, &intern, TenantId::DEFAULT)
            .expect("a MISSING main payload degrades (Ok + empty), never an Err");
        assert!(
            degraded_old.is_empty(),
            "missing blob must degrade to empty bag"
        );
    }
}
