//! Runtime value taxonomy for the M4-61 / M4-62 executor.
//!
//! [`Value`] mirrors the openCypher / ArcQL value taxonomy at v1.0
//! per ADR-038 §2 D-20. The cells flowing through [`crate::executor::Batch`]
//! columns are `Value`s; predicate evaluation (M4-62 3VL) returns
//! [`crate::executor::ThreeValued`] which is a separate type for
//! type-system-enforced boolean-vs-3VL distinction.
//!
//! # NULL discipline
//!
//! [`Value::Null`] is the openCypher 3VL "could be NULL" runtime
//! representation. NULL propagates through:
//! - **Comparisons** (`<`, `<=`, `=`, ...) — any NULL operand yields
//!   [`crate::executor::ThreeValued::Unknown`].
//! - **Arithmetic** (`+`, `-`, `*`, `/`, `%`) — any NULL operand
//!   yields `Value::Null`.
//! - **Boolean ops** — handled in [`crate::executor::three_vl`] per
//!   the ADR-038 §2 D-20 truth tables.
//!
//! NULL is distinct from `Value::Boolean(false)` for AND / OR / NOT
//! precedence — the Cypher 3VL truth-table forces this.
//!
//! # ID discipline
//!
//! [`Value::Node`] / [`Value::Relationship`] carry a stub-friendly view
//! holding the ID + label/rel-type + a property bag. v1.0-alpha tests
//! populate these via [`crate::executor::StubExecutorSubstrate`];
//! production wiring at M4-08+ will surface
//! `arcgraph_storage::router::TenantHandle`-rooted views per ADR-037
//! D-1.

use std::collections::BTreeMap;

use arcgraph_core::{
    Date, Decimal, Duration, LabelId, LocalDateTime, NodeId, RelId, TypeId, ZonedDateTime,
};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};

/// Runtime value flowing through executor batches.
///
/// The variant set is OPEN at v1.1+ per ADR-038 amendment-09 +
/// ADR-090 — temporal + decimal landed via the W23-V11-T-01 slice;
/// future point / spatial values land via subsequent amendments
/// alongside ADR-007.
///
/// # v1.1 temporal admittance (ADR-038 amendment-09)
///
/// The `Temporal(ZonedDateTime)` + `LocalDateTime(...)` + `Date(...)` +
/// `Duration(...)` + `Decimal(...)` variants opened the previously-
/// closed enum per ADR-038 amendment-09, which ratifies the K3 §2.3
/// "Gap shape" items 1-2 + K3 §4.1 items 1-4 taxonomy. (K3 §7 ¶4
/// separately requests the ADR-007 amendment that pins the wire
/// binding.) The bridges to/from `serde_json::Value` use ISO-8601
/// string encoding (lossless round-trip; the MCP boundary's JSON
/// wire shape is human-readable).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// 3VL "could be NULL" cell. Propagates per ADR-038 §2 D-20.
    Null,
    /// Boolean cell (Cypher 3VL: `true` and `false` ONLY; the
    /// "unknown" outcome is [`Value::Null`]).
    Boolean(bool),
    /// 64-bit signed integer cell.
    Integer(i64),
    /// 64-bit float cell. NaN / Inf are admissible runtime values
    /// (Cypher does not forbid them); equality with NaN is false per
    /// IEEE-754, which the executor preserves.
    Float(f64),
    /// UTF-8 string cell.
    String(String),
    /// A bound node (with an ID, optional label, property bag).
    Node(NodeView),
    /// A bound relationship (with an ID, optional rel-type, property
    /// bag, and from/to node IDs).
    Relationship(RelView),
    /// Homogeneous-or-heterogeneous list cell. Cypher 9 §3.5 admits
    /// heterogeneous lists at runtime; the executor preserves that.
    List(Vec<Value>),
    /// An openCypher map value (`{key: value, …}`): an ordered
    /// collection of string-keyed [`Value`]s. Keys are always UTF-8
    /// strings; iteration order is deterministic ([`BTreeMap`]) so
    /// `Display`-keyed GROUP BY / DISTINCT / UNION
    /// ([`crate::executor::ops::canonical_row_key`]) and byte-stable
    /// JSON / TOON projection are reproducible. Per ADR-191 (D-1). Maps
    /// participate in equality (order-independent key-set + 3VL pairwise
    /// values — D-3), comparability (`<`,`>` → `null` — D-4), and
    /// orderability (`compare_orderability` global total order — D-5),
    /// but are FENCED out of node/relationship property persistence
    /// (`literal_lift` rejects them — D-11; openCypher forbids map
    /// property values).
    Map(BTreeMap<String, Value>),
    /// A path value (ADR-193): the openCypher v9 §3 alternating
    /// node/relationship sequence `n₀, r₁, n₁, …, rₖ, nₖ`. The
    /// [`PathView`] representation STRUCTURALLY enforces the `#nodes =
    /// #rels + 1` invariant. A path is a READ/EXPRESSION value ONLY — it
    /// can NEVER be a stored property (D-12 write-op fence). Paths ARE
    /// ORDERABLE: openCypher orderability is a total order over all
    /// values, and a path sorts FIRST in the global type-order (D-11);
    /// the ordering is provided by an explicit compare arm
    /// ([`PathView::cmp_paths`]), NOT a derived `Ord`.
    Path(PathView),
    /// Zoned wall-clock instant (`TIMESTAMPTZ`). Per ADR-038
    /// amendment-09. JSON projection uses ISO-8601 string form.
    Temporal(ZonedDateTime),
    /// Local wall-clock with no zone. Per K3 §2.3 "Gap shape" item 1.
    LocalDateTime(LocalDateTime),
    /// Calendar date with no time / zone.
    Date(Date),
    /// ISO-8601 duration. Months are NOT canonicalized to nanos —
    /// see `arcgraph_core::Duration` doc.
    Duration(Duration),
    /// Fixed-point decimal `(scale, units)` per V11-T-02 companion
    /// landing.
    Decimal(Decimal),
}

/// Error surfaced by [`Value::try_from_json_value`] when a
/// [`serde_json::Value`] cannot map back into the [`Value`] taxonomy.
///
/// # Why a typed error (not `String` / `anyhow`)
///
/// The bridge surfaces enough variant information for caller-side
/// recovery: a Bolt response framer that receives an out-of-range
/// integer can choose between truncating, rejecting, or surfacing
/// `ArcQLError::TypeCheck` per the M5↔M4 contract surface. A
/// stringly-typed error would force pattern-matching on the `Display`
/// output, which is fragile.
///
/// `#[non_exhaustive]` permits adding a new variant (e.g., a future
/// `Node` reconstruction error
/// when nodes are recoverable through an arcgraph-storage handle) is
/// not a SemVer breaking change.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ValueJsonError {
    /// The JSON number cannot be losslessly represented in the
    /// [`Value`] integer / float lattice (e.g., a JSON number
    /// outside `f64` range).
    #[error("JSON number out of range for Value::Integer / Value::Float: {literal}")]
    NumberOutOfRange { literal: String },

    /// A node/relationship-shaped JSON object carries a malformed
    /// structural field (e.g. an `id` that is not a `u64`, or a `label`
    /// exceeding `u32`). Per ADR-191 (D-7), a plain JSON object that
    /// matches NEITHER the node nor the relationship shape now decodes
    /// as [`Value::Map`] (it is no longer an error); this variant is
    /// reserved for the entity-shaped-but-malformed case. The `kind`
    /// slot names the field at fault.
    #[error("unsupported JSON variant for Value lattice: {kind}")]
    UnsupportedShape { kind: &'static str },

    /// A JSON value nests deeper than [`MAX_JSON_DECODE_DEPTH`]. Per
    /// ADR-191 (D-12 / Consequences §"recursive variant"): the decode
    /// path is network-reachable (MCP / Bolt), so a recursion-depth
    /// bound rejects adversarially-deep nested-map / nested-list input
    /// rather than overflowing the stack
    /// (`feedback_security_class_first_network_surface.md`).
    #[error("JSON value nests too deep for the Value lattice: depth {depth} exceeds max {max}")]
    NestingTooDeep { depth: usize, max: usize },
}

/// Maximum nesting depth accepted by [`Value::try_from_json_value`].
///
/// The reverse JSON bridge is a network-reachable surface (the MCP
/// `raw_query` / Bolt decode paths feed it untrusted JSON). A recursive
/// `Value` (maps + lists nest arbitrarily) would stack-overflow on
/// adversarially-deep input absent a bound; we reject at the cap with
/// [`ValueJsonError::NestingTooDeep`] instead. The value (`64`) matches
/// the traversal `DEFAULT_MAX_DEPTH` convention and sits well under
/// `serde_json`'s own 128-deep parse limit (so a value that parses from
/// a string is still rejected here if it exceeds the lattice cap) and
/// the crate `#![recursion_limit = "256"]`. Per ADR-191 (D-12).
pub const MAX_JSON_DECODE_DEPTH: usize = 64;

impl Value {
    /// `true` iff this is the [`Value::Null`] variant.
    #[inline]
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Cast to a [`bool`] for predicate evaluation. Returns `Some(b)`
    /// for `Value::Boolean(b)`, `None` for any other variant
    /// (including [`Value::Null`] — the 3VL "unknown" case is handled
    /// in [`crate::executor::three_vl`]).
    #[inline]
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Cast to an [`i64`] for arithmetic / comparison. Returns
    /// `Some(n)` for `Value::Integer(n)`, `None` otherwise.
    #[inline]
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Integer(n) => Some(*n),
            _ => None,
        }
    }

    /// Cast to an [`f64`] for arithmetic / comparison. Returns
    /// `Some(n)` for `Value::Integer(n)` (widened) or
    /// `Value::Float(n)`; `None` otherwise.
    #[inline]
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Integer(n) => Some(*n as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Project this [`Value`] into a [`serde_json::Value`] per the
    /// W13β M4-81 MCP serialization bridge per ADR-038 amendment-02
    /// §M4.h ("TOON + JSON serialization for MCP") + amendment-03
    /// §M5↔M4 contract surface.
    ///
    /// # NaN / ±Inf handling
    ///
    /// Mirrors the W11ε TOON serializer's encoder convention
    /// (`crates/arcgraph-mcp/src/serializers/toon.rs` `encode_number`):
    /// non-finite floats coerce to JSON `null`. Cypher 9 admits
    /// non-finite runtime values, but JSON / TOON have no in-band
    /// representation for them; coercing to `null` matches the wire
    /// format's lossy contract and keeps the bridge total. The
    /// reverse path ([`Self::try_from_json_value`]) decodes JSON `null`
    /// as [`Value::Null`] — non-finite floats are NOT round-trippable;
    /// the proptest's value strategy excludes them per the same W12γ
    /// `materialize_proptest` discipline.
    ///
    /// # Node / Relationship / List emission
    ///
    /// - [`Value::Node`] → `{"id": <node_id_raw>, "label": <label_id_raw_or_null>, "labels": [<name>]?, "properties": {<key>: <Value-as-JSON>, ...}}`. The `labels` key (#871) carries the catalog-resolved label NAME (Neo4j-style list) and is present ONLY when [`NodeView::label_name`] is set; the numeric `label` is retained for the LabelId round-trip.
    /// - [`Value::Relationship`] → `{"id": <rel_id_raw>, "from": <from_node_id_raw>, "to": <to_node_id_raw>, "rel_type": <type_id_raw_or_null>, "type": <name>?, "properties": {<key>: <Value-as-JSON>, ...}}`. The `type` key (#871) carries the catalog-resolved rel-type NAME and is present ONLY when [`RelView::rel_type_name`] is set.
    /// - [`Value::List`] → `[<elem-as-JSON>, ...]`.
    ///
    /// The Node / Relationship JSON shape carries the structural
    /// surface the M5-07 `graph.search` / M5-11 `graph.raw_query`
    /// renderers consume; the round-trip via
    /// [`Self::try_from_json_value`] reconstructs `NodeView` / `RelView`
    /// with the same property bag. NodeId / LabelId / RelId / TypeId
    /// are `u64` newtypes per `arcgraph_core::ids` (each carries a
    /// `serde::Serialize` derive); the JSON form is the raw `u64`.
    ///
    /// # Errors
    ///
    /// Infallible. Every [`Value`] variant maps cleanly onto the JSON
    /// lattice (with NaN / Inf coerced to `null` as documented above).
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        match self {
            Value::Null => JsonValue::Null,
            Value::Boolean(b) => JsonValue::Bool(*b),
            Value::Integer(i) => JsonValue::Number(JsonNumber::from(*i)),
            Value::Float(f) => match JsonNumber::from_f64(*f) {
                Some(n) => JsonValue::Number(n),
                // NaN / ±Inf → JSON null per the TOON encoder convention.
                None => JsonValue::Null,
            },
            Value::String(s) => JsonValue::String(s.clone()),
            Value::Node(n) => {
                let mut obj = JsonMap::new();
                obj.insert(
                    "id".to_string(),
                    JsonValue::Number(JsonNumber::from(n.id.raw())),
                );
                obj.insert(
                    "label".to_string(),
                    n.label
                        .map(|l| JsonValue::Number(JsonNumber::from(u64::from(l.raw()))))
                        .unwrap_or(JsonValue::Null),
                );
                // #871 — surface the catalog-resolved label NAME as the
                // Neo4j-style `labels` list (a singleton at v1.0 single-
                // label, or empty when no name was resolved) so MCP
                // clients read `["Account"]`, never the opaque `LabelId`.
                // Emitted only when known; the numeric `label` above
                // preserves the LabelId round-trip (decoded back in
                // `json_object_to_value`).
                if let Some(name) = &n.label_name {
                    obj.insert(
                        "labels".to_string(),
                        JsonValue::Array(vec![JsonValue::String(name.clone())]),
                    );
                }
                obj.insert(
                    "properties".to_string(),
                    JsonValue::Object(properties_to_json(&n.properties)),
                );
                JsonValue::Object(obj)
            }
            Value::Relationship(r) => {
                let mut obj = JsonMap::new();
                obj.insert(
                    "id".to_string(),
                    JsonValue::Number(JsonNumber::from(r.id.raw())),
                );
                obj.insert(
                    "from".to_string(),
                    JsonValue::Number(JsonNumber::from(r.from.raw())),
                );
                obj.insert(
                    "to".to_string(),
                    JsonValue::Number(JsonNumber::from(r.to.raw())),
                );
                obj.insert(
                    "rel_type".to_string(),
                    r.rel_type
                        .map(|t| JsonValue::Number(JsonNumber::from(u64::from(t.raw()))))
                        .unwrap_or(JsonValue::Null),
                );
                // #871 — surface the catalog-resolved rel-type NAME as
                // the Neo4j-style `type` string so MCP clients read
                // `"KNOWS"`, never `"TypeId(1)"`. Emitted only when
                // known; the numeric `rel_type` above preserves the
                // TypeId round-trip.
                if let Some(name) = &r.rel_type_name {
                    obj.insert("type".to_string(), JsonValue::String(name.clone()));
                }
                obj.insert(
                    "properties".to_string(),
                    JsonValue::Object(properties_to_json(&r.properties)),
                );
                JsonValue::Object(obj)
            }
            Value::List(elems) => {
                JsonValue::Array(elems.iter().map(Value::to_json_value).collect())
            }
            // ADR-191 D-7 — a map projects as a JSON object (reusing the
            // property-bag projector; the `BTreeMap` deterministic key
            // order makes the wire form byte-stable). Nested maps recurse
            // via `properties_to_json` → `to_json_value`.
            Value::Map(m) => JsonValue::Object(properties_to_json(m)),
            // ADR-193 D-8 — a path projects as a structured JSON object
            // `{"start": <node>, "segments": [{"relationship": <rel>,
            // "end": <node>}, ...]}`. The `start`/`end` nodes reuse the
            // `Value::Node` JSON shape and `relationship` reuses the
            // `Value::Relationship` shape, so [`json_object_to_value`]'s
            // existing Node / Rel decoders reconstruct them on the
            // reverse path. The `start` + `segments` key pair is the
            // discriminator the decoder keys on (no other value shape
            // carries both).
            Value::Path(p) => {
                let mut obj = JsonMap::new();
                obj.insert(
                    "start".to_string(),
                    Value::Node(p.start.clone()).to_json_value(),
                );
                let segments: Vec<JsonValue> = p
                    .segments
                    .iter()
                    .map(|seg| {
                        let mut so = JsonMap::new();
                        so.insert(
                            "relationship".to_string(),
                            Value::Relationship(seg.rel.clone()).to_json_value(),
                        );
                        so.insert(
                            "end".to_string(),
                            Value::Node(seg.end.clone()).to_json_value(),
                        );
                        JsonValue::Object(so)
                    })
                    .collect();
                obj.insert("segments".to_string(), JsonValue::Array(segments));
                JsonValue::Object(obj)
            }
            // Temporal + decimal variants project as ISO-8601 strings
            // per ADR-090 §"Wire shape". The JSON object form
            // `{ "_type": "..", "value": ".." }` is reserved for v1.2
            // when MCP tools need to distinguish a Temporal property
            // from a String property; v1.1 uses the bare-string form
            // (matching openCypher 9 §3.4's serializer convention).
            Value::Temporal(t) => JsonValue::String(format!("{t}")),
            Value::LocalDateTime(ldt) => JsonValue::String(format!("{ldt}")),
            Value::Date(d) => JsonValue::String(format!("{d}")),
            Value::Duration(d) => JsonValue::String(format!("{d}")),
            Value::Decimal(d) => JsonValue::String(format!("{d}")),
        }
    }

    /// Decode a [`serde_json::Value`] back into a [`Value`] per the
    /// W13β M4-81 reverse bridge.
    ///
    /// The decoder is total over the JSON primitive lattice
    /// (null / bool / number / string / array) and structural over
    /// objects: an object with `id` / `from` / `to` / `rel_type` /
    /// `properties` keys reconstructs as [`Value::Relationship`]; an
    /// object with `id` / `label` / `properties` keys reconstructs as
    /// [`Value::Node`]; any OTHER object reconstructs as a
    /// [`Value::Map`] (ADR-191 D-7 — the entity shapes are matched first,
    /// so this is the bare-map fallthrough, NOT an error).
    ///
    /// # Number coercion
    ///
    /// JSON numbers route through `as_i64` first (preserving signed
    /// integer semantics), then `as_u64` for non-negative integers
    /// that overflow `i64`, then `as_f64` for fractional / large
    /// magnitudes. A JSON number that fits NONE of these surfaces
    /// [`ValueJsonError::NumberOutOfRange`].
    ///
    /// # Round-trip discipline
    ///
    /// `Value::to_json_value(&v).try_from_json_value()` yields `v`
    /// for every variant EXCEPT non-finite floats (which collapse to
    /// `Value::Null` on the encode side). The proptest's value
    /// strategy excludes non-finite floats per the W12γ
    /// `materialize_proptest` precedent.
    ///
    /// # Errors
    ///
    /// - [`ValueJsonError::NumberOutOfRange`] when the JSON number
    ///   cannot be losslessly cast to `i64` / `u64` / `f64`.
    /// - [`ValueJsonError::UnsupportedShape`] when an object IS
    ///   entity-shaped (Node / Relationship key set) but a structural
    ///   field is malformed (e.g. a non-`u64` `id`, a `label` exceeding
    ///   `u32`). A non-entity object is NOT an error — it decodes as a
    ///   [`Value::Map`] (ADR-191 D-7).
    /// - [`ValueJsonError::NestingTooDeep`] when the JSON value nests
    ///   deeper than [`MAX_JSON_DECODE_DEPTH`] (ADR-191 D-12,
    ///   network-reachable-value hardening).
    pub fn try_from_json_value(v: &JsonValue) -> Result<Self, ValueJsonError> {
        Self::try_from_json_value_depth(v, 0)
    }

    /// Decode one JSON value from a persisted property bag.
    ///
    /// Unlike [`Self::try_from_json_value`], this path treats every JSON
    /// object recursively as an open [`Value::Map`]. Stored property bags
    /// contain customer values, never internal Node / Relationship / Path
    /// result-row views, so applying entity-shape detection here would make
    /// ordinary objects such as `{id, label}` ambiguous and lossy.
    ///
    /// The scalar, list, number-coercion, and
    /// [`MAX_JSON_DECODE_DEPTH`] behavior is otherwise identical to the
    /// query-result decoder.
    ///
    /// # Errors
    ///
    /// - [`ValueJsonError::NumberOutOfRange`] when a JSON number cannot be
    ///   represented by the runtime number variants.
    /// - [`ValueJsonError::NestingTooDeep`] when the value exceeds
    ///   [`MAX_JSON_DECODE_DEPTH`].
    pub fn try_from_json_property_value(v: &JsonValue) -> Result<Self, ValueJsonError> {
        Self::try_from_json_value_depth_with_mode(v, 0, JsonObjectDecodeMode::MapOnly)
    }

    /// Depth-tracked core of [`Self::try_from_json_value`]. `depth` is
    /// the current nesting level (0 at the top of the decode); exceeding
    /// [`MAX_JSON_DECODE_DEPTH`] surfaces
    /// [`ValueJsonError::NestingTooDeep`] rather than recursing into a
    /// stack overflow. Per ADR-191 (D-12) — the decode path is
    /// network-reachable, so the recursive `Value` lattice needs a depth
    /// bound on construction from untrusted JSON.
    fn try_from_json_value_depth(v: &JsonValue, depth: usize) -> Result<Self, ValueJsonError> {
        Self::try_from_json_value_depth_with_mode(v, depth, JsonObjectDecodeMode::EntityAware)
    }

    /// Shared recursive JSON decoder. `object_mode` distinguishes query
    /// result rows, where internal entity views are meaningful, from stored
    /// property bags, where every object is customer map data.
    fn try_from_json_value_depth_with_mode(
        v: &JsonValue,
        depth: usize,
        object_mode: JsonObjectDecodeMode,
    ) -> Result<Self, ValueJsonError> {
        if depth > MAX_JSON_DECODE_DEPTH {
            return Err(ValueJsonError::NestingTooDeep {
                depth,
                max: MAX_JSON_DECODE_DEPTH,
            });
        }
        match v {
            JsonValue::Null => Ok(Value::Null),
            JsonValue::Bool(b) => Ok(Value::Boolean(*b)),
            JsonValue::Number(n) => json_number_to_value(n),
            JsonValue::String(s) => Ok(Value::String(s.clone())),
            JsonValue::Array(arr) => {
                let elems: Result<Vec<Value>, _> = arr
                    .iter()
                    .map(|e| Value::try_from_json_value_depth_with_mode(e, depth + 1, object_mode))
                    .collect();
                Ok(Value::List(elems?))
            }
            JsonValue::Object(obj) => {
                match object_mode {
                    JsonObjectDecodeMode::EntityAware => json_object_to_value(obj, depth),
                    JsonObjectDecodeMode::MapOnly => Ok(Value::Map(
                        properties_from_json_with_mode(obj, depth, object_mode)?,
                    )),
                }
            }
        }
    }
}

/// Controls whether JSON objects are interpreted as internal query-result
/// entities or unconditionally retained as customer map data.
#[derive(Clone, Copy)]
enum JsonObjectDecodeMode {
    EntityAware,
    MapOnly,
}

/// Project a property bag into a [`serde_json::Map`].
///
/// Keys are preserved verbatim (Cypher property keys are UTF-8
/// strings; `BTreeMap` guarantees deterministic iteration order so
/// round-trip preserves key order).
fn properties_to_json(props: &BTreeMap<String, Value>) -> JsonMap<String, JsonValue> {
    let mut m = JsonMap::new();
    for (k, v) in props {
        m.insert(k.clone(), v.to_json_value());
    }
    m
}

/// Project a [`serde_json::Map`] back into a property bag. `depth` is
/// the nesting level of the object itself; each value decodes at
/// `depth + 1` so the [`MAX_JSON_DECODE_DEPTH`] bound spans the whole
/// recursive descent (ADR-191 D-12).
fn properties_from_json(
    obj: &JsonMap<String, JsonValue>,
    depth: usize,
) -> Result<BTreeMap<String, Value>, ValueJsonError> {
    properties_from_json_with_mode(obj, depth, JsonObjectDecodeMode::EntityAware)
}

/// Mode-aware map descent used by the stored-property decoder to keep the
/// map-only rule recursive through nested objects and lists.
fn properties_from_json_with_mode(
    obj: &JsonMap<String, JsonValue>,
    depth: usize,
    object_mode: JsonObjectDecodeMode,
) -> Result<BTreeMap<String, Value>, ValueJsonError> {
    let mut m = BTreeMap::new();
    for (k, v) in obj {
        m.insert(
            k.clone(),
            Value::try_from_json_value_depth_with_mode(v, depth + 1, object_mode)?,
        );
    }
    Ok(m)
}

/// Decode a JSON number into the [`Value`] integer / float variants.
///
/// Order: `as_i64` (signed integer fast path) → `as_u64` (large
/// unsigned integers that don't fit `i64`) → `as_f64` (fractional /
/// scientific). A `serde_json::Number` always carries one of these
/// representations per its source code; the order priority handles
/// the i64-positive-overflow case correctly (a JSON literal `2^63`
/// fits `u64` but not `i64`; we widen to `Value::Float` for the
/// caller's downstream consistency).
fn json_number_to_value(n: &JsonNumber) -> Result<Value, ValueJsonError> {
    if let Some(i) = n.as_i64() {
        return Ok(Value::Integer(i));
    }
    if let Some(u) = n.as_u64() {
        // u64 in JSON literal — widen to f64 (round-trip lossless for
        // u <= 2^53; lossy beyond that, matching the IEEE-754 mantissa
        // budget). Per `feedback_avoid_speculative_scaffolding.md`,
        // we don't add a `Value::UnsignedInteger` variant on
        // speculation — Cypher 9's integer type is i64.
        return Ok(Value::Float(u as f64));
    }
    if let Some(f) = n.as_f64() {
        return Ok(Value::Float(f));
    }
    Err(ValueJsonError::NumberOutOfRange {
        literal: n.to_string(),
    })
}

/// Decode a JSON object into a [`Value::Node`] / [`Value::Relationship`]
/// / [`Value::Map`].
///
/// Detection rules (PRECEDENCE matters — ADR-191 D-7):
/// - Object with `id` + `from` + `to` keys → [`Value::Relationship`].
/// - Object with `id` + `label` keys → [`Value::Node`].
/// - **Any OTHER object → [`Value::Map`]** (the entity shapes are
///   checked FIRST, so `{id, label, properties}` still decodes as a
///   `Node` and `{id, from, to, …}` as a `Relationship`; only objects
///   matching NEITHER entity shape decode as a map). Keys are JSON
///   strings; nested objects recurse (each one level deeper).
///
/// The detection ordering matches the encoder: a Relationship has
/// strictly more required keys than a Node, so check Relationship
/// first; the bare-map branch is last (Rel → Node → Map precedence).
///
/// `depth` is the nesting level of this object; nested values decode at
/// `depth + 1` so the [`MAX_JSON_DECODE_DEPTH`] bound holds across the
/// whole tree.
fn json_object_to_value(
    obj: &JsonMap<String, JsonValue>,
    depth: usize,
) -> Result<Value, ValueJsonError> {
    if obj.contains_key("from") && obj.contains_key("to") && obj.contains_key("id") {
        // Relationship.
        let id_raw =
            obj.get("id")
                .and_then(JsonValue::as_u64)
                .ok_or(ValueJsonError::UnsupportedShape {
                    kind: "object missing or non-numeric `id`",
                })?;
        let from_raw = obj.get("from").and_then(JsonValue::as_u64).ok_or(
            ValueJsonError::UnsupportedShape {
                kind: "object missing or non-numeric `from`",
            },
        )?;
        let to_raw =
            obj.get("to")
                .and_then(JsonValue::as_u64)
                .ok_or(ValueJsonError::UnsupportedShape {
                    kind: "object missing or non-numeric `to`",
                })?;
        let rel_type = match obj.get("rel_type") {
            Some(JsonValue::Null) | None => None,
            Some(JsonValue::Number(n)) => {
                let raw = n.as_u64().ok_or(ValueJsonError::UnsupportedShape {
                    kind: "rel_type non-u64",
                })?;
                Some(TypeId::new(u32::try_from(raw).map_err(|_| {
                    ValueJsonError::UnsupportedShape {
                        kind: "rel_type exceeds u32",
                    }
                })?))
            }
            Some(_) => {
                return Err(ValueJsonError::UnsupportedShape {
                    kind: "rel_type non-numeric",
                });
            }
        };
        let properties = match obj.get("properties") {
            Some(JsonValue::Object(props)) => properties_from_json(props, depth)?,
            None => BTreeMap::new(),
            Some(_) => {
                return Err(ValueJsonError::UnsupportedShape {
                    kind: "properties non-object",
                });
            }
        };
        // #871 — reconstruct the resolved rel-type NAME from the
        // Neo4j-style `type` string when the encoder emitted it, so the
        // `to_json_value` → `try_from_json_value` round-trip is identity
        // for a name-resolved RelView. Absent / non-string ⇒ `None`.
        let rel_type_name = match obj.get("type") {
            Some(JsonValue::String(s)) => Some(s.clone()),
            _ => None,
        };
        Ok(Value::Relationship(RelView {
            id: RelId::new(id_raw),
            from: NodeId::new(from_raw),
            to: NodeId::new(to_raw),
            rel_type,
            rel_type_name,
            properties,
        }))
    } else if obj.contains_key("id") && obj.contains_key("label") {
        // Node.
        let id_raw =
            obj.get("id")
                .and_then(JsonValue::as_u64)
                .ok_or(ValueJsonError::UnsupportedShape {
                    kind: "object missing or non-numeric `id`",
                })?;
        let label = match obj.get("label") {
            Some(JsonValue::Null) => None,
            Some(JsonValue::Number(n)) => {
                let raw = n.as_u64().ok_or(ValueJsonError::UnsupportedShape {
                    kind: "label non-u64",
                })?;
                Some(LabelId::new(u32::try_from(raw).map_err(|_| {
                    ValueJsonError::UnsupportedShape {
                        kind: "label exceeds u32",
                    }
                })?))
            }
            None => None,
            Some(_) => {
                return Err(ValueJsonError::UnsupportedShape {
                    kind: "label non-numeric",
                });
            }
        };
        let properties = match obj.get("properties") {
            Some(JsonValue::Object(props)) => properties_from_json(props, depth)?,
            None => BTreeMap::new(),
            Some(_) => {
                return Err(ValueJsonError::UnsupportedShape {
                    kind: "properties non-object",
                });
            }
        };
        // #871 — reconstruct the resolved label NAME from the Neo4j-
        // style `labels` list (first element) when the encoder emitted
        // it, so the round-trip is identity for a name-resolved
        // NodeView. Absent / empty / non-string ⇒ `None`.
        let label_name = match obj.get("labels") {
            Some(JsonValue::Array(arr)) => arr.first().and_then(|v| match v {
                JsonValue::String(s) => Some(s.clone()),
                _ => None,
            }),
            _ => None,
        };
        Ok(Value::Node(NodeView {
            id: NodeId::new(id_raw),
            label,
            label_name,
            properties,
        }))
    } else if obj.contains_key("start") && obj.contains_key("segments") {
        // Path (ADR-193 D-8). The `start` + `segments` conjunction is a
        // discriminator NO other value shape carries (Node = id+label;
        // Rel = id+from+to; a Map (ADR-191) admits ARBITRARY string
        // keys). BINDING decode precedence is Rel → Node → Path → Map:
        // this Path branch MUST stay BEFORE any future Map catch-all
        // (which would otherwise mis-claim a `{start, segments}` object).
        // This is the precise inverse of ADR-191's "Map is last" nuance —
        // a path is a STRUCTURED object, claimed before the open-key Map.
        json_object_to_path(obj, depth)
    } else {
        // ADR-191 D-7 / ADR-193 D-8 — an object matching NEITHER the
        // Relationship, Node, nor Path shape above is a plain openCypher
        // map. Decode every key/value pair recursively into a
        // `Value::Map`. BINDING decode precedence is Rel → Node → Path →
        // Map: the Path branch above claims `{start, segments}` BEFORE
        // this open-key Map catch-all (the precise inverse of ADR-191's
        // "Map is last" — a path is a STRUCTURED object), and a
        // `{id, label, properties}` object is still a `Node`, NOT a `Map`.
        Ok(Value::Map(properties_from_json(obj, depth)?))
    }
}

/// openCypher orderability type rank (CIP2016-06-14 global sort order,
/// smallest → largest): `Path < Relationship < Node < Map < List <
/// String < Boolean < Number < temporal < NULL`. Per ADR-193 (D-11) a
/// `Path` sorts FIRST (before `Relationship`); per ADR-191 (D-5) a `Map`
/// sorts AFTER `Node`/`Relationship` and BEFORE `List` / scalars. Used by
/// [`compare_orderability`] for the cross-type tiebreak.
fn orderability_type_rank(v: &Value) -> u8 {
    match v {
        // ADR-193 D-11 — paths sort FIRST in the global type-order.
        Value::Path(_) => 0,
        Value::Relationship(_) => 1,
        Value::Node(_) => 2,
        Value::Map(_) => 3,
        Value::List(_) => 4,
        Value::String(_) => 5,
        Value::Boolean(_) => 6,
        Value::Integer(_) | Value::Float(_) => 7,
        // The temporal family is not enumerated in the ADR-191 D-5 core
        // order; pin it deterministically just after Number so map
        // tiebreaks over temporal-valued maps are total (full temporal
        // orderability is out of this slice's scope).
        Value::Temporal(_)
        | Value::LocalDateTime(_)
        | Value::Date(_)
        | Value::Duration(_)
        | Value::Decimal(_) => 8,
        Value::Null => 9,
    }
}

/// openCypher **orderability** total order over all [`Value`]s
/// (CIP2016-06-14): the order `ORDER BY` uses — total over every value
/// and NEVER erroring (distinct from comparability, where `<`/`>` on
/// maps is `null`). Within a type, the natural order; across types, the
/// global [`orderability_type_rank`]. Maps tiebreak by their sorted-key
/// sequence (lexicographic), then pairwise value orderability; lists
/// element-wise then by length. Per ADR-191 (D-5).
///
/// # Scoped invocation
///
/// `sort::compare_non_null_values` and `aggregate::compare_values` route
/// **map- and list-involved** comparisons here; the remaining cross-type
/// *scalar* pairs retain their pre-existing `Ordering::Equal`
/// (stable-sort-preserving) behavior. Full convergence of all
/// cross-type scalar comparisons to this total order is tracked as
/// OQ-191-1 (a separate TCK-ratchet-moving change), NOT this slice.
#[must_use]
pub(crate) fn compare_orderability(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (Value::Boolean(x), Value::Boolean(y)) => x.cmp(y),
        (Value::Integer(x), Value::Integer(y)) => x.cmp(y),
        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (Value::Integer(x), Value::Float(y)) => {
            (*x as f64).partial_cmp(y).unwrap_or(Ordering::Equal)
        }
        (Value::Float(x), Value::Integer(y)) => {
            x.partial_cmp(&(*y as f64)).unwrap_or(Ordering::Equal)
        }
        (Value::String(x), Value::String(y)) => x.cmp(y),
        (Value::Node(x), Value::Node(y)) => x.id.raw().cmp(&y.id.raw()),
        (Value::Relationship(x), Value::Relationship(y)) => x.id.raw().cmp(&y.id.raw()),
        (Value::List(x), Value::List(y)) => compare_list_orderability(x, y),
        (Value::Map(x), Value::Map(y)) => compare_map_orderability(x, y),
        // ADR-193 D-11 — two paths order by node-id then rel-id sequence
        // (deterministic, collision-free; distinct paths never compare
        // Equal). Without this arm the cross-type catch-all would collapse
        // two paths to Equal (same rank 0) and merge them under
        // ORDER BY / DISTINCT.
        (Value::Path(x), Value::Path(y)) => x.cmp_paths(y),
        // Cross-type (and temporal-vs-temporal, which the rank collapses
        // to Equal — temporal orderability is out of scope): the
        // openCypher global type rank.
        _ => orderability_type_rank(a).cmp(&orderability_type_rank(b)),
    }
}

/// Orderability tiebreak for two lists: element-wise by orderability,
/// then the shorter list sorts first (prefix-smaller). Per ADR-191 D-5
/// (used for nested-list map values).
fn compare_list_orderability(x: &[Value], y: &[Value]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    for (ex, ey) in x.iter().zip(y) {
        let eo = compare_orderability(ex, ey);
        if eo != Ordering::Equal {
            return eo;
        }
    }
    x.len().cmp(&y.len())
}

/// Orderability tiebreak for two maps (ADR-191 D-5): compare the sorted
/// key sequences lexicographically; on a key prefix-tie, the smaller key
/// set sorts first; on identical key sets, compare values pairwise by
/// orderability. Deterministic and collision-free — two DISTINCT maps
/// never compare `Equal` (so GROUP BY / ORDER BY never merge them).
fn compare_map_orderability(
    x: &BTreeMap<String, Value>,
    y: &BTreeMap<String, Value>,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    // 1. Sorted key sequences (BTreeMap iterates in sorted key order).
    for (kx, ky) in x.keys().zip(y.keys()) {
        let ko = kx.cmp(ky);
        if ko != Ordering::Equal {
            return ko;
        }
    }
    // Key prefix equal → the map with fewer keys sorts first.
    match x.len().cmp(&y.len()) {
        Ordering::Equal => {}
        neq => return neq,
    }
    // 2. Identical key set → pairwise value orderability.
    for (vx, vy) in x.values().zip(y.values()) {
        let vo = compare_orderability(vx, vy);
        if vo != Ordering::Equal {
            return vo;
        }
    }
    Ordering::Equal
}

/// Decode a `{start, segments:[{relationship, end}]}` object into a
/// [`Value::Path`] (ADR-193 D-8 reverse bridge). `depth` is threaded from
/// [`json_object_to_value`] so the [`MAX_JSON_DECODE_DEPTH`] bound spans
/// the whole nested path (nested node/rel sub-objects decode at
/// `depth + 1`, matching the Map / List recursion).
///
/// Structural validation: `start` must decode as a [`Value::Node`];
/// `segments` must be a JSON array whose every element is an object with
/// a `relationship` (decoding as [`Value::Relationship`]) and an `end`
/// (decoding as [`Value::Node`]). Any deviation surfaces
/// [`ValueJsonError::UnsupportedShape`] — we never silently coerce a
/// mis-shaped object into a path.
fn json_object_to_path(
    obj: &JsonMap<String, JsonValue>,
    depth: usize,
) -> Result<Value, ValueJsonError> {
    let start = match Value::try_from_json_value_depth(
        obj.get("start").ok_or(ValueJsonError::UnsupportedShape {
            kind: "path object missing `start`",
        })?,
        depth + 1,
    )? {
        Value::Node(n) => n,
        _ => {
            return Err(ValueJsonError::UnsupportedShape {
                kind: "path `start` is not a node object",
            });
        }
    };
    let seg_array = match obj.get("segments") {
        Some(JsonValue::Array(arr)) => arr,
        _ => {
            return Err(ValueJsonError::UnsupportedShape {
                kind: "path `segments` is not an array",
            });
        }
    };
    let mut segments = Vec::with_capacity(seg_array.len());
    for seg in seg_array {
        let seg_obj = match seg {
            JsonValue::Object(o) => o,
            _ => {
                return Err(ValueJsonError::UnsupportedShape {
                    kind: "path segment is not an object",
                });
            }
        };
        let rel = match Value::try_from_json_value_depth(
            seg_obj
                .get("relationship")
                .ok_or(ValueJsonError::UnsupportedShape {
                    kind: "path segment missing `relationship`",
                })?,
            depth + 1,
        )? {
            Value::Relationship(r) => r,
            _ => {
                return Err(ValueJsonError::UnsupportedShape {
                    kind: "path segment `relationship` is not a relationship object",
                });
            }
        };
        let end = match Value::try_from_json_value_depth(
            seg_obj.get("end").ok_or(ValueJsonError::UnsupportedShape {
                kind: "path segment missing `end`",
            })?,
            depth + 1,
        )? {
            Value::Node(n) => n,
            _ => {
                return Err(ValueJsonError::UnsupportedShape {
                    kind: "path segment `end` is not a node object",
                });
            }
        };
        segments.push(PathSegment { rel, end });
    }
    Ok(Value::Path(PathView { start, segments }))
}

/// Stub-friendly node view. v1.0-alpha tests populate via
/// [`crate::executor::StubExecutorSubstrate`]; production wiring will
/// surface storage-rooted views at M4-08+.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeView {
    pub id: NodeId,
    /// Resolved primary label, if any. Cypher 9 admits multi-label
    /// nodes at runtime; v1.0 grammar only admits single-label
    /// patterns (multi-label rejected at M4-22). The view records the
    /// first label observed.
    pub label: Option<LabelId>,
    /// Catalog-resolved label NAME, if known (#871). [`Self::label`]
    /// carries the interned [`LabelId`], which is opaque outside the
    /// catalog; this field carries the human-readable name the catalog
    /// reverse-resolves at materialization time (the same point the
    /// property bag is resolved to string keys). `labels(n)` and the
    /// Bolt / JSON node serializers surface THIS — never the opaque
    /// `LabelId` debug form (`"LabelId(1)"`) that #871 exposed to
    /// drivers. `None` means either the node has no label, or the name
    /// has not been reverse-resolved (the serializers then emit an
    /// empty labels list rather than leaking the id).
    ///
    /// It is a DERIVED display attribute (a pure function of
    /// [`Self::label`] + the catalog), not part of node identity —
    /// openCypher node identity is by id. It is included in the derived
    /// `PartialEq` for honesty (two views of the same node materialized
    /// through the same catalog carry the same name), but no equality /
    /// dedup path keys on it independently of `label`.
    pub label_name: Option<String>,
    /// Property bag — keyed by property name (caller resolves to
    /// [`arcgraph_core::PropertyId`] via the catalog at expression-
    /// evaluation time).
    pub properties: BTreeMap<String, Value>,
}

impl NodeView {
    /// Construct a NodeView with no properties and no resolved label
    /// name. Production materialization + the CREATE op attach the name
    /// via [`Self::with_label_name`] / direct field set (#871).
    #[must_use]
    pub fn new(id: NodeId, label: Option<LabelId>) -> Self {
        Self {
            id,
            label,
            label_name: None,
            properties: BTreeMap::new(),
        }
    }

    /// Attach the catalog-resolved label NAME. Chainable. Surfaced by
    /// `labels(n)` + the Bolt / JSON node serializers (#871).
    #[must_use]
    pub fn with_label_name(mut self, name: impl Into<String>) -> Self {
        self.label_name = Some(name.into());
        self
    }

    /// Add a property. Chainable for fluent test construction.
    #[must_use]
    pub fn with_property(mut self, key: impl Into<String>, value: Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
}

/// Stub-friendly relationship view.
#[derive(Debug, Clone, PartialEq)]
pub struct RelView {
    pub id: RelId,
    pub from: NodeId,
    pub to: NodeId,
    pub rel_type: Option<TypeId>,
    /// Catalog-resolved relationship-type NAME, if known (#871). The
    /// reverse-resolution sibling of `Self::label_name` on
    /// [`NodeView`]: [`Self::rel_type`] carries the opaque interned
    /// [`TypeId`]; this carries the human-readable name `type(r)` + the
    /// Bolt / JSON relationship serializers surface, never the
    /// `"TypeId(1)"` debug form. `None` ⇒ untyped or unresolved.
    pub rel_type_name: Option<String>,
    pub properties: BTreeMap<String, Value>,
}

impl RelView {
    /// Construct a RelView with no properties and no resolved rel-type
    /// name (#871 — production materialization + the CREATE-rel op
    /// attach it).
    #[must_use]
    pub fn new(id: RelId, from: NodeId, to: NodeId, rel_type: Option<TypeId>) -> Self {
        Self {
            id,
            from,
            to,
            rel_type,
            rel_type_name: None,
            properties: BTreeMap::new(),
        }
    }

    /// Attach the catalog-resolved relationship-type NAME. Chainable.
    /// Surfaced by `type(r)` + the Bolt / JSON serializers (#871).
    #[must_use]
    pub fn with_rel_type_name(mut self, name: impl Into<String>) -> Self {
        self.rel_type_name = Some(name.into());
        self
    }

    /// Add a property. Chainable for fluent test construction.
    #[must_use]
    pub fn with_property(mut self, key: impl Into<String>, value: Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
}

/// One segment of a [`PathView`]: a single relationship traversal
/// landing on the segment's `end` node (ADR-193 D-1/D-2).
///
/// `rel` carries the relationship in **stored** orientation
/// ([`RelView::from`] / [`RelView::to`] are storage order, NOT traversal
/// order). The path's traversal order is encoded by the node sequence
/// (the predecessor node — [`PathView::start`] for the first segment, or
/// the prior segment's `end` — followed by THIS segment's `end`).
///
/// **OQ-193-1 resolved (no `traversed_forward` flag):** the direction a
/// segment was traversed is recoverable from the adjacent node IDs
/// (`rel.from == predecessor.id` ⇒ traversed forward; otherwise
/// backward; a self-loop is direction-agnostic). Because the explicit
/// `end: NodeView` already records the actual landing node, it is
/// strictly MORE information than a `traversed_forward: bool` and
/// satisfies D-2's "or equivalent" clause — so no separate flag is
/// stored. This also keeps the JSON shape (D-8) flag-free and makes the
/// round-trip clean.
#[derive(Debug, Clone, PartialEq)]
pub struct PathSegment {
    /// The relationship traversed (stored orientation).
    pub rel: RelView,
    /// The node this segment lands on, in TRAVERSAL order.
    pub end: NodeView,
}

/// A path value (ADR-193 D-1): the openCypher v9 §3 alternating
/// node/relationship sequence `start, seg₀.rel, seg₀.end, seg₁.rel,
/// seg₁.end, …`.
///
/// The representation STRUCTURALLY enforces the openCypher §3 invariant
/// **`#nodes = #rels + 1`**: `start` plus one `end` per segment is
/// exactly `segments.len() + 1` nodes against `segments.len()`
/// relationships — the invariant cannot be violated by construction. An
/// empty (zero-length) path `MATCH p = (a)` is `PathView { start: a,
/// segments: [] }` (D-6): one node, zero relationships, `length 0`.
///
/// # Derives — and what is deliberately ABSENT
///
/// Derives `Debug, Clone, PartialEq` ONLY (matching [`NodeView`] /
/// [`RelView`]). It deliberately does NOT derive:
/// - **`Ord` / `PartialOrd`** — but NOT because paths are non-orderable.
///   openCypher paths ARE orderable (orderability is a TOTAL order over
///   all values that never errors; a path sorts FIRST in the global
///   type-order per CIP2016-06-14 — D-11). A DERIVED `Ord` would impose
///   Rust variant-declaration order (≠ openCypher orderability) and
///   `NodeView`/`RelView` carry no total `Ord` anyway. The ordering is
///   instead provided by the EXPLICIT [`PathView::cmp_paths`] arm, wired
///   into `executor::ops::sort` / `executor::ops::aggregate` (the same
///   rationale as ADR-191 D-10 — explicit compare arm, not a derive,
///   and NOT an error).
/// - **`Hash`** — not needed; a path's iteration order is intrinsic to
///   the sequence, so there is no iteration-order hazard to guard (unlike
///   the `BTreeMap` requirement a future `Value::Map` carries).
#[derive(Debug, Clone, PartialEq)]
pub struct PathView {
    /// The path's first node (the only node of a zero-length path).
    pub start: NodeView,
    /// The ordered segments. `segments.len()` is the path's hop-count
    /// (`length(p)`); empty for a zero-length path.
    pub segments: Vec<PathSegment>,
}

impl PathView {
    /// Construct a zero-length path rooted at `start` (D-6).
    #[must_use]
    pub fn new(start: NodeView) -> Self {
        Self {
            start,
            segments: Vec::new(),
        }
    }

    /// Append a segment (a relationship + the node it lands on, in
    /// traversal order). Chainable for fluent construction.
    #[must_use]
    pub fn with_segment(mut self, rel: RelView, end: NodeView) -> Self {
        self.segments.push(PathSegment { rel, end });
        self
    }

    /// The path's hop-count = number of relationships = `length(p)`
    /// (ADR-193 D-7). A zero-length path returns 0.
    #[must_use]
    pub fn hop_count(&self) -> usize {
        self.segments.len()
    }

    /// The path's nodes in TRAVERSAL order (`nodes(p)` projection, D-7):
    /// `[start, seg₀.end, seg₁.end, …]`. Always `hop_count() + 1` nodes
    /// (the `#nodes = #rels + 1` invariant).
    #[must_use]
    pub fn nodes(&self) -> Vec<NodeView> {
        let mut out = Vec::with_capacity(self.segments.len() + 1);
        out.push(self.start.clone());
        out.extend(self.segments.iter().map(|s| s.end.clone()));
        out
    }

    /// The path's relationships in TRAVERSAL order (`relationships(p)`
    /// projection, D-7): `[seg₀.rel, seg₁.rel, …]`. Relationship
    /// IDENTITY (stored `from`/`to`) is preserved — direction of
    /// traversal is read from the node sequence, not from mutating the
    /// rel.
    #[must_use]
    pub fn relationships(&self) -> Vec<RelView> {
        self.segments.iter().map(|s| s.rel.clone()).collect()
    }

    /// ADR-193 D-11 — the DETERMINISTIC intra-path orderability compare
    /// (openCypher orderability is a total order that never errors; paths
    /// sort FIRST in the global type-order, and two paths are ordered by
    /// their NODE-ID sequence lexicographically, then by their REL-ID
    /// sequence on tie). Distinct paths NEVER compare `Equal` (so they do
    /// not merge under `ORDER BY` / `DISTINCT`); two paths that compare
    /// `Equal` here are equal under D-10 too (identical node + rel ID
    /// sequences). This is an EXPLICIT compare arm, NOT a derived `Ord`
    /// (a derive would impose Rust variant order ≠ openCypher order).
    #[must_use]
    pub fn cmp_paths(&self, other: &PathView) -> std::cmp::Ordering {
        let node_ids = |p: &PathView| -> Vec<u64> {
            std::iter::once(p.start.id.raw())
                .chain(p.segments.iter().map(|s| s.end.id.raw()))
                .collect()
        };
        let rel_ids =
            |p: &PathView| -> Vec<u64> { p.segments.iter().map(|s| s.rel.id.raw()).collect() };
        // `Vec<u64>::cmp` is lexicographic (element-wise, then length),
        // which is exactly the openCypher list-orderability rule applied
        // to the id sequences.
        node_ids(self)
            .cmp(&node_ids(other))
            .then_with(|| rel_ids(self).cmp(&rel_ids(other)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_is_distinct_from_boolean_false() {
        assert!(Value::Null.is_null());
        assert!(!Value::Boolean(false).is_null());
        // Critical 3VL invariant: Null != Boolean(false).
        assert_ne!(Value::Null, Value::Boolean(false));
    }

    #[test]
    fn as_bool_returns_none_for_null() {
        // NULL is NOT a boolean for `as_bool` purposes — the 3VL
        // helper takes over from the predicate-evaluation site.
        assert_eq!(Value::Null.as_bool(), None);
        assert_eq!(Value::Boolean(true).as_bool(), Some(true));
        assert_eq!(Value::Boolean(false).as_bool(), Some(false));
    }

    #[test]
    fn as_f64_widens_integer() {
        assert_eq!(Value::Integer(42).as_f64(), Some(42.0));
        assert_eq!(Value::Float(1.5).as_f64(), Some(1.5));
        assert_eq!(Value::Null.as_f64(), None);
        assert_eq!(Value::String("x".into()).as_f64(), None);
    }

    #[test]
    fn node_view_round_trips_properties() {
        let n = NodeView::new(NodeId::new(7), Some(LabelId::new(1)))
            .with_property("name", Value::String("Alice".into()))
            .with_property("age", Value::Integer(30));
        assert_eq!(n.id, NodeId::new(7));
        assert_eq!(n.label, Some(LabelId::new(1)));
        assert_eq!(
            n.properties.get("name"),
            Some(&Value::String("Alice".into()))
        );
        assert_eq!(n.properties.get("age"), Some(&Value::Integer(30)));
    }

    #[test]
    fn rel_view_carries_endpoints_and_type() {
        let r = RelView::new(
            RelId::new(99),
            NodeId::new(1),
            NodeId::new(2),
            Some(TypeId::new(1)),
        )
        .with_property("since", Value::Integer(2020));
        assert_eq!(r.from, NodeId::new(1));
        assert_eq!(r.to, NodeId::new(2));
        assert_eq!(r.rel_type, Some(TypeId::new(1)));
        assert_eq!(r.properties.get("since"), Some(&Value::Integer(2020)));
    }

    #[test]
    fn list_value_can_be_heterogeneous() {
        // Cypher 9 §3.5 admits heterogeneous lists; the executor
        // preserves that. Forward pin against a future "homogeneous-
        // only" optimization that would silently break Cypher
        // compatibility.
        let v = Value::List(vec![
            Value::Integer(1),
            Value::String("two".into()),
            Value::Null,
        ]);
        match v {
            Value::List(elems) => assert_eq!(elems.len(), 3),
            _ => panic!("expected List"),
        }
    }

    // -----------------------------------------------------------------
    // W13β M4-81 — Value ↔ serde_json::Value bridge pin set.
    //
    // These cover the JSON half of the spawn prompt's
    // "TOON roundtrip + JSON roundtrip + per-row + per-batch +
    // nested types (4 × 2 serializers ÷ overlap = 8 distinct cases)".
    // The TOON half lives in arcgraph-mcp's
    // `tests/m4_81_materialize_serializer_roundtrip.rs` (bounded-context
    // discipline — TOON is an arcgraph-mcp concern; the bridge below
    // is the contract surface both halves share).
    // -----------------------------------------------------------------

    fn roundtrip_json(v: &Value) -> Value {
        let json = v.to_json_value();
        Value::try_from_json_value(&json).expect("decode")
    }

    /// Per-cell JSON roundtrip — primitive scalars (Null / Bool / Int /
    /// Float / String). The W12γ `materialize_proptest` excludes Float
    /// to avoid NaN, but FINITE floats round-trip cleanly via the
    /// `serde_json::Number::from_f64` path.
    #[test]
    fn json_roundtrip_per_row_primitive_scalars() {
        let row: Vec<Value> = vec![
            Value::Null,
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Integer(0),
            Value::Integer(-7),
            Value::Integer(i64::MAX),
            Value::Integer(i64::MIN),
            Value::Float(1.5),
            Value::Float(-1.234567), // arbitrary non-integer-valued finite f64
            Value::String(String::new()),
            Value::String("hello".into()),
            Value::String("emoji 🎉 + 漢字".into()),
        ];
        for cell in &row {
            assert_eq!(
                roundtrip_json(cell),
                *cell,
                "cell {cell:?} did not round-trip"
            );
        }
    }

    /// Per-batch JSON roundtrip — a multi-row Vec<Vec<Value>> whose
    /// per-cell roundtrip preserves order and shape.
    #[test]
    fn json_roundtrip_per_batch_multi_row() {
        let rows: Vec<Vec<Value>> = vec![
            vec![Value::Integer(1), Value::String("Ada".into())],
            vec![Value::Integer(2), Value::String("Bob".into())],
            vec![Value::Integer(3), Value::String("Cay".into())],
        ];
        for row in &rows {
            for cell in row {
                assert_eq!(roundtrip_json(cell), *cell);
            }
        }
        // Whole-row JSON projection composes via serde_json::Value::Array.
        let json_rows: JsonValue = JsonValue::Array(
            rows.iter()
                .map(|row| JsonValue::Array(row.iter().map(Value::to_json_value).collect()))
                .collect(),
        );
        // Round-trip via the array surface.
        let JsonValue::Array(json_arr) = json_rows else {
            panic!("expected Array")
        };
        let mut decoded: Vec<Vec<Value>> = Vec::new();
        for row_json in json_arr {
            let JsonValue::Array(cells) = row_json else {
                panic!("expected per-row Array")
            };
            decoded.push(
                cells
                    .iter()
                    .map(Value::try_from_json_value)
                    .collect::<Result<Vec<_>, _>>()
                    .expect("decode row"),
            );
        }
        assert_eq!(decoded, rows);
    }

    /// Nested-types JSON roundtrip — Node + Relationship + List, the
    /// three composite variants. The Node / Rel JSON shape is what
    /// the M5-07 / M5-11 / M5-13 renderers consume.
    #[test]
    fn json_roundtrip_nested_node_relationship_list() {
        let node = Value::Node(
            NodeView::new(NodeId::new(7), Some(LabelId::new(1)))
                .with_property("name", Value::String("Alice".into()))
                .with_property("age", Value::Integer(30)),
        );
        let rel = Value::Relationship(
            RelView::new(
                RelId::new(99),
                NodeId::new(1),
                NodeId::new(2),
                Some(TypeId::new(1)),
            )
            .with_property("since", Value::Integer(2020)),
        );
        let list = Value::List(vec![
            Value::Integer(1),
            Value::String("two".into()),
            Value::Null,
            // List-of-list — exercises the recursive descent.
            Value::List(vec![Value::Boolean(true), Value::Float(0.5)]),
        ]);
        assert_eq!(roundtrip_json(&node), node);
        assert_eq!(roundtrip_json(&rel), rel);
        assert_eq!(roundtrip_json(&list), list);
    }

    /// #871 — a node / rel carrying a catalog-resolved NAME round-trips
    /// through JSON: `to_json_value` emits the Neo4j-style `labels` /
    /// `type` name keys and `try_from_json_value` reconstructs
    /// `label_name` / `rel_type_name`. Asserts the name field DIRECTLY
    /// (not only via the derived `PartialEq`) so the oracle pins the name
    /// survival explicitly.
    #[test]
    fn json_roundtrip_preserves_resolved_label_and_type_names() {
        let node = Value::Node(
            NodeView::new(NodeId::new(7), Some(LabelId::new(1)))
                .with_label_name("Account")
                .with_property("name", Value::String("Alice".into())),
        );
        let rel = Value::Relationship(
            RelView::new(
                RelId::new(9),
                NodeId::new(1),
                NodeId::new(2),
                Some(TypeId::new(1)),
            )
            .with_rel_type_name("KNOWS"),
        );
        // Wire shape carries the resolved NAMES (Neo4j-style keys).
        assert_eq!(
            node.to_json_value().get("labels"),
            Some(&JsonValue::Array(vec![JsonValue::String("Account".into())])),
        );
        assert_eq!(
            rel.to_json_value().get("type"),
            Some(&JsonValue::String("KNOWS".into())),
        );
        // Round-trip is identity (incl. the resolved names — derived Eq).
        assert_eq!(roundtrip_json(&node), node);
        assert_eq!(roundtrip_json(&rel), rel);
        // Name survives explicitly.
        match roundtrip_json(&node) {
            Value::Node(n) => assert_eq!(n.label_name.as_deref(), Some("Account")),
            other => panic!("expected Node, got {other:?}"),
        }
        match roundtrip_json(&rel) {
            Value::Relationship(r) => assert_eq!(r.rel_type_name.as_deref(), Some("KNOWS")),
            other => panic!("expected Relationship, got {other:?}"),
        }
    }

    /// NaN / ±Inf are coerced to JSON `null` on encode, matching the
    /// W11ε TOON `encode_number` lossy contract. Round-trip yields
    /// `Value::Null` (NOT the original Float), pinned here so a future
    /// "preserve NaN" change is caught at the bridge boundary.
    #[test]
    fn json_encodes_nonfinite_floats_as_null() {
        for nf in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let v = Value::Float(nf);
            let json = v.to_json_value();
            assert_eq!(json, JsonValue::Null, "non-finite {nf} → JSON null");
            // Round-trip yields Null (lossy by spec).
            assert_eq!(roundtrip_json(&v), Value::Null);
        }
    }

    /// Node WITHOUT properties + Node with `label = None` round-trip
    /// cleanly. The JSON shape carries `label: null` and an empty
    /// `properties` object; the decoder reconstructs the same shape.
    #[test]
    fn json_roundtrip_unlabeled_node_and_empty_properties() {
        let n_no_label = Value::Node(NodeView::new(NodeId::new(42), None));
        let n_no_props = Value::Node(NodeView::new(NodeId::new(99), Some(LabelId::new(2))));
        assert_eq!(roundtrip_json(&n_no_label), n_no_label);
        assert_eq!(roundtrip_json(&n_no_props), n_no_props);
    }

    /// Rel WITHOUT properties + Rel with `rel_type = None` round-trip
    /// cleanly. Mirrors the unlabeled-node case.
    #[test]
    fn json_roundtrip_untyped_relationship() {
        let r_no_type = Value::Relationship(RelView::new(
            RelId::new(7),
            NodeId::new(1),
            NodeId::new(2),
            None,
        ));
        let r_no_props = Value::Relationship(RelView::new(
            RelId::new(8),
            NodeId::new(3),
            NodeId::new(4),
            Some(TypeId::new(1)),
        ));
        assert_eq!(roundtrip_json(&r_no_type), r_no_type);
        assert_eq!(roundtrip_json(&r_no_props), r_no_props);
    }

    /// ADR-191 D-7 — a JSON object matching NEITHER the Node nor the
    /// Relationship entity shape now decodes as a [`Value::Map`] (it was
    /// `UnsupportedShape` before; that rejection was the latent bug the
    /// map variant closes). Pins the new behavior so a future change
    /// doesn't silently regress map decode.
    #[test]
    fn try_from_json_plain_object_decodes_as_map() {
        let obj = serde_json::json!({"some_key": "some_value", "n": 7});
        let decoded = Value::try_from_json_value(&obj).expect("plain object decodes as map");
        let mut expected = BTreeMap::new();
        expected.insert("some_key".to_string(), Value::String("some_value".into()));
        expected.insert("n".to_string(), Value::Integer(7));
        assert_eq!(decoded, Value::Map(expected));
    }

    /// LabelId / TypeId are u32 newtypes; a JSON `label` exceeding
    /// `u32::MAX` surfaces `UnsupportedShape::"label exceeds u32"`.
    /// Pinned because the encoder widens to u64 for JSON wire-format
    /// cleanliness; the decoder's narrow-back step is the lossy edge.
    #[test]
    fn try_from_json_label_overflow_surfaces_error() {
        let bad = serde_json::json!({
            "id": 1,
            "label": (u32::MAX as u64) + 1,
            "properties": {},
        });
        let err = Value::try_from_json_value(&bad).unwrap_err();
        assert!(matches!(err, ValueJsonError::UnsupportedShape { .. }));
    }

    // =================================================================
    // ADR-191 — Value::Map bridge + orderability + depth-bound oracles.
    // =================================================================

    fn vmap(entries: &[(&str, Value)]) -> Value {
        Value::Map(
            entries
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        )
    }

    /// D-7 — map → JSON object → map round-trips byte-identical (incl.
    /// nested maps / lists / the previously-rejected plain-object case).
    #[test]
    fn json_roundtrip_map_nested() {
        let m = vmap(&[
            ("a", Value::Integer(1)),
            ("b", Value::String("x".into())),
            ("nested", vmap(&[("y", Value::Boolean(true))])),
            (
                "list",
                Value::List(vec![Value::Integer(1), vmap(&[("z", Value::Null)])]),
            ),
        ]);
        // Encode shape is a JSON object.
        assert!(matches!(m.to_json_value(), JsonValue::Object(_)));
        assert_eq!(roundtrip_json(&m), m, "map did not round-trip");
        // Empty map round-trips too.
        assert_eq!(
            roundtrip_json(&Value::Map(BTreeMap::new())),
            Value::Map(BTreeMap::new())
        );
    }

    /// D-7 decode PRECEDENCE pin — an entity-shaped object still decodes
    /// as Node / Relationship (NOT Map); only a non-entity object → Map.
    #[test]
    fn json_decode_precedence_entity_before_map() {
        let node = Value::Node(
            NodeView::new(NodeId::new(7), Some(LabelId::new(1)))
                .with_property("name", Value::String("Alice".into())),
        );
        let rel = Value::Relationship(
            RelView::new(
                RelId::new(9),
                NodeId::new(1),
                NodeId::new(2),
                Some(TypeId::new(1)),
            )
            .with_property("since", Value::Integer(2020)),
        );
        // Encode → decode keeps them as entities (Rel→Node→Map precedence).
        assert_eq!(roundtrip_json(&node), node, "node must NOT decode as Map");
        assert_eq!(
            roundtrip_json(&rel),
            rel,
            "relationship must NOT decode as Map"
        );
        // A plain object → Map (the fallthrough).
        let plain = serde_json::json!({"name": "Bob", "age": 30});
        assert_eq!(
            Value::try_from_json_value(&plain).unwrap(),
            vmap(&[
                ("name", Value::String("Bob".into())),
                ("age", Value::Integer(30))
            ]),
        );
    }

    /// #1383 guard — query-result rows deliberately use entity-aware JSON
    /// decoding. The stored-property map-only path must not weaken these
    /// genuine internal Node / Relationship discriminators.
    #[test]
    fn query_result_json_entity_shapes_still_decode_as_entities() {
        let node = serde_json::json!({
            "id": 5,
            "label": 2,
            "properties": {"name": "Ada"},
        });
        let relationship = serde_json::json!({
            "id": 9,
            "from": 5,
            "to": 6,
            "rel_type": 3,
            "properties": {"since": 2020},
        });

        assert!(matches!(
            Value::try_from_json_value(&node).expect("valid query-row node"),
            Value::Node(_),
        ));
        assert!(matches!(
            Value::try_from_json_value(&relationship).expect("valid query-row relationship"),
            Value::Relationship(_),
        ));
    }

    /// #1383 counterpart — the persisted-property bridge never applies the
    /// query-row entity discriminator, including recursively nested objects.
    #[test]
    fn property_json_entity_shapes_decode_as_maps_recursively() {
        let property_value = serde_json::json!({
            "descriptor": {"id": 5, "label": 2, "properties": {}},
            "edges": [{"id": 9, "from": 5, "to": 6}],
        });

        let decoded = Value::try_from_json_property_value(&property_value)
            .expect("valid persisted property map");
        let Value::Map(root) = decoded else {
            panic!("persisted property object must decode as Map");
        };
        assert!(matches!(root.get("descriptor"), Some(Value::Map(_))));
        assert!(matches!(
            root.get("edges"),
            Some(Value::List(edges)) if matches!(edges.as_slice(), [Value::Map(_)]),
        ));
    }

    /// D-12 — a JSON value nested deeper than the cap is rejected with
    /// `NestingTooDeep`, NOT a stack overflow (network-reachable hardening).
    #[test]
    fn try_from_json_deep_nesting_rejected_at_cap() {
        // Build `[[[ ... ]]]` MAX_JSON_DECODE_DEPTH + 5 deep. serde_json
        // parses up to its own 128-deep limit, so a depth in (cap, 128]
        // parses fine but is rejected by OUR decode bound.
        let depth = MAX_JSON_DECODE_DEPTH + 5;
        let mut s = String::new();
        for _ in 0..depth {
            s.push('[');
        }
        for _ in 0..depth {
            s.push(']');
        }
        let json: JsonValue = serde_json::from_str(&s).expect("serde parses (< 128 deep)");
        let err = Value::try_from_json_value(&json).unwrap_err();
        assert!(
            matches!(err, ValueJsonError::NestingTooDeep { .. }),
            "expected NestingTooDeep, got {err:?}"
        );
        // A shallow value is still accepted.
        let shallow = serde_json::json!({"a": [1, 2, {"b": 3}]});
        assert!(Value::try_from_json_value(&shallow).is_ok());
    }

    /// D-5 — orderability GLOBAL type order: a map sorts AFTER
    /// Node/Relationship and BEFORE List/String/Boolean/numeric.
    #[test]
    fn compare_orderability_map_global_type_order() {
        use std::cmp::Ordering;
        let map = vmap(&[("a", Value::Integer(1))]);
        let node = Value::Node(NodeView::new(NodeId::new(1), None));
        let rel = Value::Relationship(RelView::new(
            RelId::new(1),
            NodeId::new(1),
            NodeId::new(2),
            None,
        ));
        let list = Value::List(vec![Value::Integer(1)]);
        // Map AFTER node / relationship.
        assert_eq!(compare_orderability(&map, &node), Ordering::Greater);
        assert_eq!(compare_orderability(&map, &rel), Ordering::Greater);
        // Map BEFORE list / string / bool / numeric.
        assert_eq!(compare_orderability(&map, &list), Ordering::Less);
        assert_eq!(
            compare_orderability(&map, &Value::String("z".into())),
            Ordering::Less
        );
        assert_eq!(
            compare_orderability(&map, &Value::Boolean(false)),
            Ordering::Less
        );
        assert_eq!(
            compare_orderability(&map, &Value::Integer(0)),
            Ordering::Less
        );
        // Symmetry.
        assert_eq!(compare_orderability(&node, &map), Ordering::Less);
        assert_eq!(compare_orderability(&list, &map), Ordering::Greater);
    }

    /// D-5 — distinct maps NEVER compare Equal (no GROUP BY / ORDER BY
    /// collision); equal maps compare Equal; tiebreak by keys then values.
    #[test]
    fn compare_orderability_maps_are_deterministic_and_collision_free() {
        use std::cmp::Ordering;
        let m_a1 = vmap(&[("a", Value::Integer(1))]);
        let m_a2 = vmap(&[("a", Value::Integer(2))]);
        let m_b1 = vmap(&[("b", Value::Integer(1))]);
        let m_ab = vmap(&[("a", Value::Integer(1)), ("b", Value::Integer(2))]);
        // Equal maps → Equal.
        assert_eq!(compare_orderability(&m_a1, &m_a1.clone()), Ordering::Equal);
        // Same key, different value → value decides (never Equal).
        assert_eq!(compare_orderability(&m_a1, &m_a2), Ordering::Less);
        // Different key → key decides.
        assert_eq!(compare_orderability(&m_a1, &m_b1), Ordering::Less);
        // Key-prefix tie → shorter map sorts first.
        assert_eq!(compare_orderability(&m_a1, &m_ab), Ordering::Less);
        // No two DISTINCT maps collide.
        for (l, r) in [(&m_a1, &m_a2), (&m_a1, &m_b1), (&m_a1, &m_ab)] {
            assert_ne!(compare_orderability(l, r), Ordering::Equal);
        }
    }

    // -----------------------------------------------------------------
    // ADR-193 — Value::Path representation + projection + JSON bridge.
    // -----------------------------------------------------------------

    fn n(id: u64, label: u32) -> NodeView {
        NodeView::new(NodeId::new(id), Some(LabelId::new(label)))
    }
    fn r(id: u64, from: u64, to: u64) -> RelView {
        RelView::new(
            RelId::new(id),
            NodeId::new(from),
            NodeId::new(to),
            Some(TypeId::new(1)),
        )
    }

    /// `PathView` STRUCTURALLY enforces `#nodes = #rels + 1` (D-1), and
    /// the projections emit traversal order (D-7).
    #[test]
    fn path_projections_and_structural_invariant() {
        // a -[r1]-> b -[r2]-> c (2 hops).
        let p = PathView::new(n(1, 1))
            .with_segment(r(10, 1, 2), n(2, 1))
            .with_segment(r(11, 2, 3), n(3, 1));
        assert_eq!(p.hop_count(), 2);
        let nodes = p.nodes();
        let rels = p.relationships();
        // #nodes = #rels + 1, by construction.
        assert_eq!(nodes.len(), rels.len() + 1);
        assert_eq!(
            nodes.iter().map(|x| x.id.raw()).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            rels.iter().map(|x| x.id.raw()).collect::<Vec<_>>(),
            vec![10, 11]
        );
    }

    /// Zero-length path `p = (a)` (D-6): one node, zero rels, length 0.
    #[test]
    fn zero_length_path_is_valid() {
        let p = PathView::new(n(7, 1));
        assert_eq!(p.hop_count(), 0);
        assert_eq!(p.nodes().len(), 1);
        assert!(p.relationships().is_empty());
    }

    /// JSON round-trip (D-8 / test 10): a path → JSON object → path,
    /// value-identical, with nested node/rel properties preserved.
    #[test]
    fn path_json_round_trips_with_properties() {
        let start = NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
            .with_property("name", Value::String("a".into()));
        let rel = RelView::new(
            RelId::new(10),
            NodeId::new(1),
            NodeId::new(2),
            Some(TypeId::new(3)),
        )
        .with_property("since", Value::Integer(2020));
        let end = NodeView::new(NodeId::new(2), None).with_property("k", Value::Boolean(true));
        let path = Value::Path(PathView::new(start).with_segment(rel, end));
        assert_eq!(roundtrip_json(&path), path);

        // Zero-length path round-trips too.
        let empty = Value::Path(PathView::new(n(9, 2)));
        assert_eq!(roundtrip_json(&empty), empty);
    }

    /// Decode-precedence pin (D-8): a `{start, segments}` object decodes
    /// as Path; a `{id, label, properties}` object STILL decodes as Node
    /// (not mis-claimed as Path); a `{id, from, to}` object STILL decodes
    /// as Relationship. Locks the Rel → Node → Path ordering so a future
    /// Map catch-all (kept LAST) cannot mis-claim a path object.
    #[test]
    fn json_decode_precedence_path_vs_node_vs_rel() {
        let path_json = serde_json::json!({
            "start": {"id": 1, "label": 1, "properties": {}},
            "segments": [
                {"relationship": {"id": 10, "from": 1, "to": 2, "rel_type": 1, "properties": {}},
                 "end": {"id": 2, "label": null, "properties": {}}}
            ]
        });
        assert!(matches!(
            Value::try_from_json_value(&path_json).unwrap(),
            Value::Path(_)
        ));
        let node_json = serde_json::json!({"id": 5, "label": null, "properties": {}});
        assert!(matches!(
            Value::try_from_json_value(&node_json).unwrap(),
            Value::Node(_)
        ));
        let rel_json =
            serde_json::json!({"id": 9, "from": 1, "to": 2, "rel_type": null, "properties": {}});
        assert!(matches!(
            Value::try_from_json_value(&rel_json).unwrap(),
            Value::Relationship(_)
        ));
    }

    /// A malformed path object (segments not an array; start not a node)
    /// surfaces `UnsupportedShape` — never a silent coercion.
    #[test]
    fn malformed_path_object_surfaces_unsupported_shape() {
        let bad_segments =
            serde_json::json!({"start": {"id": 1, "label": null, "properties": {}}, "segments": 3});
        assert!(matches!(
            Value::try_from_json_value(&bad_segments),
            Err(ValueJsonError::UnsupportedShape { .. })
        ));
    }

    /// ADR-193 D-11 orderability — `cmp_paths` is DETERMINISTIC, orders by
    /// node-id sequence then rel-id sequence, and NEVER returns `Equal`
    /// for distinct paths (so they don't merge under sort/DISTINCT). This
    /// test BITES on a colliding (`_ => Equal`) or wrong-order impl.
    #[test]
    fn cmp_paths_is_deterministic_and_non_colliding() {
        use std::cmp::Ordering;
        // node-seq [1,2] < [1,3].
        let p12 = PathView::new(n(1, 1)).with_segment(r(10, 1, 2), n(2, 1));
        let p13 = PathView::new(n(1, 1)).with_segment(r(11, 1, 3), n(3, 1));
        assert_eq!(p12.cmp_paths(&p13), Ordering::Less);
        assert_eq!(p13.cmp_paths(&p12), Ordering::Greater);

        // Identical node-seq + rel-seq ⇒ Equal (these ARE the same path).
        let p12b = PathView::new(n(1, 1)).with_segment(r(10, 1, 2), n(2, 1));
        assert_eq!(p12.cmp_paths(&p12b), Ordering::Equal);

        // Same node-seq, DIFFERENT rel-seq ⇒ tiebreak on rel-id, NOT Equal
        // (distinct paths must not collide).
        let p12_rel_a = PathView::new(n(1, 1)).with_segment(r(10, 1, 2), n(2, 1));
        let p12_rel_b = PathView::new(n(1, 1)).with_segment(r(20, 1, 2), n(2, 1));
        assert_eq!(p12_rel_a.cmp_paths(&p12_rel_b), Ordering::Less);
        assert_ne!(
            p12_rel_a.cmp_paths(&p12_rel_b),
            Ordering::Equal,
            "distinct paths (differing rel id) must NOT collide"
        );

        // Shorter node-seq is a prefix ⇒ shorter sorts first (lexicographic
        // length rule).
        let p1 = PathView::new(n(1, 1));
        assert_eq!(p1.cmp_paths(&p12), Ordering::Less);
    }
}
