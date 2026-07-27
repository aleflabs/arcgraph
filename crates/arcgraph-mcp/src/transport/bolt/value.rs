//! W14δ M5-13 — Bridge between executor [`arcgraph_query::Value`] and
//! Bolt's [`PackValue`] lattice.
//!
//! The executor's [`arcgraph_query::executor::Value`] taxonomy was
//! sized for an internal vectorized executor: it has a `List` variant
//! but no `Map` (rows are positional `Vec<Value>`), and its
//! [`arcgraph_query::executor::value::NodeView`] /
//! [`arcgraph_query::executor::value::RelView`] structs use
//! ID-typed property keys + label IDs. Bolt-side values use
//! string-keyed maps and string-typed labels per the §"Type System"
//! spec.
//!
//! The bridge is deliberately one-way (executor → Bolt) at v1.0-α:
//! the server doesn't yet accept inbound RUN parameters of type Node
//! or Relationship (parameter shape is restricted to scalars, lists,
//! and maps of scalars per the openCypher subset). The inbound
//! parameter-shape rejection (Bytes / Struct PackValues arriving
//! inside a RUN's `parameters` map) lands at M5-12 alongside the
//! `QueryEngine` consumer that needs the JSON-lattice translation;
//! shipping the translator and its error taxonomy here without a
//! production consumer was flagged speculative scaffolding (review
//! finding W14δ M-2) and deferred.

use std::collections::BTreeMap;

use arcgraph_query::executor::Value as ExecValue;
use arcgraph_query::executor::eval::Parameters;
use arcgraph_query::executor::value::{NodeView, PathView, RelView};

use super::packstream::{PackValue, TAG_NODE, TAG_RELATIONSHIP};

/// Convert an executor [`ExecValue`] to a Bolt-side [`PackValue`].
///
/// At v1.0-α, [`ExecValue::Node`] / [`ExecValue::Relationship`]
/// translate to PackStream Struct values whose property maps use
/// stringified keys (the executor stores them as `String` already
/// per `NodeView::properties`'s `BTreeMap<String, Value>` shape, so
/// the bridge is direct).
///
/// # Identity-encoded `element_id`
///
/// Bolt 5.0 Node and Relationship structs require an `element_id`
/// field. v1.0-α emits a deterministic synthetic `element_id` of the
/// form `4:tenant:<numeric-id>` for nodes and `5:tenant:<numeric-id>`
/// for relationships — the type-prefix matches Neo4j's `element_id`
/// shape so existing drivers don't choke on the format. The tenant
/// slot is filled by the caller via [`exec_to_pack_with_tenant`];
/// this top-level converter uses an empty tenant slug since at the
/// value-level we don't yet have access to the tenant.
pub fn exec_to_pack(value: &ExecValue) -> PackValue {
    exec_to_pack_with_tenant(value, "")
}

/// Like [`exec_to_pack`] but takes a tenant slug used for the
/// synthetic `element_id` field on Node / Relationship structs.
pub fn exec_to_pack_with_tenant(value: &ExecValue, tenant_slug: &str) -> PackValue {
    match value {
        ExecValue::Null => PackValue::Null,
        ExecValue::Boolean(b) => PackValue::Boolean(*b),
        ExecValue::Integer(i) => PackValue::Integer(*i),
        ExecValue::Float(f) => PackValue::Float(*f),
        ExecValue::String(s) => PackValue::String(s.clone()),
        ExecValue::List(items) => PackValue::List(
            items
                .iter()
                .map(|v| exec_to_pack_with_tenant(v, tenant_slug))
                .collect(),
        ),
        ExecValue::Node(n) => pack_node_with_tenant(n, tenant_slug),
        ExecValue::Relationship(r) => pack_rel_with_tenant(r, tenant_slug),
        // ADR-191 D-7 — an openCypher map encodes as a native Bolt 5.0
        // Map (PackStream `0xA0..`), keyed in `BTreeMap` sorted-key order
        // (deterministic wire form). Nested maps/lists recurse.
        ExecValue::Map(m) => PackValue::Map(
            m.iter()
                .map(|(k, v)| (k.clone(), exec_to_pack_with_tenant(v, tenant_slug)))
                .collect(),
        ),
        // ADR-193 — a path encodes as a structured PackStream Map
        // mirroring the value-level JSON contract (D-8): `{start: <node>,
        // segments: [{relationship: <rel>, end: <node>}, ...]}`. Bolt 5.x
        // has a dedicated Path struct (tag 0x50 over UnboundRelationship
        // 0x72 + node/rel tables + index sequence); that native encoding
        // is forward-pinned to the MCP track (OQ-193-3) for the same
        // driver-support reason temporal cells ship string-form at v1.1.
        // The Map form is lossless and driver-readable today.
        ExecValue::Path(p) => pack_path_with_tenant(p, tenant_slug),
        // W23-V11-T-01 / ADR-090 — temporal + decimal cells encode as
        // Bolt UTF-8 strings (canonical ISO-8601 / decimal form). Bolt
        // 5.x has dedicated DateTime / Date / Duration structs (tags
        // 0x49, 0x44, 0x45 etc.) — forward-pinned at v1.2 when the
        // driver-side decoder support landscape is broader; v1.1 ships
        // string-form for the same reason serde_json::Value does.
        ExecValue::Temporal(t) => PackValue::String(format!("{t}")),
        ExecValue::LocalDateTime(ldt) => PackValue::String(format!("{ldt}")),
        ExecValue::Date(d) => PackValue::String(format!("{d}")),
        ExecValue::Duration(d) => PackValue::String(format!("{d}")),
        ExecValue::Decimal(d) => PackValue::String(format!("{d}")),
    }
}

/// Encode a [`NodeView`] as a Bolt 5.0 Node struct. Fields:
/// `[id, labels, properties, element_id]`.
fn pack_node_with_tenant(n: &NodeView, tenant_slug: &str) -> PackValue {
    let id = n.id.raw() as i64;
    // #871 — emit the catalog-resolved label NAME (e.g. `["Account"]`)
    // that the executor populated on `NodeView::label_name` at
    // materialization (`scan`/`expand` reverse-resolve via the intern
    // table; the CREATE op carries the verbatim name). Drivers (JS
    // neo4j-driver, Python neo4j) read `node.labels` and MUST see the
    // name, never the opaque `LabelId` debug form (`"LabelId(1)"`) the
    // pre-#871 `format!("{l:?}")` leaked. Empty list when no name is
    // resolved (unlabeled node, or — defensively — an unresolved id:
    // we never fall back to leaking the id).
    let labels = match &n.label_name {
        Some(name) => PackValue::List(vec![PackValue::String(name.clone())]),
        None => PackValue::List(vec![]),
    };
    let mut props = BTreeMap::new();
    for (k, v) in &n.properties {
        props.insert(k.clone(), exec_to_pack_with_tenant(v, tenant_slug));
    }
    let element_id = format!("4:{tenant_slug}:{id}");
    PackValue::Struct {
        tag: TAG_NODE,
        fields: vec![
            PackValue::Integer(id),
            labels,
            PackValue::Map(props),
            PackValue::String(element_id),
        ],
    }
}

/// Encode a [`RelView`] as a Bolt 5.0 Relationship struct. Fields:
/// `[id, start_id, end_id, type, properties, element_id,
/// start_element_id, end_element_id]`.
fn pack_rel_with_tenant(r: &RelView, tenant_slug: &str) -> PackValue {
    let id = r.id.raw() as i64;
    let start = r.from.raw() as i64;
    let end = r.to.raw() as i64;
    // #871 — emit the catalog-resolved rel-type NAME (e.g. `"KNOWS"`)
    // from `RelView::rel_type_name`, never the `"TypeId(1)"` debug form.
    // Empty string when unresolved (the Bolt Relationship `type` field
    // is non-nullable).
    let rel_type = match &r.rel_type_name {
        Some(name) => PackValue::String(name.clone()),
        None => PackValue::String(String::new()),
    };
    let mut props = BTreeMap::new();
    for (k, v) in &r.properties {
        props.insert(k.clone(), exec_to_pack_with_tenant(v, tenant_slug));
    }
    PackValue::Struct {
        tag: TAG_RELATIONSHIP,
        fields: vec![
            PackValue::Integer(id),
            PackValue::Integer(start),
            PackValue::Integer(end),
            rel_type,
            PackValue::Map(props),
            PackValue::String(format!("5:{tenant_slug}:{id}")),
            PackValue::String(format!("4:{tenant_slug}:{start}")),
            PackValue::String(format!("4:{tenant_slug}:{end}")),
        ],
    }
}

/// Encode a [`PathView`] as a structured PackStream Map mirroring the
/// value-level JSON contract (ADR-193 D-8): `{start: <Node struct>,
/// segments: [{relationship: <Relationship struct>, end: <Node
/// struct>}, ...]}`. The nodes/rels reuse the Bolt Node / Relationship
/// struct encoders so a driver sees the same element shapes it gets for
/// a bare `RETURN n` / `RETURN r`. v1.0 Map-form; the native Bolt Path
/// struct (tag 0x50) is forward-pinned to the MCP track (OQ-193-3).
fn pack_path_with_tenant(p: &PathView, tenant_slug: &str) -> PackValue {
    let mut obj = BTreeMap::new();
    obj.insert(
        "start".to_string(),
        pack_node_with_tenant(&p.start, tenant_slug),
    );
    let segments: Vec<PackValue> = p
        .segments
        .iter()
        .map(|seg| {
            let mut so = BTreeMap::new();
            so.insert(
                "relationship".to_string(),
                pack_rel_with_tenant(&seg.rel, tenant_slug),
            );
            so.insert(
                "end".to_string(),
                pack_node_with_tenant(&seg.end, tenant_slug),
            );
            PackValue::Map(so)
        })
        .collect();
    obj.insert("segments".to_string(), PackValue::List(segments));
    PackValue::Map(obj)
}

// =====================================================================
// Inbound: PackValue → ExecValue (#797 RUN `$param` binding)
// =====================================================================
//
// The W14δ-deferred inbound-parameter translator (review M-2) lands
// here now that its production consumer has arrived: #797 wires the
// Bolt RUN `parameters` map into the query engine so `$name` resolves.
// Shipping it WITH that consumer (not before) honors
// `feedback_avoid_speculative_scaffolding.md`.

/// Maximum container-nesting depth admitted for an inbound parameter
/// value (#797). The parameter map is the first UNTRUSTED value-surface
/// that reaches the executor's `evaluate`; this is a value-shape guard
/// at the query boundary complementing the PackStream decoder's
/// total-message-size + struct-depth bounds (per
/// `feedback_security_class_first_network_surface.md` — recursion-depth
/// bound on first-network-surface code). 64 is far above any legitimate
/// parameter shape (a vector embedding is a flat list; a properties map
/// is depth-1).
pub const MAX_PARAM_DEPTH: usize = 64;

/// Error converting an inbound Bolt RUN `parameters` value into an
/// executor [`ExecValue`] (#797).
///
/// Parameters must be the openCypher parameter shape — scalars or
/// (recursively) collections of scalars. Graph entities (Node /
/// Relationship / Path PackStream structs) and raw Bytes are NOT
/// admissible parameter VALUES (entities are query OUTPUTS, re-
/// materialized via MATCH; a caller binds ids / scalars instead).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ParamError {
    /// A PackStream Struct (Node `0x4E` / Relationship `0x52` / Path
    /// `0x50` / …) arrived as a parameter value. Rejected: parameters
    /// are scalar / collection inputs, never graph entities.
    #[error(
        "graph entities are not valid parameters (PackStream struct tag {tag:#04X}); \
         bind ids/scalars/collections, not Node/Relationship/Path"
    )]
    UnsupportedStruct {
        /// The offending PackStream struct tag byte.
        tag: u8,
    },
    /// PackStream Bytes arrived as a parameter value. The executor
    /// [`ExecValue`] lattice has no byte-blob variant at v1.0-α.
    #[error("raw byte blobs are not valid parameters at v1.0-α")]
    UnsupportedBytes,
    /// A parameter value (or nested element) exceeded
    /// [`MAX_PARAM_DEPTH`] container nesting.
    #[error("parameter nesting exceeds the maximum depth of {max}")]
    NestingTooDeep {
        /// The depth cap that was exceeded.
        max: usize,
    },
}

/// Convert one inbound Bolt [`PackValue`] (a RUN `parameters` map value)
/// into an executor [`ExecValue`] for #797 runtime parameter binding.
///
/// Admits Null / Boolean / Integer / Float / String and (recursively)
/// List / Map of the same. REJECTS PackStream Struct (Node /
/// Relationship / Path) and Bytes — see [`ParamError`].
///
/// # Errors
///
/// - [`ParamError::UnsupportedStruct`] for any PackStream Struct.
/// - [`ParamError::UnsupportedBytes`] for PackStream Bytes.
/// - [`ParamError::NestingTooDeep`] when nesting exceeds
///   [`MAX_PARAM_DEPTH`].
pub fn pack_to_exec(value: &PackValue) -> Result<ExecValue, ParamError> {
    pack_to_exec_at_depth(value, 0)
}

fn pack_to_exec_at_depth(value: &PackValue, depth: usize) -> Result<ExecValue, ParamError> {
    if depth > MAX_PARAM_DEPTH {
        return Err(ParamError::NestingTooDeep {
            max: MAX_PARAM_DEPTH,
        });
    }
    Ok(match value {
        PackValue::Null => ExecValue::Null,
        PackValue::Boolean(b) => ExecValue::Boolean(*b),
        PackValue::Integer(i) => ExecValue::Integer(*i),
        PackValue::Float(f) => ExecValue::Float(*f),
        PackValue::String(s) => ExecValue::String(s.clone()),
        PackValue::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(pack_to_exec_at_depth(it, depth + 1)?);
            }
            ExecValue::List(out)
        }
        PackValue::Map(m) => {
            let mut out = BTreeMap::new();
            for (k, v) in m {
                out.insert(k.clone(), pack_to_exec_at_depth(v, depth + 1)?);
            }
            ExecValue::Map(out)
        }
        // Inbound graph entities + byte blobs are not valid parameters.
        PackValue::Bytes(_) => return Err(ParamError::UnsupportedBytes),
        PackValue::Struct { tag, .. } => {
            return Err(ParamError::UnsupportedStruct { tag: *tag });
        }
    })
}

/// Convert an inbound Bolt RUN `parameters` map into the executor's
/// [`Parameters`] bag (#797). On a per-entry shape rejection, returns
/// the offending parameter NAME alongside the [`ParamError`] so the
/// caller can render a precise client error naming the bad parameter.
///
/// # Errors
///
/// `(name, ParamError)` for the first entry whose value is not an
/// admissible parameter shape.
pub fn pack_params_to_exec(
    params: &BTreeMap<String, PackValue>,
) -> Result<Parameters, (String, ParamError)> {
    let mut out = Parameters::with_capacity(params.len());
    for (name, value) in params {
        match pack_to_exec(value) {
            Ok(v) => {
                out.insert(name.clone(), v);
            }
            Err(e) => return Err((name.clone(), e)),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcgraph_core::{LabelId, NodeId, RelId, TypeId};

    #[test]
    fn primitives_bridge_directly() {
        assert_eq!(exec_to_pack(&ExecValue::Null), PackValue::Null);
        assert_eq!(
            exec_to_pack(&ExecValue::Boolean(true)),
            PackValue::Boolean(true)
        );
        assert_eq!(
            exec_to_pack(&ExecValue::Integer(42)),
            PackValue::Integer(42)
        );
        assert_eq!(exec_to_pack(&ExecValue::Float(2.5)), PackValue::Float(2.5));
        assert_eq!(
            exec_to_pack(&ExecValue::String("x".into())),
            PackValue::String("x".into())
        );
    }

    #[test]
    fn list_bridges_recursively() {
        let v = ExecValue::List(vec![ExecValue::Integer(1), ExecValue::Integer(2)]);
        assert_eq!(
            exec_to_pack(&v),
            PackValue::List(vec![PackValue::Integer(1), PackValue::Integer(2)])
        );
    }

    #[test]
    fn map_bridges_recursively() {
        // ADR-191 D-7 — an executor Map bridges to a native Bolt Map
        // (`BTreeMap` sorted-key order); values recurse via the shared
        // converter (same path the `List` arm above exercises).
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), ExecValue::Integer(1));
        m.insert("b".to_string(), ExecValue::String("x".into()));
        let v = ExecValue::Map(m);
        let mut expected = BTreeMap::new();
        expected.insert("a".to_string(), PackValue::Integer(1));
        expected.insert("b".to_string(), PackValue::String("x".into()));
        assert_eq!(exec_to_pack(&v), PackValue::Map(expected));
    }

    #[test]
    fn node_with_label_and_property_packs_to_struct() {
        // #871 — the node carries its catalog-resolved label NAME; the
        // Bolt struct's labels list MUST surface the name ("Person"),
        // never the opaque `LabelId` debug form. (Pre-#871 this asserted
        // only `labels.len() == 1`, which passed even on the buggy
        // `"LabelId(3)"` string — a weak oracle the fix strengthens.)
        let n = NodeView::new(NodeId::new(7), Some(LabelId::new(3)))
            .with_label_name("Person")
            .with_property("name", ExecValue::String("Alice".into()));
        let p = exec_to_pack_with_tenant(&ExecValue::Node(n), "abc");
        match p {
            PackValue::Struct { tag, fields } => {
                assert_eq!(tag, TAG_NODE);
                assert_eq!(fields.len(), 4);
                assert_eq!(fields[0], PackValue::Integer(7));
                // labels list == ["Person"] (the NAME, not "LabelId(3)").
                assert_eq!(
                    fields[1],
                    PackValue::List(vec![PackValue::String("Person".into())]),
                    "Bolt node labels must be the resolved name"
                );
                // element_id present + tenant-prefixed
                if let PackValue::String(eid) = &fields[3] {
                    assert!(eid.starts_with("4:abc:"));
                    assert!(eid.ends_with(":7"));
                } else {
                    panic!("element_id not a string");
                }
            }
            other => panic!("not a Struct: {other:?}"),
        }
    }

    #[test]
    fn rel_packs_to_struct_with_8_fields() {
        let r = RelView::new(
            RelId::new(1),
            NodeId::new(2),
            NodeId::new(3),
            Some(TypeId::new(1)),
        );
        let p = exec_to_pack_with_tenant(&ExecValue::Relationship(r), "x");
        match p {
            PackValue::Struct { tag, fields } => {
                assert_eq!(tag, TAG_RELATIONSHIP);
                assert_eq!(fields.len(), 8, "Bolt 5.0 Rel has 8 fields");
            }
            other => panic!("not a Struct: {other:?}"),
        }
    }

    /// ADR-193 D-8 / test 13 — a path packs to a structured Map
    /// `{start: <Node struct>, segments: [{relationship: <Rel struct>,
    /// end: <Node struct>}]}`. The nodes/rels reuse the existing Bolt
    /// struct encoders.
    #[test]
    fn path_packs_to_structured_map() {
        let path = PathView::new(NodeView::new(NodeId::new(1), Some(LabelId::new(1))))
            .with_segment(
                RelView::new(
                    RelId::new(10),
                    NodeId::new(1),
                    NodeId::new(2),
                    Some(TypeId::new(1)),
                ),
                NodeView::new(NodeId::new(2), None),
            );
        let packed = exec_to_pack_with_tenant(&ExecValue::Path(path), "t");
        match packed {
            PackValue::Map(obj) => {
                // start = a Node struct.
                assert!(
                    matches!(obj.get("start"), Some(PackValue::Struct { tag, .. }) if *tag == TAG_NODE),
                    "start packs as a Node struct"
                );
                // segments = a one-element list of {relationship, end} maps.
                match obj.get("segments") {
                    Some(PackValue::List(segs)) => {
                        assert_eq!(segs.len(), 1, "one segment");
                        match &segs[0] {
                            PackValue::Map(seg) => {
                                assert!(matches!(
                                    seg.get("relationship"),
                                    Some(PackValue::Struct { tag, .. }) if *tag == TAG_RELATIONSHIP
                                ));
                                assert!(matches!(
                                    seg.get("end"),
                                    Some(PackValue::Struct { tag, .. }) if *tag == TAG_NODE
                                ));
                            }
                            other => panic!("segment not a Map: {other:?}"),
                        }
                    }
                    other => panic!("segments not a List: {other:?}"),
                }
            }
            other => panic!("path did not pack to a Map: {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // #797 — inbound PackValue → ExecValue (`$param` binding)
    // -----------------------------------------------------------------

    #[test]
    fn pack_to_exec_round_trips_primitives() {
        assert_eq!(pack_to_exec(&PackValue::Null).unwrap(), ExecValue::Null);
        assert_eq!(
            pack_to_exec(&PackValue::Boolean(true)).unwrap(),
            ExecValue::Boolean(true)
        );
        assert_eq!(
            pack_to_exec(&PackValue::Integer(42)).unwrap(),
            ExecValue::Integer(42)
        );
        assert_eq!(
            pack_to_exec(&PackValue::Float(2.5)).unwrap(),
            ExecValue::Float(2.5)
        );
        assert_eq!(
            pack_to_exec(&PackValue::String("hi".into())).unwrap(),
            ExecValue::String("hi".into())
        );
    }

    #[test]
    fn pack_to_exec_round_trips_list_and_map_recursively() {
        // List of ints (the `$data` / `$embedding` shape).
        let list = PackValue::List(vec![PackValue::Integer(1), PackValue::Integer(2)]);
        assert_eq!(
            pack_to_exec(&list).unwrap(),
            ExecValue::List(vec![ExecValue::Integer(1), ExecValue::Integer(2)])
        );

        // Map of mixed scalars (a properties bag), with a nested list.
        let mut m = BTreeMap::new();
        m.insert("n".to_string(), PackValue::Integer(7));
        m.insert("s".to_string(), PackValue::String("x".into()));
        m.insert(
            "xs".to_string(),
            PackValue::List(vec![PackValue::Boolean(true)]),
        );
        let mut want = BTreeMap::new();
        want.insert("n".to_string(), ExecValue::Integer(7));
        want.insert("s".to_string(), ExecValue::String("x".into()));
        want.insert(
            "xs".to_string(),
            ExecValue::List(vec![ExecValue::Boolean(true)]),
        );
        assert_eq!(
            pack_to_exec(&PackValue::Map(m)).unwrap(),
            ExecValue::Map(want)
        );
    }

    #[test]
    fn pack_to_exec_rejects_node_struct() {
        // A Node struct (the `exec_to_pack` OUTPUT shape) is NOT a valid
        // parameter INPUT — round-tripping a query result back in as a
        // param must be rejected, not silently mis-bound.
        let node = PackValue::Struct {
            tag: TAG_NODE,
            fields: vec![
                PackValue::Integer(1),
                PackValue::List(vec![]),
                PackValue::Map(BTreeMap::new()),
                PackValue::String("4::1".into()),
            ],
        };
        assert_eq!(
            pack_to_exec(&node),
            Err(ParamError::UnsupportedStruct { tag: TAG_NODE })
        );
    }

    #[test]
    fn pack_to_exec_rejects_relationship_struct() {
        let rel = PackValue::Struct {
            tag: TAG_RELATIONSHIP,
            fields: vec![],
        };
        assert_eq!(
            pack_to_exec(&rel),
            Err(ParamError::UnsupportedStruct {
                tag: TAG_RELATIONSHIP
            })
        );
    }

    #[test]
    fn pack_to_exec_rejects_bytes() {
        assert_eq!(
            pack_to_exec(&PackValue::Bytes(vec![1, 2, 3])),
            Err(ParamError::UnsupportedBytes)
        );
    }

    #[test]
    fn pack_to_exec_rejects_a_struct_nested_inside_a_list() {
        // Defense-in-depth: a Node smuggled inside a list param is still
        // rejected (the recursion checks every element, not just the top).
        let nested = PackValue::List(vec![
            PackValue::Integer(1),
            PackValue::Struct {
                tag: TAG_NODE,
                fields: vec![],
            },
        ]);
        assert_eq!(
            pack_to_exec(&nested),
            Err(ParamError::UnsupportedStruct { tag: TAG_NODE })
        );
    }

    #[test]
    fn pack_to_exec_rejects_overdeep_nesting_without_stack_overflow() {
        // Adversarial: a list nested MAX_PARAM_DEPTH+2 deep surfaces a
        // clean `NestingTooDeep` rather than recursing to a stack
        // overflow (the security contract on the first untrusted
        // value-surface).
        let mut v = PackValue::Integer(0);
        for _ in 0..(MAX_PARAM_DEPTH + 2) {
            v = PackValue::List(vec![v]);
        }
        assert_eq!(
            pack_to_exec(&v),
            Err(ParamError::NestingTooDeep {
                max: MAX_PARAM_DEPTH
            })
        );
    }

    #[test]
    fn pack_params_to_exec_builds_bag_and_names_the_bad_param() {
        let mut params = BTreeMap::new();
        params.insert("x".to_string(), PackValue::Integer(42));
        params.insert("s".to_string(), PackValue::String("hi".into()));
        let bag = pack_params_to_exec(&params).expect("all scalar params convert");
        assert_eq!(bag.get("x"), Some(&ExecValue::Integer(42)));
        assert_eq!(bag.get("s"), Some(&ExecValue::String("hi".into())));

        // One bad entry → the offending NAME is surfaced for a precise
        // client error.
        let mut bad = BTreeMap::new();
        bad.insert(
            "qv".to_string(),
            PackValue::Struct {
                tag: TAG_NODE,
                fields: vec![],
            },
        );
        let err = pack_params_to_exec(&bad).expect_err("node param rejected");
        assert_eq!(err.0, "qv");
        assert_eq!(err.1, ParamError::UnsupportedStruct { tag: TAG_NODE });
    }
}
