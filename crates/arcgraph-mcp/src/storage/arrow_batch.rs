//! v2 M2 — Arrow `RecordBatch` batch materialization (design §M2.4;
//! ADR-230 row M2 "Arrow batch materialization").
//!
//! Column-wise materialization of a projected node scan straight from
//! the typed property payloads: per projected property, a typed Arrow
//! builder (`Int64Builder` / `Float64Builder` / `BooleanBuilder` /
//! `StringBuilder`) appends directly from the zero-decode reads — no
//! intermediate row objects, no JSON on the read path. On-disk stays
//! row-shaped typed blocks (design §M2.4 "What Arrow is NOT"); Arrow
//! is the in-memory batch IR only.
//!
//! This uses the workspace's existing Arrow 58 dependency and license
//! surface.
//!
//! # Column typing (schemaless properties → typed columns)
//!
//! A property key has no declared type in a schemaless graph, but an
//! Arrow column needs exactly one `DataType`. The resolution is a
//! deterministic, total, two-pass unification per column:
//!
//! 1. Pass 1 inspects each row's value for the key (typed lookups —
//!    scalar tags are read from the block header + payload, no
//!    allocation) and folds a column type over the lattice:
//!    `Int64 ⊔ Int64 = Int64`; `Int64 ⊔ Float64 = Float64` (the JSON
//!    number-widening the engine already applies); `Boolean ⊔ Boolean
//!    = Boolean`; `Utf8 ⊔ Utf8 = Utf8`; anything else — a mixed-type
//!    column, or a list/map-valued property — unifies to `Utf8` in the
//!    value's canonical JSON wire form (the same form the MCP wire
//!    surfaces render, so nothing is invented and nothing is dropped).
//!    Missing keys / nulls are Arrow nulls (every column nullable).
//! 2. Pass 2 appends through the unified builders.
//!
//! Deterministic + total: the same scan always produces the same
//! schema; heterogeneous data NEVER errors and NEVER silently drops a
//! value (`feedback_noop_trampoline_anti_pattern` — the fallback is
//! the documented JSON wire form, not a null hole).
//!
//! # Consumers at M2 (stated honestly)
//!
//! The committed consumers are the columnar-equivalence integration
//! test (row-scan vs RecordBatch, value-identical) and the Criterion
//! materialization bench. The PRODUCTION consumers the design names —
//! the §4.3 vectorized executor and the Arrow IPC ecosystem export —
//! land at their own milestones (M4-64b; the export needs an MCP
//! surface, ADR-004-gated). This module is the §M2.4 substrate they
//! consume, wired to the real typed read path now so the format chain
//! (M3 deltas carry final-representation blocks) does not re-cut it.
//!
//! # Budget (PD#5)
//!
//! Two passes over `rows × |projection|` typed lookups; pass 1 touches
//! only block headers + scalar payloads (no string copies), pass 2
//! copies each materialized value ONCE into its Arrow buffer. Peak
//! extra memory = the RecordBatch itself + one `Vec<BTreeMap>` of the
//! projected bags (bounded by the caller's scan range).

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_query::executor::substrate::SubstrateAccessError;
use arcgraph_query::executor::value::Value;
use arrow_array::builder::{BooleanBuilder, Float64Builder, Int64Builder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};

/// Per-column unified Arrow type (the pass-1 lattice).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColType {
    /// No non-null value seen yet.
    Unseen,
    Int64,
    Float64,
    Boolean,
    Utf8,
    /// Mixed/nested — canonical JSON wire form as Utf8.
    JsonUtf8,
}

impl ColType {
    fn join(self, v: &Value) -> ColType {
        use ColType as C;
        let vt = match v {
            Value::Null => return self,
            Value::Integer(_) => C::Int64,
            Value::Float(_) => C::Float64,
            Value::Boolean(_) => C::Boolean,
            Value::String(_) => C::Utf8,
            _ => C::JsonUtf8,
        };
        match (self, vt) {
            (C::Unseen, t) => t,
            (a, b) if a == b => a,
            // The engine's own number widening.
            (C::Int64, C::Float64) | (C::Float64, C::Int64) => C::Float64,
            // Any other mix → canonical JSON text.
            _ => C::JsonUtf8,
        }
    }
}

/// Materialize projected node bags into one Arrow [`RecordBatch`].
///
/// Schema: `node_id: UInt64 (non-null)` followed by one nullable
/// column per projected property name (input order preserved,
/// duplicates removed). `rows` are `(node_id, projected bag)` pairs —
/// the projected bags come from the typed zero-decode read
/// ([`crate::storage::property_payload::record_property_bag_projected`]),
/// so this composes with the scan exactly as design §M2.4 draws it.
pub fn projected_rows_to_record_batch(
    projected: &[String],
    rows: &[(u64, BTreeMap<String, Value>)],
) -> Result<RecordBatch, SubstrateAccessError> {
    let mut names: Vec<&str> = Vec::with_capacity(projected.len());
    for n in projected {
        if !names.contains(&n.as_str()) {
            names.push(n.as_str());
        }
    }

    // Pass 1 — unify each column's type.
    let mut col_types = vec![ColType::Unseen; names.len()];
    for (_, bag) in rows {
        for (i, name) in names.iter().enumerate() {
            if let Some(v) = bag.get(*name) {
                col_types[i] = col_types[i].join(v);
            }
        }
    }

    // Pass 2 — build the columns.
    let mut fields: Vec<Field> = Vec::with_capacity(1 + names.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(1 + names.len());
    fields.push(Field::new("node_id", DataType::UInt64, false));
    arrays.push(Arc::new(UInt64Array::from(
        rows.iter().map(|(id, _)| *id).collect::<Vec<u64>>(),
    )) as ArrayRef);

    for (i, name) in names.iter().enumerate() {
        let (dt, array): (DataType, ArrayRef) = match col_types[i] {
            ColType::Int64 => {
                let mut b = Int64Builder::with_capacity(rows.len());
                for (_, bag) in rows {
                    match bag.get(*name) {
                        Some(Value::Integer(v)) => b.append_value(*v),
                        _ => b.append_null(),
                    }
                }
                (DataType::Int64, Arc::new(b.finish()))
            }
            ColType::Float64 => {
                let mut b = Float64Builder::with_capacity(rows.len());
                for (_, bag) in rows {
                    match bag.get(*name) {
                        Some(Value::Integer(v)) => b.append_value(*v as f64),
                        Some(Value::Float(v)) => b.append_value(*v),
                        _ => b.append_null(),
                    }
                }
                (DataType::Float64, Arc::new(b.finish()))
            }
            ColType::Boolean => {
                let mut b = BooleanBuilder::with_capacity(rows.len());
                for (_, bag) in rows {
                    match bag.get(*name) {
                        Some(Value::Boolean(v)) => b.append_value(*v),
                        _ => b.append_null(),
                    }
                }
                (DataType::Boolean, Arc::new(b.finish()))
            }
            ColType::Utf8 => {
                let mut b = StringBuilder::new();
                for (_, bag) in rows {
                    match bag.get(*name) {
                        Some(Value::String(s)) => b.append_value(s),
                        _ => b.append_null(),
                    }
                }
                (DataType::Utf8, Arc::new(b.finish()))
            }
            // Mixed / nested → canonical JSON wire form. All-null /
            // never-seen columns also land here (nullable Utf8 of
            // nothing — a stable, queryable shape).
            ColType::JsonUtf8 | ColType::Unseen => {
                let mut b = StringBuilder::new();
                for (_, bag) in rows {
                    match bag.get(*name) {
                        Some(Value::Null) | None => b.append_null(),
                        Some(v) => b.append_value(v.to_json_value().to_string()),
                    }
                }
                (DataType::Utf8, Arc::new(b.finish()))
            }
        };
        fields.push(Field::new((*name).to_string(), dt, true));
        arrays.push(array);
    }

    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|e| SubstrateAccessError::Io(format!("arrow RecordBatch assembly failed: {e}")))
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, BooleanArray, Float64Array, Int64Array, StringArray};

    use super::*;

    fn bag(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn homogeneous_columns_type_natively() {
        let rows = vec![
            (
                1u64,
                bag(&[("n", Value::Integer(10)), ("s", Value::String("a".into()))]),
            ),
            (2u64, bag(&[("n", Value::Integer(20))])),
            (
                3u64,
                bag(&[("n", Value::Null), ("s", Value::String("c".into()))]),
            ),
        ];
        let proj = vec!["n".to_string(), "s".to_string()];
        let batch = projected_rows_to_record_batch(&proj, &rows).expect("batch");
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.schema().field(1).data_type(), &DataType::Int64);
        assert_eq!(batch.schema().field(2).data_type(), &DataType::Utf8);
        let n = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("i64 col");
        assert_eq!(n.value(0), 10);
        assert_eq!(n.value(1), 20);
        assert!(n.is_null(2), "Null value is an Arrow null");
        let s = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("utf8 col");
        assert_eq!(s.value(0), "a");
        assert!(s.is_null(1), "missing key is an Arrow null");
    }

    #[test]
    fn int_float_mix_widens_to_float64() {
        let rows = vec![
            (1u64, bag(&[("x", Value::Integer(1))])),
            (2u64, bag(&[("x", Value::Float(2.5))])),
        ];
        let batch = projected_rows_to_record_batch(&["x".to_string()], &rows).expect("batch");
        assert_eq!(batch.schema().field(1).data_type(), &DataType::Float64);
        let x = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("f64 col");
        assert!((x.value(0) - 1.0).abs() < f64::EPSILON);
        assert!((x.value(1) - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn heterogeneous_and_nested_fall_back_to_canonical_json_text_never_dropped() {
        let rows = vec![
            (1u64, bag(&[("m", Value::Integer(7))])),
            (2u64, bag(&[("m", Value::String("x".into()))])),
            (
                3u64,
                bag(&[("m", Value::List(vec![Value::Integer(1), Value::Integer(2)]))]),
            ),
        ];
        let batch = projected_rows_to_record_batch(&["m".to_string()], &rows).expect("batch");
        assert_eq!(batch.schema().field(1).data_type(), &DataType::Utf8);
        let m = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("utf8 col");
        assert_eq!(m.value(0), "7");
        assert_eq!(m.value(1), "\"x\"");
        assert_eq!(m.value(2), "[1,2]");
    }

    #[test]
    fn bool_column_and_node_ids() {
        let rows = vec![
            (41u64, bag(&[("ok", Value::Boolean(true))])),
            (42u64, bag(&[("ok", Value::Boolean(false))])),
        ];
        let batch = projected_rows_to_record_batch(&["ok".to_string()], &rows).expect("batch");
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("id col");
        assert_eq!((ids.value(0), ids.value(1)), (41, 42));
        let ok = batch
            .column(1)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("bool col");
        assert!(ok.value(0));
        assert!(!ok.value(1));
    }

    #[test]
    fn duplicate_projection_names_collapse() {
        let rows = vec![(1u64, bag(&[("a", Value::Integer(1))]))];
        let proj = vec!["a".to_string(), "a".to_string()];
        let batch = projected_rows_to_record_batch(&proj, &rows).expect("batch");
        assert_eq!(batch.num_columns(), 2, "node_id + one deduped column");
    }
}
