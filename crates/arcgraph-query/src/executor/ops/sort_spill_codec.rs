//! Binary spill codec for external-sort records.
//!
//! The spill layer deliberately treats a run payload as opaque bytes.  This
//! module owns the query executor's byte representation and keeps it separate
//! from spill framing, encryption, and quota accounting.  The format is
//! little-endian and versioned by [`MAGIC`]; it is an internal scratch format,
//! not a durable/on-wire compatibility promise.

use std::collections::BTreeMap;
use std::fmt;

use arcgraph_core::{
    Date, Decimal, Duration, LabelId, LocalDateTime, NodeId, RelId, TypeId, ZonedDateTime,
};

use crate::executor::value::{
    MAX_JSON_DECODE_DEPTH, NodeView, PathSegment, PathView, RelView, Value,
};

/// One row in a sorted spill run.
///
/// `ordinal` is the input sequence number used to retain stable-sort order
/// across independently generated runs. `keys` is stored alongside `row` so a
/// merge never needs to re-evaluate expressions after spilling.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SortSpillRecord {
    pub ordinal: u64,
    pub keys: Vec<Value>,
    pub row: Vec<Value>,
}

/// A defensive codec failure. All corrupt-input outcomes are typed; decoding
/// never indexes input unchecked or allocates from an unvalidated length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SortSpillCodecError {
    InvalidMagic,
    Truncated {
        needed: usize,
        remaining: usize,
    },
    TrailingBytes {
        remaining: usize,
    },
    InvalidTag(u8),
    InvalidBoolean(u8),
    InvalidOptionMarker(u8),
    InvalidUtf8,
    InvalidLength {
        kind: &'static str,
        count: usize,
        remaining: usize,
    },
    LengthOverflow {
        kind: &'static str,
        len: usize,
    },
    NestingTooDeep {
        depth: usize,
        max: usize,
    },
    DuplicateMapKey,
    InvalidTemporal {
        kind: &'static str,
    },
    AllocationFailed {
        kind: &'static str,
        count: usize,
    },
}

impl fmt::Display for SortSpillCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => f.write_str("invalid external-sort spill magic/version"),
            Self::Truncated { needed, remaining } => write!(
                f,
                "truncated external-sort spill: need {needed} bytes, have {remaining}"
            ),
            Self::TrailingBytes { remaining } => {
                write!(f, "external-sort spill has {remaining} trailing bytes")
            }
            Self::InvalidTag(tag) => write!(f, "invalid external-sort Value tag {tag}"),
            Self::InvalidBoolean(value) => {
                write!(f, "invalid external-sort boolean byte {value}")
            }
            Self::InvalidOptionMarker(value) => {
                write!(f, "invalid external-sort option marker {value}")
            }
            Self::InvalidUtf8 => f.write_str("external-sort spill contains invalid UTF-8"),
            Self::InvalidLength {
                kind,
                count,
                remaining,
            } => write!(
                f,
                "invalid external-sort {kind} length {count} for {remaining} remaining bytes"
            ),
            Self::LengthOverflow { kind, len } => {
                write!(
                    f,
                    "external-sort {kind} length {len} exceeds the format limit"
                )
            }
            Self::NestingTooDeep { depth, max } => write!(
                f,
                "external-sort Value nesting depth {depth} exceeds maximum {max}"
            ),
            Self::DuplicateMapKey => {
                f.write_str("external-sort spill map contains a duplicate key")
            }
            Self::InvalidTemporal { kind } => {
                write!(f, "external-sort spill contains an invalid {kind}")
            }
            Self::AllocationFailed { kind, count } => write!(
                f,
                "could not allocate external-sort {kind} capacity for {count} items"
            ),
        }
    }
}

impl std::error::Error for SortSpillCodecError {}

const MAGIC: &[u8; 8] = b"AGSORT01";
const JOIN_MAGIC: &[u8; 8] = b"AGJOIN01";
const EXPAND_MAGIC: &[u8; 8] = b"AGEXPD01";

const TAG_NULL: u8 = 0;
const TAG_BOOLEAN: u8 = 1;
const TAG_INTEGER: u8 = 2;
const TAG_FLOAT: u8 = 3;
const TAG_STRING: u8 = 4;
const TAG_NODE: u8 = 5;
const TAG_RELATIONSHIP: u8 = 6;
const TAG_LIST: u8 = 7;
const TAG_MAP: u8 = 8;
const TAG_PATH: u8 = 9;
const TAG_TEMPORAL: u8 = 10;
const TAG_LOCAL_DATE_TIME: u8 = 11;
const TAG_DATE: u8 = 12;
const TAG_DURATION: u8 = 13;
const TAG_DECIMAL: u8 = 14;

// Never eagerly reserve an entire attacker-provided collection count. A valid
// large collection grows fallibly as elements are successfully decoded; a
// malformed first element therefore cannot induce a large speculative heap
// allocation merely by advertising a plausible count.
const MAX_EAGER_DECODE_ITEMS: usize = 256;

/// Encode a complete spill batch. Each invocation is self-contained, which
/// lets a run reader decode batches independently while streaming a merge.
pub(super) fn encode_records(records: &[SortSpillRecord]) -> Result<Vec<u8>, SortSpillCodecError> {
    let mut encoder = Encoder::new();
    encoder.write_bytes(MAGIC, "header")?;
    encoder.write_len(records.len(), "record count")?;
    for record in records {
        encoder.write_u64(record.ordinal)?;
        encoder.write_values(&record.keys, 0, "sort-key count")?;
        encoder.write_values(&record.row, 0, "row-value count")?;
    }
    Ok(encoder.bytes)
}

/// Decode exactly one spill batch. Trailing bytes are rejected so format
/// drift, concatenation mistakes, and authenticated-but-malformed plaintext
/// cannot silently alter sort results.
pub(super) fn decode_records(bytes: &[u8]) -> Result<Vec<SortSpillRecord>, SortSpillCodecError> {
    let mut decoder = Decoder::new(bytes);
    if decoder.read_exact(MAGIC.len())? != MAGIC {
        return Err(SortSpillCodecError::InvalidMagic);
    }

    // The minimum record is ordinal + two empty-vector lengths. Checking this
    // before reserving ties the maximum allocation directly to input bytes.
    let count = decoder.read_count("record count", 16)?;
    let mut records = try_vec_with_capacity(count, "record vector")?;
    for _ in 0..count {
        let record = SortSpillRecord {
            ordinal: decoder.read_u64()?,
            keys: decoder.read_values(0, "sort-key count")?,
            row: decoder.read_values(0, "row-value count")?,
        };
        try_push(&mut records, record, "record vector")?;
    }
    let remaining = decoder.remaining();
    if remaining != 0 {
        return Err(SortSpillCodecError::TrailingBytes { remaining });
    }
    Ok(records)
}

/// Encode rows for the Grace hash-join partition runs.
///
/// The Value codec is deliberately shared with external sort so every
/// executor spill consumer has one defensive representation for the full
/// Value lattice. Join frames have their own magic/version and contain no
/// sort ordinal or evaluated-key payload.
pub(super) fn encode_join_rows(rows: &[Vec<Value>]) -> Result<Vec<u8>, SortSpillCodecError> {
    let mut encoder = Encoder::new();
    encoder.write_bytes(JOIN_MAGIC, "header")?;
    encoder.write_len(rows.len(), "join row count")?;
    for row in rows {
        encoder.write_values(row, 0, "join row value count")?;
    }
    Ok(encoder.bytes)
}

/// Decode exactly one Grace hash-join partition frame.
pub(super) fn decode_join_rows(bytes: &[u8]) -> Result<Vec<Vec<Value>>, SortSpillCodecError> {
    let mut decoder = Decoder::new(bytes);
    if decoder.read_exact(JOIN_MAGIC.len())? != JOIN_MAGIC {
        return Err(SortSpillCodecError::InvalidMagic);
    }
    // Every row carries at least its value-count u32. Tying the advertised
    // count to remaining authenticated bytes prevents speculative allocation.
    let count = decoder.read_count("join row count", 4)?;
    let mut rows = try_vec_with_capacity(count, "join row vector")?;
    for _ in 0..count {
        let row = decoder.read_values(0, "join row value count")?;
        try_push(&mut rows, row, "join row vector")?;
    }
    let remaining = decoder.remaining();
    if remaining != 0 {
        return Err(SortSpillCodecError::TrailingBytes { remaining });
    }
    Ok(rows)
}

/// Encode one FIFO expand-frontier chunk.
///
/// Expand and Grace join share the defensive Value lattice codec, but use
/// distinct magic values so a run routed to the wrong operator fails loudly
/// instead of being accepted as a plausible row batch.
pub(super) fn encode_expand_rows(rows: &[Vec<Value>]) -> Result<Vec<u8>, SortSpillCodecError> {
    encode_rows_with_magic(
        rows,
        EXPAND_MAGIC,
        "expand row count",
        "expand row value count",
    )
}

/// Decode exactly one FIFO expand-frontier chunk.
pub(super) fn decode_expand_rows(bytes: &[u8]) -> Result<Vec<Vec<Value>>, SortSpillCodecError> {
    decode_rows_with_magic(
        bytes,
        EXPAND_MAGIC,
        "expand row count",
        "expand row vector",
        "expand row value count",
    )
}

fn encode_rows_with_magic(
    rows: &[Vec<Value>],
    magic: &[u8; 8],
    count_kind: &'static str,
    value_count_kind: &'static str,
) -> Result<Vec<u8>, SortSpillCodecError> {
    let mut encoder = Encoder::new();
    encoder.write_bytes(magic, "header")?;
    encoder.write_len(rows.len(), count_kind)?;
    for row in rows {
        encoder.write_values(row, 0, value_count_kind)?;
    }
    Ok(encoder.bytes)
}

fn decode_rows_with_magic(
    bytes: &[u8],
    magic: &[u8; 8],
    count_kind: &'static str,
    vector_kind: &'static str,
    value_count_kind: &'static str,
) -> Result<Vec<Vec<Value>>, SortSpillCodecError> {
    let mut decoder = Decoder::new(bytes);
    if decoder.read_exact(magic.len())? != magic {
        return Err(SortSpillCodecError::InvalidMagic);
    }
    let count = decoder.read_count(count_kind, 4)?;
    let mut rows = try_vec_with_capacity(count, vector_kind)?;
    for _ in 0..count {
        let row = decoder.read_values(0, value_count_kind)?;
        try_push(&mut rows, row, vector_kind)?;
    }
    let remaining = decoder.remaining();
    if remaining != 0 {
        return Err(SortSpillCodecError::TrailingBytes { remaining });
    }
    Ok(rows)
}

struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn reserve(
        &mut self,
        additional: usize,
        kind: &'static str,
    ) -> Result<(), SortSpillCodecError> {
        self.bytes
            .len()
            .checked_add(additional)
            .ok_or(SortSpillCodecError::LengthOverflow {
                kind,
                len: additional,
            })?;
        self.bytes
            .try_reserve(additional)
            .map_err(|_| SortSpillCodecError::AllocationFailed {
                kind,
                count: additional,
            })
    }

    fn write_bytes(&mut self, bytes: &[u8], kind: &'static str) -> Result<(), SortSpillCodecError> {
        self.reserve(bytes.len(), kind)?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn write_u8(&mut self, value: u8) -> Result<(), SortSpillCodecError> {
        self.reserve(1, "scalar")?;
        self.bytes.push(value);
        Ok(())
    }

    fn write_i8(&mut self, value: i8) -> Result<(), SortSpillCodecError> {
        self.write_u8(value as u8)
    }

    fn write_u16(&mut self, value: u16) -> Result<(), SortSpillCodecError> {
        self.write_bytes(&value.to_le_bytes(), "scalar")
    }

    fn write_u32(&mut self, value: u32) -> Result<(), SortSpillCodecError> {
        self.write_bytes(&value.to_le_bytes(), "scalar")
    }

    fn write_i32(&mut self, value: i32) -> Result<(), SortSpillCodecError> {
        self.write_bytes(&value.to_le_bytes(), "scalar")
    }

    fn write_u64(&mut self, value: u64) -> Result<(), SortSpillCodecError> {
        self.write_bytes(&value.to_le_bytes(), "scalar")
    }

    fn write_i64(&mut self, value: i64) -> Result<(), SortSpillCodecError> {
        self.write_bytes(&value.to_le_bytes(), "scalar")
    }

    fn write_i128(&mut self, value: i128) -> Result<(), SortSpillCodecError> {
        self.write_bytes(&value.to_le_bytes(), "scalar")
    }

    fn write_len(&mut self, len: usize, kind: &'static str) -> Result<(), SortSpillCodecError> {
        let len =
            u32::try_from(len).map_err(|_| SortSpillCodecError::LengthOverflow { kind, len })?;
        self.write_u32(len)
    }

    fn write_string(&mut self, value: &str) -> Result<(), SortSpillCodecError> {
        self.write_len(value.len(), "string")?;
        self.write_bytes(value.as_bytes(), "string")
    }

    fn write_optional_u32(&mut self, value: Option<u32>) -> Result<(), SortSpillCodecError> {
        match value {
            None => self.write_u8(0),
            Some(value) => {
                self.write_u8(1)?;
                self.write_u32(value)
            }
        }
    }

    fn write_optional_string(&mut self, value: Option<&str>) -> Result<(), SortSpillCodecError> {
        match value {
            None => self.write_u8(0),
            Some(value) => {
                self.write_u8(1)?;
                self.write_string(value)
            }
        }
    }

    fn write_values(
        &mut self,
        values: &[Value],
        depth: usize,
        kind: &'static str,
    ) -> Result<(), SortSpillCodecError> {
        self.write_len(values.len(), kind)?;
        for value in values {
            self.write_value(value, depth)?;
        }
        Ok(())
    }

    fn write_map(
        &mut self,
        map: &BTreeMap<String, Value>,
        parent_depth: usize,
    ) -> Result<(), SortSpillCodecError> {
        self.write_len(map.len(), "map entry count")?;
        for (key, value) in map {
            self.write_string(key)?;
            self.write_value(value, next_depth(parent_depth)?)?;
        }
        Ok(())
    }

    fn write_node(&mut self, node: &NodeView, depth: usize) -> Result<(), SortSpillCodecError> {
        self.write_u64(node.id.raw())?;
        self.write_optional_u32(node.label.map(LabelId::raw))?;
        self.write_optional_string(node.label_name.as_deref())?;
        self.write_map(&node.properties, depth)
    }

    fn write_rel(&mut self, rel: &RelView, depth: usize) -> Result<(), SortSpillCodecError> {
        self.write_u64(rel.id.raw())?;
        self.write_u64(rel.from.raw())?;
        self.write_u64(rel.to.raw())?;
        self.write_optional_u32(rel.rel_type.map(TypeId::raw))?;
        self.write_optional_string(rel.rel_type_name.as_deref())?;
        self.write_map(&rel.properties, depth)
    }

    fn write_value(&mut self, value: &Value, depth: usize) -> Result<(), SortSpillCodecError> {
        check_depth(depth)?;
        match value {
            Value::Null => self.write_u8(TAG_NULL),
            Value::Boolean(value) => {
                self.write_u8(TAG_BOOLEAN)?;
                self.write_u8(u8::from(*value))
            }
            Value::Integer(value) => {
                self.write_u8(TAG_INTEGER)?;
                self.write_i64(*value)
            }
            Value::Float(value) => {
                self.write_u8(TAG_FLOAT)?;
                self.write_u64(value.to_bits())
            }
            Value::String(value) => {
                self.write_u8(TAG_STRING)?;
                self.write_string(value)
            }
            Value::Node(node) => {
                self.write_u8(TAG_NODE)?;
                self.write_node(node, depth)
            }
            Value::Relationship(rel) => {
                self.write_u8(TAG_RELATIONSHIP)?;
                self.write_rel(rel, depth)
            }
            Value::List(values) => {
                self.write_u8(TAG_LIST)?;
                self.write_len(values.len(), "list item count")?;
                for value in values {
                    self.write_value(value, next_depth(depth)?)?;
                }
                Ok(())
            }
            Value::Map(map) => {
                self.write_u8(TAG_MAP)?;
                self.write_map(map, depth)
            }
            Value::Path(path) => {
                self.write_u8(TAG_PATH)?;
                let child_depth = next_depth(depth)?;
                self.write_node(&path.start, child_depth)?;
                self.write_len(path.segments.len(), "path segment count")?;
                for segment in &path.segments {
                    self.write_rel(&segment.rel, child_depth)?;
                    self.write_node(&segment.end, child_depth)?;
                }
                Ok(())
            }
            Value::Temporal(value) => {
                self.write_u8(TAG_TEMPORAL)?;
                self.write_i64(value.utc_nanos())?;
                self.write_i32(value.offset_seconds())
            }
            Value::LocalDateTime(value) => {
                self.write_u8(TAG_LOCAL_DATE_TIME)?;
                self.write_i32(value.year)?;
                self.write_u16(value.ordinal)?;
                self.write_u64(value.nano_of_day)
            }
            Value::Date(value) => {
                self.write_u8(TAG_DATE)?;
                self.write_i32(value.year)?;
                self.write_u16(value.ordinal)
            }
            Value::Duration(value) => {
                self.write_u8(TAG_DURATION)?;
                self.write_i32(value.months)?;
                self.write_i64(value.nanos)
            }
            Value::Decimal(value) => {
                self.write_u8(TAG_DECIMAL)?;
                self.write_i8(value.scale)?;
                self.write_i128(value.units)
            }
        }
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], SortSpillCodecError> {
        let remaining = self.remaining();
        let end = self
            .pos
            .checked_add(len)
            .ok_or(SortSpillCodecError::Truncated {
                needed: len,
                remaining,
            })?;
        if end > self.bytes.len() {
            return Err(SortSpillCodecError::Truncated {
                needed: len,
                remaining,
            });
        }
        let start = self.pos;
        self.pos = end;
        Ok(&self.bytes[start..end])
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], SortSpillCodecError> {
        let mut array = [0_u8; N];
        array.copy_from_slice(self.read_exact(N)?);
        Ok(array)
    }

    fn read_u8(&mut self) -> Result<u8, SortSpillCodecError> {
        Ok(self.read_array::<1>()?[0])
    }

    fn read_i8(&mut self) -> Result<i8, SortSpillCodecError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, SortSpillCodecError> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    fn read_u32(&mut self) -> Result<u32, SortSpillCodecError> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_i32(&mut self) -> Result<i32, SortSpillCodecError> {
        Ok(i32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64, SortSpillCodecError> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn read_i64(&mut self) -> Result<i64, SortSpillCodecError> {
        Ok(i64::from_le_bytes(self.read_array()?))
    }

    fn read_i128(&mut self) -> Result<i128, SortSpillCodecError> {
        Ok(i128::from_le_bytes(self.read_array()?))
    }

    /// Read a u32 count and prove that at least `minimum_item_bytes` per
    /// element remains before any allocation. Every recursive Value has at
    /// least a one-byte tag, so nested counts remain bounded by input length.
    fn read_count(
        &mut self,
        kind: &'static str,
        minimum_item_bytes: usize,
    ) -> Result<usize, SortSpillCodecError> {
        let count = self.read_u32()? as usize;
        let remaining = self.remaining();
        let minimum =
            count
                .checked_mul(minimum_item_bytes)
                .ok_or(SortSpillCodecError::InvalidLength {
                    kind,
                    count,
                    remaining,
                })?;
        if minimum > remaining {
            return Err(SortSpillCodecError::InvalidLength {
                kind,
                count,
                remaining,
            });
        }
        Ok(count)
    }

    fn read_string(&mut self) -> Result<String, SortSpillCodecError> {
        let len = self.read_u32()? as usize;
        let remaining = self.remaining();
        if len > remaining {
            return Err(SortSpillCodecError::InvalidLength {
                kind: "string",
                count: len,
                remaining,
            });
        }
        let bytes = self.read_exact(len)?;
        let value = std::str::from_utf8(bytes).map_err(|_| SortSpillCodecError::InvalidUtf8)?;
        let mut owned = String::new();
        owned
            .try_reserve_exact(len)
            .map_err(|_| SortSpillCodecError::AllocationFailed {
                kind: "string",
                count: len,
            })?;
        owned.push_str(value);
        Ok(owned)
    }

    fn read_optional_u32(&mut self) -> Result<Option<u32>, SortSpillCodecError> {
        match self.read_u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.read_u32()?)),
            value => Err(SortSpillCodecError::InvalidOptionMarker(value)),
        }
    }

    fn read_optional_string(&mut self) -> Result<Option<String>, SortSpillCodecError> {
        match self.read_u8()? {
            0 => Ok(None),
            1 => self.read_string().map(Some),
            value => Err(SortSpillCodecError::InvalidOptionMarker(value)),
        }
    }

    fn read_values(
        &mut self,
        depth: usize,
        kind: &'static str,
    ) -> Result<Vec<Value>, SortSpillCodecError> {
        let count = self.read_count(kind, 1)?;
        let mut values = try_vec_with_capacity(count, "Value vector")?;
        for _ in 0..count {
            let value = self.read_value(depth)?;
            try_push(&mut values, value, "Value vector")?;
        }
        Ok(values)
    }

    fn read_map(
        &mut self,
        parent_depth: usize,
    ) -> Result<BTreeMap<String, Value>, SortSpillCodecError> {
        // Four bytes for the key length plus one byte for the Value tag.
        let count = self.read_count("map entry count", 5)?;
        let mut map = BTreeMap::new();
        for _ in 0..count {
            let key = self.read_string()?;
            let value = self.read_value(next_depth(parent_depth)?)?;
            if map.insert(key, value).is_some() {
                return Err(SortSpillCodecError::DuplicateMapKey);
            }
        }
        Ok(map)
    }

    fn read_node(&mut self, depth: usize) -> Result<NodeView, SortSpillCodecError> {
        Ok(NodeView {
            id: NodeId::new(self.read_u64()?),
            label: self.read_optional_u32()?.map(LabelId::new),
            label_name: self.read_optional_string()?,
            properties: self.read_map(depth)?,
        })
    }

    fn read_rel(&mut self, depth: usize) -> Result<RelView, SortSpillCodecError> {
        Ok(RelView {
            id: RelId::new(self.read_u64()?),
            from: NodeId::new(self.read_u64()?),
            to: NodeId::new(self.read_u64()?),
            rel_type: self.read_optional_u32()?.map(TypeId::new),
            rel_type_name: self.read_optional_string()?,
            properties: self.read_map(depth)?,
        })
    }

    fn read_value(&mut self, depth: usize) -> Result<Value, SortSpillCodecError> {
        check_depth(depth)?;
        match self.read_u8()? {
            TAG_NULL => Ok(Value::Null),
            TAG_BOOLEAN => match self.read_u8()? {
                0 => Ok(Value::Boolean(false)),
                1 => Ok(Value::Boolean(true)),
                value => Err(SortSpillCodecError::InvalidBoolean(value)),
            },
            TAG_INTEGER => self.read_i64().map(Value::Integer),
            TAG_FLOAT => self
                .read_u64()
                .map(|bits| Value::Float(f64::from_bits(bits))),
            TAG_STRING => self.read_string().map(Value::String),
            TAG_NODE => self.read_node(depth).map(Value::Node),
            TAG_RELATIONSHIP => self.read_rel(depth).map(Value::Relationship),
            TAG_LIST => {
                let count = self.read_count("list item count", 1)?;
                let mut values = try_vec_with_capacity(count, "Value vector")?;
                for _ in 0..count {
                    let value = self.read_value(next_depth(depth)?)?;
                    try_push(&mut values, value, "Value vector")?;
                }
                Ok(Value::List(values))
            }
            TAG_MAP => self.read_map(depth).map(Value::Map),
            TAG_PATH => {
                let child_depth = next_depth(depth)?;
                let start = self.read_node(child_depth)?;
                // A segment contains one minimally encoded relationship (30
                // bytes) and one minimally encoded node (14 bytes).
                let count = self.read_count("path segment count", 44)?;
                let mut segments = try_vec_with_capacity(count, "path segment vector")?;
                for _ in 0..count {
                    let segment = PathSegment {
                        rel: self.read_rel(child_depth)?,
                        end: self.read_node(child_depth)?,
                    };
                    try_push(&mut segments, segment, "path segment vector")?;
                }
                Ok(Value::Path(PathView { start, segments }))
            }
            TAG_TEMPORAL => {
                let utc_nanos = self.read_i64()?;
                let offset_seconds = self.read_i32()?;
                ZonedDateTime::from_utc_nanos_and_offset(utc_nanos, offset_seconds)
                    .map(Value::Temporal)
                    .map_err(|_| SortSpillCodecError::InvalidTemporal {
                        kind: "zoned date-time",
                    })
            }
            TAG_LOCAL_DATE_TIME => {
                let year = self.read_i32()?;
                let ordinal = self.read_u16()?;
                let nano_of_day = self.read_u64()?;
                LocalDateTime::new(year, ordinal, nano_of_day)
                    .map(Value::LocalDateTime)
                    .map_err(|_| SortSpillCodecError::InvalidTemporal {
                        kind: "local date-time",
                    })
            }
            TAG_DATE => {
                let year = self.read_i32()?;
                let ordinal = self.read_u16()?;
                Date::new(year, ordinal)
                    .map(Value::Date)
                    .map_err(|_| SortSpillCodecError::InvalidTemporal { kind: "date" })
            }
            TAG_DURATION => Ok(Value::Duration(Duration::new(
                self.read_i32()?,
                self.read_i64()?,
            ))),
            TAG_DECIMAL => {
                let scale = self.read_i8()?;
                let units = self.read_i128()?;
                Decimal::new(scale, units)
                    .map(Value::Decimal)
                    .map_err(|_| SortSpillCodecError::InvalidTemporal { kind: "decimal" })
            }
            tag => Err(SortSpillCodecError::InvalidTag(tag)),
        }
    }
}

fn check_depth(depth: usize) -> Result<(), SortSpillCodecError> {
    if depth > MAX_JSON_DECODE_DEPTH {
        return Err(SortSpillCodecError::NestingTooDeep {
            depth,
            max: MAX_JSON_DECODE_DEPTH,
        });
    }
    Ok(())
}

fn next_depth(depth: usize) -> Result<usize, SortSpillCodecError> {
    let depth = depth
        .checked_add(1)
        .ok_or(SortSpillCodecError::NestingTooDeep {
            depth,
            max: MAX_JSON_DECODE_DEPTH,
        })?;
    check_depth(depth)?;
    Ok(depth)
}

fn try_vec_with_capacity<T>(
    count: usize,
    kind: &'static str,
) -> Result<Vec<T>, SortSpillCodecError> {
    let eager_count = count.min(MAX_EAGER_DECODE_ITEMS);
    let mut values = Vec::new();
    values
        .try_reserve_exact(eager_count)
        .map_err(|_| SortSpillCodecError::AllocationFailed {
            kind,
            count: eager_count,
        })?;
    Ok(values)
}

fn try_push<T>(
    values: &mut Vec<T>,
    value: T,
    kind: &'static str,
) -> Result<(), SortSpillCodecError> {
    let count = values
        .len()
        .checked_add(1)
        .ok_or(SortSpillCodecError::AllocationFailed {
            kind,
            count: usize::MAX,
        })?;
    values
        .try_reserve(1)
        .map_err(|_| SortSpillCodecError::AllocationFailed { kind, count })?;
    values.push(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_lattice_values() -> Vec<Value> {
        let node = NodeView::new(NodeId::new(7), Some(LabelId::new(3)))
            .with_label_name("Person")
            .with_property("name", Value::String("Ada".to_owned()));
        let rel = RelView::new(
            RelId::new(11),
            NodeId::new(7),
            NodeId::new(8),
            Some(TypeId::new(4)),
        )
        .with_rel_type_name("KNOWS")
        .with_property("since", Value::Integer(2024));
        let end = NodeView::new(NodeId::new(8), None);

        let mut map = BTreeMap::new();
        map.insert("nested".to_owned(), Value::List(vec![Value::Null]));

        vec![
            Value::Null,
            Value::Boolean(true),
            Value::Integer(i64::MIN),
            Value::Float(-12.5),
            Value::String("snowman ☃".to_owned()),
            Value::Node(node.clone()),
            Value::Relationship(rel.clone()),
            Value::List(vec![Value::Integer(1), Value::String("two".to_owned())]),
            Value::Map(map),
            Value::Path(PathView::new(node).with_segment(rel, end)),
            Value::Temporal(
                ZonedDateTime::from_utc_nanos_and_offset(123_456_789, 19_800)
                    .expect("valid offset"),
            ),
            Value::LocalDateTime(
                LocalDateTime::new(2024, 60, 12_345_678).expect("valid local date-time"),
            ),
            Value::Date(Date::new(2024, 366).expect("valid leap-year date")),
            Value::Duration(Duration::new(-3, 987_654_321)),
            Value::Decimal(Decimal::new(38, i128::MIN + 1).expect("valid decimal")),
        ]
    }

    #[test]
    fn full_value_lattice_round_trips_losslessly() {
        let nan_bits = 0x7ff8_0000_0000_0042_u64;
        let records = vec![SortSpillRecord {
            ordinal: u64::MAX,
            keys: vec![Value::Float(f64::from_bits(nan_bits))],
            row: all_lattice_values(),
        }];

        let encoded = encode_records(&records).expect("encode");
        let decoded = decode_records(&encoded).expect("decode");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].ordinal, u64::MAX);
        assert_eq!(decoded[0].row, records[0].row);
        let Value::Float(decoded_nan) = &decoded[0].keys[0] else {
            panic!("expected float key");
        };
        assert_eq!(decoded_nan.to_bits(), nan_bits);

        // Covers fields whose semantic PartialEq intentionally ignores
        // presentation details (notably ZonedDateTime's offset).
        assert_eq!(
            encode_records(&decoded).expect("re-encode"),
            encoded,
            "decode/encode must preserve every original bit"
        );
    }

    #[test]
    fn corrupt_or_non_exact_input_is_rejected() {
        let encoded = encode_records(&[SortSpillRecord {
            ordinal: 1,
            keys: vec![Value::Boolean(true)],
            row: vec![],
        }])
        .expect("encode");

        for end in 0..encoded.len() {
            assert!(decode_records(&encoded[..end]).is_err(), "prefix {end}");
        }

        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(matches!(
            decode_records(&trailing),
            Err(SortSpillCodecError::TrailingBytes { remaining: 1 })
        ));

        let mut bad_bool = encoded;
        // header + record-count + ordinal + key-count + boolean tag
        bad_bool[8 + 4 + 8 + 4 + 1] = 2;
        assert!(matches!(
            decode_records(&bad_bool),
            Err(SortSpillCodecError::InvalidBoolean(2))
        ));
    }

    #[test]
    fn counts_and_nesting_are_bounded_before_allocation() {
        let mut impossible_count = MAGIC.to_vec();
        impossible_count.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            decode_records(&impossible_count),
            Err(SortSpillCodecError::InvalidLength {
                kind: "record count",
                ..
            })
        ));

        let mut nested = Value::Null;
        for _ in 0..=MAX_JSON_DECODE_DEPTH {
            nested = Value::List(vec![nested]);
        }
        assert!(matches!(
            encode_records(&[SortSpillRecord {
                ordinal: 0,
                keys: vec![nested],
                row: vec![],
            }]),
            Err(SortSpillCodecError::NestingTooDeep { .. })
        ));
    }

    #[test]
    fn expand_rows_round_trip_and_reject_join_magic() {
        let rows = vec![all_lattice_values(), vec![Value::Integer(64)]];
        let encoded = encode_expand_rows(&rows).expect("encode expand rows");
        assert_eq!(
            decode_expand_rows(&encoded).expect("decode expand rows"),
            rows
        );
        let join = encode_join_rows(&rows).expect("encode join rows");
        assert!(matches!(
            decode_expand_rows(&join),
            Err(SortSpillCodecError::InvalidMagic)
        ));
    }
}
