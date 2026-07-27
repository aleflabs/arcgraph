//! W14δ M5-13 — PackStream binary encoder + decoder.
//!
//! PackStream is Bolt's MessagePack-like wire format. This module is
//! the codec foundation the Bolt 5.0 message layer ([`super::message`])
//! pivots through. Per the spawn prompt's "Core surface" section, the
//! v1.0-α subset is:
//!
//! - Primitives: Null, Boolean, Int (TINY_INT, INT_8/16/32/64), Float
//!   (Float64), Bytes, String.
//! - Collections: List, Map.
//! - Structs: Node, Relationship, Path; v1.0-α DOES NOT ship
//!   Date/DateTime/Duration/Point as first-class struct decoders
//!   (forward-pin to v1.1+).
//!
//! # Wire layout (Bolt 5.0 §"Type System")
//!
//! Each value carries a 1-byte marker. The high nibble selects the
//! type family; the low nibble carries either the length (TINY_*) or
//! is fixed (e.g. `0xC0` = Null). Inline values:
//!
//! | Range          | Variant                          |
//! |----------------|----------------------------------|
//! | `0x00..=0x7F`  | TINY_INT (positive 0..127)       |
//! | `0x80..=0x8F`  | TINY_STRING (length 0..15)       |
//! | `0x90..=0x9F`  | TINY_LIST (length 0..15)         |
//! | `0xA0..=0xAF`  | TINY_MAP (length 0..15)          |
//! | `0xB0..=0xBF`  | TINY_STRUCT (length 0..15)       |
//! | `0xC0`         | Null                             |
//! | `0xC1`         | Float64                          |
//! | `0xC2`/`0xC3`  | Boolean false/true               |
//! | `0xC8..=0xCB`  | INT_8 / INT_16 / INT_32 / INT_64 |
//! | `0xCC..=0xCE`  | BYTES_8 / _16 / _32              |
//! | `0xD0..=0xD2`  | STRING_8 / _16 / _32             |
//! | `0xD4..=0xD6`  | LIST_8 / _16 / _32               |
//! | `0xD8..=0xDA`  | MAP_8 / _16 / _32                |
//! | `0xF0..=0xFF`  | TINY_INT (negative -16..-1)      |
//!
//! Lengths are big-endian unsigned; integers are big-endian signed.
//!
//! # Integer narrowing discipline (encode side)
//!
//! Per Bolt 5.0 §"Integer", an encoder MUST emit the **smallest**
//! variant that fits the value losslessly. The decoder MUST accept
//! any wider variant for the same value (so a peer sending a
//! conservatively-encoded `INT_64` for a value that would fit in
//! `TINY_INT` is still well-formed). Both directions are exercised
//! by the proptest (`tests/bolt_packstream_proptest.rs`).
//!
//! # Why a hand-rolled codec
//!
//! Per the spawn prompt's Core surface — "PackStream encoder/decoder
//! (Bolt's MessagePack-like binary format)" — we own the codec. The
//! crates.io options were vetted in the slice's license-gate and
//! dropped: `bolt-client` (MIT) is older and does not support Bolt
//! 5.0; `bolt-proto-5x` is MPL-2.0 (allowlisted but heavy) and pulls
//! a derive-macro surface that exceeds the v1.0-α scaffold scope. A
//! ~600 LOC hand-rolled codec keeps the dependency surface flat and
//! lets the message layer share encode/decode helpers without
//! re-routing through a foreign type system.

use std::collections::BTreeMap;

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────
// Marker constants (Bolt 5.0 §"Type System")
// ─────────────────────────────────────────────────────────────────────

const M_NULL: u8 = 0xC0;
const M_FLOAT64: u8 = 0xC1;
const M_FALSE: u8 = 0xC2;
const M_TRUE: u8 = 0xC3;
const M_INT8: u8 = 0xC8;
const M_INT16: u8 = 0xC9;
const M_INT32: u8 = 0xCA;
const M_INT64: u8 = 0xCB;
const M_BYTES8: u8 = 0xCC;
const M_BYTES16: u8 = 0xCD;
const M_BYTES32: u8 = 0xCE;
const M_STRING8: u8 = 0xD0;
const M_STRING16: u8 = 0xD1;
const M_STRING32: u8 = 0xD2;
const M_LIST8: u8 = 0xD4;
const M_LIST16: u8 = 0xD5;
const M_LIST32: u8 = 0xD6;
const M_MAP8: u8 = 0xD8;
const M_MAP16: u8 = 0xD9;
const M_MAP32: u8 = 0xDA;

const TINY_STRING_BASE: u8 = 0x80;
const TINY_LIST_BASE: u8 = 0x90;
const TINY_MAP_BASE: u8 = 0xA0;
const TINY_STRUCT_BASE: u8 = 0xB0;

// ─────────────────────────────────────────────────────────────────────
// Recursion depth cap (W14-retro IR L1-HIGH-3 Vector 2)
// ─────────────────────────────────────────────────────────────────────

/// Hard cap on nesting depth admitted by [`decode`].
///
/// Each LIST / MAP / STRUCT entry is a recursive `decode_inner` frame.
/// Without a depth cap, a `~30 KB` Bolt message of ~10K nested LIST
/// markers panics the Tokio worker via stack overflow (default 2 MiB /
/// ~200B per frame ≈ 10K depth). Cap is checked at entry to
/// `decode_inner`; over-the-line peers see [`PackError::DepthExceeded`]
/// instead of a process crash.
///
/// # Why 64
///
/// design-v2 §M5 ("Bolt protocol", line 978) lists Bolt as a v1.0-α
/// deliverable but does NOT publish a numeric nesting-depth cap, and
/// the Bolt 5.0 protocol spec itself leaves nesting depth
/// implementation-defined. Neo4j's published official drivers (Java /
/// Python / JavaScript / Go) likewise do not document a uniform cap;
/// referenced retro guidance cited 256 as a reasonable defense-in-
/// depth value. We pick **64** as the more conservative default —
/// real Bolt traffic in production rarely nests beyond depth ~10
/// (typical: result Map → List of Records → Record fields → property
/// Map → leaf primitives). 64 admits all observed shapes plus an
/// order of magnitude headroom while leaving ample stack margin on
/// Tokio's 2 MiB default worker stack (~200B per frame ≈ 10K depth
/// would be needed to panic, so we are ~150× under).
///
/// W14-retro IR L1-HIGH-3 Vector 2 (`fix/w14-retro-ir-bolt-security`):
/// without this cap, a pre-auth attacker (stub auth at v1.0-α admits
/// any principal) crashes the dispatcher task with stack overflow.
pub const MAX_PACKSTREAM_DEPTH: usize = 64;

// ─────────────────────────────────────────────────────────────────────
// Per-level List pre-allocation cap (#594 R1 H-1: depth×width OOM)
// ─────────────────────────────────────────────────────────────────────

/// Cap on the *pre-allocation* (`Vec::with_capacity`) for a single LIST
/// level in [`decode_list_body`], INDEPENDENT of the wire-declared element
/// count.
///
/// # Why a cap is needed on top of the `len > remaining` gate
///
/// [`decode_list_body`]'s per-level `len > remaining` bounds-check caps
/// ONE level's `with_capacity` to the bytes remaining. That alone does NOT
/// bound the *aggregate* pre-allocation: `decode_inner` allocates a level's
/// `items` Vec BEFORE recursing into its first element, so every live
/// recursive frame's pre-alloc coexists. A ≤ 16 MiB message nested to
/// [`MAX_PACKSTREAM_DEPTH`] (64), where each level is a `LIST_32` declaring
/// `len ≈ remaining`, makes the SUM of the 64 live `with_capacity` calls
/// ≈ `MAX_PACKSTREAM_DEPTH × remaining × size_of::<PackValue>()` (tens of
/// GiB) → `handle_alloc_error` aborts the dispatcher. The depth gate does
/// not catch it (each level is well under depth 64); the per-level byte
/// gate does not catch it (each level individually fits `remaining`); only
/// the *product* OOMs. A `max_len`-bounded fuzz corpus (#559, `max_len =
/// 4096`) cannot reach the message size that triggers it, so it would go
/// FALSE-GREEN.
///
/// # Why 4096
///
/// Pre-allocating up to 4096 slots covers the overwhelming majority of real
/// Bolt List shapes (record field lists, small result rows) with zero
/// reallocation, while bounding peak pre-alloc to `MAX_PACKSTREAM_DEPTH ×
/// LIST_PREALLOC_CAP × size_of::<PackValue>()` ≈ 8 MiB, INDEPENDENT of
/// message size. Lists longer than the cap still decode correctly: the
/// `items` Vec `push`-grows to the true element count, which is itself
/// bounded by `remaining` (every element consumes ≥ 1 wire byte), so total
/// live capacity stays proportional to the message — never to the
/// wire-declared `len` summed across levels. [`decode_map_body`] needs no
/// companion cap: its `BTreeMap` does not pre-allocate.
const LIST_PREALLOC_CAP: usize = 4096;

// ─────────────────────────────────────────────────────────────────────
// Bolt struct tags (used by the message layer; re-exported)
// ─────────────────────────────────────────────────────────────────────

/// Bolt 5.0 Node struct tag (`'N'`). Fields: `[id, labels, properties,
/// element_id]`.
pub const TAG_NODE: u8 = 0x4E;

/// Bolt 5.0 Relationship struct tag (`'R'`). Fields:
/// `[id, start_id, end_id, type, properties, element_id,
/// start_element_id, end_element_id]`.
pub const TAG_RELATIONSHIP: u8 = 0x52;

/// Bolt 5.0 UnboundRelationship struct tag (`'r'`). Fields:
/// `[id, type, properties, element_id]`.
pub const TAG_UNBOUND_RELATIONSHIP: u8 = 0x72;

/// Bolt 5.0 Path struct tag (`'P'`). Fields:
/// `[nodes, relationships, indices]`.
pub const TAG_PATH: u8 = 0x50;

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Codec-local error type for the PackStream layer. Translates to a
/// [`crate::error::MCPError::ParseError`] / `InternalError` at the
/// public boundary per `docs/codec-error-translation.md` discipline.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PackError {
    /// Encoded length exceeds the 32-bit limit Bolt admits for any
    /// container type.
    #[error("packstream length {0} exceeds 32-bit limit")]
    LengthOverflow(usize),
    /// Decoder reached end-of-buffer before finishing a value.
    #[error("packstream unexpected EOF at offset {0}")]
    UnexpectedEof(usize),
    /// Decoder saw a marker byte it does not recognize.
    #[error("packstream unknown marker 0x{marker:02X} at offset {offset}")]
    UnknownMarker { marker: u8, offset: usize },
    /// Decoder saw a string whose UTF-8 was invalid.
    #[error("packstream invalid utf-8 at offset {0}")]
    InvalidUtf8(usize),
    /// Decoder hit a value it understood but the caller did not accept
    /// (e.g., a Map key was decoded as a non-string Value).
    #[error("packstream non-string map key at offset {0}")]
    NonStringMapKey(usize),
    /// Decoder hit a struct whose tag is not one of the v1.0-α
    /// recognized tags. Forward-binds to v1.1+ types like
    /// Date/DateTime/Duration/Point.
    #[error("packstream unsupported struct tag 0x{tag:02X} at offset {offset}")]
    UnsupportedStructTag { tag: u8, offset: usize },
    /// Decoder hit nesting depth ≥ [`MAX_PACKSTREAM_DEPTH`]. Defends
    /// the dispatcher task's stack against a hostile peer that emits
    /// deeply-nested LIST/MAP/STRUCT markers (stack-overflow DoS — the
    /// `decode_inner` recursion would otherwise panic the task at
    /// ~10K depth on the default 2 MiB Tokio worker stack).
    /// W14-retro IR L1-HIGH-3 Vector 2.
    #[error("packstream nesting depth {depth} exceeds cap {max}")]
    DepthExceeded {
        /// Depth at which the cap was hit (= [`MAX_PACKSTREAM_DEPTH`]).
        depth: usize,
        /// The cap value the call exceeded; included in the variant so
        /// FAILURE messages report the boundary the operator would
        /// need to raise.
        max: usize,
    },
}

// ─────────────────────────────────────────────────────────────────────
// PackValue — the codec's lattice
// ─────────────────────────────────────────────────────────────────────

/// PackStream value lattice mirroring Bolt 5.0 §"Type System".
///
/// Distinct from [`arcgraph_query::executor::Value`]: although that type
/// now carries a `Map` variant too (ADR-191), PackStream's `Map` is
/// also used for message-extra fields and for property bags inside Node
/// / Relationship structs (which are not executor `Value`s). The bridge
/// in [`super::value`] converts an executor `Value::Map` into a
/// `PackValue::Map` at the message-encode boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum PackValue {
    /// PackStream Null marker (`0xC0`).
    Null,
    /// PackStream Boolean (`0xC2` / `0xC3`).
    Boolean(bool),
    /// PackStream signed integer. Encoder narrows to the smallest
    /// variant per §"Integer narrowing discipline".
    Integer(i64),
    /// PackStream Float64 (`0xC1`). NaN / Inf are admissible per
    /// IEEE-754; equality follows the IEEE rules.
    Float(f64),
    /// PackStream Bytes (`0xCC..=0xCE`). Used for raw binary blobs;
    /// the v1.0-α Bolt server emits Bytes rarely (mostly inside
    /// Date/Time encodings which are forward-deferred).
    Bytes(Vec<u8>),
    /// PackStream String (`0x80..=0x8F` / `0xD0..=0xD2`). UTF-8.
    String(String),
    /// PackStream List (`0x90..=0x9F` / `0xD4..=0xD6`).
    List(Vec<PackValue>),
    /// PackStream Map (`0xA0..=0xAF` / `0xD8..=0xDA`). Keys MUST be
    /// strings per Bolt 5.0 §"Map".
    Map(BTreeMap<String, PackValue>),
    /// PackStream Struct (`0xB0..=0xBF`). The tag identifies the
    /// struct family; the field vec carries the positional fields.
    Struct {
        /// The single byte that follows the marker, identifying the
        /// struct family (e.g. [`TAG_NODE`]).
        tag: u8,
        /// Positional fields. Length 0..=15 (TINY_STRUCT only — Bolt
        /// has no STRUCT_8/_16/_32; large structs are not admissible).
        fields: Vec<PackValue>,
    },
}

impl PackValue {
    /// Convenience: build a PackStream string Value from any borrow-
    /// shaped UTF-8 source.
    pub fn string(s: impl Into<String>) -> Self {
        PackValue::String(s.into())
    }

    /// Convenience: build a PackStream map from an iterable.
    pub fn map(entries: impl IntoIterator<Item = (String, PackValue)>) -> Self {
        PackValue::Map(entries.into_iter().collect())
    }

    /// Convenience: build a PackStream struct (tag + fields).
    pub fn structured(tag: u8, fields: Vec<PackValue>) -> Self {
        PackValue::Struct { tag, fields }
    }

    /// Borrow as a string; `None` on type mismatch.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PackValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Borrow as a map; `None` on type mismatch.
    pub fn as_map(&self) -> Option<&BTreeMap<String, PackValue>> {
        match self {
            PackValue::Map(m) => Some(m),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Encoder
// ─────────────────────────────────────────────────────────────────────

/// Encode a [`PackValue`] into the supplied `Vec<u8>`, appending bytes
/// at the end. Returns the number of bytes appended on success.
pub fn encode(out: &mut Vec<u8>, value: &PackValue) -> Result<usize, PackError> {
    let start = out.len();
    encode_inner(out, value)?;
    Ok(out.len() - start)
}

fn encode_inner(out: &mut Vec<u8>, value: &PackValue) -> Result<(), PackError> {
    match value {
        PackValue::Null => out.push(M_NULL),
        PackValue::Boolean(false) => out.push(M_FALSE),
        PackValue::Boolean(true) => out.push(M_TRUE),
        PackValue::Integer(i) => encode_int(out, *i),
        PackValue::Float(f) => {
            out.push(M_FLOAT64);
            out.extend_from_slice(&f.to_be_bytes());
        }
        PackValue::Bytes(b) => encode_bytes(out, b)?,
        PackValue::String(s) => encode_string(out, s)?,
        PackValue::List(items) => {
            encode_collection_header(
                out,
                items.len(),
                TINY_LIST_BASE,
                M_LIST8,
                M_LIST16,
                M_LIST32,
            )?;
            for item in items {
                encode_inner(out, item)?;
            }
        }
        PackValue::Map(entries) => {
            encode_collection_header(out, entries.len(), TINY_MAP_BASE, M_MAP8, M_MAP16, M_MAP32)?;
            for (k, v) in entries {
                encode_string(out, k)?;
                encode_inner(out, v)?;
            }
        }
        PackValue::Struct { tag, fields } => {
            // Bolt 5.0 only admits TINY_STRUCT; struct-arity > 15 is
            // out of contract. The Bolt 5.0 messages we ship max at
            // arity 8 (Relationship), so this is non-binding for
            // production callers.
            if fields.len() > 15 {
                return Err(PackError::LengthOverflow(fields.len()));
            }
            #[allow(clippy::cast_possible_truncation)] // bounded by len <= 15
            out.push(TINY_STRUCT_BASE | fields.len() as u8);
            out.push(*tag);
            for field in fields {
                encode_inner(out, field)?;
            }
        }
    }
    Ok(())
}

fn encode_int(out: &mut Vec<u8>, i: i64) {
    // §"Integer narrowing discipline": emit the smallest variant.
    if (-16..=127).contains(&i) {
        // TINY_INT — the value byte IS the marker. Negative values
        // wrap to the 0xF0..=0xFF block per two's-complement.
        out.push(i as u8);
    } else if (i8::MIN as i64..=i8::MAX as i64).contains(&i) {
        out.push(M_INT8);
        out.push(i as i8 as u8);
    } else if (i16::MIN as i64..=i16::MAX as i64).contains(&i) {
        out.push(M_INT16);
        out.extend_from_slice(&(i as i16).to_be_bytes());
    } else if (i32::MIN as i64..=i32::MAX as i64).contains(&i) {
        out.push(M_INT32);
        out.extend_from_slice(&(i as i32).to_be_bytes());
    } else {
        out.push(M_INT64);
        out.extend_from_slice(&i.to_be_bytes());
    }
}

fn encode_bytes(out: &mut Vec<u8>, b: &[u8]) -> Result<(), PackError> {
    let len = b.len();
    if len <= 0xFF {
        out.push(M_BYTES8);
        out.push(len as u8);
    } else if len <= 0xFFFF {
        out.push(M_BYTES16);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else if len <= u32::MAX as usize {
        out.push(M_BYTES32);
        out.extend_from_slice(&(len as u32).to_be_bytes());
    } else {
        return Err(PackError::LengthOverflow(len));
    }
    out.extend_from_slice(b);
    Ok(())
}

fn encode_string(out: &mut Vec<u8>, s: &str) -> Result<(), PackError> {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len <= 15 {
        out.push(TINY_STRING_BASE | len as u8);
    } else if len <= 0xFF {
        out.push(M_STRING8);
        out.push(len as u8);
    } else if len <= 0xFFFF {
        out.push(M_STRING16);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else if len <= u32::MAX as usize {
        out.push(M_STRING32);
        out.extend_from_slice(&(len as u32).to_be_bytes());
    } else {
        return Err(PackError::LengthOverflow(len));
    }
    out.extend_from_slice(bytes);
    Ok(())
}

fn encode_collection_header(
    out: &mut Vec<u8>,
    len: usize,
    tiny_base: u8,
    m8: u8,
    m16: u8,
    m32: u8,
) -> Result<(), PackError> {
    if len <= 15 {
        out.push(tiny_base | len as u8);
    } else if len <= 0xFF {
        out.push(m8);
        out.push(len as u8);
    } else if len <= 0xFFFF {
        out.push(m16);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else if len <= u32::MAX as usize {
        out.push(m32);
        out.extend_from_slice(&(len as u32).to_be_bytes());
    } else {
        return Err(PackError::LengthOverflow(len));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Decoder
// ─────────────────────────────────────────────────────────────────────

/// Decode a single PackStream value from `bytes` starting at `offset`.
/// Returns the decoded value plus the number of bytes consumed.
///
/// Nesting depth is capped at [`MAX_PACKSTREAM_DEPTH`]; deeper LIST /
/// MAP / STRUCT structures surface [`PackError::DepthExceeded`].
pub fn decode(bytes: &[u8], offset: usize) -> Result<(PackValue, usize), PackError> {
    let mut cur = offset;
    let value = decode_inner(bytes, &mut cur, 0)?;
    Ok((value, cur - offset))
}

fn decode_inner(bytes: &[u8], cur: &mut usize, depth: usize) -> Result<PackValue, PackError> {
    // Entry-side depth gate: every recursive frame increments `depth`
    // before delegating, so the cap is checked once per nesting level.
    // Primitives (Int / Float / Null / Bool / Bytes / String) hit the
    // gate harmlessly at their level and return without recursing —
    // the gate's only real consumers are LIST / MAP / STRUCT bodies.
    if depth >= MAX_PACKSTREAM_DEPTH {
        return Err(PackError::DepthExceeded {
            depth,
            max: MAX_PACKSTREAM_DEPTH,
        });
    }
    let marker = read_u8(bytes, cur)?;
    // TINY ranges first.
    match marker {
        0x00..=0x7F => return Ok(PackValue::Integer(marker as i64)),
        0xF0..=0xFF => {
            // TINY_INT negative: extend sign bit.
            return Ok(PackValue::Integer((marker as i8) as i64));
        }
        0x80..=0x8F => {
            let len = (marker & 0x0F) as usize;
            return decode_string_body(bytes, cur, len);
        }
        0x90..=0x9F => {
            let len = (marker & 0x0F) as usize;
            return decode_list_body(bytes, cur, len, depth);
        }
        0xA0..=0xAF => {
            let len = (marker & 0x0F) as usize;
            return decode_map_body(bytes, cur, len, depth);
        }
        0xB0..=0xBF => {
            let len = (marker & 0x0F) as usize;
            return decode_struct_body(bytes, cur, len, depth);
        }
        _ => {}
    }
    // Single-byte fixed markers + variable-length headers.
    match marker {
        M_NULL => Ok(PackValue::Null),
        M_FALSE => Ok(PackValue::Boolean(false)),
        M_TRUE => Ok(PackValue::Boolean(true)),
        M_FLOAT64 => {
            let mut buf = [0u8; 8];
            read_into(bytes, cur, &mut buf)?;
            Ok(PackValue::Float(f64::from_be_bytes(buf)))
        }
        M_INT8 => {
            let v = read_u8(bytes, cur)? as i8 as i64;
            Ok(PackValue::Integer(v))
        }
        M_INT16 => {
            let mut buf = [0u8; 2];
            read_into(bytes, cur, &mut buf)?;
            Ok(PackValue::Integer(i16::from_be_bytes(buf) as i64))
        }
        M_INT32 => {
            let mut buf = [0u8; 4];
            read_into(bytes, cur, &mut buf)?;
            Ok(PackValue::Integer(i32::from_be_bytes(buf) as i64))
        }
        M_INT64 => {
            let mut buf = [0u8; 8];
            read_into(bytes, cur, &mut buf)?;
            Ok(PackValue::Integer(i64::from_be_bytes(buf)))
        }
        M_BYTES8 => {
            let len = read_u8(bytes, cur)? as usize;
            decode_bytes_body(bytes, cur, len)
        }
        M_BYTES16 => {
            let mut buf = [0u8; 2];
            read_into(bytes, cur, &mut buf)?;
            decode_bytes_body(bytes, cur, u16::from_be_bytes(buf) as usize)
        }
        M_BYTES32 => {
            let mut buf = [0u8; 4];
            read_into(bytes, cur, &mut buf)?;
            decode_bytes_body(bytes, cur, u32::from_be_bytes(buf) as usize)
        }
        M_STRING8 => {
            let len = read_u8(bytes, cur)? as usize;
            decode_string_body(bytes, cur, len)
        }
        M_STRING16 => {
            let mut buf = [0u8; 2];
            read_into(bytes, cur, &mut buf)?;
            decode_string_body(bytes, cur, u16::from_be_bytes(buf) as usize)
        }
        M_STRING32 => {
            let mut buf = [0u8; 4];
            read_into(bytes, cur, &mut buf)?;
            decode_string_body(bytes, cur, u32::from_be_bytes(buf) as usize)
        }
        M_LIST8 => {
            let len = read_u8(bytes, cur)? as usize;
            decode_list_body(bytes, cur, len, depth)
        }
        M_LIST16 => {
            let mut buf = [0u8; 2];
            read_into(bytes, cur, &mut buf)?;
            decode_list_body(bytes, cur, u16::from_be_bytes(buf) as usize, depth)
        }
        M_LIST32 => {
            let mut buf = [0u8; 4];
            read_into(bytes, cur, &mut buf)?;
            decode_list_body(bytes, cur, u32::from_be_bytes(buf) as usize, depth)
        }
        M_MAP8 => {
            let len = read_u8(bytes, cur)? as usize;
            decode_map_body(bytes, cur, len, depth)
        }
        M_MAP16 => {
            let mut buf = [0u8; 2];
            read_into(bytes, cur, &mut buf)?;
            decode_map_body(bytes, cur, u16::from_be_bytes(buf) as usize, depth)
        }
        M_MAP32 => {
            let mut buf = [0u8; 4];
            read_into(bytes, cur, &mut buf)?;
            decode_map_body(bytes, cur, u32::from_be_bytes(buf) as usize, depth)
        }
        other => Err(PackError::UnknownMarker {
            marker: other,
            offset: *cur - 1,
        }),
    }
}

fn decode_string_body(bytes: &[u8], cur: &mut usize, len: usize) -> Result<PackValue, PackError> {
    let start = *cur;
    if start + len > bytes.len() {
        return Err(PackError::UnexpectedEof(start + len));
    }
    let slice = &bytes[start..start + len];
    *cur += len;
    let s = std::str::from_utf8(slice).map_err(|_| PackError::InvalidUtf8(start))?;
    Ok(PackValue::String(s.to_string()))
}

fn decode_bytes_body(bytes: &[u8], cur: &mut usize, len: usize) -> Result<PackValue, PackError> {
    let start = *cur;
    if start + len > bytes.len() {
        return Err(PackError::UnexpectedEof(start + len));
    }
    let v = bytes[start..start + len].to_vec();
    *cur += len;
    Ok(PackValue::Bytes(v))
}

fn decode_list_body(
    bytes: &[u8],
    cur: &mut usize,
    len: usize,
    depth: usize,
) -> Result<PackValue, PackError> {
    // Bounds-check the UNTRUSTED length-prefix against remaining input
    // BEFORE allocating. A LIST_32 marker can encode `len` up to u32::MAX
    // (~4.29e9); `Vec::<PackValue>::with_capacity(len)` would then request
    // len × size_of::<PackValue>() (≥ 32 B) ≈ 128 GiB, which the global
    // allocator cannot satisfy — `handle_alloc_error` aborts (or the OS
    // OOM-kills) the dispatcher task. That is an abort, NOT a catchable
    // unwind, so a caller-side `catch_unwind` cannot recover; the gate has
    // to live here, before the allocation (#577; W28-S4 `bolt_packstream_fuzz`,
    // #559). Every List element occupies ≥ 1 byte on the wire (a 1-byte
    // marker minimum), so a well-formed List of `len` items needs ≥ `len`
    // bytes remaining; anything more is unsatisfiable. This is the same
    // `start + len > bytes.len()` discipline `decode_string_body` /
    // `decode_bytes_body` already apply (per
    // `feedback_security_class_first_network_surface`: a network-facing
    // parser MUST NOT panic on adversarial bytes).
    //
    // [#594 R1 H-1] This gate bounds a SINGLE level's pre-alloc to
    // `remaining`, but NOT the AGGREGATE across nested levels: `decode_inner`
    // allocates this `items` Vec BEFORE recursing into its first element, so
    // every live frame's pre-alloc coexists. 64 levels (the
    // `MAX_PACKSTREAM_DEPTH` ceiling) each declaring `len ≈ remaining` would
    // sum to a multi-GiB pre-alloc that OOM-aborts the dispatcher even though
    // each level individually fits `remaining` (the depth gate does not catch
    // it either — each level is under depth 64). The `len.min(LIST_PREALLOC_CAP)`
    // below bounds the per-level `with_capacity` INDEPENDENTLY of the
    // wire-declared count, capping peak pre-alloc at `MAX_PACKSTREAM_DEPTH ×
    // LIST_PREALLOC_CAP × size_of::<PackValue>()` ≈ 8 MiB (see
    // `LIST_PREALLOC_CAP`). The Vec still `push`-grows to the true element
    // count, itself bounded by `remaining` (every element consumes ≥ 1 byte).
    let remaining = bytes.len().saturating_sub(*cur);
    if len > remaining {
        return Err(PackError::UnexpectedEof(cur.saturating_add(len)));
    }
    let mut items = Vec::with_capacity(len.min(LIST_PREALLOC_CAP));
    for _ in 0..len {
        items.push(decode_inner(bytes, cur, depth + 1)?);
    }
    Ok(PackValue::List(items))
}

fn decode_map_body(
    bytes: &[u8],
    cur: &mut usize,
    len: usize,
    depth: usize,
) -> Result<PackValue, PackError> {
    // Same untrusted-length-prefix discipline as `decode_list_body`. A Map
    // entry is a key (≥ 1 byte) + a value (≥ 1 byte) = ≥ 2 bytes on the
    // wire, so a well-formed Map of `len` entries needs ≥ 2·len bytes
    // remaining. `BTreeMap::new()` does not pre-allocate (so the abort
    // vector here is milder than the List's `with_capacity`), but a hostile
    // MAP_32 length-prefix (`len` up to u32::MAX) would otherwise spin the
    // decode loop up to u32::MAX times — reject it upfront for a structured
    // `UnexpectedEof` and uniform discipline (#577). `saturating_mul`
    // guards the 2·len product against overflow on a 32-bit `usize`.
    let remaining = bytes.len().saturating_sub(*cur);
    let min_needed = len.saturating_mul(2);
    if min_needed > remaining {
        return Err(PackError::UnexpectedEof(cur.saturating_add(min_needed)));
    }
    // No pre-alloc cap needed here (BTreeMap does not reserve capacity); see
    // LIST_PREALLOC_CAP on the list path for the depth×width pre-alloc bound
    // that the `Vec` path requires (#594 R1 H-1).
    let mut map = BTreeMap::new();
    for _ in 0..len {
        // Map keys MUST be strings per Bolt 5.0 §"Map".
        let key_offset = *cur;
        let key_value = decode_inner(bytes, cur, depth + 1)?;
        let key = match key_value {
            PackValue::String(s) => s,
            _ => return Err(PackError::NonStringMapKey(key_offset)),
        };
        let val = decode_inner(bytes, cur, depth + 1)?;
        map.insert(key, val);
    }
    Ok(PackValue::Map(map))
}

fn decode_struct_body(
    bytes: &[u8],
    cur: &mut usize,
    arity: usize,
    depth: usize,
) -> Result<PackValue, PackError> {
    let tag = read_u8(bytes, cur)?;
    let mut fields = Vec::with_capacity(arity);
    for _ in 0..arity {
        fields.push(decode_inner(bytes, cur, depth + 1)?);
    }
    Ok(PackValue::Struct { tag, fields })
}

fn read_u8(bytes: &[u8], cur: &mut usize) -> Result<u8, PackError> {
    if *cur >= bytes.len() {
        return Err(PackError::UnexpectedEof(*cur));
    }
    let v = bytes[*cur];
    *cur += 1;
    Ok(v)
}

fn read_into(bytes: &[u8], cur: &mut usize, buf: &mut [u8]) -> Result<(), PackError> {
    let n = buf.len();
    if *cur + n > bytes.len() {
        return Err(PackError::UnexpectedEof(*cur + n));
    }
    buf.copy_from_slice(&bytes[*cur..*cur + n]);
    *cur += n;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(v: &PackValue) -> PackValue {
        let mut buf = Vec::new();
        encode(&mut buf, v).expect("encode ok");
        let (decoded, n) = decode(&buf, 0).expect("decode ok");
        assert_eq!(n, buf.len(), "consumed all bytes");
        decoded
    }

    #[test]
    fn primitives_roundtrip_lossless() {
        // Single-shot pin for every primitive variant. Any encode-side
        // narrowing decision is reversible by the decoder.
        let cases = vec![
            PackValue::Null,
            PackValue::Boolean(true),
            PackValue::Boolean(false),
            PackValue::Integer(0),
            PackValue::Integer(127),
            PackValue::Integer(-16),
            PackValue::Integer(-17),
            PackValue::Integer(128),
            PackValue::Integer(i16::MAX as i64),
            PackValue::Integer(i32::MAX as i64),
            PackValue::Integer(i64::MIN),
            PackValue::Integer(i64::MAX),
            PackValue::Float(0.0),
            PackValue::Float(-1.5),
            PackValue::Float(f64::INFINITY),
            PackValue::String(String::new()),
            PackValue::String("hello".into()),
            PackValue::String("a".repeat(20)),
            PackValue::String("a".repeat(300)),
            PackValue::Bytes(b"abc".to_vec()),
            PackValue::Bytes(vec![0; 0]),
            PackValue::List(vec![]),
            PackValue::List(vec![PackValue::Integer(1), PackValue::Integer(2)]),
            PackValue::Map(BTreeMap::new()),
        ];
        for c in cases {
            assert_eq!(roundtrip(&c), c, "case {c:?} failed roundtrip");
        }
    }

    #[test]
    fn integer_narrowing_picks_smallest_variant() {
        // §"Integer narrowing discipline": each band picks the smallest
        // marker that fits.
        let mut buf = Vec::new();
        encode(&mut buf, &PackValue::Integer(0)).unwrap();
        // TINY_INT: just the byte itself.
        assert_eq!(buf, &[0x00]);

        buf.clear();
        encode(&mut buf, &PackValue::Integer(-1)).unwrap();
        // TINY_INT negative: two's-complement byte.
        assert_eq!(buf, &[0xFF]);

        buf.clear();
        encode(&mut buf, &PackValue::Integer(-17)).unwrap();
        // INT_8: marker + 1 body byte.
        assert_eq!(buf, &[M_INT8, 0xEF]);

        buf.clear();
        encode(&mut buf, &PackValue::Integer(200)).unwrap();
        // INT_16: marker + 2 BE bytes.
        assert_eq!(buf, &[M_INT16, 0x00, 0xC8]);

        buf.clear();
        encode(&mut buf, &PackValue::Integer(70_000)).unwrap();
        // INT_32: marker + 4 BE bytes.
        assert_eq!(buf, &[M_INT32, 0x00, 0x01, 0x11, 0x70]);

        buf.clear();
        encode(&mut buf, &PackValue::Integer(i64::MAX)).unwrap();
        // INT_64: marker + 8 BE bytes.
        let mut expected = vec![M_INT64];
        expected.extend_from_slice(&i64::MAX.to_be_bytes());
        assert_eq!(buf, expected);
    }

    #[test]
    fn list_of_mixed_types_roundtrips() {
        // Heterogeneous list (Cypher-9 admits these). Pin the codec
        // doesn't mishandle a String + Integer + Null + List sequence.
        let v = PackValue::List(vec![
            PackValue::String("x".into()),
            PackValue::Integer(7),
            PackValue::Null,
            PackValue::List(vec![PackValue::Boolean(true)]),
        ]);
        assert_eq!(roundtrip(&v), v);
    }

    #[test]
    fn map_with_mixed_value_types_roundtrips() {
        let mut m = BTreeMap::new();
        m.insert("name".into(), PackValue::String("Alice".into()));
        m.insert("age".into(), PackValue::Integer(30));
        m.insert("active".into(), PackValue::Boolean(true));
        m.insert("score".into(), PackValue::Float(2.5));
        let v = PackValue::Map(m);
        assert_eq!(roundtrip(&v), v);
    }

    #[test]
    fn node_struct_roundtrips() {
        // Bolt 5.0 Node = struct(0x4E, [id, labels, props, element_id]).
        let mut props = BTreeMap::new();
        props.insert("name".into(), PackValue::String("Alice".into()));
        let node = PackValue::Struct {
            tag: TAG_NODE,
            fields: vec![
                PackValue::Integer(42),
                PackValue::List(vec![PackValue::String("Person".into())]),
                PackValue::Map(props),
                PackValue::String("4:abc:42".into()),
            ],
        };
        assert_eq!(roundtrip(&node), node);
    }

    #[test]
    fn relationship_struct_roundtrips() {
        // Bolt 5.0 Relationship = struct(0x52, [id, start, end, type,
        // props, element_id, start_element_id, end_element_id]).
        let rel = PackValue::Struct {
            tag: TAG_RELATIONSHIP,
            fields: vec![
                PackValue::Integer(1),
                PackValue::Integer(2),
                PackValue::Integer(3),
                PackValue::String("KNOWS".into()),
                PackValue::Map(BTreeMap::new()),
                PackValue::String("5:abc:1".into()),
                PackValue::String("4:abc:2".into()),
                PackValue::String("4:abc:3".into()),
            ],
        };
        assert_eq!(roundtrip(&rel), rel);
    }

    #[test]
    fn path_struct_roundtrips() {
        // Bolt 5.0 Path = struct(0x50, [nodes, relationships, indices]).
        let path = PackValue::Struct {
            tag: TAG_PATH,
            fields: vec![
                PackValue::List(vec![PackValue::Struct {
                    tag: TAG_NODE,
                    fields: vec![
                        PackValue::Integer(1),
                        PackValue::List(vec![]),
                        PackValue::Map(BTreeMap::new()),
                        PackValue::String("4:x:1".into()),
                    ],
                }]),
                PackValue::List(vec![]),
                PackValue::List(vec![]),
            ],
        };
        assert_eq!(roundtrip(&path), path);
    }

    #[test]
    fn decode_rejects_unknown_marker() {
        // 0xC4..=0xC7 are reserved per Bolt 5.0 §"Type System".
        let err = decode(&[0xC4], 0).unwrap_err();
        match err {
            PackError::UnknownMarker { marker: 0xC4, .. } => {}
            other => panic!("expected unknown-marker, got {other:?}"),
        }
    }

    #[test]
    fn decode_rejects_truncated_string() {
        // STRING_8 with len=10 but only 3 body bytes.
        let bytes = [M_STRING8, 0x0A, b'a', b'b', b'c'];
        let err = decode(&bytes, 0).unwrap_err();
        assert!(matches!(err, PackError::UnexpectedEof(_)));
    }

    #[test]
    fn decode_rejects_invalid_utf8() {
        // STRING_8 with invalid UTF-8 body.
        let bytes = [M_STRING8, 0x02, 0xFF, 0xFE];
        let err = decode(&bytes, 0).unwrap_err();
        assert!(matches!(err, PackError::InvalidUtf8(_)));
    }

    #[test]
    fn decode_accepts_wider_int_variant_for_small_value() {
        // Per §"Integer narrowing discipline" the decoder MUST accept
        // any wider variant for the same value. Pin: a INT_64 carrying
        // value 7 decodes to Integer(7).
        let mut bytes = vec![M_INT64];
        bytes.extend_from_slice(&7i64.to_be_bytes());
        let (v, n) = decode(&bytes, 0).unwrap();
        assert_eq!(v, PackValue::Integer(7));
        assert_eq!(n, 9);
    }

    #[test]
    fn map_with_non_string_key_rejected() {
        // Hand-craft a TINY_MAP with arity 1 whose key is a TINY_INT
        // (illegal). Encoder won't emit this but a malicious peer might.
        let bytes = [
            TINY_MAP_BASE | 1, // 1-entry map
            0x07,              // TINY_INT 7 as the "key"
            M_NULL,            // value
        ];
        let err = decode(&bytes, 0).unwrap_err();
        assert!(matches!(err, PackError::NonStringMapKey(_)));
    }

    #[test]
    fn struct_arity_over_15_rejected_at_encode() {
        // Encoder MUST reject struct arity > 15 since Bolt has no
        // STRUCT_8/16/32 variants.
        let fields = (0..16).map(|i| PackValue::Integer(i as i64)).collect();
        let v = PackValue::Struct { tag: 0x55, fields };
        let mut buf = Vec::new();
        let err = encode(&mut buf, &v).unwrap_err();
        assert!(matches!(err, PackError::LengthOverflow(16)));
    }

    #[test]
    fn empty_string_uses_tiny_marker() {
        // TINY_STRING_BASE | 0 = 0x80. Single byte on the wire.
        let mut buf = Vec::new();
        encode(&mut buf, &PackValue::String(String::new())).unwrap();
        assert_eq!(buf, &[0x80]);
    }

    #[test]
    fn long_list_uses_list_32_when_needed() {
        // 70_000-element list fits LIST_32. Pin the marker selection.
        let items: Vec<PackValue> = (0..70_000).map(|_| PackValue::Null).collect();
        let v = PackValue::List(items);
        let mut buf = Vec::new();
        encode(&mut buf, &v).unwrap();
        assert_eq!(buf[0], M_LIST32);
    }

    #[test]
    fn nested_map_in_list_in_map_roundtrips() {
        // Real-world shape: HELLO extra contains a list of strings; a
        // record contains a map carrying a list of nested maps.
        let mut inner = BTreeMap::new();
        inner.insert("k".into(), PackValue::Integer(1));
        let mut outer = BTreeMap::new();
        outer.insert(
            "l".into(),
            PackValue::List(vec![PackValue::Map(inner.clone())]),
        );
        let v = PackValue::Map(outer);
        assert_eq!(roundtrip(&v), v);
    }

    #[test]
    fn string_at_each_size_band_roundtrips() {
        // 0/15/16/255/256/65535/65536-byte strings — pin each band
        // boundary so a future encoder edit can't silently slip into
        // the wrong variant.
        for len in [0, 15, 16, 255, 256, 65_535, 65_536] {
            let s = "a".repeat(len);
            let v = PackValue::String(s);
            assert_eq!(roundtrip(&v), v, "len={len}");
        }
    }

    #[test]
    fn decode_rejects_depth_exceeding_max() {
        // W14-retro IR L1-HIGH-3 Vector 2 pin: a deeply-nested
        // PackStream message MUST surface DepthExceeded rather than
        // panic the dispatcher task via stack overflow.
        //
        // Compose `MAX_PACKSTREAM_DEPTH + 1` (= 65) nested TINY_LIST(1)
        // markers. Each TINY_LIST marker is a single byte (0x91 for
        // length-1 list). Without a leaf at the bottom, the deepest
        // `decode_inner` invocation hits the depth gate before
        // reading any further bytes.
        let bytes = vec![TINY_LIST_BASE | 1; MAX_PACKSTREAM_DEPTH + 1];
        let err = decode(&bytes, 0).unwrap_err();
        match err {
            PackError::DepthExceeded { depth, max } => {
                assert_eq!(max, MAX_PACKSTREAM_DEPTH);
                assert_eq!(depth, MAX_PACKSTREAM_DEPTH);
            }
            other => panic!("expected DepthExceeded, got {other:?}"),
        }
    }

    #[test]
    fn decode_admits_depth_at_boundary() {
        // Sister-pin to depth-exceeded: the boundary case (depth ==
        // MAX_PACKSTREAM_DEPTH - 1 levels of nested LIST + leaf
        // primitive) MUST decode successfully. Catches an off-by-one
        // that would make the cap fire one level too early.
        //
        // Build (MAX_PACKSTREAM_DEPTH - 1) TINY_LIST(1) markers
        // followed by a TINY_INT leaf. Each LIST has len=1 holding
        // either the next LIST or the leaf.
        let mut bytes = vec![TINY_LIST_BASE | 1; MAX_PACKSTREAM_DEPTH - 1];
        bytes.push(0x07); // TINY_INT(7) leaf
        let (val, n) = decode(&bytes, 0).expect("boundary-depth must decode");
        assert_eq!(n, bytes.len());
        // Walk the resulting structure and verify the leaf landed at
        // the expected depth — the structure must be (MAX_PACKSTREAM_
        // DEPTH - 1) levels of List nesting around a single Integer.
        let mut cursor = &val;
        for _ in 0..(MAX_PACKSTREAM_DEPTH - 1) {
            match cursor {
                PackValue::List(items) => {
                    assert_eq!(items.len(), 1, "list arity at this level");
                    cursor = &items[0];
                }
                other => panic!("expected nested list, got {other:?}"),
            }
        }
        assert_eq!(cursor, &PackValue::Integer(7));
    }

    #[test]
    fn decode_rejects_depth_through_map_path() {
        // The depth gate must fire on the MAP recursion path too —
        // not just LIST. Compose MAX_PACKSTREAM_DEPTH + 1 nested
        // TINY_MAP(1) markers; each has a 1-char key (0x81 'k') then
        // the next map.
        let mut bytes = Vec::new();
        for _ in 0..=MAX_PACKSTREAM_DEPTH {
            bytes.push(TINY_MAP_BASE | 1); // TINY_MAP arity 1
            bytes.push(TINY_STRING_BASE | 1); // TINY_STRING len 1
            bytes.push(b'k');
        }
        bytes.push(M_NULL); // leaf value for the innermost map
        let err = decode(&bytes, 0).unwrap_err();
        assert!(
            matches!(err, PackError::DepthExceeded { .. }),
            "expected DepthExceeded, got {err:?}"
        );
    }

    #[test]
    fn decode_rejects_depth_through_struct_path() {
        // STRUCT is the third recursion path — pin it too. Compose
        // MAX_PACKSTREAM_DEPTH + 1 nested TINY_STRUCT(1, tag=0x55)
        // markers.
        let mut bytes = Vec::new();
        for _ in 0..=MAX_PACKSTREAM_DEPTH {
            bytes.push(TINY_STRUCT_BASE | 1); // TINY_STRUCT arity 1
            bytes.push(0x55); // arbitrary tag
        }
        bytes.push(M_NULL); // leaf field for the innermost struct
        let err = decode(&bytes, 0).unwrap_err();
        assert!(
            matches!(err, PackError::DepthExceeded { .. }),
            "expected DepthExceeded, got {err:?}"
        );
    }

    #[test]
    fn decode_rejects_depth_through_heterogeneous_chain() {
        // W14-retro IR R1 NIT-1 pin: the depth gate must fire on a
        // MIXED-PATH chain — STRUCT → MAP → LIST → STRUCT → MAP → LIST
        // → ... — not just the uniform LIST / MAP / STRUCT chains pinned
        // by the 3 sister tests above. Real Bolt traffic mixes container
        // types (a Record STRUCT containing a property MAP whose values
        // are LISTs of refs, etc.); a gate that fires on a uniform LIST
        // chain but holes on heterogeneous recursion would be a silent
        // hazard.
        //
        // Each cycle (STRUCT(1, tag) → MAP(1, key='k') → LIST(1))
        // advances the recursion depth by 3 (STRUCT at depth N → MAP at
        // N+1 → LIST at N+2; LIST's item is at N+3). 22 cycles cover 66
        // nesting levels — the gate fires at depth 64 inside cycle 22
        // when `decode_struct_body` recurses into its field.
        let mut bytes = Vec::new();
        for _ in 0..22 {
            bytes.push(TINY_STRUCT_BASE | 1); // STRUCT(1)
            bytes.push(0x55); //   tag
            bytes.push(TINY_MAP_BASE | 1); // MAP(1)
            bytes.push(TINY_STRING_BASE | 1); //   key: STRING(1)
            bytes.push(b'k'); //   key byte
            bytes.push(TINY_LIST_BASE | 1); //   value: LIST(1) → recurses
        }
        let err = decode(&bytes, 0).unwrap_err();
        match err {
            PackError::DepthExceeded { depth, max } => {
                assert_eq!(max, MAX_PACKSTREAM_DEPTH);
                assert_eq!(depth, MAX_PACKSTREAM_DEPTH);
            }
            other => panic!("expected DepthExceeded on heterogeneous chain, got {other:?}"),
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // #577 — untrusted length-prefix bounds gate (W28-S4
    // `bolt_packstream_fuzz`, #559). A LIST_32 / MAP_32 marker can encode
    // a `len` up to u32::MAX; before the upfront gate, `decode_list_body`'s
    // `Vec::with_capacity(len)` attempted a ≈128 GiB allocation and the
    // global allocator aborted the dispatcher process on adversarial input
    // (an abort, not a catchable unwind). These pins assert the STRUCTURED
    // error (not merely "no panic"), and the sister boundary pins prove the
    // gate uses `>` (does not over-reject the exactly-fitting case).
    //
    // [#594 R1 H-1] The per-level `len > remaining` gate does NOT bound the
    // AGGREGATE pre-alloc across nested levels: 64 levels each declaring
    // `len ≈ remaining` sum to a multi-GiB `with_capacity` (depth×width)
    // that OOM-aborts, even though every level individually fits `remaining`
    // and stays under the depth cap. `LIST_PREALLOC_CAP` closes that
    // product; `decode_64_deep_list32_chain_hits_depth_gate_without_oom`
    // below is its guard (OOM-aborts on the pre-cap code; descends cheaply
    // to the depth gate post-cap).
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn decode_64_deep_list32_chain_hits_depth_gate_without_oom() {
        // [#594 R1 H-1] Depth×width pre-alloc OOM regression. Build a chain
        // of MAX_PACKSTREAM_DEPTH (64) nested LIST_32 markers, each declaring
        // a LARGE `len` (≈ message size) so the per-level `len > remaining`
        // gate PASSES at every level. Pre-cap, `decode_list_body` called
        // `Vec::with_capacity(len)` at EACH of the 64 live recursive frames
        // simultaneously (the `items` Vec is allocated before recursing into
        // element 0), summing to ≈ 64 × len × size_of::<PackValue>() — tens
        // of GiB for a 16 MiB message → `handle_alloc_error` ABORTS the
        // process (an abort, not a catchable unwind). Neither existing gate
        // caught it: the depth gate sees each level well under 64, and the
        // per-level byte gate sees each level individually fit `remaining` —
        // only the PRODUCT OOMs. A `max_len = 4096` fuzz corpus (#559) cannot
        // reach the message size that triggers it, so it would go FALSE-GREEN.
        //
        // Post-cap (`len.min(LIST_PREALLOC_CAP)`), each level pre-allocs at
        // most 4096 slots, so the chain descends cheaply (peak pre-alloc
        // ≤ 64 × 4096 × size_of::<PackValue>() ≈ 8 MiB, INDEPENDENT of message
        // size) to the depth gate and returns DepthExceeded. This test
        // OOM-ABORTS on the pre-cap code and PASSES here — it is the real
        // guard for the cap.
        const MSG_LEN: usize = 1 << 24; // 16 MiB on-wire message
        const HEADER_BYTES: usize = 5 * MAX_PACKSTREAM_DEPTH; // 64 × LIST_32 header (5 B)
        // Declared len = bytes after the deepest (tightest) header, so the
        // `len > remaining` gate passes at every level (remaining only shrinks
        // with depth). A large len → a large pre-cap `with_capacity`.
        let declared_len = (MSG_LEN - HEADER_BYTES) as u32;
        let mut bytes = vec![0u8; MSG_LEN];
        for level in 0..MAX_PACKSTREAM_DEPTH {
            let off = level * 5;
            bytes[off] = M_LIST32;
            bytes[off + 1..off + 5].copy_from_slice(&declared_len.to_be_bytes());
        }
        // Bytes [HEADER_BYTES..MSG_LEN] are never read: decode hits the depth
        // gate at depth == MAX_PACKSTREAM_DEPTH before consuming them. They
        // exist only so `remaining` is large enough that the big `declared_len`
        // clears the per-level byte gate (the tail of the ≤ 16 MiB message an
        // attacker need not actually supply past the headers).
        let err = decode(&bytes, 0).unwrap_err();
        match err {
            PackError::DepthExceeded { depth, max } => {
                assert_eq!(max, MAX_PACKSTREAM_DEPTH);
                assert_eq!(depth, MAX_PACKSTREAM_DEPTH);
            }
            other => panic!(
                "64-deep large-len LIST_32 chain must descend cheaply to the \
                 depth gate (DepthExceeded), not OOM on per-level pre-alloc; \
                 got {other:?}"
            ),
        }
    }

    #[test]
    fn decode_rejects_list32_length_prefix_exceeding_remaining() {
        // LIST_32 declares u32::MAX items but supplies zero body bytes.
        // Pre-fix: `Vec::with_capacity(u32::MAX)` → ≈128 GiB alloc → abort.
        // Post-fix: the `len > remaining` gate returns UnexpectedEof before
        // allocating anything.
        //
        // [#594 R1 M-1] Assert the GATE-SPECIFIC offset (`header_end + len`,
        // computed with the gate's own `saturating_add` so it is correct on
        // 32-bit `usize` too), not merely "some UnexpectedEof". This is the
        // full DECLARED requirement the gate reports and is strictly stronger:
        // the pre-gate truncated-body path would surface the small loop-EOF
        // offset (≈ header_end), so a relaxed `UnexpectedEof(_)` oracle could
        // not tell the gate firing apart from an ordinary mid-element EOF.
        let mut bytes = vec![M_LIST32];
        bytes.extend_from_slice(&u32::MAX.to_be_bytes());
        // (no element bytes follow)
        let header_end = 5usize; // 1 marker byte + 4 LIST_32 length bytes
        let declared_len = u32::MAX as usize;
        let expected_off = header_end.saturating_add(declared_len); // gate's `cur + len`
        let err = decode(&bytes, 0).unwrap_err();
        assert!(
            matches!(err, PackError::UnexpectedEof(off) if off == expected_off),
            "expected UnexpectedEof({expected_off}) (header_end + len) for \
             over-long LIST_32 length-prefix, got {err:?}"
        );
    }

    #[test]
    fn decode_rejects_map32_length_prefix_exceeding_remaining() {
        // Sister-pin for the MAP_32 path. A Map of u32::MAX entries needs
        // ≥ 2·u32::MAX bytes; with zero body bytes the `2·len > remaining`
        // gate rejects upfront rather than spinning the decode loop.
        //
        // [#594 R1 M-1] Gate-specific offset oracle: `header_end + 2·len`
        // (the declared minimum byte requirement), mirroring the gate's
        // `cur.saturating_add(len.saturating_mul(2))` so it is correct on
        // 32-bit `usize` too — strictly stronger than `UnexpectedEof(_)`.
        let mut bytes = vec![M_MAP32];
        bytes.extend_from_slice(&u32::MAX.to_be_bytes());
        let header_end = 5usize; // 1 marker byte + 4 MAP_32 length bytes
        let declared_len = u32::MAX as usize;
        let expected_off = header_end.saturating_add(declared_len.saturating_mul(2));
        let err = decode(&bytes, 0).unwrap_err();
        assert!(
            matches!(err, PackError::UnexpectedEof(off) if off == expected_off),
            "expected UnexpectedEof({expected_off}) (header_end + 2·len) for \
             over-long MAP_32 length-prefix, got {err:?}"
        );
    }

    #[test]
    fn decode_rejects_list8_with_fewer_body_bytes_than_len() {
        // Realistic fuzz shape (mirrors `decode_rejects_truncated_string`):
        // a LIST_8 claims 10 items but only 3 body bytes follow. The gate
        // rejects (10 > 3 remaining) without allocating/looping.
        //
        // [#594 R1 M-1] Gate-specific offset oracle: `header_end + len`
        // (header_end = 2 for LIST_8: 1 marker + 1 length byte) — the
        // declared requirement, stronger than a bare `UnexpectedEof(_)`.
        let bytes = [M_LIST8, 0x0A, 0x01, 0x02, 0x03];
        let header_end = 2usize; // 1 marker byte + 1 LIST_8 length byte
        let declared_len = 0x0Ausize; // 10 items declared
        let expected_off = header_end.saturating_add(declared_len);
        let err = decode(&bytes, 0).unwrap_err();
        assert!(
            matches!(err, PackError::UnexpectedEof(off) if off == expected_off),
            "expected UnexpectedEof({expected_off}) (header_end + len) for \
             truncated LIST_8 body, got {err:?}"
        );
    }

    #[test]
    fn decode_accepts_list_at_exact_remaining_boundary() {
        // Off-by-one guard: a LIST_8 of 3 one-byte TINY_INT items where
        // remaining bytes == len. The gate (`len > remaining`) MUST accept
        // the exactly-fitting case (uses `>`, not `>=`).
        let bytes = [M_LIST8, 0x03, 0x01, 0x02, 0x03];
        let (v, n) = decode(&bytes, 0).expect("exact-fit list must decode");
        assert_eq!(n, bytes.len(), "consumed all bytes");
        assert_eq!(
            v,
            PackValue::List(vec![
                PackValue::Integer(1),
                PackValue::Integer(2),
                PackValue::Integer(3),
            ])
        );
    }

    #[test]
    fn decode_accepts_map_at_exact_entry_byte_boundary() {
        // Off-by-one guard for the map gate: a MAP_8 of 1 entry =
        // empty-string key (0x80, 1 byte) + Null value (0xC0, 1 byte) =
        // exactly 2 bytes, so remaining == 2·len. The `2·len > remaining`
        // gate MUST accept this exactly-fitting minimal entry.
        let bytes = [M_MAP8, 0x01, TINY_STRING_BASE, M_NULL];
        let (v, n) = decode(&bytes, 0).expect("exact-fit map must decode");
        assert_eq!(n, bytes.len(), "consumed all bytes");
        let mut expected = BTreeMap::new();
        expected.insert(String::new(), PackValue::Null);
        assert_eq!(v, PackValue::Map(expected));
    }
}
