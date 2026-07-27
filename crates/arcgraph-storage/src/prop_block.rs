//! v2 M2 — the typed inline property block codec (`PropBlockView`).
//!
//! ADR-230 row M2; `docs/design/storage-architecture-v2.md` §2.2;
//! `docs/design/m1-m2-m4-m5-impl-designs.md` §M2.1. Replaces the M1
//! slot's JSON payload with a typed binary block so point reads are
//! zero-decode (`PropBlockView::get` binary-searches ≤ 64 8-byte
//! header entries by interned `key_id` — integer compare, no parse)
//! and batch reads can materialize column-wise (§M2.4).
//!
//! # Block layout (design §M2.1, little-endian, 4-byte-aligned data)
//!
//! ```text
//! offset  size   field
//! 0       1      version (= 1)
//! 1       1      prop_count (primary header entries, ≤ 64)
//! 2       2      flags, u16 LE (bit0: has_overflow_tail; bits 1-15
//!                MUST be 0 — the high byte is reserved space)
//! 4       4      meta_check — CRC-32C, u32 LE, over the block
//!                METADATA: bytes 0..4, this field as zeros, every
//!                header entry, and the overflow tail when present.
//!                Data-region bytes are EXCLUDED (the zero-decode /
//!                bytes-touched contract; payload corruption stays an
//!                on-access fault). See `compute_block_meta_checksum`
//!                for the width-disposition record (round-3: CRC-8 was
//!                measured too narrow and widened by Director ruling).
//! 8       8×n    header entries, sorted STRICTLY ascending by key_id:
//!                  u32 key_id   — interned property key (existing InternTable)
//!                  u8  type_tag — see [`PropTag`]
//!                  u8  len      — inline payload length (0..=255)
//!                  u16 offset   — payload offset RELATIVE TO BLOCK START
//! 8+8n    var    data region: inline payloads laid out IN ENTRY
//!                ORDER, each at the NEXT 4-aligned offset, densely
//!                packed (alignment gaps only) and ending EXACTLY at
//!                the region end — placement is CANONICAL and the
//!                decoder verifies it (A1: an in-range offset/len
//!                rewrite is a loud Corrupt, never a silent redirect)
//! tail    8      overflow_ref ([`BlobRef::encode`] u64) iff flags.bit0
//! ```
//!
//! # Per-tag payload semantics
//!
//! | tag        | inline payload                                        |
//! |------------|-------------------------------------------------------|
//! | `Null`     | none (`len == 0`, `offset == 0` canonical)            |
//! | `Int64`    | 8 B `i64` LE                                          |
//! | `Float64`  | 8 B `f64` LE bit pattern                              |
//! | `Bool`     | 1 B, `0x00`/`0x01` strictly                           |
//! | `StrInline`| `len` B UTF-8 (string ≤ 255 B; `len == 0` = `""`)     |
//! | `StrRef`   | 8 B locator `{u32 off, u32 len}` into the overflow    |
//! | `Bytes`    | `len` B raw (> 255 B rejected at encode — no producer |
//! |            | exists at v1.0-α; a `BytesRef` tag lands with one)    |
//! | `Temporal` | `len` B opaque (reserved: no v1.0-α producer — the    |
//! |            | temporal family is write-rejected per ADR-152-am-02   |
//! |            | and materializes as its ISO-8601 *string* form, so    |
//! |            | the M2 encoder emits `StrInline`/`StrRef` for it; the |
//! |            | tag round-trips at the codec level for forward-compat)|
//! | `ListRef`  | `len > 0`: `len` B opaque encoded bytes inline (the   |
//! |            | nested value, ≤ 255 B — design §2.2 "into the data    |
//! |            | region"); `len == 0`: 8 B locator into the overflow   |
//! |            | (unambiguous: an encoded list is ≥ 2 B, never 0)      |
//! | `MapRef`   | same rule as `ListRef`                                |
//!
//! Nested list/map values are stored as OPAQUE encoded bytes — typed
//! at the top level, opaque below it (design §2.2). The encoding of
//! those bytes is owned by the caller (the mcp layer's JSON bridge —
//! ADR-089 §D-1 bounded-context: storage never parses JSON). Tag `0`
//! is deliberately INVALID so zeroed memory reads as loud corruption.
//!
//! # Overflow payload (one per block, design §M2.1 "overflow tail")
//!
//! Large values (> 255 B) and wide bags (> 64 props) spill to a single
//! per-block overflow payload, stored through the SAME M1 staging
//! machinery as any bag (`stage_bag`: slotted slot when small, DEC-4
//! chain when large — design §2.2 "spill to packed blob pages") and
//! addressed by the 8-byte tail ref. Layout:
//!
//! ```text
//! 0       2      magic (= 0x564F, "OV")
//! 2       2      xentry_count (properties 65+, sorted by key_id)
//! 4       4      dir_check — CRC32C over the overflow METADATA
//!                (bytes 0..4 + the whole xentry directory; data area
//!                excluded). See `compute_overflow_meta_checksum`.
//! 8       16×x   xentries: { u32 key_id, u8 type_tag, u8 pad=0,
//!                            u16 pad=0, u32 off, u32 len }
//! 8+16x   var    data area (spilled values; offsets are absolute
//!                into the overflow payload, 4-byte aligned; spilled
//!                PRIMARY values first in primary-entry order, then
//!                xentry values in directory order — xentry extents
//!                are therefore strictly sequential, verified at parse)
//! ```
//!
//! Primary-entry locators (`StrRef`, locator-form `ListRef`/`MapRef`)
//! and xentry `(off, len)` pairs both index absolutely into the
//! overflow payload. Xentries use value-type tags only (placement is
//! always `(off, len)`); `StrRef` inside an xentry is corruption.
//! **Global key ordering (A2)**: every xentry key_id is strictly
//! greater than every primary key_id — the two directories partition
//! one globally-sorted bag. [`OverflowView::parse`] takes the primary
//! max key and REJECTS a violation loudly: a cross-directory
//! duplicate would make the full read (xentry last-wins) and the
//! projected read (primary hit, overflow untouched) silently
//! disagree. A key absent from the primary header AND ≤ the max
//! primary key_id is definitively absent — the overflow is fetched
//! only when a requested key_id exceeds the primary range (design
//! §M2.3 "the overflow ref is followed lazily").
//!
//! # MVCC / CoW (design §M2.5 — the consistency anchor)
//!
//! A block is IMMUTABLE per record-version: an update encodes a new
//! block into a new slot and a new record version references it —
//! exactly the M1/DEC-4 "MVCC payload = record bytes + referenced
//! payload" model. This codec is a pure byte transform; it introduces
//! no shared mutable state and no new concurrency surface.
//!
//! # Budget (PD#5)
//!
//! - `PropBlockView::parse`: O(prop_count) integer checks over ≤ 64
//!   header entries (≤ 512 B touched), no payload reads, no
//!   allocation — plus one CRC-32C pass over the SAME metadata bytes
//!   (≤ 528 B; the `crc32c` crate uses the SSE4.2/ARMv8 CRC
//!   instructions, ≈ single-digit ns for the typical 3–8-entry bag,
//!   ≪ 0.5 µs worst-case at 64 entries; same O(metadata) class as
//!   the sweep it joins, zero allocation, no data-region reads).
//! - `get(key_id)`: binary search ≤ 64 entries ≈ ≤ 6 probes × 8 B +
//!   one bounds-checked slice — target ≪ 100 ns, zero allocation
//!   (the §4.4 slot-read envelope).
//! - encode: one O(n log n) sort (n = prop count) + O(total payload
//!   bytes) writes; no intermediate per-value allocations beyond the
//!   caller-provided [`PropValue`]s.
//!
//! # Corruption posture (build-plan §2 M2 EXIT gate 2, "loud")
//!
//! Every structural violation — unknown version, unknown/invalid tag,
//! unsorted or duplicate key_ids, out-of-range offsets/lengths,
//! NON-CANONICAL payload placement (offsets must march exactly in
//! entry order — A1), cross-directory key misorder (A2), metadata
//! checksum mismatch (A1 — catches in-range tampering that keeps the
//! structure valid, e.g. a same-width `Int64`→`Float64` retype),
//! non-canonical Bool bytes, invalid UTF-8 under `StrInline`/`StrRef`,
//! nonzero reserved bits/pads — surfaces [`PropBlockError::Corrupt`].
//! Never a silent empty/mis-typed value (the M1 JSON path's
//! warn-degrade is exactly what M2 retires). Layering: the page-grain
//! CRC32C below owns physical rot; the metadata checksums here catch
//! post-encode in-memory scribbles and keep every structural check
//! honest. Structural sweeps run FIRST (specific messages), the
//! checksum LAST (the catch-all). Payload (data-region) bytes stay
//! OUTSIDE the checksums so corruption there remains the documented
//! on-access fault and parse never touches value bytes.

use thiserror::Error;

use crate::property::BlobRef;

/// Format version byte this codec writes and accepts.
pub const PROP_BLOCK_VERSION: u8 = 1;

/// First payload byte of every typed block (== [`PROP_BLOCK_VERSION`]).
/// The mixed-store payload discriminator: an M1 legacy JSON bag always
/// begins `b'{'` (0x7B — `serde_json` object), a typed block begins
/// 0x01. Any other first byte is corruption. See the mcp read bridge.
pub const PROP_BLOCK_DISCRIMINANT: u8 = PROP_BLOCK_VERSION;

/// Max header entries in the primary block (design §M2.1 `prop_count ≤ 64`).
pub const MAX_PRIMARY_ENTRIES: usize = 64;

/// Max direct-inline payload length (`len` is a u8).
pub const MAX_INLINE_LEN: usize = 255;

/// Bytes per primary header entry.
pub const HEADER_ENTRY_SIZE: usize = 8;

/// Fixed block header size (version + prop_count + u16 flags +
/// CRC-32C meta_check). Round-3 (#1452): grew 4 → 8 to hold the
/// widened checksum — bytes 4..8 are the meta_check field; byte 3
/// (round-1's CRC-8 slot) reverted to reserved-MUST-be-zero flags
/// space. The typed format has never shipped (this branch IS the M2
/// landing), so the version byte stays 1: there is no persisted
/// 4-byte-header block anywhere to misread.
pub const BLOCK_HEADER_SIZE: usize = 8;

/// Size of the optional overflow tail ref.
pub const OVERFLOW_TAIL_SIZE: usize = 8;

/// `flags` bit0 — the block carries an overflow tail.
pub const FLAG_HAS_OVERFLOW: u16 = 0b1;

/// Overflow payload magic (`"OV"` little-endian).
pub const OVERFLOW_MAGIC: u16 = 0x564F;

/// Overflow payload fixed header size (magic + xentry_count +
/// dir_check CRC32C). The xentry directory begins here.
pub const OVERFLOW_HEADER_SIZE: usize = 8;

/// Bytes per overflow extended entry.
pub const XENTRY_SIZE: usize = 16;

/// Size of a `{u32 off, u32 len}` overflow locator payload.
const LOCATOR_SIZE: usize = 8;

// ─────────────────────────────────────────────────────────────────────
// A1 — metadata self-checksums
// ─────────────────────────────────────────────────────────────────────

/// Byte range of the block's `meta_check` field (header bytes 4..8).
const META_CHECK_RANGE: std::ops::Range<usize> = 4..8;

/// Compute the block's `meta_check` value (header bytes 4..8): CRC-32C
/// over bytes `0..4` (version, prop_count, u16 flags), four `0x00`
/// bytes in place of this field, the header entries, and the overflow
/// tail when the flags declare one. Data-region bytes are excluded by
/// construction.
///
/// # Width disposition (v2 M2 round-3, #1452 — WIDENED CRC-8 → CRC-32C)
///
/// Round-1 stamped a CRC-8/ATM (poly 0x07) into the then-reserved
/// flags-high byte; round-2 assessed widening and declined
/// (size-neutrality for the payload-parity gate). The codex re-check
/// then MEASURED the 8-bit width insufficient for the declared
/// uncrafted-scribble threat model: exhaustively enumerating all
/// 2,096,128 independent two-bit upsets of a full block's 64 key-id
/// fields, 2,378 (~1 in 881) cancelled the CRC-8 syndrome AND kept
/// strict key order AND canonical placement — clearing every
/// structural sweep and materializing values under WRONG existing
/// property names (a silent wrong read; the hot resident path reads
/// via `open_prop_trusted`, which skips the page-grain CRC32C by
/// design, so this checksum is the ONLY guard on a resident scribble).
/// Director round-3 ruling: widen to CRC-32. Disposition:
///
/// - **CRC-32C (Castagnoli, the `crc32c` crate)** — the same CRC the
///   overflow `dir_check` and the page grain already use, with
///   hardware support on x86-64/aarch64. Koopman's CRC-32C analysis
///   gives HD = 6 through 5,243 dataword bits; the worst-case covered
///   metadata span here is 8 + 64×8 + 8 = 528 B = 4,224 bits, INSIDE
///   that window — so every metadata scribble of ≤ 5 flipped bits is
///   deterministically detected. That retires both measured families:
///   the two-bit key-field population (weight 2) and the four-flip
///   same-width retype collision family (weight 4). The width gates
///   (`m2_metadata_checksum_width_gate.rs`, the collision-family gate
///   in `m2_codec_integrity_gate.rs`) MEASURE both families dead
///   rather than trusting the table; reverting the width turns them
///   RED.
/// - **The 4 bytes live in the widened header** (4 → 8; see
///   [`BLOCK_HEADER_SIZE`]): bytes 4..8 are this field, byte 3
///   reverted to reserved-MUST-be-zero flags space. The M2 EXIT
///   payload gate now pins the block at ≤ JSON + exactly this field's
///   4 bytes (the Director-ruled integrity premium).
/// - **The metadata/data boundary is UNCHANGED (round-2 was right).**
///   Payload bytes stay outside the checksum (the zero-decode
///   bytes-touched contract; the data-region locator-redirect pin in
///   `m2_codec_integrity_gate.rs`). And no unkeyed CRC of ANY width
///   resists an actor who can restamp (the forged-retype pin):
///   crafted corruption remains the encryption/AEAD layer's concern.
///   Only the WIDTH was wrong; the threat-model framing stands.
///
/// Public so integrity gates and fuzzers can build
/// recomputed-checksum tamper variants (proving the STRUCTURAL checks
/// are load-bearing independently of the checksum). Forgiving on
/// short/misdeclared inputs (ranges are clamped): shape errors are
/// the parser's job, not this helper's.
#[must_use]
pub fn compute_block_meta_checksum(block: &[u8]) -> u32 {
    if block.len() < BLOCK_HEADER_SIZE {
        return 0;
    }
    let prop_count = block[1] as usize;
    let entries_end = (BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * prop_count).min(block.len());
    let has_overflow = u16::from(block[2]) & FLAG_HAS_OVERFLOW != 0;
    let mut crc = crc32c::crc32c(&block[0..META_CHECK_RANGE.start]);
    crc = crc32c::crc32c_append(crc, &[0u8; 4]); // bytes 4..8 (this field) as zero
    crc = crc32c::crc32c_append(crc, &block[BLOCK_HEADER_SIZE..entries_end]);
    if has_overflow && block.len() >= OVERFLOW_TAIL_SIZE {
        crc = crc32c::crc32c_append(crc, &block[block.len() - OVERFLOW_TAIL_SIZE..]);
    }
    crc
}

/// Stamp the block's `meta_check` field in place. Encoder-side
/// (`build`, tail finalize/patch — the tail is INSIDE the checksum, so
/// every tail write re-stamps); public so integrity gates can build
/// recomputed-checksum tamper variants without hardcoding the field's
/// offset.
pub fn stamp_block_meta_checksum(block: &mut [u8]) {
    if block.len() >= BLOCK_HEADER_SIZE {
        let crc = compute_block_meta_checksum(block);
        block[META_CHECK_RANGE].copy_from_slice(&crc.to_le_bytes());
    }
}

/// Compute the overflow payload's `dir_check` (header bytes 4..8):
/// CRC32C over bytes `0..4` (magic + xentry_count) and the whole
/// xentry directory. Data-area bytes are excluded. Public for the
/// same recomputed-tamper gate use as
/// [`compute_block_meta_checksum`]; equally forgiving on short input.
#[must_use]
pub fn compute_overflow_meta_checksum(overflow: &[u8]) -> u32 {
    if overflow.len() < 4 {
        return 0;
    }
    let xcount = usize::from(u16::from_le_bytes([overflow[2], overflow[3]]));
    let dir_end = (OVERFLOW_HEADER_SIZE + XENTRY_SIZE * xcount).min(overflow.len());
    let crc = crc32c::crc32c(&overflow[0..4]);
    if overflow.len() < OVERFLOW_HEADER_SIZE {
        return crc;
    }
    crc32c::crc32c_append(crc, &overflow[OVERFLOW_HEADER_SIZE..dir_end])
}

/// Property value type tags (design §M2.1). Tag `0` is deliberately
/// unassigned: zeroed memory decodes as LOUD corruption, never as a
/// value. Discriminants are the on-disk bytes — never renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PropTag {
    /// JSON `null` — a storable property value in the existing bag
    /// contract (the `graph.ingest` JSON path stores and round-trips
    /// nulls today). Carries no payload. Required for the M2 EXIT
    /// gate-5 differential ("identical materialized values for every
    /// projection" vs the M1 JSON store).
    Null = 1,
    /// `i64`, 8 B LE.
    Int64 = 2,
    /// `f64`, 8 B LE bit pattern. The encoder normalizes non-finite
    /// floats to [`PropTag::Null`] BEFORE this tag is chosen (matching
    /// the M1 JSON bridge: NaN/±Inf → JSON null → `Value::Null`).
    Float64 = 3,
    /// 1 B, strictly `0x00` / `0x01`.
    Bool = 4,
    /// UTF-8 string ≤ 255 B, inline.
    StrInline = 5,
    /// UTF-8 string > 255 B; inline payload is an 8-B overflow locator.
    StrRef = 6,
    /// Raw bytes ≤ 255 B, inline (reserved: no v1.0-α producer).
    Bytes = 7,
    /// Opaque temporal encoding ≤ 255 B (reserved: no v1.0-α producer;
    /// see the module-docs table).
    Temporal = 8,
    /// Nested list, opaque encoded bytes: inline when `len > 0`,
    /// overflow locator when `len == 0`.
    ListRef = 9,
    /// Nested map, same placement rule as [`PropTag::ListRef`].
    MapRef = 10,
}

impl PropTag {
    /// Decode a tag byte. Unknown bytes (including 0) are corruption.
    pub fn from_byte(b: u8) -> Result<Self, PropBlockError> {
        Ok(match b {
            1 => Self::Null,
            2 => Self::Int64,
            3 => Self::Float64,
            4 => Self::Bool,
            5 => Self::StrInline,
            6 => Self::StrRef,
            7 => Self::Bytes,
            8 => Self::Temporal,
            9 => Self::ListRef,
            10 => Self::MapRef,
            other => {
                return Err(PropBlockError::Corrupt {
                    reason: format!("unknown property type_tag byte {other:#04x}"),
                });
            }
        })
    }

    /// The on-disk byte.
    #[must_use]
    pub fn as_byte(self) -> u8 {
        self as u8
    }
}

/// An owned property value at the storage grain. Nested list/map
/// values are OPAQUE pre-encoded bytes (the mcp layer owns that
/// encoding — ADR-089 §D-1: storage never parses JSON).
#[derive(Debug, Clone, PartialEq)]
pub enum PropValue {
    /// JSON null (see [`PropTag::Null`]).
    Null,
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit float. Encoding a non-finite value stores [`PropValue::Null`]
    /// (the M1 JSON-bridge equivalence — see [`PropTag::Float64`]).
    Float(f64),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string (any length; > 255 B spills to overflow).
    Str(String),
    /// Raw bytes (reserved; ≤ 255 B at this codec version).
    Bytes(Vec<u8>),
    /// Opaque temporal encoding (reserved; ≤ 255 B at this codec version).
    Temporal(Vec<u8>),
    /// Nested list as opaque encoded bytes (any length).
    ListOpaque(Vec<u8>),
    /// Nested map as opaque encoded bytes (any length).
    MapOpaque(Vec<u8>),
}

/// A borrowed property value read out of a block / overflow payload —
/// the zero-copy view grain (design §4.2: values are borrows from the
/// page bytes; escaping copies happen at the operator boundary, OQ-5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropValueRef<'a> {
    /// JSON null.
    Null,
    /// 64-bit signed integer (copied — scalar).
    Int(i64),
    /// 64-bit float (copied — scalar).
    Float(f64),
    /// Boolean (copied — scalar).
    Bool(bool),
    /// Borrowed UTF-8 string.
    Str(&'a str),
    /// Borrowed raw bytes (reserved).
    Bytes(&'a [u8]),
    /// Borrowed opaque temporal bytes (reserved).
    Temporal(&'a [u8]),
    /// Borrowed opaque nested-list bytes.
    ListOpaque(&'a [u8]),
    /// Borrowed opaque nested-map bytes.
    MapOpaque(&'a [u8]),
}

impl PropValueRef<'_> {
    /// Copy the borrow out into an owned [`PropValue`] (the operator-
    /// boundary escape per design OQ-5).
    #[must_use]
    pub fn to_owned_value(&self) -> PropValue {
        match *self {
            PropValueRef::Null => PropValue::Null,
            PropValueRef::Int(i) => PropValue::Int(i),
            PropValueRef::Float(f) => PropValue::Float(f),
            PropValueRef::Bool(b) => PropValue::Bool(b),
            PropValueRef::Str(s) => PropValue::Str(s.to_owned()),
            PropValueRef::Bytes(b) => PropValue::Bytes(b.to_vec()),
            PropValueRef::Temporal(b) => PropValue::Temporal(b.to_vec()),
            PropValueRef::ListOpaque(b) => PropValue::ListOpaque(b.to_vec()),
            PropValueRef::MapOpaque(b) => PropValue::MapOpaque(b.to_vec()),
        }
    }
}

/// Decode-side faults. LOUD by design: a structurally invalid typed
/// block is corruption, never an empty bag (build-plan §2 M2 EXIT 2).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PropBlockError {
    /// The block / overflow bytes violate the format contract.
    #[error("corrupt typed property block: {reason}")]
    Corrupt {
        /// What was violated (loud, specific — reviewers reject vague).
        reason: String,
    },
}

impl PropBlockError {
    fn corrupt(reason: impl Into<String>) -> Self {
        Self::Corrupt {
            reason: reason.into(),
        }
    }
}

/// Encode-side faults.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PropBlockEncodeError {
    /// A reserved-tag value exceeds the inline cap and has no overflow
    /// representation at this codec version (no v1.0-α producer exists;
    /// see the module docs' `Bytes` / `Temporal` rows).
    #[error(
        "cannot encode {kind} value of {len} bytes: exceeds the {MAX_INLINE_LEN}-byte inline \
         cap and PROP_BLOCK_VERSION={PROP_BLOCK_VERSION} defines no overflow form for it"
    )]
    ReservedValueTooLarge {
        /// `"Bytes"` or `"Temporal"`.
        kind: &'static str,
        /// The oversize length.
        len: usize,
    },
    /// Internal invariant breach (offsets exceeded u16 / u32 range).
    /// Unreachable by construction — surfaced instead of panicking.
    #[error("typed property block layout overflow (bug): {reason}")]
    LayoutOverflow {
        /// Which bound was exceeded.
        reason: String,
    },
}

// ─────────────────────────────────────────────────────────────────────
// Encoder
// ─────────────────────────────────────────────────────────────────────

/// Where one property's payload was placed by the planner pass.
enum Placement {
    /// `len == 0` payloads (Null; empty string/bytes/temporal).
    Empty { tag: PropTag },
    /// Direct inline bytes in the primary data region.
    Inline { tag: PropTag, bytes: PayloadBytes },
    /// 8-B locator in the primary data region → overflow data.
    OverflowLocator { tag: PropTag, bytes: Vec<u8> },
}

/// Inline payload bytes without heap-allocating scalars.
enum PayloadBytes {
    Scalar8([u8; 8]),
    Bool([u8; 1]),
    Owned(Vec<u8>),
}

impl PayloadBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            PayloadBytes::Scalar8(b) => b,
            PayloadBytes::Bool(b) => b,
            PayloadBytes::Owned(v) => v,
        }
    }
}

/// The encoder's output: the block body (with the tail ref slot still
/// zeroed when an overflow exists) plus the overflow payload to stage.
///
/// Two-phase by necessity: the tail stores the overflow payload's
/// [`BlobRef`], which the caller only learns AFTER staging the
/// overflow bytes through the M1 machinery. Flow:
///
/// 1. `PropBlockBuilder::build()` → `EncodedPropBlock`
/// 2. if `overflow_payload()` is `Some`, stage it (`stage_bag`) → `BlobRef`
/// 3. `into_block_bytes(Some(ref))` → the final block bytes to stage
#[derive(Debug)]
pub struct EncodedPropBlock {
    body: Vec<u8>,
    /// Byte offset of the 8-B tail ref inside `body`, when present.
    tail_at: Option<usize>,
    overflow: Option<Vec<u8>>,
}

impl EncodedPropBlock {
    /// The overflow payload the caller must stage first, if any.
    #[must_use]
    pub fn overflow_payload(&self) -> Option<&[u8]> {
        self.overflow.as_deref()
    }

    /// Finalize the block bytes. `overflow_ref` MUST be `Some` iff
    /// [`Self::overflow_payload`] was `Some` (the staged payload's ref).
    pub fn into_block_bytes(
        mut self,
        overflow_ref: Option<BlobRef>,
    ) -> Result<Vec<u8>, PropBlockEncodeError> {
        match (self.tail_at, overflow_ref) {
            (None, None) => Ok(self.body),
            (Some(at), Some(bref)) => {
                self.body[at..at + OVERFLOW_TAIL_SIZE]
                    .copy_from_slice(&bref.encode().to_le_bytes());
                // The tail is inside the metadata checksum — re-stamp.
                stamp_block_meta_checksum(&mut self.body);
                Ok(self.body)
            }
            (Some(_), None) => Err(PropBlockEncodeError::LayoutOverflow {
                reason: "block has an overflow payload but no overflow_ref was supplied".into(),
            }),
            (None, Some(_)) => Err(PropBlockEncodeError::LayoutOverflow {
                reason: "overflow_ref supplied but the block has no overflow payload".into(),
            }),
        }
    }

    /// DEFERRED finalize: return the block bytes with the overflow
    /// tail slot (when present) left ZEROED, for a stager that learns
    /// the overflow payload's [`BlobRef`] later — inside the owning
    /// transaction — and patches it via [`patch_overflow_tail`]. (The
    /// two-phase write-path flow: the mcp encoder builds the pair, the
    /// storage layer stages overflow-then-block atomically in one
    /// commit bundle.)
    pub fn into_block_bytes_deferred(self) -> Result<Vec<u8>, PropBlockEncodeError> {
        Ok(self.body)
    }
}

/// A built typed bag ready for two-phase staging: the overflow
/// payload (if any) is staged FIRST inside the owning transaction
/// (its [`BlobRef`] patches the block's tail via
/// [`patch_overflow_tail`]), then the block itself is staged. Both
/// ride the same commit bundle — atomicity is the transaction's.
///
/// Produced by the mcp encode bridge (`build_typed_bag` /
/// `reencode_json_bag_to_typed`); consumed by
/// `crud::PropertyData::TypedBlock` and the M2 migrate-on-open sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedBagParts {
    /// The encoded block. When `overflow` is `Some`, the trailing 8
    /// bytes are the zeroed tail slot the stager patches.
    pub block: Vec<u8>,
    /// The overflow payload to stage first, if the bag spilled.
    pub overflow: Option<Vec<u8>>,
}

/// Patch a deferred block's zeroed overflow tail slot with the staged
/// overflow payload's [`BlobRef`] (see
/// [`EncodedPropBlock::into_block_bytes_deferred`]).
///
/// Validates the block SHAPE before writing: version byte, the
/// overflow flag actually set, and a length that holds the tail —
/// writing a ref into a block that declares no overflow would corrupt
/// a data-region byte silently, so this is loud instead.
pub fn patch_overflow_tail(block: &mut [u8], bref: BlobRef) -> Result<(), PropBlockEncodeError> {
    let shape_err = |reason: String| PropBlockEncodeError::LayoutOverflow { reason };
    if block.len() < BLOCK_HEADER_SIZE + OVERFLOW_TAIL_SIZE {
        return Err(shape_err(format!(
            "block of {} bytes cannot carry an overflow tail",
            block.len()
        )));
    }
    if block[0] != PROP_BLOCK_VERSION {
        return Err(shape_err(format!(
            "patch_overflow_tail on non-typed-block bytes (first byte {:#04x})",
            block[0]
        )));
    }
    let flags = u16::from(block[2]);
    if flags & FLAG_HAS_OVERFLOW == 0 {
        return Err(shape_err(
            "patch_overflow_tail on a block whose overflow flag is clear".to_string(),
        ));
    }
    let at = block.len() - OVERFLOW_TAIL_SIZE;
    block[at..].copy_from_slice(&bref.encode().to_le_bytes());
    // The tail is inside the metadata checksum — re-stamp (A1).
    stamp_block_meta_checksum(block);
    Ok(())
}

/// Builds one typed property block from `(key_id, value)` pairs.
/// Duplicate key_ids resolve last-wins (mirroring the M1 JSON map
/// insert semantics). Output is byte-deterministic for a given input
/// set (global key_id sort; canonical placement order) — the
/// migration's resume-byte-identity leg depends on this.
#[derive(Debug, Default)]
pub struct PropBlockBuilder {
    entries: std::collections::BTreeMap<u32, PropValue>,
}

impl PropBlockBuilder {
    /// Empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one property. Last write per `key_id` wins.
    pub fn put(&mut self, key_id: u32, value: PropValue) -> &mut Self {
        self.entries.insert(key_id, value);
        self
    }

    /// Number of staged properties.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no properties are staged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Encode. See [`EncodedPropBlock`] for the two-phase overflow flow.
    pub fn build(self) -> Result<EncodedPropBlock, PropBlockEncodeError> {
        // Pass 1 — placement planning. BTreeMap iteration = key_id
        // ascending (the sorted-header invariant for free).
        let n_total = self.entries.len();
        let n_primary = n_total.min(MAX_PRIMARY_ENTRIES);

        let mut primary: Vec<(u32, Placement)> = Vec::with_capacity(n_primary);
        // (key_id, tag, value bytes) for props 65+ — all placed in the
        // overflow data area via xentries.
        let mut xprops: Vec<(u32, PropTag, Vec<u8>)> = Vec::with_capacity(n_total - n_primary);

        for (idx, (key_id, value)) in self.entries.into_iter().enumerate() {
            // The encoder normalizes non-finite floats to Null BEFORE
            // tag selection: the M1 JSON bridge maps NaN/±Inf → JSON
            // null → `Value::Null` on read, and the M2 EXIT gate-5
            // differential pins identical materialization.
            let value = match value {
                PropValue::Float(f) if !f.is_finite() => PropValue::Null,
                v => v,
            };
            if idx < MAX_PRIMARY_ENTRIES {
                primary.push((key_id, Self::place_primary(value)?));
            } else {
                let (tag, bytes) = Self::xentry_encode(value)?;
                xprops.push((key_id, tag, bytes));
            }
        }

        // Pass 2 — overflow payload assembly (spilled primary values
        // first, then xentry values; all offsets absolute + 4-aligned).
        let has_xentries = !xprops.is_empty();
        let spilled_primary: usize = primary
            .iter()
            .filter(|(_, p)| matches!(p, Placement::OverflowLocator { .. }))
            .count();
        let has_overflow = has_xentries || spilled_primary > 0;

        let mut overflow: Option<Vec<u8>> = None;
        // (off, len) locators for spilled primary values, in primary order.
        let mut primary_locators: Vec<(u32, u32)> = Vec::with_capacity(spilled_primary);

        if has_overflow {
            let xcount = xprops.len();
            let data_start = OVERFLOW_HEADER_SIZE + XENTRY_SIZE * xcount;
            let mut buf = vec![0u8; data_start];
            buf[0..2].copy_from_slice(&OVERFLOW_MAGIC.to_le_bytes());
            let xcount_u16 =
                u16::try_from(xcount).map_err(|_| PropBlockEncodeError::LayoutOverflow {
                    reason: format!("xentry_count {xcount} exceeds u16"),
                })?;
            buf[2..4].copy_from_slice(&xcount_u16.to_le_bytes());
            // bytes 4..8 = dir_check, stamped after the directory is
            // assembled (below).

            // Spilled primary values. Empty payloads canonicalize to
            // the (0, 0) locator (nothing is written; the decoder
            // yields an empty slice) — the zero-extent canonical form.
            for (_, placement) in &primary {
                if let Placement::OverflowLocator { bytes, .. } = placement {
                    let off = if bytes.is_empty() {
                        0
                    } else {
                        append_aligned(&mut buf, bytes)
                    };
                    primary_locators.push((
                        u32::try_from(off).map_err(|_| PropBlockEncodeError::LayoutOverflow {
                            reason: "overflow offset exceeds u32".into(),
                        })?,
                        u32::try_from(bytes.len()).map_err(|_| {
                            PropBlockEncodeError::LayoutOverflow {
                                reason: "overflow value length exceeds u32".into(),
                            }
                        })?,
                    ));
                }
            }

            // Xentry values + directory. Empty payloads canonicalize
            // to (0, 0) as above.
            for (i, (key_id, tag, bytes)) in xprops.iter().enumerate() {
                let off = if bytes.is_empty() {
                    0
                } else {
                    append_aligned(&mut buf, bytes)
                };
                let e = OVERFLOW_HEADER_SIZE + XENTRY_SIZE * i;
                buf[e..e + 4].copy_from_slice(&key_id.to_le_bytes());
                buf[e + 4] = tag.as_byte();
                // bytes e+5..e+8 stay zero (pads, validated on decode).
                let off_u32 =
                    u32::try_from(off).map_err(|_| PropBlockEncodeError::LayoutOverflow {
                        reason: "overflow offset exceeds u32".into(),
                    })?;
                let len_u32 = u32::try_from(bytes.len()).map_err(|_| {
                    PropBlockEncodeError::LayoutOverflow {
                        reason: "overflow value length exceeds u32".into(),
                    }
                })?;
                buf[e + 8..e + 12].copy_from_slice(&off_u32.to_le_bytes());
                buf[e + 12..e + 16].copy_from_slice(&len_u32.to_le_bytes());
            }
            // A1 — stamp the directory checksum (data area excluded,
            // so appended values after this point cannot invalidate it
            // — and none are: the directory is complete here).
            let dir_check = compute_overflow_meta_checksum(&buf);
            buf[4..8].copy_from_slice(&dir_check.to_le_bytes());
            overflow = Some(buf);
        }

        // Pass 3 — primary block assembly.
        let n_primary_u8 =
            u8::try_from(n_primary).map_err(|_| PropBlockEncodeError::LayoutOverflow {
                reason: format!("primary entry count {n_primary} exceeds u8"),
            })?;
        let entries_end = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * n_primary;
        let mut body = vec![0u8; entries_end];
        body[0] = PROP_BLOCK_VERSION;
        body[1] = n_primary_u8;
        let flags: u16 = if has_overflow { FLAG_HAS_OVERFLOW } else { 0 };
        body[2..4].copy_from_slice(&flags.to_le_bytes());

        let mut locator_iter = primary_locators.into_iter();
        for (i, (key_id, placement)) in primary.iter().enumerate() {
            let (tag, len, offset) = match placement {
                Placement::Empty { tag } => (*tag, 0u8, 0u16),
                Placement::Inline { tag, bytes } => {
                    let payload = bytes.as_slice();
                    let off = append_aligned(&mut body, payload);
                    let off_u16 =
                        u16::try_from(off).map_err(|_| PropBlockEncodeError::LayoutOverflow {
                            reason: "block payload offset exceeds u16".into(),
                        })?;
                    // len ≤ MAX_INLINE_LEN by placement construction.
                    let len_u8 = u8::try_from(payload.len()).map_err(|_| {
                        PropBlockEncodeError::LayoutOverflow {
                            reason: "inline payload exceeds u8 length".into(),
                        }
                    })?;
                    (*tag, len_u8, off_u16)
                }
                Placement::OverflowLocator { tag, .. } => {
                    let (off, len) = locator_iter
                        .next()
                        .expect("locator count matches spilled-primary count by construction");
                    let mut loc = [0u8; LOCATOR_SIZE];
                    loc[0..4].copy_from_slice(&off.to_le_bytes());
                    loc[4..8].copy_from_slice(&len.to_le_bytes());
                    let at = append_aligned(&mut body, &loc);
                    let at_u16 =
                        u16::try_from(at).map_err(|_| PropBlockEncodeError::LayoutOverflow {
                            reason: "block payload offset exceeds u16".into(),
                        })?;
                    // StrRef carries len == 8 (the locator itself);
                    // locator-form ListRef/MapRef carry len == 0 (the
                    // module-docs disambiguation rule).
                    let len_u8 = match tag {
                        PropTag::StrRef => LOCATOR_SIZE as u8,
                        PropTag::ListRef | PropTag::MapRef => 0,
                        other => {
                            return Err(PropBlockEncodeError::LayoutOverflow {
                                reason: format!("tag {other:?} cannot take the locator form"),
                            });
                        }
                    };
                    (*tag, len_u8, at_u16)
                }
            };
            let e = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * i;
            body[e..e + 4].copy_from_slice(&key_id.to_le_bytes());
            body[e + 4] = tag.as_byte();
            body[e + 5] = len;
            body[e + 6..e + 8].copy_from_slice(&offset.to_le_bytes());
        }

        let tail_at = if has_overflow {
            let at = body.len();
            body.extend_from_slice(&[0u8; OVERFLOW_TAIL_SIZE]);
            Some(at)
        } else {
            None
        };

        // A1 — stamp the metadata checksum (over the still-zeroed tail
        // when deferred; every later tail write re-stamps).
        stamp_block_meta_checksum(&mut body);

        Ok(EncodedPropBlock {
            body,
            tail_at,
            overflow,
        })
    }

    /// Placement decision for a primary-block property.
    fn place_primary(value: PropValue) -> Result<Placement, PropBlockEncodeError> {
        Ok(match value {
            PropValue::Null => Placement::Empty { tag: PropTag::Null },
            PropValue::Int(i) => Placement::Inline {
                tag: PropTag::Int64,
                bytes: PayloadBytes::Scalar8(i.to_le_bytes()),
            },
            PropValue::Float(f) => Placement::Inline {
                tag: PropTag::Float64,
                bytes: PayloadBytes::Scalar8(f.to_le_bytes()),
            },
            PropValue::Bool(b) => Placement::Inline {
                tag: PropTag::Bool,
                bytes: PayloadBytes::Bool([u8::from(b)]),
            },
            PropValue::Str(s) => {
                if s.is_empty() {
                    Placement::Empty {
                        tag: PropTag::StrInline,
                    }
                } else if s.len() <= MAX_INLINE_LEN {
                    Placement::Inline {
                        tag: PropTag::StrInline,
                        bytes: PayloadBytes::Owned(s.into_bytes()),
                    }
                } else {
                    Placement::OverflowLocator {
                        tag: PropTag::StrRef,
                        bytes: s.into_bytes(),
                    }
                }
            }
            PropValue::Bytes(b) => Self::place_reserved_inline(PropTag::Bytes, "Bytes", b)?,
            PropValue::Temporal(b) => {
                Self::place_reserved_inline(PropTag::Temporal, "Temporal", b)?
            }
            PropValue::ListOpaque(b) => Self::place_nested(PropTag::ListRef, b),
            PropValue::MapOpaque(b) => Self::place_nested(PropTag::MapRef, b),
        })
    }

    /// Reserved tags (`Bytes` / `Temporal`): inline-only at this codec
    /// version — no producer exists, so > 255 B is an encode error, not
    /// a silent truncation.
    fn place_reserved_inline(
        tag: PropTag,
        kind: &'static str,
        b: Vec<u8>,
    ) -> Result<Placement, PropBlockEncodeError> {
        if b.is_empty() {
            Ok(Placement::Empty { tag })
        } else if b.len() <= MAX_INLINE_LEN {
            Ok(Placement::Inline {
                tag,
                bytes: PayloadBytes::Owned(b),
            })
        } else {
            Err(PropBlockEncodeError::ReservedValueTooLarge { kind, len: b.len() })
        }
    }

    /// Nested opaque values: inline ≤ 255 B, else the locator form
    /// (`len == 0` marker — an encoded list/map is ≥ 2 B, never 0, so
    /// the marker is unambiguous; enforced at decode).
    fn place_nested(tag: PropTag, b: Vec<u8>) -> Placement {
        if b.len() <= MAX_INLINE_LEN && !b.is_empty() {
            Placement::Inline {
                tag,
                bytes: PayloadBytes::Owned(b),
            }
        } else {
            // Includes the degenerate empty case: 0-byte opaque bytes
            // route through the overflow so inline `len == 0` stays an
            // unambiguous locator marker. (No producer emits 0-byte
            // nested encodings; this is belt-and-braces determinism.)
            Placement::OverflowLocator { tag, bytes: b }
        }
    }

    /// Encode one xentry (props 65+) value: `(tag, raw bytes)`.
    /// Placement is always `(off, len)` in the overflow data area, so
    /// every type is representable at any length (module docs).
    fn xentry_encode(value: PropValue) -> Result<(PropTag, Vec<u8>), PropBlockEncodeError> {
        Ok(match value {
            PropValue::Null => (PropTag::Null, Vec::new()),
            PropValue::Int(i) => (PropTag::Int64, i.to_le_bytes().to_vec()),
            PropValue::Float(f) => (PropTag::Float64, f.to_le_bytes().to_vec()),
            PropValue::Bool(b) => (PropTag::Bool, vec![u8::from(b)]),
            // Strings in xentries always carry (off, len) — the tag is
            // the value-type `StrInline` (StrRef is primary-only; see
            // the module docs' xentry rule).
            PropValue::Str(s) => (PropTag::StrInline, s.into_bytes()),
            PropValue::Bytes(b) => (PropTag::Bytes, b),
            PropValue::Temporal(b) => (PropTag::Temporal, b),
            PropValue::ListOpaque(b) => (PropTag::ListRef, b),
            PropValue::MapOpaque(b) => (PropTag::MapRef, b),
        })
    }
}

/// Append `payload` to `buf` at the next 4-byte-aligned offset
/// (design §M2.1 "data region: inline payloads, 4-byte aligned"),
/// zero-padding the gap. Returns the payload's start offset.
fn append_aligned(buf: &mut Vec<u8>, payload: &[u8]) -> usize {
    let off = buf.len().next_multiple_of(4);
    buf.resize(off, 0);
    buf.extend_from_slice(payload);
    off
}

// ─────────────────────────────────────────────────────────────────────
// Decoder — the zero-decode view
// ─────────────────────────────────────────────────────────────────────

/// Outcome of a primary-block key lookup (design §M2.3 lazy-overflow
/// contract — the caller fetches the overflow payload ONLY on
/// [`PrimaryLookup::InOverflow`] / [`PrimaryLookup::MaybeInOverflow`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrimaryLookup<'a> {
    /// Value resolved entirely from the primary block.
    Found(PropValueRef<'a>),
    /// The key exists; its value bytes live in the overflow payload at
    /// this locator. `tag` is the value-type tag to decode with.
    InOverflow {
        /// Value type.
        tag: PropTag,
        /// Absolute offset into the overflow payload.
        off: u32,
        /// Value byte length.
        len: u32,
    },
    /// The key is definitively absent (within the primary key range,
    /// or no overflow exists).
    Absent,
    /// The key exceeds the primary range and the block has xentries —
    /// only the overflow's directory can answer (wide-bag case).
    MaybeInOverflow,
}

/// Zero-copy read view over one typed property block's bytes.
///
/// `parse` validates STRUCTURE eagerly — header fields, sortedness,
/// tag bytes, offset/length ranges — touching only the ≤ 512 B of
/// header entries, never value payloads (the fixed cost every lookup
/// pays anyway via binary search). Value bytes are touched only by
/// `get`-family calls on the requested keys — the M2 EXIT gate-1
/// "point read touching K of M properties decodes only K" contract.
#[derive(Debug, Clone, Copy)]
pub struct PropBlockView<'a> {
    bytes: &'a [u8],
    prop_count: usize,
    has_overflow: bool,
    /// End of the addressable data region (block len minus the tail).
    data_end: usize,
}

impl<'a> PropBlockView<'a> {
    /// Parse + structurally validate a typed block. LOUD on any
    /// violation (never a silent empty bag — the M2 posture).
    pub fn parse(bytes: &'a [u8]) -> Result<Self, PropBlockError> {
        if bytes.len() < BLOCK_HEADER_SIZE {
            return Err(PropBlockError::corrupt(format!(
                "block too short: {} bytes < {BLOCK_HEADER_SIZE}-byte header",
                bytes.len()
            )));
        }
        if bytes[0] != PROP_BLOCK_VERSION {
            return Err(PropBlockError::corrupt(format!(
                "unknown block version {:#04x} (this engine reads version {PROP_BLOCK_VERSION:#04x})",
                bytes[0]
            )));
        }
        let prop_count = bytes[1] as usize;
        if prop_count > MAX_PRIMARY_ENTRIES {
            return Err(PropBlockError::corrupt(format!(
                "prop_count {prop_count} exceeds the {MAX_PRIMARY_ENTRIES}-entry cap"
            )));
        }
        // Bytes 2..4 are the u16 flags (bits 1-15 reserved MUST-be-0);
        // bytes 4..8 are the metadata checksum (A1) — verified LAST,
        // after the structural sweep, so structure-explainable
        // corruption keeps its specific message.
        let flags = u16::from_le_bytes([bytes[2], bytes[3]]);
        if flags & !FLAG_HAS_OVERFLOW != 0 {
            return Err(PropBlockError::corrupt(format!(
                "reserved flag bits set: {flags:#06x}"
            )));
        }
        let has_overflow = flags & FLAG_HAS_OVERFLOW != 0;
        let entries_end = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * prop_count;
        let tail = if has_overflow { OVERFLOW_TAIL_SIZE } else { 0 };
        if bytes.len() < entries_end + tail {
            return Err(PropBlockError::corrupt(format!(
                "block length {} cannot hold {prop_count} header entries{}",
                bytes.len(),
                if has_overflow { " + overflow tail" } else { "" }
            )));
        }
        let data_end = bytes.len() - tail;

        let view = Self {
            bytes,
            prop_count,
            has_overflow,
            data_end,
        };

        // Eager structural sweep over the header entries: sortedness
        // (strictly ascending — also rejects duplicates), tag validity,
        // per-entry range checks, and the A1 CANONICAL-PLACEMENT march
        // (payload extents in entry order, each at the next 4-aligned
        // offset, densely packed — an in-range offset redirect, an
        // overlap, or a gap is loud). Integer-only; no payload bytes
        // read.
        let mut prev_key: Option<u32> = None;
        let mut cursor = entries_end;
        for i in 0..prop_count {
            let (key_id, tag, len, offset) = view.entry(i)?;
            if let Some(p) = prev_key {
                if key_id <= p {
                    return Err(PropBlockError::corrupt(format!(
                        "header entries not strictly ascending by key_id at index {i} \
                         ({key_id} after {p})"
                    )));
                }
            }
            prev_key = Some(key_id);
            view.validate_entry_extent(i, tag, len, offset, &mut cursor)?;
        }
        // A1 — the data region ends EXACTLY at the last payload (the
        // encoder never emits trailing slack); a shortened extent that
        // alignment cannot absorb, or appended garbage, lands here.
        if cursor != data_end {
            return Err(PropBlockError::corrupt(format!(
                "data region ends at {data_end} but the canonical payload layout ends at \
                 {cursor} (truncated extent or trailing bytes)"
            )));
        }
        // A1 — metadata checksum LAST: the catch-all for tampering the
        // structural sweep cannot see (e.g. a same-width Int64→Float64
        // tag retype, or a two-bit key-field upset that keeps strict
        // ordering — the round-3 width family).
        let stored = u32::from_le_bytes([
            bytes[META_CHECK_RANGE.start],
            bytes[META_CHECK_RANGE.start + 1],
            bytes[META_CHECK_RANGE.start + 2],
            bytes[META_CHECK_RANGE.start + 3],
        ]);
        let expect = compute_block_meta_checksum(bytes);
        if stored != expect {
            return Err(PropBlockError::corrupt(format!(
                "block metadata checksum mismatch: stored {stored:#010x}, computed {expect:#010x}"
            )));
        }
        Ok(view)
    }

    /// Primary header entry count (props 65+ live in the overflow).
    #[must_use]
    pub fn prop_count(&self) -> usize {
        self.prop_count
    }

    /// Whether the block carries an overflow tail.
    #[must_use]
    pub fn has_overflow(&self) -> bool {
        self.has_overflow
    }

    /// The overflow payload's [`BlobRef`], when present.
    pub fn overflow_ref(&self) -> Result<Option<BlobRef>, PropBlockError> {
        if !self.has_overflow {
            return Ok(None);
        }
        let mut raw = [0u8; OVERFLOW_TAIL_SIZE];
        raw.copy_from_slice(&self.bytes[self.data_end..self.data_end + OVERFLOW_TAIL_SIZE]);
        let raw = u64::from_le_bytes(raw);
        BlobRef::decode(raw).map(Some).ok_or_else(|| {
            PropBlockError::corrupt(format!("overflow tail {raw:#018x} is not a valid BlobRef"))
        })
    }

    /// Iterate the primary header key_ids (ascending), touching only
    /// header bytes.
    pub fn key_ids(&self) -> impl Iterator<Item = u32> + 'a {
        let bytes = self.bytes;
        (0..self.prop_count).map(move |i| {
            let e = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * i;
            u32::from_le_bytes([bytes[e], bytes[e + 1], bytes[e + 2], bytes[e + 3]])
        })
    }

    /// The largest primary key_id, if any entries exist.
    #[must_use]
    pub fn max_primary_key_id(&self) -> Option<u32> {
        if self.prop_count == 0 {
            return None;
        }
        let e = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * (self.prop_count - 1);
        Some(u32::from_le_bytes([
            self.bytes[e],
            self.bytes[e + 1],
            self.bytes[e + 2],
            self.bytes[e + 3],
        ]))
    }

    /// Point lookup by interned key (design §M2.2): binary search the
    /// ≤ 64 sorted header entries — integer compare, zero allocation,
    /// value bytes touched only on a hit.
    pub fn get(&self, key_id: u32) -> Result<PrimaryLookup<'a>, PropBlockError> {
        let mut lo = 0usize;
        let mut hi = self.prop_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let e = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * mid;
            let k = u32::from_le_bytes([
                self.bytes[e],
                self.bytes[e + 1],
                self.bytes[e + 2],
                self.bytes[e + 3],
            ]);
            match k.cmp(&key_id) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return self.resolve_entry(mid),
            }
        }
        // Not in the primary header. Definitively absent unless the
        // key sorts past the primary range on a wide (xentry) bag —
        // spilled LARGE VALUES of primary keys always have a primary
        // entry, so only key_ids beyond the primary max can live
        // exclusively in the overflow directory (module docs).
        if self.has_overflow && self.max_primary_key_id().is_none_or(|max| key_id > max) {
            return Ok(PrimaryLookup::MaybeInOverflow);
        }
        Ok(PrimaryLookup::Absent)
    }

    /// Decode entry `i`'s value (or overflow locator).
    fn resolve_entry(&self, i: usize) -> Result<PrimaryLookup<'a>, PropBlockError> {
        let (_, tag, len, offset) = self.entry(i)?;
        let inline = self.entry_payload(i, tag, len, offset)?;
        match tag {
            PropTag::Null => Ok(PrimaryLookup::Found(PropValueRef::Null)),
            PropTag::Int64 => Ok(PrimaryLookup::Found(PropValueRef::Int(i64::from_le_bytes(
                scalar8(inline),
            )))),
            PropTag::Float64 => Ok(PrimaryLookup::Found(PropValueRef::Float(
                f64::from_le_bytes(scalar8(inline)),
            ))),
            PropTag::Bool => match inline[0] {
                0 => Ok(PrimaryLookup::Found(PropValueRef::Bool(false))),
                1 => Ok(PrimaryLookup::Found(PropValueRef::Bool(true))),
                other => Err(PropBlockError::corrupt(format!(
                    "Bool payload byte {other:#04x} is not 0x00/0x01"
                ))),
            },
            PropTag::StrInline => {
                let s = std::str::from_utf8(inline).map_err(|e| {
                    PropBlockError::corrupt(format!("StrInline payload is not UTF-8: {e}"))
                })?;
                Ok(PrimaryLookup::Found(PropValueRef::Str(s)))
            }
            PropTag::Bytes => Ok(PrimaryLookup::Found(PropValueRef::Bytes(inline))),
            PropTag::Temporal => Ok(PrimaryLookup::Found(PropValueRef::Temporal(inline))),
            PropTag::StrRef => {
                let (off, olen) = read_locator(inline);
                Ok(PrimaryLookup::InOverflow {
                    tag: PropTag::StrInline,
                    off,
                    len: olen,
                })
            }
            PropTag::ListRef | PropTag::MapRef => {
                if len == 0 {
                    // Locator form (module-docs rule: an encoded
                    // list/map is never 0 bytes).
                    let (off, olen) = read_locator(inline);
                    Ok(PrimaryLookup::InOverflow {
                        tag,
                        off,
                        len: olen,
                    })
                } else if tag == PropTag::ListRef {
                    Ok(PrimaryLookup::Found(PropValueRef::ListOpaque(inline)))
                } else {
                    Ok(PrimaryLookup::Found(PropValueRef::MapOpaque(inline)))
                }
            }
        }
    }

    /// Raw header entry `i` (validated in range by `parse`).
    fn entry(&self, i: usize) -> Result<(u32, PropTag, u8, u16), PropBlockError> {
        let e = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * i;
        let key_id = u32::from_le_bytes([
            self.bytes[e],
            self.bytes[e + 1],
            self.bytes[e + 2],
            self.bytes[e + 3],
        ]);
        let tag = PropTag::from_byte(self.bytes[e + 4])?;
        let len = self.bytes[e + 5];
        let offset = u16::from_le_bytes([self.bytes[e + 6], self.bytes[e + 7]]);
        Ok((key_id, tag, len, offset))
    }

    /// The physical extent an entry's inline payload occupies.
    fn entry_extent(tag: PropTag, len: u8) -> Result<usize, PropBlockError> {
        Ok(match tag {
            PropTag::Null => {
                if len != 0 {
                    return Err(PropBlockError::corrupt(format!(
                        "Null entry with nonzero len {len}"
                    )));
                }
                0
            }
            PropTag::Int64 | PropTag::Float64 => {
                if len != 8 {
                    return Err(PropBlockError::corrupt(format!(
                        "{tag:?} entry len {len} != 8"
                    )));
                }
                8
            }
            PropTag::Bool => {
                if len != 1 {
                    return Err(PropBlockError::corrupt(format!(
                        "Bool entry len {len} != 1"
                    )));
                }
                1
            }
            PropTag::StrRef => {
                if usize::from(len) != LOCATOR_SIZE {
                    return Err(PropBlockError::corrupt(format!(
                        "StrRef entry len {len} != {LOCATOR_SIZE} (the locator size)"
                    )));
                }
                LOCATOR_SIZE
            }
            PropTag::StrInline | PropTag::Bytes | PropTag::Temporal => usize::from(len),
            // len > 0: inline nested bytes; len == 0: 8-B locator.
            PropTag::ListRef | PropTag::MapRef => {
                if len == 0 {
                    LOCATOR_SIZE
                } else {
                    usize::from(len)
                }
            }
        })
    }

    /// Range + canonical-placement validation for entry `i`
    /// (integer-only; part of the eager `parse` sweep). `cursor` is
    /// the canonical-layout march: the end of the previous non-empty
    /// payload (or the data-region start), advanced past this entry's
    /// extent on success. The encoder lays payloads out IN ENTRY ORDER
    /// at the next 4-aligned offset with no slack, so the offset is
    /// fully DETERMINED — any in-range redirect, overlap, or gap is a
    /// canonical-placement violation (A1), caught deterministically.
    fn validate_entry_extent(
        &self,
        i: usize,
        tag: PropTag,
        len: u8,
        offset: u16,
        cursor: &mut usize,
    ) -> Result<(), PropBlockError> {
        let extent = Self::entry_extent(tag, len)?;
        if extent == 0 {
            // Canonical form: zero-extent entries carry offset 0 (a
            // determinism + corruption-detection aid).
            if offset != 0 {
                return Err(PropBlockError::corrupt(format!(
                    "zero-extent entry {i} carries nonzero offset {offset}"
                )));
            }
            return Ok(());
        }
        let off = usize::from(offset);
        // A1 — canonical placement (subsumes the 4-alignment check:
        // the expected offset is 4-aligned by construction).
        let expected = cursor.next_multiple_of(4);
        if off != expected {
            return Err(PropBlockError::corrupt(format!(
                "entry {i} payload offset {off} is not the canonical layout offset {expected} \
                 (in-range redirect, overlap, or gap)"
            )));
        }
        if off + extent > self.data_end {
            return Err(PropBlockError::corrupt(format!(
                "entry {i} payload [{off}, {}) escapes the data region end {}",
                off + extent,
                self.data_end
            )));
        }
        *cursor = off + extent;
        if matches!(tag, PropTag::StrRef | PropTag::ListRef | PropTag::MapRef)
            && extent == LOCATOR_SIZE
            && !self.has_overflow
            && (tag == PropTag::StrRef || len == 0)
        {
            return Err(PropBlockError::corrupt(format!(
                "entry {i} ({tag:?}) references the overflow but the block has no overflow tail"
            )));
        }
        Ok(())
    }

    /// Entry `i`'s inline payload slice (extent validated at parse).
    fn entry_payload(
        &self,
        _i: usize,
        tag: PropTag,
        len: u8,
        offset: u16,
    ) -> Result<&'a [u8], PropBlockError> {
        let extent = Self::entry_extent(tag, len)?;
        if extent == 0 {
            return Ok(&[]);
        }
        let off = usize::from(offset);
        Ok(&self.bytes[off..off + extent])
    }
}

/// Copy an 8-byte scalar payload out (payload extent validated).
fn scalar8(b: &[u8]) -> [u8; 8] {
    let mut out = [0u8; 8];
    out.copy_from_slice(b);
    out
}

/// Read a `{u32 off, u32 len}` overflow locator.
fn read_locator(b: &[u8]) -> (u32, u32) {
    (
        u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
    )
}

// ─────────────────────────────────────────────────────────────────────
// Overflow payload view
// ─────────────────────────────────────────────────────────────────────

/// Zero-copy view over a block's overflow payload.
#[derive(Debug, Clone, Copy)]
pub struct OverflowView<'a> {
    bytes: &'a [u8],
    xentry_count: usize,
}

impl<'a> OverflowView<'a> {
    /// Parse + structurally validate an overflow payload.
    ///
    /// `primary_max_key` is the owning block's largest primary header
    /// key_id ([`PropBlockView::max_primary_key_id`]) — REQUIRED
    /// context for the **global key-ordering invariant** (v2 M2 A2,
    /// L1 review): every xentry key must be STRICTLY GREATER than
    /// every primary key. The encoder guarantees it by construction
    /// (xentries are properties 65+ of one globally-sorted bag), and
    /// the readers' lazy-overflow contract DEPENDS on it — a
    /// cross-directory duplicate makes the full read (xentry
    /// last-wins over the primary entry) and the projected read
    /// (primary hit, overflow never fetched) silently DISAGREE. The
    /// invariant is not checkable from the overflow bytes alone, so
    /// the caller passes the bound; combined with the strictly-
    /// ascending sweep below it yields whole-bag global ordering and
    /// rules out cross-directory duplicates. Pass `None` only when no
    /// primary context exists (a block whose primary header is empty)
    /// or for structure-only tooling that never resolves values.
    pub fn parse(bytes: &'a [u8], primary_max_key: Option<u32>) -> Result<Self, PropBlockError> {
        if bytes.len() < OVERFLOW_HEADER_SIZE {
            return Err(PropBlockError::corrupt(format!(
                "overflow payload too short: {} bytes < {OVERFLOW_HEADER_SIZE}-byte header",
                bytes.len()
            )));
        }
        let magic = u16::from_le_bytes([bytes[0], bytes[1]]);
        if magic != OVERFLOW_MAGIC {
            return Err(PropBlockError::corrupt(format!(
                "overflow payload magic {magic:#06x} != {OVERFLOW_MAGIC:#06x}"
            )));
        }
        let xentry_count = usize::from(u16::from_le_bytes([bytes[2], bytes[3]]));
        let dir_end = OVERFLOW_HEADER_SIZE + XENTRY_SIZE * xentry_count;
        if bytes.len() < dir_end {
            return Err(PropBlockError::corrupt(format!(
                "overflow payload length {} cannot hold {xentry_count} xentries",
                bytes.len()
            )));
        }
        let view = Self {
            bytes,
            xentry_count,
        };
        let mut prev_key: Option<u32> = None;
        // A1 — sequential-canonical march over non-empty xentry
        // extents: the encoder appends them in directory order, each
        // at the next 4-aligned offset after the previous, and the
        // payload ends exactly at the buffer end. (The march starts at
        // the first non-empty xentry's offset — the spilled-primary
        // region that precedes it is only measurable with the owning
        // block in hand.)
        let mut prev_end: Option<usize> = None;
        for i in 0..xentry_count {
            let (key_id, tag, off, len) = view.xentry(i)?;
            // A2 — global ordering: an xentry key at or below the
            // primary max is a cross-directory duplicate/misorder;
            // full and projected reads would disagree on it. LOUD.
            if let Some(pmax) = primary_max_key {
                if key_id <= pmax {
                    return Err(PropBlockError::corrupt(format!(
                        "xentry {i} key_id {key_id} is not globally ordered after the \
                         primary block's max key_id {pmax} (cross-directory duplicate \
                         or misordered overflow directory)"
                    )));
                }
            }
            if let Some(p) = prev_key {
                if key_id <= p {
                    return Err(PropBlockError::corrupt(format!(
                        "xentries not strictly ascending by key_id at index {i} \
                         ({key_id} after {p})"
                    )));
                }
            }
            prev_key = Some(key_id);
            if tag == PropTag::StrRef {
                return Err(PropBlockError::corrupt(
                    "StrRef tag inside an overflow xentry (primary-only tag)",
                ));
            }
            view.validate_xrange(i, tag, off, len)?;
            if len > 0 {
                let (off, len) = (off as usize, len as usize);
                if let Some(pe) = prev_end {
                    let expected = pe.next_multiple_of(4);
                    if off != expected {
                        return Err(PropBlockError::corrupt(format!(
                            "xentry {i} data offset {off} is not the canonical layout offset \
                             {expected} (in-range redirect, overlap, or gap)"
                        )));
                    }
                }
                prev_end = Some(off + len);
            }
        }
        // A1 — the data area ends exactly at the LAST xentry payload
        // (xentry values are appended after every spilled-primary
        // value), so trailing garbage or a shortened final extent is
        // loud. Only checkable when a non-empty xentry exists.
        if let Some(pe) = prev_end {
            if pe != bytes.len() {
                return Err(PropBlockError::corrupt(format!(
                    "overflow payload ends at {} but the canonical xentry layout ends at {pe} \
                     (truncated extent or trailing bytes)",
                    bytes.len()
                )));
            }
        }
        // A1 — directory checksum LAST: the catch-all for tampering
        // the structural sweep cannot see (e.g. a same-width xentry
        // tag retype).
        let stored = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let expect = compute_overflow_meta_checksum(bytes);
        if stored != expect {
            return Err(PropBlockError::corrupt(format!(
                "overflow directory checksum mismatch: stored {stored:#010x}, computed \
                 {expect:#010x}"
            )));
        }
        Ok(view)
    }

    /// Number of extended (wide-bag) entries.
    #[must_use]
    pub fn xentry_count(&self) -> usize {
        self.xentry_count
    }

    /// Iterate xentry key_ids (ascending).
    pub fn key_ids(&self) -> impl Iterator<Item = u32> + 'a {
        let bytes = self.bytes;
        (0..self.xentry_count).map(move |i| {
            let e = OVERFLOW_HEADER_SIZE + XENTRY_SIZE * i;
            u32::from_le_bytes([bytes[e], bytes[e + 1], bytes[e + 2], bytes[e + 3]])
        })
    }

    /// Resolve a primary-entry locator (`PrimaryLookup::InOverflow`)
    /// against this payload.
    pub fn resolve_locator(
        &self,
        tag: PropTag,
        off: u32,
        len: u32,
    ) -> Result<PropValueRef<'a>, PropBlockError> {
        let (off, len) = (off as usize, len as usize);
        if len == 0 {
            // Canonical zero-extent locator (see the encoder).
            if off != 0 {
                return Err(PropBlockError::corrupt(format!(
                    "zero-length overflow locator carries nonzero off {off}"
                )));
            }
            return self.decode_at(tag, &[]);
        }
        let dir_end = OVERFLOW_HEADER_SIZE + XENTRY_SIZE * self.xentry_count;
        if off < dir_end
            || off
                .checked_add(len)
                .is_none_or(|end| end > self.bytes.len())
        {
            return Err(PropBlockError::corrupt(format!(
                "overflow locator [{off}, {off}+{len}) escapes the data area \
                 [{dir_end}, {})",
                self.bytes.len()
            )));
        }
        self.decode_at(tag, &self.bytes[off..off + len])
    }

    /// Wide-bag lookup: binary-search the xentry directory.
    pub fn get(&self, key_id: u32) -> Result<Option<PropValueRef<'a>>, PropBlockError> {
        let mut lo = 0usize;
        let mut hi = self.xentry_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let e = OVERFLOW_HEADER_SIZE + XENTRY_SIZE * mid;
            let k = u32::from_le_bytes([
                self.bytes[e],
                self.bytes[e + 1],
                self.bytes[e + 2],
                self.bytes[e + 3],
            ]);
            match k.cmp(&key_id) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let (_, tag, off, len) = self.xentry(mid)?;
                    return self
                        .decode_at(tag, &self.bytes[off as usize..(off + len) as usize])
                        .map(Some);
                }
            }
        }
        Ok(None)
    }

    /// Raw xentry `i`.
    fn xentry(&self, i: usize) -> Result<(u32, PropTag, u32, u32), PropBlockError> {
        let e = OVERFLOW_HEADER_SIZE + XENTRY_SIZE * i;
        let key_id = u32::from_le_bytes([
            self.bytes[e],
            self.bytes[e + 1],
            self.bytes[e + 2],
            self.bytes[e + 3],
        ]);
        let tag = PropTag::from_byte(self.bytes[e + 4])?;
        if self.bytes[e + 5] != 0 || self.bytes[e + 6] != 0 || self.bytes[e + 7] != 0 {
            return Err(PropBlockError::corrupt(format!(
                "xentry {i} reserved pad bytes are nonzero"
            )));
        }
        let off = u32::from_le_bytes([
            self.bytes[e + 8],
            self.bytes[e + 9],
            self.bytes[e + 10],
            self.bytes[e + 11],
        ]);
        let len = u32::from_le_bytes([
            self.bytes[e + 12],
            self.bytes[e + 13],
            self.bytes[e + 14],
            self.bytes[e + 15],
        ]);
        Ok((key_id, tag, off, len))
    }

    /// Range/shape validation for xentry `i` (parse-time sweep).
    fn validate_xrange(
        &self,
        i: usize,
        tag: PropTag,
        off: u32,
        len: u32,
    ) -> Result<(), PropBlockError> {
        let expected = match tag {
            PropTag::Null => Some(0usize),
            PropTag::Int64 | PropTag::Float64 => Some(8),
            PropTag::Bool => Some(1),
            _ => None,
        };
        if let Some(exp) = expected {
            if len as usize != exp {
                return Err(PropBlockError::corrupt(format!(
                    "xentry {i} ({tag:?}) len {len} != {exp}"
                )));
            }
        }
        let (off, len) = (off as usize, len as usize);
        if len == 0 {
            if off != 0 {
                return Err(PropBlockError::corrupt(format!(
                    "zero-length xentry {i} carries nonzero off {off}"
                )));
            }
            return Ok(());
        }
        let dir_end = OVERFLOW_HEADER_SIZE + XENTRY_SIZE * self.xentry_count;
        if off % 4 != 0 {
            return Err(PropBlockError::corrupt(format!(
                "xentry {i} data offset {off} is not 4-byte aligned"
            )));
        }
        if off < dir_end
            || off
                .checked_add(len)
                .is_none_or(|end| end > self.bytes.len())
        {
            return Err(PropBlockError::corrupt(format!(
                "xentry {i} data [{off}, {off}+{len}) escapes the data area [{dir_end}, {})",
                self.bytes.len()
            )));
        }
        Ok(())
    }

    /// Decode raw value bytes under `tag` (overflow-side twin of the
    /// primary decode; placement is always `(off, len)` here).
    fn decode_at(&self, tag: PropTag, raw: &'a [u8]) -> Result<PropValueRef<'a>, PropBlockError> {
        Ok(match tag {
            PropTag::Null => PropValueRef::Null,
            PropTag::Int64 => {
                if raw.len() != 8 {
                    return Err(PropBlockError::corrupt("Int64 overflow value len != 8"));
                }
                PropValueRef::Int(i64::from_le_bytes(scalar8(raw)))
            }
            PropTag::Float64 => {
                if raw.len() != 8 {
                    return Err(PropBlockError::corrupt("Float64 overflow value len != 8"));
                }
                PropValueRef::Float(f64::from_le_bytes(scalar8(raw)))
            }
            PropTag::Bool => match raw {
                [0] => PropValueRef::Bool(false),
                [1] => PropValueRef::Bool(true),
                _ => {
                    return Err(PropBlockError::corrupt(
                        "Bool overflow value is not a single 0x00/0x01 byte",
                    ));
                }
            },
            PropTag::StrInline => PropValueRef::Str(std::str::from_utf8(raw).map_err(|e| {
                PropBlockError::corrupt(format!("overflow string is not UTF-8: {e}"))
            })?),
            PropTag::Bytes => PropValueRef::Bytes(raw),
            PropTag::Temporal => PropValueRef::Temporal(raw),
            PropTag::ListRef => PropValueRef::ListOpaque(raw),
            PropTag::MapRef => PropValueRef::MapOpaque(raw),
            PropTag::StrRef => {
                return Err(PropBlockError::corrupt(
                    "StrRef tag reached the overflow decoder (primary-only tag)",
                ));
            }
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// Encode a bag with no overflow expected; panics if one appears.
    fn encode_inline_only(pairs: Vec<(u32, PropValue)>) -> Vec<u8> {
        let mut b = PropBlockBuilder::new();
        for (k, v) in pairs {
            b.put(k, v);
        }
        let enc = b.build().expect("encode");
        assert!(
            enc.overflow_payload().is_none(),
            "unexpected overflow payload"
        );
        enc.into_block_bytes(None).expect("finalize")
    }

    /// Encode a bag, staging any overflow at a synthetic BlobRef.
    /// Returns (block bytes, overflow payload bytes).
    fn encode_with_overflow(pairs: Vec<(u32, PropValue)>) -> (Vec<u8>, Option<Vec<u8>>) {
        let mut b = PropBlockBuilder::new();
        for (k, v) in pairs {
            b.put(k, v);
        }
        let enc = b.build().expect("encode");
        let overflow = enc.overflow_payload().map(<[u8]>::to_vec);
        let bref = overflow.as_ref().map(|_| BlobRef::new(42, 7));
        let block = enc.into_block_bytes(bref).expect("finalize");
        (block, overflow)
    }

    /// Full-bag materialize through the view pair (test-side mirror of
    /// the mcp read bridge).
    fn materialize_all(
        block: &[u8],
        overflow: Option<&[u8]>,
    ) -> std::collections::BTreeMap<u32, PropValue> {
        let view = PropBlockView::parse(block).expect("parse block");
        let oview = overflow
            .map(|b| OverflowView::parse(b, view.max_primary_key_id()).expect("parse overflow"));
        let mut out = std::collections::BTreeMap::new();
        for key_id in view.key_ids() {
            match view.get(key_id).expect("get") {
                PrimaryLookup::Found(v) => {
                    out.insert(key_id, v.to_owned_value());
                }
                PrimaryLookup::InOverflow { tag, off, len } => {
                    let ov = oview.as_ref().expect("locator without overflow payload");
                    out.insert(
                        key_id,
                        ov.resolve_locator(tag, off, len)
                            .expect("resolve locator")
                            .to_owned_value(),
                    );
                }
                other => panic!("primary key {key_id} resolved to {other:?}"),
            }
        }
        if let Some(ov) = &oview {
            for key_id in ov.key_ids() {
                let v = ov.get(key_id).expect("xget").expect("xentry present");
                out.insert(key_id, v.to_owned_value());
            }
        }
        out
    }

    #[test]
    fn empty_block_roundtrips() {
        let block = encode_inline_only(vec![]);
        assert_eq!(block.len(), BLOCK_HEADER_SIZE);
        let view = PropBlockView::parse(&block).expect("parse");
        assert_eq!(view.prop_count(), 0);
        assert!(!view.has_overflow());
        assert_eq!(view.get(1).expect("get"), PrimaryLookup::Absent);
    }

    #[test]
    fn scalar_types_roundtrip() {
        let pairs = vec![
            (1, PropValue::Int(i64::MIN)),
            (2, PropValue::Int(-1)),
            (3, PropValue::Int(i64::MAX)),
            (4, PropValue::Float(2.5)),
            (5, PropValue::Float(-0.0)),
            (6, PropValue::Bool(true)),
            (7, PropValue::Bool(false)),
            (8, PropValue::Null),
            (9, PropValue::Str("hello".into())),
            (10, PropValue::Str(String::new())),
        ];
        let block = encode_inline_only(pairs.clone());
        let got = materialize_all(&block, None);
        for (k, v) in pairs {
            assert_eq!(got.get(&k), Some(&v), "key {k}");
        }
    }

    #[test]
    fn nonfinite_floats_normalize_to_null_matching_the_m1_json_bridge() {
        // M1: Value::Float(NaN).to_json_value() == null → reads back as
        // Value::Null. The M2 encoder must match (EXIT gate-5
        // differential).
        for f in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let block = encode_inline_only(vec![(1, PropValue::Float(f))]);
            let got = materialize_all(&block, None);
            assert_eq!(got.get(&1), Some(&PropValue::Null), "float {f}");
        }
    }

    #[test]
    fn boundary_string_lengths_roundtrip() {
        // 255 = inline max; 256 = first overflow length.
        for len in [1usize, 8, 254, 255, 256, 257, 8148, 20_000] {
            let s: String = "x".repeat(len);
            let (block, overflow) = encode_with_overflow(vec![(9, PropValue::Str(s.clone()))]);
            assert_eq!(
                overflow.is_some(),
                len > MAX_INLINE_LEN,
                "overflow presence at len {len}"
            );
            let got = materialize_all(&block, overflow.as_deref());
            assert_eq!(got.get(&9), Some(&PropValue::Str(s)), "len {len}");
        }
    }

    #[test]
    fn nested_opaque_inline_and_overflow_roundtrip() {
        let small_list = br#"[1,2,3]"#.to_vec();
        let big_list = format!("[{}]", "9,".repeat(300)).into_bytes();
        let small_map = br#"{"a":1}"#.to_vec();
        let big_map = format!(r#"{{"k":"{}"}}"#, "v".repeat(400)).into_bytes();
        let pairs = vec![
            (1, PropValue::ListOpaque(small_list.clone())),
            (2, PropValue::ListOpaque(big_list.clone())),
            (3, PropValue::MapOpaque(small_map.clone())),
            (4, PropValue::MapOpaque(big_map.clone())),
        ];
        let (block, overflow) = encode_with_overflow(pairs);
        assert!(overflow.is_some(), "big nested values must spill");
        let got = materialize_all(&block, overflow.as_deref());
        assert_eq!(got.get(&1), Some(&PropValue::ListOpaque(small_list)));
        assert_eq!(got.get(&2), Some(&PropValue::ListOpaque(big_list)));
        assert_eq!(got.get(&3), Some(&PropValue::MapOpaque(small_map)));
        assert_eq!(got.get(&4), Some(&PropValue::MapOpaque(big_map)));
    }

    #[test]
    fn prop_count_boundaries_0_1_64_65_roundtrip() {
        for n in [0usize, 1, 64, 65, 200] {
            let pairs: Vec<(u32, PropValue)> = (0..n)
                .map(|i| (i as u32 * 3 + 1, PropValue::Int(i as i64 - 7)))
                .collect();
            let (block, overflow) = encode_with_overflow(pairs.clone());
            let view = PropBlockView::parse(&block).expect("parse");
            assert_eq!(view.prop_count(), n.min(MAX_PRIMARY_ENTRIES), "n={n}");
            assert_eq!(overflow.is_some(), n > MAX_PRIMARY_ENTRIES, "n={n}");
            let got = materialize_all(&block, overflow.as_deref());
            assert_eq!(got.len(), n, "n={n}");
            for (k, v) in pairs {
                assert_eq!(got.get(&k), Some(&v), "n={n} key {k}");
            }
        }
    }

    #[test]
    fn wide_bag_absent_key_below_primary_max_needs_no_overflow_fetch() {
        // Keys 0,2,4,…,258 (130 props). Key 1 sorts below the primary
        // max and is absent → the lookup must answer WITHOUT the
        // overflow payload (design §M2.3 lazy-overflow).
        let pairs: Vec<(u32, PropValue)> = (0..130)
            .map(|i| (i * 2, PropValue::Int(i64::from(i))))
            .collect();
        let (block, _overflow) = encode_with_overflow(pairs);
        let view = PropBlockView::parse(&block).expect("parse");
        assert_eq!(view.get(1).expect("get"), PrimaryLookup::Absent);
        // A key past the primary range routes to the overflow.
        assert_eq!(
            view.get(1_000_003).expect("get"),
            PrimaryLookup::MaybeInOverflow
        );
    }

    #[test]
    fn reserved_bytes_and_temporal_roundtrip_inline_and_reject_oversize() {
        let pairs = vec![
            (1, PropValue::Bytes(vec![0xAB; 255])),
            (2, PropValue::Temporal(b"2026-07-10T00:00:00Z".to_vec())),
            (3, PropValue::Bytes(Vec::new())),
        ];
        let block = encode_inline_only(pairs.clone());
        let got = materialize_all(&block, None);
        for (k, v) in pairs {
            assert_eq!(got.get(&k), Some(&v), "key {k}");
        }
        // Oversize reserved values reject at encode (no producer at
        // v1.0-α — loud, never truncated).
        let mut b = PropBlockBuilder::new();
        b.put(1, PropValue::Bytes(vec![0u8; 256]));
        assert!(matches!(
            b.build(),
            Err(PropBlockEncodeError::ReservedValueTooLarge { kind: "Bytes", .. })
        ));
    }

    #[test]
    fn duplicate_key_resolves_last_wins() {
        let mut b = PropBlockBuilder::new();
        b.put(5, PropValue::Int(1));
        b.put(5, PropValue::Int(2));
        let block = b
            .build()
            .expect("encode")
            .into_block_bytes(None)
            .expect("finalize");
        let got = materialize_all(&block, None);
        assert_eq!(got.get(&5), Some(&PropValue::Int(2)));
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn overflow_ref_tail_roundtrips_blobref() {
        let (block, overflow) = encode_with_overflow(vec![(1, PropValue::Str("y".repeat(300)))]);
        assert!(overflow.is_some());
        let view = PropBlockView::parse(&block).expect("parse");
        let bref = view.overflow_ref().expect("tail").expect("present");
        assert_eq!(bref.page_id, 42);
        assert_eq!(bref.slot_id, 7);
    }

    #[test]
    fn encode_is_byte_deterministic() {
        let pairs = || {
            vec![
                (7, PropValue::Str("det".into())),
                (1, PropValue::Int(4)),
                (99, PropValue::ListOpaque(br#"[true]"#.to_vec())),
                (3, PropValue::Str("z".repeat(400))),
            ]
        };
        let (a_block, a_of) = encode_with_overflow(pairs());
        let (b_block, b_of) = encode_with_overflow(pairs());
        assert_eq!(a_block, b_block);
        assert_eq!(a_of, b_of);
    }

    // ── G3 — corruption is LOUD (type-safety RED) ──

    #[test]
    fn corrupt_type_tag_rejects_loud() {
        let block = encode_inline_only(vec![(1, PropValue::Int(5))]);
        for bad in [0u8, 11, 0x7F, 0xFF] {
            let mut c = block.clone();
            c[BLOCK_HEADER_SIZE + 4] = bad; // entry 0's tag byte
            let err = PropBlockView::parse(&c).expect_err("must reject");
            assert!(
                matches!(err, PropBlockError::Corrupt { ref reason } if reason.contains("type_tag")),
                "tag {bad:#04x}: {err}"
            );
        }
    }

    #[test]
    fn corrupt_version_rejects_loud() {
        let mut block = encode_inline_only(vec![(1, PropValue::Bool(true))]);
        block[0] = 2;
        assert!(matches!(
            PropBlockView::parse(&block),
            Err(PropBlockError::Corrupt { .. })
        ));
    }

    #[test]
    fn corrupt_offset_and_length_reject_loud() {
        let block = encode_inline_only(vec![(1, PropValue::Str("abcdef".into()))]);
        // Offset escaping the block.
        let mut c = block.clone();
        c[BLOCK_HEADER_SIZE + 6..BLOCK_HEADER_SIZE + 8].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(matches!(
            PropBlockView::parse(&c),
            Err(PropBlockError::Corrupt { .. })
        ));
        // Length escaping the block.
        let mut c = block.clone();
        c[BLOCK_HEADER_SIZE + 5] = 255;
        assert!(matches!(
            PropBlockView::parse(&c),
            Err(PropBlockError::Corrupt { .. })
        ));
        // Unsorted / duplicate keys.
        let mut b = PropBlockBuilder::new();
        b.put(1, PropValue::Int(1));
        b.put(2, PropValue::Int(2));
        let block2 = b
            .build()
            .expect("encode")
            .into_block_bytes(None)
            .expect("finalize");
        let mut c = block2;
        // Rewrite entry 1's key to equal entry 0's.
        let e1 = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE;
        c[e1..e1 + 4].copy_from_slice(&1u32.to_le_bytes());
        assert!(matches!(
            PropBlockView::parse(&c),
            Err(PropBlockError::Corrupt { .. })
        ));
    }

    #[test]
    fn corrupt_bool_payload_rejects_loud_on_access() {
        let block = encode_inline_only(vec![(1, PropValue::Bool(true))]);
        let mut c = block.clone();
        // Bool payload is the first data byte (aligned to 4 after the
        // 12-byte header+entry region).
        let view = PropBlockView::parse(&block).expect("parse");
        let (_, _, _, offset) = view.entry(0).expect("entry");
        c[usize::from(offset)] = 3;
        let cview = PropBlockView::parse(&c).expect("structural parse still passes");
        assert!(matches!(cview.get(1), Err(PropBlockError::Corrupt { .. })));
    }

    #[test]
    fn corrupt_utf8_rejects_loud_on_access() {
        let block = encode_inline_only(vec![(1, PropValue::Str("ab".into()))]);
        let view = PropBlockView::parse(&block).expect("parse");
        let (_, _, _, offset) = view.entry(0).expect("entry");
        let mut c = block.clone();
        c[usize::from(offset)] = 0xFF;
        let cview = PropBlockView::parse(&c).expect("structural parse still passes");
        assert!(matches!(cview.get(1), Err(PropBlockError::Corrupt { .. })));
    }

    #[test]
    fn corrupt_overflow_magic_and_pads_reject_loud() {
        let (_, overflow) =
            encode_with_overflow((0..70).map(|i| (i, PropValue::Int(i64::from(i)))).collect());
        let overflow = overflow.expect("wide bag has overflow");
        let mut c = overflow.clone();
        c[0] = 0x00;
        assert!(matches!(
            OverflowView::parse(&c, None),
            Err(PropBlockError::Corrupt { .. })
        ));
        // Pad-byte tamper WITH a recomputed directory checksum: the
        // pad validation itself must reject (not just the checksum).
        let mut c = overflow.clone();
        c[OVERFLOW_HEADER_SIZE + 5] = 1; // first xentry pad byte
        let recomputed = compute_overflow_meta_checksum(&c);
        c[4..8].copy_from_slice(&recomputed.to_le_bytes());
        let err = OverflowView::parse(&c, None).expect_err("pad byte must reject");
        assert!(
            matches!(err, PropBlockError::Corrupt { ref reason } if reason.contains("pad")),
            "{err}"
        );
        // Raw directory-region tamper WITHOUT recompute: the checksum
        // catches what structure alone cannot pin.
        let mut c = overflow.clone();
        c[4] ^= 0xFF; // a dir_check byte itself
        assert!(matches!(
            OverflowView::parse(&c, None),
            Err(PropBlockError::Corrupt { .. })
        ));
    }

    #[test]
    fn locator_without_overflow_tail_rejects_loud() {
        // Hand-build a block whose entry claims StrRef but whose flags
        // carry no overflow: parse must reject (never a dangling deref).
        let (block, _) = encode_with_overflow(vec![(1, PropValue::Str("q".repeat(300)))]);
        let mut c = block.clone();
        // Clear flags bit0 and truncate the tail.
        c[2..4].copy_from_slice(&0u16.to_le_bytes());
        c.truncate(c.len() - OVERFLOW_TAIL_SIZE);
        assert!(matches!(
            PropBlockView::parse(&c),
            Err(PropBlockError::Corrupt { .. })
        ));
    }

    // ── Property tests ──

    /// Strategy over PropValues covering every tag + boundary lengths.
    fn prop_value_strategy() -> impl Strategy<Value = PropValue> {
        prop_oneof![
            Just(PropValue::Null),
            any::<i64>().prop_map(PropValue::Int),
            // Finite floats only at the strategy level: non-finite
            // normalization has its own dedicated test above.
            any::<f64>()
                .prop_filter("finite", |f| f.is_finite())
                .prop_map(PropValue::Float),
            any::<bool>().prop_map(PropValue::Bool),
            proptest::string::string_regex("[a-zA-Z0-9 \u{00e9}\u{4e16}]{0,400}")
                .expect("regex")
                .prop_map(PropValue::Str),
            proptest::collection::vec(any::<u8>(), 0..=255).prop_map(PropValue::Bytes),
            proptest::collection::vec(any::<u8>(), 0..=255).prop_map(PropValue::Temporal),
            proptest::collection::vec(any::<u8>(), 1..=600).prop_map(PropValue::ListOpaque),
            proptest::collection::vec(any::<u8>(), 1..=600).prop_map(PropValue::MapOpaque),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(
            if cfg!(debug_assertions) { 256 } else { 1024 }
        ))]

        /// Every bag round-trips value-identically through encode →
        /// parse → materialize, at every prop count (0..=96 spans the
        /// 64-entry primary boundary), with the encoder's non-finite
        /// normalization applied to the expectation.
        #[test]
        fn roundtrip_any_bag(
            entries in proptest::collection::btree_map(
                any::<u32>(), prop_value_strategy(), 0..96
            )
        ) {
            let mut b = PropBlockBuilder::new();
            for (k, v) in &entries {
                b.put(*k, v.clone());
            }
            let enc = b.build().expect("encode");
            let overflow = enc.overflow_payload().map(<[u8]>::to_vec);
            let bref = overflow.as_ref().map(|_| BlobRef::new(9, 3));
            let block = enc.into_block_bytes(bref).expect("finalize");

            let got = materialize_all(&block, overflow.as_deref());
            prop_assert_eq!(got.len(), entries.len());
            for (k, v) in entries {
                let expect = match v {
                    PropValue::Float(f) if !f.is_finite() => PropValue::Null,
                    other => other,
                };
                prop_assert_eq!(got.get(&k), Some(&expect));
            }
        }

        /// Single-byte corruption anywhere in a block either (a) still
        /// parses+materializes (the flip hit payload bytes / padding —
        /// values may legitimately differ) or (b) rejects LOUD with
        /// `Corrupt`. It must NEVER panic. (The bounds discipline that
        /// keeps every read in range regardless of input.)
        #[test]
        fn corrupted_block_never_panics(
            entries in proptest::collection::btree_map(
                any::<u32>(), prop_value_strategy(), 1..40
            ),
            flip_at in any::<prop::sample::Index>(),
            flip_bit in 0u8..8,
        ) {
            let mut b = PropBlockBuilder::new();
            for (k, v) in &entries {
                b.put(*k, v.clone());
            }
            let enc = b.build().expect("encode");
            let overflow = enc.overflow_payload().map(<[u8]>::to_vec);
            let bref = overflow.as_ref().map(|_| BlobRef::new(9, 3));
            let mut block = enc.into_block_bytes(bref).expect("finalize");

            let at = flip_at.index(block.len());
            block[at] ^= 1 << flip_bit;

            if let Ok(view) = PropBlockView::parse(&block) {
                for key_id in view.key_ids() {
                    match view.get(key_id) {
                        Ok(PrimaryLookup::InOverflow { tag, off, len }) => {
                            if let Some(of) = &overflow {
                                if let Ok(ov) = OverflowView::parse(of, view.max_primary_key_id()) {
                                    let _ = ov.resolve_locator(tag, off, len);
                                }
                            }
                        }
                        Ok(_) | Err(PropBlockError::Corrupt { .. }) => {}
                    }
                }
            }
        }

        /// Overflow payload single-byte corruption never panics either.
        #[test]
        fn corrupted_overflow_never_panics(
            n in 65usize..120,
            flip_at in any::<prop::sample::Index>(),
            flip_bit in 0u8..8,
        ) {
            let pairs: Vec<(u32, PropValue)> = (0..n)
                .map(|i| (i as u32, PropValue::Str(format!("v{i}"))))
                .collect();
            let (_, overflow) = encode_with_overflow(pairs);
            let mut of = overflow.expect("wide bag");
            let at = flip_at.index(of.len());
            of[at] ^= 1 << flip_bit;
            if let Ok(ov) = OverflowView::parse(&of, None) {
                for k in ov.key_ids().collect::<Vec<_>>() {
                    let _ = ov.get(k);
                }
            }
        }
    }
}
