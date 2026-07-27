//! v2 M2 A1 — codec metadata-integrity RED-on-revert gates at the
//! storage grain (L1 review defect A1; promoted from the QA scratch
//! `l1_m2_codec_adversarial_scratch.rs`, flipped from assert-the-bug
//! to assert-the-contract).
//!
//! # The defect these pin closed
//!
//! `PropBlockView::parse` checked tag/size/range but NOT canonical
//! placement, non-overlap, or any self-integrity — so IN-RANGE
//! metadata corruption was silently accepted: a valid-looking offset
//! rewrite redirected one key's value to another's bytes, a
//! same-width `Int64`→`Float64` tag flip silently retyped a scalar,
//! and an in-range `len` edit silently truncated a string.
//!
//! # The fix, two load-bearing layers (each with its own RED here)
//!
//! 1. **Canonical placement** (structural, deterministic): payload
//!    extents march in entry order at the next 4-aligned offset,
//!    ending exactly at the data-region end. Catches every offset
//!    redirect / overlap / gap / net truncation — INDEPENDENT of the
//!    checksum, proven by the recomputed-checksum variants below.
//! 2. **Metadata self-checksum** (CRC-32C in header bytes 4..8 —
//!    round-3 (#1452): widened from round-1's flags-byte CRC-8 by
//!    Director ruling after the codex re-check MEASURED the 8-bit
//!    width too narrow — 2,378 of the 2,096,128 two-bit key-field
//!    upsets cancelled CRC-8 and cleared every structural sweep;
//!    CRC-32C's HD = 6 through 5,243 dataword bits covers the whole
//!    ≤ 528-B metadata span, so every ≤ 5-bit scribble is
//!    deterministically caught): catches what structure cannot pin —
//!    the same-width tag retype, the ordered two-bit key upset.
//!    Verified LAST so structural corruption keeps its specific
//!    message. Width-disposition record: the doc on
//!    `compute_block_meta_checksum` in `prop_block.rs`; the width's
//!    RED-on-revert gates live below (the collision family) and in
//!    `m2_metadata_checksum_width_gate.rs` (the exhaustive two-bit
//!    population) and `arcgraph-mcp/tests/m2_codec_read_integrity_
//!    gate.rs` (the end-to-end wrong-name read).
//!
//! Data-region (payload) bytes stay OUTSIDE the checksum by design:
//! parse never touches value bytes (the G2 bytes-touched contract),
//! and payload corruption remains the documented on-access fault.
//!
//! # Documented boundary, pinned (round-2, #1452 residual-2)
//!
//! One further test pins a CONTRACT EDGE rather than its center, so
//! the docs can never silently drift from behavior: the data-region
//! locator redirect (payload bytes are outside both the checksum and
//! the placement sweep BY DESIGN — the zero-decode G2 contract; the
//! page-grain CRC32C owns that region's rot). Round-2's OTHER edge
//! pin — the CRC-8 multi-flip collision family as a documented-keep
//! boundary — was retired by the round-3 widen: that family is now a
//! DEFECT the checksum must catch, and its test below flipped from
//! assert-the-boundary to assert-the-kill.

use arcgraph_storage::prop_block::{
    BLOCK_HEADER_SIZE, HEADER_ENTRY_SIZE, OverflowView, PrimaryLookup, PropBlockBuilder,
    PropBlockError, PropBlockView, PropValue, PropValueRef, compute_block_meta_checksum,
    stamp_block_meta_checksum,
};
use arcgraph_storage::property::BlobRef;

fn inline_block(pairs: &[(u32, PropValue)]) -> Vec<u8> {
    let mut builder = PropBlockBuilder::new();
    for (key, value) in pairs {
        builder.put(*key, value.clone());
    }
    let encoded = builder.build().expect("encode valid bag");
    assert!(encoded.overflow_payload().is_none());
    encoded.into_block_bytes(None).expect("finalize")
}

/// Re-stamp a tampered block's checksum so the STRUCTURAL layer is
/// the one under test (the tamper cannot hide behind the checksum).
fn restamp(block: &mut [u8]) {
    stamp_block_meta_checksum(block);
}

/// A1 gate #1 — in-range offset redirect. Key 1's valid, aligned
/// 8-byte extent is redirected onto key 2's equally-shaped extent;
/// the checksum is RECOMPUTED so canonical placement alone must
/// reject. RED-on-revert: drop the canonical-offset march and this
/// parses fine, silently reading 222 under key 1.
#[test]
fn corrupted_in_range_offset_redirect_is_loud_corrupt() {
    let block = inline_block(&[(1, PropValue::Int(111)), (2, PropValue::Int(222))]);
    let entry0 = BLOCK_HEADER_SIZE;
    let entry1 = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE;

    // Recomputed-checksum variant: the structural layer is load-bearing.
    let mut c = block.clone();
    let second_offset = [c[entry1 + 6], c[entry1 + 7]];
    c[entry0 + 6..entry0 + 8].copy_from_slice(&second_offset);
    restamp(&mut c);
    let err = PropBlockView::parse(&c).expect_err("in-range offset redirect must be loud");
    assert!(
        matches!(err, PropBlockError::Corrupt { ref reason } if reason.contains("canonical")),
        "canonical placement names the violation: {err}"
    );

    // Raw variant (checksum NOT recomputed) still rejects — layered.
    let mut raw = block.clone();
    raw[entry0 + 6..entry0 + 8].copy_from_slice(&second_offset);
    assert!(PropBlockView::parse(&raw).is_err());

    // Control: the untampered block parses and reads correctly.
    let view = PropBlockView::parse(&block).expect("pristine block parses");
    assert_eq!(
        view.get(1).expect("get"),
        PrimaryLookup::Found(PropValueRef::Int(111))
    );
}

/// A1 gate #2 — same-width tag retype (`Int64`→`Float64`, both 8-B
/// extents: structurally indistinguishable). Only the metadata
/// checksum can catch it. RED-on-revert: drop the checksum
/// verification and this parses fine, silently materializing
/// `Float(5e-324)` where `Int(5)` was written.
#[test]
fn corrupted_known_tag_retype_is_loud_corrupt() {
    let block = inline_block(&[(1, PropValue::Int(5))]);
    let entry0 = BLOCK_HEADER_SIZE;

    let mut c = block.clone();
    c[entry0 + 4] = 3; // Int64=2 → Float64=3, same physical extent
    let err = PropBlockView::parse(&c).expect_err("same-width tag retype must be loud");
    assert!(
        matches!(err, PropBlockError::Corrupt { ref reason } if reason.contains("checksum")),
        "only the metadata checksum can see a same-width retype: {err}"
    );

    // Scope honesty, pinned: a retype WITH a forged (recomputed)
    // checksum is out of the anti-corruption threat model — the
    // checksum defends against rot/scribbles, not an adversary who
    // can already write arbitrary block bytes AND their checksum.
    let mut forged = block.clone();
    forged[entry0 + 4] = 3;
    restamp(&mut forged);
    let view = PropBlockView::parse(&forged).expect("forged-checksum retype parses");
    assert_eq!(
        view.get(1).expect("get"),
        PrimaryLookup::Found(PropValueRef::Float(f64::from_bits(5)))
    );
}

/// A1 gate #3 — in-range variable-length truncation (`len` 6 → 5 on a
/// string; five bytes remain in-range and valid UTF-8). The canonical
/// march's exact data-region-end check rejects it even with a
/// recomputed checksum. RED-on-revert: drop the `cursor == data_end`
/// check and the recomputed variant silently reads `"abcde"`.
#[test]
fn corrupted_in_range_length_truncation_is_loud_corrupt() {
    let block = inline_block(&[(1, PropValue::Str("abcdef".into()))]);
    let entry0 = BLOCK_HEADER_SIZE;

    // Recomputed-checksum variant: structural layer load-bearing.
    let mut c = block.clone();
    c[entry0 + 5] = 5;
    restamp(&mut c);
    let err = PropBlockView::parse(&c).expect_err("in-range truncation must be loud");
    assert!(
        matches!(err, PropBlockError::Corrupt { ref reason }
            if reason.contains("truncated") || reason.contains("canonical")),
        "the canonical layout names the violation: {err}"
    );

    // Raw variant still rejects (checksum layer).
    let mut raw = block;
    raw[entry0 + 5] = 5;
    assert!(PropBlockView::parse(&raw).is_err());
}

/// The checksum helper round-trips: a freshly-encoded block carries
/// exactly the checksum the public helper computes (the gate the
/// recomputed-tamper variants above depend on).
#[test]
fn encoder_stamps_the_checksum_the_helper_computes() {
    for pairs in [
        vec![(1, PropValue::Int(1))],
        vec![
            (1, PropValue::Null),
            (2, PropValue::Str("x".repeat(300))), // overflow + tail
            (3, PropValue::Bool(true)),
        ],
    ] {
        let mut builder = PropBlockBuilder::new();
        for (k, v) in &pairs {
            builder.put(*k, v.clone());
        }
        let enc = builder.build().expect("encode");
        let bref = enc.overflow_payload().map(|_| BlobRef::new(42, 7));
        let block = enc.into_block_bytes(bref).expect("finalize");
        // Restamping a fresh block must be a byte-identical no-op —
        // the stored meta_check equals the recomputed value, without
        // this gate hardcoding the field's offset.
        let mut restamped = block.clone();
        stamp_block_meta_checksum(&mut restamped);
        assert_eq!(
            restamped, block,
            "stored meta_check equals the recomputed value"
        );
        PropBlockView::parse(&block).expect("fresh block parses");
    }
}

/// Frozen CRC-8/ATM (poly 0x07) reference over the SAME metadata
/// coverage the production checksum uses (bytes 0..4, the meta_check
/// field as zeros, header entries, tail when flagged) — the
/// HISTORICAL round-1 width, kept test-local so the width RED-on-
/// revert gates below stay decoupled from production code. A width
/// revert (production checksum narrowed back to 8 bits) reproduces
/// exactly this function's collision structure.
fn frozen_crc8_meta(block: &[u8]) -> u8 {
    const fn table() -> [u8; 256] {
        let mut table = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            let mut crc = i as u8;
            let mut b = 0;
            while b < 8 {
                crc = if crc & 0x80 != 0 {
                    (crc << 1) ^ 0x07
                } else {
                    crc << 1
                };
                b += 1;
            }
            table[i] = crc;
            i += 1;
        }
        table
    }
    const CRC8: [u8; 256] = table();
    let fold = |mut crc: u8, bytes: &[u8]| {
        for &b in bytes {
            crc = CRC8[(crc ^ b) as usize];
        }
        crc
    };
    let prop_count = block[1] as usize;
    let entries_end = (BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * prop_count).min(block.len());
    let has_overflow = u16::from(block[2]) & 0b1 != 0;
    let mut crc = fold(0, &block[0..4]);
    crc = fold(crc, &[0u8; 4]); // the meta_check field as zero
    crc = fold(crc, &block[BLOCK_HEADER_SIZE..entries_end]);
    if has_overflow && block.len() >= 8 {
        crc = fold(crc, &block[block.len() - 8..]);
    }
    crc
}

/// **Width RED-on-revert gate (round-3, #1452 — the widen's oracle
/// for the RETYPE family).** Round-2 pinned the 4-flip same-width
/// `Int64`→`Float64` collision family as CRC-8's documented
/// pigeonhole boundary; the round-3 Director ruling widened the
/// checksum to CRC-32C, so the family flips from documented edge to
/// DEFECT-that-must-die. Two prongs, both of which go RED if the
/// width is reverted:
///
/// 1. The family found under the FROZEN historical CRC-8 (it always
///    exists at 64 entries: ~2016 pair-deltas over 256 values) must
///    be caught LOUD by the production parse — a weight-4 scribble
///    inside CRC-32C's HD = 6 window cannot cancel. On a width
///    revert the four flips cancel again, every flip is same-width
///    (structure-clean), and the pre-round-3 silent whole-block
///    retype resurfaces as a parse `Ok`.
/// 2. The SAME search run against the PRODUCTION checksum must find
///    NO colliding family — the measured statement "this width has
///    no 4-flip retype collisions on this layout", asserted directly
///    rather than trusted from the HD table. An 8-bit revert makes
///    the search succeed by pigeonhole.
#[test]
fn metadata_checksum_width_kills_the_crc8_multi_flip_collision_family() {
    let pairs: Vec<(u32, PropValue)> = (1..=64u32)
        .map(|key| (key, PropValue::Int(i64::from(key))))
        .collect();
    let block = inline_block(&pairs);

    // Per-entry frozen-CRC-8 delta of the same-width Int64 → Float64
    // flip (syndrome of the single-byte tamper).
    let mut single_delta = [0u8; 64];
    for (i, delta) in single_delta.iter_mut().enumerate() {
        let mut candidate = block.clone();
        candidate[BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * i + 4] = 3;
        *delta = frozen_crc8_meta(&candidate) ^ frozen_crc8_meta(&block);
    }
    // Two DISJOINT pairs with the same pair-delta ⇒ all four flips
    // cancel under the 8-bit width. ~2016 pair-XORs over 256 possible
    // values: a disjoint collision exists for this fixed layout.
    let mut seen = std::collections::HashMap::<u8, (usize, usize)>::new();
    let mut indices = None;
    'outer: for a in 0..64usize {
        for b in a + 1..64usize {
            let delta = single_delta[a] ^ single_delta[b];
            if let Some(&(c, d)) = seen.get(&delta) {
                if a != c && a != d && b != c && b != d {
                    indices = Some([a, b, c, d]);
                    break 'outer;
                }
            } else {
                seen.insert(delta, (a, b));
            }
        }
    }
    let indices = indices.expect(
        "an 8-bit checksum must exhibit a disjoint colliding 4-flip family at 64 entries \
         (pigeonhole over 256 syndromes) — the frozen reference cannot fail to find one",
    );

    // Prong 1 — the historical family is now caught LOUD.
    let mut candidate = block.clone();
    for i in indices {
        candidate[BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * i + 4] = 3;
    }
    assert_eq!(
        frozen_crc8_meta(&candidate),
        frozen_crc8_meta(&block),
        "the four flips cancel under the frozen 8-bit width by construction",
    );
    let err = PropBlockView::parse(&candidate).expect_err(
        "the 4-flip retype family cancelled CRC-8 and survived parse pre-round-3; \
         the widened checksum must catch it LOUD — a parse Ok here means the \
         metadata checksum width was reverted",
    );
    assert!(
        matches!(err, PropBlockError::Corrupt { ref reason } if reason.contains("checksum")),
        "the metadata checksum names the violation: {err}"
    );

    // Prong 2 — the production width has NO such family, measured.
    let baseline = compute_block_meta_checksum(&block);
    let mut production_delta = [0u32; 64];
    for (i, delta) in production_delta.iter_mut().enumerate() {
        let mut candidate = block.clone();
        candidate[BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * i + 4] = 3;
        *delta = compute_block_meta_checksum(&candidate) ^ baseline;
    }
    for (i, delta) in production_delta.iter().enumerate() {
        assert_ne!(
            *delta, 0,
            "entry {i}: a single same-width retype must move the production checksum"
        );
    }
    // All 2,016 pair-deltas must be UNIQUE: with the single deltas
    // pairwise distinct (asserted above via the pair check below —
    // (a,x),(a,y) colliding would force delta_x == delta_y), any two
    // DISTINCT pairs sharing a pair-delta are automatically disjoint,
    // i.e. exactly a cancelling 4-flip family.
    let mut seen = std::collections::HashMap::<u32, (usize, usize)>::new();
    for a in 0..64usize {
        for b in a + 1..64usize {
            let delta = production_delta[a] ^ production_delta[b];
            assert_ne!(
                delta, 0,
                "entries {a}+{b}: a retype PAIR must not cancel the production checksum"
            );
            if let Some((c, d)) = seen.insert(delta, (a, b)) {
                panic!(
                    "retype pair-deltas collide: ({a},{b}) and ({c},{d}) share \
                     {delta:#010x} — a 4-flip family cancels the production checksum \
                     (the width was narrowed; this is the RED-on-revert)"
                );
            }
        }
    }
}

/// **Documented data-region boundary, pinned (round-2, #1452
/// residual-2).** A primary `StrRef` locator's 8-byte payload lives in
/// the DATA region, so neither the metadata checksum nor canonical
/// header placement covers its CONTENTS (both cover the header entry
/// that points AT it). Redirecting one locator onto another's extent
/// therefore parses clean and resolves the WRONG string — by design:
/// parse never reads value bytes (the G2 zero-decode contract), and
/// the data region's integrity is owned by the page-grain CRC32C
/// (physical rot) + the AEAD layer (adversaries). Extending the
/// metadata checksum over payload bytes is explicitly out of contract
/// (it would put every value byte on the parse path).
///
/// If this test starts FAILING with a Corrupt on the redirect, the
/// checksum's coverage grew past the metadata — that is a CONTRACT
/// change (G2 + the on-access-fault posture); re-decide it explicitly.
#[test]
fn data_region_locator_redirect_is_the_documented_on_access_boundary() {
    let first = "a".repeat(300);
    let second = "b".repeat(300);
    let mut builder = PropBlockBuilder::new();
    builder.put(1, PropValue::Str(first));
    builder.put(2, PropValue::Str(second.clone()));
    let encoded = builder.build().expect("encode");
    let overflow = encoded
        .overflow_payload()
        .expect("both strings spill")
        .to_vec();
    let mut block = encoded
        .into_block_bytes(Some(BlobRef::new(7, 1)))
        .expect("finalize");

    let view = PropBlockView::parse(&block).expect("pristine block parses");
    let first_locator = match view.get(1).expect("get") {
        PrimaryLookup::InOverflow { off, len, .. } => (off, len),
        other => panic!("expected first locator, got {other:?}"),
    };
    let second_locator = match view.get(2).expect("get") {
        PrimaryLookup::InOverflow { off, len, .. } => (off, len),
        other => panic!("expected second locator, got {other:?}"),
    };
    assert_ne!(first_locator, second_locator);

    // Locate entry 1's canonical 8-byte locator payload in the data
    // region and redirect it onto entry 2's extent.
    let first_entry = BLOCK_HEADER_SIZE;
    let locator_at = usize::from(u16::from_le_bytes([
        block[first_entry + 6],
        block[first_entry + 7],
    ]));
    block[locator_at..locator_at + 4].copy_from_slice(&second_locator.0.to_le_bytes());
    block[locator_at + 4..locator_at + 8].copy_from_slice(&second_locator.1.to_le_bytes());

    let redirected_view = PropBlockView::parse(&block)
        .expect("a locator rewrite is outside the block METADATA checksum by design");
    let overflow_view = OverflowView::parse(&overflow, redirected_view.max_primary_key_id())
        .expect("overflow directory untouched");
    let redirected = match redirected_view.get(1).expect("get") {
        PrimaryLookup::InOverflow { tag, off, len } => overflow_view
            .resolve_locator(tag, off, len)
            .expect("redirected extent resolves"),
        other => panic!("expected redirected locator, got {other:?}"),
    };
    assert_eq!(
        redirected,
        PropValueRef::Str(&second),
        "the wrong-value read IS the pinned boundary: data-region bytes \
         are outside the metadata integrity layers by design",
    );
}
