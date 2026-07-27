//! v2 M2 A1 round-3 (#1452) — metadata-checksum WIDTH RED-on-revert
//! gates at the storage grain. Promoted from the QA scratch
//! `l1_m2_crc8_random_scratch.rs` (the codex re-check's measurement),
//! flipped from assert-the-bug to assert-the-contract.
//!
//! # The defect these pin closed
//!
//! Round-1 guarded the block metadata with a CRC-8 in the reserved
//! flags byte; round-2 kept the width as a documented boundary. The
//! codex re-check then MEASURED it insufficient under the DECLARED
//! threat model (uncrafted scribbles): exhaustively enumerating all
//! 2,096,128 independent two-bit upsets of a full block's 64 key-id
//! fields, **2,378 (~1 in 881) cancelled the CRC-8 syndrome, kept
//! strict key order, and kept canonical placement** — clearing every
//! structural sweep and materializing values under WRONG existing
//! property names. The hot resident read path opens pages via
//! `open_prop_trusted` (skips the page-grain CRC32C by design), so
//! the per-block metadata checksum is the ONLY guard on a resident
//! scribble. Director round-3 ruling: widen to CRC-32 —
//! [`compute_block_meta_checksum`] is now CRC-32C in header bytes
//! 4..8 (see its width-disposition doc).
//!
//! # The gates (the measurement becomes the oracle)
//!
//! 1. The SAME exhaustive two-bit enumeration, bucketed by the
//!    PRODUCTION checksum's syndromes: the cancelling-pair count must
//!    be ZERO (CRC-32C holds HD = 6 through 5,243 dataword bits —
//!    the whole ≤ 528-B metadata span — so no ≤ 5-bit upset can
//!    cancel; measured here, not trusted). Narrow the checksum and
//!    pigeonhole floods the buckets: the count explodes and the gate
//!    goes RED naming the resurfaced silent-wrong-read population.
//! 2. A concrete two-bit key upset chosen to cancel the FROZEN
//!    historical CRC-8 (the round-1 width, kept test-local) while
//!    preserving strict key order — the exact shape that read wrong
//!    names pre-round-3 — must be caught LOUD by the checksum layer.
//!    On a width revert it parses clean again and the gate prints
//!    the wrong keys it silently materialized.
//!
//! The end-to-end sibling (the same upset staged through the
//! production crud path and read via `record_property_bag_checked`,
//! wrong NAMES and all) lives in
//! `arcgraph-mcp/tests/m2_codec_read_integrity_gate.rs`.

use std::collections::HashMap;

use arcgraph_storage::prop_block::{
    BLOCK_HEADER_SIZE, HEADER_ENTRY_SIZE, PropBlockBuilder, PropBlockError, PropBlockView,
    PropValue, compute_block_meta_checksum,
};

/// The scratch's fixture: 64 primary entries whose key ids stride
/// 1024 — real bags use sparse subsets of the tenant-wide intern-id
/// space, and the stride leaves room for low-bit upsets to stay
/// globally ordered instead of being rejected by the ordering sweep
/// (the population a checksum must catch ALONE).
fn stride_1024_block() -> Vec<u8> {
    let mut builder = PropBlockBuilder::new();
    for i in 1..=64u32 {
        builder.put(i * 1024, PropValue::Int(i64::from(i)));
    }
    builder
        .build()
        .expect("build")
        .into_block_bytes(None)
        .expect("finalize")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Flip {
    byte: usize,
    bit: u8,
}

impl Flip {
    fn apply(self, block: &mut [u8]) {
        block[self.byte] ^= 1 << self.bit;
    }
}

/// Every single-bit upset of the 64 primary key-id fields (64 entries
/// × 4 key bytes × 8 bits = 2,048 flips — the scratch's population).
fn key_field_flips() -> Vec<Flip> {
    let mut flips = Vec::with_capacity(64 * 4 * 8);
    for entry in 0..64usize {
        let key_at = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * entry;
        for byte_in_key in 0..4usize {
            for bit in 0..8u8 {
                flips.push(Flip {
                    byte: key_at + byte_in_key,
                    bit,
                });
            }
        }
    }
    flips
}

/// **Width gate #1 — the exhaustive two-bit population has ZERO
/// checksum survivors.** This is the codex re-check's measurement
/// with the verdict inverted: every pair of independent single-bit
/// key-field upsets whose syndromes cancel would reach the structural
/// sweeps with a wrong key id — at CRC-8, 2,378 of them cleared
/// EVERYTHING. At the production width no pair may cancel at all.
///
/// RED-on-revert: narrow `compute_block_meta_checksum` back to 8 bits
/// and ≥ 2,048 − 256 pairs collide by pigeonhole — the assert names
/// the count and the silent-wrong-read consequence.
#[test]
fn two_bit_key_metadata_population_has_zero_checksum_survivors() {
    let block = stride_1024_block();
    let baseline_keys: Vec<u32> = PropBlockView::parse(&block)
        .expect("baseline parse")
        .key_ids()
        .collect();
    let baseline_crc = compute_block_meta_checksum(&block);

    // Single-flip syndromes under the PRODUCTION checksum. CRC
    // linearity over GF(2): a two-flip upset cancels iff the two
    // single-flip syndromes are equal.
    let flips = key_field_flips();
    let mut by_syndrome = HashMap::<u32, Vec<Flip>>::new();
    for &flip in &flips {
        let mut one = block.clone();
        flip.apply(&mut one);
        let syndrome = compute_block_meta_checksum(&one) ^ baseline_crc;
        assert_ne!(
            syndrome, 0,
            "single-bit key upset {flip:?} must move the metadata checksum"
        );
        by_syndrome.entry(syndrome).or_default().push(flip);
    }

    let total_pairs = flips.len() * (flips.len() - 1) / 2;
    let mut cancelling_pairs = 0usize;
    let mut silent_wrong_reads = 0usize;
    let mut first = None;
    for group in by_syndrome.values() {
        for a in 0..group.len() {
            for b in a + 1..group.len() {
                cancelling_pairs += 1;
                let mut candidate = block.clone();
                group[a].apply(&mut candidate);
                group[b].apply(&mut candidate);
                assert_eq!(
                    compute_block_meta_checksum(&candidate),
                    baseline_crc,
                    "equal single-flip syndromes must cancel pairwise"
                );
                if let Ok(view) = PropBlockView::parse(&candidate) {
                    let keys: Vec<u32> = view.key_ids().collect();
                    if keys != baseline_keys {
                        silent_wrong_reads += 1;
                        first.get_or_insert((group[a], group[b], keys));
                    }
                }
            }
        }
    }

    println!(
        "two-bit key-field population: total={total_pairs}, \
         checksum_cancelling={cancelling_pairs}, silent_wrong_reads={silent_wrong_reads} \
         (CRC-8 measured 22k+/2,378 here; the width gate pins both at zero)"
    );
    assert_eq!(
        cancelling_pairs, 0,
        "{cancelling_pairs} two-bit key upsets cancel the metadata checksum \
         (first silent wrong-key read: {first:?}) — the checksum width was \
         narrowed below the round-3 CRC-32C ruling (#1452): every such pair \
         is a candidate silent wrong-name read on a resident scribble"
    );
    assert_eq!(
        silent_wrong_reads, 0,
        "two-bit key upsets survived checksum AND structure — silent wrong-key \
         reads are live again: {first:?}"
    );
}

/// Frozen CRC-8/ATM (poly 0x07) over the SAME metadata coverage the
/// production checksum uses — the HISTORICAL round-1 width. Kept
/// test-local (and deliberately duplicated across the width gates, so
/// no shared helper can be "fixed" into silently weakening them): a
/// width revert reproduces exactly this function's collision
/// structure.
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

/// **Width gate #2 — the CRC-8-surviving two-bit key upset is caught
/// LOUD.** Deterministically reconstructs the re-check's survivor
/// shape: two single-bit flips confined to each key's low 10 bits
/// (the 1024 stride guarantees strict ordering survives) whose FROZEN
/// CRC-8 syndromes cancel — 640 candidate flips over ≤ 256 syndromes,
/// so a colliding pair always exists. Pre-round-3 this exact upset
/// cleared checksum + ordering + placement and materialized wrong
/// keys; the widened checksum must reject it, and the checksum layer
/// specifically (key flips are structurally clean by construction —
/// no other sweep can see them).
///
/// RED-on-revert: narrow the checksum to CRC-8 and the pair cancels
/// the production stamp again — parse returns `Ok` and this gate
/// panics printing the wrong keys it silently read.
#[test]
fn crc8_surviving_two_bit_key_upset_is_loud_corrupt() {
    let block = stride_1024_block();
    let baseline_keys: Vec<u32> = PropBlockView::parse(&block)
        .expect("baseline parse")
        .key_ids()
        .collect();
    let frozen_baseline = frozen_crc8_meta(&block);

    // Order-preserving flip population: bits 0..10 of each key id.
    // Every key is i·1024, so its low 10 bits are ZERO — a flip there
    // always ADDS 2^b ≤ 512 (even two flips add ≤ 768 < 1024), keeping
    // every key strictly inside its (key, next-key) window.
    let mut order_safe = Vec::with_capacity(64 * 10);
    for entry in 0..64usize {
        let key_at = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * entry;
        for bit_index in 0..10u8 {
            order_safe.push(Flip {
                byte: key_at + usize::from(bit_index / 8),
                bit: bit_index % 8,
            });
        }
    }
    // First frozen-CRC-8-cancelling pair (deterministic: fixed block,
    // fixed iteration order). Pigeonhole guarantees one exists.
    let mut by_syndrome = HashMap::<u8, Flip>::new();
    let mut found = None;
    'outer: for &flip in &order_safe {
        let mut one = block.clone();
        flip.apply(&mut one);
        let syndrome = frozen_crc8_meta(&one) ^ frozen_baseline;
        if let Some(&prior) = by_syndrome.get(&syndrome) {
            found = Some((prior, flip));
            break 'outer;
        }
        by_syndrome.insert(syndrome, flip);
    }
    let (first, second) = found.expect(
        "640 order-safe key flips over ≤ 256 CRC-8 syndromes must collide \
         (pigeonhole) — the frozen reference cannot fail to find a pair",
    );

    let mut candidate = block.clone();
    first.apply(&mut candidate);
    second.apply(&mut candidate);
    assert_eq!(
        frozen_crc8_meta(&candidate),
        frozen_baseline,
        "the pair cancels the frozen 8-bit checksum by construction"
    );
    // The upset is structurally invisible: keys still strictly ascend
    // (asserted from the raw tampered header — parse cannot be used,
    // it must reject below), and placement bytes are untouched.
    let tampered_keys: Vec<u32> = (0..64usize)
        .map(|i| {
            let e = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * i;
            u32::from_le_bytes([
                candidate[e],
                candidate[e + 1],
                candidate[e + 2],
                candidate[e + 3],
            ])
        })
        .collect();
    assert!(
        tampered_keys.windows(2).all(|w| w[0] < w[1]),
        "the chosen flips preserve strict key order by construction"
    );
    assert_ne!(tampered_keys, baseline_keys, "the upset changes key ids");

    match PropBlockView::parse(&candidate) {
        Err(err) => assert!(
            matches!(err, PropBlockError::Corrupt { ref reason } if reason.contains("checksum")),
            "only the metadata checksum can see an ordered key upset: {err}"
        ),
        Ok(view) => {
            let keys: Vec<u32> = view.key_ids().collect();
            panic!(
                "two-bit key upset {first:?}+{second:?} cancelled the metadata \
                 checksum AND parsed clean — silent wrong-key read is live again \
                 (read keys {keys:?} vs written {baseline_keys:?}); the checksum \
                 width was reverted below the round-3 CRC-32C ruling (#1452)"
            );
        }
    }
}
