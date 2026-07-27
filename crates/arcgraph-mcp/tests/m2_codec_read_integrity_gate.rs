//! v2 M2 — codec read-integrity RED-on-revert gates at the mcp read
//! boundary (L1 review defects A2/A1; promoted from the QA scratch
//! `l1_m2_wrong_read_and_intern_order_scratch.rs`, flipped from
//! assert-the-bug to assert-the-contract).
//!
//! **A2 — global key ordering.** The primary header and the overflow
//! directory partition ONE globally-sorted bag: every xentry key is
//! strictly greater than every primary key. The lazy-overflow read
//! contract (design §M2.3 — a key that resolves in the primary never
//! fetches the overflow) is only sound under that invariant; a
//! cross-directory duplicate made the FULL read (xentry last-wins
//! over the primary insert) and the PROJECTED read (primary hit,
//! overflow untouched) return DIFFERENT VALUES silently — the
//! wrong-read class. `OverflowView::parse` now takes the primary max
//! key and rejects any violation loudly.
//!
//! **A1 — metadata integrity.** In-range metadata corruption (an
//! offset redirect that stays inside the data region) was silently
//! accepted and survived the full storage round trip; the canonical-
//! placement march + the metadata CRC-32C (round-3, #1452: widened
//! from round-1's flags-byte CRC-8 after the codex re-check measured
//! 8 bits too narrow) make it loud at the parse the mcp read bridge
//! performs. The storage-grain triplet (redirect / same-width retype
//! / truncation, with recomputed-checksum variants) lives in
//! `arcgraph-storage/tests/m2_codec_integrity_gate.rs`; this file
//! pins the END-TO-END read boundary: `record_property_bag_*`
//! surfaces `Err`, never a wrong value.
//!
//! RED-on-revert: drop the `key_id <= pmax` check in
//! `OverflowView::parse` and `cross_directory_duplicate_key_is_loud_
//! corrupt_never_divergent_reads` fails on its first `expect_err`
//! (the full read silently returns the xentry's value again). Drop
//! the canonical-placement march and
//! `canonical_metadata_corruption_is_loud_corrupt_through_the_read_path`
//! silently reads key `a` as 222 again. Narrow the metadata checksum
//! back to 8 bits and `uncrafted_two_bit_key_scribble_is_loud_corrupt_
//! not_a_wrong_name_read` materializes values under WRONG existing
//! property names again (the round-3 defect).

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_mcp::storage::property_payload::{
    ResolvedProjection, properties_to_property_data_typed, record_property_bag_checked,
    record_property_bag_projected,
};
use arcgraph_query::executor::value::Value;
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, read_node_with_store};
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::prop_block::{
    BLOCK_HEADER_SIZE, HEADER_ENTRY_SIZE, PropBlockBuilder, PropValue, TypedBagParts,
    stamp_block_meta_checksum,
};
use arcgraph_storage::transaction::TxnManager;

/// Build the 65-property bag (64 primary entries + 1 xentry), tamper
/// the sole xentry's key to `rekey_to`, stage + commit it through the
/// production crud path (storage stages opaque payload bytes —
/// integrity is a READ-time contract), and return the read context.
fn stage_tampered_wide_bag(intern: &InternTable, rekey_to: u32) -> (TxnManager, CrudStore, NodeId) {
    let tenant = TenantId::DEFAULT;
    let mut builder = PropBlockBuilder::new();
    for i in 1..=65u32 {
        let name = format!("k{i:03}");
        let id = intern.intern(tenant, &name).unwrap();
        assert_eq!(id.raw(), i);
        builder.put(i, PropValue::Int(i64::from(i)));
    }
    let encoded = builder.build().expect("build wide bag");
    let mut overflow = encoded
        .overflow_payload()
        .expect("65th property spills")
        .to_vec();
    let block = encoded.into_block_bytes_deferred().expect("deferred block");

    // The only xentry originally has key 65 (> every primary key, per
    // the global-ordering invariant). Re-key it INSIDE the primary
    // range — and RECOMPUTE the directory checksum (A1) so the
    // GLOBAL-ORDERING check is the load-bearing detector this gate
    // pins (a raw tamper would trip the checksum first and this gate
    // would stop being RED on an ordering-check revert).
    let xkey_at = arcgraph_storage::prop_block::OVERFLOW_HEADER_SIZE;
    overflow[xkey_at..xkey_at + 4].copy_from_slice(&rekey_to.to_le_bytes());
    let recomputed = arcgraph_storage::prop_block::compute_overflow_meta_checksum(&overflow);
    overflow[4..8].copy_from_slice(&recomputed.to_le_bytes());

    let mgr = TxnManager::new();
    let store = CrudStore::new();
    let mut tx = mgr.begin(tenant);
    let node = create_node(
        &store,
        &mut tx,
        tenant,
        LabelId::new(0),
        &PropertyData::TypedBlock(TypedBagParts {
            block,
            overflow: Some(overflow),
        }),
    )
    .expect("stage wide node");
    commit(tx, &store).expect("commit wide node");
    (mgr, store, node)
}

/// The A2 gate: a cross-directory DUPLICATE key (xentry re-keyed onto
/// primary key 1) must be loud `Corrupt` on every read that consults
/// the overflow directory — never two different `Ok` values.
#[test]
fn cross_directory_duplicate_key_is_loud_corrupt_never_divergent_reads() {
    let tenant = TenantId::DEFAULT;
    let intern = InternTable::new();
    let (mgr, store, node) = stage_tampered_wide_bag(&intern, 1);

    let read_tx = mgr.begin(tenant);
    let record = read_node_with_store(&store, &read_tx, node)
        .expect("read record")
        .expect("node exists");

    // Full-bag materialization consults the overflow → LOUD.
    let err = record_property_bag_checked(&record, store.blob_store(), &intern, tenant)
        .expect_err("cross-directory duplicate key must be loud Corrupt on the full read");
    assert!(
        err.to_string().contains("globally ordered"),
        "the error names the violated invariant: {err}"
    );

    // A projected read that routes to the overflow (key 65 sorts past
    // the primary range) → LOUD too.
    let wide_proj = ResolvedProjection::resolve(&["k065".to_string()], &intern, tenant).unwrap();
    let err =
        record_property_bag_projected(&record, store.blob_store(), &intern, tenant, &wide_proj)
            .expect_err("an overflow-consulting projected read must be loud Corrupt");
    assert!(err.to_string().contains("globally ordered"), "{err}");

    // A primary-resolved projected read never fetches the overflow
    // (the §M2.3 lazy contract — the same scoping that keeps payload-
    // byte corruption an on-access fault), so it answers from the
    // intact primary entry. The defect being pinned closed is the
    // DIVERGENT-VALUES pair: pre-fix, full read returned 65 for k001
    // while this returned 1 — both Ok, silently different. Post-fix no
    // two Ok reads can disagree: every overflow-consulting read errs.
    let narrow_proj = ResolvedProjection::resolve(&["k001".to_string()], &intern, tenant).unwrap();
    let narrow =
        record_property_bag_projected(&record, store.blob_store(), &intern, tenant, &narrow_proj)
            .expect("primary-resolved projected read (overflow untouched)");
    assert_eq!(narrow.get("k001"), Some(&Value::Integer(1)));
}

/// A2, misorder-without-duplicate variant: an xentry key BELOW the
/// primary max that collides with NO primary key (0 here) is equally
/// corrupt — it would be an unreachable ghost entry under the lazy
/// contract (a get(0) answers `Absent` from the primary alone).
#[test]
fn cross_directory_misordered_key_is_loud_corrupt_even_without_duplicate() {
    let tenant = TenantId::DEFAULT;
    let intern = InternTable::new();
    let (mgr, store, node) = stage_tampered_wide_bag(&intern, 0);

    let read_tx = mgr.begin(tenant);
    let record = read_node_with_store(&store, &read_tx, node)
        .expect("read record")
        .expect("node exists");
    let err = record_property_bag_checked(&record, store.blob_store(), &intern, tenant)
        .expect_err("misordered (ghost) xentry key must be loud Corrupt");
    assert!(err.to_string().contains("globally ordered"), "{err}");
}

/// The A1 gate at the read boundary: an in-range offset redirect —
/// key `a`'s valid, aligned 8-byte extent redirected onto key `b`'s
/// equally-shaped extent, with the block checksum RECOMPUTED so the
/// canonical-placement layer is the load-bearing detector — survives
/// staging + commit (payload bytes are opaque to storage; the page
/// CRC is computed over the already-tampered bytes) and MUST surface
/// as `Err` from the checked read. Pre-fix this returned
/// `{a: 222, b: 222}` — a silent wrong read.
#[test]
fn canonical_metadata_corruption_is_loud_corrupt_through_the_read_path() {
    let tenant = TenantId::DEFAULT;
    let intern = InternTable::new();
    let a = intern.intern(tenant, "a").unwrap();
    let b = intern.intern(tenant, "b").unwrap();

    let mut builder = PropBlockBuilder::new();
    builder.put(a.raw(), PropValue::Int(111));
    builder.put(b.raw(), PropValue::Int(222));
    let mut block = builder
        .build()
        .expect("build")
        .into_block_bytes(None)
        .expect("finalize");

    let a_entry = BLOCK_HEADER_SIZE;
    let b_entry = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE;
    let b_offset = [block[b_entry + 6], block[b_entry + 7]];
    block[a_entry + 6..a_entry + 8].copy_from_slice(&b_offset);
    stamp_block_meta_checksum(&mut block);

    let mgr = TxnManager::new();
    let store = CrudStore::new();
    let mut tx = mgr.begin(tenant);
    let node = create_node(
        &store,
        &mut tx,
        tenant,
        LabelId::new(0),
        &PropertyData::TypedBlock(TypedBagParts {
            block,
            overflow: None,
        }),
    )
    .expect("stage node");
    commit(tx, &store).expect("commit node");

    let read_tx = mgr.begin(tenant);
    let record = read_node_with_store(&store, &read_tx, node)
        .expect("read record")
        .expect("node exists");
    let err = record_property_bag_checked(&record, store.blob_store(), &intern, tenant)
        .expect_err("in-range metadata corruption must be loud Corrupt at the read boundary");
    assert!(
        err.to_string().contains("canonical"),
        "canonical placement names the violation: {err}"
    );
}

/// Frozen CRC-8/ATM (poly 0x07) over the SAME metadata coverage the
/// production checksum uses — the HISTORICAL round-1 width. Kept
/// test-local (and deliberately duplicated across the width gates, so
/// no shared helper can be "fixed" into silently weakening them): a
/// width revert reproduces exactly this function's collision
/// structure. Storage-grain siblings:
/// `arcgraph-storage/tests/m2_metadata_checksum_width_gate.rs`.
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

/// **A1 round-3 (#1452) — the production-path wrong-NAME repro, now a
/// width RED-on-revert gate.** The codex re-check measured 2,378 of
/// the 2,096,128 two-bit key-field upsets clearing CRC-8 + strict
/// ordering + canonical placement, and demonstrated one end-to-end:
/// staged through the production crud path and materialized by
/// `record_property_bag_checked` under WRONG EXISTING property names
/// (every id involved is a real interned name — an operator sees
/// plausible properties holding other properties' values). This gate
/// reconstructs that survivor shape deterministically against the
/// frozen historical CRC-8 and pins the round-3 contract: the widened
/// checksum rejects it LOUD at the read boundary.
///
/// The upset is UNCRAFTED (two independent bit flips, the original
/// CRC-32C stamp untouched) and structurally invisible: keys stride
/// 64 with flips confined to the low 6 zero bits, so ordering,
/// placement, tags and lengths all stay valid — the checksum is the
/// only layer that can see it (the resident-scribble posture:
/// `open_prop_trusted` skips the page CRC32C by design).
///
/// RED-on-revert: narrow the metadata checksum back to 8 bits and the
/// pair cancels the (reverted) stamp — the checked read returns `Ok`
/// and this gate panics printing the wrong names it materialized.
#[test]
fn uncrafted_two_bit_key_scribble_is_loud_corrupt_not_a_wrong_name_read() {
    let tenant = TenantId::DEFAULT;
    let intern = InternTable::new();
    // Intern a real tenant-wide name space: ids 1..=4160 all carry
    // names, so every id a low-bit flip can produce (≤ 4096 + 32) is
    // an EXISTING property name — the wrong-NAME (not unknown-id)
    // read is the defect shape.
    for i in 1..=4160u32 {
        let id = intern.intern(tenant, &format!("k{i:04}")).unwrap();
        assert_eq!(id.raw(), i);
    }

    // The staged bag: 64 primary entries at ids 64, 128, …, 4096
    // (stride 64 — a sparse subset of the interned space, the shape
    // real bags have). Values encode their own key for readability.
    let mut builder = PropBlockBuilder::new();
    for i in 1..=64u32 {
        builder.put(i * 64, PropValue::Int(i64::from(i * 64)));
    }
    let block = builder
        .build()
        .expect("build")
        .into_block_bytes(None)
        .expect("finalize");
    let written_names: Vec<String> = (1..=64u32).map(|i| format!("k{:04}", i * 64)).collect();

    // Deterministic survivor search under the FROZEN CRC-8: flips
    // confined to bits 0..6 of each key's low byte (zero on a 64
    // stride ⇒ a flip ADDS 2^b ≤ 32, keys stay strictly ordered and
    // every produced id stays interned). 384 flips over ≤ 256
    // syndromes ⇒ a colliding pair exists; two flips in the SAME byte
    // never collide (an intra-byte 2-bit burst is within CRC-8's
    // guaranteed detection), so the pair spans two entries — TWO
    // properties materialize under wrong names.
    let frozen_baseline = frozen_crc8_meta(&block);
    let mut by_syndrome = std::collections::HashMap::<u8, (usize, u8)>::new();
    let mut found = None;
    'outer: for entry in 0..64usize {
        let key_at = BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * entry;
        for bit in 0..6u8 {
            let mut one = block.clone();
            one[key_at] ^= 1 << bit;
            let syndrome = frozen_crc8_meta(&one) ^ frozen_baseline;
            if let Some(&(prior_entry, prior_bit)) = by_syndrome.get(&syndrome) {
                found = Some(((prior_entry, prior_bit), (entry, bit)));
                break 'outer;
            }
            by_syndrome.insert(syndrome, (entry, bit));
        }
    }
    let ((entry_a, bit_a), (entry_b, bit_b)) = found
        .expect("384 order-safe key flips over ≤ 256 CRC-8 syndromes must collide (pigeonhole)");
    assert_ne!(entry_a, entry_b, "same-byte pairs cannot cancel CRC-8");

    let mut scribbled = block;
    scribbled[BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * entry_a] ^= 1 << bit_a;
    scribbled[BLOCK_HEADER_SIZE + HEADER_ENTRY_SIZE * entry_b] ^= 1 << bit_b;
    assert_eq!(
        frozen_crc8_meta(&scribbled),
        frozen_baseline,
        "the pair cancels the frozen 8-bit checksum by construction"
    );
    // The names this scribble would forge the values under.
    let wrong_names = [
        format!("k{:04}", (entry_a as u32 + 1) * 64 + (1u32 << bit_a)),
        format!("k{:04}", (entry_b as u32 + 1) * 64 + (1u32 << bit_b)),
    ];

    // Stage + commit through the production crud path (storage stores
    // opaque payload bytes; integrity is a READ-time contract).
    let mgr = TxnManager::new();
    let store = CrudStore::new();
    let mut tx = mgr.begin(tenant);
    let node = create_node(
        &store,
        &mut tx,
        tenant,
        LabelId::new(0),
        &PropertyData::TypedBlock(TypedBagParts {
            block: scribbled,
            overflow: None,
        }),
    )
    .expect("stage scribbled node");
    commit(tx, &store).expect("commit scribbled node");

    let read_tx = mgr.begin(tenant);
    let record = read_node_with_store(&store, &read_tx, node)
        .expect("read record")
        .expect("node exists");
    match record_property_bag_checked(&record, store.blob_store(), &intern, tenant) {
        Err(err) => assert!(
            err.to_string().contains("checksum"),
            "only the metadata checksum can see an ordered key upset: {err}"
        ),
        Ok(bag) => {
            let forged: Vec<&String> = wrong_names
                .iter()
                .filter(|name| bag.contains_key(name.as_str()))
                .collect();
            panic!(
                "uncrafted two-bit key scribble (entry {entry_a} bit {bit_a} + entry \
                 {entry_b} bit {bit_b}) survived the checked read: values materialized \
                 under wrong existing names {forged:?} (written names were a stride-64 \
                 subset of {}..{}) — the metadata checksum width was reverted below \
                 the round-3 CRC-32C ruling (#1452)",
                written_names[0], written_names[63],
            );
        }
    }
}

// ─── Contract pins carried over from the L1 scratch (asserted the
//     CORRECT behavior from day one; kept as standing gates) ─────────

/// Mixed-store dispatch pin: the SAME logical bag stored as an M1
/// legacy JSON blob and as an M2 typed block materializes identically
/// through the checked read (the migrate-on-open window's core
/// invariant), including a stored Null.
#[test]
fn mixed_json_and_typed_store_dispatches_value_identically() {
    let tenant = TenantId::DEFAULT;
    let intern = InternTable::new();
    let mgr = TxnManager::new();
    let store = CrudStore::new();
    let props = vec![
        ("f".to_string(), Value::Float(0.1)),
        ("k".to_string(), Value::Integer(7)),
        ("null".to_string(), Value::Null),
    ];
    let mut json = serde_json::Map::new();
    for (key, value) in &props {
        json.insert(key.clone(), value.to_json_value());
    }
    let json_bytes = serde_json::to_vec(&serde_json::Value::Object(json)).expect("json");
    let typed =
        properties_to_property_data_typed(&props, &intern, None, tenant).expect("typed encode");

    let mut tx = mgr.begin(tenant);
    let json_node = create_node(
        &store,
        &mut tx,
        tenant,
        LabelId::new(0),
        &PropertyData::Blob(json_bytes),
    )
    .expect("json node");
    let typed_node =
        create_node(&store, &mut tx, tenant, LabelId::new(0), &typed).expect("typed node");
    commit(tx, &store).expect("mixed commit");

    let read_tx = mgr.begin(tenant);
    let json_record = read_node_with_store(&store, &read_tx, json_node)
        .expect("read json")
        .expect("json node exists");
    let typed_record = read_node_with_store(&store, &read_tx, typed_node)
        .expect("read typed")
        .expect("typed node exists");
    let json_bag = record_property_bag_checked(&json_record, store.blob_store(), &intern, tenant)
        .expect("legacy dispatch");
    let typed_bag = record_property_bag_checked(&typed_record, store.blob_store(), &intern, tenant)
        .expect("typed dispatch");
    assert_eq!(json_bag, typed_bag);
    assert_eq!(typed_bag.get("null"), Some(&Value::Null));
}

/// Reserved-tag encode pin: `Bytes`/`Temporal` values past the 255-B
/// inline cap REJECT at encode (no v1.0-α producer exists and
/// `PROP_BLOCK_VERSION = 1` defines no overflow form for them) — a
/// loud error, never a silent truncation or an undefined spill. The
/// M2.1 "values > 255 B spill" sentence applies to the tags with an
/// overflow form (strings, nested list/map); widening the reserved
/// tags is deliberately deferred to their first producer.
#[test]
fn oversize_bytes_and_temporal_are_rejected_instead_of_spilled() {
    for value in [
        PropValue::Bytes(vec![0xAB; 256]),
        PropValue::Temporal(vec![0xCD; 256]),
    ] {
        let mut builder = PropBlockBuilder::new();
        builder.put(1, value);
        let err = builder
            .build()
            .expect_err("oversize reserved-tag values reject at encode");
        assert!(err.to_string().contains("exceeds the 255-byte inline cap"));
    }
}
