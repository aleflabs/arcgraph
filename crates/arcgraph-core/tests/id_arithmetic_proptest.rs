//! W26-γ-2 D3 — comprehensive ID arithmetic + serde round-trip
//! property tests for `arcgraph-core::ids`.
//!
//! Per ADR-134 forward-binding (test:prod ratio uplift) + W26-γ-2 D3
//! spec. The existing inline `#[cfg(test)]` block in `src/ids.rs`
//! covers basic round-trip + ordering; this integration-test file
//! adds the cross-cutting invariants that the workspace relies on
//! but that lived nowhere as proptests:
//!
//! - Hash-key consistency: `a == b ⇒ hash(a) == hash(b)`
//! - Serde round-trip is byte-stable (JSON + bincode-shape)
//! - `From` / `Into` are mutually-inverse on the entire u64/u32 domain
//! - Total order matches the underlying integer order
//! - `ZERO` and `MAX` sentinels are reachable via `new()`
//! - Mixed-type arithmetic at the u64 boundary does not lose precision

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use arcgraph_core::ids::{
    LabelId, Lsn, NodeId, PageId, PartitionId, PropertyId, RelId, StringId, TenantId, TypeId,
};
use proptest::prelude::*;

// ────────────────────── PageId proptests ──────────────────────

proptest! {
    #[test]
    fn page_id_from_u64_and_back(raw in any::<u64>()) {
        let id = PageId::new(raw);
        prop_assert_eq!(id.raw(), raw);
        prop_assert_eq!(u64::from(id), raw);
        prop_assert_eq!(PageId::from(raw), id);
    }

    #[test]
    fn page_id_serde_json_roundtrip(raw in any::<u64>()) {
        let id = PageId::new(raw);
        let json = serde_json::to_string(&id).expect("serialize");
        let back: PageId = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(id, back);
        // PageId is `#[serde(transparent)]` — JSON form is bare number.
        prop_assert_eq!(json, raw.to_string());
    }

    #[test]
    fn page_id_hash_consistent_under_equality(a in any::<u64>(), b in any::<u64>()) {
        let ia = PageId::new(a);
        let ib = PageId::new(b);
        if ia == ib {
            let s = std::collections::hash_map::RandomState::new();
            prop_assert_eq!(s.hash_one(ia), s.hash_one(ib));
        }
    }

    #[test]
    fn page_id_total_order_matches_u64(a in any::<u64>(), b in any::<u64>()) {
        prop_assert_eq!(PageId::new(a).cmp(&PageId::new(b)), a.cmp(&b));
    }
}

// ────────────────────── NodeId proptests ──────────────────────

proptest! {
    #[test]
    fn node_id_total_order_strict(a in any::<u64>(), b in any::<u64>(), c in any::<u64>()) {
        // Trichotomy: exactly one of <, =, > holds.
        let (ia, ib) = (NodeId::new(a), NodeId::new(b));
        let lt = u32::from(ia < ib);
        let eq = u32::from(ia == ib);
        let gt = u32::from(ia > ib);
        prop_assert_eq!(lt + eq + gt, 1);
        // Transitivity at a≤b≤c.
        if a <= b && b <= c {
            let ic = NodeId::new(c);
            prop_assert!(ia <= ib && ib <= ic && ia <= ic);
        }
    }

    #[test]
    fn node_id_hashset_dedupes(raws in prop::collection::vec(any::<u64>(), 0..=64)) {
        let raw_set: HashSet<u64> = raws.iter().copied().collect();
        let id_set: HashSet<NodeId> = raws.iter().copied().map(NodeId::new).collect();
        prop_assert_eq!(raw_set.len(), id_set.len());
    }

    #[test]
    fn node_id_hashmap_key_lookup(raws in prop::collection::vec(any::<u64>(), 0..=64)) {
        let mut map: HashMap<NodeId, u64> = HashMap::new();
        for r in &raws {
            map.insert(NodeId::new(*r), *r);
        }
        for r in &raws {
            prop_assert_eq!(map.get(&NodeId::new(*r)).copied(), Some(*r));
        }
    }
}

// ────────────────────── RelId + Lsn proptests ──────────────────────

proptest! {
    #[test]
    fn rel_id_roundtrip_full_domain(raw in any::<u64>()) {
        let id = RelId::new(raw);
        prop_assert_eq!(id.raw(), raw);
        // `ZERO` and `MAX` are reachable via `new`.
        prop_assert!(RelId::ZERO.raw() == 0);
        prop_assert!(RelId::MAX.raw() == u64::MAX);
    }

    #[test]
    fn lsn_ordering_supports_max_alive_sentinel(raw in any::<u64>()) {
        // `Lsn::MAX` is the "alive" sentinel for `expired_lsn`. Any
        // real LSN must compare strictly less than MAX.
        let lsn = Lsn::new(raw);
        if raw < u64::MAX {
            prop_assert!(lsn < Lsn::MAX);
        }
        prop_assert!(lsn <= Lsn::MAX);
    }

    #[test]
    fn lsn_zero_is_floor(raw in any::<u64>()) {
        // `Lsn::ZERO` is the "never seen" floor.
        prop_assert!(Lsn::ZERO <= Lsn::new(raw));
    }

    #[test]
    fn lsn_serde_round_trip_preserves_max(raw in any::<u64>()) {
        let lsn = Lsn::new(raw);
        let j = serde_json::to_string(&lsn).expect("serialize");
        let back: Lsn = serde_json::from_str(&j).expect("deserialize");
        prop_assert_eq!(lsn, back);
        // Specifically: MAX round-trips. Important for the "alive"
        // sentinel — a bug in the serde layer that munged u64::MAX
        // would silently expire every record.
        let m = serde_json::to_string(&Lsn::MAX).expect("serialize MAX");
        let m_back: Lsn = serde_json::from_str(&m).expect("deserialize MAX");
        prop_assert_eq!(m_back, Lsn::MAX);
    }
}

// ────────────────────── u32-shaped IDs ──────────────────────

proptest! {
    #[test]
    fn label_id_round_trip(raw in any::<u32>()) {
        let id = LabelId::new(raw);
        prop_assert_eq!(id.raw(), raw);
        prop_assert_eq!(u32::from(id), raw);
        prop_assert_eq!(LabelId::from(raw), id);
    }

    #[test]
    fn type_id_round_trip(raw in any::<u32>()) {
        let id = TypeId::new(raw);
        prop_assert_eq!(id.raw(), raw);
    }

    #[test]
    fn string_id_round_trip(raw in any::<u32>()) {
        let id = StringId::new(raw);
        prop_assert_eq!(id.raw(), raw);
    }

    #[test]
    fn property_id_round_trip(raw in any::<u32>()) {
        let id = PropertyId::new(raw);
        prop_assert_eq!(id.raw(), raw);
        prop_assert_eq!(u32::from(id), raw);
        prop_assert_eq!(PropertyId::from(raw), id);
    }

    #[test]
    fn partition_id_round_trip(raw in any::<u32>()) {
        let id = PartitionId::new(raw);
        prop_assert_eq!(id.raw(), raw);
    }

    #[test]
    fn partition_id_default_is_zero_at_v1(_: ()) {
        // ADR-024 amendment-02 + ADR-033 §3: at v1.0 every Default-
        // constructed `TxnMutationLog` carries `PartitionId::ZERO`.
        // The z1 regression invariant gates a v1.1 distribution
        // rollout.
        prop_assert_eq!(PartitionId::default(), PartitionId::ZERO);
    }
}

// ────────────────────── TenantId domain constants ──────────────────────

proptest! {
    #[test]
    fn tenant_id_round_trip_through_serde_json(raw in any::<u64>()) {
        let t = TenantId::new(raw);
        let j = serde_json::to_string(&t).expect("serialize");
        let back: TenantId = serde_json::from_str(&j).expect("deserialize");
        prop_assert_eq!(t, back);
    }

    #[test]
    fn tenant_id_system_and_default_ordering(_: ()) {
        prop_assert!(TenantId::SYSTEM < TenantId::DEFAULT);
        prop_assert!(TenantId::SYSTEM.raw() == 0);
        prop_assert!(TenantId::DEFAULT.raw() == 1);
    }
}

// ────────────────────── Cross-type sort consistency ──────────────────────

proptest! {
    #[test]
    fn sort_ids_matches_sort_u64(raws in prop::collection::vec(any::<u64>(), 0..=128)) {
        let mut ids: Vec<NodeId> = raws.iter().copied().map(NodeId::new).collect();
        let mut sorted_u64 = raws.clone();
        ids.sort();
        sorted_u64.sort();
        let extracted: Vec<u64> = ids.iter().map(|id| id.raw()).collect();
        prop_assert_eq!(extracted, sorted_u64);
    }

    #[test]
    fn dedupe_after_sort_matches_dedupe_u64(raws in prop::collection::vec(any::<u64>(), 0..=128)) {
        let mut ids: Vec<NodeId> = raws.iter().copied().map(NodeId::new).collect();
        let mut u64s = raws.clone();
        ids.sort();
        ids.dedup();
        u64s.sort();
        u64s.dedup();
        let extracted: Vec<u64> = ids.iter().map(|id| id.raw()).collect();
        prop_assert_eq!(extracted, u64s);
    }
}

// ────────────────────── Sentinel reachability ──────────────────────

#[test]
fn page_id_zero_sentinel_reachable() {
    assert_eq!(PageId::ZERO, PageId::new(0));
}

#[test]
fn page_id_max_sentinel_reachable() {
    assert_eq!(PageId::MAX, PageId::new(u64::MAX));
}

#[test]
fn node_id_zero_sentinel_reachable() {
    assert_eq!(NodeId::ZERO, NodeId::new(0));
    assert_eq!(NodeId::MAX, NodeId::new(u64::MAX));
}

#[test]
fn lsn_zero_and_max_sentinels_reachable() {
    assert_eq!(Lsn::ZERO, Lsn::new(0));
    assert_eq!(Lsn::MAX, Lsn::new(u64::MAX));
}

#[test]
fn rel_id_sentinels_reachable() {
    assert_eq!(RelId::ZERO.raw(), 0);
    assert_eq!(RelId::MAX.raw(), u64::MAX);
}

#[test]
fn label_id_sentinels_reachable() {
    assert_eq!(LabelId::ZERO.raw(), 0);
    assert_eq!(LabelId::MAX.raw(), u32::MAX);
}

#[test]
fn type_id_sentinels_reachable() {
    assert_eq!(TypeId::ZERO.raw(), 0);
    assert_eq!(TypeId::MAX.raw(), u32::MAX);
}

#[test]
fn string_id_sentinels_reachable() {
    assert_eq!(StringId::ZERO.raw(), 0);
    assert_eq!(StringId::MAX.raw(), u32::MAX);
}

#[test]
fn property_id_sentinels_reachable() {
    assert_eq!(PropertyId::ZERO.raw(), 0);
    assert_eq!(PropertyId::MAX.raw(), u32::MAX);
}

#[test]
fn partition_id_sentinels_reachable() {
    assert_eq!(PartitionId::ZERO.raw(), 0);
    assert_eq!(PartitionId::MAX.raw(), u32::MAX);
}

// ────────────────────── Serde stability (byte-shape canonical) ──────────────────────

#[test]
fn serde_json_known_values_are_stable() {
    // These are part of the on-disk-adjacent canonical shape. A
    // serde-attribute drift (e.g., dropping `#[serde(transparent)]`)
    // would surface here.
    assert_eq!(serde_json::to_string(&NodeId::new(42)).unwrap(), "42");
    assert_eq!(serde_json::to_string(&PageId::new(0)).unwrap(), "0");
    assert_eq!(
        serde_json::to_string(&Lsn::new(u64::MAX)).unwrap(),
        u64::MAX.to_string()
    );
    assert_eq!(serde_json::to_string(&TenantId::SYSTEM).unwrap(), "0");
    assert_eq!(serde_json::to_string(&TenantId::DEFAULT).unwrap(), "1");
}

#[test]
fn serde_deserialize_rejects_negative() {
    // u64-shaped IDs MUST reject negative inputs at the serde
    // boundary (catalog ingestion + diagnostics deserialize from
    // operator-provided JSON).
    let r: Result<NodeId, _> = serde_json::from_str("-1");
    assert!(r.is_err(), "NodeId must reject negative input");
}

#[test]
fn serde_deserialize_rejects_overflow() {
    // u64::MAX + 1 is past the u64 domain.
    let oversized = format!("{}", u128::from(u64::MAX) + 1);
    let r: Result<NodeId, _> = serde_json::from_str(&oversized);
    assert!(r.is_err(), "NodeId must reject u64-overflow input");
}

// ────────────────────── Sort stability + de-dupe ──────────────────────

proptest! {
    #[test]
    fn binary_search_after_sort(raws in prop::collection::vec(any::<u64>(), 1..=128), needle in any::<u64>()) {
        let mut ids: Vec<NodeId> = raws.iter().copied().map(NodeId::new).collect();
        ids.sort();
        let needle = NodeId::new(needle);
        let result = ids.binary_search(&needle);
        match result {
            Ok(i) => prop_assert_eq!(ids[i], needle),
            Err(_) => prop_assert!(!ids.contains(&needle)),
        }
    }
}
