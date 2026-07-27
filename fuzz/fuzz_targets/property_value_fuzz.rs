#![no_main]
//! W22-DB-ε: PropertyValue + Filter matching fuzz target.
//!
//! # What this fuzzes
//!
//! [`arcgraph_vector::PropertyValue`] construction + the
//! [`arcgraph_vector::Filter::matches`] dispatch path. The matcher is
//! the v1.0-α filter-aware HNSW dispatch surface
//! (`crates/arcgraph-vector/src/hnsw/filtered.rs:263`). Per Slice F.2
//! the matcher is `O(|filter|)` per candidate; the contract is that
//! arbitrary filter ⨯ payload combinations evaluate without panicking.
//!
//! # Construction strategy
//!
//! The fuzz harness reads the byte stream as little-endian u32 words.
//! Each word's high 2 bits select a `PropertyValue` variant
//! (`U32 | U64 | StringId`); the remaining bits are the value payload.
//! `Filter` nodes are constructed via a similar tagged-byte
//! schema. Maximum filter depth is capped at 6 to bound recursion;
//! payload property count is capped at 16.
//!
//! # Assertion
//!
//! - **No panic.** `Filter::matches(payload)` MUST NOT panic on ANY
//!   filter / payload combination. The matcher walks a recursive
//!   tree; the cap prevents stack overflow.
//! - **`Any` is total.** `Filter::Any.matches(payload) == true` for
//!   every payload. The matcher's invariant per the docstring at
//!   `filtered.rs:264`.
//! - **`And(vec![])` is total-true; `Or(vec![])` is total-false.**
//!   Per the v1.0-α conjunctive-identity / disjunctive-identity
//!   semantics at `filtered.rs:280-282`.

use libfuzzer_sys::fuzz_target;

use arcgraph_vector::{Filter, PropertyValue};
use arcgraph_core::ids::{LabelId, StringId, TenantId};
use arcgraph_vector::hnsw::filtered::Payload;

const MAX_FILTER_DEPTH: usize = 6;
const MAX_PROPERTIES: usize = 16;
const MAX_LABELS: usize = 8;

fn property_value_from_word(word: u32) -> PropertyValue {
    match word & 0b11 {
        0 => PropertyValue::U32(word >> 2),
        1 => PropertyValue::U64(u64::from(word) << 2),
        2 => PropertyValue::StringId(StringId::new(word >> 2)),
        _ => PropertyValue::U32(word),
    }
}

fn next_u32(it: &mut std::slice::Iter<u8>) -> Option<u32> {
    let b0 = *it.next()?;
    let b1 = *it.next()?;
    let b2 = *it.next()?;
    let b3 = *it.next()?;
    Some(u32::from_le_bytes([b0, b1, b2, b3]))
}

fn build_payload(it: &mut std::slice::Iter<u8>) -> Payload {
    let mut payload = Payload::empty();
    // Optional tenant tag.
    if let Some(tag) = it.next() {
        if tag & 0x80 != 0 {
            if let Some(t) = next_u32(it) {
                payload.tenant_id = Some(TenantId::new(u64::from(t)));
            }
        }
    }
    // Labels.
    let n_labels = it.next().copied().unwrap_or(0) as usize % (MAX_LABELS + 1);
    for _ in 0..n_labels {
        if let Some(w) = next_u32(it) {
            payload.labels.push(LabelId::from(w));
        }
    }
    // Properties.
    let n_props = it.next().copied().unwrap_or(0) as usize % (MAX_PROPERTIES + 1);
    for _ in 0..n_props {
        let key = match next_u32(it) {
            Some(w) => StringId::new(w),
            None => return payload,
        };
        let val = match next_u32(it) {
            Some(w) => property_value_from_word(w),
            None => return payload,
        };
        payload.properties.insert(key, val);
    }
    payload
}

fn build_filter(it: &mut std::slice::Iter<u8>, depth: usize) -> Filter {
    if depth >= MAX_FILTER_DEPTH {
        return Filter::Any;
    }
    let tag = match it.next() {
        Some(t) => *t,
        None => return Filter::Any,
    };
    match tag % 7 {
        0 => Filter::Any,
        1 => match next_u32(it) {
            Some(w) => Filter::Tenant(TenantId::new(u64::from(w))),
            None => Filter::Any,
        },
        2 => match next_u32(it) {
            Some(w) => Filter::LabelEq(LabelId::from(w)),
            None => Filter::Any,
        },
        3 => {
            let n = it.next().copied().unwrap_or(0) as usize % (MAX_LABELS + 1);
            let mut ls = Vec::with_capacity(n);
            for _ in 0..n {
                if let Some(w) = next_u32(it) {
                    ls.push(LabelId::from(w));
                }
            }
            Filter::LabelIn(ls)
        }
        4 => {
            let k = match next_u32(it) {
                Some(w) => StringId::new(w),
                None => return Filter::Any,
            };
            let v = match next_u32(it) {
                Some(w) => property_value_from_word(w),
                None => return Filter::Any,
            };
            Filter::PropertyEq(k, v)
        }
        5 => {
            let n = it.next().copied().unwrap_or(0) as usize % 4;
            let mut cs = Vec::with_capacity(n);
            for _ in 0..n {
                cs.push(build_filter(it, depth + 1));
            }
            Filter::And(cs)
        }
        _ => {
            let n = it.next().copied().unwrap_or(0) as usize % 4;
            let mut cs = Vec::with_capacity(n);
            for _ in 0..n {
                cs.push(build_filter(it, depth + 1));
            }
            Filter::Or(cs)
        }
    }
}

fuzz_target!(|data: &[u8]| {
    // Bound input size — recursion + Vec construction grows linearly.
    if data.len() > 64 * 1024 {
        return;
    }
    let mut it = data.iter();
    let payload = build_payload(&mut it);
    let filter = build_filter(&mut it, 0);

    // Core contract — matches MUST NOT panic.
    let _ = filter.matches(&payload);

    // Semantic invariants — `Any` is total-true; `And(vec![])` is
    // total-true; `Or(vec![])` is total-false.
    assert!(
        Filter::Any.matches(&payload),
        "Filter::Any must match every payload"
    );
    assert!(
        Filter::And(vec![]).matches(&payload),
        "Filter::And(vec![]) must match every payload (conjunctive identity)"
    );
    assert!(
        !Filter::Or(vec![]).matches(&payload),
        "Filter::Or(vec![]) must reject every payload (disjunctive identity)"
    );
});
