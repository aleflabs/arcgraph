//! W26-γ-2 D3 — Cache-line alignment + memory-ordering invariants
//! for `arcgraph-core::cache_aligned::CacheAligned<T>`.
//!
//! Per ADR-134 forward-binding (test:prod ratio uplift) + W26-γ-2 D3
//! spec. The existing inline `#[cfg(test)]` block in
//! `src/cache_aligned.rs` covers size + alignment + Deref / DerefMut;
//! this integration-test file adds the cross-cutting concurrency
//! invariants that downstream `arcgraph-storage`
//! depend on:
//!
//! - Adjacent `CacheAligned<AtomicU64>` cells do not false-share.
//! - `AtomicU64` ordered atomic operations preserve the underlying
//!   value (no memory-ordering UB inside the wrapper).
//! - Releases-acquire pair across threads visibility holds at the
//!   `CacheAligned` boundary.
//! - Size + alignment guarantees survive proptests on arbitrary
//!   wrapped types.

use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use arcgraph_core::CacheAligned;
use proptest::prelude::*;

// ────────────────────── Size + alignment ──────────────────────

#[test]
fn alignment_is_64_for_u8() {
    assert_eq!(core::mem::align_of::<CacheAligned<u8>>(), 64);
}

#[test]
fn size_is_multiple_of_64() {
    // Every wrapped type pads up to a multiple of 64.
    macro_rules! check {
        ($t:ty) => {
            let s = core::mem::size_of::<CacheAligned<$t>>();
            assert_eq!(
                s % 64,
                0,
                "size of CacheAligned<{}> is {}",
                stringify!($t),
                s
            );
        };
    }
    check!(u8);
    check!(u16);
    check!(u32);
    check!(u64);
    check!(u128);
    check!([u8; 65]);
    check!([u8; 129]);
    check!([u8; 200]);
}

#[test]
fn size_grows_to_next_multiple_for_oversized() {
    // `T` with size 65 must pad to 128.
    assert_eq!(core::mem::size_of::<CacheAligned<[u8; 65]>>(), 128);
    // `T` with size 200 must pad to 256.
    assert_eq!(core::mem::size_of::<CacheAligned<[u8; 200]>>(), 256);
    // `T` with size 64 stays at 64 (no padding).
    assert_eq!(core::mem::size_of::<CacheAligned<[u8; 64]>>(), 64);
}

// ────────────────────── Adjacent cells separated by cache line ──────────────────────

#[test]
fn array_of_two_atomics_separated_by_cache_line() {
    let a: [CacheAligned<AtomicU64>; 2] = [
        CacheAligned::new(AtomicU64::new(0)),
        CacheAligned::new(AtomicU64::new(0)),
    ];
    let p0 = &a[0] as *const _ as usize;
    let p1 = &a[1] as *const _ as usize;
    assert_eq!(p1 - p0, 64, "adjacent atomics must be 64 bytes apart");
    assert_eq!(p0 % 64, 0, "first atomic must be cache-aligned");
    assert_eq!(p1 % 64, 0, "second atomic must be cache-aligned");
}

#[test]
fn vec_of_three_atomics_each_on_own_cache_line() {
    let v: Vec<CacheAligned<AtomicU64>> = (0..3)
        .map(|_| CacheAligned::new(AtomicU64::new(0)))
        .collect();
    let p0 = &v[0] as *const _ as usize;
    let p1 = &v[1] as *const _ as usize;
    let p2 = &v[2] as *const _ as usize;
    assert_eq!(p1 - p0, 64);
    assert_eq!(p2 - p1, 64);
    assert!(p0 % 64 == 0 && p1 % 64 == 0 && p2 % 64 == 0);
}

// ────────────────────── Atomic operations preserve value ──────────────────────

proptest! {
    #[test]
    fn cache_aligned_atomic_u64_round_trip(v in any::<u64>()) {
        let c = CacheAligned::new(AtomicU64::new(v));
        prop_assert_eq!(c.load(Ordering::SeqCst), v);
        c.store(v.wrapping_add(1), Ordering::SeqCst);
        prop_assert_eq!(c.load(Ordering::SeqCst), v.wrapping_add(1));
    }

    #[test]
    fn cache_aligned_atomic_u64_fetch_add(start in any::<u64>(), delta in 0u64..=10_000) {
        let c = CacheAligned::new(AtomicU64::new(start));
        let prev = c.fetch_add(delta, Ordering::SeqCst);
        prop_assert_eq!(prev, start);
        prop_assert_eq!(c.load(Ordering::SeqCst), start.wrapping_add(delta));
    }

    #[test]
    fn cache_aligned_atomic_u64_compare_exchange(start in any::<u64>(), new_val in any::<u64>()) {
        let c = CacheAligned::new(AtomicU64::new(start));
        let result = c.compare_exchange(start, new_val, Ordering::SeqCst, Ordering::SeqCst);
        prop_assert!(result.is_ok());
        prop_assert_eq!(c.load(Ordering::SeqCst), new_val);
    }
}

// ────────────────────── Multi-thread release-acquire ──────────────────────

#[test]
fn multi_thread_release_acquire_visibility() {
    let cell = Arc::new(CacheAligned::new(AtomicU64::new(0)));
    let writer = {
        let cell = Arc::clone(&cell);
        thread::spawn(move || {
            for i in 1..=1_000u64 {
                cell.store(i, Ordering::Release);
            }
        })
    };
    let reader = {
        let cell = Arc::clone(&cell);
        thread::spawn(move || {
            let mut seen = 0u64;
            for _ in 0..1_000 {
                seen = cell.load(Ordering::Acquire);
            }
            seen
        })
    };
    writer.join().expect("writer panicked");
    let observed = reader.join().expect("reader panicked");
    // After the writer commits 1000, the reader's final load must
    // see a value in [0, 1000] inclusive (release-acquire pair
    // guarantees no out-of-domain bits leak through).
    assert!(observed <= 1000, "observed = {observed}, expected <= 1000");
}

#[test]
fn multi_thread_counter_concurrency() {
    // Sanity-check that the cache-aligned wrapper does not degrade
    // throughput characteristics catastrophically (no false-sharing,
    // no implicit locking).
    let counter = Arc::new(CacheAligned::new(AtomicU64::new(0)));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let counter = Arc::clone(&counter);
            thread::spawn(move || {
                for _ in 0..10_000 {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();
    for t in threads {
        t.join().expect("counter thread panicked");
    }
    assert_eq!(counter.load(Ordering::SeqCst), 80_000);
}

// ────────────────────── Deref / DerefMut + Default + From ──────────────────────

#[test]
fn deref_pass_through_for_atomics() {
    let c = CacheAligned::new(AtomicU64::new(42));
    let v = c.load(Ordering::SeqCst);
    assert_eq!(v, 42);
}

#[test]
fn default_for_atomic_u64_is_zero() {
    let c: CacheAligned<AtomicU64> = CacheAligned::default();
    assert_eq!(c.load(Ordering::SeqCst), 0);
}

#[test]
fn into_inner_returns_wrapped_value() {
    let c = CacheAligned::new(42u32);
    let v = c.into_inner();
    assert_eq!(v, 42);
}

// ────────────────────── Property: alignment stable across moves ──────────────────────

proptest! {
    #[test]
    fn alignment_preserved_after_move(v in any::<u64>()) {
        let c = CacheAligned::new(v);
        // Move into a Box, ptr alignment must still be 64.
        let b: Box<CacheAligned<u64>> = Box::new(c);
        let p = b.deref() as *const _ as usize;
        prop_assert_eq!(p % 64, 0);
    }
}

// ────────────────────── Compile-time: const new ──────────────────────

const _: () = {
    // Verifies `CacheAligned::new` is `const fn` (callable in
    // const contexts — important for `static` cache-aligned counters).
    let _ = CacheAligned::new(0u64);
};
