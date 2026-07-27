//! Deterministic RNG + weighted reservoir sampling.
//!
//! No `rand` dependency: traversal determinism is contractual (same
//! `(seed, request, source order)` ⇒ same result — the PRIM-1
//! seeded-Fisher-Yates discipline), so the crate ships a tiny SplitMix64
//! and a heap-based Efraimidis–Spirakis weighted reservoir.
//!
//! Attribution (ADR-205 §D-3 cite-correctness): uniform reservoir
//! sampling is Vitter 1985 (Algorithm R) with the skip-based Algorithm L
//! due to Li 1994; the *weighted* reservoir here is Efraimidis–Spirakis
//! 2006 **A-Res** (one uniform draw per candidate, keep the top-s keys).
//! The A-ExpJ exponential-jump variant is a drop-in RNG-call optimization
//! deliberately deferred: candidates arrive pre-materialized from the
//! substrate (`Vec<BoundEdge>`), so per-item draws are nanoseconds, not
//! the bottleneck — adopt A-ExpJ only when a bench says otherwise.
//!
//! Budget (PD#5): `offer` is `O(log s)` heap work per *accepted*
//! candidate, `O(1)` compare-and-reject otherwise; memory `O(s)`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// SplitMix64 (Steele, Lea & Flood 2014 lineage; the JDK
/// `SplittableRandom` mixer). Tiny, fast, and good enough for sampling
/// keys — NOT cryptographic (documented non-goal).
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator. Identical seeds produce identical streams on
    /// every platform (the determinism contract).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform draw in the **open** interval `(0, 1)` — open at zero so
    /// `ln(u)` is finite (the reservoir key domain).
    pub fn next_unit_open(&mut self) -> f64 {
        // 53 mantissa bits; +1 in the numerator keeps the draw away from
        // exact 0.0 (2^-53 minimum), and the result is < 1.0 + epsilon
        // headroom keeps ln() strictly negative.
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 1.0) / ((1u64 << 53) as f64 + 2.0)
    }
}

/// Reservoir entry ordered by ascending key (min-heap root = the entry
/// that the next better candidate evicts).
struct Keyed<T> {
    /// ln-domain Efraimidis–Spirakis key: `ln(u) / w`, `u ∈ (0,1)`,
    /// `w > 0`. Larger is better; the heap is a min-heap on this value.
    ln_key: f64,
    item: T,
}

impl<T> PartialEq for Keyed<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ln_key.total_cmp(&other.ln_key) == Ordering::Equal
    }
}
impl<T> Eq for Keyed<T> {}
impl<T> PartialOrd for Keyed<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> Ord for Keyed<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // REVERSED comparison: BinaryHeap is a max-heap; reversing makes
        // `peek()` the smallest ln_key (the eviction candidate).
        other.ln_key.total_cmp(&self.ln_key)
    }
}

/// Heap-based Efraimidis–Spirakis **A-Res** weighted reservoir of fixed
/// capacity `s`: each candidate with weight `w > 0` draws `u ∈ (0,1)` and
/// keys as `u^(1/w)` (kept in ln-domain as `ln(u)/w` for precision); the
/// top-`s` keys win. Inclusion probability is proportional-ish to weight
/// (exactly the A-Res semantics), which is what the J6 §5 inverse-degree
/// down-weighting needs.
pub struct WeightedReservoir<T> {
    capacity: usize,
    heap: BinaryHeap<Keyed<T>>,
}

impl<T> WeightedReservoir<T> {
    /// Create a reservoir keeping at most `capacity` items.
    /// `capacity == 0` is rejected by the caller
    /// ([`crate::khop::k_hop`] surfaces `InvalidRequest`).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            heap: BinaryHeap::with_capacity(capacity.saturating_add(1)),
        }
    }

    /// Offer one candidate with weight `w` (non-finite or non-positive
    /// weights are clamped to a tiny positive floor so a pathological
    /// adapter cannot poison the heap with NaN keys).
    pub fn offer(&mut self, item: T, weight: f64, rng: &mut SplitMix64) {
        let w = if weight.is_finite() && weight > 0.0 {
            weight
        } else {
            f64::MIN_POSITIVE
        };
        let ln_key = rng.next_unit_open().ln() / w;
        if self.heap.len() < self.capacity {
            self.heap.push(Keyed { ln_key, item });
            return;
        }
        // Full: replace the minimum iff strictly better. `peek()` is the
        // smallest ln_key by the reversed Ord above. (Nested `if` rather
        // than a let-chain: MSRV 1.82 predates let-chain stabilization.)
        if let Some(min) = self.heap.peek() {
            if ln_key > min.ln_key {
                self.heap.pop();
                self.heap.push(Keyed { ln_key, item });
            }
        }
    }

    /// Drain the winners. Order is UNSPECIFIED by sampling semantics;
    /// callers that need determinism beyond set-equality sort the output
    /// (k-hop sorts by stream position — see `khop.rs`).
    #[must_use]
    pub fn into_items(self) -> Vec<T> {
        self.heap.into_iter().map(|k| k.item).collect()
    }

    /// Number of currently-held items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// True when nothing has been retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_is_deterministic_and_unit_open() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            let (x, y) = (a.next_unit_open(), b.next_unit_open());
            assert_eq!(x.to_bits(), y.to_bits());
            assert!(x > 0.0 && x < 1.0, "draw {x} outside (0,1)");
        }
    }

    #[test]
    fn reservoir_respects_capacity_and_determinism() {
        let run = |seed: u64| {
            let mut rng = SplitMix64::new(seed);
            let mut r = WeightedReservoir::new(8);
            for i in 0u64..1000 {
                r.offer(i, 1.0, &mut rng);
            }
            let mut got = r.into_items();
            got.sort_unstable();
            got
        };
        assert_eq!(run(7).len(), 8);
        assert_eq!(run(7), run(7), "same seed must reproduce the sample");
        assert_ne!(run(7), run(8), "different seeds should differ (w.h.p.)");
    }

    #[test]
    fn heavier_weights_win_more_often() {
        // 2 candidates, capacity 1, weight ratio 100:1 — the heavy item
        // should win the overwhelming majority of independent trials.
        let mut heavy_wins = 0u32;
        for seed in 0..2000u64 {
            let mut rng = SplitMix64::new(seed);
            let mut r = WeightedReservoir::new(1);
            r.offer("heavy", 100.0, &mut rng);
            r.offer("light", 1.0, &mut rng);
            if r.into_items() == vec!["heavy"] {
                heavy_wins += 1;
            }
        }
        // E[wins] ≈ 2000 · 100/101 ≈ 1980; require a loose floor so the
        // test is robust to RNG detail while still falsifying an unweighted
        // (≈1000) or inverted implementation.
        assert!(heavy_wins > 1800, "heavy won only {heavy_wins}/2000");
    }

    #[test]
    fn degenerate_weights_do_not_poison_the_heap() {
        let mut rng = SplitMix64::new(1);
        let mut r = WeightedReservoir::new(2);
        r.offer("nan", f64::NAN, &mut rng);
        r.offer("neg", -3.0, &mut rng);
        r.offer("ok", 1.0, &mut rng);
        let got = r.into_items();
        assert_eq!(got.len(), 2);
        assert!(got.contains(&"ok"), "finite-weight item must survive");
    }
}
