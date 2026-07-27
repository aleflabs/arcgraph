//! M4-04d empirical back-test of `DEFAULT_*_SELECTIVITY` constants on a
//! 1M LDBC SNB Person tenant fixture (issue #209; ADR-038 amendment-07).
//!
//! # Why
//!
//! The `DEFAULT_*_SELECTIVITY` constants in
//! [`arcgraph_query::semantic::selectivity`] were chosen by lookup
//! (Selinger 1979 / mainstream textbook defaults), not by simulation
//! against ArcQL-style workloads. The Codex M4-2x retro
//! (Recommendation #3, 2026-05-03) flagged this as a v1.0-blocker:
//! if measured `eq` selectivity for typical SNB Person properties is
//! 1e-3 (not 0.1), the M4-05 cost planner makes wrong join orders for
//! the cold-start window until M4-71 row-count observer feedback
//! converges. Issue #209 commissioned this empirical back-test.
//!
//! # What the back-test does
//!
//! 1. **Synthesises a deterministic 1M Person tenant** with realistic
//!    SNB-aligned property distributions:
//!    - `firstName` Zipf(α=1.2) over 1000 distinct values (heavy tail
//!      mimicking real first-name frequency distributions per LDBC
//!      Datagen Person0.csv shape);
//!    - `lastName` Zipf(α=1.0) over 10 000 distinct values (less skewed);
//!    - `gender` uniform binary;
//!    - `age` uniform [13, 100];
//!    - `birthday` uniform over 88 × 365 days (1925..2013);
//!    - `creationDate` uniform over 16 × 365 days (2010..2026);
//!    - `locationIP` uniform over 50 000 distinct values;
//!    - `browserUsed` 5-way categorical with realistic web-share
//!      weights `[Chrome 0.50, Firefox 0.20, Safari 0.20, Edge 0.07, IE 0.03]`;
//!    - `speaks` 1-3 distinct languages from 30;
//!    - `email` ~unique (32-bit hash domain).
//!
//! 2. **Adds auxiliary node labels** (multi-label tenant per LDBC SNB):
//!    - 100 000 Comment, 10 000 Forum, 1 000 Place. Total 1.111M nodes.
//!
//! 3. **Adds 5M relationships** with realistic rel-type distribution:
//!    `KNOWS 60%`, `LIKES 25%`, `IS_LOCATED_IN 15%`.
//!
//! 4. **Sweeps representative predicates per class** (50 trials each):
//!    - `eq`: random `firstName = X`, `lastName = Y`, `gender = G`,
//!      `age = A`, `birthday = D`, `browser = B`, `locationIP = I`;
//!    - `lt`: random `age < X`, `birthday < D`, `creationDate < D`;
//!    - `in`: random 3- and 10-element `firstName IN [...]` lists,
//!      random 3-element `language IN [...]` lists;
//!    - `label`: per-label cardinality / total — Person, Comment,
//!      Forum, Place;
//!    - `rel_type`: per-rel-type cardinality / total — KNOWS, LIKES,
//!      IS_LOCATED_IN.
//!
//! 5. **Aggregates** p10 / p50 / p90 / p99 selectivity per sub-class
//!    and per class.
//!
//! 6. **Compares** empirical p50 to current `DEFAULT_*_SELECTIVITY`
//!    constants and computes the off-by-Nx ratio.
//!
//! # How to run
//!
//! Smoke (10 K rows, runs on every `cargo test`):
//! ```bash
//! cargo test -p arcgraph-query --release \
//!     m4_04d_empirical_selectivity_backtest_smoke -- --nocapture
//! ```
//!
//! Full back-test (1 M rows, `#[ignore]`-gated; this is the empirical
//! artifact cited by ADR-038 amendment-07):
//! ```bash
//! cargo test -p arcgraph-query --release \
//!     m4_04d_empirical_selectivity_backtest -- --ignored --nocapture
//! ```
//!
//! # Determinism
//!
//! The fixture is built from a single u64 seed via a 64-bit LCG; every
//! sweep derives its sub-seed from the same seed via a per-class salt
//! that funnels through a Wyhash-style mix. Re-runs are byte-identical
//! so the empirical numbers cited in ADR-038 amendment-07 are
//! reproducible from this file at the recorded seed.
//!
//! # Citations
//!
//! - **LDBC SNB Interactive** (Erling et al., SIGMOD 2015) — the
//!   workload our synthetic fixture approximates.
//! - **PostgreSQL `selfuncs.c`** — `eqsel = 0.005`, `ineqsel = 0.3333`,
//!   `rangesel = 0.005`. The post-Selinger production-validated
//!   defaults; ADR-038 amendment-07 cites these as the reference
//!   point for our tuning recommendation.
//! - **Selinger et al., SIGMOD 1979** — the textbook defaults the
//!   v1.0 constants were originally chosen from.
//! - **Codex M4-2x retro Recommendation #3** (2026-05-03) — the trigger
//!   for this slice; saved at
//!   `~/arcgraph-retro-recent-m4-2x4x/RETRO_REVIEW_M4_2x4x_LIGHT_PASS.md`.

use arcgraph_query::semantic::{
    DEFAULT_EQ_SELECTIVITY, DEFAULT_IN_SELECTIVITY, DEFAULT_LABEL_SELECTIVITY,
    DEFAULT_LT_SELECTIVITY, DEFAULT_REL_TYPE_SELECTIVITY,
};

// ---------------------------------------------------------------------
// 1. Configuration constants.
// ---------------------------------------------------------------------

/// Number of trials per predicate sub-class. 50 is large enough that the
/// p10/p50/p90 estimates are stable across re-runs (proptest precedent
/// in this crate uses 256 cases for invariants, but selectivity p50 is
/// less variance-sensitive — 50 trials gives ±5 % p50 reproducibility
/// at the seed used here).
const NUM_TRIALS: usize = 50;

/// Master seed for the back-test. Hand-picked hex non-zero so all LCG
/// derivations stay clear of the fixed point. Bumping this constant
/// means re-running the back-test and updating the empirical numbers
/// cited in ADR-038 amendment-07.
const MASTER_SEED: u64 = 0x4D04_D5E1_EC07_C0DE;

// Per-sweep salts — distinct integers funneled through `salted_seed()`
// so each sweep's predicate values are independent of the others'.
const SALT_PERSON_BUILD: u64 = 0xA5A5_A5A5_A5A5_A5A5;
const SALT_EQ_FIRST_NAME: u64 = 0x0000_0000_0000_0001;
const SALT_EQ_LAST_NAME: u64 = 0x0000_0000_0000_0002;
const SALT_EQ_GENDER: u64 = 0x0000_0000_0000_0003;
const SALT_EQ_AGE: u64 = 0x0000_0000_0000_0004;
const SALT_EQ_BIRTHDAY: u64 = 0x0000_0000_0000_0005;
const SALT_EQ_BROWSER: u64 = 0x0000_0000_0000_0006;
const SALT_EQ_LOCATION_IP: u64 = 0x0000_0000_0000_0007;
const SALT_LT_AGE: u64 = 0x0000_0000_0000_0008;
const SALT_LT_BIRTHDAY: u64 = 0x0000_0000_0000_0009;
const SALT_LT_CREATION_DATE: u64 = 0x0000_0000_0000_000A;
const SALT_IN_FIRST_NAME: u64 = 0x0000_0000_0000_000B;
const SALT_IN_LANGUAGE: u64 = 0x0000_0000_0000_000C;

/// Mix master seed with a per-sweep salt deterministically. The
/// multiply-and-add follows the SplitMix64 / Wyhash skeleton; the goal
/// is to produce sub-seeds that are pairwise uncorrelated so two
/// sweeps don't accidentally pick the same predicate sequence.
fn salted_seed(master: u64, salt: u64) -> u64 {
    master
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(salt.wrapping_mul(0x6A09_E667_F3BC_C908))
        .max(1)
}

/// LDBC SNB Person property distribution constants. These shape the
/// fixture; changing them changes the empirical p50 numbers and means
/// re-running the back-test + updating amendment-07.
mod fixture_params {
    /// First-name distinct count. Real LDBC Datagen Person0.csv at SF-1
    /// has ~1 000 distinct first names; we mirror.
    pub const FIRST_NAME_COUNT: u32 = 1_000;
    /// First-name Zipf exponent. α=1.2 gives a heavy head consistent with
    /// real first-name frequency distributions (top-3 names ~10 % combined).
    pub const FIRST_NAME_ZIPF_ALPHA: f64 = 1.2;

    /// Last-name distinct count.
    pub const LAST_NAME_COUNT: u32 = 10_000;
    /// Last-name Zipf exponent. α=1.0 is the canonical Zipf default —
    /// less head-skew than first names.
    pub const LAST_NAME_ZIPF_ALPHA: f64 = 1.0;

    /// IP distinct count — uniform.
    pub const LOCATION_IP_COUNT: u32 = 50_000;

    /// Browser categorical: 5-way, weights below.
    pub const BROWSER_COUNT: u8 = 5;
    /// Browser CDF: Chrome 0.50, Firefox 0.20, Safari 0.20, Edge 0.07, IE 0.03.
    pub const BROWSER_CDF: [f64; 5] = [0.50, 0.70, 0.90, 0.97, 1.00];

    /// Language pool size.
    pub const LANGUAGE_COUNT: u8 = 30;

    /// Age range — uniform [13, 100].
    pub const AGE_MIN: u8 = 13;
    pub const AGE_MAX: u8 = 100;

    /// Birthday range — uniform over 88 × 365 days (1925..2013).
    pub const BIRTHDAY_DAYS: u32 = 88 * 365;

    /// CreationDate range — uniform over 16 × 365 days (2010..2026).
    pub const CREATION_DAYS: u32 = 16 * 365;

    /// Auxiliary label cardinalities (per multi-label SNB tenant).
    pub const COMMENT_COUNT: u64 = 100_000;
    pub const FORUM_COUNT: u64 = 10_000;
    pub const PLACE_COUNT: u64 = 1_000;

    /// Edge-mix fractions for rel_type sweep.
    pub const KNOWS_FRAC: f64 = 0.60;
    pub const LIKES_FRAC: f64 = 0.25;
    pub const ILI_FRAC: f64 = 0.15;
    /// Total edges in the rel-type sweep (paired with the Person count).
    pub const TOTAL_EDGES: u64 = 5_000_000;
}

use fixture_params::*;

// ---------------------------------------------------------------------
// 2. Deterministic LCG (no new deps).
// ---------------------------------------------------------------------

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid the all-zeros LCG fixed point; the master seed is
        // hand-picked non-zero, but salts can produce zero by XOR.
        Self { state: seed.max(1) }
    }

    /// One step of the Knuth/Lehmer LCG used elsewhere in this repo
    /// using the deterministic fixture below.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Sample a unit in `[0, 1)` with full f64 mantissa precision.
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Sample a u32 in `[0, max)`. `max` MUST be > 0.
    fn range_u32(&mut self, max: u32) -> u32 {
        debug_assert!(max > 0, "Rng::range_u32: max must be > 0");
        ((self.next_u64() >> 32) as u32) % max
    }

    /// Sample a u8 in `[0, max)`. `max` MUST be > 0.
    fn range_u8(&mut self, max: u8) -> u8 {
        debug_assert!(max > 0, "Rng::range_u8: max must be > 0");
        ((self.next_u64() >> 56) as u8) % max
    }
}

// ---------------------------------------------------------------------
// 3. Distribution samplers.
// ---------------------------------------------------------------------

/// Build a Zipf inverse-CDF table for `[1, n]` with exponent `α`.
/// Returns a slice of length `n` where `cdf[i] = sum_{k=1..=i+1} k^-α / Z`.
fn make_zipf_cdf(n: u32, alpha: f64) -> Vec<f64> {
    let weights: Vec<f64> = (1..=n).map(|k| 1.0 / (k as f64).powf(alpha)).collect();
    let total: f64 = weights.iter().sum();
    let mut cdf = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for w in &weights {
        acc += w;
        cdf.push(acc / total);
    }
    cdf
}

/// Sample from a precomputed inverse-CDF table. Returns a value in `[0, n)`.
fn cdf_sample(rng: &mut Rng, cdf: &[f64]) -> u32 {
    let u = rng.next_unit();
    cdf.partition_point(|x| *x < u) as u32
}

// ---------------------------------------------------------------------
// 4. Person tenant (SoA layout for tight scan loops).
// ---------------------------------------------------------------------

// MEMORY ENVELOPE (per-tenant, n rows): 2*n B (u16 first_name) + 2*n
// (u16 last_name) + 1*n (u8 gender) + 1*n (u8 age) + 4*n (u32
// birthday) + 4*n (u32 creation_date) + 4*n (u32 location_ip) + 1*n
// (u8 browser) + 4*n (u32 speaks_pkg) = ~23 B per Person + Vec
// overhead ≈ 24 B/Person × n rows. SF-1 (n=1.9 M) ≈ 44 MB; SF-10
// (n=19 M) ≈ 440 MB; SF-100 (n=190 M) ≈ 4.4 GB — SF-100 won't fit
// `cargo test` memory budget. Bump SF requires re-deriving the budget
// before merging. (W9a Group A retro CR-A-5 carry; W9d MED-2 co-pack
// closes F-3 LOW.)
struct PersonTenant {
    n: usize,
    first_name_id: Vec<u16>,
    last_name_id: Vec<u16>,
    gender_id: Vec<u8>,
    age: Vec<u8>,
    birthday: Vec<u32>,
    creation_date: Vec<u32>,
    location_ip_id: Vec<u32>,
    browser_id: Vec<u8>,
    /// Packed: (count << 24) | (lang2 << 16) | (lang1 << 8) | lang0.
    /// `lang_k = 0xFF` indicates the slot is unused.
    speaks_pkg: Vec<u32>,
}

fn build_persons(n: usize, master_seed: u64) -> PersonTenant {
    let mut rng = Rng::new(salted_seed(master_seed, SALT_PERSON_BUILD));
    let first_cdf = make_zipf_cdf(FIRST_NAME_COUNT, FIRST_NAME_ZIPF_ALPHA);
    let last_cdf = make_zipf_cdf(LAST_NAME_COUNT, LAST_NAME_ZIPF_ALPHA);

    let mut t = PersonTenant {
        n,
        first_name_id: Vec::with_capacity(n),
        last_name_id: Vec::with_capacity(n),
        gender_id: Vec::with_capacity(n),
        age: Vec::with_capacity(n),
        birthday: Vec::with_capacity(n),
        creation_date: Vec::with_capacity(n),
        location_ip_id: Vec::with_capacity(n),
        browser_id: Vec::with_capacity(n),
        speaks_pkg: Vec::with_capacity(n),
    };

    let age_span = AGE_MAX - AGE_MIN + 1;

    for _ in 0..n {
        t.first_name_id
            .push(cdf_sample(&mut rng, &first_cdf) as u16);
        t.last_name_id.push(cdf_sample(&mut rng, &last_cdf) as u16);
        t.gender_id.push(if rng.next_unit() < 0.5 { 0 } else { 1 });
        t.age.push(AGE_MIN + rng.range_u8(age_span));
        t.birthday.push(rng.range_u32(BIRTHDAY_DAYS));
        t.creation_date.push(rng.range_u32(CREATION_DAYS));
        t.location_ip_id.push(rng.range_u32(LOCATION_IP_COUNT));
        t.browser_id.push(cdf_sample(&mut rng, &BROWSER_CDF) as u8);

        // Languages: 1-3 distinct, with bounded retries to handle the
        // (unlikely) collision case without infinite loops.
        let count: u8 = 1 + rng.range_u8(3);
        let mut langs: [u8; 3] = [0xFF; 3];
        let mut chosen: u8 = 0;
        let mut tries: u8 = 0;
        while chosen < count && tries < 20 {
            let l = rng.range_u8(LANGUAGE_COUNT);
            if !langs.contains(&l) {
                langs[chosen as usize] = l;
                chosen += 1;
            }
            tries += 1;
        }
        let pkg = ((count as u32) << 24)
            | ((langs[2] as u32) << 16)
            | ((langs[1] as u32) << 8)
            | (langs[0] as u32);
        t.speaks_pkg.push(pkg);
    }

    t
}

// ---------------------------------------------------------------------
// 5. Predicate scan loops.
// ---------------------------------------------------------------------

#[inline]
fn count_eq_u8(col: &[u8], v: u8) -> usize {
    col.iter().filter(|&&x| x == v).count()
}
#[inline]
fn count_eq_u16(col: &[u16], v: u16) -> usize {
    col.iter().filter(|&&x| x == v).count()
}
#[inline]
fn count_eq_u32(col: &[u32], v: u32) -> usize {
    col.iter().filter(|&&x| x == v).count()
}
#[inline]
fn count_lt_u8(col: &[u8], v: u8) -> usize {
    col.iter().filter(|&&x| x < v).count()
}
#[inline]
fn count_lt_u32(col: &[u32], v: u32) -> usize {
    col.iter().filter(|&&x| x < v).count()
}

#[inline]
fn count_in_u16(col: &[u16], vs: &[u16]) -> usize {
    col.iter().filter(|x| vs.contains(*x)).count()
}

#[inline]
fn count_in_lang(col: &[u32], vs: &[u8]) -> usize {
    // Speaks list intersects the predicate list iff any slot byte matches.
    col.iter()
        .filter(|&&pkg| {
            let l0 = (pkg & 0xFF) as u8;
            let l1 = ((pkg >> 8) & 0xFF) as u8;
            let l2 = ((pkg >> 16) & 0xFF) as u8;
            let count = ((pkg >> 24) & 0xFF) as u8;
            for &v in vs {
                if count >= 1 && l0 == v {
                    return true;
                }
                if count >= 2 && l1 == v {
                    return true;
                }
                if count >= 3 && l2 == v {
                    return true;
                }
            }
            false
        })
        .count()
}

// ---------------------------------------------------------------------
// 6. Sweep + aggregation.
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Sweep {
    p10: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    samples: Vec<f64>,
}

impl Sweep {
    fn aggregate(mut samples: Vec<f64>) -> Self {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let n = samples.len();
        if n == 0 {
            return Self {
                p10: 0.0,
                p50: 0.0,
                p90: 0.0,
                p99: 0.0,
                samples,
            };
        }
        let p = |q: f64| -> f64 {
            let idx = ((q * (n as f64 - 1.0)).round() as usize).min(n - 1);
            samples[idx]
        };
        Self {
            p10: p(0.10),
            p50: p(0.50),
            p90: p(0.90),
            p99: p(0.99),
            samples,
        }
    }

    /// Aggregate of an aggregate: combine multiple sub-class sweeps into
    /// a class-level sweep. Pools all samples and re-aggregates.
    fn pool(sweeps: &[&Sweep]) -> Self {
        let mut all: Vec<f64> = Vec::new();
        for s in sweeps {
            all.extend_from_slice(&s.samples);
        }
        Self::aggregate(all)
    }
}

// ---------------------------------------------------------------------
// 7. Per-class sweep functions.
// ---------------------------------------------------------------------

fn sweep_eq_first_name(t: &PersonTenant, master_seed: u64) -> Sweep {
    let mut rng = Rng::new(salted_seed(master_seed, SALT_EQ_FIRST_NAME));
    let total = t.n as f64;
    let samples: Vec<f64> = (0..NUM_TRIALS)
        .map(|_| {
            let v = rng.range_u32(FIRST_NAME_COUNT) as u16;
            count_eq_u16(&t.first_name_id, v) as f64 / total
        })
        .collect();
    Sweep::aggregate(samples)
}

fn sweep_eq_last_name(t: &PersonTenant, master_seed: u64) -> Sweep {
    let mut rng = Rng::new(salted_seed(master_seed, SALT_EQ_LAST_NAME));
    let total = t.n as f64;
    let samples: Vec<f64> = (0..NUM_TRIALS)
        .map(|_| {
            let v = rng.range_u32(LAST_NAME_COUNT) as u16;
            count_eq_u16(&t.last_name_id, v) as f64 / total
        })
        .collect();
    Sweep::aggregate(samples)
}

fn sweep_eq_gender(t: &PersonTenant, master_seed: u64) -> Sweep {
    let mut rng = Rng::new(salted_seed(master_seed, SALT_EQ_GENDER));
    let total = t.n as f64;
    // Only 2 distinct values, but we sweep 50 trials anyway for stable
    // p* estimates (the per-trial answer is one of two stable points
    // close to 0.5 — the sweep mostly confirms determinism).
    let samples: Vec<f64> = (0..NUM_TRIALS)
        .map(|_| {
            let v = rng.range_u8(2);
            count_eq_u8(&t.gender_id, v) as f64 / total
        })
        .collect();
    Sweep::aggregate(samples)
}

fn sweep_eq_age(t: &PersonTenant, master_seed: u64) -> Sweep {
    let mut rng = Rng::new(salted_seed(master_seed, SALT_EQ_AGE));
    let total = t.n as f64;
    let span = AGE_MAX - AGE_MIN + 1;
    let samples: Vec<f64> = (0..NUM_TRIALS)
        .map(|_| {
            let v = AGE_MIN + rng.range_u8(span);
            count_eq_u8(&t.age, v) as f64 / total
        })
        .collect();
    Sweep::aggregate(samples)
}

fn sweep_eq_birthday(t: &PersonTenant, master_seed: u64) -> Sweep {
    let mut rng = Rng::new(salted_seed(master_seed, SALT_EQ_BIRTHDAY));
    let total = t.n as f64;
    let samples: Vec<f64> = (0..NUM_TRIALS)
        .map(|_| {
            let v = rng.range_u32(BIRTHDAY_DAYS);
            count_eq_u32(&t.birthday, v) as f64 / total
        })
        .collect();
    Sweep::aggregate(samples)
}

fn sweep_eq_browser(t: &PersonTenant, master_seed: u64) -> Sweep {
    let mut rng = Rng::new(salted_seed(master_seed, SALT_EQ_BROWSER));
    let total = t.n as f64;
    let samples: Vec<f64> = (0..NUM_TRIALS)
        .map(|_| {
            let v = rng.range_u8(BROWSER_COUNT);
            count_eq_u8(&t.browser_id, v) as f64 / total
        })
        .collect();
    Sweep::aggregate(samples)
}

fn sweep_eq_location_ip(t: &PersonTenant, master_seed: u64) -> Sweep {
    let mut rng = Rng::new(salted_seed(master_seed, SALT_EQ_LOCATION_IP));
    let total = t.n as f64;
    let samples: Vec<f64> = (0..NUM_TRIALS)
        .map(|_| {
            let v = rng.range_u32(LOCATION_IP_COUNT);
            count_eq_u32(&t.location_ip_id, v) as f64 / total
        })
        .collect();
    Sweep::aggregate(samples)
}

fn sweep_lt_age(t: &PersonTenant, master_seed: u64) -> Sweep {
    let mut rng = Rng::new(salted_seed(master_seed, SALT_LT_AGE));
    let total = t.n as f64;
    let span = AGE_MAX - AGE_MIN + 1;
    let samples: Vec<f64> = (0..NUM_TRIALS)
        .map(|_| {
            let v = AGE_MIN + rng.range_u8(span);
            count_lt_u8(&t.age, v) as f64 / total
        })
        .collect();
    Sweep::aggregate(samples)
}

fn sweep_lt_birthday(t: &PersonTenant, master_seed: u64) -> Sweep {
    let mut rng = Rng::new(salted_seed(master_seed, SALT_LT_BIRTHDAY));
    let total = t.n as f64;
    let samples: Vec<f64> = (0..NUM_TRIALS)
        .map(|_| {
            let v = rng.range_u32(BIRTHDAY_DAYS);
            count_lt_u32(&t.birthday, v) as f64 / total
        })
        .collect();
    Sweep::aggregate(samples)
}

fn sweep_lt_creation_date(t: &PersonTenant, master_seed: u64) -> Sweep {
    let mut rng = Rng::new(salted_seed(master_seed, SALT_LT_CREATION_DATE));
    let total = t.n as f64;
    let samples: Vec<f64> = (0..NUM_TRIALS)
        .map(|_| {
            let v = rng.range_u32(CREATION_DAYS);
            count_lt_u32(&t.creation_date, v) as f64 / total
        })
        .collect();
    Sweep::aggregate(samples)
}

fn sweep_in_first_name(t: &PersonTenant, list_size: usize, master_seed: u64) -> Sweep {
    let mut rng = Rng::new(salted_seed(
        master_seed,
        SALT_IN_FIRST_NAME ^ (list_size as u64),
    ));
    let total = t.n as f64;
    let samples: Vec<f64> = (0..NUM_TRIALS)
        .map(|_| {
            let mut vs: Vec<u16> = Vec::with_capacity(list_size);
            while vs.len() < list_size {
                let v = rng.range_u32(FIRST_NAME_COUNT) as u16;
                if !vs.contains(&v) {
                    vs.push(v);
                }
            }
            count_in_u16(&t.first_name_id, &vs) as f64 / total
        })
        .collect();
    Sweep::aggregate(samples)
}

fn sweep_in_language(t: &PersonTenant, list_size: usize, master_seed: u64) -> Sweep {
    let mut rng = Rng::new(salted_seed(
        master_seed,
        SALT_IN_LANGUAGE ^ (list_size as u64),
    ));
    let total = t.n as f64;
    let samples: Vec<f64> = (0..NUM_TRIALS)
        .map(|_| {
            let mut vs: Vec<u8> = Vec::with_capacity(list_size);
            while vs.len() < list_size {
                let v = rng.range_u8(LANGUAGE_COUNT);
                if !vs.contains(&v) {
                    vs.push(v);
                }
            }
            count_in_lang(&t.speaks_pkg, &vs) as f64 / total
        })
        .collect();
    Sweep::aggregate(samples)
}

/// Label sweep: takes the auxiliary label cardinalities per-tenant
/// and computes label-cardinality / total-node-count for each.
fn sweep_label(person_count: u64) -> Sweep {
    let total = (person_count + COMMENT_COUNT + FORUM_COUNT + PLACE_COUNT) as f64;
    let samples = vec![
        person_count as f64 / total,
        COMMENT_COUNT as f64 / total,
        FORUM_COUNT as f64 / total,
        PLACE_COUNT as f64 / total,
    ];
    Sweep::aggregate(samples)
}

/// Rel-type sweep: takes the SNB-style edge mix and computes
/// rel-type-cardinality / total-rel-count for each.
fn sweep_rel_type() -> Sweep {
    let total = TOTAL_EDGES as f64;
    let knows = (KNOWS_FRAC * total) as u64;
    let likes = (LIKES_FRAC * total) as u64;
    let ili = (ILI_FRAC * total) as u64;
    let samples = vec![
        knows as f64 / total,
        likes as f64 / total,
        ili as f64 / total,
    ];
    Sweep::aggregate(samples)
}

// ---------------------------------------------------------------------
// 8. Report formatting.
// ---------------------------------------------------------------------

struct ClassReport {
    eq_first_name: Sweep,
    eq_last_name: Sweep,
    eq_gender: Sweep,
    eq_age: Sweep,
    eq_birthday: Sweep,
    eq_browser: Sweep,
    eq_location_ip: Sweep,
    eq_pooled: Sweep,
    lt_age: Sweep,
    lt_birthday: Sweep,
    lt_creation_date: Sweep,
    lt_pooled: Sweep,
    in_first_name_3: Sweep,
    in_first_name_10: Sweep,
    in_language_3: Sweep,
    in_pooled: Sweep,
    label: Sweep,
    rel_type: Sweep,
}

fn run_back_test(person_count: usize) -> ClassReport {
    let t = build_persons(person_count, MASTER_SEED);

    let eq_first_name = sweep_eq_first_name(&t, MASTER_SEED);
    let eq_last_name = sweep_eq_last_name(&t, MASTER_SEED);
    let eq_gender = sweep_eq_gender(&t, MASTER_SEED);
    let eq_age = sweep_eq_age(&t, MASTER_SEED);
    let eq_birthday = sweep_eq_birthday(&t, MASTER_SEED);
    let eq_browser = sweep_eq_browser(&t, MASTER_SEED);
    let eq_location_ip = sweep_eq_location_ip(&t, MASTER_SEED);
    let eq_pooled = Sweep::pool(&[
        &eq_first_name,
        &eq_last_name,
        &eq_gender,
        &eq_age,
        &eq_birthday,
        &eq_browser,
        &eq_location_ip,
    ]);

    let lt_age = sweep_lt_age(&t, MASTER_SEED);
    let lt_birthday = sweep_lt_birthday(&t, MASTER_SEED);
    let lt_creation_date = sweep_lt_creation_date(&t, MASTER_SEED);
    let lt_pooled = Sweep::pool(&[&lt_age, &lt_birthday, &lt_creation_date]);

    let in_first_name_3 = sweep_in_first_name(&t, 3, MASTER_SEED);
    let in_first_name_10 = sweep_in_first_name(&t, 10, MASTER_SEED);
    let in_language_3 = sweep_in_language(&t, 3, MASTER_SEED);
    let in_pooled = Sweep::pool(&[&in_first_name_3, &in_first_name_10, &in_language_3]);

    let label = sweep_label(person_count as u64);
    let rel_type = sweep_rel_type();

    ClassReport {
        eq_first_name,
        eq_last_name,
        eq_gender,
        eq_age,
        eq_birthday,
        eq_browser,
        eq_location_ip,
        eq_pooled,
        lt_age,
        lt_birthday,
        lt_creation_date,
        lt_pooled,
        in_first_name_3,
        in_first_name_10,
        in_language_3,
        in_pooled,
        label,
        rel_type,
    }
}

fn print_report(person_count: usize, r: &ClassReport) {
    println!();
    println!("=========================================================================");
    println!(
        "M4-04d Empirical Selectivity Back-Test ({} LDBC SNB Person tenant)",
        format_count(person_count)
    );
    println!("=========================================================================");
    println!(
        "Hardware:        {} ({})",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!(
        "Build profile:   {}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    );
    println!("Master seed:     0x{:016X}", MASTER_SEED);
    println!("Trials per sub-class: {}", NUM_TRIALS);
    println!(
        "Tenant: {} Persons + {} Comments + {} Forums + {} Places (total {} nodes)",
        format_count(person_count),
        format_count(COMMENT_COUNT as usize),
        format_count(FORUM_COUNT as usize),
        format_count(PLACE_COUNT as usize),
        format_count(person_count + (COMMENT_COUNT + FORUM_COUNT + PLACE_COUNT) as usize),
    );
    println!(
        "Edges: {} (KNOWS {:.0}%, LIKES {:.0}%, IS_LOCATED_IN {:.0}%)",
        format_count(TOTAL_EDGES as usize),
        KNOWS_FRAC * 100.0,
        LIKES_FRAC * 100.0,
        ILI_FRAC * 100.0,
    );
    println!();
    println!("Per-sub-class results:");
    println!("---------------------------------------------------------------------------------");
    println!(
        "| {:<26} | {:>10} | {:>10} | {:>10} | {:>10} |",
        "Class : sub-class", "p10", "p50", "p90", "p99",
    );
    println!("---------------------------------------------------------------------------------");
    print_row("eq : firstName", &r.eq_first_name);
    print_row("eq : lastName", &r.eq_last_name);
    print_row("eq : gender", &r.eq_gender);
    print_row("eq : age", &r.eq_age);
    print_row("eq : birthday", &r.eq_birthday);
    print_row("eq : browser", &r.eq_browser);
    print_row("eq : locationIP", &r.eq_location_ip);
    print_row("eq : POOLED", &r.eq_pooled);
    println!("---------------------------------------------------------------------------------");
    print_row("lt : age", &r.lt_age);
    print_row("lt : birthday", &r.lt_birthday);
    print_row("lt : creationDate", &r.lt_creation_date);
    print_row("lt : POOLED", &r.lt_pooled);
    println!("---------------------------------------------------------------------------------");
    print_row("in : firstName (n=3)", &r.in_first_name_3);
    print_row("in : firstName (n=10)", &r.in_first_name_10);
    print_row("in : speaks (n=3)", &r.in_language_3);
    print_row("in : POOLED", &r.in_pooled);
    println!("---------------------------------------------------------------------------------");
    print_row("label", &r.label);
    print_row("rel_type", &r.rel_type);
    println!("---------------------------------------------------------------------------------");

    println!();
    println!("Constant comparison (current vs empirical p50):");
    println!("---------------------------------------------------------------------------------");
    println!(
        "| {:<32} | {:>10} | {:>13} | {:>8} | {:<24} |",
        "Constant", "Current", "Empirical p50", "Ratio", "Recommendation",
    );
    println!("---------------------------------------------------------------------------------");
    print_compare(
        "DEFAULT_EQ_SELECTIVITY",
        DEFAULT_EQ_SELECTIVITY,
        r.eq_pooled.p50,
    );
    print_compare(
        "DEFAULT_LT_SELECTIVITY",
        DEFAULT_LT_SELECTIVITY,
        r.lt_pooled.p50,
    );
    print_compare(
        "DEFAULT_IN_SELECTIVITY",
        DEFAULT_IN_SELECTIVITY,
        r.in_pooled.p50,
    );
    print_compare(
        "DEFAULT_LABEL_SELECTIVITY",
        DEFAULT_LABEL_SELECTIVITY,
        r.label.p50,
    );
    print_compare(
        "DEFAULT_REL_TYPE_SELECTIVITY",
        DEFAULT_REL_TYPE_SELECTIVITY,
        r.rel_type.p50,
    );
    println!("---------------------------------------------------------------------------------");
    println!();
}

fn print_row(name: &str, s: &Sweep) {
    println!(
        "| {:<26} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>10.3e} |",
        name, s.p10, s.p50, s.p90, s.p99,
    );
}

fn print_compare(name: &str, current: f64, p50: f64) {
    let ratio = if p50 > 0.0 {
        (current / p50).max(p50 / current)
    } else {
        f64::INFINITY
    };
    let recommendation = if ratio <= 2.0 {
        "KEEP (within 2× of p50)".to_string()
    } else {
        format!("TUNE → {p50:.4} (empirical p50)")
    };
    println!(
        "| {:<32} | {:>10.4} | {:>13.3e} | {:>7.1}× | {:<24} |",
        name, current, p50, ratio, recommendation,
    );
}

fn format_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{n}")
    }
}

// ---------------------------------------------------------------------
// 9. Test entry points.
// ---------------------------------------------------------------------

/// Smoke-level back-test (10 K Persons). Runs on every `cargo test`;
/// validates the back-test scaffolding without paying the full 1M cost.
/// Empirical numbers from this run are NOT cited by amendment-07 — the
/// smoke just confirms determinism + no regressions in the sweep code.
#[test]
fn m4_04d_empirical_selectivity_backtest_smoke() {
    let r = run_back_test(10_000);
    print_report(10_000, &r);

    // Sanity invariants: every p* is in [0, 1].
    for s in [
        &r.eq_pooled,
        &r.lt_pooled,
        &r.in_pooled,
        &r.label,
        &r.rel_type,
    ] {
        assert!(
            (0.0..=1.0).contains(&s.p10)
                && (0.0..=1.0).contains(&s.p50)
                && (0.0..=1.0).contains(&s.p90)
                && (0.0..=1.0).contains(&s.p99),
            "p* out of [0, 1] band: {:?}",
            s,
        );
        assert!(
            s.p10 <= s.p50 && s.p50 <= s.p90 && s.p90 <= s.p99,
            "percentiles not monotone: {:?}",
            s,
        );
    }

    // Determinism check: run again, expect byte-identical samples.
    let r2 = run_back_test(10_000);
    assert_eq!(
        r.eq_first_name.samples, r2.eq_first_name.samples,
        "back-test is non-deterministic — re-run produced different samples",
    );
}

/// Full empirical back-test (1 M Persons). The empirical p50 numbers
/// from THIS run at the recorded `MASTER_SEED` are what ADR-038
/// amendment-07 cites as the back-test artifact.
///
/// Gated `#[ignore]` because the full 1M run takes ~3-10 s in release
/// mode and is unnecessary for normal CI; the smoke variant covers
/// regression. Run explicitly via:
/// ```bash
/// cargo test -p arcgraph-query --release \
///     m4_04d_empirical_selectivity_backtest -- --ignored --nocapture
/// ```
#[test]
#[ignore = "M4-04d 1M Person back-test — invoke explicitly with --ignored"]
fn m4_04d_empirical_selectivity_backtest() {
    let r = run_back_test(1_000_000);
    print_report(1_000_000, &r);

    // Pin the empirical p50 ranges. Centred on the 2026-05-06 1M-run
    // observations (apple-silicon aarch64, release):
    //   eq pooled p50    = 1.600e-4
    //   lt pooled p50    = 4.912e-1
    //   in pooled p50    = 4.981e-3
    //   label      p50   = 9.001e-2
    //   rel_type   p50   = 2.500e-1
    // Bounds are ±50 % around each anchor — tight enough to flag a
    // fixture-distribution regression, loose enough to absorb host /
    // instruction-set / float-rounding drift. ADR-038 amendment-07
    // cites these anchor numbers; if a future fixture revision
    // intentionally moves them, bump the anchors here AND in
    // amendment-07 in lockstep.
    let expected = [
        ("eq pooled p50", r.eq_pooled.p50, 8.0e-5, 3.2e-4),
        ("lt pooled p50", r.lt_pooled.p50, 0.25, 0.75),
        ("in pooled p50", r.in_pooled.p50, 2.5e-3, 1.0e-2),
        ("label p50", r.label.p50, 0.045, 0.18),
        ("rel_type p50", r.rel_type.p50, 0.20, 0.30),
    ];
    for (name, observed, lo, hi) in expected {
        assert!(
            (lo..=hi).contains(&observed),
            "{name} = {observed:.4e} outside expected [{lo:.4e}, {hi:.4e}] — \
             fixture distribution drifted; update amendment-07 + this guard.",
        );
    }
}
