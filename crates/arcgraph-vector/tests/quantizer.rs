//! Slice E.1 quantizer integration tests.
//!
//! These tests live at the integration layer rather than alongside
//! the codec modules because they exercise the **end-to-end
//! train → encode → decode** contract specified by ADR-035 §3.3
//! and AC-1a — they're the spec-level acceptance gates, not
//! per-function unit tests (those live in
//! `crates/arcgraph-vector/src/quantizer/{sq8,binary,dispatch}.rs`).
//!
//! Per the M3.a execution plan §3.1 (Slice E.1):
//! - **`sq8_round_trip_error_bounded`**: train on 10 K samples;
//!   per-dim absolute error ≤ 1 % of the per-dim training range
//!   (per ADR-035 §3.3 SQ8 spec — the 1-bin tolerance is
//!   `(max - min) / 255 ≈ 0.39 %` of the range, well under 1 %).
//! - **`sq8_constant_dim_handled_gracefully`**: a constant
//!   dimension does NOT crash; codebook valid; warn emitted (per
//!   ADR-035 §9.3).
//! - **`binary_encode_decode_round_trip_sign_only`**: every
//!   decoded ±1 element matches the sign of the input.
//! - **`auto_quantizer_threshold_at_10m`**: the boundary at
//!   `N == 10_000_000` flips from `None` to `Sq8` (per ADR-035
//!   D-4 + Q3 ratification).

use arcgraph_vector::Encoding;
use arcgraph_vector::quantizer::{
    Sq8Trainer, auto_quantizer_for_collection, binary_decode, binary_encode, binary_encode_aligned,
};

/// Deterministic xorshift64 PRNG. Avoids the `rand` dev-dep —
/// integration tests for codecs care about reproducibility, not
/// distributional realism. The sample distributions used here
/// are uniform on `[-1, 1]` (the canonical normalized embedding
/// envelope; per ADR-035 §3.3 SQ8 measurements use this domain).
struct Xs64(u64);

impl Xs64 {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        let unit = (bits as f32) / ((1u32 << 24) as f32);
        lo + (hi - lo) * unit
    }
}

/// 10 K samples × 768 dims, per-dim error ≤ 1 % absolute on
/// uniform `[-1, 1]` inputs.
///
/// Per ADR-035 §3.3, SQ8 training fits `scale = (max - min) / 255`
/// — for uniform `[-1, 1]` samples this is ~`2/255 ≈ 0.00784`,
/// or ~0.39 % of the [-1, 1] range. The 1-bin round-trip error
/// is at most `scale / 2 ≈ 0.196 %` per dim, which is well below
/// the 1 % bound the slice spec asks for.
///
/// We intentionally probe a **fresh** sample (not one from the
/// training set) to confirm that the codebook generalizes across
/// the [-1, 1] envelope rather than only round-tripping the
/// observed samples.
#[test]
fn sq8_round_trip_error_bounded() {
    const DIM: usize = 768;
    const N: usize = 10_000;
    const RANGE_LO: f32 = -1.0;
    const RANGE_HI: f32 = 1.0;
    const ONE_PCT_OF_RANGE: f32 = 0.01 * (RANGE_HI - RANGE_LO);

    let mut rng = Xs64::new(0xCAFE_F00D_DEAD_BEEF);

    // Training samples.
    let storage: Vec<Vec<f32>> = (0..N)
        .map(|_| {
            (0..DIM)
                .map(|_| rng.next_uniform(RANGE_LO, RANGE_HI))
                .collect()
        })
        .collect();
    let samples: Vec<&[f32]> = storage.iter().map(Vec::as_slice).collect();

    let cb = Sq8Trainer.train(&samples).expect("training succeeds");
    assert_eq!(cb.dim(), DIM);

    // Probe: a fresh vector independent of the training set.
    let probe: Vec<f32> = (0..DIM)
        .map(|_| rng.next_uniform(RANGE_LO, RANGE_HI))
        .collect();
    // Per #116 closure, encode emits i8 directly (kernel-native).
    let q: Vec<i8> = cb.encode(&probe).expect("encode succeeds");
    assert_eq!(q.len(), DIM);
    let decoded = cb.decode(&q).expect("decode succeeds");
    assert_eq!(decoded.len(), DIM);

    // Per-dim absolute error must be ≤ 1 % of the input envelope
    // (= 0.02 for [-1, 1]). The actual bound is ~0.4 % of the
    // range from the ~0.39 % per-bin scale; the headroom matters
    // because the trainer's observed `(min, max)` per dim may be
    // tighter than the [-1, 1] envelope on a 10K sample.
    let mut max_err = 0.0_f32;
    for d in 0..DIM {
        let err = (probe[d] - decoded[d]).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(
            err <= ONE_PCT_OF_RANGE,
            "dim {d}: probe={} decoded={} err={} > 1% bound {}",
            probe[d],
            decoded[d],
            err,
            ONE_PCT_OF_RANGE
        );
    }
    // Tighten the upper bound: per-bin scale on a 10K uniform
    // sample is `(observed_max - observed_min) / 255`, expected
    // ≈ 2/255. The max round-trip error is half a bin (≈ 0.39 %
    // of the [-1, 1] range), well under the 1 % bound.
    let half_bin_upper = (RANGE_HI - RANGE_LO) / 2.0 / 255.0; // 0.00392
    let two_bin_upper = 2.0 * half_bin_upper; // 0.00784
    // Allow up to 2× the half-bin theoretical max to absorb the
    // (max - min) tightening (samples don't span the full
    // [-1, 1] envelope in 10K draws). This still validates the
    // 1 %-of-range target with margin.
    assert!(
        max_err <= two_bin_upper,
        "max_err={max_err} > 2-bin theoretical {two_bin_upper}"
    );
}

/// A single constant dimension does not panic, does not error;
/// the codebook is valid; non-constant dims still round-trip
/// faithfully.
///
/// Per ADR-035 §9.3, the trainer collapses the constant dim to
/// `(scale=1.0, bias=value)` and emits a `tracing::warn!` (we
/// don't assert on the warn — it requires a tracing subscriber
/// fixture; the unit tests in `quantizer/sq8.rs` cover the warn
/// path's *effect* on the codebook).
#[test]
fn sq8_constant_dim_handled_gracefully() {
    const DIM: usize = 8;
    const CONST_DIM: usize = 3; // dim index 3 is constant

    let mut rng = Xs64::new(31337);
    // 100 samples; dim 3 is always 0.42; other dims uniform in
    // [-0.5, 0.5].
    let storage: Vec<Vec<f32>> = (0..100)
        .map(|_| {
            (0..DIM)
                .map(|d| {
                    if d == CONST_DIM {
                        0.42
                    } else {
                        rng.next_uniform(-0.5, 0.5)
                    }
                })
                .collect()
        })
        .collect();
    let samples: Vec<&[f32]> = storage.iter().map(Vec::as_slice).collect();

    let cb = Sq8Trainer
        .train(&samples)
        .expect("constant dim must not fail");
    let p = cb.params();
    // Constant-dim sentinel: scale=1.0, bias=0.42.
    assert_eq!(p.scale[CONST_DIM], 1.0);
    assert!((p.bias[CONST_DIM] - 0.42).abs() < 1e-7);

    // Round-trip works for non-constant dims (the constant dim's
    // round-trip is exact at the sentinel value, off-spec for
    // other inputs but documented in §9.3).
    let probe: Vec<f32> = (0..DIM)
        .map(|d| if d == CONST_DIM { 0.42 } else { 0.1 })
        .collect();
    let q = cb.encode(&probe).expect("encode");
    let dec = cb.decode(&q).expect("decode");

    // Dim CONST_DIM round-trips exactly to the sentinel value.
    assert!(
        (dec[CONST_DIM] - 0.42).abs() < 1e-6,
        "constant dim drift: {} vs 0.42",
        dec[CONST_DIM]
    );

    // Non-constant dims round-trip within the 1 %-of-range
    // tolerance.
    for d in 0..DIM {
        if d == CONST_DIM {
            continue;
        }
        let err = (probe[d] - dec[d]).abs();
        // Per-bin scale on [-0.5, 0.5] sample is ~1/255 ≈ 0.0039.
        assert!(
            err <= 0.01,
            "dim {d}: probe={} decoded={} err={}",
            probe[d],
            dec[d],
            err
        );
    }
}

/// Sign function round-trip: encode → decode → check sign of every
/// element matches the input. Per ADR-035 §3.3 + the §9.3
/// sign(0) convention (`bit_d = (x > 0)`), `0.0` and `-0.0` both
/// decode to `-1.0`.
#[test]
fn binary_encode_decode_round_trip_sign_only() {
    const DIM: usize = 768;
    let mut rng = Xs64::new(0xBEEF_FACE_BABE_F00D);
    // Mix uniform [-1, 1] with a few exact zeros and exact 1.0s
    // / -1.0s to exercise the boundary.
    let mut input: Vec<f32> = (0..DIM).map(|_| rng.next_uniform(-1.0, 1.0)).collect();
    input[0] = 0.0;
    input[1] = -0.0;
    input[2] = 1.0;
    input[3] = -1.0;
    input[4] = f32::EPSILON;
    input[5] = -f32::EPSILON;

    let packed = binary_encode(&input);
    assert_eq!(packed.len(), DIM.div_ceil(8));
    assert_eq!(packed.len(), 96, "dim=768 packs to 96 bytes");
    let aligned = binary_encode_aligned(&input);
    assert_eq!(aligned.len(), 128, "dim=768 aligns to 128 (per S-1)");
    assert_eq!(
        aligned.len(),
        Encoding::Binary.bytes_per_vector_aligned(DIM)
    );
    assert_eq!(&aligned[..96], &packed[..]);
    assert!(
        aligned[96..].iter().all(|&b| b == 0),
        "aligned padding must be zero"
    );

    let decoded_packed = binary_decode(&packed, DIM);
    let decoded_aligned = binary_decode(&aligned, DIM);
    assert_eq!(decoded_packed, decoded_aligned);

    for i in 0..DIM {
        let expected = if input[i] > 0.0 { 1.0 } else { -1.0 };
        assert_eq!(
            decoded_packed[i], expected,
            "i={i} input={} decoded={} expected_sign={}",
            input[i], decoded_packed[i], expected
        );
    }
    // sign(0) convention: 0.0 and -0.0 both → -1.0 (bit clear).
    assert_eq!(decoded_packed[0], -1.0);
    assert_eq!(decoded_packed[1], -1.0);
    // 1.0 and -1.0 → ±1.0 directly.
    assert_eq!(decoded_packed[2], 1.0);
    assert_eq!(decoded_packed[3], -1.0);
    // EPSILON: tiny positive → +1.0; tiny negative → -1.0.
    assert_eq!(decoded_packed[4], 1.0);
    assert_eq!(decoded_packed[5], -1.0);
}

/// The 10 M auto-quantize threshold per ADR-035 D-4 + Q3
/// ratification: `N == 9_999_999` → `None`; `N == 10_000_000` →
/// `Some(Encoding::Sq8)`.
#[test]
fn auto_quantizer_threshold_at_10m() {
    assert_eq!(auto_quantizer_for_collection(9_999_999), None);
    assert_eq!(
        auto_quantizer_for_collection(10_000_000),
        Some(Encoding::Sq8)
    );
}
