//! Reference-value tests for every Slice B distance kernel.
//!
//! Each test pins the kernel against a known, hand-computable
//! result. The point is correctness — the simsimd uplift is
//! validated by the Criterion bench (`benches/distance.rs`).

use arcgraph_vector::{
    DistanceKernel,
    distance::{
        CosineF16, CosineF32, CosineSq8, HammingBinary, IpF16, IpF32, IpSq8, L2F16, L2F32, L2Sq8,
    },
};

/// Loose tolerance for the SIMD-vs-scalar approximation drift.
const EPS_F32: f32 = 1e-4;
/// F16 has ~3 decimal digits of precision; loosen accordingly.
const EPS_F16: f32 = 1e-2;
/// SQ8 / i8 cosine drift can be a few percent on small vectors.
const EPS_SQ8: f32 = 5e-2;

#[inline]
fn approx(actual: f32, expected: f32, eps: f32) -> bool {
    (actual - expected).abs() <= eps
}

// ─── F32 ─────────────────────────────────────────────────────────

#[test]
fn l2_f32_orthogonal_unit_vectors_have_distance_two() {
    // ‖ê₁ − ê₂‖² = 1² + 1² = 2
    let a: [f32; 3] = [1.0, 0.0, 0.0];
    let b: [f32; 3] = [0.0, 1.0, 0.0];
    let d = L2F32.distance(bytemuck::cast_slice(&a), bytemuck::cast_slice(&b));
    assert!(approx(d, 2.0, EPS_F32), "expected 2.0, got {d}");
}

#[test]
fn l2_f32_3_4_5_pythag_squared() {
    // ‖(3, 4)‖² = 9 + 16 = 25
    let a: [f32; 2] = [3.0, 4.0];
    let b: [f32; 2] = [0.0, 0.0];
    let d = L2F32.distance(bytemuck::cast_slice(&a), bytemuck::cast_slice(&b));
    assert!(approx(d, 25.0, EPS_F32), "expected 25.0, got {d}");
}

#[test]
fn ip_f32_textbook_example() {
    // (1, 2, 3) · (4, 5, 6) = 4 + 10 + 18 = 32
    let a: [f32; 3] = [1.0, 2.0, 3.0];
    let b: [f32; 3] = [4.0, 5.0, 6.0];
    let d = IpF32.distance(bytemuck::cast_slice(&a), bytemuck::cast_slice(&b));
    assert!(approx(d, 32.0, EPS_F32), "expected 32.0, got {d}");
}

#[test]
fn cosine_f32_same_vector_is_zero_distance() {
    // cos(0) = 1 ⇒ cosine distance = 1 - 1 = 0
    let a: [f32; 3] = [1.0, 2.0, 3.0];
    let b: [f32; 3] = [1.0, 2.0, 3.0];
    let d = CosineF32.distance(bytemuck::cast_slice(&a), bytemuck::cast_slice(&b));
    assert!(approx(d, 0.0, EPS_F32), "expected 0.0, got {d}");
}

#[test]
fn cosine_f32_orthogonal_is_one_distance() {
    // cos(π/2) = 0 ⇒ cosine distance = 1 - 0 = 1
    let a: [f32; 3] = [1.0, 0.0, 0.0];
    let b: [f32; 3] = [0.0, 1.0, 0.0];
    let d = CosineF32.distance(bytemuck::cast_slice(&a), bytemuck::cast_slice(&b));
    assert!(approx(d, 1.0, EPS_F32), "expected 1.0, got {d}");
}

#[test]
fn cosine_f32_antiparallel_is_two_distance() {
    // cos(π) = -1 ⇒ cosine distance = 1 - (-1) = 2
    let a: [f32; 3] = [1.0, 0.0, 0.0];
    let b: [f32; 3] = [-1.0, 0.0, 0.0];
    let d = CosineF32.distance(bytemuck::cast_slice(&a), bytemuck::cast_slice(&b));
    assert!(approx(d, 2.0, EPS_F32), "expected 2.0, got {d}");
}

// ─── F16 ─────────────────────────────────────────────────────────

/// Construct an `[f16; N]` byte view via the upstream `from_f32`
/// constructor. Stored on the heap (Vec) so the byte slice is
/// 2-byte-aligned by the Rust allocator.
fn f16_bytes_from_f32(values: &[f32]) -> Vec<u8> {
    let halves: Vec<simsimd::f16> = values.iter().map(|v| simsimd::f16::from_f32(*v)).collect();
    // SAFETY: simsimd::f16 has size = 2 bytes (compile-time
    // assertion in src/distance.rs) and reading those bytes
    // from the owned Vec is sound — `simsimd::f16` is an
    // upstream half-precision wrapper with no padding bytes.
    let view: &[u8] = unsafe {
        std::slice::from_raw_parts(
            halves.as_ptr().cast::<u8>(),
            halves.len() * std::mem::size_of::<simsimd::f16>(),
        )
    };
    view.to_vec()
}

#[test]
fn l2_f16_orthogonal_unit_vectors_have_distance_two() {
    let a = f16_bytes_from_f32(&[1.0, 0.0, 0.0]);
    let b = f16_bytes_from_f32(&[0.0, 1.0, 0.0]);
    let d = L2F16.distance(&a, &b);
    assert!(approx(d, 2.0, EPS_F16), "expected ~2.0, got {d}");
}

#[test]
fn ip_f16_textbook_example() {
    let a = f16_bytes_from_f32(&[1.0, 2.0, 3.0]);
    let b = f16_bytes_from_f32(&[4.0, 5.0, 6.0]);
    let d = IpF16.distance(&a, &b);
    assert!(approx(d, 32.0, EPS_F16), "expected ~32.0, got {d}");
}

#[test]
fn cosine_f16_same_vector_is_zero_distance() {
    let a = f16_bytes_from_f32(&[1.0, 2.0, 3.0]);
    let b = f16_bytes_from_f32(&[1.0, 2.0, 3.0]);
    let d = CosineF16.distance(&a, &b);
    assert!(approx(d, 0.0, EPS_F16), "expected ~0.0, got {d}");
}

// ─── SQ8 (i8) ────────────────────────────────────────────────────
//
// SQ8 vectors are stored as i8 per dimension. The byte cast goes
// from `&[u8]` via `bytemuck::cast_slice` into `&[i8]`; the byte
// slice we construct holds the canonical two's-complement
// representation of each i8, so casting via `i8::cast_from`
// (or here, casting the `i8` array's bytes) yields the same view.

fn sq8_bytes(values: &[i8]) -> Vec<u8> {
    bytemuck::cast_slice(values).to_vec()
}

#[test]
fn l2_sq8_orthogonal_vectors() {
    // (10, 0, 0) vs (0, 10, 0) → 100 + 100 = 200
    let a = sq8_bytes(&[10, 0, 0]);
    let b = sq8_bytes(&[0, 10, 0]);
    let d = L2Sq8.distance(&a, &b);
    assert!(approx(d, 200.0, 1.0), "expected ~200.0, got {d}");
}

#[test]
fn ip_sq8_textbook_example() {
    // (1, 2, 3) · (4, 5, 6) = 32
    let a = sq8_bytes(&[1, 2, 3]);
    let b = sq8_bytes(&[4, 5, 6]);
    let d = IpSq8.distance(&a, &b);
    assert!(approx(d, 32.0, 1.0), "expected ~32.0, got {d}");
}

#[test]
fn cosine_sq8_same_vector_is_zero_distance() {
    let a = sq8_bytes(&[10, 20, 30]);
    let b = sq8_bytes(&[10, 20, 30]);
    let d = CosineSq8.distance(&a, &b);
    assert!(approx(d, 0.0, EPS_SQ8), "expected ~0.0, got {d}");
}

// ─── Binary / Hamming ────────────────────────────────────────────

#[test]
fn hamming_all_bits_flipped_in_one_byte() {
    // 0xFF XOR 0x00 = 0xFF → popcount = 8.
    let a = vec![0xFFu8];
    let b = vec![0x00u8];
    let d = HammingBinary.distance(&a, &b);
    assert!(approx(d, 8.0, EPS_F32), "expected 8.0, got {d}");
}

#[test]
fn hamming_no_difference_is_zero() {
    let a = vec![0xAAu8, 0x55, 0xF0, 0x0F];
    let b = vec![0xAAu8, 0x55, 0xF0, 0x0F];
    let d = HammingBinary.distance(&a, &b);
    assert!(approx(d, 0.0, EPS_F32), "expected 0.0, got {d}");
}

#[test]
fn hamming_complement_is_full_bit_count() {
    // 4 bytes, all complemented → 32 bits differ.
    let a = vec![0xFFu8, 0xFF, 0xFF, 0xFF];
    let b = vec![0x00u8, 0x00, 0x00, 0x00];
    let d = HammingBinary.distance(&a, &b);
    assert!(approx(d, 32.0, EPS_F32), "expected 32.0, got {d}");
}

#[test]
fn hamming_alternating_pattern_one_byte_at_dim_8() {
    // 0xAA = 10101010, 0x55 = 01010101 → all 8 bits differ.
    let a = vec![0xAAu8];
    let b = vec![0x55u8];
    let d = HammingBinary.distance(&a, &b);
    assert!(approx(d, 8.0, EPS_F32), "expected 8.0, got {d}");
}

// ─── kernel tagging surface ──────────────────────────────────────

#[test]
fn every_kernel_reports_consistent_metric_and_encoding() {
    use arcgraph_vector::{Encoding, Metric};

    let pairs: &[(&dyn DistanceKernel, Metric, Encoding)] = &[
        (&L2F32, Metric::L2, Encoding::F32),
        (&IpF32, Metric::Ip, Encoding::F32),
        (&CosineF32, Metric::Cosine, Encoding::F32),
        (&L2F16, Metric::L2, Encoding::F16),
        (&IpF16, Metric::Ip, Encoding::F16),
        (&CosineF16, Metric::Cosine, Encoding::F16),
        (&L2Sq8, Metric::L2, Encoding::Sq8),
        (&IpSq8, Metric::Ip, Encoding::Sq8),
        (&CosineSq8, Metric::Cosine, Encoding::Sq8),
        (&HammingBinary, Metric::Hamming, Encoding::Binary),
    ];
    for (k, m, e) in pairs {
        assert_eq!(k.metric(), *m, "wrong metric tag");
        assert_eq!(k.encoding(), *e, "wrong encoding tag");
        assert!(
            m.is_valid_for(*e),
            "kernel reports an invalid metric × encoding pair"
        );
    }
}
