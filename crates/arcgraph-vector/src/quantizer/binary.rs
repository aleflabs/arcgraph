//! Binary (1-bit-per-dim) sign-function encoder + decoder.
//!
//! Per ADR-035 §3.3 + D-4 + S-1 fold-in:
//!
//! - **No training.** Binary encoding is the deterministic sign
//!   function: `bit[d] = (f32_vec[d] > 0)`.
//! - **Packing.** Bit `d` lives in byte `d / 8` at intra-byte
//!   position `d % 8` (LSB-first within each byte). The convention
//!   matches the natural bitset layout — bit 0 of byte 0 is the
//!   LSB. Hamming distance is order-agnostic (XOR + popcount), so
//!   the choice does not affect the distance kernel as long as the
//!   encoder is self-consistent across queries and inserts.
//! - **Output size.** [`binary_encode`] returns the unaligned
//!   packed length `(dim + 7) / 8` bytes. The arena allocator
//!   pads to the next 64-byte boundary (per ADR-035 §S-1) for
//!   cache-line-aligned slots; [`binary_encode_aligned`] applies
//!   that padding directly so callers that write straight into
//!   the arena slot can use a single call.
//!
//! ## Why the sign function and nothing else
//!
//! Binary quantization is the recall-floor encoder: it keeps only
//! the sign bit per dim, a 32× memory reduction. Recall@10 drops
//! to ~0.85 from raw F32's ~0.97 (per Qdrant 2024 measurements);
//! the rescore path (Slice E.3 for DiskANN; ADR-035 AC-2)
//! recovers it to ≥ 0.95 by re-ranking the top-K binary
//! candidates against full-precision rescore vectors.
//!
//! ## sign(0) ambiguity
//!
//! `sign(0)` is implementation-defined. We adopt the convention
//! `bit_d = (f32_d > 0)` — so `0.0` and `-0.0` both encode as
//! bit `0`. This matches the prevailing Qdrant / Faiss choice and
//! is consistent with treating a coordinate that is structurally
//! zero (e.g., a sparse-projected dim that was never populated)
//! as a non-positive contribution.
//!
//! Per ADR-035 §9.3, an all-zero vector (every coordinate exactly
//! `0.0`) encodes to all-zero bits. The Hamming distance between
//! an all-zero vector and any other vector is then equal to the
//! popcount of the other vector's positive-coordinate mask.
//! Operators rebuilding from a degenerate-input collection should
//! prefer SQ8 (which surfaces the `§9.3` per-dim warn) over
//! Binary; the binary path makes no judgment about input
//! distribution.
//!
//! ## OQ-V4 re-encode contract
//!
//! Like SQ8 (see `super::sq8`), [`binary_encode`] is
//! arena-state-free: pure function of the input slice. The
//! re-encode pass calls it per-vector under the published-flag
//! pattern (ADR-035 §5.2 step 6).

use crate::Encoding;

/// Encode a unit-step `f32` slice into a packed binary vector.
///
/// Bit `d` of byte `d / 8` is set iff `vec[d] > 0`. The output
/// length is `(vec.len() + 7) / 8`. For dim = 768 (the v1.0
/// reference dim) this is 96 bytes; arena slots pad to 128 bytes
/// per ADR-035 §S-1 — see [`binary_encode_aligned`].
///
/// Empty input yields an empty output (zero-dim case).
#[must_use]
pub fn binary_encode(vec: &[f32]) -> Vec<u8> {
    let dim = vec.len();
    let n_bytes = dim.div_ceil(8);
    let mut out = vec![0u8; n_bytes];
    for (d, &x) in vec.iter().enumerate() {
        if x > 0.0 {
            // LSB-first: bit `d % 8` of byte `d / 8`.
            out[d / 8] |= 1u8 << (d % 8);
        }
    }
    out
}

/// Encode a unit-step `f32` slice into a 64-byte cache-line-aligned
/// packed binary vector.
///
/// Equivalent to [`binary_encode`] but pads the output to
/// [`Encoding::bytes_per_vector_aligned`] for the input's dim
/// (with `Encoding::Binary` as the receiver). The padding bytes
/// are zero — they don't represent any dimension and are ignored
/// by [`binary_decode`] because the caller passes the original
/// `dim` to that function.
///
/// At dim = 768 the packed length is 96 bytes and the aligned
/// length is 128 bytes (per ADR-035 §S-1: 64-byte cache-line
/// alignment, 96 → next multiple of 64 = 128).
#[must_use]
pub fn binary_encode_aligned(vec: &[f32]) -> Vec<u8> {
    let dim = vec.len();
    let aligned_bytes = Encoding::Binary.bytes_per_vector_aligned(dim);
    let mut out = binary_encode(vec);
    out.resize(aligned_bytes, 0u8);
    out
}

/// Decode a packed binary vector back to ±1.0 `f32` per dim.
///
/// Each bit `d` of byte `d / 8` (LSB-first) maps to:
/// - `1` → `+1.0`
/// - `0` → `-1.0`
///
/// Used for reference / testing — production search reads packed
/// binary vectors directly into the Hamming distance kernel and
/// never materializes the ±1 form. See ADR-035 §3.3 ("decode is
/// for reference / testing only").
///
/// `bytes` may be longer than the strictly required `(dim + 7) /
/// 8` (e.g., aligned form): the function reads only the first
/// `dim` bits.
///
/// # Panics
///
/// Panics if `bytes.len() < (dim + 7) / 8` — i.e., the packed
/// vector is shorter than the requested dimension. This is a
/// programmer error (the encoder always produces ≥ that many
/// bytes), so we panic rather than introduce a fallible decode
/// surface for the reference codec.
#[must_use]
pub fn binary_decode(bytes: &[u8], dim: usize) -> Vec<f32> {
    let needed = dim.div_ceil(8);
    assert!(
        bytes.len() >= needed,
        "binary_decode: bytes.len()={} too short for dim={dim} (need >= {needed})",
        bytes.len()
    );
    let mut out = Vec::with_capacity(dim);
    for d in 0..dim {
        let bit = (bytes[d / 8] >> (d % 8)) & 1;
        out.push(if bit == 1 { 1.0 } else { -1.0 });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_empty_output() {
        let v: &[f32] = &[];
        assert!(binary_encode(v).is_empty());
    }

    #[test]
    fn empty_aligned_yields_zero_aligned_output() {
        // Aligned at dim=0 is `0_usize.next_multiple_of(64) = 0`,
        // so the aligned form is also empty.
        let v: &[f32] = &[];
        assert!(binary_encode_aligned(v).is_empty());
    }

    #[test]
    fn all_positive_sets_all_bits() {
        let v: Vec<f32> = vec![1.0; 16];
        let bytes = binary_encode(&v);
        assert_eq!(bytes, vec![0xFFu8, 0xFFu8]);
    }

    #[test]
    fn all_zero_sets_no_bits() {
        // sign(0) per the §9.3 convention: bit_d = (x > 0). A
        // zero coordinate encodes as bit 0.
        let v: Vec<f32> = vec![0.0; 16];
        let bytes = binary_encode(&v);
        assert_eq!(bytes, vec![0u8, 0u8]);
    }

    #[test]
    fn negative_zero_encodes_as_bit_zero() {
        // (-0.0 > 0) is false; bit clear.
        let v: Vec<f32> = vec![-0.0; 8];
        let bytes = binary_encode(&v);
        assert_eq!(bytes, vec![0u8]);
    }

    #[test]
    fn alternating_signs_match_lsb_first_layout() {
        // dims [0,1,2,3,4,5,6,7] with signs [+,-,+,-,+,-,+,-]
        // → bits [1,0,1,0,1,0,1,0] → byte 0b01010101 = 0x55.
        let v: Vec<f32> = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
        let bytes = binary_encode(&v);
        assert_eq!(bytes, vec![0x55u8]);
    }

    #[test]
    fn odd_dim_pads_with_zero_in_high_bits() {
        // dim=9 → 2 bytes; only bit 0 of byte 1 is set; bits 1..7
        // of byte 1 are unused (zero).
        let v: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let bytes = binary_encode(&v);
        assert_eq!(bytes, vec![0xFFu8, 0x01u8]);
    }

    #[test]
    fn packed_length_for_dim_768() {
        let v: Vec<f32> = vec![0.5; 768];
        let bytes = binary_encode(&v);
        assert_eq!(bytes.len(), 96);
    }

    #[test]
    fn aligned_length_for_dim_768_is_128_per_s1() {
        let v: Vec<f32> = vec![0.5; 768];
        let bytes = binary_encode_aligned(&v);
        assert_eq!(bytes.len(), 128);
        assert_eq!(bytes.len(), Encoding::Binary.bytes_per_vector_aligned(768));
        // First 96 bytes match the unaligned form; trailing 32
        // bytes are zero padding.
        assert_eq!(&bytes[..96], &binary_encode(&v)[..]);
        assert!(bytes[96..].iter().all(|&b| b == 0));
    }

    #[test]
    fn aligned_padding_is_per_s1_at_dim_513() {
        // 513 / 8 ceil = 65 bytes; next multiple of 64 = 128.
        let v: Vec<f32> = vec![0.5; 513];
        assert_eq!(binary_encode_aligned(&v).len(), 128);
    }

    #[test]
    fn decode_round_trip_sign_only() {
        let v: Vec<f32> = vec![0.1, -0.2, 3.0, -4.0, 0.0, 0.5, -0.5, 1e-9, -1e-9, 100.0];
        let bytes = binary_encode(&v);
        let dec = binary_decode(&bytes, v.len());
        for i in 0..v.len() {
            // Sign preservation: positive → +1.0, non-positive → -1.0
            // (note: 0.0 maps to -1.0 per the (x > 0) convention).
            let expected_sign = if v[i] > 0.0 { 1.0 } else { -1.0 };
            assert_eq!(
                dec[i], expected_sign,
                "i={i} input={} decoded={} expected_sign={}",
                v[i], dec[i], expected_sign
            );
        }
    }

    #[test]
    fn decode_handles_aligned_input() {
        let v: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0];
        let aligned = binary_encode_aligned(&v);
        let dec = binary_decode(&aligned, v.len());
        assert_eq!(dec, vec![1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    #[should_panic(expected = "too short")]
    fn decode_panics_on_too_short_bytes() {
        let bytes = vec![0u8]; // only 1 byte
        let _ = binary_decode(&bytes, 16); // need 2 bytes
    }

    #[test]
    fn dim_zero_decode_is_empty() {
        let dec = binary_decode(&[], 0);
        assert!(dec.is_empty());
    }
}
