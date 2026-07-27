//! RaBitQ standalone codec (ADR-209 slice 1).
//!
//! Implements the float-exact RaBitQ profile from Gao & Long
//! SIGMOD 2024: a trained centroid plus explicit random orthogonal
//! rotation, D sign bits, and two f32 factors per vector. The
//! bit-plane popcount FastScan path is wired into SSD navigation by preparing
//! B-bit query planes once per search and scoring payload sign bits as packed
//! `u64` words.
//!
//! Back-of-envelope budget (PD#5): encode = one D x D rotate,
//! roughly 590K MAC at 768d, about 0.1-0.5ms scalar and amortized
//! us-class multicore (10M corpus ~= 5.9 TFLOP, minutes, noise vs
//! the 23.7h Vamana build); estimate = O(D) FLOPs, SQ8-kernel
//! class; payload fetch 104B vs SQ8's 768B (the 10M-scale DRAM
//! traffic win is measured at slice 2/3, not asserted here); train
//! = centroid pass + one QR ~= O(N*D + D^3), seconds.

use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::{Encoding, VectorIndexError};

/// ADR-223 FastScan query quantization width.
///
/// B=4 was the lowest width that held recall@10 >= 0.95 in the 10M FastScan
/// confirmation run, and keeps the hot estimate loop to four packed bit-plane
/// passes.
pub const RABITQ_FASTSCAN_QUERY_BITS: u32 = 4;
const RABITQ_FASTSCAN_MIN_DIM: usize = 512;

/// Serialized RaBitQ codebook parameters.
///
/// The rotation is stored explicitly, row-major, rather than
/// regenerated from a seed. That keeps snapshot decoding stable
/// across PRNG or QR implementation changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RaBitQParams {
    pub dim: usize,
    pub centroid: Vec<f32>,
    pub rotation: Vec<f32>,
}

impl RaBitQParams {
    /// Construct a validated RaBitQ parameter set.
    ///
    /// Uses the codec-local [`VectorIndexError::DimensionMismatch`]
    /// convention for invalid shapes or non-finite values. The
    /// probabilistic orthonormality probe checks that the stored
    /// matrix preserves vector norms well enough to catch corrupted
    /// params without doing the trainer test's full O(D^3) oracle.
    pub fn try_new(
        dim: usize,
        centroid: Vec<f32>,
        rotation: Vec<f32>,
    ) -> Result<Self, VectorIndexError> {
        if dim == 0 {
            return Err(VectorIndexError::DimensionMismatch {
                expected: 0,
                got: 0,
            });
        }
        if centroid.len() != dim {
            return Err(VectorIndexError::DimensionMismatch {
                expected: dim,
                got: centroid.len(),
            });
        }
        let rot_len = dim
            .checked_mul(dim)
            .ok_or(VectorIndexError::DimensionMismatch {
                expected: dim,
                got: dim,
            })?;
        if rotation.len() != rot_len {
            return Err(VectorIndexError::DimensionMismatch {
                expected: rot_len,
                got: rotation.len(),
            });
        }
        if !centroid.iter().all(|x| x.is_finite()) || !rotation.iter().all(|x| x.is_finite()) {
            return Err(VectorIndexError::DimensionMismatch {
                expected: dim,
                got: dim,
            });
        }

        for probe_id in 0..8 {
            let mut v = seeded_unit_vector(dim, 0x7580_2090_0000_0000 ^ probe_id);
            apply_pt(&rotation, dim, &mut v);
            let norm = l2_norm(&v);
            if (norm - 1.0).abs() >= 1e-3 {
                return Err(VectorIndexError::DimensionMismatch {
                    expected: dim,
                    got: dim,
                });
            }
        }

        Ok(Self {
            dim,
            centroid,
            rotation,
        })
    }

    #[inline]
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RaBitQTrainer;

impl RaBitQTrainer {
    /// Train a RaBitQ codebook from same-dimension finite samples.
    pub fn train(&self, samples: &[&[f32]], seed: u64) -> Result<RaBitQCodebook, VectorIndexError> {
        if samples.is_empty() {
            return Err(VectorIndexError::DimensionMismatch {
                expected: 0,
                got: 0,
            });
        }
        let dim = samples[0].len();
        if dim == 0 {
            return Err(VectorIndexError::DimensionMismatch {
                expected: 0,
                got: 0,
            });
        }
        for sample in samples {
            if sample.len() != dim {
                return Err(VectorIndexError::DimensionMismatch {
                    expected: dim,
                    got: sample.len(),
                });
            }
            if !sample.iter().all(|x| x.is_finite()) {
                return Err(VectorIndexError::DimensionMismatch {
                    expected: dim,
                    got: dim,
                });
            }
        }

        let mut centroid = vec![0.0_f64; dim];
        for sample in samples {
            for (d, &x) in sample.iter().enumerate() {
                centroid[d] += f64::from(x);
            }
        }
        let n_inv = 1.0 / samples.len() as f64;
        let centroid: Vec<f32> = centroid.into_iter().map(|x| (x * n_inv) as f32).collect();

        let rotation = random_orthogonal(dim, seed)?;
        let params = RaBitQParams::try_new(dim, centroid, rotation)?;
        Ok(RaBitQCodebook { params })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RaBitQCodebook {
    params: RaBitQParams,
}

impl RaBitQCodebook {
    #[inline]
    #[must_use]
    pub const fn from_params(params: RaBitQParams) -> Self {
        Self { params }
    }

    #[inline]
    #[must_use]
    pub const fn params(&self) -> &RaBitQParams {
        &self.params
    }

    #[inline]
    #[must_use]
    pub fn into_params(self) -> RaBitQParams {
        self.params
    }

    #[inline]
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.params.dim
    }

    /// Encode an f32 vector as `[bits][f_o LE][n_o LE]`.
    pub fn encode(&self, vec: &[f32]) -> Result<Vec<u8>, VectorIndexError> {
        if vec.len() != self.dim() {
            return Err(VectorIndexError::DimensionMismatch {
                expected: self.dim(),
                got: vec.len(),
            });
        }
        let dim = self.dim();
        let code_bytes = dim.div_ceil(8);
        let mut out = vec![0u8; Encoding::RaBitQ.bytes_per_vector_unaligned(dim)];

        let residual = self.residual(vec);
        let n_o = l2_norm(&residual);
        let f_o = if n_o == 0.0 {
            0.0
        } else {
            let inv = 1.0 / n_o;
            let unit: Vec<f64> = residual.iter().map(|x| x * inv).collect();
            let y = rotate_pt(&self.params.rotation, dim, &unit);
            let mut sum_abs = 0.0_f64;
            for (d, yd) in y.iter().enumerate() {
                if *yd >= 0.0 {
                    out[d / 8] |= 1u8 << (d % 8);
                }
                sum_abs += yd.abs();
            }
            (sum_abs / (dim as f64).sqrt()) as f32
        };

        out[code_bytes..code_bytes + 4].copy_from_slice(&f_o.to_le_bytes());
        out[code_bytes + 4..code_bytes + 8].copy_from_slice(&(n_o as f32).to_le_bytes());
        Ok(out)
    }

    /// Encode and zero-pad to [`Encoding::RaBitQ`] aligned width.
    pub fn encode_aligned(&self, vec: &[f32]) -> Result<Vec<u8>, VectorIndexError> {
        let mut out = self.encode(vec)?;
        out.resize(Encoding::RaBitQ.bytes_per_vector_aligned(self.dim()), 0);
        Ok(out)
    }

    /// Prepare the asymmetric query vector once per search.
    pub fn prepare_query(&self, q: &[f32]) -> Result<RaBitQQuery, VectorIndexError> {
        if q.len() != self.dim() {
            return Err(VectorIndexError::DimensionMismatch {
                expected: self.dim(),
                got: q.len(),
            });
        }
        let residual = self.residual(q);
        let n_q = l2_norm(&residual);
        let y_q = if n_q == 0.0 {
            vec![0.0; self.dim()]
        } else {
            let inv = 1.0 / n_q;
            let unit: Vec<f64> = residual.iter().map(|x| x * inv).collect();
            rotate_pt(&self.params.rotation, self.dim(), &unit)
                .into_iter()
                .map(|x| x as f32)
                .collect()
        };
        Ok(RaBitQQuery::new(y_q, n_q as f32).with_fastscan(RABITQ_FASTSCAN_QUERY_BITS))
    }

    /// Estimate the unit-residual inner product `<o_bar, q_bar>`.
    #[must_use]
    pub fn estimate_ip_unit(&self, query: &RaBitQQuery, payload: &[u8]) -> f32 {
        debug_assert_eq!(query.y_q.len(), self.dim());
        estimate_ip_unit(query, payload)
    }

    /// Estimate squared L2 distance from prepared query to payload.
    #[must_use]
    pub fn estimate_l2_sq(&self, query: &RaBitQQuery, payload: &[u8]) -> f32 {
        debug_assert_eq!(query.y_q.len(), self.dim());
        estimate_l2_sq(query, payload)
    }

    /// Fallible payload parser for tests and snapshot validation.
    pub fn parse_payload<'a>(
        &self,
        payload: &'a [u8],
    ) -> Result<RaBitQPayload<'a>, VectorIndexError> {
        let need = Encoding::RaBitQ.bytes_per_vector_unaligned(self.dim());
        if payload.len() < need {
            return Err(VectorIndexError::DimensionMismatch {
                expected: need,
                got: payload.len(),
            });
        }
        let (codes, f_o, n_o) = self.parse_payload_unchecked(payload);
        Ok(RaBitQPayload { codes, f_o, n_o })
    }

    /// Reference/testing reconstruction from sign bits, norm, and
    /// centroid. RaBitQ is not invertible beyond this codeword.
    #[must_use]
    pub fn decode_reference(&self, payload: &[u8]) -> Vec<f32> {
        let (codes, _, n_o) = self.parse_payload_unchecked(payload);
        let scale = f64::from(n_o) / (self.dim() as f64).sqrt();
        let mut out = vec![0.0_f32; self.dim()];
        for (i, out_i) in out.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for d in 0..self.dim() {
                let bit = (codes[d / 8] >> (d % 8)) & 1;
                let sign = if bit == 1 { 1.0 } else { -1.0 };
                acc += f64::from(self.params.rotation[i * self.dim() + d]) * sign;
            }
            *out_i = (f64::from(self.params.centroid[i]) + scale * acc) as f32;
        }
        out
    }

    fn residual(&self, vec: &[f32]) -> Vec<f64> {
        vec.iter()
            .zip(&self.params.centroid)
            .map(|(&x, &c)| f64::from(x) - f64::from(c))
            .collect()
    }

    fn parse_payload_unchecked<'a>(&self, payload: &'a [u8]) -> (&'a [u8], f32, f32) {
        let code_bytes = self.dim().div_ceil(8);
        let codes = &payload[..code_bytes];
        let f_o = f32::from_le_bytes(payload[code_bytes..code_bytes + 4].try_into().unwrap());
        let n_o = f32::from_le_bytes(payload[code_bytes + 4..code_bytes + 8].try_into().unwrap());
        (codes, f_o, n_o)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RaBitQQuery {
    pub y_q: Vec<f32>,
    pub n_q: f32,
    pub fastscan: Option<RaBitQFastScanQuery>,
}

impl RaBitQQuery {
    #[inline]
    #[must_use]
    pub const fn new(y_q: Vec<f32>, n_q: f32) -> Self {
        Self {
            y_q,
            n_q,
            fastscan: None,
        }
    }

    #[must_use]
    pub fn with_fastscan(mut self, bits: u32) -> Self {
        self.fastscan = RaBitQFastScanQuery::prepare(&self.y_q, bits);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RaBitQFastScanQuery {
    planes: Vec<Vec<u64>>,
    plane_popcounts: Vec<u32>,
    words: usize,
    min: f64,
    inv_scale: f64,
    bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaBitQPayload<'a> {
    pub codes: &'a [u8],
    pub f_o: f32,
    pub n_o: f32,
}

/// Estimate the unit-residual inner product `<o_bar, q_bar>` from a prepared
/// RaBitQ query and one encoded payload.
///
/// This free function is the SSD nav seam: once a query is prepared, the graph
/// search path needs only the query vector and payload bytes, not the codebook
/// that created them.
#[must_use]
#[allow(unsafe_code)]
pub fn estimate_ip_unit(query: &RaBitQQuery, payload: &[u8]) -> f32 {
    let dim = query.y_q.len();
    debug_assert!(payload.len() >= Encoding::RaBitQ.bytes_per_vector_unaligned(dim));
    let code_bytes = dim.div_ceil(8);
    let f_o = f32::from_le_bytes(payload[code_bytes..code_bytes + 4].try_into().unwrap());
    if f_o == 0.0 {
        return 0.0;
    }
    if let Some(fastscan) = &query.fastscan {
        return fastscan.estimate_ip_unit(payload, dim, f_o);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2+FMA are verified present immediately above via
            // runtime CPU feature detection; the callee's `#[target_feature]`
            // re-asserts that precondition. `query.y_q` and `payload` are
            // borrowed slices, and the debug assertion above verifies payload
            // length for the query dimension in debug builds.
            return unsafe { x86_avx2_fma::estimate_ip_unit(query, payload, f_o) };
        }
    }

    estimate_ip_unit_scalar(query, payload, f_o)
}

#[inline]
fn estimate_ip_unit_scalar(query: &RaBitQQuery, payload: &[u8], f_o: f32) -> f32 {
    let dim = query.y_q.len();
    let code_bytes = dim.div_ceil(8);
    let codes = &payload[..code_bytes];
    let mut s_dot = 0.0_f64;
    for d in 0..dim {
        let bit = (codes[d / 8] >> (d % 8)) & 1;
        let y = f64::from(query.y_q[d]);
        s_dot += if bit == 1 { y } else { -y };
    }
    ((s_dot / (dim as f64).sqrt()) / f64::from(f_o)) as f32
}

impl RaBitQFastScanQuery {
    fn prepare(y_q: &[f32], bits: u32) -> Option<Self> {
        if !(4..=6).contains(&bits) || y_q.len() < RABITQ_FASTSCAN_MIN_DIM {
            return None;
        }
        let dim = y_q.len();
        let words = dim.div_ceil(64);
        let min = y_q.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        let max = y_q.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let range = max - min;
        if !range.is_finite() || range <= 1.0e-12 {
            return None;
        }
        let levels = ((1_u64 << bits) - 1) as f64;
        let scale = levels / range;
        let mut planes = vec![vec![0_u64; words]; bits as usize];
        for (d, &y) in y_q.iter().enumerate() {
            let quantized = ((f64::from(y) - min) * scale).round();
            let quantized = (quantized as i64).clamp(0, levels as i64) as u64;
            for bit in 0..bits {
                if ((quantized >> bit) & 1) == 1 {
                    planes[bit as usize][d / 64] |= 1_u64 << (d % 64);
                }
            }
        }
        let plane_popcounts = planes
            .iter()
            .map(|plane| plane.iter().map(|word| word.count_ones()).sum())
            .collect();
        Some(Self {
            planes,
            plane_popcounts,
            words,
            min,
            inv_scale: 1.0 / scale,
            bits,
        })
    }

    #[inline]
    fn estimate_ip_unit(&self, payload: &[u8], dim: usize, f_o: f32) -> f32 {
        let code_bytes = dim.div_ceil(8);
        debug_assert_eq!(self.words, dim.div_ceil(64));
        debug_assert!(payload.len() >= code_bytes + 8);

        let mut code_popcount = 0_u32;
        let mut acc = 0.0_f64;
        for word_idx in 0..self.words {
            let code_word = payload_code_word(payload, code_bytes, word_idx);
            code_popcount += code_word.count_ones();
            for bit in 0..self.bits as usize {
                let plane = self.planes[bit][word_idx];
                acc += (1_u64 << bit) as f64 * 2.0 * f64::from((code_word & plane).count_ones());
            }
        }
        for bit in 0..self.bits as usize {
            acc -= (1_u64 << bit) as f64 * f64::from(self.plane_popcounts[bit]);
        }
        let signed_code_sum = 2.0 * f64::from(code_popcount) - dim as f64;
        let s_dot = self.min * signed_code_sum + self.inv_scale * acc;
        ((s_dot / (dim as f64).sqrt()) / f64::from(f_o)) as f32
    }
}

#[inline]
fn payload_code_word(payload: &[u8], code_bytes: usize, word_idx: usize) -> u64 {
    let offset = word_idx * 8;
    let remaining = code_bytes.saturating_sub(offset);
    if remaining >= 8 {
        u64::from_le_bytes(payload[offset..offset + 8].try_into().unwrap())
    } else {
        let mut word = 0_u64;
        for i in 0..remaining {
            word |= u64::from(payload[offset + i]) << (i * 8);
        }
        word
    }
}

/// Estimate squared L2 distance from a prepared query to an encoded payload.
#[must_use]
pub fn estimate_l2_sq(query: &RaBitQQuery, payload: &[u8]) -> f32 {
    let dim = query.y_q.len();
    debug_assert!(payload.len() >= Encoding::RaBitQ.bytes_per_vector_unaligned(dim));
    let code_bytes = dim.div_ceil(8);
    let n_o = f32::from_le_bytes(payload[code_bytes + 4..code_bytes + 8].try_into().unwrap());
    let ip = estimate_ip_unit(query, payload);
    let n_o = f64::from(n_o);
    let n_q = f64::from(query.n_q);
    let est = n_o * n_o + n_q * n_q - 2.0 * n_o * n_q * f64::from(ip);
    // Estimator noise can dip slightly negative near zero.
    est.max(0.0) as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod x86_avx2_fma {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use super::{RaBitQQuery, estimate_ip_unit_scalar};
    use crate::Encoding;

    const SIGN_LUT: [[f64; 8]; 256] = sign_lut();

    const fn sign_lut() -> [[f64; 8]; 256] {
        let mut lut = [[0.0_f64; 8]; 256];
        let mut byte = 0;
        while byte < 256 {
            let mut lane = 0;
            while lane < 8 {
                lut[byte][lane] = if ((byte >> lane) & 1) == 1 { 1.0 } else { -1.0 };
                lane += 1;
            }
            byte += 1;
        }
        lut
    }

    /// AVX2+FMA RaBitQ sign-dot kernel.
    ///
    /// Budget: the scalar hot loop does one payload byte lookup, one variable
    /// bit extract, one f32→f64 widen, and one data-dependent add/sub per
    /// dimension. At 768d that is 768 unpredictable scalar iterations. This
    /// kernel consumes one packed code byte per 8 query lanes, widens to f64
    /// lanes, and accumulates with FMA, leaving only the final partial byte to
    /// the scalar fallback.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn estimate_ip_unit(query: &RaBitQQuery, payload: &[u8], f_o: f32) -> f32 {
        let dim = query.y_q.len();
        debug_assert!(payload.len() >= Encoding::RaBitQ.bytes_per_vector_unaligned(dim));
        let full_bytes = dim / 8;
        let codes = &payload[..dim.div_ceil(8)];
        let y_q = &query.y_q;

        let mut acc_lo = _mm256_setzero_pd();
        let mut acc_hi = _mm256_setzero_pd();

        for (byte_idx, &code) in codes.iter().take(full_bytes).enumerate() {
            let offset = byte_idx * 8;
            let signs = SIGN_LUT[usize::from(code)].as_ptr();
            // SAFETY:
            // - this function is `#[target_feature(enable = "avx2,fma")]`
            //   and is called only after runtime AVX2+FMA detection;
            // - `offset < full_bytes * 8 <= dim`, so `offset..offset + 8`
            //   is in-bounds for `query.y_q`;
            // - `_mm256_loadu_ps` accepts unaligned pointers derived from the
            //   valid `&[f32]` query slice;
            // - `signs` points to one 8-lane row in the static sign lookup
            //   table, so both 4-lane f64 loads are in-bounds.
            unsafe {
                let y = _mm256_loadu_ps(y_q.as_ptr().add(offset));
                let signs_lo = _mm256_loadu_pd(signs);
                let signs_hi = _mm256_loadu_pd(signs.add(4));

                let y_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(y));
                let y_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(y, 1));
                acc_lo = _mm256_fmadd_pd(y_lo, signs_lo, acc_lo);
                acc_hi = _mm256_fmadd_pd(y_hi, signs_hi, acc_hi);
            }
        }

        let mut sums = [0.0_f64; 4];
        // SAFETY: `sums` has four contiguous f64 lanes, exactly matching one
        // `__m256d` store, and unaligned stores are permitted by
        // `_mm256_storeu_pd`; AVX is enabled by this function's target feature.
        unsafe { _mm256_storeu_pd(sums.as_mut_ptr(), _mm256_add_pd(acc_lo, acc_hi)) };
        let mut s_dot = sums.iter().sum::<f64>();

        for d in (full_bytes * 8)..dim {
            let bit = (codes[d / 8] >> (d % 8)) & 1;
            let y = f64::from(y_q[d]);
            s_dot += if bit == 1 { y } else { -y };
        }

        let simd = ((s_dot / (dim as f64).sqrt()) / f64::from(f_o)) as f32;
        debug_assert!(
            {
                let scalar = estimate_ip_unit_scalar(query, payload, f_o);
                (simd - scalar).abs() <= 2.0e-7
            },
            "RaBitQ AVX2+FMA estimator diverged from scalar"
        );
        simd
    }
}

fn random_orthogonal(dim: usize, seed: u64) -> Result<Vec<f32>, VectorIndexError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cols = vec![vec![0.0_f64; dim]; dim];
    for col in &mut cols {
        for x in col {
            *x = gaussian(&mut rng);
        }
    }

    for j in 0..dim {
        for _ in 0..2 {
            for k in 0..j {
                let dot = dot(&cols[j], &cols[k]);
                let col_k = cols[k].clone();
                for (x_j, x_k) in cols[j].iter_mut().zip(col_k) {
                    *x_j -= dot * x_k;
                }
            }
        }
        let norm = l2_norm(&cols[j]);
        if norm == 0.0 || !norm.is_finite() {
            return Err(VectorIndexError::DimensionMismatch {
                expected: dim,
                got: dim,
            });
        }
        for x in &mut cols[j] {
            *x /= norm;
        }
    }

    let mut rotation = vec![0.0_f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            rotation[i * dim + j] = cols[j][i] as f32;
        }
    }
    Ok(rotation)
}

fn gaussian(rng: &mut StdRng) -> f64 {
    let u1 = rng.random_range::<f64, _>(f64::MIN_POSITIVE..1.0);
    let u2 = rng.random_range::<f64, _>(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn rotate_pt(rotation: &[f32], dim: usize, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; dim];
    for j in 0..dim {
        let mut acc = 0.0_f64;
        for i in 0..dim {
            acc += f64::from(rotation[i * dim + j]) * v[i];
        }
        out[j] = acc;
    }
    out
}

fn apply_pt(rotation: &[f32], dim: usize, v: &mut Vec<f64>) {
    *v = rotate_pt(rotation, dim, v);
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn seeded_unit_vector(dim: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut v = Vec::with_capacity(dim);
    for _ in 0..dim {
        v.push(next_unit_f64(&mut state) * 2.0 - 1.0);
    }
    let norm = l2_norm(&v);
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn next_unit_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn codebook(dim: usize) -> RaBitQCodebook {
        let samples: Vec<Vec<f32>> = (0..(dim + 3))
            .map(|i| (0..dim).map(|d| (i as f32 + d as f32) * 0.01).collect())
            .collect();
        let refs: Vec<&[f32]> = samples.iter().map(Vec::as_slice).collect();
        RaBitQTrainer.train(&refs, 7).unwrap()
    }

    #[test]
    fn payload_layout_and_lsb_packing() {
        let params = RaBitQParams::try_new(8, vec![0.0; 8], identity(8)).unwrap();
        let cb = RaBitQCodebook::from_params(params);
        let payload = cb
            .encode(&[1.0, -1.0, 1.0, -1.0, 0.0, 2.0, -2.0, 3.0])
            .unwrap();
        assert_eq!(payload.len(), 9);
        assert_eq!(payload[0], 0b1011_0101);
    }

    #[test]
    fn degenerate_object_returns_query_norm_sq() {
        let cb = codebook(4);
        let c = cb.params.centroid.clone();
        let payload = cb.encode(&c).unwrap();
        let q: Vec<f32> = c.iter().map(|x| x + 1.0).collect();
        let prepared = cb.prepare_query(&q).unwrap();
        assert_eq!(
            cb.estimate_l2_sq(&prepared, &payload),
            prepared.n_q * prepared.n_q
        );
    }

    #[test]
    fn free_estimators_match_codebook_methods_bit_for_bit() {
        let cb = codebook(16);
        let q: Vec<f32> = (0..16).map(|d| d as f32 * 0.07 - 0.3).collect();
        let v: Vec<f32> = (0..16).map(|d| (d as f32).sin() * 0.2).collect();
        let prepared = cb.prepare_query(&q).unwrap();
        let payload = cb.encode_aligned(&v).unwrap();
        assert_eq!(
            cb.estimate_ip_unit(&prepared, &payload),
            estimate_ip_unit(&prepared, &payload)
        );
        assert_eq!(
            cb.estimate_l2_sq(&prepared, &payload),
            estimate_l2_sq(&prepared, &payload)
        );
    }

    #[test]
    fn scalar_ip_unit_matches_runtime_dispatch() {
        for dim in [1, 3, 8, 31, 64, 127, 768] {
            let mut rng = StdRng::seed_from_u64(0x7580_2090_1000_0000 ^ dim as u64);
            for _case in 0..32 {
                let (query, payload, f_o) = random_query_payload(dim, &mut rng);
                let scalar = estimate_ip_unit_scalar(&query, &payload, f_o);
                let runtime = estimate_ip_unit(&query, &payload);
                assert!(
                    (runtime - scalar).abs() <= 2.0e-7,
                    "dim={dim} runtime={runtime} scalar={scalar}"
                );
            }
        }
    }

    #[test]
    fn fastscan_ip_unit_matches_spike_popcount_reference() {
        for dim in [512, 768] {
            let mut rng = StdRng::seed_from_u64(0x7580_2230_1000_0000 ^ dim as u64);
            for _case in 0..32 {
                let (query, payload, _) = random_query_payload(dim, &mut rng);
                let fast = query.clone().with_fastscan(RABITQ_FASTSCAN_QUERY_BITS);
                assert!(fast.fastscan.is_some());
                let got = estimate_ip_unit(&fast, &payload);
                let want = fastscan_reference_ip_unit(&fast, &payload);
                assert!(
                    (got - want).abs() <= 2.0e-7,
                    "dim={dim} got={got} want={want}"
                );
            }
        }
    }

    #[test]
    fn fastscan_preserves_top_k_against_scalar_on_random_768d() {
        const DIM: usize = 768;
        const QUERIES: usize = 20;
        const CANDIDATES: usize = 128;
        const K: usize = 10;

        let mut rng = StdRng::seed_from_u64(0x7580_2230_2000_0000);
        for _ in 0..QUERIES {
            let y_q: Vec<f32> = (0..DIM)
                .map(|_| rng.random_range(-1.0_f32..1.0_f32))
                .collect();
            let scalar = RaBitQQuery::new(y_q.clone(), 1.0);
            let fast = scalar.clone().with_fastscan(RABITQ_FASTSCAN_QUERY_BITS);
            assert!(fast.fastscan.is_some());
            let payloads = rank_margin_payloads(&y_q, CANDIDATES, K, &mut rng);
            let scalar_top = top_k_by_estimate(&scalar, &payloads, K);
            let fast_top = top_k_by_estimate(&fast, &payloads, K);
            assert_eq!(fast_top, scalar_top, "FastScan B=4 changed top-{K} order");
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[allow(unsafe_code)]
    #[test]
    fn avx2_fma_ip_unit_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        for dim in [1, 3, 8, 31, 64, 127, 768] {
            let mut rng = StdRng::seed_from_u64(0x7580_2090_2000_0000 ^ dim as u64);
            for _case in 0..64 {
                let (query, payload, f_o) = random_query_payload(dim, &mut rng);
                let scalar = estimate_ip_unit_scalar(&query, &payload, f_o);
                // SAFETY: this test checks AVX2+FMA runtime feature detection
                // immediately above; payloads are generated with the exact
                // RaBitQ unaligned width for `query.y_q.len()`, and the SIMD
                // helper handles non-multiple-of-8 tails with scalar cleanup.
                let simd = unsafe { x86_avx2_fma::estimate_ip_unit(&query, &payload, f_o) };
                assert!(
                    (simd - scalar).abs() <= 2.0e-7,
                    "dim={dim} simd={simd} scalar={scalar}"
                );
            }
        }
    }

    #[test]
    fn parse_rejects_short_payload() {
        let cb = codebook(9);
        let err = cb.parse_payload(&[0; 2]).unwrap_err();
        assert!(matches!(
            err,
            VectorIndexError::DimensionMismatch {
                expected: 10,
                got: 2
            }
        ));
    }

    fn identity(dim: usize) -> Vec<f32> {
        let mut p = vec![0.0; dim * dim];
        for d in 0..dim {
            p[d * dim + d] = 1.0;
        }
        p
    }

    fn random_query_payload(dim: usize, rng: &mut StdRng) -> (RaBitQQuery, Vec<u8>, f32) {
        let mut payload = vec![0u8; Encoding::RaBitQ.bytes_per_vector_unaligned(dim)];
        let code_bytes = dim.div_ceil(8);
        rng.fill_bytes(&mut payload[..code_bytes]);
        let f_o = rng.random_range(0.1_f32..2.0_f32);
        let n_o = rng.random_range(0.1_f32..3.0_f32);
        payload[code_bytes..code_bytes + 4].copy_from_slice(&f_o.to_le_bytes());
        payload[code_bytes + 4..code_bytes + 8].copy_from_slice(&n_o.to_le_bytes());

        let y_q = (0..dim)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect();
        let n_q = rng.random_range(0.1_f32..3.0_f32);
        (RaBitQQuery::new(y_q, n_q), payload, f_o)
    }

    fn top_k_by_estimate(query: &RaBitQQuery, payloads: &[Vec<u8>], k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = payloads
            .iter()
            .enumerate()
            .map(|(idx, payload)| (idx, estimate_l2_sq(query, payload)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        scored.truncate(k);
        scored.into_iter().map(|(idx, _)| idx).collect()
    }

    fn rank_margin_payloads(
        y_q: &[f32],
        candidates: usize,
        k: usize,
        rng: &mut StdRng,
    ) -> Vec<Vec<u8>> {
        let dim = y_q.len();
        let mut dims_by_abs: Vec<usize> = (0..dim).collect();
        dims_by_abs.sort_by(|&a, &b| y_q[b].abs().total_cmp(&y_q[a].abs()).then(a.cmp(&b)));
        let base_signs: Vec<bool> = y_q.iter().map(|&y| y >= 0.0).collect();
        (0..candidates)
            .map(|idx| {
                let mut signs = base_signs.clone();
                let flips = if idx < k {
                    idx * 3
                } else {
                    96 + ((idx - k) % 64)
                };
                for &dim_idx in dims_by_abs.iter().take(flips) {
                    signs[dim_idx] = !signs[dim_idx];
                }
                if idx >= k {
                    for &dim_idx in dims_by_abs.iter().skip(flips).take(96) {
                        if rng.random_range(0_u32..2) == 1 {
                            signs[dim_idx] = !signs[dim_idx];
                        }
                    }
                }
                payload_from_signs(&signs, 1.0, 1.0)
            })
            .collect()
    }

    fn payload_from_signs(signs: &[bool], f_o: f32, n_o: f32) -> Vec<u8> {
        let dim = signs.len();
        let code_bytes = dim.div_ceil(8);
        let mut payload = vec![0_u8; Encoding::RaBitQ.bytes_per_vector_aligned(dim)];
        for (d, sign) in signs.iter().enumerate() {
            if *sign {
                payload[d / 8] |= 1_u8 << (d % 8);
            }
        }
        payload[code_bytes..code_bytes + 4].copy_from_slice(&f_o.to_le_bytes());
        payload[code_bytes + 4..code_bytes + 8].copy_from_slice(&n_o.to_le_bytes());
        payload
    }

    fn fastscan_reference_ip_unit(query: &RaBitQQuery, payload: &[u8]) -> f32 {
        let dim = query.y_q.len();
        let code_bytes = dim.div_ceil(8);
        let f_o = f32::from_le_bytes(payload[code_bytes..code_bytes + 4].try_into().unwrap());
        if f_o == 0.0 {
            return 0.0;
        }
        let fastscan = query.fastscan.as_ref().expect("fastscan metadata");
        let mut code_words = vec![0_u64; fastscan.words];
        for i in 0..code_bytes {
            code_words[i / 8] |= u64::from(payload[i]) << ((i % 8) * 8);
        }
        let code_popcount: u32 = code_words.iter().map(|word| word.count_ones()).sum();
        let signed_code_sum = 2.0 * f64::from(code_popcount) - dim as f64;
        let mut acc = 0.0_f64;
        for bit in 0..fastscan.bits as usize {
            let mut and_popcount = 0_u32;
            for (code_word, plane_word) in code_words.iter().zip(&fastscan.planes[bit]) {
                and_popcount += (code_word & plane_word).count_ones();
            }
            acc += (1_u64 << bit) as f64
                * (2.0 * f64::from(and_popcount) - f64::from(fastscan.plane_popcounts[bit]));
        }
        let s_dot = fastscan.min * signed_code_sum + fastscan.inv_scale * acc;
        ((s_dot / (dim as f64).sqrt()) / f64::from(f_o)) as f32
    }
}
