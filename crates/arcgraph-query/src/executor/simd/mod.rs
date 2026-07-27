//! M4-64b — SIMD intrinsics on hot operators.
//!
//! Per ADR-038 amendment-02 §M4.f + amendment-03 §Structural-1, this
//! module ships hand-written SIMD acceleration for three hot-path
//! operators:
//!
//! - [`filter::simd_filter_i64_cmp`] — vectorized i64 comparison (eq /
//!   ne / lt / le / gt / ge) used by
//!   [`crate::executor::ops::filter::FilterOp`] when the predicate
//!   shape is `<VarRef.PropertyAccess> <BinOp> <IntLiteral>`.
//! - [`expand::simd_neighbor_match_mask`] — vectorized NodeId
//!   membership check used by [`crate::executor::ops::expand::ExpandOp`]
//!   when a `dst_allow_set` post-substrate filter is configured (forward
//!   pin: planner pushdown of `WHERE b.id IN [...]` lands at M4-72; the
//!   wiring is real today and the v1.0-alpha tests + bench exercise it
//!   directly).
//! - [`rrf::simd_rrf_scores`] — vectorized `1 / (k + rank)` score-vector
//!   utility and benchmark path. Production fusion uses the shared exact
//!   `f64` implementation in [`crate::executor::fusion`] so every public
//!   search surface returns identical scores.
//!
//! # Per-arch dispatch
//!
//! Each helper exposes a per-arch backend behind a stable scalar
//! signature; the selection is made at runtime via
//! [`is_x86_feature_detected`] / [`std::arch::is_aarch64_feature_detected`]
//! so the same binary works on machines that lack the target feature.
//! Compile-time `cfg!(target_arch = …)` gates the
//! `unsafe`-`std::arch`-intrinsic code paths so a non-matching target
//! triple cannot pull in undefined intrinsics.
//!
//! - **x86_64 + AVX2**: 256-bit lanes via `_mm256_*`. 4 × i64 / 8 × f32
//!   per register; the helpers unroll one register per iteration. Per
//!   Intel Optimization Reference §15.5.1 the AVX2 throughput on
//!   Skylake/Ice Lake is ~1 256-bit op/cycle on Port 5; the helpers
//!   deliver well above 4 lanes/cycle in practice.
//! - **AArch64 + NEON**: 128-bit lanes via `vceqq_*` / `vdivq_*`. 2 ×
//!   i64 / 4 × f32 per register; one register per iteration. Per ARM
//!   Cortex-A78 Software Optimization Guide §3.18 NEON 128-bit eq/cmp
//!   issue width is 2/cycle, so the bench expects ≥ 2× over scalar at
//!   the i64 path on Apple Silicon (M1/M2/M3 microarchitectures
//!   inherit the wide-NEON dispatch).
//! - **Scalar fallback**: portable Rust loop. Always available;
//!   exercised when neither AVX2 nor NEON is detected at runtime AND on
//!   targets that compile out the per-arch SIMD paths.
//!
//! # Safety discipline
//!
//! Every `unsafe { … }` block in this module + its sub-modules carries
//! a `// SAFETY:` comment naming the three preconditions:
//!
//! 1. **Target feature precondition**: the arch-specific intrinsics
//!    require the corresponding `target_feature` (AVX2 / NEON) be
//!    available at the call site. The runtime detection at the entry
//!    point is the load-bearing gate; the inner intrinsics inherit the
//!    gate via `#[target_feature(enable = "…")]` on the helper fn.
//! 2. **Alignment invariant**: AVX2 `_mm256_loadu_*` / NEON `vld1q_*`
//!    are documented as "unaligned-load tolerant". The slice's address
//!    is whatever Rust's allocator provides; we use the unaligned-load
//!    intrinsic variant exclusively to avoid alignment faults.
//! 3. **Length invariant**: the bulk loop processes
//!    `len - (len % LANES_PER_ITER)` elements; the trailing
//!    `len % LANES_PER_ITER` elements run through the scalar tail loop.
//!    The bulk loop's loop counter is bounded by the slice length; the
//!    intrinsics never read past the slice's end.
//!
//! # ADR provenance
//!
//! - **ADR-038 amendment-02 §M4.f** — primary M4-64b slice cite.
//! - **ADR-038 amendment-03 §Structural-1** — split out from M4-64
//!   bundled SIMD; ≥1.5× speedup gate vs scalar baseline.
//! - **ADR-038 amendment-03 §TIER-2-b** — 3VL NULL semantics; the
//!   FilterOp SIMD path preserves them.
//! - **Unsafe-code discipline** — every `unsafe` block carries a
//!   `// SAFETY:` comment naming the invariant.
//! - **Performance discipline** — every performance-sensitive helper carries
//!   a latency budget comment.

pub mod expand;
pub mod filter;
pub mod rrf;

/// Runtime-detected SIMD backend selection.
///
/// The variant set is closed at v1.0-alpha; future targets (e.g.,
/// AVX-512, RISC-V V) land alongside their respective per-arch gate.
/// Carries no payload — the per-arch helpers select intrinsics
/// internally based on this enum.
///
/// # Why not `#[non_exhaustive]`?
///
/// Under the code-quality policy "Error enum exhaustiveness" the convention is for
/// public *Error* enums; this is not an error type. The enum is
/// deliberately exhaustive: every match site must consider every
/// supported backend. Adding a backend variant in a future amendment
/// is a deliberate breaking change at the match-arm boundary, which
/// is the correct review surface for SIMD-backend additions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    /// Compile-time x86_64 + runtime AVX2 detected.
    X86Avx2,
    /// Compile-time aarch64 + runtime NEON detected.
    AArch64Neon,
    /// Scalar fallback. Always available.
    Scalar,
}

impl SimdBackend {
    /// Detect the best-available backend for the current execution.
    ///
    /// # Cost
    ///
    /// `is_*_feature_detected!` macros use cached CPUID reads so this
    /// is effectively a single-cycle conditional once the cache is
    /// warm. Operators MAY call this once per construction and cache
    /// the result (the value cannot change for the lifetime of the
    /// process).
    #[must_use]
    pub fn detect() -> Self {
        // x86_64 + AVX2 — most-common server arch.
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Self::X86Avx2;
            }
        }
        // AArch64 + NEON — Apple Silicon / ARM cloud. NEON is mandatory
        // on AArch64 per the ARMv8-A spec but the runtime check is
        // preserved for parity with the x86_64 path + future BTI-mode
        // gating.
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Self::AArch64Neon;
            }
        }
        Self::Scalar
    }

    /// Human-readable label for telemetry / EXPLAIN annotations.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::X86Avx2 => "x86_64+AVX2",
            Self::AArch64Neon => "aarch64+NEON",
            Self::Scalar => "scalar",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_a_backend_for_every_target() {
        // Pin: detection MUST always succeed (scalar fallback is the
        // safety net). A future gate that returns `Option<SimdBackend>`
        // would force every operator to handle the None case; the
        // current shape avoids that ergonomic tax.
        let _ = SimdBackend::detect();
    }

    #[test]
    fn label_renders_a_stable_string_for_telemetry() {
        // Telemetry consumers (EXPLAIN annotation, tracing fields)
        // grep on these strings; pinning prevents an accidental rename
        // from breaking downstream log scrapers.
        assert_eq!(SimdBackend::X86Avx2.label(), "x86_64+AVX2");
        assert_eq!(SimdBackend::AArch64Neon.label(), "aarch64+NEON");
        assert_eq!(SimdBackend::Scalar.label(), "scalar");
    }

    #[test]
    fn backend_is_copy_for_cheap_threading() {
        // Compile-time pin: SimdBackend MUST be Copy so the operator
        // can cache it without ownership ceremony, and so a parallel
        // executor at M4-64c can pass it to worker threads cheaply.
        fn assert_copy<T: Copy>() {}
        assert_copy::<SimdBackend>();
    }
}
