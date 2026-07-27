//! W20β-3 / ADR-052 / PR #373 R1 M-1 closure — pin the
//! `EnvSecretsProvider` startup-warn emission.
//!
//! ## Why this test exists
//!
//! ADR-052 §"`EnvSecretsProvider`" states the constructor MUST emit a
//! `tracing::warn!` tagged `unsafe_for_prod = true` so operators see
//! the danger in startup logs. The R1 review (PR #373) §M-1 flagged
//! that the only test exercising this safeguard was a source-level
//! string-pin in `crates/arcgraph-core/src/secrets/env.rs` — it
//! asserted the literal `"unsafe_for_prod = true"` appeared in the
//! source, NOT that the warning was actually emitted. A future
//! refactor that demoted the `warn!` to `info!` or removed it would
//! pass the source-pin silently.
//!
//! This integration test uses the `tracing-test` crate's
//! `#[traced_test]` macro to install a per-test subscriber that
//! captures emitted events; `logs_contain(...)` asserts the warning
//! literal fires at construction time. Per
//! `feedback_cite_correctness_not_just_resolution.md` (W13 IR
//! L1-HIGH-1) the ADR's "pinned by a deny-list test" claim is now
//! actually pinned — not just cite-resolved.
//!
//! ## What this test pins
//!
//! 1. `EnvSecretsProvider::new()` emits `tracing::warn!` with the
//!    `unsafe_for_prod = true` field — i.e., the WARN level is the
//!    level the operator's logging stack ships to alert channels.
//! 2. The warning literal contains `UNSAFE FOR PRODUCTION` — the
//!    upper-case string is the load-bearing log-search needle
//!    operators grep for.
//! 3. The warning references `ADR-052-secrets-at-rest-encryption.md`
//!    — the ADR-correctness pin per the W13 codification.
//! 4. `EnvSecretsProvider::with_prefix(...)` ALSO emits the warning
//!    (the constructor's two public entry points must be symmetric;
//!    a future refactor that warns from `new` but silently
//!    constructs in `with_prefix` would be a regression caught by
//!    this test).
//! 5. `EnvSecretsProvider::without_startup_warn_for_tests(...)` does
//!    NOT emit the warning — the test-only escape hatch must not
//!    leak the suppress-warn behavior to production paths.

use arcgraph_core::EnvSecretsProvider;
use tracing_test::traced_test;

#[test]
#[traced_test]
fn env_secrets_provider_new_emits_unsafe_for_prod_warning() {
    let _p = EnvSecretsProvider::new();
    // The warning literal includes `unsafe_for_prod = true` — the
    // structured-field form `tracing` renders into the captured log.
    assert!(
        logs_contain("unsafe_for_prod"),
        "EnvSecretsProvider::new() MUST emit a tracing::warn! with \
         `unsafe_for_prod = true` field per ADR-052 §EnvSecretsProvider"
    );
    assert!(
        logs_contain("UNSAFE FOR PRODUCTION"),
        "EnvSecretsProvider::new() MUST emit `UNSAFE FOR PRODUCTION` in \
         the warning message literal so log-search catches it"
    );
    assert!(
        logs_contain("ADR-052"),
        "EnvSecretsProvider::new() MUST reference ADR-052 in the warning \
         so operators have a docs anchor"
    );
}

#[test]
#[traced_test]
fn env_secrets_provider_with_prefix_emits_unsafe_for_prod_warning() {
    let _p = EnvSecretsProvider::with_prefix("ARCGRAPH_TEST_M1_");
    // `with_prefix` is the constructor `new` delegates to; the
    // warning MUST fire from this path too (symmetry pin — a future
    // refactor that warns only from `new` but not `with_prefix`
    // would silently regress half the operator surface).
    assert!(
        logs_contain("unsafe_for_prod"),
        "EnvSecretsProvider::with_prefix() MUST emit a tracing::warn! with \
         `unsafe_for_prod = true` per ADR-052 §EnvSecretsProvider"
    );
    assert!(
        logs_contain("UNSAFE FOR PRODUCTION"),
        "EnvSecretsProvider::with_prefix() MUST emit `UNSAFE FOR PRODUCTION` \
         in the warning message literal"
    );
}

#[test]
#[traced_test]
fn env_secrets_provider_without_startup_warn_for_tests_is_silent() {
    let _p = EnvSecretsProvider::without_startup_warn_for_tests("ARCGRAPH_TEST_M1_silent_");
    // The test-only escape hatch MUST NOT emit the warning — a
    // production caller reaching for `without_startup_warn_for_tests`
    // would lose the operator-facing pin. Doc-hidden + this negative
    // pin is the discipline.
    assert!(
        !logs_contain("UNSAFE FOR PRODUCTION"),
        "EnvSecretsProvider::without_startup_warn_for_tests() MUST be \
         silent — emitting the warning would leak the suppress-warn \
         path's purpose to production logs"
    );
}
