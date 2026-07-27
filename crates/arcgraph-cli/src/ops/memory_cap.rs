//! #1291 — served-binary default per-tenant memory cap resolution.
//!
//! The M4-64a per-tenant memory budget
//! (`arcgraph_query::executor::MemoryBudget`, ADR-038 amendment-03
//! §Structural-1) shipped mechanically correct but DISABLED: the byte
//! cap defaulted to `None` and the served binary never called
//! `set_per_tenant_cap`, so the only guard for unbudgeted tenants was
//! the `UNCAPPED_RUNAWAY_GUARD_ROWS` row-count fallback (`1 << 32` ≈
//! 4.29 B rows — effectively unbounded → OOM under a heavy query).
//!
//! This module resolves the DEFAULT byte cap the served binaries
//! (`arcgraph serve` + `arcgraph-mcp-stdio`) apply to every query, so
//! unbudgeted tenants get a real ceiling out of the box.
//!
//! # Configuration
//!
//! - **Env override:** `ARCGRAPH_TENANT_MEMORY_CAP_BYTES=<u64>` sets
//!   the cap in bytes. `0` DISABLES the cap (explicit opt-out back to
//!   the pre-#1291 unbounded posture — escape hatch for single-tenant
//!   embedded-style deployments that prefer the row-count guard).
//! - **Unset / unparseable:** [`DEFAULT_PER_TENANT_MEMORY_CAP_BYTES`]
//!   (1 GiB). Unparseable values WARN and fall back to the default
//!   (fail-SAFE: a typo keeps the ceiling rather than silently
//!   removing it — the same posture `#[serde(deny_unknown_fields)]`
//!   codifies for config structs under the code-quality policy, applied to the
//!   env-var surface's value axis).
//!
//! The env-var (rather than a new CLI flag) matches the existing
//! served-binary config precedent (`ARCGRAPH_OTLP_*` in
//! `super::tracing_init`). M5-12's rate-limit config file is
//! the designated durable home for per-tenant overrides
//! (ADR-038 amendment-03 §TIER-2-a); this default slots underneath it
//! without a new config surface.
//!
//! # Why 1 GiB (back-of-envelope under the performance-budget discipline)
//!
//! The budget charges `estimate_row_bytes` per buffered row — a
//! typical 5-column row estimates at ~300 B (24 B Vec overhead + 5 ×
//! 56 B `Value` stack + heap). 1 GiB therefore admits ~3.5 M typical
//! rows buffered across a query's blocking operators — two orders of
//! magnitude above any legitimate v1.0-α served result set (the
//! `graph.raw_query` wire cap is max_rows ≤ thousands; LDBC SF-0.1
//! working sets are well below this), while bounding the runaway
//! class: the old 4.29 B-row guard admits ~1.3 TB of 300 B rows before
//! firing — no served host survives that. A cap this generous also
//! cannot mask a >10 % perf regression on the benchmark suite (the
//! budget's reserve path is a lookup + add + compare under an
//! uncontended mutex, ~30 ns/row, and only engages on blocking
//! operators that already pay an O(row) buffer copy).
//!
//! # Scope
//!
//! The cap is enforced PER QUERY (each served query mints a budget
//! with the cap set for its tenant): one query is bounded by the cap;
//! N concurrent queries from one tenant are bounded by N × cap.
//! Cross-query per-tenant aggregation (one process-wide budget shared
//! across in-flight queries) is the M5-12 config-surface follow-up —
//! it needs the per-tenant override vocabulary that config file owns.

/// Env var overriding the per-tenant memory cap in BYTES for the
/// served binaries. `0` disables the cap (explicit opt-out).
pub const ENV_TENANT_MEMORY_CAP_BYTES: &str = "ARCGRAPH_TENANT_MEMORY_CAP_BYTES";

/// Default per-tenant memory cap: 1 GiB. See the module docs for the
/// back-of-envelope defense of the number.
pub const DEFAULT_PER_TENANT_MEMORY_CAP_BYTES: u64 = 1 << 30;

/// Resolve the per-tenant memory cap the served binary applies to
/// every query. `Some(cap_bytes)` → wire via
/// `with_per_tenant_memory_cap(cap)`; `None` → operator explicitly
/// disabled the cap (`ARCGRAPH_TENANT_MEMORY_CAP_BYTES=0`).
#[must_use]
pub fn resolve_per_tenant_memory_cap() -> Option<u64> {
    resolve_from(std::env::var(ENV_TENANT_MEMORY_CAP_BYTES).ok().as_deref())
}

/// Pure resolution core (unit-testable without env mutation).
fn resolve_from(raw: Option<&str>) -> Option<u64> {
    match raw {
        None => Some(DEFAULT_PER_TENANT_MEMORY_CAP_BYTES),
        Some(s) => match s.trim().parse::<u64>() {
            // `0` = explicit opt-out (documented escape hatch).
            Ok(0) => None,
            Ok(cap) => Some(cap),
            Err(_) => {
                // Fail-SAFE: an unparseable override keeps the default
                // ceiling rather than silently removing it.
                tracing::warn!(
                    target: "arcgraph_cli::ops::memory_cap",
                    raw = s,
                    default_bytes = DEFAULT_PER_TENANT_MEMORY_CAP_BYTES,
                    "{ENV_TENANT_MEMORY_CAP_BYTES} is not a u64 byte count; \
                     falling back to the 1 GiB default (set `0` to disable)"
                );
                Some(DEFAULT_PER_TENANT_MEMORY_CAP_BYTES)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_env_resolves_to_1gib_default() {
        // #1291 — the served binary MUST get a real byte ceiling out
        // of the box (no operator action required).
        assert_eq!(
            resolve_from(None),
            Some(DEFAULT_PER_TENANT_MEMORY_CAP_BYTES)
        );
        assert_eq!(DEFAULT_PER_TENANT_MEMORY_CAP_BYTES, 1_073_741_824);
    }

    #[test]
    fn explicit_zero_disables_the_cap() {
        // Documented escape hatch back to the pre-#1291 posture.
        assert_eq!(resolve_from(Some("0")), None);
    }

    #[test]
    fn explicit_byte_count_overrides_the_default() {
        assert_eq!(resolve_from(Some("536870912")), Some(512 * 1024 * 1024));
        // Whitespace-tolerant (shell quoting artifacts).
        assert_eq!(resolve_from(Some(" 1024 ")), Some(1024));
    }

    #[test]
    fn unparseable_override_fails_safe_to_the_default() {
        // A typo keeps the ceiling rather than silently removing it.
        for bad in ["1GB", "-1", "lots", "", "1.5e9"] {
            assert_eq!(
                resolve_from(Some(bad)),
                Some(DEFAULT_PER_TENANT_MEMORY_CAP_BYTES),
                "raw override {bad:?} must fall back to the default"
            );
        }
    }
}
