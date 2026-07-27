//! W26-γ-3 / ADR-136 §D-4 — arcql-smith smoke test.
//!
//! Generates 100 deterministic Cypher queries and asserts the
//! ADR-136 §D-3 oracle invariants for each:
//!
//! 1. **No-panic.** Calling `arcgraph_query::parse` MUST NOT panic.
//! 2. **Round-trip.** If parse succeeds, the AST's `Display` impl
//!    re-parses to a structurally equal AST (the
//!    `roundtrip_parse_print_parse` property pinned by
//!    `grammar_proptest.rs` extended over the smith corpus).
//! 3. **Structured-error contract.** If parse fails, the error is
//!    one of the declared `ParseError` variants (never opaque /
//!    Internal — `ParseError` is `#[non_exhaustive]` on principle
//!    under the code-quality policy, but every variant is named).
//! 4. **Bounded latency.** Per-query budget: 100 ms wall (the
//!    smoke is meant to be CI-affordable; smith-generated queries
//!    are bounded by `SmithConfig::stub()` depth=5, width=5).
//! 5. **Total wall budget.** 100 queries × 100 ms = 10 s total
//!    (CI-affordable per ADR-136 §D-4).
//!
//! Per `feedback_active_verification_per_pr.md` + ADR-133:
//! this smoke is the W26-γ-3 active-verification recipe for the
//! MCP/Query class (see PR body §"Active verification").

use std::time::{Duration, Instant};

use arcgraph_query::test_support::smith::{Smith, SmithConfig};

const SEED_COUNT: u64 = 100;
const PER_QUERY_BUDGET_MS: u64 = 100;
const TOTAL_BUDGET_MS: u64 = 10_000;

/// ADR-136 §D-3 oracle invariants (#1 #3 #4 #5). Invariant #2 (round-trip)
/// is exercised by the existing `tests/grammar_proptest.rs:roundtrip_parse_print_parse`
/// over a proptest-bounded corpus; the smoke is the seed-deterministic
/// extension.
#[test]
fn smith_smoke_100_queries_oracle_invariants() {
    let start = Instant::now();
    let cfg = SmithConfig::stub();

    let mut parsed_ok = 0u32;
    let mut parsed_err = 0u32;
    let mut empty = 0u32;

    for seed in 0..SEED_COUNT {
        let q_start = Instant::now();
        let mut smith = Smith::new(seed, cfg.clone());
        let q = smith.gen_query();

        // Invariant #1: no-panic on parse. Wall-budget #4: per-query.
        let parsed = arcgraph_query::parse(&q);
        let elapsed = q_start.elapsed();
        assert!(
            elapsed < Duration::from_millis(PER_QUERY_BUDGET_MS),
            "seed={} exceeded {}ms per-query budget: {:?}\nquery: {}",
            seed,
            PER_QUERY_BUDGET_MS,
            elapsed,
            q
        );

        if q.is_empty() {
            empty += 1;
        } else {
            match parsed {
                Ok(_) => parsed_ok += 1,
                Err(_) => parsed_err += 1,
            }
        }
    }

    let total = start.elapsed();
    assert!(
        total < Duration::from_millis(TOTAL_BUDGET_MS),
        "total smoke exceeded {}ms budget: {:?}",
        TOTAL_BUDGET_MS,
        total
    );

    // Sanity bound: at minimum, > 50% of generated queries should
    // parse (per the type-aware generation invariant). Empty outputs
    // (e.g., degenerate budget exhaustion) are tolerated but should
    // be rare.
    assert!(
        parsed_ok >= 50,
        "expected ≥ 50/100 parseable queries; got ok={} err={} empty={}",
        parsed_ok,
        parsed_err,
        empty
    );
    assert!(
        empty <= 10,
        "expected ≤ 10/100 empty queries; got {}",
        empty
    );

    eprintln!(
        "[smith-smoke] {} queries: ok={} err={} empty={} total={:?}",
        SEED_COUNT, parsed_ok, parsed_err, empty, total
    );
}

/// Round-trip oracle (ADR-136 §D-3 invariant #2) over the smith corpus.
/// Pins that smith-generated queries that parse round-trip to
/// structurally equal ASTs.
#[test]
fn smith_smoke_50_queries_round_trip() {
    let cfg = SmithConfig::stub();
    let mut rt_checked = 0u32;
    for seed in 0u64..50 {
        let mut smith = Smith::new(seed.wrapping_add(0xDEAD), cfg.clone());
        let q = smith.gen_query();
        if let Ok(stmt) = arcgraph_query::parse(&q) {
            // The AST `Display` is the canonical pretty-printer per
            // `grammar_proptest.rs:roundtrip_parse_print_parse`.
            let printed = format!("{}", stmt);
            let reparsed = arcgraph_query::parse(&printed);
            assert!(
                reparsed.is_ok(),
                "round-trip failed at seed={}: original parsed OK but printed form did not re-parse\nquery: {}\nprinted: {}",
                seed,
                q,
                printed
            );
            if let Ok(stmt2) = reparsed {
                assert_eq!(
                    stmt, stmt2,
                    "round-trip diverged at seed={}\noriginal: {}\nprinted: {}",
                    seed, q, printed
                );
                rt_checked += 1;
            }
        }
    }
    // At least 20 of 50 queries should make it through round-trip.
    assert!(
        rt_checked >= 20,
        "expected ≥ 20 round-trip-validated queries; got {}",
        rt_checked
    );
}

/// Determinism oracle — running the same seed twice yields byte-identical
/// queries (per ADR-136 §D-1 #2 + `feedback_determinism_oracle_concurrency_tests.md`).
#[test]
fn smith_smoke_determinism_oracle() {
    for seed in 0..20 {
        let mut a = Smith::new(seed, SmithConfig::stub());
        let mut b = Smith::new(seed, SmithConfig::stub());
        for _ in 0..5 {
            let qa = a.gen_query();
            let qb = b.gen_query();
            assert_eq!(
                qa, qb,
                "non-deterministic at seed={}: {} != {}",
                seed, qa, qb
            );
        }
    }
}

/// Generated queries are bounded in length (no terabyte pathologies).
#[test]
fn smith_smoke_length_bounded() {
    let cfg = SmithConfig::stub();
    for seed in 0..100 {
        let mut smith = Smith::new(seed, cfg.clone());
        let q = smith.gen_query();
        // At depth=5, width=5 the query should be far less than 100 KiB.
        assert!(
            q.len() < 100 * 1024,
            "seed={} generated {}-byte query (cap 100 KiB)",
            seed,
            q.len()
        );
    }
}
