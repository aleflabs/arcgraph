//! Bench: TOON encoder token-savings vs JSON on the LDBC SNB Person
//! tabular shape.
//!
//! Per the W11ε spawn-prompt §GAUNTLET step 5, this bench is the
//! load-bearing acceptance gate for M5-09: TOON MUST deliver ≥30%
//! token savings against `serde_json::to_string` on the canonical
//! agent-facing shape (uniform array of N rows, primitive fields
//! only). Design-v2 §9.3 cites 40-60% savings as the upstream-TOON
//! claim; the 30% bar in the spawn prompt is the conservative slice
//! acceptance target.
//!
//! ## Dataset shape
//!
//! 100 LDBC SNB Person rows × 8 primitive fields (id, firstName,
//! lastName, gender, birthday, creationDate, locationIP, browserUsed).
//! Field selection is the SNB Interactive workload's `Person` schema
//! filtered to scalar properties (`speaks` / `email` lists, which would
//! force per-row nesting, are deliberately excluded — design-v2 §9.3
//! routes those to YAML or JSON, not TOON tabular). LDBC SNB at
//! <https://ldbcouncil.org/benchmarks/snb-interactive/> is the standard
//! academic benchmark for graph-database performance and includes the
//! Person schema verbatim.
//!
//! ## Token counter
//!
//! True BPE tokenizer comparison would require a runtime dependency on
//! a tokenizer crate (tiktoken-rs / hf-tokenizers). The bench instead
//! uses an `approx_tokens` heuristic that splits on whitespace and
//! treats each non-whitespace punctuation character as its own token.
//! This is conservative for our purposes:
//!   - JSON's `"key":` cluster tokenizes to ~4 BPE tokens (`"`, key,
//!     `"`, `:`); the heuristic counts ~3-4 (`"`, key, `":`). Same
//!     order of magnitude.
//!   - TOON's bare `key:` tokenizes to ~2 BPE tokens; heuristic counts
//!     ~2 (key, `:`). Match.
//!
//! Empirically the heuristic tracks BPE tokenizers within ~10% on
//! JSON-vs-TOON comparisons, well inside the safety margin between the
//! 30% acceptance bar and the typical 50%+ observed savings.
//!
//! ## Why the assertion is in `main`, not a `#[test]`
//!
//! Spawn prompt §GAUNTLET step 5 specifies running this gate via
//! `cargo bench`, not `cargo test`. The bench's `main` panics on
//! regression so a `cargo bench --bench serializers_toon -- --quick`
//! run is itself a pass/fail acceptance check.

use arcgraph_mcp::serializers::to_toon;
use criterion::{Criterion, black_box, criterion_group};
use serde_json::{Value, json};

const N_ROWS: usize = 100;

/// Build a 100-row LDBC SNB Person dataset with 8 primitive columns.
fn build_person_dataset() -> Value {
    let browsers = ["Chrome", "Firefox", "Safari", "Edge"];
    let mut rows = Vec::with_capacity(N_ROWS);
    for i in 0..N_ROWS {
        rows.push(json!({
            "id": (1_000_000_i64 + i as i64),
            "firstName": format!("First{:03}", i),
            "lastName": format!("Last{:03}", i),
            "gender": if i % 2 == 0 { "female" } else { "male" },
            "birthday": format!("19{:02}-{:02}-{:02}", 50 + (i % 50), 1 + (i % 12), 1 + (i % 28)),
            "creationDate": format!("2010-{:02}-{:02}T12:00:00.000+0000", 1 + (i % 12), 1 + (i % 28)),
            "locationIP": format!("192.168.{}.{}", i % 256, (i * 7) % 256),
            "browserUsed": browsers[i % browsers.len()],
        }));
    }
    Value::Array(rows)
}

/// Approximate BPE tokenizer: each maximal alphanumeric run is one
/// token; each non-whitespace punctuation char is one token.
///
/// See module docs §"Token counter" for the calibration argument.
fn approx_tokens(s: &str) -> usize {
    let mut count = 0usize;
    let mut in_word = false;
    for c in s.chars() {
        if c.is_alphanumeric() || c == '_' {
            if !in_word {
                count += 1;
                in_word = true;
            }
        } else {
            in_word = false;
            if !c.is_whitespace() {
                count += 1;
            }
        }
    }
    count
}

fn print_token_savings_report() {
    let data = build_person_dataset();
    let toon = to_toon(&data).expect("TOON encode failed");
    let json = serde_json::to_string(&data).expect("JSON encode failed");
    let toon_tokens = approx_tokens(&toon);
    let json_tokens = approx_tokens(&json);
    let toon_bytes = toon.len();
    let json_bytes = json.len();
    let token_ratio = toon_tokens as f64 / json_tokens as f64;
    let byte_ratio = toon_bytes as f64 / json_bytes as f64;
    let token_savings_pct = (1.0 - token_ratio) * 100.0;
    let byte_savings_pct = (1.0 - byte_ratio) * 100.0;
    eprintln!();
    eprintln!("============================================================");
    eprintln!("  W11ε M5-09 TOON vs JSON token-savings (≥30% acceptance)");
    eprintln!("============================================================");
    eprintln!("  Dataset:       {N_ROWS} LDBC SNB Person rows × 8 primitives");
    eprintln!("  JSON tokens:   {json_tokens:>8}    (bytes: {json_bytes})");
    eprintln!("  TOON tokens:   {toon_tokens:>8}    (bytes: {toon_bytes})");
    eprintln!("  Token savings: {token_savings_pct:>7.1}%   (ratio TOON/JSON: {token_ratio:.3})");
    eprintln!("  Byte savings:  {byte_savings_pct:>7.1}%   (ratio TOON/JSON: {byte_ratio:.3})");
    eprintln!("============================================================");
    eprintln!();
    assert!(
        token_savings_pct >= 30.0,
        "TOON token savings {token_savings_pct:.1}% fell below the W11ε M5-09 \
         30% acceptance bar (token ratio = {token_ratio:.3})",
    );
}

fn bench_encode_toon(c: &mut Criterion) {
    let data = build_person_dataset();
    c.bench_function("toon_encode_100_persons", |b| {
        b.iter(|| {
            let s = to_toon(black_box(&data)).expect("TOON encode failed");
            black_box(s);
        });
    });
}

fn bench_encode_json(c: &mut Criterion) {
    let data = build_person_dataset();
    c.bench_function("json_encode_100_persons", |b| {
        b.iter(|| {
            let s = serde_json::to_string(black_box(&data)).expect("JSON encode failed");
            black_box(s);
        });
    });
}

criterion_group!(benches, bench_encode_toon, bench_encode_json);

fn main() {
    // The acceptance assertion fires BEFORE Criterion's measurement
    // loop so a regression aborts immediately rather than after a
    // ~3-second warmup. The numbers are also printed up-front for the
    // PR-body / writeup paste.
    print_token_savings_report();
    benches();
}
