//! M5-D3 FIX 1 (P1, LOAD-BEARING — #1518 skeptic review) — adopted
//! permanent gate for full-input resume identity.
//!
//! Originated as a skeptic probe during the M5-D3 (PR #1518) 4-skeptic
//! gate review: `InputFingerprint` used to hash only `(len, CRC of the
//! first 1 MiB)`. A crash after s2 (dup check durable), followed by an
//! input swap that PRESERVES length and the first 1 MiB but plants a
//! duplicate in the tail, then a rerun: with the head-only fingerprint,
//! the rerun resumed the stale s1/s2 manifests, the planted duplicate was
//! never seen by the phase-1 dup check, and the loader "completed" —
//! serving the OLD input's content and violating the duplicate hard-error
//! contract for the NEW input it was pointed at (violates
//! `m5_parallel.rs` "durable stages belong to THIS input or none at all").
//!
//! FIX: `InputFingerprint` now carries a full-stream CRC32C over the
//! ENTIRE input (see `m5_parallel.rs::InputFingerprint::of`), computed in
//! one sequential pass, and the resume identity check
//! (`run_pipeline`) compares the FULL fingerprint — any mismatch discards
//! the stale plan and rebuilds from scratch. This gate is now GREEN by
//! construction; the RED-on-revert is: reinstate the 1-MiB-head-only
//! fingerprint (drop `full_crc` from the comparison) and this test goes
//! red (the loader completes over the swapped input instead of refusing
//! resume / detecting the duplicate).
//!
//! Requires `--features fault-injection` (uses `ARCGRAPH_M5_CRASH_AFTER_STAGE`).

use std::fs;
use std::path::Path;

use arcgraph_cli::m5_load::{
    LoadFormat, LoadLimits, LoadOutcome, M5_RSS_CAP_BYTES, load_data_dir_with_limits,
};
use arcgraph_core::TenantId;

const LOAD_TENANT: TenantId = TenantId::new(83);

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 0xf)] as char);
    }
    out
}

fn node_line(external: &[u8]) -> String {
    format!(
        "{{\"kind\":\"node\",\"external_id\":\"{}\",\"label_or_type\":0,\
         \"float_bits\":\"0000000000000000\",\"opaque\":\"\"}}\n",
        hex(external),
    )
}

fn limits(workers: usize) -> LoadLimits {
    LoadLimits {
        workers: Some(workers),
        sort_memory_bytes: 64 * 1024,
        rss_cap_bytes: M5_RSS_CAP_BYTES,
        rss_sample_every_ms: 50,
        max_disk_bytes: None,
    }
}

/// ~13,001 lines x ~110 B ~= 1.4 MB; the tail line lands well beyond the
/// 1 MiB fingerprint head window. `tail_external` MUST be 7 bytes so A/B
/// swaps are byte-length-identical.
fn write_fixture(path: &Path, tail_external: &[u8]) -> u64 {
    assert_eq!(tail_external.len(), 7, "tail id must be 7 bytes");
    let mut body = String::new();
    for index in 0..13_000_u64 {
        body.push_str(&node_line(format!("n{index:06}").as_bytes()));
    }
    body.push_str(&node_line(tail_external));
    fs::write(path, &body).expect("write fixture");
    body.len() as u64
}

/// THE load-bearing proof (FIX 1): a stale-input resume must never serve
/// bytes that are not exactly the current input. Same length + same first
/// 1 MiB, different tail bytes — the full-stream CRC catches it.
#[test]
fn stale_input_resume_misses_tail_planted_duplicate() {
    let dir = tempfile::tempdir().expect("dir");
    let input = dir.path().join("swap.jsonl");
    let root = tempfile::tempdir().expect("root");

    // Input A: all-unique; 7-char tail id.
    let len_a = write_fixture(&input, b"zz99999");
    assert!(
        len_a > (1 << 20) + 300_000,
        "fixture must exceed the 1 MiB head window by a margin (got {len_a})"
    );

    // Crash right after the s2 (dup-check) manifest becomes durable.
    // SAFETY: single-test binary; no concurrent env readers.
    unsafe { std::env::set_var("ARCGRAPH_M5_CRASH_AFTER_STAGE", "s2-phase1-counts") };
    let crash = load_data_dir_with_limits(
        &input,
        LoadFormat::Native,
        root.path(),
        LOAD_TENANT,
        limits(4),
    )
    .expect_err("injected s2 crash must surface");
    // SAFETY: as above.
    unsafe { std::env::remove_var("ARCGRAPH_M5_CRASH_AFTER_STAGE") };
    assert!(
        format!("{crash:#}").contains("injected crash after stage"),
        "unexpected crash error: {crash:#}"
    );

    // Input B: SAME length, SAME first 1 MiB — but the tail line now
    // duplicates existing filler node n000042.
    let len_b = write_fixture(&input, b"n000042");
    assert_eq!(len_a, len_b, "A/B must be byte-length-identical");

    let outcome = load_data_dir_with_limits(
        &input,
        LoadFormat::Native,
        root.path(),
        LOAD_TENANT,
        limits(4),
    );
    match outcome {
        Err(error) => {
            let rendered = format!("{error:#}");
            assert!(
                rendered.contains("duplicate node external_id"),
                "refused, but not for the duplicate: {rendered}"
            );
            println!("OK: tail-planted duplicate detected despite the resume: {rendered}");
        }
        Ok(LoadOutcome::Loaded(report)) => {
            panic!(
                "DEFECT: loader COMPLETED over input B (tail duplicate n000042) by resuming \
                 stale stages {:?} from input A — the resume identity check failed to catch \
                 the swapped input; the dup hard error was skipped and the served store \
                 carries input A's content (records={})",
                report.resumed_stages, report.records
            );
        }
        Ok(other) => panic!("unexpected outcome: {other:?}"),
    }
}
