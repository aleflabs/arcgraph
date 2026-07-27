#![no_main]

// Placeholder fuzz target, per roadmap M0-09.
// Each parser / deserializer gets its own target under this directory.
// Run with nightly rust: `cargo +nightly fuzz run placeholder`.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = data.len();
});
