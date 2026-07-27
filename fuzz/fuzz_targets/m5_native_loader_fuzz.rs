#![no_main]
//! INV-M5.16 native loader boundary. Any panic, abort, or unbounded allocation
//! in the same parser used by `arcgraph load --format native` is a finding.

use libfuzzer_sys::fuzz_target;

fn exceeds_json_depth_cap(bytes: &[u8]) -> bool {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                if depth > arcgraph_cli::m5_load::MAX_NATIVE_RECURSION {
                    return true;
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    false
}

fuzz_target!(|data: &[u8]| {
    let result = arcgraph_cli::m5_load::fuzz_native_record_boundary(data);
    assert!(
        result.is_err() || !exceeds_json_depth_cap(data),
        "production native parser accepted an over-depth record"
    );
});
