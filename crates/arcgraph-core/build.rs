// Register the `arcgraph_sim` cfg flag for `cargo check-cfg` so
// production builds (no `--cfg arcgraph_sim`) do not warn about
// unexpected cfg names. Per ADR-135 D-3.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(arcgraph_sim)");
}
