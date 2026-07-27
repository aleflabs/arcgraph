<!--
  Every PR description must include these five sections. CI gates on presence.
  See CONTRIBUTING.md for the submission rules.
-->

## What

<!-- One sentence describing the change. -->

## Why

<!--
  Link the ADR, design-v2 section, PRD requirement, or roadmap task that
  motivates this. e.g. "Implements roadmap M1-10, design-v2 §3.4, ADR-001."
-->

## How

<!-- 2–3 bullets on approach. -->

-
-
-

## Tests

<!--
  What you ran. What numbers you saw. Paste the green output of:
    cargo test --workspace --all-features
    cargo clippy --workspace --all-targets --all-features -- -D warnings
  For perf-sensitive work, paste the relevant Criterion summary.
-->

## Risks

<!--
  What could break. What you are NOT testing. Which documented design
  constraint applies and how you satisfied it.
-->

## Checklist

- [ ] Conventional Commits style (`feat:`, `fix:`, `perf:`, etc.)
- [ ] `cargo +nightly fmt --all --check` clean
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [ ] `cargo test --workspace --all-features` green
- [ ] If new dep: Apache 2.0 or MIT-compatible license confirmed via `cargo deny check`
- [ ] If `unsafe`: each block has a `// SAFETY:` comment
- [ ] No `unwrap()` outside `#[cfg(test)]`
- [ ] If perf-sensitive: budget comment + Criterion number in this PR
- [ ] Stays inside one bounded context unless the PR is explicitly cross-crate glue
