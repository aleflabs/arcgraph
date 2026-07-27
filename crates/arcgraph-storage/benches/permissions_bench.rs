//! ADR-212 §4 budget pin: per-candidate visibility check ≈ 20–60 ns;
//! a 10k-candidate filter must stay well under 1 ms (< 5 % of engine
//! read-path p95). Regressions > 10 % block merges under the testing strategy.

use std::collections::BTreeSet;

use arcgraph_core::NodeId;
use arcgraph_storage::permissions::{PUBLIC_PRINCIPAL, PermissionIndex};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

/// 1M docs across 1k interned classes; alice holds ~1/3 of classes.
fn build_index(docs: u64, classes: u32) -> PermissionIndex {
    let idx = PermissionIndex::new();
    for c in 0..classes {
        let mut grants = BTreeSet::new();
        match c % 3 {
            0 => {
                grants.insert("alice".to_owned());
                grants.insert(format!("team-{c}"));
            }
            1 => {
                grants.insert("bob".to_owned());
            }
            _ => {
                grants.insert(PUBLIC_PRINCIPAL.to_owned());
            }
        }
        // One representative doc per class first (interns the class)…
        idx.apply_doc_acl(NodeId::new(u64::from(c)), grants);
    }
    // …then spread the remaining docs round-robin over the classes by
    // re-applying identical grant sets (interning dedups).
    for d in u64::from(classes)..docs {
        let c = (d % u64::from(classes)) as u32;
        let mut grants = BTreeSet::new();
        match c % 3 {
            0 => {
                grants.insert("alice".to_owned());
                grants.insert(format!("team-{c}"));
            }
            1 => {
                grants.insert("bob".to_owned());
            }
            _ => {
                grants.insert(PUBLIC_PRINCIPAL.to_owned());
            }
        }
        idx.apply_doc_acl(NodeId::new(d), grants);
    }
    idx
}

fn bench_permissions(c: &mut Criterion) {
    let idx = build_index(100_000, 1_000);
    let alice = idx.effective("alice");

    c.bench_function("is_visible_single", |b| {
        b.iter(|| black_box(alice.is_visible(black_box(NodeId::new(54_321)))));
    });

    c.bench_function("filter_10k_candidates", |b| {
        b.iter(|| {
            let mut kept = 0u32;
            for d in 0..10_000u64 {
                if alice.is_visible(black_box(NodeId::new(d))) {
                    kept += 1;
                }
            }
            black_box(kept)
        });
    });

    c.bench_function("effective_cold_1k_classes", |b| {
        b.iter(|| {
            // Bump generation so each resolution is a cold rebuild.
            idx.apply_doc_acl(black_box(NodeId::new(7)), {
                let mut g = BTreeSet::new();
                g.insert("alice".to_owned());
                g
            });
            black_box(idx.effective("alice"))
        });
    });
}

criterion_group!(benches, bench_permissions);
criterion_main!(benches);
