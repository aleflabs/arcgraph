//! Shared rank-fusion algorithms.
//!
//! This module is the single implementation boundary for fusion used by
//! both ArcQL execution and transport-facing search adapters. Keeping the
//! algorithm here prevents `graph.search` and `RANK BY HYBRID` from
//! acquiring different precision, tie-breaking, or rank conventions.

use std::collections::HashMap;

use arcgraph_core::NodeId;

use crate::executor::substrate::RankedHit;
use crate::executor::value::NodeView;

/// Reciprocal-Rank Fusion per Cormack SIGIR 2009.
///
/// `lists[i]` is the i-th retriever's ranked output (rank 1 = first).
/// For each node, the fused score is `Σ_i 1 / (k + rank_i)` over the
/// retrievers that produced that node.
///
/// Results are sorted by fused score descending, with `NodeId` ascending
/// as the deterministic tie-break. The returned [`RankedHit::score`] is
/// the fused score; input substrate scores affect rank order only.
#[must_use]
pub fn rrf_fuse(lists: &[Vec<RankedHit>], k: u64) -> Vec<RankedHit> {
    let k = k as f64;
    let mut scores: HashMap<NodeId, (f64, NodeView)> = HashMap::new();
    for hits in lists {
        for (rank0, hit) in hits.iter().enumerate() {
            let contribution = 1.0 / (k + (rank0 + 1) as f64);
            let entry = scores
                .entry(hit.node.id)
                .or_insert_with(|| (0.0, hit.node.clone()));
            entry.0 += contribution;
        }
    }

    let mut out: Vec<RankedHit> = scores
        .into_values()
        .map(|(score, node)| RankedHit { node, score })
        .collect();
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.node.id.raw().cmp(&b.node.id.raw()))
    });
    out
}

#[cfg(test)]
mod tests {
    use arcgraph_core::NodeId;

    use super::*;

    fn hit(id: u64) -> RankedHit {
        RankedHit {
            node: NodeView::new(NodeId::new(id), None),
            score: 0.0,
        }
    }

    #[test]
    fn opposed_lists_are_fused_with_exact_scores() {
        let fused = rrf_fuse(
            &[vec![hit(1), hit(2), hit(3)], vec![hit(2), hit(3), hit(1)]],
            60,
        );

        let ids_and_scores: Vec<(u64, f64)> = fused
            .into_iter()
            .map(|hit| (hit.node.id.raw(), hit.score))
            .collect();
        assert_eq!(
            ids_and_scores,
            vec![
                (2, 1.0 / 62.0 + 1.0 / 61.0),
                (1, 1.0 / 61.0 + 1.0 / 63.0),
                (3, 1.0 / 63.0 + 1.0 / 62.0),
            ]
        );
    }
}
