# `arcgraph-community`

Leiden-family community indexing for the ArcGraph `v0.1.0-beta` bare database
engine.

## Responsibility

- `GveLeiden` computes a static hierarchy from an in-memory graph.
- `LeidenIncremental` applies edge updates to an existing result.
- `BTreeMembershipIndex` stores forward and reverse tenant-local membership
  mappings.
- `CommunityIndexProvider` creates a tenant's index handle.
- `CommunityRefreshScheduler` runs bounded background refreshes and reports
  health through its observer interface.

The crate does not persist graph records, parse ArcQL, or expose a network
endpoint. Persistence and tenant routing belong to `arcgraph-storage`; query
and protocol surfaces consume the index through adapters. The bare server
does not expose a separate community-analysis MCP tool.
