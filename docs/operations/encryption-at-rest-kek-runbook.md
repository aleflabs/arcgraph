# WAL encryption and KEK custody

`v0.1.0-beta` can encrypt durable WAL payloads with AES-256-GCM when
`--wal-encryption` is set. It does not encrypt `pages.db`, index files, or
backup manifests. Do not describe this flag as full data-at-rest encryption.

The server creates a data-encryption key (DEK), wraps it under KEK version 1,
and stores only the wrapped DEK in `wal/wal.dek`. Startup fails closed if the
KEK cannot be resolved.

Two provider values exist:

- `os-keyring` is the default and requires a binary built with the
  `os-keyring` feature;
- `env` reads a 32-byte KEK as 64 hexadecimal characters from
  `ARCGRAPH_SECRET_ARCGRAPH_DOT_WAL_DOT_ENCRYPTION_KEY_DOT_V1` and emits an
  unsafe-for-production warning.

The environment provider is a development aid. It does not persist or rotate
keys. Production keyring provisioning is external to the ArcGraph CLI.

Preserve the KEK outside the data directory and backup. The wrapped DEK is
included by the cold-backup WAL allowlist, but it is unusable without its KEK.
Deleting or rotating away that KEK makes the encrypted WAL and its backups
unrecoverable.
