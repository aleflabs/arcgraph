# Verified restore

Restore only into a fresh data directory:

```bash
target/debug/arcgraph backup restore \
  --from ./arcgraph-backup \
  --data ./arcgraph-restored
```

Restore reads and version-checks `BACKUP_MANIFEST.json`, then verifies every
listed file's size and SHA-256 before touching the target. It refuses a target
that already contains `wal/`, `pages.db`, or `LOCK`, takes the target lock,
copies with fsync discipline, and exits.

Start the restored directory normally. Standard boot recovery replays its WAL;
there is no second restore-only recovery implementation. A corrupt or
incomplete backup fails before the target is populated.

If WAL encryption was enabled, make the same external KEK available before
starting the restored store. Missing key material is a fail-closed startup
error.
