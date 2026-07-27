# Cold backup

`arcgraph backup create` is the only backup mode in `v0.1.0-beta`. Stop the
server first. The command takes the same exclusive `LOCK` as `serve`, so it
fails if the source is live.

```bash
target/debug/arcgraph backup create \
  --data ./arcgraph-data \
  --dest ./arcgraph-backup
```

The destination must be absent or empty. A backup is a plain directory with
`BACKUP_MANIFEST.json` plus the durable allowlist: the data-directory version
and format manifests when present, `pages.db` when present, and files under
`wal/`. Runtime `LOCK` state and unrecognized files are not copied.

The manifest records the creating version and every copied file's size and
SHA-256. Creation hashes the copied bytes and fsyncs the files and
directories.

There is no hot, incremental, per-tenant, or remote-object-store backup mode
in this distribution. Retention and off-site copying are deployment policy,
not ArcGraph commands.

For encrypted WALs, preserve the external KEK separately. The wrapped DEK is
stored under `wal/` and therefore travels with the backup; the KEK does not.
See [`encryption-at-rest-kek-runbook.md`](encryption-at-rest-kek-runbook.md).
