# Operations for `v0.1.0-beta`

These runbooks cover the single-process ArcGraph bare database engine. This
distribution has no cluster, replication, connector, or online-backup service.

| Procedure | Current support |
|---|---|
| [Cold backup](backup.md) | Copies the durable allowlist under the same exclusive data lock used by the server |
| [Verified restore](restore.md) | Verifies every manifest size and SHA-256 before writing a fresh target |
| [Disaster recovery](disaster-recovery.md) | Restores one durable process from a verified cold backup |
| [Native bulk load](bulk-load.md) | Offline newline-delimited native JSON into a virgin/resumable generation |
| [Upgrade](upgrade.md) | Offline data-directory generation upgrade; incompatible formats fail closed |
| [Rollback](rollback.md) | Binary rollback rules for stamped on-disk formats |
| [TLS rotation](tls-rotation.md) | HTTPS certificate/key reload for new connections |
| [WAL encryption](encryption-at-rest-kek-runbook.md) | Optional WAL encryption; page-store encryption is not provided |

Network startup and authentication are documented separately in
[`../transports.md`](../transports.md).

The default listeners are:

- admin liveness/readiness on `127.0.0.1:8090`;
- Prometheus metrics on `127.0.0.1:9090`.

An empty `--admin-http` or `--metrics-http` value disables that listener.
Non-loopback binds require the corresponding explicit allow flag.
