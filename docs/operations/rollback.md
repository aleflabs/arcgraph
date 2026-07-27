# Binary rollback

A binary rollback is safe only when the older binary supports the data
directory's stamped format. Check compatibility before starting the older
binary.

If an upgrade published a newer generation, do not hand-edit `CURRENT`,
`VERSION`, or a generation manifest. Restore the pre-upgrade cold backup into
a fresh directory using [`restore.md`](restore.md), then start the matching
binary against that restored directory.

ArcGraph does not provide an in-place downgrade command. Query or schema
changes made after the backup are not present after this recovery method.
