# Single-node disaster recovery

The bare engine recovers one durable process; it has no replica election or
cluster failover.

1. Obtain a complete cold backup directory and, for an encrypted WAL, its
   external KEK.
2. Use the verified procedure in [`restore.md`](restore.md) with a fresh
   target.
3. Start `arcgraph serve` against that target using the same transport and
   encryption settings as the source.
4. Run `target/debug/arcgraph check --data ./arcgraph-restored` while the
   server is stopped, then run application-level count and property assertions
   after startup.

Recovery point is the end of the selected cold backup. Files written after
that backup are not reconstructed by this distribution.
