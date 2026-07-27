# Data-directory upgrade

Stop every process using the data directory and preserve a cold backup before
an upgrade.

```bash
target/debug/arcgraph migrate upgrade-data-dir \
  --data-dir ./arcgraph-data
```

The command performs one supported offline generation transition at a time
and reports `AlreadyCurrent` when no transition is needed. It uses the
data-directory lock, writes into a new generation namespace, verifies the
result, and atomically changes `CURRENT`.

`serve` checks the on-disk `VERSION` stamp and fails closed on an incompatible
format. `--adopt-legacy-datadir` only stamps an existing *unstamped* directory
that the operator knows already uses the current format; it does not convert
an incompatible store.

Do not replace the binary and bypass a refusal by editing `VERSION` or
`CURRENT`.
