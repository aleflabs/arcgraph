# Security documentation

The authoritative reporting channel, supported version, and scope are in the
repository's root [`SECURITY.md`](../../SECURITY.md).

[`dep-audit.md`](dep-audit.md) describes the Rust dependency gates. Release
signing, SBOM, SLSA, compliance, and incident-response documents from the
wider product are not claims made by this bare database distribution.

The database trust boundary includes the Rust process, stored pages and WAL,
MCP and Bolt transports, authentication and read ACLs, and tenant isolation.
