# ADR-011: Tenant isolation is foundational

- **Status:** Accepted for `v0.1.0-beta`

Tenant identity is part of storage, index, transaction, permission, and MCP
request keys. An index candidate never permits a cross-tenant read; it must be
hydrated through the same tenant-scoped storage path as a direct lookup.

Tenant 0 is reserved for system records. Public examples use tenant 1 or
higher. MCP requests supply tenant identity in their tool arguments, and HTTPS
also carries `x-arcgraph-tenant`; the values must agree. The documented Bolt
listener routes sessions to tenant 1 and uses the basic-auth username as the
read principal.

Principal ACLs narrow visibility inside a tenant. They never grant access to a
different tenant.
