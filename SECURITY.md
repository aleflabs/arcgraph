# Security Policy

## Supported versions

Security reports are accepted against the latest tagged release
(`v0.1.0-beta`) and `main`. Fixes ship on `main` and in the next tagged
release; there are no long-lived maintenance branches for this beta.

## Reporting a vulnerability

Please do **not** file a public GitHub issue for security problems.

Email the maintainer at `advaith.shesh@gmail.com` with:

- A description of the issue and the affected component (crate / protocol)
- Reproduction steps or a minimal proof of concept
- The commit SHA or release version you reproduced against
- Your disclosure timeline expectations

We aim to acknowledge within 3 business days and to ship a fix or a
mitigation within 30 days for severity HIGH / CRITICAL. Lower-severity
issues may be bundled into the next scheduled release.

## Scope

In scope:

- Memory safety or corruption bugs
- Authentication / authorization bypass (MCP tools, Bolt, HTTP)
- WAL / MVCC durability or isolation violations
- Data exfiltration via side channels

Out of scope for this beta:

- Denial of service via resource exhaustion (documented capacity limits)
- Issues requiring adversarial access to the host process

## Coordinated disclosure

We prefer private coordinated disclosure. If you have already publicly
disclosed, we will still investigate and ship a fix; we just cannot offer
embargo coordination retroactively.
