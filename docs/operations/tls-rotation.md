# HTTPS certificate rotation

The HTTPS MCP transport reads the certificate chain and matching private key
from the paths passed to `--tls-cert` and `--tls-key`. Replace both files
atomically at those paths, then send the server `SIGHUP`.

The reload path validates PEM parsing, key/certificate matching, certificate
validity, and the optional `--tls-hostname` before swapping the resolver. A
failed reload keeps the previous certificate active. New TLS connections use
the successfully reloaded material; established connections are unaffected.

Protect the private key with owner-only permissions. The loopback development
certificate procedure is in [`../transports.md`](../transports.md).

This applies to MCP over HTTPS only. The CLI exposes no Bolt certificate/key
flags, so the documented Bolt transport remains plaintext on loopback and
cannot be made routable by rotation.
