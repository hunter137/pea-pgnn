# Security policy

## Supported versions

Security fixes are provided for the latest `0.1.x` release line. Earlier alpha
releases may receive a fix only when the change can be applied safely.

## Reporting a vulnerability

Please do **not** disclose a suspected vulnerability in a public issue. Use the
repository's private vulnerability-reporting or draft-security-advisory route
when available:

<https://github.com/hunter137/pea-pgnn/security/advisories/new>

If that route is unavailable, contact the maintainer through the
[`hunter137` GitHub profile](https://github.com/hunter137) and request a private
channel. Include the affected version or commit, impact, reproduction steps,
and any suggested mitigation. Please allow time to confirm the report and
prepare a coordinated fix before public disclosure.

For ordinary bugs that do not expose data, execute code unexpectedly, bypass a
security boundary, or compromise package integrity, use the public issue
tracker instead.

## Checkpoint safety

PEA-PGNN checkpoints use PyTorch serialization. Load checkpoint files only
when you trust their source and integrity. Do not accept arbitrary `.pt`,
`.pth`, `.ckpt`, pickle, or model files from untrusted users. The package uses
PyTorch's restricted `weights_only` loading mode when the installed PyTorch
version supports it, but this is not a substitute for provenance controls.

For published checkpoints, provide a cryptographic checksum, package version,
checkpoint format version, model configuration, and a trusted download source.

