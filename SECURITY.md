# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately, not in public issues or pull requests.

- Preferred: GitHub's **Report a vulnerability** button under the *Security* tab of this
  repository (private vulnerability reporting).
- Or e-mail **security@dimensionlab.org**.

Include the affected version or commit, a description of the impact, and steps to reproduce
(a minimal input file is ideal; do not send proprietary simulation data). You will get an
acknowledgement within three working days and a first assessment within ten. We coordinate
disclosure with you and credit you in the advisory unless you prefer otherwise.

## Scope

Code in this repository. Typical concerns for this project: parsers for untrusted
simulation and mesh files (crafted input causing memory exhaustion, path traversal on
extraction, arbitrary code execution through deserialisation), handling of credentials for
dataset sources, and the integrity of published packages.

Out of scope: vulnerabilities in third-party solvers or their file formats themselves, and
the commercial Frontier platform (report those to the same address; they are handled under
your support agreement).

## Supported versions

The latest minor release receives security fixes. Before 1.0 that means the newest `0.x`.

## Supply chain

Releases are built in CI from tagged commits. Dependency licences are checked on every
change (`cargo deny`, licence allow-list for Python dependencies).
