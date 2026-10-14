# Contributing to frontier-connectors

Thanks for looking. This repository is one of the open-source parts of
[DimensionLab Frontier](https://dimensionlab.org), the platform for training and deploying
Large Physics Models. Bug reports, new connectors, dataset catalog entries, documentation
fixes and code are all welcome.

## How changes land here

frontier-connectors is developed in DimensionLab's main repository together with the rest of Frontier
and synchronised to GitHub commit by commit. That has two consequences worth knowing before
you open a pull request:

1. **Your pull request is reviewed here, then imported.** A maintainer ports your commits
   to the main repository (authorship preserved), it goes through the same CI as every
   other change, and it comes back to this repository in the next sync as a commit authored
   by you. The pull request is then closed with a link to that commit. GitHub will show it
   as "closed" rather than "merged"; the contribution graph still credits you because the
   commit carries your name and e-mail.
2. **A few files are generated.** The root `Cargo.toml` / `pyproject.toml`, `LICENSE`,
   `NOTICE`, everything under `.github/`, and the top-level Markdown files come from
   templates. Open an issue instead of a pull request if one of those needs to change.

## Sign your work (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/) instead of
a contributor licence agreement. Every commit needs a `Signed-off-by` line stating that you
have the right to submit it under the project licence:

```
git commit -s -m "fix(openfoam): read binary boundary fields"
```

The `DCO` check on your pull request fails if a commit is missing the line; fix it with
`git commit --amend -s` or `git rebase --signoff main`. If you contribute on behalf of an
employer, make sure they allow it: the sign-off is your statement that they do.

Contributions are accepted under the licence of this repository (Apache-2.0), inbound equals
outbound. We do not ask for copyright assignment.

## Ground rules

- Open an issue before a large change so nobody builds the same thing twice.
- Conventional commits (`feat(scope): ...`, `fix(scope): ...`); the changelog is generated
  from them.
- New source files start with `SPDX-License-Identifier: Apache-2.0`.
- Tests for new behaviour. Small sample files (kilobytes, not megabytes) that you have the
  right to redistribute; say where they came from in the pull request.
- **Never attach proprietary simulation data, vendor SDK files, or anything from a
  licensed solver installation** to an issue or pull request. Describe the format and point
  at public documentation instead.
- New dependencies must be permissively licensed (MIT, BSD, Apache-2.0, ISC, Zlib). Copyleft
  or proprietary tools can only be optional, imported lazily, and never required at import
  time.

## Development

Rust (stable toolchain, see `rust-toolchain.toml`):

```
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Python 3.11+ with [uv](https://docs.astral.sh/uv/):

```
python tools/bootstrap.py
uv run --no-sync pytest
uv run --no-sync ruff check .
```
The bootstrap uses the public sibling repositories, so no private checkout or unpublished package-index dependency is needed. Rust checks use the generated Cargo patches.

## Reporting security issues

Not in a public issue: see [SECURITY.md](SECURITY.md).

## Conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
