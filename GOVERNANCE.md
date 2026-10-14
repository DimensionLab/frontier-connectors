# Governance

frontier-connectors is maintained by [DimensionLab](https://dimensionlab.org), which employs the core
maintainers and owns the release process. The project is open source (Apache-2.0), not
open governance: DimensionLab decides the roadmap, and says so plainly rather than
pretending otherwise. What we commit to in return:

- **Stays open.** Code published here under Apache-2.0 stays under Apache-2.0. We will not
  relicense it; the DCO (no copyright assignment) means we could not do so unilaterally for
  contributed code anyway.
- **No crippling.** Features are not removed from the open-source packages to move them
  into the commercial product.
- **Stable contracts.** The dataset format, the connector plugin interface and the logging
  protocol follow semantic versioning and a documented data-model version; breaking changes
  go through a public issue first and ship with a migration path.
- **Public decisions.** Design changes that affect plugin authors are discussed in issues
  on this repository before they are made.

## Roles

- **Maintainers** (DimensionLab engineers) review, import and release.
- **Reviewers** are external contributors with a track record in an area (for example one
  solver format); they are asked for review on changes there and listed in `CODEOWNERS`.
  Reviewers are proposed by a maintainer after several accepted contributions.
- **Contributors** are everyone else; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Releases

Versions follow semantic versioning; before 1.0 a minor version may break. Releases are cut
from the main branch roughly monthly, with patch releases as needed. The changelog is
generated from conventional commits.

## Trademarks

"DimensionLab" and "Frontier" (as the name of this product) and the logos are trademarks of
DimensionLab. The licence does not grant rights to them. You may say your project "works
with Frontier" or is "a connector for Frontier"; do not name a product or a fork in a way
that suggests it is the official one. Details: [TRADEMARKS.md](TRADEMARKS.md).
