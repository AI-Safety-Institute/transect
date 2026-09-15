# Contributing

Contributions are welcome. This file covers the workflow;
[AGENTS.md](AGENTS.md) holds the working knowledge of the
codebase (architecture, contracts, practices) and applies to human
contributors as much as to coding agents - read it first.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it provisions Python 3.12+).

```bash
uv sync         # package + dev dependencies
make check      # the full gate: lint + typecheck + tests
make hooks      # install the pre-commit hook (once per clone)
```

Report-rendering changes additionally need the browser test
(`uv run --group ui-test playwright install chromium`, then
`uv run --group ui-test pytest`) and a real-browser look at a rendered
report - chart regressions are typically silent in Python. Check the
browser test actually ran rather than skipped (it skips without
chromium or offline): the pytest summary should not report a skip.

## Pull requests

- For anything beyond a small fix - a new feature, a new scanner, a
  new runtime dependency - open an issue first to discuss.
- Conventional Commits titles (`feat:`, `fix:`, `docs:`, `refactor:`,
  `chore:`, `test:`, `ci:`); we squash-merge, so the PR title becomes
  the commit message on `main`.
- One change or one coordinated change-set per PR.
- A PR that changes `src/transect/` bumps `version` in `pyproject.toml` in
  the same PR (CI enforces this): semver, so a breaking API or frame
  schema change bumps minor while we are pre-1.0, otherwise patch.
- `make check` must pass; CI runs the same gate on every PR, plus a
  package build check and the browser test. Report changes come with
  a screenshot or a note saying they were verified in a browser.
- A change to a user-facing contract (frame columns, API signatures,
  report sections, spec fields) includes the matching doc updates -
  see "Keep the docs true" in AGENTS.md.
- Honest framing (see AGENTS.md) applies to all user-facing copy:
  judged output is descriptive, not validated - never upgrade a
  judge's label into ground truth.

## Releases and versioning

Manual semver, no PyPI - the package installs from GitHub and versions
are git tags.

1. The version was already bumped by the PRs being released (see
   Pull requests above; `transect.__version__` reads package metadata, so
   pyproject.toml is the only place a version lives). Releasing is
   tagging `main`'s current version.
2. Tag the commit on `main`: `git pull`, then
   `git tag -a v0.2.0 -m "v0.2.0"` and `git push origin v0.2.0`.
3. Create a GitHub Release for the tag with short notes (the squash
   merge PR titles since the last tag are the changelog).

Users pin a version in their dependencies as:

```
transect @ git+https://github.com/AI-Safety-Institute/transect.git@v0.2.0
```

While the version is `0.x` the API may still move between minor
versions. `importlib.metadata.version("transect")` reports the installed
version.
