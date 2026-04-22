# Public Release Scope

This cleaned release keeps the code and documents needed to understand, run,
and extend CGARF, while excluding local artifacts that should not be published
to GitHub.

## Included

- source code under `src/`
- tests under `tests/`
- reusable scripts under `scripts/`
- configuration under `config/`
- localization inputs under `input/`
- selected technical documentation under `docs/`
- packaging / dependency metadata

## Excluded

- checked-out target repositories
- generated graphs, cached patches, experiment outputs
- runtime logs and temporary files
- vendored SWE-bench clone
- local API keys and secret-bearing notes
- ad hoc debugging scripts and one-off experiment folders

## Expected Local-Only Directories

These paths are intentionally git-ignored and should be created locally only if
you run experiments:

- `repos/`
- `data/`
- `results/`
- `demo_one_instance_output/`
- `SWE-bench/` or `external/SWE-bench/`
