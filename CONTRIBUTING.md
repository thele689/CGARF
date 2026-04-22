# Contributing to CGARF

Thanks for taking the time to contribute. This repository is a cleaned public release of a research codebase, so the most helpful contributions are the ones that improve clarity, reproducibility, and stability without obscuring the paper-aligned pipeline.

## What Is Most Helpful

- Fixes for clearly scoped bugs in the released pipeline
- Improvements to documentation, setup, and reproducibility guidance
- Additional unit tests for core modules under `src/`
- Cleanup that makes the public release easier to understand and maintain
- Paper-to-code alignment fixes when the implementation-facing docs and code drift apart

## Before You Open a Pull Request

Please try to keep changes focused and explain:

1. what problem you found
2. why the current behavior is problematic
3. how your change addresses it
4. how you verified it

For larger changes, opening an issue first is appreciated so we can align on scope before you spend time on implementation.

## Development Setup

```bash
git clone https://github.com/thele689/CGARF.git
cd CGARF

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## Running Checks

Run the lightweight public checks first:

```bash
pytest tests/unit -q
```

You can run the broader suite when your environment is prepared:

```bash
pytest tests/
```

## Style Expectations

- Keep changes as small and local as reasonably possible.
- Preserve the paper-oriented stage structure unless there is a strong reason to refactor it.
- Avoid introducing hard-coded local absolute paths, private tokens, or machine-specific assumptions.
- Prefer clear naming and straightforward control flow over clever abstractions.
- When behavior changes, update the relevant documentation in `README.md` or `docs/`.

## Data, Assets, and Large Outputs

Please do not commit:

- local repository checkouts under `repos/`
- experiment outputs under `data/`, `results/`, or `demo_one_instance_output/`
- cache directories, logs, or temporary files
- API keys, tokens, or private credentials

## Pull Request Notes

When opening a PR, include:

- a short summary of the change
- verification notes
- any limitations or follow-up work still left

If the change affects the research workflow, it is especially helpful to mention which paper stage it touches, for example `CRG`, `CG-MAD`, `SRCD`, or `TSPF`.
