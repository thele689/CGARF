# CGARF

CGARF is a causality-guided automated program repair framework that couples:

- `CG-MAD`: causal graph augmentation plus multi-agent debate for candidate reranking.
- `SRCD`: self-reflection consistency distillation for patch generation and stabilization.
- `TSPF`: two-stage patch filtering with functional evidence plus causal/group ranking.

This repository is a **clean public release** of the CGARF research codebase. It keeps the core implementation, tests, configuration, input files, and paper-facing technical appendices, while excluding local experiment artifacts, checked-out target repositories, cached datasets, demo outputs, logs, and any hard-coded secrets.

## What Is Included

- Core source code under `src/`
- Unit and integration tests under `tests/`
- Input localization JSONL files under `input/`
- Reusable scripts under `scripts/`
- Base configuration under `config/`
- Paper-facing implementation appendices under `docs/`
- Packaging metadata, dependency files, and GitHub-ready ignore rules

## What Is Intentionally Excluded

The following items are **not** vendored in this public release:

- local checked-out buggy repositories under `repos/`
- generated graphs, patches, and experiment outputs under `data/`, `results/`, `demo_one_instance_output/`
- runtime caches, temporary directories, and logs
- local SWE-bench clone / Docker artifacts
- any hard-coded API keys or private credentials

If you want to reproduce end-to-end experiments, you should prepare those assets locally after cloning this repository.

## Repository Layout

```text
CGARF/
├── src/
│   ├── common/                  # shared data structures, utilities, LLM interfaces
│   ├── phase0_integrator/       # localization input loading / integration
│   ├── phase1_causal_analysis/  # code graph, CRG, CG-MAD
│   ├── srcd/                    # SRCD (3.2.1 / 3.2.2 / 3.2.3)
│   ├── tspf/                    # TSPF (3.3)
│   ├── swebench/                # SWE-bench-facing helpers
│   └── swe_bench_evaluator/     # official harness adapters / evaluation helpers
├── tests/
├── scripts/
├── input/
├── config/
├── docs/
├── tools/
├── requirements.txt
├── requirements-dev.txt
├── requirements-swebench.txt
└── README.md
```

## Python and Installation

The current cleaned release is intended for **Python 3.10+**.

```bash
git clone https://github.com/<your-account>/CGARF.git
cd CGARF

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

## Environment Variables

Create a local `.env` file from `.env.example` and fill only the providers you actually use.

Common variables used by the codebase:

- `OPENAI_API_KEY`
- `QWEN_API_KEY`
- `QWEN_API_BASE`
- `SILICONFLOW_API_KEY`
- `SILICONFLOW_API_BASE`
- `DASHSCOPE_API_KEY`
- `DASHSCOPE_API_BASE`
- `OPENROUTER_API_KEY`
- `OPENROUTER_API_BASE`

Useful runtime controls:

- `QWEN_REQUEST_TIMEOUT_SECONDS`
- `QWEN_MAX_RETRY_ATTEMPTS`
- `QWEN_MIN_REQUEST_INTERVAL_SECONDS`
- `QWEN_RATE_LIMIT_BACKOFF_SECONDS`
- `QWEN_FORCE_REQUESTS`

## Quick Start by Phase

The codebase is organized around the paper pipeline rather than a single polished CLI. The most direct public entrypoints are the batch runners for each paper stage.

### 1. Phase 1: CRG Construction

```bash
python -m src.phase1_causal_analysis.batch_crg_constructor \
  --workspace-root . \
  --instance-id <instance_id> \
  --method orcaloca
```

### 2. Phase 1.2: CG-MAD

```bash
python -m src.phase1_causal_analysis.batch_cg_mad \
  --workspace-root . \
  --instance-id <instance_id> \
  --method orcaloca
```

### 3. Phase 3.2.1: Initial SRCD Patches

```bash
python -m src.srcd.batch_srcd \
  --workspace-root . \
  --instance-id <instance_id> \
  --method orcaloca \
  --total-sampling-budget 8
```

### 4. Phase 3.2.2: Structured Reflection

```bash
python -m src.srcd.batch_reflection \
  --workspace-root . \
  --instance-id <instance_id> \
  --current-temperature 0.2
```

### 5. Phase 3.2.3: Consistency Distillation

```bash
python -m src.srcd.batch_distillation \
  --workspace-root . \
  --instance-id <instance_id> \
  --top-k-per-candidate 5
```

### 6. Phase 3.3: TSPF

```bash
python -m src.tspf.batch_tspf \
  --workspace-root . \
  --instance-id <instance_id> \
  --distillation-json <path/to/distillation.json>
```

## External Dependencies for Full Evaluation

### SWE-bench / Official Docker Harness

This clean release does **not** vendor the official `SWE-bench` repository. If you want paper-faithful TSPF validation with the official Docker harness, clone it separately and install the optional dependencies:

```bash
pip install -r requirements-swebench.txt
git clone https://github.com/princeton-nlp/SWE-bench.git external/SWE-bench
pip install -e external/SWE-bench
```

Then point your evaluation scripts to the local SWE-bench checkout path.

### Target Repositories and Runtime Outputs

The framework expects local buggy repository checkouts and runtime output directories such as:

- `repos/`
- `data/`
- `results/`
- `demo_one_instance_output/`

They are intentionally excluded from version control and should be created locally as needed.

## Tests

Run the default test suite with:

```bash
pytest tests/
```

Some integration tests assume local data, target repositories, or optional external tooling. For public CI, start with unit tests and selectively enable integration tests in a prepared environment.

## Notes on Imports

The current codebase keeps the historical package layout under `src`, so imports use the `src.*` namespace, for example:

```python
from src.phase1_causal_analysis.cg_mad import CGMADMechanism
from src.srcd.repair_generator import RepairGenerator
from src.tspf.patch_filter import TwoStagePatchFilter
```

## Paper-Facing Technical Docs

Two paper-aligned appendix drafts are included in `docs/`:

- `docs/APPENDIX_C_CG_MAD_FINAL_ZH.md`
- `docs/APPENDIX_D_SRCD_FINAL_ZH.md`

They are useful if you want the implementation-facing description to stay aligned with the released code.

## License

MIT License. See `LICENSE`.
