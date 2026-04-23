# CGARF

[![CI](https://github.com/thele689/CGARF/actions/workflows/ci.yml/badge.svg)](https://github.com/thele689/CGARF/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#python-and-installation)

CGARF is a causality-guided automated program repair framework that couples:

- `CG-MAD`: causal graph augmentation plus multi-agent debate for candidate reranking.
- `SRCD`: self-reflection consistency distillation for patch generation and stabilization.
- `TSPF`: two-stage patch filtering with functional evidence plus causal/group ranking.

This repository is a **clean public release** of the CGARF research codebase. It keeps the core implementation, tests, configuration, input files, and paper-facing technical appendices, while excluding local experiment artifacts, checked-out target repositories, cached datasets, demo outputs, logs, and any hard-coded secrets.

## Highlights

- Causality-aware reranking via CRG construction and CG-MAD.
- Patch generation stabilized with structured self-reflection and consistency distillation.
- Two-stage filtering that combines functional evidence with causal and group-level ranking.
- Public release prepared for inspection, extension, and reproducible local experimentation.

## Project Status and Scope

This repository is aimed at **research reproduction and method inspection**, not at providing a single polished production CLI.

- The code path follows the paper pipeline stage by stage.
- Lightweight unit tests are included and run in public CI.
- Full end-to-end evaluation still requires local preparation of target repositories, SWE-bench assets, and runtime outputs.
- Some integration tests and scripts assume a prepared research environment and optional external tooling.

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

## Prerequisites

CGARF requires Docker for SWE-bench-style validation and reproducible target
repository execution. You can either build the release image locally or pull the
image used by our experiment environment:

```bash
docker pull hejiaz/swe-agent:latest
```

CGARF also requires API access to an LLM. The unified runner includes built-in
profiles for the OpenAI models used in our experiments:

- `openai-gpt-4.1` -> `gpt-4.1`
- `openai-gpt-4o` -> `gpt-4o`

For OpenAI runs, users only need to provide `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY=key_here
```

Alternatively, create a local `key.cfg` file in the repository root:

```bash
cp key.cfg.example key.cfg
```

Then edit `key.cfg` so it contains:

```bash
OPENAI_API_KEY=key_here
```

`key.cfg` is ignored by git and is loaded automatically by `evaluation/run.py`.

For open-source code models served through vLLM, CGARF uses the
OpenAI-compatible API. Example servers:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
  --download-dir /code-llm/Qwen \
  --dtype half \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 4
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve Qwen/Qwen3-Coder-30B-Instruct \
  --download-dir /code-llm/Qwen \
  --dtype half \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 4
```

Then point CGARF to the local endpoint:

```bash
export VLLM_API_BASE=http://localhost:8000/v1
export VLLM_API_KEY=EMPTY
```

```bash
git clone https://github.com/thele689/CGARF.git
cd CGARF

conda create -n cgarf python=3.10
conda activate cgarf

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

For CPU-only environments, a lighter installation path is to preinstall a CPU-only PyTorch wheel before installing the rest of the requirements, because `sentence-transformers` depends on `torch`.

## Environment Variables

Create a local `.env` file from `.env.example` and fill only the providers you actually use.

Common variables used by the codebase:

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`
- `VLLM_API_KEY`
- `VLLM_API_BASE`
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

The easiest public entry point is the unified runner:

```bash
python evaluation/run.py \
  --final_stage trace_analysis \
  --instance_ids astropy__astropy-12907 astropy__astropy-6938
```

Add the patch-search stage:

```bash
python evaluation/run.py \
  --final_stage search \
  --instance_ids astropy__astropy-12907
```

Use GPT-4o instead of the default GPT-4.1 profile:

```bash
python evaluation/run.py \
  --model-profile openai-gpt-4o \
  --final_stage trace_analysis \
  --instance_ids astropy__astropy-12907
```

Use a local vLLM server:

```bash
python evaluation/run.py \
  --model-profile vllm-qwen3-coder-30b \
  --final_stage search \
  --instance_ids astropy__astropy-12907
```

Available `--final_stage` values are:

- `crg`: build the causal relevance graph and initialize edge weights.
- `trace_analysis`: run CRG construction plus CG-MAD.
- `srcd_initial`: generate initial SRCD patches.
- `reflection`: run structured self-reflection.
- `search`: run SRCD through consistency distillation.
- `tspf`: run two-stage patch filtering on distilled patches.

The lower-level batch runners remain available when you want to inspect or rerun
one stage manually.

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

For the fastest local sanity check:

```bash
pytest tests/unit -q
```

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
- `docs/DOCKER.md`

They are useful if you want the implementation-facing description to stay aligned with the released code.

## Contributing

Contributions are welcome, especially around:

- reproducibility and environment setup improvements
- documentation and code-path clarification
- test coverage for phase-level components
- bug fixes that keep the released pipeline behavior aligned with the paper

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Security

If you find a security issue, please avoid opening a public issue with sensitive details first. See [SECURITY.md](SECURITY.md) for the preferred disclosure path.

## Citation

If this repository helps your work, please cite the CGARF paper and link to this repository release. The appendix drafts in `docs/` are included to keep the public implementation description aligned with the released code.

## License

MIT License. See `LICENSE`.
