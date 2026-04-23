# CGARF

CGARF is a causality-guided automated program repair pipeline. This repository
keeps only the code and files needed to run the released pipeline simply.

## Prerequisites

CGARF uses Docker for target-repository execution. Pull the experiment image:

```bash
docker pull hejiaz/swe-agent:latest
```

CGARF also needs an LLM API key. For the default OpenAI profile, set:

```bash
export OPENAI_API_KEY=key_here
```

Or create `key.cfg`:

```bash
cp key.cfg.example key.cfg
# edit key.cfg and set OPENAI_API_KEY=key_here
```

Built-in model profiles:

- `openai-gpt-4.1` (default)
- `openai-gpt-4o`
- `vllm-qwen2.5-coder-32b`
- `vllm-qwen3-coder-30b`

For vLLM, start an OpenAI-compatible server first, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve Qwen/Qwen3-Coder-30B-Instruct \
  --download-dir /code-llm/Qwen \
  --dtype half \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 4
```

Then set:

```bash
export VLLM_API_BASE=http://localhost:8000/v1
export VLLM_API_KEY=EMPTY
```

## Install

```bash
git clone https://github.com/thele689/CGARF.git
cd CGARF

conda create -n cgarf python=3.10
conda activate cgarf

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Run

Run causal trace analysis:

```bash
python evaluation/run.py \
  --final_stage trace_analysis \
  --instance_ids astropy__astropy-12907 astropy__astropy-6938
```

Run through patch search:

```bash
python evaluation/run.py \
  --final_stage search \
  --instance_ids astropy__astropy-12907
```

Use GPT-4o:

```bash
python evaluation/run.py \
  --model-profile openai-gpt-4o \
  --final_stage trace_analysis \
  --instance_ids astropy__astropy-12907
```

Use a local vLLM model:

```bash
python evaluation/run.py \
  --model-profile vllm-qwen3-coder-30b \
  --final_stage search \
  --instance_ids astropy__astropy-12907
```

Stages:

- `crg`: build the causal relevance graph.
- `trace_analysis`: run CRG construction and CG-MAD.
- `srcd_initial`: generate initial patches.
- `reflection`: run structured self-reflection.
- `search`: run reflection and consistency distillation.
- `tspf`: run final patch filtering when test evidence is available.

Outputs are written under `data/` and `results/`.

## Test

```bash
pip install -r requirements-dev.txt
pytest tests/unit -q
```

## Repository Layout

```text
CGARF/
├── evaluation/run.py
├── src/
│   ├── common/
│   ├── phase0_integrator/
│   ├── phase1_causal_analysis/
│   ├── srcd/
│   └── tspf/
├── input/
├── config/
├── tests/unit/
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## License

MIT License. See `LICENSE`.
