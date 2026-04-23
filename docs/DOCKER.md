# Running CGARF with Docker

This repository now includes a basic Docker setup so the public release can be installed and exercised in a consistent environment.

## What the Docker Setup Covers

The included `Dockerfile` is intended for:

- installing the cleaned public CGARF repository
- running unit tests
- opening an interactive shell for phase-level experimentation
- serving as a stable base image for local reproduction

It is **not** a full replacement for every external runtime asset used in paper-scale experiments. You still need to prepare local target repositories, runtime output directories, and optional SWE-bench assets when running the full pipeline.

The image intentionally installs only a minimal system package set and relies on Python wheels where possible, so the container stays closer to a practical public-release development image.
It also preinstalls a **CPU-only PyTorch wheel** before the rest of the Python stack so `sentence-transformers` does not pull an unnecessarily heavy CUDA-oriented dependency chain into a CPU container.

## Build the Image

```bash
docker build -t cgarf:dev .
```

For the experiment environment used with SWE-bench-style execution, you may
also pull the prebuilt image:

```bash
docker pull hejiaz/swe-agent:latest
```

Because the repository depends on modern NLP / LLM-related Python packages, the Python dependency layer can still take a while to resolve on a cold build.

## Run Unit Tests in the Container

```bash
docker run --rm cgarf:dev pytest tests/unit -q
```

## Start an Interactive Shell

```bash
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)":/workspace/CGARF \
  cgarf:dev
```

Inside the container, the repository is available at `/workspace/CGARF`.

## Use Docker Compose

Build:

```bash
docker compose build
```

Open a shell:

```bash
docker compose run --rm cgarf
```

Run unit tests:

```bash
docker compose run --rm cgarf pytest tests/unit -q
```

## Repository-Native Benchmark Environment Wrapper

Inspired by the Docker benchmark environment pattern used in other APR
projects, CGARF now includes:

- `src.environment.utils`
- `src.environment.benchmark`

The wrapper supports Docker daemon preflight checks, image existence checks,
persistent bash sessions, timeout-aware command execution, and simple smoke
tests against the mounted CGARF repository.

Build or verify the image:

```bash
python -m src.environment.benchmark \
  --workspace-root . \
  --image cgarf:dev \
  --action build
```

Run unit tests through the wrapper:

```bash
python -m src.environment.benchmark \
  --workspace-root . \
  --image cgarf:dev \
  --action test
```

Or run a smoke import check in the pulled experiment image:

```bash
python -m src.environment.benchmark \
  --workspace-root . \
  --image hejiaz/swe-agent:latest \
  --action smoke-install
```

Run an arbitrary command inside the container task environment:

```bash
python -m src.environment.benchmark \
  --workspace-root . \
  --image cgarf:dev \
  --action run \
  --command "python -m src.tspf.batch_tspf --help"
```

## Expected Local Directories for Larger Runs

For end-to-end or paper-style runs, you will usually want local directories such as:

- `repos/`
- `data/`
- `results/`
- `demo_one_instance_output/`

These are intentionally excluded from version control. Create and mount them locally as needed.

## Environment Variables

If you call LLM-backed stages from inside the container, provide a local `.env` file or pass variables explicitly. Common examples include:

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`
- `VLLM_API_KEY`
- `VLLM_API_BASE`
- `QWEN_API_KEY`
- `SILICONFLOW_API_KEY`
- `DASHSCOPE_API_KEY`
- `OPENROUTER_API_KEY`

## Running the Official SWE-bench Docker Harness from Inside the Container

Some TSPF evaluation paths rely on the official SWE-bench Docker harness. If you want the CGARF container itself to invoke Dockerized SWE-bench jobs, the inner process needs access to a Docker daemon.

The usual local approach is to mount the host Docker socket:

```bash
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)":/workspace/CGARF \
  -v /var/run/docker.sock:/var/run/docker.sock \
  cgarf:dev
```

Please use that setup carefully, because mounting the Docker socket effectively grants the container broad control over the host Docker daemon.

## Notes

- The public repository includes the Docker configuration and the public code, but not vendored buggy repositories or cached SWE-bench assets.
- If your machine already runs inside another containerized environment, you may prefer bind mounts over rebuilding images repeatedly during iteration.
