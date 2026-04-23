"""Runtime configuration helpers for CGARF command-line entry points.

The public release intentionally avoids hard-coded credentials.  This module
centralizes the small amount of environment handling needed by the runnable
entry points:

- load ``.env`` when python-dotenv is available;
- load a simple ``key.cfg`` file with ``KEY=value`` lines;
- expose built-in model profiles for the models used in the paper experiments;
- create an LLM interface from the resolved provider/model/API settings.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from loguru import logger


OPENAI_EXPERIMENT_MODELS = {
    "gpt-4.1": "gpt-4.1",
    "gpt-4o": "gpt-4o",
}

MODEL_PROFILES: Dict[str, Dict[str, Optional[str]]] = {
    "openai-gpt-4.1": {
        "provider": "openai",
        "model": "gpt-4.1",
        "api_key_env": "OPENAI_API_KEY",
        "api_base": None,
    },
    "openai-gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "api_base": None,
    },
    "vllm-qwen2.5-coder-32b": {
        "provider": "vllm",
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "api_key_env": "VLLM_API_KEY",
        "api_base_env": "VLLM_API_BASE",
        "api_key": "EMPTY",
        "api_base": "http://localhost:8000/v1",
    },
    "vllm-qwen3-coder-30b": {
        "provider": "vllm",
        "model": "Qwen/Qwen3-Coder-30B-Instruct",
        "api_key_env": "VLLM_API_KEY",
        "api_base_env": "VLLM_API_BASE",
        "api_key": "EMPTY",
        "api_base": "http://localhost:8000/v1",
    },
}

PROFILE_ALIASES = {
    "gpt-4.1": "openai-gpt-4.1",
    "gpt-4o": "openai-gpt-4o",
    "qwen2.5-coder-32b": "vllm-qwen2.5-coder-32b",
    "qwen3-coder-30b": "vllm-qwen3-coder-30b",
}


@dataclass
class LLMRuntimeConfig:
    provider: str
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    profile_name: str = "custom"
    api_key_source: str = "unset"

    def as_safe_dict(self) -> Dict[str, Optional[str]]:
        key_status = "set" if self.api_key else "unset"
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "api_base": self.api_base,
            "profile_name": self.profile_name,
            "api_key": key_status,
            "api_key_source": self.api_key_source,
        }


def _parse_key_value_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_key_cfg(path: Optional[Path], override: bool = False) -> Dict[str, str]:
    """Load ``KEY=value`` entries from a key.cfg-style file into the process env."""

    if path is None:
        return {}
    cfg_path = path.expanduser().resolve()
    values = _parse_key_value_file(cfg_path)
    for key, value in values.items():
        if override or not os.getenv(key):
            os.environ[key] = value
    if values:
        logger.info("Loaded {} entries from {}", len(values), cfg_path)
    return values


def load_runtime_environment(
    workspace_root: Optional[Path] = None,
    key_cfg: Optional[Path] = None,
) -> None:
    """Load .env and key.cfg without requiring either file to exist."""

    root = (workspace_root or Path.cwd()).resolve()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    explicit_key_cfg = key_cfg
    default_key_cfg = root / "key.cfg"
    if explicit_key_cfg is not None:
        load_key_cfg(explicit_key_cfg)
    elif default_key_cfg.exists():
        load_key_cfg(default_key_cfg)


def _profile_data(profile: Optional[str]) -> Tuple[str, Dict[str, Optional[str]]]:
    profile_name = profile or "openai-gpt-4.1"
    profile_name = PROFILE_ALIASES.get(profile_name, profile_name)
    if profile_name not in MODEL_PROFILES:
        known = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(f"Unknown model profile: {profile_name}. Known profiles: {known}")
    return profile_name, dict(MODEL_PROFILES[profile_name])


def resolve_llm_config(
    *,
    profile: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    workspace_root: Optional[Path] = None,
    key_cfg: Optional[Path] = None,
    require_api_key: bool = True,
) -> LLMRuntimeConfig:
    """Resolve provider/model/API settings for a runnable CGARF command."""

    load_runtime_environment(workspace_root=workspace_root, key_cfg=key_cfg)
    profile_name, data = _profile_data(profile)

    resolved_provider = provider or data.get("provider") or "openai"
    resolved_model = model or data.get("model") or "gpt-4.1"

    api_base_env = data.get("api_base_env")
    resolved_api_base = (
        api_base
        or (os.getenv(api_base_env) if api_base_env else None)
        or data.get("api_base")
        or (os.getenv("OPENAI_API_BASE") if resolved_provider == "openai" else None)
    )

    api_key_env = data.get("api_key_env") or (
        "OPENAI_API_KEY" if resolved_provider == "openai" else "VLLM_API_KEY"
    )
    default_api_key = data.get("api_key")
    if api_key:
        resolved_api_key = api_key
        api_key_source = "argument"
    elif os.getenv(api_key_env):
        resolved_api_key = os.getenv(api_key_env)
        api_key_source = f"env:{api_key_env}"
    elif default_api_key:
        resolved_api_key = default_api_key
        api_key_source = f"profile:{profile_name}"
    else:
        resolved_api_key = None
        api_key_source = "unset"

    if require_api_key and resolved_provider in {"openai", "vllm"} and not resolved_api_key:
        raise RuntimeError(
            f"Missing API key for provider '{resolved_provider}'. "
            f"Set {api_key_env}, pass --api-key, or create key.cfg with {api_key_env}=..."
        )

    return LLMRuntimeConfig(
        provider=resolved_provider,
        model_name=resolved_model,
        api_key=resolved_api_key,
        api_base=resolved_api_base,
        profile_name=profile_name,
        api_key_source=api_key_source,
    )


def create_configured_llm(
    *,
    profile: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    workspace_root: Optional[Path] = None,
    key_cfg: Optional[Path] = None,
    require_api_key: bool = True,
):
    """Create the configured LLM interface used by the batch runners."""

    from src.common.llm_interface import create_llm_interface

    config = resolve_llm_config(
        profile=profile,
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
        workspace_root=workspace_root,
        key_cfg=key_cfg,
        require_api_key=require_api_key,
    )
    logger.info("Using LLM runtime config: {}", config.as_safe_dict())
    llm_kwargs = {}
    if config.provider in {"openai", "vllm", "openai-compatible", "qwen"}:
        llm_kwargs["api_key"] = config.api_key
        if config.api_base:
            llm_kwargs["api_base"] = config.api_base
    return create_llm_interface(config.provider, config.model_name, **llm_kwargs)
