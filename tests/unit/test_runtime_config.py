from src.common.llm_interface import MockLLMInterface
from src.common.runtime_config import create_configured_llm, resolve_llm_config


def test_resolve_openai_profile_from_key_cfg(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "key.cfg").write_text("OPENAI_API_KEY=test-openai-key\n")

    config = resolve_llm_config(
        profile="openai-gpt-4.1",
        workspace_root=tmp_path,
        require_api_key=True,
    )

    assert config.provider == "openai"
    assert config.model_name == "gpt-4.1"
    assert config.api_key == "test-openai-key"
    assert config.api_key_source == "env:OPENAI_API_KEY"


def test_resolve_gpt4o_alias_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = resolve_llm_config(
        profile="gpt-4o",
        workspace_root=tmp_path,
        require_api_key=True,
    )

    assert config.profile_name == "openai-gpt-4o"
    assert config.model_name == "gpt-4o"


def test_resolve_vllm_profile_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_BASE", raising=False)

    config = resolve_llm_config(
        profile="vllm-qwen3-coder-30b",
        workspace_root=tmp_path,
        require_api_key=True,
    )

    assert config.provider == "vllm"
    assert config.api_key == "EMPTY"
    assert config.api_base == "http://localhost:8000/v1"
    assert config.model_name == "Qwen/Qwen3-Coder-30B-Instruct"


def test_create_configured_mock_llm(tmp_path):
    llm = create_configured_llm(
        provider="mock",
        model="mock-model",
        workspace_root=tmp_path,
        require_api_key=False,
    )

    assert isinstance(llm, MockLLMInterface)
    assert llm.model_name == "mock-model"
