from src.environment.benchmark import CGARFDockerBenchmarkEnv
from src.environment.utils import (
    DEFAULT_CONTAINER_WORKSPACE,
    build_run_command,
    generate_container_name,
)


def test_generate_container_name_sanitizes_image_name():
    name = generate_container_name("ghcr.io/demo/cgarf:latest")
    assert "/" not in name
    assert ":" not in name
    assert "ghcr.io-demo-cgarf-latest" in name


def test_build_run_command_with_workspace_and_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("A=B\n")

    command = build_run_command(
        ctr_name="cgarf-test",
        image_name="cgarf:dev",
        workspace_root=tmp_path,
        container_workspace=DEFAULT_CONTAINER_WORKSPACE,
        env_file=env_file,
        mount_docker_socket=True,
        persistent=True,
        extra_env={"PYTHONPATH": DEFAULT_CONTAINER_WORKSPACE},
    )

    assert command[:3] == ["docker", "run", "-i"]
    assert "--env-file" in command
    assert f"{tmp_path.resolve()}:{DEFAULT_CONTAINER_WORKSPACE}" in command
    assert "/var/run/docker.sock:/var/run/docker.sock" in command
    assert "cgarf:dev" in command


def test_benchmark_env_defaults(tmp_path):
    env = CGARFDockerBenchmarkEnv(workspace_root=tmp_path, image_name="cgarf:test")
    assert env.workspace_root == tmp_path.resolve()
    assert env.image_name == "cgarf:test"
    assert env.container_workspace == DEFAULT_CONTAINER_WORKSPACE
