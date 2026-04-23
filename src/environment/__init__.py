"""Docker-oriented runtime helpers for CGARF benchmark execution."""

from src.environment.utils import (
    ContainerBash,
    DockerCheckResult,
    DockerRuntimeError,
    build_image,
    ensure_docker_available,
    generate_container_name,
    get_container,
    image_exists,
    run_command_in_container,
)


def __getattr__(name):
    if name == "CGARFDockerBenchmarkEnv":
        from src.environment.benchmark import CGARFDockerBenchmarkEnv

        return CGARFDockerBenchmarkEnv
    raise AttributeError(name)


__all__ = [
    "CGARFDockerBenchmarkEnv",
    "ContainerBash",
    "DockerCheckResult",
    "DockerRuntimeError",
    "build_image",
    "ensure_docker_available",
    "generate_container_name",
    "get_container",
    "image_exists",
    "run_command_in_container",
]
