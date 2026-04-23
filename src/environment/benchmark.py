"""CGARF benchmark-oriented Docker environment wrapper.

This adapts the ideas behind OrcaLoca's benchmark environment to CGARF's public
release: image preflight, persistent bash sessions, command execution, and
repeatable smoke tests around the repository's Docker task environment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from src.environment.utils import (
    DEFAULT_CONTAINER_WORKSPACE,
    ContainerBash,
    DockerRuntimeError,
    build_image,
    copy_text_to_container,
    ensure_docker_available,
    generate_container_name,
    get_container,
    get_exit_code,
    image_exists,
    run_command_in_container,
    stop_container,
)


@dataclass
class BenchmarkRunResult:
    command: str
    output: str
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class CGARFDockerBenchmarkEnv:
    """Lifecycle manager for CGARF's Docker task environment."""

    def __init__(
        self,
        workspace_root: Path,
        image_name: str = "cgarf:dev",
        dockerfile: Optional[Path] = None,
        env_file: Optional[Path] = None,
        persistent: bool = True,
        mount_docker_socket: bool = False,
        container_workspace: str = DEFAULT_CONTAINER_WORKSPACE,
        container_name: Optional[str] = None,
        build_args: Optional[Dict[str, str]] = None,
    ):
        self.workspace_root = workspace_root.resolve()
        self.image_name = image_name
        self.dockerfile = dockerfile or self.workspace_root / "Dockerfile"
        self.env_file = env_file
        self.persistent = persistent
        self.mount_docker_socket = mount_docker_socket
        self.container_workspace = container_workspace
        self.container_name = container_name or generate_container_name(image_name)
        self.build_args = build_args or {}
        self.ctr_bash: Optional[ContainerBash] = None

    def ensure_image(self, force_rebuild: bool = False) -> None:
        ensure_docker_available()
        if force_rebuild or not image_exists(self.image_name):
            build_image(
                image_name=self.image_name,
                context_dir=self.workspace_root,
                dockerfile=self.dockerfile,
                build_args=self.build_args,
            )

    def start(self) -> ContainerBash:
        if self.ctr_bash is not None:
            return self.ctr_bash
        self.ctr_bash = get_container(
            ctr_name=self.container_name,
            image_name=self.image_name,
            workspace_root=self.workspace_root,
            container_workspace=self.container_workspace,
            env_file=self.env_file,
            mount_docker_socket=self.mount_docker_socket,
            persistent=self.persistent,
            extra_env={"PYTHONPATH": self.container_workspace},
        )
        logger.info(
            "Started CGARF Docker benchmark env: image={} container={}",
            self.image_name,
            self.container_name,
        )
        return self.ctr_bash

    def stop(self, remove: bool = False) -> None:
        if self.ctr_bash is None:
            return
        stop_container(self.ctr_bash, remove=remove)
        self.ctr_bash = None

    def copy_to_env(self, contents: str, container_path: str) -> None:
        self.start()
        copy_text_to_container(self.container_name, contents, container_path)

    def run(
        self,
        command: str,
        timeout: int = 120,
        output_log: bool = False,
        check: bool = True,
    ) -> BenchmarkRunResult:
        ctr_bash = self.start()
        output = run_command_in_container(ctr_bash, command, timeout=timeout, output_log=output_log)
        exit_code = get_exit_code(ctr_bash, timeout=10)
        result = BenchmarkRunResult(command=command, output=output, exit_code=exit_code)
        if check and not result.ok:
            raise DockerRuntimeError(
                f"Container command failed with exit code {exit_code}: {command}\n{output}"
            )
        return result

    def run_unit_tests(self, timeout: int = 600, output_log: bool = True) -> BenchmarkRunResult:
        return self.run("pytest tests/unit -q", timeout=timeout, output_log=output_log, check=True)

    def smoke_install(self, timeout: int = 120, output_log: bool = False) -> BenchmarkRunResult:
        return self.run(
            "python -c \"import src, src.tspf.official_swebench_harness, src.environment.benchmark; print('CGARF import OK')\"",
            timeout=timeout,
            output_log=output_log,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CGARF Docker benchmark environment tasks.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--image", default="cgarf:dev")
    parser.add_argument("--dockerfile", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--container-name", default=None)
    parser.add_argument("--container-workspace", default=DEFAULT_CONTAINER_WORKSPACE)
    parser.add_argument("--mount-docker-socket", action="store_true")
    parser.add_argument("--non-persistent", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--action",
        choices=["build", "test", "smoke-install", "run"],
        default="test",
    )
    parser.add_argument("--command", default="pwd")
    parser.add_argument("--output-log", action="store_true")
    args = parser.parse_args()

    env = CGARFDockerBenchmarkEnv(
        workspace_root=args.workspace_root,
        image_name=args.image,
        dockerfile=args.dockerfile,
        env_file=args.env_file,
        persistent=not args.non_persistent,
        mount_docker_socket=args.mount_docker_socket,
        container_workspace=args.container_workspace,
        container_name=args.container_name,
    )

    try:
        env.ensure_image(force_rebuild=args.force_rebuild)
        if args.action == "build":
            logger.success("Docker image is ready: {}", args.image)
            return
        if args.action == "test":
            env.run_unit_tests(timeout=args.timeout, output_log=True)
            logger.success("Container unit tests passed.")
            return
        if args.action == "smoke-install":
            env.smoke_install(timeout=args.timeout, output_log=args.output_log)
            logger.success("Container smoke install/import check passed.")
            return
        if args.action == "run":
            result = env.run(
                args.command,
                timeout=args.timeout,
                output_log=args.output_log,
                check=False,
            )
            print(result.output, end="" if result.output.endswith("\n") else "\n")
            if result.exit_code != 0:
                raise SystemExit(result.exit_code)
    finally:
        env.stop(remove=not args.non_persistent)


if __name__ == "__main__":
    main()
