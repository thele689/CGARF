"""Run official SWE-bench evaluation with a local repository bundle fallback.

The official SWE-bench Docker harness normally clones each target repository
from GitHub while building an instance image. That is faithful to the upstream
workflow, but brittle in offline or restricted Docker networks. This wrapper
keeps the official ``swebench.harness.run_evaluation`` execution path intact
and only changes the repository setup script to clone from a local git bundle
that CGARF places in the Docker build context.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import shutil


LOCAL_BUNDLE_NAME = "cgarf_local_repo.bundle"
LOCAL_BUNDLE_IN_CONTAINER = f"/root/{LOCAL_BUNDLE_NAME}"


def _install_local_repo_monkeypatches(bundle_path: Path) -> None:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Local SWE-bench repository bundle not found: {bundle_path}")

    import swebench.harness.docker_build as docker_build
    import swebench.harness.test_spec.create_scripts as create_scripts
    import swebench.harness.test_spec.python as python_scripts
    import swebench.harness.test_spec.test_spec as test_spec_module

    original_make_repo_script_list_py = python_scripts.make_repo_script_list_py
    original_get_dockerfile_instance = test_spec_module.get_dockerfile_instance
    original_build_image = docker_build.build_image

    def make_repo_script_list_py_local(
        specs,
        repo,
        repo_directory,
        base_commit,
        env_name,
    ) -> list[str]:
        commands = original_make_repo_script_list_py(
            specs,
            repo,
            repo_directory,
            base_commit,
            env_name,
        )
        if not commands:
            return commands
        commands[0] = f"git clone -o origin {LOCAL_BUNDLE_IN_CONTAINER} {repo_directory}"
        return commands

    def get_dockerfile_instance_local(*args, **kwargs) -> str:
        dockerfile = original_get_dockerfile_instance(*args, **kwargs)
        needle = "COPY ./setup_repo.sh /root/\n"
        if needle in dockerfile and LOCAL_BUNDLE_NAME not in dockerfile:
            return dockerfile.replace(
                needle,
                needle + f"COPY ./{LOCAL_BUNDLE_NAME} {LOCAL_BUNDLE_IN_CONTAINER}\n",
                1,
            )
        return dockerfile

    def build_image_with_local_bundle(*args, **kwargs):
        build_dir = kwargs.get("build_dir")
        if build_dir is None and len(args) >= 6:
            build_dir = args[5]
        if build_dir is not None:
            target = Path(build_dir) / LOCAL_BUNDLE_NAME
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle_path, target)
        return original_build_image(*args, **kwargs)

    python_scripts.make_repo_script_list_py = make_repo_script_list_py_local
    create_scripts.make_repo_script_list_py = make_repo_script_list_py_local
    test_spec_module.get_dockerfile_instance = get_dockerfile_instance_local
    docker_build.build_image = build_image_with_local_bundle


def main() -> None:
    bundle_env = os.environ.get("CGARF_SWEBENCH_LOCAL_REPO_BUNDLE")
    if not bundle_env:
        raise RuntimeError("CGARF_SWEBENCH_LOCAL_REPO_BUNDLE is required")
    _install_local_repo_monkeypatches(Path(bundle_env))
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")


if __name__ == "__main__":
    main()
