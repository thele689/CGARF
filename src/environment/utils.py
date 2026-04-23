"""Docker runtime utilities for CGARF.

This module is inspired by the container interaction pattern used in OrcaLoca's
benchmark environment. It provides:

- Docker daemon / image preflight checks with user-facing diagnostics
- persistent or one-shot interactive bash sessions inside containers
- timeout-aware command execution through a long-lived shell process
- lightweight file copy helpers for container-local setup work
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from loguru import logger


DOCKER_START_UP_DELAY = 1.0
DEFAULT_CONTAINER_WORKSPACE = "/workspace/CGARF"


class DockerRuntimeError(RuntimeError):
    """Raised when the Docker runtime is unavailable or misconfigured."""


@dataclass
class DockerCheckResult:
    ok: bool
    docker_cli_available: bool
    daemon_available: bool
    details: str = ""

    def to_dict(self) -> Dict[str, Union[bool, str]]:
        return {
            "ok": self.ok,
            "docker_cli_available": self.docker_cli_available,
            "daemon_available": self.daemon_available,
            "details": self.details,
        }


@dataclass
class ContainerBash:
    ctr_subprocess: subprocess.Popen
    ctr_name: str
    image_name: str
    persistent: bool = False
    ctr_pid: Optional[int] = None
    last_exit_code: Optional[int] = None

    def __post_init__(self) -> None:
        if self.ctr_pid is None:
            self.ctr_pid = get_bash_pid_in_docker(self.ctr_subprocess)


def _subprocess_text(
    args: Sequence[str],
    *,
    check: bool = False,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(args),
            text=True,
            capture_output=True,
            check=check,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else None,
        )
    except FileNotFoundError as exc:
        raise DockerRuntimeError(
            "Docker CLI was not found. Please install Docker and ensure `docker` is on PATH."
        ) from exc


def probe_docker_environment() -> DockerCheckResult:
    try:
        version_proc = _subprocess_text(["docker", "--version"])
    except DockerRuntimeError as exc:
        return DockerCheckResult(
            ok=False,
            docker_cli_available=False,
            daemon_available=False,
            details=str(exc),
        )

    if version_proc.returncode != 0:
        return DockerCheckResult(
            ok=False,
            docker_cli_available=False,
            daemon_available=False,
            details=(version_proc.stderr or version_proc.stdout or "").strip(),
        )

    info_proc = _subprocess_text(["docker", "info"], timeout=30)
    if info_proc.returncode != 0:
        details = (info_proc.stderr or info_proc.stdout or "").strip()
        if "permission denied" in details.lower() or "docker.sock" in details.lower():
            details = (
                details
                + "\nHint: Docker seems installed, but this process cannot talk to the daemon. "
                "Check Docker daemon status, socket permissions, or whether /var/run/docker.sock "
                "is mounted into the current container."
            ).strip()
        return DockerCheckResult(
            ok=False,
            docker_cli_available=True,
            daemon_available=False,
            details=details,
        )

    return DockerCheckResult(
        ok=True,
        docker_cli_available=True,
        daemon_available=True,
        details=(version_proc.stdout or version_proc.stderr or "").strip(),
    )


def ensure_docker_available() -> DockerCheckResult:
    result = probe_docker_environment()
    if not result.ok:
        raise DockerRuntimeError(result.details or "Docker is not ready.")
    return result


def image_exists(image_name: str) -> bool:
    ensure_docker_available()
    proc = _subprocess_text(["docker", "image", "inspect", image_name], timeout=30)
    return proc.returncode == 0


def build_image(
    image_name: str,
    context_dir: Union[str, Path],
    dockerfile: Optional[Union[str, Path]] = None,
    build_args: Optional[Dict[str, str]] = None,
    platform: Optional[str] = None,
) -> None:
    ensure_docker_available()
    command: List[str] = ["docker", "build", "-t", image_name]
    if dockerfile is not None:
        command.extend(["-f", str(dockerfile)])
    if platform:
        command.extend(["--platform", platform])
    for key, value in sorted((build_args or {}).items()):
        command.extend(["--build-arg", f"{key}={value}"])
    command.append(str(context_dir))
    logger.info("Building Docker image: {}", shlex.join(command))
    proc = subprocess.run(command, text=True)
    if proc.returncode != 0:
        raise DockerRuntimeError(f"Failed to build Docker image {image_name}")


def generate_container_name(image_name: str) -> str:
    process_id = str(os.getpid())
    current_time = str(datetime.datetime.now())
    unique_string = current_time + process_id
    hash_object = hashlib.sha256(unique_string.encode())
    image_name_sanitized = image_name.replace("/", "-").replace(":", "-")
    return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"


def _container_exists(ctr_name: str) -> bool:
    proc = _subprocess_text(["docker", "ps", "-a", "--format", "{{.Names}}"], timeout=20)
    return ctr_name in {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _container_status(ctr_name: str) -> Optional[str]:
    if not _container_exists(ctr_name):
        return None
    proc = _subprocess_text(["docker", "inspect", "-f", "{{.State.Status}}", ctr_name], timeout=20)
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def build_run_command(
    *,
    ctr_name: str,
    image_name: str,
    workspace_root: Optional[Union[str, Path]] = None,
    container_workspace: str = DEFAULT_CONTAINER_WORKSPACE,
    env_file: Optional[Union[str, Path]] = None,
    mount_docker_socket: bool = False,
    persistent: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
) -> List[str]:
    command: List[str] = ["docker", "run", "-i"]
    if persistent:
        command.extend(["-d", "--tty"])
    else:
        command.append("--rm")
    command.extend(["--name", ctr_name])
    if env_file is not None:
        command.extend(["--env-file", str(env_file)])
    if workspace_root is not None:
        command.extend(["-v", f"{Path(workspace_root).resolve()}:{container_workspace}"])
        command.extend(["-w", container_workspace])
    if mount_docker_socket:
        command.extend(["-v", "/var/run/docker.sock:/var/run/docker.sock"])
    for key, value in sorted((extra_env or {}).items()):
        command.extend(["-e", f"{key}={value}"])
    command.append(image_name)
    command.extend(["/bin/bash", "-l"])
    if persistent:
        command.append("-m")
    return command


def get_container(
    ctr_name: str,
    image_name: str,
    *,
    workspace_root: Optional[Union[str, Path]] = None,
    container_workspace: str = DEFAULT_CONTAINER_WORKSPACE,
    env_file: Optional[Union[str, Path]] = None,
    mount_docker_socket: bool = False,
    persistent: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
) -> ContainerBash:
    if not image_exists(image_name):
        raise DockerRuntimeError(
            f"Image {image_name} not found. Please build it first, for example with "
            f"`docker build -t {image_name} .`"
        )

    if persistent:
        return _get_persistent_container(
            ctr_name=ctr_name,
            image_name=image_name,
            workspace_root=workspace_root,
            container_workspace=container_workspace,
            env_file=env_file,
            mount_docker_socket=mount_docker_socket,
            extra_env=extra_env,
        )
    return _get_non_persistent_container(
        ctr_name=ctr_name,
        image_name=image_name,
        workspace_root=workspace_root,
        container_workspace=container_workspace,
        env_file=env_file,
        mount_docker_socket=mount_docker_socket,
        extra_env=extra_env,
    )


def _get_non_persistent_container(
    *,
    ctr_name: str,
    image_name: str,
    workspace_root: Optional[Union[str, Path]],
    container_workspace: str,
    env_file: Optional[Union[str, Path]],
    mount_docker_socket: bool,
    extra_env: Optional[Dict[str, str]],
) -> ContainerBash:
    startup_cmd = build_run_command(
        ctr_name=ctr_name,
        image_name=image_name,
        workspace_root=workspace_root,
        container_workspace=container_workspace,
        env_file=env_file,
        mount_docker_socket=mount_docker_socket,
        persistent=False,
        extra_env=extra_env,
    )
    logger.debug("Starting non-persistent container: {}", shlex.join(startup_cmd))
    container = subprocess.Popen(
        startup_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    time.sleep(DOCKER_START_UP_DELAY)
    output = read_with_timeout(container, lambda: [], timeout_duration=2)
    if output.strip():
        logger.warning("Unexpected container startup output: {}", output)
    return ContainerBash(container, ctr_name, image_name, persistent=False, ctr_pid=1)


def _get_persistent_container(
    *,
    ctr_name: str,
    image_name: str,
    workspace_root: Optional[Union[str, Path]],
    container_workspace: str,
    env_file: Optional[Union[str, Path]],
    mount_docker_socket: bool,
    extra_env: Optional[Dict[str, str]],
) -> ContainerBash:
    status = _container_status(ctr_name)
    if status is None:
        startup_cmd = build_run_command(
            ctr_name=ctr_name,
            image_name=image_name,
            workspace_root=workspace_root,
            container_workspace=container_workspace,
            env_file=env_file,
            mount_docker_socket=mount_docker_socket,
            persistent=True,
            extra_env=extra_env,
        )
        logger.debug("Creating persistent container: {}", shlex.join(startup_cmd))
        proc = _subprocess_text(startup_cmd, timeout=60)
        if proc.returncode != 0:
            raise DockerRuntimeError(proc.stderr or proc.stdout or f"Failed to start {ctr_name}")
    elif status == "paused":
        _subprocess_text(["docker", "unpause", ctr_name], check=True, timeout=30)
    elif status == "exited":
        _subprocess_text(["docker", "start", ctr_name], check=True, timeout=30)
    elif status not in {"running", "created"}:
        raise DockerRuntimeError(f"Unexpected persistent container status: {status}")

    startup_cmd = ["docker", "exec", "-i", ctr_name, "/bin/bash", "-l"]
    logger.debug("Starting bash in persistent container: {}", shlex.join(startup_cmd))
    container = subprocess.Popen(
        startup_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    time.sleep(DOCKER_START_UP_DELAY)
    output = read_with_timeout(container, lambda: [], timeout_duration=2)
    if output.strip():
        logger.warning("Unexpected persistent container startup output: {}", output)

    # The command runner below uses explicit shell sentinels, so persistent
    # sessions do not need procps/ps to be installed in the image. ``ps``-based
    # helpers remain available for diagnostics, but keeping startup independent
    # of them makes the wrapper work with slim SWE-bench-style images too.
    return ContainerBash(container, ctr_name, image_name, persistent=True)


def read_with_timeout(
    container: subprocess.Popen,
    pid_func: Callable[[], List[int] | List[str]],
    timeout_duration: Union[int, float],
) -> str:
    buffer = b""
    assert container.stdout is not None
    fd = container.stdout.fileno()
    end_time = time.time() + timeout_duration

    import select

    def ready_to_read(file_descriptor: int) -> bool:
        return bool(select.select([file_descriptor], [], [], 0.01)[0])

    pids: Iterable[Union[int, str]] = []
    while time.time() < end_time:
        pids = pid_func()
        if len(list(pids)) > 0:
            time.sleep(0.05)
            continue
        if ready_to_read(fd):
            data = os.read(fd, 4096)
            if data:
                buffer += data
        else:
            break
        time.sleep(0.05)

    if container.poll() is not None:
        raise DockerRuntimeError(
            f"Container subprocess exited unexpectedly.\nCurrent buffer: {buffer.decode(errors='replace')}"
        )
    if time.time() >= end_time:
        raise TimeoutError(
            "Timeout while reading from container subprocess.\n"
            f"Current buffer: {buffer.decode(errors='replace')}\nRunning PIDs: {list(pids)}"
        )
    return buffer.decode()


def read_generator_with_timeout(
    container: subprocess.Popen,
    pid_func: Callable[[], List[int]],
    timeout_duration: Union[int, float],
):
    assert container.stdout is not None
    fd = container.stdout.fileno()
    end_time = time.time() + timeout_duration

    import select

    def ready_to_read(file_descriptor: int) -> bool:
        return bool(select.select([file_descriptor], [], [], 0.01)[0])

    execution_finished = False
    pids: List[int] = []
    data = b""
    while time.time() < end_time:
        while ready_to_read(fd):
            new_data = os.read(fd, 4096)
            if new_data:
                data = data + new_data
                try:
                    decoded = data.decode()
                    data = b""
                    yield decoded
                except UnicodeDecodeError:
                    pass
                end_time = time.time() + timeout_duration
            time.sleep(0.05)
        else:
            time.sleep(0.05)
        if execution_finished:
            break
        for i in range(3):
            if i != 0:
                time.sleep(0.05)
            pids = get_int_pids(pid_func())
            execution_finished = len(pids) == 0
            if not execution_finished:
                break

    if container.poll() is not None:
        raise DockerRuntimeError("Container subprocess exited unexpectedly.")
    if time.time() >= end_time:
        raise TimeoutError(f"Timeout while reading from container subprocess. Running PIDs: {pids}")


def get_int_pids(raw_pids: Iterable[Union[int, str]]) -> List[int]:
    result: List[int] = []
    for pid in raw_pids:
        try:
            result.append(int(pid))
        except (TypeError, ValueError):
            continue
    return result


def get_background_pids(ctr_name: str) -> Tuple[List[List[str]], List[List[str]]]:
    proc = _subprocess_text(
        ["docker", "exec", ctr_name, "ps", "-eo", "pid,comm", "--no-headers"],
        timeout=20,
    )
    if proc.returncode != 0:
        raise DockerRuntimeError(proc.stderr or proc.stdout or f"Failed to inspect processes in {ctr_name}")
    pids = [line.split() for line in proc.stdout.splitlines() if line.strip()]
    pids = [entry for entry in pids if entry[1] not in {"ps"} and entry[0] != "1"]
    bash_pids = [entry for entry in pids if entry[1] == "bash"]
    other_pids = [entry for entry in pids if entry[1] != "bash"]
    return bash_pids, other_pids


def get_children_pids(ctr_name: str, parent_pid: int) -> List[int]:
    proc = _subprocess_text(
        ["docker", "exec", ctr_name, "ps", "-o", "pid=", "--ppid", str(parent_pid)],
        timeout=20,
    )
    if proc.returncode != 0:
        raise DockerRuntimeError(proc.stderr or proc.stdout or f"Failed to get child PIDs in {ctr_name}")
    return [int(line.strip()) for line in proc.stdout.splitlines() if line.strip()]


def run_command_in_container(
    ctr_bash: ContainerBash,
    command: str,
    timeout: int = 60,
    output_log: bool = False,
) -> str:
    """Run a command in the container shell and return stdout/stderr text.

    OrcaLoca's original environment tracks child PIDs via ``ps`` to decide when
    a command is finished. CGARF keeps those PID helpers available, but the
    default runner uses an explicit shell sentinel so it also works in minimal
    images that do not include ``procps``.
    """

    assert ctr_bash.ctr_subprocess.stdin is not None
    marker_seed = f"{ctr_bash.ctr_name}:{time.time_ns()}:{os.getpid()}"
    marker = "__CGARF_EXIT_" + hashlib.sha256(marker_seed.encode()).hexdigest()[:16] + "__"
    ctr_bash.ctr_subprocess.stdin.write(
        f"{command}\nprintf '\\n{marker}:%s\\n' \"$?\"\n"
    )
    ctr_bash.ctr_subprocess.stdin.flush()
    if output_log:
        logger.debug("Run command in container {}: {}", ctr_bash.ctr_name, command)
    output, exit_code = read_until_exit_marker(
        ctr_bash.ctr_subprocess,
        marker=marker,
        timeout_duration=timeout,
    )
    ctr_bash.last_exit_code = exit_code
    if output_log and output:
        print(output, end="" if output.endswith("\n") else "\n")
    return output


def get_exit_code(ctr_bash: ContainerBash, timeout: int = 10) -> int:
    if ctr_bash.last_exit_code is not None:
        return ctr_bash.last_exit_code
    assert ctr_bash.ctr_subprocess.stdin is not None
    ctr_bash.ctr_subprocess.stdin.write("echo $?\n")
    ctr_bash.ctr_subprocess.stdin.flush()
    output = read_with_timeout(
        ctr_bash.ctr_subprocess,
        lambda: get_children_pids(ctr_bash.ctr_name, int(ctr_bash.ctr_pid or 1)),
        timeout,
    )
    return int(output.strip().splitlines()[0])


def read_until_exit_marker(
    container: subprocess.Popen,
    *,
    marker: str,
    timeout_duration: Union[int, float],
) -> Tuple[str, int]:
    assert container.stdout is not None
    fd = container.stdout.fileno()
    end_time = time.time() + timeout_duration
    chunks: List[str] = []

    import select

    def ready_to_read(file_descriptor: int) -> bool:
        return bool(select.select([file_descriptor], [], [], 0.05)[0])

    marker_prefix = f"{marker}:"
    while time.time() < end_time:
        if container.poll() is not None:
            raise DockerRuntimeError(
                "Container subprocess exited unexpectedly while waiting for command sentinel."
            )
        if not ready_to_read(fd):
            time.sleep(0.05)
            continue
        data = os.read(fd, 4096)
        if not data:
            time.sleep(0.05)
            continue
        chunks.append(data.decode(errors="replace"))
        text = "".join(chunks)
        if marker_prefix in text:
            before, after = text.split(marker_prefix, 1)
            first_line = after.splitlines()[0].strip()
            try:
                return before, int(first_line)
            except ValueError as exc:
                raise DockerRuntimeError(
                    f"Invalid command sentinel exit code for marker {marker}: {first_line!r}"
                ) from exc

    raise TimeoutError(
        f"Timeout while waiting for command sentinel {marker}.\n"
        f"Current buffer: {''.join(chunks)[-4000:]}"
    )


def get_bash_pid_in_docker(ctr_subprocess: subprocess.Popen) -> int:
    assert ctr_subprocess.stdin is not None
    ctr_subprocess.stdin.write("echo $$\n")
    ctr_subprocess.stdin.flush()
    output = ""
    while output == "":
        output = read_with_timeout(ctr_subprocess, lambda: [], 5)
        time.sleep(0.05)
    return int(output.splitlines()[0])


def copy_text_to_container(
    ctr_name: str,
    contents: str,
    container_path: str,
) -> None:
    container_path_obj = Path(container_path)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
        handle.write(contents)
        temp_path = Path(handle.name)
    try:
        parent = container_path_obj.parent.as_posix()
        _subprocess_text(["docker", "exec", ctr_name, "mkdir", "-p", parent], check=True, timeout=20)
        proc = _subprocess_text(
            ["docker", "cp", str(temp_path), f"{ctr_name}:{container_path_obj.as_posix()}"],
            timeout=30,
        )
        if proc.returncode != 0:
            raise DockerRuntimeError(proc.stderr or proc.stdout or f"Failed to copy file into {ctr_name}")
    finally:
        temp_path.unlink(missing_ok=True)


def copy_file_from_container(ctr_name: str, container_path: str, host_path: Union[str, Path]) -> None:
    proc = _subprocess_text(["docker", "cp", f"{ctr_name}:{container_path}", str(host_path)], timeout=60)
    if proc.returncode != 0:
        raise DockerRuntimeError(proc.stderr or proc.stdout or f"Failed to copy {container_path} from {ctr_name}")


def pause_persistent_container(ctr_bash: ContainerBash) -> None:
    if not ctr_bash.persistent:
        return
    status = _container_status(ctr_bash.ctr_name)
    if status not in {"paused", "exited", "dead", "stopping", None}:
        proc = _subprocess_text(["docker", "pause", ctr_bash.ctr_name], timeout=30)
        if proc.returncode == 0:
            logger.info("Paused container {}", ctr_bash.ctr_name)


def stop_container(ctr_bash: ContainerBash, remove: bool = False) -> None:
    try:
        if ctr_bash.ctr_subprocess.stdin is not None:
            ctr_bash.ctr_subprocess.stdin.close()
    except Exception:
        pass
    try:
        ctr_bash.ctr_subprocess.terminate()
    except Exception:
        pass

    status = _container_status(ctr_bash.ctr_name)
    if status is None:
        return

    if remove:
        _subprocess_text(["docker", "rm", "-f", ctr_bash.ctr_name], timeout=30)
        return

    if ctr_bash.persistent:
        pause_persistent_container(ctr_bash)
