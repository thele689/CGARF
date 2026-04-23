"""Official SWE-bench Docker harness integration for CGARF TSPF.

CGARF's SRCD stages emit localized Search/Replace patches. The official
SWE-bench harness expects unified-diff predictions and evaluates them in Docker
containers. This module bridges that gap:

1. flatten distilled CGARF patches,
2. convert each Search/Replace patch to a unified diff against a local checkout,
3. write official SWE-bench predictions JSONL,
4. invoke ``swebench.harness.run_evaluation`` in a Python >=3.10 process,
5. read harness reports back into TSPF-compatible test evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from src.tspf.agentless_adapted_tests import extract_tspf_evidence
from src.tspf.patch_filter import TwoStagePatchFilter


PATCH_RE = re.compile(
    r"<<<\s*SEARCH\n(.*?)\n===\n(.*?)\n>>>\s*REPLACE",
    re.DOTALL,
)


@dataclass
class HarnessDependencyStatus:
    """Dependency check result for the official Docker harness."""

    python_executable: str
    ok: bool
    missing_modules: List[str] = field(default_factory=list)
    docker_available: bool = False
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_executable": self.python_executable,
            "ok": self.ok,
            "missing_modules": self.missing_modules,
            "docker_available": self.docker_available,
            "details": self.details,
        }


@dataclass
class UnifiedPatchCandidate:
    """One CGARF candidate converted to official SWE-bench prediction format."""

    patch_id: str
    instance_id: str
    model_name_or_path: str
    model_patch: str
    candidate_id: str
    candidate_location: str
    source_patch_content: str
    conversion_status: str
    conversion_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def prediction_dict(self) -> Dict[str, str]:
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = self.prediction_dict()
        data.update(
            {
                "patch_id": self.patch_id,
                "candidate_id": self.candidate_id,
                "candidate_location": self.candidate_location,
                "source_patch_content": self.source_patch_content,
                "conversion_status": self.conversion_status,
                "conversion_reason": self.conversion_reason,
                "metadata": self.metadata,
            }
        )
        return data


class SearchReplaceToUnifiedDiffConverter:
    """Convert CGARF Search/Replace patches into unified diffs."""

    def convert(
        self,
        repo_root: Path,
        patch_content: str,
        candidate_location: str = "",
    ) -> Tuple[Optional[str], str, str]:
        match = PATCH_RE.search(patch_content or "")
        if not match:
            return None, "invalid_search_replace_format", "No Search/Replace block found"

        search_block = match.group(1)
        replace_block = match.group(2)
        if search_block.strip() == replace_block.strip():
            return None, "no_op_search_replace", "SEARCH and REPLACE blocks are identical"

        target_files = self._candidate_target_files(repo_root, candidate_location)
        if not target_files:
            target_files = list(repo_root.rglob("*.py"))

        for file_path in target_files:
            try:
                original = file_path.read_text()
            except UnicodeDecodeError:
                continue
            if search_block not in original:
                continue
            updated = original.replace(search_block, replace_block, 1)
            rel = file_path.relative_to(repo_root).as_posix()
            diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    updated.splitlines(keepends=True),
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                )
            )
            if diff:
                return diff, "converted", f"Converted against {rel}"
            return None, "empty_unified_diff", f"Search/Replace produced no diff for {rel}"

        return None, "search_block_not_found", "SEARCH block was not found in candidate file(s)"

    def _candidate_target_files(self, repo_root: Path, candidate_location: str) -> List[Path]:
        if not candidate_location:
            return []
        raw_path = candidate_location.split("::", 1)[0]
        raw = Path(raw_path)

        candidates: List[Path] = []
        if raw.is_absolute():
            parts = list(raw.parts)
            for marker in ("repos", repo_root.name):
                if marker in parts:
                    idx = parts.index(marker)
                    rel_parts = parts[idx + 2 :] if marker == "repos" else parts[idx + 1 :]
                    if rel_parts:
                        candidate = repo_root.joinpath(*rel_parts)
                        if candidate.exists():
                            candidates.append(candidate)
            candidates.extend(repo_root.rglob(raw.name))
        else:
            candidate = repo_root / raw
            if candidate.exists():
                candidates.append(candidate)
            candidates.extend(repo_root.rglob(raw.name))

        seen = set()
        unique: List[Path] = []
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append(candidate)
        return unique


class OfficialSWEbenchHarnessAdapter:
    """Run CGARF candidate patches through the official SWE-bench harness."""

    def __init__(
        self,
        workspace_root: Path,
        repo_root: Path,
        swebench_root: Path,
        python_executable: str = "python3.10",
        dataset_name: str = "SWE-bench/SWE-bench_Lite",
        split: str = "test",
    ):
        self.workspace_root = workspace_root
        self.repo_root = repo_root
        self.swebench_root = swebench_root
        self.python_executable = python_executable
        self.dataset_name = dataset_name
        self.split = split
        self.converter = SearchReplaceToUnifiedDiffConverter()

    def check_dependencies(self) -> HarnessDependencyStatus:
        script = """
import importlib
import json
import sys
from src.environment.utils import probe_docker_environment
missing = []
import_errors = {}
for name in ["docker", "datasets", "swebench", "unidiff", "ghapi", "jsonlines", "loguru"]:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(name)
        import_errors[name] = repr(exc)
docker_probe = probe_docker_environment().to_dict()
docker_ok = bool(docker_probe.get("daemon_available", False))
details = docker_probe.get("details", "")
print(json.dumps({
    "missing": missing,
    "import_errors": import_errors,
    "docker_ok": docker_ok,
    "details": details,
    "docker_probe": docker_probe,
}))
sys.exit(0 if not missing and docker_ok else 1)
""".strip()
        env = self._subprocess_env()
        proc = subprocess.run(
            [self.python_executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
        )
        output = (proc.stdout or proc.stderr or "").strip()
        missing: List[str] = []
        docker_ok = False
        details = output
        try:
            parsed = json.loads(output) if output else {}
            missing = list(parsed.get("missing", []))
            docker_ok = bool(parsed.get("docker_ok", False))
            details = json.dumps(
                {
                    "docker": parsed.get("details", ""),
                    "docker_probe": parsed.get("docker_probe", {}),
                    "import_errors": parsed.get("import_errors", {}),
                },
                ensure_ascii=False,
            )
        except Exception:
            pass
        return HarnessDependencyStatus(
            python_executable=self.python_executable,
            ok=proc.returncode == 0,
            missing_modules=missing,
            docker_available=docker_ok,
            details=details,
        )

    def prepare_predictions_from_distillation(
        self,
        instance_id: str,
        distillation_payload: Dict[str, Any],
        output_jsonl: Path,
        max_patches: Optional[int] = None,
        model_prefix: str = "CGARF",
    ) -> List[UnifiedPatchCandidate]:
        patches = self._flatten_distilled_patches(distillation_payload)
        if max_patches is not None:
            patches = patches[:max_patches]

        converted: List[UnifiedPatchCandidate] = []
        for index, patch in enumerate(patches, start=1):
            patch_id = str(patch.get("patch_id") or f"{instance_id}::patch::{index}")
            model_name = self._model_name_for_patch(model_prefix, patch_id, index)
            patch_content = str(patch.get("patch_content") or "")
            candidate_location = str(patch.get("candidate_location") or "")
            diff, status, reason = self.converter.convert(
                repo_root=self.repo_root,
                patch_content=patch_content,
                candidate_location=candidate_location,
            )
            converted.append(
                UnifiedPatchCandidate(
                    patch_id=patch_id,
                    instance_id=instance_id,
                    model_name_or_path=model_name,
                    model_patch=diff or "",
                    candidate_id=str(patch.get("candidate_id") or ""),
                    candidate_location=candidate_location,
                    source_patch_content=patch_content,
                    conversion_status=status,
                    conversion_reason=reason,
                    metadata={
                        "distillation_score": patch.get("distillation_score"),
                        "causality_score": patch.get("causality_score"),
                        "reflection_score": patch.get("reflection_score"),
                        "generated_round": patch.get("generated_round"),
                    },
                )
            )

        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w") as handle:
            for item in converted:
                if item.conversion_status == "converted":
                    handle.write(json.dumps(item.prediction_dict(), ensure_ascii=False) + "\n")
        return converted

    def run_harness(
        self,
        predictions_jsonl: Path,
        run_id: str,
        instance_ids: Optional[List[str]] = None,
        max_workers: int = 1,
        timeout: int = 1800,
        namespace: Optional[str] = None,
        cache_level: str = "env",
        clean: bool = False,
        force_rebuild: bool = False,
        report_dir: Optional[Path] = None,
        local_repo_bundle: Optional[Path] = None,
    ) -> subprocess.CompletedProcess:
        command = [
            self.python_executable,
            "-m",
            "src.tspf.swebench_local_repo_runner"
            if local_repo_bundle is not None
            else "swebench.harness.run_evaluation",
            "--dataset_name",
            self.dataset_name,
            "--split",
            self.split,
            "--predictions_path",
            str(predictions_jsonl),
            "--max_workers",
            str(max_workers),
            "--timeout",
            str(timeout),
            "--run_id",
            run_id,
            "--cache_level",
            cache_level,
            "--clean",
            str(clean).lower(),
            "--force_rebuild",
            str(force_rebuild).lower(),
            "--namespace",
            "none" if namespace is None else namespace,
        ]
        if instance_ids:
            command.extend(["--instance_ids", *instance_ids])
        if report_dir is not None:
            command.extend(["--report_dir", str(report_dir)])

        logger.info("Running official SWE-bench harness: {}", " ".join(command))
        env = self._subprocess_env()
        if local_repo_bundle is not None:
            env["CGARF_SWEBENCH_LOCAL_REPO_BUNDLE"] = str(local_repo_bundle)
        return subprocess.run(
            command,
            cwd=str(self.workspace_root),
            env=env,
            text=True,
            capture_output=True,
        )

    def create_local_repo_bundle(self, output_bundle: Path) -> Path:
        """Create a git bundle used as an offline clone source inside Docker."""

        output_bundle.parent.mkdir(parents=True, exist_ok=True)
        if output_bundle.exists():
            output_bundle.unlink()
        subprocess.run(
            ["git", "-C", str(self.repo_root), "bundle", "create", str(output_bundle), "--all"],
            check=True,
            text=True,
            capture_output=True,
        )
        return output_bundle

    def build_tspf_evidence_from_reports(
        self,
        converted_candidates: List[UnifiedPatchCandidate],
        run_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        evidence: Dict[str, Dict[str, Any]] = {}
        for candidate in converted_candidates:
            if candidate.conversion_status != "converted":
                evidence[candidate.patch_id] = self._failed_conversion_evidence(candidate)
                continue
            report_path = (
                self.workspace_root
                / "logs"
                / "run_evaluation"
                / run_id
                / candidate.model_name_or_path.replace("/", "__")
                / candidate.instance_id
                / "report.json"
            )
            if not report_path.exists():
                evidence[candidate.patch_id] = self._missing_report_evidence(candidate, report_path)
                continue
            report = json.loads(report_path.read_text())
            instance_report = report.get(candidate.instance_id, {})
            tests_status = instance_report.get("tests_status", {})
            fail_to_pass = tests_status.get("FAIL_TO_PASS", {})
            pass_to_pass = tests_status.get("PASS_TO_PASS", {})
            reproduction_total, reproduction_passed = self._count_passed(fail_to_pass)
            regression_total, regression_passed = self._count_passed(pass_to_pass)
            resolved = bool(instance_report.get("resolved", False))
            if reproduction_total == 0:
                reproduction_total = 1
                reproduction_passed = 1 if resolved else 0

            evidence[candidate.patch_id] = {
                "official_harness": {
                    "report_path": str(report_path),
                    "resolved": resolved,
                    "model_name_or_path": candidate.model_name_or_path,
                },
                "regression": {
                    "total_tests": regression_total,
                    "passed_tests": regression_passed,
                    "pass_rate": regression_passed / regression_total if regression_total else 1.0,
                    "tests_status": pass_to_pass,
                },
                "reproduction": {
                    "total_tests": reproduction_total,
                    "passed_tests": reproduction_passed,
                    "pass_rate": reproduction_passed / reproduction_total if reproduction_total else 0.0,
                    "tests_status": fail_to_pass,
                },
            }
        return evidence

    def select_final_patch(
        self,
        distillation_payload: Dict[str, Any],
        evidence: Dict[str, Dict[str, Any]],
        mu: float = 0.6,
        max_patches: Optional[int] = 5,
    ) -> Dict[str, Any]:
        return TwoStagePatchFilter(mu=mu, require_test_evidence=True).filter_distillation_payload(
            distillation_payload,
            test_evidence=extract_tspf_evidence(evidence),
            max_patches=max_patches,
        )

    def _subprocess_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        project_root = Path(__file__).resolve().parents[2]
        compat_dir = project_root / "tools" / "python310_compat"
        pythonpath: List[str] = []
        if compat_dir.exists():
            pythonpath.append(str(compat_dir))
        pythonpath.extend([str(self.swebench_root), str(project_root), str(self.workspace_root)])
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        return env

    def _flatten_distilled_patches(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        patches: List[Dict[str, Any]] = []
        for candidate_id, candidate_result in payload.get("candidate_results", {}).items():
            candidate_location = str(candidate_result.get("candidate_location", candidate_id))
            for patch in candidate_result.get("ranked_patches", []):
                item = dict(patch)
                item.setdefault("candidate_id", candidate_id)
                item.setdefault("candidate_location", candidate_location)
                patches.append(item)
        patches.sort(key=lambda item: float(item.get("distillation_score") or 0.0), reverse=True)
        return patches

    def _model_name_for_patch(self, prefix: str, patch_id: str, index: int) -> str:
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", patch_id)[-120:]
        return f"{prefix}_{index}_{suffix}"

    def _count_passed(self, status_obj: Any) -> Tuple[int, int]:
        if not isinstance(status_obj, dict):
            return 0, 0
        success = status_obj.get("success", [])
        failure = status_obj.get("failure", [])
        error = status_obj.get("error", [])
        total = len(success) + len(failure) + len(error)
        return total, len(success)

    def _failed_conversion_evidence(self, candidate: UnifiedPatchCandidate) -> Dict[str, Any]:
        result = {
            "total_tests": 1,
            "passed_tests": 0,
            "pass_rate": 0.0,
            "conversion_status": candidate.conversion_status,
            "conversion_reason": candidate.conversion_reason,
        }
        return {
            "official_harness": {"resolved": False, "conversion_failed": True},
            "regression": result,
            "reproduction": dict(result),
        }

    def _missing_report_evidence(
        self,
        candidate: UnifiedPatchCandidate,
        report_path: Path,
    ) -> Dict[str, Any]:
        result = {
            "total_tests": 1,
            "passed_tests": 0,
            "pass_rate": 0.0,
            "missing_report": str(report_path),
        }
        return {
            "official_harness": {"resolved": False, "report_path": str(report_path)},
            "regression": result,
            "reproduction": dict(result),
        }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
