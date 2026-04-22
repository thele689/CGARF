"""Agentless-style regression/reproduction testing for TSPF.

This module adapts the Agentless testing idea to CGARF's current patch format.
Agentless evaluates SWE-bench patches with Docker and unified diffs; CGARF
currently produces localized Search/Replace patches. The implementation below
keeps the same testing semantics while making the execution path usable inside
this project:

1. verify a reproduction script on the unpatched repository,
2. apply each candidate Search/Replace patch in an isolated worktree copy,
3. run reproduction tests and regression tests,
4. emit the test-evidence JSON consumed by ``TwoStagePatchFilter``.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from src.common.llm_interface import LLMInterface
from src.tspf.patch_filter import TwoStagePatchFilter


AGENTLESS_REPRODUCTION_PROMPT = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Please generate a complete Python test that can be used to reproduce the issue.

The complete test should contain the following:
1. Necessary imports.
2. Code to reproduce the issue described in the issue text.
3. Print "Issue reproduced" if the outcome indicates that the issue is reproduced.
4. Print "Issue resolved" if the outcome indicates that the issue has been successfully resolved.
5. Print "Other issues" if the outcome indicates there are other issues with the source code.

The generated test should be usable both on the original repository and after applying a candidate patch.
Wrap the complete test in ```python ... ```.
""".strip()


@dataclass
class SearchReplacePatch:
    """Parsed CGARF Search/Replace patch."""

    search: str
    replace: str

    @property
    def is_no_op(self) -> bool:
        return self.search.strip() == self.replace.strip()


@dataclass
class CommandResult:
    """Result of running one test command/script."""

    name: str
    command: List[str]
    passed: bool
    returncode: int
    stdout: str
    stderr: str
    expected_output: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "passed": self.passed,
            "returncode": self.returncode,
            "stdout": self.stdout[-4000:],
            "stderr": self.stderr[-4000:],
            "expected_output": self.expected_output,
        }


@dataclass
class PatchApplicationResult:
    """Result of applying a Search/Replace patch to a local worktree."""

    applied: bool
    file_path: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied": self.applied,
            "file_path": self.file_path,
            "reason": self.reason,
        }


@dataclass
class EvidenceBuildResult:
    """TSPF evidence payload plus execution metadata."""

    instance_id: str
    evidence: Dict[str, Dict[str, Any]]
    reproduction_script: str
    regression_scripts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "evidence": self.evidence,
            "reproduction_script": self.reproduction_script,
            "regression_scripts": self.regression_scripts,
            "metadata": self.metadata,
        }


def extract_first_code_block(text: str) -> Optional[str]:
    """Extract the first fenced Python code block from an LLM response."""

    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def create_patch_from_code(python_code: str) -> str:
    """Create the Agentless-style reproduce_bug.py unified diff patch."""

    patch_header = (
        "diff --git a/reproduce_bug.py b/reproduce_bug.py\n"
        "new file mode 100644\n"
        "index 0000000..e69de29\n"
    )
    lines = python_code.splitlines()
    body = ["--- /dev/null", "+++ b/reproduce_bug.py", f"@@ -0,0 +1,{len(lines)} @@"]
    body.extend(f"+{line}" for line in lines)
    return patch_header + "\n".join(body) + "\n"


class ReproductionTestGenerator:
    """Generate and verify Agentless-style reproduction tests."""

    def __init__(self, llm: Optional[LLMInterface] = None):
        self.llm = llm

    def generate(self, issue_text: str) -> Tuple[str, Dict[str, Any]]:
        """Generate a reproduction script from issue text.

        If an LLM is supplied, use the Agentless prompt. Otherwise use a
        deterministic issue-derived fallback for known SWE-bench-style reports.
        """

        if self.llm is not None:
            prompt = AGENTLESS_REPRODUCTION_PROMPT.format(problem_statement=issue_text)
            raw = self.llm.generate(prompt, temperature=0.0, max_tokens=1800)
            code = extract_first_code_block(raw) or raw.strip()
            return code, {
                "source": "llm",
                "prompt_template": "agentless_generate_reproduction_tests",
                "raw_response": raw,
            }

        fallback = self._fallback_from_issue(issue_text)
        if fallback:
            return fallback, {
                "source": "deterministic_fallback",
                "reason": "issue contains a directly executable nested separability example",
            }
        raise ValueError("No LLM supplied and no deterministic reproduction fallback is available")

    def _fallback_from_issue(self, issue_text: str) -> Optional[str]:
        if "separability_matrix" not in issue_text or "Pix2Sky_TAN" not in issue_text:
            return None
        return ASTROPY_SEPARABILITY_REPRODUCTION_SCRIPT


class SearchReplacePatchApplier:
    """Apply CGARF Search/Replace patches to isolated repository copies."""

    PATCH_RE = re.compile(
        r"<<<\s*SEARCH\n(.*?)\n===\n(.*?)\n>>>\s*REPLACE",
        re.DOTALL,
    )

    def parse(self, patch_content: str) -> Optional[SearchReplacePatch]:
        match = self.PATCH_RE.search(patch_content or "")
        if not match:
            return None
        return SearchReplacePatch(match.group(1), match.group(2))

    def apply(
        self,
        worktree: Path,
        patch_content: str,
        candidate_location: str = "",
    ) -> PatchApplicationResult:
        parsed = self.parse(patch_content)
        if parsed is None:
            return PatchApplicationResult(False, reason="invalid_search_replace_format")
        if parsed.is_no_op:
            return PatchApplicationResult(False, reason="no_op_search_replace")

        target_files = self._candidate_target_files(worktree, candidate_location)
        if not target_files:
            target_files = list(worktree.rglob("*.py"))

        for file_path in target_files:
            try:
                original = file_path.read_text()
            except UnicodeDecodeError:
                continue
            if parsed.search in original:
                updated = original.replace(parsed.search, parsed.replace, 1)
                file_path.write_text(updated)
                return PatchApplicationResult(
                    True,
                    file_path=str(file_path.relative_to(worktree)),
                    reason="applied_exact_search_replace",
                )

        return PatchApplicationResult(False, reason="search_block_not_found")

    def _candidate_target_files(self, worktree: Path, candidate_location: str) -> List[Path]:
        if not candidate_location:
            return []
        raw_path = candidate_location.split("::", 1)[0]
        raw = Path(raw_path)

        candidates: List[Path] = []
        if raw.is_absolute():
            parts = list(raw.parts)
            for marker in ("repos", "astropy_astropy"):
                if marker in parts:
                    idx = parts.index(marker)
                    rel_parts = parts[idx + 2 :] if marker == "repos" else parts[idx + 1 :]
                    if rel_parts:
                        candidate = worktree.joinpath(*rel_parts)
                        if candidate.exists():
                            candidates.append(candidate)
            if raw.name:
                candidates.extend(worktree.rglob(raw.name))
        else:
            candidate = worktree / raw
            if candidate.exists():
                candidates.append(candidate)
            if raw.name:
                candidates.extend(worktree.rglob(raw.name))

        seen = set()
        unique: List[Path] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen and candidate.is_file():
                seen.add(resolved)
                unique.append(candidate)
        return unique


class LocalAgentlessTestEvidenceRunner:
    """Build regression/reproduction evidence for CGARF candidate patches."""

    def __init__(
        self,
        instance_id: str,
        repo_root: Path,
        issue_text: str,
        llm: Optional[LLMInterface] = None,
        python_executable: str = sys.executable,
        timeout: int = 120,
        work_root: Optional[Path] = None,
    ):
        self.instance_id = instance_id
        self.repo_root = repo_root
        self.issue_text = issue_text
        self.llm = llm
        self.python_executable = python_executable
        self.timeout = timeout
        self.work_root = work_root
        self.applier = SearchReplacePatchApplier()

    def build_from_distillation(
        self,
        distillation_payload: Dict[str, Any],
        max_patches: Optional[int] = None,
        reproduction_script: Optional[str] = None,
        regression_scripts: Optional[List[str]] = None,
    ) -> EvidenceBuildResult:
        patches = self._flatten_distilled_patches(distillation_payload)
        if max_patches is not None:
            patches = patches[:max_patches]

        generator_metadata: Dict[str, Any] = {}
        if reproduction_script is None:
            reproduction_script, generator_metadata = ReproductionTestGenerator(self.llm).generate(
                self.issue_text
            )
        if regression_scripts is None:
            regression_scripts = self._default_regression_scripts()

        base_reproduction = self._run_on_repo_copy(
            name="base_reproduction_verification",
            script=reproduction_script,
            expected_output="Issue reproduced",
            patch=None,
        )
        base_reproduction_verified = base_reproduction.passed

        evidence: Dict[str, Dict[str, Any]] = {}
        for patch in patches:
            patch_id = str(patch.get("patch_id") or patch.get("id") or "")
            patch_content = str(patch.get("patch_content") or "")
            candidate_location = str(patch.get("candidate_location") or "")

            patch_application, reproduction_result, regression_results = self._evaluate_patch(
                patch_id=patch_id,
                patch_content=patch_content,
                candidate_location=candidate_location,
                reproduction_script=reproduction_script,
                regression_scripts=regression_scripts,
                base_reproduction_verified=base_reproduction_verified,
            )
            regression_passed = sum(1 for item in regression_results if item.passed)
            reproduction_passed = 1 if reproduction_result.passed else 0
            evidence[patch_id] = {
                "patch_apply": patch_application.to_dict(),
                "regression": {
                    "total_tests": len(regression_results),
                    "passed_tests": regression_passed,
                    "pass_rate": regression_passed / len(regression_results)
                    if regression_results else 0.0,
                    "tests": [item.to_dict() for item in regression_results],
                },
                "reproduction": {
                    "total_tests": 1,
                    "passed_tests": reproduction_passed,
                    "pass_rate": float(reproduction_passed),
                    "base_reproduction_verified": base_reproduction_verified,
                    "test": reproduction_result.to_dict(),
                },
            }

        return EvidenceBuildResult(
            instance_id=self.instance_id,
            evidence=evidence,
            reproduction_script=reproduction_script,
            regression_scripts=regression_scripts,
            metadata={
                "repo_root": str(self.repo_root),
                "python_executable": self.python_executable,
                "timeout": self.timeout,
                "reproduction_generator": generator_metadata,
                "base_reproduction_verification": base_reproduction.to_dict(),
                "patch_count": len(patches),
            },
        )

    def _evaluate_patch(
        self,
        patch_id: str,
        patch_content: str,
        candidate_location: str,
        reproduction_script: str,
        regression_scripts: List[str],
        base_reproduction_verified: bool,
    ) -> Tuple[PatchApplicationResult, CommandResult, List[CommandResult]]:
        with self._temporary_repo_copy() as worktree:
            application = self.applier.apply(worktree, patch_content, candidate_location)
            if not application.applied:
                reproduction_result = CommandResult(
                    name="reproduction",
                    command=[],
                    passed=False,
                    returncode=1,
                    stdout="",
                    stderr=application.reason,
                    expected_output="Issue resolved",
                )
                regression_results = [
                    CommandResult(
                        name=f"regression_{idx}",
                        command=[],
                        passed=False,
                        returncode=1,
                        stdout="",
                        stderr=application.reason,
                    )
                    for idx, _ in enumerate(regression_scripts, start=1)
                ]
                return application, reproduction_result, regression_results

            reproduction_result = self._run_script(
                worktree,
                name="reproduction",
                script=reproduction_script,
                expected_output="Issue resolved" if base_reproduction_verified else None,
                disallowed_outputs=["Issue reproduced", "Other issues"],
            )
            regression_results = [
                self._run_script(
                    worktree,
                    name=f"regression_{idx}",
                    script=script,
                    expected_output=None,
                    disallowed_outputs=[],
                )
                for idx, script in enumerate(regression_scripts, start=1)
            ]
            return application, reproduction_result, regression_results

    def _run_on_repo_copy(
        self,
        name: str,
        script: str,
        expected_output: Optional[str],
        patch: Optional[Tuple[str, str, str]],
    ) -> CommandResult:
        with self._temporary_repo_copy() as worktree:
            if patch is not None:
                patch_content, candidate_location, _patch_id = patch
                application = self.applier.apply(worktree, patch_content, candidate_location)
                if not application.applied:
                    return CommandResult(
                        name=name,
                        command=[],
                        passed=False,
                        returncode=1,
                        stdout="",
                        stderr=application.reason,
                        expected_output=expected_output,
                    )
            return self._run_script(
                worktree,
                name=name,
                script=script,
                expected_output=expected_output,
                disallowed_outputs=[],
            )

    def _run_script(
        self,
        worktree: Path,
        name: str,
        script: str,
        expected_output: Optional[str],
        disallowed_outputs: Iterable[str],
    ) -> CommandResult:
        script_path = worktree / f".cgarf_{name}.py"
        script_path.write_text(script)
        command = [self.python_executable, str(script_path)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("PYTHONWARNINGS", "ignore")
        try:
            completed = subprocess.run(
                command,
                cwd=str(worktree),
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            passed = completed.returncode == 0
            if expected_output:
                passed = passed and expected_output in stdout
            for marker in disallowed_outputs:
                if marker and marker in stdout:
                    passed = False
            return CommandResult(
                name=name,
                command=command,
                passed=passed,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                expected_output=expected_output,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                name=name,
                command=command,
                passed=False,
                returncode=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\nTimeoutExpired",
                expected_output=expected_output,
            )
        finally:
            try:
                script_path.unlink()
            except FileNotFoundError:
                pass

    def _temporary_repo_copy(self):
        return _TemporaryRepoCopy(self.repo_root, self.work_root)

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

    def _default_regression_scripts(self) -> List[str]:
        if "separability_matrix" in self.issue_text and "Pix2Sky_TAN" in self.issue_text:
            return [ASTROPY_SEPARABILITY_REGRESSION_SCRIPT]
        logger.warning("No default regression script could be inferred for {}", self.instance_id)
        return []


class _TemporaryRepoCopy:
    """Context manager for a copy of the repository without the .git directory."""

    def __init__(self, repo_root: Path, work_root: Optional[Path] = None):
        self.repo_root = repo_root
        self.work_root = work_root
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.worktree: Optional[Path] = None

    def __enter__(self) -> Path:
        self.temp_dir = tempfile.TemporaryDirectory(dir=str(self.work_root) if self.work_root else None)
        self.worktree = Path(self.temp_dir.name) / self.repo_root.name
        ignore = shutil.ignore_patterns(
            ".git",
            "__pycache__",
            ".pytest_cache",
            "build",
            "dist",
            "*.egg-info",
        )
        shutil.copytree(str(self.repo_root), str(self.worktree), ignore=ignore)
        return self.worktree

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def write_evidence_json(result: EvidenceBuildResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


def extract_tspf_evidence(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return just the mapping expected by TwoStagePatchFilter."""

    if "evidence" in payload and isinstance(payload["evidence"], dict):
        return payload["evidence"]
    return payload


def run_tspf_with_evidence(
    distillation_payload: Dict[str, Any],
    evidence_payload: Dict[str, Any],
    mu: float = 0.6,
    max_patches: Optional[int] = 5,
) -> Dict[str, Any]:
    tspf = TwoStagePatchFilter(mu=mu, require_test_evidence=True)
    return tspf.filter_distillation_payload(
        distillation_payload,
        test_evidence=extract_tspf_evidence(evidence_payload),
        max_patches=max_patches,
    )


ASTROPY_SEPARABILITY_TEST_PREAMBLE = r'''
from pathlib import Path
import re

import numpy as np


class ModelDefinitionError(Exception):
    pass


class Model:
    n_inputs = 1
    n_outputs = 1
    separable = True

    def _calculate_separability_matrix(self):
        return NotImplemented

    def __and__(self, other):
        return CompoundModel("&", self, other)


class CompoundModel(Model):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
        if op == "&":
            self.n_inputs = left.n_inputs + right.n_inputs
            self.n_outputs = left.n_outputs + right.n_outputs
        else:
            raise NotImplementedError(op)
        self.separable = False


class Mapping(Model):
    def __init__(self, mapping):
        self.mapping = tuple(mapping)
        self.n_inputs = max(self.mapping) + 1 if self.mapping else 0
        self.n_outputs = len(self.mapping)
        self.separable = True


class Linear1D(Model):
    n_inputs = 1
    n_outputs = 1
    separable = True

    def __init__(self, slope):
        self.slope = slope


class Pix2Sky_TAN(Model):
    n_inputs = 2
    n_outputs = 2
    separable = False


def load_separable_matrix():
    source_path = Path("astropy/modeling/separable.py")
    source = source_path.read_text()
    source = re.sub(r"from \.core import .*\n", "", source)
    source = re.sub(r"from \.mappings import .*\n", "", source)
    ns = {
        "np": np,
        "Model": Model,
        "ModelDefinitionError": ModelDefinitionError,
        "CompoundModel": CompoundModel,
        "Mapping": Mapping,
        "__name__": "cgarf_separable_under_test",
    }
    exec(compile(source, str(source_path), "exec"), ns)
    return ns["separability_matrix"]
'''.strip()


ASTROPY_SEPARABILITY_REPRODUCTION_SCRIPT = ASTROPY_SEPARABILITY_TEST_PREAMBLE + "\n\n" + r'''


def main():
    separability_matrix = load_separable_matrix()
    cm = Linear1D(10) & Linear1D(5)
    expected = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ],
        dtype=bool,
    )
    try:
        actual = separability_matrix(Pix2Sky_TAN() & cm)
    except Exception:
        print("Other issues")
        return
    if np.array_equal(actual, expected):
        print("Issue resolved")
    else:
        print("Issue reproduced")


if __name__ == "__main__":
    main()
'''.strip()


ASTROPY_SEPARABILITY_REGRESSION_SCRIPT = ASTROPY_SEPARABILITY_TEST_PREAMBLE + "\n\n" + r'''

def assert_matrix_equal(actual, expected):
    if not np.array_equal(actual, expected):
        raise AssertionError("matrix mismatch:\nactual={}\nexpected={}".format(actual, expected))


separability_matrix = load_separable_matrix()

cm = Linear1D(10) & Linear1D(5)
assert_matrix_equal(
    separability_matrix(cm),
    np.array([[True, False], [False, True]], dtype=bool),
)

assert_matrix_equal(
    separability_matrix(Pix2Sky_TAN() & Linear1D(10) & Linear1D(5)),
    np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ],
        dtype=bool,
    ),
)

print("Regression passed")
'''.strip()
