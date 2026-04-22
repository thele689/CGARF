"""
ModuleM: Patch Filtering and Evaluation
========================================

Executes synthesized tests against repair candidates and filters/ranks patches
based on test results.

Workflow:
  1. Execute test suite for each repair
  2. Analyze test results (pass/fail/error)
  3. Score repairs based on test success
  4. Filter and rank final patch candidates
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import tempfile
import os
import re
import math
from collections import Counter
from loguru import logger

from src.common.data_structures import RepairCandidate
from src.tspf.test_synthesizer import TestStructure, TestCase


class TestStatus(Enum):
    """Status of a test execution"""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class PatchStatus(Enum):
    """Status of a patch after testing"""
    VERIFIED = "verified"  # All tests passed
    PARTIAL = "partial"  # Some tests passed
    FAILING = "failing"  # Most tests failed
    ERROR = "error"  # Execution error


@dataclass
class TestResult:
    """Result from a single test execution"""
    test_name: str
    test_status: TestStatus
    message: str = ""
    execution_time: float = 0.0
    error_trace: str = ""

    @property
    def passed(self) -> bool:
        return self.test_status == TestStatus.PASSED


@dataclass
class TestSuiteResult:
    """Results from complete test suite execution"""
    repair_id: str
    test_results: List[TestResult]
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    execution_time: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Percentage of tests that passed"""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests

    @property
    def patch_status(self) -> PatchStatus:
        """Overall patch status based on test results"""
        if self.passed_tests == self.total_tests:
            return PatchStatus.VERIFIED
        elif self.passed_tests > self.total_tests * 0.5:
            return PatchStatus.PARTIAL
        elif self.error_tests > 0:
            return PatchStatus.ERROR
        else:
            return PatchStatus.FAILING


@dataclass
class VerifiedPatch:
    """Patch that has passed test validation"""
    repair: RepairCandidate
    test_results: TestSuiteResult
    verification_score: float  # [0, 1]
    confidence: float  # [0, 1]
    failures: List[str] = field(default_factory=list)
    success_details: str = ""

    def __lt__(self, other: 'VerifiedPatch') -> bool:
        """Compare by verification score (higher is better)"""
        return self.verification_score > other.verification_score


class TestExecutor:
    """Executes test suites for repair patches"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout  # seconds

    def execute_test_structure(
        self, test_struct: TestStructure
    ) -> TestSuiteResult:
        """
        Execute test structure and collect results

        Args:
            test_struct: Generated test structure

        Returns:
            TestSuiteResult with execution details
        """
        test_code = test_struct.to_code()

        # Write test to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(test_code)
            temp_file = f.name

        try:
            # Execute test file
            result = subprocess.run(
                ["python", "-m", "pytest", temp_file, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Parse output
            return self._parse_pytest_output(
                result.stdout + result.stderr,
                test_struct.repair_id,
                len(test_struct.test_cases)
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Test execution timed out for {test_struct.repair_id}")
            return TestSuiteResult(
                repair_id=test_struct.repair_id,
                test_results=[],
                total_tests=len(test_struct.test_cases),
                passed_tests=0,
                failed_tests=len(test_struct.test_cases),
                error_tests=0
            )
        except Exception as e:
            logger.error(f"Failed to execute tests: {e}")
            return TestSuiteResult(
                repair_id=test_struct.repair_id,
                test_results=[],
                total_tests=len(test_struct.test_cases),
                passed_tests=0,
                failed_tests=0,
                error_tests=len(test_struct.test_cases)
            )
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _parse_pytest_output(
        self, output: str, repair_id: str, num_tests: int
    ) -> TestSuiteResult:
        """Parse pytest output to extract test results"""
        test_results = []
        passed = 0
        failed = 0
        errors = 0

        # Look for PASSED/FAILED patterns
        for line in output.split('\n'):
            if 'PASSED' in line:
                passed += 1
                test_results.append(TestResult(
                    test_name=line.split('::')[1] if '::' in line else "unknown",
                    test_status=TestStatus.PASSED
                ))
            elif 'FAILED' in line:
                failed += 1
                test_results.append(TestResult(
                    test_name=line.split('::')[1] if '::' in line else "unknown",
                    test_status=TestStatus.FAILED,
                    message=line
                ))
            elif 'ERROR' in line:
                errors += 1
                test_results.append(TestResult(
                    test_name=line.split('::')[1] if '::' in line else "unknown",
                    test_status=TestStatus.ERROR,
                    message=line
                ))

        # Fallback: count from summary line
        if not test_results:
            summary_match = re.search(r'(\d+) passed', output)
            if summary_match:
                passed = int(summary_match.group(1))
            summary_match = re.search(r'(\d+) failed', output)
            if summary_match:
                failed = int(summary_match.group(1))

        return TestSuiteResult(
            repair_id=repair_id,
            test_results=test_results,
            total_tests=num_tests,
            passed_tests=passed,
            failed_tests=failed,
            error_tests=errors
        )


class PatchValidator:
    """Validates repair patches through testing"""

    def __init__(self, test_executor: Optional[TestExecutor] = None):
        self.executor = test_executor or TestExecutor()

    def validate_patch(
        self, repair: RepairCandidate, test_struct: TestStructure
    ) -> VerifiedPatch:
        """
        Validate a repair patch through test execution

        Args:
            repair: Repair candidate
            test_struct: Generated test structure

        Returns:
            VerifiedPatch with validation results
        """
        # Execute tests
        test_results = self.executor.execute_test_structure(test_struct)

        # Calculate verification score: balance pass rate with repair confidence
        pass_rate = test_results.pass_rate
        repair_confidence = repair.confidence

        # Weights: test success (0.7) + repair quality (0.3)
        verification_score = 0.7 * pass_rate + 0.3 * repair_confidence

        # Determine confidence: lower if tests failed
        if test_results.patch_status == PatchStatus.VERIFIED:
            confidence = min(1.0, verification_score * 1.1)
        elif test_results.patch_status == PatchStatus.PARTIAL:
            confidence = verification_score * 0.8
        else:
            confidence = verification_score * 0.3

        # Collect failure details
        failures = [
            r.message for r in test_results.test_results 
            if r.test_status != TestStatus.PASSED
        ]

        return VerifiedPatch(
            repair=repair,
            test_results=test_results,
            verification_score=max(0.0, min(1.0, verification_score)),
            confidence=max(0.0, min(1.0, confidence)),
            failures=failures,
            success_details=f"Passed {test_results.passed_tests}/{test_results.total_tests} tests"
        )


class PatchFilter:
    """Filters and ranks patch candidates"""

    def __init__(self, min_pass_rate: float = 0.5):
        """
        Initialize filter

        Args:
            min_pass_rate: Minimum test pass rate to include patch
        """
        self.min_pass_rate = min_pass_rate
        self.validator = PatchValidator()

    def filter_repairs(
        self,
        verified_patches: List[VerifiedPatch],
        max_patches: int = 5
    ) -> List[VerifiedPatch]:
        """
        Filter and rank repair patches

        Args:
            verified_patches: List of patches with test results
            max_patches: Maximum patches to return

        Returns:
            Filtered and ranked list of patches
        """
        # Filter by pass rate
        filtered = [
            p for p in verified_patches
            if p.test_results.pass_rate >= self.min_pass_rate
        ]

        logger.info(
            f"Filtered {len(filtered)}/{len(verified_patches)} patches "
            f"(min_pass_rate={self.min_pass_rate})"
        )

        # Sort by verification score (highest first)
        filtered.sort(key=lambda p: p.verification_score, reverse=True)

        # Return top N patches
        return filtered[:max_patches]

    def rank_patches(
        self, verified_patches: List[VerifiedPatch]
    ) -> Dict[int, VerifiedPatch]:
        """
        Rank patches by verification score

        Args:
            verified_patches: List of verified patches

        Returns:
            Dict mapping rank to patch
        """
        # Sort by verification score
        sorted_patches = sorted(
            verified_patches,
            key=lambda p: p.verification_score,
            reverse=True
        )

        return {
            rank + 1: patch
            for rank, patch in enumerate(sorted_patches)
        }


@dataclass
class TSPFFunctionalResult:
    """Stage-1 functional filtering result from paper section 3.3."""

    patch_id: str
    passed: bool
    regression_passed: bool
    reproduction_passed: bool
    non_empty_edit: bool
    is_no_op_patch: bool
    regression_pass_rate: Optional[float] = None
    reproduction_pass_rate: Optional[float] = None
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "passed": self.passed,
            "regression_passed": self.regression_passed,
            "reproduction_passed": self.reproduction_passed,
            "non_empty_edit": self.non_empty_edit,
            "is_no_op_patch": self.is_no_op_patch,
            "regression_pass_rate": self.regression_pass_rate,
            "reproduction_pass_rate": self.reproduction_pass_rate,
            "reasons": list(self.reasons),
        }


@dataclass
class TSPFRankedPatch:
    """Stage-2 causality/similarity ranked patch."""

    patch_id: str
    candidate_id: str
    candidate_location: str
    patch_content: str
    causality_score: float
    similarity_score: float
    final_score: float
    functional_result: TSPFFunctionalResult
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "patch_id": self.patch_id,
            "candidate_id": self.candidate_id,
            "candidate_location": self.candidate_location,
            "patch_content": self.patch_content,
            "causality_score": self.causality_score,
            "similarity_score": self.similarity_score,
            "final_score": self.final_score,
            "functional_result": self.functional_result.to_dict(),
            "metadata": dict(self.metadata),
        }


class TwoStagePatchFilter:
    """Paper-aligned TSPF implementation.

    Stage 1 keeps only functionally acceptable patches: regression tests must
    not fail, reproduction tests must indicate the target issue is resolved,
    and the Search/Replace patch must make a real edit.

    Stage 2 ranks the valid patch set with Eq. (12):
        S_final(p_i) = mu * Caus(p_i) + (1 - mu) * S_sim(p_i)
    where S_sim follows Eq. (11), the average cosine similarity between one
    patch vector and every other valid patch vector.
    """

    def __init__(
        self,
        mu: float = 0.6,
        min_regression_pass_rate: float = 1.0,
        min_reproduction_pass_rate: float = 1.0,
        require_test_evidence: bool = True,
    ):
        if not 0.0 <= mu <= 1.0:
            raise ValueError("mu must be in [0, 1]")
        self.mu = mu
        self.min_regression_pass_rate = min_regression_pass_rate
        self.min_reproduction_pass_rate = min_reproduction_pass_rate
        self.require_test_evidence = require_test_evidence

    def filter_distillation_payload(
        self,
        distillation_payload: Dict[str, Any],
        test_evidence: Optional[Dict[str, Dict[str, Any]]] = None,
        max_patches: Optional[int] = None,
    ) -> Dict[str, Any]:
        patches = self._flatten_distilled_patches(distillation_payload)
        ranked, functional_results = self.filter_and_rank(
            patches=patches,
            test_evidence=test_evidence,
            max_patches=max_patches,
        )
        selected_patch = ranked[0].to_dict() if ranked else None
        selected_patch_id = selected_patch["patch_id"] if selected_patch else None
        selection_status = "selected" if selected_patch else "no_valid_patch"
        selection_reason = (
            "selected top-ranked patch by Eq. (12) S_final after Stage-1 functional filtering"
            if selected_patch
            else "no patch passed Stage-1 functional filtering; final patch is not selected"
        )
        return {
            "mu": self.mu,
            "min_regression_pass_rate": self.min_regression_pass_rate,
            "min_reproduction_pass_rate": self.min_reproduction_pass_rate,
            "require_test_evidence": self.require_test_evidence,
            "input_patch_count": len(patches),
            "valid_patch_count": len(ranked),
            "selection_status": selection_status,
            "selected_patch_id": selected_patch_id,
            "selected_patch": selected_patch,
            "selection_reason": selection_reason,
            "functional_results": [item.to_dict() for item in functional_results],
            "ranked_patches": [item.to_dict() for item in ranked],
        }

    def filter_and_rank(
        self,
        patches: List[Any],
        test_evidence: Optional[Dict[str, Dict[str, Any]]] = None,
        max_patches: Optional[int] = None,
    ) -> Tuple[List[TSPFRankedPatch], List[TSPFFunctionalResult]]:
        functional_results: List[TSPFFunctionalResult] = []
        valid_patches: List[Any] = []

        for patch in patches:
            result = self._functional_filter(patch, test_evidence or {})
            functional_results.append(result)
            if result.passed:
                valid_patches.append(patch)

        similarity_scores = self._compute_group_similarity(valid_patches)
        ranked: List[TSPFRankedPatch] = []
        result_by_patch_id = {item.patch_id: item for item in functional_results}

        for patch in valid_patches:
            patch_id = self._get_patch_id(patch)
            causality_score = self._extract_causality_score(patch)
            similarity_score = similarity_scores.get(patch_id, 0.0)
            final_score = self.mu * causality_score + (1.0 - self.mu) * similarity_score
            ranked.append(
                TSPFRankedPatch(
                    patch_id=patch_id,
                    candidate_id=self._get_candidate_id(patch),
                    candidate_location=self._get_candidate_location(patch),
                    patch_content=self._get_patch_content(patch),
                    causality_score=causality_score,
                    similarity_score=similarity_score,
                    final_score=final_score,
                    functional_result=result_by_patch_id[patch_id],
                    metadata=self._get_patch_metadata(patch),
                )
            )

        ranked.sort(key=lambda item: item.final_score, reverse=True)
        for rank, item in enumerate(ranked, start=1):
            item.rank = rank

        if max_patches is not None:
            ranked = ranked[:max_patches]
        return ranked, functional_results

    def _functional_filter(
        self,
        patch: Any,
        test_evidence: Dict[str, Dict[str, Any]],
    ) -> TSPFFunctionalResult:
        patch_id = self._get_patch_id(patch)
        patch_content = self._get_patch_content(patch)
        non_empty_edit = bool(patch_content.strip())
        is_no_op_patch = self._is_no_op_search_replace(patch_content)
        evidence = test_evidence.get(patch_id, {})

        regression_result = evidence.get("regression")
        reproduction_result = evidence.get("reproduction")
        regression_rate = self._pass_rate(regression_result)
        reproduction_rate = self._pass_rate(reproduction_result)

        regression_available = regression_rate is not None
        reproduction_available = reproduction_rate is not None
        regression_passed = (
            regression_rate >= self.min_regression_pass_rate
            if regression_available else not self.require_test_evidence
        )
        reproduction_passed = (
            reproduction_rate >= self.min_reproduction_pass_rate
            if reproduction_available else not self.require_test_evidence
        )

        reasons: List[str] = []
        if not non_empty_edit:
            reasons.append("empty_patch")
        if is_no_op_patch:
            reasons.append("no_op_search_replace")
        if self.require_test_evidence and not regression_available:
            reasons.append("missing_regression_test_evidence")
        if self.require_test_evidence and not reproduction_available:
            reasons.append("missing_reproduction_test_evidence")
        if regression_available and not regression_passed:
            reasons.append("regression_test_failed")
        if reproduction_available and not reproduction_passed:
            reasons.append("reproduction_test_failed")

        passed = non_empty_edit and not is_no_op_patch and regression_passed and reproduction_passed
        if passed:
            reasons.append("passed_stage1_functional_filter")

        return TSPFFunctionalResult(
            patch_id=patch_id,
            passed=passed,
            regression_passed=regression_passed,
            reproduction_passed=reproduction_passed,
            non_empty_edit=non_empty_edit,
            is_no_op_patch=is_no_op_patch,
            regression_pass_rate=regression_rate,
            reproduction_pass_rate=reproduction_rate,
            reasons=reasons,
        )

    def _compute_group_similarity(self, patches: List[Any]) -> Dict[str, float]:
        if not patches:
            return {}
        if len(patches) == 1:
            return {self._get_patch_id(patches[0]): 1.0}

        vectors = {
            self._get_patch_id(patch): self._text_vector(self._get_patch_vector_text(patch))
            for patch in patches
        }
        scores: Dict[str, float] = {}
        for patch in patches:
            patch_id = self._get_patch_id(patch)
            others = [
                other_id for other_id in vectors.keys()
                if other_id != patch_id
            ]
            if not others:
                scores[patch_id] = 1.0
                continue
            scores[patch_id] = sum(
                self._cosine_counter(vectors[patch_id], vectors[other_id])
                for other_id in others
            ) / len(others)
        return scores

    def _flatten_distilled_patches(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        patches: List[Dict[str, Any]] = []
        for candidate_id, candidate_result in payload.get("candidate_results", {}).items():
            candidate_location = str(candidate_result.get("candidate_location", candidate_id))
            for patch in candidate_result.get("ranked_patches", []):
                item = dict(patch)
                item.setdefault("candidate_id", candidate_id)
                item.setdefault("candidate_location", candidate_location)
                patches.append(item)
        return patches

    def _pass_rate(self, result: Any) -> Optional[float]:
        if result is None:
            return None
        if isinstance(result, TestSuiteResult):
            return result.pass_rate
        if isinstance(result, dict):
            if "pass_rate" in result:
                return float(result["pass_rate"])
            total = int(result.get("total_tests", 0) or 0)
            passed = int(result.get("passed_tests", 0) or 0)
            if total > 0:
                return passed / total
        return None

    def _get_patch_id(self, patch: Any) -> str:
        if isinstance(patch, dict):
            return str(patch.get("patch_id") or patch.get("id") or "")
        return str(getattr(patch, "patch_id", getattr(patch, "id", "")))

    def _get_candidate_id(self, patch: Any) -> str:
        if isinstance(patch, dict):
            return str(patch.get("candidate_id") or patch.get("candidate_location") or "")
        return str(getattr(patch, "candidate_id", getattr(patch, "location", "")))

    def _get_candidate_location(self, patch: Any) -> str:
        if isinstance(patch, dict):
            return str(patch.get("candidate_location") or patch.get("location") or "")
        return str(getattr(patch, "candidate_location", getattr(patch, "location", "")))

    def _get_patch_content(self, patch: Any) -> str:
        if isinstance(patch, dict):
            return str(patch.get("patch_content") or patch.get("content") or patch.get("repaired_code") or "")
        return str(
            getattr(
                patch,
                "patch_content",
                getattr(patch, "repaired_code", ""),
            )
        )

    def _get_patch_vector_text(self, patch: Any) -> str:
        if isinstance(patch, dict):
            return str(patch.get("embedding_text") or patch.get("patch_content") or "")
        return self._get_patch_content(patch)

    def _extract_causality_score(self, patch: Any) -> float:
        if isinstance(patch, dict):
            for key in ("causality_score", "causal_score", "causal_matching_score"):
                if key in patch:
                    return self._clamp(float(patch.get(key) or 0.0))
            causal_alignment = patch.get("causal_alignment", {}) or {}
            if isinstance(causal_alignment, dict) and "score" in causal_alignment:
                return self._clamp(float(causal_alignment.get("score") or 0.0))
            metadata = patch.get("metadata", {}) or {}
            if isinstance(metadata, dict):
                return self._extract_causality_score(metadata)
            return 0.0
        return self._clamp(float(getattr(patch, "causality_score", getattr(patch, "confidence", 0.0)) or 0.0))

    def _get_patch_metadata(self, patch: Any) -> Dict[str, Any]:
        if isinstance(patch, dict):
            return {
                key: patch.get(key)
                for key in (
                    "generated_round",
                    "reflection_score",
                    "consistency_score",
                    "embedding_similarity",
                    "distillation_score",
                    "is_no_op_patch",
                )
                if key in patch
            }
        return {}

    def _is_no_op_search_replace(self, patch_content: str) -> bool:
        match = re.search(r'<<<\s*SEARCH\n(.*?)\n===\n(.*?)\n>>>\s*REPLACE', patch_content, re.DOTALL)
        if not match:
            return False
        return match.group(1).strip() == match.group(2).strip()

    def _text_vector(self, text: str) -> Counter:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|&&|\|\|", text.lower())
        return Counter(tokens)

    def _cosine_counter(self, left: Counter, right: Counter) -> float:
        if not left or not right:
            return 0.0
        common = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in common)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return max(0.0, min(1.0, numerator / (left_norm * right_norm)))

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))


class PatchEvaluator:
    """Complete evaluation pipeline for repair patches"""

    def __init__(self, min_pass_rate: float = 0.5):
        self.filter = PatchFilter(min_pass_rate=min_pass_rate)
        self.validator = PatchValidator()

    def evaluate_repairs(
        self,
        repairs: List[RepairCandidate],
        test_structures: Dict[str, TestStructure]
    ) -> List[VerifiedPatch]:
        """
        Evaluate repair patches through testing

        Args:
            repairs: List of repair candidates
            test_structures: Dict of test structures by repair_id

        Returns:
            List of verified patches sorted by quality
        """
        verified_patches = []

        for repair in repairs:
            if repair.id not in test_structures:
                logger.warning(f"No test structure for repair {repair.id}")
                continue

            try:
                test_struct = test_structures[repair.id]
                verified = self.validator.validate_patch(repair, test_struct)
                verified_patches.append(verified)
            except Exception as e:
                logger.error(f"Failed to evaluate repair {repair.id}: {e}")

        # Filter to top patches
        top_patches = self.filter.filter_repairs(
            verified_patches, max_patches=5
        )

        logger.info(
            f"Evaluation complete: {len(top_patches)} patches passed validation"
        )

        return top_patches
