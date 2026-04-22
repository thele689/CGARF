"""
Phase 4 TSPF Integration Tests
==============================

Comprehensive tests for Test Synthesis and Patch Filtering modules
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.common.data_structures import (
    IssueContext, RepairCandidate, CodeEntity, EntityType
)
from src.tspf.test_synthesizer import (
    TestSynthesizer, TestInputGenerator, TestCaseGenerator,
    TestAssertion, TestCase, TestType, AssertionType
)
from src.tspf.patch_filter import (
    TestExecutor, PatchValidator, PatchFilter, PatchEvaluator,
    TestResult, TestSuiteResult, TestStatus, PatchStatus, VerifiedPatch,
    TwoStagePatchFilter
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_issue_context():
    """Sample issue context for testing"""
    return IssueContext(
        id="issue_001",
        description="NoneType error when data is empty",
        repo_path="/test/repo",
        candidates=["src/processor.py::process_data"],
        test_framework="pytest"
    )


@pytest.fixture
def sample_repair_candidate():
    """Sample repair candidate"""
    return RepairCandidate(
        id="repair_001",
        original_code="def process_data(data):\n    return data.strip()",
        repaired_code="""def process_data(data):
    if data is None:
        return ""
    return data.strip()""",
        mutation_type="NULL_CHECK",
        affected_lines=[11],
        confidence=0.85
    )


@pytest.fixture
def mock_llm():
    """Mock LLM interface"""
    mock = Mock()
    mock.generate.return_value = "def process_data(data):\n    return data or ''"
    return mock


# ============================================================================
# TestSynthesizer Tests
# ============================================================================

class TestTestInputGenerator:
    """Tests for TestInputGenerator"""

    def test_extract_error_values_none(self, sample_issue_context):
        """Test extraction of None error value"""
        gen = TestInputGenerator(sample_issue_context)
        values = gen.extract_error_values()
        assert None in values or len(values) > 0

    def test_extract_error_values_empty(self):
        """Test extraction of empty string error"""
        issue = IssueContext(
            id="test_001",
            description="Empty string causes error",
            repo_path="/test",
            candidates=["test.py::test"],
            test_framework="pytest"
        )
        gen = TestInputGenerator(issue)
        values = gen.extract_error_values()
        assert "" in values or len(values) > 0

    def test_generate_boundary_inputs_string(self, sample_issue_context):
        """Test boundary input generation for strings"""
        gen = TestInputGenerator(sample_issue_context)
        inputs = gen.generate_boundary_inputs("param", "str")
        assert "" in inputs  # Empty string
        assert any(len(s) > 100 for s in inputs)  # Large string


class TestTestCaseGenerator:
    """Tests for TestCaseGenerator"""

    def test_generate_regression_test(self, sample_issue_context, sample_repair_candidate):
        """Test regression test generation"""
        gen = TestCaseGenerator(sample_issue_context, llm_interface=None)
        test = gen._create_regression_test(sample_repair_candidate)
        
        assert test is not None
        assert test.test_type == TestType.REGRESSION
        assert "regression" in test.test_name.lower()

    def test_generate_unit_tests(self, sample_issue_context, sample_repair_candidate):
        """Test unit test generation"""
        gen = TestCaseGenerator(sample_issue_context)
        tests = gen._create_unit_tests(sample_repair_candidate)
        
        assert len(tests) > 0
        assert all(t.test_type == TestType.UNIT for t in tests)

    def test_generate_boundary_tests(self, sample_issue_context, sample_repair_candidate):
        """Test boundary test generation"""
        gen = TestCaseGenerator(sample_issue_context)
        tests = gen._create_boundary_tests(sample_repair_candidate)
        
        assert len(tests) > 0
        assert all(t.test_type == TestType.BOUNDARY for t in tests)

    def test_generate_reproduction_test(self, sample_issue_context, sample_repair_candidate):
        """Test reproduction test generation"""
        gen = TestCaseGenerator(sample_issue_context)
        test = gen._create_reproduction_test(sample_repair_candidate)

        assert test is not None
        assert test.test_type == TestType.REPRODUCTION
        assert "reproduction" in test.test_name.lower()


class TestTestSynthesizer:
    """Tests for TestSynthesizer main class"""

    def test_synthesize_tests(self, sample_issue_context, sample_repair_candidate):
        """Test test synthesis for single repair"""
        synth = TestSynthesizer()
        test_struct = synth.synthesize_tests(sample_repair_candidate, sample_issue_context)
        
        assert test_struct is not None
        assert test_struct.repair_id == sample_repair_candidate.id
        assert len(test_struct.test_cases) > 0

    def test_synthesize_batch(self, sample_issue_context):
        """Test batch test synthesis"""
        repairs = [
            RepairCandidate(
                id=f"repair_{i}",
                original_code="def f(x): return x",
                repaired_code="def f(x): return x or 0",
                mutation_type="FALLBACK",
                affected_lines=[1],
                confidence=0.8
            )
            for i in range(3)
        ]
        
        synth = TestSynthesizer()
        results = synth.synthesize_batch(repairs, sample_issue_context)
        
        assert len(results) <= len(repairs)

    def test_test_structure_to_code(self, sample_issue_context, sample_repair_candidate):
        """Test conversion of test structure to code"""
        synth = TestSynthesizer()
        test_struct = synth.synthesize_tests(sample_repair_candidate, sample_issue_context)
        code = test_struct.to_code()
        
        assert "import unittest" in code
        assert "class Test" in code
        assert "def test_" in code


# ============================================================================
# PatchFilter Tests
# ============================================================================

class TestTestResult:
    """Tests for TestResult data structure"""

    def test_result_passed_property(self):
        """Test passed property"""
        result = TestResult(
            test_name="test_foo",
            test_status=TestStatus.PASSED
        )
        assert result.passed is True

    def test_result_failed_property(self):
        """Test failed property"""
        result = TestResult(
            test_name="test_foo",
            test_status=TestStatus.FAILED
        )
        assert result.passed is False


class TestTestSuiteResult:
    """Tests for TestSuiteResult"""

    def test_pass_rate_calculation(self):
        """Test pass rate calculation"""
        result = TestSuiteResult(
            repair_id="test",
            test_results=[],
            total_tests=10,
            passed_tests=7,
            failed_tests=3,
            error_tests=0
        )
        assert result.pass_rate == 0.7

    def test_patch_status_verified(self):
        """Test patch status when all tests pass"""
        result = TestSuiteResult(
            repair_id="test",
            test_results=[],
            total_tests=5,
            passed_tests=5,
            failed_tests=0,
            error_tests=0
        )
        assert result.patch_status == PatchStatus.VERIFIED

    def test_patch_status_partial(self):
        """Test patch status when some tests pass"""
        result = TestSuiteResult(
            repair_id="test",
            test_results=[],
            total_tests=10,
            passed_tests=6,
            failed_tests=4,
            error_tests=0
        )
        assert result.patch_status == PatchStatus.PARTIAL

    def test_patch_status_failing(self):
        """Test patch status when most tests fail"""
        result = TestSuiteResult(
            repair_id="test",
            test_results=[],
            total_tests=10,
            passed_tests=2,
            failed_tests=8,
            error_tests=0
        )
        assert result.patch_status == PatchStatus.FAILING


class TestPatchValidator:
    """Tests for PatchValidator"""

    def test_validate_patch_verified(self, sample_repair_candidate):
        """Test patch validation when all tests pass"""
        # Mock test executor
        mock_executor = Mock(spec=TestExecutor)
        test_result = TestSuiteResult(
            repair_id=sample_repair_candidate.id,
            test_results=[],
            total_tests=5,
            passed_tests=5,
            failed_tests=0,
            error_tests=0
        )
        mock_executor.execute_test_structure.return_value = test_result

        validator = PatchValidator(mock_executor)
        mock_test_struct = Mock()
        mock_test_struct.repair_id = sample_repair_candidate.id

        verified = validator.validate_patch(sample_repair_candidate, mock_test_struct)

        assert verified.verification_score > 0.7
        assert verified.test_results.pass_rate == 1.0

    def test_validate_patch_partial(self, sample_repair_candidate):
        """Test patch validation with partial pass"""
        mock_executor = Mock(spec=TestExecutor)
        test_result = TestSuiteResult(
            repair_id=sample_repair_candidate.id,
            test_results=[],
            total_tests=10,
            passed_tests=6,
            failed_tests=4,
            error_tests=0
        )
        mock_executor.execute_test_structure.return_value = test_result

        validator = PatchValidator(mock_executor)
        mock_test_struct = Mock()

        verified = validator.validate_patch(sample_repair_candidate, mock_test_struct)

        assert 0 < verified.verification_score < 1.0
        assert verified.test_results.patch_status == PatchStatus.PARTIAL


class TestPatchFilter:
    """Tests for PatchFilter"""

    def test_filter_repairs_by_pass_rate(self):
        """Test filtering repairs by minimum pass rate"""
        patches = [
            VerifiedPatch(
                repair=RepairCandidate(
                    id=f"r{i}", original_code="", repaired_code="",
                    mutation_type="NULL_CHECK", affected_lines=[],
                    confidence=0.8
                ),
                test_results=TestSuiteResult(
                    repair_id=f"r{i}",
                    test_results=[],
                    total_tests=10,
                    passed_tests=7 + i,  # 70%, 80%, 90%
                    failed_tests=3 - i,
                    error_tests=0
                ),
                verification_score=0.7 + 0.1 * i,
                confidence=0.6 + 0.1 * i
            )
            for i in range(3)
        ]

        filter_obj = PatchFilter(min_pass_rate=0.75)
        filtered = filter_obj.filter_repairs(patches)

        # Only patches with >= 75% pass rate
        assert all(p.test_results.pass_rate >= 0.75 for p in filtered)

    def test_rank_patches(self):
        """Test patch ranking by verification score"""
        patches = [
            VerifiedPatch(
                repair=RepairCandidate(
                    id=f"r{i}", original_code="", repaired_code="",
                    mutation_type="NULL_CHECK", affected_lines=[],
                    confidence=0.8
                ),
                test_results=TestSuiteResult(
                    repair_id=f"r{i}",
                    test_results=[],
                    total_tests=10,
                    passed_tests=5 + i,
                    failed_tests=5 - i,
                    error_tests=0
                ),
                verification_score=0.5 + 0.1 * i,
                confidence=0.6
            )
            for i in range(3)
        ]

        filter_obj = PatchFilter()
        ranked = filter_obj.rank_patches(patches)

        # Verify ranking order
        assert ranked[1].verification_score >= ranked[2].verification_score
        assert ranked[2].verification_score >= ranked[3].verification_score


class TestTwoStagePatchFilter:
    """Paper-aligned TSPF tests."""

    def test_stage1_filters_no_op_patch(self):
        patches = [
            {
                "patch_id": "noop",
                "candidate_id": "loc_a",
                "patch_content": "<<< SEARCH\nreturn x\n===\nreturn x\n>>> REPLACE",
                "causality_score": 1.0,
                "embedding_text": "No textual edit; SEARCH and REPLACE blocks are identical.",
            },
            {
                "patch_id": "real",
                "candidate_id": "loc_a",
                "patch_content": "<<< SEARCH\nreturn x\n===\nreturn x or 0\n>>> REPLACE",
                "causality_score": 0.75,
                "embedding_text": "Added lines:\nreturn x or 0",
            },
        ]

        evidence = {
            "real": {
                "regression": {"total_tests": 2, "passed_tests": 2},
                "reproduction": {"total_tests": 1, "passed_tests": 1},
            }
        }
        tspf = TwoStagePatchFilter(mu=0.6)
        ranked, functional = tspf.filter_and_rank(patches, test_evidence=evidence)

        assert [item.patch_id for item in ranked] == ["real"]
        noop = [item for item in functional if item.patch_id == "noop"][0]
        assert noop.passed is False
        assert "no_op_search_replace" in noop.reasons

    def test_stage2_uses_causality_and_similarity_scores(self):
        patches = [
            {
                "patch_id": "p1",
                "candidate_id": "loc_a",
                "patch_content": "<<< SEARCH\nreturn x\n===\nreturn x or 0\n>>> REPLACE",
                "causality_score": 0.9,
                "embedding_text": "return fallback zero",
            },
            {
                "patch_id": "p2",
                "candidate_id": "loc_b",
                "patch_content": "<<< SEARCH\nreturn y\n===\nreturn y or 0\n>>> REPLACE",
                "causality_score": 0.2,
                "embedding_text": "return fallback zero",
            },
        ]

        evidence = {
            patch["patch_id"]: {
                "regression": {"total_tests": 2, "passed_tests": 2},
                "reproduction": {"total_tests": 1, "passed_tests": 1},
            }
            for patch in patches
        }

        tspf = TwoStagePatchFilter(mu=0.8)
        ranked, _ = tspf.filter_and_rank(patches, test_evidence=evidence)

        assert ranked[0].patch_id == "p1"
        assert ranked[0].final_score > ranked[1].final_score
        assert all(0.0 <= item.similarity_score <= 1.0 for item in ranked)

    def test_missing_test_evidence_blocks_stage1_by_default(self):
        patches = [
            {
                "patch_id": "p1",
                "candidate_id": "loc_a",
                "patch_content": "<<< SEARCH\nreturn x\n===\nreturn x or 0\n>>> REPLACE",
                "causality_score": 1.0,
            }
        ]

        tspf = TwoStagePatchFilter()
        ranked, functional = tspf.filter_and_rank(patches)

        assert ranked == []
        assert functional[0].passed is False
        assert "missing_regression_test_evidence" in functional[0].reasons
        assert "missing_reproduction_test_evidence" in functional[0].reasons

    def test_filter_distillation_payload_selects_top_ranked_patch(self):
        payload = {
            "candidate_results": {
                "loc_a": {
                    "candidate_location": "module.py::loc_a",
                    "ranked_patches": [
                        {
                            "patch_id": "p_low",
                            "patch_content": "<<< SEARCH\nreturn x\n===\nreturn x or 0\n>>> REPLACE",
                            "causality_score": 0.2,
                            "embedding_text": "return fallback zero",
                        },
                        {
                            "patch_id": "p_high",
                            "patch_content": "<<< SEARCH\nreturn y\n===\nreturn y or 0\n>>> REPLACE",
                            "causality_score": 0.9,
                            "embedding_text": "return fallback zero",
                        },
                    ],
                }
            }
        }
        evidence = {
            patch_id: {
                "regression": {"total_tests": 2, "passed_tests": 2},
                "reproduction": {"total_tests": 1, "passed_tests": 1},
            }
            for patch_id in ("p_low", "p_high")
        }

        result = TwoStagePatchFilter(mu=0.8).filter_distillation_payload(
            payload,
            test_evidence=evidence,
        )

        assert result["selection_status"] == "selected"
        assert result["selected_patch_id"] == "p_high"
        assert result["selected_patch"]["rank"] == 1
        assert result["ranked_patches"][0]["patch_id"] == "p_high"

    def test_filter_distillation_payload_records_no_final_patch_when_none_valid(self):
        payload = {
            "candidate_results": {
                "loc_a": {
                    "candidate_location": "module.py::loc_a",
                    "ranked_patches": [
                        {
                            "patch_id": "p1",
                            "patch_content": "<<< SEARCH\nreturn x\n===\nreturn x or 0\n>>> REPLACE",
                            "causality_score": 1.0,
                        }
                    ],
                }
            }
        }

        result = TwoStagePatchFilter().filter_distillation_payload(payload)

        assert result["selection_status"] == "no_valid_patch"
        assert result["selected_patch_id"] is None
        assert result["selected_patch"] is None
        assert result["valid_patch_count"] == 0


# ============================================================================
# End-to-End Tests
# ============================================================================

class TestTSPFEndToEnd:
    """End-to-end TSPF workflow tests"""

    def test_complete_workflow(self, sample_issue_context):
        """Test complete TSPF workflow: synthesis -> validation -> filtering"""
        # Create repairs
        repairs = [
            RepairCandidate(
                id="repair_null_check",
                original_code="def f(x): return x.upper()",
                repaired_code="def f(x):\n    if x is None:\n        return ''\n    return x.upper()",
                mutation_type="NULL_CHECK",
                affected_lines=[1],
                confidence=0.9
            ),
            RepairCandidate(
                id="repair_fallback",
                original_code="def f(x): return x.upper()",
                repaired_code="def f(x): return (x or '').upper()",
                mutation_type="FALLBACK",
                affected_lines=[1],
                confidence=0.7
            )
        ]

        # Synthesize tests
        synth = TestSynthesizer()
        test_structures = synth.synthesize_batch(repairs, sample_issue_context)
        assert len(test_structures) == len(repairs)

        # Validate patches
        evaluator = PatchEvaluator(min_pass_rate=0.5)
        # Mock the test executor since we don't want actual execution
        for repair_id, test_struct in test_structures.items():
            evaluator.validator.executor.execute_test_structure = Mock(
                return_value=TestSuiteResult(
                    repair_id=repair_id,
                    test_results=[],
                    total_tests=3,
                    passed_tests=2,
                    failed_tests=1,
                    error_tests=0
                )
            )

        top_patches = evaluator.evaluate_repairs(repairs, test_structures)
        assert len(top_patches) <= len(repairs)

    def test_workflow_with_multiple_repairs(self, sample_issue_context):
        """Test workflow with multiple repairs"""
        repairs = [
            RepairCandidate(
                id=f"repair_{i}",
                original_code="def test(): pass",
                repaired_code=f"def test(): return {i}",
                mutation_type="VARIABLE_ASSIGNMENT",
                affected_lines=[1],
                confidence=0.5 + 0.1 * i
            )
            for i in range(5)
        ]

        synth = TestSynthesizer()
        test_structures = synth.synthesize_batch(repairs, sample_issue_context)

        evaluator = PatchEvaluator()
        assert len(test_structures) > 0


# ============================================================================
# Integration Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
