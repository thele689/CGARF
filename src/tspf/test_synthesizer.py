"""
ModuleL: Test Synthesis for Repair Code Validation
===================================================

Synthesizes unit tests that verify repair code patches solve the reported issues.
Generates test cases with:
  - Input data from issue context
  - Expected behavior from issue description
  - Verification of repair-specific fixes

Workflow:
  1. Extract test requirements from issue & code
  2. Generate test inputs (normal + edge cases)
  3. Create test assertions for repair behavior
  4. Package as executable unittest code
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import ast
import re
from loguru import logger

from src.common.data_structures import IssueContext, CodeEntity, RepairCandidate


class TestType(Enum):
    """Types of tests to generate"""
    UNIT = "unit"  # Direct function call tests
    BOUNDARY = "boundary"  # Edge case tests
    INTEGRATION = "integration"  # Multi-function tests
    REGRESSION = "regression"  # Bug-specific tests
    REPRODUCTION = "reproduction"  # Tests that reproduce and then check the target failure


class AssertionType(Enum):
    """Types of assertions"""
    EQUAL = "equal"
    NOT_NONE = "not_none"
    EXCEPTION = "exception"
    NO_EXCEPTION = "no_exception"
    CONTAINS = "contains"
    GREATER = "greater"
    LESS = "less"


@dataclass
class TestAssertion:
    """Single assertion in a test"""
    assertion_type: AssertionType
    expression: str  # Python expression to assert
    expected_value: Optional[Any] = None
    error_type: Optional[type] = None
    description: str = ""

    def to_code(self) -> str:
        """Convert assertion to Python code"""
        if self.assertion_type == AssertionType.EQUAL:
            return f"self.assertEqual({self.expression}, {repr(self.expected_value)})"
        elif self.assertion_type == AssertionType.NOT_NONE:
            return f"self.assertIsNotNone({self.expression})"
        elif self.assertion_type == AssertionType.EXCEPTION:
            return f"with self.assertRaises({self.error_type.__name__}):\n            {self.expression}"
        elif self.assertion_type == AssertionType.NO_EXCEPTION:
            return f"# Should not raise: {self.expression}\n        result = {self.expression}"
        elif self.assertion_type == AssertionType.CONTAINS:
            return f"self.assertIn({repr(self.expected_value)}, {self.expression})"
        elif self.assertion_type == AssertionType.GREATER:
            return f"self.assertGreater({self.expression}, {self.expected_value})"
        elif self.assertion_type == AssertionType.LESS:
            return f"self.assertLess({self.expression}, {self.expected_value})"
        return f"# {self.description}"


@dataclass
class TestCase:
    """Generated test case"""
    test_id: str
    test_name: str
    test_type: TestType
    function_name: str
    test_inputs: Dict[str, Any]
    assertions: List[TestAssertion]
    setup_code: str = ""
    teardown_code: str = ""
    description: str = ""

    def to_code(self) -> str:
        """Convert test case to Python unittest code"""
        inputs_str = "\n        ".join(
            f"{k} = {repr(v)}" for k, v in self.test_inputs.items()
        )
        assertions_code = "\n        ".join(
            a.to_code() for a in self.assertions
        )

        code = f"""
    def {self.test_name}(self):
        \"\"\"{self.description}\"\"\"
        {inputs_str}
        
        {assertions_code}
"""
        return code


@dataclass
class TestStructure:
    """Complete test structure for a repair"""
    repair_id: str
    test_class_name: str
    test_cases: List[TestCase]
    imports: List[str] = field(default_factory=list)
    class_setup: str = ""
    class_teardown: str = ""

    def to_code(self) -> str:
        """Generate complete test file code"""
        imports = "\n".join(self.imports)
        test_cases_code = "\n".join(tc.to_code() for tc in self.test_cases)

        code = f"""\"\"\"Auto-generated tests for repair {self.repair_id}\"\"\"
import unittest
{imports}


class {self.test_class_name}(unittest.TestCase):
    {self.class_setup if self.class_setup else "pass"}
    
{test_cases_code}

    {self.class_teardown if self.class_teardown else "pass"}


if __name__ == '__main__':
    unittest.main()
"""
        return code


class TestInputGenerator:
    """Generates test inputs based on issue context"""

    def __init__(self, issue_context: IssueContext):
        self.issue = issue_context

    def extract_error_values(self) -> List[Any]:
        """Extract likely error-causing values from issue description"""
        error_values = [None, "", 0, -1, [], {}]

        # Parse issue description for hints
        desc_lower = self.issue.description.lower()
        if "none" in desc_lower or "null" in desc_lower:
            error_values = [None]
        elif "empty" in desc_lower or "blank" in desc_lower:
            error_values = ["", []]
        elif "negative" in desc_lower or "zero" in desc_lower:
            error_values = [0, -1]

        return error_values

    def generate_boundary_inputs(self, param_name: str, param_type: str) -> List[Any]:
        """Generate boundary case inputs for a parameter"""
        inputs = []

        if param_type in ("str", "string"):
            inputs = ["", " ", "a", "x" * 1000]  # empty, whitespace, single, large
        elif param_type in ("int", "integer"):
            inputs = [0, 1, -1, 2**31 - 1]  # zero, positive, negative, max
        elif param_type in ("list", "array"):
            inputs = [[], [1], [1, 2, 3], list(range(100))]  # empty, small, normal, large
        elif param_type in ("dict", "object"):
            inputs = [{}, {"a": 1}, {"a": 1, "b": 2}]
        elif param_type in ("float", "double"):
            inputs = [0.0, 1.0, -1.0, float('inf')]

        return inputs


class TestCaseGenerator:
    """Generates test cases from repair code and issue context"""

    def __init__(self, issue_context: IssueContext, llm_interface=None):
        self.issue = issue_context
        self.llm = llm_interface
        self.input_gen = TestInputGenerator(issue_context)

    def generate_for_repair(self, repair: RepairCandidate) -> List[TestCase]:
        """Generate test cases for a repair candidate"""
        test_cases = []

        # 1. Regression test: verify bug is fixed
        regression_test = self._create_regression_test(repair)
        if regression_test:
            test_cases.append(regression_test)

        # 2. Reproduction test: check the issue-triggering input is resolved
        reproduction_test = self._create_reproduction_test(repair)
        if reproduction_test:
            test_cases.append(reproduction_test)

        # 3. Unit tests: verify repair code works correctly
        unit_tests = self._create_unit_tests(repair)
        test_cases.extend(unit_tests)

        # 4. Boundary tests: edge cases
        boundary_tests = self._create_boundary_tests(repair)
        test_cases.extend(boundary_tests)

        return test_cases

    def _create_regression_test(self, repair: RepairCandidate) -> Optional[TestCase]:
        """Create test verifying the specific bug is fixed"""
        try:
            # Extract affected function from repair
            func_match = re.search(r'def\s+(\w+)\s*\(', repair.repaired_code)
            if not func_match:
                return None

            func_name = func_match.group(1)

            # Create assertion: function should not raise exception
            assertions = [
                TestAssertion(
                    assertion_type=AssertionType.NO_EXCEPTION,
                    expression=f"{func_name}()",
                    description="Bug fix: should not raise exception"
                )
            ]

            return TestCase(
                test_id=f"{repair.id}_regression",
                test_name=f"test_regression_fix_{repair.id}",
                test_type=TestType.REGRESSION,
                function_name=func_name,
                test_inputs={},
                assertions=assertions,
                description=f"Regression test for repair {repair.id}: verifies bug is fixed"
            )
        except Exception as e:
            logger.warning(f"Failed to create regression test: {e}")
            return None

    def _create_reproduction_test(self, repair: RepairCandidate) -> Optional[TestCase]:
        """Create an issue-triggering reproduction test for the target bug."""
        try:
            func_match = re.search(r'def\s+(\w+)\s*\((.*?)\)', repair.repaired_code, re.DOTALL)
            if not func_match:
                return None

            func_name = func_match.group(1)
            params = [
                item.strip().split("=")[0].strip()
                for item in func_match.group(2).split(",")
                if item.strip() and item.strip() != "self"
            ]
            issue_values = self.input_gen.extract_error_values()
            test_inputs = {}
            call_args = []
            if params:
                value = issue_values[0] if issue_values else None
                test_inputs[params[0]] = value
                call_args.append(params[0])

            expression = f"{func_name}({', '.join(call_args)})"
            return TestCase(
                test_id=f"{repair.id}_reproduction",
                test_name=f"test_reproduction_issue_{repair.id}",
                test_type=TestType.REPRODUCTION,
                function_name=func_name,
                test_inputs=test_inputs,
                assertions=[
                    TestAssertion(
                        assertion_type=AssertionType.NO_EXCEPTION,
                        expression=expression,
                        description="Issue reproduction should be resolved by the patch"
                    )
                ],
                description=f"Reproduction test for repair {repair.id}: issue-triggering input should be resolved"
            )
        except Exception as e:
            logger.warning(f"Failed to create reproduction test: {e}")
            return None

    def _create_unit_tests(self, repair: RepairCandidate) -> List[TestCase]:
        """Create unit tests for repair code"""
        tests = []

        # Extract function from repair
        try:
            tree = ast.parse(repair.repaired_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name

                    # Test 1: Function returns without error
                    test_normal = TestCase(
                        test_id=f"{repair.id}_normal",
                        test_name=f"test_{func_name}_normal_call_{repair.id}",
                        test_type=TestType.UNIT,
                        function_name=func_name,
                        test_inputs={},
                        assertions=[
                            TestAssertion(
                                assertion_type=AssertionType.NOT_NONE,
                                expression=f"{func_name}()",
                                description="Function executes without error"
                            )
                        ],
                        description=f"Unit test: {func_name} executes successfully"
                    )
                    tests.append(test_normal)

                    # Test 2: With error values (if applicable)
                    error_values = self.input_gen.extract_error_values()
                    if error_values and len(node.args.args) > 0:
                        param = node.args.args[0]
                        for err_val in error_values[:2]:  # Limit to 2 error values
                            test_error = TestCase(
                                test_id=f"{repair.id}_error_{type(err_val).__name__}",
                                test_name=f"test_{func_name}_with_error_value_{type(err_val).__name__}",
                                test_type=TestType.UNIT,
                                function_name=func_name,
                                test_inputs={param.arg: err_val},
                                assertions=[
                                    TestAssertion(
                                        assertion_type=AssertionType.NOT_NONE,
                                        expression=f"{func_name}({param.arg})",
                                        description=f"Should handle {err_val}"
                                    )
                                ],
                                description=f"Unit test: {func_name} handles {type(err_val).__name__}"
                            )
                            tests.append(test_error)
        except Exception as e:
            logger.warning(f"Failed to create unit tests: {e}")

        return tests

    def _create_boundary_tests(self, repair: RepairCandidate) -> List[TestCase]:
        """Create boundary/edge case tests"""
        tests = []

        try:
            tree = ast.parse(repair.repaired_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and len(node.args.args) > 0:
                    func_name = node.name
                    param = node.args.args[0]

                    # Generate boundary inputs
                    boundary_inputs = self.input_gen.generate_boundary_inputs(
                        param.arg, "str"  # Default to string, could be extended
                    )

                    for i, inp in enumerate(boundary_inputs[:3]):  # Limit to 3
                        test = TestCase(
                            test_id=f"{repair.id}_boundary_{i}",
                            test_name=f"test_{func_name}_boundary_case_{i}",
                            test_type=TestType.BOUNDARY,
                            function_name=func_name,
                            test_inputs={param.arg: inp},
                            assertions=[
                                TestAssertion(
                                    assertion_type=AssertionType.NOT_NONE,
                                    expression=f"{func_name}({param.arg})",
                                    description=f"Boundary case: {repr(inp)}"
                                )
                            ],
                            description=f"Boundary test: {func_name} with input {repr(inp)}"
                        )
                        tests.append(test)
        except Exception as e:
            logger.warning(f"Failed to create boundary tests: {e}")

        return tests


class TestSynthesizer:
    """Main orchestrator for test synthesis"""

    def __init__(self, llm_interface=None):
        self.llm = llm_interface

    def synthesize_tests(
        self, repair: RepairCandidate, issue_context: IssueContext
    ) -> TestStructure:
        """
        Synthesize test structure for a repair candidate

        Args:
            repair: Repair candidate code
            issue_context: Original issue context

        Returns:
            TestStructure: Complete test code and metadata
        """
        # Generate test cases
        test_gen = TestCaseGenerator(issue_context, self.llm)
        test_cases = test_gen.generate_for_repair(repair)

        logger.info(f"Generated {len(test_cases)} test cases for repair {repair.id}")

        # Create test structure
        test_structure = TestStructure(
            repair_id=repair.id,
            test_class_name=f"Test{repair.id.replace('-', '_').title()}",
            test_cases=test_cases,
            imports=[
                "from unittest.mock import Mock, patch",
                "import os",
                "import sys",
                "sys.path.insert(0, os.getenv('CGARF_ROOT', os.getcwd()))",
            ],
            class_setup="def setUp(self):\n        pass",
            class_teardown="def tearDown(self):\n        pass"
        )

        return test_structure

    def synthesize_batch(
        self, repairs: List[RepairCandidate], issue_context: IssueContext
    ) -> Dict[str, TestStructure]:
        """
        Synthesize tests for multiple repair candidates

        Args:
            repairs: List of repair candidates
            issue_context: Shared issue context

        Returns:
            Dict mapping repair_id to TestStructure
        """
        test_structures = {}

        for repair in repairs:
            try:
                test_struct = self.synthesize_tests(repair, issue_context)
                test_structures[repair.id] = test_struct
            except Exception as e:
                logger.error(f"Failed to synthesize tests for repair {repair.id}: {e}")

        logger.info(f"Synthesized tests for {len(test_structures)}/{len(repairs)} repairs")
        return test_structures
