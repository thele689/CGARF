"""SRCD repair generation with dynamic resource allocation and initial patches.

This module now serves two purposes:

1. Keep the legacy mutation/template helpers used by older integration tests.
2. Provide a paper-aligned implementation of section 3.2.1:
   - allocate sampling budget from CG-MAD candidate credibility
   - build the initial patch generation prompt from issue + code + representative path
   - generate one initial Search/Replace patch per candidate location
"""

import re
import ast
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from dataclasses import asdict, dataclass, field
from enum import Enum
from loguru import logger

from src.common.llm_interface import LLMInterface
from src.common.data_structures import IssueContext, PathEvidence, PatchCandidate
from src.common.utils import validate_patch_format
from src.phase1_causal_analysis.causal_relevance_graph import (
    CausalRelevanceGraph,
    CodeEntity as GraphCodeEntity,
    EntityType as GraphEntityType,
)


class MutationType(Enum):
    """Types of code mutations"""
    NULL_CHECK = "null_check"
    PARAMETER_MODIFICATION = "parameter_modification"
    CONDITIONAL_WRAPPING = "conditional_wrapping"
    EXCEPTION_HANDLER = "exception_handler"
    VARIABLE_ASSIGNMENT = "variable_assignment"
    FUNCTION_CALL_ADDITION = "function_call_addition"


class RepairPattern(Enum):
    """Common repair patterns"""
    NULL_CHECK = "null_check"
    EXCEPTION_HANDLER = "exception_handler"
    FALLBACK = "fallback"
    VALIDATION = "validation"
    GUARD_CLAUSE = "guard_clause"


@dataclass
class RepairCandidate:
    """A candidate repair code"""
    id: str
    original_code: str
    repaired_code: str
    mutation_type: MutationType
    affected_lines: List[int]
    confidence: float
    pattern: Optional[RepairPattern] = None
    description: str = ""


@dataclass
class RepairTemplate:
    """Template for repair patterns"""
    pattern: RepairPattern
    description: str
    before_pattern: str  # Regex to match before
    after_template: str  # Template to replace with
    applicable_to_error_types: List[str]
    confidence: float


@dataclass
class SRCDCandidateInput:
    """Structured per-candidate input for SRCD initial generation."""

    candidate_id: str
    candidate_location: str
    file_path: str
    entity_type: str
    code_context: str
    representative_path_id: str
    representative_path_summary: str
    representative_path_evidence: Dict[str, Any] = field(default_factory=dict)
    candidate_credibility: float = 0.0
    normalized_weight: float = 0.0
    allocated_samples: int = 0


@dataclass
class SRCDInitialPatchBundle:
    """Output of section 3.2.1 dynamic allocation + initial patch generation."""

    issue_id: str
    total_sampling_budget: int
    candidate_inputs: List[SRCDCandidateInput] = field(default_factory=list)
    initial_patches: List[PatchCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "total_sampling_budget": self.total_sampling_budget,
            "candidate_inputs": [asdict(item) for item in self.candidate_inputs],
            "initial_patches": [asdict(item) for item in self.initial_patches],
        }


class PatchGenerationError(RuntimeError):
    """Raised when SRCD cannot produce a real non-no-op Search/Replace patch."""


class VariableBinding:
    """Handles variable scope and binding resolution"""
    
    def __init__(self, code: str):
        """
        Initialize variable binding analyzer
        
        Args:
            code: Source code to analyze
        """
        self.code = code
        self.variables = {}
        self.scopes = {}
        self.function_calls = {}
        self._analyze_code()
    
    def _analyze_code(self):
        """Analyze code to extract variable bindings"""
        
        try:
            tree = ast.parse(self.code)
        except SyntaxError:
            logger.warning("Cannot parse code for AST analysis")
            return
        
        # Extract variables and their scopes
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in self.variables:
                    self.variables[node.id] = {'occurrences': 0, 'type': 'unknown'}
                self.variables[node.id]['occurrences'] += 1
            
            elif isinstance(node, ast.FunctionDef):
                self.scopes[node.name] = {
                    'params': [arg.arg for arg in node.args.args],
                    'body_lines': (node.lineno, node.end_lineno) if hasattr(node, 'end_lineno') else None
                }
            
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name not in self.function_calls:
                        self.function_calls[func_name] = 0
                    self.function_calls[func_name] += 1
    
    def get_variables_at_line(self, line_num: int) -> Set[str]:
        """Get variables accessible at a specific line"""
        return set(self.variables.keys())
    
    def is_variable_safe_to_modify(self, var_name: str) -> bool:
        """Check if variable can be safely modified"""
        if var_name not in self.variables:
            return False
        # Safe if not a builtin
        return var_name not in {'print', 'len', 'range', 'str', 'int'}


class MutationStrategy:
    """Generates code mutations"""
    
    def __init__(self, llm: Optional[LLMInterface] = None):
        """
        Initialize mutation strategy
        
        Args:
            llm: LLM interface for assistance (optional)
        """
        self.llm = llm
        self.logger = logger
    
    def generate_null_checks(self, code: str, variables: List[str]) -> List[str]:
        """
        Generate null check mutations
        
        Args:
            code: Original code
            variables: Variables to add null checks for
        
        Returns:
            List of mutated code variants
        """
        mutations = []
        
        lines = code.split('\n')
        
        for var in variables:
            # Find first usage of variable
            for i, line in enumerate(lines):
                if var in line and not line.strip().startswith('#'):
                    # Create null check mutation
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    
                    # Python style null check
                    check = f"{indent_str}if {var} is not None:"
                    
                    mutated_lines = lines[:i] + [check, indent_str + '    ' + line.lstrip()] + lines[i+1:]
                    mutations.append('\n'.join(mutated_lines))
                    break
        
        return mutations
    
    def generate_exception_handlers(self, code: str, error_types: List[str] = None) -> List[str]:
        """
        Generate exception handler mutations
        
        Args:
            code: Original code
            error_types: Exception types to handle
        
        Returns:
            List of mutated code with try-except
        """
        if error_types is None:
            error_types = ['Exception', 'ValueError', 'TypeError', 'AttributeError']
        
        mutations = []
        
        lines = code.split('\n')
        
        for error_type in error_types[:2]:  # Limit to 2 types
            # Wrap entire code in try-except
            indent_str = '    '
            wrapped = ['try:']
            
            for line in lines:
                if line.strip():
                    wrapped.append(indent_str + line)
                else:
                    wrapped.append('')
            
            wrapped.extend([
                f"except {error_type}:",
                f"{indent_str}# Handle {error_type}",
                f"{indent_str}pass"
            ])
            
            mutations.append('\n'.join(wrapped))
        
        return mutations
    
    def generate_fallback_mutations(self, code: str, variable: str, default_value: str = 'None') -> List[str]:
        """
        Generate fallback assignment mutations
        
        Args:
            code: Original code
            variable: Variable to add fallback for
            default_value: Default value to assign
        
        Returns:
            List of mutated code with fallbacks
        """
        mutations = []
        
        # Pattern 1: var = var or default
        fallback1 = f"{variable} = {variable} or {default_value}"
        mutations.append(code + '\n' + fallback1)
        
        # Pattern 2: var = default if var is None else var
        fallback2 = f"{variable} = {default_value} if {variable} is None else {variable}"
        mutations.append(code + '\n' + fallback2)
        
        return mutations
    
    def generate_all_mutations(
        self,
        code: str,
        issue_location: str,
        issue_type: str = 'NullPointerException'
    ) -> List[str]:
        """
        Generate all mutation types for a code snippet
        
        Args:
            code: Original code
            issue_location: Location of issue
            issue_type: Type of error (NullPointerException, ValueError, etc.)
        
        Returns:
            List of all mutation variants (up to 50)
        """
        
        all_mutations = []
        
        # Extract variables from code
        try:
            binding = VariableBinding(code)
            variables = list(binding.variables.keys())[:5]  # Top 5 variables
        except:
            variables = []
        
        # Generate null checks
        if 'Null' in issue_type or 'None' in issue_type:
            null_checks = self.generate_null_checks(code, variables)
            all_mutations.extend(null_checks)
        
        # Generate exception handlers
        exc_handlers = self.generate_exception_handlers(code)
        all_mutations.extend(exc_handlers)
        
        # Generate fallbacks
        for var in variables:
            fallbacks = self.generate_fallback_mutations(code, var)
            all_mutations.extend(fallbacks)
        
        # Limit to 50 mutations
        return all_mutations[:50]


class TemplateStrategy:
    """Template-based repair generation"""
    
    def __init__(self, llm: Optional[LLMInterface] = None):
        """
        Initialize template strategy
        
        Args:
            llm: LLM interface for pattern matching
        """
        self.llm = llm
        self.logger = logger
        self.templates = self._load_repair_templates()
    
    def _load_repair_templates(self) -> List[RepairTemplate]:
        """Load repair templates"""
        
        templates = [
            RepairTemplate(
                pattern=RepairPattern.NULL_CHECK,
                description="Add null check before accessing",
                before_pattern=r'(\w+)\.(\w+)\(',
                after_template=r'if \1 is not None: \g<0>',
                applicable_to_error_types=['NullPointerException', 'AttributeError'],
                confidence=0.8
            ),
            RepairTemplate(
                pattern=RepairPattern.EXCEPTION_HANDLER,
                description="Wrap in try-except block",
                before_pattern=r'(\w+)\(',
                after_template=r'try:\n    \g<0>\nexcept Exception:\n    pass',
                applicable_to_error_types=['Exception', 'ValueError'],
                confidence=0.7
            ),
            RepairTemplate(
                pattern=RepairPattern.FALLBACK,
                description="Add fallback value",
                before_pattern=r'(\w+) = (\w+)',
                after_template=r'\1 = \2 or None',
                applicable_to_error_types=['NullPointerException'],
                confidence=0.75
            ),
            RepairTemplate(
                pattern=RepairPattern.VALIDATION,
                description="Add validation check",
                before_pattern=r'(\w+)\(',
                after_template=r'if isinstance(\1, (str, int, float)): \g<0>',
                applicable_to_error_types=['TypeError', 'ValueError'],
                confidence=0.7
            ),
            RepairTemplate(
                pattern=RepairPattern.GUARD_CLAUSE,
                description="Add early return guard",
                before_pattern=r'def (\w+)\(',
                after_template=r'def \1(\n    # Guard clauses\g<0>',
                applicable_to_error_types=['NullPointerException'],
                confidence=0.8
            ),
        ]
        
        return templates
    
    def get_applicable_templates(
        self,
        issue_type: str,
        code_snippet: str
    ) -> List[RepairTemplate]:
        """
        Get applicable templates for issue
        
        Args:
            issue_type: Type of error (NullPointerException, ValueError, etc.)
            code_snippet: Code context
        
        Returns:
            List of applicable templates
        """
        
        applicable = []
        
        for template in self.templates:
            if issue_type in template.applicable_to_error_types:
                # Check if pattern matches code
                if re.search(template.before_pattern, code_snippet):
                    applicable.append(template)
        
        return applicable
    
    def apply_template(
        self,
        code: str,
        template: RepairTemplate
    ) -> Optional[str]:
        """
        Apply template to code
        
        Args:
            code: Original code
            template: Template to apply
        
        Returns:
            Modified code or None if template doesn't apply
        """
        
        try:
            result = re.sub(
                template.before_pattern,
                template.after_template,
                code,
                count=1
            )
            
            # Verify it's different
            if result != code:
                return result
        
        except Exception as e:
            self.logger.warning(f"Template application failed: {e}")
        
        return None
    
    def generate_from_templates(
        self,
        code: str,
        issue_type: str,
        issue_description: str
    ) -> List[str]:
        """
        Generate repairs using templates
        
        Args:
            code: Original code
            issue_type: Type of error
            issue_description: Description of issue
        
        Returns:
            List of repair variants
        """
        
        repairs = []
        
        # Get applicable templates
        templates = self.get_applicable_templates(issue_type, code)
        
        # Apply each template
        for template in templates:
            repaired = self.apply_template(code, template)
            if repaired:
                repairs.append(repaired)
        
        return repairs


class RepairGenerator:
    """Main orchestrator for repair code generation"""
    
    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        max_mutations: int = 50,
        default_total_sampling_budget: Optional[int] = None,
    ):
        """
        Initialize repair generator
        
        Args:
            llm: LLM interface for semantic operations
        """
        self.llm = llm
        self.logger = logger
        self.max_mutations = max_mutations
        self.default_total_sampling_budget = (
            default_total_sampling_budget if default_total_sampling_budget is not None else max_mutations
        )
        self.mutation_strategy = MutationStrategy(llm)
        self.template_strategy = TemplateStrategy(llm)

    # ------------------------------------------------------------------
    # Paper 3.2.1: candidate-credibility-driven dynamic resource allocation
    # ------------------------------------------------------------------

    def allocate_sampling_budget(
        self,
        candidate_inputs: List[SRCDCandidateInput],
        total_sampling_budget: int,
    ) -> List[SRCDCandidateInput]:
        """
        Allocate per-candidate sampling counts from credibility:

            w_i = Cred(l_i) / sum_j Cred(l_j)
            n_i ≈ w_i * N

        Implementation detail:
        - use largest-remainder rounding so allocations sum exactly to N
        - when N >= K, ensure every candidate keeps at least one initial slot,
          matching section 3.2.1's "first generate one initial patch per location"
        """

        if total_sampling_budget <= 0:
            raise ValueError("total_sampling_budget must be positive")
        if not candidate_inputs:
            return []

        sorted_inputs = sorted(
            candidate_inputs,
            key=lambda item: item.candidate_credibility,
            reverse=True,
        )
        total_credibility = sum(max(item.candidate_credibility, 0.0) for item in sorted_inputs)
        if total_credibility <= 0:
            uniform_weight = 1.0 / len(sorted_inputs)
            weights = [uniform_weight] * len(sorted_inputs)
        else:
            weights = [
                max(item.candidate_credibility, 0.0) / total_credibility
                for item in sorted_inputs
            ]

        raw_allocations = [weight * total_sampling_budget for weight in weights]
        allocations = [int(value) for value in raw_allocations]
        remainders = [value - int(value) for value in raw_allocations]

        remaining = total_sampling_budget - sum(allocations)
        for index in sorted(
            range(len(sorted_inputs)),
            key=lambda idx: (remainders[idx], weights[idx], -idx),
            reverse=True,
        )[:remaining]:
            allocations[index] += 1

        if total_sampling_budget >= len(sorted_inputs):
            zero_indices = [idx for idx, value in enumerate(allocations) if value == 0]
            donor_indices = sorted(
                range(len(sorted_inputs)),
                key=lambda idx: (allocations[idx], remainders[idx], weights[idx]),
                reverse=True,
            )
            for zero_idx in zero_indices:
                donor_idx = next(
                    (idx for idx in donor_indices if allocations[idx] > 1),
                    None,
                )
                if donor_idx is None:
                    break
                allocations[donor_idx] -= 1
                allocations[zero_idx] += 1

        enriched: List[SRCDCandidateInput] = []
        for item, weight, count in zip(sorted_inputs, weights, allocations):
            enriched.append(
                SRCDCandidateInput(
                    candidate_id=item.candidate_id,
                    candidate_location=item.candidate_location,
                    file_path=item.file_path,
                    entity_type=item.entity_type,
                    code_context=item.code_context,
                    representative_path_id=item.representative_path_id,
                    representative_path_summary=item.representative_path_summary,
                    representative_path_evidence=dict(item.representative_path_evidence),
                    candidate_credibility=item.candidate_credibility,
                    normalized_weight=weight,
                    allocated_samples=count,
                )
            )
        return enriched

    def build_candidate_inputs_from_cgmad(
        self,
        cg_mad_result: Dict[str, Any],
        crg: CausalRelevanceGraph,
        max_candidates: Optional[int] = None,
    ) -> List[SRCDCandidateInput]:
        """Build SRCD candidate inputs from CG-MAD output and phase1 CRG."""

        path_summaries = {
            item["path_id"]: item
            for item in cg_mad_result.get("path_summaries", [])
        }

        candidate_inputs: List[SRCDCandidateInput] = []
        assessments = list(cg_mad_result.get("candidate_assessments", []))
        assessments.sort(key=lambda item: item.get("final_credibility", 0.0), reverse=True)
        if max_candidates is not None:
            assessments = assessments[:max_candidates]

        for assessment in assessments:
            candidate_id = assessment["candidate_id"]
            representative_path = path_summaries.get(assessment.get("representative_path_id", ""))
            entity = self._resolve_candidate_entity(crg, candidate_id)
            if entity is None:
                self.logger.warning(f"Skipping SRCD candidate without entity binding: {candidate_id}")
                continue

            code_context = self._load_code_context(
                entity.file_path,
                (entity.line_start or 0, entity.line_end or 0),
            )
            path_summary_text = ""
            path_evidence: Dict[str, Any] = {}
            if representative_path:
                path_summary_text = representative_path.get("compressed_text", "")
                path_evidence = dict(representative_path.get("evidence_pack", {}))

            candidate_inputs.append(
                SRCDCandidateInput(
                    candidate_id=candidate_id,
                    candidate_location=self._format_candidate_location(entity),
                    file_path=entity.file_path,
                    entity_type=entity.entity_type.value,
                    code_context=code_context,
                    representative_path_id=assessment.get("representative_path_id", ""),
                    representative_path_summary=path_summary_text,
                    representative_path_evidence=path_evidence,
                    candidate_credibility=float(assessment.get("final_credibility", 0.0)),
                )
            )

        return candidate_inputs

    def _resolve_candidate_entity(
        self,
        crg: CausalRelevanceGraph,
        candidate_id: str,
    ) -> Optional[Any]:
        """Resolve candidate ids across runs whose absolute repo roots may differ."""

        entity = crg.code_graph.get_entity(candidate_id)
        if entity is not None:
            return entity

        candidate_file, candidate_symbol, candidate_type = self._split_candidate_id(candidate_id)
        candidate_suffix = candidate_file.replace("\\", "/").split("/repos/")[-1]

        for current_id, current_entity in crg.code_graph.entities.items():
            current_file, current_symbol, current_type = self._split_candidate_id(current_id)
            current_suffix = current_file.replace("\\", "/").split("/repos/")[-1]
            if (
                current_symbol == candidate_symbol
                and current_type == candidate_type
                and current_suffix.endswith(candidate_suffix)
            ):
                return current_entity

        return self._synthesize_entity_from_candidate_id(candidate_id)

    def _split_candidate_id(self, candidate_id: str) -> Tuple[str, str, str]:
        if "::" in candidate_id:
            file_path, remainder = candidate_id.split("::", 1)
            if ":" in remainder:
                symbol_name, symbol_type = remainder.rsplit(":", 1)
                return file_path, symbol_name, symbol_type
            return file_path, remainder, ""
        return candidate_id, "", ""

    def _synthesize_entity_from_candidate_id(self, candidate_id: str) -> Optional[GraphCodeEntity]:
        """
        Build a lightweight entity directly from phase2 ids when code-graph ids
        come from a different workspace root.
        """

        file_path, symbol_name, symbol_type = self._split_candidate_id(candidate_id)
        path = Path(file_path)
        if not path.exists():
            return None

        line_start, line_end = self._find_symbol_line_range(path, symbol_name, symbol_type)
        entity_type = self._candidate_type_to_entity_type(symbol_type)
        return GraphCodeEntity(
            id=candidate_id,
            name=symbol_name or path.name,
            entity_type=entity_type,
            file_path=str(path),
            function_name=symbol_name if entity_type == GraphEntityType.FUNCTION else None,
            class_name=symbol_name if entity_type == GraphEntityType.CLASS else None,
            variable_name=symbol_name if entity_type == GraphEntityType.VARIABLE else None,
            line_start=line_start,
            line_end=line_end,
        )

    def _candidate_type_to_entity_type(self, symbol_type: str) -> GraphEntityType:
        normalized = (symbol_type or "").lower()
        if normalized == "class":
            return GraphEntityType.CLASS
        if normalized == "variable":
            return GraphEntityType.VARIABLE
        return GraphEntityType.FUNCTION

    def _find_symbol_line_range(
        self,
        file_path: Path,
        symbol_name: str,
        symbol_type: str,
    ) -> Tuple[int, int]:
        try:
            source = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = file_path.read_text(encoding="latin-1")
        except OSError:
            return (0, 0)

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return (0, 0)

        target_is_class = (symbol_type or "").lower() == "class"
        target_is_variable = (symbol_type or "").lower() == "variable"

        for node in ast.walk(tree):
            if target_is_class and isinstance(node, ast.ClassDef) and node.name == symbol_name:
                return (node.lineno, getattr(node, "end_lineno", node.lineno))
            if not target_is_class and not target_is_variable and isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name == symbol_name:
                return (node.lineno, getattr(node, "end_lineno", node.lineno))
            if target_is_variable and isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == symbol_name:
                        return (node.lineno, getattr(node, "end_lineno", node.lineno))

        return (0, 0)

    def generate_initial_patches_from_cgmad(
        self,
        issue_context: IssueContext,
        cg_mad_result: Dict[str, Any],
        crg: CausalRelevanceGraph,
        total_sampling_budget: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> SRCDInitialPatchBundle:
        """
        Section 3.2.1 end-to-end helper:
        1. Build candidate inputs from CG-MAD credibility + representative paths
        2. Allocate dynamic sampling budget
        3. Generate exactly one initial Search/Replace patch per candidate
        """

        candidate_inputs = self.build_candidate_inputs_from_cgmad(
            cg_mad_result=cg_mad_result,
            crg=crg,
            max_candidates=max_candidates,
        )
        budget = total_sampling_budget or self.default_total_sampling_budget
        allocated_inputs = self.allocate_sampling_budget(candidate_inputs, budget)

        initial_patches: List[PatchCandidate] = []
        successful_inputs: List[SRCDCandidateInput] = []
        for index, candidate_input in enumerate(allocated_inputs):
            try:
                patch = self.generate_initial_patch(
                    issue_context=issue_context,
                    candidate_input=candidate_input,
                    patch_index=index + 1,
                )
                initial_patches.append(patch)
                successful_inputs.append(candidate_input)
            except PatchGenerationError as exc:
                self.logger.warning(
                    f"Skipping candidate without a valid non-no-op patch: "
                    f"{candidate_input.candidate_id}: {exc}"
                )

        return SRCDInitialPatchBundle(
            issue_id=issue_context.id,
            total_sampling_budget=budget,
            candidate_inputs=successful_inputs,
            initial_patches=initial_patches,
        )

    def generate_initial_patch(
        self,
        issue_context: IssueContext,
        candidate_input: SRCDCandidateInput,
        patch_index: int = 1,
    ) -> PatchCandidate:
        """Generate the initial Search/Replace patch for one candidate location."""

        prompt = self._build_initial_patch_prompt(issue_context, candidate_input)
        raw_response = ""
        if self.llm:
            try:
                raw_response = self.llm.generate(prompt, temperature=0.2, max_tokens=1400)
            except Exception as exc:
                self.logger.warning(
                    f"Initial SRCD patch generation failed for {candidate_input.candidate_id}: {exc}"
                )

        patch_content = self._normalize_search_replace_output(raw_response)
        patch_is_bad = (
            not patch_content
            or not self._is_python_search_replace_patch_valid(patch_content)
            or self._is_overbroad_exception_wrapper_patch(patch_content)
            or self._is_no_op_patch(patch_content)
        )
        if patch_is_bad and self.llm:
            retry_prompt = self._build_initial_patch_retry_prompt(issue_context, candidate_input)
            retry_response = ""
            try:
                retry_response = self.llm.generate(retry_prompt, temperature=0.15, max_tokens=1400)
            except Exception as exc:
                self.logger.warning(
                    f"Initial SRCD retry generation failed for {candidate_input.candidate_id}: {exc}"
                )
            retry_patch = self._normalize_search_replace_output(retry_response)
            if (
                retry_patch
                and self._is_python_search_replace_patch_valid(retry_patch)
                and not self._is_overbroad_exception_wrapper_patch(retry_patch)
                and not self._is_no_op_patch(retry_patch)
            ):
                patch_content = retry_patch
                patch_is_bad = False

        if patch_is_bad and self.llm:
            forced_retry_prompt = self._build_initial_patch_forced_edit_prompt(
                issue_context,
                candidate_input,
            )
            forced_retry_response = ""
            try:
                forced_retry_response = self.llm.generate(
                    forced_retry_prompt,
                    temperature=0.35,
                    max_tokens=1400,
                )
            except Exception as exc:
                self.logger.warning(
                    f"Initial SRCD forced-edit retry failed for {candidate_input.candidate_id}: {exc}"
                )
            forced_retry_patch = self._normalize_search_replace_output(forced_retry_response)
            if (
                forced_retry_patch
                and self._is_python_search_replace_patch_valid(forced_retry_patch)
                and not self._is_overbroad_exception_wrapper_patch(forced_retry_patch)
                and not self._is_no_op_patch(forced_retry_patch)
            ):
                patch_content = forced_retry_patch
                patch_is_bad = False

        if patch_is_bad:
            patch_content = self._build_fallback_search_replace(issue_context, candidate_input)
            patch_is_bad = (
                not patch_content
                or not self._is_python_search_replace_patch_valid(patch_content)
                or self._is_overbroad_exception_wrapper_patch(patch_content)
                or self._is_no_op_patch(patch_content)
            )

        if patch_is_bad:
            raise PatchGenerationError(
                "initial generation exhausted LLM retries and deterministic fallback "
                "without a valid non-no-op patch"
            )

        return PatchCandidate(
            patch_id=f"{issue_context.id}::{candidate_input.candidate_id}::initial::{patch_index}",
            location=candidate_input.candidate_id,
            patch_content=patch_content,
            generated_round=1,
            credibility_from_location=candidate_input.candidate_credibility,
        )

    def generate_refined_patch(
        self,
        issue_context: IssueContext,
        candidate_input: SRCDCandidateInput,
        reflection_payload: Dict[str, Any],
        generation_temperature: float,
        patch_index: int,
    ) -> PatchCandidate:
        """Generate a fresh next-round patch guided by section 3.2.2 reflection."""

        prompt = self._build_refinement_patch_prompt(
            issue_context=issue_context,
            candidate_input=candidate_input,
            reflection_payload=reflection_payload,
        )
        raw_response = ""
        if self.llm:
            try:
                raw_response = self.llm.generate(
                    prompt,
                    temperature=generation_temperature,
                    max_tokens=1600,
                )
            except Exception as exc:
                self.logger.warning(
                    f"Refined SRCD patch generation failed for {candidate_input.candidate_id}: {exc}"
                )

        patch_content = self._normalize_search_replace_output(raw_response)
        patch_is_bad = (
            not patch_content
            or not self._is_python_search_replace_patch_valid(patch_content)
            or self._is_overbroad_exception_wrapper_patch(patch_content)
            or self._is_no_op_patch(patch_content)
        )
        if patch_is_bad and self.llm:
            retry_prompt = self._build_refinement_patch_retry_prompt(
                issue_context=issue_context,
                candidate_input=candidate_input,
                reflection_payload=reflection_payload,
            )
            retry_response = ""
            try:
                retry_response = self.llm.generate(
                    retry_prompt,
                    temperature=min(0.95, generation_temperature + 0.1),
                    max_tokens=1600,
                )
            except Exception as exc:
                self.logger.warning(
                    f"Refined SRCD retry generation failed for {candidate_input.candidate_id}: {exc}"
                )
            retry_patch = self._normalize_search_replace_output(retry_response)
            if (
                retry_patch
                and self._is_python_search_replace_patch_valid(retry_patch)
                and not self._is_overbroad_exception_wrapper_patch(retry_patch)
                and not self._is_no_op_patch(retry_patch)
            ):
                patch_content = retry_patch
                patch_is_bad = False

        if patch_is_bad and self.llm:
            forced_retry_prompt = self._build_refinement_patch_forced_edit_prompt(
                issue_context=issue_context,
                candidate_input=candidate_input,
                reflection_payload=reflection_payload,
            )
            forced_retry_response = ""
            try:
                forced_retry_response = self.llm.generate(
                    forced_retry_prompt,
                    temperature=min(0.95, max(0.55, generation_temperature + 0.2)),
                    max_tokens=1600,
                )
            except Exception as exc:
                self.logger.warning(
                    f"Refined SRCD forced retry generation failed for {candidate_input.candidate_id}: {exc}"
                )
            forced_retry_patch = self._normalize_search_replace_output(forced_retry_response)
            if (
                forced_retry_patch
                and self._is_python_search_replace_patch_valid(forced_retry_patch)
                and not self._is_overbroad_exception_wrapper_patch(forced_retry_patch)
                and not self._is_no_op_patch(forced_retry_patch)
            ):
                patch_content = forced_retry_patch
                patch_is_bad = False

        if patch_is_bad:
            patch_content = self._build_fallback_search_replace(issue_context, candidate_input)
            patch_is_bad = (
                not patch_content
                or not self._is_python_search_replace_patch_valid(patch_content)
                or self._is_overbroad_exception_wrapper_patch(patch_content)
                or self._is_no_op_patch(patch_content)
            )

        if patch_is_bad:
            raise PatchGenerationError(
                "refinement generation exhausted LLM retries and deterministic fallback "
                "without a valid non-no-op patch"
            )

        return PatchCandidate(
            patch_id=f"{issue_context.id}::{candidate_input.candidate_id}::refined::{patch_index}",
            location=candidate_input.candidate_id,
            patch_content=patch_content,
            generated_round=patch_index,
            credibility_from_location=candidate_input.candidate_credibility,
        )

    def _build_initial_patch_prompt(
        self,
        issue_context: IssueContext,
        candidate_input: SRCDCandidateInput,
    ) -> str:
        """Prompt template aligned with appendix D.1 initial patch generation."""

        return f"""You are the initial patch generation agent in an automated program repair system.
Given an issue description, one candidate repair location, the local code context, and a representative causal path summary, generate one initial patch for this same candidate location.

Requirements:
1. The patch must be directly related to the reported failure.
2. The patch should stay aligned with the key mechanism implied by the representative path.
3. Keep the edit as local and minimal as possible.
4. Preserve unrelated semantics, interfaces, and control flow.
5. Do not wrap the whole function in a broad try/except block.
6. Do not move docstrings, add unrelated guards, rename symbols, or refactor unrelated code.
7. Even if this candidate may not be globally ideal, still produce the most plausible local fix at this same location.
8. Output must be strict Search/Replace format only, with no explanation.
9. SEARCH and REPLACE must not be identical. You must change at least one executable line.
10. Prefer the narrowest contiguous snippet that captures the real edit. It is okay to patch only a few indented lines instead of rewriting the whole function.

Issue description:
{issue_context.description}

Candidate location:
{candidate_input.candidate_location}

Candidate code context:
```python
{candidate_input.code_context}
```

Representative path summary:
{candidate_input.representative_path_summary}

Return exactly:
<<< SEARCH
...
===
...
>>> REPLACE
""".strip()

    def _build_initial_patch_retry_prompt(
        self,
        issue_context: IssueContext,
        candidate_input: SRCDCandidateInput,
    ) -> str:
        """Stricter retry prompt when the first initial patch is invalid or over-broad."""

        return f"""Generate a stricter replacement patch for the same candidate location.

The previous attempt was rejected because it was invalid, too broad, or wrapped the whole function.

Hard constraints:
- Stay at the same candidate location.
- Modify the smallest plausible local logic only.
- Do not add a broad try/except wrapper.
- Do not rewrite the whole function unless absolutely necessary.
- Prefer changing a condition, return expression, matrix-combination line, or a small local block.
- Output only strict Search/Replace format.
- SEARCH and REPLACE must differ.
- Prefer the narrowest contiguous snippet instead of a whole-function replacement.

Issue description:
{issue_context.description}

Candidate location:
{candidate_input.candidate_location}

Candidate code context:
```python
{candidate_input.code_context}
```

Representative path summary:
{candidate_input.representative_path_summary}

Return exactly:
<<< SEARCH
...
===
...
>>> REPLACE
""".strip()

    def _build_initial_patch_forced_edit_prompt(
        self,
        issue_context: IssueContext,
        candidate_input: SRCDCandidateInput,
    ) -> str:
        """Final retry that forces a concrete local change."""

        return f"""Produce one concrete local patch for the same candidate location.

Previous attempts were rejected because they were too broad or made no actual change.

Hard requirements:
- Stay at this candidate location only.
- Change at least one executable line.
- SEARCH and REPLACE must not be identical.
- Do not wrap the whole function in try/except.
- Prefer a small but real change to a condition, matrix composition, return expression, recursive combination, or local variable update.
- Output only strict Search/Replace format.
- You may patch only a small indented block; a whole-function replacement is not required.

Issue description:
{issue_context.description}

Candidate location:
{candidate_input.candidate_location}

Candidate code context:
```python
{candidate_input.code_context}
```

Representative path summary:
{candidate_input.representative_path_summary}

Return exactly:
<<< SEARCH
...
===
...
>>> REPLACE
""".strip()

    def _build_refinement_patch_prompt(
        self,
        issue_context: IssueContext,
        candidate_input: SRCDCandidateInput,
        reflection_payload: Dict[str, Any],
    ) -> str:
        """English prompt for iterative regeneration in section 3.2.2."""

        semantic = reflection_payload.get("semantic_consistency", {}) or {}
        causal = reflection_payload.get("causal_alignment", {}) or {}
        minimal = reflection_payload.get("minimal_edit", {}) or {}
        revision_suggestion = str(reflection_payload.get("revision_suggestion", "")).strip()
        edit_scaffold = self._build_candidate_edit_scaffold(candidate_input)

        return f"""You are the patch-regeneration agent in an automated program repair system.
You are generating a fresh new patch for the same candidate location after one structured reflection round.

Your goal is to regenerate a better Search/Replace patch that:
1. directly follows the revision suggestion,
2. stays aligned with the representative causal path,
3. preserves unrelated semantics,
4. keeps the edit as local as possible.

Important constraints:
- Edit only this candidate location.
- Treat the previous patch as rejected feedback context, not as a base you must continue editing.
- Regenerate a fresh patch from the issue, code context, representative path, and revision advice.
- Prefer changing the smallest necessary lines instead of wrapping the whole function.
- Do not add broad try/except blocks unless directly required by the failure mechanism.
- Do not rename symbols, refactor unrelated code, or add unrelated branches.
- If the reflection mentions another helper or function, translate that feedback into the closest plausible local fix at this candidate location instead of moving the edit elsewhere.
- Return only one Search/Replace patch and nothing else.

Issue description:
{issue_context.description}

Candidate location:
{candidate_input.candidate_location}

Candidate code context:
```python
{candidate_input.code_context}
```

Concrete local edit scaffold for this candidate:
{edit_scaffold}

Representative path summary:
{candidate_input.representative_path_summary}

Structured reflection on the rejected previous patch:
- Semantic consistency: {semantic.get("level", "unknown")} | {semantic.get("reason", "")}
- Causal alignment: {causal.get("level", "unknown")} | {causal.get("reason", "")}
- Minimal edit: {minimal.get("level", "unknown")} | {minimal.get("reason", "")}
- Revision suggestion: {revision_suggestion}

Generate a fresh new patch that directly addresses the weaknesses above.

Output format:
<<< SEARCH
...
===
...
>>> REPLACE
""".strip()

    def _build_refinement_patch_retry_prompt(
        self,
        issue_context: IssueContext,
        candidate_input: SRCDCandidateInput,
        reflection_payload: Dict[str, Any],
    ) -> str:
        """Stricter retry prompt for fresh regeneration after an invalid/no-op attempt."""

        semantic = reflection_payload.get("semantic_consistency", {}) or {}
        causal = reflection_payload.get("causal_alignment", {}) or {}
        minimal = reflection_payload.get("minimal_edit", {}) or {}
        revision_suggestion = str(reflection_payload.get("revision_suggestion", "")).strip()
        edit_scaffold = self._build_candidate_edit_scaffold(candidate_input)

        return f"""Generate a different fresh patch for the same candidate location.

The previous regeneration attempt was rejected because it was invalid, too broad, or made no real change.

Hard constraints:
- Stay at this same candidate location.
- Produce a fresh patch from the issue, code context, representative path, and revision advice.
- Change at least one executable line.
- SEARCH and REPLACE must not be identical.
- Do not wrap the whole function in try/except.
- Prefer a narrow local edit to a condition, return expression, matrix-combination line, recursive step, or local variable update.
- Do not refactor unrelated code or move the edit to another helper/function.
- Output only one strict Search/Replace patch.

Issue description:
{issue_context.description}

Candidate location:
{candidate_input.candidate_location}

Candidate code context:
```python
{candidate_input.code_context}
```

Concrete local edit scaffold for this candidate:
{edit_scaffold}

Representative path summary:
{candidate_input.representative_path_summary}

Structured reflection:
- Semantic consistency: {semantic.get("level", "unknown")} | {semantic.get("reason", "")}
- Causal alignment: {causal.get("level", "unknown")} | {causal.get("reason", "")}
- Minimal edit: {minimal.get("level", "unknown")} | {minimal.get("reason", "")}
- Revision suggestion: {revision_suggestion}

Return exactly:
<<< SEARCH
...
===
...
>>> REPLACE
""".strip()

    def _build_refinement_patch_forced_edit_prompt(
        self,
        issue_context: IssueContext,
        candidate_input: SRCDCandidateInput,
        reflection_payload: Dict[str, Any],
    ) -> str:
        """Final retry that forces a concrete fresh local edit after reflection."""

        semantic = reflection_payload.get("semantic_consistency", {}) or {}
        causal = reflection_payload.get("causal_alignment", {}) or {}
        minimal = reflection_payload.get("minimal_edit", {}) or {}
        revision_suggestion = str(reflection_payload.get("revision_suggestion", "")).strip()
        edit_scaffold = self._build_candidate_edit_scaffold(candidate_input)

        return f"""Produce one concrete fresh patch for this same candidate location.

Previous regeneration attempts were rejected because they were invalid, too broad, or made no actual change.

Hard requirements:
- Stay at this candidate location only.
- Generate a fresh patch from the issue, code context, representative path, and revision advice.
- Change at least one executable line.
- SEARCH and REPLACE must not be identical.
- Do not wrap the whole function in try/except.
- Prefer a small but real change to a condition, return expression, matrix combination, recursive combination, or local variable update.
- Do not rewrite unrelated logic, rename symbols, or move the fix to another function.
- Output only strict Search/Replace format.
- A narrow local snippet is preferred over a whole-function replacement.

Issue description:
{issue_context.description}

Candidate location:
{candidate_input.candidate_location}

Candidate code context:
```python
{candidate_input.code_context}
```

Concrete local edit scaffold for this candidate:
{edit_scaffold}

Representative path summary:
{candidate_input.representative_path_summary}

Structured reflection:
- Semantic consistency: {semantic.get("level", "unknown")} | {semantic.get("reason", "")}
- Causal alignment: {causal.get("level", "unknown")} | {causal.get("reason", "")}
- Minimal edit: {minimal.get("level", "unknown")} | {minimal.get("reason", "")}
- Revision suggestion: {revision_suggestion}

Return exactly:
<<< SEARCH
...
===
...
>>> REPLACE
""".strip()

    def _build_candidate_edit_scaffold(self, candidate_input: SRCDCandidateInput) -> str:
        """Derive local edit anchors so regeneration stays inside the current candidate."""

        candidate_name = self._extract_candidate_name(candidate_input.candidate_id)
        anchors: List[str] = []

        for lineno, line in enumerate(candidate_input.code_context.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("#", '"""', "'''")):
                continue
            if any(
                marker in stripped
                for marker in (
                    "if ",
                    "elif ",
                    "return ",
                    "isinstance(",
                    "_coord_matrix",
                    "_separable",
                    "_cstack",
                    "_compute_n_outputs",
                    "CompoundModel",
                    ".left",
                    ".right",
                    "np.",
                )
            ):
                anchors.append(f"- L{lineno}: `{stripped[:140]}`")

        if not anchors:
            for lineno, line in enumerate(candidate_input.code_context.splitlines(), start=1):
                stripped = line.strip()
                if stripped and not stripped.startswith(("#", '"""', "'''")):
                    anchors.append(f"- L{lineno}: `{stripped[:140]}`")
                if len(anchors) >= 4:
                    break

        related_symbols: List[str] = []
        evidence_nodes = candidate_input.representative_path_evidence.get("nodes", [])
        for node in evidence_nodes:
            name = str(node.get("name") or "").strip()
            file_path = str(node.get("file_path") or "")
            if (
                name
                and file_path == candidate_input.file_path
                and name != candidate_name
                and name not in related_symbols
            ):
                related_symbols.append(name)
            if len(related_symbols) >= 4:
                break

        related_text = ", ".join(related_symbols) if related_symbols else "none"
        anchor_text = "\n".join(anchors[:6]) if anchors else "- No precise anchor extracted; edit the smallest local executable block."
        return (
            f"- Candidate symbol: `{candidate_name}`\n"
            f"- Same-file related symbols from causal evidence: {related_text}\n"
            f"- Prefer SEARCH snippets anchored at one of these local statements:\n"
            f"{anchor_text}\n"
            f"- Do not replace the whole function unless no listed local statement can express the fix."
        )

    def _extract_candidate_name(self, candidate_id: str) -> str:
        _, symbol_name, _ = self._split_candidate_id(candidate_id)
        return symbol_name or candidate_id

    def _normalize_search_replace_output(self, raw_response: str) -> Optional[str]:
        """Extract a valid Search/Replace patch block from model output."""

        if not raw_response:
            return None
        match = re.search(
            r'(<<<\s*SEARCH\n.*?\n===\n.*?\n>>>\s*REPLACE)',
            raw_response,
            re.DOTALL,
        )
        if not match:
            return None
        patch = match.group(1).strip()
        return patch if validate_patch_format(patch) else None

    def _is_overbroad_exception_wrapper_patch(self, patch: str) -> bool:
        """Detect common degenerate patches that wrap the whole function in try/except."""

        search_block, replace_block = self._extract_patch_blocks(patch)
        if not search_block or not replace_block:
            return False

        search_lines = search_block.splitlines()
        replace_lines = replace_block.splitlines()
        if not search_lines or not replace_lines:
            return False

        header = search_lines[0].lstrip()
        replace_first = replace_lines[0].lstrip()
        if not (header.startswith("def ") or header.startswith("async def ")):
            return False
        if replace_first != header:
            return False

        body_lines = [line for line in replace_lines[1:] if line.strip()]
        if not body_lines:
            return False

        first_body = body_lines[0].lstrip()
        has_try = first_body.startswith("try:")
        has_catchall = any(line.lstrip().startswith("except Exception") for line in replace_lines)
        return has_try and has_catchall

    def _is_no_op_patch(self, patch: str) -> bool:
        """Reject patches whose SEARCH and REPLACE blocks are identical."""

        search_block, replace_block = self._extract_patch_blocks(patch)
        return (
            bool(search_block)
            and self._normalize_patch_snippet_for_compare(search_block)
            == self._normalize_patch_snippet_for_compare(replace_block)
        )

    def _extract_patch_blocks(self, patch_text: str) -> Tuple[str, str]:
        match = re.search(
            r'<<<\s*SEARCH\n(.*?)\n===\n(.*?)\n>>>\s*REPLACE',
            patch_text,
            re.DOTALL,
        )
        if not match:
            return ("", patch_text)
        return (match.group(1).rstrip(), match.group(2).rstrip())

    def _is_python_search_replace_patch_valid(self, patch: str) -> bool:
        """Basic syntax gate for function-level Python Search/Replace patches."""

        match = re.search(
            r'<<<\s*SEARCH\n(.*?)\n===\n(.*?)\n>>>\s*REPLACE',
            patch,
            re.DOTALL,
        )
        if not match:
            return False

        replace_block = match.group(2).rstrip()
        if not replace_block:
            return False

        return self._is_python_snippet_valid(replace_block)

    def _is_python_snippet_valid(self, snippet: str) -> bool:
        """Allow both module-level snippets and indented in-function snippets."""

        candidate = textwrap.dedent(snippet).strip()
        if not candidate:
            return False

        try:
            ast.parse(candidate)
            return True
        except SyntaxError:
            pass

        wrapped = "def _temporary_wrapper():\n"
        for line in candidate.splitlines():
            wrapped += f"    {line}\n" if line.strip() else "\n"

        try:
            ast.parse(wrapped)
            return True
        except SyntaxError:
            return False

    def _normalize_patch_snippet_for_compare(self, snippet: str) -> str:
        return textwrap.dedent(snippet).strip()

    def _build_fallback_search_replace(
        self,
        issue_context: IssueContext,
        candidate_input: SRCDCandidateInput,
    ) -> Optional[str]:
        """
        Deterministic fallback used only when LLM output is missing/invalid.

        It reuses the existing mutation/template strategies and converts the first
        non-trivial local variant into Search/Replace form.
        """

        original = candidate_input.code_context
        error_type = self._extract_error_type(issue_context.description)

        repaired_variants = self.template_strategy.generate_from_templates(
            original,
            error_type,
            issue_context.description,
        )
        if not repaired_variants:
            repaired_variants = self.mutation_strategy.generate_all_mutations(
                original,
                candidate_input.candidate_location,
                error_type,
            )

        for replacement in repaired_variants:
            if not replacement or replacement == original:
                continue
            patch = self._format_search_replace_patch(original, replacement)
            if (
                self._is_python_search_replace_patch_valid(patch)
                and not self._is_overbroad_exception_wrapper_patch(patch)
                and not self._is_no_op_patch(patch)
            ):
                return patch
        return None

    def _format_search_replace_patch(self, search_block: str, replace_block: str) -> str:
        return f"<<< SEARCH\n{search_block.rstrip()}\n===\n{replace_block.rstrip()}\n>>> REPLACE"

    def _build_function_wrapper_fallback(self, original: str) -> Optional[str]:
        """Create a syntactically valid minimal fallback for function snippets."""

        lines = original.splitlines()
        if not lines:
            return None
        header = lines[0]
        if not header.lstrip().startswith("def ") and not header.lstrip().startswith("async def "):
            return None

        body = lines[1:]
        base_indent = None
        for line in body:
            stripped = line.lstrip()
            if stripped:
                base_indent = line[: len(line) - len(stripped)]
                break
        if base_indent is None:
            base_indent = "    "

        wrapped_lines = [header, f"{base_indent}try:"]
        for line in body:
            if line.strip():
                wrapped_lines.append(f"{base_indent}    {line[len(base_indent):] if line.startswith(base_indent) else line.lstrip()}")
            else:
                wrapped_lines.append("")
        wrapped_lines.append(f"{base_indent}except Exception:")
        wrapped_lines.append(f"{base_indent}    raise")
        candidate = "\n".join(wrapped_lines)

        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            return None

    def _format_candidate_location(self, entity: Any) -> str:
        parts = [entity.file_path]
        if getattr(entity, "class_name", None):
            parts.append(entity.class_name)
        if getattr(entity, "function_name", None):
            parts.append(entity.function_name)
        parts.append(entity.entity_type.value)
        return "::".join(str(part) for part in parts if part)

    def _load_code_context(self, file_path: str, line_range: Tuple[int, int]) -> str:
        """Load a code snippet from disk when the graph cache lacks code_snippet."""

        path = Path(file_path)
        if not path.exists():
            return ""
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(encoding="latin-1").splitlines()

        start, end = line_range
        if start <= 0 or end <= 0 or start > len(lines):
            return "\n".join(lines[: min(len(lines), 60)])
        bounded_end = min(end, len(lines))
        return "\n".join(lines[start - 1:bounded_end])
    
    def generate_repairs(
        self,
        code: str,
        bug_location: str,
        issue_context: IssueContext,
        crg_path: Optional[PathEvidence] = None,
        max_mutations: int = 50
    ) -> List[RepairCandidate]:
        """
        Generate repair candidates for a bug location
        
        Args:
            code: Original code snippet
            bug_location: Location identifier (function/class name)
            issue_context: Issue description and context
            crg_path: CRG path (causal context)
            max_mutations: Maximum mutations to generate
        
        Returns:
            List of repair candidates sorted by confidence
        """
        
        self.logger.info(f"Generating repairs for {bug_location}")
        
        repairs = []
        repair_id_counter = 0
        
        # Extract error type from issue
        error_type = self._extract_error_type(issue_context.description)
        
        # Strategy 1: Mutation-based generation
        mutations = self.mutation_strategy.generate_all_mutations(
            code, bug_location, error_type
        )
        
        for i, mutated_code in enumerate(mutations[:max_mutations // 2]):
            repair = RepairCandidate(
                id=f"repair_{repair_id_counter}",
                original_code=code,
                repaired_code=mutated_code,
                mutation_type=MutationType.NULL_CHECK if 'if' in mutated_code else MutationType.EXCEPTION_HANDLER,
                affected_lines=self._find_affected_lines(code, mutated_code),
                confidence=0.6 + (0.01 * i),  # Confidence decreases with each mutation
                description=f"Mutation: {error_type}"
            )
            repairs.append(repair)
            repair_id_counter += 1
        
        # Strategy 2: Template-based generation
        templates = self.template_strategy.get_applicable_templates(error_type, code)
        
        for template in templates:
            repaired_code = self.template_strategy.apply_template(code, template)
            if repaired_code:
                repair = RepairCandidate(
                    id=f"repair_{repair_id_counter}",
                    original_code=code,
                    repaired_code=repaired_code,
                    mutation_type=MutationType.CONDITIONAL_WRAPPING,
                    affected_lines=self._find_affected_lines(code, repaired_code),
                    confidence=template.confidence,
                    pattern=template.pattern,
                    description=f"Template: {template.description}"
                )
                repairs.append(repair)
                repair_id_counter += 1
        
        # Strategy 3: LLM-guided generation (if available)
        if self.llm:
            try:
                llm_repairs = self._generate_llm_repairs(
                    code, issue_context, bug_location
                )
                repairs.extend(llm_repairs[:max_mutations // 4])
            except Exception as e:
                self.logger.warning(f"LLM repair generation failed: {e}")
        
        # Scale to max_mutations
        repairs = repairs[:max_mutations]
        
        # Sort by confidence
        repairs.sort(key=lambda r: r.confidence, reverse=True)
        
        self.logger.info(f"Generated {len(repairs)} repair candidates")
        
        return repairs
    
    def _extract_error_type(self, issue_description: str) -> str:
        """Extract error type from issue description"""
        
        # Common error patterns
        patterns = [
            r'(NullPointerException|AttributeError)',
            r'(ValueError|TypeError)',
            r'(IndexError)',
            r'(KeyError)',
            r'(FileNotFoundError)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, issue_description)
            if match:
                return match.group(1)
        
        return 'Exception'
    
    def _find_affected_lines(self, original: str, repaired: str) -> List[int]:
        """Find which lines were affected by repair"""
        
        original_lines = original.split('\n')
        repaired_lines = repaired.split('\n')
        
        affected = []
        
        for i, (orig_line, rep_line) in enumerate(zip(original_lines, repaired_lines)):
            if orig_line != rep_line:
                affected.append(i + 1)  # 1-indexed
        
        # Also include added lines
        if len(repaired_lines) > len(original_lines):
            for i in range(len(original_lines), len(repaired_lines)):
                affected.append(i + 1)
        
        return affected
    
    def _generate_llm_repairs(
        self,
        code: str,
        issue_context: IssueContext,
        bug_location: str
    ) -> List[RepairCandidate]:
        """Generate repairs using LLM"""
        
        prompt = f"""Generate 3 code repair suggestions for the following bug:

Issue: {issue_context.description}

Original Code:
```
{code}
```

Location: {bug_location}

Generate repairs that:
1. Directly address the issue
2. Are minimal (small code changes)
3. Are semantically correct

Provide each repair as a complete code snippet."""
        
        try:
            response = self.llm.generate(prompt)
            
            # Parse LLM response for repairs
            repairs = []
            
            # Split by common delimiters
            repair_blocks = re.split(r'(Repair \d|Suggestion \d|-{3,})', response)
            
            for block in repair_blocks:
                if block.strip() and not re.match(r'Repair \d|Suggestion \d', block):
                    # Extract code from block
                    code_match = re.search(r'```(?:python)?\n(.*?)\n```', block, re.DOTALL)
                    if code_match:
                        repair_code = code_match.group(1)
                        
                        repair = RepairCandidate(
                            id=f"repair_llm_{len(repairs)}",
                            original_code=code,
                            repaired_code=repair_code,
                            mutation_type=MutationType.VARIABLE_ASSIGNMENT,
                            affected_lines=[],
                            confidence=0.7,
                            description="LLM-generated repair"
                        )
                        repairs.append(repair)
            
            return repairs
        
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return []
