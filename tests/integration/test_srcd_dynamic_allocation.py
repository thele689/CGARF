"""Integration tests for SRCD section 3.2.1 dynamic resource allocation."""

from pathlib import Path

from src.common.data_structures import IssueContext
from src.common.llm_interface import MockLLMInterface
from src.phase1_causal_analysis.causal_relevance_graph import (
    CausalRelevanceGraph,
    CodeEntity,
    CodeGraph,
    EntityType,
)
from src.srcd.repair_generator import PatchGenerationError, RepairGenerator, SRCDCandidateInput


def _build_test_crg(tmp_path: Path) -> CausalRelevanceGraph:
    repo_file = tmp_path / "example.py"
    repo_file.write_text(
        "def foo(x):\n"
        "    value = x.attr\n"
        "    return value\n\n"
        "def bar(y):\n"
        "    return y + 1\n",
        encoding="utf-8",
    )

    graph = CodeGraph()
    graph.add_entity(
        CodeEntity(
            id=f"{repo_file}::foo:function",
            name="foo",
            entity_type=EntityType.FUNCTION,
            file_path=str(repo_file),
            function_name="foo",
            line_start=1,
            line_end=3,
        )
    )
    graph.add_entity(
        CodeEntity(
            id=f"{repo_file}::bar:function",
            name="bar",
            entity_type=EntityType.FUNCTION,
            file_path=str(repo_file),
            function_name="bar",
            line_start=5,
            line_end=6,
        )
    )
    return CausalRelevanceGraph(code_graph=graph, failure_evidences=[])


def test_srcd_dynamic_budget_and_initial_patch_generation(tmp_path):
    crg = _build_test_crg(tmp_path)
    issue_context = IssueContext(
        id="BUG-ALLOC-1",
        description="AttributeError when accessing x.attr in foo",
        repo_path=str(tmp_path),
        candidates=list(crg.code_graph.entities.keys()),
    )
    cg_mad_result = {
        "path_summaries": [
            {
                "path_id": "foo::path::0",
                "candidate_id": list(crg.code_graph.entities.keys())[0],
                "compressed_text": "Leaf(foo) -> Root(test failure anchor)",
                "evidence_pack": {"anchor": "test_failure"},
            },
            {
                "path_id": "bar::path::0",
                "candidate_id": list(crg.code_graph.entities.keys())[1],
                "compressed_text": "Leaf(bar) -> Root(test failure anchor)",
                "evidence_pack": {"anchor": "test_failure"},
            },
        ],
        "candidate_assessments": [
            {
                "candidate_id": list(crg.code_graph.entities.keys())[0],
                "representative_path_id": "foo::path::0",
                "final_credibility": 0.8,
            },
            {
                "candidate_id": list(crg.code_graph.entities.keys())[1],
                "representative_path_id": "bar::path::0",
                "final_credibility": 0.2,
            },
        ],
    }
    mock_llm = MockLLMInterface(
        responses={
            "search/replace": "<<< SEARCH\n"
            "def foo(x):\n"
            "    value = x.attr\n"
            "    return value\n"
            "===\n"
            "def foo(x):\n"
            "    if x is None:\n"
            "        return None\n"
            "    value = x.attr\n"
            "    return value\n"
            ">>> REPLACE"
        }
    )
    generator = RepairGenerator(llm=mock_llm, max_mutations=6)

    bundle = generator.generate_initial_patches_from_cgmad(
        issue_context=issue_context,
        cg_mad_result=cg_mad_result,
        crg=crg,
        total_sampling_budget=6,
    )

    assert len(bundle.candidate_inputs) == 2
    assert sum(item.allocated_samples for item in bundle.candidate_inputs) == 6
    assert bundle.candidate_inputs[0].allocated_samples > bundle.candidate_inputs[1].allocated_samples
    assert len(bundle.initial_patches) == 2
    assert bundle.initial_patches[0].patch_content.startswith("<<< SEARCH")
    assert ">>> REPLACE" in bundle.initial_patches[0].patch_content


class RetryLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000, **kwargs) -> str:
        self.calls += 1
        if self.calls == 1:
            return """<<< SEARCH
def foo(x):
    value = x.attr
    return value
===
def foo(x):
    try:
        value = x.attr
        return value
    except Exception:
        raise
>>> REPLACE"""
        return """<<< SEARCH
def foo(x):
    value = x.attr
    return value
===
def foo(x):
    if x is None:
        return None
    value = x.attr
    return value
>>> REPLACE"""


def test_initial_patch_retries_after_broad_wrapper(tmp_path):
    crg = _build_test_crg(tmp_path)
    candidate_id = list(crg.code_graph.entities.keys())[0]
    issue_context = IssueContext(
        id="BUG-ALLOC-2",
        description="AttributeError when accessing x.attr in foo",
        repo_path=str(tmp_path),
        candidates=[candidate_id],
    )
    cg_mad_result = {
        "path_summaries": [
            {
                "path_id": "foo::path::0",
                "candidate_id": candidate_id,
                "compressed_text": "Leaf(foo) -> Root(test failure anchor)",
                "evidence_pack": {"anchor": "test_failure"},
            }
        ],
        "candidate_assessments": [
            {
                "candidate_id": candidate_id,
                "representative_path_id": "foo::path::0",
                "final_credibility": 1.0,
            }
        ],
    }

    generator = RepairGenerator(llm=RetryLLM(), max_mutations=2)
    bundle = generator.generate_initial_patches_from_cgmad(
        issue_context=issue_context,
        cg_mad_result=cg_mad_result,
        crg=crg,
        total_sampling_budget=1,
    )

    patch_text = bundle.initial_patches[0].patch_content
    assert "except Exception" not in patch_text
    assert "if x is None" in patch_text


class NoOpRetryLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000, **kwargs) -> str:
        self.calls += 1
        if self.calls < 3:
            return """<<< SEARCH
def foo(x):
    value = x.attr
    return value
===
def foo(x):
    value = x.attr
    return value
>>> REPLACE"""
        return """<<< SEARCH
def foo(x):
    value = x.attr
    return value
===
def foo(x):
    if x is None:
        return None
    value = x.attr
    return value
>>> REPLACE"""


def test_initial_patch_retries_after_no_op_patch(tmp_path):
    crg = _build_test_crg(tmp_path)
    candidate_id = list(crg.code_graph.entities.keys())[0]
    issue_context = IssueContext(
        id="BUG-ALLOC-3",
        description="AttributeError when accessing x.attr in foo",
        repo_path=str(tmp_path),
        candidates=[candidate_id],
    )
    cg_mad_result = {
        "path_summaries": [
            {
                "path_id": "foo::path::0",
                "candidate_id": candidate_id,
                "compressed_text": "Leaf(foo) -> Root(test failure anchor)",
                "evidence_pack": {"anchor": "test_failure"},
            }
        ],
        "candidate_assessments": [
            {
                "candidate_id": candidate_id,
                "representative_path_id": "foo::path::0",
                "final_credibility": 1.0,
            }
        ],
    }

    generator = RepairGenerator(llm=NoOpRetryLLM(), max_mutations=2)
    bundle = generator.generate_initial_patches_from_cgmad(
        issue_context=issue_context,
        cg_mad_result=cg_mad_result,
        crg=crg,
        total_sampling_budget=1,
    )

    patch_text = bundle.initial_patches[0].patch_content
    assert "if x is None" in patch_text


class AlwaysNoOpLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000, **kwargs) -> str:
        self.calls += 1
        return """<<< SEARCH
def foo(x):
    return x + 1
===
def foo(x):
    return x + 1
>>> REPLACE"""


def test_initial_patch_skips_candidate_when_only_no_op_is_available(tmp_path):
    crg = _build_test_crg(tmp_path)
    candidate_id = list(crg.code_graph.entities.keys())[1]
    issue_context = IssueContext(
        id="BUG-ALLOC-4",
        description="Nested compound model separability issue",
        repo_path=str(tmp_path),
        candidates=[candidate_id],
    )
    cg_mad_result = {
        "path_summaries": [
            {
                "path_id": "bar::path::0",
                "candidate_id": candidate_id,
                "compressed_text": "Leaf(bar) -> Root(test failure anchor)",
                "evidence_pack": {"anchor": "test_failure"},
            }
        ],
        "candidate_assessments": [
            {
                "candidate_id": candidate_id,
                "representative_path_id": "bar::path::0",
                "final_credibility": 1.0,
            }
        ],
    }

    generator = RepairGenerator(llm=AlwaysNoOpLLM(), max_mutations=2)
    bundle = generator.generate_initial_patches_from_cgmad(
        issue_context=issue_context,
        cg_mad_result=cg_mad_result,
        crg=crg,
        total_sampling_budget=1,
    )

    assert bundle.initial_patches == []
    assert bundle.candidate_inputs == []


def test_fallback_never_returns_original_to_original_patch():
    generator = RepairGenerator(llm=None, max_mutations=2)
    issue_context = IssueContext(
        id="BUG-NOOP-FALLBACK",
        description="Nested compound model separability issue",
        repo_path="",
        candidates=["candidate"],
    )
    candidate = SRCDCandidateInput(
        candidate_id="candidate",
        candidate_location="candidate",
        file_path="example.py",
        entity_type="function",
        code_context="def unchanged(x):\n    return x + 1",
        representative_path_id="path",
        representative_path_summary="Leaf -> Root",
    )

    patch = generator._build_fallback_search_replace(issue_context, candidate)
    assert patch is None


def test_python_search_replace_validation_allows_indented_local_snippet():
    generator = RepairGenerator(llm=MockLLMInterface(), max_mutations=2)
    patch = """<<< SEARCH
    value = x.attr
===
    if x is None:
        return None
    value = x.attr
>>> REPLACE"""

    assert generator._is_python_search_replace_patch_valid(patch)
