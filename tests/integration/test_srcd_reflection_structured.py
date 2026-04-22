"""Paper-aligned tests for SRCD section 3.2.2 structured reflection."""

from src.common.data_structures import IssueContext, PatchCandidate
from src.srcd.repair_generator import RepairGenerator, SRCDCandidateInput
from src.srcd.reflection_scorer import ReflectionScorer


def _candidate_input() -> SRCDCandidateInput:
    return SRCDCandidateInput(
        candidate_id="parser.py::parse_json",
        candidate_location="parser.py::parse_json",
        file_path="parser.py",
        entity_type="function",
        code_context=(
            "def parse_json(data):\n"
            "    result = json.loads(data)\n"
            "    return result['key']\n"
        ),
        representative_path_id="path-1",
        representative_path_summary=(
            "Leaf(parse_json) -> json.loads[c=0.75] -> Root(anchor=JSON parsing failure)"
        ),
        representative_path_evidence={"anchor": "JSON parsing failure"},
        candidate_credibility=0.8,
        normalized_weight=0.5,
        allocated_samples=2,
    )


def _patch() -> PatchCandidate:
    return PatchCandidate(
        patch_id="patch-1",
        location="parser.py::parse_json",
        patch_content=(
            "<<< SEARCH\n"
            "result = json.loads(data)\n"
            "===\n"
            "if data is None:\n"
            "    return {}\n"
            "result = json.loads(data)\n"
            ">>> REPLACE"
        ),
        generated_round=1,
        credibility_from_location=0.6,
    )


class DummyStructuredReflectionLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        assert "semantic_consistency" in prompt
        assert "causal_alignment" in prompt
        assert "minimal_edit" in prompt
        return """{
  "semantic_consistency": {
    "level": "fully_yes",
    "reason": "The patch preserves the function contract and only guards the failing input."
  },
  "causal_alignment": {
    "level": "partially_yes",
    "reason": "The change directly targets the parse_json -> json.loads failure mechanism."
  },
  "minimal_edit": {
    "level": "partially_yes",
    "reason": "The patch only adds a narrow guard without restructuring the function."
  },
  "revision_suggestion": "Keep the guard local and avoid broader behavioral changes."
}"""


def test_score_patch_candidate_uses_five_level_structured_output():
    scorer = ReflectionScorer(DummyStructuredReflectionLLM())

    score = scorer.score_patch_candidate(
        patch=_patch(),
        candidate_input=_candidate_input(),
        issue_text="NullPointerException when parsing JSON data",
        current_temperature=0.4,
    )

    assert score.source == "llm_structured"
    assert score.semantic_consistency is not None
    assert score.semantic_consistency.level == "fully_yes"
    assert score.semantic_consistency.score == 1.0
    assert score.causal_alignment is not None
    assert score.causal_alignment.level == "partially_yes"
    assert score.causal_alignment.score == 0.75
    assert score.minimal_edit is not None
    assert score.minimal_edit.level == "partially_yes"
    assert score.minimal_edit.score == 0.75
    assert score.revision_suggestion != ""
    assert 0.0 <= score.combined_reflection <= 1.0
    assert score.suggested_temperature == 0.30000000000000004


def test_score_patch_bundle_preserves_candidate_mapping():
    scorer = ReflectionScorer(DummyStructuredReflectionLLM())
    candidate_input = _candidate_input()
    patch = _patch()

    scores = scorer.score_patch_bundle(
        issue_text="NullPointerException when parsing JSON data",
        candidate_inputs=[candidate_input],
        patches=[patch],
        current_temperature=0.2,
    )

    assert set(scores.keys()) == {"patch-1"}
    assert scores["patch-1"].repair_id == "patch-1"
    assert scores["patch-1"].source == "llm_structured"


class DummyRefinementLLM:
    def __init__(self):
        self.temperatures = []
        self.prompts = []

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000, **kwargs) -> str:
        self.temperatures.append(temperature)
        self.prompts.append(prompt)
        assert "Revision suggestion" in prompt
        return """<<< SEARCH
result = json.loads(data)
===
if data is None:
    return {}
result = json.loads(data)
>>> REPLACE"""


def test_generate_refined_patch_uses_revision_suggestion_and_temperature():
    llm = DummyRefinementLLM()
    generator = RepairGenerator(llm=llm)
    issue_context = IssueContext(
        id="BUG-001",
        description="NullPointerException when parsing JSON data",
        repo_path="/tmp/repo",
        candidates=["parser.py::parse_json"],
        test_framework="pytest",
    )
    reflection_payload = {
        "semantic_consistency": {"level": "neutral", "reason": "The patch is local but still broad."},
        "causal_alignment": {"level": "partially_not", "reason": "The patch does not target null input handling directly."},
        "minimal_edit": {"level": "partially_not", "reason": "The scope is broader than necessary."},
        "revision_suggestion": "Replace the broad wrapper with a local null-input guard.",
    }

    refined_patch = generator.generate_refined_patch(
        issue_context=issue_context,
        candidate_input=_candidate_input(),
        reflection_payload=reflection_payload,
        generation_temperature=0.35,
        patch_index=2,
    )

    assert refined_patch.generated_round == 2
    assert refined_patch.location == "parser.py::parse_json"
    assert "if data is None" in refined_patch.patch_content
    assert llm.temperatures == [0.35]
    assert "Treat the previous patch as rejected feedback context" in llm.prompts[0]
    assert "Concrete local edit scaffold for this candidate" in llm.prompts[0]
    assert _patch().patch_content not in llm.prompts[0]


class DummyRetryingRefinementLLM:
    def __init__(self):
        self.calls = 0
        self.temperatures = []

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000, **kwargs) -> str:
        self.calls += 1
        self.temperatures.append(temperature)
        if self.calls < 3:
            return """<<< SEARCH
result = json.loads(data)
===
result = json.loads(data)
>>> REPLACE"""
        return """<<< SEARCH
result = json.loads(data)
===
result = json.loads(data)
if result is None:
    return {}
>>> REPLACE"""


def test_generate_refined_patch_uses_forced_retry_before_fallback():
    llm = DummyRetryingRefinementLLM()
    generator = RepairGenerator(llm=llm)
    issue_context = IssueContext(
        id="BUG-001",
        description="NullPointerException when parsing JSON data",
        repo_path="/tmp/repo",
        candidates=["parser.py::parse_json"],
        test_framework="pytest",
    )
    reflection_payload = {
        "semantic_consistency": {"level": "fully_yes", "reason": "Keep the function contract unchanged."},
        "causal_alignment": {"level": "fully_not", "reason": "The previous patch did not change the failing logic."},
        "minimal_edit": {"level": "fully_yes", "reason": "A small local guard is enough."},
        "revision_suggestion": "Add a local guard around the failing JSON parse result handling.",
    }

    refined_patch = generator.generate_refined_patch(
        issue_context=issue_context,
        candidate_input=_candidate_input(),
        reflection_payload=reflection_payload,
        generation_temperature=0.25,
        patch_index=2,
    )

    assert llm.calls == 3
    assert llm.temperatures == [0.25, 0.35, 0.55]
    assert "if result is None" in refined_patch.patch_content
