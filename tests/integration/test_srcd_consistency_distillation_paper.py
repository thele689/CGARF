"""Paper-aligned tests for SRCD section 3.2.3 consistency distillation."""

from src.srcd.consistency_distiller import ConsistencyDistiller, SiliconFlowEmbeddingBackend


class DummyConsensusLLM:
    def __init__(self):
        self.call_count = 0
        self.model_name = "dummy-consensus-llm"

    def generate(self, prompt, temperature=0.2, max_tokens=1000, **kwargs):
        self.call_count += 1
        assert "Candidate patch set" in prompt
        return """{
  "shared_edit_intent": "Add a local guard around the failing branch.",
  "shared_target_entities": ["parse_json", "result"],
  "shared_mechanism": "Tighten the local condition before using the parsed result.",
  "shared_constraints": ["local edit", "keep interface unchanged"]
}"""


class DummyEmbeddingBackend:
    def __init__(self):
        self.model_name = "dummy-embedding"

    def encode_texts(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vec = [
                1.0 if "guard" in lower or "if " in lower else 0.2,
                1.0 if "return" in lower else 0.1,
                1.0 if "result" in lower else 0.1,
            ]
            vectors.append(vec)
        return __import__("numpy").array(vectors, dtype=float)

    def cosine_similarity(self, left, right):
        import numpy as np

        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        if denom == 0:
            return 0.0
        cosine = float(np.dot(left, right) / denom)
        return max(0.0, min(1.0, cosine))


def _round(round_id, location, reflection_score, patch_suffix):
    return {
        "round": round_id,
        "generation_temperature": 0.2 + 0.1 * (round_id - 1),
        "patch": {
            "patch_id": f"{location}::round::{round_id}",
            "location": location,
            "patch_content": (
                "<<< SEARCH\n"
                "result = parse(data)\n"
                "===\n"
                f"if data is None:\n    return None\nresult = parse(data)\n# {patch_suffix}\n"
                ">>> REPLACE"
            ),
            "generated_round": round_id,
        },
        "reflection": {
            "combined_reflection": reflection_score,
            "semantic_consistency": {"level": "fully_yes", "reason": "local"},
            "causal_alignment": {"level": "partially_yes", "reason": "aligned"},
            "minimal_edit": {"level": "fully_yes", "reason": "minimal"},
            "revision_suggestion": "Keep the guard local.",
        },
    }


def test_paper_distillation_groups_by_candidate_and_keeps_top_k():
    candidate_a = "pkg/parser.py::parse_json:function"
    candidate_b = "pkg/parser.py::parse_yaml:function"
    reflection_payload = {
        "candidate_runs": {
            candidate_a: {
                "candidate_location": candidate_a,
                "allocated_samples": 6,
                "rounds": [
                    _round(1, candidate_a, 0.70, "a1"),
                    _round(2, candidate_a, 0.75, "a2"),
                    _round(3, candidate_a, 0.65, "a3"),
                    _round(4, candidate_a, 0.60, "a4"),
                    _round(5, candidate_a, 0.55, "a5"),
                    _round(6, candidate_a, 0.50, "a6"),
                ],
            },
            candidate_b: {
                "candidate_location": candidate_b,
                "allocated_samples": 2,
                "rounds": [
                    _round(1, candidate_b, 0.40, "b1"),
                    _round(2, candidate_b, 0.45, "b2"),
                ],
            },
        }
    }

    distiller = ConsistencyDistiller(
        llm=DummyConsensusLLM(),
        embedding_backend=DummyEmbeddingBackend(),
    )
    result = distiller.distill_reflection_payload(
        reflection_payload=reflection_payload,
        top_k_per_candidate=5,
    )

    assert result["llm_model"] == "dummy-consensus-llm"
    assert result["embedding_model"] == "dummy-embedding"
    assert set(result["candidate_results"].keys()) == {candidate_a, candidate_b}

    candidate_a_result = result["candidate_results"][candidate_a]
    assert candidate_a_result["consensus_pattern"]["shared_edit_intent"] != ""
    assert len(candidate_a_result["ranked_patches"]) == 6
    assert len(candidate_a_result["kept_patch_ids"]) == 5

    ranked_scores = [item["distillation_score"] for item in candidate_a_result["ranked_patches"]]
    assert ranked_scores == sorted(ranked_scores, reverse=True)

    candidate_b_result = result["candidate_results"][candidate_b]
    assert len(candidate_b_result["kept_patch_ids"]) == 2
    for patch in candidate_b_result["ranked_patches"]:
        assert 0.0 <= patch["consistency_score"] <= 1.0
        assert 0.0 <= patch["embedding_similarity"] <= 1.0


class DummyEmbeddingResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummyEmbeddingSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        inputs = json["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        return DummyEmbeddingResponse(
            {
                "data": [
                    {"index": idx, "embedding": [float(idx + 1), 1.0, 0.0]}
                    for idx, _ in enumerate(inputs)
                ]
            }
        )


def test_siliconflow_embedding_backend_uses_embeddings_endpoint():
    session = DummyEmbeddingSession()
    backend = SiliconFlowEmbeddingBackend(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        api_key="test-key",
        api_base="https://api.siliconflow.cn/v1",
        session=session,
        timeout_seconds=7,
    )

    vectors = backend.encode_texts(["patch a", "patch b"])

    assert vectors.shape == (2, 3)
    assert session.calls[0]["url"] == "https://api.siliconflow.cn/v1/embeddings"
    assert session.calls[0]["json"]["model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert session.calls[0]["json"]["input"] == ["patch a", "patch b"]
    assert session.calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert backend.effective_model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert backend.mode == "siliconflow_api"
