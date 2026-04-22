import json
from pathlib import Path

from src.tspf.agentless_adapted_tests import (
    LocalAgentlessTestEvidenceRunner,
    run_tspf_with_evidence,
)


def test_agentless_style_evidence_selects_final_patch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text(
        "def answer():\n"
        "    return 1\n"
    )

    reproduction_script = """
from module import answer
if answer() == 2:
    print("Issue resolved")
else:
    print("Issue reproduced")
""".strip()
    regression_script = """
from module import answer
assert answer() in {1, 2}
print("Regression passed")
""".strip()

    payload = {
        "candidate_results": {
            "loc": {
                "candidate_location": str(repo / "module.py") + "::answer::function",
                "ranked_patches": [
                    {
                        "patch_id": "p_good",
                        "patch_content": "<<< SEARCH\ndef answer():\n    return 1\n===\ndef answer():\n    return 2\n>>> REPLACE",
                        "causality_score": 0.8,
                        "embedding_text": "return 2",
                    },
                    {
                        "patch_id": "p_bad",
                        "patch_content": "<<< SEARCH\ndef answer():\n    return 1\n===\ndef answer():\n    return 3\n>>> REPLACE",
                        "causality_score": 1.0,
                        "embedding_text": "return 3",
                    },
                ],
            }
        }
    }

    runner = LocalAgentlessTestEvidenceRunner(
        instance_id="toy__1",
        repo_root=repo,
        issue_text="answer should return 2",
        timeout=30,
    )
    evidence_result = runner.build_from_distillation(
        payload,
        reproduction_script=reproduction_script,
        regression_scripts=[regression_script],
    )

    assert evidence_result.evidence["p_good"]["reproduction"]["passed_tests"] == 1
    assert evidence_result.evidence["p_good"]["regression"]["passed_tests"] == 1
    assert evidence_result.evidence["p_bad"]["reproduction"]["passed_tests"] == 0

    tspf_result = run_tspf_with_evidence(
        payload,
        evidence_result.to_dict(),
        mu=0.6,
        max_patches=5,
    )

    assert tspf_result["selection_status"] == "selected"
    assert tspf_result["selected_patch_id"] == "p_good"
    assert tspf_result["selected_patch"]["functional_result"]["passed"] is True


def test_agentless_style_evidence_json_shape(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("value = 1\n")

    payload = {
        "candidate_results": {
            "loc": {
                "candidate_location": str(repo / "module.py") + "::value::variable",
                "ranked_patches": [
                    {
                        "patch_id": "p1",
                        "patch_content": "<<< SEARCH\nvalue = 1\n===\nvalue = 2\n>>> REPLACE",
                        "causality_score": 1.0,
                    }
                ],
            }
        }
    }
    runner = LocalAgentlessTestEvidenceRunner(
        instance_id="toy__2",
        repo_root=repo,
        issue_text="value should be 2",
        timeout=30,
    )
    result = runner.build_from_distillation(
        payload,
        reproduction_script="print('Issue reproduced')",
        regression_scripts=["print('Regression passed')"],
    )
    data = result.to_dict()

    assert data["instance_id"] == "toy__2"
    assert "evidence" in data
    assert data["evidence"]["p1"]["patch_apply"]["applied"] is True
    json.dumps(data)
