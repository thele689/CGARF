import json
from pathlib import Path

from src.tspf.official_swebench_harness import (
    OfficialSWEbenchHarnessAdapter,
    SearchReplaceToUnifiedDiffConverter,
)


def test_search_replace_to_unified_diff_converter(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "pkg.py"
    source.write_text("def f():\n    return 1\n")

    patch = "<<< SEARCH\ndef f():\n    return 1\n===\ndef f():\n    return 2\n>>> REPLACE"
    diff, status, reason = SearchReplaceToUnifiedDiffConverter().convert(
        repo,
        patch,
        str(source) + "::f::function",
    )

    assert status == "converted"
    assert "pkg.py" in reason
    assert "--- a/pkg.py" in diff
    assert "+++ b/pkg.py" in diff
    assert "-    return 1" in diff
    assert "+    return 2" in diff


def test_prepare_predictions_from_distillation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "pkg.py"
    source.write_text("def f():\n    return 1\n")

    payload = {
        "candidate_results": {
            "loc": {
                "candidate_location": str(source) + "::f::function",
                "ranked_patches": [
                    {
                        "patch_id": "p1",
                        "patch_content": "<<< SEARCH\ndef f():\n    return 1\n===\ndef f():\n    return 2\n>>> REPLACE",
                        "distillation_score": 0.9,
                    }
                ],
            }
        }
    }

    adapter = OfficialSWEbenchHarnessAdapter(
        workspace_root=tmp_path,
        repo_root=repo,
        swebench_root=tmp_path / "SWE-bench",
        python_executable="python3.10",
    )
    output_jsonl = tmp_path / "preds.jsonl"
    converted = adapter.prepare_predictions_from_distillation(
        "toy__1",
        payload,
        output_jsonl,
    )

    assert len(converted) == 1
    assert converted[0].conversion_status == "converted"
    lines = output_jsonl.read_text().splitlines()
    assert len(lines) == 1
    prediction = json.loads(lines[0])
    assert prediction["instance_id"] == "toy__1"
    assert prediction["model_patch"].startswith("--- a/pkg.py")


def test_build_tspf_evidence_from_official_reports(tmp_path):
    adapter = OfficialSWEbenchHarnessAdapter(
        workspace_root=tmp_path,
        repo_root=tmp_path / "repo",
        swebench_root=tmp_path / "SWE-bench",
    )
    candidate = adapter.prepare_predictions_from_distillation
    del candidate

    from src.tspf.official_swebench_harness import UnifiedPatchCandidate

    converted = [
        UnifiedPatchCandidate(
            patch_id="p1",
            instance_id="toy__1",
            model_name_or_path="CGARF_1",
            model_patch="--- a/x.py\n+++ b/x.py\n",
            candidate_id="loc",
            candidate_location="x.py::f",
            source_patch_content="patch",
            conversion_status="converted",
        )
    ]
    report_dir = tmp_path / "logs/run_evaluation/run/CGARF_1/toy__1"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "toy__1": {
                    "resolved": True,
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": ["test_repro"], "failure": [], "error": []},
                        "PASS_TO_PASS": {"success": ["test_reg"], "failure": [], "error": []},
                    },
                }
            }
        )
    )

    evidence = adapter.build_tspf_evidence_from_reports(converted, run_id="run")

    assert evidence["p1"]["reproduction"]["pass_rate"] == 1.0
    assert evidence["p1"]["regression"]["pass_rate"] == 1.0
    assert evidence["p1"]["official_harness"]["resolved"] is True
