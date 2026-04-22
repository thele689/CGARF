"""Batch entry point for the official SWE-bench Docker harness.

This runner prepares official SWE-bench predictions from CGARF distilled
Search/Replace patches, optionally invokes Docker evaluation, and converts the
resulting reports into TSPF evidence/final selection.
"""

import argparse
import json
from pathlib import Path
import re

from loguru import logger

from src.tspf.official_swebench_harness import (
    OfficialSWEbenchHarnessAdapter,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CGARF patches with official SWE-bench harness.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--distillation-json", type=Path, required=True)
    parser.add_argument("--swebench-root", type=Path, default=Path("SWE-bench"))
    parser.add_argument("--python-executable", default="python3.10")
    parser.add_argument("--dataset-name", default="SWE-bench/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-patches", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--mu", type=float, default=0.6)
    parser.add_argument("--namespace", default="none")
    parser.add_argument("--cache-level", default="instance", choices=["none", "base", "env", "instance"])
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--use-local-repo-bundle",
        action="store_true",
        help=(
            "Build a local git bundle from --repo-root and use it as the SWE-bench "
            "instance image clone source. This keeps the official harness but avoids "
            "Docker build-time GitHub access."
        ),
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-harness", action="store_true")
    parser.add_argument("--stop-on-harness-error", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or f"cgarf_official_{args.instance_id}"
    output_dir = args.workspace_root / "data" / "tspf_official"
    predictions_jsonl = output_dir / f"{args.instance_id}_predictions.jsonl"
    conversion_json = output_dir / f"{args.instance_id}_converted_candidates.json"
    evidence_json = output_dir / f"{args.instance_id}_official_test_evidence.json"
    tspf_json = output_dir / f"{args.instance_id}_official_tspf.json"
    harness_stdout = output_dir / f"{args.instance_id}_harness_stdout.txt"
    harness_stderr = output_dir / f"{args.instance_id}_harness_stderr.txt"
    local_repo_bundle = output_dir / f"{args.instance_id}_local_repo.bundle"

    adapter = OfficialSWEbenchHarnessAdapter(
        workspace_root=args.workspace_root,
        repo_root=args.repo_root,
        swebench_root=args.swebench_root,
        python_executable=args.python_executable,
        dataset_name=args.dataset_name,
        split=args.split,
    )

    dependency_status = adapter.check_dependencies()
    write_json(output_dir / f"{args.instance_id}_dependency_status.json", dependency_status.to_dict())
    if args.check_only:
        logger.info("Dependency status: {}", dependency_status.to_dict())
        return
    if not dependency_status.ok and not args.prepare_only:
        raise RuntimeError(
            "Official SWE-bench harness dependencies are not ready. "
            f"Status: {dependency_status.to_dict()}"
        )

    distillation_payload = json.loads(args.distillation_json.read_text())
    converted = adapter.prepare_predictions_from_distillation(
        instance_id=args.instance_id,
        distillation_payload=distillation_payload,
        output_jsonl=predictions_jsonl,
        max_patches=args.max_patches,
    )
    write_json(conversion_json, [item.to_dict() for item in converted])
    logger.success("Prepared {} converted candidates at {}", len(converted), conversion_json)
    logger.success("Prepared official predictions at {}", predictions_jsonl)

    if args.prepare_only:
        return

    if not args.skip_harness:
        local_repo_bundle_path = None
        if args.use_local_repo_bundle:
            local_repo_bundle_path = adapter.create_local_repo_bundle(local_repo_bundle)
            logger.success("Prepared local repository bundle at {}", local_repo_bundle_path)
        per_patch_dir = output_dir / f"{args.instance_id}_per_patch_predictions"
        per_patch_dir.mkdir(parents=True, exist_ok=True)
        stdout_parts = []
        stderr_parts = []
        for index, candidate in enumerate(converted, start=1):
            if candidate.conversion_status != "converted":
                logger.warning(
                    "Skipping harness for {} because conversion_status={}",
                    candidate.patch_id,
                    candidate.conversion_status,
                )
                continue
            safe_patch_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate.patch_id)[-160:]
            candidate_predictions = per_patch_dir / f"{index}_{safe_patch_id}.jsonl"
            candidate_predictions.write_text(
                json.dumps(candidate.prediction_dict(), ensure_ascii=False) + "\n"
            )
            proc = adapter.run_harness(
                predictions_jsonl=candidate_predictions,
                run_id=run_id,
                instance_ids=[args.instance_id],
                max_workers=args.max_workers,
                timeout=args.timeout,
                namespace=None if args.namespace == "none" else args.namespace,
                cache_level=args.cache_level,
                clean=args.clean,
                force_rebuild=args.force_rebuild,
                report_dir=output_dir,
                local_repo_bundle=local_repo_bundle_path,
            )
            stdout_parts.append(
                f"\n\n===== {candidate.patch_id} returncode={proc.returncode} =====\n{proc.stdout or ''}"
            )
            stderr_parts.append(
                f"\n\n===== {candidate.patch_id} returncode={proc.returncode} =====\n{proc.stderr or ''}"
            )
            logger.info(
                "Harness candidate {} return code: {}",
                candidate.patch_id,
                proc.returncode,
            )
            if proc.returncode != 0 and args.stop_on_harness_error:
                raise RuntimeError(
                    f"Official SWE-bench harness failed for {candidate.patch_id} "
                    f"with return code {proc.returncode}."
                )
        harness_stdout.write_text("".join(stdout_parts))
        harness_stderr.write_text("".join(stderr_parts))

    evidence = adapter.build_tspf_evidence_from_reports(converted, run_id=run_id)
    write_json(evidence_json, {"instance_id": args.instance_id, "run_id": run_id, "evidence": evidence})
    tspf_result = adapter.select_final_patch(
        distillation_payload=distillation_payload,
        evidence=evidence,
        mu=args.mu,
        max_patches=args.max_patches,
    )
    tspf_result.update(
        {
            "instance_id": args.instance_id,
            "source_distillation_json": str(args.distillation_json),
            "source_test_evidence_json": str(evidence_json),
            "official_swebench_run_id": run_id,
        }
    )
    write_json(tspf_json, tspf_result)
    logger.success(
        "Official TSPF result saved to {} with selection_status={} selected_patch_id={}",
        tspf_json,
        tspf_result.get("selection_status"),
        tspf_result.get("selected_patch_id"),
    )


if __name__ == "__main__":
    main()
