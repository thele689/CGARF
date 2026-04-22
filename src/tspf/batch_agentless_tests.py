"""Batch runner for Agentless-style TSPF test evidence.

This entry point generates regression/reproduction evidence for CGARF's
Search/Replace patches and can immediately rerun TSPF to select the final
patch from the valid candidate set.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from src.common.llm_interface import create_llm_interface
from src.tspf.agentless_adapted_tests import (
    LocalAgentlessTestEvidenceRunner,
    run_tspf_with_evidence,
    write_evidence_json,
)


def _read_issue_text(args: argparse.Namespace) -> str:
    if args.issue_text:
        return args.issue_text
    if args.issue_text_file:
        return Path(args.issue_text_file).read_text()
    raise ValueError("Provide --issue-text or --issue-text-file")


def _create_optional_llm(args: argparse.Namespace):
    if not args.use_llm_reproduction:
        return None
    return create_llm_interface(
        provider=args.llm_provider,
        model_name=args.llm_model,
        api_key=args.llm_api_key,
        api_base=args.llm_api_base,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Agentless-style regression/reproduction evidence for TSPF."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--distillation-json", type=Path, required=True)
    parser.add_argument("--issue-text", default=None)
    parser.add_argument("--issue-text-file", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-patches", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--run-tspf", action="store_true")
    parser.add_argument("--mu", type=float, default=0.6)
    parser.add_argument("--tspf-output-json", type=Path, default=None)
    parser.add_argument("--use-llm-reproduction", action="store_true")
    parser.add_argument("--llm-provider", default="qwen")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-api-base", default="https://api.siliconflow.cn/v1")
    args = parser.parse_args()

    issue_text = _read_issue_text(args)
    llm = _create_optional_llm(args)
    distillation_payload = json.loads(args.distillation_json.read_text())

    runner = LocalAgentlessTestEvidenceRunner(
        instance_id=args.instance_id,
        repo_root=args.repo_root,
        issue_text=issue_text,
        llm=llm,
        python_executable=args.python_executable or "python",
        timeout=args.timeout,
    )
    result = runner.build_from_distillation(
        distillation_payload,
        max_patches=args.max_patches,
    )

    output_json = args.output_json or (
        args.workspace_root / "data" / "tspf" / f"{args.instance_id}_test_evidence.json"
    )
    write_evidence_json(result, output_json)
    logger.success("TSPF test evidence saved to {}", output_json)

    if args.run_tspf:
        tspf_result = run_tspf_with_evidence(
            distillation_payload=distillation_payload,
            evidence_payload=result.to_dict(),
            mu=args.mu,
            max_patches=args.max_patches,
        )
        tspf_result.update(
            {
                "instance_id": args.instance_id,
                "source_distillation_json": str(args.distillation_json),
                "source_test_evidence_json": str(output_json),
            }
        )
        tspf_output = args.tspf_output_json or (
            args.workspace_root / "data" / "tspf" / f"{args.instance_id}_tspf.json"
        )
        tspf_output.parent.mkdir(parents=True, exist_ok=True)
        tspf_output.write_text(json.dumps(tspf_result, indent=2, ensure_ascii=False))
        logger.success(
            "TSPF result saved to {} with selection_status={} selected_patch_id={}",
            tspf_output,
            tspf_result.get("selection_status"),
            tspf_result.get("selected_patch_id"),
        )


if __name__ == "__main__":
    main()
