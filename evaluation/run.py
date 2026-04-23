"""Unified CGARF runner.

This script provides a compact public entry point over the paper-aligned batch
runners.  It intentionally delegates each stage to the implementation modules
under ``src/`` instead of reimplementing the pipeline logic here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.runtime_config import MODEL_PROFILES, create_configured_llm


STAGE_ORDER = {
    "crg": 0,
    "trace_analysis": 1,
    "srcd_initial": 2,
    "reflection": 3,
    "distillation": 4,
    "search": 4,
    "tspf": 5,
}

STAGE_ALIASES = {
    "trace": "trace_analysis",
    "cg_mad": "trace_analysis",
    "cg-mad": "trace_analysis",
    "srcd": "search",
}


@dataclass
class InstanceRunSummary:
    instance_id: str
    final_stage: str
    outputs: Dict[str, Optional[str]] = field(default_factory=dict)
    status: str = "completed"
    error: Optional[str] = None


def normalize_stage(stage: str) -> str:
    normalized = STAGE_ALIASES.get(stage, stage)
    if normalized not in STAGE_ORDER:
        known = ", ".join(sorted(STAGE_ORDER))
        raise ValueError(f"Unknown final_stage: {stage}. Known stages: {known}")
    return normalized


def includes_stage(final_stage: str, stage: str) -> bool:
    return STAGE_ORDER[normalize_stage(final_stage)] >= STAGE_ORDER[normalize_stage(stage)]


def default_distillation_path(workspace_root: Path, instance_id: str) -> Path:
    return workspace_root / "data" / "srcd_distillation" / f"{instance_id}_srcd_distillation.json"


def default_tspf_evidence_path(workspace_root: Path, instance_id: str) -> Optional[Path]:
    path = workspace_root / "data" / "tspf_official" / f"{instance_id}_official_test_evidence.json"
    return path if path.exists() else None


def run_instance(args, llm, instance_id: str) -> InstanceRunSummary:
    final_stage = normalize_stage(args.final_stage)
    summary = InstanceRunSummary(instance_id=instance_id, final_stage=final_stage)

    if final_stage == "crg":
        from src.phase1_causal_analysis.batch_crg_constructor import BatchCRGConstructor

        constructor = BatchCRGConstructor(
            workspace_root=str(args.workspace_root),
            llm_client=llm,
            fl_method=args.method,
            shared_workspace_root=str(args.shared_workspace_root) if args.shared_workspace_root else None,
            repo_cache_root=str(args.repo_cache_root) if args.repo_cache_root else None,
            max_paths_per_candidate=args.max_paths_per_candidate,
        )
        summary.outputs["crg"] = constructor.run_single_instance(
            instance_id,
            method=args.method,
            max_paths_per_candidate=args.max_paths_per_candidate,
        )
        return summary

    cg_mad_path: Optional[str] = None
    if includes_stage(final_stage, "trace_analysis"):
        from src.phase1_causal_analysis.batch_cg_mad import BatchCGMADRunner

        cg_mad_runner = BatchCGMADRunner(
            workspace_root=str(args.workspace_root),
            llm_client=llm,
            fl_method=args.method,
            shared_workspace_root=str(args.shared_workspace_root) if args.shared_workspace_root else None,
            repo_cache_root=str(args.repo_cache_root) if args.repo_cache_root else None,
            max_paths_per_candidate=args.max_paths_per_candidate,
        )
        cg_mad_path = cg_mad_runner.run_single_instance(
            instance_id=instance_id,
            method=args.method,
            max_paths_per_candidate=args.max_paths_per_candidate,
        )
        summary.outputs["trace_analysis"] = cg_mad_path
        if final_stage == "trace_analysis":
            return summary

    srcd_path: Optional[str] = None
    if includes_stage(final_stage, "srcd_initial"):
        from src.srcd.batch_srcd import BatchSRCDRunner

        srcd_runner = BatchSRCDRunner(
            workspace_root=str(args.workspace_root),
            llm_client=llm,
            fl_method=args.method,
            shared_workspace_root=str(args.shared_workspace_root) if args.shared_workspace_root else None,
            repo_cache_root=str(args.repo_cache_root) if args.repo_cache_root else None,
        )
        srcd_path = srcd_runner.process_instance(
            instance_id=instance_id,
            total_sampling_budget=args.total_sampling_budget,
            max_candidates=args.max_candidates,
            max_paths_per_candidate=args.max_paths_per_candidate,
            cg_mad_json=cg_mad_path,
        )
        summary.outputs["srcd_initial"] = srcd_path
        if final_stage == "srcd_initial":
            return summary

    reflection_path: Optional[str] = None
    if includes_stage(final_stage, "reflection"):
        from src.srcd.batch_reflection import BatchSRCDReflectionRunner

        reflection_runner = BatchSRCDReflectionRunner(
            workspace_root=str(args.workspace_root),
            llm_client=llm,
            fl_method=args.method,
        )
        reflection_path = reflection_runner.process_instance(
            instance_id=instance_id,
            srcd_json=srcd_path,
            current_temperature=args.current_temperature,
        )
        summary.outputs["reflection"] = reflection_path
        if final_stage == "reflection":
            return summary

    distillation_path: Optional[str] = None
    if includes_stage(final_stage, "distillation"):
        from src.srcd.batch_distillation import BatchSRCDDistillationRunner

        distillation_runner = BatchSRCDDistillationRunner(
            workspace_root=str(args.workspace_root),
            llm_client=llm,
            embedding_model=args.embedding_model,
            embedding_cache_dir=str(args.embedding_cache_dir) if args.embedding_cache_dir else None,
            device=args.embedding_device,
            embedding_backend_type=args.embedding_backend,
        )
        distillation_path = distillation_runner.process_instance(
            instance_id=instance_id,
            reflection_json=reflection_path,
            top_k_per_candidate=args.top_k_per_candidate,
        )
        summary.outputs["distillation"] = distillation_path
        if final_stage in {"distillation", "search"}:
            return summary

    if includes_stage(final_stage, "tspf"):
        from src.tspf.batch_tspf import BatchTSPFRunner

        distillation_json = Path(distillation_path) if distillation_path else default_distillation_path(
            args.workspace_root,
            instance_id,
        )
        evidence_json = args.test_evidence_json or default_tspf_evidence_path(args.workspace_root, instance_id)
        tspf_runner = BatchTSPFRunner(args.workspace_root)
        tspf_path = tspf_runner.process_instance(
            instance_id=instance_id,
            distillation_json=distillation_json,
            test_evidence_json=evidence_json,
            mu=args.mu,
            max_patches=args.max_patches,
            require_test_evidence=not args.allow_missing_test_evidence,
        )
        summary.outputs["tspf"] = str(tspf_path)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CGARF by paper-aligned stages.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--shared-workspace-root", type=Path, default=None)
    parser.add_argument("--repo-cache-root", type=Path, default=None)
    parser.add_argument("--method", default="orcaloca", choices=["orcaloca", "agentless"])
    parser.add_argument("--instance_ids", "--instance-ids", nargs="+", required=False, default=[])
    parser.add_argument(
        "--final_stage",
        "--final-stage",
        default="trace_analysis",
        help="One of: crg, trace_analysis, srcd_initial, reflection, search, tspf.",
    )
    parser.add_argument("--model-profile", default="openai-gpt-4.1")
    parser.add_argument("--provider", default=None, help="Override provider: openai, vllm, qwen, mock.")
    parser.add_argument("--model", default=None, help="Override model name.")
    parser.add_argument("--api-base", default=None, help="Override OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=None, help="Optional direct API key override.")
    parser.add_argument("--key-cfg", type=Path, default=None, help="Path to key.cfg with KEY=value entries.")
    parser.add_argument("--list-model-profiles", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve configuration and print the plan only.")

    parser.add_argument("--max-paths-per-candidate", type=int, default=100)
    parser.add_argument("--total-sampling-budget", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--current-temperature", type=float, default=0.2)
    parser.add_argument("--top-k-per-candidate", type=int, default=5)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-cache-dir", type=Path, default=None)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "siliconflow", "local"],
        default="auto",
    )
    parser.add_argument("--mu", type=float, default=0.6)
    parser.add_argument("--max-patches", type=int, default=5)
    parser.add_argument("--test-evidence-json", type=Path, default=None)
    parser.add_argument("--allow-missing-test-evidence", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.workspace_root = args.workspace_root.resolve()

    if args.list_model_profiles:
        print(json.dumps(MODEL_PROFILES, indent=2, ensure_ascii=False))
        return
    if not args.instance_ids:
        parser.error("--instance_ids is required unless --list-model-profiles is used")

    plan = {
        "workspace_root": str(args.workspace_root),
        "instances": args.instance_ids,
        "final_stage": normalize_stage(args.final_stage),
        "method": args.method,
        "model_profile": args.model_profile,
        "provider": args.provider,
        "model": args.model,
        "api_base": args.api_base,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    llm = create_configured_llm(
        profile=args.model_profile,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        workspace_root=args.workspace_root,
        key_cfg=args.key_cfg,
        require_api_key=args.provider != "mock",
    )

    summaries: List[InstanceRunSummary] = []
    for instance_id in args.instance_ids:
        try:
            logger.info("Running CGARF instance={} final_stage={}", instance_id, args.final_stage)
            summaries.append(run_instance(args, llm, instance_id))
        except Exception as exc:
            logger.exception("CGARF run failed for {}", instance_id)
            summaries.append(
                InstanceRunSummary(
                    instance_id=instance_id,
                    final_stage=normalize_stage(args.final_stage),
                    status="failed",
                    error=str(exc),
                )
            )
            if not args.continue_on_error:
                break

    payload = {
        "plan": plan,
        "results": [summary.__dict__ for summary in summaries],
    }
    summary_json = args.summary_json
    if summary_json is None:
        output_dir = args.workspace_root / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_stage = normalize_stage(args.final_stage)
        summary_json = output_dir / f"cgarf_{safe_stage}_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.success("Run summary saved to {}", summary_json)


if __name__ == "__main__":
    main()
