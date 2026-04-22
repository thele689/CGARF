import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.common.data_structures import IssueContext
from src.common.llm_interface import QwenLLMInterface
from src.phase1_causal_analysis.batch_cg_mad import BatchCGMADRunner
from src.phase1_causal_analysis.causal_relevance_graph import CausalRelevanceGraph
from src.srcd.repair_generator import RepairGenerator


class BatchSRCDRunner:
    """Run paper section 3.2.1 on top of cached CG-MAD outputs."""

    def __init__(
        self,
        workspace_root: str,
        llm_client=None,
        fl_method: str = "orcaloca",
        shared_workspace_root: Optional[str] = None,
        repo_cache_root: Optional[str] = None,
    ):
        self.workspace_root = Path(workspace_root)
        self.cg_mad_runner = BatchCGMADRunner(
            workspace_root=workspace_root,
            llm_client=llm_client,
            fl_method=fl_method,
            shared_workspace_root=shared_workspace_root,
            repo_cache_root=repo_cache_root,
        )
        self.generator = RepairGenerator(llm=llm_client)

    def _load_cg_mad_payload(self, instance_id: str, cg_mad_json: Optional[str] = None) -> Optional[dict]:
        path = Path(cg_mad_json) if cg_mad_json else (
            self.workspace_root / "data" / "cg_mad" / f"{instance_id}_cg_mad.json"
        )
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_result(self, instance_id: str, payload: dict) -> str:
        out_dir = self.workspace_root / "data" / "srcd"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{instance_id}_srcd_initial.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return str(out_path)

    def _load_cached_crg(self, context) -> Optional[CausalRelevanceGraph]:
        return self.cg_mad_runner._try_load_existing_crg(context)

    def process_instance(
        self,
        instance_id: str,
        total_sampling_budget: int = 8,
        max_candidates: Optional[int] = None,
        max_paths_per_candidate: Optional[int] = None,
        cg_mad_json: Optional[str] = None,
    ) -> Optional[str]:
        context = self.cg_mad_runner.constructor.loader.load_instance(instance_id)
        if context is None:
            logger.error(f"Failed to load instance context: {instance_id}")
            return None

        cg_mad_payload = self._load_cg_mad_payload(instance_id, cg_mad_json=cg_mad_json)
        if cg_mad_payload is None:
            logger.info(f"No cached CG-MAD found for {instance_id}, running phase2 first")
            saved = self.cg_mad_runner.process_instance(
                context=context,
                max_paths_per_candidate=max_paths_per_candidate,
            )
            if saved is None:
                return None
            cg_mad_payload = self._load_cg_mad_payload(instance_id, cg_mad_json=cg_mad_json)
            if cg_mad_payload is None:
                logger.error(f"Failed to reload CG-MAD payload after generation: {instance_id}")
                return None

        crg = self._load_cached_crg(context)
        if crg is None:
            logger.error(f"Failed to load CRG required for SRCD: {instance_id}")
            return None

        cg_mad_result = cg_mad_payload.get("cg_mad", cg_mad_payload)

        issue_context = IssueContext(
            id=context.instance_id,
            description=context.problem_statement,
            repo_path=self.cg_mad_runner.constructor.get_checkout_repo(context),
            candidates=[item["candidate_id"] for item in cg_mad_result["candidate_assessments"]],
            test_framework=context.test_framework,
            timeout_seconds=context.timeout_seconds,
            metadata={
                "repo": context.repo,
                "base_commit": context.base_commit,
                "test_paths": list(context.test_paths),
                "fail_to_pass": list(context.fail_to_pass),
                "pass_to_pass": list(context.pass_to_pass),
            },
        )

        bundle = self.generator.generate_initial_patches_from_cgmad(
            issue_context=issue_context,
            cg_mad_result=cg_mad_result,
            crg=crg,
            total_sampling_budget=total_sampling_budget,
            max_candidates=max_candidates,
        )
        payload = {
            "instance_id": context.instance_id,
            "repo": context.repo,
            "base_commit": context.base_commit,
            "method": self.cg_mad_runner.constructor.loader.method,
            "llm_model": getattr(self.generator.llm, "model_name", None),
            "llm_call_count": getattr(self.generator.llm, "call_count", None),
            "srcd_initial": bundle.to_dict(),
        }
        saved_path = self._save_result(instance_id, payload)
        logger.success(f"SRCD 3.2.1 result saved to {saved_path}")
        return saved_path


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Run SRCD 3.2.1 for SWE-Bench instances.")
    parser.add_argument("--workspace-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--shared-workspace-root", default=None)
    parser.add_argument("--repo-cache-root", default=None)
    parser.add_argument("--method", default="orcaloca")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--total-sampling-budget", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-paths-per-candidate", type=int, default=None)
    parser.add_argument("--cg-mad-json", default=None)
    args = parser.parse_args()

    runner = BatchSRCDRunner(
        workspace_root=args.workspace_root,
        llm_client=QwenLLMInterface(),
        fl_method=args.method,
        shared_workspace_root=args.shared_workspace_root,
        repo_cache_root=args.repo_cache_root,
    )
    runner.process_instance(
        instance_id=args.instance_id,
        total_sampling_budget=args.total_sampling_budget,
        max_candidates=args.max_candidates,
        max_paths_per_candidate=args.max_paths_per_candidate,
        cg_mad_json=args.cg_mad_json,
    )
