import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.common.llm_interface import QwenLLMInterface
from src.phase0_integrator.fault_localization_loader import EnhancedIssueContext
from src.phase1_causal_analysis.batch_crg_constructor import BatchCRGConstructor
from src.phase1_causal_analysis.cg_mad import CGMADMechanism
from src.phase1_causal_analysis.causal_relevance_graph import CausalRelevanceGraph


class BatchCGMADRunner:
    """Run paper section 3.1.2 on top of the current phase1 CRG build flow."""

    def __init__(
        self,
        workspace_root: str,
        llm_client=None,
        fl_method: str = "orcaloca",
        shared_workspace_root: Optional[str] = None,
        repo_cache_root: Optional[str] = None,
        max_paths_per_candidate: Optional[int] = None,
    ):
        self.workspace_root = Path(workspace_root)
        self.constructor = BatchCRGConstructor(
            workspace_root=workspace_root,
            llm_client=llm_client,
            fl_method=fl_method,
            shared_workspace_root=shared_workspace_root,
            repo_cache_root=repo_cache_root,
            max_paths_per_candidate=max_paths_per_candidate or 100,
        )

    def _save_result(self, instance_id: str, payload: dict) -> str:
        out_dir = self.workspace_root / "data" / "cg_mad"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{instance_id}_cg_mad.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return str(out_path)

    def _try_load_existing_crg(self, context: EnhancedIssueContext) -> Optional[CausalRelevanceGraph]:
        """Reuse an existing phase1 CRG when both the code graph and CRG are cached."""

        storages = [self.constructor.storage]
        if self.constructor.shared_storage:
            storages.append(self.constructor.shared_storage)

        code_graph = None
        for storage in storages:
            code_graph = storage.load_code_graph(context.repo, context.base_commit)
            if code_graph:
                break

        if code_graph is None:
            return None

        for storage in storages:
            crg = storage.load_crg(context.instance_id, code_graph)
            if crg:
                logger.info(
                    f"Reused cached phase1 CRG for {context.instance_id} from {storage.base_dir}"
                )
                return crg

        return None

    def process_instance(
        self,
        context: EnhancedIssueContext,
        max_paths_per_candidate: Optional[int] = None,
    ) -> Optional[str]:
        crg = self._try_load_existing_crg(context)
        if crg is None:
            crg = self.constructor.build_instance_crg(
                context,
                max_paths_per_candidate=max_paths_per_candidate,
            )
        if crg is None:
            return None

        mechanism = CGMADMechanism(
            crg=crg,
            issue_description=context.problem_statement,
            llm=self.constructor.llm,
            max_paths_per_candidate=max_paths_per_candidate,
        )
        result = mechanism.run()
        payload = {
            "instance_id": context.instance_id,
            "repo": context.repo,
            "base_commit": context.base_commit,
            "method": self.constructor.loader.method,
            "cg_mad": result.to_dict(),
        }
        saved_path = self._save_result(context.instance_id, payload)
        logger.success(f"CG-MAD result saved to {saved_path}")
        return saved_path

    def run_single_instance(
        self,
        instance_id: str,
        method: Optional[str] = None,
        max_paths_per_candidate: Optional[int] = None,
    ) -> Optional[str]:
        if method and method != self.constructor.loader.method:
            self.constructor.loader.set_method(method)

        context = self.constructor.loader.load_instance(instance_id)
        if not context:
            logger.error(f"Failed to load instance context: {instance_id}")
            return None

        return self.process_instance(
            context=context,
            max_paths_per_candidate=max_paths_per_candidate,
        )


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Run CG-MAD for SWE-Bench instances.")
    parser.add_argument("--workspace-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--shared-workspace-root", default=None)
    parser.add_argument("--repo-cache-root", default=None)
    parser.add_argument("--method", default="orcaloca")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--max-paths-per-candidate", type=int, default=None)
    args = parser.parse_args()

    runner = BatchCGMADRunner(
        workspace_root=args.workspace_root,
        llm_client=QwenLLMInterface(),
        fl_method=args.method,
        shared_workspace_root=args.shared_workspace_root,
        repo_cache_root=args.repo_cache_root,
        max_paths_per_candidate=args.max_paths_per_candidate,
    )
    runner.run_single_instance(
        instance_id=args.instance_id,
        method=args.method,
        max_paths_per_candidate=args.max_paths_per_candidate,
    )
