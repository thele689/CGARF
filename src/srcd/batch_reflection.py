import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.common.data_structures import IssueContext, PatchCandidate
from src.common.llm_interface import QwenLLMInterface
from src.phase0_integrator.fault_localization_loader import UnifiedFaultLocalizationLoader
from src.srcd.repair_generator import PatchGenerationError, RepairGenerator, SRCDCandidateInput
from src.srcd.reflection_scorer import ReflectionScorer


class BatchSRCDReflectionRunner:
    """Run paper section 3.2.2 on top of saved 3.2.1 initial patches."""

    def __init__(self, workspace_root: str, llm_client=None, fl_method: str = "orcaloca"):
        self.workspace_root = Path(workspace_root)
        self.llm = llm_client
        self.loader = UnifiedFaultLocalizationLoader(fl_method)
        self.generator = RepairGenerator(llm=llm_client)

    def _load_srcd_initial(self, instance_id: str, srcd_json: Optional[str] = None) -> Optional[dict]:
        path = Path(srcd_json) if srcd_json else (
            self.workspace_root / "data" / "srcd" / f"{instance_id}_srcd_initial.json"
        )
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_result(self, instance_id: str, payload: dict) -> str:
        out_dir = self.workspace_root / "data" / "srcd_reflection"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{instance_id}_srcd_reflection.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return str(out_path)

    def process_instance(
        self,
        instance_id: str,
        srcd_json: Optional[str] = None,
        current_temperature: float = 0.2,
    ) -> Optional[str]:
        srcd_payload = self._load_srcd_initial(instance_id, srcd_json=srcd_json)
        if srcd_payload is None:
            logger.error(f"Missing SRCD initial patch payload for {instance_id}")
            return None

        context = self.loader.load_instance(instance_id)
        if context is None:
            logger.error(f"Failed to load issue context for {instance_id}")
            return None

        srcd_initial = srcd_payload["srcd_initial"]
        candidate_inputs = [SRCDCandidateInput(**item) for item in srcd_initial["candidate_inputs"]]
        patches = [PatchCandidate(**item) for item in srcd_initial["initial_patches"]]

        issue_context = IssueContext(
            id=context.instance_id,
            description=context.problem_statement,
            repo_path=str(self.workspace_root / "repos"),
            candidates=[item.candidate_id for item in candidate_inputs],
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

        initial_patches_by_location = {patch.location: patch for patch in patches}
        scorer = ReflectionScorer(self.llm)
        scores = {}
        candidate_runs = {}

        for candidate_input in candidate_inputs:
            initial_patch = initial_patches_by_location.get(candidate_input.candidate_id)
            if initial_patch is None:
                logger.warning(f"Missing initial patch for candidate {candidate_input.candidate_id}")
                continue

            allocated_samples = max(1, int(candidate_input.allocated_samples or 1))
            current_patch = initial_patch
            generation_temperature = current_temperature
            rounds = []

            for round_index in range(1, allocated_samples + 1):
                reflection = scorer.score_patch_candidate(
                    patch=current_patch,
                    candidate_input=candidate_input,
                    issue_text=context.problem_statement,
                    current_temperature=generation_temperature,
                )
                scores[current_patch.patch_id] = reflection
                rounds.append(
                    {
                        "round": round_index,
                        "generation_temperature": generation_temperature,
                        "patch": {
                            "patch_id": current_patch.patch_id,
                            "location": current_patch.location,
                            "patch_content": current_patch.patch_content,
                            "generated_round": current_patch.generated_round,
                            "credibility_from_location": current_patch.credibility_from_location,
                            "final_score": current_patch.final_score,
                            "test_result": current_patch.test_result,
                        },
                        "reflection": reflection.to_dict(),
                    }
                )

                if round_index == allocated_samples:
                    break

                next_temperature = (
                    reflection.suggested_temperature
                    if reflection.suggested_temperature is not None
                    else generation_temperature
                )
                try:
                    current_patch = self.generator.generate_refined_patch(
                        issue_context=issue_context,
                        candidate_input=candidate_input,
                        reflection_payload=reflection.to_dict(),
                        generation_temperature=next_temperature,
                        patch_index=round_index + 1,
                    )
                except PatchGenerationError as exc:
                    logger.warning(
                        f"Stopping refinement for {candidate_input.candidate_id}: {exc}"
                    )
                    break
                generation_temperature = next_temperature

            candidate_runs[candidate_input.candidate_id] = {
                "candidate_id": candidate_input.candidate_id,
                "candidate_location": candidate_input.candidate_location,
                "allocated_samples": allocated_samples,
                "candidate_credibility": candidate_input.candidate_credibility,
                "normalized_weight": candidate_input.normalized_weight,
                "rounds": rounds,
            }

        payload = {
            "instance_id": instance_id,
            "repo": context.repo,
            "base_commit": context.base_commit,
            "method": self.loader.method,
            "llm_model": getattr(self.llm, "model_name", None),
            "llm_call_count": getattr(self.llm, "call_count", None),
            "current_temperature": current_temperature,
            "candidate_runs": candidate_runs,
            "reflection_scores": {
                patch_id: score.to_dict() for patch_id, score in scores.items()
            },
        }
        saved_path = self._save_result(instance_id, payload)
        logger.success(f"SRCD 3.2.2 result saved to {saved_path}")
        return saved_path


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Run SRCD 3.2.2 reflection for SWE-Bench instances.")
    parser.add_argument("--workspace-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--method", default="orcaloca")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--srcd-json", default=None)
    parser.add_argument("--current-temperature", type=float, default=0.2)
    args = parser.parse_args()

    runner = BatchSRCDReflectionRunner(
        workspace_root=args.workspace_root,
        llm_client=QwenLLMInterface(),
        fl_method=args.method,
    )
    runner.process_instance(
        instance_id=args.instance_id,
        srcd_json=args.srcd_json,
        current_temperature=args.current_temperature,
    )
