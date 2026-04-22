import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.phase0_integrator.fault_localization_loader import UnifiedFaultLocalizationLoader, EnhancedIssueContext
from src.phase1_causal_analysis.code_graph_builder import CodeGraphBuilder
from src.phase1_causal_analysis.graph_storage import StorageManager
from src.phase1_causal_analysis.causal_relevance_graph import (
    CodeGraph, CodeEntity, CausalRelevanceGraph, FailureEvidence, CRGBuilder
)
from src.phase1_causal_analysis.llm_edge_weighting import LLMEdgeWeightingStrategy
from src.common.llm_interface import QwenLLMInterface

class BatchCRGConstructor:
    """
    Orchestrates building causal relevance graphs for 300 instances.
    """
    def __init__(
        self,
        workspace_root: str,
        llm_client=None,
        fl_method: str = "orcaloca",
        shared_workspace_root: Optional[str] = None,
        repo_cache_root: Optional[str] = None,
        max_paths_per_candidate: int = 100,
    ):
        self.workspace_root = Path(workspace_root)
        self.loader = UnifiedFaultLocalizationLoader(fl_method)
        self.storage = StorageManager(base_dir=str(self.workspace_root / "data/code_graphs"))
        self.shared_storage = None
        if shared_workspace_root:
            shared_root = Path(shared_workspace_root)
            shared_base_dir = shared_root / "data/code_graphs"
            if shared_base_dir.resolve() != (self.workspace_root / "data/code_graphs").resolve():
                self.shared_storage = StorageManager(base_dir=str(shared_base_dir))
        self.llm = llm_client or QwenLLMInterface()
        self.llm_weighting = LLMEdgeWeightingStrategy(self.llm)
        self.repo_cache_root = Path(repo_cache_root) if repo_cache_root else (self.workspace_root / "repos")
        self.max_paths_per_candidate = max_paths_per_candidate
        
    import subprocess
    def get_checkout_repo(self, context: EnhancedIssueContext) -> str:
        """
        Clones the requested repository into the workspace and checks out the base_commit.
        Ensures the local directory matches the exact commit for AST parsing.
        """
        repo_name = context.repo.replace("/", "_")
        repos_dir = self.repo_cache_root
        repos_dir.mkdir(exist_ok=True)
        repo_path = repos_dir / repo_name
        
        # Original SWE-bench repo on GitHub
        repo_url = f"https://github.com/{context.repo}.git"

        if not repo_path.exists():
            logger.info(f"Cloning {repo_url} into {repo_path}...")
            import subprocess
            subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Checkout specific base commit
        logger.info(f"Checking out commit {context.base_commit} for {context.repo}...")
        import subprocess
        
        # Try standard checkout first
        try:
            subprocess.run(["git", "-C", str(repo_path), "reset", "--hard", "HEAD"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "-C", str(repo_path), "fetch", "origin"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # Fetch may fail, that's OK
            subprocess.run(["git", "-C", str(repo_path), "checkout", context.base_commit], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            # If checkout fails, try without fetch
            logger.warning(f"Failed to fetch or checkout {context.base_commit}, trying direct checkout...")
            try:
                subprocess.run(["git", "-C", str(repo_path), "checkout", context.base_commit], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e2:
                logger.error(f"Failed to checkout commit {context.base_commit}: {e2}")
                raise

        return str(repo_path)

    def _extract_failure_evidence(self, context: EnhancedIssueContext) -> List[FailureEvidence]:
        """Convert issue/test metadata into failure evidence roots for CRG construction."""
        test_paths = list(dict.fromkeys(context.test_paths))
        if not test_paths:
            return [
                FailureEvidence(
                    symptom_type="problem_statement",
                    symptom_location="issue_description",
                    symptom_message=context.problem_statement[:500],
                    test_case_id=context.instance_id,
                    stack_trace=[],
                )
            ]

        evidences = []
        for idx, test_path in enumerate(test_paths):
            evidences.append(
                FailureEvidence(
                    symptom_type="failing_test_file",
                    symptom_location=test_path,
                    symptom_message=context.problem_statement[:500],
                    test_case_id=f"{context.instance_id}::test_file::{idx}",
                    stack_trace=[],
                )
            )
        return evidences

    def _get_candidates_from_bug_locations(self, context: EnhancedIssueContext, code_graph: CodeGraph) -> List[CodeEntity]:
        """
        Extract candidate entities from bug locations.
        Matches on: class+method (highest priority), method name, line number
        range, or file fallback.
        """
        candidates = []
        seen_ids = set()
        logger.info(f"Processing {len(context.bug_locations)} bug locations for {context.instance_id}")
        
        for i, loc in enumerate(context.bug_locations):
            logger.info(
                f"  [{i}] file='{loc.file_path}' class='{loc.class_name}' method='{loc.method_name}'"
            )
            matched_entities = []
            
            # Strategy 1: Match by class + method/function name when class info exists.
            # This is important for common methods such as __init__, where method-only
            # matching would incorrectly expand one localized candidate into many classes.
            if loc.method_name:
                logger.debug(
                    f"    Strategy 1: Searching for class '{loc.class_name}' method "
                    f"'{loc.method_name}' in {loc.file_path}"
                )
                for entity in code_graph.entities.values():
                    # Check if file path matches
                    if loc.file_path not in entity.file_path:
                        continue
                    # Check if entity name matches method name
                    if (
                        entity.entity_type.value == "function"
                        and (entity.function_name == loc.method_name or entity.name == loc.method_name)
                        and (not loc.class_name or entity.class_name == loc.class_name)
                    ):
                        matched_entities.append(entity)
                        logger.debug(
                            f"      ✓ Matched {entity.class_name + '.' if entity.class_name else ''}{entity.name}"
                        )

            # Strategy 1b: If the localization names a class but strict
            # class+method matching fails, fall back to method-only matching.
            if not matched_entities and loc.method_name and loc.class_name:
                logger.debug(
                    f"    Strategy 1b: No strict class match; falling back to method "
                    f"'{loc.method_name}' in {loc.file_path}"
                )
                for entity in code_graph.entities.values():
                    if loc.file_path not in entity.file_path:
                        continue
                    if (
                        entity.entity_type.value == "function"
                        and (entity.function_name == loc.method_name or entity.name == loc.method_name)
                    ):
                        matched_entities.append(entity)
                        logger.debug(
                            f"      ✓ Fallback matched {entity.class_name + '.' if entity.class_name else ''}{entity.name}"
                        )
            
            # Strategy 2: Match by line number (if available)
            if not matched_entities and loc.line_start is not None:
                logger.debug(f"    Strategy 2: Searching by line {loc.line_start}-{loc.line_end}")

                for entity in code_graph.entities.values():
                    if loc.file_path not in entity.file_path:
                        continue
                    if entity.line_start is None or entity.line_end is None:
                        continue
                    # Check if location overlaps with entity bounds
                    if loc.line_start <= entity.line_end and (loc.line_end or loc.line_start) >= entity.line_start:
                        matched_entities.append(entity)
                        logger.debug(f"      ✓ Matched {entity.name} ({entity.line_start}-{entity.line_end})")
            
            # Strategy 3: Fallback to file entity
            if not matched_entities:
                logger.debug(f"    Strategy 3: File fallback")
                for ent_id, ent in code_graph.entities.items():
                    if ent.entity_type.value == "file" and loc.file_path in ent.file_path:
                        matched_entities.append(ent)
                        logger.debug(f"      ✓ Matched file {ent.file_path}")
                        break
            
            logger.info(f"      Result: {len(matched_entities)} entities matched")
            for entity in matched_entities:
                if entity.id in seen_ids:
                    continue
                seen_ids.add(entity.id)
                candidates.append(entity)
        
        logger.info(f"Extracted {len(candidates)} total candidate entities")
        return candidates

    def process_instance(
        self,
        context: EnhancedIssueContext,
        max_paths_per_candidate: Optional[int] = None,
    ) -> Optional[str]:
        crg = self.build_instance_crg(
            context,
            max_paths_per_candidate=max_paths_per_candidate,
        )
        if crg is None:
            return None
        return self.storage.save_crg(context.instance_id, crg)

    def build_instance_crg(
        self,
        context: EnhancedIssueContext,
        max_paths_per_candidate: Optional[int] = None,
    ) -> Optional[CausalRelevanceGraph]:
        """Build and refine the CRG for one instance without saving it."""

        instance_id = context.instance_id
        repo_id = context.repo
        base_commit = context.base_commit
        
        logger.info(f"Processing instance: {instance_id} ({repo_id} @ {base_commit})")
        
        code_graph = self.storage.load_code_graph(repo_id, base_commit)
        if not code_graph and self.shared_storage:
            code_graph = self.shared_storage.load_code_graph(repo_id, base_commit)
            if code_graph:
                logger.info(
                    f"Loaded shared CodeGraph cache for {repo_id} @ {base_commit}"
                )
        
        if not code_graph:
            repo_path = self.get_checkout_repo(context)
            if not os.path.exists(repo_path):
                logger.warning(f"Repo path does not exist for {instance_id}: {repo_path}. Skipping.")
                return None
            
            logger.info(f"Building AST CodeGraph for {repo_path}")
            # Note: in real test, this might take 10+ seconds for a big repo
            builder = CodeGraphBuilder()
            code_graph = builder.build_from_repository(repo_path)
            self.storage.save_code_graph(repo_id, base_commit, code_graph)

        evidences = self._extract_failure_evidence(context)
        candidates_L = self._get_candidates_from_bug_locations(context, code_graph)
        
        effective_max_paths = (
            max_paths_per_candidate
            if max_paths_per_candidate is not None
            else self.max_paths_per_candidate
        )

        crg_builder = CRGBuilder(
            code_graph,
            max_path_depth=8,
            max_paths_per_candidate=effective_max_paths,
            max_upstreams_per_node=3,
        )
        
        logger.info(
            "Building local CRG via backward tracing from {} candidates "
            "(max_paths_per_candidate={})...",
            len(candidates_L),
            effective_max_paths,
        )
        # Note the updated build method Signature
        local_crg = crg_builder.build(evidences, candidates_L=candidates_L)

        logger.info(f"Refining CRG weights using LLM pairwise protocol...")
        refined_crg = self.llm_weighting.apply_weights_to_crg(local_crg, evidences)
        return refined_crg

    def run_single_instance(
        self,
        instance_id: str,
        method: Optional[str] = None,
        max_paths_per_candidate: Optional[int] = None,
    ) -> Optional[str]:
        """Load one instance from Phase 0 and process it end-to-end."""

        if method and method != self.loader.method:
            self.loader.set_method(method)

        context = self.loader.load_instance(instance_id)
        if not context:
            logger.error(f"Failed to load instance context: {instance_id}")
            return None

        result = self.process_instance(
            context,
            max_paths_per_candidate=max_paths_per_candidate,
        )
        if result:
            logger.success(f"Single-instance CRG saved to {result}")
        return result

    def run_batch(self, limit: int = None):
        """Run CRG constructor across the OrcaLoca dataset"""
        logger.info("Loading OrcaLoca dataset via Phase 0 Loader...")
        self.loader.load_swe_bench()
        dataset = list(self.loader.load_instances_batch().values())
        
        if limit is not None:
            dataset = dataset[:limit]
            
        success_count = 0
        for ctx in dataset:
            try:
                result = self.process_instance(ctx)
                if result:
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to process {ctx.instance_id}: {str(e)}")
                
        logger.success(f"Batch CRG completion: Successfully processed {success_count}/{len(dataset)}.")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Build CRGs for SWE-Bench instances.")
    parser.add_argument("--workspace-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--shared-workspace-root", default=None)
    parser.add_argument("--repo-cache-root", default=None)
    parser.add_argument("--method", default="orcaloca")
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-paths-per-candidate", type=int, default=100)
    args = parser.parse_args()

    constructor = BatchCRGConstructor(
        workspace_root=args.workspace_root,
        fl_method=args.method,
        shared_workspace_root=args.shared_workspace_root,
        repo_cache_root=args.repo_cache_root,
        max_paths_per_candidate=args.max_paths_per_candidate,
    )

    if args.instance_id:
        constructor.run_single_instance(
            args.instance_id,
            method=args.method,
            max_paths_per_candidate=args.max_paths_per_candidate,
        )
    else:
        constructor.run_batch(limit=args.limit)
