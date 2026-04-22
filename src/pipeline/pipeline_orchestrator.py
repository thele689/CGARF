"""
ModuleN: CGARF Pipeline Orchestrator
====================================

Unified pipeline that integrates all Phase 1-4 modules into a complete
automated bug repair system.

Workflow:
  Input: Buggy Code + Issue Description
    ↓
  [Pipeline.run()] orchestrates:
    - Phase 1: Initialize (data structures, LLM, utilities)
    - Phase 2: CG-MAD (CRG, debate, credibility, weights)
    - Phase 3: SRCD (repair generation, scoring, distillation)
    - Phase 4: TSPF (test synthesis, patch validation, filtering)
    ↓
  Output: Top-5 Verified Patches with scores
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import yaml
import time
from datetime import datetime
from loguru import logger

from src.common.data_structures import IssueContext, CodeEntity, RepairCandidate
from src.common.llm_interface import LLMInterface, OpenAILLMInterface, MockLLMInterface, QwenLLMInterface
from src.crg.crg_builder import CRGBuilder
from src.crg.agent_manager import AgentManager
from src.crg.path_processing import PathProcessor
from src.crg.edge_weight_manager import EdgeWeightManager
from src.srcd.repair_generator import RepairGenerator
from src.srcd.reflection_scorer import ReflectionScorer
from src.srcd.consistency_distiller import ConsistencyDistiller
from src.tspf.test_synthesizer import TestSynthesizer
from src.tspf.patch_filter import PatchEvaluator
from src.tspf.patch_filter import VerifiedPatch


@dataclass
class PipelineStage:
    """Record of a single pipeline stage execution"""
    stage_name: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    status: str = "running"  # running, completed, failed
    error_message: str = ""

    def complete(self, success: bool = True, error: str = ""):
        """Mark stage as completed"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = "completed" if success else "failed"
        self.error_message = error


@dataclass
class PipelineConfig:
    """Pipeline configuration loaded from YAML"""
    llm_provider: str = "openai"  # openai, qwen, mock
    llm_model: str = "gpt-4"
    llm_api_key: Optional[str] = None
    llm_api_base: Optional[str] = None  # For custom API endpoints (Qwen, etc.)
    log_level: str = "INFO"
    
    # Phase 2 config
    phase2_agent_iterations: int = 3
    phase2_path_depth_limit: int = 20
    phase2_max_paths: int = 100
    
    # Phase 3 config
    phase3_max_repairs: int = 50
    phase3_semantic_model: str = "all-MiniLM-L6-v2"
    phase3_n_clusters: int = 5
    
    # Phase 4 config
    phase4_test_timeout: int = 30
    phase4_min_pass_rate: float = 0.5
    phase4_max_patches: int = 5
    
    # Pipeline config
    pipeline_timeout: int = 300  # 5 minutes total
    
    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'PipelineConfig':
        """Load configuration from YAML file"""
        try:
            with open(yaml_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Extract pipeline section
            pipeline_config = config_dict.get('pipeline', {})
            phase2_config = config_dict.get('phase2', {})
            phase3_config = config_dict.get('phase3', {})
            phase4_config = config_dict.get('phase4', {})
            
            return cls(
                llm_provider=pipeline_config.get('llm_provider', 'openai'),
                llm_model=pipeline_config.get('llm_model', 'gpt-4'),
                llm_api_key=pipeline_config.get('llm_api_key'),
                llm_api_base=pipeline_config.get('llm_api_base'),
                log_level=pipeline_config.get('log_level', 'INFO'),
                phase2_agent_iterations=phase2_config.get('agent_iterations', 3),
                phase2_path_depth_limit=phase2_config.get('path_depth', 20),
                phase2_max_paths=phase2_config.get('max_paths', 100),
                phase3_max_repairs=phase3_config.get('max_repairs', 50),
                phase3_semantic_model=phase3_config.get('semantic_model', 'all-MiniLM-L6-v2'),
                phase3_n_clusters=phase3_config.get('n_clusters', 5),
                phase4_test_timeout=phase4_config.get('test_timeout', 30),
                phase4_min_pass_rate=phase4_config.get('min_pass_rate', 0.5),
                phase4_max_patches=phase4_config.get('max_patches', 5),
                pipeline_timeout=pipeline_config.get('timeout', 300)
            )
        except Exception as e:
            logger.warning(f"Failed to load config from {yaml_file}: {e}, using defaults")
            return cls()


class CGARFPipeline:
    """Main CGARF pipeline orchestrator"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.llm = self._init_llm()
        self.stages: List[PipelineStage] = []
        self.start_time = 0.0
        
        logger.add(lambda msg: None, level=self.config.log_level)
        logger.info(f"CGARF Pipeline initialized with {self.config.llm_provider} LLM")

    def _init_llm(self) -> LLMInterface:
        """Initialize LLM interface"""
        if self.config.llm_provider == "openai":
            return OpenAILLMInterface(
                model_name=self.config.llm_model,
                api_key=self.config.llm_api_key
            )
        elif self.config.llm_provider == "qwen":
            return QwenLLMInterface(
                model_name=self.config.llm_model,
                api_key=self.config.llm_api_key,
                api_base=self.config.llm_api_base or "https://api.siliconflow.cn/v1"
            )
        elif self.config.llm_provider == "mock":
            return MockLLMInterface()
        else:
            logger.warning(f"Unknown LLM provider: {self.config.llm_provider}, using mock")
            return MockLLMInterface()

    def run(
        self,
        buggy_code: str,
        issue_description: str,
        file_path: str = "unknown.py"
    ) -> List[VerifiedPatch]:
        """
        Run complete CGARF pipeline

        Args:
            buggy_code: Original buggy code
            issue_description: Description of the issue/bug
            file_path: Path to the buggy file (optional)

        Returns:
            List of Top-5 verified patches sorted by quality
        """
        self.start_time = time.time()
        logger.info("=" * 70)
        logger.info(f"CGARF Pipeline started at {datetime.now().isoformat()}")
        logger.info(f"Issue: {issue_description[:100]}...")
        logger.info("=" * 70)

        try:
            # Phase 1: Initialize
            issue_context = self._run_phase1(buggy_code, issue_description, file_path)
            if not issue_context:
                return []

            # Phase 2: CG-MAD
            crg, credibilities = self._run_phase2(issue_context)
            if not crg:
                return []

            # Phase 3: SRCD
            repairs = self._run_phase3(issue_context, crg)
            if not repairs:
                return []

            # Phase 4: TSPF
            patches = self._run_phase4(repairs, issue_context)

            # Log completion
            elapsed = time.time() - self.start_time
            logger.info("=" * 70)
            logger.info(f"Pipeline completed successfully in {elapsed:.2f}s")
            logger.info(f"Top patches: {len(patches)}")
            logger.info("=" * 70)

            return patches

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return []

    def _run_phase1(
        self, buggy_code: str, issue_description: str, file_path: str
    ) -> Optional[IssueContext]:
        """Phase 1: Initialize data structures and utilities"""
        stage = PipelineStage("Phase 1: Initialize", time.time())
        self.stages.append(stage)

        try:
            # Create issue context with correct parameters
            issue_context = IssueContext(
                id="issue_001",
                description=issue_description,
                repo_path=file_path,
                candidates=["candidate_001"],  # Placeholder
                test_framework="pytest",
                timeout_seconds=self.config.pipeline_timeout,
                metadata={
                    'buggy_code': buggy_code,
                    'file_path': file_path
                }
            )

            stage.complete(True)
            logger.info(f"Phase 1: Initialized issue context")
            return issue_context

        except Exception as e:
            stage.complete(False, str(e))
            logger.error(f"Phase 1 failed: {e}")
            return None

    def _run_phase2(
        self, issue_context: IssueContext
    ) -> Tuple[Optional[Any], Optional[Dict]]:
        """Phase 2: CG-MAD analysis"""
        stage = PipelineStage("Phase 2: CG-MAD", time.time())
        self.stages.append(stage)

        try:
            # Build CRG
            crg_builder = CRGBuilder(issue_context, self.llm)
            crg = crg_builder.build()
            logger.info(f"Phase 2: Built CRG with {len(crg.nodes)} nodes")

            # Run agent debate
            agent_manager = AgentManager(crg, self.llm)
            for _ in range(self.config.phase2_agent_iterations):
                agent_manager.run_debate_round()

            # Process paths and compute credibility
            path_processor = PathProcessor(crg)
            paths = path_processor.find_candidate_paths(
                max_depth=self.config.phase2_path_depth_limit,
                max_paths=self.config.phase2_max_paths
            )

            credibilities = {
                path.id: path_processor.compute_credibility(path)
                for path in paths
            }
            logger.info(f"Phase 2: Computed credibility for {len(credibilities)} paths")

            # Fuse edge weights
            weight_manager = EdgeWeightManager(crg)
            weight_manager.fuse_weights(paths, credibilities)

            stage.complete(True)
            return crg, credibilities

        except Exception as e:
            stage.complete(False, str(e))
            logger.error(f"Phase 2 failed: {e}")
            return None, None

    def _run_phase3(
        self, issue_context: IssueContext, crg: Any
    ) -> Optional[List[VerifiedPatch]]:
        """Phase 3: SRCD - Repair generation and filtering"""
        stage = PipelineStage("Phase 3: SRCD", time.time())
        self.stages.append(stage)

        try:
            # Generate repairs
            repair_gen = RepairGenerator(max_mutations=self.config.phase3_max_repairs)
            repairs = repair_gen.generate_repairs(issue_context)
            logger.info(f"Phase 3: Generated {len(repairs)} repair candidates")

            if not repairs:
                stage.complete(False, "No repairs generated")
                return []

            # Score repairs
            scorer = ReflectionScorer()
            scored_repairs = scorer.score_repairs(repairs, issue_context)
            logger.info(f"Phase 3: Scored {len(scored_repairs)} repairs")

            # Distill repairs
            distiller = ConsistencyDistiller(
                n_clusters=self.config.phase3_n_clusters
            )
            distilled = distiller.distill_repairs(
                repairs, list(scored_repairs.values())
            )
            logger.info(f"Phase 3: Distilled to {len(distilled)} consensus repairs")

            stage.complete(True)
            return distilled

        except Exception as e:
            stage.complete(False, str(e))
            logger.error(f"Phase 3 failed: {e}")
            return []

    def _run_phase4(
        self, repairs: List[Any], issue_context: IssueContext
    ) -> List[VerifiedPatch]:
        """Phase 4: TSPF - Test synthesis and patch validation"""
        stage = PipelineStage("Phase 4: TSPF", time.time())
        self.stages.append(stage)

        try:
            # Synthesize tests
            synth = TestSynthesizer()
            test_structures = synth.synthesize_batch(repairs, issue_context)
            logger.info(f"Phase 4: Synthesized tests for {len(test_structures)} repairs")

            # Evaluate patches
            evaluator = PatchEvaluator(
                min_pass_rate=self.config.phase4_min_pass_rate
            )
            verified_patches = evaluator.evaluate_repairs(repairs, test_structures)
            logger.info(f"Phase 4: Verified {len(verified_patches)} patches")

            stage.complete(True)
            return verified_patches[:self.config.phase4_max_patches]

        except Exception as e:
            stage.complete(False, str(e))
            logger.error(f"Phase 4 failed: {e}")
            return []

    def get_stage_report(self) -> Dict[str, float]:
        """Get execution time report for each stage"""
        return {
            stage.stage_name: stage.duration
            for stage in self.stages
        }

    def get_execution_summary(self) -> str:
        """Get summary of pipeline execution"""
        total_time = time.time() - self.start_time
        stage_times = self.get_stage_report()

        summary = f"""
CGARF Pipeline Execution Summary
================================
Total Time: {total_time:.2f}s

Stage Breakdown:
"""
        for stage_name, duration in stage_times.items():
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            summary += f"  {stage_name}: {duration:.2f}s ({percentage:.1f}%)\n"

        summary += f"\nStages Completed: {len([s for s in self.stages if s.status == 'completed'])}\n"
        summary += f"Stages Failed: {len([s for s in self.stages if s.status == 'failed'])}\n"

        return summary


class PipelineFactory:
    """Factory for creating configured pipelines"""

    @staticmethod
    def create(llm_provider: str = "openai",
               model: str = "gpt-4",
               api_key: Optional[str] = None,
               api_base: Optional[str] = None) -> CGARFPipeline:
        """Create pipeline with specified LLM configuration"""
        config = PipelineConfig(
            llm_provider=llm_provider,
            llm_model=model,
            llm_api_key=api_key,
            llm_api_base=api_base
        )
        return CGARFPipeline(config)

    @staticmethod
    def create_from_config_file(config_file: str) -> CGARFPipeline:
        """Create pipeline from YAML config file"""
        config = PipelineConfig.from_yaml(config_file)
        return CGARFPipeline(config)

    @staticmethod
    def create_default() -> CGARFPipeline:
        """Create pipeline with default configuration"""
        return CGARFPipeline()

    @staticmethod
    def create_mock() -> CGARFPipeline:
        """Create pipeline with mock LLM (for testing)"""
        config = PipelineConfig(llm_provider="mock")
        return CGARFPipeline(config)
