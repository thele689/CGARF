#!/usr/bin/env python
"""
CGARF Experiments on SWE-Bench

Main script for running CGARF on SWE-Bench benchmark problems.
Supports various experiment modes: single run, batch, ablation study, etc.

Usage:
    # Basic single run
    python run_swebench_experiments.py --subset lite --max-problems 10

    # Full evaluation
    python run_swebench_experiments.py --subset lite --output-dir ./swe-bench-results

    # Ablation study
    python run_swebench_experiments.py --subset lite --mode ablation
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import traceback

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.swebench.dataset_loader import SWEBenchDataset
from src.swebench.problem_parser import SWEBenchProblemParser
from src.swebench.patch_generator import SWEBenchPatchGenerator
from src.swebench.evaluator import SWEBenchEvaluator
from src.swebench.metrics import SWEBenchMetrics
from src.pipeline.pipeline_orchestrator import PipelineFactory
from loguru import logger


class SWEBenchExperimentRunner:
    """Runs CGARF experiments on SWE-Bench."""
    
    def __init__(self,
                 output_dir: str = './swe-bench-results',
                 llm_provider: str = 'openai',
                 model: str = 'gpt-4',
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 subset: str = 'lite',
                 split: str = 'test',
                 max_problems: Optional[int] = None,
                 sample_seed: int = 42,
                 resume_from: Optional[str] = None,
                 mode: str = 'full',
                 debug: bool = False):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory for results
            llm_provider: LLM provider (openai, qwen, mock)
            model: Model name
            api_key: API key for LLM provider
            api_base: Custom API endpoint
            subset: Dataset subset (lite, verified, full)
            split: Dataset split (test, train)
            max_problems: Max problems to evaluate (None = all)
            sample_seed: Random seed for sampling
            resume_from: Resume from checkpoint
            mode: Experiment mode (full, ablation, sampling)
            debug: Enable debug logging
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(str(log_file), level="DEBUG" if debug else "INFO")
        
        self.llm_provider = llm_provider
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.subset = subset
        self.split = split
        self.max_problems = max_problems
        self.sample_seed = sample_seed
        self.resume_from = resume_from
        self.mode = mode
        
        # Initialize components
        logger.info("Initializing CGARF pipeline...")
        self.pipeline = PipelineFactory.create(
            llm_provider=llm_provider,
            model=model,
            api_key=api_key,
            api_base=api_base
        )
        
        self.dataset = SWEBenchDataset(
            split=split,
            subset=subset,
            cache_dir=str(self.output_dir / 'cache')
        )
        
        self.patch_generator = SWEBenchPatchGenerator(self.pipeline)
        self.evaluator = SWEBenchEvaluator()
        self.metrics = SWEBenchMetrics()
        
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"LLM: {llm_provider}/{model}")
        logger.info(f"Dataset: {subset} ({split})")
        logger.info(f"Mode: {mode}")
    
    def run(self) -> Dict[str, Any]:
        """Run experiment."""
        
        if self.mode == 'full':
            return self.run_full_experiment()
        elif self.mode == 'ablation':
            return self.run_ablation_study()
        elif self.mode == 'sample':
            return self.run_sample_experiment()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """Run full experiment on dataset."""
        
        logger.info("Starting full experiment...")
        
        # Load dataset
        logger.info("Loading dataset...")
        self.dataset.load()
        
        # Get problems
        problems = []
        for problem in self.dataset:
            problems.append(problem)
            if self.max_problems and len(problems) >= self.max_problems:
                break
        
        logger.info(f"Loaded {len(problems)} problems")
        
        # Save problem list
        problem_ids = [p['instance_id'] for p in problems]
        with open(self.output_dir / f"problems_{self.experiment_id}.json", 'w') as f:
            json.dump(problem_ids, f, indent=2)
        
        # Process problems
        start_time = time.time()
        
        for idx, problem in enumerate(problems):
            problem_id = problem['instance_id']
            
            try:
                logger.info(f"[{idx+1}/{len(problems)}] Processing {problem_id}...")
                
                problem_start = time.time()
                
                # Generate patch
                patch = self.patch_generator.generate_patch(problem, timeout=300)
                
                problem_time = time.time() - problem_start
                
                if patch:
                    # Verify patch
                    result = self.evaluator.verify_patch(
                        instance_id=problem_id,
                        generated_patch=patch.patch_content,
                        gold_patch=problem.get('gold_patch', ''),
                        timeout=60
                    )
                    
                    result.time_seconds = problem_time
                    
                    # Save result
                    self.results.append(result.to_dict())
                    
                    logger.info(f"  ✓ Resolved: {result.resolved} "
                               f"(similarity: {result.similarity_score:.3f}, time: {problem_time:.1f}s)")
                else:
                    logger.warning(f"  ✗ No patch generated")
                    self.results.append({
                        'instance_id': problem_id,
                        'resolved': False,
                        'test_passed': False,
                        'similarity_score': 0.0,
                        'time_seconds': problem_time,
                        'error': 'No patch generated'
                    })
                
            except Exception as e:
                logger.error(f"Error processing {problem_id}: {e}")
                traceback.print_exc()
                self.results.append({
                    'instance_id': problem_id,
                    'resolved': False,
                    'test_passed': False,
                    'similarity_score': 0.0,
                    'error': str(e)
                })
        
        elapsed = time.time() - start_time
        
        # Analyze results
        logger.info("Computing metrics...")
        metrics = self._compute_metrics()
        
        # Save results
        logger.info("Saving results...")
        self._save_results(metrics, elapsed)
        
        logger.info("\n" + self._format_summary(metrics, elapsed))
        
        return {
            'metrics': metrics,
            'elapsed_seconds': elapsed,
            'results': self.results
        }
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """Run ablation study."""
        
        logger.info("Starting ablation study...")
        
        # For now, just log that we would run ablation
        # Actual implementation would create variants with components disabled
        
        logger.error("Ablation study not yet implemented")
        
        return {}
    
    def run_sample_experiment(self) -> Dict[str, Any]:
        """Run on a small sample."""
        
        logger.info("Starting sample experiment...")
        
        # Load dataset
        self.dataset.load()
        
        # Sample problems
        sample_size = self.max_problems or 10
        problems = self.dataset.sample(sample_size, seed=self.sample_seed)
        
        logger.info(f"Sampled {len(problems)} problems")
        
        # Simplified version of full experiment
        # (same code as full_experiment but with smaller dataset)
        
        return self.run_full_experiment()
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        
        if not self.results:
            return {
                'total': 0,
                'resolved': 0,
                'resolved_percent': 0.0,
                'apply_percent': 0.0,
                'avg_similarity': 0.0
            }
        
        total = len(self.results)
        resolved = sum(1 for r in self.results if r.get('resolved', False))
        passed = sum(1 for r in self.results if r.get('test_passed', False))
        avg_similarity = sum(r.get('similarity_score', 0) for r in self.results) / total
        avg_time = sum(r.get('time_seconds', 0) for r in self.results) / total
        
        return {
            'total': total,
            'resolved': resolved,
            'resolved_percent': 100 * resolved / total,
            'apply_percent': 100 * resolved / total,  # Simplified
            'passed': passed,
            'passed_percent': 100 * passed / total,
            'avg_similarity': avg_similarity,
            'avg_time_per_problem': avg_time,
            'total_time': sum(r.get('time_seconds', 0) for r in self.results),
            'failures': sum(1 for r in self.results if r.get('error'))
        }
    
    def _save_results(self,
                     metrics: Dict[str, Any],
                     elapsed: float) -> None:
        """Save experiment results."""
        
        # Save raw results
        results_file = self.output_dir / f"results_{self.experiment_id}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'metadata': {
                    'experiment_id': self.experiment_id,
                    'timestamp': datetime.now().isoformat(),
                    'llm_provider': self.llm_provider,
                    'model': self.model,
                    'subset': self.subset,
                    'elapsed_seconds': elapsed
                },
                'metrics': metrics,
                'results': self.results
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def _format_summary(self,
                       metrics: Dict[str, Any],
                       elapsed: float) -> str:
        """Format summary output."""
        
        return f"""
╔═══════════════════════════════════════════════════════════════╗
║            CGARF SWE-Bench Experiment Results                 ║
╠═══════════════════════════════════════════════════════════════╣
║ Total Problems        : {metrics['total']:>5d}                              ║
║ Resolved              : {metrics['resolved']:>5d} ({metrics['resolved_percent']:>5.1f}%)                    ║
║ Test Passed           : {metrics['passed']:>5d} ({metrics['passed_percent']:>5.1f}%)                    ║
║ Failures              : {metrics['failures']:>5d}                              ║
╠═══════════════════════════════════════════════════════════════╣
║ Apply Rate            : {metrics['apply_percent']:>5.1f}%                             ║
║ Avg Similarity        : {metrics['avg_similarity']:>7.3f}                             ║
║ Avg Time/Problem      : {metrics['avg_time_per_problem']:>7.1f}s                             ║
║ Total Time            : {elapsed:>7.1f}s                             ║
╚═══════════════════════════════════════════════════════════════╝
        """


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description='Run CGARF experiments on SWE-Bench',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 10 problems
  %(prog)s --subset lite --max-problems 10

  # Use Aliyun Qwen (百炼)
  %(prog)s --subset lite --llm-provider qwen --model qwen3-coder-plus \\
           --api-key <your_api_key> --max-problems 10

  # Full evaluation with OpenAI
  %(prog)s --subset lite --output-dir ./results

  # Using mock LLM for testing
  %(prog)s --subset lite --llm-provider mock --max-problems 5
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./swe-bench-results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['openai', 'qwen', 'mock'],
        default='openai',
        help='LLM provider (openai, qwen, mock)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4',
        help='Model name (e.g., gpt-4, qwen3-coder-plus)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key for LLM provider'
    )
    
    parser.add_argument(
        '--api-base',
        type=str,
        default=None,
        help='Custom API endpoint (e.g., for Aliyun DashScope)'
    )
    
    parser.add_argument(
        '--subset',
        type=str,
        choices=['lite', 'verified', 'full'],
        default='lite',
        help='SWE-Bench subset'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['test', 'train'],
        default='test',
        help='Dataset split'
    )
    
    parser.add_argument(
        '--max-problems',
        type=int,
        default=None,
        help='Maximum number of problems to evaluate'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'ablation', 'sample'],
        default='full',
        help='Experiment mode'
    )
    
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Create and run experiment
    runner = SWEBenchExperimentRunner(
        output_dir=args.output_dir,
        llm_provider=args.llm_provider,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        subset=args.subset,
        split=args.split,
        max_problems=args.max_problems,
        sample_seed=args.seed,
        resume_from=args.resume_from,
        mode=args.mode,
        debug=args.debug
    )
    
    result = runner.run()
    
    return 0 if result else 1


if __name__ == '__main__':
    sys.exit(main())
