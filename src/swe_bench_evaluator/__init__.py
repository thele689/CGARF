"""
Phase 4: SWE-Bench Evaluation Framework
========================================

Evaluate CGARF-generated patches using SWE-Bench's official testing framework.

Main Classes:
  - SWEBenchEvaluator: Applies patches and runs tests
  - PatchResult: Individual patch evaluation result
  - PatchEvaluationMetrics: Aggregated metrics
"""

from .evaluator import (
    SWEBenchEvaluator,
    PatchResult,
    PatchEvaluationMetrics,
)

__all__ = [
    'SWEBenchEvaluator',
    'PatchResult',
    'PatchEvaluationMetrics',
]
