"""
SWE-Bench Integration Module

Integrates CGARF with the SWE-Bench benchmark for automated program repair evaluation.
Supports SWE-Bench Lite and Verified datasets.
"""

from .dataset_loader import SWEBenchDataset
from .problem_parser import SWEBenchProblemParser
from .patch_generator import SWEBenchPatchGenerator
from .evaluator import SWEBenchEvaluator, PatchVerificationResult
from .metrics import SWEBenchMetrics

__all__ = [
    "SWEBenchDataset",
    "SWEBenchProblemParser",
    "SWEBenchPatchGenerator",
    "SWEBenchEvaluator",
    "PatchVerificationResult",
    "SWEBenchMetrics"
]

__version__ = "0.1.0"
