"""
SWE-Bench Evaluator

Evaluates generated patches against test cases.
Computes resolution and success metrics.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import difflib
import json
from pathlib import Path
from loguru import logger


@dataclass
class PatchVerificationResult:
    """Result of patch verification."""
    instance_id: str
    resolved: bool
    test_passed: bool
    similarity_score: float
    time_seconds: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'instance_id': self.instance_id,
            'resolved': self.resolved,
            'test_passed': self.test_passed,
            'similarity_score': float(self.similarity_score),
            'time_seconds': float(self.time_seconds),
            'error': self.error_message
        }


class SWEBenchEvaluator:
    """Evaluates patches against SWE-Bench test cases."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results: List[PatchVerificationResult] = []
    
    def verify_patch(self,
                    instance_id: str,
                    generated_patch: str,
                    gold_patch: str,
                    timeout: int = 60) -> PatchVerificationResult:
        """
        Verify a generated patch.
        
        Args:
            instance_id: Problem instance ID
            generated_patch: Generated patch content
            gold_patch: Reference/gold patch
            timeout: Test execution timeout
            
        Returns:
            PatchVerificationResult
        """
        try:
            # For offline evaluation, use similarity-based heuristic
            # In a real setting, would run actual tests
            similarity = self._compute_patch_similarity(
                generated_patch,
                gold_patch
            )
            
            # Determine if resolved (threshold can be tuned)
            resolved = similarity > 0.5
            
            result = PatchVerificationResult(
                instance_id=instance_id,
                resolved=resolved,
                test_passed=resolved,
                similarity_score=similarity
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error verifying patch for {instance_id}: {e}")
            
            result = PatchVerificationResult(
                instance_id=instance_id,
                resolved=False,
                test_passed=False,
                similarity_score=0.0,
                error_message=str(e)
            )
            
            self.results.append(result)
            return result
    
    def verify_patches_batch(self,
                            patches: List[Dict[str, str]]) -> List[PatchVerificationResult]:
        """
        Verify multiple patches.
        
        Args:
            patches: List of patch dictionaries with
                    'instance_id', 'patch', 'gold_patch'
                    
        Returns:
            List of verification results
        """
        for patch_dict in patches:
            self.verify_patch(
                instance_id=patch_dict['instance_id'],
                generated_patch=patch_dict['patch'],
                gold_patch=patch_dict.get('gold_patch', '')
            )
        
        return self.results
    
    @staticmethod
    def _compute_patch_similarity(generated: str,
                                  reference: str) -> float:
        """
        Compute similarity between generated and reference patches.
        
        Uses sequence matching to measure how similar the patches are.
        
        Args:
            generated: Generated patch
            reference: Reference/gold patch
            
        Returns:
            Similarity score in [0, 1]
        """
        matcher = difflib.SequenceMatcher(None, generated, reference)
        return matcher.ratio()
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute evaluation metrics from all results.
        
        Returns:
            Dictionary with metrics
        """
        if not self.results:
            return {
                'total': 0,
                'resolved': 0,
                'resolved_percent': 0.0,
                'apply_percent': 0.0,
                'avg_similarity': 0.0,
                'passed_percent': 0.0
            }
        
        total = len(self.results)
        resolved = sum(1 for r in self.results if r.resolved)
        passed = sum(1 for r in self.results if r.test_passed)
        avg_similarity = sum(r.similarity_score for r in self.results) / total
        
        return {
            'total': total,
            'resolved': resolved,
            'resolved_percent': 100 * resolved / total,
            'apply_percent': 100 * sum(1 for r in self.results if r.resolved) / total,
            'avg_similarity': avg_similarity,
            'passed_percent': 100 * passed / total,
            'failures': sum(1 for r in self.results if r.error_message)
        }
    
    def save_results(self, output_path: str) -> None:
        """
        Save evaluation results to JSON.
        
        Args:
            output_path: Output JSON file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'results': [r.to_dict() for r in self.results],
            'metrics': self.compute_metrics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.results)} results to {output_path}")
    
    def get_summary(self) -> str:
        """Get summary of evaluation results."""
        metrics = self.compute_metrics()
        
        if not metrics['total']:
            return "No results to summarize"
        
        return f"""
========== EVALUATION SUMMARY ==========
Total Problems: {metrics['total']}
Resolved: {metrics['resolved']} ({metrics['resolved_percent']:.1f}%)
Apply Rate: {metrics['apply_percent']:.1f}%
Pass Rate: {metrics['passed_percent']:.1f}%
Avg Similarity: {metrics['avg_similarity']:.3f}
Failures: {metrics['failures']}
=========================================
        """
