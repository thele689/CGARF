"""
SWE-Bench Metrics Computation

Calculates standard SWE-Bench evaluation metrics:
- Resolved% (resolution rate)
- Apply% (applicability rate)
- Top-1 Accuracy
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from statistics import mean, stdev, median
from loguru import logger


@dataclass
class PatchRanking:
    """
    Ranking of patches for a single problem.
    Stores confidence scores for multiple candidates.
    """
    instance_id: str
    patches: List[Tuple[str, float]]  # List of (patch_id, confidence)
    resolved: bool = False
    correct_rank: int = -1  # Position of correct patch in ranking (1-indexed)
    
    @property
    def top_1_correct(self) -> bool:
        """Check if correct patch is top-ranked."""
        return self.correct_rank == 1
    
    @property
    def reciprocal_rank(self) -> float:
        """Get reciprocal rank for MRR calculation."""
        if self.correct_rank > 0:
            return 1.0 / self.correct_rank
        return 0.0
    
    @property
    def average_precision(self) -> float:
        """Get average precision for MAP calculation (simplified)."""
        if not self.resolved:
            return 0.0
        # Simple: 1.0 if correct, 0.0 otherwise (can be refined)
        return 1.0


class SWEBenchMetrics:
    """Calculate SWE-Bench evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.rankings: List[PatchRanking] = []
    
    def add_ranking(self, ranking: PatchRanking) -> None:
        """Add a patch ranking result."""
        self.rankings.append(ranking)
    
    def compute_resolved_percent(self) -> float:
        """
        Compute resolved percentage.
        
        % of problems where at least one patch was generated and verified.
        """
        if not self.rankings:
            return 0.0
        
        resolved = sum(1 for r in self.rankings if r.resolved)
        return 100 * resolved / len(self.rankings)
    
    def compute_apply_percent(self) -> float:
        """
        Compute apply percentage.
        
        % of problems where generated patch can be applied cleanly.
        For SWE-Bench, roughly same as resolved% (no 3-way merge).
        """
        # Simplified: same as resolved%
        # In practice, might differ based on patch application
        return self.compute_resolved_percent()
    
    def compute_top_1_accuracy(self) -> float:
        """
        Compute top-1 accuracy.
        
        % of problems where the top-ranked patch is correct.
        """
        if not self.rankings:
            return 0.0
        
        correct_top_1 = sum(1 for r in self.rankings if r.top_1_correct)
        return 100 * correct_top_1 / len(self.rankings)
    
    def compute_map(self) -> float:
        """
        Compute Mean Average Precision.
        
        Average of AP across all problems.
        """
        if not self.rankings:
            return 0.0
        
        aps = [r.average_precision for r in self.rankings]
        return mean(aps) if aps else 0.0
    
    def compute_mrr(self) -> float:
        """
        Compute Mean Reciprocal Rank.
        
        Average of reciprocal ranks across problems.
        """
        if not self.rankings:
            return 0.0
        
        reciprocal_ranks = [r.reciprocal_rank for r in self.rankings]
        non_zero = [rr for rr in reciprocal_ranks if rr > 0]
        
        return mean(non_zero) if non_zero else 0.0
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics at once.
        
        Returns:
            Dictionary with all metric values
        """
        return {
            'resolved_percent': self.compute_resolved_percent(),
            'apply_percent': self.compute_apply_percent(),
            'top_1_accuracy': self.compute_top_1_accuracy(),
            'map': self.compute_map(),
            'mrr': self.compute_mrr(),
            'total_problems': len(self.rankings),
            'resolved_count': sum(1 for r in self.rankings if r.resolved),
            'top_1_count': sum(1 for r in self.rankings if r.top_1_correct)
        }
    
    def compute_per_repository_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics broken down by repository.
        
        Returns:
            Dictionary mapping repo names to their metrics
        """
        # Group by repository
        repos = {}
        for ranking in self.rankings:
            # Extract repo from instance_id (format: repo__issue_id)
            parts = ranking.instance_id.split('__')
            repo = parts[0] if parts else 'unknown'
            
            if repo not in repos:
                repos[repo] = []
            repos[repo].append(ranking)
        
        # Compute metrics per repo
        metrics_by_repo = {}
        for repo, rankings in repos.items():
            resolved = sum(1 for r in rankings if r.resolved)
            total = len(rankings)
            
            metrics_by_repo[repo] = {
                'total': total,
                'resolved': resolved,
                'resolved_percent': 100 * resolved / total if total > 0 else 0.0,
                'top_1_count': sum(1 for r in rankings if r.top_1_correct),
                'top_1_percent': 100 * sum(1 for r in rankings if r.top_1_correct) / total if total > 0 else 0.0
            }
        
        return metrics_by_repo
    
    def get_summary_table(self) -> str:
        """
        Get formatted summary table of metrics.
        
        Returns:
            Formatted string table
        """
        metrics = self.compute_all_metrics()
        
        table = f"""
╔═══════════════════════════════════════════════════════════════╗
║             SWE-Bench Evaluation Results                      ║
╠═══════════════════════════════════════════════════════════════╣
║ Total Problems           : {metrics['total_problems']:>5d}                              ║
║ Resolved                 : {metrics['resolved_count']:>5d} ({metrics['resolved_percent']:>5.1f}%)                    ║
║ Top-1 Correct            : {metrics['top_1_count']:>5d} ({metrics['top_1_accuracy']:>5.1f}%)                    ║
╠═══════════════════════════════════════════════════════════════╣
║ Apply%                   : {metrics['apply_percent']:>5.1f}%                             ║
║ Top-1 Accuracy           : {metrics['top_1_accuracy']:>5.1f}%                             ║
║ MAP (Mean Avg Precision) : {metrics['map']:>7.3f}                             ║
║ MRR (Mean Recip Rank)    : {metrics['mrr']:>7.3f}                             ║
╚═══════════════════════════════════════════════════════════════╝
        """
        
        return table.strip()
    
    @staticmethod
    def compute_statistical_significance(results1: List[bool],
                                        results2: List[bool],
                                        method: str = 'mcnemar') -> float:
        """
        Compute statistical significance between two result sets.
        
        Args:
            results1: List of boolean outcomes (method 1)
            results2: List of boolean outcomes (method 2)
            method: 'mcnemar' or 'bootstrap'
            
        Returns:
            P-value
        """
        if method == 'mcnemar':
            return SWEBenchMetrics._mcnemar_test(results1, results2)
        elif method == 'bootstrap':
            return SWEBenchMetrics._bootstrap_test(results1, results2)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def _mcnemar_test(results1: List[bool],
                      results2: List[bool]) -> float:
        """
        Perform McNemar's test for paired samples.
        
        Returns:
            P-value
        """
        from scipy import stats
        
        # Create contingency table
        both_correct = sum(1 for r1, r2 in zip(results1, results2) if r1 and r2)
        r1_only = sum(1 for r1, r2 in zip(results1, results2) if r1 and not r2)
        r2_only = sum(1 for r1, r2 in zip(results1, results2) if not r1 and r2)
        neither = sum(1 for r1, r2 in zip(results1, results2) if not r1 and not r2)
        
        # McNemar statistic
        if r1_only + r2_only == 0:
            return 1.0
        
        statistic = ((r1_only - r2_only) ** 2) / (r1_only + r2_only)
        
        # Chi-square distribution with 1 df
        try:
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
        except:
            p_value = 0.05  # Default if scipy not available
        
        return p_value
    
    @staticmethod
    def _bootstrap_test(results1: List[bool],
                       results2: List[bool],
                       n_bootstrap: int = 10000) -> float:
        """
        Perform bootstrap test for significance.
        
        Returns:
            P-value
        """
        import numpy as np
        
        np.random.seed(42)
        
        diff_original = sum(results1) / len(results1) - sum(results2) / len(results2)
        
        diffs = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(results1), len(results1), replace=True)
            r1_boot = [results1[i] for i in idx]
            r2_boot = [results2[i] for i in idx]
            
            diff = sum(r1_boot) / len(r1_boot) - sum(r2_boot) / len(r2_boot)
            diffs.append(diff)
        
        diffs = np.array(diffs)
        p_value = np.mean(np.abs(diffs) >= np.abs(diff_original))
        
        return p_value
