"""Path Processing for CRG - Path compression, credibility calculation, and selection"""

import math
from typing import List, Dict, Tuple, Optional
from loguru import logger

from src.common.data_structures import PathEvidence, CRGNode, CRGEdge
from src.common.llm_interface import LLMInterface


class PathProcessor:
    """Process and evaluate paths in CRG"""
    
    def __init__(self, llm: LLMInterface, 
                 length_penalty: float = 0.1):
        """
        Initialize Path Processor
        
        Args:
            llm: LLM interface for semantic analysis
            length_penalty: Lambda parameter for path length penalty
        """
        self.llm = llm
        self.length_penalty = length_penalty
        self.logger = logger
    
    def compress_path(self, path: PathEvidence) -> str:
        """
        Compress path to human-readable chain format
        
        Format: Leaf(l_i) → n1{summary}[c=0.7] → n2{summary}[c=0.8] → ... → Root(anchor)
        
        Args:
            path: PathEvidence to compress
        
        Returns:
            Readable chain representation
        """
        
        if not path.nodes:
            return "Empty path"
        
        result = f"Leaf({path.nodes[0].entity_id})"
        
        # Add intermediate nodes with summaries and strengths
        for i in range(len(path.nodes) - 1):
            node = path.nodes[i]
            strength = "?"
            
            if i < len(path.edges):
                strength = f"{path.edges[i].strength:.1f}"
            
            # Get semantic summary (truncate if too long)
            summary = node.semantic_summary or "?"
            summary = summary[:50]  # Truncate to 50 chars
            
            result += f" → {node.entity_id}{{{summary}}}[c={strength}]"
        
        # Add root node
        result += f" → Root({path.failure_anchor})"
        
        return result
    
    def calculate_path_credibility(self, path: PathEvidence) -> float:
        """
        Calculate path credibility
        
        Credibility(π) = (∏ c_e)^(1/len) × exp(-λ × len)
        
        Args:
            path: PathEvidence to evaluate
        
        Returns:
            Credibility score [0, 1]
        """
        
        if not path.edges:
            return 0.0
        
        # Compute geometric mean of edge strengths
        edge_strengths = [e.strength for e in path.edges]
        
        if not edge_strengths or any(s <= 0 for s in edge_strengths):
            geometric_mean = 0.5  # Default if invalid
        else:
            product = 1.0
            for strength in edge_strengths:
                product *= strength
            geometric_mean = product ** (1.0 / len(edge_strengths))
        
        geometric_mean = min(max(geometric_mean, 0.0), 1.0)  # Clamp to [0, 1]
        
        # Apply length penalty
        path_length = len(path.nodes)
        length_penalty_factor = math.exp(-self.length_penalty * path_length)
        
        credibility = geometric_mean * length_penalty_factor
        
        return float(credibility)
    
    def select_representative_path(self, paths: List[PathEvidence],
                                  credibilities: List[float]) -> PathEvidence:
        """
        Select representative path with highest credibility
        
        Args:
            paths: List of PathEvidence to select from
            credibilities: Credibility scores for each path
        
        Returns:
            PathEvidence with highest credibility
        """
        
        if not paths:
            return PathEvidence()
        
        if len(paths) != len(credibilities):
            self.logger.warning("Paths and credibilities length mismatch")
            idx = 0
        else:
            # Find index of max credibility
            idx = credibilities.index(max(credibilities))
        
        return paths[idx]
    
    def rank_paths(self, paths: List[PathEvidence]) -> List[Tuple[PathEvidence, float]]:
        """
        Rank paths by credibility
        
        Args:
            paths: List of PathEvidence to rank
        
        Returns:
            List of (PathEvidence, credibility) tuples sorted by credibility
        """
        
        ranked = []
        
        for path in paths:
            cred = self.calculate_path_credibility(path)
            ranked.append((path, cred))
        
        # Sort by credibility descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def debate_path_pair(self, path1: PathEvidence, path2: PathEvidence,
                        issue_desc: str, debate_rounds: int = 5) -> Dict:
        """
        Debate two paths using multi-agent debate
        
        This is a simplified version. Full implementation will use AgentManager.
        
        Args:
            path1: First path to debate
            path2: Second path to debate
            issue_desc: Issue description for context
            debate_rounds: Number of debate rounds
        
        Returns:
            Dict with winner and statistics
        """
        
        # Format paths as readable strings for debate
        path1_str = self.compress_path(path1)
        path2_str = self.compress_path(path2)
        
        # Question: which path better explains the failure?
        question = (
            f"Issue: {issue_desc}\n\n"
            f"Path 1: {path1_str}\n"
            f"Path 2: {path2_str}\n\n"
            "Which path is more likely to be the root cause?"
        )
        
        try:
            # Use LLM to compare paths
            result = self.llm.compare_relative(
                [path1_str, path2_str],
                "Which causality path better explains the failure?"
            )
            
            winner_idx = result.get('winner_idx', 0)
            confidence = result.get('confidence', 0.5)
            
            winner_path = path1 if winner_idx == 0 else path2
            
            return {
                'winner': winner_path,
                'winner_idx': winner_idx,
                'confidence': confidence,
                'debate_log': result.get('reasoning', '')
            }
        
        except Exception as e:
            self.logger.error(f"Path debate failed: {e}")
            
            # Fallback: compare by credibility
            cred1 = self.calculate_path_credibility(path1)
            cred2 = self.calculate_path_credibility(path2)
            
            winner_path = path1 if cred1 >= cred2 else path2
            
            return {
                'winner': winner_path,
                'winner_idx': 0 if cred1 >= cred2 else 1,
                'confidence': max(cred1, cred2),
                'debate_log': 'Fallback: credibility-based selection'
            }
    
    def compute_path_statistics(self, paths: List[PathEvidence]) -> Dict:
        """
        Compute statistics about paths
        
        Args:
            paths: List of paths
        
        Returns:
            Dict with path statistics
        """
        
        if not paths:
            return {
                'total_paths': 0,
                'avg_length': 0,
                'avg_credibility': 0,
                'max_credibility': 0
            }
        
        lengths = [len(p.nodes) for p in paths]
        credibilities = [self.calculate_path_credibility(p) for p in paths]
        
        return {
            'total_paths': len(paths),
            'avg_length': sum(lengths) / len(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths),
            'avg_credibility': sum(credibilities) / len(credibilities),
            'max_credibility': max(credibilities),
            'min_credibility': min(credibilities)
        }


class PathDebater:
    """Manages path debate using multi-agent system"""
    
    def __init__(self, llm: LLMInterface, max_debate_rounds: int = 5):
        """
        Initialize Path Debater
        
        Args:
            llm: LLM interface for debate
            max_debate_rounds: Maximum rounds per debate
        """
        self.llm = llm
        self.max_debate_rounds = max_debate_rounds
        self.logger = logger
    
    def debate_two_paths(self, path1: PathEvidence, path2: PathEvidence,
                        issue_desc: str) -> Tuple[PathEvidence, float]:
        """
        Debate between two paths
        
        Simpler interface that returns winner and confidence
        
        Args:
            path1: First path
            path2: Second path
            issue_desc: Issue description
        
        Returns:
            Tuple of (winner_path, confidence_score)
        """
        
        processor = PathProcessor(self.llm)
        path1_str = processor.compress_path(path1)
        path2_str = processor.compress_path(path2)
        
        # Create debate prompt
        prompt = f"""
Issue: {issue_desc}

Path 1: {path1_str}
Path 2: {path2_str}

Which of these causal paths is more likely to explain the failure?
Consider:
1. Path coherence and semantic soundness
2. Likelihood of the causal chain
3. Connection to failure symptoms

Return JSON with:
{{"winner_idx": 0 or 1, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""
        
        try:
            result = self.llm.generate(prompt, temperature=0.7)
            
            # Parse result to determine winner
            import json
            try:
                parsed = json.loads(result)
                winner_idx = parsed.get('winner_idx', 0)
                confidence = parsed.get('confidence', 0.5)
            except:
                # Fallback: use processor's compare method
                processor_result = processor.debate_path_pair(path1, path2, issue_desc)
                winner_idx = processor_result['winner_idx']
                confidence = processor_result['confidence']
            
            winner = path1 if winner_idx == 0 else path2
            return winner, confidence
        
        except Exception as e:
            self.logger.error(f"Path debate error: {e}")
            # Fallback to credibility
            processor = PathProcessor(self.llm)
            cred1 = processor.calculate_path_credibility(path1)
            cred2 = processor.calculate_path_credibility(path2)
            
            if cred1 >= cred2:
                return path1, cred1
            else:
                return path2, cred2
    
    def debate_multiple_paths(self, paths: List[PathEvidence],
                             issue_desc: str) -> PathEvidence:
        """
        Debate among multiple paths via tournament
        
        Args:
            paths: List of paths to debate
            issue_desc: Issue description
        
        Returns:
            Winner PathEvidence
        """
        
        if not paths:
            return PathEvidence()
        
        if len(paths) == 1:
            return paths[0]
        
        # Tournament style: pairwise debates
        remaining = paths.copy()
        
        while len(remaining) > 1:
            next_round = []
            
            # Pair up paths and debate
            for i in range(0, len(remaining), 2):
                if i + 1 < len(remaining):
                    winner, _ = self.debate_two_paths(
                        remaining[i],
                        remaining[i + 1],
                        issue_desc
                    )
                    next_round.append(winner)
                else:
                    # Odd one out, advance to next round
                    next_round.append(remaining[i])
            
            remaining = next_round
        
        return remaining[0] if remaining else PathEvidence()
