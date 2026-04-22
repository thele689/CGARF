"""Edge Weight Update System - Fusion and Dynamic Reranking"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from loguru import logger

from src.common.data_structures import CRGEdge, PathEvidence
from src.crg.agent_manager import DebateResult


@dataclass
class EdgeWeightUpdate:
    """Record of edge weight update"""
    edge_id: str
    old_weight: float
    new_weight: float
    update_components: Dict[str, float]  # {'initial': 0.2, 'path': 0.3, 'loc': 0.15}
    debate_rounds: int = 0


@dataclass
class CandidateRanking:
    """Ranking of candidates by credibility"""
    location: str
    initial_credibility: float
    updated_credibility: float
    rank: int
    debate_wins: int = 0
    debate_losses: int = 0
    path_preference: Optional[str] = None


class EdgeWeightManager:
    """Manages edge weight updates via debate results"""
    
    def __init__(self, eta1: float = 0.2, eta2: float = 0.4, eta3: float = 0.4):
        """
        Initialize Edge Weight Manager
        
        Args:
            eta1: Weight for initial structure-based weights (0.2)
            eta2: Weight for path-level debate wins (0.4)
            eta3: Weight for location-level debate wins (0.4)
        """
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.logger = logger
        self.weight_history = []
    
    def update_edge_weights(self, edges: Dict[str, CRGEdge],
                           path_debate_results: List[DebateResult],
                           location_debate_results: Optional[List[DebateResult]] = None
                           ) -> Dict[str, float]:
        """
        Update edge weights using fusion formula
        
        Formula: c_ji* = η1·c_ji^(0) + η2·P_path(π_ij) + η3·P_loc(l_i)
        
        Args:
            edges: Dict mapping edge_id -> CRGEdge
            path_debate_results: Results from path-level debates
            location_debate_results: Results from location-level debates (optional)
        
        Returns:
            Dict mapping edge_id -> updated_weight
        """
        
        # Calculate P_path (path-level debate win rates)
        p_path = self._compute_path_statistics(path_debate_results)
        
        # Calculate P_loc (location-level debate win rates)
        p_loc = self._compute_location_statistics(
            location_debate_results if location_debate_results else []
        )
        
        updated_weights = {}
        
        for edge_id, edge in edges.items():
            # Get component weights
            c_initial = edge.weight  # c_ji^(0)
            
            # Retrieve path and location win rates
            path_rate = p_path.get(edge_id, 0.5)
            loc_rate = p_loc.get(edge_id, 0.5)
            
            # Fusion formula
            updated_weight = (
                self.eta1 * c_initial +
                self.eta2 * path_rate +
                self.eta3 * loc_rate
            )
            
            # Clamp to [0, 1]
            updated_weight = max(0.0, min(1.0, updated_weight))
            
            updated_weights[edge_id] = updated_weight
            
            # Record update
            update = EdgeWeightUpdate(
                edge_id=edge_id,
                old_weight=c_initial,
                new_weight=updated_weight,
                update_components={
                    'initial': self.eta1 * c_initial,
                    'path': self.eta2 * path_rate,
                    'location': self.eta3 * loc_rate
                }
            )
            self.weight_history.append(update)
        
        self.logger.info(f"Updated weights for {len(updated_weights)} edges")
        
        return updated_weights
    
    def _compute_path_statistics(self, debate_results: List[DebateResult]
                                 ) -> Dict[str, float]:
        """
        Compute path-level debate statistics
        
        Args:
            debate_results: List of path-level debate results
        
        Returns:
            Dict mapping edge_id -> win_rate
        """
        
        path_stats = {}
        
        for result in debate_results:
            # Extract edge identifiers from paths if possible
            # For now, use path index as proxy
            edge_id = f"path_{len(path_stats)}"
            
            # Win rate is the proportion of rounds won
            path_stats[edge_id] = result.win_rate
        
        return path_stats
    
    def _compute_location_statistics(self, debate_results: List[DebateResult]
                                    ) -> Dict[str, float]:
        """
        Compute location-level debate statistics
        
        Args:
            debate_results: List of location-level debate results
        
        Returns:
            Dict mapping location_id -> win_rate
        """
        
        location_stats = {}
        
        for result in debate_results:
            location_id = f"loc_{len(location_stats)}"
            location_stats[location_id] = result.win_rate
        
        return location_stats
    
    def rank_candidates(self, candidates: Dict[str, float],
                       updated_weights: Dict[str, float],
                       debate_records: Optional[Dict[str, Tuple[int, int]]] = None
                       ) -> List[CandidateRanking]:
        """
        Rank candidates by updated credibility
        
        Args:
            candidates: Dict mapping location_id -> initial_credibility
            updated_weights: Dict mapping edge_id -> updated_weight
            debate_records: Dict mapping candidate -> (wins, losses)
        
        Returns:
            List of CandidateRanking sorted by updated_credibility
        """
        
        rankings = []
        
        for idx, (location, initial_cred) in enumerate(candidates.items()):
            # Compute updated credibility by averaging related edge weights
            related_edges = [w for edge_id, w in updated_weights.items()
                           if location in str(edge_id)]
            
            if related_edges:
                edge_boost = np.mean(related_edges)
            else:
                edge_boost = 0.0
            
            # Updated credibility combines initial and edge updates
            updated_cred = 0.6 * initial_cred + 0.4 * edge_boost
            
            # Add debate statistics if available
            wins, losses = debate_records.get(location, (0, 0)) if debate_records else (0, 0)
            
            ranking = CandidateRanking(
                location=location,
                initial_credibility=initial_cred,
                updated_credibility=updated_cred,
                rank=idx + 1,
                debate_wins=wins,
                debate_losses=losses
            )
            
            rankings.append(ranking)
        
        # Sort by updated credibility (descending)
        rankings.sort(key=lambda r: r.updated_credibility, reverse=True)
        
        # Update ranks
        for idx, ranking in enumerate(rankings):
            ranking.rank = idx + 1
        
        return rankings
    
    def get_top_candidates(self, rankings: List[CandidateRanking],
                          top_k: int = 3) -> List[str]:
        """
        Get top K candidates
        
        Args:
            rankings: List of ranked candidates
            top_k: Number of top candidates to return
        
        Returns:
            List of top candidate locations
        """
        
        return [r.location for r in rankings[:top_k]]


class DynamicRerankingEngine:
    """Engine for dynamic reranking of candidates"""
    
    def __init__(self, weight_manager: EdgeWeightManager):
        """
        Initialize Dynamic Reranking Engine
        
        Args:
            weight_manager: EdgeWeightManager instance
        """
        self.weight_manager = weight_manager
        self.logger = logger
    
    def rerank_with_debates(self, initial_candidates: Dict[str, float],
                           path_debates: List[DebateResult],
                           location_debates: Optional[List[DebateResult]] = None,
                           edges: Optional[Dict[str, CRGEdge]] = None
                           ) -> Tuple[List[CandidateRanking], List[str]]:
        """
        Perform complete reranking with debate results
        
        Args:
            initial_candidates: Initial credibility scores
            path_debates: Path-level debate results
            location_debates: Location-level debate results
            edges: Edge dictionary for weight updates
        
        Returns:
            Tuple of (ranked_candidates, top_3_locations)
        """
        
        # If no edges provided, create dummy edges
        if edges is None:
            edges = {
                f"edge_{i}": CRGEdge(
                    source="",
                    target="",
                    weight=0.5,
                    source_type="FUNCTION",
                    target_type="FUNCTION",
                    edge_type="REFERENCE"
                )
                for i in range(len(initial_candidates))
            }
        
        # Update edge weights
        updated_weights = self.weight_manager.update_edge_weights(
            edges, path_debates, location_debates
        )
        
        # Build debate records
        debate_records = self._extract_debate_records(
            path_debates, location_debates
        )
        
        # Rank candidates
        rankings = self.weight_manager.rank_candidates(
            initial_candidates, updated_weights, debate_records
        )
        
        # Get top candidates
        top_candidates = self.weight_manager.get_top_candidates(rankings, top_k=3)
        
        self.logger.info(f"Reranking complete: {len(rankings)} candidates")
        self.logger.debug(f"Top 3: {top_candidates}")
        
        return rankings, top_candidates
    
    def _extract_debate_records(self, path_debates: List[DebateResult],
                               location_debates: Optional[List[DebateResult]]
                               ) -> Dict[str, Tuple[int, int]]:
        """
        Extract win/loss records from debates
        
        Args:
            path_debates: Path-level debates
            location_debates: Location-level debates
        
        Returns:
            Dict mapping candidate -> (wins, losses)
        """
        
        records = {}
        
        # Process path debates
        for debate in path_debates:
            winner_id = f"cand_{debate.winner_idx}"
            
            if winner_id not in records:
                records[winner_id] = (0, 0)
            
            wins, losses = records[winner_id]
            records[winner_id] = (wins + 1, losses)
        
        # Process location debates
        if location_debates:
            for debate in location_debates:
                winner_id = f"cand_{debate.winner_idx}"
                
                if winner_id not in records:
                    records[winner_id] = (0, 0)
                
                wins, losses = records[winner_id]
                records[winner_id] = (wins + 1, losses)
        
        return records
    
    def get_reranking_statistics(self) -> Dict:
        """Get statistics about reranking operations"""
        
        if not self.weight_manager.weight_history:
            return {
                'total_updates': 0,
                'avg_weight_change': 0,
                'max_weight_change': 0
            }
        
        weight_changes = [
            abs(u.new_weight - u.old_weight)
            for u in self.weight_manager.weight_history
        ]
        
        return {
            'total_updates': len(self.weight_manager.weight_history),
            'avg_weight_change': np.mean(weight_changes),
            'max_weight_change': np.max(weight_changes),
            'min_weight_change': np.min(weight_changes)
        }


class WeightFusionStrategy:
    """Strategy for adaptive weight fusion"""
    
    def __init__(self, base_eta: Tuple[float, float, float] = (0.2, 0.4, 0.4)):
        """
        Initialize Weight Fusion Strategy
        
        Args:
            base_eta: (η1, η2, η3) base weights
        """
        self.base_eta = base_eta
        self.logger = logger
    
    def adaptive_fusion(self, c_initial: float, p_path: float, p_loc: float,
                       debate_confidence: float) -> float:
        """
        Adaptive fusion based on debate confidence
        
        Higher confidence → higher weight on debate results
        Lower confidence → reliance on initial weights
        
        Args:
            c_initial: Initial structure-based weight
            p_path: Path-level debate win rate
            p_loc: Location-level debate win rate
            debate_confidence: Confidence in debate results [0, 1]
        
        Returns:
            Fused weight
        """
        
        # Adaptive eta based on confidence
        eta1 = self.base_eta[0] * (1 - 0.5 * debate_confidence)  # Reduce initial weight
        eta2 = self.base_eta[1] * (0.5 + debate_confidence)      # Increase path weight
        eta3 = self.base_eta[2] * (0.5 + debate_confidence)      # Increase location weight
        
        # Normalize to sum to 1
        total = eta1 + eta2 + eta3
        eta1 /= total
        eta2 /= total
        eta3 /= total
        
        # Fusion
        fused = eta1 * c_initial + eta2 * p_path + eta3 * p_loc
        
        return max(0.0, min(1.0, fused))
    
    def conservative_fusion(self, c_initial: float, p_path: float, p_loc: float
                           ) -> float:
        """
        Conservative fusion - trust initial weights more
        
        Args:
            c_initial: Initial weight
            p_path: Path debate rate
            p_loc: Location debate rate
        
        Returns:
            Fused weight (weighted towards initial)
        """
        
        return 0.5 * c_initial + 0.25 * p_path + 0.25 * p_loc
    
    def aggressive_fusion(self, c_initial: float, p_path: float, p_loc: float
                         ) -> float:
        """
        Aggressive fusion - trust debate results more
        
        Args:
            c_initial: Initial weight
            p_path: Path debate rate
            p_loc: Location debate rate
        
        Returns:
            Fused weight (weighted towards debate)
        """
        
        return 0.1 * c_initial + 0.45 * p_path + 0.45 * p_loc
