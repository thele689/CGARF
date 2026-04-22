"""Multi-Agent Debate System for CRG - Support/Oppose/Judge agents"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from loguru import logger

from src.common.llm_interface import LLMInterface, AgentType
from src.common.data_structures import PathEvidence


@dataclass
class DebateRound:
    """Single round of debate"""
    round_num: int
    support_arg: Dict
    oppose_arg: Dict
    judge_result: Dict
    support_wins: bool


@dataclass
class DebateResult:
    """Result of debate between two items"""
    item1: str
    item2: str
    rounds: List[DebateRound]
    winner: str
    winner_idx: int
    win_rate: float
    total_rounds: int


class AgentManager:
    """Manages multi-agent debate system"""
    
    def __init__(self, llm: LLMInterface, max_debate_rounds: int = 5,
                 convergence_threshold: int = 3):
        """
        Initialize Agent Manager
        
        Args:
            llm: LLM interface for agent invocation
            max_debate_rounds: Maximum rounds per debate
            convergence_threshold: Consecutive wins to declare convergence
        """
        self.llm = llm
        self.max_debate_rounds = max_debate_rounds
        self.convergence_threshold = convergence_threshold
        self.logger = logger
        self.debate_history = []
    
    def debate_items(self, item1: str, item2: str, issue_context: str,
                    debate_type: str = "path") -> DebateResult:
        """
        Run multi-round debate between two items
        
        Args:
            item1: First item (e.g., path string)
            item2: Second item to compare
            issue_context: Context (issue description)
            debate_type: Type of debate ("path", "location", etc.)
        
        Returns:
            DebateResult with winner and statistics
        """
        
        self.logger.info(f"Starting debate between items")
        
        rounds = []
        support_wins = 0
        
        for round_num in range(self.max_debate_rounds):
            # Support Agent - argues for item1
            support_arg = self._get_support_argument(
                item1, item2, issue_context
            )
            
            # Oppose Agent - argues for item2
            oppose_arg = self._get_oppose_argument(
                item1, item2, issue_context
            )
            
            # Judge Agent - decides
            judge_result = self._get_judge_decision(
                item1, item2, support_arg, oppose_arg, issue_context
            )
            
            # Track winner
            judge_winner = judge_result.get('winner', 'item1')
            item1_won = (judge_winner == 'item1')
            
            if item1_won:
                support_wins += 1
            
            # Create round record
            round_record = DebateRound(
                round_num=round_num,
                support_arg=support_arg,
                oppose_arg=oppose_arg,
                judge_result=judge_result,
                support_wins=item1_won
            )
            rounds.append(round_record)
            
            # Check for early convergence
            if self._is_converged(support_wins, round_num + 1):
                self.logger.debug(f"Debate converged at round {round_num + 1}")
                break
        
        # Determine final winner
        total_rounds = len(rounds)
        win_rate = support_wins / total_rounds if total_rounds > 0 else 0.5
        
        if support_wins > total_rounds / 2:
            winner = item1
            winner_idx = 0
        else:
            winner = item2
            winner_idx = 1
        
        result = DebateResult(
            item1=item1,
            item2=item2,
            rounds=rounds,
            winner=winner,
            winner_idx=winner_idx,
            win_rate=win_rate,
            total_rounds=total_rounds
        )
        
        self.debate_history.append(result)
        
        return result
    
    def _get_support_argument(self, item1: str, item2: str,
                            context: str) -> Dict:
        """Get support argument for item1"""
        
        prompt = f"""As a SUPPORT agent, argue why item1 is more likely than item2.

Context: {context}

Item 1: {item1}
Item 2: {item2}

Provide a structured argument with:
- claim: Main argument
- key_points: List of supporting points
- evidence: Specific evidence from the items
- reasoning: How item1 explains the failure

Return JSON format."""
        
        try:
            response = self.llm.generate(prompt, temperature=0.7)
            
            # Try to parse JSON
            try:
                arg = json.loads(response)
            except:
                # Extract key information
                arg = {
                    'claim': response.split('\n')[0][:100],
                    'key_points': [response],
                    'evidence': '',
                    'reasoning': response
                }
            
            return arg
        
        except Exception as e:
            self.logger.error(f"Support argument generation failed: {e}")
            return {
                'claim': f"Item1 seems more directly related",
                'key_points': [],
                'evidence': '',
                'reasoning': 'Failed to generate detailed argument'
            }
    
    def _get_oppose_argument(self, item1: str, item2: str,
                            context: str) -> Dict:
        """Get opposition argument against item1"""
        
        prompt = f"""As an OPPOSE agent, argue why item1 is less likely than item2.

Context: {context}

Item 1: {item1}
Item 2: {item2}

Identify weaknesses in item1 and strengths in item2:
- counter_claim: Refute item1
- weak_points: Weaknesses of item1
- strong_points: Strengths of item2
- reasoning: Why item2 is better

Return JSON format."""
        
        try:
            response = self.llm.generate(prompt, temperature=0.7)
            
            # Try to parse JSON
            try:
                arg = json.loads(response)
            except:
                arg = {
                    'counter_claim': response.split('\n')[0][:100],
                    'weak_points': [],
                    'strong_points': [],
                    'reasoning': response
                }
            
            return arg
        
        except Exception as e:
            self.logger.error(f"Oppose argument generation failed: {e}")
            return {
                'counter_claim': f"Item2 seems more relevant",
                'weak_points': [],
                'strong_points': [],
                'reasoning': 'Failed to generate detailed argument'
            }
    
    def _get_judge_decision(self, item1: str, item2: str,
                           support_arg: Dict, oppose_arg: Dict,
                           context: str) -> Dict:
        """Get judge decision on debate"""
        
        prompt = f"""As a JUDGE agent, evaluate which item is more likely.

Context: {context}

Item 1: {item1}
Support Argument: {str(support_arg)[:500]}

Item 2: {item2}
Opposition Argument: {str(oppose_arg)[:500]}

Evaluate based on:
1. Argument quality and coherence
2. Evidence strength
3. Relevance to context
4. Logical soundness

Decide: Which item (item1 or item2) is more likely?

Return JSON with:
{{"winner": "item1" or "item2", "confidence": 0.0-1.0, "key_reasons": [...]}}"""
        
        try:
            response = self.llm.generate(prompt, temperature=0.5)
            
            try:
                decision = json.loads(response)
            except:
                # Default decision based on text
                if "item1" in response.lower() and "item2" not in response.lower():
                    decision = {
                        'winner': 'item1',
                        'confidence': 0.6,
                        'key_reasons': [response[:100]]
                    }
                else:
                    decision = {
                        'winner': 'item2',
                        'confidence': 0.6,
                        'key_reasons': [response[:100]]
                    }
            
            return decision
        
        except Exception as e:
            self.logger.error(f"Judge decision generation failed: {e}")
            return {
                'winner': 'item1',
                'confidence': 0.5,
                'key_reasons': ['Failed to generate decision']
            }
    
    def _is_converged(self, support_wins: int, total_rounds: int) -> bool:
        """Check if debate has converged"""
        
        if total_rounds < self.convergence_threshold:
            return False
        
        # Check for convergence threshold wins in a row
        # Simple heuristic: if one side has more than threshold advantage
        losing_rounds = total_rounds - support_wins
        
        if support_wins >= self.convergence_threshold:
            return True
        
        if losing_rounds >= self.convergence_threshold:
            return True
        
        return False
    
    def debate_with_context(self, path1: PathEvidence, path2: PathEvidence,
                          issue_description: str) -> Tuple[PathEvidence, float]:
        """
        Debate specifically for PathEvidence objects
        
        Args:
            path1: First path
            path2: Second path
            issue_description: Issue description
        
        Returns:
            Tuple of (winner_path, confidence)
        """
        
        from src.crg.path_processing import PathProcessor
        
        processor = PathProcessor(self.llm)
        path1_str = processor.compress_path(path1)
        path2_str = processor.compress_path(path2)
        
        # Run debate
        result = self.debate_items(path1_str, path2_str, issue_description)
        
        winner_path = path1 if result.winner_idx == 0 else path2
        
        return winner_path, result.win_rate
    
    def get_debate_statistics(self) -> Dict:
        """Get statistics about all debates"""
        
        if not self.debate_history:
            return {
                'total_debates': 0,
                'avg_rounds': 0,
                'avg_win_rate': 0
            }
        
        total_debates = len(self.debate_history)
        total_rounds = sum(d.total_rounds for d in self.debate_history)
        total_win_rate = sum(d.win_rate for d in self.debate_history)
        
        return {
            'total_debates': total_debates,
            'avg_rounds': total_rounds / total_debates,
            'avg_win_rate': total_win_rate / total_debates,
            'total_rounds_sum': total_rounds
        }


class PathDebateOrchestrator:
    """Orchestrates path-level and location-level debates"""
    
    def __init__(self, agent_manager: AgentManager):
        """
        Initialize Path Debate Orchestrator
        
        Args:
            agent_manager: AgentManager instance
        """
        self.agent_manager = agent_manager
        self.logger = logger
    
    def debate_path_sets(self, path_sets: Dict[str, List[PathEvidence]],
                        issue_desc: str) -> Dict[str, PathEvidence]:
        """
        Debate paths within each candidate location
        
        Args:
            path_sets: Dict mapping candidate -> list of paths
            issue_desc: Issue description
        
        Returns:
            Dict mapping candidate -> representative path
        """
        
        representative_paths = {}
        
        for candidate, paths in path_sets.items():
            if not paths:
                continue
            
            if len(paths) == 1:
                representative_paths[candidate] = paths[0]
            else:
                from src.crg.path_processing import PathProcessor
                
                processor = PathProcessor(self.agent_manager.llm)
                winner = processor.debate_path_pair(
                    paths[0], paths[1], issue_desc
                )['winner']
                
                representative_paths[candidate] = winner
        
        return representative_paths
    
    def debate_locations(self, location_representatives: Dict[str, PathEvidence],
                        issue_desc: str) -> str:
        """
        Debate across candidate locations
        
        Args:
            location_representatives: Dict mapping candidate -> representative path
            issue_desc: Issue description
        
        Returns:
            Top candidate location
        """
        
        candidates = list(location_representatives.keys())
        
        if not candidates:
            return ""
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Pairwise debates
        remaining = candidates.copy()
        
        while len(remaining) > 1:
            next_round = []
            
            for i in range(0, len(remaining), 2):
                if i + 1 < len(remaining):
                    path1 = location_representatives[remaining[i]]
                    path2 = location_representatives[remaining[i + 1]]
                    
                    winner_path, _ = self.agent_manager.debate_with_context(
                        path1, path2, issue_desc
                    )
                    
                    # Identify which candidate won
                    if winner_path == path1:
                        next_round.append(remaining[i])
                    else:
                        next_round.append(remaining[i + 1])
                else:
                    next_round.append(remaining[i])
            
            remaining = next_round
        
        return remaining[0] if remaining else ""
