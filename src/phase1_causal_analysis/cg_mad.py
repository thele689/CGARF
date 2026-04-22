"""
Phase 2: CG-MAD (paper section 3.1.2)
=====================================

This module implements the path-level and location-level multi-agent causal
debate mechanism on top of the phase1 CRG objects.

Design goals:

1. Reuse the current `CausalRelevanceGraph` built by phase1.
2. Stay aligned with paper 3.1.2:
   - path compression into structured explanation-chain summaries
   - support / oppose / judge debate loop
   - path-level win statistics P_path
   - location-level win statistics P_loc
   - edge update fusion c* = eta1*c0 + eta2*P_path + eta3*P_loc
   - credibility / representative path / final candidate ranking
3. Remain runnable even without an available LLM, using deterministic
   structural fallbacks.

The implementation keeps the paper's semantics explicit and avoids depending on
the legacy `src/crg/*` objects, which are wired to older data structures.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

from src.common.llm_interface import LLMInterface
from .causal_relevance_graph import CausalRelevanceGraph, CodeEntity, EntityType


@dataclass
class PathNodeSummary:
    entity_id: str
    display_id: str
    name: str
    entity_type: str
    file_path: str
    semantic_summary: str
    code_snippet: str = ""
    edge_weight_to_next: Optional[float] = None


@dataclass
class PathSummary:
    path_id: str
    candidate_id: str
    root_id: str
    node_ids: List[str]
    nodes: List[PathNodeSummary]
    compressed_text: str
    initial_credibility: float
    evidence_pack: Dict[str, Any]


@dataclass
class CandidateContext:
    candidate_id: str
    display_id: str
    entity_type: str
    file_path: str
    code_snippet: str


@dataclass
class FailureAnchorPack:
    anchor_id: str
    symptom_type: str
    symptom_location: str
    symptom_message: str
    evidence_refs: List[str]


@dataclass
class EvidencePack:
    issue_text: str
    failure_anchors: List[FailureAnchorPack]
    candidate_contexts: List[CandidateContext]


@dataclass
class AgentArgument:
    stance: str
    payload: Dict[str, Any]


@dataclass
class JudgeDecision:
    winner: str
    key_reasons: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)


@dataclass
class DebateRoundRecord:
    round_index: int
    support: AgentArgument
    oppose: AgentArgument
    judge: JudgeDecision


@dataclass
class PathPairDebateRecord:
    candidate_id: str
    path_a_id: str
    path_b_id: str
    rounds: List[DebateRoundRecord]
    winner_path_id: str
    converged: bool


@dataclass
class SinglePathDebateRecord:
    candidate_id: str
    path_id: str
    rounds: List[DebateRoundRecord]
    support_win_rate: float
    converged: bool


@dataclass
class CandidatePairDebateRecord:
    candidate_a_id: str
    candidate_b_id: str
    path_a_id: str
    path_b_id: str
    rounds: List[DebateRoundRecord]
    winner_candidate_id: str
    converged: bool


@dataclass
class CandidateAssessment:
    candidate_id: str
    path_win_rates: Dict[str, float]
    location_win_rate: float
    representative_path_id: str
    representative_path_credibility: float
    final_credibility: float
    judge_reason_summary: List[str] = field(default_factory=list)


@dataclass
class CGMADResult:
    evidence_pack: Dict[str, Any]
    path_summaries: List[PathSummary]
    path_pair_debates: List[PathPairDebateRecord]
    single_path_debates: List[SinglePathDebateRecord]
    candidate_pair_debates: List[CandidatePairDebateRecord]
    path_win_rates: Dict[str, float]
    location_win_rates: Dict[str, float]
    updated_edge_weights: Dict[str, float]
    candidate_assessments: List[CandidateAssessment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_pack": dict(self.evidence_pack),
            "path_summaries": [asdict(item) for item in self.path_summaries],
            "path_pair_debates": [asdict(item) for item in self.path_pair_debates],
            "single_path_debates": [asdict(item) for item in self.single_path_debates],
            "candidate_pair_debates": [asdict(item) for item in self.candidate_pair_debates],
            "path_win_rates": dict(self.path_win_rates),
            "location_win_rates": dict(self.location_win_rates),
            "updated_edge_weights": dict(self.updated_edge_weights),
            "candidate_assessments": [asdict(item) for item in self.candidate_assessments],
        }


class CGMADMechanism:
    """
    Causal Graph-augmented Multi-Agent Debate.

    This class operates over a phase1 CRG and produces:
    - path-level debate outcomes
    - location-level debate outcomes
    - updated edge weights
    - representative path / final credibility per candidate
    """

    def __init__(
        self,
        crg: CausalRelevanceGraph,
        issue_description: str,
        llm: Optional[LLMInterface] = None,
        judge_llm: Optional[LLMInterface] = None,
        summary_llm: Optional[LLMInterface] = None,
        eta: Tuple[float, float, float] = (0.2, 0.4, 0.4),
        length_penalty: float = 0.1,
        max_rounds: int = 5,
        convergence_threshold: int = 3,
        max_paths_per_candidate: Optional[int] = None,
        code_context_radius_lines: int = 8,
        prompt_code_snippet_chars: int = 320,
        prompt_chain_text_chars: int = 3000,
        prompt_max_path_nodes: int = 32,
        sampling_temperature: float = 0.1,
        random_seed: Optional[int] = 0,
    ):
        if not math.isclose(sum(eta), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("eta weights must sum to 1")

        self.crg = crg
        self.issue_description = issue_description.strip()
        self.llm = llm
        self.judge_llm = judge_llm or llm
        self.summary_llm = summary_llm or llm
        self.eta1, self.eta2, self.eta3 = eta
        self.length_penalty = length_penalty
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.max_paths_per_candidate = max_paths_per_candidate
        self.code_context_radius_lines = code_context_radius_lines
        self.prompt_code_snippet_chars = prompt_code_snippet_chars
        self.prompt_chain_text_chars = prompt_chain_text_chars
        self.prompt_max_path_nodes = prompt_max_path_nodes
        self.sampling_temperature = sampling_temperature
        self.random_seed = random_seed
        self._path_summary_cache: Dict[str, PathSummary] = {}
        self._node_summary_cache: Dict[str, str] = {}
        self._snippet_cache: Dict[str, str] = {}
        self.evidence_pack = self._build_evidence_pack()
        logger.info("Initialized CGMADMechanism")

    def run(self) -> CGMADResult:
        all_candidate_ids = sorted(self.crg.candidate_leaf_ids)
        path_summaries = self._build_path_summaries()
        path_summaries_by_candidate = self._group_path_summaries(path_summaries, all_candidate_ids)

        path_pair_debates: List[PathPairDebateRecord] = []
        single_path_debates: List[SinglePathDebateRecord] = []
        path_win_rates: Dict[str, float] = {}

        for candidate_id, summaries in path_summaries_by_candidate.items():
            if not summaries:
                continue
            if len(summaries) == 1:
                record = self._debate_single_path(summaries[0])
                single_path_debates.append(record)
                path_win_rates[summaries[0].path_id] = record.support_win_rate
            else:
                pair_records, candidate_path_rates = self._debate_path_set(candidate_id, summaries)
                path_pair_debates.extend(pair_records)
                path_win_rates.update(candidate_path_rates)

        provisional_rep_paths = self._select_provisional_representatives(
            path_summaries_by_candidate,
            path_win_rates,
        )

        candidate_pair_debates, location_win_rates = self._debate_candidates(provisional_rep_paths)
        updated_edge_weights = self._update_edge_weights(path_summaries_by_candidate, path_win_rates, location_win_rates)
        candidate_assessments = self._assess_candidates(
            all_candidate_ids,
            path_summaries_by_candidate,
            path_win_rates,
            location_win_rates,
            updated_edge_weights,
            candidate_pair_debates,
        )

        return CGMADResult(
            evidence_pack=asdict(self.evidence_pack),
            path_summaries=path_summaries,
            path_pair_debates=path_pair_debates,
            single_path_debates=single_path_debates,
            candidate_pair_debates=candidate_pair_debates,
            path_win_rates=path_win_rates,
            location_win_rates=location_win_rates,
            updated_edge_weights=updated_edge_weights,
            candidate_assessments=candidate_assessments,
        )

    def _build_path_summaries(self) -> List[PathSummary]:
        summaries: List[PathSummary] = []

        for candidate_id, raw_paths in sorted(self.crg.paths_by_candidate.items()):
            candidate_paths = raw_paths[: self.max_paths_per_candidate] if self.max_paths_per_candidate else raw_paths
            for index, path in enumerate(candidate_paths):
                path_id = f"{candidate_id}::path::{index}"
                summary = self._summarize_path(path_id, candidate_id, path)
                summaries.append(summary)
                self._path_summary_cache[path_id] = summary

        logger.info(f"Built {len(summaries)} path summaries for CG-MAD")
        return summaries

    def _build_evidence_pack(self) -> EvidencePack:
        failure_anchors: List[FailureAnchorPack] = []
        for index, evidence in enumerate(self.crg.failure_evidences):
            refs = [f"anchor:{evidence.test_case_id}"]
            if evidence.symptom_location:
                refs.append(f"location:{evidence.symptom_location}")
            failure_anchors.append(
                FailureAnchorPack(
                    anchor_id=f"anchor:{evidence.test_case_id}",
                    symptom_type=evidence.symptom_type,
                    symptom_location=evidence.symptom_location,
                    symptom_message=evidence.symptom_message,
                    evidence_refs=refs,
                )
            )

        candidate_contexts: List[CandidateContext] = []
        for candidate_id in sorted(self.crg.candidate_leaf_ids):
            entity = self._get_entity(candidate_id)
            candidate_contexts.append(
                CandidateContext(
                    candidate_id=candidate_id,
                    display_id=self._display_entity_id(entity, candidate_id),
                    entity_type=entity.entity_type.value if entity else "unknown",
                    file_path=entity.file_path if entity else "",
                    code_snippet=self._extract_code_snippet(entity),
                )
            )

        return EvidencePack(
            issue_text=self.issue_description,
            failure_anchors=failure_anchors,
            candidate_contexts=candidate_contexts,
        )

    def _build_path_evidence_pack(
        self,
        path_id: str,
        candidate_id: str,
        path: List[str],
        nodes: Sequence[PathNodeSummary],
    ) -> Dict[str, Any]:
        anchor_refs = [anchor.anchor_id for anchor in self.evidence_pack.failure_anchors]
        chain_refs: List[str] = []
        chain_refs.extend(f"node:{node.entity_id}" for node in nodes)
        for left_node, right_node in zip(nodes[:-1], nodes[1:]):
            chain_refs.append(f"edge:{left_node.entity_id}->{right_node.entity_id}")

        candidate_context = next(
            (asdict(item) for item in self.evidence_pack.candidate_contexts if item.candidate_id == candidate_id),
            None,
        )

        return {
            "path_id": path_id,
            "candidate_id": candidate_id,
            "candidate_context": candidate_context,
            "failure_anchor_refs": anchor_refs,
            "nodes": [asdict(node) for node in nodes],
            "chain_text": self._compress_path_text(nodes),
            "evidence_refs": chain_refs + anchor_refs,
        }

    def _display_entity_id(self, entity: Optional[CodeEntity], fallback_id: str) -> str:
        if entity is None:
            return fallback_id

        rel_path = entity.file_path.split("astropy_astropy/")[-1] if entity.file_path else entity.file_path
        symbol = entity.name
        return f"{rel_path}::{symbol}"

    def _extract_code_snippet(self, entity: Optional[CodeEntity]) -> str:
        if entity is None or not entity.file_path:
            return ""
        if entity.id in self._snippet_cache:
            return self._snippet_cache[entity.id]

        try:
            with open(entity.file_path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except Exception:
            self._snippet_cache[entity.id] = ""
            return ""

        if entity.line_start is not None:
            start = max(1, entity.line_start - self.code_context_radius_lines)
            end = min(len(lines), (entity.line_end or entity.line_start) + self.code_context_radius_lines)
        else:
            start = 1
            end = min(len(lines), 40)

        snippet = "".join(lines[start - 1:end]).strip()
        self._snippet_cache[entity.id] = snippet[:2000]
        return self._snippet_cache[entity.id]

    def _summarize_path(self, path_id: str, candidate_id: str, path: List[str]) -> PathSummary:
        node_summaries: List[PathNodeSummary] = []

        for index, entity_id in enumerate(path):
            entity = self._get_entity(entity_id)
            next_id = path[index + 1] if index + 1 < len(path) else None
            edge_weight = None
            if next_id is not None:
                stored_edge = self.crg.stored_edge_for_path_step(entity_id, next_id)
                edge_weight = stored_edge.weight if stored_edge else 0.0

            node_summaries.append(
                PathNodeSummary(
                    entity_id=entity_id,
                    display_id=self._display_entity_id(entity, entity_id),
                    name=entity.name if entity else entity_id,
                    entity_type=entity.entity_type.value if entity else "unknown",
                    file_path=entity.file_path if entity else "",
                    semantic_summary=self._entity_summary(entity),
                    code_snippet=self._extract_code_snippet(entity),
                    edge_weight_to_next=edge_weight,
                )
            )

        compressed_text = self._compress_path_text(node_summaries)
        initial_credibility = self._compute_path_credibility(path, use_updated_weights=None)
        evidence_pack = self._build_path_evidence_pack(path_id, candidate_id, path, node_summaries)
        return PathSummary(
            path_id=path_id,
            candidate_id=candidate_id,
            root_id=path[-1],
            node_ids=list(path),
            nodes=node_summaries,
            compressed_text=compressed_text,
            initial_credibility=initial_credibility,
            evidence_pack=evidence_pack,
        )

    def _entity_summary(self, entity: Optional[CodeEntity]) -> str:
        if entity is None:
            return "Unknown graph entity in the explanation chain."
        if entity.id in self._node_summary_cache:
            return self._node_summary_cache[entity.id]
        if entity.semantic_summary:
            self._node_summary_cache[entity.id] = entity.semantic_summary[:200]
            return self._node_summary_cache[entity.id]

        snippet = self._extract_code_snippet(entity)
        if self.summary_llm and snippet:
            prompt = f"""
You summarize one code entity for CG-MAD path evidence.

Entity id: {self._display_entity_id(entity, entity.id)}
Entity type: {entity.entity_type.value}
Issue text:
{self.issue_description}

Code snippet:
```python
{snippet}
```

Return strict JSON:
{{
  "summary": "One or two short sentences describing the entity's role or key behavior."
}}
""".strip()
            payload = self._json_or_fallback(
                prompt,
                {"summary": self._heuristic_entity_summary(entity)},
                llm_client=self.summary_llm,
            )
            summary = str(payload.get("summary", "")).strip() or self._heuristic_entity_summary(entity)
            self._node_summary_cache[entity.id] = summary[:240]
            return self._node_summary_cache[entity.id]

        summary = self._heuristic_entity_summary(entity)
        self._node_summary_cache[entity.id] = summary
        return summary

    def _heuristic_entity_summary(self, entity: Optional[CodeEntity]) -> str:
        if entity is None:
            return "Unknown graph entity in the explanation chain."
        rel_path = entity.file_path.split("astropy_astropy/")[-1] if entity.file_path else entity.file_path
        if entity.entity_type == EntityType.FAILURE_ROOT:
            return f"Failure observation root anchored at `{rel_path}`."
        if entity.entity_type == EntityType.FILE:
            return f"Source file `{rel_path}` that contains failure-side or implementation-side context."
        if entity.entity_type == EntityType.CLASS:
            return f"Class `{entity.name}` in `{rel_path}` that groups related behavior."
        if entity.entity_type == EntityType.FUNCTION:
            return f"Function `{entity.name}` in `{rel_path}` that participates in the explanation chain."
        if entity.entity_type == EntityType.PARAMETER:
            return f"Parameter `{entity.name}` in `{rel_path}` carrying local data-flow context."
        if entity.entity_type == EntityType.VARIABLE:
            return f"Variable `{entity.name}` in `{rel_path}` used as an intermediate state in the chain."
        if entity.entity_type == EntityType.IMPORT:
            return f"Imported symbol `{entity.name}` in `{rel_path}` connecting the current module to external logic."
        return f"Entity `{entity.name}` in `{rel_path}`."

    def _compress_path_text(self, nodes: Sequence[PathNodeSummary]) -> str:
        lines = []
        for index, node in enumerate(nodes):
            edge_text = ""
            if node.edge_weight_to_next is not None:
                edge_text = f" [c={node.edge_weight_to_next:.6f}]"
            label = "Leaf" if index == 0 else ("Root" if index == len(nodes) - 1 else f"n{index}")
            lines.append(
                f"{label}: node:{node.entity_id} => {node.display_id} ({node.entity_type}){edge_text} :: {node.semantic_summary}"
            )
        for left_node, right_node in zip(nodes[:-1], nodes[1:]):
            lines.append(f"edge:{left_node.entity_id}->{right_node.entity_id}")
        return "\n".join(lines)

    def _group_path_summaries(
        self,
        path_summaries: Iterable[PathSummary],
        all_candidate_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[PathSummary]]:
        grouped: Dict[str, List[PathSummary]] = {}
        for summary in path_summaries:
            grouped.setdefault(summary.candidate_id, []).append(summary)
        for candidate_id in all_candidate_ids or []:
            grouped.setdefault(candidate_id, [])
        return grouped

    def _debate_path_set(
        self,
        candidate_id: str,
        summaries: List[PathSummary],
    ) -> Tuple[List[PathPairDebateRecord], Dict[str, float]]:
        pair_records: List[PathPairDebateRecord] = []
        win_counts = {summary.path_id: 0 for summary in summaries}
        total_pairs = math.comb(len(summaries), 2)

        for path_a, path_b in itertools.combinations(summaries, 2):
            record = self._debate_path_pair(candidate_id, path_a, path_b)
            pair_records.append(record)
            win_counts[record.winner_path_id] += 1

        rates = {
            summary.path_id: (win_counts[summary.path_id] / total_pairs if total_pairs else 1.0)
            for summary in summaries
        }
        return pair_records, rates

    def _debate_single_path(self, summary: PathSummary) -> SinglePathDebateRecord:
        rounds: List[DebateRoundRecord] = []
        support_wins = 0
        consecutive: List[str] = []

        for round_index in range(1, self.max_rounds + 1):
            support_payload = self._support_single_path(summary)
            oppose_payload = self._oppose_single_path(summary)
            judge = self._judge_single_path(summary, support_payload, oppose_payload)
            rounds.append(
                DebateRoundRecord(
                    round_index=round_index,
                    support=AgentArgument("support", support_payload),
                    oppose=AgentArgument("oppose", oppose_payload),
                    judge=judge,
                )
            )
            if judge.winner == "support":
                support_wins += 1

            consecutive.append(judge.winner)
            if len(consecutive) > self.convergence_threshold:
                consecutive.pop(0)
            if len(consecutive) == self.convergence_threshold and len(set(consecutive)) == 1:
                break

        total_rounds = len(rounds)
        support_win_rate = support_wins / total_rounds if total_rounds else 0.5
        converged = total_rounds < self.max_rounds
        return SinglePathDebateRecord(
            candidate_id=summary.candidate_id,
            path_id=summary.path_id,
            rounds=rounds,
            support_win_rate=support_win_rate,
            converged=converged,
        )

    def _debate_path_pair(
        self,
        candidate_id: str,
        path_a: PathSummary,
        path_b: PathSummary,
    ) -> PathPairDebateRecord:
        rounds: List[DebateRoundRecord] = []
        consecutive: List[str] = []
        winners: List[str] = []

        for round_index in range(1, self.max_rounds + 1):
            support_payload = self._support_pair(path_a, path_b)
            oppose_payload = self._oppose_pair(path_a, path_b)
            judge = self._judge_pair(path_a, path_b, support_payload, oppose_payload)
            rounds.append(
                DebateRoundRecord(
                    round_index=round_index,
                    support=AgentArgument("support", support_payload),
                    oppose=AgentArgument("oppose", oppose_payload),
                    judge=judge,
                )
            )

            winner_path_id = path_a.path_id if judge.winner == "A" else path_b.path_id
            winners.append(winner_path_id)
            consecutive.append(winner_path_id)
            if len(consecutive) > self.convergence_threshold:
                consecutive.pop(0)
            if len(consecutive) == self.convergence_threshold and len(set(consecutive)) == 1:
                break

        winner_path_id = max(
            (path_a.path_id, path_b.path_id),
            key=lambda path_id: (winners.count(path_id), -[path_a.path_id, path_b.path_id].index(path_id)),
        )
        converged = len(rounds) < self.max_rounds
        return PathPairDebateRecord(
            candidate_id=candidate_id,
            path_a_id=path_a.path_id,
            path_b_id=path_b.path_id,
            rounds=rounds,
            winner_path_id=winner_path_id,
            converged=converged,
        )

    def _select_provisional_representatives(
        self,
        summaries_by_candidate: Dict[str, List[PathSummary]],
        path_win_rates: Dict[str, float],
    ) -> Dict[str, PathSummary]:
        representatives: Dict[str, PathSummary] = {}
        for candidate_id, summaries in summaries_by_candidate.items():
            representatives[candidate_id] = max(
                summaries,
                key=lambda summary: (
                    path_win_rates.get(summary.path_id, 0.0),
                    summary.initial_credibility,
                ),
            )
        return representatives

    def _debate_candidates(
        self,
        representatives: Dict[str, PathSummary],
    ) -> Tuple[List[CandidatePairDebateRecord], Dict[str, float]]:
        candidate_ids = sorted(representatives)
        if len(candidate_ids) == 1:
            only_id = candidate_ids[0]
            return [], {only_id: 1.0}

        pair_records: List[CandidatePairDebateRecord] = []
        win_counts = {candidate_id: 0 for candidate_id in candidate_ids}
        total_pairs = math.comb(len(candidate_ids), 2)

        for candidate_a_id, candidate_b_id in itertools.combinations(candidate_ids, 2):
            record = self._debate_candidate_pair(
                representatives[candidate_a_id],
                representatives[candidate_b_id],
            )
            pair_records.append(record)
            win_counts[record.winner_candidate_id] += 1

        rates = {
            candidate_id: win_counts[candidate_id] / total_pairs
            for candidate_id in candidate_ids
        }
        return pair_records, rates

    def _debate_candidate_pair(
        self,
        path_a: PathSummary,
        path_b: PathSummary,
    ) -> CandidatePairDebateRecord:
        rounds: List[DebateRoundRecord] = []
        consecutive: List[str] = []
        winners: List[str] = []

        for round_index in range(1, self.max_rounds + 1):
            support_payload = self._support_candidate_pair(path_a, path_b)
            oppose_payload = self._oppose_candidate_pair(path_a, path_b)
            judge = self._judge_candidate_pair(path_a, path_b, support_payload, oppose_payload)
            rounds.append(
                DebateRoundRecord(
                    round_index=round_index,
                    support=AgentArgument("support", support_payload),
                    oppose=AgentArgument("oppose", oppose_payload),
                    judge=judge,
                )
            )

            winner_candidate_id = path_a.candidate_id if judge.winner == "A" else path_b.candidate_id
            winners.append(winner_candidate_id)
            consecutive.append(winner_candidate_id)
            if len(consecutive) > self.convergence_threshold:
                consecutive.pop(0)
            if len(consecutive) == self.convergence_threshold and len(set(consecutive)) == 1:
                break

        winner_candidate_id = max(
            (path_a.candidate_id, path_b.candidate_id),
            key=lambda candidate_id: (
                winners.count(candidate_id),
                -[path_a.candidate_id, path_b.candidate_id].index(candidate_id),
            ),
        )
        converged = len(rounds) < self.max_rounds
        return CandidatePairDebateRecord(
            candidate_a_id=path_a.candidate_id,
            candidate_b_id=path_b.candidate_id,
            path_a_id=path_a.path_id,
            path_b_id=path_b.path_id,
            rounds=rounds,
            winner_candidate_id=winner_candidate_id,
            converged=converged,
        )

    def _update_edge_weights(
        self,
        summaries_by_candidate: Dict[str, List[PathSummary]],
        path_win_rates: Dict[str, float],
        location_win_rates: Dict[str, float],
    ) -> Dict[str, float]:
        edge_proposals: Dict[Tuple[str, str], List[float]] = {}

        for candidate_id, summaries in summaries_by_candidate.items():
            p_loc = location_win_rates.get(candidate_id, 1.0)
            for summary in summaries:
                p_path = path_win_rates.get(summary.path_id, 0.0)
                for current_id, next_id in zip(summary.node_ids[:-1], summary.node_ids[1:]):
                    stored_edge = self.crg.stored_edge_for_path_step(current_id, next_id)
                    if not stored_edge:
                        continue
                    if stored_edge.is_root_connection:
                        continue
                    initial = stored_edge.initial_weight if stored_edge.initial_weight > 0 else stored_edge.weight
                    updated = self.eta1 * initial + self.eta2 * p_path + self.eta3 * p_loc
                    edge_proposals.setdefault((stored_edge.source_id, stored_edge.target_id), []).append(updated)

        updated_weights: Dict[str, float] = {}
        for edge_key, edge in self.crg.edges.items():
            if edge.is_root_connection:
                self.crg.update_edge_weight(edge.source_id, edge.target_id, 1.0)
                updated_weights[f"{edge.source_id}::{edge.target_id}"] = 1.0
                continue

            proposals = edge_proposals.get(edge_key)
            if proposals:
                final_weight = sum(proposals) / len(proposals)
            else:
                final_weight = edge.initial_weight if edge.initial_weight > 0 else edge.weight

            self.crg.update_edge_weight(edge.source_id, edge.target_id, final_weight)
            updated_weights[f"{edge.source_id}::{edge.target_id}"] = final_weight

        return updated_weights

    def _assess_candidates(
        self,
        all_candidate_ids: Sequence[str],
        summaries_by_candidate: Dict[str, List[PathSummary]],
        path_win_rates: Dict[str, float],
        location_win_rates: Dict[str, float],
        updated_edge_weights: Dict[str, float],
        candidate_pair_debates: List[CandidatePairDebateRecord],
    ) -> List[CandidateAssessment]:
        assessments: List[CandidateAssessment] = []
        reason_index = self._collect_candidate_reasons(candidate_pair_debates)

        for candidate_id in all_candidate_ids:
            summaries = summaries_by_candidate.get(candidate_id, [])
            if not summaries:
                assessments.append(
                    CandidateAssessment(
                        candidate_id=candidate_id,
                        path_win_rates={},
                        location_win_rate=0.0,
                        representative_path_id="",
                        representative_path_credibility=0.0,
                        final_credibility=0.0,
                        judge_reason_summary=["No candidate-to-root explanation path found in the CRG."],
                    )
                )
                continue

            representative = max(
                summaries,
                key=lambda summary: self._compute_path_credibility(summary.node_ids, updated_edge_weights),
            )
            representative_cred = self._compute_path_credibility(representative.node_ids, updated_edge_weights)
            assessments.append(
                CandidateAssessment(
                    candidate_id=candidate_id,
                    path_win_rates={summary.path_id: path_win_rates.get(summary.path_id, 0.0) for summary in summaries},
                    location_win_rate=location_win_rates.get(candidate_id, 1.0),
                    representative_path_id=representative.path_id,
                    representative_path_credibility=representative_cred,
                    final_credibility=representative_cred,
                    judge_reason_summary=reason_index.get(candidate_id, []),
                )
            )

        assessments.sort(key=lambda item: item.final_credibility, reverse=True)
        return assessments

    def _collect_candidate_reasons(
        self,
        candidate_pair_debates: Sequence[CandidatePairDebateRecord],
    ) -> Dict[str, List[str]]:
        reasons: Dict[str, List[str]] = {}
        for record in candidate_pair_debates:
            for round_record in record.rounds:
                for candidate_id in (record.candidate_a_id, record.candidate_b_id):
                    reasons.setdefault(candidate_id, [])
                    reasons[candidate_id].extend(round_record.judge.key_reasons[:2])
        return {
            candidate_id: list(dict.fromkeys(items))[:6]
            for candidate_id, items in reasons.items()
        }

    def _compute_path_credibility(
        self,
        path: List[str],
        use_updated_weights: Optional[Dict[str, float]],
    ) -> float:
        edge_weights: List[float] = []
        for current_id, next_id in zip(path[:-1], path[1:]):
            stored_edge = self.crg.stored_edge_for_path_step(current_id, next_id)
            if not stored_edge:
                continue
            edge_key = f"{stored_edge.source_id}::{stored_edge.target_id}"
            if use_updated_weights is None:
                weight = stored_edge.initial_weight if stored_edge.initial_weight > 0 else stored_edge.weight
            else:
                weight = use_updated_weights.get(edge_key, stored_edge.weight)
            edge_weights.append(weight)

        if not edge_weights:
            return 0.0

        product = 1.0
        for weight in edge_weights:
            product *= max(weight, 1e-12)
        geometric_mean = product ** (1.0 / len(edge_weights))
        return geometric_mean * math.exp(-self.length_penalty * len(edge_weights))

    def _path_pair_prompt_pack(self, path_a: PathSummary, path_b: PathSummary) -> Dict[str, Any]:
        return {
            "issue_text": self.evidence_pack.issue_text,
            "path_a": self._compact_path_evidence_pack(path_a),
            "path_b": self._compact_path_evidence_pack(path_b),
            "failure_anchors": [asdict(item) for item in self.evidence_pack.failure_anchors],
        }

    def _single_path_prompt_pack(self, summary: PathSummary) -> Dict[str, Any]:
        return {
            "issue_text": self.evidence_pack.issue_text,
            "path_evidence_pack": self._compact_path_evidence_pack(summary),
            "failure_anchors": [asdict(item) for item in self.evidence_pack.failure_anchors],
        }

    def _candidate_pair_prompt_pack(self, path_a: PathSummary, path_b: PathSummary) -> Dict[str, Any]:
        return {
            "issue_text": self.evidence_pack.issue_text,
            "candidate_a_representative_path": self._compact_path_evidence_pack(path_a),
            "candidate_b_representative_path": self._compact_path_evidence_pack(path_b),
            "candidate_contexts": [
                self._compact_candidate_context(asdict(item))
                for item in self.evidence_pack.candidate_contexts
            ],
            "failure_anchors": [asdict(item) for item in self.evidence_pack.failure_anchors],
        }

    def _truncate_text(self, text: Any, max_chars: int) -> str:
        value = str(text or "")
        if max_chars <= 0 or len(value) <= max_chars:
            return value
        return value[:max_chars].rstrip() + "\n...[truncated]"

    def _compact_candidate_context(self, context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if context is None:
            return None
        compact = dict(context)
        compact["code_snippet"] = self._truncate_text(
            compact.get("code_snippet", ""),
            self.prompt_code_snippet_chars,
        )
        return compact

    def _compact_node_summary(self, node: PathNodeSummary) -> Dict[str, Any]:
        return {
            "entity_id": node.entity_id,
            "display_id": node.display_id,
            "name": node.name,
            "entity_type": node.entity_type,
            "file_path": node.file_path,
            "semantic_summary": self._truncate_text(node.semantic_summary, 280),
            "code_snippet": self._truncate_text(node.code_snippet, self.prompt_code_snippet_chars),
            "edge_weight_to_next": node.edge_weight_to_next,
        }

    def _compact_path_nodes(self, summary: PathSummary) -> List[Dict[str, Any]]:
        nodes = summary.nodes
        limit = self.prompt_max_path_nodes
        if limit <= 0 or len(nodes) <= limit:
            return [self._compact_node_summary(node) for node in nodes]

        keep_head = max(1, limit // 2)
        keep_tail = max(1, limit - keep_head)
        omitted = len(nodes) - keep_head - keep_tail
        compact_nodes = [self._compact_node_summary(node) for node in nodes[:keep_head]]
        compact_nodes.append(
            {
                "entity_id": "__omitted_intermediate_nodes__",
                "display_id": "__omitted_intermediate_nodes__",
                "name": "__omitted_intermediate_nodes__",
                "entity_type": "omitted",
                "file_path": "",
                "semantic_summary": f"{omitted} intermediate nodes omitted to keep the debate prompt bounded.",
                "code_snippet": "",
                "edge_weight_to_next": None,
            }
        )
        compact_nodes.extend(self._compact_node_summary(node) for node in nodes[-keep_tail:])
        return compact_nodes

    def _compact_path_evidence_pack(self, summary: PathSummary) -> Dict[str, Any]:
        return {
            "path_id": summary.path_id,
            "candidate_id": summary.candidate_id,
            "candidate_context": self._compact_candidate_context(
                summary.evidence_pack.get("candidate_context")
            ),
            "failure_anchor_refs": self._path_anchor_refs(summary),
            "nodes": self._compact_path_nodes(summary),
            "chain_text": self._truncate_text(summary.compressed_text, self.prompt_chain_text_chars),
            "evidence_refs": list(summary.evidence_pack.get("evidence_refs", [])),
        }

    def _path_anchor_refs(self, summary: PathSummary) -> List[str]:
        refs = summary.evidence_pack.get("failure_anchor_refs") or []
        if not refs:
            refs = [anchor.anchor_id for anchor in self.evidence_pack.failure_anchors]

        normalized: List[str] = []
        for ref in refs:
            text = str(ref)
            normalized.append(text if text.startswith("anchor:") else f"anchor:{text}")
        return normalized

    def _first_path_anchor_ref(self, summary: PathSummary) -> str:
        refs = self._path_anchor_refs(summary)
        if refs:
            return refs[0]
        return f"node:{summary.root_id}"

    def _first_path_node_ref(self, summary: PathSummary) -> str:
        if summary.node_ids:
            return f"node:{summary.node_ids[0]}"
        return f"node:{summary.candidate_id}"

    def _support_pair(self, path_a: PathSummary, path_b: PathSummary) -> Dict[str, Any]:
        fallback = self._fallback_pair_argument(path_a, path_b, favor="A")
        prompt_pack = self._path_pair_prompt_pack(path_a, path_b)
        prompt = f"""
You are the Proponent in CG-MAD path-level debate.
Given the defect description and two candidate paths, argue why Path A better explains the failure.
You must cite evidence_refs using node:<id>, edge:<left_id>-><right_id>, or anchor:<id>.
Do not invent facts not present in the evidence pack. Output strict JSON only.

Issue / path evidence:
{json.dumps(prompt_pack, ensure_ascii=False, indent=2)}

Output JSON schema:
{{
  "claim": "...",
  "evidence_refs": ["node:...", "edge:...->...", "anchor:..."],
  "mechanism": "..."
}}
""".strip()
        return self._json_or_fallback(prompt, fallback, llm_client=self.llm)

    def _oppose_pair(self, path_a: PathSummary, path_b: PathSummary) -> Dict[str, Any]:
        fallback = self._fallback_pair_argument(path_a, path_b, favor="B")
        prompt_pack = self._path_pair_prompt_pack(path_a, path_b)
        prompt = f"""
You are the Skeptic in CG-MAD path-level debate.
Given the defect description and two candidate paths, argue why Path A is weaker or noisier than Path B.
Focus on semantic gaps, weak links, and unrelated intermediate entities.
You must cite weak_links using node:<id>, edge:<left_id>-><right_id>, or anchor:<id>.
Do not invent facts not present in the evidence pack. Output strict JSON only.

Issue / path evidence:
{json.dumps(prompt_pack, ensure_ascii=False, indent=2)}

Output JSON schema:
{{
  "counter_claim": "...",
  "weak_links": ["edge:...->...", "node:..."],
  "noise_flags": ["semantic_gap", "weak_edge_dependency"]
}}
""".strip()
        return self._json_or_fallback(prompt, fallback, llm_client=self.llm)

    def _judge_pair(
        self,
        path_a: PathSummary,
        path_b: PathSummary,
        support_payload: Dict[str, Any],
        oppose_payload: Dict[str, Any],
    ) -> JudgeDecision:
        fallback_winner = self._fallback_pair_winner(path_a, path_b)
        prompt_pack = self._path_pair_prompt_pack(path_a, path_b)
        prompt = f"""
You are the Evaluator in CG-MAD path-level blind judging.
You see the same defect description, the same path evidence pack, and structured summaries from Proponent and Skeptic.
Judge only from the evidence pack and the structured arguments. Do not be swayed by style.
Prefer: failure-anchor alignment, explanation-chain continuity, mechanism plausibility, intervention usefulness.
You must cite evidence_refs using node:<id>, edge:<left_id>-><right_id>, or anchor:<id>.
Output strict JSON only.

Issue / path evidence:
{json.dumps(prompt_pack, ensure_ascii=False, indent=2)}

Support JSON:
{json.dumps(support_payload, ensure_ascii=False, indent=2)}

Oppose JSON:
{json.dumps(oppose_payload, ensure_ascii=False, indent=2)}

Output JSON schema:
{{
  "winner": "A",
  "key_reasons": ["...", "..."],
  "evidence_refs": ["node:...", "edge:...->...", "anchor:..."]
}}
""".strip()
        payload = self._json_or_fallback(
            prompt,
            {
                "winner": fallback_winner,
                "key_reasons": ["Fallback structural comparison"],
                "evidence_refs": [
                    self._first_path_node_ref(path_a),
                    self._first_path_anchor_ref(path_a),
                    self._first_path_node_ref(path_b),
                    self._first_path_anchor_ref(path_b),
                ],
            },
            llm_client=self.judge_llm,
        )
        return self._judge_from_payload(payload, fallback_winner)

    def _support_single_path(self, summary: PathSummary) -> Dict[str, Any]:
        score = summary.initial_credibility
        fallback = {
            "claim": "The path maintains a plausible explanation chain from candidate to failure anchor.",
            "evidence_refs": [self._first_path_node_ref(summary), self._first_path_anchor_ref(summary)],
            "mechanism": f"Fallback structural support with initial credibility {score:.6f}.",
        }
        prompt_pack = self._single_path_prompt_pack(summary)
        prompt = f"""
You are the Proponent in CG-MAD single-path evaluation.
Given the defect description and one path evidence pack, argue why this path can mechanistically explain the failure.
You must cite evidence_refs using node:<id>, edge:<left_id>-><right_id>, or anchor:<id>.
Do not invent facts. Output strict JSON only.

Issue / path evidence:
{json.dumps(prompt_pack, ensure_ascii=False, indent=2)}

Output JSON schema:
{{
  "claim": "...",
  "evidence_refs": ["node:...", "edge:...->...", "anchor:..."],
  "mechanism": "..."
}}
""".strip()
        return self._json_or_fallback(prompt, fallback, llm_client=self.llm)

    def _oppose_single_path(self, summary: PathSummary) -> Dict[str, Any]:
        fallback = {
            "counter_claim": "The path may contain noisy or weak links.",
            "weak_links": [],
            "noise_flags": ["fallback"],
        }
        prompt_pack = self._single_path_prompt_pack(summary)
        prompt = f"""
You are the Skeptic in CG-MAD single-path evaluation.
Given the defect description and one path evidence pack, argue why the path may fail to explain the observed failure.
Focus on semantic breaks, weak links, and unrelated entities.
You must cite weak_links using node:<id>, edge:<left_id>-><right_id>, or anchor:<id>.
Do not invent facts. Output strict JSON only.

Issue / path evidence:
{json.dumps(prompt_pack, ensure_ascii=False, indent=2)}

Output JSON schema:
{{
  "counter_claim": "...",
  "weak_links": ["edge:...->...", "node:..."],
  "noise_flags": ["semantic_gap", "weak_edge_dependency"]
}}
""".strip()
        return self._json_or_fallback(prompt, fallback, llm_client=self.llm)

    def _judge_single_path(
        self,
        summary: PathSummary,
        support_payload: Dict[str, Any],
        oppose_payload: Dict[str, Any],
    ) -> JudgeDecision:
        support_score = summary.initial_credibility
        fallback_winner = "support" if support_score >= 0.3 else "oppose"
        prompt_pack = self._single_path_prompt_pack(summary)
        prompt = f"""
You are the Evaluator in CG-MAD single-path blind judging.
You see the same defect description, the same path evidence pack, and structured summaries from Proponent and Skeptic.
Judge only from the evidence pack and the structured arguments. Do not be swayed by style.
You must cite evidence_refs using node:<id>, edge:<left_id>-><right_id>, or anchor:<id>.
Output strict JSON only.

Issue / path evidence:
{json.dumps(prompt_pack, ensure_ascii=False, indent=2)}

Support JSON:
{json.dumps(support_payload, ensure_ascii=False, indent=2)}

Oppose JSON:
{json.dumps(oppose_payload, ensure_ascii=False, indent=2)}

Output JSON schema:
{{
  "winner": "support",
  "key_reasons": ["...", "..."],
  "evidence_refs": ["node:...", "edge:...->...", "anchor:..."]
}}
""".strip()
        payload = self._json_or_fallback(
            prompt,
            {
                "winner": fallback_winner,
                "key_reasons": ["Fallback single-path evaluation"],
                "evidence_refs": [f"node:{summary.node_ids[0]}", f"node:{summary.node_ids[-1]}"],
            },
            llm_client=self.judge_llm,
        )
        return self._judge_from_payload(payload, fallback_winner)

    def _support_candidate_pair(self, path_a: PathSummary, path_b: PathSummary) -> Dict[str, Any]:
        fallback = {
            "claim": "Candidate A has the more plausible root-cause position.",
            "evidence_refs": [f"node:{path_a.candidate_id}", self._first_path_anchor_ref(path_a)],
            "mechanism": "Fallback comparison favors the representative path with stronger structural credibility.",
        }
        prompt_pack = self._candidate_pair_prompt_pack(path_a, path_b)
        prompt = f"""
You are the Proponent in CG-MAD location-level debate.
Given the defect description and two representative candidate paths, argue why Candidate A is the more plausible intervention-side root cause location.
You must cite evidence_refs using node:<id>, edge:<left_id>-><right_id>, or anchor:<id>.
Do not invent facts. Output strict JSON only.

Issue / candidate evidence:
{json.dumps(prompt_pack, ensure_ascii=False, indent=2)}

Output JSON schema:
{{
  "claim": "...",
  "evidence_refs": ["node:...", "edge:...->...", "anchor:..."],
  "mechanism": "..."
}}
""".strip()
        return self._json_or_fallback(prompt, fallback, llm_client=self.llm)

    def _oppose_candidate_pair(self, path_a: PathSummary, path_b: PathSummary) -> Dict[str, Any]:
        fallback = {
            "counter_claim": "Candidate B is more plausible than Candidate A.",
            "weak_links": [f"node:{path_a.candidate_id}"],
            "noise_flags": ["fallback"],
        }
        prompt_pack = self._candidate_pair_prompt_pack(path_a, path_b)
        prompt = f"""
You are the Skeptic in CG-MAD location-level debate.
Given the defect description and two representative candidate paths, argue why Candidate A is weaker than Candidate B as the root cause location.
Focus on semantic breaks, weak links, and noisy intermediate entities.
You must cite weak_links using node:<id>, edge:<left_id>-><right_id>, or anchor:<id>.
Do not invent facts. Output strict JSON only.

Issue / candidate evidence:
{json.dumps(prompt_pack, ensure_ascii=False, indent=2)}

Output JSON schema:
{{
  "counter_claim": "...",
  "weak_links": ["edge:...->...", "node:..."],
  "noise_flags": ["semantic_gap", "weak_edge_dependency"]
}}
""".strip()
        return self._json_or_fallback(prompt, fallback, llm_client=self.llm)

    def _judge_candidate_pair(
        self,
        path_a: PathSummary,
        path_b: PathSummary,
        support_payload: Dict[str, Any],
        oppose_payload: Dict[str, Any],
    ) -> JudgeDecision:
        fallback_winner = self._fallback_pair_winner(path_a, path_b)
        prompt_pack = self._candidate_pair_prompt_pack(path_a, path_b)
        prompt = f"""
You are the Evaluator in CG-MAD location-level blind judging.
You see the same defect description, the same candidate/path evidence pack, and structured summaries from Proponent and Skeptic.
Judge only from the evidence pack and the structured arguments. Do not be swayed by style.
Prefer: failure-anchor alignment, explanation-chain continuity, whether the proposed intervention point is actionable, and whether Skeptic's pruning arguments hold.
You must cite evidence_refs using node:<id>, edge:<left_id>-><right_id>, or anchor:<id>.
Output strict JSON only.

Issue / candidate evidence:
{json.dumps(prompt_pack, ensure_ascii=False, indent=2)}

Support JSON:
{json.dumps(support_payload, ensure_ascii=False, indent=2)}

Oppose JSON:
{json.dumps(oppose_payload, ensure_ascii=False, indent=2)}

Output JSON schema:
{{
  "winner": "A",
  "key_reasons": ["...", "..."],
  "evidence_refs": ["node:...", "edge:...->...", "anchor:..."]
}}
""".strip()
        payload = self._json_or_fallback(
            prompt,
            {
                "winner": fallback_winner,
                "key_reasons": ["Fallback candidate comparison"],
                "evidence_refs": [
                    f"node:{path_a.candidate_id}",
                    self._first_path_anchor_ref(path_a),
                    f"node:{path_b.candidate_id}",
                    self._first_path_anchor_ref(path_b),
                ],
            },
            llm_client=self.judge_llm,
        )
        return self._judge_from_payload(payload, fallback_winner)

    def _judge_from_payload(self, payload: Dict[str, Any], default_winner: str) -> JudgeDecision:
        winner = str(payload.get("winner", default_winner))
        key_reasons = [str(item) for item in payload.get("key_reasons", [])][:5]
        evidence_refs = [str(item) for item in payload.get("evidence_refs", [])][:8]
        return JudgeDecision(
            winner=winner,
            key_reasons=key_reasons,
            evidence_refs=evidence_refs,
        )

    def _json_or_fallback(
        self,
        prompt: str,
        fallback: Dict[str, Any],
        llm_client: Optional[LLMInterface] = None,
    ) -> Dict[str, Any]:
        client = llm_client or self.llm
        if not client:
            return fallback
        try:
            kwargs: Dict[str, Any] = {}
            if self.random_seed is not None:
                kwargs["seed"] = self.random_seed
            logger.debug(
                "CG-MAD LLM call start model={} prompt_chars={}",
                getattr(client, "model_name", "unknown"),
                len(prompt),
            )
            response = client.generate(
                prompt,
                temperature=self.sampling_temperature,
                max_tokens=600,
                **kwargs,
            )
            logger.debug(
                "CG-MAD LLM call done model={} response_chars={}",
                getattr(client, "model_name", "unknown"),
                len(response or ""),
            )
        except Exception as exc:
            logger.debug(f"CG-MAD LLM call failed: {str(exc)[:120]}")
            return fallback

        parsed = self._extract_json(response)
        return parsed if parsed is not None else fallback

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        candidates = [text]
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            candidates.append(text[start:end])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    def _fallback_pair_argument(self, path_a: PathSummary, path_b: PathSummary, favor: str) -> Dict[str, Any]:
        favored = path_a if favor == "A" else path_b
        other = path_b if favor == "A" else path_a
        if favor == "A":
            return {
                "claim": "Fallback structural comparison favors Path A.",
                "evidence_refs": [f"node:{favored.candidate_id}", self._first_path_anchor_ref(favored)],
                "mechanism": f"{favored.path_id} has higher initial credibility than {other.path_id}.",
            }

        return {
            "counter_claim": "Fallback structural comparison favors Path B over Path A.",
            "weak_links": [f"node:{other.candidate_id}"],
            "noise_flags": ["fallback"],
        }

    def _fallback_pair_winner(self, path_a: PathSummary, path_b: PathSummary) -> str:
        score_a = path_a.initial_credibility
        score_b = path_b.initial_credibility
        return "A" if score_a >= score_b else "B"

    def _get_entity(self, entity_id: str) -> Optional[CodeEntity]:
        return self.crg.root_nodes.get(entity_id) or self.crg.code_graph.get_entity(entity_id)


if __name__ == "__main__":
    logger.info("CG-MAD module ready.")
