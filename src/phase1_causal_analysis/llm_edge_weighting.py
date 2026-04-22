"""
LLM-based initial edge weighting for paper section 3.1.1.

Implements Definition 1 and Definition 2:
- multi-upstream: pairwise relative comparison frequency
- single-upstream: deterministic weight = 1
"""

from __future__ import annotations

import itertools
import re
import time
from typing import Dict, List, Optional

from loguru import logger

from .causal_relevance_graph import (
    CausalRelevanceGraph,
    CodeEntity,
    CRGEdge,
    FailureEvidence,
)
from src.common.llm_interface import LLMInterface


class LLMEdgeWeightingStrategy:
    """Apply Definition 1 / Definition 2 initial strengths to CRG edges."""

    def __init__(
        self,
        llm_client: Optional[LLMInterface],
        temperature: float = 0.1,
        max_upstreams_per_node: int = 3,
        min_request_interval_seconds: float = 0.35,
        max_retries_on_rate_limit: int = 4,
        rate_limit_backoff_seconds: float = 5.0,
    ):
        self.llm = llm_client
        self.temperature = temperature
        self.max_upstreams_per_node = max_upstreams_per_node
        self.min_request_interval_seconds = min_request_interval_seconds
        self.max_retries_on_rate_limit = max_retries_on_rate_limit
        self.rate_limit_backoff_seconds = rate_limit_backoff_seconds
        self.cache: Dict[str, Optional[str]] = {}
        self._last_request_time = 0.0

    def _build_prompt(
        self,
        symptom_summary: str,
        downstream_node: CodeEntity,
        upstream_a: CodeEntity,
        upstream_b: CodeEntity,
    ) -> str:
        return f"""
You are implementing the pairwise comparison rule from a causality-guided APR workflow.

Observed failure evidence:
{symptom_summary}

Current downstream node j:
- type: {downstream_node.entity_type.value}
- name: {downstream_node.name}
- file: {downstream_node.file_path}

Candidate upstream node A:
- type: {upstream_a.entity_type.value}
- name: {upstream_a.name}
- file: {upstream_a.file_path}

Candidate upstream node B:
- type: {upstream_b.entity_type.value}
- name: {upstream_b.name}
- file: {upstream_b.file_path}

Question:
If fixing downstream node j is more likely to eliminate or mitigate the observed failure,
which upstream node is the more plausible explanation-side parent of j under the current
failure evidence?

Answer with exactly one line and no extra text:
Winner: A
or
Winner: B
""".strip()

    def _extract_winner(self, response: str) -> Optional[str]:
        match = re.search(r"Winner:\s*(A|B)", response or "", re.IGNORECASE)
        return match.group(1).upper() if match else None

    def _safe_generate(self, prompt: str) -> Optional[str]:
        if not self.llm:
            return None
        api_key = getattr(self.llm, "api_key", None)
        if api_key is not None and not api_key:
            return None
        for attempt in range(self.max_retries_on_rate_limit + 1):
            elapsed = time.time() - self._last_request_time
            if elapsed < self.min_request_interval_seconds:
                time.sleep(self.min_request_interval_seconds - elapsed)
            try:
                response = self.llm.generate(prompt, temperature=self.temperature, max_tokens=20)
                self._last_request_time = time.time()
                return response
            except Exception as exc:
                self._last_request_time = time.time()
                message = str(exc)
                is_rate_limit = "429" in message or "Too Many Requests" in message
                if is_rate_limit and attempt < self.max_retries_on_rate_limit:
                    sleep_seconds = self.rate_limit_backoff_seconds * (2 ** attempt)
                    logger.debug(
                        f"LLM comparison rate-limited; retrying in {sleep_seconds:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries_on_rate_limit})"
                    )
                    time.sleep(sleep_seconds)
                    continue
                logger.debug(f"LLM comparison failed: {message[:120]}")
                return None

    def apply_weights_to_crg(
        self,
        crg: CausalRelevanceGraph,
        failure_evidences: List[FailureEvidence],
    ) -> CausalRelevanceGraph:
        """
        Apply initial strengths c^(0) to CRG edges and sparsify noisy upstreams.
        """

        downstream_to_upstreams: Dict[str, List[CRGEdge]] = {}
        for edge in crg.edges.values():
            if edge.is_root_connection:
                edge.initial_weight = 1.0
                edge.weight = 1.0
                continue
            downstream_to_upstreams.setdefault(edge.source_id, []).append(edge)

        symptom_summary = "\n".join(
            f"- {evidence.symptom_type}: {evidence.symptom_message}"
            for evidence in failure_evidences
        )

        for downstream_id, upstream_edges in downstream_to_upstreams.items():
            num_upstreams = len(upstream_edges)
            if num_upstreams == 0:
                continue

            if num_upstreams == 1:
                upstream_edges[0].initial_weight = 1.0
                upstream_edges[0].weight = 1.0
                continue

            downstream_node = crg.root_nodes.get(downstream_id) or crg.code_graph.get_entity(downstream_id)
            if not downstream_node:
                continue

            pairs = list(itertools.combinations(upstream_edges, 2))
            win_counts = {edge.target_id: 0.0 for edge in upstream_edges}
            compared_pairs = 0

            for edge_a, edge_b in pairs:
                upstream_a = crg.code_graph.get_entity(edge_a.target_id)
                upstream_b = crg.code_graph.get_entity(edge_b.target_id)
                if not upstream_a or not upstream_b:
                    continue

                prompt = self._build_prompt(symptom_summary, downstream_node, upstream_a, upstream_b)
                if prompt in self.cache:
                    winner = self.cache[prompt]
                else:
                    response = self._safe_generate(prompt)
                    winner = self._extract_winner(response or "")
                    self.cache[prompt] = winner

                compared_pairs += 1
                if winner == "A":
                    win_counts[edge_a.target_id] += 1.0
                elif winner == "B":
                    win_counts[edge_b.target_id] += 1.0
                else:
                    # Unavailable / ambiguous response falls back to uniform voting.
                    win_counts[edge_a.target_id] += 0.5
                    win_counts[edge_b.target_id] += 0.5

            if compared_pairs == 0:
                continue

            for edge in upstream_edges:
                score = win_counts[edge.target_id] / compared_pairs
                edge.initial_weight = score
                edge.weight = score

        crg.prune_to_top_upstreams(self.max_upstreams_per_node)
        return crg
