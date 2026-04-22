"""
Phase 1: Causal Relevance Graph (CRG) Construction
==================================================

Implements paper section 3.1.1 "CRG Construction" with the following
constraints kept explicit in code:

1. Build a unified code graph G_CG as the structural candidate space.
2. Use only two top-level structural relation families:
   - containment
   - reference
3. Treat localization outputs as CRG leaf nodes.
4. Treat failure evidence as root-side observation entrances.
5. Enumerate candidate -> failure-anchor paths with DFS on the code graph.
6. Orient CRG edges in the symptom -> cause backtracking direction.
7. Leave initial edge strength assignment to the pairwise comparison rule
   implemented in llm_edge_weighting.py.

The implementation keeps a few pragmatic fallbacks so the system remains
runnable when failure evidence is incomplete, but the construction logic
follows the paper's 3.1.1 semantics.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
from loguru import logger


class EntityType(Enum):
    """Code entity types used in G_CG / G_CRG."""

    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    VARIABLE = "variable"
    PARAMETER = "parameter"
    IMPORT = "import"
    FAILURE_ROOT = "failure_root"


class RelationType(Enum):
    """
    Structural relation types.

    The paper constrains the code graph to two high-level relation families:
    containment and reference. For compatibility with previously cached graphs
    and surrounding utilities, legacy fine-grained values are still accepted,
    but construction now normalizes newly created relations to the two
    paper-level families.
    """

    CONTAINS = "contains"
    REFERENCES = "references"

    # Legacy / cache-compatible values.
    DEFINES = "defines"
    CALLS = "calls"
    INHERITS = "inherits"
    IMPORTS = "imports"
    READS = "reads"
    WRITES = "writes"
    ACCESSES = "accesses"

    @classmethod
    def normalized(cls, relation_type: "RelationType") -> "RelationType":
        """Map any fine-grained legacy type to the paper's two families."""

        if relation_type == cls.CONTAINS:
            return cls.CONTAINS
        return cls.REFERENCES


@dataclass
class CodeEntity:
    """Represents a code entity identified by (file, class, function, type)."""

    id: str
    name: str
    entity_type: EntityType
    file_path: str
    class_name: Optional[str] = None
    function_name: Optional[str] = None
    variable_name: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    parent_id: Optional[str] = None
    semantic_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CodeEntity) and self.id == other.id


@dataclass
class CodeRelation:
    """
    Structural candidate relation in G_CG.

    relation_type is normalized to containment/reference for new relations.
    More specific semantics, when needed, are carried in metadata.
    """

    source_id: str
    target_id: str
    relation_type: RelationType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_relation_type(self) -> RelationType:
        return RelationType.normalized(self.relation_type)


@dataclass
class FailureEvidence:
    """
    Failure evidence conditioning the CRG.

    symptom_location should be an observation-side location such as a failing
    test file, stack frame location, or other directly observed entry point.
    """

    symptom_type: str
    symptom_location: str
    symptom_message: str
    test_case_id: str
    stack_trace: List[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash((self.symptom_type, self.symptom_location, self.test_case_id))


class CodeGraph:
    """
    Unified code graph G_CG = (V_CG, E_CG).

    Entities are code syntax / semantics units; relations are the structural
    candidate space over which CRG paths are enumerated.
    """

    def __init__(self):
        self.entities: Dict[str, CodeEntity] = {}
        self.relations: List[CodeRelation] = []
        self.nx_graph = nx.DiGraph()
        logger.info("Initialized CodeGraph")

    def add_entity(self, entity: CodeEntity) -> None:
        """Add or replace a code entity."""

        self.entities[entity.id] = entity
        self.nx_graph.add_node(entity.id, entity=entity)

    def add_relation(self, relation: CodeRelation) -> None:
        """
        Add a structural relation.

        Unresolved relations are preserved in self.relations so they can be
        resolved later. They are only materialized into the NetworkX graph once
        both endpoints exist.
        """

        normalized = CodeRelation(
            source_id=relation.source_id,
            target_id=relation.target_id,
            relation_type=relation.normalized_relation_type(),
            metadata=dict(relation.metadata),
        )
        self.relations.append(normalized)

        if normalized.source_id in self.entities and normalized.target_id in self.entities:
            self.nx_graph.add_edge(
                normalized.source_id,
                normalized.target_id,
                relation_type=normalized.relation_type,
                metadata=normalized.metadata,
            )

    def rebuild_graph(self) -> None:
        """Rebuild the executable graph from entities + resolved relations."""

        self.nx_graph = nx.DiGraph()
        for entity_id, entity in self.entities.items():
            self.nx_graph.add_node(entity_id, entity=entity)

        for relation in self.relations:
            if relation.source_id in self.entities and relation.target_id in self.entities:
                self.nx_graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    relation_type=relation.relation_type,
                    metadata=relation.metadata,
                )

    def get_entity(self, entity_id: str) -> Optional[CodeEntity]:
        return self.entities.get(entity_id)

    def structural_neighbors(self, entity_id: str) -> List[str]:
        """
        Return undirected structural neighbors for DFS path enumeration.

        Paper 3.1.1 defines G_CG as the structural candidate space providing
        possible influence paths. For path enumeration, we therefore traverse
        the candidate space irrespective of the original relation direction.
        """

        if entity_id not in self.nx_graph:
            return []

        neighbors = set(self.nx_graph.successors(entity_id))
        neighbors.update(self.nx_graph.predecessors(entity_id))
        return sorted(neighbors)

    def relation_family_between(self, left_id: str, right_id: str) -> RelationType:
        """Return the normalized structural family between two adjacent nodes."""

        if self.nx_graph.has_edge(left_id, right_id):
            edge_data = self.nx_graph.get_edge_data(left_id, right_id) or {}
            relation_type = edge_data.get("relation_type")
            if isinstance(relation_type, RelationType):
                return relation_type

        if self.nx_graph.has_edge(right_id, left_id):
            edge_data = self.nx_graph.get_edge_data(right_id, left_id) or {}
            relation_type = edge_data.get("relation_type")
            if isinstance(relation_type, RelationType):
                return relation_type

        return RelationType.REFERENCES


@dataclass
class CRGEdge:
    """
    Directed edge in G_CRG.

    Direction follows the paper's backtracking convention: symptom-side /
    downstream node j -> cause-side / upstream node i.
    """

    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 0.0
    initial_weight: float = 0.0
    support_count: int = 1
    is_root_connection: bool = False

    def __post_init__(self) -> None:
        self.weight = float(min(max(self.weight, 0.0), 1.0))
        self.initial_weight = float(min(max(self.initial_weight, 0.0), 1.0))


class CausalRelevanceGraph:
    """
    Issue-conditioned CRG: G_CRG = (V_CRG, E_CRG, C).

    The graph stores:
    - candidate leaf nodes from localization
    - synthetic failure-root observation nodes
    - symptom -> cause edges
    - candidate -> root paths used as explanation chains
    """

    def __init__(self, code_graph: CodeGraph, failure_evidences: List[FailureEvidence]):
        self.code_graph = code_graph
        self.failure_evidences = failure_evidences
        self.edges: Dict[Tuple[str, str], CRGEdge] = {}
        self.root_nodes: Dict[str, CodeEntity] = {}
        self.anchor_entity_ids: Set[str] = set()
        self.candidate_leaf_ids: Set[str] = set()
        self.paths_by_candidate: Dict[str, List[List[str]]] = defaultdict(list)
        self.nx_graph = nx.DiGraph()

    def add_root_node(self, root_node: CodeEntity) -> None:
        self.root_nodes[root_node.id] = root_node
        self.nx_graph.add_node(root_node.id, entity=root_node)

    def add_edge(self, edge: CRGEdge) -> None:
        key = (edge.source_id, edge.target_id)

        if key in self.edges:
            existing = self.edges[key]
            existing.support_count += edge.support_count
            existing.is_root_connection = existing.is_root_connection or edge.is_root_connection
            existing.relation_type = edge.relation_type
            return

        self.edges[key] = edge
        self.nx_graph.add_edge(
            edge.source_id,
            edge.target_id,
            weight=edge.weight,
            initial_weight=edge.initial_weight,
            relation_type=edge.relation_type.value,
            support_count=edge.support_count,
            is_root_connection=edge.is_root_connection,
        )

    def register_path(self, candidate_id: str, path: List[str]) -> None:
        self.paths_by_candidate[candidate_id].append(path)

    def stored_edge_for_path_step(self, current_id: str, next_id: str) -> Optional[CRGEdge]:
        """
        Return the stored CRG edge corresponding to one path step.

        paths_by_candidate are stored in candidate -> ... -> failure_root order,
        while CRG edges follow the paper's symptom -> cause orientation.
        Therefore the step current -> next maps to the stored edge next -> current.
        """

        return self.edges.get((next_id, current_id))

    def path_edge_weights(self, path: List[str]) -> List[float]:
        """Return stored-edge weights aligned to a candidate->root path."""

        weights: List[float] = []
        for current_id, next_id in zip(path[:-1], path[1:]):
            edge = self.stored_edge_for_path_step(current_id, next_id)
            weights.append(edge.weight if edge else 0.0)
        return weights

    def update_edge_weight(self, source_id: str, target_id: str, new_weight: float) -> None:
        """Update one stored edge weight and keep the executable graph in sync."""

        key = (source_id, target_id)
        if key not in self.edges:
            return

        clamped = float(min(max(new_weight, 0.0), 1.0))
        edge = self.edges[key]
        edge.weight = clamped

        if self.nx_graph.has_edge(source_id, target_id):
            self.nx_graph[source_id][target_id]["weight"] = clamped

    def prune_to_top_upstreams(self, max_upstreams_per_node: int) -> None:
        """
        Keep only the strongest upstreams per downstream node.

        This implements the paper's "retain only top-ranked upstream nodes to
        weaken noisy edges" requirement.
        """

        if max_upstreams_per_node <= 0:
            return

        grouped: Dict[str, List[Tuple[Tuple[str, str], CRGEdge]]] = defaultdict(list)
        for key, edge in self.edges.items():
            grouped[edge.source_id].append((key, edge))

        keep_keys: Set[Tuple[str, str]] = set()
        for downstream_id, items in grouped.items():
            ranked = sorted(
                items,
                key=lambda item: (
                    item[1].initial_weight,
                    item[1].weight,
                    item[1].support_count,
                    item[1].is_root_connection,
                ),
                reverse=True,
            )
            for key, _ in ranked[:max_upstreams_per_node]:
                keep_keys.add(key)

        self.edges = {key: edge for key, edge in self.edges.items() if key in keep_keys}
        self._rebuild_graph()

    def _rebuild_graph(self) -> None:
        self.nx_graph = nx.DiGraph()

        for root_node in self.root_nodes.values():
            self.nx_graph.add_node(root_node.id, entity=root_node)

        for entity_id in self.candidate_leaf_ids | self.anchor_entity_ids:
            entity = self.code_graph.get_entity(entity_id)
            if entity:
                self.nx_graph.add_node(entity_id, entity=entity)

        for edge in self.edges.values():
            self.nx_graph.add_edge(
                edge.source_id,
                edge.target_id,
                weight=edge.weight,
                initial_weight=edge.initial_weight,
                relation_type=edge.relation_type.value,
                support_count=edge.support_count,
                is_root_connection=edge.is_root_connection,
            )


class EdgeWeightingStrategy:
    """Compatibility shim for legacy call sites."""

    def compute_weight(
        self,
        source_entity: CodeEntity,
        target_entity: CodeEntity,
        relation_type: RelationType,
        failure_evidences: List[FailureEvidence],
        code_graph: CodeGraph,
    ) -> float:
        raise NotImplementedError


class DataFlowWeighting(EdgeWeightingStrategy):
    """
    Compatibility shim for legacy construction paths.

    The paper's 3.1.1 initial strength is defined by pairwise upstream
    comparisons, so this strategy is not used for the final c^(0) values.
    It remains available as a lightweight fallback score.
    """

    def compute_weight(
        self,
        source_entity: CodeEntity,
        target_entity: CodeEntity,
        relation_type: RelationType,
        failure_evidences: List[FailureEvidence],
        code_graph: CodeGraph,
    ) -> float:
        base = 1.0 if relation_type == RelationType.CONTAINS else 0.5
        for evidence in failure_evidences:
            for frame in evidence.stack_trace:
                if target_entity.name in frame or target_entity.file_path in frame:
                    base = min(1.0, base + 0.2)
        return base


class CRGBuilder:
    """
    Build the structure of G_CRG according to paper section 3.1.1.

    The builder is responsible only for:
    - code-graph path enumeration
    - symptom-side root creation
    - symptom -> cause edge orientation
    - candidate/root/path bookkeeping

    Initial edge strengths c^(0) are assigned afterward by
    LLMEdgeWeightingStrategy using Definition 1 / Definition 2.
    """

    def __init__(
        self,
        code_graph: CodeGraph,
        weighting_strategy: Optional[EdgeWeightingStrategy] = None,
        max_path_depth: int = 8,
        max_paths_per_candidate: int = 100,
        max_upstreams_per_node: int = 3,
    ):
        self.code_graph = code_graph
        self.weighting_strategy = weighting_strategy
        self.max_path_depth = max_path_depth
        self.max_paths_per_candidate = max_paths_per_candidate
        self.max_upstreams_per_node = max_upstreams_per_node
        self._undirected_graph = self.code_graph.nx_graph.to_undirected(as_view=True)
        self._shortest_distance_cache: Dict[str, Dict[str, int]] = {}
        self._entities_by_name: Dict[str, List[CodeEntity]] = defaultdict(list)
        self._entities_by_file_path: Dict[str, List[CodeEntity]] = defaultdict(list)
        self._lower_file_paths: Dict[str, str] = {}
        for entity in self.code_graph.entities.values():
            for name in {
                entity.name,
                entity.function_name,
                entity.class_name,
                entity.variable_name,
            }:
                if name:
                    self._entities_by_name[name].append(entity)
            self._entities_by_file_path[entity.file_path].append(entity)
            self._lower_file_paths.setdefault(entity.file_path.lower(), entity.file_path)
        logger.info("Initialized CRGBuilder")

    def build(
        self,
        failure_evidences: List[FailureEvidence],
        candidates_L: Optional[List[CodeEntity]] = None,
    ) -> CausalRelevanceGraph:
        crg = CausalRelevanceGraph(self.code_graph, failure_evidences)

        if not candidates_L:
            logger.warning("No candidates_L provided. Returning empty CRG.")
            return crg

        candidate_ids = [
            candidate.id
            for candidate in candidates_L
            if candidate.id in self.code_graph.entities
        ]
        crg.candidate_leaf_ids.update(candidate_ids)

        root_specs = self._build_failure_roots(failure_evidences)
        for root_node, anchor_ids in root_specs:
            crg.add_root_node(root_node)
            crg.anchor_entity_ids.update(anchor_ids)
            for anchor_id in anchor_ids:
                crg.add_edge(
                    CRGEdge(
                        source_id=root_node.id,
                        target_id=anchor_id,
                        relation_type=RelationType.REFERENCES,
                        weight=0.0,
                        initial_weight=0.0,
                        is_root_connection=True,
                    )
                )

        logger.info(
            f"Building CRG with {len(candidate_ids)} candidates, "
            f"{len(root_specs)} failure roots and {len(crg.anchor_entity_ids)} anchor entities"
        )

        for candidate_id in candidate_ids:
            total_paths_for_candidate = 0

            for root_node, anchor_ids in root_specs:
                for anchor_id in anchor_ids:
                    paths = self._enumerate_paths(candidate_id, anchor_id)
                    for path in paths:
                        if total_paths_for_candidate >= self.max_paths_per_candidate:
                            break
                        full_path = list(path) + [root_node.id]
                        crg.register_path(candidate_id, full_path)
                        self._materialize_path(crg, structural_path=path, root_id=root_node.id)
                        total_paths_for_candidate += 1
                if total_paths_for_candidate >= self.max_paths_per_candidate:
                    break

            logger.debug(
                f"Candidate {candidate_id} produced "
                f"{total_paths_for_candidate} candidate->root paths"
            )

        return crg

    def _build_failure_roots(
        self,
        failure_evidences: List[FailureEvidence],
    ) -> List[Tuple[CodeEntity, List[str]]]:
        root_specs: List[Tuple[CodeEntity, List[str]]] = []

        for index, evidence in enumerate(failure_evidences):
            anchor_ids = self._match_failure_anchors(evidence)
            root_node = CodeEntity(
                id=f"failure_root::{index}",
                name=evidence.symptom_type or f"failure_root_{index}",
                entity_type=EntityType.FAILURE_ROOT,
                file_path=evidence.symptom_location or "<failure>",
                semantic_summary=evidence.symptom_message[:200],
                metadata={"test_case_id": evidence.test_case_id},
            )
            root_specs.append((root_node, anchor_ids))

        return root_specs

    def _match_failure_anchors(self, evidence: FailureEvidence) -> List[str]:
        """
        Resolve failure evidence to observation-side anchor entities.

        We keep matching heuristic and conservative:
        - exact entity id / file path matches
        - stack-trace frame hits
        - identifier tokens appearing in the symptom message
        """

        scored: Dict[str, int] = defaultdict(int)
        location_text = evidence.symptom_location or ""

        if location_text and location_text in self.code_graph.entities:
            scored[location_text] += 6

        if location_text:
            for file_path, entities in self._entities_by_file_path.items():
                if location_text != file_path and not file_path.endswith(location_text):
                    continue
                for entity in entities:
                    if entity.entity_type == EntityType.FILE:
                        scored[entity.id] += 6
                    elif entity.entity_type in {EntityType.FUNCTION, EntityType.CLASS}:
                        scored[entity.id] += 2

        for frame in evidence.stack_trace:
            frame_lower = frame.lower()
            for lower_path, file_path in self._lower_file_paths.items():
                if lower_path not in frame_lower:
                    continue
                for entity in self._entities_by_file_path[file_path]:
                    scored[entity.id] += 4

            for token in self._extract_identifier_tokens(frame):
                for entity in self._entities_by_name.get(token, []):
                    scored[entity.id] += 4

        for token in self._extract_identifier_tokens(evidence.symptom_message):
            for entity in self._entities_by_name.get(token, []):
                scored[entity.id] += 2

        if not scored and location_text:
            for file_path, entities in self._entities_by_file_path.items():
                if location_text not in file_path:
                    continue
                for entity in entities:
                    scored[entity.id] += 1

        ranked = sorted(
            scored.items(),
            key=lambda item: (
                item[1],
                self._anchor_priority(self.code_graph.get_entity(item[0])),
            ),
            reverse=True,
        )
        return [entity_id for entity_id, _ in ranked[:3]]

    def _anchor_priority(self, entity: Optional[CodeEntity]) -> int:
        if entity is None:
            return 0
        priority = {
            EntityType.FUNCTION: 4,
            EntityType.CLASS: 3,
            EntityType.VARIABLE: 2,
            EntityType.PARAMETER: 2,
            EntityType.FILE: 1,
        }
        return priority.get(entity.entity_type, 0)

    def _extract_identifier_tokens(self, text: str) -> Set[str]:
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or ""))
        return {token for token in tokens if len(token) > 2}

    def _enumerate_paths(self, start_id: str, goal_id: str) -> List[List[str]]:
        """
        Enumerate candidate->anchor paths with DFS on the structural candidate
        space G_CG.
        """

        if start_id == goal_id:
            return [[start_id]]

        paths: List[List[str]] = []
        undirected_graph = self._undirected_graph

        if start_id not in undirected_graph or goal_id not in undirected_graph:
            return paths

        shortest_distances = self._shortest_distance_cache.get(goal_id)
        if shortest_distances is None:
            try:
                shortest_distances = nx.single_source_shortest_path_length(
                    undirected_graph,
                    goal_id,
                    cutoff=self.max_path_depth,
                )
            except nx.NetworkXError:
                return paths
            self._shortest_distance_cache[goal_id] = shortest_distances

        if start_id not in shortest_distances:
            return paths

        def dfs(current_id: str, path: List[str], depth: int) -> None:
            if len(paths) >= self.max_paths_per_candidate:
                return
            if depth > self.max_path_depth:
                return
            if current_id == goal_id:
                paths.append(list(path))
                return

            remaining_depth = self.max_path_depth - depth
            candidate_neighbors = []
            for neighbor_id in undirected_graph.neighbors(current_id):
                if neighbor_id in path:
                    continue
                goal_distance = shortest_distances.get(neighbor_id)
                if goal_distance is None or goal_distance > remaining_depth - 1:
                    continue
                candidate_neighbors.append((goal_distance, neighbor_id))

            candidate_neighbors.sort(key=lambda item: (item[0], item[1]))
            for _, neighbor_id in candidate_neighbors:
                path.append(neighbor_id)
                dfs(neighbor_id, path, depth + 1)
                path.pop()

        dfs(start_id, [start_id], 0)
        return paths

    def _materialize_path(
        self,
        crg: CausalRelevanceGraph,
        structural_path: List[str],
        root_id: str,
    ) -> None:
        """
        Convert one candidate->anchor structural path into symptom->cause CRG
        edges, then connect the anchor entity to the synthetic failure root.
        """

        if len(structural_path) < 2:
            return

        # Path is [candidate, ..., anchor]. CRG edge direction must be
        # downstream/symptom-side -> upstream/cause-side, so we reverse each
        # adjacent step when materializing edges.
        for left_id, right_id in zip(structural_path[:-1], structural_path[1:]):
            crg.add_edge(
                CRGEdge(
                    source_id=right_id,
                    target_id=left_id,
                    relation_type=self.code_graph.relation_family_between(left_id, right_id),
                    weight=0.0,
                    initial_weight=0.0,
                )
            )

        crg.add_edge(
            CRGEdge(
                source_id=root_id,
                target_id=structural_path[-1],
                relation_type=RelationType.REFERENCES,
                weight=0.0,
                initial_weight=0.0,
                is_root_connection=True,
            )
        )


class PathCollector:
    """Compatibility helper for inspecting candidate->root explanation chains."""

    def __init__(self, crg: CausalRelevanceGraph):
        self.crg = crg

    def collect_paths(
        self,
        candidate_entity_id: str,
        max_path_length: int = 10,
        max_paths: int = 100,
    ) -> List[List[str]]:
        stored_paths = self.crg.paths_by_candidate.get(candidate_entity_id, [])
        filtered = [
            path
            for path in stored_paths
            if len(path) <= max_path_length + 1
        ]
        return filtered[:max_paths]


if __name__ == "__main__":
    logger.info("CRG construction module ready.")
