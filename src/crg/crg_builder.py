"""CRG (Causal Relevance Graph) Builder for extracting causal relationships"""

import re
from typing import Dict, List, Set, Optional, Tuple
from loguru import logger

from src.common.data_structures import (
    IssueContext, CodeEntity, CRGNode, CRGEdge, PathEvidence, 
    EntityType, EdgeType
)
from src.common.utils import dfs_paths, extract_function_calls, extract_variables
from src.common.llm_interface import LLMInterface


class CRGBuilder:
    """Builds Causal Relevance Graph for program repair"""
    
    def __init__(self, llm: LLMInterface, max_path_depth: int = 20, 
                 max_paths_per_location: int = 100):
        """
        Initialize CRG Builder
        
        Args:
            llm: LLM interface for semantic analysis
            max_path_depth: Maximum depth for path search
            max_paths_per_location: Maximum paths to keep per location
        """
        self.llm = llm
        self.max_path_depth = max_path_depth
        self.max_paths_per_location = max_paths_per_location
        self.logger = logger
    
    def build(self, issue: IssueContext, code_graph: Dict[str, List[str]],
             entity_map: Dict[str, CodeEntity]) -> 'CRG':
        """
        Build Causal Relevance Graph
        
        Args:
            issue: Issue context with description and candidate locations
            code_graph: Graph dict {node_id: [neighbors]}
            entity_map: Map of entity_id to CodeEntity
        
        Returns:
            CRG object with nodes, edges, and paths
        """
        
        self.logger.info(f"Building CRG for issue {issue.id}")
        
        # Step 1: Extract failure anchors from issue description
        failure_anchors = self._extract_failure_anchors(issue.description)
        
        if not failure_anchors:
            self.logger.warning("No failure anchors found, using generic root")
            failure_anchors = ["failure"]
        
        # Step 2: Create root nodes for failure anchors
        root_nodes = [
            CRGNode(
                entity_id=f"root_{anchor}",
                file_path="",
                entity_type=EntityType.FILE,
                semantic_summary=f"Failure due to: {anchor}",
                credibility=1.0,
                initial_strength=1.0
            )
            for anchor in failure_anchors
        ]
        
        # Step 3: Enumerate paths from candidates to failure anchors
        paths_by_candidate = {}
        for candidate in issue.candidates:
            paths = []
            for root in root_nodes:
                candidate_paths = self._enumerate_paths(
                    candidate, root.entity_id, code_graph
                )
                paths.extend(candidate_paths)
            
            if paths:
                paths_by_candidate[candidate] = paths
            else:
                # Create direct path if no path found
                direct_path = [candidate, root_nodes[0].entity_id]
                paths_by_candidate[candidate] = [direct_path]
        
        # Step 4: Compress paths to structured format
        compressed_paths = {}
        for candidate, paths in paths_by_candidate.items():
            compressed = [
                self._compress_path(path, entity_map)
                for path in paths[:self.max_paths_per_location]
            ]
            compressed_paths[candidate] = compressed
        
        # Step 5: Compute initial edge weights
        edge_weights = self._compute_initial_weights(
            paths_by_candidate, entity_map
        )
        
        # Step 6: Build CRG nodes and edges
        crg_nodes = root_nodes.copy()
        crg_edges = []
        
        for candidate_id, candidate_entity in entity_map.items():
            if candidate_id in issue.candidates:
                # Create CRG node for candidate
                crg_node = CRGNode(
                    entity_id=candidate_entity.entity_id,
                    file_path=candidate_entity.file_path,
                    class_name=candidate_entity.class_name,
                    function_name=candidate_entity.function_name,
                    entity_type=candidate_entity.entity_type,
                    code_snippet=candidate_entity.code_snippet,
                    line_range=candidate_entity.line_range,
                    credibility=0.5,  # Initial credibility
                    initial_strength=0.5
                )
                crg_nodes.append(crg_node)
        
        # Create edges based on paths
        for candidate, paths in paths_by_candidate.items():
            for path in paths:
                for i in range(len(path) - 1):
                    source = path[i]
                    target = path[i + 1]
                    edge_id = f"{source}→{target}"
                    
                    # Get weight if exists
                    weight = edge_weights.get(edge_id, 0.5)
                    
                    edge = CRGEdge(
                        source_id=source,
                        target_id=target,
                        edge_type=EdgeType.REFERENCE,
                        strength=weight,
                        initial_strength=weight
                    )
                    crg_edges.append(edge)
        
        self.logger.info(
            f"Built CRG with {len(crg_nodes)} nodes and {len(crg_edges)} edges"
        )
        
        return CRG(
            nodes=crg_nodes,
            edges=crg_edges,
            paths=compressed_paths,
            issue_desc=issue.description,
            failure_anchors=failure_anchors
        )
    
    def _extract_failure_anchors(self, issue_desc: str) -> List[str]:
        """Extract failure anchors from issue description"""
        
        anchors = []
        
        # Extract error messages
        error_patterns = [
            r"(?:error|exception|failure|failed|wrong|incorrect):\s*([^\n.]+)",
            r"(?:AssertionError|ValueError|TypeError|KeyError):\s*([^\n.]+)",
            r"(?:Traceback|Stack trace).*?(\w+Error)",
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, issue_desc, re.IGNORECASE)
            anchors.extend(matches)
        
        # If no anchors found, extract main keywords
        if not anchors:
            keywords = issue_desc.split()[:10]  # First 10 words
            anchors = keywords
        
        return anchors[:5]  # Limit to 5 anchors
    
    def _enumerate_paths(self, start: str, goal: str, 
                        graph: Dict[str, List[str]]) -> List[List[str]]:
        """Enumerate all paths from start to goal"""
        
        # Use reverse graph (goal -> start)
        reverse_graph = self._build_reverse_graph(graph)
        
        paths = dfs_paths(
            reverse_graph,
            goal,
            start,
            max_depth=self.max_path_depth,
            max_paths=self.max_paths_per_location
        )
        
        # Reverse paths to get start -> goal direction
        reverse_paths = [path[::-1] for path in paths]
        
        return reverse_paths if reverse_paths else [[start, goal]]
    
    def _build_reverse_graph(self, graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Build reverse graph for backward search"""
        
        reverse = {}
        for source, targets in graph.items():
            for target in targets:
                if target not in reverse:
                    reverse[target] = []
                reverse[target].append(source)
        
        return reverse
    
    def _compress_path(self, path: List[str], 
                      entity_map: Dict[str, CodeEntity]) -> PathEvidence:
        """Compress path to structured format"""
        
        nodes = []
        edges = []
        
        for node_id in path:
            if node_id in entity_map:
                entity = entity_map[node_id]
                node = CRGNode(
                    entity_id=entity.entity_id,
                    file_path=entity.file_path,
                    class_name=entity.class_name,
                    function_name=entity.function_name,
                    entity_type=entity.entity_type,
                    code_snippet=entity.code_snippet,
                    line_range=entity.line_range
                )
                
                # Generate semantic summary if not exists
                if not entity.semantic_summary:
                    node.semantic_summary = self.llm.generate_semantic_summary(
                        entity.code_snippet,
                        lines_limit=1
                    )
                else:
                    node.semantic_summary = entity.semantic_summary
                
                nodes.append(node)
            else:
                # Create placeholder node
                node = CRGNode(
                    entity_id=node_id,
                    file_path="unknown",
                    entity_type=EntityType.FILE,
                    semantic_summary=f"Unknown entity: {node_id}"
                )
                nodes.append(node)
        
        # Create edges between consecutive nodes
        for i in range(len(nodes) - 1):
            edge = CRGEdge(
                source_id=nodes[i].entity_id,
                target_id=nodes[i + 1].entity_id,
                edge_type=EdgeType.REFERENCE,
                strength=0.8
            )
            edges.append(edge)
        
        # Build path string
        path_string = " → ".join([n.entity_id for n in nodes])
        
        return PathEvidence(
            nodes=nodes,
            edges=edges,
            failure_anchor=path[-1] if path else "root",
            path_credibility=0.7,
            path_string=path_string
        )
    
    def _compute_initial_weights(self, paths: Dict[str, List[List[str]]],
                                entity_map: Dict[str, CodeEntity]) -> Dict[str, float]:
        """Compute initial edge weights using LLM relative comparison"""
        
        weights = {}
        
        for candidate, path_list in paths.items():
            for path in path_list:
                # For each edge in the path
                for i in range(len(path) - 1):
                    source = path[i]
                    target = path[i + 1]
                    edge_id = f"{source}→{target}"
                    
                    # Get parent edges for relative comparison
                    if i < len(path) - 2:
                        siblings = [path[j] for j in range(len(path)) 
                                   if path[j] != source]
                        
                        if siblings:
                            # Use LLM for relative comparison
                            comparison_objs = [
                                f"{s} (toward failure)" for s in siblings[:3]
                            ]
                            comparison_objs.append(f"{target} (candidate)")
                            
                            # Question: which is more likely on causal path to failure?
                            try:
                                result = self.llm.compare_relative(
                                    comparison_objs,
                                    "Which entity is more likely on the causal path to failure?"
                                )
                                
                                # Use confidence as weight
                                weight = result.get('confidence', 0.5)
                            except:
                                weight = 0.5
                        else:
                            weight = 0.5
                    else:
                        weight = 0.8  # High weight for final edges
                    
                    weights[edge_id] = weight
        
        return weights


class CRG:
    """Causal Relevance Graph object"""
    
    def __init__(self, nodes: List[CRGNode], edges: List[CRGEdge],
                 paths: Dict[str, List[PathEvidence]], issue_desc: str,
                 failure_anchors: List[str]):
        """
        Initialize CRG
        
        Args:
            nodes: List of CRG nodes
            edges: List of CRG edges
            paths: Dict of candidate -> list of paths
            issue_desc: Issue description
            failure_anchors: List of extracted failure anchors
        """
        self.nodes = nodes
        self.edges = edges
        self.paths = paths
        self.issue_desc = issue_desc
        self.failure_anchors = failure_anchors
        
        # Compute edge weights dict
        self.edge_weights = {}
        for edge in edges:
            edge_id = f"{edge.source_id}→{edge.target_id}"
            self.edge_weights[edge_id] = edge.strength
        
        # Compute candidate credibilities
        self.candidate_credibilities = self._compute_candidate_credibilities()
        
        # Extract representative paths
        self.representative_paths = self._extract_representative_paths()
    
    def _compute_candidate_credibilities(self) -> Dict[str, float]:
        """Compute initial credibility for each candidate"""
        
        credibilities = {}
        
        for candidate, path_list in self.paths.items():
            if path_list:
                # Average credibility of all paths for this candidate
                creds = [p.path_credibility for p in path_list]
                credibilities[candidate] = sum(creds) / len(creds)
            else:
                credibilities[candidate] = 0.3
        
        return credibilities
    
    def _extract_representative_paths(self) -> Dict[str, PathEvidence]:
        """Extract representative path for each candidate"""
        
        rep_paths = {}
        
        for candidate, path_list in self.paths.items():
            if path_list:
                # Select path with highest credibility
                rep_path = max(path_list, key=lambda p: p.path_credibility)
                rep_paths[candidate] = rep_path
            else:
                # Create default path
                rep_paths[candidate] = PathEvidence(
                    path_string=candidate,
                    failure_anchor="root",
                    path_credibility=0.3
                )
        
        return rep_paths
    
    def __repr__(self):
        return f"CRG({len(self.nodes)} nodes, {len(self.edges)} edges)"
