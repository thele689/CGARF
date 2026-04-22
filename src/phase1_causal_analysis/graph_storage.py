"""
Graph Storage and Serialization
===============================

Handles the persistence of large repository CodeGraphs and instance-specific
local Causal Relevance Graphs (CRG). Crucial for processing 300 instances 
efficiently by avoiding redundant AST parsing.
"""
import dataclasses
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from .code_graph_builder import CODE_GRAPH_SCHEMA_VERSION
from .causal_relevance_graph import (
    CodeGraph, CodeEntity, CodeRelation, EntityType, RelationType,
    CausalRelevanceGraph, FailureEvidence, CRGEdge
)

class GraphEncoder(json.JSONEncoder):
    """Custom JSON Encoder for Dataclasses and Enums"""
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (EntityType, RelationType)):
            return obj.value
        return super().default(obj)

class StorageManager:
    def __init__(self, base_dir: str = "data/code_graphs"):
        self.base_dir = Path(base_dir)
        self.repo_dir = self.base_dir / "repos"
        self.crg_dir = self.base_dir / "crgs"
        
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.crg_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized GraphStorage at {self.base_dir}")

    def save_code_graph(self, repo_id: str, commit_sha: str, graph: CodeGraph) -> str:
        """Serialize and save full repository CodeGraph"""
        safe_repo_id = repo_id.replace("/", "_")
        file_path = self.repo_dir / f"{safe_repo_id}_{commit_sha}_cg.json"
        
        data = {
            "metadata": {
                "graph_kind": "code_graph",
                "schema_version": CODE_GRAPH_SCHEMA_VERSION,
                "entity_count": len(graph.entities),
                "relation_count": len(graph.relations),
            },
            "entities": {k: dataclasses.asdict(v) for k, v in graph.entities.items() },
            "relations": [dataclasses.asdict(r) for r in graph.relations],
        }
        
        # Serialize enums properly
        for entity in data["entities"].values():
            entity["entity_type"] = entity["entity_type"].value
        for rel in data["relations"]:
            rel["relation_type"] = rel["relation_type"].value

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved CodeGraph to {file_path}")
        return str(file_path)

    def load_code_graph(self, repo_id: str, commit_sha: str) -> Optional[CodeGraph]:
        """Load repository CodeGraph from disk if exists"""
        safe_repo_id = repo_id.replace("/", "_")
        file_path = self.repo_dir / f"{safe_repo_id}_{commit_sha}_cg.json"
        
        if not file_path.exists():
            return None
            
        logger.info(f"Found cached CodeGraph at {file_path}, loading...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        schema_version = metadata.get("schema_version")
        if schema_version != CODE_GRAPH_SCHEMA_VERSION:
            logger.warning(
                "Cached CodeGraph schema mismatch at {} (found: {}, expected: {}). "
                "Rebuild required.".format(file_path, schema_version, CODE_GRAPH_SCHEMA_VERSION)
            )
            return None
            
        cg = CodeGraph()
        
        for e_id, e_dict in data["entities"].items():
            e_dict["entity_type"] = EntityType(e_dict["entity_type"])
            cg.add_entity(CodeEntity(**e_dict))
            
        for r_dict in data["relations"]:
            r_dict["relation_type"] = RelationType(r_dict["relation_type"])
            r_dict.setdefault("metadata", {})
            cg.add_relation(CodeRelation(**r_dict))

        cg.rebuild_graph()
        return cg
        
    def save_crg(self, instance_id: str, crg: CausalRelevanceGraph) -> str:
        """Save local Causal Relevance Graph with weights"""
        file_path = self.crg_dir / f"{instance_id}_crg.json"
        
        edge_data = []
        for e in crg.edges.values():  # crg.edges is a Dict, iterate over values
            edge_dict = dataclasses.asdict(e)
            edge_dict["relation_type"] = edge_dict["relation_type"].value
            edge_data.append(edge_dict)

        root_nodes = []
        for root_node in crg.root_nodes.values():
            root_dict = dataclasses.asdict(root_node)
            root_dict["entity_type"] = root_dict["entity_type"].value
            root_nodes.append(root_dict)

        data = {
            "instance_id": instance_id,
            "edges": edge_data,
            "failure_evidences": [dataclasses.asdict(fe) for fe in crg.failure_evidences],
            "root_nodes": root_nodes,
            "candidate_leaf_ids": sorted(crg.candidate_leaf_ids),
            "anchor_entity_ids": sorted(crg.anchor_entity_ids),
            "paths_by_candidate": dict(crg.paths_by_candidate),
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.debug(f"Saved CRG for {instance_id} to {file_path}")
        return str(file_path)

    def load_crg(self, instance_id: str, code_graph: CodeGraph) -> Optional[CausalRelevanceGraph]:
        """Load a saved CRG from disk and rebind it to an executable CodeGraph."""

        file_path = self.crg_dir / f"{instance_id}_crg.json"
        if not file_path.exists():
            return None

        logger.info(f"Found cached CRG at {file_path}, loading...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        failure_evidences = [FailureEvidence(**item) for item in data.get("failure_evidences", [])]
        crg = CausalRelevanceGraph(code_graph=code_graph, failure_evidences=failure_evidences)

        for root_dict in data.get("root_nodes", []):
            root_payload = dict(root_dict)
            root_payload["entity_type"] = EntityType(root_payload["entity_type"])
            crg.add_root_node(CodeEntity(**root_payload))

        for edge_dict in data.get("edges", []):
            edge_payload = dict(edge_dict)
            edge_payload["relation_type"] = RelationType(edge_payload["relation_type"])
            crg.add_edge(CRGEdge(**edge_payload))

        crg.candidate_leaf_ids.update(data.get("candidate_leaf_ids", []))
        crg.anchor_entity_ids.update(data.get("anchor_entity_ids", []))

        for candidate_id, paths in data.get("paths_by_candidate", {}).items():
            for path in paths:
                crg.register_path(candidate_id, list(path))

        return crg

if __name__ == "__main__":
    StorageManager()
