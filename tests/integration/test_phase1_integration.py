"""Integration tests for Phase 1 framework"""

import pytest
from src.common.data_structures import (
    IssueContext, CodeEntity, EntityType, PathEvidence,
    CRGNode, CRGEdge, EdgeType, PatchBatch, PatchCandidate
)
from src.common.llm_interface import MockLLMInterface, create_llm_interface
from src.common.utils import (
    set_seed, dfs_paths, cosine_similarity,
    normalize_similarity
)
import numpy as np


class TestPhase1Integration:
    """Integration tests for Phase 1"""
    
    def test_full_data_structure_workflow(self):
        """Test complete workflow with data structures"""
        
        # Create an issue
        issue = IssueContext(
            id="test_issue_1",
            description="Function returns incorrect value",
            repo_path="/test/repo",
            candidates=["src/parser.py::parse_config", "src/utils.py::apply_default"]
        )
        
        # Create code entities
        entity1 = CodeEntity(
            entity_id="src/parser.py::parse_config",
            file_path="src/parser.py",
            function_name="parse_config",
            entity_type=EntityType.FUNCTION,
            code_snippet="def parse_config(f): return json.load(f)",
            line_range=(10, 12)
        )
        
        entity2 = CodeEntity(
            entity_id="src/utils.py::apply_default",
            file_path="src/utils.py",
            function_name="apply_default",
            entity_type=EntityType.FUNCTION,
            code_snippet="def apply_default(config): return config or {}",
            line_range=(5, 7)
        )
        
        # Create path evidence
        node1 = CRGNode(
            entity_id="src/parser.py::parse_config",
            file_path="src/parser.py",
            function_name="parse_config",
            entity_type=EntityType.FUNCTION,
            credibility=0.8
        )
        
        node2 = CRGNode(
            entity_id="src/utils.py::apply_default",
            file_path="src/utils.py",
            function_name="apply_default",
            entity_type=EntityType.FUNCTION,
            credibility=0.7
        )
        
        edge = CRGEdge(
            source_id="src/parser.py::parse_config",
            target_id="src/utils.py::apply_default",
            edge_type=EdgeType.REFERENCE,
            strength=0.8,
            initial_strength=0.8
        )
        
        path = PathEvidence(
            nodes=[node1, node2],
            edges=[edge],
            failure_anchor="test.json loading",
            path_credibility=0.75,
            path_string="parse_config -> apply_default -> json"
        )
        
        # Create patches
        patch1 = PatchCandidate(
            patch_id="patch_1_0",
            location="src/parser.py::parse_config",
            patch_content="<<<SEARCH\nreturn json.load(f)\n===\nreturn json.load(f) or {}\n>>>REPLACE",
            generated_round=0,
            credibility_from_location=0.85
        )
        
        patch2 = PatchCandidate(
            patch_id="patch_1_1",
            location="src/parser.py::parse_config",
            patch_content="<<<SEARCH\ndef parse_config(f):\n===\ndef parse_config(f, default=None):\n>>>REPLACE",
            generated_round=1,
            credibility_from_location=0.85
        )
        
        # Create patch batch
        batch = PatchBatch(
            location="src/parser.py::parse_config",
            candidates=[patch1, patch2]
        )
        
        assert len(batch.candidates) == 2
        assert batch.location == "src/parser.py::parse_config"
        assert len(path.nodes) == 2
    
    def test_llm_integration(self):
        """Test LLM interface integration"""
        
        # Create LLM interface
        llm = create_llm_interface("mock", "test-model")
        
        # Test basic operations
        summary = llm.generate_semantic_summary("def foo(): pass", lines_limit=1)
        assert isinstance(summary, str)
        
        # Test agent debate
        from src.common.llm_interface import AgentType
        
        issue = "Wrong output"
        path = "node1 -> node2"
        
        support_result = llm.agent_debate(AgentType.SUPPORT, issue, path)
        assert isinstance(support_result, dict)
        
        oppose_result = llm.agent_debate(AgentType.OPPOSE, issue, path)
        assert isinstance(oppose_result, dict)
        
        # Test patch generation
        patch = llm.generate_patch(issue, "code", path)
        assert isinstance(patch, str)
    
    def test_utility_integration(self):
        """Test utility functions integration"""
        
        # Set seed for reproducibility
        set_seed(42)
        
        val1 = np.random.random()
        
        set_seed(42)
        
        val2 = np.random.random()
        
        assert val1 == val2
        
        # Test DFS paths
        graph = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": []
        }
        
        paths = dfs_paths(graph, "A", "D")
        assert len(paths) >= 2
        
        # Test similarity metrics
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        
        sim = cosine_similarity(vec1, vec2)
        assert sim == 1.0
    
    def test_configuration_loading(self):
        """Test loading configuration"""
        from src.common.utils import load_yaml
        import os
        
        config_path = "config/defaults.yaml"
        
        if os.path.exists(config_path):
            config = load_yaml(config_path)
            
            assert "cgarf" in config
            assert "cg_mad" in config["cgarf"]
            assert "srcd" in config["cgarf"]
            assert "tspf" in config["cgarf"]


class TestDataValidation:
    """Tests for data validation"""
    
    def test_issue_context_validation(self):
        """Test IssueContext validation"""
        
        issue = IssueContext(
            id="test",
            description="Test issue",
            repo_path="/repo",
            candidates=["loc1", "loc2"]
        )
        
        assert len(issue.candidates) == 2
        assert issue.timeout_seconds == 120
    
    def test_patch_candidate_validation(self):
        """Test PatchCandidate validation"""
        
        patch = PatchCandidate(
            patch_id="p1",
            location="loc1",
            patch_content="content",
            generated_round=0,
            credibility_from_location=0.8
        )
        
        assert patch.final_score == 0.0  # Initially 0
        assert patch.test_result is None  # Initially None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
