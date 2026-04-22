"""Unit tests for data structures"""

import pytest
import time
from src.common.data_structures import (
    EntityType, EdgeType, CodeEntity, CodeEdge, CRGNode, CRGEdge,
    PathEvidence, PatchCandidate, ReflectionScore, PatchBatch,
    IssueContext, RepairResult
)


class TestCodeEntity:
    """Tests for CodeEntity class"""
    
    def test_creation(self):
        """Test CodeEntity creation"""
        entity = CodeEntity(
            entity_id="file.py::MyClass::my_func",
            file_path="src/file.py",
            class_name="MyClass",
            function_name="my_func",
            entity_type=EntityType.FUNCTION,
            code_snippet="def my_func(): pass",
            line_range=(10, 15)
        )
        
        assert entity.entity_id == "file.py::MyClass::my_func"
        assert entity.file_path == "src/file.py"
        assert entity.entity_type == EntityType.FUNCTION
    
    def test_hash(self):
        """Test CodeEntity hashing"""
        entity1 = CodeEntity(
            entity_id="test_id",
            file_path="test.py",
            entity_type=EntityType.FUNCTION
        )
        entity2 = CodeEntity(
            entity_id="test_id",
            file_path="test.py",
            entity_type=EntityType.FUNCTION
        )
        
        assert hash(entity1) == hash(entity2)
    
    def test_equality(self):
        """Test CodeEntity equality"""
        entity1 = CodeEntity(
            entity_id="test_id",
            file_path="test.py",
            entity_type=EntityType.FUNCTION
        )
        entity2 = CodeEntity(
            entity_id="test_id",
            file_path="different.py",
            entity_type=EntityType.CLASS
        )
        
        assert entity1 == entity2  # Based on entity_id


class TestPathEvidence:
    """Tests for PathEvidence class"""
    
    def test_creation(self):
        """Test PathEvidence creation"""
        node1 = CRGNode(
            entity_id="node1",
            file_path="test.py",
            entity_type=EntityType.FUNCTION
        )
        node2 = CRGNode(
            entity_id="node2",
            file_path="test.py",
            entity_type=EntityType.FUNCTION
        )
        
        edge = CRGEdge(
            source_id="node1",
            target_id="node2",
            edge_type=EdgeType.REFERENCE,
            strength=0.8
        )
        
        path = PathEvidence(
            nodes=[node1, node2],
            edges=[edge],
            failure_anchor="failure_point",
            path_credibility=0.75,
            path_string="node1 -> node2"
        )
        
        assert len(path.nodes) == 2
        assert len(path.edges) == 1
        assert path.path_credibility == 0.75


class TestPatchCandidate:
    """Tests for PatchCandidate class"""
    
    def test_creation(self):
        """Test PatchCandidate creation"""
        patch = PatchCandidate(
            patch_id="patch_1",
            location="src/file.py::func",
            patch_content="<<<SEARCH\nold\n===\nnew\n>>>REPLACE",
            generated_round=0,
            credibility_from_location=0.85
        )
        
        assert patch.patch_id == "patch_1"
        assert patch.generated_round == 0
        assert patch.final_score == 0.0  # Initially 0


class TestReflectionScore:
    """Tests for ReflectionScore class"""
    
    def test_creation(self):
        """Test ReflectionScore creation"""
        score = ReflectionScore(
            semantic_consistency=0.8,
            causal_alignment=0.9,
            minimal_edit=0.7,
            combined_score=0.8
        )
        
        assert score.semantic_consistency == 0.8
        assert score.combined_score == 0.8
        assert isinstance(score.timestamp, float)
        assert score.timestamp > 0


class TestPatchBatch:
    """Tests for PatchBatch class"""
    
    def test_creation(self):
        """Test PatchBatch creation"""
        patch1 = PatchCandidate(
            patch_id="p1",
            location="loc1",
            patch_content="patch",
            generated_round=0,
            credibility_from_location=0.8
        )
        patch2 = PatchCandidate(
            patch_id="p2",
            location="loc1",
            patch_content="patch2",
            generated_round=1,
            credibility_from_location=0.8
        )
        
        batch = PatchBatch(
            location="loc1",
            candidates=[patch1, patch2]
        )
        
        assert len(batch.candidates) == 2
        assert len(batch) == 2


class TestIssueContext:
    """Tests for IssueContext class"""
    
    def test_creation(self):
        """Test IssueContext creation"""
        issue = IssueContext(
            id="issue_123",
            description="Bug description",
            repo_path="/path/to/repo",
            candidates=["file.py::func1", "file.py::func2"]
        )
        
        assert issue.id == "issue_123"
        assert len(issue.candidates) == 2


class TestRepairResult:
    """Tests for RepairResult class"""
    
    def test_success(self):
        """Test successful repair result"""
        patch = PatchCandidate(
            patch_id="p1",
            location="loc1",
            patch_content="patch",
            generated_round=0,
            credibility_from_location=0.9
        )
        
        result = RepairResult(
            issue_id="issue_1",
            patch=patch,
            success=True,
            confidence=0.92
        )
        
        assert result.success is True
        assert result.patch is not None
        assert result.confidence == 0.92
    
    def test_failure(self):
        """Test failed repair result"""
        result = RepairResult(
            issue_id="issue_1",
            patch=None,
            success=False,
            confidence=0.0
        )
        
        assert result.success is False
        assert result.patch is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
