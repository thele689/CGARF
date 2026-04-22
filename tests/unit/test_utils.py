"""Unit tests for utility functions"""

import pytest
import numpy as np
from src.common.utils import (
    set_seed, dfs_paths, get_subgraph, cosine_similarity,
    edit_distance, normalized_similarity, extract_function_calls,
    extract_variables, validate_patch_format, MetricsCounter
)


class TestSeeding:
    """Tests for random seed management"""
    
    def test_set_seed(self):
        """Test setting seed"""
        set_seed(42)
        
        val1 = np.random.random()
        
        set_seed(42)
        
        val2 = np.random.random()
        
        assert val1 == val2


class TestDFSPaths:
    """Tests for DFS path finding"""
    
    def test_simple_path(self):
        """Test finding simple path"""
        graph = {
            "A": ["B"],
            "B": ["C"],
            "C": []
        }
        
        paths = dfs_paths(graph, "A", "C")
        
        assert len(paths) > 0
        assert ["A", "B", "C"] in paths
    
    def test_multiple_paths(self):
        """Test finding multiple paths"""
        graph = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": []
        }
        
        paths = dfs_paths(graph, "A", "D")
        
        assert len(paths) >= 2
    
    def test_no_path(self):
        """Test when no path exists"""
        graph = {
            "A": ["B"],
            "C": ["D"]
        }
        
        paths = dfs_paths(graph, "A", "D")
        
        assert len(paths) == 0
    
    def test_max_depth(self):
        """Test max depth limit"""
        graph = {
            "A": ["B"],
            "B": ["C"],
            "C": ["D"],
            "D": ["E"]
        }
        
        paths = dfs_paths(graph, "A", "E", max_depth=2)
        
        # Should not find path if depth is too limited
        assert len(paths) == 0


class TestGetSubgraph:
    """Tests for subgraph extraction"""
    
    def test_extract_subgraph(self):
        """Test extracting subgraph"""
        graph = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": ["E"],
            "E": []
        }
        
        nodes = {"A", "B", "D"}
        subgraph = get_subgraph(nodes, graph)
        
        assert "A" in subgraph
        assert "B" in subgraph or "D" in subgraph


class TestSimilarityMetrics:
    """Tests for similarity metrics"""
    
    def test_cosine_similarity(self):
        """Test cosine similarity"""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        
        sim = cosine_similarity(vec1, vec2)
        
        assert sim == 1.0
    
    def test_cosine_similarity_orthogonal(self):
        """Test orthogonal vectors"""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        
        sim = cosine_similarity(vec1, vec2)
        
        assert sim == 0.0
    
    def test_edit_distance(self):
        """Test edit distance"""
        dist = edit_distance("kitten", "sitting")
        
        assert dist == 3
    
    def test_edit_distance_identical(self):
        """Test identical strings"""
        dist = edit_distance("hello", "hello")
        
        assert dist == 0
    
    def test_normalized_similarity(self):
        """Test normalized similarity"""
        sim = normalized_similarity("hello", "hello")
        
        assert sim == 1.0
    
    def test_normalized_similarity_different(self):
        """Test different strings"""
        sim = normalized_similarity("cat", "dog")
        
        assert 0 <= sim <= 1


class TestCodeAnalysis:
    """Tests for code analysis functions"""
    
    def test_extract_function_calls(self):
        """Test extracting function calls"""
        code = "result = foo(x) + bar(y, z)"
        
        calls = extract_function_calls(code)
        
        assert "foo" in calls
        assert "bar" in calls
    
    def test_extract_function_calls_no_keywords(self):
        """Test that keywords are filtered"""
        code = "if condition: pass"
        
        calls = extract_function_calls(code)
        
        assert "if" not in calls
    
    def test_extract_variables(self):
        """Test extracting variables"""
        code = "x = 10; y = x + z"
        
        variables = extract_variables(code)
        
        assert "x" in variables
        assert "y" in variables
        assert "z" in variables


class TestValidation:
    """Tests for validation functions"""
    
    def test_validate_patch_format_valid(self):
        """Test valid patch format"""
        patch = """<<<SEARCH
old code
===
new code
>>>REPLACE"""
        
        is_valid = validate_patch_format(patch)
        
        assert is_valid is True
    
    def test_validate_patch_format_invalid(self):
        """Test invalid patch format"""
        patch = "just some code"
        
        is_valid = validate_patch_format(patch)
        
        assert is_valid is False


class TestMetricsCounter:
    """Tests for metrics counter"""
    
    def test_increment(self):
        """Test incrementing metric"""
        counter = MetricsCounter()
        
        counter.increment("test_metric")
        counter.increment("test_metric", 5.0)
        
        assert counter.get("test_metric") == 6.0
    
    def test_set(self):
        """Test setting metric"""
        counter = MetricsCounter()
        
        counter.set("metric1", 42.0)
        
        assert counter.get("metric1") == 42.0
    
    def test_get_all(self):
        """Test getting all metrics"""
        counter = MetricsCounter()
        
        counter.set("m1", 1.0)
        counter.set("m2", 2.0)
        
        all_metrics = counter.get_all()
        
        assert all_metrics["m1"] == 1.0
        assert all_metrics["m2"] == 2.0
    
    def test_reset(self):
        """Test resetting metrics"""
        counter = MetricsCounter()
        
        counter.increment("metric")
        counter.reset()
        
        assert counter.get("metric") == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
