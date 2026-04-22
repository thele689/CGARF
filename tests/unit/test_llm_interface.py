"""Unit tests for LLM interface"""

import json
import pytest
from src.common.llm_interface import (
    AgentType, LLMProvider, MockLLMInterface, create_llm_interface
)


class TestMockLLMInterface:
    """Tests for MockLLMInterface"""
    
    def test_creation(self):
        """Test LLM interface creation"""
        llm = MockLLMInterface(model_name="test-model")
        
        assert llm.model_name == "test-model"
        assert llm.call_count == 0
    
    def test_simple_generation(self):
        """Test simple generation"""
        llm = MockLLMInterface()
        
        result = llm.generate("Hello", temperature=0.7)
        
        assert isinstance(result, str)
        assert llm.call_count == 1
    
    def test_custom_responses(self):
        """Test with custom responses"""
        responses = {
            "compare": "Winner is item 1",
            "summary": "A brief summary"
        }
        llm = MockLLMInterface(responses=responses)
        
        # Should return custom response if key matches
        result = llm.generate("Please compare these items")
        assert "Winner is item 1" in result or result == "Mock response placeholder"
    
    def test_multiple_calls(self):
        """Test multiple calls"""
        llm = MockLLMInterface()
        
        llm.generate("Call 1")
        llm.generate("Call 2")
        llm.generate("Call 3")
        
        assert llm.call_count == 3
    
    def test_schema_generation(self):
        """Test generation with schema"""
        llm = MockLLMInterface()
        
        schema = {
            "result": str,
            "score": float
        }
        
        result = llm.generate_with_schema("Generate JSON", schema)
        
        assert isinstance(result, dict)
    
    def test_get_stats(self):
        """Test getting statistics"""
        llm = MockLLMInterface()
        
        llm.generate("Test 1")
        llm.generate("Test 2")
        
        stats = llm.get_stats()
        
        assert stats['total_calls'] == 2
        assert 'token_stats' in stats


class TestCompareRelative:
    """Tests for relative comparison"""
    
    def test_compare_two_items(self):
        """Test comparing two items"""
        llm = MockLLMInterface()
        
        items = ["Path A -> B -> C", "Path D -> E -> F"]
        question = "Which path better explains the failure?"
        
        # This should work (will return mock response)
        result = llm.compare_relative(items, question)
        
        assert isinstance(result, dict)


class TestSemanticSummary:
    """Tests for semantic summary generation"""
    
    def test_generate_summary(self):
        """Test summary generation"""
        llm = MockLLMInterface()
        
        code = """
def parse_config(filename):
    with open(filename) as f:
        return json.load(f)
"""
        
        summary = llm.generate_semantic_summary(code, lines_limit=2)
        
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestAgentDebate:
    """Tests for agent debate"""
    
    def test_support_agent(self):
        """Test support agent"""
        llm = MockLLMInterface()
        
        issue = "Function returns wrong value"
        path = "parse_config -> load_default -> apply_default"
        
        result = llm.agent_debate(AgentType.SUPPORT, issue, path)
        
        assert isinstance(result, dict)
    
    def test_oppose_agent(self):
        """Test oppose agent"""
        llm = MockLLMInterface()
        
        issue = "Function returns wrong value"
        path = "parse_config -> load_default -> apply_default"
        
        result = llm.agent_debate(AgentType.OPPOSE, issue, path)
        
        assert isinstance(result, dict)
    
    def test_judge_agent(self):
        """Test judge agent"""
        llm = MockLLMInterface()
        
        issue = "Function returns wrong value"
        path = "parse_config -> load_default -> apply_default"
        
        context = {
            'support': "This path clearly explains the failure",
            'oppose': "This path has weak connections"
        }
        
        result = llm.agent_debate(AgentType.JUDGE, issue, path, context=context)
        
        assert isinstance(result, dict)


class TestPatchGeneration:
    """Tests for patch generation"""
    
    def test_generate_patch(self):
        """Test patch generation"""
        llm = MockLLMInterface()
        
        issue = "Wrong output format"
        code = "def format_output(data): return str(data)"
        path = "format_output -> json_encode"
        
        patch = llm.generate_patch(issue, code, path, temperature=0.7, round=0)
        
        assert isinstance(patch, str)
        assert len(patch) > 0


class TestReflectionEvaluation:
    """Tests for reflection evaluation"""
    
    def test_evaluate_reflection(self):
        """Test reflection evaluation"""
        llm = MockLLMInterface()
        
        issue = "Wrong output"
        code = "def func(): pass"
        path = "node1 -> node2"
        patch = "<<<SEARCH\nold\n===\nnew\n>>>REPLACE"
        
        result = llm.evaluate_reflection(issue, code, path, patch)
        
        assert isinstance(result, dict)


class TestConsensusExtraction:
    """Tests for consensus pattern extraction"""
    
    def test_extract_consensus(self):
        """Test consensus extraction"""
        llm = MockLLMInterface()
        
        patches = [
            "<<<SEARCH\nif x:\n===\nif x and y:\n>>>REPLACE",
            "<<<SEARCH\nif x:\n===\nif x or z:\n>>>REPLACE",
            "<<<SEARCH\nif x:\n===\nif x and z:\n>>>REPLACE"
        ]
        
        pattern = llm.extract_consensus_pattern(patches)
        
        assert isinstance(pattern, str)


class TestLLMFactory:
    """Tests for LLM factory function"""
    
    def test_create_mock_llm(self):
        """Test creating mock LLM"""
        llm = create_llm_interface("mock", "test-model")
        
        assert isinstance(llm, MockLLMInterface)
        assert llm.model_name == "test-model"
    
    def test_invalid_provider(self):
        """Test invalid provider raises error"""
        with pytest.raises(ValueError):
            create_llm_interface("invalid", "model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
