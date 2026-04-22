"""Integration tests for SRCD phase (Phase 3)"""

import pytest
from pathlib import Path

from src.common.data_structures import (
    CodeEntity, PathEvidence, IssueContext, EntityType, CRGNode, CRGEdge, EdgeType
)
from src.common.llm_interface import MockLLMInterface
from src.srcd.repair_generator import (
    RepairGenerator, MutationStrategy, TemplateStrategy,
    VariableBinding, MutationType
)
from src.srcd.reflection_scorer import (
    ReflectionScorer, SemanticSimilarityEvaluator,
    CausalRelevanceEvaluator, MinimalityEvaluator
)
from src.srcd.consistency_distiller import (
    ConsistencyDistiller, PatternExtractor, EmbeddingClusterer,
    RepairAggregator
)


@pytest.fixture
def mock_llm():
    """Create mock LLM interface"""
    return MockLLMInterface()


@pytest.fixture
def sample_issue_context():
    """Create sample issue context"""
    return IssueContext(
        id='BUG-001',
        description='NullPointerException when parsing JSON data',
        repo_path='/test/repo',
        candidates=['src/parser.py::parseJSON'],
        test_framework='pytest'
    )


@pytest.fixture
def sample_code():
    """Create sample code snippet"""
    return """def parse_json(data):
    result = json.loads(data)
    return result['key']
"""


@pytest.fixture
def sample_crg_path():
    """Create sample CRG path"""
    node1 = CRGNode(
        entity_id='parse_json',
        file_path='parser.py',
        entity_type=EntityType.FUNCTION,
        credibility=0.8
    )
    node2 = CRGNode(
        entity_id='json.loads',
        file_path='json_lib.py',
        entity_type=EntityType.FUNCTION,
        credibility=0.7
    )
    edge = CRGEdge(
        source_id='parse_json',
        target_id='json.loads',
        edge_type=EdgeType.REFERENCE,
        strength=0.8
    )
    
    return PathEvidence(
        nodes=[node1, node2],
        edges=[edge],
        failure_anchor='JSON parsing failure',
        path_credibility=0.75,
        path_string='parse_json -> json.loads'
    )


class TestRepairGeneratorIntegration:
    """Test repair code generation"""
    
    def test_mutation_strategy_null_checks(self, sample_code):
        """Test null check mutation generation"""
        
        strategy = MutationStrategy()
        
        mutations = strategy.generate_null_checks(
            sample_code,
            ['data', 'result']
        )
        
        # Should generate mutations
        assert len(mutations) > 0
        
        # Check that mutations are different from original
        for mutation in mutations:
            assert mutation != sample_code
            assert 'is not None' in mutation or 'if' in mutation
    
    def test_mutation_strategy_exception_handlers(self, sample_code):
        """Test exception handler mutation generation"""
        
        strategy = MutationStrategy()
        
        mutations = strategy.generate_exception_handlers(sample_code)
        
        # Should generate try-except mutations
        assert len(mutations) > 0
        
        for mutation in mutations:
            assert 'try:' in mutation
            assert 'except' in mutation
    
    def test_template_strategy_applicable_templates(self, sample_code):
        """Test template matching"""
        
        strategy = TemplateStrategy()
        
        templates = strategy.get_applicable_templates(
            'NullPointerException',
            sample_code
        )
        
        # Should find applicable templates
        assert len(templates) > 0
    
    def test_repair_generator_basic_flow(self, mock_llm, sample_code, sample_issue_context):
        """Test basic repair generation flow"""
        
        generator = RepairGenerator(mock_llm)
        
        repairs = generator.generate_repairs(
            code=sample_code,
            bug_location='parseJSON',
            issue_context=sample_issue_context,
            max_mutations=10
        )
        
        # Should generate repairs
        assert len(repairs) > 0
        
        # All repairs should have valid structure
        for repair in repairs:
            assert repair.id is not None
            assert repair.repaired_code != ""
            assert 0 <= repair.confidence <= 1
    
    def test_repair_candidate_structure(self, mock_llm, sample_code, sample_issue_context):
        """Test repair candidate data structure"""
        
        generator = RepairGenerator(mock_llm)
        
        repairs = generator.generate_repairs(
            sample_code, 'parseJSON', sample_issue_context, max_mutations=5
        )
        
        assert repairs[0].original_code == sample_code
        assert repairs[0].mutation_type in list(MutationType)
        assert len(repairs[0].affected_lines) >= 0


class TestReflectionScorerIntegration:
    """Test reflection scoring"""
    
    def test_semantic_similarity_evaluator(self, sample_code, sample_issue_context):
        """Test semantic similarity evaluation"""
        
        evaluator = SemanticSimilarityEvaluator()
        
        score = evaluator.evaluate(sample_code, sample_issue_context.description)
        
        # Score should be in valid range
        assert 0 <= score <= 1
    
    def test_causal_relevance_evaluator(self, sample_code, sample_crg_path):
        """Test causal relevance evaluation"""
        
        evaluator = CausalRelevanceEvaluator()
        
        score = evaluator.evaluate(
            sample_code,
            'parseJSON',
            sample_crg_path,
            'NullPointerException'
        )
        
        # Score should be in valid range
        assert 0 <= score <= 1
    
    def test_minimality_evaluator(self, sample_code):
        """Test minimality evaluation"""
        
        evaluator = MinimalityEvaluator()
        
        # Small change
        modified = sample_code + "# comment"
        score = evaluator.evaluate(modified, sample_code)
        
        # Should be high score for small change
        assert score > 0.5
        assert 0 <= score <= 1
        
        # Large change
        very_different = "def completely_different():\n    pass"
        score2 = evaluator.evaluate(very_different, sample_code)
        
        # Should be lower score for large change
        assert score2 < score
    
    def test_reflection_scorer_integration(
        self,
        mock_llm,
        sample_code,
        sample_issue_context,
        sample_crg_path
    ):
        """Test complete reflection scoring"""
        
        from src.srcd.repair_generator import RepairCandidate
        
        scorer = ReflectionScorer(mock_llm)
        
        # Create sample repair
        repair = RepairCandidate(
            id='test_repair',
            original_code=sample_code,
            repaired_code=sample_code + '\n    if data is not None:',
            mutation_type=MutationType.NULL_CHECK,
            affected_lines=[4],
            confidence=0.8
        )
        
        # Score repair
        score = scorer.score_repair(
            repair, sample_code, sample_issue_context, sample_crg_path
        )
        
        # Check score structure
        assert score.repair_id == 'test_repair'
        assert 0 <= score.semantic_score <= 1
        assert 0 <= score.causal_score <= 1
        assert 0 <= score.minimality_score <= 1
        assert 0 <= score.combined_reflection <= 1


class TestConsistencyDistillerIntegration:
    """Test consistency distillation"""
    
    def test_pattern_extractor(self, sample_code):
        """Test pattern extraction"""
        
        extractor = PatternExtractor()
        
        # Code with null check pattern
        code_with_pattern = """if data is not None:
    result = json.loads(data)
"""
        
        patterns = extractor.extract_patterns(code_with_pattern)
        
        # Should identify at least one pattern
        assert len(patterns) > 0
    
    def test_embedding_clusterer(self, mock_llm):
        """Test repair clustering"""
        
        from src.srcd.repair_generator import RepairCandidate
        
        # Create sample repairs
        repairs = [
            RepairCandidate(
                id=f'repair_{i}',
                original_code='original',
                repaired_code=f'repair variant {i}',
                mutation_type=MutationType.NULL_CHECK,
                affected_lines=[i],
                confidence=0.8
            )
            for i in range(5)
        ]
        
        clusterer = EmbeddingClusterer()
        clusters = clusterer.cluster_repairs(repairs, n_clusters=2)
        
        # Should create clusters
        assert len(clusters) > 0
        
        # All repairs should be in some cluster
        all_clustered = sum(len(c) for c in clusters.values())
        assert all_clustered == len(repairs)
    
    def test_consistency_distiller(
        self,
        mock_llm,
        sample_code,
        sample_issue_context
    ):
        """Test consistency distillation"""
        
        from src.srcd.repair_generator import RepairCandidate, RepairGenerator
        from src.srcd.reflection_scorer import ReflectionScorer
        
        # Generate repairs
        generator = RepairGenerator(mock_llm)
        repairs = generator.generate_repairs(
            sample_code, 'parseJSON', sample_issue_context, max_mutations=5
        )
        
        # Score repairs
        scorer = ReflectionScorer(mock_llm)
        reflection_scores = {}
        
        for repair in repairs:
            score = scorer.score_repair(
                repair, sample_code, sample_issue_context
            )
            reflection_scores[repair.id] = score
        
        # Distill repairs
        distiller = ConsistencyDistiller(mock_llm)
        distilled = distiller.distill_repairs(
            repairs, reflection_scores, n_clusters=2
        )
        
        # Should return distilled repairs
        assert len(distilled) > 0
        
        # Check distilled repair structure
        for repair in distilled:
            assert repair.repair.id is not None
            assert 0 <= repair.distillation_score <= 3  # Can be > 1 due to formula


class TestSRCDEndToEnd:
    """End-to-end SRCD workflow test"""
    
    def test_srcd_complete_workflow(
        self,
        mock_llm,
        sample_code,
        sample_issue_context,
        sample_crg_path
    ):
        """Test complete SRCD pipeline"""
        
        # Step 1: Generate repairs
        generator = RepairGenerator(mock_llm)
        repairs = generator.generate_repairs(
            code=sample_code,
            bug_location='parseJSON',
            issue_context=sample_issue_context,
            crg_path=sample_crg_path,
            max_mutations=10
        )
        
        assert len(repairs) > 0, "Should generate repairs"
        
        # Step 2: Score repairs (reflection)
        scorer = ReflectionScorer(mock_llm)
        reflection_scores = scorer.score_repairs(
            repairs, sample_code, sample_issue_context, sample_crg_path
        )
        
        assert len(reflection_scores) > 0, "Should score all repairs"
        
        # Step 3: Distill repairs (consistency)
        distiller = ConsistencyDistiller(mock_llm)
        distilled = distiller.distill_repairs(
            repairs, reflection_scores, n_clusters=3
        )
        
        assert len(distilled) > 0, "Should distill repairs"
        
        # Step 4: Get top repairs
        top_repairs = distiller.get_top_repairs(distilled, top_k=3)
        
        assert len(top_repairs) <= 3, "Should return at most 3 repairs"
        
        # Verify ordering (highest score first)
        for i in range(len(top_repairs) - 1):
            assert (top_repairs[i].distillation_score >=
                   top_repairs[i+1].distillation_score)
    
    def test_aggregator_statistics(self):
        """Test repair aggregator statistics"""
        
        from src.srcd.consistency_distiller import DistilledRepair
        from src.srcd.repair_generator import RepairCandidate
        
        aggregator = RepairAggregator()
        
        # Create sample distilled repairs
        distilled = [
            DistilledRepair(
                repair=RepairCandidate(
                    id=f'repair_{i}',
                    original_code='original',
                    repaired_code='repaired',
                    mutation_type=MutationType.NULL_CHECK,
                    affected_lines=[],
                    confidence=0.8
                ),
                reflection_score=0.7,
                consensus_score=0.6 + (0.05 * i),
                embedding_similarity=0.8,
                distillation_score=0.65 + (0.05 * i),
                patterns=[],
                confidence=0.75
            )
            for i in range(5)
        ]
        
        # Get statistics
        stats = aggregator.get_repair_statistics(distilled)
        
        assert stats['count'] == 5
        assert 'avg_score' in stats
        assert 'max_score' in stats
        assert 'min_score' in stats
        assert stats['max_score'] >= stats['min_score']


class TestVariableBinding:
    """Test variable binding analysis"""
    
    def test_variable_binding_extraction(self):
        """Test variable binding from code"""
        
        code = """
def process(data):
    value = data['key']
    if value is not None:
        result = str(value)
    return result
"""
        
        binding = VariableBinding(code)
        
        # Should extract variables
        assert len(binding.variables) > 0
        
        # Check specific variables
        assert 'value' in binding.variables or 'data' in binding.variables
    
    def test_safe_modification_check(self):
        """Test safe variable modification check"""
        
        code = "x = 10"
        binding = VariableBinding(code)
        
        # Custom variables should be safe
        if 'x' in binding.variables:
            assert binding.is_variable_safe_to_modify('x')
        
        # Builtins should not be safe
        assert not binding.is_variable_safe_to_modify('print')
