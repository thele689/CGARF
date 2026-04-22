"""
Phase 5 Integration Tests: Pipeline & Evaluation
=================================================

Tests for complete pipeline orchestration and evaluation metrics
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from src.pipeline.pipeline_orchestrator import (
    CGARFPipeline, PipelineConfig, PipelineFactory, PipelineStage
)
from src.evaluation.evaluation_suite import (
    MetricsCalculator, ReportGenerator, PerformanceAnalyzer,
    BenchmarkComparison, MetricsSummary
)
from src.common.data_structures import IssueContext, CodeEntity
from src.tspf.patch_filter import VerifiedPatch, TestSuiteResult, TestStatus


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def pipeline_config():
    """Sample pipeline configuration"""
    return PipelineConfig(
        llm_provider="mock",
        log_level="WARNING",
        phase2_agent_iterations=2,
        phase3_max_repairs=30,
        phase4_max_patches=5
    )


@pytest.fixture
def mock_verified_patch():
    """Mock verified patch for testing"""
    from src.common.data_structures import RepairCandidate
    
    repair = RepairCandidate(
        id="test_repair",
        original_code="def f(x):\n    return x.upper()",
        repaired_code="def f(x):\n    if x: return x.upper()\n    return ''",
        mutation_type="FALLBACK",
        affected_lines=[2],
        confidence=0.85
    )
    
    test_result = TestSuiteResult(
        repair_id="test_repair",
        test_results=[],
        total_tests=5,
        passed_tests=4,
        failed_tests=1,
        error_tests=0
    )
    
    return VerifiedPatch(
        repair=repair,
        test_results=test_result,
        verification_score=0.82,
        confidence=0.80
    )


# ============================================================================
# PipelineConfig Tests
# ============================================================================

class TestPipelineConfig:
    """Tests for PipelineConfig"""

    def test_config_defaults(self):
        """Test default configuration values"""
        config = PipelineConfig()
        assert config.llm_provider == "openai"
        assert config.pipeline_timeout == 300
        assert config.phase4_max_patches == 5

    def test_config_from_yaml(self, tmp_path):
        """Test loading configuration from YAML"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
pipeline:
  llm_provider: mock
  timeout: 600
  
phase4:
  max_patches: 10
""")
        
        config = PipelineConfig.from_yaml(str(config_file))
        assert config.llm_provider == "mock"
        assert config.pipeline_timeout == 600
        assert config.phase4_max_patches == 10


# ============================================================================
# CGARFPipeline Tests
# ============================================================================

class TestCGARFPipeline:
    """Tests for main pipeline"""

    def test_pipeline_initialization(self, pipeline_config):
        """Test pipeline initialization"""
        pipeline = CGARFPipeline(pipeline_config)
        assert pipeline.config == pipeline_config
        assert pipeline.llm is not None
        assert pipeline.start_time == 0.0

    def test_pipeline_run_integration(self, pipeline_config):
        """Test complete pipeline execution"""
        pipeline = CGARFPipeline(pipeline_config)
        
        # This would fail without full dependencies, so we mock Phase 2+
        patches = pipeline.run(
            buggy_code="def process(x):\n    return x.strip()",
            issue_description="NoneType error when x is None"
        )
        
        # At minimum, should return list (may be empty due to mocking)
        assert isinstance(patches, list)

    def test_pipeline_stage_recording(self, pipeline_config):
        """Test that pipeline records stage execution"""
        pipeline = CGARFPipeline(pipeline_config)
        assert len(pipeline.stages) == 0
        
        # Manually add a stage
        stage = PipelineStage("Test Stage", 100.0)
        pipeline.stages.append(stage)
        
        assert len(pipeline.stages) == 1
        assert pipeline.stages[0].stage_name == "Test Stage"

    def test_get_execution_summary(self, pipeline_config):
        """Test execution summary generation"""
        pipeline = CGARFPipeline(pipeline_config)
        
        stage1 = PipelineStage("Phase 1", 10.0)
        stage1.end_time = 11.0
        stage1.duration = 1.0
        stage1.status = "completed"
        
        stage2 = PipelineStage("Phase 2", 11.0)
        stage2.end_time = 14.0
        stage2.duration = 3.0
        stage2.status = "completed"
        
        pipeline.stages = [stage1, stage2]
        summary = pipeline.get_execution_summary()
        
        assert "Phase 1" in summary
        assert "Phase 2" in summary
        assert "1.0" in summary or "1.00" in summary


class TestPipelineFactory:
    """Tests for pipeline factory"""

    def test_create_default(self):
        """Test creating default pipeline"""
        try:
            pipeline = PipelineFactory.create_default()
            assert pipeline is not None
            assert pipeline.config.llm_provider == "openai"
        except ImportError as e:
            if "openai" in str(e):
                # OpenAI not installed, use mock instead for testing
                pipeline = PipelineFactory.create_mock()
                assert pipeline is not None
            else:
                raise

    def test_create_mock(self):
        """Test creating mock pipeline"""
        pipeline = PipelineFactory.create_mock()
        assert pipeline is not None
        assert pipeline.config.llm_provider == "mock"


# ============================================================================
# MetricsCalculator Tests
# ============================================================================

class TestMetricsCalculator:
    """Tests for metrics calculation"""

    def test_add_patch(self, mock_verified_patch):
        """Test adding patches to calculator"""
        calc = MetricsCalculator()
        assert len(calc.patches_evaluated) == 0
        
        calc.add_patch(mock_verified_patch, 2.5)
        assert len(calc.patches_evaluated) == 1
        assert 2.5 in calc.execution_times

    def test_repair_rate_calculation(self, mock_verified_patch):
        """Test repair rate calculation"""
        calc = MetricsCalculator()
        calc.add_patch(mock_verified_patch)
        
        rate = calc.calculate_repair_rate(total_bugs=10)
        assert 0 <= rate <= 1
        assert rate == 0.1  # 1 repaired out of 10

    def test_patch_quality_metrics(self, mock_verified_patch):
        """Test patch quality metrics"""
        calc = MetricsCalculator()
        calc.add_patch(mock_verified_patch)
        
        avg_score, avg_conf, std = calc.calculate_patch_quality_metrics()
        assert avg_score == mock_verified_patch.verification_score
        assert avg_conf == mock_verified_patch.confidence
        assert std == 0.0  # Only one patch

    def test_code_similarity(self, mock_verified_patch):
        """Test code similarity calculation"""
        calc = MetricsCalculator()
        similarity = calc.calculate_code_similarity(mock_verified_patch)
        
        assert 0 <= similarity <= 1
        # Similar code should have high similarity
        assert similarity > 0.5

    def test_complexity_delta(self, mock_verified_patch):
        """Test complexity delta calculation"""
        calc = MetricsCalculator()
        delta = calc.calculate_complexity_delta(mock_verified_patch)
        
        # Should be a number
        assert isinstance(delta, (int, float))

    def test_verified_patches_filter(self, mock_verified_patch):
        """Test getting verified patches"""
        calc = MetricsCalculator()
        calc.add_patch(mock_verified_patch)  # pass_rate = 0.8
        
        verified = calc.get_verified_patches()
        assert len(verified) == 1

    def test_high_quality_patches_filter(self, mock_verified_patch):
        """Test getting high quality patches"""
        calc = MetricsCalculator()
        calc.add_patch(mock_verified_patch)  # verification_score = 0.82
        
        high_quality = calc.get_high_quality_patches()
        assert len(high_quality) == 1

    def test_generate_summary(self, mock_verified_patch):
        """Test summary generation"""
        calc = MetricsCalculator()
        calc.add_patch(mock_verified_patch, 2.0)
        calc.add_patch(mock_verified_patch, 2.5)
        
        summary = calc.generate_summary(total_bugs=10)
        assert summary.total_bugs == 10
        assert summary.successfully_repaired == 2
        assert summary.repair_rate == 0.2


# ============================================================================
# ReportGenerator Tests
# ============================================================================

class TestReportGenerator:
    """Tests for report generation"""

    def test_json_report_generation(self, tmp_path, mock_verified_patch):
        """Test JSON report generation"""
        calc = MetricsCalculator()
        calc.add_patch(mock_verified_patch)
        
        gen = ReportGenerator(calc)
        report_file = tmp_path / "report.json"
        
        gen.generate_json_report(str(report_file))
        assert report_file.exists()

    def test_html_report_generation(self, tmp_path, mock_verified_patch):
        """Test HTML report generation"""
        calc = MetricsCalculator()
        calc.add_patch(mock_verified_patch)
        
        gen = ReportGenerator(calc)
        report_file = tmp_path / "report.html"
        
        gen.generate_html_report(str(report_file))
        assert report_file.exists()
        
        content = report_file.read_text()
        assert "CGARF" in content
        assert "Evaluation Report" in content

    def test_text_report_generation(self, mock_verified_patch):
        """Test text report generation"""
        calc = MetricsCalculator()
        calc.add_patch(mock_verified_patch)
        
        gen = ReportGenerator(calc)
        report = gen.generate_text_report()
        
        assert "CGARF" in report
        assert "Repair" in report


# ============================================================================
# PerformanceAnalyzer Tests
# ============================================================================

class TestPerformanceAnalyzer:
    """Tests for performance analysis"""

    def test_add_stage_time(self):
        """Test adding stage timing data"""
        analyzer = PerformanceAnalyzer()
        
        analyzer.add_stage_time("Phase 1", 0.5)
        analyzer.add_stage_time("Phase 2", 1.5)
        analyzer.add_stage_time("Phase 2", 1.2)  # Multiple measurements
        
        assert "Phase 1" in analyzer.stage_times
        assert len(analyzer.stage_times["Phase 2"]) == 2

    def test_stage_statistics(self):
        """Test stage statistics calculation"""
        analyzer = PerformanceAnalyzer()
        
        analyzer.add_stage_time("Phase 1", 1.0)
        analyzer.add_stage_time("Phase 1", 2.0)
        analyzer.add_stage_time("Phase 1", 3.0)
        
        stats = analyzer.get_stage_statistics()
        assert stats["Phase 1"]["min"] == 1.0
        assert stats["Phase 1"]["max"] == 3.0
        assert stats["Phase 1"]["mean"] == 2.0

    def test_bottleneck_analysis(self):
        """Test bottleneck detection"""
        analyzer = PerformanceAnalyzer()
        
        analyzer.add_stage_time("Phase 1", 0.5)
        analyzer.add_stage_time("Phase 2", 2.0)  # Slowest
        analyzer.add_stage_time("Phase 3", 1.0)
        
        report = analyzer.analyze_bottleneck()
        assert "Phase 2" in report
        assert "Slowest" in report


# ============================================================================
# BenchmarkComparison Tests
# ============================================================================

class TestBenchmarkComparison:
    """Tests for benchmark comparison"""

    def test_compare_repair_rate(self):
        """Test comparing repair rate against baseline"""
        metrics = MetricsSummary(repair_rate=0.85)
        comparison = BenchmarkComparison.compare(metrics)
        
        assert "EXCELLENT" in comparison['repair_rate']

    def test_compare_execution_time(self):
        """Test comparing execution time against baseline"""
        metrics = MetricsSummary(avg_execution_time=1.5)
        comparison = BenchmarkComparison.compare(metrics)
        
        assert "FAST" in comparison['execution_time']


# ============================================================================
# End-to-End Tests
# ============================================================================

class TestPhase5EndToEnd:
    """End-to-end tests for Phase 5"""

    def test_complete_evaluation_workflow(self, mock_verified_patch):
        """Test complete evaluation workflow"""
        # Create calculator
        calc = MetricsCalculator()
        calc.add_patch(mock_verified_patch, 2.5)
        
        # Generate reports
        gen = ReportGenerator(calc)
        text_report = gen.generate_text_report()
        
        # Analyze performance
        analyzer = PerformanceAnalyzer()
        analyzer.add_stage_time("Phase 1", 0.5)
        analyzer.add_stage_time("Phase 2", 1.5)
        
        # Compare against benchmark
        summary = calc.generate_summary(total_bugs=10)
        comparison = BenchmarkComparison.compare(summary)
        
        assert text_report is not None
        assert comparison is not None

    def test_pipeline_with_evaluation(self, pipeline_config, mock_verified_patch):
        """Test pipeline with evaluation"""
        pipeline = CGARFPipeline(pipeline_config)
        
        # Simulate successful execution
        pipeline.start_time = 100.0
        stage = PipelineStage("Test", 100.0)
        stage.end_time = 102.0
        stage.duration = 2.0
        stage.status = "completed"
        pipeline.stages = [stage]
        
        # Get summary
        summary = pipeline.get_execution_summary()
        assert "Test" in summary
        assert "2.0" in summary or "2.00" in summary


# ============================================================================
# Integration Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
