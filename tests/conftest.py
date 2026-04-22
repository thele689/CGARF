"""
Pytest configuration and shared fixtures for CGARF tests
"""

import pytest
import sys
from pathlib import Path

# Add project root to path to ensure imports work correctly
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root_path():
    """Get project root path"""
    return Path(__file__).parent.parent.absolute()


@pytest.fixture
def temp_code_file(tmp_path):
    """Create a temporary Python code file for testing"""
    code_file = tmp_path / "test_code.py"
    code_file.write_text("""
def process_data(items):
    '''Process a list of items'''
    result = []
    for item in items:
        result.append(item.upper())
    return result
""")
    return str(code_file)


@pytest.fixture
def sample_issue_context():
    """Sample issue context for testing"""
    from src.common.data_structures import IssueContext
    
    return IssueContext(
        id="issue_001",
        description="NoneType error when items is None",
        repo_path="/test/repo",
        candidates=["process_data"],
        test_framework="pytest",
        timeout_seconds=120
    )


@pytest.fixture
def mock_llm_interface():
    """Mock LLM interface for testing"""
    from src.common.llm_interface import MockLLMInterface
    return MockLLMInterface()


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Add custom markers
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# ============================================================================
# Test Collection
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Auto-mark tests based on file path
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
