"""
SWE-Bench Problem Parser

Parses and extracts information from SWE-Bench problems.
Handles repository information, issue descriptions, and test data.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import re
from loguru import logger


@dataclass
class RepositoryInfo:
    """Information about a repository."""
    owner: str
    name: str
    url: str
    
    def full_name(self) -> str:
        """Get full repository name 'owner/repo'."""
        return f"{self.owner}/{self.name}"


@dataclass
class ProblemInfo:
    """Parsed problem information."""
    instance_id: str
    repo_info: RepositoryInfo
    base_commit: str
    issue_description: str
    test_patch: str
    gold_patch: str
    hints: Optional[str] = None


class SWEBenchProblemParser:
    """Parse and extract information from SWE-Bench problems."""
    
    GITHUB_URL_PATTERN = r'https://github\.com/([^/]+)/([^/\s]+)'
    
    @staticmethod
    def parse_repo_info(repo_str: str) -> RepositoryInfo:
        """
        Parse repository information from 'owner/repo' format.
        
        Args:
            repo_str: Repository string like 'sympy/sympy'
            
        Returns:
            RepositoryInfo object
        """
        parts = repo_str.split('/')
        if len(parts) != 2:
            raise ValueError(f"Invalid repo format: {repo_str}")
        
        owner, repo_name = parts
        url = f"https://github.com/{owner}/{repo_name}"
        
        return RepositoryInfo(
            owner=owner,
            name=repo_name,
            url=url
        )
    
    @staticmethod
    def extract_issue_info(problem_statement: str) -> Dict[str, str]:
        """
        Extract key information from the issue description.
        
        Args:
            problem_statement: Full problem statement text
            
        Returns:
            Dictionary with 'title', 'description', and 'error' fields
        """
        lines = problem_statement.strip().split('\n')
        
        # Try to identify title and description
        title = lines[0] if lines else "Unknown issue"
        description = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
        
        # Extract error message if present
        error = SWEBenchProblemParser._extract_error_message(problem_statement)
        
        return {
            'title': title,
            'description': description,
            'error': error,
            'full_text': problem_statement
        }
    
    @staticmethod
    def _extract_error_message(text: str) -> Optional[str]:
        """
        Try to extract error message from problem statement.
        
        Looks for common error patterns:
        - Traceback
        - AssertionError
        - TypeError
        - AttributeError
        - etc.
        """
        error_patterns = [
            r'((?:Error|Exception)[:\s].*)',
            r'(Traceback.*?)(?:\n\n)',
            r'(assert.*?failed)',
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    @staticmethod
    def parse_test_patch(test_patch: str) -> Dict[str, str]:
        """
        Parse test patch to extract test commands and expectations.
        
        Args:
            test_patch: Test patch in unified diff format
            
        Returns:
            Dictionary with test information
        """
        return {
            'test_patch': test_patch,
            'test_type': SWEBenchProblemParser._identify_test_type(test_patch)
        }
    
    @staticmethod
    def _identify_test_type(test_patch: str) -> str:
        """Identify the type of test (pytest, unittest, etc)."""
        if 'pytest' in test_patch:
            return 'pytest'
        elif 'unittest' in test_patch:
            return 'unittest'
        elif 'doctest' in test_patch:
            return 'doctest'
        else:
            return 'custom'
    
    @staticmethod
    def parse_complete_problem(problem: Dict) -> ProblemInfo:
        """
        Parse a complete SWE-Bench problem.
        
        Args:
            problem: Raw problem dictionary from dataset
            
        Returns:
            Parsed ProblemInfo object
        """
        repo_info = SWEBenchProblemParser.parse_repo_info(problem['repo'])
        
        return ProblemInfo(
            instance_id=problem['instance_id'],
            repo_info=repo_info,
            base_commit=problem['base_commit'],
            issue_description=problem['problem_statement'],
            test_patch=problem['test_patch'],
            gold_patch=problem.get('gold_patch', ''),
            hints=problem.get('hints_text')
        )
    
    @staticmethod
    def format_for_llm(problem_info: ProblemInfo,
                      include_test: bool = False) -> str:
        """
        Format problem information for LLM input.
        
        Args:
            problem_info: Parsed problem information
            include_test: Whether to include test information
            
        Returns:
            Formatted text for LLM
        """
        text = f"""
Repository: {problem_info.repo_info.full_name()}
Commit: {problem_info.base_commit}

Issue Description:
{problem_info.issue_description}

Hints:
{problem_info.hints or 'None'}
"""
        
        if include_test:
            text += f"""
Test Patch:
{problem_info.test_patch}
"""
        
        return text.strip()
