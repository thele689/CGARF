"""
Phase 4: SWE-Bench Evaluation Framework
========================================

Apply patches to SWE-Bench instances and evaluate them using the official
SWE-Bench testing framework.

Responsibilities:
  - Convert CGARF repairs into patch files
  - Clone target repository to base_commit
  - Apply patches to the repository
  - Run test suites (FAIL_TO_PASS, PASS_TO_PASS)
  - Compute evaluation metrics (PA, SR)

Output:
  - Patch Application Rate (PA): % of patches that apply cleanly
  - Success Rate (SR): % of applied patches that pass tests
"""

import os
import tempfile
import subprocess
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class PatchResult:
    """Result of applying and testing a single patch"""
    patch_id: str
    patch_code: str
    
    # Application result
    applied: bool = False
    application_error: str = ""
    
    # Test results
    fail_to_pass: Dict[str, bool] = field(default_factory=dict)  # {test_name: passed}
    pass_to_pass: Dict[str, bool] = field(default_factory=dict)  # {test_name: still_pass}
    
    # Metrics
    fail_to_pass_rate: float = 0.0  # % of FAIL_TO_PASS tests that now pass
    pass_to_pass_rate: float = 0.0  # % of PASS_TO_PASS tests that still pass
    
    # Summary
    success: bool = False  # True if all FAIL_TO_PASS pass and all PASS_TO_PASS still pass
    test_time: float = 0.0
    error_message: str = ""


@dataclass
class PatchEvaluationMetrics:
    """Aggregated metrics for patch evaluation"""
    instance_id: str
    total_patches: int = 0
    applied_patches: int = 0
    successful_patches: int = 0
    
    # Rates
    patch_application_rate: float = 0.0  # Applied/Total
    success_rate: float = 0.0  # Successful/Applied
    
    # Details
    patch_results: List[PatchResult] = field(default_factory=list)


class SWEBenchEvaluator:
    """
    Evaluate patches using SWE-Bench testing framework
    """
    
    def __init__(self, timeout: int = 60):
        """
        Initialize evaluator
        
        Args:
            timeout: Timeout for test execution (seconds)
        """
        self.timeout = timeout
    
    def _get_repo_url(self, instance_id: str) -> Optional[str]:
        """
        Get GitHub repository URL from instance ID
        
        Args:
            instance_id: e.g., "astropy__astropy-7746"
        
        Returns:
            GitHub URL, or None if not found
        """
        # Map instance prefix to GitHub owner/repo
        prefix_map = {
            'astropy__astropy': 'astropy/astropy',
            'django__django': 'django/django',
            'matplotlib__matplotlib': 'matplotlib/matplotlib',
            'mwaskom__seaborn': 'mwaskom/seaborn',
            'pallets__flask': 'pallets/flask',
            'psf__requests': 'psf/requests',
            'pydata__xarray': 'pydata/xarray',
            'pylint-dev__pylint': 'pylint-dev/pylint',
            'pytest-dev__pytest': 'pytest-dev/pytest',
            'scikit-learn__scikit-learn': 'scikit-learn/scikit-learn',
            'sphinx-doc__sphinx': 'sphinx-doc/sphinx',
            'sympy__sympy': 'sympy/sympy',
        }
        
        for prefix, repo in prefix_map.items():
            if instance_id.startswith(prefix):
                return f"https://github.com/{repo}"
        
        logger.warning(f"Unknown repo prefix for {instance_id}")
        return None
    
    def _clone_repo(
        self,
        repo_url: str,
        base_commit: str,
        work_dir: str
    ) -> bool:
        """
        Clone repository to specific commit
        
        Args:
            repo_url: GitHub repository URL
            base_commit: Commit hash to checkout
            work_dir: Directory to clone into
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Cloning {repo_url}@{base_commit[:8]}...")
            
            # Shallow clone first
            subprocess.run(
                ['git', 'clone', '--depth', '1', repo_url, work_dir],
                capture_output=True,
                timeout=120,
                check=True
            )
            
            # Checkout specific commit
            subprocess.run(
                ['git', 'checkout', base_commit],
                cwd=work_dir,
                capture_output=True,
                timeout=60,
                check=True
            )
            
            logger.info(f"✓ Repository cloned successfully")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git operation failed: {e.stderr.decode() if e.stderr else e}")
            return False
        except subprocess.TimeoutExpired:
            logger.warning(f"Git operation timed out")
            return False
        except Exception as e:
            logger.warning(f"Error cloning repo: {e}")
            return False
    
    def _apply_patch(
        self,
        patch_content: str,
        work_dir: str
    ) -> bool:
        """
        Apply a patch file to repository
        
        Args:
            patch_content: Patch content (unified diff format)
            work_dir: Repository root directory
        
        Returns:
            True if patch applied successfully
        """
        try:
            # Write patch to temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.patch',
                dir=work_dir,
                delete=False
            ) as f:
                f.write(patch_content)
                patch_file = f.name
            
            try:
                # Apply patch
                subprocess.run(
                    ['git', 'apply', patch_file],
                    cwd=work_dir,
                    capture_output=True,
                    timeout=30,
                    check=True
                )
                logger.info(f"✓ Patch applied successfully")
                return True
            
            finally:
                os.unlink(patch_file)
        
        except subprocess.CalledProcessError as e:
            logger.warning(f"Patch apply failed: {e.stderr.decode() if e.stderr else e}")
            return False
        except Exception as e:
            logger.warning(f"Error applying patch: {e}")
            return False
    
    def _run_tests(
        self,
        test_paths: List[str],
        work_dir: str
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Run pytest on specific test paths
        
        Args:
            test_paths: List of test file paths (e.g., ["astropy/tests/test_wcs.py"])
            work_dir: Repository root
        
        Returns:
            (all_passed, test_results_dict)
        """
        results = {}
        all_passed = True
        
        for test_path in test_paths:
            full_path = os.path.join(work_dir, test_path)
            
            if not os.path.exists(full_path):
                logger.warning(f"Test path not found: {test_path}")
                results[test_path] = False
                all_passed = False
                continue
            
            try:
                result = subprocess.run(
                    ['pytest', full_path, '-xvs'],
                    cwd=work_dir,
                    capture_output=True,
                    timeout=self.timeout,
                    text=True
                )
                
                passed = result.returncode == 0
                results[test_path] = passed
                
                if not passed:
                    all_passed = False
                    logger.warning(f"  Test failed: {test_path}")
                else:
                    logger.info(f"  ✓ Test passed: {test_path}")
            
            except subprocess.TimeoutExpired:
                logger.warning(f"  Test timeout: {test_path}")
                results[test_path] = False
                all_passed = False
            except Exception as e:
                logger.warning(f"  Test error: {test_path} - {e}")
                results[test_path] = False
                all_passed = False
        
        return all_passed, results
    
    def evaluate_patch(
        self,
        patch_id: str,
        patch_content: str,
        instance_info: Dict[str, Any]
    ) -> PatchResult:
        """
        Evaluate a single patch on SWE-Bench instance
        
        Args:
            patch_id: Unique patch identifier
            patch_content: Patch code in diff format
            instance_info: SWE-Bench instance info with:
                - instance_id: Instance identifier
                - repo_url: GitHub repository URL
                - base_commit: Base commit hash
                - FAIL_TO_PASS: List of test paths that should start failing but pass after patch
                - PASS_TO_PASS: List of test paths that should continue to pass
        
        Returns:
            PatchResult with evaluation metrics
        """
        
        result = PatchResult(patch_id=patch_id, patch_code=patch_content)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Clone repository
            repo_url = instance_info.get('repo_url')
            base_commit = instance_info.get('base_commit')
            instance_id = instance_info.get('instance_id')
            
            if not repo_url:
                result.application_error = "Unknown repository"
                return result
            
            work_dir = os.path.join(tmpdir, 'repo')
            if not self._clone_repo(repo_url, base_commit, work_dir):
                result.application_error = "Failed to clone repository"
                return result
            
            # 2. Apply patch
            if not self._apply_patch(patch_content, work_dir):
                result.application_error = "Failed to apply patch"
                return result
            
            result.applied = True
            
            # 3. Run FAIL_TO_PASS tests
            fail_to_pass_paths = instance_info.get('FAIL_TO_PASS', [])
            if fail_to_pass_paths:
                logger.info(f"Running FAIL_TO_PASS tests ({len(fail_to_pass_paths)} tests)...")
                _, fail_results = self._run_tests(fail_to_pass_paths, work_dir)
                result.fail_to_pass = fail_results
                
                # Calculate rate
                if fail_results:
                    passed_count = sum(1 for v in fail_results.values() if v)
                    result.fail_to_pass_rate = passed_count / len(fail_results)
            
            # 4. Run PASS_TO_PASS tests
            pass_to_pass_paths = instance_info.get('PASS_TO_PASS', [])
            if pass_to_pass_paths:
                logger.info(f"Running PASS_TO_PASS tests ({len(pass_to_pass_paths)} tests)...")
                _, pass_results = self._run_tests(pass_to_pass_paths, work_dir)
                result.pass_to_pass = pass_results
                
                # Calculate rate
                if pass_results:
                    passed_count = sum(1 for v in pass_results.values() if v)
                    result.pass_to_pass_rate = passed_count / len(pass_results)
            
            # 5. Determine success
            # Success = all FAIL_TO_PASS pass AND all PASS_TO_PASS still pass
            fail_success = (
                not fail_to_pass_paths or  # No tests, consider success
                result.fail_to_pass_rate == 1.0
            )
            pass_success = (
                not pass_to_pass_paths or  # No tests, consider success
                result.pass_to_pass_rate == 1.0
            )
            
            result.success = fail_success and pass_success
        
        return result
    
    def evaluate_patches(
        self,
        patches: List[Dict[str, Any]],
        instance_info: Dict[str, Any]
    ) -> PatchEvaluationMetrics:
        """
        Evaluate multiple patches
        
        Args:
            patches: List of patch dicts with 'id' and 'content'
            instance_info: SWE-Bench instance information
        
        Returns:
            PatchEvaluationMetrics with aggregate results
        """
        
        instance_id = instance_info.get('instance_id', 'unknown')
        logger.info(f"\nEvaluating {len(patches)} patches for {instance_id}")
        
        metrics = PatchEvaluationMetrics(
            instance_id=instance_id,
            total_patches=len(patches)
        )
        
        for i, patch in enumerate(patches, 1):
            patch_id = patch.get('id', f'patch_{i}')
            patch_content = patch.get('content', patch.get('code', ''))
            
            logger.info(f"\n[{i}/{len(patches)}] Evaluating {patch_id}...")
            
            result = self.evaluate_patch(patch_id, patch_content, instance_info)
            metrics.patch_results.append(result)
            
            if result.applied:
                metrics.applied_patches += 1
            
            if result.success:
                metrics.successful_patches += 1
            
            # Log result
            status = "✓" if result.success else "✗"
            applied_str = "applied" if result.applied else "failed-to-apply"
            logger.info(f"  {status} {patch_id}: {applied_str}, success={result.success}")
        
        # Calculate rates
        if metrics.total_patches > 0:
            metrics.patch_application_rate = metrics.applied_patches / metrics.total_patches
        
        if metrics.applied_patches > 0:
            metrics.success_rate = metrics.successful_patches / metrics.applied_patches
        
        logger.info(f"\nEvaluation Summary for {instance_id}:")
        logger.info(f"  Patch Application Rate (PA): {metrics.patch_application_rate:.1%}")
        logger.info(f"  Success Rate (SR): {metrics.success_rate:.1%}")
        logger.info(f"  Successful patches: {metrics.successful_patches}/{metrics.applied_patches}")
        
        return metrics


# Test evaluator (requires actual SWE-Bench setup)
def test_evaluator():
    """Test SWE-Bench evaluator"""
    evaluator = SWEBenchEvaluator()
    
    # Dummy test - just verify the class loads
    print("=" * 70)
    print("SWE-Bench Evaluator initialized")
    print("=" * 70)
    print("Note: Full testing requires:")
    print("  1. Git access to clone repositories")
    print("  2. pytest installed in the environment")
    print("  3. Dependencies for each SWE-Bench repository")
    print("=" * 70)


if __name__ == "__main__":
    test_evaluator()
