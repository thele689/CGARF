"""
Phase 0: Unified Fault Localization Data Loader
=================================================

Load bug localization outputs from different methods (OrcaLoca, Agentless, CoSIL)
and integrate them with SWE-Bench instances for CGARF repair pipeline.

Workflow:
  Input: Unified JSONL files for different localization methods
    ↓
  [Select Method] Choose which localization method to use
    ↓
  [Load JSONL] Load bug_locations from method-specific JSONL
    ↓
  [Map to SWE-Bench] Find corresponding SWE-Bench instance
    ↓
  [Extract Test Info] Parse test_paths and success criteria
    ↓
  Output: EnhancedIssueContext with localization + SWE-Bench data
         for CGARF code repair algorithms
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BugLocation:
    """Single bug location candidate from fault localization"""
    file_path: str                      # e.g., "astropy/wcs/wcs.py"
    class_name: Optional[str] = None    # e.g., "WCS"
    method_name: Optional[str] = None   # e.g., "wcs_pix2world"
    line_start: Optional[int] = None    # Line number
    line_end: Optional[int] = None      # Line number
    confidence: Optional[float] = None  # Confidence score (0-1)
    source_method: str = "unknown"      # Which localization method found this


@dataclass
class LocalizationOutput:
    """Parsed localization method output"""
    instance_id: str
    method: str                         # 'orcaloca', 'agentless', or 'cosil'
    conclusion: str                     # Analysis text from localization method
    bug_locations: List[BugLocation] = field(default_factory=list)


@dataclass
class SWEBenchInstance:
    """SWE-Bench instance metadata and code"""
    instance_id: str
    repo: str                           # e.g., "astropy/astropy"
    base_commit: str
    problem_statement: str
    buggy_code: str                     # Source code at base_commit
    patch: str                          # Ground-truth patch
    test_patch: str                     # Test file patch
    test_paths: List[str] = field(default_factory=list)  # e.g., ["astropy/tests/..."]


@dataclass
class EnhancedIssueContext:
    """
    Optimized context for CGARF code repair pipeline
    Contains only information that repair algorithms actually need
    
    Core Components:
    1. Bug Localization: bug_locations (where to fix)
    2. Repository Info: repo, base_commit (code access)
    3. Test Info: test_paths, fail_to_pass, pass_to_pass (success criteria)
    4. Problem Info: instance_id, problem_statement (understanding)
    """
    # Bug localization information (ESSENTIAL)
    bug_locations: List[BugLocation]    # Ranked list of suspicious locations
    
    # Repository information (ESSENTIAL for code access)
    instance_id: str                    # e.g., "astropy__astropy-7746"
    repo: str                           # e.g., "astropy/astropy"
    base_commit: str                    # Specific commit hash to checkout
    
    # Problem/context information (IMPORTANT for understanding)
    problem_statement: str              # Issue description
    
    # Test information (ESSENTIAL for success criteria)
    test_paths: List[str]               # Files to test (e.g., ["tests/test_foo.py"])
    fail_to_pass: List[str]             # Test paths that must pass
    pass_to_pass: List[str]             # Test paths that must not break
    
    # CGARF configuration (OPTIONAL)
    repo_url: str = ""                  # Optional: full GitHub URL
    test_framework: str = "pytest"      # Test runner (pytest, unittest, etc.)
    timeout_seconds: int = 300          # Max time per repair attempt
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info for logging


class UnifiedFaultLocalizationLoader:
    """
    Load fault localization outputs from different methods
    
    Supports:
      - OrcaLoca (Observable Recursive Code Analysis)
      - Agentless (LLM-based)
      - CoSIL (Context-aware Statistical Information Localization)
    """
    
    SUPPORTED_METHODS = {
        'orcaloca': str(REPO_ROOT / 'input' / 'orcaloca.jsonl'),
        'agentless': str(REPO_ROOT / 'input' / 'agentless.jsonl'),
        'cosil': str(REPO_ROOT / 'input' / 'cosil.jsonl'),
    }
    DEFAULT_LOCAL_SWE_BENCH_ARROW = (
        str(REPO_ROOT / "data" / "swe-bench") + "/"
        "princeton-nlp___swe-bench_lite/default/0.0.0/"
        "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2/"
        "swe-bench_lite-test.arrow"
    )
    
    def __init__(self, method: str = 'orcaloca'):
        """
        Initialize loader
        
        Args:
            method: Localization method to use ('orcaloca', 'agentless', or 'cosil')
        
        Raises:
            ValueError: If method not supported
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown localization method: {method}\n"
                f"Supported methods: {list(self.SUPPORTED_METHODS.keys())}"
            )
        
        self.method = method
        self.jsonl_file = Path(self.SUPPORTED_METHODS[method])
        self.swe_bench_data = None
        
        if not self.jsonl_file.exists():
            logger.warning(f"JSONL file not found for {method}: {self.jsonl_file}")
            logger.info(f"  Please convert {method} data using convert_all_localization_methods.py")
        
        logger.info(f"UnifiedFaultLocalizationLoader initialized with method '{method}'")
        logger.info(f"  JSONL file: {self.jsonl_file}")

    def _resolve_local_swe_bench_arrow(self) -> Optional[Path]:
        """Resolve a local SWE-Bench Lite arrow file when available."""

        candidate = (
            os.getenv("SWE_BENCH_ARROW_PATH")
            or self.DEFAULT_LOCAL_SWE_BENCH_ARROW
        )
        if not candidate:
            return None

        arrow_path = Path(candidate)
        if arrow_path.exists():
            return arrow_path
        return None
    
    def set_method(self, method: str) -> None:
        """
        Switch to a different localization method
        
        Args:
            method: New localization method to use
        
        Raises:
            ValueError: If method not supported
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown method: {method}")
        
        self.method = method
        self.jsonl_file = Path(self.SUPPORTED_METHODS[method])
        logger.info(f"Switched to localization method: {method}")
    
    def load_swe_bench(self) -> None:
        """Load SWE-Bench-Lite dataset"""
        logger.info("Loading SWE-Bench-Lite dataset...")
        local_arrow = self._resolve_local_swe_bench_arrow()
        if local_arrow is not None:
            logger.info(f"Loading SWE-Bench-Lite from local arrow: {local_arrow}")
            self.swe_bench_data = Dataset.from_file(str(local_arrow))
            logger.info(f"Loaded {len(self.swe_bench_data)} SWE-Bench instances")
            return
        try:
            self.swe_bench_data = load_dataset(
                "princeton-nlp/SWE-bench_Lite", 
                split="test"
            )
            logger.info(f"Loaded {len(self.swe_bench_data)} SWE-Bench instances")
        except Exception as e:
            logger.warning(f"Failed to load SWE-Bench-Lite from HF: {e}")
            if local_arrow is not None:
                logger.info(f"Retrying with local arrow: {local_arrow}")
                self.swe_bench_data = Dataset.from_file(str(local_arrow))
                logger.info(f"Loaded {len(self.swe_bench_data)} SWE-Bench instances")
                return
            raise
    
    def _load_jsonl_record(self, line: str) -> Optional[LocalizationOutput]:
        """
        Parse a single JSONL line
        
        Expected format:
        {
          "instance_id": "astropy__astropy-12907",
          "repo": "astropy/astropy",
          "bug_locations": [
            {
              "file_path": "foo.py",
              "class_name": "Foo",
              "method_name": "bar",
              "confidence": 0.95
            }
          ],
          "conclusion": "..."
        }
        """
        try:
            data = json.loads(line)
            instance_id = data.get("instance_id")
            
            # Parse bug_locations
            bug_locations = []
            for loc in data.get("bug_locations", []):
                bug_locations.append(BugLocation(
                    file_path=loc.get("file_path", ""),
                    class_name=loc.get("class_name"),
                    method_name=loc.get("method_name"),
                    line_start=loc.get("line_start"),
                    line_end=loc.get("line_end"),
                    confidence=loc.get("confidence"),
                    source_method=self.method
                ))
            
            return LocalizationOutput(
                instance_id=instance_id,
                method=self.method,
                conclusion=data.get("conclusion", ""),
                bug_locations=bug_locations
            )
        except Exception as e:
            logger.error(f"Failed to parse JSONL record: {e}")
            return None
    
    def _find_swe_bench_instance(self, instance_id: str) -> Optional[Dict]:
        """Find SWE-Bench instance by ID"""
        if not self.swe_bench_data:
            return None
        
        for instance in self.swe_bench_data:
            if instance["instance_id"] == instance_id:
                return instance
        
        logger.warning(f"SWE-Bench instance not found: {instance_id}")
        return None
    
    def _extract_repo_info(self, repo: str) -> tuple:
        """
        Extract repo owner and name
        
        Args:
            repo: e.g., "astropy/astropy"
        
        Returns:
            (owner, repo_name): e.g., ("astropy", "astropy")
        """
        parts = repo.split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]
        return "unknown", repo
    
    def _extract_test_paths_from_patch(self, test_patch: str) -> List[str]:
        """
        Extract test file paths from unified diff patch
        
        Args:
            test_patch: Unified diff format patch
        
        Returns:
            List of test file paths
        """
        test_paths = []
        for line in test_patch.split('\n'):
            if line.startswith('--- ') or line.startswith('+++ '):
                parts = line.split('\t')
                if len(parts) > 0:
                    path = parts[0].replace('--- ', '').replace('+++ ', '').strip()
                    if path.startswith('a/') or path.startswith('b/'):
                        path = path[2:]
                    if path and not path.startswith('/dev/null'):
                        test_paths.append(path)
        return list(dict.fromkeys(test_paths))

    def _normalize_test_spec(self, value: Any) -> List[str]:
        """Normalize FAIL_TO_PASS / PASS_TO_PASS fields to lists."""

        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                return [value] if value else []
        return []
    
    def load_instance(self, instance_id: str) -> Optional[EnhancedIssueContext]:
        """
        Load a single instance combining localization + SWE-Bench
        
        Returns only information needed by CGARF repair algorithms:
        - bug_locations: Where to fix
        - repo + base_commit: Code access
        - test_paths + fail_to_pass + pass_to_pass: Success criteria
        - problem_statement: Understanding
        
        Args:
            instance_id: Instance identifier (e.g., "astropy__astropy-7746")
        
        Returns:
            EnhancedIssueContext with repair-relevant data, or None if load fails
        """
        # Load SWE-Bench data if not loaded
        if not self.swe_bench_data:
            self.load_swe_bench()
        
        # 1. Load localization output from JSONL
        localization_output = None
        if self.jsonl_file.exists():
            try:
                with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        record = self._load_jsonl_record(line.strip())
                        if record and record.instance_id == instance_id:
                            localization_output = record
                            break
            except Exception as e:
                logger.error(f"Error reading JSONL file: {e}")
        
        if not localization_output:
            logger.error(f"Localization data not found in {self.method}: {instance_id}")
            return None
        
        logger.info(f"Found {len(localization_output.bug_locations)} bug location candidates from {self.method}")
        
        # 2. Find corresponding SWE-Bench instance
        swe_bench_instance = self._find_swe_bench_instance(instance_id)
        if not swe_bench_instance:
            logger.error(f"SWE-Bench instance not found: {instance_id}")
            return None
        
        logger.info(f"Loaded SWE-Bench: {swe_bench_instance['repo']}")
        
        # 3. Extract test file paths from patch
        test_patch = swe_bench_instance.get("test_patch", "")
        test_paths = self._extract_test_paths_from_patch(test_patch)
        
        # 4. Get fail_to_pass and pass_to_pass test specs
        fail_to_pass = self._normalize_test_spec(swe_bench_instance.get("FAIL_TO_PASS", []))
        pass_to_pass = self._normalize_test_spec(swe_bench_instance.get("PASS_TO_PASS", []))
        
        # 5. Create optimized context for repair
        context = EnhancedIssueContext(
            # Bug localization (where to fix)
            bug_locations=localization_output.bug_locations,
            
            # Repository (how to access code)
            instance_id=instance_id,
            repo=swe_bench_instance["repo"],
            base_commit=swe_bench_instance["base_commit"],
            repo_url=f"https://github.com/{swe_bench_instance['repo']}",
            
            # Problem understanding (what to fix)
            problem_statement=swe_bench_instance.get("problem_statement", ""),
            
            # Test info (success criteria)
            test_paths=test_paths,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            
            # Metadata
            metadata={
                'localization_method': self.method,
                'swe_bench_version': 'lite'
            }
        )
        
        logger.info(f"Created context: {len(context.bug_locations)} candidates, {len(context.test_paths)} test files")
        return context
    
    def load_instances_batch(
        self, 
        instance_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, EnhancedIssueContext]:
        """
        Load multiple instances from localization JSONL file
        
        Args:
            instance_ids: Specific instances to load, or None to load all
            limit: Maximum number of instances to load (None = unlimited)
        
        Returns:
            Dictionary mapping instance_id to EnhancedIssueContext
        """
        results = {}
        
        if not self.jsonl_file.exists():
            logger.error(f"JSONL file not found: {self.jsonl_file}")
            logger.info(f"Available methods: {list(self.SUPPORTED_METHODS.keys())}")
            return results
        
        # Load SWE-Bench data if not loaded
        if not self.swe_bench_data:
            self.load_swe_bench()
        
        # Get list of instance IDs to load from JSONL
        localization_instances = {}
        try:
            with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    record = self._load_jsonl_record(line)
                    if record:
                        localization_instances[record.instance_id] = record
        
        except Exception as e:
            logger.error(f"Error reading JSONL file: {e}")
            return results
        
        # Determine which instances to load
        if instance_ids is None:
            instance_ids = list(localization_instances.keys())
        else:
            instance_ids = [id for id in instance_ids if id in localization_instances]
        
        # Apply limit
        if limit is not None:
            instance_ids = instance_ids[:limit]
        
        logger.info(f"Loading {len(instance_ids)} instances from {self.method} ({len(localization_instances)} total available)")
        
        for i, instance_id in enumerate(instance_ids, 1):
            logger.info(f"[{i}/{len(instance_ids)}] Loading {instance_id}...")
            
            try:
                context = self.load_instance(instance_id)
                if context:
                    results[instance_id] = context
                    logger.info(f"✓ Successfully loaded {instance_id}")
                else:
                    logger.warning(f"✗ Failed to load {instance_id}")
            except Exception as e:
                logger.error(f"✗ Exception loading {instance_id}: {e}")
            
            # Progress every 50 instances
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(instance_ids)} instances loaded")
        
        logger.info(f"✓ Successfully loaded {len(results)}/{len(instance_ids)} instances from {self.method}")
        return results
