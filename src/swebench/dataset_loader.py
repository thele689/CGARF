"""
SWE-Bench Dataset Loader

Loads and manages SWE-Bench benchmark datasets from HuggingFace.
Supports: SWE-Bench Lite, Verified, and full dataset versions.
"""

from typing import Iterator, Dict, Any, Optional, List
from pathlib import Path
import json
from loguru import logger


class SWEBenchDataset:
    """
    SWE-Bench dataset loader and manager
    
    Loads problems from HuggingFace datasets library and provides
    iteration, sampling, and filtering capabilities.
    """
    
    # Supported datasets
    LITE_DATASET = 'princeton-nlp/SWE-bench_Lite'
    VERIFIED_DATASET = 'princeton-nlp/SWE-bench_Verified'
    FULL_DATASET = 'princeton-nlp/SWE-bench'
    
    def __init__(self,
                 split: str = 'test',
                 subset: str = 'lite',
                 cache_dir: str = './data/swe-bench',
                 offline_mode: bool = False):
        """
        Initialize SWE-Bench dataset loader.
        
        Args:
            split: 'test' or 'train'
            subset: 'lite', 'verified', or 'full'
            cache_dir: Directory for caching datasets
            offline_mode: If True, only use cached data
        """
        self.split = split
        self.subset = subset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline_mode = offline_mode
        
        self._dataset = None
        self._problems = None
        
        # Map subset names to dataset identifiers
        self._subset_map = {
            'lite': self.LITE_DATASET,
            'verified': self.VERIFIED_DATASET,
            'full': self.FULL_DATASET
        }
        
        if subset not in self._subset_map:
            raise ValueError(f"Unknown subset: {subset}. "
                           f"Choose from {list(self._subset_map.keys())}")
    
    def load(self) -> 'SWEBenchDataset':
        """
        Load the dataset from HuggingFace or cache.
        
        Returns:
            self for method chaining
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library required. Install with: pip install datasets"
            )
        
        dataset_name = self._subset_map[self.subset]
        
        logger.info(f"Loading {self.subset} dataset from {dataset_name}...")
        
        # Simple load without problematic parameters
        self._dataset = load_dataset(
            dataset_name,
            split=self.split,
            cache_dir=str(self.cache_dir),
            trust_remote_code=True
        )
        
        logger.info(f"Loaded {len(self._dataset)} problems")
        
        return self
        
        return self
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over problems in the dataset."""
        if self._dataset is None:
            self.load()
        
        for example in self._dataset:
            yield self._normalize_problem(example)
    
    def __len__(self) -> int:
        """Get total number of problems."""
        if self._dataset is None:
            self.load()
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single problem by index."""
        if self._dataset is None:
            self.load()
        
        return self._normalize_problem(self._dataset[idx])
    
    def _normalize_problem(self, example: Dict) -> Dict[str, Any]:
        """Normalize problem fields for consistent access."""
        return {
            'instance_id': example['instance_id'],
            'repo': example['repo'],
            'base_commit': example['base_commit'],
            'problem_statement': example['problem_statement'],
            'test_patch': example['test_patch'],
            'gold_patch': example.get('gold_patch', ''),
            'hints_text': example.get('hints_text', ''),
            'created_at': example.get('created_at', ''),
            'version': example.get('version', '')
        }
    
    def get_subset(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get a subset of problems by indices."""
        if self._dataset is None:
            self.load()
        
        return [self._normalize_problem(self._dataset[i]) for i in indices]
    
    def sample(self,
               num_samples: int,
               seed: int = 42) -> List[Dict[str, Any]]:
        """
        Randomly sample problems from the dataset.
        
        Args:
            num_samples: Number of problems to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled problems
        """
        if self._dataset is None:
            self.load()
        
        import random
        random.seed(seed)
        
        total = len(self._dataset)
        sample_size = min(num_samples, total)
        indices = random.sample(range(total), sample_size)
        
        return self.get_subset(indices)
    
    def filter_by_repo(self, repos: List[str]) -> List[Dict[str, Any]]:
        """Filter problems by repository names."""
        if self._dataset is None:
            self.load()
        
        result = []
        for example in self._dataset:
            if example['repo'] in repos:
                result.append(self._normalize_problem(example))
        
        return result
    
    def get_repo_counts(self) -> Dict[str, int]:
        """Get count of problems per repository."""
        if self._dataset is None:
            self.load()
        
        counts = {}
        for example in self._dataset:
            repo = example['repo']
            counts[repo] = counts.get(repo, 0) + 1
        
        return counts
    
    def save_to_jsonl(self, output_path: str) -> None:
        """Save dataset to JSONL format for offline use."""
        if self._dataset is None:
            self.load()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for example in self._dataset:
                normalized = self._normalize_problem(example)
                f.write(json.dumps(normalized) + '\n')
        
        logger.info(f"Saved {len(self._dataset)} problems to {output_path}")
    
    @staticmethod
    def load_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
        """Load problems from JSONL file."""
        problems = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                problems.append(json.loads(line))
        
        logger.info(f"Loaded {len(problems)} problems from {jsonl_path}")
        return problems
