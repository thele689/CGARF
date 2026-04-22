"""Utility functions for CGARF"""

import os
import json
import logging
import random
from typing import List, Dict, Set, Tuple, Optional, Any
from pathlib import Path
import re
from functools import wraps
import time

import numpy as np
from loguru import logger


# ==================== Random Seed Management ====================

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logger.info(f"Random seed set to {seed}")


# ==================== Graph Operations ====================

def dfs_paths(graph_dict: Dict[str, List[str]], start: str, goal: str,
             max_depth: int = 20, max_paths: int = 100) -> List[List[str]]:
    """
    Find all simple paths from start to goal using DFS
    
    Args:
        graph_dict: Adjacency list representation {node: [neighbors]}
        start: Starting node
        goal: Goal node
        max_depth: Maximum depth to search
        max_paths: Maximum number of paths to return
    
    Returns:
        List of paths (each path is a list of node IDs)
    """
    
    paths = []
    visited = set()
    
    def dfs(node: str, target: str, path: List[str], depth: int):
        if depth > max_depth or len(paths) >= max_paths:
            return
        
        if node == target:
            paths.append(path.copy())
            return
        
        visited.add(node)
        
        if node in graph_dict:
            for neighbor in graph_dict[node]:
                if neighbor not in visited:
                    path.append(neighbor)
                    dfs(neighbor, target, path, depth + 1)
                    path.pop()
        
        visited.remove(node)
    
    dfs(start, goal, [start], 0)
    return paths


def get_subgraph(nodes: Set[str], edges: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Extract subgraph containing only specified nodes"""
    
    subgraph = {}
    for node in nodes:
        if node in edges:
            neighbors = [n for n in edges[node] if n in nodes]
            if neighbors:
                subgraph[node] = neighbors
    
    return subgraph


def topological_sort(graph: Dict[str, List[str]]) -> List[str]:
    """Topological sort of DAG"""
    
    visited = set()
    result = []
    
    def visit(node):
        if node in visited:
            return
        visited.add(node)
        
        if node in graph:
            for neighbor in graph[node]:
                visit(neighbor)
        
        result.append(node)
    
    for node in graph:
        visit(node)
    
    return result


# ==================== Code Analysis ====================

def extract_function_calls(code: str) -> Set[str]:
    """Extract all function calls from code"""
    
    # Pattern for function calls: word followed by (
    pattern = r'\b([a-zA-Z_]\w*)\s*\('
    matches = re.findall(pattern, code)
    
    # Filter out keywords
    keywords = {
        'if', 'elif', 'else', 'for', 'while', 'with', 'try', 'except',
        'finally', 'raise', 'return', 'yield', 'assert', 'lambda',
        'class', 'def', 'import', 'from'
    }
    
    return {m for m in matches if m not in keywords}


def extract_variables(code: str) -> Set[str]:
    """Extract variable names from code"""
    
    # Simple pattern for variable names
    pattern = r'\b([a-zA-Z_]\w*)\b'
    matches = re.findall(pattern, code)
    
    # Filter out keywords
    keywords = {
        'if', 'elif', 'else', 'for', 'while', 'with', 'try', 'except',
        'finally', 'raise', 'return', 'yield', 'assert', 'lambda',
        'class', 'def', 'import', 'from', 'True', 'False', 'None',
        'and', 'or', 'not', 'in', 'is', 'async', 'await'
    }
    
    return {m for m in matches if m not in keywords}


def get_code_context(filepath: str, start_line: int, end_line: int,
                    context_lines: int = 10) -> str:
    """Get code with surrounding context"""
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        context_start = max(0, start_line - context_lines - 1)
        context_end = min(len(lines), end_line + context_lines)
        
        context = ''.join(lines[context_start:context_end])
        return context
    except Exception as e:
        logger.error(f"Failed to get code context: {e}")
        return ""


# ==================== Similarity Metrics ====================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    
    if vec1.shape != vec2.shape:
        raise ValueError("Vector dimensions must match")
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance"""
    
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def normalized_similarity(s1: str, s2: str) -> float:
    """Compute normalized similarity (0-1) between two strings"""
    
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    
    distance = edit_distance(s1, s2)
    return 1.0 - (distance / max_len)


# Alias for backward compatibility
def normalize_similarity(s1: str, s2: str) -> float:
    """Alias for normalized_similarity - computes normalized similarity (0-1) between two strings"""
    return normalized_similarity(s1, s2)


# ==================== Decorators ====================

def retry(max_attempts: int = 3, delay: int = 1, backoff: float = 2.0):
    """Decorator for retry logic"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt} failed: {e}. "
                        f"Retrying in {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        return wrapper
    return decorator


def timing(func):
    """Decorator to measure function execution time"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        logger.debug(f"{func.__name__} took {duration:.2f}s")
        return result
    
    return wrapper


# ==================== File Operations ====================

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file"""
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        raise


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """Save data as JSON"""
    
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        raise


def load_yaml(filepath: str) -> Dict[str, Any]:
    """Load YAML file"""
    
    try:
        import yaml
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.error("pyyaml package required. Install with: pip install pyyaml")
        raise
    except Exception as e:
        logger.error(f"Failed to load YAML from {filepath}: {e}")
        raise


# ==================== Logging ====================

def setup_logger(name: str, level: str = "INFO",
                log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger with file output"""
    
    logger_instance = logger.bind(name=name)
    
    logger_instance.remove()  # Remove default handler
    
    logger_instance.add(
        lambda msg: print(msg, end=''),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
        level=level
    )
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger_instance.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
            level=level
        )
    
    return logger_instance


# ==================== Metrics ====================

class MetricsCounter:
    """Counter for tracking metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def increment(self, key: str, value: float = 1.0):
        """Increment metric"""
        
        if key not in self.metrics:
            self.metrics[key] = 0
        self.metrics[key] += value
    
    def set(self, key: str, value: float):
        """Set metric value"""
        
        self.metrics[key] = value
    
    def get(self, key: str) -> float:
        """Get metric value"""
        
        return self.metrics.get(key, 0.0)
    
    def get_all(self) -> Dict[str, float]:
        """Get all metrics"""
        
        return self.metrics.copy()
    
    def reset(self):
        """Reset all metrics"""
        
        self.metrics = {}


# ==================== Validation ====================

def validate_patch_format(patch: str) -> bool:
    """Validate Search/Replace patch format"""
    
    pattern = r'<<<\s*SEARCH\n.*?===\n.*?>>>\s*REPLACE'
    return bool(re.search(pattern, patch, re.DOTALL))


def validate_json_schema(data: Dict, schema: Dict) -> bool:
    """Validate data against schema"""
    
    try:
        for key, expected_type in schema.items():
            if key not in data:
                return False
            
            value = data[key]
            if isinstance(expected_type, dict):
                if not isinstance(value, dict):
                    return False
                if not validate_json_schema(value, expected_type):
                    return False
            else:
                if not isinstance(value, expected_type):
                    return False
        
        return True
    except:
        return False
