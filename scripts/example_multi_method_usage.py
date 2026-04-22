#!/usr/bin/env python3
"""
Example: Using UnifiedFaultLocalizationLoader in CGARF Pipeline

This script demonstrates:
1. Load instances using different localization methods
2. Compare bug_locations from different methods
3. Integrate with Phase 1-4 of CGARF pipeline
"""

import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.phase0_integrator.fault_localization_loader import (
    UnifiedFaultLocalizationLoader,
    EnhancedIssueContext,
    BugLocation
)
from loguru import logger


class MultiMethodCGARFIntegration:
    """Example CGARF pipeline with multi-method localization support"""
    
    def __init__(self, methods: List[str] = None, limit: int = 10):
        """
        Initialize with support for multiple localization methods
        
        Args:
            methods: List of methods to use ('orcaloca', 'agentless', 'cosil')
            limit: Number of instances to process per method
        """
        self.methods = methods or ['orcaloca']
        self.limit = limit
        self.loaders = {method: UnifiedFaultLocalizationLoader(method=method) 
                       for method in self.methods}
        
        logger.info(f"Initialized multi-method CGARF with methods: {self.methods}")
    
    def process_single_method(self, method: str, example_id: Optional[str] = None):
        """
        Process instances using single localization method
        
        Args:
            method: Localization method to use
            example_id: Specific instance to process (optional)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing with method: {method.upper()}")
        logger.info(f"{'='*70}")
        
        loader = self.loaders[method]
        
        # Load instances
        if example_id:
            logger.info(f"Loading single instance: {example_id}")
            contexts = {example_id: loader.load_instance(example_id)}
        else:
            logger.info(f"Loading batch ({self.limit} instances)")
            contexts = loader.load_instances_batch(limit=self.limit)
        
        if not contexts:
            logger.warning(f"No instances loaded for method: {method}")
            return []
        
        logger.info(f"✓ Loaded {len(contexts)} instances")
        
        # Process each instance through CGARF pipeline
        results = []
        for i, (instance_id, context) in enumerate(contexts.items(), 1):
            logger.info(f"\n[{i}/{len(contexts)}] Processing {instance_id}")
            logger.info(f"  Repo: {context.repo}")
            logger.info(f"  Bug locations: {len(context.bug_locations)}")
            
            # Phase 0: Already done - data loaded with localization
            result = self._process_instance(context, method)
            results.append(result)
        
        return results
    
    def _process_instance(self, context: EnhancedIssueContext, method: str) -> dict:
        """
        Process single instance through CGARF pipeline
        
        In real implementation, this would call Phase 1-4
        """
        result = {
            'instance_id': context.instance_id,
            'method': context.localization_method,
            'repo': context.repo,
        }
        
        # Phase 1: Code Analysis
        logger.info(f"  → Phase 1: Code Analysis")
        primary_location = context.bug_locations[0] if context.bug_locations else None
        if primary_location:
            logger.info(f"    Primary: {primary_location.file_path}")
            if primary_location.class_name:
                logger.info(f"    Class: {primary_location.class_name}")
            if primary_location.method_name:
                logger.info(f"    Method: {primary_location.method_name}")
        
        result['analysis'] = {
            'primary_file': primary_location.file_path if primary_location else None,
            'num_candidates': len(context.bug_locations)
        }
        
        # Phase 2: Fix Generation
        logger.info(f"  → Phase 2: Fix Generation (would use context.buggy_code)")
        result['generation'] = {
            'api_calls': 0,  # placeholder
            'patches_generated': 0
        }
        
        # Phase 3: Test Execution
        logger.info(f"  → Phase 3: Test Execution (would use context.test_paths)")
        result['testing'] = {
            'tests_run': len(context.test_paths),
            'tests_passed': 0
        }
        
        # Phase 4: Ranking
        logger.info(f"  → Phase 4: Ranking")
        result['ranking'] = {
            'candidates_ranked': 0,
            'best_score': 0.0
        }
        
        return result
    
    def compare_methods(self, instance_id: str):
        """
        Compare bug_locations from different methods for same instance
        
        Args:
            instance_id: SWE-Bench instance ID
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Comparing methods for: {instance_id}")
        logger.info(f"{'='*70}")
        
        all_locations = {}
        
        # Load from each method
        for method in self.methods:
            loader = self.loaders[method]
            context = loader.load_instance(instance_id)
            
            if context:
                all_locations[method] = context.bug_locations
                logger.info(f"\n{method.upper()}:")
                logger.info(f"  Total locations: {len(context.bug_locations)}")
                
                for i, loc in enumerate(context.bug_locations[:5], 1):
                    conf_str = f" (conf: {loc.confidence:.2f})" if loc.confidence else ""
                    logger.info(f"    {i}. {loc.file_path}{conf_str}")
            else:
                logger.warning(f"{method.upper()}: Instance not found")
                all_locations[method] = []
        
        # Compare
        if len(all_locations) > 1:
            logger.info(f"\n{'─'*70}")
            logger.info("Comparison Summary:")
            
            # Find overlapping files
            all_files = {}
            for method, locs in all_locations.items():
                for loc in locs:
                    if loc.file_path not in all_files:
                        all_files[loc.file_path] = []
                    all_files[loc.file_path].append(method)
            
            overlapping = {f: methods for f, methods in all_files.items() 
                          if len(methods) == len(self.methods)}
            
            logger.info(f"  Total unique files: {len(all_files)}")
            logger.info(f"  Overlapping files (in all methods): {len(overlapping)}")
            
            if overlapping:
                logger.info(f"\n  Overlapping locations:")
                for file_path in sorted(overlapping.keys())[:5]:
                    logger.info(f"    - {file_path}")
        
        return all_locations
    
    def collect_statistics(self) -> dict:
        """Collect statistics about different methods"""
        logger.info(f"\n{'='*70}")
        logger.info("Collection Statistics")
        logger.info(f"{'='*70}")
        
        stats = {}
        
        for method in self.methods:
            loader = self.loaders[method]
            
            # Check file
            jsonl_file = loader.jsonl_file
            if not jsonl_file.exists():
                logger.warning(f"{method}: JSONL file not found ({jsonl_file})")
                stats[method] = {'status': 'missing', 'records': 0}
                continue
            
            # Count records
            with open(jsonl_file, 'r') as f:
                records = sum(1 for line in f if line.strip())
            
            file_size = jsonl_file.stat().st_size
            
            logger.info(f"\n{method.upper()}:")
            logger.info(f"  File: {jsonl_file}")
            logger.info(f"  Status: {'✅ Ready' if records > 0 else '⚠️  Empty'}")
            logger.info(f"  Records: {records}")
            logger.info(f"  Size: {file_size / 1024:.2f} KB")
            
            stats[method] = {
                'status': 'ready' if records > 0 else 'empty',
                'records': records,
                'size_kb': file_size / 1024
            }
        
        return stats


def main():
    """Example usage"""
    
    # Example 1: Process with single method
    logger.info("EXAMPLE 1: Single Method Processing")
    logger.info("="*70)
    
    pipeline = MultiMethodCGARFIntegration(methods=['orcaloca'], limit=3)
    results = pipeline.process_single_method('orcaloca')
    
    logger.info(f"\n✓ Processed {len(results)} instances with OrcaLoca")
    
    # Example 2: Compare methods
    logger.info("\n\nEXAMPLE 2: Compare Multiple Methods")
    logger.info("="*70)
    
    pipeline = MultiMethodCGARFIntegration(
        methods=['orcaloca', 'agentless', 'cosil'],
        limit=10
    )
    
    # Try to compare (will fail for agentless/cosil if not available)
    try:
        locations = pipeline.compare_methods("django__django-11133")
    except Exception as e:
        logger.warning(f"Comparison failed: {e}")
        logger.info("(This is expected if not all methods have data)")
    
    # Example 3: Statistics
    logger.info("\n\nEXAMPLE 3: Statistics")
    logger.info("="*70)
    
    stats = pipeline.collect_statistics()
    
    logger.info("\n" + "="*70)
    logger.info("Summary:")
    logger.info("="*70)
    for method, stat in stats.items():
        status = stat['status']
        if status == 'ready':
            logger.info(f"✅ {method.upper()}: {stat['records']} records ready")
        elif status == 'empty':
            logger.info(f"⚠️  {method.upper()}: No data")
        else:
            logger.info(f"❌ {method.upper()}: File not found")


if __name__ == "__main__":
    main()
