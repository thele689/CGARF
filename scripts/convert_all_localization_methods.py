#!/usr/bin/env python3
"""
Convert multiple fault localization method outputs to unified JSONL format

Supports:
  - OrcaLoca: src/orcaloca
  - Agentless: src/agentless
  - CoSIL: src/cosil

Usage:
  python convert_all_localization_methods.py
  python convert_all_localization_methods.py --method orcaloca
  python convert_all_localization_methods.py --method agentless --input-dir ../data/agentless
"""

import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=''), 
           format="<level>{level: <8}</level> | {message}",
           level="INFO")


class LocalizationConverter:
    """Convert localization method outputs to unified JSONL format"""
    
    METHODS = {
        'orcaloca': {
            'input_dir': str(REPO_ROOT / 'input' / 'Orcaloca'),
            'output_file': str(REPO_ROOT / 'input' / 'orcaloca.jsonl'),
            'description': 'OrcaLoca - Observable Recursive Code Analysis'
        },
        'agentless': {
            'input_dir': str(REPO_ROOT / 'input' / 'Agentless'),
            'output_file': str(REPO_ROOT / 'input' / 'agentless.jsonl'),
            'description': 'Agentless - LLM-based fault localization'
        },
        'cosil': {
            'input_dir': str(REPO_ROOT / 'input' / 'CoSIL'),
            'output_file': str(REPO_ROOT / 'input' / 'cosil.jsonl'),
            'description': 'CoSIL - Context-aware Statistical Information Localization'
        }
    }
    
    def __init__(self, method: str, input_dir: Optional[str] = None):
        """
        Initialize converter
        
        Args:
            method: One of 'orcaloca', 'agentless', 'cosil'
            input_dir: Override input directory (optional)
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from: {list(self.METHODS.keys())}")
        
        self.method = method
        self.method_config = self.METHODS[method]
        self.input_dir = Path(input_dir) if input_dir else Path(self.method_config['input_dir'])
        self.output_file = Path(self.method_config['output_file'])
        
        logger.info(f"Converter initialized for {method}")
        logger.info(f"  Input: {self.input_dir}")
        logger.info(f"  Output: {self.output_file}")
    
    def _parse_instance_id(self, folder_name: str) -> tuple:
        """
        Parse instance_id to extract repo name
        
        Examples:
          'astropy__astropy-12907' → ('astropy/astropy', 'astropy__astropy-12907')
          'django__django-14182' → ('django/django', 'django__django-14182')
        
        Args:
            folder_name: Instance folder name
        
        Returns:
            (repo_name, instance_id) tuple
        """
        instance_id = folder_name
        
        # Extract repo from instance_id format: owner__repo-number
        parts = folder_name.split('__')
        if len(parts) >= 2:
            owner = parts[0]
            # repo-number format: extract repo name
            rest = '__'.join(parts[1:])
            repo_part = rest.rsplit('-', 1)[0]  # Remove trailing number
            repo = f"{owner}/{repo_part}"
        else:
            # Fallback
            repo = folder_name.replace('_', '/').rsplit('-', 1)[0]
        
        return repo, instance_id
    
    def _find_localization_file(self, folder_path: Path) -> Optional[Path]:
        """
        Find the localization output file in folder
        
        Supports multiple naming conventions:
          - searcher_*.json (OrcaLoca)
          - localization_*.json
          - results.json
          - output.json
        
        Args:
            folder_path: Path to instance folder
        
        Returns:
            Path to file, or None if not found
        """
        patterns = [
            f'searcher_{folder_path.name}.json',  # OrcaLoca format
            f'localization_{folder_path.name}.json',  # Generic format
            'localization.json',
            'results.json',
            'output.json'
        ]
        
        for pattern in patterns:
            file_path = folder_path / pattern
            if file_path.exists():
                return file_path
        
        # Try glob patterns
        for glob_pattern in ['searcher_*.json', 'localization*.json', '*.json']:
            matches = list(folder_path.glob(glob_pattern))
            if matches:
                return matches[0]
        
        return None
    
    def _load_localization_data(self, file_path: Path) -> Optional[Dict]:
        """
        Load localization data from file
        
        Args:
            file_path: Path to localization output file
        
        Returns:
            Dictionary with localization data, or None if load fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            return None
    
    def convert(self) -> bool:
        """
        Convert all instances to JSONL format
        
        Returns:
            True if successful, False otherwise
        """
        if not self.input_dir.exists():
            logger.error(f"Input directory not found: {self.input_dir}")
            logger.info(f"Please ensure {self.method} data is available at: {self.input_dir}")
            return False
        
        # Find all instance folders
        instance_folders = sorted([d for d in self.input_dir.iterdir() if d.is_dir()])
        
        if not instance_folders:
            logger.warning(f"No instance folders found in {self.input_dir}")
            return False
        
        logger.info(f"Found {len(instance_folders)} instances to convert")
        
        # Convert instances
        success_count = 0
        fail_count = 0
        records = []
        
        for i, folder in enumerate(instance_folders, 1):
            try:
                # Find localization file
                loc_file = self._find_localization_file(folder)
                if not loc_file:
                    logger.warning(f"  ✗ No localization file in {folder.name}")
                    fail_count += 1
                    continue
                
                # Load data
                data = self._load_localization_data(loc_file)
                if not data:
                    fail_count += 1
                    continue
                
                # Parse instance_id and repo
                repo, instance_id = self._parse_instance_id(folder.name)
                
                # Extract bug_locations (support multiple formats)
                bug_locations = data.get('bug_locations', 
                                        data.get('fault_locations',
                                                data.get('localization', [])))
                
                # Ensure bug_locations is a list of dicts
                if not isinstance(bug_locations, list):
                    bug_locations = []
                elif bug_locations and isinstance(bug_locations[0], str):
                    # If strings, convert to minimal format
                    bug_locations = [{'file_path': loc} for loc in bug_locations]
                
                # Create JSONL record
                record = {
                    'instance_id': instance_id,
                    'repo': repo,
                    'bug_locations': bug_locations,
                    'conclusion': data.get('conclusion', '')
                }
                
                records.append(record)
                success_count += 1
                
                if i % 50 == 0:
                    logger.info(f"  ✓ Processed {i} instances...")
            
            except Exception as e:
                logger.warning(f"  ✗ Error processing {folder.name}: {e}")
                fail_count += 1
                continue
        
        # Write JSONL file
        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Calculate statistics
            total_bug_locations = sum(len(r.get('bug_locations', [])) for r in records)
            
            logger.info(f"\n✅ Conversion completed!")
            logger.info(f"  Success: {success_count} instances")
            logger.info(f"  Failed: {fail_count} instances")
            logger.info(f"  Output: {self.output_file}")
            logger.info(f"  File size: {self.output_file.stat().st_size / 1024:.2f} KB")
            logger.info(f"  Total bug_locations: {total_bug_locations}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to write JSONL file: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert fault localization outputs to unified JSONL format'
    )
    parser.add_argument(
        '--method',
        choices=['orcaloca', 'agentless', 'cosil', 'all'],
        default='all',
        help='Localization method to convert (default: all)'
    )
    parser.add_argument(
        '--input-dir',
        help='Override input directory'
    )
    
    args = parser.parse_args()
    
    # Determine which methods to convert
    methods = ['orcaloca', 'agentless', 'cosil'] if args.method == 'all' else [args.method]
    
    logger.info("=" * 70)
    logger.info("Fault Localization Multi-Method Converter")
    logger.info("=" * 70)
    
    results = {}
    
    for method in methods:
        logger.info(f"\n[{method.upper()}] {LocalizationConverter.METHODS[method]['description']}")
        logger.info("-" * 70)
        
        try:
            converter = LocalizationConverter(method, args.input_dir)
            success = converter.convert()
            results[method] = 'SUCCESS' if success else 'FAILED'
        except Exception as e:
            logger.error(f"Error: {e}")
            results[method] = 'ERROR'
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    for method, status in results.items():
        symbol = '✅' if status == 'SUCCESS' else '❌'
        logger.info(f"  {symbol} {method.upper()}: {status}")


if __name__ == '__main__':
    main()
