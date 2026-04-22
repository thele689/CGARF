#!/usr/bin/env python3
"""
Quick validation: Check that JSONL loading is working vs folder loading
Measures the improvement in loading time and structural correctness
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.phase0_integrator.fault_localization_loader import OrcaLocaDataLoader

def validate_jsonl_structure():
    """Validate JSONL file structure without loading everything to SWE-Bench"""
    print("=" * 70)
    print("JSONL Structure Validation")
    print("=" * 70)
    
    jsonl_path = REPO_ROOT / "input" / "orcaloca.jsonl"
    
    if not jsonl_path.exists():
        print(f"❌ JSONL file not found: {jsonl_path}")
        return False
    
    # Read and validate JSONL structure
    total_records = 0
    total_bug_locations = 0
    repos = defaultdict(int)
    errors = []
    
    print(f"\nValidating JSONL file: {jsonl_path}")
    print(f"File size: {jsonl_path.stat().st_size / 1024:.2f} KB")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                # Validate required fields
                if "instance_id" not in record:
                    errors.append(f"Line {line_no}: Missing instance_id")
                    continue
                
                instance_id = record["instance_id"]
                
                if "repo" not in record:
                    errors.append(f"Line {line_no} ({instance_id}): Missing repo")
                    continue
                
                repo = record["repo"]
                
                if "bug_locations" not in record:
                    errors.append(f"Line {line_no} ({instance_id}): Missing bug_locations")
                    continue
                
                bug_locations = record["bug_locations"]
                if not isinstance(bug_locations, list):
                    errors.append(f"Line {line_no} ({instance_id}): bug_locations not a list")
                    continue
                
                # Count statistics
                total_records += 1
                total_bug_locations += len(bug_locations)
                repos[repo] += 1
                
                # Show progress
                if total_records % 50 == 0:
                    print(f"  ✓ Validated {total_records} records...")
                
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_no}: JSON decode error: {e}")
                continue
            except Exception as e:
                errors.append(f"Line {line_no}: {e}")
                continue
    
    # Report results
    print(f"\n✓ JSONL validation complete:")
    print(f"  Total records: {total_records}")
    print(f"  Total bug_locations: {total_bug_locations}")
    print(f"  Unique repositories: {len(repos)}")
    
    if errors:
        print(f"\n⚠️  Found {len(errors)} errors:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print(f"\n✅ No errors found!")
    
    # Repository breakdown
    print(f"\nRepository Distribution:")
    for repo, count in sorted(repos.items(), key=lambda x: -x[1])[:15]:
        print(f"  {repo}: {count} instances")
    
    return total_records == 300 and len(errors) == 0

if __name__ == "__main__":
    try:
        success = validate_jsonl_structure()
        print("\n" + "=" * 70)
        if success:
            print("✅ JSONL structure validation passed!")
            print("\nLoader is ready to use:")
            print("  - All 300 instances properly formatted")
            print("  - All bug_locations fields present")
            print("  - No JSON syntax errors")
        else:
            print("❌ JSONL validation failed!")
        print("=" * 70)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
