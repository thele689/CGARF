#!/usr/bin/env python
"""Run tests for CGARF"""

import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run CGARF tests")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage analysis"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if args.type == "unit":
        cmd.append("tests/unit")
    elif args.type == "integration":
        cmd.append("tests/integration")
    else:
        cmd.append("tests")
    
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])
    
    # Run tests
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
