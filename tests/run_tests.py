#!/usr/bin/env python3
"""
Test runner script for SubgraphRAG+ following testing standards.

This script provides organized test execution with proper logging and reporting.
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from src.app.log import logger


def run_command(cmd: List[str], description: str) -> bool:
    """
    Run a command and log the results.
    
    Args:
        cmd: Command to run as list of strings
        description: Description of what the command does
        
    Returns:
        True if command succeeded, False otherwise
    """
    logger.info(f"Starting {description}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        duration = time.time() - start_time
        logger.info(f"Finished {description} in {duration:.2f}s")
        
        if result.stdout:
            logger.debug(f"STDOUT: {result.stdout}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logger.error(f"Failed {description} after {duration:.2f}s")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"STDERR: {e.stderr}")
        
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
            
        return False


def run_test_suite(test_type: str, path: str, markers: Optional[str] = None) -> bool:
    """
    Run a specific test suite.
    
    Args:
        test_type: Type of tests (unit, integration, etc.)
        path: Path to test directory
        markers: Optional pytest markers to apply
        
    Returns:
        True if all tests passed, False otherwise
    """
    cmd = ["pytest", path, "-v"]
    
    if markers:
        cmd.extend(["-m", markers])
    
    # Add coverage for unit tests
    if test_type == "unit":
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    return run_command(cmd, f"{test_type} tests")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run SubgraphRAG+ tests")
    parser.add_argument(
        "test_type",
        choices=["all", "unit", "integration", "e2e", "performance", "adversarial", "smoke"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--feature",
        help="Run tests for specific feature (e.g., llm, entity_typing)"
    )
    parser.add_argument(
        "--markers",
        help="Pytest markers to apply (e.g., 'not slow')"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only fast tests (excludes slow markers)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Started test runner at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set up test paths
    test_dir = Path(__file__).parent
    
    success = True
    
    if args.fast:
        args.markers = "not slow"
    
    if args.test_type == "all":
        # Run all test types in order
        test_suites = [
            ("smoke", "smoke/"),
            ("unit", "unit/"),
            ("integration", "integration/"),
            ("adversarial", "adversarial/"),
            ("e2e", "e2e/"),
            ("performance", "performance/")
        ]
        
        for test_type, path in test_suites:
            if args.fast and test_type in ["e2e", "performance"]:
                logger.info(f"Skipping {test_type} tests (fast mode)")
                continue
                
            if not run_test_suite(test_type, str(test_dir / path), args.markers):
                success = False
                break
                
    elif args.feature:
        # Run tests for specific feature
        feature_paths = []
        
        # Check unit tests
        unit_path = test_dir / "unit" / args.feature
        if unit_path.exists():
            feature_paths.append(("unit", str(unit_path)))
        
        # Check integration tests
        integration_path = test_dir / "integration" / args.feature
        if integration_path.exists():
            feature_paths.append(("integration", str(integration_path)))
        
        # Check e2e tests
        e2e_path = test_dir / "e2e" / args.feature
        if e2e_path.exists():
            feature_paths.append(("e2e", str(e2e_path)))
        
        if not feature_paths:
            logger.error(f"No tests found for feature: {args.feature}")
            return 1
        
        for test_type, path in feature_paths:
            if not run_test_suite(test_type, path, args.markers):
                success = False
                break
                
    else:
        # Run specific test type
        path = str(test_dir / f"{args.test_type}/")
        success = run_test_suite(args.test_type, path, args.markers)
    
    if success:
        logger.info("All tests passed successfully!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 