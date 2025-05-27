#!/usr/bin/env python3
"""
Comprehensive Test Runner for SubgraphRAG+

This script runs tests according to the testing standards, with proper
categorization, timeout handling, and reporting.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.app.log import logger, CONSOLEx as CONSOLE


# Test suite configurations
TEST_SUITES = {
    "smoke": {
        "description": "Quick sanity checks (<30s total)",
        "paths": ["tests/smoke/"],
        "timeout": 60,
        "markers": "smoke",
        "parallel": True
    },
    "unit": {
        "description": "Pure logic tests, minimal mocking (<2 minutes)",
        "paths": ["tests/unit/"],
        "timeout": 180,
        "markers": "unit",
        "parallel": True
    },
    "integration": {
        "description": "Real component interactions (<5 minutes)",
        "paths": ["tests/integration/"],
        "timeout": 360,
        "markers": "integration",
        "parallel": False
    },
    "e2e": {
        "description": "End-to-end system flows (<10 minutes)",
        "paths": ["tests/e2e/"],
        "timeout": 720,
        "markers": "e2e",
        "parallel": False
    },
    "performance": {
        "description": "Load and latency benchmarks",
        "paths": ["tests/performance/"],
        "timeout": 900,
        "markers": "performance",
        "parallel": False
    },
    "adversarial": {
        "description": "Robustness and edge cases",
        "paths": ["tests/adversarial/"],
        "timeout": 600,
        "markers": "adversarial",
        "parallel": True
    },
    "all": {
        "description": "All test suites",
        "paths": ["tests/"],
        "timeout": 1800,
        "markers": None,
        "parallel": False
    }
}


def setup_test_environment():
    """Set up environment variables for testing"""
    logger.debug("Setting up test environment")
    
    # Core testing environment
    os.environ['TESTING'] = '1'
    os.environ['SUBGRAPHRAG_DISABLE_MODEL_LOADING'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Disable transformers warnings
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    
    logger.debug("✅ Test environment configured")


def run_pytest_command(
    paths: List[str],
    timeout: int,
    markers: Optional[str] = None,
    parallel: bool = False,
    verbose: bool = False,
    coverage: bool = False,
    debug: bool = False
) -> Dict[str, any]:
    """Run pytest with specified configuration"""
    
    cmd = ["python", "-m", "pytest"]
    
    # Add paths
    cmd.extend(paths)
    
    # Add markers (only if they exist in the test files)
    if markers and markers != "smoke":  # Skip marker filtering for smoke tests for now
        cmd.extend(["-m", markers])
    
    # Add verbosity
    if verbose or debug:
        cmd.append("-v")
    
    if debug:
        cmd.extend(["-s", "--tb=long"])
    else:
        cmd.append("--tb=short")
    
    # Add parallel execution (if pytest-xdist is available)
    if parallel:
        try:
            import pytest_xdist
            cmd.extend(["-n", "auto"])
        except ImportError:
            logger.debug("pytest-xdist not available, running tests sequentially")
    
    # Add coverage (if pytest-cov is available)
    if coverage:
        try:
            import pytest_cov
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:tests/coverage_html"
            ])
        except ImportError:
            logger.debug("pytest-cov not available, running tests without coverage")
    
    # Add timeout (if pytest-timeout is available)
    try:
        import pytest_timeout
        cmd.extend(["--timeout", str(timeout)])
    except ImportError:
        logger.debug("pytest-timeout not available, running tests without timeout")
    
    logger.debug(f"Running command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,  # Add buffer to pytest timeout
            cwd=project_root
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration,
            "timeout": False
        }
        
    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Test suite timed out after {duration:.1f}s",
            "duration": duration,
            "timeout": True
        }


def print_test_results(suite_name: str, result: Dict[str, any]):
    """Print formatted test results"""
    
    if result["success"]:
        CONSOLE.print(f"✅ {suite_name.upper()} TESTS PASSED", style="bold green")
    else:
        CONSOLE.print(f"❌ {suite_name.upper()} TESTS FAILED", style="bold red")
    
    CONSOLE.print(f"Duration: {result['duration']:.1f}s")
    CONSOLE.print(f"Return code: {result['returncode']}")
    
    if result["timeout"]:
        CONSOLE.print("⏰ Test suite timed out", style="bold yellow")
    
    if result["stdout"]:
        CONSOLE.print("\n[bold]STDOUT:[/bold]")
        CONSOLE.print(result["stdout"])
    
    if result["stderr"]:
        CONSOLE.print("\n[bold red]STDERR:[/bold red]")
        CONSOLE.print(result["stderr"])
    
    CONSOLE.print("-" * 80)


def run_test_suite(
    suite_name: str,
    verbose: bool = False,
    coverage: bool = False,
    debug: bool = False
) -> bool:
    """Run a specific test suite"""
    
    if suite_name not in TEST_SUITES:
        CONSOLE.print(f"❌ Unknown test suite: {suite_name}", style="bold red")
        return False
    
    suite_config = TEST_SUITES[suite_name]
    
    CONSOLE.print(f"\n🧪 Running {suite_name.upper()} tests", style="bold blue")
    CONSOLE.print(f"Description: {suite_config['description']}")
    CONSOLE.print(f"Paths: {', '.join(suite_config['paths'])}")
    CONSOLE.print(f"Timeout: {suite_config['timeout']}s")
    CONSOLE.print("-" * 80)
    
    # Check if test paths exist
    for path in suite_config['paths']:
        test_path = project_root / path
        if not test_path.exists():
            CONSOLE.print(f"⚠️  Test path does not exist: {test_path}", style="yellow")
            continue
    
    result = run_pytest_command(
        paths=suite_config['paths'],
        timeout=suite_config['timeout'],
        markers=suite_config['markers'],
        parallel=suite_config['parallel'],
        verbose=verbose,
        coverage=coverage,
        debug=debug
    )
    
    print_test_results(suite_name, result)
    
    return result["success"]


def main():
    """Main test runner function"""
    
    parser = argparse.ArgumentParser(
        description="SubgraphRAG+ Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available test suites:
  smoke        - Quick sanity checks (<30s total)
  unit         - Pure logic tests, minimal mocking (<2 minutes)
  integration  - Real component interactions (<5 minutes)
  e2e          - End-to-end system flows (<10 minutes)
  performance  - Load and latency benchmarks
  adversarial  - Robustness and edge cases
  all          - All test suites

Examples:
  python tests/run_tests.py --suite smoke
  python tests/run_tests.py --suite unit --verbose
  python tests/run_tests.py --suite all --coverage
        """
    )
    
    parser.add_argument(
        "--suite",
        choices=list(TEST_SUITES.keys()),
        default="smoke",
        help="Test suite to run (default: smoke)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed output"
    )
    
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="List available test suites and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_suites:
        CONSOLE.print("\n📋 Available Test Suites:", style="bold blue")
        for name, config in TEST_SUITES.items():
            CONSOLE.print(f"  {name:12} - {config['description']}")
        return 0
    
    # Setup environment
    setup_test_environment()
    
    # Run tests
    logger.info(f"Starting {args.suite} test suite")
    
    success = run_test_suite(
        suite_name=args.suite,
        verbose=args.verbose,
        coverage=args.coverage,
        debug=args.debug
    )
    
    if success:
        CONSOLE.print(f"\n🎉 {args.suite.upper()} tests completed successfully!", style="bold green")
        logger.info(f"{args.suite} tests passed")
        return 0
    else:
        CONSOLE.print(f"\n💥 {args.suite.upper()} tests failed!", style="bold red")
        logger.error(f"{args.suite} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 