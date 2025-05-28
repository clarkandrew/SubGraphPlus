#!/usr/bin/env python3
"""
SubgraphRAG+ Simple End-to-End Test

A simplified version of the end-to-end test that focuses on core functionality
and handles failures more gracefully. This test is designed to:

1. Verify basic API connectivity
2. Test simple query processing
3. Validate core endpoints
4. Provide clear feedback on what's working and what's not

Usage:
    python scripts/end_to_end_test_simple.py
    python scripts/end_to_end_test_simple.py --minimal
    python scripts/end_to_end_test_simple.py --help
"""

import os
import sys
import time
import json
import requests
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# RULE:import-rich-logger-correctly
from src.app.log import logger

# Rich console for beautiful output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# RULE:uppercase-constants-top
API_BASE_URL = "http://localhost:8000"
TEST_QUESTION = "What is artificial intelligence?"
TIMEOUT_SECONDS = 30
MAX_RETRIES = 3

console = Console()

class SimpleTestResults:
    """Simple container for test results"""
    def __init__(self):
        self.start_time = time.time()
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
        self.warnings = []
        self.results = {}
        
    def add_result(self, test_name: str, success: bool, details: str = "", error: str = ""):
        """Add a test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            status = "✅ PASS"
        else:
            self.tests_failed += 1
            status = "❌ FAIL"
            if error:
                self.errors.append(f"{test_name}: {error}")
        
        self.results[test_name] = {
            'status': status,
            'success': success,
            'details': details,
            'error': error
        }
        
        console.print(f"  {status} {test_name}")
        if details:
            console.print(f"    [dim]{details}[/dim]")
        if error:
            console.print(f"    [red]{error}[/red]")
    
    def add_warning(self, warning: str):
        """Add a warning"""
        self.warnings.append(warning)
        console.print(f"  [yellow]⚠️ {warning}[/yellow]")
    
    def get_total_time(self) -> float:
        return time.time() - self.start_time

def safe_request(method: str, url: str, timeout: int = TIMEOUT_SECONDS, **kwargs) -> Dict[str, Any]:
    """Make a safe HTTP request with error handling"""
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
        return {
            'success': True,
            'status_code': response.status_code,
            'response': response,
            'data': response.json() if response.content and response.headers.get('content-type', '').startswith('application/json') else response.text
        }
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': f'Request timed out after {timeout}s'
        }
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'error': 'Connection refused - server may not be running'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_api_headers() -> Dict[str, str]:
    """Get API headers including authentication if available"""
    headers = {"Content-Type": "application/json"}
    
    # Try to get API key from environment
    api_key = os.environ.get("API_KEY_SECRET")
    if api_key:
        headers["X-API-KEY"] = api_key
    
    return headers

def test_api_health(results: SimpleTestResults) -> bool:
    """Test basic API health"""
    console.print("\n[cyan]🏥 Testing API Health[/cyan]")
    
    # Test health endpoint
    result = safe_request('GET', f"{API_BASE_URL}/healthz")
    if result['success'] and result['status_code'] == 200:
        results.add_result("Health Check", True, f"Response: {result['data']}")
        health_ok = True
    else:
        results.add_result("Health Check", False, error=result.get('error', 'Unknown error'))
        health_ok = False
    
    # Test readiness endpoint
    result = safe_request('GET', f"{API_BASE_URL}/readyz")
    if result['success'] and result['status_code'] == 200:
        details = result['data']
        if isinstance(details, dict):
            checks = details.get('checks', {})
            check_summary = ", ".join([f"{k}: {v}" for k, v in checks.items()])
            results.add_result("Readiness Check", True, f"Checks: {check_summary}")
        else:
            results.add_result("Readiness Check", True, f"Response: {details}")
        ready_ok = True
    else:
        results.add_result("Readiness Check", False, error=result.get('error', 'Unknown error'))
        ready_ok = False
    
    return health_ok and ready_ok

def test_ie_endpoints(results: SimpleTestResults) -> bool:
    """Test Information Extraction endpoints"""
    console.print("\n[cyan]🤖 Testing IE Endpoints[/cyan]")
    
    # Test IE health
    result = safe_request('GET', f"{API_BASE_URL}/ie/health")
    if result['success'] and result['status_code'] == 200:
        data = result['data']
        if isinstance(data, dict):
            status = data.get('status', 'unknown')
            results.add_result("IE Health", True, f"Status: {status}")
        else:
            results.add_result("IE Health", True, f"Response: {data}")
        ie_health_ok = True
    else:
        results.add_result("IE Health", False, error=result.get('error', 'Unknown error'))
        ie_health_ok = False
    
    # Test IE info
    result = safe_request('GET', f"{API_BASE_URL}/ie/info")
    if result['success'] and result['status_code'] == 200:
        data = result['data']
        if isinstance(data, dict):
            models = data.get('models', [])
            model_count = len(models) if isinstance(models, list) else 'unknown'
            results.add_result("IE Info", True, f"Models available: {model_count}")
        else:
            results.add_result("IE Info", True, f"Response: {data}")
        ie_info_ok = True
    else:
        results.add_result("IE Info", False, error=result.get('error', 'Unknown error'))
        ie_info_ok = False
    
    return ie_health_ok and ie_info_ok

def test_simple_query(results: SimpleTestResults, minimal_mode: bool = False) -> bool:
    """Test a simple query"""
    console.print("\n[cyan]🤔 Testing Simple Query[/cyan]")
    
    query_data = {
        "question": TEST_QUESTION,
        "visualize_graph": False  # Keep it simple
    }
    
    # Use a shorter timeout for minimal mode
    timeout = 15 if minimal_mode else TIMEOUT_SECONDS
    
    headers = get_api_headers()
    result = safe_request('POST', f"{API_BASE_URL}/query", 
                         json=query_data, 
                         timeout=timeout,
                         headers=headers)
    
    if result['success']:
        if result['status_code'] == 200:
            # For streaming responses, we just check that we got a response
            results.add_result("Simple Query", True, f"Query accepted, status: {result['status_code']}")
            return True
        else:
            results.add_result("Simple Query", False, 
                             error=f"HTTP {result['status_code']}: {result.get('data', 'Unknown error')}")
            return False
    else:
        results.add_result("Simple Query", False, error=result.get('error', 'Unknown error'))
        return False

def test_graph_browse(results: SimpleTestResults) -> bool:
    """Test graph browsing endpoint"""
    console.print("\n[cyan]🕸️ Testing Graph Browse[/cyan]")
    
    params = {'limit': 10, 'page': 1}
    headers = get_api_headers()
    
    result = safe_request('GET', f"{API_BASE_URL}/graph/browse", 
                         params=params, headers=headers)
    
    if result['success']:
        if result['status_code'] == 200:
            data = result['data']
            if isinstance(data, dict):
                nodes = data.get('nodes', [])
                links = data.get('links', [])
                node_count = len(nodes) if isinstance(nodes, list) else 'unknown'
                link_count = len(links) if isinstance(links, list) else 'unknown'
                results.add_result("Graph Browse", True, f"Nodes: {node_count}, Links: {link_count}")
            else:
                results.add_result("Graph Browse", True, "Response received")
            return True
        elif result['status_code'] == 401:
            results.add_result("Graph Browse", False, error="Authentication required - set API_KEY_SECRET environment variable")
            return False
        else:
            error_msg = f"HTTP {result['status_code']}"
            if result.get('data'):
                error_msg += f": {result['data']}"
            results.add_result("Graph Browse", False, error=error_msg)
            return False
    else:
        results.add_result("Graph Browse", False, error=result.get('error', 'Unknown error'))
        return False

def test_basic_endpoints(results: SimpleTestResults) -> bool:
    """Test basic API endpoints"""
    console.print("\n[cyan]📡 Testing Basic Endpoints[/cyan]")
    
    endpoints = [
        ("/docs", "API Documentation"),
        ("/openapi.json", "OpenAPI Schema"),
    ]
    
    all_ok = True
    for endpoint, name in endpoints:
        result = safe_request('GET', f"{API_BASE_URL}{endpoint}")
        if result['success'] and result['status_code'] == 200:
            results.add_result(name, True, f"Endpoint accessible")
        else:
            results.add_result(name, False, error=result.get('error', f"HTTP {result.get('status_code', 'unknown')}"))
            all_ok = False
    
    return all_ok

def generate_simple_report(results: SimpleTestResults):
    """Generate a simple test report"""
    console.print("\n")
    console.print("=" * 80)
    
    # Summary table
    summary_table = Table(title="🚀 SubgraphRAG+ Simple E2E Test Results", box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Tests", str(results.tests_run))
    summary_table.add_row("Passed", str(results.tests_passed))
    summary_table.add_row("Failed", str(results.tests_failed))
    summary_table.add_row("Success Rate", f"{(results.tests_passed/results.tests_run*100):.1f}%" if results.tests_run > 0 else "0%")
    summary_table.add_row("Total Time", f"{results.get_total_time():.2f}s")
    
    console.print(summary_table)
    
    # Detailed results
    if results.results:
        console.print("\n")
        detail_table = Table(title="Detailed Test Results", box=box.ROUNDED)
        detail_table.add_column("Test", style="cyan")
        detail_table.add_column("Status", style="white")
        detail_table.add_column("Details", style="dim")
        
        for test_name, result in results.results.items():
            details = result.get('details', '')
            if result.get('error'):
                details = f"❌ {result['error']}"
            detail_table.add_row(test_name, result['status'], details)
        
        console.print(detail_table)
    
    # Errors and warnings
    if results.errors:
        console.print("\n")
        error_panel = Panel(
            "\n".join(f"• {error}" for error in results.errors),
            title="❌ Errors",
            border_style="red"
        )
        console.print(error_panel)
    
    if results.warnings:
        console.print("\n")
        warning_panel = Panel(
            "\n".join(f"• {warning}" for warning in results.warnings),
            title="⚠️ Warnings",
            border_style="yellow"
        )
        console.print(warning_panel)
    
    # Overall status
    if results.tests_failed == 0:
        status_text = "🎯 ALL TESTS PASSED"
        status_style = "bold green"
    elif results.tests_passed > results.tests_failed:
        status_text = "⚠️ SOME TESTS FAILED"
        status_style = "bold yellow"
    else:
        status_text = "❌ MOST TESTS FAILED"
        status_style = "bold red"
    
    console.print("\n")
    final_panel = Panel(
        f"{status_text}\n\n"
        f"Passed: {results.tests_passed}/{results.tests_run} tests\n"
        f"Time: {results.get_total_time():.2f} seconds",
        title="Overall Status",
        border_style="green" if results.tests_failed == 0 else "yellow" if results.tests_passed > results.tests_failed else "red"
    )
    console.print(final_panel)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="SubgraphRAG+ Simple End-to-End Test")
    parser.add_argument("--minimal", action="store_true", 
                       help="Run in minimal mode (shorter timeouts, basic tests only)")
    parser.add_argument("--skip-query", action="store_true",
                       help="Skip query processing test (useful if models aren't loaded)")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS,
                       help=f"Timeout for API calls in seconds (default: {TIMEOUT_SECONDS})")
    
    args = parser.parse_args()
    
    # Use local timeout variable
    timeout_seconds = args.timeout
    
    logger.info(f"Started simple E2E test at {datetime.now()}")
    
    # Display header
    mode_info = []
    if args.minimal:
        mode_info.append("MINIMAL")
    if args.skip_query:
        mode_info.append("SKIP QUERY")
    
    title = "🚀 SubgraphRAG+ Simple E2E Test"
    if mode_info:
        title += f" ({', '.join(mode_info)})"
    
    console.print(Panel(
        "Simple End-to-End Integration Test\n"
        "Tests core API functionality with graceful error handling\n\n"
        f"Mode: {'Minimal' if args.minimal else 'Standard'}\n"
                 f"Timeout: {timeout_seconds}s\n"
        f"Target: {API_BASE_URL}",
        title=title,
        border_style="blue"
    ))
    
    results = SimpleTestResults()
    
    # Run tests
    try:
        # Test 1: API Health
        api_healthy = test_api_health(results)
        
        if not api_healthy:
            results.add_warning("API not healthy - some tests may fail")
        
        # Test 2: IE Endpoints
        test_ie_endpoints(results)
        
        # Test 3: Basic Endpoints
        test_basic_endpoints(results)
        
        # Test 4: Graph Browse
        test_graph_browse(results)
        
        # Test 5: Simple Query (optional)
        if not args.skip_query:
            test_simple_query(results, minimal_mode=args.minimal)
        else:
            results.add_warning("Query test skipped by user request")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
        results.add_result("Test Suite", False, error="Interrupted by user")
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        results.add_result("Test Suite", False, error=f"Unexpected error: {e}")
    
    # Generate report
    generate_simple_report(results)
    
    # Log completion
    logger.info(f"Finished simple E2E test at {datetime.now()}")
    
    # Exit with appropriate code
    exit_code = 0 if results.tests_failed == 0 else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 