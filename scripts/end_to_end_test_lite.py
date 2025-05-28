#!/usr/bin/env python3
"""
SubgraphRAG+ End-to-End Integration Test Script (Lite Version)

This is a lighter version of the end-to-end test that focuses on core functionality
without triggering heavy model loading that might cause memory issues.

This script validates:
1. System health and dependencies
2. Basic API endpoints
3. Neo4j and FAISS connectivity
4. Graph browsing functionality
5. Basic query processing (without heavy model loading)

Usage:
    python scripts/end_to_end_test_lite.py
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# RULE:import-rich-logger-correctly
try:
    from src.app.log import logger
    from src.app.config import config, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, API_KEY_SECRET
except ImportError:
    # Fallback for different import paths
    try:
        from app.log import logger
        from app.config import config, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, API_KEY_SECRET
    except ImportError:
        # Create minimal logger if imports fail
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Try to load config manually
        import json
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        # Environment variables
        NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
        NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
        API_KEY_SECRET = os.environ.get("API_KEY_SECRET", "default_key_for_dev_only")

# Rich console for beautiful output
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich import box

# Neo4j for direct validation
try:
    from neo4j import GraphDatabase
except ImportError as e:
    logger.error(f"Missing required dependencies: {e}")
    sys.exit(1)

# RULE:uppercase-constants-top
API_BASE_URL = "http://localhost:8000"
TEST_QUESTION = "What is the main topic discussed in the document?"
TIMEOUT_SECONDS = 30  # Shorter timeout for lite version
MIN_REQUIRED_DISK_SPACE_GB = 1

console = Console()

class LiteValidationResults:
    """Container for lite validation results"""
    def __init__(self):
        self.start_time = time.time()
        self.system_health = {}
        self.api_endpoints_working = False
        self.neo4j_accessible = False
        self.faiss_accessible = False
        self.graph_browse_working = False
        self.basic_query_working = False
        self.errors = []
        self.warnings = []
        
    def add_error(self, error: str):
        self.errors.append(error)
        logger.error(error)
        
    def add_warning(self, warning: str):
        self.warnings.append(warning)
        logger.warning(warning)
        
    def get_total_time(self) -> float:
        return time.time() - self.start_time

def check_system_requirements() -> Dict[str, bool]:
    """Check basic system requirements"""
    logger.debug("Starting system requirements check")
    
    checks = {}
    
    # Python version
    current_version = sys.version_info[:2]
    checks['python_version'] = current_version >= (3, 8)
    if not checks['python_version']:
        console.print(f"[red]Python 3.8+ required, got {current_version}[/red]")
    
    # Disk space (simplified check)
    try:
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)
        checks['disk_space'] = free_space_gb >= MIN_REQUIRED_DISK_SPACE_GB
        if not checks['disk_space']:
            console.print(f"[red]Insufficient disk space: {free_space_gb:.1f}GB available, {MIN_REQUIRED_DISK_SPACE_GB}GB required[/red]")
    except Exception as e:
        checks['disk_space'] = False
        console.print(f"[red]Could not check disk space: {e}[/red]")
    
    logger.debug("Finished system requirements check")
    return checks

def check_api_health() -> Dict[str, Any]:
    """Check API endpoint health"""
    logger.debug("Starting API health check")
    
    health_status = {}
    
    # Main API health check
    try:
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=TIMEOUT_SECONDS)
        health_status['main_api'] = {
            'status': response.status_code == 200,
            'response_time_ms': response.elapsed.total_seconds() * 1000,
            'details': response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        health_status['main_api'] = {
            'status': False,
            'error': str(e)
        }
    
    # Main API readiness check
    try:
        response = requests.get(f"{API_BASE_URL}/readyz", timeout=TIMEOUT_SECONDS)
        health_status['main_api_ready'] = {
            'status': response.status_code == 200,
            'response_time_ms': response.elapsed.total_seconds() * 1000,
            'details': response.json() if response.content else {}
        }
    except Exception as e:
        health_status['main_api_ready'] = {
            'status': False,
            'error': str(e)
        }
    
    logger.debug("Finished API health check")
    return health_status

def check_neo4j_connection() -> Dict[str, Any]:
    """Check Neo4j database connection and get basic stats"""
    logger.debug("Starting Neo4j connection check")
    
    try:
        # For neo4j+s:// URIs, encryption is already specified in the URI scheme
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Test connection
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            
            if test_value != 1:
                raise Exception("Neo4j connection test failed")
            
            # Get basic statistics
            entity_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r:REL]->() RETURN count(r) as count").single()["count"]
            
            driver.close()
            
            return {
                'status': True,
                'entity_count': entity_count,
                'relationship_count': rel_count
            }
            
    except Exception as e:
        return {
            'status': False,
            'error': str(e)
        }

def test_graph_browse() -> Dict[str, Any]:
    """Test graph browsing API"""
    logger.debug("Starting graph browse test")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/graph/browse",
            params={'limit': 10, 'page': 1},
            headers={"X-API-KEY": API_KEY_SECRET},
            timeout=TIMEOUT_SECONDS
        )
        
        if response.status_code != 200:
            return {
                'status': False,
                'response_code': response.status_code,
                'error': response.text
            }
        
        data = response.json()
        
        # Validate structure
        required_fields = ['nodes', 'links', 'page', 'limit', 'total_nodes_in_filter', 'total_links_in_filter', 'has_more']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return {
                'status': False,
                'error': f"Missing required fields: {missing_fields}"
            }
        
        return {
            'status': True,
            'node_count': len(data['nodes']),
            'link_count': len(data['links']),
            'total_nodes': data['total_nodes_in_filter'],
            'total_links': data['total_links_in_filter'],
            'has_more': data['has_more']
        }
        
    except Exception as e:
        return {
            'status': False,
            'error': str(e)
        }

def test_basic_query() -> Dict[str, Any]:
    """Test basic query processing without heavy model loading"""
    logger.debug("Starting basic query test")
    
    try:
        # Use a simple query that shouldn't trigger heavy model loading
        simple_question = "test"
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "question": simple_question,
                "visualize_graph": False  # Disable graph visualization to reduce load
            },
            headers={"X-API-KEY": API_KEY_SECRET},
            timeout=TIMEOUT_SECONDS,
            stream=True
        )
        
        if response.status_code != 200:
            return {
                'status': False,
                'response_code': response.status_code,
                'error': response.text
            }
        
        # Just check that we get a response, don't parse the full stream
        # to avoid triggering heavy processing
        return {
            'status': True,
            'response_code': response.status_code,
            'note': 'Basic query endpoint accessible'
        }
        
    except Exception as e:
        return {
            'status': False,
            'error': str(e)
        }

def generate_lite_report(results: LiteValidationResults) -> None:
    """Generate lite validation report"""
    logger.debug("Generating lite validation report")
    
    # Header
    header_text = Text("🚀 SubgraphRAG+ Lite End-to-End Test Results", style="bold blue")
    header_panel = Panel(header_text, box=box.DOUBLE)
    
    console.print(header_panel)
    console.print()
    
    # System Health Table
    health_table = Table(title="System Health Checks", box=box.ROUNDED)
    health_table.add_column("Component", style="cyan")
    health_table.add_column("Status", style="green")
    health_table.add_column("Details", style="dim")
    
    for component, status in results.system_health.items():
        if isinstance(status, dict):
            status_text = "✅ HEALTHY" if status.get('status', False) else "❌ FAILED"
            details = status.get('details', status.get('error', ''))
            if isinstance(details, dict):
                details = json.dumps(details, indent=2)
        else:
            status_text = "✅ OK" if status else "❌ FAILED"
            details = ""
        
        health_table.add_row(component.replace('_', ' ').title(), status_text, str(details)[:100])
    
    console.print(health_table)
    console.print()
    
    # Core Functionality Table
    func_table = Table(title="Core Functionality Tests", box=box.ROUNDED)
    func_table.add_column("Test", style="cyan")
    func_table.add_column("Status", style="green")
    
    func_table.add_row("API Endpoints", "✅ WORKING" if results.api_endpoints_working else "❌ FAILED")
    func_table.add_row("Neo4j Database", "✅ ACCESSIBLE" if results.neo4j_accessible else "❌ FAILED")
    func_table.add_row("FAISS Index", "✅ ACCESSIBLE" if results.faiss_accessible else "❌ FAILED")
    func_table.add_row("Graph Browsing", "✅ WORKING" if results.graph_browse_working else "❌ FAILED")
    func_table.add_row("Basic Query", "✅ WORKING" if results.basic_query_working else "❌ FAILED")
    
    console.print(func_table)
    console.print()
    
    # Errors and Warnings
    if results.errors:
        error_panel = Panel(
            "\n".join(f"❌ {error}" for error in results.errors),
            title="Errors",
            border_style="red"
        )
        console.print(error_panel)
        console.print()
    
    if results.warnings:
        warning_panel = Panel(
            "\n".join(f"⚠️  {warning}" for warning in results.warnings),
            title="Warnings",
            border_style="yellow"
        )
        console.print(warning_panel)
        console.print()
    
    # Overall Status
    overall_success = (
        results.api_endpoints_working and
        results.neo4j_accessible and
        results.faiss_accessible and
        len(results.errors) == 0
    )
    
    status_text = "🎯 CORE SYSTEMS OPERATIONAL" if overall_success else "⚠️  ISSUES DETECTED"
    status_style = "bold green" if overall_success else "bold yellow"
    
    total_time = results.get_total_time()
    
    final_panel = Panel(
        f"{status_text}\n\nTotal execution time: {total_time:.2f} seconds\n\nNote: This is a lite test that avoids heavy model loading.",
        title="Overall Status",
        border_style="green" if overall_success else "yellow"
    )
    
    console.print(final_panel)

def main():
    """Main execution function"""
    logger.info(f"Started {__file__} at {datetime.now()}")
    
    console.print(Panel(
        "SubgraphRAG+ Lite End-to-End Integration Test\n"
        "This script validates core system functionality without heavy model loading",
        title="🚀 Starting Lite E2E Test",
        border_style="blue"
    ))
    
    results = LiteValidationResults()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Step 1: System Requirements
        task1 = progress.add_task("Checking system requirements...", total=100)
        req_checks = check_system_requirements()
        results.system_health.update(req_checks)
        progress.update(task1, completed=100)
        
        # Step 2: API Health Checks
        task2 = progress.add_task("Checking API health...", total=100)
        api_health = check_api_health()
        results.system_health.update(api_health)
        results.api_endpoints_working = api_health.get('main_api', {}).get('status', False)
        progress.update(task2, completed=100)
        
        # Step 3: Neo4j Connection
        task3 = progress.add_task("Testing Neo4j connection...", total=100)
        neo4j_status = check_neo4j_connection()
        results.system_health['neo4j_test'] = neo4j_status
        results.neo4j_accessible = neo4j_status['status']
        if neo4j_status['status']:
            console.print(f"[green]✅ Neo4j: {neo4j_status['entity_count']} entities, {neo4j_status['relationship_count']} relationships[/green]")
        progress.update(task3, completed=100)
        
        # Step 4: FAISS Check (via readiness endpoint)
        task4 = progress.add_task("Checking FAISS status...", total=100)
        readiness = api_health.get('main_api_ready', {}).get('details', {})
        if isinstance(readiness, dict) and 'checks' in readiness:
            results.faiss_accessible = readiness['checks'].get('faiss_index') == 'ok'
        progress.update(task4, completed=100)
        
        # Step 5: Graph Browse Test
        task5 = progress.add_task("Testing graph browsing...", total=100)
        browse_result = test_graph_browse()
        results.system_health['graph_browse'] = browse_result
        results.graph_browse_working = browse_result['status']
        if browse_result['status']:
            console.print(f"[green]✅ Graph Browse: {browse_result['node_count']} nodes, {browse_result['link_count']} links[/green]")
        progress.update(task5, completed=100)
        
        # Step 6: Basic Query Test
        task6 = progress.add_task("Testing basic query...", total=100)
        query_result = test_basic_query()
        results.system_health['basic_query'] = query_result
        results.basic_query_working = query_result['status']
        progress.update(task6, completed=100)
    
    # Generate final report
    console.print("\n")
    generate_lite_report(results)
    
    # Log completion
    logger.info(f"Finished {__file__} at {datetime.now()}")
    
    # Exit with appropriate code
    exit_code = 0 if len(results.errors) == 0 else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 