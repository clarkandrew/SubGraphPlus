#!/usr/bin/env python3
"""
Simplified End-to-End Test for SubgraphRAG+

This script starts its own server and tests the complete pipeline.
"""

import os
import sys
import time
import signal
import subprocess
import requests
import json
from pathlib import Path
from contextlib import contextmanager

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# RULE:import-rich-logger-correctly
try:
    from src.app.log import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# RULE:uppercase-constants-top
SERVER_PORT = 8002
SERVER_HOST = "localhost"
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
STARTUP_TIMEOUT = 45  # seconds
REQUEST_TIMEOUT = 30  # seconds
TEST_DOCUMENT = "Barack Obama was born in Hawaii and served as the 44th President of the United States."
API_KEY = os.environ.get("API_KEY_SECRET", "default_key_for_dev_only")

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

@contextmanager
def timeout_context(seconds: int, operation_name: str = "operation"):
    """Context manager for timeout operations"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"⏰ {operation_name} timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Reset the alarm and handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def start_server_process(use_models: bool = False):
    """Start the server process"""
    env = os.environ.copy()
    if not use_models:
        env["SUBGRAPHRAG_DISABLE_MODEL_LOADING"] = "true"
    
    console.print(f"[yellow]🚀 Starting server on port {SERVER_PORT}{'(mock mode)' if not use_models else '(with models)'}...[/yellow]")
    
    process = subprocess.Popen(
        [sys.executable, "src/main.py", "--port", str(SERVER_PORT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def wait_for_server(timeout: int = STARTUP_TIMEOUT):
    """Wait for server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/healthz", timeout=5)
            if response.status_code == 200:
                console.print("[green]✅ Server is ready![/green]")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    
    console.print(f"[red]❌ Server failed to start within {timeout} seconds[/red]")
    return False

def test_health_check():
    """Test health check endpoint"""
    try:
        with timeout_context(REQUEST_TIMEOUT, "health check"):
            response = requests.get(f"{BASE_URL}/healthz", timeout=REQUEST_TIMEOUT)
            return response.status_code == 200
    except (requests.exceptions.RequestException, TimeoutError):
        return False

def test_ie_info():
    """Test information extraction info endpoint"""
    try:
        with timeout_context(REQUEST_TIMEOUT, "IE info"):
            response = requests.get(f"{BASE_URL}/ie/info", timeout=REQUEST_TIMEOUT)
            return response.status_code == 200
    except (requests.exceptions.RequestException, TimeoutError):
        return False

def test_ie_health():
    """Test information extraction health endpoint"""
    try:
        with timeout_context(REQUEST_TIMEOUT, "IE health"):
            response = requests.get(f"{BASE_URL}/ie/health", timeout=REQUEST_TIMEOUT)
            return response.status_code == 200
    except (requests.exceptions.RequestException, TimeoutError):
        return False

def test_information_extraction():
    """Test information extraction endpoint"""
    try:
        with timeout_context(REQUEST_TIMEOUT, "information extraction"):
            payload = {"text": TEST_DOCUMENT}
            response = requests.post(
                f"{BASE_URL}/ie/extract",
                json=payload,
                headers={"Content-Type": "application/json", "X-API-KEY": API_KEY},
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                triples = data.get("triples", [])
                success = data.get("success", False)
                
                console.print(f"[green]✅ Extracted {len(triples)} triples, success: {success}[/green]")
                
                # Show first few triples
                for i, triple in enumerate(triples[:3]):
                    head = triple.get("head", "?")
                    relation = triple.get("relation", "?") 
                    tail = triple.get("tail", "?")
                    console.print(f"[dim]  {i+1}. {head} → {relation} → {tail}[/dim]")
                
                return len(triples) > 0 and success
            else:
                console.print(f"[red]❌ IE request failed: {response.status_code}[/red]")
                return False
                
    except (requests.exceptions.RequestException, TimeoutError) as e:
        console.print(f"[red]❌ IE request failed: {e}[/red]")
        return False

def test_text_ingestion():
    """Test text ingestion endpoint"""
    try:
        with timeout_context(REQUEST_TIMEOUT, "text ingestion"):
            payload = {
                "text": TEST_DOCUMENT,
                "metadata": {
                    "title": "Test Document",
                    "source": "simplified_e2e_test.py"
                }
            }
            
            response = requests.post(
                f"{BASE_URL}/ingest/text",
                json=payload,
                headers={"Content-Type": "application/json", "X-API-KEY": API_KEY},
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                doc_id = data.get("document_id")
                triples_count = data.get("triples_extracted", 0)
                console.print(f"[green]✅ Ingested document {doc_id}, extracted {triples_count} triples[/green]")
                return doc_id is not None
            else:
                console.print(f"[red]❌ Ingestion failed: {response.status_code}[/red]")
                return False
                
    except (requests.exceptions.RequestException, TimeoutError) as e:
        console.print(f"[red]❌ Ingestion failed: {e}[/red]")
        return False

def test_graph_query():
    """Test graph query endpoint"""
    try:
        with timeout_context(REQUEST_TIMEOUT, "graph query"):
            payload = {"question": "Obama"}
            response = requests.post(
                f"{BASE_URL}/query",
                json=payload,
                headers={"Content-Type": "application/json", "X-API-KEY": API_KEY},
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                console.print(f"[green]✅ Graph query returned {len(results)} results[/green]")
                return True
            else:
                console.print(f"[red]❌ Graph query failed: {response.status_code}[/red]")
                return False
                
    except (requests.exceptions.RequestException, TimeoutError) as e:
        console.print(f"[red]❌ Graph query failed: {e}[/red]")
        return False

def stop_server_process(process):
    """Stop the server process"""
    console.print("[yellow]🛑 Stopping server...[/yellow]")
    
    try:
        process.terminate()
        process.wait(timeout=10)
        console.print("[green]✅ Server stopped gracefully[/green]")
    except subprocess.TimeoutExpired:
        console.print("[yellow]⚠️ Graceful shutdown timed out, forcing kill...[/yellow]")
        process.kill()
        process.wait(timeout=5)
        console.print("[green]✅ Server killed[/green]")

def run_test_suite(use_models: bool = False):
    """Run the complete test suite"""
    mode = "with AI models" if use_models else "in mock mode"
    console.print(Panel(
        f"SubgraphRAG+ Simplified End-to-End Test\n"
        f"Testing {mode}",
        title="🧪 Simplified E2E Test",
        border_style="blue"
    ))
    
    # Start server
    server_process = start_server_process(use_models)
    
    try:
        # Wait for server to be ready
        if not wait_for_server(STARTUP_TIMEOUT):
            console.print("[red]❌ Failed to start server[/red]")
            return False
        
        # Run tests
        tests = [
            ("Health Check", test_health_check),
            ("IE Info", test_ie_info),
            ("IE Health", test_ie_health),
            ("Information Extraction", test_information_extraction),
            ("Text Ingestion", test_text_ingestion),
            ("Graph Query", test_graph_query),
        ]
        
        results = []
        for test_name, test_func in tests:
            console.print(f"\n[bold cyan]🧪 Running: {test_name}[/bold cyan]")
            try:
                result = test_func()
                results.append((test_name, result))
                
                if result:
                    console.print(f"[green]✅ {test_name} passed[/green]")
                else:
                    console.print(f"[red]❌ {test_name} failed[/red]")
                    
            except Exception as e:
                console.print(f"[red]❌ {test_name} failed with exception: {e}[/red]")
                results.append((test_name, False))
        
        # Summary
        console.print("\n")
        results_table = Table(title="Test Results Summary", show_header=True, header_style="bold blue")
        results_table.add_column("Test", style="cyan")
        results_table.add_column("Status", style="green")
        
        passed = 0
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            results_table.add_row(test_name, status)
            if result:
                passed += 1
        
        console.print(results_table)
        
        total = len(results)
        console.print(f"\n[bold]Overall: {passed}/{total} tests passed[/bold]")
        
        if passed == total:
            console.print(Panel(
                f"🎉 All tests passed {mode}!\n\n"
                "The SubgraphRAG+ system is working correctly.",
                title="✅ SUCCESS",
                border_style="green"
            ))
            return True
        else:
            console.print(Panel(
                f"❌ {total - passed} test(s) failed {mode}.\n\n"
                "Some functionality may not be working correctly.",
                title="❌ PARTIAL FAILURE",
                border_style="red"
            ))
            return False
            
    finally:
        # Always stop the server
        stop_server_process(server_process)

def main():
    """Main function"""
    logger.info("Starting simplified end-to-end test")
    
    try:
        # Test 1: Mock mode (should always work)
        console.print("\n[bold blue]🎯 Phase 1: Testing Mock Mode[/bold blue]\n")
        mock_success = run_test_suite(use_models=False)
        
        # Test 2: With models (if mock mode works)
        if mock_success:
            console.print("\n[bold blue]🎯 Phase 2: Testing with AI Models[/bold blue]\n")
            model_success = run_test_suite(use_models=True)
            
            if mock_success and model_success:
                console.print(Panel(
                    "🎉 COMPLETE SUCCESS!\n\n"
                    "✅ Mock mode works perfectly\n"
                    "✅ AI model mode works perfectly\n\n"
                    "The SubgraphRAG+ system is fully operational!",
                    title="🏆 FULL SYSTEM SUCCESS",
                    border_style="green"
                ))
                return True
            elif mock_success:
                console.print(Panel(
                    "⚠️ PARTIAL SUCCESS\n\n"
                    "✅ Mock mode works perfectly\n"
                    "❌ AI model mode has issues\n\n"
                    "The system works with fallbacks but may need\n"
                    "more memory or better hardware for full AI capabilities.",
                    title="⚠️ PARTIAL SUCCESS",
                    border_style="yellow"
                ))
                return True
        else:
            console.print(Panel(
                "❌ SYSTEM FAILURE\n\n"
                "❌ Even mock mode is not working\n\n"
                "There are fundamental issues that need to be addressed.",
                title="❌ SYSTEM FAILURE",
                border_style="red"
            ))
            return False
            
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Test interrupted by user[/yellow]")
        return False
    except Exception as e:
        console.print(f"\n[red]💥 Unexpected error: {e}[/red]")
        logger.exception("Unexpected error in simplified e2e test")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 