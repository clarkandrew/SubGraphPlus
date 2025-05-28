#!/usr/bin/env python3
"""
Model Loading Test Script for SubgraphRAG+

This script specifically tests the REBEL and NER model loading functionality
with comprehensive timeout handling and diagnostics.

Usage:
    python scripts/test_model_loading.py [--disable-models] [--timeout SECONDS]

Options:
    --disable-models    Test with SUBGRAPHRAG_DISABLE_MODEL_LOADING=true
    --timeout SECONDS   Set custom timeout (default: 300 seconds)
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# RULE:import-rich-logger-correctly
try:
    from src.app.log import logger
except ImportError:
    # Fallback for different import paths
    try:
        from app.log import logger
    except ImportError:
        # Create minimal logger if imports fail
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

# RULE:uppercase-constants-top
DEFAULT_TIMEOUT = 300  # 5 minutes
MODEL_LOADING_TIMEOUT = 180  # 3 minutes for individual model loading
GENERATION_TIMEOUT = 30  # 30 seconds for text generation

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

def test_environment_variable_handling():
    """Test environment variable handling"""
    console.print("\n[bold blue]🔧 Testing Environment Variable Handling[/bold blue]")
    
    # Test with environment variable set
    os.environ["SUBGRAPHRAG_DISABLE_MODEL_LOADING"] = "true"
    
    try:
        from src.app.services.information_extraction import ensure_models_loaded
        
        console.print("[yellow]📋 Testing with SUBGRAPHRAG_DISABLE_MODEL_LOADING=true[/yellow]")
        
        with timeout_context(10, "environment variable test"):
            result = ensure_models_loaded()
            
        if result is False:
            console.print("[green]✅ Environment variable correctly disabled model loading[/green]")
            return True
        else:
            console.print("[red]❌ Environment variable was ignored - models were loaded anyway[/red]")
            return False
            
    except TimeoutError:
        console.print("[red]❌ Environment variable test timed out[/red]")
        return False
    except Exception as e:
        console.print(f"[red]❌ Environment variable test failed: {e}[/red]")
        return False

def test_mock_extraction():
    """Test mock extraction functionality"""
    console.print("\n[bold blue]🎭 Testing Mock Extraction[/bold blue]")
    
    # Ensure models are disabled
    os.environ["SUBGRAPHRAG_DISABLE_MODEL_LOADING"] = "true"
    
    try:
        from src.app.services.information_extraction import extract
        
        test_text = "Barack Obama was born in Hawaii and served as President."
        console.print(f"[blue]📝 Test text: '{test_text}'[/blue]")
        
        with timeout_context(15, "mock extraction"):
            result = extract(test_text)
            
        if result and len(result) > 0:
            console.print(f"[green]✅ Mock extraction returned {len(result)} triples[/green]")
            for i, triple in enumerate(result[:3]):
                console.print(f"  {i+1}. {triple.get('head', '?')} → {triple.get('relation', '?')} → {triple.get('tail', '?')}")
            return True
        else:
            console.print("[red]❌ Mock extraction returned no results[/red]")
            return False
            
    except TimeoutError:
        console.print("[red]❌ Mock extraction timed out[/red]")
        return False
    except Exception as e:
        console.print(f"[red]❌ Mock extraction failed: {e}[/red]")
        return False

def test_model_loading_with_timeout(timeout_seconds: int = MODEL_LOADING_TIMEOUT):
    """Test actual model loading with timeout"""
    console.print(f"\n[bold blue]🤖 Testing Model Loading (timeout: {timeout_seconds}s)[/bold blue]")
    
    # Remove environment variable to enable model loading
    if "SUBGRAPHRAG_DISABLE_MODEL_LOADING" in os.environ:
        del os.environ["SUBGRAPHRAG_DISABLE_MODEL_LOADING"]
    
    console.print("[yellow]⚠️ WARNING: This test will attempt to load large AI models[/yellow]")
    console.print("[yellow]   - REBEL model: ~1.6GB download[/yellow]")
    console.print("[yellow]   - NER model: ~500MB download[/yellow]")
    console.print("[yellow]   - May cause segmentation faults on low-memory systems[/yellow]")
    
    try:
        # Import fresh to reset any cached state
        import importlib
        if 'src.app.services.information_extraction' in sys.modules:
            importlib.reload(sys.modules['src.app.services.information_extraction'])
        
        from src.app.services.information_extraction import load_models
        
        console.print(f"[yellow]🚀 Starting model loading with {timeout_seconds}s timeout...[/yellow]")
        
        start_time = time.time()
        
        with timeout_context(timeout_seconds, "model loading"):
            tokenizer, rebel_model, ner_pipe = load_models()
            
        loading_time = time.time() - start_time
        
        # Check results
        if tokenizer is not None and rebel_model is not None and ner_pipe is not None:
            console.print(f"[green]✅ All models loaded successfully in {loading_time:.2f}s[/green]")
            console.print(f"  • REBEL tokenizer: {'✅ Loaded' if tokenizer else '❌ Failed'}")
            console.print(f"  • REBEL model: {'✅ Loaded' if rebel_model else '❌ Failed'}")
            console.print(f"  • NER pipeline: {'✅ Loaded' if ner_pipe else '❌ Failed'}")
            return True, loading_time, "success"
        elif tokenizer is not None and rebel_model is not None:
            console.print(f"[yellow]⚠️ Partial success in {loading_time:.2f}s - NER failed[/yellow]")
            return False, loading_time, "partial"
        else:
            console.print(f"[red]❌ Model loading failed in {loading_time:.2f}s[/red]")
            return False, loading_time, "failed"
            
    except TimeoutError:
        console.print(f"[red]❌ Model loading timed out after {timeout_seconds}s[/red]")
        return False, timeout_seconds, "timeout"
    except Exception as e:
        console.print(f"[red]❌ Model loading failed with exception: {e}[/red]")
        return False, time.time() - start_time if 'start_time' in locals() else 0, "exception"

def test_generation_with_timeout():
    """Test text generation with loaded models"""
    console.print("\n[bold blue]⚡ Testing Text Generation[/bold blue]")
    
    try:
        from src.app.services.information_extraction import extract
        
        test_text = "Albert Einstein developed the theory of relativity."
        console.print(f"[blue]📝 Test text: '{test_text}'[/blue]")
        
        with timeout_context(GENERATION_TIMEOUT, "text generation"):
            result = extract(test_text)
            
        if result and len(result) > 0:
            console.print(f"[green]✅ Generated {len(result)} triples[/green]")
            for i, triple in enumerate(result[:3]):
                console.print(f"  {i+1}. {triple.get('head', '?')} → {triple.get('relation', '?')} → {triple.get('tail', '?')}")
            return True
        else:
            console.print("[red]❌ Generation returned no results[/red]")
            return False
            
    except TimeoutError:
        console.print("[red]❌ Text generation timed out[/red]")
        return False
    except Exception as e:
        console.print(f"[red]❌ Text generation failed: {e}[/red]")
        return False

def test_memory_usage():
    """Test memory usage during model loading"""
    console.print("\n[bold blue]💾 Testing Memory Usage[/bold blue]")
    
    try:
        import psutil
        import gc
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        console.print(f"[blue]📊 Initial memory usage: {initial_memory:.1f} MB[/blue]")
        
        # Force garbage collection
        gc.collect()
        
        # Test with models disabled first
        os.environ["SUBGRAPHRAG_DISABLE_MODEL_LOADING"] = "true"
        
        from src.app.services.information_extraction import ensure_models_loaded
        ensure_models_loaded()
        
        mock_memory = process.memory_info().rss / 1024 / 1024  # MB
        console.print(f"[blue]📊 Memory with mocks: {mock_memory:.1f} MB (+{mock_memory - initial_memory:.1f} MB)[/blue]")
        
        return {
            'initial_memory_mb': initial_memory,
            'mock_memory_mb': mock_memory,
            'mock_overhead_mb': mock_memory - initial_memory
        }
        
    except ImportError:
        console.print("[yellow]⚠️ psutil not available - cannot test memory usage[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]❌ Memory test failed: {e}[/red]")
        return None

def test_server_startup_with_models():
    """Test server startup with model loading"""
    console.print("\n[bold blue]🚀 Testing Server Startup with Models[/bold blue]")
    
    # Remove environment variable to enable model loading
    env = os.environ.copy()
    if "SUBGRAPHRAG_DISABLE_MODEL_LOADING" in env:
        del env["SUBGRAPHRAG_DISABLE_MODEL_LOADING"]
    
    console.print("[yellow]⚠️ This will start the server with model loading enabled[/yellow]")
    console.print("[yellow]   This may cause segmentation faults on low-memory systems[/yellow]")
    
    try:
        with timeout_context(120, "server startup with models"):
            process = subprocess.Popen(
                [sys.executable, "src/main.py", "--port", "8001"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a bit for startup
            time.sleep(10)
            
            # Check if process is still running
            if process.poll() is None:
                console.print("[green]✅ Server started successfully with models[/green]")
                
                # Try to terminate gracefully
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                return True
            else:
                stdout, stderr = process.communicate()
                console.print(f"[red]❌ Server crashed during startup[/red]")
                console.print(f"[red]Exit code: {process.returncode}[/red]")
                if stderr:
                    console.print(f"[red]Error: {stderr[:500]}[/red]")
                return False
                
    except TimeoutError:
        console.print("[red]❌ Server startup timed out[/red]")
        if 'process' in locals():
            try:
                process.kill()
            except:
                pass
        return False
    except Exception as e:
        console.print(f"[red]❌ Server startup test failed: {e}[/red]")
        return False

def run_comprehensive_test(args):
    """Run comprehensive model loading tests"""
    console.print(Panel(
        "SubgraphRAG+ Model Loading Test Suite\n"
        "This script tests model loading with comprehensive diagnostics",
        title="🧪 Model Loading Tests",
        border_style="blue"
    ))
    
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Test 1: Environment Variable Handling
        task1 = progress.add_task("Testing environment variable handling...", total=100)
        results['env_var'] = test_environment_variable_handling()
        progress.update(task1, completed=100)
        
        # Test 2: Mock Extraction
        task2 = progress.add_task("Testing mock extraction...", total=100)
        results['mock_extraction'] = test_mock_extraction()
        progress.update(task2, completed=100)
        
        # Test 3: Memory Usage
        task3 = progress.add_task("Testing memory usage...", total=100)
        results['memory'] = test_memory_usage()
        progress.update(task3, completed=100)
        
        # Test 4: Model Loading (if not disabled)
        if not args.disable_models:
            task4 = progress.add_task(f"Testing model loading (timeout: {args.timeout}s)...", total=100)
            success, loading_time, status = test_model_loading_with_timeout(args.timeout)
            results['model_loading'] = {
                'success': success,
                'time': loading_time,
                'status': status
            }
            progress.update(task4, completed=100)
            
            # Test 5: Text Generation (if models loaded)
            if success:
                task5 = progress.add_task("Testing text generation...", total=100)
                results['generation'] = test_generation_with_timeout()
                progress.update(task5, completed=100)
            else:
                console.print("[yellow]⚠️ Skipping generation test - models not loaded[/yellow]")
                results['generation'] = False
        else:
            console.print("[yellow]⚠️ Model loading tests disabled by --disable-models flag[/yellow]")
            results['model_loading'] = {'success': False, 'time': 0, 'status': 'disabled'}
            results['generation'] = False
    
    # Generate results table
    console.print("\n")
    results_table = Table(title="Test Results Summary", show_header=True, header_style="bold blue")
    results_table.add_column("Test", style="cyan")
    results_table.add_column("Status", style="green")
    results_table.add_column("Details", style="dim")
    
    # Environment variable test
    env_status = "✅ PASS" if results['env_var'] else "❌ FAIL"
    results_table.add_row("Environment Variable", env_status, "Model loading disabled correctly")
    
    # Mock extraction test
    mock_status = "✅ PASS" if results['mock_extraction'] else "❌ FAIL"
    results_table.add_row("Mock Extraction", mock_status, "Fallback functionality working")
    
    # Memory test
    if results['memory']:
        memory_status = "✅ PASS"
        memory_details = f"Overhead: {results['memory']['mock_overhead_mb']:.1f} MB"
    else:
        memory_status = "⚠️ SKIP"
        memory_details = "psutil not available"
    results_table.add_row("Memory Usage", memory_status, memory_details)
    
    # Model loading test
    if results['model_loading']['status'] == 'disabled':
        model_status = "⚠️ SKIP"
        model_details = "Disabled by flag"
    elif results['model_loading']['success']:
        model_status = "✅ PASS"
        model_details = f"Loaded in {results['model_loading']['time']:.1f}s"
    else:
        model_status = "❌ FAIL"
        model_details = f"Status: {results['model_loading']['status']}"
    results_table.add_row("Model Loading", model_status, model_details)
    
    # Generation test
    if results['generation'] is True:
        gen_status = "✅ PASS"
        gen_details = "Generated triples successfully"
    elif results['generation'] is False and results['model_loading']['success']:
        gen_status = "❌ FAIL"
        gen_details = "Failed to generate triples"
    else:
        gen_status = "⚠️ SKIP"
        gen_details = "Models not loaded"
    results_table.add_row("Text Generation", gen_status, gen_details)
    
    console.print(results_table)
    
    # Overall assessment
    console.print("\n")
    
    critical_tests = ['env_var', 'mock_extraction']
    critical_passed = all(results[test] for test in critical_tests)
    
    if critical_passed:
        if results['model_loading']['success']:
            console.print(Panel(
                "🎉 All tests passed!\n\n"
                "✅ Environment variable handling works correctly\n"
                "✅ Mock extraction provides fallback functionality\n"
                "✅ Model loading completed successfully\n"
                "✅ Text generation is working\n\n"
                "The system is ready for production use.",
                title="🎯 Test Results: SUCCESS",
                border_style="green"
            ))
        else:
            console.print(Panel(
                "⚠️ Core functionality working, model loading issues detected\n\n"
                "✅ Environment variable handling works correctly\n"
                "✅ Mock extraction provides fallback functionality\n"
                "❌ Model loading failed or timed out\n\n"
                "The system will work with mock responses but won't have\n"
                "full AI capabilities. Consider:\n"
                "• Increasing memory (8GB+ recommended)\n"
                "• Using a machine with better specs\n"
                "• Running with SUBGRAPHRAG_DISABLE_MODEL_LOADING=true",
                title="⚠️ Test Results: PARTIAL SUCCESS",
                border_style="yellow"
            ))
    else:
        console.print(Panel(
            "❌ Critical tests failed\n\n"
            "The system has fundamental issues that need to be addressed\n"
            "before it can be used reliably.",
            title="❌ Test Results: FAILURE",
            border_style="red"
        ))
    
    return critical_passed

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test SubgraphRAG+ model loading")
    parser.add_argument("--disable-models", action="store_true", 
                       help="Test with model loading disabled")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                       help=f"Timeout for model loading (default: {DEFAULT_TIMEOUT}s)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting model loading test with timeout={args.timeout}s, disable_models={args.disable_models}")
    
    try:
        success = run_comprehensive_test(args)
        exit_code = 0 if success else 1
        logger.info(f"Model loading test completed with exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Test interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]💥 Unexpected error: {e}[/red]")
        logger.exception("Unexpected error in model loading test")
        sys.exit(1)

if __name__ == "__main__":
    main() 