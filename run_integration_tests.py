#!/usr/bin/env python3
"""
Integration test runner for SubgraphRAG+
Runs tests with Docker services (Neo4j, mock LLM) running
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_test_environment():
    """Set up environment variables for integration testing"""
    # Load test environment
    env_vars = {
        'NEO4J_URI': 'neo4j://localhost:7688',
        'NEO4J_USER': 'neo4j', 
        'NEO4J_PASSWORD': 'testpassword',
        'API_KEY_SECRET': 'test_api_key_for_integration_tests',
        'MODEL_BACKEND': 'openai',
        'OPENAI_API_KEY': 'test_key',
        'OPENAI_BASE_URL': 'http://localhost:8001/v1',
        'EMBEDDING_MODEL': 'Alibaba-NLP/gte-large-en-v1.5',
        'LOG_LEVEL': 'INFO',
        'DEBUG': 'true'
    }
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")

def run_tests(test_files=None):
    """Run the integration tests"""
    if test_files is None:
        test_files = [
            'tests/test_api.py',
            'tests/test_adversarial.py', 
            'tests/test_smoke.py'
        ]
    
    print("Running integration tests with Docker services...")
    print("=" * 60)
    
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"Warning: {test_file} not found, skipping...")
            continue
            
        print(f"\nRunning {test_file}...")
        print("-" * 40)
        
        try:
            # Run pytest with verbose output
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                test_file, 
                '-v', 
                '--tb=short',
                '--no-header'
            ], capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"✅ {test_file} PASSED")
            else:
                print(f"❌ {test_file} FAILED (exit code: {result.returncode})")
                
        except KeyboardInterrupt:
            print(f"\n⚠️  {test_file} INTERRUPTED")
            break
        except Exception as e:
            print(f"❌ {test_file} ERROR: {e}")

if __name__ == "__main__":
    print("SubgraphRAG+ Integration Test Runner")
    print("=" * 60)
    
    # Setup test environment
    setup_test_environment()
    
    # Check if Docker services are running
    try:
        import requests
        
        # Test Neo4j
        neo4j_response = requests.get('http://localhost:7475', timeout=5)
        print(f"✅ Neo4j service: {neo4j_response.status_code}")
        
        # Test mock LLM
        llm_response = requests.post(
            'http://localhost:8001/v1/chat/completions',
            json={"messages": [{"role": "user", "content": "test"}]},
            timeout=5
        )
        print(f"✅ Mock LLM service: {llm_response.status_code}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not verify services: {e}")
        print("Make sure to run: docker-compose -f docker-compose.test.yml up -d")
    
    # Run tests
    test_files = sys.argv[1:] if len(sys.argv) > 1 else None
    run_tests(test_files) 