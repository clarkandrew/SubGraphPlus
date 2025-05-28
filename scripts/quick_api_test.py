#!/usr/bin/env python3
"""
Quick API diagnostic test
"""

import os
import sys
import time
import subprocess
import requests
import json
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

SERVER_PORT = 8003
BASE_URL = f"http://localhost:{SERVER_PORT}"
API_KEY = os.environ.get("API_KEY_SECRET", "default_key_for_dev_only")

def start_server():
    """Start server in background"""
    env = os.environ.copy()
    env["SUBGRAPHRAG_DISABLE_MODEL_LOADING"] = "true"
    
    process = subprocess.Popen(
        [sys.executable, "src/main.py", "--port", str(SERVER_PORT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    for _ in range(30):
        try:
            response = requests.get(f"{BASE_URL}/healthz", timeout=2)
            if response.status_code == 200:
                print(f"✅ Server started on port {SERVER_PORT}")
                return process
        except:
            pass
        time.sleep(1)
    
    print("❌ Server failed to start")
    return None

def test_endpoint(method, endpoint, data=None):
    """Test an endpoint and show detailed response"""
    headers = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}
    
    print(f"\n🧪 Testing {method} {endpoint}")
    
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}", headers=headers, timeout=10)
        else:
            response = requests.post(f"{BASE_URL}{endpoint}", json=data, headers=headers, timeout=10)
        
        print(f"Status: {response.status_code}")
        
        try:
            response_data = response.json()
            print(f"Response: {json.dumps(response_data, indent=2)}")
        except:
            print(f"Response: {response.text}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting quick API diagnostic test")
    
    # Start server
    process = start_server()
    if not process:
        return False
    
    try:
        # Test health endpoints
        test_endpoint("GET", "/healthz")
        test_endpoint("GET", "/ie/health")
        test_endpoint("GET", "/ie/info")
        
        # Test IE extraction
        test_endpoint("POST", "/ie/extract", {"text": "Barack Obama was born in Hawaii."})
        
        # Test text ingestion
        test_endpoint("POST", "/ingest/text", {
            "text": "Barack Obama was born in Hawaii.",
            "source": "test"
        })
        
        # Test query
        test_endpoint("POST", "/query", {"question": "Obama"})
        
    finally:
        # Stop server
        print("\n🛑 Stopping server...")
        process.terminate()
        process.wait(timeout=10)
        print("✅ Server stopped")

if __name__ == "__main__":
    main() 