#!/usr/bin/env python3
"""
Test script for the unified SubgraphRAG+ API with integrated IE functionality
Verifies that IE functionality is properly integrated into the main API
"""

import os
import sys
import time
import requests
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# RULE:import-rich-logger-correctly
from src.app.log import logger

# RULE:uppercase-constants-top
API_BASE_URL = "http://localhost:8000"
TEST_TEXT = "Moses led the Israelites out of Egypt."
TIMEOUT_SECONDS = 30

def test_ie_endpoints():
    """Test the integrated IE endpoints"""
    # RULE:every-src-script-must-log
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Started {__file__} at {timestamp}")
    
    # RULE:debug-trace-every-step
    logger.debug("Starting IE endpoints test")
    
    try:
        # Test IE health endpoint
        logger.info("Testing IE health endpoint...")
        response = requests.get(f"{API_BASE_URL}/ie/health", timeout=TIMEOUT_SECONDS)
        
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ IE health check passed: {health_data}")
        else:
            logger.error(f"❌ IE health check failed: {response.status_code}")
            return False
        
        # Test IE info endpoint
        logger.info("Testing IE info endpoint...")
        response = requests.get(f"{API_BASE_URL}/ie/info", timeout=TIMEOUT_SECONDS)
        
        if response.status_code == 200:
            info_data = response.json()
            logger.info(f"✅ IE info endpoint works: {info_data['model_name']}")
        else:
            logger.error(f"❌ IE info endpoint failed: {response.status_code}")
            return False
        
        # Test IE extract endpoint (this requires API key)
        api_key = os.getenv("API_KEY_SECRET")
        if api_key:
            logger.info("Testing IE extract endpoint...")
            
            headers = {"X-API-KEY": api_key}
            payload = {
                "text": TEST_TEXT,
                "max_length": 256,
                "num_beams": 3
            }
            
            response = requests.post(
                f"{API_BASE_URL}/ie/extract",
                json=payload,
                headers=headers,
                timeout=TIMEOUT_SECONDS
            )
            
            if response.status_code == 200:
                extract_data = response.json()
                triples_count = len(extract_data['triples'])
                processing_time = extract_data['processing_time']
                logger.info(f"✅ IE extract endpoint works: {triples_count} triples in {processing_time:.2f}s")
                
                # Log extracted triples
                for i, triple in enumerate(extract_data['triples']):
                    logger.info(f"  Triple {i+1}: ({triple['head']}) --[{triple['relation']}]--> ({triple['tail']})")
                
            else:
                logger.error(f"❌ IE extract endpoint failed: {response.status_code} - {response.text}")
                return False
        else:
            logger.warning("No API key found in environment, skipping extract endpoint test")
        
        logger.debug("Finished IE endpoints test")
        return True
        
    except Exception as e:
        # RULE:rich-error-handling-required
        logger.error(f"Error testing IE endpoints: {e}")
        return False

def test_main_api_endpoints():
    """Test that main API endpoints still work"""
    logger.debug("Starting main API endpoints test")
    
    try:
        # Test main health endpoint
        logger.info("Testing main API health endpoint...")
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=TIMEOUT_SECONDS)
        
        if response.status_code == 200:
            logger.info("✅ Main API health check passed")
        else:
            logger.error(f"❌ Main API health check failed: {response.status_code}")
            return False
        
        # Test readiness endpoint
        logger.info("Testing readiness endpoint...")
        response = requests.get(f"{API_BASE_URL}/readyz", timeout=TIMEOUT_SECONDS)
        
        if response.status_code in [200, 503]:  # 503 is ok if some dependencies aren't ready
            readiness_data = response.json()
            logger.info(f"✅ Readiness endpoint works: {readiness_data['status']}")
            
            # Check if REBEL model is included in readiness checks
            if 'checks' in readiness_data and 'rebel_model' in readiness_data['checks']:
                rebel_status = readiness_data['checks']['rebel_model']
                logger.info(f"✅ REBEL model check included: {rebel_status}")
            else:
                logger.warning("REBEL model not found in readiness checks")
                
        else:
            logger.error(f"❌ Readiness endpoint failed: {response.status_code}")
            return False
        
        logger.debug("Finished main API endpoints test")
        return True
        
    except Exception as e:
        # RULE:rich-error-handling-required
        logger.error(f"Error testing main API endpoints: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting unified API integration test")
    
    # Test if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=5)
        if response.status_code != 200:
            logger.error("❌ API is not running or not healthy")
            logger.error("Start the API with: uvicorn src.app.api:app --host 0.0.0.0 --port 8000")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Cannot connect to API at {API_BASE_URL}: {e}")
        logger.error("Start the API with: uvicorn src.app.api:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # Run tests
    ie_test_passed = test_ie_endpoints()
    main_api_test_passed = test_main_api_endpoints()
    
    # Summary
    if ie_test_passed and main_api_test_passed:
        logger.info("🎉 All tests passed! Unified API with IE integration is working correctly.")
        logger.info("✅ IE functionality successfully integrated into main API")
        logger.info("✅ Main API functionality preserved")
        return 0
    else:
        logger.error("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 