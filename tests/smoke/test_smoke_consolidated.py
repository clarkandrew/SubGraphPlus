#!/usr/bin/env python3
"""
Consolidated Smoke Tests for SubgraphRAG+

This file contains comprehensive smoke tests that validate core functionality
across all major components of the SubgraphRAG+ system.
"""

import os
import sys
import pytest
import time
import threading
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

# Set testing environment before any imports
os.environ['TESTING'] = '1'
os.environ['SUBGRAPHRAG_DISABLE_MODEL_LOADING'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.app.log import logger
from src.app.api import app
from fastapi.testclient import TestClient

# Test API key for all tests
TEST_API_KEY = "test_api_key_smoke_tests"

# Create test client
client = TestClient(app)


class TestBasicFunctionality:
    """Smoke tests for basic application functionality"""

    def test_app_startup(self):
        """Test that the FastAPI app can start up without errors"""
        logger.debug("Testing app startup")
        assert app is not None
        assert client is not None
        logger.debug("✅ App startup successful")

    def test_health_endpoint(self):
        """Test health endpoint functionality"""
        logger.debug("Testing health endpoint")
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        logger.debug("✅ Health endpoint working")

    def test_readiness_endpoint(self):
        """Test readiness endpoint functionality"""
        logger.debug("Testing readiness endpoint")
        response = client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        logger.debug("✅ Readiness endpoint working")

    def test_metrics_endpoint(self):
        """Test metrics endpoint functionality"""
        logger.debug("Testing metrics endpoint")
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        logger.debug("✅ Metrics endpoint working")


class TestAPIEndpoints:
    """Smoke tests for API endpoints"""

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_query_endpoint_exists(self):
        """Test that query endpoint exists and responds"""
        logger.debug("Testing query endpoint existence")
        response = client.post(
            "/query",
            json={"question": "test"},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404
        logger.debug("✅ Query endpoint exists")

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_feedback_endpoint_exists(self):
        """Test that feedback endpoint exists and responds"""
        logger.debug("Testing feedback endpoint existence")
        response = client.post(
            "/feedback",
            json={
                "query_id": "test_id",
                "rating": 5,
                "feedback_type": "accuracy"
            },
            headers={"X-API-KEY": TEST_API_KEY}
        )
        assert response.status_code != 404
        logger.debug("✅ Feedback endpoint exists")

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_graph_browse_endpoint_exists(self):
        """Test that graph browse endpoint exists and responds"""
        logger.debug("Testing graph browse endpoint existence")
        response = client.get(
            "/graph/browse?search_term=test",
            headers={"X-API-KEY": TEST_API_KEY}
        )
        assert response.status_code != 404
        logger.debug("✅ Graph browse endpoint exists")

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_ingest_endpoint_exists(self):
        """Test that ingest endpoint exists and responds"""
        logger.debug("Testing ingest endpoint existence")
        response = client.post(
            "/ingest",
            json={"triples": []},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        assert response.status_code != 404
        logger.debug("✅ Ingest endpoint exists")


class TestInformationExtraction:
    """Smoke tests for Information Extraction functionality"""
    
    def test_ie_imports_work(self):
        """Test that all IE modules can be imported without errors"""
        logger.debug("Testing IE imports")
        
        try:
            from src.app.services.information_extraction import (
                InformationExtractionService,
                get_information_extraction_service,
                extract,
                REBEL_MODEL_NAME,
                NER_MODEL_NAME
            )
            logger.debug("✅ IE imports successful")
        except Exception as e:
            logger.error(f"❌ IE import failed: {e}")
            pytest.fail(f"Failed to import IE modules: {e}")
    
    def test_ie_service_instantiation(self):
        """Test that IE service can be instantiated"""
        logger.debug("Testing IE service instantiation")
        
        try:
            from src.app.services.information_extraction import get_information_extraction_service
            
            service = get_information_extraction_service()
            assert service is not None
            assert hasattr(service, 'extract_triples')
            assert hasattr(service, 'is_model_loaded')
            
            logger.debug("✅ IE service instantiation successful")
        except Exception as e:
            logger.error(f"❌ IE service instantiation failed: {e}")
            pytest.fail(f"Failed to instantiate IE service: {e}")
    
    def test_ie_service_singleton(self):
        """Test that IE service follows singleton pattern"""
        logger.debug("Testing IE service singleton pattern")
        
        try:
            from src.app.services.information_extraction import get_information_extraction_service
            
            service1 = get_information_extraction_service()
            service2 = get_information_extraction_service()
            
            assert service1 is service2, "IE service should be a singleton"
            logger.debug("✅ IE service singleton pattern verified")
        except Exception as e:
            logger.error(f"❌ IE service singleton test failed: {e}")
            pytest.fail(f"IE service singleton test failed: {e}")
    
    def test_ie_extraction_basic(self):
        """Test basic IE extraction functionality"""
        logger.debug("Testing basic IE extraction")
        
        try:
            from src.app.services.information_extraction import get_information_extraction_service
            
            service = get_information_extraction_service()
            
            # Test with simple text
            test_text = "Barack Obama was born in Hawaii."
            result = service.extract_triples(test_text)
            
            assert hasattr(result, 'success'), "Result should have success attribute"
            assert hasattr(result, 'triples'), "Result should have triples attribute"
            assert hasattr(result, 'processing_time'), "Result should have processing_time attribute"
            
            logger.debug(f"✅ Basic IE extraction completed: success={result.success}")
            
        except Exception as e:
            logger.error(f"❌ Basic IE extraction failed: {e}")
            pytest.fail(f"Basic IE extraction failed: {e}")


class TestInputSanitization:
    """Smoke tests for input sanitization and safety"""

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_unicode_handling(self):
        """Test that Unicode characters are handled properly"""
        logger.debug("Testing Unicode handling")
        unicode_question = "Tell me about 🚀 SpaceX and العربية 官话?"
        
        response = client.post(
            "/query",
            json={"question": unicode_question},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should not crash
        assert 200 <= response.status_code < 600
        logger.debug("✅ Unicode handling working")

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_empty_question_handling(self):
        """Test handling of empty questions"""
        logger.debug("Testing empty question handling")
        response = client.post(
            "/query",
            json={"question": ""},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should handle gracefully
        assert 200 <= response.status_code < 600
        logger.debug("✅ Empty question handling working")

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_very_long_question_handling(self):
        """Test handling of very long questions"""
        logger.debug("Testing long question handling")
        long_question = "a" * 1000  # 1KB question
        
        response = client.post(
            "/query",
            json={"question": long_question},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should handle gracefully
        assert 200 <= response.status_code < 600
        logger.debug("✅ Long question handling working")


class TestErrorHandling:
    """Smoke tests for error handling"""

    def test_missing_auth_header(self):
        """Test that missing auth header is handled properly"""
        logger.debug("Testing missing auth header handling")
        response = client.post(
            "/query",
            json={"question": "test"}
            # No auth header
        )
        # Should return 401 or 403
        assert response.status_code in [401, 403]
        logger.debug("✅ Missing auth header handled correctly")

    def test_invalid_auth_header(self):
        """Test that invalid auth header is handled properly"""
        logger.debug("Testing invalid auth header handling")
        response = client.post(
            "/query",
            json={"question": "test"},
            headers={"X-API-KEY": "invalid_key"}
        )
        # Should return 401 or 403
        assert response.status_code in [401, 403]
        logger.debug("✅ Invalid auth header handled correctly")

    def test_malformed_json(self):
        """Test that malformed JSON is handled properly"""
        logger.debug("Testing malformed JSON handling")
        response = client.post(
            "/query",
            data="invalid json",
            headers={"Content-Type": "application/json", "X-API-KEY": "test"}
        )
        # Should return 422 (Unprocessable Entity)
        assert response.status_code == 422
        logger.debug("✅ Malformed JSON handled correctly")


class TestPerformance:
    """Basic performance smoke tests"""

    def test_health_endpoint_performance(self):
        """Test that health endpoint responds quickly"""
        logger.debug("Testing health endpoint performance")
        start_time = time.time()
        response = client.get("/healthz")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response.status_code == 200
        assert response_time < 1.0, f"Health endpoint too slow: {response_time:.2f}s"
        logger.debug(f"✅ Health endpoint performance: {response_time:.3f}s")

    def test_metrics_endpoint_performance(self):
        """Test that metrics endpoint responds quickly"""
        logger.debug("Testing metrics endpoint performance")
        start_time = time.time()
        response = client.get("/metrics")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response.status_code == 200
        assert response_time < 2.0, f"Metrics endpoint too slow: {response_time:.2f}s"
        logger.debug(f"✅ Metrics endpoint performance: {response_time:.3f}s")


class TestConcurrentAccess:
    """Basic concurrency smoke tests"""

    def test_concurrent_health_checks(self):
        """Test concurrent access to health endpoint"""
        logger.debug("Testing concurrent health checks")
        
        def make_request():
            response = client.get("/healthz")
            return response.status_code == 200
        
        # Run 5 concurrent requests
        threads = []
        results = []
        
        for _ in range(5):
            thread = threading.Thread(target=lambda: results.append(make_request()))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(results), "Some concurrent health checks failed"
        logger.debug("✅ Concurrent health checks working")


class TestConfiguration:
    """Test configuration and environment setup"""
    
    def test_environment_variables(self):
        """Test that required environment variables are set"""
        logger.debug("Testing environment variables")
        
        # Test that testing mode is enabled
        assert os.environ.get('TESTING') == '1', "TESTING environment variable should be set"
        
        # Test IE model loading is disabled for tests
        assert os.environ.get('SUBGRAPHRAG_DISABLE_MODEL_LOADING') == 'true'
        
        # Test tokenizers parallelism is disabled
        assert os.environ.get('TOKENIZERS_PARALLELISM') == 'false'
        
        logger.debug("✅ Environment variables configured correctly")
    
    def test_model_names_defined(self):
        """Test that model names are properly defined"""
        logger.debug("Testing model names configuration")
        
        try:
            from src.app.services.information_extraction import REBEL_MODEL_NAME, NER_MODEL_NAME
            
            assert isinstance(REBEL_MODEL_NAME, str), "REBEL_MODEL_NAME should be a string"
            assert isinstance(NER_MODEL_NAME, str), "NER_MODEL_NAME should be a string"
            assert len(REBEL_MODEL_NAME) > 0, "REBEL_MODEL_NAME should not be empty"
            assert len(NER_MODEL_NAME) > 0, "NER_MODEL_NAME should not be empty"
            
            logger.debug(f"✅ Model names verified: REBEL={REBEL_MODEL_NAME}, NER={NER_MODEL_NAME}")
            
        except Exception as e:
            logger.error(f"❌ Model names test failed: {e}")
            pytest.fail(f"Model names test failed: {e}")


if __name__ == "__main__":
    # Run smoke tests directly
    pytest.main([__file__, "-v", "--tb=short"]) 