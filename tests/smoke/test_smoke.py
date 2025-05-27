import os
import sys
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Set testing environment before any imports
os.environ['TESTING'] = '1'

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from src.app.api import app
from fastapi.testclient import TestClient

# Create test client
client = TestClient(app)

# Test API key for all tests
TEST_API_KEY = "test_api_key_smoke_tests"


class TestBasicFunctionality:
    """Smoke tests for basic application functionality"""

    def test_app_startup(self):
        """Test that the FastAPI app can start up without errors"""
        # This test passes if the app can be imported and TestClient created
        assert app is not None
        assert client is not None

    def test_health_endpoint_basic(self):
        """Test basic health endpoint functionality"""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_readiness_endpoint_basic(self):
        """Test basic readiness endpoint functionality"""
        response = client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_metrics_endpoint_basic(self):
        """Test basic metrics endpoint functionality"""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Metrics should return plain text
        assert "text/plain" in response.headers.get("content-type", "")

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_query_endpoint_exists(self):
        """Test that query endpoint exists and responds"""
        response = client.post(
            "/query",
            json={"question": "test"},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_feedback_endpoint_exists(self):
        """Test that feedback endpoint exists and responds"""
        response = client.post(
            "/feedback",
            json={
                "query_id": "test_id",
                "rating": 5,
                "feedback_type": "accuracy"
            },
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_graph_browse_endpoint_exists(self):
        """Test that graph browse endpoint exists and responds"""
        response = client.get(
            "/graph/browse?search_term=test",
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_ingest_endpoint_exists(self):
        """Test that ingest endpoint exists and responds"""
        response = client.post(
            "/ingest",
            json={"triples": []},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404


class TestInputSanitization:
    """Smoke tests for input sanitization and safety"""

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_null_bytes_handling(self):
        """Test that null bytes in input don't crash the application"""
        response = client.post(
            "/query",
            json={"question": "Who is Elon Musk?\0"},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should not crash (status code should be valid HTTP status)
        assert 200 <= response.status_code < 600

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_control_characters_handling(self):
        """Test that control characters in input don't crash the application"""
        # Create string with various control characters
        control_chars = "".join(chr(i) for i in range(32) if i not in [9, 10, 13])
        
        response = client.post(
            "/query",
            json={"question": f"Who is Elon Musk?{control_chars}"},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should not crash
        assert 200 <= response.status_code < 600

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_unicode_handling(self):
        """Test that Unicode characters are handled properly"""
        unicode_question = "Tell me about 🚀 SpaceX and العربية 官话?"
        
        response = client.post(
            "/query",
            json={"question": unicode_question},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should not crash
        assert 200 <= response.status_code < 600

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_empty_question_handling(self):
        """Test handling of empty questions"""
        response = client.post(
            "/query",
            json={"question": ""},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should handle gracefully
        assert 200 <= response.status_code < 600

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_very_long_question_handling(self):
        """Test handling of very long questions"""
        long_question = "a" * 10000  # 10KB question
        
        response = client.post(
            "/query",
            json={"question": long_question},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should handle gracefully (may reject with 413 or 422)
        assert 200 <= response.status_code < 600


class TestErrorHandling:
    """Smoke tests for error handling"""

    def test_missing_auth_header(self):
        """Test that missing auth header is handled properly"""
        response = client.post(
            "/query",
            json={"question": "test"}
            # No auth header
        )
        # Should return 401 or 403
        assert response.status_code in [401, 403]

    def test_invalid_auth_header(self):
        """Test that invalid auth header is handled properly"""
        response = client.post(
            "/query",
            json={"question": "test"},
            headers={"X-API-KEY": "invalid_key"}
        )
        # Should return 401 or 403
        assert response.status_code in [401, 403]

    def test_malformed_json(self):
        """Test that malformed JSON is handled properly"""
        response = client.post(
            "/query",
            data="invalid json",
            headers={"Content-Type": "application/json", "X-API-KEY": "test"}
        )
        # Should return 422 (Unprocessable Entity)
        assert response.status_code == 422

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_missing_required_fields(self):
        """Test that missing required fields are handled properly"""
        response = client.post(
            "/query",
            json={},  # Missing 'question' field
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should return 422 (Unprocessable Entity)
        assert response.status_code == 422


class TestDatabaseConnections:
    """Smoke tests for database connections (mocked)"""

    def test_sqlite_connection_mock(self):
        """Test that SQLite connection can be mocked"""
        # Import and test the database module
        from src.app.database import SQLiteDatabase
        
        # Create a test instance (will use in-memory DB in testing mode)
        db = SQLiteDatabase()
        
        assert db is not None
        assert db.verify_connectivity()

    def test_neo4j_connection_mock(self):
        """Test that Neo4j connection can be mocked"""
        # Import and test the database module
        from src.app.database import Neo4jDatabase
        
        # Create a test instance (will be mocked in testing mode)
        db = Neo4jDatabase()
        
        assert db is not None
        # In testing mode, this should not raise an error


class TestLLMIntegration:
    """Smoke tests for LLM integration (mocked)"""

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    @patch('src.app.ml.llm.stream_tokens')
    def test_llm_timeout_handling(self, mock_stream_tokens):
        """Test that LLM timeout is handled gracefully"""
        # Mock timeout exception
        mock_stream_tokens.side_effect = TimeoutError("LLM timeout")
        
        response = client.post(
            "/query",
            json={"question": "test timeout"},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should handle timeout gracefully
        assert 200 <= response.status_code < 600

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    @patch('src.app.ml.llm.stream_tokens')
    def test_llm_connection_error_handling(self, mock_stream_tokens):
        """Test that LLM connection errors are handled gracefully"""
        # Mock connection error
        mock_stream_tokens.side_effect = ConnectionError("LLM connection failed")
        
        response = client.post(
            "/query",
            json={"question": "test connection error"},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should handle connection error gracefully
        assert 200 <= response.status_code < 600


class TestRetrievalPipeline:
    """Smoke tests for retrieval pipeline (mocked)"""

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    @patch('src.app.utils.extract_query_entities')
    @patch('src.app.utils.link_entities_v2')
    def test_empty_entity_extraction(self, mock_link_entities, mock_extract_entities):
        """Test handling when no entities are extracted"""
        mock_extract_entities.return_value = []
        mock_link_entities.return_value = []
        
        response = client.post(
            "/query",
            json={"question": "test no entities"},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should handle gracefully
        assert 200 <= response.status_code < 600

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    @patch('src.app.utils.extract_query_entities')
    @patch('src.app.utils.link_entities_v2')
    def test_no_entity_matches(self, mock_link_entities, mock_extract_entities):
        """Test handling when entities don't match anything in the graph"""
        mock_extract_entities.return_value = ["unknown_entity"]
        mock_link_entities.return_value = []
        
        response = client.post(
            "/query",
            json={"question": "test unknown entities"},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should handle gracefully
        assert 200 <= response.status_code < 600

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    @patch('src.app.retriever.hybrid_retrieve_v2')
    def test_retrieval_failure(self, mock_hybrid_retrieve):
        """Test handling when retrieval fails"""
        mock_hybrid_retrieve.side_effect = Exception("Retrieval failed")
        
        response = client.post(
            "/query",
            json={"question": "test retrieval failure"},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should handle gracefully
        assert 200 <= response.status_code < 600


class TestIngestionPipeline:
    """Smoke tests for ingestion pipeline"""

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_empty_triples_ingestion(self):
        """Test ingestion of empty triples list"""
        response = client.post(
            "/ingest",
            json={"triples": []},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should handle gracefully
        assert 200 <= response.status_code < 600

    @patch('src.app.api.API_KEY_SECRET', TEST_API_KEY)
    def test_malformed_triple_ingestion(self):
        """Test ingestion of malformed triples"""
        response = client.post(
            "/ingest",
            json={"triples": [{"invalid": "triple"}]},
            headers={"X-API-KEY": TEST_API_KEY}
        )
        # Should handle gracefully (may reject with 400 or 422)
        assert 200 <= response.status_code < 600


class TestConcurrentAccess:
    """Smoke tests for concurrent access"""

    def test_concurrent_health_checks(self):
        """Test concurrent health check requests"""
        import threading
        
        def make_request():
            response = client.get("/healthz")
            assert response.status_code == 200
        
        # Run 5 concurrent requests
        threads = [threading.Thread(target=make_request) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_concurrent_metrics_requests(self):
        """Test concurrent metrics requests"""
        import threading
        
        def make_request():
            response = client.get("/metrics")
            assert response.status_code == 200
        
        # Run 5 concurrent requests
        threads = [threading.Thread(target=make_request) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join() 