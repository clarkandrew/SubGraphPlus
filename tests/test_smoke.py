import sys
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from src.app.api import app
from fastapi.testclient import TestClient

# Create test client
client = TestClient(app)


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
        assert data["status"] == "healthy"

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

    @pytest.fixture
    def mock_auth_header(self):
        """Mock authorization header for testing"""
        with patch('src.app.api.API_KEY_SECRET', "test_api_key"):
            return {"X-API-KEY": "test_api_key"}

    def test_query_endpoint_exists(self, mock_auth_header):
        """Test that query endpoint exists and responds"""
        response = client.post(
            "/query",
            json={"question": "test"},
            headers=mock_auth_header
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404

    def test_feedback_endpoint_exists(self, mock_auth_header):
        """Test that feedback endpoint exists and responds"""
        response = client.post(
            "/feedback",
            json={
                "query_id": "test_id",
                "rating": 5,
                "feedback_type": "accuracy"
            },
            headers=mock_auth_header
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404

    def test_graph_browse_endpoint_exists(self, mock_auth_header):
        """Test that graph browse endpoint exists and responds"""
        response = client.get(
            "/graph/browse?search_term=test",
            headers=mock_auth_header
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404

    def test_ingest_endpoint_exists(self, mock_auth_header):
        """Test that ingest endpoint exists and responds"""
        response = client.post(
            "/ingest",
            json={"triples": []},
            headers=mock_auth_header
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404


class TestInputSanitization:
    """Smoke tests for input sanitization and safety"""

    @pytest.fixture
    def mock_auth_header(self):
        """Mock authorization header for testing"""
        with patch('src.app.api.API_KEY_SECRET', "test_api_key"):
            return {"X-API-KEY": "test_api_key"}

    def test_null_bytes_handling(self, mock_auth_header):
        """Test that null bytes in input don't crash the application"""
        response = client.post(
            "/query",
            json={"question": "Who is Elon Musk?\0"},
            headers=mock_auth_header
        )
        # Should not crash (status code should be valid HTTP status)
        assert 200 <= response.status_code < 600

    def test_control_characters_handling(self, mock_auth_header):
        """Test that control characters in input don't crash the application"""
        # Create string with various control characters
        control_chars = "".join(chr(i) for i in range(32) if i not in [9, 10, 13])
        
        response = client.post(
            "/query",
            json={"question": f"Who is Elon Musk?{control_chars}"},
            headers=mock_auth_header
        )
        # Should not crash
        assert 200 <= response.status_code < 600

    def test_unicode_handling(self, mock_auth_header):
        """Test that Unicode characters are handled properly"""
        unicode_question = "Tell me about 🚀 SpaceX and العربية 官话?"
        
        response = client.post(
            "/query",
            json={"question": unicode_question},
            headers=mock_auth_header
        )
        # Should not crash
        assert 200 <= response.status_code < 600

    def test_empty_question_handling(self, mock_auth_header):
        """Test handling of empty questions"""
        response = client.post(
            "/query",
            json={"question": ""},
            headers=mock_auth_header
        )
        # Should handle gracefully
        assert 200 <= response.status_code < 600

    def test_very_long_question_handling(self, mock_auth_header):
        """Test handling of very long questions"""
        long_question = "a" * 10000  # 10KB question
        
        response = client.post(
            "/query",
            json={"question": long_question},
            headers=mock_auth_header
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

    def test_missing_required_fields(self):
        """Test that missing required fields are handled properly"""
        with patch('src.app.api.API_KEY_SECRET', "test_api_key"):
            response = client.post(
                "/query",
                json={},  # Missing 'question' field
                headers={"X-API-KEY": "test_api_key"}
            )
            # Should return 422 (Unprocessable Entity)
            assert response.status_code == 422


class TestDatabaseConnections:
    """Smoke tests for database connectivity"""

    @patch('src.app.database.sqlite_db.execute')
    def test_sqlite_connection_mock(self, mock_execute):
        """Test that SQLite operations can be mocked (indicates proper structure)"""
        mock_execute.return_value = None
        
        with patch('src.app.api.API_KEY_SECRET', "test_api_key"):
            response = client.post(
                "/feedback",
                json={
                    "query_id": "test_id",
                    "rating": 5,
                    "feedback_type": "accuracy",
                    "comments": "Test feedback"
                },
                headers={"X-API-KEY": "test_api_key"}
            )
        
        # Should not crash due to database issues
        assert 200 <= response.status_code < 600

    @patch('src.app.database.neo4j_db.run_query')
    def test_neo4j_connection_mock(self, mock_neo4j):
        """Test that Neo4j operations can be mocked (indicates proper structure)"""
        mock_neo4j.return_value = []
        
        with patch('src.app.api.API_KEY_SECRET', "test_api_key"):
            response = client.get(
                "/graph/browse?search_term=test",
                headers={"X-API-KEY": "test_api_key"}
            )
        
        # Should not crash due to database issues
        assert 200 <= response.status_code < 600


class TestLLMIntegration:
    """Smoke tests for LLM integration"""

    @pytest.fixture
    def mock_auth_header(self):
        """Mock authorization header for testing"""
        with patch('src.app.api.API_KEY_SECRET', "test_api_key"):
            return {"X-API-KEY": "test_api_key"}

    @patch('src.app.ml.llm.stream_tokens')
    def test_llm_timeout_handling(self, mock_stream_tokens, mock_auth_header):
        """Test that LLM timeouts are handled gracefully"""
        import requests
        
        # Mock LLM to raise timeout
        mock_stream_tokens.side_effect = requests.exceptions.Timeout("LLM timeout")
        
        # Mock other dependencies to reach LLM
        with patch('src.app.retriever.extract_query_entities') as mock_extract, \
             patch('src.app.retriever.link_entities_v2') as mock_link, \
             patch('src.app.retriever.hybrid_retrieve_v2') as mock_retrieve:
            
            mock_extract.return_value = ["Tesla"]
            mock_link.return_value = [("tesla", 0.9)]
            mock_retrieve.return_value = [MagicMock()]
            
            response = client.post(
                "/query",
                json={"question": "Tell me about Tesla"},
                headers=mock_auth_header
            )
            
            # Should handle timeout gracefully
            assert response.status_code == 200

    @patch('src.app.ml.llm.stream_tokens')
    def test_llm_connection_error_handling(self, mock_stream_tokens, mock_auth_header):
        """Test that LLM connection errors are handled gracefully"""
        import requests
        
        # Mock LLM to raise connection error
        mock_stream_tokens.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        # Mock other dependencies to reach LLM
        with patch('src.app.retriever.extract_query_entities') as mock_extract, \
             patch('src.app.retriever.link_entities_v2') as mock_link, \
             patch('src.app.retriever.hybrid_retrieve_v2') as mock_retrieve:
            
            mock_extract.return_value = ["Tesla"]
            mock_link.return_value = [("tesla", 0.9)]
            mock_retrieve.return_value = [MagicMock()]
            
            response = client.post(
                "/query",
                json={"question": "Tell me about Tesla"},
                headers=mock_auth_header
            )
            
            # Should handle connection error gracefully
            assert response.status_code == 200


class TestRetrievalPipeline:
    """Smoke tests for the retrieval pipeline"""

    @pytest.fixture
    def mock_auth_header(self):
        """Mock authorization header for testing"""
        with patch('src.app.api.API_KEY_SECRET', "test_api_key"):
            return {"X-API-KEY": "test_api_key"}

    @patch('src.app.retriever.extract_query_entities')
    @patch('src.app.retriever.link_entities_v2')
    def test_empty_entity_extraction(self, mock_link_entities, 
                                   mock_extract_entities, mock_auth_header):
        """Test handling when no entities are extracted"""
        mock_extract_entities.return_value = []
        mock_link_entities.return_value = []
        
        response = client.post(
            "/query",
            json={"question": "What is the meaning of life?"},
            headers=mock_auth_header
        )
        
        # Should handle gracefully
        assert response.status_code == 200

    @patch('src.app.retriever.extract_query_entities')
    @patch('src.app.retriever.link_entities_v2')
    def test_no_entity_matches(self, mock_link_entities, 
                             mock_extract_entities, mock_auth_header):
        """Test handling when entities are extracted but not found in KB"""
        mock_extract_entities.return_value = ["NonexistentEntity"]
        mock_link_entities.return_value = []  # No matches found
        
        response = client.post(
            "/query",
            json={"question": "Tell me about NonexistentEntity"},
            headers=mock_auth_header
        )
        
        # Should handle gracefully
        assert response.status_code == 200

    @patch('src.app.retriever.hybrid_retrieve_v2')
    def test_retrieval_failure(self, mock_hybrid_retrieve, mock_auth_header):
        """Test handling when retrieval fails"""
        from src.app.models import RetrievalEmpty
        
        # Mock retrieval to raise RetrievalEmpty
        mock_hybrid_retrieve.side_effect = RetrievalEmpty("No triples found")
        
        # Mock other dependencies
        with patch('src.app.retriever.extract_query_entities') as mock_extract, \
             patch('src.app.retriever.link_entities_v2') as mock_link:
            
            mock_extract.return_value = ["Tesla"]
            mock_link.return_value = [("tesla", 0.9)]
            
            response = client.post(
                "/query",
                json={"question": "Tell me about Tesla"},
                headers=mock_auth_header
            )
            
            # Should handle retrieval failure gracefully
            assert response.status_code == 200


class TestIngestionPipeline:
    """Smoke tests for the ingestion pipeline"""

    @pytest.fixture
    def mock_auth_header(self):
        """Mock authorization header for testing"""
        with patch('src.app.api.API_KEY_SECRET', "test_api_key"):
            return {"X-API-KEY": "test_api_key"}

    def test_empty_triples_ingestion(self, mock_auth_header):
        """Test ingesting empty list of triples"""
        response = client.post(
            "/ingest",
            json={"triples": []},
            headers=mock_auth_header
        )
        
        # Should handle empty ingestion gracefully
        assert response.status_code == 200

    def test_malformed_triple_ingestion(self, mock_auth_header):
        """Test ingesting malformed triples"""
        malformed_triples = [
            {
                "head": "Entity1",
                # Missing required fields
            }
        ]
        
        response = client.post(
            "/ingest",
            json={"triples": malformed_triples},
            headers=mock_auth_header
        )
        
        # Should handle malformed data gracefully (may return 422)
        assert response.status_code in [200, 400, 422]


class TestConcurrentAccess:
    """Smoke tests for concurrent access"""

    def test_concurrent_health_checks(self):
        """Test that concurrent health checks don't cause issues"""
        from concurrent.futures import ThreadPoolExecutor
        
        def make_request():
            return client.get("/healthz")
        
        # Make 5 concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            
            # All should succeed
            for future in futures:
                response = future.result()
                assert response.status_code == 200

    def test_concurrent_metrics_requests(self):
        """Test that concurrent metrics requests don't cause issues"""
        from concurrent.futures import ThreadPoolExecutor
        
        def make_request():
            return client.get("/metrics")
        
        # Make 3 concurrent requests
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            
            # All should succeed
            for future in futures:
                response = future.result()
                assert response.status_code == 200 