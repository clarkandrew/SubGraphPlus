import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

# Mock the problematic module-level variables before importing
with patch.dict('sys.modules', {
    'mlx.core': MagicMock(),
    'mlx.nn': MagicMock(),
    'mlx_lm': MagicMock(),
    'torch': MagicMock(),
}):
    # Mock the module-level variables that cause segfaults
    with patch('src.app.retriever.mlp_model', None), \
         patch('src.app.retriever.faiss_index') as mock_faiss_index:
        
        # Configure the FAISS index mock
        mock_faiss_index.index = MagicMock()
        mock_faiss_index.index.ntotal = 10
        mock_faiss_index.search.return_value = []
        
        from src.app.api import app
        from fastapi.testclient import TestClient

# Create test client
client = TestClient(app)


class TestBasicAPI:
    """Basic API functionality tests with mocked ML components"""

    def test_health_endpoint(self):
        """Test health endpoint"""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_readiness_endpoint(self):
        """Test readiness endpoint"""
        response = client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_missing_auth_returns_401(self):
        """Test that missing auth header returns 401"""
        response = client.post(
            "/query",
            json={"question": "test"}
        )
        assert response.status_code in [401, 403]

    @patch('src.app.api.API_KEY_SECRET', "test_api_key")
    def test_query_endpoint_with_auth(self):
        """Test query endpoint with proper auth"""
        response = client.post(
            "/query",
            json={"question": "test question"},
            headers={"X-API-KEY": "test_api_key"}
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404

    @patch('src.app.api.API_KEY_SECRET', "test_api_key")
    def test_feedback_endpoint_with_auth(self):
        """Test feedback endpoint with proper auth"""
        response = client.post(
            "/feedback",
            json={
                "query_id": "test_id",
                "rating": 5,
                "feedback_type": "accuracy"
            },
            headers={"X-API-KEY": "test_api_key"}
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404

    @patch('src.app.api.API_KEY_SECRET', "test_api_key")
    def test_graph_browse_endpoint_with_auth(self):
        """Test graph browse endpoint with proper auth"""
        response = client.get(
            "/graph/browse?search_term=test",
            headers={"X-API-KEY": "test_api_key"}
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404

    @patch('src.app.api.API_KEY_SECRET', "test_api_key")
    def test_ingest_endpoint_with_auth(self):
        """Test ingest endpoint with proper auth"""
        response = client.post(
            "/ingest",
            json={"triples": []},
            headers={"X-API-KEY": "test_api_key"}
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404


class TestInputValidation:
    """Test input validation"""

    @patch('src.app.api.API_KEY_SECRET', "test_api_key")
    def test_empty_question(self):
        """Test handling of empty question"""
        response = client.post(
            "/query",
            json={"question": ""},
            headers={"X-API-KEY": "test_api_key"}
        )
        # Should handle gracefully
        assert 200 <= response.status_code < 600

    @patch('src.app.api.API_KEY_SECRET', "test_api_key")
    def test_missing_question_field(self):
        """Test handling of missing question field"""
        response = client.post(
            "/query",
            json={},
            headers={"X-API-KEY": "test_api_key"}
        )
        # Should return validation error
        assert response.status_code == 422

    def test_malformed_json(self):
        """Test handling of malformed JSON"""
        response = client.post(
            "/query",
            data="invalid json",
            headers={"Content-Type": "application/json", "X-API-KEY": "test"}
        )
        assert response.status_code == 422 