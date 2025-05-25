import sys
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variable to disable model loading
os.environ['TESTING'] = '1'

# Mock all the heavy dependencies before any imports
mock_modules = {
    'mlx': MagicMock(),
    'mlx.core': MagicMock(),
    'mlx.nn': MagicMock(),
    'mlx_lm': MagicMock(),
    'torch': MagicMock(),
    'torch.nn': MagicMock(),
    'faiss': MagicMock(),
    'transformers': MagicMock(),
}

# Patch sys.modules to prevent actual imports
for module_name, mock_module in mock_modules.items():
    sys.modules[module_name] = mock_module

# Mock the config to prevent model loading
with patch('src.app.config.config') as mock_config:
    mock_config.MLP_MODEL_PATH = "/nonexistent/path"
    mock_config.FAISS_INDEX_PATH = "/nonexistent/path"
    mock_config.MODEL_BACKEND = "test"
    
    # Mock the retriever module's global variables
    with patch('src.app.retriever.mlp_model', None), \
         patch('src.app.retriever.faiss_index') as mock_faiss:
        
        # Configure FAISS mock
        mock_faiss.index = Mock()
        mock_faiss.index.ntotal = 0
        mock_faiss.search = Mock(return_value=[])
        
        # Now import the app
        from src.app.api import app
        from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_endpoint():
    """Test that health endpoint works"""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_readiness_endpoint():
    """Test that readiness endpoint responds (may be unhealthy due to mocked components)"""
    response = client.get("/readyz")
    # Should respond with either 200 (healthy) or 503 (unhealthy)
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data


def test_metrics_endpoint():
    """Test that metrics endpoint works"""
    response = client.get("/metrics")
    assert response.status_code == 200
    # Should return prometheus metrics format
    assert "text/plain" in response.headers.get("content-type", "")


def test_missing_auth():
    """Test that missing auth is handled"""
    response = client.post("/query", json={"question": "test"})
    assert response.status_code in [401, 403]


@patch('src.app.api.API_KEY_SECRET', "test_key")
def test_query_endpoint_exists():
    """Test that query endpoint exists"""
    response = client.post(
        "/query",
        json={"question": "test"},
        headers={"X-API-KEY": "test_key"}
    )
    # Should not be 404 (endpoint exists)
    assert response.status_code != 404


@patch('src.app.api.API_KEY_SECRET', "test_key")
def test_feedback_endpoint_exists():
    """Test that feedback endpoint exists"""
    response = client.post(
        "/feedback",
        json={
            "query_id": "test",
            "rating": 5,
            "feedback_type": "accuracy"
        },
        headers={"X-API-KEY": "test_key"}
    )
    # Should not be 404 (endpoint exists)
    assert response.status_code != 404


@patch('src.app.api.API_KEY_SECRET', "test_key")
def test_graph_browse_endpoint_exists():
    """Test that graph browse endpoint exists"""
    response = client.get(
        "/graph/browse?search_term=test",
        headers={"X-API-KEY": "test_key"}
    )
    # Should not be 404 (endpoint exists)
    assert response.status_code != 404


@patch('src.app.api.API_KEY_SECRET', "test_key")
def test_ingest_endpoint_exists():
    """Test that ingest endpoint exists"""
    response = client.post(
        "/ingest",
        json={"triples": []},
        headers={"X-API-KEY": "test_key"}
    )
    # Should not be 404 (endpoint exists)
    assert response.status_code != 404


def test_invalid_auth():
    """Test that invalid auth is rejected"""
    response = client.post(
        "/query",
        json={"question": "test"},
        headers={"X-API-KEY": "invalid_key"}
    )
    assert response.status_code in [401, 403]


@patch('src.app.api.API_KEY_SECRET', "test_key")
def test_empty_question_validation():
    """Test that empty questions are rejected"""
    response = client.post(
        "/query",
        json={"question": ""},
        headers={"X-API-KEY": "test_key"}
    )
    # Should return 400 for empty question
    assert response.status_code == 400


@patch('src.app.api.API_KEY_SECRET', "test_key")
def test_missing_question_validation():
    """Test that missing question field is rejected"""
    response = client.post(
        "/query",
        json={},
        headers={"X-API-KEY": "test_key"}
    )
    # Should return 422 for missing required field
    assert response.status_code == 422


def test_malformed_json():
    """Test that malformed JSON is rejected"""
    response = client.post(
        "/query",
        data="invalid json",
        headers={"Content-Type": "application/json", "X-API-KEY": "test"}
    )
    assert response.status_code == 422 