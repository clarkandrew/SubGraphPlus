import sys
import pytest
import json
import requests
from pathlib import Path
from time import sleep
from unittest.mock import patch, MagicMock

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.api import app
from app.models import Triple
from fastapi.testclient import TestClient

# Create test client
client = TestClient(app)

"""
Smoke Tests for SubgraphRAG+
These tests verify basic functionality and edge cases without detailed mocking
"""

@pytest.fixture
def mock_api_key():
    """Mock API key for testing"""
    return "test_api_key"

@pytest.fixture
def mock_auth_header(mock_api_key):
    """Mock authorization header for testing"""
    with patch('app.config.API_KEY_SECRET', mock_api_key):
        return {"X-API-KEY": mock_api_key}

class TestBasicSmoke:
    """Basic smoke tests for core functionality"""
    
    def test_api_startup(self):
        """Test that the API can start up without errors"""
        response = client.get("/healthz")
        assert response.status_code == 200
    
    def test_null_bytes_in_question(self, mock_auth_header):
        """Test handling of null bytes in question"""
        response = client.post(
            "/query",
            json={"question": "Who is Elon Musk?\0"},  # Question with null byte
            headers=mock_auth_header
        )
        # Should sanitize the input and not crash
        assert response.status_code in [200, 400, 422]
    
    def test_control_characters_in_query(self, mock_auth_header):
        """Test handling of control characters in query"""
        control_chars = "".join(chr(i) for i in range(32) if i not in [9, 10, 13])
        response = client.post(
            "/query",
            json={"question": f"Who is Elon Musk?{control_chars}"},
            headers=mock_auth_header
        )
        # Should sanitize the input and not crash
        assert response.status_code in [200, 400, 422]
    
    def test_empty_ingest(self, mock_auth_header):
        """Test ingesting empty triples list"""
        response = client.post(
            "/ingest",
            json={"triples": []},
            headers=mock_auth_header
        )
        assert response.status_code == 400
        assert "EMPTY_TRIPLES" in response.text
    
    @patch('app.ml.llm.generate_answer')
    def test_llm_timeout(self, mock_generate_answer, mock_auth_header):
        """Test LLM timeout handling"""
        # Make LLM generation take too long
        mock_generate_answer.side_effect = requests.exceptions.Timeout("LLM request timed out")
        
        response = client.post(
            "/query",
            json={"question": "Who is Elon Musk?"},
            headers=mock_auth_header,
            stream=True
        )
        
        assert response.status_code == 200
        
        # Process the streaming response
        error_found = False
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data:'):
                    data = json.loads(line[5:])
                    if 'event' in data and data['event'] == 'error':
                        error_found = True
                        assert "timeout" in data['data']['message'].lower()
                        break
        
        assert error_found

class TestEdgeCaseHandling:
    """Tests for various edge cases in API behavior"""
    
    def test_unicode_handling(self, mock_auth_header):
        """Test handling of complex Unicode in query"""
        # Mix of emojis, RTL text, special characters
        complex_question = "Tell me about 🚀 SpaceX and العربية 官话 اردو?"
        
        response = client.post(
            "/query",
            json={"question": complex_question},
            headers=mock_auth_header
        )
        
        # Should handle the Unicode without crashing
        assert response.status_code in [200, 404, 409]
    
    def test_long_feedback(self, mock_auth_header):
        """Test feedback with very long comment"""
        # Generate a very long comment (10KB)
        long_comment = "a" * 10_000
        
        response = client.post(
            "/feedback",
            json={
                "query_id": "q_abc123",
                "is_correct": False,
                "comment": long_comment
            },
            headers=mock_auth_header
        )
        
        # Should handle or truncate the long comment
        assert response.status_code in [200, 400, 422]
    
    @patch('app.api.link_entities_v2')
    def test_many_entity_candidates(self, mock_link_entities, mock_auth_header):
        """Test handling of a query with many potential entity matches"""
        # Return a large number of entity candidates
        mock_link_entities.return_value = [(f"ent{i}", 0.8) for i in range(100)]
        
        response = client.post(
            "/query",
            json={"question": "Tell me about Apple"},
            headers=mock_auth_header
        )
        
        # Should handle many candidates without crashing
        assert response.status_code == 200
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        # This is a basic concurrency test, not a full load test
        from concurrent.futures import ThreadPoolExecutor
        
        def make_request():
            return client.get("/healthz")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            
            for future in futures:
                response = future.result()
                assert response.status_code == 200

class TestResourceHandling:
    """Tests for resource usage and handling"""
    
    def test_memory_limits(self, mock_auth_header):
        """Test behavior with large payload (not exceeding limits)"""
        # Generate a question that's large but within limits (50KB)
        question = "Who is " + "a" * 49_000 + "?"
        
        response = client.post(
            "/query",
            json={"question": question},
            headers=mock_auth_header
        )
        
        # Should either accept or reject with appropriate error, not crash
        assert response.status_code in [200, 400, 413, 422]
    
    @patch('app.api.hybrid_retrieve_v2')
    def test_many_triples(self, mock_hybrid_retrieve, mock_auth_header):
        """Test handling of a very large number of retrieved triples"""
        # Return a large number of triples
        mock_hybrid_retrieve.return_value = [
            Triple(
                id=f"triple_{i}",
                head_id=f"entity_{i}",
                head_name=f"Entity {i}",
                relation_id=f"rel_{i}",
                relation_name="related to",
                tail_id=f"entity_{i+1}",
                tail_name=f"Entity {i+1}",
                confidence=0.9
            )
            for i in range(1000)
        ]
        
        response = client.post(
            "/query",
            json={"question": "Tell me about everything"},
            headers=mock_auth_header,
            stream=True
        )
        
        # Should handle the large number of triples without crashing
        assert response.status_code == 200
        
        # Just verify the stream starts properly
        started = False
        for i, line in enumerate(response.iter_lines()):
            if line:
                started = True
                break
            if i >= 5:  # Check just the first few lines
                break
        
        assert started

if __name__ == '__main__':
    pytest.main()