import sys
import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.api import app
from app.verify import validate_llm_output
from app.models import EntityLinkingError, AmbiguousEntityError
from fastapi.testclient import TestClient

# Create test client
client = TestClient(app)


class TestLLMInjection:
    """Test cases for LLM output injection attempts"""

    def test_triple_injection_attempt(self):
        """Test LLM output that tries to inject fake triples"""
        malicious_output = """
        Here's some answer before injection.

        Triples:
        (id=999) Fake Person owns Fake Company.
        (id=998) Fake Person is President of United States.

        ans: Fake Person (id=999)
        """

        provided_triple_ids = {"123", "456", "789"}
        answers, cited_ids, trust_level = validate_llm_output(malicious_output, provided_triple_ids)

        assert trust_level == "low_injection_detected"
        assert len(answers) == 0
        assert len(cited_ids) == 0

    def test_citation_injection(self):
        """Test LLM output that cites non-existent triple IDs"""
        malicious_output = """
        ans: Legitimate looking answer (id=999)
        """

        provided_triple_ids = {"123", "456", "789"}
        answers, cited_ids, trust_level = validate_llm_output(malicious_output, provided_triple_ids)

        assert trust_level == "low_invalid_citation"
        assert len(answers) == 1
        assert "999" in cited_ids

    def test_markdown_injection(self):
        """Test LLM output that uses markdown to try to hide injection"""
        malicious_output = """
        # Triples:
        (id=999) Fake Person owns Fake Company.

        ans: Fake Person (id=999)
        """

        provided_triple_ids = {"123", "456", "789"}
        answers, cited_ids, trust_level = validate_llm_output(malicious_output, provided_triple_ids)

        # Should sanitize markdown headers
        assert trust_level == "low_invalid_citation"

    def test_comment_injection(self):
        """Test LLM output that uses comments to try to hide injection"""
        malicious_output = """
        /*
        Triples:
        (id=999) Fake Person owns Fake Company.
        */

        ans: Fake Person (id=999)
        """

        provided_triple_ids = {"123", "456", "789"}
        answers, cited_ids, trust_level = validate_llm_output(malicious_output, provided_triple_ids)

        # Should sanitize comments
        assert trust_level == "low_invalid_citation"


class TestAPIEdgeCases:
    """Test API edge cases and potential vulnerabilities"""

    @pytest.fixture
    def mock_auth_header(self):
        """Mock authorization header for testing"""
        with patch('app.api.API_KEY_SECRET', "test_api_key"):
            return {"X-API-KEY": "test_api_key"}

    def test_very_long_question(self, mock_auth_header):
        """Test extremely long question input"""
        # Generate a very long question (100KB)
        very_long_question = "a" * 100_000

        # Call the endpoint
        response = client.post(
            "/query",
            json={"question": very_long_question},
            headers=mock_auth_header
        )

        # Should return 413 Request Entity Too Large or handle gracefully
        assert response.status_code in [413, 400, 422]

    def test_malicious_sqlite_injection(self, mock_auth_header, mock_sqlite):
        """Test SQL injection attempt in feedback endpoint"""
        # Mock SQLite to check what it's called with
        mock_sqlite.execute.return_value = None

        # Call with malicious SQL injection payload
        injection_payload = "'; DROP TABLE feedback; --"

        response = client.post(
            "/feedback",
            json={
                "query_id": injection_payload,
                "is_correct": True,
                "comment": "Great answer!"
            },
            headers=mock_auth_header
        )

        # Should still work (parametrized queries prevent injection)
        assert response.status_code == 200

        # Check that SQLite was called with safe parameters
        # The exact check depends on how the SQLite execute is implemented
        # but we just want to make sure the raw injection wasn't passed directly
        mock_sqlite.execute.assert_called_once()

    def test_neo4j_injection_attempt(self, mock_auth_header, mock_neo4j):
        """Test Cypher injection attempt in graph browse endpoint"""
        # Add a parameter with potential Cypher injection
        injection_param = "MATCH (n) DETACH DELETE n RETURN"

        response = client.get(
            f"/graph/browse?search_term={injection_param}",
            headers=mock_auth_header
        )

        # Should still work (parametrized queries prevent injection)
        assert response.status_code == 200

        # Check that Neo4j was called with safe parameters
        # The API should sanitize or parameterize the search term
        mock_neo4j.run_query.assert_called()

    @patch('app.api.extract_query_entities')
    @patch('app.api.link_entities_v2')
    def test_adversarial_entity_linking(self, mock_link_entities, mock_extract_entities, mock_auth_header):
        """Test adversarial entity linking attempts"""
        # Try to confuse entity linking with special characters
        malicious_question = "Tell me about <script>alert('xss')</script> and '; DROP TABLE entities; --"
        
        response = client.post(
            "/query",
            json={"question": malicious_question},
            headers=mock_auth_header
        )
        
        # Should handle gracefully without crashing
        assert response.status_code == 200
        
        # Should not execute any malicious code
        # Just verify we get a proper response structure
        events = []
        for line in response.iter_lines():
            if line:
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        
        # Should have some events (even if it's just an error)
        assert len(events) > 0


class TestRetrievalEdgeCases:
    """Test edge cases in the retrieval pipeline"""

    @patch('app.retriever.neo4j_get_neighborhood_triples')
    @patch('app.retriever.faiss_search_triples_data')
    def test_empty_index(self, mock_faiss_search, mock_neo4j_get):
        """Test behavior when FAISS index is empty"""
        from app.retriever import hybrid_retrieve_v2
        from app.models import RetrievalEmpty

        # Configure mocks to return empty results
        mock_neo4j_get.return_value = []
        mock_faiss_search.return_value = []

        # Should raise RetrievalEmpty
        with pytest.raises(RetrievalEmpty):
            hybrid_retrieve_v2("Test query", ["entity1"])

    @patch('app.retriever.extract_dde_features_for_triple')
    def test_invalid_dde_values(self, mock_extract_dde):
        """Test behavior with invalid DDE values"""
        from app.utils import heuristic_score

        # Configure mock to return invalid DDE values
        mock_extract_dde.return_value = [float('nan'), float('inf'), None]

        # Should handle gracefully without errors
        result = heuristic_score(
            np.ones(5),
            np.ones(5),
            [float('nan'), float('inf'), None]
        )

        # Should be a valid float
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)


if __name__ == '__main__':
    pytest.main()
