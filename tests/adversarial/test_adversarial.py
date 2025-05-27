import sys
import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from src.app.api import app
from src.app.verify import validate_llm_output
from src.app.models import EntityLinkingError, AmbiguousEntityError
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
        answers, cited_ids, trust_level = validate_llm_output(
            malicious_output, provided_triple_ids
        )

        assert trust_level == "low_injection_detected"
        assert len(answers) == 0
        assert len(cited_ids) == 0

    def test_citation_injection(self):
        """Test LLM output that cites non-existent triple IDs"""
        malicious_output = """
        ans: Legitimate looking answer (id=999)
        """

        provided_triple_ids = {"123", "456", "789"}
        answers, cited_ids, trust_level = validate_llm_output(
            malicious_output, provided_triple_ids
        )

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
        answers, cited_ids, trust_level = validate_llm_output(
            malicious_output, provided_triple_ids
        )

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
        answers, cited_ids, trust_level = validate_llm_output(
            malicious_output, provided_triple_ids
        )

        # Should sanitize comments
        assert trust_level == "low_invalid_citation"


class TestAPIEdgeCases:
    """Test API edge cases and potential vulnerabilities"""

    @pytest.fixture
    def mock_auth_header(self):
        """Mock authorization header for testing"""
        with patch('src.app.api.API_KEY_SECRET', "test_api_key"):
            return {"X-API-KEY": "test_api_key"}

  

    @patch('src.app.database.sqlite_db.execute')
    def test_malicious_sqlite_injection(self, mock_execute, mock_auth_header):
        """Test SQL injection attempt in feedback endpoint"""
        # Mock SQLite to check what it's called with
        mock_execute.return_value = None

        # Call with malicious SQL injection payload
        injection_payload = "'; DROP TABLE feedback; --"

        response = client.post(
            "/feedback",
            json={
                "query_id": injection_payload,
                "rating": 5,
                "feedback_type": "accuracy",
                "comments": "Great answer!"
            },
            headers=mock_auth_header
        )

        # Should still work (parametrized queries prevent injection)
        assert response.status_code == 200

        # Check that SQLite was called with safe parameters
        mock_execute.assert_called()

    @patch('src.app.database.neo4j_db.run_query')
    def test_neo4j_injection_attempt(self, mock_neo4j_run, mock_auth_header):
        """Test Cypher injection attempt in graph browse endpoint"""
        # Mock Neo4j run_query
        mock_neo4j_run.return_value = []
        
        # Add a parameter with potential Cypher injection
        injection_param = "MATCH (n) DETACH DELETE n RETURN"

        response = client.get(
            f"/graph/browse?search_term={injection_param}",
            headers=mock_auth_header
        )

        # Should still work (parametrized queries prevent injection)
        assert response.status_code == 200

        # Check that Neo4j was called (parameters should be sanitized)
        mock_neo4j_run.assert_called()

    @patch('src.app.utils.extract_query_entities')
    @patch('src.app.utils.link_entities_v2')
    def test_adversarial_entity_linking(self, mock_link_entities, 
                                      mock_extract_entities, mock_auth_header):
        """Test adversarial entity linking attempts"""
        # Mock entity extraction to return the malicious input
        mock_extract_entities.return_value = ["<script>alert('xss')</script>"]
        mock_link_entities.return_value = []
        
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


class TestRetrievalEdgeCases:
    """Test edge cases in the retrieval pipeline"""

    @patch('src.app.retriever.neo4j_get_neighborhood_triples')
    @patch('src.app.retriever.faiss_search_triples_data')
    def test_empty_index(self, mock_faiss_search, mock_neo4j_get):
        """Test behavior when FAISS index is empty"""
        from src.app.retriever import hybrid_retrieve_v2
        from src.app.models import RetrievalEmpty

        # Configure mocks to return empty results
        mock_neo4j_get.return_value = []
        mock_faiss_search.return_value = []

        # Should raise RetrievalEmpty
        with pytest.raises(RetrievalEmpty):
            hybrid_retrieve_v2("test question", ["entity1"])

    @patch('src.app.retriever.hybrid_retrieve_v2')
    def test_nan_embeddings(self, mock_hybrid_retrieve):
        """Test handling of NaN values in embeddings"""
        from src.app.models import Triple
        
        # Create a triple with NaN relevance score
        triple_with_nan = Triple(
            id="test_triple",
            head_id="entity1",
            head_name="Entity 1",
            relation_id="relation1",
            relation_name="relates to",
            tail_id="entity2",
            tail_name="Entity 2",
            properties={},
            relevance_score=float('nan')
        )
        
        mock_hybrid_retrieve.return_value = [triple_with_nan]
        
        # Should handle NaN values gracefully
        from src.app.utils import get_score_for_triple
        scored_triples = [(0.5, triple_with_nan)]
        score = get_score_for_triple(triple_with_nan.id, scored_triples)
        assert not np.isnan(score)  # Should return a valid number


class TestConcurrency:
    """Test concurrent access patterns"""

    @pytest.fixture
    def mock_auth_header(self):
        """Mock authorization header for testing"""
        with patch('src.app.api.API_KEY_SECRET', "test_api_key"):
            return {"X-API-KEY": "test_api_key"}

    def test_concurrent_health_checks(self):
        """Test concurrent health check requests"""
        from concurrent.futures import ThreadPoolExecutor
        
        def make_health_request():
            return client.get("/healthz")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_health_request) for _ in range(10)]
            
            for future in futures:
                response = future.result()
                assert response.status_code == 200

    @patch('src.app.utils.extract_query_entities')
    @patch('src.app.utils.link_entities_v2')
    def test_concurrent_queries(self, mock_link_entities, 
                              mock_extract_entities, mock_auth_header):
        """Test concurrent query requests"""
        from concurrent.futures import ThreadPoolExecutor
        
        # Mock to return no entities (fast path)
        mock_extract_entities.return_value = []
        mock_link_entities.return_value = []
        
        def make_query_request():
            return client.post(
                "/query",
                json={"question": "test question"},
                headers=mock_auth_header
            )
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_query_request) for _ in range(5)]
            
            for future in futures:
                response = future.result()
                assert response.status_code == 200


class TestResourceLimits:
    """Test resource limits and memory usage"""

    @pytest.fixture
    def mock_auth_header(self):
        """Mock authorization header for testing"""
        with patch('src.app.api.API_KEY_SECRET', "test_api_key"):
            return {"X-API-KEY": "test_api_key"}

    def test_large_feedback_comment(self, mock_auth_header):
        """Test feedback with very long comment"""
        # Generate a very long comment (10KB)
        long_comment = "a" * 10_000
        
        response = client.post(
            "/feedback",
            json={
                "query_id": "q_abc123",
                "rating": 3,
                "feedback_type": "accuracy",
                "comments": long_comment
            },
            headers=mock_auth_header
        )
        
        # Should handle or truncate the long comment
        assert response.status_code in [200, 400, 422]

    @patch('src.app.utils.extract_query_entities')
    @patch('src.app.utils.link_entities_v2')
    def test_many_entity_candidates(self, mock_link_entities, 
                                  mock_extract_entities, mock_auth_header):
        """Test handling of a query with many potential entity matches"""
        # Return a large number of entity candidates
        mock_extract_entities.return_value = ["Apple"]
        mock_link_entities.return_value = [(f"ent{i}", 0.8) for i in range(100)]
        
        response = client.post(
            "/query",
            json={"question": "Tell me about Apple"},
            headers=mock_auth_header
        )
        
        # Should handle many candidates without crashing
        assert response.status_code == 200

    @patch('src.app.retriever.hybrid_retrieve_v2')
    def test_large_triple_set(self, mock_hybrid_retrieve, mock_auth_header):
        """Test handling of a very large number of retrieved triples"""
        from src.app.models import Triple
        
        # Return a large number of triples
        mock_triples = []
        for i in range(1000):
            mock_triples.append(Triple(
                id=f"triple_{i}",
                head_id=f"entity_{i}",
                head_name=f"Entity {i}",
                relation_id=f"rel_{i}",
                relation_name="related to",
                tail_id=f"entity_{i+1}",
                tail_name=f"Entity {i+1}",
                properties={},
                relevance_score=0.9
            ))
        
        mock_hybrid_retrieve.return_value = mock_triples
        
        # Mock other dependencies for successful response
        with patch('src.app.utils.extract_query_entities') as mock_extract, \
             patch('src.app.utils.link_entities_v2') as mock_link, \
             patch('src.app.ml.llm.stream_tokens') as mock_stream, \
             patch('src.app.verify.validate_llm_output') as mock_validate:
            
            mock_extract.return_value = ["Tesla"]
            mock_link.return_value = [("tesla", 0.9)]
            mock_stream.return_value = ["Test", " response"]
            mock_validate.return_value = (["Test response"], set(), "high")
            
            response = client.post(
                "/query",
                json={"question": "Tell me about Tesla"},
                headers=mock_auth_header
            )
            
            # Should handle large result set without crashing
            assert response.status_code == 200


class TestErrorRecovery:
    """Test error recovery and resilience"""

    @pytest.fixture
    def mock_auth_header(self):
        """Mock authorization header for testing"""
        with patch('src.app.api.API_KEY_SECRET', "test_api_key"):
            return {"X-API-KEY": "test_api_key"}

    @patch('src.app.ml.llm.stream_tokens')
    def test_llm_timeout_recovery(self, mock_stream_tokens, mock_auth_header):
        """Test recovery from LLM timeout"""
        import requests
        
        # Mock LLM to raise timeout
        mock_stream_tokens.side_effect = requests.exceptions.Timeout("LLM request timed out")
        
        # Mock other dependencies for a valid pipeline up to LLM
        with patch('src.app.utils.extract_query_entities') as mock_extract, \
             patch('src.app.utils.link_entities_v2') as mock_link, \
             patch('src.app.retriever.hybrid_retrieve_v2') as mock_retrieve:
            
            mock_extract.return_value = ["Tesla"]
            mock_link.return_value = [("tesla", 0.9)]
            mock_retrieve.return_value = [MagicMock()]  # Mock triple
            
            response = client.post(
                "/query",
                json={"question": "Tell me about Tesla"},
                headers=mock_auth_header
            )
            
            # Should handle timeout gracefully
            assert response.status_code == 200
            
            # Check that we get an error in the stream
            error_found = False
            for line in response.iter_lines():
                if line:
                    try:
                        event = json.loads(line)
                        if event.get("event") == "error":
                            error_found = True
                            break
                    except json.JSONDecodeError:
                        continue
            
            assert error_found 