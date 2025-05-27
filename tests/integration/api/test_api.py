import sys
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from src.app.api import app
from src.app.models import Triple, RetrievalEmpty, EntityLinkingError, AmbiguousEntityError

# Create test client
client = TestClient(app)


@pytest.fixture
def mock_api_key():
    """Mock API key for testing"""
    return "test_api_key"


@pytest.fixture
def mock_auth_header(mock_api_key):
    """Mock authorization header for testing"""
    with patch('src.app.api.API_KEY_SECRET', mock_api_key):
        return {"X-API-KEY": mock_api_key}


@pytest.fixture
def invalid_auth_header():
    """Invalid authorization header for testing"""
    return {"X-API-KEY": "invalid_key"}


class TestHealthEndpoints:
    """Test health and readiness endpoints"""

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @patch('src.app.api.llm_health_check')
    @patch('src.app.api.embedder_health_check')
    @patch('src.app.api.faiss_index')
    @patch('src.app.api.neo4j_db')
    @patch('src.app.api.sqlite_db')
    def test_readiness_check_success(self, mock_sqlite, mock_neo4j, mock_faiss, mock_embedder_health, mock_llm_health):
        """Test readiness check when all services are healthy"""
        # Mock all health checks to return True
        mock_sqlite.verify_connectivity.return_value = True
        mock_neo4j.verify_connectivity.return_value = True
        mock_faiss.is_trained.return_value = True
        mock_embedder_health.return_value = True
        mock_llm_health.return_value = True
        
        response = client.get("/readyz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        # In testing mode, services return "mocked" or "ok"
        assert all(check in ["ok", "mocked"] for check in data["checks"].values())

    @patch('src.app.api.llm_health_check')
    @patch('src.app.api.embedder_health_check')
    @patch('src.app.api.faiss_index')
    @patch('src.app.api.neo4j_db')
    @patch('src.app.api.sqlite_db')
    def test_readiness_check_failure(self, mock_sqlite, mock_neo4j, mock_faiss, mock_embedder_health, mock_llm_health):
        """Test readiness check when some services are unhealthy"""
        # Mock some health checks to return False
        mock_sqlite.verify_connectivity.return_value = False
        mock_neo4j.verify_connectivity.return_value = True
        mock_faiss.is_trained.return_value = False
        mock_embedder_health.return_value = True
        mock_llm_health.return_value = False
        
        response = client.get("/readyz")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert "failed" in data["checks"]["sqlite"]
        assert "failed" in data["checks"]["faiss_index"]
        assert "failed" in data["checks"]["llm_backend"]


class TestAuthentication:
    """Test API authentication"""

    def test_auth_required_query(self, invalid_auth_header):
        """Test that authentication is required for query endpoint"""
        response = client.post(
            "/query", 
            json={"question": "Who is Elon Musk?"}, 
            headers=invalid_auth_header
        )
        assert response.status_code == 401

    def test_auth_required_graph_browse(self, invalid_auth_header):
        """Test that authentication is required for graph browse endpoint"""
        response = client.get("/graph/browse", headers=invalid_auth_header)
        assert response.status_code == 401

    def test_auth_required_ingest(self, invalid_auth_header):
        """Test that authentication is required for ingest endpoint"""
        response = client.post(
            "/ingest", 
            json={"triples": []}, 
            headers=invalid_auth_header
        )
        assert response.status_code == 401

    def test_auth_required_feedback(self, invalid_auth_header):
        """Test that authentication is required for feedback endpoint"""
        response = client.post(
            "/feedback", 
            json={"query_id": "test", "rating": 5}, 
            headers=invalid_auth_header
        )
        assert response.status_code == 401


class TestQueryEndpoint:
    """Test query endpoint functionality"""

    def test_query_empty_question(self, mock_auth_header):
        """Test query endpoint with empty question"""
        response = client.post(
            "/query", 
            json={"question": ""}, 
            headers=mock_auth_header
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "EMPTY_QUERY"

    def test_query_whitespace_question(self, mock_auth_header):
        """Test query endpoint with whitespace-only question"""
        response = client.post(
            "/query", 
            json={"question": "   "}, 
            headers=mock_auth_header
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "EMPTY_QUERY"

    @patch('src.app.utils.extract_query_entities')
    @patch('src.app.utils.link_entities_v2')
    def test_query_no_entity_match(self, mock_link_entities, 
                                 mock_extract_entities, mock_auth_header):
        """Test query with no entity matches"""
        # Mock entity extraction and linking to return no matches
        mock_extract_entities.return_value = ["unknown entity"]
        mock_link_entities.return_value = []  # No entity matches
        
        response = client.post(
            "/query",
            json={"question": "Who is unknown entity?"},
            headers=mock_auth_header
        )
        
        # Should return 200 with streaming response
        assert response.status_code == 200
        
        # Parse the streaming response
        events = []
        for line in response.iter_lines():
            if line:
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        
        # Should contain an error event about no entity match
        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) > 0
        assert error_events[0]["data"]["code"] == "NO_ENTITY_MATCH"

    @patch('src.app.utils.extract_query_entities')
    @patch('src.app.utils.link_entities_v2')
    @patch('src.app.retriever.hybrid_retrieve_v2')
    def test_query_retrieval_empty(self, mock_hybrid_retrieve, 
                                 mock_link_entities, mock_extract_entities, 
                                 mock_auth_header):
        """Test query when retrieval returns empty results"""
        # Mock entity extraction and linking
        mock_extract_entities.return_value = ["Elon Musk"]
        mock_link_entities.return_value = [("ent1", 0.9)]
        
        # Mock retrieval to raise RetrievalEmpty
        mock_hybrid_retrieve.side_effect = RetrievalEmpty("No relevant triples found")
        
        response = client.post(
            "/query",
            json={"question": "Who is Elon Musk?"},
            headers=mock_auth_header
        )
        
        # Should return 200 with streaming response
        assert response.status_code == 200
        
        # Parse the streaming response
        events = []
        for line in response.iter_lines():
            if line:
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        
        # Should contain an error event about no relevant triples
        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) > 0
        assert error_events[0]["data"]["code"] == "NO_RELEVANT_TRIPLES"

    @patch('src.app.utils.extract_query_entities')
    @patch('src.app.utils.link_entities_v2')
    @patch('src.app.retriever.hybrid_retrieve_v2')
    @patch('src.app.ml.llm.stream_tokens')
    @patch('src.app.verify.validate_llm_output')
    @patch('src.app.utils.triples_to_graph_data')
    def test_query_successful_response(self, mock_graph_data, mock_validate,
                                     mock_stream, mock_retrieve, mock_link,
                                     mock_extract, mock_auth_header):
        """Test successful query response"""
        # Mock entity extraction and linking
        mock_extract.return_value = ["Tesla"]
        mock_link.return_value = [("tesla_entity", 0.9)]
        
        # Mock successful retrieval
        mock_triple = Triple(
            id="triple_1",
            head_id="tesla",
            head_name="Tesla",
            relation_id="founded_by",
            relation_name="founded by",
            tail_id="elon_musk",
            tail_name="Elon Musk",
            properties={},
            relevance_score=0.9
        )
        mock_retrieve.return_value = [mock_triple]
        
        # Mock LLM streaming
        mock_stream.return_value = ["Tesla", " was", " founded", " by", " Elon", " Musk"]
        
        # Mock validation
        mock_validate.return_value = (
            ["Tesla was founded by Elon Musk"],
            {"triple_1"},
            "high"
        )
        
        # Mock graph data
        mock_graph_data.return_value = MagicMock()
        mock_graph_data.return_value.to_dict.return_value = {
            "nodes": [],
            "links": []
        }
        
        response = client.post(
            "/query",
            json={"question": "Who founded Tesla?", "visualize_graph": True},
            headers=mock_auth_header
        )
        
        assert response.status_code == 200
        
        # Verify streaming response format
        events = []
        for line in response.iter_lines():
            if line:
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        
        # Should have metadata, graph_data, tokens, citations, and end events
        event_types = [e.get("event") for e in events]
        assert "metadata" in event_types
        assert "graph_data" in event_types  # visualize_graph=True
        assert "llm_token" in event_types
        assert "citation_data" in event_types
        assert "end" in event_types


class TestGraphBrowseEndpoint:
    """Test graph browse endpoint functionality"""

    @patch('src.app.retriever.entity_search')
    def test_graph_browse_basic(self, mock_entity_search, mock_auth_header):
        """Test basic graph browse functionality"""
        # Mock the entity search to return some results
        mock_entity_search.return_value = ([], 0, True)  # triples, total, has_next
        
        response = client.get("/graph/browse", headers=mock_auth_header)
        assert response.status_code == 200
        
        data = response.json()
        assert "triples" in data
        assert "pagination" in data

    def test_graph_browse_pagination(self, mock_auth_header):
        """Test graph browse with pagination parameters"""
        response = client.get(
            "/graph/browse?page=2&limit=10", 
            headers=mock_auth_header
        )
        assert response.status_code == 200


class TestIngestEndpoint:
    """Test ingest endpoint functionality"""

    def test_ingest_empty_triples(self, mock_auth_header):
        """Test ingesting empty triples list"""
        response = client.post(
            "/ingest",
            json={"triples": []},
            headers=mock_auth_header
        )
        assert response.status_code == 400
        assert "EMPTY_TRIPLES" in response.text

    @patch('src.app.database.neo4j_db.batch_ingest_triples')
    @patch('src.app.database.faiss_index.add_embeddings')
    def test_ingest_valid_triples(self, mock_faiss, mock_neo4j, mock_auth_header):
        """Test ingesting valid triples"""
        mock_neo4j.return_value = {"created": 1, "updated": 0}
        mock_faiss.return_value = None
        
        response = client.post(
            "/ingest",
            json={
                "triples": [{
                    "head": "Tesla",
                    "relation": "founded_by",
                    "tail": "Elon Musk",
                    "head_name": "Tesla Inc.",
                    "relation_name": "founded by",
                    "tail_name": "Elon Musk"
                }],
                "source": "test",
                "batch_id": "test_batch_1"
            },
            headers=mock_auth_header
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"


class TestFeedbackEndpoint:
    """Test feedback endpoint functionality"""

    @patch('src.app.database.sqlite_db.execute')
    def test_feedback_submission(self, mock_execute, mock_auth_header):
        """Test feedback submission"""
        mock_execute.return_value = None
        
        response = client.post(
            "/feedback",
            json={
                "query_id": "test_query_123",
                "rating": 4,
                "feedback_type": "accuracy",
                "comments": "Good answer"
            },
            headers=mock_auth_header
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"
        assert "feedback_id" in data


class TestMetricsEndpoint:
    """Test metrics endpoint"""

    def test_metrics_endpoint(self):
        """Test metrics endpoint returns Prometheus format"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        # Should contain some basic metrics
        content = response.text
        assert "subgraphrag" in content.lower() or "http" in content.lower() 