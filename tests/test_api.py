import sys
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.api import app
from app.models import Triple, RetrievalEmpty, EntityLinkingError, AmbiguousEntityError

# Create test client
client = TestClient(app)

@pytest.fixture
def mock_api_key():
    """Mock API key for testing"""
    return "test_api_key"

@pytest.fixture
def mock_auth_header(mock_api_key):
    """Mock authorization header for testing"""
    with patch('app.api.API_KEY_SECRET', mock_api_key):
        return {"X-API-KEY": mock_api_key}

@pytest.fixture
def invalid_auth_header():
    """Invalid authorization header for testing"""
    return {"X-API-KEY": "invalid_key"}

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_readiness_check_success(mock_neo4j, mock_sqlite, mock_faiss_index):
    """Test readiness check endpoint with all dependencies available"""
    # Configure mocks to return True
    mock_neo4j.verify_connectivity.return_value = True
    mock_sqlite.verify_connectivity.return_value = True
    mock_faiss_index.is_trained.return_value = True
    
    # Add LLM and embedder health check mocks
    with patch('app.api.llm_health_check', return_value=True), \
         patch('app.api.embedder_health_check', return_value=True):
        
        response = client.get("/readyz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["checks"]["sqlite"] == "ok"
        assert data["checks"]["neo4j"] == "ok"
        assert data["checks"]["faiss_index"] == "ok"
        assert data["checks"]["llm_backend"] == "ok"
        assert data["checks"]["embedder"] == "ok"

def test_readiness_check_failure(mock_neo4j, mock_sqlite, mock_faiss_index):
    """Test readiness check endpoint with failed dependency"""
    # Configure mocks to simulate Neo4j failure
    mock_neo4j.verify_connectivity.return_value = False
    mock_sqlite.verify_connectivity.return_value = True
    mock_faiss_index.is_trained.return_value = True
    
    # Add LLM and embedder health check mocks
    with patch('app.api.llm_health_check', return_value=True), \
         patch('app.api.embedder_health_check', return_value=True):
        
        response = client.get("/readyz")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["checks"]["sqlite"] == "ok"
        assert data["checks"]["neo4j"] == "failed"
        assert data["checks"]["faiss_index"] == "ok"
        assert data["checks"]["llm_backend"] == "ok"
        assert data["checks"]["embedder"] == "ok"

def test_auth_required(invalid_auth_header):
    """Test that authentication is required for protected endpoints"""
    # Test query endpoint
    response = client.post("/query", json={"question": "Who is Elon Musk?"}, headers=invalid_auth_header)
    assert response.status_code == 401
    
    # Test graph browse endpoint
    response = client.get("/graph/browse", headers=invalid_auth_header)
    assert response.status_code == 401
    
    # Test ingest endpoint
    response = client.post("/ingest", json={"triples": []}, headers=invalid_auth_header)
    assert response.status_code == 401
    
    # Test feedback endpoint
    response = client.post("/feedback", json={"query_id": "test", "is_correct": True}, headers=invalid_auth_header)
    assert response.status_code == 401

def test_query_empty_question(mock_auth_header):
    """Test query endpoint with empty question"""
    response = client.post("/query", json={"question": ""}, headers=mock_auth_header)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["code"] == "EMPTY_QUERY"

@patch('app.api.extract_query_entities')
@patch('app.api.link_entities_v2')
def test_query_no_entity_match(mock_link_entities, mock_extract_entities, mock_auth_header):
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

@patch('app.api.extract_query_entities')
@patch('app.api.link_entities_v2')
@patch('app.api.hybrid_retrieve_v2')
def test_query_retrieval_empty(mock_hybrid_retrieve, mock_link_entities, mock_extract_entities, mock_auth_header):
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

@patch('app.api.extract_query_entities')
@patch('app.api.link_entities_v2')
def test_query_ambiguous_entities(mock_link_entities, mock_extract_entities, mock_auth_header):
    """Test query with ambiguous entity matches"""
    # Mock entity extraction
    mock_extract_entities.return_value = ["Apple"]
    
    # Mock entity linking to raise AmbiguousEntityError
    candidates = [
        {"id": "ent1", "name": "Apple Inc."},
        {"id": "ent2", "name": "Apple (fruit)"}
    ]
    mock_link_entities.side_effect = AmbiguousEntityError("Ambiguous entity", candidates)
    
    response = client.post(
        "/query",
        json={"question": "Tell me about Apple"},
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
    
    # Should contain an error event about ambiguous entities
    error_events = [e for e in events if e.get("event") == "error"]
    assert len(error_events) > 0
    assert error_events[0]["data"]["code"] == "AMBIGUOUS_ENTITIES"
    assert "candidates" in error_events[0]["data"]

def test_graph_browse(mock_neo4j, mock_auth_header):
    """Test graph browse endpoint"""
    # Mock Neo4j response for count query
    mock_neo4j.run_query.side_effect = [
        # First call (count query)
        [{"total": 100}],
        # Second call (data query)
        [
            {
                "source_id": "ent1",
                "source_name": "Elon Musk",
                "source_type": "Person",
                "target_id": "ent2",
                "target_name": "Tesla Inc.",
                "target_type": "Organization",
                "relation_id": "rel1",
                "relation_name": "founded"
            }
        ]
    ]
    
    # Call the endpoint
    response = client.get("/graph/browse", headers=mock_auth_header)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check pagination data
    assert data["page"] == 1
    assert data["limit"] == 500
    assert data["total_links_in_filter"] == 100
    assert data["has_more"] == False
    
    # Check nodes and links
    assert len(data["nodes"]) == 2  # Two nodes
    assert len(data["links"]) == 1  # One link
    
    # Check node data
    node_ids = [node["id"] for node in data["nodes"]]
    assert "ent1" in node_ids
    assert "ent2" in node_ids
    
    # Check link data
    assert data["links"][0]["source"] == "ent1"
    assert data["links"][0]["target"] == "ent2"
    assert data["links"][0]["relation_name"] == "founded"

def test_ingest_empty_triples(mock_auth_header):
    """Test ingest endpoint with empty triples"""
    response = client.post("/ingest", json={"triples": []}, headers=mock_auth_header)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["code"] == "EMPTY_TRIPLES"

def test_ingest_success(mock_sqlite, mock_auth_header):
    """Test successful ingest"""
    # Mock SQLite to simulate successful ingestion
    mock_sqlite.execute.return_value = None
    
    # Call the endpoint
    response = client.post(
        "/ingest", 
        json={
            "triples": [
                {"head": "Elon Musk", "relation": "founded", "tail": "Tesla Inc."},
                {"head": "Elon Musk", "relation": "founded", "tail": "SpaceX"}
            ]
        },
        headers=mock_auth_header
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert data["triples_staged"] == 2
    assert len(data["errors"]) == 0

def test_feedback_endpoint(mock_sqlite, mock_auth_header):
    """Test feedback endpoint"""
    # Mock SQLite
    mock_sqlite.execute.return_value = None
    
    # Mock file operations
    with patch('builtins.open', MagicMock()), \
         patch('json.dump', MagicMock()):
        
        # Call the endpoint
        response = client.post(
            "/feedback",
            json={
                "query_id": "q_abc123",
                "is_correct": True,
                "comment": "Great answer!",
                "expected_answer": None
            },
            headers=mock_auth_header
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert data["message"] == "Feedback recorded successfully"
        
        # Check that SQLite was called
        mock_sqlite.execute.assert_called_once()

def test_metrics_endpoint():
    """Test metrics endpoint"""
    # Need to patch the Prometheus instrumentator
    with patch('app.api.instrumentator.expose', return_value="prometheus_metrics"):
        response = client.get("/metrics")
        assert response.status_code == 200
        # Response should be plain text, not JSON
        assert response.headers["Content-Type"] == "text/plain; charset=utf-8"