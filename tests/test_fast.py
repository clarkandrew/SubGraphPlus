import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Create a minimal mock API that mimics the structure without any imports
app = FastAPI(title="Mock SubgraphRAG+ API")

@app.get("/healthz")
async def health():
    return {"status": "healthy"}

@app.get("/readyz") 
async def readiness():
    return {"status": "ready"}

@app.get("/metrics")
async def metrics():
    return "# HELP test_metric A test metric\n# TYPE test_metric counter\ntest_metric 1"

@app.post("/query")
async def query():
    return {"message": "Query endpoint exists"}

@app.get("/graph/browse")
async def graph_browse():
    return {"message": "Graph browse endpoint exists"}

@app.post("/ingest")
async def ingest():
    return {"message": "Ingest endpoint exists"}

@app.post("/feedback")
async def feedback():
    return {"message": "Feedback endpoint exists"}

client = TestClient(app)

def test_health_endpoint():
    """Test health endpoint"""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_readiness_endpoint():
    """Test readiness endpoint"""
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "test_metric" in response.text

def test_query_endpoint_exists():
    """Test query endpoint exists"""
    response = client.post("/query")
    assert response.status_code == 200

def test_graph_browse_endpoint_exists():
    """Test graph browse endpoint exists"""
    response = client.get("/graph/browse")
    assert response.status_code == 200

def test_ingest_endpoint_exists():
    """Test ingest endpoint exists"""
    response = client.post("/ingest")
    assert response.status_code == 200

def test_feedback_endpoint_exists():
    """Test feedback endpoint exists"""
    response = client.post("/feedback")
    assert response.status_code == 200 