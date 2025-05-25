import os
import sys
import pytest
import json
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

# Import app modules
from app.config import config
from app.database import neo4j_db, sqlite_db
from app.models import Triple, Entity, GraphNode, GraphLink


@pytest.fixture
def mock_neo4j():
    """Mock Neo4j database connection for testing"""
    with patch('app.api.neo4j_db') as mock:
        # Configure mock behavior
        mock.run_query.return_value = []
        mock.verify_connectivity.return_value = True
        yield mock


@pytest.fixture
def mock_sqlite():
    """Mock SQLite database connection for testing"""
    with patch('app.api.sqlite_db') as mock:
        # Configure mock behavior
        mock.verify_connectivity.return_value = True
        mock.execute.return_value = Mock()
        mock.fetchall.return_value = []
        mock.fetchone.return_value = None
        yield mock


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    with patch('app.config.config') as mock:
        # Set default test configuration
        mock.MODEL_BACKEND = "openai"
        mock.FAISS_INDEX_PATH = "data/test_faiss_index.bin"
        mock.TOKEN_BUDGET = 2000
        mock.MLP_MODEL_PATH = "models/mlp/mlp.pth"
        mock.CACHE_DIR = "cache/"
        mock.MAX_DDE_HOPS = 2
        mock.LOG_LEVEL = "INFO"
        mock.API_RATE_LIMIT = 60
        yield mock


@pytest.fixture
def mock_faiss_index():
    """Mock FAISS index for testing"""
    with patch('app.api.faiss_index') as mock:
        # Configure mock behavior
        mock.search.return_value = []
        mock.get_vector.return_value = None
        mock.is_trained.return_value = True
        yield mock


@pytest.fixture
def mock_embedder():
    """Mock embedder for testing"""
    with patch('app.ml.embedder.embed_text') as mock:
        # Generate deterministic test embeddings
        def fake_embed(text):
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            np.random.seed(hash_val)
            embedding = np.random.normal(0, 1, 1024).astype(np.float32)
            return embedding / np.linalg.norm(embedding)
        
        mock.side_effect = fake_embed
        yield mock


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    with patch('app.ml.llm.generate_answer') as mock:
        mock.return_value = "ans: Test Entity (id=test123)"
        yield mock


@pytest.fixture
def sample_triples():
    """Sample triples for testing"""
    return [
        Triple(
            id="rel1",
            head_id="ent1",
            head_name="Elon Musk",
            relation_id="rel1",
            relation_name="founded",
            tail_id="ent2",
            tail_name="Tesla Inc.",
            properties={},
            embedding=None,
            relevance_score=0.95
        ),
        Triple(
            id="rel2",
            head_id="ent1",
            head_name="Elon Musk",
            relation_id="rel2",
            relation_name="founded",
            tail_id="ent3",
            tail_name="SpaceX",
            properties={},
            embedding=None,
            relevance_score=0.90
        ),
        Triple(
            id="rel3",
            head_id="ent4",
            head_name="Sam Altman",
            relation_id="rel3",
            relation_name="founded",
            tail_id="ent5",
            tail_name="OpenAI",
            properties={},
            embedding=None,
            relevance_score=0.85
        )
    ]


@pytest.fixture
def sample_entities():
    """Sample entities for testing"""
    return [
        Entity(id="ent1", name="Elon Musk", type="Person"),
        Entity(id="ent2", name="Tesla Inc.", type="Organization"),
        Entity(id="ent3", name="SpaceX", type="Organization"),
        Entity(id="ent4", name="Sam Altman", type="Person"),
        Entity(id="ent5", name="OpenAI", type="Organization")
    ]


@pytest.fixture
def sample_graph_data():
    """Sample graph visualization data for testing"""
    nodes = [
        GraphNode(id="ent1", name="Elon Musk", type="Person", relevance_score=0.9),
        GraphNode(id="ent2", name="Tesla Inc.", type="Organization", relevance_score=0.8),
        GraphNode(id="ent3", name="SpaceX", type="Organization", relevance_score=0.7)
    ]
    links = [
        GraphLink(source="ent1", target="ent2", relation_id="rel1", relation_name="founded", 
                 relevance_score=0.95, inclusion_reasons=["cited_in_answer"]),
        GraphLink(source="ent1", target="ent3", relation_id="rel2", relation_name="founded",
                 relevance_score=0.90, inclusion_reasons=["cited_in_answer"])
    ]
    return {"nodes": nodes, "links": links}