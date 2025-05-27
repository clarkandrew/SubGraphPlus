import unittest
import pytest
import numpy as np
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.retriever import (
    hybrid_retrieve_v2, mlp_score, neo4j_get_neighborhood_triples,
    faiss_search_triples_data, entity_search, get_triple_embedding_from_faiss
)
from app.models import Triple, RetrievalEmpty


class TestRetriever(unittest.TestCase):
    
    @patch('app.retriever.faiss_index')
    def test_get_triple_embedding_from_faiss(self, mock_faiss_index):
        """Test retrieving triple embedding from FAISS"""
        # Mock faiss index to return a specific vector
        mock_vector = np.ones(1024, dtype=np.float32)
        mock_faiss_index.get_vector.return_value = mock_vector
        
        # Test with existing triple ID
        triple_id = "rel123"
        embedding = get_triple_embedding_from_faiss(triple_id)
        mock_faiss_index.get_vector.assert_called_once_with(triple_id)
        self.assertTrue(np.array_equal(embedding, mock_vector))
        
        # Reset mock
        mock_faiss_index.get_vector.reset_mock()
        mock_faiss_index.get_vector.return_value = None
        
        # Test with non-existent triple ID
        embedding = get_triple_embedding_from_faiss("nonexistent")
        self.assertEqual(embedding.shape, (1024,))  # Should return zero vector with HuggingFace dimension
        self.assertTrue(np.all(embedding == 0))  # All zeros
    
    # TODO: Fix this test - mocking is not working correctly
    # @patch('app.retriever.mlp_score')
    # def test_mlp_score_with_model(self, mock_mlp_score):
    #     """Test MLP scoring with available model"""
    #     # Mock the mlp_score function to return a specific score
    #     mock_mlp_score.return_value = 0.75
    #     
    #     # Test inputs
    #     query_emb = np.random.random(1024)
    #     triple_emb = np.random.random(1024)
    #     dde_features = [0.5, 0.3]
    #     
    #     # Call the function
    #     score = mlp_score(query_emb, triple_emb, dde_features)
    #     
    #     # Check results
    #     self.assertEqual(score, 0.75)
    #     # Check that function was called with correct args
    #     mock_mlp_score.assert_called_once_with(query_emb, triple_emb, dde_features)
    
    @patch('app.retriever.mlp_model', None)
    @patch('app.utils.heuristic_score')
    def test_mlp_score_fallback(self, mock_heuristic_score):
        """Test MLP scoring fallback when no model available"""
        # Mock heuristic score to return a specific score
        mock_heuristic_score.return_value = 0.6
        
        # Test inputs
        query_emb = np.random.random(1024)
        triple_emb = np.random.random(1024)
        dde_features = [0.5, 0.3]
        
        # Call the function
        score = mlp_score(query_emb, triple_emb, dde_features)
        
        # Check results
        self.assertEqual(score, 0.6)
        # Check that heuristic_score was called with correct args
        mock_heuristic_score.assert_called_once_with(query_emb, triple_emb, dde_features)
    
    @patch('app.retriever.neo4j_db')
    def test_neo4j_get_neighborhood_triples(self, mock_neo4j):
        """Test retrieving neighborhood triples from Neo4j"""
        # Mock Neo4j response
        mock_neo4j.run_query.return_value = [
            {
                "id": "rel1",
                "head_id": "ent1",
                "head_name": "Elon Musk",
                "relation_name": "founded",
                "tail_id": "ent2",
                "tail_name": "Tesla Inc."
            }
        ]
        
        # Call the function
        entity_ids = ["ent1"]
        triples = neo4j_get_neighborhood_triples(entity_ids, hops=2, limit=100)
        
        # Check results
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0].id, "rel1")
        self.assertEqual(triples[0].head_name, "Elon Musk")
        self.assertEqual(triples[0].relation_name, "founded")
        self.assertEqual(triples[0].tail_name, "Tesla Inc.")
        
        # Check that Neo4j was called with correct args
        mock_neo4j.run_query.assert_called_once()
        args, kwargs = mock_neo4j.run_query.call_args
        # The parameters are passed as the second positional argument (a dict)
        self.assertEqual(len(args), 2)  # query string and params dict
        params_dict = args[1]  # Second argument is the parameters dictionary
        self.assertEqual(params_dict["entity_ids"], entity_ids)
        self.assertEqual(params_dict["hops"], 2)
        self.assertEqual(params_dict["limit"], 100)
    
    @patch('app.retriever.neo4j_db')
    @patch('app.retriever.faiss_index')
    def test_faiss_search_triples_data(self, mock_faiss_index, mock_neo4j):
        """Test searching for triple data using FAISS"""
        # Mock FAISS search results
        mock_faiss_index.search.return_value = [("rel1", 0.95)]
        
        # Mock Neo4j response for retrieving triple data
        mock_neo4j.run_query.return_value = [
            {
                "id": "rel1",
                "head_id": "ent1",
                "head_name": "Elon Musk",
                "relation_name": "founded",
                "tail_id": "ent2",
                "tail_name": "Tesla Inc."
            }
        ]
        
        # Call the function
        query_embedding = np.random.random(1024)
        triples_data = faiss_search_triples_data(query_embedding, k=10)
        
        # Check results
        self.assertEqual(len(triples_data), 1)
        self.assertEqual(triples_data[0]["id"], "rel1")
        self.assertEqual(triples_data[0]["head_name"], "Elon Musk")
        self.assertEqual(triples_data[0]["relation_name"], "founded")
        self.assertEqual(triples_data[0]["tail_name"], "Tesla Inc.")
        self.assertEqual(triples_data[0]["relevance_score"], 0.95)
        
        # Check that FAISS was called with correct args
        mock_faiss_index.search.assert_called_once_with(query_embedding, 10)
        
        # Check that Neo4j was called with correct args
        mock_neo4j.run_query.assert_called_once()
        args, kwargs = mock_neo4j.run_query.call_args
        self.assertEqual(len(args), 2)  # query string and params dict
        params_dict = args[1]  # Second argument is the parameters dictionary
        self.assertEqual(params_dict["triple_ids"], ["rel1"])
    
    @patch('app.retriever.embed_query_cached')
    @patch('app.retriever.get_dde_for_entities')
    @patch('app.retriever.neo4j_get_neighborhood_triples')
    @patch('app.retriever.faiss_search_triples_data')
    @patch('app.retriever.get_triple_embedding_cached')
    @patch('app.retriever.extract_dde_features_for_triple')
    @patch('app.retriever.mlp_score_separate')
    @patch('app.retriever.greedy_connect_v2')
    def test_hybrid_retrieve_v2_success(self, mock_greedy_connect, mock_mlp_score, mock_extract_dde,
                                      mock_get_embedding, mock_faiss_search, mock_neo4j_get, 
                                      mock_get_dde, mock_embed_query):
        """Test hybrid retrieval success path"""
        # Mock function return values
        mock_embed_query.return_value = np.random.random(1024)
        mock_get_dde.return_value = {}
        
        # Create sample triples
        graph_triple = Triple(
            id="rel1", head_id="ent1", head_name="Elon Musk", relation_id="rel1",
            relation_name="founded", tail_id="ent2", tail_name="Tesla Inc.", properties={}
        )
        mock_neo4j_get.return_value = [graph_triple]
        
        # Mock FAISS search results
        mock_faiss_search.return_value = [{
            "id": "rel2", "head_id": "ent1", "head_name": "Elon Musk", "relation_id": "rel2",
            "relation_name": "founded", "tail_id": "ent3", "tail_name": "SpaceX"
        }]
        
        # Mock embedding and scoring
        mock_get_embedding.return_value = np.random.random(1024)
        mock_extract_dde.return_value = [0.5, 0.3]
        mock_mlp_score.return_value = 0.8
        
        # Mock greedy connect
        final_triples = [graph_triple]
        mock_greedy_connect.return_value = final_triples
        
        # Call the function
        result = hybrid_retrieve_v2("Who founded Tesla?", ["ent1"])
        
        # Check results
        self.assertEqual(result, final_triples)
        
        # Check that all components were called
        mock_embed_query.assert_called_once_with("Who founded Tesla?")
        mock_get_dde.assert_called_once_with(["ent1"], max_hops=2)
        mock_neo4j_get.assert_called_once()
        mock_faiss_search.assert_called_once()
        mock_mlp_score.assert_called()
        mock_greedy_connect.assert_called_once()
    
    @patch('app.retriever.embed_query_cached')
    @patch('app.retriever.get_dde_for_entities')
    @patch('app.retriever.neo4j_get_neighborhood_triples')
    @patch('app.retriever.faiss_search_triples_data')
    def test_hybrid_retrieve_v2_empty_results(self, mock_faiss_search, mock_neo4j_get,
                                            mock_get_dde, mock_embed_query):
        """Test hybrid retrieval with empty results"""
        # Mock functions to return empty results
        mock_embed_query.return_value = np.random.random(1024)
        mock_get_dde.return_value = {}
        mock_neo4j_get.return_value = []
        mock_faiss_search.return_value = []
        
        # Should raise RetrievalEmpty exception
        with self.assertRaises(RetrievalEmpty):
            hybrid_retrieve_v2("Who founded Tesla?", ["ent1"])
    
    @patch('app.retriever.neo4j_db')
    def test_entity_search(self, mock_neo4j):
        """Test entity search functionality"""
        # Mock Neo4j response
        mock_neo4j.run_query.return_value = [
            {
                "id": "ent1",
                "name": "Elon Musk",
                "type": "Person",
                "labels": ["Entity", "Person"]
            }
        ]
        
        # Call the function
        results = entity_search("Elon", limit=10)
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "ent1")
        self.assertEqual(results[0]["name"], "Elon Musk")
        self.assertEqual(results[0]["type"], "Person")
        
        # Check that Neo4j was called with correct args
        mock_neo4j.run_query.assert_called_once()
        args, kwargs = mock_neo4j.run_query.call_args
        self.assertEqual(len(args), 2)  # query string and params dict
        params_dict = args[1]  # Second argument is the parameters dictionary
        self.assertEqual(params_dict["search_term"], "Elon")
        self.assertEqual(params_dict["limit"], 10)


if __name__ == '__main__':
    unittest.main()