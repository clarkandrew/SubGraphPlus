import unittest
import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.utils import (
    count_tokens, triple_to_string, hash_text, cosine_similarity, 
    normalize_dde_value, heuristic_score, 
    get_score_for_triple, greedy_connect_v2, triples_to_graph_data
)
from app.entity_typing import get_entity_type
from app.models import Triple


class TestUtilsFunctions(unittest.TestCase):
    
    def test_count_tokens(self):
        """Test token counting function"""
        text = "Hello world"
        count = count_tokens(text)
        self.assertGreater(count, 0)
        
        # Empty text should return 0
        self.assertEqual(count_tokens(""), 0)
        
        # Longer text should have more tokens
        text_long = "Hello world " * 10
        count_long = count_tokens(text_long)
        self.assertGreater(count_long, count)
    
    def test_triple_to_string(self):
        """Test triple to string conversion"""
        triple = Triple(
            id="rel1",
            head_id="ent1",
            head_name="Elon Musk",
            relation_id="rel1",
            relation_name="founded",
            tail_id="ent2",
            tail_name="Tesla Inc.",
            properties={},
            embedding=None,
            relevance_score=None
        )
        
        string_repr = triple_to_string(triple)
        self.assertEqual(string_repr, "Elon Musk founded Tesla Inc.")
    
    def test_hash_text(self):
        """Test text hashing function"""
        text = "Sample text"
        hash_val = hash_text(text)
        
        # Hash should be consistent
        self.assertEqual(hash_val, hash_text(text))
        
        # Different texts should have different hashes
        self.assertNotEqual(hash_val, hash_text("Different text"))
        
        # Hash should be a valid MD5 hex string (32 chars)
        self.assertEqual(len(hash_val), 32)
    
    def test_cosine_similarity(self):
        """Test cosine similarity function"""
        # Identical vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 1.0)
        
        # Orthogonal vectors
        vec2 = np.array([0, 1, 0])
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 0.0)
        
        # Opposite vectors
        vec2 = np.array([-1, 0, 0])
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), -1.0)
        
        # Edge cases
        self.assertEqual(cosine_similarity(None, vec2), 0.0)
        self.assertEqual(cosine_similarity(vec1, None), 0.0)
        self.assertEqual(cosine_similarity(np.zeros(3), vec2), 0.0)
    
    def test_entity_type_detection(self):
        """Test entity type detection using new schema-driven approach"""
        # Person detection
        self.assertEqual(get_entity_type("Mr. John Smith"), "Person")
        self.assertEqual(get_entity_type("Dr. Jane Doe"), "Person")
        self.assertEqual(get_entity_type("Moses"), "Person")  # Should work with Biblical names
        
        # Organization detection
        self.assertEqual(get_entity_type("Acme Corp."), "Organization")
        self.assertEqual(get_entity_type("Tesla Inc."), "Organization")
        
        # Location detection
        self.assertEqual(get_entity_type("New York City"), "Location")
        self.assertEqual(get_entity_type("Pacific Ocean"), "Location")
        self.assertEqual(get_entity_type("Jerusalem"), "Location")  # Biblical location
        
        # Default type
        self.assertEqual(get_entity_type("Something else"), "Entity")
        
        # Test with context
        self.assertEqual(get_entity_type("Moses", context="Moses said to the people"), "Person")
        self.assertEqual(get_entity_type("Red Sea", context="crossed the Red Sea"), "Location")
    
    def test_normalize_dde_value(self):
        """Test DDE value normalization"""
        # Empty list should return 0
        self.assertEqual(normalize_dde_value([]), 0.0)
        
        # All zeros should return 0
        self.assertEqual(normalize_dde_value([0, 0, 0]), 0.0)
        
        # Sum of values, capped at 1.0
        self.assertEqual(normalize_dde_value([0.2, 0.3]), 0.5)
        self.assertEqual(normalize_dde_value([0.5, 0.6, 0.7]), 1.0)  # Capped at 1.0
    
    def test_heuristic_score(self):
        """Test heuristic scoring function"""
        query_emb = np.array([1, 0, 0])
        triple_emb = np.array([1, 0, 0])
        dde_value = [0.5]
        
        # Perfect similarity, mid-range DDE
        score = heuristic_score(query_emb, triple_emb, dde_value)
        self.assertAlmostEqual(score, 0.7 * 1.0 + 0.3 * 0.5)
        
        # No similarity, high DDE
        triple_emb = np.array([0, 1, 0])
        dde_value = [1.0]
        score = heuristic_score(query_emb, triple_emb, dde_value)
        self.assertAlmostEqual(score, 0.7 * 0.0 + 0.3 * 1.0)
    
    def test_get_score_for_triple(self):
        """Test getting score for triple from scored triples list"""
        triple1 = Triple(id="rel1", head_id="", head_name="", relation_id="", relation_name="", 
                        tail_id="", tail_name="", properties={})
        triple2 = Triple(id="rel2", head_id="", head_name="", relation_id="", relation_name="", 
                        tail_id="", tail_name="", properties={})
        
        scored_triples = [(0.9, triple1), (0.8, triple2)]
        
        # Should find the correct score
        self.assertEqual(get_score_for_triple("rel1", scored_triples), 0.9)
        self.assertEqual(get_score_for_triple("rel2", scored_triples), 0.8)
        
        # Should return 0 for unknown triple
        self.assertEqual(get_score_for_triple("unknown", scored_triples), 0.0)

    def test_triples_to_graph_data(self):
        """Test conversion of triples to graph visualization data"""
        triples = [
            Triple(
                id="rel1",
                head_id="ent1",
                head_name="Elon Musk",
                relation_id="rel1",
                relation_name="founded",
                tail_id="ent2",
                tail_name="Tesla Inc.",
                properties={},
                relevance_score=0.9
            )
        ]
        
        graph_data = triples_to_graph_data(triples, query_entities=["ent1"])
        
        # Check nodes
        self.assertEqual(len(graph_data.nodes), 2)  # Head and tail
        self.assertTrue(any(node.id == "ent1" for node in graph_data.nodes))
        self.assertTrue(any(node.id == "ent2" for node in graph_data.nodes))
        
        # Check links
        self.assertEqual(len(graph_data.links), 1)
        self.assertEqual(graph_data.links[0].source, "ent1")
        self.assertEqual(graph_data.links[0].target, "ent2")
        self.assertEqual(graph_data.links[0].relation_id, "rel1")
        self.assertEqual(graph_data.links[0].relation_name, "founded")


if __name__ == '__main__':
    unittest.main()