"""
Tests for the production-ready entity typing service
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app.entity_typing import (
    detect_entity_type,
    batch_detect_entity_types,
    schema_type_lookup,
    predict_type_with_ner,
    update_entity_type_in_schema,
    get_supported_types,
    get_type_statistics
)

class TestEntityTyping(unittest.TestCase):
    """Test the production-ready entity typing functionality"""
    
    @patch('app.entity_typing.neo4j_db')
    def test_schema_type_lookup_hit(self, mock_neo4j_db):
        """Test successful schema lookup"""
        # Mock Neo4j response
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__.return_value = "Person"
        mock_record.__bool__.return_value = True
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_neo4j_db.get_session.return_value.__enter__.return_value = mock_session
        
        result = schema_type_lookup("Jesus")
        self.assertEqual(result, "Person")
        
        # Verify the query was called
        mock_session.run.assert_called_with(
            "MATCH (e:Entity {name: $mention}) RETURN e.type as type LIMIT 1",
            {"mention": "Jesus"}
        )
    
    @patch('app.entity_typing.neo4j_db')
    def test_schema_type_lookup_miss(self, mock_neo4j_db):
        """Test schema lookup miss"""
        # Mock Neo4j response - no results
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result
        mock_neo4j_db.get_session.return_value.__enter__.return_value = mock_session
        
        result = schema_type_lookup("UnknownEntity")
        self.assertIsNone(result)
    
    @patch('app.entity_typing.get_ner_pipeline')
    def test_predict_type_with_ner_transformers(self, mock_get_pipeline):
        """Test NER prediction using transformers"""
        # Mock transformers NER pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"entity_group": "PER", "score": 0.99}]
        mock_get_pipeline.return_value = mock_pipeline
        
        result = predict_type_with_ner("John Smith")
        self.assertEqual(result, "Person")
        
        mock_pipeline.assert_called_once_with("John Smith")
    
    @patch('app.entity_typing.get_ner_pipeline')
    @patch('app.entity_typing.get_spacy_ner')
    def test_predict_type_with_ner_spacy_fallback(self, mock_get_spacy, mock_get_pipeline):
        """Test NER prediction falling back to spaCy"""
        # Mock transformers failure
        mock_get_pipeline.return_value = None
        
        # Mock spaCy success
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_ent = MagicMock()
        mock_ent.label_ = "PERSON"
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc
        mock_get_spacy.return_value = mock_nlp
        
        result = predict_type_with_ner("Jane Doe")
        self.assertEqual(result, "Person")
        
        mock_nlp.assert_called_once_with("Jane Doe")
    
    @patch('app.entity_typing.schema_type_lookup')
    @patch('app.entity_typing.predict_type_with_ner')
    def test_detect_entity_type_schema_first(self, mock_ner, mock_schema):
        """Test that schema lookup is tried first"""
        mock_schema.return_value = "Person"
        mock_ner.return_value = "Organization"  # Should not be called
        
        result = detect_entity_type("Jesus")
        self.assertEqual(result, "Person")
        
        mock_schema.assert_called_once_with("Jesus")
        mock_ner.assert_not_called()
    
    @patch('app.entity_typing.schema_type_lookup')
    @patch('app.entity_typing.predict_type_with_ner')
    def test_detect_entity_type_ner_fallback(self, mock_ner, mock_schema):
        """Test NER fallback when schema lookup misses"""
        mock_schema.return_value = None
        mock_ner.return_value = "Location"
        
        result = detect_entity_type("NewYork")
        self.assertEqual(result, "Location")
        
        mock_schema.assert_called_once_with("NewYork")
        mock_ner.assert_called_once_with("NewYork")
    
    @patch('app.entity_typing.neo4j_db')
    def test_batch_detect_entity_types(self, mock_neo4j_db):
        """Test batch entity type detection"""
        # Mock Neo4j batch response
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_records = [
            {"mention": "Jesus", "type": "Person"},
            {"mention": "Jerusalem", "type": "Location"}
        ]
        mock_result.__iter__.return_value = iter(mock_records)
        mock_session.run.return_value = mock_result
        mock_neo4j_db.get_session.return_value.__enter__.return_value = mock_session
        
        mentions = ["Jesus", "Jerusalem", "UnknownEntity"]
        
        with patch('app.entity_typing.predict_type_with_ner') as mock_ner:
            mock_ner.return_value = "Entity"
            
            result = batch_detect_entity_types(mentions)
            
            expected = {
                "Jesus": "Person",
                "Jerusalem": "Location", 
                "UnknownEntity": "Entity"
            }
            self.assertEqual(result, expected)
    
    @patch('app.entity_typing.neo4j_db')
    def test_update_entity_type_in_schema(self, mock_neo4j_db):
        """Test updating entity type in schema"""
        # Mock successful update
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_record.__bool__.return_value = True
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_neo4j_db.get_session.return_value.__enter__.return_value = mock_session
        
        result = update_entity_type_in_schema("NewEntity", "Person")
        self.assertTrue(result)
        
        # Verify the update query
        mock_session.run.assert_called_with(
            "MERGE (e:Entity {name: $mention}) SET e.type = $type RETURN e.name as name",
            {"mention": "NewEntity", "type": "Person"}
        )
    
    def test_get_supported_types(self):
        """Test getting supported entity types"""
        types = get_supported_types()
        expected_types = ["Person", "Location", "Organization", "Event", "Concept", "Entity"]
        self.assertEqual(types, expected_types)
    
    @patch('app.entity_typing.neo4j_db')
    def test_get_type_statistics(self, mock_neo4j_db):
        """Test getting type statistics"""
        # Mock Neo4j response
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_records = [
            {"type": "Person", "count": 100},
            {"type": "Location", "count": 50},
            {"type": "Organization", "count": 25}
        ]
        mock_result.__iter__.return_value = iter(mock_records)
        mock_session.run.return_value = mock_result
        mock_neo4j_db.get_session.return_value.__enter__.return_value = mock_session
        
        result = get_type_statistics()
        expected = {
            "Person": 100,
            "Location": 50,
            "Organization": 25
        }
        self.assertEqual(result, expected)
    
    def test_empty_input_handling(self):
        """Test handling of empty or None input"""
        self.assertEqual(detect_entity_type(""), "Entity")
        self.assertEqual(detect_entity_type("   "), "Entity")
        self.assertEqual(detect_entity_type(None), "Entity")
    
    @patch('app.entity_typing.schema_type_lookup')
    def test_caching_behavior(self, mock_schema):
        """Test that caching works correctly"""
        mock_schema.return_value = "Person"
        
        # First call
        result1 = detect_entity_type("TestEntity")
        # Second call should use cache
        result2 = detect_entity_type("TestEntity")
        
        self.assertEqual(result1, "Person")
        self.assertEqual(result2, "Person")
        
        # Schema lookup should only be called once due to caching
        mock_schema.assert_called_once_with("TestEntity")

if __name__ == '__main__':
    unittest.main() 