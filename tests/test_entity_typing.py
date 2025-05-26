"""
Tests for the production-ready entity typing service with OntoNotes-5 NER
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Set testing environment to disable database connections
os.environ['TESTING'] = '1'
os.environ['DISABLE_MODELS'] = '1'

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app.entity_typing import (
    detect_entity_type,
    batch_detect_entity_types,
    predict_type_with_ontonotes,
    get_supported_types,
    ONTONOTES_TO_KG
)

class TestEntityTypingOntoNotes(unittest.TestCase):
    """Test the OntoNotes-5 entity typing functionality"""
    
    def test_ontonotes_mapping(self):
        """Test that OntoNotes labels map correctly to KG buckets"""
        # Test key mappings from the spec
        self.assertEqual(ONTONOTES_TO_KG["PERSON"], "Person")
        self.assertEqual(ONTONOTES_TO_KG["GPE"], "Location")
        self.assertEqual(ONTONOTES_TO_KG["ORG"], "Organization")
        self.assertEqual(ONTONOTES_TO_KG["DATE"], "Date")
        self.assertEqual(ONTONOTES_TO_KG["CARDINAL"], "Number")
        self.assertEqual(ONTONOTES_TO_KG["_DEFAULT"], "Entity")
    
    def test_get_supported_types(self):
        """Test getting supported entity types from OntoNotes mapping"""
        types = get_supported_types()
        # Should include all unique values from ONTONOTES_TO_KG
        expected_types = set(ONTONOTES_TO_KG.values())
        self.assertEqual(set(types), expected_types)
    
    def test_empty_input_handling(self):
        """Test handling of empty or None input"""
        self.assertEqual(detect_entity_type(""), "Entity")
        self.assertEqual(detect_entity_type("   "), "Entity")
        self.assertEqual(detect_entity_type(None), "Entity")
    
    @patch('app.entity_typing.schema_type_lookup')
    @patch('app.entity_typing.get_ontonotes_ner_pipeline')
    def test_onto_fallback(self, mock_get_pipeline, mock_schema):
        """Test OntoNotes NER fallback when schema lookup misses"""
        # Mock schema miss
        mock_schema.return_value = None
        
        # Mock OntoNotes NER success
        mock_model = MagicMock()
        mock_model.predict.return_value = [[{"type": "GPE"}]]
        mock_get_pipeline.return_value = mock_model
        
        result = detect_entity_type("NewYork")
        self.assertEqual(result, "Location")  # GPE maps to Location
        
        mock_schema.assert_called_once_with("NewYork")
        mock_model.predict.assert_called_once_with(["NewYork"])
    
    @patch('app.entity_typing.schema_type_lookup')
    @patch('app.entity_typing.get_ontonotes_ner_pipeline')
    @patch('app.entity_typing.get_spacy_ner')
    def test_spacy_fallback(self, mock_get_spacy, mock_get_pipeline, mock_schema):
        """Test spaCy fallback when OntoNotes returns empty"""
        # Mock schema miss
        mock_schema.return_value = None
        
        # Mock OntoNotes failure
        mock_get_pipeline.return_value = None
        
        # Mock spaCy success
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_ent = MagicMock()
        mock_ent.label_ = "GPE"
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc
        mock_get_spacy.return_value = mock_nlp
        
        result = detect_entity_type("London")
        self.assertEqual(result, "Location")  # GPE maps to Location
        
        mock_nlp.assert_called_once_with("London")
    
    @patch('app.entity_typing.schema_type_lookup')
    @patch('app.entity_typing.get_ontonotes_ner_pipeline')
    def test_batch_cache(self, mock_get_pipeline, mock_schema):
        """Test that batch processing with identical names triggers only one NER call"""
        # Mock schema miss for all
        mock_schema.return_value = None
        
        # Mock OntoNotes NER
        mock_model = MagicMock()
        mock_model.predict.return_value = [
            [{"type": "PERSON"}],
            [{"type": "PERSON"}]  # Same entity, should be cached
        ]
        mock_get_pipeline.return_value = mock_model
        
        # Call with duplicate names
        result = batch_detect_entity_types(["John", "John"])
        
        expected = {"John": "Person", "John": "Person"}
        self.assertEqual(result, expected)
        
        # Should only call predict once for the batch, not per duplicate
        self.assertEqual(mock_model.predict.call_count, 1)
    
    @patch('app.entity_typing.schema_type_lookup')
    def test_caching_behavior(self, mock_schema):
        """Test that caching works correctly"""
        mock_schema.return_value = "Person"
        
        # Clear cache first
        detect_entity_type.cache_clear()
        
        # First call
        result1 = detect_entity_type("TestEntity")
        # Second call should use cache
        result2 = detect_entity_type("TestEntity")
        
        self.assertEqual(result1, "Person")
        self.assertEqual(result2, "Person")
        
        # Schema lookup should only be called once due to caching
        mock_schema.assert_called_once_with("TestEntity")
    
    def test_predict_type_with_ontonotes_mapping(self):
        """Test the OntoNotes prediction mapping logic"""
        with patch('app.entity_typing.get_ontonotes_ner_pipeline') as mock_get_pipeline:
            mock_model = MagicMock()
            
            # Test different OntoNotes labels
            test_cases = [
                ([{"type": "PERSON"}], "Person"),
                ([{"type": "GPE"}], "Location"),
                ([{"type": "ORG"}], "Organization"),
                ([{"type": "DATE"}], "Date"),
                ([{"type": "UNKNOWN_TYPE"}], "Entity"),  # Should default
                ([], "Entity"),  # Empty result should default
            ]
            
            mock_get_pipeline.return_value = mock_model
            
            for ner_result, expected_type in test_cases:
                mock_model.predict.return_value = [ner_result]
                result = predict_type_with_ontonotes("test")
                self.assertEqual(result, expected_type, 
                               f"Failed for NER result {ner_result}")
    
    def test_batch_detect_entity_types_deduplication(self):
        """Test that batch processing deduplicates entity names"""
        with patch('app.entity_typing.schema_type_lookup') as mock_schema:
            with patch('app.entity_typing.get_ontonotes_ner_pipeline') as mock_get_pipeline:
                # Mock schema miss for all
                mock_schema.return_value = None
                
                # Mock OntoNotes NER
                mock_model = MagicMock()
                mock_model.predict.return_value = [
                    [{"type": "PERSON"}],  # For "John"
                    [{"type": "GPE"}]      # For "London"
                ]
                mock_get_pipeline.return_value = mock_model
                
                # Call with duplicates
                result = batch_detect_entity_types(["John", "London", "John", "London"])
                
                # Should return dictionary with unique keys only
                self.assertEqual(len(result), 2)  # Only unique entities
                self.assertEqual(result["John"], "Person")
                self.assertEqual(result["London"], "Location")
                
                # Should only call predict once with unique entities
                mock_model.predict.assert_called_once()
                call_args = mock_model.predict.call_args[0][0]
                self.assertEqual(set(call_args), {"John", "London"})  # Only unique entities

if __name__ == '__main__':
    unittest.main() 