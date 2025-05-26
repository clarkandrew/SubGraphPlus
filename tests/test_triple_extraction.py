"""
Tests for the centralized triple extraction module
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app.utils.triple_extraction import (
    extract_triplets_from_rebel,
    clean_and_validate_triple,
    add_entity_types_to_triple,
    process_rebel_output,
    normalize_relation,
    Triple
)

class TestTripleExtraction(unittest.TestCase):
    """Test the centralized triple extraction functionality"""
    
    def test_extract_triplets_from_rebel(self):
        """Test REBEL output parsing"""
        # Test single triplet
        rebel_output = "<s><triplet> Jesus <subj> Bethlehem <obj> place of birth</s>"
        triplets = extract_triplets_from_rebel(rebel_output)
        
        self.assertEqual(len(triplets), 1)
        self.assertEqual(triplets[0]['head'], 'Jesus')
        self.assertEqual(triplets[0]['relation'], 'place of birth')
        self.assertEqual(triplets[0]['tail'], 'Bethlehem')
    
    def test_extract_multiple_triplets(self):
        """Test multiple triplets in one output"""
        rebel_output = "<s><triplet> Moses <subj> Israelites <obj> ethnic group <triplet> Moses <subj> Egypt <obj> place of birth</s>"
        triplets = extract_triplets_from_rebel(rebel_output)
        
        self.assertEqual(len(triplets), 2)
        self.assertEqual(triplets[0]['head'], 'Moses')
        self.assertEqual(triplets[0]['relation'], 'ethnic group')
        self.assertEqual(triplets[0]['tail'], 'Israelites')
        
        self.assertEqual(triplets[1]['head'], 'Moses')
        self.assertEqual(triplets[1]['relation'], 'place of birth')
        self.assertEqual(triplets[1]['tail'], 'Egypt')
    
    def test_clean_and_validate_triple(self):
        """Test triple cleaning and validation"""
        # Valid triple
        valid_triple = {'head': 'Jesus', 'relation': 'born in', 'tail': 'Bethlehem'}
        result = clean_and_validate_triple(valid_triple)
        self.assertIsNotNone(result)
        self.assertEqual(result['head'], 'Jesus')
        
        # Invalid triple - missing field
        invalid_triple = {'head': 'Jesus', 'tail': 'Bethlehem'}
        result = clean_and_validate_triple(invalid_triple)
        self.assertIsNone(result)
        
        # Invalid triple - empty components
        empty_triple = {'head': '', 'relation': 'born in', 'tail': 'Bethlehem'}
        result = clean_and_validate_triple(empty_triple)
        self.assertIsNone(result)
        
        # Invalid triple - too long
        long_triple = {
            'head': 'A very long entity name that exceeds reasonable limits for entity names',
            'relation': 'born in',
            'tail': 'Bethlehem'
        }
        result = clean_and_validate_triple(long_triple)
        self.assertIsNone(result)
    
    @patch('app.entity_typing.get_entity_type')
    def test_add_entity_types_to_triple(self, mock_get_entity_type):
        """Test adding entity types to triples"""
        mock_get_entity_type.side_effect = lambda entity, context=None: {
            'Jesus': 'Person',
            'Bethlehem': 'Location'
        }.get(entity, 'Entity')
        
        triple = {'head': 'Jesus', 'relation': 'born in', 'tail': 'Bethlehem'}
        result = add_entity_types_to_triple(triple)
        
        self.assertEqual(result['head_type'], 'Person')
        self.assertEqual(result['tail_type'], 'Location')
        
        # Verify context was passed
        mock_get_entity_type.assert_any_call('Jesus', context='Subject of born in')
        mock_get_entity_type.assert_any_call('Bethlehem', context='Object of born in')
    
    @patch('app.entity_typing.get_entity_type')
    def test_process_rebel_output(self, mock_get_entity_type):
        """Test complete REBEL output processing pipeline"""
        mock_get_entity_type.side_effect = lambda entity, context=None: {
            'Jesus': 'Person',
            'Bethlehem': 'Location'
        }.get(entity, 'Entity')
        
        rebel_output = "<s><triplet> Jesus <subj> Bethlehem <obj> place of birth</s>"
        triples = process_rebel_output(rebel_output)
        
        self.assertEqual(len(triples), 1)
        triple = triples[0]
        
        self.assertIsInstance(triple, Triple)
        self.assertEqual(triple.head, 'Jesus')
        self.assertEqual(triple.relation, 'place of birth')
        self.assertEqual(triple.tail, 'Bethlehem')
        self.assertEqual(triple.head_type, 'Person')
        self.assertEqual(triple.tail_type, 'Location')
        self.assertEqual(triple.confidence, 1.0)
        self.assertEqual(triple.source, 'rebel_extraction')
    
    def test_normalize_relation(self):
        """Test relation normalization"""
        # Basic normalization
        self.assertEqual(normalize_relation('place of birth'), 'born_in')
        self.assertEqual(normalize_relation('Place Of Birth'), 'born_in')
        
        # Special character removal
        self.assertEqual(normalize_relation('member-of'), 'belongs_to')
        self.assertEqual(normalize_relation('part_of'), 'belongs_to')
        
        # Unknown relation
        self.assertEqual(normalize_relation('custom relation'), 'custom_relation')
    
    def test_triple_dataclass(self):
        """Test Triple dataclass functionality"""
        triple = Triple(
            head='Jesus',
            relation='born_in',
            tail='Bethlehem',
            head_type='Person',
            tail_type='Location'
        )
        
        self.assertEqual(triple.head, 'Jesus')
        self.assertEqual(triple.relation, 'born_in')
        self.assertEqual(triple.tail, 'Bethlehem')
        self.assertEqual(triple.head_type, 'Person')
        self.assertEqual(triple.tail_type, 'Location')
        self.assertEqual(triple.confidence, 1.0)  # Default value
        self.assertEqual(triple.source, 'rebel_extraction')  # Default value
    
    def test_empty_rebel_output(self):
        """Test handling of empty or malformed REBEL output"""
        # Empty output
        triplets = extract_triplets_from_rebel("")
        self.assertEqual(len(triplets), 0)
        
        # Malformed output
        triplets = extract_triplets_from_rebel("random text without triplet markers")
        self.assertEqual(len(triplets), 0)
        
        # Incomplete triplet
        triplets = extract_triplets_from_rebel("<s><triplet> Jesus <subj> Bethlehem")
        self.assertEqual(len(triplets), 0)
    
    @patch('requests.post')
    @patch('app.entity_typing.get_entity_type')
    def test_batch_process_texts(self, mock_get_entity_type, mock_post):
        """Test batch processing of texts through IE service"""
        # Mock entity typing
        mock_get_entity_type.side_effect = lambda entity, context=None: {
            'Jesus': 'Person',
            'Bethlehem': 'Location'
        }.get(entity, 'Entity')
        
        # Mock IE service response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'triples': [
                {'head': 'Jesus', 'relation': 'place of birth', 'tail': 'Bethlehem', 'confidence': 1.0}
            ]
        }
        mock_post.return_value = mock_response
        
        from app.utils.triple_extraction import batch_process_texts
        
        texts = ["Jesus was born in Bethlehem"]
        triples = batch_process_texts(texts, "http://localhost:8003")
        
        self.assertEqual(len(triples), 1)
        triple = triples[0]
        
        self.assertEqual(triple.head, 'Jesus')
        self.assertEqual(triple.relation, 'born_in')  # Normalized
        self.assertEqual(triple.tail, 'Bethlehem')
        self.assertEqual(triple.head_type, 'Person')
        self.assertEqual(triple.tail_type, 'Location')
        self.assertEqual(triple.source, 'rebel_ie_service')
        
        # Verify API call
        mock_post.assert_called_once_with(
            "http://localhost:8003/extract",
            json={"text": "Jesus was born in Bethlehem", "max_length": 256, "num_beams": 3},
            timeout=30
        )

if __name__ == '__main__':
    unittest.main() 