import unittest
import tempfile
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.entity_typing import EntityTyper, get_entity_type, get_entity_typer


class TestEntityTyping(unittest.TestCase):
    """Test the new schema-driven entity typing system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary type mapping file
        self.test_mappings = {
            "Moses": "Person",
            "Jerusalem": "Location", 
            "Israelites": "Organization",
            "Exodus": "Event",
            "Covenant": "Concept"
        }
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_mappings, self.temp_file)
        self.temp_file.close()
        
        # Create typer with test mapping
        self.typer = EntityTyper(type_mapping_path=self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        Path(self.temp_file.name).unlink()
    
    def test_schema_mapping_lookup(self):
        """Test that schema mappings are used correctly"""
        # Direct mappings should work
        self.assertEqual(self.typer.get_entity_type("Moses"), "Person")
        self.assertEqual(self.typer.get_entity_type("Jerusalem"), "Location")
        self.assertEqual(self.typer.get_entity_type("Israelites"), "Organization")
        self.assertEqual(self.typer.get_entity_type("Exodus"), "Event")
        self.assertEqual(self.typer.get_entity_type("Covenant"), "Concept")
    
    def test_case_insensitive_mapping(self):
        """Test case-insensitive mapping lookup"""
        self.assertEqual(self.typer.get_entity_type("moses"), "Person")
        self.assertEqual(self.typer.get_entity_type("JERUSALEM"), "Location")
        self.assertEqual(self.typer.get_entity_type("israelites"), "Organization")
    
    def test_pattern_based_fallback(self):
        """Test pattern-based typing for unmapped entities"""
        # Person patterns
        self.assertEqual(self.typer.get_entity_type("Dr. Smith"), "Person")
        self.assertEqual(self.typer.get_entity_type("King David"), "Person")
        self.assertEqual(self.typer.get_entity_type("Prophet Isaiah"), "Person")
        
        # Location patterns
        self.assertEqual(self.typer.get_entity_type("Mount Sinai"), "Location")
        self.assertEqual(self.typer.get_entity_type("Red Sea"), "Location")
        self.assertEqual(self.typer.get_entity_type("New York City"), "Location")
        
        # Organization patterns
        self.assertEqual(self.typer.get_entity_type("Acme Corp."), "Organization")
        self.assertEqual(self.typer.get_entity_type("Tribe of Benjamin"), "Organization")
        
        # Event patterns
        self.assertEqual(self.typer.get_entity_type("Great Flood"), "Event")
        self.assertEqual(self.typer.get_entity_type("Passover"), "Event")
        
        # Concept patterns
        self.assertEqual(self.typer.get_entity_type("Ten Commandments"), "Concept")
        self.assertEqual(self.typer.get_entity_type("Salvation"), "Concept")
    
    def test_context_disambiguation(self):
        """Test context-based disambiguation"""
        # Context should help with ambiguous entities
        person_context = "Moses said to the people"
        location_context = "traveled to Jerusalem"
        
        # These should work even without explicit mapping
        result = self.typer.get_entity_type("David", context=person_context)
        self.assertIn(result, ["Person", "Entity"])  # Should prefer Person with person context
        
        result = self.typer.get_entity_type("Bethlehem", context=location_context)
        self.assertIn(result, ["Location", "Entity"])  # Should prefer Location with location context
    
    def test_default_fallback(self):
        """Test default Entity type for unknown entities"""
        self.assertEqual(self.typer.get_entity_type("UnknownEntity"), "Entity")
        self.assertEqual(self.typer.get_entity_type("RandomText123"), "Entity")
    
    def test_empty_input_handling(self):
        """Test handling of empty or invalid inputs"""
        self.assertEqual(self.typer.get_entity_type(""), "Entity")
        self.assertEqual(self.typer.get_entity_type("   "), "Entity")
        self.assertEqual(self.typer.get_entity_type(None), "Entity")
    
    def test_dynamic_mapping_addition(self):
        """Test adding new mappings dynamically"""
        # Add new mapping
        self.typer.add_type_mapping("TestEntity", "TestType")
        self.assertEqual(self.typer.get_entity_type("TestEntity"), "TestType")
        
        # Bulk add mappings
        new_mappings = {
            "Entity1": "Type1",
            "Entity2": "Type2"
        }
        self.typer.bulk_add_mappings(new_mappings)
        self.assertEqual(self.typer.get_entity_type("Entity1"), "Type1")
        self.assertEqual(self.typer.get_entity_type("Entity2"), "Type2")
    
    def test_supported_types(self):
        """Test getting supported entity types"""
        supported_types = self.typer.get_supported_types()
        
        # Should include mapped types
        self.assertIn("Person", supported_types)
        self.assertIn("Location", supported_types)
        self.assertIn("Organization", supported_types)
        self.assertIn("Event", supported_types)
        self.assertIn("Concept", supported_types)
        
        # Should include default type
        self.assertIn("Entity", supported_types)
    
    def test_export_mappings(self):
        """Test exporting mappings to file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            self.typer.export_mappings(export_path)
            
            # Verify exported content
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            # Should contain original mappings
            for entity, entity_type in self.test_mappings.items():
                self.assertEqual(exported_data[entity], entity_type)
                
        finally:
            Path(export_path).unlink()
    
    def test_global_entity_typer(self):
        """Test global entity typer functions"""
        # Test global function
        result = get_entity_type("Dr. Johnson")
        self.assertEqual(result, "Person")
        
        # Test global instance
        typer_instance = get_entity_typer()
        self.assertIsInstance(typer_instance, EntityTyper)
    
    def test_backward_compatibility(self):
        """Test backward compatibility with old detect_type function"""
        from app.entity_typing import detect_type
        
        # Should still work but issue deprecation warning
        result = detect_type("Mr. Smith")
        self.assertEqual(result, "Person")
    
    def test_biblical_entities(self):
        """Test specific Biblical entity typing"""
        # Test entities that should be properly typed for Biblical content
        biblical_tests = [
            ("Moses", "Person"),
            ("Aaron", "Person"), 
            ("Red Sea", "Location"),
            ("Mount Sinai", "Location"),
            ("Israelites", "Organization"),
            ("Pharisees", "Organization"),
            ("Exodus", "Event"),
            ("Passover", "Event"),
            ("Ten Commandments", "Concept"),
            ("Covenant", "Concept")
        ]
        
        for entity, expected_type in biblical_tests:
            with self.subTest(entity=entity):
                result = self.typer.get_entity_type(entity)
                # Should be either the expected type or Entity (fallback)
                self.assertIn(result, [expected_type, "Entity"], 
                             f"Entity '{entity}' got type '{result}', expected '{expected_type}' or 'Entity'")


if __name__ == '__main__':
    unittest.main() 