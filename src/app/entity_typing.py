"""
Schema-driven Entity Typing Service
Replaces naive string heuristics with proper knowledge graph schema typing
"""

import json
import logging
from typing import Dict, Optional, Set, List
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

class EntityTyper:
    """
    Production-grade entity typing using schema mappings and external knowledge bases
    """
    
    def __init__(self, type_mapping_path: Optional[str] = None):
        """
        Initialize entity typer
        
        Args:
            type_mapping_path: Path to external entity->type mapping file (CSV/JSON)
        """
        self.type_mapping = {}
        self.load_type_mapping(type_mapping_path)
        
        # Common entity patterns for fallback (much more comprehensive than original)
        self.type_patterns = {
            "Person": {
                "prefixes": ["mr.", "mrs.", "ms.", "dr.", "prof.", "professor", "sir", "lady", "lord", 
                           "rev.", "father", "mother", "brother", "sister", "saint", "st."],
                "suffixes": ["jr.", "sr.", "ii", "iii", "iv", "phd", "md", "esq."],
                "keywords": ["prophet", "apostle", "disciple", "king", "queen", "prince", "princess",
                           "emperor", "pharaoh", "caesar", "pope", "bishop", "priest", "rabbi"]
            },
            "Location": {
                "prefixes": ["mount", "mt.", "lake", "river", "sea", "ocean", "desert", "valley",
                           "city", "town", "village", "province", "state", "country", "nation"],
                "suffixes": ["city", "town", "village", "county", "state", "province", "country",
                           "island", "mountain", "river", "sea", "ocean", "desert", "valley",
                           "gate", "wall", "temple", "palace", "house"],
                "keywords": ["jerusalem", "babylon", "egypt", "israel", "judah", "galilee", "samaria"]
            },
            "Organization": {
                "prefixes": ["tribe", "house", "family", "clan", "people"],
                "suffixes": ["inc.", "corp.", "llc", "ltd.", "corporation", "company", "org",
                           "ites", "ans", "ians"],  # Biblical tribal suffixes
                "keywords": ["temple", "synagogue", "church", "council", "assembly", "congregation"]
            },
            "Event": {
                "keywords": ["exodus", "flood", "creation", "crucifixion", "resurrection", "passover",
                           "sabbath", "festival", "feast", "war", "battle", "siege", "covenant"]
            },
            "Concept": {
                "keywords": ["law", "commandment", "covenant", "promise", "prophecy", "miracle",
                           "parable", "blessing", "curse", "sin", "righteousness", "salvation"]
            }
        }
    
    def load_type_mapping(self, mapping_path: Optional[str]):
        """
        Load external entity type mapping from file
        
        Args:
            mapping_path: Path to mapping file (JSON or CSV format)
        """
        if not mapping_path:
            # Try default locations
            for default_path in ["config/entity_types.json", "data/entity_types.json"]:
                if Path(default_path).exists():
                    mapping_path = default_path
                    break
        
        if mapping_path and Path(mapping_path).exists():
            try:
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    if mapping_path.endswith('.json'):
                        self.type_mapping = json.load(f)
                    elif mapping_path.endswith('.csv'):
                        import csv
                        reader = csv.DictReader(f)
                        self.type_mapping = {row['entity']: row['type'] for row in reader}
                
                logger.info(f"Loaded {len(self.type_mapping)} entity type mappings from {mapping_path}")
            except Exception as e:
                logger.warning(f"Failed to load type mapping from {mapping_path}: {e}")
        else:
            logger.info("No external type mapping file found, using pattern-based fallback only")
    
    @lru_cache(maxsize=10000)
    def get_entity_type(self, entity_name: str, context: Optional[str] = None) -> str:
        """
        Get entity type using schema-driven approach
        
        Args:
            entity_name: Name of the entity
            context: Optional context for disambiguation
            
        Returns:
            Entity type string
        """
        if not entity_name:
            return "Entity"
        
        # Normalize entity name
        normalized_name = entity_name.strip()
        
        # 1. Check external type mapping first (highest priority)
        if normalized_name in self.type_mapping:
            return self.type_mapping[normalized_name]
        
        # 2. Check case-insensitive mapping
        normalized_lower = normalized_name.lower()
        for mapped_entity, entity_type in self.type_mapping.items():
            if mapped_entity.lower() == normalized_lower:
                return entity_type
        
        # 3. Pattern-based fallback (improved from original heuristics)
        detected_type = self._pattern_based_typing(normalized_name, context)
        
        return detected_type
    
    def _pattern_based_typing(self, entity_name: str, context: Optional[str] = None) -> str:
        """
        Pattern-based entity typing as fallback
        
        Args:
            entity_name: Entity name to type
            context: Optional context for disambiguation
            
        Returns:
            Detected entity type
        """
        name_lower = entity_name.lower()
        words = name_lower.split()
        
        # Score each type based on pattern matches
        type_scores = {}
        
        for entity_type, patterns in self.type_patterns.items():
            score = 0
            
            # Check prefixes
            for prefix in patterns.get("prefixes", []):
                if any(word.startswith(prefix) for word in words):
                    score += 2
            
            # Check suffixes  
            for suffix in patterns.get("suffixes", []):
                if any(word.endswith(suffix) for word in words):
                    score += 2
            
            # Check keywords
            for keyword in patterns.get("keywords", []):
                if keyword in name_lower:
                    score += 1
            
            if score > 0:
                type_scores[entity_type] = score
        
        # Use context for disambiguation if available
        if context and type_scores:
            context_lower = context.lower()
            for entity_type in type_scores:
                # Boost scores based on context
                if entity_type == "Person" and any(word in context_lower for word in ["said", "spoke", "told", "went", "came"]):
                    type_scores[entity_type] += 1
                elif entity_type == "Location" and any(word in context_lower for word in ["in", "at", "to", "from", "near"]):
                    type_scores[entity_type] += 1
        
        # Return highest scoring type
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return "Entity"  # Default fallback
    
    def add_type_mapping(self, entity_name: str, entity_type: str):
        """
        Add a new entity type mapping
        
        Args:
            entity_name: Entity name
            entity_type: Entity type
        """
        self.type_mapping[entity_name] = entity_type
        # Clear cache to ensure new mapping is used
        self.get_entity_type.cache_clear()
    
    def bulk_add_mappings(self, mappings: Dict[str, str]):
        """
        Add multiple entity type mappings
        
        Args:
            mappings: Dictionary of entity_name -> entity_type
        """
        self.type_mapping.update(mappings)
        self.get_entity_type.cache_clear()
    
    def get_supported_types(self) -> Set[str]:
        """Get all supported entity types"""
        types = set(self.type_mapping.values())
        types.update(self.type_patterns.keys())
        types.add("Entity")  # Default type
        return types
    
    def export_mappings(self, output_path: str):
        """
        Export current type mappings to file
        
        Args:
            output_path: Path to save mappings
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.type_mapping, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported {len(self.type_mapping)} type mappings to {output_path}")


# Global entity typer instance
_entity_typer = None

def get_entity_typer() -> EntityTyper:
    """Get global entity typer instance"""
    global _entity_typer
    if _entity_typer is None:
        _entity_typer = EntityTyper()
    return _entity_typer

def get_entity_type(entity_name: str, context: Optional[str] = None) -> str:
    """
    Get entity type using schema-driven approach
    
    Args:
        entity_name: Name of the entity
        context: Optional context for disambiguation
        
    Returns:
        Entity type string
    """
    return get_entity_typer().get_entity_type(entity_name, context)

# Backward compatibility function (replaces the old detect_type)
def detect_type(text: str) -> str:
    """
    DEPRECATED: Use get_entity_type() instead
    Maintained for backward compatibility
    """
    logger.warning("detect_type() is deprecated, use get_entity_type() instead")
    return get_entity_type(text) 