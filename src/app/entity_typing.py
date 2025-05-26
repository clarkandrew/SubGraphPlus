"""
Production-Ready Entity Typing Service
Combines schema lookup with NER fallback for robust entity classification
"""

import os
import logging
from functools import lru_cache
from typing import Optional, Dict, List
from transformers import pipeline

from .config import config
from .database import neo4j_db

logger = logging.getLogger(__name__)

# NER label mapping to our KG schema
NER_TO_KG = {
    "PER": "Person",
    "PERSON": "Person", 
    "ORG": "Organization",
    "ORGANIZATION": "Organization",
    "LOC": "Location",
    "LOCATION": "Location", 
    "GPE": "Location",  # Geopolitical entity
    "MISC": "Entity",
    "MISCELLANEOUS": "Entity"
}

@lru_cache(maxsize=1)
def get_ner_pipeline():
    """
    Initialize and cache the NER pipeline
    Uses a production-grade BERT model for entity recognition
    """
    try:
        logger.info("Loading NER pipeline: dbmdz/bert-large-cased-finetuned-conll03-english")
        return pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="first",
            device=-1  # CPU; switch to 0 for GPU
        )
    except Exception as e:
        logger.error(f"Failed to load NER pipeline: {e}")
        logger.warning("Falling back to spaCy NER")
        return None

@lru_cache(maxsize=1)
def get_spacy_ner():
    """
    Fallback to spaCy NER if transformers fails
    """
    try:
        import spacy
        logger.info("Loading spaCy model: en_core_web_sm")
        return spacy.load("en_core_web_sm")
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}")
        return None

def schema_type_lookup(mention: str) -> Optional[str]:
    """
    Look up entity type from existing Neo4j graph
    This is the primary, authoritative source
    
    Args:
        mention: Entity mention to look up
        
    Returns:
        Entity type if found in schema, None otherwise
    """
    try:
        with neo4j_db.get_session() as session:
            result = session.run(
                "MATCH (e:Entity {name: $mention}) RETURN e.type as type LIMIT 1",
                {"mention": mention}
            )
            record = result.single()
            if record and record["type"]:
                logger.debug(f"Schema lookup hit: {mention} -> {record['type']}")
                return record["type"]
            
            # Also try case-insensitive lookup
            result = session.run(
                "MATCH (e:Entity) WHERE toLower(e.name) = toLower($mention) RETURN e.type as type LIMIT 1",
                {"mention": mention}
            )
            record = result.single()
            if record and record["type"]:
                logger.debug(f"Schema lookup hit (case-insensitive): {mention} -> {record['type']}")
                return record["type"]
                
    except Exception as e:
        logger.warning(f"Schema lookup failed for '{mention}': {e}")
    
    return None

def predict_type_with_ner(mention: str) -> str:
    """
    Predict entity type using NER transformer model
    Fallback method when schema lookup misses
    
    Args:
        mention: Entity mention to classify
        
    Returns:
        Predicted entity type
    """
    # Try transformers NER first
    ner_pipeline = get_ner_pipeline()
    if ner_pipeline:
        try:
            results = ner_pipeline(mention)
            if results:
                # Take the highest confidence prediction
                best_result = max(results, key=lambda x: x.get('score', 0))
                label = best_result.get("entity_group", "MISC")
                predicted_type = NER_TO_KG.get(label.upper(), "Entity")
                logger.debug(f"NER prediction: {mention} -> {label} -> {predicted_type}")
                return predicted_type
        except Exception as e:
            logger.warning(f"Transformers NER failed for '{mention}': {e}")
    
    # Fallback to spaCy
    spacy_nlp = get_spacy_ner()
    if spacy_nlp:
        try:
            doc = spacy_nlp(mention)
            if doc.ents:
                # Take the first entity's label
                label = doc.ents[0].label_
                predicted_type = NER_TO_KG.get(label, "Entity")
                logger.debug(f"spaCy prediction: {mention} -> {label} -> {predicted_type}")
                return predicted_type
        except Exception as e:
            logger.warning(f"spaCy NER failed for '{mention}': {e}")
    
    # Ultimate fallback
    logger.debug(f"No NER prediction for '{mention}', defaulting to Entity")
    return "Entity"

@lru_cache(maxsize=4096)
def detect_entity_type(mention: str, context: Optional[str] = None) -> str:
    """
    Main entity typing function with schema-first approach
    
    Args:
        mention: Entity mention to classify
        context: Optional context (for backward compatibility)
        
    Returns:
        Entity type (Person, Location, Organization, Event, Concept, Entity)
    """
    if not mention or not mention.strip():
        return "Entity"
    
    mention = mention.strip()
    
    # 1. Schema lookup (primary)
    schema_type = schema_type_lookup(mention)
    if schema_type:
        return schema_type
    
    # 2. NER fallback (secondary)
    return predict_type_with_ner(mention)

def batch_detect_entity_types(mentions: List[str]) -> Dict[str, str]:
    """
    Batch entity type detection for efficiency
    
    Args:
        mentions: List of entity mentions
        
    Returns:
        Dictionary mapping mentions to their types
    """
    results = {}
    
    # First, batch schema lookups
    schema_hits = {}
    try:
        with neo4j_db.get_session() as session:
            # Batch query for all mentions
            result = session.run(
                "UNWIND $mentions as mention "
                "MATCH (e:Entity {name: mention}) "
                "RETURN mention, e.type as type",
                {"mentions": mentions}
            )
            for record in result:
                schema_hits[record["mention"]] = record["type"]
    except Exception as e:
        logger.warning(f"Batch schema lookup failed: {e}")
    
    # Process each mention
    ner_batch = []
    for mention in mentions:
        if mention in schema_hits:
            results[mention] = schema_hits[mention]
        else:
            ner_batch.append(mention)
    
    # Batch NER for remaining mentions
    if ner_batch:
        ner_pipeline = get_ner_pipeline()
        if ner_pipeline:
            try:
                # Process in smaller batches to avoid memory issues
                batch_size = 32
                for i in range(0, len(ner_batch), batch_size):
                    batch = ner_batch[i:i + batch_size]
                    ner_results = ner_pipeline(batch)
                    
                    for mention, ner_result in zip(batch, ner_results):
                        if ner_result:
                            label = ner_result[0].get("entity_group", "MISC") if ner_result else "MISC"
                            results[mention] = NER_TO_KG.get(label.upper(), "Entity")
                        else:
                            results[mention] = "Entity"
            except Exception as e:
                logger.warning(f"Batch NER failed: {e}")
                # Fallback to individual predictions
                for mention in ner_batch:
                    results[mention] = predict_type_with_ner(mention)
        else:
            # No NER available, default to Entity
            for mention in ner_batch:
                results[mention] = "Entity"
    
    return results

def update_entity_type_in_schema(mention: str, entity_type: str) -> bool:
    """
    Update entity type in the Neo4j schema
    Used when we want to persist NER predictions
    
    Args:
        mention: Entity mention
        entity_type: Predicted type to store
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with neo4j_db.get_session() as session:
            result = session.run(
                "MERGE (e:Entity {name: $mention}) "
                "SET e.type = $type "
                "RETURN e.name as name",
                {"mention": mention, "type": entity_type}
            )
            if result.single():
                logger.info(f"Updated schema: {mention} -> {entity_type}")
                # Clear cache for this mention
                detect_entity_type.cache_clear()
                return True
    except Exception as e:
        logger.error(f"Failed to update schema for '{mention}': {e}")
    
    return False

def get_supported_types() -> List[str]:
    """
    Get list of supported entity types
    
    Returns:
        List of supported entity type strings
    """
    return ["Person", "Location", "Organization", "Event", "Concept", "Entity"]

def get_type_statistics() -> Dict[str, int]:
    """
    Get statistics on entity types in the knowledge graph
    
    Returns:
        Dictionary with type counts
    """
    try:
        with neo4j_db.get_session() as session:
            result = session.run(
                "MATCH (e:Entity) "
                "WHERE e.type IS NOT NULL "
                "RETURN e.type as type, count(*) as count "
                "ORDER BY count DESC"
            )
            return {record["type"]: record["count"] for record in result}
    except Exception as e:
        logger.error(f"Failed to get type statistics: {e}")
        return {}

# Backward compatibility
get_entity_type = detect_entity_type

# Export main functions
__all__ = [
    "detect_entity_type",
    "get_entity_type", 
    "batch_detect_entity_types",
    "update_entity_type_in_schema",
    "get_supported_types",
    "get_type_statistics",
    "schema_type_lookup",
    "predict_type_with_ner"
] 