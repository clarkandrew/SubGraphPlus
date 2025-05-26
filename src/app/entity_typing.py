"""
Production-Ready Entity Typing Service with OntoNotes-5 NER
Schema-first approach with offline NER fallback for robust entity classification
"""

import os
import json
import logging
from functools import lru_cache
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

CONFIG = load_config()

# Load OntoNotes-5 to KG bucket mapping
def load_entity_type_mapping():
    """Load the OntoNotes-5 to KG bucket mapping from config"""
    config_path = Path(__file__).parent.parent.parent / "config" / "entity_types.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load entity type mapping: {e}")
        return {"_DEFAULT": "Entity"}

ONTONOTES_TO_KG = load_entity_type_mapping()

# Get model configurations from config
ONTONOTES_CONFIG = CONFIG.get("models", {}).get("entity_typing", {}).get("ontonotes_ner", {})
SPACY_CONFIG = CONFIG.get("models", {}).get("entity_typing", {}).get("spacy_fallback", {})

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter
    TYPING_SOURCE = Counter(
        "entity_typing_source_total",
        "How entity type was resolved",
        ["source"]
    )
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    logger.info("Prometheus metrics not available")

@lru_cache(maxsize=1)
def get_ontonotes_ner_pipeline():
    """
    Initialize and cache the OntoNotes-5 NER pipeline
    Uses configuration from config.json for model selection and parameters
    """
    try:
        import tner
        
        model_name = ONTONOTES_CONFIG.get("model", "tner/roberta-large-ontonotes5")
        local_path = ONTONOTES_CONFIG.get("local_path", "models/roberta-large-ontonotes5")
        
        logger.info(f"Loading OntoNotes-5 NER pipeline: {model_name}")
        
        # Try to load from local models directory first (offline mode)
        local_model_path = Path(local_path)
        if local_model_path.exists():
            logger.info(f"Loading model from local path: {local_model_path}")
            model = tner.TransformersNER(str(local_model_path))
        else:
            # Fallback to downloading from HuggingFace
            logger.info(f"Loading model from HuggingFace Hub: {model_name}")
            model = tner.TransformersNER(model_name)
        
        return model
    except Exception as e:
        logger.error(f"Failed to load OntoNotes NER pipeline: {e}")
        return None

@lru_cache(maxsize=1)
def get_spacy_ner():
    """
    Fallback to spaCy NER if OntoNotes model fails
    Uses configuration from config.json for model selection
    """
    try:
        import spacy
        
        model_name = SPACY_CONFIG.get("model", "en_core_web_sm")
        logger.info(f"Loading spaCy model: {model_name}")
        return spacy.load(model_name)
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
        from .database import neo4j_db
        
        with neo4j_db.get_session() as session:
            result = session.run(
                "MATCH (e:Entity {name: $mention}) RETURN e.type as type LIMIT 1",
                {"mention": mention}
            )
            record = result.single()
            if record and record["type"]:
                logger.debug(f"Schema lookup hit: {mention} -> {record['type']}")
                if HAS_METRICS:
                    TYPING_SOURCE.labels("schema").inc()
                return record["type"]
            
            # Also try case-insensitive lookup
            result = session.run(
                "MATCH (e:Entity) WHERE toLower(e.name) = toLower($mention) RETURN e.type as type LIMIT 1",
                {"mention": mention}
            )
            record = result.single()
            if record and record["type"]:
                logger.debug(f"Schema lookup hit (case-insensitive): {mention} -> {record['type']}")
                if HAS_METRICS:
                    TYPING_SOURCE.labels("schema").inc()
                return record["type"]
                
    except Exception as e:
        logger.warning(f"Schema lookup failed for '{mention}': {e}")
    
    return None

def predict_type_with_ontonotes(mention: str) -> str:
    """
    Predict entity type using OntoNotes-5 NER model
    Fallback method when schema lookup misses
    
    Args:
        mention: Entity mention to classify
        
    Returns:
        Predicted entity type mapped to KG bucket
    """
    # Try OntoNotes NER first
    ner_model = get_ontonotes_ner_pipeline()
    if ner_model:
        try:
            results = ner_model.predict([mention])
            if results and len(results) > 0 and len(results[0]) > 0:
                # Take the first entity's label
                entity_info = results[0][0]
                if isinstance(entity_info, dict) and 'type' in entity_info:
                    label = entity_info['type']
                elif isinstance(entity_info, dict) and 'entity_group' in entity_info:
                    label = entity_info['entity_group']
                else:
                    # Handle different tner output formats
                    label = str(entity_info).split('-')[-1] if '-' in str(entity_info) else str(entity_info)
                
                predicted_type = ONTONOTES_TO_KG.get(label.upper(), ONTONOTES_TO_KG.get("_DEFAULT", "Entity"))
                logger.debug(f"OntoNotes prediction: {mention} -> {label} -> {predicted_type}")
                if HAS_METRICS:
                    TYPING_SOURCE.labels("ner").inc()
                return predicted_type
        except Exception as e:
            logger.warning(f"OntoNotes NER failed for '{mention}': {e}")
    
    # Fallback to spaCy
    spacy_nlp = get_spacy_ner()
    if spacy_nlp:
        try:
            doc = spacy_nlp(mention)
            if doc.ents:
                # Take the first entity's label
                label = doc.ents[0].label_
                predicted_type = ONTONOTES_TO_KG.get(label, ONTONOTES_TO_KG.get("_DEFAULT", "Entity"))
                logger.debug(f"spaCy prediction: {mention} -> {label} -> {predicted_type}")
                if HAS_METRICS:
                    TYPING_SOURCE.labels("spacy").inc()
                return predicted_type
        except Exception as e:
            logger.warning(f"spaCy NER failed for '{mention}': {e}")
    
    # Ultimate fallback
    logger.debug(f"No NER prediction for '{mention}', defaulting to Entity")
    if HAS_METRICS:
        TYPING_SOURCE.labels("default").inc()
    return ONTONOTES_TO_KG.get("_DEFAULT", "Entity")

@lru_cache(maxsize=None)  # Use config-driven cache size
def detect_entity_type(name: str) -> str:
    """
    Main entity typing function with schema-first approach
    Cache size is configured via config.json
    
    Args:
        name: Entity mention to classify
        
    Returns:
        Entity type (Person, Location, Organization, Event, etc.)
    """
    if not name or not name.strip():
        return ONTONOTES_TO_KG.get("_DEFAULT", "Entity")
    
    name = name.strip()
    
    # 1. Schema lookup (primary)
    schema_type = schema_type_lookup(name)
    if schema_type:
        return schema_type
    
    # 2. OntoNotes NER fallback (secondary)
    return predict_type_with_ontonotes(name)

# Apply config-driven cache size
cache_size = ONTONOTES_CONFIG.get("cache_size", 4096)
detect_entity_type = lru_cache(maxsize=cache_size)(detect_entity_type.__wrapped__)

def batch_detect_entity_types(names: List[str]) -> Dict[str, str]:
    """
    Vectorised entity type detection for efficiency
    Groups unknown mentions into a single NER batch call
    Batch size is configured via config.json
    
    Args:
        names: List of entity mentions
        
    Returns:
        Dictionary mapping mentions to their types
    """
    if not names:
        return {}
    
    results = {}
    
    # First, batch schema lookups
    schema_hits = {}
    try:
        from .database import neo4j_db
        
        with neo4j_db.get_session() as session:
            # Batch query for all mentions
            result = session.run(
                "UNWIND $mentions as mention "
                "MATCH (e:Entity {name: mention}) "
                "RETURN mention, e.type as type",
                {"mentions": names}
            )
            for record in result:
                schema_hits[record["mention"]] = record["type"]
                if HAS_METRICS:
                    TYPING_SOURCE.labels("schema").inc()
    except Exception as e:
        logger.warning(f"Batch schema lookup failed: {e}")
    
    # Collect mentions that need NER
    ner_batch = []
    for name in names:
        if name in schema_hits:
            results[name] = schema_hits[name]
        else:
            ner_batch.append(name)
    
    # Batch OntoNotes NER for remaining mentions
    if ner_batch:
        ner_model = get_ontonotes_ner_pipeline()
        if ner_model:
            try:
                # Use config-driven batch size
                batch_size = ONTONOTES_CONFIG.get("batch_size", 32)
                
                # Process in smaller batches to avoid memory issues
                for i in range(0, len(ner_batch), batch_size):
                    batch = ner_batch[i:i + batch_size]
                    ner_results = ner_model.predict(batch)
                    
                    for mention, ner_result in zip(batch, ner_results):
                        if ner_result and len(ner_result) > 0:
                            entity_info = ner_result[0]
                            if isinstance(entity_info, dict) and 'type' in entity_info:
                                label = entity_info['type']
                            elif isinstance(entity_info, dict) and 'entity_group' in entity_info:
                                label = entity_info['entity_group']
                            else:
                                label = str(entity_info).split('-')[-1] if '-' in str(entity_info) else str(entity_info)
                            
                            results[mention] = ONTONOTES_TO_KG.get(label.upper(), ONTONOTES_TO_KG.get("_DEFAULT", "Entity"))
                            if HAS_METRICS:
                                TYPING_SOURCE.labels("ner").inc()
                        else:
                            results[mention] = ONTONOTES_TO_KG.get("_DEFAULT", "Entity")
                            if HAS_METRICS:
                                TYPING_SOURCE.labels("default").inc()
            except Exception as e:
                logger.warning(f"Batch OntoNotes NER failed: {e}")
                # Fallback to individual predictions
                for mention in ner_batch:
                    results[mention] = predict_type_with_ontonotes(mention)
        else:
            # No OntoNotes available, try spaCy batch or default
            spacy_nlp = get_spacy_ner()
            if spacy_nlp:
                try:
                    for mention in ner_batch:
                        doc = spacy_nlp(mention)
                        if doc.ents:
                            label = doc.ents[0].label_
                            results[mention] = ONTONOTES_TO_KG.get(label, ONTONOTES_TO_KG.get("_DEFAULT", "Entity"))
                            if HAS_METRICS:
                                TYPING_SOURCE.labels("spacy").inc()
                        else:
                            results[mention] = ONTONOTES_TO_KG.get("_DEFAULT", "Entity")
                            if HAS_METRICS:
                                TYPING_SOURCE.labels("default").inc()
                except Exception as e:
                    logger.warning(f"Batch spaCy NER failed: {e}")
                    for mention in ner_batch:
                        results[mention] = ONTONOTES_TO_KG.get("_DEFAULT", "Entity")
                        if HAS_METRICS:
                            TYPING_SOURCE.labels("default").inc()
            else:
                # No NER available, default to Entity
                for mention in ner_batch:
                    results[mention] = ONTONOTES_TO_KG.get("_DEFAULT", "Entity")
                    if HAS_METRICS:
                        TYPING_SOURCE.labels("default").inc()
    
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
        from .database import neo4j_db
        
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
    Get list of supported entity types from OntoNotes mapping
    
    Returns:
        List of supported entity type strings
    """
    return list(set(ONTONOTES_TO_KG.values()))

def get_type_statistics() -> Dict[str, int]:
    """
    Get statistics on entity types in the knowledge graph
    
    Returns:
        Dictionary with type counts
    """
    try:
        from .database import neo4j_db
        
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

def get_model_info() -> Dict[str, any]:
    """
    Get information about loaded models and configuration
    
    Returns:
        Dictionary with model information
    """
    return {
        "ontonotes_config": ONTONOTES_CONFIG,
        "spacy_config": SPACY_CONFIG,
        "entity_type_mapping": ONTONOTES_TO_KG,
        "cache_size": cache_size,
        "models_loaded": {
            "ontonotes": get_ontonotes_ner_pipeline() is not None,
            "spacy": get_spacy_ner() is not None
        }
    }

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
    "predict_type_with_ontonotes",
    "get_model_info"
] 