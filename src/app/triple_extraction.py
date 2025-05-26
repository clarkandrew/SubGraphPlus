"""
Centralized Triple Extraction Module
Consolidates REBEL parsing logic and integrates with entity typing
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Triple:
    """Represents an extracted triple with metadata"""
    head: str
    relation: str
    tail: str
    head_type: Optional[str] = None
    tail_type: Optional[str] = None
    confidence: float = 1.0
    source: str = "rebel_extraction"

def extract_triplets_from_rebel(text: str, debug: bool = False) -> List[Dict[str, str]]:
    """
    Extract triplets from REBEL model output
    
    Args:
        text: Raw model output text from REBEL
        debug: Enable debug logging
        
    Returns:
        List of triplet dictionaries with 'head', 'relation', 'tail' keys
    """
    triplets = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'

    if debug:
        logger.debug(f"Processing REBEL output: {text}")

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if debug:
            logger.debug(f"Token='{token}', Current='{current}', Subject='{subject}', Object='{object_}', Relation='{relation}'")

        if token == "<triplet>":
            current = 't'  # Start with subject after <triplet>
            if relation != '' and subject != '' and object_ != '':
                triplets.append({
                    'head': subject.strip(), 
                    'relation': relation.strip(),
                    'tail': object_.strip()
                })
                if debug:
                    logger.debug(f"Added triplet - head:{subject.strip()}, relation:{relation.strip()}, tail:{object_.strip()}")
            relation = ''
            subject = ''
            object_ = ''
        elif token == "<subj>":
            current = 's'  # Switch to object after <subj>
        elif token == "<obj>":
            current = 'o'  # Switch to relation after <obj>
        else:
            if current == 't':
                subject += ' ' + token if subject else token
            elif current == 's':
                object_ += ' ' + token if object_ else token
            elif current == 'o':
                relation += ' ' + token if relation else token

    # Add final triplet if exists
    if subject != '' and relation != '' and object_ != '':
        triplets.append({
            'head': subject.strip(), 
            'relation': relation.strip(),
            'tail': object_.strip()
        })
        if debug:
            logger.debug(f"Final triplet - head:{subject.strip()}, relation:{relation.strip()}, tail:{object_.strip()}")

    return triplets

def clean_and_validate_triple(triple_dict: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Clean and validate a triple dictionary
    
    Args:
        triple_dict: Raw triple dictionary
        
    Returns:
        Cleaned triple dictionary or None if invalid
    """
    if not all(key in triple_dict for key in ['head', 'relation', 'tail']):
        logger.warning(f"Skipping malformed triple: {triple_dict}")
        return None
    
    # Clean and validate triple components
    head = triple_dict['head'].strip()
    relation = triple_dict['relation'].strip()
    tail = triple_dict['tail'].strip()
    
    if not head or not relation or not tail:
        logger.warning(f"Skipping empty triple components: {triple_dict}")
        return None
    
    # Basic quality filters
    if len(head.split()) > 10 or len(tail.split()) > 10:
        logger.warning(f"Skipping overly long entities: {triple_dict}")
        return None
    
    if len(relation.split()) > 5:
        logger.warning(f"Skipping overly long relation: {triple_dict}")
        return None
    
    return {
        'head': head,
        'relation': relation,
        'tail': tail
    }

def add_entity_types_to_triple(triple_dict: Dict[str, str], context: Optional[str] = None) -> Dict[str, str]:
    """
    Add entity types to a triple using the schema-driven approach
    
    Args:
        triple_dict: Triple dictionary with head, relation, tail
        context: Optional context for entity typing
        
    Returns:
        Triple dictionary with added head_type and tail_type
    """
    from app.entity_typing import get_entity_type
    
    head_context = f"Subject of {triple_dict['relation']}" if context is None else context
    tail_context = f"Object of {triple_dict['relation']}" if context is None else context
    
    triple_dict['head_type'] = get_entity_type(triple_dict['head'])
    triple_dict['tail_type'] = get_entity_type(triple_dict['tail'])
    
    return triple_dict

def process_rebel_output(raw_output: str, context: Optional[str] = None, debug: bool = False) -> List[Triple]:
    """
    Complete pipeline: REBEL output → cleaned triples with entity types
    
    Args:
        raw_output: Raw output from REBEL model
        context: Optional context for entity typing
        debug: Enable debug logging
        
    Returns:
        List of Triple objects with entity types
    """
    # Extract raw triplets
    raw_triplets = extract_triplets_from_rebel(raw_output, debug=debug)
    
    processed_triples = []
    for raw_triple in raw_triplets:
        # Clean and validate
        cleaned_triple = clean_and_validate_triple(raw_triple)
        if cleaned_triple is None:
            continue
        
        # Add entity types
        typed_triple = add_entity_types_to_triple(cleaned_triple, context=context)
        
        # Create Triple object
        triple = Triple(
            head=typed_triple['head'],
            relation=typed_triple['relation'],
            tail=typed_triple['tail'],
            head_type=typed_triple['head_type'],
            tail_type=typed_triple['tail_type'],
            confidence=1.0,  # REBEL doesn't provide confidence scores
            source="rebel_extraction"
        )
        
        processed_triples.append(triple)
    
    logger.info(f"Processed {len(raw_triplets)} raw triplets into {len(processed_triples)} valid triples")
    return processed_triples

def normalize_relation(relation: str) -> str:
    """
    Normalize relation strings for consistency
    
    Args:
        relation: Raw relation string
        
    Returns:
        Normalized relation string
    """
    # Convert to lowercase and replace spaces/hyphens with underscores
    normalized = relation.lower().replace(' ', '_').replace('-', '_')
    
    # Remove special characters except underscores
    normalized = re.sub(r'[^\w_]', '', normalized)
    
    # Common relation mappings
    relation_mappings = {
        'place_of_birth': 'born_in',
        'date_of_birth': 'born_on',
        'place_of_death': 'died_in',
        'date_of_death': 'died_on',
        'ethnic_group': 'ethnicity',
        'member_of': 'belongs_to',
        'memberof': 'belongs_to',  # Handle both variations
        'part_of': 'belongs_to',
        'partof': 'belongs_to',    # Handle both variations
        'located_in': 'in',
        'capital_of': 'capital',
    }
    
    return relation_mappings.get(normalized, normalized)

def batch_process_texts(texts: List[str], ie_service_url: str, max_length: int = 256) -> List[Triple]:
    """
    Process multiple texts through REBEL IE service
    
    Args:
        texts: List of text strings to process
        ie_service_url: URL of the REBEL IE service
        max_length: Maximum sequence length for REBEL
        
    Returns:
        List of all extracted triples
    """
    import requests
    
    all_triples = []
    
    for i, text in enumerate(texts):
        try:
            response = requests.post(
                f"{ie_service_url}/extract",
                json={
                    "text": text,
                    "max_length": max_length,
                    "num_beams": 3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # Convert API response to our Triple format
                for triple_data in result['triples']:
                    triple = Triple(
                        head=triple_data['head'],
                        relation=normalize_relation(triple_data['relation']),
                        tail=triple_data['tail'],
                        confidence=triple_data.get('confidence', 1.0),
                        source="rebel_ie_service"
                    )
                    # Add entity types
                    from app.entity_typing import get_entity_type
                    triple.head_type = get_entity_type(triple.head)
                    triple.tail_type = get_entity_type(triple.tail)
                    
                    all_triples.append(triple)
                
                logger.debug(f"Processed text {i+1}/{len(texts)}: {len(result['triples'])} triples")
            else:
                logger.error(f"IE service error for text {i+1}: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error processing text {i+1}: {e}")
    
    logger.info(f"Batch processed {len(texts)} texts, extracted {len(all_triples)} total triples")
    return all_triples

def extract_triples(text: str) -> List[Dict[str, str]]:
    """
    Extract triples from text using graph-based extraction
    
    Args:
        text: Input text to extract triples from
        
    Returns:
        List of triple dictionaries with 'head', 'relation', 'tail' keys
    """
    from ..database import neo4j_db
    
    # First try REBEL extraction
    rebel_triples = process_rebel_output(text)
    
    # Then enhance with graph-based extraction
    graph_triples = []
    
    try:
        with neo4j_db.get_session() as session:
            # For each entity in the text, find its neighbors in the graph
            for triple in rebel_triples:
                # Look up head entity's neighbors
                head_result = session.run(
                    """
                    MATCH (e:Entity) WHERE toLower(e.name) = toLower($name)
                    MATCH (e)-[r:REL]->(neighbor:Entity)
                    RETURN e.name as entity, type(r) as relation, neighbor.name as neighbor
                    LIMIT 5
                    """,
                    {"name": triple.head}
                )
                
                for record in head_result:
                    graph_triples.append({
                        'head': record["entity"],
                        'relation': record["relation"],
                        'tail': record["neighbor"]
                    })
                
                # Look up tail entity's neighbors
                tail_result = session.run(
                    """
                    MATCH (e:Entity) WHERE toLower(e.name) = toLower($name)
                    MATCH (e)<-[r:REL]-(neighbor:Entity)
                    RETURN neighbor.name as entity, type(r) as relation, e.name as neighbor
                    LIMIT 5
                    """,
                    {"name": triple.tail}
                )
                
                for record in tail_result:
                    graph_triples.append({
                        'head': record["entity"],
                        'relation': record["relation"],
                        'tail': record["neighbor"]
                    })
    
    except Exception as e:
        logger.warning(f"Graph-based extraction failed: {e}")
    
    # Combine REBEL and graph-based triples
    all_triples = rebel_triples + graph_triples
    
    # Remove duplicates
    seen = set()
    unique_triples = []
    for triple in all_triples:
        key = (triple['head'], triple['relation'], triple['tail'])
        if key not in seen:
            seen.add(key)
            unique_triples.append(triple)
    
    return unique_triples 