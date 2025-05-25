import os
import json
import logging
import hashlib
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from pathlib import Path
from functools import lru_cache
import tiktoken
import networkx as nx
from rapidfuzz import fuzz
from diskcache import Cache

from app.config import config
from app.models import Triple, Entity, GraphData, GraphNode, GraphLink, RetrievalEmpty, EntityLinkingError, AmbiguousEntityError

# Set up logging
logger = logging.getLogger(__name__)

# Initialize TokenEncoder for counting tokens
try:
    TOKEN_ENCODER = tiktoken.encoding_for_model("qwen")
except Exception:
    logger.warning("Failed to load qwen tokenizer, falling back to cl100k_base")
    TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

# Constants for entity linking
MIN_CONF = 0.75  # Minimum confidence for entity linking
AMBIGUOUS_CONF = 0.60  # Threshold for ambiguous entities

# DiskCache setup
embedding_cache = Cache(directory=os.path.join('cache', 'embeddings'), size_limit=int(1e9))  # 1GB
dde_cache = Cache(directory=os.path.join('cache', 'dde'), size_limit=int(1e9))  # 1GB


def load_aliases():
    """Load entity aliases from file"""
    aliases_path = Path("config/aliases.json")
    try:
        with open(aliases_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load aliases: {e}")
        return {}


# Cache aliases in memory
ALIASES = load_aliases()


def count_tokens(text: str) -> int:
    """Count the number of tokens in a string"""
    if not text:
        return 0
    return len(TOKEN_ENCODER.encode(text))


def triple_to_string(triple: Triple) -> str:
    """Convert triple to string representation"""
    return f"{triple.head_name} {triple.relation_name} {triple.tail_name}"


def hash_text(text: str) -> str:
    """Create hash of text for caching"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


@lru_cache(maxsize=1024)
def embed_query_cached(query: str) -> np.ndarray:
    """Embed a query string with caching"""
    cache_key = f"query_{hash_text(query)}"
    
    # Check cache first
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    # Not in cache, compute embedding
    from app.ml.embedder import embed_text
    embedding = embed_text(query)
    
    # Cache result
    embedding_cache[cache_key] = embedding
    
    return embedding


def get_triple_embedding_cached(triple_id: str) -> np.ndarray:
    """Get embedding for a triple with caching"""
    cache_key = f"triple_{triple_id}"
    
    # Check cache first
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    # Not in cache, fetch from FAISS
    from app.retriever import get_triple_embedding_from_faiss
    embedding = get_triple_embedding_from_faiss(triple_id)
    
    # Cache result
    embedding_cache[cache_key] = embedding
    
    return embedding


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    if vec1 is None or vec2 is None:
        return 0.0
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def detect_type(text: str) -> str:
    """
    Simple entity type detection heuristic
    In a production system, this would be more sophisticated
    """
    text_lower = text.lower()
    
    # Check for people
    if any(title in text_lower for title in ["mr.", "mrs.", "dr.", "prof.", "professor"]):
        return "Person"
    
    # Check for organizations
    if any(suffix in text_lower for suffix in ["inc.", "corp.", "llc", "ltd", "corporation", "company"]):
        return "Organization"
    
    # Check for locations
    if any(place in text_lower for place in ["city", "town", "country", "state", "province", "ocean", "sea", "river", "mountain"]):
        return "Location"
    
    # Default type
    return "Entity"


def link_entities_v2(text_mention: str, query_context: str) -> List[Tuple[str, float]]:
    """
    Link entity mentions to the knowledge graph
    
    Args:
        text_mention: The text mention to link
        query_context: Surrounding context for disambiguation
        
    Returns:
        List of (entity_id, confidence_score) tuples
    """
    if not text_mention or len(text_mention.strip()) == 0:
        return []

    from app.database import neo4j_db
    
    # 1. Exact Match (Name or Alias)
    exact_matches = []
    
    # Check if the text exactly matches a canonical name
    query = """
    MATCH (e:Entity)
    WHERE e.name = $text_mention
    RETURN e.id as id, e.name as name, 1.0 as score
    LIMIT 5
    """
    result = neo4j_db.run_query(query, {"text_mention": text_mention})
    for record in result:
        exact_matches.append((record["id"], record["score"]))
    
    # If no exact name match, check aliases
    if not exact_matches:
        # Check if text matches any alias in our static alias dictionary
        for canonical_name, aliases in ALIASES.items():
            if text_mention in aliases:
                # Find entity by canonical name
                query = """
                MATCH (e:Entity)
                WHERE e.name = $canonical_name
                RETURN e.id as id, e.name as name
                LIMIT 1
                """
                result = neo4j_db.run_query(query, {"canonical_name": canonical_name})
                for record in result:
                    exact_matches.append((record["id"], 0.95))  # Slightly lower confidence for alias match
        
        # Check if text matches any alias in the database
        if not exact_matches:
            query = """
            MATCH (e:Entity)
            WHERE e.name <> $text_mention AND $text_mention IN e.aliases
            RETURN e.id as id, e.name as name
            LIMIT 5
            """
            result = neo4j_db.run_query(query, {"text_mention": text_mention})
            for record in result:
                exact_matches.append((record["id"], 0.95))  # Slightly lower confidence for alias match
    
    # If we have exact matches, return them
    if exact_matches:
        return exact_matches
    
    # 2. Fuzzy Match
    fuzzy_matches = []
    
    # Get candidate entities from the database
    query = """
    MATCH (e:Entity)
    RETURN e.id as id, e.name as name
    LIMIT 1000
    """
    candidates = neo4j_db.run_query(query, {})
    
    # Score candidates using RapidFuzz
    for candidate in candidates:
        entity_id = candidate["id"]
        entity_name = candidate["name"]
        
        # Use token_set_ratio for fuzzy matching - handles word reordering and partial matches
        score = fuzz.token_set_ratio(text_mention, entity_name) / 100.0
        
        # Apply confidence scaling
        confidence = score * 0.9  # Maximum 0.9 for fuzzy matches
        
        if confidence >= AMBIGUOUS_CONF:
            fuzzy_matches.append((entity_id, confidence))
    
    # Sort by confidence descending
    fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 5 fuzzy matches
    top_fuzzy_matches = fuzzy_matches[:5]
    
    # 3. Contextual Disambiguation (if multiple candidates)
    if len(top_fuzzy_matches) > 1:
        from app.ml.embedder import embed_text
        
        # Embed the query context
        query_embedding = embed_text(query_context)
        
        ranked_matches = []
        for entity_id, fuzzy_confidence in top_fuzzy_matches:
            # Get a brief description of the entity (1-hop neighborhood)
            entity_description = get_entity_description(entity_id)
            
            # Embed entity description
            entity_embedding = embed_text(entity_description)
            
            # Calculate cosine similarity for contextual relevance
            context_score = cosine_similarity(query_embedding, entity_embedding)
            
            # Combine scores - equal weight to fuzzy match and context
            final_confidence = (fuzzy_confidence * 0.5) + (context_score * 0.5)
            
            ranked_matches.append((entity_id, final_confidence))
        
        # Sort by final confidence
        ranked_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by confidence threshold
        final_matches = [(entity_id, conf) for entity_id, conf in ranked_matches if conf > AMBIGUOUS_CONF]
        
        return final_matches
    
    return top_fuzzy_matches


def get_entity_description(entity_id: str) -> str:
    """Get a text description of an entity based on its neighborhood"""
    from app.database import neo4j_db
    
    query = """
    MATCH (e:Entity {id: $entity_id})
    OPTIONAL MATCH (e)-[r:REL]-(n:Entity)
    RETURN e.name as name, e.type as type,
           collect(DISTINCT {relation: r.name, entity: n.name}) as connections
    LIMIT 1
    """
    
    result = neo4j_db.run_query(query, {"entity_id": entity_id})
    
    if not result:
        return ""
    
    record = result[0]
    name = record["name"]
    entity_type = record.get("type", "Entity")
    connections = record["connections"]
    
    # Build description
    description = f"{name} is a {entity_type}. "
    
    if connections:
        description += "It is related to: "
        relation_texts = []
        for conn in connections[:5]:  # Limit to avoid too long descriptions
            if conn.get("relation") and conn.get("entity"):
                relation_texts.append(f"{conn['relation']} {conn['entity']}")
        description += ", ".join(relation_texts)
    
    return description


def extract_query_entities(query: str) -> List[str]:
    """Extract potential entity mentions from a query"""
    # This is a simplified implementation
    # In a production system, we would use NER models
    
    # Split query into words
    words = query.split()
    
    # Extract capitalized multi-word entities (naive approach)
    entities = []
    current_entity = []
    
    for word in words:
        if word[0].isupper():
            current_entity.append(word)
        else:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
    
    # Don't forget the last entity
    if current_entity:
        entities.append(" ".join(current_entity))
    
    return entities


def get_dde_encoding(entity_id: str, max_hops: int = 2) -> Dict[str, List[str]]:
    """
    Get DDE (Directional-Distance Encoding) for an entity
    
    Args:
        entity_id: The entity ID
        max_hops: Maximum number of hops (default: 2)
        
    Returns:
        Dictionary mapping hop distance to list of entity IDs
    """
    cache_key = f"dde_{entity_id}_{max_hops}"
    
    # Check cache first
    if cache_key in dde_cache:
        return dde_cache[cache_key]
    
    # Not in cache, compute DDE
    from app.database import neo4j_db
    
    # Use apoc.path.expandConfig to find neighbors in a BFS manner
    query = """
    MATCH (e:Entity {id: $entity_id})
    CALL apoc.path.expandConfig(e, {
        minLevel: 1,
        maxLevel: $max_hops,
        uniqueness: 'NODE_GLOBAL',
        terminatorNodes: [],
        relationshipFilter: 'REL>'
    })
    YIELD path
    WITH path, length(path) as hop_distance
    RETURN hop_distance, collect(DISTINCT last(nodes(path)).id) as entity_ids
    ORDER BY hop_distance
    """
    
    result = neo4j_db.run_query(query, {"entity_id": entity_id, "max_hops": max_hops})
    
    # Build DDE encoding
    dde_encoding = {}
    for record in result:
        hop_distance = record["hop_distance"]
        entity_ids = record["entity_ids"]
        dde_encoding[f"hop{hop_distance}"] = entity_ids
    
    # Cache the result
    dde_cache[cache_key] = dde_encoding
    
    return dde_encoding


def get_dde_for_entities(entity_ids: List[str], max_hops: int = 2) -> Dict[str, Dict[str, List[str]]]:
    """
    Get DDE encodings for multiple entities
    
    Args:
        entity_ids: List of entity IDs
        max_hops: Maximum number of hops
        
    Returns:
        Dictionary mapping entity ID to its DDE encoding
    """
    dde_map = {}
    for entity_id in entity_ids:
        dde_map[entity_id] = get_dde_encoding(entity_id, max_hops)
    return dde_map


def extract_dde_features_for_triple(triple: Triple, dde_map: Dict[str, Dict[str, List[str]]]) -> List[float]:
    """
    Extract DDE features for a triple using precomputed DDE map
    
    Args:
        triple: The triple
        dde_map: Precomputed DDE map from get_dde_for_entities()
        
    Returns:
        List of DDE features
    """
    features = []
    
    # For each entity in dde_map
    for source_entity_id, dde_encoding in dde_map.items():
        # Check if head is in each hop neighborhood
        for hop in range(1, config.MAX_DDE_HOPS + 1):
            hop_key = f"hop{hop}"
            hop_entities = dde_encoding.get(hop_key, [])
            
            # Head entity features
            if triple.head_id in hop_entities:
                features.append(1.0 / hop)  # Closer hops get higher scores
            else:
                features.append(0.0)
            
            # Tail entity features
            if triple.tail_id in hop_entities:
                features.append(1.0 / hop)  # Closer hops get higher scores
            else:
                features.append(0.0)
    
    # If no features were added, use zeros
    if not features:
        features = [0.0] * (4 * len(dde_map))  # 2 entities x 2 hops x number of source entities
    
    return features


def normalize_dde_value(dde_value: List[float]) -> float:
    """Normalize DDE value to a single score in [0, 1]"""
    if not dde_value or all(v == 0 for v in dde_value):
        return 0.0
    
    # Sum of non-zero values, capped at 1.0
    return min(1.0, sum(v for v in dde_value if v > 0))


def heuristic_score(query_embedding, triple_embedding, dde_value):
    """Fallback scoring function when MLP is unavailable"""
    cosine_sim = cosine_similarity(query_embedding, triple_embedding)
    normalized_dde = normalize_dde_value(dde_value)
    return 0.7 * cosine_sim + 0.3 * normalized_dde


def heuristic_score_indexed(dde_features, index):
    """Heuristic scoring function that takes DDE features and an index
    
    Args:
        dde_features: Dictionary of DDE features with lists of values
        index: Index to select which graph to score
    
    Returns:
        Float score based on DDE features only
    """
    try:
        if not dde_features or index < 0:
            return 0.0
        
        # Extract DDE values for the specified index
        dde_values = []
        for feature_name in ['num_nodes', 'num_edges', 'avg_degree', 'density', 'clustering_coefficient']:
            if feature_name in dde_features and len(dde_features[feature_name]) > index:
                dde_values.append(dde_features[feature_name][index])
            else:
                return 0.0  # Return 0 if any feature is missing
        
        # Simple heuristic based on DDE features
        # Normalize and combine features
        num_nodes = dde_values[0]
        num_edges = dde_values[1]
        avg_degree = dde_values[2]
        density = dde_values[3]
        clustering_coeff = dde_values[4]
        
        # Simple scoring heuristic
        # Favor graphs with moderate size and good connectivity
        size_score = min(num_nodes / 100.0, 1.0)  # Normalize by 100 nodes
        connectivity_score = min(avg_degree / 10.0, 1.0)  # Normalize by degree 10
        structure_score = (density + clustering_coeff) / 2.0
        
        final_score = (size_score + connectivity_score + structure_score) / 3.0
        return max(0.0, min(1.0, final_score))  # Clamp to [0, 1]
        
    except Exception as e:
        logger.warning(f"Error in heuristic scoring: {e}")
        return 0.0


def get_score_for_triple(triple_id: str, scored_triples: List[Tuple[float, Triple]]) -> float:
    """Get the score for a triple from a list of scored triples"""
    for score, triple in scored_triples:
        if triple.id == triple_id:
            return score
    return 0.0


def greedy_connect_v2(scored_triples: List[Tuple[float, Triple]], token_budget: int) -> List[Triple]:
    """
    Greedy algorithm to build a connected subgraph from scored triples
    
    Args:
        scored_triples: List of (score, triple) tuples, sorted by score descending
        token_budget: Maximum number of tokens allowed
        
    Returns:
        List of selected triples
    """
    MIN_CONNECTED_COMPONENT_TRIPLES = 3  # Threshold for prioritizing connectivity
    
    subgraph_triples = []
    current_entities = set()
    current_tokens = 0
    
    # Phase 1: Build largest connected component from highest-scoring triples
    # Consider top N triples for initial graph construction
    candidate_pool = [triple for _, triple in scored_triples[:150]]  # Heuristic pool size
    
    temp_graph = nx.Graph()
    for triple in candidate_pool:
        temp_graph.add_edge(
            triple.head_id, 
            triple.tail_id, 
            triple_obj=triple, 
            score=get_score_for_triple(triple.id, scored_triples)
        )
    
    if not temp_graph.edges:
        # Fallback: if no edges, just take top N by score up to budget
        for score, triple in scored_triples:
            cost = count_tokens(triple_to_string(triple))
            if current_tokens + cost <= token_budget:
                subgraph_triples.append(triple)
                current_tokens += cost
            else:
                break
        return subgraph_triples
    
    # Find all connected components, sort them by sum of scores of triples within them
    components_data = []
    for component_nodes in nx.connected_components(temp_graph):
        component_edges_data = [
            data['triple_obj'] for u, v, data in temp_graph.edges(data=True)
            if u in component_nodes and v in component_nodes
        ]
        component_score = sum(get_score_for_triple(t.id, scored_triples) for t in component_edges_data)
        components_data.append({
            'nodes': component_nodes, 
            'triples': component_edges_data, 
            'score': component_score
        })
    
    components_data.sort(key=lambda x: x['score'], reverse=True)
    
    # Add triples from the highest-scored component first, respecting token budget
    for comp_data in components_data:
        comp_triples_sorted_by_score = sorted(
            comp_data['triples'], 
            key=lambda t: get_score_for_triple(t.id, scored_triples), 
            reverse=True
        )
        
        for triple in comp_triples_sorted_by_score:
            cost = count_tokens(triple_to_string(triple))
            if current_tokens + cost <= token_budget:
                if triple not in subgraph_triples:  # Avoid duplicates
                    subgraph_triples.append(triple)
                    current_entities.update([triple.head_id, triple.tail_id])
                    current_tokens += cost
            else:
                # If one triple makes it overflow, try next smaller one from this component
                continue
        
        if current_tokens >= token_budget * 0.8:  # Heuristic: if budget is mostly full
            break
    
    # Phase 2: If budget not filled and largest CC was small, add high-scoring disconnected triples
    if len(subgraph_triples) < MIN_CONNECTED_COMPONENT_TRIPLES or current_tokens < token_budget * 0.5:
        for score, triple in scored_triples:  # Iterate all originally scored triples
            if triple in subgraph_triples:
                continue  # Already added
            
            cost = count_tokens(triple_to_string(triple))
            if current_tokens + cost <= token_budget:
                subgraph_triples.append(triple)
                current_tokens += cost
            else:
                break  # Stop if budget exceeded
    
    return subgraph_triples


def triples_to_graph_data(triples: List[Triple], query_entities: List[str] = None) -> GraphData:
    """
    Convert a list of triples to GraphData for visualization
    
    Args:
        triples: List of triples
        query_entities: List of entity IDs from the original query
        
    Returns:
        GraphData object
    """
    nodes = {}
    links = []
    
    # Set of query entity IDs
    query_entity_ids = set(query_entities or [])
    
    # Process all triples to extract nodes and links
    for triple in triples:
        # Process head node
        if triple.head_id not in nodes:
            node = GraphNode(
                id=triple.head_id,
                name=triple.head_name,
                type=detect_type(triple.head_name),
                relevance_score=triple.relevance_score
            )
            
            # Add inclusion reasons
            inclusion_reasons = []
            if triple.head_id in query_entity_ids:
                inclusion_reasons.append("query_entity")
            nodes[triple.head_id] = node
        
        # Process tail node
        if triple.tail_id not in nodes:
            node = GraphNode(
                id=triple.tail_id,
                name=triple.tail_name,
                type=detect_type(triple.tail_name),
                relevance_score=triple.relevance_score
            )
            
            # Add inclusion reasons
            inclusion_reasons = []
            if triple.tail_id in query_entity_ids:
                inclusion_reasons.append("query_entity")
            nodes[triple.tail_id] = node
        
        # Create link
        link = GraphLink(
            source=triple.head_id,
            target=triple.tail_id,
            relation_id=triple.id,
            relation_name=triple.relation_name,
            relevance_score=triple.relevance_score,
            inclusion_reasons=["part_of_retrieved_subgraph"]
        )
        links.append(link)
    
    # Create GraphData
    graph_data = GraphData(
        nodes=list(nodes.values()),
        links=links
    )
    
    return graph_data