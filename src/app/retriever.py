import os
import json
import time
import logging
import numpy as np
import faiss
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any, Set
import torch
import torch.nn as nn

from app.config import config
from app.models import Triple, RetrievalEmpty
from app.utils import (
    embed_query_cached,
    get_triple_embedding_cached,
    extract_dde_features_for_triple,
    cosine_similarity,
    heuristic_score,
    greedy_connect_v2,
    get_dde_for_entities,
    heuristic_score_indexed
)
from app.database import neo4j_db

logger = logging.getLogger(__name__)

# Constants for retrieval
GRAPH_CANDIDATES_K = 100  # Max candidates from graph stage before MLP scoring
DENSE_CANDIDATES_K = 50   # Max candidates from dense stage before MLP scoring
FINAL_GRAPH_TOP_K = 60    # Triples to select from graph-favored candidates after MLP
FINAL_DENSE_TOP_K = 20    # Triples to select from dense-favored candidates after MLP


class FaissIndex:
    """FAISS index manager for triple embeddings"""
    
    def __init__(self):
        self.index = None
        self.id_map = {}  # Maps index IDs to Triple IDs
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index from disk"""
        try:
            index_path = config.FAISS_INDEX_PATH
            if os.path.exists(index_path):
                logger.info(f"Loading FAISS index from {index_path}")
                self.index = faiss.read_index(index_path)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                
                # Load ID mapping
                id_map_path = f"{index_path}.ids"
                if os.path.exists(id_map_path):
                    self.id_map = np.load(id_map_path, allow_pickle=True).item()
                    logger.info(f"Loaded ID map with {len(self.id_map)} entries")
                else:
                    logger.warning(f"ID map file not found: {id_map_path}")
            else:
                logger.warning(f"FAISS index file not found: {index_path}")
                self._create_empty_index()
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self._create_empty_index()
    
    def _create_empty_index(self):
        """Create an empty FAISS index"""
        logger.info("Creating empty FAISS index")
        # Determine dimension based on model backend
        dim = 1024  # gte-large-en-v1.5 dimension for HF and MLX
        if config.MODEL_BACKEND == "openai":
            dim = 1536
            
        # Create IVF index with PQ for efficient search
        quantizer = faiss.IndexFlatL2(dim)
        nlist = 100  # Number of clusters (Voronoi cells)
        m = 16  # Number of subquantizers
        nbits = 8  # Bits per subquantizer
        
        # Create IndexIVFPQ with ID mapping
        self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        
        # Wrap with IDMap to use our own IDs
        self.index = faiss.IndexIDMap2(self.index)
        
        # Initialize ID map
        self.id_map = {}
        
        logger.info(f"Created empty FAISS index with dimension {dim}")
    
    def is_trained(self):
        """Check if the index is trained"""
        if hasattr(self.index, 'is_trained'):
            return self.index.is_trained
        
        # For IndexIDMap, check the underlying index
        if isinstance(self.index, faiss.IndexIDMap) and hasattr(self.index.index, 'is_trained'):
            return self.index.index.is_trained
            
        return True  # Default to True for simple indices
    
    def add_vectors(self, ids: List[str], vectors: np.ndarray):
        """Add vectors to the index"""
        try:
            # Convert string IDs to integers for FAISS
            int_ids = np.array([hash(id_str) & 0x7FFFFFFF for id_str in ids], dtype=np.int64)
            
            # Map integer IDs to original string IDs
            for i, id_str in enumerate(ids):
                self.id_map[int(int_ids[i])] = id_str
            
            # Add vectors to index
            self.index.add_with_ids(vectors, int_ids)
            logger.info(f"Added {len(ids)} vectors to FAISS index")
            
            # Save ID mapping
            np.save(f"{config.FAISS_INDEX_PATH}.ids", self.id_map)
        except Exception as e:
            logger.error(f"Failed to add vectors to FAISS index: {e}")
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search index for k nearest neighbors"""
        try:
            if self.index.ntotal == 0:
                logger.warning("FAISS index is empty, returning empty results")
                return []
            
            # Search index
            distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
            
            # Map indices back to Triple IDs
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0:  # FAISS uses -1 for empty results
                    triple_id = self.id_map.get(int(idx))
                    if triple_id:
                        # Convert distance to similarity score (1 - normalized_distance)
                        distance = float(distances[0][i])
                        similarity = 1.0 - min(distance / 10.0, 1.0)  # Normalize and invert
                        results.append((triple_id, similarity))
            
            return results
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return []
    
    def get_vector(self, triple_id: str) -> Optional[np.ndarray]:
        """Get vector for a Triple ID"""
        try:
            # Find the integer ID for this triple ID
            int_id = None
            for k, v in self.id_map.items():
                if v == triple_id:
                    int_id = k
                    break
            
            if int_id is None:
                logger.warning(f"Triple ID {triple_id} not found in ID map")
                return None
            
            # Reconstruct vector
            vector = np.zeros((1, self.index.d), dtype=np.float32)
            self.index.reconstruct(int_id, vector[0])
            return vector[0]
        except Exception as e:
            logger.error(f"Failed to get vector for triple {triple_id}: {e}")
            return None


# Singleton instance
faiss_index = FaissIndex()


def get_triple_embedding_from_faiss(triple_id: str) -> np.ndarray:
    """Get embedding for a triple from FAISS"""
    embedding = faiss_index.get_vector(triple_id)
    if embedding is None:
        # Return zero vector with appropriate dimension
        dim = 1024  # gte-large-en-v1.5 dimension for HF and MLX
        if config.MODEL_BACKEND == "openai":
            dim = 1536
        embedding = np.zeros(dim, dtype=np.float32)
    return embedding


class SimpleMLP(nn.Module):
    """Simple MLP for SubgraphRAG scoring"""
    def __init__(self, input_dim=4116, hidden_dim=1024, output_dim=1):  # 1024*4 + 20 DDE features = 4116
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


def calculate_mlp_input_dim(embeddings, dde_features):
    """Calculate the input dimension for MLP based on embeddings and DDE features"""
    if embeddings is not None and len(embeddings) > 0:
        embedding_dim = embeddings.shape[-1] if hasattr(embeddings, 'shape') else len(embeddings[0])
    else:
        embedding_dim = 1024  # gte-large-en-v1.5 embedding dimension
    
    dde_dim = len(dde_features) if dde_features else 0
    return embedding_dim + dde_dim


def load_pretrained_mlp():
    """Load pre-trained SubgraphRAG MLP model"""
    try:
        mlp_path = config.MLP_MODEL_PATH
        if os.path.exists(mlp_path):
            logger.info(f"Loading pre-trained MLP from {mlp_path}")
            
            # Try to load as state dict first (safer)
            try:
                state_dict = torch.load(mlp_path, map_location=torch.device('cpu'), weights_only=True)
                model = SimpleMLP()
                model.load_state_dict(state_dict)
                return model
            except Exception as state_dict_error:
                # Fallback to loading full model (less safe but might be needed for older models)
                logger.warning(f"Could not load as state dict: {state_dict_error}, trying full model load")
                try:
                    model = torch.load(mlp_path, map_location=torch.device('cpu'), weights_only=False)
                    return model
                except Exception as full_load_error:
                    logger.warning(f"Could not load full model either: {full_load_error}")
                    return None
        else:
            logger.warning(f"Pre-trained MLP not found at {mlp_path}")
            return None
    except Exception as e:
        logger.warning(f"Could not load pre-trained MLP: {e}")
        return None


# Load MLP model (or None if not available)
mlp_model = load_pretrained_mlp()


def mlp_score(embeddings, dde_features, index):
    """Score using the MLP or fallback heuristic
    
    Args:
        embeddings: Tensor or array of embeddings
        dde_features: Dictionary of DDE features with lists of values
        index: Index to select which graph/triple to score
    
    Returns:
        Float score
    """
    if mlp_model is not None:
        try:
            # Validate inputs
            if embeddings is None or dde_features is None:
                raise ValueError("Invalid inputs")
            
            # Get embedding for the specified index
            if hasattr(embeddings, 'shape') and len(embeddings.shape) > 1:
                if index >= embeddings.shape[0]:
                    raise IndexError("Index out of bounds for embeddings")
                embedding = embeddings[index]
            else:
                if index >= len(embeddings):
                    raise IndexError("Index out of bounds for embeddings")
                embedding = embeddings[index]
            
            # Extract DDE features for the specified index
            dde_values = []
            for feature_name in ['num_nodes', 'num_edges', 'avg_degree', 'density', 'clustering_coefficient']:
                if feature_name in dde_features and len(dde_features[feature_name]) > index:
                    dde_values.append(dde_features[feature_name][index])
                else:
                    # Fallback to heuristic if DDE features are incomplete
                    return heuristic_score_indexed(dde_features, index)
            
            # Convert to torch tensors
            emb_tensor = torch.tensor(embedding, dtype=torch.float32).flatten()
            dde_tensor = torch.tensor(dde_values, dtype=torch.float32)
            
            # Concatenate features
            combined = torch.cat([emb_tensor, dde_tensor])
            
            # Check dimension compatibility
            expected_dim = calculate_mlp_input_dim(embeddings, dde_features)
            if combined.shape[0] != expected_dim:
                logger.warning(f"Dimension mismatch: expected {expected_dim}, got {combined.shape[0]}")
                # Fallback to heuristic
                return heuristic_score_indexed(dde_features, index)
            
            # Get score from MLP
            with torch.no_grad():
                score = mlp_model(combined.unsqueeze(0)).item()
            
            return score
        except Exception as e:
            logger.error(f"Error in MLP scoring: {e}, falling back to heuristic")
            return heuristic_score_indexed(dde_features, index)
    else:
        # Fallback to heuristic scoring
        return heuristic_score_indexed(dde_features, index)


def neo4j_get_neighborhood_triples(entity_ids: List[str], hops: int, limit: int) -> List[Triple]:
    """Get triples from the neighborhood of entities in Neo4j"""
    if not entity_ids:
        return []
    
    query = """
    MATCH (e:Entity)
    WHERE e.id IN $entity_ids
    CALL apoc.path.expandConfig(e, {
        minLevel: 1,
        maxLevel: $hops,
        uniqueness: 'RELATIONSHIP_GLOBAL',
        relationshipFilter: 'REL'
    })
    YIELD path
    WITH DISTINCT relationships(path) as rels
    UNWIND rels as r
    MATCH (h)-[r]->(t)
    RETURN 
        r.id as id,
        h.id as head_id,
        h.name as head_name,
        r.name as relation_name,
        t.id as tail_id,
        t.name as tail_name
    LIMIT $limit
    """
    
    result = neo4j_db.run_query(query, {
        "entity_ids": entity_ids,
        "hops": hops,
        "limit": limit
    })
    
    triples = []
    for record in result:
        triple = Triple(
            id=record["id"],
            head_id=record["head_id"],
            head_name=record["head_name"],
            relation_id=record["id"],  # Using triple ID as relation ID
            relation_name=record["relation_name"],
            tail_id=record["tail_id"],
            tail_name=record["tail_name"]
        )
        triples.append(triple)
    
    logger.info(f"Retrieved {len(triples)} triples from Neo4j neighborhood")
    return triples


def faiss_search_triples_data(query_embedding: np.ndarray, k: int = 50) -> List[Dict[str, Any]]:
    """Search FAISS for similar triples and retrieve their data from Neo4j"""
    # Search FAISS
    triple_ids_scores = faiss_index.search(query_embedding, k)
    
    if not triple_ids_scores:
        return []
    
    # Get triple IDs
    triple_ids = [t_id for t_id, _ in triple_ids_scores]
    
    # Retrieve triple data from Neo4j
    query = """
    MATCH (h)-[r:REL]->(t)
    WHERE r.id IN $triple_ids
    RETURN
        r.id as id,
        h.id as head_id,
        h.name as head_name,
        r.name as relation_name,
        t.id as tail_id,
        t.name as tail_name
    """
    
    result = neo4j_db.run_query(query, {"triple_ids": triple_ids})
    
    # Create dictionary of triple data
    triples_data = []
    for record in result:
        # Find score for this triple
        score = next((score for t_id, score in triple_ids_scores if t_id == record["id"]), 0.0)
        
        triple_data = {
            "id": record["id"],
            "head_id": record["head_id"],
            "head_name": record["head_name"],
            "relation_name": record["relation_name"],
            "relation_id": record["id"],  # Using triple ID as relation ID
            "tail_id": record["tail_id"],
            "tail_name": record["tail_name"],
            "relevance_score": score
        }
        triples_data.append(triple_data)
    
    logger.info(f"Retrieved {len(triples_data)} triples from FAISS search")
    return triples_data


def hybrid_retrieve_v2(question: str, query_entities: List[str]) -> List[Triple]:
    """
    Hybrid retrieval combining graph traversal and dense retrieval
    
    Args:
        question: The natural language question
        query_entities: List of entity IDs from the question
        
    Returns:
        List of retrieved triples
    """
    logger.info(f"Hybrid retrieve for question: '{question}' with entities: {query_entities}")
    
    # Embed query
    q_emb = embed_query_cached(question)
    
    # Get DDE for entities
    dde_map = get_dde_for_entities(query_entities, max_hops=config.MAX_DDE_HOPS)
    
    # 1. Graph Candidate Generation (focused on query_entities)
    graph_candidate_triples = neo4j_get_neighborhood_triples(
        query_entities,
        hops=config.MAX_DDE_HOPS,
        limit=GRAPH_CANDIDATES_K
    )
    
    # 2. Dense Candidate Generation
    dense_candidate_triples_data = faiss_search_triples_data(q_emb, k=DENSE_CANDIDATES_K)
    
    # Convert dense results to Triple objects
    dense_triples = [Triple(**t_data) for t_data in dense_candidate_triples_data]
    
    # 3. Score all candidates
    # Combine and deduplicate candidates
    all_candidates = []
    seen_ids = set()
    
    # Add graph candidates first
    for triple in graph_candidate_triples:
        if triple.id not in seen_ids:
            all_candidates.append(triple)
            seen_ids.add(triple.id)
    
    # Then add dense candidates
    for triple in dense_triples:
        if triple.id not in seen_ids:
            all_candidates.append(triple)
            seen_ids.add(triple.id)
    
    logger.info(f"Scoring {len(all_candidates)} unique candidate triples")
    
    if not all_candidates:
        raise RetrievalEmpty("No candidate triples found")
    
    # Score each triple
    scored_triples = []
    for triple in all_candidates:
        # Get triple embedding
        triple_embedding = get_triple_embedding_cached(triple.id)
        
        # Extract DDE features
        triple_dde_features = extract_dde_features_for_triple(triple, dde_map)
        
        # Score with MLP or fallback
        score = mlp_score(triple_embedding, triple_dde_features, triple.id)
        
        # Store score and triple
        scored_triples.append((score, triple))
        
        # Update triple relevance score for visualization
        triple.relevance_score = score
    
    # Sort by score descending
    scored_triples.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Fusion & Selection
    final_selection = []
    selected_ids = set()
    
    # Add best graph-originated triples
    for score, triple in scored_triples:
        if len(final_selection) >= FINAL_GRAPH_TOP_K:
            break
        
        if triple in graph_candidate_triples and triple.id not in selected_ids:
            final_selection.append(triple)
            selected_ids.add(triple.id)
    
    # Add best dense-originated triples (not already selected)
    for score, triple in scored_triples:
        if len(final_selection) >= (FINAL_GRAPH_TOP_K + FINAL_DENSE_TOP_K):
            break
            
        # Check if triple was originally from dense search
        is_from_dense = any(triple.id == t.id for t in dense_triples)
        
        if is_from_dense and triple.id not in selected_ids:
            final_selection.append(triple)
            selected_ids.add(triple.id)
    
    # If still under budget, fill with remaining highest scored triples
    for score, triple in scored_triples:
        if len(final_selection) >= (FINAL_GRAPH_TOP_K + FINAL_DENSE_TOP_K):
            break
            
        if triple.id not in selected_ids:
            final_selection.append(triple)
            selected_ids.add(triple.id)
    
    if not final_selection:
        raise RetrievalEmpty("No relevant triples found after scoring and fusion")
    
    logger.info(f"Selected {len(final_selection)} triples for final subgraph")
    
    # 5. Subgraph Assembly with token budget
    final_triples = greedy_connect_v2(scored_triples, token_budget=config.TOKEN_BUDGET)
    logger.info(f"Final subgraph has {len(final_triples)} triples after token budget constraints")
    
    return final_triples


def entity_search(search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for entities by name"""
    query = """
    MATCH (e:Entity)
    WHERE e.name CONTAINS $search_term
    RETURN e.id AS id, e.name AS name, e.type AS type, labels(e) AS labels
    LIMIT $limit
    """
    
    result = neo4j_db.run_query(query, {"search_term": search_term, "limit": limit})
    
    entities = []
    for record in result:
        entity = {
            "id": record["id"],
            "name": record["name"],
            "type": record["type"],
        }
        entities.append(entity)
    
    return entities