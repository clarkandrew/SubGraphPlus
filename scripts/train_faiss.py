#!/usr/bin/env python3
"""
FAISS Index Training Script

This script trains the FAISS index with real embeddings from the knowledge graph,
replacing any existing index that was trained with random data.
"""

import os
import sys
import logging
import argparse
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config
from app.database import neo4j_db
from app.ml.embedder import embed_text

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'faiss_training.log'))
    ]
)
logger = logging.getLogger(__name__)

def get_training_data(max_samples=2000):
    """Get training data from Neo4j knowledge graph"""
    logger.info("Fetching training data from Neo4j...")
    
    # Get triples from Neo4j
    query = """
    MATCH (h)-[r:REL]->(t)
    RETURN r.id as id, h.name as head_name, r.name as relation_name, t.name as tail_name
    ORDER BY r.id
    LIMIT $max_samples
    """
    
    result = neo4j_db.run_query(query, {"max_samples": max_samples})
    
    if not result:
        logger.error("No triples found in Neo4j")
        return [], []
    
    logger.info(f"Found {len(result)} triples for training")
    
    # Generate embeddings
    embeddings = []
    triple_ids = []
    
    for i, record in enumerate(tqdm(result, desc="Generating embeddings")):
        try:
            # Create triple text representation
            triple_text = f"{record['head_name']} {record['relation_name']} {record['tail_name']}"
            
            # Get embedding
            embedding = embed_text(triple_text)
            
            embeddings.append(embedding)
            triple_ids.append(record['id'])
            
        except Exception as e:
            logger.warning(f"Error processing triple {record['id']}: {e}")
            continue
    
    logger.info(f"Generated {len(embeddings)} embeddings for training")
    return embeddings, triple_ids

def create_trained_index(embeddings, triple_ids):
    """Create and train a new FAISS index with real data"""
    if not embeddings:
        logger.error("No embeddings provided for training")
        return None, None
    
    # Determine dimension
    dim = len(embeddings[0])
    logger.info(f"Creating FAISS index with dimension {dim}")
    
    # Convert to numpy array
    training_data = np.array(embeddings, dtype=np.float32)
    
    # Create index parameters based on data size
    nlist = min(4096, max(100, len(embeddings) // 100))  # Scale with data size
    m = 16  # Number of subquantizers
    nbits = 8  # Bits per subquantizer
    
    logger.info(f"Index parameters: nlist={nlist}, m={m}, nbits={nbits}")
    
    # Create quantizer and index
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    
    # Train the index
    logger.info(f"Training index with {len(training_data)} vectors...")
    index.train(training_data)
    
    # Wrap with IDMap for custom IDs
    index = faiss.IndexIDMap2(index)
    
    # Add all training data to the index
    logger.info("Adding training data to index...")
    
    # Convert string IDs to integers for FAISS
    int_ids = np.array([hash(id_str) & 0x7FFFFFFF for id_str in triple_ids], dtype=np.int64)
    
    # Create ID mapping
    id_map = {}
    for i, id_str in enumerate(triple_ids):
        id_map[int(int_ids[i])] = id_str
    
    # Add vectors with IDs
    index.add_with_ids(training_data, int_ids)
    
    logger.info(f"Created trained index with {index.ntotal} vectors")
    return index, id_map

def save_index(index, id_map, index_path):
    """Save the trained index and ID map"""
    try:
        # Create backup of existing index if it exists
        if os.path.exists(index_path):
            import time
            backup_path = f"{index_path}.{int(time.time())}.bak"
            os.rename(index_path, backup_path)
            logger.info(f"Backed up existing index to {backup_path}")
        
        # Save new index
        faiss.write_index(index, index_path)
        logger.info(f"Saved trained index to {index_path}")
        
        # Save ID map
        id_map_path = f"{index_path}.ids"
        if os.path.exists(id_map_path):
            backup_path = f"{id_map_path}.{int(time.time())}.bak"
            os.rename(id_map_path, backup_path)
            logger.info(f"Backed up existing ID map to {backup_path}")
        
        np.save(id_map_path, id_map)
        logger.info(f"Saved ID map to {id_map_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving index: {e}")
        return False

def verify_index(index_path):
    """Verify the trained index works correctly"""
    try:
        logger.info("Verifying trained index...")
        
        # Load index
        index = faiss.read_index(index_path)
        id_map_path = f"{index_path}.ids"
        id_map = np.load(id_map_path, allow_pickle=True).item()
        
        logger.info(f"Index loaded successfully: {index.ntotal} vectors, {len(id_map)} ID mappings")
        
        # Test search
        if index.ntotal > 0:
            # Create a random query vector
            dim = index.d
            query_vector = np.random.random((1, dim)).astype(np.float32)
            
            # Search
            distances, indices = index.search(query_vector, min(5, index.ntotal))
            
            logger.info(f"Test search returned {len(indices[0])} results")
            
            # Verify ID mappings
            valid_mappings = 0
            for idx in indices[0]:
                if idx >= 0 and int(idx) in id_map:
                    valid_mappings += 1
            
            logger.info(f"Valid ID mappings: {valid_mappings}/{len(indices[0])}")
            
            if valid_mappings > 0:
                logger.info("Index verification successful!")
                return True
            else:
                logger.error("No valid ID mappings found")
                return False
        else:
            logger.warning("Index is empty")
            return False
            
    except Exception as e:
        logger.error(f"Index verification failed: {e}")
        return False

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train FAISS index with real data")
    parser.add_argument("--max-samples", type=int, default=2000, 
                       help="Maximum number of triples to use for training")
    parser.add_argument("--verify", action="store_true", 
                       help="Verify the index after training")
    parser.add_argument("--force", action="store_true",
                       help="Force retraining even if index exists")
    
    args = parser.parse_args()
    
    # Check if index already exists and is trained with real data
    index_path = config.FAISS_INDEX_PATH
    if os.path.exists(index_path) and not args.force:
        try:
            index = faiss.read_index(index_path)
            if index.ntotal > 100:  # Assume if it has many vectors, it's real data
                logger.info(f"Index already exists with {index.ntotal} vectors. Use --force to retrain.")
                return 0
        except Exception:
            pass  # Continue with training if we can't load existing index
    
    logger.info("Starting FAISS index training with real data")
    
    # Get training data
    embeddings, triple_ids = get_training_data(args.max_samples)
    
    if len(embeddings) < 100:
        logger.error(f"Insufficient training data: {len(embeddings)} embeddings (need at least 100)")
        return 1
    
    # Create and train index
    index, id_map = create_trained_index(embeddings, triple_ids)
    
    if index is None:
        logger.error("Failed to create trained index")
        return 1
    
    # Save index
    if not save_index(index, id_map, index_path):
        logger.error("Failed to save trained index")
        return 1
    
    # Verify if requested
    if args.verify:
        if not verify_index(index_path):
            logger.error("Index verification failed")
            return 1
    
    logger.info("FAISS index training completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 