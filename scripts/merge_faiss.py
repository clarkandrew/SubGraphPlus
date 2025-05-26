import os
import logging
import time
import numpy as np
import faiss
import sys
from pathlib import Path
import argparse

from app.config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'faiss_merge.log'))
    ]
)
logger = logging.getLogger(__name__)

def merge_staged_embeddings():
    """Merge staged embeddings into the main FAISS index"""
    # Get FAISS index path from config
    index_path = config.FAISS_INDEX_PATH
    id_map_path = f"{index_path}.ids"
    
    # Check staging directory
    staging_dir = Path("data/faiss_staging")
    if not staging_dir.exists() or not any(staging_dir.glob("batch_*.npy")):
        logger.info("No staged embeddings found to merge")
        return 0
    
    # Load existing index or create new one
    if os.path.exists(index_path):
        logger.info(f"Loading existing FAISS index from {index_path}")
        try:
            index = faiss.read_index(index_path)
            logger.info(f"Loaded index with {index.ntotal} vectors")
            
            # Load ID map
            if os.path.exists(id_map_path):
                id_map = np.load(id_map_path, allow_pickle=True).item()
                logger.info(f"Loaded ID map with {len(id_map)} entries")
            else:
                logger.warning(f"ID map not found at {id_map_path}, creating new one")
                id_map = {}
                
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            logger.info("Creating new index")
            index, id_map = create_new_index()
    else:
        logger.info("FAISS index not found, creating new one")
        index, id_map = create_new_index()
    
    # Process all staged files
    total_added = 0
    archive_dir = staging_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    # Find all batches
    batch_files = list(staging_dir.glob("batch_*.npy"))
    batch_files = [f for f in batch_files if not f.name.endswith("_ids.npy")]
    
    logger.info(f"Found {len(batch_files)} batch files to process")
    
    for batch_file in batch_files:
        # Get corresponding IDs file
        ids_file = batch_file.with_name(f"{batch_file.stem}_ids.npy")
        if not ids_file.exists():
            logger.warning(f"IDs file not found for {batch_file}, skipping")
            continue
        
        try:
            # Load vectors and IDs
            vectors = np.load(batch_file)
            rel_ids = np.load(ids_file, allow_pickle=True)
            
            if len(vectors) != len(rel_ids):
                logger.warning(f"Mismatch between vectors ({len(vectors)}) and IDs ({len(rel_ids)}) in {batch_file}, skipping")
                continue
            
            # Convert string IDs to integers for FAISS
            int_ids = np.array([hash(id_str) & 0x7FFFFFFF for id_str in rel_ids], dtype=np.int64)
            
            # Update ID map
            for i, id_str in enumerate(rel_ids):
                id_map[int(int_ids[i])] = id_str
            
            # Add to index
            logger.info(f"Adding {len(vectors)} vectors to index")
            index.add_with_ids(vectors, int_ids)
            total_added += len(vectors)
            
            # Move processed files to archive
            batch_archive = archive_dir / batch_file.name
            ids_archive = archive_dir / ids_file.name
            
            # Move with timestamp to avoid conflicts
            timestamp = int(time.time())
            batch_file.rename(archive_dir / f"{batch_file.stem}_{timestamp}.npy")
            ids_file.rename(archive_dir / f"{ids_file.stem}_{timestamp}.npy")
            
        except Exception as e:
            logger.error(f"Error processing {batch_file}: {e}")
    
    # Save updated index and ID map if any vectors were added
    if total_added > 0:
        logger.info(f"Added {total_added} vectors to index, saving...")
        
        try:
            # Create backup of existing index if it exists
            if os.path.exists(index_path):
                backup_path = f"{index_path}.bak"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(index_path, backup_path)
                logger.info(f"Created backup of index at {backup_path}")
            
            # Save updated index
            faiss.write_index(index, index_path)
            logger.info(f"Saved updated index with {index.ntotal} vectors to {index_path}")
            
            # Save updated ID map
            np.save(id_map_path, id_map)
            logger.info(f"Saved updated ID map with {len(id_map)} entries to {id_map_path}")
            
        except Exception as e:
            logger.error(f"Failed to save updated index: {e}")
            
            # Try to restore from backup
            if os.path.exists(f"{index_path}.bak"):
                logger.info("Attempting to restore from backup")
                try:
                    os.rename(f"{index_path}.bak", index_path)
                    logger.info("Restored index from backup")
                except Exception as e2:
                    logger.error(f"Failed to restore from backup: {e2}")
    
    return total_added

def create_new_index():
    """Create a new FAISS index"""
    # Determine dimension based on model backend
    dim = 1024  # gte-large-en-v1.5 dimension for HF and MLX
    if config.MODEL_BACKEND == "openai":
        dim = 1536
    
    logger.info(f"Creating new FAISS index with dimension {dim}")
    
    # Create IVF index with PQ for efficient search
    quantizer = faiss.IndexFlatL2(dim)
    nlist = 100  # Number of clusters (Voronoi cells)
    m = 16  # Number of subquantizers
    nbits = 8  # Bits per subquantizer
    
    # Create IndexIVFPQ
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    
    # Wrap with IDMap to use our own IDs
    index = faiss.IndexIDMap2(index)
    
    # Initialize ID map
    id_map = {}
    
    # Try to train with real data first
    if not index.is_trained:
        logger.info("Training index with real data if available")
        if not train_index_with_real_data(index):
            logger.info("No real data available, training with random data")
            train_size = 1000
            train_data = np.random.random((train_size, dim)).astype(np.float32)
            index.train(train_data)
    
    return index, id_map

def train_index_with_real_data(index):
    """Train FAISS index with real embeddings from Neo4j"""
    try:
        from app.database import neo4j_db
        from app.ml.embedder import embed_text
        
        # Get a sample of triples from Neo4j for training
        query = """
        MATCH (h)-[r:REL]->(t)
        RETURN r.id as id, h.name as head_name, r.name as relation_name, t.name as tail_name
        LIMIT 2000
        """
        
        result = neo4j_db.run_query(query)
        
        if not result:
            logger.warning("No triples found in Neo4j for training")
            return False
        
        logger.info(f"Found {len(result)} triples for training")
        
        # Generate embeddings for training
        training_embeddings = []
        for i, record in enumerate(result):
            if i % 100 == 0:
                logger.info(f"Processing training triple {i+1}/{len(result)}")
            
            # Create triple text
            triple_text = f"{record['head_name']} {record['relation_name']} {record['tail_name']}"
            
            # Get embedding
            embedding = embed_text(triple_text)
            training_embeddings.append(embedding)
            
            # Limit training data to avoid memory issues
            if len(training_embeddings) >= 1000:
                break
        
        if len(training_embeddings) < 100:
            logger.warning(f"Only {len(training_embeddings)} embeddings available, using random data instead")
            return False
        
        # Convert to numpy array and train
        training_data = np.array(training_embeddings, dtype=np.float32)
        logger.info(f"Training FAISS index with {len(training_data)} real embeddings")
        index.train(training_data)
        
        return True
        
    except Exception as e:
        logger.error(f"Error training with real data: {e}")
        return False

def rebuild_index():
    """
    Rebuild the FAISS index from scratch.
    This retrains quantizers and optimizes the index.
    """
    logger.info("Rebuilding FAISS index from scratch")
    
    # Get index paths
    index_path = config.FAISS_INDEX_PATH
    id_map_path = f"{index_path}.ids"
    
    # Check if existing index exists
    if not os.path.exists(index_path) or not os.path.exists(id_map_path):
        logger.error("Cannot rebuild: No existing index found")
        return False
    
    try:
        # Load existing index
        index = faiss.read_index(index_path)
        logger.info(f"Loaded existing index with {index.ntotal} vectors")
        
        # Load ID map
        id_map = np.load(id_map_path, allow_pickle=True).item()
        logger.info(f"Loaded ID map with {len(id_map)} entries")
        
        if index.ntotal == 0:
            logger.warning("Index is empty, nothing to rebuild")
            return False
            
        # Extract all vectors and IDs
        all_ids = np.array(list(id_map.keys()), dtype=np.int64)
        all_vectors = np.zeros((len(all_ids), index.d), dtype=np.float32)
        
        for i, idx in enumerate(all_ids):
            index.reconstruct(int(idx), all_vectors[i])
        
        # Create new index with optimized parameters
        dim = index.d
        quantizer = faiss.IndexFlatL2(dim)
        nlist = min(4096, max(100, index.ntotal // 100))  # Scale nlist with data size
        m = 16  # Number of subquantizers
        nbits = 8  # Bits per subquantizer
        
        logger.info(f"Creating new IndexIVFPQ with nlist={nlist}, m={m}, nbits={nbits}")
        new_index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        
        # Train with all vectors
        logger.info("Training new index with all vectors")
        new_index.train(all_vectors)
        
        # Wrap with IDMap
        new_index = faiss.IndexIDMap2(new_index)
        
        # Add all vectors with IDs
        logger.info(f"Adding {len(all_ids)} vectors to new index")
        new_index.add_with_ids(all_vectors, all_ids)
        
        # Create backup of old index
        backup_path = f"{index_path}.{int(time.time())}.bak"
        faiss.write_index(index, backup_path)
        logger.info(f"Created backup of old index at {backup_path}")
        
        # Save new index
        faiss.write_index(new_index, index_path)
        logger.info(f"Saved rebuilt index with {new_index.ntotal} vectors")
        
        # ID map remains the same, no need to update
        
        return True
        
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Merge or rebuild FAISS index")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index completely (retrains quantizers)")
    
    args = parser.parse_args()
    
    if args.rebuild:
        success = rebuild_index()
        if success:
            logger.info("Index rebuild completed successfully")
        else:
            logger.error("Index rebuild failed")
            return 1
    else:
        # Merge staged embeddings
        added = merge_staged_embeddings()
        logger.info(f"Merged {added} vectors into FAISS index")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())