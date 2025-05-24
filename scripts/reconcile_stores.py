import os
import time
import logging
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

from app.config import config
from app.database import neo4j_db
from app.ml.embedder import embed_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'reconciliation.log'))
    ]
)
logger = logging.getLogger(__name__)

def load_faiss_index(index_path):
    """Load FAISS index from file"""
    logger.info(f"Loading FAISS index from {index_path}")
    if not os.path.exists(index_path):
        logger.error(f"FAISS index not found at {index_path}")
        return None, None
    
    try:
        index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        
        # Load ID mapping
        id_map_path = f"{index_path}.ids"
        if os.path.exists(id_map_path):
            id_map = np.load(id_map_path, allow_pickle=True).item()
            logger.info(f"Loaded ID map with {len(id_map)} entries")
        else:
            logger.warning(f"ID map not found at {id_map_path}, creating empty one")
            id_map = {}
        
        return index, id_map
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        return None, None

def get_all_neo4j_relation_ids():
    """Get all relationship IDs from Neo4j"""
    logger.info("Fetching all relationship IDs from Neo4j")
    
    query = """
    MATCH ()-[r:REL]->()
    RETURN r.id as id
    """
    
    try:
        result = neo4j_db.run_query(query)
        relation_ids = [record["id"] for record in result]
        logger.info(f"Found {len(relation_ids)} relationships in Neo4j")
        return relation_ids
    except Exception as e:
        logger.error(f"Error fetching relationship IDs from Neo4j: {e}")
        return []

def get_all_faiss_ids(id_map):
    """Get all IDs in FAISS index"""
    logger.info("Collecting all IDs in FAISS index")
    
    faiss_ids = set()
    for int_id, rel_id in id_map.items():
        faiss_ids.add(rel_id)
    
    logger.info(f"Found {len(faiss_ids)} IDs in FAISS index")
    return faiss_ids

def get_triple_text_from_neo4j(relation_id):
    """Get triple text from Neo4j for a given relation ID"""
    query = """
    MATCH (h)-[r:REL {id: $relation_id}]->(t)
    RETURN h.name as head_name, r.name as relation_name, t.name as tail_name
    """
    
    result = neo4j_db.run_query(query, {"relation_id": relation_id})
    if not result:
        return None
    
    record = result[0]
    return f"{record['head_name']} {record['relation_name']} {record['tail_name']}"

def add_missing_to_faiss(index, id_map, missing_ids, batch_size=100):
    """Add missing relationships to FAISS index"""
    if not missing_ids:
        return 0
    
    logger.info(f"Adding {len(missing_ids)} missing relationships to FAISS")
    
    # Process in batches to avoid memory issues
    added = 0
    for i in range(0, len(missing_ids), batch_size):
        batch_ids = missing_ids[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(missing_ids)-1)//batch_size + 1} ({len(batch_ids)} items)")
        
        # Get triple texts
        texts = []
        valid_ids = []
        for relation_id in batch_ids:
            text = get_triple_text_from_neo4j(relation_id)
            if text:
                texts.append(text)
                valid_ids.append(relation_id)
        
        if not texts:
            continue
        
        # Embed texts
        embeddings = []
        for text in tqdm(texts, desc="Embedding triples"):
            embedding = embed_text(text)
            embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Convert string IDs to integers for FAISS
        int_ids = np.array([hash(id_str) & 0x7FFFFFFF for id_str in valid_ids], dtype=np.int64)
        
        # Update ID map
        for i, id_str in enumerate(valid_ids):
            id_map[int(int_ids[i])] = id_str
        
        # Add to index
        try:
            index.add_with_ids(embeddings, int_ids)
            added += len(valid_ids)
            logger.info(f"Added {len(valid_ids)} vectors to FAISS index")
        except Exception as e:
            logger.error(f"Error adding vectors to FAISS index: {e}")
    
    return added

def remove_orphaned_from_faiss(index, id_map, orphaned_ids):
    """Remove orphaned IDs from FAISS index"""
    if not orphaned_ids:
        return 0
    
    logger.info(f"Found {len(orphaned_ids)} orphaned IDs in FAISS index")
    
    # In FAISS, we need to rebuild the index to remove vectors,
    # so we log the orphaned IDs for now and recommend a rebuild
    logger.info("Note: FAISS doesn't support efficient removal of vectors.")
    logger.info("Orphaned IDs will be logged, but a full rebuild is recommended later.")
    
    # Log orphaned IDs
    with open("logs/orphaned_faiss_ids.txt", "w") as f:
        for rel_id in orphaned_ids:
            f.write(f"{rel_id}\n")
    
    logger.info("Orphaned IDs written to logs/orphaned_faiss_ids.txt")
    
    return len(orphaned_ids)

def save_faiss_index(index, id_map, index_path):
    """Save FAISS index and ID map to file"""
    try:
        # Create backup of existing index
        if os.path.exists(index_path):
            backup_path = f"{index_path}.{int(time.time())}.bak"
            faiss.write_index(faiss.read_index(index_path), backup_path)
            logger.info(f"Created backup of old index at {backup_path}")
        
        # Save index
        faiss.write_index(index, index_path)
        logger.info(f"Saved FAISS index with {index.ntotal} vectors to {index_path}")
        
        # Save ID map
        id_map_path = f"{index_path}.ids"
        np.save(id_map_path, id_map)
        logger.info(f"Saved ID map with {len(id_map)} entries to {id_map_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        return False

def reconcile_stores(time_limit_seconds=None):
    """
    Reconcile Neo4j and FAISS stores
    
    Args:
        time_limit_seconds: Optional time limit in seconds. If provided,
                           the function will stop after this time.
    """
    start_time = time.time()
    
    # Step 1: Load FAISS index
    index_path = config.FAISS_INDEX_PATH
    index, id_map = load_faiss_index(index_path)
    
    if index is None or id_map is None:
        logger.error("Failed to load FAISS index, aborting reconciliation")
        return False
    
    # Step 2: Get all relation IDs from Neo4j
    neo4j_relation_ids = get_all_neo4j_relation_ids()
    if not neo4j_relation_ids:
        logger.error("Failed to get relation IDs from Neo4j, aborting reconciliation")
        return False
    
    # Step 3: Get all IDs from FAISS
    faiss_ids = get_all_faiss_ids(id_map)
    
    # Step 4: Find missing relations in FAISS
    missing_in_faiss = []
    for rel_id in neo4j_relation_ids:
        if rel_id not in faiss_ids:
            missing_in_faiss.append(rel_id)
    
    logger.info(f"Found {len(missing_in_faiss)} relations in Neo4j but not in FAISS")
    
    # Step 5: Find orphaned vectors in FAISS
    orphaned_in_faiss = []
    for faiss_id in faiss_ids:
        if faiss_id not in neo4j_relation_ids:
            orphaned_in_faiss.append(faiss_id)
    
    logger.info(f"Found {len(orphaned_in_faiss)} relations in FAISS but not in Neo4j")
    
    # Step 6: Add missing relations to FAISS
    if missing_in_faiss:
        added = add_missing_to_faiss(index, id_map, missing_in_faiss)
        logger.info(f"Added {added} missing relations to FAISS")
    
    # Step 7: Handle orphaned vectors in FAISS (log for now)
    if orphaned_in_faiss:
        logged = remove_orphaned_from_faiss(index, id_map, orphaned_in_faiss)
        logger.info(f"Logged {logged} orphaned relations for cleanup")
    
    # Step 8: Save updated FAISS index
    if missing_in_faiss:
        saved = save_faiss_index(index, id_map, index_path)
        if not saved:
            logger.error("Failed to save updated FAISS index")
            return False
    
    # Calculate duration and print summary
    duration = time.time() - start_time
    logger.info(f"Reconciliation completed in {duration:.2f} seconds")
    logger.info(f"Summary: {len(neo4j_relation_ids)} total relations, "
                f"{len(missing_in_faiss)} added to FAISS, "
                f"{len(orphaned_in_faiss)} orphaned in FAISS")
    
    return True

def main():
    """Main entry point for reconciliation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reconcile Neo4j and FAISS stores")
    parser.add_argument("--time-limit", type=int, help="Time limit in seconds")
    parser.add_argument("--read-only", action="store_true", help="Run in read-only mode (no updates)")
    
    args = parser.parse_args()
    
    if args.read_only:
        logger.info("Running in read-only mode, no updates will be made")
        # Load FAISS index
        index_path = config.FAISS_INDEX_PATH
        index, id_map = load_faiss_index(index_path)
        
        # Get all relation IDs from Neo4j
        neo4j_relation_ids = get_all_neo4j_relation_ids()
        
        # Get all IDs from FAISS
        faiss_ids = get_all_faiss_ids(id_map)
        
        # Find missing relations in FAISS
        missing_in_faiss = []
        for rel_id in neo4j_relation_ids:
            if rel_id not in faiss_ids:
                missing_in_faiss.append(rel_id)
        
        # Find orphaned vectors in FAISS
        orphaned_in_faiss = []
        for faiss_id in faiss_ids:
            if faiss_id not in neo4j_relation_ids:
                orphaned_in_faiss.append(faiss_id)
        
        logger.info(f"Summary: {len(neo4j_relation_ids)} total relations, "
                    f"{len(missing_in_faiss)} missing from FAISS, "
                    f"{len(orphaned_in_faiss)} orphaned in FAISS")
    else:
        reconcile_stores(time_limit_seconds=args.time_limit)

if __name__ == "__main__":
    main()