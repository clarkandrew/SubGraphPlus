import os
import logging
import time
import sqlite3
import numpy as np
import uuid
import sys
from pathlib import Path
import argparse

from app.database import sqlite_db, neo4j_db
from app.ml.embedder import embed_text
from app.entity_typing import detect_entity_type, batch_detect_entity_types
from app.utils.triple_extraction import process_rebel_output, batch_process_texts

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'ingest_worker.log'))
    ]
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def process_batch():
    """Process a batch of pending triples from staging table"""
    # Get a batch of pending triples
    pending_triples = sqlite_db.fetchall(
        "SELECT id, h_text, r_text, t_text FROM staging_triples WHERE status = 'pending' ORDER BY id LIMIT ?",
        (BATCH_SIZE,)
    )
    
    if not pending_triples:
        logger.info("No pending triples to process")
        return 0
    
    logger.info(f"Processing batch of {len(pending_triples)} triples")
    
    # Lists to track results
    processed_ids = []
    errored_ids = []
    pending_faiss_embeddings = []
    
    # Process each triple
    for triple in pending_triples:
        triple_id = triple['id']
        head_text = triple['h_text']
        relation_text = triple['r_text']
        tail_text = triple['t_text']
        
        # Try to process with retries
        success = False
        errors = []
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Process in Neo4j transaction
                rel_id = ingest_triple_to_neo4j(head_text, relation_text, tail_text)
                
                if rel_id:
                    # Add to pending FAISS embeddings
                    triple_text = f"{head_text} {relation_text} {tail_text}"
                    embedding = embed_text(triple_text)
                    pending_faiss_embeddings.append((rel_id, embedding))
                    
                    processed_ids.append(triple_id)
                    success = True
                    break
                else:
                    errors.append(f"Failed to get relation ID (attempt {attempt})")
            except Exception as e:
                errors.append(f"Attempt {attempt}: {str(e)}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        
        if not success:
            logger.error(f"Failed to process triple ID {triple_id} after {MAX_RETRIES} attempts: {'; '.join(errors)}")
            errored_ids.append(triple_id)
    
    # Update status in SQLite for processed triples
    if processed_ids:
        placeholders = ','.join(['?'] * len(processed_ids))
        sqlite_db.execute(
            f"UPDATE staging_triples SET status = 'processed', processed_at = CURRENT_TIMESTAMP WHERE id IN ({placeholders})",
            processed_ids
        )
        logger.info(f"Marked {len(processed_ids)} triples as processed")
    
    # Update status in SQLite for errored triples
    if errored_ids:
        placeholders = ','.join(['?'] * len(errored_ids))
        sqlite_db.execute(
            f"UPDATE staging_triples SET status = 'error', processed_at = CURRENT_TIMESTAMP WHERE id IN ({placeholders})",
            errored_ids
        )
        logger.info(f"Marked {len(errored_ids)} triples as error")
        
        # Write failed triples to dead letter queue
        write_to_dead_letter_queue(errored_ids)
    
    # Store embeddings for FAISS
    if pending_faiss_embeddings:
        store_faiss_embeddings(pending_faiss_embeddings)
    
    return len(processed_ids)

def process_batch_optimized():
    """
    Optimized batch processing using the new entity typing approach
    Batches entity type detection for better performance
    """
    # Get a batch of pending triples
    pending_triples = sqlite_db.fetchall(
        "SELECT id, h_text, r_text, t_text FROM staging_triples WHERE status = 'pending' ORDER BY id LIMIT ?",
        (BATCH_SIZE,)
    )
    
    if not pending_triples:
        logger.info("No pending triples to process")
        return 0
    
    logger.info(f"Processing optimized batch of {len(pending_triples)} triples")
    
    # Extract all unique entity mentions for batch typing
    all_entities = set()
    for triple in pending_triples:
        all_entities.add(triple['h_text'])
        all_entities.add(triple['t_text'])
    
    # Batch detect entity types for all entities
    logger.debug(f"Batch typing {len(all_entities)} unique entities")
    entity_types = batch_detect_entity_types(list(all_entities))
    
    # Lists to track results
    processed_ids = []
    errored_ids = []
    pending_faiss_embeddings = []
    
    # Process each triple with pre-computed entity types
    for triple in pending_triples:
        triple_id = triple['id']
        head_text = triple['h_text']
        relation_text = triple['r_text']
        tail_text = triple['t_text']
        
        # Get pre-computed entity types
        head_type = entity_types.get(head_text, "Entity")
        tail_type = entity_types.get(tail_text, "Entity")
        
        # Try to process with retries
        success = False
        errors = []
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Process in Neo4j transaction with pre-computed types
                rel_id = ingest_triple_to_neo4j_optimized(
                    head_text, relation_text, tail_text, head_type, tail_type
                )
                
                if rel_id:
                    # Add to pending FAISS embeddings
                    triple_text = f"{head_text} {relation_text} {tail_text}"
                    embedding = embed_text(triple_text)
                    pending_faiss_embeddings.append((rel_id, embedding))
                    
                    processed_ids.append(triple_id)
                    success = True
                    break
                else:
                    errors.append(f"Failed to get relation ID (attempt {attempt})")
            except Exception as e:
                errors.append(f"Attempt {attempt}: {str(e)}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        
        if not success:
            logger.error(f"Failed to process triple ID {triple_id} after {MAX_RETRIES} attempts: {'; '.join(errors)}")
            errored_ids.append(triple_id)
    
    # Update status in SQLite for processed triples
    if processed_ids:
        placeholders = ','.join(['?'] * len(processed_ids))
        sqlite_db.execute(
            f"UPDATE staging_triples SET status = 'processed', processed_at = CURRENT_TIMESTAMP WHERE id IN ({placeholders})",
            processed_ids
        )
        logger.info(f"Marked {len(processed_ids)} triples as processed")
    
    # Update status in SQLite for errored triples
    if errored_ids:
        placeholders = ','.join(['?'] * len(errored_ids))
        sqlite_db.execute(
            f"UPDATE staging_triples SET status = 'error', processed_at = CURRENT_TIMESTAMP WHERE id IN ({placeholders})",
            errored_ids
        )
        logger.info(f"Marked {len(errored_ids)} triples as error")
        
        # Write failed triples to dead letter queue
        write_to_dead_letter_queue(errored_ids)
    
    # Store embeddings for FAISS
    if pending_faiss_embeddings:
        store_faiss_embeddings(pending_faiss_embeddings)
    
    return len(processed_ids)

def ingest_triple_to_neo4j(head_text, relation_text, tail_text):
    """
    Ingest a triple into Neo4j.
    Returns the relation ID if successful, None otherwise.
    """
    # Generate a transaction function
    def _tx_function(tx):
        # Create head entity if it doesn't exist
        result = tx.run("""
        MERGE (h:Entity {name: $head_text})
        ON CREATE SET h.id = $head_id, h.type = $head_type
        RETURN h.id as head_id
        """, head_text=head_text, head_id=str(uuid.uuid4()), head_type=detect_entity_type(head_text))
        head_id = result.single()["head_id"]
        
        # Create tail entity if it doesn't exist
        result = tx.run("""
        MERGE (t:Entity {name: $tail_text})
        ON CREATE SET t.id = $tail_id, t.type = $tail_type
        RETURN t.id as tail_id
        """, tail_text=tail_text, tail_id=str(uuid.uuid4()), tail_type=detect_entity_type(tail_text))
        tail_id = result.single()["tail_id"]
        
        # Create relationship if it doesn't exist
        result = tx.run("""
        MATCH (h:Entity {id: $head_id})
        MATCH (t:Entity {id: $tail_id})
        MERGE (h)-[r:REL {name: $relation_text}]->(t)
        ON CREATE SET r.id = $rel_id
        RETURN r.id as rel_id
        """, head_id=head_id, tail_id=tail_id, relation_text=relation_text, rel_id=str(uuid.uuid4()))
        rel_id = result.single()["rel_id"]
        
        return rel_id
    
    try:
        # Run the transaction
        rel_id = neo4j_db.run_transaction(_tx_function)
        return rel_id
    except Exception as e:
        logger.error(f"Error in Neo4j transaction: {str(e)}")
        return None

def ingest_triple_to_neo4j_optimized(head_text, relation_text, tail_text, head_type, tail_type):
    """
    Optimized version that uses pre-computed entity types
    Returns the relation ID if successful, None otherwise.
    """
    # Generate a transaction function
    def _tx_function(tx):
        # Create head entity if it doesn't exist
        result = tx.run("""
        MERGE (h:Entity {name: $head_text})
        ON CREATE SET h.id = $head_id, h.type = $head_type
        RETURN h.id as head_id
        """, head_text=head_text, head_id=str(uuid.uuid4()), head_type=head_type)
        head_id = result.single()["head_id"]
        
        # Create tail entity if it doesn't exist
        result = tx.run("""
        MERGE (t:Entity {name: $tail_text})
        ON CREATE SET t.id = $tail_id, t.type = $tail_type
        RETURN t.id as tail_id
        """, tail_text=tail_text, tail_id=str(uuid.uuid4()), tail_type=tail_type)
        tail_id = result.single()["tail_id"]
        
        # Create relationship if it doesn't exist
        result = tx.run("""
        MATCH (h:Entity {id: $head_id})
        MATCH (t:Entity {id: $tail_id})
        MERGE (h)-[r:REL {name: $relation_text}]->(t)
        ON CREATE SET r.id = $rel_id
        RETURN r.id as rel_id
        """, head_id=head_id, tail_id=tail_id, relation_text=relation_text, rel_id=str(uuid.uuid4()))
        rel_id = result.single()["rel_id"]
        
        return rel_id
    
    try:
        # Run the transaction
        rel_id = neo4j_db.run_transaction(_tx_function)
        return rel_id
    except Exception as e:
        logger.error(f"Error in Neo4j transaction: {str(e)}")
        return None

def store_faiss_embeddings(embeddings):
    """
    Store embeddings for FAISS indexing.
    embeddings is a list of (rel_id, embedding) tuples.
    """
    if not embeddings:
        return
    
    # Create staging directory if it doesn't exist
    staging_dir = Path("data/faiss_staging")
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract IDs and vectors
    rel_ids = [e[0] for e in embeddings]
    vectors = np.array([e[1] for e in embeddings], dtype=np.float32)
    
    # Generate timestamp for unique filenames
    timestamp = int(time.time())
    
    # Save vectors and IDs to files
    vectors_file = staging_dir / f"batch_{timestamp}.npy"
    ids_file = staging_dir / f"batch_{timestamp}_ids.npy"
    
    np.save(vectors_file, vectors)
    np.save(ids_file, rel_ids)
    
    logger.info(f"Saved {len(embeddings)} embeddings to FAISS staging: {vectors_file}")

def write_to_dead_letter_queue(error_ids):
    """Write failed triple IDs to dead letter queue for later investigation"""
    if not error_ids:
        return
    
    # Get the failed triples
    placeholders = ','.join(['?'] * len(error_ids))
    failed_triples = sqlite_db.fetchall(
        f"SELECT id, h_text, r_text, t_text, status FROM staging_triples WHERE id IN ({placeholders})",
        error_ids
    )
    
    # Create dead letter queue directory if it doesn't exist
    dlq_dir = Path("data/dead_letter_queue")
    dlq_dir.mkdir(parents=True, exist_ok=True)
    
    # Write to CSV file
    import csv
    timestamp = int(time.time())
    dlq_file = dlq_dir / f"failed_triples_{timestamp}.csv"
    
    with open(dlq_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'head', 'relation', 'tail', 'status'])
        for triple in failed_triples:
            writer.writerow([
                triple['id'], 
                triple['h_text'], 
                triple['r_text'], 
                triple['t_text'],
                triple['status']
            ])
    
    logger.info(f"Wrote {len(failed_triples)} failed triples to dead letter queue: {dlq_file}")

def main():
    """Main worker function"""
    global BATCH_SIZE
    
    parser = argparse.ArgumentParser(description="Process pending triples from staging to Neo4j and FAISS")
    parser.add_argument("--process-all", action="store_true", help="Process all pending triples")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size for processing (default: {BATCH_SIZE})")
    parser.add_argument("--max-batches", type=int, help="Maximum number of batches to process")
    parser.add_argument("--use-optimized", action="store_true", default=True, help="Use optimized batch processing with schema-first entity typing (default: True)")
    parser.add_argument("--legacy-mode", action="store_true", help="Use legacy processing mode (disables optimizations)")
    
    args = parser.parse_args()
    
    # Update batch size if specified
    BATCH_SIZE = args.batch_size
    
    # Determine which processing function to use
    if args.legacy_mode:
        process_function = process_batch
        logger.info("Using legacy processing mode")
    else:
        process_function = process_batch_optimized
        logger.info("Using optimized processing mode with schema-first entity typing")
    
    logger.info("Starting ingest worker")
    
    # Test embedder initialization early to catch model loading issues
    try:
        logger.info("Initializing embedder (this may take time on first run to download models)...")
        test_embedding = embed_text("test")
        if test_embedding is None or len(test_embedding) == 0:
            logger.error("Embedder test failed - got empty result")
            return 1
        logger.info(f"Embedder initialized successfully, embedding dimension: {len(test_embedding)}")
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {str(e)}")
        logger.error("Cannot proceed without working embedder")
        return 1
    
    # Test entity typing system
    try:
        logger.info("Testing entity typing system...")
        test_type = detect_entity_type("Jesus")
        logger.info(f"Entity typing test successful: 'Jesus' -> {test_type}")
    except Exception as e:
        logger.error(f"Failed to initialize entity typing: {str(e)}")
        logger.error("Cannot proceed without working entity typing")
        return 1
    
    try:
        # Process batches until there are no more pending triples
        total_processed = 0
        batch_count = 0
        
        while True:
            start_time = time.time()
            processed = process_function()
            elapsed = time.time() - start_time
            batch_count += 1
            
            if processed == 0:
                # No more triples to process
                logger.info("No more pending triples to process")
                break
                
            total_processed += processed
            logger.info(f"Processed batch {batch_count} of {processed} triples in {elapsed:.2f} seconds")
            
            # Check if we've reached max batches
            if args.max_batches and batch_count >= args.max_batches:
                logger.info(f"Reached maximum batch limit of {args.max_batches}")
                break
            
            # Short delay between batches
            time.sleep(0.5)
        
        logger.info(f"Ingest worker completed successfully, processed {total_processed} triples in {batch_count} batches")
    
    except KeyboardInterrupt:
        logger.info("Ingest worker interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Ingest worker failed: {str(e)}")
        return 1
    
    logger.info("Ingest worker exiting normally")
    return 0

if __name__ == "__main__":
    sys.exit(main())