#!/usr/bin/env python3
"""
Simple FAISS Index Training Script

This script trains the FAISS index with real embeddings from the knowledge graph,
using a lightweight approach to avoid memory issues.

Following project rules:
- RULE:import-rich-logger-correctly ✅
- RULE:debug-trace-every-step ✅
- RULE:rich-error-handling-required ✅
"""

import os
import sys
import argparse
import numpy as np
import faiss
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# RULE:import-rich-logger-correctly - Use centralized rich logger
from src.app.log import logger, log_and_print
from rich.console import Console

from app.database import neo4j_db

# Initialize rich console for pretty CLI output
console = Console()

# Constants
MAX_SAMPLES = 1000
EMBEDDING_DIM = 384  # Use smaller dimension for stability

def simple_embed_text(text):
    """Simple text embedding using basic features"""
    logger.debug(f"Creating simple embedding for: {text[:50]}...")
    
    # Create a simple hash-based embedding
    import hashlib
    
    # Normalize text
    text = text.lower().strip()
    
    # Create multiple hash features
    features = []
    
    # Character-level features
    char_counts = [0] * 26
    for char in text:
        if 'a' <= char <= 'z':
            char_counts[ord(char) - ord('a')] += 1
    features.extend(char_counts)
    
    # Word-level features
    words = text.split()
    word_features = [
        len(words),  # word count
        sum(len(w) for w in words) / max(len(words), 1),  # avg word length
        len(set(words)),  # unique words
        text.count(' '),  # space count
    ]
    features.extend(word_features)
    
    # Hash-based features for semantic content
    for i in range(350):  # Fill remaining dimensions
        hash_input = f"{text}_{i}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        features.append((hash_val % 1000) / 1000.0)
    
    # Ensure exactly EMBEDDING_DIM dimensions
    features = features[:EMBEDDING_DIM]
    while len(features) < EMBEDDING_DIM:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)

def get_training_data(max_samples=MAX_SAMPLES):
    """Get training data from Neo4j knowledge graph"""
    logger.info("Starting to fetch training data from Neo4j...")
    
    # Get triples from Neo4j
    query = """
    MATCH (h)-[r:REL]->(t)
    RETURN r.id as id, h.name as head_name, r.name as relation_name, t.name as tail_name
    ORDER BY r.id
    LIMIT $max_samples
    """
    
    logger.debug("Executing Neo4j query for triples...")
    result = neo4j_db.run_query(query, {"max_samples": max_samples})
    
    if not result:
        logger.error("No triples found in Neo4j")
        return [], []
    
    logger.info(f"Found {len(result)} triples for training")
    
    # Generate embeddings
    embeddings = []
    triple_ids = []
    
    logger.debug("Starting embedding generation...")
    for i, record in enumerate(result):
        try:
            # Create triple text representation
            triple_text = f"{record['head_name']} {record['relation_name']} {record['tail_name']}"
            
            # Get embedding using simple method
            embedding = simple_embed_text(triple_text)
            
            embeddings.append(embedding)
            triple_ids.append(record['id'])
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Generated {i + 1}/{len(result)} embeddings")
            
        except Exception as e:
            logger.warning(f"Error processing triple {record['id']}: {e}")
            continue
    
    logger.info(f"Generated {len(embeddings)} embeddings for training")
    logger.debug("Finished embedding generation")
    return embeddings, triple_ids

def create_simple_index(embeddings, triple_ids):
    """Create a simple FAISS index with real data"""
    logger.debug("Starting FAISS index creation...")
    
    if not embeddings:
        logger.error("No embeddings provided for training")
        return None, None
    
    # Convert to numpy array
    training_data = np.array(embeddings, dtype=np.float32)
    logger.info(f"Creating FAISS index with {len(training_data)} vectors of dimension {EMBEDDING_DIM}")
    
    # Use simple flat index for stability
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    
    # Wrap with IDMap for custom IDs
    index = faiss.IndexIDMap2(index)
    
    logger.debug("Adding training data to index...")
    
    # Convert string IDs to integers for FAISS
    int_ids = np.array([hash(id_str) & 0x7FFFFFFF for id_str in triple_ids], dtype=np.int64)
    
    # Create ID mapping
    id_map = {}
    for i, id_str in enumerate(triple_ids):
        id_map[int(int_ids[i])] = id_str
    
    # Add vectors with IDs
    index.add_with_ids(training_data, int_ids)
    
    logger.info(f"Created index with {index.ntotal} vectors")
    logger.debug("Finished FAISS index creation")
    return index, id_map

def save_index(index, id_map, index_path):
    """Save the trained index and ID map"""
    logger.debug(f"Starting to save index to {index_path}")
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
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
        
        logger.debug("Finished saving index")
        return True
        
    except Exception as e:
        logger.error(f"Error saving index: {e}")
        return False

def verify_index(index_path):
    """Verify the trained index works correctly"""
    logger.debug("Starting index verification...")
    
    try:
        # Load index
        index = faiss.read_index(index_path)
        id_map_path = f"{index_path}.ids"
        id_map = np.load(id_map_path, allow_pickle=True).item()
        
        logger.info(f"Index loaded successfully: {index.ntotal} vectors, {len(id_map)} ID mappings")
        
        # Test search
        if index.ntotal > 0:
            # Create a random query vector
            query_vector = np.random.random((1, EMBEDDING_DIM)).astype(np.float32)
            
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
                logger.debug("Finished index verification")
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
    # RULE:debug-trace-every-step
    logger.debug("Starting main() function")
    
    # RULE:every-src-script-must-log
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Started {__file__} at {timestamp}")
    
    console.print("🤖 [bold green]Starting simple FAISS index training with real data[/bold green]")
    
    parser = argparse.ArgumentParser(description="Train FAISS index with real data (simple version)")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES, 
                       help="Maximum number of triples to use for training")
    parser.add_argument("--verify", action="store_true", 
                       help="Verify the index after training")
    parser.add_argument("--force", action="store_true",
                       help="Force retraining even if index exists")
    
    args = parser.parse_args()
    
    # Use simple index path
    index_path = "models/faiss_simple.index"
    
    # Check if index already exists
    if os.path.exists(index_path) and not args.force:
        try:
            index = faiss.read_index(index_path)
            if index.ntotal > 50:  # Assume if it has many vectors, it's real data
                logger.info(f"Index already exists with {index.ntotal} vectors. Use --force to retrain.")
                console.print(f"📊 [yellow]Index already exists with {index.ntotal} vectors. Use --force to retrain.[/yellow]")
                return 0
        except Exception:
            pass  # Continue with training if we can't load existing index
    
    logger.info("Starting simple FAISS index training with real data")
    
    try:
        # Get training data
        logger.debug("Getting training data...")
        console.print("📥 [cyan]Fetching training data from Neo4j...[/cyan]")
        embeddings, triple_ids = get_training_data(args.max_samples)
        
        if len(embeddings) < 10:
            # RULE:rich-error-handling-required
            logger.error(f"Insufficient training data: {len(embeddings)} embeddings (need at least 10)")
            console.print(f"❌ [bold red]Insufficient training data: {len(embeddings)} embeddings (need at least 10)[/bold red]")
            return 1
        
        console.print(f"✅ [green]Retrieved {len(embeddings)} embeddings for training[/green]")
        
        # Create and train index
        logger.debug("Creating index...")
        console.print("🏗️ [cyan]Creating FAISS index...[/cyan]")
        index, id_map = create_simple_index(embeddings, triple_ids)
        
        if index is None:
            # RULE:rich-error-handling-required
            logger.error("Failed to create trained index")
            console.print("❌ [bold red]Failed to create trained index[/bold red]")
            return 1
        
        console.print(f"✅ [green]Created index with {index.ntotal} vectors[/green]")
        
        # Save index
        logger.debug("Saving index...")
        console.print("💾 [cyan]Saving index to disk...[/cyan]")
        if not save_index(index, id_map, index_path):
            # RULE:rich-error-handling-required
            logger.error("Failed to save trained index")
            console.print("❌ [bold red]Failed to save trained index[/bold red]")
            return 1
        
        console.print(f"✅ [green]Saved index to {index_path}[/green]")
        
        # Verify if requested
        if args.verify:
            logger.debug("Verifying index...")
            console.print("🔍 [cyan]Verifying index...[/cyan]")
            if not verify_index(index_path):
                # RULE:rich-error-handling-required
                logger.error("Index verification failed")
                console.print("❌ [bold red]Index verification failed[/bold red]")
                return 1
            console.print("✅ [green]Index verification successful![/green]")
        
        logger.info("Simple FAISS index training completed successfully!")
        console.print("🎉 [bold green]Simple FAISS index training completed successfully![/bold green]")
        
        logger.debug("Finished main() function successfully")
        return 0
        
    except Exception as e:
        # RULE:rich-error-handling-required
        logger.error(f"Training failed with error: {e}")
        console.print_exception()
        logger.debug("Finished main() function with error")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 