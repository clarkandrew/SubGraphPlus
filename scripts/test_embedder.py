#!/usr/bin/env python3
"""
Test script to verify embedder works correctly
Run this before setup to catch embedding model issues early
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app.ml.embedder import embed_text

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_embedder():
    """Test the embedder initialization and basic functionality"""
    logger.info("Testing embedder initialization...")
    
    try:
        # Test basic embedding
        start_time = time.time()
        logger.info("Testing basic text embedding...")
        embedding = embed_text("This is a test sentence")
        elapsed = time.time() - start_time
        
        if embedding is None:
            logger.error("❌ Embedding returned None")
            return False
            
        if len(embedding) == 0:
            logger.error("❌ Embedding returned empty array")
            return False
            
        logger.info(f"✅ Basic embedding successful in {elapsed:.2f}s")
        logger.info(f"   Embedding shape: {embedding.shape}")
        logger.info(f"   Embedding type: {type(embedding)}")
        
        # Test multiple embeddings
        start_time = time.time()
        logger.info("Testing multiple embeddings...")
        test_texts = [
            "First test sentence",
            "Second test sentence", 
            "Third test sentence"
        ]
        
        for i, text in enumerate(test_texts):
            emb = embed_text(text)
            if emb is None or len(emb) == 0:
                logger.error(f"❌ Failed on embedding {i+1}")
                return False
                
        elapsed = time.time() - start_time
        logger.info(f"✅ Multiple embeddings successful in {elapsed:.2f}s")
        
        # Test empty string
        logger.info("Testing empty string handling...")
        empty_emb = embed_text("")
        if empty_emb is None:
            logger.error("❌ Empty string returned None")
            return False
        logger.info("✅ Empty string handled correctly")
        
        logger.info("🎉 All embedder tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedder test failed with exception: {str(e)}")
        logger.exception("Full traceback:")
        return False

def main():
    logger.info("=" * 50)
    logger.info("SubGraphRAG+ Embedder Test")
    logger.info("=" * 50)
    
    success = test_embedder()
    
    if success:
        logger.info("✅ Embedder is working correctly!")
        logger.info("You can now run the full setup script.")
        return 0
    else:
        logger.error("❌ Embedder test failed!")
        logger.error("Please check your configuration and dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 