import numpy as np
import os
from typing import Dict, List, Union
from functools import lru_cache

from app.config import config, OPENAI_API_KEY, EMBEDDING_MODEL

# Add testing check at the top
TESTING = os.getenv('TESTING', '').lower() in ('1', 'true', 'yes')

# RULE:import-rich-logger-correctly - Use centralized rich logger
from ..log import logger, log_and_print
from rich.console import Console

# Initialize rich console for pretty CLI output
console = Console()

# Check for Hugging Face transformers (primary for embeddings)
# Skip during testing to prevent slow imports
HF_AVAILABLE = False
if not TESTING:
    try:
        from sentence_transformers import SentenceTransformer
        HF_AVAILABLE = True
        logger.info("Sentence Transformers is available and will be used for embeddings")
    except ImportError:
        logger.warning("Sentence Transformers not available, will not be able to use local embeddings")

# Check for OpenAI (fallback)
# Skip during testing to prevent API calls
OPENAI_AVAILABLE = False
if not TESTING:
    try:
        from openai import OpenAI
        if OPENAI_API_KEY:
            OPENAI_AVAILABLE = True
            logger.info("OpenAI API is available and will be used as embedding fallback")
        else:
            logger.warning("OpenAI API key not set, will not be able to use OpenAI embedding fallback")
    except ImportError:
        logger.warning("OpenAI package not installed, will not be able to use OpenAI embedding fallback")


# Cache for embedding models
@lru_cache(maxsize=1)
def get_hf_model():
    """Get or initialize the Hugging Face model (primary)"""
    if not HF_AVAILABLE:
        raise ImportError("Sentence Transformers not available")
    
    model_name = EMBEDDING_MODEL
    
    # Set up cache directory to models/embeddings (where user moved the models)
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'models', 'embeddings')
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info(f"Loading HuggingFace embedding model: {model_name}")
    logger.info(f"Using cache directory: {cache_dir}")
    
    return SentenceTransformer(model_name, cache_folder=cache_dir, trust_remote_code=True)


@lru_cache(maxsize=1)
def get_openai_client():
    """Get or initialize the OpenAI client (fallback)"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available")
    return OpenAI(api_key=OPENAI_API_KEY)


def huggingface_embed(text: str) -> np.ndarray:
    """Embed text using Hugging Face SentenceTransformers (primary)"""
    if not HF_AVAILABLE:
        raise ImportError("Hugging Face Sentence Transformers not available")
    
    logger.info(f"Embedding with HF: '{text[:50]}...'")
    model = get_hf_model()
    embedding = model.encode(text, convert_to_numpy=True)
    
    # Ensure it's a numpy array and normalized
    embedding = np.array(embedding).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding


def openai_embed(text: str) -> np.ndarray:
    """Embed text using OpenAI API (fallback)"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available")
    
    logger.info(f"Embedding with OpenAI (fallback): '{text[:50]}...'")
    client = get_openai_client()
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = np.array(response.data[0].embedding).astype(np.float32)
        
        # CRITICAL FIX: OpenAI embeddings are 1536-dim but we need 1024-dim to match HuggingFace
        # Truncate to first 1024 dimensions to maintain consistency
        if embedding.shape[0] > 1024:
            logger.warning(f"Truncating OpenAI embedding from {embedding.shape[0]} to 1024 dimensions for consistency")
            embedding = embedding[:1024]
        elif embedding.shape[0] < 1024:
            # Pad with zeros if somehow smaller
            logger.warning(f"Padding OpenAI embedding from {embedding.shape[0]} to 1024 dimensions")
            padding = np.zeros(1024 - embedding.shape[0], dtype=np.float32)
            embedding = np.concatenate([embedding, padding])
            
        # Renormalize after truncation/padding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
        
    except Exception as e:
        logger.error(f"OpenAI embedding error: {e}")
        # Return zeros as fallback with correct dimension
        return np.zeros(1024).astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    """Embed text using local transformers (HF primary, OpenAI fallback)"""
    if not text:
        # Return zero vector with appropriate dimension (default 1024 for gte-large-en-v1.5)
        return np.zeros(1024).astype(np.float32)
    
    # Try HuggingFace first (primary)
    if HF_AVAILABLE:
        try:
            return huggingface_embed(text)
        except Exception as e:
            logger.warning(f"HuggingFace embedding failed: {e}, trying OpenAI fallback")
    
    # Try OpenAI fallback
    if OPENAI_AVAILABLE:
        try:
            return openai_embed(text)
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}, returning zero vector")
    
    # Last resort: return zero vector
    logger.error("All embedding backends failed, returning zero vector")
    return np.zeros(1024).astype(np.float32)


def embed_batch(texts: List[str]) -> np.ndarray:
    """Embed a batch of texts"""
    embeddings = []
    for text in texts:
        embeddings.append(embed_text(text))
    return np.array(embeddings)


def health_check() -> bool:
    """Check if the embedding model is working"""
    try:
        # Try embedding a simple text
        embedding = embed_text("test")
        # Check if the embedding is a non-zero vector
        return embedding is not None and np.any(embedding)
    except Exception as e:
        logger.error(f"Embedding health check failed: {e}")
        return False