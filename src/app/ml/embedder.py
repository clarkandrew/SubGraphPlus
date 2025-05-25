import numpy as np
import logging
import os
from typing import Dict, List, Union
from functools import lru_cache

from app.config import config, OPENAI_API_KEY, USE_MLX, MLX_EMBEDDING_MODEL, MLX_EMBEDDING_MODEL_PATH, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Check for MLX availability and user preference
MLX_AVAILABLE = False
if USE_MLX:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        MLX_AVAILABLE = True
        logger.info("MLX is available and enabled via USE_MLX=true")
    except ImportError:
        logger.warning("MLX requested via USE_MLX=true but MLX not available. Install with: pip install mlx")
        logger.warning("Falling back to non-MLX backends")
else:
    logger.info("MLX disabled via USE_MLX=false or not set")

# Check for Hugging Face transformers
HF_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    HF_AVAILABLE = True
    logger.info("Sentence Transformers is available and will be used for HF embedding if selected")
except ImportError:
    logger.warning("Sentence Transformers not available, will not be able to use HF embedding")

# Check for OpenAI
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    if OPENAI_API_KEY:
        OPENAI_AVAILABLE = True
        logger.info("OpenAI API is available and will be used for embedding if selected")
    else:
        logger.warning("OpenAI API key not set, will not be able to use OpenAI embedding")
except ImportError:
    logger.warning("OpenAI package not installed, will not be able to use OpenAI embedding")


# Cache for embedding models
@lru_cache(maxsize=1)
def get_hf_model():
    """Get or initialize the Hugging Face model"""
    if not HF_AVAILABLE:
        raise ImportError("Sentence Transformers not available")
    model_name = EMBEDDING_MODEL
    logger.info(f"Loading HuggingFace embedding model: {model_name}")
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_mlx_model():
    """Get or initialize the MLX embedding model"""
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available or not enabled. Set USE_MLX=true and install MLX.")
    
    # Try to load MLX embedding model
    try:
        # Import MLX sentence transformers if available
        from mlx_embedding import SentenceTransformer as MLXSentenceTransformer
        
        model_path = MLX_EMBEDDING_MODEL_PATH or MLX_EMBEDDING_MODEL
        logger.info(f"Loading MLX embedding model: {model_path}")
        return MLXSentenceTransformer(model_path)
    except ImportError:
        logger.warning("MLX sentence transformers not available, using placeholder")
        # Return a placeholder that can at least generate consistent embeddings
        return None


@lru_cache(maxsize=1)
def get_openai_client():
    """Get or initialize the OpenAI client"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available")
    return OpenAI(api_key=OPENAI_API_KEY)


def mlx_embed(text: str) -> np.ndarray:
    """Embed text using MLX"""
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available or not enabled. Set USE_MLX=true and install MLX.")
    
    model = get_mlx_model()
    
    if model is not None:
        # Use actual MLX model if available
        try:
            logger.info(f"Embedding with MLX model: '{text[:50]}...'")
            embedding = model.encode(text, convert_to_numpy=True)
            embedding = np.array(embedding).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            logger.warning(f"MLX model failed: {e}, falling back to deterministic embedding")
    
    # Fallback to deterministic embedding for development/testing
    logger.info(f"Embedding with MLX (deterministic fallback): '{text[:50]}...'")
    
    # Create deterministic embedding based on text hash
    import hashlib
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    np.random.seed(hash_val)
    
    # Generate a 384-dim embedding vector (typical for small embedding models)
    embedding = np.random.normal(0, 1, 384).astype(np.float32)
    # Normalize to unit length
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding


def huggingface_embed(text: str) -> np.ndarray:
    """Embed text using Hugging Face SentenceTransformers"""
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
    """Embed text using OpenAI API"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available")
    
    logger.info(f"Embedding with OpenAI: '{text[:50]}...'")
    client = get_openai_client()
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = np.array(response.data[0].embedding).astype(np.float32)
        return embedding
    except Exception as e:
        logger.error(f"OpenAI embedding error: {e}")
        # Return zeros as fallback
        return np.zeros(1536).astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    """Embed text using the configured backend"""
    if not text:
        # Return zero vector with appropriate dimension based on backend
        dimensions = {
            "mlx": 384,
            "hf": 384,
            "openai": 1536
        }
        return np.zeros(dimensions.get(config.MODEL_BACKEND, 384)).astype(np.float32)
    
    try:
        if config.MODEL_BACKEND == "mlx":
            return mlx_embed(text)
        elif config.MODEL_BACKEND == "hf":
            return huggingface_embed(text)
        elif config.MODEL_BACKEND == "openai":
            return openai_embed(text)
        else:
            logger.error(f"Unknown model backend: {config.MODEL_BACKEND}")
            # Default to HF if available
            return huggingface_embed(text) if HF_AVAILABLE else np.zeros(384).astype(np.float32)
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        # Return zeros as fallback
        if config.MODEL_BACKEND == "openai":
            return np.zeros(1536).astype(np.float32)
        else:
            return np.zeros(384).astype(np.float32)


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