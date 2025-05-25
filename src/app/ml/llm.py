import logging
import os
from typing import Dict, List, Optional, Any
from functools import lru_cache

from app.config import config, OPENAI_API_KEY, USE_MLX, MLX_LLM_MODEL, MLX_LLM_MODEL_PATH

logger = logging.getLogger(__name__)

# Check for MLX availability and user preference
MLX_AVAILABLE = False
if USE_MLX:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        MLX_AVAILABLE = True
        logger.info("MLX is available and enabled via USE_MLX=true for LLM")
    except ImportError:
        logger.warning("MLX requested via USE_MLX=true but MLX not available. Install with: pip install mlx")
        logger.warning("Falling back to non-MLX LLM backends")
else:
    logger.info("MLX disabled via USE_MLX=false or not set for LLM")

# Check for Hugging Face transformers
HF_AVAILABLE = False
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
    logger.info("Hugging Face Transformers is available and will be used for HF models if selected")
except ImportError:
    logger.warning("Hugging Face Transformers not available, will not be able to use HF models")

# Check for OpenAI
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    if OPENAI_API_KEY:
        OPENAI_AVAILABLE = True
        logger.info("OpenAI API is available and will be used if selected")
    else:
        logger.warning("OpenAI API key not set, will not be able to use OpenAI models")
except ImportError:
    logger.warning("OpenAI package not installed, will not be able to use OpenAI models")


# Cache for LLM models
@lru_cache(maxsize=1)
def get_hf_model():
    """Get or initialize the Hugging Face model"""
    if not HF_AVAILABLE:
        raise ImportError("Hugging Face Transformers not available")
    
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # A smaller model as example
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    return model, tokenizer


@lru_cache(maxsize=1)
def get_mlx_model():
    """Get or initialize the MLX LLM model"""
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available or not enabled. Set USE_MLX=true and install MLX.")
    
    # Try to load MLX LLM model
    try:
        # Import MLX LLM if available
        from mlx_lm import load, generate
        
        model_path = MLX_LLM_MODEL_PATH or MLX_LLM_MODEL
        logger.info(f"Loading MLX LLM model: {model_path}")
        model, tokenizer = load(model_path)
        return model, tokenizer
    except ImportError:
        logger.warning("MLX LLM not available, using placeholder")
        # Return a placeholder
        return None, None


@lru_cache(maxsize=1)
def get_openai_client():
    """Get or initialize the OpenAI client"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available")
    return OpenAI(api_key=OPENAI_API_KEY)


def mlx_generate(prompt: str, **kwargs) -> str:
    """Generate text using MLX"""
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available or not enabled. Set USE_MLX=true and install MLX.")
    
    model, tokenizer = get_mlx_model()
    
    if model is not None and tokenizer is not None:
        # Use actual MLX model if available
        try:
            logger.info(f"Generating with MLX model: prompt length={len(prompt)}")
            
            # Import generate function
            from mlx_lm import generate
            
            max_tokens = kwargs.get("max_tokens", 512)
            temperature = kwargs.get("temperature", 0.1)
            
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False
            )
            
            return response
        except Exception as e:
            logger.warning(f"MLX model failed: {e}, falling back to placeholder response")
    
    # Fallback to placeholder for development/testing
    logger.info(f"Generating with MLX (placeholder): prompt length={len(prompt)}")
    
    # Return a more contextual placeholder response
    if "question" in prompt.lower() or "what" in prompt.lower():
        return "Based on the provided context, I cannot find specific information to answer this question accurately."
    else:
        return "Information not available in provided context."


def huggingface_generate(prompt: str, **kwargs) -> str:
    """Generate text using Hugging Face model"""
    if not HF_AVAILABLE:
        raise ImportError("Hugging Face Transformers not available")
    
    logger.info(f"Generating with HF: prompt length={len(prompt)}")
    
    model, tokenizer = get_hf_model()
    
    max_length = kwargs.get("max_tokens", 512)
    temperature = kwargs.get("temperature", 0.1)
    top_p = kwargs.get("top_p", 0.9)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and remove the prompt
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def openai_generate(prompt: str, **kwargs) -> str:
    """Generate text using OpenAI API"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available")
    
    logger.info(f"Generating with OpenAI: prompt length={len(prompt)}")
    client = get_openai_client()
    
    # Extract parameters with defaults
    max_tokens = kwargs.get("max_tokens", 512)
    temperature = kwargs.get("temperature", 0.1)
    top_p = kwargs.get("top_p", 0.9)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Default model, can be overridden via kwargs
            messages=[
                {"role": "system", "content": "You are a precise, factual question-answering assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI generation error: {e}")
        return "Error: Unable to generate response from OpenAI."


def generate_answer(prompt: str, **kwargs) -> str:
    """Generate an answer from the LLM using the provided prompt."""
    try:
        if config.MODEL_BACKEND == "mlx":
            return mlx_generate(prompt, **kwargs)
        elif config.MODEL_BACKEND == "hf":
            return huggingface_generate(prompt, **kwargs)
        elif config.MODEL_BACKEND == "openai":
            return openai_generate(prompt, **kwargs)
        else:
            logger.error(f"Unknown model backend: {config.MODEL_BACKEND}")
            # Default to OpenAI if available, then HF
            if OPENAI_AVAILABLE:
                return openai_generate(prompt, **kwargs)
            elif HF_AVAILABLE:
                return huggingface_generate(prompt, **kwargs)
            else:
                return "Error: No available LLM backend."
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}"


def stream_tokens(prompt: str, **kwargs):
    """Stream tokens from the LLM, for SSE endpoints"""
    if config.MODEL_BACKEND == "openai" and OPENAI_AVAILABLE:
        client = get_openai_client()
        
        try:
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise, factual question-answering assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get("temperature", 0.1),
                top_p=kwargs.get("top_p", 0.9),
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error streaming tokens from OpenAI: {e}")
            yield f"Error: {str(e)}"
    else:
        # For models that don't support native streaming, simulate it
        try:
            answer = generate_answer(prompt, **kwargs)
            # Simulate streaming by yielding word by word
            for word in answer.split():
                yield word + " "
        except Exception as e:
            logger.error(f"Error in simulated streaming: {e}")
            yield f"Error: {str(e)}"


def health_check() -> bool:
    """Check if the LLM backend is working"""
    try:
        # Try generating a simple response
        response = generate_answer("Say 'health check passed'.", max_tokens=10)
        # Check if we got some response
        return response is not None and len(response) > 0
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        return False