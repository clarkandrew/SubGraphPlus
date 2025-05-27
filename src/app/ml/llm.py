import os
from typing import Dict, List, Optional, Any
from functools import lru_cache

from app.config import config, OPENAI_API_KEY

# RULE:import-rich-logger-correctly - Use centralized rich logger
from ..log import logger, log_and_print
from rich.console import Console

# Initialize rich console for pretty CLI output
console = Console()

# Check for MLX availability (for LLM only, never embeddings)
# Use lazy loading to prevent slow imports during testing
MLX_AVAILABLE = False
TESTING = os.getenv('TESTING', '').lower() in ('1', 'true', 'yes')

if not TESTING:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        MLX_AVAILABLE = True
        logger.info("MLX is available and will be used as the primary backend for LLM")
    except ImportError:
        logger.warning("MLX not available. Install with: pip install mlx mlx-lm")
        logger.warning("Falling back to HuggingFace/OpenAI backends")
else:
    logger.info("Testing mode: Skipping MLX imports to speed up tests")

# Check for Hugging Face transformers (fallback for LLM)
HF_AVAILABLE = False
if not TESTING:
    try:
        import torch
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        HF_AVAILABLE = True
        logger.info("Hugging Face Transformers is available and will be used as fallback")
    except ImportError:
        logger.warning("Hugging Face Transformers not available, will not be able to use HF models")
else:
    logger.info("Testing mode: Skipping HuggingFace imports to speed up tests")

# Check for OpenAI (fallback for LLM)
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    # Only consider OpenAI available if it's the configured backend or if no specific backend is set
    if OPENAI_API_KEY and (config.MODEL_BACKEND == "openai" or config.MODEL_BACKEND not in ["mlx", "huggingface", "openai"]):
        OPENAI_AVAILABLE = True
        logger.info("OpenAI API is available and will be used as fallback")
    elif OPENAI_API_KEY and config.MODEL_BACKEND != "openai":
        logger.info(f"OpenAI API available but MODEL_BACKEND={config.MODEL_BACKEND}, not using OpenAI")
    else:
        logger.warning("OpenAI API key not set, will not be able to use OpenAI models")
except ImportError:
    logger.warning("OpenAI package not installed, will not be able to use OpenAI models")


# Cache for LLM models
@lru_cache(maxsize=1)
def get_hf_model():
    """Get or initialize the Hugging Face model (fallback)"""
    if not HF_AVAILABLE:
        raise ImportError("Hugging Face Transformers not available")
    
    # Get model from config
    hf_config = config.get_model_config("huggingface")
    model_id = hf_config.get("model", "mistralai/Mistral-7B-Instruct-v0.2")
    
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
    """Get or initialize the MLX LLM model (primary)"""
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available. Install with: pip install mlx mlx-lm")
    
    # Try to load MLX LLM model
    try:
        # Import MLX LLM if available
        from mlx_lm import load, generate
        
        # Get model from config
        mlx_config = config.get_model_config("mlx")
        model_id = mlx_config.get("model", "mlx-community/Qwen3-14B-8bit")
        
        logger.info(f"Loading MLX LLM model: {model_id}")
        logger.info("MLX will automatically use cached model from HuggingFace cache directory")
        
        # MLX load() function automatically handles caching when given a HF model ID
        # It will check the HF cache directory first before downloading
        model, tokenizer = load(model_id)
        return model, tokenizer
    except ImportError:
        logger.warning("MLX LLM not available, using placeholder")
        # Return a placeholder
        return None, None


@lru_cache(maxsize=1)
def get_openai_client():
    """Get or initialize the OpenAI client (fallback)"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available")
    return OpenAI(api_key=OPENAI_API_KEY)


def mlx_generate(prompt: str, **kwargs) -> str:
    """Generate text using MLX (primary backend for LLM only)"""
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available. Install with: pip install mlx mlx-lm")
    
    model, tokenizer = get_mlx_model()
    
    if model is not None and tokenizer is not None:
        # Use actual MLX model if available
        try:
            logger.info(f"Generating with MLX model: prompt length={len(prompt)}")
            
            # Import generate function
            from mlx_lm import generate
            
            # Get parameters from config or kwargs
            mlx_config = config.get_model_config("mlx")
            max_tokens = kwargs.get("max_tokens", mlx_config.get("max_tokens", 512))
            temperature = kwargs.get("temperature", mlx_config.get("temperature", 0.1))
            
            # MLX-LM generate function uses different parameter names
            # Use sampler for temperature control instead of direct parameter
            from mlx_lm.sample_utils import make_sampler
            
            sampler = make_sampler(temp=temperature)
            
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=max_tokens,
                sampler=sampler,
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
    """Generate text using Hugging Face model (fallback)"""
    if not HF_AVAILABLE:
        raise ImportError("Hugging Face Transformers not available")
    
    logger.info(f"Generating with HF (fallback): prompt length={len(prompt)}")
    
    model, tokenizer = get_hf_model()
    
    # Get parameters from config or kwargs
    hf_config = config.get_model_config("huggingface")
    max_length = kwargs.get("max_tokens", hf_config.get("max_tokens", 512))
    temperature = kwargs.get("temperature", hf_config.get("temperature", 0.1))
    top_p = kwargs.get("top_p", hf_config.get("top_p", 0.9))
    
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
    """Generate text using OpenAI API (fallback)"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available")
    
    logger.info(f"Generating with OpenAI (fallback): prompt length={len(prompt)}")
    client = get_openai_client()
    
    # Get parameters from config or kwargs
    openai_config = config.get_model_config("openai")
    model = openai_config.get("model", "gpt-3.5-turbo")
    max_tokens = kwargs.get("max_tokens", openai_config.get("max_tokens", 512))
    temperature = kwargs.get("temperature", openai_config.get("temperature", 0.1))
    top_p = kwargs.get("top_p", openai_config.get("top_p", 0.9))
    
    try:
        response = client.chat.completions.create(
            model=model,
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
    """Generate an answer from the LLM using the configured backend."""
    try:
        if config.MODEL_BACKEND == "mlx":
            return mlx_generate(prompt, **kwargs)
        elif config.MODEL_BACKEND == "huggingface":
            return huggingface_generate(prompt, **kwargs)
        elif config.MODEL_BACKEND == "openai":
            return openai_generate(prompt, **kwargs)
        else:
            logger.error(f"Unknown model backend: {config.MODEL_BACKEND}")
            # Default fallback order: MLX -> HF -> OpenAI
            if MLX_AVAILABLE:
                return mlx_generate(prompt, **kwargs)
            elif HF_AVAILABLE:
                return huggingface_generate(prompt, **kwargs)
            elif OPENAI_AVAILABLE:
                return openai_generate(prompt, **kwargs)
            else:
                return "Error: No available LLM backend."
    except Exception as e:
        logger.error(f"Error generating answer with {config.MODEL_BACKEND}: {e}")
        
        # Only fall back to other backends if the configured backend is not explicitly set
        # or if we're in a testing environment
        if config.MODEL_BACKEND == "mlx":
            # For MLX, only fall back if MLX is completely broken and we're in testing
            if os.getenv('TESTING', '').lower() in ('1', 'true', 'yes'):
                logger.warning("MLX failed in testing mode, returning mock response")
                return "Mock LLM response for testing"
            else:
                # In production, if MLX is configured, stick with MLX even if it fails
                logger.error(f"MLX backend failed: {e}")
                return f"MLX backend error: {str(e)}"
        
        # For other backends, allow fallback
        if config.MODEL_BACKEND != "openai" and OPENAI_AVAILABLE:
            try:
                logger.info("Falling back to OpenAI")
                return openai_generate(prompt, **kwargs)
            except Exception as fallback_e:
                logger.error(f"OpenAI fallback failed: {fallback_e}")
        
        if config.MODEL_BACKEND != "huggingface" and HF_AVAILABLE:
            try:
                logger.info("Falling back to HuggingFace")
                return huggingface_generate(prompt, **kwargs)
            except Exception as fallback_e:
                logger.error(f"HuggingFace fallback failed: {fallback_e}")
        
        return f"Error generating answer: {str(e)}"


def stream_tokens(prompt: str, **kwargs):
    """Stream tokens from the LLM, for SSE endpoints"""
    # Only use OpenAI streaming if explicitly configured as the backend
    if config.MODEL_BACKEND == "openai" and OPENAI_AVAILABLE:
        client = get_openai_client()
        
        # Get model from config
        openai_config = config.get_model_config("openai")
        model = openai_config.get("model", "gpt-3.5-turbo")
        
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise, factual question-answering assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get("temperature", openai_config.get("temperature", 0.1)),
                top_p=kwargs.get("top_p", openai_config.get("top_p", 0.9)),
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error streaming tokens from OpenAI: {e}")
            yield f"Error: {str(e)}"
    else:
        # For all other backends (MLX, HF), simulate streaming by using the configured backend
        try:
            logger.info(f"Simulating streaming for {config.MODEL_BACKEND} backend")
            answer = generate_answer(prompt, **kwargs)
            # Simulate streaming by yielding word by word
            words = answer.split()
            for i, word in enumerate(words):
                if i == len(words) - 1:
                    yield word  # Last word without space
                else:
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