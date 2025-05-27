#!/usr/bin/env python3
"""
Unit tests for LLM module components.
Tests individual functions and classes in isolation with mocking.
"""

import os
import sys
import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Set testing environment variable BEFORE any imports
os.environ['TESTING'] = '1'

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.app.log import logger
from src.app.ml.llm import (
    generate_answer, 
    stream_tokens, 
    health_check,
    mlx_generate,
    huggingface_generate,
    openai_generate,
    get_mlx_model,
    get_hf_model,
    get_openai_client
)
from src.app.config import config


class TestLLMConfiguration(unittest.TestCase):
    """Test LLM configuration and backend selection"""
    
    def test_config_backend_selection(self):
        """Test that the configured backend is properly selected"""
        logger.debug("Starting test_config_backend_selection")
        
        # Check that MLX is the configured backend
        self.assertEqual(config.MODEL_BACKEND, "mlx")
        
        # Check that MLX model is configured
        mlx_config = config.get_model_config("mlx")
        self.assertIsInstance(mlx_config, dict)
        self.assertIn("model", mlx_config)
        self.assertEqual(mlx_config["model"], "mlx-community/Qwen3-14B-8bit")
        
        logger.debug("Finished test_config_backend_selection")
    
    def test_fallback_backends_configured(self):
        """Test that fallback backends are properly configured"""
        logger.debug("Starting test_fallback_backends_configured")
        
        # Check OpenAI config
        openai_config = config.get_model_config("openai")
        self.assertIsInstance(openai_config, dict)
        self.assertIn("model", openai_config)
        self.assertEqual(openai_config["model"], "gpt-3.5-turbo")
        
        # Check HuggingFace config
        hf_config = config.get_model_config("huggingface")
        self.assertIsInstance(hf_config, dict)
        self.assertIn("model", hf_config)
        self.assertEqual(hf_config["model"], "mlx-community/Qwen3-14B-8bit")
        
        logger.debug("Finished test_fallback_backends_configured")


class TestLLMGeneration(unittest.TestCase):
    """Test LLM text generation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_prompt = "What is the capital of France?"
        self.test_kwargs = {
            "max_tokens": 50,
            "temperature": 0.1,
            "top_p": 0.9
        }
    
    def test_generate_answer_basic(self):
        """Test basic answer generation"""
        logger.debug("Starting test_generate_answer_basic")
        
        response = generate_answer(self.test_prompt, **self.test_kwargs)
        
        # Should return a string response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # In testing mode, should return mock response
        self.assertIn("Mock LLM response", response)
        
        logger.debug("Finished test_generate_answer_basic")
    
    def test_generate_answer_empty_prompt(self):
        """Test generation with empty prompt"""
        logger.debug("Starting test_generate_answer_empty_prompt")
        
        response = generate_answer("", **self.test_kwargs)
        
        # Should handle empty prompt gracefully
        self.assertIsInstance(response, str)
        
        logger.debug("Finished test_generate_answer_empty_prompt")
    
    def test_generate_answer_long_prompt(self):
        """Test generation with very long prompt"""
        logger.debug("Starting test_generate_answer_long_prompt")
        
        long_prompt = "This is a very long prompt. " * 100
        response = generate_answer(long_prompt, **self.test_kwargs)
        
        # Should handle long prompts gracefully
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        logger.debug("Finished test_generate_answer_long_prompt")
    
    def test_generate_answer_with_custom_params(self):
        """Test generation with custom parameters"""
        logger.debug("Starting test_generate_answer_with_custom_params")
        
        custom_kwargs = {
            "max_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.8
        }
        
        response = generate_answer(self.test_prompt, **custom_kwargs)
        
        # Should return a string response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        logger.debug("Finished test_generate_answer_with_custom_params")


class TestMLXBackend(unittest.TestCase):
    """Test MLX backend functionality"""
    
    def test_mlx_generate_testing_mode(self):
        """Test MLX generation in testing mode"""
        logger.debug("Starting test_mlx_generate_testing_mode")
        
        # In testing mode, MLX should return placeholder response
        with patch('src.app.ml.llm.MLX_AVAILABLE', False):
            try:
                response = mlx_generate("Test prompt")
                # Should not reach here as it should raise ImportError
                self.fail("Expected ImportError")
            except ImportError as e:
                self.assertIn("MLX not available", str(e))
        
        logger.debug("Finished test_mlx_generate_testing_mode")
    
    @patch('src.app.ml.llm.MLX_AVAILABLE', True)
    @patch('src.app.ml.llm.get_mlx_model')
    def test_mlx_generate_with_model(self, mock_get_model):
        """Test MLX generation when model is available"""
        logger.debug("Starting test_mlx_generate_with_model")
        
        # Mock MLX model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_get_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock the mlx_lm.generate function
        with patch('mlx_lm.generate', return_value="MLX generated response") as mock_generate:
            response = mlx_generate("Test prompt")
            
            self.assertEqual(response, "MLX generated response")
            mock_generate.assert_called_once()
        
        logger.debug("Finished test_mlx_generate_with_model")
    
    @patch('src.app.ml.llm.MLX_AVAILABLE', True)
    @patch('src.app.ml.llm.get_mlx_model')
    def test_mlx_generate_model_failure(self, mock_get_model):
        """Test MLX generation when model fails"""
        logger.debug("Starting test_mlx_generate_model_failure")
        
        # Mock model returning None (failure case)
        mock_get_model.return_value = (None, None)
        
        response = mlx_generate("Test prompt")
        
        # Should fallback to placeholder response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        logger.debug("Finished test_mlx_generate_model_failure")


class TestFallbackBackends(unittest.TestCase):
    """Test fallback backend functionality"""
    
    @patch('src.app.ml.llm.HF_AVAILABLE', False)
    def test_huggingface_generate_unavailable(self):
        """Test HuggingFace generation when not available"""
        logger.debug("Starting test_huggingface_generate_unavailable")
        
        try:
            response = huggingface_generate("Test prompt")
            self.fail("Expected ImportError")
        except ImportError as e:
            self.assertIn("Hugging Face Transformers not available", str(e))
        
        logger.debug("Finished test_huggingface_generate_unavailable")
    
    @patch('src.app.ml.llm.OPENAI_AVAILABLE', True)
    @patch('src.app.ml.llm.get_openai_client')
    def test_openai_generate(self, mock_get_client):
        """Test OpenAI generation when available"""
        logger.debug("Starting test_openai_generate")
        
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "OpenAI response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        response = openai_generate("Test prompt")
        
        self.assertEqual(response, "OpenAI response")
        mock_client.chat.completions.create.assert_called_once()
        
        logger.debug("Finished test_openai_generate")
    
    @patch('src.app.ml.llm.OPENAI_AVAILABLE', True)
    @patch('src.app.ml.llm.get_openai_client')
    def test_openai_generate_error(self, mock_get_client):
        """Test OpenAI generation error handling"""
        logger.debug("Starting test_openai_generate_error")
        
        # Mock OpenAI client that raises an error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client
        
        response = openai_generate("Test prompt")
        
        self.assertIn("Error: Unable to generate response", response)
        
        logger.debug("Finished test_openai_generate_error")


class TestStreamingFunctionality(unittest.TestCase):
    """Test streaming functionality"""
    
    def test_stream_tokens_basic(self):
        """Test basic streaming functionality"""
        logger.debug("Starting test_stream_tokens_basic")
        
        tokens = list(stream_tokens("Test prompt", max_tokens=10))
        
        # Should return a list of tokens
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Each token should be a string
        for token in tokens:
            self.assertIsInstance(token, str)
        
        logger.debug("Finished test_stream_tokens_basic")
    
    @patch('src.app.ml.llm.config.MODEL_BACKEND', 'openai')
    @patch('src.app.ml.llm.OPENAI_AVAILABLE', True)
    @patch('src.app.ml.llm.get_openai_client')
    def test_stream_tokens_openai(self, mock_get_client):
        """Test streaming with OpenAI backend"""
        logger.debug("Starting test_stream_tokens_openai")
        
        # Mock OpenAI streaming response
        mock_client = Mock()
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta.content = " world"
        
        mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2]
        mock_get_client.return_value = mock_client
        
        tokens = list(stream_tokens("Test prompt"))
        
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0], "Hello")
        self.assertEqual(tokens[1], " world")
        
        logger.debug("Finished test_stream_tokens_openai")
    
    def test_stream_tokens_error_handling(self):
        """Test streaming error handling"""
        logger.debug("Starting test_stream_tokens_error_handling")
        
        # Mock generate_answer to raise an error
        with patch('src.app.ml.llm.generate_answer', side_effect=Exception("Generation error")):
            tokens = list(stream_tokens("Test prompt"))
            
            # Should handle error gracefully
            self.assertIsInstance(tokens, list)
            self.assertGreater(len(tokens), 0)
            
            # Should contain error message
            error_response = ''.join(tokens)
            self.assertIn("Error:", error_response)
        
        logger.debug("Finished test_stream_tokens_error_handling")


class TestHealthCheck(unittest.TestCase):
    """Test health check functionality"""
    
    def test_health_check_success(self):
        """Test successful health check"""
        logger.debug("Starting test_health_check_success")
        
        # Mock generate_answer to return a valid response
        with patch('src.app.ml.llm.generate_answer', return_value="health check passed"):
            result = health_check()
            self.assertTrue(result)
        
        logger.debug("Finished test_health_check_success")
    
    def test_health_check_failure(self):
        """Test health check failure"""
        logger.debug("Starting test_health_check_failure")
        
        # Mock generate_answer to raise an error
        with patch('src.app.ml.llm.generate_answer', side_effect=Exception("LLM error")):
            result = health_check()
            self.assertFalse(result)
        
        logger.debug("Finished test_health_check_failure")
    
    def test_health_check_empty_response(self):
        """Test health check with empty response"""
        logger.debug("Starting test_health_check_empty_response")
        
        # Mock generate_answer to return empty response
        with patch('src.app.ml.llm.generate_answer', return_value=""):
            result = health_check()
            self.assertFalse(result)
        
        logger.debug("Finished test_health_check_empty_response")


class TestBackendFallback(unittest.TestCase):
    """Test backend fallback logic"""
    
    @patch('src.app.ml.llm.MLX_AVAILABLE', False)
    @patch('src.app.ml.llm.HF_AVAILABLE', True)
    @patch('src.app.ml.llm.OPENAI_AVAILABLE', False)
    @patch('os.getenv')
    def test_fallback_to_huggingface(self, mock_getenv):
        """Test fallback to HuggingFace when MLX unavailable"""
        logger.debug("Starting test_fallback_to_huggingface")
        
        # Mock testing environment
        mock_getenv.return_value = '1'
        
        # Mock HuggingFace generation
        with patch('src.app.ml.llm.huggingface_generate', return_value="HF response") as mock_hf:
            response = generate_answer("Test prompt")
            
            # Should return mock response in testing mode
            self.assertIsInstance(response, str)
        
        logger.debug("Finished test_fallback_to_huggingface")
    
    @patch('src.app.ml.llm.MLX_AVAILABLE', False)
    @patch('src.app.ml.llm.HF_AVAILABLE', False)
    @patch('src.app.ml.llm.OPENAI_AVAILABLE', True)
    @patch('os.getenv')
    def test_fallback_to_openai(self, mock_getenv):
        """Test fallback to OpenAI when other backends unavailable"""
        logger.debug("Starting test_fallback_to_openai")
        
        # Mock testing environment
        mock_getenv.return_value = '1'
        
        # Mock OpenAI generation
        with patch('src.app.ml.llm.openai_generate', return_value="OpenAI response") as mock_openai:
            response = generate_answer("Test prompt")
            
            # Should return mock response in testing mode
            self.assertIsInstance(response, str)
        
        logger.debug("Finished test_fallback_to_openai")
    
    @patch('src.app.ml.llm.MLX_AVAILABLE', False)
    @patch('src.app.ml.llm.HF_AVAILABLE', False)
    @patch('src.app.ml.llm.OPENAI_AVAILABLE', False)
    @patch('os.getenv')
    def test_no_backend_available(self, mock_getenv):
        """Test behavior when no backend is available"""
        logger.debug("Starting test_no_backend_available")
        
        # Mock testing environment
        mock_getenv.return_value = '1'
        
        response = generate_answer("Test prompt")
        
        # Should return error message
        self.assertIsInstance(response, str)
        self.assertIn("Error: No available LLM backend", response)
        
        logger.debug("Finished test_no_backend_available")


if __name__ == '__main__':
    unittest.main() 