#!/usr/bin/env python3
"""
Integration tests for LLM module.
Tests component interactions and real functionality with controlled environments.
"""

import os
import sys
import pytest
import unittest
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Set testing environment variable BEFORE any imports
os.environ['TESTING'] = '1'

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.app.log import logger
from src.app.ml.llm import (
    generate_answer, 
    stream_tokens, 
    health_check,
    MLX_AVAILABLE,
    HF_AVAILABLE,
    OPENAI_AVAILABLE
)
from src.app.config import config


class TestLLMIntegration(unittest.TestCase):
    """Test LLM integration with other system components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_prompts = [
            "What is the capital of France?",
            "Explain what machine learning is in one sentence.",
            "List three benefits of renewable energy.",
            "What is 2 + 2?",
            "Define artificial intelligence."
        ]
        
        self.test_kwargs = {
            "max_tokens": 100,
            "temperature": 0.1,
            "top_p": 0.9
        }
    
    def test_end_to_end_generation(self):
        """Test end-to-end LLM generation workflow"""
        logger.debug("Starting test_end_to_end_generation")
        
        for prompt in self.test_prompts[:2]:  # Test first 2 prompts
            with self.subTest(prompt=prompt):
                # Test the complete generation workflow
                response = generate_answer(prompt, **self.test_kwargs)
                
                # Validate response
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 0)
                
                # In testing mode, should return mock response
                self.assertIn("Mock LLM response", response)
        
        logger.debug("Finished test_end_to_end_generation")
    
    def test_parameter_validation(self):
        """Test parameter validation across the system"""
        logger.debug("Starting test_parameter_validation")
        
        prompt = "Test prompt"
        
        # Test various parameter combinations
        parameter_sets = [
            {"max_tokens": 50, "temperature": 0.0, "top_p": 1.0},
            {"max_tokens": 100, "temperature": 0.5, "top_p": 0.9},
            {"max_tokens": 200, "temperature": 1.0, "top_p": 0.8},
        ]
        
        for params in parameter_sets:
            with self.subTest(params=params):
                response = generate_answer(prompt, **params)
                
                # Should handle all parameter combinations
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 0)
        
        logger.debug("Finished test_parameter_validation")
    
    def test_concurrent_generation(self):
        """Test concurrent LLM generation requests"""
        logger.debug("Starting test_concurrent_generation")
        
        import threading
        import queue
        
        results = queue.Queue()
        
        def generate_response(prompt, index):
            try:
                response = generate_answer(f"{prompt} (Request {index})", max_tokens=50)
                results.put((index, response, None))
            except Exception as e:
                results.put((index, None, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=generate_response, 
                args=("Test concurrent prompt", i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        responses = []
        errors = []
        while not results.empty():
            index, response, error = results.get()
            if error:
                errors.append((index, error))
            else:
                responses.append((index, response))
        
        # Validate results
        self.assertEqual(len(responses), 5, f"Expected 5 responses, got {len(responses)}")
        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")
        
        for index, response in responses:
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        
        logger.debug("Finished test_concurrent_generation")
    
    def test_streaming_integration(self):
        """Test streaming integration with the system"""
        logger.debug("Starting test_streaming_integration")
        
        prompt = "Count from 1 to 5"
        
        # Test streaming functionality
        tokens = list(stream_tokens(prompt, max_tokens=50))
        
        # Validate streaming
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Each token should be a string
        for token in tokens:
            self.assertIsInstance(token, str)
        
        # Reconstruct full response
        full_response = ''.join(tokens)
        self.assertIsInstance(full_response, str)
        self.assertGreater(len(full_response), 0)
        
        logger.debug("Finished test_streaming_integration")
    
    def test_health_check_integration(self):
        """Test health check integration with system monitoring"""
        logger.debug("Starting test_health_check_integration")
        
        # Test health check
        is_healthy = health_check()
        
        # Should return a boolean
        self.assertIsInstance(is_healthy, bool)
        
        # In testing mode with mocked responses, should be healthy
        self.assertTrue(is_healthy)
        
        logger.debug("Finished test_health_check_integration")
    
    def test_backend_fallback_integration(self):
        """Test backend fallback integration"""
        logger.debug("Starting test_backend_fallback_integration")
        
        prompt = "Test backend fallback"
        
        # Test with different backend configurations
        original_backend = config.MODEL_BACKEND
        
        try:
            # Test MLX backend (default)
            response_mlx = generate_answer(prompt, max_tokens=50)
            self.assertIsInstance(response_mlx, str)
            self.assertGreater(len(response_mlx), 0)
            
            # In testing mode, all backends return mock responses
            self.assertIn("Mock LLM response", response_mlx)
            
        finally:
            # Restore original backend
            config.MODEL_BACKEND = original_backend
        
        logger.debug("Finished test_backend_fallback_integration")
    
    def test_error_handling_integration(self):
        """Test error handling integration across components"""
        logger.debug("Starting test_error_handling_integration")
        
        # Test with various error conditions
        test_cases = [
            ("", "empty prompt"),
            ("x" * 10000, "very long prompt"),
            ("Test prompt with special chars: !@#$%^&*()", "special characters"),
        ]
        
        for prompt, description in test_cases:
            with self.subTest(description=description):
                try:
                    response = generate_answer(prompt, max_tokens=50)
                    
                    # Should handle gracefully
                    self.assertIsInstance(response, str)
                    # Empty response is acceptable for edge cases
                    
                except Exception as e:
                    # If an exception occurs, it should be a known type
                    self.assertIsInstance(e, (ValueError, TypeError, RuntimeError))
        
        logger.debug("Finished test_error_handling_integration")


class TestLLMConfigurationIntegration(unittest.TestCase):
    """Test LLM configuration integration with the system"""
    
    def test_config_loading_integration(self):
        """Test configuration loading integration"""
        logger.debug("Starting test_config_loading_integration")
        
        # Test that configuration is properly loaded
        self.assertEqual(config.MODEL_BACKEND, "mlx")
        
        # Test model configurations
        mlx_config = config.get_model_config("mlx")
        self.assertIsInstance(mlx_config, dict)
        self.assertIn("model", mlx_config)
        self.assertIn("max_tokens", mlx_config)
        self.assertIn("temperature", mlx_config)
        
        openai_config = config.get_model_config("openai")
        self.assertIsInstance(openai_config, dict)
        self.assertIn("model", openai_config)
        
        hf_config = config.get_model_config("huggingface")
        self.assertIsInstance(hf_config, dict)
        self.assertIn("model", hf_config)
        
        logger.debug("Finished test_config_loading_integration")
    
    def test_backend_availability_integration(self):
        """Test backend availability integration"""
        logger.debug("Starting test_backend_availability_integration")
        
        # Test backend availability flags
        self.assertIsInstance(MLX_AVAILABLE, bool)
        self.assertIsInstance(HF_AVAILABLE, bool)
        self.assertIsInstance(OPENAI_AVAILABLE, bool)
        
        # In testing mode, MLX and HF should be False
        self.assertFalse(MLX_AVAILABLE)
        self.assertFalse(HF_AVAILABLE)
        
        logger.debug("Finished test_backend_availability_integration")


class TestLLMPerformanceIntegration(unittest.TestCase):
    """Test LLM performance integration"""
    
    def test_response_time_integration(self):
        """Test response time integration with system performance"""
        logger.debug("Starting test_response_time_integration")
        
        prompt = "What is artificial intelligence?"
        
        # Measure response time
        start_time = time.time()
        response = generate_answer(prompt, max_tokens=100)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Validate response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Response time should be reasonable (less than 5 seconds in testing mode)
        self.assertLess(response_time, 5.0, f"Response time too slow: {response_time:.2f}s")
        
        logger.debug(f"Response time: {response_time:.3f}s")
        logger.debug("Finished test_response_time_integration")
    
    def test_memory_usage_integration(self):
        """Test memory usage integration"""
        logger.debug("Starting test_memory_usage_integration")
        
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate multiple responses
        for i in range(10):
            response = generate_answer(f"Test prompt {i}", max_tokens=50)
            self.assertIsInstance(response, str)
        
        # Force garbage collection
        gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for testing)
        self.assertLess(memory_increase, 100, f"Memory increase too high: {memory_increase:.2f}MB")
        
        logger.debug(f"Memory usage: {initial_memory:.2f}MB -> {final_memory:.2f}MB (+{memory_increase:.2f}MB)")
        logger.debug("Finished test_memory_usage_integration")


if __name__ == '__main__':
    unittest.main() 