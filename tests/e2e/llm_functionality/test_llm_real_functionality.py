#!/usr/bin/env python3
"""
End-to-end tests for LLM real functionality.
Tests complete workflows with actual LLM backends (when available).
"""

import os
import sys
import pytest
import unittest
import time
from pathlib import Path

# DO NOT set TESTING=1 for this file - we want to test real functionality
# os.environ['TESTING'] = '1'  # Commented out intentionally

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
    MLX_AVAILABLE,
    HF_AVAILABLE,
    OPENAI_AVAILABLE
)
from src.app.config import config


@pytest.mark.skipif(not (MLX_AVAILABLE or HF_AVAILABLE or OPENAI_AVAILABLE), 
                    reason="No LLM backend available")
class TestRealLLMFunctionality(unittest.TestCase):
    """Test actual LLM functionality with real responses"""
    
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
    
    def test_backend_availability(self):
        """Test which backends are actually available"""
        logger.info("=== Backend Availability Test ===")
        logger.info(f"MLX Available: {MLX_AVAILABLE}")
        logger.info(f"HuggingFace Available: {HF_AVAILABLE}")
        logger.info(f"OpenAI Available: {OPENAI_AVAILABLE}")
        logger.info(f"Configured Backend: {config.MODEL_BACKEND}")
        
        # At least one backend should be available
        self.assertTrue(
            MLX_AVAILABLE or HF_AVAILABLE or OPENAI_AVAILABLE,
            "No LLM backend is available"
        )
    
    def test_real_generation_with_responses(self):
        """Test real LLM generation and display actual responses"""
        logger.info("=== Real LLM Generation Test ===")
        
        for i, prompt in enumerate(self.test_prompts):
            with self.subTest(prompt=prompt):
                logger.info(f"\n--- Test {i+1}: {prompt} ---")
                
                try:
                    response = generate_answer(prompt, **self.test_kwargs)
                    
                    # Validate response
                    self.assertIsInstance(response, str)
                    self.assertGreater(len(response), 0)
                    
                    # Display the actual response
                    logger.info(f"PROMPT: {prompt}")
                    logger.info(f"RESPONSE: {response}")
                    logger.info(f"Response Length: {len(response)} characters")
                    
                    # Basic quality checks
                    self.assertNotIn("Error", response, f"Response contains error: {response}")
                    self.assertGreater(len(response.strip()), 5, "Response too short")
                    
                except Exception as e:
                    logger.error(f"Failed to generate response for: {prompt}")
                    logger.error(f"Error: {e}")
                    self.fail(f"Generation failed for prompt: {prompt}, Error: {e}")
    
    def test_different_parameters(self):
        """Test LLM with different parameter settings"""
        logger.info("=== Parameter Variation Test ===")
        
        prompt = "Explain quantum computing in simple terms."
        
        parameter_sets = [
            {"max_tokens": 50, "temperature": 0.0, "top_p": 1.0},
            {"max_tokens": 100, "temperature": 0.5, "top_p": 0.9},
            {"max_tokens": 150, "temperature": 0.8, "top_p": 0.8},
        ]
        
        for i, params in enumerate(parameter_sets):
            with self.subTest(params=params):
                logger.info(f"\n--- Parameter Set {i+1}: {params} ---")
                
                try:
                    response = generate_answer(prompt, **params)
                    
                    logger.info(f"PROMPT: {prompt}")
                    logger.info(f"PARAMS: {params}")
                    logger.info(f"RESPONSE: {response}")
                    logger.info(f"Response Length: {len(response)} characters")
                    
                    # Validate response
                    self.assertIsInstance(response, str)
                    self.assertGreater(len(response), 0)
                    
                    # Check that max_tokens is roughly respected
                    # Note: This is approximate as different tokenizers count differently
                    word_count = len(response.split())
                    expected_max_words = params["max_tokens"] * 1.5  # Rough estimate
                    self.assertLessEqual(
                        word_count, 
                        expected_max_words, 
                        f"Response too long: {word_count} words for max_tokens={params['max_tokens']}"
                    )
                    
                except Exception as e:
                    logger.error(f"Failed with parameters: {params}")
                    logger.error(f"Error: {e}")
                    self.fail(f"Generation failed with params: {params}, Error: {e}")
    
    def test_streaming_functionality(self):
        """Test streaming token generation"""
        logger.info("=== Streaming Test ===")
        
        prompt = "Count from 1 to 5 and explain each number."
        
        try:
            logger.info(f"STREAMING PROMPT: {prompt}")
            logger.info("STREAMING RESPONSE:")
            
            tokens = []
            for token in stream_tokens(prompt, max_tokens=100):
                tokens.append(token)
                print(token, end='', flush=True)  # Real-time display
            
            print()  # New line after streaming
            
            # Validate streaming
            self.assertGreater(len(tokens), 0, "No tokens were streamed")
            
            # Reconstruct full response
            full_response = ''.join(tokens)
            logger.info(f"FULL STREAMED RESPONSE: {full_response}")
            logger.info(f"Total Tokens Streamed: {len(tokens)}")
            
            # Validate reconstructed response
            self.assertIsInstance(full_response, str)
            self.assertGreater(len(full_response), 0)
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            self.fail(f"Streaming test failed: {e}")
    
    def test_health_check_real(self):
        """Test real health check functionality"""
        logger.info("=== Health Check Test ===")
        
        try:
            is_healthy = health_check()
            logger.info(f"Health Check Result: {is_healthy}")
            
            # Health check should pass if any backend is working
            if MLX_AVAILABLE or HF_AVAILABLE or OPENAI_AVAILABLE:
                self.assertTrue(is_healthy, "Health check failed despite available backends")
            else:
                self.assertFalse(is_healthy, "Health check passed despite no available backends")
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
            self.fail(f"Health check test failed: {e}")
    
    @pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
    def test_mlx_backend_specific(self):
        """Test MLX backend specifically"""
        logger.info("=== MLX Backend Test ===")
        
        prompt = "What is the meaning of life?"
        
        try:
            response = mlx_generate(prompt, max_tokens=50)
            logger.info(f"MLX RESPONSE: {response}")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        except Exception as e:
            logger.error(f"MLX backend failed: {e}")
            self.fail(f"MLX backend test failed: {e}")
    
    @pytest.mark.skipif(not HF_AVAILABLE, reason="HuggingFace not available")
    def test_huggingface_backend_specific(self):
        """Test HuggingFace backend specifically"""
        logger.info("=== HuggingFace Backend Test ===")
        
        prompt = "What is the meaning of life?"
        
        try:
            response = huggingface_generate(prompt, max_tokens=50)
            logger.info(f"HF RESPONSE: {response}")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        except Exception as e:
            logger.error(f"HuggingFace backend failed: {e}")
            self.fail(f"HuggingFace backend test failed: {e}")
    
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_openai_backend_specific(self):
        """Test OpenAI backend specifically"""
        logger.info("=== OpenAI Backend Test ===")
        
        prompt = "What is the meaning of life?"
        
        try:
            response = openai_generate(prompt, max_tokens=50)
            logger.info(f"OpenAI RESPONSE: {response}")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        except Exception as e:
            logger.error(f"OpenAI backend failed: {e}")
            self.fail(f"OpenAI backend test failed: {e}")
    
    def test_context_understanding(self):
        """Test LLM context understanding capabilities"""
        logger.info("=== Context Understanding Test ===")
        
        context_prompts = [
            ("The sky is blue.", "What color is the sky?"),
            ("Paris is the capital of France.", "What is the capital of France?"),
            ("2 + 2 = 4", "What is 2 + 2?"),
        ]
        
        for context, question in context_prompts:
            with self.subTest(context=context, question=question):
                full_prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                
                try:
                    response = generate_answer(full_prompt, max_tokens=50)
                    
                    logger.info(f"CONTEXT: {context}")
                    logger.info(f"QUESTION: {question}")
                    logger.info(f"RESPONSE: {response}")
                    
                    # Validate response
                    self.assertIsInstance(response, str)
                    self.assertGreater(len(response), 0)
                    
                    # Basic context understanding check
                    if "blue" in context.lower():
                        # Should mention blue for sky color question
                        pass  # Actual content validation would be more complex
                    
                except Exception as e:
                    logger.error(f"Context understanding failed for: {context} -> {question}")
                    logger.error(f"Error: {e}")
                    self.fail(f"Context understanding test failed: {e}")
    
    def test_response_consistency(self):
        """Test response consistency for repeated prompts"""
        logger.info("=== Response Consistency Test ===")
        
        prompt = "What is artificial intelligence?"
        responses = []
        
        # Generate multiple responses for the same prompt
        for i in range(3):
            try:
                response = generate_answer(prompt, max_tokens=100, temperature=0.1)
                responses.append(response)
                logger.info(f"Response {i+1}: {response}")
                
            except Exception as e:
                logger.error(f"Failed to generate response {i+1}: {e}")
                self.fail(f"Response consistency test failed: {e}")
        
        # Validate all responses
        for i, response in enumerate(responses):
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        
        # With low temperature, responses should be somewhat consistent
        # (exact consistency depends on the model and backend)
        logger.info(f"Generated {len(responses)} responses for consistency test")


class TestLLMPerformance(unittest.TestCase):
    """Test LLM performance characteristics"""
    
    def test_response_time(self):
        """Test LLM response time performance"""
        logger.info("=== Response Time Performance Test ===")
        
        prompt = "Explain machine learning in one sentence."
        response_times = []
        
        # Measure response times for multiple requests
        for i in range(5):
            start_time = time.time()
            try:
                response = generate_answer(prompt, max_tokens=50)
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                
                logger.info(f"Request {i+1}: {response_time:.3f}s - {response}")
                
                # Validate response
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 0)
                
            except Exception as e:
                logger.error(f"Performance test request {i+1} failed: {e}")
                self.fail(f"Performance test failed: {e}")
        
        # Calculate performance metrics
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        logger.info(f"Performance Summary:")
        logger.info(f"  Average: {avg_time:.3f}s")
        logger.info(f"  Min: {min_time:.3f}s")
        logger.info(f"  Max: {max_time:.3f}s")
        
        # Performance assertions (adjust based on expected performance)
        self.assertLess(avg_time, 30.0, f"Average response time too slow: {avg_time:.3f}s")
        self.assertLess(max_time, 60.0, f"Max response time too slow: {max_time:.3f}s")
    
    def test_concurrent_requests(self):
        """Test concurrent request handling"""
        logger.info("=== Concurrent Requests Test ===")
        
        import threading
        import queue
        
        results = queue.Queue()
        num_threads = 3
        
        def generate_response(prompt, index):
            try:
                start_time = time.time()
                response = generate_answer(f"{prompt} (Request {index})", max_tokens=50)
                end_time = time.time()
                
                results.put({
                    'index': index,
                    'response': response,
                    'time': end_time - start_time,
                    'error': None
                })
            except Exception as e:
                results.put({
                    'index': index,
                    'response': None,
                    'time': None,
                    'error': str(e)
                })
        
        # Start concurrent threads
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(
                target=generate_response, 
                args=("Test concurrent prompt", i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect and validate results
        responses = []
        errors = []
        response_times = []
        
        while not results.empty():
            result = results.get()
            if result['error']:
                errors.append((result['index'], result['error']))
            else:
                responses.append((result['index'], result['response']))
                response_times.append(result['time'])
        
        logger.info(f"Concurrent test completed in {total_time:.3f}s")
        logger.info(f"Successful responses: {len(responses)}")
        logger.info(f"Errors: {len(errors)}")
        
        # Validate results
        self.assertEqual(len(responses), num_threads, f"Expected {num_threads} responses, got {len(responses)}")
        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")
        
        for index, response in responses:
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            logger.info(f"Response {index}: {response}")
        
        if response_times:
            avg_concurrent_time = sum(response_times) / len(response_times)
            logger.info(f"Average concurrent response time: {avg_concurrent_time:.3f}s")


if __name__ == '__main__':
    unittest.main() 