#!/usr/bin/env python3
"""
Performance tests for LLM module.
Tests load handling, latency, and resource usage.
"""

import os
import sys
import pytest
import unittest
import time
import threading
import queue
import statistics
from pathlib import Path

# Set testing environment variable BEFORE any imports
os.environ['TESTING'] = '1'

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.app.log import logger
from src.app.ml.llm import (
    generate_answer, 
    stream_tokens, 
    health_check
)


class TestLLMLatency(unittest.TestCase):
    """Test LLM latency characteristics"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.test_prompt = "What is machine learning?"
        self.test_kwargs = {
            "max_tokens": 50,
            "temperature": 0.1,
            "top_p": 0.9
        }
    
    def test_single_request_latency(self):
        """Test latency for single requests"""
        logger.info("=== Single Request Latency Test ===")
        
        latencies = []
        
        # Measure latency for multiple single requests
        for i in range(10):
            start_time = time.time()
            response = generate_answer(self.test_prompt, **self.test_kwargs)
            end_time = time.time()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            # Validate response
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            
            logger.debug(f"Request {i+1}: {latency:.3f}s")
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        logger.info(f"Latency Statistics (n={len(latencies)}):")
        logger.info(f"  Average: {avg_latency:.3f}s")
        logger.info(f"  Median: {median_latency:.3f}s")
        logger.info(f"  Min: {min_latency:.3f}s")
        logger.info(f"  Max: {max_latency:.3f}s")
        logger.info(f"  Std Dev: {std_latency:.3f}s")
        
        # Performance assertions (adjust based on expected performance)
        self.assertLess(avg_latency, 1.0, f"Average latency too high: {avg_latency:.3f}s")
        self.assertLess(max_latency, 2.0, f"Max latency too high: {max_latency:.3f}s")
    
    def test_streaming_latency(self):
        """Test latency for streaming responses"""
        logger.info("=== Streaming Latency Test ===")
        
        prompt = "Count from 1 to 10"
        
        # Measure time to first token and total streaming time
        start_time = time.time()
        first_token_time = None
        token_count = 0
        
        for token in stream_tokens(prompt, max_tokens=50):
            if first_token_time is None:
                first_token_time = time.time() - start_time
            token_count += 1
        
        total_time = time.time() - start_time
        
        logger.info(f"Streaming Performance:")
        logger.info(f"  Time to first token: {first_token_time:.3f}s")
        logger.info(f"  Total streaming time: {total_time:.3f}s")
        logger.info(f"  Tokens streamed: {token_count}")
        
        if token_count > 0:
            avg_token_time = total_time / token_count
            logger.info(f"  Average time per token: {avg_token_time:.3f}s")
            
            # Performance assertions
            self.assertLess(first_token_time, 1.0, f"Time to first token too high: {first_token_time:.3f}s")
            self.assertLess(avg_token_time, 0.1, f"Average token time too high: {avg_token_time:.3f}s")
        
        self.assertGreater(token_count, 0, "No tokens were streamed")
    
    def test_health_check_latency(self):
        """Test health check latency"""
        logger.info("=== Health Check Latency Test ===")
        
        latencies = []
        
        # Measure health check latency multiple times
        for i in range(5):
            start_time = time.time()
            is_healthy = health_check()
            end_time = time.time()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            # Validate health check
            self.assertIsInstance(is_healthy, bool)
            
            logger.debug(f"Health check {i+1}: {latency:.3f}s - {is_healthy}")
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        logger.info(f"Health Check Latency:")
        logger.info(f"  Average: {avg_latency:.3f}s")
        logger.info(f"  Max: {max_latency:.3f}s")
        
        # Health checks should be fast
        self.assertLess(avg_latency, 0.5, f"Health check latency too high: {avg_latency:.3f}s")
        self.assertLess(max_latency, 1.0, f"Max health check latency too high: {max_latency:.3f}s")


class TestLLMLoad(unittest.TestCase):
    """Test LLM load handling capabilities"""
    
    def test_concurrent_load(self):
        """Test concurrent request load handling"""
        logger.info("=== Concurrent Load Test ===")
        
        num_threads = 10
        requests_per_thread = 5
        total_requests = num_threads * requests_per_thread
        
        results = queue.Queue()
        
        def worker_thread(thread_id):
            """Worker thread function"""
            thread_results = []
            
            for request_id in range(requests_per_thread):
                try:
                    start_time = time.time()
                    response = generate_answer(
                        f"Test request {thread_id}-{request_id}", 
                        max_tokens=30
                    )
                    end_time = time.time()
                    
                    thread_results.append({
                        'thread_id': thread_id,
                        'request_id': request_id,
                        'latency': end_time - start_time,
                        'response_length': len(response),
                        'success': True,
                        'error': None
                    })
                    
                except Exception as e:
                    thread_results.append({
                        'thread_id': thread_id,
                        'request_id': request_id,
                        'latency': None,
                        'response_length': 0,
                        'success': False,
                        'error': str(e)
                    })
            
            results.put(thread_results)
        
        # Start load test
        start_time = time.time()
        threads = []
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        all_results = []
        while not results.empty():
            thread_results = results.get()
            all_results.extend(thread_results)
        
        # Analyze results
        successful_requests = [r for r in all_results if r['success']]
        failed_requests = [r for r in all_results if not r['success']]
        
        success_rate = len(successful_requests) / len(all_results) * 100
        
        logger.info(f"Load Test Results:")
        logger.info(f"  Total requests: {len(all_results)}")
        logger.info(f"  Successful: {len(successful_requests)}")
        logger.info(f"  Failed: {len(failed_requests)}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info(f"  Total time: {total_time:.3f}s")
        logger.info(f"  Requests per second: {len(all_results) / total_time:.2f}")
        
        if successful_requests:
            latencies = [r['latency'] for r in successful_requests]
            avg_latency = statistics.mean(latencies)
            max_latency = max(latencies)
            
            logger.info(f"  Average latency: {avg_latency:.3f}s")
            logger.info(f"  Max latency: {max_latency:.3f}s")
            
            # Performance assertions
            self.assertGreaterEqual(success_rate, 95.0, f"Success rate too low: {success_rate:.1f}%")
            self.assertLess(avg_latency, 2.0, f"Average latency under load too high: {avg_latency:.3f}s")
        
        # Log any failures
        if failed_requests:
            logger.warning("Failed requests:")
            for req in failed_requests[:5]:  # Log first 5 failures
                logger.warning(f"  Thread {req['thread_id']}, Request {req['request_id']}: {req['error']}")
    
    def test_sustained_load(self):
        """Test sustained load over time"""
        logger.info("=== Sustained Load Test ===")
        
        duration_seconds = 30  # Run for 30 seconds
        request_interval = 0.5  # Request every 0.5 seconds
        
        results = []
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration_seconds:
            request_start = time.time()
            
            try:
                response = generate_answer(
                    f"Sustained load request {request_count}", 
                    max_tokens=30
                )
                request_end = time.time()
                
                results.append({
                    'request_id': request_count,
                    'timestamp': request_start,
                    'latency': request_end - request_start,
                    'response_length': len(response),
                    'success': True,
                    'error': None
                })
                
            except Exception as e:
                request_end = time.time()
                results.append({
                    'request_id': request_count,
                    'timestamp': request_start,
                    'latency': request_end - request_start,
                    'response_length': 0,
                    'success': False,
                    'error': str(e)
                })
            
            request_count += 1
            
            # Wait for next request interval
            elapsed = time.time() - request_start
            if elapsed < request_interval:
                time.sleep(request_interval - elapsed)
        
        total_time = time.time() - start_time
        
        # Analyze sustained load results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        success_rate = len(successful_requests) / len(results) * 100 if results else 0
        
        logger.info(f"Sustained Load Results:")
        logger.info(f"  Duration: {total_time:.1f}s")
        logger.info(f"  Total requests: {len(results)}")
        logger.info(f"  Successful: {len(successful_requests)}")
        logger.info(f"  Failed: {len(failed_requests)}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info(f"  Average RPS: {len(results) / total_time:.2f}")
        
        if successful_requests:
            latencies = [r['latency'] for r in successful_requests]
            avg_latency = statistics.mean(latencies)
            
            # Check for latency degradation over time
            first_half = latencies[:len(latencies)//2]
            second_half = latencies[len(latencies)//2:]
            
            if first_half and second_half:
                first_half_avg = statistics.mean(first_half)
                second_half_avg = statistics.mean(second_half)
                latency_increase = (second_half_avg - first_half_avg) / first_half_avg * 100
                
                logger.info(f"  First half avg latency: {first_half_avg:.3f}s")
                logger.info(f"  Second half avg latency: {second_half_avg:.3f}s")
                logger.info(f"  Latency increase: {latency_increase:.1f}%")
                
                # Performance assertions
                self.assertLess(latency_increase, 50.0, f"Latency degradation too high: {latency_increase:.1f}%")
        
        # Overall performance assertions
        self.assertGreaterEqual(success_rate, 90.0, f"Sustained load success rate too low: {success_rate:.1f}%")


class TestLLMResourceUsage(unittest.TestCase):
    """Test LLM resource usage characteristics"""
    
    def test_memory_usage(self):
        """Test memory usage during LLM operations"""
        logger.info("=== Memory Usage Test ===")
        
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil not available for memory testing")
        
        process = psutil.Process()
        
        # Get baseline memory usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Baseline memory: {baseline_memory:.2f} MB")
        
        # Perform multiple LLM operations
        for i in range(20):
            response = generate_answer(f"Memory test request {i}", max_tokens=50)
            self.assertIsInstance(response, str)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        logger.info(f"Final memory: {final_memory:.2f} MB")
        logger.info(f"Memory increase: {memory_increase:.2f} MB")
        
        # Memory increase should be reasonable
        self.assertLess(memory_increase, 100, f"Memory increase too high: {memory_increase:.2f} MB")
    
    def test_cpu_usage_pattern(self):
        """Test CPU usage patterns during LLM operations"""
        logger.info("=== CPU Usage Pattern Test ===")
        
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil not available for CPU testing")
        
        # Monitor CPU usage during operations
        cpu_samples = []
        
        def monitor_cpu():
            for _ in range(10):  # Sample for 10 seconds
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_samples.append(cpu_percent)
        
        # Start CPU monitoring in background
        import threading
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Perform LLM operations
        for i in range(5):
            response = generate_answer(f"CPU test request {i}", max_tokens=50)
            self.assertIsInstance(response, str)
            time.sleep(1)
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        if cpu_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)
            
            logger.info(f"CPU Usage During LLM Operations:")
            logger.info(f"  Average: {avg_cpu:.1f}%")
            logger.info(f"  Max: {max_cpu:.1f}%")
            logger.info(f"  Samples: {cpu_samples}")
            
            # CPU usage should be reasonable (not pegging the CPU)
            self.assertLess(avg_cpu, 80.0, f"Average CPU usage too high: {avg_cpu:.1f}%")


if __name__ == '__main__':
    unittest.main() 