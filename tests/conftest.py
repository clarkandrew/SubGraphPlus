#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for all tests.
"""

import os
import sys
import json
import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import shutil

# Disable model loading during tests to prevent segfaults on Apple Silicon
os.environ["SUBGRAPHRAG_DISABLE_MODEL_LOADING"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Force HuggingFace to use offline mode - don't check for updates
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Add parent directory to path so tests can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from src.app.log import logger


@pytest.fixture(scope="session")
def test_fixtures_dir():
    """Get the test fixtures directory path"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_prompts(test_fixtures_dir):
    """Load sample prompts from fixtures"""
    with open(test_fixtures_dir / "sample_prompts.json", "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def expected_responses(test_fixtures_dir):
    """Load expected responses from fixtures"""
    with open(test_fixtures_dir / "expected_responses.json", "r") as f:
        return json.load(f)


@pytest.fixture
def testing_environment():
    """Ensure testing environment is set up correctly"""
    original_testing = os.environ.get('TESTING', '')
    os.environ['TESTING'] = '1'
    yield
    os.environ['TESTING'] = original_testing


@pytest.fixture
def non_testing_environment():
    """Set up non-testing environment for e2e tests"""
    original_testing = os.environ.get('TESTING', '')
    if 'TESTING' in os.environ:
        del os.environ['TESTING']
    yield
    os.environ['TESTING'] = original_testing


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing"""
    return {
        'query_embedding': np.random.random(1024).astype(np.float32),
        'triple_embedding': np.random.random(1024).astype(np.float32),
        'dde_features': [1.0, 0.5, 0.0, 1.0, 0.8]
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing"""
    return "Mock LLM response for testing"


@pytest.fixture
def mock_mlx_model():
    """Mock MLX model for testing"""
    mock_model = Mock()
    mock_tokenizer = Mock()
    
    # Mock model output
    mock_output = Mock()
    mock_output.item.return_value = 0.75
    mock_model.return_value = mock_output
    
    return mock_model, mock_tokenizer


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "OpenAI mock response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        # Write some dummy data
        f.write(b"dummy model data")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def performance_test_config():
    """Configuration for performance tests"""
    return {
        'max_response_time': 30.0,
        'max_memory_increase_mb': 100,
        'min_success_rate': 95.0,
        'concurrent_requests': 10,
        'load_test_duration': 30,
        'request_interval': 0.5
    }


@pytest.fixture
def llm_test_parameters():
    """Standard LLM test parameters"""
    return {
        'max_tokens': 50,
        'temperature': 0.1,
        'top_p': 0.9
    }


@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests"""
    logger.info("Starting test execution")
    yield
    logger.info("Test execution completed")


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    mock_config = Mock()
    mock_config.MODEL_BACKEND = "mlx"
    mock_config.MLP_MODEL_PATH = "models/mlp/mlp.pth"
    
    # Mock model configurations
    mock_config.get_model_config.side_effect = lambda backend: {
        "mlx": {
            "model": "mlx-community/Qwen3-14B-8bit",
            "max_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9
        },
        "openai": {
            "model": "gpt-3.5-turbo",
            "max_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9
        },
        "huggingface": {
            "model": "mlx-community/Qwen3-14B-8bit",
            "max_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9
        }
    }.get(backend, {})
    
    return mock_config


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_mlx: mark test as requiring MLX backend"
    )
    config.addinivalue_line(
        "markers", "requires_openai: mark test as requiring OpenAI API"
    )
    config.addinivalue_line(
        "markers", "requires_hf: mark test as requiring HuggingFace"
    )


# Skip conditions for different backends
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip markers"""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)


# Test data generators
@pytest.fixture
def generate_test_prompts():
    """Generate test prompts of various lengths"""
    def _generate(count=5, min_length=10, max_length=100):
        prompts = []
        for i in range(count):
            length = np.random.randint(min_length, max_length)
            prompt = f"Test prompt {i}: " + "word " * (length // 5)
            prompts.append(prompt.strip())
        return prompts
    return _generate


@pytest.fixture
def generate_batch_data():
    """Generate batch test data"""
    def _generate(batch_size=10):
        return {
            'query_embeddings': [np.random.random(1024).astype(np.float32) for _ in range(batch_size)],
            'triple_embeddings': [np.random.random(1024).astype(np.float32) for _ in range(batch_size)],
            'dde_features': [[1.0, 0.5, 0.0, 1.0, 0.8] for _ in range(batch_size)],
            'prompts': [f"Batch test prompt {i}" for i in range(batch_size)]
        }
    return _generate


# Cleanup fixtures
@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """Clean up test artifacts after test session"""
    yield
    
    # Clean up any temporary files or directories created during testing
    test_dir = Path(__file__).parent
    
    # Remove any .pyc files
    for pyc_file in test_dir.rglob("*.pyc"):
        try:
            pyc_file.unlink()
        except FileNotFoundError:
            pass
    
    # Remove __pycache__ directories
    for pycache_dir in test_dir.rglob("__pycache__"):
        try:
            pycache_dir.rmdir()
        except (FileNotFoundError, OSError):
            pass
    
    logger.info("Test cleanup completed")


# Error handling fixtures
@pytest.fixture
def capture_exceptions():
    """Capture and log exceptions during testing"""
    exceptions = []
    
    def _capture(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exceptions.append(e)
            logger.error(f"Captured exception: {e}")
            raise
    
    yield _capture, exceptions


# Performance monitoring fixtures
@pytest.fixture
def performance_monitor():
    """Monitor performance during tests"""
    import time
    import psutil
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    duration = end_time - start_time
    memory_increase = end_memory - start_memory
    
    logger.info(f"Test performance: {duration:.3f}s, memory: +{memory_increase:.2f}MB")


# Mock backend availability
@pytest.fixture
def mock_backend_availability():
    """Mock backend availability for testing"""
    def _mock(mlx=False, hf=False, openai=False):
        with patch('src.app.ml.llm.MLX_AVAILABLE', mlx), \
             patch('src.app.ml.llm.HF_AVAILABLE', hf), \
             patch('src.app.ml.llm.OPENAI_AVAILABLE', openai):
            yield
    return _mock