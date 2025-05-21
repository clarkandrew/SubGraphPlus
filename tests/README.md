# SubgraphRAG+ Test Suite

## Overview
This directory contains the test suite for SubgraphRAG+, including unit tests, integration tests, smoke tests, and adversarial tests. The tests are designed to verify the correctness and robustness of the system.

## Test Categories

### Unit Tests
- **test_utils.py**: Tests for utility functions including entity linking, DDE feature extraction, etc.
- **test_retriever.py**: Tests for the hybrid retrieval system, including graph traversal and vector search.

### Integration Tests
- **test_api.py**: Tests for API endpoints, request handling, and response formatting.

### Smoke Tests
- **test_smoke.py**: Basic functionality tests and edge case handling to ensure the system doesn't crash under unexpected conditions.

### Adversarial Tests
- **test_adversarial.py**: Tests designed to stress the system with malicious inputs, injection attempts, and other adversarial scenarios.

## Running Tests

### Using the Test Script
```bash
# Run all tests
./run_tests.sh

# Run specific test categories
./run_tests.sh -t unit
./run_tests.sh -t smoke
./run_tests.sh -t adversarial

# Generate coverage report
./run_tests.sh -c

# Run tests with verbose output
./run_tests.sh -v
```

### Using pytest Directly
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py

# Run specific test
pytest tests/test_api.py::test_health_check

# Generate coverage report
pytest --cov=app
```

## Test Fixtures and Mocks

The tests use a variety of fixtures and mocks to simulate different components of the system:

- **mock_auth_header**: Simulates authenticated requests
- **mock_neo4j**: Mocks Neo4j database connections and queries
- **mock_sqlite**: Mocks SQLite database operations
- **mock_faiss_index**: Mocks FAISS vector index operations
- **mock_llm**: Mocks language model responses

These fixtures are defined in `conftest.py` and are automatically available to all test functions.

## Error Cases and Edge Conditions Tested

- Authentication failures
- Malformed input handling
- Empty/null inputs
- Very large inputs
- Malicious injection attempts (SQL, XSS, etc.)
- Unicode and special character handling
- Timeouts and network failures
- Resource limitations
- Concurrent request handling

## Adding New Tests

When adding new tests:

1. Choose the appropriate test file based on the component being tested
2. Use existing fixtures where possible
3. Follow the naming convention: `test_<functionality>_<scenario>`
4. Document any new fixtures in `conftest.py`
5. Ensure tests are deterministic (no random behavior)
6. Add appropriate assertions for both positive and negative cases

## Performance Testing

For performance benchmarking, use the evaluation framework in the `evaluation` directory instead of the test suite.