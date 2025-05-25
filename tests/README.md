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

# SubGraphPlus Tests

This directory contains comprehensive tests for the SubGraphPlus MLP model and related functionality.

## Test Structure

### MLP Model Tests (`test_mlp_model.py`)

The test suite is organized into three main test classes:

#### 1. `TestMLPModel` - Core MLP Functionality
- **Model Creation & Forward Pass**: Tests basic MLP model instantiation and forward propagation
- **Model Loading**: Tests loading pretrained models from disk with proper error handling
- **Input Dimension Calculation**: Validates correct calculation of input dimensions based on embeddings and DDE features
- **Scoring Functions**: Tests both MLP-based scoring and heuristic fallback scoring
- **Input Validation**: Ensures graceful handling of invalid inputs with fallback to heuristic scoring
- **DDE Feature Extraction**: Validates the structure and content of DDE (Dynamic Data Exchange) features

#### 2. `TestMLPIntegration` - Integration Tests
- **Retrieval Pipeline Integration**: Tests MLP integration within the full retrieval pipeline
- **Batch Scoring**: Validates scoring multiple graphs/triples in batch
- **Model Persistence**: Tests saving and loading model state dictionaries

#### 3. `TestMLPEdgeCases` - Edge Cases & Error Handling
- **Empty Features**: Handles empty DDE feature sets gracefully
- **NaN/Inf Values**: Robust handling of invalid numerical values in embeddings
- **Large Feature Vectors**: Performance with very large DDE feature vectors
- **Device Mismatch**: Handling CPU/GPU device mismatches

## Running the Tests

### Prerequisites

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Ensure all dependencies are installed (pytest should be available in the venv)

### Running All MLP Tests

```bash
# Run all MLP model tests
python -m pytest tests/test_mlp_model.py -v

# Run with coverage
python -m pytest tests/test_mlp_model.py --cov=src/app/retriever --cov=src/app/utils -v
```

### Running Specific Test Classes

```bash
# Run only core MLP functionality tests
python -m pytest tests/test_mlp_model.py::TestMLPModel -v

# Run only integration tests
python -m pytest tests/test_mlp_model.py::TestMLPIntegration -v

# Run only edge case tests
python -m pytest tests/test_mlp_model.py::TestMLPEdgeCases -v
```

### Running Individual Tests

```bash
# Test model creation
python -m pytest tests/test_mlp_model.py::TestMLPModel::test_simple_mlp_model_creation -v

# Test model loading
python -m pytest tests/test_mlp_model.py::TestMLPModel::test_load_pretrained_mlp_success -v

# Test scoring functionality
python -m pytest tests/test_mlp_model.py::TestMLPModel::test_mlp_score_with_model -v
```

## Test Coverage

The test suite covers:

- ✅ **Model Architecture**: SimpleMLP class with configurable dimensions
- ✅ **Model Loading**: Safe loading with weights_only=True and fallback mechanisms
- ✅ **Scoring Functions**: Both MLP-based and heuristic scoring methods
- ✅ **Input Validation**: Robust error handling and fallback mechanisms
- ✅ **DDE Features**: Dynamic Data Exchange feature extraction and validation
- ✅ **Integration**: End-to-end pipeline integration testing
- ✅ **Edge Cases**: NaN/Inf handling, empty inputs, dimension mismatches
- ✅ **Persistence**: Model saving and loading with state dictionaries

## Key Features Tested

### MLP Model Architecture
- Input dimension: 773 (768 embedding dimensions + 5 DDE features)
- Hidden layers: 64 neurons each
- Output: Single score value
- Activation: ReLU

### DDE Features
The tests validate the following DDE (Dynamic Data Exchange) features:
- `num_nodes`: Number of nodes in the subgraph
- `num_edges`: Number of edges in the subgraph  
- `avg_degree`: Average node degree
- `density`: Graph density (0-1)
- `clustering_coefficient`: Clustering coefficient (0-1)

### Error Handling
- Graceful fallback to heuristic scoring when MLP fails
- Proper handling of missing or corrupted model files
- Input validation with meaningful error messages
- Device compatibility (CPU/GPU) handling

## Expected Test Results

When all tests pass, you should see:
```
21 passed, X warnings in Y.ZZs
```

The warnings are typically related to:
- FAISS library deprecation warnings (safe to ignore)
- PyTorch tensor construction recommendations (informational)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the virtual environment is activated and all dependencies are installed
2. **Model Loading Failures**: Check that the test creates temporary model files correctly
3. **Dimension Mismatches**: Verify that the SimpleMLP input dimension matches the test data (773)
4. **Device Errors**: Tests are designed to run on CPU; GPU is not required

### Debug Mode

Run tests with additional debugging:
```bash
python -m pytest tests/test_mlp_model.py -v -s --tb=long
```

This will show:
- Detailed test output (`-s`)
- Verbose test names (`-v`) 
- Full tracebacks on failures (`--tb=long`)