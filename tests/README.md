# SubgraphRAG+ Test Suite

This directory contains the comprehensive test suite for the SubgraphRAG+ project, organized by test type and scope following testing best practices.

## Test Structure

```
tests/
├── README.md
├── conftest.py                 # shared fixtures & config
│
├── unit/                       # pure unit tests, no I/O or ML models
│   ├── entity_typing/
│   │   └── test_entity_typing.py
│   ├── triple_extraction/
│   │   └── test_triple_extraction.py
│   ├── mlp/
│   │   └── test_mlp.py
│   ├── llm/
│   │   └── test_llm.py
│   └── utils/
│       └── test_utils.py
│
├── integration/                # tests that spin up components & talk over interfaces
│   ├── api/
│   │   └── test_api.py
│   ├── embedder/
│   │   └── test_embedder.py
│   ├── retriever/
│   │   └── test_retriever.py
│   ├── llm/
│   │   └── test_llm_integration.py
│   └── general/
│       └── test_integration.py
│
├── e2e/                        # full-system smoke & domain workflows
│   └── llm_functionality/
│       └── test_llm_real_functionality.py
│
├── performance/                # load & latency benchmarks
│   └── test_llm_performance.py
│
├── adversarial/                # edge-case and adversarial robustness tests
│   └── test_adversarial.py
│
├── smoke/                      # very quick sanity checks
│   ├── test_smoke.py
│   ├── test_basic.py
│   ├── test_minimal.py
│   ├── test_ultra_minimal.py
│   └── test_fast.py
│
└── fixtures/                   # static data & mocks
    ├── expected_responses.json
    └── sample_prompts.json
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual functions and classes in isolation
- **Organization**: Grouped by feature/module (entity_typing, triple_extraction, mlp, llm, utils)
- **Characteristics**: 
  - Use extensive mocking to isolate components
  - Fast execution (< 1 second per test)
  - High coverage of edge cases and error conditions
  - No external dependencies or I/O operations

### Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions and workflows
- **Organization**: Grouped by feature/module (api, embedder, retriever, llm, general)
- **Characteristics**:
  - Limited mocking, focus on real component integration
  - Moderate execution time (1-10 seconds per test)
  - Test configuration loading, backend fallbacks, etc.
  - Spin up components and test interfaces

### End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete workflows with real backends
- **Organization**: Grouped by domain workflows (llm_functionality)
- **Characteristics**:
  - Minimal mocking, real functionality testing
  - Longer execution time (10+ seconds per test)
  - Require actual LLM backends to be available
  - Full-system smoke tests

### Performance Tests (`tests/performance/`)
- **Purpose**: Test load handling, latency, and resource usage
- **Characteristics**:
  - Measure response times, memory usage, concurrent handling
  - Long execution time (30+ seconds per test)
  - Generate performance metrics and reports
  - Load & latency benchmarks

### Adversarial Tests (`tests/adversarial/`)
- **Purpose**: Test edge cases and adversarial robustness
- **Characteristics**:
  - Test system behavior under stress
  - Edge case validation
  - Security and robustness testing
  - Malformed input handling

### Smoke Tests (`tests/smoke/`)
- **Purpose**: Very quick sanity checks
- **Characteristics**:
  - Ultra-fast execution (< 0.5 seconds per test)
  - Basic functionality verification
  - CI/CD pipeline health checks
  - Minimal resource usage

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run by Category
```bash
# Unit tests only (fast)
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests (requires real backends)
pytest tests/e2e/

# Performance tests (slow)
pytest tests/performance/

# Adversarial tests
pytest tests/adversarial/

# Smoke tests (ultra-fast)
pytest tests/smoke/
```

### Run by Feature/Module
```bash
# Entity typing tests
pytest tests/unit/entity_typing/

# Triple extraction tests
pytest tests/unit/triple_extraction/

# LLM tests (unit + integration)
pytest tests/unit/llm/ tests/integration/llm/

# API tests
pytest tests/integration/api/

# Embedder tests
pytest tests/integration/embedder/
```

### Run by Markers
```bash
# Run only fast tests
pytest -m "not slow"

# Run tests that require MLX
pytest -m "requires_mlx"

# Run performance tests
pytest -m "performance"

# Run smoke tests
pytest -m "smoke"
```

### Run Specific Test Files
```bash
# LLM unit tests
pytest tests/unit/llm/test_llm.py

# LLM integration tests
pytest tests/integration/llm/test_llm_integration.py

# Real LLM functionality (requires backends)
pytest tests/e2e/llm_functionality/test_llm_real_functionality.py

# Performance tests
pytest tests/performance/test_llm_performance.py

# Adversarial tests
pytest tests/adversarial/test_adversarial.py
```

## Test Configuration

### Environment Variables
- `TESTING=1`: Enable testing mode (mocked responses)
- `TESTING=0`: Disable testing mode (real LLM backends)

### Backend Requirements
- **MLX Tests**: Require MLX installation and model cache
- **OpenAI Tests**: Require `OPENAI_API_KEY` environment variable
- **HuggingFace Tests**: Require transformers and torch installation

### Test Fixtures
- `sample_prompts`: Pre-defined test prompts of various types
- `expected_responses`: Expected response patterns and quality checks
- `mock_llm_response`: Standard mock response for testing mode
- `performance_test_config`: Configuration for performance tests

## LLM Test Coverage

### Unit Tests (`test_llm.py`)
- ✅ Configuration validation
- ✅ Backend selection logic
- ✅ Generation with different parameters
- ✅ Error handling and fallbacks
- ✅ Streaming functionality
- ✅ Health check mechanisms
- ✅ Mock model interactions

### Integration Tests (`test_llm_integration.py`)
- ✅ End-to-end generation workflow
- ✅ Parameter validation across system
- ✅ Concurrent request handling
- ✅ Backend fallback integration
- ✅ Configuration loading
- ✅ Memory and performance monitoring

### E2E Tests (`test_llm_real_functionality.py`)
- ✅ Real backend availability detection
- ✅ Actual response generation and validation
- ✅ Parameter variation testing
- ✅ Streaming with real backends
- ✅ Context understanding capabilities
- ✅ Response consistency testing

### Performance Tests (`test_llm_performance.py`)
- ✅ Single request latency measurement
- ✅ Streaming performance metrics
- ✅ Concurrent load testing
- ✅ Sustained load over time
- ✅ Memory usage monitoring
- ✅ CPU usage pattern analysis

## MLP Test Coverage

### Unit Tests (`test_mlp.py`)
- ✅ Model loading and configuration
- ✅ Scoring with mock models
- ✅ Fallback to heuristic scoring
- ✅ Input validation and error handling
- ✅ Architecture compatibility testing
- ✅ Performance characteristics

## Test Quality Metrics

### Current Status
- **Total Tests**: 50+ tests across all categories
- **Unit Test Coverage**: 95%+ of LLM and MLP modules
- **Integration Coverage**: All major component interactions
- **Performance Benchmarks**: Latency, throughput, resource usage
- **E2E Validation**: Real functionality when backends available

### Performance Benchmarks
- **Unit Test Speed**: < 1s per test
- **Integration Test Speed**: 1-10s per test
- **LLM Response Time**: < 30s (real backends)
- **Memory Usage**: < 100MB increase during testing
- **Concurrent Handling**: 10+ simultaneous requests

## Continuous Integration

### Test Execution Strategy
1. **PR Validation**: Unit + Integration tests (fast feedback)
2. **Nightly Builds**: Full test suite including E2E and performance
3. **Release Validation**: Complete test suite with real backends

### Test Environment Matrix
- **Testing Mode**: Fast execution with mocks
- **MLX Backend**: Real MLX model testing (when available)
- **OpenAI Backend**: API integration testing (with key)
- **HuggingFace Backend**: Transformer model testing

## Contributing

### Adding New Tests
1. Choose appropriate test category (unit/integration/e2e/performance)
2. Use existing fixtures and patterns
3. Follow naming conventions: `test_<functionality>.py`
4. Add appropriate pytest markers
5. Update this README if adding new test categories

### Test Best Practices
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Mock external dependencies in unit tests
- Use real components in integration tests when possible
- Add performance assertions for critical paths
- Clean up resources in test teardown

### Debugging Tests
```bash
# Run with verbose output
pytest -v tests/unit/test_llm.py

# Run with debug logging
pytest -s --log-cli-level=DEBUG tests/

# Run specific test method
pytest tests/unit/test_llm.py::TestLLMGeneration::test_generate_answer_basic
```