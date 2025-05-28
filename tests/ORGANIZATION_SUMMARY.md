# SubgraphRAG+ Test Organization Summary

This document summarizes the consolidated test structure following testing standards and best practices.

## Test Structure Overview

```
tests/
├── smoke/                          # Quick sanity checks (<30s total)
│   ├── test_smoke_consolidated.py  # Main smoke tests for all components
│   ├── test_ie_consolidated.py     # IE-specific smoke tests
│   └── test_ie_real_models.py      # Real model loading tests (optional)
├── unit/                           # Pure logic tests (<2 minutes)
│   ├── services/
│   │   ├── test_information_extraction.py
│   │   └── test_ingestion.py
│   ├── entity_typing/
│   ├── triple_extraction/
│   ├── llm/
│   ├── mlp/
│   └── utils/
├── integration/                    # Real component interactions (<5 minutes)
│   ├── api/
│   │   └── test_ie_endpoints.py
│   ├── embedder/
│   ├── retriever/
│   └── general/
├── e2e/                           # End-to-end system flows (<10 minutes)
│   └── test_ie_pipeline_e2e.py
├── performance/                   # Load and latency benchmarks
├── adversarial/                   # Robustness and edge cases
├── fixtures/                      # Test data and configurations
├── conftest.py                    # Shared fixtures and configuration
├── run_tests.py                   # Comprehensive test runner
├── Makefile                       # Convenient test commands
└── README.md                      # Test documentation
```

## Consolidation Changes Made

### Removed Duplicate Files
- **Smoke Tests**: Consolidated 17 duplicate IE smoke test files into 2 clean files
- **Debug Files**: Removed all debug test files (test_ie_debug*.py)
- **Minimal Tests**: Removed redundant minimal test variations
- **Old Runners**: Removed old IE-specific test runner

### Consolidated Files
1. **`test_smoke_consolidated.py`**: Main smoke tests covering:
   - Basic app functionality
   - API endpoints
   - Information extraction
   - Input sanitization
   - Error handling
   - Performance basics
   - Concurrency
   - Configuration

2. **`test_ie_consolidated.py`**: IE-specific smoke tests covering:
   - Import validation
   - Service instantiation
   - Singleton pattern
   - Model loading (when enabled)
   - Extraction functionality
   - Error handling
   - Performance characteristics

### Updated Infrastructure
1. **`run_tests.py`**: Comprehensive test runner with:
   - Proper test suite categorization
   - Timeout handling
   - Environment setup
   - Rich logging and reporting
   - Coverage support
   - Debug modes

2. **`Makefile`**: Clean targets following standards:
   - Individual test suite targets
   - Coverage reporting
   - CI/CD targets
   - Development helpers

## Test Categories and Standards

### Smoke Tests (tests/smoke/)
- **Purpose**: Quick validation that core functionality works
- **Duration**: <30 seconds total
- **Scope**: Basic operations, health checks, configuration validation
- **Execution**: Parallel where possible

### Unit Tests (tests/unit/)
- **Purpose**: Test individual functions and classes in isolation
- **Duration**: <2 minutes total
- **Scope**: Pure logic, minimal external dependencies
- **Mocking**: Only for true externalities (network, file system, time)

### Integration Tests (tests/integration/)
- **Purpose**: Test real interactions between components
- **Duration**: <5 minutes total
- **Scope**: API endpoints, database operations, service integrations
- **Mocking**: Minimal - prefer real implementations

### End-to-End Tests (tests/e2e/)
- **Purpose**: Test complete user workflows
- **Duration**: <10 minutes total
- **Scope**: Full system behavior from input to output
- **Mocking**: None - test the real system

### Performance Tests (tests/performance/)
- **Purpose**: Validate latency, throughput, and resource usage
- **Scope**: Load testing, memory profiling, concurrent operations
- **Metrics**: Response times, memory usage, CPU utilization

### Adversarial Tests (tests/adversarial/)
- **Purpose**: Test system robustness and edge cases
- **Scope**: Malformed inputs, boundary conditions, error scenarios
- **Focus**: Security, stability, graceful degradation

## Running Tests

### Quick Commands
```bash
# Default (smoke tests)
make test

# Specific test suites
make test-unit
make test-integration
make test-e2e

# With coverage
make test-coverage

# All tests
make test-all
```

### Advanced Usage
```bash
# Using the test runner directly
python tests/run_tests.py --suite smoke
python tests/run_tests.py --suite unit --verbose
python tests/run_tests.py --suite all --coverage

# List available suites
python tests/run_tests.py --list-suites
```

## Environment Configuration

### Test Environment Variables
- `TESTING=1`: Enable testing mode
- `SUBGRAPHRAG_DISABLE_MODEL_LOADING=true`: Disable model loading for tests
- `TOKENIZERS_PARALLELISM=false`: Prevent threading issues
- `PYTORCH_ENABLE_MPS_FALLBACK=1`: Apple Silicon compatibility

### Model Loading Strategy
- **Unit/Smoke Tests**: Models disabled by default to prevent segfaults
- **Integration Tests**: Can enable models with environment variable
- **E2E Tests**: Use real models when available
- **Performance Tests**: Real models for accurate benchmarks

## Best Practices Implemented

### Testing Philosophy
- ✅ Test behavior, not implementation
- ✅ Maximize realism (minimal mocking)
- ✅ Comprehensive coverage (normal, edge, error flows)
- ✅ Fast feedback (smoke tests first)

### Test Implementation
- ✅ One behavior per test
- ✅ Arrange-Act-Assert structure
- ✅ Meaningful test names
- ✅ Clear assertions with descriptive messages

### Mocking Guidelines
- ✅ Mock only externalities
- ✅ Never mock own code
- ✅ Document all mocks
- ✅ Prefer real implementations

### Performance Considerations
- ✅ Fast unit tests (<1s each)
- ✅ Parallel execution where safe
- ✅ Proper resource cleanup
- ✅ Timeout protection

## Quality Metrics

### Coverage Targets
- Unit Tests: >95% line coverage
- Integration Tests: >90% feature coverage
- E2E Tests: >80% user workflow coverage

### Performance Budgets
- Unit Test Suite: <2 minutes
- Integration Test Suite: <5 minutes
- E2E Test Suite: <10 minutes
- Smoke Test Suite: <30 seconds

### Quality Gates
- All tests must pass before merge
- Coverage cannot decrease
- Performance budgets enforced
- No skipped tests without justification

## Information Extraction (IE) Testing

### IE Test Strategy
- **Real Models**: Use cached models for integration/e2e tests
- **Mocked Models**: Use mocks for unit tests to avoid model loading overhead
- **Performance**: Measure extraction speed and memory usage
- **Robustness**: Test with various text inputs and edge cases

### IE-Specific Considerations
- Model loading disabled by default to prevent Apple Silicon segfaults
- Threading conflicts avoided with proper environment variables
- FastAPI TestClient issues worked around with mocking
- Real model tests available when explicitly enabled

## Continuous Integration

### CI Test Execution
- All test categories run on every PR
- Coverage enforcement (>90% for critical modules)
- Performance regression detection
- Flake rate monitoring (<1%)

### CI Targets
```bash
make ci-smoke      # Quick CI validation
make ci-unit       # Unit tests with coverage
make ci-integration # Integration tests
make ci-all        # Full CI test suite
```

## Development Workflow

### Development Targets
```bash
make dev-fast      # Quick development tests
make dev-ie        # IE-specific tests
make dev-watch     # Watch mode for TDD
```

### Test-Driven Development
1. Write failing test first
2. Implement minimal fix
3. Refactor for clarity
4. Run full suite before commit

## Maintenance

### Regular Tasks
- Quarterly test audits to prune obsolete tests
- Performance budget reviews
- Flake rate monitoring and fixes
- Coverage drift alerts

### Test Health Metrics
- Flake Rate: <1%
- Test Maintenance Ratio: <10% of development time
- Bug Escape Rate: <5% of releases 