# SubgraphRAG+ Test Suite

This directory contains comprehensive tests for the SubgraphRAG+ Information Extraction (IE) pipeline, organized according to testing best practices.

## Test Organization

```
tests/
├── unit/           # Pure logic tests, minimal mocking
├── integration/    # Real component interactions
├── e2e/            # End-to-end system flows
├── performance/    # Load and latency tests
├── adversarial/    # Robustness and edge cases
├── smoke/          # Quick sanity checks
├── fixtures/       # Test data and configurations
├── conftest.py     # Shared fixtures
└── README.md       # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual functions and classes in isolation
- **Scope**: Pure logic, minimal external dependencies
- **Mocking**: Only for true externalities (network, file system, time)
- **Speed**: Fast (<1s per test)

### Integration Tests (`tests/integration/`)
- **Purpose**: Test real interactions between components
- **Scope**: API endpoints, database operations, service integrations
- **Mocking**: Minimal - prefer real implementations
- **Speed**: Medium (1-10s per test)

### End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete user workflows
- **Scope**: Full system behavior from input to output
- **Mocking**: None - test the real system
- **Speed**: Slower (10-60s per test)

### Performance Tests (`tests/performance/`)
- **Purpose**: Validate latency, throughput, and resource usage
- **Scope**: Load testing, memory profiling, concurrent operations
- **Metrics**: Response times, memory usage, CPU utilization

### Adversarial Tests (`tests/adversarial/`)
- **Purpose**: Test system robustness and edge cases
- **Scope**: Malformed inputs, boundary conditions, error scenarios
- **Focus**: Security, stability, graceful degradation

### Smoke Tests (`tests/smoke/`)
- **Purpose**: Quick validation that core functionality works
- **Scope**: Basic operations, health checks, configuration validation
- **Speed**: Very fast (<30s total)

## Running Tests

### Quick Start
```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-e2e
make test-smoke

# Run with coverage
make test-coverage
```

### Detailed Test Runner
```bash
# Use the comprehensive test runner
python tests/run_tests.py --help

# Run specific test suites
python tests/run_tests.py --suite smoke
python tests/run_tests.py --suite unit
python tests/run_tests.py --suite integration
```

## Test Data Management

### Fixtures (`tests/fixtures/`)
- Static test data for reproducible tests
- Versioned configurations and sample inputs
- Expected outputs for validation

### Dynamic Test Data
- Generated using factories and builders
- Property-based testing for edge cases
- Sanitized real-world examples

## Information Extraction (IE) Tests

### Core IE Functionality
- **Model Loading**: Verify REBEL and NER models load correctly
- **Triple Extraction**: Test extraction accuracy and format
- **Entity Typing**: Validate entity classification
- **Service Integration**: Test IE service wrapper and API

### Test Strategy
- **Real Models**: Use cached models for integration/e2e tests
- **Mocked Models**: Use mocks for unit tests to avoid model loading overhead
- **Performance**: Measure extraction speed and memory usage
- **Robustness**: Test with various text inputs and edge cases

## Best Practices

### Test Writing
1. **One Behavior Per Test**: Each test validates one specific behavior
2. **Arrange-Act-Assert**: Clear test structure
3. **Meaningful Names**: `test_<function>_<scenario>_<expected>()`
4. **Clear Assertions**: Descriptive error messages

### Mocking Guidelines
1. **Mock Only Externalities**: Network, file system, time, randomness
2. **Never Mock Your Own Code**: Test real implementations
3. **Document Mocks**: Explain why real code cannot be used
4. **Prefer Real Over Mock**: Default to testing actual implementations

### Performance Considerations
1. **Fast Unit Tests**: Keep under 1 second each
2. **Parallel Execution**: Tests must be independent
3. **Resource Cleanup**: Proper teardown of test resources
4. **Timeout Protection**: Prevent hanging tests

## Continuous Integration

### Test Execution
- All test categories run on every PR
- Coverage enforcement (>90% for critical modules)
- Performance regression detection
- Flake rate monitoring (<1%)

### Quality Gates
- All tests must pass before merge
- Coverage cannot decrease
- Performance budgets enforced
- No skipped tests without justification

## Troubleshooting

### Common Issues
1. **Model Loading Failures**: Check `SUBGRAPHRAG_DISABLE_MODEL_LOADING` environment variable
2. **Segmentation Faults**: Use mocked tests on Apple Silicon for model-heavy operations
3. **Hanging Tests**: Check for threading conflicts with FastAPI TestClient
4. **Import Errors**: Verify PYTHONPATH includes project root

### Debug Mode
```bash
# Run with verbose output
python tests/run_tests.py --verbose --debug

# Run single test with debugging
pytest -xvs tests/unit/services/test_information_extraction.py::test_extract_triples_success
```

## Contributing

### Adding New Tests
1. Choose appropriate test category based on scope
2. Follow naming conventions and structure
3. Add fixtures to `tests/fixtures/` if needed
4. Update this README if adding new test patterns

### Test Review Checklist
- [ ] Tests follow Arrange-Act-Assert pattern
- [ ] Minimal and justified mocking
- [ ] Clear, descriptive test names
- [ ] Proper error handling and edge cases
- [ ] Performance considerations addressed
- [ ] Documentation updated if needed

## Metrics and Monitoring

### Coverage Targets
- Unit Tests: >95% line coverage
- Integration Tests: >90% feature coverage
- E2E Tests: >80% user workflow coverage

### Performance Budgets
- Unit Test Suite: <2 minutes
- Integration Test Suite: <5 minutes
- E2E Test Suite: <10 minutes
- Smoke Test Suite: <30 seconds

### Quality Metrics
- Flake Rate: <1%
- Test Maintenance Ratio: <10% of development time
- Bug Escape Rate: <5% of releases