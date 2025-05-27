# SubgraphRAG+ Test Makefile
# 
# This Makefile provides convenient targets for running tests according to
# the testing standards with proper categorization and timeout handling.

.PHONY: help test test-smoke test-unit test-integration test-e2e test-performance test-adversarial test-coverage test-debug clean

# Default target
help:
	@echo "SubgraphRAG+ Test Suite"
	@echo "======================="
	@echo ""
	@echo "Available targets:"
	@echo "  test              - Run smoke tests (default, quick sanity check)"
	@echo "  test-smoke        - Run smoke tests (<30s total)"
	@echo "  test-unit         - Run unit tests (<2 minutes)"
	@echo "  test-integration  - Run integration tests (<5 minutes)"
	@echo "  test-e2e          - Run end-to-end tests (<10 minutes)"
	@echo "  test-performance  - Run performance tests"
	@echo "  test-adversarial  - Run adversarial/robustness tests"
	@echo "  test-all          - Run all test suites"
	@echo "  test-coverage     - Run unit tests with coverage report"
	@echo "  test-debug        - Run smoke tests with debug output"
	@echo "  clean             - Clean test artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make test                    # Quick smoke tests"
	@echo "  make test-unit               # Unit tests only"
	@echo "  make test-coverage           # Unit tests with coverage"
	@echo "  make test-all                # All test suites"

# Default test target (smoke tests for quick feedback)
test: test-smoke

# Individual test suite targets
test-smoke:
	@echo "🧪 Running smoke tests..."
	python run_tests.py --suite smoke

test-unit:
	@echo "🧪 Running unit tests..."
	python run_tests.py --suite unit

test-integration:
	@echo "🧪 Running integration tests..."
	python run_tests.py --suite integration

test-e2e:
	@echo "🧪 Running end-to-end tests..."
	python run_tests.py --suite e2e

test-performance:
	@echo "🧪 Running performance tests..."
	python run_tests.py --suite performance

test-adversarial:
	@echo "🧪 Running adversarial tests..."
	python run_tests.py --suite adversarial

test-all:
	@echo "🧪 Running all test suites..."
	python run_tests.py --suite all

# Special targets
test-coverage:
	@echo "🧪 Running unit tests with coverage..."
	python run_tests.py --suite unit --coverage

test-debug:
	@echo "🧪 Running smoke tests with debug output..."
	python run_tests.py --suite smoke --debug --verbose

# Utility targets
clean:
	@echo "🧹 Cleaning test artifacts..."
	rm -rf tests/coverage_html/
	rm -rf tests/.coverage
	rm -rf tests/.pytest_cache/
	rm -rf tests/__pycache__/
	find tests/ -name "*.pyc" -delete
	find tests/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Test artifacts cleaned"

# Continuous Integration targets
ci-smoke:
	@echo "🚀 CI: Running smoke tests..."
	timeout 60s python run_tests.py --suite smoke --verbose

ci-unit:
	@echo "🚀 CI: Running unit tests with coverage..."
	timeout 180s python run_tests.py --suite unit --coverage --verbose

ci-integration:
	@echo "🚀 CI: Running integration tests..."
	timeout 360s python run_tests.py --suite integration --verbose

ci-all:
	@echo "🚀 CI: Running all tests..."
	timeout 1800s python run_tests.py --suite all --coverage --verbose

# Development targets
dev-fast:
	@echo "⚡ Running fast development tests..."
	python run_tests.py --suite smoke --verbose

dev-ie:
	@echo "🔍 Running IE-specific tests..."
	python -m pytest smoke/test_ie_consolidated.py unit/services/test_information_extraction.py -v

dev-watch:
	@echo "👀 Running tests in watch mode..."
	python -m pytest smoke/ --verbose --tb=short -f

# Help target for listing available test suites
list-suites:
	python run_tests.py --list-suites 