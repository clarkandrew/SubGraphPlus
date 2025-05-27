# SubgraphRAG+ Test Suite Makefile
# Provides convenient commands for running organized tests

.PHONY: help test test-all test-unit test-integration test-e2e test-performance test-adversarial test-smoke
.PHONY: test-fast test-feature test-llm test-entity test-triple test-api test-embedder test-retriever
.PHONY: coverage clean

# Default target
help:
	@echo "SubgraphRAG+ Test Suite Commands"
	@echo "================================"
	@echo ""
	@echo "Test Categories:"
	@echo "  test-all         Run all tests (smoke -> unit -> integration -> adversarial -> e2e -> performance)"
	@echo "  test-unit        Run unit tests only (fast, isolated)"
	@echo "  test-integration Run integration tests (component interactions)"
	@echo "  test-e2e         Run end-to-end tests (requires real backends)"
	@echo "  test-performance Run performance/load tests (slow)"
	@echo "  test-adversarial Run adversarial/edge-case tests"
	@echo "  test-smoke       Run smoke tests (ultra-fast sanity checks)"
	@echo ""
	@echo "Quick Options:"
	@echo "  test-fast        Run only fast tests (smoke + unit + integration)"
	@echo "  test             Alias for test-fast"
	@echo ""
	@echo "Feature-Specific:"
	@echo "  test-llm         Run all LLM tests (unit + integration + e2e)"
	@echo "  test-entity      Run entity typing tests"
	@echo "  test-triple      Run triple extraction tests"
	@echo "  test-api         Run API tests"
	@echo "  test-embedder    Run embedder tests"
	@echo "  test-retriever   Run retriever tests"
	@echo ""
	@echo "Utilities:"
	@echo "  coverage         Generate test coverage report"
	@echo "  clean            Clean test artifacts and cache"

# Main test targets
test: test-fast

test-all:
	python run_tests.py all

test-unit:
	python run_tests.py unit

test-integration:
	python run_tests.py integration

test-e2e:
	python run_tests.py e2e

test-performance:
	python run_tests.py performance

test-adversarial:
	python run_tests.py adversarial

test-smoke:
	python run_tests.py smoke

test-fast:
	python run_tests.py all --fast

# Feature-specific tests
test-llm:
	python run_tests.py --feature llm

test-entity:
	python run_tests.py --feature entity_typing

test-triple:
	python run_tests.py --feature triple_extraction

test-api:
	python run_tests.py --feature api

test-embedder:
	python run_tests.py --feature embedder

test-retriever:
	python run_tests.py --feature retriever

# Coverage and utilities
coverage:
	pytest unit/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	rm -rf logs/*.log 2>/dev/null || true
	@echo "Test artifacts cleaned"

# CI/CD targets
ci-fast:
	python run_tests.py smoke
	python run_tests.py unit
	python run_tests.py integration --markers "not slow"

ci-full:
	python run_tests.py all

# Development targets
dev-watch:
	pytest-watch unit/ integration/ --clear

dev-debug:
	pytest unit/ integration/ -v -s --tb=long 