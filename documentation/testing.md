# SubgraphRAG+ Testing Guide

## Overview

This document describes the testing strategy, coverage, and instructions for running tests in the SubgraphRAG+ project.

---

## Test Types

- **Unit Tests**: Validate individual functions and modules (e.g., utils, retriever, entity linking).
- **Integration Tests**: Validate end-to-end flows (e.g., API endpoints, ingestion, retrieval).
- **Adversarial/Edge Case Tests**: Ensure robustness against malformed input, empty queries, ambiguous entities, etc.
- **Benchmark/Evaluation**: Scripts for performance and accuracy evaluation.

---

## Directory Structure

- `tests/` — All test code
  - `test_utils.py` — Utility function tests
  - `test_retriever.py` — Retriever logic tests
  - `test_api.py` — API endpoint tests
  - `conftest.py` — Pytest fixtures and mocks

---

## Running Tests

### **Locally**

```bash
make test
```
or
```bash
pytest -v
```

### **Inside Docker**

```bash
docker-compose run subgraphrag pytest -v
```

---

## Coverage

- **Entity Linking**: Extraction, alias/fuzzy/contextual linking, ambiguous/negative cases
- **Retriever**: Graph/dense retrieval, MLP/heuristic scoring, fusion logic
- **API**: All endpoints, error handling, streaming, feedback, ingestion
- **Ingestion**: Staging, deduplication, Neo4j/FAISS sync
- **Evaluation**: End-to-end QA accuracy, latency, and trust metrics

---

## Adding New Tests

- Place new test files in `tests/` and name them `test_*.py`
- Use pytest fixtures from `conftest.py` for mocks
- Use FastAPI’s `TestClient` for API endpoint tests

---

## CI Integration

- All tests are run automatically in GitHub Actions on each PR.
- Linting and smoke tests are required for merge.

---

## Troubleshooting

- Check logs in `logs/` for failures.
- Use `pytest -s` for verbose output.
- For database-related tests, ensure Neo4j is running and accessible.
