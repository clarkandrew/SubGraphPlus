# SubgraphRAG+ Developer Guide

## Introduction

Welcome to the SubgraphRAG+ developer guide! This document provides essential information for developers working with the SubgraphRAG+ codebase. Whether you're extending functionality, fixing bugs, or integrating the system into your own applications, this guide will help you understand the system's architecture and development workflow.

## Getting Started

### Development Environment Setup

1. **Clone the repository**

```bash
git clone https://github.com/clarkandrew/SubgraphRAGPlus.git
cd SubgraphRAGPlus
```

2. **Set up your development environment**

```bash
make setup-dev
```

3. **Configure environment variables**

```bash
# Create a .env file
cp .env.example .env
# Edit .env with your settings
nano .env
```

4. **Start Neo4j for development**

```bash
make neo4j-start
```

### Development Workflow

1. **Create a feature branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**

3. **Run the linter and tests**

```bash
make lint
make test
```

4. **Start the development server**

```bash
make serve
```

## Project Structure

### Key Directories

- `app/`: Core application code
  - `api.py`: FastAPI application and endpoints
  - `config.py`: Configuration management
  - `database.py`: Database connections
  - `models.py`: Data models
  - `retriever.py`: Retrieval logic
  - `utils.py`: Utility functions
  - `verify.py`: Output verification
  - `ml/`: Machine learning modules
- `config/`: Configuration files
- `data/`: Data storage
- `docs/`: Documentation
- `migrations/`: Neo4j schema migrations
- `models/`: ML model storage
- `prompts/`: Prompt templates
- `scripts/`: Utility scripts
- `tests/`: Test suite

### Key Files

- `main.py`: Application entry point
- `Makefile`: Common development commands
- `requirements.txt`: Production dependencies
- `requirements-dev.txt`: Development dependencies

## Core Development Concepts

### Configuration System

SubgraphRAG+ uses a JSON-based configuration system with schema validation:

- `config/config.schema.json`: Defines the schema for validation
- `config/config.json`: Main configuration file
- `app/config.py`: Configuration loading and validation

To add a new configuration parameter:

1. Add it to `config.schema.json` with appropriate type and description
2. Access it in code via `config.YOUR_PARAMETER`

### Database Layer

#### Neo4j Connection

The Neo4j connection is managed in `app/database.py` with the `Neo4jDatabase` class:

```python
# Example: Run a Neo4j query
result = neo4j_db.run_query(
    "MATCH (e:Entity {name: $name}) RETURN e",
    {"name": "Tesla Inc."}
)
```

#### SQLite Connection

SQLite operations use the `SQLiteDatabase` class:

```python
# Example: Execute a SQLite query
sqlite_db.execute(
    "INSERT INTO feedback (query_id, is_correct) VALUES (?, ?)",
    (query_id, is_correct)
)
```

### Entity Linking

Entity linking is a critical component that connects natural language mentions to knowledge graph entities:

1. Extraction: `extract_query_entities()` extracts potential entity mentions
2. Linking: `link_entities_v2()` links mentions to KG entities
3. Resolution: Handles exact matches, aliases, and fuzzy matching

### Hybrid Retrieval

The hybrid retrieval system combines graph traversal and vector search:

1. Graph retrieval: `neo4j_get_neighborhood_triples()`
2. Dense retrieval: `faiss_search_triples_data()`
3. Candidate fusion: `hybrid_retrieve_v2()`
4. Subgraph assembly: `greedy_connect_v2()`

### LLM Integration

The LLM abstraction layer in `app/ml/llm.py` supports multiple backends:

- OpenAI API
- Hugging Face models
- MLX (Apple Silicon)

To add a new LLM backend:

1. Add a new implementation function in `llm.py`
2. Update the `generate_answer()` function to include your backend
3. Add appropriate configuration in `config.schema.json`

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific tests
pytest tests/test_retriever.py -v
```

### Test Structure

- `tests/test_*.py`: Test files
- `conftest.py`: Pytest fixtures
- `tests/resources/`: Test data

### Mocking Dependencies

For tests that require mocking:

```python
@pytest.fixture
def mock_neo4j():
    with patch('app.database.neo4j_db') as mock:
        # Configure mock
        yield mock
```

## API Extension

### Adding a New Endpoint

1. Define request/response models in `app/models.py`
2. Add the endpoint in `app/api.py`:

```python
@app.post("/your-endpoint")
async def your_endpoint(
    request: YourRequestModel,
    api_key: str = Depends(get_api_key)
):
    # Implementation
    return {"result": "success"}
```

### Adding API Documentation

- Update OpenAPI tags and descriptions in the endpoint decorator
- Add examples using FastAPI's `example` parameter

## Deployment

### Production Deployment

1. Build the Docker image:

```bash
docker build -t subgraphrag-plus .
```

2. Run with Docker Compose:

```bash
docker-compose up -d
```

### Environment Variables for Production

- `NEO4J_URI`: Neo4j connection URI
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `API_KEY_SECRET`: API authentication key
- `MODEL_BACKEND`: LLM backend to use

## Troubleshooting

### Common Issues

#### Neo4j Connection Issues

- Check Neo4j is running: `docker ps | grep neo4j`
- Verify connection settings in `.env` file
- Ensure APOC plugin is installed

#### FAISS Index Issues

- Check index file exists: `ls -la data/faiss_index.bin`
- Rebuild index: `make rebuild-faiss-index`
- Check embeddings exist in staging: `ls -la data/faiss_staging/`

#### LLM Backend Issues

- Check API keys for external services
- Verify model paths for local models
- Check logs for specific errors: `tail -f logs/app.log`

### Logging

Access logs in the `logs/` directory:

- `app.log`: Main application logs
- `ingest_worker.log`: Ingestion worker logs
- `migration.log`: Schema migration logs

## Performance Optimization

### FAISS Optimization

- Adjust `nlist` parameter for IVF index based on dataset size
- Consider using GPU-enabled FAISS for large datasets
- Use periodic retraining: `python scripts/merge_faiss.py --rebuild`

### Neo4j Query Optimization

- Use parameterized queries to leverage query caching
- Add appropriate indexes for frequently filtered properties
- Use EXPLAIN to analyze query performance

### Embedding Caching

- Adjust cache size in `app/utils.py` for your memory constraints
- Monitor cache hit rates through logs
- Consider persistent caching for frequent queries

## Contributing

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Document classes and functions with docstrings
- Keep functions focused on a single responsibility

### Pull Request Process

1. Ensure all tests pass: `make test`
2. Run linting checks: `make lint`
3. Update documentation if necessary
4. Submit PR with clear description of changes

## Further Resources

- [API Reference](api_reference.md)
- [Architecture Documentation](architecture.md)
- [System Overview](overview.md)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
