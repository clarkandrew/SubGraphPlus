# 🚀 SubgraphRAG+

> Knowledge Graph Question Answering with Hybrid Retrieval and Visualization

SubgraphRAG+ is a knowledge graph-powered QA system that combines structured graph traversal with dense vector retrieval to provide accurate, contextual answers with explanatory visualizations. Built on the original SubgraphRAG research, this enhanced version adds dynamic knowledge graph ingestion, hybrid retrieval, and enterprise-grade API features.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## 🌟 Key Features

- **Hybrid Retrieval**: Combines graph traversal and semantic search for optimal recall
- **Dynamic KG Ingestion**: Real-time triple ingestion with deduplication
- **SSE Streaming**: Token-by-token LLM response with citations and graph data
- **Visualization-Ready**: D3.js compatible graph data with relevance scores
- **Enterprise-Grade API**: OpenAPI compliant with comprehensive documentation

## 📂 Directory Structure

```
SubgraphRAG+/
├── src/app/           # Core application code
│   ├── api.py         # FastAPI endpoints
│   ├── models.py      # Data models
│   ├── config.py      # Configuration management
│   ├── database.py    # Database connections
│   ├── retriever.py   # Hybrid retrieval engine
│   └── ml/            # ML components
├── bin/               # Executable scripts for setup and operation
├── scripts/           # Python utility scripts
├── tools/             # Development and maintenance utilities
├── deployment/        # Docker and infrastructure files
├── config/            # Configuration files
├── data/              # Data storage
├── docs/              # Detailed documentation
├── tests/             # Test suite
├── main.py            # Application entry point
└── Makefile           # Build and development commands
```

## 📋 Prerequisites

- **Docker Setup** (Recommended):
  - Docker Engine 20.10+
  - Docker Compose v2
  - 4GB+ RAM allocated to Docker
  - 10GB+ free disk space

- **Local Development** (Alternative):
  - Python 3.11+
  - Neo4j 4.4+ with APOC plugin
  - SQLite3
  - Virtual environment (venv, conda, etc.)
  - (Optional) OpenAI API key for OpenAI backend

## 🚀 Quick Start

### Option 1: Docker Setup (Recommended)

```bash
# Clone and setup
git clone https://github.com/clarkandrew/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Complete setup in one command
make setup-all

# Test the system
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: changeme" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who founded Tesla?", "visualize_graph": true}'
```

### Option 2: Local Development Setup

```bash
# Clone repository
git clone https://github.com/clarkandrew/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Quick start with interactive setup
./bin/start.sh

# Or manual setup
./bin/setup_dev.sh --use-local-neo4j
./bin/run.sh
```

### Access Points

- **API Documentation**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474 (neo4j/password)
- **API Endpoint**: http://localhost:8000

## 🔧 Development Commands

### Make Commands (Docker-based)

```bash
# Setup and development
make setup-all          # Complete setup with Docker
make serve              # Start development server
make test               # Run test suite
make lint               # Code quality checks

# Docker operations
make docker-start       # Start Docker services
make docker-stop        # Stop Docker services
make docker-build       # Build Docker images

# Database operations
make neo4j-start        # Start Neo4j container
make migrate-schema     # Run database migrations
make ingest-sample      # Load sample data

# Utilities
make clean              # Clean temporary files
make help               # Show all commands
```

### Shell Scripts (All setups)

```bash
# Core operations
./bin/start.sh          # Interactive setup and start
./bin/run.sh            # Start API server
./bin/run_tests.sh      # Run tests
./bin/setup_dev.sh      # Development environment setup

# Database management
./bin/install_neo4j.sh  # Install Neo4j locally
./bin/setup_docker.sh   # Docker environment setup

# Utilities
./bin/demo.sh           # Interactive demo
./bin/backup.sh         # Backup system data
./bin/run_benchmark.sh  # Performance benchmarks
```

## ⚙️ Configuration

### Environment Variables (.env file)

```bash
# Database
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Security
API_KEY_SECRET=changeme_in_production

# Model Backend
MODEL_BACKEND=openai
# OPENAI_API_KEY=your_key_here
```

### Application Config (config/config.json)

```json
{
  "MODEL_BACKEND": "openai",
  "FAISS_INDEX_PATH": "data/faiss_index.bin",
  "TOKEN_BUDGET": 4000,
  "MLP_MODEL_PATH": "models/mlp_pretrained.pt",
  "CACHE_DIR": "cache/",
  "MAX_DDE_HOPS": 2,
  "LOG_LEVEL": "INFO",
  "API_RATE_LIMIT": 60
}
```

## 📊 API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Main QA endpoint with SSE streaming |
| `/graph/browse` | GET | Browse knowledge graph |
| `/ingest` | POST | Batch ingest triples |
| `/feedback` | POST | Submit answer feedback |
| `/healthz` | GET | Health check |
| `/readyz` | GET | Readiness check |

### Example Usage

```bash
# Ask a question
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: changeme" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the relationship between Tesla and SpaceX?",
    "visualize_graph": true,
    "max_tokens": 500
  }'

# Ingest new data
curl -X POST "http://localhost:8000/ingest" \
  -H "X-API-KEY: changeme" \
  -H "Content-Type: application/json" \
  -d '{
    "triples": [
      {
        "head": "Elon Musk",
        "relation": "founded",
        "tail": "Neuralink",
        "head_name": "Elon Musk",
        "relation_name": "founded",
        "tail_name": "Neuralink"
      }
    ]
  }'
```

## 🧪 Testing

```bash
# Run all tests
make test
# or
./bin/run_tests.sh

# Run specific test categories
pytest tests/test_api.py -v
pytest tests/test_retriever.py -v
pytest tests/test_database.py -v
```

## 📚 Documentation

- **[Architecture](docs/architecture.md)**: System design and components
- **[API Reference](docs/api_reference.md)**: Detailed API documentation
- **[Deployment](docs/deployment.md)**: Production deployment guide
- **[Troubleshooting](docs/troubleshooting.md)**: Common issues and solutions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the SubgraphRAG research framework
- Uses Neo4j for graph storage and FAISS for vector search
- Powered by FastAPI and modern Python ecosystem
