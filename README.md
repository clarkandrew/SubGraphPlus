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
├── app/               # Core application code
├── bin/               # Shell scripts for setup and operation
├── config/            # Configuration files
├── data/              # Data storage
├── docs/              # Additional documentation
├── evaluation/        # Benchmarking and evaluation tools
├── examples/          # Example scripts and usage demos
├── migrations/        # Neo4j schema migrations
├── models/            # ML model storage
├── prompts/           # Prompt templates
├── scripts/           # Utility Python scripts
└── tests/             # Test suite
```

## 📋 Prerequisites

- **Docker Setup** (Recommended for most users):
  - Docker Engine 20.10+
  - Docker Compose v2
  - 4GB+ RAM allocated to Docker
  - 10GB+ free disk space

- **Local Development** (Alternative setup):
  - Python 3.11+
  - Neo4j 4.4+ with APOC plugin (can be installed via `bin/install_neo4j.sh`)
  - SQLite3
  - Virtual environment (venv, conda, etc.)
  - (Optional) OpenAI API key for OpenAI backend
  - (Optional) GPU for faster local model inference

## 🚀 Quick Start: Choose ONE Setup Method

### Method 1: Setup with Make (Recommended for Most Users)

```bash
# Clone repo
git clone https://github.com/clarkandrew/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Complete setup in one command (requires Docker)
make setup-all

# Test the system
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: changeme" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who founded Tesla?", "visualize_graph": true}'

# Access interfaces
# - API Documentation: http://localhost:8000/docs
# - Neo4j Browser: http://localhost:7474 (user: neo4j, password: password)
```

### Method 2: Setup with Shell Scripts (Without Docker)

```bash
# Clone repo
git clone https://github.com/clarkandrew/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Make scripts executable
chmod +x bin/*.sh

# 1. Install Neo4j locally (skip if you already have Neo4j 4.4+ with APOC)
./bin/install_neo4j.sh

# 2. Set up development environment (with local Neo4j)
./bin/setup_dev.sh --use-local-neo4j

# 3. Start the server
./bin/run.sh

# Access interfaces
# - API Documentation: http://localhost:8000/docs
# - Neo4j Browser: http://localhost:7474 (user: neo4j, password: password)
```

## 🛠️ Development Commands
## 🔧 Configuration & Commands

### Available Tools

#### 1. Makefile Commands (Docker-based Development)

```bash
# Complete setup in one step
make setup-all                   # Complete setup with everything

# Core development commands
make setup-dev                   # Install development dependencies
make serve                       # Start the development server
make serve-prod                  # Start production server

# Testing and quality
make test                        # Run all tests
make lint                        # Run code quality checks

# Database management
make neo4j-start                 # Start Neo4j database (Docker)
make neo4j-stop                  # Stop Neo4j database
make migrate-schema              # Initialize database schema

# Data operations
make ingest-sample               # Load sample data
make rebuild-faiss-index         # Rebuild the FAISS index

# Docker operations
make docker-start                # Start all Docker containers
make docker-stop                 # Stop all Docker containers

# Show all available commands
make help
```

#### 2. Shell Scripts (Non-Docker Development)

All scripts are in the `bin/` directory:

```bash
# Basic setup and operation
./bin/setup_dev.sh [--use-local-neo4j]   # Set up development environment
./bin/run.sh [--port PORT]               # Start the API server
./bin/run_tests.sh                       # Run test suite
./bin/install_neo4j.sh                   # Install Neo4j locally
./bin/backup.sh                          # Backup the system

# Advanced operations
./bin/run_benchmark.sh                   # Run performance benchmarks
./bin/demo.sh                            # Run interactive demo
```

### Configuration Options

Create a `.env` file in the project root with these settings:

```
# Neo4j Connection
NEO4J_URI=neo4j://localhost:7687    # For local Neo4j
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
USE_LOCAL_NEO4J=true                # Set to use local Neo4j

# API Security
API_KEY_SECRET=changeme_in_production

# Model Backend (options: openai, hf, mlx)
MODEL_BACKEND=openai
# OPENAI_API_KEY=your_openai_key    # Required for OpenAI backend
```

Edit `config/config.json` for additional settings:

```json
{
  "MODEL_BACKEND": "openai",
  "FAISS_INDEX_PATH": "data/faiss/index",
  "TOKEN_BUDGET": 8000,
  "MLP_MODEL_PATH": "models/mlp",
  "CACHE_DIR": "cache"
}
```

## 📈 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Main QA endpoint with SSE streaming |
| `/graph/browse` | GET | Browse knowledge graph with pagination |
| `/ingest` | POST | Batch ingest triples |
| `/feedback` | POST | Submit feedback on answers |
| `/healthz` | GET | Health check |
| `/readyz` | GET | Readiness check |
| `/metrics` | GET | Prometheus metrics |

See API documentation at http://localhost:8000/docs when running the server.

## 📋 Common Tasks

### 1. Adding Custom Data

```bash
# Prepare your triples in CSV format (head,relation,tail)
# Then ingest them:

# Using Make
make ingest-data FILE=path/to/your/triples.csv

# Using Python script
python scripts/stage_ingest.py --file path/to/your/triples.csv
python scripts/ingest_worker.py --process-all
python scripts/merge_faiss.py
```

### 2. Backing Up and Restoring

```bash
# Create backup
./bin/backup.sh create

# Restore backup
./bin/backup.sh restore backup_20240501_120000
```

### 3. Running Tests

```bash
# All tests
make test

# Specific tests
./bin/run_tests.sh -t unit        # Unit tests only
./bin/run_tests.sh -t integration # Integration tests only
./bin/run_tests.sh -c             # With coverage report
```

### 4. Production Deployment

For production deployment, use the Docker setup with these changes:

1. Update `API_KEY_SECRET` to a strong random value
2. Configure HTTPS (using Nginx/Traefik as reverse proxy)
3. Set resource limits in docker-compose.yml
4. Enable regular backups with `./bin/backup.sh create`

See `docs/deployment.md` for detailed production setup instructions.

## 📝 License

This project is licensed under the Apache License 2.0 - see the `LICENSE` file for details.

## 🌟 Acknowledgements

Based on the original [SubgraphRAG paper](https://arxiv.org/abs/2401.09863) (ICLR 2025) with significant enhancements.
