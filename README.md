# 🚀 SubgraphRAG+

> Knowledge Graph Question Answering with Hybrid Retrieval and Visualization

SubgraphRAG+ is a knowledge graph-powered QA system that combines structured graph traversal with dense vector retrieval to provide accurate, contextual answers with explanatory visualizations. Built on the original SubgraphRAG research, this enhanced version adds dynamic knowledge graph ingestion, hybrid retrieval, and enterprise-grade API features.

[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/index.md)
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
├── bin/               # Shell scripts for running and managing the system
├── config/            # Configuration files
├── data/              # Data storage
├── docs/              # Documentation and guides
├── evaluation/        # Benchmarking and evaluation tools
├── examples/          # Example scripts and usage demos
├── migrations/        # Neo4j schema migrations
├── models/            # ML model storage
├── prompts/           # Prompt templates
├── scripts/           # Utility Python scripts
└── tests/             # Test suite
```

## 📋 Prerequisites

- Docker and Docker Compose (recommended deployment method)
- OR for local development:
  - Python 3.11+
  - Neo4j (4.4+) with APOC plugin (Docker or [local installation](docs/dev_environment.md#step-5-set-up-neo4j-for-development))
  - SQLite3
  - (Optional) Local model support: MLX (Apple Silicon) or Hugging Face models
  - (Optional) OpenAI API key (for OpenAI backend)

## 🚀 Quick Start

### One-Step Setup Using Make (Recommended)

```bash
# Clone repo (if needed)
git clone https://github.com/yourusername/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Run the complete setup target (handles everything automatically)
make setup-all
```

This command will:
- Set up your virtual environment
- Install all dependencies
- Start Neo4j using Docker
- Download necessary models
- Initialize the database
- Load sample data
- Run the tests to verify everything works

The Makefile is your central command hub for all operations in this project!
```bash
# Show all available commands
make help
```

For complete setup instructions, see our [Getting Started Guide](./docs/getting_started.md) or [Developer Environment Guide](./docs/dev_environment.md).

2. **Test with a Query**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: changeme" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who founded Tesla?", "visualize_graph": true}'
```

3. **Access interfaces**
   - API Documentation: http://localhost:8000/docs
   - Neo4j Browser: http://localhost:7474 (user: neo4j, password: password)

### Alternative Setup Methods

#### Docker Manual Setup

```bash
# Start everything with Docker Compose
make docker-start

# Initialize with sample data
make ingest-sample
```

#### Local Development Environment

```bash
# Install development dependencies
make setup-dev

# Start Neo4j (with Docker)
make neo4j-start
# OR install Neo4j locally
./bin/install_neo4j.sh

# Download MLP model
make get-pretrained-mlp

# Initialize database
make migrate-schema
```

For detailed developer setup instructions, see our [Getting Started Guide](./docs/getting_started.md) and [Development Environment Guide](./docs/dev_environment.md).

2. **Ingest Sample Data**

```bash
# Stage and ingest sample triples
make ingest-sample

# Merge vectors into FAISS index
python scripts/merge_faiss.py
```

3. **Start the API Server**

```bash
make serve
```

## 🔧 Configuration

For complete configuration details, see our [Configuration Guide](./docs/deployment.md#configuration).

### Docker Configuration

When using Docker, configure the system through these methods:

#### Docker Environment Variables

Edit these in `docker-compose.yml`:

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | `bolt://subgraphrag_neo4j:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |
| `API_KEY_SECRET` | API key for authentication | `changeme` |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI backend) | None |
| `MODEL_BACKEND` | Model backend to use | `openai` |

#### Docker Volume Management

The system uses Docker volumes to persist data:
- `neo4j_data`: Neo4j database files
- `app_data`: Application data including SQLite DB
- `app_models`: ML models
- `app_cache`: Cache data
- `app_logs`: Application logs

### Local Configuration

#### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | `neo4j://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |
| `API_KEY_SECRET` | API key for authentication | (auto-generated) |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI backend) | None |

#### Config Options (config/config.json)

| Option | Description | Default |
|--------|-------------|---------|
| `MODEL_BACKEND` | Model backend to use (`mlx`, `openai`, `hf`) | `openai` |
| `FAISS_INDEX_PATH` | Path to FAISS index file | `data/faiss_index.bin` |
| `TOKEN_BUDGET` | Maximum tokens for context window | `4000` |
| `MLP_MODEL_PATH` | Path to pre-trained SubgraphRAG MLP model | `models/mlp_pretrained.pt` |

## 📈 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Main QA endpoint with SSE streaming |
| `/graph/browse` | GET | Browse knowledge graph with pagination |
| `/ingest` | POST | Batch ingest triples |
| `/feedback` | POST | Submit feedback on answers |
| `/healthz` | GET | Health check |
| `/readyz` | GET | Readiness check |
| `/metrics` | GET | Prometheus metrics |

For detailed information about the API endpoints, request/response formats, and SSE events, refer to `docs/api_reference.md`.

For detailed API documentation, start the server and visit `http://localhost:8000/docs`.

### Common Tasks Using Makefile

The Makefile is the primary way to interact with this project. Here are the most common commands:

```bash
# Complete setup in one step
make setup-all                   # Complete setup with everything

# Core development commands
make setup-dev                   # Install development dependencies
make setup-prod                  # Install production dependencies
make serve                       # Start the development server
make serve-prod                  # Start the production server

# Testing and quality
make test                        # Run all tests
make lint                        # Run code quality checks

# Database management
make neo4j-start                 # Start Neo4j database
make neo4j-stop                  # Stop Neo4j database
make migrate-schema              # Initialize database schema

# Data operations
make ingest-sample               # Load sample data
make rebuild-faiss-index         # Rebuild the FAISS index

# Docker operations
make docker-start                # Start all Docker containers
make docker-stop                 # Stop all Docker containers

# Help
make help                        # Show all available commands
```

Simply run `make help` to see all available commands and what they do.

### Using Make Commands (Local Development)

```bash
# Run tests
make test

# Run linting
make lint

# Rebuild FAISS index
make rebuild-faiss-index

# Run benchmarks
make benchmark

# Reset all data (use with caution)
make reset

# Complete setup in one step
make setup-all
```

## 🧠 Pre-trained MLP Integration

SubgraphRAG+ uses a pre-trained MLP model for triple scoring:

### Using Docker

The model will be downloaded automatically when the container starts if it's not present.

### Local Development

1. **Option 1**: Automatic download (requires internet)
   ```bash
   make get-pretrained-mlp
   # or
   python scripts/download_models.py
   ```

2. **Option 2**: Manual acquisition
   - Visit the [SubgraphRAG Colab notebook](https://colab.research.google.com/drive/...)
   - Run the notebook with `KGQA_DATASET="webqsp"`
   - Download the resulting `cpt.pth` file
   - Place it at `models/mlp_pretrained.pt`

## 🐳 Docker Deployment Details

### Container Architecture

The system consists of two main containers:
1. **subgraphrag_neo4j**: Neo4j graph database with APOC plugin
2. **subgraphrag_api**: FastAPI application with all dependencies

### Data Persistence

All data is stored in Docker volumes for persistence between restarts:
- `neo4j_data`: Neo4j database files
- `app_data`: Application data including SQLite
- `app_models`: ML models storage
- `app_cache`: Cache storage
- `app_logs`: Application logs

### Scaling and Production Use

For production:
1. Update `API_KEY_SECRET` in docker-compose.yml
2. Configure proper HTTPS termination (e.g., with Nginx or Traefik)
3. Set `OPENAI_API_KEY` if using the OpenAI model backend
4. Consider setting resource limits for containers

## 🛠️ Operational Features

For more information about operational aspects, see our [Operations Guide](./docs/deployment.md#operations).

### Backup and Restore

The system includes a comprehensive backup and restore solution:

- Full backup of Neo4j database, SQLite staging DB, FAISS indices, and configuration
- Metadata tracking for all backups with timestamps and component status
- Selective restoration of specific backups
- Docker-aware operation for containerized deployments
- Graceful handling of missing components (works even if Docker or Neo4j is unavailable)

### Advanced Evaluation

- Precision, recall, and F1 score metrics with ground truth support
- Robustness evaluation with adversarial test questions
- Detailed HTML reports with visualizations
- Entity linking accuracy and hallucination detection

## 📝 License

This project is licensed under the Apache License 2.0 - see the `LICENSE` file for details.

## 📚 Documentation

For complete documentation, visit the [docs](./docs) directory:

- [Documentation Index](./docs/index.md) - Full documentation index
- [Getting Started Guide](./docs/getting_started.md) - Install and setup instructions
- [Developer Environment Guide](./docs/dev_environment.md) - Setting up for development
- [API Reference](./docs/api_reference.md) - API endpoint details
- [Architecture Overview](./docs/architecture.md) - System design and components
- [Deployment Guide](./docs/deployment.md) - Production deployment
- [Testing Guide](./docs/testing.md) - Running and writing tests
- [Evaluation Guide](./docs/evaluation.md) - Benchmarking and evaluation

## 🌟 Acknowledgements

Based on the original [SubgraphRAG paper](https://arxiv.org/abs/2401.09863) (ICLR 2025) with significant enhancements.