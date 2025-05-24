# 🚀 SubgraphRAG+

> Knowledge Graph Question Answering with Hybrid Retrieval and Visualization

SubgraphRAG+ is a knowledge graph-powered QA system that combines structured graph traversal with dense vector retrieval to provide accurate, contextual answers with explanatory visualizations. Built on the original SubgraphRAG research, this enhanced version adds dynamic knowledge graph ingestion, hybrid retrieval, and enterprise-grade API features.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

## 🌟 Key Features

- **🔀 Hybrid Retrieval**: Combines graph traversal and semantic search for optimal recall and precision
- **🔄 Dynamic KG Ingestion**: Real-time triple ingestion with intelligent deduplication and validation
- **📡 SSE Streaming**: Token-by-token LLM response delivery with live citations and graph data
- **📊 Interactive Visualization**: D3.js compatible graph data with relevance scores and subgraph highlighting
- **🏢 Enterprise-Grade API**: OpenAPI 3.0 compliant with comprehensive documentation, rate limiting, and monitoring
- **🧠 Multi-LLM Support**: Compatible with OpenAI, HuggingFace, Anthropic, and local models
- **⚡ Performance Optimized**: FAISS vector indexing, Neo4j graph optimization, and intelligent caching
- **🔒 Production Ready**: Docker deployment, health checks, metrics, and comprehensive logging

## 📂 Project Structure

```
SubgraphRAG+/
├── src/                       # Source code
│   ├── main.py               # Application entry point
│   └── app/                  # Core application
│       ├── api.py           # FastAPI routes and endpoints
│       ├── models.py        # Pydantic data models
│       ├── config.py        # Configuration management
│       ├── database.py      # Database connections (Neo4j, SQLite)
│       ├── retriever.py     # Hybrid retrieval engine
│       └── ml/              # Machine learning components
├── bin/                      # Executable scripts
│   ├── start.sh            # Interactive setup and start
│   ├── setup_dev.sh        # Development environment setup
│   ├── run.sh              # Server startup script
│   ├── run_tests.sh        # Test execution
│   └── install_neo4j.sh    # Neo4j local installation
├── scripts/                  # Python utilities
│   ├── stage_ingest.py     # Data staging for ingestion
│   ├── ingest_worker.py    # Background ingestion processing
│   ├── merge_faiss.py      # FAISS index management
│   └── migrate_schema.py   # Database schema migrations
├── deployment/               # Infrastructure and deployment
│   ├── docker-compose.yml  # Docker service definition
│   ├── Dockerfile          # Container build instructions
│   └── nginx.conf          # Reverse proxy configuration
├── tests/                   # Comprehensive test suite
│   ├── unit/               # Unit tests for components
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end workflow tests
├── config/                  # Configuration files
│   └── config.json         # Application configuration
├── data/                    # Data storage and caches
│   ├── faiss/              # Vector index storage
│   ├── neo4j/              # Graph database data
│   └── staging.db          # SQLite ingestion queue
├── models/                  # Machine learning models
├── docs/                    # Detailed documentation
│   ├── architecture.md     # System architecture
│   ├── api_reference.md    # API documentation
│   ├── deployment.md       # Production deployment
│   └── development.md      # Development guide
├── Makefile                 # Build and development commands
└── requirements.txt         # Python dependencies
```

## 📋 Prerequisites

### Docker Setup (Recommended)
- **Docker Engine**: 20.10+ with Docker Compose v2
- **System Resources**: 4GB+ RAM, 10GB+ free disk space
- **Network**: Internet access for model downloads

### Local Development Setup
- **Python**: 3.11+ with pip and venv
- **Neo4j**: 4.4+ with APOC plugin installed
- **System Tools**: SQLite3, curl, git
- **Optional**: OpenAI API key for advanced LLM features

## 🚀 Quick Start

### Option 1: One-Command Docker Setup

```bash
# Clone repository
git clone https://github.com/clarkandrew/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Complete automated setup
make setup-all

# Test the system
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: changeme" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who founded Tesla?", "visualize_graph": true}'
```

### Option 2: Interactive Setup

```bash
# Clone and run interactive setup
git clone https://github.com/clarkandrew/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Interactive setup with guidance
./bin/start.sh
```

### Option 3: Manual Local Development

```bash
# Setup Python environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements-dev.txt

# Setup local Neo4j (automated)
./bin/install_neo4j.sh

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Initialize and start
make migrate-schema
make ingest-sample
python src/main.py --reload
```

## 🌐 Access Points

After successful setup, access these endpoints:

| Service | URL | Description |
|---------|-----|-------------|
| **API Documentation** | http://localhost:8000/docs | Interactive Swagger UI |
| **API Endpoints** | http://localhost:8000 | Main API server |
| **Neo4j Browser** | http://localhost:7474 | Graph database UI (neo4j/password) |
| **Health Check** | http://localhost:8000/healthz | System health status |
| **Metrics** | http://localhost:8000/metrics | Prometheus metrics |

## 🔧 Development Commands

### Make Commands (Recommended)

```bash
# Essential commands
make help                    # Show all available commands
make setup-all              # Complete setup with Docker
make serve                   # Start development server
make test                    # Run comprehensive tests
make test-coverage           # Generate coverage reports

# Code quality
make lint                    # Check code style and quality
make format                  # Auto-format with black/isort
make typecheck               # Type checking with mypy

# Docker operations
make docker-start            # Start all Docker services
make docker-stop             # Stop Docker services
make docker-dev              # Development mode with hot reload
make logs                    # View service logs

# Database operations
make neo4j-start             # Start Neo4j container
make neo4j-restart           # Restart Neo4j with clean state
make migrate-schema          # Apply database migrations
make ingest-sample           # Load sample knowledge data

# Maintenance
make clean                   # Remove temporary files
make reset                   # Reset all data (use with caution)
```

### Shell Scripts (Cross-platform)

```bash
# Core operations
./bin/start.sh               # Interactive setup and start
./bin/run.sh                 # Start API server
./bin/setup_dev.sh           # Development environment setup
./bin/run_tests.sh           # Execute test suite

# Database and infrastructure
./bin/install_neo4j.sh       # Install Neo4j locally
./bin/setup_docker.sh        # Docker environment setup

# Tools and utilities
./bin/demo.sh                # Interactive system demo
./bin/backup.sh              # Backup system data and models
./bin/run_benchmark.sh       # Performance benchmarking
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# API Security
API_KEY_SECRET=changeme_in_production

# LLM Backend Configuration
MODEL_BACKEND=openai
OPENAI_API_KEY=your_api_key_here

# Performance Settings
WORKERS=4
MAX_CONNECTIONS=100
CACHE_SIZE=1000

# Logging and Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

### Application Configuration

Customize `config/config.json` for your deployment:

```json
{
  "MODEL_BACKEND": "openai",
  "FAISS_INDEX_PATH": "data/faiss_index.bin",
  "TOKEN_BUDGET": 4000,
  "MLP_MODEL_PATH": "models/mlp_pretrained.pt",
  "CACHE_DIR": "cache/",
  "MAX_DDE_HOPS": 2,
  "LOG_LEVEL": "INFO",
  "API_RATE_LIMIT": 60,
  "ENABLE_CORS": true,
  "CORS_ORIGINS": ["http://localhost:3000"],
  "MAX_QUERY_LENGTH": 1000,
  "EMBEDDING_CACHE_SIZE": 10000
}
```

## 📊 API Overview

### Core Endpoints

| Method | Endpoint | Description | Features |
|--------|----------|-------------|----------|
| `POST` | `/query` | Question answering | SSE streaming, graph visualization |
| `GET` | `/graph/browse` | Browse knowledge graph | Pagination, filtering |
| `POST` | `/ingest` | Batch data ingestion | Validation, deduplication |
| `POST` | `/feedback` | Submit user feedback | Model improvement |
| `GET` | `/healthz` | Health monitoring | Dependency checks |

### Example API Usage

#### Ask a Question with Visualization

```bash
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: changeme" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the relationship between Tesla and SpaceX?",
    "visualize_graph": true,
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

#### Ingest Knowledge Triples

```bash
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

#### Browse Knowledge Graph

```bash
curl "http://localhost:8000/graph/browse?entity=Tesla&limit=10" \
  -H "X-API-KEY: changeme"
```

## 🧪 Testing and Quality Assurance

### Running Tests

```bash
# Complete test suite
make test

# With coverage reporting
make test-coverage

# Specific test categories
pytest tests/unit/ -v           # Unit tests
pytest tests/integration/ -v    # Integration tests
pytest tests/e2e/ -v           # End-to-end tests

# Performance testing
make benchmark
```

### Code Quality Checks

```bash
# All quality checks
make lint format typecheck

# Individual checks
flake8 src/ tests/ scripts/     # Style checking
black src/ tests/ scripts/      # Code formatting
mypy src/ tests/ scripts/       # Type checking
```

## 📚 Documentation

Comprehensive documentation is available:

- **[🏗️ Architecture Guide](docs/architecture.md)**: System design, components, and data flow
- **[📖 API Reference](docs/api_reference.md)**: Detailed endpoint documentation with examples
- **[🚀 Deployment Guide](docs/deployment.md)**: Production deployment with Docker and Kubernetes
- **[💻 Development Guide](docs/development.md)**: Contributing, testing, and extending the system

## 🔧 Common Tasks

### Adding Custom Data

```python
# Using the Python client
import requests

response = requests.post(
    "http://localhost:8000/ingest",
    headers={"X-API-KEY": "your-key"},
    json={
        "triples": [
            {
                "head": "Your Entity",
                "relation": "has_property",
                "tail": "Property Value"
            }
        ]
    }
)
```

### Monitoring System Health

```bash
# Quick health check
curl http://localhost:8000/healthz

# Detailed readiness check
curl http://localhost:8000/readyz

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Backup and Recovery

```bash
# Create backup
./bin/backup.sh create

# List available backups
./bin/backup.sh list

# Restore from backup
./bin/backup.sh restore backup_20240101
```

## 🚀 Production Deployment

For production deployment, see the [Deployment Guide](docs/deployment.md) which covers:

- Docker Compose production setup
- Kubernetes deployment manifests
- Security best practices
- Monitoring and logging setup
- Performance optimization
- Backup and recovery procedures

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](docs/development.md) for:

1. **Development setup** and environment configuration
2. **Code standards** and formatting requirements
3. **Testing strategies** and coverage requirements
4. **Pull request process** and review guidelines
5. **Architecture decisions** and extension points

### Quick Contributing Steps

```bash
# Fork and clone
git clone https://github.com/yourusername/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
make test lint format

# Commit and push
git commit -m "feat: add amazing feature"
git push origin feature/amazing-feature

# Create pull request
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Research Foundation**: Built on the SubgraphRAG research framework
- **Core Technologies**: Neo4j for graph storage, FAISS for vector search, FastAPI for web framework
- **ML Integration**: OpenAI, HuggingFace, and local model support
- **Community**: Thanks to all contributors and users who help improve this project

## 📞 Support and Community

- **Issues**: [GitHub Issues](https://github.com/clarkandrew/SubgraphRAGPlus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/clarkandrew/SubgraphRAGPlus/discussions)
- **Documentation**: Comprehensive docs in the `/docs` directory
- **Examples**: Sample queries and integrations in `/examples`

---

**Ready to explore knowledge graphs?** Start with `make setup-all` and begin querying your data in minutes! 🚀
