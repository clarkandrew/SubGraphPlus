# SubgraphRAG+

<div align="center">

![SubgraphRAG+ Logo](https://img.shields.io/badge/SubgraphRAG+-Knowledge%20Graph%20QA-blue?style=for-the-badge&logo=neo4j&logoColor=white)

**Production-Ready Knowledge Graph Question Answering with Hybrid Retrieval**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![Neo4j](https://img.shields.io/badge/neo4j-4.4+-red.svg)](https://neo4j.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🏗️ Architecture](#️-architecture) • [🤝 Contributing](#-contributing)

</div>

---

## 🌟 Overview

SubgraphRAG+ is an advanced knowledge graph-powered question answering system that combines structured graph traversal with semantic vector search. It provides contextual answers with real-time visualizations through a production-ready REST API.

### ✨ Key Features

- **🔀 Hybrid Retrieval**: Combines Neo4j graph traversal with FAISS vector search
- **🔄 Real-time Ingestion**: Dynamic knowledge graph updates with validation
- **📡 Streaming API**: Server-sent events with live citations and graph data  
- **📊 Interactive Visualization**: D3.js-compatible graph data with relevance scoring
- **🧠 Multi-LLM Support**: OpenAI, HuggingFace, Anthropic, MLX (Apple Silicon)
- **⚡ High Performance**: Optimized with caching, indexing, and MLP scoring
- **🏢 Production Ready**: Docker deployment, monitoring, health checks

### 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  Hybrid         │    │  Knowledge      │
│   REST API      │───▶│  Retriever      │───▶│  Graph (Neo4j)  │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │  Vector Index   │              │
         ▼              │  (FAISS)        │              ▼
┌─────────────────┐    └─────────────────┘    ┌─────────────────┐
│   LLM Backend   │                           │  MLP Scoring    │
│  (OpenAI/HF/MLX)│                           │  Model          │
└─────────────────┘                           └─────────────────┘
```

---

## 🚀 Quick Start

Choose your setup method based on your needs:

### 🐳 Production Setup (Docker - Recommended)

**For production deployments and users who want everything set up automatically:**

```bash
# Clone the repository
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# One-command setup and start
./bin/setup_docker.sh

# Verify it's working
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: default_key_for_dev_only" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?", "visualize_graph": true}'
```

### 🔧 Development Setup (Local)

**For developers who want to modify code or contribute:**

```bash
# Clone the repository
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Interactive development setup (recommended)
./bin/setup_dev.sh

# The script will guide you through:
# - Python environment setup
# - Dependency installation  
# - Neo4j configuration
# - Sample data loading
# - Configuration files

# Start the development server
source venv/bin/activate
python src/main.py --reload
```

### ⚡ Quick Setup Commands

**If you prefer using make commands:**

```bash
# Development environment
make setup-dev          # Uses bin/setup_dev.sh internally

# Docker environment  
make setup-all          # Uses bin/setup_docker.sh internally

# Individual components
make neo4j-start        # Start just Neo4j
make serve              # Start development server
make test               # Run test suite
```

> **💡 Setup Method Guide**: 
> - **`./bin/` scripts** → Interactive setup with user prompts and error handling
> - **`make` commands** → Automated workflows for CI/CD and quick operations
> - Use **bin scripts** for initial setup, **make** for daily development

---

## 📖 Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| **[📚 Documentation Hub](docs/README.md)** | Complete documentation index | All users |
| **[🛠️ Installation Guide](docs/installation.md)** | Detailed setup instructions | New users |
| **[🏗️ Architecture Guide](docs/architecture.md)** | System design and components | Developers, Architects |
| **[🔧 Development Guide](docs/development.md)** | Contributing and local dev | Contributors |
| **[🚀 Deployment Guide](docs/deployment.md)** | Production deployment | DevOps, SysAdmins |
| **[📡 API Reference](docs/api_reference.md)** | Complete API documentation | Integrators |
| **[🔧 Configuration](docs/configuration.md)** | Settings and environment vars | All users |
| **[🩺 Troubleshooting](docs/troubleshooting.md)** | Common issues and solutions | All users |

### 🍎 Apple Silicon Users

For optimized performance on M1/M2/M3 Macs:
- See **[MLX Integration Guide](docs/mlx.md)** for native Apple Silicon acceleration
- Use `./bin/setup_dev.sh` which auto-detects and configures MLX

---

## 🛠️ System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows with WSL2
- **Python**: 3.9+ (tested up to 3.13)
- **Memory**: 4GB RAM
- **Storage**: 10GB free space
- **Docker**: 20.10+ with Compose v2 (for Docker setup)

### Recommended for Production
- **CPU**: 4+ cores
- **Memory**: 8GB+ RAM  
- **Storage**: 50GB+ SSD
- **Network**: Stable internet connection for LLM APIs

---

## 🚦 API Usage

### Basic Query

```python
import requests

# Query with graph visualization
response = requests.post(
    "http://localhost:8000/query",
    headers={"X-API-KEY": "your-api-key"},
    json={
        "question": "What is machine learning?",
        "visualize_graph": True,
        "max_context_triples": 50
    }
)

# Stream the response
for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8'))
        print(f"Type: {data['type']}, Content: {data['content']}")
```

### Health Check

```bash
# Basic health check
curl http://localhost:8000/healthz

# Comprehensive readiness check
curl http://localhost:8000/readyz
```

### Graph Browsing

```bash
# Browse the knowledge graph
curl "http://localhost:8000/graph/browse?limit=100&search_term=AI" \
  -H "X-API-KEY: your-api-key"
```

---

## 🔧 Development Workflow

### Daily Development Commands

```bash
# Start development server
make serve                    # or: python src/main.py --reload

# Run tests
make test                     # Run full test suite
make test-smoke              # Quick smoke tests
make test-api                # API integration tests

# Code quality
make lint                     # Check code style
make format                   # Auto-format code

# Database operations
make neo4j-start             # Start Neo4j container
make migrate-schema          # Apply database migrations
make ingest-sample           # Load sample data
```

### Project Structure

```
SubgraphRAGPlus/
├── 📁 src/                   # Application source code
│   ├── 📄 main.py           # Application entry point
│   └── 📁 app/              # Core application modules
│       ├── 📄 api.py        # FastAPI REST endpoints
│       ├── 📄 retriever.py  # Hybrid retrieval engine
│       ├── 📄 database.py   # Neo4j & SQLite connections
│       └── 📁 ml/           # ML models (LLM, embeddings, MLP)
├── 📁 bin/                  # Setup and utility scripts
├── 📁 scripts/              # Python utilities and tools
├── 📁 tests/                # Comprehensive test suite
├── 📁 docs/                 # Documentation
├── 📁 config/               # Configuration files
├── 📁 deployment/           # Docker and infrastructure
├── 📄 Makefile             # Development commands
└── 📄 requirements.txt     # Python dependencies
```

---

## 🏢 Production Deployment

### Docker Production

```bash
# Production deployment
cd deployment/
docker-compose -f docker-compose.prod.yml up -d

# Scale API instances
docker-compose -f docker-compose.prod.yml up -d --scale api=3

# Monitor services
docker-compose -f docker-compose.prod.yml logs -f
```

### Environment Configuration

Essential production environment variables:

```bash
# Core database settings
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-production-password

# API security
API_KEY_SECRET=your-secure-api-key

# LLM backend (choose one)
OPENAI_API_KEY=your-openai-key
# or configure HuggingFace/MLX in config.json

# Production settings
LOG_LEVEL=INFO
API_RATE_LIMIT=100
WORKERS=4
```

### Monitoring Endpoints

- **Health Check**: `GET /healthz` - Basic liveness probe
- **Readiness Check**: `GET /readyz` - Dependency health with detailed status
- **Metrics**: `GET /metrics` - Prometheus-compatible metrics
- **API Docs**: `GET /docs` - Interactive OpenAPI documentation
- **Neo4j Browser**: http://localhost:7474 - Database management interface

---

## 🧪 Testing

### Running Tests

```bash
# Full test suite
make test

# Specific test categories
python -m pytest tests/test_api.py -v          # API tests
python -m pytest tests/test_retriever.py -v    # Retrieval tests
python -m pytest tests/test_mlp_model.py -v    # MLP model tests

# With coverage report
python -m pytest --cov=src tests/ --cov-report=html
```

### Test Structure

- **Unit Tests**: Individual component testing
- **Integration Tests**: Multi-component workflows
- **API Tests**: REST endpoint validation
- **Smoke Tests**: Basic system functionality
- **Performance Tests**: Benchmarking and load testing

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Quick Contribution Setup

```bash
# 1. Fork and clone
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# 2. Setup development environment
./bin/setup_dev.sh --run-tests

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make changes and test
make test
make lint

# 5. Submit a pull request
```

### Development Guidelines

- **Code Style**: Follow PEP 8 with Black formatting
- **Testing**: Add tests for new features
- **Documentation**: Update docs for user-facing changes
- **Commits**: Use conventional commit messages

See **[Contributing Guide](docs/contributing.md)** for detailed information.

---

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Support & Community

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/your-username/SubgraphRAGPlus/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/your-username/SubgraphRAGPlus/discussions)
- **📖 Documentation Issues**: [Create an Issue](https://github.com/your-username/SubgraphRAGPlus/issues)
- **💬 General Questions**: [GitHub Discussions](https://github.com/your-username/SubgraphRAGPlus/discussions)

---

## 🙏 Acknowledgments

- Original SubgraphRAG research by Microsoft Research
- Neo4j and FAISS communities for graph and vector database technologies
- FastAPI, PyTorch, and Python ecosystem contributors
- Contributors and users of this project

---

<div align="center">

**[⭐ Star this repository](https://github.com/your-username/SubgraphRAGPlus) if you find it useful!**

**Made with ❤️ for the Knowledge Graph community**

</div>
