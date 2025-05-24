# SubgraphRAG+

<div align="center">

![SubgraphRAG+ Logo](https://img.shields.io/badge/SubgraphRAG+-Enhanced%20Knowledge%20Graph%20QA-blue?style=for-the-badge)

**Advanced Knowledge Graph Question Answering with Hybrid Retrieval and Real-time Visualization**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![Neo4j](https://img.shields.io/badge/neo4j-4.4+-red.svg)](https://neo4j.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🔧 Development](#-development) • [🏢 Deployment](#-deployment) • [🤝 Contributing](#-contributing)

</div>

---

## 🌟 Overview

SubgraphRAG+ is a production-ready knowledge graph-powered question answering system that combines structured graph traversal with dense vector retrieval. Built on cutting-edge research, it provides accurate, contextual answers with interactive visualizations and comprehensive API features.

### ✨ Key Features

- **🔀 Hybrid Retrieval**: Combines graph traversal and semantic search for optimal accuracy
- **🔄 Real-time Ingestion**: Dynamic knowledge graph updates with intelligent validation
- **📡 Streaming Responses**: Token-by-token delivery with live citations and graph data
- **📊 Interactive Visualizations**: D3.js-compatible graph data with relevance highlighting
- **🏢 Enterprise API**: OpenAPI 3.0 compliant with rate limiting and monitoring
- **🧠 Multi-LLM Support**: OpenAI, HuggingFace, Anthropic, MLX (Apple Silicon), and local model compatibility
- **⚡ High Performance**: FAISS indexing, Neo4j optimization, and intelligent caching
- **🔒 Production Ready**: Docker deployment, health checks, metrics, and comprehensive logging

## 🚀 Quick Start

Choose your preferred setup method:

### 🐳 Docker Setup (Recommended for Production)

**Prerequisites:**
- Docker Engine 20.10+ with Docker Compose v2
- 4GB+ RAM, 10GB+ free disk space

```bash
# Clone the repository
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# One-command setup and start
make setup-all

# Verify installation
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?", "visualize_graph": true}'
```

### 🐍 Local Development Setup (Recommended for Development)

**Prerequisites:**
- Python 3.9+ (tested up to Python 3.13)
- Git

```bash
# Clone the repository
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Interactive setup (recommended - handles everything automatically)
make setup-dev

# Alternative: Use the setup script directly with options
./bin/setup_dev.sh --help  # See all options

# Quick non-interactive setup (skips tests and sample data)
make setup-dev-quick
```

The setup script will:
- ✅ Detect and validate your Python version
- ✅ Create and configure a virtual environment
- ✅ Install all dependencies
- ✅ Set up Neo4j (Docker or local)
- ✅ Create configuration files
- ✅ Initialize database schema
- ✅ Load sample data (optional)
- ✅ Run tests (optional)

> **🍎 Apple Silicon Users**: For optimized performance on M1/M2/M3 Macs, see [MLX Installation Guide](docs/installation.md#-mlx-installation-apple-silicon) to enable native Apple Silicon acceleration.

### 🐍 Quick Validation

After setup, verify your installation:

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs  # macOS
# or visit http://localhost:8000/docs in your browser
```

## 📖 Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| [🏗️ Architecture Guide](docs/architecture.md) | System design and components | Developers, Architects |
| [🔧 Development Guide](docs/development.md) | Local development setup | Contributors, Developers |
| [🚀 Deployment Guide](docs/deployment.md) | Production deployment | DevOps, System Admins |
| [📡 API Reference](docs/api_reference.md) | Complete API documentation | API Consumers, Integrators |

## 🛠️ System Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 10GB free space
- **OS**: Linux, macOS, or Windows with WSL2

### Recommended for Production
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 50GB+ SSD
- **Network**: Stable internet connection

## 🔧 Development

### Development Commands

```bash
# Setup commands
make setup-dev          # Full interactive development setup
make setup-dev-quick    # Quick non-interactive setup
make setup-all          # Docker-based setup

# Development server
make serve              # Start development server
make test               # Run comprehensive tests
make lint               # Check code quality
make format             # Auto-format code

# Database operations
make neo4j-start        # Start Neo4j container
make migrate-schema     # Apply database migrations
make ingest-sample      # Load sample data

# Maintenance
make clean              # Remove temporary files
make docker-logs        # View service logs
```

### Project Structure

```
SubgraphRAG+/
├── 📁 src/                     # Application source code
│   ├── 📄 main.py             # Application entry point
│   └── 📁 app/                # Core application modules
├── 📁 bin/                     # Executable scripts
├── 📁 scripts/                 # Python utilities
├── 📁 deployment/              # Docker and infrastructure
├── 📁 tests/                   # Comprehensive test suite
├── 📁 docs/                    # Documentation
├── 📁 config/                  # Configuration files
├── 📄 Makefile                 # Development commands
└── 📄 requirements.txt         # Python dependencies
```

## 🏢 Deployment

### Docker Production Deployment

```bash
# Production setup
cd deployment/
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale api=3
```

### Environment Configuration

Essential environment variables:

```bash
# Core settings
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-password
API_KEY_SECRET=your-secret-key

# Optional: OpenAI integration
OPENAI_API_KEY=your-openai-key

# Production settings
LOG_LEVEL=INFO
API_RATE_LIMIT=100
```

### Health Monitoring

- **Health endpoint**: `GET /health`
- **Metrics endpoint**: `GET /metrics`
- **Neo4j Browser**: http://localhost:7474

## 🌐 API Usage

### Basic Query

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    headers={"X-API-KEY": "your-api-key"},
    json={
        "question": "What is machine learning?",
        "visualize_graph": True,
        "max_results": 10
    }
)

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Sources: {len(data['sources'])} found")
```

### Streaming Response

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query/stream",
    headers={"X-API-KEY": "your-api-key"},
    json={"question": "Explain artificial intelligence"},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](docs/development.md) for details.

### Quick Contribution Setup

```bash
# Fork and clone your fork
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Setup development environment
./bin/setup_dev.sh

# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes and test
make test
make lint

# Submit a pull request
```

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/your-username/SubgraphRAGPlus/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/your-username/SubgraphRAGPlus/discussions)
- **📖 Documentation**: [docs/](docs/)
- **💬 Community**: [GitHub Discussions](https://github.com/your-username/SubgraphRAGPlus/discussions)

## 🙏 Acknowledgments

- Original SubgraphRAG research team
- Neo4j and FAISS communities
- FastAPI and Python ecosystem contributors

---

<div align="center">

**[⭐ Star this repository](https://github.com/your-username/SubgraphRAGPlus) if you find it useful!**

Made with ❤️ by the SubgraphRAG+ team

</div>
