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

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd SubGraphPlus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

SubgraphRAG+ uses a **clean two-tier configuration system** that separates secrets from application settings:

- **`.env`**: Secrets and environment-specific values (never commit to git)
- **`config/config.json`**: Application settings, models, and parameters (version controlled)

This separation follows security best practices and makes deployment across environments simple.

### Quick Setup

```bash
# Copy the example and customize with your secrets
cp .env.example .env
nano .env  # Add your actual credentials and API keys
```

### Application Configuration (`config/config.json`)

The main configuration file controls all application behavior:

```json
{
  "models": {
    "backend": "mlx",
    "llm": {
      "mlx": {
        "model": "mlx-community/Qwen3-14B-8bit",
        "max_tokens": 512,
        "temperature": 0.1
      }
    },
    "embeddings": {
      "model": "Alibaba-NLP/gte-large-en-v1.5",
      "backend": "transformers"
    }
  },
  "retrieval": {
    "token_budget": 4000,
    "max_dde_hops": 2,
    "similarity_threshold": 0.7
  }
}
```

### Environment Variables (`.env`)

Contains **only secrets and environment-specific values**:

```bash
# === Database Credentials ===
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password

# === API Security ===
API_KEY_SECRET=your_secret_key

# === API Keys ===
OPENAI_API_KEY=sk-your-key  # Required for OpenAI backend
HF_TOKEN=hf_your-token      # Optional for private HF models

# === Environment ===
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=false
```

**Key Principles:**
- **Secrets in `.env`** - Never commit credentials to version control
- **Settings in `config.json`** - Application configuration is version controlled  
- **No duplication** - Each setting has one clear location
- **Environment overrides** - `.env` can override `config.json` defaults when needed

### 3. Database Setup

Start Neo4j database:

```bash
# Using Docker
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/your_password \
    neo4j:latest

# Or use Neo4j Desktop/AuraDB and update NEO4J_URI in .env
```

### 4. Run the Application

```bash
# Start the FastAPI server
python -m uvicorn src.app.api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 📋 API Documentation

### Health & Monitoring
- `GET /health` - Health check
- `GET /ready` - Readiness check (includes model status)
- `GET /metrics` - Prometheus metrics

### Core Endpoints
- `POST /query` - Ask questions using RAG
- `POST /ingest` - Add documents to knowledge base
- `POST /feedback` - Provide feedback on responses
- `GET /graph/browse` - Browse knowledge graph

### Authentication
All endpoints (except health/metrics) require API key authentication:
```bash
curl -H "X-API-Key: your_api_key" http://localhost:8000/query
```

## ⚙️ Configuration

SubgraphRAG+ uses a **clean two-tier configuration system** that separates secrets from application settings:

- **`.env`**: Secrets and environment-specific values (never commit to git)
- **`config/config.json`**: Application settings, models, and parameters (version controlled)

This separation follows security best practices and makes deployment across environments simple.

### Quick Setup

```bash
# Copy the example and customize with your secrets
cp .env.example .env
nano .env  # Add your actual credentials and API keys
```

### Application Configuration (`config/config.json`)

The main configuration file controls all application behavior:

```json
{
  "models": {
    "backend": "mlx",
    "llm": {
      "mlx": {
        "model": "mlx-community/Qwen3-14B-8bit",
        "max_tokens": 512,
        "temperature": 0.1
      }
    },
    "embeddings": {
      "model": "Alibaba-NLP/gte-large-en-v1.5",
      "backend": "transformers"
    }
  },
  "retrieval": {
    "token_budget": 4000,
    "max_dde_hops": 2,
    "similarity_threshold": 0.7
  }
}
```

### Environment Variables (`.env`)

Contains **only secrets and environment-specific values**:

```bash
# === Database Credentials ===
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password

# === API Security ===
API_KEY_SECRET=your_secret_key

# === API Keys ===
OPENAI_API_KEY=sk-your-key  # Required for OpenAI backend
HF_TOKEN=hf_your-token      # Optional for private HF models

# === Environment ===
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=false
```

**Key Principles:**
- **Secrets in `.env`** - Never commit credentials to version control
- **Settings in `config.json`** - Application configuration is version controlled  
- **No duplication** - Each setting has one clear location
- **Environment overrides** - `.env` can override `config.json` defaults when needed

## 🧪 Testing

```bash
# Run all tests
python -m pytest

# Run minimal API tests (fast)
TESTING=1 python -m pytest tests/test_minimal.py -v

# Run with coverage
python -m pytest --cov=src tests/
```

The `TESTING=1` environment variable skips expensive operations (model loading, database connections) for faster testing.

## 🏗️ Architecture

### Core Components

- **API Layer** (`src/app/api.py`): FastAPI application with endpoints
- **Configuration** (`src/app/config.py`): Centralized configuration management
- **Database** (`src/app/database.py`): Neo4j and SQLite database interfaces
- **ML Models** (`src/app/ml/`): LLM and embedding model abstractions
- **Retrieval** (`src/app/retriever.py`): RAG retrieval logic
- **Utils** (`src/app/utils.py`): Shared utilities

### Data Flow

1. **Ingestion**: Documents → Embeddings → Neo4j Graph + Vector Index
2. **Query**: Question → Embedding → Graph Retrieval → LLM → Response
3. **Feedback**: User feedback → SQLite → Model improvement

## 🔧 Development

### Adding New LLM Backends

1. Create a new class in `src/app/ml/llm.py` implementing the `LLMInterface`
2. Add backend configuration to `config/config.json`
3. Update the factory function in `get_llm_model()`

### Adding New Embedding Backends

1. Create a new class in `src/app/ml/embedder.py` implementing the `EmbedderInterface`
2. Add backend configuration to `config/config.json`
3. Update the factory function in `get_embedder()`

### Configuration Schema

The configuration system supports:
- **Type validation**: Automatic type checking and conversion
- **Environment overrides**: Override any config value via environment variables
- **Nested configurations**: Hierarchical settings with dot notation
- **Default values**: Fallback values for optional settings

## 📊 Monitoring

### Prometheus Metrics

Available at `/metrics`:
- HTTP request metrics
- Response times
- Error rates
- Custom application metrics

### Logging

Structured logging with configurable levels:
```bash
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
LOG_FILE=logs/app.log  # Optional file output
```

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment-Specific Configs

Create different config files for each environment:
- `config/config.json` (default)
- `config/config.production.json`
- `config/config.staging.json`

Set `CONFIG_FILE` environment variable to override:
```bash
CONFIG_FILE=config/config.production.json python -m uvicorn src.app.api:app
```

## 🔒 Security

- **API Key Authentication**: All endpoints protected
- **Input Validation**: Pydantic models for request validation
- **Rate Limiting**: Built-in FastAPI rate limiting
- **CORS**: Configurable cross-origin resource sharing

## 📈 Performance

### Optimization Features

- **Lazy Loading**: Models loaded only when needed
- **Connection Pooling**: Efficient database connections
- **Caching**: Response and embedding caching
- **Apple Silicon**: MLX backend for M1/M2/M3 optimization

### Benchmarks

- **Cold Start**: ~2-3 seconds (with model loading)
- **Query Response**: ~200-500ms (cached embeddings)
- **Ingestion**: ~100-200 docs/minute

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `TESTING=1 python -m pytest`
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed and virtual environment is activated

**Database Connection**: Verify Neo4j is running and credentials in `.env` are correct

**Model Loading**: Check model names in `config/config.json` and ensure sufficient disk space

**API Key Issues**: Verify `API_KEY_SECRET` is set and using correct header format

### Getting Help

- Check the logs: `tail -f logs/app.log`
- Run health checks: `curl http://localhost:8000/health`
- Test configuration: `TESTING=1 python -c "from src.app.config import config; print(config)"`

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
    headers={"X-API-Key": "your-api-key"},
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

SubgraphRAG+ uses a hybrid configuration approach following security best practices:

- **`.env`** - Secrets and environment-specific values
- **`config/config.json`** - Application settings and model configurations

#### 🔒 Environment Variables (.env)

Essential production environment variables for secrets and environment-specific settings:

```bash
# === Database Credentials ===
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-production-password

# === API Security ===
API_KEY_SECRET=your-secure-api-key

# === API Keys ===
OPENAI_API_KEY=your-openai-api-key  # Required if using OpenAI backend
HF_TOKEN=your-hf-token              # Optional, for private HuggingFace models

# === Environment Settings ===
ENVIRONMENT=production              # development|staging|production
LOG_LEVEL=INFO                      # DEBUG|INFO|WARNING|ERROR|CRITICAL
DEBUG=false

# === Optional: Custom Model Paths ===
# MLX_LLM_MODEL_PATH=/path/to/custom/mlx/model
# HF_MODEL_PATH=/path/to/custom/hf/model
```

#### ⚙️ Application Configuration (config/config.json)

Model settings and application configuration:

```json
{
  "models": {
    "backend": "mlx",
    "llm": {
      "mlx": {
        "model": "mlx-community/Qwen3-14B-8bit",
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.9
      },
      "openai": {
        "model": "gpt-3.5-turbo",
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.9
      }
    },
    "embeddings": {
      "model": "Alibaba-NLP/gte-large-en-v1.5",
      "backend": "transformers"
    }
  },
  "retrieval": {
    "token_budget": 4000,
    "max_dde_hops": 2,
    "similarity_threshold": 0.7
  },
  "performance": {
    "cache_size": 1000,
    "api_rate_limit": 60,
    "timeout_seconds": 30
  }
}
```

#### 🔑 Configuration Best Practices

1. **Never commit secrets**: Keep `.env` in `.gitignore`
2. **Use environment overrides**: Local `.env` can override `config.json` defaults
3. **Embedding consistency**: Always use `transformers` backend for embeddings (never MLX)
4. **Backend separation**: MLX for LLM only, transformers for embeddings only

#### 🍎 Apple Silicon (MLX) Configuration

For optimal performance on M1/M2/M3 Macs:

```bash
# In .env
LOG_LEVEL=DEBUG  # To see MLX initialization logs
```

```json
// In config/config.json
{
  "models": {
    "backend": "mlx",
    "llm": {
      "mlx": {
        "model": "mlx-community/Qwen3-14B-8bit",
        "max_tokens": 1024,
        "temperature": 0.1
      }
    },
    "embeddings": {
      "model": "Alibaba-NLP/gte-large-en-v1.5",
      "backend": "transformers"
    }
  }
}
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

## 🚀 Key Improvements Over Original SubgraphRAG

### 1. **Production-Grade Information Extraction**
- **REBEL IE Service**: Uses Babelscape/rebel-large for proper triple extraction from raw text
- **Schema-Driven Entity Typing**: Replaces naive string heuristics with authoritative type mappings
- **Domain Adaptability**: Works with Biblical text, legal documents, scientific papers, etc.
- **Offline Operation**: No external API dependencies, fully self-contained

### 2. **Dynamic Knowledge Graph Construction**
- **Live Ingestion Pipeline**: Build KGs from any text corpus in real-time
- **Incremental Updates**: Add new content without rebuilding entire graph
- **Quality Control**: Deduplication, validation, and error handling

### 3. **Enhanced Retrieval & Reasoning**
- **Hybrid Retrieval**: Combines graph traversal with dense vector search
- **MLP-Based Scoring**: Uses original SubgraphRAG MLP (no retraining needed)
- **Budget-Aware Assembly**: Optimizes subgraph size for LLM context windows

### 4. **Enterprise-Ready Architecture**
- **Microservices**: Containerized IE service, API layer, database components
- **Monitoring**: Health checks, metrics, logging, alerting
- **Scalability**: Horizontal scaling, caching, batch processing

## 📖 Quick Start with Biblical Text

```bash
# 1. Start the full stack
make docker-start

# 2. Start the IE service
uvicorn src.app.ie_service:app --host 0.0.0.0 --port 8003

# 3. Ingest Biblical text
python scripts/ingest_with_ie.py data/genesis.txt

# 4. Process staged triples
python scripts/ingest_worker.py --process-all

# 5. Query the knowledge graph
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who parted the Red Sea?"}'
```

The system will:
1. Extract triples using REBEL: `(Moses, parted, Red Sea)`
2. Type entities using schema: `Moses → Person, Red Sea → Location`
3. Build knowledge graph with proper relationships
4. Answer queries with precise citations and subgraph evidence
