# 🚀 SubgraphRAG+

> Knowledge Graph Question Answering with Hybrid Retrieval and Visualization

SubgraphRAG+ is a knowledge graph-powered QA system that combines structured graph traversal with dense vector retrieval to provide accurate, contextual answers with explanatory visualizations. Built on the original SubgraphRAG research, this enhanced version adds dynamic knowledge graph ingestion, hybrid retrieval, and enterprise-grade API features.

## 🌟 Key Features

- **Hybrid Retrieval**: Combines graph traversal and semantic search for optimal recall
- **Dynamic KG Ingestion**: Real-time triple ingestion with deduplication
- **SSE Streaming**: Token-by-token LLM response with citations and graph data
- **Visualization-Ready**: D3.js compatible graph data with relevance scores
- **Enterprise-Grade API**: OpenAPI compliant with comprehensive documentation

## 📋 Prerequisites

- Docker and Docker Compose (recommended deployment method)
- OR for local development:
  - Python 3.11+
  - Neo4j (4.4+) with APOC plugin
  - SQLite3
  - (Optional) Local model support: MLX (Apple Silicon) or Hugging Face models
  - (Optional) OpenAI API key (for OpenAI backend)

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Start the System with Docker**

```bash
# Clone repo (if needed)
git clone https://github.com/yourusername/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Start everything with Docker Compose
./docker-setup.sh start

# Initialize with sample data (optional)
./docker-setup.sh sample-data
```

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

### Local Development (Alternative)

1. **Setup Local Development Environment**

```bash
# Clone repo (if needed)
git clone https://github.com/yourusername/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Using the setup script
./setup.sh

# OR manually:
# Install dependencies
make setup-dev

# Download pre-trained MLP model
make get-pretrained-mlp

# Start Neo4j (assuming Docker is installed)
make neo4j-start

# Apply schema migration
make migrate-schema
```

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

## 📂 Project Structure

```
SubgraphRAG+/
├── app/               # Core application code
│   ├── api.py         # FastAPI application and endpoints
│   ├── config.py      # Configuration management
│   ├── database.py    # Neo4j and SQLite connections
│   ├── models.py      # Data models
│   ├── retriever.py   # Hybrid retrieval logic
│   ├── utils.py       # Utility functions
│   ├── verify.py      # Output verification
│   └── ml/            # Machine learning modules
│       ├── embedder.py    # Text embedding
│       └── llm.py         # Language model interface
├── config/            # Configuration files
├── data/              # Data storage
├── docs/              # Documentation
├── evaluation/        # Evaluation and benchmarking
│   ├── benchmark.py           # Benchmark script
│   ├── sample_questions.json  # Standard test questions
│   ├── adversarial_questions.json # Robustness test questions
│   └── ground_truth.json      # Ground truth for metrics
├── migrations/        # Neo4j schema migrations
├── models/            # ML model storage
├── prompts/           # Prompt templates
├── scripts/           # Utility scripts
│   ├── backup_restore.py      # Backup/restore functionality
│   ├── download_models.py     # MLP model download
│   └── ...
├── tests/             # Test suite
│   ├── test_adversarial.py    # Adversarial tests
│   ├── test_api.py            # API tests
│   ├── test_smoke.py          # Smoke and edge case tests
│   └── ...
├── docker-setup.sh    # Docker management script
├── run.sh             # Local application runner script
├── run_tests.sh       # Test runner script
├── run_benchmark.sh   # Benchmark runner script
├── backup.sh          # Backup operations script
├── Dockerfile         # Docker image definition
└── docker-compose.yml # Docker services configuration
```

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

For detailed information about the API endpoints, request/response formats, and SSE events, refer to `documentation/api_reference.md`.

For detailed API documentation, start the server and visit `http://localhost:8000/docs`.

## 🔧 Common Tasks

### Using Docker (Recommended)

```bash
# Start/stop the system
./docker-setup.sh start         # Start all services
./docker-setup.sh stop          # Stop all services
./docker-setup.sh restart       # Restart all services
./docker-setup.sh status        # Show service status

# Working with data
./docker-setup.sh sample-data   # Initialize with sample data
./docker-setup.sh backup        # Create data backup

# Monitoring and debugging
./docker-setup.sh logs          # View all service logs
./docker-setup.sh resources     # Check container resource usage

# Development tasks
./docker-setup.sh rebuild       # Rebuild and restart services
./docker-setup.sh api-shell     # Open shell in API container
./docker-setup.sh neo4j-shell   # Open shell in Neo4j container
./docker-setup.sh tests         # Run tests in container
```

### Using Local Shell Scripts

If developing locally without Docker, use these scripts:

```bash
# Setup environment
./setup.sh                      # Complete environment setup
./setup.sh --skip-neo4j         # Setup without Neo4j
./setup.sh --skip-models        # Setup without downloading models

# Run the application
./run.sh

# Run tests (with various options)
./run_tests.sh                  # Run all tests
./run_tests.sh -t unit          # Run only unit tests
./run_tests.sh -t smoke         # Run smoke tests
./run_tests.sh -t adversarial   # Run adversarial tests
./run_tests.sh -c               # Generate coverage report

# Run benchmarks
./run_benchmark.sh              # Run standard benchmark
./run_benchmark.sh -a           # Run adversarial benchmark
./run_benchmark.sh -r           # Generate detailed HTML report
./run_benchmark.sh -g evaluation/ground_truth.json  # Use ground truth

# Backup and restore operations
./backup.sh backup              # Create a new backup
./backup.sh restore -i backup_20230101_120000  # Restore specific backup
./backup.sh list                # List available backups
```

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

## 🌟 Acknowledgements

Based on the original [SubgraphRAG paper](https://arxiv.org/abs/2401.09863) (ICLR 2025) with significant enhancements.