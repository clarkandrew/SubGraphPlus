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

- Python 3.11+
- Neo4j (4.4+) with APOC plugin
- SQLite3
- (Optional) Local model support: MLX (Apple Silicon) or Hugging Face models
- (Optional) OpenAI API key (for OpenAI backend)

## 🚀 Quick Start

1. **Setup Development Environment**

```bash
# Clone repo (if needed)
git clone https://github.com/yourusername/SubgraphRAGPlus.git
cd SubgraphRAGPlus

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

4. **Test with a Query**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: default_key_for_dev_only" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who founded Tesla?", "visualize_graph": true}'
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | `neo4j://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |
| `API_KEY_SECRET` | API key for authentication | (auto-generated) |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI backend) | None |

### Config Options (config/config.json)

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
├── run.sh             # Application runner script
├── run_tests.sh       # Test runner script
├── run_benchmark.sh   # Benchmark runner script
└── backup.sh          # Backup operations script
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

### Using Shell Scripts

We've added convenient shell scripts to simplify common operations:

```bash
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

### Using Make Commands

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

1. **Option 1**: Automatic download (requires internet)
   ```bash
   make get-pretrained-mlp
   ```

2. **Option 2**: Manual acquisition
   - Visit the [SubgraphRAG Colab notebook](https://colab.research.google.com/drive/...)
   - Run the notebook with `KGQA_DATASET="webqsp"`
   - Download the resulting `cpt.pth` file
   - Place it at `models/mlp_pretrained.pt`

## 📝 License

This project is licensed under the Apache License 2.0 - see the `LICENSE` file for details.

## 🛠️ Operational Features

### Backup and Restore

The system includes a comprehensive backup and restore solution:

- Full backup of Neo4j database, SQLite staging DB, FAISS indices, and configuration
- Metadata tracking for all backups with timestamps and component status
- Selective restoration of specific backups
- Docker-aware operation for containerized deployments

### Advanced Evaluation

- Precision, recall, and F1 score metrics with ground truth support
- Robustness evaluation with adversarial test questions
- Detailed HTML reports with visualizations
- Entity linking accuracy and hallucination detection

## 🌟 Acknowledgements

Based on the original [SubgraphRAG paper](https://arxiv.org/abs/2401.09863) (ICLR 2025) with significant enhancements.