# 🚀 Getting Started with SubgraphRAG+

This guide will help you get up and running with SubgraphRAG+ quickly. You have two options for running the system:
1. **Docker environment** (recommended for quick start and production)
2. **Local development environment** (recommended for development and customization)

## Prerequisites

### For Docker Environment
- Docker Engine 20.10+
- Docker Compose v2
- At least 4GB of RAM allocated to Docker
- 10GB of free disk space

### For Local Development
- Python 3.11+
- Neo4j 4.4+ with APOC plugin
- SQLite3
- Git
- A virtual environment manager (venv, conda, etc.)
- (Optional) GPU support for local model inference

## Docker Environment Setup

Using Docker is the fastest way to get started with SubgraphRAG+ and ensures consistent behavior across platforms.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SubgraphRAG+.git
cd SubgraphRAG+
```

### 2. Start the System

```bash
./bin/docker-setup.sh start
```

This command will:
- Build the Docker images if they don't exist
- Start the Neo4j database container
- Start the SubgraphRAG+ API container
- Initialize the necessary volumes

### 3. Load Sample Data (Optional)

To try the system with pre-loaded sample data:

```bash
./bin/docker-setup.sh sample-data
```

### 4. Access the System

- **API Documentation**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474 (user: neo4j, password: password)
- **API Endpoint**: http://localhost:8000

### 5. Test with a Simple Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: changeme" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who founded Tesla?", "visualize_graph": true}'
```

### 6. Stopping the System

```bash
./bin/docker-setup.sh stop
```

## Local Development Environment Setup

For development, customization, or contributing to the project, setting up a local environment is recommended.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SubgraphRAG+.git
cd SubgraphRAG+
```

### 2. Create and Activate a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Linux/MacOS)
source venv/bin/activate

# Activate it (Windows)
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all dependencies including development tools
pip install -r requirements-dev.txt

# OR use the setup script for a complete setup
./bin/setup.sh
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```
# Neo4j Connection
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# API Security
API_KEY_SECRET=changeme_in_production

# Model Backend
MODEL_BACKEND=openai
# Uncomment and fill if using OpenAI API
# OPENAI_API_KEY=your_openai_api_key

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Logging
LOG_LEVEL=INFO
```

### 5. Start Neo4j (If Not Already Running)

```bash
# Using Docker (recommended for development)
make neo4j-start

# OR manually install Neo4j Community Edition from https://neo4j.com/download/
```

### 6. Initialize the Database Schema

```bash
python scripts/migrate_schema.py
```

### 7. Download Pre-trained MLP Model

```bash
make get-pretrained-mlp
# OR
python scripts/download_models.py
```

### 8. Load Sample Data (Optional)

```bash
# Stage and ingest sample triples
python scripts/stage_ingest.py --sample
python scripts/ingest_worker.py --process-all
python scripts/merge_faiss.py
```

### 9. Start the Development Server

```bash
# Start the development server with hot reload
make serve
# OR
python main.py --reload
```

### 10. Access the System

- **API Documentation**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474 (user: neo4j, password: password)
- **API Endpoint**: http://localhost:8000

## Common Tasks

### Docker Environment

```bash
# View logs
./bin/docker-setup.sh logs

# Rebuild images after changes
./bin/docker-setup.sh rebuild

# Check resource usage
./bin/docker-setup.sh resources

# Access shell in API container
./bin/docker-setup.sh api-shell

# Run tests in container
./bin/docker-setup.sh tests
```

### Local Development Environment

```bash
# Run all tests
./bin/run_tests.sh
# OR
pytest

# Run linting checks
make lint

# Rebuild FAISS index
make rebuild-faiss-index

# Run benchmarks
./bin/run_benchmark.sh

# Create a backup
./bin/backup.sh backup
```

## Updating Configuration

### Docker Environment

Edit the configuration values in `docker-compose.yml` or create a `.env` file in the project root.

### Local Development

Edit the file at `config/config.json` or set environment variables directly.

## Project Structure

```
SubgraphRAG+/
├── app/               # Core application code
├── bin/               # Shell scripts
├── config/            # Configuration files
├── data/              # Data storage
├── docs/              # Documentation
├── evaluation/        # Benchmarking tools
├── examples/          # Example scripts
├── migrations/        # Neo4j schema migrations
├── models/            # ML model storage
├── prompts/           # Prompt templates
├── scripts/           # Utility scripts
└── tests/             # Test suite
```

## Next Steps

- Read [API Reference](./api_reference.md) for detailed API documentation
- Explore [Architecture](./architecture.md) for system design details
- Check [Developer Guide](./developer_guide.md) for contribution guidelines
- See [Deployment](./deployment.md) for production deployment tips

## Troubleshooting

If you encounter issues:

1. Check the logs:
   ```bash
   # Docker
   ./bin/docker-setup.sh logs
   
   # Local
   cat logs/app.log
   ```

2. Ensure Neo4j is running:
   ```bash
   # Docker
   docker ps | grep neo4j
   
   # Local
   curl http://localhost:7474
   ```

3. Verify that environment variables are set correctly

For more help, refer to the [Troubleshooting](./troubleshooting.md) guide or open an issue on GitHub.