# 🚀 Getting Started with SubgraphRAG+

This guide will help you get up and running with SubgraphRAG+ quickly. The simplest way to start is with our one-step setup command using Make, but you also have other options:

1. **One-step Make setup** (recommended for everyone)
2. **Docker environment** (alternative for production)
3. **Manual local development environment** (alternative for advanced customization)

## Prerequisites

### For All Methods
- Git
- Python 3.9+ (Python 3.11+ recommended)
- 10GB of free disk space

### Additional Requirements for Docker
- Docker Engine 20.10+
- Docker Compose v2
- At least 4GB of RAM allocated to Docker

### Alternative Requirements (Non-Docker)
- Neo4j (4.4+) installed locally if not using Docker
- APOC plugin for Neo4j

### Optional Components
- GPU support for local model inference
- A virtual environment manager (venv, conda, etc.) - our quickstart script will create one for you if needed

## One-Step Makefile Setup (Recommended)

The `setup-all` Make target automates the entire setup process, making it easy to get started regardless of your experience level.

### 1. Clone the Repository

```bash
git clone https://github.com/clarkandrew/SubgraphRAG+.git
cd SubgraphRAG+
```

### 2. Run the Complete Setup Command

```bash
make setup-all
```

This single command will:
- Set up a Python virtual environment
- Install all dependencies
- Start Neo4j using Docker
- Download pre-trained models
- Initialize the database schema
- Load sample data
- Run tests to verify everything works

### 3. Understanding the Makefile

The Makefile is the central command hub for this project. You can run any of these targets separately as needed:

```bash
# Install dependencies only
make setup-dev

# Start Neo4j with Docker
make neo4j-start

# Download models only
make get-pretrained-mlp

# Run tests only
make test

# See all available commands
make help
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
make neo4j-stop
# OR if using Docker Compose
make docker-stop
```

## Alternative Setup Methods

If you prefer more control over the setup process, or if the quickstart doesn't work for your environment, you can use these alternative methods.

### Docker Environment Setup

Using Docker provides consistent behavior across platforms and is ideal for production deployments.

```bash
# Start the system
make docker-start

# Load sample data
make ingest-sample
```

### Local Neo4j Setup (Without Docker)

If you prefer not to use Docker, you can install Neo4j directly on your system:

#### Option 1: Neo4j Desktop (Recommended for development)

1. Download Neo4j Desktop from [neo4j.com/download](https://neo4j.com/download/)
2. Install and run Neo4j Desktop
3. Create a new project and add a local database (version 4.4+ recommended)
4. Set password to "password" (or update your .env file accordingly)
5. Start the database
6. Install the APOC plugin via the Plugins tab in Neo4j Desktop

#### Option 2: Install Neo4j with Homebrew (macOS)

```bash
# Install Neo4j
brew install neo4j

# Start the service
brew services start neo4j

# Set password (will prompt for password change)
cypher-shell -u neo4j -p neo4j

# Install APOC plugin (may require additional steps)
```

### Manual Local Development Setup

For advanced development and customization:

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Linux/MacOS)
source venv/bin/activate
# OR on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Option 1: Start Neo4j with Docker
make neo4j-start

# Option 2: If using locally installed Neo4j
# Ensure Neo4j is running:
# - If using Neo4j Desktop: Start from the application
# - If using Homebrew: brew services start neo4j

# Apply database migrations
python scripts/migrate_schema.py

# Download models
python scripts/download_models.py

# Load sample data
python scripts/stage_ingest.py --sample
python scripts/ingest_worker.py --process-all
python scripts/merge_faiss.py
```

### Configuration

All setup methods will create default configuration, but you can customize it:

#### Environment Variables (.env file)

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

#### Configuration File (config/config.json)

For more detailed configuration, edit `config/config.json` after initial setup.

### Starting the Server

After setup is complete (using any method), start the server:

```bash
# Start the development server
make serve

# OR for production mode
make serve-prod

# OR manually
source venv/bin/activate
python main.py --reload
```

### Accessing the System

Once the server is running:

- **API Documentation**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474 (user: neo4j, password: password)
- **API Endpoint**: http://localhost:8000

## Common Tasks

### Quick Reference

```bash
# Complete setup in one step
make setup-all

# Start the server
make serve

# Run all tests
make test

# Start Neo4j (Docker)
make neo4j-start
```

### Docker Management

```bash
# Start Docker Compose stack
make docker-start

# Stop Docker Compose stack
make docker-stop

# For additional Docker operations, you can use docker/docker-compose commands directly:
docker compose logs  # View logs
docker stats         # Check resource usage
docker exec -it subgraphrag_api /bin/bash  # Access shell in API container
```

### Development Tasks

```bash
# Run linting checks
make lint

# Rebuild FAISS index
make rebuild-faiss-index

# Run benchmarks
./bin/run_benchmark.sh
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

1. **Try the complete setup command first**:
   ```bash
   # It automatically handles most common setup issues
   make setup-all
   ```

2. Check the logs:
   ```bash
   # Docker
   docker compose logs

   # Local
   cat logs/app.log
   ```

3. Ensure Neo4j is running:
   ```bash
   # Docker
   docker ps | grep neo4j

   # Local installation
   curl http://localhost:7474
   # Or check via Neo4j Desktop application
   ```

4. Verify that environment variables are set correctly

For more detailed help, refer to the [Troubleshooting](./troubleshooting.md) guide or open an issue on GitHub.
