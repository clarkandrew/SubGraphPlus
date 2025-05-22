# 🚀 Getting Started with SubgraphRAG+

This guide will help you get up and running with SubgraphRAG+ quickly. The simplest way to start is with our one-step quickstart script, but you also have other options:

1. **One-step quickstart script** (recommended for everyone)
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

### Optional Components
- GPU support for local model inference
- A virtual environment manager (venv, conda, etc.) - our quickstart script will create one for you if needed

## One-Step Quickstart (Recommended)

The quickstart script automates the entire setup process, making it easy to get started regardless of your experience level.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SubgraphRAG+.git
cd SubgraphRAG+
```

### 2. Run the Quickstart Script

```bash
./bin/quickstart.sh
```

This single command will:
- Set up a Python virtual environment
- Install all dependencies
- Start Neo4j using Docker
- Download pre-trained models
- Initialize the database schema
- Load sample data
- Run tests to verify everything works

### 3. Customizing the Quickstart

The quickstart script has several options:

```bash
# Skip running tests
./bin/quickstart.sh --skip-tests

# Skip Docker setup (if you have Neo4j installed separately)
./bin/quickstart.sh --skip-docker

# Skip loading sample data
./bin/quickstart.sh --skip-sample-data

# Use production dependencies only
./bin/quickstart.sh --prod

# Use a specific Python version
./bin/quickstart.sh --python python3.10

# See all options
./bin/quickstart.sh --help
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

## Alternative Setup Methods

If you prefer more control over the setup process, or if the quickstart doesn't work for your environment, you can use these alternative methods.

### Docker Environment Setup

Using Docker provides consistent behavior across platforms and is ideal for production deployments.

```bash
# Start the system
./bin/docker-setup.sh start

# Load sample data
./bin/docker-setup.sh sample-data
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

# Start Neo4j (requires Docker)
make neo4j-start

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
# If using quickstart or manual setup
source venv/bin/activate
python main.py --reload

# OR use the convenience script
./bin/run.sh
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
./bin/quickstart.sh

# Start the server
./bin/run.sh

# Run all tests
./bin/run_tests.sh

# Create a backup
./bin/backup.sh backup
```

### Docker Management

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

1. **Try the quickstart script first**:
   ```bash
   # It automatically handles most common setup issues
   ./bin/quickstart.sh
   ```

2. Check the logs:
   ```bash
   # Docker
   ./bin/docker-setup.sh logs
   
   # Local
   cat logs/app.log
   ```

3. Ensure Neo4j is running:
   ```bash
   # Docker
   docker ps | grep neo4j
   
   # Local
   curl http://localhost:7474
   ```

4. Verify that environment variables are set correctly

For more detailed help, refer to the [Troubleshooting](./troubleshooting.md) guide or open an issue on GitHub.