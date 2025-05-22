# Setting Up Your Development Environment for SubgraphRAG+

This guide provides detailed instructions for setting up a comprehensive development environment for SubgraphRAG+. Whether you're contributing to the project or customizing it for your own use, these steps will help you establish a proper development setup.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+** - The core programming language
- **Git** - For version control
- **pip** - For package management (comes with Python)
- **Docker & Docker Compose** (optional) - For containerized development
- **make** - For using the project's Makefile shortcuts

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/SubgraphRAG+.git
cd SubgraphRAG+
```

## Step 2: Set Up a Virtual Environment

### Using venv (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/macOS
source venv/bin/activate

# On Windows
# venv\Scripts\activate
```

### Using conda (Alternative)

```bash
# Create a conda environment
conda create -n subgraphrag python=3.11
conda activate subgraphrag
```

## Step 3: Install Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

This installs both the production and development dependencies, which include:
- Testing tools (pytest, pytest-cov)
- Linting tools (flake8, black, isort)
- Type checking (mypy)
- Documentation tools (mkdocs)

## Step 4: Configure Pre-commit Hooks

We use pre-commit hooks to ensure code quality before commits.

```bash
# Install pre-commit
pip install pre-commit

# Set up the pre-commit hooks
pre-commit install
```

## Step 5: Set Up Neo4j for Development

### Option A: Using Docker (Recommended)

```bash
# Start Neo4j with Docker
make neo4j-start
```

This will start a Neo4j container with the APOC plugin and expose ports 7474 (HTTP) and 7687 (Bolt).

### Option B: Manual Installation

1. [Download Neo4j Community Edition](https://neo4j.com/download/)
2. Install the APOC plugin
3. Configure Neo4j to accept connections with these settings in `neo4j.conf`:
   ```
   dbms.connector.bolt.enabled=true
   dbms.connector.bolt.listen_address=0.0.0.0:7687
   dbms.connector.http.enabled=true
   dbms.connector.http.listen_address=0.0.0.0:7474
   ```

## Step 6: Create Local Configuration

Create a `.env` file in the project root with your development settings:

```
# Neo4j Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# API Security
API_KEY_SECRET=dev_key_only_for_local_use

# Model Backend
MODEL_BACKEND=hf
# Or if using OpenAI:
# MODEL_BACKEND=openai
# OPENAI_API_KEY=your_openai_api_key

# Development Settings
DEBUG=True
LOG_LEVEL=DEBUG
RELOAD=True
```

## Step 7: Initialize the Database Schema

```bash
# Apply database migrations
python scripts/migrate_schema.py
```

## Step 8: Download Pre-trained Models

```bash
# Download models (MLP, embeddings, etc.)
python scripts/download_models.py
```

## Step 9: Set Up Test Data

```bash
# Load sample data for testing
python scripts/stage_ingest.py --sample
python scripts/ingest_worker.py --process-all
python scripts/merge_faiss.py
```

## Step 10: Running the Development Server

```bash
# Start the development server with auto-reload
python main.py --reload

# Or using Make
make serve
```

The API will be available at http://localhost:8000 with Swagger documentation at http://localhost:8000/docs.

## Step 11: Setting Up Your Editor

### VS Code

1. Install the following extensions:
   - Python
   - Black Formatter
   - isort
   - Flake8
   - Neo4j Cypher
   - Docker

2. Configure settings.json with:
   ```json
   {
     "python.formatting.provider": "black",
     "python.linting.enabled": true,
     "python.linting.flake8Enabled": true,
     "editor.formatOnSave": true,
     "editor.codeActionsOnSave": {
       "source.organizeImports": true
     }
   }
   ```

### PyCharm

1. Install the following plugins:
   - Black
   - isort
   - Flake8
   - Cypher

2. Configure External Tools:
   - Black: `$PyInterpreterDirectory$/black $FilePath$`
   - isort: `$PyInterpreterDirectory$/isort $FilePath$`

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v

# Run with filtering
pytest tests/test_api.py::test_query_endpoint
```

### Linting and Formatting

```bash
# Run all linting checks
make lint

# Format code
black app/ tests/ scripts/

# Sort imports
isort app/ tests/ scripts/

# Type checking
mypy app/
```

### Submitting Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add your meaningful commit message"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request on GitHub

## Docker Development Environment

For a consistent development environment, you can use Docker:

```bash
# Build and start the development environment
./bin/docker-setup.sh rebuild

# Access the API container shell
./bin/docker-setup.sh api-shell

# Run tests in the container
./bin/docker-setup.sh tests
```

## Troubleshooting

### Common Issues

1. **Neo4j Connection Problems**
   - Ensure Neo4j is running: `docker ps | grep neo4j`
   - Check connection string in `.env`
   - Try accessing Neo4j Browser: http://localhost:7474

2. **Package Import Errors**
   - Verify your virtual environment is activated
   - Update dependencies: `pip install -r requirements-dev.txt`

3. **Model Loading Errors**
   - Confirm models are downloaded: `ls -la models/`
   - Check model paths in config

4. **Port Conflicts**
   - Check if ports are in use: `lsof -i :8000`
   - Change the port in your configuration

### Logs

Check the logs for detailed error information:

```bash
# Application logs
cat logs/app.log

# Debug logs
cat logs/debug.log
```

## Advanced Development Topics

### Custom Model Integration

To add your own model backend:

1. Create a new module in `app/ml/backends/`
2. Implement the required interface classes
3. Register your backend in `app/ml/llm.py`

### Database Migrations

To create a new database migration:

1. Create a new migration file in `migrations/neo4j/`
2. Add the migration to the version list in `scripts/migrate_schema.py`
3. Test the migration: `python scripts/migrate_schema.py --dry-run`

### Benchmarking

Run benchmarks to test performance:

```bash
./bin/run_benchmark.sh
```

Results will be saved to `evaluation/results/`.

## Next Steps

- Review the [Architecture Documentation](./architecture.md)
- Consult the [API Reference](./api_reference.md)
- Explore the [Developer Guide](./developer_guide.md)

Happy coding!