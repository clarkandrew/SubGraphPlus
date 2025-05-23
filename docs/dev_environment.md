# Setting Up Your Development Environment for SubgraphRAG+

This guide provides detailed instructions for setting up a comprehensive development environment for SubgraphRAG+. Whether you're contributing to the project or customizing it for your own use, these steps will help you establish a proper development setup.

## Quick Setup (Recommended)

The fastest way to set up a complete development environment is to use our Make setup command:

```bash
# Clone the repository
git clone https://github.com/yourusername/SubgraphRAG+.git
cd SubgraphRAG+

# Run the complete setup command (with Docker)
make setup-all

# OR with locally installed Neo4j
./bin/setup_dev.sh --use-local-neo4j
```

This will automatically:
- Set up a Python virtual environment
- Install all development dependencies
- Start Neo4j using Docker (or connect to local Neo4j if using `--use-local-neo4j` flag)
- Download pre-trained models
- Initialize the database schema
- Load sample data
- Run tests to verify everything works

After running this command, you'll have a fully functional development environment ready to go!

For those who prefer a more manual approach or need to customize the setup process, follow the detailed steps below.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+** - The core programming language
- **Git** - For version control
- **pip** - For package management (comes with Python)
- **One of the following**:
  - **Docker & Docker Compose** - For containerized Neo4j
  - **Neo4j** (4.4+) - Locally installed with APOC plugin
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

### Option A: Using Docker

```bash
# Start Neo4j with Docker
make neo4j-start
```

This will start a Neo4j container with the APOC plugin and expose ports 7474 (HTTP) and 7687 (Bolt).

### Option B: Using Neo4j Desktop (Recommended for local development)

1. Download [Neo4j Desktop](https://neo4j.com/download/)
2. Install and run Neo4j Desktop
3. Create a new project and add a local database (version 4.4+ recommended)
4. Set password to "password" (or update your .env file accordingly)
5. Start the database
6. Install the APOC plugin via the Plugins tab in Neo4j Desktop

### Option C: Using Homebrew (macOS)

```bash
# Install Neo4j
brew install neo4j

# Start the service
brew services start neo4j

# Set password (will prompt for password change)
cypher-shell -u neo4j -p neo4j

# Install APOC plugin
wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.4.0.10/apoc-4.4.0.10-all.jar
mv apoc-4.4.0.10-all.jar /usr/local/var/neo4j/plugins/
brew services restart neo4j
```

### Option D: Manual Installation

1. [Download Neo4j Community Edition](https://neo4j.com/download/)
2. Install the APOC plugin
3. Configure Neo4j to accept connections with these settings in `neo4j.conf`:
   ```
   dbms.connector.bolt.enabled=true
   dbms.connector.bolt.listen_address=0.0.0.0:7687
   dbms.connector.http.enabled=true
   dbms.connector.http.listen_address=0.0.0.0:7474
   dbms.security.procedures.unrestricted=apoc.*
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

### Using Make for Development Tasks

```bash
# Set up the entire development environment
make setup-all

# Install dependencies only
make setup-dev

# Start Neo4j with Docker
make neo4j-start

# OR if using local Neo4j
# Ensure your Neo4j installation is running

# Download models only
make get-pretrained-mlp

# Initialize database schema
make migrate-schema

# Run tests only
make test

# See all available commands
make help
```

### Running Tests

```bash
# Run all tests with the test script
./bin/run_tests.sh

# Or run pytest directly
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

## Development Environments

### Docker Development Environment

For a consistent development environment, you can use Docker:

```bash
# Start Docker Compose stack
make docker-start

# Access the API container shell (using Docker directly)
docker exec -it subgraphrag_api /bin/bash

# Run tests
make test
```

### Local Neo4j Development Environment

You can also develop using a locally installed Neo4j instance:

```bash
# Set up development environment with local Neo4j
./bin/setup_dev.sh --use-local-neo4j

# Start the development server
python main.py --reload

# Run tests
make test
```

## Troubleshooting

### Common Issues

1. **First Try the Complete Setup Command**

   - If you're having setup issues, try the automated approach:
   - `make setup-all` (for Docker-based setup)
   - `./bin/setup_dev.sh --use-local-neo4j` (for local Neo4j setup)
   - This often resolves environment-related issues automatically

2. **Neo4j Connection Problems**
   - For Docker-based Neo4j:
     - Ensure Neo4j container is running: `docker ps | grep neo4j`
   - For locally installed Neo4j:
     - Check if Neo4j service is running:
       - Neo4j Desktop: Check the database status in the application
       - Homebrew: `brew services list | grep neo4j`
       - Linux: `systemctl status neo4j` 
   - Check connection string in `.env`
   - Try accessing Neo4j Browser: http://localhost:7474

3. **Package Import Errors**
   - Verify your virtual environment is activated
   - Update dependencies: `pip install -r requirements-dev.txt`

4. **Model Loading Errors**
   - Confirm models are downloaded: `ls -la models/`
   - Check model paths in config

5. **Port Conflicts**
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

For more comprehensive troubleshooting, consult our [Troubleshooting Guide](./troubleshooting.md).

Happy coding!