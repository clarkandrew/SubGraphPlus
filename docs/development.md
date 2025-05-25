# Development Guide

This guide covers development workflows, testing strategies, and contribution guidelines for SubgraphRAG+.

## Development Environment Setup

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git
- Neo4j (optional for local development)

### Quick Development Setup

```bash
# Clone repository
git clone https://github.com/clarkandrew/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Setup development environment
make dev-setup
# OR
./bin/setup_dev.sh

# Start development server
make dev-start
# OR
./bin/run.sh --dev
```

### Development with Docker

```bash
# Start all services in development mode
make docker-dev

# View logs
make logs

# Restart specific service
docker-compose restart subgraphrag
```

### Docker Troubleshooting

#### Common Docker Issues

**1. Docker Daemon Not Running**
```bash
# Error: Cannot connect to the Docker daemon
# Solution: Start Docker daemon

# macOS (Docker Desktop)
open -a Docker
# OR use helper script
./bin/start_docker.sh

# Linux (systemd)
sudo systemctl start docker
sudo systemctl enable docker

# Linux (service)
sudo service docker start
```

**2. Docker Compose File Not Found**
```bash
# Error: no configuration file provided: not found
# Make sure you're in the project root directory with docker-compose.yml

# Check if file exists
ls -la docker-compose.yml

# If missing, create from template
cp deployment/docker-compose.yml .
```

**3. Port Already in Use**
```bash
# Error: Port 7474 or 7687 is already allocated
# Check what's using the ports
lsof -i :7474
lsof -i :7687

# Stop conflicting services
sudo systemctl stop neo4j  # Linux
brew services stop neo4j   # macOS with Homebrew

# Or use different ports in docker-compose.yml
```

**4. Permission Issues**
```bash
# Error: Permission denied while trying to connect to Docker daemon
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo (not recommended for development)
sudo docker-compose up -d
```

**5. Container Won't Start**
```bash
# Check container status
docker ps -a

# View container logs
docker logs subgraphrag_neo4j
docker logs subgraphrag_api

# Remove and recreate containers
docker-compose down
docker-compose up -d --force-recreate
```

**6. Neo4j Container Issues**
```bash
# Check Neo4j health
docker exec subgraphrag_neo4j wget -O - http://localhost:7474 2>/dev/null

# Access Neo4j shell
docker exec -it subgraphrag_neo4j cypher-shell -u neo4j -p password

# Reset Neo4j data
docker-compose down
docker volume rm subgraphrag_neo4j_data
docker-compose up -d neo4j
```

#### Alternative Neo4j Setup Options

If Docker continues to cause issues, you can use these alternatives:

**Option 1: Use Local Neo4j Installation**
```bash
# Setup with local Neo4j
./bin/setup_dev.sh --use-local-neo4j

# Install Neo4j locally
# macOS
brew install neo4j
brew services start neo4j

# Ubuntu/Debian
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 4.4' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j
sudo systemctl start neo4j
```

**Option 2: Skip Neo4j for Initial Development**
```bash
# Setup without Neo4j (limited functionality)
./bin/setup_dev.sh --skip-neo4j

# You can add Neo4j later once Docker issues are resolved
```

**Option 3: Use Neo4j Aura (Cloud)**
```bash
# Create free account at https://neo4j.com/aura/
# Update .env with your cloud credentials:
# NEO4J_URI=neo4j+s://your-instance.neo4j.io
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your-password

./bin/setup_dev.sh --skip-neo4j
```

#### Docker Cleanup Commands

```bash
# Remove all containers and volumes
docker-compose down -v

# Clean up Docker system
docker system prune -a

# Remove specific volumes
docker volume ls
docker volume rm subgraphrag_neo4j_data

# Force remove containers
docker rm -f subgraphrag_neo4j subgraphrag_api

# Rebuild images
docker-compose build --no-cache
```

#### Verification Steps

After resolving Docker issues, verify your setup:

```bash
# Check Docker is running
docker info

# Check containers are running
docker ps

# Test Neo4j connection
curl http://localhost:7474

# Test API connection
curl http://localhost:8000/health

# Run setup again if needed
./bin/setup_dev.sh
```

### Local Development (without Docker)

```bash
# Install Neo4j locally
./bin/install_neo4j.sh

# Setup Python environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configure environment
cp .env.example .env
# Edit .env with local settings

# Start development server
python -m uvicorn src.app.api:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
SubgraphRAGPlus/
├── src/app/                    # Core application code
│   ├── api.py                 # FastAPI application and routes
│   ├── models.py              # Data models and schemas
│   ├── config.py              # Configuration management
│   ├── database.py            # Database connections and utilities
│   └── retriever.py           # Hybrid retrieval implementation
├── bin/                       # Executable scripts
│   ├── start.sh              # Main startup script
│   ├── setup_dev.sh          # Development environment setup
│   ├── run_tests.sh          # Test runner
│   └── run_benchmark.sh      # Performance benchmarking
├── scripts/                   # Data processing and utilities
│   ├── stage_ingest.py       # Data staging for ingestion
│   ├── ingest_worker.py      # Background ingestion worker
│   └── merge_faiss.py        # FAISS index management
├── tests/                     # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── e2e/                  # End-to-end tests
├── config/                    # Configuration files
├── data/                      # Data storage
├── models/                    # ML model files
└── docs/                      # Documentation
```

## Development Workflows

### Code Style and Formatting

```bash
# Format code
make format
# OR
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Lint code
make lint
# OR
flake8 src/ tests/ scripts/
pylint src/ tests/ scripts/

# Type checking
make typecheck
# OR
mypy src/ tests/ scripts/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add new feature"

# Push and create PR
git push origin feature/your-feature-name
```

### Commit Message Convention

Follow conventional commits:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` formatting changes
- `refactor:` code refactoring
- `test:` adding tests
- `chore:` maintenance tasks

## Testing

### Running Tests

```bash
# Run all tests
make test
# OR
./bin/run_tests.sh

# Run specific test categories
make test-unit
make test-integration
make test-e2e

# Run with coverage
make test-coverage

# Run specific test file
pytest tests/unit/test_retriever.py -v

# Run with debugging
pytest tests/unit/test_retriever.py -v -s --pdb
```

### Test Structure

#### Unit Tests

Test individual components in isolation:

```python
# tests/unit/test_retriever.py
import pytest
from unittest.mock import Mock, patch
from src.app.retriever import HybridRetriever

class TestHybridRetriever:
    @pytest.fixture
    def mock_neo4j_driver(self):
        return Mock()
    
    @pytest.fixture
    def mock_faiss_index(self):
        return Mock()
    
    @pytest.fixture
    def retriever(self, mock_neo4j_driver, mock_faiss_index):
        with patch('src.app.retriever.neo4j.GraphDatabase.driver', return_value=mock_neo4j_driver):
            with patch('src.app.retriever.faiss.read_index', return_value=mock_faiss_index):
                return HybridRetriever()
    
    def test_retrieve_candidates(self, retriever):
        # Test candidate retrieval logic
        candidates = retriever.retrieve_candidates("test query")
        assert isinstance(candidates, list)
```

#### Integration Tests

Test component interactions:

```python
# tests/integration/test_api_database.py
import pytest
from fastapi.testclient import TestClient
from src.app.api import app

class TestAPIDatabase:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_query_endpoint_with_database(self, client):
        response = client.post(
            "/query",
            json={"question": "What is machine learning?"},
            headers={"X-API-KEY": "test-key"}
        )
        assert response.status_code == 200
```

#### End-to-End Tests

Test complete user workflows:

```python
# tests/e2e/test_complete_workflow.py
import pytest
import requests

class TestCompleteWorkflow:
    def test_ingest_and_query_workflow(self):
        # Test complete data ingestion and querying
        base_url = "http://localhost:8000"
        
        # Ingest data
        ingest_response = requests.post(
            f"{base_url}/ingest",
            json={"triples": [["AI", "is", "artificial intelligence"]]},
            headers={"X-API-KEY": "test-key"}
        )
        assert ingest_response.status_code == 200
        
        # Query data
        query_response = requests.post(
            f"{base_url}/query",
            json={"question": "What is AI?"},
            headers={"X-API-KEY": "test-key"}
        )
        assert query_response.status_code == 200
```

### Test Data Management

```bash
# Create test database
make test-db-setup

# Load test data
python scripts/load_test_data.py

# Clean test data
make test-db-clean
```

### Performance Testing

```bash
# Run benchmarks
make benchmark
# OR
./bin/run_benchmark.sh

# Load testing
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## Debugging

### Local Debugging

```python
# Add breakpoints in code
import pdb; pdb.set_trace()

# Or use ipdb for better interface
import ipdb; ipdb.set_trace()

# Debug with pytest
pytest tests/unit/test_retriever.py -v -s --pdb
```

### Docker Debugging

```bash
# Access running container
docker exec -it subgraphrag_api bash

# View container logs
docker logs subgraphrag_api -f

# Debug with remote debugger
# Add to your code:
import debugpy
debugpy.listen(("0.0.0.0", 5678))
debugpy.wait_for_client()
```

### Database Debugging

```bash
# Access Neo4j browser
open http://localhost:7474

# Run Cypher queries
docker exec -it subgraphrag_neo4j cypher-shell -u neo4j -p password

# Check database status
python -c "from src.app.database import test_connection; test_connection()"
```

## Development Tools

### Useful Make Commands

```bash
make help              # Show all available commands
make dev-setup         # Setup development environment
make dev-start         # Start development server
make test              # Run all tests
make lint              # Run linting
make format            # Format code
make clean             # Clean temporary files
make docker-build      # Build Docker images
make docker-dev        # Start development with Docker
```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm

1. Set interpreter to `./venv/bin/python`
2. Enable Black formatter
3. Configure pytest as test runner
4. Set up remote debugging for Docker

### Environment Variables for Development

```bash
# .env.development
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
API_KEY_SECRET=dev-secret-key
MODEL_BACKEND=mock
ENABLE_CORS=true
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
```

## Contributing

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch** from `main`
3. **Make your changes** following code style guidelines
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run the test suite** and ensure all tests pass
7. **Submit a pull request** with a clear description

### Code Review Guidelines

- **Code Quality**: Follow PEP 8 and project conventions
- **Testing**: Ensure adequate test coverage (>80%)
- **Documentation**: Update docstrings and README if needed
- **Performance**: Consider performance implications
- **Security**: Review for security vulnerabilities

### Issue Reporting

When reporting issues, include:
- **Environment details** (OS, Python version, Docker version)
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Error messages and logs**
- **Configuration details** (sanitized)

### Feature Requests

For new features:
- **Use case description** and motivation
- **Proposed implementation** approach
- **Breaking changes** if any
- **Documentation requirements**

## Advanced Development

### Custom Model Integration

```python
# src/app/models/custom_model.py
from abc import ABC, abstractmethod

class CustomLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

class MyCustomModel(CustomLLM):
    def generate(self, prompt: str, **kwargs) -> str:
        # Your custom implementation
        return "Generated response"

# Register in config
CUSTOM_MODELS = {
    "my_model": MyCustomModel
}
```

### Plugin Development

```python
# src/app/plugins/example_plugin.py
from src.app.plugins.base import BasePlugin

class ExamplePlugin(BasePlugin):
    def __init__(self):
        self.name = "example"
    
    def process_query(self, query: str) -> str:
        # Custom query processing
        return f"Processed: {query}"
    
    def process_response(self, response: str) -> str:
        # Custom response processing
        return f"Enhanced: {response}"
```

### Performance Profiling

```python
# Profile code execution
import cProfile
import pstats

def profile_function():
    # Your code here
    pass

cProfile.run('profile_function()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler scripts/ingest_worker.py

# Line-by-line profiling
@profile
def memory_intensive_function():
    # Your code here
    pass
```

## 🔧 Troubleshooting

### Quick Diagnostic Reference

| Issue Type | Diagnostic Command | Common Fix |
|------------|-------------------|------------|
| Docker Problems | `docker ps \| grep subgraphrag` | `make docker-restart` |
| Neo4j Connection | `curl http://localhost:7474` | Check `.env` credentials |
| API Issues | `curl -i http://localhost:8000/healthz` | View logs in `logs/app.log` |
| Model Problems | `ls -la models/` | `python scripts/download_models.py` |

### Common Development Issues

#### Environment Setup Problems

**Virtual Environment Issues**
```bash
# Clean environment setup
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

**Package Installation Failures**
```bash
# Update package managers
pip install --upgrade pip setuptools wheel

# Install with verbose output for debugging
pip install -v -r requirements-dev.txt

# For macOS with M1/M2 chips
export ARCHFLAGS="-arch arm64"
pip install -r requirements-dev.txt
```

#### Docker Development Issues

**Containers Won't Start**
```bash
# Check Docker daemon
docker info

# Check port conflicts
sudo lsof -i :8000 :7474 :7687

# Reset Docker environment
make docker-stop
docker system prune -f
make docker-start
```

**Container Performance Issues**
```bash
# Monitor resource usage
docker stats

# Increase Docker resources (Docker Desktop)
# Settings > Resources > Advanced
# RAM: 4GB+, CPU: 2+ cores
```

#### Neo4j Development Issues

**Connection Problems**
```bash
# Test Neo4j connectivity
curl http://localhost:7474

# Check Neo4j logs
docker logs subgraphrag_neo4j

# Reset Neo4j data
make neo4j-restart
```

**Authentication Issues**
```bash
# Verify credentials in .env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Test connection with cypher-shell
echo "RETURN 1;" | cypher-shell -u neo4j -p password
```

#### API Development Issues

**Server Won't Start**
```bash
# Check for port conflicts
sudo lsof -i :8000

# Start with verbose logging
LOG_LEVEL=DEBUG python src/main.py

# Check application logs
tail -f logs/app.log
```

**API Authentication Problems**
```bash
# Test API key
curl -H "X-API-KEY: changeme" http://localhost:8000/healthz

# Verify environment variable
echo $API_KEY_SECRET
```

#### Testing Issues

**Test Failures**
```bash
# Run tests with verbose output
pytest -v -s tests/

# Run specific failing test
pytest -v tests/test_specific.py::test_function

# Check test database
sqlite3 data/test_staging.db ".tables"
```

**Coverage Issues**
```bash
# Generate detailed coverage report
pytest --cov=src --cov-report=html tests/
open htmlcov/index.html
```

### Performance Debugging

#### Slow Query Performance
```bash
# Profile Neo4j queries
# In Neo4j Browser:
PROFILE MATCH (n)-[r]-(m) RETURN n,r,m LIMIT 100

# Check FAISS index
python -c "
import faiss
index = faiss.read_index('data/faiss/index.bin')
print(f'Index size: {index.ntotal}')
"
```

#### Memory Issues
```bash
# Monitor memory usage
python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / 1024**3:.1f}GB')
"

# Profile memory usage
pip install memory-profiler
python -m memory_profiler src/main.py
```

### Data Issues

#### Ingestion Problems
```bash
# Check ingestion queue
python scripts/stage_ingest.py --status

# Process queue manually
python scripts/ingest_worker.py --process-all

# Rebuild FAISS index
python scripts/merge_faiss.py --rebuild
```

#### Data Consistency Issues
```bash
# Check for duplicate entities
python scripts/reconcile_stores.py --check

# Validate data integrity
python scripts/validate_data.py --full-check
```

### Error Message Reference

| Error Pattern | Likely Cause | Solution |
|---------------|--------------|----------|
| `Connection refused` | Service not running | Start required service |
| `No module named 'X'` | Missing dependency | `pip install X` |
| `FAISS index not found` | Missing index file | `python scripts/merge_faiss.py --init` |
| `API key not valid` | Authentication error | Check X-API-KEY header |
| `Neo4j.exceptions.ServiceUnavailable` | Neo4j connection issue | Verify Neo4j is running |
| `sqlite3.OperationalError` | Database lock/corruption | Restart application |

### Getting Help

#### Self-Diagnosis Steps
1. **Check logs**: `tail -f logs/app.log`
2. **Verify services**: `make healthcheck`
3. **Test connectivity**: Run diagnostic commands above
4. **Check configuration**: Verify `.env` and `config/config.json`

#### Reporting Issues
When reporting bugs, include:
```bash
# System information
python --version
docker --version
uname -a

# Application state
make healthcheck
docker ps
ls -la data/ models/ logs/

# Error logs
tail -50 logs/app.log
```

#### Community Support
- **GitHub Issues**: [Report bugs and request features](https://github.com/clarkandrew/SubgraphRAGPlus/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/clarkandrew/SubgraphRAGPlus/discussions)
- **Documentation**: Check other guides in `/docs` directory

This development guide provides comprehensive information for contributing to and extending SubgraphRAG+ while maintaining code quality and following best practices. 