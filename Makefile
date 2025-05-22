# Makefile for SubgraphRAG+

# Variables
PYTHON = python3
PIP = pip
DOCKER = docker
DOCKER_COMPOSE = docker compose

# Neo4j Container Setup
NEO4J_PASSWORD ?= password

# Directories
DATA_DIR = ./data
CACHE_DIR = ./cache
MODELS_DIR = ./models
LOGS_DIR = ./logs

# Install dependencies
.PHONY: setup-dev
setup-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements-dev.txt
	@echo "Creating necessary directories..."
	mkdir -p $(DATA_DIR) $(CACHE_DIR) $(MODELS_DIR) $(LOGS_DIR)
	@echo "Development setup complete!"

# Install production dependencies
.PHONY: setup-prod
setup-prod:
	@echo "Installing production dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Creating necessary directories..."
	mkdir -p $(DATA_DIR) $(CACHE_DIR) $(MODELS_DIR) $(LOGS_DIR)
	@echo "Production setup complete!"

# Run linting
.PHONY: lint
lint:
	@echo "Running linting checks..."
	flake8 app/ tests/ scripts/
	black --check app/ tests/ scripts/
	isort --check app/ tests/ scripts/
	@echo "Linting complete!"

# Run tests
.PHONY: test
test:
	@echo "Running tests..."
	@mkdir -p logs
	@touch logs/app.log
	@if [ ! -d "venv" ]; then \
		echo "Virtual environment not found. Creating..."; \
		python3 -m venv venv; \
		. venv/bin/activate && pip install -r requirements-dev.txt; \
	fi
	@echo "Checking if Neo4j is running..."
	@if ! docker ps | grep -q neo4j; then \
		echo "Neo4j is not running. Starting Neo4j..."; \
		$(MAKE) neo4j-start; \
		echo "Waiting for Neo4j to initialize..."; \
		sleep 10; \
	fi
	@echo "Running tests now..."
	@. venv/bin/activate && pytest tests/ -v --disable-warnings || { \
		echo "Tests failed. Please check the error messages above."; \
		exit 1; \
	}
	@echo "Tests complete!"

# Run server in development mode
.PHONY: serve
serve:
	@echo "Starting SubgraphRAG+ API server..."
	@mkdir -p logs
	@if [ ! -d "venv" ]; then \
		echo "Virtual environment not found. Creating..."; \
		python3 -m venv venv; \
		. venv/bin/activate && pip install -r requirements-dev.txt; \
	fi
	@echo "Checking if Neo4j is running..."
	@if ! docker ps | grep -q neo4j; then \
		echo "Neo4j is not running. Starting Neo4j..."; \
		$(MAKE) neo4j-start; \
		echo "Waiting for Neo4j to initialize..."; \
		sleep 10; \
	fi
	@echo "Starting server now..."
	@. venv/bin/activate && $(PYTHON) main.py --reload
	
# Run server in production mode
.PHONY: serve-prod
serve-prod:
	@echo "Starting SubgraphRAG+ API server in production mode..."
	@mkdir -p logs
	@if [ ! -d "venv" ]; then \
		echo "Virtual environment not found. Creating..."; \
		python3 -m venv venv; \
		. venv/bin/activate && pip install -r requirements.txt; \
	fi
	@echo "Checking if Neo4j is running..."
	@if ! docker ps | grep -q neo4j; then \
		echo "Neo4j is not running. Starting Neo4j..."; \
		$(MAKE) neo4j-start; \
		echo "Waiting for Neo4j to initialize..."; \
		sleep 10; \
	fi
	@echo "Starting production server now..."
	@. venv/bin/activate && uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 4

# Start Neo4j container
.PHONY: neo4j-start
neo4j-start:
	@echo "Starting Neo4j container..."
	$(DOCKER) run --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/$(NEO4J_PASSWORD) -e NEO4J_PLUGINS='["apoc"]' -v $(PWD)/$(DATA_DIR)/neo4j:/data -d neo4j:4.4

# Stop Neo4j container
.PHONY: neo4j-stop
neo4j-stop:
	@echo "Stopping Neo4j container..."
	$(DOCKER) stop neo4j
	$(DOCKER) rm neo4j

# Download pre-trained MLP model
.PHONY: get-pretrained-mlp
get-pretrained-mlp:
	@echo "Downloading pre-trained MLP model..."
	mkdir -p $(MODELS_DIR)
	@if [ -f $(MODELS_DIR)/mlp_pretrained.pt ]; then \
		echo "Model already exists"; \
	else \
		$(PYTHON) scripts/download_pretrained_mlp.py; \
	fi

# Ingest sample data
.PHONY: ingest-sample
ingest-sample:
	@echo "Ingesting sample data..."
	$(PYTHON) scripts/stage_ingest.py --file data/sample_data/sample_triples.csv
	$(PYTHON) scripts/ingest_worker.py

# Run demo with quickstart
.PHONY: demo_quickstart
demo_quickstart:
	@echo "Running SubgraphRAG+ demo quickstart..."
	$(PYTHON) scripts/demo_quickstart.py

# Rebuild FAISS index
.PHONY: rebuild-faiss-index
rebuild-faiss-index:
	@echo "Rebuilding FAISS index..."
	$(PYTHON) scripts/rebuild_faiss_index.py

# Run benchmarks
.PHONY: benchmark
benchmark:
	@echo "Running benchmark tests..."
	$(PYTHON) scripts/benchmark.py

# Clean cache and temporary files
.PHONY: clean
clean:
	@echo "Cleaning cache and temporary files..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf $(CACHE_DIR)/*
	@echo "Clean complete!"

# Full reset - USE WITH CAUTION
.PHONY: reset
reset:
	@echo "WARNING: This will reset all data and caches. Press Enter to continue or Ctrl+C to cancel."
	@read dummy
	rm -rf $(CACHE_DIR)/*
	rm -rf $(DATA_DIR)/staging.db
	rm -rf $(DATA_DIR)/faiss_staging/*
	@echo "Reset complete!"

# Create basic migrations
.PHONY: migrate-schema
migrate-schema:
	@echo "Running Neo4j schema migrations..."
	$(PYTHON) scripts/migrate_schema.py --target-version kg_v1

# Check API health
.PHONY: health-check
health-check:
	@echo "Checking API health..."
	curl -s http://localhost:8000/healthz | jq

# Check API readiness
.PHONY: ready-check
ready-check:
	@echo "Checking API readiness..."
	curl -s http://localhost:8000/readyz | jq

# Generate API documentation
.PHONY: generate-api-docs
generate-api-docs:
	@echo "Generating API documentation..."
	$(PYTHON) scripts/generate_openapi.py

# Start Docker Compose stack
.PHONY: docker-start
docker-start:
	@echo "Starting Docker Compose stack..."
	$(DOCKER_COMPOSE) up -d

# Stop Docker Compose stack
.PHONY: docker-stop
docker-stop:
	@echo "Stopping Docker Compose stack..."
	$(DOCKER_COMPOSE) down

# Help command
.PHONY: help
help:
	@echo "SubgraphRAG+ Makefile Commands:"
	@echo "  setup-dev         : Install development dependencies"
	@echo "  setup-prod        : Install production dependencies"
	@echo "  lint              : Run linting checks"
	@echo "  test              : Run tests"
	@echo "  serve             : Start server in development mode"
	@echo "  serve-prod        : Start server in production mode"
	@echo "  neo4j-start       : Start Neo4j container"
	@echo "  neo4j-stop        : Stop Neo4j container"
	@echo "  get-pretrained-mlp: Download pre-trained MLP model"
	@echo "  ingest-sample     : Ingest sample data"
	@echo "  demo_quickstart   : Run demo quickstart"
	@echo "  rebuild-faiss-index: Rebuild FAISS index"
	@echo "  benchmark         : Run benchmarks"
	@echo "  clean             : Clean cache and temporary files"
	@echo "  reset             : Reset all data (use with caution)"
	@echo "  migrate-schema    : Run Neo4j schema migrations"
	@echo "  health-check      : Check API health"
	@echo "  ready-check       : Check API readiness"
	@echo "  generate-api-docs : Generate API documentation"
	@echo "  docker-start      : Start Docker Compose stack"
	@echo "  docker-stop       : Stop Docker Compose stack"
	@echo "  help              : Show this help message"

# Complete setup - one command to set up everything
.PHONY: setup-all
setup-all:
	@echo "Starting complete setup of SubgraphRAG+..."
	@echo "Step 1: Installing dependencies..."
	$(MAKE) setup-dev
	@echo "Step 2: Starting Neo4j database..."
	$(MAKE) neo4j-start
	@echo "Step 3: Downloading pre-trained models..."
	$(MAKE) get-pretrained-mlp
	@echo "Step 4: Initializing database schema..."
	$(MAKE) migrate-schema
	@echo "Step 5: Loading sample data..."
	$(MAKE) ingest-sample
	@echo "Step 6: Running tests to verify setup..."
	$(MAKE) test
	@echo "================================================================="
	@echo "✅ Setup complete! You can now run 'make serve' to start the API."
	@echo "================================================================="

# Shortcut for setup-all (to make it more discoverable)
.PHONY: start
start: setup-all

# Default target
.PHONY: default
default: help