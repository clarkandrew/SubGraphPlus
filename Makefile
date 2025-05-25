# SubgraphRAG+ Makefile
# Production-ready build and development commands

# ============================================================================
# Configuration
# ============================================================================

# Python and package management
PYTHON := python3
PIP := pip3
VENV := venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

# Docker configuration
DOCKER := docker
DOCKER_COMPOSE := docker-compose
COMPOSE_FILE := docker-compose.yml
COMPOSE_PROD_FILE := deployment/docker-compose.prod.yml

# Project directories
SRC_DIR := src
TESTS_DIR := tests
DOCS_DIR := docs
CONFIG_DIR := config
DATA_DIR := data
CACHE_DIR := cache
LOGS_DIR := logs

# Application settings
APP_MODULE := app.api:app
HOST := 0.0.0.0
PORT := 8000
WORKERS := 4

# ============================================================================
# Help and Default Target
# ============================================================================

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "🚀 SubgraphRAG+ Development Commands"
	@echo ""
	@echo "📦 Setup Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep "Setup" | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🔧 Development Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep "Development" | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🐳 Docker Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep "Docker" | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🗄️ Database Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep "Database" | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🧪 Testing & Quality Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep "Testing\|Quality" | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🛠️ Utility Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep "Utility" | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Setup Commands
# ============================================================================

.PHONY: setup-all
setup-all: ## Setup - Complete Docker setup (recommended)
	@echo "🚀 Setting up SubgraphRAG+ with Docker..."
	@./bin/setup_docker.sh
	@echo "✅ Setup complete! Access: http://localhost:8000/docs"

.PHONY: setup-dev
setup-dev: ## Setup - Development environment (uses interactive script)
	@echo "🔧 Running development environment setup..."
	@./bin/setup_dev.sh

.PHONY: setup-dev-quick
setup-dev-quick: ## Setup - Quick development setup (non-interactive)
	@echo "🔧 Running quick development setup..."
	@./bin/setup_dev.sh --skip-tests --skip-sample-data

.PHONY: setup-prod
setup-prod: venv install-prod ## Setup - Production environment
	@echo "🏢 Production environment ready!"

.PHONY: venv
venv: ## Setup - Create Python virtual environment (manual method)
	@echo "⚠️  Note: Consider using 'make setup-dev' for full setup instead"
	@if [ ! -d "$(VENV)" ]; then \
		echo "📦 Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
		$(VENV_PIP) install --upgrade pip setuptools wheel; \
	fi

.PHONY: install-dev
install-dev: venv ## Setup - Install development dependencies (manual method)
	@echo "⚠️  Note: Consider using 'make setup-dev' for full setup instead"
	@echo "📦 Installing development dependencies..."
	@$(VENV_PIP) install -r requirements-dev.txt
	@$(VENV_PIP) install -e .

.PHONY: install-prod
install-prod: venv ## Setup - Install production dependencies (manual method)
	@echo "⚠️  Note: Consider using 'make setup-dev' for full setup instead"
	@echo "📦 Installing production dependencies..."
	@$(VENV_PIP) install -r requirements.txt
	@$(VENV_PIP) install -e .

# ============================================================================
# Development Commands
# ============================================================================

.PHONY: serve
serve: ## Development - Start development server
	@echo "🚀 Starting development server..."
	@$(VENV_PYTHON) $(SRC_DIR)/main.py --reload --host $(HOST) --port $(PORT)

.PHONY: serve-prod
serve-prod: ## Development - Start production server
	@echo "🏢 Starting production server..."
	@$(VENV_PYTHON) -m uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT) --workers $(WORKERS)

.PHONY: shell
shell: ## Development - Start Python shell with app context
	@echo "🐍 Starting Python shell..."
	@$(VENV_PYTHON) -c "from app.api import app; import IPython; IPython.embed()"

# ============================================================================
# Docker Commands
# ============================================================================

.PHONY: docker-build
docker-build: ## Docker - Build all Docker images
	@echo "🐳 Building Docker images..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) build

.PHONY: docker-start
docker-start: ## Docker - Start all services
	@echo "🐳 Starting Docker services..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d
	@echo "✅ Services started! API: http://localhost:8000"

.PHONY: docker-stop
docker-stop: ## Docker - Stop all services
	@echo "🛑 Stopping Docker services..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down

.PHONY: docker-restart
docker-restart: docker-stop docker-start ## Docker - Restart all services

.PHONY: docker-logs
docker-logs: ## Docker - View service logs
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f

.PHONY: docker-clean
docker-clean: ## Docker - Clean up containers and volumes
	@echo "🧹 Cleaning Docker resources..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down -v
	@$(DOCKER) system prune -f

# ============================================================================
# Database Commands
# ============================================================================

.PHONY: neo4j-start
neo4j-start: ## Database - Start Neo4j container
	@echo "🗄️ Starting Neo4j..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d neo4j
	@echo "⏳ Waiting for Neo4j to be ready..."
	@sleep 30
	@echo "✅ Neo4j ready! Browser: http://localhost:7474"

.PHONY: neo4j-stop
neo4j-stop: ## Database - Stop Neo4j container
	@echo "🛑 Stopping Neo4j..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) stop neo4j

.PHONY: neo4j-restart
neo4j-restart: neo4j-stop neo4j-start ## Database - Restart Neo4j

.PHONY: migrate-schema
migrate-schema: ## Database - Apply schema migrations
	@echo "🔄 Applying database migrations..."
	@$(VENV_PYTHON) scripts/migrate_schema.py

.PHONY: ingest-sample
ingest-sample: ## Database - Load sample data
	@echo "📊 Loading sample data..."
	@$(VENV_PYTHON) scripts/stage_ingest.py --sample
	@$(VENV_PYTHON) scripts/ingest_worker.py --process-all

# ============================================================================
# Testing & Quality Commands
# ============================================================================

.PHONY: test
test: ## Testing - Run test suite
	@echo "🧪 Running tests..."
	@$(VENV_PYTHON) -m pytest $(TESTS_DIR)/ -v

.PHONY: test-coverage
test-coverage: ## Testing - Run tests with coverage
	@echo "🧪 Running tests with coverage..."
	@$(VENV_PYTHON) -m pytest $(TESTS_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=term

.PHONY: test-integration
test-integration: ## Testing - Run integration tests
	@echo "🧪 Running integration tests..."
	@$(VENV_PYTHON) -m pytest $(TESTS_DIR)/integration/ -v

.PHONY: lint
lint: ## Quality - Run code linting
	@echo "🔍 Running linting..."
	@$(VENV_PYTHON) -m flake8 $(SRC_DIR)/ $(TESTS_DIR)/ scripts/
	@$(VENV_PYTHON) -m pylint $(SRC_DIR)/ --disable=C0114,C0115,C0116

.PHONY: format
format: ## Quality - Format code
	@echo "✨ Formatting code..."
	@$(VENV_PYTHON) -m black $(SRC_DIR)/ $(TESTS_DIR)/ scripts/
	@$(VENV_PYTHON) -m isort $(SRC_DIR)/ $(TESTS_DIR)/ scripts/

.PHONY: typecheck
typecheck: ## Quality - Run type checking
	@echo "🔍 Running type checks..."
	@$(VENV_PYTHON) -m mypy $(SRC_DIR)/ --ignore-missing-imports

.PHONY: security
security: ## Quality - Run security checks
	@echo "🔒 Running security checks..."
	@$(VENV_PYTHON) -m bandit -r $(SRC_DIR)/ -f json

.PHONY: quality
quality: lint typecheck security ## Quality - Run all quality checks

# ============================================================================
# Utility Commands
# ============================================================================

.PHONY: health
health: ## Utility - Check API health
	@echo "🏥 Checking API health..."
	@curl -s http://localhost:8000/health | jq '.' || echo "❌ API not responding"

.PHONY: docs-serve
docs-serve: ## Utility - Serve documentation locally
	@echo "📚 Serving documentation..."
	@cd $(DOCS_DIR) && $(PYTHON) -m http.server 8080

.PHONY: clean
clean: ## Utility - Clean temporary files
	@echo "🧹 Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	@rm -rf $(CACHE_DIR)/* $(LOGS_DIR)/*
	@echo "✅ Clean complete!"

.PHONY: reset
reset: ## Utility - Reset all data (⚠️ DESTRUCTIVE)
	@echo "⚠️  WARNING: This will delete all data!"
	@echo "Press Enter to continue or Ctrl+C to cancel..."
	@read dummy
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down -v
	@rm -rf $(DATA_DIR)/faiss $(DATA_DIR)/neo4j $(DATA_DIR)/staging.db
	@rm -rf $(CACHE_DIR)/* $(LOGS_DIR)/*
	@echo "🔄 Reset complete!"

.PHONY: backup
backup: ## Utility - Create data backup
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@tar -czf backups/backup_$(shell date +%Y%m%d_%H%M%S).tar.gz $(DATA_DIR)/ $(CONFIG_DIR)/
	@echo "✅ Backup created in backups/"

.PHONY: env-check
env-check: ## Utility - Check environment setup
	@echo "🔍 Environment Check:"
	@echo "Python: $(shell $(PYTHON) --version 2>/dev/null || echo 'Not found')"
	@echo "Docker: $(shell $(DOCKER) --version 2>/dev/null || echo 'Not found')"
	@echo "Docker Compose: $(shell $(DOCKER_COMPOSE) --version 2>/dev/null || echo 'Not found')"
	@echo "Virtual Environment: $(shell [ -d $(VENV) ] && echo 'Present' || echo 'Missing')"

# ============================================================================
# Production Deployment
# ============================================================================

.PHONY: deploy-prod
deploy-prod: ## Production - Deploy to production
	@echo "🚀 Deploying to production..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_PROD_FILE) up -d --build
	@echo "✅ Production deployment complete!"

.PHONY: deploy-stop
deploy-stop: ## Production - Stop production deployment
	@echo "🛑 Stopping production deployment..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_PROD_FILE) down

# ============================================================================
# Development Workflow Shortcuts
# ============================================================================

.PHONY: demo_quickstart
demo_quickstart: ## Workflow - Complete demo setup and test query
	@echo "🚀 Running demo quickstart..."
	@$(VENV_PYTHON) examples/demo_quickstart.py

.PHONY: get-pretrained-mlp
get-pretrained-mlp: ## Setup - Download or create pre-trained MLP model
	@echo "📥 Setting up pre-trained MLP model..."
	@if [ ! -f "models/mlp_pretrained.pt" ]; then \
		echo "Pre-trained MLP model not found. Creating placeholder..."; \
		$(VENV_PYTHON) -c "import torch; import torch.nn as nn; \
		class SimpleMLP(nn.Module): \
			def __init__(self): \
				super().__init__(); \
				self.pred = nn.Sequential(nn.Linear(4116, 1024), nn.ReLU(), nn.Linear(1024, 1)); \
			def forward(self, x): return self.pred(x); \
		torch.save(SimpleMLP(), 'models/mlp_pretrained.pt')"; \
		echo "✅ Placeholder MLP model created at models/mlp_pretrained.pt"; \
		echo "ℹ️  For production use, train a real model using the SubgraphRAG Colab notebook"; \
	else \
		echo "✅ Pre-trained MLP model already exists"; \
	fi

.PHONY: dev
dev: setup-dev neo4j-start migrate-schema serve ## Workflow - Complete development setup and start

.PHONY: quick-start
quick-start: docker-start ## Workflow - Quick start with Docker

.PHONY: full-test
full-test: quality test test-integration ## Workflow - Run all tests and quality checks

# ============================================================================
# Maintenance Commands
# ============================================================================

.PHONY: update-deps
update-deps: ## Maintenance - Update Python dependencies
	@echo "📦 Updating dependencies..."
	@$(VENV_PIP) install --upgrade pip setuptools wheel
	@$(VENV_PIP) install --upgrade -r requirements-dev.txt

.PHONY: rebuild-index
rebuild-index: ## Maintenance - Rebuild FAISS index
	@echo "🔄 Rebuilding FAISS index..."
	@$(VENV_PYTHON) scripts/merge_faiss.py

# ============================================================================
# Special Targets
# ============================================================================

# Ensure directories exist
$(DATA_DIR) $(CACHE_DIR) $(LOGS_DIR):
	@mkdir -p $@

# Clean up on exit
.PHONY: cleanup
cleanup:
	@echo "🧹 Performing cleanup..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down 2>/dev/null || true