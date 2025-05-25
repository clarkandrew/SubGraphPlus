# 🔧 Troubleshooting Guide

This comprehensive guide helps you diagnose and resolve common issues with SubgraphRAG+.

## 🚨 Quick Diagnostics

### Health Check Commands

```bash
# Basic system health
curl http://localhost:8000/healthz

# Detailed component status (includes dependencies)
curl http://localhost:8000/readyz

# Check if Docker services are running
docker ps
docker-compose ps

# View recent logs
docker-compose logs --tail=50

# Quick API test
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "visualize_graph": false}'
```

### Common Status Indicators

| Status | Meaning | Action |
|--------|---------|--------|
| ✅ `"status": "ok"` | Component working | No action needed |
| ⚠️ `"status": "degraded"` | Partial functionality | Check specific component |
| ❌ `"status": "failed"` | Component failed | Follow troubleshooting steps |

---

## 🛠️ Setup & Installation Issues

### Setup Script Hangs or Times Out

**Symptoms:**
- Setup script hangs at "Starting ingest worker" or model download
- Process appears frozen for >10 minutes
- Memory usage spikes during setup

**Root Causes:**
1. **Large model download**: The embedding model (434MB+) takes time to download
2. **Model loading timeout**: Model initialization can take several minutes
3. **Memory constraints**: Insufficient RAM for model loading
4. **Configuration conflicts**: Mismatched settings between `.env` and `config.json`

**Solutions:**

#### Option 1: Use Setup Script Timeout (Recommended)
```bash
# Setup with built-in timeout protection
./bin/setup_dev.sh --skip-tests

# If it times out, manually complete:
source venv/bin/activate
python scripts/ingest_worker.py --process-all
```

#### Option 2: Test Components Individually
```bash
# Test embedder first
source venv/bin/activate
python scripts/test_embedder.py

# Test database connections
python -c "from src.app.database import neo4j_db, sqlite_db; print('Neo4j:', neo4j_db.verify_connectivity()); print('SQLite: OK')"
```

#### Option 3: Skip Heavy Components Initially
```bash
# Minimal setup first
./bin/setup_dev.sh --skip-sample-data --skip-tests

# Add components later
make migrate-schema
make ingest-sample
```

#### Option 4: Use Lighter Models
Edit `.env` to use smaller models temporarily:
```bash
# For faster setup, use smaller embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Or use OpenAI backend (no local models)
MODEL_BACKEND=openai
OPENAI_API_KEY=your_key_here
```

### Configuration Conflicts

**Symptoms:**
- Setup logs show conflicting backend configurations
- Services start but API calls fail
- Model loading errors

**Solutions:**
```bash
# Ensure consistency between files
# Check .env file:
cat .env | grep MODEL_BACKEND

# Check config.json:
cat config/config.json | jq '.MODEL_BACKEND'

# They should match. If not, update one:
# Edit .env or use:
echo 'MODEL_BACKEND=hf' >> .env
```

### Model Download Issues

**Symptoms:**
- Slow or failed model downloads
- Timeouts during HuggingFace model loading
- Disk space errors

**Solutions:**
```bash
# Pre-download models manually
pip install huggingface_hub
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2

# Set custom cache directory
export HF_HOME=/path/to/large/disk/cache
export TRANSFORMERS_CACHE=$HF_HOME

# Check available disk space
df -h

# Clean old models if needed
rm -rf ~/.cache/huggingface
```

### Memory Issues During Setup

**Symptoms:**
- System becomes unresponsive
- "Killed" process messages
- Out of memory errors

**Solutions:**
```bash
# Check system memory
free -h

# Close unnecessary applications
# Use swap file if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Use lighter backends:
# MLX (Apple Silicon - more memory efficient)
USE_MLX_LLM=true
MODEL_BACKEND=mlx

# Or OpenAI (no local models)
MODEL_BACKEND=openai
OPENAI_API_KEY=your_key
```

---

## 🐳 Docker Issues

### Docker Daemon Not Running

**Symptoms:**
- `Cannot connect to the Docker daemon at unix:///var/run/docker.sock`
- `docker: command not found`

**Solutions:**

#### macOS
```bash
# Start Docker Desktop
open -a Docker

# Wait for Docker to start (check menu bar icon)
# Verify Docker is running
docker info
```

#### Linux
```bash
# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (if not already done)
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker is running
docker info
```

#### Windows (WSL2)
```bash
# Start Docker Desktop from Windows
# Ensure WSL2 integration is enabled in Docker Desktop settings

# In WSL2 terminal, verify connection
docker info
```

### Docker Compose Issues

**Symptoms:**
- `no configuration file provided: not found`
- `service "xxx" failed to build`

**Solutions:**
```bash
# Verify docker-compose.yml exists
ls -la docker-compose.yml

# Check Docker Compose version
docker-compose --version

# Use newer Docker Compose syntax if needed
docker compose up -d  # instead of docker-compose up -d

# Rebuild containers if needed
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Port Conflicts

**Symptoms:**
- `Port 7474 is already allocated`
- `Port 7687 is already allocated`
- `Port 8000 is already allocated`

**Solutions:**
```bash
# Check what's using the ports
lsof -i :7474  # Neo4j HTTP
lsof -i :7687  # Neo4j Bolt
lsof -i :8000  # API server

# Stop conflicting services
# For Neo4j:
brew services stop neo4j    # macOS
sudo systemctl stop neo4j   # Linux

# Kill specific processes
sudo kill -9 $(lsof -t -i:7474)

# Or modify ports in docker-compose.yml
# Change the left side of port mappings:
# "7475:7474" instead of "7474:7474"
```

### Container Startup Issues

**Symptoms:**
- Containers exit immediately
- Services show as "unhealthy"

**Solutions:**
```bash
# Check container logs
docker-compose logs neo4j
docker-compose logs api

# Check container status
docker ps -a

# Restart specific service
docker-compose restart neo4j

# Complete reset
docker-compose down -v
docker-compose up -d
```

---

## 🐍 Python Environment Issues

### Module Import Errors

**Symptoms:**
- `ModuleNotFoundError: No module named 'app'`
- `ImportError: cannot import name 'xxx'`

**Solutions:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Check if you're in the right directory
pwd  # Should be in SubgraphRAGPlus root

# Reinstall in development mode
pip install -e .

# Verify installation
python -c "from src.app.api import app; print('✅ Import successful')"

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Virtual Environment Issues

**Symptoms:**
- `command not found: python`
- Wrong Python version
- Package installation fails

**Solutions:**
```bash
# Recreate virtual environment
rm -rf venv
python3.11 -m venv venv  # Use specific Python version
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install dependencies
pip install -r requirements-dev.txt

# Verify Python version
python --version  # Should be 3.9+
```

### Dependency Conflicts

**Symptoms:**
- Package version conflicts
- `pip` installation failures
- Import errors after installation

**Solutions:**
```bash
# Clean installation
pip freeze > current_packages.txt
pip uninstall -r current_packages.txt -y
pip install -r requirements-dev.txt

# Check for conflicts
pip check

# Use specific versions if needed
pip install "torch>=2.1.0,<3.0"
pip install "transformers>=4.35.0,<5.0"
```

---

## 🗄️ Database Issues

### 1. Neo4j Connection Failed

**Symptoms:**
- `Failed to connect to Neo4j`
- `ServiceUnavailable: Connection refused`

**Solutions:**
```bash
# Check if Neo4j container is running
docker ps | grep neo4j

# Check Neo4j logs
docker logs subgraphrag_neo4j

# Wait for Neo4j to fully start (can take 30-60 seconds)
sleep 30
curl http://localhost:7474

# Reset Neo4j container
docker-compose down
docker volume rm $(docker volume ls -q | grep neo4j)
docker-compose up -d neo4j

# Check Neo4j browser
open http://localhost:7474  # Default: neo4j/password
```

### 2. Neo4j Authentication Issues

**Symptoms:**
- `AuthError: The client is unauthorized`
- `Invalid username or password`

**Solutions:**
```bash
# Check environment variables
echo $NEO4J_USER
echo $NEO4J_PASSWORD

# Reset Neo4j password
docker exec -it subgraphrag_neo4j neo4j-admin set-initial-password newpassword

# Update .env file
NEO4J_PASSWORD=newpassword

# Restart application
docker-compose restart api
```

### 3. Database Schema Issues

**Symptoms:**
- `Node/Relationship not found`
- Schema migration errors

**Solutions:**
```bash
# Run schema migrations
python scripts/migrate_schema.py

# Check current schema
docker exec -it subgraphrag_neo4j cypher-shell -u neo4j -p password "CALL db.schema.visualization()"

# Reset database (⚠️ DESTRUCTIVE)
docker exec -it subgraphrag_neo4j cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n"

# Reapply migrations
python scripts/migrate_schema.py
```

### 4. SQLite Issues

**Symptoms:**
- `database is locked`
- `no such table` errors

**Solutions:**
```bash
# Check if SQLite file exists
ls -la data/staging.db

# Check for lock files
ls -la data/staging.db*

# Remove lock files (if safe)
rm data/staging.db-wal data/staging.db-shm

# Reset SQLite database
rm data/staging.db
python scripts/stage_ingest.py --sample  # Recreates database
```

---

## 🔌 API Issues

### 1. API Server Won't Start

**Symptoms:**
- `Address already in use`
- `Permission denied`
- Server exits immediately

**Solutions:**
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill conflicting process
sudo kill -9 $(lsof -t -i:8000)

# Start on different port
python src/main.py --port 8001

# Check for permission issues
chmod +x src/main.py

# Run with debug mode
DEBUG=true python src/main.py --reload
```

### 2. API Authentication Issues

**Symptoms:**
- `401 Unauthorized`
- `Invalid API key`

**Solutions:**
```bash
# Check API key in .env
grep API_KEY_SECRET .env

# Test with correct API key
curl -H "X-API-KEY: your-api-key" http://localhost:8000/health

# Generate new API key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. API Response Issues

**Symptoms:**
- `500 Internal Server Error`
- Slow responses
- Timeout errors

**Solutions:**
```bash
# Check API logs
docker-compose logs api

# Test with simple query
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'

# Check system resources
docker stats

# Increase timeout settings in config
```

---

## 🧠 ML/AI Issues

### 1. Model Download Failures

**Symptoms:**
- `Failed to download model`
- `Connection timeout`
- `Model not found`

**Solutions:**
```bash
# Check internet connection
curl -I https://huggingface.co

# Download models manually
python scripts/download_models.py

# Use local models instead
# Edit config.json to use local model paths

# Check available disk space
df -h
```

### 2. OpenAI API Issues

**Symptoms:**
- `OpenAI API key not found`
- `Rate limit exceeded`
- `Invalid request`

**Solutions:**
```bash
# Check API key
echo $OPENAI_API_KEY

# Test API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Use alternative model backend
# Edit config.json: "MODEL_BACKEND": "huggingface"
```

### 3. Vector Index Issues

**Symptoms:**
- `FAISS index not found`
- `Vector dimension mismatch`

**Solutions:**
```bash
# Check if FAISS index exists
ls -la data/faiss/

# Rebuild FAISS index
python scripts/merge_faiss.py

# Clear and rebuild
rm -rf data/faiss/*
python scripts/stage_ingest.py --sample
python scripts/ingest_worker.py --process-all
```

### 4. MLX Issues (Apple Silicon)

**Symptoms:**
- `No module named 'mlx'`
- `MLX not available on this platform`
- `Failed to load MLX model`
- `MLX memory allocation failed`

**Solutions:**

#### MLX Installation Issues
```bash
# Verify you're on Apple Silicon
python -c "import platform; print(f'Processor: {platform.processor()}')"
# Should show 'arm' for Apple Silicon

# Check macOS version (requires 13.3+)
sw_vers

# Reinstall MLX
pip uninstall mlx mlx-lm
pip install mlx mlx-lm

# Verify MLX installation
python -c "import mlx.core as mx; print(f'✅ MLX version: {mx.__version__}')"
```

#### MLX Model Loading Issues
```bash
# Check available disk space (models can be large)
df -h

# Clear MLX cache
rm -rf ~/.cache/huggingface/transformers/models--mlx*

# Test with smaller model
echo "MLX_LLM_MODEL=mlx-community/Mistral-7B-Instruct-v0.2-8bit-mlx" >> .env

# Check internet connectivity
ping huggingface.co

# Manual model download
python -c "
from mlx_lm import load
try:
    model, tokenizer = load('mlx-community/Mistral-7B-Instruct-v0.2-4bit-mlx')
    print('✅ Model loaded successfully')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

#### MLX Memory Issues
```bash
# Check available memory
vm_stat | grep "Pages free"

# Reduce memory usage
echo "MLX_MEMORY_LIMIT=4096" >> .env  # 4GB limit
echo "MLX_LLM_MODEL=mlx-community/Mistral-7B-Instruct-v0.2-8bit-mlx" >> .env

# Monitor MLX memory usage
python -c "
import mlx.core as mx
print(f'Active memory: {mx.metal.get_active_memory() / 1024**3:.2f} GB')
print(f'Cache memory: {mx.metal.get_cache_memory() / 1024**3:.2f} GB')
"

# Clear MLX memory cache
python -c "import mlx.core as mx; mx.metal.clear_cache()"
```

#### MLX Configuration Issues
```bash
# Verify MLX environment variables
grep MLX .env

# Test MLX configuration
python -c "
import os
os.environ['USE_MLX_LLM'] = 'true'
from app.ml.llm import LLMService
from app.ml.embedder import EmbedderService

try:
    llm = LLMService()
    print('✅ MLX LLM service initialized')
    
    embedder = EmbedderService()
    print('✅ MLX embedder service initialized')
except Exception as e:
    print(f'❌ Error: {e}')
"

# Fallback to non-MLX mode
echo "USE_MLX_LLM=false" >> .env
echo "MODEL_BACKEND=openai" >> .env
```

#### MLX Performance Issues
```bash
# Check system temperature (thermal throttling)
sudo powermetrics --samplers smc -n 1 | grep -i temp

# Optimize MLX settings
echo "MLX_USE_UNIFIED_MEMORY=true" >> .env
echo "MLX_NUM_THREADS=4" >> .env

# Use quantized models for better performance
echo "MLX_LLM_MODEL=mlx-community/Mistral-7B-Instruct-v0.2-8bit-mlx" >> .env
```

---

## 🔧 Performance Issues

### 1. Slow Query Responses

**Symptoms:**
- Queries take >30 seconds
- Timeout errors

**Solutions:**
```bash
# Check system resources
docker stats
htop  # or top

# Optimize Neo4j memory
# Edit docker-compose.yml:
# NEO4J_dbms_memory_heap_initial__size=1G
# NEO4J_dbms_memory_heap_max__size=2G

# Check query complexity
# Use simpler test queries first

# Enable query caching
# Edit config.json: "ENABLE_CACHE": true
```

### 2. High Memory Usage

**Symptoms:**
- System becomes unresponsive
- Out of memory errors

**Solutions:**
```bash
# Check memory usage
docker stats
free -h

# Reduce batch sizes
# Edit config.json: "BATCH_SIZE": 10

# Limit concurrent requests
# Edit config.json: "MAX_WORKERS": 2

# Restart services to clear memory
docker-compose restart
```

---

## 🔄 Complete Reset Procedures

### 1. Soft Reset (Preserve Data)

```bash
# Restart all services
docker-compose restart

# Clear caches only
rm -rf cache/*
rm -rf logs/*

# Restart application
python src/main.py --reload
```

### 2. Hard Reset (Remove All Data)

```bash
# ⚠️ WARNING: This will delete all data

# Stop all services
docker-compose down -v

# Remove all data
rm -rf data/faiss data/neo4j cache logs

# Remove Docker volumes
docker volume prune -f

# Start fresh
make setup-all
```

### 3. Development Environment Reset

```bash
# Remove virtual environment
rm -rf venv

# Clean Python cache
find . -type d -name "__pycache__" -delete
find . -name "*.pyc" -delete

# Recreate environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

---

## 📊 Diagnostic Commands

### System Information

```bash
# Operating system
uname -a

# Python version
python --version

# Docker version
docker --version
docker-compose --version

# Available memory
free -h  # Linux
vm_stat  # macOS

# Disk space
df -h
```

### Service Status

```bash
# Docker containers
docker ps -a

# Docker networks
docker network ls

# Docker volumes
docker volume ls

# Process list
ps aux | grep python
ps aux | grep neo4j
```

### Log Collection

```bash
# Collect all logs for support
mkdir -p debug_logs
docker-compose logs > debug_logs/docker.log
cp logs/*.log debug_logs/ 2>/dev/null || true
cp .env debug_logs/env.txt 2>/dev/null || true
tar -czf debug_logs.tar.gz debug_logs/

# Share debug_logs.tar.gz when reporting issues
```

---

## 🆘 Getting Help

If these troubleshooting steps don't resolve your issue:

### 1. Check Existing Resources
- [GitHub Issues](https://github.com/your-username/SubgraphRAGPlus/issues)
- [GitHub Discussions](https://github.com/your-username/SubgraphRAGPlus/discussions)
- [Installation Guide](installation.md)

### 2. Create a Bug Report

Include the following information:

```bash
# System information
uname -a
python --version
docker --version

# Error messages (exact text)
# Steps to reproduce
# Expected vs actual behavior

# Attach debug logs
tar -czf debug_logs.tar.gz debug_logs/
```

### 3. Community Support

- **💬 Discussions**: General questions and community help
- **🐛 Issues**: Bug reports and feature requests
- **📖 Documentation**: Check all docs in the `docs/` folder

---

**💡 Pro Tip**: Most issues can be resolved by ensuring Docker is running, virtual environment is activated, and all dependencies are properly installed. When in doubt, try a complete reset!** 