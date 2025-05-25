# 🚀 Installation Guide

This guide provides step-by-step instructions for installing SubgraphRAG+ in different environments.

## 📋 Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 2 cores | 4+ cores |
| **RAM** | 4GB | 8GB+ |
| **Storage** | 10GB free | 50GB+ SSD |
| **OS** | Linux, macOS, Windows (WSL2) | Linux/macOS |

### Software Dependencies

#### For Docker Setup (Recommended)
- **Docker Engine**: 20.10+ 
- **Docker Compose**: v2.0+
- **Git**: Latest version

#### For Local Development
- **Python**: 3.9+ (3.11+ recommended)
- **Git**: Latest version
- **Node.js**: 16+ (for frontend development)

## 🐳 Docker Installation (Recommended)

### 1. Install Docker

#### macOS
```bash
# Install Docker Desktop
brew install --cask docker
# OR download from https://docker.com/products/docker-desktop

# Start Docker Desktop
open -a Docker
```

#### Linux (Ubuntu/Debian)
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

#### Windows
1. Install [Docker Desktop for Windows](https://docker.com/products/docker-desktop)
2. Enable WSL2 integration
3. Restart your system

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# One-command setup
make setup-all

# Verify installation
curl http://localhost:8000/health
```

### 3. Access the Application

| Service | URL | Description |
|---------|-----|-------------|
| **API** | http://localhost:8000 | Main API server |
| **API Docs** | http://localhost:8000/docs | Interactive documentation |
| **Neo4j Browser** | http://localhost:7474 | Graph database UI |

## 🐍 Local Python Installation

### 1. Install Python Dependencies

#### macOS
```bash
# Install Python 3.11 (recommended)
brew install python@3.11

# Verify installation
python3.11 --version
```

#### Linux (Ubuntu/Debian)
```bash
# Install Python 3.11
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install additional dependencies
sudo apt install build-essential libssl-dev libffi-dev
```

#### Windows (WSL2)
```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### 2. Setup Project Environment

```bash
# Clone the repository
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt

# Install project in development mode
pip install -e .
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (use your preferred editor)
nano .env  # or vim, code, etc.
```

**Required environment variables:**
```bash
# Database
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-password

# API Security
API_KEY_SECRET=your-secret-key

# Optional: OpenAI integration
OPENAI_API_KEY=your-openai-key
```

### 4. Install and Setup Neo4j

#### Option A: Docker Neo4j (Recommended)
```bash
# Start Neo4j container
make neo4j-start

# Apply schema migrations
make migrate-schema
```

#### Option B: Local Neo4j Installation

**macOS:**
```bash
# Install Neo4j
brew install neo4j

# Start Neo4j service
brew services start neo4j

# Set initial password
neo4j-admin set-initial-password your-password
```

**Linux:**
```bash
# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 4.4' | sudo tee /etc/apt/sources.list.d/neo4j.list

# Install Neo4j
sudo apt update
sudo apt install neo4j

# Start Neo4j service
sudo systemctl enable neo4j
sudo systemctl start neo4j

# Set initial password
sudo neo4j-admin set-initial-password your-password
```

### 5. Initialize and Start

```bash
# Apply database migrations
python scripts/migrate_schema.py

# Load sample data (optional)
python scripts/stage_ingest.py --sample
python scripts/ingest_worker.py --process-all

# Start the application
python src/main.py --reload
```

## 🍎 MLX Installation (Apple Silicon)

MLX provides optimized machine learning on Apple Silicon Macs. This section is only relevant for Apple Silicon (M1/M2/M3) Mac users.

### 1. Prerequisites

- **Apple Silicon Mac** (M1, M2, or M3 chip)
- **macOS 13.3+** (Ventura or later)
- **Python 3.9+** (3.11+ recommended)

### 2. Install MLX

```bash
# Activate your virtual environment first
source venv/bin/activate

# Install MLX and related packages
pip install mlx mlx-lm

# Verify MLX installation
python -c "import mlx.core as mx; print(f'✅ MLX installed: {mx.__version__}')"
```

### 3. Configure MLX Environment

```bash
# Enable MLX in your .env file
echo "USE_MLX_LLM=true" >> .env
echo "MODEL_BACKEND=mlx" >> .env

# Set MLX models (these will be downloaded automatically)
echo "MLX_LLM_MODEL=mlx-community/Mistral-7B-Instruct-v0.2-4bit-mlx" >> .env
echo "MLX_EMBEDDING_MODEL=mlx-community/all-MiniLM-L6-v2-mlx" >> .env
```

### 4. Download MLX Models

Models will be downloaded automatically on first use, but you can pre-download them:

```bash
# Create models directory
mkdir -p models/mlx

# Download LLM model (optional - will auto-download)
python -c "
from mlx_lm import load
model, tokenizer = load('mlx-community/Mistral-7B-Instruct-v0.2-4bit-mlx')
print('✅ LLM model downloaded')
"

# Download embedding model (optional - will auto-download)
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('✅ Embedding model downloaded')
"
```

### 5. Verify MLX Setup

```bash
# Test MLX functionality
python -c "
import os
os.environ['USE_MLX_LLM'] = 'true'
from app.ml.llm import LLMService
from app.ml.embedder import EmbedderService

# Test LLM
llm = LLMService()
print('✅ MLX LLM service initialized')

# Test embedder
embedder = EmbedderService()
print('✅ MLX embedder service initialized')
"
```

### 6. Performance Optimization

For optimal MLX performance:

```bash
# Set memory allocation (in .env)
echo "MLX_MEMORY_LIMIT=8192" >> .env  # 8GB limit

# Enable unified memory
echo "MLX_USE_UNIFIED_MEMORY=true" >> .env

# Set thread count (optional)
echo "MLX_NUM_THREADS=4" >> .env
```

### MLX Troubleshooting

#### Common MLX Issues

**1. MLX Import Error**
```bash
# Error: No module named 'mlx'
# Solution: Ensure you're on Apple Silicon
python -c "import platform; print(platform.processor())"
# Should show 'arm' for Apple Silicon

# Reinstall MLX
pip uninstall mlx mlx-lm
pip install mlx mlx-lm
```

**2. Model Download Issues**
```bash
# Error: Failed to download model
# Solution: Check internet connection and disk space
df -h  # Check disk space
ping huggingface.co  # Check connectivity

# Clear cache and retry
rm -rf ~/.cache/huggingface
```

**3. Memory Issues**
```bash
# Error: Out of memory
# Solution: Reduce model size or memory limit
echo "MLX_LLM_MODEL=mlx-community/Mistral-7B-Instruct-v0.2-8bit-mlx" >> .env
echo "MLX_MEMORY_LIMIT=4096" >> .env
```

**4. Performance Issues**
```bash
# Check system resources
top -l 1 | grep "CPU usage"
vm_stat | grep "Pages free"

# Monitor MLX memory usage
python -c "
import mlx.core as mx
print(f'MLX memory: {mx.metal.get_active_memory() / 1024**3:.2f} GB')
"
```

## �� Development Setup

For contributors and developers:

```bash
# Clone your fork
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Setup development environment
./bin/setup_dev.sh

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
make test
```

## ✅ Verification

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed readiness check
curl http://localhost:8000/readyz

# Test API functionality
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?"}'
```

### Component Status

```bash
# Check Neo4j connection
curl http://localhost:8000/health | jq '.neo4j'

# Check vector database
curl http://localhost:8000/health | jq '.faiss'

# View system metrics
curl http://localhost:8000/metrics
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Docker Daemon Not Running

**Error:** `Cannot connect to the Docker daemon`

**Solution:**
```bash
# macOS: Start Docker Desktop
open -a Docker

# Linux: Start Docker service
sudo systemctl start docker

# Verify Docker is running
docker info
```

#### 2. Port Conflicts

**Error:** `Port 7474/7687 is already allocated`

**Solution:**
```bash
# Check what's using the ports
lsof -i :7474
lsof -i :7687

# Stop conflicting services
brew services stop neo4j    # macOS
sudo systemctl stop neo4j   # Linux

# Or modify ports in docker-compose.yml
```

#### 3. Python Import Errors

**Error:** `ModuleNotFoundError: No module named 'app'`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall in development mode
pip install -e .

# Verify installation
python -c "from app.api import app; print('✅ Import successful')"
```

#### 4. Neo4j Connection Failed

**Error:** `Failed to connect to Neo4j`

**Solution:**
```bash
# Check Neo4j status
docker ps | grep neo4j

# View Neo4j logs
docker logs subgraphrag_neo4j

# Reset Neo4j container
docker-compose down
docker-compose up -d neo4j

# Wait for startup and retry
sleep 30
curl http://localhost:7474
```

#### 5. Permission Issues (Linux)

**Error:** `Permission denied`

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x bin/*.sh
```

### Advanced Troubleshooting

#### Complete Reset

If all else fails:

```bash
# Stop all services
docker-compose down -v
docker system prune -a

# Remove virtual environment
rm -rf venv

# Clean data directories
rm -rf data/faiss data/neo4j cache logs

# Start fresh
./bin/setup_dev.sh
```

#### Debug Mode

Enable detailed logging:

```bash
# Set debug environment
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with verbose output
./bin/setup_dev.sh --verbose

# Check logs
tail -f logs/app.log
```

## 🆘 Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: [GitHub Issues](https://github.com/your-username/SubgraphRAGPlus/issues)
2. **Search discussions**: [GitHub Discussions](https://github.com/your-username/SubgraphRAGPlus/discussions)
3. **Create a new issue** with:
   - Your operating system and version
   - Docker version (`docker --version`)
   - Python version (`python --version`)
   - Complete error messages
   - Steps to reproduce

## 📚 Next Steps

After successful installation:

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Read the documentation**: Check out [docs/README.md](README.md)
3. **Try examples**: See [API Usage Examples](api_examples.md)
4. **Join the community**: [GitHub Discussions](https://github.com/your-username/SubgraphRAGPlus/discussions)

---

**🎉 Congratulations! SubgraphRAG+ is now installed and ready to use.** 