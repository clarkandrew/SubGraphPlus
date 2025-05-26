# 🛠️ Installation Guide

This guide provides step-by-step instructions for installing SubgraphRAG+ on different platforms and environments.

## 🎯 Choose Your Installation Method

| Method | Best For | Time | Requirements |
|--------|----------|------|-------------|
| **[🎯 Quick Demo](#-quick-demo-installation)** | First-time users, evaluation | 2-5 min | Python 3.9+ |
| **[🐳 Docker Setup](#-docker-installation-recommended)** | Production, quick start | 5-10 min | Docker |
| **[🔧 Development Setup](#-development-installation)** | Contributors, customization | 10-15 min | Python 3.9+ |
| **[🍎 Apple Silicon](#-apple-silicon-optimization)** | M1/M2/M3 Mac optimization | 15-20 min | macOS + Python |

---

## 📋 Prerequisites

### System Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|--------|
| **CPU** | 2 cores | 4+ cores | Intel or Apple Silicon |
| **RAM** | 4GB | 8GB+ | More needed for large models |
| **Storage** | 10GB free | 50GB+ SSD | Fast storage improves performance |
| **OS** | Linux, macOS, Windows (WSL2) | Linux/macOS preferred |

### Software Requirements

#### All Methods
- **Git**: Latest version for repository cloning

#### Docker Method
- **Docker Engine**: 20.10+ with Docker Compose v2
- **4GB+ RAM** available for containers

#### Development Method  
- **Python**: 3.9+ (3.11+ recommended, tested up to 3.13)
- **pip**: Latest version

---

## 🎯 Quick Demo Installation

**Best for: First-time users who want to see SubgraphRAG+ in action immediately**

This is the fastest way to get SubgraphRAG+ running with minimal setup:

### Prerequisites
- Python 3.9+ installed
- 4GB+ RAM available
- Internet connection for model downloads

### One-Command Demo

```bash
# Clone and run demo in one command
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus
python examples/demo_quickstart.py --help
```

### Demo Features

The demo script provides:
- **📋 Progress Indicators**: Clear step-by-step feedback
- **⚡ Smart Performance**: Automatically skips steps if already completed
- **🔧 Flexible Options**: Multiple configuration options
- **💡 Helpful Errors**: Clear guidance when issues occur

### Demo Options

```bash
# Full demo (recommended for first run)
python examples/demo_quickstart.py

# Quick demo (skip database setup)
python examples/demo_quickstart.py --skip-neo4j --skip-data

# Custom port (if 8000 is busy)
python examples/demo_quickstart.py --port 8080

# See all options
python examples/demo_quickstart.py --help
```

### What the Demo Does

1. **🔧 Environment Setup** (30s): Creates virtual environment and installs dependencies
2. **🗄️ Database Check** (15s): Verifies Neo4j connection (skippable)
3. **🧠 Model Validation** (10s): Checks for required MLP model
4. **📥 Data Population** (60s): Loads sample knowledge graph (skippable if present)
5. **🚀 Server Startup** (30s): Starts API server with health checks
6. **🧪 Live Demo** (15s): Runs example query to show functionality

**Total Time: 2-5 minutes** (depending on options and system speed)

### Demo Troubleshooting

If the demo fails, try these options:

```bash
# Skip Neo4j if you don't have it installed
python examples/demo_quickstart.py --skip-neo4j

# Skip data ingestion if it's slow
python examples/demo_quickstart.py --skip-data

# Use different port if 8000 is busy
python examples/demo_quickstart.py --port 8001

# Minimal demo for testing
python examples/demo_quickstart.py --skip-neo4j --skip-data --port 8001
```

### After the Demo

Once the demo completes successfully, you can:
- **Explore the API**: Visit http://localhost:8000/docs
- **Try queries**: Use the interactive API documentation
- **Set up full installation**: Follow the Docker or Development setup below
- **Read documentation**: Check out the [Architecture Guide](architecture.md)

---

## 🐳 Docker Installation (Recommended)

**Best for: Production deployments, quick evaluation, users who want everything configured automatically**

### Step 1: Install Docker

#### macOS
```bash
# Install Docker Desktop
brew install --cask docker
# OR download from https://docker.com/products/docker-desktop

# Start Docker Desktop and wait for it to be ready
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

# Verify installation
docker --version
docker compose version
```

#### Windows
1. Install [Docker Desktop for Windows](https://docker.com/products/docker-desktop)
2. Enable WSL2 integration in Docker Desktop settings
3. Restart your system

### Step 2: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Run the interactive Docker setup script
./bin/setup_docker.sh

# Or use the make shortcut
make setup-all
```

The setup script will:
- ✅ Check Docker installation and requirements
- ✅ Pull and configure all necessary container images
- ✅ Set up Neo4j database with proper configuration
- ✅ Configure networking and environment variables
- ✅ Start all services and verify they're working
- ✅ Load sample data for testing

### Step 3: Verify Installation

```bash
# Check service status
docker ps

# Test API health
curl http://localhost:8000/healthz

# Test a simple query
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: default_key_for_dev_only" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?", "visualize_graph": true}'
```

### Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| **API Server** | http://localhost:8000 | API Key: `default_key_for_dev_only` |
| **API Documentation** | http://localhost:8000/docs | - |
| **Neo4j Browser** | http://localhost:7474 | User: `neo4j`, Password: `password` |

---

## 🔧 Development Installation

**Best for: Contributors, developers who want to modify code, advanced customization**

### Step 1: Install Python

#### macOS
```bash
# Install Python 3.11 (recommended)
brew install python@3.11

# Verify installation
python3.11 --version  # Should show 3.11.x
```

#### Linux (Ubuntu/Debian)
```bash
# Install Python 3.11 and development tools
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev \
                 build-essential libssl-dev libffi-dev git

# Verify installation
python3.11 --version
```

#### Windows (WSL2 Required)
```bash
# Update package list
sudo apt update

# Install Python and tools
sudo apt install python3.11 python3.11-venv python3.11-dev \
                 build-essential git

# Verify installation
python3.11 --version
```

### Step 2: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Run the interactive development setup script
./bin/setup_dev.sh

# Or for advanced users, use options:
./bin/setup_dev.sh --python python3.11 --run-tests
```

The setup script will:
- ✅ Detect and validate your Python version
- ✅ Create and configure a virtual environment
- ✅ Install all Python dependencies
- ✅ Set up Neo4j (Docker or guide you through local installation)
- ✅ Create configuration files from templates
- ✅ Initialize database schema and indexes
- ✅ Download required AI models
- ✅ Load sample data (optional)
- ✅ Run tests to verify installation (optional)

### Step 3: Manual Configuration (if needed)

If the setup script can't auto-configure everything:

```bash
# Activate virtual environment
source venv/bin/activate

# Create configuration from template
cp .env.example .env

# Edit configuration with your preferred editor
nano .env  # or vim, code, etc.
```

**Essential configuration variables:**
```bash
# Database connection
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# API security
API_KEY_SECRET=your-secret-key-here

# Choose your LLM backend
MODEL_BACKEND=openai
OPENAI_API_KEY=your-openai-key

# Or use HuggingFace
# MODEL_BACKEND=hf

# Or use MLX on Apple Silicon
# MODEL_BACKEND=mlx
# USE_MLX_LLM=true
```

### Step 4: Start Development Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start development server with auto-reload
python src/main.py --reload

# Or use make command
make serve
```

### Step 5: Verify Installation

```bash
# Run health checks
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz

# Run test suite
make test

# Test a query
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"question": "test query", "visualize_graph": false}'
```

---

## 🍎 Apple Silicon Optimization

**Best for: M1/M2/M3 Mac users who want optimal performance**

### Why MLX?
- **Native Performance**: Up to 2x faster inference on Apple Silicon
- **Memory Efficient**: Lower memory usage compared to PyTorch
- **Integrated**: Seamless integration with existing workflows

### Installation Steps

```bash
# Follow development installation first
./bin/setup_dev.sh

# The setup script will auto-detect Apple Silicon and prompt for MLX
# Or manually install MLX:
pip install mlx mlx-lm

# Configure MLX in .env
echo "MODEL_BACKEND=mlx" >> .env
echo "USE_MLX_LLM=true" >> .env
```

### MLX Configuration

Create or update `config/config.json`:
```json
{
  "MODEL_BACKEND": "mlx",
  "MLX_MODEL_PATH": "mlx-community/Mistral-7B-Instruct-v0.2-8bit-mlx",
  "MLX_MAX_TOKENS": 2048,
  "TOKEN_BUDGET": 8000
}
```

### Verify MLX Setup

```bash
# Test MLX installation
python -c "import mlx.core as mx; print('MLX version:', mx.__version__)"

# Test MLX LLM
python scripts/test_embedder.py

# Start server with MLX
make serve
```

See **[MLX Integration Guide](mlx.md)** for advanced MLX configuration.

---

## 🚦 Verification & Testing

### Quick Health Check

After any installation method:

```bash
# 1. Basic health
curl http://localhost:8000/healthz

# 2. Dependency check
curl http://localhost:8000/readyz

# 3. Simple query test
curl -X POST "http://localhost:8000/query" \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "visualize_graph": true}'
```

### Expected Results

1. **Health check** should return: `{"status": "ok"}`
2. **Readiness check** should show all dependencies as `"ok"`
3. **Query test** should return streaming response with answer and graph data

### Run Test Suite

```bash
# Full test suite (development setup only)
make test

# Quick smoke tests
make test-smoke

# API integration tests
make test-api
```

---

## 🛠️ Common Setup Commands

### Docker Environment
```bash
make setup-all          # Full Docker setup
make docker-start       # Start services
make docker-stop        # Stop services
make docker-logs        # View logs
```

### Development Environment
```bash
make setup-dev          # Full development setup
make serve              # Start development server
make test               # Run test suite
make lint               # Check code quality
```

### Database Operations
```bash
make neo4j-start        # Start Neo4j container
make migrate-schema     # Apply database schema
make ingest-sample      # Load sample data
```

---

## 🆘 Troubleshooting

### Quick Fixes

**Setup script hangs:**
```bash
# Use timeout protection
./bin/setup_dev.sh --skip-tests --skip-sample_data
```

**Docker issues:**
```bash
# Restart Docker
docker system prune -f
./bin/setup_docker.sh
```

**Module import errors:**
```bash
# Reinstall in development mode
source venv/bin/activate
pip install -e .
```

**Neo4j connection issues:**
```bash
# Check Neo4j status
docker ps | grep neo4j
curl http://localhost:7474
```

### Comprehensive Troubleshooting

For detailed troubleshooting, see **[Troubleshooting Guide](troubleshooting.md)**.

### Get Help

- **🐛 Installation Issues**: [GitHub Issues](https://github.com/your-username/SubgraphRAGPlus/issues)
- **❓ Setup Questions**: [GitHub Discussions](https://github.com/your-username/SubgraphRAGPlus/discussions)
- **📖 Documentation**: [docs/README.md](README.md)

---

## 🎉 Next Steps

After successful installation:

1. **📡 Learn the API**: [API Reference](api_reference.md)
2. **🏗️ Understand Architecture**: [Architecture Guide](architecture.md)
3. **🔧 Contribute**: [Development Guide](development.md)
4. **🚀 Deploy**: [Deployment Guide](deployment.md)

---

<div align="center">

**🎊 Installation Complete!** 

Visit **[http://localhost:8000/docs](http://localhost:8000/docs)** to explore the API

</div> 