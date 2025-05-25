# 📚 SubgraphRAG+ Documentation

Welcome to the comprehensive documentation for **SubgraphRAG+**, an advanced knowledge graph-powered question answering system that combines graph traversal, dense retrieval, and machine learning for intelligent information extraction.

## 🚀 Quick Start

| I want to... | Go to... | Time needed |
|--------------|----------|-------------|
| **🏃‍♂️ Get started quickly** | [Installation Guide](installation.md) → Docker Setup | 5-10 min |
| **🔧 Set up for development** | [Installation Guide](installation.md) → Development Setup | 10-15 min |
| **🍎 Optimize for Apple Silicon** | [Installation Guide](installation.md) → Apple Silicon | 15-20 min |
| **🐛 Fix issues** | [Troubleshooting Guide](troubleshooting.md) | As needed |
| **📡 Learn the API** | [API Reference](api_reference.md) | 15-30 min |

---

## 📖 Documentation Structure

### 🛠️ Setup & Installation
- **[📦 Installation Guide](installation.md)** - Complete setup instructions for all platforms
- **[🆘 Troubleshooting](troubleshooting.md)** - Solutions for common issues and problems
- **[⚙️ Configuration](configuration.md)** - Environment variables and system configuration

### 🏗️ Architecture & Development  
- **[🏛️ Architecture Guide](architecture.md)** - System design and component overview
- **[🔧 Development Guide](development.md)** - Contributing, testing, and development workflow
- **[🤝 Contributing](contributing.md)** - Guidelines for contributors and code standards

### 📡 API & Usage
- **[📋 API Reference](api_reference.md)** - Complete API documentation with examples
- **[🚀 Deployment Guide](deployment.md)** - Production deployment and scaling

### 🧠 Advanced Topics
- **[🍎 MLX Integration](mlx.md)** - Apple Silicon optimization with MLX
- **[🤖 MLP Retriever](mlp_retriever.md)** - Technical details on the MLP scoring model

---

## 🎯 Choose Your Path

### 👤 End Users
**Goal**: Use SubgraphRAG+ for question answering

1. **Start here**: [Installation Guide](installation.md) → Docker Setup
2. **Learn the API**: [API Reference](api_reference.md)
3. **Get help**: [Troubleshooting Guide](troubleshooting.md)

### 👨‍💻 Developers  
**Goal**: Contribute to or customize SubgraphRAG+

1. **Setup**: [Installation Guide](installation.md) → Development Setup
2. **Understand**: [Architecture Guide](architecture.md)
3. **Develop**: [Development Guide](development.md)
4. **Contribute**: [Contributing Guide](contributing.md)

### 🚀 DevOps/Deployment
**Goal**: Deploy SubgraphRAG+ in production

1. **Setup**: [Installation Guide](installation.md) → Docker Setup
2. **Configure**: [Configuration Guide](configuration.md)
3. **Deploy**: [Deployment Guide](deployment.md)
4. **Monitor**: [Troubleshooting Guide](troubleshooting.md)

### 🔬 Researchers
**Goal**: Understand and extend the system

1. **Architecture**: [Architecture Guide](architecture.md)
2. **MLP Details**: [MLP Retriever Guide](mlp_retriever.md)
3. **Development**: [Development Guide](development.md)
4. **Apple Silicon**: [MLX Integration](mlx.md)

---

## 🏛️ System Overview

SubgraphRAG+ is a sophisticated question-answering system that combines:

### Core Components
- **🧠 Knowledge Graph**: Neo4j-powered graph database
- **🔍 Dense Retrieval**: FAISS-based vector search  
- **🤖 MLP Scoring**: Machine learning relevance scoring
- **🌐 REST API**: FastAPI-based web service
- **🍎 MLX Support**: Apple Silicon optimization

### Key Features
- **Hybrid Retrieval**: Combines graph traversal with semantic search
- **Intelligent Scoring**: ML-powered relevance ranking
- **Streaming Responses**: Real-time answer generation
- **Graph Visualization**: Interactive subgraph exploration
- **Multi-Platform**: Docker, local, and Apple Silicon support

### Supported Models
- **OpenAI**: GPT-3.5/4 integration
- **HuggingFace**: Open-source model support
- **MLX**: Optimized Apple Silicon inference
- **Local Models**: Self-hosted deployment options

---

## 🛠️ Quick Setup Commands

### Docker (Recommended)
```bash
git clone <repository-url>
cd SubgraphRAGPlus
./bin/setup_docker.sh
# Visit http://localhost:8000/docs
```

### Development
```bash
git clone <repository-url>
cd SubgraphRAGPlus
./bin/setup_dev.sh
source venv/bin/activate
make serve
```

### Apple Silicon (MLX)
```bash
./bin/setup_dev.sh  # Auto-detects Apple Silicon
# Configure MLX in .env
make serve
```

---

## 📊 Documentation Quality

### ✅ What's Covered
- **Complete Installation**: All platforms and methods
- **Comprehensive API**: Every endpoint documented
- **Troubleshooting**: Common issues and solutions
- **Architecture**: System design and components
- **Development**: Contributing and testing
- **Deployment**: Production setup and scaling

### 🎯 Documentation Principles
- **User-Focused**: Organized by user goals and use cases
- **Action-Oriented**: Clear steps and commands
- **Comprehensive**: Covers all aspects of the system
- **Maintained**: Kept up-to-date with code changes
- **Accessible**: Clear language and good formatting

---

## 🆘 Getting Help

### 📖 Self-Service
1. **Search this documentation** for your specific issue
2. **Check [Troubleshooting](troubleshooting.md)** for common problems
3. **Review [API Reference](api_reference.md)** for usage questions

### 🤝 Community Support
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/your-username/SubgraphRAGPlus/issues)
- **❓ Questions**: [GitHub Discussions](https://github.com/your-username/SubgraphRAGPlus/discussions)
- **💡 Feature Requests**: [GitHub Issues](https://github.com/your-username/SubgraphRAGPlus/issues)

### 📝 Contributing to Documentation
Found an error or want to improve the docs? See our [Contributing Guide](contributing.md) for how to submit documentation improvements.

---

## 🗺️ Documentation Map

```
docs/
├── README.md                 # 👈 You are here - Main navigation
├── installation.md           # 📦 Complete setup guide
├── troubleshooting.md        # 🆘 Problem solving
├── configuration.md          # ⚙️ System configuration
├── architecture.md           # 🏛️ System design
├── development.md            # 🔧 Development workflow
├── contributing.md           # 🤝 Contribution guidelines
├── api_reference.md          # 📡 API documentation
├── deployment.md             # 🚀 Production deployment
├── mlx.md                   # 🍎 Apple Silicon optimization
└── mlp_retriever.md         # 🤖 MLP technical details
```

---

## 🎉 What's Next?

After reading this overview:

1. **🚀 Get Started**: Choose your installation method in the [Installation Guide](installation.md)
2. **🔍 Explore**: Try the API using the [API Reference](api_reference.md)
3. **🏗️ Understand**: Learn the system architecture in the [Architecture Guide](architecture.md)
4. **🤝 Contribute**: Join the community via the [Contributing Guide](contributing.md)

---

<div align="center">

**📚 Happy Learning!**

*SubgraphRAG+ Documentation - Your guide to intelligent question answering*

</div> 