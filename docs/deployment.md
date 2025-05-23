# SubgraphRAG+ Deployment Guide

This document outlines best practices for deploying SubgraphRAG+ in different environments - from development to production.

## Table of Contents

1. [Overview](#overview)
2. [Docker Deployment (Recommended)](#docker-deployment-recommended)
3. [Local Neo4j Deployment](#local-neo4j-deployment)
4. [Local Development Setup](#local-development-setup)
5. [Production Deployment](#production-deployment)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)

## Overview

SubgraphRAG+ combines several components that need to work together:

- **Neo4j**: Graph database storing knowledge triples
- **FastAPI**: API server with endpoints for querying, ingestion, etc.
- **SQLite**: Staging database for ingestion
- **FAISS**: Vector index for dense retrieval
- **ML Models**: For triple scoring and LLM integration

## Docker Deployment (Recommended)

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB of available RAM
- At least 10GB of disk space

### Quick Start

```bash
# Clone the repository
git clone https://github.com/clarkandrew/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Make scripts executable
chmod +x make-executable.sh
./make-executable.sh

# Start the system with Docker
./docker-setup.sh start

# Wait for the system to fully start (usually takes 30-60 seconds)
# Then initialize with sample data
./docker-setup.sh sample-data

# Run the interactive demo
./demo.sh
```

### Configuration

Edit `docker-compose.yml` to customize:

- **API_KEY_SECRET**: Change to a secure value
- **OPENAI_API_KEY**: Add if using OpenAI backend
- **Port mappings**: Change if ports 7474, 7687, or 8000 conflict
- **Volume mounts**: Customize if needed

### Resource Management

Docker volumes are used to persist data:

- `neo4j_data`: Neo4j database files
- `app_data`: Application data including SQLite
- `app_models`: ML models storage
- `app_cache`: Cache storage
- `app_logs`: Application logs

### Backup and Restore

```bash
# Create backup
./docker-setup.sh backup

# Backups are stored in ./backups
```

## Local Neo4j Deployment

If you prefer not to use Docker for deploying Neo4j, you can install and run Neo4j directly on your server or local machine.

### Prerequisites

- Operating system: Linux, macOS, or Windows
- Java Runtime Environment (JRE) 11+

### Installation Options

#### Option 1: Neo4j Desktop (Best for development)

1. Download Neo4j Desktop from [neo4j.com/download](https://neo4j.com/download/)
2. Install and run Neo4j Desktop
3. Create a new project and add a local database (version 4.4+ recommended)
4. Set password to a secure value
5. Install the APOC plugin via the Plugins tab
6. Start the database

#### Option 2: Direct Installation

##### On macOS with Homebrew:

```bash
# Install Neo4j
brew install neo4j

# Start the service
brew services start neo4j

# Set password (will prompt for password change)
cypher-shell -u neo4j -p neo4j

# Install APOC plugin manually
# 1. Download the APOC plugin JAR
wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.4.0.10/apoc-4.4.0.10-all.jar
# 2. Move it to Neo4j plugins directory
mv apoc-4.4.0.10-all.jar /usr/local/var/neo4j/plugins/
# 3. Restart Neo4j
brew services restart neo4j
```

##### On Linux (Debian/Ubuntu):

```bash
# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
sudo apt-get update

# Install Neo4j
sudo apt-get install neo4j=1:4.4.0

# Start Neo4j service
sudo systemctl start neo4j

# Set password
cypher-shell -u neo4j -p neo4j

# Install APOC plugin manually
sudo wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.4.0.10/apoc-4.4.0.10-all.jar -P /var/lib/neo4j/plugins/
sudo systemctl restart neo4j
```

### Configuration

Edit the Neo4j configuration file to enable remote connections and APOC procedures:

```bash
# Location varies by installation method:
# - Homebrew: /usr/local/var/neo4j/conf/neo4j.conf
# - Linux: /etc/neo4j/neo4j.conf
# - Neo4j Desktop: Via the Settings tab in the application

# Add or uncomment these lines:
dbms.connector.bolt.enabled=true
dbms.connector.bolt.listen_address=0.0.0.0:7687
dbms.connector.http.enabled=true
dbms.connector.http.listen_address=0.0.0.0:7474
dbms.security.procedures.unrestricted=apoc.*
```

### Connecting SubgraphRAG+ to local Neo4j

Update your `.env` file with the appropriate connection details:

```
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

## Local Development Setup

### Prerequisites

- Python 3.11+
- Neo4j 4.4+ with APOC plugin (Docker or locally installed)
- SQLite3
- Git

### Setup Steps

```bash
# Clone repository
git clone https://github.com/clarkandrew/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Make scripts executable
chmod +x make-executable.sh
./make-executable.sh

# Run setup script with local Neo4j option
./setup.sh --use-local-neo4j

# If using locally installed Neo4j, ensure it's running:
# - Neo4j Desktop: Start from the application
# - Homebrew: brew services start neo4j
# - Linux: sudo systemctl start neo4j

# Start the API server
./run.sh

# In another terminal, initialize with sample data
python scripts/stage_ingest.py --sample
python scripts/ingest_worker.py --process-all
python scripts/merge_faiss.py

# Run tests
./run_tests.sh
```

### Configuration

- `.env`: Set environment variables
- `config/config.json`: Configure application settings

## Production Deployment

For production environments, consider these additional steps:

### Security Enhancements

1. **API Authentication**:
   - Change the default API key to a strong, randomly generated value
   - Consider implementing JWT or OAuth2 for more robust authentication

2. **HTTPS/TLS**:
   - Deploy behind a reverse proxy (Nginx, Traefik) with TLS termination
   - Use valid SSL certificates (Let's Encrypt)

3. **Network Security**:
   - Use Docker networks to isolate components
   - Restrict external access to Neo4j (use internal Docker network only)
   - Implement proper firewall rules

### High Availability Setup

1. **Neo4j Clustering**:
   - Consider Neo4j Enterprise for clustering
   - Configure causal clustering with core and read replicas

2. **API Scaling**:
   - Deploy multiple API instances behind a load balancer
   - Use Docker Swarm or Kubernetes for orchestration

3. **Backup Strategy**:
   - Schedule regular backups (Neo4j, SQLite, FAISS)
   - Test restore procedures regularly
   - Consider offsite backup storage

### Performance Optimization

1. **Neo4j Tuning**:
   - Allocate sufficient memory for heap and page cache
   - Use appropriate index strategies for your data
   - Configure query timeouts

2. **Hardware Recommendations**:
   - Neo4j: Memory-optimized instance (16+ GB RAM)
   - API: Balanced instance (4+ GB RAM)
   - Consider SSD storage for all components

3. **Monitoring**:
   - Set up Prometheus + Grafana for metrics visualization
   - Configure alerting for key thresholds

### Sample Docker Compose for Production

```yaml
services:
  traefik:
    image: traefik:v2.10
    command:
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.myresolver.acme.tlschallenge=true"
      - "--certificatesresolvers.myresolver.acme.email=your@email.com"
      - "--certificatesresolvers.myresolver.acme.storage=/letsencrypt/acme.json"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - "/var/run/docker.sock:/var/run/docker.sock:ro"
      - "letsencrypt:/letsencrypt"
    networks:
      - traefik-public

  neo4j:
    image: neo4j:4.4
    container_name: subgraphrag_neo4j
    environment:
      - NEO4J_AUTH=neo4j/strong-password-here
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_memory_heap_initial__size=2G
      - NEO4J_dbms_memory_heap_max__size=4G
      - NEO4J_dbms_memory_pagecache_size=4G
    volumes:
      - neo4j_data:/data
    networks:
      - subgraphrag-net
    restart: unless-stopped

  subgraphrag:
    build: .
    container_name: subgraphrag_api
    depends_on:
      - neo4j
    environment:
      - NEO4J_URI=bolt://subgraphrag_neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=strong-password-here
      - API_KEY_SECRET=very-strong-api-key-here
      - MODEL_BACKEND=openai
      - OPENAI_API_KEY=your-openai-key
    volumes:
      - app_data:/app/data
      - app_models:/app/models
      - app_cache:/app/cache
      - app_logs:/app/logs
    networks:
      - subgraphrag-net
      - traefik-public
    restart: unless-stopped
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.subgraphrag.rule=Host(`api.yourdomain.com`)"
      - "traefik.http.routers.subgraphrag.entrypoints=websecure"
      - "traefik.http.routers.subgraphrag.tls.certresolver=myresolver"
      - "traefik.http.services.subgraphrag.loadbalancer.server.port=8000"

networks:
  subgraphrag-net:
    driver: bridge
  traefik-public:
    external: true

volumes:
  neo4j_data:
  app_data:
  app_models:
  app_cache:
  app_logs:
  letsencrypt:
```

## Monitoring and Maintenance

### Health Checks

SubgraphRAG+ provides several endpoints for monitoring:

- `/healthz`: Basic application health
- `/readyz`: Dependency health (Neo4j, SQLite, FAISS, LLM backend)
- `/metrics`: Prometheus-format metrics

### Regular Maintenance Tasks

1. **Backup**:
   ```bash
   ./docker-setup.sh backup
   ```

2. **Neo4j Index Optimization**:
   ```bash
   docker exec subgraphrag_neo4j cypher-shell -u neo4j -p password "CALL db.indexes()"
   ```

3. **FAISS Index Rebuilding** (if performance degrades):
   ```bash
   docker exec subgraphrag_api python scripts/merge_faiss.py --rebuild
   ```

4. **Log Rotation**:
   - Configure log rotation for Neo4j and application logs
   - Example logrotate configuration can be found in `docs/logrotate.conf`

## Security Considerations

### API Key Management

- Rotate API keys regularly
- Store API keys securely (environment variables, secrets management)
- Implement rate limiting for API endpoints

### Data Protection

- Encrypt sensitive data at rest
- Configure Neo4j authentication properly
- Implement proper backup encryption
- Consider data retention policies for logs and feedback data

### Authentication Flow

1. All requests must include the API key in the `X-API-KEY` header
2. The API validates the key before processing the request
3. Failed authentication attempts are logged to SQLite
4. Rate limiting is applied to prevent brute force attacks

## Troubleshooting

### Common Issues

**Neo4j Connection Failures**:
- For Docker Neo4j:
  - Check Neo4j logs: `docker logs subgraphrag_neo4j`
  - Verify Neo4j is running: `docker ps | grep neo4j`
- For local Neo4j:
  - Check logs:
    - Homebrew: `cat /usr/local/var/log/neo4j/neo4j.log`
    - Linux: `sudo journalctl -u neo4j`
    - Neo4j Desktop: View logs in the application
  - Verify Neo4j is running:
    - Homebrew: `brew services list | grep neo4j`
    - Linux: `sudo systemctl status neo4j`
    - Neo4j Desktop: Check status in the application
- Ensure correct password in environment variables

**API Server Errors**:
- Check API logs: `docker logs subgraphrag_api`
- Verify environment variables are set correctly
- Ensure all volumes are properly mounted

**FAISS Index Issues**:
- Rebuild the index: `docker exec subgraphrag_api python scripts/merge_faiss.py --rebuild`
- Check disk space for FAISS index

**Missing Models**:
- Verify models directory contains expected files
- Run: `docker exec subgraphrag_api python scripts/download_models.py`

### Getting Help

If you encounter persistent issues:

1. Check the GitHub repository for known issues
2. Consult the troubleshooting section in the documentation
3. Open an issue on GitHub with detailed information about your problem
