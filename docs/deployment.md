# SubgraphRAG+ Deployment Guide

This document provides concise instructions for deploying SubgraphRAG+ in different environments.

## Components Overview

SubgraphRAG+ consists of these core components:

- **Neo4j**: Graph database (stores knowledge triples)
- **FastAPI**: API server (provides query and ingestion endpoints)
- **SQLite**: Staging database (manages ingestion queue)
- **FAISS**: Vector index (enables semantic search)
- **ML Models**: Pre-trained models (triple scoring, LLM integration)

## Docker Deployment (Recommended)

### Quick Start

```bash
# Clone and set up
git clone https://github.com/clarkandrew/SubgraphRAGPlus.git
cd SubgraphRAGPlus
chmod +x bin/*.sh

# Start system with Docker
make docker-start
# OR
./bin/docker-setup.sh start

# Load sample data (after system starts)
make ingest-sample
# OR
./bin/docker-setup.sh sample-data
```

### Configuration

1. Edit `docker-compose.yml` environment variables:
   - `API_KEY_SECRET`: Set a strong secret key
   - `OPENAI_API_KEY`: Add if using OpenAI
   - `NEO4J_PASSWORD`: Change default password

2. Persistent storage uses Docker volumes:
   - `neo4j_data`: Neo4j database
   - `app_data`: Application data
   - `app_models`: ML models
   - `app_cache`: Cache storage
   - `app_logs`: Application logs

3. Backup and restore:
   ```bash
   # Create backup
   ./bin/backup.sh create
   
   # List backups
   ./bin/backup.sh list
   
   # Restore specific backup
   ./bin/backup.sh restore backup_20230101_120000
   ```

## Local Neo4j Deployment

For development without Docker, use our automated installer or set up Neo4j manually.

### Automated Installation

```bash
# Install Neo4j locally with our script
./bin/install_neo4j.sh

# Options:
./bin/install_neo4j.sh --version 4.4.30 --sudo
```

### Manual Installation Options

#### Neo4j Desktop (Recommended for development)
1. Download from [neo4j.com/download](https://neo4j.com/download/)
2. Create a database (4.4+)
3. Install APOC plugin via Plugins tab
4. Set secure password

#### Package Managers

**macOS:**
```bash
brew install neo4j
brew services start neo4j
cypher-shell -u neo4j -p neo4j  # Set password
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install neo4j
sudo systemctl start neo4j
```

### Required Configuration

Edit Neo4j configuration (varies by installation method):
```
dbms.connector.bolt.enabled=true
dbms.connector.bolt.listen_address=0.0.0.0:7687
dbms.connector.http.enabled=true
dbms.connector.http.listen_address=0.0.0.0:7474
dbms.security.procedures.unrestricted=apoc.*
```

Set up `.env` file:
```
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
USE_LOCAL_NEO4J=true
```

## Local Development Setup

```bash
# Clone repository
git clone https://github.com/clarkandrew/SubgraphRAGPlus.git
cd SubgraphRAGPlus

# Make scripts executable
chmod +x bin/*.sh

# Option 1: Setup with local Neo4j
./bin/setup_dev.sh --use-local-neo4j

# Option 2: Setup with Docker Neo4j
./bin/setup_dev.sh

# Start the API server
./bin/run.sh

# Load sample data
python scripts/stage_ingest.py --sample
python scripts/ingest_worker.py --process-all
python scripts/merge_faiss.py

# Run tests
./bin/run_tests.sh
```

Configuration files:
- `.env`: Environment variables
- `config/config.json`: Application settings

## Production Deployment

This section covers essential considerations for production deployment.

### Security Checklist

1. **API Security:**
   - Generate a strong random API key
   - Implement rate limiting
   - Use HTTPS/TLS with a reverse proxy

2. **Network Security:**
   - Isolate components with Docker networks
   - Restrict Neo4j access to internal network
   - Configure proper firewall rules

3. **Data Security:**
   - Encrypt sensitive data
   - Schedule regular backups
   - Rotate credentials periodically

### High Availability

1. **Neo4j Clustering** (Enterprise):
   - Core servers for writes (3+ recommended)
   - Read replicas for scaling queries

2. **API Scaling:**
   - Multiple API instances behind load balancer
   - Docker Swarm or Kubernetes orchestration

3. **Resource Recommendations:**
   - Neo4j: 16GB+ RAM, SSD storage
   - API: 4GB+ RAM
   - Regular backups with retention policy

### Production Docker-Compose Example

```yaml
services:
  traefik:
    image: traefik:v2.10
    command:
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.myresolver.acme.tlschallenge=true"
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
    environment:
      - NEO4J_AUTH=neo4j/strong-password-here
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_memory_heap_max__size=4G
    volumes:
      - neo4j_data:/data
    networks:
      - internal-net
    restart: unless-stopped

  api:
    build: .
    depends_on:
      - neo4j
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_PASSWORD=strong-password-here
      - API_KEY_SECRET=very-strong-api-key-here
    volumes:
      - app_data:/app/data
      - app_models:/app/models
    networks:
      - internal-net
      - traefik-public
    restart: unless-stopped
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=Host(`api.yourdomain.com`)"
      - "traefik.http.routers.api.tls.certresolver=myresolver"

networks:
  internal-net:
    internal: true
  traefik-public:
    external: true

volumes:
  neo4j_data:
  app_data:
  app_models:
  letsencrypt:
```

## Monitoring and Maintenance

### Health Monitoring

SubgraphRAG+ provides built-in monitoring endpoints:

- `/healthz`: Basic application health (liveness probe)
- `/readyz`: Dependency health checks (Neo4j, SQLite, FAISS, LLM)
- `/metrics`: Prometheus-compatible metrics

### Maintenance Tasks

1. **Scheduled Backups:**
   ```bash
   # Automated backup script - recommended for cron jobs
   ./bin/backup.sh create --name scheduled-backup
   ```

2. **Database Maintenance:**
   ```bash
   # Check and optimize Neo4j indexes
   ./bin/neo4j-maintenance.sh optimize-indexes
   
   # Rebuild FAISS index (if search performance degrades)
   python scripts/merge_faiss.py --rebuild
   ```

3. **Log Management:**
   - Configure log rotation for all components
   - Implement log aggregation (ELK/Graylog)
   - Monitor for error patterns

### Security Best Practices

1. **API Security:**
   - Rotate API keys every 90 days
   - Use environment variables for secrets
   - Implement rate limiting (60 requests/minute default)
   - Log all authentication failures
   
2. **Data Protection:**
   - Set proper Neo4j authentication
   - Encrypt sensitive data
   - Implement retention policies
   - Secure backup encryption

3. **Access Control:**
   - Restrict Neo4j to internal network
   - Use TLS for all communications
   - Implement network segmentation
   - Audit logging for sensitive operations

## Troubleshooting

### Quick Diagnostic Commands

```bash
# Check system status
./bin/system-check.sh

# View logs
tail -n 100 logs/app.log

# Test Neo4j connection
./bin/test-connection.sh neo4j

# Test API
curl -i http://localhost:8000/healthz
```

### Common Issues and Solutions

#### Neo4j Connection Problems

**Docker Setup:**
```bash
# Check Neo4j container
docker logs subgraphrag_neo4j
docker ps | grep neo4j

# Restart Neo4j
make neo4j-restart
```

**Local Installation:**
```bash
# Check Neo4j status
brew services list | grep neo4j    # macOS
systemctl status neo4j            # Linux

# View Neo4j logs
cat /usr/local/var/log/neo4j/neo4j.log  # macOS
journalctl -u neo4j                     # Linux
```

#### API Issues

- Check logs: `cat logs/app.log`
- Verify environment variables in `.env`
- Test endpoint: `curl -i http://localhost:8000/healthz`
- Restart server: `./bin/run.sh`

#### Data Problems

- Rebuild FAISS index: `python scripts/merge_faiss.py --rebuild`
- Download missing models: `python scripts/download_models.py`
- Check disk space: `df -h`
- Verify permissions: `ls -la data/`

### Getting Help

Report issues on GitHub with:
- Error messages and logs
- System information
- Steps to reproduce
- Configuration (remove sensitive data)
