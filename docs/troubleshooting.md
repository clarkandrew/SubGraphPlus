# Troubleshooting Guide for SubgraphRAG+

This guide will help you diagnose and resolve common issues you might encounter when working with SubgraphRAG+.

## Table of Contents
- [Docker Environment Issues](#docker-environment-issues)
- [Local Development Issues](#local-development-issues)
- [Neo4j Connection Problems](#neo4j-connection-problems)
- [API Issues](#api-issues)
- [Data Ingestion Problems](#data-ingestion-problems)
- [Performance Issues](#performance-issues)
- [Model Loading Issues](#model-loading-issues)
- [Common Error Messages](#common-error-messages)

## Docker Environment Issues

### Containers Won't Start

**Symptoms:**
- `./bin/docker-setup.sh start` fails
- Containers exit immediately after starting

**Possible Solutions:**
1. Check Docker service status:
   ```bash
   docker info
   ```

2. View detailed logs:
   ```bash
   docker logs subgraphrag_api
   docker logs subgraphrag_neo4j
   ```

3. Ensure ports aren't already in use:
   ```bash
   sudo lsof -i :8000
   sudo lsof -i :7474
   sudo lsof -i :7687
   ```

4. Check Docker disk space:
   ```bash
   docker system df
   ```

5. Try with clean volumes:
   ```bash
   ./bin/docker-setup.sh stop
   docker volume rm subgraphrag_neo4j_data subgraphrag_app_data
   ./bin/docker-setup.sh start
   ```

### Container Exited With Error

**Symptoms:**
- Container starts but exits with an error code
- You see "Exited (1)" in `docker ps -a`

**Possible Solutions:**
1. Check container logs:
   ```bash
   docker logs subgraphrag_api
   ```

2. Check for environment variable issues:
   ```bash
   docker exec subgraphrag_api env
   ```

3. Ensure container has sufficient resources:
   ```bash
   docker stats
   ```

## Local Development Issues

### Package Installation Failures

**Symptoms:**
- `pip install` commands fail
- Import errors when running Python scripts

**Possible Solutions:**
1. Update pip:
   ```bash
   pip install --upgrade pip
   ```

2. Check for conflicting packages:
   ```bash
   pip check
   ```

3. Try with a fresh virtual environment:
   ```bash
   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. Install system dependencies (Ubuntu/Debian):
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential libsqlite3-dev python3-dev
   ```

### Server Won't Start

**Symptoms:**
- `python main.py` crashes or exits with an error
- Server doesn't respond on localhost:8000

**Possible Solutions:**
1. Check for port conflicts:
   ```bash
   sudo lsof -i :8000
   ```

2. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. Check environment variables:
   ```bash
   cat .env
   ```

4. Look for errors in the log files:
   ```bash
   cat logs/app.log
   ```

## Neo4j Connection Problems

### Can't Connect to Neo4j

**Symptoms:**
- "Connection refused" errors
- "Unable to connect to Neo4j" messages

**Possible Solutions:**
1. Ensure Neo4j is running:
   ```bash
   # For Docker
   docker ps | grep neo4j
   
   # For local install
   ps aux | grep neo4j
   ```

2. Check Neo4j connection settings:
   ```bash
   # In .env file
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=password
   ```

3. Test connection with another tool:
   ```bash
   curl http://localhost:7474
   ```

4. Check firewall settings:
   ```bash
   sudo ufw status
   ```

### Authentication Failures

**Symptoms:**
- "Invalid credentials" errors
- "Authorization failed" messages

**Possible Solutions:**
1. Verify credentials in .env or config.json:
   ```
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=password
   ```

2. Reset Neo4j password (for local installation):
   ```bash
   neo4j-admin set-initial-password newpassword
   ```

3. For Docker, update password and recreate container:
   ```bash
   # In docker-compose.yml
   environment:
     - NEO4J_AUTH=neo4j/newpassword
   ```

## API Issues

### API Key Authentication Failures

**Symptoms:**
- 401 Unauthorized responses
- "Invalid API key" messages

**Possible Solutions:**
1. Check the API key in your request headers:
   ```bash
   curl -H "X-API-KEY: your_api_key" ...
   ```

2. Verify API key configuration:
   ```
   # In .env file
   API_KEY_SECRET=your_api_key
   ```

3. For Docker, check the API key in the container:
   ```bash
   docker exec subgraphrag_api bash -c 'echo $API_KEY_SECRET'
   ```

### API Endpoint Not Found

**Symptoms:**
- 404 Not Found responses
- "Not Found" or "URL not found" messages

**Possible Solutions:**
1. Check URL path:
   ```bash
   # Correct format
   curl http://localhost:8000/query
   ```

2. Ensure server is running:
   ```bash
   curl http://localhost:8000/healthz
   ```

3. Check server logs for routing errors:
   ```bash
   cat logs/app.log
   ```

## Data Ingestion Problems

### Triples Not Being Ingested

**Symptoms:**
- Ingested data doesn't appear in queries
- No error messages, but data is missing

**Possible Solutions:**
1. Check ingestion queue status:
   ```bash
   python scripts/stage_ingest.py --status
   ```

2. Check for validation errors:
   ```bash
   sqlite3 data/staging.db "SELECT * FROM error_log ORDER BY timestamp DESC LIMIT 10;"
   ```

3. Process the ingestion queue manually:
   ```bash
   python scripts/ingest_worker.py --process-all
   ```

4. Ensure FAISS index is updated:
   ```bash
   python scripts/merge_faiss.py
   ```

### Duplicate Entities

**Symptoms:**
- Same entity appears multiple times in results
- Redundant nodes in graph visualization

**Possible Solutions:**
1. Check alias configuration:
   ```bash
   cat config/aliases.json
   ```

2. Run the deduplication process:
   ```bash
   python scripts/reconcile_stores.py
   ```

3. Verify with Neo4j Browser:
   ```cypher
   MATCH (n) WHERE n.name = "EntityName" RETURN n
   ```

## Performance Issues

### Slow Query Performance

**Symptoms:**
- Queries take a long time to complete
- Timeouts or very slow responses

**Possible Solutions:**
1. Check Neo4j query performance:
   ```bash
   # In Neo4j Browser
   PROFILE MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 100
   ```

2. Optimize FAISS index:
   ```bash
   python scripts/rebuild_faiss_index.py --optimize
   ```

3. Increase cache size:
   ```bash
   # In config/config.json
   "CACHE_SIZE": 1000
   ```

4. Check hardware resources:
   ```bash
   # For Docker
   docker stats
   
   # For local
   top
   ```

### High Memory Usage

**Symptoms:**
- System becomes sluggish
- "Out of memory" errors

**Possible Solutions:**
1. Reduce model size:
   ```bash
   # In config/config.json
   "MODEL_BACKEND": "hf",
   "HF_MODEL": "smaller-model-name"
   ```

2. Limit batch sizes:
   ```bash
   # In config/config.json
   "INGEST_BATCH_SIZE": 100
   ```

3. For Docker, increase container memory limit:
   ```yaml
   # In docker-compose.yml
   services:
     subgraphrag:
       deploy:
         resources:
           limits:
             memory: 4G
   ```

## Model Loading Issues

### Can't Load MLP Model

**Symptoms:**
- "Model file not found" errors
- "Failed to load pre-trained model" messages

**Possible Solutions:**
1. Download the model manually:
   ```bash
   ./bin/setup.sh --models-only
   ```

2. Check model path configuration:
   ```bash
   # In config/config.json
   "MLP_MODEL_PATH": "models/mlp_pretrained.pt"
   ```

3. Check permissions on model directory:
   ```bash
   ls -la models/
   ```

### OpenAI API Issues

**Symptoms:**
- "API key not found" errors
- OpenAI request failures

**Possible Solutions:**
1. Verify API key:
   ```bash
   # In .env file
   OPENAI_API_KEY=your_key_here
   ```

2. Check for API key environment variable:
   ```bash
   echo $OPENAI_API_KEY
   ```

3. Switch to a local model backend:
   ```bash
   # In config/config.json
   "MODEL_BACKEND": "hf"
   ```

## Common Error Messages

### "Failed to connect to Neo4j: Connection refused"

**Possible Solutions:**
1. Ensure Neo4j is running
2. Check if the port is correct (default: 7687)
3. See if there's a firewall blocking the connection
4. Verify that Neo4j is configured to accept external connections

### "No module named X"

**Possible Solutions:**
1. Install the missing package:
   ```bash
   pip install X
   ```
2. Make sure you're in the correct virtual environment
3. Check if the package is listed in requirements.txt

### "FAISS index not found"

**Possible Solutions:**
1. Initialize the FAISS index:
   ```bash
   python scripts/merge_faiss.py --init
   ```
2. Check the index path in config/config.json
3. Look for disk space issues or permission problems

### "MLFlow model not found"

**Possible Solutions:**
1. Download the models:
   ```bash
   python scripts/download_models.py
   ```
2. Check model registry paths
3. Verify that model filenames match the expected names

## Further Assistance

If you're still experiencing issues after trying the solutions in this guide:

1. Check the logs in the `logs/` directory
2. Look for similar issues in the GitHub repository issues section
3. Post a detailed description of your problem, including:
   - Error messages
   - System specifications
   - Steps to reproduce
   - Configuration details (sanitized of secrets)

For urgent production issues, contact the maintainers directly through the channels listed in the support documentation.