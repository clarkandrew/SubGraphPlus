# SubgraphRAG+ Troubleshooting Guide

This concise guide helps resolve common issues with SubgraphRAG+.

## Quick Reference

| Issue Type | Diagnostic Command | Common Fix |
|------------|-------------------|------------|
| Docker Problems | `docker ps \| grep subgraphrag` | `./bin/docker-setup.sh restart` |
| Neo4j Connection | `curl http://localhost:7474` | Check `.env` credentials |
| API Issues | `curl -i http://localhost:8000/healthz` | View logs in `logs/app.log` |
| Model Problems | `ls -la models/` | `python scripts/download_models.py` |

## Common Issues

### Docker Environment

#### Containers Won't Start
- Check Docker: `docker info`
- Check ports: `sudo lsof -i :8000` (also check 7474, 7687)
- Reset volumes: `docker volume rm subgraphrag_neo4j_data subgraphrag_app_data`

#### Container Crashes
- View logs: `docker logs subgraphrag_api`
- Check resources: `docker stats`

### Development Setup

#### Package Installation Fails
- Update pip: `pip install --upgrade pip`
- Fresh environment:
  ```bash
  rm -rf venv
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

#### Server Won't Start
- Check port conflicts: `sudo lsof -i :8000`
- Verify dependencies: `pip install -r requirements.txt`
- Check logs: `cat logs/app.log`

### Neo4j Issues

#### Connection Failures
- Verify running: `docker ps | grep neo4j` or `ps aux | grep neo4j`
- Test connection: `curl http://localhost:7474`
- Check credentials in `.env`:
  ```
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=password
  ```

#### Authentication Problems
- Reset Neo4j password (local): `neo4j-admin set-initial-password newpassword`
- Reset password (Docker): Update `NEO4J_AUTH=neo4j/newpassword` in docker-compose.yml

### API Problems

#### API Key Authentication
- Check header: `curl -H "X-API-KEY: your_key" ...`
- Verify key in `.env`: `API_KEY_SECRET=your_key`

#### Endpoint Not Found
- Check URL format: `curl http://localhost:8000/query`
- Verify server: `curl http://localhost:8000/healthz`

### Data Operations

#### Ingestion Issues
- Check queue: `python scripts/stage_ingest.py --status`
- Process manually: `python scripts/ingest_worker.py --process-all`
- Update FAISS: `python scripts/merge_faiss.py`

#### Duplicate Entities
- Run deduplication: `python scripts/reconcile_stores.py`
- Check with Neo4j Browser: `MATCH (n) WHERE n.name = "EntityName" RETURN n`

### Performance

#### Slow Queries
- Profile Neo4j queries: `PROFILE MATCH (n)-[r]-(m) RETURN n,r,m LIMIT 100`
- Rebuild FAISS: `python scripts/merge_faiss.py --rebuild`
- Increase cache: Set `"CACHE_SIZE": 1000` in config.json

#### Memory Issues
- Use smaller models: Change `MODEL_BACKEND` in config.json
- Limit batch sizes: Set `"INGEST_BATCH_SIZE": 100` in config.json
- For Docker, add memory limits in docker-compose.yml

### Models

#### MLP Model Issues
- Download manually: `python scripts/download_models.py`
- Check path: Verify `MLP_MODEL_PATH` in config.json

#### OpenAI Issues
- Check API key in `.env`: `OPENAI_API_KEY=your_key`
- Switch model: Set `MODEL_BACKEND` to `hf` in config.json

## Error Message Reference

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| "Connection refused" | Neo4j not running | Start Neo4j service |
| "No module named X" | Missing dependency | `pip install X` |
| "FAISS index not found" | Missing index | `python scripts/merge_faiss.py --init` |
| "API key not valid" | Authentication error | Check X-API-KEY header |

## Getting Help

If the above solutions don't work:
1. Check `logs/app.log` for errors
2. Search existing GitHub issues
3. Submit a new issue with:
   - Error messages
   - System details
   - Steps to reproduce