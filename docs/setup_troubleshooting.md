# Setup Troubleshooting Guide

## Common Setup Issues

### Setup Script Hangs at "Starting ingest worker"

**Problem**: The development setup script hangs after displaying "Starting ingest worker" and never completes.

**Root Cause**: This is typically caused by one of the following issues:

1. **Large model download**: The `Alibaba-NLP/gte-large-en-v1.5` embedding model (434MB) takes time to download on first use
2. **Model loading timeout**: The model initialization can take several minutes, especially on slower machines
3. **Configuration mismatch**: Conflicting settings between `.env` and `config/config.json`

**Solutions**:

#### Option 1: Use the Fixed Setup Script (Recommended)
The updated `bin/setup_dev.sh` includes a 10-minute timeout for the ingest worker:

```bash
./bin/setup_dev.sh --skip-tests
```

If it times out, you can manually run the ingest later:
```bash
source venv/bin/activate
python scripts/ingest_worker.py --process-all
```

#### Option 2: Test Embedder First
Before running the full setup, test if the embedder works:

```bash
source venv/bin/activate
python scripts/test_embedder.py
```

This will identify embedding issues early without running the full setup.

#### Option 3: Skip Sample Data
Skip the sample data loading during setup:

```bash
./bin/setup_dev.sh --skip-sample-data --skip-tests
```

Then load sample data manually later:
```bash
source venv/bin/activate
python scripts/stage_ingest.py --sample
python scripts/ingest_worker.py --process-all
python scripts/merge_faiss.py
```

#### Option 4: Use Different Backend
Temporarily switch to a smaller embedding model by editing `.env`:

```bash
# Change this:
EMBEDDING_MODEL=Alibaba-NLP/gte-large-en-v1.5

# To this (smaller, faster model):
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**Note**: Changing the embedding model may affect MLP model compatibility.

### Configuration Mismatch

**Problem**: The setup logs show conflicting backend configurations.

**Root Cause**: The `MODEL_BACKEND` setting in `.env` differs from `config/config.json`.

**Solution**: Ensure consistency between files:

`.env`:
```bash
MODEL_BACKEND=hf
```

`config/config.json`:
```json
{
  "MODEL_BACKEND": "hf",
  ...
}
```

### Model Download Issues

**Problem**: Slow or failed model downloads.

**Solutions**:

1. **Use faster internet connection** for initial setup
2. **Pre-download models** using Hugging Face CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli download Alibaba-NLP/gte-large-en-v1.5
   ```

3. **Use local model cache** by setting `HF_HOME`:
   ```bash
   export HF_HOME=/path/to/your/model/cache
   ```

### Memory Issues

**Problem**: System runs out of memory during model loading.

**Solutions**:

1. **Close other applications** before running setup
2. **Use MLX backend** on Apple Silicon (more memory efficient):
   ```bash
   USE_MLX_LLM=true
   MODEL_BACKEND=mlx
   ```

3. **Use OpenAI backend** (no local model required):
   ```bash
   MODEL_BACKEND=openai
   OPENAI_API_KEY=your_api_key_here
   ```

## Verification Steps

After setup completes, verify everything works:

1. **Test embedder**:
   ```bash
   python scripts/test_embedder.py
   ```

2. **Check database connections**:
   ```bash
   python -c "from src.app.database import neo4j_db, sqlite_db; print('Neo4j:', neo4j_db.health_check()); print('SQLite: OK')"
   ```

3. **Verify sample data**:
   ```bash
   sqlite3 data/staging.db "SELECT COUNT(*) FROM staging_triples;"
   ```

4. **Test API**:
   ```bash
   source venv/bin/activate
   python src/main.py --reload &
   curl http://localhost:8000/health
   ```

## Getting Help

If issues persist:

1. **Check logs**: Look in `logs/ingest_worker.log` for detailed error messages
2. **Run with debug**: Set `LOG_LEVEL=DEBUG` in `.env`
3. **Check system resources**: Ensure sufficient RAM and disk space
4. **Verify dependencies**: Run `pip list` to check installed packages

## Environment-Specific Issues

### Apple Silicon (M1/M2/M3)
- **Use MLX**: Set `USE_MLX_LLM=true` for better performance
- **Install MLX**: `pip install mlx mlx-lm`

### Linux/Ubuntu
- **Install system dependencies**: `sudo apt-get install build-essential`
- **Use system Python**: Avoid conda environments if possible

### Windows
- **Use WSL2**: Recommended for better compatibility
- **Long path support**: Enable long path support in Windows 