# ⚙️ Configuration Reference

This guide covers all configuration options for SubgraphRAG+, including environment variables, configuration files, and runtime settings.

## 📁 Configuration Files

### Main Configuration

**Location**: `config/config.json`

```json
{
  "MODEL_BACKEND": "openai",
  "FAISS_INDEX_PATH": "data/faiss_index.bin",
  "TOKEN_BUDGET": 4000,
  "MLP_MODEL_PATH": "models/mlp_pretrained.pt",
  "CACHE_DIR": "cache/",
  "MAX_DDE_HOPS": 2,
  "LOG_LEVEL": "INFO",
  "API_RATE_LIMIT": 60,
  "ENABLE_CORS": true,
  "CORS_ORIGINS": ["http://localhost:3000"],
  "MAX_QUERY_LENGTH": 1000,
  "EMBEDDING_CACHE_SIZE": 10000,
  "BATCH_SIZE": 32,
  "MAX_WORKERS": 4,
  "ENABLE_CACHE": true,
  "CACHE_TTL": 3600
}
```

### Schema Configuration

**Location**: `config/config.schema.json`

Defines the JSON schema for validating the main configuration file.

### Aliases Configuration

**Location**: `config/aliases.json`

```json
{
  "entity_aliases": {
    "AI": ["Artificial Intelligence", "Machine Learning", "ML"],
    "ML": ["Machine Learning", "Artificial Intelligence", "AI"]
  },
  "relation_aliases": {
    "is_a": ["type_of", "instance_of", "kind_of"],
    "part_of": ["component_of", "belongs_to"]
  }
}
```

## 🌍 Environment Variables

### Core Database Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEO4J_URI` | Neo4j connection URI | `neo4j://localhost:7687` | ✅ |
| `NEO4J_USER` | Neo4j username | `neo4j` | ✅ |
| `NEO4J_PASSWORD` | Neo4j password | `password` | ✅ |
| `NEO4J_DATABASE` | Neo4j database name | `neo4j` | ❌ |

### API Security

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `API_KEY_SECRET` | Secret key for API authentication | `changeme` | ✅ |
| `JWT_SECRET_KEY` | JWT token secret (if using JWT) | - | ❌ |
| `JWT_ALGORITHM` | JWT algorithm | `HS256` | ❌ |
| `JWT_EXPIRATION_HOURS` | JWT token expiration | `24` | ❌ |

### LLM Integration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | ❌ |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-3.5-turbo` | ❌ |
| `OPENAI_EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-ada-002` | ❌ |
| `HUGGINGFACE_API_KEY` | HuggingFace API key | - | ❌ |
| `ANTHROPIC_API_KEY` | Anthropic API key | - | ❌ |

### MLX Configuration (Apple Silicon)

> **Note**: MLX is only available on Apple Silicon Macs and requires separate installation: `pip install mlx mlx-lm`

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `USE_MLX` | Enable MLX functionality | `false` | ❌ |
| `MLX_LLM_MODEL` | MLX LLM model identifier | `mlx-community/Mistral-7B-Instruct-v0.2-4bit-mlx` | ❌ |
| `MLX_LLM_MODEL_PATH` | Local path to MLX LLM model | - | ❌ |
| `MLX_EMBEDDING_MODEL` | MLX embedding model identifier | `mlx-community/all-MiniLM-L6-v2-mlx` | ❌ |
| `MLX_EMBEDDING_MODEL_PATH` | Local path to MLX embedding model | - | ❌ |
| `EMBEDDING_MODEL` | Standard embedding model (non-MLX) | `all-MiniLM-L6-v2` | ❌ |

**MLX Model Examples:**
- **LLM Models**: `mlx-community/Mistral-7B-Instruct-v0.2-4bit-mlx`, `mlx-community/Meta-Llama-3-8B-Instruct-4bit-mlx`
- **Embedding Models**: `mlx-community/all-MiniLM-L6-v2-mlx`, `mlx-community/bge-small-en-v1.5-mlx`

### Performance Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `WORKERS` | Number of worker processes | `4` | ❌ |
| `MAX_CONNECTIONS` | Maximum database connections | `100` | ❌ |
| `CACHE_SIZE` | Cache size limit | `1000` | ❌ |
| `BATCH_SIZE` | Processing batch size | `32` | ❌ |
| `REQUEST_TIMEOUT` | Request timeout in seconds | `30` | ❌ |

### Logging and Monitoring

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LOG_LEVEL` | Logging level | `INFO` | ❌ |
| `LOG_FORMAT` | Log format | `json` | ❌ |
| `ENABLE_METRICS` | Enable Prometheus metrics | `true` | ❌ |
| `METRICS_PORT` | Metrics server port | `9090` | ❌ |
| `SENTRY_DSN` | Sentry error tracking DSN | - | ❌ |

### Development Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DEBUG` | Enable debug mode | `false` | ❌ |
| `RELOAD` | Enable auto-reload | `false` | ❌ |
| `TESTING` | Enable testing mode | `false` | ❌ |
| `MOCK_MODELS` | Use mock models | `false` | ❌ |

## 🔧 Configuration Examples

### Development Environment

```bash
# .env.development
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true
MOCK_MODELS=true
NEO4J_PASSWORD=dev_password
API_KEY_SECRET=dev_secret_key
CACHE_SIZE=100
WORKERS=1
```

### Production Environment

```bash
# .env.production
DEBUG=false
LOG_LEVEL=INFO
RELOAD=false
MOCK_MODELS=false
NEO4J_PASSWORD=secure_production_password
API_KEY_SECRET=secure_production_secret
OPENAI_API_KEY=your_openai_key
CACHE_SIZE=10000
WORKERS=4
ENABLE_METRICS=true
SENTRY_DSN=your_sentry_dsn
```

### Testing Environment

```bash
# .env.testing
TESTING=true
DEBUG=true
LOG_LEVEL=WARNING
MOCK_MODELS=true
NEO4J_PASSWORD=test_password
API_KEY_SECRET=test_secret
CACHE_SIZE=10
WORKERS=1
```

### MLX Environment (Apple Silicon)

```bash
# .env.mlx
# Enable MLX for Apple Silicon Macs
USE_MLX=true
MODEL_BACKEND=mlx

# MLX LLM Configuration
MLX_LLM_MODEL=mlx-community/Mistral-7B-Instruct-v0.2-4bit-mlx
# MLX_LLM_MODEL_PATH=models/mlx_llm  # Optional: use local model

# MLX Embedding Configuration  
MLX_EMBEDDING_MODEL=mlx-community/all-MiniLM-L6-v2-mlx
# MLX_EMBEDDING_MODEL_PATH=models/mlx_embedding  # Optional: use local model

# Standard settings
NEO4J_PASSWORD=your_password
API_KEY_SECRET=your_secret_key
LOG_LEVEL=INFO
CACHE_SIZE=5000
WORKERS=2
```

## 📊 Model Configuration

### OpenAI Settings

```json
{
  "openai": {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "embedding_model": "text-embedding-ada-002",
    "embedding_dimensions": 1536
  }
}
```

### HuggingFace Settings

```json
{
  "huggingface": {
    "model": "microsoft/DialoGPT-medium",
    "tokenizer": "microsoft/DialoGPT-medium",
    "max_length": 1000,
    "temperature": 0.7,
    "do_sample": true,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dimensions": 384
  }
}
```

### Local Model Settings

```json
{
  "local": {
    "model_path": "models/local_model.bin",
    "tokenizer_path": "models/tokenizer/",
    "device": "cuda",
    "batch_size": 16,
    "max_length": 512
  }
}
```

## 🗄️ Database Configuration

### Neo4j Settings

```bash
# Neo4j specific environment variables
NEO4J_dbms_memory_heap_initial__size=1G
NEO4J_dbms_memory_heap_max__size=2G
NEO4J_dbms_memory_pagecache_size=1G
NEO4J_dbms_default__listen__address=0.0.0.0
NEO4J_dbms_connector_bolt_listen__address=0.0.0.0:7687
NEO4J_dbms_connector_http_listen__address=0.0.0.0:7474
```

### Connection Pool Settings

```json
{
  "neo4j_pool": {
    "max_connection_lifetime": 3600,
    "max_connection_pool_size": 50,
    "connection_acquisition_timeout": 60,
    "connection_timeout": 30,
    "max_retry_time": 30
  }
}
```

### SQLite Settings

```json
{
  "sqlite": {
    "database_path": "data/staging.db",
    "timeout": 30,
    "check_same_thread": false,
    "isolation_level": null
  }
}
```

## 🚀 Performance Tuning

### Memory Settings

```json
{
  "memory": {
    "faiss_index_memory_limit": "2GB",
    "embedding_cache_size": 10000,
    "query_cache_size": 1000,
    "result_cache_ttl": 3600
  }
}
```

### Concurrency Settings

```json
{
  "concurrency": {
    "max_workers": 4,
    "max_concurrent_requests": 100,
    "request_queue_size": 1000,
    "worker_timeout": 300
  }
}
```

### Caching Configuration

```json
{
  "cache": {
    "enabled": true,
    "backend": "redis",
    "redis_url": "redis://localhost:6379/0",
    "default_ttl": 3600,
    "max_size": "1GB"
  }
}
```

## 🔒 Security Configuration

### API Security

```json
{
  "security": {
    "api_key_header": "X-API-KEY",
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 60,
      "burst_size": 10
    },
    "cors": {
      "enabled": true,
      "origins": ["http://localhost:3000"],
      "methods": ["GET", "POST"],
      "headers": ["Content-Type", "X-API-KEY"]
    }
  }
}
```

### Data Protection

```json
{
  "data_protection": {
    "encrypt_sensitive_data": true,
    "encryption_key_env": "DATA_ENCRYPTION_KEY",
    "mask_pii_in_logs": true,
    "audit_logging": true
  }
}
```

## 📝 Logging Configuration

### Log Levels

- `DEBUG`: Detailed information for debugging
- `INFO`: General information about system operation
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for failures
- `CRITICAL`: Critical errors that may cause system failure

### Log Formats

```json
{
  "logging": {
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
      "json": {
        "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
        "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
      },
      "standard": {
        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
      }
    },
    "handlers": {
      "console": {
        "class": "logging.StreamHandler",
        "formatter": "standard",
        "level": "INFO"
      },
      "file": {
        "class": "logging.handlers.RotatingFileHandler",
        "filename": "logs/app.log",
        "maxBytes": 10485760,
        "backupCount": 5,
        "formatter": "json",
        "level": "DEBUG"
      }
    },
    "root": {
      "level": "INFO",
      "handlers": ["console", "file"]
    }
  }
}
```

## 🔄 Configuration Validation

### Schema Validation

The system automatically validates configuration against the JSON schema in `config/config.schema.json`. Invalid configurations will prevent startup.

### Environment Variable Validation

```python
# Example validation rules
REQUIRED_VARS = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "API_KEY_SECRET"]
OPTIONAL_VARS = ["OPENAI_API_KEY", "DEBUG", "LOG_LEVEL"]

# Type validation
INT_VARS = ["WORKERS", "CACHE_SIZE", "API_RATE_LIMIT"]
BOOL_VARS = ["DEBUG", "ENABLE_METRICS", "ENABLE_CORS"]
```

## 🛠️ Configuration Management

### Loading Order

1. Default values from code
2. Configuration file (`config/config.json`)
3. Environment variables (override config file)
4. Command-line arguments (override environment variables)

### Dynamic Configuration

Some settings can be updated at runtime:

```python
# Update cache size
POST /admin/config
{
  "cache_size": 5000
}

# Update log level
POST /admin/config
{
  "log_level": "DEBUG"
}
```

### Configuration Backup

```bash
# Backup current configuration
make backup

# Restore configuration
make restore backup_20240101_120000
```

## 🧪 Testing Configuration

### Test Environment Variables

```bash
# Minimal test configuration
export TESTING=true
export NEO4J_PASSWORD=test
export API_KEY_SECRET=test
export MOCK_MODELS=true
```

### Configuration for CI/CD

```yaml
# GitHub Actions example
env:
  TESTING: true
  NEO4J_PASSWORD: test
  API_KEY_SECRET: test
  MOCK_MODELS: true
  LOG_LEVEL: WARNING
```

## 🔍 Troubleshooting Configuration

### Common Issues

1. **Missing required environment variables**
   ```bash
   # Check required variables
   make env-check
   ```

2. **Invalid configuration format**
   ```bash
   # Validate configuration
   python -c "from app.config import Config; Config()"
   ```

3. **Permission issues with config files**
   ```bash
   # Fix permissions
   chmod 644 config/*.json
   ```

### Debug Configuration

```bash
# Print current configuration
python -c "from app.config import config; import json; print(json.dumps(config.dict(), indent=2))"

# Check environment variables
env | grep -E "(NEO4J|API|OPENAI|DEBUG)"
```

---

**💡 Pro Tip**: Use environment-specific `.env` files and never commit sensitive values to version control. Use tools like `python-dotenv` to load environment variables from files during development.** 