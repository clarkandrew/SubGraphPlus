# ⚙️ Configuration Reference

This guide covers the complete configuration system for SubgraphRAG+, which follows security best practices by separating secrets from application settings.

## 🏗️ Configuration Architecture

SubgraphRAG+ uses a **hybrid configuration approach**:

- **`.env`** - Secrets and environment-specific values (never commit to version control)
- **`config/config.json`** - Application settings and model configurations (version controlled)
- **`config/config.schema.json`** - JSON schema for validation
- **`config/aliases.json`** - Entity and relation aliases for improved matching

## 📁 Configuration Files

### 🔒 Environment Variables (.env)

**Location**: `.env` (root directory)

Contains secrets and environment-specific settings that should **never be committed to version control**.

```bash
# === Database Credentials ===
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-password

# === API Security ===
API_KEY_SECRET=your-secure-api-key

# === API Keys ===
OPENAI_API_KEY=sk-your-openai-key  # Required if using OpenAI backend
HF_TOKEN=hf_your-token             # Optional, for private HuggingFace models

# === Environment Settings ===
ENVIRONMENT=development            # development|staging|production
LOG_LEVEL=INFO                     # DEBUG|INFO|WARNING|ERROR|CRITICAL
DEBUG=false

# === Optional: Custom Model Paths ===
# MLX_LLM_MODEL_PATH=/path/to/custom/mlx/model
# HF_MODEL_PATH=/path/to/custom/hf/model

# === Optional: Logging ===
# LOG_FILE=logs/app.log
```

### ⚙️ Application Configuration (config.json)

**Location**: `config/config.json`

Contains application settings, model configurations, and performance tuning parameters.

```json
{
  "application": {
    "name": "SubgraphRAG+",
    "version": "1.0.0",
    "description": "Enhanced RAG system with subgraph retrieval and multi-modal ML capabilities"
  },
  "models": {
    "backend": "mlx",
    "llm": {
      "mlx": {
        "model": "mlx-community/Qwen3-14B-8bit",
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.9
      },
      "openai": {
        "model": "gpt-3.5-turbo",
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.9
      },
      "huggingface": {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.9
      }
    },
    "embeddings": {
      "model": "Alibaba-NLP/gte-large-en-v1.5",
      "backend": "transformers",
      "cache_dir": "models/embeddings/"
    },
    "mlp": {
      "model_path": "models/mlp/mlp.pth"
    }
  },
  "data": {
    "faiss_index_path": "data/faiss_index.bin",
    "neo4j": {
      "default_database": "neo4j"
    }
  },
  "retrieval": {
    "token_budget": 4000,
    "max_dde_hops": 2,
    "similarity_threshold": 0.7,
    "max_results": 10
  },
  "performance": {
    "cache_size": 1000,
    "ingest_batch_size": 100,
    "api_rate_limit": 60,
    "timeout_seconds": 30
  },
  "paths": {
    "cache_dir": "cache/",
    "models_dir": "models/",
    "data_dir": "data/",
    "logs_dir": "logs/"
  }
}
```

### 📋 Schema Configuration (config.schema.json)

**Location**: `config/config.schema.json`

JSON schema for validating the main configuration file:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["application", "models", "data", "retrieval"],
  "properties": {
    "models": {
      "type": "object",
      "required": ["backend", "llm", "embeddings"],
      "properties": {
        "backend": {
          "type": "string",
          "enum": ["mlx", "openai", "huggingface"]
        },
        "embeddings": {
          "properties": {
            "backend": {
              "enum": ["transformers"],
              "description": "Always use transformers for embeddings"
            }
          }
        }
      }
    }
  }
}
```

### 🏷️ Aliases Configuration (aliases.json)

**Location**: `config/aliases.json`

Entity and relation aliases for improved matching:

```json
{
  "entity_aliases": {
    "AI": ["Artificial Intelligence", "Machine Learning", "ML"],
    "ML": ["Machine Learning", "Artificial Intelligence", "AI"],
    "NLP": ["Natural Language Processing", "Language Processing"]
  },
  "relation_aliases": {
    "is_a": ["type_of", "instance_of", "kind_of"],
    "part_of": ["component_of", "belongs_to"],
    "related_to": ["associated_with", "connected_to"]
  }
}
```

## 🌍 Environment Variables Reference

### Database Credentials

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEO4J_URI` | Neo4j connection URI | `neo4j://localhost:7687` | ✅ |
| `NEO4J_USER` | Neo4j username | `neo4j` | ✅ |
| `NEO4J_PASSWORD` | Neo4j password | `password` | ✅ |

### API Security

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `API_KEY_SECRET` | Secret key for API authentication | `default_key_for_dev_only` | ✅ |

### API Keys

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key (required if using OpenAI backend) | - | ❌ |
| `HF_TOKEN` | HuggingFace API token (for private models) | - | ❌ |

### Environment Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Environment type | `development` | ❌ |
| `LOG_LEVEL` | Logging level | `INFO` | ❌ |
| `DEBUG` | Enable debug mode | `false` | ❌ |
| `LOG_FILE` | Log file path (optional) | - | ❌ |

### Custom Model Paths

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MLX_LLM_MODEL_PATH` | Local path to MLX LLM model | - | ❌ |
| `HF_MODEL_PATH` | Local path to HuggingFace model | - | ❌ |

## 🧠 Model Configuration

### LLM Backend Selection

Configure the primary LLM backend in `config.json`:

```json
{
  "models": {
    "backend": "mlx"  // Options: "mlx", "openai", "huggingface"
  }
}
```

**Backend Options**:
- **`mlx`** - Apple Silicon optimized (M1/M2/M3 Macs)
- **`openai`** - OpenAI API (requires `OPENAI_API_KEY`)
- **`huggingface`** - Local HuggingFace transformers

### Embedding Configuration

> **⚠️ CRITICAL**: Embeddings **always use transformers**, never MLX. The embedding model must match the model used to train the MLP.

```json
{
  "models": {
    "embeddings": {
      "model": "Alibaba-NLP/gte-large-en-v1.5",
      "backend": "transformers",  // Always "transformers"
      "cache_dir": "models/embeddings/"
    }
  }
}
```

**Supported Embedding Models**:
- `Alibaba-NLP/gte-large-en-v1.5` (default, 1024 dimensions)
- `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- `sentence-transformers/all-mpnet-base-v2` (768 dimensions)

### MLX Configuration (Apple Silicon)

For optimal performance on M1/M2/M3 Macs:

```json
{
  "models": {
    "backend": "mlx",
    "llm": {
      "mlx": {
        "model": "mlx-community/Qwen3-14B-8bit",
        "max_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.9
      }
    },
    "embeddings": {
      "model": "Alibaba-NLP/gte-large-en-v1.5",
      "backend": "transformers"  // Never use MLX for embeddings
    }
  }
}
```

**Recommended MLX Models**:
- `mlx-community/Qwen3-14B-8bit` (current default)
- `mlx-community/Mistral-7B-Instruct-v0.2-4bit-mlx`
- `mlx-community/Meta-Llama-3-8B-Instruct-4bit-mlx`

## 🔧 Configuration Examples

### Development Environment

```bash
# .env
NEO4J_PASSWORD=dev_password
API_KEY_SECRET=dev_secret_key
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true
```

```json
// config/config.json
{
  "models": {
    "backend": "mlx",
    "llm": {
      "mlx": {"model": "mlx-community/Qwen3-14B-8bit"}
    }
  },
  "performance": {
    "cache_size": 100,
    "api_rate_limit": 10
  }
}
```

### Production Environment

```bash
# .env
NEO4J_URI=neo4j+s://your-production-instance.neo4j.io
NEO4J_PASSWORD=secure_production_password
API_KEY_SECRET=secure_production_secret
OPENAI_API_KEY=sk-your-production-key
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false
```

```json
// config/config.json
{
  "models": {
    "backend": "openai",
    "llm": {
      "openai": {
        "model": "gpt-4",
        "max_tokens": 1024
      }
    }
  },
  "performance": {
    "cache_size": 10000,
    "api_rate_limit": 100,
    "timeout_seconds": 60
  }
}
```

### Testing Environment

```bash
# .env.testing
TESTING=true
NEO4J_PASSWORD=test_password
API_KEY_SECRET=test_secret
LOG_LEVEL=WARNING
DEBUG=true
```

## 🔑 Best Practices

### Security

1. **Never commit `.env`** - Add to `.gitignore`
2. **Use strong secrets** - Generate secure API keys and passwords
3. **Rotate credentials** - Regularly update API keys and passwords
4. **Environment separation** - Use different credentials per environment

### Model Configuration

1. **Embedding consistency** - Never change the embedding model after MLP training
2. **Backend separation** - Use MLX for LLM only, transformers for embeddings only
3. **Resource management** - Adjust cache sizes based on available memory
4. **Performance tuning** - Test different model parameters for your use case

### Monitoring

1. **Log levels** - Use `DEBUG` in development, `INFO` in production
2. **Health checks** - Monitor `/healthz` and `/readyz` endpoints
3. **Metrics** - Enable Prometheus metrics for production monitoring
4. **Error tracking** - Consider integrating with Sentry for error reporting

## 🚨 Troubleshooting

### Common Configuration Issues

**MLX not working on Apple Silicon**:
```bash
# Install MLX dependencies
pip install mlx mlx-lm

# Verify in .env
LOG_LEVEL=DEBUG  # To see MLX logs
```

**Embedding model mismatch**:
```json
// Ensure this matches your MLP training model
{
  "models": {
    "embeddings": {
      "model": "Alibaba-NLP/gte-large-en-v1.5",
      "backend": "transformers"
    }
  }
}
```

**Database connection issues**:
```bash
# Check .env settings
NEO4J_URI=neo4j://localhost:7687  # or neo4j+s:// for cloud
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

For more troubleshooting help, see [Troubleshooting Guide](troubleshooting.md). 