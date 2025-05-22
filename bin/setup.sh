#!/bin/bash
# Script to setup the SubgraphRAG+ environment
# Created as part of the implementation of the TODO list

# Set script to exit on error
set -e

# Define colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "=================================================="
echo "           SubgraphRAG+ Environment Setup         "
echo "=================================================="
echo -e "${NC}"

# Parse command line arguments
INSTALL_DEPS=true
CREATE_VENV=true
SETUP_NEO4J=true
DOWNLOAD_MODELS=true
INITIALIZE_DB=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-deps)
      INSTALL_DEPS=false
      shift
      ;;
    --skip-venv)
      CREATE_VENV=false
      shift
      ;;
    --skip-neo4j)
      SETUP_NEO4J=false
      shift
      ;;
    --skip-models)
      DOWNLOAD_MODELS=false
      shift
      ;;
    --skip-db-init)
      INITIALIZE_DB=false
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --skip-deps       Skip dependencies installation"
      echo "  --skip-venv       Skip virtual environment creation"
      echo "  --skip-neo4j      Skip Neo4j setup"
      echo "  --skip-models     Skip model download"
      echo "  --skip-db-init    Skip database initialization"
      echo "  -h, --help        Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create required directories
echo -e "${GREEN}Creating required directories...${NC}"
mkdir -p data/faiss
mkdir -p logs
mkdir -p cache/dde
mkdir -p config
mkdir -p models

# Create virtual environment if requested
if [ "$CREATE_VENV" = true ]; then
  echo -e "${GREEN}Creating virtual environment...${NC}"
  if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping...${NC}"
  else
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
  fi

  # Activate virtual environment
  echo -e "${GREEN}Activating virtual environment...${NC}"
  source venv/bin/activate
fi

# Install dependencies if requested
if [ "$INSTALL_DEPS" = true ]; then
  echo -e "${GREEN}Installing dependencies...${NC}"
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install -r requirements-dev.txt
fi

# Set up Neo4j if requested
if [ "$SETUP_NEO4J" = true ]; then
  echo -e "${GREEN}Setting up Neo4j...${NC}"
  if command -v docker &> /dev/null; then
    if docker ps | grep -q subgraphrag_neo4j; then
      echo -e "${YELLOW}Neo4j container already running. Skipping...${NC}"
    else
      echo -e "${GREEN}Starting Neo4j container...${NC}"
      docker-compose up -d neo4j
      echo -e "${GREEN}Waiting for Neo4j to start...${NC}"
      sleep 10
    fi
  else
    echo -e "${YELLOW}Docker not found. Please install Docker or set up Neo4j manually.${NC}"
  fi
fi

# Create default configuration if it doesn't exist
echo -e "${GREEN}Checking for configuration file...${NC}"
if [ ! -f "config/config.json" ]; then
  echo -e "${YELLOW}Configuration file not found. Creating default...${NC}"
  echo '{
    "MODEL_BACKEND": "openai",
    "FAISS_INDEX_PATH": "data/faiss/index",
    "TOKEN_BUDGET": 8000,
    "MLP_MODEL_PATH": "models/mlp",
    "CACHE_DIR": "cache",
    "LOG_LEVEL": "INFO"
  }' > config/config.json
  echo -e "${GREEN}Created default configuration.${NC}"
fi

# Create schema file if it doesn't exist
if [ ! -f "config/config.schema.json" ]; then
  echo -e "${YELLOW}Schema file not found. Creating default...${NC}"
  echo '{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["MODEL_BACKEND", "FAISS_INDEX_PATH", "TOKEN_BUDGET"],
    "properties": {
      "MODEL_BACKEND": {
        "type": "string",
        "enum": ["openai", "hf", "mlx"],
        "description": "Backend for language model (OpenAI, HuggingFace, or MLX)"
      },
      "FAISS_INDEX_PATH": {
        "type": "string",
        "description": "Path to FAISS index file"
      },
      "TOKEN_BUDGET": {
        "type": "integer",
        "minimum": 100,
        "description": "Maximum number of tokens for LLM context window"
      },
      "MLP_MODEL_PATH": {
        "type": "string",
        "default": "models/mlp",
        "description": "Path to pre-trained MLP model for triple scoring"
      },
      "CACHE_DIR": {
        "type": "string",
        "default": "cache",
        "description": "Directory for caching embeddings, DDE features, etc."
      }
    }
  }' > config/config.schema.json
  echo -e "${GREEN}Created default schema.${NC}"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
  echo -e "${YELLOW}.env file not found. Creating default...${NC}"
  echo 'NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
API_KEY_SECRET=default_key_for_dev_only
# OPENAI_API_KEY=your_key_here' > .env
  echo -e "${GREEN}Created default .env file.${NC}"
  echo -e "${YELLOW}Remember to update the .env file with your own values.${NC}"
fi

# Download models if requested
if [ "$DOWNLOAD_MODELS" = true ]; then
  echo -e "${GREEN}Checking for MLP models...${NC}"
  if [ -d "models/mlp" ]; then
    echo -e "${YELLOW}MLP model already exists. Skipping...${NC}"
  else
    echo -e "${GREEN}Downloading MLP model...${NC}"
    python scripts/download_models.py
  fi
fi

# Initialize database if requested
if [ "$INITIALIZE_DB" = true ]; then
  echo -e "${GREEN}Initializing database...${NC}"
  
  # Check if Neo4j is available
  if command -v docker &> /dev/null && docker ps | grep -q subgraphrag_neo4j; then
    echo -e "${GREEN}Running Neo4j schema migration...${NC}"
    python scripts/migrate_schema.py
  else
    echo -e "${YELLOW}Neo4j not available. Skipping schema migration.${NC}"
  fi
  
  # Check if SQLite staging database exists
  if [ ! -f "data/staging.db" ]; then
    echo -e "${GREEN}Initializing SQLite staging database...${NC}"
    python -c "
import sqlite3, os
os.makedirs('data', exist_ok=True)
conn = sqlite3.connect('data/staging.db')
conn.execute('''CREATE TABLE IF NOT EXISTS staging_triples
             (head TEXT, relation TEXT, tail TEXT, head_name TEXT, relation_name TEXT, tail_name TEXT, 
              status TEXT DEFAULT 'pending', timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.execute('''CREATE TABLE IF NOT EXISTS error_log
             (triple_id INTEGER, error TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.execute('''CREATE TABLE IF NOT EXISTS auth_log
             (ip_hash TEXT, attempt_count INTEGER DEFAULT 1, last_attempt DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.execute('''CREATE INDEX IF NOT EXISTS idx_staging_status ON staging_triples(status)''')
conn.commit()
conn.close()
print('SQLite staging database initialized.')
"
  else
    echo -e "${YELLOW}SQLite staging database already exists. Skipping...${NC}"
  fi
fi

# Make shell scripts executable
echo -e "${GREEN}Making shell scripts executable...${NC}"
chmod +x *.sh

echo -e "${BLUE}"
echo "=================================================="
echo "           Setup completed successfully!          "
echo "=================================================="
echo -e "${NC}"
echo -e "${GREEN}You can now run the application with:${NC}"
echo -e "  ${YELLOW}./run.sh${NC}"
echo -e "${GREEN}For more options, see:${NC}"
echo -e "  ${YELLOW}./run.sh --help${NC}"
echo -e "  ${YELLOW}./run_tests.sh --help${NC}"
echo -e "  ${YELLOW}./run_benchmark.sh --help${NC}"
echo -e "  ${YELLOW}./backup.sh --help${NC}"