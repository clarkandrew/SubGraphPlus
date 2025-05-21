#!/bin/bash
# Script to run SubgraphRAG+ application
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
echo "           SubgraphRAG+ Application Runner        "
echo "=================================================="
echo -e "${NC}"

# Check for virtualenv
if [ -d "venv" ]; then
  echo -e "${GREEN}Activating virtual environment...${NC}"
  source venv/bin/activate
else
  echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
  python -m venv venv
  source venv/bin/activate
  echo -e "${GREEN}Installing requirements...${NC}"
  pip install -r requirements.txt
  pip install -r requirements-dev.txt
fi

# Check if Neo4j is running in Docker
if command -v docker &> /dev/null; then
  if ! docker ps | grep -q subgraphrag_neo4j; then
    echo -e "${YELLOW}Neo4j container not running. Starting...${NC}"
    docker-compose up -d neo4j
    echo -e "${GREEN}Waiting for Neo4j to start...${NC}"
    sleep 10
  else
    echo -e "${GREEN}Neo4j container is running.${NC}"
  fi
else
  echo -e "${YELLOW}Docker not found. Make sure Neo4j is running manually.${NC}"
fi

# Set environment variables if .env file exists
if [ -f ".env" ]; then
  echo -e "${GREEN}Loading environment variables...${NC}"
  export $(grep -v '^#' .env | xargs)
fi

# Check if models are available
if [ ! -d "models/mlp" ]; then
  echo -e "${YELLOW}MLP model not found. Downloading...${NC}"
  python scripts/download_models.py
fi

# Check for configuration file
if [ ! -f "config/app_config.json" ]; then
  echo -e "${YELLOW}Configuration file not found. Creating default...${NC}"
  mkdir -p config
  echo '{
    "MODEL_BACKEND": "openai",
    "FAISS_INDEX_PATH": "data/faiss/index",
    "TOKEN_BUDGET": 8000,
    "MLP_MODEL_PATH": "models/mlp",
    "CACHE_DIR": "cache"
  }' > config/app_config.json
fi

# Create directories if they don't exist
mkdir -p data/faiss
mkdir -p logs
mkdir -p cache

# Run application
echo -e "${GREEN}Starting SubgraphRAG+ application...${NC}"
uvicorn main:app --reload --host 0.0.0.0 --port 8000