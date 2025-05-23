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

# Check if .env has USE_LOCAL_NEO4J flag
USE_LOCAL_NEO4J=false
if [ -f ".env" ] && grep -q "USE_LOCAL_NEO4J=true" .env; then
  USE_LOCAL_NEO4J=true
fi

# Handle Neo4j - either Docker or local installation
if [ "$USE_LOCAL_NEO4J" = true ]; then
  echo -e "${YELLOW}Using local Neo4j installation...${NC}"
  
  # Check if Neo4j is running locally (basic check)
  if command -v cypher-shell &> /dev/null; then
    echo -e "${GREEN}Neo4j client tools found. Checking connection...${NC}"
    # Try to connect (will fail silently if not running)
    NEO4J_USER=${NEO4J_USER:-"neo4j"}
    NEO4J_PASSWORD=${NEO4J_PASSWORD:-"password"}
    if echo "RETURN 1;" | cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" &> /dev/null; then
      echo -e "${GREEN}Successfully connected to local Neo4j.${NC}"
    else
      echo -e "${YELLOW}Could not connect to local Neo4j. Please ensure it's running.${NC}"
      echo -e "${YELLOW}Tips for starting Neo4j:${NC}"
      echo -e "  - Neo4j Desktop: Start through the application interface"
      echo -e "  - macOS Homebrew: brew services start neo4j"
      echo -e "  - Linux systemd: sudo systemctl start neo4j"
      echo -e "  - Windows: Start through Neo4j Desktop or services"
    fi
  else
    echo -e "${YELLOW}Neo4j client tools not found. Assuming Neo4j is running.${NC}"
    echo -e "${YELLOW}If you encounter connection errors, please ensure Neo4j is running.${NC}"
  fi
else
  # Check if Neo4j is running in Docker
  if command -v docker &> /dev/null; then
    if ! docker ps | grep -q subgraphrag_neo4j; then
      echo -e "${YELLOW}Neo4j container not running. Starting...${NC}"
      if command -v docker-compose &> /dev/null; then
        docker-compose up -d neo4j
      else
        # Try docker compose (v2)
        docker compose up -d neo4j
      fi
      echo -e "${GREEN}Waiting for Neo4j to start...${NC}"
      sleep 10
    else
      echo -e "${GREEN}Neo4j container is running.${NC}"
    fi
  else
    echo -e "${YELLOW}Docker not found. Make sure Neo4j is running manually.${NC}"
    echo -e "${YELLOW}If you're using a local Neo4j installation, add USE_LOCAL_NEO4J=true to your .env file.${NC}"
  fi
fi

# Set environment variables if .env file exists
if [ -f ".env" ]; then
  echo -e "${GREEN}Loading environment variables...${NC}"
  set -a # automatically export all variables
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Check if models are available
if [ ! -d "models/mlp" ]; then
  echo -e "${YELLOW}MLP model not found. Downloading...${NC}"
  python scripts/download_models.py
fi

# Check for configuration file
if [ ! -f "config/config.json" ]; then
  echo -e "${YELLOW}Configuration file not found. Creating default...${NC}"
  mkdir -p config
  echo '{
    "MODEL_BACKEND": "openai",
    "FAISS_INDEX_PATH": "data/faiss/index",
    "TOKEN_BUDGET": 8000,
    "MLP_MODEL_PATH": "models/mlp",
    "CACHE_DIR": "cache",
    "LOG_LEVEL": "INFO"
  }' > config/config.json
  echo -e "${GREEN}Created default configuration in config/config.json${NC}"
  echo -e "${YELLOW}You may need to update the configuration for your environment.${NC}"
fi

# Create directories if they don't exist
mkdir -p data/faiss
mkdir -p logs
mkdir -p cache

# Check if database is initialized
if [ ! -f "data/staging.db" ]; then
  echo -e "${YELLOW}No staging database found. Initializing...${NC}"
  python scripts/migrate_schema.py
fi

# Parse command line arguments
PORT=8000
DEBUG=true
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --no-debug)
      DEBUG=false
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --port PORT      Port to run the server on (default: 8000)"
      echo "  --no-debug       Run without debug mode (no auto-reload)"
      echo "  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run '$0 --help' for usage information."
      exit 1
      ;;
  esac
done

# Run application
echo -e "${GREEN}Starting SubgraphRAG+ application...${NC}"
if [ "$DEBUG" = true ]; then
  echo -e "${GREEN}Running in debug mode with auto-reload enabled${NC}"
  uvicorn main:app --reload --host 0.0.0.0 --port "$PORT"
else
  echo -e "${GREEN}Running in production mode${NC}"
  uvicorn main:app --host 0.0.0.0 --port "$PORT"
fi