#!/bin/bash
# SubgraphRAG+ Development Environment Setup Script
# This script sets up a development environment using virtualenv

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
echo "    SubgraphRAG+ Development Environment Setup    "
echo "=================================================="
echo -e "${NC}"

# Define variables
VENV_DIR="venv"
PYTHON_VERSION="python3.9"
ROOT_DIR=$(pwd)
SKIP_TESTS=true
SKIP_NEO4J=false
SKIP_SAMPLE_DATA=false
PYTHON_PREFERRED_VERSION=""
USE_LOCAL_NEO4J=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-tests)
      SKIP_TESTS=false
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=true
      shift
      ;;
    --skip-neo4j)
      SKIP_NEO4J=true
      shift
      ;;
    --skip-sample-data)
      SKIP_SAMPLE_DATA=true
      shift
      ;;
    --python)
      PYTHON_PREFERRED_VERSION="$2"
      shift 2
      ;;
    --use-local-neo4j)
      USE_LOCAL_NEO4J=true
      shift
      ;;
    -h|--help)
      echo "Usage: ./bin/setup_dev.sh [options]"
      echo ""
      echo "Options:"
      echo "  --run-tests          Run tests (tests are skipped by default)"
      echo "  --skip-tests         Skip running tests (default behavior)"
      echo "  --skip-neo4j         Skip Neo4j setup (if you have it running separately)"
      echo "  --skip-sample-data   Skip loading sample data"
      echo "  --python VERSION     Use specific Python version (e.g., python3.11)"
      echo "  --use-local-neo4j    Use locally installed Neo4j instead of Docker"
      echo "                       (Requires Neo4j 4.4+ with APOC plugin installed)"
      echo "  -h, --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './bin/setup_dev.sh --help' for usage information."
      exit 1
      ;;
  esac
done

# Function to check and select Python version
select_python_version() {
  if [ -n "$PYTHON_PREFERRED_VERSION" ]; then
    if command -v "$PYTHON_PREFERRED_VERSION" >/dev/null 2>&1; then
      PYTHON_VERSION="$PYTHON_PREFERRED_VERSION"
      echo -e "${GREEN}Using specified Python version: $PYTHON_VERSION${NC}"
    else
      echo -e "${YELLOW}Specified Python version $PYTHON_PREFERRED_VERSION not found, using fallback${NC}"
    fi
  fi

  # Try to find Python 3.9+ if no preferred version is specified or it wasn't found
  if [ "$PYTHON_VERSION" = "python3" ]; then
    for ver in python3.11 python3.10 python3.9; do
      if command -v "$ver" >/dev/null 2>&1; then
        PYTHON_VERSION="$ver"
        echo -e "${GREEN}Selected Python version: $PYTHON_VERSION${NC}"
        break
      fi
    done
  fi

  # Check Python version meets minimum requirements
  PYTHON_VERSION_CHECK=$($PYTHON_VERSION -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  if [ "$(echo "$PYTHON_VERSION_CHECK < 3.9" | bc)" -eq 1 ]; then
    echo -e "${RED}Error: Python 3.9 or higher is required. Found version $PYTHON_VERSION_CHECK${NC}"
    echo "Please install a compatible Python version and try again."
    exit 1
  fi
}

# Function to set up virtual environment
setup_venv() {
  echo -e "${BLUE}Setting up Python virtual environment...${NC}"

  # Check if venv exists and create if it doesn't
  if [ ! -d "$VENV_DIR" ]; then
    echo -e "${GREEN}Creating new virtual environment...${NC}"
    $PYTHON_VERSION -m venv "$VENV_DIR"
  else
    echo -e "${YELLOW}Virtual environment already exists. Reusing it.${NC}"
  fi

  # Activate virtual environment
  source "$VENV_DIR/bin/activate"

  # Upgrade pip, setuptools, and wheel
  echo -e "${GREEN}Upgrading pip, setuptools, and wheel...${NC}"
  pip install -q --upgrade pip setuptools wheel

  # Install development dependencies
  echo -e "${GREEN}Installing development dependencies...${NC}"
  pip install -q -r requirements-dev.txt

  # Install the package in development mode
  echo -e "${GREEN}Installing SubGraphRAG+ in development mode...${NC}"
  pip install -e .

  echo -e "${GREEN}Virtual environment setup complete.${NC}"
}

# Function to create necessary directories
create_directories() {
  echo -e "${BLUE}Creating necessary directories...${NC}"

  # Create application directories
  mkdir -p data/faiss data/sample_data
  mkdir -p cache logs models
  mkdir -p config

  echo -e "${GREEN}Directories created.${NC}"
}

# Function to set up Neo4j database
setup_neo4j() {
  if [ "$SKIP_NEO4J" = true ]; then
    echo -e "${YELLOW}Skipping Neo4j setup as requested.${NC}"
    echo "Please ensure your Neo4j database is configured properly."
    return 0
  fi

  echo -e "${BLUE}Setting up Neo4j database...${NC}"

  if [ "$USE_LOCAL_NEO4J" = true ]; then
    echo -e "${BLUE}Using locally installed Neo4j...${NC}"
    
    # Check if Neo4j is installed locally
    if command -v neo4j >/dev/null 2>&1 || command -v cypher-shell >/dev/null 2>&1 || [ -d "/Applications/Neo4j Desktop.app" ]; then
      echo -e "${GREEN}Neo4j appears to be installed.${NC}"
      echo -e "${YELLOW}Please ensure your Neo4j instance is running and accessible at neo4j://localhost:7687${NC}"
      echo -e "${YELLOW}with username 'neo4j' and password 'password' (or update your .env file).${NC}"
      
      # Give user time to start Neo4j if needed
      echo -e "${YELLOW}If Neo4j is not running, please start it now...${NC}"
      sleep 5
    else
      echo -e "${YELLOW}Neo4j does not appear to be installed.${NC}"
      echo -e "${YELLOW}Please install Neo4j using one of these methods:${NC}"
      echo -e "1. Download Neo4j Desktop: https://neo4j.com/download/"
      echo -e "2. Use Homebrew: brew install neo4j"
      echo -e "   Then start with: brew services start neo4j"
      echo -e "   Set password with: cypher-shell -u neo4j -p neo4j"
      echo -e "${YELLOW}Press any key to continue once Neo4j is installed and running...${NC}"
      read -n 1 -s
    fi
  else
    # Check if Docker is installed
    if ! command -v docker >/dev/null 2>&1; then
      echo -e "${RED}Docker is not installed. Please install Docker and try again.${NC}"
      echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
      echo -e "${YELLOW}You can run this script with --skip-neo4j flag to skip this step if you have Neo4j installed separately.${NC}"
      echo -e "${YELLOW}Or use --use-local-neo4j if you have Neo4j installed locally.${NC}"
      exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
      echo -e "${RED}Docker daemon is not running.${NC}"
      echo -e "${YELLOW}Please start Docker and try again. You can:${NC}"
      echo -e "1. Start Docker Desktop application"
      echo -e "2. Or run: ${BLUE}sudo systemctl start docker${NC} (Linux)"
      echo -e "3. Or run: ${BLUE}open -a Docker${NC} (macOS)"
      echo -e ""
      echo -e "${YELLOW}Alternative options:${NC}"
      echo -e "• Use --skip-neo4j flag to skip Neo4j setup"
      echo -e "• Use --use-local-neo4j if you have Neo4j installed locally"
      echo -e ""
      echo -e "${YELLOW}Would you like to continue without Docker? [y/N]${NC}"
      read -r response
      if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Continuing without Docker. Please ensure Neo4j is available separately.${NC}"
        SKIP_NEO4J=true
        return 0
      else
        exit 1
      fi
    fi

    # Check if docker-compose is installed
    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
      echo -e "${RED}Docker Compose is not installed. Please install Docker Compose and try again.${NC}"
      echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
      exit 1
    fi

    # Check if Neo4j container is already running
    if docker ps | grep -q "subgraphrag_neo4j"; then
      echo -e "${YELLOW}Neo4j container is already running.${NC}"
    else
      echo -e "${GREEN}Starting Neo4j container...${NC}"
      
      # Try to start using docker compose (newer format first)
      if docker compose version >/dev/null 2>&1; then
        if ! docker compose up -d neo4j 2>/dev/null; then
          echo -e "${RED}Failed to start Neo4j with 'docker compose'.${NC}"
          echo -e "${YELLOW}Trying with 'docker-compose'...${NC}"
          docker-compose up -d neo4j
        fi
      else
        docker-compose up -d neo4j
      fi

      # Wait for Neo4j to start
      echo -e "${YELLOW}Waiting for Neo4j to start (this may take a minute)...${NC}"
      sleep 15
      
      # Check if container started successfully
      if ! docker ps | grep -q "subgraphrag_neo4j"; then
        echo -e "${RED}Failed to start Neo4j container. Please check Docker logs:${NC}"
        echo -e "${BLUE}docker logs subgraphrag_neo4j${NC}"
        echo -e "${YELLOW}You can continue with --skip-neo4j flag or --use-local-neo4j if you have Neo4j installed locally.${NC}"
        exit 1
      else
        echo -e "${GREEN}Neo4j container started successfully.${NC}"
      fi
    fi
  fi

  echo -e "${GREEN}Neo4j setup complete.${NC}"
}

# Function to create configuration files
create_config_files() {
  echo -e "${GREEN}Checking for configuration files...${NC}"

  # Create default configuration if it doesn't exist
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
}

# Function to download pre-trained models
download_models() {
  echo -e "${BLUE}Downloading pre-trained models...${NC}"

  # Check if models directory exists
  if [ -d "models/mlp" ]; then
    echo -e "${YELLOW}MLP model already exists. Skipping download...${NC}"
  else
    echo -e "${GREEN}Downloading MLP model...${NC}"
    source "$VENV_DIR/bin/activate"
    python scripts/download_models.py
  fi

  echo -e "${GREEN}Model download complete.${NC}"
}

# Function to initialize database schema
init_schema() {
  echo -e "${BLUE}Initializing database schema...${NC}"

  source "$VENV_DIR/bin/activate"

  # Check if Neo4j is available
  if [ "$SKIP_NEO4J" = false ]; then
    # Simple Neo4j connectivity test using Python
    NEO4J_TEST=$(python -c "
import os
from dotenv import load_dotenv
load_dotenv()
try:
    from neo4j import GraphDatabase
    uri = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'password')
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        result = driver.session().run('RETURN 1 as n')
        print('connected')
except Exception as e:
    print(f'error: {str(e)}')
" 2>/dev/null)
    
    if [[ $NEO4J_TEST == *"connected"* ]]; then
      echo -e "${GREEN}Neo4j connection successful.${NC}"
      echo -e "${GREEN}Running Neo4j schema migration...${NC}"
      python scripts/migrate_schema.py
    else
      echo -e "${YELLOW}Neo4j connection failed: $NEO4J_TEST${NC}"
      echo -e "${YELLOW}Skipping schema migration. Make sure Neo4j is running and credentials are correct in .env file.${NC}"
    fi
  else
    echo -e "${YELLOW}Neo4j setup skipped. Skipping schema migration.${NC}"
  fi

  # Initialize SQLite staging database
  echo -e "${GREEN}Initializing SQLite staging database...${NC}"
  if [ ! -f "data/staging.db" ]; then
    mkdir -p data
    echo -e "${GREEN}Creating SQLite staging database...${NC}"
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

  echo -e "${GREEN}Schema initialization complete.${NC}"
}

# Function to load sample data
load_sample_data() {
  if [ "$SKIP_SAMPLE_DATA" = true ]; then
    echo -e "${YELLOW}Skipping sample data loading as requested.${NC}"
    return 0
  fi

  # Check Neo4j connectivity first
  NEO4J_TEST=$(python -c "
import os
from dotenv import load_dotenv
load_dotenv()
try:
    from neo4j import GraphDatabase
    uri = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'password')
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        result = driver.session().run('RETURN 1 as n')
        print('connected')
except Exception as e:
    print(f'error: {str(e)}')
" 2>/dev/null)

  if [[ $NEO4J_TEST != *"connected"* ]]; then
    echo -e "${YELLOW}Neo4j connection failed. Cannot load sample data without Neo4j.${NC}"
    echo -e "${YELLOW}Make sure Neo4j is running and credentials are correct in .env file.${NC}"
    return 1
  fi

  echo -e "${BLUE}Loading sample data...${NC}"

  source "$VENV_DIR/bin/activate"

  # Create directories if they don't exist
  mkdir -p data/faiss data/sample_data

  # Stage sample data
  echo -e "${GREEN}Staging sample data...${NC}"
  python scripts/stage_ingest.py --sample

  # Process ingestion queue with timeout
  echo -e "${GREEN}Processing ingestion queue (this may take a few minutes on first run due to model download)...${NC}"
  
  # Add timeout to prevent hanging - 10 minutes should be enough for model download and processing
  timeout 600 python scripts/ingest_worker.py --process-all
  INGEST_EXIT_CODE=$?
  
  if [ $INGEST_EXIT_CODE -eq 124 ]; then
    echo -e "${RED}Ingest worker timed out after 10 minutes.${NC}"
    echo -e "${YELLOW}This might be due to slow model download. You can run the ingest manually later with:${NC}"
    echo -e "${YELLOW}  source venv/bin/activate && python scripts/ingest_worker.py --process-all${NC}"
    echo -e "${YELLOW}Continuing with setup...${NC}"
  elif [ $INGEST_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}Ingest worker failed with exit code $INGEST_EXIT_CODE.${NC}"
    echo -e "${YELLOW}You can run the ingest manually later with:${NC}"
    echo -e "${YELLOW}  source venv/bin/activate && python scripts/ingest_worker.py --process-all${NC}"
    echo -e "${YELLOW}Continuing with setup...${NC}"
  else
    echo -e "${GREEN}Ingest worker completed successfully.${NC}"
  fi

  # Merge FAISS index
  echo -e "${GREEN}Merging FAISS index...${NC}"
  python scripts/merge_faiss.py

  echo -e "${GREEN}Sample data loaded.${NC}"
}

# Function to run tests
run_tests() {
  if [ "$SKIP_TESTS" = true ]; then
    echo -e "${YELLOW}Skipping tests (default behavior).${NC}"
    echo -e "${YELLOW}Use --run-tests flag to enable test execution.${NC}"
    return 0
  fi

  echo -e "${BLUE}Running tests...${NC}"

  source "$VENV_DIR/bin/activate"

  # Run tests
  if [ -f "bin/run_tests.sh" ]; then
    ./bin/run_tests.sh
  else
    echo -e "${YELLOW}Test script not found, running pytest directly...${NC}"
    pytest
  fi

  echo -e "${GREEN}Tests completed.${NC}"
}

# Function to show next steps
show_next_steps() {
  echo -e "${BLUE}"
  echo "=================================================="
  echo "    SubgraphRAG+ Development Setup Complete!      "
  echo "=================================================="
  echo -e "${NC}"
  echo -e "${GREEN}You can now:${NC}"
  echo -e "1. Start the development server: ${YELLOW}source venv/bin/activate && python src/main.py --reload${NC}"
  echo -e "   OR use the run script: ${YELLOW}./bin/run.sh${NC}"
  echo -e ""
  echo -e "2. Access the API at: ${YELLOW}http://localhost:8000${NC}"
  echo -e "3. View API documentation at: ${YELLOW}http://localhost:8000/docs${NC}"
  echo -e "4. Access Neo4j browser at: ${YELLOW}http://localhost:7474${NC}"
  if [ "$USE_LOCAL_NEO4J" = true ]; then
    echo -e "   (Make sure you're using the credentials from your .env file)"
  else
    echo -e "   (username: neo4j, password: password)"
  fi
  echo -e ""
  echo -e "5. Run tests: ${YELLOW}./bin/run_tests.sh${NC}"
  echo -e "6. Run benchmarks: ${YELLOW}./bin/run_benchmark.sh${NC}"
  echo -e ""
  echo -e "${YELLOW}Note:${NC} If you used local Neo4j installation (--use-local-neo4j), ensure it's running"
  echo -e "before starting the development server."
  echo -e ""
  echo -e "For more information, see the documentation in the ${YELLOW}docs/${NC} directory."
}

# Make scripts executable
make_scripts_executable() {
  echo -e "${GREEN}Making scripts executable...${NC}"
  chmod +x bin/*.sh
  echo -e "${GREEN}Done.${NC}"
}

# Main function
main() {
  cd "$ROOT_DIR" || exit 1

  echo -e "${GREEN}Starting development environment setup...${NC}"

  select_python_version
  create_directories
  setup_venv
  create_config_files
  setup_neo4j
  download_models
  init_schema
  load_sample_data
  run_tests
  make_scripts_executable
  show_next_steps
}

# Run the main function
main
