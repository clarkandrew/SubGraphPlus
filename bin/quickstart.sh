#!/bin/bash
# SubgraphRAG+ Quickstart Script
# This script provides a one-stop solution for setting up the SubgraphRAG+ environment

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
echo "           SubgraphRAG+ Quickstart                "
echo "=================================================="
echo -e "${NC}"

# Define variables
VENV_DIR="venv"
PYTHON_VERSION="python3"
ROOT_DIR=$(pwd)
MODE="dev"
SKIP_TESTS=false
SKIP_DOCKER=false
SKIP_SAMPLE_DATA=false
PYTHON_PREFERRED_VERSION=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prod)
      MODE="prod"
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=true
      shift
      ;;
    --skip-docker)
      SKIP_DOCKER=true
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
    -h|--help)
      echo "Usage: ./bin/quickstart.sh [options]"
      echo ""
      echo "Options:"
      echo "  --prod               Setup for production environment (default: development)"
      echo "  --skip-tests         Skip running tests"
      echo "  --skip-docker        Skip Docker setup (local environment only)"
      echo "  --skip-sample-data   Skip loading sample data"
      echo "  --python VERSION     Use specific Python version (e.g., python3.11)"
      echo "  -h, --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './bin/quickstart.sh --help' for usage information."
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
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  
  # Upgrade pip, setuptools, and wheel
  echo -e "${GREEN}Upgrading pip, setuptools, and wheel...${NC}"
  pip install --upgrade pip setuptools wheel
  
  # Install dependencies based on mode
  if [ "$MODE" = "dev" ]; then
    echo -e "${GREEN}Installing development dependencies...${NC}"
    pip install -r requirements-dev.txt
  else
    echo -e "${GREEN}Installing production dependencies...${NC}"
    pip install -r requirements.txt
  fi
  
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
  if [ "$SKIP_DOCKER" = true ]; then
    echo -e "${YELLOW}Skipping Neo4j Docker setup as requested.${NC}"
    echo "Please ensure your Neo4j database is configured properly."
    return 0
  fi
  
  echo -e "${BLUE}Setting up Neo4j database...${NC}"
  
  # Check if Docker is installed
  if ! command -v docker >/dev/null 2>&1; then
    echo -e "${RED}Docker is not installed. Please install Docker and try again.${NC}"
    echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
    echo -e "${YELLOW}You can run this script with --skip-docker flag to skip this step if you have Neo4j installed separately.${NC}"
    exit 1
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
    if docker compose version >/dev/null 2>&1; then
      docker compose up -d neo4j
    else
      docker-compose up -d neo4j
    fi
    
    # Wait for Neo4j to start
    echo -e "${YELLOW}Waiting for Neo4j to start (this may take a minute)...${NC}"
    sleep 10
  fi
  
  echo -e "${GREEN}Neo4j setup complete.${NC}"
}

# Function to download pre-trained models
download_models() {
  echo -e "${BLUE}Downloading pre-trained models...${NC}"
  
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  
  # Download models
  python scripts/download_models.py
  
  echo -e "${GREEN}Model download complete.${NC}"
}

# Function to initialize database schema
init_schema() {
  echo -e "${BLUE}Initializing database schema...${NC}"
  
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  
  # Apply migrations
  python scripts/migrate_schema.py
  
  echo -e "${GREEN}Schema initialization complete.${NC}"
}

# Function to load sample data
load_sample_data() {
  if [ "$SKIP_SAMPLE_DATA" = true ]; then
    echo -e "${YELLOW}Skipping sample data loading as requested.${NC}"
    return 0
  fi
  
  echo -e "${BLUE}Loading sample data...${NC}"
  
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  
  # Stage sample data
  python scripts/stage_ingest.py --sample
  
  # Process ingestion queue
  python scripts/ingest_worker.py --process-all
  
  # Merge FAISS index
  python scripts/merge_faiss.py
  
  echo -e "${GREEN}Sample data loaded.${NC}"
}

# Function to run tests
run_tests() {
  if [ "$SKIP_TESTS" = true ]; then
    echo -e "${YELLOW}Skipping tests as requested.${NC}"
    return 0
  fi
  
  echo -e "${BLUE}Running tests...${NC}"
  
  # shellcheck disable=SC1091
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
  echo "          SubgraphRAG+ Setup Complete!            "
  echo "=================================================="
  echo -e "${NC}"
  echo -e "${GREEN}You can now:${NC}"
  echo -e "1. Start the server: ${YELLOW}source venv/bin/activate && python main.py${NC}"
  echo -e "   OR use the run script: ${YELLOW}./bin/run.sh${NC}"
  echo -e ""
  echo -e "2. Access the API at: ${YELLOW}http://localhost:8000${NC}"
  echo -e "3. View API documentation at: ${YELLOW}http://localhost:8000/docs${NC}"
  echo -e "4. Access Neo4j browser at: ${YELLOW}http://localhost:7474${NC}"
  echo -e "   (username: neo4j, password: password)"
  echo -e ""
  echo -e "For more information, see the documentation in the ${YELLOW}docs/${NC} directory."
}

# Main setup process
main() {
  cd "$ROOT_DIR" || exit 1
  
  select_python_version
  setup_venv
  create_directories
  setup_neo4j
  download_models
  init_schema
  load_sample_data
  run_tests
  show_next_steps
}

# Run the main setup process
main