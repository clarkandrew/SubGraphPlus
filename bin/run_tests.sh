#!/bin/bash
# Script to run SubgraphRAG+ tests
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
echo "           SubgraphRAG+ Test Runner              "
echo "=================================================="
echo -e "${NC}"

# Define variables
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false
ADVERSARIAL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--type)
      TEST_TYPE="$2"
      shift 2
      ;;
    -c|--coverage)
      COVERAGE=true
      shift
      ;;
    -v|--verbose)
      VERBOSE=true
      shift
      ;;
    -a|--adversarial)
      ADVERSARIAL=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  -t, --type TYPE       Type of tests to run (all, unit, integration, smoke, adversarial)"
      echo "  -c, --coverage        Generate coverage report"
      echo "  -v, --verbose         Verbose output"
      echo "  -a, --adversarial     Run only adversarial tests"
      echo "  -h, --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

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

# Set environment variables if .env file exists
if [ -f ".env" ]; then
  echo -e "${GREEN}Loading environment variables...${NC}"
  set -a # automatically export all variables
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Create directories if they don't exist
mkdir -p logs
mkdir -p cache
mkdir -p data/faiss

# Check for Neo4j connectivity if we're running integration tests
if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "integration" ]]; then
  # Check if .env has USE_LOCAL_NEO4J flag
  USE_LOCAL_NEO4J=false
  if [ -f ".env" ] && grep -q "USE_LOCAL_NEO4J=true" .env; then
    USE_LOCAL_NEO4J=true
  fi
  
  if [ "$USE_LOCAL_NEO4J" = true ]; then
    echo -e "${YELLOW}Using local Neo4j installation for tests...${NC}"
    
    # Check if Neo4j is running locally (basic check)
    NEO4J_USER=${NEO4J_USER:-"neo4j"}
    NEO4J_PASSWORD=${NEO4J_PASSWORD:-"password"}
    NEO4J_URI=${NEO4J_URI:-"neo4j://localhost:7687"}
    
    # Python-based check since it will be used in tests anyway
    if python -c "
from neo4j import GraphDatabase
import sys
try:
    with GraphDatabase.driver('$NEO4J_URI', auth=('$NEO4J_USER', '$NEO4J_PASSWORD')) as driver:
        result = driver.session().run('RETURN 1 as n')
        list(result)
        print('Neo4j connection successful')
        sys.exit(0)
except Exception as e:
    print(f'Neo4j connection error: {str(e)}')
    sys.exit(1)
" 2>/dev/null; then
      echo -e "${GREEN}Successfully connected to local Neo4j.${NC}"
    else
      echo -e "${YELLOW}Warning: Could not connect to local Neo4j. Some tests may fail.${NC}"
      echo -e "${YELLOW}Tips for starting Neo4j:${NC}"
      echo -e "  - Neo4j Desktop: Start through the application interface"
      echo -e "  - macOS Homebrew: brew services start neo4j"
      echo -e "  - Linux systemd: sudo systemctl start neo4j"
    fi
  else
    # Check for Docker-based Neo4j
    if command -v docker &> /dev/null; then
      if ! docker ps | grep -q subgraphrag_neo4j; then
        echo -e "${YELLOW}Neo4j container not running. Some tests may fail.${NC}"
      else
        echo -e "${GREEN}Neo4j container is running.${NC}"
      fi
    else
      echo -e "${YELLOW}Docker not found. If you're using a local Neo4j installation, add USE_LOCAL_NEO4J=true to your .env file.${NC}"
    fi
  fi
fi

# Run the selected tests
if [ "$COVERAGE" = true ]; then
  COVERAGE_CMD="coverage run -m"
else
  COVERAGE_CMD=""
fi

if [ "$VERBOSE" = true ]; then
  PYTEST_OPTS="-v"
else
  PYTEST_OPTS=""
fi

echo -e "${GREEN}Running $TEST_TYPE tests...${NC}"

case $TEST_TYPE in
  "unit")
    $COVERAGE_CMD pytest $PYTEST_OPTS tests/test_utils.py tests/test_retriever.py
    ;;
  "integration")
    echo -e "${YELLOW}Running integration tests. Ensure Neo4j is running.${NC}"
    $COVERAGE_CMD pytest $PYTEST_OPTS tests/test_api.py
    ;;
  "smoke")
    $COVERAGE_CMD pytest $PYTEST_OPTS tests/test_smoke.py
    ;;
  "adversarial"|"adv")
    $COVERAGE_CMD pytest $PYTEST_OPTS tests/test_adversarial.py
    ;;
  "all")
    if [ "$ADVERSARIAL" = true ]; then
      $COVERAGE_CMD pytest $PYTEST_OPTS tests/test_adversarial.py
    else
      $COVERAGE_CMD pytest $PYTEST_OPTS
    fi
    ;;
  *)
    echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
    exit 1
    ;;
esac

# Generate coverage report if requested
if [ "$COVERAGE" = true ]; then
  echo -e "${GREEN}Generating coverage report...${NC}"
  coverage report
  coverage html
  echo -e "${GREEN}HTML coverage report generated in htmlcov/index.html${NC}"
fi

echo -e "${BLUE}"
echo "=================================================="
echo "           Tests completed                        "
echo "=================================================="
echo -e "${NC}"