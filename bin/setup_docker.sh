#!/bin/bash
# SubgraphRAG+ Docker Environment Setup Script
# This script sets up a Docker-based environment for SubgraphRAG+

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
echo "      SubgraphRAG+ Docker Environment Setup       "
echo "=================================================="
echo -e "${NC}"

# Define variables
SKIP_SAMPLE_DATA=false
REBUILD=false
FORCE_PULL=false
DETACHED=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-sample-data)
      SKIP_SAMPLE_DATA=false
      shift
      ;;
    --skip-sample-data)
      SKIP_SAMPLE_DATA=true
      shift
      ;;
    --rebuild)
      REBUILD=true
      shift
      ;;
    --pull)
      FORCE_PULL=true
      shift
      ;;
    --foreground)
      DETACHED=false
      shift
      ;;
    -h|--help)
      echo "Usage: ./bin/setup_docker.sh [options]"
      echo ""
      echo "Options:"
      echo "  --with-sample-data  Load sample data after setup (default)"
      echo "  --skip-sample-data  Skip loading sample data"
      echo "  --rebuild           Force rebuild of Docker images"
      echo "  --pull              Force pull latest base images"
      echo "  --foreground        Run in foreground (not detached)"
      echo "  -h, --help          Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './bin/setup_docker.sh --help' for usage information."
      exit 1
      ;;
  esac
done

# Function to check if Docker and Docker Compose are installed
check_docker() {
  echo -e "${BLUE}Checking Docker and Docker Compose installation...${NC}"
  
  if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker and try again.${NC}"
    echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
  fi
  
  if ! docker info &> /dev/null; then
    echo -e "${RED}Docker daemon is not running or you don't have permission to use Docker.${NC}"
    echo "Please start Docker and ensure you have the necessary permissions."
    exit 1
  fi
  
  # Check for docker-compose (either standalone or as docker compose plugin)
  if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose and try again.${NC}"
    echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
  fi
  
  echo -e "${GREEN}Docker and Docker Compose are properly installed.${NC}"
}

# Function to check for required files
check_required_files() {
  echo -e "${BLUE}Checking for required files...${NC}"
  
  if [ ! -f "docker-compose.yml" ] && [ ! -f "compose.yaml" ]; then
    echo -e "${RED}Docker Compose file not found. Please make sure you're in the project root directory.${NC}"
    exit 1
  fi
  
  if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}Dockerfile not found. Please make sure you're in the project root directory.${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}All required files found.${NC}"
}

# Function to create required directories
create_directories() {
  echo -e "${BLUE}Creating required directories...${NC}"
  
  mkdir -p data/faiss
  mkdir -p logs
  mkdir -p cache
  mkdir -p models
  mkdir -p config
  mkdir -p backups
  
  echo -e "${GREEN}Directories created.${NC}"
}

# Function to create env file if it doesn't exist
create_env_file() {
  echo -e "${BLUE}Checking for .env file...${NC}"
  
  if [ ! -f ".env" ]; then
    echo -e "${YELLOW}.env file not found. Creating default...${NC}"
    echo 'NEO4J_URI=neo4j://subgraphrag_neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
API_KEY_SECRET=default_key_for_dev_only
# OPENAI_API_KEY=your_key_here' > .env
    echo -e "${GREEN}Created default .env file.${NC}"
    echo -e "${YELLOW}Remember to update the .env file with your own values.${NC}"
  else
    echo -e "${GREEN}.env file exists, skipping creation.${NC}"
  fi
}

# Function to pull or build Docker images
build_images() {
  echo -e "${BLUE}Preparing Docker images...${NC}"
  
  if [ "$FORCE_PULL" = true ]; then
    echo -e "${YELLOW}Pulling latest base images...${NC}"
    if docker compose version &> /dev/null; then
      docker compose pull
    else
      docker-compose pull
    fi
  fi
  
  if [ "$REBUILD" = true ]; then
    echo -e "${YELLOW}Rebuilding Docker images...${NC}"
    if docker compose version &> /dev/null; then
      docker compose build --no-cache
    else
      docker-compose build --no-cache
    fi
  else
    echo -e "${YELLOW}Building Docker images if needed...${NC}"
    if docker compose version &> /dev/null; then
      docker compose build
    else
      docker-compose build
    fi
  fi
  
  echo -e "${GREEN}Docker images ready.${NC}"
}

# Function to start Docker services
start_services() {
  echo -e "${BLUE}Starting Docker services...${NC}"
  
  if [ "$DETACHED" = true ]; then
    if docker compose version &> /dev/null; then
      docker compose up -d
    else
      docker-compose up -d
    fi
    
    echo -e "${GREEN}Services started in detached mode.${NC}"
  else
    if docker compose version &> /dev/null; then
      docker compose up
    else
      docker-compose up
    fi
  fi
  
  # Only continue for detached mode since foreground mode blocks
  if [ "$DETACHED" = true ]; then
    echo -e "${YELLOW}Waiting for services to initialize...${NC}"
    sleep 10
    
    # Check if services are running
    if docker ps | grep -q "subgraphrag_neo4j"; then
      echo -e "${GREEN}Neo4j is running.${NC}"
    else
      echo -e "${RED}Neo4j failed to start. Check logs with 'docker logs subgraphrag_neo4j'${NC}"
    fi
    
    if docker ps | grep -q "subgraphrag_api"; then
      echo -e "${GREEN}SubgraphRAG+ API is running.${NC}"
    else
      echo -e "${RED}SubgraphRAG+ API failed to start. Check logs with 'docker logs subgraphrag_api'${NC}"
    fi
  fi
}

# Function to load sample data
load_sample_data() {
  if [ "$SKIP_SAMPLE_DATA" = true ]; then
    echo -e "${YELLOW}Skipping sample data loading as requested.${NC}"
    return 0
  fi
  
  if [ "$DETACHED" = false ]; then
    echo -e "${YELLOW}Cannot load sample data in foreground mode. Please run manually after startup.${NC}"
    return 0
  fi
  
  echo -e "${BLUE}Loading sample data...${NC}"
  
  echo -e "${GREEN}Adding sample triples to the system...${NC}"
  docker exec subgraphrag_api python scripts/stage_ingest.py --sample
  
  echo -e "${GREEN}Processing ingestion queue...${NC}"
  docker exec subgraphrag_api python scripts/ingest_worker.py --process-all
  
  echo -e "${GREEN}Merging FAISS index...${NC}"
  docker exec subgraphrag_api python scripts/merge_faiss.py
  
  echo -e "${GREEN}Sample data initialization completed.${NC}"
}

# Function to display final information
show_info() {
  if [ "$DETACHED" = false ]; then
    # Don't show info in foreground mode as it's still running
    return 0
  fi
  
  echo -e "${BLUE}"
  echo "=================================================="
  echo "        SubgraphRAG+ Docker Setup Complete!       "
  echo "=================================================="
  echo -e "${NC}"
  echo -e "${GREEN}You can now access:${NC}"
  echo -e "1. Neo4j Browser: ${YELLOW}http://localhost:7474${NC}"
  echo -e "   Username: neo4j, Password: password"
  echo -e ""
  echo -e "2. API Endpoint:  ${YELLOW}http://localhost:8000${NC}"
  echo -e "3. API Documentation: ${YELLOW}http://localhost:8000/docs${NC}"
  echo -e ""
  echo -e "${GREEN}Useful commands:${NC}"
  echo -e "- View logs: ${YELLOW}docker logs -f subgraphrag_api${NC}"
  echo -e "- Stop services: ${YELLOW}./bin/docker-setup.sh stop${NC}"
  echo -e "- Run tests: ${YELLOW}docker exec subgraphrag_api pytest${NC}"
  echo -e "- Create backup: ${YELLOW}./bin/docker-setup.sh backup${NC}"
  echo -e ""
  echo -e "For more information, see the documentation in the ${YELLOW}docs/${NC} directory."
}

# Main function
main() {
  echo -e "${GREEN}Starting Docker-based environment setup...${NC}"
  
  check_docker
  check_required_files
  create_directories
  create_env_file
  build_images
  start_services
  load_sample_data
  show_info
}

# Run the main function
main