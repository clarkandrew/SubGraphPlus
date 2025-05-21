#!/bin/bash
# SubgraphRAG+ Docker Setup Script
# This script helps set up and manage the SubgraphRAG+ Docker environment

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

# Function to check if Docker and Docker Compose are installed
check_docker() {
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
  
  if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose and try again.${NC}"
    echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
  fi
}

# Function to start the system
start_system() {
  echo -e "${GREEN}Starting SubgraphRAG+ with Docker Compose...${NC}"
  
  # Use docker compose if available, otherwise fallback to docker-compose
  if docker compose version &> /dev/null; then
    docker compose up -d
  else
    docker-compose up -d
  fi
  
  echo -e "${GREEN}Waiting for services to initialize...${NC}"
  sleep 5
  
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
  
  echo -e "${BLUE}"
  echo "=================================================="
  echo "           SubgraphRAG+ is now running!           "
  echo "=================================================="
  echo -e "${NC}"
  echo "Neo4j Browser: http://localhost:7474"
  echo "API Endpoint:  http://localhost:8000"
  echo "API Docs:      http://localhost:8000/docs"
}

# Function to stop the system
stop_system() {
  echo -e "${YELLOW}Stopping SubgraphRAG+ services...${NC}"
  
  # Use docker compose if available, otherwise fallback to docker-compose
  if docker compose version &> /dev/null; then
    docker compose down
  else
    docker-compose down
  fi
  
  echo -e "${GREEN}All services stopped.${NC}"
}

# Function to rebuild the system
rebuild_system() {
  echo -e "${YELLOW}Rebuilding SubgraphRAG+ services...${NC}"
  
  # Use docker compose if available, otherwise fallback to docker-compose
  if docker compose version &> /dev/null; then
    docker compose build --no-cache
    docker compose up -d
  else
    docker-compose build --no-cache
    docker-compose up -d
  fi
  
  echo -e "${GREEN}System rebuilt and restarted.${NC}"
}

# Function to view logs
view_logs() {
  echo -e "${GREEN}Viewing logs for all services (press Ctrl+C to exit)...${NC}"
  
  # Use docker compose if available, otherwise fallback to docker-compose
  if docker compose version &> /dev/null; then
    docker compose logs -f
  else
    docker-compose logs -f
  fi
}

# Function to show the status of services
show_status() {
  echo -e "${GREEN}Current status of SubgraphRAG+ services:${NC}"
  
  # Use docker compose if available, otherwise fallback to docker-compose
  if docker compose version &> /dev/null; then
    docker compose ps
  else
    docker-compose ps
  fi
}

# Function to check containers' resource usage
show_resources() {
  echo -e "${GREEN}Resource usage of SubgraphRAG+ containers:${NC}"
  docker stats --no-stream subgraphrag_api subgraphrag_neo4j
}

# Function to initialize the volume with sample data
init_sample_data() {
  echo -e "${GREEN}Initializing system with sample data...${NC}"
  
  # Check if system is running
  if ! docker ps | grep -q "subgraphrag_api"; then
    echo -e "${YELLOW}SubgraphRAG+ is not running. Starting services first...${NC}"
    start_system
    sleep 5
  fi
  
  echo -e "${GREEN}Adding sample triples to the system...${NC}"
  docker exec subgraphrag_api python scripts/stage_ingest.py --sample
  
  echo -e "${GREEN}Processing ingestion queue...${NC}"
  docker exec subgraphrag_api python scripts/ingest_worker.py --process-all
  
  echo -e "${GREEN}Merging FAISS index...${NC}"
  docker exec subgraphrag_api python scripts/merge_faiss.py
  
  echo -e "${GREEN}Sample data initialization completed.${NC}"
}

# Function to access the API shell
api_shell() {
  echo -e "${GREEN}Opening shell in the API container...${NC}"
  docker exec -it subgraphrag_api /bin/bash
}

# Function to access the Neo4j shell
neo4j_shell() {
  echo -e "${GREEN}Opening shell in the Neo4j container...${NC}"
  docker exec -it subgraphrag_neo4j /bin/bash
}

# Function to run tests
run_tests() {
  echo -e "${GREEN}Running tests in the API container...${NC}"
  docker exec subgraphrag_api pytest
}

# Function to create a backup
create_backup() {
  echo -e "${GREEN}Creating backup...${NC}"
  BACKUP_NAME="subgraphrag_backup_$(date +%Y%m%d_%H%M%S)"
  mkdir -p ./backups
  
  echo -e "${YELLOW}Creating Neo4j backup...${NC}"
  docker exec subgraphrag_neo4j neo4j-admin dump --database=neo4j --to=/data/${BACKUP_NAME}.dump
  docker cp subgraphrag_neo4j:/data/${BACKUP_NAME}.dump ./backups/
  
  echo -e "${YELLOW}Creating SQLite backup...${NC}"
  docker exec subgraphrag_api sqlite3 data/staging.db ".backup '/tmp/${BACKUP_NAME}.db'"
  docker cp subgraphrag_api:/tmp/${BACKUP_NAME}.db ./backups/
  
  echo -e "${YELLOW}Creating FAISS backup...${NC}"
  docker exec subgraphrag_api tar -czf /tmp/${BACKUP_NAME}_faiss.tar.gz -C /app/data faiss
  docker cp subgraphrag_api:/tmp/${BACKUP_NAME}_faiss.tar.gz ./backups/
  
  echo -e "${GREEN}Backup completed to ./backups/${BACKUP_NAME}.*${NC}"
}

# Function to show help
show_help() {
  echo -e "${BLUE}SubgraphRAG+ Docker Management Script${NC}"
  echo ""
  echo "Usage: $0 [command]"
  echo ""
  echo "Commands:"
  echo "  start         Start the SubgraphRAG+ services"
  echo "  stop          Stop the SubgraphRAG+ services"
  echo "  restart       Restart the SubgraphRAG+ services"
  echo "  rebuild       Rebuild and restart the services"
  echo "  status        Show status of the services"
  echo "  logs          View the logs of all services"
  echo "  resources     Show resource usage of containers"
  echo "  sample-data   Initialize the system with sample data"
  echo "  api-shell     Open a shell in the API container"
  echo "  neo4j-shell   Open a shell in the Neo4j container"
  echo "  tests         Run tests in the API container"
  echo "  backup        Create a backup of all data"
  echo "  help          Show this help message"
  echo ""
}

# Check if Docker is installed
check_docker

# Parse command line arguments
if [ $# -eq 0 ]; then
  show_help
  exit 0
fi

case "$1" in
  start)
    start_system
    ;;
  stop)
    stop_system
    ;;
  restart)
    stop_system
    start_system
    ;;
  rebuild)
    rebuild_system
    ;;
  status)
    show_status
    ;;
  logs)
    view_logs
    ;;
  resources)
    show_resources
    ;;
  sample-data)
    init_sample_data
    ;;
  api-shell)
    api_shell
    ;;
  neo4j-shell)
    neo4j_shell
    ;;
  tests)
    run_tests
    ;;
  backup)
    create_backup
    ;;
  help|--help|-h)
    show_help
    ;;
  *)
    echo -e "${RED}Unknown command: $1${NC}"
    show_help
    exit 1
    ;;
esac