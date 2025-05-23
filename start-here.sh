#!/bin/bash

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print welcome banner
echo -e "${BLUE}=================================================="
echo -e "            SubgraphRAG+ Starter                 "
echo -e "==================================================${NC}"
echo

# Check for make
if ! command -v make &> /dev/null; then
    echo -e "${YELLOW}Error: 'make' is not installed or not in your PATH.${NC}"
    echo -e "Please install make before continuing:"
    echo -e "  - Ubuntu/Debian: sudo apt-get install make"
    echo -e "  - macOS: xcode-select --install"
    echo -e "  - Windows: Install Make through chocolatey or WSL"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Error: Python 3 is not installed or not in your PATH.${NC}"
    echo -e "Please install Python 3.9+ before continuing:"
    echo -e "  - Download from: https://www.python.org/downloads/"
    exit 1
fi

# Make scripts executable
chmod +x bin/*.sh 2>/dev/null

# Ask user what to do
echo -e "${GREEN}Welcome to SubgraphRAG+!${NC}"
echo -e "What would you like to do?"
echo
echo -e "1) ${GREEN}Setup with Docker${NC} (recommended)"
echo -e "2) ${GREEN}Setup without Docker${NC} (local Neo4j installation)"
echo -e "3) View available commands"
echo -e "4) Run tests only"
echo -e "5) Start the server (if already set up)"
echo -e "6) Exit"
echo
read -p "Enter your choice (1-6): " choice

case "$choice" in
    1)
        echo -e "${GREEN}Running setup with Docker...${NC}"
        echo -e "${YELLOW}This will set up Neo4j in Docker and install all dependencies.${NC}"
        echo -e "${YELLOW}Requires Docker to be installed and running.${NC}"
        read -p "Proceed? (y/n): " confirm
        if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
            make setup-all
        else
            echo -e "${YELLOW}Setup canceled.${NC}"
            exit 0
        fi
        ;;
    2)
        echo -e "${GREEN}Running setup without Docker...${NC}"
        echo -e "${YELLOW}This will install Neo4j locally and set up the development environment.${NC}"
        
        # Check if Neo4j is already installed
        if command -v neo4j &> /dev/null || command -v cypher-shell &> /dev/null || [ -d "/Applications/Neo4j Desktop.app" ]; then
            echo -e "${GREEN}Neo4j appears to be already installed.${NC}"
            read -p "Use existing Neo4j installation? (y/n): " use_existing
            if [[ "$use_existing" == "y" || "$use_existing" == "Y" ]]; then
                ./bin/setup_dev.sh --use-local-neo4j
            else
                ./bin/install_neo4j.sh
                ./bin/setup_dev.sh --use-local-neo4j
            fi
        else
            echo -e "${YELLOW}Neo4j not detected. Installing...${NC}"
            ./bin/install_neo4j.sh
            ./bin/setup_dev.sh --use-local-neo4j
        fi
        ;;
    3)
        echo -e "${GREEN}Available commands:${NC}"
        echo -e "${BLUE}Make commands (Docker-based):${NC}"
        make help
        echo
        echo -e "${BLUE}Shell scripts (for all setups):${NC}"
        echo -e "  ./bin/run.sh               - Start the API server"
        echo -e "  ./bin/run_tests.sh         - Run tests"
        echo -e "  ./bin/install_neo4j.sh     - Install Neo4j locally (without Docker)"
        echo -e "  ./bin/setup_dev.sh         - Set up development environment"
        echo -e "  ./bin/backup.sh            - Backup/restore the system"
        ;;
    4)
        echo -e "${GREEN}Running tests...${NC}"
        ./bin/run_tests.sh
        ;;
    5)
        echo -e "${GREEN}Starting server...${NC}"
        ./bin/run.sh
        ;;
    6)
        echo -e "${GREEN}Exiting. Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice.${NC}"
        echo -e "Please run this script again and enter a number from 1-6."
        exit 1
        ;;
esac

echo
if [ "$choice" == "1" ] || [ "$choice" == "2" ]; then
    echo -e "${GREEN}Setup complete! 🎉${NC}"
    echo -e "Access the API at: http://localhost:8000"
    echo -e "API documentation: http://localhost:8000/docs"
    echo -e "Neo4j browser: http://localhost:7474"
    
    echo
    read -p "Would you like to start the server now? (y/n): " start_server
    if [[ "$start_server" == "y" || "$start_server" == "Y" ]]; then
        echo -e "${GREEN}Starting server...${NC}"
        ./bin/run.sh
    else
        if [ "$choice" == "1" ]; then
            echo -e "${GREEN}To start the server later, run: ${YELLOW}make serve${NC} or ${YELLOW}./bin/run.sh${NC}"
        else
            echo -e "${GREEN}To start the server later, run: ${YELLOW}./bin/run.sh${NC}"
        fi
    fi
fi