#!/bin/bash

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Ask user what to do
echo -e "${GREEN}Welcome to SubgraphRAG+!${NC}"
echo -e "What would you like to do?"
echo
echo -e "1) ${GREEN}Quick setup - Do everything automatically${NC} (recommended)"
echo -e "2) View available commands"
echo -e "3) Run tests only"
echo -e "4) Start the server (if already set up)"
echo -e "5) Exit"
echo
read -p "Enter your choice (1-5): " choice

case "$choice" in
    1)
        echo -e "${GREEN}Running complete setup...${NC}"
        make setup-all
        ;;
    2)
        echo -e "${GREEN}Available commands:${NC}"
        make help
        ;;
    3)
        echo -e "${GREEN}Running tests...${NC}"
        make test
        ;;
    4)
        echo -e "${GREEN}Starting server...${NC}"
        make serve
        ;;
    5)
        echo -e "${GREEN}Exiting. Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "${YELLOW}Invalid choice.${NC}"
        echo -e "Please run this script again and enter a number from 1-5."
        exit 1
        ;;
esac

echo
if [ "$choice" == "1" ]; then
    echo -e "${GREEN}Setup complete! 🎉${NC}"
    echo -e "You can now start the server with: ${YELLOW}make serve${NC}"
    echo -e "Access the API at: http://localhost:8000"
    echo -e "API documentation: http://localhost:8000/docs"
    echo -e "Neo4j browser: http://localhost:7474"
    
    echo
    read -p "Would you like to start the server now? (y/n): " start_server
    if [[ "$start_server" == "y" || "$start_server" == "Y" ]]; then
        echo -e "${GREEN}Starting server...${NC}"
        make serve
    else
        echo -e "${GREEN}To start the server later, run: ${YELLOW}make serve${NC}"
    fi
fi