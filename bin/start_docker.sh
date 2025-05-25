#!/bin/bash
# Helper script to start Docker on different operating systems

# Define colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Docker Startup Helper${NC}"
echo "=========================="

# Detect operating system
OS=$(uname -s)

case "$OS" in
    Darwin*)
        echo -e "${YELLOW}Detected macOS${NC}"
        if [ -d "/Applications/Docker.app" ]; then
            echo -e "${GREEN}Starting Docker Desktop...${NC}"
            open -a Docker
            echo -e "${YELLOW}Waiting for Docker to start...${NC}"
            # Wait for Docker to be ready
            while ! docker info >/dev/null 2>&1; do
                echo -n "."
                sleep 2
            done
            echo -e "\n${GREEN}Docker is now running!${NC}"
        else
            echo -e "${RED}Docker Desktop not found in /Applications/Docker.app${NC}"
            echo -e "${YELLOW}Please install Docker Desktop from: https://docs.docker.com/desktop/install/mac-install/${NC}"
            exit 1
        fi
        ;;
    Linux*)
        echo -e "${YELLOW}Detected Linux${NC}"
        echo -e "${GREEN}Attempting to start Docker service...${NC}"
        if command -v systemctl >/dev/null 2>&1; then
            sudo systemctl start docker
            sudo systemctl enable docker
            echo -e "${GREEN}Docker service started.${NC}"
        elif command -v service >/dev/null 2>&1; then
            sudo service docker start
            echo -e "${GREEN}Docker service started.${NC}"
        else
            echo -e "${RED}Unable to start Docker service automatically.${NC}"
            echo -e "${YELLOW}Please start Docker manually and ensure your user is in the docker group:${NC}"
            echo -e "sudo usermod -aG docker \$USER"
            echo -e "newgrp docker"
            exit 1
        fi
        ;;
    CYGWIN*|MINGW32*|MSYS*|MINGW*)
        echo -e "${YELLOW}Detected Windows${NC}"
        echo -e "${YELLOW}Please start Docker Desktop manually.${NC}"
        echo -e "You can find it in your Start Menu or by running 'Docker Desktop' from the command line."
        exit 1
        ;;
    *)
        echo -e "${RED}Unknown operating system: $OS${NC}"
        echo -e "${YELLOW}Please start Docker manually for your operating system.${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}Docker is ready to use!${NC}" 