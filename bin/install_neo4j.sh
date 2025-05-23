#!/bin/bash
# Script to install Neo4j locally without Docker
# For SubgraphRAG+ development and deployment

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
echo "          Neo4j Local Installation Script         "
echo "=================================================="
echo -e "${NC}"

# Define variables
NEO4J_VERSION="4.4.30"
APOC_VERSION="4.4.0.15"
INSTALL_DIR=""
USE_SUDO=false
UPDATE_ENV=true
START_AFTER_INSTALL=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      NEO4J_VERSION="$2"
      shift 2
      ;;
    --apoc-version)
      APOC_VERSION="$2"
      shift 2
      ;;
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --sudo)
      USE_SUDO=true
      shift
      ;;
    --no-update-env)
      UPDATE_ENV=false
      shift
      ;;
    --no-start)
      START_AFTER_INSTALL=false
      shift
      ;;
    -h|--help)
      echo "Usage: ./bin/install_neo4j.sh [options]"
      echo ""
      echo "Options:"
      echo "  --version VERSION     Neo4j version to install (default: 4.4.30)"
      echo "  --apoc-version VER    APOC plugin version (default: 4.4.0.15)"
      echo "  --install-dir DIR     Installation directory (defaults to platform-specific)"
      echo "  --sudo                Use sudo for installation (Linux only)"
      echo "  --no-update-env       Don't update .env file with Neo4j connection details"
      echo "  --no-start            Don't start Neo4j after installation"
      echo "  -h, --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './bin/install_neo4j.sh --help' for usage information."
      exit 1
      ;;
  esac
done

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=linux;;
    Darwin*)    OS_TYPE=macos;;
    CYGWIN*)    OS_TYPE=windows;;
    MINGW*)     OS_TYPE=windows;;
    *)          OS_TYPE="unknown"
esac

echo -e "${GREEN}Detected operating system: ${OS_TYPE}${NC}"

# Set default installation directory if not specified
if [ -z "$INSTALL_DIR" ]; then
  case $OS_TYPE in
    macos)
      INSTALL_DIR="$HOME/neo4j"
      ;;
    linux)
      if [ "$USE_SUDO" = true ]; then
        INSTALL_DIR="/opt/neo4j"
      else
        INSTALL_DIR="$HOME/neo4j"
      fi
      ;;
    windows)
      INSTALL_DIR="$HOME/neo4j"
      ;;
    *)
      echo -e "${RED}Unsupported operating system: $OS_TYPE${NC}"
      exit 1
      ;;
  esac
fi

# Function to install Neo4j on macOS
install_macos() {
  if command -v brew &> /dev/null; then
    echo -e "${GREEN}Installing Neo4j using Homebrew...${NC}"
    
    # Check if specific version is requested
    if [ "$NEO4J_VERSION" != "4.4.30" ]; then
      echo -e "${YELLOW}Warning: Specific version $NEO4J_VERSION requested, but Homebrew may install the latest version.${NC}"
      echo -e "${YELLOW}Consider using Neo4j Desktop for specific versions: https://neo4j.com/download/${NC}"
    fi
    
    # Install Neo4j
    brew install neo4j
    
    # Set installation directory for later use
    ACTUAL_INSTALL_DIR="$(brew --prefix)/opt/neo4j"
    PLUGINS_DIR="$(brew --prefix)/var/neo4j/plugins"
    CONF_DIR="$(brew --prefix)/etc/neo4j"
    
    # Install APOC plugin
    echo -e "${GREEN}Installing APOC plugin version $APOC_VERSION...${NC}"
    mkdir -p "$PLUGINS_DIR"
    curl -L "https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/${APOC_VERSION}/apoc-${APOC_VERSION}-all.jar" -o "$PLUGINS_DIR/apoc-${APOC_VERSION}-all.jar"
    
    # Update configuration to allow APOC procedures
    if grep -q "dbms.security.procedures.unrestricted" "$CONF_DIR/neo4j.conf"; then
      sed -i '' 's/^#dbms.security.procedures.unrestricted=.*/dbms.security.procedures.unrestricted=apoc.*/' "$CONF_DIR/neo4j.conf"
    else
      echo "dbms.security.procedures.unrestricted=apoc.*" >> "$CONF_DIR/neo4j.conf"
    fi
    
    # Ensure remote connections are allowed
    if grep -q "dbms.default_listen_address" "$CONF_DIR/neo4j.conf"; then
      sed -i '' 's/^#dbms.default_listen_address=.*/dbms.default_listen_address=0.0.0.0/' "$CONF_DIR/neo4j.conf"
    else
      echo "dbms.default_listen_address=0.0.0.0" >> "$CONF_DIR/neo4j.conf"
    fi
    
    # Start Neo4j service
    if [ "$START_AFTER_INSTALL" = true ]; then
      echo -e "${GREEN}Starting Neo4j service...${NC}"
      brew services start neo4j
      
      # Wait for Neo4j to start
      echo -e "${YELLOW}Waiting for Neo4j to start...${NC}"
      sleep 10
      
      # Reset password to default
      echo -e "${YELLOW}Setting Neo4j password (default: 'password')...${NC}"
      if cypher-shell -u neo4j -p neo4j -d system "ALTER USER neo4j SET PASSWORD 'password'"; then
        echo -e "${GREEN}Password set successfully.${NC}"
      else
        echo -e "${YELLOW}Password may be already set. If you need to reset it, run:${NC}"
        echo -e "  cypher-shell -u neo4j -p your_current_password -d system \"ALTER USER neo4j SET PASSWORD 'password'\""
      fi
    fi
    
    echo -e "${GREEN}Neo4j installation completed!${NC}"
    echo -e "${GREEN}Neo4j is installed at: $ACTUAL_INSTALL_DIR${NC}"
    echo -e "${GREEN}Configuration file: $CONF_DIR/neo4j.conf${NC}"
    echo -e "${GREEN}Plugins directory: $PLUGINS_DIR${NC}"
    
  else
    echo -e "${YELLOW}Homebrew not found. Installing from binary package...${NC}"
    install_from_binary
  fi
}

# Function to install Neo4j on Linux
install_linux() {
  if [ -f /etc/debian_version ] && command -v apt-get &> /dev/null; then
    echo -e "${GREEN}Detected Debian/Ubuntu system, installing with apt...${NC}"
    
    # Add Neo4j repository key
    if [ "$USE_SUDO" = true ]; then
      sudo wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
      echo "deb https://debian.neo4j.com stable ${NEO4J_VERSION}" | sudo tee /etc/apt/sources.list.d/neo4j.list
      sudo apt-get update
      
      # Install Neo4j
      sudo apt-get install -y neo4j
      
      # Set directories for later use
      ACTUAL_INSTALL_DIR="/var/lib/neo4j"
      PLUGINS_DIR="/var/lib/neo4j/plugins"
      CONF_DIR="/etc/neo4j"
      
      # Install APOC plugin
      echo -e "${GREEN}Installing APOC plugin version $APOC_VERSION...${NC}"
      sudo curl -L "https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/${APOC_VERSION}/apoc-${APOC_VERSION}-all.jar" -o "$PLUGINS_DIR/apoc-${APOC_VERSION}-all.jar"
      
      # Update configuration to allow APOC procedures
      if sudo grep -q "dbms.security.procedures.unrestricted" "$CONF_DIR/neo4j.conf"; then
        sudo sed -i 's/^#dbms.security.procedures.unrestricted=.*/dbms.security.procedures.unrestricted=apoc.*/' "$CONF_DIR/neo4j.conf"
      else
        echo "dbms.security.procedures.unrestricted=apoc.*" | sudo tee -a "$CONF_DIR/neo4j.conf"
      fi
      
      # Ensure remote connections are allowed
      if sudo grep -q "dbms.default_listen_address" "$CONF_DIR/neo4j.conf"; then
        sudo sed -i 's/^#dbms.default_listen_address=.*/dbms.default_listen_address=0.0.0.0/' "$CONF_DIR/neo4j.conf"
      else
        echo "dbms.default_listen_address=0.0.0.0" | sudo tee -a "$CONF_DIR/neo4j.conf"
      fi
      
      # Start Neo4j service
      if [ "$START_AFTER_INSTALL" = true ]; then
        echo -e "${GREEN}Starting Neo4j service...${NC}"
        sudo systemctl enable neo4j
        sudo systemctl start neo4j
        
        # Wait for Neo4j to start
        echo -e "${YELLOW}Waiting for Neo4j to start...${NC}"
        sleep 15
        
        # Reset password to default
        echo -e "${YELLOW}Setting Neo4j password (default: 'password')...${NC}"
        if sudo -u neo4j cypher-shell -u neo4j -p neo4j -d system "ALTER USER neo4j SET PASSWORD 'password'"; then
          echo -e "${GREEN}Password set successfully.${NC}"
        else
          echo -e "${YELLOW}Password may be already set. If you need to reset it, run:${NC}"
          echo -e "  sudo -u neo4j cypher-shell -u neo4j -p your_current_password -d system \"ALTER USER neo4j SET PASSWORD 'password'\""
        fi
      fi
    else
      echo -e "${YELLOW}Not using sudo. Installing from binary package...${NC}"
      install_from_binary
    fi
  else
    echo -e "${YELLOW}APT package manager not found or not a Debian-based system.${NC}"
    echo -e "${YELLOW}Installing from binary package...${NC}"
    install_from_binary
  fi
}

# Function to install Neo4j from binary package (for all platforms)
install_from_binary() {
  echo -e "${GREEN}Installing Neo4j from binary package...${NC}"
  
  # Determine the appropriate package URL based on OS
  case $OS_TYPE in
    macos)
      PACKAGE_URL="https://dist.neo4j.org/neo4j-community-${NEO4J_VERSION}-unix.tar.gz"
      ;;
    linux)
      PACKAGE_URL="https://dist.neo4j.org/neo4j-community-${NEO4J_VERSION}-unix.tar.gz"
      ;;
    windows)
      PACKAGE_URL="https://dist.neo4j.org/neo4j-community-${NEO4J_VERSION}-windows.zip"
      ;;
  esac
  
  # Create installation directory
  mkdir -p "$INSTALL_DIR"
  
  # Download and extract Neo4j
  echo -e "${GREEN}Downloading Neo4j ${NEO4J_VERSION}...${NC}"
  if command -v curl &> /dev/null; then
    curl -L "$PACKAGE_URL" -o /tmp/neo4j.archive
  elif command -v wget &> /dev/null; then
    wget "$PACKAGE_URL" -O /tmp/neo4j.archive
  else
    echo -e "${RED}Neither curl nor wget are available. Please install one of them and try again.${NC}"
    exit 1
  fi
  
  # Extract the archive
  echo -e "${GREEN}Extracting Neo4j...${NC}"
  case $OS_TYPE in
    windows)
      if command -v unzip &> /dev/null; then
        unzip -q /tmp/neo4j.archive -d "$INSTALL_DIR"
      else
        echo -e "${RED}Unzip utility not found. Please install unzip and try again.${NC}"
        exit 1
      fi
      ;;
    *)
      tar -xf /tmp/neo4j.archive -C "$INSTALL_DIR" --strip-components=1
      ;;
  esac
  
  # Clean up the downloaded archive
  rm /tmp/neo4j.archive
  
  # Set directories for later use
  ACTUAL_INSTALL_DIR="$INSTALL_DIR"
  PLUGINS_DIR="$INSTALL_DIR/plugins"
  CONF_DIR="$INSTALL_DIR/conf"
  
  # Install APOC plugin
  echo -e "${GREEN}Installing APOC plugin version $APOC_VERSION...${NC}"
  if command -v curl &> /dev/null; then
    curl -L "https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/${APOC_VERSION}/apoc-${APOC_VERSION}-all.jar" -o "$PLUGINS_DIR/apoc-${APOC_VERSION}-all.jar"
  elif command -v wget &> /dev/null; then
    wget "https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/${APOC_VERSION}/apoc-${APOC_VERSION}-all.jar" -O "$PLUGINS_DIR/apoc-${APOC_VERSION}-all.jar"
  fi
  
  # Update configuration to allow APOC procedures
  if grep -q "dbms.security.procedures.unrestricted" "$CONF_DIR/neo4j.conf"; then
    sed -i.bak 's/^#dbms.security.procedures.unrestricted=.*/dbms.security.procedures.unrestricted=apoc.*/' "$CONF_DIR/neo4j.conf"
  else
    echo "dbms.security.procedures.unrestricted=apoc.*" >> "$CONF_DIR/neo4j.conf"
  fi
  
  # Ensure remote connections are allowed
  if grep -q "dbms.default_listen_address" "$CONF_DIR/neo4j.conf"; then
    sed -i.bak 's/^#dbms.default_listen_address=.*/dbms.default_listen_address=0.0.0.0/' "$CONF_DIR/neo4j.conf"
  else
    echo "dbms.default_listen_address=0.0.0.0" >> "$CONF_DIR/neo4j.conf"
  fi
  
  # Set execution permissions on Neo4j binaries
  chmod +x "$INSTALL_DIR/bin/neo4j"
  chmod +x "$INSTALL_DIR/bin/cypher-shell"
  
  # Start Neo4j service (binary installation)
  if [ "$START_AFTER_INSTALL" = true ]; then
    echo -e "${GREEN}Starting Neo4j server...${NC}"
    "$INSTALL_DIR/bin/neo4j" start
    
    # Wait for Neo4j to start
    echo -e "${YELLOW}Waiting for Neo4j to start...${NC}"
    sleep 15
    
    # Reset password to default
    echo -e "${YELLOW}Setting Neo4j password (default: 'password')...${NC}"
    if "$INSTALL_DIR/bin/cypher-shell" -u neo4j -p neo4j -d system "ALTER USER neo4j SET PASSWORD 'password'"; then
      echo -e "${GREEN}Password set successfully.${NC}"
    else
      echo -e "${YELLOW}Password may be already set. If you need to reset it, run:${NC}"
      echo -e "  $INSTALL_DIR/bin/cypher-shell -u neo4j -p your_current_password -d system \"ALTER USER neo4j SET PASSWORD 'password'\""
    fi
  fi
}

# Main installation logic
case $OS_TYPE in
  macos)
    install_macos
    ;;
  linux)
    install_linux
    ;;
  windows)
    echo -e "${YELLOW}For Windows, we recommend using Neo4j Desktop.${NC}"
    echo -e "${YELLOW}Download from: https://neo4j.com/download/${NC}"
    echo -e "${YELLOW}Alternatively, you can install using binary package.${NC}"
    install_from_binary
    ;;
  *)
    echo -e "${RED}Unsupported operating system: $OS_TYPE${NC}"
    exit 1
    ;;
esac

# Update .env file with Neo4j connection details
if [ "$UPDATE_ENV" = true ]; then
  echo -e "${GREEN}Updating .env file with Neo4j connection details...${NC}"
  
  # Check if .env file exists
  if [ -f ".env" ]; then
    # Update existing entries or add new ones
    if grep -q "NEO4J_URI=" ".env"; then
      sed -i.bak 's|^NEO4J_URI=.*|NEO4J_URI=neo4j://localhost:7687|' ".env"
    else
      echo "NEO4J_URI=neo4j://localhost:7687" >> ".env"
    fi
    
    if grep -q "NEO4J_USER=" ".env"; then
      sed -i.bak 's/^NEO4J_USER=.*/NEO4J_USER=neo4j/' ".env"
    else
      echo "NEO4J_USER=neo4j" >> ".env"
    fi
    
    if grep -q "NEO4J_PASSWORD=" ".env"; then
      sed -i.bak 's/^NEO4J_PASSWORD=.*/NEO4J_PASSWORD=password/' ".env"
    else
      echo "NEO4J_PASSWORD=password" >> ".env"
    fi
    
    if grep -q "USE_LOCAL_NEO4J=" ".env"; then
      sed -i.bak 's/^USE_LOCAL_NEO4J=.*/USE_LOCAL_NEO4J=true/' ".env"
    else
      echo "USE_LOCAL_NEO4J=true" >> ".env"
    fi
    
    # Clean up backup files
    find . -name "*.bak" -type f -delete
  else
    # Create new .env file
    cat > ".env" <<EOL
# Neo4j Connection (local installation)
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
USE_LOCAL_NEO4J=true

# API Security (change this in production)
API_KEY_SECRET=changeme

# Uncomment and set OPENAI_API_KEY if you're using OpenAI backend
# OPENAI_API_KEY=your_openai_key_here

# Logging level
LOG_LEVEL=INFO
EOL
  fi
  
  echo -e "${GREEN}.env file updated with Neo4j connection details.${NC}"
fi

# Print summary information
echo -e "${BLUE}"
echo "=================================================="
echo "           Neo4j Installation Complete!           "
echo "=================================================="
echo -e "${NC}"
echo -e "${GREEN}Neo4j Information:${NC}"
echo -e " - Installation path: ${ACTUAL_INSTALL_DIR}"
echo -e " - Plugins directory: ${PLUGINS_DIR}"
echo -e " - Configuration file: ${CONF_DIR}/neo4j.conf"
echo -e " - Neo4j version: ${NEO4J_VERSION}"
echo -e " - APOC plugin version: ${APOC_VERSION}"
echo -e ""
echo -e "${GREEN}Connection Information:${NC}"
echo -e " - URI: neo4j://localhost:7687"
echo -e " - HTTP interface: http://localhost:7474"
echo -e " - Username: neo4j"
echo -e " - Password: password"
echo -e ""
echo -e "${GREEN}Next Steps:${NC}"
echo -e " 1. Access Neo4j Browser at http://localhost:7474"
echo -e " 2. Run './bin/setup_dev.sh --use-local-neo4j' to set up the development environment"
echo -e " 3. Run './bin/run.sh' to start the application server"
echo -e ""
echo -e "${YELLOW}For production environments, be sure to:${NC}"
echo -e " - Secure your Neo4j installation with a strong password"
echo -e " - Configure proper authentication and network settings"
echo -e " - Consider setting up a Neo4j cluster for high availability"
echo -e ""
echo -e "${GREEN}Documentation: https://neo4j.com/docs/${NC}"