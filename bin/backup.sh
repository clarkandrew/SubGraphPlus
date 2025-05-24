#!/bin/bash
# Script for SubgraphRAG+ backup operations
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
echo "           SubgraphRAG+ Backup Utility            "
echo "=================================================="
echo -e "${NC}"

# Define variables
ACTION="backup"
BACKUP_ID=""
BACKUP_DIR="backups"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    backup|restore|list)
      ACTION="$1"
      shift
      ;;
    -i|--id)
      BACKUP_ID="$2"
      shift 2
      ;;
    -d|--dir)
      BACKUP_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [action] [options]"
      echo "Actions:"
      echo "  backup               Create a new backup (default)"
      echo "  restore              Restore from a backup"
      echo "  list                 List available backups"
      echo ""
      echo "Options:"
      echo "  -i, --id ID          Backup ID for restore (default: latest)"
      echo "  -d, --dir DIR        Backup directory (default: backups)"
      echo "  -h, --help           Show this help message"
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

# Build backup command
BACKUP_CMD="python scripts/backup_restore.py --action $ACTION"

if [ -n "$BACKUP_ID" ]; then
  BACKUP_CMD="$BACKUP_CMD --backup-id $BACKUP_ID"
fi

if [ -n "$BACKUP_DIR" ]; then
  BACKUP_CMD="$BACKUP_CMD --backup-dir $BACKUP_DIR"
fi

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Print action information
case $ACTION in
  "backup")
    echo -e "${GREEN}Creating new backup in $BACKUP_DIR...${NC}"
    ;;
  "restore")
    if [ -n "$BACKUP_ID" ]; then
      echo -e "${YELLOW}Restoring from backup $BACKUP_ID...${NC}"
    else
      echo -e "${YELLOW}Restoring from latest backup...${NC}"
    fi
    ;;
  "list")
    echo -e "${GREEN}Listing available backups in $BACKUP_DIR...${NC}"
    ;;
esac

# Run backup command
$BACKUP_CMD

echo -e "${BLUE}"
echo "=================================================="
echo "           Operation completed                    "
echo "=================================================="
echo -e "${NC}"