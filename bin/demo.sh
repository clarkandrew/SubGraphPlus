#!/bin/bash
# SubgraphRAG+ Demo Script
# This script demonstrates the core functionality of SubgraphRAG+

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
echo "          SubgraphRAG+ Interactive Demo           "
echo "=================================================="
echo -e "${NC}"

# Check if using Docker or local development
if command -v docker &> /dev/null && docker ps --format "{{.Names}}" | grep -q "subgraphrag_api"; then
    USING_DOCKER=true
    echo -e "${GREEN}Detected Docker deployment${NC}"
    API_KEY=$(docker exec subgraphrag_api bash -c 'echo $API_KEY_SECRET')
    if [ -z "$API_KEY" ]; then
        API_KEY="changeme"
    fi
else
    USING_DOCKER=false
    echo -e "${GREEN}Detected local deployment${NC}"
    # Try to read from .env file
    if [ -f ".env" ]; then
        API_KEY=$(grep API_KEY_SECRET .env | cut -d= -f2)
    fi
    if [ -z "$API_KEY" ]; then
        API_KEY="default_key_for_dev_only"
    fi
fi

# Function to make API requests
make_request() {
    local endpoint=$1
    local data=$2
    local method=${3:-POST}
    
    echo -e "${YELLOW}Making $method request to $endpoint...${NC}"
    
    local result=$(curl -s -X $method "http://localhost:8000$endpoint" \
        -H "X-API-KEY: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$data")
    
    echo -e "${GREEN}Response:${NC}"
    echo $result | jq '.' || echo $result
    echo ""
}

# Function to make streaming API requests
make_streaming_request() {
    local endpoint=$1
    local data=$2
    
    echo -e "${YELLOW}Making streaming request to $endpoint...${NC}"
    echo -e "${GREEN}Response (stream):${NC}"
    
    curl -N -X POST "http://localhost:8000$endpoint" \
        -H "X-API-KEY: $API_KEY" \
        -H "Content-Type: application/json" \
        -H "Accept: text/event-stream" \
        -d "$data"
    
    echo ""
}

# Function to check API health
check_health() {
    echo -e "${YELLOW}Checking API health...${NC}"
    local health=$(curl -s "http://localhost:8000/healthz")
    
    if echo "$health" | grep -q "ok"; then
        echo -e "${GREEN}API is healthy!${NC}"
        return 0
    else
        echo -e "${RED}API is not healthy${NC}"
        return 1
    fi
}

# Check if API is running
if ! check_health; then
    echo -e "${RED}API is not running. Starting it...${NC}"
    
    if [ "$USING_DOCKER" = true ]; then
        echo -e "${YELLOW}Starting Docker containers...${NC}"
        ./bin/docker-setup.sh start
        sleep 5
    else
        echo -e "${YELLOW}Starting local API...${NC}"
        ./bin/run.sh &
        sleep 5
    fi
    
    # Check again
    if ! check_health; then
        echo -e "${RED}Failed to start API. Please check logs and try again.${NC}"
        exit 1
    fi
fi

# Check if we have sample data (try to make a simple query)
echo -e "${YELLOW}Checking if sample data is loaded...${NC}"
sample_check=$(curl -s -X POST "http://localhost:8000/query" \
    -H "X-API-KEY: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"question": "Who founded Tesla?", "visualize_graph": true}')

if echo "$sample_check" | grep -q "NO_ENTITY_MATCH\|NO_RELEVANT_TRIPLES"; then
    echo -e "${YELLOW}Sample data not detected. Initializing with sample data...${NC}"
    
    if [ "$USING_DOCKER" = true ]; then
        ./bin/docker-setup.sh sample-data
    else
        # Load sample data locally
        python scripts/stage_ingest.py --sample
        python scripts/ingest_worker.py --process-all
        python scripts/merge_faiss.py
    fi
fi

# Interactive demo
echo -e "${BLUE}"
echo "=================================================="
echo "             Interactive Demo Menu                "
echo "=================================================="
echo -e "${NC}"

while true; do
    echo "Select an option:"
    echo "1. Query: Who founded Tesla?"
    echo "2. Query: What companies does Elon Musk run?"
    echo "3. Query: Where is the headquarters of OpenAI?"
    echo "4. Browse knowledge graph (first 10 entries)"
    echo "5. Submit feedback on a query"
    echo "6. Custom query (enter your own question)"
    echo "7. Exit demo"
    read -p "Enter your choice (1-7): " choice
    
    case $choice in
        1)
            make_streaming_request "/query" '{"question": "Who founded Tesla?", "visualize_graph": true}'
            ;;
        2)
            make_streaming_request "/query" '{"question": "What companies does Elon Musk run?", "visualize_graph": true}'
            ;;
        3)
            make_streaming_request "/query" '{"question": "Where is the headquarters of OpenAI?", "visualize_graph": true}'
            ;;
        4)
            make_request "/graph/browse?page=1&limit=10" "{}" "GET"
            ;;
        5)
            make_request "/feedback" '{"query_id": "demo_query", "is_correct": true, "comment": "Great answer!", "expected_answer": null}'
            ;;
        6)
            read -p "Enter your question: " custom_question
            make_streaming_request "/query" "{\"question\": \"$custom_question\", \"visualize_graph\": true}"
            ;;
        7)
            echo -e "${GREEN}Exiting demo. Thanks for trying SubgraphRAG+!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Please try again.${NC}"
            ;;
    esac
    
    echo -e "${BLUE}=================================================${NC}"
done