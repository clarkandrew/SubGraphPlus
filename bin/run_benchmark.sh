#!/bin/bash
# Script to run SubgraphRAG+ benchmarks
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
echo "           SubgraphRAG+ Benchmark Runner          "
echo "=================================================="
echo -e "${NC}"

# Define variables
INPUT_FILE="evaluation/sample_questions.json"
OUTPUT_FILE="evaluation/results.json"
METRICS_FILE="evaluation/metrics.json"
GROUND_TRUTH_FILE=""
DETAILED_REPORT=false
ADVERSARIAL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input)
      INPUT_FILE="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    -m|--metrics)
      METRICS_FILE="$2"
      shift 2
      ;;
    -g|--ground-truth)
      GROUND_TRUTH_FILE="$2"
      shift 2
      ;;
    -r|--report)
      DETAILED_REPORT=true
      shift
      ;;
    -a|--adversarial)
      ADVERSARIAL=true
      INPUT_FILE="evaluation/adversarial_questions.json"
      OUTPUT_FILE="evaluation/adversarial_results.json"
      METRICS_FILE="evaluation/adversarial_metrics.json"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  -i, --input FILE       Input questions file (default: evaluation/sample_questions.json)"
      echo "  -o, --output FILE      Output results file (default: evaluation/results.json)"
      echo "  -m, --metrics FILE     Output metrics file (default: evaluation/metrics.json)"
      echo "  -g, --ground-truth FILE Ground truth file for advanced metrics"
      echo "  -r, --report           Generate detailed HTML report"
      echo "  -a, --adversarial      Run adversarial benchmark"
      echo "  -h, --help             Show this help message"
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
  export $(grep -v '^#' .env | xargs)
fi

# Check if Neo4j is running in Docker
if command -v docker &> /dev/null; then
  if ! docker ps | grep -q subgraphrag_neo4j; then
    echo -e "${YELLOW}Neo4j container not running. Starting...${NC}"
    docker-compose up -d neo4j
    echo -e "${GREEN}Waiting for Neo4j to start...${NC}"
    sleep 10
  else
    echo -e "${GREEN}Neo4j container is running.${NC}"
  fi
else
  echo -e "${YELLOW}Docker not found. Make sure Neo4j is running manually.${NC}"
fi

# Create directories if they don't exist
mkdir -p logs
mkdir -p $(dirname "$OUTPUT_FILE")
mkdir -p $(dirname "$METRICS_FILE")

# Build benchmark command
BENCHMARK_CMD="python evaluation/benchmark.py --input $INPUT_FILE --output $OUTPUT_FILE --metrics $METRICS_FILE"

if [ -n "$GROUND_TRUTH_FILE" ]; then
  BENCHMARK_CMD="$BENCHMARK_CMD --ground-truth $GROUND_TRUTH_FILE"
fi

if [ "$DETAILED_REPORT" = true ]; then
  BENCHMARK_CMD="$BENCHMARK_CMD --detailed-report"
fi

# Print information
echo -e "${GREEN}Running benchmark with:${NC}"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Metrics: $METRICS_FILE"
if [ -n "$GROUND_TRUTH_FILE" ]; then
  echo "Ground Truth: $GROUND_TRUTH_FILE"
fi
if [ "$DETAILED_REPORT" = true ]; then
  echo "Generating detailed HTML report"
fi

# Run benchmark
echo -e "${GREEN}Starting benchmark...${NC}"
$BENCHMARK_CMD

# Print summary
if [ -f "$METRICS_FILE" ]; then
  echo -e "${GREEN}Benchmark completed. Summary:${NC}"
  
  # Extract key metrics using jq if available, otherwise use grep
  if command -v jq &> /dev/null; then
    echo -e "${BLUE}Success Rate:${NC} $(jq '.successful_queries / .total_questions' $METRICS_FILE)"
    echo -e "${BLUE}Average Latency:${NC} $(jq '.avg_duration' $METRICS_FILE) seconds"
    echo -e "${BLUE}Precision:${NC} $(jq '.precision' $METRICS_FILE)"
    echo -e "${BLUE}Recall:${NC} $(jq '.recall' $METRICS_FILE)"
    echo -e "${BLUE}F1 Score:${NC} $(jq '.f1_score' $METRICS_FILE)"
  else
    echo -e "${YELLOW}Install jq for better metrics display${NC}"
    echo "See full metrics in: $METRICS_FILE"
  fi
else
  echo -e "${RED}Benchmark failed. Check logs for details.${NC}"
fi

echo -e "${BLUE}"
echo "=================================================="
echo "           Benchmark completed                    "
echo "=================================================="
echo -e "${NC}"