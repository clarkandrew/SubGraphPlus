#!/bin/bash

# SubgraphRAG+ End-to-End Test Runner
# This script sets up the environment and runs the comprehensive end-to-end test

set -e  # Exit on any error

echo "🚀 SubgraphRAG+ End-to-End Test Runner"
echo "======================================"

# Check if we're in the project root
if [ ! -f "scripts/end_to_end_test.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Set default environment variables if not already set
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export API_KEY_SECRET="${API_KEY_SECRET:-default_key_for_dev_only}"
export NEO4J_URI="${NEO4J_URI:-neo4j://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

# Parse command line arguments
MINIMAL_MODE=false
SKIP_MODELS=false
TIMEOUT=300

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            MINIMAL_MODE=true
            export E2E_MINIMAL_MODE=true
            shift
            ;;
        --skip-models)
            SKIP_MODELS=true
            export E2E_SKIP_MODEL_TESTS=true
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --minimal      Run in minimal mode (only requires API server)"
            echo "  --skip-models  Skip model-dependent tests"
            echo "  --timeout SEC  Set timeout for API calls (default: 300)"
            echo "  --help         Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  API_KEY_SECRET    API key for authentication"
            echo "  NEO4J_URI         Neo4j database URI"
            echo "  NEO4J_USER        Neo4j username"
            echo "  NEO4J_PASSWORD    Neo4j password"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo ""
echo "📋 Test Configuration:"
echo "  Minimal mode: $MINIMAL_MODE"
echo "  Skip models: $SKIP_MODELS"
echo "  Timeout: ${TIMEOUT}s"
echo "  API Base URL: http://localhost:8000"
echo ""

# Check Python dependencies
echo "🔍 Checking Python dependencies..."
python3 -c "import requests, rich, neo4j, faiss, numpy" 2>/dev/null || {
    echo "❌ Missing required Python dependencies"
    echo "Please install: pip install requests rich neo4j-driver faiss-cpu numpy"
    exit 1
}

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/test_documents
mkdir -p test_results
mkdir -p logs

# Check if API server is running (unless in minimal mode)
if [ "$MINIMAL_MODE" = false ]; then
    echo "🔍 Checking if API server is running..."
    if ! curl -s http://localhost:8000/healthz > /dev/null; then
        echo "❌ API server is not running on http://localhost:8000"
        echo ""
        echo "Please start the API server first:"
        echo "  python src/main.py --port 8000"
        echo ""
        echo "Or run in minimal mode:"
        echo "  $0 --minimal"
        exit 1
    fi
    echo "✅ API server is running"
fi

# Run the test
echo ""
echo "🧪 Starting end-to-end test..."
echo "This may take several minutes, especially on first run with model loading."
echo ""

# Build the command
CMD="python3 scripts/end_to_end_test.py"
if [ "$MINIMAL_MODE" = true ]; then
    CMD="$CMD --minimal"
fi
if [ "$SKIP_MODELS" = true ]; then
    CMD="$CMD --skip-models"
fi
CMD="$CMD --timeout $TIMEOUT"

# Run the test
$CMD

# Check exit code
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 End-to-end test completed successfully!"
    echo "📊 Check test_results/ directory for detailed results"
else
    echo "❌ End-to-end test failed with exit code $EXIT_CODE"
    echo "📋 Check the output above for error details"
fi

exit $EXIT_CODE 