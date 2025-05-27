import os
import sys
import logging
import uvicorn
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file before anything else
from dotenv import load_dotenv
load_dotenv()

# Set up logging before importing app modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'app.log'))
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up required directories and environment"""
    # Ensure required directories exist
    for directory in ['data', 'logs', 'cache', 'models']:
        os.makedirs(directory, exist_ok=True)
    
    # Check for config file
    config_path = Path('config/config.json')
    if not config_path.exists():
        logger.warning(f"Configuration file not found at {config_path}")
        logger.info("Creating default configuration...")
        
        # Create config directory if it doesn't exist
        os.makedirs('config', exist_ok=True)
        
        # Write default config
        default_config = '''{
  "MODEL_BACKEND": "openai",
  "FAISS_INDEX_PATH": "data/faiss_index.bin",
  "TOKEN_BUDGET": 4000,
  "MLP_MODEL_PATH": "models/mlp/mlp.pth",
  "CACHE_DIR": "cache/",
  "MAX_DDE_HOPS": 2,
  "LOG_LEVEL": "INFO",
  "API_RATE_LIMIT": 60
}'''
        with open(config_path, 'w') as f:
            f.write(default_config)
        
        logger.info(f"Default configuration created at {config_path}")
    
    # Check for environment variables
    required_env_vars = ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.info("Setting default values for development (NOT SECURE)")
        
        if 'NEO4J_URI' in missing_vars:
            os.environ['NEO4J_URI'] = 'neo4j://localhost:7687'
        if 'NEO4J_USER' in missing_vars:
            os.environ['NEO4J_USER'] = 'neo4j'
        if 'NEO4J_PASSWORD' in missing_vars:
            os.environ['NEO4J_PASSWORD'] = 'password'
    
    if 'API_KEY_SECRET' not in os.environ:
        import uuid
        api_key = str(uuid.uuid4())
        os.environ['API_KEY_SECRET'] = api_key
        logger.warning(f"API_KEY_SECRET not set, using generated key: {api_key}")

def run_app(host: str = '0.0.0.0', port: int = 8000, reload: bool = False):
    """Run the FastAPI application with uvicorn"""
    import src.app.api  # Import here after environment setup
    
    logger.info(f"Starting SubgraphRAG+ server on {host}:{port}")
    uvicorn.run(
        "src.app.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SubgraphRAG+ API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Run application
    run_app(args.host, args.port, args.reload)

# Setup environment for uvicorn direct usage
setup_environment()

# Import and expose the FastAPI app for uvicorn
from src.app.api import app