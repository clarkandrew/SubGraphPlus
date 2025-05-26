import os
import sys
import time
import logging
import subprocess
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, cwd=None, env=None):
    """Run a shell command and return the result"""
    logger.info(f"Running: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True,
            cwd=cwd,
            env=env
        )
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with code {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False

def setup_environment():
    """Set up the environment for the demo"""
    logger.info("Setting up environment...")
    
    # Create required directories
    directories = [
        'data',
        'logs',
        'cache',
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Set up environment variables for Neo4j
    if 'NEO4J_URI' not in os.environ:
        os.environ['NEO4J_URI'] = 'neo4j://localhost:7687'
    if 'NEO4J_USER' not in os.environ:
        os.environ['NEO4J_USER'] = 'neo4j'
    if 'NEO4J_PASSWORD' not in os.environ:
        os.environ['NEO4J_PASSWORD'] = 'password'
    
    # Generate API key if not set
    if 'API_KEY_SECRET' not in os.environ:
        import uuid
        api_key = str(uuid.uuid4())
        os.environ['API_KEY_SECRET'] = api_key
        logger.info(f"Generated API key: {api_key}")

def check_neo4j():
    """Check if Neo4j is running"""
    logger.info("Checking Neo4j connection...")
    
    sys.path.append(str(Path(__file__).parent.parent))
    from src.app.database import neo4j_db
    
    try:
        neo4j_db.verify_connectivity()
        logger.info("Neo4j connection successful")
        return True
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        logger.info("Please make sure Neo4j is running with APOC plugin")
        logger.info("You can start Neo4j with: make neo4j-start")
        return False

def migrate_schema():
    """Run Neo4j schema migrations"""
    logger.info("Applying Neo4j schema migrations...")
    return run_command("python scripts/migrate_schema.py --target-version kg_v2")

def ingest_sample_data():
    """Ingest sample data"""
    logger.info("Ingesting sample data...")
    
    # First stage the data
    if not run_command("python scripts/stage_ingest.py --file data/sample_data/sample_triples.csv"):
        return False
    
    # Then process it
    if not run_command("python scripts/ingest_worker.py"):
        return False
    
    # Merge into FAISS index
    if not run_command("python scripts/merge_faiss.py"):
        return False
    
    return True

def ensure_mlp_model():
    """Ensure the pre-trained MLP model is available"""
    logger.info("Checking for pre-trained MLP model...")
    
    model_path = Path("models/mlp_pretrained.pt")
    if model_path.exists():
        logger.info(f"Pre-trained MLP model found at {model_path}")
        return True
    
    logger.info("Pre-trained MLP model not found, creating placeholder...")
    
    # Create a simple placeholder model for demo purposes
    try:
        import torch
        import torch.nn as nn
        
        # Import the existing SimpleMLP class from the retriever module
        sys.path.append(str(Path(__file__).parent.parent))
        from src.app.retriever import SimpleMLP
        
        # Create model with standard dimensions
        model = SimpleMLP(input_dim=4116, hidden_dim=1024, output_dim=1)
        
        # Save the model state dict in the expected format
        checkpoint = {
            'config': {'input_dim': 4116, 'hidden_dim': 1024, 'output_dim': 1},
            'model_state_dict': model.state_dict()
        }
        
        # Ensure the models directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, model_path)
        logger.info(f"Created placeholder MLP model at {model_path}")
        return True
    except ImportError as e:
        logger.warning(f"PyTorch or SimpleMLP not available: {e}, skipping MLP model creation")
        return False
    except Exception as e:
        logger.error(f"Error creating MLP model: {e}")
        return False

def start_server(port=8000):
    """Start the API server"""
    logger.info(f"Starting API server on port {port}...")
    
    # Start the server in a subprocess
    server_process = subprocess.Popen(
        [sys.executable, "src/main.py", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server to start
    logger.info("Waiting for server to start...")
    time.sleep(5)
    
    # Check if server is running
    try:
        import requests
        response = requests.get(f"http://localhost:{port}/healthz")
        if response.status_code == 200:
            logger.info("Server started successfully")
            return server_process
        else:
            logger.error(f"Server health check failed: {response.status_code}")
            server_process.terminate()
            return None
    except Exception as e:
        logger.error(f"Error checking server health: {e}")
        server_process.terminate()
        return None

def run_demo_query(port=8000):
    """Run a demo query against the API"""
    logger.info("Running demo query...")
    
    import requests
    import json
    
    api_key = os.environ.get('API_KEY_SECRET', 'default_key_for_dev_only')
    
    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': api_key
    }
    
    data = {
        'question': 'Who founded Tesla?',
        'visualize_graph': True
    }
    
    try:
        response = requests.post(
            f'http://localhost:{port}/query',
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            logger.info("Query successful!")
            logger.info(f"Response: {response.text[:100]}...")
            return True
        else:
            logger.error(f"Query failed with status {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error running query: {e}")
        return False

def main():
    """Main function to run the demo quickstart"""
    parser = argparse.ArgumentParser(description="SubgraphRAG+ Demo Quickstart")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j checks (for CI environments)")
    args = parser.parse_args()
    
    logger.info("Starting SubgraphRAG+ Demo Quickstart")
    
    # Setup environment
    setup_environment()
    
    # Check Neo4j
    if not args.skip_neo4j and not check_neo4j():
        return 1
    
    # Migrate schema
    if not args.skip_neo4j and not migrate_schema():
        logger.error("Schema migration failed")
        return 1
    
    # Ensure MLP model
    if not ensure_mlp_model():
        logger.warning("MLP model setup failed, continuing with fallback")
    
    # Ingest sample data
    if not ingest_sample_data():
        logger.error("Data ingestion failed")
        return 1
    
    # Start server
    server_process = start_server(args.port)
    if server_process is None:
        logger.error("Failed to start server")
        return 1
    
    try:
        # Run demo query
        run_demo_query(args.port)
        
        # Keep server running for a while
        logger.info("Server is running. Press Ctrl+C to stop...")
        time.sleep(60)
    finally:
        # Stop server
        logger.info("Stopping server...")
        server_process.terminate()
        server_process.wait()
    
    logger.info("Demo quickstart completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())