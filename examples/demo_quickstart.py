import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# RULE:import-rich-logger-correctly - Use centralized rich logger
from src.app.log import logger, log_and_print
from rich.console import Console

# Initialize rich console for pretty CLI output
console = Console()

def run_command(command, cwd=None, env=None):
    """Run a shell command and return success status"""
    logger.info(f"🔄 Running: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Command completed successfully")
            if result.stdout.strip():
                logger.debug(f"Output: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"❌ Command failed with return code {result.returncode}")
            if result.stderr.strip():
                logger.error(f"Error: {result.stderr.strip()}")
            if result.stdout.strip():
                logger.error(f"Output: {result.stdout.strip()}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Command timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Error running command: {e}")
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
    logger.info("🔍 Checking Neo4j connection...")
    
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from src.app.database import neo4j_db
        logger.info("📦 Database module loaded")
        
        # Quick connectivity test with timeout
        logger.info("🔌 Testing Neo4j connectivity...")
        neo4j_db.verify_connectivity()
        logger.info("✅ Neo4j connection successful")
        
        # Quick data check
        result = neo4j_db.run_query("MATCH (n) RETURN count(n) as count LIMIT 1")
        if result:
            count = result[0].get('count', 0)
            logger.info(f"📊 Neo4j has {count} nodes")
        
        return True
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")
        logger.info("💡 Please make sure Neo4j is running with APOC plugin")
        logger.info("💡 You can start Neo4j with: make neo4j-start")
        return False

def migrate_schema():
    """Run Neo4j schema migrations"""
    logger.info("🔄 Applying Neo4j schema migrations...")
    return run_command("python scripts/migrate_schema.py --target-version kg_v2")

def ingest_sample_data():
    """Ingest sample data"""
    logger.info("📥 Checking sample data status...")
    
    # Check if we already have sufficient data
    sys.path.append(str(Path(__file__).parent.parent))
    try:
        from src.app.database import neo4j_db
        result = neo4j_db.run_query("MATCH (n) RETURN count(n) as count")
        if result and result[0].get('count', 0) > 50:
            logger.info(f"✅ Neo4j already has {result[0]['count']} nodes - skipping data ingestion")
            return True
    except Exception as e:
        logger.warning(f"Could not check existing data: {e}")
    
    logger.info("📥 Ingesting sample data...")
    
    # Check if sample data file exists
    sample_file = Path("data/sample_data/sample_triples.csv")
    if not sample_file.exists():
        logger.warning(f"⚠️  Sample data file not found at {sample_file}")
        logger.info("🔄 Using knowledge graph population script instead...")
        return run_command("python scripts/populate_knowledge_graph.py")
    
    # First stage the data
    logger.info("📋 Step 1/3: Staging sample data...")
    if not run_command(f"python scripts/stage_ingest.py --file {sample_file}"):
        return False
    
    # Then process it
    logger.info("⚙️  Step 2/3: Processing staged data...")
    if not run_command("python scripts/ingest_worker.py --process-all"):
        return False
    
    # Train FAISS index with real data
    logger.info("🧠 Step 3/3: Training FAISS index...")
    if not run_command("python scripts/train_faiss_simple.py --verify"):
        logger.warning("⚠️  FAISS training failed, but continuing...")
    
    logger.info("✅ Sample data ingestion completed")
    return True

def ensure_mlp_model():
    """Ensure the pre-trained MLP model is available"""
    logger.info("Checking for pre-trained MLP model...")
    
    # Check if MLP model exists
    model_path = Path("models/mlp/mlp.pth")
    if not model_path.exists():
        print(f"⚠️  MLP model not found at {model_path}")
        print("   You can train it using: notebooks/train_SubGraph_MLP.ipynb")
        print("   Or download from the provided link")
        return False
    
    logger.info(f"Pre-trained MLP model found at {model_path}")
    return True

def start_server(port=8000):
    """Start the API server"""
    logger.info(f"🚀 Starting API server on port {port}...")
    
    # Set up environment for server
    server_env = os.environ.copy()
    # Don't set TESTING=1 for demo - we want to show real functionality
    
    # Start the server in a subprocess
    server_process = subprocess.Popen(
        [sys.executable, "src/main.py", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=server_env
    )
    
    # Wait for server to start with progress indicators
    logger.info("⏳ Waiting for server to start...")
    max_attempts = 10
    for attempt in range(max_attempts):
        time.sleep(1)
        logger.info(f"🔄 Attempt {attempt + 1}/{max_attempts}: Checking server health...")
        
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/healthz", timeout=2)
            if response.status_code == 200:
                logger.info("✅ Server started successfully")
                return server_process
        except requests.exceptions.RequestException:
            # Server not ready yet, continue waiting
            continue
        except Exception as e:
            logger.warning(f"⚠️  Health check error: {e}")
            continue
    
    # If we get here, server didn't start properly
    logger.error("❌ Server failed to start within timeout")
    try:
        server_process.terminate()
        server_process.wait(timeout=5)
    except:
        server_process.kill()
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
    parser.add_argument("--skip-data", action="store_true", help="Skip data ingestion if data already exists")
    args = parser.parse_args()
    
    logger.info("🎯 Starting SubgraphRAG+ Demo Quickstart")
    logger.info("=" * 50)
    
    # Setup environment
    logger.info("📋 Step 1/6: Setting up environment...")
    setup_environment()
    logger.info("✅ Environment setup completed")
    
    # Check Neo4j
    if not args.skip_neo4j:
        logger.info("📋 Step 2/6: Checking Neo4j...")
        if not check_neo4j():
            logger.error("❌ Neo4j check failed - please start Neo4j and try again")
            return 1
        logger.info("✅ Neo4j check completed")
    else:
        logger.info("⏭️  Step 2/6: Skipping Neo4j checks (CI mode)")
    
    # Migrate schema
    if not args.skip_neo4j:
        logger.info("📋 Step 3/6: Migrating database schema...")
        if not migrate_schema():
            logger.error("❌ Schema migration failed")
            return 1
        logger.info("✅ Schema migration completed")
    else:
        logger.info("⏭️  Step 3/6: Skipping schema migration (CI mode)")
    
    # Ensure MLP model
    logger.info("📋 Step 4/6: Checking MLP model...")
    if not ensure_mlp_model():
        logger.warning("⚠️  MLP model setup failed, continuing with fallback")
    else:
        logger.info("✅ MLP model check completed")
    
    # Ingest sample data
    if not args.skip_data:
        logger.info("📋 Step 5/6: Ingesting sample data...")
        if not ingest_sample_data():
            logger.error("❌ Data ingestion failed")
            return 1
        logger.info("✅ Data ingestion completed")
    else:
        logger.info("⏭️  Step 5/6: Skipping data ingestion")
    
    # Start server
    logger.info("📋 Step 6/6: Starting API server...")
    server_process = start_server(args.port)
    if server_process is None:
        logger.error("❌ Failed to start server")
        return 1
    logger.info("✅ Server startup completed")
    
    logger.info("=" * 50)
    logger.info("🎉 Demo quickstart setup completed successfully!")
    logger.info(f"🌐 Server is running at: http://localhost:{args.port}")
    logger.info(f"🔍 Health check: http://localhost:{args.port}/healthz")
    logger.info(f"📚 API docs: http://localhost:{args.port}/docs")
    
    try:
        # Run demo query
        logger.info("🧪 Running demo query...")
        if run_demo_query(args.port):
            logger.info("✅ Demo query completed successfully")
        else:
            logger.warning("⚠️  Demo query failed, but server is still running")
        
        # Keep server running for a while
        logger.info("🔄 Server is running. Press Ctrl+C to stop...")
        logger.info("💡 You can now test the API endpoints manually")
        time.sleep(60)
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    finally:
        # Stop server
        logger.info("🛑 Stopping server...")
        try:
            server_process.terminate()
            server_process.wait(timeout=10)
            logger.info("✅ Server stopped gracefully")
        except:
            server_process.kill()
            logger.info("🔨 Server force-killed")
    
    logger.info("🏁 Demo quickstart completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())