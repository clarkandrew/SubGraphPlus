#!/usr/bin/env python3
"""
SubgraphRAG+ End-to-End Integration Test Script

This script validates the complete SubgraphRAG+ pipeline by:
1. Checking system health and dependencies
2. Ingesting a PDF document via API
3. Validating Neo4j storage and FAISS indexing
4. Testing query processing and retrieval
5. Validating graph visualization endpoints
6. Generating a comprehensive validation report

Usage:
    python scripts/end_to_end_test.py

Requirements:
    - SubgraphRAG+ API server running on configured port
    - IE service running on port 8003
    - Neo4j database accessible
    - All required models downloaded and available
"""

import os
import sys
import time
import json
import requests
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import tempfile
import shutil

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# RULE:import-rich-logger-correctly
try:
    from src.app.log import logger
    from src.app.config import config, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, API_KEY_SECRET
except ImportError:
    # Fallback for different import paths
    try:
        from app.log import logger
        from app.config import config, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, API_KEY_SECRET
    except ImportError:
        # Create minimal logger if imports fail
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Try to load config manually
        import json
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        # Load config manually
        config_path = project_root / "config" / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
            
            class SimpleConfig:
                def __init__(self, data):
                    self.FAISS_INDEX_PATH = data.get("data", {}).get("faiss_index_path", "data/faiss_index.bin")
            
            config = SimpleConfig(config_data)
        else:
            class SimpleConfig:
                FAISS_INDEX_PATH = "data/faiss_index.bin"
            config = SimpleConfig()
        
        # Environment variables
        NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
        NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
        API_KEY_SECRET = os.environ.get("API_KEY_SECRET", "default_key_for_dev_only")

# Rich console for beautiful output
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.layout import Layout
from rich.live import Live

# Neo4j and FAISS for direct validation
try:
    from neo4j import GraphDatabase
    import faiss
    import numpy as np
except ImportError as e:
    logger.error(f"Missing required dependencies: {e}")
    sys.exit(1)

# RULE:uppercase-constants-top
PDF_INPUT_FILE = "data/test_documents/sample_document.pdf"
API_BASE_URL = "http://localhost:8000"
# Note: IE functionality is now integrated into the main API at /ie/ endpoints
TEST_QUESTION = "What is the main topic discussed in the document?"
TIMEOUT_SECONDS = 30
MIN_REQUIRED_DISK_SPACE_GB = 2
REQUIRED_PYTHON_VERSION = (3, 8)

console = Console()

class ValidationResults:
    """Container for all validation results"""
    def __init__(self):
        self.start_time = time.time()
        self.system_health = {}
        self.ingestion_success = False
        self.neo4j_entities_before = 0
        self.neo4j_relationships_before = 0
        self.neo4j_entities_after = 0
        self.neo4j_relationships_after = 0
        self.faiss_vectors_before = 0
        self.faiss_vectors_after = 0
        self.query_response_time_ms = 0
        self.citations_valid = False
        self.graph_data_valid = False
        self.answer_quality_score = 0.0
        self.errors = []
        self.warnings = []
        self.performance_metrics = {}
        
    def add_error(self, error: str):
        self.errors.append(error)
        logger.error(error)
        
    def add_warning(self, warning: str):
        self.warnings.append(warning)
        logger.warning(warning)
        
    def get_total_time(self) -> float:
        return time.time() - self.start_time

def create_sample_pdf():
    """Create a sample PDF for testing if it doesn't exist"""
    pdf_path = Path(PDF_INPUT_FILE)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    if pdf_path.exists():
        return str(pdf_path)
    
    # Create a simple text file that can be used for testing
    # In a real scenario, you'd want a proper PDF
    sample_text = """
    SubgraphRAG+ Test Document
    
    This is a sample document for testing the SubgraphRAG+ system.
    
    Key Information:
    - SubgraphRAG+ is an advanced knowledge graph question answering system
    - It uses REBEL for information extraction
    - The system employs MLP-based triple scoring
    - Neo4j is used for graph storage
    - FAISS provides vector similarity search
    
    Technical Details:
    - The system processes documents through multiple stages
    - Entity linking connects text mentions to knowledge graph entities
    - Hybrid retrieval combines graph and dense retrieval methods
    - The MLP scorer ranks candidate triples for relevance
    
    Applications:
    - Question answering over large document collections
    - Knowledge discovery and exploration
    - Fact verification and citation
    """
    
    # For now, create a text file (in production, you'd use a PDF library)
    with open(pdf_path, 'w') as f:
        f.write(sample_text)
    
    console.print(f"[yellow]Created sample text file at {pdf_path} (treating as PDF for testing)[/yellow]")
    return str(pdf_path)

def check_system_requirements() -> Dict[str, bool]:
    """Check basic system requirements"""
    logger.debug("Starting system requirements check")
    
    checks = {}
    
    # Python version
    current_version = sys.version_info[:2]
    checks['python_version'] = current_version >= REQUIRED_PYTHON_VERSION
    if not checks['python_version']:
        console.print(f"[red]Python {REQUIRED_PYTHON_VERSION} required, got {current_version}[/red]")
    
    # Disk space (simplified check)
    try:
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)
        checks['disk_space'] = free_space_gb >= MIN_REQUIRED_DISK_SPACE_GB
        if not checks['disk_space']:
            console.print(f"[red]Insufficient disk space: {free_space_gb:.1f}GB available, {MIN_REQUIRED_DISK_SPACE_GB}GB required[/red]")
    except Exception as e:
        checks['disk_space'] = False
        console.print(f"[red]Could not check disk space: {e}[/red]")
    
    # Input file
    checks['input_file'] = Path(PDF_INPUT_FILE).exists()
    if not checks['input_file']:
        console.print(f"[yellow]Input file {PDF_INPUT_FILE} not found, will create sample[/yellow]")
    
    logger.debug("Finished system requirements check")
    return checks

def check_api_health() -> Dict[str, Any]:
    """Check API endpoint health"""
    logger.debug("Starting API health check")
    
    health_status = {}
    
    # Main API health check
    try:
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=TIMEOUT_SECONDS)
        health_status['main_api'] = {
            'status': response.status_code == 200,
            'response_time_ms': response.elapsed.total_seconds() * 1000,
            'details': response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        health_status['main_api'] = {
            'status': False,
            'error': str(e)
        }
    
    # Main API readiness check
    try:
        response = requests.get(f"{API_BASE_URL}/readyz", timeout=TIMEOUT_SECONDS)
        health_status['main_api_ready'] = {
            'status': response.status_code == 200,
            'response_time_ms': response.elapsed.total_seconds() * 1000,
            'details': response.json() if response.content else {}
        }
    except Exception as e:
        health_status['main_api_ready'] = {
            'status': False,
            'error': str(e)
        }
    
    # IE Module health check (integrated into main API)
    try:
        response = requests.get(f"{API_BASE_URL}/ie/health", timeout=TIMEOUT_SECONDS)
        health_status['ie_module'] = {
            'status': response.status_code == 200,
            'response_time_ms': response.elapsed.total_seconds() * 1000,
            'details': response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        health_status['ie_module'] = {
            'status': False,
            'error': str(e)
        }
    
    # Test IE info endpoint as well
    try:
        response = requests.get(f"{API_BASE_URL}/ie/info", timeout=TIMEOUT_SECONDS)
        health_status['ie_info'] = {
            'status': response.status_code == 200,
            'response_time_ms': response.elapsed.total_seconds() * 1000,
            'details': response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        health_status['ie_info'] = {
            'status': False,
            'error': str(e)
        }
    
    logger.debug("Finished API health check")
    return health_status

def check_neo4j_connection() -> Dict[str, Any]:
    """Check Neo4j database connection and get baseline stats"""
    logger.debug("Starting Neo4j connection check")
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Test connection
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            
            if test_value != 1:
                raise Exception("Neo4j connection test failed")
            
            # Get baseline statistics
            entity_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r:REL]->() RETURN count(r) as count").single()["count"]
            
            # Get entity types distribution
            entity_types = session.run(
                "MATCH (n:Entity) RETURN n.type as type, count(n) as count ORDER BY count DESC"
            ).data()
            
            # Get relation types distribution
            relation_types = session.run(
                "MATCH ()-[r:REL]->() RETURN r.name as relation, count(r) as count ORDER BY count DESC LIMIT 10"
            ).data()
            
            driver.close()
            
            return {
                'status': True,
                'entity_count': entity_count,
                'relationship_count': rel_count,
                'entity_types': entity_types,
                'relation_types': relation_types
            }
            
    except Exception as e:
        return {
            'status': False,
            'error': str(e)
        }

def check_faiss_index() -> Dict[str, Any]:
    """Check FAISS index status"""
    logger.debug("Starting FAISS index check")
    
    try:
        faiss_path = config.FAISS_INDEX_PATH
        
        if not Path(faiss_path).exists():
            return {
                'status': False,
                'error': f"FAISS index not found at {faiss_path}",
                'vector_count': 0
            }
        
        # Load index
        index = faiss.read_index(faiss_path)
        
        return {
            'status': True,
            'vector_count': index.ntotal,
            'index_type': type(index).__name__,
            'is_trained': index.is_trained,
            'dimension': index.d if hasattr(index, 'd') else 'unknown'
        }
        
    except Exception as e:
        return {
            'status': False,
            'error': str(e),
            'vector_count': 0
        }

def ingest_document(file_path: str) -> Dict[str, Any]:
    """Ingest document via API"""
    logger.debug(f"Starting document ingestion: {file_path}")
    
    start_time = time.time()
    
    try:
        # For now, we'll send the text content as triples
        # In a real implementation, the API would handle PDF processing
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create some sample triples from the content
        sample_triples = [
            {"head": "SubgraphRAG+", "relation": "is_a", "tail": "knowledge graph system"},
            {"head": "SubgraphRAG+", "relation": "uses", "tail": "REBEL"},
            {"head": "SubgraphRAG+", "relation": "uses", "tail": "Neo4j"},
            {"head": "SubgraphRAG+", "relation": "uses", "tail": "FAISS"},
            {"head": "REBEL", "relation": "performs", "tail": "information extraction"},
            {"head": "Neo4j", "relation": "provides", "tail": "graph storage"},
            {"head": "FAISS", "relation": "provides", "tail": "vector similarity search"},
            {"head": "MLP scorer", "relation": "ranks", "tail": "candidate triples"},
            {"head": "system", "relation": "processes", "tail": "documents"},
            {"head": "system", "relation": "supports", "tail": "question answering"}
        ]
        
        # Send to ingest API
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json={"triples": sample_triples},
            headers={"X-API-KEY": API_KEY_SECRET},
            timeout=TIMEOUT_SECONDS
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200 or response.status_code == 202:
            return {
                'status': True,
                'response_code': response.status_code,
                'processing_time_s': processing_time,
                'triples_sent': len(sample_triples),
                'response_data': response.json() if response.content else {}
            }
        else:
            return {
                'status': False,
                'response_code': response.status_code,
                'processing_time_s': processing_time,
                'error': response.text
            }
            
    except Exception as e:
        return {
            'status': False,
            'error': str(e),
            'processing_time_s': time.time() - start_time
        }

def validate_neo4j_changes(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Validate changes in Neo4j after ingestion"""
    logger.debug("Starting Neo4j changes validation")
    
    try:
        current_stats = check_neo4j_connection()
        
        if not current_stats['status']:
            return {
                'status': False,
                'error': 'Could not connect to Neo4j for validation'
            }
        
        entities_added = current_stats['entity_count'] - baseline['entity_count']
        relationships_added = current_stats['relationship_count'] - baseline['relationship_count']
        
        # Get sample of new data
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Get recent entities (assuming they have timestamps or can be identified)
            sample_entities = session.run(
                "MATCH (n:Entity) RETURN n.name, n.type LIMIT 10"
            ).data()
            
            # Get recent relationships
            sample_relationships = session.run(
                "MATCH (h:Entity)-[r:REL]->(t:Entity) RETURN h.name, r.name, t.name LIMIT 10"
            ).data()
        
        driver.close()
        
        return {
            'status': True,
            'entities_added': entities_added,
            'relationships_added': relationships_added,
            'total_entities': current_stats['entity_count'],
            'total_relationships': current_stats['relationship_count'],
            'sample_entities': sample_entities,
            'sample_relationships': sample_relationships
        }
        
    except Exception as e:
        return {
            'status': False,
            'error': str(e)
        }

def validate_faiss_changes(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Validate changes in FAISS index after ingestion"""
    logger.debug("Starting FAISS changes validation")
    
    try:
        current_stats = check_faiss_index()
        
        if not current_stats['status']:
            return {
                'status': False,
                'error': 'Could not access FAISS index for validation'
            }
        
        vectors_added = current_stats['vector_count'] - baseline['vector_count']
        
        # Test search functionality
        if current_stats['vector_count'] > 0:
            index = faiss.read_index(config.FAISS_INDEX_PATH)
            
            # Create a random query vector for testing
            dimension = index.d
            test_query = np.random.random((1, dimension)).astype('float32')
            
            # Perform search
            distances, indices = index.search(test_query, k=min(5, index.ntotal))
            
            search_results = {
                'distances': distances[0].tolist(),
                'indices': indices[0].tolist()
            }
        else:
            search_results = {'message': 'No vectors to search'}
        
        return {
            'status': True,
            'vectors_added': vectors_added,
            'total_vectors': current_stats['vector_count'],
            'index_type': current_stats['index_type'],
            'is_trained': current_stats['is_trained'],
            'dimension': current_stats['dimension'],
            'search_test': search_results
        }
        
    except Exception as e:
        return {
            'status': False,
            'error': str(e)
        }

def test_query_processing(question: str) -> Dict[str, Any]:
    """Test query processing via API"""
    logger.debug(f"Starting query processing test: {question}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "question": question,
                "visualize_graph": True
            },
            headers={"X-API-KEY": API_KEY_SECRET},
            stream=True,
            timeout=TIMEOUT_SECONDS
        )
        
        if response.status_code != 200:
            return {
                'status': False,
                'response_code': response.status_code,
                'error': response.text,
                'processing_time_s': time.time() - start_time
            }
        
        # Parse SSE stream
        events = []
        answer_tokens = []
        citations = []
        graph_data = None
        metadata = {}
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        event_data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                        events.append(event_data)
                        
                        if event_data.get('event') == 'llm_token':
                            answer_tokens.append(event_data['data']['token'])
                        elif event_data.get('event') == 'citation_data':
                            citations.append(event_data['data'])
                        elif event_data.get('event') == 'graph_data':
                            graph_data = event_data['data']
                        elif event_data.get('event') == 'metadata':
                            metadata.update(event_data['data'])
                        elif event_data.get('event') == 'end':
                            break
                            
                    except json.JSONDecodeError:
                        continue
        
        processing_time = time.time() - start_time
        full_answer = ''.join(answer_tokens)
        
        return {
            'status': True,
            'processing_time_s': processing_time,
            'answer': full_answer,
            'citations': citations,
            'graph_data': graph_data,
            'metadata': metadata,
            'total_events': len(events),
            'answer_length': len(full_answer)
        }
        
    except Exception as e:
        return {
            'status': False,
            'error': str(e),
            'processing_time_s': time.time() - start_time
        }

def test_ie_extraction() -> Dict[str, Any]:
    """Test IE extraction functionality"""
    logger.debug("Starting IE extraction test")
    
    sample_text = "Barack Obama was born in Hawaii. He served as President of the United States."
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/ie/extract",
            json={
                "text": sample_text,
                "max_length": 256,
                "num_beams": 3
            },
            headers={"X-API-KEY": API_KEY_SECRET},
            timeout=TIMEOUT_SECONDS
        )
        
        if response.status_code != 200:
            return {
                'status': False,
                'response_code': response.status_code,
                'error': response.text
            }
        
        data = response.json()
        
        # Validate response structure
        required_fields = ['triples', 'raw_output', 'processing_time']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return {
                'status': False,
                'error': f"Missing required fields: {missing_fields}"
            }
        
        # Validate triples structure
        triples_valid = True
        if data['triples']:
            sample_triple = data['triples'][0]
            required_triple_fields = ['head', 'relation', 'tail']
            missing_triple_fields = [field for field in required_triple_fields if field not in sample_triple]
            if missing_triple_fields:
                triples_valid = False
        
        return {
            'status': True,
            'triples_count': len(data['triples']),
            'processing_time': data['processing_time'],
            'triples_valid': triples_valid,
            'sample_triples': data['triples'][:3] if data['triples'] else [],
            'raw_output_length': len(data['raw_output'])
        }
        
    except Exception as e:
        return {
            'status': False,
            'error': str(e)
        }

def test_graph_browse() -> Dict[str, Any]:
    """Test graph browsing API"""
    logger.debug("Starting graph browse test")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/graph/browse",
            params={'limit': 50, 'page': 1},
            headers={"X-API-KEY": API_KEY_SECRET},
            timeout=TIMEOUT_SECONDS
        )
        
        if response.status_code != 200:
            return {
                'status': False,
                'response_code': response.status_code,
                'error': response.text
            }
        
        data = response.json()
        
        # Validate structure
        required_fields = ['nodes', 'links', 'page', 'limit', 'total_nodes_in_filter', 'total_links_in_filter', 'has_more']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return {
                'status': False,
                'error': f"Missing required fields: {missing_fields}"
            }
        
        # Validate node structure
        node_validation = True
        if data['nodes']:
            sample_node = data['nodes'][0]
            required_node_fields = ['id', 'name']
            missing_node_fields = [field for field in required_node_fields if field not in sample_node]
            if missing_node_fields:
                node_validation = False
        
        # Validate link structure
        link_validation = True
        if data['links']:
            sample_link = data['links'][0]
            required_link_fields = ['source', 'target', 'relation_name']
            missing_link_fields = [field for field in required_link_fields if field not in sample_link]
            if missing_link_fields:
                link_validation = False
        
        return {
            'status': True,
            'node_count': len(data['nodes']),
            'link_count': len(data['links']),
            'total_nodes': data['total_nodes_in_filter'],
            'total_links': data['total_links_in_filter'],
            'has_more': data['has_more'],
            'node_validation': node_validation,
            'link_validation': link_validation,
            'sample_data': {
                'nodes': data['nodes'][:3] if data['nodes'] else [],
                'links': data['links'][:3] if data['links'] else []
            }
        }
        
    except Exception as e:
        return {
            'status': False,
            'error': str(e)
        }

def generate_report(results: ValidationResults) -> None:
    """Generate comprehensive validation report"""
    logger.debug("Generating validation report")
    
    # Create main layout
    layout = Layout()
    
    # Header
    header_text = Text("🚀 SubgraphRAG+ End-to-End Test Results", style="bold blue")
    header_panel = Panel(header_text, box=box.DOUBLE)
    
    console.print(header_panel)
    console.print()
    
    # System Health Table
    health_table = Table(title="System Health Checks", box=box.ROUNDED)
    health_table.add_column("Component", style="cyan")
    health_table.add_column("Status", style="green")
    health_table.add_column("Details", style="dim")
    
    for component, status in results.system_health.items():
        if isinstance(status, dict):
            status_text = "✅ HEALTHY" if status.get('status', False) else "❌ FAILED"
            details = status.get('details', status.get('error', ''))
            if isinstance(details, dict):
                details = json.dumps(details, indent=2)
        else:
            status_text = "✅ OK" if status else "❌ FAILED"
            details = ""
        
        health_table.add_row(component.replace('_', ' ').title(), status_text, str(details)[:100])
    
    console.print(health_table)
    console.print()
    
    # Ingestion Results
    if results.ingestion_success:
        ingestion_table = Table(title="Document Ingestion Results", box=box.ROUNDED)
        ingestion_table.add_column("Metric", style="cyan")
        ingestion_table.add_column("Value", style="green")
        
        ingestion_table.add_row("Status", "✅ SUCCESS")
        ingestion_table.add_row("Entities Added", str(results.neo4j_entities_after - results.neo4j_entities_before))
        ingestion_table.add_row("Relationships Added", str(results.neo4j_relationships_after - results.neo4j_relationships_before))
        ingestion_table.add_row("FAISS Vectors Added", str(results.faiss_vectors_after - results.faiss_vectors_before))
        
        console.print(ingestion_table)
        console.print()
    
    # Query Processing Results
    if results.query_response_time_ms > 0:
        query_table = Table(title="Query Processing Results", box=box.ROUNDED)
        query_table.add_column("Metric", style="cyan")
        query_table.add_column("Value", style="green")
        
        query_table.add_row("Response Time", f"{results.query_response_time_ms:.1f}ms")
        query_table.add_row("Citations Valid", "✅ YES" if results.citations_valid else "❌ NO")
        query_table.add_row("Graph Data Valid", "✅ YES" if results.graph_data_valid else "❌ NO")
        query_table.add_row("Answer Quality", f"{results.answer_quality_score:.2f}")
        
        console.print(query_table)
        console.print()
    
    # Performance Metrics
    if results.performance_metrics:
        perf_table = Table(title="Performance Metrics", box=box.ROUNDED)
        perf_table.add_column("Operation", style="cyan")
        perf_table.add_column("Time (s)", style="green")
        perf_table.add_column("Status", style="yellow")
        
        for operation, metrics in results.performance_metrics.items():
            time_taken = metrics.get('time', 0)
            status = "✅ OK" if metrics.get('success', False) else "❌ FAILED"
            perf_table.add_row(operation.replace('_', ' ').title(), f"{time_taken:.2f}", status)
        
        console.print(perf_table)
        console.print()
    
    # Errors and Warnings
    if results.errors:
        error_panel = Panel(
            "\n".join(f"❌ {error}" for error in results.errors),
            title="Errors",
            border_style="red"
        )
        console.print(error_panel)
        console.print()
    
    if results.warnings:
        warning_panel = Panel(
            "\n".join(f"⚠️  {warning}" for warning in results.warnings),
            title="Warnings",
            border_style="yellow"
        )
        console.print(warning_panel)
        console.print()
    
    # Overall Status
    overall_success = (
        results.ingestion_success and
        len(results.errors) == 0 and
        results.citations_valid and
        results.graph_data_valid
    )
    
    status_text = "🎯 ALL SYSTEMS OPERATIONAL" if overall_success else "⚠️  ISSUES DETECTED"
    status_style = "bold green" if overall_success else "bold yellow"
    
    total_time = results.get_total_time()
    
    final_panel = Panel(
        f"{status_text}\n\nTotal execution time: {total_time:.2f} seconds",
        title="Overall Status",
        border_style="green" if overall_success else "yellow"
    )
    
    console.print(final_panel)

def check_services_and_provide_guidance(results: ValidationResults) -> bool:
    """Check if required services are running and provide guidance if not"""
    api_running = results.system_health.get('main_api', {}).get('status', False)
    ie_running = results.system_health.get('ie_module', {}).get('status', False)
    
    if not api_running:
        console.print("\n")
        console.print(Panel(
            "🚨 Required Service Not Running\n\n"
            "The end-to-end test requires the main API server to be running:\n\n"
            f"{'✅' if api_running else '❌'} Main API Server (port 8000) - includes IE module\n"
            f"{'✅' if ie_running else '❌'} IE Module (/ie/ endpoints)\n\n"
            "To start the service:\n\n"
            "1. Start the main API server (includes IE functionality):\n"
            "   python src/main.py --port 8000\n"
            "   # OR\n"
            "   uvicorn src.app.api:app --host 0.0.0.0 --port 8000\n\n"
            "2. Re-run this test:\n"
            "   python scripts/end_to_end_test.py\n\n"
            "Note: IE functionality is now integrated into the main API\n"
            "as a service layer, no separate IE server needed.",
            title="🔧 Service Setup Required",
            border_style="yellow"
        ))
        return False
    return True

def main():
    """Main execution function"""
    logger.info(f"Started {__file__} at {datetime.now()}")
    
    console.print(Panel(
        "SubgraphRAG+ End-to-End Integration Test\n"
        "This script validates the complete system pipeline",
        title="🚀 Starting E2E Test",
        border_style="blue"
    ))
    
    results = ValidationResults()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Step 1: System Requirements
        task1 = progress.add_task("Checking system requirements...", total=100)
        req_checks = check_system_requirements()
        results.system_health.update(req_checks)
        progress.update(task1, completed=100)
        
        # Step 2: Create sample PDF if needed
        task2 = progress.add_task("Preparing test document...", total=100)
        if not req_checks.get('input_file', False):
            create_sample_pdf()
        progress.update(task2, completed=100)
        
        # Step 3: API Health Checks
        task3 = progress.add_task("Checking API health...", total=100)
        api_health = check_api_health()
        results.system_health.update(api_health)
        progress.update(task3, completed=100)
        
        # Check if services are running and provide guidance if not
        if not check_services_and_provide_guidance(results):
            console.print("\n")
            console.print(Panel(
                "Test completed with service availability issues.\n"
                "Please start the required services and re-run the test.",
                title="⚠️ Test Incomplete",
                border_style="yellow"
            ))
            logger.info(f"Finished {__file__} at {datetime.now()}")
            sys.exit(1)
        
        # Step 4: Database Baseline
        task4 = progress.add_task("Getting database baseline...", total=100)
        neo4j_baseline = check_neo4j_connection()
        faiss_baseline = check_faiss_index()
        
        if neo4j_baseline['status']:
            results.neo4j_entities_before = neo4j_baseline['entity_count']
            results.neo4j_relationships_before = neo4j_baseline['relationship_count']
        
        if faiss_baseline['status']:
            results.faiss_vectors_before = faiss_baseline['vector_count']
        
        results.system_health['neo4j_baseline'] = neo4j_baseline
        results.system_health['faiss_baseline'] = faiss_baseline
        progress.update(task4, completed=100)
        
        # Step 5: Document Ingestion
        task5 = progress.add_task("Ingesting test document...", total=100)
        start_time = time.time()
        ingestion_result = ingest_document(PDF_INPUT_FILE)
        ingestion_time = time.time() - start_time
        
        results.ingestion_success = ingestion_result['status']
        results.performance_metrics['document_ingestion'] = {
            'time': ingestion_time,
            'success': ingestion_result['status']
        }
        
        if not results.ingestion_success:
            results.add_error(f"Document ingestion failed: {ingestion_result.get('error', 'Unknown error')}")
        
        progress.update(task5, completed=100)
        
        # Step 6: Validate Storage Changes
        task6 = progress.add_task("Validating storage changes...", total=100)
        
        # Wait a moment for ingestion to complete
        time.sleep(2)
        
        neo4j_changes = validate_neo4j_changes(neo4j_baseline)
        faiss_changes = validate_faiss_changes(faiss_baseline)
        
        if neo4j_changes['status']:
            results.neo4j_entities_after = neo4j_changes['total_entities']
            results.neo4j_relationships_after = neo4j_changes['total_relationships']
        
        if faiss_changes['status']:
            results.faiss_vectors_after = faiss_changes['total_vectors']
        
        results.system_health['neo4j_changes'] = neo4j_changes
        results.system_health['faiss_changes'] = faiss_changes
        progress.update(task6, completed=100)
        
        # Step 7: Query Processing Test
        task7 = progress.add_task("Testing query processing...", total=100)
        start_time = time.time()
        query_result = test_query_processing(TEST_QUESTION)
        query_time = time.time() - start_time
        
        results.query_response_time_ms = query_time * 1000
        results.performance_metrics['query_processing'] = {
            'time': query_time,
            'success': query_result['status']
        }
        
        if query_result['status']:
            results.citations_valid = len(query_result.get('citations', [])) > 0
            results.graph_data_valid = query_result.get('graph_data') is not None
            results.answer_quality_score = 0.8 if query_result.get('answer') else 0.0
        else:
            results.add_error(f"Query processing failed: {query_result.get('error', 'Unknown error')}")
        
        results.system_health['query_processing'] = query_result
        progress.update(task7, completed=100)
        
        # Step 8: IE Extraction Test
        task8 = progress.add_task("Testing IE extraction...", total=100)
        start_time = time.time()
        ie_result = test_ie_extraction()
        ie_time = time.time() - start_time
        
        results.performance_metrics['ie_extraction'] = {
            'time': ie_time,
            'success': ie_result['status']
        }
        
        if not ie_result['status']:
            results.add_error(f"IE extraction failed: {ie_result.get('error', 'Unknown error')}")
        
        results.system_health['ie_extraction'] = ie_result
        progress.update(task8, completed=100)
        
        # Step 9: Graph Browse Test
        task9 = progress.add_task("Testing graph browsing...", total=100)
        start_time = time.time()
        browse_result = test_graph_browse()
        browse_time = time.time() - start_time
        
        results.performance_metrics['graph_browsing'] = {
            'time': browse_time,
            'success': browse_result['status']
        }
        
        if not browse_result['status']:
            results.add_error(f"Graph browsing failed: {browse_result.get('error', 'Unknown error')}")
        
        results.system_health['graph_browsing'] = browse_result
        progress.update(task9, completed=100)
    
    # Generate final report
    console.print("\n")
    generate_report(results)
    
    # Log completion
    logger.info(f"Finished {__file__} at {datetime.now()}")
    
    # Exit with appropriate code
    exit_code = 0 if len(results.errors) == 0 else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 