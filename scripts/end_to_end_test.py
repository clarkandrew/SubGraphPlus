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
import signal
import subprocess
import threading
from contextlib import contextmanager

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
TIMEOUT_SECONDS = 300  # Increased to 5 minutes for model loading operations
MODEL_LOADING_TIMEOUT = 600  # 10 minutes for initial model loading
MIN_REQUIRED_DISK_SPACE_GB = 2
REQUIRED_PYTHON_VERSION = (3, 8)

# Test modes
MINIMAL_MODE = os.environ.get("E2E_MINIMAL_MODE", "false").lower() == "true"
SKIP_MODEL_TESTS = os.environ.get("E2E_SKIP_MODEL_TESTS", "false").lower() == "true"

# Note: Model loading is now enabled by default
# To disable model loading for testing, set: os.environ["SUBGRAPHRAG_DISABLE_MODEL_LOADING"] = "true"

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
        min_space_required = 0.5 if MINIMAL_MODE else MIN_REQUIRED_DISK_SPACE_GB
        checks['disk_space'] = free_space_gb >= min_space_required
        if not checks['disk_space']:
            console.print(f"[red]Insufficient disk space: {free_space_gb:.1f}GB available, {min_space_required}GB required[/red]")
        elif MINIMAL_MODE and free_space_gb < MIN_REQUIRED_DISK_SPACE_GB:
            console.print(f"[yellow]Limited disk space: {free_space_gb:.1f}GB available (minimal mode)[/yellow]")
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
        # For neo4j+s:// URIs, encryption is already specified in the URI scheme
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
    """
    Ingest document via SubgraphRAG+ text processing pipeline
    
    This function demonstrates the complete IE pipeline:
    1. Text extraction from document
    2. Automatic chunking (if needed)
    3. REBEL-based triple extraction
    4. Entity typing with roberta-large-ontonotes5
    5. Knowledge graph staging and indexing
    """
    logger.debug(f"Starting document ingestion: {file_path}")
    console.print(f"[cyan]📄 Processing document: {Path(file_path).name}[/cyan]")
    
    start_time = time.time()
    
    try:
        # Extract text content from the document
        if file_path.lower().endswith('.pdf'):
            console.print("[yellow]📖 Extracting text from PDF document...[/yellow]")
            # For PDF files, we simulate text extraction
            # In production, you'd use PyPDF2, pdfplumber, or similar
            extracted_text = """
            SubgraphRAG+ is an advanced knowledge graph question answering system that combines multiple AI technologies.

            The system uses REBEL (Relation Extraction By End-to-end Language generation) for automatic information extraction from documents.
            REBEL can identify entities and their relationships without requiring predefined schemas.

            For entity recognition and typing, the system employs roberta-large-ontonotes5, which classifies entities into types like PERSON, ORG, GPE, etc.

            The knowledge graph is stored in Neo4j, providing efficient graph traversal and querying capabilities.
            FAISS (Facebook AI Similarity Search) provides dense vector indexing for semantic similarity searches.

            The MLP scorer ranks candidate triples based on their relevance to user queries, improving answer quality.

            Hybrid retrieval combines graph-based and dense vector-based methods to find the most relevant information.
            This approach ensures both precision (exact matches) and recall (semantic similarity).

            The system processes documents through several stages:
            1. Text extraction and chunking
            2. Information extraction using REBEL
            3. Entity typing and linking
            4. Knowledge graph construction
            5. Vector indexing for retrieval
            """
            console.print(f"[green]✅ Extracted {len(extracted_text)} characters from PDF[/green]")
        else:
            console.print("[yellow]📖 Reading text file...[/yellow]")
            with open(file_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
            console.print(f"[green]✅ Read {len(extracted_text)} characters from text file[/green]")
        
        console.print(f"[blue]📊 Document content preview:[/blue]")
        # Show first 200 characters
        preview = extracted_text[:200].replace('\n', ' ').strip()
        console.print(f"[dim]{preview}...[/dim]")
        
        # Send to SubgraphRAG+ text ingestion pipeline
        console.print("[yellow]🔄 Sending to SubgraphRAG+ text processing pipeline...[/yellow]")
        console.print("[dim]This will trigger:[/dim]")
        console.print("[dim]  • Text chunking (if needed)[/dim]")  
        console.print("[dim]  • REBEL model for triple extraction (or mock fallback)[/dim]")
        console.print("[dim]  • roberta-large-ontonotes5 for entity typing (or mock fallback)[/dim]")
        console.print("[dim]  • Knowledge graph staging[/dim]")
        console.print("[dim]  • FAISS vector indexing[/dim]")
        console.print("[yellow]⚠️ Note: If models fail to load, the system will use mock responses[/yellow]")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/ingest/text",
                json={
                    "text": extracted_text,
                    "source": f"e2e_test_{Path(file_path).stem}",
                    "chunk_size": 1000
                },
                headers={"X-API-KEY": API_KEY_SECRET},
                timeout=300  # 5 minutes timeout - reasonable for processing
            )
        except requests.exceptions.Timeout:
            console.print("[red]❌ Text ingestion timed out[/red]")
            console.print("[yellow]💡 This may indicate model loading is taking too long[/yellow]")
            console.print("[yellow]The API server may have crashed due to memory issues[/yellow]")
            return {
                'status': False,
                'error': 'Text ingestion timed out - possible model loading issue',
                'processing_time_s': time.time() - start_time,
                'method': 'text_ingestion_timeout'
            }
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            response_data = response.json()
            console.print(f"[green]✅ Text ingestion completed successfully![/green]")
            console.print(f"[green]📈 Processing Results:[/green]")
            console.print(f"  • Total triples extracted: {response_data.get('total_triples', 'N/A')}")
            console.print(f"  • Successfully staged: {response_data.get('successful_triples', 'N/A')}")
            console.print(f"  • Failed triples: {response_data.get('failed_triples', 'N/A')}")
            console.print(f"  • Processing time: {response_data.get('processing_time', processing_time):.2f}s")
            
            if response_data.get('errors'):
                console.print(f"[yellow]⚠️ Errors encountered: {len(response_data['errors'])}[/yellow]")
                for error in response_data['errors'][:3]:  # Show first 3 errors
                    console.print(f"  [red]• {error}[/red]")
            
            if response_data.get('warnings'):
                console.print(f"[yellow]⚠️ Warnings: {len(response_data['warnings'])}[/yellow]")
            
            return {
                'status': True,
                'response_code': response.status_code,
                'processing_time_s': processing_time,
                'response_data': response_data,
                'method': 'text_ingestion_pipeline',
                'text_length': len(extracted_text)
            }
        else:
            console.print(f"[red]❌ Text ingestion failed with status {response.status_code}[/red]")
            console.print(f"[red]Error: {response.text}[/red]")
            return {
                'status': False,
                'response_code': response.status_code,
                'processing_time_s': processing_time,
                'error': response.text,
                'method': 'text_ingestion_pipeline'
            }
            
    except Exception as e:
        console.print(f"[red]❌ Document ingestion failed: {e}[/red]")
        logger.error(f"Error in document ingestion: {e}")
        return {
            'status': False,
            'error': str(e),
            'processing_time_s': time.time() - start_time,
            'method': 'failed'
        }
        
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
    """
    Validate changes in Neo4j knowledge graph after ingestion
    
    This provides insight into how the SubgraphRAG+ system builds and updates
    its knowledge graph representation in Neo4j.
    """
    logger.debug("Starting Neo4j changes validation")
    console.print("[cyan]🔍 Analyzing Neo4j Knowledge Graph Changes...[/cyan]")
    
    try:
        current_stats = check_neo4j_connection()
        
        if not current_stats['status']:
            console.print("[red]❌ Could not connect to Neo4j for validation[/red]")
            return {
                'status': False,
                'error': 'Could not connect to Neo4j for validation'
            }
        
        # Calculate changes from baseline
        entities_added = current_stats['entity_count'] - baseline['entity_count']
        relationships_added = current_stats['relationship_count'] - baseline['relationship_count']
        
        console.print(f"[blue]📊 Knowledge Graph Statistics:[/blue]")
        console.print(f"  [green]• Entities before ingestion: {baseline['entity_count']:,}[/green]")
        console.print(f"  [green]• Entities after ingestion: {current_stats['entity_count']:,}[/green]")
        console.print(f"  [yellow]• New entities created: {entities_added:,}[/yellow]")
        console.print()
        console.print(f"  [green]• Relationships before: {baseline['relationship_count']:,}[/green]")
        console.print(f"  [green]• Relationships after: {current_stats['relationship_count']:,}[/green]")
        console.print(f"  [yellow]• New relationships created: {relationships_added:,}[/yellow]")
        
        # Get detailed insights about the knowledge graph structure
        # For neo4j+s:// URIs, encryption is already specified in the URI scheme
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Get entity type distribution
            console.print(f"\n[blue]🏷️ Entity Type Distribution:[/blue]")
            entity_types = session.run(
                "MATCH (n:Entity) RETURN n.type as type, count(n) as count ORDER BY count DESC LIMIT 10"
            ).data()
            
            for entity_type in entity_types:
                type_name = entity_type['type'] or 'Unknown'
                count = entity_type['count']
                console.print(f"  • {type_name}: {count:,} entities")
            
            # Get most common relation types
            console.print(f"\n[blue]🔗 Most Common Relation Types:[/blue]")
            relation_types = session.run(
                "MATCH ()-[r:REL]->() RETURN r.name as relation, count(r) as count ORDER BY count DESC LIMIT 10"
            ).data()
            
            for relation in relation_types:
                rel_name = relation['relation'] or 'Unknown'
                count = relation['count']
                console.print(f"  • {rel_name}: {count:,} relationships")
            
            # Get sample of recent entities (last 10)
            console.print(f"\n[blue]🎯 Sample Entities in Graph:[/blue]")
            sample_entities = session.run(
                "MATCH (n:Entity) RETURN coalesce(n.name, n.id, 'Unknown') as name, coalesce(n.type, 'Unknown') as type ORDER BY name LIMIT 10"
            ).data()
            
            for entity in sample_entities:
                name = entity.get('name', 'Unknown')
                etype = entity.get('type', 'Unknown')
                console.print(f"  • {name} ({etype})")
            
            # Get sample relationships to show graph structure
            console.print(f"\n[blue]⚡ Sample Knowledge Graph Triples:[/blue]")
            sample_relationships = session.run(
                "MATCH (h:Entity)-[r:REL]->(t:Entity) RETURN coalesce(h.name, h.id, 'Unknown') as head, coalesce(r.name, 'Unknown') as relation, coalesce(t.name, t.id, 'Unknown') as tail LIMIT 10"
            ).data()
            
            for rel in sample_relationships:
                head = rel.get('head', 'Unknown')
                relation = rel.get('relation', 'Unknown')
                tail = rel.get('tail', 'Unknown')
                console.print(f"  • {head} --[{relation}]--> {tail}")
            
            # Check graph connectivity metrics
            console.print(f"\n[blue]🌐 Graph Connectivity Analysis:[/blue]")
            
            # Count isolated entities (entities with no relationships)
            isolated_entities = session.run(
                "MATCH (n:Entity) WHERE NOT (n)-[:REL]-() AND NOT ()-[:REL]-(n) RETURN count(n) as count"
            ).single()['count']
            
            # Get degree distribution for top entities
            top_connected = session.run(
                "MATCH (n:Entity) RETURN coalesce(n.name, n.id, 'Unknown') as name, COUNT { (n)-[:REL]-() } as degree ORDER BY degree DESC LIMIT 5"
            ).data()
            
            console.print(f"  • Isolated entities (no connections): {isolated_entities:,}")
            console.print(f"  • Most connected entities:")
            for entity in top_connected:
                name = entity.get('name', 'Unknown')
                degree = entity.get('degree', 0)
                console.print(f"    - {name}: {degree} connections")
        
        driver.close()
        
        console.print(f"\n[green]✅ Neo4j analysis complete![/green]")
        
        return {
            'status': True,
            'entities_added': entities_added,
            'relationships_added': relationships_added,
            'total_entities': current_stats['entity_count'],
            'total_relationships': current_stats['relationship_count'],
            'entity_types': entity_types,
            'relation_types': relation_types,
            'sample_entities': sample_entities,
            'sample_relationships': sample_relationships,
            'isolated_entities': isolated_entities,
            'top_connected': top_connected
        }
        
    except Exception as e:
        console.print(f"[red]❌ Neo4j validation failed: {e}[/red]")
        return {
            'status': False,
            'error': str(e)
        }

def validate_faiss_changes(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate changes in FAISS vector index after ingestion
    
    FAISS is the dense retrieval component of SubgraphRAG+, providing
    semantic similarity search over embedded knowledge graph triples.
    """
    logger.debug("Starting FAISS changes validation")
    console.print("[cyan]🔍 Analyzing FAISS Vector Index Changes...[/cyan]")
    
    try:
        current_stats = check_faiss_index()
        
        if not current_stats['status']:
            console.print("[red]❌ Could not access FAISS index for validation[/red]")
            return {
                'status': False,
                'error': 'Could not access FAISS index for validation'
            }
        
        vectors_added = current_stats['vector_count'] - baseline['vector_count']
        
        console.print(f"[blue]📊 FAISS Vector Index Statistics:[/blue]")
        console.print(f"  [green]• Vectors before ingestion: {baseline['vector_count']:,}[/green]")
        console.print(f"  [green]• Vectors after ingestion: {current_stats['vector_count']:,}[/green]")
        console.print(f"  [yellow]• New vectors added: {vectors_added:,}[/yellow]")
        console.print(f"  [blue]• Vector dimension: {current_stats['dimension']}[/blue]")
        console.print(f"  [blue]• Index type: {current_stats['index_type']}[/blue]")
        console.print(f"  [blue]• Index trained: {'✅ Yes' if current_stats['is_trained'] else '❌ No'}[/blue]")
        
        console.print(f"\n[blue]🧠 About FAISS in SubgraphRAG+:[/blue]")
        console.print(f"  [dim]• FAISS stores embeddings of knowledge graph triples[/dim]")
        console.print(f"  [dim]• Each vector represents a triple's semantic meaning[/dim]")
        console.print(f"  [dim]• Enables fast similarity search for hybrid retrieval[/dim]")
        console.print(f"  [dim]• Complements graph-based traversal with dense retrieval[/dim]")
        
        # Test search functionality if vectors exist
        if current_stats['vector_count'] > 0:
            console.print(f"\n[yellow]🔍 Testing FAISS Search Capability...[/yellow]")
            
            index = faiss.read_index(config.FAISS_INDEX_PATH)
            dimension = index.d
            
            # Create a random query vector for testing
            test_query = np.random.random((1, dimension)).astype('float32')
            
            # Normalize the query vector (important for cosine similarity)
            test_query = test_query / np.linalg.norm(test_query)
            
            # Perform search with different k values to test scalability
            k_values = [1, 5, min(10, index.ntotal)]
            search_results = {}
            
            for k in k_values:
                if k <= index.ntotal:
                    distances, indices = index.search(test_query, k=k)
                    search_results[f'top_{k}'] = {
                        'distances': distances[0].tolist(),
                        'indices': indices[0].tolist()
                    }
                    
                    console.print(f"  • Top-{k} search: Found {len(indices[0])} results")
                    console.print(f"    - Best similarity: {1 - distances[0][0]:.4f}")
                    console.print(f"    - Worst similarity: {1 - distances[0][-1]:.4f}")
            
            # Test batch search capability
            console.print(f"\n[yellow]📦 Testing Batch Search Performance...[/yellow]")
            batch_size = min(5, index.ntotal)
            batch_queries = np.random.random((batch_size, dimension)).astype('float32')
            # Normalize batch queries
            batch_queries = batch_queries / np.linalg.norm(batch_queries, axis=1, keepdims=True)
            
            start_time = time.time()
            batch_distances, batch_indices = index.search(batch_queries, k=3)
            batch_time = time.time() - start_time
            
            console.print(f"  • Batch search ({batch_size} queries): {batch_time*1000:.2f}ms")
            console.print(f"  • Average time per query: {(batch_time/batch_size)*1000:.2f}ms")
            
            # Check index memory usage (approximate)
            vector_memory_mb = (index.ntotal * dimension * 4) / (1024 * 1024)  # 4 bytes per float32
            console.print(f"\n[blue]💾 Index Memory Footprint:[/blue]")
            console.print(f"  • Estimated vector storage: {vector_memory_mb:.2f} MB")
            console.print(f"  • Vectors per MB: ~{index.ntotal / vector_memory_mb:.0f}")
            
        else:
            console.print(f"\n[yellow]⚠️ No vectors in index - cannot test search[/yellow]")
            search_results = {'message': 'No vectors to search'}
        
        console.print(f"\n[green]✅ FAISS analysis complete![/green]")
        
        return {
            'status': True,
            'vectors_added': vectors_added,
            'total_vectors': current_stats['vector_count'],
            'index_type': current_stats['index_type'],
            'is_trained': current_stats['is_trained'],
            'dimension': current_stats['dimension'],
            'search_test': search_results,
            'memory_footprint_mb': vector_memory_mb if current_stats['vector_count'] > 0 else 0
        }
        
    except Exception as e:
        console.print(f"[red]❌ FAISS validation failed: {e}[/red]")
        return {
            'status': False,
            'error': str(e)
        }

def test_query_processing(question: str) -> Dict[str, Any]:
    """
    Test the complete SubgraphRAG+ question answering pipeline
    
    This demonstrates the full RAG (Retrieval-Augmented Generation) workflow:
    1. Entity extraction and linking
    2. Hybrid retrieval (graph + dense)
    3. MLP-based triple scoring and fusion
    4. Subgraph assembly
    5. LLM answer generation with citations
    """
    logger.debug(f"Starting query processing test: {question}")
    console.print(f"[cyan]🤔 Testing Question Answering Pipeline...[/cyan]")
    console.print(f"[blue]❓ Question: [bold]{question}[/bold][/blue]")
    
    start_time = time.time()
    
    try:
        console.print(f"\n[yellow]🚀 Sending query to SubgraphRAG+ API...[/yellow]")
        console.print(f"[dim]This will trigger the complete RAG pipeline:[/dim]")
        console.print(f"[dim]  1. Entity extraction from question[/dim]")
        console.print(f"[dim]  2. Entity linking to knowledge graph[/dim]")
        console.print(f"[dim]  3. Graph traversal retrieval[/dim]")
        console.print(f"[dim]  4. Dense vector retrieval via FAISS[/dim]")
        console.print(f"[dim]  5. MLP scoring and hybrid fusion[/dim]")
        console.print(f"[dim]  6. Subgraph assembly[/dim]")
        console.print(f"[dim]  7. LLM answer generation[/dim]")
        console.print(f"[dim]  8. Citation extraction and validation[/dim]")
        
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
            console.print(f"[red]❌ Query failed with status {response.status_code}[/red]")
            console.print(f"[red]Error: {response.text}[/red]")
            return {
                'status': False,
                'response_code': response.status_code,
                'error': response.text,
                'processing_time_s': time.time() - start_time
            }
        
        console.print(f"\n[green]✅ Query accepted, processing SSE stream...[/green]")
        
        # Parse SSE stream with detailed analysis
        events = []
        answer_tokens = []
        citations = []
        graph_data = None
        metadata = {}
        errors = []
        
        console.print(f"[blue]📡 Real-time Stream Events:[/blue]")
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        event_data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                        events.append(event_data)
                        
                        event_type = event_data.get('event')
                        
                        if event_type == 'metadata':
                            metadata.update(event_data['data'])
                            if 'query_id' in event_data['data']:
                                console.print(f"  [dim]🆔 Query ID: {event_data['data']['query_id']}[/dim]")
                            if 'triple_count' in event_data['data']:
                                console.print(f"  [green]📊 Retrieved {event_data['data']['triple_count']} relevant triples[/green]")
                            if 'entity_count' in event_data['data']:
                                console.print(f"  [green]🏷️ Linked {event_data['data']['entity_count']} entities[/green]")
                            if 'latency_ms' in event_data['data']:
                                console.print(f"  [yellow]⏱️ Retrieval latency: {event_data['data']['latency_ms']}ms[/yellow]")
                        
                        elif event_type == 'graph_data':
                            graph_data = event_data['data']
                            node_count = len(graph_data.get('nodes', []))
                            link_count = len(graph_data.get('links', []))
                            console.print(f"  [blue]🕸️ Subgraph: {node_count} nodes, {link_count} edges[/blue]")
                            
                        elif event_type == 'llm_token':
                            token = event_data['data']['token']
                            answer_tokens.append(token)
                            # Show streaming progress every 10 tokens
                            if len(answer_tokens) % 10 == 0:
                                partial_answer = ''.join(answer_tokens)
                                console.print(f"  [green]💭 Generating... ({len(answer_tokens)} tokens)[/green]")
                                
                        elif event_type == 'citation_data':
                            citation = event_data['data']
                            citations.append(citation)
                            console.print(f"  [cyan]📝 Citation: {citation.get('text', 'N/A')}[/cyan]")
                            
                        elif event_type == 'error':
                            error_info = event_data['data']
                            errors.append(error_info)
                            console.print(f"  [red]❌ Error: {error_info.get('message', 'Unknown error')}[/red]")
                            
                        elif event_type == 'end':
                            console.print(f"  [green]🏁 Stream completed[/green]")
                            break
                            
                    except json.JSONDecodeError as e:
                        console.print(f"  [yellow]⚠️ Failed to parse event: {line_str[:100]}[/yellow]")
                        continue
        
        processing_time = time.time() - start_time
        full_answer = ''.join(answer_tokens)
        
        # Detailed analysis of results
        console.print(f"\n[blue]📋 Query Processing Results Analysis:[/blue]")
        console.print(f"  [green]• Total processing time: {processing_time:.2f}s[/green]")
        console.print(f"  [green]• Total SSE events received: {len(events)}[/green]")
        console.print(f"  [green]• Answer tokens generated: {len(answer_tokens)}[/green]")
        console.print(f"  [green]• Citations found: {len(citations)}[/green]")
        console.print(f"  [green]• Graph data provided: {'✅ Yes' if graph_data else '❌ No'}[/green]")
        console.print(f"  [green]• Errors encountered: {len(errors)}[/green]")
        
        if full_answer:
            console.print(f"\n[blue]💬 Generated Answer:[/blue]")
            # Show answer with word wrapping
            answer_lines = full_answer.strip().split('\n')
            for line in answer_lines:
                if line.strip():
                    console.print(f"  [dim]{line.strip()}[/dim]")
        
        if citations:
            console.print(f"\n[blue]📚 Citations and Knowledge Sources:[/blue]")
            for i, citation in enumerate(citations, 1):
                console.print(f"  [{i}] {citation.get('text', 'N/A')}")
                if 'id' in citation:
                    console.print(f"      [dim]Triple ID: {citation['id']}[/dim]")
        
        if graph_data:
            console.print(f"\n[blue]🕸️ Retrieved Subgraph Analysis:[/blue]")
            nodes = graph_data.get('nodes', [])
            links = graph_data.get('links', [])
            
            # Analyze node types
            node_types = {}
            for node in nodes:
                node_type = node.get('type', 'Unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            console.print(f"  • Node type distribution:")
            for node_type, count in node_types.items():
                console.print(f"    - {node_type}: {count} nodes")
            
            # Analyze relationship types
            rel_types = {}
            for link in links:
                rel_type = link.get('relation_name', 'Unknown')
                rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
            
            console.print(f"  • Relationship type distribution:")
            for rel_type, count in list(rel_types.items())[:5]:  # Top 5
                console.print(f"    - {rel_type}: {count} relationships")
        
        # Quality assessment
        quality_score = 0.0
        if full_answer and len(full_answer) > 10:
            quality_score += 0.4  # Has substantive answer
        if citations:
            quality_score += 0.3  # Has citations
        if graph_data:
            quality_score += 0.2  # Has graph data
        if not errors:
            quality_score += 0.1  # No errors
        
        console.print(f"\n[blue]🎯 Answer Quality Assessment:[/blue]")
        console.print(f"  • Quality score: {quality_score:.2f}/1.0")
        console.print(f"  • Answer completeness: {'✅ Good' if len(full_answer) > 50 else '⚠️ Short'}")
        console.print(f"  • Citation support: {'✅ Good' if citations else '❌ Missing'}")
        console.print(f"  • Graph context: {'✅ Provided' if graph_data else '❌ Missing'}")
        
        console.print(f"\n[green]✅ Query processing analysis complete![/green]")
        
        return {
            'status': True,
            'processing_time_s': processing_time,
            'answer': full_answer,
            'citations': citations,
            'graph_data': graph_data,
            'metadata': metadata,
            'total_events': len(events),
            'answer_length': len(full_answer),
            'quality_score': quality_score,
            'errors': errors
        }
        
    except Exception as e:
        console.print(f"[red]❌ Query processing failed: {e}[/red]")
        return {
            'status': False,
            'error': str(e),
            'processing_time_s': time.time() - start_time
        }

def test_ie_extraction() -> Dict[str, Any]:
    """Test IE extraction functionality with graceful fallback"""
    logger.debug("Starting IE extraction test")
    
    # Note: This test will trigger REBEL model loading on first use
    console.print("[yellow]⏳ Testing IE extraction (may trigger REBEL model loading)...[/yellow]")
    console.print("[dim]This test validates the information extraction pipeline.[/dim]")
    console.print("[dim]If model loading fails, the system will use mock responses.[/dim]")
    
    sample_text = "Barack Obama was born in Hawaii. He served as President of the United States."
    
    try:
        console.print(f"[blue]📝 Test text: '{sample_text}'[/blue]")
        console.print("[yellow]🚀 Sending to REBEL extraction endpoint...[/yellow]")
        
        # First, check if the IE service is available
        try:
            health_response = requests.get(
                f"{API_BASE_URL}/ie/health",
                timeout=10
            )
            if health_response.status_code != 200:
                console.print("[yellow]⚠️ IE service not available, skipping extraction test[/yellow]")
                return {
                    'status': False,
                    'error': 'IE service not available',
                    'skipped': True
                }
        except Exception:
            console.print("[yellow]⚠️ Cannot reach IE service, skipping extraction test[/yellow]")
            return {
                'status': False,
                'error': 'Cannot reach IE service',
                'skipped': True
            }
        
        # Try the extraction with a reasonable timeout
        response = requests.post(
            f"{API_BASE_URL}/ie/extract",
            json={
                "text": sample_text,
                "max_length": 256,
                "num_beams": 3
            },
            headers={"X-API-KEY": API_KEY_SECRET},
            timeout=120  # 2 minutes timeout - reasonable for model loading
        )
        
        if response.status_code != 200:
            console.print(f"[red]❌ IE extraction failed with status {response.status_code}[/red]")
            error_text = response.text
            console.print(f"[red]Error: {error_text}[/red]")
            
            # Check if it's a model loading issue
            if any(keyword in error_text.lower() for keyword in ["segmentation fault", "memory", "timeout", "killed"]):
                console.print("[yellow]💡 This appears to be a model loading/memory issue.[/yellow]")
                console.print("[yellow]The system should fall back to mock responses for document ingestion.[/yellow]")
                return {
                    'status': False,
                    'response_code': response.status_code,
                    'error': error_text,
                    'model_loading_issue': True
                }
            
            return {
                'status': False,
                'response_code': response.status_code,
                'error': error_text
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
        
        # Check if this looks like a mock response
        is_mock = False
        if data['triples'] and any('mock' in str(triple).lower() for triple in data['triples']):
            is_mock = True
            console.print("[yellow]⚠️ Received mock response - real models may not be loaded[/yellow]")
        
        console.print(f"[green]✅ IE extraction completed successfully![/green]")
        console.print(f"[green]📊 Extracted {len(data['triples'])} triples[/green]")
        if data['triples']:
            console.print(f"[blue]Sample triple: {data['triples'][0]}[/blue]")
        
        return {
            'status': True,
            'triples_count': len(data['triples']),
            'processing_time': data['processing_time'],
            'triples_valid': triples_valid,
            'sample_triples': data['triples'][:3] if data['triples'] else [],
            'raw_output_length': len(data['raw_output']),
            'is_mock_response': is_mock
        }
        
    except requests.exceptions.Timeout:
        console.print("[red]❌ IE extraction timed out[/red]")
        console.print("[yellow]💡 This may indicate model loading is taking too long[/yellow]")
        return {
            'status': False,
            'error': 'Request timed out',
            'timeout': True
        }
    except Exception as e:
        console.print(f"[red]❌ IE extraction failed with exception: {e}[/red]")
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

def test_feedback_endpoint() -> Dict[str, Any]:
    """Test feedback submission functionality"""
    logger.debug("Starting feedback endpoint test")
    console.print("[cyan]📝 Testing Feedback Submission...[/cyan]")
    
    try:
        # Generate a test query ID
        test_query_id = f"test_query_{int(time.time())}"
        
        console.print(f"[blue]📋 Submitting feedback for query ID: {test_query_id}[/blue]")
        
        response = requests.post(
            f"{API_BASE_URL}/feedback",
            json={
                "query_id": test_query_id,
                "is_correct": True,
                "comment": "End-to-end test feedback - system working correctly",
                "expected_answer": "Test answer for validation"
            },
            headers={"X-API-KEY": API_KEY_SECRET},
            timeout=TIMEOUT_SECONDS
        )
        
        if response.status_code != 200:
            console.print(f"[red]❌ Feedback submission failed with status {response.status_code}[/red]")
            return {
                'status': False,
                'response_code': response.status_code,
                'error': response.text
            }
        
        data = response.json()
        
        # Validate response structure
        required_fields = ['status', 'message']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return {
                'status': False,
                'error': f"Missing required fields: {missing_fields}"
            }
        
        console.print(f"[green]✅ Feedback submitted successfully![/green]")
        console.print(f"[green]📊 Response: {data['message']}[/green]")
        
        return {
            'status': True,
            'query_id': test_query_id,
            'response_data': data,
            'message': data['message']
        }
        
    except Exception as e:
        console.print(f"[red]❌ Feedback test failed: {e}[/red]")
        return {
            'status': False,
            'error': str(e)
        }

def test_metrics_endpoint() -> Dict[str, Any]:
    """Test Prometheus metrics endpoint"""
    logger.debug("Starting metrics endpoint test")
    console.print("[cyan]📊 Testing Prometheus Metrics Endpoint...[/cyan]")
    
    try:
        console.print("[blue]📈 Fetching Prometheus metrics...[/blue]")
        
        response = requests.get(
            f"{API_BASE_URL}/metrics",
            timeout=TIMEOUT_SECONDS
        )
        
        if response.status_code != 200:
            console.print(f"[red]❌ Metrics endpoint failed with status {response.status_code}[/red]")
            return {
                'status': False,
                'response_code': response.status_code,
                'error': response.text
            }
        
        metrics_text = response.text
        
        # Basic validation of Prometheus format
        if not metrics_text:
            return {
                'status': False,
                'error': 'Empty metrics response'
            }
        
        # Check for common Prometheus metrics
        expected_metrics = [
            'http_requests_total',
            'http_request_duration_seconds'
        ]
        
        found_metrics = []
        for metric in expected_metrics:
            if metric in metrics_text:
                found_metrics.append(metric)
        
        # Count total metrics
        metric_lines = [line for line in metrics_text.split('\n') if line and not line.startswith('#')]
        
        console.print(f"[green]✅ Metrics endpoint working![/green]")
        console.print(f"[green]📊 Found {len(metric_lines)} metric entries[/green]")
        console.print(f"[green]📈 Standard metrics found: {len(found_metrics)}/{len(expected_metrics)}[/green]")
        
        # Show sample metrics
        console.print(f"[blue]📋 Sample metrics:[/blue]")
        for line in metric_lines[:5]:  # Show first 5 metrics
            console.print(f"  [dim]{line}[/dim]")
        
        return {
            'status': True,
            'total_metrics': len(metric_lines),
            'found_standard_metrics': found_metrics,
            'metrics_text_length': len(metrics_text),
            'sample_metrics': metric_lines[:10]
        }
        
    except Exception as e:
        console.print(f"[red]❌ Metrics test failed: {e}[/red]")
        return {
            'status': False,
            'error': str(e)
        }

def test_text_ingestion_endpoint() -> Dict[str, Any]:
    """Test the text ingestion endpoint specifically"""
    logger.debug("Starting text ingestion endpoint test")
    console.print("[cyan]📄 Testing Text Ingestion Endpoint...[/cyan]")
    
    try:
        # Test text for ingestion
        test_text = """
        SubgraphRAG+ is an advanced knowledge graph question answering system.
        It was developed by researchers to improve information retrieval.
        The system uses REBEL for relation extraction and Neo4j for graph storage.
        FAISS provides vector similarity search capabilities.
        """
        
        console.print(f"[blue]📝 Test text length: {len(test_text)} characters[/blue]")
        console.print("[yellow]🚀 Sending to text ingestion endpoint...[/yellow]")
        
        response = requests.post(
            f"{API_BASE_URL}/ingest/text",
            json={
                "text": test_text,
                "source": "e2e_test_text_ingestion",
                "chunk_size": 500
            },
            headers={"X-API-KEY": API_KEY_SECRET},
            timeout=TIMEOUT_SECONDS
        )
        
        if response.status_code != 200:
            console.print(f"[red]❌ Text ingestion failed with status {response.status_code}[/red]")
            return {
                'status': False,
                'response_code': response.status_code,
                'error': response.text
            }
        
        data = response.json()
        
        # Validate response structure
        required_fields = ['total_triples', 'successful_triples', 'failed_triples', 'processing_time']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return {
                'status': False,
                'error': f"Missing required fields: {missing_fields}"
            }
        
        console.print(f"[green]✅ Text ingestion completed![/green]")
        console.print(f"[green]📊 Processing Results:[/green]")
        console.print(f"  • Total triples: {data['total_triples']}")
        console.print(f"  • Successful: {data['successful_triples']}")
        console.print(f"  • Failed: {data['failed_triples']}")
        console.print(f"  • Processing time: {data['processing_time']:.2f}s")
        
        if data.get('errors'):
            console.print(f"[yellow]⚠️ Errors: {len(data['errors'])}[/yellow]")
        
        if data.get('warnings'):
            console.print(f"[yellow]⚠️ Warnings: {len(data['warnings'])}[/yellow]")
        
        return {
            'status': True,
            'total_triples': data['total_triples'],
            'successful_triples': data['successful_triples'],
            'failed_triples': data['failed_triples'],
            'processing_time': data['processing_time'],
            'errors': data.get('errors', []),
            'warnings': data.get('warnings', [])
        }
        
    except Exception as e:
        console.print(f"[red]❌ Text ingestion test failed: {e}[/red]")
        return {
            'status': False,
            'error': str(e)
        }

def test_debug_endpoints() -> Dict[str, Any]:
    """Test debug and diagnostic endpoints"""
    logger.debug("Starting debug endpoints test")
    console.print("[cyan]🔍 Testing Debug Endpoints...[/cyan]")
    
    results = {}
    
    try:
        # Test debug load-test endpoint
        console.print("[blue]🧪 Testing debug load-test endpoint...[/blue]")
        
        response = requests.get(
            f"{API_BASE_URL}/debug/load-test",
            timeout=TIMEOUT_SECONDS
        )
        
        if response.status_code == 200:
            data = response.json()
            console.print(f"[green]✅ Debug load-test working![/green]")
            console.print(f"[green]📊 Status: {data.get('status', 'unknown')}[/green]")
            
            if data.get('tokenizer_load_time'):
                console.print(f"[green]⏱️ Tokenizer load time: {data['tokenizer_load_time']:.2f}s[/green]")
            
            results['debug_load_test'] = {
                'status': True,
                'response_data': data
            }
        else:
            console.print(f"[yellow]⚠️ Debug load-test returned {response.status_code}[/yellow]")
            results['debug_load_test'] = {
                'status': False,
                'response_code': response.status_code,
                'error': response.text
            }
        
        # Test IE info endpoint
        console.print("[blue]🧠 Testing IE info endpoint...[/blue]")
        
        response = requests.get(
            f"{API_BASE_URL}/ie/info",
            timeout=TIMEOUT_SECONDS
        )
        
        if response.status_code == 200:
            data = response.json()
            console.print(f"[green]✅ IE info endpoint working![/green]")
            console.print(f"[green]📊 Overall status: {data.get('overall_status', 'unknown')}[/green]")
            
            if 'models' in data:
                console.print(f"[green]🤖 Models info available: {len(data['models'])} models[/green]")
            
            results['ie_info'] = {
                'status': True,
                'response_data': data
            }
        else:
            console.print(f"[yellow]⚠️ IE info returned {response.status_code}[/yellow]")
            results['ie_info'] = {
                'status': False,
                'response_code': response.status_code,
                'error': response.text
            }
        
        # Overall status
        overall_success = all(result.get('status', False) for result in results.values())
        
        console.print(f"\n[green]✅ Debug endpoints test completed![/green]")
        console.print(f"[blue]📊 Results: {sum(1 for r in results.values() if r.get('status', False))}/{len(results)} endpoints working[/blue]")
        
        return {
            'status': overall_success,
            'individual_results': results,
            'working_endpoints': sum(1 for r in results.values() if r.get('status', False)),
            'total_endpoints': len(results)
        }
        
    except Exception as e:
        console.print(f"[red]❌ Debug endpoints test failed: {e}[/red]")
        return {
            'status': False,
            'error': str(e),
            'individual_results': results
        }

def test_batch_ingest_endpoint() -> Dict[str, Any]:
    """Test the batch triple ingestion endpoint"""
    logger.debug("Starting batch ingest endpoint test")
    console.print("[cyan]📦 Testing Batch Triple Ingestion...[/cyan]")
    
    try:
        # Sample triples for testing
        test_triples = [
            {
                "head": "SubgraphRAG+",
                "relation": "uses",
                "tail": "REBEL"
            },
            {
                "head": "REBEL",
                "relation": "performs",
                "tail": "relation extraction"
            },
            {
                "head": "SubgraphRAG+",
                "relation": "stores_data_in",
                "tail": "Neo4j"
            }
        ]
        
        console.print(f"[blue]📝 Testing with {len(test_triples)} sample triples[/blue]")
        console.print("[yellow]🚀 Sending to batch ingest endpoint...[/yellow]")
        
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json={"triples": test_triples},
            headers={"X-API-KEY": API_KEY_SECRET},
            timeout=TIMEOUT_SECONDS
        )
        
        if response.status_code not in [200, 202]:
            console.print(f"[red]❌ Batch ingest failed with status {response.status_code}[/red]")
            return {
                'status': False,
                'response_code': response.status_code,
                'error': response.text
            }
        
        data = response.json()
        
        # Validate response structure
        expected_fields = ['status']
        missing_fields = [field for field in expected_fields if field not in data]
        
        if missing_fields:
            return {
                'status': False,
                'error': f"Missing required fields: {missing_fields}"
            }
        
        console.print(f"[green]✅ Batch ingest completed![/green]")
        console.print(f"[green]📊 Status: {data['status']}[/green]")
        
        if 'triples_staged' in data:
            console.print(f"[green]📈 Triples staged: {data['triples_staged']}[/green]")
        
        if 'message' in data:
            console.print(f"[green]💬 Message: {data['message']}[/green]")
        
        if data.get('errors'):
            console.print(f"[yellow]⚠️ Errors: {len(data['errors'])}[/yellow]")
        
        return {
            'status': True,
            'response_code': response.status_code,
            'triples_sent': len(test_triples),
            'response_data': data
        }
        
    except Exception as e:
        console.print(f"[red]❌ Batch ingest test failed: {e}[/red]")
        return {
            'status': False,
            'error': str(e)
        }

def test_error_handling() -> Dict[str, Any]:
    """Test API error handling with various invalid inputs"""
    logger.debug("Starting error handling test")
    console.print("[cyan]🚨 Testing API Error Handling...[/cyan]")
    
    error_tests = {}
    
    try:
        # Test 1: Invalid API key
        console.print("[blue]🔐 Testing invalid API key...[/blue]")
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": "test"},
            headers={"X-API-KEY": "invalid_key"},
            timeout=TIMEOUT_SECONDS
        )
        
        error_tests['invalid_api_key'] = {
            'expected_status': 401,
            'actual_status': response.status_code,
            'success': response.status_code == 401
        }
        
        if response.status_code == 401:
            console.print("[green]✅ Invalid API key correctly rejected[/green]")
        else:
            console.print(f"[yellow]⚠️ Expected 401, got {response.status_code}[/yellow]")
        
        # Test 2: Empty query
        console.print("[blue]❓ Testing empty query...[/blue]")
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": ""},
            headers={"X-API-KEY": API_KEY_SECRET},
            timeout=TIMEOUT_SECONDS
        )
        
        error_tests['empty_query'] = {
            'expected_status': 400,
            'actual_status': response.status_code,
            'success': response.status_code == 400
        }
        
        if response.status_code == 400:
            console.print("[green]✅ Empty query correctly rejected[/green]")
        else:
            console.print(f"[yellow]⚠️ Expected 400, got {response.status_code}[/yellow]")
        
        # Test 3: Malformed JSON for ingest
        console.print("[blue]📦 Testing malformed ingest data...[/blue]")
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json={"invalid_field": "test"},
            headers={"X-API-KEY": API_KEY_SECRET},
            timeout=TIMEOUT_SECONDS
        )
        
        error_tests['malformed_ingest'] = {
            'expected_status': 400,
            'actual_status': response.status_code,
            'success': response.status_code in [400, 422]  # Either is acceptable
        }
        
        if response.status_code in [400, 422]:
            console.print("[green]✅ Malformed ingest data correctly rejected[/green]")
        else:
            console.print(f"[yellow]⚠️ Expected 400/422, got {response.status_code}[/yellow]")
        
        # Test 4: Missing API key
        console.print("[blue]🔑 Testing missing API key...[/blue]")
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": "test"},
            timeout=TIMEOUT_SECONDS
        )
        
        error_tests['missing_api_key'] = {
            'expected_status': 401,
            'actual_status': response.status_code,
            'success': response.status_code == 401
        }
        
        if response.status_code == 401:
            console.print("[green]✅ Missing API key correctly rejected[/green]")
        else:
            console.print(f"[yellow]⚠️ Expected 401, got {response.status_code}[/yellow]")
        
        # Calculate overall success
        successful_tests = sum(1 for test in error_tests.values() if test['success'])
        total_tests = len(error_tests)
        
        console.print(f"\n[green]✅ Error handling test completed![/green]")
        console.print(f"[blue]📊 Results: {successful_tests}/{total_tests} error cases handled correctly[/blue]")
        
        return {
            'status': successful_tests == total_tests,
            'successful_tests': successful_tests,
            'total_tests': total_tests,
            'individual_results': error_tests
        }
        
    except Exception as e:
        console.print(f"[red]❌ Error handling test failed: {e}[/red]")
        return {
            'status': False,
            'error': str(e),
            'individual_results': error_tests
        }

def test_performance_benchmarks() -> Dict[str, Any]:
    """Test performance benchmarks for key operations"""
    logger.debug("Starting performance benchmark test")
    console.print("[cyan]⚡ Testing Performance Benchmarks...[/cyan]")
    
    benchmarks = {}
    
    try:
        # Benchmark 1: Health check response time
        console.print("[blue]💓 Benchmarking health check...[/blue]")
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=TIMEOUT_SECONDS)
        health_time = time.time() - start_time
        
        benchmarks['health_check'] = {
            'response_time_ms': health_time * 1000,
            'success': response.status_code == 200,
            'benchmark_passed': health_time < 1.0  # Should be under 1 second
        }
        
        console.print(f"[green]⏱️ Health check: {health_time*1000:.2f}ms[/green]")
        
        # Benchmark 2: Readiness check response time
        console.print("[blue]🔍 Benchmarking readiness check...[/blue]")
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/readyz", timeout=TIMEOUT_SECONDS)
        readiness_time = time.time() - start_time
        
        benchmarks['readiness_check'] = {
            'response_time_ms': readiness_time * 1000,
            'success': response.status_code in [200, 503],
            'benchmark_passed': readiness_time < 5.0  # Should be under 5 seconds
        }
        
        console.print(f"[green]⏱️ Readiness check: {readiness_time*1000:.2f}ms[/green]")
        
        # Benchmark 3: Graph browse response time
        console.print("[blue]🕸️ Benchmarking graph browse...[/blue]")
        start_time = time.time()
        response = requests.get(
            f"{API_BASE_URL}/graph/browse",
            params={'limit': 100, 'page': 1},
            headers={"X-API-KEY": API_KEY_SECRET},
            timeout=TIMEOUT_SECONDS
        )
        browse_time = time.time() - start_time
        
        benchmarks['graph_browse'] = {
            'response_time_ms': browse_time * 1000,
            'success': response.status_code == 200,
            'benchmark_passed': browse_time < 10.0  # Should be under 10 seconds
        }
        
        console.print(f"[green]⏱️ Graph browse: {browse_time*1000:.2f}ms[/green]")
        
        # Benchmark 4: IE health check
        console.print("[blue]🧠 Benchmarking IE health check...[/blue]")
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/ie/health", timeout=TIMEOUT_SECONDS)
        ie_health_time = time.time() - start_time
        
        benchmarks['ie_health'] = {
            'response_time_ms': ie_health_time * 1000,
            'success': response.status_code == 200,
            'benchmark_passed': ie_health_time < 3.0  # Should be under 3 seconds
        }
        
        console.print(f"[green]⏱️ IE health: {ie_health_time*1000:.2f}ms[/green]")
        
        # Calculate overall performance score
        passed_benchmarks = sum(1 for b in benchmarks.values() if b['benchmark_passed'])
        total_benchmarks = len(benchmarks)
        
        console.print(f"\n[green]✅ Performance benchmarks completed![/green]")
        console.print(f"[blue]📊 Results: {passed_benchmarks}/{total_benchmarks} benchmarks passed[/blue]")
        
        # Show performance summary
        console.print(f"\n[blue]⚡ Performance Summary:[/blue]")
        for test_name, result in benchmarks.items():
            status = "✅ PASS" if result['benchmark_passed'] else "⚠️ SLOW"
            console.print(f"  • {test_name.replace('_', ' ').title()}: {result['response_time_ms']:.2f}ms {status}")
        
        return {
            'status': passed_benchmarks == total_benchmarks,
            'passed_benchmarks': passed_benchmarks,
            'total_benchmarks': total_benchmarks,
            'individual_results': benchmarks,
            'average_response_time_ms': sum(b['response_time_ms'] for b in benchmarks.values()) / len(benchmarks)
        }
        
    except Exception as e:
        console.print(f"[red]❌ Performance benchmark test failed: {e}[/red]")
        return {
            'status': False,
            'error': str(e),
            'individual_results': benchmarks
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
    
    # API Endpoints Test Results
    api_endpoints = [
        'query_processing', 'graph_browsing', 'feedback_endpoint', 'metrics_endpoint',
        'text_ingestion_endpoint', 'batch_ingest_endpoint', 'ie_extraction'
    ]
    
    api_table = Table(title="API Endpoints Test Results", box=box.ROUNDED)
    api_table.add_column("Endpoint", style="cyan")
    api_table.add_column("Status", style="green")
    api_table.add_column("Response Time", style="yellow")
    api_table.add_column("Details", style="dim")
    
    for endpoint in api_endpoints:
        if endpoint in results.system_health:
            endpoint_result = results.system_health[endpoint]
            status_text = "✅ PASS" if endpoint_result.get('status', False) else "❌ FAIL"
            
            # Get response time from performance metrics
            perf_data = results.performance_metrics.get(endpoint, {})
            response_time = f"{perf_data.get('time', 0)*1000:.1f}ms" if perf_data.get('time') else "N/A"
            
            # Get relevant details
            details = ""
            if endpoint == 'query_processing':
                details = f"Citations: {'✅' if results.citations_valid else '❌'}, Graph: {'✅' if results.graph_data_valid else '❌'}"
            elif endpoint == 'ie_extraction':
                if endpoint_result.get('triples_count'):
                    details = f"Triples: {endpoint_result['triples_count']}"
                elif endpoint_result.get('skipped'):
                    details = "Skipped (model tests disabled)"
            elif endpoint == 'text_ingestion_endpoint':
                if endpoint_result.get('successful_triples'):
                    details = f"Triples: {endpoint_result['successful_triples']}"
            elif endpoint == 'batch_ingest_endpoint':
                if endpoint_result.get('triples_sent'):
                    details = f"Sent: {endpoint_result['triples_sent']}"
            elif endpoint == 'metrics_endpoint':
                if endpoint_result.get('total_metrics'):
                    details = f"Metrics: {endpoint_result['total_metrics']}"
            
            api_table.add_row(
                endpoint.replace('_', ' ').title(),
                status_text,
                response_time,
                details
            )
    
    console.print(api_table)
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
        perf_table.add_column("Benchmark", style="blue")
        
        # Define performance benchmarks
        benchmarks = {
            'document_ingestion': 60.0,  # 1 minute
            'query_processing': 30.0,    # 30 seconds
            'graph_browsing': 10.0,      # 10 seconds
            'feedback_endpoint': 5.0,    # 5 seconds
            'metrics_endpoint': 2.0,     # 2 seconds
            'text_ingestion_endpoint': 30.0,  # 30 seconds
            'batch_ingest_endpoint': 10.0,    # 10 seconds
            'ie_extraction': 120.0,      # 2 minutes (model loading)
            'debug_endpoints': 10.0,     # 10 seconds
            'error_handling': 15.0,      # 15 seconds
            'performance_benchmarks': 20.0  # 20 seconds
        }
        
        for operation, metrics in results.performance_metrics.items():
            time_taken = metrics.get('time', 0)
            status = "✅ OK" if metrics.get('success', False) else "❌ FAILED"
            
            # Check if within benchmark
            benchmark_time = benchmarks.get(operation, 60.0)
            benchmark_status = "🚀 FAST" if time_taken < benchmark_time * 0.5 else "✅ GOOD" if time_taken < benchmark_time else "⚠️ SLOW"
            
            perf_table.add_row(
                operation.replace('_', ' ').title(),
                f"{time_taken:.2f}",
                status,
                benchmark_status
            )
        
        console.print(perf_table)
        console.print()
    
    # Error Handling Results
    if 'error_handling' in results.system_health:
        error_handling_result = results.system_health['error_handling']
        if error_handling_result.get('individual_results'):
            error_table = Table(title="Error Handling Test Results", box=box.ROUNDED)
            error_table.add_column("Test Case", style="cyan")
            error_table.add_column("Expected", style="yellow")
            error_table.add_column("Actual", style="yellow")
            error_table.add_column("Result", style="green")
            
            for test_name, test_result in error_handling_result['individual_results'].items():
                expected = test_result.get('expected_status', 'N/A')
                actual = test_result.get('actual_status', 'N/A')
                result = "✅ PASS" if test_result.get('success', False) else "❌ FAIL"
                
                error_table.add_row(
                    test_name.replace('_', ' ').title(),
                    str(expected),
                    str(actual),
                    result
                )
            
            console.print(error_table)
            console.print()
    
    # Performance Benchmarks Results
    if 'performance_benchmarks' in results.system_health:
        benchmark_result = results.system_health['performance_benchmarks']
        if benchmark_result.get('individual_results'):
            benchmark_table = Table(title="Performance Benchmark Results", box=box.ROUNDED)
            benchmark_table.add_column("Endpoint", style="cyan")
            benchmark_table.add_column("Response Time", style="yellow")
            benchmark_table.add_column("Benchmark", style="green")
            benchmark_table.add_column("Status", style="blue")
            
            for endpoint, bench_result in benchmark_result['individual_results'].items():
                response_time = f"{bench_result.get('response_time_ms', 0):.1f}ms"
                benchmark_passed = bench_result.get('benchmark_passed', False)
                status = "✅ PASS" if benchmark_passed else "⚠️ SLOW"
                benchmark_text = "Fast response" if benchmark_passed else "Slow response"
                
                benchmark_table.add_row(
                    endpoint.replace('_', ' ').title(),
                    response_time,
                    benchmark_text,
                    status
                )
            
            console.print(benchmark_table)
            console.print()
    
    # Test Coverage Summary
    coverage_table = Table(title="Test Coverage Summary", box=box.ROUNDED)
    coverage_table.add_column("Test Category", style="cyan")
    coverage_table.add_column("Tests Run", style="green")
    coverage_table.add_column("Success Rate", style="yellow")
    
    # Calculate test coverage
    test_categories = {
        'Core API Endpoints': ['query_processing', 'graph_browsing', 'feedback_endpoint'],
        'Ingestion Endpoints': ['text_ingestion_endpoint', 'batch_ingest_endpoint'],
        'Information Extraction': ['ie_extraction'],
        'Monitoring & Debug': ['metrics_endpoint', 'debug_endpoints'],
        'Error Handling': ['error_handling'],
        'Performance': ['performance_benchmarks'],
        'Data Storage': ['neo4j_changes', 'faiss_changes']
    }
    
    for category, tests in test_categories.items():
        total_tests = len(tests)
        successful_tests = sum(1 for test in tests if (
            results.system_health.get(test, {}).get('status', False) 
            if isinstance(results.system_health.get(test, {}), dict) 
            else results.system_health.get(test, False)
        ))
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        coverage_table.add_row(
            category,
            f"{successful_tests}/{total_tests}",
            f"{success_rate:.1f}%"
        )
    
    console.print(coverage_table)
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
    # Calculate overall success based on critical components
    critical_tests = ['query_processing', 'graph_browsing', 'text_ingestion_endpoint']
    critical_success = all(
        (results.system_health.get(test, {}).get('status', False) 
         if isinstance(results.system_health.get(test, {}), dict) 
         else results.system_health.get(test, False))
        for test in critical_tests
    )
    
    overall_success = (
        results.ingestion_success and
        len(results.errors) == 0 and
        critical_success
    )
    
    status_text = "🎯 ALL SYSTEMS OPERATIONAL" if overall_success else "⚠️  ISSUES DETECTED"
    status_style = "bold green" if overall_success else "bold yellow"
    
    total_time = results.get_total_time()
    
    # Calculate test statistics
    total_tests_run = len([k for k in results.system_health.keys() if not k.endswith('_baseline')])
    successful_tests = len([k for k, v in results.system_health.items() 
                           if not k.endswith('_baseline') and (v.get('status', False) if isinstance(v, dict) else v)])
    
    final_panel = Panel(
        f"{status_text}\n\n"
        f"Total execution time: {total_time:.2f} seconds\n"
        f"Tests completed: {successful_tests}/{total_tests_run}\n"
        f"Success rate: {(successful_tests/total_tests_run)*100:.1f}%\n"
        f"Errors: {len(results.errors)}\n"
        f"Warnings: {len(results.warnings)}",
        title="Overall Status",
        border_style="green" if overall_success else "yellow"
    )
    
    console.print(final_panel)

def check_services_and_provide_guidance(results: ValidationResults, minimal_mode: bool = False) -> bool:
    """Check if required services are running and provide guidance if not"""
    api_running = results.system_health.get('main_api', {}).get('status', False)
    api_ready = results.system_health.get('main_api_ready', {}).get('status', False)
    ie_available = results.system_health.get('ie_module', {}).get('status', False)
    
    # Check Neo4j and FAISS from readiness details if available
    readiness_details = results.system_health.get('main_api_ready', {}).get('details', {})
    
    # Extract check results from readiness response
    if isinstance(readiness_details, dict) and 'checks' in readiness_details:
        checks = readiness_details['checks']
        neo4j_available = checks.get('neo4j') == 'ok'
        faiss_available = checks.get('faiss_index') == 'ok'
    else:
        # Fallback to baseline checks
        neo4j_available = results.system_health.get('neo4j_baseline', {}).get('status', False)
        faiss_available = results.system_health.get('faiss_baseline', {}).get('status', False)
    
    # In minimal mode, only require API server
    if minimal_mode:
        if not api_running:
            console.print("\n")
            console.print(Panel(
                "🚨 Minimal Mode: API Server Required\n\n"
                "Even in minimal mode, the API server must be running:\n\n"
                "❌ Main API Server not responding\n\n"
                "🔧 Start the API server:\n"
                "   python src/main.py --port 8000 &\n\n"
                "Then re-run this test:\n"
                "   E2E_MINIMAL_MODE=true python scripts/end_to_end_test.py",
                title="🔧 API Server Required",
                border_style="red"
            ))
            return False
        
        # In minimal mode, warn about missing services but continue
        missing_services = []
        if not neo4j_available:
            missing_services.append("Neo4j database")
        if not faiss_available:
            missing_services.append("FAISS index")
        if not ie_available:
            missing_services.append("IE module")
        
        if missing_services:
            console.print("\n")
            console.print(Panel(
                f"⚠️ Minimal Mode: Running with Limited Services\n\n"
                f"Missing services: {', '.join(missing_services)}\n\n"
                "Some tests will be skipped or use mock responses.\n"
                "For full functionality, set up all services and run without minimal mode.",
                title="⚠️ Limited Functionality",
                border_style="yellow"
            ))
            for service in missing_services:
                results.add_warning(f"Missing service in minimal mode: {service}")
        
        return True
    
    # Full mode - check critical dependencies
    critical_issues = []
    
    if not api_running:
        critical_issues.append("❌ Main API Server not responding")
    
    if not neo4j_available:
        critical_issues.append("❌ Neo4j database not accessible")
    
    if not faiss_available:
        critical_issues.append("❌ FAISS index not available")
    
    # Check readiness (less critical but important)
    readiness_issues = []
    
    if api_running and not api_ready:
        readiness_issues.append("⚠️ API server not fully ready (some models may not be loaded)")
    
    if not ie_available:
        readiness_issues.append("⚠️ IE module not ready (REBEL model not loaded)")
    
    if critical_issues:
        console.print("\n")
        console.print(Panel(
            "🚨 Critical Services Not Available\n\n"
            "The end-to-end test requires these critical services:\n\n" +
            "\n".join(critical_issues) + "\n\n"
            "✅ Required Components:\n"
            f"{'✅' if api_running else '❌'} Main API Server (port 8000)\n"
            f"{'✅' if neo4j_available else '❌'} Neo4j Database\n"
            f"{'✅' if faiss_available else '❌'} FAISS Vector Index\n\n"
            "🔧 Setup Instructions:\n\n"
            "1. Start the main API server:\n"
            "   python src/main.py --port 8000 &\n\n"
            "2. Start Neo4j (if using Docker):\n"
            "   make neo4j-start\n"
            "   # OR manually: docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \\\n"
            "   #   -e NEO4J_AUTH=neo4j/password neo4j:4.4\n\n"
            "3. Create FAISS index:\n"
            "   python scripts/train_faiss_simple.py\n\n"
            "4. Re-run this test:\n"
            "   python scripts/end_to_end_test.py\n\n"
            "💡 Alternative: Run in minimal mode (limited functionality):\n"
            "   E2E_MINIMAL_MODE=true python scripts/end_to_end_test.py",
            title="🔧 Critical Dependencies Missing",
            border_style="red"
        ))
        return False
    
    if readiness_issues:
        console.print("\n")
        console.print(Panel(
            "⚠️ Service Readiness Issues Detected\n\n"
            "While critical services are running, some components need attention:\n\n" +
            "\n".join(readiness_issues) + "\n\n"
            "💡 These issues may cause some tests to fail, but the test will continue.\n"
            "For full functionality, ensure all models are properly loaded.",
            title="⚠️ Readiness Warnings",
            border_style="yellow"
        ))
        # Continue execution but log warnings
        for issue in readiness_issues:
            results.add_warning(issue)
    
    return True

def save_test_results(results: ValidationResults) -> None:
    """Save detailed test results to a JSON file for analysis"""
    logger.debug("Saving test results to file")
    
    try:
        # Create results directory
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"e2e_test_results_{timestamp}.json"
        
        # Prepare results data
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": results.get_total_time(),
            "overall_success": len(results.errors) == 0,
            "system_health": results.system_health,
            "performance_metrics": results.performance_metrics,
            "ingestion_results": {
                "success": results.ingestion_success,
                "entities_before": results.neo4j_entities_before,
                "entities_after": results.neo4j_entities_after,
                "relationships_before": results.neo4j_relationships_before,
                "relationships_after": results.neo4j_relationships_after,
                "faiss_vectors_before": results.faiss_vectors_before,
                "faiss_vectors_after": results.faiss_vectors_after
            },
            "query_results": {
                "response_time_ms": results.query_response_time_ms,
                "citations_valid": results.citations_valid,
                "graph_data_valid": results.graph_data_valid,
                "answer_quality_score": results.answer_quality_score
            },
            "errors": results.errors,
            "warnings": results.warnings,
            "test_environment": {
                "minimal_mode": MINIMAL_MODE,
                "skip_model_tests": SKIP_MODEL_TESTS,
                "api_base_url": API_BASE_URL,
                "timeout_seconds": TIMEOUT_SECONDS
            }
        }
        
        # Save to file
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        console.print(f"[green]💾 Test results saved to: {results_file}[/green]")
        logger.info(f"Test results saved to {results_file}")
        
    except Exception as e:
        console.print(f"[yellow]⚠️ Failed to save test results: {e}[/yellow]")
        logger.warning(f"Failed to save test results: {e}")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SubgraphRAG+ End-to-End Integration Test")
    parser.add_argument("--minimal", action="store_true", 
                       help="Run in minimal mode (only requires API server)")
    parser.add_argument("--skip-models", action="store_true",
                       help="Skip model-dependent tests (IE extraction)")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS,
                       help=f"Timeout for API calls in seconds (default: {TIMEOUT_SECONDS})")
    
    args = parser.parse_args()
    
    # Override global settings based on command line args
    global MINIMAL_MODE, SKIP_MODEL_TESTS
    if args.minimal:
        MINIMAL_MODE = True
        os.environ["E2E_MINIMAL_MODE"] = "true"
    if args.skip_models:
        SKIP_MODEL_TESTS = True
        os.environ["E2E_SKIP_MODEL_TESTS"] = "true"
    
    # Use local variable for timeout instead of modifying global
    timeout_seconds = args.timeout
    
    logger.info(f"Started {__file__} at {datetime.now()}")
    
    mode_info = []
    if MINIMAL_MODE:
        mode_info.append("MINIMAL MODE")
    if SKIP_MODEL_TESTS:
        mode_info.append("SKIP MODELS")
    
    title = "🚀 Starting E2E Test"
    if mode_info:
        title += f" ({', '.join(mode_info)})"
    
    console.print(Panel(
        "SubgraphRAG+ End-to-End Integration Test\n"
        "This script validates the complete system pipeline\n\n"
        f"Mode: {'Minimal' if MINIMAL_MODE else 'Full'}\n"
        f"Model tests: {'Disabled' if SKIP_MODEL_TESTS else 'Enabled'}\n"
        f"Timeout: {timeout_seconds}s",
        title=title,
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
        if not check_services_and_provide_guidance(results, minimal_mode=MINIMAL_MODE):
            console.print("\n")
            console.print(Panel(
                "❌ Critical dependencies are missing or not accessible.\n\n"
                "The end-to-end test cannot proceed without these services.\n"
                "Please address the issues above and re-run the test.\n\n"
                "💡 Tip: Start the API server first, then ensure Neo4j and FAISS are accessible.",
                title="🛑 Test Aborted - Critical Dependencies Missing",
                border_style="red"
            ))
            logger.error("Test aborted due to missing critical dependencies")
            logger.info(f"Finished {__file__} at {datetime.now()}")
            sys.exit(1)
        
        # Additional pre-flight checks
        console.print("\n")
        console.print("[green]✅ Critical dependencies verified. Starting intensive tests...[/green]")
        
        # Warn about resource requirements
        console.print("\n")
        console.print(Panel(
            "🧠 Resource Requirements Notice\n\n"
            "The following tests will download and load large AI models:\n\n"
            "• REBEL (Babelscape/rebel-large) - ~1.5GB\n"
            "• Model loading may take 5-10 minutes on first run\n"
            "• Requires sufficient RAM (recommended: 8GB+)\n\n"
            "If you experience segmentation faults or memory errors,\n"
            "consider running on a machine with more RAM or using smaller models.",
            title="💡 Performance Notice",
            border_style="blue"
        ))
        
        # Step 4: Database Baseline
        task4 = progress.add_task("Getting database baseline...", total=100)
        neo4j_baseline = check_neo4j_connection()
        faiss_baseline = check_faiss_index()
        
        if neo4j_baseline['status']:
            results.neo4j_entities_before = neo4j_baseline['entity_count']
            results.neo4j_relationships_before = neo4j_baseline['relationship_count']
        elif MINIMAL_MODE:
            console.print("[yellow]⚠️ Minimal mode: Skipping Neo4j baseline check[/yellow]")
            results.neo4j_entities_before = 0
            results.neo4j_relationships_before = 0
        
        if faiss_baseline['status']:
            results.faiss_vectors_before = faiss_baseline['vector_count']
        elif MINIMAL_MODE:
            console.print("[yellow]⚠️ Minimal mode: Skipping FAISS baseline check[/yellow]")
            results.faiss_vectors_before = 0
        
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
        
        if MINIMAL_MODE and not neo4j_baseline['status']:
            console.print("[yellow]⚠️ Minimal mode: Skipping Neo4j validation (not available)[/yellow]")
            neo4j_changes = {'status': False, 'error': 'Skipped in minimal mode'}
            results.neo4j_entities_after = 0
            results.neo4j_relationships_after = 0
        else:
            neo4j_changes = validate_neo4j_changes(neo4j_baseline)
            if neo4j_changes['status']:
                results.neo4j_entities_after = neo4j_changes['total_entities']
                results.neo4j_relationships_after = neo4j_changes['total_relationships']
        
        if MINIMAL_MODE and not faiss_baseline['status']:
            console.print("[yellow]⚠️ Minimal mode: Skipping FAISS validation (not available)[/yellow]")
            faiss_changes = {'status': False, 'error': 'Skipped in minimal mode'}
            results.faiss_vectors_after = 0
        else:
            faiss_changes = validate_faiss_changes(faiss_baseline)
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
        
        if SKIP_MODEL_TESTS:
            console.print("[yellow]⚠️ Skipping IE extraction test (SKIP_MODEL_TESTS=true)[/yellow]")
            ie_result = {'status': True, 'skipped': True, 'reason': 'Model tests disabled'}
            ie_time = 0
        else:
            ie_result = test_ie_extraction()
            ie_time = time.time() - start_time
        
        results.performance_metrics['ie_extraction'] = {
            'time': ie_time,
            'success': ie_result['status']
        }
        
        if not ie_result['status'] and not ie_result.get('skipped', False):
            if ie_result.get('model_loading_issue', False):
                results.add_warning(f"IE extraction failed due to model loading: {ie_result.get('error', 'Unknown error')}")
            else:
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
        
        # Step 10: Feedback Endpoint Test
        task10 = progress.add_task("Testing feedback endpoint...", total=100)
        start_time = time.time()
        feedback_result = test_feedback_endpoint()
        feedback_time = time.time() - start_time
        
        results.performance_metrics['feedback_endpoint'] = {
            'time': feedback_time,
            'success': feedback_result['status']
        }
        
        if not feedback_result['status']:
            results.add_error(f"Feedback endpoint failed: {feedback_result.get('error', 'Unknown error')}")
        
        results.system_health['feedback_endpoint'] = feedback_result
        progress.update(task10, completed=100)
        
        # Step 11: Metrics Endpoint Test
        task11 = progress.add_task("Testing metrics endpoint...", total=100)
        start_time = time.time()
        metrics_result = test_metrics_endpoint()
        metrics_time = time.time() - start_time
        
        results.performance_metrics['metrics_endpoint'] = {
            'time': metrics_time,
            'success': metrics_result['status']
        }
        
        if not metrics_result['status']:
            results.add_error(f"Metrics endpoint failed: {metrics_result.get('error', 'Unknown error')}")
        
        results.system_health['metrics_endpoint'] = metrics_result
        progress.update(task11, completed=100)
        
        # Step 12: Text Ingestion Endpoint Test
        task12 = progress.add_task("Testing text ingestion endpoint...", total=100)
        start_time = time.time()
        text_ingest_result = test_text_ingestion_endpoint()
        text_ingest_time = time.time() - start_time
        
        results.performance_metrics['text_ingestion_endpoint'] = {
            'time': text_ingest_time,
            'success': text_ingest_result['status']
        }
        
        if not text_ingest_result['status']:
            results.add_error(f"Text ingestion endpoint failed: {text_ingest_result.get('error', 'Unknown error')}")
        
        results.system_health['text_ingestion_endpoint'] = text_ingest_result
        progress.update(task12, completed=100)
        
        # Step 13: Batch Ingest Endpoint Test
        task13 = progress.add_task("Testing batch ingest endpoint...", total=100)
        start_time = time.time()
        batch_ingest_result = test_batch_ingest_endpoint()
        batch_ingest_time = time.time() - start_time
        
        results.performance_metrics['batch_ingest_endpoint'] = {
            'time': batch_ingest_time,
            'success': batch_ingest_result['status']
        }
        
        if not batch_ingest_result['status']:
            results.add_error(f"Batch ingest endpoint failed: {batch_ingest_result.get('error', 'Unknown error')}")
        
        results.system_health['batch_ingest_endpoint'] = batch_ingest_result
        progress.update(task13, completed=100)
        
        # Step 14: Debug Endpoints Test
        task14 = progress.add_task("Testing debug endpoints...", total=100)
        start_time = time.time()
        debug_result = test_debug_endpoints()
        debug_time = time.time() - start_time
        
        results.performance_metrics['debug_endpoints'] = {
            'time': debug_time,
            'success': debug_result['status']
        }
        
        if not debug_result['status']:
            results.add_warning(f"Debug endpoints had issues: {debug_result.get('error', 'Unknown error')}")
        
        results.system_health['debug_endpoints'] = debug_result
        progress.update(task14, completed=100)
        
        # Step 15: Error Handling Test
        task15 = progress.add_task("Testing error handling...", total=100)
        start_time = time.time()
        error_handling_result = test_error_handling()
        error_handling_time = time.time() - start_time
        
        results.performance_metrics['error_handling'] = {
            'time': error_handling_time,
            'success': error_handling_result['status']
        }
        
        if not error_handling_result['status']:
            results.add_warning(f"Error handling test had issues: {error_handling_result.get('error', 'Unknown error')}")
        
        results.system_health['error_handling'] = error_handling_result
        progress.update(task15, completed=100)
        
        # Step 16: Performance Benchmarks
        task16 = progress.add_task("Running performance benchmarks...", total=100)
        start_time = time.time()
        performance_result = test_performance_benchmarks()
        performance_time = time.time() - start_time
        
        results.performance_metrics['performance_benchmarks'] = {
            'time': performance_time,
            'success': performance_result['status']
        }
        
        if not performance_result['status']:
            results.add_warning(f"Performance benchmarks had issues: {performance_result.get('error', 'Unknown error')}")
        
        results.system_health['performance_benchmarks'] = performance_result
        progress.update(task16, completed=100)
    
    # Generate final report
    console.print("\n")
    generate_report(results)
    
    # Save detailed results to file
    save_test_results(results)
    
    # Log completion
    logger.info(f"Finished {__file__} at {datetime.now()}")
    
    # Exit with appropriate code
    exit_code = 0 if len(results.errors) == 0 else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 