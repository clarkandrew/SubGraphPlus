#!/usr/bin/env python3
"""
Enhanced Ingestion Script with Information Extraction
Integrates REBEL IE service for proper triple extraction from raw text
"""

import os
import sys
import json
import logging
import argparse
import requests
import time
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.database import sqlite_db
from app.entity_typing import get_entity_type

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'ingest_with_ie.log'))
    ]
)
logger = logging.getLogger(__name__)

class IEIngestPipeline:
    """
    Information Extraction + Ingestion Pipeline
    """
    
    def __init__(self, ie_service_url: str = "http://localhost:8003"):
        """
        Initialize the IE ingestion pipeline
        
        Args:
            ie_service_url: URL of the REBEL IE service
        """
        self.ie_service_url = ie_service_url
        self.session = requests.Session()
        
        # Test IE service connection
        self._test_ie_service()
    
    def _test_ie_service(self):
        """Test connection to IE service"""
        try:
            response = self.session.get(f"{self.ie_service_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"IE service healthy: {health_data}")
            else:
                logger.warning(f"IE service health check failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to IE service at {self.ie_service_url}: {e}")
            logger.error("Make sure the IE service is running: uvicorn src.app.ie_service:app --host 0.0.0.0 --port 8003")
            sys.exit(1)
    
    def extract_triples_from_text(self, text: str, max_length: int = 256) -> List[Dict[str, Any]]:
        """
        Extract triples from raw text using REBEL IE service
        
        Args:
            text: Raw text to extract triples from
            max_length: Maximum sequence length for REBEL
            
        Returns:
            List of extracted triples with metadata
        """
        try:
            payload = {
                "text": text,
                "max_length": max_length,
                "num_beams": 3
            }
            
            response = self.session.post(
                f"{self.ie_service_url}/extract",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Extracted {len(result['triples'])} triples from text (took {result['processing_time']:.2f}s)")
                return result['triples']
            else:
                logger.error(f"IE service error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error calling IE service: {e}")
            return []
    
    def stage_triples(self, triples: List[Dict[str, Any]], source: str = "ie_extraction"):
        """
        Stage extracted triples for ingestion
        
        Args:
            triples: List of triple dictionaries
            source: Source identifier for the triples
        """
        staged_count = 0
        
        for triple in triples:
            try:
                # Validate triple structure
                if not all(key in triple for key in ['head', 'relation', 'tail']):
                    logger.warning(f"Skipping malformed triple: {triple}")
                    continue
                
                # Clean and validate triple components
                head = triple['head'].strip()
                relation = triple['relation'].strip()
                tail = triple['tail'].strip()
                
                if not head or not relation or not tail:
                    logger.warning(f"Skipping empty triple components: {triple}")
                    continue
                
                # Get entity types using schema-driven approach
                head_type = get_entity_type(head, context=f"Subject of {relation}")
                tail_type = get_entity_type(tail, context=f"Object of {relation}")
                
                # Stage the triple
                sqlite_db.stage_triple(
                    head_text=head,
                    relation_text=relation,
                    tail_text=tail,
                    source=source,
                    metadata={
                        "head_type": head_type,
                        "tail_type": tail_type,
                        "confidence": triple.get('confidence', 1.0),
                        "extraction_method": "rebel_ie"
                    }
                )
                staged_count += 1
                
            except Exception as e:
                logger.error(f"Error staging triple {triple}: {e}")
        
        logger.info(f"Staged {staged_count} triples from IE extraction")
        return staged_count
    
    def process_text_file(self, file_path: str, chunk_size: int = 1000):
        """
        Process a text file by extracting triples from chunks
        
        Args:
            file_path: Path to text file
            chunk_size: Size of text chunks to process
        """
        logger.info(f"Processing text file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return
        
        # Split content into chunks
        chunks = self._split_text_into_chunks(content, chunk_size)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        total_triples = 0
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Extract triples from chunk
            triples = self.extract_triples_from_text(chunk)
            
            if triples:
                # Stage triples
                count = self.stage_triples(triples, source=f"file:{Path(file_path).name}")
                total_triples += count
            
            # Small delay to avoid overwhelming the IE service
            time.sleep(0.1)
        
        logger.info(f"Processed {file_path}: extracted and staged {total_triples} total triples")
    
    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into overlapping chunks for better relation extraction
        
        Args:
            text: Text to split
            chunk_size: Target chunk size in characters
            
        Returns:
            List of text chunks
        """
        # Split by sentences first
        sentences = text.replace('\n', ' ').split('. ')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, start new chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_json_triples(self, file_path: str):
        """
        Process pre-extracted triples from JSON file
        
        Args:
            file_path: Path to JSON file with triples
        """
        logger.info(f"Processing JSON triples file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                triples = data
            elif isinstance(data, dict) and 'triples' in data:
                triples = data['triples']
            else:
                logger.error(f"Unsupported JSON format in {file_path}")
                return
            
            count = self.stage_triples(triples, source=f"json:{Path(file_path).name}")
            logger.info(f"Processed {file_path}: staged {count} triples")
            
        except Exception as e:
            logger.error(f"Failed to process JSON file {file_path}: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Ingest text with IE extraction")
    parser.add_argument("input", help="Input file path (text or JSON)")
    parser.add_argument("--ie-url", default="http://localhost:8003", 
                       help="IE service URL")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Text chunk size for processing")
    parser.add_argument("--format", choices=['text', 'json'], default='auto',
                       help="Input format (auto-detected by default)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = IEIngestPipeline(ie_service_url=args.ie_url)
    
    # Determine input format
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    if args.format == 'auto':
        format_type = 'json' if input_path.suffix.lower() == '.json' else 'text'
    else:
        format_type = args.format
    
    # Process input
    if format_type == 'json':
        pipeline.process_json_triples(args.input)
    else:
        pipeline.process_text_file(args.input, chunk_size=args.chunk_size)
    
    logger.info("Ingestion complete. Run 'python scripts/ingest_worker.py --process-all' to process staged triples.")


if __name__ == "__main__":
    main() 