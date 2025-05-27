#!/usr/bin/env python3
"""
Enhanced Ingestion Script with Information Extraction
Uses the new services architecture for proper separation of concerns
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# RULE:import-rich-logger-correctly
from src.app.log import logger
from app.services.ingestion import get_ingestion_service

# RULE:uppercase-constants-top
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_API_URL = "http://localhost:8000"

class IEIngestPipeline:
    """
    Information Extraction + Ingestion Pipeline
    Uses the new services architecture for better separation of concerns
    """
    
    def __init__(self, api_url: str = DEFAULT_API_URL, api_key: str = None):
        """
        Initialize the IE ingestion pipeline
        
        Args:
            api_url: URL of the unified SubgraphRAG+ API (for fallback)
            api_key: API key for authentication (for fallback)
        """
        # RULE:debug-trace-every-step
        logger.debug("Starting IEIngestPipeline initialization")
        
        # Use services architecture for primary functionality
        self.ingestion_service = get_ingestion_service()
        
        # Keep API details for fallback/compatibility
        self.api_url = api_url
        self.api_key = api_key
        
        logger.debug("Finished IEIngestPipeline initialization")
    
    # Legacy methods for compatibility
    def extract_triples_from_text(self, text: str, max_length: int = 256) -> List[Dict[str, Any]]:
        """Extract triples from raw text (legacy compatibility method)"""
        result = self.ingestion_service.ie_service.extract_triples(text, max_length)
        return result.triples if result.success else []
    
    def stage_triples(self, triples: List[Dict[str, Any]], source: str = "ie_extraction") -> int:
        """Stage extracted triples for ingestion (legacy compatibility method)"""
        result = self.ingestion_service.stage_triples_batch(triples, source)
        return result.successful_triples
    
    def process_text_file(self, file_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Process a text file by extracting triples from chunks
        
        Args:
            file_path: Path to text file
            chunk_size: Size of text chunks to process
        """
        logger.info(f"Processing text file: {file_path}")
        logger.debug(f"Starting text file processing with chunk size {chunk_size}")
        
        # Delegate to ingestion service
        result = self.ingestion_service.process_text_file(file_path, chunk_size)
        
        # Log results
        logger.info(f"Processed {file_path}: extracted {result.total_triples} triples, staged {result.successful_triples}")
        
        if result.errors:
            logger.warning(f"Encountered {len(result.errors)} errors during processing:")
            for error in result.errors[:5]:  # Log first 5 errors
                logger.warning(f"  - {error}")
        
        if result.warnings:
            logger.warning(f"Encountered {len(result.warnings)} warnings during processing:")
            for warning in result.warnings[:5]:  # Log first 5 warnings
                logger.warning(f"  - {warning}")
        
        return result
    
    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into overlapping chunks for better relation extraction
        
        Args:
            text: Text to split
            chunk_size: Target chunk size in characters
            
        Returns:
            List of text chunks
        """
        logger.debug(f"Starting text splitting with chunk size {chunk_size}")
        
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
        
        logger.debug(f"Finished text splitting: {len(chunks)} chunks created")
        return chunks
    
    def process_json_triples(self, file_path: str):
        """
        Process pre-extracted triples from JSON file
        
        Args:
            file_path: Path to JSON file with triples
        """
        logger.info(f"Processing JSON triples file: {file_path}")
        logger.debug("Starting JSON triples processing")
        
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
            
            # Delegate to ingestion service
            result = self.ingestion_service.stage_triples_batch(
                triples=triples, 
                source=f"json:{Path(file_path).name}"
            )
            
            logger.info(f"Processed {file_path}: staged {result.successful_triples} triples")
            
            if result.errors:
                logger.warning(f"Encountered {len(result.errors)} errors during JSON processing")
            
            return result
            
        except Exception as e:
            # RULE:rich-error-handling-required
            logger.error(f"Failed to process JSON file {file_path}: {e}")
            return None
        
        logger.debug("Finished JSON triples processing")


def main():
    """Main entry point"""
    # RULE:every-src-script-must-log
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Started {__file__} at {timestamp}")
    
    parser = argparse.ArgumentParser(description="Ingest text with IE extraction via unified API")
    parser.add_argument("input", help="Input file path (text or JSON)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, 
                       help="Unified API URL")
    parser.add_argument("--api-key", 
                       help="API key for authentication")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                       help="Text chunk size for processing")
    parser.add_argument("--format", choices=['text', 'json'], default='auto',
                       help="Input format (auto-detected by default)")
    
    args = parser.parse_args()
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.getenv("API_KEY_SECRET")
    if not api_key:
        logger.warning("No API key provided. Some endpoints may require authentication.")
    
    # Initialize pipeline
    pipeline = IEIngestPipeline(api_url=args.api_url, api_key=api_key)
    
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