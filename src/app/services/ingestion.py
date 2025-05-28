"""
Ingestion Service
Handles text processing, chunking, entity typing, and data pipeline logic
Coordinates between information extraction and database staging
"""

import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# RULE:import-rich-logger-correctly
from ..log import logger
from ..database import sqlite_db
from ..entity_typing import get_entity_type
from .information_extraction import get_information_extraction_service

# RULE:uppercase-constants-top
DEFAULT_CHUNK_SIZE = 1000
MAX_CHUNK_SIZE = 5000
MIN_CHUNK_SIZE = 100

@dataclass
class IngestionResult:
    """Result from ingestion processing"""
    total_triples: int
    successful_triples: int
    failed_triples: int
    processing_time: float
    errors: List[str]
    warnings: List[str]

@dataclass
class ChunkProcessingResult:
    """Result from processing a single text chunk"""
    chunk_index: int
    triples_extracted: int
    triples_staged: int
    processing_time: float
    errors: List[str]

class IngestionService:
    """
    Service for ingesting and processing text into knowledge graph
    Handles text chunking, information extraction, entity typing, and staging
    """
    
    def __init__(self):
        """Initialize the Ingestion Service"""
        # RULE:debug-trace-every-step
        logger.debug("Starting IngestionService initialization")
        
        self.ie_service = get_information_extraction_service()
        
        logger.debug("Finished IngestionService initialization")
    
    async def process_text_file(self, file_path: str, chunk_size: int = 2000) -> IngestionResult:
        """
        Process a text file by extracting triples from chunks
        
        Args:
            file_path: Path to text file
            chunk_size: Size of text chunks to process
            
        Returns:
            IngestionResult with processing statistics
        """
        logger.info(f"Processing text file: {file_path}")
        logger.debug(f"Starting text file processing with chunk size {chunk_size}")
        
        start_time = time.time()
        errors = []
        warnings = []
        total_triples = 0
        successful_triples = 0
        failed_triples = 0
        
        try:
            # Read file content
            content = self._read_file_content(file_path)
            if content is None:
                errors.append(f"Failed to read file {file_path}")
                return IngestionResult(
                    total_triples=0,
                    successful_triples=0,
                    failed_triples=0,
                    processing_time=time.time() - start_time,
                    errors=errors,
                    warnings=warnings
                )
            
            # Split content into chunks
            chunks = self._split_text_into_chunks(content, chunk_size)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                chunk_result = await self._process_text_chunk(
                    chunk, 
                    chunk_index=i,
                    source=f"file:{Path(file_path).name}"
                )
                
                total_triples += chunk_result.triples_extracted
                successful_triples += chunk_result.triples_staged
                failed_triples += (chunk_result.triples_extracted - chunk_result.triples_staged)
                errors.extend(chunk_result.errors)
                
                # Small delay to avoid overwhelming the system
                time.sleep(0.1)
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {file_path}: extracted {total_triples} triples, staged {successful_triples}")
            
            return IngestionResult(
                total_triples=total_triples,
                successful_triples=successful_triples,
                failed_triples=failed_triples,
                processing_time=processing_time,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            # RULE:rich-error-handling-required
            logger.error(f"Error processing file {file_path}: {e}")
            errors.append(f"File processing error: {e}")
            
            return IngestionResult(
                total_triples=0,
                successful_triples=0,
                failed_triples=0,
                processing_time=time.time() - start_time,
                errors=errors,
                warnings=warnings
            )
    
    async def process_text_content(self, text: str, source: str = "direct_input") -> IngestionResult:
        """
        Process raw text content directly
        
        Args:
            text: Raw text content to process
            source: Source identifier for the content
            
        Returns:
            IngestionResult with processing statistics
        """
        logger.debug(f"Processing text content of length {len(text)}")
        
        start_time = time.time()
        
        chunk_result = await self._process_text_chunk(text, chunk_index=0, source=source)
        
        return IngestionResult(
            total_triples=chunk_result.triples_extracted,
            successful_triples=chunk_result.triples_staged,
            failed_triples=chunk_result.triples_extracted - chunk_result.triples_staged,
            processing_time=time.time() - start_time,
            errors=chunk_result.errors,
            warnings=[]
        )
    
    def stage_triples_batch(self, triples: List[Dict[str, Any]], source: str = "batch_input") -> IngestionResult:
        """
        Stage a batch of pre-extracted triples
        
        Args:
            triples: List of triple dictionaries
            source: Source identifier for the triples
            
        Returns:
            IngestionResult with staging statistics
        """
        logger.debug(f"Staging batch of {len(triples)} triples")
        
        start_time = time.time()
        staged_count = 0
        errors = []
        
        for i, triple in enumerate(triples):
            try:
                if self._stage_single_triple(triple, source):
                    staged_count += 1
            except Exception as e:
                # RULE:rich-error-handling-required
                logger.error(f"Error staging triple {i}: {e}")
                errors.append(f"Triple {i}: {e}")
        
        processing_time = time.time() - start_time
        logger.info(f"Staged {staged_count}/{len(triples)} triples from batch")
        
        return IngestionResult(
            total_triples=len(triples),
            successful_triples=staged_count,
            failed_triples=len(triples) - staged_count,
            processing_time=processing_time,
            errors=errors,
            warnings=[]
        )
    
    async def _process_text_chunk(self, text: str, chunk_index: int, source: str) -> ChunkProcessingResult:
        """
        Process a single text chunk through the full pipeline
        
        Args:
            text: Text chunk to process
            chunk_index: Index of the chunk for logging
            source: Source identifier
            
        Returns:
            ChunkProcessingResult with processing statistics
        """
        logger.debug(f"Processing chunk {chunk_index} with {len(text)} characters")
        
        start_time = time.time()
        errors = []
        
        # Extract triples using IE service
        extraction_result = await self.ie_service.extract_triples(text)
        
        if not extraction_result.success:
            errors.append(f"IE extraction failed: {extraction_result.error_message}")
            return ChunkProcessingResult(
                chunk_index=chunk_index,
                triples_extracted=0,
                triples_staged=0,
                processing_time=time.time() - start_time,
                errors=errors
            )
        
        # Stage extracted triples
        staged_count = 0
        for triple in extraction_result.triples:
            try:
                if self._stage_single_triple(triple, source):
                    staged_count += 1
            except Exception as e:
                errors.append(f"Staging error: {e}")
        
        processing_time = time.time() - start_time
        logger.debug(f"Chunk {chunk_index}: extracted {len(extraction_result.triples)}, staged {staged_count}")
        
        return ChunkProcessingResult(
            chunk_index=chunk_index,
            triples_extracted=len(extraction_result.triples),
            triples_staged=staged_count,
            processing_time=processing_time,
            errors=errors
        )
    
    def _stage_single_triple(self, triple: Dict[str, Any], source: str) -> bool:
        """
        Stage a single triple with entity typing and validation
        
        Args:
            triple: Triple dictionary
            source: Source identifier
            
        Returns:
            bool: True if staging succeeded, False otherwise
        """
        try:
            # Validate triple structure
            if not all(key in triple for key in ['head', 'relation', 'tail']):
                logger.warning(f"Skipping malformed triple: {triple}")
                return False
            
            # Clean and validate triple components
            head = str(triple['head']).strip()
            relation = str(triple['relation']).strip()
            tail = str(triple['tail']).strip()
            
            if not head or not relation or not tail:
                logger.warning(f"Skipping empty triple components: {triple}")
                return False
            
            # Get entity types using schema-driven approach
            head_type = get_entity_type(head)
            tail_type = get_entity_type(tail)
            
            # Stage the triple
            sqlite_db.execute(
                "INSERT INTO staging_triples (h_text, r_text, t_text, status, source, metadata) VALUES (?, ?, ?, 'pending', ?, ?)",
                (
                    head,
                    relation,
                    tail,
                    source,
                    {
                        "head_type": head_type,
                        "tail_type": tail_type,
                        "confidence": triple.get('confidence', 1.0),
                        "extraction_method": "rebel_ie"
                    }
                )
            )
            return True
            
        except Exception as e:
            if "UNIQUE constraint failed" in str(e):
                logger.debug(f"Duplicate triple: {triple}")
                return True  # Count duplicates as successful
            else:
                # RULE:rich-error-handling-required
                logger.error(f"Error staging triple {triple}: {e}")
                raise e
    
    def _read_file_content(self, file_path: str) -> Optional[str]:
        """
        Read content from a file with error handling
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: File content or None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            # RULE:rich-error-handling-required
            logger.error(f"Failed to read file {file_path}: {e}")
            return None
    
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
        
        # Validate chunk size
        if chunk_size < MIN_CHUNK_SIZE:
            chunk_size = MIN_CHUNK_SIZE
            logger.warning(f"Chunk size too small, using minimum: {MIN_CHUNK_SIZE}")
        elif chunk_size > MAX_CHUNK_SIZE:
            chunk_size = MAX_CHUNK_SIZE
            logger.warning(f"Chunk size too large, using maximum: {MAX_CHUNK_SIZE}")
        
        # Split by sentences first for better coherence
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


# Global service instance for singleton pattern
_ingestion_service_instance = None

def get_ingestion_service() -> IngestionService:
    """
    Get the global Ingestion Service instance
    Implements singleton pattern for consistency
    """
    global _ingestion_service_instance
    if _ingestion_service_instance is None:
        _ingestion_service_instance = IngestionService()
    return _ingestion_service_instance 