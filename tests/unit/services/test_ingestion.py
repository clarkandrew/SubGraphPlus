"""
Unit tests for Ingestion Service

Tests the ingestion pipeline including text processing, triple staging,
and batch operations without external dependencies.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from typing import List, Dict, Any

from src.app.services.ingestion import (
    IngestionService,
    get_ingestion_service,
    IngestionResult,
    ChunkProcessingResult,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    DEFAULT_CHUNK_SIZE
)


class TestIngestionService:
    """Test the IngestionService class"""
    
    def test_service_initialization_creates_instance(self):
        """Test that service initializes correctly"""
        # Arrange & Act
        service = IngestionService()
        
        # Assert
        assert service is not None
        assert hasattr(service, 'process_text_content')
        assert hasattr(service, 'process_text_file')
        assert hasattr(service, 'stage_triples_batch')
        assert service.ie_service is not None
    
    def test_singleton_pattern_returns_same_instance(self):
        """Test that get_ingestion_service returns singleton"""
        # Arrange & Act
        service1 = get_ingestion_service()
        service2 = get_ingestion_service()
        
        # Assert
        assert service1 is service2


class TestTextChunking:
    """Test text chunking functionality"""
    
    def test_split_text_into_chunks_respects_chunk_size(self):
        """Test that text is split according to chunk size"""
        # Arrange
        service = IngestionService()
        long_text = "This is a sentence. " * 100  # Create long text
        chunk_size = 200
        
        # Act
        chunks = service._split_text_into_chunks(long_text, chunk_size)
        
        # Assert
        assert len(chunks) > 1  # Should be split into multiple chunks
        for chunk in chunks:
            assert len(chunk) <= chunk_size + 50  # Allow some flexibility for sentence boundaries
    
    def test_split_text_into_chunks_handles_small_chunk_size(self):
        """Test handling of chunk size below minimum"""
        # Arrange
        service = IngestionService()
        text = "This is a test sentence."
        small_chunk_size = 10  # Below MIN_CHUNK_SIZE
        
        # Act
        chunks = service._split_text_into_chunks(text, small_chunk_size)
        
        # Assert
        assert len(chunks) >= 1
        # Should use minimum chunk size, not the requested small size
    
    def test_split_text_into_chunks_handles_large_chunk_size(self):
        """Test handling of chunk size above maximum"""
        # Arrange
        service = IngestionService()
        text = "This is a test sentence."
        large_chunk_size = 10000  # Above MAX_CHUNK_SIZE
        
        # Act
        chunks = service._split_text_into_chunks(text, large_chunk_size)
        
        # Assert
        assert len(chunks) == 1  # Should fit in one chunk
        assert len(chunks[0]) == len(text)
    
    def test_split_text_preserves_sentence_boundaries(self):
        """Test that chunking preserves sentence boundaries when possible"""
        # Arrange
        service = IngestionService()
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunk_size = 30  # Should split between sentences
        
        # Act
        chunks = service._split_text_into_chunks(text, chunk_size)
        
        # Assert
        assert len(chunks) > 1
        # Each chunk should end with a complete sentence (or be the last chunk)
        for i, chunk in enumerate(chunks[:-1]):  # All but last chunk
            assert chunk.strip().endswith('.')
    
    def test_split_text_handles_empty_input(self):
        """Test handling of empty text input"""
        # Arrange
        service = IngestionService()
        empty_text = ""
        
        # Act
        chunks = service._split_text_into_chunks(empty_text, DEFAULT_CHUNK_SIZE)
        
        # Assert
        assert len(chunks) == 1
        assert chunks[0] == ""


class TestFileProcessing:
    """Test file processing functionality"""
    
    @patch('builtins.open', new_callable=mock_open, read_data="Test file content.")
    @patch.object(IngestionService, '_process_text_chunk')
    def test_process_text_file_reads_and_processes_content(self, mock_process_chunk, mock_file):
        """Test that file processing reads content and processes chunks"""
        # Arrange
        service = IngestionService()
        mock_process_chunk.return_value = ChunkProcessingResult(
            chunk_index=0,
            triples_extracted=2,
            triples_staged=2,
            processing_time=0.1,
            errors=[]
        )
        
        # Act
        result = service.process_text_file("test_file.txt")
        
        # Assert
        mock_file.assert_called_once_with("test_file.txt", 'r', encoding='utf-8')
        mock_process_chunk.assert_called_once()
        assert isinstance(result, IngestionResult)
        assert result.successful_triples == 2
        assert result.total_triples == 2
    
    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_process_text_file_handles_missing_file(self, mock_file):
        """Test handling of missing file"""
        # Arrange
        service = IngestionService()
        
        # Act
        result = service.process_text_file("nonexistent_file.txt")
        
        # Assert
        assert isinstance(result, IngestionResult)
        assert result.total_triples == 0
        assert result.successful_triples == 0
        assert len(result.errors) > 0
        assert "Failed to read file" in result.errors[0]
    
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_process_text_file_handles_permission_error(self, mock_file):
        """Test handling of file permission errors"""
        # Arrange
        service = IngestionService()
        
        # Act
        result = service.process_text_file("protected_file.txt")
        
        # Assert
        assert isinstance(result, IngestionResult)
        assert result.total_triples == 0
        assert len(result.errors) > 0


class TestTextContentProcessing:
    """Test direct text content processing"""
    
    @patch.object(IngestionService, '_process_text_chunk')
    def test_process_text_content_calls_chunk_processor(self, mock_process_chunk):
        """Test that text content processing delegates to chunk processor"""
        # Arrange
        service = IngestionService()
        mock_process_chunk.return_value = ChunkProcessingResult(
            chunk_index=0,
            triples_extracted=3,
            triples_staged=2,
            processing_time=0.2,
            errors=["One error"]
        )
        
        # Act
        result = service.process_text_content("Test text content")
        
        # Assert
        mock_process_chunk.assert_called_once_with("Test text content", chunk_index=0, source="direct_input")
        assert isinstance(result, IngestionResult)
        assert result.total_triples == 3
        assert result.successful_triples == 2
        assert result.failed_triples == 1
        assert len(result.errors) == 1
    
    def test_process_text_content_with_custom_source(self):
        """Test text processing with custom source identifier"""
        # Arrange
        service = IngestionService()
        
        with patch.object(service, '_process_text_chunk') as mock_process:
            mock_process.return_value = ChunkProcessingResult(
                chunk_index=0,
                triples_extracted=1,
                triples_staged=1,
                processing_time=0.1,
                errors=[]
            )
            
            # Act
            result = service.process_text_content("Test text", source="custom_source")
            
            # Assert
            mock_process.assert_called_once_with("Test text", chunk_index=0, source="custom_source")


class TestChunkProcessing:
    """Test individual chunk processing"""
    
    @patch('src.app.services.ingestion.get_entity_type')
    @patch('src.app.services.ingestion.sqlite_db')
    def test_process_text_chunk_extracts_and_stages_triples(self, mock_db, mock_get_type):
        """Test that chunk processing extracts triples and stages them"""
        # Arrange
        service = IngestionService()
        mock_get_type.return_value = "ENTITY"
        mock_db.execute.return_value = None
        
        # Mock IE service
        mock_extraction_result = Mock()
        mock_extraction_result.success = True
        mock_extraction_result.triples = [
            {"head": "Obama", "relation": "born in", "tail": "Hawaii", "confidence": 0.9}
        ]
        service.ie_service.extract_triples = Mock(return_value=mock_extraction_result)
        
        # Act
        result = service._process_text_chunk("Obama was born in Hawaii", 0, "test_source")
        
        # Assert
        assert isinstance(result, ChunkProcessingResult)
        assert result.triples_extracted == 1
        assert result.triples_staged == 1
        assert result.chunk_index == 0
        service.ie_service.extract_triples.assert_called_once()
        mock_db.execute.assert_called_once()
    
    def test_process_text_chunk_handles_ie_failure(self):
        """Test handling of IE extraction failure"""
        # Arrange
        service = IngestionService()
        
        # Mock IE service failure
        mock_extraction_result = Mock()
        mock_extraction_result.success = False
        mock_extraction_result.error_message = "Model failed"
        service.ie_service.extract_triples = Mock(return_value=mock_extraction_result)
        
        # Act
        result = service._process_text_chunk("Test text", 0, "test_source")
        
        # Assert
        assert isinstance(result, ChunkProcessingResult)
        assert result.triples_extracted == 0
        assert result.triples_staged == 0
        assert len(result.errors) > 0
        assert "IE extraction failed" in result.errors[0]


class TestTripleStaging:
    """Test triple staging functionality"""
    
    @patch('src.app.services.ingestion.get_entity_type')
    @patch('src.app.services.ingestion.sqlite_db')
    def test_stage_single_triple_inserts_to_database(self, mock_db, mock_get_type):
        """Test that single triple staging inserts to database"""
        # Arrange
        service = IngestionService()
        mock_get_type.return_value = "PERSON"
        mock_db.execute.return_value = None
        
        triple = {
            "head": "Barack Obama",
            "relation": "born in",
            "tail": "Hawaii",
            "confidence": 0.95
        }
        
        # Act
        result = service._stage_single_triple(triple, "test_source")
        
        # Assert
        assert result is True
        mock_db.execute.assert_called_once()
        # Verify the SQL call structure
        call_args = mock_db.execute.call_args
        assert "INSERT INTO staging_triples" in call_args[0][0]
        assert call_args[0][1] == ("Barack Obama", "born in", "Hawaii", "test_source", {
            "head_type": "PERSON",
            "tail_type": "PERSON", 
            "confidence": 0.95,
            "extraction_method": "rebel_ie"
        })
    
    @patch('src.app.services.ingestion.get_entity_type')
    @patch('src.app.services.ingestion.sqlite_db')
    def test_stage_single_triple_handles_duplicate(self, mock_db, mock_get_type):
        """Test handling of duplicate triple insertion"""
        # Arrange
        service = IngestionService()
        mock_get_type.return_value = "ENTITY"
        mock_db.execute.side_effect = Exception("UNIQUE constraint failed")
        
        triple = {"head": "A", "relation": "rel", "tail": "B"}
        
        # Act
        result = service._stage_single_triple(triple, "test_source")
        
        # Assert
        assert result is True  # Duplicates are considered successful
    
    def test_stage_single_triple_validates_structure(self):
        """Test validation of triple structure"""
        # Arrange
        service = IngestionService()
        
        # Test missing keys
        invalid_triple = {"head": "A", "relation": "rel"}  # Missing tail
        
        # Act
        result = service._stage_single_triple(invalid_triple, "test_source")
        
        # Assert
        assert result is False
    
    def test_stage_single_triple_validates_empty_values(self):
        """Test validation of empty triple values"""
        # Arrange
        service = IngestionService()
        
        # Test empty values
        empty_triple = {"head": "", "relation": "rel", "tail": "B"}
        
        # Act
        result = service._stage_single_triple(empty_triple, "test_source")
        
        # Assert
        assert result is False


class TestBatchTripleStaging:
    """Test batch triple staging"""
    
    @patch.object(IngestionService, '_stage_single_triple')
    def test_stage_triples_batch_processes_all_triples(self, mock_stage_single):
        """Test that batch staging processes all provided triples"""
        # Arrange
        service = IngestionService()
        mock_stage_single.return_value = True
        
        triples = [
            {"head": "A", "relation": "rel1", "tail": "B"},
            {"head": "C", "relation": "rel2", "tail": "D"},
            {"head": "E", "relation": "rel3", "tail": "F"}
        ]
        
        # Act
        result = service.stage_triples_batch(triples, "batch_source")
        
        # Assert
        assert isinstance(result, IngestionResult)
        assert result.total_triples == 3
        assert result.successful_triples == 3
        assert result.failed_triples == 0
        assert mock_stage_single.call_count == 3
    
    @patch.object(IngestionService, '_stage_single_triple')
    def test_stage_triples_batch_handles_partial_failures(self, mock_stage_single):
        """Test batch staging with some failures"""
        # Arrange
        service = IngestionService()
        mock_stage_single.side_effect = [True, False, True]  # Second one fails
        
        triples = [
            {"head": "A", "relation": "rel1", "tail": "B"},
            {"head": "C", "relation": "rel2", "tail": "D"},
            {"head": "E", "relation": "rel3", "tail": "F"}
        ]
        
        # Act
        result = service.stage_triples_batch(triples)
        
        # Assert
        assert result.total_triples == 3
        assert result.successful_triples == 2
        assert result.failed_triples == 1
    
    @patch.object(IngestionService, '_stage_single_triple')
    def test_stage_triples_batch_handles_exceptions(self, mock_stage_single):
        """Test batch staging with exceptions during processing"""
        # Arrange
        service = IngestionService()
        mock_stage_single.side_effect = [True, Exception("Database error"), True]
        
        triples = [
            {"head": "A", "relation": "rel1", "tail": "B"},
            {"head": "C", "relation": "rel2", "tail": "D"},
            {"head": "E", "relation": "rel3", "tail": "F"}
        ]
        
        # Act
        result = service.stage_triples_batch(triples)
        
        # Assert
        assert result.total_triples == 3
        assert result.successful_triples == 2  # First and third succeed
        assert len(result.errors) > 0
        assert "Database error" in str(result.errors)


class TestEndToEndIngestion:
    """Test complete ingestion workflows"""
    
    @patch('builtins.open', new_callable=mock_open, read_data="Barack Obama was born in Hawaii. He became President.")
    @patch('src.app.services.ingestion.get_entity_type')
    @patch('src.app.services.ingestion.sqlite_db')
    def test_complete_file_ingestion_workflow(self, mock_db, mock_get_type, mock_file):
        """Test complete file ingestion from start to finish"""
        # Arrange
        service = IngestionService()
        mock_get_type.return_value = "PERSON"
        mock_db.execute.return_value = None
        
        # Mock IE service
        mock_extraction_result = Mock()
        mock_extraction_result.success = True
        mock_extraction_result.triples = [
            {"head": "Barack Obama", "relation": "born in", "tail": "Hawaii", "confidence": 0.9},
            {"head": "Barack Obama", "relation": "became", "tail": "President", "confidence": 0.8}
        ]
        service.ie_service.extract_triples = Mock(return_value=mock_extraction_result)
        
        # Act
        result = service.process_text_file("test_file.txt", chunk_size=100)
        
        # Assert
        assert isinstance(result, IngestionResult)
        assert result.total_triples == 2
        assert result.successful_triples == 2
        assert result.failed_triples == 0
        assert len(result.errors) == 0
        assert result.processing_time > 0
        
        # Verify file was read
        mock_file.assert_called_once_with("test_file.txt", 'r', encoding='utf-8')
        
        # Verify IE was called
        service.ie_service.extract_triples.assert_called()
        
        # Verify database staging
        assert mock_db.execute.call_count == 2  # Two triples staged


@pytest.mark.performance
class TestIngestionPerformance:
    """Test ingestion performance characteristics"""
    
    def test_text_processing_completes_within_reasonable_time(self):
        """Test that text processing doesn't take unreasonably long"""
        # Arrange
        service = IngestionService()
        test_text = "Barack Obama was born in Hawaii. " * 10  # Moderate length text
        
        # Act
        start_time = time.time()
        result = service.process_text_content(test_text)
        elapsed_time = time.time() - start_time
        
        # Assert
        # Should complete within 60 seconds (allowing for model loading)
        assert elapsed_time < 60.0
        assert result.processing_time < 60.0
    
    def test_chunking_performance_scales_linearly(self):
        """Test that chunking performance scales reasonably with text size"""
        # Arrange
        service = IngestionService()
        small_text = "Test sentence. " * 100
        large_text = "Test sentence. " * 1000
        
        # Act
        start_time = time.time()
        small_chunks = service._split_text_into_chunks(small_text, DEFAULT_CHUNK_SIZE)
        small_time = time.time() - start_time
        
        start_time = time.time()
        large_chunks = service._split_text_into_chunks(large_text, DEFAULT_CHUNK_SIZE)
        large_time = time.time() - start_time
        
        # Assert
        # Large text should take more time but not exponentially more
        assert large_time < small_time * 20  # Allow for some overhead but not exponential
        assert len(large_chunks) > len(small_chunks)


@pytest.mark.integration
class TestIngestionIntegration:
    """Integration tests that use real components where possible"""
    
    def test_ingestion_service_integrates_with_ie_service(self):
        """Test that ingestion service properly integrates with IE service"""
        # Arrange
        service = IngestionService()
        test_text = "Barack Obama was born in Hawaii."
        
        # Act - Use real IE service integration
        result = service.process_text_content(test_text)
        
        # Assert
        assert isinstance(result, IngestionResult)
        # Should either succeed or fail gracefully
        assert result.processing_time > 0
        if result.successful_triples > 0:
            assert result.total_triples >= result.successful_triples
        else:
            # If no triples extracted, should have error or be empty input
            assert len(result.errors) > 0 or test_text.strip() == "" 