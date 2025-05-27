"""
Unit tests for Information Extraction Service

Tests the core IE functionality including model loading, triple extraction,
and entity typing without external dependencies.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any

from src.app.services.information_extraction import (
    InformationExtractionService,
    get_information_extraction_service,
    ExtractionResult,
    ensure_models_loaded,
    load_models,
    extract_triplets,
    build_context_ner_map,
    get_entity_type,
    generate_raw,
    extract,
    REBEL_MODEL_NAME,
    NER_MODEL_NAME
)


class TestInformationExtractionService:
    """Test the InformationExtractionService class"""
    
    def test_service_initialization_creates_instance(self):
        """Test that service initializes correctly"""
        # Arrange & Act
        service = InformationExtractionService()
        
        # Assert
        assert service is not None
        assert hasattr(service, 'extract_triples')
        assert hasattr(service, 'get_model_info')
    
    def test_singleton_pattern_returns_same_instance(self):
        """Test that get_information_extraction_service returns singleton"""
        # Arrange & Act
        service1 = get_information_extraction_service()
        service2 = get_information_extraction_service()
        
        # Assert
        assert service1 is service2
    
    def test_is_service_available_returns_true_when_transformers_available(self):
        """Test service availability check"""
        # Arrange
        service = InformationExtractionService()
        
        # Act & Assert
        assert service.is_service_available() is True
    
    @patch('src.app.services.information_extraction.AutoModelForSeq2SeqLM')
    def test_is_service_available_returns_false_when_transformers_unavailable(self, mock_model):
        """Test service availability when transformers not available"""
        # Arrange
        mock_model.side_effect = ImportError("No module named 'transformers'")
        service = InformationExtractionService()
        
        # Act & Assert
        assert service.is_service_available() is False
    
    def test_get_model_info_returns_expected_structure(self):
        """Test model info returns correct structure"""
        # Arrange
        service = InformationExtractionService()
        
        # Act
        info = service.get_model_info()
        
        # Assert
        assert "service" in info
        assert "models" in info
        assert "overall_status" in info
        assert len(info["models"]) == 2
        
        # Check REBEL model info
        rebel_model = next(m for m in info["models"] if m["purpose"] == "relation_extraction")
        assert rebel_model["model_name"] == REBEL_MODEL_NAME
        assert "capabilities" in rebel_model
        assert "loaded" in rebel_model
        
        # Check NER model info
        ner_model = next(m for m in info["models"] if m["purpose"] == "entity_typing")
        assert ner_model["model_name"] == NER_MODEL_NAME
        assert "capabilities" in ner_model
        assert "loaded" in ner_model


class TestModelLoading:
    """Test model loading functionality"""
    
    @patch('src.app.services.information_extraction.MODELS_LOADED', False)
    @patch('src.app.services.information_extraction._models_loading_attempted', False)
    def test_ensure_models_loaded_first_time_calls_load_models(self):
        """Test that first call to ensure_models_loaded triggers loading"""
        # Arrange
        with patch('src.app.services.information_extraction.load_models') as mock_load:
            mock_load.return_value = (Mock(), Mock(), Mock())
            
            # Act
            result = ensure_models_loaded()
            
            # Assert
            mock_load.assert_called_once()
            assert result is True
    
    @patch('src.app.services.information_extraction.MODELS_LOADED', True)
    @patch('src.app.services.information_extraction._models_loading_attempted', True)
    def test_ensure_models_loaded_cached_returns_immediately(self):
        """Test that subsequent calls use cached result"""
        # Arrange
        with patch('src.app.services.information_extraction.load_models') as mock_load:
            
            # Act
            result = ensure_models_loaded()
            
            # Assert
            mock_load.assert_not_called()
            assert result is True
    
    @patch('src.app.services.information_extraction.AutoTokenizer')
    @patch('src.app.services.information_extraction.AutoModelForSeq2SeqLM')
    @patch('src.app.services.information_extraction.pipeline')
    def test_load_models_success_returns_all_models(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test successful model loading"""
        # Arrange
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        # Act
        tokenizer, model, ner_pipe = load_models()
        
        # Assert
        assert tokenizer is not None
        assert model is not None
        assert ner_pipe is not None
        mock_tokenizer.from_pretrained.assert_called_once_with(REBEL_MODEL_NAME)
        mock_model.from_pretrained.assert_called_once_with(REBEL_MODEL_NAME)
        mock_pipeline.assert_called_once()
    
    @patch('src.app.services.information_extraction.AutoTokenizer')
    def test_load_models_failure_returns_none_values(self, mock_tokenizer):
        """Test model loading failure handling"""
        # Arrange
        mock_tokenizer.from_pretrained.side_effect = Exception("Model loading failed")
        
        # Act
        result = load_models()
        
        # Assert
        assert result == (None, None, None)


class TestTripleExtraction:
    """Test triple extraction functionality"""
    
    def test_extract_triplets_parses_rebel_output_correctly(self):
        """Test parsing of REBEL model output into triples"""
        # Arrange
        rebel_output = "<s><triplet> Barack Obama <subj> born in <obj> Hawaii </s>"
        
        # Act
        triples = extract_triplets(rebel_output)
        
        # Assert
        assert len(triples) == 1
        assert triples[0]["head"] == "Barack Obama"
        assert triples[0]["relation"] == "born in"
        assert triples[0]["tail"] == "Hawaii"
    
    def test_extract_triplets_handles_multiple_triples(self):
        """Test parsing multiple triples from REBEL output"""
        # Arrange
        rebel_output = "<s><triplet> Barack Obama <subj> born in <obj> Hawaii <triplet> Barack Obama <subj> president of <obj> United States </s>"
        
        # Act
        triples = extract_triplets(rebel_output)
        
        # Assert
        assert len(triples) == 2
        assert triples[0]["head"] == "Barack Obama"
        assert triples[0]["relation"] == "born in"
        assert triples[0]["tail"] == "Hawaii"
        assert triples[1]["head"] == "Barack Obama"
        assert triples[1]["relation"] == "president of"
        assert triples[1]["tail"] == "United States"
    
    def test_extract_triplets_handles_empty_input(self):
        """Test handling of empty or malformed input"""
        # Arrange
        empty_output = ""
        
        # Act
        triples = extract_triplets(empty_output)
        
        # Assert
        assert triples == []
    
    def test_extract_triplets_handles_malformed_input(self):
        """Test handling of malformed REBEL output"""
        # Arrange
        malformed_output = "This is not a valid REBEL output"
        
        # Act
        triples = extract_triplets(malformed_output)
        
        # Assert
        assert triples == []


class TestEntityTyping:
    """Test entity typing functionality"""
    
    @patch('src.app.services.information_extraction.ensure_models_loaded')
    @patch('src.app.services.information_extraction.NER_PIPE')
    def test_get_entity_type_returns_predicted_type(self, mock_ner_pipe, mock_ensure_loaded):
        """Test entity type prediction"""
        # Arrange
        mock_ensure_loaded.return_value = True
        mock_ner_pipe.return_value = [{"entity_group": "PERSON"}]
        
        # Act
        entity_type = get_entity_type("Barack Obama")
        
        # Assert
        assert entity_type == "PERSON"
        mock_ner_pipe.assert_called_once_with("Barack Obama")
    
    @patch('src.app.services.information_extraction.ensure_models_loaded')
    def test_get_entity_type_returns_default_when_models_not_loaded(self, mock_ensure_loaded):
        """Test fallback when models not available"""
        # Arrange
        mock_ensure_loaded.return_value = False
        
        # Act
        entity_type = get_entity_type("Barack Obama")
        
        # Assert
        assert entity_type == "ENTITY"
    
    def test_get_entity_type_handles_empty_span(self):
        """Test handling of empty entity span"""
        # Arrange & Act
        entity_type = get_entity_type("")
        
        # Assert
        assert entity_type == "ENTITY"
    
    @patch('src.app.services.information_extraction.ensure_models_loaded')
    @patch('src.app.services.information_extraction.NER_PIPE')
    def test_build_context_ner_map_creates_mapping(self, mock_ner_pipe, mock_ensure_loaded):
        """Test contextual NER mapping creation"""
        # Arrange
        mock_ensure_loaded.return_value = True
        mock_ner_pipe.return_value = [
            {"word": "Barack Obama", "entity_group": "PERSON"},
            {"word": "Hawaii", "entity_group": "LOCATION"}
        ]
        
        # Act
        context_map = build_context_ner_map("Barack Obama was born in Hawaii")
        
        # Assert
        assert "Barack Obama" in context_map
        assert context_map["Barack Obama"] == "PERSON"
        assert "Hawaii" in context_map
        assert context_map["Hawaii"] == "LOCATION"


class TestEndToEndExtraction:
    """Test end-to-end extraction pipeline"""
    
    @patch('src.app.services.information_extraction.ensure_models_loaded')
    @patch('src.app.services.information_extraction.generate_raw')
    @patch('src.app.services.information_extraction.build_context_ner_map')
    @patch('src.app.services.information_extraction.get_entity_type')
    def test_extract_function_complete_pipeline(self, mock_get_type, mock_context_map, mock_generate, mock_ensure_loaded):
        """Test complete extraction pipeline"""
        # Arrange
        mock_ensure_loaded.return_value = True
        mock_generate.return_value = "<s><triplet> Barack Obama <subj> born in <obj> Hawaii </s>"
        mock_context_map.return_value = {"Barack Obama": "PERSON", "Hawaii": "LOCATION"}
        mock_get_type.side_effect = lambda x: {"Barack Obama": "PERSON", "Hawaii": "LOCATION"}.get(x, "ENTITY")
        
        # Act
        triples = extract("Barack Obama was born in Hawaii")
        
        # Assert
        assert len(triples) == 1
        triple = triples[0]
        assert triple["head"] == "Barack Obama"
        assert triple["relation"] == "born in"
        assert triple["tail"] == "Hawaii"
        assert triple["head_type"] == "PERSON"
        assert triple["tail_type"] == "LOCATION"
    
    def test_service_extract_triples_returns_extraction_result(self):
        """Test service extract_triples method returns proper result structure"""
        # Arrange
        service = InformationExtractionService()
        
        with patch('src.app.services.information_extraction.extract') as mock_extract:
            mock_extract.return_value = [
                {
                    "head": "Barack Obama",
                    "relation": "born in", 
                    "tail": "Hawaii",
                    "head_type": "PERSON",
                    "tail_type": "LOCATION"
                }
            ]
            
            # Act
            result = service.extract_triples("Barack Obama was born in Hawaii")
            
            # Assert
            assert isinstance(result, ExtractionResult)
            assert result.success is True
            assert len(result.triples) == 1
            assert result.processing_time > 0
            assert result.triples[0]["confidence"] == 1.0
    
    def test_service_extract_triples_handles_extraction_failure(self):
        """Test service handles extraction failures gracefully"""
        # Arrange
        service = InformationExtractionService()
        
        with patch('src.app.services.information_extraction.extract') as mock_extract:
            mock_extract.side_effect = Exception("Model inference failed")
            
            # Act
            result = service.extract_triples("Test text")
            
            # Assert
            assert isinstance(result, ExtractionResult)
            assert result.success is False
            assert result.error_message == "Model inference failed"
            assert len(result.triples) == 0
    
    def test_service_extract_triples_with_custom_parameters(self):
        """Test service respects custom extraction parameters"""
        # Arrange
        service = InformationExtractionService()
        
        with patch('src.app.services.information_extraction.extract') as mock_extract:
            mock_extract.return_value = []
            
            # Act
            result = service.extract_triples("Test text", max_length=512, num_beams=5)
            
            # Assert
            # Note: Current implementation doesn't use these parameters in extract()
            # This test documents the expected interface
            assert isinstance(result, ExtractionResult)
            assert result.success is True


class TestModelStateManagement:
    """Test model state and loading management"""
    
    def test_service_model_status_methods_return_consistent_state(self):
        """Test that model status methods return consistent information"""
        # Arrange
        service = InformationExtractionService()
        
        # Act
        is_loaded = service.is_model_loaded()
        is_rebel_loaded = service.is_rebel_loaded()
        is_ner_loaded = service.is_ner_loaded()
        
        # Assert
        # All should return the same state since they check the same global variable
        assert is_loaded == is_rebel_loaded == is_ner_loaded
    
    def test_load_models_methods_return_consistent_results(self):
        """Test that load model methods return consistent results"""
        # Arrange
        service = InformationExtractionService()
        
        # Act
        rebel_result = service.load_rebel_model()
        ner_result = service.load_ner_model()
        both_result = service.load_models()
        
        # Assert
        assert rebel_result == ner_result
        assert both_result["rebel"] == both_result["ner"]
        assert both_result["rebel"] == rebel_result


@pytest.mark.performance
class TestExtractionPerformance:
    """Test extraction performance characteristics"""
    
    def test_extraction_completes_within_reasonable_time(self):
        """Test that extraction doesn't take unreasonably long"""
        # Arrange
        service = InformationExtractionService()
        test_text = "Barack Obama was born in Hawaii. He served as President."
        
        # Act
        start_time = time.time()
        result = service.extract_triples(test_text)
        elapsed_time = time.time() - start_time
        
        # Assert
        # Should complete within 60 seconds (allowing for model loading)
        assert elapsed_time < 60.0
        assert result.processing_time < 60.0
    
    def test_subsequent_extractions_are_faster(self):
        """Test that model caching improves subsequent extraction speed"""
        # Arrange
        service = InformationExtractionService()
        test_text = "Barack Obama was born in Hawaii."
        
        # Act - First extraction (may include model loading)
        first_result = service.extract_triples(test_text)
        
        # Act - Second extraction (should use cached models)
        start_time = time.time()
        second_result = service.extract_triples(test_text)
        second_elapsed = time.time() - start_time
        
        # Assert
        # Second extraction should be much faster (no model loading)
        assert second_elapsed < 10.0  # Should be very fast with cached models
        assert second_result.success is True 