#!/usr/bin/env python3
"""
Consolidated Information Extraction (IE) Smoke Tests

This file consolidates all IE smoke tests into a single, comprehensive test suite
that validates core IE functionality quickly and reliably.
"""

import os
import sys
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.app.log import logger


class TestIESmoke:
    """Consolidated smoke tests for Information Extraction functionality"""
    
    def test_ie_imports_work(self):
        """Test that all IE modules can be imported without errors"""
        logger.debug("Starting IE import test")
        
        try:
            from src.app.services.information_extraction import (
                InformationExtractionService,
                get_information_extraction_service,
                extract,
                REBEL_MODEL_NAME,
                NER_MODEL_NAME
            )
            logger.debug("✅ IE imports successful")
            assert True
        except Exception as e:
            logger.error(f"❌ IE import failed: {e}")
            pytest.fail(f"Failed to import IE modules: {e}")
    
    def test_ie_service_instantiation(self):
        """Test that IE service can be instantiated"""
        logger.debug("Starting IE service instantiation test")
        
        try:
            from src.app.services.information_extraction import get_information_extraction_service
            
            service = get_information_extraction_service()
            assert service is not None
            assert hasattr(service, 'extract_triples')
            assert hasattr(service, 'is_model_loaded')
            
            logger.debug("✅ IE service instantiation successful")
        except Exception as e:
            logger.error(f"❌ IE service instantiation failed: {e}")
            pytest.fail(f"Failed to instantiate IE service: {e}")
    
    def test_ie_service_singleton(self):
        """Test that IE service follows singleton pattern"""
        logger.debug("Starting IE service singleton test")
        
        try:
            from src.app.services.information_extraction import get_information_extraction_service
            
            service1 = get_information_extraction_service()
            service2 = get_information_extraction_service()
            
            assert service1 is service2, "IE service should be a singleton"
            logger.debug("✅ IE service singleton pattern verified")
        except Exception as e:
            logger.error(f"❌ IE service singleton test failed: {e}")
            pytest.fail(f"IE service singleton test failed: {e}")
    
    @pytest.mark.skipif(
        os.environ.get("SUBGRAPHRAG_DISABLE_MODEL_LOADING") == "true",
        reason="Model loading disabled for testing"
    )
    def test_ie_model_loading_real(self):
        """Test real model loading (only if models are enabled)"""
        logger.debug("Starting real IE model loading test")
        
        try:
            from src.app.services.information_extraction import load_models
            
            start_time = time.time()
            tokenizer, rebel_model, ner_pipe = load_models()
            load_time = time.time() - start_time
            
            assert tokenizer is not None, "REBEL tokenizer should be loaded"
            assert rebel_model is not None, "REBEL model should be loaded"
            assert ner_pipe is not None, "NER pipeline should be loaded"
            
            logger.debug(f"✅ Real models loaded in {load_time:.2f}s")
            
            # Test that models are functional
            test_text = "Barack Obama was born in Hawaii."
            
            # Test NER pipeline
            ner_results = ner_pipe(test_text)
            assert isinstance(ner_results, list), "NER should return a list"
            
            logger.debug("✅ Real model functionality verified")
            
        except Exception as e:
            logger.error(f"❌ Real model loading failed: {e}")
            pytest.fail(f"Real model loading failed: {e}")
    
    def test_ie_extraction_mocked(self):
        """Test IE extraction with mocked models"""
        logger.debug("Starting mocked IE extraction test")
        
        try:
            with patch('src.app.services.information_extraction.load_models') as mock_load:
                # Mock the models
                mock_tokenizer = Mock()
                mock_model = Mock()
                mock_ner_pipe = Mock()
                
                # Configure tokenizer mock
                mock_tokenizer.return_value = {
                    'input_ids': [[1, 2, 3, 4, 5]],
                    'attention_mask': [[1, 1, 1, 1, 1]]
                }
                
                # Configure model mock to return REBEL format
                mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
                mock_tokenizer.decode.return_value = "<triplet> Barack Obama <subj> place of birth <obj> Hawaii"
                
                # Configure NER mock
                mock_ner_pipe.return_value = [
                    {'entity': 'PERSON', 'word': 'Barack Obama', 'start': 0, 'end': 12},
                    {'entity': 'GPE', 'word': 'Hawaii', 'start': 25, 'end': 31}
                ]
                
                mock_load.return_value = (mock_tokenizer, mock_model, mock_ner_pipe)
                
                # Test extraction
                from src.app.services.information_extraction import extract
                
                test_text = "Barack Obama was born in Hawaii."
                triples = extract(test_text)
                
                assert isinstance(triples, list), "Extract should return a list"
                logger.debug(f"✅ Mocked extraction returned {len(triples)} triples")
                
        except Exception as e:
            logger.error(f"❌ Mocked extraction failed: {e}")
            pytest.fail(f"Mocked extraction failed: {e}")
    
    def test_ie_service_extract_triples_mocked(self):
        """Test IE service extract_triples method with mocked models"""
        logger.debug("Starting IE service extract_triples test")
        
        try:
            with patch('src.app.services.information_extraction.load_models') as mock_load:
                # Mock the models
                mock_tokenizer = Mock()
                mock_model = Mock()
                mock_ner_pipe = Mock()
                
                # Configure mocks for successful extraction
                mock_tokenizer.return_value = {
                    'input_ids': [[1, 2, 3, 4, 5]],
                    'attention_mask': [[1, 1, 1, 1, 1]]
                }
                mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
                mock_tokenizer.decode.return_value = "<triplet> Barack Obama <subj> place of birth <obj> Hawaii"
                mock_ner_pipe.return_value = [
                    {'entity': 'PERSON', 'word': 'Barack Obama', 'start': 0, 'end': 12},
                    {'entity': 'GPE', 'word': 'Hawaii', 'start': 25, 'end': 31}
                ]
                
                mock_load.return_value = (mock_tokenizer, mock_model, mock_ner_pipe)
                
                # Test service extraction
                from src.app.services.information_extraction import get_information_extraction_service
                
                service = get_information_extraction_service()
                result = service.extract_triples("Barack Obama was born in Hawaii.")
                
                assert hasattr(result, 'success'), "Result should have success attribute"
                assert hasattr(result, 'triples'), "Result should have triples attribute"
                assert hasattr(result, 'processing_time'), "Result should have processing_time attribute"
                
                logger.debug(f"✅ Service extraction completed: success={result.success}")
                
        except Exception as e:
            logger.error(f"❌ Service extraction test failed: {e}")
            pytest.fail(f"Service extraction test failed: {e}")
    
    def test_ie_error_handling(self):
        """Test IE error handling with invalid inputs"""
        logger.debug("Starting IE error handling test")
        
        try:
            from src.app.services.information_extraction import get_information_extraction_service
            
            service = get_information_extraction_service()
            
            # Test with empty string
            result = service.extract_triples("")
            assert hasattr(result, 'success'), "Result should have success attribute"
            
            # Test with None
            result = service.extract_triples(None)
            assert hasattr(result, 'success'), "Result should have success attribute"
            
            # Test with very long string
            long_text = "word " * 1000
            result = service.extract_triples(long_text)
            assert hasattr(result, 'success'), "Result should have success attribute"
            
            logger.debug("✅ IE error handling tests passed")
            
        except Exception as e:
            logger.error(f"❌ IE error handling test failed: {e}")
            pytest.fail(f"IE error handling test failed: {e}")
    
    def test_ie_performance_basic(self):
        """Test basic IE performance characteristics"""
        logger.debug("Starting IE performance test")
        
        try:
            from src.app.services.information_extraction import get_information_extraction_service
            
            service = get_information_extraction_service()
            
            # Test extraction timing
            test_text = "Barack Obama was born in Hawaii and became President."
            
            start_time = time.time()
            result = service.extract_triples(test_text)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Basic performance assertions
            assert processing_time < 30.0, f"Extraction took too long: {processing_time:.2f}s"
            
            if hasattr(result, 'processing_time'):
                assert result.processing_time >= 0, "Processing time should be non-negative"
            
            logger.debug(f"✅ IE performance test passed: {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ IE performance test failed: {e}")
            pytest.fail(f"IE performance test failed: {e}")


class TestIEConfiguration:
    """Test IE configuration and environment setup"""
    
    def test_ie_model_names_defined(self):
        """Test that model names are properly defined"""
        logger.debug("Starting IE model names test")
        
        try:
            from src.app.services.information_extraction import REBEL_MODEL_NAME, NER_MODEL_NAME
            
            assert isinstance(REBEL_MODEL_NAME, str), "REBEL_MODEL_NAME should be a string"
            assert isinstance(NER_MODEL_NAME, str), "NER_MODEL_NAME should be a string"
            assert len(REBEL_MODEL_NAME) > 0, "REBEL_MODEL_NAME should not be empty"
            assert len(NER_MODEL_NAME) > 0, "NER_MODEL_NAME should not be empty"
            
            logger.debug(f"✅ Model names verified: REBEL={REBEL_MODEL_NAME}, NER={NER_MODEL_NAME}")
            
        except Exception as e:
            logger.error(f"❌ Model names test failed: {e}")
            pytest.fail(f"Model names test failed: {e}")
    
    def test_ie_environment_variables(self):
        """Test IE environment variable handling"""
        logger.debug("Starting IE environment variables test")
        
        # Test model loading disable flag
        disable_flag = os.environ.get("SUBGRAPHRAG_DISABLE_MODEL_LOADING")
        logger.debug(f"Model loading disabled: {disable_flag}")
        
        # Test tokenizers parallelism
        tokenizers_parallel = os.environ.get("TOKENIZERS_PARALLELISM")
        logger.debug(f"Tokenizers parallelism: {tokenizers_parallel}")
        
        # Test transformers offline mode
        transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        logger.debug(f"Transformers offline: {transformers_offline}")
        
        logger.debug("✅ Environment variables checked")


if __name__ == "__main__":
    # Run smoke tests directly
    pytest.main([__file__, "-v", "--tb=short"]) 