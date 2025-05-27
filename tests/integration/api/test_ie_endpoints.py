"""
Integration tests for Information Extraction API endpoints

Tests the IE API endpoints with real FastAPI application and service integration.
"""

import pytest
import json
import time
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from src.main import app
from src.app.config import API_KEY_SECRET


class TestIEHealthEndpoint:
    """Test IE health check endpoint"""
    
    def test_ie_health_endpoint_returns_status(self):
        """Test that IE health endpoint returns proper status structure"""
        # Arrange
        client = TestClient(app)
        
        # Act
        response = client.get("/ie/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models" in data
        assert "overall_status" in data
        
        # Check models structure
        assert "rebel" in data["models"]
        assert "ner" in data["models"]
        
        # Check REBEL model info
        rebel_info = data["models"]["rebel"]
        assert rebel_info["name"] == "Babelscape/rebel-large"
        assert rebel_info["purpose"] == "relation_extraction"
        assert "status" in rebel_info
        
        # Check NER model info
        ner_info = data["models"]["ner"]
        assert ner_info["name"] == "tner/roberta-large-ontonotes5"
        assert ner_info["purpose"] == "entity_typing"
        assert "status" in ner_info
    
    def test_ie_health_endpoint_accessible_without_auth(self):
        """Test that health endpoint doesn't require authentication"""
        # Arrange
        client = TestClient(app)
        
        # Act - No API key provided
        response = client.get("/ie/health")
        
        # Assert
        assert response.status_code == 200  # Should not require auth


class TestIEInfoEndpoint:
    """Test IE model info endpoint"""
    
    def test_ie_info_endpoint_returns_model_details(self):
        """Test that IE info endpoint returns detailed model information"""
        # Arrange
        client = TestClient(app)
        
        # Act
        response = client.get("/ie/info")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "service" in data
        assert "models" in data
        assert "overall_status" in data
        
        # Should have exactly 2 models
        assert len(data["models"]) == 2
        
        # Check model details
        for model in data["models"]:
            assert "model_name" in model
            assert "description" in model
            assert "capabilities" in model
            assert "input_format" in model
            assert "output_format" in model
            assert "loaded" in model
            assert "purpose" in model
    
    def test_ie_info_endpoint_accessible_without_auth(self):
        """Test that info endpoint doesn't require authentication"""
        # Arrange
        client = TestClient(app)
        
        # Act - No API key provided
        response = client.get("/ie/info")
        
        # Assert
        assert response.status_code == 200  # Should not require auth


class TestIEExtractEndpoint:
    """Test IE extraction endpoint"""
    
    def test_extract_endpoint_requires_authentication(self):
        """Test that extract endpoint requires API key"""
        # Arrange
        client = TestClient(app)
        payload = {"text": "Barack Obama was born in Hawaii."}
        
        # Act - No API key
        response = client.post("/ie/extract", json=payload)
        
        # Assert
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]
    
    def test_extract_endpoint_rejects_invalid_api_key(self):
        """Test that extract endpoint rejects invalid API key"""
        # Arrange
        client = TestClient(app)
        payload = {"text": "Barack Obama was born in Hawaii."}
        headers = {"X-API-KEY": "invalid_key"}
        
        # Act
        response = client.post("/ie/extract", json=payload, headers=headers)
        
        # Assert
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]
    
    def test_extract_endpoint_accepts_valid_api_key(self):
        """Test that extract endpoint accepts valid API key"""
        # Arrange
        client = TestClient(app)
        payload = {"text": "Barack Obama was born in Hawaii."}
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        response = client.post("/ie/extract", json=payload, headers=headers)
        
        # Assert
        # Should not be 401 (may be other status depending on model availability)
        assert response.status_code != 401
    
    def test_extract_endpoint_validates_request_structure(self):
        """Test that extract endpoint validates request structure"""
        # Arrange
        client = TestClient(app)
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Test missing text field
        invalid_payload = {"max_length": 256}
        
        # Act
        response = client.post("/ie/extract", json=invalid_payload, headers=headers)
        
        # Assert
        assert response.status_code == 422  # Validation error
    
    def test_extract_endpoint_handles_empty_text(self):
        """Test extraction with empty text"""
        # Arrange
        client = TestClient(app)
        payload = {"text": ""}
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        response = client.post("/ie/extract", json=payload, headers=headers)
        
        # Assert
        # Should handle gracefully (either succeed with empty result or proper error)
        assert response.status_code in [200, 400, 500]
        if response.status_code == 200:
            data = response.json()
            assert "triples" in data
            assert "processing_time" in data
    
    @patch('src.app.services.information_extraction.extract')
    def test_extract_endpoint_returns_proper_response_structure(self, mock_extract):
        """Test that extract endpoint returns proper response structure"""
        # Arrange
        client = TestClient(app)
        mock_extract.return_value = [
            {
                "head": "Barack Obama",
                "relation": "born in",
                "tail": "Hawaii",
                "head_type": "PERSON",
                "tail_type": "LOCATION"
            }
        ]
        
        payload = {"text": "Barack Obama was born in Hawaii."}
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        response = client.post("/ie/extract", json=payload, headers=headers)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "triples" in data
        assert "raw_output" in data
        assert "processing_time" in data
        
        # Check triples structure
        assert len(data["triples"]) == 1
        triple = data["triples"][0]
        assert "head" in triple
        assert "relation" in triple
        assert "tail" in triple
        assert "head_type" in triple
        assert "tail_type" in triple
        assert "confidence" in triple
        
        # Check values
        assert triple["head"] == "Barack Obama"
        assert triple["relation"] == "born in"
        assert triple["tail"] == "Hawaii"
        assert triple["head_type"] == "PERSON"
        assert triple["tail_type"] == "LOCATION"
    
    def test_extract_endpoint_respects_custom_parameters(self):
        """Test that extract endpoint accepts custom parameters"""
        # Arrange
        client = TestClient(app)
        payload = {
            "text": "Barack Obama was born in Hawaii.",
            "max_length": 512,
            "num_beams": 5
        }
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        response = client.post("/ie/extract", json=payload, headers=headers)
        
        # Assert
        # Should accept the request (may succeed or fail based on model availability)
        assert response.status_code != 422  # Not a validation error
    
    @patch('src.app.services.information_extraction.extract')
    def test_extract_endpoint_handles_extraction_failure(self, mock_extract):
        """Test handling of extraction failures"""
        # Arrange
        client = TestClient(app)
        mock_extract.side_effect = Exception("Model inference failed")
        
        payload = {"text": "Barack Obama was born in Hawaii."}
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        response = client.post("/ie/extract", json=payload, headers=headers)
        
        # Assert
        assert response.status_code == 500
        assert "Extraction failed" in response.json()["detail"]


class TestTextIngestEndpoint:
    """Test text ingestion endpoint"""
    
    def test_ingest_text_endpoint_requires_authentication(self):
        """Test that ingest endpoint requires API key"""
        # Arrange
        client = TestClient(app)
        payload = {"text": "Barack Obama was born in Hawaii."}
        
        # Act - No API key
        response = client.post("/ingest/text", json=payload)
        
        # Assert
        assert response.status_code == 401
    
    def test_ingest_text_endpoint_validates_request_structure(self):
        """Test that ingest endpoint validates request structure"""
        # Arrange
        client = TestClient(app)
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Test missing text field
        invalid_payload = {"source": "test"}
        
        # Act
        response = client.post("/ingest/text", json=invalid_payload, headers=headers)
        
        # Assert
        assert response.status_code == 422  # Validation error
    
    @patch('src.app.services.ingestion.get_ingestion_service')
    def test_ingest_text_endpoint_returns_proper_response_structure(self, mock_get_service):
        """Test that ingest endpoint returns proper response structure"""
        # Arrange
        client = TestClient(app)
        
        # Mock ingestion service
        mock_service = Mock()
        mock_result = Mock()
        mock_result.total_triples = 2
        mock_result.successful_triples = 2
        mock_result.failed_triples = 0
        mock_result.processing_time = 1.5
        mock_result.errors = []
        mock_result.warnings = []
        mock_service.process_text_content.return_value = mock_result
        mock_get_service.return_value = mock_service
        
        payload = {"text": "Barack Obama was born in Hawaii."}
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        response = client.post("/ingest/text", json=payload, headers=headers)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "total_triples" in data
        assert "successful_triples" in data
        assert "failed_triples" in data
        assert "processing_time" in data
        assert "errors" in data
        assert "warnings" in data
        
        # Check values
        assert data["total_triples"] == 2
        assert data["successful_triples"] == 2
        assert data["failed_triples"] == 0
        assert data["processing_time"] == 1.5
    
    def test_ingest_text_endpoint_accepts_custom_parameters(self):
        """Test that ingest endpoint accepts custom parameters"""
        # Arrange
        client = TestClient(app)
        payload = {
            "text": "Barack Obama was born in Hawaii.",
            "source": "custom_source",
            "chunk_size": 500
        }
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        response = client.post("/ingest/text", json=payload, headers=headers)
        
        # Assert
        # Should accept the request (may succeed or fail based on service availability)
        assert response.status_code != 422  # Not a validation error


@pytest.mark.integration
class TestIEEndpointIntegration:
    """Integration tests using real services where possible"""
    
    def test_ie_health_reflects_actual_service_state(self):
        """Test that health endpoint reflects actual service state"""
        # Arrange
        client = TestClient(app)
        
        # Act
        response = client.get("/ie/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Health status should be consistent with actual service state
        assert data["status"] in ["healthy", "partial", "unhealthy"]
        assert data["overall_status"] in ["ready", "partial", "not_loaded"]
        
        # Model statuses should be consistent
        rebel_status = data["models"]["rebel"]["status"]
        ner_status = data["models"]["ner"]["status"]
        assert rebel_status in ["loaded", "not_loaded"]
        assert ner_status in ["loaded", "not_loaded"]
    
    def test_ie_info_provides_accurate_model_information(self):
        """Test that info endpoint provides accurate model information"""
        # Arrange
        client = TestClient(app)
        
        # Act
        response = client.get("/ie/info")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Should have accurate model names
        model_names = [model["model_name"] for model in data["models"]]
        assert "Babelscape/rebel-large" in model_names
        assert "tner/roberta-large-ontonotes5" in model_names
        
        # Should have proper capabilities listed
        for model in data["models"]:
            assert isinstance(model["capabilities"], list)
            assert len(model["capabilities"]) > 0
    
    @pytest.mark.slow
    def test_extract_endpoint_with_real_service_integration(self):
        """Test extraction endpoint with real service (may load models)"""
        # Arrange
        client = TestClient(app)
        payload = {"text": "Barack Obama was born in Hawaii."}
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        start_time = time.time()
        response = client.post("/ie/extract", json=payload, headers=headers)
        elapsed_time = time.time() - start_time
        
        # Assert
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "triples" in data
            assert "processing_time" in data
            assert isinstance(data["triples"], list)
            
            # If triples were extracted, they should have proper structure
            for triple in data["triples"]:
                assert "head" in triple
                assert "relation" in triple
                assert "tail" in triple
                assert "confidence" in triple
        
        # Should complete within reasonable time (allowing for model loading)
        assert elapsed_time < 120.0  # 2 minutes max


@pytest.mark.performance
class TestIEEndpointPerformance:
    """Performance tests for IE endpoints"""
    
    def test_health_endpoint_responds_quickly(self):
        """Test that health endpoint responds quickly"""
        # Arrange
        client = TestClient(app)
        
        # Act
        start_time = time.time()
        response = client.get("/ie/health")
        elapsed_time = time.time() - start_time
        
        # Assert
        assert response.status_code == 200
        assert elapsed_time < 1.0  # Should be very fast
    
    def test_info_endpoint_responds_quickly(self):
        """Test that info endpoint responds quickly"""
        # Arrange
        client = TestClient(app)
        
        # Act
        start_time = time.time()
        response = client.get("/ie/info")
        elapsed_time = time.time() - start_time
        
        # Assert
        assert response.status_code == 200
        assert elapsed_time < 1.0  # Should be very fast
    
    @patch('src.app.services.information_extraction.extract')
    def test_extract_endpoint_performance_with_mocked_service(self, mock_extract):
        """Test extract endpoint performance with mocked service"""
        # Arrange
        client = TestClient(app)
        mock_extract.return_value = [
            {"head": "Obama", "relation": "born in", "tail": "Hawaii", "head_type": "PERSON", "tail_type": "LOCATION"}
        ]
        
        payload = {"text": "Barack Obama was born in Hawaii."}
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        start_time = time.time()
        response = client.post("/ie/extract", json=payload, headers=headers)
        elapsed_time = time.time() - start_time
        
        # Assert
        assert response.status_code == 200
        # With mocked service, should be very fast
        assert elapsed_time < 2.0


@pytest.mark.adversarial
class TestIEEndpointRobustness:
    """Adversarial tests for IE endpoints"""
    
    def test_extract_endpoint_handles_very_long_text(self):
        """Test extraction with very long text input"""
        # Arrange
        client = TestClient(app)
        long_text = "Barack Obama was born in Hawaii. " * 1000  # Very long text
        payload = {"text": long_text}
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        response = client.post("/ie/extract", json=payload, headers=headers)
        
        # Assert
        # Should handle gracefully (either succeed or proper error)
        assert response.status_code in [200, 400, 413, 500]
    
    def test_extract_endpoint_handles_special_characters(self):
        """Test extraction with special characters and unicode"""
        # Arrange
        client = TestClient(app)
        special_text = "Barack Obama 🇺🇸 was born in Hawaii 🏝️. He speaks English & español."
        payload = {"text": special_text}
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        response = client.post("/ie/extract", json=payload, headers=headers)
        
        # Assert
        # Should handle gracefully
        assert response.status_code in [200, 400, 500]
    
    def test_extract_endpoint_handles_malformed_json(self):
        """Test extraction with malformed JSON"""
        # Arrange
        client = TestClient(app)
        headers = {"X-API-KEY": API_KEY_SECRET, "Content-Type": "application/json"}
        
        # Act
        response = client.post("/ie/extract", data="invalid json", headers=headers)
        
        # Assert
        assert response.status_code == 422  # JSON parsing error
    
    def test_extract_endpoint_handles_extreme_parameters(self):
        """Test extraction with extreme parameter values"""
        # Arrange
        client = TestClient(app)
        payload = {
            "text": "Barack Obama was born in Hawaii.",
            "max_length": 999999,  # Very large
            "num_beams": 100       # Very large
        }
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        response = client.post("/ie/extract", json=payload, headers=headers)
        
        # Assert
        # Should handle gracefully (either succeed or proper error)
        assert response.status_code in [200, 400, 500] 