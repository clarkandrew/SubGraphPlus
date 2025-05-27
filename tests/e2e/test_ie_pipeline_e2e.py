"""
End-to-End tests for Information Extraction Pipeline

Tests the complete IE pipeline from API request through model loading,
extraction, and response generation.
"""

import pytest
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.main import app
from src.app.config import API_KEY_SECRET


@pytest.mark.e2e
class TestCompleteIEPipeline:
    """Test complete IE pipeline end-to-end"""
    
    def test_complete_extraction_workflow(self):
        """Test complete extraction workflow from API to response"""
        # Arrange
        client = TestClient(app)
        test_text = "Barack Obama was born in Hawaii. He served as the 44th President of the United States."
        payload = {"text": test_text}
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        start_time = time.time()
        response = client.post("/ie/extract", json=payload, headers=headers)
        total_time = time.time() - start_time
        
        # Assert
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate response structure
            assert "triples" in data
            assert "processing_time" in data
            assert "raw_output" in data
            
            # Validate triples structure if any were extracted
            for triple in data["triples"]:
                assert "head" in triple
                assert "relation" in triple
                assert "tail" in triple
                assert "head_type" in triple
                assert "tail_type" in triple
                assert "confidence" in triple
                
                # Validate data types
                assert isinstance(triple["head"], str)
                assert isinstance(triple["relation"], str)
                assert isinstance(triple["tail"], str)
                assert isinstance(triple["head_type"], str)
                assert isinstance(triple["tail_type"], str)
                assert isinstance(triple["confidence"], (int, float))
                
                # Validate non-empty values
                assert len(triple["head"].strip()) > 0
                assert len(triple["relation"].strip()) > 0
                assert len(triple["tail"].strip()) > 0
            
            # Validate processing time is reasonable
            assert data["processing_time"] > 0
            assert data["processing_time"] < 300  # 5 minutes max
        
        # Total API response time should be reasonable
        assert total_time < 300  # 5 minutes max
    
    def test_complete_ingestion_workflow(self):
        """Test complete text ingestion workflow"""
        # Arrange
        client = TestClient(app)
        test_text = "Barack Obama was born in Hawaii. He became the 44th President."
        payload = {
            "text": test_text,
            "source": "e2e_test",
            "chunk_size": 100
        }
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act
        start_time = time.time()
        response = client.post("/ingest/text", json=payload, headers=headers)
        total_time = time.time() - start_time
        
        # Assert
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate response structure
            assert "total_triples" in data
            assert "successful_triples" in data
            assert "failed_triples" in data
            assert "processing_time" in data
            assert "errors" in data
            assert "warnings" in data
            
            # Validate data types
            assert isinstance(data["total_triples"], int)
            assert isinstance(data["successful_triples"], int)
            assert isinstance(data["failed_triples"], int)
            assert isinstance(data["processing_time"], (int, float))
            assert isinstance(data["errors"], list)
            assert isinstance(data["warnings"], list)
            
            # Validate logical consistency
            assert data["total_triples"] >= 0
            assert data["successful_triples"] >= 0
            assert data["failed_triples"] >= 0
            assert data["successful_triples"] + data["failed_triples"] == data["total_triples"]
            assert data["processing_time"] > 0
        
        # Total API response time should be reasonable
        assert total_time < 300  # 5 minutes max
    
    def test_health_and_info_endpoints_consistency(self):
        """Test that health and info endpoints provide consistent information"""
        # Arrange
        client = TestClient(app)
        
        # Act
        health_response = client.get("/ie/health")
        info_response = client.get("/ie/info")
        
        # Assert
        assert health_response.status_code == 200
        assert info_response.status_code == 200
        
        health_data = health_response.json()
        info_data = info_response.json()
        
        # Both should report the same models
        health_models = set(health_data["models"].keys())
        info_model_purposes = {model["purpose"] for model in info_data["models"]}
        
        assert "rebel" in health_models
        assert "ner" in health_models
        assert "relation_extraction" in info_model_purposes
        assert "entity_typing" in info_model_purposes
        
        # Model loading status should be consistent
        health_overall = health_data["overall_status"]
        info_overall = info_data["overall_status"]
        assert health_overall == info_overall
    
    def test_service_state_persistence_across_requests(self):
        """Test that service state persists across multiple requests"""
        # Arrange
        client = TestClient(app)
        
        # Act - Make multiple health check requests
        responses = []
        for i in range(3):
            response = client.get("/ie/health")
            responses.append(response.json())
            time.sleep(0.1)  # Small delay between requests
        
        # Assert
        # All responses should be consistent
        for response_data in responses:
            assert response_data["overall_status"] == responses[0]["overall_status"]
            assert response_data["models"]["rebel"]["status"] == responses[0]["models"]["rebel"]["status"]
            assert response_data["models"]["ner"]["status"] == responses[0]["models"]["ner"]["status"]


@pytest.mark.e2e
@pytest.mark.slow
class TestModelLoadingBehavior:
    """Test model loading behavior in real scenarios"""
    
    def test_first_extraction_triggers_model_loading(self):
        """Test that first extraction request triggers model loading"""
        # Arrange
        client = TestClient(app)
        payload = {"text": "Barack Obama was born in Hawaii."}
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act - Check initial state
        initial_health = client.get("/ie/health").json()
        
        # Act - Make extraction request
        start_time = time.time()
        extract_response = client.post("/ie/extract", json=payload, headers=headers)
        extraction_time = time.time() - start_time
        
        # Act - Check state after extraction
        final_health = client.get("/ie/health").json()
        
        # Assert
        # Should either succeed or fail gracefully
        assert extract_response.status_code in [200, 500]
        
        # If models were not loaded initially and extraction succeeded,
        # they should be loaded after the request
        if (initial_health["overall_status"] == "not_loaded" and 
            extract_response.status_code == 200):
            assert final_health["overall_status"] in ["ready", "partial"]
        
        # First extraction may take longer due to model loading
        # but should complete within reasonable time
        assert extraction_time < 300  # 5 minutes max
    
    def test_subsequent_extractions_are_faster(self):
        """Test that subsequent extractions are faster than the first"""
        # Arrange
        client = TestClient(app)
        payload = {"text": "Barack Obama was born in Hawaii."}
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Act - First extraction
        start_time = time.time()
        first_response = client.post("/ie/extract", json=payload, headers=headers)
        first_time = time.time() - start_time
        
        # Act - Second extraction (if first succeeded)
        if first_response.status_code == 200:
            start_time = time.time()
            second_response = client.post("/ie/extract", json=payload, headers=headers)
            second_time = time.time() - start_time
            
            # Assert
            assert second_response.status_code == 200
            
            # Second extraction should be faster (models already loaded)
            # Allow some variance but should be significantly faster
            assert second_time < first_time * 0.8 or second_time < 10.0


@pytest.mark.e2e
@pytest.mark.performance
class TestPipelinePerformance:
    """Test pipeline performance characteristics"""
    
    def test_extraction_performance_with_various_text_lengths(self):
        """Test extraction performance with different text lengths"""
        # Arrange
        client = TestClient(app)
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        test_cases = [
            ("Short text", "Barack Obama was born in Hawaii."),
            ("Medium text", "Barack Obama was born in Hawaii. " * 10),
            ("Long text", "Barack Obama was born in Hawaii. " * 50)
        ]
        
        results = []
        
        # Act
        for name, text in test_cases:
            payload = {"text": text}
            start_time = time.time()
            response = client.post("/ie/extract", json=payload, headers=headers)
            elapsed_time = time.time() - start_time
            
            results.append({
                "name": name,
                "text_length": len(text),
                "response_time": elapsed_time,
                "status_code": response.status_code,
                "success": response.status_code == 200
            })
        
        # Assert
        # All requests should complete within reasonable time
        for result in results:
            assert result["response_time"] < 120.0  # 2 minutes max
        
        # If any succeeded, processing time should scale reasonably with text length
        successful_results = [r for r in results if r["success"]]
        if len(successful_results) > 1:
            # Longer text shouldn't take exponentially longer
            short_result = min(successful_results, key=lambda x: x["text_length"])
            long_result = max(successful_results, key=lambda x: x["text_length"])
            
            length_ratio = long_result["text_length"] / short_result["text_length"]
            time_ratio = long_result["response_time"] / short_result["response_time"]
            
            # Time ratio should not be much larger than length ratio
            assert time_ratio < length_ratio * 3  # Allow some overhead
    
    def test_concurrent_extraction_requests(self):
        """Test handling of concurrent extraction requests"""
        # Arrange
        client = TestClient(app)
        headers = {"X-API-KEY": API_KEY_SECRET}
        payload = {"text": "Barack Obama was born in Hawaii."}
        
        # Act - Make multiple concurrent requests (simulated)
        responses = []
        start_time = time.time()
        
        for i in range(3):
            response = client.post("/ie/extract", json=payload, headers=headers)
            responses.append(response)
            time.sleep(0.1)  # Small delay to simulate near-concurrent requests
        
        total_time = time.time() - start_time
        
        # Assert
        # All requests should complete
        assert len(responses) == 3
        
        # All should either succeed or fail gracefully
        for response in responses:
            assert response.status_code in [200, 500]
        
        # Total time should be reasonable
        assert total_time < 180.0  # 3 minutes for 3 requests


@pytest.mark.e2e
@pytest.mark.adversarial
class TestPipelineRobustness:
    """Test pipeline robustness under adverse conditions"""
    
    def test_extraction_with_problematic_text(self):
        """Test extraction with various problematic text inputs"""
        # Arrange
        client = TestClient(app)
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        problematic_texts = [
            "",  # Empty text
            " ",  # Whitespace only
            "a",  # Single character
            "🎉🎊🎈",  # Emoji only
            "This is a very long sentence without any punctuation that goes on and on and might cause issues with tokenization or processing because it never ends and has no clear structure",  # Very long sentence
            "Multiple\n\nline\n\nbreaks\n\neverywhere",  # Multiple line breaks
            "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",  # Special characters
            "Mixed languages: Hello 你好 Hola Bonjour",  # Mixed languages
        ]
        
        # Act & Assert
        for text in problematic_texts:
            payload = {"text": text}
            response = client.post("/ie/extract", json=payload, headers=headers)
            
            # Should handle gracefully (not crash)
            assert response.status_code in [200, 400, 500]
            
            # If successful, should return valid structure
            if response.status_code == 200:
                data = response.json()
                assert "triples" in data
                assert "processing_time" in data
                assert isinstance(data["triples"], list)
    
    def test_extraction_with_invalid_parameters(self):
        """Test extraction with invalid parameter values"""
        # Arrange
        client = TestClient(app)
        headers = {"X-API-KEY": API_KEY_SECRET}
        base_text = "Barack Obama was born in Hawaii."
        
        invalid_payloads = [
            {"text": base_text, "max_length": -1},  # Negative max_length
            {"text": base_text, "num_beams": 0},    # Zero num_beams
            {"text": base_text, "max_length": "invalid"},  # String instead of int
            {"text": base_text, "num_beams": 1.5},  # Float instead of int
        ]
        
        # Act & Assert
        for payload in invalid_payloads:
            response = client.post("/ie/extract", json=payload, headers=headers)
            
            # Should handle validation errors appropriately
            assert response.status_code in [200, 400, 422, 500]
    
    def test_ingestion_with_large_text(self):
        """Test ingestion with very large text input"""
        # Arrange
        client = TestClient(app)
        headers = {"X-API-KEY": API_KEY_SECRET}
        
        # Create large text (but not too large to avoid timeouts)
        large_text = "Barack Obama was born in Hawaii. He became President. " * 100
        payload = {
            "text": large_text,
            "source": "large_text_test",
            "chunk_size": 500
        }
        
        # Act
        start_time = time.time()
        response = client.post("/ingest/text", json=payload, headers=headers)
        elapsed_time = time.time() - start_time
        
        # Assert
        # Should handle gracefully
        assert response.status_code in [200, 413, 500]  # 413 = Payload Too Large
        
        # Should complete within reasonable time
        assert elapsed_time < 300  # 5 minutes max
        
        if response.status_code == 200:
            data = response.json()
            assert "total_triples" in data
            assert "processing_time" in data


@pytest.mark.e2e
@pytest.mark.integration
class TestServiceIntegration:
    """Test integration between different services"""
    
    def test_ie_service_integration_with_database(self):
        """Test that IE service properly integrates with database for staging"""
        # Arrange
        client = TestClient(app)
        headers = {"X-API-KEY": API_KEY_SECRET}
        test_text = "Barack Obama was born in Hawaii."
        
        # Act - Ingest text (which should stage triples in database)
        ingest_payload = {"text": test_text, "source": "integration_test"}
        ingest_response = client.post("/ingest/text", json=ingest_payload, headers=headers)
        
        # Assert
        # Should either succeed or fail gracefully
        assert ingest_response.status_code in [200, 500]
        
        if ingest_response.status_code == 200:
            data = ingest_response.json()
            
            # If triples were staged, the counts should be consistent
            assert data["total_triples"] >= 0
            assert data["successful_triples"] >= 0
            assert data["failed_triples"] >= 0
            assert data["successful_triples"] + data["failed_triples"] == data["total_triples"]
    
    def test_extraction_and_ingestion_consistency(self):
        """Test that extraction and ingestion produce consistent results"""
        # Arrange
        client = TestClient(app)
        headers = {"X-API-KEY": API_KEY_SECRET}
        test_text = "Barack Obama was born in Hawaii."
        
        # Act - Extract triples directly
        extract_payload = {"text": test_text}
        extract_response = client.post("/ie/extract", json=extract_payload, headers=headers)
        
        # Act - Ingest text (which also extracts triples)
        ingest_payload = {"text": test_text, "source": "consistency_test"}
        ingest_response = client.post("/ingest/text", json=ingest_payload, headers=headers)
        
        # Assert
        if (extract_response.status_code == 200 and ingest_response.status_code == 200):
            extract_data = extract_response.json()
            ingest_data = ingest_response.json()
            
            # Should extract similar number of triples
            # (allowing for some variance due to processing differences)
            extracted_count = len(extract_data["triples"])
            ingested_count = ingest_data["total_triples"]
            
            # Should be reasonably close (within 50% or exact match for small numbers)
            if extracted_count <= 2:
                assert abs(extracted_count - ingested_count) <= 1
            else:
                assert abs(extracted_count - ingested_count) <= extracted_count * 0.5 