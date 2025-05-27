import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Set testing environment before any imports
os.environ['TESTING'] = '1'

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from src.validation import get_validation_metrics


class TestValidationMetricsSimple:
    """Simple validation metrics tests using mocks"""

    def test_get_validation_metrics_success_with_mock(self):
        """Test successful retrieval of validation metrics using mock database"""
        # Create mock database
        mock_db = MagicMock()
        
        # Mock database responses
        mock_db.fetchone.return_value = (5,)  # Model exists with 5 records
        mock_db.fetchall.return_value = [
            (True, "Good answer"),
            (True, "Accurate"),
            (False, "Incorrect"),
            (True, "Perfect")
        ]
        
        result = get_validation_metrics("test_model", mock_db)
        
        assert result["status"] == "success"
        assert result["message"] == "Validation metrics retrieved successfully"
        assert result["metrics"] is not None
        
        metrics = result["metrics"]
        assert metrics["total_validations"] == 4
        assert metrics["correct_validations"] == 3
        assert metrics["incorrect_validations"] == 1
        assert metrics["accuracy"] == 0.75

    def test_get_validation_metrics_model_not_found_with_mock(self):
        """Test handling when model is not found using mock database"""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)  # No records found
        
        result = get_validation_metrics("nonexistent_model", mock_db)
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
        assert result["metrics"] is None

    def test_get_validation_metrics_no_validation_results_with_mock(self):
        """Test handling when model exists but has no validation results"""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (3,)  # Model exists
        mock_db.fetchall.return_value = []  # No validation results
        
        result = get_validation_metrics("lonely_model", mock_db)
        
        assert result["status"] == "error"
        assert "No validation results found" in result["message"]
        assert result["metrics"] is None

    def test_get_validation_metrics_no_database(self):
        """Test handling when no database is provided"""
        result = get_validation_metrics("test_model", None)
        
        assert result["status"] == "error"
        assert "Database not available" in result["message"]
        assert result["metrics"] is None

    def test_get_validation_metrics_database_error_with_mock(self):
        """Test handling of database errors using mock"""
        mock_db = MagicMock()
        mock_db.fetchone.side_effect = Exception("Database connection failed")
        
        result = get_validation_metrics("test_model", mock_db)
        
        assert result["status"] == "error"
        assert "Error getting validation metrics" in result["message"]
        assert result["metrics"] is None

    def test_get_validation_metrics_all_correct_with_mock(self):
        """Test metrics calculation when all validations are correct"""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (1,)  # Model exists
        mock_db.fetchall.return_value = [
            (True, "Perfect answer"),
            (True, "Excellent"),
            (True, "Great")
        ]
        
        result = get_validation_metrics("perfect_model", mock_db)
        
        assert result["status"] == "success"
        metrics = result["metrics"]
        assert metrics["accuracy"] == 1.0
        assert metrics["total_validations"] == 3
        assert metrics["correct_validations"] == 3
        assert metrics["incorrect_validations"] == 0

    def test_get_validation_metrics_all_incorrect_with_mock(self):
        """Test metrics calculation when all validations are incorrect"""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (1,)  # Model exists
        mock_db.fetchall.return_value = [
            (False, "Wrong answer"),
            (False, "Incorrect")
        ]
        
        result = get_validation_metrics("bad_model", mock_db)
        
        assert result["status"] == "success"
        metrics = result["metrics"]
        assert metrics["accuracy"] == 0.0
        assert metrics["total_validations"] == 2
        assert metrics["correct_validations"] == 0
        assert metrics["incorrect_validations"] == 2

    def test_get_validation_metrics_empty_model_id_with_mock(self):
        """Test handling of empty model ID"""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)  # No records for empty string
        
        result = get_validation_metrics("", mock_db)
        
        assert result["status"] == "error"
        assert result["metrics"] is None

    def test_get_validation_metrics_large_dataset_with_mock(self):
        """Test performance with larger dataset using mock"""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (1,)  # Model exists
        
        # Create 100 alternating results (50% accuracy)
        mock_results = [(i % 2 == 0, f"Comment {i}") for i in range(100)]
        mock_db.fetchall.return_value = mock_results
        
        result = get_validation_metrics("large_model", mock_db)
        
        assert result["status"] == "success"
        assert result["metrics"]["total_validations"] == 100
        assert result["metrics"]["accuracy"] == 0.5

    def test_get_validation_metrics_database_calls(self):
        """Test that the correct database calls are made"""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (5,)
        mock_db.fetchall.return_value = [(True, "Good")]
        
        get_validation_metrics("test_model", mock_db)
        
        # Verify the correct SQL queries were called
        assert mock_db.fetchone.called
        assert mock_db.fetchall.called
        
        # Check the SQL query patterns
        fetchone_call = mock_db.fetchone.call_args
        fetchall_call = mock_db.fetchall.call_args
        
        assert "staging_triples" in fetchone_call[0][0]
        assert "feedback" in fetchall_call[0][0]
        assert "test_model" in str(fetchone_call[0][1])
        assert "test_model" in str(fetchall_call[0][1])


class TestValidationMetricsIntegration:
    """Integration tests for validation metrics without database locking issues"""

    @patch('src.validation.sqlite_db')
    def test_get_validation_metrics_with_global_db_fallback(self, mock_global_db):
        """Test that the function falls back to global sqlite_db when no db is provided"""
        mock_global_db.fetchone.return_value = (1,)
        mock_global_db.fetchall.return_value = [(True, "Good")]
        
        # Call without providing db parameter
        result = get_validation_metrics("test_model")
        
        assert result["status"] == "success"
        assert mock_global_db.fetchone.called
        assert mock_global_db.fetchall.called

    @patch('src.validation.sqlite_db', None)
    def test_get_validation_metrics_no_global_db(self):
        """Test handling when global sqlite_db is None (testing mode)"""
        result = get_validation_metrics("test_model")
        
        assert result["status"] == "error"
        assert "Database not available" in result["message"]
        assert result["metrics"] is None 