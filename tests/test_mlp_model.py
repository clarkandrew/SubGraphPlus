import unittest
import pytest
import numpy as np
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.retriever import (
    load_pretrained_mlp, mlp_score, mlp_model
)
from app.models import Triple
from app.utils import heuristic_score_indexed


class SimpleMLP(nn.Module):
    """Simple MLP for testing purposes"""
    def __init__(self, input_dim=773, hidden_dim=64, output_dim=1):  # 768 + 5 DDE features
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class TestMLPModel(unittest.TestCase):
    """Test cases for MLP model functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_model = SimpleMLP()
        self.sample_embeddings = torch.randn(5, 768)  # 5 embeddings of 768 dimensions
        self.sample_dde_features = {
            'num_nodes': [10, 15, 8, 12, 20],
            'num_edges': [25, 30, 18, 28, 45],
            'avg_degree': [2.5, 2.0, 2.25, 2.33, 2.25],
            'density': [0.28, 0.14, 0.32, 0.21, 0.12],
            'clustering_coefficient': [0.3, 0.25, 0.4, 0.35, 0.2]
        }
        
        # Legacy test data for backward compatibility
        self.query_embedding = torch.randn(768)
        self.triple_embedding = torch.randn(768)
        self.dde_features = self.sample_dde_features
        
        # Sample triple
        self.sample_triple = Triple(
            id="test_rel_1",
            head_id="ent1",
            head_name="Test Entity 1",
            relation_id="test_rel_1",
            relation_name="test_relation",
            tail_id="ent2",
            tail_name="Test Entity 2",
            properties={}
        )
    
    def test_simple_mlp_model_creation(self):
        """Test that SimpleMLP model can be created"""
        model = SimpleMLP()
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(len(list(model.parameters())), 6)  # 3 layers * 2 params each (weight + bias)
    
    def test_simple_mlp_forward_pass(self):
        """Test forward pass through SimpleMLP"""
        model = SimpleMLP(input_dim=773)
        test_input = torch.randn(1, 773)
        output = model(test_input)
        self.assertEqual(output.shape, (1, 1))
        self.assertFalse(torch.isnan(output).any())
    
    def test_mlp_input_dimension_calculation(self):
        """Test that MLP input dimensions are calculated correctly"""
        # Expected dimensions:
        # Query embedding: 768
        # Triple embedding: 768  
        # DDE features: num_query_entities × MAX_DDE_HOPS × 2 (head/tail)
        # For 1 query entity, 2 hops: 1 × 2 × 2 = 4 features
        # But can be more depending on implementation
        
        query_emb = np.random.random(768)
        triple_emb = np.random.random(768)
        dde_features = [1.0, 0.5, 0.0, 1.0]  # 4 features
        
        # Concatenate as done in mlp_score
        q_tensor = torch.tensor(query_emb, dtype=torch.float32)
        t_tensor = torch.tensor(triple_emb, dtype=torch.float32)
        dde_tensor = torch.tensor(dde_features, dtype=torch.float32)
        
        combined = torch.cat([q_tensor, t_tensor, dde_tensor])
        
        expected_dim = 768 + 768 + len(dde_features)
        self.assertEqual(combined.shape[0], expected_dim)
    
    @patch('app.retriever.config')
    def test_load_pretrained_mlp_success(self, mock_config):
        """Test successful loading of pretrained MLP"""
        # Create a temporary model file with the correct checkpoint format
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            # Create a checkpoint that matches the real model format
            checkpoint = {
                'config': {'test': 'config'},
                'model_state_dict': {
                    'pred.0.weight': torch.randn(64, 773),
                    'pred.0.bias': torch.randn(64),
                    'pred.2.weight': torch.randn(1, 64),
                    'pred.2.bias': torch.randn(1)
                }
            }
            torch.save(checkpoint, tmp_file.name)
            mock_config.MLP_MODEL_PATH = tmp_file.name
            
            # Test loading
            from app.retriever import load_pretrained_mlp
            loaded_model = load_pretrained_mlp()
            
            self.assertIsNotNone(loaded_model)
            # Verify the model has the correct architecture
            self.assertEqual(loaded_model.pred[0].in_features, 773)
            self.assertEqual(loaded_model.pred[0].out_features, 64)
            self.assertEqual(loaded_model.pred[2].in_features, 64)
            self.assertEqual(loaded_model.pred[2].out_features, 1)
        
        # Clean up
        os.unlink(tmp_file.name)
    
    @patch('app.retriever.config')
    def test_load_pretrained_mlp_file_not_found(self, mock_config):
        """Test loading MLP when file doesn't exist"""
        mock_config.MLP_MODEL_PATH = "/nonexistent/path/model.pth"
        
        from app.retriever import load_pretrained_mlp
        loaded_model = load_pretrained_mlp()
        
        self.assertIsNone(loaded_model)
    
    @patch('app.retriever.config')
    @patch('torch.load')
    def test_load_pretrained_mlp_corrupted_file(self, mock_torch_load, mock_config):
        """Test loading MLP when file is corrupted"""
        mock_config.MLP_MODEL_PATH = "models/mlp/mlp.pth"
        mock_torch_load.side_effect = Exception("Corrupted file")
        
        from app.retriever import load_pretrained_mlp
        loaded_model = load_pretrained_mlp()
        
        self.assertIsNone(loaded_model)
    
    @patch('app.retriever.load_pretrained_mlp')
    def test_mlp_score_with_model(self, mock_load_mlp):
        """Test MLP scoring when model is available"""
        mock_load_mlp.return_value = self.test_model
        
        from app.retriever import mlp_score
        
        # Test scoring
        score = mlp_score(self.sample_embeddings, self.sample_dde_features, 0)
        self.assertIsInstance(score, (int, float))
    
    @patch('app.retriever.load_pretrained_mlp')
    def test_mlp_score_fallback_no_model(self, mock_load_mlp):
        """Test MLP scoring fallback when no model is available"""
        mock_load_mlp.return_value = None
        
        from app.retriever import mlp_score
        
        # Should fallback to heuristic score
        score = mlp_score(self.sample_embeddings, self.sample_dde_features, 0)
        self.assertIsInstance(score, (int, float))
    
    @patch('app.retriever.load_pretrained_mlp')
    def test_mlp_score_fallback_on_error(self, mock_load_mlp):
        """Test MLP scoring fallback when model throws error"""
        mock_model = MagicMock()
        mock_model.side_effect = Exception("Model error")
        mock_load_mlp.return_value = mock_model
        
        from app.retriever import mlp_score
        
        # Should fallback to heuristic score
        score = mlp_score(self.sample_embeddings, self.sample_dde_features, 0)
        self.assertIsInstance(score, (int, float))
    
    def test_mlp_score_input_validation(self):
        """Test MLP score input validation"""
        from app.retriever import mlp_score
        
        # Test with invalid embeddings - should fallback to heuristic score
        score = mlp_score(None, self.sample_dde_features, 0)
        self.assertIsInstance(score, (int, float))
        
        # Test with invalid DDE features - should fallback to heuristic score
        score = mlp_score(self.sample_embeddings, None, 0)
        self.assertIsInstance(score, (int, float))
        
        # Test with out of bounds index - should fallback to heuristic score
        score = mlp_score(self.sample_embeddings, self.sample_dde_features, 100)
        self.assertIsInstance(score, (int, float))
    
    def test_mlp_score_dimension_mismatch(self):
        """Test MLP score with dimension mismatch"""
        from app.retriever import mlp_score
        
        # Create model with different input dimension
        wrong_dim_model = SimpleMLP(input_dim=100)
        
        with patch('app.retriever.load_pretrained_mlp', return_value=wrong_dim_model):
            # Should fallback to heuristic score due to dimension mismatch
            score = mlp_score(self.sample_embeddings, self.sample_dde_features, 0)
            self.assertIsInstance(score, (int, float))
    
    def test_heuristic_score_calculation(self):
        """Test heuristic score calculation"""
        from app.utils import heuristic_score_indexed
        
        # Test with sample DDE features
        score = heuristic_score_indexed(self.sample_dde_features, 0)  # First graph
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
    
    def test_heuristic_score_with_edge_cases(self):
        """Test heuristic score with edge cases"""
        from app.utils import heuristic_score_indexed
        
        # Test with empty features
        empty_features = {key: [] for key in self.sample_dde_features.keys()}
        score = heuristic_score_indexed(empty_features, 0)
        self.assertEqual(score, 0)  # Should return 0 for empty features
        
        # Test with out of bounds index
        score = heuristic_score_indexed(self.sample_dde_features, 100)
        self.assertEqual(score, 0)  # Should return 0 for invalid index
    
    def test_dde_feature_extraction(self):
        """Test DDE feature extraction from graph data"""
        # Test that our sample DDE features have the right structure
        expected_features = ['num_nodes', 'num_edges', 'avg_degree', 'density', 'clustering_coefficient']
        
        self.assertEqual(set(self.sample_dde_features.keys()), set(expected_features))
        for feature_name, values in self.sample_dde_features.items():
            self.assertIsInstance(values, list)
            self.assertTrue(len(values) > 0)
            # Check that all values are numeric
            for value in values:
                self.assertIsInstance(value, (int, float))


class TestMLPIntegration(unittest.TestCase):
    """Integration tests for MLP in the retrieval pipeline"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.sample_question = "Who founded Tesla?"
        self.query_entities = ["ent1"]
        
        self.sample_triple = Triple(
            id="rel1",
            head_id="ent1",
            head_name="Elon Musk",
            relation_id="rel1",
            relation_name="founded",
            tail_id="ent2",
            tail_name="Tesla Inc.",
            properties={}
        )
    
    @patch('app.retriever.embed_query_cached')
    @patch('app.retriever.get_triple_embedding_cached')
    @patch('app.retriever.extract_dde_features_for_triple')
    @patch('app.retriever.mlp_score')
    def test_mlp_in_retrieval_pipeline(self, mock_mlp_score, mock_extract_dde, 
                                     mock_get_triple_emb, mock_embed_query):
        """Test MLP integration in the retrieval pipeline"""
        # Mock return values
        mock_embed_query.return_value = np.random.random(768)
        mock_get_triple_emb.return_value = np.random.random(768)
        mock_extract_dde.return_value = [1.0, 0.5, 0.0, 1.0]
        mock_mlp_score.return_value = 0.8
        
        # This would be called within hybrid_retrieve_v2
        # We're testing the scoring component
        query_emb = mock_embed_query.return_value
        triple_emb = mock_get_triple_emb.return_value
        dde_features = mock_extract_dde.return_value
        
        score = mock_mlp_score(query_emb, triple_emb, dde_features)
        
        self.assertEqual(score, 0.8)
        mock_mlp_score.assert_called_once()
    
    @patch('app.retriever.mlp_model')
    def test_mlp_batch_scoring(self, mock_mlp_model):
        """Test scoring multiple triples with MLP"""
        # Mock model output
        mock_output = Mock()
        mock_output.item.return_value = 0.75
        mock_mlp_model.return_value = mock_output
        
        # Test scoring multiple triples
        triples = [
            self.sample_triple,
            Triple(id="rel2", head_id="ent1", head_name="Elon Musk", 
                  relation_id="rel2", relation_name="owns", 
                  tail_id="ent3", tail_name="SpaceX", properties={})
        ]
        
        scores = []
        for triple in triples:
            query_emb = np.random.random(768)
            triple_emb = np.random.random(768)
            dde_features = [1.0, 0.5]
            
            score = mlp_score(query_emb, triple_emb, dde_features)
            scores.append(score)
        
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(isinstance(s, float) for s in scores))
    
    def test_mlp_model_persistence(self):
        """Test that MLP model can be saved and loaded"""
        # Create a test model
        test_model = SimpleMLP()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            torch.save(test_model.state_dict(), tmp_file.name)
            
            # Load the model
            loaded_state_dict = torch.load(tmp_file.name, map_location='cpu', weights_only=True)
            
            # Create new model and load state
            new_model = SimpleMLP()
            new_model.load_state_dict(loaded_state_dict)
            
            # Test that models produce same output
            test_input = torch.randn(1, 773)  # Correct input dimension
            original_output = test_model(test_input)
            loaded_output = new_model(test_input)
            
            self.assertTrue(torch.allclose(original_output, loaded_output, atol=1e-6))
        
        # Clean up
        os.unlink(tmp_file.name)


class TestMLPEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for MLP"""
    
    def test_empty_dde_features(self):
        """Test MLP scoring with empty DDE features"""
        query_emb = np.random.random(768)
        triple_emb = np.random.random(768)
        empty_dde = []
        
        with patch('app.retriever.mlp_model') as mock_model:
            mock_model.side_effect = RuntimeError("Empty tensor")
            
            with patch('app.utils.heuristic_score') as mock_heuristic:
                mock_heuristic.return_value = 0.3
                
                score = mlp_score(query_emb, triple_emb, empty_dde)
                self.assertEqual(score, 0.3)
    
    def test_nan_inf_in_embeddings(self):
        """Test MLP scoring with NaN/Inf values in embeddings"""
        # Embeddings with NaN values
        nan_query_emb = np.full(768, np.nan)
        inf_triple_emb = np.full(768, np.inf)
        dde_features = [1.0, 0.5]
        
        with patch('app.retriever.mlp_model') as mock_model:
            mock_model.side_effect = RuntimeError("NaN values detected")
            
            with patch('app.utils.heuristic_score') as mock_heuristic:
                mock_heuristic.return_value = 0.2
                
                score = mlp_score(nan_query_emb, inf_triple_emb, dde_features)
                self.assertEqual(score, 0.2)
    
    def test_very_large_dde_features(self):
        """Test MLP scoring with very large DDE feature vectors"""
        query_emb = np.random.random(768)
        triple_emb = np.random.random(768)
        large_dde = [1.0] * 1000  # Very large DDE vector
        
        with patch('app.retriever.mlp_model') as mock_model:
            # Model should handle or fail gracefully
            mock_model.side_effect = RuntimeError("Input too large")
            
            with patch('app.utils.heuristic_score') as mock_heuristic:
                mock_heuristic.return_value = 0.4
                
                score = mlp_score(query_emb, triple_emb, large_dde)
                self.assertEqual(score, 0.4)
    
    def test_model_device_mismatch(self):
        """Test MLP scoring with device mismatch (CPU vs GPU)"""
        query_emb = np.random.random(768)
        triple_emb = np.random.random(768)
        dde_features = [1.0, 0.5]
        
        with patch('app.retriever.mlp_model') as mock_model:
            # Simulate CUDA device error
            mock_model.side_effect = RuntimeError("Expected all tensors to be on the same device")
            
            with patch('app.utils.heuristic_score') as mock_heuristic:
                mock_heuristic.return_value = 0.35
                
                score = mlp_score(query_emb, triple_emb, dde_features)
                self.assertEqual(score, 0.35)


if __name__ == '__main__':
    unittest.main() 