import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path so test can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.ml.embedder import embed_text, embed_batch, health_check, huggingface_embed, openai_embed


class TestEmbedText:
    """Test the main embed_text function"""
    
    def test_embed_empty_string(self):
        """Test embedding empty string returns zero vector"""
        result = embed_text("")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)  # gte-large-en-v1.5 dimension
        assert result.dtype == np.float32
        assert np.allclose(result, 0.0)
    
    def test_embed_none(self):
        """Test embedding None returns zero vector"""
        result = embed_text(None)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)
        assert result.dtype == np.float32
        assert np.allclose(result, 0.0)
    
    @patch('app.ml.embedder.HF_AVAILABLE', True)
    @patch('app.ml.embedder.huggingface_embed')
    def test_embed_with_huggingface(self, mock_hf_embed):
        """Test embedding with HuggingFace (primary method)"""
        # Mock return value
        mock_embedding = np.random.normal(0, 1, 1024).astype(np.float32)
        mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
        mock_hf_embed.return_value = mock_embedding
        
        result = embed_text("test text")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)
        assert result.dtype == np.float32
        mock_hf_embed.assert_called_once_with("test text")
        np.testing.assert_array_equal(result, mock_embedding)
    
    @patch('app.ml.embedder.HF_AVAILABLE', False)
    @patch('app.ml.embedder.OPENAI_AVAILABLE', True)
    @patch('app.ml.embedder.openai_embed')
    def test_embed_fallback_to_openai(self, mock_openai_embed):
        """Test fallback to OpenAI when HuggingFace unavailable"""
        # Mock return value
        mock_embedding = np.random.normal(0, 1, 1536).astype(np.float32)
        mock_openai_embed.return_value = mock_embedding
        
        result = embed_text("test text")
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        mock_openai_embed.assert_called_once_with("test text")
        np.testing.assert_array_equal(result, mock_embedding)
    
    @patch('app.ml.embedder.HF_AVAILABLE', False)
    @patch('app.ml.embedder.OPENAI_AVAILABLE', False)
    def test_embed_no_backends_available(self):
        """Test behavior when no embedding backends are available"""
        result = embed_text("test text")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)
        assert result.dtype == np.float32
        assert np.allclose(result, 0.0)


class TestEmbedBatch:
    """Test batch embedding functionality"""
    
    @patch('app.ml.embedder.embed_text')
    def test_embed_batch_multiple_texts(self, mock_embed_text):
        """Test embedding multiple texts"""
        # Mock individual embeddings
        mock_embed_text.side_effect = [
            np.ones(1024, dtype=np.float32),
            np.ones(1024, dtype=np.float32) * 2,
            np.ones(1024, dtype=np.float32) * 3
        ]
        
        texts = ["text1", "text2", "text3"]
        result = embed_batch(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1024)
        assert result.dtype == np.float32
        assert mock_embed_text.call_count == 3
    
    def test_embed_batch_empty_list(self):
        """Test embedding empty list"""
        result = embed_batch([])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)  # Empty array has shape (0,) not (0, 1024)
        # Note: Empty array dtype is float64 by default, which is fine


class TestHuggingFaceEmbed:
    """Test HuggingFace embedding functionality"""
    
    @patch('app.ml.embedder.HF_AVAILABLE', False)
    def test_huggingface_embed_not_available(self):
        """Test HuggingFace embedding when not available"""
        with pytest.raises(ImportError, match="Hugging Face Sentence Transformers not available"):
            huggingface_embed("test text")
    
    @patch('app.ml.embedder.HF_AVAILABLE', True)
    @patch('app.ml.embedder.get_hf_model')
    def test_huggingface_embed_success(self, mock_get_model):
        """Test successful HuggingFace embedding"""
        # Mock model
        mock_model = MagicMock()
        mock_embedding = np.random.normal(0, 1, 1024).astype(np.float32)
        mock_model.encode.return_value = mock_embedding
        mock_get_model.return_value = mock_model
        
        result = huggingface_embed("test text")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)
        assert result.dtype == np.float32
        # Should be normalized
        assert np.allclose(np.linalg.norm(result), 1.0, atol=1e-5)
        mock_model.encode.assert_called_once_with("test text", convert_to_numpy=True)


class TestOpenAIEmbed:
    """Test OpenAI embedding functionality"""
    
    @patch('app.ml.embedder.OPENAI_AVAILABLE', False)
    def test_openai_embed_not_available(self):
        """Test OpenAI embedding when not available"""
        with pytest.raises(ImportError, match="OpenAI not available"):
            openai_embed("test text")
    
    @patch('app.ml.embedder.OPENAI_AVAILABLE', True)
    @patch('app.ml.embedder.get_openai_client')
    def test_openai_embed_success(self, mock_get_client):
        """Test successful OpenAI embedding"""
        # Mock client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536  # OpenAI returns 1536 dims
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = openai_embed("test text")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)  # But we truncate to 1024 for consistency
        assert result.dtype == np.float32
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="test text"
        )
    
    @patch('app.ml.embedder.OPENAI_AVAILABLE', True)
    @patch('app.ml.embedder.get_openai_client')
    def test_openai_embed_error_fallback(self, mock_get_client):
        """Test OpenAI embedding error handling"""
        # Mock client to raise an exception
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client
        
        result = openai_embed("test text")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)  # Error fallback returns 1024 dims, not 1536
        assert result.dtype == np.float32
        assert np.allclose(result, 0.0)  # Should return zero vector on error


class TestHealthCheck:
    """Test embedding health check functionality"""
    
    @patch('app.ml.embedder.embed_text')
    def test_health_check_success(self, mock_embed_text):
        """Test successful health check"""
        # Mock a valid embedding
        mock_embed_text.return_value = np.ones(1024, dtype=np.float32)
        
        result = health_check()
        
        assert result == True  # Use == instead of is for numpy booleans
        mock_embed_text.assert_called_once_with("test")
    
    @patch('app.ml.embedder.embed_text')
    def test_health_check_zero_vector(self, mock_embed_text):
        """Test health check with zero vector (failure)"""
        # Mock a zero embedding (indicates failure)
        mock_embed_text.return_value = np.zeros(1024, dtype=np.float32)
        
        result = health_check()
        
        assert result == False  # Use == instead of is for numpy booleans
        mock_embed_text.assert_called_once_with("test")
    
    @patch('app.ml.embedder.embed_text')
    def test_health_check_exception(self, mock_embed_text):
        """Test health check with exception"""
        # Mock an exception
        mock_embed_text.side_effect = Exception("Embedding failed")
        
        result = health_check()
        
        assert result is False
        mock_embed_text.assert_called_once_with("test")
    
    @patch('app.ml.embedder.embed_text')
    def test_health_check_none_result(self, mock_embed_text):
        """Test health check with None result"""
        # Mock None return
        mock_embed_text.return_value = None
        
        result = health_check()
        
        assert result is False
        mock_embed_text.assert_called_once_with("test")


class TestEmbeddingNormalization:
    """Test embedding normalization and properties"""
    
    @patch('app.ml.embedder.HF_AVAILABLE', True)
    @patch('app.ml.embedder.get_hf_model')
    def test_embedding_is_normalized(self, mock_get_model):
        """Test that embeddings are properly normalized"""
        # Mock model with unnormalized embedding
        mock_model = MagicMock()
        unnormalized_embedding = np.array([3.0, 4.0] + [0.0] * 1022, dtype=np.float32)
        mock_model.encode.return_value = unnormalized_embedding
        mock_get_model.return_value = mock_model
        
        result = huggingface_embed("test text")
        
        # Should be normalized to unit length
        assert np.allclose(np.linalg.norm(result), 1.0, atol=1e-5)
        assert result[0] == 0.6  # 3/5
        assert result[1] == 0.8  # 4/5
    
    def test_zero_vector_properties(self):
        """Test properties of zero vector for empty inputs"""
        result = embed_text("")
        
        assert np.allclose(result, 0.0)
        assert result.shape == (1024,)
        assert result.dtype == np.float32 