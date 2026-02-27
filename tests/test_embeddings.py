import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.embeddings import get_embedding_model, embed_text


def test_get_embedding_model_success():
    """Should load embedding model successfully"""
    with patch("src.ingestion.embeddings.HuggingFaceEmbeddings") as mock:
        mock.return_value = MagicMock()
        model = get_embedding_model()
        assert model is not None
        mock.assert_called_once()


def test_get_embedding_model_failure():
    """Should raise exception if model fails to load"""
    with patch("src.ingestion.embeddings.HuggingFaceEmbeddings") as mock:
        mock.side_effect = Exception("Model not found")
        with pytest.raises(Exception):
            get_embedding_model()


def test_embed_text_returns_vector():
    """Should return a list of floats"""
    mock_model = MagicMock()
    mock_model.embed_query.return_value = [0.1, 0.2, 0.3]

    vector = embed_text("hello world", mock_model)

    assert isinstance(vector, list)
    assert len(vector) == 3
    assert all(isinstance(v, float) for v in vector)


def test_embed_text_calls_model():
    """Should call embed_query with correct text"""
    mock_model = MagicMock()
    mock_model.embed_query.return_value = [0.1] * 384

    embed_text("test text", mock_model)

    mock_model.embed_query.assert_called_once_with("test text")