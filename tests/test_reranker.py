import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.retrieval.reranker import rerank_chunks


def make_docs(texts):
    return [Document(page_content=t) for t in texts]


def test_rerank_returns_top_k():
    """Should return at most k chunks"""
    docs = make_docs(["doc1", "doc2", "doc3", "doc4", "doc5"])

    with patch("src.retrieval.reranker.get_reranker") as mock:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.1, 0.8, 0.3, 0.5]
        mock.return_value = mock_model

        results = rerank_chunks("question", docs, k=3)

    assert len(results) == 3


def test_rerank_orders_by_score():
    """Highest scored chunk should be first"""
    docs = make_docs(["low score doc", "high score doc", "mid score doc"])

    with patch("src.retrieval.reranker.get_reranker") as mock:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        mock.return_value = mock_model

        results = rerank_chunks("question", docs, k=3)

    assert results[0].page_content == "high score doc"


def test_rerank_empty_chunks():
    """Should handle empty chunk list"""
    with patch("src.retrieval.reranker.get_reranker") as mock:
        mock_model = MagicMock()
        mock_model.predict.return_value = []
        mock.return_value = mock_model

        results = rerank_chunks("question", [], k=3)

    assert results == []