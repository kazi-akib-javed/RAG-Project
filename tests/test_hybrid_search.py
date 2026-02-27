import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.retrieval.hybrid_search import tokenize, bm25_search, hybrid_search


def make_docs(texts):
    return [Document(page_content=t) for t in texts]


def test_tokenize_basic():
    """Should split text into lowercase tokens"""
    tokens = tokenize("Hello World")
    assert tokens == ["hello", "world"]


def test_tokenize_empty():
    """Should return empty list for empty string"""
    assert tokenize("") == []

def test_bm25_search_returns_k_results():
    """Should return at most k results"""
    docs = make_docs(["salary info", "tax deduction", "pension fund", "gross pay"])
    results = bm25_search("salary", docs, k=2)
    assert len(results) <= 2


def test_bm25_search_relevance():
    """Most relevant doc should rank first"""
    docs = make_docs([
        "the weather is nice today",
        "net salary is 492 EUR",
        "cats and dogs",
    ])
    results = bm25_search("salary", docs, k=1)
    assert "salary" in results[0].page_content.lower()


def test_hybrid_search_deduplicates():
    """Should not return duplicate chunks"""
    docs = make_docs([
        "net salary 492 EUR",
        "gross pay 618 EUR",
        "tax deduction 68 EUR",
    ])

    with patch("src.retrieval.hybrid_search.retrieve_chunks", create=True) as mock_retrieve:
        mock_retrieve.return_value = [docs[0], docs[1]]
        results = hybrid_search("salary", docs, k=3)

    contents = [r.page_content for r in results]
    assert len(contents) == len(set(contents))