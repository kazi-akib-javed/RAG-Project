import pytest
from langchain_core.documents import Document
from src.ingestion.text_splitter import split_documents


def test_split_documents_basic():
    """Should split long document into multiple chunks"""
    docs = [Document(page_content="word " * 200)]
    chunks = split_documents(docs)
    assert len(chunks) > 1
    assert len(chunks[0].page_content) <= 500


def test_split_documents_short():
    """Short document should remain as single chunk"""
    docs = [Document(page_content="short text")]
    chunks = split_documents(docs)
    assert len(chunks) == 1


def test_split_documents_empty():
    """Empty document list should return empty list"""
    chunks = split_documents([])
    assert chunks == []


def test_split_documents_custom_chunk_size():
    """Should respect custom chunk size"""
    docs = [Document(page_content="word " * 100)]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)
    for chunk in chunks:
        assert len(chunk.page_content) <= 100


def test_split_documents_preserves_metadata():
    """Should preserve document metadata in chunks"""
    docs = [Document(
        page_content="word " * 200,
        metadata={"source": "test.pdf", "page": 1}
    )]
    chunks = split_documents(docs)
    assert chunks[0].metadata["source"] == "test.pdf"
    assert chunks[0].metadata["page"] == 1