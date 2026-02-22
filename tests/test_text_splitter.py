from src.ingestion.text_splitter import split_documents
from langchain_core.documents import Document

def test_split_documents():
    docs = [Document(page_content="word " * 200)]
    chunks = split_documents(docs)
    assert len(chunks) > 1
    assert len(chunks[0].page_content) <= 500