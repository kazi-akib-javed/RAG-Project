import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


# ── Singleton tests ───────────────────────────────────────────────
def test_singleton_returns_same_instance():
    """Should return same instance on multiple calls"""
    from src.ingestion.embeddings import EmbeddingModelSingleton
    EmbeddingModelSingleton.reset()

    with patch("src.ingestion.embeddings.HuggingFaceEmbeddings") as mock:
        mock.return_value = MagicMock()
        instance1 = EmbeddingModelSingleton.get_instance()
        instance2 = EmbeddingModelSingleton.get_instance()
        assert instance1 is instance2
        mock.assert_called_once()

    EmbeddingModelSingleton.reset()


def test_singleton_reset_creates_new_instance():
    """Should create new instance after reset"""
    from src.ingestion.embeddings import EmbeddingModelSingleton
    EmbeddingModelSingleton.reset()

    with patch("src.ingestion.embeddings.HuggingFaceEmbeddings") as mock:
        mock.side_effect = [MagicMock(), MagicMock()]
        instance1 = EmbeddingModelSingleton.get_instance()
        EmbeddingModelSingleton.reset()
        instance2 = EmbeddingModelSingleton.get_instance()
        assert instance1 is not instance2

    EmbeddingModelSingleton.reset()


# ── Factory tests ─────────────────────────────────────────────────
def test_factory_creates_groq_llm():
    """Should create Groq LLM for groq provider"""
    from src.generation.generator import LLMFactory
    with patch("src.generation.generator.ChatGroq") as mock:
        mock.return_value = MagicMock()
        llm = LLMFactory.create("groq")
        assert llm is not None
        mock.assert_called_once()


def test_factory_raises_for_unknown_provider():
    """Should raise ValueError for unknown provider"""
    from src.generation.generator import LLMFactory
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMFactory.create("unknown_provider")


# ── Repository tests ──────────────────────────────────────────────
def test_session_repository_create():
    """Should call create_session"""
    from src.database.repository import SupabaseSessionRepository
    repo = SupabaseSessionRepository()
    with patch("src.database.repository.create_session") as mock:
        mock.return_value = {"id": "123", "name": "Test"}
        result = repo.create("Test")
        mock.assert_called_once_with("Test")
        assert result["name"] == "Test"


def test_message_repository_save():
    """Should call save_message"""
    from src.database.repository import SupabaseMessageRepository
    repo = SupabaseMessageRepository()
    with patch("src.database.repository.save_message") as mock:
        mock.return_value = {"id": "msg1"}
        repo.save("session1", "user", "hello")
        mock.assert_called_once_with("session1", "user", "hello")


def test_document_repository_add():
    """Should call add_session_document"""
    from src.database.repository import SupabaseDocumentRepository
    repo = SupabaseDocumentRepository()
    with patch("src.database.repository.add_session_document") as mock:
        mock.return_value = {"id": "doc1"}
        repo.add("session1", "file.pdf")
        mock.assert_called_once_with("session1", "file.pdf")


# ── Strategy tests ────────────────────────────────────────────────
def test_strategy_default_is_hybrid_with_reranking():
    """Default strategy should be HybridWithReranking"""
    from src.retrieval.strategy import RetrieverContext, HybridWithRerankingStrategy
    ctx = RetrieverContext()
    assert isinstance(ctx._strategy, HybridWithRerankingStrategy)


def test_strategy_can_be_switched():
    """Should switch strategy at runtime"""
    from src.retrieval.strategy import (
        RetrieverContext,
        SemanticRetrievalStrategy,
        HybridRetrievalStrategy,
    )
    ctx = RetrieverContext()
    ctx.set_strategy(SemanticRetrievalStrategy())
    assert isinstance(ctx._strategy, SemanticRetrievalStrategy)
    ctx.set_strategy(HybridRetrievalStrategy())
    assert isinstance(ctx._strategy, HybridRetrievalStrategy)


def test_strategy_retrieve_calls_correct_method():
    """Strategy retrieve should call underlying implementation"""
    from src.retrieval.strategy import RetrieverContext
    from langchain_core.documents import Document

    mock_strategy = MagicMock()
    mock_strategy.retrieve.return_value = [Document(page_content="test")]

    ctx = RetrieverContext(strategy=mock_strategy)
    result = ctx.retrieve("question", [], k=3)

    mock_strategy.retrieve.assert_called_once_with("question", [], k=3)
    assert len(result) == 1