import logging
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from src.config import TOP_K_RESULTS

logger = logging.getLogger(__name__)


class RetrievalStrategy(ABC):
    """Abstract retrieval strategy"""

    @abstractmethod
    def retrieve(self, question: str, chunks: list, k: int = TOP_K_RESULTS) -> list:
        pass


class SemanticRetrievalStrategy(RetrievalStrategy):
    """Retrieval using semantic search only"""

    def retrieve(self, question: str, chunks: list, k: int = TOP_K_RESULTS) -> list:
        logger.info("Using semantic retrieval strategy")
        from src.retrieval.retriever import retrieve_chunks
        return retrieve_chunks(question, k=k)


class HybridRetrievalStrategy(RetrievalStrategy):
    """Retrieval using BM25 + semantic search"""

    def retrieve(self, question: str, chunks: list, k: int = TOP_K_RESULTS) -> list:
        logger.info("Using hybrid retrieval strategy")
        from src.retrieval.hybrid_search import hybrid_search
        return hybrid_search(question, chunks, k=k)


class HybridWithRerankingStrategy(RetrievalStrategy):
    """Retrieval using BM25 + semantic + cross-encoder reranking"""

    def retrieve(self, question: str, chunks: list, k: int = TOP_K_RESULTS) -> list:
        logger.info("Using hybrid + reranking retrieval strategy")
        from src.retrieval.hybrid_search import hybrid_search
        from src.retrieval.reranker import rerank_chunks
        results = hybrid_search(question, chunks, k=k)
        return rerank_chunks(question, results, k=k)


class RetrieverContext:
    """Context class that uses a retrieval strategy"""

    def __init__(self, strategy: RetrievalStrategy = None):
        self._strategy = strategy or HybridWithRerankingStrategy()

    def set_strategy(self, strategy: RetrievalStrategy):
        logger.info(f"Switching retrieval strategy to: {strategy.__class__.__name__}")
        self._strategy = strategy

    def retrieve(self, question: str, chunks: list, k: int = TOP_K_RESULTS) -> list:
        return self._strategy.retrieve(question, chunks, k=k)