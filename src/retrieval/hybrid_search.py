import logging
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.config import TOP_K_RESULTS

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list:
    """Simple whitespace tokenizer"""
    return text.lower().split()


def bm25_search(query: str, chunks: list, k: int = TOP_K_RESULTS) -> list:
    """Search chunks using BM25 keyword search"""
    logger.info(f"Running BM25 search for: '{query}'")

    try:
        # tokenize all chunks
        tokenized_chunks = [tokenize(chunk.page_content) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        # search
        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # get top k indices sorted by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        results = [chunks[i] for i in top_indices]
        logger.info(f"BM25 returned {len(results)} chunks")
        return results

    except Exception as e:
        logger.error(f"BM25 search failed: {e}")
        raise


def hybrid_search(query: str, chunks: list, k: int = TOP_K_RESULTS) -> list:
    """Combine semantic and BM25 results"""
    logger.info(f"Running hybrid search for: '{query}'")

    try:
        from src.retrieval.retriever import retrieve_chunks

        # semantic search results
        semantic_results = retrieve_chunks(query, k=k)

        # bm25 search results
        bm25_results = bm25_search(query, chunks, k=k)

        # combine and deduplicate
        seen = set()
        combined = []

        for doc in semantic_results + bm25_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)

        logger.info(f"Hybrid search returned {len(combined)} unique chunks")
        return combined[:k * 2]  # return top k*2 unique chunks

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise