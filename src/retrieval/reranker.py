import logging
from sentence_transformers import CrossEncoder
from src.config import TOP_K_RESULTS

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def get_reranker():
    """Load cross-encoder reranker model"""
    logger.info(f"Loading reranker model: {RERANKER_MODEL}")
    try:
        model = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")
        raise


def rerank_chunks(question: str, chunks: list, k: int = TOP_K_RESULTS) -> list:
    """Rerank chunks using cross-encoder and return top k"""
    logger.info(f"Reranking {len(chunks)} chunks for question: '{question}'")

    try:
        model = get_reranker()

        # create pairs of (question, chunk_text)
        pairs = [[question, chunk.page_content] for chunk in chunks]

        # score each pair
        scores = model.predict(pairs)

        # sort chunks by score descending
        scored_chunks = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True
        )

        # return top k chunks
        top_chunks = [chunk for _, chunk in scored_chunks[:k]]

        logger.info(f"Reranking complete — returning top {len(top_chunks)} chunks")

        # log scores for visibility
        for i, (score, chunk) in enumerate(scored_chunks[:k]):
            logger.debug(f"Chunk {i+1} score: {score:.4f} — {chunk.page_content[:80]}...")

        return top_chunks

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise