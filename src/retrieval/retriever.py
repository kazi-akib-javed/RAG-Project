import logging

from src.ingestion.vector_store import load_vector_store
from src.config import FAISS_DB_PATH, TOP_K_RESULTS

logger = logging.getLogger(__name__)


def get_retriever(save_path=FAISS_DB_PATH, k=TOP_K_RESULTS):
    """Load vector store and return retriever"""
    logger.info(f"Initializing retriever with top_k={k}")

    try:
        vector_store = load_vector_store(save_path)
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        logger.info(f"Retriever ready — will return top {k} chunks")
        return retriever

    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        raise


def retrieve_chunks(question: str, save_path=FAISS_DB_PATH, k=TOP_K_RESULTS):
    """Retrieve relevant chunks for a question"""
    logger.info(f"Retrieving chunks for question: '{question}'")

    try:
        retriever = get_retriever(save_path, k)
        chunks = retriever.invoke(question)

        logger.info(f"Retrieved {len(chunks)} chunks successfully")

        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i+1}: {chunk.page_content[:100]}...")

        return chunks

    except Exception as e:
        logger.error(f"Failed to retrieve chunks for question '{question}': {e}")
        raise