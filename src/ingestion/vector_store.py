import logging

from langchain_community.vectorstores import FAISS

from src.ingestion.embeddings import get_embedding_model
from src.config import FAISS_DB_PATH

logger = logging.getLogger(__name__)


def create_vector_store(chunks, save_path=FAISS_DB_PATH):
    """Create and save vector store from chunks"""
    logger.info(f"Creating vector store from {len(chunks)} chunks")

    try:
        embedding_model = get_embedding_model()
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model,
        )
        vector_store.save_local(save_path)
        logger.info(f"Vector store created and saved to {save_path}")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise


def load_vector_store(save_path=FAISS_DB_PATH):
    """Load existing vector store from disk"""
    logger.info(f"Loading vector store from {save_path}")

    try:
        embedding_model = get_embedding_model()
        vector_store = FAISS.load_local(
            save_path,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Vector store loaded successfully from {save_path}")
        return vector_store

    except FileNotFoundError:
        logger.error(f"Vector store not found at {save_path}. Run ingestion first.")
        raise

    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise