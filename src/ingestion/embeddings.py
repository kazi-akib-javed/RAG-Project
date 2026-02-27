import logging
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingModelSingleton:
    """Singleton to ensure embedding model is loaded only once"""

    _instance = None

    @classmethod
    def get_instance(cls) -> HuggingFaceEmbeddings:
        if cls._instance is None:
            logger.info(f"Loading embedding model for the first time: {EMBEDDING_MODEL}")
            try:
                cls._instance = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                logger.info("Embedding model loaded and cached successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        else:
            logger.debug("Returning cached embedding model")
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton — useful for testing"""
        cls._instance = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Get the singleton embedding model instance"""
    return EmbeddingModelSingleton.get_instance()


def embed_text(text: str, model) -> list:
    """Embed a single text string"""
    logger.info("Embedding text query")
    try:
        vector = model.embed_query(text)
        logger.info(f"Embedding generated successfully — size: {len(vector)}")
        return vector
    except Exception as e:
        logger.error(f"Failed to embed text: {e}")
        raise