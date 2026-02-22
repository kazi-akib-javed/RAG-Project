import logging

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def get_embedding_model():
    """Load a free embedding model from HuggingFace"""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")

    try:
        model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def embed_text(text: str, model):
    """Embed a single text string"""
    logger.info("Embedding text query")

    try:
        vector = model.embed_query(text)
        logger.info(f"Embedding generated successfully — size: {len(vector)}")
        return vector

    except Exception as e:
        logger.error(f"Failed to embed text: {e}")
        raise