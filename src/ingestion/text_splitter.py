import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into smaller chunks"""
    logger.info(
        f"Splitting {len(documents)} documents "
        f"with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
    )

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = splitter.split_documents(documents)

        logger.info(f"Successfully split into {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Failed to split documents: {e}")
        raise