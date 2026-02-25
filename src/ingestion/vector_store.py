import json
import logging
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

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

        # save raw chunks to disk for BM25
        chunks_path = os.path.join(save_path, "chunks.json")
        chunks_data = [
            {"page_content": c.page_content, "metadata": c.metadata}
            for c in chunks
        ]
        with open(chunks_path, "w") as f:
            json.dump(chunks_data, f)

        logger.info(f"Vector store and chunks saved to {save_path}")
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


def load_chunks(session_id: str) -> list:
    """Load raw chunks for a specific session"""
    save_path = get_session_store_path(session_id)
    logger.info(f"Loading chunks for session: {session_id}")

    try:
        chunks_path = os.path.join(save_path, "chunks.json")
        with open(chunks_path, "r") as f:
            chunks_data = json.load(f)

        chunks = [
            Document(
                page_content=c["page_content"],
                metadata=c["metadata"]
            )
            for c in chunks_data
        ]

        logger.info(f"Loaded {len(chunks)} chunks for session")
        return chunks

    except FileNotFoundError:
        logger.error(f"No chunks found for session {session_id}")
        raise

    except Exception as e:
        logger.error(f"Failed to load chunks: {e}")
        raise


def add_to_vector_store(chunks, session_id: str):
    """Add new chunks to session-specific vector store"""
    save_path = get_session_store_path(session_id)
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Adding {len(chunks)} chunks to vector store: {save_path}")

    try:
        embedding_model = get_embedding_model()

        if os.path.exists(os.path.join(save_path, "index.faiss")):
            vector_store = FAISS.load_local(
                save_path,
                embedding_model,
                allow_dangerous_deserialization=True,
            )
            new_store = FAISS.from_documents(
                documents=chunks,
                embedding=embedding_model,
            )
            vector_store.merge_from(new_store)
            logger.info("Merged new chunks into existing vector store")
        else:
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embedding_model,
            )
            logger.info("Created new vector store for session")

        vector_store.save_local(save_path)

        # update chunks.json
        chunks_path = os.path.join(save_path, "chunks.json")
        existing_chunks = []
        if os.path.exists(chunks_path):
            with open(chunks_path, "r") as f:
                existing_chunks = json.load(f)

        new_chunks_data = [
            {"page_content": c.page_content, "metadata": c.metadata}
            for c in chunks
        ]
        all_chunks = existing_chunks + new_chunks_data

        with open(chunks_path, "w") as f:
            json.dump(all_chunks, f)

        logger.info(f"Session vector store now has {len(all_chunks)} chunks")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to add to vector store: {e}")
        raise


def get_session_store_path(session_id: str) -> str:
    """Get vector store path for a specific session"""
    return f"faiss_db/{session_id}"