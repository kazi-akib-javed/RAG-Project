import logging
import os

from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)


def load_pdf(file_path: str):
    """Load a PDF file and return list of pages"""
    logger.info(f"Loading PDF: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        logger.info(f"Successfully loaded {len(pages)} pages from {file_path}")
        return pages

    except Exception as e:
        logger.error(f"Failed to load PDF {file_path}: {e}")
        raise


def load_all_pdfs(folder_path: str):
    """Load all PDFs from a folder"""
    logger.info(f"Scanning folder for PDFs: {folder_path}")

    if not os.path.exists(folder_path):
        logger.error(f"Folder not found: {folder_path}")
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_documents = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        logger.warning(f"No PDF files found in {folder_path}")
        return all_documents

    for file in pdf_files:
        file_path = os.path.join(folder_path, file)
        docs = load_pdf(file_path)
        all_documents.extend(docs)

    logger.info(f"Total pages loaded: {len(all_documents)} from {len(pdf_files)} PDFs")
    return all_documents