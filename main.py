import logging
import os

from dotenv import load_dotenv

from src.ingestion.document_loader import load_all_pdfs
from src.ingestion.text_splitter import split_documents
from src.ingestion.vector_store import create_vector_store, load_vector_store
from src.retrieval.retriever import retrieve_chunks
from src.generation.generator import generate_answer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_documents():
    """Load, chunk and store all PDFs from data folder"""
    logger.info("Starting document ingestion pipeline")
    pages = load_all_pdfs("data")
    chunks = split_documents(pages)
    create_vector_store(chunks)
    logger.info("Document ingestion complete")


def chat():
    """Main chat loop"""

    if not os.path.exists("faiss_db"):
        logger.warning("No vector store found. Running ingestion first...")
        ingest_documents()

    logger.info("RAG Chatbot is ready")
    print("\n🤖 RAG Chatbot ready! Type 'quit' to exit\n")

    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() == "quit":
                logger.info("User exited the chatbot")
                print("Goodbye!")
                break

            logger.info(f"User question: {question}")
            chunks = retrieve_chunks(question)
            answer = generate_answer(question, chunks)

            logger.info("Answer generated successfully")
            print(f"\nBot: {answer}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            logger.info("Chatbot interrupted by user")
            print("\nGoodbye!")
            break

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print("Something went wrong. Please try again.")


if __name__ == "__main__":
    chat()