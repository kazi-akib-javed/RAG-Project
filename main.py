from src.ingestion.document_loader import load_all_pdfs
from src.ingestion.text_splitter import split_documents
from src.ingestion.vector_store import create_vector_store, load_vector_store
from src.retrieval.retriever import retrieve_chunks
from src.generation.generator import generate_answer
from dotenv import load_dotenv
import os

load_dotenv()

def ingest_documents():
    """Load, chunk and store all PDFs from data folder"""
    print("\n--- Ingesting documents ---")
    pages = load_all_pdfs("data")
    chunks = split_documents(pages)
    create_vector_store(chunks)
    print("--- Ingestion complete ---\n")


def chat():
    """Main chat loop"""
    
    # check if vector store exists
    if not os.path.exists("faiss_db"):
        print("No vector store found. Ingesting documents first...")
        ingest_documents()
    
    print("\n🤖 RAG Chatbot ready! Type 'quit' to exit\n")
    
    while True:
        question = input("You: ").strip()
        
        if not question:
            continue
        
        if question.lower() == "quit":
            print("Goodbye!")
            break
        
        # retrieve and generate
        chunks = retrieve_chunks(question)
        answer = generate_answer(question, chunks)
        
        print(f"\nBot: {answer}\n")
        print("-" * 50)


if __name__ == "__main__":
    chat()