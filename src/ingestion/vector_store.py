from langchain_community.vectorstores import FAISS
from src.ingestion.embeddings import get_embedding_model

def create_vector_store(chunks, save_path="faiss_db"):
    """Create and save vector store from chunks"""
    
    embedding_model = get_embedding_model()
    
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    
    vector_store.save_local(save_path)
    print(f"Vector store created and saved to {save_path}")
    return vector_store


def load_vector_store(save_path="faiss_db"):
    """Load existing vector store from disk"""
    
    embedding_model = get_embedding_model()
    
    vector_store = FAISS.load_local(
        save_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    
    print(f"Vector store loaded from {save_path}")
    return vector_store