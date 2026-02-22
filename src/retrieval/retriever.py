from src.ingestion.vector_store import load_vector_store

def get_retriever(save_path="faiss_db", k=3):
    """Load vector store and return retriever"""
    
    vector_store = load_vector_store(save_path)
    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )
    
    print(f"Retriever ready — will return top {k} chunks")
    return retriever


def retrieve_chunks(question: str, save_path="faiss_db", k=3):
    """Retrieve relevant chunks for a question"""
    
    retriever = get_retriever(save_path, k)
    chunks = retriever.invoke(question)
    
    print(f"\nFound {len(chunks)} relevant chunks for: '{question}'")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.page_content[:200])
    
    return chunks