from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """Load a free embedding model from HuggingFace"""
    
    model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    print("Embedding model loaded!")
    return model


def embed_text(text: str, model):
    """Embed a single text string"""
    
    vector = model.embed_query(text)
    print(f"Embedding size: {len(vector)}")
    return vector