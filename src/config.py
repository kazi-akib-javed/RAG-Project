import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Text splitting
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K_RESULTS = 3
FAISS_DB_PATH = "faiss_db"

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Memory
CHAT_HISTORY_LIMIT = 6

# Reranker
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Validation
def validate_config():
    """Raise early if required env vars are missing"""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Copy .env.example to .env and fill in the values."
        )


validate_config()