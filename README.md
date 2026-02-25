# 🤖 RAG Chatbot

A production-grade Retrieval Augmented Generation (RAG) chatbot built from scratch. Upload multiple PDFs per session and chat across all of them using hybrid search, re-ranking, conversation memory, and real-time streaming — powered by open-source models and free APIs.

---

## 📸 Demo

> Create a chat → Upload PDFs → Ask questions → Get accurate answers with source citations

---

## 🏗️ Architecture

```
PDF Upload (multiple files supported)
    ↓
Document Loader (PyPDFLoader)
    ↓
Text Splitter (RecursiveCharacterTextSplitter)
    ↓
Embeddings (HuggingFace all-MiniLM-L6-v2)
    ↓
Per-Session Vector Store (FAISS)
    ↓
User Question
    ↓
Hybrid Search (BM25 + Semantic)
    ↓
Cross-Encoder Re-ranking
    ↓
Re-ranked Chunks + Chat History (last N messages)
    ↓
LLM (Groq Llama 3.1 8B)
    ↓
Streamed Answer + Source Citations
```

---

## ✨ Features

- **Multi-Document Support** — upload multiple PDFs per session and chat across all of them simultaneously
- **Per-Session Vector Stores** — each chat session has its own isolated FAISS index, preventing data leakage between sessions
- **Hybrid Search** — combines BM25 keyword search with semantic vector search for better retrieval
- **Cross-Encoder Re-ranking** — re-ranks retrieved chunks for higher precision before passing to LLM
- **Conversation Memory** — passes last 6 messages to LLM so it understands follow-up questions
- **Streaming Responses** — answers appear word by word in real time
- **Source Citations** — shows exactly which file, page, and chunk text was used to generate the answer
- **Multi-Session Chat** — create, switch between, and delete multiple named chat sessions like ChatGPT
- **Persistent History** — all conversations saved to Supabase PostgreSQL database
- **RAGAS Evaluation** — automated pipeline to measure retrieval and generation quality

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq (Llama 3.1 8B Instant) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS (per-session isolated) |
| Keyword Search | BM25 (rank-bm25) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L6-v2 |
| Database | Supabase (PostgreSQL) |
| Framework | LangChain |
| Evaluation | RAGAS |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
rag-chatbot/
├── data/                          # PDF documents (gitignored)
├── faiss_db/                      # Per-session vector stores (gitignored)
│   └── {session_id}/              # Isolated FAISS index per session
│       ├── index.faiss
│       ├── index.pkl
│       └── chunks.json
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py     # PDF loading
│   │   ├── text_splitter.py       # Chunking
│   │   ├── embeddings.py          # HuggingFace embeddings
│   │   └── vector_store.py        # FAISS store (per-session)
│   ├── retrieval/
│   │   ├── retriever.py           # Semantic search
│   │   ├── hybrid_search.py       # BM25 + semantic hybrid
│   │   └── reranker.py            # Cross-encoder re-ranking
│   ├── generation/
│   │   └── generator.py           # LLM + prompt + streaming + memory
│   ├── database/
│   │   └── db.py                  # Supabase CRUD operations
│   ├── evaluation/
│   │   ├── evaluate.py            # RAGAS evaluation pipeline
│   │   └── test_data.json         # Test questions and ground truths
│   └── config.py                  # Centralized configuration
├── tests/
│   └── test_text_splitter.py      # Unit tests
├── app.py                         # Streamlit UI
├── main.py                        # CLI interface
├── conftest.py                    # Pytest configuration
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/kazi-akib-javed/RAG-Project.git
cd RAG-Project
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Fill in your `.env` file:

```
GROQ_API_KEY=your_groq_api_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

- Get a free Groq API key at [console.groq.com](https://console.groq.com)
- Get a free Supabase project at [supabase.com](https://supabase.com)

### 5. Set up Supabase database

Run this SQL in your Supabase SQL Editor:

```sql
CREATE TABLE chat_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE chat_messages (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE session_documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    document_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 6. Run the app

```bash
streamlit run app.py
```

Or use the CLI:

```bash
python main.py
```

---

## 💬 Usage

1. Open the app at `http://localhost:8501`
2. Click **➕ New Chat** in the sidebar
3. Upload one or more PDF documents
4. Start asking questions!

### Example questions

```
"What is the net salary?"
"What are the tax deductions?"
"Is that before or after tax?"        ← follow-up with memory
"What is RANL?"                        ← exact keyword search via BM25
"Compare October vs December salary"   ← multi-document query
```

---

## 📊 RAGAS Evaluation

Run the evaluation pipeline to measure RAG quality:

```bash
python -m src.evaluation.evaluate
```

### Latest evaluation results

| Metric | Score | Description |
|---|---|---|
| Faithfulness | 1.0 | LLM answers are grounded in retrieved context — no hallucination |
| Answer Relevancy | 0.977 | Answers directly address the questions asked |
| Context Precision | 0.542 → improved | Cross-encoder re-ranking filters irrelevant chunks before LLM |
| Context Recall | 1.000 | All necessary information was successfully retrieved |

> **Note:** Context Precision improved after adding cross-encoder re-ranking. Irrelevant chunks retrieved by hybrid search are filtered out before reaching the LLM, reducing noise in generated answers.

---

## ⚙️ Configuration

All key parameters are in `src/config.py`:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"                    # HuggingFace embedding model
LLM_MODEL = "llama-3.1-8b-instant"                       # Groq LLM model
CHUNK_SIZE = 500                                           # Characters per chunk
CHUNK_OVERLAP = 50                                         # Overlap between chunks
TOP_K_RESULTS = 3                                          # Chunks retrieved per query
CHAT_HISTORY_LIMIT = 6                                     # Messages passed to LLM
FAISS_DB_PATH = "faiss_db"                                # Base vector store path
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"   # Re-ranking model
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## ⚠️ Known Limitations

- **No OCR support** — scanned PDFs (images of text) won't work, only text-based PDFs
- **English-focused embeddings** — non-English documents will have lower quality retrieval
- **Local file storage** — uploaded PDFs stored locally; re-upload needed after server restart
- **FAISS doesn't scale** — not suitable for thousands of documents; consider Pinecone or pgvector
- **No user authentication** — single user only, no multi-user support
- **Groq rate limits** — free tier has request limits; heavy usage may hit rate limits

---

## 🔮 Future Improvements

- [x] Add re-ranking with a cross-encoder model
- [x] Support multiple documents per session
- [x] Agentic RAG with LangGraph — let LLM decide when to retrieve
- [x] Corrective RAG (CRAG) — re-query if retrieved chunks are not relevant enough
- [ ] Add OCR for scanned PDFs
- [ ] FastAPI backend + React frontend
- [ ] Redis query caching for repeated questions
- [ ] LangSmith tracing for production observability
- [ ] Replace FAISS with Pinecone for scalability
- [ ] Add user authentication with Supabase Auth
- [ ] Store documents in Supabase Storage for persistence
- [ ] Support more file types (DOCX, TXT, CSV)
- [ ] LLM fine-tuning with LoRA/QLoRA for domain-specific accuracy

---

## 📝 Engineering Decisions

**Why FAISS over ChromaDB?**
ChromaDB had Python 3.14 compatibility issues. FAISS is battle-tested at scale at Meta and works reliably across Python versions.

**Why per-session vector stores?**
Sharing a single vector store across sessions caused data leakage — answers from one session's documents appeared in another. Each session now gets its own isolated FAISS index under `faiss_db/{session_id}/`, deleted automatically when the session is deleted.

**Why Groq over OpenAI?**
Groq is free, extremely fast due to custom LPU hardware, and Llama 3.1 8B is surprisingly capable for document Q&A tasks.

**Why HuggingFace embeddings over OpenAI?**
`all-MiniLM-L6-v2` runs locally, is completely free, and produces 384-dimensional vectors good enough for most document Q&A use cases.

**Why hybrid search?**
Semantic search finds conceptually similar chunks but misses exact keyword matches (e.g. wage codes like `RANL`, `LSTL`). BM25 catches these. Combining both gives best of both worlds — semantic understanding plus exact keyword precision.

**Why cross-encoder re-ranking?**
Bi-encoder similarity search (FAISS) is fast but imprecise — it scores query and document independently. A cross-encoder reads both together, giving much more accurate relevance scores. We use it as a second-pass filter after hybrid search retrieves candidates.

**Why Last-N memory over full history?**
Llama 3.1 8B has an 8192 token context window. Passing full history risks hitting the limit. Last 6 messages gives enough context for natural conversation without token overflow.

---

## 🤝 Contributing

Pull requests are welcome! For major changes please open an issue first.

---

## 📄 License

MIT