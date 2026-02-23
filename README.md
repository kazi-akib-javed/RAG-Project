# 🤖 RAG Chatbot

A production-grade Retrieval Augmented Generation (RAG) chatbot built from scratch. Chat with your PDF documents using hybrid search, conversation memory, and real-time streaming — powered by open-source models and free APIs.

---

## 📸 Demo

> Upload a PDF → Ask questions → Get accurate answers with source citations

---

## 🏗️ Architecture

```
PDF Upload
    ↓
Document Loader (PyPDFLoader)
    ↓
Text Splitter (RecursiveCharacterTextSplitter)
    ↓
Embeddings (HuggingFace all-MiniLM-L6-v2)
    ↓
Vector Store (FAISS)
    ↓
User Question
    ↓
Hybrid Search (BM25 + Semantic)
    ↓
Re-ranked Chunks + Chat History
    ↓
LLM (Groq Llama 3.1 8B)
    ↓
Streamed Answer + Source Citations
```

---

## ✨ Features

- **Hybrid Search** — combines BM25 keyword search with semantic vector search for better retrieval
- **Conversation Memory** — passes last 6 messages to LLM so it understands follow-up questions
- **Streaming Responses** — answers appear word by word in real time
- **Source Citations** — shows exactly which chunks and page numbers were used to generate the answer
- **Multi-Session Chat** — create, switch between, and delete multiple chat sessions like ChatGPT
- **Persistent History** — all conversations saved to Supabase PostgreSQL database
- **Document Management** — each chat session remembers which document it was using
- **RAGAS Evaluation** — automated pipeline to measure retrieval and generation quality

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq (Llama 3.1 8B Instant) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Keyword Search | BM25 (rank-bm25) |
| Database | Supabase (PostgreSQL) |
| Framework | LangChain |
| Evaluation | RAGAS |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
rag-chatbot/
├── data/                          # PDF documents (gitignored)
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py     # PDF loading
│   │   ├── text_splitter.py       # Chunking
│   │   ├── embeddings.py          # HuggingFace embeddings
│   │   └── vector_store.py        # FAISS store
│   ├── retrieval/
│   │   ├── retriever.py           # Semantic search
│   │   └── hybrid_search.py       # BM25 + semantic hybrid
│   ├── generation/
│   │   └── generator.py           # LLM + prompt + streaming
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
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
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
    document_name TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE chat_messages (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
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
3. Upload a PDF document
4. Start asking questions!

### Example questions

```
"What is the net salary?"
"What are the tax deductions?"
"Is that before or after tax?"   ← follow-up with memory
"What is RANL?"                  ← exact keyword search via BM25
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
| Faithfulness | 1.0 | LLM answers are grounded in retrieved context |
| Answer Relevancy | 0.977 | Answers directly address the questions |
| Context Precision | 0.542 | Proportion of retrieved chunks that are relevant |
| Context Recall | 1.000 | All necessary information was retrieved |

---

## ⚙️ Configuration

All key parameters are in `src/config.py`:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # HuggingFace embedding model
LLM_MODEL = "llama-3.1-8b-instant"      # Groq LLM model
CHUNK_SIZE = 500                          # Characters per chunk
CHUNK_OVERLAP = 50                        # Overlap between chunks
TOP_K_RESULTS = 3                         # Chunks retrieved per query
CHAT_HISTORY_LIMIT = 6                    # Messages passed to LLM
FAISS_DB_PATH = "faiss_db"               # Vector store location
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
- **Single document per session** — each chat session is tied to one PDF at a time
- **FAISS doesn't scale** — not suitable for thousands of documents; consider Pinecone or pgvector for large scale
- **No user authentication** — single user only, no multi-user support
- **File persistence** — uploaded PDFs are stored locally; if the server restarts, files need to be re-uploaded

---

## 🔮 Future Improvements

- [ ] Add re-ranking with a cross-encoder model
- [ ] Support multiple documents per session
- [ ] Add OCR for scanned PDFs
- [ ] Replace FAISS with Pinecone for scalability
- [ ] Add user authentication with Supabase Auth
- [ ] Store documents in Supabase Storage for persistence
- [ ] Add document deletion management
- [ ] Support more file types (DOCX, TXT, CSV)

---

## 📝 Engineering Decisions

**Why FAISS over ChromaDB?**
ChromaDB had Python 3.14 compatibility issues. FAISS is battle-tested at scale at Meta and works reliably across Python versions.

**Why Groq over OpenAI?**
Groq is free, extremely fast due to custom LPU hardware, and Llama 3.1 8B is surprisingly capable for document Q&A tasks.

**Why HuggingFace embeddings over OpenAI?**
`all-MiniLM-L6-v2` runs locally, is completely free, and produces 384-dimensional vectors that are good enough for most single-document use cases.

**Why Last-N memory over full history?**
Llama 3.1 8B has an 8192 token context window. Passing full history risks hitting the limit. Last 6 messages gives enough context for natural conversation without token overflow.

**Why hybrid search?**
Semantic search finds conceptually similar chunks but misses exact keyword matches (e.g. wage codes like `RANL`, `LSTL`). BM25 catches these. Combining both gives best of both worlds.

---

## 🤝 Contributing

Pull requests are welcome! For major changes please open an issue first.

---

## 📄 License

MIT