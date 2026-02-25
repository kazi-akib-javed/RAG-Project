import logging
import os
import shutil

import streamlit as st
from dotenv import load_dotenv

from src.ingestion.document_loader import load_pdf
from src.ingestion.text_splitter import split_documents
from src.ingestion.vector_store import add_to_vector_store
from src.generation.generator import build_chat_history
from src.database.db import (
    add_session_document,
    create_session,
    get_all_sessions,
    delete_session,
    save_message,
    get_session_messages,
    rename_session,
    remove_session_document,
    get_session_documents,
)
from src.agent.graph import rag_graph
from src.agent.state import AgentState

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# ── Session state defaults ─────────────────────────────────────────
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "current_session_name" not in st.session_state:
    st.session_state.current_session_name = None

if "current_documents" not in st.session_state:
    st.session_state.current_documents = []

if "session_docs" not in st.session_state:
    st.session_state.session_docs = []

if "sessions_cache" not in st.session_state:
    st.session_state.sessions_cache = get_all_sessions()

if "messages_cache" not in st.session_state:
    st.session_state.messages_cache = []

if "processing_upload" not in st.session_state:
    st.session_state.processing_upload = False

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 RAG Chatbot")
    st.divider()

    # new chat button
    if st.button("➕ New Chat", use_container_width=True):
        session = create_session("New Chat")
        st.session_state.current_session_id = session["id"]
        st.session_state.current_session_name = session["name"]
        st.session_state.session_docs = []
        st.session_state.current_documents = []
        st.session_state.messages_cache = []
        st.session_state.sessions_cache = get_all_sessions()
        st.rerun()

    st.divider()

    # upload document
    st.subheader("📁 Upload Document")
    uploaded_files = st.file_uploader(
        "Add PDFs to this chat",
        type="pdf",
        accept_multiple_files=True
    )

    # show documents in current session
    if st.session_state.current_session_id:
        docs = st.session_state.session_docs
        if docs:
            st.caption(f"📄 {len(docs)} document(s) in this chat:")
            for doc in docs:
                col1, col2 = st.columns([8, 2])
                with col1:
                    st.caption(f"• {doc['document_name']}")
                with col2:
                    if st.button("✕", key=f"remove_doc_{doc['id']}"):
                        remove_session_document(doc["id"])
                        st.session_state.session_docs = get_session_documents(
                            st.session_state.current_session_id
                        )
                        st.session_state.current_documents = [
                            d["document_name"] for d in st.session_state.session_docs
                        ]
                        st.rerun()
        else:
            st.warning("⚠️ No documents in this chat")

    if uploaded_files and not st.session_state.get("processing_upload"):
        if st.session_state.current_session_id is None:
            st.warning("Please create a new chat first!")
        else:
            existing = [d["document_name"] for d in st.session_state.session_docs]
            new_files = [f for f in uploaded_files if f.name not in existing]

            if new_files:
                st.session_state.processing_upload = True
                for uploaded_file in new_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        save_path = os.path.join("data", uploaded_file.name)
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        pages = load_pdf(save_path)
                        chunks = split_documents(pages)
                        add_to_vector_store(chunks, st.session_state.current_session_id)

                        add_session_document(
                            st.session_state.current_session_id,
                            uploaded_file.name,
                        )
                        st.session_state.current_documents.append(uploaded_file.name)

                    st.success(f"✅ {uploaded_file.name} added!")
                    logger.info(f"Document added: {uploaded_file.name}")

                st.session_state.session_docs = get_session_documents(
                    st.session_state.current_session_id
                )
                st.session_state.processing_upload = False
                st.rerun()
            else:
                for f in uploaded_files:
                    st.warning(f"'{f.name}' already added!")

    st.divider()

    # chat history list
    st.subheader("💬 Chat History")
    for session in st.session_state.sessions_cache:
        col1, col2 = st.columns([8, 2])

        with col1:
            if st.button(
                session["name"],
                key=f"session_{session['id']}",
                use_container_width=True,
            ):
                st.session_state.current_session_id = session["id"]
                st.session_state.current_session_name = session["name"]
                docs = get_session_documents(session["id"])
                st.session_state.session_docs = docs
                st.session_state.current_documents = [
                    d["document_name"] for d in docs
                ]
                st.session_state.messages_cache = get_session_messages(session["id"])
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"delete_{session['id']}"):
                session_store_path = f"faiss_db/{session['id']}"
                if os.path.exists(session_store_path):
                    shutil.rmtree(session_store_path)
                delete_session(session["id"])
                if st.session_state.current_session_id == session["id"]:
                    st.session_state.current_session_id = None
                    st.session_state.current_session_name = None
                    st.session_state.current_documents = []
                    st.session_state.session_docs = []
                    st.session_state.messages_cache = []
                st.session_state.sessions_cache = get_all_sessions()
                st.rerun()


# ── Main chat area ─────────────────────────────────────────────────
if st.session_state.current_session_id is None:
    st.title("🤖 RAG Chatbot")
    st.info("Click ➕ New Chat in the sidebar to start a conversation!")

else:
    st.title(f"💬 {st.session_state.current_session_name}")

    # display messages from cache
    for message in st.session_state.messages_cache:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat input
    if question := st.chat_input("Ask a question about your documents..."):

        # show and save user message
        with st.chat_message("user"):
            st.markdown(question)
        save_message(st.session_state.current_session_id, "user", question)
        st.session_state.messages_cache.append({
            "role": "user",
            "content": question
        })

        # rename chat to first question if still default
        if st.session_state.current_session_name == "New Chat":
            new_name = question[:40] + "..." if len(question) > 40 else question
            rename_session(st.session_state.current_session_id, new_name)
            st.session_state.current_session_name = new_name
            st.session_state.sessions_cache = get_all_sessions()
            logger.info(f"Session renamed to: {new_name}")

        # generate and save answer
        with st.chat_message("assistant"):
            try:
                # build chat history from cache
                previous_messages = st.session_state.messages_cache[:-1]
                chat_history = build_chat_history(previous_messages)

                if not st.session_state.current_documents:
                    st.warning("Please upload a document first!")
                else:
                    with st.spinner("Thinking..."):
                        # run agentic RAG graph
                        result = rag_graph.invoke(AgentState(
                            question=question,
                            chat_history=chat_history,
                            documents=[],
                            answer="",
                            needs_retrieval=False,
                            documents_relevant=False,
                            rewrite_count=0,
                            session_id=st.session_state.current_session_id,
                        ))

                    answer = result["answer"]
                    st.markdown(answer)

                    # show source citations if retrieval was used
                    if result["needs_retrieval"] and result["documents"]:
                        with st.expander("📄 View Sources"):
                            for i, chunk in enumerate(result["documents"]):
                                st.markdown(f"**Source {i+1}**")
                                st.caption(
                                    f"📁 {chunk.metadata.get('source', 'Unknown')} "
                                    f"| Page {chunk.metadata.get('page', '?') + 1}"
                                )
                                st.info(chunk.page_content[:300] + "...")
                                st.divider()
                    elif not result["needs_retrieval"]:
                        st.caption("💭 Answered from conversation memory")

                    save_message(
                        st.session_state.current_session_id,
                        "assistant",
                        answer,
                    )
                    st.session_state.messages_cache.append({
                        "role": "assistant",
                        "content": answer
                    })
                    logger.info("Agent answer saved to database")

            except Exception as e:
                st.error("Something went wrong. Please try again.")
                logger.error(f"Error running agent: {e}")