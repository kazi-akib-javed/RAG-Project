import logging
import os
import shutil

import streamlit as st
from dotenv import load_dotenv

from src.ingestion.document_loader import load_pdf, load_all_pdfs
from src.ingestion.text_splitter import split_documents
from src.ingestion.vector_store import create_vector_store
from src.retrieval.retriever import retrieve_chunks
from src.generation.generator import build_chat_history, generate_answer_stream
from src.database.db import (
    create_session,
    get_all_sessions,
    delete_session,
    save_message,
    get_session_messages,
    rename_session,
)
from src.retrieval.hybrid_search import hybrid_search
from src.ingestion.vector_store import load_chunks

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")


def setup_vector_store():
    """Ingest documents if vector store doesn't exist"""
    if not os.path.exists("faiss_db"):
        with st.spinner("Ingesting documents..."):
            pages = load_all_pdfs("data")
            chunks = split_documents(pages)
            create_vector_store(chunks)


# ── Session state defaults ─────────────────────────────────────────
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "current_session_name" not in st.session_state:
    st.session_state.current_session_name = None


# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 RAG Chatbot")
    st.divider()

    # new chat button
    if st.button("➕ New Chat", use_container_width=True):
        session = create_session("New Chat")
        st.session_state.current_session_id = session["id"]
        st.session_state.current_session_name = session["name"]
        st.rerun()

    st.divider()

    # upload document
    st.subheader("📁 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=False, max_upload_size=10)

    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            if os.path.exists("faiss_db"):
                shutil.rmtree("faiss_db")
                logger.info("Old vector store cleared")

            save_path = os.path.join("data", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pages = load_pdf(save_path)
            chunks = split_documents(pages)
            create_vector_store(chunks)

        st.success(f"✅ {uploaded_file.name} ready!")
        logger.info(f"Document uploaded: {uploaded_file.name}")

    st.divider()

    # chat history list
    st.subheader("💬 Chat History")
    sessions = get_all_sessions()

    for session in sessions:
        col1, col2 = st.columns([8, 2])

        with col1:
            if st.button(
                session["name"],
                key=f"session_{session['id']}",
                use_container_width=True,
            ):
                st.session_state.current_session_id = session["id"]
                st.session_state.current_session_name = session["name"]
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"delete_{session['id']}"):
                delete_session(session["id"])
                if st.session_state.current_session_id == session["id"]:
                    st.session_state.current_session_id = None
                    st.session_state.current_session_name = None
                st.rerun()


# ── Main chat area ─────────────────────────────────────────────────
if st.session_state.current_session_id is None:
    st.title("🤖 RAG Chatbot")
    st.info("Click ➕ New Chat in the sidebar to start a conversation!")

else:
    st.title(f"💬 {st.session_state.current_session_name}")

    setup_vector_store()

    # load and display messages from database
    messages = get_session_messages(st.session_state.current_session_id)
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat input
    if question := st.chat_input("Ask a question about your documents..."):

        # show and save user message
        with st.chat_message("user"):
            st.markdown(question)
        save_message(st.session_state.current_session_id, "user", question)

        # rename chat to first question if still default
        if st.session_state.current_session_name == "New Chat":
            # truncate to 40 chars if question is too long
            new_name = question[:40] + "..." if len(question) > 40 else question
            rename_session(st.session_state.current_session_id, new_name)
            st.session_state.current_session_name = new_name
            logger.info(f"Session renamed to: {new_name}")

        # generate and save answer
        with st.chat_message("assistant"):
            try:
                all_messages = get_session_messages(
                    st.session_state.current_session_id
                )
                previous_messages = all_messages[:-1]
                chat_history = build_chat_history(previous_messages)

                all_chunks = load_chunks()
                chunks = hybrid_search(question, all_chunks)

                # stream the response
                answer = st.write_stream(
                    generate_answer_stream(question, chunks, chat_history)
                )

                # display source chunks
                with st.expander("📄 View Sources"):
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Source {i+1}**")
                        st.caption(
                            f"📁 {chunk.metadata.get('source', 'Unknown')} "
                            f"| Page {chunk.metadata.get('page', '?') + 1}"
                        )
                        st.info(chunk.page_content[:300] + "...")
                        st.divider()

                # save complete answer after streaming
                save_message(
                    st.session_state.current_session_id,
                    "assistant",
                    answer,
                )
                logger.info("Streamed answer saved to database")

            except Exception as e:
                st.error("Something went wrong. Please try again.")
                logger.error(f"Error streaming answer: {e}")
        st.rerun()