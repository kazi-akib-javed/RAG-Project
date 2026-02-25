from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """State that flows through the RAG agent graph"""
    question: str
    chat_history: List[BaseMessage]
    documents: List[Document]
    answer: str
    needs_retrieval: bool
    documents_relevant: bool
    rewrite_count: int
    session_id: str