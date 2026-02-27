import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.agent.state import AgentState


def make_state(**kwargs):
    defaults = {
        "question": "what is the salary?",
        "chat_history": [],
        "documents": [],
        "answer": "",
        "needs_retrieval": False,
        "documents_relevant": False,
        "rewrite_count": 0,
        "session_id": "test-session-id",
    }
    return {**defaults, **kwargs}


def test_check_retrieval_needed_no_history():
    """Should always retrieve when no chat history"""
    from src.agent.nodes import check_retrieval_needed
    state = make_state(chat_history=[])
    result = check_retrieval_needed(state)
    assert result["needs_retrieval"] is True


def test_rewrite_query_increments_count():
    """Should increment rewrite count"""
    from src.agent.nodes import rewrite_query

    with patch("src.agent.nodes.get_llm") as mock_llm:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "rewritten question"
        mock_llm.return_value.__or__ = MagicMock(return_value=mock_chain)

        state = make_state(rewrite_count=0)
        with patch("src.agent.nodes.StrOutputParser"):
            with patch("src.agent.nodes.ChatPromptTemplate"):
                result = rewrite_query(state)

        assert result["rewrite_count"] == 1


def test_grade_documents_no_docs():
    """Should mark as not relevant when no documents"""
    from src.agent.nodes import grade_documents
    state = make_state(documents=[])
    result = grade_documents(state)
    assert result["documents_relevant"] is False


def test_graph_has_correct_nodes():
    """Graph should have all required nodes"""
    from src.agent.graph import build_rag_graph
    graph = build_rag_graph()
    node_names = list(graph.nodes.keys())
    assert "check_retrieval_needed" in node_names
    assert "retrieve" in node_names
    assert "grade_documents" in node_names
    assert "rewrite_query" in node_names
    assert "generate" in node_names
    assert "generate_from_memory" in node_names