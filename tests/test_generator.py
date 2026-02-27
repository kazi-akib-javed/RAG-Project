import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from src.generation.generator import build_chat_history, generate_answer_stream


def test_build_chat_history_empty():
    """Should return empty list for no messages"""
    result = build_chat_history([])
    assert result == []


def test_build_chat_history_converts_roles():
    """Should convert user/assistant roles to LangChain messages"""
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    result = build_chat_history(messages)
    assert isinstance(result[0], HumanMessage)
    assert isinstance(result[1], AIMessage)
    assert result[0].content == "hello"
    assert result[1].content == "hi there"


def test_build_chat_history_limit():
    """Should respect CHAT_HISTORY_LIMIT"""
    messages = [
        {"role": "user", "content": f"message {i}"}
        for i in range(20)
    ]
    result = build_chat_history(messages)
    from src.config import CHAT_HISTORY_LIMIT
    assert len(result) <= CHAT_HISTORY_LIMIT


def test_generate_answer_stream_no_chunks():
    """Should yield I dont know when no chunks provided"""
    result = list(generate_answer_stream("question", []))
    assert "".join(result) == "I don't know"


def test_generate_answer_stream_calls_llm():
    """Should call LLM with correct inputs"""
    docs = [Document(page_content="salary is 492 EUR")]

    with patch("src.generation.generator.get_llm") as mock_llm:
        mock_chain = MagicMock()
        mock_chain.stream.return_value = [MagicMock(content="492 EUR")]
        mock_llm.return_value.__or__ = MagicMock(return_value=mock_chain)

        list(generate_answer_stream("what is salary?", docs))

        mock_llm.assert_called_once()