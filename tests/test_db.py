import pytest
from unittest.mock import patch, MagicMock
from src.database.db import (
    create_session,
    get_all_sessions,
    delete_session,
    save_message,
    get_session_messages,
    rename_session,
)


def mock_supabase():
    """Helper to create mock Supabase client"""
    mock = MagicMock()
    return mock


def test_create_session_success():
    """Should create session and return data"""
    with patch("src.database.db.get_supabase_client") as mock_client:
        mock = mock_supabase()
        mock.table.return_value.insert.return_value.execute.return_value.data = [
            {"id": "123", "name": "Test Chat"}
        ]
        mock_client.return_value = mock

        result = create_session("Test Chat")
        assert result["name"] == "Test Chat"
        assert result["id"] == "123"


def test_get_all_sessions_returns_list():
    """Should return list of sessions"""
    with patch("src.database.db.get_supabase_client") as mock_client:
        mock = mock_supabase()
        mock.table.return_value.select.return_value.order.return_value.execute.return_value.data = [
            {"id": "1", "name": "Chat 1"},
            {"id": "2", "name": "Chat 2"},
        ]
        mock_client.return_value = mock

        result = get_all_sessions()
        assert len(result) == 2


def test_save_message_success():
    """Should save message and return data"""
    with patch("src.database.db.get_supabase_client") as mock_client:
        mock = mock_supabase()
        mock.table.return_value.insert.return_value.execute.return_value.data = [
            {"id": "msg1", "role": "user", "content": "hello"}
        ]
        mock_client.return_value = mock

        result = save_message("session1", "user", "hello")
        assert result["role"] == "user"
        assert result["content"] == "hello"


def test_rename_session_calls_update():
    """Should call update with new name"""
    with patch("src.database.db.get_supabase_client") as mock_client:
        mock = mock_supabase()
        mock_client.return_value = mock

        rename_session("session1", "New Name")

        mock.table.return_value.update.assert_called_once_with({"name": "New Name"})