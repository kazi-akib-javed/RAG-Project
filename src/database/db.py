import logging
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_supabase_client() -> Client:
    """Initialize and return Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

    return create_client(url, key)


# ── Session management ────────────────────────────────────────────

def create_session(name: str) -> dict:
    """Create a new chat session"""
    try:
        client = get_supabase_client()
        response = client.table("chat_sessions").insert({"name": name}).execute()
        session = response.data[0]
        logger.info(f"Created new session: {session['id']} — {name}")
        return session

    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise


def get_all_sessions() -> list:
    """Get all chat sessions ordered by newest first"""
    try:
        client = get_supabase_client()
        response = (
            client.table("chat_sessions")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data

    except Exception as e:
        logger.error(f"Failed to fetch sessions: {e}")
        raise


def delete_session(session_id: str) -> None:
    """Delete a chat session and its messages"""
    try:
        client = get_supabase_client()
        client.table("chat_sessions").delete().eq("id", session_id).execute()
        logger.info(f"Deleted session: {session_id}")

    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise


# ── Message management ────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str) -> dict:
    """Save a message to a session"""
    try:
        client = get_supabase_client()
        response = (
            client.table("chat_messages")
            .insert({
                "session_id": session_id,
                "role": role,
                "content": content,
            })
            .execute()
        )
        logger.info(f"Saved {role} message to session {session_id}")
        return response.data[0]

    except Exception as e:
        logger.error(f"Failed to save message: {e}")
        raise


def get_session_messages(session_id: str) -> list:
    """Get all messages for a session ordered by time"""
    try:
        client = get_supabase_client()
        response = (
            client.table("chat_messages")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at")
            .execute()
        )
        return response.data

    except Exception as e:
        logger.error(f"Failed to fetch messages: {e}")
        raise

def rename_session(session_id: str, new_name: str) -> None:
    """Rename a chat session"""
    try:
        client = get_supabase_client()
        client.table("chat_sessions").update(
            {"name": new_name}
        ).eq("id", session_id).execute()
        logger.info(f"Renamed session {session_id} to: {new_name}")

    except Exception as e:
        logger.error(f"Failed to rename session: {e}")
        raise