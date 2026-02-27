import logging
from abc import ABC, abstractmethod
from src.database.db import (
    create_session,
    get_all_sessions,
    delete_session,
    save_message,
    get_session_messages,
    rename_session,
    add_session_document,
    get_session_documents,
    remove_session_document,
)

logger = logging.getLogger(__name__)


class SessionRepository(ABC):
    """Abstract repository for chat sessions"""

    @abstractmethod
    def create(self, name: str) -> dict:
        pass

    @abstractmethod
    def get_all(self) -> list:
        pass

    @abstractmethod
    def delete(self, session_id: str) -> None:
        pass

    @abstractmethod
    def rename(self, session_id: str, name: str) -> None:
        pass


class MessageRepository(ABC):
    """Abstract repository for chat messages"""

    @abstractmethod
    def save(self, session_id: str, role: str, content: str) -> dict:
        pass

    @abstractmethod
    def get_all(self, session_id: str) -> list:
        pass


class DocumentRepository(ABC):
    """Abstract repository for session documents"""

    @abstractmethod
    def add(self, session_id: str, document_name: str) -> dict:
        pass

    @abstractmethod
    def get_all(self, session_id: str) -> list:
        pass

    @abstractmethod
    def remove(self, document_id: str) -> None:
        pass


class SupabaseSessionRepository(SessionRepository):
    """Supabase implementation of SessionRepository"""

    def create(self, name: str) -> dict:
        logger.info(f"Creating session: {name}")
        return create_session(name)

    def get_all(self) -> list:
        logger.info("Fetching all sessions")
        return get_all_sessions()

    def delete(self, session_id: str) -> None:
        logger.info(f"Deleting session: {session_id}")
        delete_session(session_id)

    def rename(self, session_id: str, name: str) -> None:
        logger.info(f"Renaming session {session_id} to: {name}")
        rename_session(session_id, name)


class SupabaseMessageRepository(MessageRepository):
    """Supabase implementation of MessageRepository"""

    def save(self, session_id: str, role: str, content: str) -> dict:
        logger.info(f"Saving {role} message to session {session_id}")
        return save_message(session_id, role, content)

    def get_all(self, session_id: str) -> list:
        logger.info(f"Fetching messages for session {session_id}")
        return get_session_messages(session_id)


class SupabaseDocumentRepository(DocumentRepository):
    """Supabase implementation of DocumentRepository"""

    def add(self, session_id: str, document_name: str) -> dict:
        logger.info(f"Adding document {document_name} to session {session_id}")
        return add_session_document(session_id, document_name)

    def get_all(self, session_id: str) -> list:
        logger.info(f"Fetching documents for session {session_id}")
        return get_session_documents(session_id)

    def remove(self, document_id: str) -> None:
        logger.info(f"Removing document {document_id}")
        remove_session_document(document_id)