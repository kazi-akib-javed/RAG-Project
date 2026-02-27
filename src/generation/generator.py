import logging
import os
import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY, LLM_MODEL, CHAT_HISTORY_LIMIT

load_dotenv()
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question based only on the context below.
If you don't know the answer from the context, say "I don't know".
If the user refers to something from chat history, use that to understand their question.

Context:
{context}
"""


class LLMFactory:
    """Factory to create LLM instances — easy to swap models"""

    @staticmethod
    def create(provider: str = "groq") -> ChatGroq:
        """Create LLM instance by provider name"""
        logger.info(f"Creating LLM for provider: {provider}")

        if provider == "groq":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is missing from environment variables")
            return ChatGroq(
                model=LLM_MODEL,
                api_key=GROQ_API_KEY,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}. Supported: groq")


def get_llm():
    """Get LLM instance using factory"""
    try:
        llm = LLMFactory.create("groq")
        logger.info("LLM loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        raise


def build_chat_history(messages: list) -> list:
    """Convert last N messages from DB into LangChain message format"""
    recent = messages[-CHAT_HISTORY_LIMIT:]
    history = []
    for msg in recent:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    logger.info(f"Built chat history with {len(history)} messages")
    return history


def generate_answer_stream(question: str, chunks: list, chat_history: list = []):
    """Stream answer token by token"""
    logger.info(f"Streaming answer for question: '{question}'")

    try:
        if not chunks:
            logger.warning("No chunks provided — cannot generate answer")
            yield "I don't know"
            return

        llm = get_llm()
        context = "\n\n".join([chunk.page_content for chunk in chunks])
        logger.debug(f"Context length: {len(context)} characters")

        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        chain = prompt | llm
        for chunk in chain.stream({
            "context": context,
            "chat_history": chat_history,
            "question": question,
        }):
            yield chunk.content
            time.sleep(0.02)

    except Exception as e:
        logger.error(f"Failed to stream answer: {e}")
        raise