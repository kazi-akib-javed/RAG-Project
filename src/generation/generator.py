import logging
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, LLM_MODEL

load_dotenv()

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question based only on the context below.
If you don't know the answer from the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:
"""


def get_llm():
    """Load Groq LLM"""
    logger.info(f"Loading LLM: {LLM_MODEL}")

    try:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing from environment variables")

        llm = ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
        )
        logger.info("LLM loaded successfully")
        return llm

    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        raise


def generate_answer(question: str, chunks: list):
    """Generate answer from question and retrieved chunks"""
    logger.info(f"Generating answer for question: '{question}'")

    try:
        if not chunks:
            logger.warning("No chunks provided — cannot generate answer")
            return "I don't know"

        llm = get_llm()
        context = "\n\n".join([chunk.page_content for chunk in chunks])

        logger.debug(f"Context length: {len(context)} characters")

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "question": question,
        })

        logger.info("Answer generated successfully")
        return response.content

    except Exception as e:
        logger.error(f"Failed to generate answer: {e}")
        raise