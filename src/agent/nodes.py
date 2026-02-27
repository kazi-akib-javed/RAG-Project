import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from src.agent.state import AgentState

from src.ingestion.vector_store import load_chunks
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_chunks

from src.generation.generator import get_llm  # import instead of redefining

logger = logging.getLogger(__name__)


# ── Node 1: Check if retrieval is needed ─────────────────────────
def check_retrieval_needed(state: AgentState) -> AgentState:
    """Decide if we need to retrieve documents or can answer from memory"""
    logger.info("Checking if retrieval is needed")

    question = state["question"]
    chat_history = state["chat_history"]

    # if no chat history, always retrieve
    if not chat_history:
        logger.info("No chat history — retrieval needed")
        return {**state, "needs_retrieval": True}

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("""
You are deciding whether a question requires searching documents or can be answered from conversation history.

Chat History:
{chat_history}

Question: {question}

Answer with ONLY "YES" if documents need to be searched, or "NO" if the question can be answered from chat history alone.
Examples of NO: follow-up questions, clarifications, questions about previous answers
Examples of YES: new topics, specific document facts, numbers, names
""")

    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in chat_history[-4:]
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "question": question,
        "chat_history": history_text
    })

    needs_retrieval = "YES" in result.upper()
    logger.info(f"Retrieval needed: {needs_retrieval}")
    return {**state, "needs_retrieval": needs_retrieval}


# ── Node 2: Retrieve documents ────────────────────────────────────
def retrieve(state: AgentState) -> AgentState:
    """Retrieve relevant chunks using hybrid search + reranking"""
    logger.info(f"Retrieving documents for: {state['question']}")

    if not state.get("session_id"):
        logger.error("No session_id in state")
        return {**state, "documents": []}

    try:
        from src.retrieval.strategy import RetrieverContext
        all_chunks = load_chunks(state["session_id"])
        retriever = RetrieverContext()  # uses HybridWithRerankingStrategy by default
        chunks = retriever.retrieve(state["question"], all_chunks)
        logger.info(f"Retrieved {len(chunks)} chunks")
        return {**state, "documents": chunks}
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {**state, "documents": []}

# ── Node 3: Grade documents ───────────────────────────────────────
def grade_documents(state: AgentState) -> AgentState:
    """Check if retrieved documents are actually relevant"""
    logger.info("Grading document relevance")

    question = state["question"]
    documents = state["documents"]

    if not documents:
        logger.warning("No documents to grade — skipping grading")
        return {**state, "documents_relevant": False}

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("""
You are checking if retrieved documents contain information that could help answer a question.
Be generous — if the documents contain ANY related information, answer YES.

Question: {question}

Retrieved content:
{context}

Do these documents contain information related to the question?
Answer with ONLY "YES" or "NO".
""")

    context = "\n\n".join([doc.page_content[:300] for doc in documents])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "context": context})

    relevant = "NO" not in result.upper()
    logger.info(f"Grading result: '{result.strip()}' — relevant: {relevant}")
    return {**state, "documents_relevant": relevant}


# ── Node 4: Rewrite query ─────────────────────────────────────────
def rewrite_query(state: AgentState) -> AgentState:
    """Rewrite the question to get better retrieval results"""
    logger.info(f"Rewriting query (attempt {state['rewrite_count'] + 1})")

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("""
The original question failed to retrieve relevant documents.
Rewrite it to be more specific and likely to find relevant information.

Original question: {question}

Rewritten question (return ONLY the rewritten question, nothing else):
""")

    chain = prompt | llm | StrOutputParser()
    new_question = chain.invoke({"question": state["question"]})
    logger.info(f"Rewritten question: {new_question}")

    return {
        **state,
        "question": new_question.strip(),
        "rewrite_count": state["rewrite_count"] + 1
    }


# ── Node 5: Generate answer from documents ────────────────────────
def generate(state: AgentState) -> AgentState:
    """Generate answer from retrieved documents and chat history"""
    logger.info("Generating answer from documents")

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a precise document assistant.
RULES:
- Answer ONLY using exact information from the context below
- NEVER invent or estimate numbers, percentages, or values
- If the exact answer is not in the context, say: "I could not find this in the document"
- Quote specific values directly from the context when possible
- Do not mix up values from different parts of the context
- If the user refers to something from chat history, use that to understand their question

Context:
{context}"""),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])

    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "chat_history": state["chat_history"],
        "question": state["question"],
    })

    logger.info("Answer generated from documents")
    return {**state, "answer": answer}


# ── Node 6: Generate from memory ─────────────────────────────────
def generate_from_memory(state: AgentState) -> AgentState:
    """Generate answer from chat history without retrieval"""
    logger.info("Generating answer from chat history (no retrieval)")

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer based on the conversation history."),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "chat_history": state["chat_history"],
        "question": state["question"],
    })

    logger.info("Answer generated from memory")
    return {**state, "answer": answer}