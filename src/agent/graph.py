import logging
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    check_retrieval_needed,
    retrieve,
    grade_documents,
    rewrite_query,
    generate,
    generate_from_memory,
)

logger = logging.getLogger(__name__)

MAX_REWRITES = 2


def decide_retrieval(state: AgentState) -> str:
    """Edge: should we retrieve or answer from memory?"""
    if state["needs_retrieval"]:
        return "retrieve"
    return "generate_from_memory"


def decide_after_grading(state: AgentState) -> str:
    """Edge: are documents relevant or do we need to rewrite?"""
    if state["documents_relevant"]:
        return "generate"
    if state["rewrite_count"] >= MAX_REWRITES:
        logger.warning("Max rewrites reached — generating with available docs")
        return "generate"
    return "rewrite_query"


def build_rag_graph():
    """Build and compile the agentic RAG graph"""
    graph = StateGraph(AgentState)

    # add nodes
    graph.add_node("check_retrieval_needed", check_retrieval_needed)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("generate_from_memory", generate_from_memory)

    # set entry point
    graph.set_entry_point("check_retrieval_needed")

    # add edges
    graph.add_conditional_edges(
        "check_retrieval_needed",
        decide_retrieval,
        {
            "retrieve": "retrieve",
            "generate_from_memory": "generate_from_memory",
        }
    )

    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        }
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("generate_from_memory", END)

    return graph.compile()


# compile once at module level
rag_graph = build_rag_graph()