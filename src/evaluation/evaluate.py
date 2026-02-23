import json
import logging

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.retrieval.hybrid_search import hybrid_search
from src.ingestion.vector_store import load_chunks
from src.generation.generator import generate_answer_stream
from src.config import GROQ_API_KEY, LLM_MODEL, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def load_test_data(path: str = "src/evaluation/test_data.json") -> list:
    """Load test questions and ground truths"""
    logger.info(f"Loading test data from {path}")
    with open(path, "r") as f:
        return json.load(f)


def run_evaluation():
    """Run RAGAS evaluation on the RAG pipeline"""
    logger.info("Starting RAGAS evaluation")

    # load test data
    test_data = load_test_data()

    # load chunks for retrieval
    all_chunks = load_chunks()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in test_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info(f"Evaluating question: '{question}'")

        # retrieve chunks
        chunks = hybrid_search(question, all_chunks)
        context_texts = [chunk.page_content for chunk in chunks]

        # generate answer
        answer = ""
        for token in generate_answer_stream(question, chunks):
            answer += token

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(ground_truth)

        logger.info(f"Answer: {answer[:100]}...")

    # build dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # wrap LLM and embeddings for RAGAS
    llm = LangchainLLMWrapper(
        ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    # run evaluation
    logger.info("Running RAGAS metrics...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    # print results
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    df = results.to_pandas()
    print(df.to_string())
    print("\nAverage Scores:")
    print(f"  Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"  Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")
    print(f"  Context Precision: {df['context_precision'].mean():.3f}")
    print(f"  Context Recall:    {df['context_recall'].mean():.3f}")
    print("=" * 50)

    # save results to file
    df.to_csv("src/evaluation/results.csv", index=False)
    logger.info("Results saved to src/evaluation/results.csv")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()