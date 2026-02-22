from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    """Load Groq LLM"""
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    print("LLM loaded!")
    return llm


def generate_answer(question: str, chunks: list):
    """Generate answer from question and retrieved chunks"""
    
    llm = get_llm()
    
    # combine chunks into one context
    context = "\n\n".join([chunk.page_content for chunk in chunks])
    
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer the question based only on the context below.
    If you don't know the answer from the context, say "I don't know".
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """)
    
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    return response.content