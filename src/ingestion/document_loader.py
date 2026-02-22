from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf(file_path: str):
    """Load a PDF file and return list of pages"""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    print(f"Loaded {len(pages)} pages from {file_path}")
    return pages


def load_all_pdfs(folder_path: str):
    """Load all PDFs from a folder"""
    
    all_documents = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            docs = load_pdf(file_path)
            all_documents.extend(docs)
    
    print(f"Total pages loaded: {len(all_documents)}")
    return all_documents