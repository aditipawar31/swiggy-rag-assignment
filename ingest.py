"""
PDF ingestion and vector store creation for Swiggy Annual Report RAG application.

This module handles:
- Loading PDF files using PyMuPDF
- Extracting and cleaning text with page metadata
- Splitting text into chunks
- Generating embeddings
- Creating and saving FAISS vector store
"""

import os
import fitz  # PyMuPDF
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def extract_text_from_pdf(pdf_path: str) -> list[Document]:
    """
    Extract text from PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects with text and page metadata
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF cannot be read
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    documents = []
    
    try:
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            
            # Clean text: remove extra whitespace and special characters
            text = clean_text(text)
            
            if text.strip():  # Only add non-empty pages
                doc = Document(
                    page_content=text,
                    metadata={"page": page_num + 1, "source": pdf_path}
                )
                documents.append(doc)
        
        pdf_document.close()
        print(f"✓ Extracted text from {len(documents)} pages")
        
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
    
    return documents


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and special characters.
    
    Args:
        text: Raw text from PDF
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Remove common header/footer artifacts
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.isdigit():  # Skip empty or page number lines
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)


def split_into_chunks(documents: list[Document], 
                     chunk_size: int = 800, 
                     chunk_overlap: int = 100) -> list[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks from documents")
    
    return chunks


def build_vectorstore(pdf_path: str, 
                     faiss_index_dir: str = "faiss_index",
                     force_rebuild: bool = False) -> FAISS:
    """
    Build FAISS vector store from PDF.
    
    Pipeline:
    1. Extract text from PDF with page metadata
    2. Split into chunks (800 chars, 100 overlap)
    3. Generate embeddings using sentence-transformers
    4. Create and save FAISS index
    
    Args:
        pdf_path: Path to PDF file
        faiss_index_dir: Directory to save FAISS index
        force_rebuild: If True, rebuild index even if it exists
        
    Returns:
        FAISS vector store object
        
    Raises:
        FileNotFoundError: If PDF not found
        Exception: If processing fails
    """
    # Check if index already exists and force_rebuild is False
    if os.path.exists(faiss_index_dir) and not force_rebuild:
        print(f"✓ Loading existing FAISS index from {faiss_index_dir}")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.load_local(
            faiss_index_dir, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    
    print("Building vector store...")
    
    # Step 1: Extract text from PDF
    documents = extract_text_from_pdf(pdf_path)
    
    # Step 2: Split into chunks
    chunks = split_into_chunks(documents)
    
    # Step 3: Generate embeddings
    print("Generating embeddings using sentence-transformers...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Step 4: Create FAISS index
    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Step 5: Save index to disk
    os.makedirs(faiss_index_dir, exist_ok=True)
    vectorstore.save_local(faiss_index_dir)
    print(f"✓ FAISS index saved to {faiss_index_dir}")
    
    return vectorstore


if __name__ == "__main__":
    # Example usage
    pdf_path = "swiggy_annual_report.pdf"
    vectorstore = build_vectorstore(pdf_path)
    print("Vector store ready!")
