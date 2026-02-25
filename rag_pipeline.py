"""
RAG (Retrieval-Augmented Generation) pipeline for Swiggy Annual Report Q&A.

This module handles:
- Loading the FAISS vector store
- Setting up the retrieval chain with Groq LLM
- Processing user queries and retrieving answers with sources
"""

import os
from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


def load_vectorstore(faiss_index_dir: str = "faiss_index") -> FAISS:
    """
    Load FAISS vector store from disk.
    
    Args:
        faiss_index_dir: Directory where FAISS index is stored
        
    Returns:
        FAISS vector store object
        
    Raises:
        FileNotFoundError: If FAISS index not found
    """
    if not os.path.exists(faiss_index_dir):
        raise FileNotFoundError(
            f"FAISS index not found at {faiss_index_dir}. "
            f"Please run ingest.py first."
        )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.load_local(
        faiss_index_dir, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print(f"✓ Loaded FAISS index from {faiss_index_dir}")
    return vectorstore


def create_rag_chain(vectorstore: FAISS):
    """
    Create a RAG chain with Groq LLM using LCEL (LangChain Expression Language).
    
    Components:
    - Retriever: FAISS similarity search with k=4
    - LLM: ChatGroq with llama3-8b-8192
    - Prompt: Custom template ensuring grounded answers
    
    Args:
        vectorstore: FAISS vector store
        
    Returns:
        RAG chain (LCEL runnable)
        
    Raises:
        ValueError: If GROQ_API_KEY not set
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in .env file or environment."
        )
    
    # Create retriever (top 4 similar chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,  # Low temperature for more grounded answers
        api_key=api_key,
        max_tokens=512
    )
    
    # Create prompt template
    prompt_template = """You are an AI assistant that answers questions strictly based on the provided context from the Swiggy Annual Report.
If the answer is not present in the context, say "I don't have enough information in the document to answer this."
Do NOT make up any information.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Format retrieved documents into context
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Create LCEL chain: question -> retriever -> format -> prompt -> llm
    chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
    )
    
    print("✓ RAG chain created with Groq LLM")
    return chain, retriever


def get_answer(query: str, 
               vectorstore: Optional[FAISS] = None,
               faiss_index_dir: str = "faiss_index") -> dict:
    """
    Get answer to a user query using RAG pipeline.
    
    Process:
    1. Load vectorstore if not provided
    2. Create RAG chain
    3. Retrieve relevant chunks from FAISS
    4. Generate answer using Groq LLM
    5. Extract source documents with page numbers
    
    Args:
        query: User's question
        vectorstore: Optional pre-loaded FAISS vectorstore
        faiss_index_dir: Directory of FAISS index
        
    Returns:
        Dictionary with keys:
            - "answer": Final answer from LLM
            - "sources": List of dicts with "content" and "page" keys
            
    Raises:
        ValueError: If GROQ_API_KEY not set or query empty
        FileNotFoundError: If FAISS index not found
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Load vectorstore if not provided
    if vectorstore is None:
        vectorstore = load_vectorstore(faiss_index_dir)
    
    # Create RAG chain
    chain, retriever = create_rag_chain(vectorstore)
    
    # Get answer
    try:
        # Run the chain
        answer = chain.invoke(query)
        
        # Retrieve source documents
        source_docs = retriever.invoke(query)
        
        # Format sources with page numbers
        sources = []
        for doc in source_docs:
            source_dict = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "page": doc.metadata.get("page", "Unknown"),
                "full_content": doc.page_content
            }
            sources.append(source_dict)
        
        return {
            "answer": answer.content if hasattr(answer, 'content') else str(answer),
            "sources": sources,
            "num_sources": len(sources)
        }
        
    except Exception as e:
        raise Exception(f"Error generating answer: {str(e)}")


def get_answer_with_vectorstore(query: str, vectorstore: FAISS) -> dict:
    """
    Get answer using a pre-loaded vectorstore (optimized for session state).
    
    Args:
        query: User's question
        vectorstore: Pre-loaded FAISS vectorstore
        
    Returns:
        Dictionary with "answer" and "sources"
    """
    return get_answer(query, vectorstore=vectorstore)


if __name__ == "__main__":
    # Example usage
    try:
        # Load vectorstore
        vectorstore = load_vectorstore()
        
        # Test query
        test_query = "What is Swiggy's business model?"
        result = get_answer_with_vectorstore(test_query, vectorstore)
        
        print("\n" + "="*50)
        print(f"Query: {test_query}")
        print("="*50)
        print(f"Answer: {result['answer']}")
        print(f"\nSources ({result['num_sources']} chunks):")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n[Source {i} - Page {source['page']}]")
            print(source['content'])
            
    except Exception as e:
        print(f"Error: {str(e)}")
