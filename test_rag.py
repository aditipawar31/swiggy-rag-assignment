"""
Test script to demonstrate RAG pipeline usage without Streamlit.
This shows how to use the RAG system programmatically.

Run: python test_rag.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG modules
from ingest import build_vectorstore
from rag_pipeline import get_answer_with_vectorstore, load_vectorstore


def test_rag_pipeline():
    """Test the complete RAG pipeline."""
    
    print("=" * 70)
    print("🍽️  SWIGGY ANNUAL REPORT - RAG Q&A SYSTEM TEST")
    print("=" * 70)
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: GROQ_API_KEY not found in .env file")
        print("Please create .env with: GROQ_API_KEY=your_key_here")
        return
    print("✓ GROQ_API_KEY configured\n")
    
    # Define PDF path
    pdf_path = "swiggy_annual_report.pdf"
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: {pdf_path} not found")
        print("Please download from: https://investors.swiggy.com/annual-reports")
        print("And save as: swiggy_annual_report.pdf")
        return
    print(f"✓ PDF file found: {pdf_path}\n")
    
    # Step 1: Build or load vectorstore
    print("-" * 70)
    print("STEP 1: Building/Loading Vector Store")
    print("-" * 70)
    try:
        vectorstore = build_vectorstore(pdf_path, force_rebuild=False)
        print("✓ Vector store ready!\n")
    except Exception as e:
        print(f"❌ Error building vectorstore: {str(e)}")
        return
    
    # Step 2: Test queries
    print("-" * 70)
    print("STEP 2: Testing Queries")
    print("-" * 70)
    
    test_queries = [
        "What is Swiggy's business model?",
        "How many cities does Swiggy operate in?",
        "What are Swiggy's revenue streams?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i}: {query}")
        print(f"{'─' * 70}")
        
        try:
            result = get_answer_with_vectorstore(query, vectorstore)
            
            # Display answer
            print(f"\n📝 ANSWER:\n{result['answer']}\n")
            
            # Display sources
            print(f"📚 SOURCES ({result['num_sources']} chunks retrieved):")
            for j, source in enumerate(result['sources'], 1):
                print(f"\n  [{j}] Page {source['page']}:")
                print(f"      {source['content'][:150]}...")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 70)
    print("✓ TEST COMPLETE")
    print("=" * 70)
    print("\nTo use the interactive UI, run: streamlit run app.py")
    print("=" * 70)


def test_direct_answer(query: str):
    """
    Get answer for a specific query.
    
    Usage example:
        python test_rag.py "What is Swiggy?"
    """
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: GROQ_API_KEY not found in .env file")
        return
    
    pdf_path = "swiggy_annual_report.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: {pdf_path} not found")
        return
    
    try:
        vectorstore = load_vectorstore()
    except:
        try:
            vectorstore = build_vectorstore(pdf_path)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return
    
    try:
        result = get_answer_with_vectorstore(query, vectorstore)
        
        print("\n" + "=" * 70)
        print("SWIGGY ANNUAL REPORT Q&A")
        print("=" * 70)
        print(f"\nQ: {query}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources ({result['num_sources']} chunks):")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n  [{i}] Page {source['page']}:")
            print(f"      {source['content'][:100]}...")
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Direct query from command line
        query = " ".join(sys.argv[1:])
        test_direct_answer(query)
    else:
        # Run full test pipeline
        test_rag_pipeline()
