"""
Streamlit UI for Swiggy Annual Report RAG Application.

Features:
- PDF upload and processing
- Q&A interface
- Source document display
- Session state management for efficient vectorstore caching
"""

import os
import sys
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG modules
from ingest import build_vectorstore
from rag_pipeline import get_answer_with_vectorstore, load_vectorstore


# Page configuration
st.set_page_config(
    page_title="Swiggy Annual Report - AI Q&A",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .answer-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 5px;
        border-left: 4px solid #007bff;
        margin-bottom: 10px;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables for caching."""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "pdf_loaded" not in st.session_state:
        st.session_state.pdf_loaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []


def load_or_create_vectorstore(pdf_path: str) -> bool:
    """
    Load or create vectorstore for the PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if st.session_state.vectorstore is None:
            with st.spinner("🔄 Building FAISS vector store..."):
                st.session_state.vectorstore = build_vectorstore(pdf_path)
                st.session_state.pdf_loaded = True
            return True
        return True
    except FileNotFoundError as e:
        st.error(f"❌ PDF not found: {str(e)}")
        return False
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return False


def display_answer_section(result: dict):
    """Display the answer and source documents in formatted sections."""
    # Display answer
    st.markdown("### 📝 Answer")
    st.markdown(
        f'<div class="answer-box">{result["answer"]}</div>',
        unsafe_allow_html=True
    )
    
    # Display sources
    if result.get("sources"):
        st.markdown("### 📚 Supporting Context")
        with st.expander(
            f"View {result['num_sources']} retrieved chunks from the document",
            expanded=False
        ):
            for i, source in enumerate(result["sources"], 1):
                with st.container():
                    st.markdown(
                        f'<div class="source-box">'
                        f'<strong>Source {i} - Page {source["page"]}</strong><br/>'
                        f'{source["content"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )


def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Title
    st.title("🍽️ Swiggy Annual Report - AI Q&A Assistant")
    st.markdown("*Powered by RAG, LangChain, and Groq LLM*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Setup")
        
        # PDF upload/selection
        st.subheader("Document Management")
        
        # Option to upload or use existing
        pdf_option = st.radio(
            "Choose PDF source:",
            ["Use Existing FAISS Index", "Upload New PDF"],
            key="pdf_option"
        )
        
        pdf_path = None
        
        if pdf_option == "Upload New PDF":
            uploaded_file = st.file_uploader(
                "Upload Swiggy Annual Report PDF",
                type="pdf"
            )
            
            if uploaded_file is not None:
                # Save uploaded file
                pdf_path = f"swiggy_annual_report_{uploaded_file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✓ PDF saved as {pdf_path}")
        else:
            # Use existing FAISS index
            if os.path.exists("faiss_index"):
                st.info("✓ FAISS index found. Using existing index.")
                pdf_path = "existing"  # Flag to load from FAISS
            else:
                st.warning("⚠️ No FAISS index found. Please upload a PDF first.")
        
        # Process button
        if pdf_path:
            if st.button("🚀 Process Document", use_container_width=True):
                if pdf_path != "existing":
                    success = load_or_create_vectorstore(pdf_path)
                    if success:
                        st.success("✓ Document processed successfully!")
                else:
                    try:
                        with st.spinner("Loading existing FAISS index..."):
                            st.session_state.vectorstore = load_vectorstore()
                            st.session_state.pdf_loaded = True
                        st.success("✓ FAISS index loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error loading FAISS index: {str(e)}")
        
        # API Key status
        st.subheader("API Configuration")
        if os.getenv("GROQ_API_KEY"):
            st.success("✓ GROQ_API_KEY configured")
        else:
            st.error("⚠️ GROQ_API_KEY not set. Add to .env file.")
        
        # Information
        st.markdown("---")
        st.subheader("📖 Information")
        st.markdown("""
        **How to use:**
        1. Upload PDF or use existing FAISS index
        2. Click "Process Document"
        3. Ask questions in the main area
        4. View answers with source citations
        
        **Get Groq API Key:**
        Visit https://console.groq.com
        
        **Download Swiggy Report:**
        https://investors.swiggy.com/annual-reports
        """)
    
    # Main content area
    if not st.session_state.pdf_loaded:
        st.info(
            "👈 Please upload a PDF and click 'Process Document' in the sidebar to get started."
        )
    else:
        st.success("✓ Document ready for queries!")
        
        # Question input
        st.markdown("### 💭 Ask a Question")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "Enter your question about the Swiggy Annual Report:",
                placeholder="e.g., What is Swiggy's revenue growth?",
                label_visibility="collapsed"
            )
        
        with col2:
            search_button = st.button("🔍 Ask", use_container_width=True)
        
        # Process query
        if search_button or query:
            if query and query.strip():
                try:
                    with st.spinner("🤔 Thinking..."):
                        result = get_answer_with_vectorstore(
                            query,
                            st.session_state.vectorstore
                        )
                    
                    display_answer_section(result)
                    
                except ValueError as e:
                    st.markdown(
                        f'<div class="error-box">'
                        f'❌ Validation Error: {str(e)}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.markdown(
                        f'<div class="error-box">'
                        f'❌ Error: {str(e)}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.warning("Please enter a question!")
        
        # Example questions
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "What is Swiggy's business model and revenue streams?",
            "What are the key risks mentioned in the annual report?",
            "How many cities does Swiggy operate in?",
            "What is Swiggy's market share compared to competitors?",
            "What are the financial highlights for the latest year?"
        ]
        
        col_size = 3
        cols = st.columns(col_size)
        for idx, question in enumerate(example_questions):
            with cols[idx % col_size]:
                if st.button(question, use_container_width=True, key=f"example_{idx}"):
                    try:
                        with st.spinner("🤔 Thinking..."):
                            result = get_answer_with_vectorstore(
                                question,
                                st.session_state.vectorstore
                            )
                        display_answer_section(result)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
