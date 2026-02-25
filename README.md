# 🍽️ Swiggy Annual Report - RAG Q&A Assistant

A complete **Retrieval-Augmented Generation (RAG)** application for answering questions about the Swiggy Annual Report using AI. Built with LangChain, FAISS, sentence-transformers embeddings, and Groq LLM.

## 📋 Features

✅ **PDF Processing** - Load and chunk the Swiggy Annual Report  
✅ **Semantic Search** - FAISS vector store for fast retrieval  
✅ **Free LLM** - Groq API (llama-3.3-70b-versatile) with free tier  
✅ **No API Keys for Embeddings** - Using open-source sentence-transformers  
✅ **Persistent Storage** - FAISS index saved to disk (no re-processing needed)  
✅ **Source Citations** - See which page each answer comes from  
✅ **Grounded Responses** - LLM only answers from provided context  
✅ **Beautiful UI** - Streamlit interface with session state management  

## 🏗️ Architecture

```
PDF Document
     ↓
 [PyMuPDF Extraction]
     ↓
 [Text Cleaning]
     ↓
 [RecursiveCharacterTextSplitter]
 (800 chars, 100 overlap)
     ↓
 [sentence-transformers Embeddings]
 (all-MiniLM-L6-v2)
     ↓
 [FAISS Vector Store]
 (Persisted to disk)
     ↓
 [Similarity Search Retriever] ← User Query
 (k=4 top chunks)
     ↓
 [Groq LLM (llama-3.3-70b-versatile)]
 (With prompt template for grounded answers)
     ↓
 Final Answer + Source Pages
```

## 📂 Project Structure

```
swiggy-rag/
├── app.py                      # Streamlit UI
├── rag_pipeline.py             # RAG core logic
├── ingest.py                   # PDF processing
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── .env                        # Environment variables (create this)
├── faiss_index/                # Persisted FAISS index (auto-created)
└── swiggy_annual_report.pdf   # PDF file (not included, download separately)
```

## 🚀 Quick Start

### 1. Clone or Setup Project

```bash
cd c:\Users\Admin\Desktop\Assignment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This will install:
- `langchain` and `langchain-community` for RAG framework
- `langchain-groq` for Groq LLM integration
- `sentence-transformers` for embeddings
- `faiss-cpu` for vector storage
- `pymupdf` (fitz) for PDF reading
- `streamlit` for UI
- `python-dotenv` for environment variables

### 3. Get Groq API Key (FREE)

1. Visit: https://console.groq.com
2. Sign up (free)
3. Create API key
4. Copy your key

### 4. Create `.env` File

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_api_key_here
```

### 5. Get PDF Document

1. Download Swiggy Annual Report from: https://investors.swiggy.com/annual-reports
2. Save as `swiggy_annual_report.pdf` in project directory
   - Or upload via UI in Streamlit app

### 6. Run Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🔧 How It Works

### Phase 1: Ingestion (`ingest.py`)

1. **Extract Text**: PyMuPDF reads PDF page by page
2. **Clean Text**: Remove whitespace, special chars, headers/footers
3. **Chunk**: Split into 800-character chunks with 100-character overlap
4. **Embed**: Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
5. **Store**: Save FAISS index to disk in `faiss_index/` folder

**Function**: `build_vectorstore(pdf_path: str) -> FAISS`

### Phase 2: RAG Pipeline (`rag_pipeline.py`)

1. **Load**: Retrieve FAISS index from disk
2. **Retriever**: Similarity search returns top 4 relevant chunks
3. **Prompt**: Custom template ensures grounded answers
4. **LLM**: Groq llama-3.3-70b-versatile generates answer
5. **Sources**: Return retrieved chunks with page numbers

**Function**: `get_answer(query: str) -> dict`

```python
{
    "answer": "The answer to your question...",
    "sources": [
        {
            "content": "Relevant text excerpt...",
            "page": 15,
            "full_content": "..."
        },
        ...
    ],
    "num_sources": 4
}
```

### Phase 3: UI (`app.py`)

- **Streamlit Interface** with sidebar and main area
- **Session State** keeps vectorstore in memory (no reloading)
- **PDF Upload** or use existing FAISS index
- **Q&A Interface** with spinner and formatted results
- **Source Display** with expandable chunks and page numbers

## 📝 Key Components

### `ingest.py` Functions

```python
extract_text_from_pdf(pdf_path: str) -> list[Document]
    # Extract text with page metadata

clean_text(text: str) -> str
    # Clean and normalize text

split_into_chunks(documents, chunk_size=800, chunk_overlap=100) -> list[Document]
    # Split using RecursiveCharacterTextSplitter

build_vectorstore(pdf_path: str, force_rebuild=False) -> FAISS
    # Main pipeline: Extract → Chunk → Embed → Store
```

### `rag_pipeline.py` Functions

```python
load_vectorstore(faiss_index_dir: str) -> FAISS
    # Load FAISS index from disk

create_rag_chain(vectorstore: FAISS) -> RetrievalQA
    # Create LangChain RetrievalQA with Groq

get_answer(query: str, vectorstore=None) -> dict
    # Main query function with sources

get_answer_with_vectorstore(query: str, vectorstore: FAISS) -> dict
    # Optimized for session state (no reload)
```

## 💡 Example Queries

Test with these questions:

1. **"What is Swiggy's business model and revenue streams?"**
2. **"How many cities does Swiggy operate in?"**
3. **"What are the key risks mentioned in the annual report?"**
4. **"What is Swiggy's market share?"**
5. **"What are the financial highlights?"**
6. **"Who are Swiggy's major competitors?"**
7. **"What is the company's profitability?"**
8. **"How does Swiggy handle food safety?"**

## ⚙️ Configuration

### Chunk Size Tuning

In `ingest.py`:
```python
split_into_chunks(documents, chunk_size=800, chunk_overlap=100)
```

- **Larger chunks** (1000+): More context, fewer results
- **Smaller chunks** (500-): More precise, more results

### Retriever k Value

In `rag_pipeline.py`:
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

- **k=4**: Default (balance of speed and completeness)
- **k=3**: Faster, more focused
- **k=5+**: More context, slower

### LLM Parameters

In `rag_pipeline.py`:
```python
ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,      # 0=factual, 1=creative
    max_tokens=512        # Response length limit
)
```

## 🛠️ Troubleshooting

### PDF not found
```
FileNotFoundError: PDF file not found
```
**Solution**: Ensure PDF is in project directory or upload via UI

### GROQ_API_KEY not set
```
ValueError: GROQ_API_KEY not found
```
**Solution**: 
1. Create `.env` file
2. Add `GROQ_API_KEY=your_key`
3. Restart app

### FAISS index not found
```
FileNotFoundError: FAISS index not found at faiss_index
```
**Solution**: Click "Process Document" in sidebar to build index

### Slow embedding generation
```
Generating embeddings using sentence-transformers...
```
**Solution**: First run is slow (downloads model). Subsequent runs are cached locally.

### API Rate Limit
```
RateLimitError from Groq API
```
**Solution**: Free tier has limits. Wait a moment and retry.

## 📊 Performance Metrics

- **PDF Processing**: ~10-30 seconds (first run)
- **Embedding Generation**: ~5-15 seconds (first run, scales with PDF size)
- **Query Response**: ~2-5 seconds (includes retrieval + LLM inference)
- **Subsequent Queries**: ~2-3 seconds (FAISS already in memory)

## 🔒 Privacy & Security

- ✅ **No Cloud Storage**: FAISS index stored locally
- ✅ **No API for Embeddings**: sentence-transformers runs locally
- ✅ **Only Query → Groq**: Only user queries sent to API
- ✅ **No PDF Storage**: PDF processed once, only chunks stored

## 📚 Source Documentation

- **Swiggy Annual Report**: https://investors.swiggy.com/annual-reports
- **Groq Console**: https://console.groq.com
- **LangChain Docs**: https://python.langchain.com
- **FAISS**: https://github.com/facebookresearch/faiss
- **sentence-transformers**: https://www.sbert.net

## 📄 Tech Stack

| Component | Technology | Why? |
|-----------|-----------|------|
| PDF Loading | PyMuPDF (fitz) | Fast, no dependencies, good text extraction |
| Text Splitting | LangChain RecursiveCharacterTextSplitter | Preserves context, smart separators |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Fast, accurate, free, no API needed |
| Vector Store | FAISS | Efficient, persistent, no external DB |
| LLM | Groq llama-3.3-70b-versatile | Fast, free tier, good quality |
| Framework | LangChain | Unified interface, easy to customize |
| UI | Streamlit | Fast prototyping, beautiful, interactive |

## 🎯 What Makes This Good for Production Use

1. **No Hallucination**: Prompt explicitly tells LLM to only use context
2. **Source Attribution**: Users see exactly where answer comes from
3. **Persistent Storage**: FAISS index doesn't rebuild every run
4. **Session Caching**: Vectorstore cached in memory during session
5. **Error Handling**: Graceful failures with informative messages
6. **Free Stack**: No paid APIs required (Groq has free tier)
7. **Fully Commented**: Every function has docstrings

## 📖 File Descriptions

### `app.py` (500+ lines)
Streamlit UI with:
- Sidebar for PDF management
- Session state for vectorstore caching
- Q&A interface
- Example questions
- Source display with collapsible sections

### `rag_pipeline.py` (200+ lines)
RAG core with:
- FAISS loader
- RetrievalQA setup
- Custom prompt template
- Source formatting

### `ingest.py` (250+ lines)
Document processing with:
- PDF extraction (page metadata)
- Text cleaning
- Chunking with overlap
- Embedding generation
- FAISS persistence

### `requirements.txt`
All dependencies with pinned versions for reproducibility

### `README.md`
This comprehensive guide

## 🚀 Next Steps / Enhancements

- Add support for multiple PDFs
- Implement chat history
- Add query result caching
- Support for different LLM providers
- Fine-tune embeddings on domain-specific data
- Add document summarization feature
- Implement streaming responses
- Add authentication for multi-user deployment

## 📞 Support

For issues:
1. Check error messages in console
2. Verify `.env` file has correct `GROQ_API_KEY`
3. Ensure all dependencies installed: `pip install -r requirements.txt`
4. Check Groq free tier status at https://console.groq.com

## 📄 License

This project is provided as-is for educational and commercial use.

---

**Built with ❤️ using LangChain, FAISS, and Groq**
