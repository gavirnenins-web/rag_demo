# PDF Chat using Microsoft Agent Framework + Ollama

A small local RAG application that lets you upload up to five PDFs and chat with them.

## Stack

- Python
- Streamlit for the UI
- PyPDF for PDF text extraction
- Ollama for the local chat model and embeddings
- ChromaDB for local vector indexing/storage
- Microsoft Agent Framework for the agent and retrieval tool

Microsoft Agent Framework supports Ollama directly, and Python agents can use normal Python functions as tools. The retrieval function in this project is exposed as a MAF tool.

## How it works

1. Upload up to five PDFs.
2. Click **Build index**.
3. Text is extracted page by page.
4. Text is split into small overlapping chunks.
5. Ollama creates an embedding for every chunk.
6. ChromaDB stores the vectors, chunks, and source/page metadata locally in `chroma_db/`.
7. When you ask a question, the MAF agent calls `search_documents`.
9. The query is embedded, ChromaDB finds the closest chunks, and those chunks are given to the agent.
10. Ollama generates the final answer.

## Setup

Install Ollama first and make sure it is running.

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.




### Vector storage

This version uses ChromaDB instead of FAISS. ChromaDB stores the embeddings, document chunks, and PDF metadata in the local `chroma_db/` directory. It uses cosine similarity for retrieval. No Azure account is required.
