import asyncio
import json
import os
import re
from pathlib import Path
from typing import Annotated

import chromadb
from pypdf import PdfReader
from pydantic import Field
from agent_framework import Agent, tool
from agent_framework.ollama import OllamaChatClient, OllamaEmbeddingClient

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
DOCS_FILE = CHROMA_DIR / "documents.json"
COLLECTION_NAME = "pdf_documents"

CHAT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TOP_K = 5


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def split_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end]
        if end < len(text):
            last_space = piece.rfind(" ")
            if last_space > size * 0.65:
                end = start + last_space
                piece = text[start:end]
        chunks.append(piece.strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_pdfs(uploaded_files) -> tuple[list[dict], list[str]]:
    chunks = []
    document_names = []
    total_pages = 0

    for uploaded in uploaded_files:
        document_names.append(uploaded.name)
        pdf_path = DATA_DIR / uploaded.name
        pdf_path.write_bytes(uploaded.getvalue())

        reader = PdfReader(str(pdf_path))
        total_pages += len(reader.pages)
        for page_number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            for chunk_number, chunk in enumerate(split_text(text), start=1):
                chunks.append(
                    {
                        "id": len(chunks),
                        "text": chunk,
                        "source": uploaded.name,
                        "page": page_number,
                        "chunk": chunk_number,
                    }
                )

    return chunks, document_names, total_pages


async def embed_texts(texts: list[str]) -> list[list[float]]:
    client = OllamaEmbeddingClient(model=EMBED_MODEL, host=OLLAMA_HOST)
    try:
        result = await client.get_embeddings(texts)
        return [item.vector for item in result]
    finally:
        close = getattr(client, "close", None)
        if close:
            maybe = close()
            if asyncio.iscoroutine(maybe):
                await maybe


def get_collection():
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(uploaded_files) -> dict:
    if not uploaded_files:
        raise ValueError("Upload at least one PDF.")
    if len(uploaded_files) > 5:
        raise ValueError("You can upload a maximum of 5 PDFs.")

    DATA_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)

    chunks, document_names, pages = extract_pdfs(uploaded_files)
    if not chunks:
        raise ValueError("No readable text was found in the PDFs.")

    vectors = asyncio.run(embed_texts([item["text"] for item in chunks]))

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Add in batches so larger PDFs do not exceed Chroma's batch limit.
    batch_size = 4000
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        batch_chunks = chunks[start:end]
        collection.add(
            ids=[str(item["id"]) for item in batch_chunks],
            documents=[item["text"] for item in batch_chunks],
            embeddings=vectors[start:end],
            metadatas=[
                {
                    "source": item["source"],
                    "page": item["page"],
                    "chunk": item["chunk"],
                }
                for item in batch_chunks
            ],
        )

    DOCS_FILE.write_text(
        json.dumps({"documents": document_names, "pages": pages}, indent=2),
        encoding="utf-8",
    )

    return {"chunks": len(chunks), "pages": pages, "documents": len(document_names)}


def index_exists() -> bool:
    if not DOCS_FILE.exists() or not CHROMA_DIR.exists():
        return False
    try:
        collection = get_collection()
        return collection.count() > 0
    except Exception:
        return False


def get_index_info() -> dict:
    if not index_exists():
        return {"documents": 0, "chunks": 0}
    docs = json.loads(DOCS_FILE.read_text(encoding="utf-8"))
    collection = get_collection()
    return {"documents": len(docs.get("documents", [])), "chunks": collection.count()}


def search_documents(query: str, top_k: int = TOP_K) -> str:
    """Search the local ChromaDB vector store and return relevant passages."""
    if not index_exists():
        return "No document index exists yet."

    collection = get_collection()
    query_vector = asyncio.run(embed_texts([query]))[0]
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    results = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "?")
        similarity = 1 - float(distance)
        results.append(
            f"SOURCE: {source} | PAGE: {page} | SCORE: {similarity:.3f}\n"
            f"TEXT: {text}"
        )

    return "\n\n".join(results) if results else "No relevant passages were found."


@tool(name="search_documents", description="Search the uploaded PDFs for passages relevant to the user's question.")
def search_documents_tool(
    query: Annotated[str, Field(description="The user's question or the information to find in the PDFs.")],
) -> str:
    """Retrieve relevant passages from the local PDF vector index."""
    return search_documents(query)


async def ask_agent(question: str) -> str:
    client = OllamaChatClient(model=CHAT_MODEL, host=OLLAMA_HOST)
    agent = Agent(
        client=client,
        name="PdfAssistant",
        instructions=(
            "You answer questions using the uploaded PDFs. "
            "Always use the search_documents tool before answering document questions. "
            "Only use information supported by the retrieved passages. "
            "If the answer is not in the documents, say that it was not found in the PDFs. "
            "Keep answers clear and reasonably short. At the end, mention the source PDF and page "
            "for the main facts when available."
        ),
        tools=[search_documents_tool],
    )
    try:
        result = await agent.run(question)
        return result.text
    finally:
        close = getattr(client, "close", None)
        if close:
            maybe = close()
            if asyncio.iscoroutine(maybe):
                await maybe
