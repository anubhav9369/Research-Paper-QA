# src/rag_pipeline.py — RAG Pipeline with sentence-transformers
import os
import time
import hashlib
from typing import Optional
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec


_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list, embedder) -> list:
    vectors = embedder.encode(texts, show_progress_bar=False, batch_size=32)
    return vectors.tolist()


def embed_query(query: str, embedder) -> list:
    vector = embedder.encode([query], show_progress_bar=False)
    return vector[0].tolist()


def get_pinecone_index(index_name: str = "research-papers"):
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set in environment.")

    pc = Pinecone(api_key=api_key)
    existing = [i.name for i in pc.list_indexes()]

    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    else:
        info = pc.describe_index(index_name)
        if info.dimension != 384:
            pc.delete_index(index_name)
            time.sleep(2)
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

    return pc.Index(index_name)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def upsert_paper(paper_id, paper_title, chunks, embedder, index, batch_size=100):
    vectors = embed_texts(chunks, embedder)
    records = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        records.append({
            "id": f"{paper_id}_chunk_{i}",
            "values": vector,
            "metadata": {
                "paper_id": paper_id,
                "paper_title": paper_title,
                "chunk_index": i,
                "text": chunk[:1000],
            }
        })
    total = 0
    for i in range(0, len(records), batch_size):
        index.upsert(vectors=records[i:i+batch_size])
        total += len(records[i:i+batch_size])
    return total


def semantic_search(query, embedder, index, top_k=5, filter_paper_id=None):
    query_vector = embed_query(query, embedder)
    filter_dict = {"paper_id": {"$eq": filter_paper_id}} if filter_paper_id else None
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True, filter=filter_dict)
    matches = []
    for match in results.matches:
        matches.append({
            "text": match.metadata.get("text", ""),
            "score": round(match.score, 4),
            "paper_title": match.metadata.get("paper_title", "Unknown"),
            "paper_id": match.metadata.get("paper_id", ""),
            "chunk_index": match.metadata.get("chunk_index", 0),
        })
    return matches


def make_paper_id(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()[:12]


def delete_paper(paper_id, index):
    try:
        results = index.query(vector=[0.0]*384, top_k=1000,
                              filter={"paper_id": {"$eq": paper_id}}, include_metadata=False)
        ids = [m.id for m in results.matches]
        if ids:
            index.delete(ids=ids)
        return len(ids)
    except Exception as e:
        print(f"[Delete error] {e}")
        return 0