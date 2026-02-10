import os
import pickle
import faiss
import numpy as np
from pypdf import PdfReader
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_PATH = "index/faiss.index"
META_PATH = "index/meta.pkl"

def chunk_text(text, size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def embed_texts(texts):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([d.embedding for d in resp.data]).astype("float32")

def ingest_pdfs(data_dir="data"):
    all_chunks = []
    metadata = []

    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(".pdf"):
            continue

        reader = PdfReader(os.path.join(data_dir, fname))
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        chunks = chunk_text(full_text)

        all_chunks.extend(chunks)
        metadata.extend([{"source": fname}] * len(chunks))

    embeddings = embed_texts(all_chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "meta": metadata}, f)

    return len(all_chunks)
