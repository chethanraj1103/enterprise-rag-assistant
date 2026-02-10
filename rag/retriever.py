import pickle
import faiss
import numpy as np
import os
import httpx

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN")

INDEX_PATH = "index/faiss.index"
META_PATH = "index/meta.pkl"

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    store = pickle.load(f)

def embed_query(q):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    r = httpx.post(
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}",
        headers=headers,
        json={"inputs": q},
        timeout=60
    )
    emb = np.array(r.json()).mean(axis=0)
    return np.array([emb]).astype("float32")

def retrieve(query, k=4):
    q_emb = embed_query(query)
    D, I = index.search(q_emb, k)

    return [{
        "text": store["chunks"][i],
        "source": store["meta"][i]["source"]
    } for i in I[0]]
