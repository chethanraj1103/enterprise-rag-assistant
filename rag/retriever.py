import pickle
import faiss
import numpy as np
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_PATH = "index/faiss.index"
META_PATH = "index/meta.pkl"

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    store = pickle.load(f)

def embed_query(q):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[q]
    )
    return np.array([resp.data[0].embedding]).astype("float32")

def retrieve(query, k=4):
    q_emb = embed_query(query)
    D, I = index.search(q_emb, k)

    results = []
    for idx in I[0]:
        results.append({
            "text": store["chunks"][idx],
            "source": store["meta"][idx]["source"]
        })
    return results
