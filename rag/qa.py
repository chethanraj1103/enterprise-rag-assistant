import os
from groq import Groq
from rag.retriever import retrieve

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chat_history = []

def answer_question(question):
    docs = retrieve(question, k=3)
    context = "\n\n".join([f"Source: {d['source']}\n{d['text']}" for d in docs])

    history_text = ""
    for q, a in chat_history[-5:]:
        history_text += f"Q: {q}\nA: {a}\n\n"

    prompt = f"""
You are an enterprise assistant. Use ONLY the context below.
If the answer is not in the context, say you don't know.

Chat History:
{history_text}

Context:
{context}

Question:
{question}
"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        print("LLM error:", e)
        answer = "The assistant is temporarily unavailable. Please try again."


    sources = list(set([d["source"] for d in docs]))
    chat_history.append((question, answer))

    return answer, sources
